"""Unit tests for topologicpy.PyG using a lightweight fake torch_geometric layer."""

from __future__ import annotations

import importlib
import sys
import types
import zipfile
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn as nn


@pytest.fixture(autouse=True)
def _suppress_expected_topologicpy_output(capfd):
    """Keep expected TopologicPy diagnostic prints out of normal pytest output."""
    capfd.readouterr()
    yield
    capfd.readouterr()


def _install_fake_torch_geometric(monkeypatch):
    tg_mod = types.ModuleType("torch_geometric")
    data_mod = types.ModuleType("torch_geometric.data")
    loader_mod = types.ModuleType("torch_geometric.loader")
    nn_mod = types.ModuleType("torch_geometric.nn")
    transforms_mod = types.ModuleType("torch_geometric.transforms")

    class Data:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        def to(self, device):
            for key, value in list(vars(self).items()):
                if torch.is_tensor(value):
                    setattr(self, key, value.to(device))
            return self

    def _clone_single_graph(data):
        if not hasattr(data, "batch") or data.batch is None:
            data.batch = torch.zeros(int(data.x.shape[0]), dtype=torch.long)
        return data

    def _batch_graphs(items):
        items = list(items)
        if len(items) == 1:
            return _clone_single_graph(items[0])
        x_parts = []
        edge_parts = []
        y_parts = []
        batch_parts = []
        node_offset = 0
        for graph_i, data in enumerate(items):
            x_parts.append(data.x)
            if getattr(data, "edge_index", None) is not None and data.edge_index.numel() > 0:
                edge_parts.append(data.edge_index + node_offset)
            if hasattr(data, "y") and data.y is not None:
                y_parts.append(data.y.reshape(-1))
            batch_parts.append(torch.full((int(data.x.shape[0]),), graph_i, dtype=torch.long))
            node_offset += int(data.x.shape[0])
        out = Data(x=torch.cat(x_parts, dim=0), batch=torch.cat(batch_parts, dim=0))
        out.edge_index = torch.cat(edge_parts, dim=1) if edge_parts else torch.empty((2, 0), dtype=torch.long)
        if y_parts:
            out.y = torch.cat(y_parts, dim=0)
        return out

    class DataLoader:
        def __init__(self, dataset, batch_size=1, shuffle=False):
            self.dataset = list(dataset or [])
            self.batch_size = max(1, int(batch_size or 1))
            self.shuffle = bool(shuffle)

        def __iter__(self):
            data = list(self.dataset)
            # Deterministic tests: ignore shuffle while preserving API shape.
            for start in range(0, len(data), self.batch_size):
                yield _batch_graphs(data[start:start + self.batch_size])

        def __len__(self):
            if not self.dataset:
                return 0
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    class _LinearConv(nn.Module):
        def __init__(self, in_channels, out_channels, *args, **kwargs):
            super().__init__()
            self.linear = nn.Linear(int(in_channels), int(out_channels))

        def forward(self, x, edge_index):
            return self.linear(x)

    def _pool(x, batch, reducer):
        if batch is None or batch.numel() == 0:
            return reducer(x, dim=0, keepdim=True).values if reducer is torch.max else reducer(x, dim=0, keepdim=True)
        n_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
        outs = []
        for i in range(n_graphs):
            chunk = x[batch == i]
            if chunk.numel() == 0:
                outs.append(torch.zeros((x.shape[1],), dtype=x.dtype, device=x.device))
            elif reducer is torch.max:
                outs.append(torch.max(chunk, dim=0).values)
            else:
                outs.append(reducer(chunk, dim=0))
        return torch.stack(outs, dim=0)

    def global_mean_pool(x, batch):
        return _pool(x, batch, torch.mean)

    def global_max_pool(x, batch):
        return _pool(x, batch, torch.max)

    def global_add_pool(x, batch):
        return _pool(x, batch, torch.sum)

    class RandomLinkSplit:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def __call__(self, data):
            return data, data, data

    data_mod.Data = Data
    loader_mod.DataLoader = DataLoader
    nn_mod.SAGEConv = _LinearConv
    nn_mod.GCNConv = _LinearConv
    nn_mod.GATv2Conv = _LinearConv
    nn_mod.global_mean_pool = global_mean_pool
    nn_mod.global_max_pool = global_max_pool
    nn_mod.global_add_pool = global_add_pool
    transforms_mod.RandomLinkSplit = RandomLinkSplit

    monkeypatch.setitem(sys.modules, "torch_geometric", tg_mod)
    monkeypatch.setitem(sys.modules, "torch_geometric.data", data_mod)
    monkeypatch.setitem(sys.modules, "torch_geometric.loader", loader_mod)
    monkeypatch.setitem(sys.modules, "torch_geometric.nn", nn_mod)
    monkeypatch.setitem(sys.modules, "torch_geometric.transforms", transforms_mod)


def _load_pyg_module(monkeypatch):
    _install_fake_torch_geometric(monkeypatch)
    sys.modules.pop("topologicpy.PyG", None)
    return importlib.import_module("topologicpy.PyG")


def _write_graph_dataset(path: Path, graph_count: int = 4, empty_edges: bool = False):
    path.mkdir(parents=True, exist_ok=True)
    graph_rows = []
    node_rows = []
    edge_rows = []
    for i in range(graph_count):
        gid = f"G{i}"
        graph_rows.append({
            "graph_id": gid,
            "label": "class_a" if i % 2 == 0 else "class_b",
            "feat_0": float(i),
            "ontology_class": "top:Graph",
            "category": "sample",
            "uri": f"urn:graph:{gid}",
        })
        node_rows.extend([
            {"graph_id": gid, "node_id": f"{gid}_node_b", "label": "wall", "feat_0": 2.0, "feat_1": 0.0, "ontology_class": "top:Vertex", "train_mask": True, "val_mask": False, "test_mask": False},
            {"graph_id": gid, "node_id": f"{gid}_node_a", "label": "room", "feat_0": 1.0, "feat_1": 1.0, "ontology_class": "top:Vertex", "train_mask": False, "val_mask": True, "test_mask": False},
        ])
        if not empty_edges:
            edge_rows.append({
                "graph_id": gid,
                "src_id": f"{gid}_node_b",
                "dst_id": f"{gid}_node_a",
                "label": "edge_b" if i % 2 else "edge_a",
                "feat_0": 3.0,
                "ontology_class": "top:Edge",
                "train_mask": True,
                "val_mask": False,
                "test_mask": False,
            })
    pd.DataFrame(graph_rows).to_csv(path / "graphs.csv", index=False)
    pd.DataFrame(node_rows).to_csv(path / "nodes.csv", index=False)
    edge_cols = ["graph_id", "src_id", "dst_id", "label", "feat_0", "ontology_class", "train_mask", "val_mask", "test_mask"]
    pd.DataFrame(edge_rows, columns=edge_cols).to_csv(path / "edges.csv", index=False)


def test_config_feature_helpers_and_summary(monkeypatch, tmp_path):
    pyg_mod = _load_pyg_module(monkeypatch)
    _write_graph_dataset(tmp_path)

    pyg = pyg_mod.PyG.ByCSVPath(
        str(tmp_path),
        level="graph",
        task="classification",
        use_gpu=False,
        shuffle=False,
        batch_size=1,
        hidden_dims=(8,),
    )

    assert pyg._feature_columns(pd.DataFrame(columns=["feat_10", "feat_2", "feat_x", "other"]), "feat") == ["feat_2", "feat_10", "feat_x"]
    assert pyg._infer_num_classes(["b", "a", "a"]) == 2
    summary = pyg.Summary()
    assert summary["level"] == "graph"
    assert summary["num_graphs"] == 4
    assert summary["num_outputs"] == 2
    assert summary["ontology"] is True

    updated = pyg.SetHyperparameters(hidden_dims=[4, 4], activation="gelu", dropout=0.0)
    assert updated["hidden_dims"] == (4, 4)
    assert pyg.model is not None


def test_graph_csv_loading_remaps_string_node_ids_and_encodes_labels(monkeypatch, tmp_path):
    pyg_mod = _load_pyg_module(monkeypatch)
    _write_graph_dataset(tmp_path)

    pyg = pyg_mod.PyG.ByCSVPath(
        str(tmp_path),
        level="graph",
        task="classification",
        use_gpu=False,
        shuffle=False,
        batch_size=1,
        hidden_dims=(4,),
    )

    first = pyg.data_list[0]
    assert tuple(first.x.shape) == (2, 2)
    assert tuple(first.edge_index.shape) == (2, 1)
    # nodes are sorted by node_id, so G0_node_a -> 0 and G0_node_b -> 1.
    assert first.edge_index.tolist() == [[1], [0]]
    assert first.y.item() == pyg._label_encoders["graph"]["class_a"]
    assert pyg._label_encoders["graph"] == {"class_a": 0, "class_b": 1}
    assert pyg._label_decoders["graph"] == {0: "class_a", 1: "class_b"}

    meta = pyg.MetadataByGraphID("G0")
    assert meta["graph"]["ontology_class"] == "top:Graph"
    assert meta["nodes"][0]["ontology_class"] == "top:Vertex"
    assert meta["edges"][0]["ontology_class"] == "top:Edge"
    md_copy = pyg.OntologyMetadata()
    md_copy["graphs"]["G0"]["ontology_class"] = "mutated"
    assert pyg.MetadataByGraphID("G0")["graph"]["ontology_class"] == "top:Graph"


def test_node_and_edge_level_labels_and_masks(monkeypatch, tmp_path):
    pyg_mod = _load_pyg_module(monkeypatch)
    _write_graph_dataset(tmp_path, graph_count=2)

    node_pyg = pyg_mod.PyG.ByCSVPath(
        str(tmp_path),
        level="node",
        task="classification",
        use_gpu=False,
        shuffle=False,
        batch_size=1,
        hidden_dims=(4,),
    )
    node_data = node_pyg.data_list[0]
    assert node_pyg._label_encoders["node"] == {"room": 0, "wall": 1}
    assert node_data.y.tolist() == [0, 1]
    assert node_data.train_mask.tolist() == [False, True]
    assert node_data.val_mask.tolist() == [True, False]
    assert node_data.test_mask.tolist() == [False, False]

    edge_pyg = pyg_mod.PyG.ByCSVPath(
        str(tmp_path),
        level="edge",
        task="classification",
        use_gpu=False,
        shuffle=False,
        batch_size=1,
        hidden_dims=(4,),
    )
    edge_data = edge_pyg.data_list[0]
    assert edge_pyg._label_encoders["edge"] == {"edge_a": 0, "edge_b": 1}
    assert edge_data.edge_y.tolist() == [0]
    assert edge_data.edge_train_mask.tolist() == [True]
    assert edge_data.edge_val_mask.tolist() == [False]
    assert edge_data.edge_test_mask.tolist() == [False]


def test_missing_endpoint_and_duplicate_node_ids_raise_clear_errors(monkeypatch, tmp_path):
    pyg_mod = _load_pyg_module(monkeypatch)
    _write_graph_dataset(tmp_path, graph_count=1)
    edges = pd.read_csv(tmp_path / "edges.csv")
    edges.loc[0, "src_id"] = "missing_node"
    edges.to_csv(tmp_path / "edges.csv", index=False)

    with pytest.raises(ValueError, match="Edge references a node_id"):
        pyg_mod.PyG.ByCSVPath(str(tmp_path), level="graph", use_gpu=False, hidden_dims=(4,))

    _write_graph_dataset(tmp_path, graph_count=1)
    nodes = pd.read_csv(tmp_path / "nodes.csv")
    nodes.loc[1, "node_id"] = nodes.loc[0, "node_id"]
    nodes.to_csv(tmp_path / "nodes.csv", index=False)

    with pytest.raises(ValueError, match="Duplicate node_id"):
        pyg_mod.PyG.ByCSVPath(str(tmp_path), level="graph", use_gpu=False, hidden_dims=(4,))


def test_empty_edges_create_valid_edge_tensors(monkeypatch, tmp_path):
    pyg_mod = _load_pyg_module(monkeypatch)
    _write_graph_dataset(tmp_path, graph_count=2, empty_edges=True)

    pyg = pyg_mod.PyG.ByCSVPath(
        str(tmp_path),
        level="graph",
        task="classification",
        use_gpu=False,
        shuffle=False,
        batch_size=1,
        hidden_dims=(4,),
    )

    data = pyg.data_list[0]
    assert tuple(data.edge_index.shape) == (2, 0)
    assert hasattr(data, "edge_attr")
    assert tuple(data.edge_attr.shape) == (0, 1)


def test_holdout_grouping_and_predict_metadata_alignment(monkeypatch, tmp_path):
    pyg_mod = _load_pyg_module(monkeypatch)
    _write_graph_dataset(tmp_path, graph_count=4)
    graphs = pd.read_csv(tmp_path / "graphs.csv")
    graphs["group"] = ["A", "B", "C", "D"]
    graphs.to_csv(tmp_path / "graphs.csv", index=False)

    pyg = pyg_mod.PyG.ByCSVPath(
        str(tmp_path),
        level="graph",
        task="classification",
        use_gpu=False,
        shuffle=False,
        split=(0.5, 0.25, 0.25),
        holdout_group_by="group",
        batch_size=1,
        hidden_dims=(4,),
    )

    assert [int(d.topologicpy_graph_index.item()) for d in pyg.train_set] == [0, 1]
    assert [int(d.topologicpy_graph_index.item()) for d in pyg.val_set] == [2]
    assert [int(d.topologicpy_graph_index.item()) for d in pyg.test_set] == [3]

    out_all = pyg.Predict(split="all")
    assert out_all["index"].tolist() == [0, 1, 2, 3]
    assert len(out_all["metadata"]) == 4
    assert out_all["metadata"][0]["uri"] == "urn:graph:G0"

    out_test = pyg.Predict(split="test")
    assert out_test["index"].tolist() == [3]
    assert out_test["metadata"][0]["uri"] == "urn:graph:G3"


def test_save_and_load_preserve_schema_and_label_encoders(monkeypatch, tmp_path):
    pyg_mod = _load_pyg_module(monkeypatch)
    data_dir = tmp_path / "data"
    _write_graph_dataset(data_dir, graph_count=3)

    pyg = pyg_mod.PyG.ByCSVPath(
        str(data_dir),
        level="graph",
        task="classification",
        use_gpu=False,
        shuffle=False,
        batch_size=1,
        hidden_dims=(4,),
    )
    save_path = tmp_path / "model.pt"
    pyg.SaveModel(str(save_path))
    assert save_path.exists()

    loaded = pyg_mod.PyG.ByCSVPath(
        str(data_dir),
        level="graph",
        task="classification",
        use_gpu=False,
        shuffle=False,
        batch_size=1,
        hidden_dims=(4,),
    )
    loaded._label_encoders = {"graph": {}, "node": {}, "edge": {}}
    loaded._label_decoders = {"graph": {}, "node": {}, "edge": {}}
    loaded.LoadModel(str(save_path), strict=True)

    assert loaded._label_encoders["graph"] == {"class_a": 0, "class_b": 1}
    assert loaded._label_decoders["graph"] == {0: "class_a", 1: "class_b"}
    assert loaded._feature_schema["node_feat_cols"] == ["feat_0", "feat_1"]
    assert loaded._num_outputs == 2


def test_metric_helpers_and_plot_cv_summary(monkeypatch, tmp_path):
    pyg_mod = _load_pyg_module(monkeypatch)
    _write_graph_dataset(tmp_path, graph_count=3)
    pyg = pyg_mod.PyG.ByCSVPath(str(tmp_path), level="graph", task="classification", use_gpu=False, hidden_dims=(4,))

    cls = pyg._classification_metrics(torch.tensor([0, 1]).numpy(), torch.tensor([0, 1]).numpy(), prefix="x_")
    assert cls["x_accuracy"] == 1.0
    reg = pyg._regression_metrics(torch.tensor([1.0]).numpy(), torch.tensor([1.5]).numpy(), prefix="r_")
    assert reg["r_r2"] == 0.0

    pyg.cv_report = {
        "fold_metrics": [
            {"fold": 0, "accuracy": 0.5, "f1": 0.4, "precision": 0.3, "recall": 0.2},
            {"fold": 1, "accuracy": 1.0, "f1": 0.9, "precision": 0.8, "recall": 0.7},
        ],
        "mean_accuracy": 0.75,
        "std_accuracy": 0.25,
    }
    fig = pyg.PlotCrossValidationSummary(metrics=["accuracy"])
    assert len(fig.data) >= 2


def test_invalid_inputs_and_dependency_fallback(monkeypatch, tmp_path):
    pyg_mod = _load_pyg_module(monkeypatch)
    missing = tmp_path / "missing"
    with pytest.raises(ValueError, match="path does not exist"):
        pyg_mod.PyG(path=str(missing), config=pyg_mod._RunConfig())

    _write_graph_dataset(tmp_path, graph_count=1)
    with pytest.raises(ValueError, match="split must be a 3-tuple"):
        pyg = pyg_mod.PyG.ByCSVPath(str(tmp_path), use_gpu=False, hidden_dims=(4,))
        pyg.SetHyperparameters(split=(0.8, 0.2))

    # Import the same source without fake torch_geometric. This verifies the
    # documentation/import fallback path without relying on the global package import.
    for key in ["torch_geometric", "torch_geometric.data", "torch_geometric.loader", "torch_geometric.nn", "torch_geometric.transforms"]:
        monkeypatch.delitem(sys.modules, key, raising=False)
    sys.modules.pop("topologicpy.PyG", None)
    mod = importlib.import_module("topologicpy.PyG")
    if mod._PYG_IMPORT_ERROR is not None:
        with pytest.raises(ImportError):
            mod.PyG.ByCSVPath(str(tmp_path))
