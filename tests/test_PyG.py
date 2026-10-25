"""Regression coverage for PyG data, training, inference and checkpoint contracts.

Run with PyTorch, torch-geometric, pandas, scikit-learn and pytest installed.
"""
from __future__ import annotations

import copy
import importlib
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("sklearn")
from topologicpy.PyG import PyG


@pytest.fixture(autouse=True)
def cpu_determinism():
    torch.manual_seed(123)
    torch.set_num_threads(1)


@pytest.fixture
def dataset(tmp_path):
    def make(name="data", graphs=20, nodes=12, continuous=False, labels=True,
             mask=True, features=False, only_b=False):
        path = tmp_path / name
        path.mkdir()
        gs, ns, es = [], [], []
        for gid in range(graphs):
            row = {"graph_id": gid, "building": gid // 2, "name": f"graph {gid}"}
            if labels:
                row["label"] = float(gid) / 10 if continuous else ("B" if only_b or gid % 2 else "A")
            if features:
                row["feat_weather"] = .5
            gs.append(row)
            for i in range(nodes):
                row = {"graph_id": gid, "node_id": f"node-{i:02}",
                       "feat_0": float(i) / nodes, "feat_1": float(gid) / 20,
                       "name": f"node {i}"}
                if labels: row["label"] = float(i) / 10 if continuous else ("B" if only_b or i % 2 else "A")
                if mask:
                    row.update(train_mask=i % 4 < 2, val_mask=i % 4 == 2, test_mask=i % 4 == 3)
                ns.append(row)
            for j, (a, b) in enumerate(( (i, (i + d) % nodes) for i in range(nodes) for d in (1, 3))):
                row = {"graph_id": gid, "src_id": f"node-{a:02}", "dst_id": f"node-{b:02}", "name": f"edge {j}"}
                if labels: row["label"] = float(j) / 10 if continuous else ("B" if only_b or j % 2 else "A")
                if features: row["feat_distance"] = 1.
                if mask: row.update(train_mask=j % 4 < 2, val_mask=j % 4 == 2, test_mask=j % 4 == 3)
                es.append(row)
        for file, rows in (("graphs", gs), ("nodes", ns), ("edges", es)):
            pd.DataFrame(rows).to_csv(path / (file + ".csv"), index=False)
        return path
    return make


def runner(path, level="graph", task="classification", **kwargs):
    args = dict(hidden_dims=(8, 8), dropout=0., use_gpu=False, epochs=2, batch_size=4)
    if task == "regression":
        args.update(graphLabelType="continuous", nodeLabelType="continuous", edgeLabelType="continuous")
    args.update(kwargs)
    return PyG.ByCSVPath(str(path), level=level, task=task, **args)


@pytest.mark.parametrize("conv", ["sage", "gcn", "gatv2"])
@pytest.mark.parametrize("level,task", [("graph", "classification"), ("node", "classification"),
    ("edge", "classification"), ("graph", "regression"), ("node", "regression"),
    ("edge", "regression"), ("link", "link_prediction")])
def test_real_training_evaluation_and_prediction(dataset, conv, level, task):
    model = runner(dataset(graphs=10, continuous=task == "regression"), level, task, conv=conv)
    initial = copy.deepcopy(model.model.state_dict())
    history = model.Train()
    assert all(np.isfinite(history["train_loss"]))
    assert any(not torch.equal(v, initial[k]) for k, v in model.model.state_dict().items())
    assert all(np.isfinite(list(model.Validate().values())))
    assert all(np.isfinite(list(model.Test().values())))
    prediction = model.Predict(return_logits=True)
    assert "pred" in prediction and "logits" in prediction
    if task == "classification":
        assert prediction["class_labels"] == ["A", "B"]
        values = np.concatenate(prediction["pred"]) if isinstance(prediction["pred"], list) else prediction["pred"]
        assert set(values) <= {"A", "B"}


def test_link_split_is_fixed_and_rng_is_preserved(dataset):
    model = runner(dataset(graphs=2), "link", "link_prediction")
    rng = torch.get_rng_state().clone()
    model._BuildLinkSplits()
    assert torch.equal(rng, torch.get_rng_state())
    first = model.Predict(split="test", return_logits=True)
    cached = copy.deepcopy(model._link_splits)
    model.Train()
    second = model.Predict(split="test")
    third = model.Predict(split="test")
    np.testing.assert_array_equal(first["y_true"], second["y_true"])
    np.testing.assert_array_equal(second["score"], third["score"])
    assert model.Test() == model.Test()
    for before, after in zip(cached, model._link_splits):
        for split in ("train", "val", "test"):
            assert torch.equal(before[split].edge_label_index, after[split].edge_label_index)
        train_edges = {tuple(x) for x in after["train"].edge_index.t().tolist()}
        for split in ("val", "test"):
            positives = after[split].edge_label_index[:, after[split].edge_label.bool()].t().tolist()
            assert train_edges.isdisjoint(map(tuple, positives))


@pytest.mark.parametrize("level", ["graph", "node", "edge"])
def test_checkpoint_preserves_label_mapping_and_logits(dataset, tmp_path, level):
    model = runner(dataset(), level)
    model.Train()
    path = str(tmp_path / "model.pt")
    model.SaveModel(path)
    subset = runner(dataset(name="subset", graphs=2, only_b=True), level)
    subset.LoadModel(path)
    assert subset._label_encoders[level] == {"A": 0, "B": 1}
    subset._build_data_list()
    assert subset._label_encoders[level]["B"] == 1
    out = subset.Predict(return_logits=True)
    true = out["y_true"] if level == "graph" else np.concatenate(out["y_true"])
    assert set(true) == {"B"}
    restored = runner(model.path, level)
    restored.LoadModel(path)
    expected, actual = model.Predict(return_logits=True), restored.Predict(return_logits=True)
    if level == "graph":
        np.testing.assert_allclose(expected["logits"], actual["logits"])
    else:
        for a, b in zip(expected["logits"], actual["logits"]): np.testing.assert_allclose(a, b)


@pytest.mark.parametrize("level", ["graph", "node", "edge"])
def test_unlabelled_inference(dataset, tmp_path, level):
    model = runner(dataset(), level)
    model.Train()
    checkpoint = str(tmp_path / "model.pt")
    model.SaveModel(checkpoint)
    unseen = runner(dataset(name="unseen", graphs=2, labels=False, mask=False), level, inference=True)
    unseen.LoadModel(checkpoint)
    out = unseen.Predict(return_logits=True)
    assert "y_true" not in out
    assert out["class_labels"] == ["A", "B"]
    with pytest.raises(ValueError, match="inference"):
        unseen.Train()


@pytest.mark.parametrize("level,task", [("graph", "classification"), ("node", "classification"),
    ("edge", "classification"), ("node", "regression"), ("edge", "regression")])
def test_prediction_attachment_updates_original_data(dataset, level, task):
    model = runner(dataset(continuous=task == "regression"), level, task)
    out = model.Predict(attach_to_data=True, return_logits=True, return_embeddings=True)
    for index, data in enumerate(model.data_list):
        np.testing.assert_array_equal(data.pred, out["pred"][index])
        np.testing.assert_allclose(data.logits, out["logits"][index])
        if level != "graph": assert data.emb.shape[0] == data.num_nodes


def test_live_optimizer_settings_and_transactional_validation(dataset):
    model = runner(dataset())
    net = model.model
    model.SetHyperparameters(lr=.02, weight_decay=.03)
    assert model.model is net
    assert model.optimizer.param_groups[0]["lr"] == .02
    assert model.optimizer.param_groups[0]["weight_decay"] == .03
    model.SetHyperparameters(optimizer="adamw")
    assert isinstance(model.optimizer, torch.optim.AdamW)
    assert model.model is net
    with pytest.raises(ValueError): model.SetHyperparameters(lr=.4, dropout=2)
    assert model.config.lr == .02
    with pytest.raises(ValueError, match="new ByCSVPath"): model.SetHyperparameters(level="node")
    with pytest.raises(ValueError, match="Unknown"): model.SetHyperparameters(typo_lr=.2)


def test_features_are_supported_without_ignored_feature_warnings(dataset):
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        model = runner(dataset(features=True))
    assert not any("ignore" in str(w.message).lower() for w in caught)
    assert model.Summary()["edge_feature_dim"] == 1
    assert model.Summary()["graph_feature_dim"] == 1


@pytest.mark.parametrize("level", ["graph", "node", "edge", "link"])
def test_early_stopping_restores_best_weights(dataset, level, monkeypatch):
    model = runner(dataset(), level, "link_prediction" if level == "link" else "classification",
                   epochs=6, early_stopping=True, early_stopping_patience=1)
    parameter = next(model.model.parameters())
    losses = iter([1., 2.])
    snapshots = []
    def train(loader):
        with torch.no_grad(): parameter.add_(1)
        snapshots.append(parameter.detach().clone())
        return 1.
    monkeypatch.setattr(model, "_train_epoch_" + level, train)
    monkeypatch.setattr(model, "_eval_epoch_" + level, lambda loader: next(losses))
    model.Train()
    assert len(model.history["train_loss"]) == 2
    torch.testing.assert_close(parameter, snapshots[0])


def test_grouped_cv_preserves_state_and_separates_test_validation(dataset):
    model = runner(dataset(graphs=24), holdout_group_by="building", early_stopping=True,
                   early_stopping_patience=1)
    model.Train()
    net, optimizer, cfg, hist = model.model, model.optimizer, model.config, model.history
    weights = copy.deepcopy(net.state_dict())
    report = model.CrossValidate(k_folds=3)
    assert model.model is net and model.optimizer is optimizer and model.config is cfg and model.history is hist
    for key, value in net.state_dict().items(): torch.testing.assert_close(value, weights[key])
    for fold in report["fold_metrics"]:
        sets = [{i // 2 for i in fold[key]} for key in ("train_indices", "val_indices", "test_indices")]
        assert all(sets) and not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])


def test_cv_state_restored_on_failure(dataset, monkeypatch):
    model = runner(dataset())
    net, opt, history = model.model, model.optimizer, model.history
    def fail(*args): raise RuntimeError("failure during fold")
    monkeypatch.setattr(model, "_train_epoch_graph", fail)
    with pytest.raises(RuntimeError, match="during fold"): model.CrossValidate(k_folds=2)
    assert model.model is net and model.optimizer is opt and model.history is history


@pytest.mark.parametrize("bad", [(-.1, .5, .6), (.8, .1), (.8, .1, float("nan")), (.8, .1, .2)])
def test_invalid_split_rejected_at_creation(dataset, bad):
    with pytest.raises(ValueError, match="split"): runner(dataset(), split=bad)


def test_masks_parse_strings_and_reject_overlap(dataset):
    path = dataset(graphs=2)
    file = path / "nodes.csv"
    frame = pd.read_csv(file)
    for key in ("train_mask", "val_mask", "test_mask"):
        frame[key] = frame[key].map({True: "1", False: "0"})
    frame.to_csv(file, index=False)
    model = runner(path, "node")
    assert int(model.data_list[0].train_mask.sum()) == 6
    frame.loc[0, "val_mask"] = "1"
    frame.to_csv(file, index=False)
    with pytest.raises(ValueError, match="overlap"): runner(path, "node")
    assert PyG._ParseMask(["False", "True", "0", "1"]).tolist() == [False, True, False, True]
    with pytest.raises(ValueError): PyG._ParseMask(["perhaps"])


def test_empty_supervision_is_explicit_and_empty_graphs_are_skipped(dataset):
    path = dataset(graphs=2)
    file = path / "nodes.csv"
    frame = pd.read_csv(file)
    frame.loc[frame.graph_id == 0, "train_mask"] = False
    frame.to_csv(file, index=False)
    model = runner(path, "node")
    assert np.isfinite(model.Train()["train_loss"]).all()
    frame["train_mask"] = False
    frame.to_csv(file, index=False)
    model = runner(path, "node")
    with pytest.raises(ValueError, match="No supervised"): model.Train()


def test_required_and_optional_features_on_checkpoint_load(dataset, tmp_path):
    model = runner(dataset(), optional_feature_columns=("feat_1",))
    checkpoint = str(tmp_path / "model.pt")
    model.SaveModel(checkpoint)
    path = dataset(name="missing", graphs=2, labels=False, mask=False)
    file = path / "nodes.csv"
    frame = pd.read_csv(file).drop(columns="feat_1")
    frame.to_csv(file, index=False)
    unseen = runner(path, inference=True)
    unseen.LoadModel(checkpoint)
    assert unseen.data_list[0].x.shape[1] == 2
    assert torch.count_nonzero(unseen.data_list[0].x[:, 1]) == 0
    frame = frame.rename(columns={"feat_0": "feat_other"})
    frame.to_csv(file, index=False)
    unseen = runner(path, inference=True)
    with pytest.raises(ValueError, match="Missing required"): unseen.LoadModel(checkpoint)


def test_safe_checkpoint_load_never_silently_retries(dataset, tmp_path, monkeypatch):
    model = runner(dataset())
    real_load = torch.load
    calls = []
    def reject(*args, **kwargs):
        calls.append(kwargs.get("weights_only"))
        raise RuntimeError("unsafe payload")
    monkeypatch.setattr(torch, "load", reject)
    with pytest.raises(ValueError, match="trusted=True"): model.LoadModel(str(tmp_path / "x.pt"))
    assert calls == [True]
    checkpoint = str(tmp_path / "trusted.pt")
    model.SaveModel(checkpoint)
    calls.clear()
    def require_trust(*args, **kwargs):
        calls.append(kwargs.get("weights_only"))
        if kwargs.get("weights_only"): raise RuntimeError("legacy metadata")
        return real_load(*args, **kwargs)
    monkeypatch.setattr(torch, "load", require_trust)
    model.LoadModel(checkpoint, trusted=True)
    assert calls == [True, False]


@pytest.mark.parametrize("level", ["node", "edge"])
def test_masked_predictions_align_logits_metadata_and_original_attachment(dataset, level):
    model = runner(dataset(graphs=2), level)
    out = model.Predict(split="test", return_logits=True, attach_to_data=True)
    for i, (pred, logits, rows, selected) in enumerate(zip(out["pred"], out["logits"], out["metadata"], out["index"])):
        assert len(pred) == len(logits) == len(rows) == len(selected)
        assert model.data_list[i].pred.shape[0] > len(pred)
        assert len(model.data_list[i].logits) == len(model.data_list[i].pred)
        for row, index in zip(rows, selected): assert row["name"] == f"{level} {index}"


def test_link_splits_reproduce_across_instances_and_restore_all_rngs(dataset):
    import random
    path = dataset(graphs=2)
    a = runner(path, "link", "link_prediction")
    python_state, numpy_state = random.getstate(), np.random.get_state()
    a._BuildLinkSplits()
    assert random.getstate() == python_state
    after = np.random.get_state()
    assert after[0] == numpy_state[0] and np.array_equal(after[1], numpy_state[1]) and after[2:] == numpy_state[2:]
    random.seed(983)
    np.random.seed(342)
    torch.manual_seed(987)
    b = runner(path, "link", "link_prediction")
    b._BuildLinkSplits()
    for ga, gb in zip(a._link_splits, b._link_splits):
        for split in ("train", "val", "test"):
            torch.testing.assert_close(ga[split].edge_label_index, gb[split].edge_label_index)
            torch.testing.assert_close(ga[split].edge_label, gb[split].edge_label)


def test_zero_split_ratios_respected_and_small_folds_rejected(dataset):
    model = runner(dataset(), split=(1., 0., 0.))
    assert len(model.train_set) == 20 and not model.val_set and not model.test_set
    assert np.isfinite(model.Train()["train_loss"]).all()
    with pytest.raises(ValueError, match="No supervised"): model.Test()
    model.SetHyperparameters(early_stopping=True)
    with pytest.raises(ValueError, match="No supervised"): model.Train()
    model.SetHyperparameters(early_stopping=False)
    with pytest.raises(ValueError, match="k_folds"): model.CrossValidate(k_folds=30)


def test_inference_unknown_label_and_raw_weight_checkpoints(dataset, tmp_path):
    model = runner(dataset())
    model.Train()
    path = str(tmp_path / "raw.pt")
    model.SaveModel(path, include_config=False)
    other = runner(model.path)
    other.LoadModel(path)
    np.testing.assert_allclose(model.Predict(return_logits=True)["logits"], other.Predict(return_logits=True)["logits"])
    labelled = dataset(name="unknown")
    file = labelled / "graphs.csv"
    frame = pd.read_csv(file)
    frame["label"] = "C"
    frame.to_csv(file, index=False)
    model.SaveModel(str(tmp_path / "full.pt"))
    unknown = runner(labelled)
    with pytest.raises(ValueError, match="Unknown graph categorical label"):
        unknown.LoadModel(str(tmp_path / "full.pt"))


@pytest.mark.parametrize("level", ["graph", "node", "edge"])
def test_numeric_labels_are_decoded(dataset, level):
    path = dataset()
    for name in ("graphs", "nodes", "edges"):
        file = path / (name + ".csv")
        frame = pd.read_csv(file)
        frame["label"] = frame["label"].map({"A": 10, "B": 20})
        frame.to_csv(file, index=False)
    out = runner(path, level).Predict()
    assert out["class_labels"] == [10, 20]
    pred = np.concatenate(out["pred"]) if isinstance(out["pred"], list) else out["pred"]
    assert set(pred) <= {10, 20}


def test_csv_export_load_train_roundtrip(tmp_path):
    from topologicpy.TGraph import TGraph
    graphs = []
    for i in range(10):
        graph = TGraph.ByMeshData([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1], [1, 2], [2, 0]],
                                 vertexDictionaries=[{"temperature": j + i} for j in range(3)])
        graph._dictionary["label"] = "A" if i % 2 else "B"
        graphs.append(graph)
    path = tmp_path / "exported"
    assert TGraph.ExportToCSV(graphs, str(path), nodeFeaturesKeys=["temperature"])
    model = runner(path)
    assert np.isfinite(model.Train()["train_loss"]).all()
    assert np.isfinite(list(model.Test().values())).all()


@pytest.mark.parametrize("level", ["graph", "node", "edge"])
def test_regression_learns_simple_synthetic_relationship(dataset, level):
    model = runner(dataset(continuous=True), level, "regression", epochs=35, lr=.02, hidden_dims=(16, 16))
    losses = model.Train()["train_loss"]
    assert losses[-1] < .5 * losses[0]
    assert model.Test()["test_rmse"] < 1.


def test_cv_without_early_stopping_does_not_use_test_loss(dataset, monkeypatch):
    model = runner(dataset())
    def forbidden(loader): raise AssertionError("Outer test loss must not control fitting")
    monkeypatch.setattr(model, "_eval_epoch_graph", forbidden)
    assert len(model.CrossValidate(k_folds=2)["fold_metrics"]) == 2


def test_link_checkpoint_roundtrip_reuses_same_split(dataset, tmp_path):
    model = runner(dataset(graphs=2), "link", "link_prediction")
    model.Train()
    checkpoint = str(tmp_path / "link.pt")
    model.SaveModel(checkpoint)
    other = runner(model.path, "link", "link_prediction")
    other.LoadModel(checkpoint)
    a, b = model.Predict(return_logits=True), other.Predict(return_logits=True)
    np.testing.assert_array_equal(a["y_true"], b["y_true"])
    np.testing.assert_allclose(a["logits"], b["logits"])


def test_grouped_zero_split_ratios_and_config_changes(dataset):
    model = runner(dataset(), holdout_group_by="building", split=(1., 0., 0.))
    assert len(model.train_set) == 20 and not model.val_set and not model.test_set
    model.SetHyperparameters(split=(.6, .2, .2))
    assert tuple(map(len, (model.train_set, model.val_set, model.test_set))) == (12, 4, 4)
    before = model.config
    with pytest.raises(ValueError, match="nonmissing"): model.SetHyperparameters(holdout_group_by="missing")
    assert model.config is before


@pytest.mark.parametrize("settings", [
    {"dropout": -1}, {"dropout": 1}, {"lr": 0}, {"lr": float("nan")},
    {"weight_decay": -1}, {"hidden_dims": ()}, {"hidden_dims": (0,)},
    {"epochs": 0}, {"batch_size": 0}, {"random_state": -1},
    {"random_state": 2**32}, {"link_val_ratio": .6, "link_test_ratio": .5},
    {"gradient_clip_norm": 0}, {"conv": "unknown"}, {"optimizer": "unknown"},
])
def test_invalid_configuration_rejected_before_loading_data(tmp_path, settings):
    with pytest.raises(ValueError): runner(tmp_path / "nonexistent", **settings)


def test_partial_masks_and_nonfinite_inputs_are_rejected(dataset):
    path = dataset()
    file = path / "nodes.csv"
    frame = pd.read_csv(file)
    frame.drop(columns="test_mask").to_csv(file, index=False)
    with pytest.raises(ValueError, match="all three"): runner(path, "node")
    frame.loc[0, "feat_0"] = float("nan")
    frame.to_csv(file, index=False)
    with pytest.raises(ValueError, match="finite"): runner(path, "node")


def test_live_node_split_changes_rebuild_fallback_masks(dataset):
    model = runner(dataset(mask=False), "node")
    old = int(model.data_list[0].train_mask.sum())
    model.SetHyperparameters(split=(.5, .25, .25))
    assert int(model.data_list[0].train_mask.sum()) == 6 != old
    assert tuple(int(getattr(model.data_list[0], key).sum()) for key in ("train_mask", "val_mask", "test_mask")) == (6, 3, 3)


@pytest.mark.parametrize("conv", ["sage", "gcn", "gatv2"])
@pytest.mark.parametrize("level,task", [("graph", "classification"), ("node", "classification"),
    ("edge", "classification"), ("graph", "regression"), ("node", "regression"),
    ("edge", "regression"), ("link", "link_prediction")])
def test_edge_and_graph_features_affect_all_forward_paths(dataset, conv, level, task):
    model = runner(dataset(graphs=10, features=True, continuous=task == "regression"), level, task, conv=conv)
    model.Train()
    assert np.isfinite(list(model.Validate().values())).all()
    assert np.isfinite(list(model.Test().values())).all()
    data = model.data_list[0].clone().to(model.device)
    data.x.requires_grad_(True)
    data.edge_attr.requires_grad_(True)
    data.u.requires_grad_(True)
    model.model.eval()
    encoded = model._Encode(data)
    weights = torch.arange(encoded.numel(), dtype=encoded.dtype).reshape(encoded.shape) + 1
    (encoded * weights).sum().backward()
    assert data.edge_attr.grad is not None and data.edge_attr.grad.abs().sum() > 0
    assert data.u.grad is not None and data.u.grad.abs().sum() > 0
    def predict_one(item):
        node_emb = model._Encode(item)
        if level == "graph":
            batch = torch.zeros(item.num_nodes, dtype=torch.long)
            return model.model["head"](node_emb, batch, item.u)
        if level == "node":
            return model.model["head"](node_emb)
        if level == "edge":
            return model.model["head"](node_emb, item.edge_index, item.edge_attr)
        return model.model["predictor"](node_emb, item.edge_index)
    baseline = model._Encode(data).detach()
    baseline_prediction = predict_one(data).detach()
    changed_edges = data.clone()
    changed_edges.edge_attr = torch.randn_like(data.edge_attr) * 5
    assert not torch.allclose(baseline, model._Encode(changed_edges))
    assert not torch.allclose(baseline_prediction, predict_one(changed_edges))
    changed_graph = data.clone()
    changed_graph.u = data.u.detach() + 3
    assert not torch.allclose(baseline, model._Encode(changed_graph))
    assert not torch.allclose(baseline_prediction, predict_one(changed_graph))
    out = model.Predict(return_logits=True)
    assert "logits" in out


@pytest.mark.parametrize("conv", ["sage", "gcn", "gatv2"])
@pytest.mark.parametrize("level,task", [("graph", "classification"), ("node", "classification"),
                                      ("edge", "regression"), ("link", "link_prediction")])
def test_feature_aware_checkpoint_and_unlabelled_inference(dataset, tmp_path, conv, level, task):
    path = dataset(graphs=10, features=True, continuous=task == "regression")
    model = runner(path, level, task, conv=conv)
    model.Train()
    checkpoint = str(tmp_path / "features.pt")
    model.SaveModel(checkpoint)
    restored = runner(path, level, task, conv="sage")
    restored.LoadModel(checkpoint)
    a, b = model.Predict(return_logits=True), restored.Predict(return_logits=True)
    if isinstance(a["logits"], list):
        for left, right in zip(a["logits"], b["logits"]): np.testing.assert_allclose(left, right)
    else:
        np.testing.assert_allclose(a["logits"], b["logits"])
    if level != "link":
        unseen = runner(dataset(name="unlabelled_features", graphs=2, features=True, labels=False, mask=False),
                        level, task, inference=True)
        unseen.LoadModel(checkpoint)
        out = unseen.Predict()
        assert "y_true" not in out
        assert unseen.data_list[0].u.shape == (1, 1)


@pytest.mark.parametrize("conv", ["sage", "gcn", "gatv2"])
def test_graph_feature_batching_matches_independent_predictions(dataset, conv):
    from torch_geometric.data import Batch
    model = runner(dataset(graphs=4, features=True), conv=conv)
    for i, data in enumerate(model.data_list): data.u[:] = i * 1.3
    model.model.eval()
    def forward(data):
        return model.model["head"](model._Encode(data), data.batch, data.u)
    together = forward(Batch.from_data_list(model.data_list)).detach()
    apart = torch.cat([forward(Batch.from_data_list([data])) for data in model.data_list]).detach()
    torch.testing.assert_close(together, apart)


def test_graph_feature_only_signal_is_learned(dataset):
    path = dataset(graphs=30, continuous=True, features=True)
    nodes = pd.read_csv(path / "nodes.csv")
    nodes[["feat_0", "feat_1"]] = 0.
    nodes.to_csv(path / "nodes.csv", index=False)
    graphs = pd.read_csv(path / "graphs.csv")
    graphs["feat_weather"] = graphs["graph_id"] / 30
    graphs["label"] = 2 * graphs["feat_weather"] - 1
    graphs.to_csv(path / "graphs.csv", index=False)
    model = runner(path, task="regression", epochs=55, lr=.02, hidden_dims=(16, 16))
    loss = model.Train()["train_loss"]
    assert loss[-1] < .05 * loss[0]
    assert model.Test()["test_rmse"] < .15


def test_edge_feature_only_signal_is_learned(dataset):
    path = dataset(graphs=4, continuous=True, features=True)
    nodes = pd.read_csv(path / "nodes.csv")
    nodes[["feat_0", "feat_1"]] = 0.
    nodes.to_csv(path / "nodes.csv", index=False)
    edges = pd.read_csv(path / "edges.csv")
    edges["feat_distance"] = np.random.default_rng(12).uniform(-1, 1, len(edges))
    edges["label"] = 2 * edges["feat_distance"] + .3
    edges.to_csv(path / "edges.csv", index=False)
    model = runner(path, "edge", "regression", epochs=65, lr=.02, hidden_dims=(16, 16))
    loss = model.Train()["train_loss"]
    assert loss[-1] < .05 * loss[0]
    assert model.Test()["test_rmse"] < .15


def test_graph_and_edge_features_allow_edgeless_graphs(dataset):
    path = dataset(graphs=3, features=True)
    file = path / "edges.csv"
    pd.read_csv(file).iloc[:0].to_csv(file, index=False)
    for conv in ("sage", "gcn", "gatv2"):
        model = runner(path, conv=conv)
        assert np.isfinite(model.Predict(return_logits=True)["logits"]).all()


def test_legacy_ignored_feature_checkpoint_requires_retraining(dataset, tmp_path):
    model = runner(dataset(features=True))
    file = str(tmp_path / "legacy.pt")
    model.SaveModel(file)
    payload = torch.load(file, weights_only=True)
    del payload["feature_support_version"]
    torch.save(payload, file)
    with pytest.raises(ValueError, match="legacy checkpoint ignored"):
        model.LoadModel(file)


@pytest.mark.parametrize("optional", [False, True])
def test_graph_and_edge_checkpoint_feature_schema(dataset, tmp_path, optional):
    cols = ("feat_weather", "feat_distance") if optional else ()
    model = runner(dataset(features=True), optional_feature_columns=cols)
    checkpoint = str(tmp_path / "schema.pt")
    model.SaveModel(checkpoint)
    path = dataset(name="missing_graph_edge", features=True, labels=False, mask=False)
    for name, col in (("graphs", "feat_weather"), ("edges", "feat_distance")):
        file = path / (name + ".csv")
        pd.read_csv(file).drop(columns=col).to_csv(file, index=False)
    unseen = runner(path, inference=True)
    if optional:
        unseen.LoadModel(checkpoint)
        assert unseen.data_list[0].u.shape == (1, 1)
        assert torch.count_nonzero(unseen.data_list[0].u) == 0
        assert torch.count_nonzero(unseen.data_list[0].edge_attr) == 0
        assert np.isfinite(unseen.Predict(return_logits=True)["logits"]).all()
    else:
        with pytest.raises(ValueError, match="Missing required"):
            unseen.LoadModel(checkpoint)


def test_legacy_node_only_checkpoint_still_loads(dataset, tmp_path):
    model = runner(dataset())
    file = str(tmp_path / "node_only.pt")
    model.SaveModel(file)
    payload = torch.load(file, weights_only=True)
    del payload["feature_support_version"]
    torch.save(payload, file)
    other = runner(model.path)
    other.LoadModel(file)
    np.testing.assert_allclose(model.Predict(return_logits=True)["logits"], other.Predict(return_logits=True)["logits"])


def test_feature_aware_cv_and_raw_weight_roundtrip(dataset, tmp_path):
    model = runner(dataset(features=True), early_stopping=True, early_stopping_patience=1,
                   holdout_group_by="building")
    model.Train()
    prediction = model.Predict(return_logits=True)["logits"]
    assert len(model.CrossValidate(k_folds=3)["fold_metrics"]) == 3
    np.testing.assert_allclose(prediction, model.Predict(return_logits=True)["logits"])
    file = str(tmp_path / "raw_features.pt")
    model.SaveModel(file, include_config=False)
    other = runner(model.path)
    other.LoadModel(file)
    np.testing.assert_allclose(prediction, other.Predict(return_logits=True)["logits"])


@pytest.mark.parametrize("conv", ["sage", "gcn", "gatv2"])
@pytest.mark.parametrize("feature_kind", ["graph", "edge"])
@pytest.mark.parametrize("level,task", [("graph", "classification"), ("node", "classification"),
                                      ("edge", "regression"), ("link", "link_prediction")])
def test_graph_or_edge_features_work_independently(dataset, conv, feature_kind, level, task):
    path = dataset(graphs=10, features=True, continuous=task == "regression")
    filename, column = ("edges", "feat_distance") if feature_kind == "graph" else ("graphs", "feat_weather")
    file = path / (filename + ".csv")
    pd.read_csv(file).drop(columns=column).to_csv(file, index=False)
    model = runner(path, level, task, conv=conv)
    assert model._edge_dim == int(feature_kind == "edge")
    assert model._graph_dim == int(feature_kind == "graph")
    assert np.isfinite(model.Train()["train_loss"]).all()
    assert np.isfinite(list(model.Test().values())).all()
    out = model.Predict(return_logits=True)
    values = np.concatenate(out["logits"]) if isinstance(out["logits"], list) else out["logits"]
    assert np.isfinite(values).all()
