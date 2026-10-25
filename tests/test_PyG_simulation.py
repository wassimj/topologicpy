"""Simulation-ready PyG regression coverage; uses real PyTorch/PyG."""
from __future__ import annotations
import copy
import random
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("sklearn")
from topologicpy.PyG import PyG
from test_PyG import dataset, runner, cpu_determinism


def vectorize(path, level, physical=False):
    name = {"graph": "graphs", "node": "nodes", "edge": "edges"}[level]
    file = path / (name + ".csv")
    frame = pd.read_csv(file)
    values = frame["label"].to_numpy(dtype=float)
    frame["vx"], frame["vy"] = values, 2 * values + 1
    frame.drop(columns="label").to_csv(file, index=False)
    if physical:
        nodes = pd.read_csv(path / "nodes.csv")
        nodes["x"], nodes["y"], nodes["z"] = np.arange(len(nodes)), 0., 0.
        nodes["volume"], nodes["boundary"] = 1. + np.arange(len(nodes)) % 3, np.arange(len(nodes)) % 2 == 0
        nodes.to_csv(path / "nodes.csv", index=False)
        graph = pd.read_csv(path / "graphs.csv")
        graph["time"], graph["dt"] = graph.graph_id * .1, .1
        graph.to_csv(path / "graphs.csv", index=False)
    return path


def test_training_reproducibility_ignores_and_preserves_outside_rng(dataset):
    path = dataset(features=True)
    outside = PyG._CaptureRNG()
    a = runner(path, dropout=.3, epochs=4)
    a.Train()
    after = PyG._CaptureRNG()
    assert outside["python"] == after["python"]
    assert outside["numpy"] == after["numpy"]
    assert torch.equal(outside["torch"], after["torch"])
    random.seed(1233); np.random.seed(322); torch.manual_seed(333)
    b = runner(path, dropout=.3, epochs=4)
    b.Train()
    for key, value in a.model.state_dict().items():
        torch.testing.assert_close(value, b.model.state_dict()[key], rtol=0, atol=0)
    c = runner(path, dropout=.3, random_state=1234)
    assert any(not torch.equal(v, c.model.state_dict()[k]) for k, v in a.model.state_dict().items())


@pytest.mark.parametrize("defect", ["duplicate_graph", "null_graph", "duplicate_node", "null_node", "orphan_node", "orphan_edge", "null_endpoint", "unknown_endpoint"])
def test_strict_csv_identity_validation(dataset, defect):
    path = dataset(graphs=2)
    name = "graphs" if "graph" in defect else "nodes" if "node" in defect else "edges"
    file = path / (name + ".csv")
    frame = pd.read_csv(file)
    if defect == "duplicate_graph": frame = pd.concat([frame, frame.iloc[:1]])
    elif defect == "null_graph": frame.loc[0, "graph_id"] = None
    elif defect == "duplicate_node": frame = pd.concat([frame, frame.iloc[:1]])
    elif defect == "null_node": frame.loc[0, "node_id"] = None
    elif defect in ("orphan_node", "orphan_edge"): frame.loc[0, "graph_id"] = 999
    elif defect == "null_endpoint": frame.loc[0, "src_id"] = None
    else: frame.loc[0, "src_id"] = "not-a-node"
    frame.to_csv(file, index=False)
    with pytest.raises(ValueError): runner(path)


@pytest.mark.parametrize("conv", ["sage", "gcn", "gatv2"])
@pytest.mark.parametrize("level", ["graph", "node", "edge"])
def test_vector_regression_shapes_physical_units_and_checkpoints(dataset, tmp_path, conv, level):
    path = vectorize(dataset(continuous=True, features=True), level)
    model = runner(path, level, "regression", target_columns=("vx", "vy"), target_units=("m/s", "m/s"),
                   normalize_features=True, normalize_targets=True, conv=conv)
    assert model._num_outputs == 2
    assert np.isfinite(model.Train()["train_loss"]).all()
    metrics = model.Test()
    assert "test_rmse_vx" in metrics and "test_rmse_vy" in metrics
    out = model.Predict(return_logits=True, attach_to_data=True)
    values = out["pred"] if level == "graph" else np.concatenate(out["pred"])
    assert values.ndim == 2 and values.shape[1] == 2
    truth = out["y_true"] if level == "graph" else np.concatenate(out["y_true"])
    np.testing.assert_allclose(truth[:, 1], 2 * truth[:, 0] + 1, rtol=1e-5, atol=1e-5)
    logits = out["logits"] if level == "graph" else np.concatenate(out["logits"])
    scaler = model._scalers["targets"]
    np.testing.assert_allclose(values, logits * scaler["scale"] + scaler["mean"], atol=1e-5)
    file = str(tmp_path / "vector.pt")
    model.SaveModel(file)
    other = runner(path, level, "regression", target_columns=("vx", "vy"))
    other.LoadModel(file)
    result = other.Predict()["pred"]
    if level != "graph": result = np.concatenate(result)
    np.testing.assert_allclose(result, values)


@pytest.mark.parametrize("level", ["node", "edge"])
def test_whole_simulation_splits_and_fold_scalers(dataset, level):
    path = vectorize(dataset(continuous=True, features=True), level)
    model = runner(path, level, "regression", split_mode="graph", holdout_group_by="building",
                   target_columns=("vx", "vy"), normalize_features=True, normalize_targets=True,
                   early_stopping=True, early_stopping_patience=1)
    sets = [{int(d.topologicpy_graph_index) // 2 for d in data} for data in (model.train_set, model.val_set, model.test_set)]
    assert all(sets) and not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    prefix = "edge_" if level == "edge" else ""
    for data in model.test_set:
        assert getattr(data, prefix + "test_mask").all()
        assert not getattr(data, prefix + "train_mask").any()
    model.Train()
    before = copy.deepcopy(model._scalers)
    report = model.CrossValidate(k_folds=3)
    assert model._scalers == before
    for fold in report["fold_metrics"]:
        groups = [{i // 2 for i in fold[key]} for key in ("train_indices", "val_indices", "test_indices")]
        assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        expected = np.mean([i / 20 for i in fold["train_indices"]])
        assert fold["normalization"]["x"]["mean"][1] == pytest.approx(expected, abs=1e-6)
    assert np.isfinite(list(model.Test().values())).all()


def test_target_and_feature_scalers_exclude_heldout_values(dataset):
    path = dataset(graphs=10, continuous=True)
    file = path / "nodes.csv"
    frame = pd.read_csv(file)
    held = ~frame.train_mask
    frame.loc[held, "feat_0"] = 10000.
    frame.loc[held, "label"] = 50000.
    frame.to_csv(file, index=False)
    model = runner(path, "node", "regression", normalize_features=True, normalize_targets=True)
    assert model._scalers["x"]["mean"][0] == pytest.approx(frame.loc[frame.train_mask, "feat_0"].mean())
    assert model._scalers["targets"]["mean"][0] == pytest.approx(frame.loc[frame.train_mask, "label"].mean())
    assert model.data_list[0].x[~model.data_list[0].train_mask, 0].min() > 1000


def test_graph_cross_validation_scalers_fit_only_inner_training(dataset):
    path = dataset(continuous=True, features=True)
    model = runner(path, task="regression", normalize_features=True, normalize_targets=True,
                   early_stopping=True, holdout_group_by="building")
    report = model.CrossValidate(k_folds=3)
    for fold in report["fold_metrics"]:
        expected = np.mean([i / 10 for i in fold["train_indices"]])
        assert fold["normalization"]["targets"]["mean"][0] == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("level,task", [("graph", "classification"), ("node", "regression"), ("edge", "regression"), ("link", "link_prediction")])
def test_checkpoint_resume_matches_uninterrupted_training(dataset, tmp_path, level, task):
    path = dataset(continuous=task == "regression", features=True)
    settings = dict(dropout=.25, normalize_features=True, normalize_targets=task == "regression")
    full = runner(path, level, task, epochs=6, **settings)
    full.Train()
    part = runner(path, level, task, epochs=2, **settings)
    part.Train()
    file = str(tmp_path / "resume.pt")
    part.SaveModel(file)
    restored = runner(path, level, task, **settings)
    restored.LoadModel(file, resume_training=True)
    assert restored._epochs_completed == 2
    restored.Train(epochs=4, resume=True)
    assert restored._epochs_completed == 6
    assert len(restored.history["train_loss"]) == 6
    for key, value in full.model.state_dict().items():
        torch.testing.assert_close(value, restored.model.state_dict()[key], rtol=0, atol=0)
    np.testing.assert_allclose(full.history["train_loss"], restored.history["train_loss"], rtol=0, atol=0)


def test_resume_rejects_changed_dataset(dataset, tmp_path):
    path = dataset()
    model = runner(path)
    model.Train()
    file = str(tmp_path / "resume.pt")
    model.SaveModel(file)
    nodes = pd.read_csv(path / "nodes.csv")
    nodes.loc[0, "feat_0"] += 100
    nodes.to_csv(path / "nodes.csv", index=False)
    other = runner(path)
    with pytest.raises(ValueError, match="different dataset"):
        other.LoadModel(file, resume_training=True)


class StructuredNodeModel(torch.nn.Module):
    def __init__(self, context):
        super().__init__()
        self.linear = torch.nn.Linear(context["node_dim"], context["output_dim"])
        self.seen_hierarchy = False
    def forward(self, data):
        assert data.pos.shape[0] == data.num_nodes
        assert data.dt.numel() == 1
        if getattr(data, "hierarchy", None) is not None:
            self.seen_hierarchy = True
            assert data.num_graphs == 1
        return {"prediction": self.linear(data.x), "embedding": data.x}


def test_registered_factory_receives_structured_inputs_and_reloads(dataset, tmp_path):
    path = vectorize(dataset(continuous=True), "node", physical=True)
    PyG.RegisterModel("structured-regression", StructuredNodeModel)
    model = runner(path, "node", "regression", target_columns=("vx", "vy"), model_name="structured-regression",
                   node_weight_column="volume", boundary_column="boundary", normalize_targets=True)
    for row in model._identities:
        model.SetGraphData(row["graph_id"], hierarchy={"cluster_index": torch.arange(12) // 3})
    model.Train()
    assert model.model.seen_hierarchy
    assert np.isfinite(list(model.Test().values())).all()
    assert len(model.Predict(return_embeddings=True)["emb"]) == 20
    file = str(tmp_path / "custom.pt")
    model.SaveModel(file)
    other = runner(path, "node", "regression", target_columns=("vx", "vy"), node_weight_column="volume", boundary_column="boundary")
    other.LoadModel(file, resume_training=True)
    assert other.config.model_name == "structured-regression"
    assert other.data_list[0].hierarchy["cluster_index"].tolist() == (torch.arange(12) // 3).tolist()
    other.Train(epochs=1, resume=True)
    assert other._epochs_completed == 3
    PyG._ModelFactories.pop("structured-regression")
    missing = runner(path, "node", "regression", target_columns=("vx", "vy"))
    with pytest.raises(ValueError, match="Register custom model"): missing.LoadModel(file)


def test_hierarchy_inputs_and_source_ids_are_validated_without_ontology(dataset):
    model = runner(dataset(), ontology=False)
    ids = model.Predict()["graph_ids"]
    assert ids == list(range(20))
    with pytest.raises(ValueError): model.SetGraphData(999, dt=.1)
    with pytest.raises(ValueError): model.SetGraphData(0, dt=-1)
    with pytest.raises(ValueError): model.SetGraphData(0, node_weight=[1, 2])
    with pytest.raises(ValueError): model.SetGraphData(0, hierarchy={"bad": [1, 2]})
    model.SetGraphData(0, hierarchy={"cluster_index": torch.arange(12) // 2})
    assert model._Loader(model.data_list, batch_size=10).batch_size == 1


def test_physics_metrics_known_vector_weighting_boundaries_and_rollout():
    prediction = np.asarray([[1., 2.], [3., 4.]])
    truth = np.zeros_like(prediction)
    out = PyG.PhysicsMetrics(prediction, truth, weights=[1, 3], boundaryMask=[True, False], componentNames=["vx", "vy"])
    assert out["components"]["vx"]["rmse"] == pytest.approx(np.sqrt(7))
    assert out["components"]["vy"]["rmse"] == pytest.approx(np.sqrt(13))
    assert out["rmse"] == pytest.approx(np.sqrt(10))
    assert out["boundary"]["count"] == 1
    assert out["boundary"]["mae"] == 1.5
    p = np.stack([np.ones((2, 2)), np.ones((2, 2)) * 3])
    rollout = PyG.EvaluateRollout(p, np.zeros_like(p), weights=[1, 3], timeStep=[.1, .2])
    np.testing.assert_allclose(rollout["times"], [.1, .3])
    assert rollout["rmse"] == [1., 3.] and rollout["rmse_growth"] == 2.
    with pytest.raises(ValueError): PyG.PhysicsMetrics(prediction, truth, weights=[0, 0])
    with pytest.raises(ValueError): PyG.EvaluateRollout(p, np.zeros_like(p), timeStep=0)


def test_model_physics_evaluation_uses_units_weights_and_boundaries(dataset):
    path = vectorize(dataset(continuous=True), "node", physical=True)
    model = runner(path, "node", "regression", target_columns=("vx", "vy"), target_units=("m/s", "m/s"),
                   node_weight_column="volume", boundary_column="boundary", normalize_targets=True)
    model.Train()
    result = model.EvaluatePhysics("test")
    assert result["units"] == ["m/s", "m/s"]
    assert result["count"] > 0 and result["vector_rmse"] > 0
    assert set(result["components"]) == {"vx", "vy"}


def test_early_stopping_resume_preserves_continuation_and_best_weights(dataset, tmp_path):
    path = dataset(features=True)
    settings = dict(dropout=.2, early_stopping=True, early_stopping_patience=3)
    full = runner(path, epochs=7, **settings)
    full.Train()
    part = runner(path, epochs=2, **settings)
    part.Train()
    file = str(tmp_path / "early.pt")
    part.SaveModel(file)
    other = runner(path, **settings)
    other.LoadModel(file, resume_training=True)
    other.Train(epochs=5, resume=True)
    for key, value in full.model.state_dict().items():
        torch.testing.assert_close(value, other.model.state_dict()[key], rtol=0, atol=0)
    np.testing.assert_allclose(full.history["val_loss"], other.history["val_loss"], rtol=0, atol=0)


def test_inference_uses_saved_scalers_without_refitting_on_new_data(dataset, tmp_path):
    path = vectorize(dataset(continuous=True, features=True), "node")
    model = runner(path, "node", "regression", target_columns=("vx", "vy"), normalize_features=True, normalize_targets=True)
    model.Train()
    file = str(tmp_path / "scaled.pt")
    model.SaveModel(file)
    unseen_path = dataset(name="unseen", labels=False, mask=False, features=True)
    csv = unseen_path / "nodes.csv"
    frame = pd.read_csv(csv)
    frame["feat_0"] += 1000
    frame.to_csv(csv, index=False)
    unseen = runner(unseen_path, "node", "regression", inference=True)
    unseen.LoadModel(file)
    assert unseen._scalers == model._scalers
    assert unseen.Predict()["pred"][0].shape == (12, 2)
    assert unseen.data_list[0].x[:, 0].min() > 1000
    with pytest.raises(ValueError, match="full checkpoint"): model.SaveModel(str(tmp_path / "bad.pt"), include_config=False)


def test_prediction_and_evaluation_do_not_advance_external_rng(dataset):
    model = runner(dataset(), dropout=.2)
    model.Train()
    before = PyG._CaptureRNG()
    model.Predict(); model.Test(); model.Validate()
    after = PyG._CaptureRNG()
    assert torch.equal(before["torch"], after["torch"])
    assert before["python"] == after["python"] and before["numpy"] == after["numpy"]


def test_vector_parity_selects_one_physical_component(dataset, monkeypatch):
    from topologicpy.Plotly import Plotly
    path = vectorize(dataset(continuous=True), "node")
    model = runner(path, "node", "regression", target_columns=("vx", "vy"), normalize_targets=True)
    monkeypatch.setattr(Plotly, "FigureByCorrelation", staticmethod(lambda **kwargs: kwargs))
    with pytest.raises(ValueError, match="Select a component"): model.PlotParity()
    result = model.PlotParity(component="vy")
    output = model.Predict(split="test")
    expected = np.concatenate(output["y_true"])[:, 1]
    np.testing.assert_allclose(result["actual"], expected)


def test_model_validation_and_checkpoint_option_errors(dataset, tmp_path):
    path = dataset()
    with pytest.raises(ValueError): runner(path, task="classification", target_columns=("vx",))
    with pytest.raises(ValueError): runner(path, normalize_targets=True)
    with pytest.raises(ValueError): runner(path, model_options={"bad": object()})
    with pytest.raises(ValueError): PyG.RegisterModel("default", lambda context: None)
    PyG.RegisterModel("not-a-module", lambda context: {})
    with pytest.raises(ValueError, match="nn.Module"): runner(path, model_name="not-a-module")
    PyG._ModelFactories.pop("not-a-module")
    model = runner(path)
    model.SaveModel(str(tmp_path / "raw.pt"), include_config=False)
    with pytest.raises(ValueError, match="full training checkpoint"):
        model.LoadModel(str(tmp_path / "raw.pt"), resume_training=True)


def test_seed_update_before_training_matches_new_seed_instance(dataset):
    path = dataset(features=True)
    a = runner(path)
    a.SetHyperparameters(random_state=123)
    b = runner(path, random_state=123)
    a.Train(); b.Train()
    for key, value in a.model.state_dict().items():
        torch.testing.assert_close(value, b.model.state_dict()[key], rtol=0, atol=0)


def test_early_stop_resume_after_nonbest_epoch_is_exact(dataset, tmp_path, monkeypatch):
    path = dataset(features=True)
    settings = dict(dropout=.2, early_stopping=True, early_stopping_patience=2)
    schedule = [1., .8, 1.2, 1.4, 1.6, 1.8]
    def fixed_validation(model):
        monkeypatch.setattr(model, "_eval_epoch_graph", lambda loader: schedule[model._epochs_completed])
    full = runner(path, epochs=6, **settings)
    fixed_validation(full)
    full.Train()
    part = runner(path, epochs=3, **settings)
    fixed_validation(part)
    part.Train()
    file = str(tmp_path / "nonbest.pt")
    part.SaveModel(file)
    other = runner(path, **settings)
    other.LoadModel(file, resume_training=True)
    fixed_validation(other)
    other.Train(epochs=3, resume=True)
    assert full._epochs_completed == other._epochs_completed == 4
    assert full.history["val_loss"] == other.history["val_loss"]
    for key, value in full.model.state_dict().items():
        torch.testing.assert_close(value, other.model.state_dict()[key], rtol=0, atol=0)
    other.Train(epochs=2, resume=True)
    assert other._epochs_completed == 4
    for key, value in full.model.state_dict().items():
        torch.testing.assert_close(value, other.model.state_dict()[key], rtol=0, atol=0)


def test_custom_vector_output_cannot_silently_transpose_components(dataset):
    class BadVectorModel(torch.nn.Module):
        def __init__(self, context):
            super().__init__()
            self.linear = torch.nn.Linear(context["node_dim"], context["output_dim"])
        def forward(self, data): return self.linear(data.x).T
    PyG.RegisterModel("bad-vector-shape", BadVectorModel)
    path = vectorize(dataset(continuous=True), "node")
    model = runner(path, "node", "regression", target_columns=("vx", "vy"), model_name="bad-vector-shape")
    with pytest.raises(ValueError, match="Custom prediction shape"): model.Train()
    PyG._ModelFactories.pop("bad-vector-shape")
