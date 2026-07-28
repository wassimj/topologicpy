"""Unit tests for topologicpy.ANN.

These tests intentionally use tiny CSV fixtures and very short training runs so
that they remain suitable for the full TopologicPy CI matrix.
"""

import importlib
import math
import sys
import types

import numpy as np
import pandas as pd
import pytest
import torch

ann_module = importlib.import_module("topologicpy.ANN")
ANN = ann_module.ANN
_TabularDataset = ann_module._TabularDataset
_activation = ann_module._activation
_parse_feat_cell = ann_module._parse_feat_cell
_pick_device = ann_module._pick_device


@pytest.fixture(autouse=True)
def _suppress_expected_topologicpy_output(capfd):
    """Keep expected diagnostic prints out of normal pytest output."""
    capfd.readouterr()
    yield
    capfd.readouterr()


def _write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _classification_rows(n=12, masks=False):
    rows = []
    for i in range(n):
        row = {
            "id": f"r{i}",
            "feat": f"[{float(i % 2)}, {float((i // 2) % 2)}]",
            "label": "A" if i % 2 == 0 else "B",
        }
        if masks:
            row["train_mask"] = i < 8
            row["val_mask"] = 8 <= i < 10
            row["test_mask"] = i >= 10
        rows.append(row)
    return rows


def _small_classification_ann(path):
    ann = ANN.ByCSVPath(
        str(path),
        task="classification",
        split=(8, 2, 2),
        useMasksIfPresent=False,
        device="cpu",
        randomState=7,
    )
    ann.SetHyperparameters(
        epochs=2,
        batch_size=4,
        lr=0.01,
        hidden_dims=(4,),
        dropout=0.0,
        batch_norm=False,
        early_stopping=False,
        verbose=False,
        device="cpu",
    )
    return ann


def test_parse_feat_cell_accepts_supported_encodings_and_numeric_string():
    assert _parse_feat_cell([1, 2.5]) == [1.0, 2.5]
    assert _parse_feat_cell((1, 2)) == [1.0, 2.0]
    assert _parse_feat_cell(np.asarray([1, 2])) == [1.0, 2.0]
    assert _parse_feat_cell("[1, 2, 3]") == [1.0, 2.0, 3.0]
    assert _parse_feat_cell("1, 2, 3") == [1.0, 2.0, 3.0]
    assert _parse_feat_cell("1.25") == [1.25]
    assert _parse_feat_cell(2) == [2.0]
    assert _parse_feat_cell(2.5) == [2.5]
    assert _parse_feat_cell("") is None
    assert _parse_feat_cell("not numeric") is None


def test_activation_and_device_helpers():
    assert isinstance(_activation("relu"), torch.nn.ReLU)
    assert isinstance(_activation("gelu"), torch.nn.GELU)
    assert isinstance(_activation("elu"), torch.nn.ELU)
    assert isinstance(_activation("tanh"), torch.nn.Tanh)

    with pytest.raises(ValueError):
        _activation("not-an-activation")

    assert _pick_device("cpu").type == "cpu"
    assert _pick_device("cuda").type in ["cpu", "cuda"]
    assert _pick_device("auto").type in ["cpu", "cuda"]


def test_tabular_dataset_with_and_without_labels():
    X = np.asarray([[1, 2], [3, 4]], dtype=float)
    y = np.asarray([0, 1], dtype=np.int64)

    labelled = _TabularDataset(X, y)
    assert len(labelled) == 2
    x0, y0 = labelled[0]
    assert torch.allclose(x0, torch.tensor([1.0, 2.0]))
    assert int(y0) == 0

    unlabelled = _TabularDataset(X, None)
    assert torch.allclose(unlabelled[1], torch.tensor([3.0, 4.0]))


def test_by_csv_path_parses_feat_column_and_accepts_percentage_style_split(tmp_path):
    csv_path = _write_csv(tmp_path / "classification.csv", _classification_rows(10))

    ann = ANN.ByCSVPath(
        str(csv_path),
        task="classification",
        split=(80, 10, 10),
        useMasksIfPresent=False,
        device="cpu",
        randomState=3,
    )

    assert ann.X.shape == (10, 2)
    assert len(ann.idx_train) == 8
    assert len(ann.idx_val) == 1
    assert len(ann.idx_test) == 1
    assert ann.num_classes == 2
    assert set(ann.class_to_index.keys()) == {"A", "B"}
    assert "rows=10" in ann.Summary()
    assert "features=2" in ann.Summary()


def test_by_csv_path_uses_mask_columns_when_present(tmp_path):
    csv_path = _write_csv(tmp_path / "masked.csv", _classification_rows(12, masks=True))

    ann = ANN.ByCSVPath(str(csv_path), task="classification", device="cpu")

    assert ann.ids.tolist() == [f"r{i}" for i in range(12)]
    assert ann.idx_train.tolist() == list(range(8))
    assert ann.idx_val.tolist() == [8, 9]
    assert ann.idx_test.tolist() == [10, 11]


def test_by_csv_path_with_explicit_feature_columns_and_missing_prediction_columns(tmp_path):
    train_path = _write_csv(
        tmp_path / "columns_train.csv",
        [
            {"id": "a", "x": 0.0, "y": 1.0, "z": 2.0, "label": 0},
            {"id": "b", "x": 1.0, "y": 0.0, "z": 2.0, "label": 1},
            {"id": "c", "x": 0.0, "y": 0.0, "z": 1.0, "label": 0},
            {"id": "d", "x": 1.0, "y": 1.0, "z": 0.0, "label": 1},
        ],
    )
    pred_path = _write_csv(
        tmp_path / "columns_pred.csv",
        [
            {"id": "p0", "x": 0.5, "y": 0.5},
            {"id": "p1", "x": 1.5, "y": 0.0},
        ],
    )

    ann = ANN.ByCSVPath(
        str(train_path),
        task="classification",
        featuresKeys=["x", "y", "z"],
        split=(2, 1, 1),
        device="cpu",
        randomState=1,
    )
    ann.SetHyperparameters(hidden_dims=(3,), dropout=0.0, batch_norm=False, batch_size=2, device="cpu")

    out = ann.Predict(path=str(pred_path), attach_to_df=True)

    assert ann._feature_schema_mode == "columns"
    assert ann._feature_schema_keys == ["x", "y", "z"]
    assert out["pred"].shape == (2,)
    assert "z" in out["df"].columns
    assert out["df"]["z"].tolist() == [0.0, 0.0]


def test_predict_new_feat_csv_trims_and_pads_to_frozen_training_dimension(tmp_path):
    train_path = _write_csv(tmp_path / "feat_train.csv", _classification_rows(6))
    pred_path = _write_csv(
        tmp_path / "feat_pred.csv",
        [
            {"id": "p0", "feat": "[1.0]"},
            {"id": "p1", "feat": "[0.0, 1.0, 99.0]"},
        ],
    )

    ann = ANN.ByCSVPath(
        str(train_path),
        task="classification",
        split=(4, 1, 1),
        useMasksIfPresent=False,
        device="cpu",
    )
    ann.SetHyperparameters(hidden_dims=(4,), dropout=0.0, batch_norm=False, batch_size=2, device="cpu")

    out = ann.Predict(path=str(pred_path), return_proba=True, return_logits=True, attach_to_df=True)

    assert ann._feature_schema_mode == "feat"
    assert ann._feature_schema_dim == 2
    assert out["pred"].shape == (2,)
    assert out["proba"].shape == (2, 2)
    assert out["logits"].shape == (2, 2)
    assert "pred" in out["df"].columns


def test_set_hyperparameters_rebuilds_resplits_and_rejects_unknown_key(tmp_path):
    csv_path = _write_csv(tmp_path / "classification.csv", _classification_rows(12))
    ann = ANN.ByCSVPath(str(csv_path), task="classification", split=(8, 2, 2), device="cpu")
    old_model = ann.model

    ann.SetHyperparameters(hidden_dims=(3,), dropout=0.0, batch_norm=False, split=(6, 3, 3), device="cpu")

    assert ann.model is not old_model
    assert ann.config.hidden_dims == (3,)
    assert len(ann.idx_train) == 6
    assert len(ann.idx_val) == 3
    assert len(ann.idx_test) == 3

    with pytest.raises(ValueError):
        ann.SetHyperparameters(not_a_real_hyperparameter=True)


def test_classification_train_validate_test_predict_and_save_load(tmp_path):
    csv_path = _write_csv(tmp_path / "classification.csv", _classification_rows(12, masks=True))
    ann = _small_classification_ann(csv_path)

    history = ann.Train()
    val = ann.Validate()
    test = ann.Test()
    pred = ann.Predict(return_proba=True, return_logits=True, attach_to_df=True)

    assert len(history["train_loss"]) == 2
    assert len(history["val_loss"]) == 2
    assert all(math.isfinite(v) for v in history["train_loss"])
    assert "accuracy" in val
    assert "f1_macro" in val
    assert val["y_true"].shape[0] == 2
    assert test["y_pred"].shape[0] == 2
    assert pred["pred"].shape == (12,)
    assert pred["proba"].shape == (12, 2)
    assert pred["logits"].shape == (12, 2)
    assert "pred_label" in pred
    assert "proba_0" in pred["df"].columns

    model_path = tmp_path / "ann_checkpoint.pt"
    ann.SaveModel(str(model_path))
    assert model_path.exists()

    loaded = ANN()
    loaded.LoadModel(str(model_path))
    loaded_pred = loaded.Predict(path=str(csv_path), attach_to_df=False)
    assert loaded_pred["pred"].shape == (12,)


def test_save_model_appends_pt_extension(tmp_path):
    csv_path = _write_csv(tmp_path / "classification.csv", _classification_rows(6))
    ann = ANN.ByCSVPath(str(csv_path), task="classification", split=(4, 1, 1), device="cpu")
    ann.SetHyperparameters(hidden_dims=(2,), dropout=0.0, batch_norm=False, batch_size=2, device="cpu")

    out_path = tmp_path / "model_without_extension"
    ann.SaveModel(str(out_path))

    assert (tmp_path / "model_without_extension.pt").exists()


def test_load_raw_state_dict_requires_existing_model_context(tmp_path):
    csv_path = _write_csv(tmp_path / "classification.csv", _classification_rows(6))
    ann = ANN.ByCSVPath(str(csv_path), task="classification", split=(4, 1, 1), device="cpu")
    ann.SetHyperparameters(hidden_dims=(2,), dropout=0.0, batch_norm=False, batch_size=2, device="cpu")

    raw_path = tmp_path / "raw.pt"
    ann.SaveModel(str(raw_path), include_config=False)

    clone = ANN.ByCSVPath(str(csv_path), task="classification", split=(4, 1, 1), device="cpu")
    clone.SetHyperparameters(hidden_dims=(2,), dropout=0.0, batch_norm=False, batch_size=2, device="cpu")
    clone.LoadModel(str(raw_path))
    out = clone.Predict(attach_to_df=False)
    assert out["pred"].shape == (6,)


def test_regression_train_validate_test_and_predict(tmp_path):
    rows = []
    for i in range(12):
        rows.append({
            "id": f"r{i}",
            "feat": f"[{float(i)}, {float(i % 3)}]",
            "label": float(2 * i + 1),
            "train_mask": i < 8,
            "val_mask": 8 <= i < 10,
            "test_mask": i >= 10,
        })
    csv_path = _write_csv(tmp_path / "regression.csv", rows)

    ann = ANN.ByCSVPath(str(csv_path), task="regression", device="cpu")
    ann.SetHyperparameters(
        epochs=1,
        batch_size=4,
        lr=0.001,
        hidden_dims=(4,),
        dropout=0.0,
        batch_norm=False,
        early_stopping=False,
        verbose=False,
        device="cpu",
    )

    history = ann.Train()
    val = ann.Validate()
    test = ann.Test()
    pred = ann.Predict(attach_to_df=True)

    assert len(history["train_loss"]) == 1
    assert "mae" in val
    assert "rmse" in val
    assert "r2" in val
    assert test["y_pred"].shape == (2,)
    assert pred["pred"].shape == (12,)
    assert "pred" in pred["df"].columns


def test_empty_training_or_validation_splits_raise_clear_errors(tmp_path):
    csv_path = _write_csv(tmp_path / "tiny.csv", _classification_rows(2))

    empty_val = ANN.ByCSVPath(
        str(csv_path),
        task="classification",
        split=(1, 0, 1),
        useMasksIfPresent=False,
        device="cpu",
    )
    empty_val.SetHyperparameters(hidden_dims=(2,), dropout=0.0, batch_norm=False, batch_size=2, device="cpu")

    with pytest.raises(RuntimeError, match="validation split is empty"):
        empty_val.Train()
    with pytest.raises(RuntimeError, match="requested split is empty"):
        empty_val.Validate()

    empty_train = ANN.ByCSVPath(
        str(csv_path),
        task="classification",
        split=(0, 1, 1),
        useMasksIfPresent=False,
        device="cpu",
    )
    empty_train.SetHyperparameters(hidden_dims=(2,), dropout=0.0, batch_norm=False, batch_size=2, device="cpu")

    with pytest.raises(RuntimeError, match="training split is empty"):
        empty_train.Train()


def test_invalid_inputs_raise_clear_exceptions(tmp_path):
    missing_path = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        ANN.ByCSVPath(str(missing_path), device="cpu")

    bad_feat = _write_csv(tmp_path / "bad_feat.csv", [{"feat": "abc", "label": 0}])
    with pytest.raises(ValueError, match="Could not parse features"):
        ANN.ByCSVPath(str(bad_feat), device="cpu")

    bad_task = _write_csv(tmp_path / "bad_task.csv", [{"feat": "[1, 2]", "label": 0}])
    with pytest.raises(ValueError, match="Unknown task"):
        ANN.ByCSVPath(str(bad_task), task="not-a-task", device="cpu")

    bad_split = _write_csv(tmp_path / "bad_split.csv", _classification_rows(4))
    with pytest.raises(ValueError, match="non-negative"):
        ANN.ByCSVPath(str(bad_split), split=(-1, 1, 1), useMasksIfPresent=False, device="cpu")
    with pytest.raises(ValueError, match="sum to a positive"):
        ANN.ByCSVPath(str(bad_split), split=(0, 0, 0), useMasksIfPresent=False, device="cpu")


def test_predict_requires_model_and_loaded_data(tmp_path):
    ann = ANN()
    with pytest.raises(RuntimeError, match="No model"):
        ann.Predict()

    csv_path = _write_csv(tmp_path / "classification.csv", _classification_rows(4))
    trained_context = ANN.ByCSVPath(str(csv_path), task="classification", split=(2, 1, 1), device="cpu")
    trained_context.SetHyperparameters(hidden_dims=(2,), dropout=0.0, batch_norm=False, batch_size=2, device="cpu")

    no_data = ANN()
    no_data.model = trained_context.model
    no_data.config = trained_context.config
    with pytest.raises(RuntimeError, match="No dataset loaded"):
        no_data.Predict()

    bad_pred = _write_csv(tmp_path / "bad_pred.csv", [{"feat": "abc"}])
    with pytest.raises(ValueError, match="Could not parse features"):
        trained_context.Predict(path=str(bad_pred))


def test_plot_history_uses_go_figure_when_history_exists(monkeypatch):
    class FakeScatter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeFigure:
        def __init__(self):
            self.traces = []
            self.layout = None

        def add_trace(self, trace):
            self.traces.append(trace)

        def update_layout(self, **kwargs):
            self.layout = kwargs

    fake_go = types.SimpleNamespace(Figure=FakeFigure, Scatter=FakeScatter)
    monkeypatch.setattr(ann_module, "go", fake_go)

    ann = ANN()
    ann.history = {"train_loss": [1.0, 0.5], "val_loss": [1.2, 0.7]}
    fig = ann.PlotHistory(title="History")

    assert isinstance(fig, FakeFigure)
    assert len(fig.traces) == 2
    assert fig.layout["title"] == "History"


def test_plot_history_requires_history(monkeypatch):
    class FakeFigure:
        pass

    fake_go = types.SimpleNamespace(Figure=FakeFigure)
    monkeypatch.setattr(ann_module, "go", fake_go)

    ann = ANN()
    with pytest.raises(RuntimeError, match="No history"):
        ann.PlotHistory()


def test_plot_confusion_matrix_row_normalization_and_fake_plotly(monkeypatch):
    captured = {}

    class FakeFig:
        def __init__(self):
            self.xaxes = None
            self.yaxes = None

        def update_xaxes(self, **kwargs):
            self.xaxes = kwargs

        def update_yaxes(self, **kwargs):
            self.yaxes = kwargs

    class FakePlotly:
        @staticmethod
        def FigureByConfusionMatrix(**kwargs):
            captured.update(kwargs)
            return FakeFig()

    fake_plotly_module = types.ModuleType("topologicpy.Plotly")
    fake_plotly_module.Plotly = FakePlotly
    monkeypatch.setitem(sys.modules, "topologicpy.Plotly", fake_plotly_module)
    monkeypatch.setattr(
        ann_module,
        "confusion_matrix",
        lambda y_true, y_pred, labels=None: np.asarray([[2, 1], [0, 3]], dtype=int),
    )

    ann = ANN()
    ann.config.task = "classification"
    ann.idx_test = np.asarray([0, 1, 2, 3])
    ann._evaluate_split = lambda idx: {
        "y_true": np.asarray([0, 0, 1, 1]),
        "y_pred": np.asarray([0, 1, 1, 1]),
    }

    fig = ann.PlotConfusionMatrix(split="test", normalize=True, mantissa=1)
    matrix = np.asarray(captured["matrix"], dtype=float)

    assert fig.xaxes["ticktext"] == ["0", "1"]
    assert fig.yaxes["ticktext"] == ["0", "1"]
    assert matrix.shape == (2, 2)
    assert np.allclose(matrix.sum(axis=1), np.asarray([1.0, 1.0]))
    assert captured["maxValue"] == 1.0


def test_plot_confusion_matrix_rejects_regression():
    ann = ANN()
    ann.config.task = "regression"
    with pytest.raises(ValueError, match="Only valid for classification"):
        ann.PlotConfusionMatrix()


def test_plot_parity_rejects_classification():
    ann = ANN()
    ann.config.task = "classification"
    with pytest.raises(ValueError, match="Only valid for regression"):
        ann.PlotParity()
