# Copyright (c) 2026
# TopologicPy-style ANN helper for tabular ML using PyTorch (no PyG required).

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Union, Dict, Any, Callable

import os
import ast
import json

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Optional dependencies (kept optional to match TopologicPy-style soft deps)
try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px = None
    go = None

try:
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
    )
except Exception:
    accuracy_score = f1_score = precision_score = recall_score = None
    confusion_matrix = None
    mean_absolute_error = mean_squared_error = r2_score = None


@dataclass
class _ANNConfig:
    """
    Internal run configuration for ANN.
    """
    task: str = "classification"             # "classification" | "regression"
    device: str = "auto"                     # "auto" | "cpu" | "cuda"
    random_state: int = 42
    shuffle: bool = True

    # Dataset schema
    label_header: str = "label"
    feat_header: str = "feat"
    id_header: str = "id"
    features_keys: Optional[List[str]] = None  # if not None, build features from explicit columns

    # Splits: either masks in CSV or ratio split
    use_masks_if_present: bool = True
    split: Tuple[float, float, float] = (0.8, 0.1, 0.1)   # train/val/test

    # Training
    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"                 # "adam" | "adamw" | "sgd"
    gradient_clip_norm: Optional[float] = 1.0
    early_stopping: bool = True
    early_stopping_patience: int = 12

    # Model
    hidden_dims: Tuple[int, ...] = (128, 128)
    activation: str = "relu"                 # "relu" | "gelu" | "elu" | "tanh"
    dropout: float = 0.2
    batch_norm: bool = True

    # Outputs / logging
    verbose: bool = True


class _TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray], ids: Optional[np.ndarray] = None):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = None if y is None else y
        self.ids = None if ids is None else ids

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        x = self.X[idx]
        if self.y is None:
            return x
        y = self.y[idx]
        return x, y


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _pick_device(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _parse_feat_cell(cell: Any) -> Optional[List[float]]:
    """
    Parse a CSV cell holding features. Supports:
    - Python list string: "[1,2,3]"
    - JSON list string:   "[1,2,3]"
    - Comma-separated:    "1,2,3"
    - Already list/tuple
    """
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return None
    if isinstance(cell, (list, tuple, np.ndarray)):
        return [float(v) for v in cell]
    if isinstance(cell, (int, float)):
        return [float(cell)]
    s = str(cell).strip()
    if not s:
        return None
    # try JSON / python literal
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return [float(x) for x in v]
    except Exception:
        pass
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)):
            return [float(x) for x in v]
    except Exception:
        pass
    # comma-separated
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip() != ""]
        try:
            return [float(p) for p in parts]
        except Exception:
            return None
    return None


def _activation(name: str) -> nn.Module:
    n = (name or "relu").lower()
    if n == "relu":
        return nn.ReLU()
    if n == "gelu":
        return nn.GELU()
    if n == "elu":
        return nn.ELU()
    if n == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown activation '{name}'.")


class _MLP(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dims: Tuple[int, ...],
                 out_dim: int,
                 activation: str = "relu",
                 dropout: float = 0.0,
                 batch_norm: bool = False):
        super().__init__()
        dims = [in_dim] + list(hidden_dims)
        layers: List[nn.Module] = []
        act = _activation(activation)

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(act)
            if dropout and dropout > 0:
                layers.append(nn.Dropout(p=float(dropout)))

        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ANN:
    """
    ANN is a TopologicPy-style helper class for regular tabular ML using PyTorch.

    The class mirrors the high-level API of :class:`topologicpy.PyG.PyG` but targets
    non-graph datasets (rows in a CSV file).

    Typical workflow
    ----------------
    1. Load dataset from CSV:
       >>> ann = ANN.ByCSVPath("data.csv", task="classification")

    2. Set hyperparameters / model shape:
       >>> ann.SetHyperparameters(hidden_dims=(128,128), batch_norm=True, dropout=0.2)

    3. Train, validate, test:
       >>> ann.Train()
       >>> ann.Validate()
       >>> ann.Test()

    4. Predict unseen rows:
       >>> yhat = ann.Predict(return_proba=True)

    Dataset format
    --------------
    Minimal (recommended):
      - feat : a list-like feature column (string list/JSON/comma-separated)
      - label: class id (int) for classification, float for regression

    Alternative:
      - provide explicit feature columns using `featuresKeys=[...]` (e.g. ["x","y","z"]).

    Optional split masks:
      - train_mask, val_mask, test_mask columns (0/1 or True/False)
      If masks exist and `use_masks_if_present=True`, they are used instead of ratio split.
    """

    def __init__(self):
        self.config = _ANNConfig()
        self.device = _pick_device(self.config.device)

        # Data (numpy)
        self.df: Optional[pd.DataFrame] = None
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.ids: Optional[np.ndarray] = None

        # Split indices
        self.idx_train: Optional[np.ndarray] = None
        self.idx_val: Optional[np.ndarray] = None
        self.idx_test: Optional[np.ndarray] = None

        # Model + training state
        self.model: Optional[nn.Module] = None
        self.num_features: Optional[int] = None
        self.num_classes: Optional[int] = None  # classification only

        self.history: Dict[str, List[float]] = {}
        self._trained: bool = False

        # Label mapping (classification)
        self.class_to_index: Optional[Dict[Any, int]] = None
        self.index_to_class: Optional[Dict[int, Any]] = None

    # -------------------------------------------------------------------------
    # Constructors
    # -------------------------------------------------------------------------

    @staticmethod
    def ByCSVPath(path: str,
                 task: str = "classification",
                 labelHeader: str = "label",
                 featHeader: str = "feat",
                 idHeader: str = "id",
                 featuresKeys: Optional[List[str]] = None,
                 useMasksIfPresent: bool = True,
                 split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                 device: str = "auto",
                 randomState: int = 42,
                 shuffle: bool = True):
        """
        Create an ANN instance by reading a dataset from a CSV file path.

        Parameters
        ----------
        path : str
            Path to a CSV file.
        task : str, optional
            "classification" or "regression". Default is "classification".
        labelHeader : str, optional
            Label column name. Default is "label".
        featHeader : str, optional
            Feature column name (list-like string). Default is "feat".
        idHeader : str, optional
            Optional ID column name. Default is "id".
        featuresKeys : list, optional
            If provided, features will be built from these explicit columns
            instead of parsing `featHeader`.
        useMasksIfPresent : bool, optional
            If True and mask columns exist, use them. Default is True.
        split : tuple(float,float,float), optional
            Train/val/test ratio used when masks are not provided. Default is (0.8,0.1,0.1).
        device : str, optional
            "auto", "cpu", or "cuda". Default is "auto".
        randomState : int, optional
            RNG seed for splitting and training determinism. Default is 42.
        shuffle : bool, optional
            Whether to shuffle prior to ratio split. Default is True.

        Returns
        -------
        ANN
            A configured ANN instance with data loaded and splits created.
        """
        ann = ANN()
        ann.config.task = task
        ann.config.label_header = labelHeader
        ann.config.feat_header = featHeader
        ann.config.id_header = idHeader
        ann.config.features_keys = featuresKeys
        ann.config.use_masks_if_present = useMasksIfPresent
        ann.config.split = split
        ann.config.device = device
        ann.config.random_state = randomState
        ann.config.shuffle = shuffle

        ann.device = _pick_device(device)
        _seed_everything(randomState)

        ann._load_csv(path)
        ann._prepare_xy()
        ann._make_splits()
        ann._build_model()

        return ann

    # -------------------------------------------------------------------------
    # Core helpers
    # -------------------------------------------------------------------------

    def _load_csv(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"ANN.ByCSVPath - Error: CSV file not found: {path}")
        self.df = pd.read_csv(path)

    def _prepare_xy(self) -> None:
        if self.df is None:
            raise RuntimeError("ANN - Error: No dataframe loaded.")

        df = self.df
        cfg = self.config

        # IDs
        self.ids = df[cfg.id_header].to_numpy() if cfg.id_header in df.columns else None

        # Features
        if cfg.features_keys is not None and len(cfg.features_keys) > 0:
            missing = [k for k in cfg.features_keys if k not in df.columns]
            if missing:
                raise ValueError(f"ANN - Error: Missing feature columns: {missing}")
            X = df[cfg.features_keys].to_numpy(dtype=float)
        else:
            if cfg.feat_header not in df.columns:
                raise ValueError(f"ANN - Error: Feature column '{cfg.feat_header}' not found.")
            feats = []
            for i, cell in enumerate(df[cfg.feat_header].values):
                v = _parse_feat_cell(cell)
                if v is None:
                    raise ValueError(f"ANN - Error: Could not parse features at row {i}.")
                feats.append(v)
            # enforce rectangular
            d0 = len(feats[0])
            for i, v in enumerate(feats):
                if len(v) != d0:
                    raise ValueError(f"ANN - Error: Non-rectangular features at row {i}: expected {d0}, got {len(v)}")
            X = np.asarray(feats, dtype=float)

        self.X = X
        self.num_features = int(X.shape[1])

        # Labels
        if cfg.label_header in df.columns:
            y_raw = df[cfg.label_header].values
        else:
            y_raw = None

        if y_raw is None:
            self.y = None
            self.num_classes = None
            return

        if cfg.task == "classification":
            # Create a stable mapping for possibly-string labels
            uniq = pd.unique(y_raw)
            uniq_sorted = sorted(list(uniq), key=lambda z: str(z))
            self.class_to_index = {c: i for i, c in enumerate(uniq_sorted)}
            self.index_to_class = {i: c for c, i in self.class_to_index.items()}
            y = np.asarray([self.class_to_index[v] for v in y_raw], dtype=np.int64)
            self.num_classes = int(len(uniq_sorted))
            self.y = y
        elif cfg.task == "regression":
            y = np.asarray(y_raw, dtype=float)
            self.num_classes = None
            self.y = y
        else:
            raise ValueError(f"ANN - Error: Unknown task '{cfg.task}'.")

    def _make_splits(self) -> None:
        if self.X is None:
            raise RuntimeError("ANN - Error: No features loaded.")

        n = int(self.X.shape[0])
        df = self.df
        cfg = self.config

        # Mask-based split (if present)
        if df is not None and cfg.use_masks_if_present:
            mask_cols = {"train_mask", "val_mask", "test_mask"}
            if mask_cols.issubset(set(df.columns)):
                tr = df["train_mask"].astype(bool).to_numpy()
                va = df["val_mask"].astype(bool).to_numpy()
                te = df["test_mask"].astype(bool).to_numpy()
                self.idx_train = np.where(tr)[0]
                self.idx_val = np.where(va)[0]
                self.idx_test = np.where(te)[0]
                return

        # Ratio split
        idx = np.arange(n)
        if cfg.shuffle:
            rng = np.random.default_rng(cfg.random_state)
            rng.shuffle(idx)

        a, b, c = cfg.split
        if not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0 and 0.0 <= c <= 1.0):
            raise ValueError("ANN - Error: split ratios must be in [0,1].")
        s = a + b + c
        if abs(s - 1.0) > 1e-6:
            # normalize if user passed, e.g., (80,10,10)
            a, b, c = a / s, b / s, c / s

        n_train = int(round(a * n))
        n_val = int(round(b * n))
        n_test = n - n_train - n_val

        self.idx_train = idx[:n_train]
        self.idx_val = idx[n_train:n_train + n_val]
        self.idx_test = idx[n_train + n_val:n_train + n_val + n_test]

    def _build_model(self) -> None:
        if self.num_features is None:
            raise RuntimeError("ANN - Error: num_features is unknown. Load data first.")

        cfg = self.config
        if cfg.task == "classification":
            if self.num_classes is None:
                raise RuntimeError("ANN - Error: num_classes is unknown for classification.")
            out_dim = int(self.num_classes)
        else:
            out_dim = 1

        self.model = _MLP(
            in_dim=int(self.num_features),
            hidden_dims=tuple(cfg.hidden_dims),
            out_dim=out_dim,
            activation=cfg.activation,
            dropout=float(cfg.dropout),
            batch_norm=bool(cfg.batch_norm)
        ).to(self.device)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def Summary(self) -> str:
        """
        Return a concise string summary of the ANN configuration and dataset shape.

        Returns
        -------
        str
            Human-readable summary.
        """
        n = int(self.X.shape[0]) if self.X is not None else 0
        d = int(self.num_features) if self.num_features is not None else 0
        s = [
            f"ANN(task={self.config.task}, device={self.device.type})",
            f"rows={n}, features={d}",
            f"split={self.config.split}, shuffle={self.config.shuffle}, seed={self.config.random_state}",
            f"hidden_dims={self.config.hidden_dims}, act={self.config.activation}, dropout={self.config.dropout}, bn={self.config.batch_norm}",
        ]
        if self.config.task == "classification":
            s.append(f"classes={self.num_classes}")
        return "\n".join(s)

    def SetHyperparameters(self, **kwargs) -> None:
        """
        Update training/model hyperparameters.

        Parameters
        ----------
        **kwargs : dict
            Any attribute of the internal config. Common keys include:
            epochs, batch_size, lr, weight_decay, optimizer, hidden_dims,
            activation, dropout, batch_norm, early_stopping, early_stopping_patience,
            gradient_clip_norm, split, shuffle, random_state, device.

        Returns
        -------
        None

        Notes
        -----
        If a model-shaping parameter changes (e.g. hidden_dims, activation, batch_norm),
        the model is rebuilt automatically.
        """
        rebuild_keys = {"hidden_dims", "activation", "dropout", "batch_norm", "task"}
        split_keys = {"split", "shuffle", "random_state", "use_masks_if_present"}
        device_keys = {"device"}

        do_rebuild = False
        do_resplit = False
        do_device = False

        for k, v in kwargs.items():
            if not hasattr(self.config, k):
                raise ValueError(f"ANN.SetHyperparameters - Error: Unknown config key '{k}'.")
            setattr(self.config, k, v)
            if k in rebuild_keys:
                do_rebuild = True
            if k in split_keys:
                do_resplit = True
            if k in device_keys:
                do_device = True

        if do_device:
            self.device = _pick_device(self.config.device)
            if self.model is not None:
                self.model.to(self.device)

        if do_resplit:
            _seed_everything(self.config.random_state)
            self._make_splits()

        if do_rebuild:
            self._build_model()

    def SaveModel(self, path: str, include_config: bool = True) -> None:
        """
        Save model weights (and optionally config) to disk.

        Parameters
        ----------
        path : str
            Output file path. If it does not end with ``.pt``, it is appended.
        include_config : bool, optional
            If True, saves a checkpoint dict that includes config fields to support
            robust reload. Default is True.

        Returns
        -------
        None
        """
        if self.model is None:
            raise RuntimeError("ANN.SaveModel - Error: No model to save.")
        if not path.lower().endswith(".pt"):
            path += ".pt"

        if include_config:
            payload = {
                "state_dict": self.model.state_dict(),
                "config_fields": asdict(self.config),
                "num_features": int(self.num_features) if self.num_features is not None else None,
                "num_classes": int(self.num_classes) if self.num_classes is not None else None,
                "class_to_index": self.class_to_index,
                "index_to_class": self.index_to_class,
            }
            torch.save(payload, path)
        else:
            torch.save(self.model.state_dict(), path)

    def LoadModel(self, path: str, strict: bool = True, rebuild_from_checkpoint: bool = True) -> None:
        """
        Load model weights from disk (backward compatible).

        Parameters
        ----------
        path : str
            Path to a ``.pt`` checkpoint file.
        strict : bool, optional
            Passed to ``load_state_dict``. Default is True.
        rebuild_from_checkpoint : bool, optional
            If True and the checkpoint includes config fields, rebuild the model
            before loading. Default is True.

        Returns
        -------
        None
        """
        # Silence future warning by being explicit
        try:
            obj = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            obj = torch.load(path, map_location=self.device)

        if isinstance(obj, dict) and "state_dict" in obj:
            if rebuild_from_checkpoint and "config_fields" in obj and isinstance(obj["config_fields"], dict):
                # restore config
                for k, v in obj["config_fields"].items():
                    if hasattr(self.config, k):
                        setattr(self.config, k, v)

                self.device = _pick_device(self.config.device)
                self.num_features = obj.get("num_features", self.num_features)
                self.num_classes = obj.get("num_classes", self.num_classes)
                self.class_to_index = obj.get("class_to_index", self.class_to_index)
                self.index_to_class = obj.get("index_to_class", self.index_to_class)

                self._build_model()

            state = obj["state_dict"]
        else:
            # raw state_dict
            state = obj
            if self.model is None:
                self._build_model()

        self.model.load_state_dict(state, strict=strict)
        self.model.to(self.device)
        self.model.eval()

    # -------------------------------------------------------------------------
    # Training / Evaluation
    # -------------------------------------------------------------------------

    def _make_loader(self, idx: np.ndarray, shuffle: bool = False) -> DataLoader:
        X = self.X[idx]
        y = None if self.y is None else self.y[idx]
        ds = _TabularDataset(X, y)
        return DataLoader(ds, batch_size=int(self.config.batch_size), shuffle=bool(shuffle))

    def _loss_fn(self) -> nn.Module:
        if self.config.task == "classification":
            return nn.CrossEntropyLoss()
        return nn.MSELoss()

    def _optimizer(self) -> torch.optim.Optimizer:
        if self.model is None:
            raise RuntimeError("ANN - Error: No model built.")
        opt = (self.config.optimizer or "adamw").lower()
        if opt == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        if opt == "adamw":
            return torch.optim.AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        if opt == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay, momentum=0.9)
        raise ValueError(f"ANN - Error: Unknown optimizer '{self.config.optimizer}'.")

    def Train(self) -> Dict[str, List[float]]:
        """
        Train the ANN model.

        Returns
        -------
        dict
            Training history dictionary with per-epoch curves. Typical keys:
            "train_loss", "val_loss" and task-specific metric keys.
        """
        if self.model is None:
            self._build_model()
        if self.idx_train is None or self.idx_val is None:
            self._make_splits()

        cfg = self.config
        loss_fn = self._loss_fn()
        opt = self._optimizer()

        train_loader = self._make_loader(self.idx_train, shuffle=True)
        val_loader = self._make_loader(self.idx_val, shuffle=False)

        self.history = {"train_loss": [], "val_loss": []}
        best_val = float("inf")
        best_state = None
        patience = 0

        for epoch in range(int(cfg.epochs)):
            self.model.train()
            train_losses = []
            for batch in train_loader:
                if cfg.task == "classification":
                    x, y = batch
                    y = torch.as_tensor(y, dtype=torch.long)
                else:
                    x, y = batch
                    y = torch.as_tensor(y, dtype=torch.float32)

                x = x.to(self.device)
                y = y.to(self.device)

                opt.zero_grad()
                out = self.model(x)
                if cfg.task == "regression":
                    out = out.squeeze(-1)

                loss = loss_fn(out, y)
                loss.backward()

                if cfg.gradient_clip_norm is not None and cfg.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(cfg.gradient_clip_norm))

                opt.step()
                train_losses.append(float(loss.detach().cpu().item()))

            train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

            # validation
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    if cfg.task == "classification":
                        x, y = batch
                        y = torch.as_tensor(y, dtype=torch.long)
                    else:
                        x, y = batch
                        y = torch.as_tensor(y, dtype=torch.float32)

                    x = x.to(self.device)
                    y = y.to(self.device)

                    out = self.model(x)
                    if cfg.task == "regression":
                        out = out.squeeze(-1)

                    loss = loss_fn(out, y)
                    val_losses.append(float(loss.detach().cpu().item()))

            val_loss = float(np.mean(val_losses)) if val_losses else float("nan")

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if cfg.verbose:
                print(f"Epoch {epoch+1:03d}/{cfg.epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

            # early stopping
            if cfg.early_stopping:
                if val_loss < best_val - 1e-12:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
                    if patience >= int(cfg.early_stopping_patience):
                        if cfg.verbose:
                            print("Early stopping triggered.")
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state, strict=True)

        self.model.eval()
        self._trained = True
        return self.history

    def _evaluate_split(self, idx: np.ndarray) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("ANN - Error: No model built.")
        if self.y is None:
            raise RuntimeError("ANN - Error: No labels available for evaluation.")

        cfg = self.config
        loader = self._make_loader(idx, shuffle=False)

        y_true = []
        y_pred = []
        y_prob = []

        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                x, y = batch
                x = x.to(self.device)

                out = self.model(x)

                if cfg.task == "classification":
                    probs = torch.softmax(out, dim=1)
                    pred = torch.argmax(probs, dim=1).cpu().numpy()
                    y_prob.append(probs.cpu().numpy())
                    y_pred.append(pred)
                    y_true.append(np.asarray(y, dtype=np.int64))
                else:
                    pred = out.squeeze(-1).cpu().numpy().astype(float)
                    y_pred.append(pred)
                    y_true.append(np.asarray(y, dtype=float))

        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)

        metrics: Dict[str, Any] = {}

        if cfg.task == "classification":
            if accuracy_score is None:
                metrics["accuracy"] = float("nan")
            else:
                metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
                metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
                metrics["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
                metrics["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
            metrics["y_true"] = y_true
            metrics["y_pred"] = y_pred
            metrics["y_prob"] = np.concatenate(y_prob, axis=0) if y_prob else None
        else:
            if mean_absolute_error is None:
                metrics["mae"] = float("nan")
                metrics["rmse"] = float("nan")
                metrics["r2"] = float("nan")
            else:
                metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
                metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                metrics["r2"] = float(r2_score(y_true, y_pred))
            metrics["y_true"] = y_true
            metrics["y_pred"] = y_pred

        return metrics

    def Validate(self) -> Dict[str, Any]:
        """
        Evaluate the model on the validation split.

        Returns
        -------
        dict
            Metrics dictionary. For classification includes accuracy/f1_macro/etc.
            For regression includes mae/rmse/r2. Always includes y_true and y_pred.
        """
        if self.idx_val is None:
            raise RuntimeError("ANN.Validate - Error: No validation split.")
        return self._evaluate_split(self.idx_val)

    def Test(self) -> Dict[str, Any]:
        """
        Evaluate the model on the test split.

        Returns
        -------
        dict
            Metrics dictionary. For classification includes accuracy/f1_macro/etc.
            For regression includes mae/rmse/r2. Always includes y_true and y_pred.
        """
        if self.idx_test is None:
            raise RuntimeError("ANN.Test - Error: No test split.")
        return self._evaluate_split(self.idx_test)

    def Predict(self,
                path: Optional[str] = None,
                return_proba: bool = False,
                return_logits: bool = False,
                attach_to_df: bool = True) -> Dict[str, Any]:
        """
        Predict outputs for the loaded dataset or a new CSV dataset.

        Parameters
        ----------
        path : str, optional
            If provided, load a new CSV file and run inference on it. If None,
            predicts on the currently loaded dataset.
        return_proba : bool, optional
            If True and task is classification, return class probabilities.
            Default is False.
        return_logits : bool, optional
            If True and task is classification, return raw logits.
            Default is False.
        attach_to_df : bool, optional
            If True, attach predictions back to ``self.df`` (or newly loaded df)
            as columns: "pred" and optionally "proba_*". Default is True.

        Returns
        -------
        dict
            Prediction package with keys:
            - "pred" : numpy array of predictions (class indices or regression values)
            - "proba" : numpy array (N,C) if requested for classification
            - "logits": numpy array (N,C) if requested for classification
            - "df" : pandas DataFrame if attach_to_df=True
        """
        if self.model is None:
            raise RuntimeError("ANN.Predict - Error: No model. Train or load a model first.")

        # If predicting new CSV, we reuse schema (features_keys / feat_header)
        if path is not None:
            df = pd.read_csv(path)
            cfg = self.config
            if cfg.features_keys is not None and len(cfg.features_keys) > 0:
                missing = [k for k in cfg.features_keys if k not in df.columns]
                if missing:
                    raise ValueError(f"ANN.Predict - Error: Missing feature columns in new CSV: {missing}")
                X = df[cfg.features_keys].to_numpy(dtype=float)
            else:
                if cfg.feat_header not in df.columns:
                    raise ValueError(f"ANN.Predict - Error: Feature column '{cfg.feat_header}' not found in new CSV.")
                feats = []
                for i, cell in enumerate(df[cfg.feat_header].values):
                    v = _parse_feat_cell(cell)
                    if v is None:
                        raise ValueError(f"ANN.Predict - Error: Could not parse features at row {i} in new CSV.")
                    feats.append(v)
                X = np.asarray(feats, dtype=float)

            work_df = df
        else:
            if self.X is None or self.df is None:
                raise RuntimeError("ANN.Predict - Error: No dataset loaded.")
            X = self.X
            work_df = self.df

        ds = _TabularDataset(X, y=None)
        loader = DataLoader(ds, batch_size=int(self.config.batch_size), shuffle=False)

        preds = []
        probas = []
        logits_all = []

        self.model.eval()
        with torch.no_grad():
            for x in loader:
                x = x.to(self.device)
                out = self.model(x)

                if self.config.task == "classification":
                    logits = out
                    probs = torch.softmax(logits, dim=1)
                    pred = torch.argmax(probs, dim=1)
                    preds.append(pred.cpu().numpy())
                    if return_proba:
                        probas.append(probs.cpu().numpy())
                    if return_logits:
                        logits_all.append(logits.cpu().numpy())
                else:
                    pred = out.squeeze(-1)
                    preds.append(pred.cpu().numpy().astype(float))

        pred = np.concatenate(preds, axis=0)
        out_pkg: Dict[str, Any] = {"pred": pred}

        if self.config.task == "classification":
            # decode to original labels if mapping exists
            if self.index_to_class is not None:
                decoded = np.asarray([self.index_to_class[int(i)] for i in pred], dtype=object)
                out_pkg["pred_label"] = decoded
                if attach_to_df:
                    work_df["pred_label"] = decoded
            if attach_to_df:
                work_df["pred"] = pred
            if return_proba:
                proba = np.concatenate(probas, axis=0) if probas else None
                out_pkg["proba"] = proba
                if attach_to_df and proba is not None:
                    for c in range(proba.shape[1]):
                        work_df[f"proba_{c}"] = proba[:, c]
            if return_logits:
                logits = np.concatenate(logits_all, axis=0) if logits_all else None
                out_pkg["logits"] = logits
        else:
            if attach_to_df:
                work_df["pred"] = pred

        if attach_to_df:
            out_pkg["df"] = work_df

        return out_pkg

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------

    def PlotHistory(self, title: str = "Learning Curves"):
        """
        Plot training history curves (losses) using Plotly.

        Parameters
        ----------
        title : str, optional
            Plot title. Default is "Learning Curves".

        Returns
        -------
        plotly.graph_objects.Figure
            A Plotly figure.

        Notes
        -----
        Requires Plotly.
        """
        if go is None:
            raise ImportError("Plotly is required. Install plotly to use PlotHistory.")
        if not self.history:
            raise RuntimeError("ANN.PlotHistory - Error: No history found. Train first.")

        fig = go.Figure()
        if "train_loss" in self.history:
            fig.add_trace(go.Scatter(y=self.history["train_loss"], mode="lines", name="train_loss"))
        if "val_loss" in self.history:
            fig.add_trace(go.Scatter(y=self.history["val_loss"], mode="lines", name="val_loss"))
        fig.update_layout(title=title, xaxis_title="Epoch", yaxis_title="Loss")
        return fig

    def PlotConfusionMatrix(self,
                            split: str = "test",
                            normalize: bool = False,
                            title: Optional[str] = None):
        """
        Plot a confusion matrix for classification.

        Parameters
        ----------
        split : {"train","val","validate","validation","test","all"}, optional
            Which split(s) to evaluate. Default is "test".
        normalize : bool, optional
            If True, row-normalize the confusion matrix. Default is False.
        title : str, optional
            Custom title. If None, uses an automatic title.

        Returns
        -------
        plotly.graph_objects.Figure
            Confusion matrix heatmap (Plotly).

        Notes
        -----
        Requires scikit-learn and plotly.
        """
        if self.config.task != "classification":
            raise ValueError("ANN.PlotConfusionMatrix - Error: Only valid for classification.")
        if px is None or confusion_matrix is None:
            raise ImportError("Plotly and scikit-learn are required for confusion matrices.")

        s = (split or "test").lower()
        if s in ("validate", "validation"):
            s = "val"

        if s == "train":
            m = self._evaluate_split(self.idx_train)
        elif s == "val":
            m = self._evaluate_split(self.idx_val)
        elif s == "test":
            m = self._evaluate_split(self.idx_test)
        elif s == "all":
            yt = []
            yp = []
            for idx in (self.idx_train, self.idx_val, self.idx_test):
                mm = self._evaluate_split(idx)
                yt.append(mm["y_true"])
                yp.append(mm["y_pred"])
            y_true = np.concatenate(yt, axis=0)
            y_pred = np.concatenate(yp, axis=0)
            m = {"y_true": y_true, "y_pred": y_pred}
        else:
            raise ValueError("ANN.PlotConfusionMatrix - Error: split must be train/val/test/all.")

        y_true = m["y_true"]
        y_pred = m["y_pred"]

        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        if normalize:
            cm = cm.astype(float)
            cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)

        if title is None:
            title = f"Confusion Matrix ({s})"

        fig = px.imshow(cm, x=labels, y=labels, text_auto=True, aspect="auto")
        fig.update_layout(title=title, xaxis_title="Predicted", yaxis_title="True")
        # Force all tick labels to show
        fig.update_xaxes(tickmode="array", tickvals=list(range(len(labels))), ticktext=[str(x) for x in labels])
        fig.update_yaxes(tickmode="array", tickvals=list(range(len(labels))), ticktext=[str(x) for x in labels])
        return fig

    def PlotParity(self,
                   split: str = "test",
                   title: Optional[str] = None,
                   show_identity: bool = True,
                   show_best_fit: bool = True,
                   point_size: int = 6):
        """
        Plot a parity plot (true vs predicted) for regression.

        Parameters
        ----------
        split : {"train","val","validate","validation","test","all"}, optional
            Which split(s) to evaluate. Default is "test".
        title : str, optional
            Custom title. If None, uses an automatic title.
        show_identity : bool, optional
            If True, plot y=x. Default is True.
        show_best_fit : bool, optional
            If True, plot least-squares fit line. Default is True.
        point_size : int, optional
            Marker size. Default is 6.

        Returns
        -------
        plotly.graph_objects.Figure
            Parity scatter plot.

        Notes
        -----
        Requires plotly.
        """
        if self.config.task != "regression":
            raise ValueError("ANN.PlotParity - Error: Only valid for regression.")
        if go is None:
            raise ImportError("Plotly is required for PlotParity.")

        s = (split or "test").lower()
        if s in ("validate", "validation"):
            s = "val"

        if s == "train":
            m = self._evaluate_split(self.idx_train)
        elif s == "val":
            m = self._evaluate_split(self.idx_val)
        elif s == "test":
            m = self._evaluate_split(self.idx_test)
        elif s == "all":
            yt = []
            yp = []
            for idx in (self.idx_train, self.idx_val, self.idx_test):
                mm = self._evaluate_split(idx)
                yt.append(mm["y_true"])
                yp.append(mm["y_pred"])
            m = {"y_true": np.concatenate(yt, axis=0), "y_pred": np.concatenate(yp, axis=0)}
        else:
            raise ValueError("ANN.PlotParity - Error: split must be train/val/test/all.")

        y_true = np.asarray(m["y_true"], dtype=float)
        y_pred = np.asarray(m["y_pred"], dtype=float)

        if title is None:
            title = f"Parity Plot ({s})"

        mae = float(np.mean(np.abs(y_pred - y_true)))
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        eps = 1e-12
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = 1.0 - ss_res / (ss_tot + eps)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_true, y=y_pred, mode="markers",
            name="Predictions", marker=dict(size=int(point_size))
        ))

        mn = float(min(np.min(y_true), np.min(y_pred)))
        mx = float(max(np.max(y_true), np.max(y_pred)))

        if show_identity:
            fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines", name="Ideal (y=x)", hoverinfo="skip"))

        if show_best_fit and len(y_true) >= 2:
            a, b = np.polyfit(y_true, y_pred, 1)
            fig.add_trace(go.Scatter(
                x=[mn, mx], y=[a * mn + b, a * mx + b],
                mode="lines", name=f"Best fit (y={a:.3g}x+{b:.3g})", hoverinfo="skip"
            ))

        fig.update_layout(
            title=f"{title} — MAE={mae:.4g}, RMSE={rmse:.4g}, R²={r2:.4g}",
            xaxis_title="True",
            yaxis_title="Predicted"
        )
        return fig
