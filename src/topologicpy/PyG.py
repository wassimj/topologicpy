# Copyright (C) 2026
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

"""
TopologicPy: PyTorch Geometric (PyG) helper class
=================================================

This module provides a clean, beginner-friendly interface to:

1) Load TopologicPy-exported CSV datasets (graphs.csv, nodes.csv, edges.csv)
2) Train / validate / test models for:
   - Graph-level prediction (classification or regression)
   - Node-level prediction (classification or regression)
   - Edge-level prediction (classification or regression)
   - Link prediction (binary edge existence)

3) Report performance metrics and interactive Plotly visualisations.

User-controlled hyperparameters (medium-level)
----------------------------------------------
- Cross-validation: holdout or k-fold (graph-level)
- Network topology: number of hidden layers and neurons per layer (hidden_dims)
- GNN backbone: conv type (sage/gcn/gatv2), activation, dropout, batch_norm, residual
- Training: epochs, batch_size, lr, weight_decay, optimizer (adam/adamw),
           gradient clipping, early stopping

CSV assumptions
---------------
- graphs.csv contains at least: graph_id, label, and optional graph feature columns:
    feat_0, feat_1, ...

- nodes.csv contains at least: graph_id, node_id, label, optional masks, and feature columns:
    feat_0, feat_1, ...

- edges.csv contains at least: graph_id, src_id, dst_id, label, optional masks, and feature columns:
    feat_0, feat_1, ...

Notes
-----
- This module intentionally avoids auto-install behaviour.
- It aims to be easy to read and modify by non-ML experts.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Literal

import os
import math
import random
import copy

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import SAGEConv, GCNConv, GATv2Conv, global_mean_pool, global_max_pool, global_add_pool
    from torch_geometric.transforms import RandomLinkSplit
except Exception as _e:
    torch = nn = F = Data = DataLoader = SAGEConv = GCNConv = GATv2Conv = None
    global_mean_pool = global_max_pool = global_add_pool = None
    RandomLinkSplit = None
    _PYG_IMPORT_ERROR = _e
else:
    _PYG_IMPORT_ERROR = None


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader
# from torch_geometric.nn import (
#     SAGEConv, GCNConv, GATv2Conv,
#     global_mean_pool, global_max_pool, global_add_pool
# )
# from torch_geometric.transforms import RandomLinkSplit

try:
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support, confusion_matrix,
        mean_absolute_error, mean_squared_error, r2_score
    )
except Exception as _e:
    accuracy_score = precision_recall_fscore_support = confusion_matrix = None
    mean_absolute_error = mean_squared_error = r2_score = None
    _SKLEARN_IMPORT_ERROR = _e
else:
    _SKLEARN_IMPORT_ERROR = None

try:
    import plotly.graph_objects as go
    import plotly.express as px
except Exception as _e:
    go = px = None
    _PLOTLY_IMPORT_ERROR = _e
else:
    _PLOTLY_IMPORT_ERROR = None


LabelType = Literal["categorical", "continuous"]
Level = Literal["graph", "node", "edge", "link"]
TaskKind = Literal["classification", "regression", "link_prediction"]
ConvKind = Literal["sage", "gcn", "gatv2"]
PoolingKind = Literal["mean", "max", "sum"]


@dataclass
class _RunConfig:

    verbose : bool = False

    # ----------------------------
    # Task selection
    # ----------------------------
    level: Level = "graph"            # "graph" | "node" | "edge" | "link"
    task: TaskKind = "classification" # "classification" | "regression" | "link_prediction"

    # label types (graph/node/edge)
    graph_label_type: LabelType = "categorical"
    node_label_type: LabelType = "categorical"
    edge_label_type: LabelType = "categorical"

    # ----------------------------
    # CSV headers
    # ----------------------------
    graph_id_header: str = "graph_id"
    graph_label_header: str = "label"
    graph_features_header: str = "feat"

    node_id_header: str = "node_id"
    node_label_header: str = "label"
    node_features_header: str = "feat"

    edge_src_header: str = "src_id"
    edge_dst_header: str = "dst_id"
    edge_label_header: str = "label"
    edge_features_header: str = "feat"

    # masks (optional)
    node_train_mask_header: str = "train_mask"
    node_val_mask_header: str = "val_mask"
    node_test_mask_header: str = "test_mask"

    edge_train_mask_header: str = "train_mask"
    edge_val_mask_header: str = "val_mask"
    edge_test_mask_header: str = "test_mask"

    # ----------------------------
    # Cross-validation / splitting
    # ----------------------------
    cv: Literal["holdout", "kfold"] = "holdout"
    split: Tuple[float, float, float] = (0.8, 0.1, 0.1)  # used for holdout
    k_folds: int = 5                                    # used for kfold (graph-level only)
    k_shuffle: bool = True
    k_stratify: bool = True                             # only if categorical labels exist
    random_state: int = 42
    shuffle: bool = True                                # affects holdout + in-graph mask fallback

    # link prediction split (within each graph)
    link_val_ratio: float = 0.1
    link_test_ratio: float = 0.1
    link_is_undirected: bool = False

    # ----------------------------
    # Training hyperparameters
    # ----------------------------
    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: Literal["adam", "adamw"] = "adam"
    gradient_clip_norm: Optional[float] = None
    early_stopping: bool = False
    early_stopping_patience: int = 10
    use_gpu: bool = True

    # ----------------------------
    # Network topology / model hyperparameters
    # ----------------------------
    conv: ConvKind = "sage"
    hidden_dims: Tuple[int, ...] = (64, 64)  # explicit per-layer widths (controls depth)
    activation: Literal["relu", "gelu", "elu"] = "relu"
    dropout: float = 0.1
    batch_norm: bool = False
    residual: bool = False
    pooling: PoolingKind = "mean"   # only for graph-level


# -----------------------------------------------------------------------------
# Optional dependency guard
# -----------------------------------------------------------------------------
# Sphinx/autodoc imports this module when building documentation. In environments
# where PyTorch / PyTorch-Geometric are not installed, importing must still
# succeed so the documentation page is not blank.

if _PYG_IMPORT_ERROR is None:
    class _GNNBackbone(nn.Module):
        """
        Shared GNN encoder that produces node embeddings.
        """

        def __init__(self,
                     in_dim: int,
                     hidden_dims: Tuple[int, ...],
                     conv: ConvKind = "sage",
                     activation: str = "relu",
                     dropout: float = 0.1,
                     batch_norm: bool = False,
                     residual: bool = False):
            super().__init__()
            if in_dim <= 0:
                raise ValueError("in_dim must be > 0. Your dataset has no node features columns.")
            if hidden_dims is None or len(hidden_dims) == 0:
                raise ValueError("hidden_dims must contain at least one layer width, e.g. (64, 64).")

            self.dropout = float(dropout)
            self.use_bn = bool(batch_norm)
            self.use_residual = bool(residual)

            if activation == "relu":
                self.act = F.relu
            elif activation == "gelu":
                self.act = F.gelu
            elif activation == "elu":
                self.act = F.elu
            else:
                raise ValueError("Unsupported activation. Use 'relu', 'gelu', or 'elu'.")

            dims = [int(in_dim)] + [int(d) for d in hidden_dims]

            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()

            for i in range(1, len(dims)):
                in_ch, out_ch = dims[i - 1], dims[i]

                if conv == "sage":
                    self.convs.append(SAGEConv(in_ch, out_ch))
                elif conv == "gcn":
                    self.convs.append(GCNConv(in_ch, out_ch))
                elif conv == "gatv2":
                    self.convs.append(GATv2Conv(in_ch, out_ch, heads=1, concat=False))
                else:
                    raise ValueError(f"Unsupported conv='{conv}'.")

                if self.use_bn:
                    self.bns.append(nn.BatchNorm1d(out_ch))

            self.out_dim = dims[-1]

        def forward(self, x, edge_index):
            h = x
            for i, conv in enumerate(self.convs):
                h_in = h
                h = conv(h, edge_index)
                if self.use_bn:
                    h = self.bns[i](h)
                h = self.act(h)

                if self.use_residual and h_in.shape == h.shape:
                    h = h + h_in

                h = F.dropout(h, p=self.dropout, training=self.training)
            return h


    class _GraphHead(nn.Module):
        def __init__(self, in_dim: int, out_dim: int, pooling: PoolingKind = "mean", dropout: float = 0.1):
            super().__init__()
            self.dropout = float(dropout)

            if pooling == "mean":
                self.pool = global_mean_pool
            elif pooling == "max":
                self.pool = global_max_pool
            elif pooling == "sum":
                self.pool = global_add_pool
            else:
                raise ValueError("GraphHead requires pooling in {'mean','max','sum'}.")

            self.mlp = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(in_dim, out_dim),
            )

        def forward(self, node_emb, batch):
            g = self.pool(node_emb, batch)
            return self.mlp(g)


    class _NodeHead(nn.Module):
        def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(in_dim, out_dim),
            )

        def forward(self, node_emb):
            return self.mlp(node_emb)


    class _EdgeHead(nn.Module):
        """
        Edge prediction head using concatenation of endpoint embeddings.
        """
        def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(in_dim * 2, in_dim),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(in_dim, out_dim),
            )

        def forward(self, node_emb, edge_index):
            src, dst = edge_index[0], edge_index[1]
            h = torch.cat([node_emb[src], node_emb[dst]], dim=-1)
            return self.mlp(h)


    class _LinkPredictor(nn.Module):
        """
        Binary link predictor (edge exists or not).
        """
        def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim * 2, hidden),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden, 1),
            )

        def forward(self, node_emb, edge_label_index):
            src, dst = edge_label_index[0], edge_label_index[1]
            h = torch.cat([node_emb[src], node_emb[dst]], dim=-1)
            return self.net(h).squeeze(-1)  # logits


    class PyG:
        """
        A clean PyTorch Geometric interface for TopologicPy-exported CSV datasets.

        You can control medium-level hyperparameters by passing keyword arguments to ByCSVPath,
        for example:

        pyg = PyG.ByCSVPath(
            path="C:/dataset",
            level="graph",
            task="classification",
            graphLabelType="categorical",
            cv="kfold",
            k_folds=5,
            conv="gatv2",
            hidden_dims=(128, 128, 64),
            activation="gelu",
            batch_norm=True,
            residual=True,
            dropout=0.2,
            lr=1e-3,
            optimizer="adamw",
            early_stopping=True,
            early_stopping_patience=10,
            gradient_clip_norm=1.0
        )
        """

        # ---------
        # Creation
        # ---------
        @staticmethod
        def ByCSVPath(path: str,
                      level: Level = "graph",
                      task: TaskKind = "classification",
                      graphLabelType: LabelType = "categorical",
                      nodeLabelType: LabelType = "categorical",
                      edgeLabelType: LabelType = "categorical",
                      **kwargs) -> "PyG":
            """
            Creates a :class:`~topologicpy.PyG.PyG` instance from a TopologicPy-exported CSV dataset folder.

            The dataset folder is expected to contain **three** files:

            - ``graphs.csv`` : one row per graph (graph-level labels/features)
            - ``nodes.csv``  : one row per node (node-level labels/features/masks)
            - ``edges.csv``  : one row per edge (edge-level labels/features/masks)

            The created instance immediately loads the CSVs, builds a list of
            :class:`torch_geometric.data.Data` objects, performs an initial holdout split
            (for graph-level tasks), and builds a default model according to the provided
            configuration.

            Parameters
            ----------
            path : str
                Path to the dataset folder that contains ``graphs.csv``, ``nodes.csv``,
                and ``edges.csv``.
            level : {"graph", "node", "edge", "link"}, optional
                The prediction level:

                - ``"graph"``: graph-level labels in ``graphs.csv``
                - ``"node"`` : node-level labels in ``nodes.csv``
                - ``"edge"`` : edge-level labels in ``edges.csv``
                - ``"link"`` : link prediction (binary edge existence)
            task : {"classification", "regression", "link_prediction"}, optional
                The learning task. For ``level="link"`` this should be
                ``"link_prediction"``.
            graphLabelType : {"categorical", "continuous"}, optional
                Label type for graph-level targets (used when ``level="graph"``).
            nodeLabelType : {"categorical", "continuous"}, optional
                Label type for node-level targets (used when ``level="node"``).
            edgeLabelType : {"categorical", "continuous"}, optional
                Label type for edge-level targets (used when ``level="edge"``).
            **kwargs : dict
                Optional overrides for any field in :class:`~topologicpy.PyG._RunConfig`.
                Common examples include ``conv``, ``hidden_dims``, ``activation``,
                ``dropout``, ``batch_norm``, ``residual``, ``pooling``, ``epochs``,
                ``batch_size``, ``lr``, ``weight_decay``, and cross-validation options.

            Returns
            -------
            PyG
                The created :class:`~topologicpy.PyG.PyG` instance.

            Raises
            ------
            ValueError
                If the path does not exist, required CSV files are missing, or no node
                feature columns are found.

            Examples
            --------
            . pyg = PyG.ByCSVPath(path="C:/dataset", level="graph", task="classification")
            . history = pyg.Train(epochs=50)
            """
            if _PYG_IMPORT_ERROR is not None:
                raise ImportError(
                    "topologicpy.PyG requires optional dependencies. Install with the PyG extras "
                    "(e.g. torch + torch_geometric) and try again."
                ) from _PYG_IMPORT_ERROR
        
            cfg = _RunConfig(level=level, task=task,
                             graph_label_type=graphLabelType,
                             node_label_type=nodeLabelType,
                             edge_label_type=edgeLabelType)

            # allow override of any config field via kwargs
            for k, v in kwargs.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)

            return PyG(path=path, config=cfg)

        def __init__(self, path: str, config: _RunConfig):
            self.path = path
            self.config = config

            self.device = torch.device("cuda:0" if (config.use_gpu and torch.cuda.is_available()) else "cpu")

            self.graph_df: Optional[pd.DataFrame] = None
            self.nodes_df: Optional[pd.DataFrame] = None
            self.edges_df: Optional[pd.DataFrame] = None

            self.data_list: List[Data] = []
            self.train_set: Optional[List[Data]] = None
            self.val_set: Optional[List[Data]] = None
            self.test_set: Optional[List[Data]] = None

            self.model: Optional[nn.Module] = None
            self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
            self.cv_report: Optional[Dict[str, Union[float, List[Dict[str, float]]]]] = None

            self._num_outputs: int = 1

            # Saved training schema (used to keep feature dimensions consistent during inference)
            self._in_dim: Optional[int] = None
            self._feature_schema: Optional[Dict[str, List[str]]] = None
            self._freeze_num_outputs: bool = False

            self._load_csv()
            self._build_data_list()
            self._split_holdout()

            self._build_model()


        # ----------------------------
        # Convenience: hyperparameters
        # ----------------------------
        def SetHyperparameters(self, **kwargs) -> Dict[str, Union[str, int, float, bool, Tuple]]:
            """
            Set one or more configuration values (hyperparameters) on this instance.

            This method updates :attr:`~topologicpy.PyG.PyG.config` fields using keyword
            arguments. If any *model-shaping* setting changes (e.g. ``conv``, ``hidden_dims``,
            ``activation``, ``dropout``, ``batch_norm``, ``residual``, ``pooling``), the
            model is rebuilt automatically.

            Parameters
            ----------
            **kwargs : dict
                Key/value pairs matching fields in :class:`~topologicpy.PyG._RunConfig`.
                Unknown keys are ignored.

            Returns
            -------
            dict
                A compact configuration summary (same as :meth:`~topologicpy.PyG.PyG.Summary`).

            Raises
            ------
            ValueError
                If an attempted setting fails validation (e.g. malformed ``split`` or empty
                ``hidden_dims``).

            Notes
            -----
            - For graph-level tasks, changing ``split`` affects holdout splitting. You may
              want to call :meth:`~topologicpy.PyG.PyG.ByCSVPath` again (or re-instantiate)
              if you need a fresh split with new ratios.
            - For node/edge tasks, masks are taken from CSV columns if present; otherwise
              they are generated using ``split`` ratios within each graph.
            """
            cfg = self.config
            changed_model = False

            for k, v in kwargs.items():
                if not hasattr(cfg, k):
                    if cfg.verbose:
                        print(f"PyG.SetHyperparameters - Warning: Unknown parameter '{k}' ignored.")
                    continue

                # Basic validation / normalisation
                if k == "split":
                    if (not isinstance(v, (tuple, list))) or len(v) != 3:
                        raise ValueError("split must be a 3-tuple, e.g. (0.8, 0.1, 0.1).")
                    s = float(v[0]) + float(v[1]) + float(v[2])
                    if abs(s - 1.0) > 1e-3:
                        raise ValueError("split ratios must sum to 1.")
                    v = (float(v[0]), float(v[1]), float(v[2]))

                if k == "hidden_dims":
                    if isinstance(v, list):
                        v = tuple(v)
                    if (not isinstance(v, tuple)) or len(v) == 0:
                        raise ValueError("hidden_dims must be a non-empty tuple, e.g. (64, 64).")
                    v = tuple(int(x) for x in v)
                    changed_model = True

                if k in ["conv", "activation", "dropout", "batch_norm", "residual", "pooling"]:
                    changed_model = True

                setattr(cfg, k, v)

            # rebuild model if needed
            if changed_model:
                self._build_model()

            return self.Summary()

        def Summary(self) -> Dict[str, Union[str, int, float, bool, Tuple]]:
            """
            Return a compact summary of the current configuration and dataset size.

            Returns
            -------
            dict
                A dictionary containing key configuration choices such as ``level``, ``task``,
                network options (``conv``, ``hidden_dims``, etc.), training hyperparameters,
                current device, and basic dataset counts.

            Notes
            -----
            This is intended to be a lightweight, ReadTheDocs-friendly snapshot suitable for
            logging and reproducibility.
            """
            cfg = self.config
            return {
                "level": cfg.level,
                "task": cfg.task,
                "graph_label_type": cfg.graph_label_type,
                "node_label_type": cfg.node_label_type,
                "edge_label_type": cfg.edge_label_type,
                "cv": cfg.cv,
                "split": cfg.split,
                "k_folds": cfg.k_folds,
                "conv": cfg.conv,
                "hidden_dims": cfg.hidden_dims,
                "activation": cfg.activation,
                "dropout": cfg.dropout,
                "batch_norm": cfg.batch_norm,
                "residual": cfg.residual,
                "pooling": cfg.pooling,
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "optimizer": cfg.optimizer,
                "gradient_clip_norm": cfg.gradient_clip_norm,
                "early_stopping": cfg.early_stopping,
                "early_stopping_patience": cfg.early_stopping_patience,
                "device": str(self.device),
                "num_graphs": len(self.data_list),
                "num_outputs": int(self._num_outputs),
            }

        # ----------------------------
        # Convenience: CV visualisation
        # ----------------------------
        def PlotCrossValidationSummary(self,
                                      cv_report: Optional[Dict[str, Union[float, List[Dict[str, float]]]]] = None,
                                      metrics: Optional[List[str]] = None,
                                      show_mean_std: bool = True):
            if go is None or px is None:
                raise ImportError('Plotly is required for plotting methods. Install plotly to use this feature.')
            """
            Create a Plotly figure summarising k-fold cross-validation performance.

            Parameters
            ----------
            cv_report : dict, optional
                The output of :meth:`~topologicpy.PyG.PyG.CrossValidate`. If ``None``,
                the method uses :attr:`~topologicpy.PyG.PyG.cv_report`.
            metrics : list[str], optional
                Metrics to include. If ``None`` a default set is chosen based on task:

                - classification: ``["accuracy", "f1", "precision", "recall"]``
                - regression: ``["mae", "rmse", "r2"]``
            show_mean_std : bool, optional
                If ``True``, adds mean and +/-std reference lines when present in ``cv_report``.

            Returns
            -------
            plotly.graph_objects.Figure
                A grouped bar chart (one group per fold, one bar per metric).

            Raises
            ------
            ValueError
                If no cross-validation report is available, or it does not contain fold metrics.
            """
            if cv_report is None:
                cv_report = self.cv_report
            if cv_report is None:
                raise ValueError("No cross-validation report found. Run CrossValidate() first or pass cv_report.")

            fold_metrics = cv_report.get("fold_metrics", [])
            if not fold_metrics:
                raise ValueError("cv_report has no fold_metrics.")

            # default metrics
            if metrics is None:
                if self.config.task == "regression":
                    metrics = ["mae", "rmse", "r2"]
                else:
                    metrics = ["accuracy", "f1", "precision", "recall"]

            folds = [int(fm.get("fold", i)) for i, fm in enumerate(fold_metrics)]

            fig = go.Figure()
            for met in metrics:
                vals = [float(fm.get(met, 0.0)) for fm in fold_metrics]
                fig.add_trace(go.Bar(name=met, x=folds, y=vals))

                if show_mean_std:
                    mean_k = f"mean_{met}"
                    std_k = f"std_{met}"
                    if mean_k in cv_report and std_k in cv_report:
                        mu = float(cv_report[mean_k])
                        sd = float(cv_report[std_k])
                        # mean line
                        fig.add_trace(go.Scatter(
                            x=[min(folds), max(folds)], y=[mu, mu],
                            mode="lines", name=f"{met} mean", line=dict(dash="dash")
                        ))
                        # +/- std (as band using two lines)
                        fig.add_trace(go.Scatter(
                            x=[min(folds), max(folds)], y=[mu + sd, mu + sd],
                            mode="lines", name=f"{met} +std", line=dict(dash="dot")
                        ))
                        fig.add_trace(go.Scatter(
                            x=[min(folds), max(folds)], y=[mu - sd, mu - sd],
                            mode="lines", name=f"{met} -std", line=dict(dash="dot")
                        ))

            fig.update_layout(
                barmode="group",
                title="Cross-Validation Summary",
                xaxis_title="Fold",
                yaxis_title="Metric Value"
            )
            return fig

        # ----------------
        # Dataset loading
        # ----------------
        def _load_csv(self):
            if not isinstance(self.path, str) or (not os.path.exists(self.path)):
                raise ValueError("PyG - Error: path does not exist.")

            gpath = os.path.join(self.path, "graphs.csv")
            npath = os.path.join(self.path, "nodes.csv")
            epath = os.path.join(self.path, "edges.csv")

            if not os.path.exists(gpath) or not os.path.exists(npath) or not os.path.exists(epath):
                raise ValueError("PyG - Error: graphs.csv, nodes.csv, edges.csv must exist in the folder.")

            self.graph_df = pd.read_csv(gpath)
            self.nodes_df = pd.read_csv(npath)
            self.edges_df = pd.read_csv(epath)

        def _feature_columns(self, df: pd.DataFrame, prefix: str) -> List[str]:
            cols = [c for c in df.columns if c.startswith(prefix + "_")]
            def _key(c):
                parts = c.rsplit("_", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    return int(parts[1])
                return 10**9
            return sorted(cols, key=_key)

        
        def _current_feature_schema(self) -> Dict[str, List[str]]:
            """Return the current feature-column schema inferred from the loaded CSVs."""
            cfg = self.config
            assert self.graph_df is not None and self.nodes_df is not None and self.edges_df is not None
            return {
                "graph_feat_cols": self._feature_columns(self.graph_df, cfg.graph_features_header),
                "node_feat_cols": self._feature_columns(self.nodes_df, cfg.node_features_header),
                "edge_feat_cols": self._feature_columns(self.edges_df, cfg.edge_features_header),
            }

        @staticmethod
        def _ensure_feature_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
            """Ensure that *all* columns in ``cols`` exist in ``df`` (missing columns are added as zeros)."""
            if df is None:
                return df
            for c in cols:
                if c not in df.columns:
                    df[c] = 0.0
            return df
        @staticmethod
        def _infer_num_classes(values: np.ndarray) -> int:
            uniq = np.unique(values[~pd.isna(values)])
            return int(len(uniq))

        def _build_data_list(self):
            assert self.graph_df is not None and self.nodes_df is not None and self.edges_df is not None

            cfg = self.config
            gdf = self.graph_df
            ndf = self.nodes_df
            edf = self.edges_df

            graph_feat_cols = self._feature_columns(gdf, cfg.graph_features_header)
            node_feat_cols = self._feature_columns(ndf, cfg.node_features_header)
            edge_feat_cols = self._feature_columns(edf, cfg.edge_features_header)

            # If a saved feature schema exists (e.g. loaded from a trained checkpoint),
            # enforce it so that node/edge feature dimensions match the trained model.
            if getattr(self, "_feature_schema", None):
                sch = self._feature_schema or {}
                if "graph_feat_cols" in sch:
                    graph_feat_cols = list(sch["graph_feat_cols"])
                if "node_feat_cols" in sch:
                    node_feat_cols = list(sch["node_feat_cols"])
                if "edge_feat_cols" in sch:
                    edge_feat_cols = list(sch["edge_feat_cols"])

                # Add any missing columns as zeros (common when one-hot columns are absent in an unseen dataset).
                gdf = self._ensure_feature_columns(gdf, graph_feat_cols)
                ndf = self._ensure_feature_columns(ndf, node_feat_cols)
                edf = self._ensure_feature_columns(edf, edge_feat_cols)

                # Persist back (so later calls see the aligned frames)
                self.graph_df, self.nodes_df, self.edges_df = gdf, ndf, edf

            if len(node_feat_cols) == 0:
                raise ValueError(
                    f"PyG - Error: No node feature columns found. "
                    f"Expected columns starting with '{cfg.node_features_header}_'."
                )

            for gid in gdf[cfg.graph_id_header].unique():
                g_row = gdf[gdf[cfg.graph_id_header] == gid]
                g_nodes = ndf[ndf[cfg.graph_id_header] == gid].sort_values(cfg.node_id_header)
                g_edges = edf[edf[cfg.graph_id_header] == gid]

                x = torch.tensor(g_nodes[node_feat_cols].values, dtype=torch.float32)
                edge_index = torch.tensor(
                    g_edges[[cfg.edge_src_header, cfg.edge_dst_header]].values.T,
                    dtype=torch.long
                )

                data = Data(x=x, edge_index=edge_index)

                if len(edge_feat_cols) > 0:
                    data.edge_attr = torch.tensor(g_edges[edge_feat_cols].values, dtype=torch.float32)

                # graph-level
                if cfg.level == "graph":
                    y_val = g_row[cfg.graph_label_header].values[0]
                    if cfg.graph_label_type == "categorical":
                        data.y = torch.tensor([int(y_val)], dtype=torch.long)
                    else:
                        data.y = torch.tensor([float(y_val)], dtype=torch.float32)

                    if len(graph_feat_cols) > 0:
                        data.u = torch.tensor(g_row[graph_feat_cols].values[0], dtype=torch.float32)

                # node-level
                if cfg.level == "node":
                    y_vals = g_nodes[cfg.node_label_header].values
                    if cfg.node_label_type == "categorical":
                        data.y = torch.tensor(y_vals.astype(int), dtype=torch.long)
                    else:
                        data.y = torch.tensor(y_vals.astype(float), dtype=torch.float32)
                    data.train_mask, data.val_mask, data.test_mask = self._get_or_make_node_masks(g_nodes)

                # edge-level
                if cfg.level == "edge":
                    y_vals = g_edges[cfg.edge_label_header].values
                    if cfg.edge_label_type == "categorical":
                        data.edge_y = torch.tensor(y_vals.astype(int), dtype=torch.long)
                    else:
                        data.edge_y = torch.tensor(y_vals.astype(float), dtype=torch.float32)
                    data.edge_train_mask, data.edge_val_mask, data.edge_test_mask = self._get_or_make_edge_masks(g_edges)

                self.data_list.append(data)

            # output dimensionality
            if not getattr(self, "_freeze_num_outputs", False):
                if cfg.level == "graph":
                    self._num_outputs = self._infer_num_classes(gdf[cfg.graph_label_header].values) if cfg.graph_label_type == "categorical" else 1
                elif cfg.level == "node":
                    self._num_outputs = self._infer_num_classes(ndf[cfg.node_label_header].values) if cfg.node_label_type == "categorical" else 1
                elif cfg.level == "edge":
                    self._num_outputs = self._infer_num_classes(edf[cfg.edge_label_header].values) if cfg.edge_label_type == "categorical" else 1
                elif cfg.level == "link":
                    self._num_outputs = 1
                else:
                    raise ValueError("Unsupported level.")

            # Cache input dimension for convenience (used when saving checkpoints)
            self._in_dim = int(self.data_list[0].x.shape[1]) if self.data_list else self._in_dim
            # Cache input dimension for convenience (used when saving checkpoints)
            self._in_dim = int(self.data_list[0].x.shape[1]) if self.data_list else self._in_dim

        def _get_or_make_node_masks(self, g_nodes: pd.DataFrame):
            cfg = self.config
            cols = g_nodes.columns

            if (cfg.node_train_mask_header in cols) and (cfg.node_val_mask_header in cols) and (cfg.node_test_mask_header in cols):
                train_mask = torch.tensor(g_nodes[cfg.node_train_mask_header].astype(bool).values, dtype=torch.bool)
                val_mask = torch.tensor(g_nodes[cfg.node_val_mask_header].astype(bool).values, dtype=torch.bool)
                test_mask = torch.tensor(g_nodes[cfg.node_test_mask_header].astype(bool).values, dtype=torch.bool)
                return train_mask, val_mask, test_mask

            n = len(g_nodes)
            idx = list(range(n))
            if cfg.shuffle:
                random.Random(cfg.random_state).shuffle(idx)

            n_train = max(1, int(cfg.split[0] * n))
            n_val = max(1, int(cfg.split[1] * n))
            n_test = max(0, n - n_train - n_val)

            train_idx = set(idx[:n_train])
            val_idx = set(idx[n_train:n_train + n_val])
            test_idx = set(idx[n_train + n_val:n_train + n_val + n_test])

            train_mask = torch.tensor([i in train_idx for i in range(n)], dtype=torch.bool)
            val_mask = torch.tensor([i in val_idx for i in range(n)], dtype=torch.bool)
            test_mask = torch.tensor([i in test_idx for i in range(n)], dtype=torch.bool)
            return train_mask, val_mask, test_mask

        def _get_or_make_edge_masks(self, g_edges: pd.DataFrame):
            cfg = self.config
            cols = g_edges.columns

            if (cfg.edge_train_mask_header in cols) and (cfg.edge_val_mask_header in cols) and (cfg.edge_test_mask_header in cols):
                train_mask = torch.tensor(g_edges[cfg.edge_train_mask_header].astype(bool).values, dtype=torch.bool)
                val_mask = torch.tensor(g_edges[cfg.edge_val_mask_header].astype(bool).values, dtype=torch.bool)
                test_mask = torch.tensor(g_edges[cfg.edge_test_mask_header].astype(bool).values, dtype=torch.bool)
                return train_mask, val_mask, test_mask

            n = len(g_edges)
            idx = list(range(n))
            if cfg.shuffle:
                random.Random(cfg.random_state).shuffle(idx)

            n_train = max(1, int(cfg.split[0] * n))
            n_val = max(1, int(cfg.split[1] * n))
            n_test = max(0, n - n_train - n_val)

            train_idx = set(idx[:n_train])
            val_idx = set(idx[n_train:n_train + n_val])
            test_idx = set(idx[n_train + n_val:n_train + n_val + n_test])

            train_mask = torch.tensor([i in train_idx for i in range(n)], dtype=torch.bool)
            val_mask = torch.tensor([i in val_idx for i in range(n)], dtype=torch.bool)
            test_mask = torch.tensor([i in test_idx for i in range(n)], dtype=torch.bool)
            return train_mask, val_mask, test_mask

        # ----------------------------
        # Holdout split (graph-level)
        # ----------------------------
        def _split_holdout(self):
            cfg = self.config
            if cfg.level in ["node", "edge", "link"]:
                self.train_set = self.data_list
                self.val_set = self.data_list
                self.test_set = self.data_list
                return

            n = len(self.data_list)
            idx = list(range(n))
            if cfg.shuffle:
                random.Random(cfg.random_state).shuffle(idx)

            n_train = max(1, int(cfg.split[0] * n))
            n_val = max(1, int(cfg.split[1] * n))
            n_test = max(0, n - n_train - n_val)

            train_idx = idx[:n_train]
            val_idx = idx[n_train:n_train + n_val]
            test_idx = idx[n_train + n_val:n_train + n_val + n_test]

            self.train_set = [self.data_list[i] for i in train_idx]
            self.val_set = [self.data_list[i] for i in val_idx]
            self.test_set = [self.data_list[i] for i in test_idx]

        # --------------
        # Model building
        # --------------
        def _build_model(self):
            cfg = self.config

            if getattr(self, "_in_dim", None) is not None: # Prefer to get the input dimension from the stored value, not the current dataset.
                in_dim = int(self._in_dim)
            else:
                in_dim = int(self.data_list[0].x.shape[1])  # fallback (training-time)

            encoder = _GNNBackbone(
                in_dim=in_dim,
                hidden_dims=cfg.hidden_dims,
                conv=cfg.conv,
                activation=cfg.activation,
                dropout=cfg.dropout,
                batch_norm=cfg.batch_norm,
                residual=cfg.residual
            )

            if cfg.level == "graph":
                head = _GraphHead(encoder.out_dim, self._num_outputs, pooling=cfg.pooling, dropout=cfg.dropout)
                self.model = nn.ModuleDict({"encoder": encoder, "head": head}).to(self.device)
            elif cfg.level == "node":
                head = _NodeHead(encoder.out_dim, self._num_outputs, dropout=cfg.dropout)
                self.model = nn.ModuleDict({"encoder": encoder, "head": head}).to(self.device)
            elif cfg.level == "edge":
                head = _EdgeHead(encoder.out_dim, self._num_outputs, dropout=cfg.dropout)
                self.model = nn.ModuleDict({"encoder": encoder, "head": head}).to(self.device)
            elif cfg.level == "link":
                predictor = _LinkPredictor(encoder.out_dim, hidden=max(32, encoder.out_dim), dropout=cfg.dropout)
                self.model = nn.ModuleDict({"encoder": encoder, "predictor": predictor}).to(self.device)
            else:
                raise ValueError("Unsupported level.")

            if cfg.optimizer == "adamw":
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            else:
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

            if cfg.level == "link":
                self.criterion = nn.BCEWithLogitsLoss()
            else:
                if cfg.task == "regression":
                    self.criterion = nn.MSELoss()
                else:
                    self.criterion = nn.CrossEntropyLoss()

        def _apply_gradients(self):
            cfg = self.config
            if cfg.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(cfg.gradient_clip_norm))
            self.optimizer.step()

        # -----------------------
        # Training / evaluation
        # -----------------------
        def Train(self, epochs: Optional[int] = None, batch_size: Optional[int] = None) -> Dict[str, List[float]]:
            """
            Train the model using the current configuration.

            Training behaviour depends on :attr:`~topologicpy.PyG.PyG.config.level`:

            - ``"graph"``: uses the current holdout split (train/val sets)
            - ``"node"`` : uses in-graph boolean masks (``train_mask``, ``val_mask``)
            - ``"edge"`` : uses in-graph boolean masks (``edge_train_mask``, ``edge_val_mask``)
            - ``"link"`` : uses :class:`torch_geometric.transforms.RandomLinkSplit` per graph

            Parameters
            ----------
            epochs : int, optional
                If provided, overrides ``config.epochs`` for this run.
            batch_size : int, optional
                If provided, overrides ``config.batch_size`` for this run. For node/edge/link
                tasks the loader uses ``batch_size=1`` (one graph at a time).

            Returns
            -------
            dict
                Training history dictionary with keys ``"train_loss"`` and ``"val_loss"``.
                Each value is a list of floats (one per epoch).

            Notes
            -----
            - For graph-level tasks, early stopping can be enabled via
              ``config.early_stopping`` and ``config.early_stopping_patience``.
            - For k-fold cross-validation on graph-level tasks, use
              :meth:`~topologicpy.PyG.PyG.CrossValidate` instead.
            """
            cfg = self.config
            if epochs is not None:
                cfg.epochs = int(epochs)
            if batch_size is not None:
                cfg.batch_size = int(batch_size)

            self.history = {"train_loss": [], "val_loss": []}

            if cfg.level == "graph":
                train_loader = DataLoader(self.train_set, batch_size=cfg.batch_size, shuffle=True)
                val_loader = DataLoader(self.val_set, batch_size=cfg.batch_size, shuffle=False)

                best_val = float("inf")
                patience = 0

                for _ in range(cfg.epochs):
                    tr = self._train_epoch_graph(train_loader)
                    va = self._eval_epoch_graph(val_loader)
                    self.history["train_loss"].append(tr)
                    self.history["val_loss"].append(va)

                    if cfg.early_stopping:
                        if va < best_val - 1e-9:
                            best_val = va
                            patience = 0
                        else:
                            patience += 1
                            if patience >= int(cfg.early_stopping_patience):
                                break

            elif cfg.level == "node":
                train_loader = DataLoader(self.data_list, batch_size=1, shuffle=True)
                val_loader = DataLoader(self.data_list, batch_size=1, shuffle=False)
                for _ in range(cfg.epochs):
                    tr = self._train_epoch_node(train_loader)
                    va = self._eval_epoch_node(val_loader)
                    self.history["train_loss"].append(tr)
                    self.history["val_loss"].append(va)

            elif cfg.level == "edge":
                train_loader = DataLoader(self.data_list, batch_size=1, shuffle=True)
                val_loader = DataLoader(self.data_list, batch_size=1, shuffle=False)
                for _ in range(cfg.epochs):
                    tr = self._train_epoch_edge(train_loader)
                    va = self._eval_epoch_edge(val_loader)
                    self.history["train_loss"].append(tr)
                    self.history["val_loss"].append(va)

            elif cfg.level == "link":
                train_loader = DataLoader(self.data_list, batch_size=1, shuffle=True)
                val_loader = DataLoader(self.data_list, batch_size=1, shuffle=False)
                for _ in range(cfg.epochs):
                    tr = self._train_epoch_link(train_loader)
                    va = self._eval_epoch_link(val_loader)
                    self.history["train_loss"].append(tr)
                    self.history["val_loss"].append(va)

            else:
                raise ValueError("Unsupported level.")

            return self.history

        def CrossValidate(self,
                          k_folds: Optional[int] = None,
                          epochs: Optional[int] = None,
                          batch_size: Optional[int] = None) -> Dict[str, Union[float, List[Dict[str, float]]]]:
            """
            Perform k-fold cross-validation for graph-level tasks.

            This method rebuilds and retrains a fresh model per fold, evaluates on the fold's
            held-out set, and returns fold-wise metrics along with mean/std aggregates.

            Parameters
            ----------
            k_folds : int, optional
                Number of folds. Defaults to ``config.k_folds``.
            epochs : int, optional
                Training epochs per fold. Defaults to ``config.epochs``.
            batch_size : int, optional
                Batch size for DataLoader. Defaults to ``config.batch_size``.

            Returns
            -------
            dict
                A dictionary of the form::

                    {
                      "fold_metrics": [{"fold": 0, ...}, {"fold": 1, ...}, ...],
                      "mean_<metric>": ...,
                      "std_<metric>": ...
                    }

            Raises
            ------
            ValueError
                If called for non-graph levels, or if ``k_folds < 2``.

            Notes
            -----
            - Stratified folding is available for categorical graph labels when
              ``config.k_stratify`` is ``True``.
            - Cross-validation is intentionally limited to graph-level tasks; node/edge tasks
              typically rely on per-graph masks rather than splitting graphs.
            """
            cfg = self.config
            if cfg.level != "graph":
                raise ValueError("CrossValidate is supported for graph-level tasks only.")

            if k_folds is None:
                k_folds = int(cfg.k_folds)
            if k_folds < 2:
                raise ValueError("k_folds must be >= 2.")
            if epochs is not None:
                cfg.epochs = int(epochs)
            if batch_size is not None:
                cfg.batch_size = int(batch_size)

            n = len(self.data_list)
            indices = np.arange(n)

            # Stratification labels (optional)
            y = None
            if cfg.k_stratify and cfg.task == "classification" and cfg.graph_label_type == "categorical":
                y = np.array([int(d.y.item()) for d in self.data_list], dtype=int)

            rng = np.random.RandomState(cfg.random_state)
            if cfg.k_shuffle:
                rng.shuffle(indices)

            # Build folds
            if y is None:
                folds = np.array_split(indices, k_folds)
            else:
                folds = [np.array([], dtype=int) for _ in range(k_folds)]
                classes = np.unique(y)
                for c in classes:
                    cls_idx = indices[y[indices] == c]
                    cls_chunks = np.array_split(cls_idx, k_folds)
                    for fi in range(k_folds):
                        folds[fi] = np.concatenate([folds[fi], cls_chunks[fi]])
                folds = [rng.permutation(f) for f in folds]

            fold_metrics: List[Dict[str, float]] = []
            base_config = copy.deepcopy(cfg)

            for fi in range(k_folds):
                test_idx = folds[fi]
                train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != fi])

                train_set = [self.data_list[i] for i in train_idx.tolist()]
                test_set = [self.data_list[i] for i in test_idx.tolist()]

                # Fresh model per fold
                self.config = copy.deepcopy(base_config)
                self._build_model()
                self.history = {"train_loss": [], "val_loss": []}

                train_loader = DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True)
                test_loader = DataLoader(test_set, batch_size=self.config.batch_size, shuffle=False)

                best_loss = float("inf")
                patience = 0

                for _ in range(self.config.epochs):
                    tr_loss = self._train_epoch_graph(train_loader)
                    te_loss = self._eval_epoch_graph(test_loader)
                    self.history["train_loss"].append(tr_loss)
                    self.history["val_loss"].append(te_loss)

                    if self.config.early_stopping:
                        if te_loss < best_loss - 1e-9:
                            best_loss = te_loss
                            patience = 0
                        else:
                            patience += 1
                            if patience >= int(self.config.early_stopping_patience):
                                break

                # Metrics (unprefixed) for the fold
                metrics = self._metrics_graph(test_loader, prefix="")
                metrics["fold"] = float(fi)
                fold_metrics.append(metrics)

            # Restore original config and rebuild model
            self.config = copy.deepcopy(base_config)
            self._build_model()

            # Aggregate
            summary: Dict[str, Union[float, List[Dict[str, float]]]] = {"fold_metrics": fold_metrics}
            metric_keys = [k for k in fold_metrics[0].keys()] if fold_metrics else []
            metric_keys = [k for k in metric_keys if k != "fold"]

            for k in metric_keys:
                vals = np.array([fm[k] for fm in fold_metrics], dtype=float)
                summary[f"mean_{k}"] = float(np.mean(vals))
                summary[f"std_{k}"] = float(np.std(vals))

            self.cv_report = summary
            return summary

        def Validate(self) -> Dict[str, float]:
            """
            Compute metrics on the validation split.

            Returns
            -------
            dict
                A dictionary of metric values. Key names are prefixed depending on task:

                - graph-level: keys are prefixed with ``"val_"``
                - node/edge/link: keys are prefixed with ``"val_"`` via internal helpers

            Raises
            ------
            ValueError
                If the configured level is unsupported.
            """
            cfg = self.config
            if cfg.level == "graph":
                loader = DataLoader(self.val_set, batch_size=cfg.batch_size, shuffle=False)
                return self._metrics_graph(loader, prefix="val_")
            if cfg.level == "node":
                loader = DataLoader(self.data_list, batch_size=1, shuffle=False)
                return self._metrics_node(loader, split="val")
            if cfg.level == "edge":
                loader = DataLoader(self.data_list, batch_size=1, shuffle=False)
                return self._metrics_edge(loader, split="val")
            if cfg.level == "link":
                loader = DataLoader(self.data_list, batch_size=1, shuffle=False)
                return self._metrics_link(loader, split="val")
            raise ValueError("Unsupported level.")

        def Test(self) -> Dict[str, float]:
            """
            Compute metrics on the test split.

            Returns
            -------
            dict
                A dictionary of metric values. Key names are prefixed depending on task:

                - graph-level: keys are prefixed with ``"test_"``
                - node/edge/link: keys are prefixed with ``"test_"`` via internal helpers

            Raises
            ------
            ValueError
                If the configured level is unsupported.
            """
            cfg = self.config
            if cfg.level == "graph":
                loader = DataLoader(self.test_set, batch_size=cfg.batch_size, shuffle=False)
                return self._metrics_graph(loader, prefix="test_")
            if cfg.level == "node":
                loader = DataLoader(self.data_list, batch_size=1, shuffle=False)
                return self._metrics_node(loader, split="test")
            if cfg.level == "edge":
                loader = DataLoader(self.data_list, batch_size=1, shuffle=False)
                return self._metrics_edge(loader, split="test")
            if cfg.level == "link":
                loader = DataLoader(self.data_list, batch_size=1, shuffle=False)
                return self._metrics_link(loader, split="test")
            raise ValueError("Unsupported level.")
        
        def Predict(self,
                    split: str = "all",
                    threshold: float = 0.5,
                    return_logits: bool = False,
                    return_probs: bool = True,
                    return_embeddings: bool = False,
                    attach_to_data: bool = False,
                    pred_key: str = "pred",
                    prob_key: str = "prob",
                    logits_key: str = "logits",
                    emb_key: str = "emb") -> Dict[str, object]:
            """Run inference (prediction) using the current model on the loaded dataset.

            This method is designed for **post-training** workflows, including the common pattern
            of *train → save → reload → predict on unseen data*. It performs forward passes only
            (no gradient computation) and returns predictions in a compact, serializable form.

            Behaviour depends on :attr:`~topologicpy.PyG.PyG.config.level`:

            - ``"graph"``: graph-level prediction using a mini-batched :class:`~torch_geometric.loader.DataLoader`
            - ``"node"`` : node-level prediction using node masks (``train_mask``, ``val_mask``, ``test_mask``)
            - ``"edge"`` : edge-level prediction using edge masks (``edge_train_mask``, ``edge_val_mask``, ``edge_test_mask``)
            - ``"link"`` : link prediction using :class:`~torch_geometric.transforms.RandomLinkSplit` per graph

            Parameters
            ----------
            split : str, optional
                The subset to predict. Supported values depend on ``config.level``:

                - graph-level: ``"train"``, ``"val"``, ``"test"``, ``"all"``
                - node-level : ``"train"``, ``"val"``, ``"test"``, ``"all"`` (``"all"`` returns full-length vectors)
                - edge-level : ``"train"``, ``"val"``, ``"test"``, ``"all"`` (``"all"`` returns full-length vectors)
                - link-level : ``"train"``, ``"val"``, ``"test"`` (``"all"`` is treated as ``"test"``)

                Default is ``"all"``.
            threshold : float, optional
                Threshold for converting link-prediction probabilities into binary labels.
                Only used when ``config.level == "link"``. Default is 0.5.
            return_logits : bool, optional
                If True, includes raw model outputs (logits) in the returned dictionary.
                For regression tasks, logits are the raw predictions. Default is False.
            return_probs : bool, optional
                If True, includes probabilities/scores when applicable:

                - classification: softmax probabilities
                - link prediction: sigmoid probabilities
                - regression: ignored (no probabilities)

                Default is True.
            return_embeddings : bool, optional
                If True, includes the **node embeddings** produced by the GNN backbone
                (the output of ``model["encoder"]``) for each predicted batch/graph.
                Default is False.
            attach_to_data : bool, optional
                If True, attaches prediction tensors to each :class:`~torch_geometric.data.Data`
                object in :attr:`~topologicpy.PyG.PyG.data_list` using keys ``pred_key``,
                ``prob_key``, ``logits_key``, and ``emb_key``. This is useful for downstream
                processing (e.g., exporting to CSV or mapping back to Topologic entities).
                Default is False.
            pred_key : str, optional
                Attribute name to attach predicted labels/values to each Data object when
                ``attach_to_data`` is True. Default is ``"pred"``.
            prob_key : str, optional
                Attribute name to attach probabilities/scores to each Data object when
                ``attach_to_data`` is True. Default is ``"prob"``.
            logits_key : str, optional
                Attribute name to attach logits/raw outputs to each Data object when
                ``attach_to_data`` is True. Default is ``"logits"``.
            emb_key : str, optional
                Attribute name to attach encoder embeddings to each Data object when
                ``attach_to_data`` is True. Default is ``"emb"``.

            Returns
            -------
            dict
                A dictionary containing (at minimum) the key ``"pred"`` with predictions.

                **Graph-level**
                    - ``"pred"``: ``(N,)`` predicted class indices or regression values
                    - ``"y_true"``: ``(N,)`` true labels/targets if present
                    - ``"index"``: ``(N,)`` integer indices aligned with ``self.data_list`` order

                **Node/Edge-level**
                    - ``"pred"``: list of arrays (one per graph) unless ``split != "all"``
                    - ``"y_true"``: list of arrays (one per graph) if present
                    - ``"mask"``: mask name used when ``split in {train,val,test}``

                **Link-level**
                    - ``"score"``: sigmoid probabilities for edge_label_index samples
                    - ``"pred"``: binary predictions derived from ``threshold``
                    - ``"y_true"``: binary ground truth labels for sampled links

            Raises
            ------
            ValueError
                If ``split`` or ``config.level`` is unsupported, or if the model is not initialised.

            Notes
            -----
            - This method assumes you have already called :meth:`~topologicpy.PyG.PyG.ByCSVPath`
            (or otherwise populated :attr:`~topologicpy.PyG.PyG.data_list`), and that
            :attr:`~topologicpy.PyG.PyG.model` is loaded/initialised (e.g., via
            :meth:`~topologicpy.PyG.PyG.Train` or :meth:`~topologicpy.PyG.PyG.LoadModel`).
            - For classification tasks, the returned class indices follow the encoding present in the CSV labels.
            """
            if self.model is None:
                raise ValueError("PyG - Error: Model is not initialised. Train or LoadModel first.")

            cfg = self.config
            split = (split or "all").lower().strip()

            # ----------------------------
            # Graph-level predictions
            # ----------------------------
            if cfg.level == "graph":
                if split == "train":
                    data_src = self.train_set
                elif split in ("val", "valid", "validation"):
                    data_src = self.val_set
                elif split == "test":
                    data_src = self.test_set
                elif split == "all":
                    data_src = self.data_list
                else:
                    raise ValueError("PyG - Error: split must be one of {'train','val','test','all'} for graph-level.")

                loader = DataLoader(data_src, batch_size=cfg.batch_size, shuffle=False)

                self.model.eval()
                all_logits, all_probs, all_pred, all_true, all_idx, all_emb = [], [], [], [], [], []

                # Build a stable index mapping back to self.data_list order
                # For train/val/test subsets, we map by object identity.
                id_to_index = {id(d): i for i, d in enumerate(self.data_list)}

                with torch.no_grad():
                    for batch in loader:
                        # When DataLoader batches, object identity is not preserved; keep it simple:
                        # we return indices only for 'all'. For subsets we return -1.
                        batch = batch.to(self.device)
                        node_emb = self.model["encoder"](batch.x, batch.edge_index)
                        logits = self.model["head"](node_emb, batch.batch)

                        if return_embeddings:
                            # store per-batch node embeddings on CPU
                            all_emb.append(node_emb.detach().cpu())

                        if cfg.task == "regression":
                            pred = logits.squeeze(-1).detach().cpu().numpy()
                            all_pred.extend(pred.tolist())
                            if hasattr(batch, "y") and batch.y is not None:
                                all_true.extend(batch.y.squeeze(-1).detach().cpu().numpy().tolist())
                            if return_logits:
                                all_logits.append(logits.detach().cpu())
                        else:
                            probs = F.softmax(logits, dim=-1)
                            pred = probs.argmax(dim=-1)
                            all_pred.extend(pred.detach().cpu().numpy().tolist())
                            if hasattr(batch, "y") and batch.y is not None:
                                all_true.extend(batch.y.detach().cpu().numpy().tolist())
                            if return_probs:
                                all_probs.append(probs.detach().cpu())
                            if return_logits:
                                all_logits.append(logits.detach().cpu())

                        if split == "all":
                            # For 'all' we can safely report sequential indices in data_list order
                            # by tracking running count.
                            pass

                # Index for graph-level: sequential for chosen subset.
                all_idx = list(range(len(all_pred)))

                out = {
                    "pred": np.array(all_pred),
                    "index": np.array(all_idx)
                }
                if len(all_true) > 0:
                    out["y_true"] = np.array(all_true)
                if return_probs and len(all_probs) > 0:
                    out["prob"] = torch.cat(all_probs, dim=0).numpy()
                if return_logits and len(all_logits) > 0:
                    out["logits"] = torch.cat(all_logits, dim=0).numpy()
                if return_embeddings and len(all_emb) > 0:
                    out["emb"] = all_emb  # list of torch tensors on CPU (variable sized)

                # Optional attachment (best-effort): only for split='all' since we need alignment
                if attach_to_data and split == "all":
                    # We need per-graph predictions; with graph batching this is already per-graph.
                    # Attach to each Data in data_list.
                    for i, d in enumerate(self.data_list):
                        setattr(d, pred_key, out["pred"][i])
                        if "y_true" in out:
                            setattr(d, "y_true", out["y_true"][i])
                    if "prob" in out:
                        for i, d in enumerate(self.data_list):
                            setattr(d, prob_key, out["prob"][i])
                    if "logits" in out:
                        for i, d in enumerate(self.data_list):
                            setattr(d, logits_key, out["logits"][i])

                return out

            # ----------------------------
            # Node-level predictions
            # ----------------------------
            if cfg.level == "node":
                loader = DataLoader(self.data_list, batch_size=1, shuffle=False)
                self.model.eval()

                if split == "train":
                    mask_name = "train_mask"
                elif split in ("val", "valid", "validation"):
                    mask_name = "val_mask"
                elif split == "test":
                    mask_name = "test_mask"
                elif split == "all":
                    mask_name = None
                else:
                    raise ValueError("PyG - Error: split must be one of {'train','val','test','all'} for node-level.")

                preds_per_graph, true_per_graph = [], []
                probs_per_graph, logits_per_graph, emb_per_graph = [], [], []

                with torch.no_grad():
                    for data in loader:
                        data = data.to(self.device)
                        node_emb = self.model["encoder"](data.x, data.edge_index)
                        logits = self.model["head"](node_emb)

                        if return_embeddings:
                            emb_cpu = node_emb.detach().cpu()
                            emb_per_graph.append(emb_cpu)

                        if cfg.task == "regression":
                            pred_all = logits.squeeze(-1)
                            if mask_name is not None:
                                mask = getattr(data, mask_name)
                                pred = pred_all[mask]
                                y_true = data.y[mask] if hasattr(data, "y") else None
                            else:
                                pred = pred_all
                                y_true = data.y if hasattr(data, "y") else None

                            preds_per_graph.append(pred.detach().cpu().numpy())
                            if y_true is not None:
                                true_per_graph.append(y_true.detach().cpu().numpy())
                            if return_logits:
                                logits_per_graph.append(logits.detach().cpu().numpy())
                        else:
                            probs_all = F.softmax(logits, dim=-1)
                            pred_all = probs_all.argmax(dim=-1)
                            if mask_name is not None:
                                mask = getattr(data, mask_name)
                                pred = pred_all[mask]
                                probs = probs_all[mask]
                                y_true = data.y[mask] if hasattr(data, "y") else None
                            else:
                                pred = pred_all
                                probs = probs_all
                                y_true = data.y if hasattr(data, "y") else None

                            preds_per_graph.append(pred.detach().cpu().numpy())
                            if y_true is not None:
                                true_per_graph.append(y_true.detach().cpu().numpy())
                            if return_probs:
                                probs_per_graph.append(probs.detach().cpu().numpy())
                            if return_logits:
                                logits_per_graph.append(logits.detach().cpu().numpy())

                        if attach_to_data:
                            # Attach full-length predictions to the underlying Data object (CPU arrays).
                            # For split!=all, we still attach full-length arrays for convenience.
                            d0 = data.cpu()
                            if cfg.task == "regression":
                                setattr(d0, pred_key, pred_all.detach().cpu().numpy())
                            else:
                                setattr(d0, pred_key, pred_all.detach().cpu().numpy())
                                if return_probs:
                                    setattr(d0, prob_key, probs_all.detach().cpu().numpy())
                                if return_logits:
                                    setattr(d0, logits_key, logits.detach().cpu().numpy())
                            if return_embeddings:
                                setattr(d0, emb_key, node_emb.detach().cpu().numpy())

                out = {"pred": preds_per_graph}
                if len(true_per_graph) > 0:
                    out["y_true"] = true_per_graph
                if mask_name is not None:
                    out["mask"] = mask_name
                if return_probs and len(probs_per_graph) > 0:
                    out["prob"] = probs_per_graph
                if return_logits and len(logits_per_graph) > 0:
                    out["logits"] = logits_per_graph
                if return_embeddings and len(emb_per_graph) > 0:
                    out["emb"] = emb_per_graph
                return out

            # ----------------------------
            # Edge-level predictions
            # ----------------------------
            if cfg.level == "edge":
                loader = DataLoader(self.data_list, batch_size=1, shuffle=False)
                self.model.eval()

                if split == "train":
                    mask_name = "edge_train_mask"
                elif split in ("val", "valid", "validation"):
                    mask_name = "edge_val_mask"
                elif split == "test":
                    mask_name = "edge_test_mask"
                elif split == "all":
                    mask_name = None
                else:
                    raise ValueError("PyG - Error: split must be one of {'train','val','test','all'} for edge-level.")

                preds_per_graph, true_per_graph = [], []
                probs_per_graph, logits_per_graph, emb_per_graph = [], [], []

                with torch.no_grad():
                    for data in loader:
                        data = data.to(self.device)
                        node_emb = self.model["encoder"](data.x, data.edge_index)
                        logits = self.model["head"](node_emb, data.edge_index)

                        if return_embeddings:
                            emb_per_graph.append(node_emb.detach().cpu())

                        if cfg.task == "regression":
                            pred_all = logits.squeeze(-1)
                            if mask_name is not None:
                                mask = getattr(data, mask_name)
                                pred = pred_all[mask]
                                y_true = data.edge_y[mask] if hasattr(data, "edge_y") else None
                            else:
                                pred = pred_all
                                y_true = data.edge_y if hasattr(data, "edge_y") else None

                            preds_per_graph.append(pred.detach().cpu().numpy())
                            if y_true is not None:
                                true_per_graph.append(y_true.detach().cpu().numpy())
                            if return_logits:
                                logits_per_graph.append(logits.detach().cpu().numpy())
                        else:
                            probs_all = F.softmax(logits, dim=-1)
                            pred_all = probs_all.argmax(dim=-1)
                            if mask_name is not None:
                                mask = getattr(data, mask_name)
                                pred = pred_all[mask]
                                probs = probs_all[mask]
                                y_true = data.edge_y[mask] if hasattr(data, "edge_y") else None
                            else:
                                pred = pred_all
                                probs = probs_all
                                y_true = data.edge_y if hasattr(data, "edge_y") else None

                            preds_per_graph.append(pred.detach().cpu().numpy())
                            if y_true is not None:
                                true_per_graph.append(y_true.detach().cpu().numpy())
                            if return_probs:
                                probs_per_graph.append(probs.detach().cpu().numpy())
                            if return_logits:
                                logits_per_graph.append(logits.detach().cpu().numpy())

                        if attach_to_data:
                            d0 = data.cpu()
                            setattr(d0, pred_key, pred_all.detach().cpu().numpy())
                            if cfg.task != "regression" and return_probs:
                                setattr(d0, prob_key, probs_all.detach().cpu().numpy())
                            if return_logits:
                                setattr(d0, logits_key, logits.detach().cpu().numpy())
                            if return_embeddings:
                                setattr(d0, emb_key, node_emb.detach().cpu().numpy())

                out = {"pred": preds_per_graph}
                if len(true_per_graph) > 0:
                    out["y_true"] = true_per_graph
                if mask_name is not None:
                    out["mask"] = mask_name
                if return_probs and len(probs_per_graph) > 0:
                    out["prob"] = probs_per_graph
                if return_logits and len(logits_per_graph) > 0:
                    out["logits"] = logits_per_graph
                if return_embeddings and len(emb_per_graph) > 0:
                    out["emb"] = emb_per_graph
                return out

            # ----------------------------
            # Link prediction
            # ----------------------------
            if cfg.level == "link":
                loader = DataLoader(self.data_list, batch_size=1, shuffle=False)
                self.model.eval()

                use_split = split
                if use_split == "all":
                    use_split = "test"
                if use_split not in ("train", "val", "test"):
                    raise ValueError("PyG - Error: split must be one of {'train','val','test'} for link prediction.")

                split_tf = RandomLinkSplit(
                    num_val=cfg.link_val_ratio,
                    num_test=cfg.link_test_ratio,
                    is_undirected=cfg.link_is_undirected,
                    add_negative_train_samples=True,
                    neg_sampling_ratio=1.0
                )

                all_score, all_true = [], []
                with torch.no_grad():
                    for data in loader:
                        tr, va, te = split_tf(data)
                        use = {"train": tr, "val": va, "test": te}[use_split].to(self.device)
                        node_emb = self.model["encoder"](use.x, use.edge_index)
                        logits = self.model["predictor"](node_emb, use.edge_label_index)
                        score = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
                        y = use.edge_label.detach().cpu().numpy().reshape(-1)
                        all_score.extend(score.tolist())
                        all_true.extend(y.tolist())

                score_arr = np.array(all_score)
                true_arr = np.array(all_true)
                pred_arr = (score_arr >= float(threshold)).astype(int)

                out = {"score": score_arr, "pred": pred_arr, "y_true": true_arr, "threshold": float(threshold)}
                if return_logits:
                    # logits are not stored batch-wise here; expose scores only unless needed
                    pass
                return out

            raise ValueError("Unsupported level.")

        # --------
        # Epochs
        # --------
        def _loss_from_logits(self, logits, y, task: TaskKind):
            if task == "regression":
                pred = logits.squeeze(-1)
                return self.criterion(pred.float(), y.float())
            return self.criterion(logits, y.long())

        def _train_epoch_graph(self, loader):
            self.model.train()
            losses = []
            for batch in loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                node_emb = self.model["encoder"](batch.x, batch.edge_index)
                logits = self.model["head"](node_emb, batch.batch)
                loss = self._loss_from_logits(logits, batch.y, self.config.task)
                loss.backward()
                self._apply_gradients()
                losses.append(float(loss.detach().cpu()))
            return float(np.mean(losses)) if losses else 0.0

        @torch.no_grad()
        def _eval_epoch_graph(self, loader):
            self.model.eval()
            losses = []
            for batch in loader:
                batch = batch.to(self.device)
                node_emb = self.model["encoder"](batch.x, batch.edge_index)
                logits = self.model["head"](node_emb, batch.batch)
                loss = self._loss_from_logits(logits, batch.y, self.config.task)
                losses.append(float(loss.detach().cpu()))
            return float(np.mean(losses)) if losses else 0.0

        def _train_epoch_node(self, loader):
            self.model.train()
            losses = []
            for data in loader:
                data = data.to(self.device)
                self.optimizer.zero_grad()
                node_emb = self.model["encoder"](data.x, data.edge_index)
                logits = self.model["head"](node_emb)
                mask = data.train_mask
                loss = self._loss_from_logits(logits[mask], data.y[mask], self.config.task)
                loss.backward()
                self._apply_gradients()
                losses.append(float(loss.detach().cpu()))
            return float(np.mean(losses)) if losses else 0.0

        @torch.no_grad()
        def _eval_epoch_node(self, loader):
            self.model.eval()
            losses = []
            for data in loader:
                data = data.to(self.device)
                node_emb = self.model["encoder"](data.x, data.edge_index)
                logits = self.model["head"](node_emb)
                mask = data.val_mask
                loss = self._loss_from_logits(logits[mask], data.y[mask], self.config.task)
                losses.append(float(loss.detach().cpu()))
            return float(np.mean(losses)) if losses else 0.0

        def _train_epoch_edge(self, loader):
            self.model.train()
            losses = []
            for data in loader:
                data = data.to(self.device)
                self.optimizer.zero_grad()
                node_emb = self.model["encoder"](data.x, data.edge_index)
                logits = self.model["head"](node_emb, data.edge_index)
                mask = data.edge_train_mask
                loss = self._loss_from_logits(logits[mask], data.edge_y[mask], self.config.task)
                loss.backward()
                self._apply_gradients()
                losses.append(float(loss.detach().cpu()))
            return float(np.mean(losses)) if losses else 0.0

        @torch.no_grad()
        def _eval_epoch_edge(self, loader):
            self.model.eval()
            losses = []
            for data in loader:
                data = data.to(self.device)
                node_emb = self.model["encoder"](data.x, data.edge_index)
                logits = self.model["head"](node_emb, data.edge_index)
                mask = data.edge_val_mask
                loss = self._loss_from_logits(logits[mask], data.edge_y[mask], self.config.task)
                losses.append(float(loss.detach().cpu()))
            return float(np.mean(losses)) if losses else 0.0

        def _train_epoch_link(self, loader):
            self.model.train()
            losses = []
            split = RandomLinkSplit(
                num_val=self.config.link_val_ratio,
                num_test=self.config.link_test_ratio,
                is_undirected=self.config.link_is_undirected,
                add_negative_train_samples=True,
                neg_sampling_ratio=1.0
            )
            for data in loader:
                train_data, _, _ = split(data)
                train_data = train_data.to(self.device)
                self.optimizer.zero_grad()
                node_emb = self.model["encoder"](train_data.x, train_data.edge_index)
                logits = self.model["predictor"](node_emb, train_data.edge_label_index)
                loss = self.criterion(logits, train_data.edge_label.float())
                loss.backward()
                self._apply_gradients()
                losses.append(float(loss.detach().cpu()))
            return float(np.mean(losses)) if losses else 0.0

        @torch.no_grad()
        def _eval_epoch_link(self, loader):
            self.model.eval()
            losses = []
            split = RandomLinkSplit(
                num_val=self.config.link_val_ratio,
                num_test=self.config.link_test_ratio,
                is_undirected=self.config.link_is_undirected,
                add_negative_train_samples=True,
                neg_sampling_ratio=1.0
            )
            for data in loader:
                _, val_data, _ = split(data)
                val_data = val_data.to(self.device)
                node_emb = self.model["encoder"](val_data.x, val_data.edge_index)
                logits = self.model["predictor"](node_emb, val_data.edge_label_index)
                loss = self.criterion(logits, val_data.edge_label.float())
                losses.append(float(loss.detach().cpu()))
            return float(np.mean(losses)) if losses else 0.0

        # --------------
        # Metrics helpers
        # --------------
        @torch.no_grad()
        def _predict_graph(self, loader):
            self.model.eval()
            y_true, y_pred = [], []
            for batch in loader:
                batch = batch.to(self.device)
                node_emb = self.model["encoder"](batch.x, batch.edge_index)
                out = self.model["head"](node_emb, batch.batch)
                if self.config.task == "regression":
                    y_true.extend(batch.y.squeeze(-1).detach().cpu().numpy().tolist())
                    y_pred.extend(out.squeeze(-1).detach().cpu().numpy().tolist())
                else:
                    probs = F.softmax(out, dim=-1)
                    y_true.extend(batch.y.detach().cpu().numpy().tolist())
                    y_pred.extend(probs.argmax(dim=-1).detach().cpu().numpy().tolist())
            return np.array(y_true), np.array(y_pred)

        @torch.no_grad()
        def _predict_node(self, loader, mask_name: str):
            self.model.eval()
            y_true, y_pred = [], []
            for data in loader:
                data = data.to(self.device)
                node_emb = self.model["encoder"](data.x, data.edge_index)
                out = self.model["head"](node_emb)
                mask = getattr(data, mask_name)
                if self.config.task == "regression":
                    y_true.extend(data.y[mask].detach().cpu().numpy().tolist())
                    y_pred.extend(out.squeeze(-1)[mask].detach().cpu().numpy().tolist())
                else:
                    probs = F.softmax(out, dim=-1)
                    y_true.extend(data.y[mask].detach().cpu().numpy().tolist())
                    y_pred.extend(probs.argmax(dim=-1)[mask].detach().cpu().numpy().tolist())
            return np.array(y_true), np.array(y_pred)

        @torch.no_grad()
        def _predict_edge(self, loader, mask_name: str):
            self.model.eval()
            y_true, y_pred = [], []
            for data in loader:
                data = data.to(self.device)
                node_emb = self.model["encoder"](data.x, data.edge_index)
                out = self.model["head"](node_emb, data.edge_index)
                mask = getattr(data, mask_name)
                if self.config.task == "regression":
                    y_true.extend(data.edge_y[mask].detach().cpu().numpy().tolist())
                    y_pred.extend(out.squeeze(-1)[mask].detach().cpu().numpy().tolist())
                else:
                    probs = F.softmax(out, dim=-1)
                    y_true.extend(data.edge_y[mask].detach().cpu().numpy().tolist())
                    y_pred.extend(probs.argmax(dim=-1)[mask].detach().cpu().numpy().tolist())
            return np.array(y_true), np.array(y_pred)

        @torch.no_grad()
        def _predict_link(self, loader, split_name: str):
            self.model.eval()
            split = RandomLinkSplit(
                num_val=self.config.link_val_ratio,
                num_test=self.config.link_test_ratio,
                is_undirected=self.config.link_is_undirected,
                add_negative_train_samples=True,
                neg_sampling_ratio=1.0
            )
            y_true, y_score = [], []
            for data in loader:
                tr, va, te = split(data)
                use = {"train": tr, "val": va, "test": te}[split_name]
                use = use.to(self.device)
                node_emb = self.model["encoder"](use.x, use.edge_index)
                logits = self.model["predictor"](node_emb, use.edge_label_index)
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                y = use.edge_label.detach().cpu().numpy()
                y_true.extend(y.tolist())
                y_score.extend(probs.tolist())
            return np.array(y_true), np.array(y_score)

        # ----------
        # Public metrics API
        # ----------
        def _metrics_graph(self, loader, prefix: str):
            y_true, y_pred = self._predict_graph(loader)
            if self.config.task == "regression":
                return self._regression_metrics(y_true, y_pred, prefix=prefix)
            return self._classification_metrics(y_true, y_pred, prefix=prefix)

        def _metrics_node(self, loader, split: Literal["train", "val", "test"]):
            mask = "train_mask" if split == "train" else ("val_mask" if split == "val" else "test_mask")
            y_true, y_pred = self._predict_node(loader, mask)
            if self.config.task == "regression":
                return self._regression_metrics(y_true, y_pred, prefix=f"{split}_")
            return self._classification_metrics(y_true, y_pred, prefix=f"{split}_")

        def _metrics_edge(self, loader, split: Literal["train", "val", "test"]):
            mask = "edge_train_mask" if split == "train" else ("edge_val_mask" if split == "val" else "edge_test_mask")
            y_true, y_pred = self._predict_edge(loader, mask)
            if self.config.task == "regression":
                return self._regression_metrics(y_true, y_pred, prefix=f"{split}_")
            return self._classification_metrics(y_true, y_pred, prefix=f"{split}_")

        def _metrics_link(self, loader, split: Literal["train", "val", "test"]):
            y_true, y_score = self._predict_link(loader, split)
            y_pred = (y_score >= 0.5).astype(int)
            return self._classification_metrics(y_true, y_pred, prefix=f"{split}_")

        @staticmethod
        def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> Dict[str, float]:
            acc = float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0
            prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
            return {
                f"{prefix}accuracy": float(acc),
                f"{prefix}precision": float(prec),
                f"{prefix}recall": float(rec),
                f"{prefix}f1": float(f1),
            }

        @staticmethod
        def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> Dict[str, float]:
            if len(y_true) == 0:
                return {f"{prefix}mae": 0.0, f"{prefix}rmse": 0.0, f"{prefix}r2": 0.0}
            mae = float(mean_absolute_error(y_true, y_pred))
            rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
            r2 = float(r2_score(y_true, y_pred))
            return {f"{prefix}mae": mae, f"{prefix}rmse": rmse, f"{prefix}r2": r2}

        # -----------------
        # Plotly reporting
        # -----------------
        def PlotHistory(self):
            if go is None or px is None:
                raise ImportError('Plotly is required for plotting methods. Install plotly to use this feature.')
            """
            Plot training and validation loss curves (Plotly).

            Returns
            -------
            plotly.graph_objects.Figure
                A line chart showing ``train_loss`` and ``val_loss`` over epochs.

            Notes
            -----
            Call :meth:`~topologicpy.PyG.PyG.Train` (or :meth:`~topologicpy.PyG.PyG.CrossValidate`)
            before plotting to populate :attr:`~topologicpy.PyG.PyG.history`.
            """
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=self.history["train_loss"], mode="lines+markers", name="Train Loss"))
            fig.add_trace(go.Scatter(y=self.history["val_loss"], mode="lines+markers", name="Val Loss"))
            fig.update_layout(title="Training History", xaxis_title="Epoch", yaxis_title="Loss")
            return fig

        def PlotConfusionMatrix(self,
                                split: str = "test",
                                normalize: bool = False,
                                title: str = None,
                                xTitle: str = "Actual Categories",
                                yTitle: str = "Predicted Categories",
                                width: int = 950,
                                height: int = 500,
                                showScale: bool = True,
                                colorScale: str = "viridis",
                                colorSamples: int = 10,
                                backgroundColor: str = 'rgba(0,0,0,0)',
                                marginLeft: int = 0,
                                marginRight: int = 0,
                                marginTop: int = 40,
                                marginBottom: int = 0,
                                minValue=None,
                                maxValue=None,
                                baseFontSize = 16,
                                tickFontSize = 14,
                                titleFontSize = 22,
                                axisTitleFontSize = 16,
                                annotationFontSize=18):
            """
            Plot a confusion matrix for classification tasks using TopologicPy's Plotly helper.

            Notes
            -----
            This method computes the confusion matrix (rows=Actual, cols=Predicted) and
            delegates plotting to ``Plotly.FigureByConfusionMatrix``. It then explicitly
            enforces both X and Y tick labels to ensure categories appear correctly.
            """
            if confusion_matrix is None:
                raise ImportError("scikit-learn is required. Install scikit-learn to use this feature.")

            try:
                from topologicpy.Plotly import Plotly
            except Exception as e:
                raise ImportError("topologicpy.Plotly is required to plot the confusion matrix.") from e

            import numpy as np

            if self.config.task != "classification" or self.config.level == "link":
                raise ValueError("PlotConfusionMatrix is only available for classification (graph/node/edge).")

            split_l = (split or "test").lower()
            if split_l in ("validate", "validation"):
                split_l = "val"

            def _y_for_one_split(one_split: str):
                if self.config.level == "graph":
                    if one_split == "train":
                        loader = DataLoader(self.train_set, batch_size=self.config.batch_size, shuffle=False)
                    elif one_split == "val":
                        loader = DataLoader(self.val_set, batch_size=self.config.batch_size, shuffle=False)
                    else:
                        loader = DataLoader(self.test_set, batch_size=self.config.batch_size, shuffle=False)
                    return self._predict_graph(loader)

                elif self.config.level == "node":
                    loader = DataLoader(self.data_list, batch_size=1, shuffle=False)
                    mask = "train_mask" if one_split == "train" else ("val_mask" if one_split == "val" else "test_mask")
                    return self._predict_node(loader, mask)

                else:  # edge
                    loader = DataLoader(self.data_list, batch_size=1, shuffle=False)
                    mask = "edge_train_mask" if one_split == "train" else ("edge_val_mask" if one_split == "val" else "edge_test_mask")
                    return self._predict_edge(loader, mask)

            # ---- Collect y_true/y_pred
            if split_l == "all":
                y_true_all, y_pred_all = [], []
                for s in ("train", "val", "test"):
                    yt, yp = _y_for_one_split(s)
                    if yt is not None and len(yt):
                        y_true_all.append(yt)
                        y_pred_all.append(yp)
                if not y_true_all:
                    raise RuntimeError("No labels found for confusion matrix (split='all').")
                y_true = np.concatenate(y_true_all, axis=0)
                y_pred = np.concatenate(y_pred_all, axis=0)
            else:
                if split_l not in ("train", "val", "test"):
                    raise ValueError(f"Unknown split '{split}'. Use train/val/test/all.")
                y_true, y_pred = _y_for_one_split(split_l)
                if y_true is None or len(y_true) == 0:
                    raise RuntimeError(f"No labels found for confusion matrix (split='{split_l}').")

            # ---- Ensure all categories appear
            labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
            labels_str = [str(c) for c in labels]

            cm = confusion_matrix(y_true, y_pred, labels=labels)

            if normalize:
                cm = cm.astype(float)
                cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)

            if title is None:
                title = f"Confusion Matrix ({split_l})"

            if minValue is None:
                minValue = 0.0
            if maxValue is None:
                maxValue = 1.0 if normalize else float(np.max(cm)) if cm.size else 1.0

            debug = cm.tolist()
            for i in debug:
                print(i)
            # ---- Delegate to TopologicPy Plotly
            fig = Plotly.FigureByConfusionMatrix(
                matrix=cm.tolist(),
                categories=labels_str,
                minValue=minValue,
                maxValue=maxValue,
                title=title,
                xTitle=xTitle,
                yTitle=yTitle,
                width=width,
                height=height,
                showScale=showScale,
                colorScale=colorScale,
                colorSamples=colorSamples,
                backgroundColor=backgroundColor,
                marginLeft=marginLeft,
                marginRight=marginRight,
                marginTop=marginTop,
                marginBottom=marginBottom,
                baseFontSize = baseFontSize,
                tickFontSize = tickFontSize,
                titleFontSize = titleFontSize,
                axisTitleFontSize = axisTitleFontSize,
                annotationFontSize = annotationFontSize
            )

            # ---- FIX: force BOTH axes ticks to show categories (prevents sequential Y labels)
            n = len(labels_str)
            tickvals = list(range(n))
            fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=labels_str)
            fig.update_yaxes(tickmode="array", tickvals=tickvals, ticktext=labels_str)

            return fig

        def PlotParity(self,
                    split: str = "test",
                    title: str = None,
                    show_identity: bool = True,
                    show_best_fit: bool = True,
                    point_size: int = 6):
            """
            Plot a parity (true vs predicted) plot for regression tasks (Plotly).

            Parameters
            ----------
            split : {"train", "val", "validate", "validation", "test", "all"}, optional
                Which split to evaluate.

                - "train"            : training split
                - "val"/"validate"   : validation split
                - "test"             : test split
                - "all"              : concatenates train+val+test

                Default is ``"test"``.

            title : str, optional
                Custom plot title. If None, an automatic title is generated.

            show_identity : bool, optional
                If True, draws the ``y=x`` reference line. Default is True.

            show_best_fit : bool, optional
                If True, draws a least-squares best-fit line (ŷ = a·y + b). Default is True.

            point_size : int, optional
                Marker size. Default is 6.

            Returns
            -------
            plotly.graph_objects.Figure
                Scatter plot of true vs predicted values with optional reference lines.

            Raises
            ------
            ValueError
                If called when ``config.task`` is not ``"regression"`` or when ``config.level`` is ``"link"``.
            RuntimeError
                If no regression labels are found for the requested split(s).

            Notes
            -----
            - For node/edge regression, the method uses the corresponding boolean masks on each
            graph and aggregates across all graphs.
            - This method relies on :meth:`_predict_graph`, :meth:`_predict_node`, and :meth:`_predict_edge`.
            """
            if go is None or px is None:
                raise ImportError("Plotly is required for plotting methods. Install plotly to use this feature.")

            import numpy as np

            if self.config.task != "regression" or self.config.level == "link":
                raise ValueError("Parity plot is only available for regression (graph/node/edge).")

            split_l = (split or "test").lower()
            if split_l in ("validate", "validation"):
                split_l = "val"

            def _y_for_one_split(one_split: str):
                if self.config.level == "graph":
                    if one_split == "train":
                        loader = DataLoader(self.train_set, batch_size=self.config.batch_size, shuffle=False)
                    elif one_split == "val":
                        loader = DataLoader(self.val_set, batch_size=self.config.batch_size, shuffle=False)
                    else:
                        loader = DataLoader(self.test_set, batch_size=self.config.batch_size, shuffle=False)
                    return self._predict_graph(loader)

                elif self.config.level == "node":
                    loader = DataLoader(self.data_list, batch_size=1, shuffle=False)
                    mask = "train_mask" if one_split == "train" else ("val_mask" if one_split == "val" else "test_mask")
                    return self._predict_node(loader, mask)

                else:  # edge
                    loader = DataLoader(self.data_list, batch_size=1, shuffle=False)
                    mask = "edge_train_mask" if one_split == "train" else ("edge_val_mask" if one_split == "val" else "edge_test_mask")
                    return self._predict_edge(loader, mask)

            # ---- Collect y_true/y_pred for split or splits
            if split_l == "all":
                y_true_all = []
                y_pred_all = []
                for s in ("train", "val", "test"):
                    yt, yp = _y_for_one_split(s)
                    if yt is not None and len(yt):
                        y_true_all.append(yt)
                        y_pred_all.append(yp)
                if not y_true_all:
                    raise RuntimeError("No labels found for parity plot (split='all').")
                y_true = np.concatenate(y_true_all, axis=0)
                y_pred = np.concatenate(y_pred_all, axis=0)
            else:
                if split_l not in ("train", "val", "test"):
                    raise ValueError(f"Unknown split '{split}'. Use train/val/test/all.")
                y_true, y_pred = _y_for_one_split(split_l)
                if y_true is None or len(y_true) == 0:
                    raise RuntimeError(f"No labels found for parity plot (split='{split_l}').")

            # ---- Build figure
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_true, y=y_pred,
                mode="markers",
                name="Predictions",
                marker=dict(size=int(point_size))
            ))

            mn = float(min(np.min(y_true), np.min(y_pred))) if len(y_true) else 0.0
            mx = float(max(np.max(y_true), np.max(y_pred))) if len(y_true) else 1.0

            if show_identity:
                fig.add_trace(go.Scatter(
                    x=[mn, mx], y=[mn, mx],
                    mode="lines",
                    name="Ideal (y=x)",
                    hoverinfo="skip"
                ))

            if show_best_fit and len(y_true) >= 2:
                a, b = np.polyfit(y_true.astype(float), y_pred.astype(float), 1)
                fig.add_trace(go.Scatter(
                    x=[mn, mx], y=[a * mn + b, a * mx + b],
                    mode="lines",
                    name=f"Best fit (y={a:.3g}x+{b:.3g})",
                    hoverinfo="skip"
                ))

            # lightweight metrics
            eps = 1e-12
            y_true_f = y_true.astype(float)
            y_pred_f = y_pred.astype(float)
            mae = float(np.mean(np.abs(y_pred_f - y_true_f)))
            rmse = float(np.sqrt(np.mean((y_pred_f - y_true_f) ** 2)))
            ss_res = float(np.sum((y_true_f - y_pred_f) ** 2))
            ss_tot = float(np.sum((y_true_f - np.mean(y_true_f)) ** 2))
            r2 = 1.0 - ss_res / (ss_tot + eps)

            if title is None:
                title = f"Parity Plot ({split_l})"

            fig.update_layout(
                title=f"{title} — MAE={mae:.4g}, RMSE={rmse:.4g}, R²={r2:.4g}",
                xaxis_title="True",
                yaxis_title="Predicted"
            )
            return fig

        def SaveModel(self, path: str, include_config: bool = True):
            """
            Save the model to disk.

            Parameters
            ----------
            path : str
                Output file path. If the extension is not ``.pt``, it is appended automatically.
            include_config : bool, optional
                If True, saves enough configuration alongside weights to rebuild the model on load.
                Default is True.

            Returns
            -------
            None
            """
            if not path.lower().endswith(".pt"):
                path = path + ".pt"

            if self.model is None:
                raise RuntimeError("PyG.SaveModel - Error: No model to save.")

            if include_config:
                # Save full config fields (not only Summary) so future options can be restored too.
                cfg = self.config
                cfg_fields = {k: getattr(cfg, k) for k in vars(cfg).keys()}
                payload = {
                    "state_dict": self.model.state_dict(),
                    "config_fields": cfg_fields,
                    "summary": self.Summary(),
                    # Schema required for safe inference on unseen datasets
                    "in_dim": int(self._in_dim) if getattr(self, "_in_dim", None) is not None else int(self.data_list[0].x.shape[1]) if self.data_list else None,
                    "num_outputs": int(self._num_outputs),
                    "feature_schema": self._current_feature_schema() if (self.graph_df is not None and self.nodes_df is not None and self.edges_df is not None) else None,
                }
                torch.save(payload, path)
            else:
                torch.save(self.model.state_dict(), path)


        def LoadModel(self, path: str, strict: bool = True, rebuild_from_checkpoint: bool = True):
            """
            Load model weights from disk.

            This method is backward compatible with older ``.pt`` files that contain only
            a raw ``state_dict``. If the file contains a checkpoint dict produced by
            :meth:`~topologicpy.PyG.PyG.SaveModel` with ``include_config=True``, the model
            can be rebuilt automatically to match the saved architecture.

            Parameters
            ----------
            path : str
                Path to a ``.pt`` file.
            strict : bool, optional
                Passed to ``load_state_dict``. Default is True.
            rebuild_from_checkpoint : bool, optional
                If True and the checkpoint contains saved config fields, rebuilds the model
                before loading weights. Default is True.

            Returns
            -------
            None
            """
            if self.model is None:
                # In case someone creates PyG without model building (unlikely), build now.
                self._build_model()

            # Try safe loading first (future default behavior)
            try:
                obj = torch.load(path, map_location=self.device, weights_only=True)
            except TypeError:
                # Older PyTorch version (no weights_only argument)
                obj = torch.load(path, map_location=self.device)


            # New format: dict checkpoint
            if isinstance(obj, dict) and ("state_dict" in obj):
                state = obj["state_dict"]

                if rebuild_from_checkpoint and ("config_fields" in obj) and isinstance(obj["config_fields"], dict):
                    # Restore config fields that affect model shape first
                    # (conv/hidden_dims/activation/dropout/batch_norm/residual/pooling)
                    cfg_fields = obj["config_fields"]

                    model_keys = {"conv", "hidden_dims", "activation", "dropout", "batch_norm", "residual", "pooling",
                                "level", "task", "graph_label_type", "node_label_type", "edge_label_type"}

                    # Apply only fields that exist on this version of _RunConfig
                    for k, v in cfg_fields.items():
                        if hasattr(self.config, k) and (k in model_keys):
                            setattr(self.config, k, v)

                    # IMPORTANT: rebuild model to match checkpoint architecture

                    # Restore training-time schema if present (keeps feature dims + head dims stable).
                    if "in_dim" in obj and obj.get("in_dim", None) is not None:
                        self._in_dim = int(obj["in_dim"])
                    if "num_outputs" in obj and obj.get("num_outputs", None) is not None:
                        self._num_outputs = int(obj["num_outputs"])
                        self._freeze_num_outputs = True
                    if "feature_schema" in obj and isinstance(obj.get("feature_schema"), dict):
                        self._feature_schema = obj.get("feature_schema")

                    # If we already have CSVs loaded (common when doing: ByCSVPath(unseen) -> LoadModel()),
                    # rebuild data_list using the saved schema so tensors match the trained model.
                    if (self.graph_df is not None) and (self.nodes_df is not None) and (self.edges_df is not None):
                        self.data_list = []
                        self._build_data_list()
                        self._split_holdout()

                    self._build_model()

            # Old format: state_dict only
            else:
                state = obj

            
            # Restore training-time schema if present even if we did not rebuild from checkpoint
            # (useful when predicting on unseen datasets).
            if isinstance(obj, dict):
                if ("in_dim" in obj) and (obj.get("in_dim", None) is not None):
                    self._in_dim = int(obj["in_dim"])
                if ("num_outputs" in obj) and (obj.get("num_outputs", None) is not None):
                    self._num_outputs = int(obj["num_outputs"])
                    self._freeze_num_outputs = True
                if ("feature_schema" in obj) and isinstance(obj.get("feature_schema"), dict):
                    self._feature_schema = obj.get("feature_schema")

                # Align already-loaded CSVs to the schema (if any) so prediction tensors match.
                if self._feature_schema and (self.graph_df is not None) and (self.nodes_df is not None) and (self.edges_df is not None):
                    self.data_list = []
                    self._build_data_list()
                    self._split_holdout()
# Finally load weights
            self.model.load_state_dict(state, strict=strict)
            self.model.to(self.device)
            self.model.eval()

else:
    class PyG:
        """PyTorch Geometric (PyG) helper.

        This class requires optional dependencies (PyTorch + PyTorch-Geometric).
        The module remains importable without them so documentation can build.
        """

        @staticmethod
        def ByCSVPath(*args, **kwargs):
            raise ImportError(
                "topologicpy.PyG requires optional dependencies. Install torch + torch_geometric "
                "(and related packages) to use this class."
            ) from _PYG_IMPORT_ERROR

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "topologicpy.PyG requires optional dependencies. Install torch + torch_geometric "
                "(and related packages) to use this class."
            ) from _PYG_IMPORT_ERROR
