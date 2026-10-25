# Copyright (C) 2026
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free Software
# Foundation, either version 3.0 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

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

- Only configured feat_* columns are numeric model inputs. Coordinate columns
  x/y/z are not implicitly added to node features. Edge attributes condition
  message passing and edge predictions. Graph attributes condition node embeddings
  for every task and are also concatenated with pooled graph embeddings.
- Ontology metadata columns are optional and are preserved as metadata, not used as
  numeric model features unless the user explicitly encodes them as feat_* columns.
  Recommended metadata columns include:
    ontology_class, category, uri, ifc_class, ifc_guid, source, derived_from, generated_by.

Notes
-----
- This module intentionally avoids auto-install behaviour.
- It aims to be easy to read and modify by non-ML experts.

"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Literal

import os
import math
import random
import copy
import hashlib
import json
import tempfile
from contextlib import contextmanager

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

    target_columns: Tuple[str, ...] = ()
    target_units: Tuple[str, ...] = ()
    split_mode: str = "entity"  # "graph" holds out complete simulations
    normalize_features: bool = False
    normalize_targets: bool = False
    deterministic: bool = False
    model_name: str = "default"
    model_options: Optional[dict] = None
    position_columns: Tuple[str, ...] = ("x", "y", "z")
    node_weight_column: Optional[str] = None
    edge_weight_column: Optional[str] = None
    boundary_column: Optional[str] = None
    time_column: str = "time"
    time_step_column: str = "dt"
    inference: bool = False
    optional_feature_columns: Tuple[str, ...] = ()
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

    # ----------------------------
    # Ontology / semantic metadata
    # ----------------------------
    ontology: bool = True
    ontology_class_header: str = "ontology_class"
    category_header: str = "category"
    uri_header: str = "uri"
    ifc_class_header: str = "ifc_class"
    ifc_guid_header: str = "ifc_guid"
    source_header: str = "source"
    derived_from_header: str = "derived_from"
    generated_by_header: str = "generated_by"

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
    holdout_group_by: Optional[str] = None              # graph_df column name for group-aware graph-level holdout

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
    activation: Literal["relu", "gelu", "elu", "tanh"] = "relu"
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
        Feature-aware GNN encoder. Graph context conditions node inputs.
        GATv2 uses native edge attention; GraphSAGE/GCN add an averaged learned
        edge-conditioned message residual. With no extra features the original
        node-only architecture and parameter names are preserved.
        """

        def __init__(self,
                     in_dim: int,
                     hidden_dims: Tuple[int, ...],
                     conv: ConvKind = "sage",
                     activation: str = "relu",
                     dropout: float = 0.1,
                     batch_norm: bool = False,
                     residual: bool = False,
                     edge_dim: int = 0,
                     graph_dim: int = 0):
            super().__init__()
            self.edge_dim = int(edge_dim)
            self.graph_dim = int(graph_dim)
            self.conv_kind = conv
            if self.graph_dim:
                self.graph_projection = nn.Linear(self.graph_dim, in_dim)
            if self.edge_dim and conv != "gatv2":
                self.edge_messages = nn.ModuleList()
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
            elif activation == "tanh":
                self.act = torch.tanh
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
                    self.convs.append(GATv2Conv(in_ch, out_ch, heads=1, concat=False,
                                              edge_dim=self.edge_dim or None))
                else:
                    raise ValueError(f"Unsupported conv='{conv}'.")

                if self.edge_dim and conv != "gatv2":
                    self.edge_messages.append(nn.Sequential(
                        nn.Linear(in_ch + self.edge_dim, out_ch), nn.ReLU(),
                        nn.Linear(out_ch, out_ch)))
                if self.use_bn:
                    self.bns.append(nn.BatchNorm1d(out_ch))

            self.out_dim = dims[-1]

        def forward(self, x, edge_index, edge_attr=None, u=None, batch=None):
            h = x
            if self.graph_dim:
                if u is None or u.ndim != 2 or u.shape[1] != self.graph_dim:
                    raise ValueError("Graph features must have shape [num_graphs, graph_dim].")
                if batch is None:
                    batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
                h = h + self.graph_projection(u)[batch]
            if self.edge_dim and (edge_attr is None or edge_attr.shape != (edge_index.shape[1], self.edge_dim)):
                raise ValueError("Edge features must align with edge_index and the trained feature dimension.")
            for i, conv in enumerate(self.convs):
                h_in = h
                if self.edge_dim and self.conv_kind == "gatv2":
                    h = conv(h, edge_index, edge_attr=edge_attr)
                else:
                    h = conv(h, edge_index)
                    if self.edge_dim:
                        src, dst = edge_index
                        messages = self.edge_messages[i](torch.cat((h_in[src], edge_attr), dim=-1))
                        aggregate = h.new_zeros(h.shape)
                        aggregate.index_add_(0, dst, messages)
                        degree = torch.bincount(dst, minlength=h.shape[0]).clamp_min(1).to(h.dtype)
                        h = h + aggregate / degree.unsqueeze(-1)
                if self.use_bn:
                    h = self.bns[i](h)
                h = self.act(h)

                if self.use_residual and h_in.shape == h.shape:
                    h = h + h_in

                h = F.dropout(h, p=self.dropout, training=self.training)
            return h


    class _GraphHead(nn.Module):
        def __init__(self, in_dim: int, out_dim: int, pooling: PoolingKind = "mean", dropout: float = 0.1, graph_dim: int = 0):
            super().__init__()
            self.graph_dim = int(graph_dim)
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
                nn.Linear(in_dim + self.graph_dim, in_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(in_dim, out_dim),
            )

        def forward(self, node_emb, batch, u=None):
            g = self.pool(node_emb, batch)
            if self.graph_dim:
                if u is None or u.shape != (g.shape[0], self.graph_dim):
                    raise ValueError("Graph feature rows must align with pooled graphs.")
                g = torch.cat((g, u), dim=-1)
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
        Edge prediction head using endpoint embeddings and optional edge attributes.
        """
        def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1, edge_dim: int = 0):
            super().__init__()
            self.edge_dim = int(edge_dim)
            self.mlp = nn.Sequential(
                nn.Linear(in_dim * 2 + self.edge_dim, in_dim),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(in_dim, out_dim),
            )

        def forward(self, node_emb, edge_index, edge_attr=None):
            src, dst = edge_index[0], edge_index[1]
            h = torch.cat([node_emb[src], node_emb[dst]], dim=-1)
            if self.edge_dim:
                if edge_attr is None or edge_attr.shape != (h.shape[0], self.edge_dim):
                    raise ValueError("Edge prediction attributes must align with predicted edges.")
                h = torch.cat((h, edge_attr), dim=-1)
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
            return self.net(h).reshape(-1)  # logits


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

        @staticmethod
        def _MSAMLP(inputs, outputs, hidden):
            return nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, outputs), nn.LayerNorm(outputs))

        class _MSAGraphBlock(nn.Module):
            def __init__(self, width):
                super().__init__()
                self.edge = PyG._MSAMLP(3 * width, width, width)
                self.node = PyG._MSAMLP(2 * width, width, width)
            def forward(self, nodes, edges, index):
                src, dst = index
                edges = edges + self.edge(torch.cat((nodes[src], nodes[dst], edges), -1))
                aggregate = nodes.new_zeros(nodes.shape)
                aggregate.index_add_(0, dst, edges)
                count = torch.bincount(dst, minlength=len(nodes)).clamp_min(1).to(nodes.dtype)
                nodes = nodes + self.node(torch.cat((nodes, aggregate / count[:, None]), -1))
                return nodes, edges

        class _MSAProcessor(nn.Module):
            def __init__(self, width, heads, dropout, attention):
                super().__init__()
                self.fine = PyG._MSAGraphBlock(width)
                self.coarse = PyG._MSAGraphBlock(width)
                self.attention_enabled = attention
                if attention:
                    self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
                    self.attention_norm = nn.LayerNorm(width)
                    self.ffn = nn.Sequential(nn.Linear(width, 2 * width), nn.ReLU(), nn.Dropout(dropout), nn.Linear(2 * width, width))
                    self.ffn_norm = nn.LayerNorm(width)
                self.fuse = PyG._MSAMLP(2 * width, width, width)

        class _MSAModel(nn.Module):
            """Velocity-increment surrogate with fine/coarse graph processing.

            A single scene is processed per batch. Coarse attention is masked between
            disconnected air regions; fine reconstruction retains a learned skip path.
            """
            def __init__(self, context):
                super().__init__()
                options = context["options"]
                width, blocks, heads = options.get("latent_dim", 128), options.get("processor_blocks", 4), options.get("attention_heads", 2)
                self.multiscale = options.get("multiscale", True)
                self.pooling = options.get("aggregation", "gru")
                self.graph_dim, self.edge_dim = context["graph_dim"], context["edge_dim"]
                self.pos_dim = context["position_dim"]
                if self.pos_dim not in (2, 3): raise ValueError("MSA-GNN needs 2D/3D positions.")
                self.node_encoder = PyG._MSAMLP(context["node_dim"] + self.pos_dim + self.graph_dim + 1, width, width)
                self.edge_encoder = PyG._MSAMLP(self.edge_dim + self.pos_dim + 1, width, width)
                self.coarse_edge_encoder = PyG._MSAMLP(self.pos_dim + 1, width, width)
                if self.pooling == "gru": self.group_gru = nn.GRU(width, width, batch_first=True)
                self.group_projection = PyG._MSAMLP(width,width,width)
                self.processors = nn.ModuleList([PyG._MSAProcessor(width, heads, options.get("dropout", .1), options.get("attention", True)) for _ in range(blocks)])
                self.decoder = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width,width), nn.ReLU(), nn.Linear(width, context["output_dim"]))
            def _aggregate(self, nodes, hierarchy):
                groups = hierarchy["cluster_index"]
                k = len(hierarchy["coarse_pos"])
                if self.pooling == "mean":
                    result = nodes.new_zeros((k,nodes.shape[1]))
                    result.index_add_(0,groups,nodes)
                    return self.group_projection(result / torch.bincount(groups,minlength=k).clamp_min(1)[:,None])
                order,ptr = hierarchy["member_index"],hierarchy["group_ptr"]
                lengths = ptr[1:]-ptr[:-1]
                sorting = lengths.argsort(descending=True,stable=True)
                times = torch.arange(int(lengths.max()),device=nodes.device)[:,None]
                valid = times < lengths[sorting][None,:]
                offsets = (ptr[sorting][None,:]+times)[valid]
                # Pack only actual member vectors; avoid K*max_group*H feature padding.
                packed = nn.utils.rnn.PackedSequence(nodes[order[offsets]],valid.sum(1).cpu(),sorting,sorting.argsort())
                _,hidden = self.group_gru(packed)
                return self.group_projection(hidden[0])
            def forward(self, data):
                if getattr(data, "num_graphs", 1) != 1:
                    raise ValueError("MSA-GNN processes one scene per batch to isolate attention and coarse indices.")
                if getattr(data, "pos", None) is None: raise ValueError("MSA-GNN requires physical node positions.")
                dt = getattr(data, "dt", None)
                if dt is None or dt.numel() != 1 or not bool((dt > 0).all()):
                    raise ValueError("MSA-GNN requires a positive prediction time step.")
                pos = data.pos
                scale = (pos.max(0).values - pos.min(0).values).norm().clamp_min(1e-8)
                centered = (pos - pos.mean(0)) / scale
                parts = [data.x, centered, dt.reshape(1, 1).expand(len(pos), 1)]
                if self.graph_dim: parts.append(data.u.expand(len(pos), -1))
                nodes = self.node_encoder(torch.cat(parts, -1))
                src, dst = data.edge_index
                relative = (pos[src] - pos[dst]) / scale
                geometry = torch.cat((relative, relative.norm(dim=-1, keepdim=True)), -1)
                if self.edge_dim: geometry = torch.cat((geometry, data.edge_attr), -1)
                edges = self.edge_encoder(geometry)
                if self.multiscale:
                    hierarchy = data.hierarchy
                    coarse_pos, coarse_index = hierarchy["coarse_pos"], hierarchy["coarse_edge_index"]
                    a, b = coarse_index
                    rel = (coarse_pos[a] - coarse_pos[b]) / scale
                    coarse_edges = self.coarse_edge_encoder(torch.cat((rel, rel.norm(dim=-1, keepdim=True)), -1))
                    components = hierarchy["coarse_component"]
                    mask = components[:, None] != components[None, :]
                for block in self.processors:
                    nodes, edges = block.fine(nodes, edges, data.edge_index)
                    if not self.multiscale: continue
                    coarse = self._aggregate(nodes, hierarchy)
                    coarse, coarse_edges = block.coarse(coarse, coarse_edges, coarse_index)
                    if block.attention_enabled:
                        attended, _ = block.attention(coarse[None], coarse[None], coarse[None], attn_mask=mask, need_weights=False)
                        coarse = block.attention_norm(coarse + attended[0])
                        coarse = block.ffn_norm(coarse + block.ffn(coarse))
                    nodes = nodes + block.fuse(torch.cat((nodes, coarse[hierarchy["cluster_index"]]), -1))
                return {"prediction": self.decoder(nodes), "embedding": nodes}

        @staticmethod
        def BuildMSAHierarchy(positions, edgeIndex, nClusters=None, ratio=.1, maxClusters=128, seed=42):
            """Build deterministic sparse, barrier-aware spectral graph coarsening.

            Spatially weighted mesh adjacency defines affinity. Spectral seeds are
            expanded by graph shortest paths, guaranteeing connected clusters without
            crossing disconnected air regions. No dense fine-mesh affinity is created.
            """
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import connected_components, laplacian, dijkstra
            from scipy.sparse.linalg import eigsh
            from sklearn.cluster import KMeans
            pos = np.asarray(positions.detach().cpu() if torch.is_tensor(positions) else positions, dtype=float)
            index = np.asarray(edgeIndex.detach().cpu() if torch.is_tensor(edgeIndex) else edgeIndex)
            if pos.ndim != 2 or pos.shape[1] not in (2, 3) or not len(pos) or not np.isfinite(pos).all():
                raise ValueError("positions must be finite [nodes, 2 or 3].")
            if index.ndim != 2 or index.shape[0] != 2 or not np.issubdtype(index.dtype, np.integer):
                raise ValueError("edgeIndex must be integer [2, edges].")
            n = len(pos)
            if index.size and (index.min() < 0 or index.max() >= n): raise ValueError("Mesh edge endpoint out of range.")
            if not isinstance(maxClusters, int) or maxClusters < 1 or not 0 < ratio <= 1:
                raise ValueError("maxClusters must be positive and ratio in (0,1].")
            if not isinstance(seed, int) or not 0 <= seed < 2**32: raise ValueError("Invalid hierarchy seed.")
            if nClusters is not None and (not isinstance(nClusters, int) or nClusters < 1 or nClusters > min(n, maxClusters)):
                raise ValueError("nClusters must be positive and within mesh/maxClusters limits.")
            order = np.lexsort(tuple(pos[:, i] for i in reversed(range(pos.shape[1]))))
            inverse = np.empty(n, dtype=int); inverse[order] = np.arange(n)
            points, pairs = pos[order], inverse[index]
            src, dst = pairs
            nonself = src != dst
            src, dst = src[nonself], dst[nonself]
            distances = np.linalg.norm(points[src] - points[dst], axis=1)
            sigma = np.median(distances[distances > 0]) if (distances > 0).any() else 1.
            weights = np.maximum(np.exp(-.5 * (distances / sigma) ** 2), 1e-12)
            affinity = coo_matrix((np.r_[weights, weights], (np.r_[src, dst], np.r_[dst, src])), shape=(n, n)).tocsr()
            # Duplicate directed mesh records should not alter affinity.
            counts = coo_matrix((np.ones(2 * len(src)), (np.r_[src, dst], np.r_[dst, src])), shape=(n,n)).tocsr()
            if affinity.nnz: affinity.data /= counts.data
            nc, component = connected_components(affinity, directed=False)
            k = nClusters if nClusters is not None else min(maxClusters, max(nc, int(math.ceil(ratio * n))))
            if nc > maxClusters or k < nc: raise ValueError("Cluster budget cannot merge disconnected air regions; increase maxClusters/nClusters.")
            sizes = np.bincount(component)
            allocation = np.ones(nc, dtype=int)
            for _ in range(k - nc):
                candidates = np.where(allocation < sizes)[0]
                winner = candidates[np.argmax(sizes[candidates] / allocation[candidates])]
                allocation[winner] += 1
            labels = np.empty(n, dtype=int); offset = 0
            rng = np.random.default_rng(seed)
            for c in range(nc):
                members = np.flatnonzero(component == c); count = int(allocation[c])
                if count == len(members):
                    local_labels = np.arange(count)
                elif count == 1:
                    local_labels = np.zeros(len(members), dtype=int)
                else:
                    matrix = affinity[members][:, members]
                    L = laplacian(matrix, normed=True)
                    _, embedding = eigsh(L, k=count, which="SM", v0=rng.normal(size=len(members)))
                    embedding /= np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-12)
                    fit = KMeans(count, n_init=10, random_state=seed).fit(embedding)
                    seeds = []
                    for center in fit.cluster_centers_:
                        ranked = np.argsort(np.linalg.norm(embedding - center, axis=1), kind="stable")
                        seeds.append(next(int(v) for v in ranked if int(v) not in seeds))
                    costs = matrix.copy(); costs.data = 1. / costs.data
                    _, _, sources = dijkstra(costs, directed=False, indices=seeds, min_only=True, return_predecessors=True)
                    seed_map = {v: j for j, v in enumerate(seeds)}
                    local_labels = np.asarray([seed_map[int(v)] for v in sources])
                labels[members] = local_labels + offset
                offset += count
            original_labels = np.empty(n, dtype=int); original_labels[order] = labels
            cluster_positions = np.stack([pos[original_labels == i].mean(0) for i in range(k)])
            coarse_components = np.asarray([component[np.flatnonzero(labels == i)[0]] for i in range(k)])
            a, b = original_labels[index[0]], original_labels[index[1]]
            coarse_pairs = sorted({(int(x), int(y)) for x, y in zip(a, b) if x != y} | {(int(y), int(x)) for x, y in zip(a, b) if x != y})
            coarse_index = np.asarray(coarse_pairs, dtype=np.int64).T if coarse_pairs else np.empty((2, 0), dtype=np.int64)
            members = np.concatenate([order[labels == i] for i in range(k)])
            ptr = np.r_[0, np.cumsum(np.bincount(original_labels, minlength=k))]
            return {"cluster_index": torch.tensor(original_labels, dtype=torch.long),
                    "coarse_edge_index": torch.tensor(coarse_index, dtype=torch.long),
                    "coarse_pos": torch.tensor(cluster_positions, dtype=torch.float32),
                    "coarse_component": torch.tensor(coarse_components, dtype=torch.long),
                    "member_index": torch.tensor(members, dtype=torch.long),
                    "group_ptr": torch.tensor(ptr, dtype=torch.long)}

        def _PrepareMSAHierarchies(self):
            if self.config.model_name != "msa_gnn": return
            options = self.config.model_options or {}
            cache = getattr(self, "_msa_hierarchy_cache", {})
            auto = getattr(self, "_msa_auto_hierarchy", {})
            for i, data in enumerate(self.data_list):
                if getattr(data, "pos", None) is None or getattr(data, "dt", None) is None:
                    raise ValueError("MSA-GNN needs positions and time steps on every graph.")
                if not options.get("multiscale", True): continue
                digest = hashlib.sha256(data.pos.numpy().tobytes() + data.edge_index.numpy().tobytes()).hexdigest()
                settings = {"nClusters": options.get("n_clusters"), "ratio": options.get("coarsening_ratio", .1),
                            "maxClusters": options.get("max_coarse_nodes", 128), "seed": self.config.random_state}
                key = digest + json.dumps(settings, sort_keys=True)
                graph_id = self._label_key(self._identities[i]["graph_id"])
                if getattr(data, "hierarchy", None) is None or (graph_id in auto and auto[graph_id] != key):
                    if key not in cache: cache[key] = self.BuildMSAHierarchy(data.pos, data.edge_index, **settings)
                    data.hierarchy = cache[key]
                    auto[graph_id] = key
                self._ValidateMSAHierarchy(data)
                graph_id = self._identities[i]["graph_id"]
                self._external_simulation.setdefault(self._label_key(graph_id), {})["hierarchy"] = data.hierarchy
            self._msa_hierarchy_cache = cache
            self._msa_auto_hierarchy = auto

        @staticmethod
        def _ValidateMSAHierarchy(data):
            h = data.hierarchy
            keys = ("cluster_index", "coarse_edge_index", "coarse_pos", "coarse_component", "member_index", "group_ptr")
            if not all(k in h and torch.is_tensor(h[k]) for k in keys): raise ValueError("Incomplete MSA hierarchy.")
            n, k = data.num_nodes, len(h["coarse_pos"])
            labels, members, ptr = h["cluster_index"], h["member_index"], h["group_ptr"]
            if k < 1 or labels.dtype != torch.long or labels.shape != (n,) or labels.min() < 0 or labels.max() >= k:
                raise ValueError("Invalid MSA cluster assignments.")
            if members.dtype != torch.long or members.shape != (n,) or not torch.equal(members.sort().values, torch.arange(n, device=members.device)):
                raise ValueError("MSA member order must be a node permutation.")
            if ptr.dtype != torch.long or ptr.shape != (k+1,) or ptr[0] != 0 or ptr[-1] != n or not bool((ptr[1:] > ptr[:-1]).all()):
                raise ValueError("Invalid MSA group pointers.")
            for group in range(k):
                if not bool((labels[members[ptr[group]:ptr[group+1]]] == group).all()): raise ValueError("MSA hierarchy membership mismatch.")
            edges = h["coarse_edge_index"]
            if edges.dtype != torch.long or edges.ndim != 2 or edges.shape[0] != 2 or (edges.numel() and (edges.min() < 0 or edges.max() >= k)):
                raise ValueError("Invalid coarse edges.")
            if h["coarse_pos"].shape != (k, data.pos.shape[1]) or not torch.isfinite(h["coarse_pos"]).all() or h["coarse_component"].shape != (k,) or h["coarse_component"].dtype != torch.long or (h["coarse_component"] < 0).any():
                raise ValueError("Invalid coarse geometry/component data.")

        @staticmethod
        def _CFDArray(value, name):
            array = np.asarray(value)
            if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
                raise ValueError(f"CFD {name} must contain finite numeric values.")
            return array

        @staticmethod
        def ReadCFDSequence(paths, velocityKey="U", location="point", boundaryKey=None, weightKey=None, timeStep=.1):
            """Read a safe NPZ trajectory or fixed-mesh VTK/VTU files via optional meshio.

            NPZ keys: positions[N,2/3], edge_index[2,E], velocity[T,N,2/3], with optional
            times[T], boundary_type[N], weights[N], graph_features[T,G] and
            extra_node_features[N,F]. VTK data can be point- or cell-centred; mesh identity
            and node/cell ordering must remain fixed across the supplied files.
            """
            if isinstance(paths, (str, os.PathLike)):
                if str(paths).lower().endswith(".npz"):
                    with np.load(paths, allow_pickle=False) as data:
                        required = ("positions", "edge_index", "velocity")
                        if any(k not in data for k in required): raise ValueError("NPZ is missing positions/edge_index/velocity.")
                        return {k: data[k].copy() for k in data.files}
                paths = [paths]
            if not paths: raise ValueError("Provide CFD frame paths.")
            if location not in ("point", "cell"): raise ValueError("location must be point or cell.")
            if not math.isfinite(timeStep) or timeStep <= 0: raise ValueError("timeStep must be positive.")
            try:
                import meshio
            except ImportError as error:
                raise ImportError("VTK/VTU import requires meshio; install it explicitly.") from error
            frames, first, positions, index, tags, weights = [], None, None, None, None, None
            for path in paths:
                mesh = meshio.read(path)
                signature = (np.asarray(mesh.points), [(block.type, np.asarray(block.data)) for block in mesh.cells])
                if first is None:
                    first = signature
                    positions, index, blocks = PyG._CFDMeshGraph(mesh, location)
                    if location == "point":
                        tags = mesh.point_data.get(boundaryKey) if boundaryKey else None
                        weights = mesh.point_data.get(weightKey) if weightKey else None
                    else:
                        if boundaryKey: tags = PyG._CFDCellField(mesh, boundaryKey, blocks)
                        if weightKey: weights = PyG._CFDCellField(mesh, weightKey, blocks)
                else:
                    if not np.array_equal(signature[0], first[0]) or len(signature[1]) != len(first[1]) or any(a[0] != b[0] or not np.array_equal(a[1], b[1]) for a,b in zip(signature[1], first[1])):
                        raise ValueError("CFD frame mesh/order changed; remeshing requires explicit field transfer.")
                if location == "point":
                    if velocityKey not in mesh.point_data: raise ValueError(f"Missing point velocity field {velocityKey!r}.")
                    velocity = mesh.point_data[velocityKey]
                else:
                    velocity = PyG._CFDCellField(mesh, velocityKey, blocks)
                frames.append(np.asarray(velocity))
                # Boundary classification and integration weights are static mesh data.
                if location == "point":
                    current_tags = mesh.point_data.get(boundaryKey) if boundaryKey else None
                    current_weights = mesh.point_data.get(weightKey) if weightKey else None
                else:
                    current_tags = PyG._CFDCellField(mesh, boundaryKey, blocks) if boundaryKey else None
                    current_weights = PyG._CFDCellField(mesh, weightKey, blocks) if weightKey else None
                if boundaryKey and (tags is None or not np.array_equal(tags, current_tags)): raise ValueError("Boundary classification missing or changed across frames.")
                if weightKey and (weights is None or not np.array_equal(weights, current_weights)): raise ValueError("Integration weights missing or changed across frames.")
            result = {"positions": positions, "edge_index": index, "velocity": np.stack(frames),
                      "times": np.arange(len(frames)) * timeStep}
            if tags is not None: result["boundary_type"] = np.asarray(tags).reshape(-1)
            if weights is not None: result["weights"] = np.asarray(weights).reshape(-1)
            return result

        @staticmethod
        def _CFDCellField(mesh, key, blocks):
            if key not in mesh.cell_data: raise ValueError(f"Missing cell field {key!r}.")
            return np.concatenate([np.asarray(mesh.cell_data[key][i]) for i in blocks], axis=0)

        @staticmethod
        def _CFDMeshGraph(mesh, location):
            """Build vertex edges or face-sharing cell adjacency for standard linear cells."""
            shapes = {
                "line": (1, [(0,), (1,)], [(0,1)]),
                "triangle": (2, [(0,1),(1,2),(2,0)], [(0,1),(1,2),(2,0)]),
                "quad": (2, [(0,1),(1,2),(2,3),(3,0)], [(0,1),(1,2),(2,3),(3,0)]),
                "tetra": (3, [(0,1,2),(0,1,3),(0,2,3),(1,2,3)], [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]),
                "hexahedron": (3, [(0,1,2,3),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)], [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]),
                "wedge": (3, [(0,1,2),(3,4,5),(0,1,4,3),(1,2,5,4),(2,0,3,5)], [(0,1),(1,2),(2,0),(3,4),(4,5),(5,3),(0,3),(1,4),(2,5)]),
                "pyramid": (3, [(0,1,2,3),(0,1,4),(1,2,4),(2,3,4),(3,0,4)], [(0,1),(1,2),(2,3),(3,0),(0,4),(1,4),(2,4),(3,4)])}
            if not mesh.cells or any(block.type not in shapes for block in mesh.cells):
                raise ValueError("Supported CFD cells: line, triangle, quad, tetra, hexahedron, wedge and pyramid.")
            dimension = max(shapes[b.type][0] for b in mesh.cells)
            blocks = [i for i,b in enumerate(mesh.cells) if shapes[b.type][0] == dimension]
            pairs, centers, faces = set(), [], {}
            for block_id in blocks:
                block = mesh.cells[block_id]
                for cell in block.data:
                    if location == "point":
                        for a,b in shapes[block.type][2]: pairs.add(tuple(sorted((int(cell[a]),int(cell[b])))))
                    else:
                        cell_id = len(centers)
                        centers.append(mesh.points[cell].mean(0))
                        for face in shapes[block.type][1]:
                            key = tuple(sorted(int(cell[j]) for j in face))
                            if key in faces:
                                if len(faces[key]) >= 2: raise ValueError("Non-manifold CFD cell face.")
                                pairs.add((faces[key][0], cell_id)); faces[key].append(cell_id)
                            else: faces[key] = [cell_id]
            directed = sorted(pairs | {(b,a) for a,b in pairs})
            index = np.asarray(directed, dtype=np.int64).T if directed else np.empty((2,0),dtype=np.int64)
            return (np.asarray(mesh.points) if location == "point" else np.asarray(centers)), index, blocks

        @staticmethod
        def ByCFDPaths(path, sequences, velocityKey="U", location="point", boundaryKey=None, weightKey=None, timeStep=.1, **kwargs):
            """Import independent CFD file sequences and construct a trainable dataset."""
            values = [PyG.ReadCFDSequence(files, velocityKey, location, boundaryKey, weightKey, timeStep) for files in sequences]
            return PyG.ByCFDSequences(path, values, timeStep=timeStep, **kwargs)

        @staticmethod
        def ByCFDSequences(path, sequences, timeStep=.1, overwrite=False, **kwargs):
            """Write/load fixed-mesh transient velocity trajectories as increment targets.

            Each sequence is a dictionary or consolidated NPZ path. Boundary types are
            0=interior, 1=prescribed wall, 2=prescribed inlet, 3=outlet. Only types 1/2
            are enforced during rollout. All input fields use physical units.
            """
            if not sequences or not math.isfinite(timeStep) or timeStep <= 0: raise ValueError("Provide sequences and a positive timeStep.")
            sequences = [PyG.ReadCFDSequence(v) if isinstance(v,(str,os.PathLike)) else v for v in sequences]
            graphs, nodes, edges, dimensions, timesteps = [], [], [], None, []
            feature_width, extra_width, graph_names = None, None, None
            for sequence_id, sequence in enumerate(sequences):
                if not isinstance(sequence, dict): raise ValueError("Each CFD sequence must be a dictionary or NPZ path.")
                positions = PyG._CFDArray(sequence["positions"], "positions").astype(float)
                velocity = PyG._CFDArray(sequence["velocity"], "velocity").astype(float)
                index = np.asarray(sequence["edge_index"])
                if positions.ndim != 2 or positions.shape[1] not in (2,3) or not len(positions): raise ValueError("Invalid CFD positions.")
                n = len(positions)
                if positions.shape[1] == 2: positions = np.c_[positions, np.zeros(n)]
                if velocity.ndim != 3 or velocity.shape[0] < 2 or velocity.shape[1] != n or velocity.shape[2] not in (2,3):
                    raise ValueError("velocity must have shape [at least 2 frames, nodes, 2/3].")
                t, _, d = velocity.shape
                if dimensions is not None and dimensions != d: raise ValueError("All sequences need the same velocity dimension.")
                dimensions = d
                if index.ndim != 2 or index.shape[0] != 2 or not np.issubdtype(index.dtype, np.integer) or (index.size and (index.min()<0 or index.max()>=n)):
                    raise ValueError("Invalid CFD edge_index.")
                pairs = sorted({(int(a),int(b)) for a,b in index.T if a != b} | {(int(b),int(a)) for a,b in index.T if a != b})
                times = PyG._CFDArray(sequence.get("times", np.arange(t)*timeStep), "times")
                if times.shape != (t,) or not (np.diff(times)>0).all(): raise ValueError("CFD times must increase strictly.")
                tags = np.asarray(sequence.get("boundary_type", np.zeros(n,dtype=int)))
                if tags.shape != (n,) or not np.isin(tags, (0,1,2,3)).all(): raise ValueError("boundary_type must use 0/1/2/3.")
                tags = tags.astype(int)
                weights = PyG._CFDArray(sequence.get("weights", np.ones(n)), "weights")
                if weights.shape != (n,) or (weights < 0).any() or weights.sum()<=0: raise ValueError("CFD weights must be nonnegative with positive sum.")
                extra = PyG._CFDArray(sequence.get("extra_node_features", np.empty((n,0))), "extra_node_features")
                if extra.ndim != 2 or extra.shape[0] != n: raise ValueError("extra_node_features must be static [nodes, features].")
                global_features = sequence.get("graph_features", np.empty((t,0)))
                names = sorted(global_features) if isinstance(global_features, dict) else None
                if isinstance(global_features, dict):
                    columns = []
                    for key in sorted(global_features):
                        value = np.asarray(global_features[key])
                        columns.append(np.full(t, value.item()) if value.ndim == 0 else value)
                    global_features = np.stack(columns, axis=1) if columns else np.empty((t,0))
                global_features = PyG._CFDArray(global_features, "graph_features")
                if global_features.ndim == 1: global_features = np.broadcast_to(global_features, (t,len(global_features)))
                if global_features.ndim != 2 or global_features.shape[0] != t: raise ValueError("graph_features must be [frames, features] or static [features].")
                names = names or [f"graph_{j}" for j in range(global_features.shape[1])]
                if graph_names is not None and graph_names != names:
                    raise ValueError("Named graph feature schemas must match across CFD sequences.")
                graph_names = names
                if feature_width is not None and (feature_width != global_features.shape[1] or extra_width != extra.shape[1]):
                    raise ValueError("CFD feature schemas must match across sequences.")
                feature_width, extra_width = global_features.shape[1], extra.shape[1]
                for frame in range(t-1):
                    graph_id = len(graphs); dt = float(times[frame+1]-times[frame]); timesteps.append(dt)
                    row = {"graph_id":graph_id,"label":0,"simulation_id":sequence_id,"frame_index":frame,"time":float(times[frame]),"dt":dt}
                    row.update({f"feat_{j}":float(v) for j,v in enumerate(global_features[frame])}); graphs.append(row)
                    delta = velocity[frame+1]-velocity[frame]
                    for i in range(n):
                        features = np.r_[velocity[frame,i], np.eye(4)[tags[i]], extra[i]]
                        row = {"graph_id":graph_id,"node_id":i,"x":positions[i,0],"y":positions[i,1],"z":positions[i,2],
                               "boundary_mask":bool(tags[i] in (1,2)),"volume":float(weights[i])}
                        row.update({f"feat_{j}":float(v) for j,v in enumerate(features)})
                        row.update({f"delta_v{axis}":float(delta[i,j]) for j,axis in enumerate("xyz"[:d])}); nodes.append(row)
                    for a,b in pairs:
                        relative = positions[a]-positions[b]
                        row = {"graph_id":graph_id,"src_id":a,"dst_id":b}
                        row.update({f"feat_{j}":float(v) for j,v in enumerate(np.r_[relative,np.linalg.norm(relative)])}); edges.append(row)
            options = {"velocity_feature_columns":[f"feat_{j}" for j in range(dimensions)],
                       "time_steps":sorted(set(round(v,12) for v in timesteps))}
            options.update(kwargs.pop("model_options", {}) or {})
            defaults = dict(level="node", task="regression", nodeLabelType="continuous", model_name="msa_gnn",
                            target_columns=tuple("delta_v"+a for a in "xyz"[:dimensions]), target_units=("m/s",)*dimensions,
                            split_mode="graph", holdout_group_by="simulation_id", split=(.6,.2,.2), normalize_features=True, normalize_targets=True,
                            node_weight_column="volume", boundary_column="boundary_mask", model_options=options)
            defaults.update(kwargs)
            if defaults["level"] != "node" or defaults["task"] != "regression" or defaults["target_columns"] != tuple("delta_v"+a for a in "xyz"[:dimensions]):
                raise ValueError("CFD adapters use node velocity-increment regression.")
            validation = _RunConfig()
            spellings = {"graphLabelType":"graph_label_type","nodeLabelType":"node_label_type","edgeLabelType":"edge_label_type"}
            for key,value in defaults.items():
                attr = spellings.get(key,key)
                if not hasattr(validation,attr): raise ValueError(f"Unknown hyperparameter: {key}.")
                setattr(validation,attr,value)
            PyG._ValidateConfig(validation)
            path = os.fspath(path); os.makedirs(path, exist_ok=True)
            expected = [os.path.join(path, name) for name in ("graphs.csv","nodes.csv","edges.csv","msa_manifest.json")]
            if not overwrite and any(os.path.exists(name) for name in expected): raise FileExistsError("CFD dataset files already exist; pass overwrite=True to replace those files only.")
            edge_columns = ["graph_id","src_id","dst_id"]+[f"feat_{j}" for j in range(4)]
            frames = [pd.DataFrame(graphs),pd.DataFrame(nodes),pd.DataFrame(edges,columns=edge_columns)]
            for filename,frame in zip(expected[:3],frames):
                PyG._WriteCFDFile(filename,frame)
            manifest = {"format_version":1,"target_kind":"velocity_increment","velocity_dimension":dimensions,
                        "velocity_feature_columns":options["velocity_feature_columns"],"target_columns":list(defaults["target_columns"]),
                        "time_steps":options["time_steps"],"boundary_types":{"interior":0,"wall":1,"inlet":2,"outlet":3},
                        "sequences":len(sequences),"transitions":len(graphs),"graph_feature_names":graph_names,
                        "loader_options":{k:getattr(validation,spellings.get(k,k)) for k in defaults}}
            PyG._WriteCFDFile(expected[3],manifest)
            return PyG.ByCSVPath(path, **defaults)

        def _MSAVelocityIndices(self):
            columns = (self.config.model_options or {}).get("velocity_feature_columns")
            schema = self._current_feature_schema()["node_feat_cols"]
            if columns is None or isinstance(columns,str) or len(columns) != self._num_outputs or len(set(columns)) != len(columns) or any(name not in schema for name in columns):
                raise ValueError("Set model_options.velocity_feature_columns to the current physical velocity input columns.")
            return [schema.index(name) for name in columns]

        def _MSACurrentVelocity(self, data):
            indices = self._MSAVelocityIndices()
            values = data.x[:,indices]
            scaler = self._scalers.get("x")
            if scaler:
                values = values * values.new_tensor(scaler["scale"])[indices] + values.new_tensor(scaler["mean"])[indices]
            return values

        def _MSASetVelocity(self, data, physical):
            indices = self._MSAVelocityIndices()
            value = physical
            scaler = self._scalers.get("x")
            if scaler:
                value = (value-value.new_tensor(scaler["mean"])[indices])/value.new_tensor(scaler["scale"])[indices]
            data.x = data.x.clone()
            data.x[:,indices] = value
            return data

        def _MSAEncodeIncrement(self, physical):
            scaler = self._scalers.get("targets")
            if scaler: return (physical-physical.new_tensor(scaler["mean"]))/physical.new_tensor(scaler["scale"])
            return physical

        def _MSANoise(self, data):
            std = (self.config.model_options or {}).get("noise_std", 0.)
            if not std: return data
            data = data.clone()
            indices = self._MSAVelocityIndices()
            noise = torch.randn_like(data.x[:,indices]) * std
            mask = getattr(data,"boundary_mask",None)
            if mask is not None: noise = noise.masked_fill(mask[:,None],0.)
            data.x[:,indices] += noise
            scaler = self._scalers.get("x")
            raw_noise = noise if scaler is None else noise*noise.new_tensor(scaler["scale"])[indices]
            target_scale = self._scalers.get("targets",{}).get("scale",[1.]*self._num_outputs)
            data.y = data.y - raw_noise/raw_noise.new_tensor(target_scale)
            return data

        def _MSAMultiStepEpoch(self, loader):
            options = self.config.model_options or {}
            horizon, decay = options.get("rollout_steps",1), options.get("rollout_loss_decay",1.)
            if self.config.split_mode != "graph" or not self.config.holdout_group_by:
                raise ValueError("Multi-step training requires whole-simulation grouped splits.")
            if "simulation_id" not in self.graph_df or "frame_index" not in self.graph_df:
                raise ValueError("Multi-step training requires CFD sequence identity/frame metadata.")
            lookup, owner = {}, {}
            for i,row in self.graph_df.iterrows():
                key = (row["simulation_id"],int(row["frame_index"]))
                if key in lookup: raise ValueError("Duplicate CFD simulation/frame identity.")
                lookup[key] = i
            for split in ("train","val","test"):
                for d in self._DataForSplit(split):
                    i = int(d.topologicpy_graph_index)
                    simulation = self.graph_df.iloc[i]["simulation_id"]
                    if simulation in owner and owner[simulation] != split: raise ValueError("A simulation crosses training/evaluation splits.")
                    owner[simulation] = split
            train_ids = {int(d.topologicpy_graph_index) for d in self.train_set}
            self.model.train(); losses = []
            for batch in loader:
                start = int(batch.topologicpy_graph_index.reshape(-1)[0])
                row = self.graph_df.iloc[start]
                current = self._MSANoise(batch.to(self.device))
                self.optimizer.zero_grad()
                total, mass = None, 0.
                for step in range(horizon):
                    index = lookup.get((row["simulation_id"],int(row["frame_index"])+step))
                    if index not in train_ids: break
                    reference = self.data_list[index].clone().to(self.device)
                    if reference.pos.shape != current.pos.shape or not torch.equal(reference.pos,current.pos) or not torch.equal(reference.edge_index,current.edge_index):
                        raise ValueError("CFD rollout mesh identity changed.")
                    truth_next = self._MSACurrentVelocity(reference) + self._PhysicalTensor(reference.y)
                    state = self._MSACurrentVelocity(current)
                    target = self._MSAEncodeIncrement(truth_next-state)
                    prediction,_ = self._Forward(current)
                    loss = self._loss_from_logits(prediction,target,"regression")
                    weight = decay**step
                    total = weight*loss if total is None else total+weight*loss
                    mass += weight
                    next_state = state+self._PhysicalTensor(prediction)
                    mask = getattr(reference,"boundary_mask",None)
                    if mask is not None: next_state = torch.where(mask[:,None],truth_next,next_state)
                    next_index = lookup.get((row["simulation_id"],int(row["frame_index"])+step+1))
                    if next_index not in train_ids: break
                    current = self.data_list[next_index].clone().to(self.device)
                    self._MSASetVelocity(current,next_state)
                if total is None: continue
                loss = total/mass; loss.backward(); self._apply_gradients()
                losses.append(float(loss.detach().cpu()))
            return float(np.mean(losses)) if losses else 0.

        @torch.no_grad()
        def Rollout(self, graphID, steps, timeStep=None, boundaryValues=None, graphFeatures=None, reference=None):
            """Autoregress physical velocities from a fixed mesh, predicting increments.

            Boundary values, if supplied, are physical [steps,nodes,components] next-step
            Dirichlet velocities; omitted values hold the initial boundary state fixed.
            graphFeatures are physical [steps,graph_features]. Returned states include
            the initial field. Reference, if supplied, includes that initial state too.
            """
            if self.config.level != "node" or self.config.task != "regression": raise ValueError("Rollout requires node velocity-increment regression.")
            if not isinstance(steps,int) or isinstance(steps,bool) or steps < 1: raise ValueError("steps must be a positive integer.")
            i = next((i for i,row in enumerate(self._identities) if row["graph_id"] == graphID),None)
            if i is None: raise ValueError("Unknown rollout graph ID.")
            data = self.data_list[i].clone().to(self.device)
            if getattr(data,"dt",None) is None: raise ValueError("Rollout requires a physical prediction time step.")
            dt = np.asarray(float(data.dt.item()) if timeStep is None else timeStep,dtype=float)
            if dt.ndim == 0: dt = np.full(steps,dt.item())
            if dt.shape != (steps,) or not np.isfinite(dt).all() or (dt<=0).any(): raise ValueError("Invalid rollout time steps.")
            allowed = (self.config.model_options or {}).get("time_steps")
            if allowed is None and self.config.time_step_column in self.graph_df:
                allowed = self.graph_df[self.config.time_step_column].unique().tolist()
            if allowed and not (self.config.model_options or {}).get("allow_unseen_time_step",False):
                if any(not np.isclose(value,allowed,rtol=1e-5,atol=1e-9).any() for value in dt):
                    raise ValueError("Prediction interval was not represented in training data.")
            state = self._MSACurrentVelocity(data)
            n,d = state.shape
            mask = getattr(data,"boundary_mask",torch.zeros(n,dtype=torch.bool,device=self.device))
            boundary = state.clone()
            if boundaryValues is not None:
                boundaryValues = np.asarray(boundaryValues,dtype=float)
                if boundaryValues.shape == (n,d): boundaryValues = np.broadcast_to(boundaryValues,(steps,n,d))
                if boundaryValues.shape != (steps,n,d) or not np.isfinite(boundaryValues).all(): raise ValueError("Invalid rollout boundaryValues shape/values.")
            if graphFeatures is not None:
                graphFeatures = np.asarray(graphFeatures,dtype=float)
                if graphFeatures.shape != (steps,self._graph_dim) or not np.isfinite(graphFeatures).all(): raise ValueError("Invalid rollout graphFeatures shape/values.")
            initial_time = float(data.time.item()) if getattr(data,"time",None) is not None else 0.
            states, increments, times = [state.cpu().numpy().copy()], [], [initial_time]
            mode = self.model.training; self.model.eval()
            try:
                for step in range(steps):
                    data.dt = state.new_tensor([dt[step]])
                    data.time = state.new_tensor([times[-1]])
                    if graphFeatures is not None:
                        value = state.new_tensor(graphFeatures[step:step+1])
                        scaler = self._scalers.get("u")
                        if scaler: value = (value-value.new_tensor(scaler["mean"]))/value.new_tensor(scaler["scale"])
                        data.u = value
                    batch = next(iter(self._Loader([data],batch_size=1,shuffle=False))).to(self.device)
                    prediction,_ = self._Forward(batch)
                    delta = self._PhysicalTensor(prediction)
                    next_state = state+delta
                    prescribed = boundary if boundaryValues is None else state.new_tensor(boundaryValues[step])
                    next_state = torch.where(mask[:,None],prescribed,next_state)
                    if not bool(torch.isfinite(next_state).all()): raise FloatingPointError("Non-finite autoregressive velocity field.")
                    increments.append((next_state-state).cpu().numpy().copy())
                    state = next_state
                    states.append(state.cpu().numpy().copy()); times.append(times[-1]+float(dt[step]))
                    self._MSASetVelocity(data,state)
            finally:
                self.model.train(mode)
            result = {"velocity":np.stack(states),"increments":np.stack(increments),"times":np.asarray(times),
                      "node_ids":copy.deepcopy(self._identities[i]["node_ids"]),"graph_id":graphID,"boundary_mask":mask.cpu().numpy()}
            if reference is not None:
                reference = np.asarray(reference)
                if reference.shape != result["velocity"].shape: raise ValueError("Reference must include the initial state and match the rollout shape.")
                weight = getattr(data,"node_weight",None)
                result["metrics"] = self.EvaluateRollout(result["velocity"][1:],reference[1:],
                    None if weight is None else weight.cpu().numpy(),mask.cpu().numpy(),componentNames=["v"+a for a in "xyz"[:d]],timeStep=dt)
            return result

        @staticmethod
        def ByCFDDataset(path, **kwargs):
            """Load a previously exported CFD dataset using its increment/feature manifest."""
            with open(os.path.join(path,"msa_manifest.json"),encoding="utf-8") as stream:
                manifest = json.load(stream)
            if manifest.get("format_version") != 1 or manifest.get("target_kind") != "velocity_increment":
                raise ValueError("Unsupported CFD dataset manifest.")
            options = dict(manifest["loader_options"])
            options.update(kwargs)
            return PyG.ByCSVPath(os.fspath(path),**options)

        @staticmethod
        def _WriteCFDFile(filename, value):
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w",encoding="utf-8",newline="",dir=os.path.dirname(filename),prefix=".msa-",suffix=".tmp",delete=False) as stream:
                    temporary = stream.name
                    if isinstance(value,pd.DataFrame): value.to_csv(stream,index=False)
                    else: json.dump(value,stream,indent=2)
                os.replace(temporary,filename)
            finally:
                if temporary and os.path.exists(temporary): os.unlink(temporary)

        _ModelFactories = {}

        @staticmethod
        def RegisterModel(name, factory):
            """Register a factory(context)->nn.Module with forward(data)->prediction.

            The returned module may alternatively return a dictionary with 'prediction'
            and optional 'embedding' tensors. Factories are registered by name in each
            process; executable Python code is never serialized into a checkpoint.
            """
            if not isinstance(name, str) or not name or name in ("default", "msa_gnn") or not callable(factory):
                raise ValueError("Use a nonempty custom name and a callable factory.")
            PyG._ModelFactories[name] = factory

        @staticmethod
        def _CaptureRNG():
            state = np.random.get_state()
            return {"python": random.getstate(),
                    "numpy": (state[0], state[1].tolist(), state[2], state[3], state[4]),
                    "torch": torch.get_rng_state().clone(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}

        @staticmethod
        def _RestoreRNG(state):
            random.setstate(state["python"])
            n = state["numpy"]
            np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
            torch.set_rng_state(state["torch"].cpu())
            if state.get("cuda") and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])

        @contextmanager
        def _RandomContext(self, reset=False):
            if self.device.type == "cuda": torch.cuda.init()
            outside = self._CaptureRNG()
            deterministic = torch.are_deterministic_algorithms_enabled()
            warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
            try:
                if self._rng_state is not None and not reset:
                    self._RestoreRNG(self._rng_state)
                else:
                    random.seed(self.config.random_state)
                    np.random.seed(self.config.random_state)
                    torch.random.default_generator.manual_seed(self.config.random_state)
                    if self.device.type == "cuda": torch.cuda.manual_seed_all(self.config.random_state)
                if self.config.deterministic:
                    torch.use_deterministic_algorithms(True)
                yield
            finally:
                self._rng_state = self._CaptureRNG()
                self._RestoreRNG(outside)
                torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)

        def _ValidateIdentity(self):
            cfg = self.config
            frames = (self.graph_df, self.nodes_df, self.edges_df)
            required = ((cfg.graph_id_header,), (cfg.graph_id_header, cfg.node_id_header),
                        (cfg.graph_id_header, cfg.edge_src_header, cfg.edge_dst_header))
            for name, frame, columns in zip(("graphs", "nodes", "edges"), frames, required):
                missing = [c for c in columns if c not in frame]
                if missing: raise ValueError(f"{name}.csv missing identity columns: {missing}.")
                if frame[list(columns)].isna().any().any():
                    raise ValueError(f"{name}.csv contains missing identity values.")
            if self.graph_df[cfg.graph_id_header].duplicated().any():
                raise ValueError("graphs.csv contains duplicate graph_id records.")
            if self.nodes_df.duplicated([cfg.graph_id_header, cfg.node_id_header]).any():
                raise ValueError("nodes.csv contains duplicate node_id records within a graph.")
            known = set(self.graph_df[cfg.graph_id_header])
            for name, frame in zip(("nodes", "edges"), frames[1:]):
                if not set(frame[cfg.graph_id_header]) <= known:
                    raise ValueError(f"{name}.csv contains orphan graph references.")
            if not known: raise ValueError("Dataset contains no graphs.")
            if cfg.holdout_group_by:
                if cfg.holdout_group_by not in self.graph_df or self.graph_df[cfg.holdout_group_by].isna().any():
                    raise ValueError("holdout_group_by must name a nonmissing graph column.")

        def _Targets(self, frame, entity):
            columns = list(self.config.target_columns) or [getattr(self.config, entity + "_label_header")]
            if any(col not in frame for col in columns):
                raise ValueError(f"Missing target columns: {columns}.")
            values = frame[columns].to_numpy(dtype=float)
            if not np.isfinite(values).all(): raise ValueError("Regression targets must be finite.")
            tensor = torch.tensor(values, dtype=torch.float32)
            return tensor if len(columns) > 1 else tensor[:, 0]

        def _PhysicalTensor(self, values):
            """Restore target units while retaining a vector's component dimension."""
            shaped = values.reshape(-1, self._num_outputs)
            scaler = self._scalers.get("targets")
            if scaler:
                shaped = shaped * shaped.new_tensor(scaler["scale"]) + shaped.new_tensor(scaler["mean"])
            return shaped if self._num_outputs > 1 else shaped[:, 0]

        def _RegressionArray(self, values):
            return self._PhysicalTensor(values).detach().cpu().numpy()

        def _GraphSplit(self):
            return self.config.level == "graph" or self.config.split_mode == "graph"

        def _DataForSplit(self, split):
            if self._GraphSplit():
                return getattr(self, {"train": "train_set", "val": "val_set", "test": "test_set"}[split])
            return self.data_list

        def _Loader(self, data, batch_size=1, shuffle=False):
            # A hierarchy is deliberately processed one scene at a time; arbitrary
            # coarse-level index tensors cannot use PyG's fine-node batch offsets.
            if self.config.model_name == "msa_gnn" or any(getattr(d, "hierarchy", None) is not None for d in data): batch_size = 1
            generator = None if shuffle else torch.Generator().manual_seed(self.config.random_state)
            return DataLoader(data, batch_size=batch_size, shuffle=shuffle, generator=generator)

        def _Forward(self, data):
            """Structured model contract shared by every training/inference path."""
            if self.config.model_name != "default":
                result = self.model(data)
                if isinstance(result, dict):
                    prediction, embedding = result["prediction"], result.get("embedding")
                else:
                    prediction, embedding = result, None
                if not torch.is_tensor(prediction): raise ValueError("Custom model must return a prediction tensor.")
                level = self.config.level
                count = getattr(data, "num_graphs", 1) if level == "graph" else data.num_nodes if level == "node" else data.num_edges if level == "edge" else data.edge_label_index.shape[1]
                scalar = self.config.task != "classification" and self._num_outputs == 1
                expected = (count, self._num_outputs)
                if tuple(prediction.shape) != expected and not (scalar and tuple(prediction.shape) == (count,)):
                    raise ValueError(f"Custom prediction shape must be {expected}, with scalar outputs optionally squeezed.")
                if level == "link": prediction = prediction.reshape(-1)
                if embedding is not None and (not torch.is_tensor(embedding) or embedding.ndim != 2 or embedding.shape[0] != data.num_nodes):
                    raise ValueError("Custom embeddings must have one feature vector per node.")
                return prediction, embedding
            embedding = self._Encode(data)
            level = self.config.level
            if level == "graph":
                prediction = self.model["head"](embedding, data.batch, getattr(data, "u", None))
            elif level == "node":
                prediction = self.model["head"](embedding)
            elif level == "edge":
                prediction = self.model["head"](embedding, data.edge_index, getattr(data, "edge_attr", None))
            else:
                prediction = self.model["predictor"](embedding, data.edge_label_index)
            return prediction, embedding

        def _AttachSimulation(self, data, graph, nodes, edges):
            cfg = self.config
            present = [col in nodes for col in cfg.position_columns]
            if any(present) and not all(present): raise ValueError("Supply all configured position columns or none.")
            if present and all(present):
                pos = nodes[list(cfg.position_columns)].to_numpy(dtype=float)
                if not np.isfinite(pos).all(): raise ValueError("Positions must be finite.")
                data.pos = torch.tensor(pos, dtype=torch.float32)
            for column, frame, attr in ((cfg.node_weight_column, nodes, "node_weight"),
                                        (cfg.edge_weight_column, edges, "edge_weight")):
                if column:
                    if column not in frame: raise ValueError(f"Missing physical weight column {column!r}.")
                    values = frame[column].to_numpy(dtype=float)
                    if not np.isfinite(values).all() or (values < 0).any():
                        raise ValueError("Physical weights must be finite and nonnegative.")
                    setattr(data, attr, torch.tensor(values, dtype=torch.float32))
            if cfg.boundary_column:
                if cfg.boundary_column not in nodes: raise ValueError("Missing boundary column.")
                data.boundary_mask = self._ParseMask(nodes[cfg.boundary_column].values)
            for column, attr in ((cfg.time_column, "time"), (cfg.time_step_column, "dt")):
                if column in graph:
                    value = float(graph[column].iloc[0])
                    if not math.isfinite(value) or (attr == "dt" and value <= 0):
                        raise ValueError("Time must be finite and time step must be positive.")
                    setattr(data, attr, torch.tensor([value], dtype=torch.float32))

        def SetGraphData(self, graphID, **attributes):
            """Attach simulation inputs by source graph ID, independent of ontology metadata.

            Accepted attributes are pos, boundary_mask, node_weight, edge_weight, time,
            dt and a hierarchy dictionary of tensors. Hierarchies are processed one
            graph per batch until a model provides a specialized hierarchy batch policy.
            """
            index = next((i for i, row in enumerate(self._identities) if row["graph_id"] == graphID), None)
            if index is None: raise ValueError(f"Unknown graph ID {graphID!r}.")
            data = self.data_list[index]
            updates = {}
            for key, value in attributes.items():
                if key not in ("pos", "boundary_mask", "node_weight", "edge_weight", "time", "dt", "hierarchy"):
                    raise ValueError(f"Unsupported simulation attribute {key!r}.")
                if key == "hierarchy":
                    if not isinstance(value, dict) or not all(isinstance(k, str) and torch.is_tensor(v) for k, v in value.items()):
                        raise ValueError("hierarchy must be a dictionary of tensors.")
                    updates[key] = {k: v.detach().cpu().clone() for k, v in value.items()}
                    continue
                tensor = torch.as_tensor(value).detach().cpu().clone()
                if key == "boundary_mask":
                    tensor = self._ParseMask(tensor.tolist())
                elif not torch.isfinite(tensor).all():
                    raise ValueError("Simulation inputs must be finite.")
                if key in ("node_weight", "edge_weight") and (tensor.ndim != 1 or (tensor < 0).any()):
                    raise ValueError("Physical weights must be nonnegative vectors.")
                count = data.num_edges if key == "edge_weight" else data.num_nodes
                if key in ("node_weight", "edge_weight", "boundary_mask") and tensor.shape != (count,):
                    raise ValueError("Simulation attribute size does not match the mesh.")
                if key == "pos" and (tensor.ndim != 2 or tensor.shape[0] != count):
                    raise ValueError("pos must contain one coordinate vector per node.")
                if key in ("time", "dt"):
                    if tensor.numel() != 1 or (key == "dt" and tensor.item() <= 0):
                        raise ValueError("time/dt must be scalar and dt must be positive.")
                    tensor = tensor.reshape(1)
                updates[key] = tensor
            if self.config.model_name == "msa_gnn" and "hierarchy" in updates:
                candidate = data.clone()
                for key,value in updates.items(): setattr(candidate,key,value)
                self._ValidateMSAHierarchy(candidate)
            for key, value in updates.items(): setattr(data, key, value)
            if "hierarchy" in updates:
                getattr(self,"_msa_auto_hierarchy",{}).pop(self._label_key(graphID),None)
            self._external_simulation.setdefault(self._label_key(graphID), {}).update(copy.deepcopy(updates))
            self._link_splits = None
            return self

        def _SetGraphMasks(self, train, val, test):
            prefix = "edge_" if self.config.level == "edge" else ""
            sets = [set(v) for v in (train, val, test)]
            for i, data in enumerate(self.data_list):
                n = data.num_edges if prefix else data.num_nodes
                for split, selected in zip(("train", "val", "test"), sets):
                    setattr(data, prefix + split + "_mask", torch.full((n,), i in selected, dtype=torch.bool))

        def _TrainingValues(self, key):
            result = []
            level = self.config.level
            if level == "link":
                self._BuildLinkSplits()
                source = [row["train"] for row in self._link_splits]
            else:
                source = self._DataForSplit("train")
            for data in source:
                value = getattr(data, key, None)
                if value is None: continue
                if level == "node" and key in ("x", "y"):
                    value = value[data.train_mask]
                elif level == "edge" and key in ("edge_attr", "edge_y"):
                    value = value[data.edge_train_mask]
                if level in ("node", "edge") and key == "u":
                    mask = data.edge_train_mask if level == "edge" else data.train_mask
                    if not bool(mask.any()): continue
                if value.numel(): result.append(value)
            return result

        def _PrepareNormalization(self, refit=False):
            cfg = self.config
            if refit and not self._scalers_frozen: self._scalers = {}
            # Always start from raw tensors; repeated application never compounds scaling.
            for data, raw in zip(self.data_list, self._raw_tensors):
                for key, value in raw.items(): setattr(data, key, value.clone())
            self._link_splits = None
            if not self._scalers_frozen and not cfg.inference:
                specs = []
                if cfg.normalize_features:
                    specs += [("x", self._in_dim), ("edge_attr", self._edge_dim), ("u", self._graph_dim)]
                if cfg.normalize_targets:
                    specs += [("edge_y" if cfg.level == "edge" else "y", self._num_outputs)]
                for key, width in specs:
                    name = "targets" if key in ("y", "edge_y") else key
                    if not width or name in self._scalers: continue
                    values = self._TrainingValues(key)
                    if not values: raise ValueError(f"No training values available to fit {key} normalization.")
                    matrix = torch.cat([v.reshape(-1, width).double() for v in values])
                    mean, scale = matrix.mean(0), matrix.std(0, unbiased=False)
                    scale = torch.where(scale > 1e-12, scale, torch.ones_like(scale))
                    self._scalers[name] = {"mean": mean.tolist(), "scale": scale.tolist()}
            for data, raw in zip(self.data_list, self._raw_tensors):
                for key, value in raw.items():
                    name = "targets" if key in ("y", "edge_y") else key
                    scaler = self._scalers.get(name)
                    if scaler is None: continue
                    width = len(scaler["mean"])
                    matrix = value.reshape(-1, width)
                    if matrix.shape[1] != width: raise ValueError("Scaler dimensions do not match the feature schema.")
                    scaled = (matrix - matrix.new_tensor(scaler["mean"])) / matrix.new_tensor(scaler["scale"])
                    setattr(data, key, scaled.reshape(value.shape))
            self._link_splits = None

        @staticmethod
        def PhysicsMetrics(prediction, truth, weights=None, boundaryMask=None, componentNames=None):
            """Physical-unit scalar/vector errors, optional spatial weighting and boundaries."""
            p, y = np.asarray(prediction, dtype=float), np.asarray(truth, dtype=float)
            if p.shape != y.shape or p.ndim not in (1, 2) or not p.size:
                raise ValueError("prediction/truth must have matching nonempty [samples, components] shapes.")
            if not np.isfinite(p).all() or not np.isfinite(y).all(): raise ValueError("Physical values must be finite.")
            p, y = p.reshape(len(p), -1), y.reshape(len(y), -1)
            n, k = p.shape
            w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
            if w.shape != (n,) or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
                raise ValueError("weights must be finite, nonnegative and have positive total weight.")
            names = list(componentNames) if componentNames else [f"component_{i}" for i in range(k)]
            if len(names) != k or len(set(names)) != k: raise ValueError("componentNames must match the distinct target components.")
            error = p - y
            mse = np.average(error ** 2, axis=0, weights=w)
            mae = np.average(np.abs(error), axis=0, weights=w)
            def r2(j):
                mean = np.average(y[:, j], weights=w)
                denominator = np.sum(w * (y[:, j] - mean) ** 2)
                numerator = np.sum(w * error[:, j] ** 2)
                return float(1 - numerator / denominator) if denominator > 0 else float(numerator == 0)
            components = {name: {"mae": float(mae[j]), "rmse": float(np.sqrt(mse[j])), "r2": r2(j)} for j, name in enumerate(names)}
            result = {"count": n, "mae": float(mae.mean()), "rmse": float(np.sqrt(mse.mean())),
                      "r2": float(np.mean([v["r2"] for v in components.values()])),
                      "vector_rmse": float(np.sqrt(mse.sum())),
                      "vector_mae": float(np.average(np.linalg.norm(error, axis=1), weights=w)),
                      "components": components}
            if boundaryMask is not None:
                mask = np.asarray(boundaryMask)
                if mask.shape != (n,) or not np.isin(mask, (0, 1)).all(): raise ValueError("boundaryMask must be boolean and match samples.")
                mask = mask.astype(bool)
                result["boundary"] = PyG.PhysicsMetrics(p[mask], y[mask], w[mask], componentNames=names) if mask.any() and w[mask].sum() > 0 else {"count": 0}
            return result

        @staticmethod
        def EvaluateRollout(prediction, truth, weights=None, boundaryMask=None, componentNames=None, timeStep=1.0):
            """Evaluate supplied physical rollout sequences [steps, samples, components]."""
            p, y = np.asarray(prediction), np.asarray(truth)
            if p.shape != y.shape or p.ndim not in (2, 3) or not len(p):
                raise ValueError("Rollouts require matching nonempty time-major sequences.")
            dt = np.asarray(timeStep, dtype=float)
            if dt.ndim == 0: dt = np.full(len(p), float(dt))
            if dt.shape != (len(p),) or not np.isfinite(dt).all() or (dt <= 0).any():
                raise ValueError("timeStep must be positive scalar or one value per step.")
            def at(values, step):
                if values is None: return None
                arr = np.asarray(values)
                return arr[step] if arr.ndim == 2 else arr
            steps = [PyG.PhysicsMetrics(p[i], y[i], at(weights, i), at(boundaryMask, i), componentNames) for i in range(len(p))]
            rmse = [row["rmse"] for row in steps]
            return {"times": np.cumsum(dt).tolist(), "steps": steps, "rmse": rmse,
                    "rmse_growth": float(rmse[-1] - rmse[0]), "mean_rmse": float(np.mean(rmse))}

        def EvaluatePhysics(self, split="test"):
            """Evaluate model outputs in physical units with mesh weights/boundary masks."""
            if self.config.task != "regression": raise ValueError("Physics evaluation requires regression.")
            self._CheckSupervision(split)
            output = self.Predict(split=split)
            level = self.config.level
            if level == "graph":
                prediction, truth = output["pred"], output["y_true"]
                weights, boundary = None, None
            else:
                prediction = np.concatenate(output["pred"], axis=0)
                truth = np.concatenate(output["y_true"], axis=0)
                weights, boundary = [], []
                for data in self.data_list:
                    prefix = "edge_" if level == "edge" else ""
                    mask = getattr(data, prefix + split + "_mask").cpu().numpy()
                    n = data.num_edges if level == "edge" else data.num_nodes
                    w = getattr(data, "edge_weight" if level == "edge" else "node_weight", torch.ones(n)).cpu().numpy()
                    b = getattr(data, "boundary_mask", torch.zeros(n, dtype=torch.bool)).cpu().numpy() if level == "node" else np.zeros(n, dtype=bool)
                    weights.extend(w[mask]); boundary.extend(b[mask])
            names = self.config.target_columns or None
            metrics = self.PhysicsMetrics(prediction, truth, weights, boundary, names)
            metrics["units"] = list(self.config.target_units)
            return metrics

        def _CSVSignature(self):
            digest = hashlib.sha256()
            for frame in (self.graph_df, self.nodes_df, self.edges_df):
                digest.update(repr(list(frame.columns)).encode())
                digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
            return digest.hexdigest()

        def _DatasetSignature(self):
            digest = hashlib.sha256()
            for frame in (self.graph_df, self.nodes_df, self.edges_df):
                digest.update(repr(list(frame.columns)).encode())
                digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
            settings = {k: getattr(self.config, k) for k in ("level", "task", "target_columns", "split_mode", "split",
                        "holdout_group_by", "random_state", "shuffle", "link_val_ratio", "link_test_ratio", "link_is_undirected")}
            digest.update(json.dumps(settings, sort_keys=True).encode())
            for data in self.data_list:
                for key in ("pos", "boundary_mask", "node_weight", "edge_weight", "time", "dt", "hierarchy"):
                    value = getattr(data, key, None)
                    values = sorted(value.items()) if isinstance(value, dict) else [(key, value)]
                    for name, tensor in values:
                        if torch.is_tensor(tensor):
                            digest.update(name.encode()); digest.update(str(tensor.shape).encode())
                            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            return digest.hexdigest()

        @staticmethod
        def _ValidateConfig(cfg):
            """Validate configuration before loading data or mutating a live runner."""
            for key in ("target_columns", "target_units", "position_columns"):
                value = getattr(cfg, key)
                if isinstance(value, str) or not isinstance(value, (tuple, list)):
                    raise ValueError(f"{key} must be a sequence of names.")
                if any(not isinstance(v, str) or not v for v in value):
                    raise ValueError(f"{key} entries must be nonempty strings.")
                setattr(cfg, key, tuple(value))
            if len(set(cfg.target_columns)) != len(cfg.target_columns):
                raise ValueError("target_columns must be distinct.")
            if cfg.target_columns and cfg.task != "regression":
                raise ValueError("target_columns are supported for regression only.")
            if cfg.normalize_targets and cfg.task != "regression":
                raise ValueError("Target normalization requires regression.")
            if cfg.target_units and len(cfg.target_units) != (len(cfg.target_columns) or 1):
                raise ValueError("target_units must match the number of target components.")
            if cfg.split_mode not in ("entity", "graph"):
                raise ValueError("split_mode must be 'entity' or 'graph'.")
            if cfg.level == "link" and cfg.split_mode == "graph":
                raise ValueError("Link tasks use fixed edge splits; whole-graph splits require a node/edge/graph task.")
            if not isinstance(cfg.model_name, str) or not cfg.model_name:
                raise ValueError("model_name must be a nonempty string.")
            if cfg.model_name not in ("default", "msa_gnn") and cfg.model_name not in PyG._ModelFactories:
                raise ValueError(f"Register custom model {cfg.model_name!r} before constructing/loading it.")
            if cfg.model_options is not None and not isinstance(cfg.model_options, dict):
                raise ValueError("model_options must be a dictionary.")
            try:
                json.dumps(cfg.model_options, allow_nan=False)
            except (TypeError, ValueError) as error:
                raise ValueError("model_options must contain JSON-compatible finite values.") from error
            if cfg.model_name == "msa_gnn":
                if cfg.level != "node" or cfg.task != "regression" or len(cfg.target_columns) not in (2, 3):
                    raise ValueError("MSA-GNN requires node regression with two or three velocity-increment target columns.")
                options = cfg.model_options or {}
                known = {"latent_dim","processor_blocks","attention_heads","max_coarse_nodes","rollout_steps",
                         "aggregation","attention","multiscale","dropout","coarsening_ratio","n_clusters",
                         "noise_std","rollout_loss_decay","velocity_feature_columns","time_steps","allow_unseen_time_step"}
                if set(options)-known: raise ValueError(f"Unknown MSA options: {sorted(set(options)-known)}.")
                for flag in ("attention","multiscale","allow_unseen_time_step"):
                    if flag in options and not isinstance(options[flag],bool): raise ValueError(f"MSA {flag} must be boolean.")
                if "time_steps" in options and (not isinstance(options["time_steps"],(list,tuple)) or not options["time_steps"] or any(not isinstance(v,(int,float)) or v<=0 for v in options["time_steps"])):
                    raise ValueError("MSA time_steps must be a positive numeric sequence.")
                for name in ("latent_dim", "processor_blocks", "attention_heads", "max_coarse_nodes", "rollout_steps"):
                    value = options.get(name, {"latent_dim":128,"processor_blocks":4,"attention_heads":2,"max_coarse_nodes":128,"rollout_steps":1}[name])
                    if not isinstance(value, int) or isinstance(value, bool) or value < 1: raise ValueError(f"Invalid MSA option {name}.")
                if options.get("latent_dim", 128) % options.get("attention_heads", 2): raise ValueError("MSA latent_dim must be divisible by attention_heads.")
                if options.get("aggregation", "gru") not in ("gru", "mean"): raise ValueError("MSA aggregation must be gru or mean.")
                if not 0 <= options.get("dropout", .1) < 1 or not 0 < options.get("coarsening_ratio", .1) <= 1:
                    raise ValueError("Invalid MSA dropout/coarsening ratio.")
                if options.get("noise_std", 0.) < 0 or not 0 < options.get("rollout_loss_decay", 1.) <= 1:
                    raise ValueError("Invalid MSA noise/rollout decay.")
            for key in ("normalize_features", "normalize_targets", "deterministic"):
                if not isinstance(getattr(cfg, key), bool):
                    raise ValueError(f"{key} must be boolean.")
            enums = {"level": ("graph", "node", "edge", "link"),
                     "task": ("classification", "regression", "link_prediction"),
                     "conv": ("sage", "gcn", "gatv2"), "optimizer": ("adam", "adamw"),
                     "activation": ("relu", "gelu", "elu", "tanh"),
                     "pooling": ("mean", "max", "sum"), "cv": ("holdout", "kfold")}
            for key, values in enums.items():
                if getattr(cfg, key) not in values:
                    raise ValueError(f"Unsupported {key}: {getattr(cfg, key)!r}.")
            if (cfg.level == "link") != (cfg.task == "link_prediction"):
                raise ValueError("Use task='link_prediction' exactly when level='link'.")
            entity = {"graph": "graph", "node": "node", "edge": "edge"}.get(cfg.level)
            for key in ("graph_label_type", "node_label_type", "edge_label_type"):
                if getattr(cfg, key) not in ("categorical", "continuous"):
                    raise ValueError(f"Invalid {key}.")
            if entity and getattr(cfg, entity + "_label_type") != ("continuous" if cfg.task == "regression" else "categorical"):
                raise ValueError("Regression requires continuous labels; classification requires categorical labels.")
            if not isinstance(cfg.split, (tuple, list)) or len(cfg.split) != 3:
                raise ValueError("split must contain three ratios.")
            ratios = np.asarray(cfg.split, dtype=float)
            if not np.isfinite(ratios).all() or (ratios < 0).any() or not np.isclose(ratios.sum(), 1):
                raise ValueError("split ratios must be finite, nonnegative and sum to 1.")
            ratios = ratios / ratios.sum()
            cfg.split = tuple(float(v) for v in ratios)
            if not cfg.hidden_dims or any(not isinstance(v, (int, np.integer)) or isinstance(v, bool) or v <= 0 for v in cfg.hidden_dims):
                raise ValueError("hidden_dims must contain positive integers.")
            cfg.hidden_dims = tuple(int(v) for v in cfg.hidden_dims)
            for key in ("epochs", "batch_size", "early_stopping_patience", "k_folds"):
                value = getattr(cfg, key)
                if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < (2 if key == "k_folds" else 1):
                    raise ValueError(f"{key} must be a positive integer (k_folds >= 2).")
                setattr(cfg, key, int(value))
            if not isinstance(cfg.random_state, (int, np.integer)) or not 0 <= cfg.random_state < 2**32:
                raise ValueError("random_state must be an integer in [0, 2**32).")
            cfg.random_state = int(cfg.random_state)
            for key in ("lr", "weight_decay", "dropout", "link_val_ratio", "link_test_ratio"):
                value = float(getattr(cfg, key))
                if not math.isfinite(value) or value < 0 or (key == "lr" and value == 0):
                    raise ValueError(f"Invalid {key}.")
                setattr(cfg, key, value)
            if cfg.dropout >= 1 or cfg.link_val_ratio + cfg.link_test_ratio >= 1:
                raise ValueError("dropout and the sum of link holdout ratios must be < 1.")
            if cfg.gradient_clip_norm is not None and (not math.isfinite(float(cfg.gradient_clip_norm)) or cfg.gradient_clip_norm <= 0):
                raise ValueError("gradient_clip_norm must be positive and finite.")
            if isinstance(cfg.optional_feature_columns, str):
                raise ValueError("optional_feature_columns must be a sequence of feature names.")
            cfg.optional_feature_columns = tuple(cfg.optional_feature_columns)

        def _SplitCounts(self, n):
            ratios = np.asarray(self.config.split)
            counts = np.floor(ratios * n).astype(int)
            fractions = ratios * n - counts
            for i in sorted(range(3), key=lambda i: (-fractions[i], i))[:n - int(counts.sum())]:
                counts[i] += 1
            return tuple(int(v) for v in counts)

        @staticmethod
        def _ParseMask(values):
            result = []
            for value in values:
                if isinstance(value, str):
                    token = value.strip().lower()
                    if token not in ("true", "false", "1", "0"):
                        raise ValueError(f"Invalid boolean mask value: {value!r}.")
                    result.append(token in ("true", "1"))
                elif pd.isna(value) or value not in (True, False, 0, 1):
                    raise ValueError(f"Invalid boolean mask value: {value!r}.")
                else:
                    result.append(bool(value))
            return torch.tensor(result, dtype=torch.bool)

        def _Masks(self, frame, entity):
            cfg = self.config
            names = [getattr(cfg, f"{entity}_{part}_mask_header") for part in ("train", "val", "test")]
            present = [name in frame for name in names]
            if any(present) and not all(present):
                raise ValueError("Supply all three mask columns or none.")
            if all(present):
                masks = tuple(self._ParseMask(frame[name].values) for name in names)
                if (sum(mask.int() for mask in masks) > 1).any():
                    raise ValueError("Training, validation and test masks must not overlap.")
                return masks
            n = len(frame)
            idx = list(range(n))
            if cfg.shuffle:
                random.Random(cfg.random_state).shuffle(idx)
            cuts = np.cumsum([0] + list(self._SplitCounts(n)))
            masks = []
            for j in range(3):
                selected = set(idx[cuts[j]:cuts[j+1]])
                masks.append(torch.tensor([i in selected for i in range(n)], dtype=torch.bool))
            return tuple(masks)

        def _Encode(self, data):
            """Use the same node, edge and graph inputs on every forward path."""
            return self.model["encoder"](
                data.x, data.edge_index, getattr(data, "edge_attr", None),
                getattr(data, "u", None), getattr(data, "batch", None))

        def _BuildLinkSplits(self):
            if self.config.level != "link" or self._link_splits is not None:
                return
            cfg = self.config
            transform = RandomLinkSplit(num_val=cfg.link_val_ratio, num_test=cfg.link_test_ratio,
                                        is_undirected=cfg.link_is_undirected,
                                        add_negative_train_samples=True, neg_sampling_ratio=1.0)
            python_state = random.getstate()
            numpy_state = np.random.get_state()
            try:
                random.seed(cfg.random_state)
                np.random.seed(cfg.random_state)
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(cfg.random_state)
                    self._link_splits = [dict(zip(("train", "val", "test"), transform(d.clone()))) for d in self.data_list]
            finally:
                random.setstate(python_state)
                np.random.set_state(numpy_state)

        def _LinkData(self, data, split_name):
            self._BuildLinkSplits()
            index = int(data.topologicpy_graph_index.reshape(-1)[0])
            return self._link_splits[index][split_name].clone().to(self.device)

        def _CheckSupervision(self, split):
            if self.config.inference:
                raise ValueError("This is an inference-only dataset; training and evaluation require labelled data.")
            level = self.config.level
            if level == "graph":
                count = len(getattr(self, {"train": "train_set", "val": "val_set", "test": "test_set"}[split]))
            elif level == "link":
                self._BuildLinkSplits()
                count = sum(int(d[split].edge_label.numel()) for d in self._link_splits)
            else:
                key = ("edge_" if level == "edge" else "") + split + "_mask"
                count = sum(int(getattr(d, key).sum()) for d in self.data_list)
            if count == 0:
                raise ValueError(f"No supervised samples in the {split} split.")

        def _Fit(self, train_loader, val_loader, train_epoch, eval_epoch):
            best, best_state, best_optimizer, best_rng, patience = float("inf"), None, None, None, 0
            if getattr(self, "_continue_history", False) and getattr(self, "_early_state", None):
                state = self._early_state
                best, best_state, best_optimizer, best_rng, patience = state["loss"], state["model"], state["optimizer"], state["rng"], state["patience"]
                if patience >= self.config.early_stopping_patience:
                    if best_state is not None:
                        self.model.load_state_dict(best_state)
                        self.optimizer.load_state_dict(best_optimizer)
                        self._RestoreRNG(best_rng)
                    return self.history
            if not getattr(self, "_continue_history", False):
                self.history = {"train_loss": [], "val_loss": []}
                self._epochs_completed = 0
            for _ in range(self.config.epochs):
                tr = train_epoch(train_loader)
                va = eval_epoch(val_loader) if val_loader is not None else float("nan")
                self.history["train_loss"].append(tr)
                self.history["val_loss"].append(va)
                self._epochs_completed += 1
                if not math.isfinite(tr) or (val_loader is not None and not math.isfinite(va)):
                    raise ValueError("Non-finite training or validation loss.")
                if self.config.early_stopping:
                    if va < best - 1e-9:
                        best, best_state, best_optimizer, best_rng, patience = va, copy.deepcopy(self.model.state_dict()), copy.deepcopy(self.optimizer.state_dict()), self._CaptureRNG(), 0
                    else:
                        patience += 1
                        if patience >= self.config.early_stopping_patience: break
            if self.config.early_stopping:
                self._early_state = {"loss": best, "model": best_state, "optimizer": best_optimizer,
                                     "rng": best_rng, "patience": patience}
                self._continuation_state = {"model": copy.deepcopy(self.model.state_dict()),
                                            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
                                            "rng": self._CaptureRNG()}
            if best_state is not None:
                self.model.load_state_dict(best_state)
                self.optimizer.load_state_dict(best_optimizer)
                self._RestoreRNG(best_rng)
            return self.history

        def _DecodePredictions(self, out, entity):
            if self.config.task != "classification":
                return out
            decoder = self._label_decoders[entity]
            decode = lambda a: np.asarray([decoder[int(i)] for i in np.asarray(a).reshape(-1)])
            out["class_index"] = out["pred"]
            out["class_labels"] = [decoder[i] for i in range(self._num_outputs)]
            out["pred"] = [decode(a) for a in out["pred"]] if isinstance(out["pred"], list) else decode(out["pred"])
            if "y_true" in out:
                out["y_true_index"] = out["y_true"]
                out["y_true"] = [decode(a) for a in out["y_true"]] if isinstance(out["y_true"], list) else decode(out["y_true"])
            return out

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
                      ontology: bool = True,
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
            ontology : bool, optional
                If True, preserves ontology and semantic metadata columns from
                ``graphs.csv``, ``nodes.csv``, and ``edges.csv``. These columns are
                stored as metadata and are not converted into numeric feature tensors.
                Default is True.
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
                             edge_label_type=edgeLabelType,
                             ontology=bool(ontology))

            # allow override of any config field via kwargs
            for k, v in kwargs.items():
                if not hasattr(cfg, k):
                    raise ValueError(f"Unknown hyperparameter: {k}.")
                setattr(cfg, k, v)

            return PyG(path=path, config=cfg)

        def __init__(self, path: str, config: _RunConfig):
            self._ValidateConfig(config)
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

            # Encoders for categorical labels. CrossEntropyLoss requires labels
            # to be contiguous integers in [0, num_classes-1], while CSV labels
            # may be strings or arbitrary integer codes.
            self._label_encoders: Dict[str, Dict[object, int]] = {"graph": {}, "node": {}, "edge": {}}
            self._label_decoders: Dict[str, Dict[int, object]] = {"graph": {}, "node": {}, "edge": {}}

            # Saved training schema (used to keep feature dimensions consistent during inference)
            self._in_dim: Optional[int] = None
            self._feature_schema: Optional[Dict[str, List[str]]] = None
            self._freeze_num_outputs: bool = False
            self._freeze_label_encoders = False
            self._link_splits = None
            self._scalers = {}
            self._scalers_frozen = False
            self._rng_state = None
            self._epochs_completed = 0
            self._raw_tensors = []
            self._identities = []
            self._training_signature = None
            self._external_simulation = {}

            # Ontology / semantic metadata is deliberately kept outside numeric
            # feature tensors. It is keyed by graph index and graph id so that it
            # can be carried through summaries, predictions, and checkpoints.
            self.ontology_metadata: Dict[str, object] = {
                "enabled": bool(config.ontology),
                "columns": {"graph": [], "node": [], "edge": []},
                "graphs": {},
                "nodes": {},
                "edges": {},
            }

            self._load_csv()
            self._build_data_list()
            self._split_holdout()
            self._PrepareMSAHierarchies()
            self._PrepareNormalization()

            self._build_model()


        # ----------------------------
        # Convenience: hyperparameters
        # ----------------------------
        def SetHyperparameters(self, **kwargs):
            """Apply validated settings to the active runner.

            Parameters
            ----------
            **kwargs : dict
                Configuration fields to update. Optimizer parameters take effect
                immediately. Architecture changes rebuild the model and optimizer.
                Changes to task, label types, inference mode, or CSV schema require
                a new instance. Link split changes also require a new instance.

            Returns
            -------
            dict
                The current :meth:`Summary`.

            Raises
            ------
            ValueError
                An unknown, invalid, or unsupported setting was supplied.
            """
            cfg = copy.deepcopy(self.config)
            immutable = {"level", "task", "inference", "ontology", "optional_feature_columns",
                         "target_columns", "target_units", "normalize_features", "normalize_targets", "split_mode",
                         "position_columns", "node_weight_column", "edge_weight_column", "boundary_column",
                         "time_column", "time_step_column"}
            for key, value in kwargs.items():
                if not hasattr(cfg, key):
                    raise ValueError(f"Unknown hyperparameter: {key}.")
                if (key in immutable or key.endswith("_header") or key.endswith("_label_type")) and value != getattr(cfg, key):
                    raise ValueError(f"Changing {key} requires a new ByCSVPath instance.")
                setattr(cfg, key, value)
            self._ValidateConfig(cfg)
            old = self.config
            changed = {key for key in kwargs if getattr(old, key) != getattr(cfg, key)}
            if changed & {"link_val_ratio", "link_test_ratio", "link_is_undirected"} or (cfg.level == "link" and "random_state" in changed):
                raise ValueError("Changing link splits requires a new instance to prevent evaluation leakage.")
            if cfg.holdout_group_by and (cfg.holdout_group_by not in self.graph_df or self.graph_df[cfg.holdout_group_by].isna().any()):
                raise ValueError("holdout_group_by must name a nonmissing graph column.")
            if self._epochs_completed and changed & {"split", "shuffle", "random_state", "holdout_group_by"}:
                raise ValueError("Changing trained experiment splits/seeds requires a new instance.")
            self.config = cfg
            self.device = torch.device("cuda:0" if cfg.use_gpu and torch.cuda.is_available() else "cpu")
            if changed & {"conv", "hidden_dims", "activation", "dropout", "batch_norm", "residual", "pooling", "model_name", "model_options", "random_state"}:
                self._PrepareMSAHierarchies()
                self._build_model()
            else:
                self.model.to(self.device)
                for state in self.optimizer.state.values():
                    for key, value in state.items():
                        if torch.is_tensor(value):
                            state[key] = value.to(self.device)
                if "optimizer" in changed:
                    optimizer = torch.optim.AdamW if cfg.optimizer == "adamw" else torch.optim.Adam
                    self.optimizer = optimizer(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
                else:
                    for group in self.optimizer.param_groups:
                        group.update(lr=cfg.lr, weight_decay=cfg.weight_decay)
            if "early_stopping" in changed:
                self._early_state, self._continuation_state = None, None
            if changed & {"split", "shuffle", "random_state", "holdout_group_by"}:
                if cfg.level in ("node", "edge"):
                    self._build_data_list()
                self._split_holdout()
                self._PrepareMSAHierarchies()
                self._PrepareNormalization(refit=True)
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
                "holdout_group_by": cfg.holdout_group_by,
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
                "ontology": bool(getattr(cfg, "ontology", False)),
                "ontology_metadata_columns": dict((getattr(self, "ontology_metadata", {}) or {}).get("columns", {})),
                "num_graphs": len(self.data_list),
                "num_outputs": int(self._num_outputs),
                "target_columns": list(self.config.target_columns),
                "split_mode": self.config.split_mode,
                "model_name": self.config.model_name,
                "epochs_completed": self._epochs_completed,
                "normalization": copy.deepcopy(self._scalers),
                "edge_feature_dim": int(self._edge_dim),
                "graph_feature_dim": int(self._graph_dim),
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

        # ----------------------------
        # Ontology / semantic metadata
        # ----------------------------
        def _ontology_base_columns(self) -> List[str]:
            """
            Returns the canonical ontology metadata column names for CSV preservation.

            These are not interpreted as numeric features. They are retained as
            metadata so predictions and trained checkpoints can be traced back to
            semantic TopologicPy, IFC, graph database, or RDF entities.
            """
            cfg = self.config
            cols = [
                getattr(cfg, "ontology_class_header", "ontology_class"),
                getattr(cfg, "category_header", "category"),
                getattr(cfg, "uri_header", "uri"),
                getattr(cfg, "ifc_class_header", "ifc_class"),
                getattr(cfg, "ifc_guid_header", "ifc_guid"),
                getattr(cfg, "source_header", "source"),
                getattr(cfg, "derived_from_header", "derived_from"),
                getattr(cfg, "generated_by_header", "generated_by"),
                "label",
                "name",
                "description",
            ]
            # Preserve order and remove duplicates.
            return list(dict.fromkeys([c for c in cols if isinstance(c, str) and c.strip()]))

        def _ontology_columns(self, df: pd.DataFrame, entity: str) -> List[str]:
            """
            Returns ontology/semantic metadata columns present in the given dataframe.

            Parameters
            ----------
            df : pandas.DataFrame
                The input dataframe.
            entity : str
                One of ``"graph"``, ``"node"``, or ``"edge"``.
            """
            if df is None or not getattr(self.config, "ontology", True):
                return []

            cfg = self.config
            candidate_cols = self._ontology_base_columns()

            if entity == "graph":
                candidate_cols += [cfg.graph_id_header, cfg.graph_label_header]
            elif entity == "node":
                candidate_cols += [cfg.graph_id_header, cfg.node_id_header, cfg.node_label_header]
            elif entity == "edge":
                candidate_cols += [cfg.graph_id_header, cfg.edge_src_header, cfg.edge_dst_header, cfg.edge_label_header]

            candidate_cols = list(dict.fromkeys(candidate_cols))
            return [c for c in candidate_cols if c in df.columns]

        @staticmethod
        def _clean_metadata_value(value):
            """
            Converts numpy/pandas scalar values to serialisable Python values.
            """
            try:
                if pd.isna(value):
                    return None
            except Exception:
                pass
            if isinstance(value, np.generic):
                return value.item()
            return value

        def _row_metadata(self, row, columns: List[str]) -> Dict[str, object]:
            """
            Returns a clean metadata dictionary from a pandas row.
            """
            meta = {}
            for col in columns:
                try:
                    value = self._clean_metadata_value(row[col])
                except Exception:
                    value = None
                if value is not None:
                    meta[col] = value
            return meta

        def _reset_ontology_metadata(self, graph_cols=None, node_cols=None, edge_cols=None):
            """
            Resets ontology metadata containers.
            """
            self.ontology_metadata = {
                "enabled": bool(getattr(self.config, "ontology", True)),
                "columns": {
                    "graph": list(graph_cols or []),
                    "node": list(node_cols or []),
                    "edge": list(edge_cols or []),
                },
                "graphs": {},
                "nodes": {},
                "edges": {},
            }

        def OntologyMetadata(self) -> Dict[str, object]:
            """
            Returns preserved ontology and semantic metadata for the loaded dataset.

            Returns
            -------
            dict
                A dictionary with ``graphs``, ``nodes``, ``edges`` and ``columns``
                sections. Metadata is keyed by graph index and graph id where possible.
            """
            return copy.deepcopy(getattr(self, "ontology_metadata", {}) or {})

        def MetadataByGraphID(self, graphID) -> Dict[str, object]:
            """
            Returns preserved metadata for one graph id.

            Parameters
            ----------
            graphID : any
                The graph id value as stored in ``graphs.csv``.

            Returns
            -------
            dict
                A dictionary containing graph, node, and edge metadata.
            """
            md = getattr(self, "ontology_metadata", {}) or {}
            key = str(graphID)
            return {
                "graph": copy.deepcopy((md.get("graphs", {}) or {}).get(key, {})),
                "nodes": copy.deepcopy((md.get("nodes", {}) or {}).get(key, [])),
                "edges": copy.deepcopy((md.get("edges", {}) or {}).get(key, [])),
            }

        # ----------------
        # Dataset loading
        # ----------------
        def _load_csv(self):
            if not isinstance(self.path, str) or not os.path.isdir(self.path):
                raise ValueError("Dataset path must be an existing directory.")
            frames = []
            for name in ("graphs", "nodes", "edges"):
                path = os.path.join(self.path, name + ".csv")
                if not os.path.isfile(path):
                    raise ValueError("graphs.csv, nodes.csv and edges.csv are required.")
                frames.append(pd.read_csv(path))
            self.graph_df, self.nodes_df, self.edges_df = frames
            self._ValidateIdentity()

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
        def _ensure_feature_columns(df, cols, optional=()):
            missing = [col for col in cols if col not in df]
            required = [col for col in missing if col not in optional]
            if required:
                raise ValueError(f"Missing required feature columns: {required}.")
            df = df.copy()
            for col in missing:
                df[col] = 0.0
            return df
        @staticmethod
        def _label_key(value):
            """Return a stable, hashable key for a CSV label value or node id."""
            try:
                if pd.isna(value):
                    return None
            except Exception:
                pass
            if isinstance(value, np.generic):
                value = value.item()
            try:
                hash(value)
                return value
            except Exception:
                return str(value)

        @staticmethod
        def _sorted_label_keys(values) -> List[object]:
            keys = []
            for value in values:
                key = PyG._label_key(value)
                if key is not None and key not in keys:
                    keys.append(key)
            return sorted(keys, key=lambda x: (str(type(x)), str(x)))

        @staticmethod
        def _infer_num_classes(values: np.ndarray) -> int:
            return int(len(PyG._sorted_label_keys(values)))

        def _fit_label_encoder(self, entity: str, values) -> Dict[object, int]:
            """Fit a stable categorical label encoder for graph/node/edge labels."""
            keys = self._sorted_label_keys(values)
            encoder = {key: i for i, key in enumerate(keys)}
            self._label_encoders[entity] = encoder
            self._label_decoders[entity] = {i: key for key, i in encoder.items()}
            return encoder

        def _encode_label_value(self, entity: str, value) -> int:
            key = self._label_key(value)
            encoder = self._label_encoders.get(entity, {})
            if key not in encoder:
                raise ValueError(f"PyG - Error: Unknown {entity} categorical label {value!r}.")
            return int(encoder[key])

        def _encode_label_values(self, entity: str, values) -> np.ndarray:
            return np.array([self._encode_label_value(entity, v) for v in values], dtype=np.int64)

        def _build_data_list(self):
            assert self.graph_df is not None and self.nodes_df is not None and self.edges_df is not None

            self._ValidateIdentity()
            self.data_list = []
            self._identities = []
            self._link_splits = None
            cfg = self.config
            gdf = self.graph_df
            ndf = self.nodes_df
            edf = self.edges_df

            graph_feat_cols = self._feature_columns(gdf, cfg.graph_features_header)
            node_feat_cols = self._feature_columns(ndf, cfg.node_features_header)
            edge_feat_cols = self._feature_columns(edf, cfg.edge_features_header)

            graph_meta_cols = self._ontology_columns(gdf, "graph")
            node_meta_cols = self._ontology_columns(ndf, "node")
            edge_meta_cols = self._ontology_columns(edf, "edge")
            self._reset_ontology_metadata(graph_meta_cols, node_meta_cols, edge_meta_cols)

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
                gdf = self._ensure_feature_columns(gdf, graph_feat_cols, cfg.optional_feature_columns)
                ndf = self._ensure_feature_columns(ndf, node_feat_cols, cfg.optional_feature_columns)
                edf = self._ensure_feature_columns(edf, edge_feat_cols, cfg.optional_feature_columns)

                # Persist back (so later calls see the aligned frames)
                self.graph_df, self.nodes_df, self.edges_df = gdf, ndf, edf

            if len(node_feat_cols) == 0:
                raise ValueError(
                    f"PyG - Error: No node feature columns found. "
                    f"Expected columns starting with '{cfg.node_features_header}_'."
                )

            self._edge_dim = len(edge_feat_cols)
            self._graph_dim = len(graph_feat_cols)
            for frame, cols in ((gdf, graph_feat_cols), (ndf, node_feat_cols), (edf, edge_feat_cols)):
                if cols and not np.isfinite(frame[cols].to_numpy(dtype=float)).all():
                    raise ValueError("Numeric features must be finite.")

            # Fit categorical encoders once per dataset. This fixes string labels
            # and arbitrary integer label codes such as {10, 20}; PyTorch
            # CrossEntropyLoss requires contiguous zero-based targets.
            if not self._freeze_label_encoders and cfg.graph_label_type == "categorical" and cfg.graph_label_header in gdf.columns:
                self._fit_label_encoder("graph", gdf[cfg.graph_label_header].values)
            if not self._freeze_label_encoders and cfg.node_label_type == "categorical" and cfg.node_label_header in ndf.columns:
                self._fit_label_encoder("node", ndf[cfg.node_label_header].values)
            if not self._freeze_label_encoders and cfg.edge_label_type == "categorical" and cfg.edge_label_header in edf.columns:
                self._fit_label_encoder("edge", edf[cfg.edge_label_header].values)

            for graph_index, gid in enumerate(gdf[cfg.graph_id_header].unique()):
                g_row = gdf[gdf[cfg.graph_id_header] == gid]
                g_nodes = ndf[ndf[cfg.graph_id_header] == gid].sort_values(cfg.node_id_header)
                g_edges = edf[edf[cfg.graph_id_header] == gid]

                if len(g_nodes) == 0:
                    raise ValueError(f"PyG - Error: Graph {gid!r} has no nodes.")

                x = torch.tensor(g_nodes[node_feat_cols].values, dtype=torch.float32)

                # PyG edge_index is local and zero-based. TopologicPy CSVs may
                # contain node ids that are strings, GUIDs, or non-contiguous
                # integer ids, so always remap src_id/dst_id to local row indices.
                node_ids = list(g_nodes[cfg.node_id_header].values)
                node_id_to_local = {}
                for local_i, node_id in enumerate(node_ids):
                    key = self._label_key(node_id)
                    if key in node_id_to_local:
                        raise ValueError(f"PyG - Error: Duplicate node_id {node_id!r} in graph {gid!r}.")
                    node_id_to_local[key] = int(local_i)

                edge_pairs = []
                for row_i, row in g_edges.iterrows():
                    src_key = self._label_key(row[cfg.edge_src_header])
                    dst_key = self._label_key(row[cfg.edge_dst_header])
                    if src_key not in node_id_to_local or dst_key not in node_id_to_local:
                        raise ValueError(
                            "PyG - Error: Edge references a node_id that is not present in nodes.csv "
                            f"for graph {gid!r} at edges.csv row {row_i!r}: "
                            f"{row[cfg.edge_src_header]!r} -> {row[cfg.edge_dst_header]!r}."
                        )
                    edge_pairs.append([node_id_to_local[src_key], node_id_to_local[dst_key]])

                if edge_pairs:
                    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
                else:
                    edge_index = torch.empty((2, 0), dtype=torch.long)

                data = Data(x=x, edge_index=edge_index)
                if graph_feat_cols:
                    data.u = torch.tensor(g_row[graph_feat_cols].values[:1], dtype=torch.float32)

                # Preserve graph identity as a tensor-safe index. Full ontology
                # metadata is stored externally in self.ontology_metadata to avoid
                # polluting PyG mini-batches with strings/dictionaries.
                data.topologicpy_graph_index = torch.tensor([int(graph_index)], dtype=torch.long)

                if getattr(cfg, "ontology", True):
                    gid_key = str(gid)
                    graph_meta = self._row_metadata(g_row.iloc[0], graph_meta_cols) if len(g_row) > 0 else {}
                    node_meta = [self._row_metadata(row, node_meta_cols) for _, row in g_nodes.iterrows()]
                    edge_meta = [self._row_metadata(row, edge_meta_cols) for _, row in g_edges.iterrows()]
                    self.ontology_metadata["graphs"][gid_key] = graph_meta
                    self.ontology_metadata["graphs"][int(graph_index)] = graph_meta
                    self.ontology_metadata["nodes"][gid_key] = node_meta
                    self.ontology_metadata["nodes"][int(graph_index)] = node_meta
                    self.ontology_metadata["edges"][gid_key] = edge_meta
                    self.ontology_metadata["edges"][int(graph_index)] = edge_meta

                if len(edge_feat_cols) > 0:
                    if len(g_edges) > 0:
                        data.edge_attr = torch.tensor(g_edges[edge_feat_cols].values, dtype=torch.float32)
                    else:
                        data.edge_attr = torch.empty((0, len(edge_feat_cols)), dtype=torch.float32)

                # graph-level
                if cfg.level == "graph" and not cfg.inference:
                    y_val = g_row[cfg.graph_label_header].values[0] if cfg.graph_label_type == "categorical" else 0.
                    if cfg.graph_label_type == "continuous" and not math.isfinite(float(y_val)):
                        raise ValueError("Regression targets must be finite.")
                    if cfg.graph_label_type == "categorical":
                        data.y = torch.tensor([self._encode_label_value("graph", y_val)], dtype=torch.long)
                    else:
                        data.y = self._Targets(g_row, "graph")


                # node-level
                if cfg.level == "node" and not cfg.inference:
                    y_vals = g_nodes[cfg.node_label_header].values if cfg.node_label_type == "categorical" else np.asarray([])
                    if cfg.node_label_type == "continuous" and not np.isfinite(y_vals.astype(float)).all():
                        raise ValueError("Regression targets must be finite.")
                    if cfg.node_label_type == "categorical":
                        data.y = torch.tensor(self._encode_label_values("node", y_vals), dtype=torch.long)
                    else:
                        data.y = self._Targets(g_nodes, "node")
                    data.train_mask, data.val_mask, data.test_mask = self._get_or_make_node_masks(g_nodes) if cfg.split_mode == "entity" else (torch.zeros(len(g_nodes), dtype=torch.bool),) * 3

                # edge-level
                if cfg.level == "edge" and not cfg.inference:
                    y_vals = g_edges[cfg.edge_label_header].values if cfg.edge_label_type == "categorical" else np.asarray([])
                    if cfg.edge_label_type == "continuous" and not np.isfinite(y_vals.astype(float)).all():
                        raise ValueError("Regression targets must be finite.")
                    if cfg.edge_label_type == "categorical":
                        data.edge_y = torch.tensor(self._encode_label_values("edge", y_vals), dtype=torch.long)
                    else:
                        data.edge_y = self._Targets(g_edges, "edge")
                    data.edge_train_mask, data.edge_val_mask, data.edge_test_mask = self._get_or_make_edge_masks(g_edges) if cfg.split_mode == "entity" else (torch.zeros(len(g_edges), dtype=torch.bool),) * 3

                self._AttachSimulation(data, g_row, g_nodes, g_edges)
                for key, value in self._external_simulation.get(self._label_key(gid), {}).items():
                    setattr(data, key, copy.deepcopy(value))
                self._identities.append({"graph_id": self._clean_metadata_value(gid),
                                         "node_ids": [self._clean_metadata_value(v) for v in node_ids],
                                         "edge_endpoints": [(self._clean_metadata_value(row[cfg.edge_src_header]), self._clean_metadata_value(row[cfg.edge_dst_header])) for _, row in g_edges.iterrows()]})
                self.data_list.append(data)

            # output dimensionality
            if cfg.inference and cfg.task == "regression" and not self._freeze_num_outputs:
                self._num_outputs = len(cfg.target_columns) or 1
            if not cfg.inference and not getattr(self, "_freeze_num_outputs", False):
                if cfg.level == "graph":
                    self._num_outputs = self._infer_num_classes(gdf[cfg.graph_label_header].values) if cfg.graph_label_type == "categorical" else (len(cfg.target_columns) or 1)
                elif cfg.level == "node":
                    self._num_outputs = self._infer_num_classes(ndf[cfg.node_label_header].values) if cfg.node_label_type == "categorical" else (len(cfg.target_columns) or 1)
                elif cfg.level == "edge":
                    self._num_outputs = self._infer_num_classes(edf[cfg.edge_label_header].values) if cfg.edge_label_type == "categorical" else (len(cfg.target_columns) or 1)
                elif cfg.level == "link":
                    self._num_outputs = 1
                else:
                    raise ValueError("Unsupported level.")

            if not self.data_list:
                raise ValueError("Dataset contains no graphs.")
            self._raw_tensors = [{key: getattr(d, key).clone() for key in ("x", "edge_attr", "u", "y", "edge_y") if torch.is_tensor(getattr(d, key, None))} for d in self.data_list]
            # Cache input dimension for convenience (used when saving checkpoints)
            self._in_dim = int(self.data_list[0].x.shape[1]) if self.data_list else self._in_dim

        def _get_or_make_node_masks(self, g_nodes):
            return self._Masks(g_nodes, "node")

        def _get_or_make_edge_masks(self, g_edges):
            return self._Masks(g_edges, "edge")

        # ----------------------------
        # Holdout split (graph-level)
        # ----------------------------
        def _split_holdout(self):
            cfg = self.config
            if cfg.level in ["node", "edge", "link"] and cfg.split_mode != "graph":
                self.train_set = self.data_list
                self.val_set = self.data_list
                self.test_set = self.data_list
                return

            n = len(self.data_list)
            if n == 0:
                self.train_set = []
                self.val_set = []
                self.test_set = []
                return

            def _counts(total):
                counts = list(self._SplitCounts(total))
                active = [i for i,ratio in enumerate(cfg.split) if ratio > 0]
                if total >= len(active):
                    for i in active:
                        if counts[i] == 0:
                            donor = max(active,key=lambda j:counts[j])
                            counts[donor] -= 1; counts[i] += 1
                return tuple(counts)

            # Group-aware holdout split for graph-level tasks.
            # All graphs with the same group value stay in the same split.
            if getattr(cfg, "holdout_group_by", None):
                if self.graph_df is None:
                    raise ValueError("PyG - Error: graph_df is not loaded, so group-aware holdout split cannot be computed.")
                if cfg.holdout_group_by not in self.graph_df.columns:
                    raise ValueError(
                        f"PyG - Error: holdout_group_by='{cfg.holdout_group_by}' was not found in graphs.csv columns."
                    )

                gids = list(self.graph_df[cfg.graph_id_header].unique())
                gid_to_index = {gid: i for i, gid in enumerate(gids)}
                group_series = (
                    self.graph_df[[cfg.graph_id_header, cfg.holdout_group_by]]
                    .drop_duplicates(subset=[cfg.graph_id_header])
                    .set_index(cfg.graph_id_header)[cfg.holdout_group_by]
                )

                groups = [group_series.loc[gid] for gid in gids]
                if pd.isna(groups).any():
                    raise ValueError("Group identifiers must not be missing.")
                unique_groups = list(dict.fromkeys(groups))
                if cfg.shuffle:
                    random.Random(cfg.random_state).shuffle(unique_groups)

                g_train, g_val, g_test = _counts(len(unique_groups))
                train_groups = set(unique_groups[:g_train])
                val_groups = set(unique_groups[g_train:g_train + g_val])
                test_groups = set(unique_groups[g_train + g_val:g_train + g_val + g_test])

                train_idx = [gid_to_index[gid] for gid, grp in zip(gids, groups) if grp in train_groups]
                val_idx = [gid_to_index[gid] for gid, grp in zip(gids, groups) if grp in val_groups]
                test_idx = [gid_to_index[gid] for gid, grp in zip(gids, groups) if grp in test_groups]

                if len(train_idx) == 0:
                    raise ValueError("PyG - Error: group-aware holdout split produced an empty training set.")
                if (len(val_idx) == 0) and (len(unique_groups) >= 3) and cfg.split[1] > 0:
                    raise ValueError("PyG - Error: group-aware holdout split produced an empty validation set.")
                if (len(test_idx) == 0) and (len(unique_groups) >= 3) and cfg.split[2] > 0:
                    raise ValueError("PyG - Error: group-aware holdout split produced an empty test set.")
            else:
                idx = list(range(n))
                if cfg.shuffle:
                    random.Random(cfg.random_state).shuffle(idx)

                n_train, n_val, n_test = _counts(n)
                train_idx = idx[:n_train]
                val_idx = idx[n_train:n_train + n_val]
                test_idx = idx[n_train + n_val:n_train + n_val + n_test]

            self.train_set = [self.data_list[i] for i in train_idx]
            self.val_set = [self.data_list[i] for i in val_idx]
            self.test_set = [self.data_list[i] for i in test_idx]
            if cfg.level in ("node", "edge"):
                self._SetGraphMasks(train_idx, val_idx, test_idx)

        # --------------
        # Model building
        # --------------
        def _build_model(self):
            with self._RandomContext(reset=True):
                self._BuildModel()
            self._epochs_completed = 0
            self._early_state = None
            self._continuation_state = None
            self._training_signature = None
            self.history = {"train_loss": [], "val_loss": []}

        def _BuildModel(self):
            cfg = self.config

            if getattr(self, "_in_dim", None) is not None: # Prefer to get the input dimension from the stored value, not the current dataset.
                in_dim = int(self._in_dim)
            else:
                in_dim = int(self.data_list[0].x.shape[1])  # fallback (training-time)

            if cfg.model_name != "default":
                factory = self._MSAModel if cfg.model_name == "msa_gnn" else self._ModelFactories.get(cfg.model_name)
                if factory is None:
                    raise ValueError(f"Register custom model {cfg.model_name!r} before constructing/loading it.")
                context = {"node_dim": in_dim, "edge_dim": self._edge_dim,
                           "graph_dim": self._graph_dim, "output_dim": self._num_outputs,
                           "level": cfg.level, "task": cfg.task,
                           "target_columns": tuple(cfg.target_columns),
                           "target_units": tuple(cfg.target_units),
                           "position_dim": int(self.data_list[0].pos.shape[1]) if getattr(self.data_list[0], "pos", None) is not None else 0,
                           "options": copy.deepcopy(cfg.model_options or {})}
                self.model = factory(context)
                if not isinstance(self.model, nn.Module): raise ValueError("Model factory must return nn.Module.")
                self.model = self.model.to(self.device)
            else:
                encoder = _GNNBackbone(
                    in_dim=in_dim,
                    hidden_dims=cfg.hidden_dims,
                    conv=cfg.conv,
                    activation=cfg.activation,
                    dropout=cfg.dropout,
                    batch_norm=cfg.batch_norm,
                    residual=cfg.residual,
                    edge_dim=self._edge_dim,
                    graph_dim=self._graph_dim
                )

                if cfg.level == "graph":
                    head = _GraphHead(encoder.out_dim, self._num_outputs, pooling=cfg.pooling, dropout=cfg.dropout, graph_dim=self._graph_dim)
                    self.model = nn.ModuleDict({"encoder": encoder, "head": head}).to(self.device)
                elif cfg.level == "node":
                    head = _NodeHead(encoder.out_dim, self._num_outputs, dropout=cfg.dropout)
                    self.model = nn.ModuleDict({"encoder": encoder, "head": head}).to(self.device)
                elif cfg.level == "edge":
                    head = _EdgeHead(encoder.out_dim, self._num_outputs, dropout=cfg.dropout, edge_dim=self._edge_dim)
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
        def Train(self, epochs=None, batch_size=None, resume=False):
            """Train deterministically within an isolated random context.

            epochs specifies additional epochs. resume=True preserves accumulated
            history/epoch count after LoadModel(..., resume_training=True) or an earlier
            Train call. Model weights and optimizer are retained on subsequent calls.
            """
            updates = {k: v for k, v in (("epochs", epochs), ("batch_size", batch_size)) if v is not None}
            if updates: self.SetHyperparameters(**updates)
            self._CheckSupervision("train")
            signature = self._DatasetSignature()
            if resume and self._training_signature is not None and signature != self._training_signature:
                raise ValueError("Cannot resume training on a different dataset or split.")
            level = self.config.level
            has_val = True
            try:
                self._CheckSupervision("val")
            except ValueError:
                if self.config.early_stopping: raise
                has_val = False
            train_data, val_data = self._DataForSplit("train"), self._DataForSplit("val")
            size = self.config.batch_size if level == "graph" else 1
            train_loader = self._Loader(train_data, batch_size=size, shuffle=self.config.shuffle)
            val_loader = self._Loader(val_data, batch_size=size, shuffle=False) if has_val else None
            self._continue_history = bool(resume)
            continuation = getattr(self, "_continuation_state", None)
            if resume and continuation:
                self.model.load_state_dict(continuation["model"])
                self.optimizer.load_state_dict(continuation["optimizer"])
                for group in self.optimizer.param_groups:
                    group.update(lr=self.config.lr, weight_decay=self.config.weight_decay)
                self._rng_state = copy.deepcopy(continuation["rng"])
            with self._RandomContext():
                result = self._Fit(train_loader, val_loader, getattr(self, "_train_epoch_" + level), getattr(self, "_eval_epoch_" + level))
            self._training_signature = signature
            return result

        def CrossValidate(self, k_folds=None, epochs=None, batch_size=None):
            """Independent graph/group folds with fold-specific scalers and validation.

            Supports graph tasks and node/edge tasks using split_mode='graph'. The
            caller's trained model, data, optimizer, scalers, RNG and history are retained.
            Reports include fold memberships and fitted normalization statistics.
            """
            if self.config.inference or self.config.level == "link" or not self._GraphSplit():
                raise ValueError("CrossValidate requires labelled graph-level or whole-graph node/edge tasks.")
            from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, GroupShuffleSplit, train_test_split
            cfg = copy.deepcopy(self.config)
            if epochs is not None: cfg.epochs = epochs
            if batch_size is not None: cfg.batch_size = batch_size
            if k_folds is not None: cfg.k_folds = k_folds
            self._ValidateConfig(cfg)
            indices = np.arange(len(self.data_list))
            groups = None
            if cfg.holdout_group_by:
                groups = self.graph_df[cfg.holdout_group_by].to_numpy()
                if cfg.k_folds > len(set(groups)): raise ValueError("k_folds exceeds the number of groups.")
                outer = GroupKFold(cfg.k_folds).split(indices, groups=groups)
            elif cfg.k_stratify and cfg.task == "classification" and cfg.level == "graph":
                y = np.asarray([int(d.y.item()) for d in self.data_list])
                if min(np.bincount(y)) < cfg.k_folds:
                    raise ValueError("Each class must have at least k_folds samples for stratification.")
                outer = StratifiedKFold(cfg.k_folds, shuffle=cfg.k_shuffle,
                                       random_state=cfg.random_state if cfg.k_shuffle else None).split(indices, y)
            else:
                if cfg.k_folds > len(indices): raise ValueError("k_folds exceeds the number of graphs.")
                outer = KFold(cfg.k_folds, shuffle=cfg.k_shuffle,
                              random_state=cfg.random_state if cfg.k_shuffle else None).split(indices)
            folds = []
            for fi, (outer_train, test_idx) in enumerate(outer):
                train_idx, val_idx = outer_train, np.asarray([], dtype=int)
                seed = (cfg.random_state + fi) % 2**32
                if cfg.early_stopping:
                    if groups is not None:
                        if len(set(groups[outer_train])) < 2:
                            raise ValueError("Early stopping requires at least two training groups per fold.")
                        tr, va = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(outer_train, groups=groups[outer_train]))
                        train_idx, val_idx = outer_train[tr], outer_train[va]
                    else:
                        if len(outer_train) < 2: raise ValueError("Early stopping requires an inner validation set.")
                        train_idx, val_idx = train_test_split(outer_train, test_size=.2, random_state=seed)
                fold = copy.copy(self)
                fold.config = copy.deepcopy(cfg)
                fold.config.random_state = seed
                fold.data_list = copy.deepcopy(self.data_list)
                fold._scalers, fold._scalers_frozen, fold._rng_state = {}, False, None
                fold.train_set = [fold.data_list[i] for i in train_idx]
                fold.val_set = [fold.data_list[i] for i in val_idx]
                fold.test_set = [fold.data_list[i] for i in test_idx]
                if cfg.level in ("node", "edge"): fold._SetGraphMasks(train_idx, val_idx, test_idx)
                fold._PrepareNormalization(refit=True)
                fold._build_model()
                size = cfg.batch_size if cfg.level == "graph" else 1
                train_loader = fold._Loader(fold.train_set, batch_size=size, shuffle=cfg.shuffle)
                val_loader = fold._Loader(fold.val_set, batch_size=size, shuffle=False) if len(val_idx) else None
                fold._continue_history = False
                with fold._RandomContext():
                    fold._Fit(train_loader, val_loader, getattr(fold, "_train_epoch_" + cfg.level), getattr(fold, "_eval_epoch_" + cfg.level))
                loader = fold._Loader(fold.test_set, batch_size=size, shuffle=False)
                if cfg.level == "graph": metrics = fold._metrics_graph(loader, prefix="")
                else:
                    metrics = getattr(fold, "_metrics_" + cfg.level)(loader, split="test")
                    metrics = {key.removeprefix("test_"): value for key, value in metrics.items()}
                metrics.update(fold=fi, train_indices=train_idx.tolist(), val_indices=val_idx.tolist(), test_indices=test_idx.tolist(), normalization=copy.deepcopy(fold._scalers))
                folds.append(metrics)
            report = {"fold_metrics": folds}
            for key, value in folds[0].items():
                if key == "fold" or not isinstance(value, (int, float)): continue
                values = [f[key] for f in folds]
                report["mean_" + key], report["std_" + key] = float(np.mean(values)), float(np.std(values))
            self.cv_report = report
            return report

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
            self._CheckSupervision("val")
            cfg = self.config
            if cfg.level == "graph":
                loader = self._Loader(self.val_set, batch_size=cfg.batch_size, shuffle=False)
                return self._metrics_graph(loader, prefix="val_")
            if cfg.level == "node":
                loader = self._Loader(self.data_list, batch_size=1, shuffle=False)
                return self._metrics_node(loader, split="val")
            if cfg.level == "edge":
                loader = self._Loader(self.data_list, batch_size=1, shuffle=False)
                return self._metrics_edge(loader, split="val")
            if cfg.level == "link":
                loader = self._Loader(self.data_list, batch_size=1, shuffle=False)
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
            self._CheckSupervision("test")
            cfg = self.config
            if cfg.level == "graph":
                loader = self._Loader(self.test_set, batch_size=cfg.batch_size, shuffle=False)
                return self._metrics_graph(loader, prefix="test_")
            if cfg.level == "node":
                loader = self._Loader(self.data_list, batch_size=1, shuffle=False)
                return self._metrics_node(loader, split="test")
            if cfg.level == "edge":
                loader = self._Loader(self.data_list, batch_size=1, shuffle=False)
                return self._metrics_edge(loader, split="test")
            if cfg.level == "link":
                loader = self._Loader(self.data_list, batch_size=1, shuffle=False)
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
            - Classification ``pred`` and ``y_true`` contain original CSV labels.
              ``class_index`` / ``y_true_index`` retain numeric indices, and
              ``class_labels`` defines probability and logit column order.
            """
            if self.model is None:
                raise ValueError("PyG - Error: Model is not initialised. Train or LoadModel first.")

            cfg = self.config
            split = (split or "all").lower().strip()

            # ----------------------------
            # Graph-level predictions
            # ----------------------------
            if cfg.level == "graph":
                id_to_index = {id(d): i for i, d in enumerate(self.data_list)}
                if split == "train":
                    data_src = list(self.train_set or [])
                    selected_indices = [id_to_index.get(id(d), -1) for d in data_src]
                elif split in ("val", "valid", "validation"):
                    data_src = list(self.val_set or [])
                    selected_indices = [id_to_index.get(id(d), -1) for d in data_src]
                elif split == "test":
                    data_src = list(self.test_set or [])
                    selected_indices = [id_to_index.get(id(d), -1) for d in data_src]
                elif split == "all":
                    data_src = list(self.data_list)
                    selected_indices = list(range(len(data_src)))
                else:
                    raise ValueError("PyG - Error: split must be one of {'train','val','test','all'} for graph-level.")

                loader = self._Loader(data_src, batch_size=cfg.batch_size, shuffle=False)

                self.model.eval()
                all_logits, all_probs, all_pred, all_true, all_idx, all_emb = [], [], [], [], [], []

                with torch.no_grad():
                    for batch in loader:
                        # Preserve dataset indices independently of collated object identity.
                        batch = batch.to(self.device)
                        logits, node_emb = self._Forward(batch)

                        if return_embeddings:
                            # store per-batch node embeddings on CPU
                            if node_emb is None: raise ValueError("Custom model did not return embeddings.")
                            all_emb.append(node_emb.detach().cpu())

                        if cfg.task == "regression":
                            pred = self._RegressionArray(logits)
                            all_pred.extend(pred.tolist())
                            if hasattr(batch, "y") and batch.y is not None:
                                all_true.extend(self._RegressionArray(batch.y).tolist())
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

                # Index for graph-level: original self.data_list indices for the chosen subset.
                all_idx = selected_indices[:len(all_pred)]

                out = {
                    "pred": np.array(all_pred),
                    "index": np.array(all_idx),
                    "graph_ids": [self._identities[i]["graph_id"] for i in all_idx]
                }
                if getattr(cfg, "ontology", True):
                    md = getattr(self, "ontology_metadata", {}) or {}
                    graph_md = md.get("graphs", {}) or {}
                    selected_metadata = []
                    selected_metadata = [copy.deepcopy(graph_md.get(i, {})) for i in all_idx]
                    out["metadata"] = selected_metadata
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
                        setattr(d, pred_key, self._DecodePredictions({"pred": np.asarray([out["pred"][i]])}, "graph")["pred"][0])
                        if "y_true" in out:
                            setattr(d, "y_true", self._DecodePredictions({"pred": np.asarray([out["y_true"][i]])}, "graph")["pred"][0])
                    if "prob" in out:
                        for i, d in enumerate(self.data_list):
                            setattr(d, prob_key, out["prob"][i])
                    if "logits" in out:
                        for i, d in enumerate(self.data_list):
                            setattr(d, logits_key, out["logits"][i])

                if attach_to_data and split == "all" and return_embeddings and all_emb:
                    embeddings = torch.cat(all_emb, dim=0).split([d.num_nodes for d in self.data_list])
                    for d, embedding in zip(self.data_list, embeddings):
                        setattr(d, emb_key, embedding.numpy())
                return self._DecodePredictions(out, cfg.level)

            # ----------------------------
            # Node-level predictions
            # ----------------------------
            if cfg.level == "node":
                loader = self._Loader(self.data_list, batch_size=1, shuffle=False)
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
                        logits, node_emb = self._Forward(data)

                        if return_embeddings:
                            if node_emb is None: raise ValueError("Custom model did not return embeddings.")
                            emb_cpu = node_emb.detach().cpu()
                            emb_per_graph.append(emb_cpu)

                        if cfg.task == "regression":
                            pred_all = self._PhysicalTensor(logits)
                            if mask_name is not None:
                                mask = getattr(data, mask_name)
                                pred = pred_all[mask]
                                y_true = self._PhysicalTensor(data.y[mask]) if getattr(data, "y", None) is not None else None
                            else:
                                pred = pred_all
                                y_true = self._PhysicalTensor(data.y) if getattr(data, "y", None) is not None else None

                            preds_per_graph.append(pred.detach().cpu().numpy())
                            if y_true is not None:
                                true_per_graph.append(y_true.detach().cpu().numpy())
                            if return_logits:
                                logits_per_graph.append((logits[getattr(data, mask_name)] if mask_name else logits).detach().cpu().numpy())
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
                                logits_per_graph.append((logits[getattr(data, mask_name)] if mask_name else logits).detach().cpu().numpy())

                        if attach_to_data:
                            # Attach full-length predictions to the underlying Data object (CPU arrays).
                            # For split!=all, we still attach full-length arrays for convenience.
                            d0 = self.data_list[int(data.topologicpy_graph_index.reshape(-1)[0])]
                            if cfg.task == "regression":
                                setattr(d0, pred_key, self._DecodePredictions({"pred": pred_all.detach().cpu().numpy()}, cfg.level)["pred"])
                                if return_logits:
                                    setattr(d0, logits_key, logits.detach().cpu().numpy())
                            else:
                                setattr(d0, pred_key, self._DecodePredictions({"pred": pred_all.detach().cpu().numpy()}, cfg.level)["pred"])
                                if return_probs:
                                    setattr(d0, prob_key, probs_all.detach().cpu().numpy())
                                if return_logits:
                                    setattr(d0, logits_key, logits.detach().cpu().numpy())
                            if return_embeddings:
                                setattr(d0, emb_key, node_emb.detach().cpu().numpy())

                out = {"pred": preds_per_graph, "identities": copy.deepcopy(self._identities)}
                if getattr(cfg, "ontology", True):
                    md = getattr(self, "ontology_metadata", {}) or {}
                    out["metadata"] = [copy.deepcopy((md.get("nodes", {}) or {}).get(i, [])) for i in range(len(preds_per_graph))]
                if len(true_per_graph) > 0:
                    out["y_true"] = true_per_graph
                if mask_name is not None:
                    out["mask"] = mask_name
                    out["index"] = [getattr(d, mask_name).nonzero(as_tuple=False).reshape(-1).cpu().numpy() for d in self.data_list]
                    if "metadata" in out:
                        out["metadata"] = [[rows[int(j)] for j in ix] if rows else [] for rows, ix in zip(out["metadata"], out["index"])]
                if return_probs and len(probs_per_graph) > 0:
                    out["prob"] = probs_per_graph
                if return_logits and len(logits_per_graph) > 0:
                    out["logits"] = logits_per_graph
                if return_embeddings and len(emb_per_graph) > 0:
                    out["emb"] = emb_per_graph
                return self._DecodePredictions(out, cfg.level)

            # ----------------------------
            # Edge-level predictions
            # ----------------------------
            if cfg.level == "edge":
                loader = self._Loader(self.data_list, batch_size=1, shuffle=False)
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
                        logits, node_emb = self._Forward(data)

                        if return_embeddings:
                            if node_emb is None: raise ValueError("Custom model did not return embeddings.")
                            emb_per_graph.append(node_emb.detach().cpu())

                        if cfg.task == "regression":
                            pred_all = self._PhysicalTensor(logits)
                            if mask_name is not None:
                                mask = getattr(data, mask_name)
                                pred = pred_all[mask]
                                y_true = self._PhysicalTensor(data.edge_y[mask]) if getattr(data, "edge_y", None) is not None else None
                            else:
                                pred = pred_all
                                y_true = self._PhysicalTensor(data.edge_y) if getattr(data, "edge_y", None) is not None else None

                            preds_per_graph.append(pred.detach().cpu().numpy())
                            if y_true is not None:
                                true_per_graph.append(y_true.detach().cpu().numpy())
                            if return_logits:
                                logits_per_graph.append((logits[getattr(data, mask_name)] if mask_name else logits).detach().cpu().numpy())
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
                                logits_per_graph.append((logits[getattr(data, mask_name)] if mask_name else logits).detach().cpu().numpy())

                        if attach_to_data:
                            d0 = self.data_list[int(data.topologicpy_graph_index.reshape(-1)[0])]
                            setattr(d0, pred_key, self._DecodePredictions({"pred": pred_all.detach().cpu().numpy()}, cfg.level)["pred"])
                            if cfg.task != "regression" and return_probs:
                                setattr(d0, prob_key, probs_all.detach().cpu().numpy())
                            if return_logits:
                                setattr(d0, logits_key, logits.detach().cpu().numpy())
                            if return_embeddings:
                                setattr(d0, emb_key, node_emb.detach().cpu().numpy())

                out = {"pred": preds_per_graph, "identities": copy.deepcopy(self._identities)}
                if getattr(cfg, "ontology", True):
                    md = getattr(self, "ontology_metadata", {}) or {}
                    out["metadata"] = [copy.deepcopy((md.get("edges", {}) or {}).get(i, [])) for i in range(len(preds_per_graph))]
                if len(true_per_graph) > 0:
                    out["y_true"] = true_per_graph
                if mask_name is not None:
                    out["mask"] = mask_name
                    out["index"] = [getattr(d, mask_name).nonzero(as_tuple=False).reshape(-1).cpu().numpy() for d in self.data_list]
                    if "metadata" in out:
                        out["metadata"] = [[rows[int(j)] for j in ix] if rows else [] for rows, ix in zip(out["metadata"], out["index"])]
                if return_probs and len(probs_per_graph) > 0:
                    out["prob"] = probs_per_graph
                if return_logits and len(logits_per_graph) > 0:
                    out["logits"] = logits_per_graph
                if return_embeddings and len(emb_per_graph) > 0:
                    out["emb"] = emb_per_graph
                return self._DecodePredictions(out, cfg.level)

            # ----------------------------
            # Link prediction
            # ----------------------------
            if cfg.level == "link":
                loader = self._Loader(self.data_list, batch_size=1, shuffle=False)
                self.model.eval()

                use_split = split
                if use_split == "all":
                    use_split = "test"
                if use_split not in ("train", "val", "test"):
                    raise ValueError("PyG - Error: split must be one of {'train','val','test'} for link prediction.")


                all_score, all_true, all_logits = [], [], []
                with torch.no_grad():
                    for data in loader:
                        use = self._LinkData(data, use_split)
                        logits, node_emb = self._Forward(use)
                        score = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
                        y = use.edge_label.detach().cpu().numpy().reshape(-1)
                        all_logits.extend(logits.detach().cpu().numpy().reshape(-1).tolist())
                        all_score.extend(score.tolist())
                        all_true.extend(y.tolist())

                score_arr = np.array(all_score)
                true_arr = np.array(all_true)
                pred_arr = (score_arr >= float(threshold)).astype(int)

                out = {"score": score_arr, "pred": pred_arr, "y_true": true_arr, "threshold": float(threshold)}
                if return_logits:
                    out["logits"] = np.asarray(all_logits)
                return out

            raise ValueError("Unsupported level.")

        # --------
        # Epochs
        # --------
        def _loss_from_logits(self, logits, y, task):
            if task == "regression":
                expected = y.numel()
                if y.ndim > 1 and logits.shape != y.shape:
                    raise ValueError("Prediction and target component shapes do not match.")
                if logits.numel() != expected:
                    raise ValueError("Prediction and target shapes do not match.")
                return self.criterion(logits.reshape_as(y).float(), y.float())
            return self.criterion(logits, y.long())

        def _train_epoch_graph(self, loader):
            self.model.train()
            losses = []
            for batch in loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                logits, node_emb = self._Forward(batch)
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
                logits, node_emb = self._Forward(batch)
                loss = self._loss_from_logits(logits, batch.y, self.config.task)
                losses.append(float(loss.detach().cpu()))
            return float(np.mean(losses)) if losses else 0.0

        def _train_epoch_node(self, loader):
            if self.config.model_name == "msa_gnn" and (self.config.model_options or {}).get("rollout_steps",1) > 1:
                return self._MSAMultiStepEpoch(loader)
            self.model.train()
            losses = []
            for data in loader:
                data = data.to(self.device)
                if self.config.model_name == "msa_gnn": data = self._MSANoise(data)
                self.optimizer.zero_grad()
                logits, node_emb = self._Forward(data)
                mask = data.train_mask
                if not bool(mask.any()):
                    continue
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
                logits, node_emb = self._Forward(data)
                mask = data.val_mask
                if not bool(mask.any()):
                    continue
                loss = self._loss_from_logits(logits[mask], data.y[mask], self.config.task)
                losses.append(float(loss.detach().cpu()))
            return float(np.mean(losses)) if losses else 0.0

        def _train_epoch_edge(self, loader):
            self.model.train()
            losses = []
            for data in loader:
                data = data.to(self.device)
                self.optimizer.zero_grad()
                logits, node_emb = self._Forward(data)
                mask = data.edge_train_mask
                if not bool(mask.any()):
                    continue
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
                logits, node_emb = self._Forward(data)
                mask = data.edge_val_mask
                if not bool(mask.any()):
                    continue
                loss = self._loss_from_logits(logits[mask], data.edge_y[mask], self.config.task)
                losses.append(float(loss.detach().cpu()))
            return float(np.mean(losses)) if losses else 0.0

        def _train_epoch_link(self, loader):
            self.model.train()
            losses = []
            for data in loader:
                train_data = self._LinkData(data, "train")
                if train_data.edge_label.numel() == 0:
                    continue
                self.optimizer.zero_grad()
                logits, node_emb = self._Forward(train_data)
                loss = self.criterion(logits, train_data.edge_label.float())
                loss.backward()
                self._apply_gradients()
                losses.append(float(loss.detach().cpu()))
            return float(np.mean(losses)) if losses else 0.0

        @torch.no_grad()
        def _eval_epoch_link(self, loader):
            self.model.eval()
            losses = []
            for data in loader:
                val_data = self._LinkData(data, "val")
                if val_data.edge_label.numel() == 0:
                    continue
                logits, node_emb = self._Forward(val_data)
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
                out, node_emb = self._Forward(batch)
                if self.config.task == "regression":
                    y_true.extend(self._RegressionArray(batch.y).tolist())
                    y_pred.extend(self._RegressionArray(out).tolist())
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
                out, node_emb = self._Forward(data)
                mask = getattr(data, mask_name)
                if self.config.task == "regression":
                    y_true.extend(self._RegressionArray(data.y[mask]).tolist())
                    y_pred.extend(self._RegressionArray(out)[mask.cpu().numpy()].tolist())
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
                out, node_emb = self._Forward(data)
                mask = getattr(data, mask_name)
                if self.config.task == "regression":
                    y_true.extend(self._RegressionArray(data.edge_y[mask]).tolist())
                    y_pred.extend(self._RegressionArray(out)[mask.cpu().numpy()].tolist())
                else:
                    probs = F.softmax(out, dim=-1)
                    y_true.extend(data.edge_y[mask].detach().cpu().numpy().tolist())
                    y_pred.extend(probs.argmax(dim=-1)[mask].detach().cpu().numpy().tolist())
            return np.array(y_true), np.array(y_pred)

        @torch.no_grad()
        def _predict_link(self, loader, split_name: str):
            self.model.eval()
            y_true, y_score = [], []
            for data in loader:
                use = self._LinkData(data, split_name)
                logits, node_emb = self._Forward(use)
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
            if accuracy_score is None or precision_recall_fscore_support is None:
                raise ImportError("scikit-learn is required for classification metrics.") from _SKLEARN_IMPORT_ERROR
            if len(y_true) == 0:
                return {f"{prefix}accuracy": 0.0, f"{prefix}precision": 0.0, f"{prefix}recall": 0.0, f"{prefix}f1": 0.0}
            acc = float(accuracy_score(y_true, y_pred))
            prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
            return {
                f"{prefix}accuracy": float(acc),
                f"{prefix}precision": float(prec),
                f"{prefix}recall": float(rec),
                f"{prefix}f1": float(f1),
            }

        def _regression_metrics(self, y_true, y_pred, prefix=""):
            if mean_absolute_error is None or mean_squared_error is None or r2_score is None:
                raise ImportError("scikit-learn is required for regression metrics.") from _SKLEARN_IMPORT_ERROR
            if len(y_true) == 0: raise ValueError("No supervised samples to evaluate.")
            mae = float(mean_absolute_error(y_true, y_pred))
            rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
            r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else 0.
            result = {prefix + "mae": mae, prefix + "rmse": rmse, prefix + "r2": r2}
            if self._num_outputs > 1:
                for i, name in enumerate(self.config.target_columns):
                    result[prefix + "mae_" + name] = float(mean_absolute_error(y_true[:, i], y_pred[:, i]))
                    result[prefix + "rmse_" + name] = float(math.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])))
                    result[prefix + "r2_" + name] = float(r2_score(y_true[:, i], y_pred[:, i])) if len(y_true) >= 2 else 0.
            return result

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
                                minValue: int = None,
                                maxValue: int = None,
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
                                baseFontSize: int = 16,
                                tickFontSize: int = 14,
                                titleFontSize: int = 22,
                                axisTitleFontSize: int = 16,
                                annotationFontSize: int = 18,
                                grayScale: bool = False,
                                mantissa: int = 6):
            """
            Returns a Plotly Figure of the confusion matrix of the inference. Actual categories are displayed on the X-Axis,
            Predicted categories are displayed on the Y-Axis.

            Parameters
            ----------
            split : str , optional
                Which split(s) to evaluate. Options are: {"train","val","validate","validation","test","all"}. Default is "test".
            normalize : bool, optional
                If True, row-normalize the confusion matrix. Default is False.
            minValue : float , optional
                The desired minimum value to use for the color scale. If set to None, the minimum value found in the input data will be used.
            maxValue : float , optional
                The desired maximum value to use for the color scale. If set to None, the maximum value found in the input data will be used.
            title : str , optional
                The desired title to display. Default is "Confusion Matrix".
            xTitle : str , optional
                The desired X-axis title to display. Default is "Actual Categories".
            yTitle : str , optional
                The desired Y-axis title to display. Default is "Predicted Categories".
            width : int , optional
                The desired width of the figure. Default is 950.
            height : int , optional
                The desired height of the figure. Default is 500.
            showScale : bool , optional
                If set to True, a color scale is shown on the right side of the figure. Default is True.
            colorScale : str , optional
                The desired type of plotly color scales to use (e.g. "Viridis", "Plasma"). Default is "Viridis".
            colorSamples : int , optional
                The number of discrete color samples to use for displaying the data. Default is 10.
            backgroundColor : list or str , optional
                The desired background color (see docstring above). Default is transparent.
            marginLeft, marginRight, marginTop, marginBottom : int , optional
                Plot margins in pixels.
            baseFontSize : int , optional
                The base font size. Default is 16.
            tickFontSize : int , optional
                The tick font size. Default is 14.
            titleFontSize : int , optional
                The title font size. Default is 22.
            axisTitleFontSize : int , optional
                The axis title font size. Default is 16.
            annotationFontSize : int , optional
                The annotation font size. Default is 18.
            grayScale : bool , optional
                If set to True, the figure is rendered in grayscale. Default is False.
            mantissa : int , optional
                The desired length of the mantissa. Default is 6.

            Returns
            -------
            plotly.graph_objects.Figure
                The created plotly figure.
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
                        loader = self._Loader(self.train_set, batch_size=self.config.batch_size, shuffle=False)
                    elif one_split == "val":
                        loader = self._Loader(self.val_set, batch_size=self.config.batch_size, shuffle=False)
                    else:
                        loader = self._Loader(self.test_set, batch_size=self.config.batch_size, shuffle=False)
                    return self._predict_graph(loader)

                elif self.config.level == "node":
                    loader = self._Loader(self.data_list, batch_size=1, shuffle=False)
                    mask = "train_mask" if one_split == "train" else ("val_mask" if one_split == "val" else "test_mask")
                    return self._predict_node(loader, mask)

                else:  # edge
                    loader = self._Loader(self.data_list, batch_size=1, shuffle=False)
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

                # Round
                cm = np.round(cm, decimals=mantissa)

                # Enforce exact row sum = 1
                col_sums = cm.sum(axis=0)
                diff = 1.0 - col_sums

                # Add correction to the largest element in each column
                for j in range(cm.shape[1]):
                    i = np.argmax(cm[j])
                    cm[i, j] += diff[i]

            if title is None:
                title = f"Confusion Matrix ({split_l})"

            if minValue is None:
                minValue = 0.0
            if maxValue is None:
                maxValue = 1.0 if normalize else float(np.max(cm)) if cm.size else 1.0

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
                annotationFontSize = annotationFontSize,
                grayScale = grayScale
            )

            n = len(labels_str)
            tickvals = list(range(n))
            fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=labels_str)
            fig.update_yaxes(tickmode="array", tickvals=tickvals, ticktext=labels_str)

            return fig

        def PlotParity(self,
                split: str = "test",
                title: str = None,
                xTitle: str = "Actual Values",
                yTitle: str = "Predicted Values",
                showIdentity: bool = True,
                showBestFit: bool = True,
                dotSize: int = 6,
                dotColor: str = "blue",
                lineColor: str = "red",
                width: int = 800,
                height: int = 600,
                theme: str = "default",
                backgroundColor: str = 'rgba(0,0,0,0)',
                marginLeft: int = 0,
                marginRight: int = 0,
                marginTop: int = 40,
                marginBottom: int = 0, component=None):
            """
            Plot a parity / correlation plot for regression tasks by delegating to
            ``Plotly.FigureByCorrelation``.

            Parameters
            ----------
            split : {"train", "val", "validate", "validation", "test", "all"}, optional
                Which split to evaluate. Default is ``"test"``.
            title : str, optional
                Custom plot title. If None, an automatic title is generated.
            xTitle : str, optional
                The X-axis title. Default is ``"Actual Values"``.
            yTitle : str, optional
                The Y-axis title. Default is ``"Predicted Values"``.
            showIdentity : bool, optional
                If set to true, shows the 45 degree line.
            showBestFit : bool, optional
                If set to True, draws the best fit line through the data.
            dotSize : int, optional
                The marker size
            dotColor : str, optional
                Dot color passed to ``Plotly.FigureByCorrelation``.
            lineColor : str, optional
                Best-fit line color passed to ``Plotly.FigureByCorrelation``.
            width : int, optional
                Figure width in pixels.
            height : int, optional
                Figure height in pixels.
            theme : str, optional
                Plotly theme. Options are ``"dark"``, ``"light"``, ``"default"``.
            backgroundColor : str, optional
                Figure background color.
            marginLeft : int, optional
                Left margin in pixels.
            marginRight : int, optional
                Right margin in pixels.
            marginTop : int, optional
                Top margin in pixels.
            marginBottom : int, optional
                Bottom margin in pixels.

            Returns
            -------
            plotly.graph_objects.Figure
                A correlation figure of actual vs predicted values.

            Raises
            ------
            ValueError
                If called when ``config.task`` is not ``"regression"`` or when
                ``config.level`` is ``"link"``.
            RuntimeError
                If no regression labels are found for the requested split(s).

            Notes
            -----
            - For node/edge regression, the method uses the corresponding boolean masks
            on each graph and aggregates across all graphs.
            - This method relies on :meth:`_predict_graph`, :meth:`_predict_node`,
            and :meth:`_predict_edge`.
            - ``show_identity``, ``show_best_fit``, and ``point_size`` are kept only
            for API compatibility. The delegated Plotly method always shows the
            best-fit line and 45-degree line, and does not expose point size.  
            """
            if go is None or px is None:
                raise ImportError("Plotly is required for plotting methods. Install plotly to use this feature.")

            import numpy as np
            from topologicpy.Plotly import Plotly

            if self.config.task != "regression" or self.config.level == "link":
                raise ValueError("Parity plot is only available for regression (graph/node/edge).")

            if self._num_outputs > 1 and component is None:
                raise ValueError("Select a component for a vector regression parity plot.")
            if isinstance(component, str):
                if component not in self.config.target_columns: raise ValueError("Unknown target component.")
                component = self.config.target_columns.index(component)
            if component is not None and (not isinstance(component, int) or not 0 <= component < self._num_outputs):
                raise ValueError("Invalid target component index.")
            def select(values):
                values = np.asarray(values)
                return values[:, component] if values.ndim == 2 and component is not None else values.reshape(-1)
            split_l = (split or "test").lower()
            if split_l in ("validate", "validation"):
                split_l = "val"

            def _y_for_one_split(one_split: str):
                if self.config.level == "graph":
                    if one_split == "train":
                        loader = self._Loader(self.train_set, batch_size=self.config.batch_size, shuffle=False)
                    elif one_split == "val":
                        loader = self._Loader(self.val_set, batch_size=self.config.batch_size, shuffle=False)
                    else:
                        loader = self._Loader(self.test_set, batch_size=self.config.batch_size, shuffle=False)
                    return self._predict_graph(loader)

                elif self.config.level == "node":
                    loader = self._Loader(self.data_list, batch_size=1, shuffle=False)
                    mask = "train_mask" if one_split == "train" else ("val_mask" if one_split == "val" else "test_mask")
                    return self._predict_node(loader, mask)

                else:  # edge
                    loader = self._Loader(self.data_list, batch_size=1, shuffle=False)
                    mask = "edge_train_mask" if one_split == "train" else ("edge_val_mask" if one_split == "val" else "edge_test_mask")
                    return self._predict_edge(loader, mask)

            # Collect y_true / y_pred
            if split_l == "all":
                y_true_all = []
                y_pred_all = []
                for s in ("train", "val", "test"):
                    yt, yp = _y_for_one_split(s)
                    if yt is not None and len(yt):
                        y_true_all.append(select(yt))
                        y_pred_all.append(select(yp))
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
                y_true = select(y_true)
                y_pred = select(y_pred)

            # Lightweight metrics for title
            eps = 1e-12
            y_true_f = y_true.astype(float)
            y_pred_f = y_pred.astype(float)
            mae = float(np.mean(np.abs(y_pred_f - y_true_f)))
            rmse = float(np.sqrt(np.mean((y_pred_f - y_true_f) ** 2)))
            ss_res = float(np.sum((y_true_f - y_pred_f) ** 2))
            ss_tot = float(np.sum((y_true_f - np.mean(y_true_f)) ** 2))
            r2 = 1.0 - ss_res / (ss_tot + eps)

            if title is None:
                title = f"Parity Plot ({split_l}) — MAE={mae:.4g}, RMSE={rmse:.4g}, R²={r2:.4g}"

            fig = Plotly.FigureByCorrelation(
                actual=y_true_f.tolist(),
                predicted=y_pred_f.tolist(),
                title=title,
                xTitle=xTitle,
                yTitle=yTitle,
                showIdentity=showIdentity,
                showBestFit=showBestFit,
                dotSize=dotSize,
                dotColor=dotColor,
                lineColor=lineColor,
                width=width,
                height=height,
                theme=theme,
                backgroundColor=backgroundColor,
                marginLeft=marginLeft,
                marginRight=marginRight,
                marginTop=marginTop,
                marginBottom=marginBottom
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
            if not include_config and self._scalers:
                raise ValueError("Normalization requires a full checkpoint; use include_config=True.")
            if not path.lower().endswith(".pt"):
                path = path + ".pt"

            if self.model is None:
                raise RuntimeError("PyG.SaveModel - Error: No model to save.")

            if include_config:
                # Save full config fields (not only Summary) so future options can be restored too.
                cfg = self.config
                cfg_fields = {k: getattr(cfg, k) for k in vars(cfg).keys()}
                payload = {
                    "feature_support_version": 1,
                    "normalization": copy.deepcopy(self._scalers),
                    "csv_signature": self._CSVSignature(),
                    "simulation_inputs": copy.deepcopy(self._external_simulation),
                    "msa_auto_hierarchy": copy.deepcopy(getattr(self,"_msa_auto_hierarchy",{})),
                    "training_state": {"optimizer": self.optimizer.state_dict(),
                                       "rng": copy.deepcopy(self._rng_state),
                                       "epochs_completed": self._epochs_completed,
                                       "history": copy.deepcopy(self.history),
                                       "early_stopping": copy.deepcopy(getattr(self, "_early_state", None)),
                                       "continuation": copy.deepcopy(getattr(self, "_continuation_state", None)),
                                       "dataset_signature": self._training_signature or self._DatasetSignature()},
                    "state_dict": self.model.state_dict(),
                    "config_fields": cfg_fields,
                    "summary": self.Summary(),
                    # Schema required for safe inference on unseen datasets
                    "in_dim": int(self._in_dim) if getattr(self, "_in_dim", None) is not None else int(self.data_list[0].x.shape[1]) if self.data_list else None,
                    "num_outputs": int(self._num_outputs),
                    "feature_schema": self._current_feature_schema() if (self.graph_df is not None and self.nodes_df is not None and self.edges_df is not None) else None,
                    "ontology_metadata": copy.deepcopy(getattr(self, "ontology_metadata", {}) or {}),
                    "label_encoders": copy.deepcopy(getattr(self, "_label_encoders", {}) or {}),
                    "label_decoders": copy.deepcopy(getattr(self, "_label_decoders", {}) or {}),
                }
                torch.save(payload, path)
            else:
                torch.save(self.model.state_dict(), path)


        def LoadModel(self, path: str, strict: bool = True, rebuild_from_checkpoint: bool = True, trusted: bool = False, resume_training: bool = False):
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
            trusted : bool, optional
                Permit unrestricted deserialisation only after safe loading fails.
                Default is False. Enable only for a checkpoint from a trusted source.
            resume_training : bool, optional
                Restore optimizer, random state, epoch/history and early-stopping
                continuation. Requires the same labelled dataset and split. Call
                Train(..., resume=True) to append additional epochs.

            Returns
            -------
            None
            """
            if self.model is None:
                # In case someone creates PyG without model building (unlikely), build now.
                self._build_model()

            try:
                obj = torch.load(path, map_location=self.device, weights_only=True)
            except Exception as error:
                if not trusted:
                    raise ValueError("Checkpoint could not be loaded safely. Use trusted=True only for a checkpoint you trust.") from error
                try:
                    obj = torch.load(path, map_location=self.device, weights_only=False)
                except TypeError:
                    obj = torch.load(path, map_location=self.device)

            if isinstance(obj, dict) and "state_dict" in obj:
                state = obj["state_dict"]
                schema = obj.get("feature_schema") or {}
                if not obj.get("feature_support_version") and (schema.get("edge_feat_cols") or schema.get("graph_feat_cols")):
                    raise ValueError("This legacy checkpoint ignored edge/graph features. Retrain it to use feature-aware models.")
                if rebuild_from_checkpoint:
                    cfg = copy.deepcopy(self.config)
                    for key, value in obj.get("config_fields", {}).items():
                        if hasattr(cfg, key) and key not in {"inference", "use_gpu", "verbose"}:
                            setattr(cfg, key, value)
                    self._ValidateConfig(cfg)
                    self.config = cfg
                self._in_dim = obj.get("in_dim", self._in_dim)
                self._num_outputs = int(obj.get("num_outputs", self._num_outputs))
                self._freeze_num_outputs = True
                self._feature_schema = obj.get("feature_schema", self._feature_schema)
                self._scalers = copy.deepcopy(obj.get("normalization", {}))
                self._scalers_frozen = True
                if obj.get("label_encoders"):
                    self._label_encoders = copy.deepcopy(obj["label_encoders"])
                    self._label_decoders = copy.deepcopy(obj["label_decoders"])
                    self._freeze_label_encoders = True
                if obj.get("csv_signature") == self._CSVSignature():
                    self._msa_auto_hierarchy = copy.deepcopy(obj.get("msa_auto_hierarchy",{}))
                    for graph_id, attributes in obj.get("simulation_inputs", {}).items():
                        self._external_simulation.setdefault(graph_id, copy.deepcopy(attributes))
                if self._feature_schema:
                    self._build_data_list()
                    self._split_holdout()
                    self._PrepareMSAHierarchies()
                    self._PrepareNormalization()
                if rebuild_from_checkpoint:
                    self._build_model()
            else:
                state = obj

            # Finally load weights
            self.model.load_state_dict(state, strict=strict)
            self.model.to(self.device)
            self.model.eval()
            if resume_training:
                training = obj.get("training_state") if isinstance(obj, dict) else None
                if self.config.inference or not training or not training.get("rng"):
                    raise ValueError("Resuming requires a labelled dataset and a full training checkpoint.")
                if training["dataset_signature"] != self._DatasetSignature():
                    raise ValueError("Cannot resume training on a different dataset or split.")
                self.optimizer.load_state_dict(training["optimizer"])
                self._rng_state = copy.deepcopy(training["rng"])
                self._epochs_completed = int(training["epochs_completed"])
                self._early_state = copy.deepcopy(training.get("early_stopping"))
                self._continuation_state = copy.deepcopy(training.get("continuation"))
                self.history = copy.deepcopy(training["history"])
                self._training_signature = training["dataset_signature"]

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
