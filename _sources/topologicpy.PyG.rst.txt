topologicpy.PyG module
======================

.. automodule:: topologicpy.PyG
   :members:
   :undoc-members:
   :show-inheritance:

Usage and compatibility notes
-----------------------------

* Classification ``Predict`` results contain original CSV labels in ``pred``
  and ``y_true``. Numeric indices are available in ``class_index`` and
  ``y_true_index``. ``class_labels`` gives the probability/logit column order.
* Use ``ByCSVPath(path, inference=True, ...)`` for unlabelled input, then
  ``LoadModel`` with a full training checkpoint before predicting. Training and
  evaluation require a labelled instance. Masks are unnecessary for inference.
* Checkpoints preserve training feature order and categorical encoders. Missing
  required features raise ``ValueError``. Features intentionally absent from a
  new dataset may be named in ``optional_feature_columns`` during training;
  only these columns are filled with zero when loading a checkpoint.
* ``LoadModel`` uses safe deserialisation by default. Legacy checkpoints that
  require unrestricted deserialisation must be explicitly loaded with
  ``trusted=True``, and only when their source is trusted.
* Graph features from ``graphs.csv`` are stored as one row per graph in ``u``.
  A learned projection conditions node embeddings for every prediction level;
  graph predictions also concatenate these features with pooled embeddings.
* Edge features from ``edges.csv`` condition message passing in every backbone.
  GATv2 uses its native edge-aware attention. GraphSAGE and GCN add a learned
  edge-conditioned message residual, averaged over incoming neighbours. Edge
  prediction heads additionally concatenate each edge's own attributes with its
  endpoint embeddings. Link prediction uses attributes on the observed message
  edges; held-out candidate edges do not supply attributes to the predictor.
* Existing node-only checkpoints remain compatible. Older checkpoints that
  supplied edge/graph features but ignored them require retraining; loading them
  now raises an explicit error. Coordinates must still be explicitly encoded as
  numeric features.
* Link prediction retains one seeded split for each graph. Split settings cannot
  be changed on a live instance; construct a new instance for a new experiment.
  Validation and test samples remain fixed across epochs and repeated calls.
* ``CrossValidate`` honours ``holdout_group_by`` and creates a separate inner
  validation set when early stopping is enabled. It restores the caller's model,
  optimizer and history, including when a fold fails. Fold reports include the
  train, validation and test indices for inspection.
* Early stopping is supported for graph, node, edge and link tasks. The best
  weights and optimizer state are restored. Empty evaluation splits raise an
  explicit error rather than returning misleading scores.
* ``SetHyperparameters`` applies optimizer updates to the active optimizer and
  validates settings before use. Architecture changes rebuild the model;
  changes to the task, labels or CSV schema require a new instance. Unknown
  configuration keys now raise ``ValueError``.
* Split ratios are honoured, including zero ratios. Counts use largest-remainder
  allocation; small datasets may have empty validation/test splits. Explicit
  masks must be valid booleans and mutually exclusive.

Regression tests
----------------

``tests/test_PyG.py`` exercises real PyTorch/PyG training and prediction for all
three backbones and graph/node/edge classification and regression, plus link
prediction. It covers fixed link splits, safe checkpoint round trips, original
label decoding, unlabelled inference, feature schemas, prediction attachment,
configuration updates, grouped cross-validation, early stopping, invalid masks
and synthetic learning convergence.

Graph and edge feature inputs
-----------------------------

Use numeric columns with the configured feature prefix in each CSV. For example,
``graphs.csv`` may include ``feat_inlet_speed`` and ``feat_temperature``;
``edges.csv`` may include ``feat_distance`` and ``feat_aperture_area``. Existing
``TGraph.ExportToCSV`` options ``graphFeaturesKeys`` and ``edgeFeaturesKeys``
produce these inputs from graph and edge dictionaries. Feature dimensions are
inferred automatically; no additional model flag is needed.

Graph, node and edge classification/regression and link prediction all use the
feature-aware encoder. Training, validation, testing, cross-validation, prediction
and checkpoint reload share the same forward-input contract. Graph contexts are
kept separate when multiple graphs are batched. Required/optional feature schema
checks apply to graph and edge features as well as node features.

The regression suite also verifies nonzero feature gradients, prediction changes
when only graph or edge attributes change, batched-versus-independent results,
feature-aware checkpoints and learning tasks where the only predictive signal is
in graph or edge attributes.

Simulation-ready training
-------------------------

Vector targets and complete simulation splits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Regression accepts ``target_columns`` from the table corresponding to the
prediction level. For example, node targets ``vx``, ``vy`` and ``vz`` belong in
``nodes.csv``. Multi-component predictions retain shape ``[samples, components]``;
scalar targets keep their existing one-dimensional output. Standard metrics also
report each named component. ``PlotParity(component="vx")`` selects a component
instead of combining unlike quantities in one scatter plot.

For node/edge prediction, ``split_mode="entity"`` retains the original within-graph
masks. ``split_mode="graph"`` holds out entire graphs and applies each graph's
split to all its node/edge targets, overriding per-entity CSV masks. Set
``holdout_group_by="simulation_id"`` or a layout/building column in ``graphs.csv``
to keep every snapshot from a simulation or building in the same subset. Grouped
cross-validation supports these whole-graph node/edge tasks too.

.. code-block:: python

   model = PyG.ByCSVPath(
       path="airflow_dataset",
       level="node", task="regression", nodeLabelType="continuous",
       target_columns=("vx", "vy", "vz"),
       target_units=("m/s", "m/s", "m/s"),
       split_mode="graph", holdout_group_by="simulation_id",
       normalize_features=True, normalize_targets=True,
       random_state=42,
   )
   model.Train(epochs=100)
   metrics = model.Test()

Training-only normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Normalization is optional and disabled by default. Feature scalers and target
scalers are fitted using training data only. Entity-split node/edge tasks use
training masks for the corresponding feature/target rows; whole-graph tasks use
only training graphs. Constant columns receive unit scale. Every cross-validation
fold fits separate scalers using its own inner training subset, excluding both
validation and test data. Inference uses the saved training scalers without
refitting. ``Predict`` and metrics return targets/predictions in physical units;
``return_logits=True`` exposes the raw model outputs in training units.

Full checkpoints store the scaler statistics and target names/units. Exporting
raw weights with ``include_config=False`` is rejected when normalization is active
because weights alone cannot reproduce preprocessing or physical-unit outputs.

Reproducibility and continuation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The seed controls model initialization, training sampling and dropout inside an
isolated random context. Python, NumPy and PyTorch random states outside the
runner are restored after training. Non-shuffling evaluation loaders use local
random generators. ``deterministic=True`` additionally requests deterministic
PyTorch algorithms; this may raise when an operation has no deterministic device
implementation. CPU reproducibility is regression-tested; GPU execution is not.

.. code-block:: python

   model.SaveModel("airflow.pt")
   continued = PyG.ByCSVPath("airflow_dataset", level="node",
                            task="regression", nodeLabelType="continuous",
                            target_columns=("vx", "vy", "vz"))
   continued.LoadModel("airflow.pt", resume_training=True)
   continued.Train(epochs=20, resume=True)  # additional epochs

The full checkpoint retains optimizer state, random states, epoch count/history,
scalers and early-stopping continuation. Early stopping still exposes the best
weights for prediction, while retaining the last training state for exact
continuation. Resume requires the same labelled CSV data, split configuration
and physical inputs; mismatches raise an error. Changing a trained experiment's
split/seed requires a new instance. Changing the seed before training rebuilds
the model consistently.

Structured inputs and model factories
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``PyG.RegisterModel(name, factory)`` registers a named factory that receives a
context dictionary with node/edge/graph/output dimensions, task, target names and
units, position dimension and ``model_options``. It returns a ``torch.nn.Module``
whose ``forward(data)`` returns a prediction tensor, or a dictionary containing
``prediction`` and optional ``embedding``. Classification expects logits;
regression expects training-unit outputs with the target component dimension.
Link models receive the fixed query pairs in ``data.edge_label_index``.

The model receives a PyG ``Data``/``Batch`` with ``x``, ``edge_index``, optional
``edge_attr`` and ``u``, plus any configured simulation fields:

* ``position_columns=("x", "y", "z")`` in nodes.csv populates ``data.pos``.
  All configured coordinate columns must be present or all absent. The default
  backbone does not implicitly add these coordinates to numeric node features.
* ``node_weight_column`` and ``edge_weight_column`` populate physical integration
  weights, such as cell volumes or face areas. Weights are finite and nonnegative.
* ``boundary_column`` populates ``boundary_mask`` from explicit booleans.
* ``time_column`` and ``time_step_column`` in graphs.csv populate ``time`` and
  positive ``dt``. Their defaults are ``time`` and ``dt``.

``SetGraphData(source_graph_id, ...)`` can also attach positions, masks, weights,
time, time step and a tensor dictionary named ``hierarchy``. A supplied hierarchy
forces one graph per batch so coarse indices never receive incorrect fine-node
batch offsets. This carries externally constructed hierarchies; it does not
construct an MSA-GNN hierarchy. Attached inputs are retained in checkpoints and
restored for matching CSV data. New inference geometry retains its own inputs.

Factories must be registered by the application before loading a custom model's
checkpoint. Checkpoints contain the factory name/options and weights, never
executable factory code. Built-in GraphSAGE/GCN/GATv2 models remain the default.

Physical evaluation
~~~~~~~~~~~~~~~~~~~

``EvaluatePhysics(split="test")`` evaluates regression predictions in physical
units, using configured mesh weights and boundary masks. It returns component
MAE/RMSE/R-squared, vector errors and boundary-specific errors, plus declared
units. ``PhysicsMetrics(prediction, truth, ...)`` applies the same checks to
external arrays.

``EvaluateRollout(prediction, truth, weights=..., boundaryMask=...,
componentNames=..., timeStep=...)`` assesses supplied time-major sequences. It
reports per-step metrics, accumulated physical times and RMSE growth. This is
rollout evaluation; an autoregressive simulator will be implemented with MSA-GNN.

Strict input identities
~~~~~~~~~~~~~~~~~~~~~~~

CSV loading rejects duplicate graph records, duplicate node IDs within a graph,
missing identity values, orphan graph references and edges with unknown endpoints.
Parallel edges remain allowed. Prediction outputs retain source graph IDs and
node/edge identities independently of ontology metadata.

``tests/test_PyG_simulation.py`` covers these APIs, including exact resumed versus
uninterrupted CPU training with dropout, early-stopping continuation, vector
regression, fold-specific scalers and weighted physical metrics.

MSA-GNN airflow surrogate
-------------------------

``model_name="msa_gnn"`` selects the built-in multi-scale attention model; no
manual factory registration is required. It is intended for node-level velocity
increment regression on fixed airflow meshes, with two or three target components.
The default architecture uses a latent width of 128, four processor blocks, two
attention heads and GRU aggregation.

The encoder combines current node features, normalized geometric coordinates,
graph conditions and the physical prediction interval. Edge encoding uses relative
positions/distances plus supplied edge attributes. Each processor performs local
fine-graph message passing, cluster aggregation, coarse-graph message passing,
coarse self-attention and a feed-forward residual, followed by learned fine-node
reconstruction with a skip connection. The decoder predicts velocity increments.

This is a paper-inspired implementation, not a claim of reproducing the paper's
numerical results. Deliberate choices include sparse barrier-aware coarsening,
connected shortest-path cluster assignment, deterministic geometric ordering for
GRU inputs, and fine/coarse processing with skip reconstruction in each block.
Message aggregation uses degree-normalized incoming sums. Benchmark these choices
on the intended CFD dataset before drawing accuracy or speed conclusions.

Hierarchy construction
~~~~~~~~~~~~~~~~~~~~~~~

``BuildMSAHierarchy(positions, edgeIndex, nClusters=None, ratio=.1,
maxClusters=128, seed=42)`` constructs a reusable hierarchy. Affinity exists only
on physical mesh edges and is weighted by distance. Sparse spectral embeddings
choose cluster seeds, then shortest-path assignment produces connected clusters.
Disconnected air regions remain separate. Attention is masked between disconnected
components. No dense fine-mesh affinity matrix is constructed.

The returned tensor dictionary contains cluster assignments, coarse edges and
positions, component IDs, and deterministic within-group membership order. The
runner caches it by geometry, connectivity and coarsening settings. Hierarchies
and settings are checkpointed; changed automatic coarsening settings rebuild the
hierarchy. Explicit external hierarchies are respected and validated. One scene
is processed per batch so unrelated scenes cannot share attention or coarse-index
offsets. Packed GRU aggregation materializes only actual hidden-state vectors,
not a dense cluster-count by maximum-group-length feature array.

CFD sequence adapters
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   sequences = [
       {
           "positions": positions,           # [nodes, 2 or 3]
           "edge_index": mesh_edges,         # [2, edges], integer
           "velocity": velocities,           # [frames, nodes, 2 or 3], physical units
           "times": times,                   # [frames], strictly increasing
           "boundary_type": boundary_types,  # 0 interior, 1 wall, 2 inlet, 3 outlet
           "weights": cell_volumes,          # [nodes]; optional
           "graph_features": inlet_settings, # [frames, graph_features]; optional
       },
       # Independent simulations may use different fixed meshes/node counts.
   ]
   model = PyG.ByCFDSequences(
       "airflow_dataset", sequences,
       model_options={"latent_dim": 128, "processor_blocks": 4,
                      "attention_heads": 2, "noise_std": .01,
                      "rollout_steps": 3},
       epochs=100, use_gpu=True,
   )
   model.Train()
   model.SaveModel("airflow.pt")

The adapter writes graphs.csv, nodes.csv, edges.csv and msa_manifest.json. It
creates consecutive velocity-increment targets and stores the current velocities,
static boundary-type encodings and optional static extra_node_features. It makes
mesh connections bidirectional and assigns opposite displacement vectors to
opposite directions. Graph conditions can be numeric arrays or a dictionary of
named scalar/time-varying conditions; named schemas must match across simulations.

Default splitting is 60/20/20 by simulation_id, with training-only feature/target
normalization. Requested nonzero graph holdouts receive a sample/group when the
number of independent groups permits. Fewer independent simulations may leave
some evaluation splits empty; evaluation then raises rather than inventing scores.
Node integration weights default to uniform weights when actual weights are not
provided; uniform weights are not estimates of physical cell volumes.

``ReadCFDSequence("simulation.npz")`` reads consolidated NumPy trajectories using
allow_pickle=False. ``ReadCFDSequence(frame_paths, velocityKey="U",
location="point", boundaryKey=None, weightKey=None, timeStep=.1)`` reads ordered
VTK/VTU files using optional meshio. Supported linear cells are line, triangle,
quad, tetra, hexahedron, wedge and pyramid. Cell mode creates cell-centre nodes and
face-sharing adjacency, retaining cell-centred field values. Point mode retains
point fields and mesh edges. Unknown cell types, missing fields and changing mesh
identity/order are rejected. Supply integration weights through a field or the
sequence dictionary; they are not inferred from cell geometry.

``ByCFDPaths`` imports independent lists of CFD frame paths and delegates to the
same adapter. ``ByCFDDataset(path)`` reloads an exported dataset using its saved
schema, split and model options. To resume, load a full checkpoint with
resume_training=True and then call Train(..., resume=True). overwrite=True replaces
only the adapter's four named dataset files and leaves unrelated files intact.
The current CSV/PyG data path is memory-resident; large production CFD datasets
must fit the available memory. Remeshing, native OpenFOAM case parsing, thermal
fields and moving geometric boundaries are outside this first implementation.

Training and ablations
~~~~~~~~~~~~~~~~~~~~~~~

Optional noise_std perturbs current velocity input features during training,
leaving prescribed boundaries unchanged. Target increments are corrected for the
same physical perturbation, including when input/target scalers differ. A value
of zero disables noise.

rollout_steps=1 uses ordinary one-step training. Larger values train differentiable
short autoregressive windows, using actual next-frame conditions and reapplying
known boundaries. Entire simulations must stay in one split. rollout_loss_decay
weights later steps; its default is 1. Validation/early stopping currently uses
one-step validation loss, while EvaluateRollout measures long-horizon behaviour.

Options attention=False and multiscale=False support ablations. aggregation="mean"
replaces GRU aggregation. For a controlled architecture comparison, use identical
sequences, splits, normalization, training budgets and one-step protocols. The
existing feature-aware GNN can use the same exported increment dataset by loading
it with model_name="default".

Supported model_options are latent_dim, processor_blocks, attention_heads,
max_coarse_nodes, n_clusters, coarsening_ratio, aggregation, attention, multiscale,
dropout, noise_std, rollout_steps, rollout_loss_decay, velocity_feature_columns,
time_steps and allow_unseen_time_step. The adapter supplies velocity-column and
time-step metadata automatically. Manual CSV users must provide velocity-column
metadata before using input noise or autoregressive rollout.

Physical rollout
~~~~~~~~~~~~~~~~~

.. code-block:: python

   result = model.Rollout(
       graphID=initial_graph_id, steps=30,
       boundaryValues=prescribed_next_velocities,  # [steps, nodes, components]
       graphFeatures=known_inlet_conditions,       # [steps, graph_features]
       reference=reference_sequence,               # optional, includes initial state
   )

Rollout repeatedly predicts an increment, restores physical units, updates velocity,
applies prescribed wall/inlet values and re-encodes the next input. Type-3 outlets
are not clamped. Omitted boundaryValues hold the initial prescribed boundary values
fixed. Only prescribed boundary entries are used, even when full-node arrays are
supplied. Graph conditions and boundary values are caller-known inputs, not
predicted future interior reference values.

Returned velocity contains the initial field followed by every predicted field;
increments, physical times, source node IDs and the boundary mask are also returned.
With reference supplied, physical component/vector/weighted/boundary errors and
error growth are added. Inputs are cloned and model mode is restored. Non-finite
rollouts raise. Prediction intervals absent from training are rejected unless the
caller explicitly opts into allow_unseen_time_step=True; doing so does not establish
accuracy at that interval. Boundaries are enforced, but no pressure projection or
explicit incompressibility/PDE constraint is imposed.

Example and regression tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``examples/msa_gnn_airflow.py`` trains on clearly labelled synthetic discrete
flow dynamics, saves a checkpoint and rollout arrays, and writes a self-contained
HTML preview. Run with --compare for the feature-aware GNN and no-attention
multi-scale baselines under shared one-step protocols. Run separately with
--training-rollout-steps 3 to exercise short-rollout training. Synthetic examples
are implementation demonstrations, not CFD or real-building validation.

``tests/test_PyG_MSA.py`` checks sparse connected clustering, permutation behaviour,
attention gradients, packed GRU correctness, learning, two/three-component fields,
VTK/VTU and safe NPZ ingestion, boundary conditions, long rollouts, ablations,
noise correction, exact multi-step checkpoint continuation and reload settings.

Optional dependencies
~~~~~~~~~~~~~~~~~~~~~~

From this source checkout, install the learning and VTK/VTU reader dependencies
explicitly with ``python -m pip install -e ".[pyg,cfd]"``. The pyg extra supplies
PyTorch, PyTorch Geometric and scikit-learn; cfd supplies meshio and google-crc32c. NumPy, SciPy,
pandas and Plotly are core dependencies. No import or method installs packages.
NPZ/in-memory sequence adapters do not require meshio.

Official cylinder-flow verification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``PyG.DownloadMeshGraphNets(path, counts={"train": 8, "valid": 2, "test": 2})``
retrieves bounded prefixes of the official MeshGraphNets cylinder-flow TFRecords.
Each prefix has a manifest recording its source URL, original record indices,
byte count and SHA256. Download and ingestion verify TFRecord CRC32C checksums.
``PyG.ReadMeshGraphNets(record_path, meta_path)`` yields fixed-mesh trajectory
dictionaries without requiring TensorFlow. Optional google-crc32c accelerates
checksum verification; a Python fallback is provided. Triangle edges are made
bidirectional and node weights are mass-lumped triangle areas. Pressure is omitted
by default; includePressure=True also returns the physical vertex pressure field.

The reader maps NORMAL to interior, INFLOW to prescribed inlet, OUTFLOW to outlet,
and other labels to prescribed boundaries. Its supervision_mask selects NORMAL
and OUTFLOW nodes, matching the reference velocity-update supervision. The
boundary values remain fixed during rollout unless supplied by the caller.

ByCFDSequences also accepts split="train"/"valid"/"val"/"test", simulation_id and
frame_offset on every input sequence. These preserve official trajectory membership
and original frame indices when using short windows. All sequences must supply a
split, or none may do so. The generated official_split column is selected through
split_column; a simulation cannot cross splits. CrossValidate refuses to remix
explicit splits. Optional per-node supervision_mask excludes prescribed nodes from
loss and supervised metrics. Feature/target scaling is fitted on training data.

Run the bounded GPU benchmark from a source checkout::

   python examples/msa_gnn_cylinder_flow.py --data cylinder-data --output cylinder-results --epochs 12

It uses the first 8/2/2 official trajectories, three 16-transition training windows
per trajectory, and two validation windows. Checkpoints are selected by validation
20-step rollout error. Test trajectories are evaluated only after selection, from
source frame 1 over 200 predicted steps. The reported cumulative MSE averages both
velocity components, all nodes and predicted frames 1 through the stated horizon,
excluding the initial field, following the reference cfd_eval protocol. A constant
initial-velocity persistence baseline is included. Training uses area weights;
the official-style evaluation metric is unweighted. This compact, short-budget run
is an implementation verification, not a full published benchmark reproduction.

``tests/test_PyG_MeshGraphNets.py`` checks TF Example decoding, CRC validation,
truncation handling, bounded range downloads and cache reuse, official split
isolation, supervision masks and checkpoint reload.

Finite Volume Graph Network (FVGN)
----------------------------------

``model_name="fvgn"`` now defaults to the reference-compatible formulation of
Li et al., *Predicting unsteady incompressible fluid dynamics with finite volume
informed neural network*, Physics of Fluids 36, 043601 (2024),
https://doi.org/10.1063/5.0197425. The reference code is pinned to author commit
668cce81546868c7e4cc1b9340690b6cc296fd50. Numerical fixtures are generated by
executing that code; the TopologicPy implementation uses native PyTorch and does
not require downloading the author repository or installing torch-scatter.

The earlier independent SI-gradient implementation is retained explicitly as
``model_options={"formulation": "physical", ...}``. Existing exported datasets and
checkpoints without a formulation marker are migrated to that legacy mode in
memory, preserving their computational meaning. They are not silently interpreted
as the corrected reference model. ``Summary()["model_formulation"]`` identifies
the active mode. Earlier unstable checkpoints are not repaired by loading them.

Reference architecture and update
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default reference formulation uses:

* Fixed XY-planar triangular meshes, with distinct mesh vertices, cells and faces.
* Cell-centred velocity, velocity differences on faces, relative cell positions,
  face length and nine-class face-type encoding.
* 128 latent features, 15 processor blocks, two hidden layers per MLP, SiLU and
  LayerNorm except at the decoder output.
* Directed split-channel aggregation: the two halves of each face representation
  go to its two vertices. Vertex features are averaged at each triangle cell.
* The reference residual order: the edge MLP receives the cell MLP output before
  the cell residual is added. Cells and faces then receive their residual updates.
* Five decoder channels: normalised face u, v, p and two learned compensation
  components. These are not four physical velocity-gradient components.
* The author's online normalisers, including their initial unit pseudocount, and
  BatchNorm of the geometric length*dt/mean-adjacent-area coefficient.
* Integration in normalised space, with pressure detached from the momentum
  gradient path. The learned compensation is summed as in the author code. The
  resulting velocity increment is inverse-normalised before updating cell state.

This update should not be interpreted as the standalone conservative SI flux
operator. In particular, the learned compensation and normalised geometry do not
establish exact physical local conservation. Pressure learns through its own
supervision. A variance-roundoff clamp and a finite log-loss floor are explicit
numerical safeguards beyond the source implementation; ordinary finite golden
cases match its results.

Graph features condition cell and face encoders. Static node features beyond the
adapter's velocity/type fields and custom adjacency columns beyond its four
geometry columns are optional extensions. Both directed rows of custom adjacency
fields are included. The adapter's four cell-type indicators and geometry columns
are replaced by the reference face-type/geometric encoding, not redundantly fed
into the reference network. Exact author parity is checked with those optional
extensions absent.

Geometry and sequence input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``BuildFVGNMesh(positions, cells, faceTypes=None)`` validates physical geometry and
builds centres, areas, unique faces, owner/neighbour cells, normals, lengths and
cell adjacency. positions is [vertices,2] or constant-Z [vertices,3]; cells is
integer [cells,3]. Faces are indexed lexicographically by their two vertex IDs.
Physical labels are 0=internal, 1=wall, 2=inlet, 3=outlet. Exterior faces default
to walls. Degenerate, duplicate, non-manifold and non-planar cells are rejected.
The function does not globally detect intersections between arbitrary triangles.

``ByFVGNSequences(path, sequences, fieldLocation="cell", overwrite=False, **kwargs)``
exports CFD tables and compressed mesh archives, then constructs the runner.
Each sequence supplies:

* positions, cells and velocity[frames,cells,2].
* Optional times[frames], density (default 1 kg/m3) and kinematic_viscosity
  (default .001 m2/s). These defaults are not inferred from the source simulation.
* face_types[faces] in the public BuildFVGNMesh face order.
* Optional face_velocity[frames,faces,2] for exact face supervision and known
  prescribed boundary inputs. Otherwise face targets are interpolated, and
  boundary_velocity[faces,2] is required for inlets; walls default to zero.
* face_pressure[frames,faces] or pressure[frames,cells] for pressure supervision.
* Optional graph_features, extra_node_features, supervision_mask and
  cell_boundary_type, with one row per cell where applicable.
* Optional official split="train"/"valid"/"val"/"test", simulation_id and
  frame_offset. Simulations cannot cross splits; overlapping windows are rejected.

``fieldLocation="vertex"`` accepts vertex velocity and pressure. Reference cell
velocities use the source converter's squared-distance vertex weights, not a
centroid average. Reference face targets use endpoint means. The implementation
also matches sorted triangle vertices, max-ID-first directed vertex pairs,
greater-(x,y)-first neighbouring cell pairs, exterior self-loops and float32 Heron
areas. CSV graph nodes remain in triangle order with IDs 0..cells-1.

In the reference mode, provided face labels are preserved as encoder metadata
even if they differ from physical boundary topology, enabling literal comparison
with the author's converter. It is the caller's responsibility to provide correct
inlet/outlet/wall labels. The physical mode retains stricter topological label
validation. Re-export mesh geometry through this adapter rather than replacing
cell positions alone.

Training and loss
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   model = PyG.ByFVGNSequences(
       "fvgn_dataset", sequences,
       model_options={"formulation": "reference", "latent_dim": 128,
                      "processor_blocks": 15, "momentum_loss_weight": 10.,
                      "face_velocity_loss_weight": 1., "pressure_loss_weight": 1.,
                      "continuity_loss_weight": 0., "noise_std": .02},
       epochs=100, lr=.001, optimizer="adamw", use_gpu=True,
   )
   model.Train()
   model.SaveModel("fvgn.pt")
   restored = PyG.ByCFDDataset("fvgn_dataset")
   restored.LoadModel("fvgn.pt")

The default reference loss matches the source's direct_mean_loss configuration:
log(10*momentum_MSE + face_velocity_MSE + face_pressure_MSE), using its normalised
quantities. Face velocity supervision includes internal/outlet-labelled faces;
pressure supervision includes all faces. Pressure targets are required by default for training;
``inference=True`` permits prediction without future pressure labels;
explicitly set pressure_loss_weight=0 for velocity-only data. The reference uses an
online face-pressure normaliser, so pressure_scale is rejected in this mode.

continuity_loss_weight defaults to zero to match the executable source defaults.
The author's loss helper accidentally references an undefined continuity variable
and falls back to zero even when configured otherwise. TopologicPy does not copy
that defect: an explicit nonzero weight adds the computed normalised continuity
residual to the loss. That is a deliberate correction, not exact source-loss parity.

noise_std is a physical velocity standard deviation (m/s), matching the reference
noise function. Inputs receive noise and increment targets are corrected. Random
edge feature-direction flips default to enabled, following the source data loader;
random_edge_flip=False disables them for controlled comparisons. Two normaliser
warm-up passes occur before the first optimizer update; configure
normalizer_warmup_passes (default 2) or normalizer_accumulations (default 10000000)
as needed. Validation and inference never update statistics. Checkpoints preserve
online moments, geometry BatchNorm, warm-up completion, optimizer and RNG state.
CPU exact checkpoint continuation is tested. CUDA reductions need not be bit-exact.

Other supported options are velocity_feature_columns, time_steps and
allow_unseen_time_step. Training is one-step and one mesh is processed per batch.
Existing generic feature/target normalisation can remain enabled; physical values
are recovered before the reference online normalisers are applied. These choices
and training budgets must be matched explicitly when comparing published results.

Prediction and boundaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``PredictFVGN(graphID)`` returns physical velocity increment, raw decoded face
velocity/pressure, physical face-divergence diagnostics and normalized_compensation.
It does not expose training targets. Generic Predict/Test continue to report
velocity increments in physical units. No reference viscous gradient is returned,
because its two compensation outputs are not physical gradient components.

``Rollout(graphID, steps, faceBoundaryValues=schedule)`` updates cell velocity
states. schedule is [steps,faces,2] of caller-known wall/inlet inputs. If omitted,
the stored conditions for the first prediction are held fixed. Future interior
reference fields never enter the encoder or update. In the reference formulation,
these conditions constrain the encoder inputs; decoded face velocities are not
hard-clamped before integration, matching the author model. Existing boundaryValues
can additionally prescribe explicit cell velocities. ``graphFeatures`` supplies known
global conditions. Returned fields include face pressure and cell divergence.
Non-finite predictions raise FloatingPointError with the failing step.

Physical formulation
~~~~~~~~~~~~~~~~~~~~~

``formulation="physical"`` retains the earlier full-width aggregation, seven-output
U/p/grad(U) decoder and SI finite-volume integration. FVGNIntegrate remains a public,
differentiable physical operator whose shared internal-face fluxes cancel between
neighbours. This is a separate experimental formulation, not the reference update.
Its earlier small Cylinder Flow trials failed at step 8 and are retained as evidence.
Its noise is in normalised feature units, pressure_scale is a physical pressure
scale, and its loss uses the earlier unlogged physical/normalised residual mixture.

Verification and known reference defects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``tests/test_PyG_FVGN_reference.py`` uses a compact golden fixture generated by the
unchanged author model, integration, loss and autograd code. The fixture's commit,
SHA256 and compatibility details are recorded in tests/data/fvgn-author-golden.json.
Tests cover forward/loss/gradient parity, pressure detachment, frozen inference
statistics, reference-field isolation, feature extensions, weighted vertex and face
interpolation, exact CPU resume and legacy checkpoint migration. The separate
physical tests retain geometric/analytic conservation checks and CUDA coverage.

The literal author mesh converter also has a substring condition in which
"compressible" matches "incompressible". Its default main configuration therefore
leaves some walls/outlets labelled NORMAL. The control records this behaviour and
does not silently repair the author files. The production example derives explicit
wall/inlet/outlet labels from the original dataset. These are distinct preprocessing
configurations; neither should be confused with a verified reproduction of the
published trained model.

``examples/fvgn_cylinder_flow.py`` provides a small official-split diagnostic.
It defaults to reference mode and loads pressure. --formulation physical selects
the earlier variant. --latent-dim 128 --processor-blocks 15 uses the published
width/depth; the lightweight example defaults to 32/2. Failed horizons, persistence,
reference conversion and checkpoint checks are recorded. Finite long rollouts do
not establish physical accuracy. The small training budget is not a publication
benchmark reproduction, and the tested corrected checkpoints remain less accurate
than persistence over the evaluated long horizons.
