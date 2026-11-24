# TopologicPy

<img src="https://topologic.app/wp-content/uploads/2023/02/topologicpy-logo-no-loop.gif" alt="topologicpy logo" width="250" loop="1">

# An AI-Powered Spatial Modelling and Analysis Software Library for Architecture, Engineering, and Construction

## Introduction
Welcome to **TopologicPy** (rhymes with *apple pie*). TopologicPy is a powerful, open-source Python library for spatial modelling, topology, graph analysis, and computational design in Architecture, Engineering, and Construction (AEC). It enables the creation and analysis of information-rich representations of architectural spaces, buildings, infrastructure, and other spatial systems by integrating **geometry, topology, semantics, graphs, and computation** within a coherent programming framework.

At the heart of TopologicPy is support for **non-manifold topology (NMT)**. Unlike conventional solid-modelling approaches, NMT allows entities of different dimensionalities to coexist within a single coherent model. Vertices, edges, surfaces, volumes, and their assemblies can therefore represent complex spatial and analytical systems while explicitly preserving their connectivity and adjacency relationships. This makes TopologicPy particularly well suited to applications in architectural space planning, building analysis, structural modelling, circulation and navigation, energy and environmental analysis, and other workflows in which relationships between entities are as important as their geometric form.

TopologicPy supports **mixed-dimensional models** naturally. For example, lines may represent beams, columns, circulation paths, or network connections; surfaces may represent walls, slabs, zones, or analytical boundaries; and volumes may represent rooms, buildings, or other spatial regions. These entities can coexist within the same topological structure and carry associated dictionaries and semantic information. This provides a flexible foundation for translating between geometric models, Building Information Models (BIM), analytical models, and graph-based representations.

A major strength of TopologicPy is its ability to transform spatial and topological models into **graphs**. Buildings and other spatial systems can be represented through nodes and edges describing adjacency, accessibility, containment, visibility, connectivity, or other relationships. TopologicPy provides graph algorithms for path finding, centrality, community detection, spatial analysis, and related operations, while also supporting integration with **Graph Machine Learning (GML)** workflows. These capabilities make it possible to investigate spatial configurations computationally and to apply machine-learning techniques to problems such as classification, prediction, pattern recognition, and inference over connected spatial data.

TopologicPy also supports **semantic and knowledge-based representations**. Its graph and ontology capabilities can be used to construct knowledge graphs, represent relationships derived from BIM and IFC data, perform reasoning, and connect spatial models with broader semantic information. This provides a foundation for workflows involving knowledge graphs, rule-based reasoning, graph databases, and emerging approaches such as GraphRAG and AI-assisted spatial analysis.

TopologicPy provides a comprehensive Python **Application Programming Interface (API)** designed for scripting, Jupyter notebooks, computational design workflows, research applications, and integration into larger software systems. Its geometry and topology capabilities are powered by **PythonOCC/OpenCASCADE**, providing access to the mature Open CASCADE Technology (OCCT) geometric modelling kernel through Python.

Interoperability is an important part of the library. TopologicPy supports widely used formats and workflows including **IFC, BREP, OBJ, STL, DXF, JSON, CSV, RDF/TTL, and HBJSON**, among others. Its IFC capabilities enable geometric, topological, spatial, and semantic information to be extracted from Building Information Models and transformed into structures suitable for analysis, graph generation, reasoning, and downstream computational workflows.

TopologicPy is distributed as open-source software under the **[GNU Lesser General Public License (LGPL)](https://opensource.org/license/lgpl-3-0)**. This allows it to be used, extended, embedded, and integrated into both research and professional workflows without dependence on proprietary subscription-based platforms. The library is designed to provide an extensible computational foundation on which researchers, designers, engineers, educators, and software developers can build their own spatial analysis and design tools.

TopologicPy can be used interactively in Python and Jupyter environments, incorporated into automated analysis pipelines, or embedded within larger applications. By bringing together **geometry, non-manifold topology, graph computation, semantics, and artificial intelligence**, TopologicPy provides a flexible platform for computationally understanding and working with the spatial relationships that underpin the built environment.

## Installation

TopologicPy uses [PythonOCC](https://github.com/tpaviot/pythonocc-core) as its sole supported geometry and topology kernel. PythonOCC includes compiled bindings to Open CASCADE Technology (OCCT), so it must be installed in the same Python environment as TopologicPy.

**Python 3.12 is recommended.** The examples below use the PythonOCC 8.0.1 release series. Choose one of the following installation routes:

| Installation route | Recommended for |
| --- | --- |
| Conda / conda-forge | Most users; installs TopologicPy and native dependencies together. |
| Conda + pip | Users who want the latest TopologicPy release from PyPI. |
| Pixi (no Conda application required) | Users who prefer a lightweight, standalone environment manager. |
| Build PythonOCC from source + pip | Advanced users who cannot use conda-forge packages or need their own Python installation. |

### 1. Recommended: Conda

If Conda is not already installed, install [Miniforge](https://github.com/conda-forge/miniforge#download) for your operating system and processor architecture. Miniforge uses conda-forge by default. Existing Miniconda or Anaconda installations can also be used.

Open a terminal (on Windows, use the Miniforge or Anaconda Prompt, or a PowerShell session configured for Conda). Check that Conda is available:

```bash
conda --version
```

Create a dedicated environment with TopologicPy and PythonOCC:

```bash
conda create -n topologicpy --override-channels -c conda-forge python=3.12 "pythonocc-core=8.0.1*" topologicpy
conda activate topologicpy
```

The package manager installs the native OCCT libraries and Python dependencies automatically. Do not install PythonOCC separately with pip.

To update TopologicPy within this environment:

```bash
conda activate topologicpy
conda update --override-channels -c conda-forge topologicpy
```

### 2. Conda + pip: Install the latest TopologicPy from PyPI

The PyPI release may be newer than the conda-forge release. To use it, first create a fresh environment containing PythonOCC, then install TopologicPy with pip:

```bash
conda create -n topologicpy-pip --override-channels -c conda-forge python=3.12 "pythonocc-core=8.0.1*" pip
conda activate topologicpy-pip
python -m pip install --upgrade topologicpy
```

If you already have a suitable Conda environment, activate it and install PythonOCC before installing TopologicPy. A new environment is preferred to avoid dependency conflicts.

```bash
conda install --override-channels -c conda-forge "pythonocc-core=8.0.1*" pip
python -m pip install --upgrade topologicpy
```

Use `python -m pip` rather than a standalone `pip` command to ensure that TopologicPy is installed into the active interpreter. Keep native dependencies managed by Conda; use pip to update TopologicPy in this environment.

### 3. Without Conda: Pixi

[Pixi](https://pixi.prefix.dev/latest/) is a standalone package and environment manager. It does not require Conda, Miniconda, Anaconda or an existing Python installation. It can install the same precompiled PythonOCC and OCCT packages from conda-forge.

#### Install Pixi

Follow the [official Pixi installation instructions](https://pixi.prefix.dev/latest/installation/) for Windows, macOS or Linux. Open a new terminal after installation and check:

```bash
pixi --version
```

#### Create a TopologicPy workspace

Run these commands in a terminal from a directory where you want to keep your project:

```bash
pixi init --channel conda-forge topologicpy-project
cd topologicpy-project
pixi add python=3.12 "pythonocc-core=8.0.1*" topologicpy
```

To use the latest PyPI release instead, start with a fresh workspace and use the following commands in place of the previous block:

```bash
pixi init --channel conda-forge topologicpy-pypi
cd topologicpy-pypi
pixi add python=3.12 "pythonocc-core=8.0.1*"
pixi add --pypi topologicpy
```

Run Python in the workspace:

```bash
pixi run python
```

To run an existing script:

```bash
pixi run python my_script.py
```

To work interactively in the managed environment, run `pixi shell`. To update the workspace dependencies, run `pixi update`. Pixi records the package requirements and resolved versions in its manifest and lock file.

Although Pixi uses conda-forge packages, it does not install or invoke the Conda application. It manages its own Python interpreter; it does not install PythonOCC into your existing system Python or virtual environment.

### 4. Without Conda or conda-forge packages: Build PythonOCC from source

If you need to use an existing standard Python installation and cannot use precompiled conda-forge packages, build PythonOCC against a compatible OCCT release using the [official PythonOCC build guide](https://github.com/tpaviot/pythonocc-core/blob/master/INSTALL.md). This is an advanced installation route, not a pip-only installation.

1. Install Python 3.12 and create a virtual environment:

   ```bash
   python -m venv .venv
   ```

   Activate it on Windows (PowerShell):

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   On macOS or Linux:

   ```bash
   source .venv/bin/activate
   ```

2. Install the platform-specific C++ compiler, CMake, SWIG and other native build dependencies specified in the upstream guide.
3. Obtain or compile a compatible OCCT 8.0.1 release and the corresponding PythonOCC source release. Use the matching tagged source and its build instructions rather than mixing arbitrary versions.
4. Configure and compile PythonOCC using the Python interpreter in your virtual environment. Ensure its installed `OCC` package is on that interpreter's module search path and that the matching OCCT shared libraries are discoverable at runtime. The upstream guide gives platform-specific CMake options and native library configuration for Linux and Windows; other platforms may require additional build work.
5. Once the PythonOCC verification command below succeeds, install TopologicPy:

   ```bash
   python -m pip install --upgrade topologicpy
   ```

#### Why not `pip install pythonocc`?

The modern PythonOCC releases are not officially published as PyPI wheels. The historical package named `pythonocc-core` on PyPI is not a current release and must not be used for this installation. The supported binary distribution is [pythonocc-core on conda-forge](https://anaconda.org/conda-forge/pythonocc-core). Third-party repackaged wheels are not an upstream-supported distribution and are not recommended for this installation.

### 5. Verify the installation

In the activated Conda environment, the Pixi shell, or the virtual environment with a successful native build, run:

```bash
python -c "import sys; print(sys.executable); from OCC.Core.gp import gp_Pnt; print(gp_Pnt(1., 2., 3.).X())"
```

The last line should be `1.0`. This checks that the native bindings can load, not just that a package is listed as installed.

Next, test TopologicPy's geometry construction:

```bash
python -c "from topologicpy.Cell import Cell; from topologicpy.Topology import Topology; c = Cell.Prism(); assert Topology.IsInstance(c, 'Cell'); assert len(Topology.Faces(c)) == 6; print('TopologicPy geometry OK')"
```

If using Pixi without opening `pixi shell`, prefix these commands with `pixi run`.

### 6. Jupyter notebooks (optional)

In an activated Conda or source-build environment:

```bash
python -m pip install jupyterlab ipykernel
python -m ipykernel install --user --name topologicpy --display-name "Python (TopologicPy)"
python -m jupyterlab
```

Select **Python (TopologicPy)** as the notebook kernel. For Pixi, use `pixi add jupyterlab ipykernel` and `pixi run jupyter lab` from the workspace, selecting its Python kernel.

### Troubleshooting

* `ModuleNotFoundError: No module named 'OCC'`: PythonOCC is absent from the interpreter you are running. Check `python -c "import sys; print(sys.executable)"` and activate the correct environment.
* DLL load failures or missing shared libraries: ensure the correct environment is activated and that OCCT and PythonOCC were installed together for the same platform and Python version. Avoid copying compiled modules between environments.
* Package resolution fails: use a fresh Python 3.12 environment and check the [available PythonOCC builds](https://anaconda.org/conda-forge/pythonocc-core/files) for your platform.
* Imports work in a terminal but not in Jupyter: select the kernel associated with the environment in which PythonOCC was installed.

## Prerequisites

Use Python 3.12 for the installation examples above. The usable Python versions depend on the intersection of TopologicPy's package requirements and the available PythonOCC builds. Check the [TopologicPy PyPI metadata](https://pypi.org/project/topologicpy/) and [PythonOCC package builds](https://anaconda.org/conda-forge/pythonocc-core/files) if selecting another Python version.

### Core Dependencies

TopologicPy depends on the following Python libraries:

<details>

<summary><b>Expand to view dependencies</b></summary>

* [numpy](https://numpy.org/) >= 1.18.0
* [scipy](https://scipy.org/) >= 1.4.1
* [pandas](https://pandas.pydata.org/)
* [shapely](https://shapely.readthedocs.io/)
* [plotly](https://plotly.com/python/)
* [lark](https://github.com/lark-parser/lark)
* [webcolors](https://pypi.org/project/webcolors/)
* [nbformat](https://nbformat.readthedocs.io/)

</details>

These Python dependencies are installed automatically by the package manager when TopologicPy is installed. PythonOCC must be available through one of the installation routes above.

### Geometry and Topology Backend

[PythonOCC](https://github.com/tpaviot/pythonocc-core) provides the native geometry and topology implementation. Follow one of the installation routes above to install it before using TopologicPy. The distribution package is named `pythonocc-core`; its Python import namespace is `OCC`.

### Optional features

TopologicPy never installs packages during import or a method call. Install
optional features explicitly in the Python environment running your application:

```bash
python -m pip install "topologicpy[sun]"
python -m pip install "topologicpy[graph]"
python -m pip install "topologicpy[honeybee]"
python -m pip install "topologicpy[kuzu]"
python -m pip install "topologicpy[ladybugdb]"
python -m pip install "topologicpy[ifc]"
```

The `sun` extra provides PyEphem; `graph` provides NetworkX, igraph, Graphviz,
PyVis and RDFLib; `honeybee` provides the Honeybee and Ladybug Tools packages.
The database extras provide their respective drivers, and `ifc` provides
IfcOpenShell. Extras can be combined, for example `topologicpy[sun,graph]`.
Graphviz rendering also requires the Graphviz application to be installed.

Optional dependencies are imported when their feature is used. A missing
dependency raises `ImportError` with an installation hint instead of modifying
the environment or returning an unexplained empty result. NumPy, SciPy and
webcolors remain required dependencies and are installed by the package manager.

## How to start using TopologicPy
1. Open your favourite python editor ([jupyter notebook](https://jupyter.org/) is highly recommended)
1. Type 'import topologicpy'
1. Start using the API


## Ontology and Semantic Web Support

TopologicPy includes a formal ontology for representing topology, geometry, graphs, spatial relationships, building information, provenance, and analytical properties using RDF, RDFS, and OWL.

The ontology is designed to interoperate with:

* RDF / RDFS / OWL
* BOT (Building Topology Ontology)
* Brick Schema
* IFC and Linked Building Data workflows
* Graph databases such as Neo4j and Kùzu
* GraphRAG and AI reasoning systems

The canonical TopologicPy namespace is:

`@prefix top: <http://w3id.org/topologicpy#> .`

The ontology is persistently identified through w3id.org. The canonical source is maintained in the repository under `ontology/topologicpy.ttl`; documentation copies are generated from that source.

### Ontology Resources

* Ontology namespace: `http://w3id.org/topologicpy#`
* Ontology document: `http://w3id.org/topologicpy`
* Current ontology specification:
  `https://wassimj.github.io/topologicpy/ontology/topologicpy.ttl`
* Canonical ontology source:
  `https://github.com/wassimj/topologicpy/blob/main/ontology/topologicpy.ttl`

### Example

```ttl
@prefix top: <http://w3id.org/topologicpy#> .
@prefix inst: <http://w3id.org/topologicpy/instance#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

inst:room_101 a top:Room ;
    rdfs:label "Room 101" ;
    top:area "24.6"^^xsd:double ;
    top:adjacentTo inst:corridor_1 .

inst:corridor_1 a top:Space ;
    rdfs:label "Corridor 1" .
```

For graph relationships, TopologicPy preserves both the structural edge and its semantic predicate:

```ttl
inst:rel_0 a top:Relationship ;
    top:startsAt inst:room_101 ;
    top:endsAt inst:corridor_1 ;
    top:hasPredicate top:adjacentTo .

inst:room_101 top:adjacentTo inst:corridor_1 .
```

### Python Example

```python
from topologicpy.Ontology import Ontology

Ontology.SetClass(cell, "top:Room")
Ontology.SetLabel(cell, "Room 101")
Ontology.SetCategory(cell, "space")

ttl = Ontology.TTLString(cell)
print(ttl)
```



### TGraph graph API

Graph-producing library methods now return `TGraph`. Navigation, database
retrieval, GQL, GraphRAG, plotting, and topology utilities no longer depend on
`Graph.py` or a native geometry Graph class.

Use `TGraph.Vertices(graph)` and `TGraph.Edges(graph)` for records. Request
`asTopologic=True` when geometry objects are needed. `TGraph.ShortestPath`
returns vertex indices by default, and `TGraph.Dictionary` returns a Python
dictionary.

The older `graphType="graph"` and `asTGraph=False` selector spellings are retained
for call compatibility, but they also produce `TGraph`. `Plotly.DataByGraph`
delegates to `Plotly.DataByTGraph`.

See the [TGraph capability guide](docs/TGraph_Capability_Parity.md) for graph
overlays, alternative routes, total depth, weighted analysis, flow details and
visualization styling options.

## API Documentation
API documentation can be found at [https://topologicpy.readthedocs.io](https://topologicpy.readthedocs.io)

## How to cite topologicpy
If you wish to cite the actual software, you can use:

**Jabi, W. (2024). topologicpy. pypi.org. http://doi.org/10.5281/zenodo.11555172**

To cite one of the main papers that defines topologicpy, you can use:

**Jabi, W., & Chatzivasileiadi, A. (2021). Topologic: Exploring Spatial Reasoning Through Geometry, Topology, and Semantics. In S. Eloy, D. Leite Viana, F. Morais, & J. Vieira Vaz (Eds.), Formal Methods in Architecture (pp. 277–285). Springer International Publishing. https://doi.org/10.1007/978-3-030-57509-0_25**

Or you can import the following .bib formatted references into your favourite reference manager
```
@misc{Jabi2025,
   author = {Wassim Jabi},
   doi = {https://doi.org/10.5281/zenodo.11555173},
   title = {topologicpy},
   url = {http://pypi.org/projects/topologicpy},
   year = {2025},
}
```
```
  @inbook{Jabi2021,
   abstract = {Topologic is a software modelling library that supports a comprehensive conceptual framework for the hierarchical spatial representation of buildings based on the data structures and concepts of non-manifold topology (NMT). Topologic supports conceptual design and spatial reasoning through the integration of geometry, topology, and semantics. This enables architects and designers to reflect on their design decisions before the complexities of building information modelling (BIM) set in. We summarize below related work on NMT starting in the late 1980s, describe Topologic’s software architecture, methods, and classes, and discuss how Topologic’s features support conceptual design and spatial reasoning. We also report on a software usability workshop that was conducted to validate a software evaluation methodology and reports on the collected qualitative data. A reflection on Topologic’s features and software architecture illustrates how it enables a fundamental shift from pursuing fidelity of design form to pursuing fidelity of design intent.},
   author = {Wassim Jabi and Aikaterini Chatzivasileiadi},
   city = {Cham},
   doi = {10.1007/978-3-030-57509-0_25},
   editor = {Sara Eloy and David Leite Viana and Franklim Morais and Jorge Vieira Vaz},
   isbn = {978-3-030-57509-0},
   journal = {Formal Methods in Architecture},
   pages = {277-285},
   publisher = {Springer International Publishing},
   title = {Topologic: Exploring Spatial Reasoning Through Geometry, Topology, and Semantics},
   url = {https://link.springer.com/10.1007/978-3-030-57509-0_25},
   year = {2021},
}
```

topologicpy: © 2026 Wassim Jabi

## IDS assessment

Use `from topologicpy.IDS import IDS` to assess IFC model information against
IDS 1.0 without IfcOpenShell or IfcTester. Install the optional XML packages with
`pip install "topologicpy[ids]"`. See [IDS API and verification](docs/IDS.md).

## Analytical spatial models from IFC

`IFC.AnalyticalModel` reconstructs reviewable room-only spatial geometry from IFC
evidence, including independent CellComplexes for separate buildings. The engine
is included in `IFC.py`; no external reconstruction package or IfcOpenShell is
needed. World coordinates are the default. Curved/stepped roofs, boundary evidence,
editable plans, fidelity reports and verified save/load are supported.

```python
from topologicpy.IFC import IFC

result = IFC.AnalyticalModel("building.ifc")
complexes = result.cell_complexes
IFC.AnalyticalModelSave(result, "analytical-model")
```

This feature is experimental: verified geometry does not certify simulation
readiness or full IFC second-level-boundary compliance. See the
[analytical IFC guide](docs/IFC-AnalyticalModel.md) for settings, coordinates,
building separation, limitations and validation.
