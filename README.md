# TopologicPy

<img src="https://topologic.app/wp-content/uploads/2023/02/topologicpy-logo-no-loop.gif" alt="topologicpy logo" width="250" loop="1">

# An AI-Powered Spatial Modelling and Analysis Software Library for Architecture, Engineering, and Construction

## Introduction
Welcome to **TopologicPy** (rhymes with *apple pie*). TopologicPy is a powerful, open-source Python library for spatial modelling, topology, graph analysis, and computational design in Architecture, Engineering, and Construction (AEC). It enables the creation and analysis of information-rich representations of architectural spaces, buildings, infrastructure, and other spatial systems by integrating **geometry, topology, semantics, graphs, and computation** within a coherent programming framework.

At the heart of TopologicPy is support for **non-manifold topology (NMT)**. Unlike conventional solid-modelling approaches, NMT allows entities of different dimensionalities to coexist within a single coherent model. Vertices, edges, surfaces, volumes, and their assemblies can therefore represent complex spatial and analytical systems while explicitly preserving their connectivity and adjacency relationships. This makes TopologicPy particularly well suited to applications in architectural space planning, building analysis, structural modelling, circulation and navigation, energy and environmental analysis, and other workflows in which relationships between entities are as important as their geometric form.

TopologicPy supports **mixed-dimensional models** naturally. For example, lines may represent beams, columns, circulation paths, or network connections; surfaces may represent walls, slabs, zones, or analytical boundaries; and volumes may represent rooms, buildings, or other spatial regions. These entities can coexist within the same topological structure and carry associated dictionaries and semantic information. This provides a flexible foundation for translating between geometric models, Building Information Models (BIM), analytical models, and graph-based representations.

A major strength of TopologicPy is its ability to transform spatial and topological models into **graphs**. Buildings and other spatial systems can be represented through nodes and edges describing adjacency, accessibility, containment, visibility, connectivity, or other relationships. TopologicPy provides graph algorithms for path finding, centrality, community detection, spatial analysis, and related operations, while also supporting integration with **Graph Machine Learning (GML)** workflows. These capabilities make it possible to investigate spatial configurations computationally and to apply machine-learning techniques to problems such as classification, prediction, pattern recognition, and inference over connected spatial data.

TopologicPy also supports **semantic and knowledge-based representations**. Its graph and ontology capabilities can be used to construct knowledge graphs, represent relationships derived from BIM and IFC data, perform reasoning, and connect spatial models with broader semantic information. This provides a foundation for workflows involving knowledge graphs, rule-based reasoning, graph databases, and emerging approaches such as GraphRAG and AI-assisted spatial analysis.

TopologicPy provides a comprehensive Python **Application Programming Interface (API)** designed for scripting, Jupyter notebooks, computational design workflows, research applications, and integration into larger software systems. Its geometry and topology capabilities are currently powered primarily by **PythonOCC/OpenCASCADE**, providing access to the mature Open CASCADE geometric modelling kernel through Python. TopologicPy also maintains support for alternative topology backends, allowing the Python API to remain increasingly independent of any single underlying geometry engine.

Interoperability is an important part of the library. TopologicPy supports widely used formats and workflows including **IFC, BREP, OBJ, STL, DXF, JSON, CSV, RDF/TTL, and HBJSON**, among others. Its IFC capabilities enable geometric, topological, spatial, and semantic information to be extracted from Building Information Models and transformed into structures suitable for analysis, graph generation, reasoning, and downstream computational workflows.

TopologicPy is distributed as open-source software under the **[GNU Lesser General Public License (LGPL)](https://opensource.org/license/lgpl-3-0)**. This allows it to be used, extended, embedded, and integrated into both research and professional workflows without dependence on proprietary subscription-based platforms. The library is designed to provide an extensible computational foundation on which researchers, designers, engineers, educators, and software developers can build their own spatial analysis and design tools.

TopologicPy can be used interactively in Python and Jupyter environments, incorporated into automated analysis pipelines, or embedded within larger applications. By bringing together **geometry, non-manifold topology, graph computation, semantics, and artificial intelligence**, TopologicPy provides a flexible platform for computationally understanding and working with the spatial relationships that underpin the built environment.

## Installation

### Recommended: Conda

TopologicPy can be installed from conda-forge:

```bash
conda create -n topologicpy -c conda-forge topologicpy
conda activate topologicpy
```

### Pip

TopologicPy can also be installed from PyPI:

```bash
pip install topologicpy --upgrade
```

## Prerequisites

TopologicPy requires Python 3.8 or newer and currently supports Python versions below 3.15.

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

These dependencies are installed automatically when TopologicPy is installed.

### Geometry and Topology Backend

TopologicPy requires a geometry and topology backend.

When TopologicPy is installed from **conda-forge**, the recommended
[pythonocc-core](https://github.com/tpaviot/pythonocc-core) backend is installed automatically:

```bash
conda create -n topologicpy -c conda-forge topologicpy
conda activate topologicpy
```

## How to start using TopologicPy
1. Open your favourite python editor ([jupyter notebook](https://jupyter.org/) is highly recommended)
1. Type 'import topologicpy'
1. Start using the API


## Ontology and Semantic Web Support

topologicpy now includes a formal ontology specification that provides a semantic framework for representing geometry, topology, graphs, spatial relationships, building information, provenance, and analytical metrics.

The ontology enables interoperability with:

* RDF / RDFS / OWL
* BOT (Building Topology Ontology)
* Brick Schema
* IFC and Linked Building Data workflows
* Graph databases such as Neo4j and Kùzu
* GraphRAG and AI reasoning systems

The canonical namespace is:

`@prefix top: <http://w3id.org/topologicpy#> .`

The ontology is persistently identified through w3id.org and physically hosted through GitHub Pages.

### Ontology Resources

* Ontology namespace: `http://w3id.org/topologicpy#`
* Ontology document: `http://w3id.org/topologicpy`
* Current ontology specification:
  `https://wassimj.github.io/topologicpy/ontology/topologicpy.ttl`
* Ontology source folder:
  `https://github.com/wassimj/topologicpy/tree/main/ontology`

### Example

```ttl
@prefix top: <http://w3id.org/topologicpy#> .

:room_101 a top:Room ;
    top:hasArea "24.6"^^xsd:double ;
    top:adjacentTo :corridor_1 .

:wall_12 a top:Wall ;
    top:bounds :room_101 .
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
