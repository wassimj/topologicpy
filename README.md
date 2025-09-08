# topologicpy

<img src="https://topologic.app/wp-content/uploads/2023/02/topologicpy-logo-no-loop.gif" alt="topologicpy logo" width="250" loop="1">

# An AI-Powered Spatial Modelling and Analysis Software Library for Architecture, Engineering, and Construction

## Introduction
Welcome to topologicpy (rhymes with apple pie). topologicpy is an open-source python 3 implementation of [Topologic](https://topologic.app) which is a powerful spatial modelling and analysis software library that revolutionizes the way you design architectural spaces, buildings, and artefacts. Topologic's advanced features enable you to create hierarchical and topological information-rich 3D representations that offer unprecedented flexibility and control in your design process. With the integration of geometry, topology, information, and artificial intelligence, Topologic enriches Building Information Models with Building *Intelligence* Models.

Two of Topologic's main strengths are its support for *defeaturing* and *encoded meshing*. By simplifying the geometry of a model and removing small or unnecessary details not needed for analysis, defeaturing allows for faster and more accurate analysis while maintaining topological consistency. This feature enables you to transform low-quality, heavy BIM models into high-quality, lightweight representations ready for rigorous analysis effortlessly. Encoded meshing allows you to use the same base elements available in your commercial BIM platform to cleanly build 3D information-encoded models that match your exacting specifications.

Topologic's versatility extends to entities with mixed dimensionalities, enabling structural models, for example, to be represented coherently. Lines can represent columns and beams, surfaces can represent walls and slabs, and volumes can represent solids. Even non-building entities like structural loads can be efficiently attached to the structure. This approach creates mixed-dimensional models that are highly compatible with structural analysis simulation software.

Topologic's graph-based representation makes it a natural fit for integrating with Graph Machine Learning (GML), an exciting new branch of artificial intelligence. With GML, you can process vast amounts of connected data and extract valuable insights quickly and accurately. Topologic's intelligent algorithms for graph and node classification take GML to the next level by using the extracted data to classify building typologies, predict associations, and complete missing information in building information models. This integration empowers you to leverage the historical knowledge embedded in your databases and make informed decisions about your current design projects. With Topologic and GML, you can streamline your workflow, enhance your productivity, and achieve your project goals with greater efficiency and precision.

Experience Topologic's comprehensive and well-documented Application Protocol Interface (API) and enjoy the freedom and flexibility that Topologic offers in your architectural design process. Topologic uses cutting-edge C++-based non-manifold topology (NMT) core technology ([Open CASCADE](https://www.opencascade.com/)), and python bindings. Interacting with Topologic is easily accomplished through a command-Line interface and scripts, visual data flow programming (VDFP) plugins for popular BIM software, and cloud-based interfaces through [Streamlit](https://streamlit.io/). You can easily interact with Topologic in various ways to perform design and analysis tasks or even seamlessly customize and embed it in your own in-house software and workflows. Plus, Topologic includes several industry-standard methods for data transport including IFC, OBJ, BREP, HBJSON, CSV, as well serializing through cloud-based services such as [Speckle](https://speckle.systems/).

Topologic’s open-source philosophy and licensing ([AGPLv3](https://www.gnu.org/licenses/agpl-3.0.en.html)) enables you to achieve your design vision with minimal incremental costs, ensuring a high return on investment. You control and own your information outright, and nothing is ever trapped in an expensive subscription model. Topologic empowers you to build and share data apps with ease, giving you the flexibility to choose between local or cloud-based options and the peace of mind to focus on what matters most. 

Join the revolution in architectural design with Topologic. Try it today and see the difference for yourself.

## Installation
topologicpy can be installed using the **pip** command as such:

`pip install topologicpy --upgrade`

## Prerequisites

topologicpy depends on the following python libraries which will be installed automatically from pip:

<details>
<summary>
<b>Expand to view dependencies</b>
</summary>
* [numpy](http://numpy.org) >= 1.24.0
* [scipy](http://scipy.org) >= 1.10.0
* [plotly](http://plotly.com/) >= 5.11.0
* [ifcopenshell](http://ifcopenshell.org/) >=0.7.9
* [ipfshttpclient](https://pypi.org/project/ipfshttpclient/) >= 0.7.0
* [web3](https://web3py.readthedocs.io/en/stable/) >=5.30.0
* [openstudio](https://openstudio.net/) >= 3.4.0
* [topologic_core](https://pypi.org/project/topologic_core/) >= 6.0.6
* [lbt-ladybug](https://pypi.org/project/lbt-ladybug/) >= 0.25.161
* [lbt-honeybee](https://pypi.org/project/lbt-honeybee/) >= 0.6.12
* [honeybee-energy](https://pypi.org/project/honeybee-energy/) >= 1.91.49
* [json](https://docs.python.org/3/library/json.html) >= 2.0.9
* [py2neo](https://py2neo.org/) >= 2021.2.3
* [pyvisgraph](https://github.com/TaipanRex/pyvisgraph) >= 0.2.1
* [specklepy](https://github.com/specklesystems/specklepy) >= 2.7.6
* [pandas](https://pandas.pydata.org/) >= 1.4.2
* [scipy](https://scipy.org/) >= 1.8.1
* [dgl](https://github.com/dmlc/dgl) >= 0.8.2

</details>

## How to start using Topologic
1. Open your favourite python editor ([jupyter notebook](https://jupyter.org/) is highly recommended)
1. Type 'import topologicpy'
1. Start using the API

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

topologicpy: © 2025 Wassim Jabi

Topologic: © 2025 Cardiff University and UCL
