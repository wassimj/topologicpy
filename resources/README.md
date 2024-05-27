# topologicpy

<img src="https://topologic.app/wp-content/uploads/2023/02/topologicpy-logo-no-loop.gif" alt="topologicpy logo" width="250" loop="1">

## Ontology Overview
The TopologicPy 3D Spatial Modeling Ontology is designed to facilitate the detailed representation and manipulation of 3D models through non-manifold BREP (Boundary Representation) structures. Its primary purpose is to provide a comprehensive and structured framework for defining the topology of 3D models, encompassing various elements such as vertices, edges, wires, faces, shells, cells, cell complexes, and clusters. The ontology addresses the domain of computational geometry and spatial modeling, specifically targeting applications in CAD (Computer-Aided Design), BIM (Building Information Modeling), and other 3D modeling environments. By standardizing the representation of complex topological relationships, this ontology aims to enhance interoperability, data exchange, and the accuracy of 3D spatial analysis and visualization.

## Prefix and Namespace
The TopologicPy 3D Spatial Modeling Ontology uses the following prefix and namespace:

Prefix: top

Namespace URI: http://w3id.org/topologicpy/#

The full namespace declaration in Turtle (TTL) format is:
```
@prefix top: <http://w3id.org/topologicpy/#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
```

## Authors and Contributors
### Author
Name: Wassim Jabi

Email: wassim.jabi@gmail.com

### Contributor
Name: Theo Dounas

Email: Theo.Dounas@uantwerpen.be

## Class Descriptions

* Topology: The base class for all topological elements.
* Vertex: Represents a point in 3D space.
* Edge: Represents a line segment connecting two vertices.
* Wire: Represents a sequence of connected edges.
* Face: Represents a flat surface bounded by a wire and optionally containing holes.
* Shell: Represents a collection of faces forming a segmented surface.
* Cell: Represents a volumetric element bounded by faces.
* CellComplex: Represents a collection of cells forming a segmented volumetric element.
* Cluster: Represents a group of related objects.
* Grid: Represents a spatial structure dividing space into regular intervals.
* Dictionary: Represents a key-value store for metadata.
* Aperture: Represents an element indicating an opening or hole.
* Context: Represents the environment or settings in which objects exist.
* Vector: Represents a mathematical entity with magnitude and direction.
* Matrix: Represents a rectangular array of numbers used for transformations.
* Graph: Represents a collection of nodes and edges.

## Object Properties
* hasDictionary: Links a topology object to its dictionary.
* hasStartVertex: Links an edge to its start vertex.
* hasEndVertex: Links an edge to its end vertex.
* hasEdges: Links an object to its edges.
* hasVertices: Links an object to its vertices.
* hasExternalBoundary: Links an object to its external boundary.
* hasInternalBoundaries: Links an object to its internal boundaries.
* hasFaces: Links an object to its faces.
* hasWires: Links an object to its wires.
* hasShells: Links an object to its shells.
* hasCells: Links an object to its cells.
* hasClusterMember: Links a cluster to its members.
* hasFreeCells: Links a cluster to its free cells.
* hasFreeShells: Links a cluster to its free shells.
* hasFreeFaces: Links a cluster to its free faces.
* hasFreeWires: Links a cluster to its free wires.
* hasFreeEdges: Links a cluster to its free edges.
* hasFreeVertices: Links a cluster to its free vertices.
* hasKeys: Links a dictionary to its keys.
* hasValues: Links a dictionary to its values.
* Data Properties
* hasX: The X coordinate of a vertex.
* hasY: The Y coordinate of a vertex.
* hasZ: The Z coordinate of a vertex.
* hasAngle: The angle at the start vertex of an edge.
* hasLength: The length of an edge.

## Usage
This ontology provides a structured way to describe 3D models and their topological relationships. It can be used in various applications, including CAD systems, BIM (Building Information Modeling), and other 3D modeling environments.

## License
This ontology is released under the MIT License. For more information, see the LICENSE file.

## Usage
This ontology provides a structured way to describe 3D models and their topological relationships. It can be used in various applications, including CAD systems, BIM (Building Information Modeling), and other 3D modeling environments.
## License
This ontology is licensed under the MIT License.
For more information, see the LICENSE file included with this repository.
## Contact
For any questions or feedback, please contact wassim.jabi@gmail.com.

