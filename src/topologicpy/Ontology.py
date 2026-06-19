# Copyright (C) 2026
# TopologicPy
#
# This file is part of TopologicPy.
#
# TopologicPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.
#
# TopologicPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.

class Ontology:
    """
    A lightweight ontology helper class for TopologicPy.

    The Ontology class provides a dictionary-based semantic layer for TopologicPy
    topologies and graphs. It does not replace TopologicPy dictionaries. Instead,
    it standardises semantic keys such as ontology class, category, label, URI,
    IFC class, IFC GUID, source, and provenance.

    The class intentionally avoids mandatory RDF/OWL dependencies. It can export
    simple RDF Turtle strings/files directly from TopologicPy dictionaries and
    graph connectivity.

    Recommended canonical dictionary keys
    -------------------------------------
    ontology_class : str
        The TopologicPy ontology class, for example "top:Room", "top:Wall",
        "top:AdjacencyGraph".
    ontology_uri : str
        The full URI for the ontology class or entity.
    label : str
        A human-readable label.
    category : str
        A broad category such as "space", "element", "graph", "surface".
    ifc_class : str
        IFC entity type, for example "IfcSpace".
    ifc_guid : str
        IFC GlobalId.
    source : str
        Source file, database, method, or process.
    derived_from : str
        Identifier or URI of the source object.
    generated_by : str
        Name of the method or process that generated the object.
    """

    # -------------------------------------------------------------------------
    # Canonical dictionary keys
    # -------------------------------------------------------------------------

    ONTOLOGY_CLASS_KEY = "ontology_class"
    ONTOLOGY_URI_KEY = "ontology_uri"
    LABEL_KEY = "label"
    CATEGORY_KEY = "category"
    IFC_CLASS_KEY = "ifc_class"
    IFC_GUID_KEY = "ifc_guid"
    SOURCE_KEY = "source"
    DERIVED_FROM_KEY = "derived_from"
    GENERATED_BY_KEY = "generated_by"
    URI_KEY = "uri"

    # -------------------------------------------------------------------------
    # Namespaces
    # -------------------------------------------------------------------------

    NAMESPACES = {
        "bot": "https://w3id.org/bot#",
        "brick": "https://brickschema.org/schema/Brick#",
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "xsd": "http://www.w3.org/2001/XMLSchema#",
        "owl": "http://www.w3.org/2002/07/owl#",
        "top": "http://w3id.org/topologicpy#",
    }

    # -------------------------------------------------------------------------
    # Core ontology hierarchy
    # -------------------------------------------------------------------------

    TOP_SUPERCLASSES = {
        "top:Topology": [],
        "top:Vertex": ["top:Topology"],
        "top:Edge": ["top:Topology"],
        "top:Wire": ["top:Topology"],
        "top:Face": ["top:Topology"],
        "top:Shell": ["top:Topology"],
        "top:Cell": ["top:Topology"],
        "top:CellComplex": ["top:Topology"],
        "top:Cluster": ["top:Topology"],
        "top:Graph": [],
        "top:Grid": [],
        "top:Dictionary": [],
        "top:Context": [],
        "top:Vector": [],
        "top:Matrix": [],
        "top:Project": [],

        "top:Point": ["top:Vertex"],
        "top:Node": ["top:Vertex"],
        "top:Relationship": ["top:Edge"],
        "top:Interface": ["top:Face"],
        "top:Surface": ["top:Face"],
        "top:Aperture": ["top:Face", "top:Element"],

        "top:Zone": ["top:Cell"],
        "top:Space": ["top:Zone"],
        "top:Room": ["top:Space"],
        "top:ThermalZone": ["top:Space"],
        "top:Storey": ["top:Zone"],
        "top:Building": ["top:Zone"],
        "top:Site": ["top:Zone"],

        "top:Element": ["top:Topology"],
        "top:Wall": ["top:Element"],
        "top:Door": ["top:Element"],
        "top:Window": ["top:Element"],
        "top:Slab": ["top:Element"],
        "top:Roof": ["top:Element"],
        "top:Column": ["top:Element"],
        "top:Beam": ["top:Element"],
        "top:Stair": ["top:Element"],
        "top:Railing": ["top:Element"],
        "top:Opening": ["top:Element"],
        "top:Furniture": ["top:Element"],
        "top:Equipment": ["top:Element"],

        "top:SpatialGraph": ["top:Graph"],
        "top:AdjacencyGraph": ["top:SpatialGraph"],
        "top:VisibilityGraph": ["top:SpatialGraph"],
        "top:CirculationGraph": ["top:SpatialGraph"],
        "top:ConnectivityGraph": ["top:SpatialGraph"],
        "top:KnowledgeGraph": ["top:Graph"],
    }

    # -------------------------------------------------------------------------
    # IFC to TopologicPy ontology mappings
    # -------------------------------------------------------------------------

    IFC_TO_TOP = {
        "IfcProject": "top:Project",
        "IfcSite": "top:Site",
        "IfcBuilding": "top:Building",
        "IfcBuildingStorey": "top:Storey",
        "IfcSpace": "top:Space",
        "IfcZone": "top:Zone",

        "IfcWall": "top:Wall",
        "IfcWallStandardCase": "top:Wall",
        "IfcCurtainWall": "top:Wall",
        "IfcDoor": "top:Door",
        "IfcWindow": "top:Window",
        "IfcSlab": "top:Slab",
        "IfcRoof": "top:Roof",
        "IfcColumn": "top:Column",
        "IfcBeam": "top:Beam",
        "IfcStair": "top:Stair",
        "IfcStairFlight": "top:Stair",
        "IfcRailing": "top:Railing",
        "IfcOpeningElement": "top:Opening",
        "IfcVirtualElement": "top:Element",
        "IfcFurnishingElement": "top:Furniture",
        "IfcFurniture": "top:Furniture",
        "IfcFlowTerminal": "top:Equipment",
        "IfcDistributionElement": "top:Equipment",
        "IfcBuildingElementProxy": "top:Element",

        "IfcRelSpaceBoundary": "top:Interface",
    }

    TOP_TO_BOT = {
        "top:Site": "bot:Site",
        "top:Building": "bot:Building",
        "top:Storey": "bot:Storey",
        "top:Zone": "bot:Zone",
        "top:Space": "bot:Space",
        "top:Room": "bot:Space",
        "top:Element": "bot:Element",
        "top:Wall": "bot:Element",
        "top:Door": "bot:Element",
        "top:Window": "bot:Element",
        "top:Slab": "bot:Element",
        "top:Roof": "bot:Element",
        "top:Column": "bot:Element",
        "top:Beam": "bot:Element",
        "top:Interface": "bot:Interface",
    }

    TOP_CATEGORIES = {
        "top:Project": "project",
        "top:Grid": "utility",
        "top:Dictionary": "metadata",
        "top:Context": "context",
        "top:Vector": "mathematics",
        "top:Matrix": "mathematics",
        "top:Site": "site",
        "top:Building": "building",
        "top:Storey": "storey",
        "top:Zone": "space",
        "top:Space": "space",
        "top:Room": "space",
        "top:ThermalZone": "space",
        "top:Element": "element",
        "top:Wall": "element",
        "top:Door": "element",
        "top:Window": "element",
        "top:Slab": "element",
        "top:Roof": "element",
        "top:Column": "element",
        "top:Beam": "element",
        "top:Opening": "element",
        "top:Aperture": "element",
        "top:Interface": "interface",
        "top:Graph": "graph",
        "top:SpatialGraph": "graph",
        "top:AdjacencyGraph": "graph",
        "top:VisibilityGraph": "graph",
        "top:CirculationGraph": "graph",
        "top:ConnectivityGraph": "graph",
        "top:KnowledgeGraph": "graph",
        "top:Vertex": "topology",
        "top:Edge": "topology",
        "top:Wire": "topology",
        "top:Face": "topology",
        "top:Shell": "topology",
        "top:Cell": "topology",
        "top:CellComplex": "topology",
        "top:Cluster": "topology",
    }


    # -------------------------------------------------------------------------
    # Ontology specification fragments
    # -------------------------------------------------------------------------

    CLASS_COMMENTS = {
        "top:Topology": "A superclass of Vertex, Edge, Wire, Face, Shell, Cell, CellComplex, and Cluster.",
        "top:Vertex": "A point in 3D space defined by X, Y, Z coordinates.",
        "top:Edge": "A line segment connecting two vertices.",
        "top:Wire": "A sequence of connected edges.",
        "top:Face": "A bounded two-dimensional surface, optionally with internal boundaries.",
        "top:Shell": "A collection of faces that share edges and form a segmented surface.",
        "top:Cell": "A volumetric element bounded by faces.",
        "top:CellComplex": "A collection of cells that share faces.",
        "top:Cluster": "A heterogeneous collection of related topologies.",
        "top:Graph": "A collection of nodes and relationships represented by vertices and edges.",
        "top:Grid": "A spatial structure dividing space into regular intervals.",
        "top:Dictionary": "A key-value store for semantic, analytical, and provenance metadata.",
        "top:Aperture": "An opening or aperture associated with a topology.",
        "top:Context": "The environment or settings in which topologies are interpreted.",
        "top:Vector": "A mathematical entity with magnitude and direction.",
        "top:Matrix": "A rectangular array of numbers used for transformations.",
        "top:Project": "A project-level container for a model or dataset.",
    }

    OBJECT_PROPERTIES = {
        "top:hasDictionary": ("top:Topology", "top:Dictionary", "The dictionary of a topology or graph."),
        "top:hasStartVertex": ("top:Edge", "top:Vertex", "The starting vertex of an edge."),
        "top:hasEndVertex": ("top:Edge", "top:Vertex", "The ending vertex of an edge."),
        "top:hasVertices": ("top:Topology", "top:Vertex", "The vertices that belong to a topology or graph."),
        "top:hasEdges": ("top:Topology", "top:Edge", "The edges that belong to a topology or graph."),
        "top:hasWires": ("top:Topology", "top:Wire", "The wires that belong to a topology."),
        "top:hasFaces": ("top:Topology", "top:Face", "The faces that belong to a topology."),
        "top:hasShells": ("top:Topology", "top:Shell", "The shells that belong to a topology."),
        "top:hasCells": ("top:Topology", "top:Cell", "The cells that belong to a topology."),
        "top:hasCellComplexes": ("top:Cluster", "top:CellComplex", "The cell complexes that belong to a cluster."),
        "top:hasExternalBoundary": ("top:Topology", "top:Topology", "The external boundary of a topology."),
        "top:hasInternalBoundaries": ("top:Topology", "top:Topology", "The internal boundaries of a topology."),
        "top:hasFreeVertices": ("top:Cluster", "top:Vertex", "Free vertices in a cluster."),
        "top:hasFreeEdges": ("top:Cluster", "top:Edge", "Free edges in a cluster."),
        "top:hasFreeWires": ("top:Cluster", "top:Wire", "Free wires in a cluster."),
        "top:hasFreeFaces": ("top:Cluster", "top:Face", "Free faces in a cluster."),
        "top:hasFreeShells": ("top:Cluster", "top:Shell", "Free shells in a cluster."),
        "top:hasFreeCells": ("top:Cluster", "top:Cell", "Free cells in a cluster."),
        "top:connectsTo": ("top:Topology", "top:Topology", "The topologies connected to a topology."),
        "top:connectsTo": ("top:Topology", "top:Topology", "Alias of top:connectsTo."),
        "top:adjacentTo": ("top:Topology", "top:Topology", "The topologies adjacent to a topology."),
        "top:interfaceOf": ("top:Topology", "top:Topology", "The topologies connected or separated by an interface."),
        "top:isPartOf": ("top:Topology", "top:Topology", "The topology that contains this topology."),
        "top:containsElement": ("top:Topology", "top:Topology", "A topology contained by this topology."),
        "top:hasDirection": ("top:Topology", "top:Vector", "The direction vector of a topology."),
        "top:hasNode": ("top:Graph", "top:Vertex", "A node that belongs to a graph."),
        "top:hasRelationship": ("top:Graph", "top:Edge", "A relationship that belongs to a graph."),
    }

    DATA_PROPERTIES = {
        "top:hasX": ("top:Vertex", "xsd:double", "The X coordinate of a vertex."),
        "top:hasY": ("top:Vertex", "xsd:double", "The Y coordinate of a vertex."),
        "top:hasZ": ("top:Vertex", "xsd:double", "The Z coordinate of a vertex."),
        "top:hasLength": ("top:Topology", "xsd:double", "The length of an edge or wire."),
        "top:hasArea": ("top:Topology", "xsd:double", "The area of a face, shell, cell, or cell complex."),
        "top:hasVolume": ("top:Topology", "xsd:double", "The volume of a cell or cell complex."),
        "top:hasMantissa": ("top:Topology", "xsd:integer", "The number of decimal places used to report values."),
        "top:hasUnit": ("top:Topology", "xsd:string", "The unit of measurement."),
        "top:createdAt": ("top:Topology", "xsd:dateTime", "The creation timestamp."),
        "top:updatedAt": ("top:Topology", "xsd:dateTime", "The last update timestamp."),
        "top:category": ("top:Topology", "xsd:string", "The broad category of a topology or graph."),
    }

    PROPERTY_ALIASES = {
        "startsAt": "hasStartVertex",
        "endsAt": "hasEndVertex",
        "connectedTo": "connectsTo",
        "x": "hasX",
        "y": "hasY",
        "z": "hasZ",
        "length": "hasLength",
        "area": "hasArea",
        "volume": "hasVolume",
        "mantissa": "hasMantissa",
        "unit": "hasUnit",
    }

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _dictionary(topology):
        """Returns a TopologicPy or Python dictionary from the input topology or graph."""
        if topology is None:
            return None

        record_dict = Ontology._record_dictionary(topology)
        if record_dict is not None:
            return record_dict

        if Ontology._is_tgraph(topology):
            try:
                from topologicpy.TGraph import TGraph
                return TGraph.Dictionary(topology)
            except Exception:
                return None

        try:
            from topologicpy.Topology import Topology
            return Topology.Dictionary(topology)
        except Exception:
            return None
    @staticmethod
    def _is_tgraph(obj):
        """Returns True if the input object is a TGraph."""
        try:
            from topologicpy.TGraph import TGraph
            return isinstance(obj, TGraph)
        except Exception:
            return False

    @staticmethod
    def _is_graph_like(obj):
        """Returns True if the input object is a legacy Graph or TGraph."""
        if obj is None:
            return False
        if Ontology._is_tgraph(obj):
            return True
        try:
            from topologicpy.Topology import Topology
            return bool(Topology.IsInstance(obj, "graph"))
        except Exception:
            return False

    @staticmethod
    def _graph_vertices(graph, asTopologic: bool = False):
        """Returns graph vertices from either a legacy Graph or TGraph."""
        if graph is None:
            return []
        if Ontology._is_tgraph(graph):
            try:
                from topologicpy.TGraph import TGraph
                return TGraph.Vertices(graph, asTopologic=asTopologic, activeOnly=True) or []
            except Exception:
                return []
        try:
            from topologicpy.Graph import Graph
            return Graph.Vertices(graph) or []
        except Exception:
            return []

    @staticmethod
    def _graph_edges(graph, asTopologic: bool = False):
        """Returns graph edges from either a legacy Graph or TGraph."""
        if graph is None:
            return []
        if Ontology._is_tgraph(graph):
            try:
                from topologicpy.TGraph import TGraph
                return TGraph.Edges(graph, asTopologic=asTopologic, activeOnly=True) or []
            except Exception:
                return []
        try:
            from topologicpy.Graph import Graph
            return Graph.Edges(graph) or []
        except Exception:
            return []

    @staticmethod
    def _graph_edge_endpoints(graph, edge, vertex_records=None):
        """Returns the start and end vertices/records of a graph edge."""
        if graph is None or edge is None:
            return None, None
        if Ontology._is_tgraph(graph):
            try:
                src = edge.get("src", None) if isinstance(edge, dict) else None
                dst = edge.get("dst", None) if isinstance(edge, dict) else None
                if isinstance(src, int) and isinstance(dst, int):
                    if vertex_records is None:
                        vertex_records = Ontology._graph_vertices(graph, asTopologic=False)
                    by_index = {}
                    for i, v in enumerate(vertex_records or []):
                        if isinstance(v, dict):
                            by_index[v.get("index", i)] = v
                    return by_index.get(src), by_index.get(dst)
            except Exception:
                return None, None
            return None, None
        try:
            from topologicpy.Edge import Edge
            return Edge.StartVertex(edge), Edge.EndVertex(edge)
        except Exception:
            pass
        try:
            from topologicpy.Topology import Topology
            return Topology.StartVertex(edge), Topology.EndVertex(edge)
        except Exception:
            pass
        try:
            from topologicpy.Graph import Graph
            return Graph.StartVertex(graph, edge), Graph.EndVertex(graph, edge)
        except Exception:
            return None, None

    @staticmethod
    def _record_dictionary(obj):
        """Returns a Python dictionary from a TGraph record-like object."""
        if isinstance(obj, dict):
            d = obj.get("dictionary", obj)
            if isinstance(d, dict):
                return d
        return None

    @staticmethod
    def _record_coordinates(obj, default=None):
        """Returns XYZ coordinates from a TGraph vertex record-like object."""
        default = default if default is not None else [0.0, 0.0, 0.0]
        if isinstance(obj, dict):
            d = obj.get("dictionary", obj)
            if isinstance(d, dict):
                x = d.get("x", obj.get("x", None))
                y = d.get("y", obj.get("y", None))
                z = d.get("z", obj.get("z", None))
                try:
                    return [float(x), float(y), float(z)]
                except Exception:
                    pass
            try:
                coords = obj.get("coordinates", None)
                if isinstance(coords, (list, tuple)) and len(coords) >= 3:
                    return [float(coords[0]), float(coords[1]), float(coords[2])]
            except Exception:
                pass
        return list(default)

    @staticmethod
    def _set_value(topology, key, value, silent=False):
        """Sets a dictionary key/value pair on the input topology, graph, or TGraph record."""
        try:
            from topologicpy.Dictionary import Dictionary
        except Exception:
            Dictionary = None

        if topology is None:
            if not silent:
                print("Ontology._set_value - Error: The input topology is None. Returning None.")
            return None

        if key is None:
            if not silent:
                print("Ontology._set_value - Error: The input key is None. Returning None.")
            return None

        if isinstance(topology, dict):
            d = topology.get("dictionary", topology)
            if not isinstance(d, dict):
                d = {}
            d[key] = value
            if "dictionary" in topology:
                topology["dictionary"] = d
            return topology

        if Ontology._is_tgraph(topology):
            try:
                from topologicpy.TGraph import TGraph
                d = TGraph.Dictionary(topology)
                d[key] = value
                topology.SetDictionary(d)
                return topology
            except Exception as e:
                if not silent:
                    print("Ontology._set_value - Error: Could not set the TGraph dictionary value. Returning None.")
                    print("Error:", e)
                return None

        try:
            from topologicpy.Topology import Topology
            d = Topology.Dictionary(topology)
            d = Dictionary.SetValueAtKey(d, key, value)
            Topology.SetDictionary(topology, d)
            return topology
        except Exception as e:
            if not silent:
                print("Ontology._set_value - Error: Could not set the dictionary value. Returning None.")
                print("Error:", e)
            return None

    @staticmethod
    def _value(topology, key, defaultValue=None):
        """Returns a dictionary value from the input topology, graph, or TGraph record."""
        d = Ontology._dictionary(topology)
        if isinstance(d, dict):
            return d.get(key, defaultValue)
        try:
            from topologicpy.Dictionary import Dictionary
            return Dictionary.ValueAtKey(d, key, defaultValue)
        except Exception:
            return defaultValue

    @staticmethod
    def _as_list(item):
        """Returns item as a list."""
        if item is None:
            return []
        if isinstance(item, list):
            return item
        if isinstance(item, tuple):
            return list(item)
        return [item]

    @staticmethod
    def _safe_string(value):
        """Returns a safe string representation for RDF literals and identifiers."""
        if value is None:
            return ""
        return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")

    @staticmethod
    def _safe_local_name(value):
        """Returns a conservative RDF local name."""
        import re

        if value is None:
            return "unnamed"
        s = str(value).strip()
        if s == "":
            return "unnamed"
        s = re.sub(r"[^A-Za-z0-9_\\-]", "_", s)
        if len(s) == 0:
            s = "unnamed"
        if s[0].isdigit():
            s = "id_" + s
        return s

    @staticmethod
    def _is_number(value):
        """Returns True if value is an int or float but not a bool."""
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    @staticmethod
    def _rdf_literal(value):
        """Returns an RDF literal string with a basic datatype."""
        if isinstance(value, bool):
            return '"' + str(value).lower() + '"^^xsd:boolean'
        if isinstance(value, int) and not isinstance(value, bool):
            return '"' + str(value) + '"^^xsd:integer'
        if isinstance(value, float):
            return '"' + str(value) + '"^^xsd:double'
        return '"' + Ontology._safe_string(value) + '"'

    @staticmethod
    def _uri_for_topology(topology, prefix="inst"):
        """Returns a stable URI-like QName for a topology, graph, or TGraph entity."""
        guid = Ontology._value(topology, Ontology.IFC_GUID_KEY, None)
        uri = Ontology._value(topology, Ontology.URI_KEY, None)
        label = Ontology._value(topology, Ontology.LABEL_KEY, None)
        local_id = Ontology._value(topology, "id", None)

        if uri:
            if ":" in str(uri):
                return str(uri)
            return prefix + ":" + Ontology._safe_local_name(uri)

        if guid:
            return prefix + ":" + Ontology._safe_local_name(guid)

        if local_id:
            return prefix + ":" + Ontology._safe_local_name(local_id)

        if label:
            return prefix + ":" + Ontology._safe_local_name(label)

        try:
            from topologicpy.Topology import Topology
            uid = Topology.UUID(topology, silent=True)
            return prefix + ":" + Ontology._safe_local_name(uid)
        except Exception:
            return prefix + ":" + Ontology._safe_local_name(id(topology))

    # -------------------------------------------------------------------------
    # Namespace methods
    # -------------------------------------------------------------------------

    @staticmethod
    def Namespaces():
        """
        Returns the default ontology namespace dictionary.

        Returns
        -------
        dict
            The namespace dictionary.
        """

        return dict(Ontology.NAMESPACES)

    @staticmethod
    def Namespace(prefix, defaultValue=None):
        """
        Returns the namespace URI associated with the input prefix.

        Parameters
        ----------
        prefix : str
            The namespace prefix.
        defaultValue : any , optional
            The value to return if the prefix is not found. Default is None.

        Returns
        -------
        str
            The namespace URI.
        """

        if prefix is None:
            return defaultValue
        return Ontology.NAMESPACES.get(prefix, defaultValue)

    @staticmethod
    def ExpandQName(qname, defaultValue=None):
        """
        Expands a QName into a full URI.

        Parameters
        ----------
        qname : str
            The QName, for example "top:Room".
        defaultValue : any , optional
            The value to return if the QName cannot be expanded. Default is None.

        Returns
        -------
        str
            The expanded URI.
        """

        if not isinstance(qname, str):
            return defaultValue
        if ":" not in qname:
            return defaultValue
        prefix, local = qname.split(":", 1)
        ns = Ontology.Namespace(prefix)
        if ns is None:
            return defaultValue
        return ns + local

    # -------------------------------------------------------------------------
    # Class and category methods
    # -------------------------------------------------------------------------

    @staticmethod
    def Class(topology, defaultValue=None):
        """
        Returns the ontology class assigned to the input topology.

        Parameters
        ----------
        topology : topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        defaultValue : any , optional
            The value to return if no ontology class is found. Default is None.

        Returns
        -------
        str
            The ontology class.
        """

        return Ontology._value(topology, Ontology.ONTOLOGY_CLASS_KEY, defaultValue)

    @staticmethod
    def SetClass(topology, ontologyClass, setCategory=True, setURI=True, silent=False):
        """
        Sets the ontology class of the input topology.

        Parameters
        ----------
        topology : topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        ontologyClass : str
            The ontology class, for example "top:Room".
        setCategory : bool , optional
            If True, the category is also set if it can be inferred. Default is True.
        setURI : bool , optional
            If True, ontology_uri is also set if the class can be expanded. Default is True.
        silent : bool , optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        """

        if topology is None:
            if not silent:
                print("Ontology.SetClass - Error: The input topology is None. Returning None.")
            return None
        if not isinstance(ontologyClass, str) or ontologyClass.strip() == "":
            if not silent:
                print("Ontology.SetClass - Error: The input ontologyClass is not a valid string. Returning None.")
            return None

        ontologyClass = ontologyClass.strip()
        topology = Ontology._set_value(topology, Ontology.ONTOLOGY_CLASS_KEY, ontologyClass, silent=silent)

        if topology is not None and setCategory:
            category = Ontology.CategoryByClass(ontologyClass)
            if category is not None:
                topology = Ontology.SetCategory(topology, category, silent=silent)

        if topology is not None and setURI:
            uri = Ontology.ExpandQName(ontologyClass)
            if uri is not None:
                topology = Ontology._set_value(topology, Ontology.ONTOLOGY_URI_KEY, uri, silent=silent)

        return topology

    @staticmethod
    def Category(topology, defaultValue=None):
        """
        Returns the category assigned to the input topology.

        Parameters
        ----------
        topology : topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        defaultValue : any , optional
            The value to return if no category is found. Default is None.

        Returns
        -------
        str
            The category.
        """

        return Ontology._value(topology, Ontology.CATEGORY_KEY, defaultValue)

    @staticmethod
    def SetCategory(topology, category, silent=False):
        """
        Sets the category of the input topology.

        Parameters
        ----------
        topology : topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        category : str
            The category.
        silent : bool , optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        """

        if not isinstance(category, str) or category.strip() == "":
            if not silent:
                print("Ontology.SetCategory - Error: The input category is not a valid string. Returning None.")
            return None
        return Ontology._set_value(topology, Ontology.CATEGORY_KEY, category.strip(), silent=silent)

    @staticmethod
    def Label(topology, defaultValue=None):
        """
        Returns the label assigned to the input topology.

        Parameters
        ----------
        topology : topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        defaultValue : any , optional
            The value to return if no label is found. Default is None.

        Returns
        -------
        str
            The label.
        """

        return Ontology._value(topology, Ontology.LABEL_KEY, defaultValue)

    @staticmethod
    def SetLabel(topology, label, silent=False):
        """
        Sets the label of the input topology.

        Parameters
        ----------
        topology : topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        label : str
            The label.
        silent : bool , optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        """

        return Ontology._set_value(topology, Ontology.LABEL_KEY, label, silent=silent)

    @staticmethod
    def URI(topology, defaultValue=None):
        """
        Returns the URI assigned to the input topology.

        Parameters
        ----------
        topology : topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        defaultValue : any , optional
            The value to return if no URI is found. Default is None.

        Returns
        -------
        str
            The URI.
        """

        return Ontology._value(topology, Ontology.URI_KEY, defaultValue)

    @staticmethod
    def SetURI(topology, uri, silent=False):
        """
        Sets the URI of the input topology.

        Parameters
        ----------
        topology : topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        uri : str
            The URI.
        silent : bool , optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        """

        return Ontology._set_value(topology, Ontology.URI_KEY, uri, silent=silent)

    @staticmethod
    def CategoryByClass(ontologyClass, defaultValue=None):
        """
        Returns a category for the input ontology class.

        Parameters
        ----------
        ontologyClass : str
            The ontology class.
        defaultValue : any , optional
            The value to return if no category is found. Default is None.

        Returns
        -------
        str
            The category.
        """

        if ontologyClass is None:
            return defaultValue
        ontologyClass = str(ontologyClass).strip()
        if ontologyClass in Ontology.TOP_CATEGORIES:
            return Ontology.TOP_CATEGORIES[ontologyClass]
        return defaultValue

    @staticmethod
    def Superclasses(ontologyClass, transitive=True):
        """
        Returns the superclasses of the input ontology class.

        Parameters
        ----------
        ontologyClass : str
            The ontology class.
        transitive : bool , optional
            If True, all transitive superclasses are returned. If False, only direct
            superclasses are returned. Default is True.

        Returns
        -------
        list
            The list of superclasses.
        """

        if ontologyClass is None:
            return []

        ontologyClass = str(ontologyClass).strip()
        direct = Ontology.TOP_SUPERCLASSES.get(ontologyClass, [])
        if not transitive:
            return list(direct)

        result = []
        stack = list(direct)
        while len(stack) > 0:
            cls = stack.pop(0)
            if cls not in result:
                result.append(cls)
                stack.extend(Ontology.TOP_SUPERCLASSES.get(cls, []))
        return result

    @staticmethod
    def IsA(topology, ontologyClass, transitive=True):
        """
        Returns True if the input topology is an instance of the input ontology class.

        Parameters
        ----------
        topology : topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        ontologyClass : str
            The ontology class to test against.
        transitive : bool , optional
            If True, superclass inheritance is considered. Default is True.

        Returns
        -------
        bool
            True if the topology is an instance of ontologyClass. Otherwise False.
        """

        assignedClass = Ontology.Class(topology)
        if assignedClass is None or ontologyClass is None:
            return False

        assignedClass = str(assignedClass).strip()
        ontologyClass = str(ontologyClass).strip()

        if assignedClass == ontologyClass:
            return True

        if transitive:
            return ontologyClass in Ontology.Superclasses(assignedClass, transitive=True)

        return False

    # -------------------------------------------------------------------------
    # IFC mappings
    # -------------------------------------------------------------------------

    @staticmethod
    def ClassByIFCClass(ifcClass, defaultValue="top:Element"):
        """
        Returns a TopologicPy ontology class from an IFC class.

        Parameters
        ----------
        ifcClass : str
            The IFC class, for example "IfcSpace".
        defaultValue : any , optional
            The value to return if the IFC class is not mapped. Default is "top:Element".

        Returns
        -------
        str
            The TopologicPy ontology class.
        """

        if ifcClass is None:
            return defaultValue
        ifcClass = str(ifcClass).strip()
        return Ontology.IFC_TO_TOP.get(ifcClass, defaultValue)

    @staticmethod
    def BOTClassByClass(ontologyClass, defaultValue=None):
        """
        Returns a BOT class from a TopologicPy ontology class.

        Parameters
        ----------
        ontologyClass : str
            The TopologicPy ontology class.
        defaultValue : any , optional
            The value to return if the class is not mapped. Default is None.

        Returns
        -------
        str
            The BOT class.
        """

        if ontologyClass is None:
            return defaultValue

        ontologyClass = str(ontologyClass).strip()
        if ontologyClass in Ontology.TOP_TO_BOT:
            return Ontology.TOP_TO_BOT[ontologyClass]

        for superclass in Ontology.Superclasses(ontologyClass, transitive=True):
            if superclass in Ontology.TOP_TO_BOT:
                return Ontology.TOP_TO_BOT[superclass]

        return defaultValue

    @staticmethod
    def AnnotateIFC(topology,
                    ifcClass=None,
                    ifcGUID=None,
                    ifcName=None,
                    source=None,
                    silent=False):
        """
        Annotates a topology using IFC metadata.

        Parameters
        ----------
        topology : topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        ifcClass : str , optional
            The IFC class. Default is None.
        ifcGUID : str , optional
            The IFC GlobalId. Default is None.
        ifcName : str , optional
            The IFC name. If specified, it is stored as the label. Default is None.
        source : str , optional
            The source file or source identifier. Default is None.
        silent : bool , optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The annotated topology or graph.
        """

        if topology is None:
            if not silent:
                print("Ontology.AnnotateIFC - Error: The input topology is None. Returning None.")
            return None

        if ifcClass is not None:
            topology = Ontology._set_value(topology, Ontology.IFC_CLASS_KEY, ifcClass, silent=silent)
            ontologyClass = Ontology.ClassByIFCClass(ifcClass, defaultValue=None)
            if ontologyClass is not None:
                topology = Ontology.SetClass(topology, ontologyClass, silent=silent)

        if ifcGUID is not None:
            topology = Ontology._set_value(topology, Ontology.IFC_GUID_KEY, ifcGUID, silent=silent)

        if ifcName is not None:
            topology = Ontology.SetLabel(topology, ifcName, silent=silent)

        if source is not None:
            topology = Ontology._set_value(topology, Ontology.SOURCE_KEY, source, silent=silent)

        return topology

    # -------------------------------------------------------------------------
    # Topologic class inference
    # -------------------------------------------------------------------------

    @staticmethod
    def ClassByTopology(topology, defaultValue="top:Topology"):
        """
        Returns a TopologicPy ontology class inferred from the topology type.

        Parameters
        ----------
        topology : topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        defaultValue : any , optional
            The value to return if the topology type cannot be inferred. Default is "top:Topology".

        Returns
        -------
        str
            The inferred ontology class.
        """

        if topology is None:
            return defaultValue

        if Ontology._is_tgraph(topology):
            return "top:Graph"

        try:
            from topologicpy.Topology import Topology

            type_name = None
            try:
                type_name = Topology.TypeAsString(topology)
            except Exception:
                pass

            if isinstance(type_name, str):
                type_name_lower = type_name.lower()
                mapping = {
                    "vertex": "top:Vertex",
                    "edge": "top:Edge",
                    "wire": "top:Wire",
                    "face": "top:Face",
                    "shell": "top:Shell",
                    "cell": "top:Cell",
                    "cellcomplex": "top:CellComplex",
                    "cluster": "top:Cluster",
                    "graph": "top:Graph",
                    "tgraph": "top:Graph",
                }
                if type_name_lower in mapping:
                    return mapping[type_name_lower]

            if Topology.IsInstance(topology, "vertex"):
                return "top:Vertex"
            if Topology.IsInstance(topology, "edge"):
                return "top:Edge"
            if Topology.IsInstance(topology, "wire"):
                return "top:Wire"
            if Topology.IsInstance(topology, "face"):
                return "top:Face"
            if Topology.IsInstance(topology, "shell"):
                return "top:Shell"
            if Topology.IsInstance(topology, "cell"):
                return "top:Cell"
            if Topology.IsInstance(topology, "cellcomplex"):
                return "top:CellComplex"
            if Topology.IsInstance(topology, "cluster"):
                return "top:Cluster"
            if Topology.IsInstance(topology, "graph"):
                return "top:Graph"
        except Exception:
            pass

        return defaultValue

    @staticmethod
    def Annotate(topology,
                 ontologyClass=None,
                 category=None,
                 label=None,
                 uri=None,
                 source=None,
                 derivedFrom=None,
                 generatedBy=None,
                 inferClass=False,
                 silent=False):
        """
        Annotates a topology using canonical ontology dictionary keys.

        Parameters
        ----------
        topology : topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        ontologyClass : str , optional
            The ontology class. Default is None.
        category : str , optional
            The category. Default is None.
        label : str , optional
            The label. Default is None.
        uri : str , optional
            The URI. Default is None.
        source : str , optional
            The source identifier. Default is None.
        derivedFrom : str , optional
            The source entity from which this topology was derived. Default is None.
        generatedBy : str , optional
            The process or method that generated this topology. Default is None.
        inferClass : bool , optional
            If True and ontologyClass is None, the ontology class is inferred from
            the Topologic type. Default is False.
        silent : bool , optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The annotated topology or graph.
        """

        if topology is None:
            if not silent:
                print("Ontology.Annotate - Error: The input topology is None. Returning None.")
            return None

        if ontologyClass is None and inferClass:
            ontologyClass = Ontology.ClassByTopology(topology, defaultValue=None)

        if ontologyClass is not None:
            topology = Ontology.SetClass(topology, ontologyClass, silent=silent)

        if category is not None:
            topology = Ontology.SetCategory(topology, category, silent=silent)

        if label is not None:
            topology = Ontology.SetLabel(topology, label, silent=silent)

        if uri is not None:
            topology = Ontology.SetURI(topology, uri, silent=silent)

        if source is not None:
            topology = Ontology._set_value(topology, Ontology.SOURCE_KEY, source, silent=silent)

        if derivedFrom is not None:
            topology = Ontology._set_value(topology, Ontology.DERIVED_FROM_KEY, derivedFrom, silent=silent)

        if generatedBy is not None:
            topology = Ontology._set_value(topology, Ontology.GENERATED_BY_KEY, generatedBy, silent=silent)

        return topology

    @staticmethod
    def AnnotateSubtopologies(topology,
                              vertices=True,
                              edges=True,
                              wires=True,
                              faces=True,
                              shells=True,
                              cells=True,
                              cellComplexes=True,
                              inferClass=True,
                              silent=False):
        """
        Annotates the requested subtopologies with inferred ontology classes.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        vertices : bool , optional
            If True, annotate vertices. Default is True.
        edges : bool , optional
            If True, annotate edges. Default is True.
        wires : bool , optional
            If True, annotate wires. Default is True.
        faces : bool , optional
            If True, annotate faces. Default is True.
        shells : bool , optional
            If True, annotate shells. Default is True.
        cells : bool , optional
            If True, annotate cells. Default is True.
        cellComplexes : bool , optional
            If True, annotate cell complexes. Default is True.
        inferClass : bool , optional
            If True, infer ontology classes from Topologic types. Default is True.
        silent : bool , optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Topology
            The input topology.
        """

        if topology is None:
            if not silent:
                print("Ontology.AnnotateSubtopologies - Error: The input topology is None. Returning None.")
            return None

        try:
            from topologicpy.Topology import Topology

            selectors = []
            if vertices:
                selectors.append("vertices")
            if edges:
                selectors.append("edges")
            if wires:
                selectors.append("wires")
            if faces:
                selectors.append("faces")
            if shells:
                selectors.append("shells")
            if cells:
                selectors.append("cells")
            if cellComplexes:
                selectors.append("cellcomplexes")

            for selector in selectors:
                try:
                    subtopologies = Topology.SubTopologies(topology, subTopologyType=selector)
                except Exception:
                    subtopologies = []
                for subtopology in subtopologies:
                    Ontology.Annotate(subtopology, inferClass=inferClass, silent=True)
            return topology
        except Exception as e:
            if not silent:
                print("Ontology.AnnotateSubtopologies - Error: Could not annotate subtopologies. Returning input topology.")
                print("Error:", e)
            return topology

    # -------------------------------------------------------------------------
    # Dictionary normalisation
    # -------------------------------------------------------------------------

    @staticmethod
    def NormalizeDictionary(topology,
                            labelKeys=None,
                            categoryKeys=None,
                            ifcClassKeys=None,
                            ifcGUIDKeys=None,
                            silent=False):
        """
        Normalises common dictionary keys into canonical ontology keys.

        Parameters
        ----------
        topology : topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        labelKeys : list , optional
            Candidate keys for the label. Default is ["name", "Name", "LongName", "ifc_name", "label"].
        categoryKeys : list , optional
            Candidate keys for the category. Default is ["category", "type", "ObjectType"].
        ifcClassKeys : list , optional
            Candidate keys for the IFC class. Default is ["ifc_class", "IfcClass", "class", "type"].
        ifcGUIDKeys : list , optional
            Candidate keys for the IFC GUID. Default is ["ifc_guid", "GlobalId", "global_id", "guid"].
        silent : bool , optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        """

        try:
            from topologicpy.Dictionary import Dictionary
        except Exception:
            Dictionary = None

        if topology is None:
            if not silent:
                print("Ontology.NormalizeDictionary - Error: The input topology is None. Returning None.")
            return None

        labelKeys = labelKeys or ["name", "Name", "LongName", "ifc_name", "label"]
        categoryKeys = categoryKeys or ["category", "type", "ObjectType"]
        ifcClassKeys = ifcClassKeys or ["ifc_class", "IfcClass", "class", "type"]
        ifcGUIDKeys = ifcGUIDKeys or ["ifc_guid", "GlobalId", "global_id", "guid"]

        d = Ontology._dictionary(topology)
        if d is None:
            return topology

        def first_value(keys):
            for key in keys:
                value = Dictionary.ValueAtKey(d, key, None)
                if value is not None and value != "":
                    return value
            return None

        label = first_value(labelKeys)
        category = first_value(categoryKeys)
        ifcClass = first_value(ifcClassKeys)
        ifcGUID = first_value(ifcGUIDKeys)

        if label is not None:
            topology = Ontology.SetLabel(topology, label, silent=silent)
        if category is not None:
            topology = Ontology.SetCategory(topology, str(category).lower(), silent=silent)
        if ifcClass is not None:
            topology = Ontology.AnnotateIFC(topology, ifcClass=ifcClass, silent=silent)
        if ifcGUID is not None:
            topology = Ontology._set_value(topology, Ontology.IFC_GUID_KEY, ifcGUID, silent=silent)

        return topology

    # -------------------------------------------------------------------------
    # Triple and Turtle export
    # -------------------------------------------------------------------------

    @staticmethod
    def Triples(topology,
                subject=None,
                includeDictionaries=True,
                includeBOT=True,
                includeType=True,
                includeLabel=True,
                includeCategory=True,
                namespacePrefix="inst",
                silent=False):
        """
        Returns RDF-like triples from a topology or graph dictionary.

        Parameters
        ----------
        topology : topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        subject : str , optional
            The subject QName. If None, one is inferred. Default is None.
        includeDictionaries : bool , optional
            If True, dictionary entries are exported as top properties. Default is True.
        includeBOT : bool , optional
            If True, a BOT rdf:type triple is added when a mapping exists. Default is True.
        includeType : bool , optional
            If True, rdf:type triples are exported. Default is True.
        includeLabel : bool , optional
            If True, rdfs:label is exported when a label exists. Default is True.
        includeCategory : bool , optional
            If True, top:category is exported when a category exists. Default is True.
        namespacePrefix : str , optional
            The instance namespace prefix. Default is "inst".
        silent : bool , optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            A list of triples. Each triple is a tuple of the form (subject, predicate, object).
        """

        try:
            from topologicpy.Dictionary import Dictionary
        except Exception:
            Dictionary = None

        if topology is None:
            if not silent:
                print("Ontology.Triples - Error: The input topology is None. Returning an empty list.")
            return []

        triples = []
        if subject is None:
            subject = Ontology._uri_for_topology(topology, prefix=namespacePrefix)

        ontologyClass = Ontology.Class(topology)
        label = Ontology.Label(topology)
        category = Ontology.Category(topology)

        if includeType and ontologyClass is not None:
            triples.append((subject, "rdf:type", ontologyClass))
            if includeBOT:
                botClass = Ontology.BOTClassByClass(ontologyClass)
                if botClass is not None:
                    triples.append((subject, "rdf:type", botClass))

        if includeLabel and label is not None:
            triples.append((subject, "rdfs:label", Ontology._rdf_literal(label)))

        if includeCategory and category is not None:
            triples.append((subject, "top:category", Ontology._rdf_literal(category)))

        if includeDictionaries:
            d = Ontology._dictionary(topology)
            if d is not None:
                if isinstance(d, dict):
                    keys = list(d.keys())
                    def _dict_value(_d, _key):
                        return _d.get(_key, None)
                else:
                    try:
                        keys = Dictionary.Keys(d)
                    except Exception:
                        keys = []
                    def _dict_value(_d, _key):
                        try:
                            return Dictionary.ValueAtKey(_d, _key, None)
                        except Exception:
                            return None
                skip_keys = {
                    Ontology.ONTOLOGY_CLASS_KEY,
                    Ontology.ONTOLOGY_URI_KEY,
                    Ontology.LABEL_KEY,
                    Ontology.CATEGORY_KEY,
                    Ontology.URI_KEY,
                }
                for key in keys:
                    if key in skip_keys:
                        continue
                    value = _dict_value(d, key)
                    if value is None:
                        continue
                    predicate = Ontology.PropertyQName(key)
                    if isinstance(value, (list, tuple)):
                        for v in value:
                            triples.append((subject, predicate, Ontology._rdf_literal(v)))
                    else:
                        triples.append((subject, predicate, Ontology._rdf_literal(value)))

        return triples

    @staticmethod
    def GraphTriples(graph,
                     includeVertices=True,
                     includeEdges=True,
                     includeDictionaries=True,
                     includeBOT=True,
                     namespacePrefix="inst",
                     silent=False):
        """
        Returns RDF-like triples from a TopologicPy graph.

        Parameters
        ----------
        graph : topologic_core.Graph or topologicpy.TGraph
            The input graph.
        includeVertices : bool , optional
            If True, vertex triples are exported. Default is True.
        includeEdges : bool , optional
            If True, edge triples are exported. Default is True.
        includeDictionaries : bool , optional
            If True, dictionary entries are exported. Default is True.
        includeBOT : bool , optional
            If True, BOT rdf:type triples are added when mappings exist. Default is True.
        namespacePrefix : str , optional
            The instance namespace prefix. Default is "inst".
        silent : bool , optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            A list of triples.
        """

        triples = []

        if graph is None:
            if not silent:
                print("Ontology.GraphTriples - Error: The input graph is None. Returning an empty list.")
            return triples

        try:
            graph_subject = Ontology._uri_for_topology(graph, prefix=namespacePrefix)
            triples.extend(Ontology.Triples(graph,
                                            subject=graph_subject,
                                            includeDictionaries=includeDictionaries,
                                            includeBOT=includeBOT,
                                            namespacePrefix=namespacePrefix,
                                            silent=True))

            vertex_records = []
            if includeVertices:
                vertex_records = Ontology._graph_vertices(graph, asTopologic=False)
                for i, vertex in enumerate(vertex_records):
                    if Ontology.Class(vertex) is None:
                        Ontology.Annotate(vertex, ontologyClass="top:Node", silent=True)
                    v_subject = Ontology._uri_for_topology(vertex, prefix=namespacePrefix)
                    triples.append((graph_subject, "top:hasNode", v_subject))
                    triples.extend(Ontology.Triples(vertex,
                                                    subject=v_subject,
                                                    includeDictionaries=includeDictionaries,
                                                    includeBOT=includeBOT,
                                                    namespacePrefix=namespacePrefix,
                                                    silent=True))

            if includeEdges:
                if not vertex_records:
                    vertex_records = Ontology._graph_vertices(graph, asTopologic=False)
                edges = Ontology._graph_edges(graph, asTopologic=False)
                for i, edge in enumerate(edges):
                    if Ontology.Class(edge) is None:
                        Ontology.Annotate(edge, ontologyClass="top:Relationship", silent=True)
                    e_subject = Ontology._uri_for_topology(edge, prefix=namespacePrefix)
                    # TGraph edge records commonly lack stable labels/URIs, so provide
                    # a deterministic fallback edge subject.
                    if isinstance(edge, dict):
                        d = Ontology._record_dictionary(edge) or {}
                        if not any(d.get(k) for k in [Ontology.URI_KEY, Ontology.IFC_GUID_KEY, Ontology.LABEL_KEY, "id"]):
                            src = edge.get("src", "u")
                            dst = edge.get("dst", "v")
                            e_subject = f"{namespacePrefix}:edge_{Ontology._safe_local_name(str(src))}_{Ontology._safe_local_name(str(dst))}_{i}"
                    triples.append((graph_subject, "top:hasRelationship", e_subject))
                    triples.extend(Ontology.Triples(edge,
                                                    subject=e_subject,
                                                    includeDictionaries=includeDictionaries,
                                                    includeBOT=includeBOT,
                                                    namespacePrefix=namespacePrefix,
                                                    silent=True))

                    sv, ev = Ontology._graph_edge_endpoints(graph, edge, vertex_records=vertex_records)

                    if sv is not None:
                        triples.append((e_subject, "top:hasStartVertex", Ontology._uri_for_topology(sv, prefix=namespacePrefix)))
                    if ev is not None:
                        triples.append((e_subject, "top:hasEndVertex", Ontology._uri_for_topology(ev, prefix=namespacePrefix)))
                    if sv is not None and ev is not None:
                        triples.append((Ontology._uri_for_topology(sv, prefix=namespacePrefix), "top:connectsTo", Ontology._uri_for_topology(ev, prefix=namespacePrefix)))
            return triples
        except Exception as e:
            if not silent:
                print("Ontology.GraphTriples - Error: Could not create graph triples. Returning partial list.")
                print("Error:", e)
            return triples

    @staticmethod
    def TurtleFromTriples(triples,
                          namespaces=None,
                          instanceNamespace="http://w3id.org/topologicpy/instance#",
                          includeHeader=True):
        """
        Returns a Turtle string from the input triples.

        Parameters
        ----------
        triples : list
            A list of triples. Each triple must be a tuple of the form
            (subject, predicate, object).
        namespaces : dict , optional
            Namespace dictionary. Default is None.
        instanceNamespace : str , optional
            The URI for the "inst" namespace. Default is "http://w3id.org/topologicpy/instance#".
        includeHeader : bool , optional
            If True, namespace prefixes are included. Default is True.

        Returns
        -------
        str
            The Turtle string.
        """

        namespaces = namespaces or Ontology.NAMESPACES
        namespaces = dict(namespaces)
        if "inst" not in namespaces:
            namespaces["inst"] = instanceNamespace

        lines = []

        if includeHeader:
            for prefix, uri in namespaces.items():
                lines.append("@prefix " + prefix + ": <" + uri + "> .")
            lines.append("")

        for triple in triples:
            if triple is None or len(triple) != 3:
                continue
            s, p, o = triple
            if o is None:
                continue
            lines.append(str(s) + " " + str(p) + " " + str(o) + " .")

        return "\n".join(lines) + "\n"


    @staticmethod
    def PropertyQName(key, defaultPrefix="top"):
        """
        Returns a canonical ontology property QName from a dictionary key.

        Parameters
        ----------
        key : str
            The dictionary key or property name.
        defaultPrefix : str , optional
            The default namespace prefix to use if the key is not already a QName.
            Default is "top".

        Returns
        -------
        str
            The canonical property QName.
        """

        if key is None:
            return None
        key = str(key).strip()
        if key == "":
            return None
        if ":" in key:
            return key
        key = Ontology.PROPERTY_ALIASES.get(key, key)
        return str(defaultPrefix) + ":" + Ontology._safe_local_name(key)

    @staticmethod
    def OntologyTriples(includeBOT=True):
        """
        Returns triples describing the TopologicPy ontology itself.

        Parameters
        ----------
        includeBOT : bool , optional
            If True, BOT alignment triples are included where appropriate. Default is True.

        Returns
        -------
        list
            A list of triples describing classes, properties, subclass relations, and comments.
        """

        triples = []
        triples.append(("top:TopologicPyOntology", "rdf:type", "owl:Ontology"))

        for cls, comment in Ontology.CLASS_COMMENTS.items():
            triples.append((cls, "rdf:type", "owl:Class"))
            for superclass in Ontology.Superclasses(cls, transitive=False):
                triples.append((cls, "rdfs:subClassOf", superclass))
            triples.append((cls, "rdfs:comment", Ontology._rdf_literal(comment)))
            if includeBOT:
                botClass = Ontology.BOTClassByClass(cls)
                if botClass is not None:
                    triples.append((cls, "rdfs:subClassOf", botClass))

        for cls, superclasses in Ontology.TOP_SUPERCLASSES.items():
            if cls not in Ontology.CLASS_COMMENTS:
                triples.append((cls, "rdf:type", "owl:Class"))
            for superclass in superclasses:
                triples.append((cls, "rdfs:subClassOf", superclass))

        for prop, data in Ontology.OBJECT_PROPERTIES.items():
            domain, range_, comment = data
            triples.append((prop, "rdf:type", "owl:ObjectProperty"))
            triples.append((prop, "rdfs:domain", domain))
            triples.append((prop, "rdfs:range", range_))
            triples.append((prop, "rdfs:comment", Ontology._rdf_literal(comment)))

        for prop, data in Ontology.DATA_PROPERTIES.items():
            domain, range_, comment = data
            triples.append((prop, "rdf:type", "owl:DatatypeProperty"))
            triples.append((prop, "rdfs:domain", domain))
            triples.append((prop, "rdfs:range", range_))
            triples.append((prop, "rdfs:comment", Ontology._rdf_literal(comment)))

        if includeBOT:
            bot_subproperties = {
                "top:hasVertices": "bot:hasSubElement",
                "top:hasEdges": "bot:hasSubElement",
                "top:hasWires": "bot:hasSubElement",
                "top:hasFaces": "bot:hasSubElement",
                "top:hasShells": "bot:hasSubElement",
                "top:hasCells": "bot:hasSubElement",
                "top:hasCellComplexes": "bot:hasSubElement",
                "top:hasFreeVertices": "bot:hasSubElement",
                "top:hasFreeEdges": "bot:hasSubElement",
                "top:hasFreeWires": "bot:hasSubElement",
                "top:hasFreeFaces": "bot:hasSubElement",
                "top:hasFreeShells": "bot:hasSubElement",
                "top:hasFreeCells": "bot:hasSubElement",
                "top:connectsTo": "bot:connectsTo",
                "top:adjacentTo": "bot:adjacentTo",
                "top:interfaceOf": "bot:interfaceOf",
                "top:containsElement": "bot:containsElement",
                "top:isPartOf": "bot:isPartOf",
            }
            for prop, bot_prop in bot_subproperties.items():
                triples.append((prop, "rdfs:subPropertyOf", bot_prop))

        return triples

    @staticmethod
    def OntologyTTLString(includeBOT=True,
                          instanceNamespace="http://w3id.org/topologicpy/instance#"):
        """
        Returns the TopologicPy ontology specification as a Turtle string.

        Parameters
        ----------
        includeBOT : bool , optional
            If True, BOT alignment triples are included. Default is True.
        instanceNamespace : str , optional
            The instance namespace URI. Default is "http://w3id.org/topologicpy/instance#".

        Returns
        -------
        str
            The ontology Turtle string.
        """

        triples = Ontology.OntologyTriples(includeBOT=includeBOT)
        return Ontology.TurtleFromTriples(triples,
                                          namespaces=Ontology.NAMESPACES,
                                          instanceNamespace=instanceNamespace,
                                          includeHeader=True)

    @staticmethod
    def ExportOntologyTTL(path,
                          includeBOT=True,
                          instanceNamespace="http://w3id.org/topologicpy/instance#",
                          silent=False):
        """
        Exports the TopologicPy ontology specification as a Turtle file.

        Parameters
        ----------
        path : str
            The output Turtle file path.
        includeBOT : bool , optional
            If True, BOT alignment triples are included. Default is True.
        instanceNamespace : str , optional
            The instance namespace URI. Default is "http://w3id.org/topologicpy/instance#".
        silent : bool , optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        str
            The input path if successful. Otherwise None.
        """

        if path is None:
            if not silent:
                print("Ontology.ExportOntologyTTL - Error: The input path is None. Returning None.")
            return None
        try:
            ttl = Ontology.OntologyTTLString(includeBOT=includeBOT,
                                             instanceNamespace=instanceNamespace)
            with open(path, "w", encoding="utf-8") as f:
                f.write(ttl)
            return path
        except Exception as e:
            if not silent:
                print("Ontology.ExportOntologyTTL - Error: Could not export ontology Turtle file. Returning None.")
                print("Error:", e)
            return None

    @staticmethod
    def ExportTTL(topology,
                  path,
                  includeGraph=True,
                  includeDictionaries=True,
                  includeBOT=True,
                  namespacePrefix="inst",
                  instanceNamespace="http://w3id.org/topologicpy/instance#",
                  silent=False):
        """
        Exports a topology or graph as a Turtle file.

        Parameters
        ----------
        topology : topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        path : str
            The output Turtle file path.
        includeGraph : bool , optional
            If True and the input is a graph, graph vertices and edges are exported.
            Default is True.
        includeDictionaries : bool , optional
            If True, dictionary entries are exported as top properties. Default is True.
        includeBOT : bool , optional
            If True, BOT rdf:type triples are added when mappings exist. Default is True.
        namespacePrefix : str , optional
            The instance namespace prefix. Default is "inst".
        instanceNamespace : str , optional
            The instance namespace URI. Default is "http://w3id.org/topologicpy/instance#".
        silent : bool , optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        str
            The input path if successful. Otherwise None.
        """

        if topology is None:
            if not silent:
                print("Ontology.ExportTTL - Error: The input topology is None. Returning None.")
            return None

        if path is None:
            if not silent:
                print("Ontology.ExportTTL - Error: The input path is None. Returning None.")
            return None

        try:
            from topologicpy.Topology import Topology

            if includeGraph and Ontology._is_graph_like(topology):
                triples = Ontology.GraphTriples(topology,
                                                includeDictionaries=includeDictionaries,
                                                includeBOT=includeBOT,
                                                namespacePrefix=namespacePrefix,
                                                silent=silent)
            else:
                triples = Ontology.Triples(topology,
                                           includeDictionaries=includeDictionaries,
                                           includeBOT=includeBOT,
                                           namespacePrefix=namespacePrefix,
                                           silent=silent)

            ttl = Ontology.TurtleFromTriples(triples, instanceNamespace=instanceNamespace)
            with open(path, "w", encoding="utf-8") as f:
                f.write(ttl)
            return path
        except Exception as e:
            if not silent:
                print("Ontology.ExportTTL - Error: Could not export Turtle file. Returning None.")
                print("Error:", e)
            return None

    @staticmethod
    def TTLString(topology,
                  includeGraph=True,
                  includeDictionaries=True,
                  includeBOT=True,
                  namespacePrefix="inst",
                  instanceNamespace="http://w3id.org/topologicpy/instance#",
                  silent=False):
        """
        Returns a Turtle string from a topology or graph.

        Parameters
        ----------
        topology : topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        includeGraph : bool , optional
            If True and the input is a graph, graph vertices and edges are exported.
            Default is True.
        includeDictionaries : bool , optional
            If True, dictionary entries are exported as top properties. Default is True.
        includeBOT : bool , optional
            If True, BOT rdf:type triples are added when mappings exist. Default is True.
        namespacePrefix : str , optional
            The instance namespace prefix. Default is "inst".
        instanceNamespace : str , optional
            The instance namespace URI. Default is "http://w3id.org/topologicpy/instance#".
        silent : bool , optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        str
            The Turtle string.
        """

        if topology is None:
            if not silent:
                print("Ontology.TTLString - Error: The input topology is None. Returning None.")
            return None

        try:
            from topologicpy.Topology import Topology

            if includeGraph and Ontology._is_graph_like(topology):
                triples = Ontology.GraphTriples(topology,
                                                includeDictionaries=includeDictionaries,
                                                includeBOT=includeBOT,
                                                namespacePrefix=namespacePrefix,
                                                silent=silent)
            else:
                triples = Ontology.Triples(topology,
                                           includeDictionaries=includeDictionaries,
                                           includeBOT=includeBOT,
                                           namespacePrefix=namespacePrefix,
                                           silent=silent)

            return Ontology.TurtleFromTriples(triples, instanceNamespace=instanceNamespace)
        except Exception as e:
            if not silent:
                print("Ontology.TTLString - Error: Could not create Turtle string. Returning None.")
                print("Error:", e)
            return None

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    @staticmethod
    def Validate(topology,
                 requireClass=True,
                 requireCategory=False,
                 requireLabel=False,
                 requireURI=False,
                 checkClassKnown=True,
                 checkCategory=True,
                 checkQName=True,
                 silent=False):
        """
        Validates the ontology annotation of a topology or graph object.

        Parameters
        ----------
        topology : topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        requireClass : bool , optional
            If True, ``ontology_class`` must be present. Default is True.
        requireCategory : bool , optional
            If True, ``category`` must be present. Default is False.
        requireLabel : bool , optional
            If True, ``label`` must be present. Default is False.
        requireURI : bool , optional
            If True, ``uri`` or ``ontology_uri`` must be present. Default is False.
        checkClassKnown : bool , optional
            If True, ``ontology_class`` must be in the built-in ontology class map or be expandable as a QName. Default is True.
        checkCategory : bool , optional
            If True, validates ``category`` against the inferred category where possible. Default is True.
        checkQName : bool , optional
            If True, validates QName prefixes against ``Ontology.NAMESPACES``. Default is True.
        silent : bool , optional
            If True, warnings are suppressed. Default is False.

        Returns
        -------
        dict
            A validation report with keys ``ok``, ``errors``, ``warnings``, and ``dictionary``.
        """

        report = {
            "ok": False,
            "errors": [],
            "warnings": [],
            "dictionary": {},
        }

        if topology is None:
            report["errors"].append("The input topology is None.")
            return report

        d = Ontology._dictionary(topology)
        if isinstance(d, dict):
            report["dictionary"] = dict(d)
        else:
            try:
                from topologicpy.Dictionary import Dictionary
                if d is not None:
                    report["dictionary"] = dict(Dictionary.PythonDictionary(d) or {})
            except Exception:
                report["dictionary"] = {}

        ontologyClass = report["dictionary"].get(Ontology.ONTOLOGY_CLASS_KEY, None)
        category = report["dictionary"].get(Ontology.CATEGORY_KEY, None)
        label = report["dictionary"].get(Ontology.LABEL_KEY, None)
        uri = report["dictionary"].get(Ontology.URI_KEY, None)
        ontologyURI = report["dictionary"].get(Ontology.ONTOLOGY_URI_KEY, None)

        if requireClass and (ontologyClass is None or str(ontologyClass).strip() == ""):
            report["errors"].append("Missing ontology_class.")

        if requireCategory and (category is None or str(category).strip() == ""):
            report["errors"].append("Missing category.")

        if requireLabel and (label is None or str(label).strip() == ""):
            report["errors"].append("Missing label.")

        if requireURI and (uri is None or str(uri).strip() == "") and (ontologyURI is None or str(ontologyURI).strip() == ""):
            report["errors"].append("Missing uri or ontology_uri.")

        if ontologyClass is not None and str(ontologyClass).strip() != "":
            ontologyClass = str(ontologyClass).strip()

            if checkQName and ":" in ontologyClass:
                prefix = ontologyClass.split(":", 1)[0]
                if prefix not in Ontology.NAMESPACES:
                    report["errors"].append(f"Unknown ontology_class prefix: {prefix}.")

            if checkClassKnown:
                known = ontologyClass in Ontology.TOP_SUPERCLASSES
                expandable = Ontology.ExpandQName(ontologyClass, defaultValue=None) is not None
                if not known and not expandable:
                    report["warnings"].append(f"ontology_class is not known and cannot be expanded: {ontologyClass}.")

            if checkCategory and category is not None and str(category).strip() != "":
                expected = Ontology.CategoryByClass(ontologyClass, defaultValue=None)
                if expected is not None and str(category).strip().lower() != str(expected).strip().lower():
                    report["warnings"].append(
                        f"Category '{category}' does not match inferred category '{expected}' for {ontologyClass}."
                    )

            if ontologyURI is not None and str(ontologyURI).strip() != "":
                expanded = Ontology.ExpandQName(ontologyClass, defaultValue=None)
                if expanded is not None and str(ontologyURI).strip() != str(expanded).strip():
                    report["warnings"].append("ontology_uri does not match expanded ontology_class.")

        report["ok"] = len(report["errors"]) == 0

        if not silent and (report["errors"] or report["warnings"]):
            for error in report["errors"]:
                print("Ontology.Validate - Error:", error)
            for warning in report["warnings"]:
                print("Ontology.Validate - Warning:", warning)

        return report

    @staticmethod
    def ValidateGraph(graph,
                      requireClass=True,
                      requireVertexClasses=True,
                      requireEdgeClasses=True,
                      requireLabels=False,
                      checkConnectivity=True,
                      silent=False):
        """
        Validates ontology annotations on a graph, its vertices, and its edges.

        Parameters
        ----------
        graph : topologic_core.Graph or topologicpy.TGraph
            The input graph.
        requireClass : bool , optional
            If True, the graph itself must have ``ontology_class``. Default is True.
        requireVertexClasses : bool , optional
            If True, every vertex must have ``ontology_class``. Default is True.
        requireEdgeClasses : bool , optional
            If True, every edge must have ``ontology_class``. Default is True.
        requireLabels : bool , optional
            If True, graph elements must have labels. Default is False.
        checkConnectivity : bool , optional
            If True, validates that edge endpoints can be resolved. Default is True.
        silent : bool , optional
            If True, warnings are suppressed. Default is False.

        Returns
        -------
        dict
            A validation report with aggregate status and per-element reports.
        """

        report = {
            "ok": False,
            "errors": [],
            "warnings": [],
            "graph": None,
            "vertices": [],
            "edges": [],
        }

        if graph is None:
            report["errors"].append("The input graph is None.")
            return report

        report["graph"] = Ontology.Validate(graph,
                                             requireClass=requireClass,
                                             requireLabel=requireLabels,
                                             silent=True)
        if not report["graph"].get("ok", False):
            for error in report["graph"].get("errors", []):
                report["errors"].append("Graph: " + error)
        for warning in report["graph"].get("warnings", []):
            report["warnings"].append("Graph: " + warning)

        vertices = Ontology._graph_vertices(graph, asTopologic=False)
        edges = Ontology._graph_edges(graph, asTopologic=False)

        for i, vertex in enumerate(vertices):
            r = Ontology.Validate(vertex,
                                  requireClass=requireVertexClasses,
                                  requireLabel=requireLabels,
                                  silent=True)
            r["index"] = i
            report["vertices"].append(r)
            if not r.get("ok", False):
                for error in r.get("errors", []):
                    report["errors"].append(f"Vertex {i}: {error}")
            for warning in r.get("warnings", []):
                report["warnings"].append(f"Vertex {i}: {warning}")

        for i, edge in enumerate(edges):
            r = Ontology.Validate(edge,
                                  requireClass=requireEdgeClasses,
                                  requireLabel=False,
                                  silent=True)
            r["index"] = i

            if checkConnectivity:
                sv, ev = Ontology._graph_edge_endpoints(graph, edge, vertex_records=vertices)
                if sv is None or ev is None:
                    r.setdefault("errors", []).append("Could not resolve edge start/end vertices.")
                    r["ok"] = False

            report["edges"].append(r)
            if not r.get("ok", False):
                for error in r.get("errors", []):
                    report["errors"].append(f"Edge {i}: {error}")
            for warning in r.get("warnings", []):
                report["warnings"].append(f"Edge {i}: {warning}")

        report["ok"] = len(report["errors"]) == 0

        if not silent and (report["errors"] or report["warnings"]):
            for error in report["errors"]:
                print("Ontology.ValidateGraph - Error:", error)
            for warning in report["warnings"]:
                print("Ontology.ValidateGraph - Warning:", warning)

        return report

    @staticmethod
    def ValidateTTLString(ttlString, silent=False):
        """
        Validates a Turtle string using RDFLib when available.

        Parameters
        ----------
        ttlString : str
            The Turtle string.
        silent : bool , optional
            If True, errors are suppressed. Default is False.

        Returns
        -------
        dict
            A validation report with keys ``ok``, ``errors``, ``warnings``, and ``triple_count``.
        """

        report = {"ok": False, "errors": [], "warnings": [], "triple_count": 0}
        if not isinstance(ttlString, str) or ttlString.strip() == "":
            report["errors"].append("The input ttlString is not a valid string.")
            return report
        try:
            import rdflib
        except Exception as e:
            report["warnings"].append("RDFLib is not installed, so syntax validation could not be performed.")
            report["ok"] = True
            return report
        try:
            g = rdflib.Graph()
            g.parse(data=ttlString, format="turtle")
            report["triple_count"] = len(g)
            report["ok"] = True
            return report
        except Exception as e:
            report["errors"].append(str(e))
            if not silent:
                print("Ontology.ValidateTTLString - Error:", e)
            return report

    @staticmethod
    def ValidateTTLFile(path, silent=False):
        """
        Validates a Turtle file using RDFLib when available.

        Parameters
        ----------
        path : str
            The Turtle file path.
        silent : bool , optional
            If True, errors are suppressed. Default is False.

        Returns
        -------
        dict
            A validation report.
        """

        if path is None:
            return {"ok": False, "errors": ["The input path is None."], "warnings": [], "triple_count": 0}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return Ontology.ValidateTTLString(f.read(), silent=silent)
        except Exception as e:
            if not silent:
                print("Ontology.ValidateTTLFile - Error:", e)
            return {"ok": False, "errors": [str(e)], "warnings": [], "triple_count": 0}

    # -------------------------------------------------------------------------
    # RDFLib export/import
    # -------------------------------------------------------------------------

    @staticmethod
    def RDFGraph(topology,
                 includeGraph=True,
                 includeDictionaries=True,
                 includeBOT=True,
                 namespacePrefix="inst",
                 instanceNamespace="http://w3id.org/topologicpy/instance#",
                 silent=False):
        """
        Returns an RDFLib graph from a topology or graph.

        Parameters
        ----------
        topology : topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        includeGraph : bool , optional
            If True and the input is a graph, vertices and edges are exported. Default is True.
        includeDictionaries : bool , optional
            If True, dictionary keys are exported as data properties. Default is True.
        includeBOT : bool , optional
            If True, BOT type triples are included where mapped. Default is True.
        namespacePrefix : str , optional
            The instance namespace prefix. Default is "inst".
        instanceNamespace : str , optional
            The instance namespace URI. Default is "http://w3id.org/topologicpy/instance#".
        silent : bool , optional
            If True, warnings are suppressed. Default is False.

        Returns
        -------
        rdflib.Graph
            The RDFLib graph, or None if RDFLib is unavailable.
        """

        try:
            import rdflib
            from rdflib import Graph as RDFLibGraph
            from rdflib import Namespace, URIRef, Literal
            from rdflib.namespace import RDF, RDFS, XSD, OWL
        except Exception as e:
            if not silent:
                print("Ontology.RDFGraph - Error: RDFLib is required. Install it with 'pip install rdflib'. Returning None.")
                print("Error:", e)
            return None

        def expand(term):
            if term is None:
                return None
            if isinstance(term, str) and term.startswith('"'):
                return literal_from_turtle(term)
            if isinstance(term, str) and ":" in term and not term.startswith("http"):
                prefix, local = term.split(":", 1)
                if prefix in namespaces:
                    return URIRef(namespaces[prefix] + local)
            if isinstance(term, str) and (term.startswith("http://") or term.startswith("https://")):
                return URIRef(term)
            return URIRef(str(term))

        def literal_from_turtle(text):
            text = str(text)
            if "^^" not in text:
                return Literal(text[1:-1] if text.startswith('"') and text.endswith('"') else text)
            value, dtype = text.split("^^", 1)
            value = value.strip()
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            dtype_ref = expand(dtype.strip())
            if str(dtype_ref) == str(XSD.integer):
                try:
                    return Literal(int(value), datatype=XSD.integer)
                except Exception:
                    return Literal(value, datatype=XSD.integer)
            if str(dtype_ref) == str(XSD.double):
                try:
                    return Literal(float(value), datatype=XSD.double)
                except Exception:
                    return Literal(value, datatype=XSD.double)
            if str(dtype_ref) == str(XSD.boolean):
                return Literal(str(value).lower() == "true", datatype=XSD.boolean)
            return Literal(value, datatype=dtype_ref)

        namespaces = dict(Ontology.NAMESPACES)
        namespaces[namespacePrefix] = instanceNamespace

        rdf_graph = RDFLibGraph()
        for prefix, uri in namespaces.items():
            rdf_graph.bind(prefix, Namespace(uri))

        try:
            from topologicpy.Topology import Topology
            if includeGraph and Ontology._is_graph_like(topology):
                triples = Ontology.GraphTriples(topology,
                                                includeDictionaries=includeDictionaries,
                                                includeBOT=includeBOT,
                                                namespacePrefix=namespacePrefix,
                                                silent=silent)
            else:
                triples = Ontology.Triples(topology,
                                           includeDictionaries=includeDictionaries,
                                           includeBOT=includeBOT,
                                           namespacePrefix=namespacePrefix,
                                           silent=silent)
            for s, p, o in triples:
                rdf_graph.add((expand(s), expand(p), expand(o)))
            return rdf_graph
        except Exception as e:
            if not silent:
                print("Ontology.RDFGraph - Error: Could not build RDF graph. Returning None.")
                print("Error:", e)
            return None

    @staticmethod
    def RDFString(topology,
                  format="turtle",
                  includeGraph=True,
                  includeDictionaries=True,
                  includeBOT=True,
                  namespacePrefix="inst",
                  instanceNamespace="http://w3id.org/topologicpy/instance#",
                  silent=False):
        """
        Serializes a topology or graph to an RDF string using RDFLib.

        Parameters
        ----------
        topology : topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        format : str , optional
            RDFLib serialization format, e.g. "turtle", "xml", "json-ld", "nt". Default is "turtle".
        includeGraph : bool , optional
            If True and the input is a graph, vertices and edges are exported. Default is True.
        includeDictionaries : bool , optional
            If True, dictionary keys are exported. Default is True.
        includeBOT : bool , optional
            If True, BOT type triples are included where mapped. Default is True.
        namespacePrefix : str , optional
            The instance namespace prefix. Default is "inst".
        instanceNamespace : str , optional
            The instance namespace URI. Default is "http://w3id.org/topologicpy/instance#".
        silent : bool , optional
            If True, warnings are suppressed. Default is False.

        Returns
        -------
        str
            The serialized RDF string, or None on failure.
        """

        g = Ontology.RDFGraph(topology,
                              includeGraph=includeGraph,
                              includeDictionaries=includeDictionaries,
                              includeBOT=includeBOT,
                              namespacePrefix=namespacePrefix,
                              instanceNamespace=instanceNamespace,
                              silent=silent)
        if g is None:
            return None
        try:
            result = g.serialize(format=format)
            if isinstance(result, bytes):
                return result.decode("utf-8")
            return str(result)
        except Exception as e:
            if not silent:
                print("Ontology.RDFString - Error: Could not serialize RDF graph. Returning None.")
                print("Error:", e)
            return None

    @staticmethod
    def ExportRDF(topology,
                  path,
                  format=None,
                  includeGraph=True,
                  includeDictionaries=True,
                  includeBOT=True,
                  namespacePrefix="inst",
                  instanceNamespace="http://w3id.org/topologicpy/instance#",
                  silent=False):
        """
        Exports a topology or graph to an RDF file using RDFLib.

        Parameters
        ----------
        topology : topologic_core.Topology, topologic_core.Graph, or topologicpy.TGraph
            The input topology or graph.
        path : str
            The output file path.
        format : str , optional
            RDFLib serialization format. If None, it is inferred from the extension. Default is None.
        includeGraph : bool , optional
            If True and the input is a graph, vertices and edges are exported. Default is True.
        includeDictionaries : bool , optional
            If True, dictionary keys are exported. Default is True.
        includeBOT : bool , optional
            If True, BOT type triples are included where mapped. Default is True.
        namespacePrefix : str , optional
            The instance namespace prefix. Default is "inst".
        instanceNamespace : str , optional
            The instance namespace URI. Default is "http://w3id.org/topologicpy/instance#".
        silent : bool , optional
            If True, warnings are suppressed. Default is False.

        Returns
        -------
        str
            The output path, or None on failure.
        """

        if path is None:
            if not silent:
                print("Ontology.ExportRDF - Error: The input path is None. Returning None.")
            return None
        if format is None:
            ext = str(path).lower().rsplit(".", 1)[-1] if "." in str(path) else "ttl"
            format = {
                "ttl": "turtle",
                "rdf": "xml",
                "xml": "xml",
                "jsonld": "json-ld",
                "json": "json-ld",
                "nt": "nt",
                "n3": "n3",
            }.get(ext, "turtle")
        g = Ontology.RDFGraph(topology,
                              includeGraph=includeGraph,
                              includeDictionaries=includeDictionaries,
                              includeBOT=includeBOT,
                              namespacePrefix=namespacePrefix,
                              instanceNamespace=instanceNamespace,
                              silent=silent)
        if g is None:
            return None
        try:
            g.serialize(destination=path, format=format)
            return path
        except Exception as e:
            if not silent:
                print("Ontology.ExportRDF - Error: Could not export RDF file. Returning None.")
                print("Error:", e)
            return None

    @staticmethod
    def GraphByRDFGraph(rdfGraph,
                        graphSubject=None,
                        namespacePrefix="inst",
                        tolerance=0.0001,
                        asTGraph: bool = False,
                        silent=False):
        """
        Reconstructs a TopologicPy graph from an RDFLib graph exported by Ontology.RDFGraph or Ontology.ExportRDF.

        Parameters
        ----------
        rdfGraph : rdflib.Graph
            The input RDFLib graph.
        graphSubject : str , optional
            The graph subject URI or QName. If None, the first subject with ``top:hasNode`` is used. Default is None.
        namespacePrefix : str , optional
            The preferred instance namespace prefix for stored ``uri`` values. Default is "inst".
        tolerance : float , optional
            The graph creation tolerance. Default is 0.0001.
        asTGraph : bool , optional
            If True, returns a topologicpy.TGraph. If False, returns a legacy Graph. Default is False.
        silent : bool , optional
            If True, warnings are suppressed. Default is False.

        Returns
        -------
        topologic_core.Graph or topologicpy.TGraph
            The reconstructed graph, or None on failure.
        """

        try:
            import rdflib
            from rdflib import URIRef, Literal
            from rdflib.namespace import RDF, RDFS
            from topologicpy.Vertex import Vertex
            from topologicpy.Edge import Edge
            from topologicpy.Topology import Topology
            from topologicpy.Dictionary import Dictionary
        except Exception as e:
            if not silent:
                print("Ontology.GraphByRDFGraph - Error: Missing dependency or TopologicPy class. Returning None.")
                print("Error:", e)
            return None

        if rdfGraph is None:
            if not silent:
                print("Ontology.GraphByRDFGraph - Error: The input rdfGraph is None. Returning None.")
            return None

        ns = {k: URIRef(v) for k, v in Ontology.NAMESPACES.items()}
        TOP = Ontology.NAMESPACES["top"]
        RDF_TYPE = RDF.type
        RDFS_LABEL = RDFS.label

        def uri(qname):
            if qname is None:
                return None
            if isinstance(qname, URIRef):
                return qname
            qname = str(qname)
            if qname.startswith("http://") or qname.startswith("https://"):
                return URIRef(qname)
            if ":" in qname:
                prefix, local = qname.split(":", 1)
                if prefix in Ontology.NAMESPACES:
                    return URIRef(Ontology.NAMESPACES[prefix] + local)
            return URIRef(qname)

        def qname(term):
            try:
                return rdfGraph.namespace_manager.normalizeUri(term).replace("<", "").replace(">", "")
            except Exception:
                return str(term)

        def local_key(term):
            q = qname(term)
            if ":" in q:
                return q.split(":", 1)[1]
            if "#" in q:
                return q.rsplit("#", 1)[1]
            if "/" in q:
                return q.rsplit("/", 1)[1]
            return q

        def literal_value(value):
            if isinstance(value, Literal):
                try:
                    return value.toPython()
                except Exception:
                    return str(value)
            return str(value)

        hasNode = uri("top:hasNode")
        hasRelationship = uri("top:hasRelationship")
        hasStartVertex = uri("top:hasStartVertex")
        hasEndVertex = uri("top:hasEndVertex")
        hasX = uri("top:hasX")
        hasY = uri("top:hasY")
        hasZ = uri("top:hasZ")
        categoryP = uri("top:category")

        graph_subject = uri(graphSubject) if graphSubject is not None else None
        if graph_subject is None:
            for s, _, _ in rdfGraph.triples((None, hasNode, None)):
                graph_subject = s
                break
        if graph_subject is None:
            if not silent:
                print("Ontology.GraphByRDFGraph - Error: Could not find a graph subject with top:hasNode. Returning None.")
            return None

        node_subjects = list(rdfGraph.objects(graph_subject, hasNode))
        rel_subjects = list(rdfGraph.objects(graph_subject, hasRelationship))

        def subject_props(subject):
            props = {Ontology.URI_KEY: qname(subject)}
            for obj in rdfGraph.objects(subject, RDF_TYPE):
                q = qname(obj)
                if isinstance(q, str) and q.startswith("top:"):
                    props[Ontology.ONTOLOGY_CLASS_KEY] = q
                    expanded = Ontology.ExpandQName(q, defaultValue=None)
                    if expanded is not None:
                        props[Ontology.ONTOLOGY_URI_KEY] = expanded
                    cat = Ontology.CategoryByClass(q, defaultValue=None)
                    if cat is not None and Ontology.CATEGORY_KEY not in props:
                        props[Ontology.CATEGORY_KEY] = cat
                    break
            for obj in rdfGraph.objects(subject, RDFS_LABEL):
                props[Ontology.LABEL_KEY] = literal_value(obj)
                break
            for obj in rdfGraph.objects(subject, categoryP):
                props[Ontology.CATEGORY_KEY] = literal_value(obj)
                break
            skip = {RDF_TYPE, RDFS_LABEL, categoryP, hasStartVertex, hasEndVertex}
            for _, p, o in rdfGraph.triples((subject, None, None)):
                if p in skip:
                    continue
                if p in (hasNode, hasRelationship):
                    continue
                if isinstance(o, Literal):
                    key = local_key(p)
                    if key not in props:
                        props[key] = literal_value(o)
            return props

        vertices = []
        vertex_dictionaries = []
        subject_to_index = {}
        subject_to_vertex = {}
        for i, subject in enumerate(node_subjects):
            props = subject_props(subject)
            x = props.get("x", None)
            y = props.get("y", None)
            z = props.get("z", None)
            if x is None:
                objs = list(rdfGraph.objects(subject, hasX))
                x = literal_value(objs[0]) if objs else float(i)
            if y is None:
                objs = list(rdfGraph.objects(subject, hasY))
                y = literal_value(objs[0]) if objs else 0.0
            if z is None:
                objs = list(rdfGraph.objects(subject, hasZ))
                z = literal_value(objs[0]) if objs else 0.0
            try:
                x = float(x)
            except Exception:
                x = float(i)
            try:
                y = float(y)
            except Exception:
                y = 0.0
            try:
                z = float(z)
            except Exception:
                z = 0.0
            props.setdefault("id", Ontology._safe_local_name(qname(subject)))
            props.setdefault("x", x)
            props.setdefault("y", y)
            props.setdefault("z", z)
            vertex_dictionaries.append(props)
            subject_to_index[subject] = i
            if not asTGraph:
                v = Vertex.ByCoordinates(x, y, z)
                v = Topology.SetDictionary(v, Dictionary.ByPythonDictionary(props))
                subject_to_vertex[subject] = v
                vertices.append(v)
            else:
                vertices.append(dict(props))

        edge_dictionaries = []
        edges = []
        for subject in rel_subjects:
            start_objs = list(rdfGraph.objects(subject, hasStartVertex))
            end_objs = list(rdfGraph.objects(subject, hasEndVertex))
            if not start_objs or not end_objs:
                continue
            props = subject_props(subject)
            props.setdefault("id", Ontology._safe_local_name(qname(subject)))
            if asTGraph:
                src = subject_to_index.get(start_objs[0])
                dst = subject_to_index.get(end_objs[0])
                if src is None or dst is None:
                    continue
                ed = dict(props)
                ed["src"] = src
                ed["dst"] = dst
                edge_dictionaries.append(ed)
            else:
                sv = subject_to_vertex.get(start_objs[0])
                ev = subject_to_vertex.get(end_objs[0])
                if sv is None or ev is None:
                    continue
                try:
                    e = Edge.ByStartVertexEndVertex(sv, ev)
                except Exception:
                    try:
                        e = Edge.ByVertices([sv, ev], tolerance=tolerance, silent=True)
                    except Exception:
                        e = None
                if e is None:
                    continue
                e = Topology.SetDictionary(e, Dictionary.ByPythonDictionary(props))
                edges.append(e)

        if not vertices:
            return None

        graph_props = subject_props(graph_subject)
        graph_props.setdefault(Ontology.ONTOLOGY_CLASS_KEY, "top:Graph")
        graph_props.setdefault(Ontology.CATEGORY_KEY, "graph")

        if asTGraph:
            try:
                from topologicpy.TGraph import TGraph
                graph = TGraph.ByVerticesEdges(vertices=vertices,
                                               edges=edge_dictionaries,
                                               dictionary=graph_props,
                                               directed=True,
                                               allowSelfLoops=True,
                                               allowParallelEdges=True,
                                               tolerance=tolerance,
                                               silent=silent,
                                               ontology=False)
                return graph
            except Exception as e:
                if not silent:
                    print("Ontology.GraphByRDFGraph - Error: Could not create TGraph. Returning None.")
                    print("Error:", e)
                return None

        try:
            from topologicpy.Graph import Graph
            graph = Graph.ByVerticesEdges(vertices, edges, tolerance=tolerance, silent=silent)
        except TypeError:
            try:
                from topologicpy.Graph import Graph
                graph = Graph.ByVerticesEdges(vertices, edges)
            except Exception:
                graph = None
        if graph is None:
            return None

        graph = Topology.SetDictionary(graph, Dictionary.ByPythonDictionary(graph_props))
        return graph

    @staticmethod
    def GraphByRDFFile(path,
                       format=None,
                       graphSubject=None,
                       namespacePrefix="inst",
                       tolerance=0.0001,
                       asTGraph: bool = False,
                       silent=False):
        """
        Reconstructs a TopologicPy graph from an RDF/Turtle file.

        Parameters
        ----------
        path : str
            The RDF file path.
        format : str , optional
            RDFLib parser format. If None, inferred from the extension. Default is None.
        graphSubject : str , optional
            The graph subject URI or QName. Default is None.
        namespacePrefix : str , optional
            The instance namespace prefix. Default is "inst".
        tolerance : float , optional
            Graph creation tolerance. Default is 0.0001.
        asTGraph : bool , optional
            If True, returns a topologicpy.TGraph. If False, returns a legacy Graph. Default is False.
        silent : bool , optional
            If True, warnings are suppressed. Default is False.

        Returns
        -------
        topologic_core.Graph or topologicpy.TGraph
            The reconstructed graph, or None on failure.
        """

        if path is None:
            if not silent:
                print("Ontology.GraphByRDFFile - Error: The input path is None. Returning None.")
            return None
        try:
            import rdflib
        except Exception as e:
            if not silent:
                print("Ontology.GraphByRDFFile - Error: RDFLib is required. Install it with 'pip install rdflib'. Returning None.")
                print("Error:", e)
            return None
        if format is None:
            ext = str(path).lower().rsplit(".", 1)[-1] if "." in str(path) else "ttl"
            format = {
                "ttl": "turtle",
                "rdf": "xml",
                "xml": "xml",
                "jsonld": "json-ld",
                "json": "json-ld",
                "nt": "nt",
                "n3": "n3",
            }.get(ext, "turtle")
        try:
            g = rdflib.Graph()
            for prefix, uri in Ontology.NAMESPACES.items():
                g.bind(prefix, uri)
            g.parse(path, format=format)
            return Ontology.GraphByRDFGraph(g,
                                            graphSubject=graphSubject,
                                            namespacePrefix=namespacePrefix,
                                            tolerance=tolerance,
                                            asTGraph=asTGraph,
                                            silent=silent)
        except Exception as e:
            if not silent:
                print("Ontology.GraphByRDFFile - Error: Could not parse RDF file. Returning None.")
                print("Error:", e)
            return None

    @staticmethod
    def GraphByTTL(path,
                   graphSubject=None,
                   namespacePrefix="inst",
                   tolerance=0.0001,
                   asTGraph: bool = False,
                   silent=False):
        """
        Reconstructs a TopologicPy graph from a Turtle file.

        Parameters
        ----------
        path : str
            The Turtle file path.
        graphSubject : str , optional
            The graph subject URI or QName. Default is None.
        namespacePrefix : str , optional
            The instance namespace prefix. Default is "inst".
        tolerance : float , optional
            Graph creation tolerance. Default is 0.0001.
        asTGraph : bool , optional
            If True, returns a topologicpy.TGraph. If False, returns a legacy Graph. Default is False.
        silent : bool , optional
            If True, warnings are suppressed. Default is False.

        Returns
        -------
        topologic_core.Graph or topologicpy.TGraph
            The reconstructed graph, or None on failure.
        """

        return Ontology.GraphByRDFFile(path,
                                       format="turtle",
                                       graphSubject=graphSubject,
                                       namespacePrefix=namespacePrefix,
                                       tolerance=tolerance,
                                       asTGraph=asTGraph,
                                       silent=silent)

    @staticmethod
    def GraphByTTLString(ttlString,
                         graphSubject=None,
                         namespacePrefix="inst",
                         tolerance=0.0001,
                         asTGraph: bool = False,
                         silent=False):
        """
        Reconstructs a TopologicPy graph from a Turtle string.

        Parameters
        ----------
        ttlString : str
            The Turtle string.
        graphSubject : str , optional
            The graph subject URI or QName. Default is None.
        namespacePrefix : str , optional
            The instance namespace prefix. Default is "inst".
        tolerance : float , optional
            Graph creation tolerance. Default is 0.0001.
        asTGraph : bool , optional
            If True, returns a topologicpy.TGraph. If False, returns a legacy Graph. Default is False.
        silent : bool , optional
            If True, warnings are suppressed. Default is False.

        Returns
        -------
        topologic_core.Graph or topologicpy.TGraph
            The reconstructed graph, or None on failure.
        """

        try:
            import rdflib
        except Exception as e:
            if not silent:
                print("Ontology.GraphByTTLString - Error: RDFLib is required. Install it with 'pip install rdflib'. Returning None.")
                print("Error:", e)
            return None
        if not isinstance(ttlString, str) or ttlString.strip() == "":
            if not silent:
                print("Ontology.GraphByTTLString - Error: The input ttlString is not valid. Returning None.")
            return None
        try:
            g = rdflib.Graph()
            for prefix, uri in Ontology.NAMESPACES.items():
                g.bind(prefix, uri)
            g.parse(data=ttlString, format="turtle")
            return Ontology.GraphByRDFGraph(g,
                                            graphSubject=graphSubject,
                                            namespacePrefix=namespacePrefix,
                                            tolerance=tolerance,
                                            asTGraph=asTGraph,
                                            silent=silent)
        except Exception as e:
            if not silent:
                print("Ontology.GraphByTTLString - Error: Could not parse Turtle string. Returning None.")
                print("Error:", e)
            return None

