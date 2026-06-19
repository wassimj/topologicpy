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

from __future__ import annotations

from typing import Any, Dict, Optional


class GraphDB:
    """
    A provider-neutral dispatcher for graph database backends.

    GraphDB is intentionally lightweight. It does not implement graph database
    logic itself. Instead, it stores a provider name and a backend manager, then
    dispatches calls to the corresponding TopologicPy database class, currently
    Kuzu or Neo4j.

    The expected usage pattern is:

        from topologicpy.GraphDB import GraphDB
        from topologicpy.Kuzu import Kuzu

        manager = Kuzu.Manager("my_db")
        graphdb = GraphDB.ByParameters(provider="kuzu", manager=manager)

        GraphDB.EnsureSchema(graphdb)
        GraphDB.UpsertGraph(graphdb, graph, graphIDKey="graph_id",
                            vertexIDKey="node_id", vertexLabelKey="label")

    For Neo4j:

        from topologicpy.GraphDB import GraphDB
        from topologicpy.Neo4j import Neo4j

        driver = Neo4j.Connect(url, username, password)
        graphdb = GraphDB.ByParameters(provider="neo4j", manager=driver,
                                       database="neo4j")

        GraphDB.EnsureSchema(graphdb)
    """

    # -------------------------------------------------------------------------
    # Construction and inspection
    # -------------------------------------------------------------------------

    @staticmethod
    def ByParameters(provider: str,
                     manager=None,
                     database: str = None,
                     options: dict = None,
                     silent: bool = False) -> dict:
        """
        Creates a provider-neutral graph database descriptor.

        Parameters
        ----------
        provider : str
            The backend provider name. Supported values are "kuzu" and "neo4j" and "ladybug".
        manager : object , optional
            The backend-specific manager. For Kuzu, this is usually the object
            returned by Kuzu.Manager(...). For Neo4j, this is usually the driver
            returned by Neo4j.Connect(...).
        database : str , optional
            Optional database name. This is mainly used by Neo4j. Default is None.
        options : dict , optional
            Optional provider-specific settings. Default is None.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        dict
            A graph database descriptor, or None on error.
        """
        try:
            if provider is None:
                if not silent:
                    print("GraphDB.ByParameters - Error: The input provider is None. Returning None.")
                return None
            provider = str(provider).strip().lower()
            aliases = {
                "kuzu": "kuzu",
                "kùzu": "kuzu",
                "neo4j": "neo4j",
                "neo": "neo4j",
                "ladybug": "ladybugdb",
                "ladybugdb": "ladybugdb",
                "lbug": "ladybugdb",
            }
            provider = aliases.get(provider, provider)
            if provider not in ["kuzu", "neo4j", "ladybugdb"]:
                if not silent:
                    print(f"GraphDB.ByParameters - Error: Unsupported provider '{provider}'. Returning None.")
                return None
            return {
                "provider": provider,
                "manager": manager,
                "database": database,
                "options": dict(options or {}),
            }
        except Exception as e:
            if not silent:
                print(f"GraphDB.ByParameters - Error: {e}. Returning None.")
            return None

    @staticmethod
    def Provider(graphdb, silent: bool = False):
        """
        Returns the provider string stored in the input graph database descriptor.

        Parameters
        ----------
        graphdb : dict
            The input graph database descriptor.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        str
            The provider string, or None on error.
        """
        try:
            return str(graphdb.get("provider")).strip().lower()
        except Exception as e:
            if not silent:
                print(f"GraphDB.Provider - Error: {e}. Returning None.")
            return None

    @staticmethod
    def Manager(graphdb, silent: bool = False):
        """
        Returns the backend-specific manager stored in the input graph database descriptor.

        Parameters
        ----------
        graphdb : dict
            The input graph database descriptor.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        object
            The backend manager, or None on error.
        """
        try:
            return graphdb.get("manager")
        except Exception as e:
            if not silent:
                print(f"GraphDB.Manager - Error: {e}. Returning None.")
            return None

    @staticmethod
    def Database(graphdb, silent: bool = False):
        """
        Returns the database name stored in the input graph database descriptor.

        Parameters
        ----------
        graphdb : dict
            The input graph database descriptor.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        str
            The database name, or None.
        """
        try:
            return graphdb.get("database")
        except Exception as e:
            if not silent:
                print(f"GraphDB.Database - Error: {e}. Returning None.")
            return None

    @staticmethod
    def Options(graphdb, silent: bool = False):
        """
        Returns the options dictionary stored in the input graph database descriptor.

        Parameters
        ----------
        graphdb : dict
            The input graph database descriptor.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        dict
            The options dictionary, or None on error.
        """
        try:
            return dict(graphdb.get("options") or {})
        except Exception as e:
            if not silent:
                print(f"GraphDB.Options - Error: {e}. Returning None.")
            return None

    # -------------------------------------------------------------------------
    # Internal dispatch helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _backend(graphdb, silent: bool = False):
        """
        Returns the backend class corresponding to the provider.

        Parameters
        ----------
        graphdb : dict
            The input graph database descriptor.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        class
            The backend class, or None on error.
        """
        provider = GraphDB.Provider(graphdb, silent=silent)
        if provider == "kuzu":
            try:
                from topologicpy.Kuzu import Kuzu
            except Exception:
                try:
                    from Kuzu import Kuzu
                except Exception as e:
                    if not silent:
                        print(f"GraphDB._backend - Error: Could not import Kuzu: {e}. Returning None.")
                    return None
            return Kuzu

        if provider == "neo4j":
            try:
                from topologicpy.Neo4j import Neo4j
            except Exception:
                try:
                    from Neo4j import Neo4j
                except Exception as e:
                    if not silent:
                        print(f"GraphDB._backend - Error: Could not import Neo4j: {e}. Returning None.")
                    return None
            return Neo4j
        
        if provider == "ladybugdb":
            try:
                from topologicpy.LadybugDB import LadybugDB
            except Exception:
                try:
                    from LadybugDB import LadybugDB
                except Exception as e:
                    if not silent:
                        print(f"GraphDB._backend - Error: Could not import LadybugDB: {e}. Returning None.")
                    return None
            return LadybugDB

        if not silent:
            print(f"GraphDB._backend - Error: Unsupported provider '{provider}'. Returning None.")
        return None

    @staticmethod
    def _manager_database(graphdb):
        manager = graphdb.get("manager") if isinstance(graphdb, dict) else None
        database = graphdb.get("database") if isinstance(graphdb, dict) else None
        return manager, database


    # -------------------------------------------------------------------------
    # Ontology helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def OntologyKeys() -> list:
        """
        Returns the canonical ontology-related dictionary keys that should be
        preserved by graph database backends.

        Returns
        -------
        list
            The list of ontology-related dictionary keys.
        """
        return [
            "ontology_class",
            "ontology_uri",
            "uri",
            "label",
            "category",
            "ifc_class",
            "ifc_guid",
            "source",
            "derived_from",
            "generated_by",
        ]

    @staticmethod
    def _ontology_option(graphdb, key: str, defaultValue=None):
        """Returns an ontology option from graphdb['options']."""
        try:
            options = GraphDB.Options(graphdb, silent=True) or {}
            return options.get(key, defaultValue)
        except Exception:
            return defaultValue

    @staticmethod
    @staticmethod
    def _annotate_graph_ontology(graph,
                                 ontology: bool = True,
                                 graphClass: str = "top:Graph",
                                 vertexClass: str = "top:Node",
                                 edgeClass: str = "top:Relationship",
                                 generatedBy: str = "GraphDB.UpsertGraph",
                                 silent: bool = True):
        """
        Annotates a TopologicPy Graph or TGraph, its vertices, and its edges using
        the TopologicPy ontology.

        This method is intentionally defensive and non-breaking. If ontology
        support is unavailable, or if a backend returns an object that cannot be
        inspected as a TopologicPy graph, the input graph is returned unchanged.
        """
        if ontology is False or graph is None:
            return graph

        # ------------------------------------------------------------------
        # TGraph path. This must be checked before Topology.IsInstance(...,
        # "Graph") because Topology may intentionally treat TGraph as graph-
        # compatible.
        # ------------------------------------------------------------------
        try:
            from topologicpy.TGraph import TGraph
            is_tgraph = isinstance(graph, TGraph)
        except Exception:
            TGraph = None
            is_tgraph = False

        if is_tgraph:
            try:
                if hasattr(TGraph, "_OntologyAnnotateGraph"):
                    graph = TGraph._OntologyAnnotateGraph(
                        graph,
                        graphClass=graphClass,
                        vertexClass=vertexClass,
                        edgeClass=edgeClass,
                        generatedBy=generatedBy,
                        ontology=ontology,
                        silent=True,
                    )
            except TypeError:
                try:
                    graph = TGraph._OntologyAnnotateGraph(
                        graph,
                        graphClass=graphClass,
                        generatedBy=generatedBy,
                        ontology=ontology,
                        silent=True,
                    )
                except Exception:
                    pass
            except Exception:
                pass

            # Ensure graph-level metadata exists even if the TGraph ontology
            # helper is absent or older.
            try:
                gd = TGraph.Dictionary(graph)
                gd = dict(gd) if isinstance(gd, dict) else {}
                gd.setdefault("ontology_class", graphClass)
                gd.setdefault("category", "graph")
                gd.setdefault("generated_by", generatedBy)
                graph.SetDictionary(gd)
            except Exception:
                pass

            # Ensure database-friendly vertex ids and ontology metadata.
            try:
                vertices = TGraph.Vertices(graph, asTopologic=False, activeOnly=True) or []
            except Exception:
                vertices = []

            for i, vertex in enumerate(vertices):
                try:
                    index = vertex.get("index", i)
                    d = dict(vertex.get("dictionary", {}) or {})
                    d.setdefault("ontology_class", vertexClass)
                    d.setdefault("category", "node")
                    d.setdefault("id", index)
                    try:
                        graph.SetVertexDictionary(index, d)
                    except Exception:
                        graph._vertices[index]["dictionary"] = d
                except Exception:
                    continue

            # Ensure edge ontology metadata.
            try:
                edges = TGraph.Edges(graph, asTopologic=False, activeOnly=True) or []
            except Exception:
                edges = []

            for edge in edges:
                try:
                    index = edge.get("index")
                    if index is None:
                        continue
                    d = dict(edge.get("dictionary", {}) or {})
                    d.setdefault("ontology_class", edgeClass)
                    d.setdefault("category", "relationship")
                    try:
                        graph.SetEdgeDictionary(index, d)
                    except Exception:
                        graph._edges[index]["dictionary"] = d
                except Exception:
                    continue

            return graph

        # ------------------------------------------------------------------
        # Legacy Graph path.
        # ------------------------------------------------------------------
        try:
            from topologicpy.Topology import Topology
            from topologicpy.Graph import Graph
            from topologicpy.Ontology import Ontology
            from topologicpy.Dictionary import Dictionary
        except Exception:
            return graph

        try:
            if not Topology.IsInstance(graph, "graph"):
                return graph
        except Exception:
            return graph

        def _class(obj):
            try:
                return Ontology.Class(obj, defaultValue=None)
            except Exception:
                return None

        def _category(obj):
            try:
                return Ontology.Category(obj, defaultValue=None)
            except Exception:
                return None

        def _value(obj, key, defaultValue=None):
            try:
                d = Topology.Dictionary(obj)
                return Dictionary.ValueAtKey(d, key, defaultValue)
            except Exception:
                return defaultValue

        try:
            if _class(graph) is None:
                graph = Ontology.Annotate(graph,
                                          ontologyClass=graphClass,
                                          category="graph",
                                          generatedBy=generatedBy,
                                          silent=True)
            else:
                graph = Ontology.Annotate(graph,
                                          generatedBy=generatedBy,
                                          silent=True)
        except Exception:
            pass

        try:
            vertices = Graph.Vertices(graph) or []
        except Exception:
            vertices = []

        for i, vertex in enumerate(vertices):
            try:
                if _class(vertex) is None:
                    vertex = Ontology.Annotate(vertex,
                                               ontologyClass=vertexClass,
                                               category="node",
                                               silent=True)
                elif _category(vertex) is None:
                    vertex = Ontology.SetCategory(vertex, "node", silent=True)

                # Preserve a graph-database-friendly id if none exists.
                if _value(vertex, "id", None) is None:
                    d = Topology.Dictionary(vertex)
                    d = Dictionary.SetValueAtKey(d, "id", i)
                    Topology.SetDictionary(vertex, d, silent=True)
            except Exception:
                continue

        try:
            edges = Graph.Edges(graph, vertices) or []
        except TypeError:
            try:
                edges = Graph.Edges(graph) or []
            except Exception:
                edges = []
        except Exception:
            edges = []

        for edge in edges:
            try:
                if _class(edge) is None:
                    edge = Ontology.Annotate(edge,
                                             ontologyClass=edgeClass,
                                             category="relationship",
                                             silent=True)
                elif _category(edge) is None:
                    edge = Ontology.SetCategory(edge, "relationship", silent=True)
            except Exception:
                continue

        return graph

    @staticmethod
    def _annotate_graph_result(result,
                               ontology: bool = True,
                               graphClass: str = "top:Graph",
                               vertexClass: str = "top:Node",
                               edgeClass: str = "top:Relationship",
                               generatedBy: str = "GraphDB.GraphByID",
                               silent: bool = True):
        """Annotates a returned graph or list of graphs, if possible."""
        if ontology is False or result is None:
            return result
        if isinstance(result, list):
            return [GraphDB._annotate_graph_ontology(g,
                                                     ontology=ontology,
                                                     graphClass=graphClass,
                                                     vertexClass=vertexClass,
                                                     edgeClass=edgeClass,
                                                     generatedBy=generatedBy,
                                                     silent=silent) for g in result]
        return GraphDB._annotate_graph_ontology(result,
                                                ontology=ontology,
                                                graphClass=graphClass,
                                                vertexClass=vertexClass,
                                                edgeClass=edgeClass,
                                                generatedBy=generatedBy,
                                                silent=silent)

    @staticmethod
    def _call(graphdb, methodName: str, *args, silent: bool = False, **kwargs):
        """
        Calls a backend method.

        GraphDB["options"] are only injected for methods that benefit from them,
        such as UpsertGraph.

        Precedence:
        explicit kwargs > graphdb["options"] > backend defaults
        """
        try:
            backend = GraphDB._backend(graphdb, silent=silent)
            if backend is None:
                return None

            manager = GraphDB.Manager(graphdb, silent=silent)
            if manager is None:
                if not silent:
                    print(f"GraphDB.{methodName} - Error: The graph database manager is None. Returning None.")
                return None

            try:
                method = getattr(backend, methodName)
            except Exception:
                if not silent:
                    print(f"GraphDB.{methodName} - Error: Backend does not implement {methodName}. Returning None.")
                return None

            provider = GraphDB.Provider(graphdb, silent=True)

            # Start with explicit kwargs only.
            call_kwargs = dict(kwargs)

            # Only inject graphdb options into methods that are graph-persistence related.
            methods_using_options = {
                "UpsertGraph",
            }

            if methodName in methods_using_options:
                options = GraphDB.Options(graphdb, silent=True) or {}

                merged = dict(options)
                merged.update(call_kwargs)
                call_kwargs = merged

                # GraphDB-level ontology options are consumed by GraphDB before
                # dispatch. Do not pass them to backend-specific UpsertGraph
                # methods unless those methods explicitly add their own support.
                for _key in [
                    "ontology",
                    "ontologyGraphClass",
                    "ontologyVertexClass",
                    "ontologyEdgeClass",
                    "ontologyGeneratedBy",
                ]:
                    call_kwargs.pop(_key, None)

            # Neo4j database context.
            if provider == "neo4j" and "database" not in call_kwargs:
                db = GraphDB.Database(graphdb, silent=True)
                if db is not None:
                    call_kwargs["database"] = db

            # Kuzu does not use Neo4j database names.
            if provider in ["kuzu", "ladybugdb"]:
                call_kwargs.pop("database", None)

            if "silent" not in call_kwargs:
                call_kwargs["silent"] = silent

            return method(manager, *args, **call_kwargs)

        except Exception as e:
            if not silent:
                print(f"GraphDB.{methodName} - Error: {e}. Returning None.")
            return None

    # -------------------------------------------------------------------------
    # Schema and persistence
    # -------------------------------------------------------------------------

    @staticmethod
    def EnsureSchema(graphdb, silent: bool = False) -> bool:
        """
        Ensures that the backend-specific schema/indexes exist.

        Parameters
        ----------
        graphdb : dict
            The graph database descriptor.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        result = GraphDB._call(graphdb, "EnsureSchema", silent=silent)
        return bool(result)

    @staticmethod
    def ByCSVPath(
        graphdb,
        path,
        graphIDHeader="graph_id", graphLabelHeader="label", graphFeaturesHeader="feat", graphFeaturesKeys=None,
        edgeSRCHeader="src_id", edgeDSTHeader="dst_id", edgeLabelHeader="label",
        edgeTrainMaskHeader="train_mask", edgeValidateMaskHeader="val_mask", edgeTestMaskHeader="test_mask",
        edgeFeaturesHeader="feat", edgeFeaturesKeys=None,
        nodeIDHeader="node_id", nodeLabelHeader="label",
        nodeTrainMaskHeader="train_mask", nodeValidateMaskHeader="val_mask", nodeTestMaskHeader="test_mask",
        nodeFeaturesHeader="feat", nodeXHeader="X", nodeYHeader="Y", nodeZHeader="Z",
        nodeFeaturesKeys=None,
        tolerance=0.0001,
        ontology: bool = True,
        silent=False):
        """
        Reads CSV graph data and upserts all returned graphs into the selected backend.

        The signature mirrors the TopologicPy CSV graph import/export format and the backend-specific ByCSVPath methods in Kuzu and Neo4j.

        Parameters
        ----------
        graphdb : dict
            The graph database descriptor.
        path : str
            The CSV dataset path.
        graphIDHeader : str , optional
            The graph id column/header. Default is "graph_id".
        graphLabelHeader : str , optional
            The graph label column/header. Default is "label".
        graphFeaturesHeader : str , optional
            The graph features prefix/header. Default is "feat".
        graphFeaturesKeys : list , optional
            Optional graph feature keys. Default is None.
        edgeSRCHeader : str , optional
            The edge source id column/header. Default is "src_id".
        edgeDSTHeader : str , optional
            The edge destination id column/header. Default is "dst_id".
        edgeLabelHeader : str , optional
            The edge label column/header. Default is "label".
        edgeTrainMaskHeader : str , optional
            The edge train mask column/header. Default is "train_mask".
        edgeValidateMaskHeader : str , optional
            The edge validation mask column/header. Default is "val_mask".
        edgeTestMaskHeader : str , optional
            The edge test mask column/header. Default is "test_mask".
        edgeFeaturesHeader : str , optional
            The edge features prefix/header. Default is "feat".
        edgeFeaturesKeys : list , optional
            Optional edge feature keys. Default is None.
        nodeIDHeader : str , optional
            The node id column/header. Default is "node_id".
        nodeLabelHeader : str , optional
            The node label column/header. Default is "label".
        nodeTrainMaskHeader : str , optional
            The node train mask column/header. Default is "train_mask".
        nodeValidateMaskHeader : str , optional
            The node validation mask column/header. Default is "val_mask".
        nodeTestMaskHeader : str , optional
            The node test mask column/header. Default is "test_mask".
        nodeFeaturesHeader : str , optional
            The node features prefix/header. Default is "feat".
        nodeXHeader : str , optional
            The node X-coordinate column/header. Default is "X".
        nodeYHeader : str , optional
            The node Y-coordinate column/header. Default is "Y".
        nodeZHeader : str , optional
            The node Z-coordinate column/header. Default is "Z".
        nodeFeaturesKeys : list , optional
            Optional node feature keys. Default is None.
        tolerance : float , optional
            The tolerance passed to the backend CSV importer. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        dict
            A dictionary containing the number of upserted graphs and their ids.
        """
        return GraphDB._call(
            graphdb,
            "ByCSVPath",
            path,
            graphIDHeader=graphIDHeader, graphLabelHeader=graphLabelHeader,
            graphFeaturesHeader=graphFeaturesHeader, graphFeaturesKeys=graphFeaturesKeys,
            edgeSRCHeader=edgeSRCHeader, edgeDSTHeader=edgeDSTHeader, edgeLabelHeader=edgeLabelHeader,
            edgeTrainMaskHeader=edgeTrainMaskHeader, edgeValidateMaskHeader=edgeValidateMaskHeader,
            edgeTestMaskHeader=edgeTestMaskHeader, edgeFeaturesHeader=edgeFeaturesHeader,
            edgeFeaturesKeys=edgeFeaturesKeys,
            nodeIDHeader=nodeIDHeader, nodeLabelHeader=nodeLabelHeader,
            nodeTrainMaskHeader=nodeTrainMaskHeader, nodeValidateMaskHeader=nodeValidateMaskHeader,
            nodeTestMaskHeader=nodeTestMaskHeader, nodeFeaturesHeader=nodeFeaturesHeader,
            nodeXHeader=nodeXHeader, nodeYHeader=nodeYHeader, nodeZHeader=nodeZHeader,
            nodeFeaturesKeys=nodeFeaturesKeys,
            tolerance=tolerance,
            ontology=ontology,
            silent=silent,
        )

    @staticmethod
    def UpsertGraph(graphdb,
                    graph,
                    graphIDKey: str = "graph_id",
                    vertexIDKey: str = "id",
                    vertexLabelKey: str = "label",
                    defaultVertexLabel: str = "Node",
                    vertexCategoryKey: str = "category",
                    defaultVertexCategory: str = "Node",
                    edgeLabelKey: str = "label",
                    defaultEdgeLabel: str = "CONNECTED_TO",
                    edgeCategoryKey: str = "category",
                    defaultEdgeCategory: str = "Edge",
                    bidirectional: bool = True,
                    overwrite: bool = False,
                    mantissa: int = 6,
                    ontology: bool = True,
                    ontologyGraphClass: str = "top:Graph",
                    ontologyVertexClass: str = "top:Node",
                    ontologyEdgeClass: str = "top:Relationship",
                    silent: bool = False) -> str:
        """
        Upserts a TopologicPy graph into the selected backend.

        Parameters
        ----------
        graphdb : dict
            The graph database descriptor. See GraphDB.ByParameters(...).
        graph : topologic_core.Graph or topologicpy.TGraph
            The graph to be upserted.
        graphIDKey : str , optional
            The dictionary key used to retrieve the graph id. Default is "id".
        vertexIDKey : str , optional
            The dictionary key used to retrieve each vertex id. Default is "id".
        vertexLabelKey : str , optional
            The dictionary key used to retrieve each vertex label. Default is "label".
        defaultVertexLabel : str , optional
            The default vertex label to use if no vertex label is found. Default is "Node".
        vertexCategoryKey : str , optional
            The dictionary key used to retrieve each vertex category. Default is "category".
        defaultVertexCategory : str , optional
            The default vertex category to use if no vertex category is found. Default is "Node".
        edgeLabelKey : str , optional
            The dictionary key used to retrieve each edge label. Default is "label".
        defaultEdgeLabel : str , optional
            The default edge label to use if no edge label is found. Default is "CONNECTED_TO".
        edgeCategoryKey : str , optional
            The dictionary key used to retrieve each edge category. Default is "category".
        defaultEdgeCategory : str , optional
            The default edge category to use if no edge category is found. Default is "Edge".
        bidirectional : bool , optional
            If set to True, edges are written bidirectionally where supported by the backend. Default is True.
        overwrite : bool , optional
            If set to True, an existing graph with the same graph id is overwritten where supported by the backend. Default is False.
        mantissa : int , optional
            The number of decimal places to use when extracting coordinate data. Default is 6.
        ontology : bool , optional
            If True, annotates the graph, vertices, and edges with TopologicPy ontology
            classes before upserting. Default is True.
        ontologyGraphClass : str , optional
            The ontology class assigned to the graph if none exists. Default is "top:Graph".
        ontologyVertexClass : str , optional
            The ontology class assigned to vertices if none exists. Default is "top:Node".
        ontologyEdgeClass : str , optional
            The ontology class assigned to edges if none exists. Default is "top:Relationship".
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        str
            The graph id used, or None on error.
        """

        # GraphDB is the safest place to enforce ontology adherence before
        # the graph is persisted by a backend. This remains non-breaking because
        # it only adds dictionary keys when they are missing.
        try:
            options = GraphDB.Options(graphdb, silent=True) or {}
            ontology = bool(options.get("ontology", ontology))
            ontologyGraphClass = options.get("ontologyGraphClass", ontologyGraphClass)
            ontologyVertexClass = options.get("ontologyVertexClass", ontologyVertexClass)
            ontologyEdgeClass = options.get("ontologyEdgeClass", ontologyEdgeClass)
        except Exception:
            pass

        graph = GraphDB._annotate_graph_ontology(graph,
                                                 ontology=ontology,
                                                 graphClass=ontologyGraphClass,
                                                 vertexClass=ontologyVertexClass,
                                                 edgeClass=ontologyEdgeClass,
                                                 generatedBy="GraphDB.UpsertGraph",
                                                 silent=True)

        kwargs = {
            "graph": graph,
            "graphIDKey": graphIDKey,
            "vertexIDKey": vertexIDKey,
            "vertexLabelKey": vertexLabelKey,
            "defaultVertexLabel": defaultVertexLabel,
            "vertexCategoryKey": vertexCategoryKey,
            "defaultVertexCategory": defaultVertexCategory,
            "edgeLabelKey": edgeLabelKey,
            "defaultEdgeLabel": defaultEdgeLabel,
            "edgeCategoryKey": edgeCategoryKey,
            "defaultEdgeCategory": defaultEdgeCategory,
            "bidirectional": bidirectional,
            "overwrite": overwrite,
            "mantissa": mantissa,
            "ontology": ontology,
            "silent": silent
        }

        return GraphDB._call(graphdb, "UpsertGraph", **kwargs)

    @staticmethod
    def GraphByID(graphdb, graphID: str, ontology: bool = True, silent: bool = False):
        """
        Constructs a TopologicPy graph from the selected backend using a graph id.

        Parameters
        ----------
        graphdb : dict
            The graph database descriptor.
        graphID : str
            The graph id to retrieve.
        ontology : bool , optional
            If True, annotates the returned graph with TopologicPy ontology classes
            where missing. Default is True.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Graph or topologicpy.TGraph
            A TopologicPy Graph or TGraph, or None on error.
        """
        result = GraphDB._call(graphdb, "GraphByID", graphID, silent=silent)
        return GraphDB._annotate_graph_result(result,
                                              ontology=ontology,
                                              generatedBy="GraphDB.GraphByID",
                                              silent=True)

    @staticmethod
    def GraphsByQuery(graphdb, query: str, parameters: dict = None, ontology: bool = True, silent: bool = False):
        """
        Executes a backend query and returns matching TopologicPy graphs.

        Parameters
        ----------
        graphdb : dict
            The graph database descriptor.
        query : str
            The backend query string. For Kuzu and Neo4j this is Cypher.
        parameters : dict , optional
            Query parameters. Default is None.
        ontology : bool , optional
            If True, annotates the returned graphs with TopologicPy ontology classes
            where missing. Default is True.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            A list of TopologicPy graphs, or None on error.
        """
        result = GraphDB._call(graphdb, "GraphsByQuery", query, parameters=parameters, silent=silent)
        return GraphDB._annotate_graph_result(result,
                                              ontology=ontology,
                                              generatedBy="GraphDB.GraphsByQuery",
                                              silent=True)

    @staticmethod
    def DeleteGraph(graphdb, graphID: str, silent: bool = False) -> bool:
        """
        Deletes a graph by id from the selected backend.

        Parameters
        ----------
        graphdb : dict
            The graph database descriptor.
        graphID : str
            The graph id to delete.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        bool
            True on success, False otherwise.
        """
        return bool(GraphDB._call(graphdb, "DeleteGraph", graphID, silent=silent))

    @staticmethod
    def EmptyDatabase(graphdb, dropSchema: bool = False, recreateSchema: bool = True, silent: bool = False) -> bool:
        """
        Empties the selected backend database.

        Parameters
        ----------
        graphdb : dict
            The graph database descriptor.
        dropSchema : bool , optional
            If True, drops the schema where the backend supports it. Default is False.
        recreateSchema : bool , optional
            If True and dropSchema is True, recreates schema afterwards. Default is True.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        bool
            True on success, False otherwise.
        """
        return bool(
            GraphDB._call(
                graphdb,
                "EmptyDatabase",
                dropSchema=dropSchema,
                recreateSchema=recreateSchema,
                silent=silent,
            )
        )

    @staticmethod
    def ListGraphs(graphdb, where: dict = None, limit: int = 100, offset: int = 0, silent: bool = False):
        """
        Lists graph metadata from the selected backend.

        Parameters
        ----------
        graphdb : dict
            The graph database descriptor.
        where : dict , optional
            Simple backend-supported filters. Default is None.
        limit : int , optional
            Maximum number of records to return. Default is 100.
        offset : int , optional
            Number of records to skip. Default is 0.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            A list of graph metadata dictionaries, or None on error.
        """
        return GraphDB._call(graphdb, "ListGraphs", where=where, limit=limit, offset=offset, silent=silent)


    # -------------------------------------------------------------------------
    # Ontology-oriented query helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def VerticesByOntologyClass(graphdb, ontologyClass: str, limit: int = 100, silent: bool = False):
        """
        Returns backend rows for vertices/nodes with the requested ontology class.

        Parameters
        ----------
        graphdb : dict
            The graph database descriptor.
        ontologyClass : str
            The requested ontology class, for example "top:Room".
        limit : int , optional
            Maximum number of rows to return. Default is 100.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            Backend-normalized rows, or None on error.
        """
        provider = GraphDB.Provider(graphdb, silent=silent)
        if provider == "neo4j":
            query = "MATCH (n) WHERE n.ontology_class = $ontologyClass RETURN n LIMIT $limit"
            return GraphDB.Query(graphdb, query, parameters={"ontologyClass": ontologyClass, "limit": int(limit)}, silent=silent)
        if provider == "kuzu":
            query = "MATCH (n) WHERE n.ontology_class = $ontologyClass RETURN n LIMIT $limit"
            return GraphDB.Query(graphdb, query, parameters={"ontologyClass": ontologyClass, "limit": int(limit)}, silent=silent)
        if not silent:
            print(f"GraphDB.VerticesByOntologyClass - Error: Unsupported provider '{provider}'. Returning None.")
        return None

    @staticmethod
    def VerticesByCategory(graphdb, category: str, limit: int = 100, silent: bool = False):
        """
        Returns backend rows for vertices/nodes with the requested category.

        Parameters
        ----------
        graphdb : dict
            The graph database descriptor.
        category : str
            The requested category, for example "space", "node", or "element".
        limit : int , optional
            Maximum number of rows to return. Default is 100.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            Backend-normalized rows, or None on error.
        """
        provider = GraphDB.Provider(graphdb, silent=silent)
        if provider in ("neo4j", "kuzu"):
            query = "MATCH (n) WHERE n.category = $category RETURN n LIMIT $limit"
            return GraphDB.Query(graphdb, query, parameters={"category": category, "limit": int(limit)}, silent=silent)
        if not silent:
            print(f"GraphDB.VerticesByCategory - Error: Unsupported provider '{provider}'. Returning None.")
        return None

    # -------------------------------------------------------------------------
    # Corpus analytics for GraphRAG
    # -------------------------------------------------------------------------

    @staticmethod
    def FetchAllPairs(graphdb, undirected: bool = True, silent: bool = False):
        """
        Returns all label-pair counts from the selected backend.

        Parameters
        ----------
        graphdb : dict
            The graph database descriptor.
        undirected : bool , optional
            If True, normalizes direction so A-B and B-A are counted together.
            Default is True.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            A list of dictionaries describing label pairs and counts.
        """
        return GraphDB._call(graphdb, "FetchAllPairs", undirected=undirected, silent=silent)

    @staticmethod
    def CandidateCountsForLabels(graphdb, labels, excludeLabels=None, limit: int = 50, silent: bool = False):
        """
        Returns likely candidate labels connected to the input labels.

        Parameters
        ----------
        graphdb : dict
            The graph database descriptor.
        labels : list
            The input labels.
        excludeLabels : list , optional
            Labels to exclude from candidates. Default is None.
        limit : int , optional
            Maximum number of candidates to return. Default is 50.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            A list of candidate label dictionaries.
        """
        return GraphDB._call(
            graphdb,
            "CandidateCountsForLabels",
            labels,
            excludeLabels=excludeLabels,
            limit=limit,
            silent=silent,
        )

    @staticmethod
    def MaxNeighborsForLabel(graphdb, label, silent: bool = False):
        """
        Returns the maximum observed neighbour count for vertices with the input label.

        Parameters
        ----------
        graphdb : dict
            The graph database descriptor.
        label : str
            The input vertex label.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        int
            The maximum neighbour count, or None on error.
        """
        return GraphDB._call(graphdb, "MaxNeighborsForLabel", label, silent=silent)

    @staticmethod
    def FindBestExampleForLabel(graphdb, label, attachTo=None, silent: bool = False):
        """
        Returns a representative database example for the input label.

        Parameters
        ----------
        graphdb : dict
            The graph database descriptor.
        label : str
            The input vertex label.
        attachTo : str , optional
            Optional label of the node to which the candidate is expected to attach.
            Default is None.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        dict
            A representative example dictionary, or None on error.
        """
        return GraphDB._call(graphdb, "FindBestExampleForLabel", label, attachTo=attachTo, silent=silent)

    # -------------------------------------------------------------------------
    # Generic query/execute passthroughs
    # -------------------------------------------------------------------------

    @staticmethod
    def Execute(graphdb, query: str, parameters: dict = None, write: bool = False, silent: bool = False):
        """
        Executes a backend query and returns raw backend rows where supported.

        Parameters
        ----------
        graphdb : dict
            The graph database descriptor.
        query : str
            The query to execute.
        parameters : dict , optional
            Query parameters. Default is None.
        write : bool , optional
            If True, executes a write query. Default is False.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            Backend-normalized result rows, or None on error.
        """
        provider = GraphDB.Provider(graphdb, silent=silent)
        manager = GraphDB.Manager(graphdb, silent=silent)
        database = GraphDB.Database(graphdb, silent=True)
        if manager is None:
            if not silent:
                print("GraphDB.Execute - Error: The graph database manager is None. Returning None.")
            return None

        try:
            if provider == "kuzu":
                return manager.exec(query, parameters or {}, write=write)
            if provider == "ladybugdb":
                return manager.exec(query, parameters or {}, write=write)
            if provider == "neo4j":
                backend = GraphDB._backend(graphdb, silent=silent)
                if backend is None:
                    return None
                return backend.Execute(manager, query, parameters=parameters or {}, write=write, database=database, silent=silent)
            if not silent:
                print(f"GraphDB.Execute - Error: Unsupported provider '{provider}'. Returning None.")
            return None
        except Exception as e:
            if not silent:
                print(f"GraphDB.Execute - Error: {e}. Returning None.")
            return None

    @staticmethod
    def Query(graphdb, query: str, parameters: dict = None, silent: bool = False):
        """
        Executes a read query against the selected backend.

        Parameters
        ----------
        graphdb : dict
            The graph database descriptor.
        query : str
            The query to execute.
        parameters : dict , optional
            Query parameters. Default is None.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            Backend-normalized result rows, or None on error.
        """
        return GraphDB.Execute(graphdb, query, parameters=parameters, write=False, silent=silent)
