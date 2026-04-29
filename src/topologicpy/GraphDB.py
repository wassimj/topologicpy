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
            The backend provider name. Supported values are "kuzu" and "neo4j".
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
            }
            provider = aliases.get(provider, provider)
            if provider not in ["kuzu", "neo4j"]:
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

        if not silent:
            print(f"GraphDB._backend - Error: Unsupported provider '{provider}'. Returning None.")
        return None

    @staticmethod
    def _manager_database(graphdb):
        manager = graphdb.get("manager") if isinstance(graphdb, dict) else None
        database = graphdb.get("database") if isinstance(graphdb, dict) else None
        return manager, database

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

            # Neo4j database context.
            if provider == "neo4j" and "database" not in call_kwargs:
                db = GraphDB.Database(graphdb, silent=True)
                if db is not None:
                    call_kwargs["database"] = db

            # Kuzu does not use Neo4j database names.
            if provider == "kuzu":
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
        silent=False):
        """
        Reads CSV graph data using Graph.ByCSVPath and upserts all returned graphs
        into the selected backend.

        The signature mirrors Graph.ByCSVPath and the backend-specific ByCSVPath
        methods in Kuzu and Neo4j.

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
            The tolerance passed to Graph.ByCSVPath. Default is 0.0001.
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
            silent=silent,
        )

    @staticmethod
    def UpsertGraph(graphdb,
                    graph,
                    mantissa: int = 6,
                    silent: bool = False) -> str:
        """
        Upserts a TopologicPy graph into the selected backend.

        Parameters
        ----------
        graphDB: dict
            The graph database descriptor. See GraphDB.ByParameters(...).
        graph: topologic_core.Graph
            The graph to be upserted.
        mantissa : int , optional
            The number of decimal places to use when extracting mesh data. Default is 6.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        str
            The graph id used, or None on error.
        """

        # Make sure your backend graph implementatation supports the following options.

        kwargs = {
            "graph": graph,
            "mantissa": mantissa,
            "silent": silent
        }

        return GraphDB._call(graphdb,
                             "UpsertGraph",
                             **kwargs,
        )

    @staticmethod
    def GraphByID(graphdb, graphID: str, silent: bool = False):
        """
        Constructs a TopologicPy graph from the selected backend using a graph id.

        Parameters
        ----------
        graphdb : dict
            The graph database descriptor.
        graphID : str
            The graph id to retrieve.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Graph
            A TopologicPy graph, or None on error.
        """
        return GraphDB._call(graphdb, "GraphByID", graphID, silent=silent)

    @staticmethod
    def GraphsByQuery(graphdb, query: str, parameters: dict = None, silent: bool = False):
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
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            A list of TopologicPy graphs, or None on error.
        """
        return GraphDB._call(graphdb, "GraphsByQuery", query, parameters=parameters, silent=silent)

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
