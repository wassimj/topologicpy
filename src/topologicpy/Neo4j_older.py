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

import json
import os
import re
import warnings
from typing import Any, Dict, List, Optional

try:
    import neo4j
    from neo4j import GraphDatabase
except Exception:
    warnings.warn("Neo4j - Error: Could not import neo4j. Please install it using pip install neo4j.")
    neo4j = None
    GraphDatabase = None


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------


def _make_json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_make_json_safe(v) for v in value]
    return str(value)


def _json_dumps(value: Any) -> str:
    try:
        return json.dumps(value if value is not None else {}, ensure_ascii=False)
    except Exception:
        try:
            return json.dumps(_make_json_safe(value), ensure_ascii=False)
        except Exception:
            return "{}"


def _json_loads(value: Any, default: Any = None) -> Any:
    if default is None:
        default = {}
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return default


def _normalize_label(label: Any) -> str:
    if label is None:
        return ""
    return str(label).strip()


def _unique_preserve_order(values: List[Any]) -> List[Any]:
    seen = set()
    out = []
    for value in values or []:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _record_to_dict(record: Any) -> Dict[str, Any]:
    if record is None:
        return {}
    if isinstance(record, dict):
        return dict(record)
    try:
        return {key: record[key] for key in record.keys()}
    except Exception:
        pass
    try:
        return dict(record)
    except Exception:
        return {}


def _records_to_dicts(records: Any) -> List[Dict[str, Any]]:
    if records is None:
        return []
    if isinstance(records, list):
        return [_record_to_dict(r) for r in records]
    return [_record_to_dict(records)]


def _value_from_dict(dictionary: Any, key: str = None, default: Any = None) -> Any:
    if key is None:
        return default
    try:
        from topologicpy.Dictionary import Dictionary
        return Dictionary.ValueAtKey(dictionary, key, default)
    except Exception:
        try:
            return dictionary.get(key, default)
        except Exception:
            return default


def _python_dictionary(dictionary: Any) -> Dict[str, Any]:
    if dictionary is None:
        return {}
    if isinstance(dictionary, dict):
        return dict(dictionary)
    try:
        from topologicpy.Dictionary import Dictionary
        d = Dictionary.PythonDictionary(dictionary)
        return dict(d or {})
    except Exception:
        return {}


class Neo4j:
    """
    Neo4j helper class for TopologicPy.

    This implementation keeps the existing static-method style while adding a
    canonical graph-corpus API compatible with Kuzu.py:

    - EnsureSchema
    - ByCSVPath
    - UpsertGraph
    - GraphByID
    - GraphsByQuery
    - DeleteGraph
    - EmptyDatabase
    - ListGraphs
    - FetchAllPairs
    - CandidateCountsForLabels
    - MaxNeighborsForLabel
    - FindBestExampleForLabel

    Canonical storage model:

    (:Graph {id, label, num_nodes, num_edges, props})
    (:Vertex {id, graph_id, label, x, y, z, props})
    (:Vertex)-[:Edge {label, props, graph_id}]->(:Vertex)
    """

    # -------------------------------------------------------------------------
    # Low-level utilities
    # -------------------------------------------------------------------------

    @staticmethod
    def _is_driver(driver):
        return driver is not None and hasattr(driver, "session") and hasattr(driver, "close")

    @staticmethod
    def _sanitize_identifier(value, default="X"):
        if value is None:
            value = default
        value = str(value).strip()
        if len(value) < 1:
            value = default
        value = re.sub(r"[^0-9a-zA-Z_]", "_", value)
        if len(value) < 1:
            value = default
        if not value[0].isalpha() and value[0] != "_":
            value = "_" + value
        return value

    @staticmethod
    def _node_properties(node):
        props = dict(node.items())
        props["id"] = getattr(node, "element_id", None)
        props["labels"] = list(getattr(node, "labels", []))
        return props

    @staticmethod
    def _relationship_properties(relationship):
        props = dict(relationship.items())
        props["id"] = getattr(relationship, "element_id", None)
        props["type"] = getattr(relationship, "type", None)
        return props

    # -------------------------------------------------------------------------
    # Connection and execution
    # -------------------------------------------------------------------------

    @staticmethod
    def Connect(url, username, password, database=None, silent=False):
        """
        Returns a Neo4j driver/manager created from the input connection parameters.

        Parameters
        ----------
        url : str
            The URL of the neo4j database
        username : str
            The neo4j user name
        password : str
            The neo4j password
        database : str, optional
            The name of the neo4j database
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Neo4jDriver
            The neo4j driver/manager
        """
        if GraphDatabase is None:
            if not silent:
                print("Neo4j.Connect - Error: Could not import neo4j. Returning None.")
            return None
        try:
            driver = GraphDatabase.driver(url, auth=(username, password))
            try:
                if database:
                    driver.verify_connectivity(database=database)
                else:
                    driver.verify_connectivity()
            except TypeError:
                driver.verify_connectivity()
            return driver
        except Exception as ex:
            if not silent:
                print("Neo4j.Connect - Error: Could not connect to the Neo4j server. Returning None.")
                print(ex)
            return None
    
    @staticmethod
    def Manager(url: str = None, username: str = None, password: str = None, database=None, silent: bool = False):
        """
        Returns a Neo4j manager created from the input connection parameters.

        Parameters
        ----------
        url : str
            The URL of the neo4j database
        username : str
            The neo4j user name
        password : str
            The neo4j password
        database : str, optional
            The name of the neo4j database
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Neo4jDriver
            The neo4j driver/manager
        """
        if GraphDatabase is None:
            if not silent:
                print("Neo4j.Connect - Error: Could not import neo4j. Returning None.")
            return None
        try:
            manager = GraphDatabase.driver(url, auth=(username, password))
            try:
                if database:
                    manager.verify_connectivity(database=database)
                else:
                    manager.verify_connectivity()
            except TypeError:
                manager.verify_connectivity()
            return manager
        except Exception as ex:
            if not silent:
                print("Neo4j.Connect - Error: Could not connect to the Neo4j server. Returning None.")
                print(ex)
            return None

    @staticmethod
    def Close(driver, silent=False):
        if not Neo4j._is_driver(driver):
            if not silent:
                print("Neo4j.Close - Error: The input driver is not a valid Neo4j driver. Returning False.")
            return False
        try:
            driver.close()
            return True
        except Exception as ex:
            if not silent:
                print("Neo4j.Close - Error: Could not close the driver. Returning False.")
                print(ex)
            return False

    @staticmethod
    def Execute(driver, cypher, parameters=None, write=False, database=None, silent=False):
        """
        Executes the input Cypher statement and returns the resulting records.
        """
        if not Neo4j._is_driver(driver):
            if not silent:
                print("Neo4j.Execute - Error: The input driver is not a valid Neo4j driver. Returning None.")
            return None
        if not isinstance(cypher, str) or len(cypher.strip()) < 1:
            if not silent:
                print("Neo4j.Execute - Error: The input cypher is not a valid string. Returning None.")
            return None
        parameters = parameters or {}

        def _run(tx):
            result = tx.run(cypher, parameters)
            return list(result)

        try:
            kwargs = {}
            if database:
                kwargs["database"] = database
            with driver.session(**kwargs) as session:
                if write:
                    try:
                        return session.execute_write(_run)
                    except AttributeError:
                        return session.write_transaction(_run)
                try:
                    return session.execute_read(_run)
                except AttributeError:
                    return session.read_transaction(_run)
        except Exception as ex:
            if not silent:
                print("Neo4j.Execute - Error: Could not execute the Cypher statement. Returning None.")
                print(ex)
            return None

    @staticmethod
    def Query(driver, cypher, parameters=None, database=None, silent=False):
        return Neo4j.Execute(driver=driver, cypher=cypher, parameters=parameters, write=False, database=database, silent=silent)

    @staticmethod
    def BatchExecute(driver, cypher, data, batchSize=1000, database=None, silent=False):
        if not isinstance(data, list):
            if not silent:
                print("Neo4j.BatchExecute - Error: The input data is not a valid list. Returning False.")
            return False
        if len(data) < 1:
            return True
        try:
            batchSize = max(1, int(batchSize))
        except Exception:
            batchSize = 1000
        for i in range(0, len(data), batchSize):
            batch = data[i:i + batchSize]
            result = Neo4j.Execute(driver, cypher, parameters={"rows": batch}, write=True, database=database, silent=silent)
            if result is None:
                return False
        return True

    # -------------------------------------------------------------------------
    # Schema and database management
    # -------------------------------------------------------------------------

    @staticmethod
    def EnsureSchema(manager, database=None, silent: bool = False) -> bool:
        """
        Ensures the canonical TopologicPy graph-corpus schema exists in Neo4j.

        Parameters
        ----------
        manager : neo4j.Driver
            The Neo4j driver.
        database : str , optional
            The database name. Default is None.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        if not Neo4j._is_driver(manager):
            if not silent:
                print("Neo4j.EnsureSchema - Error: The input manager is not a valid Neo4j driver. Returning False.")
            return False
        try:
            statements = [
                "CREATE CONSTRAINT graph_id_unique IF NOT EXISTS FOR (g:Graph) REQUIRE g.id IS UNIQUE",
                "CREATE CONSTRAINT vertex_graph_id_id_unique IF NOT EXISTS FOR (v:Vertex) REQUIRE (v.graph_id, v.id) IS UNIQUE",
                "CREATE INDEX vertex_graph_id_index IF NOT EXISTS FOR (v:Vertex) ON (v.graph_id)",
                "CREATE INDEX vertex_label_index IF NOT EXISTS FOR (v:Vertex) ON (v.label)",
                "CREATE INDEX edge_graph_id_index IF NOT EXISTS FOR ()-[r:Edge]-() ON (r.graph_id)",
            ]
            for stmt in statements:
                Neo4j.Execute(manager, stmt, write=True, database=database, silent=True)
            return True
        except Exception as ex:
            if not silent:
                print(f"Neo4j.EnsureSchema - Error: {ex}. Returning False.")
            return False

    @staticmethod
    def Reset(driver, database=None, silent=False):
        return Neo4j.EmptyDatabase(driver, dropSchema=True, recreateSchema=False, database=database, silent=silent)

    @staticmethod
    def EmptyDatabase(manager, dropSchema: bool = False, recreateSchema: bool = True, database=None, silent: bool = False) -> bool:
        """
        Empties the Neo4j database. If dropSchema is True, known constraints and
        indexes are also dropped when possible.
        """
        if not Neo4j._is_driver(manager):
            if not silent:
                print("Neo4j.EmptyDatabase - Error: The input manager is not a valid Neo4j driver. Returning False.")
            return False
        try:
            result = Neo4j.Execute(manager, "MATCH (n) DETACH DELETE n", write=True, database=database, silent=silent)
            if result is None:
                return False
            if dropSchema:
                constraints = Neo4j.Query(manager, "SHOW CONSTRAINTS", database=database, silent=True) or []
                for rec in constraints:
                    row = _record_to_dict(rec)
                    name = row.get("name")
                    if name:
                        Neo4j.Execute(manager, f"DROP CONSTRAINT `{name}` IF EXISTS", write=True, database=database, silent=True)
                indexes = Neo4j.Query(manager, "SHOW INDEXES", database=database, silent=True) or []
                for rec in indexes:
                    row = _record_to_dict(rec)
                    name = row.get("name")
                    if name:
                        Neo4j.Execute(manager, f"DROP INDEX `{name}` IF EXISTS", write=True, database=database, silent=True)
            if recreateSchema:
                Neo4j.EnsureSchema(manager, database=database, silent=silent)
            return True
        except Exception as ex:
            if not silent:
                print(f"Neo4j.EmptyDatabase - Error: {ex}. Returning False.")
            return False

    @staticmethod
    def CreateIndex(driver, label, property, indexName=None, ifNotExists=True, database=None, silent=False):
        label = Neo4j._sanitize_identifier(label, default="Node")
        property = Neo4j._sanitize_identifier(property, default="id")
        if indexName is None:
            indexName = f"idx_{label}_{property}"
        indexName = Neo4j._sanitize_identifier(indexName)
        ifClause = " IF NOT EXISTS" if ifNotExists else ""
        cypher = f"CREATE INDEX {indexName}{ifClause} FOR (n:{label}) ON (n.{property})"
        result = Neo4j.Execute(driver, cypher, write=True, database=database, silent=silent)
        return result is not None

    @staticmethod
    def CreateConstraint(driver, label, property, unique=True, constraintName=None, ifNotExists=True, database=None, silent=False):
        label = Neo4j._sanitize_identifier(label, default="Node")
        if isinstance(property, str):
            properties = [property]
        elif isinstance(property, (list, tuple)):
            properties = list(property)
        else:
            properties = ["id"]
        properties = [Neo4j._sanitize_identifier(p, default="id") for p in properties]
        if constraintName is None:
            suffix = "unique" if unique else "exists"
            constraintName = f"con_{label}_{'_'.join(properties)}_{suffix}"
        constraintName = Neo4j._sanitize_identifier(constraintName)
        ifClause = " IF NOT EXISTS" if ifNotExists else ""
        if unique:
            if len(properties) == 1:
                cypher = f"CREATE CONSTRAINT {constraintName}{ifClause} FOR (n:{label}) REQUIRE n.{properties[0]} IS UNIQUE"
            else:
                tuple_expr = "(" + ", ".join([f"n.{p}" for p in properties]) + ")"
                cypher = f"CREATE CONSTRAINT {constraintName}{ifClause} FOR (n:{label}) REQUIRE {tuple_expr} IS UNIQUE"
        else:
            if len(properties) > 1:
                if not silent:
                    print("Neo4j.CreateConstraint - Warning: Existence constraints do not support composite properties.")
                return False
            cypher = f"CREATE CONSTRAINT {constraintName}{ifClause} FOR (n:{label}) REQUIRE n.{properties[0]} IS NOT NULL"
        result = Neo4j.Execute(driver, cypher, write=True, database=database, silent=silent)
        return result is not None

    @staticmethod
    def Schema(driver, database=None, silent=False):
        indexes = Neo4j.Query(driver, "SHOW INDEXES", database=database, silent=silent)
        constraints = Neo4j.Query(driver, "SHOW CONSTRAINTS", database=database, silent=silent)
        if indexes is None or constraints is None:
            return None
        return {"indexes": _records_to_dicts(indexes), "constraints": _records_to_dicts(constraints)}

    # -------------------------------------------------------------------------
    # General database inspection
    # -------------------------------------------------------------------------

    @staticmethod
    def CountNodes(driver, label=None, database=None, silent=False):
        if label:
            label = Neo4j._sanitize_identifier(label, default="Node")
            cypher = f"MATCH (n:{label}) RETURN count(n) AS count"
        else:
            cypher = "MATCH (n) RETURN count(n) AS count"
        result = Neo4j.Query(driver, cypher, database=database, silent=silent)
        rows = _records_to_dicts(result)
        return rows[0].get("count") if rows else None

    @staticmethod
    def CountRelationships(driver, relationshipType=None, database=None, silent=False):
        if relationshipType:
            relationshipType = Neo4j._sanitize_identifier(relationshipType, default="CONNECTED_TO")
            cypher = f"MATCH ()-[r:{relationshipType}]->() RETURN count(r) AS count"
        else:
            cypher = "MATCH ()-[r]->() RETURN count(r) AS count"
        result = Neo4j.Query(driver, cypher, database=database, silent=silent)
        rows = _records_to_dicts(result)
        return rows[0].get("count") if rows else None

    @staticmethod
    def Labels(driver, database=None, silent=False):
        result = Neo4j.Query(driver, "CALL db.labels()", database=database, silent=silent)
        rows = _records_to_dicts(result)
        labels = []
        for row in rows:
            labels.append(row.get("label") if "label" in row else next(iter(row.values()), None))
        return [x for x in labels if x is not None]

    @staticmethod
    def RelationshipTypes(driver, database=None, silent=False):
        result = Neo4j.Query(driver, "CALL db.relationshipTypes()", database=database, silent=silent)
        rows = _records_to_dicts(result)
        types = []
        for row in rows:
            types.append(row.get("relationshipType") if "relationshipType" in row else next(iter(row.values()), None))
        return [x for x in types if x is not None]

    @staticmethod
    def Info(driver, database=None, silent=False):
        return {
            "nodeCount": Neo4j.CountNodes(driver, database=database, silent=silent),
            "relationshipCount": Neo4j.CountRelationships(driver, database=database, silent=silent),
            "labels": Neo4j.Labels(driver, database=database, silent=silent),
            "relationshipTypes": Neo4j.RelationshipTypes(driver, database=database, silent=silent),
        }

    @staticmethod
    def MatchNodes(driver, label=None, properties=None, database=None, silent=False):
        properties = properties or {}
        clauses = []
        parameters = {}
        label_clause = ":" + Neo4j._sanitize_identifier(label, default="Node") if label else ""
        for i, (k, v) in enumerate(properties.items()):
            pk = Neo4j._sanitize_identifier(k, default=f"p{i}")
            clauses.append(f"n.{pk} = $p{i}")
            parameters[f"p{i}"] = v
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        result = Neo4j.Query(driver, f"MATCH (n{label_clause}){where} RETURN n", parameters=parameters, database=database, silent=silent)
        rows = _records_to_dicts(result)
        out = []
        for row in rows:
            node = row.get("n")
            if node is not None:
                out.append(Neo4j._node_properties(node))
        return out

    @staticmethod
    def DeleteNodes(driver, label=None, properties=None, database=None, silent=False):
        properties = properties or {}
        clauses = []
        parameters = {}
        label_clause = ":" + Neo4j._sanitize_identifier(label, default="Node") if label else ""
        for i, (k, v) in enumerate(properties.items()):
            pk = Neo4j._sanitize_identifier(k, default=f"p{i}")
            clauses.append(f"n.{pk} = $p{i}")
            parameters[f"p{i}"] = v
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        result = Neo4j.Execute(driver, f"MATCH (n{label_clause}){where} DETACH DELETE n", parameters=parameters, write=True, database=database, silent=silent)
        return result is not None

    @staticmethod
    def DeleteRelationships(driver, relationshipType=None, database=None, silent=False):
        rel_clause = ":" + Neo4j._sanitize_identifier(relationshipType, default="CONNECTED_TO") if relationshipType else ""
        result = Neo4j.Execute(driver, f"MATCH ()-[r{rel_clause}]->() DELETE r", write=True, database=database, silent=silent)
        return result is not None

    @staticmethod
    def Validate(driver, idKey="topologic_id", database=None, silent=False):
        idKey = Neo4j._sanitize_identifier(idKey, default="topologic_id")
        duplicate = Neo4j.Query(driver, f"MATCH (n) WHERE n.{idKey} IS NOT NULL WITH n.{idKey} AS id, count(n) AS c WHERE c > 1 RETURN count(id) AS count", database=database, silent=silent)
        missing = Neo4j.Query(driver, f"MATCH (n) WHERE n.{idKey} IS NULL RETURN count(n) AS count", database=database, silent=silent)
        orphan = Neo4j.Query(driver, "MATCH (n) WHERE NOT (n)--() RETURN count(n) AS count", database=database, silent=silent)
        duplicate_rows = _records_to_dicts(duplicate)
        missing_rows = _records_to_dicts(missing)
        orphan_rows = _records_to_dicts(orphan)
        if not duplicate_rows or not missing_rows or not orphan_rows:
            return None
        return {
            "duplicateNodeIds": duplicate_rows[0].get("count", 0),
            "missingNodeIds": missing_rows[0].get("count", 0),
            "orphanNodes": orphan_rows[0].get("count", 0),
        }

    @staticmethod
    def ToDataFrame(driver, cypher, parameters=None, database=None, silent=False):
        try:
            import pandas as pd
        except Exception:
            if not silent:
                print("Neo4j.ToDataFrame - Error: Could not import pandas. Returning None.")
            return None
        records = Neo4j.Query(driver, cypher, parameters=parameters, database=database, silent=silent)
        if records is None:
            return None
        rows = []
        for row in _records_to_dicts(records):
            clean = {}
            for key, value in row.items():
                if hasattr(value, "items") and hasattr(value, "element_id"):
                    try:
                        value = dict(value.items())
                    except Exception:
                        pass
                clean[key] = value
            rows.append(clean)
        return pd.DataFrame(rows)

    # -------------------------------------------------------------------------
    # Canonical graph persistence API compatible with Kuzu.py
    # -------------------------------------------------------------------------

    @staticmethod
    def UpsertGraph(manager,
                    graph,
                    database: str = None,
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
                    silent: bool = False) -> str:
        """
        Upserts a TopologicPy graph into Neo4j using the canonical schema.

        Parameters
        ----------
        manager
            The graph database manager/driver.
        graph : topologic_core.Graph
            The input TopologicPy graph.
        database : str , optional
            The Neo4j database name. Default is None.
        graphIDKey : str , optional
            The graph dictionary key under which the graph id is stored. Default is "graph_id".
        vertexIDKey : str , optional
            The vertex dictionary key under which the vertex id is stored. Default is "id".
        vertexLabelKey : str , optional
            The vertex dictionary key under which the vertex label is stored. Default is "label".
        defaultVertexLabel : str , optional
            The default vertex label if no vertex label is found. Default is "Node".
        vertexCategoryKey : str , optional
            The vertex dictionary key used for category metadata. Default is "category".
        defaultVertexCategory : any , optional
            The default vertex category if none is found. Default is "Node".
        edgeLabelKey : str , optional
            The edge dictionary key under which the edge label/type is stored. Default is "label".
        defaultEdgeLabel : str , optional
            The default edge label if no edge label is found. Default is "CONNECTED_TO".
        edgeCategoryKey : str , optional
            The edge dictionary key used for category metadata. Default is "category".
        defaultEdgeCategory : any , optional
            The default edge category if none is found. Default is "Edge".
        bidirectional : bool , optional
            If set to True, reverse edges are also written. Default is True.
        overwrite : bool , optional
            If set to True, an existing graph with the same id is deleted before import. Default is False.
        mantissa : int , optional
            The number of decimal places to use when extracting mesh data. Default is 6.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        str
            The graph id used, or None on error.
        """

        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology
        from topologicpy.Graph import Graph
        import re

        def _safe_string(value, default: str = "") -> str:
            if value is None:
                return default
            value = str(value).strip()
            return value if value != "" else default

        def _safe_id(value, fallback: str) -> str:
            """
            Converts an arbitrary value into a Neo4j-safe local identifier.
            The returned id is graph-local, not globally unique.
            """
            value = _safe_string(value, fallback)

            value = re.sub(r"[^A-Za-z0-9_\-]", "_", value)
            value = re.sub(r"_+", "_", value)
            value = value.strip("_- ")

            if value == "":
                value = fallback

            return value

        try:
            if not Neo4j.EnsureSchema(manager, database=database, silent=silent):
                return None

            graph_dict = Topology.Dictionary(graph)
            gid = _value_from_dict(graph_dict, graphIDKey, None) if graphIDKey is not None else None

            if gid is None or str(gid).strip() == "":
                gid = Topology.UUID(graph)

            gid = str(gid).strip()

            # ------------------------------------------------------------
            # Check whether this graph already exists.
            # ------------------------------------------------------------
            exists_result = Neo4j.Query(
                manager,
                """
                MATCH (g:Graph {id:$id})
                RETURN count(g) > 0 AS exists
                """,
                parameters={"id": gid},
                database=database,
                silent=True,
            )

            exists = False
            if exists_result:
                try:
                    exists = bool(exists_result[0]["exists"])
                except Exception:
                    exists = False

            if exists and not overwrite:
                if not silent:
                    print("Neo4j.UpsertGraph - Error: The graph already exists and overwrite is False. Returning None.")
                return None

            if exists and overwrite:
                Neo4j.DeleteGraph(manager, gid, database=database, silent=True)

            # ------------------------------------------------------------
            # Extract graph data.
            # ------------------------------------------------------------
            g_props = _python_dictionary(graph_dict)
            g_label = str(g_props.get("label", ""))

            mesh_data = Graph.MeshData(graph, mantissa=mantissa)
            verts = mesh_data.get("vertices", []) or []
            v_props = mesh_data.get("vertexDictionaries", []) or []
            edges = mesh_data.get("edges", []) or []
            e_props = mesh_data.get("edgeDictionaries", []) or []

            edge_count = len(edges) * (2 if bidirectional else 1)

            # ------------------------------------------------------------
            # Store graph card.
            # ------------------------------------------------------------
            result = Neo4j.Execute(
                manager,
                """
                CREATE (g:Graph {
                    id:$id,
                    label:$label,
                    num_nodes:$num_nodes,
                    num_edges:$num_edges,
                    props:$props
                })
                RETURN g.id AS id
                """,
                parameters={
                    "id": gid,
                    "label": g_label,
                    "num_nodes": int(len(verts)),
                    "num_edges": int(edge_count),
                    "props": _json_dumps(g_props),
                },
                write=True,
                database=database,
                silent=silent,
            )

            if result is None:
                return None

            # ------------------------------------------------------------
            # Build canonical vertices.
            #
            # Important:
            # Vertex.id remains local to the graph.
            # Vertex.uid is globally unique and should be used for robust
            # database matching.
            # ------------------------------------------------------------
            vertex_ids = []
            vertex_uids = []
            vertex_rows = []
            used_vertex_ids = set()

            for i, xyz in enumerate(verts):
                props = dict(v_props[i] or {}) if i < len(v_props) else {}

                try:
                    x, y, z = xyz
                except Exception:
                    if not silent:
                        print(f"Neo4j.UpsertGraph - Warning: Invalid vertex coordinates at index {i}. Skipping vertex.")
                    vertex_ids.append(None)
                    vertex_uids.append(None)
                    continue

                if vertexIDKey is not None:
                    raw_vid = props.get(vertexIDKey, None)
                else:
                    raw_vid = None

                fallback_vid = f"n{i + 1}"
                vid = _safe_id(raw_vid, fallback_vid)

                # Ensure uniqueness within this graph import.
                base_vid = vid
                counter = 2
                while vid in used_vertex_ids:
                    vid = f"{base_vid}_{counter}"
                    counter += 1

                used_vertex_ids.add(vid)

                # Globally unique database id.
                uid = f"{gid}:{vid}"

                if vertexIDKey is not None:
                    props[vertexIDKey] = vid

                props["_db_id"] = uid
                props["_local_id"] = vid
                props["graph_id"] = gid

                label = props.get(vertexLabelKey, None) if vertexLabelKey is not None else None
                label = _safe_string(label, defaultVertexLabel if defaultVertexLabel is not None else str(i))

                if vertexLabelKey is not None:
                    props[vertexLabelKey] = label

                if vertexCategoryKey is not None:
                    category = props.get(vertexCategoryKey, defaultVertexCategory)
                    if category is not None:
                        props[vertexCategoryKey] = category

                vertex_ids.append(vid)
                vertex_uids.append(uid)

                vertex_rows.append({
                    "uid": uid,
                    "id": vid,
                    "graph_id": gid,
                    "label": label,
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                    "props": _json_dumps(props),
                })

            if vertex_rows:
                ok = Neo4j.BatchExecute(
                    manager,
                    """
                    UNWIND $rows AS row
                    CREATE (:Vertex {
                        uid: row.uid,
                        id: row.id,
                        graph_id: row.graph_id,
                        label: row.label,
                        x: row.x,
                        y: row.y,
                        z: row.z,
                        props: row.props
                    })
                    """,
                    vertex_rows,
                    batchSize=1000,
                    database=database,
                    silent=silent,
                )

                if not ok:
                    return None

            # ------------------------------------------------------------
            # Build canonical edges.
            #
            # Critical fix:
            # Edges are matched by Vertex.uid, not by Vertex.id alone.
            # This prevents cross-graph contamination when multiple graphs
            # reuse local vertex ids such as n1, n2, door, 0, 1, etc.
            # ------------------------------------------------------------
            edge_rows = []

            for i, edge_indices in enumerate(edges):
                try:
                    a_index = int(edge_indices[0])
                    b_index = int(edge_indices[1])
                except Exception:
                    continue

                if (
                    a_index < 0 or
                    b_index < 0 or
                    a_index >= len(vertex_uids) or
                    b_index >= len(vertex_uids)
                ):
                    continue

                a_uid = vertex_uids[a_index]
                b_uid = vertex_uids[b_index]

                if a_uid is None or b_uid is None:
                    continue

                props = dict(e_props[i] or {}) if i < len(e_props) else {}

                label = props.get(edgeLabelKey, None) if edgeLabelKey is not None else None

                if label is None or str(label).strip() == "":
                    label = props.get("type", None)

                if label is None or str(label).strip() == "":
                    label = defaultEdgeLabel

                label = _safe_string(label, defaultEdgeLabel)

                if edgeLabelKey is not None:
                    props[edgeLabelKey] = label

                if edgeCategoryKey is not None:
                    category = props.get(edgeCategoryKey, defaultEdgeCategory)
                    if category is not None:
                        props[edgeCategoryKey] = category

                props["graph_id"] = gid

                edge_rows.append({
                    "a_uid": a_uid,
                    "b_uid": b_uid,
                    "graph_id": gid,
                    "label": label,
                    "props": _json_dumps(props),
                })

                if bidirectional and a_uid != b_uid:
                    edge_rows.append({
                        "a_uid": b_uid,
                        "b_uid": a_uid,
                        "graph_id": gid,
                        "label": label,
                        "props": _json_dumps(props),
                    })

            if edge_rows:
                ok = Neo4j.BatchExecute(
                    manager,
                    """
                    UNWIND $rows AS row
                    MATCH (a:Vertex {uid: row.a_uid})
                    MATCH (b:Vertex {uid: row.b_uid})
                    CREATE (a)-[:Edge {
                        graph_id: row.graph_id,
                        label: row.label,
                        props: row.props
                    }]->(b)
                    """,
                    edge_rows,
                    batchSize=1000,
                    database=database,
                    silent=silent,
                )

                if not ok:
                    return None

            return gid

        except Exception as ex:
            if not silent:
                print(f"Neo4j.UpsertGraph - Error: {ex}. Returning None.")
            return None

    @staticmethod
    def GraphByID(manager, graphID: str, database=None, silent: bool = False):
        """
        Constructs a TopologicPy graph from Neo4j using the canonical graph id.
        """
        try:
            from topologicpy.Graph import Graph
            from topologicpy.Dictionary import Dictionary
            from topologicpy.Vertex import Vertex
            from topologicpy.Edge import Edge
            from topologicpy.Topology import Topology

            graphID = str(graphID)
            rows_g = _records_to_dicts(Neo4j.Query(
                manager,
                """
                MATCH (g:Graph {id:$id})
                RETURN g.id AS id, g.label AS label, g.num_nodes AS num_nodes, g.num_edges AS num_edges, g.props AS props
                """,
                parameters={"id": graphID},
                database=database,
                silent=silent,
            ))
            if not rows_g:
                return None

            g_row = rows_g[0]
            g_props = dict(_json_loads(g_row.get("props"), {}))
            if "label" not in g_props and g_row.get("label") is not None:
                g_props["label"] = g_row.get("label")
            g_dict = Dictionary.ByPythonDictionary(g_props)

            rows_v = _records_to_dicts(Neo4j.Query(
                manager,
                """
                MATCH (v:Vertex)
                WHERE v.graph_id = $gid
                RETURN v.id AS id, v.label AS label, v.x AS x, v.y AS y, v.z AS z, v.props AS props
                ORDER BY v.id
                """,
                parameters={"gid": graphID},
                database=database,
                silent=silent,
            ))

            id_to_vertex = {}
            vertices = []
            for row in rows_v:
                x = row.get("x") if row.get("x") is not None else 0.0
                y = row.get("y") if row.get("y") is not None else 0.0
                z = row.get("z") if row.get("z") is not None else 0.0
                v = Vertex.ByCoordinates(float(x), float(y), float(z))
                props = dict(_json_loads(row.get("props"), {}))
                if "id" not in props:
                    props["id"] = row.get("id")
                if "label" not in props:
                    props["label"] = row.get("label") or ""
                d = Dictionary.ByPythonDictionary(props)
                v = Topology.SetDictionary(v, d)
                id_to_vertex[row.get("id")] = v
                vertices.append(v)

            rows_e = _records_to_dicts(Neo4j.Query(
                manager,
                """
                MATCH (a:Vertex)-[r:Edge]->(b:Vertex)
                WHERE a.graph_id = $gid AND b.graph_id = $gid AND r.graph_id = $gid
                RETURN a.id AS a_id, b.id AS b_id, r.label AS label, r.props AS props
                """,
                parameters={"gid": graphID},
                database=database,
                silent=silent,
            ))

            edges = []
            for row in rows_e:
                va = id_to_vertex.get(row.get("a_id"))
                vb = id_to_vertex.get(row.get("b_id"))
                if va is None or vb is None:
                    continue
                e = Edge.ByStartVertexEndVertex(va, vb)
                props = dict(_json_loads(row.get("props"), {}))
                if "label" not in props:
                    props["label"] = row.get("label") or "connect"
                d = Dictionary.ByPythonDictionary(props)
                e = Topology.SetDictionary(e, d)
                edges.append(e)

            if not vertices:
                return None
            g = Graph.ByVerticesEdges(vertices, edges)
            g = Topology.SetDictionary(g, g_dict)
            return g
        except Exception as ex:
            if not silent:
                print(f"Neo4j.GraphByID - Error: {ex}. Returning None.")
            return None

    @staticmethod
    def GraphsByQuery(manager, query: str, parameters: dict = None, database=None, silent: bool = False):
        """
        Executes a Cypher query and returns a list of TopologicPy graphs constructed
        directly from the returned nodes, relationships, and paths.

        The method supports query results that return:
        - Neo4j nodes
        - Neo4j relationships
        - Neo4j paths
        - lists containing nodes, relationships, or paths
        - dictionaries containing nodes, relationships, or paths

        If the query returns only nodes, the result is a graph containing isolated
        vertices. If the query returns relationships or paths, the corresponding
        edges are included.

        Parameters
        ----------
        manager : neo4j.Driver
            The Neo4j driver.
        query : str
            The Cypher query.
        parameters : dict , optional
            Query parameters. Default is None.
        database : str , optional
            Neo4j database name. Default is None.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            A list containing one TopologicPy graph constructed from the query result,
            or an empty list if no graph elements are returned.
        """
        try:
            from topologicpy.Graph import Graph
            from topologicpy.Vertex import Vertex
            from topologicpy.Edge import Edge
            from topologicpy.Topology import Topology
            from topologicpy.Dictionary import Dictionary

            records = Neo4j.Query(
                manager,
                query,
                parameters=parameters or {},
                database=database,
                silent=silent,
            )

            if records is None:
                return None

            node_by_element_id = {}
            rels = []

            def _node_element_id(node):
                try:
                    return node.element_id
                except Exception:
                    try:
                        return str(node.id)
                    except Exception:
                        return str(id(node))

            def _rel_element_id(rel):
                try:
                    return rel.element_id
                except Exception:
                    try:
                        return str(rel.id)
                    except Exception:
                        return str(id(rel))

            def _node_props(node):
                try:
                    return dict(node.items())
                except Exception:
                    return {}

            def _rel_props(rel):
                try:
                    return dict(rel.items())
                except Exception:
                    return {}

            def _is_node(value):
                return hasattr(value, "labels") and hasattr(value, "items")

            def _is_relationship(value):
                return (
                    hasattr(value, "start_node")
                    and hasattr(value, "end_node")
                    and hasattr(value, "type")
                )

            def _is_path(value):
                return hasattr(value, "nodes") and hasattr(value, "relationships")

            def _add_node(node):
                if not _is_node(node):
                    return

                eid = _node_element_id(node)
                if eid in node_by_element_id:
                    return

                props = _node_props(node)

                x = props.get("x", 0.0)
                y = props.get("y", 0.0)
                z = props.get("z", 0.0)

                try:
                    x = float(x)
                except Exception:
                    x = 0.0
                try:
                    y = float(y)
                except Exception:
                    y = 0.0
                try:
                    z = float(z)
                except Exception:
                    z = 0.0

                v = Vertex.ByCoordinates(x, y, z)

                # Preserve useful Neo4j metadata.
                props = dict(props)
                props["_neo4j_element_id"] = eid
                try:
                    props["_neo4j_labels"] = list(node.labels)
                except Exception:
                    pass

                d = Dictionary.ByPythonDictionary(props)
                v = Topology.SetDictionary(v, d)

                node_by_element_id[eid] = v

            def _add_relationship(rel):
                if not _is_relationship(rel):
                    return

                try:
                    start_node = rel.start_node
                    end_node = rel.end_node
                except Exception:
                    return

                _add_node(start_node)
                _add_node(end_node)

                sid = _node_element_id(start_node)
                eid = _node_element_id(end_node)

                if sid not in node_by_element_id or eid not in node_by_element_id:
                    return

                rels.append(rel)

            def _consume(value):
                if value is None:
                    return

                if _is_path(value):
                    try:
                        for n in value.nodes:
                            _add_node(n)
                    except Exception:
                        pass
                    try:
                        for r in value.relationships:
                            _add_relationship(r)
                    except Exception:
                        pass
                    return

                if _is_relationship(value):
                    _add_relationship(value)
                    return

                if _is_node(value):
                    _add_node(value)
                    return

                if isinstance(value, dict):
                    for v in value.values():
                        _consume(v)
                    return

                if isinstance(value, (list, tuple, set)):
                    for v in value:
                        _consume(v)
                    return

            for record in records:
                try:
                    for key in record.keys():
                        _consume(record[key])
                except Exception:
                    try:
                        for value in record:
                            _consume(value)
                    except Exception:
                        pass

            vertices = list(node_by_element_id.values())
            edges = []
            used_rels = set()

            for rel in rels:
                rid = _rel_element_id(rel)
                if rid in used_rels:
                    continue
                used_rels.add(rid)

                try:
                    sv = node_by_element_id.get(_node_element_id(rel.start_node))
                    ev = node_by_element_id.get(_node_element_id(rel.end_node))
                except Exception:
                    sv = None
                    ev = None

                if sv is None or ev is None:
                    continue

                e = Edge.ByStartVertexEndVertex(sv, ev)

                props = _rel_props(rel)
                props = dict(props)
                props["_neo4j_element_id"] = rid
                props["type"] = getattr(rel, "type", None)

                if "label" not in props and props.get("type") is not None:
                    props["label"] = props.get("type")

                d = Dictionary.ByPythonDictionary(props)
                e = Topology.SetDictionary(e, d)

                edges.append(e)

            if not vertices:
                return []

            g = Graph.ByVerticesEdges(vertices, edges)
            return [g]

        except Exception as ex:
            if not silent:
                print(f"Neo4j.GraphsByQuery - Error: {ex}. Returning None.")
            return None
    
    @staticmethod
    def DeleteGraph(manager, graphID: str, database=None, silent: bool = False) -> bool:
        try:
            graphID = str(graphID)
            Neo4j.Execute(
                manager,
                """
                MATCH (a:Vertex)-[r:Edge]->(b:Vertex)
                WHERE a.graph_id = $gid AND b.graph_id = $gid AND r.graph_id = $gid
                DELETE r
                """,
                parameters={"gid": graphID},
                write=True,
                database=database,
                silent=silent,
            )
            Neo4j.Execute(manager, "MATCH (v:Vertex) WHERE v.graph_id = $gid DELETE v", parameters={"gid": graphID}, write=True, database=database, silent=silent)
            Neo4j.Execute(manager, "MATCH (g:Graph) WHERE g.id = $gid DELETE g", parameters={"gid": graphID}, write=True, database=database, silent=silent)
            return True
        except Exception as ex:
            if not silent:
                print(f"Neo4j.DeleteGraph - Error: {ex}. Returning False.")
            return False

    @staticmethod
    def ListGraphs(manager, where: dict = None, limit: int = 100, offset: int = 0, database=None, silent: bool = False) -> list:
        try:
            where = where or {}
            conds = []
            parameters = {}
            if where.get("id"):
                conds.append("g.id = $id")
                parameters["id"] = str(where["id"])
            if where.get("label"):
                conds.append("g.label CONTAINS $label_sub")
                parameters["label_sub"] = str(where["label"])
            if where.get("props_contains"):
                conds.append("g.props CONTAINS $props_sub")
                parameters["props_sub"] = str(where["props_contains"])
            if where.get("props_equals"):
                conds.append("g.props = $props_equals")
                parameters["props_equals"] = str(where["props_equals"])
            if where.get("min_nodes") is not None:
                conds.append("g.num_nodes >= $min_nodes")
                parameters["min_nodes"] = int(where["min_nodes"])
            if where.get("max_nodes") is not None:
                conds.append("g.num_nodes <= $max_nodes")
                parameters["max_nodes"] = int(where["max_nodes"])
            if where.get("min_edges") is not None:
                conds.append("g.num_edges >= $min_edges")
                parameters["min_edges"] = int(where["min_edges"])
            if where.get("max_edges") is not None:
                conds.append("g.num_edges <= $max_edges")
                parameters["max_edges"] = int(where["max_edges"])
            parameters["offset"] = max(0, int(offset or 0))
            parameters["limit"] = max(0, int(limit or 100))
            where_clause = "WHERE " + " AND ".join(conds) if conds else ""
            rows = _records_to_dicts(Neo4j.Query(
                manager,
                f"""
                MATCH (g:Graph)
                {where_clause}
                RETURN g.id AS id, g.label AS label, g.num_nodes AS num_nodes, g.num_edges AS num_edges, g.props AS props
                ORDER BY g.id
                SKIP $offset LIMIT $limit
                """,
                parameters=parameters,
                database=database,
                silent=silent,
            ))
            return rows
        except Exception as ex:
            if not silent:
                print(f"Neo4j.ListGraphs - Error: {ex}. Returning None.")
            return None

    @staticmethod
    def ByCSVPath(
        manager,
        path,
        graphIDHeader="graph_id", graphLabelHeader="label", graphFeaturesHeader="feat", graphFeaturesKeys=None,
        edgeSRCHeader="src_id", edgeDSTHeader="dst_id", edgeLabelHeader="label",
        edgeTrainMaskHeader="train_mask", edgeValidateMaskHeader="val_mask", edgeTestMaskHeader="test_mask",
        edgeFeaturesHeader="feat", edgeFeaturesKeys=None,
        nodeIDHeader="node_id", nodeLabelHeader="label",
        nodeTrainMaskHeader="train_mask", nodeValidateMaskHeader="val_mask", nodeTestMaskHeader="test_mask",
        nodeFeaturesHeader="feat", nodeXHeader="X", nodeYHeader="Y", nodeZHeader="Z",
        nodeFeaturesKeys=None,
        tolerance=0.0001, database=None, silent=False):
        """
        Reads CSV graph data using Graph.ByCSVPath and upserts all returned graphs into Neo4j.

        The signature mirrors Graph.ByCSVPath, with an extra optional database parameter.
        """
        try:
            from topologicpy.Graph import Graph

            Neo4j.EnsureSchema(manager, database=database, silent=silent)
            graphs = Graph.ByCSVPath(
                path=path,
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
                tolerance=tolerance, silent=silent)
            if graphs is None:
                if not silent:
                    print("Neo4j.ByCSVPath - Error: Graph.ByCSVPath returned None. Returning None.")
                return None
            if not isinstance(graphs, list):
                graphs = [graphs]
            graph_ids = []
            for graph in graphs:
                gid = Neo4j.UpsertGraph(
                    manager,
                    graph,
                    graphIDKey=graphIDHeader,
                    vertexIDKey=nodeIDHeader,
                    vertexLabelKey=nodeLabelHeader,
                    database=database,
                    silent=silent)
                if gid is not None:
                    graph_ids.append(gid)
            return {"graphs_upserted": len(graph_ids), "graph_ids": graph_ids}
        except Exception as ex:
            if not silent:
                print(f"Neo4j.ByCSVPath - Error: {ex}. Returning None.")
            return None

    # -------------------------------------------------------------------------
    # Corpus analytics for graph generation / GraphRAG workflows
    # -------------------------------------------------------------------------

    @staticmethod
    def FetchAllPairs(manager, undirected: bool = True, database=None, silent: bool = False) -> list:
        try:
            rows = _records_to_dicts(Neo4j.Query(
                manager,
                """
                MATCH (a:Vertex)-[r:Edge]->(b:Vertex)
                RETURN a.label AS a_label, b.label AS b_label, count(*) AS count
                ORDER BY count DESC
                """,
                database=database,
                silent=silent,
            ))
            if not undirected:
                return rows
            counts = {}
            for row in rows:
                a = _normalize_label(row.get("a_label"))
                b = _normalize_label(row.get("b_label"))
                if a == "" or b == "":
                    continue
                key = tuple(sorted([a, b]))
                counts[key] = counts.get(key, 0) + int(row.get("count") or 0)
            out = []
            for (a, b), count in sorted(counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1])):
                out.append({"a_label": a, "b_label": b, "pair": [a, b], "count": count})
            return out
        except Exception as ex:
            if not silent:
                print(f"Neo4j.FetchAllPairs - Error: {ex}. Returning None.")
            return None

    @staticmethod
    def MaxNeighborsForLabel(manager, label: str, database=None, silent: bool = False) -> int:
        try:
            label = _normalize_label(label)
            if label == "":
                return 0

            rows = _records_to_dicts(Neo4j.Query(
                manager,
                """
                MATCH (v:Vertex)
                WHERE v.label = $label
                OPTIONAL MATCH (v)-[e:Edge]-(o:Vertex)
                WHERE v.graph_id = o.graph_id
                AND e.graph_id = v.graph_id
                WITH v, count(DISTINCT o) AS degree
                RETURN coalesce(max(degree), 0) AS max_degree
                """,
                parameters={"label": label},
                database=database,
                silent=silent,
            ))

            if not rows:
                return 0

            return int(rows[0].get("max_degree") or 0)

        except Exception as ex:
            if not silent:
                print(f"Neo4j.MaxNeighborsForLabel - Error: {ex}. Returning 0.")
            return 0

    @staticmethod
    def CandidateCountsForLabels(manager, labels, excludeLabels=None, limit: int = 50, database=None, silent: bool = False) -> list:
        try:
            if labels is None:
                labels = []
            if isinstance(labels, str):
                labels = [labels]

            labels = [_normalize_label(x) for x in labels if _normalize_label(x) != ""]
            labels = _unique_preserve_order(labels)

            exclude = [_normalize_label(x) for x in (excludeLabels or []) if _normalize_label(x) != ""]
            exclude = _unique_preserve_order(exclude)

            # Case-insensitive comparison lists.
            labels_lc = _unique_preserve_order([str(x).lower() for x in labels])
            exclude_lc = _unique_preserve_order([str(x).lower() for x in exclude])

            if not labels_lc:
                return []

            rows = _records_to_dicts(Neo4j.Query(
                manager,
                """
                MATCH (a:Vertex)-[:Edge]-(b:Vertex)
                WITH
                    toLower(coalesce(a.label, "")) AS a_label_lc,
                    toLower(coalesce(b.label, "")) AS b_label_lc
                WHERE a_label_lc IN $labels_lc
                AND NOT b_label_lc IN $exclude_lc
                AND b_label_lc <> ""
                RETURN b_label_lc AS label, count(*) AS count
                ORDER BY count DESC, label ASC
                LIMIT $limit
                """,
                parameters={
                    "labels_lc": labels_lc,
                    "exclude_lc": exclude_lc,
                    "limit": max(1, int(limit or 50)),
                },
                database=database,
                silent=silent,
            ))

            out = []
            exclude_set = set(exclude_lc)

            for row in rows:
                lbl = _normalize_label(row.get("label"))
                lbl_lc = str(lbl).lower()

                if lbl_lc == "" or lbl_lc in exclude_set:
                    continue

                out.append({
                    "label": lbl_lc,
                    "count": int(row.get("count") or 0),
                })

            return out

        except Exception as ex:
            if not silent:
                print(f"Neo4j.CandidateCountsForLabels - Error: {ex}. Returning None.")
            return None

    @staticmethod
    def FindBestExampleForLabel(manager, label: str, attachTo=None, database=None, silent: bool = False) -> dict:
        """
        Returns a representative corpus vertex for the input label.

        If attachTo is supplied, the method prefers an example adjacent to a
        vertex whose label equals attachTo. The returned dictionary contains
        id, graph_id, label, x, y, z, props, attach_label, and frequency where
        available.
        """
        try:
            label = _normalize_label(label)
            attach_label = _normalize_label(attachTo)
            if label == "":
                return None

            if attach_label != "":
                rows = _records_to_dicts(Neo4j.Query(
                    manager,
                    """
                    MATCH (a:Vertex)-[:Edge]-(b:Vertex)
                    WHERE b.label = $label AND a.label = $attach
                    WITH b, a, count(*) AS frequency
                    RETURN b.id AS id, b.graph_id AS graph_id, b.label AS label,
                           b.x AS x, b.y AS y, b.z AS z, b.props AS props,
                           a.label AS attach_label, frequency AS frequency
                    ORDER BY frequency DESC, b.id ASC
                    LIMIT 1
                    """,
                    parameters={"label": label, "attach": attach_label},
                    database=database,
                    silent=silent,
                ))
                if rows:
                    row = rows[0]
                    row["props"] = _json_loads(row.get("props"), {})
                    return row

            rows = _records_to_dicts(Neo4j.Query(
                manager,
                """
                MATCH (b:Vertex)
                WHERE b.label = $label
                OPTIONAL MATCH (b)-[:Edge]-(o:Vertex)
                WITH b, count(DISTINCT o) AS degree
                RETURN b.id AS id, b.graph_id AS graph_id, b.label AS label,
                       b.x AS x, b.y AS y, b.z AS z, b.props AS props,
                       null AS attach_label, degree AS frequency
                ORDER BY degree DESC, b.id ASC
                LIMIT 1
                """,
                parameters={"label": label},
                database=database,
                silent=silent,
            ))
            if not rows:
                return None
            row = rows[0]
            row["props"] = _json_loads(row.get("props"), {})
            return row
        except Exception as ex:
            if not silent:
                print(f"Neo4j.FindBestExampleForLabel - Error: {ex}. Returning None.")
            return None

    # -------------------------------------------------------------------------
    # Compatibility wrappers for older Neo4j.py workflows
    # -------------------------------------------------------------------------

    @staticmethod
    def ByGraph(driver,
                graph,
                graphID: str = None,
                graphIDKey: str = "graph_id",
                nodeLabelKey: str = "label",
                defaultNodeLabel: str = "Node",
                nodeCategoryKey: str = "category",
                defaultNodeCategory: str = None,
                relationshipTypeKey: str = "label",
                defaultRelationshipType: str = "CONNECTED_TO",
                relationshipCategoryKey: str = "category",
                defaultRelationshipCategory: str = None,
                bidirectional: bool = True,
                deleteAll: bool = False,
                createIndex: bool = True,
                createConstraint: bool = False,
                mantissa: int = 6,
                tolerance: float = 0.0001,
                database: str = None,
                overwrite: bool = True,
                silent: bool = False):
        """
        Writes the input Topologic graph to Neo4j.

        Compatibility wrapper: internally uses Neo4j.UpsertGraph with the
        canonical schema. The input driver is returned on success.
        """
        try:
            from topologicpy.Topology import Topology
            from topologicpy.Dictionary import Dictionary

            if deleteAll:
                Neo4j.EmptyDatabase(driver, dropSchema=False, recreateSchema=True, database=database, silent=silent)
            if graphID is not None and str(graphID).strip() != "":
                d = Topology.Dictionary(graph)
                d = Dictionary.SetValueAtKey(d, graphIDKey, str(graphID).strip())
                Topology.SetDictionary(graph, d)
            gid = Neo4j.UpsertGraph(driver, graph, graphIDKey=graphIDKey, vertexIDKey=None, vertexLabelKey=nodeLabelKey, mantissa=mantissa, database=database, silent=silent)
            return driver if gid is not None else None
        except Exception as ex:
            if not silent:
                print(f"Neo4j.ByGraph - Error: {ex}. Returning None.")
            return None

    @staticmethod
    def MergeGraph(driver,
                   graph,
                   nodeLabelKey="label",
                   defaultNodeLabel="Node",
                   nodeCategoryKey="category",
                   defaultNodeCategory=None,
                   relationshipTypeKey="label",
                   defaultRelationshipType="CONNECTED_TO",
                   relationshipCategoryKey="category",
                   defaultRelationshipCategory=None,
                   bidirectional=True,
                   createIndex=True,
                   createConstraint=False,
                   mantissa=6,
                   tolerance=0.0001,
                   database=None,
                   silent=False):
        """
        Compatibility wrapper. Uses canonical UpsertGraph and returns the driver.
        """
        gid = Neo4j.UpsertGraph(driver, graph, graphIDKey="graph_id", vertexIDKey=None, vertexLabelKey=nodeLabelKey, mantissa=mantissa, database=database, silent=silent)
        return driver if gid is not None else None



    @staticmethod
    def Neighborhood(driver,
                     nodeId,
                     depth: int = 1,
                     xMin: float = -0.5,
                     yMin: float = -0.5,
                     zMin: float = -0.5,
                     xMax: float = 0.5,
                     yMax: float = 0.5,
                     zMax: float = 0.5,
                     tolerance: float = 0.0001,
                     silent: bool = False):
        """
        Returns the neighborhood of the input Neo4j node as a Topologic graph.

        Parameters
        ----------
        driver : neo4j.Driver
            The input Neo4j driver.
        nodeId : str
            The Neo4j internal element id of the source node.
        depth : int , optional
            The neighborhood depth. Default is 1.
        xMin : float , optional
            The minimum random X coordinate to use when a node does not contain
            an ``x`` property. Default is -0.5.
        yMin : float , optional
            The minimum random Y coordinate to use when a node does not contain
            a ``y`` property. Default is -0.5.
        zMin : float , optional
            The minimum random Z coordinate to use when a node does not contain
            a ``z`` property. Default is -0.5.
        xMax : float , optional
            The maximum random X coordinate to use when a node does not contain
            an ``x`` property. Default is 0.5.
        yMax : float , optional
            The maximum random Y coordinate to use when a node does not contain
            a ``y`` property. Default is 0.5.
        zMax : float , optional
            The maximum random Z coordinate to use when a node does not contain
            a ``z`` property. Default is 0.5.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Graph
            The returned Topologic graph.
        """
        if driver is None or not hasattr(driver, "session"):
            if not silent:
                print("Neo4j.Neighborhood - Error: The input driver is not a valid Neo4j driver. Returning None.")
            return None

        if not isinstance(nodeId, str) or len(nodeId) < 1:
            if not silent:
                print("Neo4j.Neighborhood - Error: The input nodeId is not a valid string. Returning None.")
            return None

        if not isinstance(depth, int) or depth < 1:
            if not silent:
                print("Neo4j.Neighborhood - Error: The input depth is not a valid positive integer. Returning None.")
            return None

        cypher = f"""
        MATCH (n)
        WHERE elementId(n) = $nodeId
        OPTIONAL MATCH p=(n)-[*1..{depth}]-(m)
        RETURN n AS result
        UNION
        MATCH (n)
        WHERE elementId(n) = $nodeId
        OPTIONAL MATCH p=(n)-[*1..{depth}]-(m)
        WITH p
        WHERE p IS NOT NULL
        RETURN p AS result
        """

        return Neo4j.ToGraph(driver,
                             cypher=cypher,
                             parameters={"nodeId": nodeId},
                             xMin=xMin,
                             yMin=yMin,
                             zMin=zMin,
                             xMax=xMax,
                             yMax=yMax,
                             zMax=zMax,
                             tolerance=tolerance,
                             silent=silent)
    @staticmethod
    def ToGraph(driver,
                graphID: str = None,
                cypher: str = None,
                parameters: dict = None,
                xMin: float = -0.5,
                yMin: float = -0.5,
                zMin: float = -0.5,
                xMax: float = 0.5,
                yMax: float = 0.5,
                zMax: float = 0.5,
                database: str = None,
                tolerance: float = 0.0001,
                silent: bool = False):
        """
        Returns a Topologic graph from Neo4j.

        If graphID is supplied, the canonical graph with that id is returned.
        Otherwise, the method imports graph entities returned by the supplied
        Cypher query, or by a generic whole-database query when cypher is None.
        """
        if graphID is not None:
            return Neo4j.GraphByID(driver, graphID, database=database, silent=silent)

        # Generic fallback for arbitrary Neo4j graph queries.
        try:
            import random
            from topologicpy.Vertex import Vertex
            from topologicpy.Edge import Edge
            from topologicpy.Graph import Graph
            from topologicpy.Topology import Topology
            from topologicpy.Dictionary import Dictionary
            try:
                from neo4j.graph import Node, Relationship, Path
            except Exception:
                Node = Relationship = Path = None

            if cypher is None:
                cypher = "MATCH p=(a)-[r]->(b) RETURN p AS result UNION MATCH (n) RETURN n AS result"
            records = Neo4j.Query(driver, cypher, parameters=parameters or {}, database=database, silent=silent)
            if records is None:
                return None

            nodes_by_id = {}
            rels_by_id = {}

            def collect(value):
                if value is None:
                    return
                if Node is not None and isinstance(value, Node):
                    nodes_by_id[value.element_id] = value
                    return
                if Relationship is not None and isinstance(value, Relationship):
                    rels_by_id[value.element_id] = value
                    try:
                        nodes_by_id[value.start_node.element_id] = value.start_node
                        nodes_by_id[value.end_node.element_id] = value.end_node
                    except Exception:
                        pass
                    return
                if Path is not None and isinstance(value, Path):
                    for n in value.nodes:
                        nodes_by_id[n.element_id] = n
                    for r in value.relationships:
                        rels_by_id[r.element_id] = r
                        try:
                            nodes_by_id[r.start_node.element_id] = r.start_node
                            nodes_by_id[r.end_node.element_id] = r.end_node
                        except Exception:
                            pass
                    return
                if isinstance(value, dict):
                    for v in value.values():
                        collect(v)
                    return
                if isinstance(value, (list, tuple, set)):
                    for v in value:
                        collect(v)
                    return

            for rec in records:
                row = _record_to_dict(rec)
                for value in row.values():
                    collect(value)

            topologic_vertices = {}
            vertices = []
            edges = []
            for eid, node in nodes_by_id.items():
                props = dict(node.items())
                x = props.get("x", random.uniform(xMin, xMax))
                y = props.get("y", random.uniform(yMin, yMax))
                z = props.get("z", random.uniform(zMin, zMax))
                v = Vertex.ByCoordinates(x, y, z)
                props["id"] = eid
                props["labels"] = list(node.labels)
                d = Dictionary.ByPythonDictionary(props)
                v = Topology.SetDictionary(v, d)
                topologic_vertices[eid] = v
                vertices.append(v)

            for _, rel in rels_by_id.items():
                sv = topologic_vertices.get(rel.start_node.element_id)
                ev = topologic_vertices.get(rel.end_node.element_id)
                if sv is None or ev is None:
                    continue
                e = Edge.ByStartVertexEndVertex(sv, ev)
                props = dict(rel.items())
                props["id"] = rel.element_id
                props["type"] = rel.type
                d = Dictionary.ByPythonDictionary(props)
                e = Topology.SetDictionary(e, d)
                edges.append(e)
            return Graph.ByVerticesEdges(vertices, edges) if vertices else None
        except Exception as ex:
            if not silent:
                print(f"Neo4j.ToGraph - Error: {ex}. Returning None.")
            return None
