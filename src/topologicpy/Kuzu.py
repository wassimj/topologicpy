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

import contextlib
import json
import os
import threading
import time
import warnings
from typing import Any, Dict, List, Optional

try:
    import kuzu
except Exception:
    print("Kuzu - Installing required kuzu library.")
    try:
        os.system("pip install kuzu")
    except Exception:
        os.system("pip install kuzu --user")
    try:
        import kuzu
    except Exception:
        warnings.warn("Kuzu - Error: Could not import Kuzu.")
        kuzu = None


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


def _rows(result: Any) -> List[Dict[str, Any]]:
    if result is None:
        return []
    try:
        return result.rows_as_dict().get_all()
    except Exception:
        pass
    try:
        return result.get_all()
    except Exception:
        pass
    if isinstance(result, list):
        out = []
        for row in result:
            if isinstance(row, dict):
                out.append(row)
            else:
                try:
                    out.append(dict(row))
                except Exception:
                    out.append({"value": row})
        return out
    return []


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


def _object_to_dict(value: Any) -> Dict[str, Any]:
    """Best-effort conversion of Kuzu node/rel values to dictionaries."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    try:
        return dict(value)
    except Exception:
        pass
    out = {}
    for name in ("id", "graph_id", "label", "x", "y", "z", "props"):
        try:
            out[name] = getattr(value, name)
        except Exception:
            try:
                out[name] = value[name]
            except Exception:
                pass
    try:
        # Some Kuzu objects expose properties through _properties.
        props = getattr(value, "_properties")
        if isinstance(props, dict):
            out.update(props)
    except Exception:
        pass
    return out


def _safe_local_id(raw: Any, fallback: Any) -> str:
    raw = fallback if raw is None or str(raw).strip() == "" else raw
    raw = str(raw).strip()
    if raw == "":
        raw = str(fallback)
    return raw


def _ontology_scalar(props: Dict[str, Any], key: str, default: Any = "") -> Any:
    """Returns a scalar ontology/provenance value from a props dictionary."""
    try:
        value = props.get(key, default)
    except Exception:
        value = default
    if value is None:
        return default
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _annotate_graph_for_ontology(graph, ontology: bool = True, generatedBy: str = "Kuzu.UpsertGraph", silent: bool = True):
    """
    Best-effort ontology annotation before graph persistence.

    This intentionally fails silently so Kuzu remains usable without Ontology.py.
    """
    if not ontology:
        return graph
    try:
        from topologicpy.Graph import Graph
        if hasattr(Graph, "_AnnotateOntology"):
            return Graph._AnnotateOntology(graph, graphClass="top:Graph", generatedBy=generatedBy, silent=silent)
    except Exception:
        pass
    try:
        from topologicpy.Ontology import Ontology
        graph = Ontology.Annotate(graph, ontologyClass="top:Graph", category="graph", generatedBy=generatedBy, silent=True)
        try:
            from topologicpy.Graph import Graph
            vertices = Graph.Vertices(graph) or []
            edges = Graph.Edges(graph) or []
        except Exception:
            vertices = []
            edges = []
        for v in vertices:
            try:
                Ontology.Annotate(v, ontologyClass="top:Node", category="node", silent=True)
            except Exception:
                pass
        for e in edges:
            try:
                Ontology.Annotate(e, ontologyClass="top:Relationship", category="relationship", silent=True)
            except Exception:
                pass
    except Exception:
        pass
    return graph


class _DBCache:
    def __init__(self):
        self._lock = threading.RLock()
        self._cache: Dict[str, "kuzu.Database"] = {}

    def get(self, path: str) -> "kuzu.Database":
        if kuzu is None:
            raise RuntimeError("Kuzu - Error: Kuzu is not available.")
        if path is None:
            raise ValueError("Kuzu - Error: The input path is None.")
        path = os.path.abspath(os.path.expanduser(str(path)))
        with self._lock:
            db = self._cache.get(path)
            if db is None:
                db = kuzu.Database(path)
                self._cache[path] = db
            return db


class _WriteGate:
    def __init__(self):
        self._lock = threading.RLock()

    @contextlib.contextmanager
    def hold(self):
        with self._lock:
            yield


_db_cache = _DBCache()
_write_gate = _WriteGate()


class _ConnectionPool:
    def __init__(self, db: "kuzu.Database"):
        self.db = db
        self._local = threading.local()

    def _ensure(self) -> "kuzu.Connection":
        if not hasattr(self._local, "conn"):
            self._local.conn = kuzu.Connection(self.db)
        return self._local.conn

    @contextlib.contextmanager
    def connection(self, write: bool = False):
        conn = self._ensure()
        if write:
            with _write_gate.hold():
                yield conn
        else:
            yield conn


class _Mgr:
    def __init__(self, db_path: str):
        self.db_path = os.path.abspath(os.path.expanduser(str(db_path)))
        self._db = _db_cache.get(self.db_path)
        self._pool = _ConnectionPool(self._db)

    @contextlib.contextmanager
    def read(self):
        with self._pool.connection(write=False) as c:
            yield c

    @contextlib.contextmanager
    def write(self):
        with self._pool.connection(write=True) as c:
            yield c

    def exec(self, query: str, params: Optional[dict] = None, write: bool = False, retries: int = 5, backoff: float = 0.15):
        params = params or {}
        attempt = 0
        while True:
            try:
                with (self.write() if write else self.read()) as c:
                    res = c.execute(query, parameters=params)
                    try:
                        with res:
                            return _rows(res)
                    except Exception:
                        return _rows(res)
            except Exception:
                attempt += 1
                if (not write) or attempt > retries:
                    raise
                time.sleep(backoff * attempt)

    def ensure_schema(self):
        self.exec(
            """
            CREATE NODE TABLE IF NOT EXISTS Graph(
                id STRING,
                label STRING,
                ontology_class STRING,
                category STRING,
                uri STRING,
                source STRING,
                generated_by STRING,
                derived_from STRING,
                num_nodes INT64,
                num_edges INT64,
                props STRING,
                PRIMARY KEY(id)
            );
            """,
            write=True,
        )
        # Kuzu node tables currently require a single-column primary key. We keep
        # Vertex.id as a backend-global storage key and preserve the local vertex id
        # in props[vertexIDKey]. The storage key is normally "<graph_id>:<local_id>".
        self.exec(
            """
            CREATE NODE TABLE IF NOT EXISTS Vertex(
                id STRING,
                graph_id STRING,
                label STRING,
                ontology_class STRING,
                category STRING,
                uri STRING,
                source STRING,
                generated_by STRING,
                derived_from STRING,
                ifc_class STRING,
                ifc_guid STRING,
                x DOUBLE,
                y DOUBLE,
                z DOUBLE,
                props STRING,
                PRIMARY KEY(id)
            );
            """,
            write=True,
        )
        self.exec(
            """
            CREATE REL TABLE IF NOT EXISTS Edge(
                FROM Vertex TO Vertex,
                graph_id STRING,
                label STRING,
                ontology_class STRING,
                category STRING,
                uri STRING,
                source STRING,
                generated_by STRING,
                derived_from STRING,
                ifc_class STRING,
                ifc_guid STRING,
                props STRING
            );
            """,
            write=True,
        )


class Kuzu:
    # -------------------------------------------------------------------------
    # Core
    # -------------------------------------------------------------------------

    @staticmethod
    def EnsureSchema(manager, silent: bool = False) -> bool:
        try:
            manager.ensure_schema()
            return True
        except Exception as e:
            if not silent:
                print(f"Kuzu.EnsureSchema - Error: {e}. Returning False.")
            return False

    @staticmethod
    def Database(path: str, silent: bool = False):
        try:
            return _db_cache.get(path)
        except Exception as e:
            if not silent:
                print(f"Kuzu.Database - Error: {e}. Returning None.")
            return None

    @staticmethod
    def Connection(manager, silent: bool = False):
        try:
            return manager._pool._ensure()
        except Exception as e:
            if not silent:
                print(f"Kuzu.Connection - Error: {e}. Returning None.")
            return None

    @staticmethod
    def Manager(path: str, silent: bool = False):
        try:
            return _Mgr(path)
        except Exception as e:
            if not silent:
                print(f"Kuzu.Manager - Error: {e}. Returning None.")
            return None

    @staticmethod
    def Execute(manager, query: str, parameters: dict = None, write: bool = False, silent: bool = False):
        try:
            return manager.exec(query, parameters or {}, write=write)
        except Exception as e:
            if not silent:
                print(f"Kuzu.Execute - Error: {e}. Returning None.")
            return None

    @staticmethod
    def Query(manager, query: str, parameters: dict = None, silent: bool = False):
        return Kuzu.Execute(manager, query, parameters=parameters, write=False, silent=silent)

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    @staticmethod
    def UpsertGraph(manager,
                    graph,
                    graphIDKey: str = "id",
                    vertexIDKey: str = "id",
                    vertexLabelKey: str = "label",
                    defaultVertexLabel: str = "Node",
                    vertexCategoryKey: str = "category",
                    defaultVertexCategory="Node",
                    edgeLabelKey: str = "label",
                    defaultEdgeLabel: str = "CONNECTED_TO",
                    edgeCategoryKey: str = "category",
                    defaultEdgeCategory="Edge",
                    bidirectional: bool = True,
                    overwrite: bool = False,
                    mantissa: int = 6,
                    tolerance: float = 0.0001,
                    database: str = None,
                    ontology: bool = True,
                    silent: bool = False) -> str:
        """
        Upserts a TopologicPy graph into Kuzu using the canonical GraphDB schema.

        The signature mirrors Neo4j.UpsertGraph. The database argument is accepted
        for GraphDB compatibility and ignored because Kuzu database identity is the
        manager/path. If ontology is True, the graph, vertices, and edges are
        annotated with TopologicPy ontology metadata before persistence.
        """
        try:
            from topologicpy.Graph import Graph
            from topologicpy.Topology import Topology

            graph = _annotate_graph_for_ontology(graph, ontology=ontology, generatedBy="Kuzu.UpsertGraph", silent=True)

            manager.ensure_schema()

            graph_dict = Topology.Dictionary(graph)
            gid = _value_from_dict(graph_dict, graphIDKey, None) if graphIDKey is not None else None
            if gid is None or str(gid).strip() == "":
                gid = Topology.UUID(graph)
            gid = str(gid).strip()

            exists_rows = manager.exec(
                """
                MATCH (g:Graph)
                WHERE g.id = $gid
                RETURN COUNT(g) > 0 AS exists;
                """,
                {"gid": gid},
                write=False,
            ) or []
            exists = False
            if exists_rows:
                try:
                    exists = bool(exists_rows[0].get("exists"))
                except Exception:
                    exists = False

            if exists and not overwrite:
                if not silent:
                    print("Kuzu.UpsertGraph - Error: The graph already exists and overwrite is False. Returning None.")
                return None

            if overwrite:
                Kuzu.DeleteGraph(manager, gid, silent=True)

            g_props = _python_dictionary(graph_dict)
            g_label = str(g_props.get("label", ""))

            mesh_data = Graph.MeshData(graph, mantissa=mantissa)
            verts = mesh_data.get("vertices", [])
            v_props = mesh_data.get("vertexDictionaries", [])
            edges = mesh_data.get("edges", [])
            e_props = mesh_data.get("edgeDictionaries", [])

            edge_count = len(edges) * (2 if bidirectional else 1)
            manager.exec(
                """
                CREATE (g:Graph {
                    id:$id,
                    label:$label,
                    ontology_class:$ontology_class,
                    category:$category,
                    uri:$uri,
                    source:$source,
                    generated_by:$generated_by,
                    derived_from:$derived_from,
                    num_nodes:$num_nodes,
                    num_edges:$num_edges,
                    props:$props
                });
                """,
                {
                    "id": gid,
                    "label": g_label,
                    "ontology_class": _ontology_scalar(g_props, "ontology_class", "top:Graph"),
                    "category": _ontology_scalar(g_props, "category", "graph"),
                    "uri": _ontology_scalar(g_props, "uri", ""),
                    "source": _ontology_scalar(g_props, "source", ""),
                    "generated_by": _ontology_scalar(g_props, "generated_by", "Kuzu.UpsertGraph"),
                    "derived_from": _ontology_scalar(g_props, "derived_from", ""),
                    "num_nodes": int(len(verts)),
                    "num_edges": int(edge_count),
                    "props": _json_dumps(g_props),
                },
                write=True,
            )

            vertex_ids = []
            used_storage_ids = set()
            for i, xyz in enumerate(verts):
                props = dict(v_props[i] or {}) if i < len(v_props) else {}
                x, y, z = xyz

                raw_vid = _safe_local_id(props.get(vertexIDKey, None) if vertexIDKey is not None else None, i)
                if vertexIDKey is not None:
                    props[vertexIDKey] = raw_vid
                props["graph_id"] = gid

                storage_id = raw_vid if raw_vid.startswith(f"{gid}:") else f"{gid}:{raw_vid}"
                if storage_id in used_storage_ids:
                    suffix = 2
                    base = storage_id
                    while storage_id in used_storage_ids:
                        storage_id = f"{base}_{suffix}"
                        suffix += 1
                    # Preserve the modified local id when there was a duplicate inside one graph.
                    if vertexIDKey is not None:
                        props[vertexIDKey] = storage_id.split(":", 1)[-1]
                used_storage_ids.add(storage_id)
                props["_db_id"] = storage_id

                label = props.get(vertexLabelKey, None) if vertexLabelKey is not None else None
                if label is None or str(label).strip() == "":
                    label = defaultVertexLabel if defaultVertexLabel is not None else str(i)
                label = str(label).strip()

                if vertexCategoryKey is not None:
                    category = props.get(vertexCategoryKey, defaultVertexCategory)
                    if category is not None:
                        props[vertexCategoryKey] = category

                vertex_ids.append(storage_id)
                manager.exec(
                    """
                    CREATE (v:Vertex {
                        id:$id,
                        graph_id:$gid,
                        label:$label,
                        ontology_class:$ontology_class,
                        category:$category,
                        uri:$uri,
                        source:$source,
                        generated_by:$generated_by,
                        derived_from:$derived_from,
                        ifc_class:$ifc_class,
                        ifc_guid:$ifc_guid,
                        props:$props,
                        x:$x,
                        y:$y,
                        z:$z
                    });
                    """,
                    {
                        "id": storage_id,
                        "gid": gid,
                        "label": label,
                        "ontology_class": _ontology_scalar(props, "ontology_class", "top:Node"),
                        "category": _ontology_scalar(props, "category", "node"),
                        "uri": _ontology_scalar(props, "uri", ""),
                        "source": _ontology_scalar(props, "source", ""),
                        "generated_by": _ontology_scalar(props, "generated_by", ""),
                        "derived_from": _ontology_scalar(props, "derived_from", ""),
                        "ifc_class": _ontology_scalar(props, "ifc_class", ""),
                        "ifc_guid": _ontology_scalar(props, "ifc_guid", ""),
                        "x": float(x),
                        "y": float(y),
                        "z": float(z),
                        "props": _json_dumps(props),
                    },
                    write=True,
                )

            for i, edge_indices in enumerate(edges):
                try:
                    a_index = int(edge_indices[0])
                    b_index = int(edge_indices[1])
                except Exception:
                    continue
                if a_index < 0 or b_index < 0 or a_index >= len(vertex_ids) or b_index >= len(vertex_ids):
                    continue

                props = dict(e_props[i] or {}) if i < len(e_props) else {}
                label = props.get(edgeLabelKey, None) if edgeLabelKey is not None else None
                if label is None or str(label).strip() == "":
                    label = props.get("type", None)
                if label is None or str(label).strip() == "":
                    label = defaultEdgeLabel
                label = str(label).strip()
                if edgeLabelKey is not None:
                    props[edgeLabelKey] = label
                if edgeCategoryKey is not None:
                    category = props.get(edgeCategoryKey, defaultEdgeCategory)
                    if category is not None:
                        props[edgeCategoryKey] = category
                props["graph_id"] = gid

                a_id = vertex_ids[a_index]
                b_id = vertex_ids[b_index]
                rel_rows = [(a_id, b_id)]
                if bidirectional and a_id != b_id:
                    rel_rows.append((b_id, a_id))
                for start_id, end_id in rel_rows:
                    manager.exec(
                        """
                        MATCH (a:Vertex {id:$a}), (b:Vertex {id:$b})
                        CREATE (a)-[:Edge {
                            graph_id:$gid,
                            label:$label,
                            ontology_class:$ontology_class,
                            category:$category,
                            uri:$uri,
                            source:$source,
                            generated_by:$generated_by,
                            derived_from:$derived_from,
                            ifc_class:$ifc_class,
                            ifc_guid:$ifc_guid,
                            props:$props
                        }]->(b);
                        """,
                        {
                            "a": start_id,
                            "b": end_id,
                            "gid": gid,
                            "label": label,
                            "ontology_class": _ontology_scalar(props, "ontology_class", "top:Relationship"),
                            "category": _ontology_scalar(props, "category", "relationship"),
                            "uri": _ontology_scalar(props, "uri", ""),
                            "source": _ontology_scalar(props, "source", ""),
                            "generated_by": _ontology_scalar(props, "generated_by", ""),
                            "derived_from": _ontology_scalar(props, "derived_from", ""),
                            "ifc_class": _ontology_scalar(props, "ifc_class", ""),
                            "ifc_guid": _ontology_scalar(props, "ifc_guid", ""),
                            "props": _json_dumps(props),
                        },
                        write=True,
                    )
            return gid
        except Exception as e:
            if not silent:
                print(f"Kuzu.UpsertGraph - Error: {e}. Returning None.")
            return None

    @staticmethod
    def GraphByID(manager, graphID: str, ontology: bool = True, silent: bool = False):
        try:
            from topologicpy.Graph import Graph
            from topologicpy.Dictionary import Dictionary
            from topologicpy.Vertex import Vertex
            from topologicpy.Edge import Edge
            from topologicpy.Topology import Topology

            manager.ensure_schema()
            graphID = str(graphID)
            rows_g = manager.exec(
                """
                MATCH (g:Graph)
                WHERE g.id = $id
                RETURN g.id AS id, g.label AS label,
                       g.ontology_class AS ontology_class, g.category AS category,
                       g.uri AS uri, g.source AS source,
                       g.generated_by AS generated_by, g.derived_from AS derived_from,
                       g.num_nodes AS num_nodes, g.num_edges AS num_edges, g.props AS props;
                """,
                {"id": graphID},
                write=False,
            ) or []
            if not rows_g:
                return None

            g_row = rows_g[0]
            g_props = dict(_json_loads(g_row.get("props"), {}))
            if "label" not in g_props and g_row.get("label") is not None:
                g_props["label"] = g_row.get("label")
            for _k in ["ontology_class", "category", "uri", "source", "generated_by", "derived_from"]:
                if _k not in g_props and g_row.get(_k) not in [None, ""]:
                    g_props[_k] = g_row.get(_k)
            g_dict = Dictionary.ByPythonDictionary(g_props)

            rows_v = manager.exec(
                """
                MATCH (v:Vertex)
                WHERE v.graph_id = $gid
                RETURN v.id AS id, v.label AS label,
                       v.ontology_class AS ontology_class, v.category AS category,
                       v.uri AS uri, v.source AS source,
                       v.generated_by AS generated_by, v.derived_from AS derived_from,
                       v.ifc_class AS ifc_class, v.ifc_guid AS ifc_guid,
                       v.x AS x, v.y AS y, v.z AS z, v.props AS props
                ORDER BY v.id;
                """,
                {"gid": graphID},
                write=False,
            ) or []

            id_to_vertex = {}
            vertices = []
            for row in rows_v:
                x = row.get("x") if row.get("x") is not None else 0.0
                y = row.get("y") if row.get("y") is not None else 0.0
                z = row.get("z") if row.get("z") is not None else 0.0
                v = Vertex.ByCoordinates(float(x), float(y), float(z))
                props = dict(_json_loads(row.get("props"), {}))
                if "id" not in props:
                    # If no local id was preserved, use the backend storage id.
                    props["id"] = row.get("id")
                if "label" not in props:
                    props["label"] = row.get("label") or ""
                for _k in ["ontology_class", "category", "uri", "source", "generated_by", "derived_from", "ifc_class", "ifc_guid"]:
                    if _k not in props and row.get(_k) not in [None, ""]:
                        props[_k] = row.get(_k)
                d = Dictionary.ByPythonDictionary(props)
                v = Topology.SetDictionary(v, d)
                id_to_vertex[row.get("id")] = v
                vertices.append(v)

            rows_e = manager.exec(
                """
                MATCH (a:Vertex)-[r:Edge]->(b:Vertex)
                WHERE a.graph_id = $gid AND b.graph_id = $gid
                RETURN a.id AS a_id, b.id AS b_id, r.label AS label,
                       r.ontology_class AS ontology_class, r.category AS category,
                       r.uri AS uri, r.source AS source,
                       r.generated_by AS generated_by, r.derived_from AS derived_from,
                       r.ifc_class AS ifc_class, r.ifc_guid AS ifc_guid,
                       r.props AS props;
                """,
                {"gid": graphID},
                write=False,
            ) or []

            edges_out = []
            seen_pairs = set()
            for row in rows_e:
                a_id = row.get("a_id")
                b_id = row.get("b_id")
                # Suppress reverse duplicate edges caused by bidirectional storage.
                pair_key = tuple(sorted([str(a_id), str(b_id)]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                va = id_to_vertex.get(a_id)
                vb = id_to_vertex.get(b_id)
                if va is None or vb is None:
                    continue
                e = Edge.ByStartVertexEndVertex(va, vb)
                props = dict(_json_loads(row.get("props"), {}))
                if "label" not in props:
                    props["label"] = row.get("label") or "connect"
                for _k in ["ontology_class", "category", "uri", "source", "generated_by", "derived_from", "ifc_class", "ifc_guid"]:
                    if _k not in props and row.get(_k) not in [None, ""]:
                        props[_k] = row.get(_k)
                d = Dictionary.ByPythonDictionary(props)
                e = Topology.SetDictionary(e, d)
                edges_out.append(e)

            if not vertices:
                return None
            g = Graph.ByVerticesEdges(vertices, edges_out, ontology=ontology)
            g = Topology.SetDictionary(g, g_dict)
            if ontology:
                try:
                    from topologicpy.Graph import Graph as _Graph
                    g = _Graph._OntologyAnnotateGraph(g, graphClass="top:Graph", generatedBy="Kuzu.GraphByID", ontology=True, silent=True)
                except Exception:
                    pass
            return g
        except Exception as e:
            if not silent:
                print(f"Kuzu.GraphByID - Error: {e}. Returning None.")
            return None

    @staticmethod
    def GraphsByQuery(manager, query: str, parameters: dict = None, params: dict = None, ontology: bool = True, silent: bool = False):
        """
        Executes a Kuzu query and returns a list containing a TopologicPy graph
        constructed from the returned rows. If rows return graph_id/gid only,
        full graphs are reconstructed using GraphByID.
        """
        try:
            from topologicpy.Graph import Graph
            from topologicpy.Vertex import Vertex
            from topologicpy.Edge import Edge
            from topologicpy.Topology import Topology
            from topologicpy.Dictionary import Dictionary

            manager.ensure_schema()
            qparams = parameters if parameters is not None else (params or {})
            rows = manager.exec(query, qparams, write=False) or []
            if not rows:
                return []

            # If the result only exposes graph ids, return full graph(s).
            gids = []
            has_graph_objects = False
            for row in rows:
                for value in row.values():
                    if isinstance(value, dict) or hasattr(value, "_properties"):
                        has_graph_objects = True
                        break
                gid = row.get("graph_id") or row.get("gid")
                if gid is not None and gid not in gids:
                    gids.append(gid)
            if gids and not has_graph_objects:
                graphs = []
                for gid in gids:
                    g = Kuzu.GraphByID(manager, str(gid), ontology=ontology, silent=silent)
                    if g is not None:
                        graphs.append(g)
                return graphs

            node_rows = {}
            rel_rows = []

            def _consume(value):
                if value is None:
                    return
                if isinstance(value, dict):
                    # Direct canonical vertex row.
                    if "id" in value and ("x" in value or "label" in value or "graph_id" in value):
                        node_rows[str(value.get("id"))] = dict(value)
                        return
                    for v in value.values():
                        _consume(v)
                    return
                if isinstance(value, (list, tuple, set)):
                    for v in value:
                        _consume(v)
                    return
                obj = _object_to_dict(value)
                if obj:
                    if "id" in obj and ("x" in obj or "label" in obj or "graph_id" in obj):
                        node_rows[str(obj.get("id"))] = obj
                    elif "a_id" in obj or "b_id" in obj:
                        rel_rows.append(obj)

            for row in rows:
                # Prefer explicit n/r/m style query outputs.
                for value in row.values():
                    _consume(value)
                if "n" in row:
                    _consume(row.get("n"))
                if "m" in row:
                    _consume(row.get("m"))
                if "r" in row:
                    rel_rows.append(_object_to_dict(row.get("r")) or row.get("r"))
                if "a_id" in row and "b_id" in row:
                    rel_rows.append(dict(row))

            # Fallback for scalar canonical rows.
            for row in rows:
                if "id" in row and row.get("id") is not None:
                    node_rows[str(row.get("id"))] = row

            id_to_vertex = {}
            vertices = []
            for nid, row in node_rows.items():
                props = dict(_json_loads(row.get("props"), {}))
                if "id" not in props:
                    props["id"] = row.get("id")
                if "label" not in props and row.get("label") is not None:
                    props["label"] = row.get("label")
                x = row.get("x", 0.0)
                y = row.get("y", 0.0)
                z = row.get("z", 0.0)
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
                v = Topology.SetDictionary(v, Dictionary.ByPythonDictionary(props))
                id_to_vertex[nid] = v
                vertices.append(v)

            edges_out = []
            seen = set()
            for rel in rel_rows:
                if not isinstance(rel, dict):
                    rel = _object_to_dict(rel)
                a_id = rel.get("a_id") or rel.get("a") or rel.get("src") or rel.get("source")
                b_id = rel.get("b_id") or rel.get("b") or rel.get("dst") or rel.get("target")
                # Kuzu relationship objects may not expose endpoints, so queries that
                # need edges should return a.id AS a_id and b.id AS b_id as well.
                if a_id is None or b_id is None:
                    continue
                key = tuple(sorted([str(a_id), str(b_id)]))
                if key in seen:
                    continue
                seen.add(key)
                va = id_to_vertex.get(str(a_id))
                vb = id_to_vertex.get(str(b_id))
                if va is None or vb is None:
                    continue
                e = Edge.ByStartVertexEndVertex(va, vb)
                props = dict(_json_loads(rel.get("props"), {}))
                if "label" not in props and rel.get("label") is not None:
                    props["label"] = rel.get("label")
                e = Topology.SetDictionary(e, Dictionary.ByPythonDictionary(props))
                edges_out.append(e)

            if not vertices:
                return []
            g = Graph.ByVerticesEdges(vertices, edges_out, ontology=ontology)
            if ontology:
                try:
                    g = Graph._OntologyAnnotateGraph(g, graphClass="top:Graph", generatedBy="Kuzu.GraphsByQuery", ontology=True, silent=True)
                except Exception:
                    pass
            return [g]
        except Exception as e:
            if not silent:
                print(f"Kuzu.GraphsByQuery - Error: {e}. Returning None.")
            return None

    @staticmethod
    def DeleteGraph(manager, graphID: str, silent: bool = False) -> bool:
        try:
            manager.ensure_schema()
            graphID = str(graphID)
            # Delete relationships first. Prefer r.graph_id, but keep endpoint fallback
            # for databases created by older versions without Edge.graph_id.
            try:
                manager.exec(
                    """
                    MATCH (a:Vertex)-[r:Edge]->(b:Vertex)
                    WHERE r.graph_id = $gid
                    DELETE r;
                    """,
                    {"gid": graphID},
                    write=True,
                )
            except Exception:
                manager.exec(
                    """
                    MATCH (a:Vertex)-[r:Edge]->(b:Vertex)
                    WHERE a.graph_id = $gid AND b.graph_id = $gid
                    DELETE r;
                    """,
                    {"gid": graphID},
                    write=True,
                )
            manager.exec("MATCH (v:Vertex) WHERE v.graph_id = $gid DELETE v;", {"gid": graphID}, write=True)
            manager.exec("MATCH (g:Graph) WHERE g.id = $gid DELETE g;", {"gid": graphID}, write=True)
            return True
        except Exception as e:
            if not silent:
                print(f"Kuzu.DeleteGraph - Error: {e}. Returning False.")
            return False

    @staticmethod
    def EmptyDatabase(manager, dropSchema: bool = False, recreateSchema: bool = True, silent: bool = False) -> bool:
        try:
            if dropSchema:
                for stmt in [
                    "DROP TABLE IF EXISTS Edge;",
                    "DROP TABLE IF EXISTS Vertex;",
                    "DROP TABLE IF EXISTS Graph;",
                ]:
                    try:
                        manager.exec(stmt, write=True)
                    except Exception as e:
                        if not silent:
                            print(f"Kuzu.EmptyDatabase - Warning: {e}")
                if recreateSchema:
                    manager.ensure_schema()
                return True

            manager.ensure_schema()
            manager.exec("MATCH (a)-[r]->(b) DELETE r;", write=True)
            manager.exec("MATCH (v:Vertex) DELETE v;", write=True)
            manager.exec("MATCH (g:Graph) DELETE g;", write=True)
            return True
        except Exception as e:
            if not silent:
                print(f"Kuzu.EmptyDatabase - Error: {e}. Returning False.")
            return False

    @staticmethod
    def ListGraphs(manager, where: dict = None, limit: int = 100, offset: int = 0, silent: bool = False) -> list[dict]:
        try:
            manager.ensure_schema()
            where = where or {}
            conds = []
            params = {}
            if where.get("id"):
                conds.append("g.id = $id")
                params["id"] = str(where["id"])
            if where.get("label"):
                conds.append("g.label CONTAINS $label_sub")
                params["label_sub"] = str(where["label"])
            if where.get("ontology_class"):
                conds.append("g.ontology_class = $ontology_class")
                params["ontology_class"] = str(where["ontology_class"])
            if where.get("category"):
                conds.append("g.category = $category")
                params["category"] = str(where["category"])
            if where.get("source"):
                conds.append("g.source CONTAINS $source")
                params["source"] = str(where["source"])
            if where.get("generated_by"):
                conds.append("g.generated_by = $generated_by")
                params["generated_by"] = str(where["generated_by"])
            if where.get("props_contains"):
                conds.append("g.props CONTAINS $props_sub")
                params["props_sub"] = str(where["props_contains"])
            if where.get("props_equals"):
                conds.append("g.props = $props_equals")
                params["props_equals"] = str(where["props_equals"])
            if where.get("min_nodes") is not None:
                conds.append("g.num_nodes >= $min_nodes")
                params["min_nodes"] = int(where["min_nodes"])
            if where.get("max_nodes") is not None:
                conds.append("g.num_nodes <= $max_nodes")
                params["max_nodes"] = int(where["max_nodes"])
            if where.get("min_edges") is not None:
                conds.append("g.num_edges >= $min_edges")
                params["min_edges"] = int(where["min_edges"])
            if where.get("max_edges") is not None:
                conds.append("g.num_edges <= $max_edges")
                params["max_edges"] = int(where["max_edges"])
            where_clause = ("WHERE " + " AND ".join(conds)) if conds else ""
            params["offset"] = max(0, int(offset or 0))
            params["limit"] = max(0, int(limit or 100))
            return manager.exec(
                f"""
                MATCH (g:Graph)
                {where_clause}
                RETURN g.id AS id, g.label AS label,
                       g.ontology_class AS ontology_class, g.category AS category,
                       g.uri AS uri, g.source AS source,
                       g.generated_by AS generated_by, g.derived_from AS derived_from,
                       g.num_nodes AS num_nodes, g.num_edges AS num_edges, g.props AS props
                ORDER BY g.id
                SKIP $offset LIMIT $limit;
                """,
                params,
                write=False,
            ) or []
        except Exception as e:
            if not silent:
                print(f"Kuzu.ListGraphs - Error: {e}. Returning None.")
            return None

    # -------------------------------------------------------------------------
    # CSV import: Graph.ByCSVPath -> Kuzu.UpsertGraph
    # -------------------------------------------------------------------------

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
        tolerance=0.0001, ontology: bool = True, silent=False):
        try:
            from topologicpy.Graph import Graph
            manager.ensure_schema()
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
                tolerance=tolerance, ontology=ontology, silent=silent)
            if graphs is None:
                if not silent:
                    print("Kuzu.ByCSVPath - Error: Graph.ByCSVPath returned None. Returning None.")
                return None
            if not isinstance(graphs, list):
                graphs = [graphs]
            graph_ids = []
            for graph in graphs:
                gid = Kuzu.UpsertGraph(
                    manager,
                    graph,
                    graphIDKey=graphIDHeader,
                    vertexIDKey=nodeIDHeader,
                    vertexLabelKey=nodeLabelHeader,
                    edgeLabelKey=edgeLabelHeader,
                    overwrite=True,
                    bidirectional=True,
                    tolerance=tolerance,
                    ontology=ontology,
                    silent=silent,
                )
                if gid is not None:
                    graph_ids.append(gid)
            return {"graphs_upserted": len(graph_ids), "graph_ids": graph_ids}
        except Exception as e:
            if not silent:
                print(f"Kuzu.ByCSVPath - Error: {e}. Returning None.")
            return None

    # -------------------------------------------------------------------------
    # Corpus analytics
    # -------------------------------------------------------------------------

    @staticmethod
    def FetchAllPairs(manager, undirected: bool = True, silent: bool = False) -> list[dict]:
        try:
            manager.ensure_schema()
            rows = manager.exec(
                """
                MATCH (a:Vertex)-[r:Edge]->(b:Vertex)
                RETURN a.label AS a_label, b.label AS b_label, COUNT(*) AS count
                ORDER BY count DESC;
                """,
                write=False,
            ) or []
            counts = {}
            for row in rows:
                a = _normalize_label(row.get("a_label"))
                b = _normalize_label(row.get("b_label"))
                if not a or not b:
                    continue
                key = tuple(sorted([a, b])) if undirected else (a, b)
                counts[key] = counts.get(key, 0) + int(row.get("count", 0) or 0)
            out = []
            for key, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
                a, b = key
                out.append({"a_label": a, "b_label": b, "pair": [a, b], "count": int(count)})
            return out
        except Exception as e:
            if not silent:
                print(f"Kuzu.FetchAllPairs - Error: {e}. Returning None.")
            return None

    @staticmethod
    def CandidateCountsForLabels(manager, labels, excludeLabels=None, limit: int = 50, silent: bool = False) -> list[dict]:
        try:
            manager.ensure_schema()
            if isinstance(labels, str):
                labels = [labels]
            labels = [_normalize_label(x) for x in (labels or []) if _normalize_label(x)]
            exclude = set([_normalize_label(x) for x in (excludeLabels or []) if _normalize_label(x)])
            if not labels:
                return []

            rows = manager.exec(
                """
                MATCH (a:Vertex)-[r:Edge]->(b:Vertex)
                WHERE a.label IN $labels
                RETURN b.label AS label, a.label AS attach_to, COUNT(*) AS count
                UNION ALL
                MATCH (a:Vertex)-[r:Edge]->(b:Vertex)
                WHERE b.label IN $labels
                RETURN a.label AS label, b.label AS attach_to, COUNT(*) AS count;
                """,
                {"labels": labels},
                write=False,
            ) or []

            accum = {}
            for row in rows:
                label = _normalize_label(row.get("label"))
                attach = _normalize_label(row.get("attach_to"))
                if not label or label in exclude:
                    continue
                c = int(row.get("count", 0) or 0)
                rec = accum.setdefault(label, {"label": label, "count": 0, "attach_to_counts": {}})
                rec["count"] += c
                if attach:
                    rec["attach_to_counts"][attach] = rec["attach_to_counts"].get(attach, 0) + c
            out = []
            for rec in accum.values():
                rec["attach_to_labels"] = [k for k, _ in sorted(rec["attach_to_counts"].items(), key=lambda kv: kv[1], reverse=True)]
                out.append(rec)
            out.sort(key=lambda r: r.get("count", 0), reverse=True)
            return out[: max(1, int(limit or 50))]
        except Exception as e:
            if not silent:
                print(f"Kuzu.CandidateCountsForLabels - Error: {e}. Returning None.")
            return None

    @staticmethod
    def MaxNeighborsForLabel(manager, label, silent: bool = False):
        try:
            manager.ensure_schema()
            label = _normalize_label(label)
            if not label:
                return None
            rows = manager.exec(
                """
                MATCH (v:Vertex)
                WHERE v.label = $label
                OPTIONAL MATCH (v)-[:Edge]-(n:Vertex)
                WITH v, COUNT(DISTINCT n.id) AS degree
                RETURN MAX(degree) AS max_neighbors;
                """,
                {"label": label},
                write=False,
            ) or []
            if not rows:
                return None
            value = rows[0].get("max_neighbors")
            return None if value is None else int(value)
        except Exception as e:
            if not silent:
                print(f"Kuzu.MaxNeighborsForLabel - Error: {e}. Returning None.")
            return None

    @staticmethod
    def FindBestExampleForLabel(manager, label, attachTo=None, silent: bool = False):
        try:
            manager.ensure_schema()
            label = _normalize_label(label)
            attachTo = _normalize_label(attachTo)
            if not label:
                return None
            if attachTo:
                rows = manager.exec(
                    """
                    MATCH (v:Vertex)-[:Edge]-(n:Vertex)
                    WHERE v.label = $label AND n.label = $attach
                    WITH v, COUNT(DISTINCT n.id) AS degree
                    RETURN v.id AS id, v.id AS vertex_id, v.graph_id AS graph_id, v.label AS label,
                           v.x AS x, v.y AS y, v.z AS z, v.props AS props, degree AS degree
                    ORDER BY degree DESC
                    LIMIT 1;
                    """,
                    {"label": label, "attach": attachTo},
                    write=False,
                ) or []
            else:
                rows = manager.exec(
                    """
                    MATCH (v:Vertex)
                    WHERE v.label = $label
                    OPTIONAL MATCH (v)-[:Edge]-(n:Vertex)
                    WITH v, COUNT(DISTINCT n.id) AS degree
                    RETURN v.id AS id, v.id AS vertex_id, v.graph_id AS graph_id, v.label AS label,
                           v.x AS x, v.y AS y, v.z AS z, v.props AS props, degree AS degree
                    ORDER BY degree DESC
                    LIMIT 1;
                    """,
                    {"label": label},
                    write=False,
                ) or []
            if not rows:
                return None
            row = dict(rows[0])
            props = dict(_json_loads(row.get("props"), {}))
            row["props"] = props
            # Add neighbor labels/counts for the selected vertex.
            neigh = manager.exec(
                """
                MATCH (v:Vertex)-[:Edge]-(n:Vertex)
                WHERE v.id = $id
                RETURN n.label AS label, COUNT(*) AS count
                ORDER BY count DESC;
                """,
                {"id": row.get("id")},
                write=False,
            ) or []
            counts = {}
            for r in neigh:
                lab = _normalize_label(r.get("label"))
                if lab:
                    counts[lab] = counts.get(lab, 0) + int(r.get("count", 0) or 0)
            labels = list(counts.keys())
            row["neighbour_labels"] = labels
            row["neighbor_labels"] = labels
            row["neighbour_counts"] = counts
            row["neighbor_counts"] = counts
            return row
        except Exception as e:
            if not silent:
                print(f"Kuzu.FindBestExampleForLabel - Error: {e}. Returning None.")
            return None


    # -------------------------------------------------------------------------
    # Ontology-oriented queries
    # -------------------------------------------------------------------------

    @staticmethod
    def OntologyClasses(manager, elementType: str = "all", silent: bool = False):
        """
        Returns ontology class counts for graphs, vertices, edges, or all stored elements.

        Parameters
        ----------
        manager : Kuzu manager
            The input Kuzu manager.
        elementType : str , optional
            One of "graph", "vertex", "edge", or "all". Default is "all".
        silent : bool , optional
            If True, suppresses warning/error messages. Default is False.

        Returns
        -------
        list
            A list of dictionaries with element_type, ontology_class, and count.
        """
        try:
            manager.ensure_schema()
            elementType = str(elementType or "all").strip().lower()
            rows = []
            if elementType in ["graph", "graphs", "all"]:
                rows += manager.exec(
                    """
                    MATCH (g:Graph)
                    WHERE g.ontology_class IS NOT NULL AND g.ontology_class <> ""
                    RETURN "graph" AS element_type, g.ontology_class AS ontology_class, COUNT(*) AS count
                    ORDER BY count DESC;
                    """,
                    write=False,
                ) or []
            if elementType in ["vertex", "vertices", "node", "nodes", "all"]:
                rows += manager.exec(
                    """
                    MATCH (v:Vertex)
                    WHERE v.ontology_class IS NOT NULL AND v.ontology_class <> ""
                    RETURN "vertex" AS element_type, v.ontology_class AS ontology_class, COUNT(*) AS count
                    ORDER BY count DESC;
                    """,
                    write=False,
                ) or []
            if elementType in ["edge", "edges", "relationship", "relationships", "all"]:
                rows += manager.exec(
                    """
                    MATCH (:Vertex)-[r:Edge]->(:Vertex)
                    WHERE r.ontology_class IS NOT NULL AND r.ontology_class <> ""
                    RETURN "edge" AS element_type, r.ontology_class AS ontology_class, COUNT(*) AS count
                    ORDER BY count DESC;
                    """,
                    write=False,
                ) or []
            return rows
        except Exception as e:
            if not silent:
                print(f"Kuzu.OntologyClasses - Error: {e}. Returning None.")
            return None

    @staticmethod
    def GraphsByOntologyClass(manager, ontologyClass: str, silent: bool = False):
        """
        Returns TopologicPy graphs whose stored graph ontology_class matches the input class.
        """
        try:
            ontologyClass = str(ontologyClass or "").strip()
            if ontologyClass == "":
                return []
            rows = manager.exec(
                """
                MATCH (g:Graph)
                WHERE g.ontology_class = $ontology_class
                RETURN g.id AS graph_id;
                """,
                {"ontology_class": ontologyClass},
                write=False,
            ) or []
            graphs = []
            for row in rows:
                gid = row.get("graph_id")
                if gid is None:
                    continue
                graph = Kuzu.GraphByID(manager, str(gid), ontology=ontology, silent=silent)
                if graph is not None:
                    graphs.append(graph)
            return graphs
        except Exception as e:
            if not silent:
                print(f"Kuzu.GraphsByOntologyClass - Error: {e}. Returning None.")
            return None

    @staticmethod
    def VerticesByOntologyClass(manager, ontologyClass: str, graphID: str = None, silent: bool = False):
        """
        Returns stored vertex rows matching the input ontology class.

        Parameters
        ----------
        manager : Kuzu manager
            The input Kuzu manager.
        ontologyClass : str
            The ontology class, for example "top:Room".
        graphID : str , optional
            Optional graph id filter. Default is None.
        silent : bool , optional
            If True, suppresses warning/error messages. Default is False.

        Returns
        -------
        list
            A list of vertex row dictionaries. The props value is decoded when possible.
        """
        try:
            ontologyClass = str(ontologyClass or "").strip()
            if ontologyClass == "":
                return []
            params = {"ontology_class": ontologyClass}
            graph_clause = ""
            if graphID is not None:
                params["graph_id"] = str(graphID)
                graph_clause = "AND v.graph_id = $graph_id"
            rows = manager.exec(
                f"""
                MATCH (v:Vertex)
                WHERE v.ontology_class = $ontology_class
                {graph_clause}
                RETURN v.id AS id, v.graph_id AS graph_id, v.label AS label,
                       v.ontology_class AS ontology_class, v.category AS category,
                       v.uri AS uri, v.source AS source,
                       v.generated_by AS generated_by, v.derived_from AS derived_from,
                       v.ifc_class AS ifc_class, v.ifc_guid AS ifc_guid,
                       v.x AS x, v.y AS y, v.z AS z, v.props AS props
                ORDER BY v.graph_id, v.id;
                """,
                params,
                write=False,
            ) or []
            for row in rows:
                if isinstance(row, dict) and "props" in row:
                    row["props"] = _json_loads(row.get("props"), {})
            return rows
        except Exception as e:
            if not silent:
                print(f"Kuzu.VerticesByOntologyClass - Error: {e}. Returning None.")
            return None

