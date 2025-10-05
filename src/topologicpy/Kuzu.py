from __future__ import annotations
import threading, contextlib, time, json
from typing import Dict, Any, List, Optional

import os
import warnings

try:
    import kuzu
except:
    print("Kuzu - Installing required kuzu library.")
    try:
        os.system("pip install kuzu")
    except:
        os.system("pip install kuzu --user")
    try:
        import kuzu
    except:
        warnings.warn("Kuzu - Error: Could not import Kuzu.")
        kuzu = None


class _DBCache:
    """
    One kuzu.Database per path. Thread-safe and process-local.
    """
    def __init__(self):
        self._lock = threading.RLock()
        self._cache: Dict[str, "kuzu.Database"] = {}

    def get(self, path: str) -> "kuzu.Database":
        if kuzu is None:
            raise "Kuzu - Error: Kuzu is not available"
        with self._lock:
            db = self._cache.get(path)
            if db is None:
                db = kuzu.Database(path)
                self._cache[path] = db
            return db

class _WriteGate:
    """
    Serialize writes to avoid IO lock contention.
    """
    def __init__(self):
        self._lock = threading.RLock()

    @contextlib.contextmanager
    def hold(self):
        with self._lock:
            yield

_db_cache = _DBCache()
_write_gate = _WriteGate()

class _ConnectionPool:
    """
    Per-thread kuzu.Connection pool bound to a Database instance.
    """
    def __init__(self, db: "kuzu.Database"):
        self.db = db
        self._local = threading.local()

    def _ensure(self) -> "kuzu.Connection":
        if not hasattr(self._local, "conn"):
            self._local.conn = kuzu.Connection(self.db)
        return self._local.conn

    @contextlib.contextmanager
    def connection(self, write: bool = False, retries: int = 5, backoff: float = 0.15):
        conn = self._ensure()
        if not write:
            yield conn
            return
        # Serialize writes and retry transient failures
        with _write_gate.hold():
            attempt = 0
            while True:
                try:
                    yield conn
                    break
                except Exception as e:
                    attempt += 1
                    if attempt > retries:
                        raise f"Kuzu write failed after {retries} retries: {e}"
                    time.sleep(backoff * attempt)

class _Mgr:
    """
    Lightweight facade (per-db-path) providing read/write execution and schema bootstrap.
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._db = _db_cache.get(db_path)
        self._pool = _ConnectionPool(self._db)

    @contextlib.contextmanager
    def read(self):
        with self._pool.connection(write=False) as c:
            yield c

    @contextlib.contextmanager
    def write(self):
        with self._pool.connection(write=True) as c:
            yield c

    def exec(self, query: str, params: Optional[dict] = None, write: bool = False):
        with (self.write() if write else self.read()) as c:
            with c.execute(query, parameters=params or {}) as res:
                try:
                    return res.rows_as_dict().get_all()
                except Exception:
                    return None

    def ensure_schema(self):
        # Node tables
        self.exec("""
        CREATE NODE TABLE IF NOT EXISTS Graph(
            id STRING,
            label STRING,
            num_nodes INT64,
            num_edges INT64,
            props STRING,
            PRIMARY KEY(id)
        );
        """, write=True)
        self.exec("""
        CREATE NODE TABLE IF NOT EXISTS Vertex(
            id STRING,
            graph_id STRING,
            label STRING,
            x DOUBLE,
            y DOUBLE,
            z DOUBLE,
            props STRING,
            PRIMARY KEY(id)
        );
        """, write=True)
        
        # Relationship tables
        self.exec("""
        CREATE REL TABLE IF NOT EXISTS Edge(FROM Vertex TO Vertex, label STRING, props STRING);
        """, write=True)

        # Figure out later if we need sessions and steps
        # self.exec("""
        # CREATE NODE TABLE IF NOT EXISTS Session(
        #     id STRING,
        #     title STRING,
        #     created_at STRING,
        #     PRIMARY KEY(id)
        # );
        # """, write=True)
        # self.exec("""
        # CREATE NODE TABLE IF NOT EXISTS Step(
        #     id STRING,
        #     session_id STRING,
        #     idx INT64,
        #     action STRING,
        #     ok BOOL,
        #     message STRING,
        #     snapshot_before STRING,
        #     snapshot_after STRING,
        #     evidence STRING,
        #     created_at STRING,
        #     PRIMARY KEY(id)
        # );
        # """, write=True)
        # self.exec("CREATE REL TABLE IF NOT EXISTS SessionHasStep(FROM Session TO Step);", write=True)


class Kuzu:
    # ---------- Core (DB + Connection + Schema) ----------
    @staticmethod
    def EnsureSchema(manager, silent: bool = False) -> bool:
        """
        Ensures the required Kùzu schema exists in the database at `path`.

        Parameters
        ----------
        manager : Kuzu.Manager
            Path to the Kùzu database. It will be created if it does not exist.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        try:
            manager.ensure_schema()
            return True
        except Exception as e:
            if not silent:
                print(f"Kuzu.EnsureSchema - Error: {e}. Returning False.")
            return False

    @staticmethod
    def Database(path: str, silent: bool = False):
        """
        Returns the underlying `kuzu.Database` instance for `path`.

        Parameters
        ----------
        path : str
            Path to the Kùzu database. It will be created if it does not exist.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        kuzu.Database
            The Kuzu database found at the path.
        """
        try:
            return _db_cache.get(path)
        except Exception as e:
            if not silent:
                print(f"Kuzu.Database - Error: {e}. Returning None.")
            return None

    @staticmethod
    def Connection(manager, silent: bool = False):
        """
        Returns a `kuzu.Connection` bound to the database at `path`.

        Parameters
        ----------
        manager : Kuzu.Manager
            The Manager to the Kùzu database.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        kuzu.Connection
            The Kuzu live connection. Do NOT use across threads.
        """
        try:
            with manager.read() as c:
                return c  # Note: returns a live connection (do not use across threads)
        except Exception as e:
            if not silent:
                print(f"Kuzu.Connection - Error: {e}. Returning None.")
            return None        
    
    @staticmethod
    def Manager(path: str, silent: bool = False):
        """
        Returns a lightweight manager bound to the database at `path`.
        Parameters
        ----------
        path : str
            Path to the Kùzu database. It will be created if it does not exist.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        Kuzu.Manager
            The Kuzu Manager.
        """
        try:
            return _Mgr(path)
        except Exception as e:
            if not silent:
                print(f"Kuzu.Manager - Error: {e}. Returning None.")
            return None 

    @staticmethod
    def UpsertGraph(manager,
                    graph,
                    graphIDKey: str = None,
                    vertexIDKey: str = None,
                    vertexLabelKey: str = None,
                    mantissa: int = 6,
                    silent: bool = False) -> str:
        """
        Upserts (deletes prior + inserts new) a TopologicPy graph.

        Parameters
        ----------
        manager : Kuzu.Manager
            The Kuzu database manager.
        graph : topologicpy.Graph
            The input TopologicPy graph.
        graphIDKey : str , optional
            The graph dictionary key under which the graph ID is stored. If None, a UUID is generated and stored under 'id'.
        vertexIDKey : str , optional
            The vertex dictionary key under which the vertex ID is stored. If None, a UUID is generated and stored under 'id'.
        edgeIDKey : str , optional
            The edge dictionary key under which the edge ID is stored. If None, a UUID is generated and stored under 'id'.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        str
            The graph_id used.
        """
        from topologicpy.Graph import Graph
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        d = Topology.Dictionary(graph)
        if graphIDKey is None:
            gid =  Topology.UUID(graph)
        else:
            gid = Dictionary.ValueAtKey(d, graphIDKey, Topology.UUID(graph))
        g_props = Dictionary.PythonDictionary(d)
        mesh_data = Graph.MeshData(graph, mantissa=mantissa)
        verts = mesh_data['vertices']
        v_props = mesh_data['vertexDictionaries']
        edges = mesh_data['edges']
        e_props = mesh_data['edgeDictionaries']
        num_nodes = len(verts)
        num_edges = len(edges)
        try:
            manager.ensure_schema()
            # Upsert Graph
            manager.exec("MATCH (g:Graph) WHERE g.id = $id DELETE g;", {"id": gid}, write=True)
            manager.exec("""
                CREATE (g:Graph {id:$id, num_nodes:$num_nodes, num_edges: $num_edges, props:$props});
            """, {"id": gid, "num_nodes": num_nodes, "num_edges": num_edges, "props": json.dumps(g_props)}, write=True)

            # Remove existing vertices/edges for this graph_id
            manager.exec("""
                MATCH (a:Vertex)-[r:Edge]->(b:Vertex)
                WHERE a.graph_id = $gid AND b.graph_id = $gid
                DELETE r;
            """, {"gid": gid}, write=True)
            manager.exec("MATCH (v:Vertex) WHERE v.graph_id = $gid DELETE v;", {"gid": gid}, write=True)

            # Insert vertices
            for i, v in enumerate(verts):
                x,y,z = v
                if vertexIDKey is None:
                    vid = f"{gid}:{i}"
                else:
                    vid = v_props[i].get(vertexIDKey, f"{gid}:{i}")
                if vertexLabelKey is None:
                    label = str(i)
                else:
                    label = v_props[i].get(vertexIDKey, str(i))
                manager.exec("""
                    CREATE (v:Vertex {id:$id, graph_id:$gid, label:$label, props:$props, x:$x, y:$y, z:$z});
                """, {"id": vid, "gid": gid, "label": label, "x": x, "y": y, "z": z,
                        "props": json.dumps(v_props[i])}, write=True)

            # Insert edges
            for i, e in enumerate(edges):
                a_id = v_props[e[0]].get(vertexIDKey, f"{gid}:{e[0]}")
                b_id = v_props[e[1]].get(vertexIDKey, f"{gid}:{e[1]}")
                manager.exec("""
                    MATCH (a:Vertex {id:$a}), (b:Vertex {id:$b})
                    CREATE (a)-[:Edge {label:$label, props:$props}]->(b);
                """, {"a": a_id, "b": b_id,
                        "label": e_props[i].get("label", str(i)),
                        "props": json.dumps(e_props[i])}, write=True)

            return gid
        except Exception as e:
            if not silent:
                print(f"Kuzu.UpsertGraph - Error: {e}. Returning None.")
            return None

    @staticmethod
    def GraphByID(manager, graphID: str, silent: bool = False):
        """
        Constructs a TopologicPy graph from from Kùzu using the graphID input parameter.

        Parameters
        ----------
        manager : Kuzu.Manager
            The manager of the Kùzu database.
        graphID : str , optional
            The graph ID to retrieve from Kùzu.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologicpy.Graph
            A new TopologicPy Graph, or None on error.
        """
        import random
        from topologicpy.Graph import Graph
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        try:
            manager.ensure_schema()
            # Read the Graph
            g = manager.exec("""
                    MATCH (g:Graph) WHERE g.id = $id
                    RETURN g.id AS id, g.num_nodes AS num_nodes, g.num_edges AS num_edges, g.props AS props
                    ;
                     """, {"id": graphID}, write=False) or None
            if g is None:
                return None
            g = g[0]
            g_dict = dict(json.loads(g.get("props") or "{}") or {})
            g_dict = Dictionary.ByPythonDictionary(g_dict)
            # Read vertices
            rows_v = manager.exec("""
                MATCH (v:Vertex) WHERE v.graph_id = $gid
                RETURN v.id AS id, v.label AS label, v.x AS x, v.y AS y, v.z AS z, v.props AS props
                ORDER BY id;
            """, {"gid": graphID}, write=False) or []

            id_to_vertex = {}
            vertices = []
            for row in rows_v:
                try:
                    x = row.get("x", random.uniform(0,1000))
                    y = row.get("y", random.uniform(0,1000))
                    z = row.get("z", random.uniform(0,1000))
                except:
                    x = random.uniform(0,1000)
                    y = random.uniform(0,1000)
                    z = random.uniform(0,1000)
                v = Vertex.ByCoordinates(x,y,z)
                props = {}
                try:
                    props = json.loads(row.get("props") or "{}")
                except Exception:
                    props = {}
                # Ensure 'label' key present
                props = dict(props or {})
                if "label" not in props:
                    props["label"] = row.get("label") or ""
                d = Dictionary.ByKeysValues(list(props.keys()), list(props.values()))
                v = Topology.SetDictionary(v, d)
                id_to_vertex[row["id"]] = v
                vertices.append(v)

            # Read edges
            rows_e = manager.exec("""
                MATCH (a:Vertex)-[r:Edge]->(b:Vertex)
                WHERE a.graph_id = $gid AND b.graph_id = $gid
                RETURN a.id AS a_id, b.id AS b_id, r.label AS label, r.props AS props;
            """, {"gid": graphID}, write=False) or []
            edges = []
            for row in rows_e:
                va = id_to_vertex.get(row["a_id"])
                vb = id_to_vertex.get(row["b_id"])
                if not va or not vb:
                    continue
                e = Edge.ByStartVertexEndVertex(va, vb)
                props = {}
                try:
                    props = json.loads(row.get("props") or "{}")
                except Exception:
                    props = {}
                props = dict(props or {})
                if "label" not in props:
                    props["label"] = row.get("label") or "connect"
                d = Dictionary.ByKeysValues(list(props.keys()), list(props.values()))
                e = Topology.SetDictionary(e, d)
                edges.append(e)
            if len(vertices) > 0:
                g = Graph.ByVerticesEdges(vertices, edges)
                g = Topology.SetDictionary(g, g_dict)
            else:
                g = None
            return g
        except Exception as e:
            if not silent:
                print(f"Kuzu.GraphByID - Error: {e}. Returning None.")
            return None

    @staticmethod
    def GraphsByQuery(
        manager,
        query: str,
        params: dict = None,
        silent: bool = False,
    ):
        """
        Executes a Kùzu Cypher query and returns a list of TopologicPy Graphs.

        The method will:
        1) run the query,
        2) extract distinct graph IDs from the result set.
        3) reconstruct each graph via Kuzu.GraphByID(...).

        Parameters
        ----------
        manager : Kuzu.Manager
            The manager of the Kùzu database.
        query : str
            A valid Kùzu Cypher query.
        params : dict , optional
            Parameters to pass with the query.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list[topologic_core.Graph]
            A list of reconstructed TopologicPy graphs.
        
        """

        try:
            manager.ensure_schema()
            rows = manager.exec(query, params or {}, write=False) or []

            # Collect distinct graph IDs
            gids = []
            for r in rows:
                gid = r.get('graph_id')

                # Fallback: try to infer from common id fields like "<graph_id>:<i>"
                if gid is None:
                    for k in ("a_id", "b_id", "id"):
                        v = r.get(k)
                        if isinstance(v, str) and ":" in v:
                            gid = v.split(":", 1)[0]
                            break

                if gid and gid not in gids:
                    gids.append(gid)

            # Reconstruct each graph
            graphs = []
            for gid in gids:
                g = Kuzu.GraphByID(path, gid, silent=silent)
                if g is not None:
                    graphs.append(g)
            return graphs

        except Exception as e:
            if not silent:
                print(f"Kuzu.GraphsByQuery - Error: {e}. Returning None.")
            return None

    @staticmethod
    def DeleteGraph(manager, graphID: str, silent: bool = False) -> bool:
        """
        Deletes a graph (vertices, edges, and graphCard) by id.

        Parameters
        ----------
        manager : Kuzu.Manager
            The manager of the Kùzu database.
        graphID : str
            The id of the graph to be deleted.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
        Returns
        -------
        bool
            True on success, False otherwise.
        """
        try:
            manager.ensure_schema()
            # Delete edges
            manager.exec("""
                MATCH (a:Vertex)-[r:Edge]->(b:Vertex)
                WHERE a.graph_id = $gid AND b.graph_id = $gid
                DELETE r;
            """, {"gid": graphID}, write=True)
            # Delete vertices
            manager.exec("MATCH (v:Vertex) WHERE v.graph_id = $gid DELETE v;", {"gid": graphID}, write=True)
            # Delete card
            manager.exec("MATCH (g:Graph) WHERE g.id = $gid DELETE g;", {"gid": graphID}, write=True)
            return True
        except Exception as e:
            if not silent:
                print(f"Kuzu.DeleteGraph - Error: {e}. Returning False.")
            return False
    
    @staticmethod
    def EmptyDatabase(manager, dropSchema: bool = False, recreateSchema: bool = True, silent: bool = False) -> bool:
        """
        Empties the Kùzu database at `db_path`.

        Two modes:
        - Soft clear (default): delete ALL relationships, then ALL nodes across all tables.
        - Hard reset (drop_schema=True): drop known node/rel tables, optionally recreate schema.

        Parameters
        ----------
        manager : Kuzu Manager
            The manager of the Kùzu database.
        dropSchema : bool , optional
            If True, DROP the known tables instead of deleting rows. Default False.
        recreateSchema : bool , optional
            If True and drop_schema=True, re-create the minimal schema after dropping. Default True.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        bool
            True on success, False otherwise.
        """
        try:
            manager.ensure_schema()

            if dropSchema:
                # Drop relationship tables FIRST (to release dependencies), then node tables.
                # IF EXISTS is convenient; if your Kùzu version doesn't support it, remove and ignore exceptions.
                for stmt in [
                    "DROP TABLE IF EXISTS Edge;",
                    "DROP TABLE IF EXISTS Vertex;",
                    "DROP TABLE IF EXISTS Graph;",
                ]:
                    try:
                        manager.exec(stmt, write=True)
                    except Exception as _e:
                        if not silent:
                            print(f"Kuzu.EmptyDatabase - Warning dropping table: {_e}")

                if recreateSchema:
                    manager.ensure_schema()
                return True

            # Soft clear: remove all relationships, then all nodes (covers all labels/tables).
            # Delete all edges (any direction)
            manager.exec("MATCH (a)-[r]->(b) DELETE r;", write=True)
            # Delete all nodes (from all node tables)
            manager.exec("MATCH (n) DELETE n;", write=True)
            return True

        except Exception as e:
            if not silent:
                print(f"Kuzu.EmptyDatabase - Error: {e}. Returning False.")
            return False

    @staticmethod
    def ListGraphs(manager, where: dict = None, limit: int = 100, offset: int = 0, silent: bool = False) -> list[dict]:
        """
        Lists Graph metadata with simple filtering and pagination.

        Parameters
        ----------
        manager : Kuzu.Manager
            The manager of the Kùzu database.
        where : dict , optional
            The filter python dictionaries. Supported filters in `where` (all optional):
            - id (exact match)
            - label (substring match)
            - props_contains (substring match against JSON/text in `props`)
            - props_equals (exact string match against `props`)
            - min_nodes / max_nodes (integers)
            - min_edges / max_edges (integers)
        limit : int , optional
            The desired limit of returned Graphs. Default is 100.
        offset : int , optional
            The desired offset of the returned Graphs (skips the first number of Graphs specified by the offset and returns the remaining cards up to the specified limit). The offset is useful if pagination is needed. Default is 0.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The list of found Graph python dictionaries.

        """

        manager.ensure_schema()
        where = where or {}

        conds: list[str] = []
        params: dict = {}

        if "id" in where and where["id"]:
            conds.append("g.id = $id")
            params["id"] = str(where["id"])

        if "label" in where and where["label"]:
            # Cypher-style infix CONTAINS
            conds.append("g.label CONTAINS $label_sub")
            params["label_sub"] = str(where["label"])

        if "props_contains" in where and where["props_contains"]:
            conds.append("g.props CONTAINS $props_sub")
            params["props_sub"] = str(where["props_contains"])

        if "props_equals" in where and where["props_equals"]:
            conds.append("g.props = $props_equals")
            params["props_equals"] = str(where["props_equals"])

        if "min_nodes" in where and where["min_nodes"] is not None:
            conds.append("g.num_nodes >= $min_nodes")
            params["min_nodes"] = int(where["min_nodes"])

        if "max_nodes" in where and where["max_nodes"] is not None:
            conds.append("g.num_nodes <= $max_nodes")
            params["max_nodes"] = int(where["max_nodes"])

        if "min_edges" in where and where["min_edges"] is not None:
            conds.append("g.num_edges >= $min_edges")
            params["min_edges"] = int(where["min_edges"])

        if "max_edges" in where and where["max_edges"] is not None:
            conds.append("g.num_edges <= $max_edges")
            params["max_edges"] = int(where["max_edges"])

        where_clause = ("WHERE " + " AND ".join(conds)) if conds else ""
        q = f"""
            MATCH (g:Graph)
            {where_clause}
            RETURN g.id AS id, g.label AS label,
                g.num_nodes AS num_nodes, g.num_edges AS num_edges,
                g.props AS props
            ORDER BY id
            SKIP $__offset LIMIT $__limit;
        """
 
        params["__offset"] = max(0, int(offset or 0))
        params["__limit"] = max(0, int(limit or 100))

        return manager.exec(q, params, write=False) or []
    

    @staticmethod
    def ByCSVPath(
        manager,
        path: str,
        graphIDPrefix: str = "g",
        graphIDHeader="graph_id",
        graphLabelHeader="label",
        edgeSRCHeader="src_id",
        edgeDSTHeader="dst_id",
        edgeLabelHeader="label",
        nodeIDHeader="node_id",
        nodeLabelHeader="label",
        nodeXHeader="X",
        nodeYHeader="Y",
        nodeZHeader="Z",
        silent: bool = False,
    ) -> Dict[str, Any]:
        """
        Load node/edge/graph CSVs from a folder (using its .yaml meta) and upsert them
        directly into Kùzu using the schema defined in Kuzu.py:

        - NODE TABLE Graph(id STRING PRIMARY KEY, label STRING, num_nodes INT64, num_edges INT64, props STRING)
        - NODE TABLE Vertex(id STRING PRIMARY KEY, graph_id STRING, label STRING, x DOUBLE, y DOUBLE, z DOUBLE, props STRING)
        - REL  TABLE Edge(FROM Vertex TO Vertex, label STRING, props STRING)

        Parameters
        ----------
        manager : Kuzu.Manager
            An initialized Kùzu manager; must provide ensure_schema() and exec(query, params, write=True/False).
        path : str
            Folder containing a dataset YAML (e.g., meta.yaml) that points to nodes/edges/graphs CSVs.
        graphIDPrefix : str
            Prefix for materialized graph IDs (default "g"); e.g., graph 0 -> "g0".
        graphIDHeader : str , optional
            The column header string used to specify the graph id. Default is "graph_id".
        graphLabelHeader : str , optional
            The column header string used to specify the graph label. Default is "label".
        edgeSRCHeader : str , optional
            The column header string used to specify the source vertex id of edges. Default is "src_id".
        edgeDSTHeader : str , optional
            The column header string used to specify the destination vertex id of edges. Default is "dst_id".
        edgeLabelHeader : str , optional
            The column header string used to specify the label of edges. Default is "label".
        nodeIDHeader : str , optional
            The column header string used to specify the id of nodes. Default is "node_id".
        nodeLabelHeader : str , optional
            The column header string used to specify the label of nodes. Default is "label".
        nodeXHeader : str , optional
            The column header string used to specify the X coordinate of nodes. Default is "X".
        nodeYHeader : str , optional
            The column header string used to specify the Y coordinate of nodes. Default is "Y".
        nodeZHeader : str , optional
            The column header string used to specify the Z coordinate of nodes. Default is "Z".
        silent : bool
            If True, suppress warnings.

        Returns
        -------
        dict
            {"graphs_upserted": int, "graph_ids": [str, ...]}
        """
        import os
        import glob
        import json
        import numbers
        import pandas as pd
        import yaml
        import random

        # ---------- Helpers (mirroring your CSV loader’s patterns) ----------
        def _find_yaml_files(folder_path: str):
            return glob.glob(os.path.join(folder_path, "*.yaml"))

        def _read_yaml(file_path: str):
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            edge_data = data.get("edge_data", [])
            node_data = data.get("node_data", [])
            graph_data = data.get("graph_data", {})
            edges_rel = edge_data[0].get("file_name") if edge_data else None
            nodes_rel = node_data[0].get("file_name") if node_data else None
            graphs_rel = graph_data.get("file_name")
            return graphs_rel, edges_rel, nodes_rel

        def _props_from_row(row: pd.Series, exclude: set) -> str:
            d = {}
            for k, v in row.items():
                if k in exclude:
                    continue
                # normalize NaN -> None for clean JSON
                if isinstance(v, float) and pd.isna(v):
                    d[k] = None
                else:
                    d[k] = v
            try:
                return json.dumps(d, ensure_ascii=False)
            except Exception:
                # Fallback: stringify everything
                return json.dumps({k: (None if v is None else str(v)) for k, v in d.items()}, ensure_ascii=False)

        # ---------- Validate path and locate YAML/CSVs ----------
        if not os.path.exists(path) or not os.path.isdir(path):
            if not silent:
                print("ByCSVPath - Error: path must be an existing folder. Returning None.")
            return None

        yaml_files = _find_yaml_files(path)
        if len(yaml_files) < 1:
            if not silent:
                print("ByCSVPath - Error: no YAML file found in the folder. Returning None.")
            return None
        yaml_file = yaml_files[0]
        graphs_rel, edges_rel, nodes_rel = _read_yaml(yaml_file)

        # Resolve CSV paths
        graphs_csv = os.path.join(path, graphs_rel) if graphs_rel else None
        edges_csv = os.path.join(path, edges_rel) if edges_rel else None
        nodes_csv = os.path.join(path, nodes_rel) if nodes_rel else None

        if not edges_csv or not os.path.exists(edges_csv):
            if not silent:
                print("ByCSVPath - Error: edges CSV not found. Returning None.")
            return None
        if not nodes_csv or not os.path.exists(nodes_csv):
            if not silent:
                print("ByCSVPath - Error: nodes CSV not found. Returning None.")
            return None

        # ---------- Load CSVs ----------
        nodes_df = pd.read_csv(nodes_csv)
        edges_df = pd.read_csv(edges_csv)
        graphs_df = pd.read_csv(graphs_csv) if graphs_csv and os.path.exists(graphs_csv) else pd.DataFrame()

        # Required columns
        for req_cols, df_name, df in [
            ({graphIDHeader, nodeIDHeader}, "nodes", nodes_df),
            ({graphIDHeader, edgeSRCHeader, edgeDSTHeader}, "edges", edges_df),
        ]:
            missing = req_cols.difference(df.columns)
            if missing:
                raise ValueError(f"ByCSVPath - {df_name}.csv is missing required columns: {missing}")

        # Graph IDs present in the data
        gids = pd.Index([]).union(nodes_df[graphIDHeader].dropna().unique()).union(
            edges_df[graphIDHeader].dropna().unique()
        )

        # Prepare graphs_df lookup if provided
        graphs_by_gid = {}
        if graphIDHeader in graphs_df.columns:
            graphs_by_gid = {gid: g.iloc[0].to_dict() for gid, g in graphs_df.groupby(graphIDHeader, dropna=False)}

        # ---------- Ensure schema ----------
        manager.ensure_schema()  # Graph, Vertex, Edge

        # ---------- Upsert per graph ----------
        materialized_graph_ids = []
        for raw_gid in gids:
            gid_str = f"{graphIDPrefix}{int(raw_gid) if str(raw_gid).isdigit() else str(raw_gid)}"
            materialized_graph_ids.append(gid_str)

            nsub = nodes_df[nodes_df[graphIDHeader] == raw_gid].copy()
            esub = edges_df[edges_df[graphIDHeader] == raw_gid].copy()

            # Graph info
            gcard_src = graphs_by_gid.get(raw_gid, {})
            g_label = str(gcard_src.get(graphLabelHeader, "")) if gcard_src else ""
            g_props = _props_from_row(pd.Series(gcard_src), exclude={graphIDHeader, graphLabelHeader}) if gcard_src else "{}"
            num_nodes = int(nsub.shape[0])
            num_edges = int(esub.shape[0])

            # Remove any existing data for this graph id, then re-insert
            manager.exec("""
                MATCH (a:Vertex)-[r:Edge]->(b:Vertex)
                WHERE a.graph_id = $gid AND b.graph_id = $gid
                DELETE r;
            """, {"gid": gid_str}, write=True)
            manager.exec("MATCH (v:Vertex) WHERE v.graph_id = $gid DELETE v;", {"gid": gid_str}, write=True)
            manager.exec("MATCH (g:Graph) WHERE g.id = $gid DELETE g;", {"gid": gid_str}, write=True)

            manager.exec("""
                CREATE (g:Graph {id:$id, label:$label, num_nodes:$num_nodes, num_edges:$num_edges, props:$props});
            """, {
                "id": gid_str,
                "label": g_label,
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "props": g_props,
            }, write=True)

            # Insert vertices
            for _, row in nsub.iterrows():
                node_id = row[nodeIDHeader]
                vid = f"{gid_str}:{node_id}"
                v_label = str(row[nodeLabelHeader]) if "label" in row and pd.notna(row[nodeLabelHeader]) else str(node_id)

                # X/Y/Z may be missing or non-numeric; store a random numeric value in that case
                def _num_or_none(val):
                    try:
                        return float(val)
                    except Exception:
                        return None

                x = _num_or_none(row[nodeXHeader]) if nodeXHeader in row else random.uniform(0,1000)
                y = _num_or_none(row[nodeYHeader]) if nodeYHeader in row else random.uniform(0,1000)
                z = _num_or_none(row[nodeZHeader]) if nodeZHeader in row else random.uniform(0,1000)

                props = _props_from_row(row, exclude={graphIDHeader, nodeIDHeader, nodeLabelHeader, nodeXHeader, nodeYHeader, nodeZHeader})
                manager.exec("""
                    CREATE (v:Vertex {id:$id, graph_id:$gid, label:$label, x:$x, y:$y, z:$z, props:$props});
                """, {"id": vid, "gid": gid_str, "label": v_label, "x": x, "y": y, "z": z, "props": props}, write=True)

            # Insert edges (Edge)
            for _, row in esub.iterrows():
                a_id = f"{gid_str}:{row[edgeSRCHeader]}"
                b_id = f"{gid_str}:{row[edgeDSTHeader]}"
                e_label = str(row[edgeLabelHeader]) if edgeLabelHeader in row and pd.notna(row[edgeLabelHeader]) else "connect"
                e_props = _props_from_row(row, exclude={graphIDHeader, edgeSRCHeader, edgeDSTHeader, edgeLabelHeader})

                manager.exec("""
                    MATCH (a:Vertex {id:$a_id}), (b:Vertex {id:$b_id})
                    CREATE (a)-[:Edge {label:$label, props:$props}]->(b);
                """, {"a_id": a_id, "b_id": b_id, "label": e_label, "props": e_props}, write=True)

        return {"graphs_upserted": len(materialized_graph_ids), "graph_ids": materialized_graph_ids}



