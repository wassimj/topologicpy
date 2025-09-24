from __future__ import annotations
import threading, contextlib, time, json
from typing import Dict, Any, List, Optional


# Optional TopologicPy imports (make this file safe to import without TopologicPy)
from topologicpy.Graph import Graph
from topologicpy.Vertex import Vertex
from topologicpy.Edge import Edge
from topologicpy.Dictionary import Dictionary
from topologicpy.Topology import Topology

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
        CREATE NODE TABLE IF NOT EXISTS GraphCard(
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
        CREATE REL TABLE IF NOT EXISTS CONNECT(FROM Vertex TO Vertex, label STRING, props STRING);
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
    """
    TopologicPy-style class of static methods for Kùzu integration.

    Notes
    -----
    - All methods are *static* to match TopologicPy's style.
    - Graph persistence:
        * Vertices: stored in `Vertex` with (id, graph_id, label, props JSON)
        * Edges: stored as `CONNECT` relations a->b with label + props JSON
        * We assume undirected design intent; only one CONNECT is stored (a->b),
          but TopologicPy Graph treats edges as undirected by default.
    """

    # ---------- Core (DB + Connection + Schema) ----------
    @staticmethod
    def EnsureSchema(db_path: str, silent: bool = False) -> bool:
        """
        Ensures the required Kùzu schema exists in the database at `db_path`.

        Parameters
        ----------
        db_path : str
            Path to the Kùzu database directory. It will be created if it does not exist.
        silent : bool , optional
            If True, suppresses error messages. Default is False.

        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        try:
            mgr = _Mgr(db_path)
            mgr.ensure_schema()
            return True
        except Exception as e:
            if not silent:
                print(f"Kuzu.EnsureSchema - Error: {e}")
            return False

    @staticmethod
    def Database(db_path: str):
        """
        Returns the underlying `kuzu.Database` instance for `db_path`.
        """
        return _db_cache.get(db_path)

    @staticmethod
    def Connection(db_path: str):
        """
        Returns a `kuzu.Connection` bound to the database at `db_path`.
        """
        mgr = _Mgr(db_path)
        with mgr.read() as c:
            return c  # Note: returns a live connection (do not use across threads)

    # ---------- Graph <-> DB Conversion ----------

    @staticmethod
    def UpsertGraph(db_path: str,
                    graph,
                    graphIDKey: Optional[str] = None,
                    vertexIDKey: Optional[str] = None,
                    vertexLabelKey: Optional[str] = None,
                    mantissa: int = 6,
                    silent: bool = False) -> str:
        """
        Upserts (deletes prior + inserts new) a TopologicPy graph and its GraphCard.

        Parameters
        ----------
        db_path : str
            Kùzu database path.
        graph : topologicpy.Graph
            The input TopologicPy graph.
        graphIDKey : str , optional
            The graph dictionary key under which the graph ID is stored. If None, a UUID is generated.
        title, domain, geo, time_start, time_end, summary : str , optional
            Optional metadata for GraphCard.
        silent : bool , optional
            If True, suppresses error messages. Default is False.

        Returns
        -------
        str
            The graph_id used.
        """
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
        mgr = _Mgr(db_path)
        try:
            mgr.ensure_schema()
            # Upsert GraphCard
            mgr.exec("MATCH (g:GraphCard) WHERE g.id = $id DELETE g;", {"id": gid}, write=True)
            mgr.exec("""
                CREATE (g:GraphCard {id:$id, num_nodes:$num_nodes, num_edges: $num_edges, props:$props});
            """, {"id": gid, "num_nodes": num_nodes, "num_edges": num_edges, "props": json.dumps(g_props)}, write=True)

            # Remove existing vertices/edges for this graph_id
            mgr.exec("""
                MATCH (a:Vertex)-[r:CONNECT]->(b:Vertex)
                WHERE a.graph_id = $gid AND b.graph_id = $gid
                DELETE r;
            """, {"gid": gid}, write=True)
            mgr.exec("MATCH (v:Vertex) WHERE v.graph_id = $gid DELETE v;", {"gid": gid}, write=True)

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
                mgr.exec("""
                    CREATE (v:Vertex {id:$id, graph_id:$gid, label:$label, props:$props, x:$x, y:$y, z:$z});
                """, {"id": vid, "gid": gid, "label": label, "x": x, "y": y, "z": z,
                        "props": json.dumps(v_props[i])}, write=True)

            # Insert edges
            for i, e in enumerate(edges):
                a_id = v_props[e[0]].get(vertexIDKey, f"{gid}:{e[0]}")
                b_id = v_props[e[1]].get(vertexIDKey, f"{gid}:{e[1]}")
                mgr.exec("""
                    MATCH (a:Vertex {id:$a}), (b:Vertex {id:$b})
                    CREATE (a)-[:CONNECT {label:$label, props:$props}]->(b);
                """, {"a": a_id, "b": b_id,
                        "label": e_props[i].get("label", str(i)),
                        "props": json.dumps(e_props[i])}, write=True)

            return gid
        except Exception as e:
            if not silent:
                print(f"Kuzu.UpsertGraph - Error: {e}")
            raise

    @staticmethod
    def GraphByID(db_path: str, graphID: str, silent: bool = False):
        """
        Reads a graph with id `graph_id` from Kùzu and constructs a TopologicPy graph.

        Returns
        -------
        topologicpy.Graph
            A new TopologicPy Graph, or None on error.
        """
        # if TGraph is None:
        #     raise _KuzuError("TopologicPy is required to use Kuzu.ReadTopologicGraph.")
        import random
        mgr = _Mgr(db_path)

        try:
            mgr.ensure_schema()
            # Read the GraphCard
            g = mgr.exec("""
                    MATCH (g:GraphCard) WHERE g.id = $id
                    RETURN g.id AS id, g.num_nodes AS num_nodes, g.num_edges AS num_edges, g.props AS props
                    ;
                     """, {"id": graphID}, write=False) or None
            if g is None:
                return None
            g = g[0]
            g_dict = dict(json.loads(g.get("props") or "{}") or {})
            g_dict = Dictionary.ByPythonDictionary(g_dict)
            # Read vertices
            rows_v = mgr.exec("""
                MATCH (v:Vertex) WHERE v.graph_id = $gid
                RETURN v.id AS id, v.label AS label, v.x AS x, v.y AS y, v.z AS z, v.props AS props
                ORDER BY id;
            """, {"gid": graphID}, write=False) or []

            id_to_vertex = {}
            vertices = []
            for row in rows_v:
                try:
                    x = row.get("x") or random.uniform(0,1000)
                    y = row.get("y") or random.uniform(0,1000)
                    z = row.get("z") or random.uniform(0,1000)
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
            rows_e = mgr.exec("""
                MATCH (a:Vertex)-[r:CONNECT]->(b:Vertex)
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
                print(f"Kuzu.GraphByID - Error: {e}")
            return None

    @staticmethod
    def GraphsByQuery(
        db_path: str,
        query: str,
        params: dict | None = None,
        graphIDKey: str = "graph_id",
        silent: bool = False,
    ):
        """
        Executes a Kùzu Cypher query and returns a list of TopologicPy Graphs.
        The query should return at least one column identifying each graph.
        By default this column is expected to be named 'graph_id', but you can
        override that via `graph_id_field`.

        The method will:
        1) run the query,
        2) extract distinct graph IDs from the result set (using `graph_id_field`
            if present; otherwise it attempts to infer IDs from common fields like
            'a_id', 'b_id', or 'id' that look like '<graph_id>:<vertex_index>'),
        3) reconstruct each graph via Kuzu.ReadTopologicGraph(...).

        Parameters
        ----------
        db_path : str
            Path to the Kùzu database directory.
        query : str
            A valid Kùzu Cypher query.
        params : dict , optional
            Parameters to pass with the query.
        graph_id_field : str , optional
            The field name in the query result that contains the graph ID(s).
            Default is "graph_id".
        silent : bool , optional
            If True, suppresses errors and returns an empty list on failure.

        Returns
        -------
        list[topologicpy.Graph]
            A list of reconstructed TopologicPy graphs.
        """
        # if TGraph is None:
        #     raise _KuzuError("TopologicPy is required to use Kuzu.GraphsFromQuery.")

        try:
            mgr = _Mgr(db_path)
            mgr.ensure_schema()
            rows = mgr.exec(query, params or {}, write=False) or []

            # Collect distinct graph IDs
            gids = []
            for r in rows:
                gid = r.get(graphIDKey)

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
                g = Kuzu.GraphByID(db_path, gid, silent=True)
                if g is not None:
                    graphs.append(g)
            return graphs

        except Exception as e:
            if not silent:
                print(f"Kuzu.GraphsByQuery - Error: {e}")
            return []

    @staticmethod
    def DeleteGraph(db_path: str, graph_id: str, silent: bool = False) -> bool:
        """
        Deletes a graph (vertices/edges) and its GraphCard by id.
        """
        try:
            mgr = _Mgr(db_path)
            mgr.ensure_schema()
            # Delete edges
            mgr.exec("""
                MATCH (a:Vertex)-[r:CONNECT]->(b:Vertex)
                WHERE a.graph_id = $gid AND b.graph_id = $gid
                DELETE r;
            """, {"gid": graph_id}, write=True)
            # Delete vertices
            mgr.exec("MATCH (v:Vertex) WHERE v.graph_id = $gid DELETE v;", {"gid": graph_id}, write=True)
            # Delete card
            mgr.exec("MATCH (g:GraphCard) WHERE g.id = $gid DELETE g;", {"gid": graph_id}, write=True)
            return True
        except Exception as e:
            if not silent:
                print(f"Kuzu.DeleteGraph - Error: {e}")
            return False
    
    @staticmethod
    def EmptyDatabase(db_path: str, drop_schema: bool = False, recreate_schema: bool = True, silent: bool = False) -> bool:
        """
        Empties the Kùzu database at `db_path`.

        Two modes:
        - Soft clear (default): delete ALL relationships, then ALL nodes across all tables.
        - Hard reset (drop_schema=True): drop known node/rel tables, optionally recreate schema.

        Parameters
        ----------
        db_path : str
            Path to the Kùzu database directory.
        drop_schema : bool , optional
            If True, DROP the known tables instead of deleting rows. Default False.
        recreate_schema : bool , optional
            If True and drop_schema=True, re-create the minimal schema after dropping. Default True.
        silent : bool , optional
            Suppress errors if True. Default False.

        Returns
        -------
        bool
            True on success, False otherwise.
        """
        try:
            mgr = _Mgr(db_path)
            # Ensure DB exists (does not create tables unless needed)
            mgr.ensure_schema()

            if drop_schema:
                # Drop relationship tables FIRST (to release dependencies), then node tables.
                # IF EXISTS is convenient; if your Kùzu version doesn't support it, remove and ignore exceptions.
                for stmt in [
                    "DROP TABLE IF EXISTS CONNECT;",
                    "DROP TABLE IF EXISTS Vertex;",
                    "DROP TABLE IF EXISTS GraphCard;",
                ]:
                    try:
                        mgr.exec(stmt, write=True)
                    except Exception as _e:
                        if not silent:
                            print(f"Kuzu.EmptyDatabase - Warning dropping table: {_e}")

                if recreate_schema:
                    mgr.ensure_schema()
                return True

            # Soft clear: remove all relationships, then all nodes (covers all labels/tables).
            # Delete all edges (any direction)
            mgr.exec("MATCH (a)-[r]->(b) DELETE r;", write=True)
            # Delete all nodes (from all node tables)
            mgr.exec("MATCH (n) DELETE n;", write=True)
            return True

        except Exception as e:
            if not silent:
                print(f"Kuzu.EmptyDatabase - Error: {e}")
            return False

