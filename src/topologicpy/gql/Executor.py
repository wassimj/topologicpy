# -*- coding: utf-8 -*-

"""
Minimal GQL-like executor for TopologicPy.

This executor uses a dictionary-backed GQL working graph internally. The first
query against a TopologicPy Graph converts it into a working graph containing
Python lists of vertices and edges plus index maps. Subsequent GQL operations
operate on that working graph and avoid rebuilding or traversing the TopologicPy
Graph until Executor.TopologicGraph(...) is explicitly requested.

Phase 1:
- MATCH one node-edge-node pattern
- ontology-prefixed labels such as top:Room and top:adjacentTo
- WHERE with AND / OR / parentheses
- RETURN / RETURN *
- RETURN DISTINCT
- COUNT(...)
- RETURN aliases with AS
- ORDER BY, SKIP, LIMIT

Phase 2:
- CREATE node or node-edge-node patterns
- MERGE node or node-edge-node patterns
- MATCH ... SET property assignments
- MATCH ... DELETE bound variables

Read queries return list[dict]. Mutation queries return:
    {
        "graph": updated_graph,
        "rows": projected_rows,
        "action": "CREATE" | "MERGE" | "SET" | "DELETE",
        "created": int,
        "matched": int,
        "updated": int,
        "deleted": int,
    }

For mutation queries, "graph" is normally a GQL working graph. Convert it back
to a TopologicPy Graph only when needed using Executor.TopologicGraph(graph).
"""

from typing import Any, Dict, List, Tuple

try:
    from topologicpy.gql.Parser import AggregateExpression, ReturnClause, ReturnItem
except Exception:
    AggregateExpression = None
    ReturnClause = None
    ReturnItem = None


class Executor:
    """Minimal GQL-like executor for TopologicPy Graph objects."""

    _GRAPH_CACHE = {}

    @staticmethod
    def _is_working_graph(graph):
        return isinstance(graph, dict) and graph.get("type") == "GQLWorkingGraph"

    @staticmethod
    def _graph_cache_key(graph):
        return id(graph)

    @staticmethod
    def _ensure_working_graph(graph, silent: bool = False):
        """Returns a dictionary-backed GQL working graph.

        If graph is already a GQL working graph, it is returned unchanged. If it
        is a TopologicPy Graph, its vertices and edges are read once and cached
        as a Python-level working graph. All subsequent GQL operations should use
        the working graph to avoid expensive TopologicPy graph calls.
        """

        if Executor._is_working_graph(graph):
            return graph

        if graph is None:
            return None

        key = Executor._graph_cache_key(graph)
        cache = Executor._GRAPH_CACHE.setdefault(key, {})

        working = cache.get("working_graph")
        if Executor._is_working_graph(working):
            return working

        try:
            from topologicpy.Graph import Graph
            vertices = Graph.Vertices(graph)
            edges = Graph.Edges(graph)
        except Exception as e:
            if not silent:
                print("GQL.Executor._ensure_working_graph - Error: Could not read TopologicPy graph. Returning None.")
                print("Error:", e)
            return None

        vertices = vertices if isinstance(vertices, list) else []
        edges = edges if isinstance(edges, list) else []

        working = {
            "type": "GQLWorkingGraph",
            "vertices": vertices,
            "edges": edges,
            "vertex_by_index": {},
            "source_graph": graph,
            "topologic_graph": graph,
            "dirty": False,
        }
        Executor._rebuild_working_indexes(working)
        cache["working_graph"] = working
        return working

    @staticmethod
    def _rebuild_working_indexes(working_graph):
        """Rebuilds the index -> vertex lookup for a GQL working graph."""

        if not Executor._is_working_graph(working_graph):
            return working_graph

        vertex_by_index = {}
        vertices = working_graph.get("vertices", []) or []

        for fallback_index, vertex in enumerate(vertices):
            value = Executor._dictionary_value(vertex, "index", None)
            normalised = Executor._normalise_index(value)

            if normalised is not None:
                vertex_by_index[normalised] = vertex
                vertex_by_index[str(normalised)] = vertex

            # Conservative fallback for legacy or newly created vertices.
            vertex_by_index.setdefault(fallback_index, vertex)
            vertex_by_index.setdefault(str(fallback_index), vertex)

        working_graph["vertex_by_index"] = vertex_by_index
        return working_graph

    @staticmethod
    def TopologicGraph(graph, ontology: bool = True, silent: bool = False):
        """
        Returns a TopologicPy Graph from a TopologicPy graph or GQL working graph.

        Parameters
        ----------
        graph : topologic_core.Graph or dict
            A TopologicPy graph or a GQL working graph returned by GQL.Query / GQL.Mutate.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Graph or None
            The corresponding TopologicPy graph.
        """

        if graph is None:
            return None

        if not Executor._is_working_graph(graph):
            return graph

        if graph.get("dirty") is False and graph.get("topologic_graph") is not None:
            return graph.get("topologic_graph")

        vertices = graph.get("vertices", []) or []
        edges = graph.get("edges", []) or []

        updated = Executor._graph_by_vertices_edges(vertices, edges, ontology=ontology, silent=silent)
        if updated is not None:
            graph["topologic_graph"] = updated
            graph["dirty"] = False

        return updated

    @staticmethod
    def _graph_edges(graph):
        working = Executor._ensure_working_graph(graph)
        if Executor._is_working_graph(working):
            edges = working.get("edges", [])
            return edges if isinstance(edges, list) else []
        return []

    @staticmethod
    def _graph_vertices(graph):
        working = Executor._ensure_working_graph(graph)
        if Executor._is_working_graph(working):
            vertices = working.get("vertices", [])
            return vertices if isinstance(vertices, list) else []
        return []

    @staticmethod
    def _normalise_index(value):
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            text = value.strip()
            if text:
                try:
                    number = float(text)
                    if number.is_integer():
                        return int(number)
                except Exception:
                    pass
                return text
        return value

    @staticmethod
    def _ontology_term(value, default_prefix: str = "top"):
        """Returns a normalised ontology term, e.g. ``top:Room``."""
        if value is None:
            return None
        text = str(value).strip()
        if text == "":
            return None
        if ":" in text:
            prefix, local = text.split(":", 1)
            prefix = prefix.strip() or default_prefix
            local = local.strip()
            if local == "":
                return None
            return f"{prefix}:{local}"
        return text

    @staticmethod
    def _ontology_local_name(value):
        """Returns the local part of an ontology term."""
        term = Executor._ontology_term(value)
        if term is None:
            return None
        if ":" in term:
            return term.split(":", 1)[1]
        return term

    @staticmethod
    def _ontology_match(candidate, query) -> bool:
        """Ontology-aware string comparison.

        A query for ``top:Room`` matches ``top:Room`` and also the local label
        ``Room``. A query for ``Room`` matches ``Room`` and ``top:Room``.
        """
        if query is None:
            return True
        if candidate is None:
            return False
        q = str(query).strip()
        c = str(candidate).strip()
        if q == "" or c == "":
            return False
        if c == q:
            return True
        return str(Executor._ontology_local_name(c)).lower() == str(Executor._ontology_local_name(q)).lower()

    @staticmethod
    def _ontology_category_from_class(ontology_class, fallback=None):
        """Best-effort category lookup using Ontology.py, with safe fallback."""
        if ontology_class is None:
            return fallback
        try:
            from topologicpy.Ontology import Ontology
            if hasattr(Ontology, "CategoryByClass"):
                value = Ontology.CategoryByClass(ontology_class, silent=True)
                if value is not None:
                    return value
        except Exception:
            pass
        local = Executor._ontology_local_name(ontology_class)
        if local is None:
            return fallback
        low = str(local).lower()
        if low in ("vertex", "node"):
            return "node"
        if low in ("edge", "relationship", "adjacentto", "connectsto", "connects"):
            return "relationship"
        if low in ("room", "space", "storey", "building", "site"):
            return "space"
        return fallback

    @staticmethod
    def _ontology_annotate_pattern_data(data: Dict[str, Any], pattern, is_edge: bool = False):
        """Adds ontology keys to pattern-created topology dictionaries."""
        label = getattr(pattern, "label", None)
        if label is None:
            return data

        label_text = str(label).strip()
        if label_text == "":
            return data

        if ":" in label_text:
            ontology_class = Executor._ontology_term(label_text)
            local = Executor._ontology_local_name(ontology_class)
            data["ontology_class"] = ontology_class
            # Keep a clean local label for legacy graph/GQL workflows.
            data["label"] = local
            category = Executor._ontology_category_from_class(
                ontology_class,
                fallback=("relationship" if is_edge else "node")
            )
            if category is not None:
                data.setdefault("category", category)
        else:
            data["label"] = label_text
            data.setdefault("ontology_class", f"top:{label_text}" if label_text[:1].isupper() else None)
            if data.get("ontology_class") is None:
                data.pop("ontology_class", None)

        return data

    @staticmethod
    def _vertex_by_index_map(graph):
        working = Executor._ensure_working_graph(graph)
        if Executor._is_working_graph(working):
            if not isinstance(working.get("vertex_by_index"), dict):
                Executor._rebuild_working_indexes(working)
            return working.get("vertex_by_index", {})
        return {}

    @staticmethod
    def _edge_endpoint_vertices(edge, vertex_by_index=None, vertices=None):
        vertex_by_index = vertex_by_index or {}

        src = Executor._normalise_index(Executor._dictionary_value(edge, "src", None))
        dst = Executor._normalise_index(Executor._dictionary_value(edge, "dst", None))

        src_vertex = vertex_by_index.get(src)
        dst_vertex = vertex_by_index.get(dst)

        if src_vertex is not None and dst_vertex is not None:
            return src_vertex, dst_vertex

        # Fallback for older graphs that do not yet carry src/dst dictionaries.
        try:
            from topologicpy.Edge import Edge
            return Edge.StartVertex(edge), Edge.EndVertex(edge)
        except Exception:
            return None, None

    @staticmethod
    def _next_vertex_index(graph):
        vertices = Executor._graph_vertices(graph)
        max_index = -1
        for fallback_index, vertex in enumerate(vertices):
            value = Executor._normalise_index(Executor._dictionary_value(vertex, "index", None))
            if isinstance(value, int) and value > max_index:
                max_index = value
            elif value is None and fallback_index > max_index:
                max_index = fallback_index
        return max_index + 1

    @staticmethod
    def ClearCache(graph=None):
        if graph is None:
            Executor._GRAPH_CACHE.clear()
            return

        if Executor._is_working_graph(graph):
            source_graph = graph.get("source_graph")
            if source_graph is not None:
                Executor._GRAPH_CACHE.pop(id(source_graph), None)
            return

        Executor._GRAPH_CACHE.pop(id(graph), None)

    @staticmethod
    def Execute(graph, ast, silent: bool = False):
        """Executes a parsed GQL-like AST against a TopologicPy or GQL working graph."""

        if graph is None:
            if not silent:
                print("GQL.Executor.Execute - Error: The input graph is None. Returning None.")
            return None

        if ast is None:
            if not silent:
                print("GQL.Executor.Execute - Error: The input AST is None. Returning None.")
            return None

        working_graph = Executor._ensure_working_graph(graph, silent=silent)
        if working_graph is None:
            return None

        try:
            query_type = str(getattr(ast, "query_type", "MATCH")).upper()

            if query_type == "MATCH":
                return Executor._execute_match(working_graph, ast, silent=silent)
            if query_type == "CREATE":
                return Executor._execute_create(working_graph, ast, silent=silent)
            if query_type == "MERGE":
                return Executor._execute_merge(working_graph, ast, silent=silent)
            if query_type == "SET":
                return Executor._execute_set(working_graph, ast, silent=silent)
            if query_type == "DELETE":
                return Executor._execute_delete(working_graph, ast, silent=silent)

            if not silent:
                print(f"GQL.Executor.Execute - Error: Unsupported query type '{query_type}'. Returning None.")
            return None

        except Exception as e:
            if not silent:
                print("GQL.Executor.Execute - Error: Could not execute query. Returning None.")
                print("Error:", e)
            return None

    @staticmethod
    def _execute_match(graph, ast, silent: bool = False):
        fast_count = Executor._fast_count_match(graph, ast, silent=False)

        if fast_count is not None:
            return fast_count

        rows = Executor._match(graph, ast.match, silent=silent)

        if ast.where is not None:
            rows = [
                row for row in rows
                if Executor._evaluate_where_clause(row, ast.where, silent=silent)
            ]

        order_by = getattr(ast, "order_by", None)
        returns = getattr(ast, "returns", None)

        if Executor._has_aggregate_return(returns):
            rows = Executor._aggregate(rows, returns)
        else:
            rows = Executor._project(rows, returns)

            if order_by is not None:
                rows = Executor._order_by(rows, order_by)

            if Executor._is_distinct_return(returns):
                rows = Executor._distinct(rows)

        skip = getattr(ast, "skip", None)
        if isinstance(skip, int) and skip > 0:
            rows = rows[skip:]

        limit = getattr(ast, "limit", None)
        if isinstance(limit, int) and limit >= 0:
            rows = rows[:limit]

        return rows

    @staticmethod
    def _execute_create(graph, ast, silent: bool = False):
        updated_graph, row, created = Executor._create_pattern(graph, ast.create, silent=silent)
        rows = [row] if row else []

        if ast.returns is not None:
            rows = Executor._project(rows, ast.returns)

        return {
            "graph": updated_graph,
            "rows": rows,
            "action": "CREATE",
            "created": created,
            "matched": 0,
            "updated": 0,
            "deleted": 0,
        }

    @staticmethod
    def _execute_merge(graph, ast, silent: bool = False):
        pattern = ast.merge
        rows = Executor._match_pattern_or_node(graph, pattern, silent=silent)

        if rows:
            projected = Executor._project(rows, ast.returns) if ast.returns is not None else rows
            return {
                "graph": graph,
                "rows": projected,
                "action": "MERGE",
                "created": 0,
                "matched": len(rows),
                "updated": 0,
                "deleted": 0,
            }

        updated_graph, row, created = Executor._create_pattern(graph, pattern, silent=silent)
        rows = [row] if row else []
        if ast.returns is not None:
            rows = Executor._project(rows, ast.returns)

        return {
            "graph": updated_graph,
            "rows": rows,
            "action": "MERGE",
            "created": created,
            "matched": 0,
            "updated": 0,
            "deleted": 0,
        }

    @staticmethod
    def _execute_set(graph, ast, silent: bool = False):
        rows = Executor._rows_for_match(graph, ast, silent=silent)
        updated_count = 0

        for row in rows:
            for item in getattr(ast, "set_items", []) or []:
                obj = row.get(item.variable)
                if obj is None:
                    continue
                Executor._set_dictionary_values(obj, {item.property: item.value})
                updated_count += 1

        working = Executor._ensure_working_graph(graph, silent=silent)
        if Executor._is_working_graph(working) and updated_count > 0:
            working["dirty"] = True
            working["topologic_graph"] = None

        output_rows = Executor._finalize_rows(rows, ast) if ast.returns is not None else rows

        return {
            "graph": graph,
            "rows": output_rows,
            "action": "SET",
            "created": 0,
            "matched": len(rows),
            "updated": updated_count,
            "deleted": 0,
        }

    @staticmethod
    def _execute_delete(graph, ast, silent: bool = False):
        rows = Executor._rows_for_match(graph, ast, silent=silent)
        delete_names = set(getattr(ast, "delete_items", []) or [])

        edges_to_delete = []
        vertices_to_delete = []

        for row in rows:
            for name in delete_names:
                obj = row.get(name)
                if obj is None:
                    continue
                if Executor._is_edge(obj):
                    edges_to_delete.append(obj)
                elif Executor._is_vertex(obj):
                    vertices_to_delete.append(obj)

        edges_to_delete = Executor._unique_topologies(edges_to_delete)
        vertices_to_delete = Executor._unique_topologies(vertices_to_delete)

        working = Executor._ensure_working_graph(graph, silent=silent)
        if working is None:
            return {
                "graph": graph,
                "rows": [],
                "action": "DELETE",
                "created": 0,
                "matched": len(rows),
                "updated": 0,
                "deleted": 0,
            }

        old_vertices = working.get("vertices", []) or []
        old_edges = working.get("edges", []) or []
        vertex_by_index = Executor._vertex_by_index_map(working)

        kept_vertices = []
        removed_vertex_count = 0
        for vertex in old_vertices:
            if Executor._contains_topology(vertices_to_delete, vertex):
                removed_vertex_count += 1
            else:
                kept_vertices.append(vertex)

        kept_edges = []
        removed_edge_count = 0
        for edge in old_edges:
            remove_edge = Executor._contains_topology(edges_to_delete, edge)
            if not remove_edge:
                sv, ev = Executor._edge_endpoint_vertices(edge, vertex_by_index, old_vertices)
                if (
                    Executor._contains_topology(vertices_to_delete, sv)
                    or Executor._contains_topology(vertices_to_delete, ev)
                ):
                    remove_edge = True

            if remove_edge:
                removed_edge_count += 1
            else:
                kept_edges.append(edge)

        working["vertices"] = kept_vertices
        working["edges"] = kept_edges
        working["dirty"] = True
        working["topologic_graph"] = None
        Executor._rebuild_working_indexes(working)

        output_rows = Executor._project(rows, ast.returns) if ast.returns is not None else []

        return {
            "graph": working,
            "rows": output_rows,
            "action": "DELETE",
            "created": 0,
            "matched": len(rows),
            "updated": 0,
            "deleted": removed_vertex_count + removed_edge_count,
        }

    @staticmethod
    def _rows_for_match(graph, ast, silent: bool = False) -> List[Dict[str, Any]]:
        rows = Executor._match_pattern_or_node(graph, ast.match, silent=silent)

        if ast.where is not None:
            rows = [
                row for row in rows
                if Executor._evaluate_where_clause(row, ast.where, silent=silent)
            ]

        return rows

    @staticmethod
    def _finalize_rows(rows: List[Dict[str, Any]], ast) -> List[Dict[str, Any]]:
        if ast.returns is not None:
            rows = Executor._project(rows, ast.returns)

        order_by = getattr(ast, "order_by", None)
        if order_by is not None:
            rows = Executor._order_by(rows, order_by)

        skip = getattr(ast, "skip", None)
        if isinstance(skip, int) and skip > 0:
            rows = rows[skip:]

        limit = getattr(ast, "limit", None)
        if isinstance(limit, int) and limit >= 0:
            rows = rows[:limit]

        return rows

    @staticmethod
    def _match_pattern_or_node(graph, pattern, silent: bool = False) -> List[Dict[str, Any]]:
        if pattern is None:
            return []
        if getattr(pattern, "edge", None) is None or getattr(pattern, "right_node", None) is None:
            return Executor._match_node(graph, pattern.left_node, silent=silent)
        return Executor._match(graph, pattern, silent=silent)

    @staticmethod
    def _match_node(graph, node_pattern, silent: bool = False) -> List[Dict[str, Any]]:
        vertices = Executor._graph_vertices(graph)
        if not isinstance(vertices, list):
            vertices = []

        variable = getattr(node_pattern, "variable", None) or "_"
        results = []

        for vertex in vertices:
            if Executor._vertex_matches(vertex, node_pattern):
                results.append({variable: vertex})

        return results

    @staticmethod
    def _match(graph, match_pattern, silent: bool = False) -> List[Dict[str, Any]]:
        """Returns rows matching a node-edge-node pattern using src/dst dictionaries."""

        results = []
        edges = Executor._graph_edges(graph)
        if not isinstance(edges, list):
            edges = []

        vertices = Executor._graph_vertices(graph)
        vertex_by_index = Executor._vertex_by_index_map(graph)

        left_pattern = match_pattern.left_node
        edge_pattern = match_pattern.edge
        right_pattern = match_pattern.right_node

        for edge in edges:
            if not Executor._edge_matches(edge, edge_pattern):
                continue

            src_vertex, dst_vertex = Executor._edge_endpoint_vertices(edge, vertex_by_index, vertices)
            if src_vertex is None or dst_vertex is None:
                continue

            if edge_pattern.direction == "out":
                candidate_pairs = [(src_vertex, dst_vertex)]
            elif edge_pattern.direction == "in":
                candidate_pairs = [(dst_vertex, src_vertex)]
            else:
                candidate_pairs = [(src_vertex, dst_vertex), (dst_vertex, src_vertex)]

            for left_vertex, right_vertex in candidate_pairs:
                if not Executor._vertex_matches(left_vertex, left_pattern):
                    continue
                if not Executor._vertex_matches(right_vertex, right_pattern):
                    continue

                row = {}
                if left_pattern.variable:
                    row[left_pattern.variable] = left_vertex
                if right_pattern.variable:
                    row[right_pattern.variable] = right_vertex
                if edge_pattern.variable:
                    row[edge_pattern.variable] = edge

                results.append(row)

        return results

    @staticmethod
    def _vertex_matches(vertex, node_pattern) -> bool:
        if vertex is None:
            return False

        label = getattr(node_pattern, "label", None)
        if label:
            vertex_label = Executor._dictionary_value(vertex, "label")
            vertex_category = Executor._dictionary_value(vertex, "category")
            vertex_ontology_class = Executor._dictionary_value(vertex, "ontology_class")
            vertex_type = Executor._dictionary_value(vertex, "type")

            if not (
                Executor._ontology_match(vertex_label, label)
                or Executor._ontology_match(vertex_category, label)
                or Executor._ontology_match(vertex_ontology_class, label)
                or Executor._ontology_match(vertex_type, label)
            ):
                return False

        properties = getattr(node_pattern, "properties", None) or {}
        for key, value in properties.items():
            if Executor._dictionary_value(vertex, key) != value:
                return False

        return True

    @staticmethod
    def _edge_matches(edge, edge_pattern) -> bool:
        if edge is None:
            return False

        label = getattr(edge_pattern, "label", None)
        if label:
            edge_label = Executor._dictionary_value(edge, "label")
            edge_category = Executor._dictionary_value(edge, "category")
            edge_ontology_class = Executor._dictionary_value(edge, "ontology_class")
            edge_type = Executor._dictionary_value(edge, "type")

            if not (
                Executor._ontology_match(edge_label, label)
                or Executor._ontology_match(edge_category, label)
                or Executor._ontology_match(edge_ontology_class, label)
                or Executor._ontology_match(edge_type, label)
            ):
                return False

        properties = getattr(edge_pattern, "properties", None) or {}
        for key, value in properties.items():
            if Executor._dictionary_value(edge, key) != value:
                return False

        return True

    @staticmethod
    def _create_pattern(graph, pattern, silent: bool = False):
        """Creates a node or node-edge-node pattern in the GQL working graph only."""

        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge

        working = Executor._ensure_working_graph(graph, silent=silent)
        if working is None or pattern is None:
            return graph, {}, 0

        row = {}
        created = 0

        existing_vertices = working.get("vertices", []) or []
        existing_edges = working.get("edges", []) or []

        left_index = Executor._next_vertex_index(working)
        right_index = left_index + 1

        left = Vertex.ByCoordinates(left_index, 0, 0)
        Executor._apply_pattern_dictionary(left, pattern.left_node)
        Executor._set_dictionary_values(left, {"index": left_index})
        if pattern.left_node.variable:
            row[pattern.left_node.variable] = left
        created += 1

        if getattr(pattern, "edge", None) is None or getattr(pattern, "right_node", None) is None:
            working["vertices"] = existing_vertices + [left]
            working["dirty"] = True
            working["topologic_graph"] = None
            Executor._rebuild_working_indexes(working)
            return working, row, created

        right = Vertex.ByCoordinates(right_index, 0, 0)
        Executor._apply_pattern_dictionary(right, pattern.right_node)
        Executor._set_dictionary_values(right, {"index": right_index})
        if pattern.right_node.variable:
            row[pattern.right_node.variable] = right
        created += 1

        direction = getattr(pattern.edge, "direction", "out")
        if direction == "in":
            edge = Edge.ByVertices(right, left)
            src_index = right_index
            dst_index = left_index
        else:
            edge = Edge.ByVertices(left, right)
            src_index = left_index
            dst_index = right_index

        Executor._apply_pattern_dictionary(edge, pattern.edge)
        Executor._set_dictionary_values(edge, {"src": src_index, "dst": dst_index})
        if pattern.edge.variable:
            row[pattern.edge.variable] = edge
        created += 1

        working["vertices"] = existing_vertices + [left, right]
        working["edges"] = existing_edges + [edge]
        working["dirty"] = True
        working["topologic_graph"] = None
        Executor._rebuild_working_indexes(working)

        return working, row, created

    @staticmethod
    def _apply_pattern_dictionary(topology, pattern):
        data = {}
        is_edge = isinstance(pattern, object) and pattern.__class__.__name__.lower().startswith("edge")
        data = Executor._ontology_annotate_pattern_data(data, pattern, is_edge=is_edge)
        properties = getattr(pattern, "properties", None) or {}
        data.update(properties)
        Executor._set_dictionary_values(topology, data)
        return topology

    @staticmethod
    def _evaluate_where_clause(row: Dict[str, Any], where_clause, silent: bool = False) -> bool:
        expression = getattr(where_clause, "expression", where_clause)
        return Executor._evaluate_boolean_expression(row, expression, silent=silent)

    @staticmethod
    def _evaluate_boolean_expression(row: Dict[str, Any], expression, silent: bool = False) -> bool:
        if hasattr(expression, "variable") and hasattr(expression, "property"):
            return Executor._evaluate_where(row, expression, silent=silent)

        operator = getattr(expression, "operator", None)
        left = getattr(expression, "left", None)
        right = getattr(expression, "right", None)

        if operator == "AND":
            return Executor._evaluate_boolean_expression(row, left, silent=silent) and Executor._evaluate_boolean_expression(row, right, silent=silent)

        if operator == "OR":
            return Executor._evaluate_boolean_expression(row, left, silent=silent) or Executor._evaluate_boolean_expression(row, right, silent=silent)

        return False

    @staticmethod
    def _evaluate_where(row: Dict[str, Any], predicate, silent: bool = False) -> bool:
        obj = row.get(predicate.variable)
        if obj is None:
            return False
        actual_value = Executor._dictionary_value(obj, predicate.property)
        return Executor._compare(actual_value, predicate.operator, predicate.value, silent=silent)

    @staticmethod
    def _compare(left, operator: str, right, silent: bool = False) -> bool:
        try:
            left = Executor._coerce_number(left)
            right = Executor._coerce_number(right)
            if operator in ("=", "=="):
                return left == right
            if operator in ("!=", "<>"):
                return left != right
            if operator == ">":
                return left > right
            if operator == "<":
                return left < right
            if operator == ">=":
                return left >= right
            if operator == "<=":
                return left <= right
            return False
        except Exception:
            if not silent:
                print("GQL.Executor - Warning: Could not evaluate predicate.")
                print("Left:", left)
                print("Operator:", operator)
                print("Right:", right)
            return False

    @staticmethod
    def _project(rows: List[Dict[str, Any]], return_clause) -> List[Dict[str, Any]]:
        if return_clause is None:
            return rows

        if isinstance(return_clause, list):
            items = return_clause
            distinct = False
        else:
            items = getattr(return_clause, "items", [])
            distinct = bool(getattr(return_clause, "distinct", False))

        if items == ["*"]:
            projected = list(rows)
        elif Executor._contains_aggregate(items):
            projected = [Executor._aggregate_row(rows, items)]
        else:
            projected = []
            for row in rows:
                result = {}
                for item in items:
                    key, value = Executor._evaluate_return_item(row, item)
                    result[key] = value
                projected.append(result)

        if distinct:
            projected = Executor._distinct(projected)
        return projected

    @staticmethod
    def _contains_aggregate(items: List[Any]) -> bool:
        for item in items:
            expression = getattr(item, "expression", item)
            if Executor._is_aggregate(expression):
                return True
        return False

    @staticmethod
    def _is_aggregate(expression: Any) -> bool:
        return hasattr(expression, "function") and str(getattr(expression, "function", "")).upper() == "COUNT"

    @staticmethod
    def _aggregate_row(rows: List[Dict[str, Any]], items: List[Any]) -> Dict[str, Any]:
        result = {}
        for item in items:
            expression = getattr(item, "expression", item)
            alias = getattr(item, "alias", None)
            if Executor._is_aggregate(expression):
                key = alias or Executor._expression_key(expression)
                result[key] = Executor._count(rows, expression)
            else:
                key = alias or Executor._expression_key(expression)
                value = Executor._evaluate_expression(rows[0], expression)[1] if rows else None
                result[key] = value
        return result

    @staticmethod
    def _count(rows: List[Dict[str, Any]], expression) -> int:
        argument = getattr(expression, "argument", "*")
        distinct = bool(getattr(expression, "distinct", False))
        if argument == "*":
            if not distinct:
                return len(rows)
            return len({Executor._hashable_row(row) for row in rows})
        values = []
        for row in rows:
            _, value = Executor._evaluate_expression(row, argument)
            if value is not None:
                values.append(value)
        if distinct:
            return len({Executor._make_hashable(value) for value in values})
        return len(values)

    @staticmethod
    def _evaluate_return_item(row: Dict[str, Any], item: Any) -> Tuple[str, Any]:
        expression = getattr(item, "expression", item)
        alias = getattr(item, "alias", None)
        key, value = Executor._evaluate_expression(row, expression)
        return alias or key, value

    @staticmethod
    def _evaluate_expression(row: Dict[str, Any], expression: Any) -> Tuple[str, Any]:
        if isinstance(expression, tuple) and len(expression) == 2:
            variable, prop = expression
            obj = row.get(variable)
            return f"{variable}.{prop}", Executor._dictionary_value(obj, prop)
        key = str(expression)
        return key, row.get(key)

    @staticmethod
    def _expression_key(expression: Any) -> str:
        if Executor._is_aggregate(expression):
            arg = getattr(expression, "argument", "*")
            arg_text = f"{arg[0]}.{arg[1]}" if isinstance(arg, tuple) and len(arg) == 2 else str(arg)
            prefix = "DISTINCT " if bool(getattr(expression, "distinct", False)) else ""
            return f"COUNT({prefix}{arg_text})"
        if isinstance(expression, tuple) and len(expression) == 2:
            return f"{expression[0]}.{expression[1]}"
        return str(expression)

    @staticmethod
    def _distinct(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        unique_rows = []
        for row in rows:
            key = Executor._hashable_row(row)
            if key in seen:
                continue
            seen.add(key)
            unique_rows.append(row)
        return unique_rows

    @staticmethod
    def _hashable_row(row: Dict[str, Any]):
        return tuple((key, Executor._make_hashable(value)) for key, value in sorted(row.items(), key=lambda pair: pair[0]))

    @staticmethod
    def _make_hashable(value: Any):
        if isinstance(value, dict):
            return tuple((key, Executor._make_hashable(val)) for key, val in sorted(value.items(), key=lambda pair: pair[0]))
        if isinstance(value, (list, tuple, set)):
            return tuple(Executor._make_hashable(item) for item in value)
        try:
            hash(value)
            return value
        except Exception:
            return repr(value)

    @staticmethod
    def _order_by(rows: List[Dict[str, Any]], order_by: Dict[str, Any]) -> List[Dict[str, Any]]:
        order_items = order_by.get("items", [])
        sorted_rows = list(rows)
        for order_item in reversed(order_items):
            item = order_item.get("item")
            direction = order_item.get("direction", "ASC")
            key = f"{item[0]}.{item[1]}" if isinstance(item, tuple) and len(item) == 2 else str(item)
            reverse = direction == "DESC"
            sorted_rows = sorted(sorted_rows, key=lambda row: (row.get(key) is None, row.get(key)), reverse=reverse)
        return sorted_rows

    @staticmethod
    def _dictionary_value(topology, key: str, default=None):
        """Safely returns a dictionary value from a Topologic topology."""

        if topology is None:
            return default

        if isinstance(topology, (str, int, float, bool, tuple, list, dict)):
            return default

        from contextlib import redirect_stdout
        import io

        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        try:
            if not Topology.IsInstance(topology, "topology"):
                return default
        except Exception:
            return default

        try:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                d = Topology.Dictionary(topology)
        except Exception:
            return default

        try:
            return Dictionary.ValueAtKey(d, key, default)
        except Exception:
            return default

    @staticmethod
    def _dictionary_to_python(topology) -> Dict[str, Any]:
        """Safely converts a Topologic dictionary to a Python dictionary."""

        if topology is None:
            return {}

        if isinstance(topology, (str, int, float, bool, tuple, list, dict)):
            return {}

        from contextlib import redirect_stdout
        import io

        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        try:
            if not Topology.IsInstance(topology, "topology"):
                return {}
        except Exception:
            return {}

        try:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                d = Topology.Dictionary(topology)
        except Exception:
            return {}

        if d is None:
            return {}

        try:
            if hasattr(Dictionary, "PythonDictionary"):
                result = Dictionary.PythonDictionary(d)
                return result if isinstance(result, dict) else {}

            if hasattr(Dictionary, "Keys"):
                keys = Dictionary.Keys(d)
                return {
                    key: Dictionary.ValueAtKey(d, key, None)
                    for key in keys
                }
        except Exception:
            return {}

        return {}

    @staticmethod
    def _set_dictionary_values(topology, values: Dict[str, Any]):
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        data = Executor._dictionary_to_python(topology)
        data.update(values or {})
        d = Dictionary.ByKeysValues(list(data.keys()), list(data.values()))
        return Topology.SetDictionary(topology, d)

    @staticmethod
    def _graph_by_vertices_edges(vertices, edges, ontology: bool = True, silent: bool = False):
        from topologicpy.Graph import Graph

        try:
            result = Graph.ByVerticesEdges(vertices, edges, ontology=ontology, silent=silent)
            return result
        except TypeError:
            try:
                result = Graph.ByVerticesEdges(vertices, edges, silent=silent)
            except TypeError:
                result = Graph.ByVerticesEdges(vertices, edges)
            if ontology and result is not None:
                try:
                    from topologicpy.Graph import Graph as _Graph
                    if hasattr(_Graph, "AnnotateOntology"):
                        result = _Graph.AnnotateOntology(result, silent=True)
                except Exception:
                    pass
            return result
        except Exception as e:
            if not silent:
                print("GQL.Executor._graph_by_vertices_edges - Warning: Could not rebuild graph.")
                print("Error:", e)
            return None

    @staticmethod
    def _prime_graph_cache(graph, vertices=None, edges=None):
        working = Executor._ensure_working_graph(graph)
        if not Executor._is_working_graph(working):
            return
        if isinstance(vertices, list):
            working["vertices"] = vertices
        if isinstance(edges, list):
            working["edges"] = edges
        working["dirty"] = True
        working["topologic_graph"] = None
        Executor._rebuild_working_indexes(working)

    @staticmethod
    def _add_vertex(graph, vertex, silent: bool = False):
        working = Executor._ensure_working_graph(graph, silent=silent)
        if Executor._is_working_graph(working):
            vertices = working.get("vertices", []) or []
            working["vertices"] = vertices + [vertex]
            working["dirty"] = True
            working["topologic_graph"] = None
            Executor._rebuild_working_indexes(working)
            return working
        return graph

    @staticmethod
    def _add_edge(graph, edge, silent: bool = False):
        working = Executor._ensure_working_graph(graph, silent=silent)
        if Executor._is_working_graph(working):
            edges = working.get("edges", []) or []
            working["edges"] = edges + [edge]
            working["dirty"] = True
            working["topologic_graph"] = None
            return working
        return graph

    @staticmethod
    def _remove_edge(graph, edge, silent: bool = False):
        return Executor._rebuild_graph_without(graph, remove_edges=[edge])

    @staticmethod
    def _remove_vertex(graph, vertex, silent: bool = False):
        return Executor._rebuild_graph_without(graph, remove_vertices=[vertex])

    @staticmethod
    def _rebuild_graph_without(graph, remove_vertices=None, remove_edges=None):
        remove_vertices = remove_vertices or []
        remove_edges = remove_edges or []
        working = Executor._ensure_working_graph(graph)
        if working is None:
            return graph
        vertices = working.get("vertices", []) or []
        edges = working.get("edges", []) or []
        vertex_by_index = Executor._vertex_by_index_map(working)
        kept_vertices = [v for v in vertices if not Executor._contains_topology(remove_vertices, v)]
        kept_edges = []
        for e in edges:
            if Executor._contains_topology(remove_edges, e):
                continue
            sv, ev = Executor._edge_endpoint_vertices(e, vertex_by_index, vertices)
            if Executor._contains_topology(remove_vertices, sv) or Executor._contains_topology(remove_vertices, ev):
                continue
            kept_edges.append(e)
        working["vertices"] = kept_vertices
        working["edges"] = kept_edges
        working["dirty"] = True
        working["topologic_graph"] = None
        Executor._rebuild_working_indexes(working)
        return working

    @staticmethod
    def _contains_topology(collection, topology) -> bool:
        return any(Executor._same_topology(item, topology) for item in collection)

    @staticmethod
    def _unique_topologies(items):
        result = []
        for item in items:
            if not Executor._contains_topology(result, item):
                result.append(item)
        return result

    @staticmethod
    def _same_topology(a, b) -> bool:
        if a is b:
            return True
        try:
            from topologicpy.Topology import Topology
            return bool(Topology.IsSame(a, b))
        except Exception:
            return False

    @staticmethod
    def _is_vertex(obj) -> bool:
        try:
            from topologicpy.Topology import Topology
            return bool(Topology.IsInstance(obj, "vertex"))
        except Exception:
            return obj.__class__.__name__.lower().endswith("vertex")

    @staticmethod
    def _is_edge(obj) -> bool:
        try:
            from topologicpy.Topology import Topology
            return bool(Topology.IsInstance(obj, "edge"))
        except Exception:
            return obj.__class__.__name__.lower().endswith("edge")

    @staticmethod
    def _coerce_number(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, int) or isinstance(value, float):
            return value
        if isinstance(value, str):
            text = value.strip()
            if text.replace(".", "", 1).replace("-", "", 1).isdigit():
                return float(text) if "." in text else int(text)
            return text
        return value

    @staticmethod
    def _has_aggregate_return(returns) -> bool:
        if not isinstance(returns, dict):
            return False

        for item in returns.get("items", []):
            expr = item.get("expr") if isinstance(item, dict) else item
            if isinstance(expr, dict) and expr.get("type") == "count":
                return True

        return False

    @staticmethod
    def _aggregate(rows: List[Dict[str, Any]], returns) -> List[Dict[str, Any]]:
        result = {}

        for item in returns.get("items", []):
            expr = item.get("expr") if isinstance(item, dict) else item
            alias = item.get("alias") if isinstance(item, dict) else None

            if not (isinstance(expr, dict) and expr.get("type") == "count"):
                continue

            arg = expr.get("arg")
            key = alias or Executor._return_key(expr)

            if arg == "*":
                result[key] = len(rows)
                continue

            distinct = bool(arg.get("distinct")) if isinstance(arg, dict) else False
            inner_expr = arg.get("expr") if isinstance(arg, dict) else arg

            values = []
            for row in rows:
                values.append(Executor._resolve_return_value(row, inner_expr))

            if distinct:
                values = list({Executor._hashable_value(v) for v in values})

            result[key] = len(values)

        return [result]

    @staticmethod
    def _return_key(expression: Any) -> str:
        if isinstance(expression, dict):
            expr_type = expression.get("type")
            if expr_type == "count":
                arg = expression.get("arg", "*")
                return f"COUNT({arg})"
        return Executor._expression_key(expression)

    @staticmethod
    def _resolve_return_value(row: Dict[str, Any], expression: Any):
        return Executor._evaluate_expression(row, expression)[1]

    @staticmethod
    def _hashable_value(value: Any):
        return Executor._make_hashable(value)

    @staticmethod
    def _is_distinct_return(returns) -> bool:
        if not isinstance(returns, dict):
            return False

        return bool(returns.get("distinct", False))

    @staticmethod
    def _fast_count_match(graph, ast, silent: bool = False):
        """Fast path for simple COUNT(*) queries over the working edge list."""

        if getattr(ast, "where", None) is not None:
            return None
        if getattr(ast, "order_by", None) is not None:
            return None
        if getattr(ast, "skip", None) is not None:
            return None
        if getattr(ast, "limit", None) is not None:
            return None

        returns = getattr(ast, "returns", None)
        items = getattr(returns, "items", None)
        if not isinstance(items, list) or len(items) != 1:
            return None

        item = items[0]
        expression = getattr(item, "expression", None)
        alias = getattr(item, "alias", None)
        if not Executor._is_aggregate(expression):
            return None

        function = str(getattr(expression, "function", "")).upper()
        argument = getattr(expression, "argument", None)
        distinct = bool(getattr(expression, "distinct", False))
        if function != "COUNT" or argument != "*" or distinct:
            return None

        match_pattern = getattr(ast, "match", None)
        if match_pattern is None:
            return None

        left_node = getattr(match_pattern, "left_node", None)
        right_node = getattr(match_pattern, "right_node", None)
        edge_pattern = getattr(match_pattern, "edge", None)
        if left_node is None or right_node is None or edge_pattern is None:
            return None

        if getattr(left_node, "label", None) is not None:
            return None
        if getattr(right_node, "label", None) is not None:
            return None
        if getattr(left_node, "properties", None):
            return None
        if getattr(right_node, "properties", None):
            return None
        if getattr(edge_pattern, "label", None) is not None:
            return None
        if getattr(edge_pattern, "properties", None):
            return None

        edges = Executor._graph_edges(graph)
        direction = getattr(edge_pattern, "direction", "out")
        count = len(edges) * 2 if direction == "undirected" else len(edges)
        key = alias or Executor._expression_key(expression)
        return [{key: count}]
