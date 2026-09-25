# Copyright (C) 2026
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free Software
# Foundation, either version 3.0 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import copy
import json


@dataclass
class _CacheEntry:
    signature: Any
    result: Any


@dataclass
class _EvaluationContext:
    graph: Any
    plan_version: Any = None
    plan: Optional[dict] = None
    cache: Dict[int, _CacheEntry] = field(default_factory=dict)
    lineage: Dict[int, List[dict]] = field(default_factory=dict)
    evaluations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    last_elapsed: float = 0.0
    last_target: Optional[int] = None


_CONTEXTS: Dict[int, _EvaluationContext] = {}


class CSG:
    """Parametric, provenance-aware constructive solid geometry over TGraph.

    TGraph is the public expression/dependency graph.  Runtime geometry, caches,
    dirty state and OCCT/BRepGraph lineage are deliberately kept outside the
    serialisable TGraph and are private implementation details.

    The graph contains two vertex kinds:

    * ``source``: a TopologicPy topology stored as the TGraph vertex
      representation.
    * ``operation``: an abstract CSG operation.  Directed operand edges point
      from dependencies to the operation and carry an integer ``operand`` role.

    Evaluation is incremental.  Each node is fingerprinted from its expression
    data and input fingerprints; unchanged nodes reuse their cached topology.
    PythonOCC operations capture exact BRepTools/BRepGraph lineage through the
    private provenance hook when available.  No BRepGraph identifiers escape
    this public class.
    """

    SCHEMA = "topologicpy.csg/2"
    SOURCE = "source"
    OPERATION = "operation"

    _ALIASES = {
        "union": "union",
        "fuse": "union",
        "intersection": "intersection",
        "intersect": "intersection",
        "common": "intersection",
        "difference": "difference",
        "subtract": "difference",
        "subtraction": "difference",
        "cut": "difference",
        "xor": "xor",
        "symdif": "xor",
        "symmetricdifference": "xor",
        "symmetric_difference": "xor",
        "merge": "merge",
        "impose": "impose",
        "imprint": "imprint",
        "slice": "slice",
        "transform": "transform",
    }

    _NARY = {"union", "intersection", "xor", "merge"}
    _BINARY = {"difference", "impose", "imprint", "slice"}
    _UNARY = {"transform"}

    # ------------------------------------------------------------------
    # TGraph construction
    # ------------------------------------------------------------------

    @staticmethod
    def Init(dictionary: Optional[dict] = None):
        """Return an empty directed CSG expression TGraph."""
        from topologicpy.TGraph import TGraph

        d = {
            "type": "CSG",
            "schema": CSG.SCHEMA,
        }
        if isinstance(dictionary, dict):
            d["data"] = copy.deepcopy(dictionary)
        return TGraph(
            directed=True,
            allowSelfLoops=False,
            allowParallelEdges=False,
            dictionary=d,
        )

    @staticmethod
    def Source(
        graph,
        topology,
        name: Optional[str] = None,
        data: Optional[dict] = None,
        silent: bool = False,
    ) -> Optional[int]:
        """Add a source topology node and return its stable TGraph index."""
        from topologicpy.TGraph import TGraph
        from topologicpy.Topology import Topology

        if not isinstance(graph, TGraph):
            if not silent:
                print("CSG.Source - Error: graph is not a TGraph. Returning None.")
            return None
        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print("CSG.Source - Error: topology is invalid. Returning None.")
            return None

        d = {
            "csg_kind": CSG.SOURCE,
            "name": str(name) if name is not None else None,
            "topology_type": Topology.TypeAsString(topology),
        }
        if isinstance(data, dict):
            d["data"] = copy.deepcopy(data)
        index = graph.AddVertex(dictionary=d, representation=topology, silent=silent)
        CSG._invalidate_plan(graph)
        return index

    @staticmethod
    def Operation(
        graph,
        operation: str,
        inputs: Sequence[Any],
        name: Optional[str] = None,
        matrix: Optional[Sequence[Sequence[float]]] = None,
        data: Optional[dict] = None,
        silent: bool = False,
    ) -> Optional[int]:
        """Add an operation node and directed operand edges.

        Operand ordering is encoded only on edges (``operand=0,1,...``).  This
        removes the duplicated ``a_id``/``b_id`` state of the experimental CSG.
        """
        from topologicpy.TGraph import TGraph

        if not isinstance(graph, TGraph):
            if not silent:
                print("CSG.Operation - Error: graph is not a TGraph. Returning None.")
            return None
        op = CSG._operation_name(operation)
        if op is None:
            if not silent:
                print(f"CSG.Operation - Error: unknown operation '{operation}'. Returning None.")
            return None
        indices = [CSG._node_index(item) for item in (inputs or [])]
        if any(index is None for index in indices):
            if not silent:
                print("CSG.Operation - Error: one or more inputs are invalid. Returning None.")
            return None
        if not CSG._arity_ok(op, len(indices)):
            if not silent:
                print(f"CSG.Operation - Error: operation '{op}' has invalid arity {len(indices)}. Returning None.")
            return None
        active = set(TGraph.ActiveVertexIndices(graph))
        if any(index not in active for index in indices):
            if not silent:
                print("CSG.Operation - Error: one or more input nodes are not active. Returning None.")
            return None

        d = {
            "csg_kind": CSG.OPERATION,
            "operation": op,
            "name": str(name) if name is not None else None,
        }
        if matrix is not None:
            m = CSG._matrix(matrix)
            if m is None:
                if not silent:
                    print("CSG.Operation - Error: matrix is not a valid 4x4 matrix. Returning None.")
                return None
            d["matrix"] = m
        if isinstance(data, dict):
            d["data"] = copy.deepcopy(data)

        node = graph.AddVertex(dictionary=d, representation=None, silent=silent)
        if node is None:
            return None
        for operand, source in enumerate(indices):
            edge_d = {
                "csg_role": "operand",
                "operand": operand,
                "role": CSG._operand_role(operand),
            }
            edge = graph.AddEdge(
                source,
                node,
                directed=True,
                dictionary=edge_d,
                representation=None,
                silent=silent,
            )
            if edge is None:
                graph.RemoveVertex(node, silent=True)
                return None
        CSG._invalidate_plan(graph)
        return node

    @staticmethod
    def Union(graph, *inputs, **kwargs):
        return CSG.Operation(graph, "union", inputs, **kwargs)

    @staticmethod
    def Intersect(graph, *inputs, **kwargs):
        return CSG.Operation(graph, "intersection", inputs, **kwargs)

    @staticmethod
    def Difference(graph, a, b, **kwargs):
        return CSG.Operation(graph, "difference", [a, b], **kwargs)

    @staticmethod
    def XOR(graph, *inputs, **kwargs):
        return CSG.Operation(graph, "xor", inputs, **kwargs)

    @staticmethod
    def Merge(graph, *inputs, **kwargs):
        return CSG.Operation(graph, "merge", inputs, **kwargs)

    @staticmethod
    def Impose(graph, a, b, **kwargs):
        return CSG.Operation(graph, "impose", [a, b], **kwargs)

    @staticmethod
    def Imprint(graph, a, b, **kwargs):
        return CSG.Operation(graph, "imprint", [a, b], **kwargs)

    @staticmethod
    def Slice(graph, a, b, **kwargs):
        return CSG.Operation(graph, "slice", [a, b], **kwargs)

    @staticmethod
    def Transform(graph, source, matrix, **kwargs):
        return CSG.Operation(graph, "transform", [source], matrix=matrix, **kwargs)

    # ------------------------------------------------------------------
    # Editing
    # ------------------------------------------------------------------

    @staticmethod
    def SetSource(graph, node, topology, data: Optional[dict] = None, silent: bool = False) -> bool:
        """Replace a source topology and invalidate only dependent cached nodes."""
        from topologicpy.TGraph import TGraph
        from topologicpy.Topology import Topology

        if not isinstance(graph, TGraph) or not Topology.IsInstance(topology, "Topology"):
            return False
        index = CSG._node_index(node)
        record = CSG._vertex_record(graph, index)
        if record is None:
            return False
        d = dict(record.get("dictionary") or {})
        if d.get("csg_kind") != CSG.SOURCE:
            return False
        d["topology_type"] = Topology.TypeAsString(topology)
        if data is not None:
            d["data"] = copy.deepcopy(data) if isinstance(data, dict) else data
        graph.SetVertexDictionary(index, d)
        record = CSG._vertex_record(graph, index)
        if record is None:
            return False
        record["representation"] = topology
        CSG.Invalidate(graph, index, downstream=True)
        return True

    @staticmethod
    def Rewire(graph, operationNode, inputs: Sequence[Any], silent: bool = False) -> bool:
        """Replace all operand edges of an operation node."""
        from topologicpy.TGraph import TGraph

        if not isinstance(graph, TGraph):
            return False
        node = CSG._node_index(operationNode)
        record = CSG._vertex_record(graph, node)
        if record is None:
            return False
        d = record.get("dictionary") or {}
        op = CSG._operation_name(d.get("operation"))
        if d.get("csg_kind") != CSG.OPERATION or op is None:
            return False
        indices = [CSG._node_index(item) for item in (inputs or [])]
        if any(index is None for index in indices) or not CSG._arity_ok(op, len(indices)):
            return False

        for edge in list(TGraph.IncomingEdges(graph, node) or []):
            edge_index = edge.get("index") if isinstance(edge, dict) else edge
            if edge_index is not None:
                graph.RemoveEdge(edge_index, silent=True)
        for operand, source in enumerate(indices):
            if graph.AddEdge(
                source,
                node,
                directed=True,
                dictionary={
                    "csg_role": "operand",
                    "operand": operand,
                    "role": CSG._operand_role(operand),
                },
                silent=silent,
            ) is None:
                return False
        CSG._invalidate_plan(graph)
        CSG.Invalidate(graph, node, downstream=True)
        return True

    # ------------------------------------------------------------------
    # Compilation / validation
    # ------------------------------------------------------------------

    @staticmethod
    def Compile(graph, force: bool = False) -> Optional[dict]:
        """Compile the TGraph into an O(V+E) CSG evaluation plan."""
        from topologicpy.TGraph import TGraph

        if not isinstance(graph, TGraph):
            return None
        context = CSG._context(graph)
        version = getattr(graph, "_version", None)
        if not force and context.plan is not None and context.plan_version == version:
            return context.plan

        vertices_list = TGraph.Vertices(graph, copy=False, asTopologic=False, active=True) or []
        edges_list = TGraph.Edges(graph, asTopologic=False, active=True, copy=False) or []
        vertices = {v.get("index"): v for v in vertices_list if isinstance(v, dict) and isinstance(v.get("index"), int)}
        incoming = {index: [] for index in vertices}
        outgoing = {index: [] for index in vertices}
        errors = []
        warnings = []

        for edge in edges_list:
            if not isinstance(edge, dict):
                continue
            src = edge.get("src")
            dst = edge.get("dst")
            if src not in vertices or dst not in vertices:
                errors.append(f"Edge {edge.get('index')} references an inactive or missing vertex.")
                continue
            if not bool(edge.get("directed", getattr(graph, "_directed", True))):
                errors.append(f"Edge {edge.get('index')} is not directed.")
                continue
            edge_d = edge.get("dictionary") or {}
            operand = edge_d.get("operand")
            if not isinstance(operand, int) or operand < 0:
                errors.append(f"Edge {edge.get('index')} is missing a non-negative integer operand role.")
                continue
            incoming[dst].append((operand, src, edge))
            outgoing[src].append((dst, edge))

        for index, record in vertices.items():
            d = record.get("dictionary") or {}
            kind = d.get("csg_kind")
            if kind == CSG.SOURCE:
                if incoming[index]:
                    errors.append(f"Source node {index} has incoming operand edges.")
                if record.get("representation") is None:
                    errors.append(f"Source node {index} has no runtime topology representation.")
            elif kind == CSG.OPERATION:
                op = CSG._operation_name(d.get("operation"))
                if op is None:
                    errors.append(f"Operation node {index} has an unknown operation.")
                    continue
                ordered = sorted(incoming[index], key=lambda item: item[0])
                operands = [item[0] for item in ordered]
                if operands != list(range(len(operands))):
                    errors.append(f"Operation node {index} operand roles must be contiguous from zero.")
                if not CSG._arity_ok(op, len(ordered)):
                    errors.append(f"Operation node {index} ('{op}') has invalid arity {len(ordered)}.")
                incoming[index] = ordered
                if op == "transform" and CSG._matrix(d.get("matrix")) is None:
                    errors.append(f"Transform node {index} has no valid 4x4 matrix.")
            else:
                errors.append(f"Node {index} has unknown csg_kind '{kind}'.")

        indegree = {index: len(incoming[index]) for index in vertices}
        queue = [index for index, value in indegree.items() if value == 0]
        queue.sort()
        order = []
        cursor = 0
        while cursor < len(queue):
            u = queue[cursor]
            cursor += 1
            order.append(u)
            for v, _edge in outgoing[u]:
                indegree[v] -= 1
                if indegree[v] == 0:
                    queue.append(v)
        if len(order) != len(vertices):
            errors.append("The CSG expression graph contains a directed cycle.")

        roots = sorted(index for index in vertices if len(outgoing[index]) == 0)
        leaves = sorted(index for index in vertices if len(incoming[index]) == 0)
        if not roots and vertices:
            errors.append("The CSG expression graph has no root.")
        if len(roots) > 1:
            warnings.append("The CSG expression graph has multiple roots; Evaluate requires an explicit target node.")

        plan = {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "vertices": vertices,
            "edges": edges_list,
            "incoming": incoming,
            "outgoing": outgoing,
            "order": order,
            "roots": roots,
            "leaves": leaves,
            "version": version,
        }
        context.plan = plan
        context.plan_version = version
        return plan

    @staticmethod
    def Validate(graph) -> dict:
        plan = CSG.Compile(graph, force=True)
        if plan is None:
            return {"valid": False, "errors": ["Input is not a TGraph."], "warnings": [], "roots": [], "leaves": []}
        return {
            "valid": bool(plan["valid"]),
            "errors": list(plan["errors"]),
            "warnings": list(plan["warnings"]),
            "roots": list(plan["roots"]),
            "leaves": list(plan["leaves"]),
            "order": list(plan["order"]),
        }

    @staticmethod
    def Roots(graph) -> list:
        plan = CSG.Compile(graph)
        return list(plan["roots"]) if plan else []

    @staticmethod
    def Leaves(graph) -> list:
        plan = CSG.Compile(graph)
        return list(plan["leaves"]) if plan else []

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def Evaluate(
        graph,
        node: Any = None,
        cache: bool = True,
        lineage: bool = True,
        silent: bool = False,
    ):
        """Evaluate one CSG node, reusing unchanged subexpressions."""
        from topologicpy.Topology import Topology

        plan = CSG.Compile(graph)
        if not plan or not plan["valid"]:
            if not silent:
                message = "; ".join(plan["errors"] if plan else ["Invalid graph."])
                print(f"CSG.Evaluate - Error: {message} Returning None.")
            return None

        target = CSG._node_index(node)
        if target is None:
            if len(plan["roots"]) != 1:
                if not silent:
                    print("CSG.Evaluate - Error: graph does not have exactly one root; specify node. Returning None.")
                return None
            target = plan["roots"][0]
        if target not in plan["vertices"]:
            if not silent:
                print("CSG.Evaluate - Error: target node is not active. Returning None.")
            return None

        needed = CSG._ancestors_including(plan, target)
        context = CSG._context(graph)
        signatures = {}
        results = {}
        started = perf_counter()

        for index in plan["order"]:
            if index not in needed:
                continue
            record = plan["vertices"][index]
            d = record.get("dictionary") or {}
            kind = d.get("csg_kind")
            input_nodes = [item[1] for item in plan["incoming"][index]]
            input_signatures = tuple(signatures[item] for item in input_nodes)

            if kind == CSG.SOURCE:
                representation = record.get("representation")
                signature = CSG._source_signature(representation, d)
            else:
                signature = ("operation", CSG._freeze(CSG._expression_dictionary(d)), input_signatures)

            signatures[index] = signature
            cached = context.cache.get(index)
            if cache and cached is not None and cached.signature == signature:
                results[index] = cached.result
                context.cache_hits += 1
                continue

            context.cache_misses += 1
            if kind == CSG.SOURCE:
                result = record.get("representation")
                if not Topology.IsInstance(result, "Topology"):
                    if not silent:
                        print(f"CSG.Evaluate - Error: source node {index} has no valid topology. Returning None.")
                    return None
                context.lineage.pop(index, None)
            else:
                operands = [results[item] for item in input_nodes]
                result, history_records = CSG._evaluate_operation(
                    index,
                    d,
                    input_nodes,
                    operands,
                    capture_lineage=bool(lineage),
                    silent=silent,
                )
                if result is None:
                    if not silent:
                        print(f"CSG.Evaluate - Error: operation node {index} failed. Returning None.")
                    return None
                if lineage:
                    context.lineage[index] = history_records
                else:
                    context.lineage.pop(index, None)

            results[index] = result
            if cache:
                context.cache[index] = _CacheEntry(signature=signature, result=result)

        context.evaluations += 1
        context.last_elapsed = perf_counter() - started
        context.last_target = target
        return results.get(target)

    @staticmethod
    def Result(graph, node: Any = None):
        context = CSG._context(graph, create=False)
        if context is None:
            return None
        target = CSG._node_index(node)
        if target is None:
            target = context.last_target
        entry = context.cache.get(target) if target is not None else None
        return entry.result if entry is not None else None

    @staticmethod
    def Invalidate(graph, node: Any = None, downstream: bool = True) -> bool:
        """Invalidate cached evaluation and lineage state."""
        context = CSG._context(graph, create=False)
        if context is None:
            return True
        if node is None:
            context.cache.clear()
            context.lineage.clear()
            return True
        index = CSG._node_index(node)
        if index is None:
            return False
        affected = {index}
        if downstream:
            plan = CSG.Compile(graph)
            if plan:
                stack = [index]
                while stack:
                    u = stack.pop()
                    for v, _edge in plan["outgoing"].get(u, []):
                        if v not in affected:
                            affected.add(v)
                            stack.append(v)
        for item in affected:
            context.cache.pop(item, None)
            context.lineage.pop(item, None)
        return True

    @staticmethod
    def ClearRuntime(graph=None) -> None:
        """Drop private evaluation/cache/lineage state without changing TGraph data."""
        if graph is None:
            _CONTEXTS.clear()
        else:
            _CONTEXTS.pop(id(graph), None)

    @staticmethod
    def Status(graph) -> dict:
        context = CSG._context(graph, create=False)
        if context is None:
            return {
                "evaluations": 0,
                "cacheHits": 0,
                "cacheMisses": 0,
                "cachedNodes": 0,
                "lineageOperations": 0,
                "lineageRecords": 0,
                "lastElapsed": 0.0,
                "lastTarget": None,
            }
        return {
            "evaluations": context.evaluations,
            "cacheHits": context.cache_hits,
            "cacheMisses": context.cache_misses,
            "cachedNodes": len(context.cache),
            "lineageOperations": len(context.lineage),
            "lineageRecords": sum(len(records) for records in context.lineage.values()),
            "lastElapsed": context.last_elapsed,
            "lastTarget": context.last_target,
        }

    # ------------------------------------------------------------------
    # Exact lineage queries
    # ------------------------------------------------------------------

    @staticmethod
    def History(
        graph,
        operation: Any = None,
        relation: Optional[str] = None,
        sourceNode: Any = None,
        resultType: Optional[str] = None,
    ) -> list:
        """Return backend-neutral exact lineage records captured during evaluation."""
        context = CSG._context(graph, create=False)
        if context is None:
            return []
        operation_index = CSG._node_index(operation)
        source_index = CSG._node_index(sourceNode)
        relation_l = str(relation).lower() if relation is not None else None
        type_l = str(resultType).lower() if resultType is not None else None
        output = []
        items = context.lineage.items() if operation_index is None else [(operation_index, context.lineage.get(operation_index, []))]
        for op_node, records in items:
            for record in records:
                if relation_l is not None and str(record.get("relation", "")).lower() != relation_l:
                    continue
                if source_index is not None and record.get("sourceNode") != source_index:
                    continue
                if type_l is not None and str(record.get("resultType", "")).lower() != type_l:
                    continue
                try:
                    from topologicpy.pythonocc_backend._csg_lineage import materialise_record
                    output.append(materialise_record(record))
                except Exception:
                    public_record = {
                        key: value
                        for key, value in record.items()
                        if not str(key).startswith("_")
                    }
                    public_record.setdefault("source", None)
                    public_record.setdefault("result", None)
                    output.append(public_record)
        return output

    @staticmethod
    def GeneratedBy(graph, operation, topologyType: Optional[str] = None) -> list:
        return CSG._history_topologies(graph, operation, "generated", topologyType)

    @staticmethod
    def ModifiedBy(graph, operation, topologyType: Optional[str] = None) -> list:
        return CSG._history_topologies(graph, operation, "modified", topologyType)

    @staticmethod
    def UnchangedBy(graph, operation, topologyType: Optional[str] = None) -> list:
        return CSG._history_topologies(graph, operation, "unchanged", topologyType)

    @staticmethod
    def DeletedBy(graph, operation, topologyType: Optional[str] = None) -> list:
        records = CSG.History(graph, operation=operation, relation="deleted")
        result = [record.get("source") for record in records]
        if topologyType is not None:
            t = str(topologyType).lower()
            result = [topology for topology, record in zip(result, records) if str(record.get("sourceType", "")).lower() == t]
        return CSG._unique_topologies([item for item in result if item is not None])

    @staticmethod
    def Origins(graph, topology, operation: Any = None) -> list:
        """Trace a result subtopology backwards to source expression nodes."""
        plan = CSG.Compile(graph)
        if plan is None:
            return []
        records = CSG.History(graph, operation=operation)
        frontier = [topology]
        seen = set()
        origins = []
        while frontier:
            current = frontier.pop()
            key = CSG._topology_identity_key(current)
            if key in seen:
                continue
            seen.add(key)
            for record in records:
                target = record.get("result")
                if target is None or not CSG._same_topology(target, current):
                    continue
                source = record.get("source")
                source_node = record.get("sourceNode")
                if source is None:
                    continue
                source_record = plan["vertices"].get(source_node)
                source_kind = (source_record.get("dictionary") or {}).get("csg_kind") if isinstance(source_record, dict) else None
                if source_kind == CSG.SOURCE:
                    origins.append({
                        "sourceNode": source_node,
                        "source": source,
                        "viaOperation": record.get("operationNode"),
                        "relation": record.get("relation"),
                    })
                else:
                    frontier.append(source)
        return CSG._unique_origin_records(origins)

    @staticmethod
    def Descendants(graph, topology, operation: Any = None) -> list:
        """Trace a source subtopology forward through captured exact lineage."""
        records = CSG.History(graph, operation=operation)
        frontier = [topology]
        seen = set()
        results = []
        while frontier:
            current = frontier.pop()
            key = CSG._topology_identity_key(current)
            if key in seen:
                continue
            seen.add(key)
            for record in records:
                source = record.get("source")
                target = record.get("result")
                if source is None or not CSG._same_topology(source, current):
                    continue
                if target is not None:
                    results.append(target)
                    frontier.append(target)
        return CSG._unique_topologies(results)

    @staticmethod
    def LineageGraph(graph, operation: Any = None):
        """Return captured subtopology lineage as a TGraph.

        This is intentionally a *derived* TGraph.  BRepGraph stays private.
        """
        from topologicpy.TGraph import TGraph

        records = CSG.History(graph, operation=operation)
        lineage = TGraph(
            directed=True,
            allowSelfLoops=False,
            allowParallelEdges=True,
            dictionary={"type": "CSGLineage", "schema": CSG.SCHEMA},
        )
        buckets = {}

        def vertex_for(topology, role, metadata):
            if topology is None:
                return None
            key = CSG._topology_identity_key(topology)
            bucket = buckets.setdefault(key, [])
            for existing_topology, index in bucket:
                if CSG._same_topology(existing_topology, topology):
                    return index
            d = {"role": role}
            d.update(metadata)
            index = lineage.AddVertex(dictionary=d, representation=topology, silent=True)
            bucket.append((topology, index))
            return index

        for record in records:
            source = record.get("source")
            target = record.get("result")
            if source is None or target is None:
                continue
            s = vertex_for(source, "source", {"csgNode": record.get("sourceNode"), "topologyType": record.get("sourceType")})
            t = vertex_for(target, "result", {"csgNode": record.get("operationNode"), "topologyType": record.get("resultType")})
            if s is None or t is None:
                continue
            lineage.AddEdge(
                s,
                t,
                directed=True,
                dictionary={
                    "relation": record.get("relation"),
                    "operation": record.get("operation"),
                    "operationNode": record.get("operationNode"),
                    "stage": record.get("stage"),
                },
                silent=True,
            )
        return lineage

    # ------------------------------------------------------------------
    # Persistence: BREP only at the boundary, never as runtime state
    # ------------------------------------------------------------------

    @staticmethod
    def Data(graph, includeBREP: bool = True) -> Optional[dict]:
        """Return a persistence-safe CSG dictionary."""
        from topologicpy.TGraph import TGraph
        from topologicpy.Topology import Topology

        if not isinstance(graph, TGraph):
            return None
        vertices = []
        for record in TGraph.Vertices(graph, copy=False, asTopologic=False, active=True) or []:
            d = copy.deepcopy(record.get("dictionary") or {})
            item = {"index": record.get("index"), "dictionary": d}
            if d.get("csg_kind") == CSG.SOURCE and includeBREP:
                topology = record.get("representation")
                try:
                    item["brep"] = Topology.BREPString(topology)
                except Exception:
                    item["brep"] = None
            vertices.append(item)
        edges = []
        for edge in TGraph.Edges(graph, asTopologic=False, active=True, copy=False) or []:
            edges.append({
                "src": edge.get("src"),
                "dst": edge.get("dst"),
                "dictionary": copy.deepcopy(edge.get("dictionary") or {}),
            })
        return {
            "schema": CSG.SCHEMA,
            "dictionary": copy.deepcopy(TGraph.Dictionary(graph) or {}),
            "vertices": vertices,
            "edges": edges,
        }

    @staticmethod
    def ByData(data: dict, silent: bool = False):
        """Reconstruct a CSG TGraph from :meth:`Data`."""
        from topologicpy.Topology import Topology

        if not isinstance(data, dict) or data.get("schema") != CSG.SCHEMA:
            if not silent:
                print("CSG.ByData - Error: invalid CSG data. Returning None.")
            return None
        graph = CSG.Init(dictionary=(data.get("dictionary") or {}).get("data"))
        mapping = {}
        for item in sorted(data.get("vertices") or [], key=lambda value: value.get("index", -1)):
            d = copy.deepcopy(item.get("dictionary") or {})
            kind = d.get("csg_kind")
            representation = None
            if kind == CSG.SOURCE:
                brep = item.get("brep")
                if not isinstance(brep, str) or not brep:
                    if not silent:
                        print("CSG.ByData - Error: a source node is missing BREP data. Returning None.")
                    return None
                representation = Topology.ByBREPString(brep, silent=True)
                if representation is None:
                    return None
            index = graph.AddVertex(dictionary=d, representation=representation, silent=silent)
            mapping[item.get("index")] = index
        for edge in data.get("edges") or []:
            src = mapping.get(edge.get("src"))
            dst = mapping.get(edge.get("dst"))
            if src is None or dst is None:
                return None
            if graph.AddEdge(src, dst, directed=True, dictionary=copy.deepcopy(edge.get("dictionary") or {}), silent=silent) is None:
                return None
        return graph

    @staticmethod
    def JSONString(graph, indent: Optional[int] = None) -> Optional[str]:
        data = CSG.Data(graph, includeBREP=True)
        return json.dumps(data, indent=indent) if data is not None else None

    @staticmethod
    def ByJSONString(string: str, silent: bool = False):
        try:
            data = json.loads(string)
        except Exception:
            return None
        return CSG.ByData(data, silent=silent)

    @staticmethod
    def Export(graph, path: str, indent: int = 2, overwrite: bool = False, silent: bool = False) -> Optional[bool]:
        if not isinstance(path, str) or not path.strip():
            return None
        import os
        if os.path.exists(path) and not overwrite:
            if not silent:
                print("CSG.Export - Error: path already exists and overwrite is False. Returning None.")
            return None
        string = CSG.JSONString(graph, indent=indent)
        if string is None:
            return None
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(string)
            return True
        except Exception:
            return None

    @staticmethod
    def ByPath(path: str, silent: bool = False):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return CSG.ByJSONString(handle.read(), silent=silent)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _context(graph, create: bool = True) -> Optional[_EvaluationContext]:
        if graph is None:
            return None
        key = id(graph)
        context = _CONTEXTS.get(key)
        if context is not None and context.graph is graph:
            return context
        if not create:
            return None
        context = _EvaluationContext(graph=graph)
        _CONTEXTS[key] = context
        return context

    @staticmethod
    def _invalidate_plan(graph) -> None:
        context = CSG._context(graph, create=False)
        if context is not None:
            context.plan = None
            context.plan_version = None

    @staticmethod
    def _node_index(node) -> Optional[int]:
        if isinstance(node, int) and not isinstance(node, bool):
            return node
        if isinstance(node, dict):
            index = node.get("index")
            return index if isinstance(index, int) else None
        return None

    @staticmethod
    def _vertex_record(graph, index: Optional[int]):
        if index is None:
            return None
        try:
            from topologicpy.TGraph import TGraph
            return TGraph.Vertex(graph, index, copy=False, active=False, asTopologic=False, silent=True)
        except Exception:
            return None

    @staticmethod
    def _operation_name(operation) -> Optional[str]:
        if not isinstance(operation, str):
            return None
        return CSG._ALIASES.get(operation.strip().lower())

    @staticmethod
    def _arity_ok(operation: str, count: int) -> bool:
        if operation in CSG._UNARY:
            return count == 1
        if operation in CSG._BINARY:
            return count == 2
        if operation in CSG._NARY:
            return count >= 2
        return False

    @staticmethod
    def _operand_role(index: int) -> str:
        if index == 0:
            return "A"
        if index == 1:
            return "B"
        return str(index)

    @staticmethod
    def _matrix(matrix) -> Optional[list]:
        if matrix is None or not isinstance(matrix, (list, tuple)) or len(matrix) != 4:
            return None
        result = []
        try:
            for row in matrix:
                if not isinstance(row, (list, tuple)) or len(row) != 4:
                    return None
                result.append([float(value) for value in row])
        except Exception:
            return None
        return result

    @staticmethod
    def _ancestors_including(plan: dict, target: int) -> set:
        needed = {target}
        stack = [target]
        while stack:
            node = stack.pop()
            for _operand, source, _edge in plan["incoming"].get(node, []):
                if source not in needed:
                    needed.add(source)
                    stack.append(source)
        return needed

    @staticmethod
    def _expression_dictionary(dictionary: dict) -> dict:
        return {
            key: value
            for key, value in (dictionary or {}).items()
            if key not in {"topology_type"}
        }

    @staticmethod
    def _freeze(value):
        if isinstance(value, dict):
            return tuple(sorted((str(k), CSG._freeze(v)) for k, v in value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(CSG._freeze(item) for item in value)
        if isinstance(value, set):
            return tuple(sorted(CSG._freeze(item) for item in value))
        try:
            hash(value)
            return value
        except Exception:
            return repr(value)

    @staticmethod
    def _source_signature(topology, dictionary: dict):
        shape = getattr(topology, "shape", None)
        try:
            shape_hash = hash(shape) if shape is not None else None
        except Exception:
            shape_hash = None
        semantic = {}
        try:
            from topologicpy.Topology import Topology
            from topologicpy.Dictionary import Dictionary
            d = Topology.Dictionary(topology, silent=True)
            semantic = Dictionary.PythonDictionary(d, silent=True) or {}
        except Exception:
            semantic = {}
        return (
            "source",
            id(topology),
            shape_hash,
            CSG._freeze(CSG._expression_dictionary(dictionary)),
            CSG._freeze(semantic),
        )

    @staticmethod
    def _lineage_capture(operation_node: int, role_nodes: dict, stage: int, sink: list, enabled: bool):
        if not enabled:
            return nullcontext()
        try:
            from topologicpy.pythonocc_backend._csg_lineage import capture
            return capture(operation_node, role_nodes, stage=stage, sink=sink)
        except Exception:
            return nullcontext()

    @staticmethod
    def _evaluate_operation(index: int, dictionary: dict, input_nodes: list, operands: list, capture_lineage: bool, silent: bool):
        from topologicpy.Topology import Topology

        op = CSG._operation_name(dictionary.get("operation"))
        if op is None:
            return None, []
        history_records = []

        if op == "transform":
            matrix = CSG._matrix(dictionary.get("matrix"))
            if matrix is None or len(operands) != 1:
                return None, history_records
            role_nodes = {"source": input_nodes[0], "self": input_nodes[0]}
            with CSG._lineage_capture(index, role_nodes, 0, history_records, capture_lineage):
                try:
                    result = Topology.Transform(operands[0], matrix, transferDictionaries=True, silent=silent)
                except TypeError:
                    result = Topology.Transform(operands[0], matrix, silent=silent)
            return result, history_records

        result = operands[0]
        left_node = input_nodes[0]
        for stage, (right_node, right) in enumerate(zip(input_nodes[1:], operands[1:])):
            role_nodes = {
                "self": left_node,
                "source": left_node,
                "other": right_node,
                "tool": right_node,
            }
            with CSG._lineage_capture(index, role_nodes, stage, history_records, capture_lineage):
                result = CSG._boolean_call(op, result, right, silent=silent)
            if result is None:
                return None, history_records
            left_node = index

        matrix = dictionary.get("matrix")
        if matrix is not None:
            matrix = CSG._matrix(matrix)
            if matrix is None:
                return None, history_records
            role_nodes = {"source": index, "self": index}
            with CSG._lineage_capture(index, role_nodes, len(operands), history_records, capture_lineage):
                try:
                    result = Topology.Transform(result, matrix, transferDictionaries=True, silent=silent)
                except TypeError:
                    result = Topology.Transform(result, matrix, silent=silent)
        return result, history_records

    @staticmethod
    def _boolean_call(operation: str, a, b, silent: bool = False):
        """Execute one binary TopologicPy Boolean with provenance enabled.

        The public TopologicPy Boolean API uses ``tranDict``. The PythonOCC
        backend uses ``transferDictionary`` internally. CSG deliberately asks
        the public API to transfer dictionaries because that is also the path
        through which exact BRepTools history reaches the private CSG lineage
        capture hook. CSG control metadata remains on TGraph and is never
        copied onto result topology dictionaries.
        """
        from topologicpy.Topology import Topology

        mapping = {
            "union": "Union",
            "intersection": "Intersect",
            "difference": "Difference",
            "merge": "Merge",
            "impose": "Impose",
            "imprint": "Imprint",
            "slice": "Slice",
        }

        if operation == "xor":
            fn = (
                getattr(Topology, "SymmetricDifference", None)
                or getattr(Topology, "SymDif", None)
            )
        else:
            fn = getattr(Topology, mapping.get(operation, ""), None)

        if not callable(fn):
            return None

        # Public TopologicPy Boolean contract.
        try:
            return fn(a, b, tranDict=True, silent=silent)
        except TypeError:
            pass

        # Direct/backend compatibility fallbacks. Keep these after tranDict so
        # a public call cannot silently bypass exact provenance/history capture.
        try:
            return fn(a, b, transferDictionary=True, silent=silent)
        except TypeError:
            pass

        try:
            return fn(a, b, transferDictionaries=True, silent=silent)
        except TypeError:
            pass

        try:
            return fn(a, b, silent=silent)
        except TypeError:
            return fn(a, b)

    @staticmethod
    def _history_topologies(graph, operation, relation: str, topologyType: Optional[str]) -> list:
        records = CSG.History(graph, operation=operation, relation=relation, resultType=topologyType)
        return CSG._unique_topologies([record.get("result") for record in records if record.get("result") is not None])

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
    def _topology_identity_key(topology):
        shape = getattr(topology, "shape", None)
        try:
            return ("shape", hash(shape)) if shape is not None else ("object", id(topology))
        except Exception:
            return ("object", id(topology))

    @staticmethod
    def _unique_topologies(topologies: Iterable[Any]) -> list:
        buckets = {}
        result = []
        for topology in topologies or []:
            key = CSG._topology_identity_key(topology)
            bucket = buckets.setdefault(key, [])
            if any(CSG._same_topology(topology, existing) for existing in bucket):
                continue
            bucket.append(topology)
            result.append(topology)
        return result

    @staticmethod
    def _unique_origin_records(records: Iterable[dict]) -> list:
        result = []
        seen = set()
        for record in records or []:
            source = record.get("source")
            key = (record.get("sourceNode"), CSG._topology_identity_key(source))
            if key in seen:
                continue
            seen.add(key)
            result.append(record)
        return result
