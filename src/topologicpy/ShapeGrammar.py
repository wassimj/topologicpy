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


@dataclass(frozen=True)
class _Rule:
    index: int
    input: Any
    output: Any
    title: str
    description: str
    operation: str
    matrix: Optional[Tuple[Tuple[float, ...], ...]]
    parameters: Tuple[Tuple[str, Any], ...]
    metadata: Dict[str, Any]


@dataclass
class _CompiledRule:
    rule: _Rule
    topology_type: Any
    counts: Tuple[int, int, int, int]
    dictionary: Dict[str, Any]
    divide_tool: Any = None
    fingerprint: Any = None


@dataclass
class _ApplyCacheEntry:
    result: Any
    history: List[dict]


@dataclass
class _Runtime:
    generation: int = 0
    compiled_generation: int = -1
    compiled: Dict[int, _CompiledRule] = field(default_factory=dict)
    by_type: Dict[Any, List[int]] = field(default_factory=dict)
    match_cache: Dict[Any, Tuple[dict, ...]] = field(default_factory=dict)
    apply_cache: Dict[Any, _ApplyCacheEntry] = field(default_factory=dict)
    applications: Dict[int, dict] = field(default_factory=dict)
    next_application: int = 0
    match_cache_hits: int = 0
    match_cache_misses: int = 0
    apply_cache_hits: int = 0
    apply_cache_misses: int = 0
    similarity_tests: int = 0
    last_elapsed: float = 0.0
    last_application: Optional[int] = None


class ShapeGrammar:
    """A compiled, provenance-aware topology shape grammar.

    ``ShapeGrammar`` stores rules that match a topology pattern and apply a
    replacement, Boolean, transform, or procedural division operation to a
    matching target topology.

    The public API is backend-neutral. On the PythonOCC backend, rule application
    participates in TopologicPy's exact native provenance path: OCCT history is
    captured from the operation already being executed and BRepGraph is used,
    when available, as the exact final-result membership index. BRepGraph and
    OCCT objects never appear in this public API.

    Rules are compiled and indexed by topology type and structural counts.
    Matching therefore avoids calling ``Topology.IsSimilar`` for rules that can
    be rejected cheaply. Match results and rule applications are cached.

    Rule geometry is treated as immutable after ``AddRule``. If a rule input or
    output topology is mutated externally, call ``Invalidate`` before matching
    or applying rules again.

    Matrix semantics
    ----------------
    ``Topology.IsSimilar(rule_input, target)`` returns the matrix that maps the
    rule input to the target. For replacement and Boolean rules, ``matrix`` on a
    rule is an optional local preparation transform for the rule output/tool.
    The tool is transformed once by ``match_matrix @ rule_matrix`` and the
    operation is executed directly against the actual target topology.

    For ``Transform`` rules, the rule matrix is a transform expressed in the
    local coordinate frame of the rule input. It is conjugated into the matched
    target frame before being applied to the target.

    For ``Divide`` rules, division planes are compiled once in the coordinate
    frame of the rule input and mapped into the target by the match transform.
    """

    SCHEMA = "topologicpy.shapegrammar/2"

    _OPERATION_SPECS = {
        "replace": ("Replace", "Replace the matched target with the mapped rule output."),
        "transform": ("Transform", "Transform the matched target by a rule-local 4x4 matrix."),
        "union": ("Union", "Union the matched target with the mapped rule output."),
        "difference": ("Difference", "Subtract the mapped rule output from the matched target."),
        "xor": ("Symmetric Difference", "Compute the symmetric difference of target and mapped rule output."),
        "intersection": ("Intersect", "Intersect the matched target with the mapped rule output."),
        "merge": ("Merge", "Merge the matched target with the mapped rule output."),
        "slice": ("Slice", "Slice the matched target using the mapped rule output."),
        "impose": ("Impose", "Impose the mapped rule output on the matched target."),
        "imprint": ("Imprint", "Imprint the mapped rule output on the matched target."),
        "divide": ("Divide", "Divide the matched target along rule-local x, y and z axes."),
    }

    _ALIASES = {
        "replace": "replace",
        "transform": "transform",
        "union": "union",
        "fuse": "union",
        "difference": "difference",
        "subtract": "difference",
        "subtraction": "difference",
        "cut": "difference",
        "symmetricdifference": "xor",
        "symmetricaldifference": "xor",
        "symdif": "xor",
        "xor": "xor",
        "intersect": "intersection",
        "intersection": "intersection",
        "common": "intersection",
        "merge": "merge",
        "slice": "slice",
        "impose": "impose",
        "imprint": "imprint",
        "divide": "divide",
    }

    _OUTPUT_OPERATIONS = {
        "replace", "union", "difference", "xor", "intersection",
        "merge", "slice", "impose", "imprint",
    }

    def __init__(self, title: str = "Untitled", description: str = ""):
        self.title = str(title) if title is not None else "Untitled"
        self.description = str(description) if description is not None else ""
        self._rules: Dict[int, _Rule] = {}
        self._next_rule = 0
        self._runtime = _Runtime()

    # ------------------------------------------------------------------
    # Public rule/operation model
    # ------------------------------------------------------------------

    @property
    def rules(self) -> list:
        """Return public rule descriptors in insertion order."""
        return self.Rules()

    @property
    def operations(self) -> list:
        """Return public operation descriptors."""
        return self.Operations()

    def Operations(self) -> list:
        result = []
        for canonical, (title, description) in self._OPERATION_SPECS.items():
            item = {"title": title, "operation": canonical, "description": description}
            if canonical == "divide":
                item.update({"uSides": 2, "vSides": 2, "wSides": 2})
            result.append(item)
        return result

    def OperationTitles(self) -> list:
        return [self._OPERATION_SPECS[key][0] for key in self._OPERATION_SPECS]

    def OperationByTitle(self, title):
        if title is None:
            return None
        canonical = self._normalise_operation(title)
        if canonical is None:
            return None
        display, description = self._OPERATION_SPECS[canonical]
        result = {"title": display, "operation": canonical, "description": description}
        if canonical == "divide":
            result.update({"uSides": 2, "vSides": 2, "wSides": 2})
        return result

    def AddRule(
        self,
        input,
        output=None,
        title: str = "Untitled Rule",
        description: str = "",
        operation: Any = "Replace",
        matrix: Optional[Sequence[Sequence[float]]] = None,
        uSides: int = 2,
        vSides: int = 2,
        wSides: int = 2,
        metadata: Optional[dict] = None,
        silent: bool = False,
    ) -> Optional[int]:
        """Add a rule and return its stable integer rule identifier.

        ``input`` is the pattern topology. ``output`` is required by replacement
        and Boolean operations and is authored in the same rule-local coordinate
        frame as ``input``. ``Transform`` and ``Divide`` do not use ``output``.
        """
        return self._add_rule(
            index=None,
            input=input,
            output=output,
            title=title,
            description=description,
            operation=operation,
            matrix=matrix,
            uSides=uSides,
            vSides=vSides,
            wSides=wSides,
            metadata=metadata,
            silent=silent,
        )

    def Rule(self, rule) -> Optional[dict]:
        index = self._rule_index(rule)
        stored = self._rules.get(index) if index is not None else None
        return self._public_rule(stored) if stored is not None else None

    def Rules(self) -> list:
        return [self._public_rule(self._rules[index]) for index in sorted(self._rules)]

    def RemoveRule(self, rule, silent: bool = False) -> bool:
        index = self._rule_index(rule)
        if index is None or index not in self._rules:
            if not silent:
                print("ShapeGrammar.RemoveRule - Error: rule is not valid. Returning False.")
            return False
        del self._rules[index]
        self._touch_rules()
        return True

    def ClearRules(self) -> None:
        self._rules.clear()
        self._touch_rules()

    def Invalidate(self) -> None:
        """Invalidate compiled rule, match, and application caches.

        Call this after externally mutating topology dictionaries used by rules.
        Existing application history remains available.
        """
        self._touch_rules()

    def Compile(self, force: bool = False) -> dict:
        """Compile and index the current rules and return a compact summary."""
        if force:
            self._runtime.compiled_generation = -1
        self._ensure_compiled()
        return {
            "rules": len(self._rules),
            "compiledRules": len(self._runtime.compiled),
            "types": len(self._runtime.by_type),
            "generation": self._runtime.generation,
        }

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def ApplicableRules(
        self,
        topology,
        keys: Optional[Sequence[str]] = None,
        removeCoplanarFaces: bool = False,
        mantissa: int = 6,
        epsilon: float = 0.1,
        tolerance: float = 0.0001,
        cache: bool = True,
        silent: bool = False,
    ) -> Optional[list]:
        """Return deterministic match descriptors for all applicable rules.

        Each descriptor contains ``rule`` (the rule id), ``matrix`` (the
        rule-input -> target similarity transform), and basic rule metadata.
        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print("ShapeGrammar.ApplicableRules - Error: topology is not valid. Returning None.")
            return None
        try:
            epsilon = float(epsilon)
            tolerance = float(tolerance)
            mantissa = int(mantissa)
        except Exception:
            if not silent:
                print("ShapeGrammar.ApplicableRules - Error: invalid numeric matching parameters. Returning None.")
            return None
        if not 0.0 <= epsilon <= 1.0 or tolerance < 0.0:
            if not silent:
                print("ShapeGrammar.ApplicableRules - Error: epsilon/tolerance out of range. Returning None.")
            return None

        self._ensure_compiled()
        match_keys = tuple(key for key in (keys or []) if isinstance(key, str))
        target_fp = self._topology_fingerprint(topology)
        cache_key = (
            self._runtime.generation,
            target_fp,
            match_keys,
            bool(removeCoplanarFaces),
            mantissa,
            epsilon,
            tolerance,
        )
        if cache and cache_key in self._runtime.match_cache:
            self._runtime.match_cache_hits += 1
            return [self._copy_match(item) for item in self._runtime.match_cache[cache_key]]

        self._runtime.match_cache_misses += 1
        target_type = Topology.Type(topology)
        target_counts = self._structural_counts(topology)
        target_dictionary = self._python_dictionary(topology)
        candidates = self._runtime.by_type.get(target_type, [])
        result = []

        for rule_id in candidates:
            compiled = self._runtime.compiled[rule_id]
            if match_keys and not self._dictionary_matches(target_dictionary, compiled.dictionary, match_keys):
                continue
            if not removeCoplanarFaces and not self._counts_compatible(target_counts, compiled.counts, epsilon):
                continue

            self._runtime.similarity_tests += 1
            try:
                similar = Topology.IsSimilar(
                    compiled.rule.input,
                    topology,
                    removeCoplanarFaces=bool(removeCoplanarFaces),
                    mantissa=mantissa,
                    epsilon=epsilon,
                    tolerance=tolerance,
                    silent=True,
                )
            except TypeError:
                similar = Topology.IsSimilar(compiled.rule.input, topology)
            except Exception:
                similar = (False, None)

            if isinstance(similar, (list, tuple)) and len(similar) >= 2:
                status, matrix = bool(similar[0]), similar[1]
            else:
                status, matrix = bool(similar), None
            if not status:
                continue
            if matrix is None:
                matrix = self._identity_matrix()
            matrix = self._matrix(matrix)
            if matrix is None:
                continue
            result.append(self._match_descriptor(compiled.rule, matrix))

        if cache:
            self._runtime.match_cache[cache_key] = tuple(self._copy_match(item) for item in result)
        return result

    def Match(self, topology, **kwargs):
        """Alias for :meth:`ApplicableRules`."""
        return self.ApplicableRules(topology, **kwargs)

    # ------------------------------------------------------------------
    # Rule application
    # ------------------------------------------------------------------

    def ApplyRule(
        self,
        topology,
        rule,
        matrix: Optional[Sequence[Sequence[float]]] = None,
        keys: Optional[Sequence[str]] = None,
        removeCoplanarFaces: bool = False,
        mantissa: int = 6,
        epsilon: float = 0.1,
        tolerance: float = 0.0001,
        force: bool = False,
        cache: bool = True,
        lineage: bool = True,
        silent: bool = False,
    ):
        """Apply one rule to ``topology`` and return the resulting topology.

        ``rule`` may be a rule id, a descriptor returned by ``Rule``, or a match
        descriptor returned by ``ApplicableRules``. If ``matrix`` is omitted,
        the method computes the rule-input -> target match transform. Set
        ``force=True`` to apply with an identity match transform when the target
        does not match the rule input.
        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print("ShapeGrammar.ApplyRule - Error: topology is not valid. Returning None.")
            return None

        supplied_match = rule if isinstance(rule, dict) and "rule" in rule else None
        rule_id = self._rule_index(rule)
        stored = self._rules.get(rule_id) if rule_id is not None else None
        if stored is None:
            if not silent:
                print("ShapeGrammar.ApplyRule - Error: rule is not valid. Returning None.")
            return None

        if matrix is None and supplied_match is not None:
            matrix = supplied_match.get("matrix")
        if matrix is None:
            one = self._match_rule(
                topology,
                rule_id,
                keys=keys,
                removeCoplanarFaces=removeCoplanarFaces,
                mantissa=mantissa,
                epsilon=epsilon,
                tolerance=tolerance,
                silent=True,
            )
            if one is not None:
                matrix = one.get("matrix")
            elif force:
                matrix = self._identity_matrix()
            else:
                if not silent:
                    print("ShapeGrammar.ApplyRule - Error: rule is not applicable to topology. Returning None.")
                return None

        match_matrix = self._matrix(matrix)
        if match_matrix is None:
            if not silent:
                print("ShapeGrammar.ApplyRule - Error: matrix is not a valid 4x4 numeric matrix. Returning None.")
            return None

        self._ensure_compiled()
        compiled = self._runtime.compiled.get(rule_id)
        if compiled is None:
            return None

        application = self._runtime.next_application
        self._runtime.next_application += 1
        target_fp = self._topology_fingerprint(topology)
        apply_key = (
            self._runtime.generation,
            rule_id,
            compiled.fingerprint,
            target_fp,
            self._freeze(match_matrix),
            bool(lineage),
        )
        started = perf_counter()

        cached_entry = self._runtime.apply_cache.get(apply_key) if cache else None
        if cached_entry is not None:
            self._runtime.apply_cache_hits += 1
            result = cached_entry.result
            history = self._history_for_application(cached_entry.history, application, rule_id)
            cached = True
        else:
            self._runtime.apply_cache_misses += 1
            result, history = self._execute_rule(
                topology=topology,
                compiled=compiled,
                match_matrix=match_matrix,
                application=application,
                lineage=bool(lineage),
                tolerance=float(tolerance),
                silent=silent,
            )
            if result is None:
                return None
            cached = False
            if cache:
                self._runtime.apply_cache[apply_key] = _ApplyCacheEntry(
                    result=result,
                    history=self._history_for_application(history, -1, rule_id),
                )

        elapsed = perf_counter() - started
        record = {
            "application": application,
            "rule": rule_id,
            "title": stored.title,
            "operation": stored.operation,
            "input": topology,
            "result": result,
            "matrix": self._copy_matrix(match_matrix),
            "cached": cached,
            "elapsed": elapsed,
            "history": history,
        }
        self._runtime.applications[application] = record
        self._runtime.last_application = application
        self._runtime.last_elapsed = elapsed
        return result

    def ApplyMatch(self, topology, match: dict, **kwargs):
        """Apply a descriptor returned by :meth:`ApplicableRules`."""
        return self.ApplyRule(topology, match, **kwargs)

    def Apply(self, topology, rule, **kwargs):
        """Alias for :meth:`ApplyRule`."""
        return self.ApplyRule(topology, rule, **kwargs)

    def Application(self, application: Optional[int] = None) -> Optional[dict]:
        if application is None:
            application = self._runtime.last_application
        if not isinstance(application, int):
            return None
        record = self._runtime.applications.get(application)
        if record is None:
            return None
        result = {key: value for key, value in record.items() if key != "history"}
        result["historyCount"] = len(record.get("history") or [])
        return result

    # ------------------------------------------------------------------
    # Runtime status and exact lineage
    # ------------------------------------------------------------------

    def ClearRuntime(self, clearHistory: bool = False) -> None:
        """Clear match/application caches; optionally clear application history."""
        self._runtime.match_cache.clear()
        self._runtime.apply_cache.clear()
        self._runtime.match_cache_hits = 0
        self._runtime.match_cache_misses = 0
        self._runtime.apply_cache_hits = 0
        self._runtime.apply_cache_misses = 0
        self._runtime.similarity_tests = 0
        self._runtime.last_elapsed = 0.0
        if clearHistory:
            self._runtime.applications.clear()
            self._runtime.next_application = 0
            self._runtime.last_application = None

    def Status(self) -> dict:
        records = [
            history_record
            for application in self._runtime.applications.values()
            for history_record in (application.get("history") or [])
        ]
        return {
            "rules": len(self._rules),
            "compiledRules": len(self._runtime.compiled),
            "matchCacheEntries": len(self._runtime.match_cache),
            "applyCacheEntries": len(self._runtime.apply_cache),
            "matchCacheHits": self._runtime.match_cache_hits,
            "matchCacheMisses": self._runtime.match_cache_misses,
            "applyCacheHits": self._runtime.apply_cache_hits,
            "applyCacheMisses": self._runtime.apply_cache_misses,
            "similarityTests": self._runtime.similarity_tests,
            "applications": len(self._runtime.applications),
            "lineageRecords": len(records),
            "usedBRepGraph": any(bool(record.get("usedBRepGraph")) for record in records),
            "lastApplication": self._runtime.last_application,
            "lastElapsed": self._runtime.last_elapsed,
        }

    def History(
        self,
        application: Optional[int] = None,
        rule: Any = None,
        relation: Optional[str] = None,
        sourceRole: Optional[str] = None,
        resultType: Optional[str] = None,
    ) -> list:
        """Return backend-neutral exact lineage records from rule applications."""
        rule_id = self._rule_index(rule) if rule is not None else None
        relation_l = str(relation).lower() if relation is not None else None
        role_l = str(sourceRole).lower() if sourceRole is not None else None
        type_l = str(resultType).lower() if resultType is not None else None
        if application is None:
            applications = [self._runtime.applications[key] for key in sorted(self._runtime.applications)]
        else:
            record = self._runtime.applications.get(application)
            applications = [record] if record is not None else []

        output = []
        for app in applications:
            if rule_id is not None and app.get("rule") != rule_id:
                continue
            for record in app.get("history") or []:
                if relation_l is not None and str(record.get("relation", "")).lower() != relation_l:
                    continue
                if role_l is not None and str(record.get("sourceRole", "")).lower() != role_l:
                    continue
                if type_l is not None and str(record.get("resultType", "")).lower() != type_l:
                    continue
                output.append(dict(record))
        return output

    def GeneratedBy(self, application: Optional[int] = None, topologyType: Optional[str] = None) -> list:
        return self._history_results(application, "generated", topologyType)

    def ModifiedBy(self, application: Optional[int] = None, topologyType: Optional[str] = None) -> list:
        return self._history_results(application, "modified", topologyType)

    def UnchangedBy(self, application: Optional[int] = None, topologyType: Optional[str] = None) -> list:
        return self._history_results(application, "unchanged", topologyType)

    def DeletedBy(self, application: Optional[int] = None, topologyType: Optional[str] = None) -> list:
        records = self.History(application=application, relation="deleted")
        if topologyType is not None:
            t = str(topologyType).lower()
            records = [r for r in records if str(r.get("sourceType", "")).lower() == t]
        return self._unique_topologies([r.get("source") for r in records if r.get("source") is not None])

    def Origins(self, topology, application: Optional[int] = None) -> list:
        """
        Trace a result subtopology backwards to terminal rule-application sources.

        Identity/self-loop lineage records such as ``unchanged`` relationships are
        treated as terminal provenance steps rather than as upstream predecessors.
        This prevents a source topology from being incorrectly considered its own
        ancestor.

        Parameters
        ----------
        topology : topologicpy topology
            The result topology whose origins are to be traced.
        application : int, optional
            If specified, restrict the search to this rule application. If None,
            all recorded applications are searched.

        Returns
        -------
        list
            A list of dictionaries describing the terminal origins.
        """
        if topology is None:
            return []

        records = self.History(application=application)
        if not records:
            return []

        frontier = [topology]
        seen = set()
        origins = []

        def is_self_loop(record) -> bool:
            """Return True when a lineage record maps a topology to itself."""
            source = record.get("source")
            result = record.get("result")

            if source is None or result is None:
                return False

            return self._same_topology(source, result)

        def has_upstream_producer(source, current_record) -> bool:
            """
            Return True if ``source`` was produced by another non-trivial lineage
            record.

            Identity/unchanged records are deliberately ignored because they do
            not represent an earlier geometric derivation.
            """
            for prior in records:
                if prior is current_record:
                    continue

                prior_result = prior.get("result")
                if prior_result is None:
                    continue

                if not self._same_topology(prior_result, source):
                    continue

                # An unchanged/self-loop relationship does not constitute an
                # upstream producer. Following it would create a provenance cycle.
                if is_self_loop(prior):
                    continue

                return True

            return False

        while frontier:
            current = frontier.pop()

            if current is None:
                continue

            key = self._topology_identity_key(current)
            if key in seen:
                continue
            seen.add(key)

            incoming = [
                record
                for record in records
                if (
                    record.get("result") is not None
                    and self._same_topology(record.get("result"), current)
                )
            ]

            for record in incoming:
                source = record.get("source")
                if source is None:
                    continue

                if has_upstream_producer(source, record):
                    source_key = self._topology_identity_key(source)
                    if source_key not in seen:
                        frontier.append(source)
                    continue

                role = str(
                    record.get(
                        "sourceNode",
                        record.get("sourceRole", "source"),
                    )
                )

                if role == "target":
                    kind = "target"
                elif role == "ruleOutput":
                    kind = "ruleOutput"
                elif role == "divideTool":
                    kind = "proceduralTool"
                else:
                    kind = "source"

                origins.append(
                    {
                        "kind": kind,
                        "application": record.get("application"),
                        "rule": record.get("rule"),
                        "source": source,
                        "relation": record.get("relation"),
                    }
                )

        return self._unique_origin_records(origins)

    def Descendants(self, topology, application: Optional[int] = None) -> list:
        """Trace a source subtopology forward through captured lineage."""
        records = self.History(application=application)
        frontier = [topology]
        seen = set()
        output = []
        while frontier:
            current = frontier.pop()
            key = self._topology_identity_key(current)
            if key in seen:
                continue
            seen.add(key)
            for record in records:
                source = record.get("source")
                target = record.get("result")
                if source is None or not self._same_topology(source, current):
                    continue
                if target is not None:
                    output.append(target)
                    frontier.append(target)
        return self._unique_topologies(output)

    def LineageGraph(self, application: Optional[int] = None):
        """Return exact captured subtopology lineage as a derived ``TGraph``."""
        from topologicpy.TGraph import TGraph

        graph = TGraph(
            directed=True,
            allowSelfLoops=True,
            allowParallelEdges=True,
            dictionary={"type": "ShapeGrammarLineage", "schema": self.SCHEMA},
        )
        buckets = {}

        def vertex_for(topology, role, metadata):
            if topology is None:
                return graph.AddVertex(dictionary={"role": role, **metadata}, representation=None, silent=True)
            key = self._topology_identity_key(topology)
            bucket = buckets.setdefault(key, [])
            for existing, index in bucket:
                if self._same_topology(existing, topology):
                    return index
            index = graph.AddVertex(dictionary={"role": role, **metadata}, representation=topology, silent=True)
            bucket.append((topology, index))
            return index

        for record in self.History(application=application):
            source = record.get("source")
            target = record.get("result")
            s = vertex_for(source, "source", {
                "application": record.get("application"),
                "rule": record.get("rule"),
                "topologyType": record.get("sourceType"),
            })
            if target is None:
                t = vertex_for(None, "deleted", {
                    "application": record.get("application"),
                    "rule": record.get("rule"),
                })
            else:
                t = vertex_for(target, "result", {
                    "application": record.get("application"),
                    "rule": record.get("rule"),
                    "topologyType": record.get("resultType"),
                })
            graph.AddEdge(
                s,
                t,
                directed=True,
                dictionary={
                    "relation": record.get("relation"),
                    "operation": record.get("operation"),
                    "application": record.get("application"),
                    "rule": record.get("rule"),
                    "stage": record.get("stage"),
                    "usedBRepGraph": bool(record.get("usedBRepGraph")),
                },
                silent=True,
            )
        return graph

    def DerivationGraph(self):
        """Return the rule-application history as a topology-state ``TGraph``."""
        from topologicpy.TGraph import TGraph

        graph = TGraph(
            directed=True,
            allowSelfLoops=True,
            allowParallelEdges=True,
            dictionary={"type": "ShapeGrammarDerivation", "schema": self.SCHEMA},
        )
        buckets = {}

        def state_vertex(topology):
            key = self._topology_identity_key(topology)
            bucket = buckets.setdefault(key, [])
            for existing, index in bucket:
                if self._same_topology(existing, topology):
                    return index
            index = graph.AddVertex(
                dictionary={"role": "state", "topologyType": self._topology_type_name(topology)},
                representation=topology,
                silent=True,
            )
            bucket.append((topology, index))
            return index

        for application in sorted(self._runtime.applications):
            record = self._runtime.applications[application]
            source = state_vertex(record.get("input"))
            target = state_vertex(record.get("result"))
            graph.AddEdge(
                source,
                target,
                directed=True,
                dictionary={
                    "application": application,
                    "rule": record.get("rule"),
                    "title": record.get("title"),
                    "operation": record.get("operation"),
                    "cached": bool(record.get("cached")),
                    "elapsed": record.get("elapsed"),
                },
                silent=True,
            )
        return graph

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def Data(self, includeBREP: bool = True) -> dict:
        """Return a persistence-safe dictionary. Runtime caches/history are omitted."""
        from topologicpy.Topology import Topology

        rules = []
        for index in sorted(self._rules):
            rule = self._rules[index]
            item = {
                "id": rule.index,
                "title": rule.title,
                "description": rule.description,
                "operation": rule.operation,
                "matrix": self._copy_matrix(rule.matrix),
                "parameters": dict(rule.parameters),
                "metadata": copy.deepcopy(rule.metadata),
            }
            if includeBREP:
                item["inputBREP"] = Topology.BREPString(rule.input)
                item["outputBREP"] = Topology.BREPString(rule.output) if rule.output is not None else None
            rules.append(item)
        return {
            "schema": self.SCHEMA,
            "title": self.title,
            "description": self.description,
            "rules": rules,
        }

    @classmethod
    def ByData(cls, data: dict, silent: bool = False):
        from topologicpy.Topology import Topology

        if not isinstance(data, dict) or data.get("schema") != cls.SCHEMA:
            if not silent:
                print("ShapeGrammar.ByData - Error: invalid ShapeGrammar data. Returning None.")
            return None
        grammar = cls(title=data.get("title", "Untitled"), description=data.get("description", ""))
        for item in sorted(data.get("rules") or [], key=lambda x: x.get("id", -1)):
            input_brep = item.get("inputBREP")
            if not isinstance(input_brep, str) or not input_brep:
                if not silent:
                    print("ShapeGrammar.ByData - Error: rule input BREP is missing. Returning None.")
                return None
            rule_input = Topology.ByBREPString(input_brep, silent=True)
            output_brep = item.get("outputBREP")
            rule_output = Topology.ByBREPString(output_brep, silent=True) if isinstance(output_brep, str) and output_brep else None
            params = item.get("parameters") or {}
            created = grammar._add_rule(
                index=item.get("id"),
                input=rule_input,
                output=rule_output,
                title=item.get("title", "Untitled Rule"),
                description=item.get("description", ""),
                operation=item.get("operation", "replace"),
                matrix=item.get("matrix"),
                uSides=params.get("uSides", 2),
                vSides=params.get("vSides", 2),
                wSides=params.get("wSides", 2),
                metadata=item.get("metadata") or {},
                silent=silent,
            )
            if created is None:
                return None
        return grammar

    def JSONString(self, indent: Optional[int] = None) -> Optional[str]:
        try:
            return json.dumps(self.Data(includeBREP=True), indent=indent)
        except Exception:
            return None

    @classmethod
    def ByJSONString(cls, string: str, silent: bool = False):
        try:
            data = json.loads(string)
        except Exception:
            return None
        return cls.ByData(data, silent=silent)

    def Export(self, path: str, indent: int = 2, overwrite: bool = False, silent: bool = False) -> Optional[bool]:
        if not isinstance(path, str) or not path.strip():
            return None
        import os
        if os.path.exists(path) and not overwrite:
            if not silent:
                print("ShapeGrammar.Export - Error: path already exists and overwrite is False. Returning None.")
            return None
        string = self.JSONString(indent=indent)
        if string is None:
            return None
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(string)
            return True
        except Exception:
            return None

    @classmethod
    def ByPath(cls, path: str, silent: bool = False):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return cls.ByJSONString(handle.read(), silent=silent)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    def ClusterByInputOutput(self, input, output, silent: bool = False):
        from topologicpy.Vertex import Vertex
        from topologicpy.Cell import Cell
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Cluster import Cluster

        if not Topology.IsInstance(input, "Topology") or not Topology.IsInstance(output, "Topology"):
            if not silent:
                print("ShapeGrammar.ClusterByInputOutput - Error: input/output is not a valid topology. Returning None.")
            return None

        def scaled_copy(topology, x_offset):
            bb = Topology.BoundingBox(topology)
            if bb is None:
                return None
            centroid = Topology.Centroid(bb)
            d = Topology.Dictionary(bb)
            xmin = Dictionary.ValueAtKey(d, "xmin", 0)
            ymin = Dictionary.ValueAtKey(d, "ymin", 0)
            zmin = Dictionary.ValueAtKey(d, "zmin", 0)
            xmax = Dictionary.ValueAtKey(d, "xmax", xmin)
            ymax = Dictionary.ValueAtKey(d, "ymax", ymin)
            zmax = Dictionary.ValueAtKey(d, "zmax", zmin)
            extent = max(float(xmax) - float(xmin), float(ymax) - float(ymin), float(zmax) - float(zmin))
            scale = 1.0 / extent if extent > 0.0 else 1.0
            item = Topology.Translate(topology, -Vertex.X(centroid), -Vertex.Y(centroid), -Vertex.Z(centroid))
            item = Topology.Scale(item, x=scale, y=scale, z=scale)
            return Topology.Translate(item, x_offset, 0, 0)

        left = scaled_copy(input, 0.5)
        right = scaled_copy(output, 2.5)
        if left is None or right is None:
            return None
        cylinder = Cell.Cylinder(radius=0.04, height=0.4, placement="bottom")
        cylinder = Topology.Rotate(cylinder, axis=[0, 1, 0], angle=90)
        cylinder = Topology.Translate(cylinder, 1.25, 0, 0)
        cone = Cell.Cone(baseRadius=0.1, topRadius=0, height=0.15, placement="bottom")
        cone = Topology.Rotate(cone, axis=[0, 1, 0], angle=90)
        cone = Topology.Translate(cone, 1.65, 0, 0)
        cluster = Cluster.ByTopologies([left, right, cylinder, cone])
        return Topology.Place(cluster, originA=Topology.Centroid(cluster), originB=Vertex.Origin())

    def ClusterByRule(self, rule, silent: bool = False):
        descriptor = self.Rule(rule)
        if descriptor is None:
            if not silent:
                print("ShapeGrammar.ClusterByRule - Error: rule is not valid. Returning None.")
            return None
        self._ensure_compiled()
        compiled = self._runtime.compiled.get(descriptor["id"])
        if compiled is None:
            return None
        output, _history = self._execute_rule(
            topology=descriptor["input"],
            compiled=compiled,
            match_matrix=self._identity_matrix(),
            application=-1,
            lineage=False,
            tolerance=0.0001,
            silent=silent,
        )
        if output is None:
            return None
        return self.ClusterByInputOutput(descriptor["input"], output, silent=silent)

    def FigureByInputOutput(self, input, output, silent: bool = False):
        from topologicpy.Plotly import Plotly
        cluster = self.ClusterByInputOutput(input, output, silent=silent)
        if cluster is None:
            return None
        return Plotly.FigureByData(Plotly.DataByTopology(cluster))

    def FigureByRule(self, rule, silent: bool = False):
        cluster = self.ClusterByRule(rule, silent=silent)
        if cluster is None:
            return None
        from topologicpy.Plotly import Plotly
        return Plotly.FigureByData(Plotly.DataByTopology(cluster))

    # ------------------------------------------------------------------
    # Private rule compilation / matching helpers
    # ------------------------------------------------------------------

    def _add_rule(
        self,
        *,
        index: Optional[int],
        input,
        output,
        title,
        description,
        operation,
        matrix,
        uSides,
        vSides,
        wSides,
        metadata,
        silent,
    ) -> Optional[int]:
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(input, "Topology"):
            if not silent:
                print("ShapeGrammar.AddRule - Error: input is not a valid topology. Returning None.")
            return None
        canonical = self._normalise_operation(operation)
        if canonical is None:
            if not silent:
                print("ShapeGrammar.AddRule - Error: operation is not valid. Returning None.")
            return None

        if canonical in self._OUTPUT_OPERATIONS:
            if not Topology.IsInstance(output, "Topology"):
                if not silent:
                    print("ShapeGrammar.AddRule - Error: this operation requires a valid output topology. Returning None.")
                return None
        elif output is not None:
            if not silent:
                print("ShapeGrammar.AddRule - Error: Transform/Divide rules do not accept an output topology. Returning None.")
            return None

        matrix_value = self._matrix(matrix) if matrix is not None else None
        if matrix is not None and matrix_value is None:
            if not silent:
                print("ShapeGrammar.AddRule - Error: matrix is not a valid 4x4 numeric matrix. Returning None.")
            return None
        if canonical == "transform" and matrix_value is None:
            if not silent:
                print("ShapeGrammar.AddRule - Error: Transform requires a matrix. Returning None.")
            return None

        try:
            u = int(uSides)
            v = int(vSides)
            w = int(wSides)
        except Exception:
            if not silent:
                print("ShapeGrammar.AddRule - Error: divide side counts must be integers. Returning None.")
            return None
        if min(u, v, w) < 1:
            if not silent:
                print("ShapeGrammar.AddRule - Error: divide side counts must be >= 1. Returning None.")
            return None

        metadata_value = copy.deepcopy(metadata or {})
        if not isinstance(metadata_value, dict):
            if not silent:
                print("ShapeGrammar.AddRule - Error: metadata must be a dictionary. Returning None.")
            return None
        try:
            json.dumps(metadata_value)
        except Exception:
            if not silent:
                print("ShapeGrammar.AddRule - Error: metadata must be JSON-serialisable. Returning None.")
            return None

        if index is None:
            index = self._next_rule
            self._next_rule += 1
        else:
            try:
                index = int(index)
            except Exception:
                return None
            if index in self._rules:
                return None
            self._next_rule = max(self._next_rule, index + 1)

        params = (("uSides", u), ("vSides", v), ("wSides", w)) if canonical == "divide" else tuple()
        stored = _Rule(
            index=index,
            input=input,
            output=output,
            title=str(title) if title is not None else "Untitled Rule",
            description=str(description) if description is not None else "",
            operation=canonical,
            matrix=self._matrix_tuple(matrix_value),
            parameters=params,
            metadata=metadata_value,
        )
        self._rules[index] = stored
        self._touch_rules()
        return index

    def _touch_rules(self):
        self._runtime.generation += 1
        self._runtime.compiled_generation = -1
        self._runtime.compiled.clear()
        self._runtime.by_type.clear()
        self._runtime.match_cache.clear()
        self._runtime.apply_cache.clear()

    def _ensure_compiled(self):
        if self._runtime.compiled_generation == self._runtime.generation:
            return
        from topologicpy.Topology import Topology

        compiled = {}
        by_type = {}
        for index in sorted(self._rules):
            rule = self._rules[index]
            topology_type = Topology.Type(rule.input)
            counts = self._structural_counts(rule.input)
            dictionary = self._python_dictionary(rule.input)
            divide_tool = self._divide_tool(rule) if rule.operation == "divide" else None
            fingerprint = (
                self._topology_fingerprint(rule.input),
                self._topology_fingerprint(rule.output) if rule.output is not None else None,
                rule.operation,
                self._freeze(rule.matrix),
                self._freeze(rule.parameters),
            )
            item = _CompiledRule(
                rule=rule,
                topology_type=topology_type,
                counts=counts,
                dictionary=dictionary,
                divide_tool=divide_tool,
                fingerprint=fingerprint,
            )
            compiled[index] = item
            by_type.setdefault(topology_type, []).append(index)
        self._runtime.compiled = compiled
        self._runtime.by_type = by_type
        self._runtime.compiled_generation = self._runtime.generation

    def _match_rule(self, topology, rule_id: int, **kwargs) -> Optional[dict]:
        self._ensure_compiled()
        compiled = self._runtime.compiled.get(rule_id)
        if compiled is None:
            return None
        from topologicpy.Topology import Topology

        try:
            epsilon = float(kwargs.get("epsilon", 0.1))
            tolerance = float(kwargs.get("tolerance", 0.0001))
            mantissa = int(kwargs.get("mantissa", 6))
        except Exception:
            return None
        remove_coplanar = bool(kwargs.get("removeCoplanarFaces", False))
        keys = tuple(k for k in (kwargs.get("keys") or []) if isinstance(k, str))
        if Topology.Type(topology) != compiled.topology_type:
            return None
        if keys and not self._dictionary_matches(self._python_dictionary(topology), compiled.dictionary, keys):
            return None
        if not remove_coplanar and not self._counts_compatible(self._structural_counts(topology), compiled.counts, epsilon):
            return None
        self._runtime.similarity_tests += 1
        try:
            similar = Topology.IsSimilar(
                compiled.rule.input,
                topology,
                removeCoplanarFaces=remove_coplanar,
                mantissa=mantissa,
                epsilon=epsilon,
                tolerance=tolerance,
                silent=True,
            )
        except TypeError:
            similar = Topology.IsSimilar(compiled.rule.input, topology)
        except Exception:
            return None
        if isinstance(similar, (list, tuple)) and len(similar) >= 2:
            status, matrix = bool(similar[0]), similar[1]
        else:
            status, matrix = bool(similar), None
        if not status:
            return None
        matrix = self._matrix(matrix if matrix is not None else self._identity_matrix())
        return self._match_descriptor(compiled.rule, matrix) if matrix is not None else None

    # ------------------------------------------------------------------
    # Private execution / provenance helpers
    # ------------------------------------------------------------------

    def _execute_rule(self, topology, compiled, match_matrix, application, lineage, tolerance, silent):
        from topologicpy.Topology import Topology
        from topologicpy.Matrix import Matrix

        rule = compiled.rule
        sink = []
        stage = 0

        def capture(role_nodes):
            nonlocal stage
            current = stage
            stage += 1
            if not lineage:
                return nullcontext()
            try:
                from topologicpy.pythonocc_backend._csg_lineage import capture as lineage_capture
                return lineage_capture(application, role_nodes, stage=current, sink=sink)
            except Exception:
                return nullcontext()

        def transform(source, matrix_value, role):
            if source is None:
                return None
            if self._is_identity_matrix(matrix_value):
                return source
            with capture({"source": role, "self": role}):
                try:
                    return Topology.Transform(source, matrix_value, transferDictionaries=True, silent=silent)
                except TypeError:
                    try:
                        return Topology.Transform(source, matrix_value, silent=silent)
                    except TypeError:
                        return Topology.Transform(source, matrix_value)

        if rule.operation == "transform":
            local = self._copy_matrix(rule.matrix)
            inverse = Matrix.Invert(match_matrix, silent=True)
            if inverse is None:
                return None, []
            world = Matrix.Multiply(Matrix.Multiply(match_matrix, local), inverse)
            result = transform(topology, world, "target")

        elif rule.operation == "divide":
            tool = compiled.divide_tool
            if tool is None:
                result = topology
                if lineage:
                    sink.extend(self._identity_lineage_records(
                        topology, application, "target", "Divide", stage
                    ))
            else:
                tool_matrix = self._compose_matrices(match_matrix, self._copy_matrix(rule.matrix))
                aligned_tool = transform(tool, tool_matrix, "divideTool")
                if aligned_tool is None:
                    return None, []
                with capture({"self": "target", "source": "target", "other": "divideTool", "tool": "divideTool"}):
                    result = self._boolean_call("slice", topology, aligned_tool, silent=silent)
        else:
            tool = rule.output
            tool_matrix = self._compose_matrices(match_matrix, self._copy_matrix(rule.matrix))
            aligned_tool = transform(tool, tool_matrix, "ruleOutput")
            if aligned_tool is None:
                return None, []
            if rule.operation == "replace":
                result = aligned_tool
                if lineage and aligned_tool is tool:
                    sink.extend(self._identity_lineage_records(
                        tool, application, "ruleOutput", "Replace", stage
                    ))
            else:
                with capture({"self": "target", "source": "target", "other": "ruleOutput", "tool": "ruleOutput"}):
                    result = self._boolean_call(rule.operation, topology, aligned_tool, silent=silent)

        if result is None:
            return None, []
        return result, self._materialise_history(sink, application, rule.index)

    def _boolean_call(self, operation: str, a, b, silent: bool = False):
        """Execute one binary TopologicPy Boolean with exact provenance enabled.

        The public TopologicPy Boolean API uses ``tranDict``. On the PythonOCC
        backend that public flag reaches the native ``transferDictionary`` path,
        which is also where BRepTools history is exposed to the private lineage
        capture hook. The fallback spellings are retained only for backend/API
        compatibility; ``tranDict=True`` is deliberately attempted first.
        """
        from topologicpy.Topology import Topology

        names = {
            "union": "Union",
            "difference": "Difference",
            "intersection": "Intersect",
            "merge": "Merge",
            "slice": "Slice",
            "impose": "Impose",
            "imprint": "Imprint",
        }

        if operation == "xor":
            fn = (
                getattr(Topology, "SymmetricDifference", None)
                or getattr(Topology, "SymDif", None)
            )
        else:
            fn = getattr(Topology, names.get(operation, ""), None)

        if not callable(fn):
            return None

        try:
            return fn(a, b, tranDict=True, silent=silent)
        except TypeError:
            pass
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

    def _materialise_history(self, records: list, application: int, rule: int) -> list:
        output = []
        for record in records or []:
            if record.get("_synthetic"):
                public = {
                    key: value for key, value in record.items()
                    if key not in {"_synthetic"}
                }
            else:
                try:
                    from topologicpy.pythonocc_backend._csg_lineage import materialise_record
                    public = materialise_record(record)
                except Exception:
                    public = {
                        key: value for key, value in record.items()
                        if not str(key).startswith("_")
                    }
                    public.setdefault("source", None)
                    public.setdefault("result", None)
            public["application"] = application
            public["rule"] = rule
            public["operationNode"] = application
            output.append(public)
        return output

    def _identity_lineage_records(self, topology, application, source_node, operation, stage):
        """Create exact unchanged records when an operation reuses geometry verbatim."""
        from topologicpy.Topology import Topology

        items = [topology]
        for name in ("Vertices", "Edges", "Wires", "Faces", "Shells", "Cells"):
            method = getattr(Topology, name, None)
            if not callable(method):
                continue
            try:
                values = method(topology, silent=True) or []
            except TypeError:
                try:
                    values = method(topology) or []
                except Exception:
                    values = []
            except Exception:
                values = []
            items.extend(values)
        items = self._unique_topologies(items)
        return [
            self._synthetic_private_record(
                application, source_node, operation, "unchanged", item, item, stage
            )
            for item in items
        ]

    def _synthetic_private_record(self, application, source_node, operation, relation, source, result, stage):
        return {
            "_synthetic": True,
            "application": application,
            "operationNode": application,
            "sourceNode": source_node,
            "sourceRole": source_node,
            "stage": stage,
            "operation": operation,
            "relation": relation,
            "source": source,
            "result": result,
            "sourceType": self._topology_type_name(source),
            "resultType": self._topology_type_name(result),
            "usedBRepGraph": False,
        }

    def _history_for_application(self, history, application, rule):
        result = []
        for record in history or []:
            item = dict(record)
            old_operation = item.get("operationNode")
            if item.get("sourceNode") == old_operation:
                item["sourceNode"] = application
            item["application"] = application
            item["rule"] = rule
            item["operationNode"] = application
            result.append(item)
        return result

    # ------------------------------------------------------------------
    # Private low-level helpers
    # ------------------------------------------------------------------

    def _normalise_operation(self, operation) -> Optional[str]:
        if operation is None:
            return "replace"
        if isinstance(operation, dict):
            operation = operation.get("operation", operation.get("title"))
        if not isinstance(operation, str):
            return None
        key = "".join(ch for ch in operation.strip().lower() if ch.isalnum())
        return self._ALIASES.get(key)

    def _rule_index(self, rule) -> Optional[int]:
        if isinstance(rule, int) and not isinstance(rule, bool):
            return rule
        if isinstance(rule, dict):
            value = rule.get("rule", rule.get("id"))
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        return None

    def _public_rule(self, rule: _Rule) -> dict:
        display = self._OPERATION_SPECS[rule.operation][0]
        params = dict(rule.parameters)
        return {
            "id": rule.index,
            "input": rule.input,
            "output": rule.output,
            "title": rule.title,
            "description": rule.description,
            "operation": display,
            "canonicalOperation": rule.operation,
            "matrix": self._copy_matrix(rule.matrix),
            "uSides": params.get("uSides"),
            "vSides": params.get("vSides"),
            "wSides": params.get("wSides"),
            "metadata": copy.deepcopy(rule.metadata),
        }

    def _match_descriptor(self, rule: _Rule, matrix) -> dict:
        return {
            "rule": rule.index,
            "title": rule.title,
            "operation": self._OPERATION_SPECS[rule.operation][0],
            "canonicalOperation": rule.operation,
            "matrix": self._copy_matrix(matrix),
            "metadata": copy.deepcopy(rule.metadata),
        }

    @staticmethod
    def _copy_match(match: dict) -> dict:
        result = dict(match)
        result["matrix"] = ShapeGrammar._copy_matrix(match.get("matrix"))
        result["metadata"] = copy.deepcopy(match.get("metadata") or {})
        return result

    @staticmethod
    def _is_4x4_matrix(matrix) -> bool:
        return ShapeGrammar._matrix(matrix) is not None

    @staticmethod
    def _matrix(matrix):
        if not isinstance(matrix, (list, tuple)) or len(matrix) != 4:
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
    def _matrix_tuple(matrix):
        if matrix is None:
            return None
        return tuple(tuple(float(value) for value in row) for row in matrix)

    @staticmethod
    def _copy_matrix(matrix):
        if matrix is None:
            return None
        return [[float(value) for value in row] for row in matrix]

    @staticmethod
    def _identity_matrix():
        return [[1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]]

    @staticmethod
    def _is_identity_matrix(matrix, tolerance: float = 1e-12) -> bool:
        if matrix is None:
            return True
        identity = ShapeGrammar._identity_matrix()
        try:
            return all(abs(float(matrix[i][j]) - identity[i][j]) <= tolerance for i in range(4) for j in range(4))
        except Exception:
            return False

    @staticmethod
    def _compose_matrices(match_matrix, preparation_matrix):
        if preparation_matrix is None:
            return ShapeGrammar._copy_matrix(match_matrix)
        if match_matrix is None:
            return ShapeGrammar._copy_matrix(preparation_matrix)
        try:
            from topologicpy.Matrix import Matrix
            return Matrix.Multiply(match_matrix, preparation_matrix)
        except Exception:
            a = match_matrix
            b = preparation_matrix
            return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]

    @staticmethod
    def _structural_counts(topology) -> Tuple[int, int, int, int]:
        from topologicpy.Topology import Topology
        def count(method):
            try:
                return len(method(topology, silent=True) or [])
            except TypeError:
                try:
                    return len(method(topology) or [])
                except Exception:
                    return 0
            except Exception:
                return 0
        return (
            count(Topology.Vertices),
            count(Topology.Edges),
            count(Topology.Faces),
            count(Topology.Cells),
        )

    @staticmethod
    def _counts_compatible(a, b, epsilon: float) -> bool:
        for x, y in zip(a, b):
            maximum = max(x, y)
            if maximum > 0 and abs(x - y) / maximum > epsilon:
                return False
        return True

    @staticmethod
    def _python_dictionary(topology) -> dict:
        if topology is None:
            return {}
        try:
            from topologicpy.Topology import Topology
            from topologicpy.Dictionary import Dictionary
            d = Topology.Dictionary(topology, silent=True)
            return Dictionary.PythonDictionary(d, silent=True) or {}
        except TypeError:
            try:
                d = Topology.Dictionary(topology)
                return Dictionary.PythonDictionary(d) or {}
            except Exception:
                return {}
        except Exception:
            return {}

    @staticmethod
    def _dictionary_matches(target: dict, pattern: dict, keys: Sequence[str]) -> bool:
        return all(target.get(key, None) == pattern.get(key, None) for key in keys)

    def _topology_fingerprint(self, topology):
        if topology is None:
            return None
        shape = getattr(topology, "shape", None)
        try:
            shape_hash = hash(shape) if shape is not None else hash(topology)
        except Exception:
            shape_hash = id(shape) if shape is not None else id(topology)
        try:
            shape_id = id(shape) if shape is not None else id(topology)
        except Exception:
            shape_id = id(topology)
        try:
            from topologicpy.Topology import Topology
            topology_type = Topology.Type(topology)
        except Exception:
            topology_type = topology.__class__.__name__
        return (shape_id, shape_hash, topology_type, self._freeze(self._python_dictionary(topology)))

    @staticmethod
    def _freeze(value):
        if isinstance(value, dict):
            return tuple(sorted((str(k), ShapeGrammar._freeze(v)) for k, v in value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(ShapeGrammar._freeze(v) for v in value)
        if isinstance(value, set):
            return tuple(sorted(ShapeGrammar._freeze(v) for v in value))
        try:
            hash(value)
            return value
        except Exception:
            return repr(value)

    def _divide_tool(self, rule: _Rule):
        from topologicpy.Topology import Topology
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster

        params = dict(rule.parameters)
        u_sides = int(params.get("uSides", 1))
        v_sides = int(params.get("vSides", 1))
        w_sides = int(params.get("wSides", 1))
        vertices = Topology.Vertices(rule.input, silent=True) or []
        if not vertices:
            return None
        try:
            xs = [float(Vertex.X(v)) for v in vertices]
            ys = [float(Vertex.Y(v)) for v in vertices]
            zs = [float(Vertex.Z(v)) for v in vertices]
        except Exception:
            return None
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        zmin, zmax = min(zs), max(zs)
        dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
        tolerance = 1e-12
        cx, cy, cz = xmin + 0.5 * dx, ymin + 0.5 * dy, zmin + 0.5 * dz
        faces = []

        if w_sides > 1 and abs(dz) > tolerance:
            origin = Vertex.ByCoordinates(cx, cy, zmin)
            base = Face.Rectangle(origin=origin, width=max(dx * 1.1, tolerance), length=max(dy * 1.1, tolerance))
            for i in range(1, w_sides):
                faces.append(Topology.Translate(base, 0, 0, dz * i / w_sides))
        if u_sides > 1 and abs(dx) > tolerance:
            origin = Vertex.ByCoordinates(xmin, cy, cz)
            base = Face.Rectangle(origin=origin, width=max(dz * 1.1, tolerance), length=max(dy * 1.1, tolerance), direction=[1, 0, 0])
            for i in range(1, u_sides):
                faces.append(Topology.Translate(base, dx * i / u_sides, 0, 0))
        if v_sides > 1 and abs(dy) > tolerance:
            origin = Vertex.ByCoordinates(cx, ymin, cz)
            base = Face.Rectangle(origin=origin, width=max(dx * 1.1, tolerance), length=max(dz * 1.1, tolerance), direction=[0, 1, 0])
            for i in range(1, v_sides):
                faces.append(Topology.Translate(base, 0, dy * i / v_sides, 0))
        return Cluster.ByTopologies(faces) if faces else None

    def _history_results(self, application, relation, topology_type):
        records = self.History(application=application, relation=relation, resultType=topology_type)
        return self._unique_topologies([record.get("result") for record in records if record.get("result") is not None])

    @staticmethod
    def _same_topology(a, b) -> bool:
        if a is b:
            return True
        if a is None or b is None:
            return False
        try:
            from topologicpy.Topology import Topology
            return bool(Topology.IsSame(a, b, silent=True))
        except TypeError:
            try:
                return bool(Topology.IsSame(a, b))
            except Exception:
                return False
        except Exception:
            return False

    @staticmethod
    def _topology_identity_key(topology):
        if topology is None:
            return (None, None)
        shape = getattr(topology, "shape", None)
        try:
            return (topology.__class__.__name__, hash(shape) if shape is not None else hash(topology))
        except Exception:
            return (topology.__class__.__name__, id(shape) if shape is not None else id(topology))

    def _unique_topologies(self, topologies: Iterable[Any]) -> list:
        output = []
        buckets = {}
        for topology in topologies:
            if topology is None:
                continue
            key = self._topology_identity_key(topology)
            bucket = buckets.setdefault(key, [])
            if any(self._same_topology(topology, existing) for existing in bucket):
                continue
            bucket.append(topology)
            output.append(topology)
        return output

    def _unique_origin_records(self, records: Iterable[dict]) -> list:
        output = []
        buckets = set()
        for record in records:
            source = record.get("source")
            key = (
                record.get("kind"),
                record.get("application"),
                record.get("rule"),
                self._topology_identity_key(source),
            )
            if key in buckets:
                continue
            buckets.add(key)
            output.append(record)
        return output

    @staticmethod
    def _topology_type_name(topology) -> Optional[str]:
        if topology is None:
            return None
        name = topology.__class__.__name__
        known = {"Vertex", "Edge", "Wire", "Face", "Shell", "Cell", "CellComplex", "Cluster"}
        if name in known:
            return name
        try:
            from topologicpy.Topology import Topology
            for candidate in ("Vertex", "Edge", "Wire", "Face", "Shell", "Cell", "CellComplex", "Cluster"):
                if Topology.IsInstance(topology, candidate):
                    return candidate
        except Exception:
            pass
        return name


__all__ = ["ShapeGrammar"]
