# Copyright (C) 2026
# Wassim Jabi <wassimj@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free Software
# Foundation, either version 3.0 of the License, or (at your option) any later version.

from __future__ import annotations

from typing import Any, Iterable, Optional
import copy


class Provenance:
    """
    Backend-neutral provenance/lineage container for TopologicPy operations.

    ``History()`` returns the exact captured source -> result records emitted by
    the topology kernel. ``Records()`` returns a semantic view in which internal
    implementation stages (for example same-domain unification after a Boolean)
    are collapsed while meaningful operation/application stages are retained.

    The semantic ``Graph()`` is a derived TGraph whose vertex representations
    are the actual source/result TopologicPy (sub)topologies. Relationship and
    operation metadata live on directed edges.
    """

    SCHEMA = "topologicpy.provenance/1"

    _RELATION_PRECEDENCE = {
        "unchanged": 0,
        "modified": 1,
        "generated": 2,
        "deleted": 3,
    }

    def __init__(
        self,
        records: Optional[Iterable[dict]] = None,
        *,
        operation: Optional[str] = None,
        sources: Optional[dict] = None,
        result: Any = None,
        supported: bool = True,
        metadata: Optional[dict] = None,
    ):
        self._history = [
            copy.copy(record)
            for record in (records or [])
            if isinstance(record, dict)
        ]
        self.operation = str(operation) if operation is not None else None
        self.sources = dict(sources or {})
        self.result = result
        self.supported = bool(supported)
        self.metadata = copy.deepcopy(metadata) if isinstance(metadata, dict) else {}

    @staticmethod
    def ByRecords(
        records: Optional[Iterable[dict]] = None,
        *,
        operation: Optional[str] = None,
        sources: Optional[dict] = None,
        result: Any = None,
        supported: bool = True,
        metadata: Optional[dict] = None,
    ) -> "Provenance":
        """Create a provenance object from already-materialised lineage records."""
        return Provenance(
            records,
            operation=operation,
            sources=sources,
            result=result,
            supported=supported,
            metadata=metadata,
        )

    @staticmethod
    def _same(a, b) -> bool:
        if a is b:
            return True
        if a is None or b is None:
            return False
        try:
            from topologicpy.Topology import Topology
            return bool(Topology.IsSame(a, b))
        except Exception:
            return False

    @staticmethod
    def _identity_key(topology):
        if topology is None:
            return ("none", None)
        shape = getattr(topology, "shape", None)
        try:
            if shape is not None:
                return ("shape", topology.__class__.__name__, hash(shape))
        except Exception:
            pass
        return ("object", topology.__class__.__name__, id(topology))

    @staticmethod
    def _type_name(topology) -> Optional[str]:
        if topology is None:
            return None
        try:
            from topologicpy.Topology import Topology
            return Topology.TypeAsString(topology)
        except Exception:
            return topology.__class__.__name__

    @staticmethod
    def _group_key(record: dict):
        if record.get("operationNode") is not None:
            return ("operationNode", record.get("operationNode"))
        if record.get("application") is not None:
            return ("application", record.get("application"))
        return ("direct", 0)

    @staticmethod
    def _combine_relation(current: Optional[str], new: Optional[str]) -> str:
        current = str(current or "unchanged").lower()
        new = str(new or "unchanged").lower()
        if Provenance._RELATION_PRECEDENCE.get(new, 1) > Provenance._RELATION_PRECEDENCE.get(current, 1):
            return new
        return current

    @staticmethod
    def _copy_public_metadata(record: dict) -> dict:
        keys = (
            "operationNode",
            "sourceNode",
            "sourceRole",
            "operation",
            "application",
            "rule",
            "title",
            "usedBRepGraph",
        )
        return {key: record.get(key) for key in keys if record.get(key) is not None}

    @staticmethod
    def _find_matches(records: list, field: str, topology) -> list:
        return [
            record for record in records
            if record.get(field) is not None
            and Provenance._same(record.get(field), topology)
        ]

    @staticmethod
    def _is_internal_role(role) -> bool:
        return str(role or "").lower() in {"source", "intermediate", "internal"}

    def History(
        self,
        *,
        relation: Optional[str] = None,
        topologyType: Optional[str] = None,
        sourceRole: Optional[str] = None,
    ) -> list:
        """Return exact captured lineage records, optionally filtered."""
        relation_l = str(relation).lower() if relation is not None else None
        type_l = str(topologyType).lower() if topologyType is not None else None
        role_l = str(sourceRole).lower() if sourceRole is not None else None

        output = []
        for record in self._history:
            if relation_l is not None and str(record.get("relation", "")).lower() != relation_l:
                continue
            if role_l is not None and str(record.get("sourceRole", "")).lower() != role_l:
                continue
            if type_l is not None:
                result_type = str(record.get("resultType") or self._type_name(record.get("result")) or "").lower()
                source_type = str(record.get("sourceType") or self._type_name(record.get("source")) or "").lower()
                if result_type != type_l and source_type != type_l:
                    continue
            output.append(copy.copy(record))
        return output

    def _semantic_records_for_group(self, records: list) -> list:
        usable = [r for r in records if r.get("source") is not None]
        if not usable:
            return []

        consumed_results = []
        for candidate in usable:
            source = candidate.get("source")
            if source is None:
                continue
            for producer in usable:
                target = producer.get("result")
                if target is not None and self._same(target, source):
                    consumed_results.append(target)
                    break

        final_records = []
        for record in usable:
            target = record.get("result")
            if target is None:
                continue
            if any(self._same(target, consumed) for consumed in consumed_results):
                continue
            final_records.append(record)

        if not final_records:
            final_records = [r for r in usable if r.get("result") is not None]

        semantic = []

        def walk_back(target, relation, trail, template):
            incoming = self._find_matches(usable, "result", target)
            if not incoming:
                return

            for record in incoming:
                source = record.get("source")
                if source is None:
                    continue
                source_key = self._identity_key(source)
                if source_key in trail:
                    continue

                combined = self._combine_relation(relation, record.get("relation"))
                role = record.get("sourceRole")
                predecessors = self._find_matches(usable, "result", source)

                is_leaf = (not predecessors or not self._is_internal_role(role))

                if is_leaf:
                    item = self._copy_public_metadata(record)
                    for key, value in self._copy_public_metadata(template).items():
                        if key not in item or item.get(key) is None:
                            item[key] = value
                    item.update({
                        "source": source,
                        "result": template.get("result"),
                        "sourceType": record.get("sourceType") or self._type_name(source),
                        "resultType": template.get("resultType") or self._type_name(template.get("result")),
                        "relation": combined,
                    })
                    semantic.append(item)
                    continue

                walk_back(source, combined, trail | {source_key}, template)

        for terminal in final_records:
            target = terminal.get("result")
            if target is None:
                continue
            walk_back(
                target,
                terminal.get("relation"),
                {self._identity_key(target)},
                terminal,
            )

        for record in usable:
            if str(record.get("relation", "")).lower() == "deleted":
                semantic.append(copy.copy(record))

        return self._dedupe_records(semantic)

    def _dedupe_records(self, records: Iterable[dict]) -> list:
        output = []
        buckets = {}
        for record in records or []:
            source = record.get("source")
            target = record.get("result")
            key = (
                self._identity_key(source),
                self._identity_key(target),
                self._group_key(record),
            )
            bucket = buckets.setdefault(key, [])
            matched = None
            for existing in bucket:
                if self._same(existing.get("source"), source) and self._same(existing.get("result"), target):
                    matched = existing
                    break
            if matched is None:
                item = copy.copy(record)
                bucket.append(item)
                output.append(item)
            else:
                matched["relation"] = self._combine_relation(
                    matched.get("relation"),
                    record.get("relation"),
                )
                matched["usedBRepGraph"] = bool(
                    matched.get("usedBRepGraph") or record.get("usedBRepGraph")
                )
        return output

    def Records(
        self,
        *,
        relation: Optional[str] = None,
        topologyType: Optional[str] = None,
        sourceRole: Optional[str] = None,
        detailed: bool = False,
    ) -> list:
        """
        Return semantic lineage records by default.

        Set ``detailed=True`` to return exact kernel history.
        """
        if detailed:
            return self.History(
                relation=relation,
                topologyType=topologyType,
                sourceRole=sourceRole,
            )

        groups = {}
        order = []
        for record in self._history:
            key = self._group_key(record)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(record)

        records = []
        for key in order:
            records.extend(self._semantic_records_for_group(groups[key]))

        relation_l = str(relation).lower() if relation is not None else None
        type_l = str(topologyType).lower() if topologyType is not None else None
        role_l = str(sourceRole).lower() if sourceRole is not None else None

        output = []
        for record in records:
            if relation_l is not None and str(record.get("relation", "")).lower() != relation_l:
                continue
            if role_l is not None and str(record.get("sourceRole", "")).lower() != role_l:
                continue
            if type_l is not None:
                result_type = str(record.get("resultType") or self._type_name(record.get("result")) or "").lower()
                source_type = str(record.get("sourceType") or self._type_name(record.get("source")) or "").lower()
                if result_type != type_l and source_type != type_l:
                    continue
            output.append(copy.copy(record))
        return output

    def Origins(self, topology, *, detailed: bool = False) -> list:
        """Return provenance records whose result is the input topology."""
        return [
            record for record in self.Records(detailed=detailed)
            if record.get("result") is not None
            and self._same(record.get("result"), topology)
        ]

    def Descendants(self, topology, *, detailed: bool = False) -> list:
        """Return provenance records whose source is the input topology."""
        return [
            record for record in self.Records(detailed=detailed)
            if record.get("source") is not None
            and self._same(record.get("source"), topology)
        ]

    def Generated(self, topologyType: Optional[str] = None) -> list:
        return self.Records(relation="generated", topologyType=topologyType)

    def Modified(self, topologyType: Optional[str] = None) -> list:
        return self.Records(relation="modified", topologyType=topologyType)

    def Unchanged(self, topologyType: Optional[str] = None) -> list:
        return self.Records(relation="unchanged", topologyType=topologyType)

    def Deleted(self, topologyType: Optional[str] = None) -> list:
        return self.Records(relation="deleted", topologyType=topologyType)

    def Graph(
        self,
        topologyType: Optional[str] = None,
        *,
        includeDeleted: bool = False,
        detailed: bool = False,
    ):
        """
        Return provenance as a directed TGraph.

        Non-deleted vertices carry the actual TopologicPy topology as the TGraph
        ``representation``. Edge dictionaries carry the semantic relationship.
        """
        from topologicpy.TGraph import TGraph

        records = self.Records(
            topologyType=topologyType,
            detailed=detailed,
        )
        if not includeDeleted:
            records = [
                record for record in records
                if str(record.get("relation", "")).lower() != "deleted"
                and record.get("result") is not None
            ]

        graph = TGraph(
            directed=True,
            allowSelfLoops=False,
            allowParallelEdges=True,
            dictionary={
                "type": "Provenance",
                "schema": self.SCHEMA,
                "operation": self.operation,
                "detailed": bool(detailed),
                **copy.deepcopy(self.metadata),
            },
        )

        entities = []

        def entity_for(topology):
            if topology is None:
                return None
            for item in entities:
                if self._same(item["topology"], topology):
                    return item
            item = {
                "topology": topology,
                "asSource": False,
                "asResult": False,
                "sourceRoles": set(),
                "index": None,
            }
            entities.append(item)
            return item

        for record in records:
            s = entity_for(record.get("source"))
            if s is not None:
                s["asSource"] = True
                if record.get("sourceRole") is not None:
                    s["sourceRoles"].add(str(record.get("sourceRole")))
            t = entity_for(record.get("result"))
            if t is not None:
                t["asResult"] = True

        for item in entities:
            if item["asSource"] and item["asResult"]:
                role = "intermediate"
            elif item["asSource"]:
                role = "source"
            else:
                role = "result"
            dictionary = {
                "role": role,
                "topologyType": self._type_name(item["topology"]),
            }
            if item["sourceRoles"]:
                dictionary["sourceRoles"] = sorted(item["sourceRoles"])
            item["index"] = graph.AddVertex(
                dictionary=dictionary,
                representation=item["topology"],
                silent=True,
            )

        def index_for(topology):
            if topology is None:
                return None
            for item in entities:
                if self._same(item["topology"], topology):
                    return item["index"]
            return None

        deleted_counter = 0
        for record in records:
            source_index = index_for(record.get("source"))
            target_index = index_for(record.get("result"))

            if record.get("result") is None and includeDeleted:
                deleted_counter += 1
                target_index = graph.AddVertex(
                    dictionary={
                        "role": "deleted",
                        "topologyType": record.get("sourceType"),
                        "deleted": True,
                        "ordinal": deleted_counter,
                    },
                    representation=None,
                    silent=True,
                )

            if source_index is None or target_index is None:
                continue

            edge_dictionary = {"relation": record.get("relation")}
            for key in (
                "operation",
                "operationNode",
                "sourceNode",
                "sourceRole",
                "application",
                "rule",
                "title",
                "usedBRepGraph",
            ):
                if record.get(key) is not None:
                    edge_dictionary[key] = record.get(key)

            graph.AddEdge(
                source_index,
                target_index,
                directed=True,
                dictionary=edge_dictionary,
                representation=None,
                silent=True,
            )

        return graph

    def __len__(self):
        return len(self.Records())

    def __repr__(self):
        return (
            f"Provenance(records={len(self.Records())}, "
            f"history={len(self._history)}, supported={self.supported})"
        )
