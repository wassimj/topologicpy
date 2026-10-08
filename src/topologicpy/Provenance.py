# Copyright (C) 2026
# Wassim Jabi <wassimj@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free Software
# Foundation, either version 3.0 of the License, or (at your option) any later version.

from __future__ import annotations

from typing import Any, Iterable, Optional
import copy
import uuid
import weakref


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

    # Stable provenance identities are owned by Provenance, not by transient
    # TopologicPy wrappers and not by user dictionaries. Native OCCT identities
    # are confirmed exclusively with TopoDS_Shape.IsSame(); hashes are only
    # bucket accelerators.
    _SHAPE_ENTITY_BUCKETS = {}
    _OBJECT_ENTITY_IDS = weakref.WeakKeyDictionary()
    _OBJECT_ENTITY_FALLBACK = {}

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

        # Materialise stable endpoint identities while the exact operation
        # sources/result are available. This makes independently-created
        # Provenance objects composable later without geometric matching.
        self._initialise_entity_ids()

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
    def Compose(*provenances) -> "Provenance":
        """Compose multiple provenance objects into one staged provenance.

        Composition is identity-based. Consecutive operation boundaries are
        stitched only when the previous result root is exactly the same native
        topology as one of the next operation's source roots. Subtopologies on
        that shared boundary are then reconciled with native ``IsSame`` only;
        no coordinate, centroid, tolerance, bounding-box, or other geometric
        fallback is used.
        """
        items = [p for p in provenances if isinstance(p, Provenance)]
        if not items:
            return Provenance.ByRecords(
                [],
                operation="Compose",
                sources={},
                result=None,
                supported=False,
                metadata={"composed": True, "stageCount": 0},
            )

        # Work on semantic records from each input first. Keep explicit endpoint
        # IDs on the records so that later graph/traversal code does not need to
        # rediscover cross-operation identity.
        semantic_inputs = []
        for provenance in items:
            semantic = [copy.copy(record) for record in provenance.Records()]
            for record in semantic:
                provenance._stamp_record(record)
            semantic_inputs.append(semantic)

        # Reconcile each adjacent operation boundary. The common case is:
        #
        #   result_1, p1 = Op1(...)
        #   result_2, p2 = Op2(result_1, ...)
        #
        # ``p1.result`` and one root in ``p2.sources`` therefore denote the
        # same exact topology. Seed that authoritative boundary and force both
        # sides of the composed records to use its Provenance-owned entity IDs.
        boundary_matches = []
        for boundary_index in range(len(items) - 1):
            left = items[boundary_index]
            right = items[boundary_index + 1]
            boundary_root = left.result
            if boundary_root is None:
                boundary_matches.append(0)
                continue

            continuation = any(
                value is not None and Provenance._same(boundary_root, value)
                for value in right.sources.values()
            )
            if not continuation:
                boundary_matches.append(0)
                continue

            boundary_entities = Provenance._topology_entities(boundary_root)
            boundary_index_map = Provenance._boundary_index(boundary_entities)

            left_ids = set()
            right_ids = set()

            # Previous stage result-state -> boundary.
            for record in semantic_inputs[boundary_index]:
                target = record.get("result")
                entity_id = Provenance._boundary_match_id(
                    boundary_index_map,
                    boundary_entities,
                    target,
                )
                if entity_id is not None:
                    record["resultProvenanceId"] = entity_id
                    left_ids.add(entity_id)

            # Next stage source-state -> same boundary.
            for record in semantic_inputs[boundary_index + 1]:
                source = record.get("source")
                entity_id = Provenance._boundary_match_id(
                    boundary_index_map,
                    boundary_entities,
                    source,
                )
                if entity_id is not None:
                    record["sourceProvenanceId"] = entity_id
                    right_ids.add(entity_id)

            # Diagnostic metadata counts actual IDs present on both sides of
            # the operation boundary, i.e. the entities that can become
            # semantic "intermediate" graph vertices.
            boundary_matches.append(len(left_ids & right_ids))

        records = []
        sources = {}
        operations = []
        next_stage = 0

        for provenance_index, provenance in enumerate(items):
            semantic = semantic_inputs[provenance_index]

            # Preserve distinct stages already present in a composed provenance,
            # while remapping them into one monotonically increasing stage space.
            local_stage_map = {}
            for record in semantic:
                existing_stage = record.get("provenanceStage")
                if existing_stage is not None:
                    local_key = ("stage", existing_stage)
                else:
                    local_key = provenance._group_key(record)

                if local_key not in local_stage_map:
                    local_stage_map[local_key] = next_stage
                    next_stage += 1

                item = copy.copy(record)
                item["provenanceStage"] = local_stage_map[local_key]
                item["provenanceInput"] = provenance_index
                provenance._stamp_record(item)
                records.append(item)

            for key, value in provenance.sources.items():
                sources[f"{provenance_index}:{key}"] = value

            if provenance.metadata.get("composed"):
                operations.extend(
                    str(value)
                    for value in provenance.metadata.get("operations", [])
                )
            elif provenance.operation is not None:
                operations.append(provenance.operation)

        last = items[-1]
        metadata = {
            "composed": True,
            "stageCount": next_stage,
            "operations": operations,
            "boundaryMatches": boundary_matches,
        }

        # Inputs already seeded their roots during capture; boundary stitching
        # above has reconciled every composed endpoint. Construct the container
        # without reseeding all historical operands and the final result again.
        # Keep ordinary ByRecords construction unchanged for uninitialised input.
        composed = Provenance.__new__(Provenance)
        composed._history = [copy.copy(record) for record in records]
        composed.operation = "Compose"
        composed.sources = dict(sources)
        composed.result = last.result
        composed.supported = all(p.supported for p in items)
        composed.metadata = copy.deepcopy(metadata)
        for record in composed._history:
            composed._stamp_record(record)
        return composed

    @staticmethod
    def _native_shape(topology):
        if topology is None:
            return None

        try:
            shape = getattr(topology, "shape", None)
        except Exception:
            shape = None

        if shape is not None:
            return shape

        # Permit native TopoDS_Shape objects in private/synthetic paths without
        # importing OCC here. ShapeType + IsSame form the exact identity API we
        # need.
        if hasattr(topology, "ShapeType") and hasattr(topology, "IsSame"):
            return topology

        try:
            getter = getattr(topology, "GetOcctShape", None)
            if callable(getter):
                return getter()
        except Exception:
            pass

        return None

    @staticmethod
    def _native_same(shape_a, shape_b) -> bool:
        if shape_a is shape_b:
            return shape_a is not None
        if shape_a is None or shape_b is None:
            return False
        try:
            return bool(shape_a.IsSame(shape_b))
        except Exception:
            return False

    @staticmethod
    def _native_bucket_key(shape):
        if shape is None:
            return None

        try:
            shape_type = int(shape.ShapeType())
        except Exception:
            try:
                shape_type = str(shape.ShapeType())
            except Exception:
                shape_type = None

        try:
            shape_hash = hash(shape)
        except Exception:
            try:
                shape_hash = int(shape.HashCode(2147483647))
            except Exception:
                shape_hash = None

        return (shape_type, shape_hash)

    @classmethod
    def _new_entity_id(cls) -> str:
        return "prov_" + uuid.uuid4().hex

    @classmethod
    def _shape_entity_id(cls, shape, create: bool = True) -> Optional[str]:
        if shape is None:
            return None

        key = cls._native_bucket_key(shape)
        bucket = cls._SHAPE_ENTITY_BUCKETS.setdefault(key, [])

        for existing_shape, entity_id in bucket:
            if cls._native_same(existing_shape, shape):
                return entity_id

        # Hashes are accelerators only. OCCT permits no correctness assumption
        # here, so confirm against other buckets before creating a new ID.
        for other_key, other_bucket in cls._SHAPE_ENTITY_BUCKETS.items():
            if other_key == key:
                continue
            for existing_shape, entity_id in other_bucket:
                if cls._native_same(existing_shape, shape):
                    bucket.append((shape, entity_id))
                    return entity_id

        if not create:
            return None

        entity_id = cls._new_entity_id()
        bucket.append((shape, entity_id))
        return entity_id

    @classmethod
    def _object_entity_id(cls, topology, create: bool = True) -> Optional[str]:
        if topology is None:
            return None

        try:
            entity_id = cls._OBJECT_ENTITY_IDS.get(topology)
            if entity_id is not None:
                return entity_id
        except Exception:
            pass

        object_key = id(topology)
        fallback = cls._OBJECT_ENTITY_FALLBACK.get(object_key)
        if fallback is not None:
            existing, entity_id = fallback
            if existing is topology:
                return entity_id

        if not create:
            return None

        entity_id = cls._new_entity_id()
        try:
            cls._OBJECT_ENTITY_IDS[topology] = entity_id
            return entity_id
        except Exception:
            cls._OBJECT_ENTITY_FALLBACK[object_key] = (topology, entity_id)
            return entity_id

    @classmethod
    def _entity_id(cls, topology, create: bool = True) -> Optional[str]:
        if topology is None:
            return None

        shape = cls._native_shape(topology)
        if shape is not None:
            return cls._shape_entity_id(shape, create=create)

        return cls._object_entity_id(topology, create=create)

    @classmethod
    def _same(cls, a, b) -> bool:
        """Return exact TopologicPy identity; never geometric equivalence."""
        if a is b:
            return True
        if a is None or b is None:
            return False

        # Captured native endpoints can be compared directly. Routing each
        # pair through public IsSame revalidates up to ten wrapper types on
        # both sides, dominating semantic reduction on even small histories.
        # This is the same native identity predicate used by the backend;
        # a False answer is authoritative, never geometric coincidence.
        shape_a = cls._native_shape(a)
        shape_b = cls._native_shape(b)
        if shape_a is not None and shape_b is not None:
            try:
                if not shape_a.IsNull() and not shape_b.IsNull():
                    return bool(shape_a.IsSame(shape_b))
            except Exception:
                # Null, legacy or unavailable native APIs retain public logic.
                pass

        # Preserve TopologicPy's exact identity contract. On PythonOCC this
        # checks native TopoDS_Shape.IsSame first, then the wrapper UUID.
        try:
            from topologicpy.Topology import Topology
            result = Topology.IsSame(a, b, silent=True)
            if result is not None:
                return bool(result)
        except Exception:
            pass

        shape_a = cls._native_shape(a)
        shape_b = cls._native_shape(b)

        if shape_a is not None or shape_b is not None:
            if shape_a is None or shape_b is None:
                return False
            return cls._native_same(shape_a, shape_b)

        try:
            method = getattr(a, "IsSame", None)
            if callable(method):
                return bool(method(b))
        except Exception:
            pass

        return False

    @staticmethod
    def _identity_key(topology):
        if topology is None:
            return ("none", None)

        shape = getattr(topology, "shape", None)

        try:
            if shape is not None:
                return (
                    "shape",
                    topology.__class__.__name__,
                    hash(shape),
                )
        except Exception:
            pass

        return (
            "object",
            topology.__class__.__name__,
            id(topology),
        )

    @classmethod
    def _topology_entities(cls, root) -> list:
        """Return root + exact public subtopologies, deduplicated by entity ID."""
        if root is None:
            return []

        output = []
        seen = set()

        def add(item):
            if item is None:
                return
            entity_id = cls._entity_id(item)
            key = entity_id if entity_id is not None else ("object", id(item))
            if key in seen:
                return
            seen.add(key)
            output.append(item)

        add(root)

        try:
            from topologicpy.Topology import Topology
        except Exception:
            return output

        for name in (
            "Vertices",
            "Edges",
            "Wires",
            "Faces",
            "Shells",
            "Cells",
            "CellComplexes",
        ):
            method = getattr(Topology, name, None)
            if not callable(method):
                continue
            try:
                values = method(root, silent=True) or []
            except TypeError:
                try:
                    values = method(root) or []
                except Exception:
                    values = []
            except Exception:
                values = []

            for value in values:
                add(value)

        return output

    @classmethod
    def _boundary_index(cls, entities: Iterable[Any]) -> dict:
        result = {}
        for entity in entities or []:
            shape = cls._native_shape(entity)
            key = cls._native_bucket_key(shape) if shape is not None else ("object", id(entity))
            result.setdefault(key, []).append(
                (entity, cls._entity_id(entity))
            )
        return result

    @classmethod
    def _boundary_match_id(
        cls,
        index: dict,
        entities: Iterable[Any],
        topology,
    ) -> Optional[str]:
        if topology is None:
            return None

        shape = cls._native_shape(topology)
        key = cls._native_bucket_key(shape) if shape is not None else ("object", id(topology))
        candidates = list((index or {}).get(key, []))

        for entity, entity_id in candidates:
            if cls._same(entity, topology):
                return entity_id

        # The bucket is only an accelerator. If it misses, still perform exact
        # native identity confirmation across the boundary.
        for entity in entities or []:
            if cls._same(entity, topology):
                return cls._entity_id(entity)

        return None

    def _stamp_record(self, record: dict, *, overwrite: bool = False) -> dict:
        if not isinstance(record, dict):
            return record

        for field, key in (
            ("source", "sourceProvenanceId"),
            ("result", "resultProvenanceId"),
        ):
            topology = record.get(field)
            if topology is None:
                continue
            if not overwrite and record.get(key) is not None:
                continue
            entity_id = self._entity_id(topology)
            if entity_id is not None:
                record[key] = entity_id

        return record

    def _initialise_entity_ids(self) -> None:
        # Seed operation inputs first so raw history source endpoints inherit the
        # identity of the actual public operands.
        for root in self.sources.values():
            self._topology_entities(root)

        # Stamp exact kernel/materialised history.
        for record in self._history:
            self._stamp_record(record)

        # Seed the authoritative returned topology after history capture. Exact
        # IsSame reuses IDs already assigned to final history images.
        self._topology_entities(self.result)

        # One final stamp covers any record endpoint that only became reachable
        # after authoritative result seeding.
        for record in self._history:
            self._stamp_record(record)

    def _record_entity_id(self, record: dict, field: str) -> Optional[str]:
        key = "sourceProvenanceId" if field == "source" else "resultProvenanceId"
        value = record.get(key)
        if value is not None:
            return str(value)
        return self._entity_id(record.get(field))

    def _record_matches_topology(self, record: dict, field: str, topology) -> bool:
        value = record.get(field)
        if value is None or topology is None:
            return False

        # Direct-operation identity is authoritative. A provisional provenance
        # ID must never split two wrappers that TopologicPy considers identical.
        if self._same(value, topology):
            return True

        # Reconciled IDs remain the exact cross-operation identity channel used
        # by composed provenance when wrappers are no longer IsSame.
        record_id = self._record_entity_id(record, field)
        topology_id = self._entity_id(topology, create=False)

        return (
            record_id is not None
            and topology_id is not None
            and record_id == topology_id
        )

    def _records_share_endpoint(
        self,
        record_a: dict,
        field_a: str,
        record_b: dict,
        field_b: str,
    ) -> bool:
        a = record_a.get(field_a)
        b = record_b.get(field_b)

        if a is None or b is None:
            return False

        # Within one captured operation, exact topology identity wins over any
        # provisional IDs assigned before semantic reduction.
        if self._same(a, b):
            return True

        # Across composed boundaries, explicitly reconciled IDs are the stable
        # identity channel.
        id_a = self._record_entity_id(record_a, field_a)
        id_b = self._record_entity_id(record_b, field_b)

        return (
            id_a is not None
            and id_b is not None
            and id_a == id_b
        )

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
        stage = record.get("provenanceStage")
        if stage is not None:
            return (
                "provenanceStage",
                stage,
                record.get("operationNode"),
                record.get("application"),
            )
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
            "provenanceStage",
            "provenanceInput",
        )
        return {key: record.get(key) for key in keys if record.get(key) is not None}

    @staticmethod
    def _find_matches(records: list, field: str, topology) -> list:
        return [
            record
            for record in records
            if record.get(field) is not None
            and Provenance._same(
                record.get(field),
                topology,
            )
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
        usable = [
            record
            for record in records
            if record.get("source") is not None
        ]

        if not usable:
            return []

        consumed_results = []

        for candidate in usable:
            source = candidate.get("source")

            if source is None:
                continue

            for producer in usable:
                target = producer.get("result")

                if (
                    target is not None
                    and self._same(target, source)
                ):
                    consumed_results.append(target)
                    break

        final_records = []

        for record in usable:
            target = record.get("result")

            if target is None:
                continue

            if any(
                self._same(target, consumed)
                for consumed in consumed_results
            ):
                continue

            final_records.append(record)

        if not final_records:
            final_records = [
                record
                for record in usable
                if record.get("result") is not None
            ]

        semantic = []

        def walk_back(
            target,
            relation,
            trail,
            template,
        ):
            incoming = self._find_matches(
                usable,
                "result",
                target,
            )

            if not incoming:
                return

            for record in incoming:
                source = record.get("source")

                if source is None:
                    continue

                combined = self._combine_relation(
                    relation,
                    record.get("relation"),
                )

                # An unchanged public operand is a real state transition.
                # An internal unchanged link is only a post-processing identity:
                # its public predecessors are already in this incoming set.
                # Emitting it would introduce an internal face as another source.
                if (
                    not self._is_internal_role(record.get("sourceRole"))
                    and str(
                        record.get(
                            "relation",
                            "",
                        )
                    ).lower()
                    == "unchanged"
                    and record.get("result") is not None
                    and self._same(
                        source,
                        record.get("result"),
                    )
                ):
                    item = self._copy_public_metadata(
                        record
                    )

                    for key, value in (
                        self._copy_public_metadata(
                            template
                        ).items()
                    ):
                        if (
                            key not in item
                            or item.get(key) is None
                        ):
                            item[key] = value

                    item.update({
                        "source":
                            source,

                        "result":
                            template.get("result"),

                        "sourceType":
                            record.get("sourceType")
                            or self._type_name(source),

                        "resultType":
                            template.get("resultType")
                            or self._type_name(
                                template.get("result")
                            ),

                        "relation":
                            combined,
                    })

                    semantic.append(item)
                    continue

                source_key = self._identity_key(
                    source
                )

                if source_key in trail:
                    continue

                role = record.get("sourceRole")

                predecessors = self._find_matches(
                    usable,
                    "result",
                    source,
                )

                # Internal Boolean/post-processing states are never public provenance
                # origins. If an internal state has an exact predecessor, continue
                # tracing backwards. If it does not, discard that unresolved internal
                # state rather than exposing it as a semantic source.
                if self._is_internal_role(role):
                    if predecessors:
                        walk_back(
                            source,
                            combined,
                            trail | {source_key},
                            template,
                        )
                    continue

                # Non-internal roles (for example "self" and "other") are the public
                # operand states and therefore terminate the semantic back-trace.
                item = self._copy_public_metadata(
                    record
                )

                for key, value in (
                    self._copy_public_metadata(
                        template
                    ).items()
                ):
                    if (
                        key not in item
                        or item.get(key) is None
                    ):
                        item[key] = value

                item.update({
                    "source":
                        source,

                    "result":
                        template.get("result"),

                    "sourceType":
                        record.get("sourceType")
                        or self._type_name(source),

                    "resultType":
                        template.get("resultType")
                        or self._type_name(
                            template.get("result")
                        ),

                    "relation":
                        combined,
                })

                semantic.append(item)

        for terminal in final_records:
            target = terminal.get("result")

            if target is None:
                continue

            walk_back(
                target,
                terminal.get("relation"),
                {
                    self._identity_key(
                        target
                    )
                },
                terminal,
            )

        for record in usable:
            if (
                str(
                    record.get(
                        "relation",
                        "",
                    )
                ).lower()
                == "deleted"
            ):
                semantic.append(
                    copy.copy(record)
                )

        return self._dedupe_records(
            semantic
        )

    def _dedupe_records(self, records: Iterable[dict]) -> list:
        # Direct provenance retains the pre-Compose deduplication semantics.
        if not self.metadata.get("composed"):
            output = []
            buckets = {}

            def legacy_identity_key(topology):
                if topology is None:
                    return ("none", None)

                shape = getattr(topology, "shape", None)

                try:
                    if shape is not None:
                        return (
                            "shape",
                            topology.__class__.__name__,
                            hash(shape),
                        )
                except Exception:
                    pass

                return (
                    "object",
                    topology.__class__.__name__,
                    id(topology),
                )

            for record in records or []:
                source = record.get("source")
                target = record.get("result")

                key = (
                    legacy_identity_key(source),
                    legacy_identity_key(target),
                    self._group_key(record),
                )

                bucket = buckets.setdefault(key, [])
                matched = None

                for existing in bucket:
                    if (
                        self._same(
                            existing.get("source"),
                            source,
                        )
                        and self._same(
                            existing.get("result"),
                            target,
                        )
                    ):
                        matched = existing
                        break

                if matched is None:
                    item = copy.copy(record)
                    self._stamp_record(item)
                    bucket.append(item)
                    output.append(item)
                else:
                    matched["relation"] = self._combine_relation(
                        matched.get("relation"),
                        record.get("relation"),
                    )
                    matched["usedBRepGraph"] = bool(
                        matched.get("usedBRepGraph")
                        or record.get("usedBRepGraph")
                    )

            return output

        # Composed provenance remains ID-based because adjacent operations may
        # contain different wrappers for the same reconciled entity.
        output = []
        buckets = {}

        for record in records or []:
            item = copy.copy(record)
            self._stamp_record(item)

            source_id = self._record_entity_id(
                item,
                "source",
            )
            target_id = self._record_entity_id(
                item,
                "result",
            )

            key = (
                source_id,
                target_id,
                self._group_key(item),
            )

            matched = buckets.get(key)

            if matched is None:
                buckets[key] = item
                output.append(item)
                continue

            matched["relation"] = self._combine_relation(
                matched.get("relation"),
                item.get("relation"),
            )
            matched["usedBRepGraph"] = bool(
                matched.get("usedBRepGraph")
                or item.get("usedBRepGraph")
            )

        return output

    def _authoritative_final_entities(
        self,
        topologyType: Optional[str] = None,
    ) -> list:
        """Return actual final entities contained in ``self.result``."""
        if self.result is None:
            return []

        try:
            from topologicpy.Topology import Topology
        except Exception:
            return []

        requested = str(topologyType).lower() if topologyType is not None else None
        result = []

        def add(topology):
            if topology is None:
                return
            for existing in result:
                if self._same(existing, topology):
                    return
            result.append(topology)

        try:
            root_type = str(Topology.TypeAsString(self.result) or "").lower()
        except Exception:
            root_type = ""

        if requested is None or root_type == requested:
            add(self.result)

        extractors = {
            "vertex": "Vertices",
            "edge": "Edges",
            "wire": "Wires",
            "face": "Faces",
            "shell": "Shells",
            "cell": "Cells",
            "cellcomplex": "CellComplexes",
        }

        names = [requested] if requested in extractors else list(extractors)

        for name in names:
            method_name = extractors.get(name)
            if method_name is None:
                continue
            method = getattr(Topology, method_name, None)
            if method is None:
                continue

            try:
                items = method(self.result, silent=True) or []
            except TypeError:
                try:
                    items = method(self.result) or []
                except Exception:
                    items = []
            except Exception:
                items = []

            for item in items:
                add(item)

        return result

    def _reconcile_final_records(
        self,
        records: list,
        topologyType: Optional[str] = None,
    ) -> list:
        """Keep only terminal records that belong to the returned result."""
        if (
            self.result is None
            or self.metadata.get("composed")
        ):
            return records

        finals = self._authoritative_final_entities(
            topologyType=topologyType
        )

        if not finals:
            return records

        output = []

        for record in records or []:
            relation = str(
                record.get(
                    "relation",
                    "",
                )
            ).lower()

            target = record.get("result")

            if target is None:
                if relation == "deleted":
                    output.append(record)
                continue

            if any(
                self._same(
                    target,
                    final,
                )
                for final in finals
            ):
                output.append(record)

        return output

    def _restore_missing_final_records(
        self,
        records: list,
        topologyType: Optional[str] = None,
    ) -> list:
        if (
            self.result is None
            or self.metadata.get("composed")
        ):
            return records

        finals = self._authoritative_final_entities(
            topologyType=topologyType
        )

        if not finals:
            return records

        output = [
            copy.copy(record)
            for record in (records or [])
        ]

        def has_target(final):
            for record in output:
                target = record.get("result")

                if (
                    target is not None
                    and self._same(
                        target,
                        final,
                    )
                ):
                    return True

            return False

        for final in finals:
            if has_target(final):
                continue

            candidates = []

            for record in self._history:
                target = record.get("result")

                if (
                    target is None
                    or not self._same(
                        target,
                        final,
                    )
                ):
                    continue

                record_type = (
                    record.get("resultType")
                    or self._type_name(target)
                )

                if (
                    topologyType is not None
                    and str(
                        record_type or ""
                    ).lower()
                    != str(
                        topologyType
                    ).lower()
                ):
                    continue

                candidates.append(record)

            candidates.sort(
                key=lambda record: (
                    self._is_internal_role(
                        record.get("sourceRole")
                    ),
                    0
                    if str(
                        record.get(
                            "relation",
                            "",
                        )
                    ).lower()
                    == "unchanged"
                    else 1,
                )
            )

            restored = None

            for record in candidates:
                source = record.get("source")

                if source is None:
                    continue

                if not self._is_internal_role(
                    record.get("sourceRole")
                ):
                    restored = copy.copy(
                        record
                    )
                    break

                incoming = self._find_matches(
                    self._history,
                    "result",
                    source,
                )

                for predecessor in incoming:
                    predecessor_source = (
                        predecessor.get("source")
                    )

                    if predecessor_source is None:
                        continue

                    if self._is_internal_role(
                        predecessor.get(
                            "sourceRole"
                        )
                    ):
                        continue

                    restored = (
                        self._copy_public_metadata(
                            predecessor
                        )
                    )

                    for key, value in (
                        self._copy_public_metadata(
                            record
                        ).items()
                    ):
                        if (
                            key not in restored
                            or restored.get(key) is None
                        ):
                            restored[key] = value

                    restored.update({
                        "source":
                            predecessor_source,

                        "result":
                            final,

                        "sourceType":
                            predecessor.get(
                                "sourceType"
                            )
                            or self._type_name(
                                predecessor_source
                            ),

                        "resultType":
                            record.get(
                                "resultType"
                            )
                            or self._type_name(
                                final
                            ),

                        "relation":
                            self._combine_relation(
                                predecessor.get(
                                    "relation"
                                ),
                                record.get(
                                    "relation"
                                ),
                            ),
                    })

                    break

                if restored is not None:
                    break

            if restored is None:
                for record in self._history:
                    source = record.get("source")

                    if (
                        source is None
                        or not self._same(
                            source,
                            final,
                        )
                    ):
                        continue

                    if self._is_internal_role(
                        record.get("sourceRole")
                    ):
                        continue

                    source_type = (
                        record.get("sourceType")
                        or self._type_name(
                            source
                        )
                    )

                    if (
                        topologyType is not None
                        and str(
                            source_type or ""
                        ).lower()
                        != str(
                            topologyType
                        ).lower()
                    ):
                        continue

                    restored = (
                        self._copy_public_metadata(
                            record
                        )
                    )

                    restored.update({
                        "source":
                            source,

                        "result":
                            final,

                        "sourceType":
                            source_type,

                        "resultType":
                            self._type_name(
                                final
                            ),

                        "relation":
                            "unchanged",
                    })

                    break

            if restored is not None:
                output.append(restored)

        return self._dedupe_records(
            output
        )

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

        # A composed Provenance is built from the already-semantic Records()
        # of its constituent provenances. Do NOT run semantic reduction again:
        # doing so can reinterpret an unchanged source->result record as an
        # internal link whenever that same native entity also appears as a
        # source in the stage. Direct provenances may restore such records from
        # their authoritative result, but composed provenance intentionally has
        # no single authoritative result per intermediate stage. A second
        # reduction therefore destroys exactly the boundary records that
        # Compose needs for stitching.
        if self.metadata.get("composed"):
            records = [copy.copy(record) for record in self._history]
        else:
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

            # A direct Boolean has one authoritative returned topology.
            # Drop stale raw/pre-normalisation terminal entities that are not
            # actual subtopologies of that returned result.
            records = self._reconcile_final_records(
                records,
                topologyType=topologyType,
            )

            records = self._restore_missing_final_records(
                records,
                topologyType=topologyType,
            )

        relation_l = str(relation).lower() if relation is not None else None
        type_l = str(topologyType).lower() if topologyType is not None else None
        role_l = str(sourceRole).lower() if sourceRole is not None else None

        output = []
        composed = bool(self.metadata.get("composed"))

        for record in records:
            if relation_l is not None and str(record.get("relation", "")).lower() != relation_l:
                continue
            if role_l is not None and str(record.get("sourceRole", "")).lower() != role_l:
                continue

            if type_l is not None:
                result_type = str(
                    record.get("resultType")
                    or self._type_name(record.get("result"))
                    or ""
                ).lower()
                source_type = str(
                    record.get("sourceType")
                    or self._type_name(record.get("source"))
                    or ""
                ).lower()

                if composed:
                    # Compose stores the already-semantic records of each
                    # constituent operation. For a typed composed view, use
                    # the same result-side authority as a direct provenance
                    # query after final-result reconciliation. A relationship
                    # with a real result belongs to the requested topology
                    # type only when that result has the requested type.
                    # This prevents cross-dimensional native history such as
                    # Face -> Edge/Vertex from surfacing non-Face vertices in
                    # Graph(topologyType="Face"). Deleted records have no
                    # result, so their source type remains authoritative.
                    if record.get("result") is None:
                        if source_type != type_l:
                            continue
                    elif result_type != type_l:
                        continue
                else:
                    if result_type != type_l and source_type != type_l:
                        continue

            output.append(copy.copy(record))

        return output


    def Origins(self, topology, *, detailed: bool = False) -> list:
        """Return the origins of ``topology``.

        For composed provenance this traverses backwards across staged operation
        boundaries using stable provenance entity IDs. No geometric matching is
        performed.
        """
        records = self.Records(detailed=detailed)
        topology_id = self._entity_id(topology)

        if not self.metadata.get("composed"):
            return [
                record
                for record in records
                if record.get("result") is not None
                and (
                    (
                        topology_id is not None
                        and self._record_entity_id(record, "result") == topology_id
                    )
                    or (
                        topology_id is None
                        and self._same(record.get("result"), topology)
                    )
                )
            ]

        staged = [
            record
            for record in records
            if record.get("provenanceStage") is not None
        ]

        matching = [
            record
            for record in staged
            if record.get("result") is not None
            and self._record_entity_id(record, "result") == topology_id
        ]
        if not matching:
            return []

        terminal_stage = max(
            int(record.get("provenanceStage"))
            for record in matching
        )
        output = []

        def walk(target, target_id, result_stage, relation=None, trail=None):
            trail = set() if trail is None else set(trail)

            incoming = [
                record
                for record in staged
                if int(record.get("provenanceStage")) == result_stage
                and record.get("result") is not None
                and self._record_entity_id(record, "result") == target_id
            ]

            for record in incoming:
                source = record.get("source")
                if source is None:
                    continue

                source_id = self._record_entity_id(record, "source")
                state_key = (
                    int(record.get("provenanceStage")),
                    source_id,
                )
                if state_key in trail:
                    continue

                combined = self._combine_relation(
                    relation,
                    record.get("relation"),
                )

                previous_stage = (
                    int(record.get("provenanceStage")) - 1
                )

                predecessors = [
                    predecessor
                    for predecessor in staged
                    if int(predecessor.get("provenanceStage"))
                    == previous_stage
                    and predecessor.get("result") is not None
                    and self._record_entity_id(
                        predecessor,
                        "result",
                    ) == source_id
                ]

                if predecessors:
                    walk(
                        source,
                        source_id,
                        previous_stage,
                        combined,
                        trail | {state_key},
                    )
                    continue

                item = self._copy_public_metadata(record)
                item.update({
                    "source": source,
                    "result": topology,
                    "sourceType": (
                        record.get("sourceType")
                        or self._type_name(source)
                    ),
                    "resultType": self._type_name(topology),
                    "relation": combined,
                    "sourceProvenanceId": source_id,
                    "resultProvenanceId": topology_id,
                })
                self._stamp_record(item)
                output.append(item)

        walk(
            topology,
            topology_id,
            terminal_stage,
            trail={(terminal_stage + 1, topology_id)},
        )

        return self._dedupe_records(output)

    def Descendants(self, topology, *, detailed: bool = False) -> list:
        """Return the descendants of ``topology``.

        For composed provenance this traverses forwards across staged operation
        boundaries using stable provenance entity IDs. No geometric matching is
        performed.
        """
        records = self.Records(detailed=detailed)
        topology_id = self._entity_id(topology)

        if not self.metadata.get("composed"):
            return [
                record
                for record in records
                if record.get("source") is not None
                and (
                    (
                        topology_id is not None
                        and self._record_entity_id(record, "source") == topology_id
                    )
                    or (
                        topology_id is None
                        and self._same(record.get("source"), topology)
                    )
                )
            ]

        staged = [
            record
            for record in records
            if record.get("provenanceStage") is not None
        ]

        matching = [
            record
            for record in staged
            if record.get("source") is not None
            and self._record_entity_id(record, "source") == topology_id
        ]
        if not matching:
            return []

        initial_stage = min(
            int(record.get("provenanceStage"))
            for record in matching
        )
        output = []

        def walk(source, source_id, source_stage, relation=None, trail=None):
            trail = set() if trail is None else set(trail)

            outgoing = [
                record
                for record in staged
                if int(record.get("provenanceStage")) == source_stage
                and record.get("source") is not None
                and self._record_entity_id(record, "source") == source_id
            ]

            for record in outgoing:
                target = record.get("result")
                if target is None:
                    continue

                target_id = self._record_entity_id(record, "result")
                state_key = (
                    int(record.get("provenanceStage")) + 1,
                    target_id,
                )
                if state_key in trail:
                    continue

                combined = self._combine_relation(
                    relation,
                    record.get("relation"),
                )

                next_stage = (
                    int(record.get("provenanceStage")) + 1
                )

                successors = [
                    successor
                    for successor in staged
                    if int(successor.get("provenanceStage")) == next_stage
                    and successor.get("source") is not None
                    and self._record_entity_id(
                        successor,
                        "source",
                    ) == target_id
                ]

                if successors:
                    walk(
                        target,
                        target_id,
                        next_stage,
                        combined,
                        trail | {state_key},
                    )
                    continue

                item = self._copy_public_metadata(record)
                item.update({
                    "source": topology,
                    "result": target,
                    "sourceType": self._type_name(topology),
                    "resultType": (
                        record.get("resultType")
                        or self._type_name(target)
                    ),
                    "relation": combined,
                    "sourceProvenanceId": topology_id,
                    "resultProvenanceId": target_id,
                })
                self._stamp_record(item)
                output.append(item)

        walk(
            topology,
            topology_id,
            initial_stage,
            trail={(initial_stage, topology_id)},
        )

        return self._dedupe_records(output)

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
        ``representation``. For composed provenance, graph vertices are keyed by
        ``(semantic boundary, provenance entity ID)``. This is what permits the
        result-state of one operation to become the source-state of the next
        without geometric matching.
        """
        from topologicpy.TGraph import TGraph

        records = self.Records(
            topologyType=topologyType,
            detailed=detailed,
        )

        for record in records:
            self._stamp_record(record)

        if not includeDeleted:
            records = [
                record
                for record in records
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
        composed = bool(self.metadata.get("composed"))
        direct_operation = self.result is not None and not composed

        def entity_for(topology, state=None, provenance_id=None):
            if topology is None:
                return None

            provenance_id = (
                provenance_id
                if provenance_id is not None
                else self._entity_id(topology)
            )

            for item in entities:
                if item.get("state") != state:
                    continue

                # A direct operation is one exact semantic identity domain.
                # Ignore provisional provenance IDs when collapsing wrappers.
                if direct_operation:
                    if self._same(
                        item["topology"],
                        topology,
                    ):
                        return item
                    continue

                # A composed provenance spans independent operation instances.
                # Here the reconciled provenance ID is authoritative.
                existing_id = item.get("provenanceId")

                if (
                    provenance_id is not None
                    and existing_id is not None
                ):
                    if provenance_id == existing_id:
                        return item
                    continue

                if self._same(
                    item["topology"],
                    topology,
                ):
                    return item

            item = {
                "topology": topology,
                "state": state,
                "provenanceId": provenance_id,
                "asSource": False,
                "asResult": False,
                "sourceRoles": set(),
                "index": None,
            }

            entities.append(item)
            return item

        for record in records:
            stage = record.get("provenanceStage")

            if composed and stage is not None:
                source_state = ("boundary", int(stage))
                result_state = ("boundary", int(stage) + 1)
            else:
                source_state = "source" if direct_operation else None
                result_state = "result" if direct_operation else None

            source_item = entity_for(
                record.get("source"),
                state=source_state,
                provenance_id=self._record_entity_id(
                    record,
                    "source",
                ),
            )
            if source_item is not None:
                source_item["asSource"] = True
                if record.get("sourceRole") is not None:
                    source_item["sourceRoles"].add(
                        str(record.get("sourceRole"))
                    )

            result_item = entity_for(
                record.get("result"),
                state=result_state,
                provenance_id=self._record_entity_id(
                    record,
                    "result",
                ),
            )
            if result_item is not None:
                result_item["asResult"] = True

        stage_count = 0
        if composed:
            try:
                stage_count = int(self.metadata.get("stageCount", 0) or 0)
            except Exception:
                stage_count = 0
            if stage_count <= 0:
                stages = [
                    int(record.get("provenanceStage"))
                    for record in records
                    if record.get("provenanceStage") is not None
                ]
                stage_count = (max(stages) + 1) if stages else 0

        for item in entities:
            if direct_operation:
                role = item.get("state")
            elif composed:
                state = item.get("state")
                boundary = None
                if (
                    isinstance(state, tuple)
                    and len(state) == 2
                    and state[0] == "boundary"
                ):
                    try:
                        boundary = int(state[1])
                    except Exception:
                        boundary = None

                # Role in a composed graph is boundary-aware. Only the last
                # semantic boundary is a public result state. A topology
                # produced by an earlier stage is intermediate even if it is
                # not consumed by the next stage (for example, a Face removed
                # by a later Difference). A topology first introduced as a
                # source at an internal boundary remains an external source.
                if boundary == 0:
                    role = "source" if item["asSource"] else "intermediate"
                elif stage_count > 0 and boundary == stage_count:
                    role = "result" if item["asResult"] else "source"
                elif item["asResult"]:
                    role = "intermediate"
                elif item["asSource"]:
                    role = "source"
                else:
                    role = "intermediate"
            elif item["asSource"] and item["asResult"]:
                role = "intermediate"
            elif item["asSource"]:
                role = "source"
            else:
                role = "result"

            dictionary = {
                "role": role,
                "topologyType": self._type_name(item["topology"]),
            }

            if item.get("provenanceId") is not None:
                dictionary["provenanceId"] = item["provenanceId"]

            if item["sourceRoles"]:
                dictionary["sourceRoles"] = sorted(
                    item["sourceRoles"]
                )

            item["index"] = graph.AddVertex(
                dictionary=dictionary,
                representation=item["topology"],
                silent=True,
            )

        def index_for(topology, state=None, provenance_id=None):
            if topology is None:
                return None

            provenance_id = (
                provenance_id
                if provenance_id is not None
                else self._entity_id(topology)
            )

            for item in entities:
                if item.get("state") != state:
                    continue

                if direct_operation:
                    if self._same(
                        item["topology"],
                        topology,
                    ):
                        return item["index"]
                    continue

                existing_id = item.get("provenanceId")

                if (
                    provenance_id is not None
                    and existing_id is not None
                ):
                    if provenance_id == existing_id:
                        return item["index"]
                    continue

                if self._same(
                    item["topology"],
                    topology,
                ):
                    return item["index"]

            return None

        deleted_counter = 0

        for record in records:
            stage = record.get("provenanceStage")

            if composed and stage is not None:
                source_state = ("boundary", int(stage))
                result_state = ("boundary", int(stage) + 1)
            else:
                source_state = "source" if direct_operation else None
                result_state = "result" if direct_operation else None

            source_index = index_for(
                record.get("source"),
                state=source_state,
                provenance_id=self._record_entity_id(
                    record,
                    "source",
                ),
            )
            target_index = index_for(
                record.get("result"),
                state=result_state,
                provenance_id=self._record_entity_id(
                    record,
                    "result",
                ),
            )

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

            edge_dictionary = {
                "relation": record.get("relation"),
            }

            for key in (
                "operation",
                "operationNode",
                "sourceNode",
                "sourceRole",
                "application",
                "rule",
                "title",
                "usedBRepGraph",
                "provenanceStage",
                "provenanceInput",
                "sourceProvenanceId",
                "resultProvenanceId",
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
