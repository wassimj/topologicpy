# Copyright (C) 2026
# This file is part of TopologicPy.
#
# TopologicPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class _GraphRAGConfig:
    """
    Lightweight configuration object used by the GraphRAG static methods.
    """
    graphdb: Any = None
    llm: Any = None
    vertexIDKey: str = "id"
    vertexLabelKey: str = "label"
    edgeLabelKey: str = "label"
    ontology: bool = True
    ontologyClassKey: str = "ontology_class"
    categoryKey: str = "category"
    uriKey: str = "uri"
    sourceKey: str = "source"
    generatedByKey: str = "generated_by"
    derivedFromKey: str = "derived_from"
    defaultGraphOntologyClass: str = "top:KnowledgeGraph"
    defaultVertexOntologyClass: str = "top:Node"
    defaultEdgeOntologyClass: str = "top:Relationship"
    defaultVertexCategory: str = "node"
    defaultEdgeCategory: str = "relationship"
    defaultGeneratedBy: str = "GraphRAG.Generate"
    defaultEdgeLabel: str = "suggested"
    tolerance: float = 0.0001
    maxCandidates: int = 12
    maxPairs: int = 40
    promptContext: str = None
    silent: bool = False

    def __repr__(self):
        return (
            "_GraphRAGConfig("
            f"graphdb={self.graphdb!r}, "
            f"llm={self.llm!r}, "
            f"vertexIDKey={self.vertexIDKey!r}, "
            f"vertexLabelKey={self.vertexLabelKey!r}, "
            f"edgeLabelKey={self.edgeLabelKey!r}, "
            f"ontology={self.ontology!r}, "
            f"ontologyClassKey={self.ontologyClassKey!r}, "
            f"categoryKey={self.categoryKey!r}, "
            f"defaultEdgeLabel={self.defaultEdgeLabel!r}, "
            f"tolerance={self.tolerance!r}, "
            f"maxCandidates={self.maxCandidates!r}, "
            f"maxPairs={self.maxPairs!r}, "
            f"silent={self.silent!r})"
        )


class GraphRAG:
    """
    A minimal graph-native retrieval-augmented generation workflow.

    The class follows the TopologicPy static-method style. Create a GraphRAG
    object with GraphRAG.ByParameters(...), then pass that object to the other
    static methods:

        grag = GraphRAG.ByParameters(graphdb=graphdb, llm=llm)
        result = GraphRAG.Generate(grag, graph, description="...")

    Supported actions
    -----------------
    - add_node: add a vertex and optionally connect it to an existing vertex.
    - connect: connect two existing vertices.
    - stop: stop the generation loop.
    """

    # ---------------------------------------------------------------------
    # Constructors
    # ---------------------------------------------------------------------
    @staticmethod
    def ByParameters(
        graphdb=None,
        llm=None,
        promptContext: str = None,
        vertexIDKey: str = "id",
        vertexLabelKey: str = "label",
        edgeLabelKey: str = "label",
        ontology: bool = True,
        ontologyClassKey: str = "ontology_class",
        categoryKey: str = "category",
        uriKey: str = "uri",
        sourceKey: str = "source",
        generatedByKey: str = "generated_by",
        derivedFromKey: str = "derived_from",
        defaultGraphOntologyClass: str = "top:KnowledgeGraph",
        defaultVertexOntologyClass: str = "top:Node",
        defaultEdgeOntologyClass: str = "top:Relationship",
        defaultVertexCategory: str = "node",
        defaultEdgeCategory: str = "relationship",
        defaultGeneratedBy: str = "GraphRAG.Generate",
        defaultEdgeLabel: str = "suggested",
        tolerance: float = 0.0001,
        maxCandidates: int = 12,
        maxPairs: int = 40,
        silent: bool = False,
    ):
        """
        Creates and returns a lightweight GraphRAG configuration object.
        """
        try:
            return _GraphRAGConfig(
                graphdb=graphdb,
                llm=llm,
                promptContext=promptContext,
                vertexIDKey=vertexIDKey,
                vertexLabelKey=vertexLabelKey,
                edgeLabelKey=edgeLabelKey,
                ontology=bool(ontology),
                ontologyClassKey=ontologyClassKey,
                categoryKey=categoryKey,
                uriKey=uriKey,
                sourceKey=sourceKey,
                generatedByKey=generatedByKey,
                derivedFromKey=derivedFromKey,
                defaultGraphOntologyClass=defaultGraphOntologyClass,
                defaultVertexOntologyClass=defaultVertexOntologyClass,
                defaultEdgeOntologyClass=defaultEdgeOntologyClass,
                defaultVertexCategory=defaultVertexCategory,
                defaultEdgeCategory=defaultEdgeCategory,
                defaultGeneratedBy=defaultGeneratedBy,
                defaultEdgeLabel=defaultEdgeLabel,
                tolerance=tolerance,
                maxCandidates=max(1, int(maxCandidates or 12)),
                maxPairs=max(1, int(maxPairs or 40)),
                silent=silent,
            )
        except Exception as e:
            if not silent:
                print(f"GraphRAG.ByParameters - Error: {e}. Returning None.")
            return None

    # ---------------------------------------------------------------------
    # Public workflow methods
    # ---------------------------------------------------------------------
    @staticmethod
    def Generate(
        grag,
        graph,
        description: str = "",
        maxSteps: int = 10,
        patience: int = 4,
        automatic: bool = False,
        approvalFunction: Callable[[Dict[str, Any], str], str] = None,
        verbose: bool = True,
        silent: bool = False,
    ) -> Dict[str, Any]:
        """
        Iteratively grows or edits the input graph using corpus evidence and an LLM.
        """

        import re

        def _get_value(obj, key, default=None):
            if obj is None:
                return default
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        def _normalise_label(label):
            if label is None:
                return None
            label = str(label).strip()
            if not label:
                return None

            # Convert common LLM variants to corpus-style labels.
            label = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", label)
            label = label.lower()
            label = label.replace("-", "_")
            label = label.replace(" ", "_")
            label = re.sub(r"[^a-z0-9_]+", "", label)
            label = re.sub(r"_+", "_", label)
            label = label.strip("_")
            return label if label else None

        def _extract_label_from_best_example(best_example, fallback=None):
            """
            Accepts several plausible return formats from GraphDB.FindBestExampleForLabel:
            - "living_room"
            - {"label": "living_room", ...}
            - {"best_label": "living_room", ...}
            - {"node": {"label": "living_room"}, ...}
            - {"example": {"label": "living_room"}, ...}
            - tuple/list where one item is a dict containing a label
            """

            if best_example is None:
                return fallback

            if isinstance(best_example, str):
                return best_example

            if isinstance(best_example, dict):
                for key in ("label", "best_label", "canonical_label", "resolved_label", "node_label"):
                    value = best_example.get(key, None)
                    if isinstance(value, str) and value.strip():
                        return value

                for key in ("node", "vertex", "example", "best_example"):
                    value = best_example.get(key, None)
                    if isinstance(value, dict):
                        for label_key in ("label", "best_label", "canonical_label", "resolved_label", "node_label"):
                            label = value.get(label_key, None)
                            if isinstance(label, str) and label.strip():
                                return label

                return fallback

            if isinstance(best_example, (list, tuple)):
                for item in best_example:
                    label = _extract_label_from_best_example(item, fallback=None)
                    if isinstance(label, str) and label.strip():
                        return label

            return fallback

        def _manager_from_grag(grag):
            """
            Tries common names without assuming the internal structure of grag.
            """
            for key in ("manager", "graphDB", "graphDb", "graphdb", "db", "database"):
                value = _get_value(grag, key, None)
                if value is not None:
                    return value
            return grag

        def _find_best_example_for_label(grag, label, silent=False):
            """
            Calls GraphDB.FindBestExampleForLabel defensively, because the exact
            implementation/signature may vary between GraphDB backends.
            """

            manager = _manager_from_grag(grag)

            # Prefer a directly attached graphDB object if available.
            graph_db = None
            for key in ("GraphDB", "graphDB", "graphDb", "graphdb"):
                graph_db = _get_value(grag, key, None)
                if graph_db is not None:
                    break

            candidates = []

            if graph_db is not None and hasattr(graph_db, "FindBestExampleForLabel"):
                candidates.append(graph_db.FindBestExampleForLabel)

            # Fall back to a globally/imported GraphDB class.
            try:
                from topologicpy.GraphDB import GraphDB
                if hasattr(GraphDB, "FindBestExampleForLabel"):
                    candidates.append(GraphDB.FindBestExampleForLabel)
            except Exception:
                pass

            last_error = None

            for fn in candidates:
                call_patterns = (
                    lambda: fn(manager, label, silent=silent),
                    lambda: fn(manager, label),
                    lambda: fn(grag, label, silent=silent),
                    lambda: fn(grag, label),
                    lambda: fn(label, silent=silent),
                    lambda: fn(label),
                )

                for call in call_patterns:
                    try:
                        return call()
                    except TypeError as e:
                        last_error = e
                        continue
                    except Exception as e:
                        last_error = e
                        break

            if not silent:
                print(f"GraphRAG.Generate - Warning: Could not call FindBestExampleForLabel for label '{label}'. Error: {last_error}")

            return None

        def _resolve_label_with_best_example(grag, label, silent=False):
            """
            Resolves an LLM-proposed label to a corpus label using:
            1. Raw label.
            2. Normalised label.
            3. Simple synonym fallback.
            4. FindBestExampleForLabel.
            """

            raw_label = label
            normalised_label = _normalise_label(label)

            if normalised_label is None:
                return None, raw_label, None

            synonyms = {
                "living": "living_room",
                "lounge": "living_room",
                "sitting_room": "living_room",
                "sitting": "living_room",
                "dining": "dining_room",
                "bed": "bedroom",
                "bath": "bathroom",
                "wc": "toilet",
                "water_closet": "toilet",
                "entry": "entrance",
                "entryway": "entrance",
                "hall": "corridor",
                "hallway": "corridor",
            }

            candidate_labels = []
            for candidate in (raw_label, normalised_label, synonyms.get(normalised_label, None)):
                if isinstance(candidate, str) and candidate.strip() and candidate not in candidate_labels:
                    candidate_labels.append(candidate)

            for candidate_label in candidate_labels:
                best_example = _find_best_example_for_label(
                    grag,
                    candidate_label,
                    silent=silent,
                )

                resolved_label = _extract_label_from_best_example(
                    best_example,
                    fallback=None,
                )

                if isinstance(resolved_label, str) and resolved_label.strip():
                    return resolved_label, raw_label, best_example

            return normalised_label, raw_label, None


        def _enrich_action_labels_from_state(state, action):
            """
            Adds a_label and b_label to connect/remove_edge actions when the action
            identifies nodes by id only.

            This is for clarity, validation, logging, and downstream corpus checks.
            """
            if not isinstance(state, dict) or not isinstance(action, dict):
                return action

            action = dict(action)
            action_name = str(action.get("action", "")).strip().lower()

            if action_name not in ("connect", "remove_edge"):
                return action

            ids = state.get("ids", []) or []
            labels = state.get("labels", []) or []

            id_to_label = {}
            for i, node_id in enumerate(ids):
                if i < len(labels):
                    id_to_label[str(node_id)] = labels[i]

            a_id = action.get("a_id") or action.get("src") or action.get("source") or action.get("from") or action.get("id")
            b_id = action.get("b_id") or action.get("dst") or action.get("target") or action.get("to") or action.get("attach_to_id")

            if a_id is not None and not action.get("a_label"):
                action["a_label"] = id_to_label.get(str(a_id))

            if b_id is not None and not action.get("b_label"):
                action["b_label"] = id_to_label.get(str(b_id))

            return action


        def _resolve_action_labels(grag, action, silent=False):
            """
            Resolves all plausible label-bearing fields in an LLM action before
            approval and before ApplyAction.

            This prevents raw LLM labels such as 'Living' from reaching the graph
            database when the corpus uses canonical labels such as 'living_room'.
            """


            if not isinstance(action, dict):
                return action

            resolved_action = GraphRAG.NormalizeAction(action)

            label_keys = (
                "label",
                "a_label",
                "b_label",
                "source_label",
                "target_label",
                "src_label",
                "dst_label",
                "from_label",
                "to_label",
                "neighbor_label",
                "neighbour_label",
                "node_label",
                "vertex_label",
                "new_label",
            )

            resolved_labels = {}

            for key in label_keys:
                if key not in resolved_action:
                    continue

                value = resolved_action.get(key, None)
                if not isinstance(value, str) or not value.strip():
                    continue

                resolved_label, raw_label, best_example = _resolve_label_with_best_example(
                    grag,
                    value,
                    silent=silent,
                )

                if isinstance(resolved_label, str) and resolved_label.strip():
                    resolved_action[key] = resolved_label
                    resolved_labels[key] = {
                        "raw": raw_label,
                        "resolved": resolved_label,
                        "best_example": best_example,
                    }

            if resolved_labels:
                existing = resolved_action.get("_resolved_labels", {})
                if not isinstance(existing, dict):
                    existing = {}
                existing.update(resolved_labels)
                resolved_action["_resolved_labels"] = existing

            return resolved_action

        effective_silent = GraphRAG._effective_silent(grag, silent)
        records = []
        stagnant = 0
        status = "completed"

        if grag is None:
            return {
                "ok": False,
                "status": "error",
                "message": "The input grag object is None.",
                "graph": graph,
                "steps": [],
            }

        if graph is None:
            return {
                "ok": False,
                "status": "error",
                "message": "The input graph is None.",
                "graph": graph,
                "steps": [],
            }

        for step in range(1, max(1, int(maxSteps or 1)) + 1):
            summary_before = GraphRAG.SummarizeGraph(
                grag,
                graph,
                silent=effective_silent,
            )

            evidence = GraphRAG.Evidence(
                grag,
                summary_before,
                silent=effective_silent,
            )

            raw_action = GraphRAG.PickAction(
                grag,
                summary_before,
                evidence,
                description=description,
                silent=effective_silent,
            )

            if not isinstance(raw_action, dict) or not raw_action:
                status = "llm_failed"
                records.append({
                    "step": step,
                    "ok": False,
                    "status": status,
                    "action": raw_action,
                    "message": "LLM returned no usable action.",
                    "summary_before": summary_before,
                    "summary_after": summary_before,
                    "evidence": evidence,
                })
                break

            action = _resolve_action_labels(
                grag,
                raw_action,
                silent=effective_silent,
            )
            action = _enrich_action_labels_from_state(state, action)

            if verbose and not effective_silent:
                print(f"STEP: {step}")
                print("json_action:", action.get("action") if isinstance(action, dict) else None)
                print("json_label:", action.get("label") if isinstance(action, dict) else None)
                print("json_id:", action.get("id") if isinstance(action, dict) else None)
                print("json_a_id:", action.get("a_id") if isinstance(action, dict) else None)
                print("json_b_id:", action.get("b_id") if isinstance(action, dict) else None)
                print("json_attach_to_id:", action.get("attach_to_id") if isinstance(action, dict) else None)
                print("json_a_label:", action.get("a_label") if isinstance(action, dict) else None)
                print("json_b_label:", action.get("b_label") if isinstance(action, dict) else None)
                print("json_attach_to_label:", action.get("attach_to_label") if isinstance(action, dict) else None)
                print("json_edge_label:", action.get("edge_label") if isinstance(action, dict) else None)
                if isinstance(action, dict) and action.get("_resolved_labels"):
                    print("resolved_labels:", action.get("_resolved_labels"))

            action_name = str(action.get("action", "stop")).strip().lower()

            if action_name in ("stop", "done", "finish"):
                status = "stopped"
                records.append({
                    "step": step,
                    "ok": True,
                    "status": status,
                    "action": action,
                    "raw_action": raw_action,
                    "message": action.get("reason", "The LLM chose to stop."),
                    "summary_before": summary_before,
                    "summary_after": summary_before,
                    "evidence": evidence,
                })

                if verbose and not effective_silent:
                    print("→ The process stopped.")

                break

            decision = "accept" if automatic else GraphRAG.ApproveAction(
                action,
                approvalFunction=approvalFunction,
                silent=effective_silent,
            )

            if decision == "stop":
                status = "stopped_by_user"
                records.append({
                    "step": step,
                    "ok": True,
                    "status": status,
                    "action": action,
                    "raw_action": raw_action,
                    "message": "Stopped by user.",
                    "summary_before": summary_before,
                    "summary_after": summary_before,
                    "evidence": evidence,
                })
                break

            if decision == "ignore":
                stagnant += 1
                records.append({
                    "step": step,
                    "ok": True,
                    "status": "ignored",
                    "action": action,
                    "raw_action": raw_action,
                    "message": "Action ignored.",
                    "summary_before": summary_before,
                    "summary_after": summary_before,
                    "evidence": evidence,
                })
            else:
                apply_result = GraphRAG.ApplyAction(
                    grag,
                    graph,
                    action,
                    silent=effective_silent,
                )

                if not isinstance(apply_result, dict):
                    apply_result = {
                        "ok": False,
                        "graph": graph,
                        "message": "GraphRAG.ApplyAction returned no usable result.",
                    }

                if apply_result.get("graph") is not None:
                    graph = apply_result.get("graph")

                summary_after = GraphRAG.SummarizeGraph(
                    grag,
                    graph,
                    silent=effective_silent,
                )

                changed = bool(apply_result.get("ok")) and (summary_after != summary_before)
                stagnant = 0 if changed else stagnant + 1

                records.append({
                    "step": step,
                    "ok": bool(apply_result.get("ok")),
                    "status": "applied" if apply_result.get("ok") else "failed",
                    "action": action,
                    "raw_action": raw_action,
                    "message": apply_result.get("message", ""),
                    "summary_before": summary_before,
                    "summary_after": summary_after,
                    "evidence": evidence,
                })

                if verbose and not effective_silent:
                    print("→", apply_result.get("message", ""))
                    print("Latest Number of Nodes:", summary_after.get("num_nodes"))
                    print("Latest Labels:", [n.get("label") for n in summary_after.get("nodes", [])])

            if stagnant >= max(1, int(patience or 1)):
                status = "patience_exhausted"
                if verbose and not effective_silent:
                    print("→ Stopping because patience was exhausted.")
                break
        else:
            status = "max_steps_reached"

        return {
            "ok": True,
            "status": status,
            "graph": graph,
            "steps": records,
            "num_steps": len(records),
        }
    @staticmethod
    def Generate_old(
        grag,
        graph,
        description: str = "",
        maxSteps: int = 10,
        patience: int = 4,
        automatic: bool = False,
        approvalFunction: Callable[[Dict[str, Any], str], str] = None,
        verbose: bool = True,
        silent: bool = False,
    ) -> Dict[str, Any]:
        """
        Iteratively grows or edits the input graph using corpus evidence and an LLM.
        """
        effective_silent = GraphRAG._effective_silent(grag, silent)
        records = []
        stagnant = 0
        status = "completed"

        if grag is None:
            return {"ok": False, "status": "error", "message": "The input grag object is None.", "graph": graph, "steps": []}
        if graph is None:
            return {"ok": False, "status": "error", "message": "The input graph is None.", "graph": graph, "steps": []}

        for step in range(1, max(1, int(maxSteps or 1)) + 1):
            summary_before = GraphRAG.SummarizeGraph(grag, graph, silent=effective_silent)
            evidence = GraphRAG.Evidence(grag, summary_before, silent=effective_silent)
            action = GraphRAG.PickAction(grag, summary_before, evidence, description=description, silent=effective_silent)

            if verbose and not effective_silent:
                print(f"STEP: {step}")
                print("json_action:", action.get("action") if isinstance(action, dict) else None)
                print("json_a_label:", action.get("a_label") if isinstance(action, dict) else None)
                print("json_b_label:", action.get("b_label") if isinstance(action, dict) else None)

            if not isinstance(action, dict) or not action:
                status = "llm_failed"
                records.append({
                    "step": step,
                    "ok": False,
                    "status": status,
                    "action": action,
                    "message": "LLM returned no usable action.",
                    "summary_before": summary_before,
                    "summary_after": summary_before,
                    "evidence": evidence,
                })
                break

            action_name = str(action.get("action", "stop")).strip().lower()
            if action_name in ("stop", "done", "finish"):
                status = "stopped"
                records.append({
                    "step": step,
                    "ok": True,
                    "status": status,
                    "action": action,
                    "message": action.get("reason", "The LLM chose to stop."),
                    "summary_before": summary_before,
                    "summary_after": summary_before,
                    "evidence": evidence,
                })
                if verbose and not effective_silent:
                    print("→ The process stopped.")
                break

            decision = "accept" if automatic else GraphRAG.ApproveAction(
                action,
                approvalFunction=approvalFunction,
                silent=effective_silent,
            )

            if decision == "stop":
                status = "stopped_by_user"
                records.append({
                    "step": step,
                    "ok": True,
                    "status": status,
                    "action": action,
                    "message": "Stopped by user.",
                    "summary_before": summary_before,
                    "summary_after": summary_before,
                    "evidence": evidence,
                })
                break

            if decision == "ignore":
                stagnant += 1
                records.append({
                    "step": step,
                    "ok": True,
                    "status": "ignored",
                    "action": action,
                    "message": "Action ignored.",
                    "summary_before": summary_before,
                    "summary_after": summary_before,
                    "evidence": evidence,
                })
            else:
                apply_result = GraphRAG.ApplyAction(grag, graph, action, silent=effective_silent)
                if isinstance(apply_result, dict) and apply_result.get("graph") is not None:
                    graph = apply_result.get("graph")
                summary_after = GraphRAG.SummarizeGraph(grag, graph, silent=effective_silent)
                changed = bool(apply_result.get("ok")) and (summary_after != summary_before)
                stagnant = 0 if changed else stagnant + 1
                records.append({
                    "step": step,
                    "ok": bool(apply_result.get("ok")),
                    "status": "applied" if apply_result.get("ok") else "failed",
                    "action": action,
                    "message": apply_result.get("message", ""),
                    "summary_before": summary_before,
                    "summary_after": summary_after,
                    "evidence": evidence,
                })

                if verbose and not effective_silent:
                    print("→", apply_result.get("message", ""))
                    print("Latest Number of Nodes:", summary_after.get("num_nodes"))
                    print("Latest Labels:", [n.get("label") for n in summary_after.get("nodes", [])])

            if stagnant >= max(1, int(patience or 1)):
                status = "patience_exhausted"
                if verbose and not effective_silent:
                    print("→ Stopping because patience was exhausted.")
                break
        else:
            status = "max_steps_reached"

        return {"ok": True, "status": status, "graph": graph, "steps": records, "num_steps": len(records)}

    @staticmethod
    def PickAction(grag, graphSummary, evidence=None, description: str = "", silent: bool = False):
        """
        Asks the configured LLM to select the next graph-edit action.
        """
        effective_silent = GraphRAG._effective_silent(grag, silent)
        try:
            from topologicpy.LLM import LLM

            llm = getattr(grag, "llm", None)
            if llm is None:
                if not effective_silent:
                    print("GraphRAG.PickAction - Error: No LLM object was supplied. Returning {}.")
                return {}

            prompt = GraphRAG.Prompt(grag, graphSummary, evidence=evidence, description=description, silent=effective_silent)
            prompt += """

Return ONLY one valid JSON object. Do not use markdown. Do not use ```json. Do not explain.
The JSON object must follow this exact shape:
{
  "action": "add_node",
  "label": "Bedroom",
  "id": "n4",
  "attach_to_id": "n2",
  "attach_to_label": "Living",
  "edge_label": "connect",
  "x": 7.0,
  "y": 0.0,
  "z": 0.0,
  "reason": "Short reason."
}
Allowed action values are only:
- add_node
- connect
- stop
"""
            raw = LLM.Prompt(llm, prompt, temperature=0, maxOutputTokens=4096, silent=effective_silent)
            if not effective_silent:
                print("\nRAW LLM RESPONSE:")
                print(raw)
                print("END RAW LLM RESPONSE\n")

            action = GraphRAG._coerce_json(raw)
            if isinstance(action, list):
                action = action[0] if action else {}
            if not isinstance(action, dict) or not action.get("action"):
                if not effective_silent:
                    print("GraphRAG.PickAction - Error: Could not parse complete JSON action.")
                return {}
            return GraphRAG.NormalizeAction(action)
        except Exception as e:
            if not effective_silent:
                print(f"GraphRAG.PickAction - Error: {e}. Returning {{}}.")
            return {}

    @staticmethod
    def ApplyAction(grag, graph, action: Dict[str, Any], silent: bool = False) -> Dict[str, Any]:
        """
        Applies one action to the input graph.
        """
        effective_silent = GraphRAG._effective_silent(grag, silent)
        if graph is None:
            return {"ok": False, "graph": graph, "message": "The input graph is None."}
        if not isinstance(action, dict):
            return {"ok": False, "graph": graph, "message": "The input action is not a dictionary."}

        action = GraphRAG.NormalizeAction(action)
        name = str(action.get("action", "stop")).strip().lower()
        if name in ("stop", "done", "finish"):
            return {"ok": True, "graph": graph, "message": "No graph edit was applied."}
        if name == "add_node":
            return GraphRAG._apply_add_node(grag, graph, action, silent=effective_silent)
        if name == "connect":
            return GraphRAG._apply_connect(grag, graph, action, silent=effective_silent)
        return {"ok": False, "graph": graph, "message": f"Unsupported action: {name}."}

    @staticmethod
    def Evidence(grag, graphSummary: Dict[str, Any], silent: bool = False) -> Dict[str, Any]:
        """
        Retrieves compact corpus evidence from GraphDB for the current graph state.
        """
        effective_silent = GraphRAG._effective_silent(grag, silent)
        labels = [n.get("label") for n in (graphSummary or {}).get("nodes", []) if n.get("label")]
        labels = GraphRAG._unique(labels)
        evidence = {"candidate_counts": [], "max_neighbors": {}, "pairs": [], "expandable_nodes": []}

        graphdb = getattr(grag, "graphdb", None)
        if graphdb is None:
            return evidence

        try:
            from topologicpy.GraphDB import GraphDB
        except Exception as e:
            if not effective_silent:
                print(f"GraphRAG.Evidence - Error importing GraphDB: {e}.")
            return evidence

        try:
            evidence["candidate_counts"] = GraphDB.CandidateCountsForLabels(
                graphdb,
                labels,
                limit=getattr(grag, "maxCandidates", 12),
                silent=effective_silent,
            ) or []
        except Exception as e:
            evidence["candidate_counts"] = []
            if not effective_silent:
                print(f"GraphRAG.Evidence - Warning: CandidateCountsForLabels failed: {e}")

        for label in labels:
            try:
                evidence["max_neighbors"][label] = GraphDB.MaxNeighborsForLabel(graphdb, label, silent=effective_silent)
            except Exception:
                evidence["max_neighbors"][label] = None

        candidate_labels = GraphRAG._candidate_labels(evidence.get("candidate_counts", []))
        all_labels = GraphRAG._unique(labels + candidate_labels)[: max(1, getattr(grag, "maxCandidates", 12) * 2)]
        try:
            pairs = GraphDB.FetchAllPairs(graphdb, undirected=True, silent=effective_silent) or []
            evidence["pairs"] = GraphRAG._filter_pairs_by_labels(pairs, all_labels, limit=getattr(grag, "maxPairs", 40))
        except Exception as e:
            evidence["pairs"] = []
            if not effective_silent:
                print(f"GraphRAG.Evidence - Warning: FetchAllPairs failed: {e}")

        evidence["expandable_nodes"] = GraphRAG.ExpandableNodes(graphSummary, evidence, silent=effective_silent)
        return evidence

    @staticmethod
    def ExpandableNodes(graphSummary: Dict[str, Any], evidence: Dict[str, Any] = None, silent: bool = False) -> List[Dict[str, Any]]:
        """
        Returns nodes whose current degree is lower than the maximum found in the corpus.
        """
        evidence = evidence or {}
        max_neighbors = evidence.get("max_neighbors", {}) or {}
        nodes = (graphSummary or {}).get("nodes", []) or []
        out = []
        for n in nodes:
            label = n.get("label")
            degree = int(n.get("degree", 0) or 0)
            raw_max = max_neighbors.get(label)
            max_degree = GraphRAG._max_neighbor_value(raw_max)
            if max_degree is None:
                out.append({**n, "max_degree": None, "reason": "not_found_in_db"})
            elif degree < max_degree:
                out.append({**n, "max_degree": max_degree, "reason": "below_corpus_max"})
        return out

    @staticmethod
    def SummarizeGraph(grag, graph, silent: bool = False) -> Dict[str, Any]:
        """
        Summarises the current working graph as JSON-friendly nodes and edges.
        """
        effective_silent = GraphRAG._effective_silent(grag, silent)
        try:
            from topologicpy.Vertex import Vertex
            from topologicpy.Graph import Graph
            from topologicpy.Topology import Topology
            from topologicpy.Dictionary import Dictionary

            vertex_id_key = getattr(grag, "vertexIDKey", "id")
            vertex_label_key = getattr(grag, "vertexLabelKey", "label")
            edge_label_key = getattr(grag, "edgeLabelKey", "label")
            vertices = Graph.Vertices(graph) or []
            edges = Graph.Edges(graph) or []
            nodes = []

            for i, v in enumerate(vertices):
                d = Topology.Dictionary(v)
                props = GraphRAG._python_dictionary(d)
                vid = Dictionary.ValueAtKey(d, vertex_id_key, None)
                if vid is None or str(vid).strip() == "":
                    vid = props.get(vertex_id_key) or f"n{i}"
                label = Dictionary.ValueAtKey(d, vertex_label_key, None)
                if label is None or str(label).strip() == "":
                    label = props.get(vertex_label_key) or str(vid)
                x, y, z = GraphRAG._vertex_xyz(v)
                degree = len(Graph.AdjacentVertices(graph, v) or [])
                props = GraphRAG._ensure_ontology_props(
                    grag,
                    props,
                    entity="vertex",
                    label=label,
                    entity_id=vid,
                    generated_by="GraphRAG.SummarizeGraph",
                )
                ontology_class = GraphRAG._ontology_class_from_props(
                    grag,
                    props,
                    default=getattr(grag, "defaultVertexOntologyClass", "top:Node"),
                )
                nodes.append({
                    "id": str(vid),
                    "label": str(label),
                    "ontology_class": ontology_class,
                    "category": props.get(GraphRAG._ontology_key(grag, "categoryKey", "category")),
                    "degree": int(degree),
                    "x": x,
                    "y": y,
                    "z": z,
                    "props": props,
                })

            out_edges = []
            for e in edges:
                src = None
                dst = None
                try:
                    sv = GraphRAG._start_vertex(e)
                    ev = GraphRAG._end_vertex(e)
                    si = Vertex.Index(vertex=sv, vertices=vertices, strict=False, tolerance=getattr(grag, "tolerance", 0.0001))
                    ei = Vertex.Index(vertex=ev, vertices=vertices, strict=False, tolerance=getattr(grag, "tolerance", 0.0001))
                    if si is not None and 0 <= si < len(nodes):
                        src = nodes[si]["id"]
                    if ei is not None and 0 <= ei < len(nodes):
                        dst = nodes[ei]["id"]
                except Exception:
                    src = None
                    dst = None
                d = Topology.Dictionary(e)
                props = GraphRAG._python_dictionary(d)
                label = Dictionary.ValueAtKey(d, edge_label_key, props.get(edge_label_key, "connect"))
                props = GraphRAG._ensure_ontology_props(
                    grag,
                    props,
                    entity="edge",
                    label=label,
                    entity_id=f"{src}_{dst}" if src is not None and dst is not None else None,
                    generated_by="GraphRAG.SummarizeGraph",
                )
                ontology_class = GraphRAG._ontology_class_from_props(
                    grag,
                    props,
                    default=getattr(grag, "defaultEdgeOntologyClass", "top:Relationship"),
                )
                out_edges.append({
                    "src": src,
                    "dst": dst,
                    "label": str(label),
                    "ontology_class": ontology_class,
                    "category": props.get(GraphRAG._ontology_key(grag, "categoryKey", "category")),
                    "props": props,
                })

            return {"nodes": nodes, "edges": out_edges, "num_nodes": len(nodes), "num_edges": len(out_edges)}
        except Exception as e:
            if not effective_silent:
                print(f"GraphRAG.SummarizeGraph - Error: {e}. Returning empty summary.")
            return {"nodes": [], "edges": [], "num_nodes": 0, "num_edges": 0}

    @staticmethod
    def Prompt(grag, graphSummary: Dict[str, Any], evidence: Dict[str, Any] = None, description: str = "", silent: bool = False) -> str:
        """
        Builds the prompt used for graph-edit action selection.
        """
        evidence = evidence or {}
        payload = {
            "target_description": description or "",
            "current_graph": {
                "nodes": [{
                    "id": n["id"],
                    "label": n["label"],
                    "ontology_class": n.get("ontology_class"),
                    "category": n.get("category"),
                    "degree": n["degree"],
                } for n in (graphSummary or {}).get("nodes", [])],
                "edges": [{
                    "src": e["src"],
                    "dst": e["dst"],
                    "label": e["label"],
                    "ontology_class": e.get("ontology_class"),
                    "category": e.get("category"),
                } for e in (graphSummary or {}).get("edges", [])],
            },
            "corpus_evidence": {
                "candidate_counts": (evidence.get("candidate_counts") or [])[: getattr(grag, "maxCandidates", 12)],
                "max_neighbors": evidence.get("max_neighbors") or {},
                "expandable_nodes": evidence.get("expandable_nodes") or [],
                "pairs": (evidence.get("pairs") or [])[: getattr(grag, "maxPairs", 40)],
            },
            "allowed_actions": ["add_node", "connect", "stop"],
            "rules": [
                "Return one JSON object only.",
                "Use action='add_node' to add a new node and optionally connect it to an existing node.",
                "Any new node id must be unique and must not appear in current_graph.nodes[*].id.",
                "Use action='connect' only when both nodes already exist in current_graph.nodes.",
                "Use action='stop' when the graph already satisfies the target description or no justified edit remains.",
                "Prefer candidate labels and label-pairs supported by corpus_evidence.",
                "Preserve ontology_class and category metadata when they are present.",
                "New nodes should use ontology_class='top:Node' unless corpus evidence gives a more specific class.",
                "New edges should use ontology_class='top:Relationship' unless corpus evidence gives a more specific class.",
                "Do not invent existing node ids. Existing ids must come from current_graph.nodes.",
                "Use only node ids that appear in current_graph.nodes[*].id.",
                "If unsure, use attach_to_label instead of attach_to_id.",
                "Coordinates x, y, z are optional. Omit them unless necessary.",
                "Do not connect a node if its current degree is already equal to or greater than the corpus maximum degree for that label.",
            ],
        }
        context = getattr(grag, "promptContext", None)
        if context is None or len(str(context).strip()) < 1:
            context = "You are assisting with graph-native structure generation."
        prompt = context.strip()
        if not prompt.endswith("."):
            prompt += "."
        prompt += " Choose exactly one next graph-edit action.\n\n"
        prompt += json.dumps(payload, ensure_ascii=False, indent=2)
        return prompt

    @staticmethod
    def ActionSchema() -> Dict[str, Any]:
        """
        Returns the JSON schema for graph-edit actions.
        """
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["add_node", "connect", "stop"]},
                "label": {"type": "string"},
                "id": {"type": "string"},
                "a_id": {"type": "string"},
                "b_id": {"type": "string"},
                "a_label": {"type": "string"},
                "b_label": {"type": "string"},
                "attach_to_id": {"type": "string"},
                "attach_to_label": {"type": "string"},
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
                "edge_label": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["action", "reason"],
            "additionalProperties": True,
        }

    @staticmethod
    def NormalizeAction(action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalizes common action aliases and field aliases.
        """
        if not isinstance(action, dict):
            return {}

        out = dict(action)

        name = str(out.get("action", "stop")).strip().lower()
        aliases = {
            "add": "add_node",
            "add_vertex": "add_node",
            "add vertex": "add_node",
            "node": "add_node",

            "connect_vertices": "connect",
            "connect vertices": "connect",
            "connect_nodes": "connect",
            "connect nodes": "connect",
            "add_edge": "connect",
            "add edge": "connect",
            "edge": "connect",
            "link": "connect",
            "join": "connect",

            "remove": "remove_node",
            "delete": "remove_node",
            "delete_node": "remove_node",
            "delete node": "remove_node",
            "remove_vertex": "remove_node",
            "remove vertex": "remove_node",
            "delete_vertex": "remove_node",
            "delete vertex": "remove_node",

            "remove_edge": "remove_edge",
            "remove edge": "remove_edge",
            "delete_edge": "remove_edge",
            "delete edge": "remove_edge",
            "unlink": "remove_edge",
            "disconnect": "remove_edge",

            "finish": "stop",
            "done": "stop",
        }
        out["action"] = aliases.get(name, name)

        # --------------------------------------------------
        # add_node aliases
        # --------------------------------------------------
        if out["action"] == "add_node":
            if "label" not in out:
                for key in ("b_label", "new_label", "node_label", "vertex_label"):
                    if key in out and out.get(key) is not None:
                        out["label"] = out.get(key)
                        break

            if "attach_to_id" not in out:
                for key in ("a_id", "src", "source", "from", "from_id", "source_id", "start", "start_id"):
                    if key in out and out.get(key) is not None:
                        out["attach_to_id"] = out.get(key)
                        break

            if "attach_to_label" not in out:
                for key in ("a_label", "src_label", "source_label", "from_label", "start_label"):
                    if key in out and out.get(key) is not None:
                        out["attach_to_label"] = out.get(key)
                        break

        # --------------------------------------------------
        # connect aliases
        # --------------------------------------------------
        if out["action"] == "connect":
            # The LLM often returns add_node-shaped fields for connect.
            if "a_id" not in out:
                for key in ("a_id", "src", "source", "from", "from_id", "source_id", "start", "start_id", "id"):
                    if key in out and out.get(key) is not None:
                        out["a_id"] = out.get(key)
                        break

            if "b_id" not in out:
                for key in ("b_id", "dst", "target", "to", "to_id", "target_id", "end", "end_id", "attach_to_id"):
                    if key in out and out.get(key) is not None:
                        out["b_id"] = out.get(key)
                        break

            if "a_label" not in out:
                for key in ("a_label", "src_label", "source_label", "from_label", "start_label"):
                    if key in out and out.get(key) is not None:
                        out["a_label"] = out.get(key)
                        break

            if "b_label" not in out:
                for key in ("b_label", "dst_label", "target_label", "to_label", "end_label", "attach_to_label"):
                    if key in out and out.get(key) is not None:
                        out["b_label"] = out.get(key)
                        break

            # In connect actions, "label" is commonly the edge label, not a node label.
            if "edge_label" not in out and out.get("label") is not None:
                out["edge_label"] = out.get("label")

            # Prevent later label-resolution code from treating this as a node label.
            if "label" in out:
                out["_raw_label"] = out.get("label")
                out.pop("label", None)

        # --------------------------------------------------
        # remove_node aliases
        # --------------------------------------------------
        if out["action"] == "remove_node":
            if "id" not in out:
                for key in ("node_id", "vertex_id", "a_id", "src", "source", "target"):
                    if key in out and out.get(key) is not None:
                        out["id"] = out.get(key)
                        break

            if "label" not in out:
                for key in ("node_label", "vertex_label", "a_label", "src_label", "source_label", "target_label"):
                    if key in out and out.get(key) is not None:
                        out["label"] = out.get(key)
                        break

        # --------------------------------------------------
        # remove_edge aliases
        # --------------------------------------------------
        if out["action"] == "remove_edge":
            if "a_id" not in out:
                for key in ("src", "source", "from", "from_id", "source_id", "start", "start_id", "id"):
                    if key in out and out.get(key) is not None:
                        out["a_id"] = out.get(key)
                        break

            if "b_id" not in out:
                for key in ("dst", "target", "to", "to_id", "target_id", "end", "end_id", "attach_to_id"):
                    if key in out and out.get(key) is not None:
                        out["b_id"] = out.get(key)
                        break

            if "a_label" not in out:
                for key in ("src_label", "source_label", "from_label", "start_label"):
                    if key in out and out.get(key) is not None:
                        out["a_label"] = out.get(key)
                        break

            if "b_label" not in out:
                for key in ("dst_label", "target_label", "to_label", "end_label", "attach_to_label"):
                    if key in out and out.get(key) is not None:
                        out["b_label"] = out.get(key)
                        break
        return out
    @staticmethod
    def ApproveAction(action: Dict[str, Any], approvalFunction: Callable[[Dict[str, Any], str], str] = None, silent: bool = False) -> str:
        """
        Requests approval for an action.
        """
        message = json.dumps(action or {}, ensure_ascii=False, indent=2)
        if approvalFunction is not None:
            try:
                decision = str(approvalFunction(action, message)).strip().lower()
                return decision if decision in ("accept", "ignore", "stop") else "ignore"
            except Exception:
                return "ignore"
        if silent:
            return "ignore"
        print("Proposed action:")
        print(message)
        try:
            value = input("Apply this action? [y]es / [n]o / [s]top: ").strip().lower()
        except Exception:
            return "ignore"
        if value in ("y", "yes", "a", "accept", "apply"):
            return "accept"
        if value in ("s", "stop", "q", "quit"):
            return "stop"
        return "ignore"

    # ---------------------------------------------------------------------
    # Local workflow helpers retained outside Graph API by design
    # ---------------------------------------------------------------------
    @staticmethod
    def unique_id(graph=None, prefix: str = "n", key: str = "id", silent: bool = False) -> str:
        """
        Returns a unique vertex ID for the input graph.
        """
        try:
            existing = set()
            if graph is not None:
                from topologicpy.Graph import Graph
                from topologicpy.Topology import Topology
                from topologicpy.Dictionary import Dictionary
                for v in Graph.Vertices(graph) or []:
                    d = Topology.Dictionary(v)
                    value = Dictionary.ValueAtKey(d, key, None)
                    if value is not None:
                        existing.add(str(value))
            while True:
                candidate = f"{prefix}{uuid.uuid4().hex[:8]}"
                if candidate not in existing:
                    return candidate
        except Exception:
            return f"{prefix}{uuid.uuid4().hex[:8]}"

    @staticmethod
    def list_working_nodes_edges(grag, graph, silent: bool = False) -> Dict[str, Any]:
        """
        Returns the current graph nodes and edges.
        """
        summary = GraphRAG.SummarizeGraph(grag, graph, silent=silent)
        return {"nodes": summary.get("nodes", []), "edges": summary.get("edges", [])}

    @staticmethod
    def _apply_add_node(grag, graph, action: Dict[str, Any], silent: bool = False) -> Dict[str, Any]:
        try:
            from topologicpy.Vertex import Vertex
            from topologicpy.Edge import Edge
            from topologicpy.Graph import Graph
            from topologicpy.Topology import Topology
            from topologicpy.Dictionary import Dictionary

            vertex_id_key = getattr(grag, "vertexIDKey", "id")
            vertex_label_key = getattr(grag, "vertexLabelKey", "label")
            edge_label_key = getattr(grag, "edgeLabelKey", "label")
            tolerance = getattr(grag, "tolerance", 0.0001)

            label = str(action.get("label") or action.get("b_label") or "Node")
            vid = GraphRAG._unique_action_id(graph, action.get("id"), prefix="n", key=vertex_id_key, silent=silent)
            x = float(action.get("x", 0.0) or 0.0)
            y = float(action.get("y", 0.0) or 0.0)
            z = float(action.get("z", 0.0) or 0.0)

            vertex = Vertex.ByCoordinates(x, y, z)
            props = GraphRAG._ensure_ontology_props(
                grag,
                {
                    vertex_id_key: vid,
                    vertex_label_key: label,
                },
                entity="vertex",
                label=label,
                entity_id=vid,
                generated_by="GraphRAG._apply_add_node",
            )
            d = Dictionary.ByKeysValues(list(props.keys()), list(props.values()))
            vertex = Topology.SetDictionary(vertex, d)
            graph = Graph.AddVertex(graph, vertex, tolerance=tolerance, silent=silent)

            attach_vertex = None
            attach_id = action.get("attach_to_id") or action.get("a_id")
            attach_label = action.get("attach_to_label") or action.get("a_label")
            if attach_id:
                attach_vertex = GraphRAG._find_vertex_by_id(grag, graph, attach_id)
            if attach_vertex is None and attach_label:
                attach_vertex = GraphRAG._find_vertex_by_label(grag, graph, attach_label)

            if attach_vertex is not None:
                if not GraphRAG._can_add_connection(grag, graph, attach_vertex, silent=silent):
                    return {"ok": False, "graph": graph, "vertex": vertex, "message": GraphRAG._degree_limit_message(grag, graph, attach_vertex)}
                edge = Edge.ByVertices([attach_vertex, vertex], tolerance=tolerance, silent=True)
                if edge is not None:
                    edge_label = action.get("edge_label") or getattr(grag, "defaultEdgeLabel", "suggested")
                    edge_props = GraphRAG._ensure_ontology_props(
                        grag,
                        {edge_label_key: edge_label},
                        entity="edge",
                        label=edge_label,
                        entity_id=f"{attach_id or 'unknown'}_{vid}",
                        generated_by="GraphRAG._apply_add_node",
                    )
                    ed = Dictionary.ByKeysValues(list(edge_props.keys()), list(edge_props.values()))
                    edge = Topology.SetDictionary(edge, ed)
                    graph = Graph.AddEdge(graph, edge, transferVertexDictionaries=False, transferEdgeDictionaries=False, tolerance=tolerance, silent=silent)

            return {"ok": True, "graph": graph, "vertex": vertex, "message": f"Added node '{label}' with id '{vid}'."}
        except Exception as e:
            if not silent:
                print(f"GraphRAG._apply_add_node - Error: {e}.")
            return {"ok": False, "graph": graph, "message": str(e)}

    @staticmethod
    def _apply_connect(grag, graph, action: Dict[str, Any], silent: bool = False) -> Dict[str, Any]:
        try:
            from topologicpy.Edge import Edge
            from topologicpy.Graph import Graph
            from topologicpy.Topology import Topology
            from topologicpy.Dictionary import Dictionary

            edge_label_key = getattr(grag, "edgeLabelKey", "label")
            tolerance = getattr(grag, "tolerance", 0.0001)

            a = GraphRAG._find_vertex_by_id(grag, graph, action.get("a_id"))
            b = GraphRAG._find_vertex_by_id(grag, graph, action.get("b_id"))
            if a is None and action.get("a_label"):
                a = GraphRAG._find_vertex_by_label(grag, graph, action.get("a_label"))
            if b is None and action.get("b_label"):
                b = GraphRAG._find_vertex_by_label(grag, graph, action.get("b_label"))
            if a is None or b is None:
                return {"ok": False, "graph": graph, "message": "Could not find both vertices to connect."}
            if not GraphRAG._can_add_connection(grag, graph, a, silent=silent):
                return {"ok": False, "graph": graph, "message": GraphRAG._degree_limit_message(grag, graph, a)}
            if not GraphRAG._can_add_connection(grag, graph, b, silent=silent):
                return {"ok": False, "graph": graph, "message": GraphRAG._degree_limit_message(grag, graph, b)}

            edge = Edge.ByVertices([a, b], tolerance=tolerance, silent=True)
            if edge is None:
                return {"ok": False, "graph": graph, "message": "Could not create edge."}
            edge_label = action.get("edge_label") or getattr(grag, "defaultEdgeLabel", "suggested")
            edge_props = GraphRAG._ensure_ontology_props(
                grag,
                {edge_label_key: edge_label},
                entity="edge",
                label=edge_label,
                entity_id=f"{action.get('a_id') or action.get('a_label')}_{action.get('b_id') or action.get('b_label')}",
                generated_by="GraphRAG._apply_connect",
            )
            ed = Dictionary.ByKeysValues(list(edge_props.keys()), list(edge_props.values()))
            edge = Topology.SetDictionary(edge, ed)
            graph = Graph.AddEdge(graph, edge, transferVertexDictionaries=False, transferEdgeDictionaries=False, tolerance=tolerance, silent=silent)
            return {"ok": True, "graph": graph, "edge": edge, "message": "Connected vertices."}
        except Exception as e:
            if not silent:
                print(f"GraphRAG._apply_connect - Error: {e}.")
            return {"ok": False, "graph": graph, "message": str(e)}

    @staticmethod
    def _find_vertex_by_id(grag, graph, value):
        if value is None:
            return None
        try:
            from topologicpy.Graph import Graph
            from topologicpy.Topology import Topology
            from topologicpy.Dictionary import Dictionary
            key = getattr(grag, "vertexIDKey", "id")
            needle = str(value)
            for v in Graph.Vertices(graph) or []:
                if str(Dictionary.ValueAtKey(Topology.Dictionary(v), key, "")) == needle:
                    return v
        except Exception:
            return None
        return None

    @staticmethod
    def _find_vertex_by_label(grag, graph, value):
        if value is None:
            return None
        try:
            from topologicpy.Graph import Graph
            from topologicpy.Topology import Topology
            from topologicpy.Dictionary import Dictionary
            key = getattr(grag, "vertexLabelKey", "label")
            needle = str(value).lower()
            for v in Graph.Vertices(graph) or []:
                if str(Dictionary.ValueAtKey(Topology.Dictionary(v), key, "")).lower() == needle:
                    return v
        except Exception:
            return None
        return None

    @staticmethod
    def _can_add_connection(grag, graph, vertex, silent: bool = False) -> bool:
        try:
            from topologicpy.Graph import Graph
            from topologicpy.Topology import Topology
            from topologicpy.Dictionary import Dictionary
            from topologicpy.GraphDB import GraphDB
            graphdb = getattr(grag, "graphdb", None)
            if graphdb is None:
                return True
            label = Dictionary.ValueAtKey(Topology.Dictionary(vertex), getattr(grag, "vertexLabelKey", "label"), None)
            if label is None:
                return True
            current_degree = len(Graph.AdjacentVertices(graph, vertex) or [])
            max_degree = GraphDB.MaxNeighborsForLabel(graphdb, label, silent=silent)
            max_degree = GraphRAG._max_neighbor_value(max_degree)
            if max_degree is None:
                return True
            return current_degree < max_degree
        except Exception:
            return True

    @staticmethod
    def _degree_limit_message(grag, graph, vertex) -> str:
        try:
            from topologicpy.Graph import Graph
            from topologicpy.Topology import Topology
            from topologicpy.Dictionary import Dictionary
            from topologicpy.GraphDB import GraphDB
            label = Dictionary.ValueAtKey(Topology.Dictionary(vertex), getattr(grag, "vertexLabelKey", "label"), "Unknown")
            current_degree = len(Graph.AdjacentVertices(graph, vertex) or [])
            max_degree = None
            graphdb = getattr(grag, "graphdb", None)
            if graphdb is not None:
                max_degree = GraphRAG._max_neighbor_value(GraphDB.MaxNeighborsForLabel(graphdb, label, silent=True))
            return f"Degree limit reached for '{label}': current degree = {current_degree}, corpus max degree = {max_degree}."
        except Exception:
            return "Degree limit reached."

    @staticmethod
    def _unique_action_id(graph, suggested_id: str = None, prefix: str = "n", key: str = "id", silent: bool = False) -> str:
        try:
            from topologicpy.Graph import Graph
            from topologicpy.Topology import Topology
            from topologicpy.Dictionary import Dictionary
            existing = set()
            for v in Graph.Vertices(graph) or []:
                value = Dictionary.ValueAtKey(Topology.Dictionary(v), key, None)
                if value is not None:
                    existing.add(str(value).strip())
            suggested_id = str(suggested_id or "").strip()
            if suggested_id and suggested_id not in existing:
                return suggested_id
            i = 1
            while True:
                candidate = f"{prefix}{i}"
                if candidate not in existing:
                    return candidate
                i += 1
        except Exception:
            return GraphRAG.unique_id(graph, prefix=prefix, key=key, silent=silent)

    @staticmethod
    def _ontology_enabled(grag) -> bool:
        try:
            return bool(getattr(grag, "ontology", True))
        except Exception:
            return True

    @staticmethod
    def _ontology_key(grag, name: str, default: str) -> str:
        try:
            value = getattr(grag, name, default)
            return str(value) if value is not None else default
        except Exception:
            return default

    @staticmethod
    def _ensure_ontology_props(grag,
                               props: Dict[str, Any],
                               entity: str = "vertex",
                               label: Any = None,
                               entity_id: Any = None,
                               source: Any = None,
                               generated_by: Any = None) -> Dict[str, Any]:
        """
        Adds ontology metadata to a plain Python dictionary without overwriting
        existing user-supplied values.
        """
        props = dict(props or {})
        if not GraphRAG._ontology_enabled(grag):
            return props

        ontology_key = GraphRAG._ontology_key(grag, "ontologyClassKey", "ontology_class")
        category_key = GraphRAG._ontology_key(grag, "categoryKey", "category")
        uri_key = GraphRAG._ontology_key(grag, "uriKey", "uri")
        source_key = GraphRAG._ontology_key(grag, "sourceKey", "source")
        generated_by_key = GraphRAG._ontology_key(grag, "generatedByKey", "generated_by")

        entity_lc = str(entity or "").strip().lower()
        if entity_lc in ("graph", "knowledgegraph", "knowledge_graph"):
            default_class = getattr(grag, "defaultGraphOntologyClass", "top:KnowledgeGraph")
            default_category = "graph"
        elif entity_lc in ("edge", "relationship", "rel"):
            default_class = getattr(grag, "defaultEdgeOntologyClass", "top:Relationship")
            default_category = getattr(grag, "defaultEdgeCategory", "relationship")
        else:
            default_class = getattr(grag, "defaultVertexOntologyClass", "top:Node")
            default_category = getattr(grag, "defaultVertexCategory", "node")

        props.setdefault(ontology_key, default_class)
        props.setdefault(category_key, default_category)

        if label is not None and str(label).strip() != "":
            props.setdefault("label", str(label))

        if entity_id is not None and str(entity_id).strip() != "":
            # Use a lightweight, relative instance identifier. This avoids binding
            # generated graphs to a website while remaining TTL/GraphDB friendly.
            safe_id = str(entity_id).strip().replace(" ", "_")
            props.setdefault(uri_key, f"topologicpy:grag/{entity_lc}/{safe_id}")

        if source is not None and str(source).strip() != "":
            props.setdefault(source_key, str(source))

        generated_by = generated_by or getattr(grag, "defaultGeneratedBy", "GraphRAG.Generate")
        if generated_by is not None and str(generated_by).strip() != "":
            props.setdefault(generated_by_key, str(generated_by))

        return props

    @staticmethod
    def _ontology_class_from_props(grag, props: Dict[str, Any], default: str = None):
        props = props or {}
        key = GraphRAG._ontology_key(grag, "ontologyClassKey", "ontology_class")
        return props.get(key, props.get("ontologyClass", props.get("class", default)))

    @staticmethod
    def _coerce_json(text):
        if isinstance(text, dict):
            return text
        if isinstance(text, list):
            return text
        if text is None:
            return {}
        text = str(text).strip()
        if text.startswith("```"):
            text = text.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(text)
        except Exception:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                return {}
        return {}

    @staticmethod
    def _candidate_labels(candidate_counts) -> List[str]:
        labels = []
        for item in candidate_counts or []:
            if isinstance(item, dict):
                labels.append(item.get("label") or item.get("candidate") or item.get("neighbor_label") or item.get("value"))
            elif isinstance(item, (list, tuple)) and len(item) > 0:
                labels.append(item[0])
            elif item is not None:
                labels.append(item)
        return GraphRAG._unique(labels)

    @staticmethod
    def _filter_pairs_by_labels(pairs, labels, limit: int = 40) -> List[Any]:
        if not labels:
            return list(pairs or [])[:limit]
        label_set = {str(x).lower() for x in labels if x is not None}
        out = []
        for p in pairs or []:
            values = []
            if isinstance(p, dict):
                values = [p.get("a"), p.get("b"), p.get("src"), p.get("dst"), p.get("source"), p.get("target"), p.get("labelA"), p.get("labelB")]
            elif isinstance(p, (list, tuple)):
                values = list(p[:2])
            if any(str(v).lower() in label_set for v in values if v is not None):
                out.append(p)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _max_neighbor_value(value):
        if value is None:
            return None
        if isinstance(value, dict):
            for k in ("max_neighbors", "max", "degree", "count", "value"):
                if k in value:
                    return GraphRAG._max_neighbor_value(value[k])
            return None
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return None
            if len(value) == 1:
                return GraphRAG._max_neighbor_value(value[0])
            for item in reversed(value):
                mv = GraphRAG._max_neighbor_value(item)
                if mv is not None:
                    return mv
            return None
        try:
            return int(value)
        except Exception:
            try:
                return int(float(value))
            except Exception:
                return None

    @staticmethod
    def _vertex_xyz(vertex):
        try:
            from topologicpy.Vertex import Vertex
            return Vertex.X(vertex), Vertex.Y(vertex), Vertex.Z(vertex)
        except Exception:
            return None, None, None

    @staticmethod
    def _start_vertex(edge):
        try:
            from topologicpy.Edge import Edge
            return Edge.StartVertex(edge)
        except Exception:
            try:
                from topologicpy.Core import Core
                return Core.InstanceCall(edge, "StartVertex")
            except Exception:
                return None

    @staticmethod
    def _end_vertex(edge):
        try:
            from topologicpy.Edge import Edge
            return Edge.EndVertex(edge)
        except Exception:
            try:
                from topologicpy.Core import Core
                return Core.InstanceCall(edge, "EndVertex")
            except Exception:
                return None

    @staticmethod
    def _topology_key(topology) -> str:
        try:
            return str(id(topology))
        except Exception:
            return str(topology)

    @staticmethod
    def _python_dictionary(dictionary: Any) -> Dict[str, Any]:
        if dictionary is None:
            return {}
        if isinstance(dictionary, dict):
            return dict(dictionary)
        try:
            from topologicpy.Dictionary import Dictionary
            return dict(Dictionary.PythonDictionary(dictionary) or {})
        except Exception:
            return {}

    @staticmethod
    def _unique(values: List[Any]) -> List[Any]:
        seen = set()
        out = []
        for value in values or []:
            if value is None:
                continue
            value = str(value)
            if value not in seen:
                seen.add(value)
                out.append(value)
        return out


    # ---------------------------------------------------------------------
    # Matrix-backed generation helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _matrix_label_signature(label) -> str:
        """
        Returns a separator-insensitive label signature for matching labels.
        """
        if label is None:
            return ""
        import re
        label = str(label).strip().lower()
        label = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", label)
        return re.sub(r"[^a-z0-9]+", "", label)

    @staticmethod
    def _matrix_label_variants(label) -> List[str]:
        """
        Returns conservative label variants for corpus lookup.
        """
        if label is None:
            return []
        import re
        raw = str(label).strip()
        if not raw:
            return []
        spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", raw).strip().lower()
        spaced = re.sub(r"[_\-]+", " ", spaced)
        spaced = re.sub(r"\s+", " ", spaced).strip()
        underscored = re.sub(r"\s+", "_", spaced)
        compact = GraphRAG._matrix_label_signature(raw)
        synonyms = {
            "living": ["living room", "living_room", "lounge", "sitting room", "sitting_room"],
            "lounge": ["living room", "living_room"],
            "sitting": ["living room", "living_room"],
            "sittingroom": ["living room", "living_room"],
            "dining": ["dining room", "dining_room"],
            "bed": ["bedroom"],
            "bath": ["bathroom"],
            "wc": ["toilet", "water closet", "water_closet"],
            "entry": ["entrance", "entryway", "front door", "front_door"],
            "entryway": ["entrance"],
            "frontdoor": ["front door", "front_door", "entrance"],
            "hall": ["corridor", "hallway"],
            "hallway": ["corridor", "hall"],
        }
        variants = [raw, spaced, underscored, compact]
        variants += synonyms.get(compact, [])
        return GraphRAG._unique([v for v in variants if isinstance(v, str) and v.strip()])

    @staticmethod
    def _matrix_extract_label_from_best_example(best_example, fallback=None):
        """
        Extracts a label from common GraphDB.FindBestExampleForLabel return shapes.
        """
        if best_example is None:
            return fallback
        if isinstance(best_example, str):
            return best_example
        if isinstance(best_example, dict):
            for key in ("label", "best_label", "canonical_label", "resolved_label", "node_label", "vertex_label"):
                value = best_example.get(key)
                if isinstance(value, str) and value.strip():
                    return value
            for key in ("node", "vertex", "example", "best_example", "source", "target"):
                value = best_example.get(key)
                if isinstance(value, dict):
                    label = GraphRAG._matrix_extract_label_from_best_example(value, fallback=None)
                    if isinstance(label, str) and label.strip():
                        return label
            for value in best_example.values():
                if isinstance(value, (dict, list, tuple)):
                    label = GraphRAG._matrix_extract_label_from_best_example(value, fallback=None)
                    if isinstance(label, str) and label.strip():
                        return label
            return fallback
        if isinstance(best_example, (list, tuple)):
            for item in best_example:
                label = GraphRAG._matrix_extract_label_from_best_example(item, fallback=None)
                if isinstance(label, str) and label.strip():
                    return label
        return fallback

    @staticmethod
    def _matrix_find_best_example_for_label(grag, label, silent: bool = False):
        """
        Calls GraphDB.FindBestExampleForLabel on the configured graph database.
        """
        graphdb = getattr(grag, "graphdb", None)
        if graphdb is None or label is None:
            return None
        try:
            from topologicpy.GraphDB import GraphDB
        except Exception:
            try:
                from GraphDB import GraphDB
            except Exception:
                return None
        try:
            return GraphDB.FindBestExampleForLabel(graphdb, label, silent=silent)
        except TypeError:
            try:
                return GraphDB.FindBestExampleForLabel(graphdb, label)
            except Exception:
                return None
        except Exception:
            return None

    @staticmethod
    def _matrix_resolve_corpus_label(grag, label, fallback=None, return_example: bool = False, silent: bool = False):
        """
        Resolves a label to the actual corpus label when possible.
        """
        if label is None:
            return (fallback, None) if return_example else fallback
        for candidate in GraphRAG._matrix_label_variants(label):
            best_example = GraphRAG._matrix_find_best_example_for_label(grag, candidate, silent=silent)
            resolved = GraphRAG._matrix_extract_label_from_best_example(best_example, fallback=None)
            if isinstance(resolved, str) and resolved.strip():
                return (resolved, best_example) if return_example else resolved
        return (fallback, None) if return_example else fallback

    @staticmethod
    def _matrix_pair_values(pair) -> List[Any]:
        """
        Extracts plausible endpoint labels from a corpus pair record.
        """
        if isinstance(pair, dict):
            candidates = [
                (pair.get("a"), pair.get("b")),
                (pair.get("src"), pair.get("dst")),
                (pair.get("source"), pair.get("target")),
                (pair.get("labelA"), pair.get("labelB")),
                (pair.get("label_a"), pair.get("label_b")),
                (pair.get("a_label"), pair.get("b_label")),
                (pair.get("source_label"), pair.get("target_label")),
                (pair.get("src_label"), pair.get("dst_label")),
            ]
            for a, b in candidates:
                if a is not None and b is not None:
                    return [a, b]
            values = []
            for key in ("labels", "pair"):
                value = pair.get(key)
                if isinstance(value, (list, tuple)) and len(value) >= 2:
                    return [value[0], value[1]]
            return values
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            return [pair[0], pair[1]]
        return []

    @staticmethod
    def _matrix_corpus_pair_supported(grag, label_a, label_b, pairs=None, silent: bool = False) -> bool:
        """
        Returns True if a label pair is found in the corpus pair evidence.
        """
        graphdb = getattr(grag, "graphdb", None)
        if graphdb is None or label_a is None or label_b is None:
            return False
        if pairs is None:
            try:
                from topologicpy.GraphDB import GraphDB
            except Exception:
                try:
                    from GraphDB import GraphDB
                except Exception:
                    return False
            try:
                pairs = GraphDB.FetchAllPairs(graphdb, undirected=True, silent=silent) or []
            except Exception:
                pairs = []
        sig_a = GraphRAG._matrix_label_signature(label_a)
        sig_b = GraphRAG._matrix_label_signature(label_b)
        if not sig_a or not sig_b:
            return False
        for pair in pairs or []:
            values = GraphRAG._matrix_pair_values(pair)
            if len(values) < 2:
                continue
            pa = GraphRAG._matrix_label_signature(values[0])
            pb = GraphRAG._matrix_label_signature(values[1])
            if (pa == sig_a and pb == sig_b) or (pa == sig_b and pb == sig_a):
                return True
        return False

    @staticmethod
    def _matrix_seed_state(grag, graph, silent: bool = False) -> Dict[str, Any]:
        """
        Converts a TopologicPy graph into an editable Python matrix state.
        """
        summary = GraphRAG.SummarizeGraph(grag, graph, silent=silent)
        nodes = []
        seen_ids = set()
        for i, node in enumerate((summary or {}).get("nodes", []) or []):
            nid = str(node.get("id") if node.get("id") is not None else f"n{i}")
            if nid in seen_ids:
                nid = GraphRAG._matrix_unique_id({"nodes": nodes}, suggested_id=nid, prefix="n")
            seen_ids.add(nid)
            label = str(node.get("label") if node.get("label") is not None else nid)
            props = dict(node.get("props") or {})
            props = GraphRAG._ensure_ontology_props(
                grag,
                props,
                entity="vertex",
                label=label,
                entity_id=nid,
                generated_by="GraphRAG._matrix_seed_state",
            )
            nodes.append({
                "id": nid,
                "label": label,
                "ontology_class": GraphRAG._ontology_class_from_props(grag, props, getattr(grag, "defaultVertexOntologyClass", "top:Node")),
                "category": props.get(GraphRAG._ontology_key(grag, "categoryKey", "category")),
                "x": node.get("x"),
                "y": node.get("y"),
                "z": node.get("z"),
                "props": props,
            })
        id_to_index = {node["id"]: i for i, node in enumerate(nodes)}
        n = len(nodes)
        matrix = [[0 for _ in range(n)] for _ in range(n)]
        edge_labels = {}
        for edge in (summary or {}).get("edges", []) or []:
            src = edge.get("src")
            dst = edge.get("dst")
            if src in id_to_index and dst in id_to_index and src != dst:
                i = id_to_index[src]
                j = id_to_index[dst]
                matrix[i][j] = 1
                matrix[j][i] = 1
                edge_labels[tuple(sorted((src, dst)))] = str(edge.get("label") or getattr(grag, "defaultEdgeLabel", "suggested"))
        return {"nodes": nodes, "matrix": matrix, "edge_labels": edge_labels}

    @staticmethod
    def _matrix_summary(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Summarises an editable matrix state using the same shape as SummarizeGraph.
        """
        nodes = []
        matrix = state.get("matrix", []) or []
        for i, node in enumerate(state.get("nodes", []) or []):
            degree = 0
            if i < len(matrix):
                degree = sum(1 for v in matrix[i] if bool(v))
            props = dict(node.get("props") or {})
            props = GraphRAG._ensure_ontology_props(
                None,
                props,
                entity="vertex",
                label=node.get("label"),
                entity_id=node.get("id"),
                generated_by="GraphRAG._matrix_summary",
            )
            nodes.append({
                "id": str(node.get("id")),
                "label": str(node.get("label")),
                "ontology_class": props.get("ontology_class"),
                "category": props.get("category"),
                "degree": int(degree),
                "x": node.get("x"),
                "y": node.get("y"),
                "z": node.get("z"),
                "props": props,
            })
        edges = []
        edge_labels = state.get("edge_labels", {}) or {}
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                try:
                    if matrix[i][j]:
                        src = nodes[i]["id"]
                        dst = nodes[j]["id"]
                        label = edge_labels.get(tuple(sorted((src, dst))), "connect")
                        props = GraphRAG._ensure_ontology_props(
                            None,
                            {},
                            entity="edge",
                            label=label,
                            entity_id=f"{src}_{dst}",
                            generated_by="GraphRAG._matrix_summary",
                        )
                        edges.append({
                            "src": src,
                            "dst": dst,
                            "label": label,
                            "ontology_class": props.get("ontology_class"),
                            "category": props.get("category"),
                            "props": props,
                        })
                except Exception:
                    continue
        return {"nodes": nodes, "edges": edges, "num_nodes": len(nodes), "num_edges": len(edges)}

    @staticmethod
    def _matrix_unique_id(state: Dict[str, Any], suggested_id=None, prefix: str = "n") -> str:
        """
        Returns a unique node id in the editable matrix state.
        """
        existing = {str(n.get("id")) for n in state.get("nodes", []) or [] if n.get("id") is not None}
        suggested = str(suggested_id or "").strip()
        if suggested and suggested not in existing:
            return suggested
        i = 1
        while True:
            candidate = f"{prefix}{i}"
            if candidate not in existing:
                return candidate
            i += 1

    @staticmethod
    def _matrix_find_index_by_id(state: Dict[str, Any], value) -> Optional[int]:
        """
        Finds a node index by id.
        """
        if value is None:
            return None
        needle = str(value).strip()
        for i, node in enumerate(state.get("nodes", []) or []):
            if str(node.get("id")).strip() == needle:
                return i
        return None

    @staticmethod
    def _matrix_find_index_by_label(state: Dict[str, Any], value) -> Optional[int]:
        """
        Finds a node index by separator-insensitive label.
        """
        if value is None:
            return None
        sig = GraphRAG._matrix_label_signature(value)
        if not sig:
            return None
        for i, node in enumerate(state.get("nodes", []) or []):
            if GraphRAG._matrix_label_signature(node.get("label")) == sig:
                return i
        return None

    @staticmethod
    def _matrix_degree(state: Dict[str, Any], index: int) -> int:
        """
        Returns the degree of a node in the editable matrix state.
        """
        matrix = state.get("matrix", []) or []
        if index is None or index < 0 or index >= len(matrix):
            return 0
        return sum(1 for v in matrix[index] if bool(v))

    @staticmethod
    def _matrix_max_degree_for_label(grag, label, silent: bool = False):
        """
        Returns the corpus maximum degree for a label after resolving it to a corpus label.
        """
        graphdb = getattr(grag, "graphdb", None)
        if graphdb is None or label is None:
            return None
        try:
            from topologicpy.GraphDB import GraphDB
        except Exception:
            try:
                from GraphDB import GraphDB
            except Exception:
                return None
        resolved = GraphRAG._matrix_resolve_corpus_label(grag, label, fallback=label, silent=silent)
        candidates = GraphRAG._unique([resolved, label] + GraphRAG._matrix_label_variants(label))
        for candidate in candidates:
            try:
                value = GraphDB.MaxNeighborsForLabel(graphdb, candidate, silent=silent)
                max_degree = GraphRAG._max_neighbor_value(value)
                if max_degree is not None:
                    return max_degree
            except Exception:
                continue
        return None

    @staticmethod
    def _matrix_can_add_connection(grag, state: Dict[str, Any], index: int, silent: bool = False) -> bool:
        """
        Checks whether a node can accept one more connection according to corpus max degree.
        """
        if index is None:
            return False
        nodes = state.get("nodes", []) or []
        if index < 0 or index >= len(nodes):
            return False
        label = nodes[index].get("label")
        max_degree = GraphRAG._matrix_max_degree_for_label(grag, label, silent=silent)
        if max_degree is None:
            return True
        if max_degree <= 0:
            return True
        return GraphRAG._matrix_degree(state, index) < max_degree

    @staticmethod
    def _matrix_degree_limit_message(grag, state: Dict[str, Any], index: int, silent: bool = False) -> str:
        """
        Reports why a node cannot accept another connection.
        """
        try:
            node = (state.get("nodes", []) or [])[index]
            label = node.get("label", "Unknown")
            current_degree = GraphRAG._matrix_degree(state, index)
            max_degree = GraphRAG._matrix_max_degree_for_label(grag, label, silent=silent)
            return f"Degree limit reached for '{label}': current degree = {current_degree}, corpus max degree = {max_degree}."
        except Exception:
            return "Degree limit reached."

    @staticmethod
    def _matrix_connect_indices(grag, state: Dict[str, Any], i: int, j: int, edge_label: str = None, silent: bool = False):
        """
        Adds an undirected edge to the editable matrix state.
        """
        nodes = state.get("nodes", []) or []
        matrix = state.get("matrix", []) or []
        if i is None or j is None or i < 0 or j < 0 or i >= len(nodes) or j >= len(nodes):
            return {"ok": False, "message": "Could not find both nodes to connect."}
        if i == j:
            return {"ok": False, "message": "Cannot connect a node to itself."}
        if matrix[i][j]:
            return {"ok": False, "message": "The requested edge already exists."}
        if not GraphRAG._matrix_can_add_connection(grag, state, i, silent=silent):
            return {"ok": False, "message": GraphRAG._matrix_degree_limit_message(grag, state, i, silent=silent)}
        if not GraphRAG._matrix_can_add_connection(grag, state, j, silent=silent):
            return {"ok": False, "message": GraphRAG._matrix_degree_limit_message(grag, state, j, silent=silent)}
        matrix[i][j] = 1
        matrix[j][i] = 1
        key = tuple(sorted((str(nodes[i].get("id")), str(nodes[j].get("id")))))
        state.setdefault("edge_labels", {})[key] = str(edge_label or getattr(grag, "defaultEdgeLabel", "suggested"))
        return {"ok": True, "message": "Connected nodes."}

    @staticmethod
    def _matrix_apply_action(grag, state: Dict[str, Any], action: Dict[str, Any], evidence=None, silent: bool = False) -> Dict[str, Any]:
        """
        Applies a graph-edit action to the editable matrix state.
        """
        action = GraphRAG.NormalizeAction(action)
        name = str(action.get("action", "stop")).strip().lower()
        if name in ("stop", "done", "finish"):
            return {"ok": True, "state": state, "message": "No graph edit was applied."}

        if name == "add_node":
            label = str(action.get("label") or action.get("b_label") or "Node").strip() or "Node"
            resolved_label = GraphRAG._matrix_resolve_corpus_label(grag, label, fallback=label, silent=silent)
            label = resolved_label or label
            vid = GraphRAG._matrix_unique_id(state, action.get("id"), prefix="n")
            props = GraphRAG._ensure_ontology_props(
                grag,
                {"id": str(vid), "label": str(label)},
                entity="vertex",
                label=label,
                entity_id=vid,
                generated_by="GraphRAG._matrix_apply_action",
            )
            node = {
                "id": str(vid),
                "label": str(label),
                "ontology_class": GraphRAG._ontology_class_from_props(grag, props, getattr(grag, "defaultVertexOntologyClass", "top:Node")),
                "category": props.get(GraphRAG._ontology_key(grag, "categoryKey", "category")),
                "x": action.get("x"),
                "y": action.get("y"),
                "z": action.get("z"),
                "props": props,
            }
            state.setdefault("nodes", []).append(node)
            for row in state.setdefault("matrix", []):
                row.append(0)
            state["matrix"].append([0 for _ in range(len(state["nodes"]))])
            new_index = len(state["nodes"]) - 1

            attach_id = action.get("attach_to_id") or action.get("a_id") or action.get("src") or action.get("source") or action.get("from")
            attach_label = action.get("attach_to_label") or action.get("a_label") or action.get("src_label") or action.get("source_label") or action.get("from_label")
            attach_index = GraphRAG._matrix_find_index_by_id(state, attach_id)
            if attach_index is None and attach_label:
                attach_index = GraphRAG._matrix_find_index_by_label(state, attach_label)
            if attach_index is not None:
                connect_result = GraphRAG._matrix_connect_indices(grag, state, attach_index, new_index, action.get("edge_label"), silent=silent)
                if not connect_result.get("ok"):
                    return {"ok": True, "state": state, "message": f"Added node '{label}' with id '{vid}', but did not connect it. {connect_result.get('message', '')}"}
            return {"ok": True, "state": state, "message": f"Added node '{label}' with id '{vid}'."}

        if name == "connect":
            a_id = action.get("a_id") or action.get("src") or action.get("source") or action.get("from") or action.get("from_id") or action.get("source_id") or action.get("start") or action.get("start_id")
            b_id = action.get("b_id") or action.get("dst") or action.get("target") or action.get("to") or action.get("to_id") or action.get("target_id") or action.get("end") or action.get("end_id")
            a_label = action.get("a_label") or action.get("src_label") or action.get("source_label") or action.get("from_label") or action.get("start_label")
            b_label = action.get("b_label") or action.get("dst_label") or action.get("target_label") or action.get("to_label") or action.get("end_label")
            i = GraphRAG._matrix_find_index_by_id(state, a_id)
            j = GraphRAG._matrix_find_index_by_id(state, b_id)
            if i is None and a_label:
                i = GraphRAG._matrix_find_index_by_label(state, a_label)
            if j is None and b_label:
                j = GraphRAG._matrix_find_index_by_label(state, b_label)
            result = GraphRAG._matrix_connect_indices(grag, state, i, j, action.get("edge_label"), silent=silent)
            return {"ok": bool(result.get("ok")), "state": state, "message": result.get("message", "")}

        if name in ("remove_edge", "delete_edge"):
            pairs = (evidence or {}).get("all_pairs") or (evidence or {}).get("pairs")
            a_id = action.get("a_id") or action.get("src") or action.get("source") or action.get("from") or action.get("from_id") or action.get("source_id") or action.get("start") or action.get("start_id")
            b_id = action.get("b_id") or action.get("dst") or action.get("target") or action.get("to") or action.get("to_id") or action.get("target_id") or action.get("end") or action.get("end_id")
            a_label = action.get("a_label") or action.get("src_label") or action.get("source_label") or action.get("from_label") or action.get("start_label")
            b_label = action.get("b_label") or action.get("dst_label") or action.get("target_label") or action.get("to_label") or action.get("end_label")
            i = GraphRAG._matrix_find_index_by_id(state, a_id)
            j = GraphRAG._matrix_find_index_by_id(state, b_id)
            if i is None and a_label:
                i = GraphRAG._matrix_find_index_by_label(state, a_label)
            if j is None and b_label:
                j = GraphRAG._matrix_find_index_by_label(state, b_label)
            nodes = state.get("nodes", []) or []
            matrix = state.get("matrix", []) or []
            if i is None or j is None or i >= len(nodes) or j >= len(nodes):
                return {"ok": False, "state": state, "message": "Could not find both nodes for remove_edge."}
            if not matrix[i][j]:
                return {"ok": False, "state": state, "message": "The requested edge does not exist."}
            label_i = nodes[i].get("label")
            label_j = nodes[j].get("label")
            if GraphRAG._matrix_corpus_pair_supported(grag, label_i, label_j, pairs=pairs, silent=silent):
                return {"ok": False, "state": state, "message": f"Refused to remove corpus-supported edge '{label_i}' - '{label_j}'."}
            matrix[i][j] = 0
            matrix[j][i] = 0
            state.get("edge_labels", {}).pop(tuple(sorted((str(nodes[i].get("id")), str(nodes[j].get("id"))))), None)
            return {"ok": True, "state": state, "message": f"Removed unsupported edge '{label_i}' - '{label_j}'."}

        if name in ("remove_node", "delete_node"):
            node_id = action.get("id") or action.get("node_id") or action.get("a_id") or action.get("src")
            node_label = action.get("label") or action.get("node_label") or action.get("a_label") or action.get("src_label")
            i = GraphRAG._matrix_find_index_by_id(state, node_id)
            if i is None and node_label:
                i = GraphRAG._matrix_find_index_by_label(state, node_label)
            nodes = state.get("nodes", []) or []
            matrix = state.get("matrix", []) or []
            if i is None or i < 0 or i >= len(nodes):
                return {"ok": False, "state": state, "message": "Could not find node for remove_node."}
            label = nodes[i].get("label")
            supported = GraphRAG._matrix_resolve_corpus_label(grag, label, fallback=None, silent=silent)
            if supported is not None:
                return {"ok": False, "state": state, "message": f"Refused to remove corpus-supported node '{label}'."}
            removed_id = str(nodes[i].get("id"))
            removed_label = str(label)
            nodes.pop(i)
            matrix.pop(i)
            for row in matrix:
                if i < len(row):
                    row.pop(i)
            # Rebuild edge label keys because node ids may have been removed.
            valid_ids = {str(node.get("id")) for node in nodes}
            edge_labels = {}
            for key, value in (state.get("edge_labels", {}) or {}).items():
                if isinstance(key, tuple) and len(key) == 2 and key[0] in valid_ids and key[1] in valid_ids:
                    edge_labels[key] = value
            state["edge_labels"] = edge_labels
            return {"ok": True, "state": state, "message": f"Removed unsupported node '{removed_label}' with id '{removed_id}'."}

        return {"ok": False, "state": state, "message": f"Unsupported action: {name}."}

    @staticmethod
    def _matrix_materialise_graph(grag, state: Dict[str, Any], silent: bool = False):
        """
        Converts the editable matrix state to a TopologicPy graph at the end of generation.

        The materialised graph preserves ontology metadata on graph, vertex, and
        edge dictionaries. It avoids Graph.ByAdjacencyMatrix so edge dictionaries
        can be authored explicitly.
        """
        try:
            from topologicpy.Graph import Graph
            from topologicpy.Vertex import Vertex
            from topologicpy.Edge import Edge
            from topologicpy.Topology import Topology
            from topologicpy.Dictionary import Dictionary

            vertex_id_key = getattr(grag, "vertexIDKey", "id")
            vertex_label_key = getattr(grag, "vertexLabelKey", "label")
            edge_label_key = getattr(grag, "edgeLabelKey", "label")

            nodes = state.get("nodes", []) or []
            matrix = state.get("matrix", []) or []
            edge_labels = state.get("edge_labels", {}) or {}

            vertices = []
            for i, node in enumerate(nodes):
                x = node.get("x", None)
                y = node.get("y", None)
                z = node.get("z", None)
                try:
                    x = float(x) if x is not None else float(i)
                except Exception:
                    x = float(i)
                try:
                    y = float(y) if y is not None else 0.0
                except Exception:
                    y = 0.0
                try:
                    z = float(z) if z is not None else 0.0
                except Exception:
                    z = 0.0

                v = Vertex.ByCoordinates(x, y, z)
                props = dict(node.get("props") or {})
                props[vertex_id_key] = str(node.get("id"))
                props[vertex_label_key] = str(node.get("label"))
                props.setdefault("id", str(node.get("id")))
                props.setdefault("label", str(node.get("label")))
                props = GraphRAG._ensure_ontology_props(
                    grag,
                    props,
                    entity="vertex",
                    label=node.get("label"),
                    entity_id=node.get("id"),
                    generated_by="GraphRAG._matrix_materialise_graph",
                )
                d = Dictionary.ByKeysValues(list(props.keys()), list(props.values()))
                v = Topology.SetDictionary(v, d)
                vertices.append(v)

            edges = []
            for i in range(len(vertices)):
                for j in range(i + 1, len(vertices)):
                    try:
                        if not matrix[i][j]:
                            continue
                    except Exception:
                        continue
                    e = Edge.ByVertices([vertices[i], vertices[j]], silent=True)
                    if e is None:
                        continue
                    a_id = str(nodes[i].get("id"))
                    b_id = str(nodes[j].get("id"))
                    label = edge_labels.get(tuple(sorted((a_id, b_id))), getattr(grag, "defaultEdgeLabel", "suggested"))
                    props = GraphRAG._ensure_ontology_props(
                        grag,
                        {edge_label_key: label},
                        entity="edge",
                        label=label,
                        entity_id=f"{a_id}_{b_id}",
                        generated_by="GraphRAG._matrix_materialise_graph",
                    )
                    d = Dictionary.ByKeysValues(list(props.keys()), list(props.values()))
                    e = Topology.SetDictionary(e, d)
                    edges.append(e)

            if not vertices:
                return None

            graph = Graph.ByVerticesEdges(vertices, edges)

            graph_props = GraphRAG._ensure_ontology_props(
                grag,
                {},
                entity="graph",
                label="GraphRAG Result",
                entity_id=str(uuid.uuid4()),
                generated_by=getattr(grag, "defaultGeneratedBy", "GraphRAG.Generate"),
            )
            graph_props.setdefault("label", "GraphRAG Result")
            graph_props.setdefault("graph_id", f"grag_{uuid.uuid4().hex[:12]}")
            gd = Dictionary.ByKeysValues(list(graph_props.keys()), list(graph_props.values()))
            graph = Topology.SetDictionary(graph, gd)
            return graph
        except Exception as e:
            if not silent:
                print(f"GraphRAG._matrix_materialise_graph - Error: {e}. Returning None.")
            return None

    @staticmethod
    def Generate(
        grag,
        graph,
        description: str = "",
        maxSteps: int = 10,
        patience: int = 4,
        automatic: bool = False,
        approvalFunction: Callable[[Dict[str, Any], str], str] = None,
        verbose: bool = True,
        silent: bool = False,
    ) -> Dict[str, Any]:
        """
        Iteratively grows or edits the input graph using corpus evidence and an LLM.

        This implementation edits a temporary Python adjacency-matrix state and
        materialises a TopologicPy graph only once, at the end of generation. This
        avoids allowing geometric coincidence or core graph mutation behaviour to
        block the intended topological edits.
        """
        effective_silent = GraphRAG._effective_silent(grag, silent)
        records = []
        stagnant = 0
        status = "completed"

        if grag is None:
            return {"ok": False, "status": "error", "message": "The input grag object is None.", "graph": graph, "steps": []}
        if graph is None:
            return {"ok": False, "status": "error", "message": "The input graph is None.", "graph": graph, "steps": []}

        state = GraphRAG._matrix_seed_state(grag, graph, silent=effective_silent)
        final_graph = graph

        allowed_actions = ("add_node", "connect", "remove_node", "delete_node", "remove_edge", "delete_edge", "stop", "done", "finish")

        for step in range(1, max(1, int(maxSteps or 1)) + 1):
            summary_before = GraphRAG._matrix_summary(state)
            evidence = GraphRAG.Evidence(grag, summary_before, silent=effective_silent)

            raw_action = GraphRAG.PickAction(grag, summary_before, evidence, description=description, silent=effective_silent)

            if not isinstance(raw_action, dict) or not raw_action:
                stagnant += 1
                records.append({
                    "step": step,
                    "ok": False,
                    "status": "ignored_bad_llm_response",
                    "action": raw_action,
                    "raw_action": raw_action,
                    "message": "LLM returned no usable action. Ignoring this response and continuing.",
                    "summary_before": summary_before,
                    "summary_after": summary_before,
                    "evidence": evidence,
                })
                if verbose and not effective_silent:
                    print(f"STEP: {step}")
                    print("→ Ignoring malformed or unusable LLM response.")
                if stagnant >= max(1, int(patience or 1)):
                    status = "patience_exhausted"
                    if verbose and not effective_silent:
                        print("→ Stopping because patience was exhausted.")
                    break
                continue

            action = GraphRAG.NormalizeAction(raw_action)
            action_name = str(action.get("action", "")).strip().lower()

            if action_name not in allowed_actions:
                stagnant += 1
                records.append({
                    "step": step,
                    "ok": False,
                    "status": "ignored_invalid_action",
                    "action": action,
                    "raw_action": raw_action,
                    "message": f"Invalid action '{action_name}'. Ignoring this response and continuing.",
                    "summary_before": summary_before,
                    "summary_after": summary_before,
                    "evidence": evidence,
                })
                if verbose and not effective_silent:
                    print(f"STEP: {step}")
                    print(f"→ Ignoring invalid action: {action_name}")
                if stagnant >= max(1, int(patience or 1)):
                    status = "patience_exhausted"
                    if verbose and not effective_silent:
                        print("→ Stopping because patience was exhausted.")
                    break
                continue

            if verbose and not effective_silent:
                print(f"STEP: {step}")
                print("json_action:", action.get("action") if isinstance(action, dict) else None)
                print("json_label:", action.get("label") if isinstance(action, dict) else None)
                print("json_a_label:", action.get("a_label") if isinstance(action, dict) else None)
                print("json_b_label:", action.get("b_label") if isinstance(action, dict) else None)
                print("json_attach_to_label:", action.get("attach_to_label") if isinstance(action, dict) else None)

            if action_name in ("stop", "done", "finish"):
                status = "stopped"
                records.append({
                    "step": step,
                    "ok": True,
                    "status": status,
                    "action": action,
                    "raw_action": raw_action,
                    "message": action.get("reason", "The LLM chose to stop."),
                    "summary_before": summary_before,
                    "summary_after": summary_before,
                    "evidence": evidence,
                })
                if verbose and not effective_silent:
                    print("→ The process stopped.")
                break

            decision = "accept" if automatic else GraphRAG.ApproveAction(action, approvalFunction=approvalFunction, silent=effective_silent)

            if decision == "stop":
                status = "stopped_by_user"
                records.append({
                    "step": step,
                    "ok": True,
                    "status": status,
                    "action": action,
                    "raw_action": raw_action,
                    "message": "Stopped by user.",
                    "summary_before": summary_before,
                    "summary_after": summary_before,
                    "evidence": evidence,
                })
                break

            if decision == "ignore":
                stagnant += 1
                records.append({
                    "step": step,
                    "ok": True,
                    "status": "ignored",
                    "action": action,
                    "raw_action": raw_action,
                    "message": "Action ignored.",
                    "summary_before": summary_before,
                    "summary_after": summary_before,
                    "evidence": evidence,
                })
                if stagnant >= max(1, int(patience or 1)):
                    status = "patience_exhausted"
                    if verbose and not effective_silent:
                        print("→ Stopping because patience was exhausted.")
                    break
                continue

            apply_result = GraphRAG._matrix_apply_action(grag, state, action, evidence=evidence, silent=effective_silent)
            if not isinstance(apply_result, dict):
                apply_result = {"ok": False, "state": state, "message": "GraphRAG._matrix_apply_action returned no usable result."}
            if isinstance(apply_result.get("state"), dict):
                state = apply_result.get("state")

            summary_after = GraphRAG._matrix_summary(state)
            changed = bool(apply_result.get("ok")) and (summary_after != summary_before)
            stagnant = 0 if changed else stagnant + 1

            records.append({
                "step": step,
                "ok": bool(apply_result.get("ok")),
                "status": "applied" if apply_result.get("ok") else "failed",
                "action": action,
                "raw_action": raw_action,
                "message": apply_result.get("message", ""),
                "summary_before": summary_before,
                "summary_after": summary_after,
                "evidence": evidence,
            })

            if verbose and not effective_silent:
                print("→", apply_result.get("message", ""))
                print("Latest Number of Nodes:", summary_after.get("num_nodes"))
                print("Latest Labels:", [n.get("label") for n in summary_after.get("nodes", [])])

            if stagnant >= max(1, int(patience or 1)):
                status = "patience_exhausted"
                if verbose and not effective_silent:
                    print("→ Stopping because patience was exhausted.")
                break
        else:
            status = "max_steps_reached"

        materialised = GraphRAG._matrix_materialise_graph(grag, state, silent=effective_silent)
        if materialised is not None:
            final_graph = materialised
        else:
            status = "materialisation_failed" if status == "completed" else status

        return {
            "ok": materialised is not None,
            "status": status,
            "graph": final_graph,
            "steps": records,
            "num_steps": len(records),
            "matrix_state": state,
        }

    @staticmethod
    def _effective_silent(grag, silent: bool = False) -> bool:
        return bool(silent or getattr(grag, "silent", False))
