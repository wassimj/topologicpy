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

        def _resolve_action_labels(grag, action, silent=False):
            """
            Resolves all plausible label-bearing fields in an LLM action before
            approval and before ApplyAction.

            This prevents raw LLM labels such as 'Living' from reaching the graph
            database when the corpus uses canonical labels such as 'living_room'.
            """

            if not isinstance(action, dict):
                return action

            resolved_action = dict(action)

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

            if verbose and not effective_silent:
                print(f"STEP: {step}")
                print("json_action:", action.get("action") if isinstance(action, dict) else None)
                print("json_label:", action.get("label") if isinstance(action, dict) else None)
                print("json_a_label:", action.get("a_label") if isinstance(action, dict) else None)
                print("json_b_label:", action.get("b_label") if isinstance(action, dict) else None)
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
                nodes.append({"id": str(vid), "label": str(label), "degree": int(degree), "x": x, "y": y, "z": z, "props": props})

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
                out_edges.append({"src": src, "dst": dst, "label": str(label), "props": props})

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
                "nodes": [{"id": n["id"], "label": n["label"], "degree": n["degree"]} for n in (graphSummary or {}).get("nodes", [])],
                "edges": [{"src": e["src"], "dst": e["dst"], "label": e["label"]} for e in (graphSummary or {}).get("edges", [])],
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
        Normalizes common action aliases.
        """
        if not isinstance(action, dict):
            return {}
        out = dict(action)
        name = str(out.get("action", "stop")).strip().lower()
        aliases = {
            "add": "add_node",
            "add_vertex": "add_node",
            "node": "add_node",
            "connect_nodes": "connect",
            "add_edge": "connect",
            "edge": "connect",
            "finish": "stop",
            "done": "stop",
        }
        out["action"] = aliases.get(name, name)
        if "label" not in out and "b_label" in out and out["action"] == "add_node":
            out["label"] = out.get("b_label")
        if "attach_to_id" not in out and "a_id" in out and out["action"] == "add_node":
            out["attach_to_id"] = out.get("a_id")
        if "attach_to_label" not in out and "a_label" in out and out["action"] == "add_node":
            out["attach_to_label"] = out.get("a_label")
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
            d = Dictionary.ByKeysValues([vertex_id_key, vertex_label_key], [vid, label])
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
                    ed = Dictionary.ByKeysValues([edge_label_key], [action.get("edge_label") or getattr(grag, "defaultEdgeLabel", "suggested")])
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
            ed = Dictionary.ByKeysValues([edge_label_key], [action.get("edge_label") or getattr(grag, "defaultEdgeLabel", "suggested")])
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

    @staticmethod
    def _effective_silent(grag, silent: bool = False) -> bool:
        return bool(silent or getattr(grag, "silent", False))
