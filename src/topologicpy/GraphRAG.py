# Copyright (C) 2026
# This file is part of TopologicPy.
#
# TopologicPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import json
import math
import random
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


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

    Responsibilities
    ----------------
    - Summarise the current working TopologicPy graph.
    - Query a graph database corpus for candidate labels and relationships.
    - Ask an LLM for a strict JSON graph-edit action.
    - Apply supported graph edits using the existing TopologicPy Graph API.

    Supported actions
    -----------------
    - add_node: add a vertex and optionally connect it to an existing vertex.
    - connect: connect two existing vertices.
    - stop: stop the generation loop.
    """



    @staticmethod
    def _can_add_connection(grag, graph, vertex, silent: bool = False) -> bool:
        """
        Returns True if the input vertex can accept one more connection based on
        the maximum degree observed in the graph database corpus.
        """
        try:
            from topologicpy.Graph import Graph
            from topologicpy.Topology import Topology
            from topologicpy.Dictionary import Dictionary
            from topologicpy.GraphDB import GraphDB

            vertex_label_key = getattr(grag, "vertexLabelKey", "label")
            graphdb = getattr(grag, "graphdb", None)

            if graphdb is None:
                return True

            d = Topology.Dictionary(vertex)
            label = Dictionary.ValueAtKey(d, vertex_label_key, None)

            if label is None:
                return True

            current_degree = len(Graph.AdjacentVertices(graph, vertex) or [])
            max_degree = GraphDB.MaxNeighborsForLabel(
                graphdb,
                label,
                silent=silent,
            )

            max_degree = GraphRAG._max_neighbor_value(max_degree)

            # If the label was not found in the corpus, do not block it.
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

            vertex_label_key = getattr(grag, "vertexLabelKey", "label")
            graphdb = getattr(grag, "graphdb", None)

            d = Topology.Dictionary(vertex)
            label = Dictionary.ValueAtKey(d, vertex_label_key, "Unknown")
            current_degree = len(Graph.AdjacentVertices(graph, vertex) or [])

            max_degree = None
            if graphdb is not None:
                max_degree = GraphDB.MaxNeighborsForLabel(graphdb, label, silent=True)
                max_degree = GraphRAG._max_neighbor_value(max_degree)

            return f"Degree limit reached for '{label}': current degree = {current_degree}, corpus max degree = {max_degree}."

        except Exception:
            return "Degree limit reached."

    @staticmethod
    def _unique_action_id(graph, suggested_id: str = None, prefix: str = "n", key: str = "id", silent: bool = False) -> str:
        """
        Returns a graph-unique ID, preserving the LLM-suggested ID if it is unused.

        If the suggested ID already exists, the method generates a new sequential ID
        such as n5, n6, n7, etc.
        """
        try:
            from topologicpy.Graph import Graph
            from topologicpy.Topology import Topology
            from topologicpy.Dictionary import Dictionary

            existing = set()
            for v in Graph.Vertices(graph) or []:
                d = Topology.Dictionary(v)
                value = Dictionary.ValueAtKey(d, key, None)
                if value is not None:
                    existing.add(str(value).strip())

            suggested_id = str(suggested_id or "").strip()
            if suggested_id and suggested_id not in existing:
                return suggested_id

            # Try sequential IDs first: n1, n2, n3...
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

        # Remove common markdown fences.
        if text.startswith("```"):
            text = text.replace("```json", "").replace("```", "").strip()

        try:
            return json.loads(text)
        except Exception:
            pass

        # Extract first JSON object.
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            candidate = text[start:end + 1]
            try:
                return json.loads(candidate)
            except Exception:
                return {}

        return {}
    # -------------------------------------------------------------------------
    # Constructors
    # -------------------------------------------------------------------------
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

        Parameters
        ----------
        graphdb : dict , optional
            The GraphDB object used for corpus retrieval.
        llm : _LLMConfig , optional
            The LLM object returned by LLM.ByParameters.
        promptContext: str , optional
            The desired prompt context string. Default is None.
        vertexIDKey : str , optional
            The vertex dictionary key that stores the vertex ID. Default is "id".
        vertexLabelKey : str , optional
            The vertex dictionary key that stores the vertex label. Default is "label".
        edgeLabelKey : str , optional
            The edge dictionary key that stores the edge label. Default is "label".
        defaultEdgeLabel : str , optional
            The default label for newly created edges. Default is "suggested".
        tolerance : float , optional
            The geometric tolerance used by Graph methods. Default is 0.0001.
        maxCandidates : int , optional
            The maximum number of candidate labels included in an LLM prompt. Default is 12.
        maxPairs : int , optional
            The maximum number of corpus label-pair records included in an LLM prompt. Default is 40.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        _GraphRAGConfig
            The GraphRAG configuration object.
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

    # -------------------------------------------------------------------------
    # Public workflow methods
    # -------------------------------------------------------------------------
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

        Parameters
        ----------
        grag : _GraphRAGConfig
            The GraphRAG object returned by GraphRAG.ByParameters.
        graph : topologic_core.Graph
            The working TopologicPy graph to edit.
        description : str , optional
            A natural-language description of the target graph. Default is "".
        maxSteps : int , optional
            The maximum number of generation steps. Default is 10.
        patience : int , optional
            The maximum number of consecutive non-changing steps before stopping. Default is 4.
        automatic : bool , optional
            If True, apply accepted LLM actions without interactive confirmation. Default is False.
        approvalFunction : callable , optional
            A callback of the form approvalFunction(action, message) -> "accept" | "ignore" | "stop".
            If None and automatic is False, terminal input is used.
        verbose : bool , optional
            If True, prints step diagnostics. Default is True.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        dict
            A result dictionary containing the final graph, step records, and status.
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

            decision = "accept" if automatic else GraphRAG.ApproveAction(action, approvalFunction=approvalFunction, silent=effective_silent)
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

        return {
            "ok": True,
            "status": status,
            "graph": graph,
            "steps": records,
            "num_steps": len(records),
        }

    @staticmethod
    def PickAction(
        grag,
        graphSummary,
        evidence=None,
        description: str = "",
        silent: bool = False,
    ):
        effective_silent = GraphRAG._effective_silent(grag, silent)
        try:
            from topologicpy.LLM import LLM

            llm = getattr(grag, "llm", None)
            if llm is None:
                if not effective_silent:
                    print("GraphRAG.PickAction - Error: No LLM object was supplied. Returning {}.")
                return {}

            prompt = GraphRAG.Prompt(
                grag,
                graphSummary,
                evidence=evidence,
                description=description,
                silent=effective_silent,
            )

            prompt += """

    Return ONLY one valid JSON object.
    Do not use markdown.
    Do not use ```json.
    Do not explain.

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

            raw = LLM.Prompt(
                llm,
                prompt,
                temperature=0,
                maxOutputTokens=4096,
                silent=effective_silent,
            )

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
    # @staticmethod
    # def PickAction(
    #     grag,
    #     graphSummary: Dict[str, Any],
    #     evidence: Dict[str, Any] = None,
    #     description: str = "",
    #     silent: bool = False,
    # ) -> Dict[str, Any]:
    #     """
    #     Asks the configured LLM to select the next graph-edit action.

    #     Parameters
    #     ----------
    #     grag : _GraphRAGConfig
    #         The GraphRAG object returned by GraphRAG.ByParameters.
    #     graphSummary : dict
    #         A dictionary returned by GraphRAG.SummarizeGraph.
    #     evidence : dict , optional
    #         A dictionary returned by GraphRAG.Evidence. Default is None.
    #     description : str , optional
    #         A natural-language description of the target graph. Default is "".
    #     silent : bool , optional
    #         If set to True, error and warning messages are suppressed. Default is False.

    #     Returns
    #     -------
    #     dict
    #         The selected action dictionary, or {} on failure.
    #     """
    #     effective_silent = GraphRAG._effective_silent(grag, silent)
    #     try:
    #         from topologicpy.LLM import LLM

    #         llm = getattr(grag, "llm", None)
    #         if llm is None:
    #             if not effective_silent:
    #                 print("GraphRAG.PickAction - Error: No LLM object was supplied. Returning {}.")
    #             return {}

    #         prompt = GraphRAG.Prompt(grag, graphSummary, evidence=evidence, description=description, silent=effective_silent)

    #         for attempt in range(2):
    #             action = LLM.JSON(
    #                 llm,
    #                 prompt + "\n\nReturn ONLY one valid JSON object. Do not use markdown. Do not explain.",
    #                 schema=GraphRAG.ActionSchema(),
    #                 temperature=0,
    #                 maxOutputTokens=512,
    #                 silent=effective_silent,
    #             )

    #             if isinstance(action, list):
    #                 action = action[0] if action else {}

    #             if isinstance(action, dict) and action.get("action"):
    #                 return GraphRAG.NormalizeAction(action)

    #         return {}
    #     except Exception as e:
    #         if not effective_silent:
    #             print(f"GraphRAG.PickAction - Error: {e}. Returning {{}}.")
    #         return {}

    @staticmethod
    def ApplyAction(grag, graph, action: Dict[str, Any], silent: bool = False) -> Dict[str, Any]:
        """
        Applies one action to the input graph.

        Parameters
        ----------
        grag : _GraphRAGConfig
            The GraphRAG object returned by GraphRAG.ByParameters.
        graph : topologic_core.Graph
            The graph to edit.
        action : dict
            The action dictionary. Supported actions are "add_node", "connect", and "stop".
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        dict
            A result dictionary containing ok, graph, and message keys.
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

        Parameters
        ----------
        grag : _GraphRAGConfig
            The GraphRAG object returned by GraphRAG.ByParameters.
        graphSummary : dict
            A dictionary returned by GraphRAG.SummarizeGraph.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        dict
            Corpus evidence for prompt construction.
        """
        effective_silent = GraphRAG._effective_silent(grag, silent)
        labels = [n.get("label") for n in (graphSummary or {}).get("nodes", []) if n.get("label")]
        labels = GraphRAG._unique(labels)
        evidence = {
            "candidate_counts": [],
            "max_neighbors": {},
            "pairs": [],
            "expandable_nodes": [],
        }

        graphdb = getattr(grag, "graphdb", None)
        if graphdb is None:
            return evidence

        try:
            from topologicpy.GraphDB import GraphDB
        except Exception:
            try:
                from GraphDB import GraphDB
            except Exception as e:
                if not effective_silent:
                    print(f"GraphRAG.Evidence - Error importing GraphDB: {e}.")
                return evidence

        # Candidate labels and counts.
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

        # Max-neighbour statistics for currently present labels.
        for label in labels:
            try:
                evidence["max_neighbors"][label] = GraphDB.MaxNeighborsForLabel(
                    graphdb,
                    label,
                    silent=effective_silent,
                )
            except Exception:
                evidence["max_neighbors"][label] = None

        # Pair frequencies. GraphDB.FetchAllPairs returns backend-wide pair
        # frequencies. We filter and truncate locally so backends can keep a
        # simple common signature.
        candidate_labels = GraphRAG._candidate_labels(evidence.get("candidate_counts", []))
        all_labels = GraphRAG._unique(labels + candidate_labels)[: max(1, getattr(grag, "maxCandidates", 12) * 2)]
        try:
            pairs = GraphDB.FetchAllPairs(
                graphdb,
                undirected=True,
                silent=effective_silent,
            ) or []
            evidence["pairs"] = GraphRAG._filter_pairs_by_labels(
                pairs,
                all_labels,
                limit=getattr(grag, "maxPairs", 40),
            )
        except Exception as e:
            evidence["pairs"] = []
            if not effective_silent:
                print(f"GraphRAG.Evidence - Warning: FetchAllPairs failed: {e}")

        # Expandable node calculation.
        evidence["expandable_nodes"] = GraphRAG.ExpandableNodes(graphSummary, evidence, silent=effective_silent)
        return evidence

    @staticmethod
    def ExpandableNodes(graphSummary: Dict[str, Any], evidence: Dict[str, Any] = None, silent: bool = False) -> List[Dict[str, Any]]:
        """
        Returns nodes whose current degree is lower than the maximum found in the corpus.

        Parameters
        ----------
        graphSummary : dict
            A dictionary returned by GraphRAG.SummarizeGraph.
        evidence : dict , optional
            A dictionary returned by GraphRAG.Evidence. Default is None.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            A list of expandable node dictionaries.
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

        Parameters
        ----------
        grag : _GraphRAGConfig
            The GraphRAG object returned by GraphRAG.ByParameters.
        graph : topologic_core.Graph
            The graph to summarise.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        dict
            A graph summary dictionary.
        """
        from topologicpy.Vertex import Vertex

        effective_silent = GraphRAG._effective_silent(grag, silent)
        try:
            from topologicpy.Graph import Graph
            from topologicpy.Topology import Topology
            from topologicpy.Dictionary import Dictionary

            vertex_id_key = getattr(grag, "vertexIDKey", "id")
            vertex_label_key = getattr(grag, "vertexLabelKey", "label")
            edge_label_key = getattr(grag, "edgeLabelKey", "label")

            vertices = Graph.Vertices(graph) or []
            edges = Graph.Edges(graph) or []

            vertex_to_id = {}
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
                vertex_to_id[GraphRAG._topology_key(v)] = str(vid)
                nodes.append({
                    "id": str(vid),
                    "label": str(label),
                    "degree": int(degree),
                    "x": x,
                    "y": y,
                    "z": z,
                    "props": props,
                })

            out_edges = []
            for i, e in enumerate(edges):
                try:
                    sv = GraphRAG._start_vertex(e)
                    ev = GraphRAG._end_vertex(e)
                    src = None
                    dst = None
                    try:
                        si = Vertex.Index(vertex=sv, vertices=vertices, strict=False, tolerance=getattr(grag, "tolerance", 0.0001))
                        if si is not None and si >= 0 and si < len(nodes):
                            src = nodes[si]["id"]
                    except Exception:
                        pass
                    try:
                        ei = Vertex.Index(vertex=ev, vertices=vertices, strict=False, tolerance=getattr(grag, "tolerance", 0.0001))
                        if ei is not None and ei >= 0 and ei < len(nodes):
                            dst = nodes[ei]["id"]
                    except Exception:
                        pass
                except Exception:
                    src = None
                    dst = None
                d = Topology.Dictionary(e)
                props = GraphRAG._python_dictionary(d)
                label = Dictionary.ValueAtKey(d, edge_label_key, props.get(edge_label_key, "connect"))
                out_edges.append({
                    "src": src,
                    "dst": dst,
                    "label": str(label),
                    "props": props,
                })

            return {
                "nodes": nodes,
                "edges": out_edges,
                "num_nodes": len(nodes),
                "num_edges": len(out_edges),
            }
        except Exception as e:
            if not effective_silent:
                print(f"GraphRAG.SummarizeGraph - Error: {e}. Returning empty summary.")
            return {"nodes": [], "edges": [], "num_nodes": 0, "num_edges": 0}

    @staticmethod
    def Prompt(
        grag,
        graphSummary: Dict[str, Any],
        evidence: Dict[str, Any] = None,
        description: str = "",
        silent: bool = False,
    ) -> str:
        """
        Builds the prompt used for graph-edit action selection.

        Parameters
        ----------
        grag : _GraphRAGConfig
            The GraphRAG object returned by GraphRAG.ByParameters.
        graphSummary : dict
            A dictionary returned by GraphRAG.SummarizeGraph.
        evidence : dict , optional
            A dictionary returned by GraphRAG.Evidence. Default is None.
        description : str , optional
            A natural-language description of the target graph. Default is "".
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        str
            The prompt string.
        """
        evidence = evidence or {}

        payload = {
            "target_description": description or "",
            "current_graph": {
                "nodes": [
                    {
                        "id": n["id"],
                        "label": n["label"],
                        "degree": n["degree"],
                    }
                    for n in (graphSummary or {}).get("nodes", [])
                ],
                "edges": [
                    {
                        "src": e["src"],
                        "dst": e["dst"],
                        "label": e["label"],
                    }
                    for e in (graphSummary or {}).get("edges", [])
                ],
            },
            "corpus_evidence": {
                "candidate_counts": (evidence.get("candidate_counts") or [])[
                    : getattr(grag, "maxCandidates", 12)
                ],
                "max_neighbors": evidence.get("max_neighbors") or {},
                "expandable_nodes": evidence.get("expandable_nodes") or [],
                "pairs": (evidence.get("pairs") or [])[
                    : getattr(grag, "maxPairs", 40)
                ],
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
                "Do not connect a node if its current degree is already equal to or greater than the corpus maximum degree for that label."
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

        Returns
        -------
        dict
            The action schema.
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

        Parameters
        ----------
        action : dict
            The action dictionary to normalize.

        Returns
        -------
        dict
            The normalized action dictionary.
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

        Parameters
        ----------
        action : dict
            The proposed action dictionary.
        approvalFunction : callable , optional
            A callback of the form approvalFunction(action, message) -> "accept" | "ignore" | "stop".
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        str
            "accept", "ignore", or "stop".
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

    # -------------------------------------------------------------------------
    # Local workflow helpers retained outside Graph API by design
    # -------------------------------------------------------------------------
    @staticmethod
    def unique_id(graph=None, prefix: str = "n", key: str = "id", silent: bool = False) -> str:
        """
        Returns a unique vertex ID for the input graph.

        Parameters
        ----------
        graph : topologic_core.Graph , optional
            The graph to inspect for existing IDs. Default is None.
        prefix : str , optional
            The ID prefix. Default is "n".
        key : str , optional
            The dictionary key under which IDs are stored. Default is "id".
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        str
            A unique ID string.
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

        This helper intentionally remains part of the GraphRAG workflow rather
        than being added to the Graph API.

        Parameters
        ----------
        grag : _GraphRAGConfig
            The GraphRAG object returned by GraphRAG.ByParameters.
        graph : topologic_core.Graph
            The graph to inspect.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        dict
            A dictionary with nodes and edges keys.
        """
        summary = GraphRAG.SummarizeGraph(grag, graph, silent=silent)
        return {"nodes": summary.get("nodes", []), "edges": summary.get("edges", [])}

    # -------------------------------------------------------------------------
    # Private action implementations
    # -------------------------------------------------------------------------
    @staticmethod
    def _apply_add_node(grag, graph, action: Dict[str, Any], silent: bool = False) -> Dict[str, Any]:
        try:
            from topologicpy.Graph import Graph
            from topologicpy.Vertex import Vertex
            from topologicpy.Edge import Edge
            from topologicpy.Topology import Topology
            from topologicpy.Dictionary import Dictionary

            vertex_id_key = getattr(grag, "vertexIDKey", "id")
            vertex_label_key = getattr(grag, "vertexLabelKey", "label")
            edge_label_key = getattr(grag, "edgeLabelKey", "label")
            tolerance = getattr(grag, "tolerance", 0.0001)

            label = str(action.get("label") or action.get("b_label") or "Node").strip()
            # vid = str(action.get("id") or GraphRAG.unique_id(graph, prefix="n", key=vertex_id_key)).strip()
            suggested_id = str(action.get("id") or "").strip()
            vid = GraphRAG._unique_action_id(
                graph,
                suggested_id=suggested_id,
                prefix="n",
                key=vertex_id_key,
                silent=silent,
            )

            attach_vertex = GraphRAG._vertex_from_action(grag, graph, action, prefix="attach_to")
            x, y, z = GraphRAG._coordinates_for_new_node(grag, graph, action, attach_vertex, label)

            v = Vertex.ByCoordinates(x, y, z)
            props = {
                vertex_id_key: vid,
                vertex_label_key: label,
                "generated_by": "GraphRAG",
            }
            if action.get("reason"):
                props["reason"] = str(action.get("reason"))
            d = Dictionary.ByKeysValues(list(props.keys()), list(props.values()))
            v = Topology.SetDictionary(v, d)

            g2 = Graph.AddVertex(graph, v, tolerance=tolerance, silent=silent)
            if g2 is not None:
                graph = g2

            if attach_vertex is not None:
                existing_edge = Graph.Edge(graph, attach_vertex, v, tolerance=tolerance, silent=True)
                if existing_edge is None:
                    edge = Edge.ByStartVertexEndVertex(attach_vertex, v)
                    edge_label = str(action.get("edge_label") or getattr(grag, "defaultEdgeLabel", "suggested"))
                    ed = Dictionary.ByKeysValues([edge_label_key, "generated_by"], [edge_label, "GraphRAG"])
                    edge = Topology.SetDictionary(edge, ed)
                    g3 = Graph.AddEdge(
                        graph,
                        edge,
                        transferVertexDictionaries=False,
                        transferEdgeDictionaries=False,
                        tolerance=tolerance,
                        silent=silent,
                    )
                    if g3 is not None:
                        graph = g3
                return {"ok": True, "graph": graph, "message": f"Added node '{label}' and connected it."}
            return {"ok": True, "graph": graph, "message": f"Added node '{label}'."}
        except Exception as e:
            if not silent:
                print(f"GraphRAG._apply_add_node - Error: {e}. Returning unchanged graph.")
            return {"ok": False, "graph": graph, "message": f"Could not add node: {e}"}

    @staticmethod
    def _apply_connect(grag, graph, action: Dict[str, Any], silent: bool = False) -> Dict[str, Any]:
        try:
            from topologicpy.Graph import Graph
            from topologicpy.Edge import Edge
            from topologicpy.Topology import Topology
            from topologicpy.Dictionary import Dictionary


            edge_label_key = getattr(grag, "edgeLabelKey", "label")
            tolerance = getattr(grag, "tolerance", 0.0001)

            va = GraphRAG._vertex_from_action(grag, graph, action, prefix="a")
            vb = GraphRAG._vertex_from_action(grag, graph, action, prefix="b")

            if va is None or vb is None:
                return {"ok": False, "graph": graph, "message": "Could not find both vertices to connect."}
            if va == vb:
                return {"ok": False, "graph": graph, "message": "Cannot connect a vertex to itself."}
            
            existing = Graph.Edge(graph, va, vb, tolerance=tolerance, silent=True)
            if existing is not None:
                return {"ok": True, "graph": graph, "message": "The requested edge already exists."}
            
            if not GraphRAG._can_add_connection(grag, graph, va, silent=silent):
                return {
                    "ok": False,
                    "graph": graph,
                    "message": GraphRAG._degree_limit_message(grag, graph, va),
                }

            if not GraphRAG._can_add_connection(grag, graph, vb, silent=silent):
                return {
                    "ok": False,
                    "graph": graph,
                    "message": GraphRAG._degree_limit_message(grag, graph, vb),
                }

            edge = Edge.ByStartVertexEndVertex(va, vb)
            edge_label = str(action.get("edge_label") or getattr(grag, "defaultEdgeLabel", "suggested"))
            ed = Dictionary.ByKeysValues([edge_label_key, "generated_by"], [edge_label, "GraphRAG"])
            edge = Topology.SetDictionary(edge, ed)
            g2 = Graph.AddEdge(
                graph,
                edge,
                transferVertexDictionaries=False,
                transferEdgeDictionaries=False,
                tolerance=tolerance,
                silent=silent,
            )
            if g2 is not None:
                graph = g2
            return {"ok": True, "graph": graph, "message": "Connected the requested vertices."}
        except Exception as e:
            if not silent:
                print(f"GraphRAG._apply_connect - Error: {e}. Returning unchanged graph.")
            return {"ok": False, "graph": graph, "message": f"Could not connect vertices: {e}"}

    # -------------------------------------------------------------------------
    # Private utility methods
    # -------------------------------------------------------------------------
    @staticmethod
    def _vertex_from_action(grag, graph, action: Dict[str, Any], prefix: str = "a"):
        try:
            from topologicpy.Graph import Graph
            vertex_id_key = getattr(grag, "vertexIDKey", "id")
            vertex_label_key = getattr(grag, "vertexLabelKey", "label")
            vid = action.get(f"{prefix}_id")
            label = action.get(f"{prefix}_label")
            if vid is not None and str(vid).strip() != "":
                v = Graph.VertexByKeyValue(graph, vertex_id_key, str(vid), silent=True)
                if v is not None:
                    return v
            if label is not None and str(label).strip() != "":
                return Graph.VertexByKeyValue(graph, vertex_label_key, str(label), silent=True)
            return None
        except Exception:
            return None

    @staticmethod
    def _coordinates_for_new_node(grag, graph, action: Dict[str, Any], attach_vertex=None, label: str = None) -> Tuple[float, float, float]:
        def _num(value, default=None):
            try:
                if value is None:
                    return default
                value = float(value)
                if math.isnan(value) or math.isinf(value):
                    return default
                return value
            except Exception:
                return default

        x = _num(action.get("x"))
        y = _num(action.get("y"))
        z = _num(action.get("z"))
        if x is not None and y is not None and z is not None:
            return x, y, z

        # Try GraphDB example coordinates if the method exists.
        try:
            try:
                from topologicpy.GraphDB import GraphDB
            except Exception:
                from GraphDB import GraphDB
            graphdb = getattr(grag, "graphdb", None)
            if graphdb is not None and label:
                attach_label = action.get("attach_to_label") or action.get("a_label")
                example = GraphDB.FindBestExampleForLabel(graphdb, label, attachTo=attach_label, silent=True)
                if isinstance(example, dict):
                    ex = _num(example.get("x"), None)
                    ey = _num(example.get("y"), None)
                    ez = _num(example.get("z"), None)
                    if ex is not None and ey is not None and ez is not None:
                        return ex, ey, ez
        except Exception:
            pass

        if attach_vertex is not None:
            ax, ay, az = GraphRAG._vertex_xyz(attach_vertex)
            return ax + random.uniform(1.0, 3.0), ay + random.uniform(1.0, 3.0), az

        # Fallback: use a point just outside the current bounding box if available.
        try:
            from topologicpy.Graph import Graph
            vertices = Graph.Vertices(graph) or []
            coords = [GraphRAG._vertex_xyz(v) for v in vertices]
            if coords:
                max_x = max(c[0] for c in coords)
                avg_y = sum(c[1] for c in coords) / len(coords)
                avg_z = sum(c[2] for c in coords) / len(coords)
                return max_x + 2.0, avg_y, avg_z
        except Exception:
            pass
        return random.uniform(0, 100), random.uniform(0, 100), 0.0

    @staticmethod
    def _filter_pairs_by_labels(pairs: List[Any], labels: List[str], limit: int = 40) -> List[Dict[str, Any]]:
        """
        Filters backend pair-frequency records by a set of labels.

        Backends may return pair dictionaries with different but common key names,
        such as a_label/b_label, label_a/label_b, src_label/dst_label, or
        source/target. This helper normalizes those records for prompting.
        """
        label_set = set([str(label) for label in labels or [] if label is not None])
        normalized = []
        for row in pairs or []:
            if not isinstance(row, dict):
                continue

            a = (
                row.get("a_label")
                or row.get("label_a")
                or row.get("src_label")
                or row.get("source_label")
                or row.get("source")
                or row.get("a")
            )
            b = (
                row.get("b_label")
                or row.get("label_b")
                or row.get("dst_label")
                or row.get("target_label")
                or row.get("target")
                or row.get("b")
            )

            if a is None or b is None:
                continue
            a = str(a)
            b = str(b)
            if label_set and a not in label_set and b not in label_set:
                continue

            count = (
                row.get("count")
                if row.get("count") is not None
                else row.get("frequency")
                if row.get("frequency") is not None
                else row.get("n")
            )
            try:
                count = int(count)
            except Exception:
                count = 1

            out = dict(row)
            out["a_label"] = a
            out["b_label"] = b
            out["count"] = count
            normalized.append(out)

        normalized.sort(key=lambda r: int(r.get("count", 0) or 0), reverse=True)
        return normalized[: max(1, int(limit or 40))]

    @staticmethod
    def _candidate_labels(candidate_counts: List[Any]) -> List[str]:
        labels = []
        for row in candidate_counts or []:
            if isinstance(row, dict):
                for k in ("label", "candidate", "candidate_label", "neighbor_label", "b_label"):
                    if row.get(k):
                        labels.append(str(row.get(k)))
                        break
            elif row is not None:
                labels.append(str(row))
        return GraphRAG._unique(labels)

    @staticmethod
    def _max_neighbor_value(value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, dict):
            for k in ("max_neighbors", "max_degree", "degree", "count", "value"):
                if k in value:
                    return GraphRAG._max_neighbor_value(value.get(k))
            return None
        if isinstance(value, list):
            if not value:
                return None
            return GraphRAG._max_neighbor_value(value[0])
        try:
            return int(value)
        except Exception:
            return None

    @staticmethod
    def _vertex_xyz(vertex) -> Tuple[float, float, float]:
        def _call(cls_name, method_name):
            try:
                module = __import__(f"topologicpy.{cls_name}", fromlist=[cls_name])
                cls = getattr(module, cls_name)
                return getattr(cls, method_name)(vertex)
            except Exception:
                return None
        x = _call("Vertex", "X")
        y = _call("Vertex", "Y")
        z = _call("Vertex", "Z")
        if x is not None and y is not None and z is not None:
            return float(x), float(y), float(z)
        try:
            return float(vertex.X()), float(vertex.Y()), float(vertex.Z())
        except Exception:
            return 0.0, 0.0, 0.0

    @staticmethod
    def _start_vertex(edge):
        try:
            from topologicpy.Edge import Edge
            return Edge.StartVertex(edge)
        except Exception:
            try:
                return edge.StartVertex()
            except Exception:
                return None

    @staticmethod
    def _end_vertex(edge):
        try:
            from topologicpy.Edge import Edge
            return Edge.EndVertex(edge)
        except Exception:
            try:
                return edge.EndVertex()
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
