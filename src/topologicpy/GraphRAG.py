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
$1
            except Exception:
                return None

    @staticmethod
    def _end_vertex(edge):
        try:
            from topologicpy.Edge import Edge
            return Edge.EndVertex(edge)
        except Exception:
            try:
                return edge.EndVertex() # Hook to Core
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
