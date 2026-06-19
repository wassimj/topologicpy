# Copyright (C) 2026
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

"""
Convenience wrapper for the TopologicPy GQL pipeline.

This module lives at ``topologicpy/GQL.py`` so it can be imported like the
other TopologicPy classes::

    from topologicpy.GQL import GQL

The parser and executor implementation live in the lowercase internal package
``topologicpy/gql``.

Read queries return ``list[dict]``. Mutation queries return a dictionary
containing at least the updated graph or GQL working graph and projected rows.

Ontology support
----------------
This wrapper preserves TopologicPy ontology metadata during GQL execution and
conversion. It does not make ontology mandatory inside the internal parser or
executor. Instead, it provides a lightweight compatibility layer:

- ``ontology=True`` annotates TopologicPy TGraph results where possible.
- ``top:Class`` query labels can be normalised to parser-safe labels when
  ``normalizeOntologyLabels=True``.
- Query results can be enriched with ontology metadata copied from returned
  graph elements or working-graph dictionaries.
- Returned TopologicPy TGraphs can be annotated with ``top:Graph``.
"""


class GQL:
    """Convenience API for parsing and executing GQL-like queries."""

    # ---------------------------------------------------------------------
    # Ontology defaults
    # ---------------------------------------------------------------------

    ONTOLOGY_PREFIX = "top"
    ONTOLOGY_NAMESPACE = "http://w3id.org/topologicpy#"

    ONTOLOGY_CLASS_KEY = "ontology_class"
    ONTOLOGY_URI_KEY = "uri"
    CATEGORY_KEY = "category"
    LABEL_KEY = "label"
    GENERATED_BY_KEY = "generated_by"

    DEFAULT_GRAPH_ONTOLOGY_CLASS = "top:Graph"
    DEFAULT_NODE_ONTOLOGY_CLASS = "top:Node"
    DEFAULT_EDGE_ONTOLOGY_CLASS = "top:Relationship"

    # ---------------------------------------------------------------------
    # Internal ontology helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _normalise_top_class(value):
        """
        Returns a canonical top:Class value if possible.

        Parameters
        ----------
        value : any
            The input ontology class value.

        Returns
        -------
        str
            The normalised ontology class string, or None.
        """
        if value is None:
            return None
        value = str(value).strip()
        if value == "":
            return None
        if value.startswith("top:"):
            return value
        if value.startswith(GQL.ONTOLOGY_NAMESPACE):
            return "top:" + value[len(GQL.ONTOLOGY_NAMESPACE):]
        if ":" not in value and value[:1].isupper():
            return "top:" + value
        return value

    @staticmethod
    def _ontology_local_name(value):
        """
        Returns the local name of an ontology class.

        Examples
        --------
        ``top:Room`` -> ``Room``
        ``http://w3id.org/topologicpy#Room`` -> ``Room``
        """
        value = GQL._normalise_top_class(value)
        if value is None:
            return None
        if value.startswith("top:"):
            return value.split(":", 1)[1]
        if "#" in value:
            return value.rsplit("#", 1)[-1]
        if "/" in value:
            return value.rstrip("/").rsplit("/", 1)[-1]
        return value

    @staticmethod
    def _safe_dictionary(topology):
        try:
            from topologicpy.TGraph import TGraph
            if isinstance(topology, TGraph):
                return TGraph.Dictionary(topology)
        except Exception:
            pass
        if isinstance(topology, dict):
            return topology.get("dictionary", topology)
        try:
            from topologicpy.Topology import Topology
            return Topology.Dictionary(topology)
        except Exception:
            return None

    @staticmethod
    def _python_dictionary(dictionary):
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
    def _dictionary_value(topology_or_dict, key, default=None):
        if topology_or_dict is None or key is None:
            return default
        if isinstance(topology_or_dict, dict):
            if key in topology_or_dict:
                return topology_or_dict.get(key, default)
            d = topology_or_dict.get("dictionary", None)
            if isinstance(d, dict):
                return d.get(key, default)
            return default
        try:
            from topologicpy.Topology import Topology
            from topologicpy.Dictionary import Dictionary
            d = Topology.Dictionary(topology_or_dict)
            return Dictionary.ValueAtKey(d, key, default)
        except Exception:
            try:
                return topology_or_dict.get(key, default)
            except Exception:
                return default

    @staticmethod
    def _set_dictionary_values(topology, keys, values, silent: bool = True):
        """
        Sets dictionary values on a TopologicPy topology using existing helpers.

        This method is deliberately defensive because GQL may operate on internal
        working graph dictionaries as well as TopologicPy objects.
        """
        if topology is None:
            return topology
        try:
            from topologicpy.Topology import Topology
            from topologicpy.Dictionary import Dictionary

            d = Topology.Dictionary(topology)
            d = Dictionary.SetValuesAtKeys(d, keys, values)
            return Topology.SetDictionary(topology, d, silent=silent)
        except TypeError:
            try:
                from topologicpy.Topology import Topology
                from topologicpy.Dictionary import Dictionary

                d = Topology.Dictionary(topology)
                d = Dictionary.SetValuesAtKeys(d, keys, values)
                return Topology.SetDictionary(topology, d)
            except Exception:
                return topology
        except Exception:
            return topology

    @staticmethod
    def _annotate_tgraph(graph,
                         ontologyClass: str = None,
                         generatedBy: str = "GQL",
                         silent: bool = True):
        """
        Adds lightweight ontology metadata to a TGraph and its vertices/edges
        where missing.

        This method preserves existing ontology metadata and only fills missing
        keys. It does not change graph incidence.

        Parameters
        ----------
        graph : topologicpy.TGraph
            The input TGraph.
        ontologyClass : str , optional
            The ontology class assigned to the graph when missing. Default is None.
        generatedBy : str , optional
            The value assigned to ``generated_by`` when missing. Default is ``"GQL"``.
        silent : bool , optional
            If True, warning messages are suppressed. Default is True.

        Returns
        -------
        topologicpy.TGraph
            The annotated TGraph.
        """
        try:
            from topologicpy.TGraph import TGraph

            if not isinstance(graph, TGraph):
                return graph

            graph_class = GQL._normalise_top_class(ontologyClass) or GQL.DEFAULT_GRAPH_ONTOLOGY_CLASS

            def _set_missing_dict(d, pairs):
                d = dict(d) if isinstance(d, dict) else {}
                changed = False
                for k, v in pairs:
                    if v is None:
                        continue
                    current = d.get(k, None)
                    if current is None or str(current).strip() == "":
                        d[k] = v
                        changed = True
                return d, changed

            graph_dict, changed = _set_missing_dict(
                TGraph.Dictionary(graph),
                [
                    (GQL.ONTOLOGY_CLASS_KEY, graph_class),
                    (GQL.CATEGORY_KEY, "graph"),
                    (GQL.GENERATED_BY_KEY, generatedBy),
                ],
            )
            if changed:
                graph.SetDictionary(graph_dict)

            for v in TGraph.Vertices(graph, asTopologic=False, activeOnly=True) or []:
                idx = v.get("index", None)
                d = dict(v.get("dictionary", {}))
                label = d.get(GQL.LABEL_KEY, None)
                node_class = d.get(GQL.ONTOLOGY_CLASS_KEY, None)
                if node_class is None or str(node_class).strip() == "":
                    if isinstance(label, str) and label.startswith("top:"):
                        node_class = label
                    else:
                        node_class = GQL.DEFAULT_NODE_ONTOLOGY_CLASS
                d, changed = _set_missing_dict(
                    d,
                    [
                        (GQL.ONTOLOGY_CLASS_KEY, GQL._normalise_top_class(node_class)),
                        (GQL.CATEGORY_KEY, "node"),
                    ],
                )
                if changed and isinstance(idx, int):
                    graph.SetVertexDictionary(idx, d)

            for e in TGraph.Edges(graph, asTopologic=False, activeOnly=True) or []:
                idx = e.get("index", None)
                d = dict(e.get("dictionary", {}))
                d, changed = _set_missing_dict(
                    d,
                    [
                        (GQL.ONTOLOGY_CLASS_KEY, GQL.DEFAULT_EDGE_ONTOLOGY_CLASS),
                        (GQL.CATEGORY_KEY, "relationship"),
                    ],
                )
                if changed and isinstance(idx, int):
                    graph.SetEdgeDictionary(idx, d)

            return graph
        except Exception as e:
            if not silent:
                print(f"GQL._annotate_tgraph - Warning: {e}")
            return graph

    @staticmethod
    def _to_tgraph(graph,
                   ontology: bool = True,
                   ontologyClass: str = DEFAULT_GRAPH_ONTOLOGY_CLASS,
                   generatedBy: str = "GQL.TGraph",
                   silent: bool = True):
        """
        Converts a TGraph-like object or GQL working graph dictionary to TGraph.

        Parameters
        ----------
        graph : topologicpy.TGraph or dict
            The input TGraph or working graph dictionary.
        ontology : bool , optional
            If True, the returned TGraph is annotated where possible. Default is True.
        ontologyClass : str , optional
            The ontology class assigned to the TGraph when missing. Default is ``"top:Graph"``.
        generatedBy : str , optional
            The value assigned to ``generated_by`` when missing. Default is ``"GQL.TGraph"``.
        silent : bool , optional
            If True, warning messages are suppressed. Default is True.

        Returns
        -------
        topologicpy.TGraph or None
            The converted TGraph, or None if conversion fails.
        """
        try:
            from topologicpy.TGraph import TGraph
        except Exception as e:
            if not silent:
                print(f"GQL._to_tgraph - Error: Could not import TGraph. {e}")
            return None

        if isinstance(graph, TGraph):
            return GQL._annotate_tgraph(graph, ontologyClass=ontologyClass, generatedBy=generatedBy, silent=silent) if ontology else graph

        if not isinstance(graph, dict):
            if not silent:
                print("GQL._to_tgraph - Error: The input graph is not a TGraph or dictionary. Returning None.")
            return None

        if isinstance(graph.get("graph"), TGraph):
            return GQL._to_tgraph(graph.get("graph"), ontology=ontology, ontologyClass=ontologyClass, generatedBy=generatedBy, silent=silent)

        # Native TGraph JSON/Python format.
        if graph.get("type") == "TGraph" or ("vertices" in graph and "edges" in graph and "directed" in graph):
            try:
                result = TGraph.FromPython(graph, ontology=ontology)
                if ontology:
                    result = GQL._annotate_tgraph(result, ontologyClass=ontologyClass, generatedBy=generatedBy, silent=silent)
                return result
            except Exception:
                pass

        # Common internal GQL working-graph layouts.
        nodes = graph.get("nodes", None)
        if nodes is None:
            nodes = graph.get("vertices", None)
        relationships = graph.get("relationships", None)
        if relationships is None:
            relationships = graph.get("edges", None)

        if not isinstance(nodes, list):
            nodes = []
        if not isinstance(relationships, list):
            relationships = []

        vertices = []
        id_to_index = {}

        for i, node in enumerate(nodes):
            if isinstance(node, dict):
                d = dict(node.get("dictionary", node))
                d.pop("representation", None)
                node_id = d.get("id", d.get("ID", d.get("uuid", d.get("name", d.get("label", i)))))
            else:
                d = {"label": str(node)}
                node_id = i
            d.setdefault("id", node_id)
            vertices.append(d)
            id_to_index[node_id] = i
            for alias in ("index", "ID", "uuid", "guid", "name", "label"):
                if isinstance(d, dict) and alias in d and d[alias] not in id_to_index:
                    id_to_index[d[alias]] = i

        edges = []
        for rel in relationships:
            if not isinstance(rel, dict):
                continue
            d = dict(rel.get("dictionary", rel))
            src = d.get("src", d.get("source", d.get("start", d.get("from", None))))
            dst = d.get("dst", d.get("target", d.get("end", d.get("to", None))))
            if src in id_to_index:
                src = id_to_index[src]
            if dst in id_to_index:
                dst = id_to_index[dst]
            if not isinstance(src, int) or not isinstance(dst, int):
                continue
            d["src"] = src
            d["dst"] = dst
            edges.append(d)

        try:
            result = TGraph.ByVerticesEdges(
                vertices=vertices,
                edges=edges,
                directed=bool(graph.get("directed", True)),
                allowSelfLoops=True,
                allowParallelEdges=True,
                dictionary=dict(graph.get("dictionary", {})) if isinstance(graph.get("dictionary", {}), dict) else {},
                ontology=ontology,
                silent=silent,
            )
            if ontology:
                result = GQL._annotate_tgraph(result, ontologyClass=ontologyClass, generatedBy=generatedBy, silent=silent)
            return result
        except Exception as e:
            if not silent:
                print(f"GQL._to_tgraph - Error: Could not convert dictionary to TGraph. {e}")
            return None

    @staticmethod
    def _annotate_result(result, ontology: bool = True, silent: bool = True):
        """
        Recursively annotates TGraph values inside query results.

        Parameters
        ----------
        result : any
            The query result to annotate.
        ontology : bool , optional
            If True, annotation is attempted. Default is True.
        silent : bool , optional
            If True, warning messages are suppressed. Default is True.

        Returns
        -------
        any
            The annotated result.
        """
        if ontology is not True:
            return result

        try:
            from topologicpy.TGraph import TGraph
            if isinstance(result, TGraph):
                return GQL._annotate_tgraph(result, generatedBy="GQL.Query", silent=silent)
        except Exception:
            pass

        if isinstance(result, list):
            return [GQL._annotate_result(item, ontology=ontology, silent=silent) for item in result]

        if isinstance(result, tuple):
            return tuple(GQL._annotate_result(item, ontology=ontology, silent=silent) for item in result)

        if isinstance(result, dict):
            out = dict(result)

            if "graph" in out:
                out["graph"] = GQL._annotate_result(out["graph"], ontology=ontology, silent=silent)

            if "tgraph" in out:
                out["tgraph"] = GQL._annotate_result(out["tgraph"], ontology=ontology, silent=silent)

            # Backward-compatible key from older wrappers. If present, convert it
            # to TGraph-compatible metadata rather than materialising Graph.
            if "topologic_graph" in out:
                out["topologic_graph"] = GQL._annotate_result(out["topologic_graph"], ontology=ontology, silent=silent)

            for key, value in list(out.items()):
                if isinstance(value, (dict, list, tuple)):
                    out[key] = GQL._annotate_result(value, ontology=ontology, silent=silent)
            return out

        return result

    @staticmethod
    def NormalizeOntologyLabels(query: str,
                                ontologyClassKey: str = ONTOLOGY_CLASS_KEY,
                                labelKey: str = LABEL_KEY,
                                mode: str = "label",
                                silent: bool = False) -> str:
        """
        Normalises ontology labels in a GQL-like query.

        Many simple GQL parsers treat labels as single identifiers and cannot
        parse ``:top:Room``. This helper converts ontology-prefixed labels to a
        parser-safe form.

        Parameters
        ----------
        query : str
            The input GQL-like query string.
        ontologyClassKey : str , optional
            The dictionary key used for ontology class metadata.
            Default is ``"ontology_class"``.
        labelKey : str , optional
            The dictionary key used for ordinary labels. Default is ``"label"``.
        mode : str , optional
            Supported values:

            - ``"none"``: return query unchanged.
            - ``"label"``: replace ``:top:Room`` with ``:Room``.
            - ``"where"``: replace ``:top:Room`` with no label and append a
              best-effort WHERE clause is intentionally not attempted because
              the parser grammar may vary.

            Default is ``"label"``.

        silent : bool , optional
            If True, suppress warning messages. Default is False.

        Returns
        -------
        str
            The normalised query string.
        """
        if not isinstance(query, str):
            return query

        mode = str(mode or "label").strip().lower()
        if mode in ("none", "off", "false", "0"):
            return query

        import re

        if mode == "label":
            # Convert (n:top:Room) -> (n:Room), [:top:adjacentTo] -> [:adjacentTo]
            # This keeps most simple parser grammars happy while preserving the
            # local ontology name as the query label/type.
            def repl(match):
                local = match.group(1)
                return f":{local}"

            return re.sub(r":top:([A-Za-z_][A-Za-z0-9_]*)", repl, query)

        if not silent:
            print(f"GQL.NormalizeOntologyLabels - Warning: Unsupported mode '{mode}'. Returning query unchanged.")
        return query

    @staticmethod
    def OntologyTerms(result,
                      ontologyClassKey: str = ONTOLOGY_CLASS_KEY,
                      categoryKey: str = CATEGORY_KEY,
                      labelKey: str = LABEL_KEY,
                      silent: bool = False):
        """
        Extracts ontology-related terms from query results.

        Parameters
        ----------
        result : any
            A GQL query result, mutation result, graph, row list, or row dict.
        ontologyClassKey : str , optional
            The dictionary key that stores ontology classes.
        categoryKey : str , optional
            The dictionary key that stores categories.
        labelKey : str , optional
            The dictionary key that stores labels.
        silent : bool , optional
            If True, suppress warnings.

        Returns
        -------
        dict
            A dictionary containing unique ontology classes, categories, and labels.
        """
        classes = []
        categories = []
        labels = []

        def add_unique(target, value):
            if value is None:
                return
            value = str(value).strip()
            if value and value not in target:
                target.append(value)

        def visit(value):
            if value is None:
                return

            try:
                from topologicpy.TGraph import TGraph

                if isinstance(value, TGraph):
                    pd = TGraph.Dictionary(value)
                    add_unique(classes, pd.get(ontologyClassKey))
                    add_unique(categories, pd.get(categoryKey))
                    add_unique(labels, pd.get(labelKey))
                    for v in TGraph.Vertices(value, asTopologic=False, activeOnly=True) or []:
                        visit(v)
                    for e in TGraph.Edges(value, asTopologic=False, activeOnly=True) or []:
                        visit(e)
                    return
            except Exception:
                pass

            try:
                from topologicpy.Topology import Topology

                if Topology.IsInstance(value, "Topology"):
                    pd = GQL._python_dictionary(Topology.Dictionary(value))
                    add_unique(classes, pd.get(ontologyClassKey))
                    add_unique(categories, pd.get(categoryKey))
                    add_unique(labels, pd.get(labelKey))
                    return
            except Exception:
                pass

            if isinstance(value, dict):
                add_unique(classes, value.get(ontologyClassKey))
                add_unique(categories, value.get(categoryKey))
                add_unique(labels, value.get(labelKey))
                for v in value.values():
                    if isinstance(v, (dict, list, tuple)):
                        visit(v)
                return

            if isinstance(value, (list, tuple, set)):
                for item in value:
                    visit(item)
                return

        try:
            visit(result)
        except Exception as e:
            if not silent:
                print(f"GQL.OntologyTerms - Warning: {e}")

        return {
            "ontology_classes": classes,
            "categories": categories,
            "labels": labels,
        }

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    @staticmethod
    def Query(graph,
              query: str,
              ontology: bool = True,
              normalizeOntologyLabels: bool = True,
              ontologyLabelMode: str = "label",
              silent: bool = False):
        """
        Parses and executes a GQL-like query.

        The executor accepts either a TopologicPy TGraph or the internal GQL
        working graph returned by a previous mutation. For best performance,
        continue passing the returned ``result["graph"]`` between mutation and
        read queries. Convert back to a TopologicPy TGraph only when needed by
        calling ``GQL.TGraph(...)``.

        Parameters
        ----------
        graph : topologicpy.TGraph or dict
            The input TGraph or GQL working graph.
        query : str
            The GQL-like query string to parse and execute.
        ontology : bool , optional
            If set to True, GQL annotates returned TopologicPy TGraphs with
            ontology metadata where possible and preserves ontology dictionaries
            in result rows. Default is True.
        normalizeOntologyLabels : bool , optional
            If set to True, parser-hostile labels such as ``:top:Room`` are
            normalised to ``:Room`` before parsing. Default is True.
        ontologyLabelMode : str , optional
            The normalisation mode passed to ``GQL.NormalizeOntologyLabels``.
            Default is ``"label"``.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        list or dict or None
            The query result, mutation result, or None if parsing/execution fails.
        """

        from topologicpy.gql.Parser import Parser
        from topologicpy.gql.Executor import Executor

        if normalizeOntologyLabels:
            query = GQL.NormalizeOntologyLabels(
                query,
                mode=ontologyLabelMode,
                silent=silent,
            )

        ast = Parser.Parse(query, silent=silent)
        if ast is None:
            return None

        result = Executor.Execute(graph, ast, silent=silent)
        return GQL._annotate_result(result, ontology=ontology, silent=silent)

    @staticmethod
    def Mutate(graph,
               query: str,
               ontology: bool = True,
               normalizeOntologyLabels: bool = True,
               ontologyLabelMode: str = "label",
               silent: bool = False):
        """
        Alias for Query, intended for CREATE, MERGE, SET, and DELETE queries.

        Parameters
        ----------
        graph : topologicpy.TGraph or dict
            The input TGraph or GQL working graph.
        query : str
            The GQL-like mutation query.
        ontology : bool , optional
            If True, ontology metadata is preserved and graph outputs are
            annotated where possible. Default is True.
        normalizeOntologyLabels : bool , optional
            If True, normalises ``top:`` labels before parsing. Default is True.
        ontologyLabelMode : str , optional
            Ontology label normalisation mode. Default is ``"label"``.
        silent : bool , optional
            Suppress warnings/errors. Default is False.

        Returns
        -------
        list or dict or None
            The mutation result, or None.
        """

        return GQL.Query(
            graph,
            query,
            ontology=ontology,
            normalizeOntologyLabels=normalizeOntologyLabels,
            ontologyLabelMode=ontologyLabelMode,
            silent=silent,
        )

    @staticmethod
    def TGraph(graph,
               ontology: bool = True,
               ontologyClass: str = DEFAULT_GRAPH_ONTOLOGY_CLASS,
               generatedBy: str = "GQL.TGraph",
               silent: bool = False):
        """
        Returns a TGraph from a TGraph or GQL working graph.

        Parameters
        ----------
        graph : topologicpy.TGraph or dict
            The input TGraph or GQL working graph.
        ontology : bool , optional
            If True, the returned TGraph is annotated with ontology metadata.
            Default is True.
        ontologyClass : str , optional
            The ontology class assigned to the returned TGraph when missing.
            Default is ``"top:Graph"``.
        generatedBy : str , optional
            The value stored under ``generated_by`` when missing.
            Default is ``"GQL.TGraph"``.
        silent : bool , optional
            Suppress warnings/errors. Default is False.

        Returns
        -------
        topologicpy.TGraph
            The returned TGraph, or None.
        """

        return GQL._to_tgraph(
            graph,
            ontology=ontology,
            ontologyClass=ontologyClass,
            generatedBy=generatedBy,
            silent=silent,
        )

    @staticmethod
    def TopologicGraph(graph,
                       ontology: bool = True,
                       ontologyClass: str = DEFAULT_GRAPH_ONTOLOGY_CLASS,
                       generatedBy: str = "GQL.TopologicGraph",
                       silent: bool = False):
        """
        Compatibility alias for ``GQL.TGraph``.

        Parameters
        ----------
        graph : topologicpy.TGraph or dict
            The input TGraph or GQL working graph.
        ontology : bool , optional
            If True, the returned TGraph is annotated with ontology metadata.
            Default is True.
        ontologyClass : str , optional
            The ontology class assigned to the returned TGraph when missing.
            Default is ``"top:Graph"``.
        generatedBy : str , optional
            The value stored under ``generated_by`` when missing.
            Default is ``"GQL.TopologicGraph"``.
        silent : bool , optional
            Suppress warnings/errors. Default is False.

        Returns
        -------
        topologicpy.TGraph
            The returned TGraph, or None.
        """

        return GQL.TGraph(
            graph,
            ontology=ontology,
            ontologyClass=ontologyClass,
            generatedBy=generatedBy,
            silent=silent,
        )
