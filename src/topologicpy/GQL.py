# -*- coding: utf-8 -*-

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

- ``ontology=True`` annotates TopologicPy graph results where possible.
- ``top:Class`` query labels can be normalised to parser-safe labels when
  ``normalizeOntologyLabels=True``.
- Query results can be enriched with ontology metadata copied from returned
  graph elements or working-graph dictionaries.
- Returned TopologicPy graphs can be annotated with ``top:Graph``.
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
            return topology_or_dict.get(key, default)
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
    def _annotate_topologic_graph(graph,
                                  ontologyClass: str = None,
                                  generatedBy: str = "GQL",
                                  silent: bool = True):
        """
        Adds lightweight ontology metadata to a TopologicPy graph and its
        vertices/edges where missing.

        This method preserves existing ontology metadata and only fills missing
        keys. It does not change graph topology.
        """
        try:
            from topologicpy.Graph import Graph
            from topologicpy.Topology import Topology
            from topologicpy.Dictionary import Dictionary

            if not Topology.IsInstance(graph, "Graph"):
                return graph

            graph_class = GQL._normalise_top_class(ontologyClass) or GQL.DEFAULT_GRAPH_ONTOLOGY_CLASS

            def _value(t, key, default=None):
                try:
                    return Dictionary.ValueAtKey(Topology.Dictionary(t), key, default)
                except Exception:
                    return default

            def _set_missing(t, pairs):
                keys = []
                values = []
                for k, v in pairs:
                    if v is None:
                        continue
                    current = _value(t, k, None)
                    if current is None or str(current).strip() == "":
                        keys.append(k)
                        values.append(v)
                if keys:
                    return GQL._set_dictionary_values(t, keys, values, silent=silent)
                return t

            graph = _set_missing(
                graph,
                [
                    (GQL.ONTOLOGY_CLASS_KEY, graph_class),
                    (GQL.CATEGORY_KEY, "graph"),
                    (GQL.GENERATED_BY_KEY, generatedBy),
                ],
            )

            vertices = Graph.Vertices(graph) or []
            for v in vertices:
                label = _value(v, GQL.LABEL_KEY, None)
                node_class = _value(v, GQL.ONTOLOGY_CLASS_KEY, None)
                if node_class is None or str(node_class).strip() == "":
                    if isinstance(label, str) and label.startswith("top:"):
                        node_class = label
                    else:
                        node_class = GQL.DEFAULT_NODE_ONTOLOGY_CLASS
                _set_missing(
                    v,
                    [
                        (GQL.ONTOLOGY_CLASS_KEY, GQL._normalise_top_class(node_class)),
                        (GQL.CATEGORY_KEY, "node"),
                    ],
                )

            edges = Graph.Edges(graph) or []
            for e in edges:
                _set_missing(
                    e,
                    [
                        (GQL.ONTOLOGY_CLASS_KEY, GQL.DEFAULT_EDGE_ONTOLOGY_CLASS),
                        (GQL.CATEGORY_KEY, "relationship"),
                    ],
                )

            return graph
        except Exception as e:
            if not silent:
                print(f"GQL._annotate_topologic_graph - Warning: {e}")
            return graph

    @staticmethod
    def _annotate_result(result, ontology: bool = True, silent: bool = True):
        """
        Recursively annotates TopologicPy graph values inside query results.
        """
        if ontology is not True:
            return result

        try:
            from topologicpy.Topology import Topology
        except Exception:
            Topology = None

        if Topology is not None:
            try:
                if Topology.IsInstance(result, "Graph"):
                    return GQL._annotate_topologic_graph(result, generatedBy="GQL.Query", silent=silent)
            except Exception:
                pass

        if isinstance(result, list):
            return [GQL._annotate_result(item, ontology=ontology, silent=silent) for item in result]

        if isinstance(result, tuple):
            return tuple(GQL._annotate_result(item, ontology=ontology, silent=silent) for item in result)

        if isinstance(result, dict):
            out = dict(result)

            # Mutation results commonly store the working graph under "graph".
            if "graph" in out:
                out["graph"] = GQL._annotate_result(out["graph"], ontology=ontology, silent=silent)

            # Some executors return an already materialised TopologicPy graph.
            if "topologic_graph" in out:
                out["topologic_graph"] = GQL._annotate_result(out["topologic_graph"], ontology=ontology, silent=silent)

            # Enrich row dictionaries without overwriting.
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
                prefix = match.group(1)
                local = match.group(2)
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
                from topologicpy.Topology import Topology
                from topologicpy.Graph import Graph

                if Topology.IsInstance(value, "Graph"):
                    d = Topology.Dictionary(value)
                    pd = GQL._python_dictionary(d)
                    add_unique(classes, pd.get(ontologyClassKey))
                    add_unique(categories, pd.get(categoryKey))
                    add_unique(labels, pd.get(labelKey))
                    for v in Graph.Vertices(value) or []:
                        visit(v)
                    for e in Graph.Edges(value) or []:
                        visit(e)
                    return

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

        The executor accepts either a TopologicPy graph or the internal GQL
        working graph returned by a previous mutation. For best performance,
        continue passing the returned ``result["graph"]`` between mutation and
        read queries. Convert back to a TopologicPy graph only when needed by
        calling ``GQL.TopologicGraph(...)``.

        Parameters
        ----------
        graph : topologic_core.Graph or dict
            The input TopologicPy graph or GQL working graph.
        query : str
            The GQL-like query string to parse and execute.
        ontology : bool , optional
            If set to True, GQL annotates returned TopologicPy graphs with
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
        graph : topologic_core.Graph or dict
            The input TopologicPy graph or GQL working graph.
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
    def TopologicGraph(graph,
                       ontology: bool = True,
                       ontologyClass: str = DEFAULT_GRAPH_ONTOLOGY_CLASS,
                       generatedBy: str = "GQL.TopologicGraph",
                       silent: bool = False):
        """
        Returns a TopologicPy Graph from a TopologicPy graph or GQL working graph.

        Parameters
        ----------
        graph : topologic_core.Graph or dict
            The input TopologicPy graph or GQL working graph.
        ontology : bool , optional
            If True, the returned graph is annotated with ontology metadata.
            Default is True.
        ontologyClass : str , optional
            The ontology class assigned to the returned graph when missing.
            Default is ``"top:Graph"``.
        generatedBy : str , optional
            The value stored under ``generated_by`` when missing.
            Default is ``"GQL.TopologicGraph"``.
        silent : bool , optional
            Suppress warnings/errors. Default is False.

        Returns
        -------
        topologic_core.Graph
            The returned TopologicPy Graph, or None.
        """

        from topologicpy.gql.Executor import Executor

        result = Executor.TopologicGraph(graph, silent=silent)
        if ontology:
            result = GQL._annotate_topologic_graph(
                result,
                ontologyClass=ontologyClass,
                generatedBy=generatedBy,
                silent=silent,
            )
        return result
