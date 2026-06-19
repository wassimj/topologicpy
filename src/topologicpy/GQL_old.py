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
"""


class GQL:
    """Convenience API for parsing and executing GQL-like queries."""

    @staticmethod
    def Query(graph, query: str, silent: bool = False):
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

        ast = Parser.Parse(query, silent=silent)
        if ast is None:
            return None

        return Executor.Execute(graph, ast, silent=silent)

    @staticmethod
    def Mutate(graph, query: str, silent: bool = False):
        """Alias for Query, intended for CREATE, MERGE, SET, and DELETE queries."""

        return GQL.Query(graph, query, silent=silent)

    @staticmethod
    def TopologicGraph(graph, silent: bool = False):
        """Returns a TopologicPy Graph from a TopologicPy graph or GQL working graph."""

        from topologicpy.gql.Executor import Executor

        return Executor.TopologicGraph(graph, silent=silent)
