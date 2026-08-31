# Copyright (C) 2026
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3.0 of the License, or (at your option)
# any later version.

from __future__ import annotations

import copy
import uuid

from topologicpy.Core import Core


def _is_pythonocc_backend() -> bool:
    try:
        return "pythonocc" in Core.Backend().__class__.__name__.lower()
    except Exception:
        return False


class Context:
    """One semantic relationship between a Content and one host topology.

    ``parameters`` are optional relationship metadata. They can contain UV
    coordinates for a face, a scalar parameter for an edge, or any other
    meaningful structured data. No universal u/v/w parameterisation is assumed.
    """

    def __init__(self, content=None, host=None, parameters=None, dictionary=None, uuid_value=None):
        self.content = content
        self.host = host
        self.parameters = copy.deepcopy(parameters) if parameters is not None else None
        self.dictionary = dict(dictionary) if isinstance(dictionary, dict) else {}
        self._uuid = str(uuid_value) if uuid_value else str(uuid.uuid4())

    def Topology(self):
        """Returns the host topology. Retained as the legacy compatibility name."""
        if isinstance(self, Context):
            return self.host
        try:
            return Core.InstanceCall(self, "Topology")
        except Exception:
            return None

    def Host(self):
        """Returns the host topology of this Context."""
        if isinstance(self, Context):
            return self.host
        try:
            return Core.InstanceCall(self, "Topology")
        except Exception:
            return None

    def Content(self):
        """Returns the semantic Content participating in this Context."""
        return self.content if isinstance(self, Context) else None

    def Parameters(self):
        """Returns a deep copy of the optional relationship parameters."""
        return copy.deepcopy(self.parameters) if isinstance(self, Context) else None

    def Dictionary(self):
        """Returns a shallow copy of the Context dictionary."""
        if isinstance(self, Context):
            return dict(self.dictionary)
        try:
            return Core.InstanceCall(self, "GetDictionary")
        except Exception:
            return {}

    def SetDictionary(self, dictionary):
        """Sets the Context dictionary and returns this Context."""
        if isinstance(self, Context):
            self.dictionary = dict(dictionary) if isinstance(dictionary, dict) else {}
            return self
        try:
            return Core.InstanceCall(self, "SetDictionary", dictionary)
        except Exception:
            return self

    @staticmethod
    def ByContentHost(content, host, parameters=None, dictionary=None, silent: bool = False):
        """Creates or returns the Context linking ``content`` to ``host``."""
        from topologicpy.Content import Content
        from topologicpy.SemanticManager import SemanticManager
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(host, "Topology"):
            if not silent:
                print("Context.ByContentHost - Error: The input host parameter is not a valid topology. Returning None.")
            return None

        if not isinstance(content, Content) and not Topology.IsInstance(content, "Topology"):
            if not silent:
                print("Context.ByContentHost - Error: The input content parameter is not valid. Returning None.")
            return None

        _, context = SemanticManager.GetInstance().register(
            content,
            host,
            parameters=parameters,
            context_dictionary=dictionary,
        )
        return context

    @staticmethod
    def ByTopologyParameters(topology, u=0.5, v=0.5, w=0.5):
        """Creates a Context specification for a host topology.

        On PythonOCC this legacy factory creates an *unbound* Context whose host
        is ``topology`` and whose optional parameters contain u/v/w. The Context
        becomes bound to a Content when passed to ``Aperture.ByTopologyContext``
        or registered through ``SemanticManager``. On TopologicCore the native
        Context implementation is used unchanged.
        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(topology, "Topology"):
            print("Context.ByTopologyParameters - Error: The input topology parameter is not a valid topologic topology. Returning None.")
            return None

        if _is_pythonocc_backend():
            try:
                parameters = {"u": float(u), "v": float(v), "w": float(w)}
            except Exception:
                print("Context.ByTopologyParameters - Error: The input parameters are not numeric. Returning None.")
                return None
            return Context(content=None, host=topology, parameters=parameters)

        try:
            return Core.Context.ByTopologyParameters(topology, u, v, w)
        except Exception:
            print("Context.ByTopologyParameters - Error: The operation failed. Returning None.")
            return None


__all__ = ["Context"]
