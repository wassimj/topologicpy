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


class Context:
    """One semantic relationship between a Content and one host topology.

    ``parameters`` are optional relationship metadata. They can contain UV
    coordinates for a face, a scalar parameter for an edge, or any other
    meaningful structured data. No universal u/v/w parameterisation is assumed.

    Context is a TopologicPy semantic object and is intentionally independent of
    the active geometry kernel.
    """

    def __init__(self, content=None, host=None, parameters=None, dictionary=None, uuid_value=None):
        self.content = content
        self.host = host
        self.parameters = copy.deepcopy(parameters) if parameters is not None else None
        self.dictionary = dict(dictionary) if isinstance(dictionary, dict) else {}
        self._uuid = str(uuid_value) if uuid_value else str(uuid.uuid4())

    @staticmethod
    def _is_native_context(value) -> bool:
        if value is None:
            return False
        try:
            return isinstance(value, Core.Namespace("Context"))
        except Exception:
            return False

    def Topology(self):
        """Returns the host topology. Retained as the legacy compatibility name."""
        if isinstance(self, Context):
            return self.host
        if not Context._is_native_context(self):
            return None
        try:
            return Core.InstanceCall(self, "Topology")
        except Exception:
            return None

    def Host(self):
        """Returns the host topology of this Context."""
        return Context.Topology(self)

    def Content(self):
        """Returns the semantic Content participating in this Context."""
        return self.content if isinstance(self, Context) else None

    def Parameters(self):
        """Returns a deep copy of the optional relationship parameters."""
        if isinstance(self, Context):
            return copy.deepcopy(self.parameters)
        return None

    def Dictionary(self):
        """Returns a copy of the Context dictionary."""
        if isinstance(self, Context):
            return dict(self.dictionary)
        if not Context._is_native_context(self):
            return {}
        try:
            dictionary = Core.InstanceCall(self, "GetDictionary")
            return dictionary if dictionary is not None else {}
        except Exception:
            return {}

    def SetDictionary(self, dictionary):
        """Sets the Context dictionary and returns this Context."""
        if isinstance(self, Context):
            self.dictionary = dict(dictionary) if isinstance(dictionary, dict) else {}
            return self
        if not Context._is_native_context(self):
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
        """Creates an unbound Context specification for a host topology.

        This historical factory is now backend independent. The returned Context
        stores the host topology and the supplied u/v/w values as optional
        relationship parameters. It becomes bound to Content when passed to
        ``Aperture.ByTopologyContext`` or registered through SemanticManager.
        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(topology, "Topology"):
            print("Context.ByTopologyParameters - Error: The input topology parameter is not a valid topologic topology. Returning None.")
            return None

        try:
            parameters = {"u": float(u), "v": float(v), "w": float(w)}
        except Exception:
            print("Context.ByTopologyParameters - Error: The input parameters are not numeric. Returning None.")
            return None

        return Context(content=None, host=topology, parameters=parameters)


__all__ = ["Context"]
