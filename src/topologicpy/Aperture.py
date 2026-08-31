# Copyright (C) 2026
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3.0 of the License, or (at your option)
# any later version.

from __future__ import annotations

from topologicpy.Content import Content
from topologicpy.Core import Core


def _is_pythonocc_backend() -> bool:
    try:
        return "pythonocc" in Core.Backend().__class__.__name__.lower()
    except Exception:
        return False


def _mark_aperture_topology(topology):
    """Preserves the historical ``type=Aperture`` topology marker."""
    try:
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology
        dictionary = Topology.Dictionary(topology, silent=True)
        dictionary = Dictionary.SetValueAtKey(dictionary, "type", "Aperture")
        marked = Topology.SetDictionary(topology, dictionary, silent=True)
        if Topology.IsInstance(marked, "Topology"):
            return marked
    except Exception:
        pass
    return topology


class Aperture(Content):
    """A specialised Content representing an aperture such as a window or door.

    One Aperture can participate in multiple Context relationships, for example
    one to its host Face and another to the containing room Cell.
    """

    def Topology(self):
        """Returns the topology represented by this Aperture."""
        if isinstance(self, Aperture):
            return self.topology
        try:
            return Core.Aperture.Topology(self)
        except Exception:
            return None

    @staticmethod
    def ByTopologyHost(topology, host, parameters=None, dictionary=None, silent: bool = False):
        """Creates or returns an Aperture represented by ``topology`` in ``host``."""
        from topologicpy.SemanticManager import SemanticManager
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print("Aperture.ByTopologyHost - Error: The input topology parameter is not a valid topology. Returning None.")
            return None
        if not Topology.IsInstance(host, "Topology"):
            if not silent:
                print("Aperture.ByTopologyHost - Error: The input host parameter is not a valid topology. Returning None.")
            return None

        topology = _mark_aperture_topology(topology)
        aperture, _ = SemanticManager.GetInstance().register(
            topology,
            host,
            aperture=True,
            parameters=parameters,
            content_dictionary=dictionary,
        )
        return aperture

    @staticmethod
    def ByTopologyContext(topology, context):
        """Creates an Aperture from a represented topology and a Context.

        This remains compatible with the historical API. On PythonOCC the
        supplied Context is bound to the unique Aperture Content. On
        TopologicCore the native Aperture implementation is used unchanged.
        """
        from topologicpy.Context import Context
        from topologicpy.SemanticManager import SemanticManager
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(topology, "Topology"):
            print("Aperture.ByTopologyContext - Error: The input topology parameter is not a valid topologic topology. Returning None.")
            return None
        if not Topology.IsInstance(context, "Context"):
            print("Aperture.ByTopologyContext - Error: The input context parameter is not a valid topologic context. Returning None.")
            return None

        if not _is_pythonocc_backend():
            try:
                return Core.Aperture.ByTopologyContext(topology, context)
            except Exception:
                print("Aperture.ByTopologyContext - Error: The operation failed. Returning None.")
                return None

        host = context.Host() if isinstance(context, Context) else None
        if not Topology.IsInstance(host, "Topology"):
            print("Aperture.ByTopologyContext - Error: The input context does not reference a valid host topology. Returning None.")
            return None

        topology = _mark_aperture_topology(topology)
        aperture, _ = SemanticManager.GetInstance().register(
            topology,
            host,
            aperture=True,
            parameters=context.Parameters() if isinstance(context, Context) else None,
            context_dictionary=context.Dictionary() if isinstance(context, Context) else None,
            context=context if isinstance(context, Context) else None,
        )
        return aperture


__all__ = ["Aperture"]
