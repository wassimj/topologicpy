# Copyright (C) 2026
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3.0 of the License, or (at your option)
# any later version.

from __future__ import annotations

import uuid


class Content:
    """A semantic object represented geometrically by a TopologicPy topology.

    A Content can participate in zero or more :class:`Context` relationships.
    The represented topology owns the geometry; this object owns only semantic
    identity and relationship-level metadata.
    """

    def __init__(self, topology, dictionary=None, uuid_value=None):
        self.topology = topology
        self.dictionary = dict(dictionary) if isinstance(dictionary, dict) else {}
        self._uuid = str(uuid_value) if uuid_value else str(uuid.uuid4())

    def Topology(self):
        """Returns the topology that geometrically represents this Content."""
        return self.topology

    def Contexts(self):
        """Returns all Context relationships associated with this Content."""
        from topologicpy.SemanticManager import SemanticManager
        return SemanticManager.GetInstance().contexts_for_content(self)

    def Dictionary(self):
        """Returns a shallow copy of the semantic dictionary."""
        return dict(self.dictionary)

    def SetDictionary(self, dictionary):
        """Sets the semantic dictionary and returns this Content."""
        self.dictionary = dict(dictionary) if isinstance(dictionary, dict) else {}
        return self

    @staticmethod
    def ByTopology(topology, dictionary=None, silent: bool = False):
        """Returns the unique semantic Content represented by ``topology``."""
        from topologicpy.Topology import Topology
        from topologicpy.SemanticManager import SemanticManager

        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print("Content.ByTopology - Error: The input topology parameter is not a valid topology. Returning None.")
            return None

        return SemanticManager.GetInstance().content_for_topology(
            topology,
            create=True,
            dictionary=dictionary,
        )


__all__ = ["Content"]
