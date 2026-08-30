# Copyright (C) 2026
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3.0 of the License, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Topology
========

Public router for the TopologicPy Topology class.

The implementation is selected according to the active TopologicPy Core
backend:

- TopologicCore -> TopologyLegacy.Topology
- PythonOCC    -> TopologyPythonOCC.Topology

This module intentionally exposes only the selected class under the public
name ``Topology`` so existing imports remain unchanged:

    from topologicpy.Topology import Topology
"""

from __future__ import annotations

from topologicpy.Core import Core


def _topology_class():
    """
    Returns the Topology implementation associated with the active Core backend.

    Returns
    -------
    type
        The selected Topology class.

    Raises
    ------
    ImportError
        If no supported Core backend is active.
    """

    try:
        backend = Core.Backend()
    except Exception as error:
        raise ImportError(
            "Topology - Could not determine the active TopologicPy Core backend."
        ) from error

    if backend is None:
        raise ImportError(
            "Topology - No active TopologicPy Core backend was found."
        )

    backend_name = backend.__class__.__name__

    if backend_name == "TopologicCoreBackend":
        from topologicpy.TopologyLegacy import Topology
        return Topology

    if backend_name == "PythonOCCBackend":
        from topologicpy.TopologyPythonOCC import Topology
        return Topology

    raise ImportError(
        "Topology - Unsupported TopologicPy Core backend: "
        f"{backend_name}."
    )


Topology = _topology_class()

__all__ = ["Topology"]

del _topology_class