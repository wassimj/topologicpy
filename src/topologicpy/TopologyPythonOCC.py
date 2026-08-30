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
PythonOCC implementation of the public Topology class.

During the migration away from the legacy dual-backend Topology
implementation, this class temporarily inherits methods that have not yet been
rewritten specifically for PythonOCC.

Legacy methods frequently refer to ``Topology`` by its module-global name.
After creating the PythonOCC subclass, the legacy module's global ``Topology``
reference is therefore rebound to this subclass. This provides late dispatch
for calls such as::

    Topology.Faces(...)
    Topology.IsInstance(...)
    Topology.Triangulate(...)

inside inherited legacy methods.

As methods are progressively rewritten in this class, calls made by inherited
methods will consequently resolve to the new PythonOCC implementations.

This compatibility bridge is temporary. Once the PythonOCC Topology class is
complete, inheritance from TopologyLegacy and the module-global rebinding
should both be removed.
"""

from __future__ import annotations

import topologicpy.TopologyLegacy as _topology_legacy_module


# Preserve a private reference to the actual frozen legacy class before the
# legacy module's public Topology symbol is rebound below.
TopologyLegacy = _topology_legacy_module.Topology


class Topology(TopologyLegacy):
    """
    PythonOCC-first Topology implementation.

    Methods will progressively replace those inherited from TopologyLegacy.
    The inheritance exists only as a temporary migration mechanism.
    """

    pass


# ---------------------------------------------------------------------------
# Transitional late-dispatch bridge
# ---------------------------------------------------------------------------
#
# Functions inherited from TopologyLegacy retain the global namespace of the
# module in which they were originally defined. Thus a legacy method containing:
#
#     Topology.Faces(...)
#
# looks up ``Topology`` in TopologyLegacy.py, not in this module.
#
# Rebind that module-global symbol to the PythonOCC subclass so inherited
# methods call the progressively modernized PythonOCC implementation.
#
# This assignment occurs only when TopologyPythonOCC is imported. Under the
# TopologicCore backend, the public router imports TopologyLegacy directly and
# this module is never imported, so TopologicCore continues to use the original
# frozen class unchanged.
#
_topology_legacy_module.Topology = Topology


__all__ = ["Topology"]