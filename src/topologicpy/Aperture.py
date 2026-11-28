# Copyright (C) 2026
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free Software
# Foundation, either version 3.0 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

from topologicpy.Core import Core

class Aperture:
    """Semantic aperture API. An aperture is a native topology tagged as content."""

    @staticmethod
    def Topology(aperture):
        """Return the tagged native topology, without creating another object."""
        return Core.Aperture.Topology(aperture)

    @staticmethod
    def ByTopologyContext(topology, context):
        """Tag and attach the supplied topology to the explicit context host.

        The supplied topology is returned, not a separate aperture wrapper.
        Its dictionary and geometric type are retained.
        """
        from topologicpy.Topology import Topology
        if not Topology.IsInstance(topology, "Topology") or not Topology.IsInstance(context, "Context"):
            print("Aperture.ByTopologyContext - Error: Invalid topology or context.")
            return None
        try:
            return Core.Aperture.ByTopologyContext(topology, context)
        except Exception as exc:
            print(f"Aperture.ByTopologyContext - Error: {exc}")
            return None
