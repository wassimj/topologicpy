# Copyright (C) 2025
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

import topologic_core as topologic

class Aperture():
    @staticmethod
    def Topology(aperture):
        """
        Returns the topology of the input aperture.
        
        Parameters
        ----------
        aperture : Aperture
            The input aperture.

        Returns
        -------
        Topology
            The topology of the input aperture.

        """
        from topologicpy.Topology import Topology
        if not Topology.IsInstance(aperture, "aperture"):
            print("Aperture.Topology - Error: The input aperture parameter is not a valid topologic aperture. Returning None.")
            return None
        return topologic.Aperture.Topology(aperture) # Hook to Core

    @staticmethod
    def ByTopologyContext(topology, context):
        """
        Creates an aperture object represented by the input topology and one that belongs to the input context.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology that represents the aperture.
        context : Context
            The context of the aperture. See Context class.

        Returns
        -------
        Aperture
            The created aperture.

        """
        from topologicpy.Topology import Topology
        if not Topology.IsInstance(topology, "Topology"):
            print("Aperture.ByTopologyContext - Error: The input topology parameter is not a valid topologic topology. Returning None.")
            return None
        if not Topology.IsInstance(context, "Context"):
            print("Aperture.ByTopologyContext - Error: The input context parameter is not a valid topologic context. Returning None.")
            return None
        aperture = None
        try:
            aperture = topologic.Aperture.ByTopologyContext(topology, context) # Hook to Core
        except:
            print("Aperture.ByTopologyContext - Error: The operation failed. Returning None.")
            aperture = None
        return aperture
    