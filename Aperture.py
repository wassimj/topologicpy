# Copyright (C) 2024
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

import topologicpy
import topologic_core as topologic

class Aperture(topologic.Aperture):
    @staticmethod
    def Topology(aperture: topologic.Aperture) -> topologic.Topology:
        """
        Returns the topology of the input aperture.
        
        Parameters
        ----------
        aperture : topologic.Aperture
            The input aperture.

        Returns
        -------
        topologic.Topology
            The topology of the input aperture.

        """
        if not isinstance(aperture, topologic.Aperture):
            print("Aperture.Topology - Error: The input aperture parameter is not a valid topologic aperture. Returning None.")
            return None
        return topologic.Aperture.Topology(aperture)

    @staticmethod
    def ByTopologyContext(topology: topologic.Topology, context: topologic.Context) -> topologic.Aperture:
        """
        Creates an aperture object represented by the input topology and one that belongs to the input context.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology that represents the aperture.
        context : topologic.Context
            The context of the aperture. See Context class.

        Returns
        -------
        topologic.Aperture
            The created aperture.

        """
        if not isinstance(topology, topologic.Topology):
            print("Aperture.ByTopologyContext - Error: The input topology parameter is not a valid topologic topology. Returning None.")
            return None
        if not isinstance(context, topologic.Context):
            print("Aperture.ByTopologyContext - Error: The input context parameter is not a valid topologic context. Returning None.")
            return None
        aperture = None
        try:
            aperture = topologic.Aperture.ByTopologyContext(topology, context)
        except:
            print("Aperture.ByTopologyContext - Error: The operation failed. Returning None.")
            aperture = None
        return aperture
    