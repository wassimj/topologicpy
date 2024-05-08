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

import topologic_core as topologic

class Context:
    @staticmethod
    def ByTopologyParameters(topology, u = 0.5, v = 0.5, w = 0.5):
        """
        Creates a context object represented by the input topology.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        u : float , optional
            The input *u* parameter. This defines the relative parameteric location of the content object along the *u* axis.
        v : TYPE
            The input *v* parameter. This defines the relative parameteric location of the content object along the *v* axis..
        w : TYPE
            The input *w* parameter. This defines the relative parameteric location of the content object along the *w* axis.

        Returns
        -------
        topologic.Context
            The created context object. See Aperture.ByObjectContext.

        """
        if not isinstance(topology, topologic.Topology):
            print("Context.ByTopologyParameters - Error: The input topology parameter is not a valid topologic topology. Returning None.")
            return None
        
        context = None
        try:
            context = topologic.Context.ByTopologyParameters(topology, u, v, w)
        except:
            print("Context.ByTopologyParameters - Error: The operation failed. Returning None.")
            context = None
        return context
    
    @staticmethod
    def Topology(context):
        """
        Returns the topology of the input context.
        
        Parameters
        ----------
        context : topologic.Context
            The input context.

        Returns
        -------
        topologic.Topology
            The topology of the input context.

        """
        if not isinstance(context, topologic.Context):
            print("Context.Topology - Error: The input context parameter is not a valid topologic context. Returning None.")
            return None
        topology = None
        try:
            topology = context.Topology()
        except:
            print("Context.Topology - Error: The operation failed. Returning None.")
            topology = None
        return topology
