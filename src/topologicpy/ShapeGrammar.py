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

class ShapeGrammar:
    def __init__(self):
        self.title = "Untitled" # Stores the title of the topology grammar.
        self.description = "" # Stores the description of the grammar.
        self.rules = []  # Stores transformation rules of the topology grammar.
        # Operations
        # Replace
        replace = {"title": "Replace",
                   "description": "Replaces the input topology with the output topology.",
                   "uSides": None,
                   "vSides": None,
                   "wSides": None}
        # Transform
        transform = {"title": "Transform",
                   "description": "Transforms the input topology using the specified matrix.",
                   "uSides": None,
                   "vSides": None,
                   "wSides": None}
        # Union
        union = {"title": "Union",
                   "description": "Unions the input topology and the output topology.",
                   "uSides": None,
                   "vSides": None,
                   "wSides": None}
        # Difference
        difference = {"title": "Difference",
                   "description": "Subtracts the output topology from the input topology.",
                   "uSides": None,
                   "vSides": None,
                   "wSides": None}
        # Difference
        symdif = {"title": "Symmetric Difference",
                   "description": "Calculates the symmetrical difference of the input topology and the output topology.",
                   "uSides": None,
                   "vSides": None,
                   "wSides": None}
        # Intersect
        intersect = {"title": "Intersect",
                   "description": "Intersects the input topology and the output topology.",
                   "uSides": None,
                   "vSides": None,
                   "wSides": None}
        # Merge
        merge = {"title": "Merge",
                   "description": "Merges the input topology and the output topology.",
                   "uSides": None,
                   "vSides": None,
                   "wSides": None}
        # Slice
        slice = {"title": "Slice",
                   "description": "Slices the input topology using the output topology.",
                   "uSides": None,
                   "vSides": None,
                   "wSides": None}
        # Impose
        impose = {"title": "Impose",
                   "description": "Imposes the output topology on the input topology.",
                   "uSides": None,
                   "vSides": None,
                   "wSides": None}
        # Imprint
        imprint = {"title": "Imprint",
                   "description": "Imposes the output topology on the input topology.",
                   "uSides": None,
                   "vSides": None,
                   "wSides": None}
        # Divide
        divide = {"title": "Divide",
                   "description": "Divides the input topology along the x, y, and z axes using the specified number of sides (uSides, vSides, wSides)",
                   "uSides": 2,
                   "vSides": 2,
                   "wSides": 2}
        self.operations = [replace, transform, union, difference, symdif, intersect, merge, slice, impose, imprint, divide]

    def OperationTitles(self):
        """
        Returns the list of available operation titles.

        Parameters
        ----------
        
        Returns
        -------
        list
            The requested list of operation titles
        """
        return [op["title"] for op in self.operations]
    
    def OperationByTitle(self, title):
        """
        Returns the operation given the input title string

        Parameters
        ----------
        title : str
            The input operation str. See OperationTitles for list of operations.
        
        Returns
        -------
        ShapeGrammar.Operation
            The requested operation
        """
        for op in self.operations:
            op_title = op["title"]
            if title.lower() in op_title.lower():
                return op
        return None

    def AddRule(self,
                input,
                output,
                title : str = "Untitled Rule",
                description: str = "",
                operation : dict = None,
                matrix: list = None,
                silent: bool = False):
        """
        Adds a rule to the topology grammar.

        Parameters
        ----------
        input : topologic_core.Topology
            The linput topology of the rule.
        output : topologic_core.Topology
            The output topology of the rule.
        title : str , optional
            The title of the rule. The default is "Untitled Rule"
        description : str, optional
            The description of the rule. The default is "".
        operation : dict , optional
            The desired rule operation. See Rule Operations. If set to None, the replacement rule is applied. The default is None.
        matrix : list
            The 4x4 transformation matrix that tranforms the output topology to the input topology. If set to None, no transformation is applied. The default is None.
        silent : bool, optional
            If True, suppresses error/warning messages. Default is False.
        
        Returns
        -------
        None
            This method does not return a value.
        """
        from topologicpy.Topology import Topology

        def is_4x4_matrix(matrix):
            return (
                isinstance(matrix, list) and
                len(matrix) == 4 and
                all(isinstance(row, list) and len(row) == 4 for row in matrix)
            )

        if not Topology.IsInstance(input, "Topology"):
            if not silent:
                print("ShapeGrammar.AddRule - Error: The input input parameter is not a valid topology. Returning None.")
            return None
        if not output == None:
            if not Topology.IsInstance(output, "Topology"):
                if not silent:
                    print("ShapeGrammar.AddRule - Error: The input output parameter is not a valid topology. Returning None.")
                return None
        if not operation == None:
            if not operation in self.operations:
                if not silent:
                    print("ShapeGrammar.AddRule - Error: The input operation parameter is not a valid operation. Returning None.")
                return None
        if not matrix == None:
            if not is_4x4_matrix(matrix):
                if not silent:
                    print("ShapeGrammar.AddRule - Error: The input matrix parameter is not a valid matrix. Returning None.")
                return None
        
        self.rules.append({"input":input,
                           "output": output,
                           "title": title,
                           "description": description,
                           "operation": operation,
                           "matrix": matrix
                           })

    def ApplicableRules(self, topology, keys: list = None, silent: bool = False):
        """
        Returns rules applicable to the input topology.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology
        keys : list , optional
            The list of dictionary keys to semantically match the rules. The default is None which means dictionaries are not considered.
        silent : bool, optional
            If True, suppresses error/warning messages. Default is False.
        
        Returns
        -------
        list
            The list of applicable rules.
        """
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print("ShapeGrammar.ApplicableRules - Error: The input topology parameter is not a valid topology. Returning None.")
            return None
        
        ap_rules = []
        ap_trans = []
        d = Topology.Dictionary(topology)
        for i, rule in enumerate(self.rules):
            dict_status = True
            input = rule["input"]
            # If there is a list of keys specified, check that the values match
            if isinstance(keys, list):
                d_input = Topology.Dictionary(input)
                for j, key in enumerate(keys):
                    if not Dictionary.ValueAtKey(d, key, None) == Dictionary.ValueAtKey(d_input, key, None):
                        dict_status = False
                        break
            #If it passed the dictionary key test, then check topology similarity
            if dict_status:
                topology_status, mat = Topology.IsSimilar(rule["input"], topology)
                if topology_status:
                    ap_rules.append(rule)
                    ap_trans.append(mat)
        return ap_rules, ap_trans

    def ApplyRule(self, topology, rule: dict = None, matrix: list = None, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns rules applicable to the input topology.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology
        rule : dict , optional
            The desired rule to apply. The default is None.
        matrix : list
            The 4x4 transformation matrix that tranforms the output topology to the input topology. If set to None, no transformation is applied. The default is None.
        mantissa : int, optional
            Decimal precision. Default is 6.
        tolerance : float, optional
            The desired Tolerance. Not used here but included for API compatibility. Default is 0.0001.
        silent : bool, optional
            If True, suppresses error/warning messages. Default is False.
        
        Returns
        -------
        topologic_core.Topology
            The transformed topology
        """

        from topologicpy.Topology import Topology
        from topologicpy.Cluster import Cluster
        from topologicpy.Face import Face
        from topologicpy.Vertex import Vertex

        def is_4x4_matrix(matrix):
            return (
                isinstance(matrix, list) and
                len(matrix) == 4 and
                all(isinstance(row, list) and len(row) == 4 for row in matrix)
            )
        
        def bb(topology):
            vertices = Topology.Vertices(topology)
            x = []
            y = []
            z = []
            for aVertex in vertices:
                x.append(Vertex.X(aVertex, mantissa=mantissa))
                y.append(Vertex.Y(aVertex, mantissa=mantissa))
                z.append(Vertex.Z(aVertex, mantissa=mantissa))
            x_min = min(x)
            y_min = min(y)
            z_min = min(z)
            maxX = max(x)
            maxY = max(y)
            maxZ = max(z)
            return [x_min, y_min, z_min, maxX, maxY, maxZ]
        
        def slice(topology, uSides, vSides, wSides):
            x_min, y_min, z_min, maxX, maxY, maxZ = bb(topology)
            centroid = Vertex.ByCoordinates(x_min+(maxX-x_min)*0.5, y_min+(maxY-y_min)*0.5, z_min+(maxZ-z_min)*0.5)
            wOrigin = Vertex.ByCoordinates(Vertex.X(centroid, mantissa=mantissa), Vertex.Y(centroid, mantissa=mantissa), z_min)
            wFace = Face.Rectangle(origin=wOrigin, width=(maxX-x_min)*1.1, length=(maxY-y_min)*1.1)
            wFaces = []
            wOffset = (maxZ-z_min)/wSides
            for i in range(wSides-1):
                wFaces.append(Topology.Translate(wFace, 0,0,wOffset*(i+1)))
            uOrigin = Vertex.ByCoordinates(x_min, Vertex.Y(centroid, mantissa=mantissa), Vertex.Z(centroid, mantissa=mantissa))
            uFace = Face.Rectangle(origin=uOrigin, width=(maxZ-z_min)*1.1, length=(maxY-y_min)*1.1, direction=[1,0,0])
            uFaces = []
            uOffset = (maxX-x_min)/uSides
            for i in range(uSides-1):
                uFaces.append(Topology.Translate(uFace, uOffset*(i+1),0,0))
            vOrigin = Vertex.ByCoordinates(Vertex.X(centroid, mantissa=mantissa), y_min, Vertex.Z(centroid, mantissa=mantissa))
            vFace = Face.Rectangle(origin=vOrigin, width=(maxX-x_min)*1.1, length=(maxZ-z_min)*1.1, direction=[0,1,0])
            vFaces = []
            vOffset = (maxY-y_min)/vSides
            for i in range(vSides-1):
                vFaces.append(Topology.Translate(vFace, 0,vOffset*(i+1),0))
            all_faces = uFaces+vFaces+wFaces
            if len(all_faces) > 0:
                f_clus = Cluster.ByTopologies(uFaces+vFaces+wFaces)
                return Topology.Slice(topology, f_clus, tolerance=tolerance)
            else:
                return topology

        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print("ShapeGrammar.ApplyRule - Error: The input topology parameter is not a valid topology. Returning None.")
            return None
        if not matrix == None:
            if not is_4x4_matrix(matrix):
                if not silent:
                    print("ShapeGrammar.ApplyRule - Error: The input matrix parameter is not a valid matrix. Returning None.")
                return None
        
        if not rule == None:
            input = rule["input"]
            output = rule["output"]
            r_matrix = rule["matrix"]
            operation = rule["operation"]
            if not operation == None:
                op_title = operation["title"]
            else:
                op_title = "None"

            result_output = topology
            temp_output = None
            if not output == None:
                temp_output = output
            # Transform the output topology to the input topology to prepare it for final transformation
            if not r_matrix == None and not output == None:
                    temp_output = Topology.Transform(output, r_matrix)

            if "replace" in op_title.lower():
                result_output = temp_output
            elif "transform" in op_title.lower():
                result_output = Topology.Transform(topology, r_matrix)
            elif "union" in op_title.lower():
                result_output = Topology.Union(input, temp_output)
            elif "difference" in op_title.lower():
                result_output = Topology.Difference(input, temp_output)
            elif "symmetric difference" in op_title.lower():
                result_output = Topology.SymmetricDifference(input, temp_output)
            elif "intersect" in op_title.lower():
                result_output = Topology.Intersect(input, temp_output)
            elif "merge" in op_title.lower():
                result_output = Topology.Merge(input, temp_output)
            elif "slice" in op_title.lower():
                result_output = Topology.Slice(input, temp_output)
            elif "impose" in op_title.lower():
                result_output = Topology.Impose(input, temp_output)
            elif "imprint" in op_title.lower():
                result_output = Topology.Imprint(input, temp_output)
            elif "divide" in op_title.lower():
                uSides = operation["uSides"]
                vSides = operation["vSides"]
                wSides = operation["wSides"]
                if not uSides == None and not vSides == None and not wSides == None:
                    result_output = slice(input, uSides, vSides, wSides)
        
        # Finally, transform the result to the input topology
        if not matrix == None:
            result_output = Topology.Transform(result_output, matrix)
        
        return result_output

    def ClusterByInputOutput(self, input, output, silent: bool = False):
        """
        Returns the Plotly figure of the input and output topologies as a rule.

        Parameters
        ----------
        input : topologic_core.Topology
            The input topology
        output : topologic_core.Topology
            The output topology
        silent : bool, optional
            If True, suppresses error/warning messages. Default is False.
        
        Returns
        -------
        This function does not return a value
        """

        from topologicpy.Vertex import Vertex
        from topologicpy.Cell import Cell
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Cluster import Cluster

        if not Topology.IsInstance(input, "Topology"):
            if not silent:
                print("ShapeGrammar.ClusterByInputOutput - Error: The input topology parameter is not a valid topology. Returning None.")
            return None
        if not Topology.IsInstance(output, "Topology"):
            if not silent:
                print("ShapeGrammar.ClusterByInputOutput - Error: The output topology parameter is not a valid topology. Returning None.")
            return None

        input_bb = Topology.BoundingBox(input)
        input_centroid = Topology.Centroid(input_bb)
        input_d = Topology.Dictionary(input_bb)
        xmin = Dictionary.ValueAtKey(input_d, "xmin")
        ymin = Dictionary.ValueAtKey(input_d, "ymin")
        zmin = Dictionary.ValueAtKey(input_d, "zmin")
        xmax = Dictionary.ValueAtKey(input_d, "xmax")
        ymax = Dictionary.ValueAtKey(input_d, "ymax")
        zmax = Dictionary.ValueAtKey(input_d, "zmax")
        input_width = xmax-xmin
        input_length = ymax-ymin
        input_height = zmax-zmin
        input_max = max(input_width, input_length, input_height)
        sf = 1/input_max
        temp_input = Topology.Translate(input, -Vertex.X(input_centroid), -Vertex.Y(input_centroid), -Vertex.Z(input_centroid))
        temp_input = Topology.Scale(temp_input, x=sf, y=sf, z=sf)
        temp_input = Topology.Translate(temp_input, 0.5, 0, 0)

        output_bb = Topology.BoundingBox(output)
        output_centroid = Topology.Centroid(output_bb)
        output_d = Topology.Dictionary(output_bb)
        xmin = Dictionary.ValueAtKey(output_d, "xmin")
        ymin = Dictionary.ValueAtKey(output_d, "ymin")
        zmin = Dictionary.ValueAtKey(output_d, "zmin")
        xmax = Dictionary.ValueAtKey(output_d, "xmax")
        ymax = Dictionary.ValueAtKey(output_d, "ymax")
        zmax = Dictionary.ValueAtKey(output_d, "zmax")
        output_width = xmax-xmin
        output_length = ymax-ymin
        output_height = zmax-zmin
        output_max = max(output_width, output_length, output_height)
        sf = 1/output_max
        temp_output = Topology.Translate(output, -Vertex.X(output_centroid), -Vertex.Y(output_centroid), -Vertex.Z(output_centroid))
        temp_output = Topology.Scale(temp_output, x=sf, y=sf, z=sf)
        temp_output = Topology.Translate(temp_output, 2.5, 0, 0)

        cyl = Cell.Cylinder(radius=0.04, height=0.4, placement="bottom")
        cyl=Topology.Rotate(cyl, axis=[0,1,0], angle=90)
        cyl = Topology.Translate(cyl, 1.25, 0, 0)

        cone = Cell.Cone(baseRadius=0.1, topRadius=0, height=0.15, placement="bottom")
        cone=Topology.Rotate(cone, axis=[0,1,0], angle=90)
        cone = Topology.Translate(cone, 1.65, 0, 0)
        cluster = Cluster.ByTopologies([temp_input, temp_output, cyl, cone])
        cluster = Topology.Place(cluster, originA=Topology.Centroid(cluster), originB=Vertex.Origin())
        return cluster
    
    def ClusterByRule(self, rule, silent: bool = False):
        """
        Returns the Plotly figure of the input rule.

        Parameters
        ----------
        rule : dict
            The input rule
        silent : bool, optional
            If True, suppresses error/warning messages. Default is False.
        
        Returns
        -------
        topologic_core.Cluster
            The created rule cluster
        """
                
        if not isinstance(rule, dict):
            if not silent:
                print("ShapeGrammar.ClusterByRule - Error: The input rule parameter is not a valid rule. Returning None.")
            return None
        input = rule["input"]
        output = self.ApplyRule(input, rule)
        return self.ClusterByInputOutput(input, output, silent=silent)
    
    def FigureByInputOutput(self, input, output, silent: bool = False):
        """
        Returns the Plotly figure of the input and output topologies as a rule.

        Parameters
        ----------
        input : topologic_core.Topology
            The input topology
        output : topologic_core.Topology
            The output topology
        silent : bool, optional
            If True, suppresses error/warning messages. Default is False.
        
        Returns
        -------
        Plotly.Figure
            The created plotly figure.
        """

        from topologicpy.Topology import Topology
        from topologicpy.Plotly import Plotly

        if not Topology.IsInstance(input, "Topology"):
            if not silent:
                print("ShapeGrammar.FigureByInputOutput - Error: The input topology parameter is not a valid topology. Returning None.")
            return None
        if not Topology.IsInstance(output, "Topology"):
            if not silent:
                print("ShapeGrammar.FigureByInputOutput - Error: The output topology parameter is not a valid topology. Returning None.")
            return None

        cluster = self.ClusterByInputOutput(input, output, silent=silent)
        data = Plotly.DataByTopology(cluster)
        fig = Plotly.FigureByData(data)
        return fig
    
    def FigureByRule(self, rule, silent: bool = False):
        """
        Returns the Plotly figure of the input rule.

        Parameters
        ----------
        rule : dict
            The input rule
        silent : bool, optional
            If True, suppresses error/warning messages. Default is False.
        
        Returns
        -------
        Plotly.Figure
            The create plotly figure
        """
        from topologicpy.Topology import Topology
        if not isinstance(rule, dict):
            if not silent:
                print("ShapeGrammar.DrawRule - Error: The input rule parameter is not a valid rule. Returning None.")
            return None
        input = rule["input"]
        output = self.ApplyRule(input, rule)
        return self.FigureByInputOutput(input, output, silent=silent)