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
import os
import warnings

try:
    import numpy as np
except:
    print("Cluster - Installing required numpy library.")
    try:
        os.system("pip install numpy")
    except:
        os.system("pip install numpy --user")
    try:
        import numpy as np
        print("Cluster - numpy library installed correctly.")
    except:
        warnings.warn("Cluster - Error: Could not import numpy.")

try:
    from scipy.spatial.distance import pdist, squareform
except:
    print("Cluster - Installing required scipy library.")
    try:
        os.system("pip install scipy")
    except:
        os.system("pip install scipy --user")
    try:
        from scipy.spatial.distance import pdist, squareform
        print("Cluster - scipy library installed correctly.")
    except:
        warnings.warn("Cluster - Error: Could not import scipy.")

class Cluster():
    @staticmethod
    def ByFormula(formula, xRange=None, yRange=None, xString="X", yString="Y"):
        """
        Creates a cluster of vertices by evaluating the input formula for a range of x values and, optionally, a range of y values.

        Parameters
        ----------
        formula : str
            A string representing the formula to be evaluated.
            For 2D formulas (i.e. Z = 0), use either 'X' (uppercase) or 'Y' (uppercase) for the independent variable.
            For 3D formulas, use 'X' and 'Y' (uppercase) for the independent variables. The Z value will be evaluated.
            For 3D formulas, both xRange and yRange MUST be specified.
            You can use standard math functions like 'sin', 'cos', 'tan', 'sqrt', etc.
            For example, 'X**2 + 2*X - sqrt(X)' or 'cos(abs(X)+abs(Y))'
        xRange : tuple , optional
            A tuple (start, end, step) representing the range of X values for which the formula should be evaluated.
            For example, to evaluate Y for X values from -5 to 5 with a step of 0.1, you should specify xRange=(-5, 5, 0.1).
            If the xRange is set to None or not specified:
                The method assumes that the formula uses the yString (e.g. 'Y' as in 'Y**2 + 2*Y - sqrt(Y)')
                The method will attempt to evaluate X based on the specified yRange.
                xRange and yRange CANNOT be None or unspecified at the same time. One or the other must be specified.
        yRange : tuple , optional
            A tuple (start, end, step) representing the range of Y values for which the formula should be evaluated.
            For example, to evaluate X for Y values from -5 to 5 with a step of 0.1, you should specify yRange=(-5,5,0.1).
            If the yRange is set to None or not specified:
                The method assumes that the formula uses the xString (e.g. 'X' as in 'X**2 + 2*X - sqrt(X)')
                The method will attempt to evaluate Y based on the specified xRange.
                xRange and yRange CANNOT be None or unspecified at the same time. One or the other must be specified.
        xString : str , optional
            The string used to represent the X independent variable. The default is 'X' (uppercase).
        yString : str , optional
            The string used to represent the Y independent variable. The default is 'Y' (uppercase).

        Returns:
            topologic_core.Cluster
                The created cluster of vertices.
        """
        from topologicpy.Vertex import Vertex
        import math
        if xRange == None and yRange == None:
            print("Cluster.ByFormula - Error: Both ranges cannot be None at the same time. Returning None.")
            return None
        if xString.islower():
            print("Cluster.ByFormula - Error: the input xString cannot lowercase. Please consider using uppercase (e.g. X). Returning None.")
            return None
        if yString == 'y':
            print("Cluster.ByFormula - Error: the input yString cannot be lowercase. Please consider using uppercase (e.g. Y). Returning None.")
            return None
        
        x_values = []
        y_values = []
        if not xRange == None:
            x_start, x_end, x_step = xRange
            x = x_start
            while x < x_end:
                x_values.append(x)
                x = x + x_step
            x_values.append(x_end)
        
        if not yRange == None:
            y_start, y_end, y_step = yRange
            y = y_start
            while y < y_end:
                y_values.append(y)
                y = y + y_step
            y_values.append(y_end)

        # Evaluate the formula for each x and y value
        x_return = []
        y_return = []
        z_return = []
        if len(x_values) > 0 and len(y_values) > 0: # Both X and Y exist, compute Z.
            for x in x_values:
                for y in y_values:
                    x_return.append(x)
                    y_return.append(y)
                    formula1 = formula.replace(xString, str(x)).replace(yString, str(y)).replace('sqrt', 'math.sqrt').replace('sin', 'math.sin').replace('cos', 'math.cos').replace('tan', 'math.tan').replace('radians', 'math.radians').replace('pi', 'math.pi')
                    z_return.append(eval(formula1))
        elif len(x_values) == 0 and len(y_values) > 0: # Only Y exists, compute X, Z is always 0.
            for y in y_values:
                y_return.append(y)
                formula1 = formula.replace(xString, str(y)).replace('sqrt', 'math.sqrt').replace('sin', 'math.sin').replace('cos', 'math.cos').replace('tan', 'math.tan').replace('radians', 'math.radians').replace('pi', 'math.pi')
                x_return.append(eval(formula1))
                z_return.append(0)
        else: # Only X exists, compute Y, Z is always 0.
            for x in x_values:
                x_return.append(x)
                formula1 = formula.replace(xString, str(x)).replace('sqrt', 'math.sqrt').replace('sin', 'math.sin').replace('cos', 'math.cos').replace('tan', 'math.tan').replace('radians', 'math.radians').replace('pi', 'math.pi')
                y_return.append(eval(formula1))
                z_return.append(0)
        vertices = []
        for i in range(len(x_return)):
            vertices.append(Vertex.ByCoordinates(x_return[i], y_return[i], z_return[i]))
        return Cluster.ByTopologies(vertices)
    
    @staticmethod
    def ByTopologies(*topologies, transferDictionaries: bool = False, silent=False):
        """
        Creates a topologic Cluster from the input list of topologies. The input can be individual topologies each as an input argument or a list of topologies stored in one input argument.

        Parameters
        ----------
        *topologies : topologic_core.Topology
            One or more instances of `topologic_core.Topology` to be processed.
        transferDictionaries : bool , optional
            If set to True, the dictionaries from the input topologies are merged and transferred to the cluster. Otherwise they are not. The default is False.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.
        
        Returns
        -------
        topologic_core.Cluster
            The created topologic Cluster.

        """
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology
        from topologicpy.Helper import Helper
        import inspect
        
        if len(topologies) == 0:
            if not silent:
                print("Cluster.ByTopologies - Error: The input topologies parameter is an empty list. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
            return None
        if len(topologies) == 1:
            topologies = topologies[0]
            if isinstance(topologies, list):
                if len(topologies) == 0:
                    if not silent:
                        print("Cluster.ByTopologies - Error: The input topologies parameter is an empty list. Returning None.")
                        curframe = inspect.currentframe()
                        calframe = inspect.getouterframes(curframe, 2)
                        print('caller name:', calframe[1][3])
                    return None
                else:
                    topologyList = [x for x in topologies if Topology.IsInstance(x, "Topology")]
                    if len(topologyList) == 0:
                        if not silent:
                            print("Cluster.ByTopologies - Error: The input topologies parameter does not contain any valid topologies. Returning None.")
                            curframe = inspect.currentframe()
                            calframe = inspect.getouterframes(curframe, 2)
                            print('caller name:', calframe[1][3])
                        return None
            else:
                if not silent:
                    print("Cluster.ByTopologies - Warning: The input topologies parameter contains only one topology. Returning the same topology.")
                    curframe = inspect.currentframe()
                    calframe = inspect.getouterframes(curframe, 2)
                    print('caller name:', calframe[1][3])
                return topologies
        else:
            topologyList = Helper.Flatten(list(topologies))
            topologyList = [x for x in topologyList if Topology.IsInstance(x, "Topology")]
        if len(topologyList) == 0:
            if not silent:
                print("Cluster.ByTopologies - Error: The input parameters do not contain any valid topologies. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
            return None
        cluster = topologic.Cluster.ByTopologies(topologyList, False) # Hook to Core
        dictionaries = []
        for t in topologyList:
            d = Topology.Dictionary(t)
            keys = Dictionary.Keys(d)
            if isinstance(keys, list):
                if len(keys) > 0:
                    dictionaries.append(d)
        if len(dictionaries) > 0:
            if len(dictionaries) > 1:
                d = Dictionary.ByMergedDictionaries(dictionaries, silent=silent)
            else:
                d = dictionaries[0]
                cluster = Topology.SetDictionary(cluster, d)
        return cluster

    @staticmethod
    def CellComplexes(cluster) -> list:
        """
        Returns the cellComplexes of the input cluster.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input cluster.

        Returns
        -------
        list
            The list of cellComplexes.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Cluster"):
            print("Cluster.CellComplexes - Error: The input cluster parameter is not a valid topologic cluster. Returning None.")
            return None
        cellComplexes = []
        _ = cluster.CellComplexes(None, cellComplexes) # Hook to Core
        return cellComplexes

    @staticmethod
    def Cells(cluster) -> list:
        """
        Returns the cells of the input cluster.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input cluster.

        Returns
        -------
        list
            The list of cells.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Cluster"):
            print("Cluster.Cells - Error: The input cluster parameter is not a valid topologic cluster. Returning None.")
            return None
        cells = []
        _ = cluster.Cells(None, cells) # Hook to Core
        return cells

    @staticmethod
    def DBSCAN(topologies, selectors=None, keys=["x", "y", "z"], epsilon: float = 0.5, minSamples: int = 2):
        """
        Clusters the input vertices based on the Density-Based Spatial Clustering of Applications with Noise (DBSCAN) method. See https://en.wikipedia.org/wiki/DBSCAN

        Parameters
        ----------
        topologies : list
            The input list of topologies to be clustered.
        selectors : list , optional
            If the list of topologies are not vertices then please provide a corresponding list of selectors (vertices) that represent the topologies for clustering. For example, these can be the centroids of the topologies.
            If set to None, the list of topologies is expected to be a list of vertices. The default is None.
        keys : list, optional
            The keys in the embedded dictionaries in the topologies. If specified, the values at these keys will be added to the dimensions to be clustered. The values must be numeric. If you wish the x, y, z location to be included,
            make sure the keys list includes "X", "Y", and/or "Z" (case insensitive). The default is ["x", "y", "z"]
        epsilon : float , optional
            The maximum radius around a data point within which other points are considered to be part of the same sense region (cluster). The default is 0.5. 
        minSamples : int , optional
            The minimum number of points required to form a dense region (cluster). The default is 2.

        Returns
        -------
        list, list
            The list of clusters and the list of vertices considered to be noise if any (otherwise returns None).

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        def dbscan_3d_indices(data, eps, min_samples):
            """
            DBSCAN clustering algorithm for 3D points.

            Parameters:
            - data: NumPy array, input data points with X, Y, and Z coordinates.
            - eps: float, maximum distance between two samples for one to be considered as in the neighborhood of the other.
            - min_samples: int, the number of samples (or total weight) in a neighborhood for a point to be considered as a core point.

            Returns:
            - clusters: List of lists, each list containing the indices of points in a cluster.
            - noise: List of indices, indices of points labeled as noise.
            """

            # Compute pairwise distances
            dists = squareform(pdist(data))

            # Initialize labels and cluster ID
            labels = np.full(data.shape[0], -1)
            cluster_id = 0

            # Iterate through each point
            for i in range(data.shape[0]):
                if labels[i] != -1:
                    continue  # Skip already processed points

                # Find neighbors within epsilon distance
                neighbors = np.where(dists[i] < eps)[0]

                if len(neighbors) < min_samples:
                    # Label as noise
                    labels[i] = -1
                else:
                    # Expand cluster
                    cluster_id += 1
                    expand_cluster_3d_indices(labels, dists, i, neighbors, cluster_id, eps, min_samples)

            # Organize indices into clusters and noise
            clusters = [list(np.where(labels == cid)[0]) for cid in range(1, cluster_id + 1)]
            noise = list(np.where(labels == -1)[0])

            return clusters, noise

        def expand_cluster_3d_indices(labels, dists, point_index, neighbors, cluster_id, eps, min_samples):
            """
            Expand the cluster around a core point for 3D points.

            Parameters:
            - labels: NumPy array, cluster labels for each data point.
            - dists: NumPy array, pairwise distances between data points.
            - point_index: int, index of the core point.
            - neighbors: NumPy array, indices of neighbors.
            - cluster_id: int, current cluster ID.
            - eps: float, maximum distance between two samples for one to be considered as in the neighborhood of the other.
            - min_samples: int, the number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
            """
            labels[point_index] = cluster_id

            i = 0
            while i < len(neighbors):
                current_neighbor = neighbors[i]

                if labels[current_neighbor] == -1:
                    labels[current_neighbor] = cluster_id

                    new_neighbors = np.where(dists[current_neighbor] < eps)[0]
                    if len(new_neighbors) >= min_samples:
                        neighbors = np.concatenate([neighbors, new_neighbors])

                elif labels[current_neighbor] == 0:
                    labels[current_neighbor] = cluster_id

                i += 1
        
        if not isinstance(topologies, list):
            print("Cluster.DBSCAN - Error: The input vertices parameter is not a valid list. Returning None.")
            return None, None
        topologyList = [t for t in topologies if Topology.IsInstance(t, "Topology")]
        if len(topologyList) < 1:
            print("Cluster.DBSCAN - Error: The input vertices parameter does not contain any valid vertices. Returning None.")
            return None, None
        if len(topologyList) < minSamples:
            print("Cluster.DBSCAN - Error: The input minSamples parameter cannot be larger than the number of vertices. Returning None.")
            return None, None
        
        if not isinstance(selectors, list):
            check_vertices = [t for t in topologyList if not Topology.IsInstance(t, "Vertex")]
            if len(check_vertices) > 0:
                print("Cluster.DBSCAN - Error: The input selectors parameter is not a valid list and this is needed since the list of topologies contains objects of type other than a topologic_core.Vertex. Returning None.")
                return None, None
        else:
            selectors = [s for s in selectors if Topology.IsInstance(s, "Vertex")]
            if len(selectors) < 1:
                check_vertices = [t for t in topologyList if not Topology.IsInstance(t, "Vertex")]
                if len(check_vertices) > 0:
                    print("Cluster.DBSCAN - Error: The input selectors parameter does not contain any valid vertices and this is needed since the list of topologies contains objects of type other than a topologic_core.Vertex. Returning None.")
                    return None, None
            if not len(selectors) == len(topologyList):
                print("Cluster.DBSCAN - Error: The input topologies and selectors parameters do not have the same length. Returning None.")
                return None, None
        if not isinstance(keys, list):
            print("Cluster.DBSCAN - Error: The input keys parameter is not a valid list. Returning None.")
            return None
        

        data = []
        if selectors == None:
            for t in topologyList:
                elements = []
                if keys:
                    d = Topology.Dictionary(t)
                    for key in keys:
                        if key.lower() == "x":
                            value = Vertex.X(t)
                        elif key.lower() == "y":
                            value = Vertex.Y(t)
                        elif key.lower() == "z":
                            value = Vertex.Z(t)
                        else:
                            value = Dictionary.ValueAtKey(d, key)
                        if value != None:
                            elements.append(value)
                data.append(elements)
        else:
            for i, s in enumerate(selectors):
                elements = []
                if keys:
                    d = Topology.Dictionary(topologyList[i])
                    for key in keys:
                        if key.lower() == "x":
                            value = Vertex.X(s)
                        elif key.lower() == "y":
                            value = Vertex.Y(s)
                        elif key.lower() == "z":
                            value = Vertex.Z(s)
                        else:
                            value = Dictionary.ValueAtKey(d, key)
                        if value != None:
                            elements.append(value)
                data.append(elements)
        #coords = [[Vertex.X(v), Vertex.Y(v), Vertex.Z(v)] for v in vertexList]
        clusters, noise = dbscan_3d_indices(np.array(data), epsilon, minSamples)
        tp_clusters = []
        for cluster in clusters:
            tp_clusters.append(Cluster.ByTopologies([topologyList[i] for i in cluster]))
        vert_group = []
        tp_noise = None
        if len(noise) > 0:
            tp_noise = Cluster.ByTopologies([topologyList[i] for i in noise])
        return tp_clusters, tp_noise

    @staticmethod
    def Edges(cluster) -> list:
        """
        Returns the edges of the input cluster.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input cluster.

        Returns
        -------
        list
            The list of edges.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Cluster"):
            print("Cluster.Edges - Error: The input cluster parameter is not a valid topologic cluster. Returning None.")
            return None
        edges = []
        _ = cluster.Edges(None, edges) # Hook to Core
        return edges

    @staticmethod
    def Faces(cluster) -> list:
        """
        Returns the faces of the input cluster.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input cluster.

        Returns
        -------
        list
            The list of faces.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Cluster"):
            print("Cluster.Faces - Error: The input cluster parameter is not a valid topologic cluster. Returning None.")
            return None
        faces = []
        _ = cluster.Faces(None, faces) # Hook to Core
        return faces

    @staticmethod
    def FreeCells(cluster, tolerance: float = 0.0001) -> list:
        """
        Returns the free cells of the input cluster that are not part of a higher topology.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input cluster.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        list
            The list of free cells.

        """
        from topologicpy.CellComplex import CellComplex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Cluster"):
            print("Cluster.FreeCells - Error: The input cluster parameter is not a valid topologic cluster. Returning None.")
            return None
        allCells = Cluster.Cells(cluster)
        if len(allCells) < 1:
            return []
        allCellsCluster = Cluster.ByTopologies(allCells)
        freeCells = []
        cellComplexes = Cluster.CellComplexes(cluster)
        cellComplexesCells = []
        for cellComplex in cellComplexes:
            tempCells = CellComplex.Cells(cellComplex)
            cellComplexesCells += tempCells
        if len(cellComplexesCells) == 0:
            return allCells
        cellComplexesCluster = Cluster.ByTopologies(cellComplexesCells)
        resultingCluster = Topology.Boolean(allCellsCluster, cellComplexesCluster, operation="difference", tolerance=tolerance)
        if resultingCluster == None:
            return []
        if Topology.IsInstance(resultingCluster, "Cell"):
            return [resultingCluster]
        result = Topology.SubTopologies(resultingCluster, subTopologyType="cell")
        if result == None:
            return [] #Make sure you return an empty list instead of None
        return result
    
    @staticmethod
    def FreeShells(cluster, tolerance: float = 0.0001) -> list:
        """
        Returns the free shells of the input cluster that are not part of a higher topology.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input cluster.
        tolerance : float, optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        list
            The list of free shells.

        """
        from topologicpy.Cell import Cell
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Cluster"):
            print("Cluster.FreeShells - Error: The input cluster parameter is not a valid topologic cluster. Returning None.")
            return None
        allShells = Cluster.Shells(cluster)
        if len(allShells) < 1:
            return []
        allShellsCluster = Cluster.ByTopologies(allShells)
        cells = Cluster.Cells(cluster)
        cellsShells = []
        for cell in cells:
            tempShells = Cell.Shells(cell)
            cellsShells += tempShells
        if len(cellsShells) == 0:
            return allShells
        cellsCluster = Cluster.ByTopologies(cellsShells)
        resultingCluster = Topology.Boolean(allShellsCluster, cellsCluster, operation="difference", tolerance=tolerance)
        if resultingCluster == None:
            return []
        if Topology.IsInstance(resultingCluster, "Shell"):
            return [resultingCluster]
        result = Topology.SubTopologies(resultingCluster, subTopologyType="shell")
        if result == None:
            return [] #Make sure you return an empty list instead of None
        return result
    
    @staticmethod
    def FreeFaces(cluster, tolerance: float = 0.0001) -> list:
        """
        Returns the free faces of the input cluster that are not part of a higher topology.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input cluster.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        list
            The list of free faces.

        """
        from topologicpy.Shell import Shell
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Cluster"):
            print("Cluster.FreeFaces - Error: The input cluster parameter is not a valid topologic cluster. Returning None.")
            return None
        allFaces = Cluster.Faces(cluster)
        if len(allFaces) < 1:
            return []
        allFacesCluster = Cluster.ByTopologies(allFaces)
        shells = Cluster.Shells(cluster)
        shellFaces = []
        for shell in shells:
            tempFaces = Shell.Faces(shell)
            shellFaces += tempFaces
        if len(shellFaces) == 0:
            return allFaces
        shellCluster = Cluster.ByTopologies(shellFaces)
        resultingCluster = Topology.Boolean(allFacesCluster, shellCluster, operation="difference", tolerance=tolerance)
        if resultingCluster == None:
            return []
        if Topology.IsInstance(resultingCluster, "Face"):
            return [resultingCluster]
        result = Topology.SubTopologies(resultingCluster, subTopologyType="face")
        if result == None:
            return [] #Make sure you return an empty list instead of None
        return result

    @staticmethod
    def FreeWires(cluster, tolerance: float = 0.0001) -> list:
        """
        Returns the free wires of the input cluster that are not part of a higher topology.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input cluster.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        list
            The list of free wires.

        """
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Cluster"):
            print("Cluster.FreeWires - Error: The input cluster parameter is not a valid topologic cluster. Returning None.")
            return None
        allWires = Cluster.Wires(cluster)
        if len(allWires) < 1:
            return []
        allWiresCluster = Cluster.ByTopologies(allWires)
        faces = Cluster.Faces(cluster)
        facesWires = []
        for face in faces:
            tempWires = Face.Wires(face)
            facesWires += tempWires
        if len(facesWires) == 0:
            return allWires
        facesCluster = Cluster.ByTopologies(facesWires)
        resultingCluster = Topology.Boolean(allWiresCluster, facesCluster, operation="difference", tolerance=tolerance)
        if resultingCluster == None:
            return []
        if Topology.IsInstance(resultingCluster, "Wire"):
            return [resultingCluster]
        result = Topology.SubTopologies(resultingCluster, subTopologyType="wire")
        if not result:
            return [] #Make sure you return an empty list instead of None
        return result
    
    @staticmethod
    def FreeEdges(cluster, tolerance: float = 0.0001) -> list:
        """
        Returns the free edges of the input cluster that are not part of a higher topology.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input cluster.
        tolerance : float, optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        list
            The list of free edges.

        """
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Cluster"):
            print("Cluster.FreeEdges - Error: The input cluster parameter is not a valid topologic cluster. Returning None.")
            return None
        allEdges = Cluster.Edges(cluster)
        if len(allEdges) < 1:
            return []
        allEdgesCluster = Cluster.ByTopologies(allEdges)
        wires = Cluster.Wires(cluster)
        wireEdges = []
        for wire in wires:
            tempEdges = Wire.Edges(wire)
            wireEdges += tempEdges
        if len(wireEdges) == 0:
            return allEdges
        wireCluster = Cluster.ByTopologies(wireEdges)
        resultingCluster = Topology.Boolean(allEdgesCluster, wireCluster, operation="difference", tolerance=tolerance)
        if resultingCluster == None:
            return []
        if Topology.IsInstance(resultingCluster, "Edge"):
            return [resultingCluster]
        result = Topology.SubTopologies(resultingCluster, subTopologyType="edge")
        if result == None:
            return [] #Make sure you return an empty list instead of None
        return result
    
    @staticmethod
    def FreeVertices(cluster, tolerance: float = 0.0001) -> list:
        """
        Returns the free vertices of the input cluster that are not part of a higher topology.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input cluster.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        list
            The list of free vertices.

        """
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Cluster"):
            print("Cluster.FreeVertices - Error: The input cluster parameter is not a valid topologic cluster. Returning None.")
            return None
        allVertices = Topology.Vertices(cluster)
        if len(allVertices) < 1:
            return []
        allVerticesCluster = Cluster.ByTopologies(allVertices)
        edges = Topology.Edges(cluster)
        edgesVertices = []
        for edge in edges:
            tempVertices = Topology.Vertices(edge)
            edgesVertices += tempVertices
        if len(edgesVertices) == 0:
            return allVertices
        edgesCluster = Cluster.ByTopologies(edgesVertices)
        resultingCluster = Topology.Boolean(allVerticesCluster, edgesCluster, operation="difference", tolerance=tolerance)
        if Topology.IsInstance(resultingCluster, "Vertex"):
            return [resultingCluster]
        if resultingCluster == None:
            return []
        result = Topology.SubTopologies(resultingCluster, subTopologyType="vertex")
        if result == None:
            return [] #Make sure you return an empty list instead of None
        return result
    
    @staticmethod
    def FreeTopologies(cluster, tolerance: float = 0.0001) -> list:
        """
        Returns the free topologies of the input cluster that are not part of a higher topology.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input cluster.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        list
            The list of free topologies.

        """
        topologies = Cluster.FreeVertices(cluster, tolerance=tolerance)
        topologies += Cluster.FreeEdges(cluster, tolerance=tolerance)
        topologies += Cluster.FreeWires(cluster, tolerance=tolerance)
        topologies += Cluster.FreeFaces(cluster, tolerance=tolerance)
        topologies += Cluster.FreeShells(cluster, tolerance=tolerance)
        topologies += Cluster.FreeCells(cluster, tolerance=tolerance)
        topologies += Cluster.CellComplexes(cluster)

        return topologies
    
    @staticmethod
    def HighestType(cluster) -> int:
        """
        Returns the type of the highest dimension subtopology found in the input cluster.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input cluster.

        Returns
        -------
        int
            The type of the highest dimension subtopology found in the input cluster.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Cluster"):
            print("Cluster.HighestType - Error: The input cluster parameter is not a valid topologic cluster. Returning None.")
            return None
        cellComplexes = Topology.CellComplexes(cluster)
        if len(cellComplexes) > 0:
            return Topology.TypeID("CellComplex")
        cells = Topology.Cells(cluster)
        if len(cells) > 0:
            return Topology.TypeID("Cell")
        shells = Topology.Shells(cluster)
        if len(shells) > 0:
            return Topology.TypeID("Shell")
        faces = Topology.Faces(cluster)
        if len(faces) > 0:
            return Topology.TypeID("Face")
        wires = Cluster.Wires(cluster)
        if len(wires) > 0:
            return Topology.TypeID("Wire")
        edges = Topology.Edges(cluster)
        if len(edges) > 0:
            return Topology.TypeID("Edge")
        vertices = Topology.Vertices(cluster)
        if len(vertices) > 0:
            return Topology.TypeID("Vertex")
    
    @staticmethod
    def K_Means(topologies, selectors=None, keys=["x", "y", "z"], k=4, maxIterations=100, centroidKey="k_centroid"):
        """
        Clusters the input topologies using K-Means clustering. See https://en.wikipedia.org/wiki/K-means_clustering

        Parameters
        ----------
        topologies : list
            The input list of topologies. If this is not a list of topologic vertices then please provide a list of selectors
        selectors : list , optional
            If the list of topologies are not vertices then please provide a corresponding list of selectors (vertices) that represent the topologies for clustering. For example, these can be the centroids of the topologies.
            If set to None, the list of topologies is expected to be a list of vertices. The default is None.
        keys : list, optional
            The keys in the embedded dictionaries in the topologies. If specified, the values at these keys will be added to the dimensions to be clustered. The values must be numeric. If you wish the x, y, z location to be included,
            make sure the keys list includes "X", "Y", and/or "Z" (case insensitive). The default is ["x", "y", "z"]
        k : int , optional
            The desired number of clusters. The default is 4.
        maxIterations : int , optional
            The desired maximum number of iterations for the clustering algorithm
        centroidKey : str , optional
            The desired dictionary key under which to store the cluster's centroid (this is not to be confused with the actual geometric centroid of the cluster). The default is "k_centroid"

        Returns
        -------
        list
            The created list of clusters.

        """
        from topologicpy.Helper import Helper
        from topologicpy.Vertex import Vertex
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology

        def k_means(data, vertices, k=4, maxIterations=100):
            import random
            def euclidean_distance(p, q):
                return sum((pi - qi) ** 2 for pi, qi in zip(p, q)) ** 0.5

            # Initialize k centroids randomly
            centroids = random.sample(data, k)

            for _ in range(maxIterations):
                # Assign each data point to the nearest centroid
                clusters = [[] for _ in range(k)]
                clusters_v = [[] for _ in range(k)]
                for i, point in enumerate(data):
                    distances = [euclidean_distance(point, centroid) for centroid in centroids]
                    nearest_centroid_index = distances.index(min(distances))
                    clusters[nearest_centroid_index].append(point)
                    clusters_v[nearest_centroid_index].append(vertices[i])

                # Compute the new centroids as the mean of the points in each cluster
                new_centroids = []
                for cluster in clusters:
                    if not cluster:
                        # If a cluster is empty, keep the previous centroid
                        new_centroids.append(centroids[clusters.index(cluster)])
                    else:
                        new_centroids.append([sum(dim) / len(cluster) for dim in zip(*cluster)])

                # Check if the centroids have converged
                if new_centroids == centroids:
                    break

                centroids = new_centroids

            return {'clusters': clusters, 'clusters_v': clusters_v, 'centroids': centroids}

        if not isinstance(topologies, list):
            print("Cluster.K_Means - Error: The input topologies parameter is not a valid list. Returning None.")
            return None
        topologies = [t for t in topologies if Topology.IsInstance(t, "Topology")]
        if len(topologies) < 1:
            print("Cluster.K_Means - Error: The input topologies parameter does not contain any valid topologies. Returning None.")
            return None
        if not isinstance(selectors, list):
            check_vertices = [v for v in topologies if not Topology.IsInstance(v, "Vertex")]
            if len(check_vertices) > 0:
                print("Cluster.K_Means - Error: The input selectors parameter is not a valid list and this is needed since the list of topologies contains objects of type other than a topologic_core.Vertex. Returning None.")
                return None
        else:
            selectors = [s for s in selectors if Topology.IsInstance(s, "Vertex")]
            if len(selectors) < 1:
                check_vertices = [v for v in topologies if not Topology.IsInstance(v, "Vertex")]
                if len(check_vertices) > 0:
                    print("Cluster.K_Means - Error: The input selectors parameter does not contain any valid vertices and this is needed since the list of topologies contains objects of type other than a topologic_core.Vertex. Returning None.")
                    return None
            if not len(selectors) == len(topologies):
                print("Cluster.K_Means - Error: The input topologies and selectors parameters do not have the same length. Returning None.")
                return None
        if not isinstance(keys, list):
            print("Cluster.K_Means - Error: The input keys parameter is not a valid list. Returning None.")
            return None
        if not isinstance(k , int):
            print("Cluster.K_Means - Error: The input k parameter is not a valid integer. Returning None.")
            return None
        if k < 1:
            print("Cluster.K_Means - Error: The input k parameter is less than one. Returning None.")
            return None
        if len(topologies) < k:
            print("Cluster.K_Means - Error: The input topologies parameter is less than the specified number of clusters. Returning None.")
            return None
        if len(topologies) == k:
            t_clusters = []
            for topology in topologies:
                t_cluster = Cluster.ByTopologies([topology])
                for key in keys:
                        if key.lower() == "x":
                            value = Vertex.X(t)
                        elif key.lower() == "y":
                            value = Vertex.Y(t)
                        elif key.lower() == "z":
                            value = Vertex.Z(t)
                        else:
                            value = Dictionary.ValueAtKey(d, key)
                        if value != None:
                            elements.append(value)
                d = Dictionary.ByKeysValues([centroidKey], [elements])
                t_cluster = Topology.SetDictionary(t_cluster, d)
                t_clusters.append(t_cluster)
            return t_clusters
        
        data = []
        if selectors == None:
            for t in topologies:
                elements = []
                if keys:
                    d = Topology.Dictionary(t)
                    for key in keys:
                        if key.lower() == "x":
                            value = Vertex.X(t)
                        elif key.lower() == "y":
                            value = Vertex.Y(t)
                        elif key.lower() == "z":
                            value = Vertex.Z(t)
                        else:
                            value = Dictionary.ValueAtKey(d, key)
                        if value != None:
                            elements.append(value)
                data.append(elements)
        else:
            for i, s in enumerate(selectors):
                elements = []
                if keys:
                    d = Topology.Dictionary(topologies[i])
                    for key in keys:
                        if key.lower() == "x":
                            value = Vertex.X(s)
                        elif key.lower() == "y":
                            value = Vertex.Y(s)
                        elif key.lower() == "z":
                            value = Vertex.Z(s)
                        else:
                            value = Dictionary.ValueAtKey(d, key)
                        if value != None:
                            elements.append(value)
                data.append(elements)
        if len(data) == 0:
            print("Cluster.K_Means - Error: Could not perform the operation. Returning None.")
            return None
        if selectors:
            dict = k_means(data, selectors, k=k, maxIterations=maxIterations)
        else:
            dict = k_means(data, topologies, k=k, maxIterations=maxIterations)
        clusters = dict['clusters_v']
        centroids = dict['centroids']
        t_clusters = []
        for i, cluster in enumerate(clusters):
            cluster_vertices = []
            for v in cluster:
                if selectors == None:
                    cluster_vertices.append(v)
                else:
                    index = selectors.index(v)
                    cluster_vertices.append(topologies[index])
            cluster = Cluster.ByTopologies(cluster_vertices)
            d = Dictionary.ByKeysValues([centroidKey], [centroids[i]])
            cluster = Topology.SetDictionary(cluster, d)
            t_clusters.append(cluster)
        return t_clusters

    @staticmethod
    def MergeCells(cells, tolerance=0.0001):
        """
        Creates a cluster that contains cellComplexes where it can create them plus any additional free cells.

        Parameters
        ----------
        cells : list
            The input list of cells.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Cluster
            The created cluster with merged cells as possible.

        """

        from topologicpy.CellComplex import CellComplex
        from topologicpy.Topology import Topology

        def find_cell_complexes(cells, adjacency_test, tolerance=0.0001):
            cell_complexes = []
            remaining_cells = set(cells)

            def explore_complex(cell_complex, remaining, tolerance=0.0001):
                new_cells = set()
                for cell in remaining:
                    if any(adjacency_test(cell, existing_cell, tolerance=tolerance) for existing_cell in cell_complex):
                        new_cells.add(cell)
                return new_cells

            while remaining_cells:
                current_cell = remaining_cells.pop()
                current_complex = {current_cell}
                current_complex.update(explore_complex(current_complex, remaining_cells, tolerance=tolerance))
                cell_complexes.append(current_complex)
                remaining_cells -= current_complex

            return cell_complexes

        # Example adjacency test function (replace this with your actual implementation)
        def adjacency_test(cell1, cell2, tolerance=0.0001):
            return Topology.IsInstance(Topology.Merge(cell1, cell2, tolerance=tolerance), "CellComplex")

        if not isinstance(cells, list):
            print("Cluster.MergeCells - Error: The input cells parameter is not a valid list of cells. Returning None.")
            return None
        #cells = [cell for cell in cells if Topology.IsInstance(cell, "Cell")]
        if len(cells) < 1:
            print("Cluster.MergeCells - Error: The input cells parameter does not contain any valid cells. Returning None.")
            return None
        
        complexes = find_cell_complexes(cells, adjacency_test)
        cellComplexes = []
        cells = []
        for aComplex in complexes:
            aComplex = list(aComplex)
            if len(aComplex) > 1:
                cc = CellComplex.ByCells(aComplex, silent=True)
                if Topology.IsInstance(cc, "CellComplex"):
                    cellComplexes.append(cc)
            elif len(aComplex) == 1:
                if Topology.IsInstance(aComplex[0], "Cell"):
                    cells.append(aComplex[0])
        return Cluster.ByTopologies(cellComplexes+cells)
    
    @staticmethod
    def MysticRose(wire= None, origin= None, radius: float = 0.5, sides: int = 16, perimeter: bool = True, direction: list = [0, 0, 1], placement:str = "center", tolerance: float = 0.0001):
        """
        Creates a mystic rose.

        Parameters
        ----------
        wire : topologic_core.Wire , optional
            The input Wire. if set to None, a circle with the input parameters is created. Otherwise, the input parameters are ignored.
        origin : topologic_core.Vertex , optional
            The location of the origin of the circle. The default is None which results in the circle being placed at (0, 0, 0).
        radius : float , optional
            The radius of the mystic rose. The default is 1.
        sides : int , optional
            The number of sides of the mystic rose. The default is 16.
        perimeter : bool , optional
            If True, the perimeter edges are included in the output. The default is True.
        direction : list , optional
            The vector representing the up direction of the mystic rose. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the mystic rose. This can be "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Cluster
            The created mystic rose (cluster of edges).

        """
        import topologicpy
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from itertools import combinations

        if wire == None:
            wire = Wire.Circle(origin=origin, radius=radius, sides=sides, fromAngle=0, toAngle=360, close=True, direction=direction, placement=placement, tolerance=tolerance)
        if not Wire.IsClosed(wire):
            print("Cluster.MysticRose - Error: The input wire parameter is not a closed topologic wire. Returning None.")
            return None
        vertices = Topology.Vertices(wire)
        indices = list(range(len(vertices)))
        combs = [[comb[0],comb[1]] for comb in combinations(indices, 2) if not (abs(comb[0]-comb[1]) == 1) and not (abs(comb[0]-comb[1]) == len(indices)-1)]
        edges = []
        if perimeter:
            edges = Wire.Edges(wire)
        for comb in combs:
            edges.append(Edge.ByVertices([vertices[comb[0]], vertices[comb[1]]], tolerance=tolerance))
        return Cluster.ByTopologies(edges)
    
    @staticmethod
    def Shells(cluster) -> list:
        """
        Returns the shells of the input cluster.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input cluster.

        Returns
        -------
        list
            The list of shells.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Cluster"):
            print("Cluster.Shells - Error: The input cluster parameter is not a valid topologic cluster. Returning None.")
            return None
        shells = []
        _ = cluster.Shells(None, shells) # Hook to Core
        return shells

    @staticmethod
    def Simplify(cluster):
        """
        Simplifies the input cluster if possible. For example, if the cluster contains only one cell, that cell is returned.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input cluster.

        Returns
        -------
        topologic_core.Topology or list
            The simplification of the cluster.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Cluster"):
            print("Cluster.Simplify - Error: The input cluster parameter is not a valid topologic cluster. Returning None.")
            return None
        resultingTopologies = []
        topCC = Topology.CellComplexes(cluster)
        topCells = Topology.Cells(cluster)
        topShells = Topology.Shells(cluster)
        topFaces = Topology.Faces(cluster)
        topWires = Topology.Wires(cluster)
        topEdges = Topology.Edges(cluster)
        topVertices = Topology.Vertices(cluster)
        if len(topCC) == 1:
            cc = topCC[0]
            ccVertices = Topology.Vertices(cc)
            if len(topVertices) == len(ccVertices):
                resultingTopologies.append(cc)
        if len(topCC) == 0 and len(topCells) == 1:
            cell = topCells[0]
            ccVertices = Topology.Vertices(cell)
            if len(topVertices) == len(ccVertices):
                resultingTopologies.append(cell)
        if len(topCC) == 0 and len(topCells) == 0 and len(topShells) == 1:
            shell = topShells[0]
            ccVertices = Topology.Vertices(shell)
            if len(topVertices) == len(ccVertices):
                resultingTopologies.append(shell)
        if len(topCC) == 0 and len(topCells) == 0 and len(topShells) == 0 and len(topFaces) == 1:
            face = topFaces[0]
            ccVertices = Topology.Vertices(face)
            if len(topVertices) == len(ccVertices):
                resultingTopologies.append(face)
        if len(topCC) == 0 and len(topCells) == 0 and len(topShells) == 0 and len(topFaces) == 0 and len(topWires) == 1:
            wire = topWires[0]
            ccVertices = Topology.Vertices(wire)
            if len(topVertices) == len(ccVertices):
                resultingTopologies.append(wire)
        if len(topCC) == 0 and len(topCells) == 0 and len(topShells) == 0 and len(topFaces) == 0 and len(topWires) == 0 and len(topEdges) == 1:
            edge = topEdges[0]
            ccVertices = Topology.Vertices(edge)
            if len(topVertices) == len(ccVertices):
                resultingTopologies.append(edge)
        if len(topCC) == 0 and len(topCells) == 0 and len(topShells) == 0 and len(topFaces) == 0 and len(topWires) == 0 and len(topEdges) == 0 and len(topVertices) == 1:
            vertex = topVertices[0]
            resultingTopologies.append(vertex)
        if len(resultingTopologies) == 1:
            return resultingTopologies[0]
        return cluster

    @staticmethod
    def Tripod(size: float = 1.0,
               radius: float = 0.03,
               sides: int = 4,
               faceColorKey="faceColor",
               xColor = "red",
               yColor = "green",
               zColor = "blue",
               matrix=None):
        """
        Creates a color-coded Axes tripod for X, Y, and Z axes. X-Axis is red, Y-Axis is "green", and Z-Axis= "blue"

        Parameters
        ----------
        size : float , optional
            The desired size of the tripod. The default is 1.0.
        radius : float , optional
            The desired radiues of the tripod. The default is 0.03
        sides : int , optional
            The desired number of sides of the tripod. The default is 4.
        faceColorKey : str , optional
            The dictionary key under which to store the colors of the axes.
        xColor : str , optional
            The color to use for the X axis. The default is "red".
        yColor : str , optional
            The color to use for the Y axis. The default is "green".
        zColor : str , optional
            The color to use for the Z axis. The default is "blue".
        matrix : list , optional
            The desired 4X4 transformation matrix to use for transforming the tripod. The default is None which means the tripod will be placed at the origin and will be axis-aligned. 

        Returns
        -------
        topologic_core.Cluster
            The created tripod

        """

        from topologicpy.Cell import Cell
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        cyl = Cell.Cylinder(radius=radius, height=size - size*0.3, uSides=sides, placement="bottom")
        cone = Cell.Cone(baseRadius=radius*2.25, height=size*0.3, placement="bottom", uSides=sides)
        cone = Topology.Translate(cone, 0, 0, size - size*0.3)
        z_arrow = Topology.Union(cyl, cone)
        x_arrow = Topology.Rotate(z_arrow, axis=[0,1,0], angle=90)
        y_arrow = Topology.Rotate(z_arrow, axis=[1,0,0], angle=-90)

        x_faces = Topology.Faces(x_arrow)
        for x_face in x_faces:
            d = Dictionary.ByKeyValue(faceColorKey, xColor)
            x_face = Topology.SetDictionary(x_face, d)
        y_faces = Topology.Faces(y_arrow)
        for y_face in y_faces:
            d = Dictionary.ByKeyValue(faceColorKey, yColor)
            y_face = Topology.SetDictionary(y_face, d)
        z_faces = Topology.Faces(z_arrow)
        for z_face in z_faces:
            d = Dictionary.ByKeyValue(faceColorKey, zColor)
            z_face = Topology.SetDictionary(z_face, d)

        cluster = Cluster.ByTopologies(x_arrow, y_arrow, z_arrow)
        if not matrix == None:
            cluster = Topology.Transform(cluster, matrix=matrix)
        return cluster

    @staticmethod
    def Vertices(cluster) -> list:
        """
        Returns the vertices of the input cluster.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input cluster.

        Returns
        -------
        list
            The list of vertices.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Cluster"):
            print("Cluster.Vertices - Error: The input cluster parameter is not a valid topologic cluster. Returning None.")
            return None
        vertices = []
        _ = cluster.Vertices(None, vertices) # Hook to Core
        return vertices

    @staticmethod
    def Wires(cluster) -> list:
        """
        Returns the wires of the input cluster.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input cluster.

        Returns
        -------
        list
            The list of wires.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Cluster"):
            print("Cluster.Wires - Error: The input cluster parameter is not a valid topologic cluster. Returning None.")
            return None
        wires = []
        _ = cluster.Wires(None, wires) # Hook to Core
        return wires

    