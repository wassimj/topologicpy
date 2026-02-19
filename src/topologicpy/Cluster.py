# Copyright (C) 2026
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
            . The method assumes that the formula uses the yString (e.g. 'Y' as in 'Y**2 + 2*Y - sqrt(Y)')
            . The method will attempt to evaluate X based on the specified yRange.
            . xRange and yRange CANNOT be None or unspecified at the same time. One or the other must be specified.
        yRange : tuple , optional
            A tuple (start, end, step) representing the range of Y values for which the formula should be evaluated.
            For example, to evaluate X for Y values from -5 to 5 with a step of 0.1, you should specify yRange=(-5,5,0.1).
            If the yRange is set to None or not specified:
            . The method assumes that the formula uses the xString (e.g. 'X' as in 'X**2 + 2*X - sqrt(X)')
            . The method will attempt to evaluate Y based on the specified xRange.
            . xRange and yRange CANNOT be None or unspecified at the same time. One or the other must be specified.
        xString : str , optional
            The string used to represent the X independent variable. Default is 'X' (uppercase).
        yString : str , optional
            The string used to represent the Y independent variable. Default is 'Y' (uppercase).

        Returns
        -------
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
            If set to True, the dictionaries from the input topologies are merged and transferred to the cluster. Otherwise they are not. Default is False.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
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
                print("Topologies:", topologies)
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
                print('caller name:', calframe[1][2])
            return None
        if len(topologies) == 1:
            topologies = topologies[0]
            if isinstance(topologies, list):
                if len(topologies) == 0:
                    if not silent:
                        print("Cluster.ByTopologies - Error: The input topologies parameter is an empty list. Returning None.")
                        print("Topologies:", topologies)
                        curframe = inspect.currentframe()
                        calframe = inspect.getouterframes(curframe, 2)
                        print('caller name:', calframe[1][3])
                        print('caller name:', calframe[1][2])
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
    def Topologies(cluster, tolerance: float = 0.0001, silent: bool = False) -> list:
        """
        Returns the topologies of the input cluster.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input cluster.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The list of cellComplexes.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Cluster"):
            if not silent:
                print("Cluster.Topologies - Error: The input cluster parameter is not a valid topologic cluster. Returning None.")
            return None
        topologies = []
        topologies.extend(Cluster.CellComplexes(cluster, silent=silent))
        topologies.extend(Cluster.FreeCells(cluster, tolerance=tolerance, silent=silent))
        topologies.extend(Cluster.FreeShells(cluster, tolerance=tolerance, silent=silent))
        topologies.extend(Cluster.FreeFaces(cluster, tolerance=tolerance, silent=silent))
        topologies.extend(Cluster.FreeWires(cluster, tolerance=tolerance, silent=silent))
        topologies.extend(Cluster.FreeEdges(cluster, tolerance=tolerance, silent=silent))
        topologies.extend(Cluster.FreeVertices(cluster, tolerance=tolerance, silent=silent))
        return topologies
        
    @staticmethod
    def CellComplexes(cluster, silent: bool = False) -> list:
        """
        Returns the cellComplexes of the input cluster.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input cluster.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The list of cellComplexes.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Cluster"):
            if not silent:
                print("Cluster.CellComplexes - Error: The input cluster parameter is not a valid topologic cluster. Returning None.")
            return None
        cellComplexes = []
        _ = cluster.CellComplexes(None, cellComplexes) # Hook to Core
        return cellComplexes

    @staticmethod
    def Cells(cluster, silent: bool = False) -> list:
        """
        Returns the cells of the input cluster.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input cluster.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The list of cells.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Cluster"):
            if not silent:
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
            If set to None, the list of topologies is expected to be a list of vertices. Default is None.
        keys : list, optional
            The keys in the embedded dictionaries in the topologies. If specified, the values at these keys will be added to the dimensions to be clustered. The values must be numeric. If you wish the x, y, z location to be included,
            make sure the keys list includes "X", "Y", and/or "Z" (case insensitive). Default is ["x", "y", "z"]
        epsilon : float , optional
            The maximum radius around a data point within which other points are considered to be part of the same sense region (cluster). Default is 0.5. 
        minSamples : int , optional
            The minimum number of points required to form a dense region (cluster). Default is 2.

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
    def ExternalBoundary(cluster, silent: bool = False):
        """
        Returns the external boundary of the input cluster.

        Parameters
        ----------
        cluster : topologic_core.Clusterx
            The input cluster.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Cluster
            The external boundary of the input cluster.

        """
        from topologicpy.Face import Face
        from topologicpy.Cell import Cell
        from topologicpy.CellComplex import CellComplex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Cluster"):
            if not silent:
                print("Cluster.ExternalBoundary - Error: The input cellComplex parameter is not a valid cellComplex. Returning None.")
            return None
        
        cellComplexes = Cluster.CellComplexes(cluster)
        cells = Cluster.FreeCells(cluster)
        shells = Cluster.FreeShells(cluster)
        faces = Cluster.FreeFaces(cluster)
        wires = Cluster.FreeWires(cluster)
        edges = Cluster.FreeEdges(cluster)
        vertices = Cluster.FreeVertices(cluster)

        eb_list = []
        for cc in cellComplexes:
            eb_list.append(CellComplex.ExternalBoundary(cc))
        for c in cells:
            eb = Cell.ExternalBoundary(c)
            c2 = Cell.ByShell(eb)
            if Topology.IsInstance(c2, "cell"):
                eb_list.append(c2)
        for f in faces:
            eb = Face.ExternalBoundary(f)
            ibList = Face.InternalBoundaries(f)
            f2 = Face.ByWires(eb, ibList)
            if Topology.IsInstance(f2, "face"):
                eb_list.append(Face.ExternalBoundary(f2))
        eb_list.extend(shells)
        eb_list.extend(wires)
        eb_list.extend(edges)
        eb_list.extend(vertices)
        if len(eb_list) > 0:
            return Cluster.ByTopologies(eb_list)
        return cluster

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
    def FreeCells(cluster, tolerance: float = 0.0001, silent: bool = False) -> list:
        """
        Returns the free cells of the input cluster that are not part of a higher topology.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input cluster.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The list of free cells.

        """
        from topologicpy.CellComplex import CellComplex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Cluster"):
            if not silent:
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
        resultingCluster = Topology.Difference(allCellsCluster, cellComplexesCluster, tolerance=tolerance)
        if resultingCluster == None:
            return []
        if Topology.IsInstance(resultingCluster, "Cell"):
            return [resultingCluster]
        result = Topology.SubTopologies(resultingCluster, subTopologyType="cell")
        if result == None:
            return [] #Make sure you return an empty list instead of None
        return result
    
    @staticmethod
    def FreeShells(cluster, tolerance: float = 0.0001, silent: bool = False) -> list:
        """
        Returns the free shells of the input cluster that are not part of a higher topology.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input cluster.
        tolerance : float, optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The list of free shells.

        """
        from topologicpy.Cell import Cell
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Cluster"):
            if not silent:
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
        resultingCluster = Topology.Difference(allShellsCluster, cellsCluster, tolerance=tolerance)
        if resultingCluster == None:
            return []
        if Topology.IsInstance(resultingCluster, "Shell"):
            return [resultingCluster]
        result = Topology.SubTopologies(resultingCluster, subTopologyType="shell")
        if result == None:
            return [] #Make sure you return an empty list instead of None
        return result
    
    @staticmethod
    def FreeFaces(cluster, tolerance: float = 0.0001, silent: bool = False) -> list:
        """
        Returns the free faces of the input cluster that are not part of a higher topology.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input cluster.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The list of free faces.

        """
        from topologicpy.Shell import Shell
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Cluster"):
            if not silent:
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
        resultingCluster = Topology.Difference(allFacesCluster, shellCluster, tolerance=tolerance)
        if resultingCluster == None:
            return []
        if Topology.IsInstance(resultingCluster, "Face"):
            return [resultingCluster]
        result = Topology.SubTopologies(resultingCluster, subTopologyType="face")
        if result == None:
            return [] #Make sure you return an empty list instead of None
        return result

    @staticmethod
    def FreeWires(cluster, tolerance: float = 0.0001, silent: bool = False) -> list:
        """
        Returns the free wires of the input cluster that are not part of a higher topology.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input cluster.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The list of free wires.

        """
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Cluster"):
            if not silent:
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
        resultingCluster = Topology.Difference(allWiresCluster, facesCluster, tolerance=tolerance)
        if resultingCluster == None:
            return []
        if Topology.IsInstance(resultingCluster, "Wire"):
            return [resultingCluster]
        result = Topology.SubTopologies(resultingCluster, subTopologyType="wire")
        if not result:
            return [] #Make sure you return an empty list instead of None
        return result
    
    @staticmethod
    def FreeEdges(cluster, tolerance: float = 0.0001, silent: bool = False) -> list:
        """
        Returns the free edges of the input cluster that are not part of a higher topology.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input cluster.
        tolerance : float, optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The list of free edges.

        """
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Cluster"):
            if not silent:
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
        resultingCluster = Topology.Difference(allEdgesCluster, wireCluster, tolerance=tolerance)
        if resultingCluster == None:
            return []
        if Topology.IsInstance(resultingCluster, "Edge"):
            return [resultingCluster]
        result = Topology.SubTopologies(resultingCluster, subTopologyType="edge")
        if result == None:
            return [] #Make sure you return an empty list instead of None
        return result
    
    @staticmethod
    def FreeVertices(cluster, tolerance: float = 0.0001, silent: bool = False) -> list:
        """
        Returns the free vertices of the input cluster that are not part of a higher topology.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input cluster.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The list of free vertices.

        """
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Cluster"):
            if not silent:
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
        resultingCluster = Topology.Difference(allVerticesCluster, edgesCluster, tolerance=tolerance)
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
            The desired tolerance. Default is 0.0001.

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
    def KMeans(topologies,
                selectors=None,
                keys=["x", "y", "z"],
                k=4,
                maxIterations=100,
                centroidKey="k_centroid",
                distanceMeasure: str = "euclidean",   # "euclidean", "sqeuclidean", "manhattan", "chebyshev", "cosine", "mahalanobis"
                init: str = "kmeans++",              # "kmeans++" or "random"
                nInit: int = 10,                     # best-of-n restarts (like sklearn)
                tol: float = 1e-6,                   # convergence tolerance on centroid shift
                standardize: bool = False,           # z-score standardization
                normalize: bool = False,             # L2-normalize rows (useful for cosine / spherical k-means)
                randomSeed: int = None,
                mantissa: int = 6,
                tolerance: float = 0.0001,
                silent: bool = False):
        """
        Clusters the input topologies using K-Means-like clustering.

        Best-practice upgrades vs the legacy implementation:
        - k-means++ initialization (default) for faster, stabler convergence.
        - Multiple restarts (nInit) and selects best (lowest inertia / objective).
        - Vectorized numpy core for speed.
        - More distance measures:
            * euclidean / sqeuclidean : classic k-means (centroid = mean)
            * manhattan              : k-medians update (centroid = coordinate-wise median)
            * chebyshev              : centroid = coordinate-wise midrange (0.5*(min+max)) (robust-ish heuristic)
            * cosine                 : spherical k-means (normalize; centroid = normalized mean)
            * mahalanobis            : mean in whitened space (uses global covariance)
        - Robust empty-cluster handling (reseed with farthest point).

        Notes
        - Classic k-means objective is squared Euclidean; for other metrics this function uses the
        appropriate/standard centroid update where available (k-medians, spherical k-means), or a
        reasonable heuristic (chebyshev).
        - Feature extraction:
            * If keys include "x","y","z" (case-insensitive), these are taken from the selector vertex (or topology vertex).
            * Any other key is read from the topology dictionary and must be numeric.
            * Missing/None values are replaced by 0.0 (warns unless silent).

        Parameters
        ----------
        topologies : list
            The input list of topologies.
        selectors : list , optional
            If topologies are not vertices, provide selector vertices (e.g., centroids) of equal length.
            If None, topologies must be vertices.
        keys : list, optional
            Keys to build the feature vector. Include "x","y","z" (any case) for coordinates.
        k : int , optional
            Number of clusters.
        maxIterations : int , optional
            Maximum iterations.
        centroidKey : str , optional
            Dictionary key under which to store the cluster centroid feature vector.
        distanceMeasure : str , optional
            See list above.
        init : str , optional
            "kmeans++" or "random".
        nInit : int , optional
            Number of random restarts; best result returned.
        tol : float , optional
            Convergence tolerance on centroid movement.
        standardize : bool , optional
            Z-score features before clustering (recommended when mixing units).
        normalize : bool , optional
            L2-normalize feature rows (recommended for cosine / spherical k-means).
        randomSeed : int , optional
            RNG seed.
        mantissa : int , optional
            Rounding precision when storing centroid.
        tolerance : float , optional
            Tolerance (kept for API consistency).
        silent : bool , optional
            Suppress warnings/errors.

        Returns
        -------
        list
            The created list of clusters (topologic_core.Cluster), each with centroidKey stored in its dictionary.
        """
        
        import math
        import numpy as np

        from topologicpy.Vertex import Vertex
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology
        from topologicpy.Cluster import Cluster

        # --------------------------
        # Validation
        # --------------------------
        if not isinstance(topologies, list):
            if not silent:
                print("Cluster.KMeans - Error: The input topologies parameter is not a valid list. Returning None.")
            return None

        topologies = [t for t in topologies if Topology.IsInstance(t, "Topology")]
        if len(topologies) < 1:
            if not silent:
                print("Cluster.KMeans - Error: The input topologies parameter does not contain any valid topologies. Returning None.")
            return None

        if selectors is None:
            # Require vertices
            non_vertices = [t for t in topologies if not Topology.IsInstance(t, "Vertex")]
            if len(non_vertices) > 0:
                if not silent:
                    print("Cluster.KMeans - Error: selectors is None but topologies include non-Vertex objects. Returning None.")
                return None
        else:
            if not isinstance(selectors, list):
                if not silent:
                    print("Cluster.KMeans - Error: The input selectors parameter is not a valid list. Returning None.")
                return None
            selectors = [s for s in selectors if Topology.IsInstance(s, "Vertex")]
            if len(selectors) != len(topologies):
                if not silent:
                    print("Cluster.KMeans - Error: topologies and selectors must have the same length. Returning None.")
                return None
            if len(selectors) < 1:
                if not silent:
                    print("Cluster.KMeans - Error: selectors does not contain any valid vertices. Returning None.")
                return None

        if not isinstance(keys, list) or len(keys) < 1:
            if not silent:
                print("Cluster.KMeans - Error: The input keys parameter is not a valid non-empty list. Returning None.")
            return None

        if not isinstance(k, int) or k < 1:
            if not silent:
                print("Cluster.KMeans - Error: The input k parameter is not a valid integer >= 1. Returning None.")
            return None

        n = len(topologies)
        if n < k:
            if not silent:
                print("Cluster.KMeans - Error: The number of topologies is less than k. Returning None.")
            return None

        if not isinstance(maxIterations, int) or maxIterations < 1:
            if not silent:
                print("Cluster.KMeans - Error: maxIterations must be an integer >= 1. Returning None.")
            return None

        distanceMeasure = (distanceMeasure or "euclidean").strip().lower()
        init = (init or "kmeans++").strip().lower()
        if init not in ["kmeans++", "random"]:
            init = "kmeans++"

        if not isinstance(nInit, int) or nInit < 1:
            nInit = 1

        rng = np.random.default_rng(randomSeed)

        # --------------------------
        # Feature extraction
        # --------------------------
        def _safe_float(val):
            try:
                if val is None:
                    return None
                f = float(val)
                if math.isnan(f) or math.isinf(f):
                    return None
                return f
            except Exception:
                return None

        def _feature_row(topology, selector_vertex):
            row = []
            d = Topology.Dictionary(topology)
            for key in keys:
                kl = str(key).lower()
                if kl == "x":
                    v = Vertex.X(selector_vertex)
                    fv = _safe_float(v)
                elif kl == "y":
                    v = Vertex.Y(selector_vertex)
                    fv = _safe_float(v)
                elif kl == "z":
                    v = Vertex.Z(selector_vertex)
                    fv = _safe_float(v)
                else:
                    v = Dictionary.ValueAtKey(d, key)
                    fv = _safe_float(v)
                if fv is None:
                    if not silent:
                        print(f"Cluster.KMeans - Warning: Non-numeric or missing value for key '{key}'. Using 0.0.")
                    fv = 0.0
                row.append(fv)
            return row

        selector_list = selectors if selectors is not None else topologies
        X = np.array([_feature_row(topologies[i], selector_list[i]) for i in range(n)], dtype=float)

        if X.ndim != 2 or X.shape[0] != n or X.shape[1] < 1:
            if not silent:
                print("Cluster.KMeans - Error: Could not build a valid feature matrix. Returning None.")
            return None

        # Optional preprocessing
        X_work = X.copy()

        if standardize:
            mu = X_work.mean(axis=0)
            sigma = X_work.std(axis=0)
            sigma[sigma == 0] = 1.0
            X_work = (X_work - mu) / sigma

        if normalize:
            norms = np.linalg.norm(X_work, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            X_work = X_work / norms

        # --------------------------
        # Distance utilities
        # --------------------------
        def _pairwise_distances(Xa, C):
            """
            Returns D shape (n, k): distance from each Xa row to each centroid row in C.
            """
            if distanceMeasure in ["euclidean", "sqeuclidean"]:
                # Use squared distances for stability/speed; take sqrt only if requested.
                # d^2 = ||x||^2 + ||c||^2 - 2 x·c
                x2 = np.sum(Xa * Xa, axis=1, keepdims=True)          # (n,1)
                c2 = np.sum(C * C, axis=1, keepdims=True).T          # (1,k)
                d2 = np.maximum(x2 + c2 - 2.0 * (Xa @ C.T), 0.0)
                if distanceMeasure == "euclidean":
                    return np.sqrt(d2)
                return d2

            if distanceMeasure == "manhattan":
                return np.sum(np.abs(Xa[:, None, :] - C[None, :, :]), axis=2)

            if distanceMeasure == "chebyshev":
                return np.max(np.abs(Xa[:, None, :] - C[None, :, :]), axis=2)

            if distanceMeasure == "cosine":
                # 1 - cosine similarity (in [0,2]); assumes rows are normalized for spherical k-means
                # Still works without normalization but then includes magnitude effects.
                xnorm = np.linalg.norm(Xa, axis=1, keepdims=True)
                cnorm = np.linalg.norm(C, axis=1, keepdims=True).T
                xnorm[xnorm == 0] = 1.0
                cnorm[cnorm == 0] = 1.0
                sim = (Xa @ C.T) / (xnorm * cnorm)
                sim = np.clip(sim, -1.0, 1.0)
                return 1.0 - sim

            if distanceMeasure == "mahalanobis":
                # Global covariance (whitening) on X_work
                # d(x,c) = sqrt( (x-c)^T S^{-1} (x-c) )
                # We'll precompute inv covariance in outer scope for speed.
                raise RuntimeError("Internal: mahalanobis handled via whitening.")

            # Fallback
            return _pairwise_distances(Xa, C)

        # Mahalanobis via whitening
        if distanceMeasure == "mahalanobis":
            # Regularized covariance
            cov = np.cov(X_work, rowvar=False)
            cov = np.atleast_2d(cov)
            reg = 1e-9 * np.eye(cov.shape[0])
            try:
                inv_cov = np.linalg.inv(cov + reg)
            except Exception:
                inv_cov = np.linalg.pinv(cov + reg)

            # whiten transform via Cholesky of inv_cov if possible; else use eig
            try:
                L = np.linalg.cholesky(inv_cov)   # inv_cov = L L^T
                Xw = X_work @ L
            except Exception:
                w, V = np.linalg.eigh(inv_cov)
                w[w < 0] = 0
                Xw = X_work @ (V @ np.diag(np.sqrt(w)) @ V.T)
            X_work = Xw
            # Now Mahalanobis distance reduces to Euclidean in whitened space
            distanceMeasure = "euclidean"

        # --------------------------
        # Initialization
        # --------------------------
        def _init_centroids_random(Xa, k):
            idx = rng.choice(Xa.shape[0], size=k, replace=False)
            return Xa[idx].copy()

        def _init_centroids_kmeanspp(Xa, k):
            n = Xa.shape[0]
            # pick first centroid uniformly
            c_idx = [int(rng.integers(0, n))]
            C = [Xa[c_idx[0]].copy()]

            # maintain closest squared distance to any chosen centroid
            # use squared euclidean in working space for kmeans++ selection (standard approach)
            closest_d2 = np.sum((Xa - C[0]) ** 2, axis=1)

            for _ in range(1, k):
                probs = closest_d2 / (closest_d2.sum() if closest_d2.sum() > 0 else 1.0)
                next_idx = int(rng.choice(n, p=probs))
                c_idx.append(next_idx)
                C.append(Xa[next_idx].copy())
                d2_new = np.sum((Xa - Xa[next_idx]) ** 2, axis=1)
                closest_d2 = np.minimum(closest_d2, d2_new)

            return np.vstack(C)

        # --------------------------
        # Centroid updates
        # --------------------------
        def _update_centroids(Xa, labels, k):
            C = np.zeros((k, Xa.shape[1]), dtype=float)
            counts = np.zeros(k, dtype=int)

            for j in range(k):
                mask = (labels == j)
                counts[j] = int(mask.sum())
                if counts[j] == 0:
                    continue
                Xj = Xa[mask]
                if distanceMeasure in ["sqeuclidean", "euclidean"]:
                    C[j] = Xj.mean(axis=0)
                elif distanceMeasure == "manhattan":
                    C[j] = np.median(Xj, axis=0)
                elif distanceMeasure == "chebyshev":
                    C[j] = 0.5 * (Xj.min(axis=0) + Xj.max(axis=0))
                elif distanceMeasure == "cosine":
                    # spherical k-means: mean then renormalize
                    cj = Xj.mean(axis=0)
                    norm = np.linalg.norm(cj)
                    C[j] = cj / (norm if norm > 0 else 1.0)
                else:
                    C[j] = Xj.mean(axis=0)
            return C, counts

        def _objective(Xa, C, labels):
            # Inertia-like objective (sum of distances or squared distances depending on metric)
            if distanceMeasure == "sqeuclidean":
                d2 = np.sum((Xa - C[labels]) ** 2, axis=1)
                return float(d2.sum())
            D = _pairwise_distances(Xa, C)
            return float(D[np.arange(Xa.shape[0]), labels].sum())

        # --------------------------
        # Core solve (single run)
        # --------------------------
        def _solve_once(Xa):
            if init == "random":
                C = _init_centroids_random(Xa, k)
            else:
                C = _init_centroids_kmeanspp(Xa, k)

            prev_obj = None

            for _it in range(maxIterations):
                D = _pairwise_distances(Xa, C)
                labels = np.argmin(D, axis=1)

                C_new, counts = _update_centroids(Xa, labels, k)

                # Empty cluster handling: reseed empties to farthest points
                if np.any(counts == 0):
                    # distance to assigned centroid
                    dist_to_assigned = D[np.arange(Xa.shape[0]), labels]
                    # sort farthest-first
                    far_order = np.argsort(-dist_to_assigned)
                    empties = np.where(counts == 0)[0].tolist()
                    used = set()
                    for j in empties:
                        # pick farthest point not already used for reseeding
                        pick = None
                        for idx in far_order:
                            if int(idx) not in used:
                                pick = int(idx)
                                break
                        if pick is None:
                            pick = int(far_order[0])
                        used.add(pick)
                        C_new[j] = Xa[pick].copy()

                # Convergence check: centroid shift
                shift = np.linalg.norm(C_new - C)
                C = C_new

                obj = _objective(Xa, C, labels)
                if prev_obj is not None and abs(prev_obj - obj) <= tol * (abs(prev_obj) + tol):
                    break
                if shift <= tol:
                    break
                prev_obj = obj

            # final labels/objective
            D = _pairwise_distances(Xa, C)
            labels = np.argmin(D, axis=1)
            obj = _objective(Xa, C, labels)
            return labels, C, obj

        # --------------------------
        # Best-of-nInit
        # --------------------------
        best = None
        best_obj = float("inf")
        for _ in range(nInit):
            labels, C_work, obj = _solve_once(X_work)
            if obj < best_obj:
                best_obj = obj
                best = (labels.copy(), C_work.copy())

        labels, C_work = best

        # --------------------------
        # Map centroids back to original feature space (for storage)
        # --------------------------
        # If we standardized, unstandardize stored centroids.
        # If we normalized, stored centroids are in normalized space (that’s appropriate for cosine/spherical);
        # we still store them as-is, since they represent the model centroid in feature space used.
        C_store = C_work.copy()
        if standardize:
            # reverse z-score
            C_store = (C_store * sigma) + mu

        # Round centroids for storage
        C_store = np.round(C_store.astype(float), mantissa).tolist()

        # --------------------------
        # Build Topologic clusters
        # --------------------------
        t_clusters = []
        for j in range(k):
            idxs = np.where(labels == j)[0].tolist()
            if len(idxs) == 0:
                continue  # should not happen after reseeding, but keep safe

            # cluster members are ORIGINAL topologies (not selectors)
            members = [topologies[i] for i in idxs]
            t_cluster = Cluster.ByTopologies(members)

            d = Dictionary.ByKeysValues([centroidKey], [C_store[j]])
            t_cluster = Topology.SetDictionary(t_cluster, d)
            t_clusters.append(t_cluster)

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
            The desired tolerance. Default is 0.0001.

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
    def MysticRose(wire= None, origin= None, radius: float = 0.5, sides: int = 16, perimeter: bool = True, direction: list = [0, 0, 1], placement:str = "center", tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a mystic rose.

        Parameters
        ----------
        wire : topologic_core.Wire , optional
            The input Wire. if set to None, a circle with the input parameters is created. Otherwise, the input parameters are ignored.
        origin : topologic_core.Vertex , optional
            The location of the origin of the circle. Default is None which results in the circle being placed at (0, 0, 0).
        radius : float , optional
            The radius of the mystic rose. Default is 1.
        sides : int , optional
            The number of sides of the mystic rose. Default is 16.
        perimeter : bool , optional
            If True, the perimeter edges are included in the output. Default is True.
        direction : list , optional
            The vector representing the up direction of the mystic rose. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the mystic rose. This can be "center", or "lowerleft". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

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
            edges.append(Edge.ByVertices([vertices[comb[0]], vertices[comb[1]]], tolerance=tolerance, silent=silent))
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
            The desired size of the tripod. Default is 1.0.
        radius : float , optional
            The desired radiues of the tripod. Default is 0.03
        sides : int , optional
            The desired number of sides of the tripod. Default is 4.
        faceColorKey : str , optional
            The dictionary key under which to store the colors of the axes.
        xColor : str , optional
            The color to use for the X axis. Default is "red".
        yColor : str , optional
            The color to use for the Y axis. Default is "green".
        zColor : str , optional
            The color to use for the Z axis. Default is "blue".
        matrix : list , optional
            The desired 4X4 transformation matrix to use for transforming the tripod. Default is None which means the tripod will be placed at the origin and will be axis-aligned. 

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

    