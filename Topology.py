#from base64 import b16encode
#from tkinter import N
import topologicpy
import topologic

import uuid
import json
import os
import numpy as np
from numpy import arctan, pi, signbit
from numpy.linalg import norm
import math
from scipy.spatial import ConvexHull
'''
try:
    import ifcopenshell
    import ifcopenshell.geom
except:
    raise Exception("Error: TopologyByImportedIFC: ifcopenshell is not present on your system. Install BlenderBIM or ifcopenshell to resolve.")
'''

class Topology():
    @staticmethod
    def AddApertures(topology, apertures, exclusive=False, subTopologyType=None, tolerance=0.0001):
        """
        Adds the input list of apertures to the input topology or to its subtpologies based on the input subTopologyType.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        apertures : list
            The input list of apertures.
        exclusive : bool , optional
            If set to True, one (sub)topology will accept only one aperture. Otherwise, one (sub)topology can accept multiple apertures. The default is False.
        subTopologyType : string , optional
            The subtopology type to which to add the apertures. This can be "cell", "face", "edge", or "vertex". It is case insensitive. If set to None, the apertures will be added to the input topology. The defaul is None.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Topology
            The input topology with the apertures added to it.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Cluster import Cluster
        from topologicpy.Aperture import Aperture
        def processApertures(subTopologies, apertures, exclusive=False, tolerance=0.001):
            usedTopologies = []
            for subTopology in subTopologies:
                    usedTopologies.append(0)
            ap = 1
            for aperture in apertures:
                apCenter = Topology.InternalVertex(aperture, tolerance)
                for i in range(len(subTopologies)):
                    subTopology = subTopologies[i]
                    if exclusive == True and usedTopologies[i] == 1:
                        continue
                    if Vertex.Distance(apCenter, subTopology) < tolerance:
                        context = topologic.Context.ByTopologyParameters(subTopology, 0.5, 0.5, 0.5)
                        _ = Aperture.ByTopologyContext(aperture, context)
                        if exclusive == True:
                            usedTopologies[i] = 1
                ap = ap + 1
            return None

        if not isinstance(topology, topologic.Topology):
            return None
        if not apertures:
            return topology
        if not isinstance(apertures, list):
            return None
        apertures = [x for x in apertures if isinstance(x , topologic.Topology)]
        if len(apertures) < 1:
            return topology
        if not subTopologyType:
            subTopologyType = "self"
        if not subTopologyType.lower() in ["self", "cell", "face", "edge", "vertex"]:
            return None
        if subTopologyType.lower() == "self":
            subTopologies = [topology]
        else:
            subTopologies = Topology.SubTopologies(topology, subTopologyType)
        processApertures(subTopologies, apertures, exclusive, tolerance)
        return topology
    
    @staticmethod
    def AddContent(topology, contents, subTopologyType=None, tolerance=0.0001):
        """
        Adds the input list of contents to the input topology or to its subtpologies based on the input subTopologyType.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        conntents : list
            The input list of contents.
        subTopologyType : string , optional
            The subtopology type to which to add the contents. This can be "cellcomplex", "cell", "shell", "face", "wire", "edge", or "vertex". It is case insensitive. If set to None, the contents will be added to the input topology. The defaul is None.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Topology
            The input topology with the contents added to it.

        """
        if not isinstance(topology, topologic.Topology):
            return None
        if not contents:
            return topology
        if not isinstance(contents, list):
            return None
        contents = [x for x in contents if isinstance(x, topologic.Topology)]
        if len(contents) < 1:
            return topology
        if not subTopologyType:
            subTopologyType = "self"
        if not subTopologyType.lower() in ["self", "cellcomplex", "cell", "shell", "face", "wire", "edge", "vertex"]:
            return None
        if subTopologyType.lower() == "vertex":
            t = topologic.Vertex.Type()
        elif subTopologyType.lower() == "edge":
            t = topologic.Edge.Type()
        elif subTopologyType.lower() == "wire":
            t = topologic.Wire.Type()
        elif subTopologyType.lower() == "face":
            t = topologic.Face.Type()
        elif subTopologyType.lower() == "shell":
            t = topologic.Shell.Type()
        elif subTopologyType.lower() == "cell":
            t = topologic.Cell.Type()
        elif subTopologyType.lower() == "cellcomplex":
            t = topologic.CellComplex.Type()
        else:
            t = 0
        return topology.AddContents(contents, t)
    
    @staticmethod
    def AddDictionary(topology, dictionary):
        """
        Adds the input dictionary to the input topology.

        Parameters
        ----------
        topology : topologic.topology
            The input topology.
        dictionary : topologic.Dictionary
            The input dictionary.

        Returns
        -------
        topologic.Topology
            The input topology with the input dictionary added to it.

        """
        
        from topologicpy.Dictionary import Dictionary
        if not isinstance(topology, topologic.Topology):
            return None
        if not isinstance(dictionary, topologic.Dictionary):
            return None
        tDict = Topology.Dictionary(topology)
        if len(tDict.Keys()) < 1:
            _ = topology.SetDictionary(dictionary)
        else:
            newDict = Dictionary.ByMergedDictionaries([tDict, dictionary])
            if newDict:
                _ = topology.SetDictionary(newDict)
        return topology
    
    @staticmethod
    def AdjacentTopologies(topology, hostTopology, topologyType=None):
        """
        Returns the topologies, as specified by the input topology type, adjacent to the input topology witin the input host topology.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        hostTopology : topologic.Topology
            The host topology in which to search.
        topologyType : str
            The type of topology for which to search. This can be one of "vertex", "edge", "wire", "face", "shell", "cell", "cellcomplex". It is case-insensitive. If it is set to None, the type will be set to the same type as the input topology. The default is None.

        Returns
        -------
        adjacentTopologies : list
            The list of adjacent topologies.

        """
        if not isinstance(topology, topologic.Topology):
            return None
        if not isinstance(hostTopology, topologic.Topology):
            return None
        if not topologyType:
            topologyType = Topology.TypeAsString(topology).lower()
        if not isinstance(topologyType, str):
            return None
        if not topologyType.lower() in ["vertex", "edge", "wire", "face", "shell", "cell", "cellcomplex"]:
            return None
        adjacentTopologies = []
        error = False
        if isinstance(topology, topologic.Vertex):
            if topologyType.lower() == "vertex":
                try:
                    _ = topology.AdjacentVertices(hostTopology, adjacentTopologies)
                except:
                    try:
                        _ = topology.Vertices(hostTopology, adjacentTopologies)
                    except:
                        error = True
            elif topologyType.lower() == "edge":
                try:
                    _ = topologic.VertexUtility.AdjacentEdges(topology, hostTopology, adjacentTopologies)
                except:
                    try:
                        _ = topology.Edges(hostTopology, adjacentTopologies)
                    except:
                        error = True
            elif topologyType.lower() == "wire":
                try:
                    _ = topologic.VertexUtility.AdjacentWires(topology, hostTopology, adjacentTopologies)
                except:
                    try:
                        _ = topology.Wires(hostTopology, adjacentTopologies)
                    except:
                        error = True
            elif topologyType.lower() == "face":
                try:
                    _ = topologic.VertexUtility.AdjacentFaces(topology, hostTopology, adjacentTopologies)
                except:
                    try:
                        _ = topology.Faces(hostTopology, adjacentTopologies)
                    except:
                        error = True
            elif topologyType.lower() == "shell":
                try:
                    _ = topologic.VertexUtility.AdjacentShells(topology, hostTopology, adjacentTopologies)
                except:
                    try:
                        _ = topology.Shells(hostTopology, adjacentTopologies)
                    except:
                        error = True
            elif topologyType.lower() == "cell":
                try:
                    _ = topologic.VertexUtility.AdjacentCells(topology, hostTopology, adjacentTopologies)
                except:
                    try:
                        _ = topology.Cells(hostTopology, adjacentTopologies)
                    except:
                        error = True
            elif topologyType.lower() == "cellcomplex":
                try:
                    _ = topologic.VertexUtility.AdjacentCellComplexes(topology, hostTopology, adjacentTopologies)
                except:
                    try:
                        _ = topology.CellComplexes(hostTopology, adjacentTopologies)
                    except:
                        error = True
        elif isinstance(topology, topologic.Edge):
            if topologyType.lower() == "vertex":
                try:
                    _ = topology.Vertices(hostTopology, adjacentTopologies)
                except:
                    error = True
            elif topologyType.lower() == "edge":
                try:
                    _ = topology.AdjacentEdges(hostTopology, adjacentTopologies)
                except:
                    error = True
            elif topologyType.lower() == "wire":
                try:
                    _ = topologic.EdgeUtility.AdjacentWires(topology, adjacentTopologies)
                except:
                    try:
                        _ = topology.Wires(hostTopology, adjacentTopologies)
                    except:
                        error = True
            elif topologyType.lower() == "face":
                try:
                    _ = topologic.EdgeUtility.AdjacentFaces(topology, adjacentTopologies)
                except:
                    try:
                        _ = topology.Faces(hostTopology, adjacentTopologies)
                    except:
                        error = True
            elif topologyType.lower() == "shell":
                try:
                    _ = topologic.EdgeUtility.AdjacentShells(adjacentTopologies)
                except:
                    try:
                        _ = topology.Shells(hostTopology, adjacentTopologies)
                    except:
                        error = True
            elif topologyType.lower() == "cell":
                try:
                    _ = topologic.EdgeUtility.AdjacentCells(adjacentTopologies)
                except:
                    try:
                        _ = topology.Cells(hostTopology, adjacentTopologies)
                    except:
                        error = True
            elif topologyType.lower() == "cellcomplex":
                try:
                    _ = topologic.EdgeUtility.AdjacentCellComplexes(adjacentTopologies)
                except:
                    try:
                        _ = topology.CellComplexes(hostTopology, adjacentTopologies)
                    except:
                        error = True
        elif isinstance(topology, topologic.Wire):
            if topologyType.lower() == "vertex":
                try:
                    _ = topologic.WireUtility.AdjacentVertices(topology, adjacentTopologies)
                except:
                    try:
                        _ = topology.Vertices(hostTopology, adjacentTopologies)
                    except:
                        error = True
            elif topologyType.lower() == "edge":
                try:
                    _ = topologic.WireUtility.AdjacentEdges(topology, adjacentTopologies)
                except:
                    try:
                        _ = topology.Edges(hostTopology, adjacentTopologies)
                    except:
                        error = True
            elif topologyType.lower() == "wire":
                try:
                    _ = topologic.WireUtility.AdjacentWires(topology, adjacentTopologies)
                except:
                    try:
                        _ = topology.Wires(hostTopology, adjacentTopologies)
                    except:
                        error = True
            elif topologyType.lower() == "face":
                try:
                    _ = topologic.WireUtility.AdjacentFaces(topology, adjacentTopologies)
                except:
                    try:
                        _ = topology.Faces(hostTopology, adjacentTopologies)
                    except:
                        error = True
            elif topologyType.lower() == "shell":
                try:
                    _ = topologic.WireUtility.AdjacentShells(adjacentTopologies)
                except:
                    try:
                        _ = topology.Shells(hostTopology, adjacentTopologies)
                    except:
                        error = True
            elif topologyType.lower() == "cell":
                try:
                    _ = topologic.WireUtility.AdjacentCells(adjacentTopologies)
                except:
                    try:
                        _ = topology.Cells(hostTopology, adjacentTopologies)
                    except:
                        error = True
            elif topologyType.lower() == "cellcomplex":
                try:
                    _ = topologic.WireUtility.AdjacentCellComplexes(adjacentTopologies)
                except:
                    try:
                        _ = topology.CellComplexes(hostTopology, adjacentTopologies)
                    except:
                        error = True
        elif isinstance(topology, topologic.Face):
            if topologyType.lower() == "vertex":
                try:
                    _ = topologic.FaceUtility.AdjacentVertices(topology, adjacentTopologies)
                except:
                    try:
                        _ = topology.Vertices(hostTopology, adjacentTopologies)
                    except:
                        error = True
            elif topologyType.lower() == "edge":
                try:
                    _ = topologic.FaceUtility.AdjacentEdges(topology, adjacentTopologies)
                except:
                    try:
                        _ = topology.Edges(hostTopology, adjacentTopologies)
                    except:
                        error = True
            elif topologyType.lower() == "wire":
                try:
                    _ = topologic.FaceUtility.AdjacentWires(topology, adjacentTopologies)
                except:
                    try:
                        _ = topology.Wires(hostTopology, adjacentTopologies)
                    except:
                        error = True
            elif topologyType.lower() == "face":
                _ = topology.AdjacentFaces(hostTopology, adjacentTopologies)
            elif topologyType.lower() == "shell":
                try:
                    _ = topologic.FaceUtility.AdjacentShells(adjacentTopologies)
                except:
                    try:
                        _ = topology.Shells(hostTopology, adjacentTopologies)
                    except:
                        error = True
            elif topologyType.lower() == "cell":
                try:
                    _ = topologic.FaceUtility.AdjacentCells(adjacentTopologies)
                except:
                    try:
                        _ = topology.Cells(hostTopology, adjacentTopologies)
                    except:
                        error = True
            elif topologyType.lower() == "cellcomplex":
                try:
                    _ = topologic.FaceUtility.AdjacentCellComplexes(adjacentTopologies)
                except:
                    try:
                        _ = topology.CellComplexes(hostTopology, adjacentTopologies)
                    except:
                        error = True
        elif isinstance(topology, topologic.Shell):
            if topologyType.lower() == "vertex":
                try:
                    _ = topologic.ShellUtility.AdjacentVertices(topology, adjacentTopologies)
                except:
                    try:
                        _ = topology.Vertices(hostTopology, adjacentTopologies)
                    except:
                        error = True
            elif topologyType.lower() == "edge":
                try:
                    _ = topologic.ShellUtility.AdjacentEdges(topology, adjacentTopologies)
                except:
                    try:
                        _ = topology.Edges(hostTopology, adjacentTopologies)
                    except:
                        error = True
            elif topologyType.lower() == "wire":
                try:
                    _ = topologic.ShellUtility.AdjacentWires(topology, adjacentTopologies)
                except:
                    try:
                        _ = topology.Wires(hostTopology, adjacentTopologies)
                    except:
                        error = True
            elif topologyType.lower() == "face":
                try:
                    _ = topologic.ShellUtility.AdjacentFaces(topology, adjacentTopologies)
                except:
                    try:
                        _ = topology.Faces(hostTopology, adjacentTopologies)
                    except:
                        error = True
            elif topologyType.lower() == "shell":
                try:
                    _ = topologic.ShellUtility.AdjacentShells(adjacentTopologies)
                except:
                    try:
                        _ = topology.Shells(hostTopology, adjacentTopologies)
                    except:
                        error = True
            elif topologyType.lower() == "cell":
                try:
                    _ = topologic.ShellUtility.AdjacentCells(adjacentTopologies)
                except:
                    try:
                        _ = topology.Cells(hostTopology, adjacentTopologies)
                    except:
                        error = True
            elif topologyType.lower() == "cellcomplex":
                try:
                    _ = topologic.ShellUtility.AdjacentCellComplexes(adjacentTopologies)
                except:
                    try:
                        _ = topology.CellComplexes(hostTopology, adjacentTopologies)
                    except:
                        error = True
        elif isinstance(topology, topologic.Cell):
            if topologyType.lower() == "vertex":
                try:
                    _ = topologic.CellUtility.AdjacentVertices(topology, adjacentTopologies)
                except:
                    try:
                        _ = topology.Vertices(hostTopology, adjacentTopologies)
                    except:
                        error = True
            elif topologyType.lower() == "edge":
                try:
                    _ = topologic.CellUtility.AdjacentEdges(topology, adjacentTopologies)
                except:
                    try:
                        _ = topology.Edges(hostTopology, adjacentTopologies)
                    except:
                        error = True
            elif topologyType.lower() == "wire":
                try:
                    _ = topologic.CellUtility.AdjacentWires(topology, adjacentTopologies)
                except:
                    try:
                        _ = topology.Wires(hostTopology, adjacentTopologies)
                    except:
                        error = True
            elif topologyType.lower() == "face":
                try:
                    _ = topologic.CellUtility.AdjacentFaces(topology, adjacentTopologies)
                except:
                    try:
                        _ = topology.Faces(hostTopology, adjacentTopologies)
                    except:
                        error = True
            elif topologyType.lower() == "shell":
                try:
                    _ = topologic.CellUtility.AdjacentShells(adjacentTopologies)
                except:
                    try:
                        _ = topology.Shells(hostTopology, adjacentTopologies)
                    except:
                        error = True
            elif topologyType.lower() == "cell":
                try:
                    _ = topology.AdjacentCells(hostTopology, adjacentTopologies)
                except:
                    try:
                        _ = topology.Cells(hostTopology, adjacentTopologies)
                    except:
                        error = True
            elif topologyType.lower() == "cellcomplex":
                try:
                    _ = topologic.CellUtility.AdjacentCellComplexes(adjacentTopologies)
                except:
                    try:
                        _ = topology.CellComplexes(hostTopology, adjacentTopologies)
                    except:
                        error = True
        elif isinstance(topology, topologic.CellComplex):
            if topologyType.lower() == "vertex":
                try:
                    _ = topology.Vertices(hostTopology, adjacentTopologies)
                except:
                    error = True
            elif topologyType.lower() == "edge":
                try:
                    _ = topology.Edges(hostTopology, adjacentTopologies)
                except:
                    error = True
            elif topologyType.lower() == "wire":
                try:
                    _ = topology.Wires(hostTopology, adjacentTopologies)
                except:
                    error = True
            elif topologyType.lower() == "face":
                try:
                    _ = topology.Faces(hostTopology, adjacentTopologies)
                except:
                    error = True
            elif topologyType.lower() == "shell":
                try:
                    _ = topology.Shells(hostTopology, adjacentTopologies)
                except:
                    error = True
            elif topologyType.lower() == "cell":
                try:
                    _ = topology.Cells(hostTopology, adjacentTopologies)
                except:
                    error = True
            elif topologyType.lower() == "cellcomplex":
                raise Exception("Topology.AdjacentTopologies - Error: Cannot search for adjacent topologies of a CellComplex")
        elif isinstance(topology, topologic.Cluster):
            raise Exception("Topology.AdjacentTopologies - Error: Cannot search for adjacent topologies of a Cluster")
        if error:
            raise Exception("Topology.AdjacentTopologies - Error: Failure in search for adjacent topologies of type "+topologyType)
        return adjacentTopologies

    @staticmethod
    def Analyze(topology):
        """
        Returns an analysis string that describes the input topology.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.

        Returns
        -------
        str
            The analysis string.

        """
        if not isinstance(topology, topologic.Topology):
            return None
        return topologic.Topology.Analyze(topology)
    
    @staticmethod
    def Apertures(topology):
        """
        Returns the apertures of the input topology.
        
        Parameters
        ----------
        topology : topologic.Topology
            The input topology.

        Returns
        -------
        list
            The list of apertures beloning to the input topology.

        """
        if not isinstance(topology, topologic.Topology):
            return None
        apertures = []
        _ = topology.Apertures(apertures)
        return apertures

    @staticmethod
    def ApertureTopologies(topology):
        """
        Returns the aperture topologies of the input topology.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.

        Returns
        -------
        list
            The list of aperture topologies found in the input topology.

        """
        from topologicpy.Aperture import Aperture
        if not isinstance(topology, topologic.Topology):
            return None
        apertures = Topology.Apertures(topology)
        apTopologies = []
        for aperture in apertures:
            apTopologies.append(Aperture.ApertureTopology(aperture))
        return apTopologies
    
    @staticmethod
    def Boolean(topologyA, topologyB, operation="union", tranDict=False, tolerance=0.0001):
        """
        Execute the input boolean operation type on the input operand topologies and return the result. See https://en.wikipedia.org/wiki/Boolean_operation.

        Parameters
        ----------
        topologyA : topologic.Topology
            The first input topology.
        topologyB : topologic.Topology
            The second input topology.
        operation : str , optional
            The boolean operation. This can be one of "union", "difference", "intersect", "symdif", "merge", "slice", "impose", "imprint". It is case insensitive. The default is "union".
        tranDict : bool , optional
            If set to True the dictionaries of the operands are merged and transferred to the result. The default is False.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Topology
            the resultant topology.

        """
        from topologicpy.Dictionary import Dictionary
        
        if not isinstance(topologyA, topologic.Topology):
            return None
        if not isinstance(topologyB, topologic.Topology):
            return None
        if not isinstance(operation, str):
            return None
        if not operation.lower() in ["union", "difference", "intersect", "symdif", "merge", "slice", "impose", "imprint"]:
            return None
        if not isinstance(tranDict, bool):
            return None
        topologyC = None
        try:
            if operation.lower() == "union":
                topologyC = topologyA.Union(topologyB, False)
            elif operation.lower() == "difference":
                topologyC = topologyA.Difference(topologyB, False)
            elif operation.lower() == "intersect":
                topologyC = topologyA.Intersect(topologyB, False)
            elif operation.lower() == "symdif":
                topologyC = topologyA.XOR(topologyB, False)
            elif operation.lower() == "merge":
                topologyC = topologyA.Merge(topologyB, False)
            elif operation.lower() == "slice":
                topologyC = topologyA.Slice(topologyB, False)
            elif operation.lower() == "impose":
                topologyC = topologyA.Impose(topologyB, False)
            elif operation.lower() == "imprint":
                topologyC = topologyA.Imprint(topologyB, False)
            else:
                return None
        except:
            return None
        if tranDict == True:
            sourceVertices = []
            sourceEdges = []
            sourceFaces = []
            sourceCells = []
            hidimA = Topology.HighestType(topologyA)
            hidimB = Topology.HighestType(topologyB)
            hidimC = Topology.HighestType(topologyC)
            verticesA = []
            if topologyA.Type() == topologic.Vertex.Type():
                verticesA.append(topologyA)
            elif hidimA >= topologic.Vertex.Type():
                _ = topologyA.Vertices(None, verticesA)
                for aVertex in verticesA:
                    sourceVertices.append(aVertex)
            verticesB = []
            if topologyB.Type() == topologic.Vertex.Type():
                verticesB.append(topologyB)
            elif hidimB >= topologic.Vertex.Type():
                _ = topologyB.Vertices(None, verticesB)
                for aVertex in verticesB:
                    sourceVertices.append(aVertex)
            sinkVertices = []
            if topologyC.Type() == topologic.Vertex.Type():
                sinkVertices.append(topologyC)
            elif hidimC >= topologic.Vertex.Type():
                _ = topologyC.Vertices(None, sinkVertices)
            _ = Topology.TransferDictionaries(sourceVertices, sinkVertices, tolerance)
            if topologyA.Type() == topologic.Edge.Type():
                sourceEdges.append(topologyA)
            elif hidimA >= topologic.Edge.Type():
                edgesA = []
                _ = topologyA.Edges(None, edgesA)
                for anEdge in edgesA:
                    sourceEdges.append(anEdge)
            if topologyB.Type() == topologic.Edge.Type():
                sourceEdges.append(topologyB)
            elif hidimB >= topologic.Edge.Type():
                edgesB = []
                _ = topologyB.Edges(None, edgesB)
                for anEdge in edgesB:
                    sourceEdges.append(anEdge)
            sinkEdges = []
            if topologyC.Type() == topologic.Edge.Type():
                sinkEdges.append(topologyC)
            elif hidimC >= topologic.Edge.Type():
                _ = topologyC.Edges(None, sinkEdges)
            _ = Topology.TransferDictionaries(sourceEdges, sinkEdges, tolerance)

            if topologyA.Type() == topologic.Face.Type():
                sourceFaces.append(topologyA)
            elif hidimA >= topologic.Face.Type():
                facesA = []
                _ = topologyA.Faces(None, facesA)
                for aFace in facesA:
                    sourceFaces.append(aFace)
            if topologyB.Type() == topologic.Face.Type():
                sourceFaces.append(topologyB)
            elif hidimB >= topologic.Face.Type():
                facesB = []
                _ = topologyB.Faces(None, facesB)
                for aFace in facesB:
                    sourceFaces.append(aFace)
            sinkFaces = []
            if topologyC.Type() == topologic.Face.Type():
                sinkFaces.append(topologyC)
            elif hidimC >= topologic.Face.Type():
                _ = topologyC.Faces(None, sinkFaces)
            _ = Topology.TransferDictionaries(sourceFaces, sinkFaces, tolerance)
            if topologyA.Type() == topologic.Cell.Type():
                sourceCells.append(topologyA)
            elif hidimA >= topologic.Cell.Type():
                cellsA = []
                _ = topologyA.Cells(None, cellsA)
                for aCell in cellsA:
                    sourceCells.append(aCell)
            if topologyB.Type() == topologic.Cell.Type():
                sourceCells.append(topologyB)
            elif hidimB >= topologic.Cell.Type():
                cellsB = []
                _ = topologyB.Cells(None, cellsB)
                for aCell in cellsB:
                    sourceCells.append(aCell)
            sinkCells = []
            if topologyC.Type() == topologic.Cell.Type():
                sinkCells.append(topologyC)
            elif hidimC >= topologic.Cell.Type():
                _ = topologyC.Cells(None, sinkCells)
            _ = Topology.TransferDictionaries(sourceCells, sinkCells, tolerance)
        return topologyC

    
    @staticmethod
    def BoundingBox(topology, optimize=0, axes="xyz"):
        """
        Returns a cell representing a bounding box of the input topology. The returned cell contains a dictionary with keys "xrot", "yrot", and "zrot" that represents rotations around the X,Y, and Z axes. If applied in the order of Z,Y,Z, the resulting box will become axis-aligned.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        optimize : int , optional
            If set to an integer from 1 (low optimization) to 10 (high optimization), the method will attempt to optimize the bounding box so that it reduces its surface area. The default is 0 which will result in an axis-aligned bounding box. The default is 0.
        axes : str , optional
            Sets what axes are to be used for rotating the bounding box. This can be any permutation or substring of "xyz". It is not case sensitive. The default is "xyz".
        Returns
        -------
        topologic.Cell
            The bounding box of the input topology.

        """
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Cell import Cell
        from topologicpy.Cluster import Cluster
        from topologicpy.Dictionary import Dictionary
        def bb(topology):
            vertices = []
            _ = topology.Vertices(None, vertices)
            x = []
            y = []
            z = []
            for aVertex in vertices:
                x.append(aVertex.X())
                y.append(aVertex.Y())
                z.append(aVertex.Z())
            minX = min(x)
            minY = min(y)
            minZ = min(z)
            maxX = max(x)
            maxY = max(y)
            maxZ = max(z)
            return [minX, minY, minZ, maxX, maxY, maxZ]

        if not isinstance(topology, topologic.Topology):
            return None
        axes = axes.lower()
        x_flag = "x" in axes
        y_flag = "y" in axes
        z_flag = "z" in axes
        if not x_flag and not y_flag and not z_flag:
            return None
        vertices = Topology.SubTopologies(topology, subTopologyType="vertex")
        topology = Cluster.ByTopologies(vertices)
        boundingBox = bb(topology)
        minX = boundingBox[0]
        minY = boundingBox[1]
        minZ = boundingBox[2]
        maxX = boundingBox[3]
        maxY = boundingBox[4]
        maxZ = boundingBox[5]
        w = abs(maxX - minX)
        l = abs(maxY - minY)
        h = abs(maxZ - minZ)
        best_area = 2*l*w + 2*l*h + 2*w*h
        orig_area = best_area
        best_x = 0
        best_y = 0
        best_z = 0
        best_bb = boundingBox
        origin = Topology.Centroid(topology)
        optimize = min(max(optimize, 0), 10)
        if optimize > 0:
            factor = (round(((11 - optimize)/30 + 0.57), 2))
            flag = False
            for n in range(10,0,-1):
                if flag:
                    break
                if x_flag:
                    xa = n
                    xb = 90+n
                    xc = n
                else:
                    xa = 0
                    xb = 1
                    xc = 1
                if y_flag:
                    ya = n
                    yb = 90+n
                    yc = n
                else:
                    ya = 0
                    yb = 1
                    yc = 1
                if z_flag:
                    za = n
                    zb = 90+n
                    zc = n
                else:
                    za = 0
                    zb = 1
                    zc = 1
                for x in range(xa,xb,xc):
                    if flag:
                        break
                    for y in range(ya,yb,yc):
                        if flag:
                            break
                        for z in range(za,zb,zc):
                            if flag:
                                break
                            t = Topology.Rotate(topology, origin=origin, x=0,y=0,z=1, degree=z)
                            t = Topology.Rotate(t, origin=origin, x=0,y=1,z=0, degree=y)
                            t = Topology.Rotate(t, origin=origin, x=1,y=0,z=0, degree=x)
                            minX, minY, minZ, maxX, maxY, maxZ = bb(t)
                            w = abs(maxX - minX)
                            l = abs(maxY - minY)
                            h = abs(maxZ - minZ)
                            area = 2*l*w + 2*l*h + 2*w*h
                            if area < orig_area*factor:
                                best_area = area
                                best_x = x
                                best_y = y
                                best_z = z
                                best_bb = [minX, minY, minZ, maxX, maxY, maxZ]
                                flag = True
                                break
                            if area < best_area:
                                best_area = area
                                best_x = x
                                best_y = y
                                best_z = z
                                best_bb = [minX, minY, minZ, maxX, maxY, maxZ]
                        
        else:
            best_bb = boundingBox

        minX, minY, minZ, maxX, maxY, maxZ = best_bb
        vb1 = topologic.Vertex.ByCoordinates(minX, minY, minZ)
        vb2 = topologic.Vertex.ByCoordinates(maxX, minY, minZ)
        vb3 = topologic.Vertex.ByCoordinates(maxX, maxY, minZ)
        vb4 = topologic.Vertex.ByCoordinates(minX, maxY, minZ)

        vt1 = topologic.Vertex.ByCoordinates(minX, minY, maxZ)
        vt2 = topologic.Vertex.ByCoordinates(maxX, minY, maxZ)
        vt3 = topologic.Vertex.ByCoordinates(maxX, maxY, maxZ)
        vt4 = topologic.Vertex.ByCoordinates(minX, maxY, maxZ)
        baseWire = Wire.ByVertices([vb1, vb2, vb3, vb4], close=True)
        baseFace = Face.ByWire(baseWire)
        box = Cell.ByThickenedFace(baseFace, thickness=abs(maxZ-minZ), bothSides=False)
        #topWire = Wire.ByVertices([vt1, vt2, vt3, vt4], close=True)
        #wires = [baseWire, topWire]
        #box = Cell.ByWires(wires)
        box = Topology.Rotate(box, origin=origin, x=1,y=0,z=0, degree=-best_x)
        box = Topology.Rotate(box, origin=origin, x=0,y=1,z=0, degree=-best_y)
        box = Topology.Rotate(box, origin=origin, x=0,y=0,z=1, degree=-best_z)
        dictionary = Dictionary.ByKeysValues(["xrot","yrot","zrot"], [best_x, best_y, best_z])
        box = Topology.SetDictionary(box, dictionary)
        return box

    @staticmethod
    def ByImportedBRep(path):
        """
        Create a topology by importing it from a BRep file path.

        Parameters
        ----------
        path : str
            The path to the BRep file.

        Returns
        -------
        topology : topologic.Topology
            The created topology.

        """
        topology = None
        brep_file = open(path)
        if not brep_file:
            return None
        brep_string = brep_file.read()
        topology = Topology.ByString(brep_string)
        brep_file.close()
        return topology
    '''
    @staticmethod
    def ByImportedIFC(filePath, typeList):
        """
        NOT DONE YET

        Parameters
        ----------
        filePath : TYPE
            DESCRIPTION.
        typeList : TYPE
            DESCRIPTION.

        Returns
        -------
        returnList : TYPE
            DESCRIPTION.

        """
        
        def processKeysValues(keys, values):
            if len(keys) != len(values):
                raise Exception("DictionaryByKeysValues - Keys and Values do not have the same length")
            stl_keys = []
            stl_values = []
            for i in range(len(keys)):
                if isinstance(keys[i], str):
                    stl_keys.append(keys[i])
                else:
                    stl_keys.append(str(keys[i]))
                if isinstance(values[i], list) and len(values[i]) == 1:
                    value = values[i][0]
                else:
                    value = values[i]
                if isinstance(value, bool):
                    if value == False:
                        stl_values.append(topologic.IntAttribute(0))
                    else:
                        stl_values.append(topologic.IntAttribute(1))
                elif isinstance(value, int):
                    stl_values.append(topologic.IntAttribute(value))
                elif isinstance(value, float):
                    stl_values.append(topologic.DoubleAttribute(value))
                elif isinstance(value, str):
                    stl_values.append(topologic.StringAttribute(value))
                elif isinstance(value, list):
                    l = []
                    for v in value:
                        if isinstance(v, bool):
                            l.append(topologic.IntAttribute(v))
                        elif isinstance(v, int):
                            l.append(topologic.IntAttribute(v))
                        elif isinstance(v, float):
                            l.append(topologic.DoubleAttribute(v))
                        elif isinstance(v, str):
                            l.append(topologic.StringAttribute(v))
                    stl_values.append(topologic.ListAttribute(l))
                else:
                    raise Exception("Error: Value type is not supported. Supported types are: Boolean, Integer, Double, String, or List.")
            myDict = topologic.Dictionary.ByKeysValues(stl_keys, stl_values)
            return myDict

        def triangulate(faces):
            triangles = []
            for aFace in faces:
                ib = []
                _ = aFace.InternalBoundaries(ib)
                if len(ib) != 0:
                    print("Found Internal Boundaries")
                    faceTriangles = []
                    topologic.FaceUtility.Triangulate(aFace, 0.0, faceTriangles)
                    print("Length of Face Triangles:", len(faceTriangles))
                    for aFaceTriangle in faceTriangles:
                        triangles.append(aFaceTriangle)
                else:
                    triangles.append(aFace)
            return triangles
        
        settings = ifcopenshell.geom.settings()
        settings.set(settings.USE_BREP_DATA,True)
        settings.set(settings.SEW_SHELLS,True)
        settings.set(settings.USE_WORLD_COORDS,False)

        ifc_file = ifcopenshell.open(filePath)
        if len(typeList) < 1:
            typeList = ifc_file.types()
        returnList = []
        for aType in typeList:
            products = ifc_file.by_type(aType)
            for p in products:
                try:
                    cr = ifcopenshell.geom.create_shape(settings, p)
                    brepString = cr.geometry.brep_data
                    topology = topologic.Topology.ByString(brepString)
                    if topology.Type() == 8:
                        triangles = triangulate([topology])
                        topology = topologic.Cluster.ByTopologies(triangles)
                    elif topology.Type() > 8:
                        faces = []
                        _ = topology.Faces(None, faces)
                        triangles = triangulate(faces)
                        topology = topologic.Cluster.ByTopologies(triangles)
                    keys = []
                    values = []
                    keys.append("TOPOLOGIC_color")
                    values.append([1.0,1.0,1.0,1.0])
                    keys.append("TOPOLOGIC_id")
                    values.append(str(uuid.uuid4()))
                    keys.append("TOPOLOGIC_name")
                    values.append(p.Name)
                    keys.append("TOPOLOGIC_type")
                    values.append(topology.GetTypeAsString())
                    keys.append("IFC_id")
                    values.append(str(p.GlobalId))
                    keys.append("IFC_name")
                    values.append(p.Name)
                    keys.append("IFC_type")
                    values.append(p.is_a())
                    for definition in p.IsDefinedBy:
                        # To support IFC2X3, we need to filter our results.
                        if definition.is_a('IfcRelDefinesByProperties'):
                            property_set = definition.RelatingPropertyDefinition
                            for property in property_set.HasProperties:
                                if property.is_a('IfcPropertySingleValue'):
                                    keys.append(property.Name)
                                    values.append(property.NominalValue.wrappedValue)
                    topDict = processKeysValues(keys, values)
                    _ = topology.SetDictionary(topDict)
                except:
                    continue
                returnList.append(topology)
        return returnList

    @staticmethod
    def ByImportedIPFS(hash_, url, port):
        """
        NOT DONE YET.

        Parameters
        ----------
        hash : TYPE
            DESCRIPTION.
        url : TYPE
            DESCRIPTION.
        port : TYPE
            DESCRIPTION.

        Returns
        -------
        topology : TYPE
            DESCRIPTION.

        """
        # hash, url, port = item
        url = url.replace('http://','')
        url = '/dns/'+url+'/tcp/'+port+'/https'
        client = ipfshttpclient.connect(url)
        brepString = client.cat(hash_).decode("utf-8")
        topology = Topology.ByString(brepString)
        return topology
    '''
    @staticmethod
    def ByImportedJSONMK1(filePath, tolerance=0.0001):
        """
        Imports the topology from a JSON file.

        Parameters
        ----------
        filePath : str
            The file path to the json file.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        list
            The list of imported topologies.

        """
        from topologicpy.Dictionary import Dictionary

        

        def processApertures(subTopologies, apertures, exclusive, tolerance):
            usedTopologies = []
            for subTopology in subTopologies:
                    usedTopologies.append(0)
            ap = 1
            for aperture in apertures:
                apCenter = Topology.InternalVertex(aperture, tolerance)
                for i in range(len(subTopologies)):
                    subTopology = subTopologies[i]
                    if exclusive == True and usedTopologies[i] == 1:
                        continue
                    if topologic.VertexUtility.Distance(apCenter, subTopology) < tolerance:
                        context = topologic.Context.ByTopologyParameters(subTopology, 0.5, 0.5, 0.5)
                        _ = topologic.Aperture.ByTopologyContext(aperture, context)
                        if exclusive == True:
                            usedTopologies[i] = 1
                ap = ap + 1
            return None

        def getApertures(apertureList):
            returnApertures = []
            for item in apertureList:
                aperture = Topology.ByString(item['brep'])
                dictionary = item['dictionary']
                keys = list(dictionary.keys())
                values = []
                for key in keys:
                    values.append(dictionary[key])
                topDictionary = Dictionary.ByKeysValues(keys, values)
                if len(keys) > 0:
                    _ = aperture.SetDictionary(topDictionary)
                returnApertures.append(aperture)
            return returnApertures
        
        def assignDictionary(dictionary):
            selector = dictionary['selector']
            pydict = dictionary['dictionary']
            v = topologic.Vertex.ByCoordinates(selector[0], selector[1], selector[2])
            d = Dictionary.ByPythonDictionary(pydict)
            _ = v.SetDictionary(d)
            return v

        topology = None
        file = open(filePath)
        if file:
            topologies = []
            jsondata = json.load(file)
            for jsonItem in jsondata:
                brep = jsonItem['brep']
                topology = Topology.ByString(brep)
                dictionary = jsonItem['dictionary']
                topDictionary = Dictionary.ByPythonDictionary(dictionary)
                topology = Topology.SetDictionary(topology, topDictionary)
                cellApertures = getApertures(jsonItem['cellApertures'])
                cells = []
                try:
                    _ = topology.Cells(None, cells)
                except:
                    pass
                processApertures(cells, cellApertures, False, 0.001)
                faceApertures = getApertures(jsonItem['faceApertures'])
                faces = []
                try:
                    _ = topology.Faces(None, faces)
                except:
                    pass
                processApertures(faces, faceApertures, False, 0.001)
                edgeApertures = getApertures(jsonItem['edgeApertures'])
                edges = []
                try:
                    _ = topology.Edges(None, edges)
                except:
                    pass
                processApertures(edges, edgeApertures, False, 0.001)
                vertexApertures = getApertures(jsonItem['vertexApertures'])
                vertices = []
                try:
                    _ = topology.Vertices(None, vertices)
                except:
                    pass
                processApertures(vertices, vertexApertures, False, 0.001)
                cellDataList = jsonItem['cellDictionaries']
                cellSelectors = []
                for cellDataItem in cellDataList:
                    cellSelectors.append(assignDictionary(cellDataItem))
                Topology.TransferDictionariesBySelectors(topology=topology, selectors=cellSelectors, tranVertices=False, tranEdges=False, tranFaces=False, tranCells=True, tolerance=0.0001)
                faceDataList = jsonItem['faceDictionaries']
                faceSelectors = []
                for faceDataItem in faceDataList:
                    faceSelectors.append(assignDictionary(faceDataItem))
                Topology.TransferDictionariesBySelectors(topology=topology, selectors=faceSelectors, tranVertices=False, tranEdges=False, tranFaces=True, tranCells=False, tolerance=0.0001)
                edgeDataList = jsonItem['edgeDictionaries']
                edgeSelectors = []
                for edgeDataItem in edgeDataList:
                    edgeSelectors.append(assignDictionary(edgeDataItem))
                Topology.TransferDictionariesBySelectors(topology=topology, selectors=edgeSelectors, tranVertices=False, tranEdges=True, tranFaces=False, tranCells=False, tolerance=0.0001)
                vertexDataList = jsonItem['vertexDictionaries']
                vertexSelectors = []
                for vertexDataItem in vertexDataList:
                    vertexSelectors.append(assignDictionary(vertexDataItem))
                Topology.TransferDictionariesBySelectors(topology=topology, selectors=vertexSelectors, tranVertices=True, tranEdges=False, tranFaces=False, tranCells=False, tolerance=0.0001)
                topologies.append(topology)
            return topologies
        return None

    @staticmethod
    def ByImportedJSONMK2(filePath, tolerance=0.0001):
        """
        Imports the topology from a JSON file.

        Parameters
        ----------
        filePath : str
            The file path to the json file.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        list
            The list of imported topologies.

        """
        from topologicpy.Dictionary import Dictionary
        def getApertures(apertureList, folderPath):
            returnApertures = []
            for item in apertureList:
                brepFileName = item['brep']
                brepFilePath = os.path.join(folderPath, brepFileName+".brep")
                brepFile = open(brepFilePath)
                if brepFile:
                    brepString = brepFile.read()
                    aperture = Topology.ByString(brepString)
                    brepFile.close()
                dictionary = item['dictionary']
                keys = list(dictionary.keys())
                values = []
                for key in keys:
                    values.append(dictionary[key])
                topDictionary = Dictionary.ByKeysValues(keys, values)
                if len(keys) > 0:
                    _ = aperture.SetDictionary(topDictionary)
                returnApertures.append(aperture)
            return returnApertures

        def processApertures(subTopologies, apertures, exclusive, tolerance):
            usedTopologies = []
            for subTopology in subTopologies:
                    usedTopologies.append(0)
            ap = 1
            for aperture in apertures:
                apCenter = Topology.InternalVertex(aperture, tolerance)
                for i in range(len(subTopologies)):
                    subTopology = subTopologies[i]
                    if exclusive == True and usedTopologies[i] == 1:
                        continue
                    if topologic.VertexUtility.Distance(apCenter, subTopology) < tolerance:
                        context = topologic.Context.ByTopologyParameters(subTopology, 0.5, 0.5, 0.5)
                        _ = topologic.Aperture.ByTopologyContext(aperture, context)
                        if exclusive == True:
                            usedTopologies[i] = 1
                ap = ap + 1
            return None

        def assignDictionary(dictionary):
            selector = dictionary['selector']
            pydict = dictionary['dictionary']
            v = topologic.Vertex.ByCoordinates(selector[0], selector[1], selector[2])
            d = Dictionary.ByPythonDictionary(pydict)
            _ = v.SetDictionary(d)
            return v

        topology = None
        jsonFile = open(filePath)
        folderPath = os.path.dirname(filePath)
        if jsonFile:
            topologies = []
            jsondata = json.load(jsonFile)
            for jsonItem in jsondata:
                brepFileName = jsonItem['brep']
                brepFilePath = os.path.join(folderPath, brepFileName+".brep")
                brepFile = open(brepFilePath)
                if brepFile:
                    brepString = brepFile.read()
                    topology = Topology.ByString(brepString)
                    brepFile.close()
                #topology = topologic.Topology.ByString(brep)
                dictionary = jsonItem['dictionary']
                topDictionary = Dictionary.ByPythonDictionary(dictionary)
                _ = topology.SetDictionary(topDictionary)
                cellApertures = getApertures(jsonItem['cellApertures'], folderPath)
                cells = []
                try:
                    _ = topology.Cells(None, cells)
                except:
                    pass
                processApertures(cells, cellApertures, False, 0.001)
                faceApertures = getApertures(jsonItem['faceApertures'], folderPath)
                faces = []
                try:
                    _ = topology.Faces(None, faces)
                except:
                    pass
                processApertures(faces, faceApertures, False, 0.001)
                edgeApertures = getApertures(jsonItem['edgeApertures'], folderPath)
                edges = []
                try:
                    _ = topology.Edges(None, edges)
                except:
                    pass
                processApertures(edges, edgeApertures, False, 0.001)
                vertexApertures = getApertures(jsonItem['vertexApertures'], folderPath)
                vertices = []
                try:
                    _ = topology.Vertices(None, vertices)
                except:
                    pass
                processApertures(vertices, vertexApertures, False, 0.001)
                cellDataList = jsonItem['cellDictionaries']
                cellSelectors = []
                for cellDataItem in cellDataList:
                    cellSelectors.append(assignDictionary(cellDataItem))
                Topology.TransferDictionariesBySelectors(topology, cellSelectors, tranVertices=False, tranEdges=False, tranFaces=False, tranCells=True, tolerance=tolerance)
                faceDataList = jsonItem['faceDictionaries']
                faceSelectors = []
                for faceDataItem in faceDataList:
                    faceSelectors.append(assignDictionary(faceDataItem))
                Topology.TransferDictionariesBySelectors(topology, faceSelectors, tranVertices=False, tranEdges=False, tranFaces=True, tranCells=False, tolerance=tolerance)
                edgeDataList = jsonItem['edgeDictionaries']
                edgeSelectors = []
                for edgeDataItem in edgeDataList:
                    edgeSelectors.append(assignDictionary(edgeDataItem))
                Topology.TransferDictionariesBySelectors(topology, edgeSelectors, tranVertices=False, tranEdges=True, tranFaces=False, tranCells=False, tolerance=tolerance)
                vertexDataList = jsonItem['vertexDictionaries']
                vertexSelectors = []
                for vertexDataItem in vertexDataList:
                    vertexSelectors.append(assignDictionary(vertexDataItem))
                Topology.TransferDictionariesBySelectors(topology, vertexSelectors, tranVertices=True, tranEdges=False, tranFaces=False, tranCells=False, tolerance=tolerance)
                topologies.append(topology)
            return topologies
        return None

    @staticmethod
    def ByOCCTShape(occtShape):
        """
        Creates a topology from the input OCCT shape. See https://dev.opencascade.org/doc/overview/html/occt_user_guides__modeling_data.html.

        Parameters
        ----------
        occtShape : topologic.TopoDS_Shape
            The inoput OCCT Shape.

        Returns
        -------
        topologic.Topology
            The created topology.

        """
        return topologic.Topology.ByOcctShape(occtShape, "")
    
    @staticmethod
    def ByString(string):
        """
        Creates a topology from the input brep string

        Parameters
        ----------
        string : str
            The input brep string.

        Returns
        -------
        topologic.Topology
            The created topology.

        """
        if not isinstance(string, str):
            return None
        returnTopology = None
        try:
            returnTopology = topologic.Topology.ByString(string)
        except:
            returnTopology = None
        return returnTopology
    
    @staticmethod
    def CenterOfMass(topology):
        """
        Returns the center of mass of the input topology. See https://en.wikipedia.org/wiki/Center_of_mass.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.

        Returns
        -------
        topologic.Vertex
            The center of mass of the input topology.

        """
        if not isinstance(topology, topologic.Topology):
            return None
        return topology.CenterOfMass()
    
    @staticmethod
    def Centroid(topology):
        """
        Returns the centroid of the vertices of the input topology. See https://en.wikipedia.org/wiki/Centroid.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.

        Returns
        -------
        topologic.Vertex
            The centroid of the input topology.

        """
        if not isinstance(topology, topologic.Topology):
            return None
        return topology.Centroid()
    
    @staticmethod
    def ClusterFaces(topology, tolerance=0.0001):
        """
        Clusters the faces of the input topology by their direction.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        list
            The list of clusters of faces where faces in the same cluster have the same direction.

        """
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster
        def angle_between(v1, v2):
            u1 = v1 / norm(v1)
            u2 = v2 / norm(v2)
            y = u1 - u2
            x = u1 + u2
            if norm(x) == 0:
                return 0
            a0 = 2 * arctan(norm(y) / norm(x))
            if (not signbit(a0)) or signbit(pi - a0):
                return a0
            elif signbit(a0):
                return 0
            else:
                return pi

        def collinear(v1, v2, tol):
            ang = angle_between(v1, v2)
            if math.isnan(ang) or math.isinf(ang):
                raise Exception("Face.IsCollinear - Error: Could not determine the angle between the input faces")
            elif abs(ang) < tol or abs(pi - ang) < tol:
                return True
            return False
        
        def sumRow(matrix, i):
            return np.sum(matrix[i,:])
        
        def buildSimilarityMatrix(samples, tol):
            numOfSamples = len(samples)
            matrix = np.zeros(shape=(numOfSamples, numOfSamples))
            for i in range(len(matrix)):
                for j in range(len(matrix)):
                    if collinear(samples[i], samples[j], tol):
                        matrix[i,j] = 1
            return matrix

        def determineRow(matrix):
            maxNumOfOnes = -1
            row = -1
            for i in range(len(matrix)):
                if maxNumOfOnes < sumRow(matrix, i):
                    maxNumOfOnes = sumRow(matrix, i)
                    row = i
            return row

        def categorizeIntoClusters(matrix):
            groups = []
            while np.sum(matrix) > 0:
                group = []
                row = determineRow(matrix)
                indexes = addIntoGroup(matrix, row)
                groups.append(indexes)
                matrix = deleteChosenRowsAndCols(matrix, indexes)
            return groups

        def addIntoGroup(matrix, ind):
            change = True
            indexes = []
            for col in range(len(matrix)):
                if matrix[ind, col] == 1:
                    indexes.append(col)
            while change == True:
                change = False
                numIndexes = len(indexes)
                for i in indexes:
                    for col in range(len(matrix)):
                        if matrix[i, col] == 1:
                            if col not in indexes:
                                indexes.append(col)
                numIndexes2 = len(indexes)
                if numIndexes != numIndexes2:
                    change = True
            return indexes

        def deleteChosenRowsAndCols(matrix, indexes):
            for i in indexes:
                matrix[i,:] = 0
                matrix[:,i] = 0
            return matrix
        faces = []
        _ = topology.Faces(None, faces)
        normals = []
        for aFace in faces:
            normals.append(Face.NormalAtParameters(aFace, 0.5, 0.5, "XYZ", 3))
        # build a matrix of similarity
        mat = buildSimilarityMatrix(normals, tolerance)
        categories = categorizeIntoClusters(mat)
        returnList = []
        for aCategory in categories:
            tempList = []
            if len(aCategory) > 0:
                for index in aCategory:
                    tempList.append(faces[index])
                returnList.append(Cluster.SelfMerge(Cluster.ByTopologies(tempList)))
        return returnList

    @staticmethod
    def Contents(topology):
        """
        Returns the contents of the input topology

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.

        Returns
        -------
        list
            The list of contents of the input topology.

        """
        contents = []
        _ = topology.Contents(contents)
        return contents
    
    @staticmethod
    def Contexts(topology):
        """
        Returns the list of contexts of the input topology

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.

        Returns
        -------
        list
            The list of contexts of the input topology.

        """
        contexts = []
        _ = topology.Contexts(contexts)
        return contexts

    @staticmethod
    def ConvexHull(topology, tolerance=0.0001):
        """
        Creates a convex hull

        Parameters
        ----------
        topology : topologic.Topology
            The input Topology.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Topology
            The created convex hull of the input topology.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Shell import Shell
        from topologicpy.Cell import Cell
        from topologicpy.Cluster import Cluster
        def convexHull3D(item, tol, option):
            if item:
                vertices = []
                _ = item.Vertices(None, vertices)
                pointList = []
                for v in vertices:
                    pointList.append([v.X(), v.Y(), v.Z()])
                points = np.array(pointList)
                if option:
                    hull = ConvexHull(points, qhull_options=option)
                else:
                    hull = ConvexHull(points)
                faces = []
                for simplex in hull.simplices:
                    edges = []
                    for i in range(len(simplex)-1):
                        sp = hull.points[simplex[i]]
                        ep = hull.points[simplex[i+1]]
                        sv = Vertex.ByCoordinates(sp[0], sp[1], sp[2])
                        ev = Vertex.ByCoordinates(ep[0], ep[1], ep[2])
                        edges.append(Edge.ByVertices([sv, ev]))
                    sp = hull.points[simplex[-1]]
                    ep = hull.points[simplex[0]]
                    sv = Vertex.ByCoordinates(sp[0], sp[1], sp[2])
                    ev = Vertex.ByCoordinates(ep[0], ep[1], ep[2])
                    edges.append(Edge.ByVertices([sv, ev]))
                    faces.append(Face.ByWire(Wire.ByEdges(edges)))
            try:
                c = Cell.ByFaces(faces, tol)
                return c
            except:
                returnTopology = Cluster.SelfMerge(Cluster.ByTopologies(faces))
                if returnTopology.Type() == 16:
                    return Shell.ExternalBoundary(returnTopology)
        returnObject = None
        try:
            returnObject = convexHull3D(topology, tolerance, None)
        except:
            returnObject = convexHull3D(topology, tolerance, 'QJ')
        return returnObject
    
    @staticmethod
    def Copy(topology):
        """
        Returns a copy of the input topology

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.

        Returns
        -------
        topologic.Topology
            A copy of the input topology.

        """
        return topologic.Topology.DeepCopy(topology)
    
    @staticmethod
    def Dictionary(topology):
        """
        Returns the dictionary of the input topology

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.

        Returns
        -------
        topologic.Dictionary
            The dictionary of the input topology.

        """
        if not isinstance(topology, topologic.Topology):
            return None
        return topology.GetDictionary()
    
    @staticmethod
    def Dimensionality(topology):
        """
        Returns the dimensionality of the input topology

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.

        Returns
        -------
        int
            The dimensionality of the input topology.

        """
        return topology.Dimensionality()
    
    @staticmethod
    def Divide(topology, tool, transferDictionary=False, addNestingDepth=False):
        """
        Divides the input topology by the input tool and places the results in the contents of the input topology.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        tool : topologic.Topology
            the tool used to divide the input topology.
        transferDictionary : bool , optional
            If set to True the dictionary of the input topology is transferred to the divided topologies.
        addNestingDepth : bool , optional
            If set to True the nesting depth of the division is added to the dictionaries of the divided topologies.

        Returns
        -------
        topologic.Topology
            The input topology with the divided topologies added to it as contents.

        """
        
        from topologicpy.Dictionary import Dictionary
        
        try:
            _ = topology.Divide(tool, False) # Don't transfer dictionaries just yet
        except:
            raise Exception("TopologyDivide - Error: Divide operation failed.")
        nestingDepth = "1"
        keys = ["nesting_depth"]
        values = [nestingDepth]

        if not addNestingDepth and not transferDictionary:
            return topology

        contents = []
        _ = topology.Contents(contents)
        for i in range(len(contents)):
            if not addNestingDepth and transferDictionary:
                parentDictionary = Topology.Dictionary(topology)
                if parentDictionary != None:
                    _ = contents[i].SetDictionary(parentDictionary)
            if addNestingDepth and transferDictionary:
                parentDictionary = Topology.Dictionary(topology)
                if parentDictionary != None:
                    keys = Dictionary.Keys(parentDictionary)
                    values = Dictionary.Values(parentDictionary)
                    if ("nesting_depth" in keys):
                        nestingDepth = parentDictionary.ValueAtKey("nesting_depth").StringValue()
                    else:
                        keys.append("nesting_depth")
                        values.append(nestingDepth)
                    parentDictionary = Dictionary.ByKeysValues(keys, values)
                else:
                    keys = ["nesting_depth"]
                    values = [nestingDepth]
                parentDictionary = Dictionary.ByKeysValues(keys, values)
                _ = topology.SetDictionary(parentDictionary)
                values[keys.index("nesting_depth")] = nestingDepth+"_"+str(i+1)
                d = Dictionary.ByKeysValues(keys, values)
                _ = contents[i].SetDictionary(d)
            if addNestingDepth and  not transferDictionary:
                parentDictionary = Topology.Dictionary(topology)
                if parentDictionary != None:
                    keys, values = Dictionary.ByKeysValues(parentDictionary)
                    if ("nesting_depth" in keys):
                        nestingDepth = parentDictionary.ValueAtKey("nesting_depth").StringValue()
                    else:
                        keys.append("nesting_depth")
                        values.append(nestingDepth)
                    parentDictionary = Dictionary.ByKeysValues(keys, values)
                else:
                    keys = ["nesting_depth"]
                    values = [nestingDepth]
                parentDictionary = Dictionary.ByKeysValues(keys, values)
                _ = topology.SetDictionary(parentDictionary)
                keys = ["nesting_depth"]
                v = nestingDepth+"_"+str(i+1)
                values = [v]
                d = Dictionary.ByKeysValues(keys, values)
                _ = contents[i].SetDictionary(d)
        return topology
    
    @staticmethod
    def Explode(topology, origin=None, scale=1.25, typeFilter=None, axes="xyz"):
        """
        Explodes the input topology. See https://en.wikipedia.org/wiki/Exploded-view_drawing.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        origin : topologic.Vertex , optional
            The origin of the explosion. If set to None, the centroid of the input topology will be used. The defaul is None.
        scale : float , optional
            The scale factor of the explosion. The default is 1.25.
        typeFilter : str , optional
            The type of the subtopologies to explode. This can be any of "vertex", "edge", "face", or "cell". If set to None, a subtopology one level below the type of the input topology will be used. The default is None.
        axes : str , optional
            Sets what axes are to be used for exploding the topology. This can be any permutation or substring of "xyz". It is not case sensitive. The default is "xyz".
        
        Returns
        -------
        topologic.Cluster
            The exploded topology.

        """
        from topologicpy.Cluster import Cluster
        from topologicpy.Graph import Graph

        def processClusterTypeFilter(cluster):
            if len(Cluster.CellComplexes()) > 0:
                return "cell"
            elif len(Cluster.Cells()) > 0:
                return "face"
            elif len(Cluster.Shells()) > 0:
                return "face"
            elif len(Cluster.Faces()) > 0:
                return "edge"
            elif len(Cluster.Wires()) > 0:
                return "edge"
            elif len(Cluster.Edges()) > 0:
                return "vertex"
            else:
                return "self"

        def getTypeFilter(topology):
            typeFilter = "self"
            if isinstance(topology, topologic.Vertex):
                typeFilter = "self"
            elif isinstance(topology, topologic.Edge):
                typeFilter = "vertex"
            elif isinstance(topology, topologic.Wire):
                typeFilter = "edge"
            elif isinstance(topology, topologic.Face):
                typeFilter = "edge"
            elif isinstance(topology, topologic.Shell):
                typeFilter = "face"
            elif isinstance(topology, topologic.Cell):
                typeFilter = "face"
            elif isinstance(topology, topologic.CellComplex):
                typeFilter = "cell"
            elif isinstance(topology, topologic.Cluster):
                typeFilter = processClusterTypeFilter(topology)
            elif isinstance(topology, topologic.Graph):
                typeFilter = "edge"
            return typeFilter
        
        if not isinstance(topology, topologic.Topology):
            return None
        if not isinstance(origin, topologic.Vertex):
            origin = Topology.CenterOfMass(topology)
        if not typeFilter:
            typeFilter = getTypeFilter(topology)
        if not isinstance(typeFilter, str):
            return None
        if not isinstance(axes, str):
            return None
        axes = axes.lower()
        x_flag = "x" in axes
        y_flag = "y" in axes
        z_flag = "z" in axes
        if not x_flag and not y_flag and not z_flag:
            return None

        topologies = []
        newTopologies = []
        if isinstance(topology, topologic.Graph):
            topology = Graph.Topology(topology)

        if typeFilter.lower() == "self":
            topologies = [topology]
        else:
            topologies = Topology.SubTopologies(topology, subTopologyType=typeFilter.lower())
        for aTopology in topologies:
            c = Topology.RelevantSelector(aTopology)
            oldX = c.X()
            oldY = c.Y()
            oldZ = c.Z()
            if x_flag:
                newX = (oldX - origin.X())*scale + origin.X()
            else:
                newX = oldX
            if y_flag:
                newY = (oldY - origin.Y())*scale + origin.Y()
            else:
                newY = oldY
            if z_flag:
                newZ = (oldZ - origin.Z())*scale + origin.Z()
            else:
                newZ = oldZ
            xT = newX - oldX
            yT = newY - oldY
            zT = newZ - oldZ
            newTopology = Topology.Translate(aTopology, xT, yT, zT)
            newTopologies.append(newTopology)
        return Cluster.ByTopologies(newTopologies)

    
    @staticmethod
    def ExportToBRep(topology, filePath, overwrite=True, version=3):
        """
        Exports the input topology to a BREP file. See https://dev.opencascade.org/doc/occt-6.7.0/overview/html/occt_brep_format.html#:~:text=BREP%20format%20is%20used%20to,triangulations%2C%20space%20location%20and%20orientation.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        filePath : str
            The input file path.
        overwrite : bool , optional
            If set to True the ouptut file will overwrite any pre-existing file. Otherwise, it won't.
        version : int , optional
            Replaces the BREP version number and header to make the file compatible with older version of OCCT.

        Returns
        -------
        bool
            True if the export operation is successful. False otherwise.

        """
        if not isinstance(topology, topologic.Topology):
            return None
        if not isinstance(filePath, str):
            return None
        # Make sure the file extension is .BREP
        ext = filePath[len(filePath)-5:len(filePath)]
        if ext.lower() != ".brep":
            filePath = filePath+".brep"
        f = None
        try:
            if overwrite == True:
                f = open(filePath, "w")
            else:
                f = open(filePath, "x") # Try to create a new File
        except:
            raise Exception("Error: Could not create a new file at the following location: "+filePath)
        if (f):
            if version < 3:
                topology = Topology.SelfMerge(topology)
                s = topology.String().replace("V3, (c) Open Cascade", "V1, (c) Matra-Datavision")
                s = "DBRep_DrawableShape\n"+s
            else:
                s = topology.String()
            f.write(s)
            f.close()    
            return True
        return False

    '''
    @staticmethod
    def ExportToIPFS(topology, url, port, user, password):
        """
        NOT DONE YET

        Parameters
        ----------
        topology : TYPE
            DESCRIPTION.
        url : TYPE
            DESCRIPTION.
        port : TYPE
            DESCRIPTION.
        user : TYPE
            DESCRIPTION.
        password : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # topology, url, port, user, password = item
        
        def exportToBREP(topology, filePath, overwrite):
            # Make sure the file extension is .BREP
            ext = filePath[len(filePath)-5:len(filePath)]
            if ext.lower() != ".brep":
                filePath = filePath+".brep"
            f = None
            try:
                if overwrite == True:
                    f = open(filePath, "w")
                else:
                    f = open(filePath, "x") # Try to create a new File
            except:
                raise Exception("Error: Could not create a new file at the following location: "+filePath)
            if (f):
                topString = topology.String()
                f.write(topString)
                f.close()	
                return True
            return False
        
        filePath = os.path.expanduser('~')+"/tempFile.brep"
        if exportToBREP(topology, filePath, True):
            url = url.replace('http://','')
            url = '/dns/'+url+'/tcp/'+port+'/https'
            client = ipfshttpclient.connect(url, auth=(user, password))
            newfile = client.add(filePath)
            os.remove(filePath)
            return newfile['Hash']
        return ''
    '''

    @staticmethod
    def ExportToJSONMK1(topologies, filePath, overwrite=False, tolerance=0.0001):
        """
        Export the input list of topologies to a JSON file

        Parameters
        ----------
        topologies : list
            The input list of topologies.
        filePath : str
            The path to the JSON file.
        overwrite : bool , optional
            If set to True, any existing file will be overwritten. The default is False.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        bool
            The status of exporting the JSON file. If True, the operation was successful. Otherwise, it was unsuccesful.

        """

        from topologicpy.Dictionary import Dictionary
        def cellAperturesAndDictionaries(topology, tol):
            cells = []
            try:
                _ = topology.Cells(None, cells)
            except:
                return [[],[],[]]
            cellApertures = []
            cellDictionaries = []
            cellSelectors = []
            for aCell in cells:
                tempApertures = []
                _ = aCell.Apertures(tempApertures)
                for anAperture in tempApertures:
                    cellApertures.append(anAperture)
                cellDictionary = Topology.Dictionary(aCell)
                if len(cellDictionary.keys()) > 0:
                    cellDictionaries.append(cellDictionary)
                    iv = topologic.CellUtility.InternalVertex(aCell, tol)
                    cellSelectors.append([iv.X(), iv.Y(), iv.Z()])
            return [cellApertures, cellDictionaries, cellSelectors]

        def faceAperturesAndDictionaries(topology, tol):
            faces = []
            try:
                _ = topology.Faces(None, faces)
            except:
                return [[],[],[]]
            faceApertures = []
            faceDictionaries = []
            faceSelectors = []
            for aFace in faces:
                tempApertures = []
                _ = aFace.Apertures(tempApertures)
                for anAperture in tempApertures:
                    faceApertures.append(anAperture)
                faceDictionary = Topology.Dictionary(aFace)
                if len(faceDictionary.keys()) > 0:
                    faceDictionaries.append(faceDictionary)
                    iv = topologic.FaceUtility.InternalVertex(aFace, tol)
                    faceSelectors.append([iv.X(), iv.Y(), iv.Z()])
            return [faceApertures, faceDictionaries, faceSelectors]

        def edgeAperturesAndDictionaries(topology, tol):
            edges = []
            try:
                _ = topology.Edges(None, edges)
            except:
                return [[],[],[]]
            edgeApertures = []
            edgeDictionaries = []
            edgeSelectors = []
            for anEdge in edges:
                tempApertures = []
                _ = anEdge.Apertures(tempApertures)
                for anAperture in tempApertures:
                    edgeApertures.append(anAperture)
                edgeDictionary = Topology.Dictionary(anEdge)
                if len(edgeDictionary.keys()) > 0:
                    edgeDictionaries.append(edgeDictionary)
                    iv = topologic.EdgeUtility.PointAtParameter(anEdge, 0.5)
                    edgeSelectors.append([iv.X(), iv.Y(), iv.Z()])
            return [edgeApertures, edgeDictionaries, edgeSelectors]

        def vertexAperturesAndDictionaries(topology, tol):
            vertices = []
            try:
                _ = topology.Vertices(None, vertices)
            except:
                return [[],[],[]]
            vertexApertures = []
            vertexDictionaries = []
            vertexSelectors = []
            for aVertex in vertices:
                tempApertures = []
                _ = aVertex.Apertures(tempApertures)
                for anAperture in tempApertures:
                    vertexApertures.append(anAperture)
                vertexDictionary = Topology.Dictionary(aVertex)
                if len(vertexDictionary.keys()) > 0:
                    vertexDictionaries.append(vertexDictionary)
                    vertexSelectors.append([aVertex.X(), aVertex.Y(), aVertex.Z()])
            return [vertexApertures, vertexDictionaries, vertexSelectors]
        
        def apertureDicts(apertureList):
            apertureDicts = []
            for anAperture in apertureList:
                apertureData = {}
                apertureData['brep'] = anAperture.String()
                apertureData['dictionary'] = Topology.Dictionary(anAperture)
                apertureDicts.append(apertureData)
            return apertureDicts

        def subTopologyDicts(dicts, selectors):
            returnDicts = []
            for i in range(len(dicts)):
                data = {}
                data['dictionary'] = dicts[i]
                data['selector'] = selectors[i]
                returnDicts.append(data)
            return returnDicts

        def getTopologyData(topology, tol):
            returnDict = {}
            brep = topology.String()
            dictionary = Topology.Dictionary(topology)
            returnDict['brep'] = brep
            returnDict['dictionary'] = dictionary
            cellApertures, cellDictionaries, cellSelectors = cellAperturesAndDictionaries(topology, tol)
            faceApertures, faceDictionaries, faceSelectors = faceAperturesAndDictionaries(topology, tol)
            edgeApertures, edgeDictionaries, edgeSelectors = edgeAperturesAndDictionaries(topology, tol)
            vertexApertures, vertexDictionaries, vertexSelectors = vertexAperturesAndDictionaries(topology, tol)
            returnDict['cellApertures'] = apertureDicts(cellApertures)
            returnDict['faceApertures'] = apertureDicts(faceApertures)
            returnDict['edgeApertures'] = apertureDicts(edgeApertures)
            returnDict['vertexApertures'] = apertureDicts(vertexApertures)
            returnDict['cellDictionaries'] = subTopologyDicts(cellDictionaries, cellSelectors)
            returnDict['faceDictionaries'] = subTopologyDicts(faceDictionaries, faceSelectors)
            returnDict['edgeDictionaries'] = subTopologyDicts(edgeDictionaries, edgeSelectors)
            returnDict['vertexDictionaries'] = subTopologyDicts(vertexDictionaries, vertexSelectors)
            return returnDict

        if not (isinstance(topologies,list)):
            topologies = [topologies]
        # filePath = item[1]
        # tol = item[2]
        # Make sure the file extension is .json
        ext = filePath[len(filePath)-5:len(filePath)]
        if ext.lower() != ".json":
            filePath = filePath+".json"
        f = None
        try:
            if overwrite == True:
                f = open(filePath, "w")
            else:
                f = open(filePath, "x") # Try to create a new File
        except:
            raise Exception("Error: Could not create a new file at the following location: "+filePath)
        if (f):
            jsondata = []
            for topology in topologies:
                jsondata.append(getTopologyData(topology, tolerance))
            json.dump(jsondata, f, indent=4, sort_keys=True)
            f.close()    
            return True
        return False

    
    @staticmethod
    def ExportToJSONMK2(topologies, folderPath, fileName, overwrite=False, tolerance=0.0001):
        """
        Export the input list of topologies to a JSON file

        Parameters
        ----------
        topologies : list
            The input list of topologies.
        folderPath : list
            The path to the folder containing the json file and brep files.
        filePath : str
            The path to the JSON file.
        overwrite : bool , optional
            If set to True, any existing file will be overwritten. The default is False.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        bool
            The status of exporting the JSON file. If True, the operation was successful. Otherwise, it was unsuccesful.

        """

        def cellAperturesAndDictionaries(topology, tol):
            if topology.Type() <= 32:
                return [[],[],[]]
            cells = []
            try:
                _ = topology.Cells(None, cells)
            except:
                return [[],[],[]]
            cellApertures = []
            cellDictionaries = []
            cellSelectors = []
            for aCell in cells:
                tempApertures = []
                _ = aCell.Apertures(tempApertures)
                for anAperture in tempApertures:
                    cellApertures.append(anAperture)
                cellDictionary = Topology.Dictionary(aCell)
                if len(cellDictionary.keys()) > 0:
                    cellDictionaries.append(cellDictionary)
                    iv = topologic.CellUtility.InternalVertex(aCell, tol)
                    cellSelectors.append([iv.X(), iv.Y(), iv.Z()])
            return [cellApertures, cellDictionaries, cellSelectors]

        def faceAperturesAndDictionaries(topology, tol):
            if topology.Type() <= 8:
                return [[],[],[]]
            faces = []
            try:
                _ = topology.Faces(None, faces)
            except:
                return [[],[],[]]
            faceApertures = []
            faceDictionaries = []
            faceSelectors = []
            for aFace in faces:
                tempApertures = []
                _ = aFace.Apertures(tempApertures)
                for anAperture in tempApertures:
                    faceApertures.append(anAperture)
                faceDictionary = Topology.Dictionary(aFace)
                if len(faceDictionary.keys()) > 0:
                    faceDictionaries.append(faceDictionary)
                    iv = topologic.FaceUtility.InternalVertex(aFace, tol)
                    faceSelectors.append([iv.X(), iv.Y(), iv.Z()])
            return [faceApertures, faceDictionaries, faceSelectors]

        def edgeAperturesAndDictionaries(topology, tol):
            if topology.Type() <= 2:
                return [[],[],[]]
            edges = []
            try:
                _ = topology.Edges(None, edges)
            except:
                return [[],[],[]]
            edgeApertures = []
            edgeDictionaries = []
            edgeSelectors = []
            for anEdge in edges:
                tempApertures = []
                _ = anEdge.Apertures(tempApertures)
                for anAperture in tempApertures:
                    edgeApertures.append(anAperture)
                edgeDictionary = Topology.Dictionary(anEdge)
                if len(edgeDictionary.keys()) > 0:
                    edgeDictionaries.append(edgeDictionary)
                    iv = topologic.EdgeUtility.PointAtParameter(anEdge, 0.5)
                    edgeSelectors.append([iv.X(), iv.Y(), iv.Z()])
            return [edgeApertures, edgeDictionaries, edgeSelectors]

        def vertexAperturesAndDictionaries(topology, tol):
            if topology.Type() <= 1:
                return [[],[],[]]
            vertices = []
            try:
                _ = topology.Vertices(None, vertices)
            except:
                return [[],[],[]]
            vertexApertures = []
            vertexDictionaries = []
            vertexSelectors = []
            for aVertex in vertices:
                tempApertures = []
                _ = aVertex.Apertures(tempApertures)
                for anAperture in tempApertures:
                    vertexApertures.append(anAperture)
                vertexDictionary = Topology.Dictionary(aVertex)
                if len(vertexDictionary.keys()) > 0:
                    vertexDictionaries.append(vertexDictionary)
                    vertexSelectors.append([aVertex.X(), aVertex.Y(), aVertex.Z()])
            return [vertexApertures, vertexDictionaries, vertexSelectors]


        def apertureDicts(apertureList, brepName, folderPath):
            apertureDicts = []
            for index, anAperture in enumerate(apertureList):
                apertureName = brepName+"_aperture_"+str(index+1).zfill(5)
                brepFilePath = os.path.join(folderPath, apertureName+".brep")
                brepFile = open(brepFilePath, "w")
                brepFile.write(anAperture.String())
                brepFile.close()
                apertureData = {}
                apertureData['brep'] = apertureName
                apertureData['dictionary'] = Topology.Dictionary(anAperture)
                apertureDicts.append(apertureData)
            return apertureDicts

        def subTopologyDicts(dicts, selectors):
            returnDicts = []
            for i in range(len(dicts)):
                data = {}
                data['dictionary'] = dicts[i]
                data['selector'] = selectors[i]
                returnDicts.append(data)
            return returnDicts

        def getTopologyData(topology, brepName, folderPath, tol):
            returnDict = {}
            #brep = topology.String()
            dictionary = Topology.Dictionary(topology)
            returnDict['brep'] = brepName
            returnDict['dictionary'] = dictionary
            cellApertures, cellDictionaries, cellSelectors = cellAperturesAndDictionaries(topology, tol)
            faceApertures, faceDictionaries, faceSelectors = faceAperturesAndDictionaries(topology, tol)
            edgeApertures, edgeDictionaries, edgeSelectors = edgeAperturesAndDictionaries(topology, tol)
            vertexApertures, vertexDictionaries, vertexSelectors = vertexAperturesAndDictionaries(topology, tol)
            returnDict['cellApertures'] = apertureDicts(cellApertures, brepName, folderPath)
            returnDict['faceApertures'] = apertureDicts(faceApertures, brepName, folderPath)
            returnDict['edgeApertures'] = apertureDicts(edgeApertures, brepName, folderPath)
            returnDict['vertexApertures'] = apertureDicts(vertexApertures, brepName, folderPath)
            returnDict['cellDictionaries'] = subTopologyDicts(cellDictionaries, cellSelectors)
            returnDict['faceDictionaries'] = subTopologyDicts(faceDictionaries, faceSelectors)
            returnDict['edgeDictionaries'] = subTopologyDicts(edgeDictionaries, edgeSelectors)
            returnDict['vertexDictionaries'] = subTopologyDicts(vertexDictionaries, vertexSelectors)
            return returnDict
        
        if not (isinstance(topologies,list)):
            topologies = [topologies]
        # folderPath = item[1]
        # fileName = item[2]
        # tol = item[3]
        # Make sure the file extension is .json
        ext = fileName[len(fileName)-5:len(fileName)]
        if ext.lower() != ".json":
            fileName = fileName+".json"
        jsonFile = None
        jsonFilePath = os.path.join(folderPath, fileName)
        try:
            if overwrite == True:
                jsonFile = open(jsonFilePath, "w")
            else:
                jsonFile = open(jsonFilePath, "x") # Try to create a new File
        except:
            raise Exception("Error: Could not create a new file at the following location: "+jsonFilePath)
        if (jsonFilePath):
            jsondata = []
            for index, topology in enumerate(topologies):
                brepName = "topology_"+str(index+1).zfill(5)
                brepFilePath = os.path.join(folderPath, brepName+".brep")
                brepFile = open(brepFilePath, "w")
                brepFile.write(topology.String())
                brepFile.close()
                jsondata.append(getTopologyData(topology, brepName, folderPath, tolerance))
            json.dump(jsondata, jsonFile, indent=4, sort_keys=True)
            jsonFile.close()    
            return True
        return False
    
    @staticmethod
    def Filter(topologies, topologyType="vertex", searchType="any", key=None, value=None):
        """
        Filters the input list of topologies based on the input parameters.

        Parameters
        ----------
        topologies : list
            The input list of topologies.
        topologyType : str , optional
            The type of topology to filter by. This can be one of "vertex", "edge", "wire", "face", "shell", "cell", "cellcomplex", or "cluster". It is case insensitive. The default is "vertex".
        searchType : str , optional
            The type of search query to conduct in the topology's dictionary. This can be one of "any", "equal to", "contains", "starts with", "ends with", "not equal to", "does not contain". The default is "any".
        key : str , optional
            The dictionary key to search within. The default is None which means it will filter by topology type only.
        value : str , optional
            The value to search for at the specified key. The default is None which means it will filter by topology type only.

        Returns
        -------
        list
            The list of filtered topologies.

        """
        from topologicpy.Dictionary import Dictionary

        def listToString(item):
            returnString = ""
            if isinstance(item, list):
                if len(item) < 2:
                    return str(item[0])
                else:
                    returnString = item[0]
                    for i in range(1, len(item)):
                        returnString = returnString+str(item[i])
            return returnString
        
        filteredTopologies = []
        otherTopologies = []
        for aTopology in topologies:
            if not aTopology:
                continue
            if (topologyType.lower() == "any") or (aTopology.GetTypeAsString().lower() == topologyType.lower()):
                if value == "" or key == "":
                    filteredTopologies.append(aTopology)
                else:
                    if isinstance(value, list):
                        value.sort()
                        value = str(value)
                    value.replace("*",".+")
                    value = value.lower()
                    d = Topology.Dictionary(aTopology)
                    v = Dictionary.ValueAtKey(d, key)
                    if v != None:
                        v = v.lower()
                        if searchType.lower() == "equal to":
                            searchResult = (value == v)
                        elif searchType.lower() == "contains":
                            searchResult = (value in v)
                        elif searchType.lower() == "starts with":
                            searchResult = (value == v[0: len(value)])
                        elif searchType.lower() == "ends with":
                            searchResult = (value == v[len(v)-len(value):len(v)])
                        elif searchType.lower() == "not equal to":
                            searchResult = not (value == v)
                        elif searchType.lower() == "does not contain":
                            searchResult = not (value in v)
                        else:
                            searchResult = False
                        if searchResult:
                            filteredTopologies.append(aTopology)
                        else:
                            otherTopologies.append(aTopology)
                    else:
                        otherTopologies.append(aTopology)
            else:
                otherTopologies.append(aTopology)
        return [filteredTopologies, otherTopologies]

    def Flatten(topology, origin=None, vector=[0,0,1]):
        """
        Flattens the input topology such that the input origin is located at the world origin and the input topology is rotated such that the input vector is pointed in the positive Z axis.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        origin : topologic.Vertex , optional
            The input origin. If set to None, the center of mass of the input topology will be place the world origin. The default is None.
        vector : list , optional
            The input direction vector. The input topology will be rotated such that this vector is pointed in the positive Z axis.

        Returns
        -------
        topologic.Topology
            The flattened topology.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        if not isinstance(topology, topologic.Topology):
            return None
        world_origin = Vertex.ByCoordinates(0,0,0)
        if origin == None:
            origin = topology.CenterOfMass()
        x1 = origin.X()
        y1 = origin.Y()
        z1 = origin.Z()
        x2 = origin.X() + vector[0]
        y2 = origin.Y() + vector[1]
        z2 = origin.Z() + vector[2]
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1    
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        phi = math.degrees(math.atan2(dy, dx)) # Rotation around Y-Axis
        if dist < 0.0001:
            theta = 0
        else:
            theta = math.degrees(math.acos(dz/dist)) # Rotation around Z-Axis
        flat_topology = Topology.Translate(topology, -origin.X(), -origin.Y(), -origin.Z())
        flat_topology = Topology.Rotate(flat_topology, world_origin, 0, 0, 1, -phi)
        flat_topology = Topology.Rotate(flat_topology, world_origin, 0, 1, 0, -theta)
        return flat_topology
    
    @staticmethod
    def Geometry(topology):
        """
        Returns the geometry (mesh data format) of the input topology as a dictionary of vertices, edges, and faces.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.

        Returns
        -------
        dict
            A dictionary containing the vertices, edges, and faces data. The keys found in the dictionary are "vertices", "edges", and "faces".

        """
        
        def getSubTopologies(topology, subTopologyClass):
            topologies = []
            if subTopologyClass == topologic.Vertex:
                _ = topology.Vertices(None, topologies)
            elif subTopologyClass == topologic.Edge:
                _ = topology.Edges(None, topologies)
            elif subTopologyClass == topologic.Wire:
                _ = topology.Wires(None, topologies)
            elif subTopologyClass == topologic.Face:
                _ = topology.Faces(None, topologies)
            elif subTopologyClass == topologic.Shell:
                _ = topology.Shells(None, topologies)
            elif subTopologyClass == topologic.Cell:
                _ = topology.Cells(None, topologies)
            elif subTopologyClass == topologic.CellComplex:
                _ = topology.CellComplexes(None, topologies)
            return topologies

        def triangulateFace(face):
            faceTriangles = []
            for i in range(0,5,1):
                try:
                    _ = topologic.FaceUtility.Triangulate(face, float(i)*0.1, faceTriangles)
                    return faceTriangles
                except:
                    continue
            faceTriangles.append(face)
            return faceTriangles

        vertices = []
        edges = []
        faces = []
        if topology == None:
            return [None, None, None]
        topVerts = []
        if (topology.Type() == 1): #input is a vertex, just add it and process it
            topVerts.append(topology)
        else:
            _ = topology.Vertices(None, topVerts)
        for aVertex in topVerts:
            try:
                vertices.index([aVertex.X(), aVertex.Y(), aVertex.Z()]) # Vertex already in list
            except:
                vertices.append([aVertex.X(), aVertex.Y(), aVertex.Z()]) # Vertex not in list, add it.
        topEdges = []
        if (topology.Type() == 2): #Input is an Edge, just add it and process it
            topEdges.append(topology)
        elif (topology.Type() > 2):
            _ = topology.Edges(None, topEdges)
        for anEdge in topEdges:
            e = []
            sv = anEdge.StartVertex()
            ev = anEdge.EndVertex()
            try:
                svIndex = vertices.index([sv.X(), sv.Y(), sv.Z()])
            except:
                vertices.append([sv.X(), sv.Y(), sv.Z()])
                svIndex = len(vertices)-1
            try:
                evIndex = vertices.index([ev.X(), ev.Y(), ev.Z()])
            except:
                vertices.append([ev.X(), ev.Y(), ev.Z()])
                evIndex = len(vertices)-1
            e.append(svIndex)
            e.append(evIndex)
            if ([e[0], e[1]] not in edges) and ([e[1], e[0]] not in edges):
                edges.append(e)
        topFaces = []
        if (topology.Type() == 8): # Input is a Face, just add it and process it
            topFaces.append(topology)
        elif (topology.Type() > 8):
            _ = topology.Faces(None, topFaces)
        for aFace in topFaces:
            ib = []
            _ = aFace.InternalBoundaries(ib)
            if(len(ib) > 0):
                triFaces = triangulateFace(aFace)
                for aTriFace in triFaces:
                    wire = aTriFace.ExternalBoundary()
                    faceVertices = getSubTopologies(wire, topologic.Vertex)
                    f = []
                    for aVertex in faceVertices:
                        try:
                            fVertexIndex = vertices.index([aVertex.X(), aVertex.Y(), aVertex.Z()])
                        except:
                            vertices.append([aVertex.X(), aVertex.Y(), aVertex.Z()])
                            fVertexIndex = len(vertices)-1
                        f.append(fVertexIndex)
                    faces.append(f)
            else:
                wire =  aFace.ExternalBoundary()
                #wire = topologic.WireUtility.RemoveCollinearEdges(wire, 0.1) #This is an angle Tolerance
                faceVertices = getSubTopologies(wire, topologic.Vertex)
                f = []
                for aVertex in faceVertices:
                    try:
                        fVertexIndex = vertices.index([aVertex.X(), aVertex.Y(), aVertex.Z()])
                    except:
                        vertices.append([aVertex.X(), aVertex.Y(), aVertex.Z()])
                        fVertexIndex = len(vertices)-1
                    f.append(fVertexIndex)
                faces.append(f)
        if len(vertices) == 0:
            vertices = [[]]
        if len(edges) == 0:
            edges = [[]]
        if len(faces) == 0:
            faces = [[]]
        return {"vertices":vertices, "edges":edges, "faces":faces}

    @staticmethod
    def HighestType(topology):
        """
        Returns the highest topology type found in the input topology.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.

        Returns
        -------
        int
            The highest type found in the input topology.

        """
        from topologicpy.Cluster import Cluster
        if (topology.Type() == topologic.Cluster.Type()):
            return Cluster.HighestType(topology)
        else:
            return(topology.Type())

    @staticmethod
    def InternalVertex(topology, tolerance=0.0001):
        """
        Returns an vertex guaranteed to be inside the input topology.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        tolerance : float , ptional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Vertex
            A vertex guaranteed to be inside the input topology.

        """
        if not isinstance(topology, topologic.Topology):
            return None
        vst = None
        classType = topology.Type()
        if classType == 64: #CellComplex
            tempCells = []
            _ = topology.Cells(tempCells)
            tempCell = tempCells[0]
            vst = topologic.CellUtility.InternalVertex(tempCell, tolerance)
        elif classType == 32: #Cell
            vst = topologic.CellUtility.InternalVertex(topology, tolerance)
        elif classType == 16: #Shell
            tempFaces = []
            _ = topology.Faces(None, tempFaces)
            tempFace = tempFaces[0]
            vst = topologic.FaceUtility.InternalVertex(tempFace, tolerance)
        elif classType == 8: #Face
            vst = topologic.FaceUtility.InternalVertex(topology, tolerance)
        elif classType == 4: #Wire
            if topology.IsClosed():
                internalBoundaries = []
                tempFace = topologic.Face.ByExternalInternalBoundaries(topology, internalBoundaries)
                vst = topologic.FaceUtility.InternalVertex(tempFace, tolerance)
            else:
                tempEdges = []
                _ = topology.Edges(None, tempEdges)
                vst = topologic.EdgeUtility.PointAtParameter(tempEdges[0], 0.5)
        elif classType == 2: #Edge
            vst = topologic.EdgeUtility.PointAtParameter(topology, 0.5)
        elif classType == 1: #Vertex
            vst = topology
        else:
            vst = topology.Centroid()
        return vst

    @staticmethod
    def IsInside(topology, vertex, tolerance=0.0001):
        """
        Returns True if the input vertex is inside the input topology. Returns False otherwise.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        vertex : topologic.Vertex
            The input Vertex.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        bool
            True if the input vertex is inside the input topology. False otherwise.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Cell import Cell
        from topologicpy.Shell import Shell
        from topologicpy.CellComplex import CellComplex
        from topologicpy.Cluster import Cluster
        if not isinstance(topology, topologic.Topology):
            return None
        if not isinstance(vertex, topologic.Vertex):
            return None
        is_inside = False
        if topology.Type() == topologic.Vertex.Type():
            try:
                is_inside = (Vertex.Distance(vertex, topology) <= tolerance)
            except:
                is_inside = False
            return is_inside
        elif topology.Type() == topologic.Edge.Type():
            u = Edge.ParameterAtVertex(topology, vertex)
            d = Vertex.Distance(vertex, topology)
            if u:
                is_inside = (0 <= u <= 1) and (d <= tolerance)              
            else:
                is_inside = False
            return is_inside
        elif topology.Type() == topologic.Wire.Type():
            edges = Wire.Edges(topology)
            for edge in edges:
                is_inside = (Vertex.Distance(vertex, edge) <= tolerance)
                if is_inside:
                    return is_inside
        elif topology.Type() == topologic.Face.Type():
            return Face.IsInside(topology, vertex, tolerance)
        elif topology.Type() == topologic.Shell.Type():
            faces = Shell.Faces(topology)
            for face in faces:
                is_inside = Face.IsInside(face, vertex, tolerance)
                if is_inside:
                    return is_inside
        elif topology.Type() == topologic.Cell.Type():
            return Cell.IsInside(topology, vertex, tolerance)
        elif topology.Type() == topologic.CellComplex.Type():
            cells = CellComplex.Cells(topology)
            for cell in cells:
                is_inside = Cell.IsInside(cell, vertex, tolerance)
                if is_inside:
                    return is_inside
        elif topology.Type() == topologic.Cluster.Type():
            cells = Cluster.Cells(topology)
            faces = Cluster.Faces(topology)
            edges = Cluster.Edges(topology)
            vertices = Cluster.Vertices(topology)
            subTopologies = []
            if isinstance(cells, list):
                subTopologies += cells
            if isinstance(faces, list):
                subTopologies += faces
            if isinstance(edges, list):
                subTopologies += edges
            if isinstance(vertices, list):
                subTopologies += vertices
            for subTopology in subTopologies:
                is_inside = Topology.IsInside(subTopology, vertex, tolerance)
                if is_inside:
                    return is_inside
        return False

    @staticmethod
    def IsPlanar(topology, tolerance=0.0001):
        """
        Returns True if all the vertices of the input topology are co-planar. Returns False otherwise.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        bool
            True if all the vertices of the input topology are co-planar. False otherwise.

        """
        
        def isOnPlane(v, plane, tolerance):
            x, y, z = v
            a, b, c, d = plane
            if math.fabs(a*x + b*y + c*z + d) <= tolerance:
                return True
            return False

        def plane(v1, v2, v3):
            a1 = v2.X() - v1.X() 
            b1 = v2.Y() - v1.Y() 
            c1 = v2.Z() - v1.Z() 
            a2 = v3.X() - v1.X() 
            b2 = v3.Y() - v1.Y() 
            c2 = v3.Z() - v1.Z() 
            a = b1 * c2 - b2 * c1 
            b = a2 * c1 - a1 * c2 
            c = a1 * b2 - b1 * a2 
            d = (- a * v1.X() - b * v1.Y() - c * v1.Z())
            return [a,b,c,d]

        if not isinstance(topology, topologic.Topology):
            return None
        vertices = []
        _ = topology.Vertices(None, vertices)

        result = True
        if len(vertices) <= 3:
            result = True
        else:
            p = plane(vertices[0], vertices[1], vertices[2])
            for i in range(len(vertices)):
                if isOnPlane([vertices[i].X(), vertices[i].Y(), vertices[i].Z()], p, tolerance) == False:
                    result = False
                    break
        return result
    
    @staticmethod
    def IsSame(topologyA, topologyB):
        """
        Returns True of the input topologies are the same topology. Returns False otherwise.

        Parameters
        ----------
        topologyA : topologic.Topology
            The first input topology.
        topologyB : topologic.Topology
            The second input topology.

        Returns
        -------
        bool
            True of the input topologies are the same topology. False otherwise.

        """
        return topologic.Topology.IsSame(topologyA, topologyB)
    
    @staticmethod
    def MergeAll(topologies):
        """
        Merge all the input topologies.

        Parameters
        ----------
        topologies : list
            The list of input topologies.

        Returns
        -------
        sets : list
            The sets of merged topologies.

        """
        resultTopology = topologies[0]
        for i in range(1, len(topologies)):
            resultTopology = resultTopology.Union(topologies[i])
        cells = []
        _ = resultTopology.Cells(None, cells)
        unused = []
        for i in range(len(topologies)):
            unused.append(True)
        sets = []
        for i in range(len(cells)):
            sets.append([])
        for i in range(len(topologies)):
            if unused[i]:
                iv = topologic.CellUtility.InternalVertex(topologies[i], 0.0001)
                for j in range(len(cells)):
                    if (topologic.CellUtility.Contains(cells[j], iv, 0.0001) == 0):
                        sets[j].append(topologies[i])
                        unused[i] = False
        return sets
    
    @staticmethod
    def OCCTShape(topology):
        """
        Returns the occt shape of the input topology.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.

        Returns
        -------
        topologic.TopoDS_Shape
            The OCCT Shape.

        """
        return topology.GetOcctShape()
    
    def Orient(topology, origin=None, dirA=[0,0,1], dirB=[0,0,1], tolerance=0.0001):
        """
        Orients the input topology such that the input such that the input dirA vector is parallel to the input dirB vector.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        origin : topologic.Vertex , optional
            The input origin. If set to None, the center of mass of the input topology will be used to locate the input topology. The default is None.
        dirA : list , optional
            The first input direction vector. The input topology will be rotated such that this vector is parallel to the input dirB vector. The default is [0,0,1].
        dirB : list , optional
            The target direction vector. The input topology will be rotated such that the input dirA vector is parallel to this vector. The default is [0,0,1].
        tolerance : float , optional
            The desired tolerance. The default is 0.0001

        Returns
        -------
        topologic.Topology
            The flattened topology.

        """
        from topologicpy.Vertex import Vertex
        if not isinstance(topology, topologic.Topology):
            return None
        if not isinstance(origin, topologic.Vertex):
            origin = topology.CenterOfMass()
        topology = Topology.Flatten(topology, origin=origin, vector=dirA)
        x1 = origin.X()
        y1 = origin.Y()
        z1 = origin.Z()
        x2 = origin.X() + dirB[0]
        y2 = origin.Y() + dirB[1]
        z2 = origin.Z() + dirB[2]
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1    
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        phi = math.degrees(math.atan2(dy, dx)) # Rotation around Y-Axis
        if dist < tolerance:
            theta = 0
        else:
            theta = math.degrees(math.acos(dz/dist)) # Rotation around Z-Axis
        world_origin = Vertex.ByCoordinates(0,0,0)
        returnTopology = Topology.Rotate(topology, world_origin, 0, 0, 1, -phi)
        returnTopology = Topology.Rotate(returnTopology, world_origin, 0, 1, 0, -theta)
        returnTopology = Topology.Place(returnTopology, world_origin, origin)
        return returnTopology

    @staticmethod
    def Place(topology, oldLocation=None, newLocation=None):
        """
        Places the input topology at the specified location.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        oldLocation : topologic.Vertex , optional
            The old location to use as the origin of the movement. If set to None, the centroid of the input topology is used. The default is None.
        newLocation : topologic.Vertex , optional
            The new location at which to place the topology. If set to None, the world origin (0,0,0) is used. The default is None.

        Returns
        -------
        topologic.Topology
            The placed topology.

        """
        from topologicpy.Vertex import Vertex
        if not isinstance(topology, topologic.Topology):
            return None
        if not isinstance(oldLocation, topologic.Vertex):
            oldLocation = Topology.Centroid(topology)
        if not isinstance(newLocation, topologic.Vertex):
            newLocation = Vertex.ByCoordinates(0,0,0)

        x = newLocation.X() - oldLocation.X()
        y = newLocation.Y() - oldLocation.Y()
        z = newLocation.Z() - oldLocation.Z()
        newTopology = None
        try:
            newTopology = Topology.Translate(topology, x, y, z)
        except:
            print("ERROR: (Topologic>TopologyUtility.Place) operation failed. Returning None.")
            newTopology = None
        return newTopology
    
    def RelevantSelector(topology, tolerance=0.0001):
        """
        Returns the relevant selector (vertex) of the input topology

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Vertex
            The relevant selector.

        """
        if topology.Type() == topologic.Vertex.Type():
            return topology
        elif topology.Type() == topologic.Edge.Type():
            return topologic.EdgeUtility.PointAtParameter(topology, 0.5)
        elif topology.Type() == topologic.Face.Type():
            return topologic.FaceUtility.InternalVertex(topology, tolerance)
        elif topology.Type() == topologic.Cell.Type():
            return topologic.CellUtility.InternalVertex(topology, tolerance)
        else:
            return topology.CenterOfMass()

    @staticmethod
    def RemoveCollinearEdges(topology, angTolerance=0.1, tolerance=0.0001):
        """
        Removes the collinear edges of the input topology

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        angTolerance : float , optional
            The desired angular tolerance. The default is 0.1.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Topology
            The input topology with the collinear edges removed.

        """
        
        def toDegrees(ang):
            import math
            return ang * 180 / math.pi

        # From https://gis.stackexchange.com/questions/387237/deleting-collinear-vertices-from-polygon-feature-class-using-arcpy
        def are_collinear(v2, wire, tolerance=0.5):
            edges = []
            _ = v2.Edges(wire, edges)
            if len(edges) == 2:
                ang = toDegrees(topologic.EdgeUtility.AngleBetween(edges[0], edges[1]))
                if -tolerance <= ang <= tolerance:
                    return True
                else:
                    return False
            else:
                raise Exception("Topology.RemoveCollinearEdges - Error: This method only applies to manifold closed wires")

        #----------------------------------------------------------------------
        def get_redundant_vertices(vertices, wire, angTol):
            """get redundant vertices from a line shape vertices"""
            indexes_of_vertices_to_remove = []
            start_idx, middle_index, end_index = 0, 1, 2
            for i in range(len(vertices)):
                v1, v2, v3 = vertices[start_idx:end_index + 1]
                if are_collinear(v2, wire, angTol):
                    indexes_of_vertices_to_remove.append(middle_index)

                start_idx += 1
                middle_index += 1
                end_index += 1
                if end_index == len(vertices):
                    break
            if are_collinear(vertices[0], wire, angTol):
                indexes_of_vertices_to_remove.append(0)
            return indexes_of_vertices_to_remove

        def processWire(wire, angTol):
            vertices = []
            _ = wire.Vertices(None, vertices)
            redundantIndices = get_redundant_vertices(vertices, wire, angTol)
            # Check if first vertex is also collinear
            if are_collinear(vertices[0], wire, angTol):
                redundantIndices.append(0)
            cleanedVertices = []
            for i in range(len(vertices)):
                if (i in redundantIndices) == False:
                    cleanedVertices.append(vertices[i])
            edges = []
            for i in range(len(cleanedVertices)-1):
                edges.append(topologic.Edge.ByStartVertexEndVertex(cleanedVertices[i], cleanedVertices[i+1]))
            edges.append(topologic.Edge.ByStartVertexEndVertex(cleanedVertices[-1], cleanedVertices[0]))
            return topologic.Wire.ByEdges(edges)
            #return topologic.WireUtility.RemoveCollinearEdges(wire, angTol) #This is an angle Tolerance
        
        returnTopology = topology
        t = topology.Type()
        if (t == 1) or (t == 2) or (t == 128): #Vertex or Edge or Cluster, return the original topology
            return returnTopology
        elif (t == 4): #wire
            returnTopology = processWire(topology, angTolerance)
            return returnTopology
        elif (t == 8): #Face
            extBoundary = processWire(topology.ExternalBoundary(), angTolerance)
            internalBoundaries = []
            _ = topology.InternalBoundaries(internalBoundaries)
            cleanIB = []
            for ib in internalBoundaries:
                cleanIB.append(processWire(ib, angTolerance))
            try:
                returnTopology = topologic.Face.ByExternalInternalBoundaries(extBoundary, cleanIB)
            except:
                returnTopology = topology
            return returnTopology
        faces = []
        _ = topology.Faces(None, faces)
        stl_final_faces = []
        for aFace in faces:
            extBoundary = processWire(aFace.ExternalBoundary(), angTolerance)
            internalBoundaries = []
            _ = aFace.InternalBoundaries(internalBoundaries)
            cleanIB = []
            for ib in internalBoundaries:
                cleanIB.append(processWire(ib, angTolerance))
            stl_final_faces.append(topologic.Face.ByExternalInternalBoundaries(extBoundary, cleanIB))
        returnTopology = topology
        if t == 16: # Shell
            try:
                returnTopology = topologic.Shell.ByFaces(stl_final_faces, tolerance)
            except:
                returnTopology = topology
        elif t == 32: # Cell
            try:
                returnTopology = topologic.Cell.ByFaces(stl_final_faces, tolerance)
            except:
                returnTopology = topology
        elif t == 64: #CellComplex
            try:
                returnTopology = topologic.CellComplex.ByFaces(stl_final_faces, tolerance)
            except:
                returnTopology = topology
        return returnTopology

    
    @staticmethod
    def RemoveContent(topology, contents):
        """
        Removes the input content list from the input topology

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        contentList : list
            The input list of contents.

        Returns
        -------
        topologic.Topology
            The input topology with the input list of contents removed.

        """
        if isinstance(contents, list) == False:
            contents = [contents]
        return topology.RemoveContents(contents)
    
    @staticmethod
    def RemoveCoplanarFaces(topology, angTolerance=0.1, tolerance=0.0001):
        """
        Removes coplanar faces in the input topology

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        angTolerance : float , optional
            The desired angular tolerance for removing coplanar faces. The default is 0.1.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Topology
            The input topology with coplanar faces merged into one face.

        """
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Shell import Shell
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        t = topology.Type()
        if (t == 1) or (t == 2) or (t == 4) or (t == 8) or (t == 128):
            return topology
        clusters = Topology.ClusterFaces(topology, tolerance=tolerance)
        faces = []
        for aCluster in clusters:
            aCluster = Cluster.SelfMerge(aCluster)
            shells = Cluster.Shells(aCluster)
            tempFaces = Cluster.Faces(aCluster)
            if not shells or len(shells) < 1:
                if isinstance(aCluster, topologic.Shell):
                    shells = [aCluster]
                else:
                    tempFaces = Topology.SubTopologies(aCluster, subTopologyType="Face")
                    temp_shell = Shell.ByFaces(tempFaces)
                    if isinstance(temp_shell, list):
                        shells = temp_shell
                    else:
                        shells = [temp_shell]
            if len(shells) > 1:
                for aShell in shells:
                    junk_faces = Shell.Faces(aShell)
                    if len(junk_faces) > 2:
                        aFace = Face.ByShell(aShell, angTolerance)
                    aFace = Face.ByShell(aShell, angTolerance)
                    if isinstance(aFace, topologic.Face):
                        faces.append(aFace)
                    else:
                        for f in Shell.Faces(aShell):
                            faces.append(f)
                    for tempFace in tempFaces:
                        isInside = False
                        for tempShell in shells:
                            if Topology.IsInside(tempShell, Face.InternalVertex(tempFace), tolerance=tolerance):
                                isInside = True
                                break;
                        if not isInside:
                            faces.append(tempFace)
            else:
                cFaces = Cluster.Faces(aCluster)
                if cFaces:
                    for aFace in cFaces:
                        faces.append(aFace)
        returnTopology = None
        finalFaces = []
        for aFace in faces:
            eb = Face.ExternalBoundary(aFace)
            ibList = Face.InternalBoundaries(aFace)
            try:
                eb = Wire.RemoveCollinearEdges(eb, angTolerance=angTolerance)
            except:
                pass
            finalIbList = []
            if ibList:
                for ib in ibList:
                    temp_ib = ib
                    try:
                        temp_ib = Wire.RemoveCollinearEdges(ib, angTolerance=angTolerance)
                    except:
                        pass
                    finalIbList.append(temp_ib)
            finalFaces.append(Face.ByWires(eb, finalIbList))
        faces = finalFaces
        if t == 16:
            returnTopology = topologic.Shell.ByFaces(faces, tolerance)
            if not returnTopology:
                returnTopology = topologic.Cluster.ByTopologies(faces, False)
        elif t == 32:
            returnTopology = topologic.Cell.ByFaces(faces, tolerance)
            if not returnTopology:
                returnTopology = topologic.Shell.ByFaces(faces, tolerance)
            if not returnTopology:
                returnTopology = topologic.Cluster.ByTopologies(faces, False)
        elif t == 64:
            returnTopology = topologic.CellComplex.ByFaces(faces, tolerance, False)
            if not returnTopology:
                returnTopology = topologic.Cell.ByFaces(faces, tolerance)
            if not returnTopology:
                returnTopology = topologic.Shell.ByFaces(faces, tolerance)
            if not returnTopology:
                returnTopology = topologic.Cluster.ByTopologies(faces, False)
        return returnTopology

    
    @staticmethod
    def Rotate(topology, origin=None, x=0, y=0, z=1, degree=0):
        """
        Rototates the input topology

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        origin : topologic.Vertex , optional
            The origin (center) of the rotation. If set to None, the world origin (0,0,0) is used. The default is None.
        x : float , optional
            The 'x' component of the rotation axis. The default is 0.
        y : float , optional
            The 'y' component of the rotation axis. The default is 0.
        z : float , optional
            The 'z' component of the rotation axis. The default is 0.
        degree : float , optional
            The angle of rotation in degrees. The default is 0.

        Returns
        -------
        topologic.Topology
            The rotated topology.

        """
        from topologicpy.Vertex import Vertex
        if not isinstance(topology, topologic.Topology):
            return None
        if not origin:
            origin = Vertex.ByCoordinates(0,0,0)
        if not isinstance(origin, topologic.Vertex):
            return None
        return topologic.TopologyUtility.Rotate(topology, origin, x, y, z, degree)
    
    @staticmethod
    def Scale(topology, origin=None, x=1, y=1, z=1):
        """
        Scales the input topology

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        origin : topologic.Vertex , optional
            The origin (center) of the scaling. If set to None, the world origin (0,0,0) is used. The default is None.
        x : float , optional
            The 'x' component of the scaling factor. The default is 1.
        y : float , optional
            The 'y' component of the scaling factor. The default is 1.
        z : float , optional
            The 'z' component of the scaling factor. The default is 1..

        Returns
        -------
        topologic.Topology
            The scaled topology.

        """
        from topologicpy.Vertex import Vertex
        if not isinstance(topology, topologic.Topology):
            return None
        if not origin:
            origin = Vertex.ByCoordinates(0,0,0)
        if not isinstance(origin, topologic.Vertex):
            return None
        newTopology = None
        try:
            newTopology = topologic.TopologyUtility.Scale(topology, origin, x, y, z)
        except:
            print("ERROR: (Topologic>TopologyUtility.Scale) operation failed. Returning None.")
            newTopology = None
        return newTopology

    
    @staticmethod
    def SelectSubTopology(topology, selector, subTopologyType="vertex"):
        """
        Returns the subtopology within the input topology based on the input selector and the subTopologyType.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        selector : topologic.Vertex
            A vertex located on the desired subtopology.
        subTopologyType : str , optional.
            The desired subtopology type. This can be of "vertex", "edge", "wire", "face", "shell", "cell", or "cellcomplex". It is case insensitive. The default is "vertex".

        Returns
        -------
        topologic.Topology
            The selected subtopology.

        """
        if not isinstance(topology, topologic.Topology):
            return None
        if not isinstance(selector, topologic.Vertex):
            return None
        t = 1
        if topologyType.lower() == "vertex":
            t = 1
        elif topologyType.lower() == "edge":
            t = 2
        elif topologyType.lower() == "wire":
            t = 4
        elif topologyType.lower() == "face":
            t = 8
        elif topologyType.lower() == "shell":
            t = 16
        elif topologyType.lower() == "cell":
            t = 32
        elif topologyType.lower() == "cellcomplex":
            t = 64
        return topology.SelectSubtopology(selector, t)

    
    @staticmethod
    def SelfMerge(topology):
        """
        Self merges the input topology to return the most logical topology type given the input data.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.

        Returns
        -------
        topologic.Topology
            The self-merged topology.

        """
        if topology.Type() != 128:
            topology = topologic.Cluster.ByTopologies([topology])
        resultingTopologies = []
        topCC = []
        _ = topology.CellComplexes(None, topCC)
        topCells = []
        _ = topology.Cells(None, topCells)
        topShells = []
        _ = topology.Shells(None, topShells)
        topFaces = []
        _ = topology.Faces(None, topFaces)
        topWires = []
        _ = topology.Wires(None, topWires)
        topEdges = []
        _ = topology.Edges(None, topEdges)
        topVertices = []
        _ = topology.Vertices(None, topVertices)
        if len(topCC) == 1:
            cc = topCC[0]
            ccVertices = []
            _ = cc.Vertices(None, ccVertices)
            if len(topVertices) == len(ccVertices):
                resultingTopologies.append(cc)
        if len(topCC) == 0 and len(topCells) == 1:
            cell = topCells[0]
            ccVertices = []
            _ = cell.Vertices(None, ccVertices)
            if len(topVertices) == len(ccVertices):
                resultingTopologies.append(cell)
        if len(topCC) == 0 and len(topCells) == 0 and len(topShells) == 1:
            shell = topShells[0]
            ccVertices = []
            _ = shell.Vertices(None, ccVertices)
            if len(topVertices) == len(ccVertices):
                resultingTopologies.append(shell)
        if len(topCC) == 0 and len(topCells) == 0 and len(topShells) == 0 and len(topFaces) == 1:
            face = topFaces[0]
            ccVertices = []
            _ = face.Vertices(None, ccVertices)
            if len(topVertices) == len(ccVertices):
                resultingTopologies.append(face)
        if len(topCC) == 0 and len(topCells) == 0 and len(topShells) == 0 and len(topFaces) == 0 and len(topWires) == 1:
            wire = topWires[0]
            ccVertices = []
            _ = wire.Vertices(None, ccVertices)
            if len(topVertices) == len(ccVertices):
                resultingTopologies.append(wire)
        if len(topCC) == 0 and len(topCells) == 0 and len(topShells) == 0 and len(topFaces) == 0 and len(topWires) == 0 and len(topEdges) == 1:
            edge = topEdges[0]
            ccVertices = []
            _ = edge.Vertices(None, ccVertices)
            if len(topVertices) == len(ccVertices):
                resultingTopologies.append(edge)
        if len(topCC) == 0 and len(topCells) == 0 and len(topShells) == 0 and len(topFaces) == 0 and len(topWires) == 0 and len(topEdges) == 0 and len(topVertices) == 1:
            vertex = topVertices[0]
            resultingTopologies.append(vertex)
        if len(resultingTopologies) == 1:
            return resultingTopologies[0]
        return topology.SelfMerge()

    
    @staticmethod
    def SetDictionary(topology, dictionary):
        """
        Sets the input topology's dictionary to the input dictionary

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        dictionary : topologic.Dictionary
            The input dictionary.

        Returns
        -------
        topologic.Topology
            The input topology with the input dictionary set in it.

        """
        if not isinstance(topology, topologic.Topology):
            return None
        if not isinstance(dictionary, topologic.Dictionary):
            return topology
        if len(dictionary.Keys()) < 1:
            return topology
        _ = topology.SetDictionary(dictionary)
        return topology
    
    @staticmethod
    def SharedTopologies(topologyA, topologyB, vertices=True, edges=False, wires=False, faces=False):
        """
        Returns the shared topologies between the two input topologies

        Parameters
        ----------
        topologyA : topologic.Topology
            The first input topology.
        topologyB : topologic.Topology
            The second input topology.
        vertices : bool , optional
            If set to True, shared vertices will be returned. The default is True.
        edges : bool , optional
            If set to True, shared edges will be returned. The default is False.
        wires : bool , optional
            If set to True, shared wires will be returned. The default is False.
        faces : bool , optional
            If set to True, shared faces will be returned. The default is False.

        Returns
        -------
        dict
            A dictionary with the list of vertices, edges, wires, and faces. The keys are "vertices", "edges", "wires", and "faces".

        """
        vOutput = []
        eOutput = []
        wOutput = []
        fOutput = []
        if vertices:
            _ = topologyA.SharedTopologies(topologyB, 1, vOutput)
        if edges:
            _ = topologyA.SharedTopologies(topologyB, 2, eOutput)
        if wires:
            _ = topologyA.SharedTopologies(topologyB, 4, wOutput)
        if faces:
            _ = topologyA.SharedTopologies(topologyB, 8, fOutput)
        return {"vertices":vOutput, "edges":eOutput, "wires":wOutput, "faces":fOutput}

    
    @staticmethod
    def SortBySelectors(topologies, selectors, exclusive=False, tolerance=0.0001):
        """
        Sorts the input list of topologies according to the input list of selectors.

        Parameters
        ----------
        topologies : list
            The input list of topologies.
        selectors : list
            The input list of selectors (vertices).
        exclusive : bool , optional
            If set to True only one selector can be used to select on topology. The default is False.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        dict
            A dictionary containing the list of sorted and unsorted topologies. The keys are "sorted" and "unsorted".

        """
        usedTopologies = []
        sortedTopologies = []
        unsortedTopologies = []
        for i in range(len(topologies)):
            usedTopologies.append(0)
        
        for i in range(len(selectors)):
            found = False
            for j in range(len(topologies)):
                if usedTopologies[j] == 0:
                    if Topology.IsInside(topologies[j], selectors[i], tolerance):
                        sortedTopologies.append(topologies[j])
                        if exclusive == True:
                            usedTopologies[j] = 1
                        found = True
                        break
            if found == False:
                sortedTopologies.append(None)
        for i in range(len(usedTopologies)):
            if usedTopologies[i] == 0:
                unsortedTopologies.append(topologies[i])
        return {"sorted":sortedTopologies, "unsorted":unsortedTopologies}
    
    @staticmethod
    def Spin(topology, origin=None, triangulate=True, dirX=0, dirY=0, dirZ=1, degree=360, sides=16,
                     tolerance=0.0001):
        """
        Spins the input topology around an axis to create a new topology.See https://en.wikipedia.org/wiki/Solid_of_revolution.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        origin : topologic.Vertex
            The origin (center) of the spin.
        triangulate : bool , optional
            If set to True, the result will be triangulated. The default is True.
        dirX : float , optional
            The 'x' component of the axis (vector) of spin. The default is 0.
        dirY : float , optional
            The 'y' component of the axis (vector) of spin. The default is 0.
        dirZ : float , optional
            The 'z' component of the axis (vector) of spin. The default is 1.
        degree : float , optional
            The angle in degrees for the spin. The default is 360.
        sides : int , optional
            The desired number of sides. The default is 16.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Topology
            The spun topology.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Shell import Shell
        from topologicpy.Cell import Cell
        from topologicpy.CellComplex import CellComplex

        if not origin:
            origin = Vertex.ByCoordinates(0,0,0)
        if not isinstance(topology, topologic.Topology):
            return None
        if not isinstance(origin, topologic.Vertex):
            return None
        topologies = []
        unit_degree = degree / float(sides)
        for i in range(sides+1):
            topologies.append(topologic.TopologyUtility.Rotate(topology, origin, dirX, dirY, dirZ, unit_degree*i))
        returnTopology = None
        if topology.Type() == topologic.Vertex.Type():
            returnTopology = Wire.ByVertices(topologies, False)
        elif topology.Type() == topologic.Edge.Type():
            try:
                returnTopology = Shell.ByWires(topologies,triangulate=triangulate, tolerance=tolerance)
            except:
                try:
                    returnTopology = topologic.Cluster.ByTopologies(topologies)
                except:
                    returnTopology = None
        elif topology.Type() == topologic.Wire.Type():
            if topology.IsClosed():
                returnTopology = Cell.ByWires(topologies, triangulate=triangulate, tolerance=tolerance)
                try:
                    returnTopology = Cell.ByWires(topologies, triangulate=triangulate, tolerance=tolerance)
                    returnTopology = Cell.ExternalBoundary(returnTopology)
                    returnTopology = Cell.ByShell(returnTopology)
                except:
                    try:
                        returnTopology = CellComplex.ByWires(topologies, tolerance)
                        try:
                            returnTopology = returnTopology.ExternalBoundary()
                        except:
                            pass
                    except:
                        try:
                            returnTopology = Shell.ByWires(topologies, triangulate=triangulate, tolerance=tolerance)
                        except:
                            try:
                                returnTopology = topologic.Cluster.ByTopologies(topologies)
                            except:
                                returnTopology = None
            else:
                Shell.ByWires(topologies, triangulate=triangulate, tolerance=tolerance)
                try:
                    returnTopology = Shell.ByWires(topologies, triangulate=triangulate, tolerance=tolerance)
                except:
                    try:
                        returnTopology = topologic.Cluster.ByTopologies(topologies)
                    except:
                        returnTopology = None
        elif topology.Type() == topologic.Face.Type():
            external_wires = []
            for t in topologies:
                external_wires.append(topologic.Face.ExternalBoundary(t))
            try:
                returnTopology = CellComplex.ByWires(external_wires, tolerance)
            except:
                try:
                    returnTopology = Shell.ByWires(external_wires, triangulate=triangulate, tolerance=tolerance)
                except:
                    try:
                        returnTopology = topologic.Cluster.ByTopologies(topologies)
                    except:
                        returnTopology = None
        else:
            returnTopology = Topology.SelfMerge(topologic.Cluster.ByTopologies(topologies))
        if returnTopology.Type() == topologic.Shell.Type():
            try:
                new_t = topologic.Cell.ByShell(returnTopology)
                if new_t:
                    returnTopology = new_t
            except:
                pass
        return returnTopology

    
    @staticmethod
    def String(topology):
        """
        Return the BRep string of the input topology.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.

        Returns
        -------
        str
            The BRep string.

        """
        if not isinstance(topology, topologic.Topology):
            return None
        return topology.String()
    
    @staticmethod
    def SubTopologies(topology, subTopologyType="vertex"):
        """
        Returns the subtopologies of the input topology as specified by the subTopologyType input string.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        subTopologyType : str , optional
            The requested subtopology type. This can be one of "vertex", "edge", "wire", "face", "shell", "cell", "cellcomplex", "cluster". It is case insensitive. The default is "vertex".

        Returns
        -------
        list
            The list of subtopologies.

        """
        if not isinstance(topology, topologic.Topology):
            return None
        if topology.GetTypeAsString().lower() == subTopologyType.lower():
            return [topology]
        subTopologies = []
        if subTopologyType.lower() == "vertex":
            _ = topology.Vertices(None, subTopologies)
        elif subTopologyType.lower() == "edge":
            _ = topology.Edges(None, subTopologies)
        elif subTopologyType.lower() == "wire":
            _ = topology.Wires(None, subTopologies)
        elif subTopologyType.lower() == "face":
            _ = topology.Faces(None, subTopologies)
        elif subTopologyType.lower() == "shell":
            _ = topology.Shells(None, subTopologies)
        elif subTopologyType.lower() == "cell":
            _ = topology.Cells(None, subTopologies)
        elif subTopologyType.lower() == "cellcomplex":
            _ = topology.CellComplexes(None, subTopologies)
        elif subTopologyType.lower() == "cluster":
            _ = topology.Clusters(None, subTopologies)
        elif subTopologyType.lower() == "aperture":
            _ = topology.Apertures(subTopologies)
        return subTopologies

    
    @staticmethod
    def SuperTopologies(topology, hostTopology, topologyType = None):
        """
        Returns the supertopologies connected to the input topology.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        hostTopology : topologic.Topology
            The host to topology in which to search for ther supertopologies.
        topologyType : str , optional
            The topology type to search for. This can be any of "edge", "wire", "face", "shell", "cell", "cellcomplex", "cluster". It is case insensitive. If set to None, the immediate supertopology type is searched for. The default is None.

        Returns
        -------
        list
            The list of supertopologies connected to the input topology.

        """
        
        if not isinstance(topology, topologic.Topology):
            return None
        if not isinstance(hostTopology, topologic.Topology):
            return None

        superTopologies = []

        if not topologyType:
            typeID = 2*Topology.TypeID(topology)
        else:
            typeID = Topology.TypeID(topologyType)
        if topology.Type() >= typeID:
            return None #The user has asked for a topology type lower than the input topology
        elif typeID == topologic.Edge.Type():
            topology.Edges(hostTopology, superTopologies)
        elif typeID == topologic.Wire.Type():
            topology.Wires(hostTopology, superTopologies)
        elif typeID == topologic.Face.Type():
            topology.Faces(hostTopology, superTopologies)
        elif typeID == topologic.Shell.Type():
            topology.Shells(hostTopology, superTopologies)
        elif typeID == topologic.Cell.Type():
            topology.Cells(hostTopology, superTopologies)
        elif typeID == topologic.CellComplex.Type():
            topology.CellComplexes(hostTopology, superTopologies)
        elif typeID == topologic.Cluster.Type():
            topology.Cluster(hostTopology, superTopologies)
        else:
            return None
        return superTopologies
    
    @staticmethod
    def SymmetricDifference(topologyA, topologyB, tranDict=False):
        """
        Return the symmetric difference (XOR) of the two input topologies. See https://en.wikipedia.org/wiki/Symmetric_difference.

        Parameters
        ----------
        topologyA : topologic.Topology
            The first input topology.
        topologyB : topologic.Topology
            The second input topology.
        tranDict : bool , opetional
            If set to True, the dictionaries of the input topologies are transferred to the resulting topology.

        Returns
        -------
        topologic.Topology
            The symmetric difference of the two input topologies.

        """
        topologyC = None
        try:
            topologyC = topologyA.XOR(topologyB, tranDict)
        except:
            print("ERROR: (Topologic>Topology.SymmetricDifference) operation failed. Returning None.")
            topologyC = None
        return topologyC
    
    @staticmethod
    def TransferDictionaries(sources, sinks, tolerance=0.0001):
        """
        Transfers the dictionaries from the list of sources to the list of sinks.

        Parameters
        ----------
        sources : list
            The list of topologies from which to transfer the dictionaries.
        sinks : list
            The list of topologies to which to transfer the dictionaries.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        dict
            Returns a dictionary with the lists of sources and sinks. The keys are "sinks" and "sources".

        """
        from topologicpy.Dictionary import Dictionary
        if not isinstance(sources, list):
            return None
        if not isinstance(sinks, list):
            return None
        sources = [x for x in sources if isinstance(x, topologic.Topology)]
        sinks = [x for x in sinks if isinstance(x, topologic.Topology)]
        if len(sources) < 1:
            return None
        if len(sinks) < 1:
            return None
        for sink in sinks:
            sinkKeys = []
            sinkValues = []
            #iv = Topology.RelevantSelector(sink, tolerance)
            j = 1
            for source in sources:
                if Topology.IsInside(sink, source, tolerance):
                    d = Topology.Dictionary(source)
                    if d == None:
                        continue
                    stlKeys = d.Keys()
                    if len(stlKeys) > 0:
                        sourceKeys = d.Keys()
                        for aSourceKey in sourceKeys:
                            if aSourceKey not in sinkKeys:
                                sinkKeys.append(aSourceKey)
                                sinkValues.append("")
                        for i in range(len(sourceKeys)):
                            index = sinkKeys.index(sourceKeys[i])
                            sourceValue = Dictionary.ValueAtKey(d, sourceKeys[i])
                            if sourceValue != None:
                                if sinkValues[index] != "":
                                    if isinstance(sinkValues[index], list):
                                        sinkValues[index].append(sourceValue)
                                    else:
                                        sinkValues[index] = [sinkValues[index], sourceValue]
                                else:
                                    sinkValues[index] = sourceValue
                    break;
            if len(sinkKeys) > 0 and len(sinkValues) > 0:
                newDict = Dictionary.ByKeysValues(sinkKeys, sinkValues)
                _ = sink.SetDictionary(newDict)
        return {"sources": sources, "sinks": sinks}

    
    @staticmethod
    def TransferDictionariesBySelectors(topology, selectors, tranVertices=False, tranEdges=False, tranFaces=False, tranCells=False, tolerance=0.0001):
        """
        Transfers the dictionaries of the list of selectors to the subtopologies of the input topology based on the input parameters.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        selectors : list
            The list of input selectors from which to transfer the dictionaries.
        tranVertices : bool , optional
            If True transfer dictionaries to the vertices of the input topology.
        tranEdges : bool , optional
            If True transfer dictionaries to the edges of the input topology.
        tranFaces : bool , optional
            If True transfer dictionaries to the faces of the input topology.
        tranCells : bool , optional
            If True transfer dictionaries to the cells of the input topology.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topology.Topology
            The input topology with the dictionaries transferred to its subtopologies.

        """
        if not isinstance(topology, topologic.Topology):
            return None
        if not isinstance(selectors, list):
            return None
        selectors = [x for x in selectors if isinstance(x, topologic.Topology)]
        if len(selectors) < 1:
            return None
        sinkEdges = []
        sinkFaces = []
        sinkCells = []
        hidimSink = Topology.HighestType(topology)
        if tranVertices == True:
            sinkVertices = []
            if topology.Type() == topologic.Vertex.Type():
                sinkVertices.append(topology)
            elif hidimSink >= topologic.Vertex.Type():
                topology.Vertices(None, sinkVertices)
            _ = Topology.TransferDictionaries(selectors, sinkVertices, tolerance)
        if tranEdges == True:
            sinkEdges = []
            if topology.Type() == topologic.Edge.Type():
                sinkEdges.append(topology)
            elif hidimSink >= topologic.Edge.Type():
                topology.Edges(None, sinkEdges)
                _ = Topology.TransferDictionaries(selectors, sinkEdges, tolerance)
        if tranFaces == True:
            sinkFaces = []
            if topology.Type() == topologic.Face.Type():
                sinkFaces.append(topology)
            elif hidimSink >= topologic.Face.Type():
                topology.Faces(None, sinkFaces)
            _ = Topology.TransferDictionaries(selectors, sinkFaces, tolerance)
        if tranCells == True:
            sinkCells = []
            if topology.Type() == topologic.Cell.Type():
                sinkCells.append(topology)
            elif hidimSink >= topologic.Cell.Type():
                topology.Cells(None, sinkCells)
            _ = Topology.TransferDictionaries(selectors, sinkCells, tolerance)
        return topology

    
    @staticmethod
    def Transform(topology, matrix):
        """
        Transforms the input topology by the input 4X4 transformation matrix.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        matrix : list
            The input 4x4 transformation matrix.

        Returns
        -------
        topologic.Topology
            The transformed topology.

        """
        kTranslationX = 0.0
        kTranslationY = 0.0
        kTranslationZ = 0.0
        kRotation11 = 1.0
        kRotation12 = 0.0
        kRotation13 = 0.0
        kRotation21 = 0.0
        kRotation22 = 1.0
        kRotation23 = 0.0
        kRotation31 = 0.0
        kRotation32 = 0.0
        kRotation33 = 1.0

        kTranslationX = matrix[0][3]
        kTranslationY = matrix[1][3]
        kTranslationZ = matrix[2][3]
        kRotation11 = matrix[0][0]
        kRotation12 = matrix[0][1]
        kRotation13 = matrix[0][2]
        kRotation21 = matrix[1][0]
        kRotation22 = matrix[1][1]
        kRotation23 = matrix[1][2]
        kRotation31 = matrix[2][0]
        kRotation32 = matrix[2][1]
        kRotation33 = matrix[2][2]

        return topologic.TopologyUtility.Transform(topology, kTranslationX, kTranslationY, kTranslationZ, kRotation11, kRotation12, kRotation13, kRotation21, kRotation22, kRotation23, kRotation31, kRotation32, kRotation33)

    @staticmethod
    def Translate(topology, x=0, y=0, z=0):
        """
        Translates (moves) the input topology.

        Parameters
        ----------
        topology : topologic.topology
            The input topology.
        x : float , optional
            The x translation value. The default is 0.
        y : float , optional
            The y translation value. The default is 0.
        z : float , optional
            The z translation value. The default is 0.

        Returns
        -------
        topologic.Topology
            The translated topology.

        """
        if not isinstance(topology, topologic.Topology):
            return None
        return topologic.TopologyUtility.Translate(topology, x, y, z)

    
    @staticmethod
    def Triangulate(topology, tolerance=0.0001):
        """
        Triangulates the input topology.

        Parameters
        ----------
        topology : topologic.Topology
            The input topologgy.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Topology
            The triangulated topology.

        """
        from topologicpy.Face import Face
        from topologicpy.Shell import Shell
        from topologicpy.Cell import Cell
        from topologicpy.CellComplex import CellComplex
        if not isinstance(topology, topologic.Topology):
            return None
        t = topology.Type()
        if (t == 1) or (t == 2) or (t == 4) or (t == 128):
            return topology
        topologyFaces = []
        _ = topology.Faces(None, topologyFaces)
        faceTriangles = []
        for aFace in topologyFaces:
            triFaces = Face.Triangulate(aFace)
            for triFace in triFaces:
                faceTriangles.append(triFace)
        if t == 8 or t == 16: # Face or Shell
            return Shell.ByFaces(faceTriangles, tolerance)
        elif t == 32: # Cell
            return Cell.ByFaces(faceTriangles, tolerance)
        elif t == 64: #CellComplex
            return CellComplex.ByFaces(faceTriangles, tolerance)

    
    @staticmethod
    def Type(topology):
        """
        Returns the type of the input topology.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.

        Returns
        -------
        int
            The type of the input topology.

        """
        if not isinstance(topology, topologic.Topology):
            return None
        return topology.Type()
    
    @staticmethod
    def TypeAsString(topology):
        """
        Returns the type of the input topology as a string.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.

        Returns
        -------
        str
            The type of the topology as a string.

        """
        if not isinstance(topology, topologic.Topology):
            return None
        return topology.GetTypeAsString()
    
    @staticmethod
    def TypeID(topologyType=None):
        """
        Returns the type id of the input topologyType string.

        Parameters
        ----------
        topologyType : str , optional
            The input topology type string. This could be one of "vertex", "edge", "wire", "face", "shell", "cell", "cellcomplex", "cluster". It is case insensitive. The default is None.

        Returns
        -------
        int
            The type id of the input topologyType string.

        """

        if not isinstance(topologyType, str):
            return None
        topologyType = topologyType.lower()
        if not topologyType in ["vertex", "edge", "wire", "face", "shell", "cell", "cellcomplex", "cluster"]:
            return None
        typeID = None
        if topologyType == "vertex":
            typeID = topologic.Vertex.Type()
        elif topologyType == "edge":
            typeID = topologic.Edge.Type()
        elif topologyType == "wire":
            typeID = topologic.Wire.Type()
        elif topologyType == "face":
            typeID = topologic.Face.Type()
        elif topologyType == "shell":
            typeID = topologic.Shell.Type()
        elif topologyType == "cell":
            typeID = topologic.Cell.Type()
        elif topologyType == "cellComplex":
            typeID = topologic.CellComplex.Type()
        elif topologyType == "cluster":
            typeID = topologic.Cluster.Type()
        return typeID