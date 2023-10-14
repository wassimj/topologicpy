#from base64 import b16encode
#from tkinter import N
import topologicpy
import topologic
from topologicpy.Aperture import Aperture

import uuid
import json
import os
import numpy as np
from numpy import arctan, pi, signbit
from numpy.linalg import norm
import math

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
            The subtopology type to which to add the apertures. This can be "cell", "face", "edge", or "vertex". It is case insensitive. If set to None, the apertures will be added to the input topology. The default is None.
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
            print("Topology.AddApertures - Error: The input topology is not a valid topology. Returning None.")
            return None
        if not apertures:
            return topology
        if not isinstance(apertures, list):
            print("Topology.AddApertures - Error: the input apertures is not a list. Returning None.")
            return None
        apertures = [x for x in apertures if isinstance(x , topologic.Topology)]
        if len(apertures) < 1:
            return topology
        if not subTopologyType:
            subTopologyType = "self"
        if not subTopologyType.lower() in ["self", "cell", "face", "edge", "vertex"]:
            print("Topology.AddApertures - Error: the input subtopology type is not a recognized type. Returning None.")
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
            The subtopology type to which to add the contents. This can be "cellcomplex", "cell", "shell", "face", "wire", "edge", or "vertex". It is case insensitive. If set to None, the contents will be added to the input topology. The default is None.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Topology
            The input topology with the contents added to it.

        """
        if not isinstance(topology, topologic.Topology):
            print("Topology.AddContent - Error: the input topology is not a valid topology. Returning None.")
            return None
        if not contents:
            return topology
        if not isinstance(contents, list):
            print("Topology.AddContent - Error: the input contents is not a list. Returning None.")
            return None
        contents = [x for x in contents if isinstance(x, topologic.Topology)]
        if len(contents) < 1:
            return topology
        if not subTopologyType:
            subTopologyType = "self"
        if not subTopologyType.lower() in ["self", "cellcomplex", "cell", "shell", "face", "wire", "edge", "vertex"]:
            print("Topology.AddContent - Error: the input subtopology type is not a recognized type. Returning None.")
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
            print("Topology.AddDictionary - Error: the input topology is not a valid topology. Returning None.")
            return None
        if not isinstance(dictionary, topologic.Dictionary):
            print("Topology.AddDictionary - Error: the input dictionary is not a dictionary. Returning None.")
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
            print("Topology.AdjacentTopologies - Error: the input topology is not a valid topology. Returning None.")
            return None
        if not isinstance(hostTopology, topologic.Topology):
            print("Topology.AdjacentTopologies - Error: the input hostTopology is not a valid topology. Returning None.")
            return None
        if not topologyType:
            topologyType = Topology.TypeAsString(topology).lower()
        if not isinstance(topologyType, str):
            print("Topology.AdjacentTopologies - Error: the input topologyType is not a string. Returning None.")
            return None
        if not topologyType.lower() in ["vertex", "edge", "wire", "face", "shell", "cell", "cellcomplex"]:
            print("Topology.AdjacentTopologies - Error: the input topologyType is not a recognized type. Returning None.")
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
            print("Topology.Analyze - Error: the input topology is not a valid topology. Returning None.")
            return None
        return topologic.Topology.Analyze(topology)
    
    @staticmethod
    def Apertures(topology, subTopologyType=None):
        """
        Returns the apertures of the input topology.
        
        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        subTopologyType : string , optional
            The subtopology type from which to retrieve the apertures. This can be "cell", "face", "edge", or "vertex" or "all". It is case insensitive. If set to "all", then all apertures will be returned. If set to None, the apertures will be retrieved only from the input topology. The default is None.
        
        Returns
        -------
        list
            The list of apertures belonging to the input topology.

        """
        if not isinstance(topology, topologic.Topology):
            print("Topology.Apertures - Error: the input topology is not a valid topology. Returning None.")
            return None
        
        apertures = []
        subTopologies = []
        if not subTopologyType:
            _ = topology.Apertures(apertures)
        elif subTopologyType.lower() == "vertex":
            subTopologies = Topology.Vertices(topology)
        elif subTopologyType.lower() == "edge":
            subTopologies = Topology.Edges(topology)
        elif subTopologyType.lower() == "face":
            subTopologies = Topology.Faces(topology)
        elif subTopologyType.lower() == "cell":
            subTopologies = Topology.Cells(topology)
        elif subTopologyType.lower() == "all":
            _ = topology.Apertures(apertures)
            subTopologies = Topology.Vertices(topology)
            subTopologies += Topology.Edges(topology)
            subTopologies += Topology.Faces(topology)
            subTopologies += Topology.Cells(topology)
        else:
            print("Topology.Apertures - Error: the input subtopologyType is not a recognized type. Returning None.")
            return None
        for subTopology in subTopologies:
            apertures += Topology.Apertures(subTopology, subTopologyType=None)
        return apertures


    @staticmethod
    def ApertureTopologies(topology, subTopologyType=None):
        """
        Returns the aperture topologies of the input topology.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        subTopologyType : string , optional
            The subtopology type from which to retrieve the apertures. This can be "cell", "face", "edge", or "vertex" or "all". It is case insensitive. If set to "all", then all apertures will be returned. If set to None, the apertures will be retrieved only from the input topology. The default is None.
       
        Returns
        -------
        list
            The list of aperture topologies found in the input topology.

        """
        from topologicpy.Aperture import Aperture
        if not isinstance(topology, topologic.Topology):
            print("Topology.ApertureTopologies - Error: the input topology is not a valid topology. Returning None.")
            return None
        apertures = Topology.Apertures(topology=topology, subTopologyType=subTopologyType)
        apTopologies = []
        for aperture in apertures:
            apTopologies.append(Aperture.Topology(aperture))
        return apTopologies
    
    @staticmethod
    def Union(topologyA, topologyB, tranDict=False, tolerance=0.0001):
        """
        See Topology.Boolean()

        """
        return Topology.Boolean(topologyA, topologyB, operation="union", tranDict=tranDict, tolerance=tolerance)
    
    @staticmethod
    def Difference(topologyA, topologyB, tranDict=False, tolerance=0.0001):
        """
        See Topology.Boolean().

        """
        return Topology.Boolean(topologyA=topologyA, topologyB=topologyB, operation="difference", tranDict=tranDict, tolerance=tolerance)
    
    @staticmethod
    def Intersect(topologyA, topologyB, tranDict=False, tolerance=0.0001):
        """
        See Topology.Boolean().

        """
        return Topology.Boolean(topologyA=topologyA, topologyB=topologyB, operation="intersect", tranDict=tranDict, tolerance=tolerance)
    
    @staticmethod
    def SymmetricDifference(topologyA, topologyB, tranDict=False, tolerance=0.0001):
        """
        See Topology.Boolean().

        """
        return Topology.Boolean(topologyA=topologyA, topologyB=topologyB, operation="symdif", tranDict=tranDict, tolerance=tolerance)
    
    @staticmethod
    def SymDif(topologyA, topologyB, tranDict=False, tolerance=0.0001):
        """
        See Topology.Boolean().

        """
        return Topology.Boolean(topologyA=topologyA, topologyB=topologyB, operation="symdif", tranDict=tranDict, tolerance=tolerance)
    
    @staticmethod
    def XOR(topologyA, topologyB, tranDict=False, tolerance=0.0001):
        """
        See Topology.Boolean().

        """
        return Topology.Boolean(topologyA=topologyA, topologyB=topologyB, operation="symdif", tranDict=tranDict, tolerance=tolerance)
    
    @staticmethod
    def Merge(topologyA, topologyB, tranDict=False, tolerance=0.0001):
        """
        See Topology.Boolean().

        """
        return Topology.Boolean(topologyA=topologyA, topologyB=topologyB, operation="merge", tranDict=tranDict, tolerance=tolerance)
    
    @staticmethod
    def Slice(topologyA, topologyB, tranDict=False, tolerance=0.0001):
        """
        See Topology.Boolean().

        """
        return Topology.Boolean(topologyA=topologyA, topologyB=topologyB, operation="slice", tranDict=tranDict, tolerance=tolerance)
    
    @staticmethod
    def Merge(topologyA, topologyB, tranDict=False, tolerance=0.0001):
        """
        See Topology.Boolean().

        """
        return Topology.Boolean(topologyA=topologyA, topologyB=topologyB, operation="merge", tranDict=tranDict, tolerance=tolerance)
    
    @staticmethod
    def Impose(topologyA, topologyB, tranDict=False, tolerance=0.0001):
        """
        See Topology.Boolean().

        """
        return Topology.Boolean(topologyA=topologyA, topologyB=topologyB, operation="impose", tranDict=tranDict, tolerance=tolerance)
    
    @staticmethod
    def Imprint(topologyA, topologyB, tranDict=False, tolerance=0.0001):
        """
        See Topology.Boolean().

        """
        return Topology.Boolean(topologyA=topologyA, topologyB=topologyB, operation="imprint", tranDict=tranDict, tolerance=tolerance)
    
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
            print("Topology.Boolean - Error: the input topologyA is not a valid topology. Returning None.")
            return None
        if not isinstance(topologyB, topologic.Topology):
            print("Topology.Boolean - Error: the input topologyB is not a valid topology. Returning None.")
            return None
        if not isinstance(operation, str):
            print("Topology.Boolean - Error: the input operation is not a valid string. Returning None.")
            return None
        if not operation.lower() in ["union", "difference", "intersect", "symdif", "merge", "slice", "impose", "imprint"]:
            print("Topology.Boolean - Error: the input operation is not a recognized operation. Returning None.")
            return None
        if not isinstance(tranDict, bool):
            print("Topology.Boolean - Error: the input tranDict is not a valid boolean. Returning None.")
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
                print("Topology.Boolean - Error: the boolean operation failed. Returning None.")
                return None
        except:
            print("Topology.Boolean - Error: the boolean operation failed. Returning None.")
            return None
        if tranDict == True:
            sourceVertices = []
            sourceEdges = []
            sourceFaces = []
            sourceCells = []
            sinkVertices = []
            sinkEdges = []
            sinkFaces = []
            sinkCells = []
            hidimA = Topology.HighestType(topologyA)
            hidimB = Topology.HighestType(topologyB)
            hidimC = Topology.HighestType(topologyC)

            if topologyA.Type() == topologic.Vertex.Type():
                sourceVertices += [topologyA]
            elif hidimA >= topologic.Vertex.Type():
                sourceVertices += Topology.Vertices(topologyA)
            if topologyB.Type() == topologic.Vertex.Type():
                sourceVertices += [topologyB]
            elif hidimB >= topologic.Vertex.Type():
                sourceVertices += Topology.Vertices(topologyB)
            if topologyC.Type() == topologic.Vertex.Type():
                sinkVertices = [topologyC]
            elif hidimC >= topologic.Vertex.Type():
                sinkVertices = Topology.Vertices(topologyC)
            if len(sourceVertices) > 0 and len(sinkVertices) > 0:
                _ = Topology.TransferDictionaries(sourceVertices, sinkVertices, tolerance)

            if topologyA.Type() == topologic.Edge.Type():
                sourceEdges += [topologyA]
            elif hidimA >= topologic.Edge.Type():
                sourceEdges += Topology.Edges(topologyA)
            if topologyB.Type() == topologic.Edge.Type():
                sourceEdges += [topologyB]
            elif hidimB >= topologic.Edge.Type():
                sourceEdges += Topology.Edges(topologyB)
            if topologyC.Type() == topologic.Edge.Type():
                sinkEdges = [topologyC]
            elif hidimC >= topologic.Edge.Type():
                sinkEdges = Topology.Edges(topologyC)
            if len(sourceEdges) > 0 and len(sinkEdges) > 0:
                _ = Topology.TransferDictionaries(sourceEdges, sinkEdges, tolerance)

            if topologyA.Type() == topologic.Face.Type():
                sourceFaces += [topologyA]
            elif hidimA >= topologic.Face.Type():
                sourceFaces += Topology.Faces(topologyA)
            if topologyB.Type() == topologic.Face.Type():
                sourceFaces += [topologyB]
            elif hidimB >= topologic.Face.Type():
                sourceFaces += Topology.Faces(topologyB)
            if topologyC.Type() == topologic.Face.Type():
                sinkFaces += [topologyC]
            elif hidimC >= topologic.Face.Type():
                sinkFaces += Topology.Faces(topologyC)
            if len(sourceFaces) > 0 and len(sinkFaces) > 0:
                _ = Topology.TransferDictionaries(sourceFaces, sinkFaces, tolerance)

            if topologyA.Type() == topologic.Cell.Type():
                sourceCells += [topologyA]
            elif hidimA >= topologic.Cell.Type():
                sourceCells += Topology.Cells(topologyA)
            if topologyB.Type() == topologic.Cell.Type():
                sourceCells += [topologyB]
            elif hidimB >= topologic.Cell.Type():
                sourceCells += Topology.Cells(topologyB)
            if topologyC.Type() == topologic.Cell.Type():
                sinkCells = [topologyC]
            elif hidimC >= topologic.Cell.Type():
                sinkCells = Topology.Cells(topologyC)
            if len(sourceCells) > 0 and len(sinkCells) > 0:
                print("919: Trying to transfer dictionaries of cells!")
                _ = Topology.TransferDictionaries(sourceCells, sinkCells, tolerance)
        return topologyC

    
    @staticmethod
    def BoundingBox(topology, optimize=0, axes="xyz", tolerance=0.0001):
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
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
        Returns
        -------
        topologic.Cell or topologic.Face
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
            print("Topology.BoundingBox - Error: the input topology is not a valid topology. Returning None.")
            return None
        if not isinstance(axes, str):
            print("Topology.BoundingBox - Error: the input axes is not a valid string. Returning None.")
            return None
        axes = axes.lower()
        x_flag = "x" in axes
        y_flag = "y" in axes
        z_flag = "z" in axes
        if not x_flag and not y_flag and not z_flag:
            print("Topology.BoundingBox - Error: the input axes is not a recognized string. Returning None.")
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

        baseWire = Wire.ByVertices([vb1, vb2, vb3, vb4], close=True)
        baseFace = Face.ByWire(baseWire)
        if abs(maxZ-minZ) < tolerance:
            box = baseFace
        else:
            box = Cell.ByThickenedFace(baseFace, planarize=False, thickness=abs(maxZ-minZ), bothSides=False)
        box = Topology.Rotate(box, origin=origin, x=1,y=0,z=0, degree=-best_x)
        box = Topology.Rotate(box, origin=origin, x=0,y=1,z=0, degree=-best_y)
        box = Topology.Rotate(box, origin=origin, x=0,y=0,z=1, degree=-best_z)
        dictionary = Dictionary.ByKeysValues(["xrot","yrot","zrot", "minx", "miny", "minz", "maxx", "maxy", "maxz", "width", "length", "height"], [best_x, best_y, best_z, minX, minY, minZ, maxX, maxY, maxZ, (maxX-minX), (maxY-minY), (maxZ-minZ)])
        box = Topology.SetDictionary(box, dictionary)
        return box

    @staticmethod
    def BREPString(topology, version=3):
        """
        Returns the BRep string of the input topology.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        version : int , optional
            The desired BRep version number. The default is 3.

        Returns
        -------
        str
            The BREP string.

        """
        if not isinstance(topology, topologic.Topology):
            print("Topology.BREPString - Error: the input topology is not a valid topology. Returning None.")
            return None
        st = None
        try:
            st = topologic.Topology.String(topology, version)
        except:
            try:
                st = topologic.Topology.BREPString(topology, version)
            except:
                st = None
        return st
    
    @staticmethod
    def ByGeometry(vertices=[], edges=[], faces=[], color=[1.0,1.0,1.0,1.0], id=None, name=None, lengthUnit="METERS", outputMode="default", tolerance=0.0001):
        """
        Create a topology by the input lists of vertices, edges, and faces.

        Parameters
        ----------
        vertices : list
            The input list of vertices in the form of [x, y, z]
        edges : list , optional
            The input list of edges in the form of [i, j] where i and j are vertex indices.
        faces : list , optional
            The input list of faces in the form of [i, j, k, l, ...] where the items in the list are vertex indices. The face is assumed to be closed to the last vertex is connected to the first vertex automatically.
        color : list , optional
            The desired color of the object in the form of [r, g, b, a] where the components are between 0 and 1 and represent red, blue, green, and alpha (transparency) repsectively. The default is [1.0, 1.0, 1.0, 1.0].
        id : str , optional
            The desired ID of the object. If set to None, an automatic uuid4 will be assigned to the object. The default is None.
        name : str , optional
            The desired name of the object. If set to None, a default name "Topologic_[topology_type]" will be assigned to the object. The default is None.
        lengthUnit : str , optional
            The length unit used for the object. The default is "METERS"
        outputMode : str , optional
            The desired otuput mode of the object. This can be "wire", "shell", "cell", "cellcomplex", or "default". It is case insensitive. The default is "default".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topology : topologic.Topology
            The created topology. The topology will have a dictionary embedded in it that records the input attributes (color, id, lengthUnit, name, type)

        """
        def topologyByFaces(faces, outputMode, tolerance):
            output = None
            if len(faces) == 1:
                return faces[0]
            if outputMode.lower() == "cell":
                output = Cell.ByFaces(faces, tolerance=tolerance)
                if output:
                    return output
                else:
                    return None
            if outputMode.lower() == "cellcomplex":
                output = CellComplex.ByFaces(faces, tolerance=tolerance)
                if output:
                    return output
                else:
                    return None
            if outputMode.lower() == "shell":
                output = Shell.ByFaces(faces, tolerance)
                if output:
                    return output
                else:
                    return None
            if outputMode.lower() == "default":
                output = Cluster.ByTopologies(faces)
                if output:
                    return output
            return output
        def topologyByEdges(edges, outputMode):
            output = None
            if len(edges) == 1:
                return edges[0]
            output = Cluster.ByTopologies(edges)
            if outputMode.lower() == "wire":
                output = Topology.SelfMerge(output)
                if isinstance(output, topologic.Wire):
                    return output
                else:
                    return None
            return output
        def edgesByVertices(vertices, topVerts):
            if len(vertices) < 2:
                return []
            edges = []
            for i in range(len(vertices)-1):
                v1 = vertices[i]
                v2 = vertices[i+1]
                e1 = Edge.ByVertices([topVerts[v1], topVerts[v2]])
                edges.append(e1)
            # connect the last vertex to the first one
            v1 = vertices[-1]
            v2 = vertices[0]
            e1 = Edge.ByVertices([topVerts[v1], topVerts[v2]])
            edges.append(e1)
            return edges
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Shell import Shell
        from topologicpy.Cell import Cell
        from topologicpy.CellComplex import CellComplex
        from topologicpy.Cluster import Cluster
        from topologicpy.Dictionary import Dictionary
        import uuid
        returnTopology = None
        topVerts = []
        topEdges = []
        topFaces = []
        vertices = [v for v in vertices if not v == []]
        edges = [e for e in edges if not e == []]
        faces = [f for f in faces if not f == []]
        if len(vertices) > 0:
            for aVertex in vertices:
                v = Vertex.ByCoordinates(aVertex[0], aVertex[1], aVertex[2])
                topVerts.append(v)
        else:
            return None
        if (outputMode.lower == "wire") and (len(edges) > 0):
            for anEdge in edges:
                topEdge = Edge.ByVertices([topVerts[anEdge[0]], topVerts[anEdge[1]]])
                topEdges.append(topEdge)
            if len(topEdges) > 0:
                returnTopology = topologyByEdges(topEdges)
        elif len(faces) > 0:
            for aFace in faces:
                faceEdges = edgesByVertices(aFace, topVerts)
                if len(faceEdges) > 2:
                    faceWire = Wire.ByEdges(faceEdges)
                    try:
                        topFace = Face.ByWire(faceWire)
                        topFaces.append(topFace)
                    except:
                        pass
            if len(topFaces) > 0:
                returnTopology = topologyByFaces(topFaces, outputMode=outputMode, tolerance=tolerance)
        elif len(edges) > 0:
            for anEdge in edges:
                topEdge = Edge.ByVertices([topVerts[anEdge[0]], topVerts[anEdge[1]]])
                topEdges.append(topEdge)
            if len(topEdges) > 0:
                returnTopology = topologyByEdges(topEdges, outputMode=outputMode)
        else:
            returnTopology = Cluster.ByTopologies(topVerts)
        if returnTopology:
            keys = []
            values = []
            keys.append("TOPOLOGIC_color")
            keys.append("TOPOLOGIC_id")
            keys.append("TOPOLOGIC_name")
            keys.append("TOPOLOGIC_type")
            keys.append("TOPOLOGIC_length_unit")
            if color:
                if isinstance(color, tuple):
                    color = list(color)
                elif isinstance(color, list):
                    if isinstance(color[0], tuple):
                        color = list(color[0])
                values.append(color)
            else:
                values.append([1.0,1.0,1.0,1.0])
            if id:
                values.append(id)
            else:
                values.append(str(uuid.uuid4()))
            if name:
                values.append(name)
            else:
                values.append("Topologic_"+Topology.TypeAsString(returnTopology))
            values.append(Topology.TypeAsString(returnTopology))
            values.append(lengthUnit)
            topDict = Dictionary.ByKeysValues(keys, values)
            Topology.SetDictionary(returnTopology, topDict)
        return returnTopology

    @staticmethod
    def ByBREPFile(file):
        """
        Imports a topology from a BREP file.

        Parameters
        ----------
        file : file object
            The BREP file.

        Returns
        -------
        topologic.Topology
            The imported topology.

        """
        topology = None
        if not file:
            print("Topology.ByBREPFile - Error: the input file is not a valid file. Returning None.")
            return None
        brep_string = file.read()
        topology = Topology.ByBREPString(brep_string)
        file.close()
        return topology
    
    @staticmethod
    def ByBREPPath(path):
        """
        IMports a topology from a BREP file path.

        Parameters
        ----------
        path : str
            The path to the BREP file.

        Returns
        -------
        topologic.Topology
            The imported topology.

        """
        if not path:
            print("Topology.ByBREPPath - Error: the input path is not a valid path. Returning None.")
            return None
        try:
            file = open(path)
        except:
            print("Topology.ByBREPPath - Error: the BREP file is not a valid file. Returning None.")
            return None
        return Topology.ByBREPFile(file)
    
    @staticmethod
    def ByIFCFile(file, transferDictionaries=False):
        """
        Create a topology by importing it from an IFC file.

        Parameters
        ----------
        file : file object
            The input IFC file.
        transferDictionaries : bool , optional
            If set to True, the dictionaries from the IFC file will be transfered to the topology. Otherwise, they won't. The default is False.
        
        Returns
        -------
        list
            The created list of topologies.
        
        """
        import multiprocessing
        from topologicpy.Cluster import Cluster
        from topologicpy.Dictionary import Dictionary
        import uuid
        import sys
        import subprocess

        try:
            import ifcopenshell
            import ifcopenshell.geom
        except:
            call = [sys.executable, '-m', 'pip', 'install', 'ifcopenshell', '-t', sys.path[0]]
            subprocess.run(call)
            try:
                import ifcopenshell
                import ifcopenshell.geom
            except:
                print("Topology.ByIFCFile - ERROR: Could not import ifcopenshell")
        if not file:
            print("Topology.ByIFCFile - Error: the input file is not a valid file. Returning None.")
            return None
        topologies = []
        settings = ifcopenshell.geom.settings()
        settings.set(settings.DISABLE_TRIANGULATION, True)
        settings.set(settings.USE_BREP_DATA, True)
        settings.set(settings.USE_WORLD_COORDS, True)
        settings.set(settings.SEW_SHELLS, True)
        iterator = ifcopenshell.geom.iterator(settings, file, multiprocessing.cpu_count())
        if iterator.initialize():
            while True:
                shape = iterator.get()
                brep = shape.geometry.brep_data
                topology = Topology.ByBREPString(brep)
                if transferDictionaries:
                        keys = []
                        values = []
                        keys.append("TOPOLOGIC_color")
                        values.append([1.0,1.0,1.0,1.0])
                        keys.append("TOPOLOGIC_id")
                        values.append(str(uuid.uuid4()))
                        keys.append("TOPOLOGIC_name")
                        values.append(shape.name)
                        keys.append("TOPOLOGIC_type")
                        values.append(Topology.TypeAsString(topology))
                        keys.append("IFC_id")
                        values.append(str(shape.id))
                        keys.append("IFC_guid")
                        values.append(str(shape.guid))
                        keys.append("IFC_unique_id")
                        values.append(str(shape.unique_id))
                        keys.append("IFC_name")
                        values.append(shape.name)
                        keys.append("IFC_type")
                        values.append(shape.type)
                        d = Dictionary.ByKeysValues(keys, values)
                        topology = Topology.SetDictionary(topology, d)
                topologies.append(topology)
                if not iterator.next():
                    break
        return topologies

    @staticmethod
    def ByIFCPath(path, transferDictionaries=False):
        """
        Create a topology by importing it from an IFC file path.

        Parameters
        ----------
        path : str
            The path to the IFC file.
        transferDictionaries : bool , optional
            If set to True, the dictionaries from the IFC file will be transfered to the topology. Otherwise, they won't. The default is False.
        
        Returns
        -------
        list
            The created list of topologies.
        
        """
        import sys, subprocess
        try:
            import ifcopenshell
            import ifcopenshell.geom
        except:
            call = [sys.executable, '-m', 'pip', 'install', 'ifcopenshell', '-t', sys.path[0]]
            subprocess.run(call)
            try:
                import ifcopenshell
                import ifcopenshell.geom
            except:
                print("Topology.ByIFCPath - ERROR: Could not import ifcopenshell")

        if not path:
            print("Topology.ByIFCPath - Error: the input path is not a valid path. Returning None.")
            return None
        try:
            file = ifcopenshell.open(path)
        except:
            print("Topology.ByIFCPath - Error: the input file is not a valid file. Returning None.")
            file = None
        if not file:
            print("Topology.ByIFCPath - Error: the input file is not a valid file. Returning None.")
            return None
        return Topology.ByIFCFile(file, transferDictionaries=transferDictionaries)
    
    '''
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
        topology = Topology.ByBREPString(brepString)
        return topology
    '''
    @staticmethod
    def ByJSONFile(file, tolerance=0.0001):
        """
        Imports the topology from a JSON file.

        Parameters
        ----------
        file : file object
            The input JSON file.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        list
            The list of imported topologies.

        """
        if not file:
            print("Topology.ByJSONFile - Error: the input file is not a valid file. Returning None.")
            return None
        jsonData = json.load(file)
        jsonString = json.dumps(jsonData)
        return Topology.ByJSONString(jsonString, tolerance=tolerance)
    
    @staticmethod
    def ByJSONString(string, tolerance=0.0001):
        """
        Imports the topology from a JSON string.

        Parameters
        ----------
        string : str
            The input JSON string.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        list or topologicpy.Topology
            The list of imported topologies. If the list only contains one element, it returns that element.

        """

        from topologicpy.Dictionary import Dictionary
        from topologicpy.Context import Context
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Shell import Shell
        from topologicpy.Cell import Cell
        from topologicpy.CellComplex import CellComplex
        from topologicpy.Cluster import Cluster
        from topologicpy.Aperture import Aperture
        from tqdm.auto import tqdm
        import time

        def getUUID(topology, uuidKey="uuid"):
            d = Topology.Dictionary(topology)
            if d == None:
                uuidOne = str(uuid.uuid1())
                d = Dictionary.ByKeyValue(uuidKey, uuidOne)
            elif uuidKey not in Dictionary.Keys(d):
                uuidOne = str(uuid.uuid1())
                d = Dictionary.SetValueAtKey(d, uuidKey, uuidOne)
                topology = Topology.SetDictionary(topology, d)
            else:
                uuidOne = Dictionary.ValueAtKey(d, uuidKey)
            return uuidOne
            
        def find_json_item(json_list, key, value):
            for item in json_list:
                if key in item and item[key] == value:
                    return item
            return None

        def buildAperture(j_aperture):
            j_vertices = []
            j_edges = []
            j_wires = []
            j_faces = []
            j_shells = []
            j_cells = []
            j_cellComplexes = []
            for jsonItem in j_aperture:
                topology_type = jsonItem['type']
                if topology_type.lower() == "vertex":
                    j_vertices.append(jsonItem)
                elif topology_type.lower() == "edge":
                    j_edges.append(jsonItem)
                elif topology_type.lower() == "wire":
                    j_wires.append(jsonItem)
                elif topology_type.lower() == "face":
                    j_faces.append(jsonItem)
                elif topology_type.lower() == "shell":
                    j_shells.append(jsonItem)
                elif topology_type.lower() == "cell":
                    j_cells.append(jsonItem)
                elif topology_type.lower() == "cellcomplex":
                    j_cellComplexes.append(jsonItem)

            vertices = [buildVertex(j_v) for j_v in j_vertices]
            edges = [buildEdge(j_e, j_vertices, uuidKey="uuid") for j_e in j_edges]
            edges = [x for x in edges if x is not None]
            wires = [buildWire(j_w, j_edges, j_vertices, uuidKey="uuid") for j_w in j_wires]
            faces = [buildFace(j_f, j_wires, j_edges, j_vertices, uuidKey="uuid") for j_f in j_faces]
            faces = [x for x in faces if x is not None]
            shells = [buildShell(j_s, j_faces, j_wires, j_edges, j_vertices, uuidKey="uuid") for j_s in j_shells]
            cells = [buildCell(j_c, j_shells, j_faces, j_wires, j_edges, j_vertices, uuidKey="uuid") for j_c in j_cells]
            cellComplexes = [buildCellComplex(j_cc, j_cells, j_shells, j_faces, j_wires, j_edges, j_vertices, uuidKey="uuid") for j_cc in j_cellComplexes]
            if len(cellComplexes) > 0:
                everything = cellComplexes
            elif len(cells) > 0:
                everything = cells
            elif len(shells) > 0:
                everything = shells
            elif len(faces) > 0:
                everything = faces
            elif len(wires) > 0:
                everything = wires
            elif len(edges) > 0:
                everything = edges
            elif len(vertices) > 0:
                everything = vertices
            else:
                return None
            if len(everything) == 1:
                aperture = everything[0]
            else:
                aperture = Topology.SelfMerge(Cluster.ByTopologies(everything))
            return aperture

        def buildVertex(json_item):
            x, y, z = json_item['coordinates']
            d = json_item['dictionary']
            v = Vertex.ByCoordinates(x,y,z)
            v = Topology.SetDictionary(v, Dictionary.ByPythonDictionary(d))
            apertures = [buildAperture(j_ap) for j_ap in json_item['apertures']]
            context = Context.ByTopologyParameters(v, u=0.5, v=0.5, w=0.5)
            for ap in apertures:
                _ = Aperture.ByTopologyContext(ap, context)
            return v

        def buildEdge(json_item, j_vertices, uuidKey="uuid"):
            edge_vertices = json_item['vertices']
            vertices = []
            for j_v in edge_vertices:
                vertices.append(buildVertex(find_json_item(j_vertices, uuidKey, j_v)))
            e = Edge.ByVertices(vertices)
            if e is None:
                return None
            d = json_item['dictionary']
            e = Topology.SetDictionary(e, Dictionary.ByPythonDictionary(d))
            apertures = [buildAperture(j_ap) for j_ap in json_item['apertures']]
            context = Context.ByTopologyParameters(e, u=0.5, v=0.5, w=0.5)
            for ap in apertures:
                _ = Aperture.ByTopologyContext(ap, context)
            return e

        def buildWire(json_item, j_edges, j_vertices, uuidKey="uuid"):
            wire_edges = json_item['edges']
            edges = []
            for j_e in wire_edges:
                edges.append(buildEdge(find_json_item(j_edges, uuidKey, j_e), j_vertices, uuidKey=uuidKey))
            w = Wire.ByEdges(edges)
            d = json_item['dictionary']
            w = Topology.SetDictionary(w, Dictionary.ByPythonDictionary(d))
            apertures = [buildAperture(j_ap) for j_ap in json_item['apertures']]
            context = Context.ByTopologyParameters(w, u=0.5, v=0.5, w=0.5)
            for ap in apertures:
                _ = Aperture.ByTopologyContext(ap, context)
            return w

        def buildFace(json_item, j_wires, j_edges, j_vertices, uuidKey="uuid",x=0):
            bn = 298
            if x >= bn:
                print("        Entering buildFace function")
            face_wires = json_item['wires']
            external_boundary = buildWire(find_json_item(j_wires, uuidKey, face_wires[0]), j_edges, j_vertices, uuidKey=uuidKey)
            if not isinstance(external_boundary, topologic.Wire):
                print("Topology.ByJSONString - ERROR: Something went wrong with original external boundary. Returning None.")
                return None
            temp_boundary = Wire.Planarize(external_boundary)
            if temp_boundary == None or not isinstance(temp_boundary, topologic.Wire):
                print("Topology.ByJSONString - Error: Something went wrong with external boundary. Returning None.")
                return None
            else:
                external_boundary = temp_boundary
            internal_boundaries = []
            for j_w in face_wires[1:]:
                ib = buildWire(find_json_item(j_wires, uuidKey, j_w),j_edges, j_vertices, uuidKey=uuidKey)
                ib = Wire.Planarize(ib)
                internal_boundaries.append(ib)
            f = Face.ByWire(external_boundary)
            if f is None:
                return None
            if Face.Area(f) < 0:
                external_boundary = Wire.Invert(external_boundary)
            f = Face.ByWires(external_boundary, internal_boundaries)
            d = json_item['dictionary']
            f = Topology.SetDictionary(f, Dictionary.ByPythonDictionary(d))
            apertures = [buildAperture(j_ap) for j_ap in json_item['apertures']]
            if len(apertures) > 0:
                context = Context.ByTopologyParameters(f, u=0.5, v=0.5, w=0.5)
                for ap in apertures:
                    _ = Aperture.ByTopologyContext(ap, context)
            if x >= 298:
                print("        Exiting buildFace function with face:", f)
            return f

        def buildShell(json_item, j_faces, j_wires, j_edges, j_vertices, uuidKey="uuid"):
            shell_faces = json_item['faces']
            faces = []
            for j_f in shell_faces:
                faces.append(buildFace(find_json_item(j_faces, uuidKey, j_f), j_wires, j_edges, j_vertices, uuidKey=uuidKey))
            faces = [x for x in faces if x is not None]
            s = Shell.ByFaces(faces)
            d = json_item['dictionary']
            s = Topology.SetDictionary(s, Dictionary.ByPythonDictionary(d))
            apertures = [buildAperture(j_ap) for j_ap in json_item['apertures']]
            if len(apertures) > 0:
                context = Context.ByTopologyParameters(s, u=0.5, v=0.5, w=0.5)
                for ap in apertures:
                    _ = Aperture.ByTopologyContext(ap, context)
            return s

        def buildCell(json_item, j_shells, j_faces, j_wires, j_edges, j_vertices, uuidKey="uuid"):
            cell_shells = json_item['shells']
            shells = []
            external_boundary = buildShell(find_json_item(j_shells, uuidKey, cell_shells[0]), j_faces, j_wires, j_edges, j_vertices, uuidKey=uuidKey)
            internal_boundaries = []
            for j_s in cell_shells[1:]:
                internal_boundaries.append(buildShell(find_json_item(j_shells, uuidKey, j_s), j_faces, j_wires, j_edges, j_vertices, uuidKey=uuidKey))
            c = Cell.ByShell(external_boundary)
            for ib in internal_boundaries:
                ib_c = Cell.ByShell(ib)
                c = Topology.Difference(c, ib_c)
            d = json_item['dictionary']
            c = Topology.SetDictionary(c, Dictionary.ByPythonDictionary(d))
            apertures = [buildAperture(j_ap) for j_ap in json_item['apertures']]
            context = Context.ByTopologyParameters(c, u=0.5, v=0.5, w=0.5)
            for ap in apertures:
                _ = Aperture.ByTopologyContext(ap, context)
            return c

        def buildCellComplex(json_item, j_cells, j_shells, j_faces, j_wires, j_edges, j_vertices, uuidKey="uuid"):
            cc_cells = json_item['cells']
            cells = []
            for j_c in cc_cells:
                cells.append(buildCell(find_json_item(j_cells, uuidKey, j_c), j_shells, j_faces, j_wires, j_edges, j_vertices, uuidKey=uuidKey))
            cc = CellComplex.ByCells(cells)
            d = json_item['dictionary']
            cc = Topology.SetDictionary(cc, Dictionary.ByPythonDictionary(d))
            apertures = [buildAperture(j_ap) for j_ap in json_item['apertures']]
            context = Context.ByTopologyParameters(cc, u=0.5, v=0.5, w=0.5)
            for ap in apertures:
                _ = Aperture.ByTopologyContext(ap, context)
            return cc

        def addAperturesUUID(topology, uuidKey="uuid"):
            topology_apertures = [Aperture.Topology(ap) for ap in Topology.Apertures(topology)]
            apertures_uuid = []
            for top_a in topology_apertures:
                uuid = getUUID(top_a, uuidKey=uuidKey)
                apertures_uuid.append(uuid)
            d = Topology.Dictionary(topology)
            d = Dictionary.SetValueAtKey(d, 'apertures', apertures_uuid)
            topology = Topology.SetDictionary(topology, d)
            s = Topology.InternalVertex(topology)
            s = Topology.SetDictionary(s, d)
            return topology, s, topology_apertures

        def findAperture(uuid, apertures, uuidKey="uuid"):
            for ap in apertures:
                d = Topology.Dictionary(ap)
                ap_uuid = Dictionary.ValueAtKey(d, uuidKey)
                if uuid == ap_uuid:
                    return ap
            return None

        def setApertures(topology, allApertures, uuidKey="uuid"):
            apertures = []
            d = Topology.Dictionary(topology)
            apertures_uuid = Dictionary.ValueAtKey(d, 'apertures')
            if not isinstance(apertures_uuid, list):
                apertures_uuid = [apertures_uuid]
            for ap_uuid in apertures_uuid:
                ap = findAperture(ap_uuid, allApertures, uuidKey=uuidKey)
                if ap != None:
                    apertures.append(ap)
            context = Context.ByTopologyParameters(topology, u=0.5, v=0.5, w=0.5)
            for ap in apertures:
                _ = Aperture.ByTopologyContext(ap, context)
            return topology
                    
                
        jsondata = json.loads(string)
        if not isinstance(jsondata, list):
            jsondata = [jsondata]

        j_vertices = []
        j_edges = []
        j_wires = []
        j_faces = []
        j_shells = []
        j_cells = []
        j_cellComplexes = []

        vertices = []
        edges = []
        wires = []
        faces = []
        shells = []
        cells = []
        cellComplexes = []

        for jsonItem in jsondata:
            topology_type = jsonItem['type']
            if topology_type.lower() == "vertex":
                j_vertices.append(jsonItem)
            elif topology_type.lower() == "edge":
                j_edges.append(jsonItem)
            elif topology_type.lower() == "wire":
                j_wires.append(jsonItem)
            elif topology_type.lower() == "face":
                j_faces.append(jsonItem)
            elif topology_type.lower() == "shell":
                j_shells.append(jsonItem)
            elif topology_type.lower() == "cell":
                j_cells.append(jsonItem)
            elif topology_type.lower() == "cellcomplex":
                j_cellComplexes.append(jsonItem)

        print("1. Importing", len(j_vertices),"vertices and selectors")
        vertices = [buildVertex(j_v) for j_v in j_vertices]
        vertex_selectors = []
        all_vertex_apertures = []
        for v in vertices:
            v, s, vertex_apertures = addAperturesUUID(v, uuidKey="uuid")
            all_vertex_apertures += vertex_apertures
            vertex_selectors.append(s)
        print("   Done importing vertices and selectors")

        print("2. Importing", len(j_edges),"edges and selectors")
        edges = [buildEdge(j_e, j_vertices, uuidKey="uuid") for j_e in j_edges]
        edge_selectors = []
        all_edge_apertures = []
        for e in edges:
            e, s, edge_apertures = addAperturesUUID(e, uuidKey="uuid")
            all_edge_apertures += edge_apertures
            edge_selectors.append(s)
        print("   Done importing edges and selectors")

        print("3. Importing", len(j_wires),"wires and selectors")
        wires = [buildWire(j_w, j_edges, j_vertices, uuidKey="uuid") for j_w in j_wires]
        wire_selectors = []
        all_wire_apertures = []
        for w in wires:
            w, s, wire_apertures = addAperturesUUID(w, uuidKey="uuid")
            all_wire_apertures += wire_apertures
            wire_selectors.append(s)
        print("   Done importing wires and selectors")

        print("4. Importing", len(j_faces),"faces and selectors")
        faces = []
        xx = 0
        for j_f in tqdm(j_faces[298:302]):
            xx = j_faces.index(j_f)
            print("   ",xx,"Building face")
            f = buildFace(j_f, j_wires, j_edges, j_vertices, uuidKey="uuid", x=xx)
            print("   ",xx,"Received face:", f)
            print("   ",xx,"Appending face to list", f)
            faces.append(f)
            print("   ",xx,"Done appending face to list")
            xx += 1
            print("   Fetching next item:", xx)
        print("   Done importing faces")
        face_selectors = []
        all_face_apertures = []
        for f in faces:
            f, s, face_apertures = addAperturesUUID(f, uuidKey="uuid")
            all_face_apertures += face_apertures
            face_selectors.append(s)
        print("   Done building face selectors")

        print("5. Building shells and selectors")
        shells = [buildShell(j_s, j_faces, j_wires, j_edges, j_vertices, uuidKey="uuid") for j_s in j_shells]
        shell_selectors = []
        all_shell_apertures = []
        for sh in shells:
            sh, s, shell_apertures = addAperturesUUID(sh, uuidKey="uuid")
            all_shell_apertures += shell_apertures
            shell_selectors.append(s)
        print("   Done building shells and selectors")

        print("6. Building cells and selectors")
        cells = [buildCell(j_c, j_shells, j_faces, j_wires, j_edges, j_vertices, uuidKey="uuid") for j_c in j_cells]
        cell_selectors = []
        all_cell_apertures = []
        for c in cells:
            c, s, cell_apertures = addAperturesUUID(c, uuidKey="uuid")
            all_cell_apertures += cell_apertures
            cell_selectors.append(s)
        print("   Done building cells and selectors")

        print("7. Building cellcomplexes and selectors")
        cellComplexes = [buildCellComplex(j_cc, j_cells, j_shells, j_faces, j_wires, j_edges, j_vertices, uuidKey="uuid") for j_cc in j_cellComplexes]
        cellComplex_selectors = []
        all_cellComplex_apertures = []
        for cc in cellComplexes:
            cc, s, cellComplex_apertures = addAperturesUUID(cc, uuidKey="uuid")
            all_cellComplex_apertures += cellComplex_apertures
            cellComplex_selectors.append(s)
        print("   Done building cellcomplexes and selectors")

        everything = vertices+edges+wires+faces+shells+cells+cellComplexes
        toplevelTopologies = []
        for ev in everything:
            d = Topology.Dictionary(ev)
            if Dictionary.ValueAtKey(d,'toplevel') == True:
                toplevelTopologies.append(ev)

        print("Found toplevelTopologies", len(toplevelTopologies))
        for tp in tqdm(toplevelTopologies):
            _ = Topology.TransferDictionariesBySelectors(tp, vertex_selectors, tranVertices=True, tolerance=tolerance)
            _ = Topology.TransferDictionariesBySelectors(tp, edge_selectors, tranEdges=True, tolerance=tolerance)
            _ = Topology.TransferDictionariesBySelectors(tp, face_selectors, tranFaces=True, tolerance=tolerance)
            _ = Topology.TransferDictionariesBySelectors(tp, cell_selectors, tranCells=True, tolerance=tolerance)
            tp_vertices = Topology.Vertices(tp)
            for tp_vertex in tp_vertices:
                tp_vertex = setApertures(tp_vertex, all_vertex_apertures, uuidKey="uuid")
            tp_edges = Topology.Edges(tp)
            for tp_edge in tp_edges:
                tp_edge = setApertures(tp_edge, all_edge_apertures, uuidKey="uuid")
            tp_faces = Topology.Faces(tp)
            for tp_face in tp_faces:
                tp_face = setApertures(tp_face, all_face_apertures, uuidKey="uuid")
            tp_cells = Topology.Cells(tp)
            for tp_cell in tp_cells:
                tp_cell = setApertures(tp_cell, all_cell_apertures, uuidKey="uuid")
    
        if len(toplevelTopologies) == 1:
            return toplevelTopologies[0]
        else:
            return toplevelTopologies
    
    @staticmethod
    def ByJSONPath(path, tolerance=0.0001):
        """
        Imports the topology from a JSON file.

        Parameters
        ----------
        path : str
            The file path to the json file.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        list
            The list of imported topologies.

        """
        if not path:
            print("Topology.ByJSONPath - Error: the input path is not a valid path. Returning None.")
            return None
        data = None
        with open(path) as file:
            data = Topology.ByJSONFile(file=file, tolerance=tolerance)
        return data

    @staticmethod
    def ByOBJString(string, transposeAxes = True, progressBar=False, renderer="notebook", tolerance=0.0001):
        """
        Creates a topology from the input Waverfront OBJ string. This is a very experimental method and only works with simple planar solids. Materials and Colors are ignored.

        Parameters
        ----------
        string : str
            The input OBJ string.
        transposeAxes : bool , optional
            If set to True the Z and Y coordinates are transposed so that Y points "up" 
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topology
            The created topology.

        """
        from topologicpy.Vertex import Vertex
        def parse(lines):
            vertices = []
            faces = []
            for i in range(len(lines)):
                l = lines[i].replace(",", " ")
                s = l.split()
                if isinstance(s, list):
                    if len(s) > 3:
                        if s[0].lower() == "v":
                            vertices.append([float(s[1]), float(s[2]), float(s[3])])
                        elif s[0].lower() == "f":
                            temp_faces = []
                            for j in range(1,len(s)):
                                f = s[j].split("/")[0]
                                temp_faces.append(int(f)-1)
                            faces.append(temp_faces)
            return [vertices, faces]
        
        def parsetqdm(lines):
            vertices = []
            faces = []
            for i in tqdm(range(len(lines))):
                s = lines[i].split()
                if isinstance(s, list):
                    if len(s) > 3:
                        if s[0].lower() == "v":
                                vertices.append([float(s[1]), float(s[2]), float(s[3])])
                        elif s[0].lower() == "f":
                            temp_faces = []
                            for j in range(1,len(s)):
                                f = s[j].split("/")[0]
                                temp_faces.append(int(f)-1)
                            faces.append(temp_faces)
            return [vertices, faces]
        
        
        lines = string.split("\n")
        if lines:
            if progressBar:
                if renderer.lower() == "notebook":
                    from tqdm.notebook import tqdm
                else:
                    from tqdm import tqdm
                vertices, faces = parsetqdm(lines)
            else:
                vertices, faces = parse(lines)
        if vertices or faces:
            topology = Topology.ByGeometry(vertices = vertices, faces = faces, outputMode="default", tolerance=tolerance)
            if transposeAxes == True:
                topology = Topology.Rotate(topology, Vertex.Origin(), 1,0,0,90)
            return topology
        print("Topology.ByOBJString - Error: Could not find vertices or faces. Returning None.")
        return None

    @staticmethod
    def ByOBJFile(file, transposeAxes = True, progressBar=False, renderer="notebook", tolerance=0.0001):
        """
        Imports the topology from a Weverfront OBJ file. This is a very experimental method and only works with simple planar solids. Materials and Colors are ignored.

        Parameters
        ----------
        file : file object
            The input OBJ file.
        transposeAxes : bool , optional
            If set to True the Z and Y coordinates are transposed so that Y points "up" 
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topology
            The imported topology.

        """
        if not file:
            print("Topology.ByOBJFile - Error: the input file is not a valid file. Returning None.")
            return None
        obj_string = file.read()
        topology = Topology.ByOBJString(obj_string)
        file.close()
        return topology
    
    @staticmethod
    def ByOBJPath(path, transposeAxes = True, progressBar=False, renderer="notebook", tolerance=0.0001):
        """
        Imports the topology from a Weverfront OBJ file path. This is a very experimental method and only works with simple planar solids. Materials and Colors are ignored.

        Parameters
        ----------
        path : str
            The file path to the OBJ file.
        transposeAxes : bool , optional
            If set to True the Z and Y coordinates are transposed so that Y points "up" 
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topology
            The imported topology.

        """
        if not path:
            print("Topology.ByOBJPath - Error: the input path is not a valid path. Returning None.")
            return None
        try:
            file = open(path)
        except:
            print("Topology.ByOBJPath - Error: the OBJ file is not a valid file. Returning None.")
            return None
        return Topology.ByOBJFile(file, transposeAxes=transposeAxes, progressBar=progressBar, renderer=renderer, tolerance=tolerance)

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
    def ByBREPString(string):
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
            print("Topology.ByBREPString - Error: the input string is not a valid string. Returning None.")
            return None
        returnTopology = None
        try:
            returnTopology = topologic.Topology.ByString(string)
        except:
            print("Topology.ByBREPString - Error: the input string is not a valid string. Returning None.")
            returnTopology = None
        return returnTopology
    
    @staticmethod
    def ByXYZFile(file, frameIdKey="id", vertexIdKey="id"):
        """
        Imports the topology from an XYZ file path. This is a very experimental method. While variants of the format exist, topologicpy reads XYZ files that conform to the following:
        An XYZ file can be made out of one or more frames. Each frame will be stored in a sepatate topologic cluster.
        First line: total number of vertices in the frame. This must be an integer. No other words or characters are allowed on this line.
        Second line: frame label. This is free text and will be stored in the dictionary of each frame (topologic.Cluster)
        All other lines: vertex_label, x, y, and z coordinates, separated by spaces, tabs, or commas. The vertex label must be one word with no spaces. It is stored in the dictionary of each vertex.

        Example:
        3
        Frame 1
        A 5.67 -3.45 2.61
        B 3.91 -1.91 4
        A 3.2 1.2 -12.3
        4
        Frame 2
        B 5.47 -3.45 2.61
        B 3.91 -1.93 3.1
        A 3.2 1.2 -22.4
        A 3.2 1.2 -12.3
        3
        Frame 3
        A 5.67 -3.45 2.61
        B 3.91 -1.91 4
        C 3.2 1.2 -12.3

        Parameters
        ----------
        file : file object
            The input XYZ file.
        frameIdKey : str , optional
            The desired id key to use to store the ID of each frame in its dictionary. The default is "id".
        vertexIdKey : str , optional
            The desired id key to use to store the ID of each point in its dictionary. The default is "id".

        Returns
        -------
        list
            The list of frames (topologic.Cluster).

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Cluster import Cluster
        from topologicpy.Dictionary import Dictionary

        def parse(lines):
            frames = []
            line_index = 0
            while line_index < len(lines):
                try:
                    n_vertices = int(lines[line_index])
                except:
                    return frames
                frame_label = lines[line_index+1][:-1]
                vertices = []
                for i in range(n_vertices):
                    one_line = lines[line_index+2+i]
                    s = one_line.split()
                    vertex_label = s[0]
                    v = Vertex.ByCoordinates(float(s[1]), float(s[2]), float(s[3]))
                    vertex_dict = Dictionary.ByKeysValues([vertexIdKey], [vertex_label])
                    v = Topology.SetDictionary(v, vertex_dict)
                    vertices.append(v)
                frame = Cluster.ByTopologies(vertices)
                frame_dict = Dictionary.ByKeysValues([frameIdKey], [frame_label])
                frame = Topology.SetDictionary(frame, frame_dict)
                frames.append(frame)
                line_index = line_index + 2 + n_vertices
            return frames
        
        if not file:
            print("Topology.ByXYZFile - Error: the input file is not a valid file. Returning None.")
            return None
        lines = []
        for lineo, line in enumerate(file):
                lines.append(line)
        if len(lines) > 0:
            frames = parse(lines)
        file.close()
        return frames
    
    @staticmethod
    def ByXYZPath(path, frameIdKey="id", vertexIdKey="id"):
        """
        Imports the topology from an XYZ file path. This is a very experimental method. While variants of the format exist, topologicpy reads XYZ files that conform to the following:
        An XYZ file can be made out of one or more frames. Each frame will be stored in a sepatate topologic cluster.
        First line: total number of vertices in the frame. This must be an integer. No other words or characters are allowed on this line.
        Second line: frame label. This is free text and will be stored in the dictionary of each frame (topologic.Cluster)
        All other lines: vertex_label, x, y, and z coordinates, separated by spaces, tabs, or commas. The vertex label must be one word with no spaces. It is stored in the dictionary of each vertex.

        Example:
        3
        Frame 1
        A 5.67 -3.45 2.61
        B 3.91 -1.91 4
        A 3.2 1.2 -12.3
        4
        Frame 2
        B 5.47 -3.45 2.61
        B 3.91 -1.93 3.1
        A 3.2 1.2 -22.4
        A 3.2 1.2 -12.3
        3
        Frame 3
        A 5.67 -3.45 2.61
        B 3.91 -1.91 4
        C 3.2 1.2 -12.3

        Parameters
        ----------
        path : str
            The input XYZ file path.
        frameIdKey : str , optional
            The desired id key to use to store the ID of each frame in its dictionary. The default is "id".
        vertexIdKey : str , optional
            The desired id key to use to store the ID of each point in its dictionary. The default is "id".

        Returns
        -------
        list
            The list of frames (topologic.Cluster).

        """
        if not path:
            print("Topology.ByXYZPath - Error: the input path is not a valid path. Returning None.")
            return None
        try:
            file = open(path)
        except:
            print("Topology.ByXYZPath - Error: the XYZ file is not a valid file. Returning None.")
            return None
        return Topology.ByXYZFile(file, frameIdKey=frameIdKey, vertexIdKey=frameIdKey)
    
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
            print("Topology.CenterofMass - Error: the input topology is not a valid topology. Returning None.")
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
            print("Topology.Centroid - Error: the input topology is not a valid topology. Returning None.")
            return None
        if isinstance(topology, topologic.Aperture):
            return Aperture.Topology(topology).Centroid()
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
        if not isinstance(topology, topologic.Topology):
            print("Topology.ClusterFaces - Error: the input topology is not a valid topology. Returning None.")
            return None
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
                returnList.append(Topology.SelfMerge(Cluster.ByTopologies(tempList)))
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
        if not isinstance(topology, topologic.Topology):
            print("Topology.Contents - Error: the input topology is not a valid topology. Returning None.")
            return None
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
        if not isinstance(topology, topologic.Topology):
            print("Topology.Contexts - Error: the input topology is not a valid topology. Returning None.")
            return None
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
        import sys
        import subprocess

        try:
            from scipy.spatial import ConvexHull
        except:
            call = [sys.executable, '-m', 'pip', 'install', 'scipy', '-t', sys.path[0]]
            subprocess.run(call)
            try:
                from scipy.spatial import ConvexHull
            except:
                print("Topology.ConvexHull - Error: Could not import scipy. Returning None.")
                return None
        
        def convexHull3D(item, tolerance, option):
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
                c = Cell.ByFaces(faces, tolerance=tolerance)
                return c
            except:
                returnTopology = Topology.SelfMerge(Cluster.ByTopologies(faces))
                if returnTopology.Type() == 16:
                    return Shell.ExternalBoundary(returnTopology)
        if not isinstance(topology, topologic.Topology):
            print("Topology.ConvexHull - Error: the input topology is not a valid topology. Returning None.")
            return None
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
        if not isinstance(topology, topologic.Topology):
            print("Topology.Copy - Error: the input topology is not a valid topology. Returning None.")
            return None
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
            print("Topology.Dictionary - Error: the input topology is not a valid topology. Returning None.")
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
        if not isinstance(topology, topologic.Topology):
            print("Topology.Dimensionality - Error: the input topology is not a valid topology. Returning None.")
            return None
        return topology.Dimensionality()
    
    @staticmethod
    def Divide(topologyA, topologyB, transferDictionary=False, addNestingDepth=False):
        """
        Divides the input topology by the input tool and places the results in the contents of the input topology.

        Parameters
        ----------
        topologyA : topologic.Topology
            The input topology to be divided.
        topologyB : topologic.Topology
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
        if not isinstance(topologyA, topologic.Topology):
            print("Topology.Divide - Error: the input topologyA is not a valid topology. Returning None.")
            return None
        if not isinstance(topologyB, topologic.Topology):
            print("Topology.Divide - Error: the input topologyB is not a valid topology. Returning None.")
            return None
        try:
            _ = topologyA.Divide(topologyB, False) # Don't transfer dictionaries just yet
        except:
            raise Exception("TopologyDivide - Error: Divide operation failed.")
        nestingDepth = "1"
        keys = ["nesting_depth"]
        values = [nestingDepth]

        if not addNestingDepth and not transferDictionary:
            return topologyA

        contents = []
        _ = topologyA.Contents(contents)
        for i in range(len(contents)):
            if not addNestingDepth and transferDictionary:
                parentDictionary = Topology.Dictionary(topologyA)
                if parentDictionary != None:
                    _ = contents[i].SetDictionary(parentDictionary)
            if addNestingDepth and transferDictionary:
                parentDictionary = Topology.Dictionary(topologyA)
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
                _ = Topology.SetDictionary(topologyA, parentDictionary)
                values[keys.index("nesting_depth")] = nestingDepth+"_"+str(i+1)
                d = Dictionary.ByKeysValues(keys, values)
                _ = contents[i].SetDictionary(d)
            if addNestingDepth and  not transferDictionary:
                parentDictionary = Topology.Dictionary(topologyA)
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
                _ = Topology.SetDictionary(topologyA, parentDictionary)
                keys = ["nesting_depth"]
                v = nestingDepth+"_"+str(i+1)
                values = [v]
                d = Dictionary.ByKeysValues(keys, values)
                _ = Topology.SetDictionary(contents[i], d)
        return topologyA
    
    @staticmethod
    def Explode(topology, origin=None, scale=1.25, typeFilter=None, axes="xyz"):
        """
        Explodes the input topology. See https://en.wikipedia.org/wiki/Exploded-view_drawing.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        origin : topologic.Vertex , optional
            The origin of the explosion. If set to None, the centroid of the input topology will be used. The default is None.
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
            if len(Cluster.CellComplexes(cluster)) > 0:
                return "cell"
            elif len(Cluster.Cells(cluster)) > 0:
                return "face"
            elif len(Cluster.Shells(cluster)) > 0:
                return "face"
            elif len(Cluster.Faces(cluster)) > 0:
                return "edge"
            elif len(Cluster.Wires(cluster)) > 0:
                return "edge"
            elif len(Cluster.Edges(cluster)) > 0:
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
            print("Topology.Explode - Error: the input topology is not a valid topology. Returning None.")
            return None
        if not isinstance(origin, topologic.Vertex):
            origin = Topology.CenterOfMass(topology)
        if not typeFilter:
            typeFilter = getTypeFilter(topology)
        if not isinstance(typeFilter, str):
            print("Topology.Explode - Error: the input typeFilter is not a valid string. Returning None.")
            return None
        if not isinstance(axes, str):
            print("Topology.Explode - Error: the input axes is not a valid string. Returning None.")
            return None
        axes = axes.lower()
        x_flag = "x" in axes
        y_flag = "y" in axes
        z_flag = "z" in axes
        if not x_flag and not y_flag and not z_flag:
            print("Topology.Explode - Error: the input axes is not a valid string. Returning None.")
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
    def ExportToBREP(topology, path, overwrite=False, version=3):
        """
        Exports the input topology to a BREP file. See https://dev.opencascade.org/doc/occt-6.7.0/overview/html/occt_brep_format.html.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        path : str
            The input file path.
        overwrite : bool , optional
            If set to True the ouptut file will overwrite any pre-existing file. Otherwise, it won't. The default is False.
        version : int , optional
            The desired version number for the BREP file. The default is 3.

        Returns
        -------
        bool
            True if the export operation is successful. False otherwise.

        """
        from os.path import exists
        if not isinstance(topology, topologic.Topology):
            print("Topology.ExportToBREP - Error: the input topology is not a valid topology. Returning None.")
            return None
        if not isinstance(path, str):
            print("Topology.ExportToBREP - Error: the input path is not a valid string. Returning None.")
            return None
        # Make sure the file extension is .brep
        ext = path[len(path)-5:len(path)]
        if ext.lower() != ".brep":
            path = path+".brep"
        if not overwrite and exists(path):
            print("Topology.ExportToBREP - Error: a file already exists at the specified path and overwrite is set to False. Returning None.")
            return None
        f = None
        try:
            if overwrite == True:
                f = open(path, "w")
            else:
                f = open(path, "x") # Try to create a new File
        except:
            raise Exception("Error: Could not create a new file at the following location: "+path)
        if (f):
            s = Topology.BREPString(topology, version)
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
        
        def exportToBREP(topology, path, overwrite):
            # Make sure the file extension is .brep
            ext = path[len(path)-5:len(path)]
            if ext.lower() != ".brep":
                path = path+".brep"
            f = None
            try:
                if overwrite == True:
                    f = open(path, "w")
                else:
                    f = open(path, "x") # Try to create a new File
            except:
                raise Exception("Error: Could not create a new file at the following location: "+path)
            if (f):
                topString = topology.BREPString()
                f.write(topString)
                f.close()	
                return True
            return False
        
        path = os.path.expanduser('~')+"/tempFile.brep"
        if exportToBREP(topology, path, True):
            url = url.replace('http://','')
            url = '/dns/'+url+'/tcp/'+port+'/https'
            client = ipfshttpclient.connect(url, auth=(user, password))
            newfile = client.add(path)
            os.remove(path)
            return newfile['Hash']
        return ''
    '''
    @staticmethod
    def ExportToJSON(topologies, path, overwrite=False):
        """
        Exports the input list of topologies to a JSON file.

        Parameters
        ----------
        topologies : list
            The input list of topologies.
        path : str
            The path to the JSON file.
        overwrite : bool , optional
            If set to True the ouptut file will overwrite any pre-existing file. Otherwise, it won't. The default is False.

        Returns
        -------
        bool
            The status of exporting the JSON file. If True, the operation was successful. Otherwise, it was unsuccesful.

        """
        from os.path import exists
        # Make sure the file extension is .json
        ext = path[len(path)-5:len(path)]
        if ext.lower() != ".json":
            path = path+".json"
        if not overwrite and exists(path):
            print("Topology.ExportToJSON - Error: a file already exists at the specified path and overwrite is set to False. Returning None.")
            return None
        f = None
        try:
            if overwrite == True:
                f = open(path, "w")
            else:
                f = open(path, "x") # Try to create a new File
        except:
            raise Exception("Error: Could not create a new file at the following location: "+path)
        if (f):
            jsondata = json.loads(Topology.JSONString(topologies))
            if jsondata != None:
                json.dump(jsondata, f, indent=4, sort_keys=True)
                f.close()
                return True
            else:
                f.close()
                return False
        return False

    @staticmethod
    def Fix(topology, topologyType="CellComplex"):
        """
        Attempts to fix the input topology to matched the desired output type.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology
        topologyType : str , optional
            The desired output topology type. This must be one of "vertex", "edge", "wire", "face", "shell", "cell", "cellcomplex", "cluster". It is case insensitive. The default is "CellComplex"
        
        Returns
        -------
        topologic.Topology
            The output topology in the desired type.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Shell import Shell
        from topologicpy.Cell import Cell
        from topologicpy.CellComplex import CellComplex
        from topologicpy.Cluster import Cluster

        topology = Cluster.ByTopologies([topology])
        a_type = Topology.TypeAsString(topology).lower()
        b_type = topologyType.lower()
        if b_type not in ["vertex", "edge", "wire", "face", "shell", "cell", "cellcomplex", "cluster"]:
            print("Topology.Fix - Error: Desired topologyType is not recognized. Returning original topology.")
            return topology
        if a_type == b_type:
            return topology
        if b_type == "cluster":
            topology = Topology.SelfMerge(topology)
            return Cluster.ByTopologies([topology])
        if b_type == "cellcomplex":
            topology = Topology.SelfMerge(topology)
            if Topology.TypeAsString(topology).lower() == "cellcomplex":
                return topology
            cells = Topology.Cells(topology)
            if len(cells) < 2:
                print("Topology.Fix - Error: Desired topologyType cannot be achieved. Returning original topology.")
                return topology
            return_topology = CellComplex.ByCells(cells)
            if return_topology == None:
                print("Topology.Fix - Error: Desired topologyType cannot be achieved. Returning original topology.")
                return topology
            if Topology.TypeAsString(topology).lower() == "cellcomplex":
                return return_topology
            faces = Topology.Faces(topology)
            if len(faces) < 3:
                print("Topology.Fix - Error: Desired topologyType cannot be achieved. Returning original topology.")
                return topology
            return_topology = CellComplex.ByFaces(faces)
            if return_topology == None:
                print("Topology.Fix - Error: Desired topologyType cannot be achieved. Returning original topology.")
                return topology
            if Topology.TypeAsString(return_topology).lower() == "cellcomplex":
                return return_topology
            print("6 Topology.Fix - Error: Desired topologyType cannot be achieved. Returning original topology.")
            return topology
        if b_type == "cell":
            topology = Topology.SelfMerge(topology)
            if Topology.TypeAsString(topology).lower() == "cell":
                return topology
            if Topology.TypeAsString(topology).lower() == "cellComplex":
                return Cell.Complex.ExternalBoundary(topology)
            faces = Topology.Faces(topology)
            if len(faces) < 3:
                print("Topology.Fix - Error: Desired topologyType cannot be achieved. Returning original topology.")
                return topology
            return_topology = Cell.ByFaces(faces)
            if return_topology == None:
                return_topology = CellComplex.ByFaces(faces)
                print("Return Topology:", return_topology)
                if return_topology == None:
                    print("Topology.Fix - Error: Desired topologyType cannot be achieved. Returning original topology.")
                    return topology
                elif len(Topology.Cells(return_topology)) < 1:
                    print("Topology.Fix - Error: Desired topologyType cannot be achieved. Returning original topology.")
                    return topology
                return_topology = CellComplex.ExternalBoundary(return_topology)
            if return_topology == None:
                print("Topology.Fix - Error: Desired topologyType cannot be achieved. Returning original topology.")
                return topology
            if Topology.TypeAsString(return_topology).lower() == "cell":
                return return_topology
            print("Topology.Fix - Error: Desired topologyType cannot be achieved. Returning original topology.")
            return topology
        if b_type == "shell":
            topology = Topology.SelfMerge(topology)
            if Topology.TypeAsString(topology).lower() == "shell":
                return topology
            faces = Topology.Faces(topology)
            if len(faces) < 2:
                print("Topology.Fix - Error: Desired topologyType cannot be achieved. Returning original topology.")
                return topology
            return_topology = Shell.ByFaces(faces)
            if Topology.TypeAsString(return_topology).lower() == "shell":
                return return_topology
            print("Topology.Fix - Error: Desired topologyType cannot be achieved. Returning original topology.")
            return topology
        if b_type == "face":
            topology = Topology.SelfMerge(topology)
            if Topology.TypeAsString(topology).lower() == "face":
                return topology
            wires = Topology.Wires(topology)
            if len(wires) < 1:
                print("Topology.Fix - Error: Desired topologyType cannot be achieved. Returning original topology.")
                return topology
            return_topology = Face.ByWire(wires[0])
            if Topology.TypeAsString(return_topology).lower() == "face":
                return return_topology
            print("Topology.Fix - Error: Desired topologyType cannot be achieved. Returning original topology.")
            return topology
        if b_type == "wire":
            topology = Topology.SelfMerge(topology)
            if Topology.TypeAsString(topology).lower() == "wire":
                return topology
            edges = Topology.Edges(topology)
            if len(edges) < 2:
                print("Topology.Fix - Error: Desired topologyType cannot be achieved. Returning original topology.")
                return topology
            return_topology = Wire.ByEdges(edges)
            if Topology.TypeAsString(return_topology).lower() == "wire":
                return return_topology
            print("Topology.Fix - Error: Desired topologyType cannot be achieved. Returning original topology.")
            return topology
        if b_type == "edge":
            topology = Topology.SelfMerge(topology)
            if Topology.TypeAsString(topology).lower() == "edge":
                return topology
            vertices = Topology.Vertices(topology)
            if len(vertices) < 2:
                print("Topology.Fix - Error: Desired topologyType cannot be achieved. Returning original topology.")
                return topology
            return_topology = Edge.ByVertices(vertices)
            if Topology.TypeAsString(return_topology).lower() == "edge":
                return return_topology
            print("Topology.Fix - Error: Desired topologyType cannot be achieved. Returning original topology.")
            return topology
        if b_type == "vertex":
            topology = Topology.SelfMerge(topology)
            if Topology.TypeAsString(topology).lower() == "vertex":
                return topology
            vertices = Topology.Vertices(topology)
            if len(vertices) < 1:
                print("Topology.Fix - Error: Desired topologyType cannot be achieved. Returning original topology.")
                return topology
            return_topology = vertices[0]
            if Topology.TypeAsString(return_topology).lower() == "vertex":
                return return_topology
            print("Topology.Fix - Error: Desired topologyType cannot be achieved. Returning original topology.")
            return topology
        return topology

    @staticmethod
    def JSONString(topologies):
        """
        Exports the input list of topologies to a JSON string

        Parameters
        ----------
        topologies : list or topologic.Topology
            The input list of topologies or a single topology.

        Returns
        -------
        bool
            The status of exporting the JSON file. If True, the operation was successful. Otherwise, it was unsuccesful.

        """

        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Cell import Cell
        from topologicpy.Dictionary import Dictionary

        def getUUID(topology, uuidKey="uuid"):
            d = Topology.Dictionary(topology)
            if d == None:
                print("Dictionary is None for", topology)
            if uuidKey not in Dictionary.Keys(d):
                uuidOne = str(uuid.uuid1())
                d = Dictionary.SetValueAtKey(d, uuidKey, uuidOne)
                topology = Topology.SetDictionary(topology, d)
            else:
                uuidOne = Dictionary.ValueAtKey(d, uuidKey)
            return uuidOne

        def getVertex(topology, uuidKey="uuid"):
            returnDict = {}
            uuidOne = getUUID(topology, uuidKey=uuidKey)
            returnDict['type'] = "Vertex"
            returnDict['uuid'] = uuidOne
            returnDict['coordinates'] = [Vertex.X(topology), Vertex.Y(topology), Vertex.Z(topology)]
            returnDict['dictionary'] = Dictionary.PythonDictionary(Topology.Dictionary(topology))
            return returnDict

        def getEdge(topology, uuidKey="uuid"):
            returnDict = {}
            uuidOne = getUUID(topology, uuidKey=uuidKey)
            returnDict['type'] = "Edge"
            returnDict['uuid'] = uuidOne
            returnDict['vertices'] = [getUUID(v, uuidKey=uuidKey) for v in Edge.Vertices(topology)]
            returnDict['dictionary'] = Dictionary.PythonDictionary(Topology.Dictionary(topology))
            return returnDict

        def getWire(topology, uuidKey="uuid"):
            returnDict = {}
            uuidOne = getUUID(topology, uuidKey="uuid")
            returnDict['type'] = "Wire"
            returnDict['uuid'] = uuidOne
            returnDict['edges'] = [getUUID(e, uuidKey=uuidKey) for e in Topology.Edges(topology)]
            returnDict['dictionary'] = Dictionary.PythonDictionary(Topology.Dictionary(topology))
            return returnDict

        def getFace(topology, uuidKey="uuid"):
            apertures = [Aperture.Topology(ap) for ap in Topology.Apertures(topology)]
            returnDict = {}
            uuidOne = getUUID(topology, uuidKey=uuidKey)
            returnDict['type'] = "Face"
            returnDict['uuid'] = uuidOne
            wires = []
            external_boundary = Face.ExternalBoundary(topology)
            wires.append(getUUID(Face.ExternalBoundary(topology), uuidKey=uuidKey))
            internal_boundaries = [getUUID(ib, uuidKey=uuidKey) for ib in Face.InternalBoundaries(topology)]
            wires += internal_boundaries
            returnDict['wires'] = wires
            dictionary = Dictionary.PythonDictionary(Topology.Dictionary(topology))
            returnDict['dictionary'] = dictionary
            return returnDict

        def getShell(topology, uuidKey="uuid"):
            returnDict = {}
            uuidOne = getUUID(topology, uuidKey=uuidKey)
            returnDict['type'] = "Shell"
            returnDict['uuid'] = uuidOne
            returnDict['faces'] = [getUUID(f, uuidKey=uuidKey) for f in Topology.Faces(topology)]
            returnDict['dictionary'] = Dictionary.PythonDictionary(Topology.Dictionary(topology))
            return returnDict

        def getCell(topology, uuidKey="uuid"):
            returnDict = {}
            uuidOne = getUUID(topology, uuidKey=uuidKey)
            returnDict['type'] = "Cell"
            returnDict['uuid'] = uuidOne
            shells = []
            external_boundary = Cell.ExternalBoundary(topology)
            shells.append(getUUID(external_boundary, uuidKey=uuidKey))
            internal_boundaries = [getUUID(ib, uuidKey=uuidKey) for ib in Cell.InternalBoundaries(topology)]
            shells += internal_boundaries
            returnDict['shells'] = shells
            dictionary = Dictionary.PythonDictionary(Topology.Dictionary(topology))
            returnDict['dictionary'] = dictionary
            return returnDict

        def getCellComplex(topology, uuidKey="uuid"):
            returnDict = {}
            uuidOne = getUUID(topology, uuidKey=uuidKey)
            returnDict['type'] = "CellComplex"
            returnDict['uuid'] = uuidOne
            returnDict['cells'] = [getUUID(c, uuidKey=uuidKey) for c in Topology.Cells(topology)]
            returnDict['dictionary'] = Dictionary.PythonDictionary(Topology.Dictionary(topology))
            return returnDict


        def getApertureData(topology, topLevel="False", uuidKey="uuid"):
            json_data = []
            if isinstance(topology, topologic.Vertex):
                d = getVertex(topology, uuidKey=uuidKey)
            elif isinstance(topology, topologic.Edge):
                d = getEdge(topology, uuidKey=uuidKey)
            elif isinstance(topology, topologic.Wire):
                d = getWire(topology, uuidKey=uuidKey)
            elif isinstance(topology, topologic.Face):
                d = getFace(topology, uuidKey=uuidKey)
            elif isinstance(topology, topologic.Shell):
                d = getShell(topology, uuidKey=uuidKey)
            elif isinstance(topology, topologic.Cell):
                d = getCell(topology, uuidKey=uuidKey)
            elif isinstance(topology, topologic.CellComplex):
                d = getCellComplex(topology, uuidKey=uuidKey)
            d['dictionary']['toplevel'] = topLevel
            json_data += getSubTopologyData(topology, uuidKey=uuidKey)
            apertures = [Aperture.Topology(ap) for ap in Topology.Apertures(topology)]
            aperture_data = []
            for ap in apertures:
                aperture_data.append(getApertureData(ap, topLevel=False, uuidKey=uuidKey))
            d['apertures'] = aperture_data
            json_data.append(d)
            return json_data

        def getSubTopologyData(topology, uuidKey="uuid"):
            json_data = []
            vertices = Topology.Vertices(topology)
            for v in vertices:
                d = getVertex(v, uuidKey=uuidKey)
                d['dictionary']['toplevel'] = False
                apertures = [Aperture.Topology(ap) for ap in Topology.Apertures(v)]
                aperture_data = []
                for ap in apertures:
                    aperture_data.append(getApertureData(ap, topLevel=False, uuidKey=uuidKey))
                d['apertures'] = aperture_data
                json_data.append(d)
            edges = Topology.Edges(topology)
            for e in edges:
                d = getEdge(e, uuidKey=uuidKey)
                d['dictionary']['toplevel'] = False
                apertures = [Aperture.Topology(ap) for ap in Topology.Apertures(e)]
                aperture_data = []
                for ap in apertures:
                    aperture_data.append(getApertureData(ap, topLevel=False, uuidKey=uuidKey))
                d['apertures'] = aperture_data
                json_data.append(d)
            wires = Topology.Wires(topology)
            for w in wires:
                d = getWire(w, uuidKey=uuidKey)
                d['dictionary']['toplevel'] = False
                apertures = [Aperture.Topology(ap) for ap in Topology.Apertures(w)]
                aperture_data = []
                for ap in apertures:
                    aperture_data.append(getApertureData(ap, topLevel=False, uuidKey=uuidKey))
                d['apertures'] = aperture_data
                json_data.append(d)
            faces = Topology.Faces(topology)
            for f in faces:
                d = getFace(f, uuidKey=uuidKey)
                d['dictionary']['toplevel'] = False
                apertures = [Aperture.Topology(ap) for ap in Topology.Apertures(f)]
                aperture_data = []
                for ap in apertures:
                    aperture_data.append(getApertureData(ap, topLevel=False, uuidKey=uuidKey))
                d['apertures'] = aperture_data
                json_data.append(d)
            shells = Topology.Shells(topology)
            for s in shells:
                d = getShell(s, uuidKey=uuidKey)
                d['dictionary']['toplevel'] = False
                apertures = [Aperture.Topology(ap) for ap in Topology.Apertures(s)]
                aperture_data = []
                for ap in apertures:
                    aperture_data.append(getApertureData(ap, topLevel=False, uuidKey=uuidKey))
                d['apertures'] = aperture_data
                json_data.append(d)
            cells = Topology.Cells(topology)
            for c in cells:
                d = getCell(c, uuidKey=uuidKey)
                d['dictionary']['toplevel'] = False
                apertures = [Aperture.Topology(ap) for ap in Topology.Apertures(c)]
                aperture_data = []
                for ap in apertures:
                    aperture_data.append(getApertureData(ap, topLevel=False, uuidKey=uuidKey))
                d['apertures'] = aperture_data
                json_data.append(d)
            cellComplexes = Topology.CellComplexes(topology)
            for cc in cellComplexes:
                d = getCellComplex(cc, uuidKey=uuidKey)
                d['dictionary']['toplevel'] = False
                apertures = [Aperture.Topology(ap) for ap in Topology.Apertures(cc)]
                aperture_data = []
                for ap in apertures:
                    aperture_data.append(getApertureData(ap, topLevel=False, uuidKey=uuidKey))
                d['apertures'] = aperture_data
                json_data.append(d)
            return json_data

        def getJSONData(topology, topLevel=False, uuidKey="uuid"):
            json_data = []
            if isinstance(topology, topologic.Vertex):
                d = getVertex(topology, uuidKey=uuidKey)
            elif isinstance(topology, topologic.Edge):
                d = getEdge(topology, uuidKey=uuidKey)
            elif isinstance(topology, topologic.Wire):
                d = getWire(topology, uuidKey=uuidKey)
            elif isinstance(topology, topologic.Face):
                d = getFace(topology, uuidKey=uuidKey)
            elif isinstance(topology, topologic.Shell):
                d = getShell(topology, uuidKey=uuidKey)
            elif isinstance(topology, topologic.Cell):
                d = getCell(topology, uuidKey=uuidKey)
            elif isinstance(topology, topologic.CellComplex):
                d = getCellComplex(topology, uuidKey=uuidKey)
            d['dictionary']['toplevel'] = topLevel
            json_data += getSubTopologyData(topology, uuidKey=uuidKey)
            apertures = [Aperture.Topology(ap) for ap in Topology.Apertures(topology)]
            aperture_data = []
            for ap in apertures:
                aperture_data.append(getApertureData(ap, topLevel=False, uuidKey=uuidKey))
            d['apertures'] = aperture_data
            json_data.append(d)
            return json_data
        json_data = []
        if not isinstance(topologies, list):
            topologies = [topologies]
        topologies = [x for x in topologies if isinstance(x, topologic.Topology)]
        for topology in topologies:
            json_data += getJSONData(topology, topLevel=True, uuidKey="uuid")
        json_string = json.dumps(json_data, indent=4, sort_keys=False)
        return json_string
    
    @staticmethod
    def OBJString(topology, transposeAxes=True):
        """
        Returns the Wavefront string of the input topology. This is very experimental and outputs a simple solid topology.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        transposeAxes : bool , optional
            If set to True the Z and Y coordinates are transposed so that Y points "up" 

        Returns
        -------
        str
            The Wavefront OBJ string of the input topology

        """
        from topologicpy.Helper import Helper
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face

        if not isinstance(topology, topologic.Topology):
            print("Topology.ExportToOBJ - Error: the input topology is not a valid topology. Returning None.")
            return None
        
        lines = []
        version = Helper.Version()
        lines.append("# topologicpy "+version)
        d = Topology.Geometry(topology)
        vertices = d['vertices']
        faces = d['faces']
        tVertices = []
        if transposeAxes:
            for v in vertices:
                tVertices.append([v[0], v[2], v[1]])
            vertices = tVertices
        for v in vertices:
            lines.append("v "+str(v[0])+" "+str(v[1])+" "+str(v[2]))
        for f in faces:
            line = "f"
            for j in f:
                line = line+" "+str(j+1)
            lines.append(line)
        finalLines = lines[0]
        for i in range(1,len(lines)):
            finalLines = finalLines+"\n"+lines[i]
        return finalLines
    
    @staticmethod
    def ExportToOBJ(topology, path, transposeAxes=True, overwrite=False):
        """
        Exports the input topology to a Wavefront OBJ file. This is very experimental and outputs a simple solid topology.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        path : str
            The input file path.
        transposeAxes : bool , optional
            If set to True the Z and Y coordinates are transposed so that Y points "up" 
        overwrite : bool , optional
            If set to True the ouptut file will overwrite any pre-existing file. Otherwise, it won't. The default is False.

        Returns
        -------
        bool
            True if the export operation is successful. False otherwise.

        """
        from os.path import exists

        if not isinstance(topology, topologic.Topology):
            print("Topology.ExportToOBJ - Error: the input topology is not a valid topology. Returning None.")
            return None
        if not overwrite and exists(path):
            print("Topology.ExportToOBJ - Error: a file already exists at the specified path and overwrite is set to False. Returning None.")
            return None
        
        # Make sure the file extension is .obj
        ext = path[len(path)-4:len(path)]
        if ext.lower() != ".obj":
            path = path+".obj"
        status = False
        objString = Topology.OBJString(topology, transposeAxes=transposeAxes)
        with open(path, "w") as f:
            f.writelines(objString)
            f.close()
            status = True
        return status

    @staticmethod
    def Filter(topologies, topologyType="any", searchType="any", key=None, value=None):
        """
        Filters the input list of topologies based on the input parameters.

        Parameters
        ----------
        topologies : list
            The input list of topologies.
        topologyType : str , optional
            The type of topology to filter by. This can be one of "any", "vertex", "edge", "wire", "face", "shell", "cell", "cellcomplex", or "cluster". It is case insensitive. The default is "any".
        searchType : str , optional
            The type of search query to conduct in the topology's dictionary. This can be one of "any", "equal to", "contains", "starts with", "ends with", "not equal to", "does not contain". The default is "any".
        key : str , optional
            The dictionary key to search within. The default is None which means it will filter by topology type only.
        value : str , optional
            The value to search for at the specified key. The default is None which means it will filter by topology type only.

        Returns
        -------
        dict
            A dictionary of filtered and other elements. The dictionary has two keys:
            - "filtered" The filtered topologies.
            - "other" The other topologies that did not meet the filter criteria.

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
            if (topologyType.lower() == "any") or (Topology.TypeAsString(aTopology).lower() == topologyType.lower()):
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
        return {"filtered": filteredTopologies, "other": otherTopologies}

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
        if not isinstance(topology, topologic.Topology):
            print("Topology.Flatten - Error: the input topology is not a valid topology. Returning None.")
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
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Cell import Cell
        from topologicpy.CellComplex import CellComplex
        from topologicpy.Aperture import Aperture
        if not isinstance(topology, topologic.Topology):
            print("Topology.InternalVertex - Error: the input topology is not a valid topology. Returning None.")
            return None
        vst = None
        classType = topology.Type()
        if isinstance(topology, topologic.CellComplex): #CellComplex
            tempCell = Topology.Cells(topology)[0]
            vst = Cell.InternalVertex(tempCell, tolerance)
        elif isinstance(topology, topologic.Cell): #Cell
            vst = Cell.InternalVertex(topology, tolerance)
        elif isinstance(topology, topologic.Shell): #Shell
            tempFace = Topology.Faces(topology)[0]
            vst = Face.InternalVertex(tempFace, tolerance)
        elif isinstance(topology, topologic.Face): #Face
            vst = Face.InternalVertex(topology, tolerance)
        elif isinstance(topology, topologic.Wire): #Wire
            if topology.IsClosed():
                internalBoundaries = []
                try:
                    tempFace = topologic.Face.ByExternalInternalBoundaries(topology, internalBoundaries)
                    vst = Face.InternalVertex(tempFace, tolerance)
                except:
                    vst = Topology.Centroid(topology)
            else:
                tempEdge = Topology.Edges(topology)[0]
                vst = Edge.VertexByParameter(tempEdge, 0.5)
        elif isinstance(topology, topologic.Edge): #Edge
            vst = Edge.VertexByParameter(topology, 0.5)
        elif isinstance(topology, topologic.Vertex): #Vertex
            vst = topology
        elif isinstance(topology, topologic.Aperture): #Aperture
            vst = Face.InternalVertex(Aperture.Topology(topology), tolerance)
        else:
            vst = Topology.Centroid(topology)
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
            print("Topology.IsInside - Error: the input topology is not a valid topology. Returning None.")
            return None
        if not isinstance(vertex, topologic.Vertex):
            print("Topology.IsInside - Error: the input vertex is not a valid vertex. Returning None.")
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
            if not u == None:
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
            print("Topology.IsPlanar", topology)
            print("Topology.IsPlanar - Error: the input topology is not a valid topology. Returning None.")
            return None
        vertices = Topology.Vertices(topology)

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
        if not isinstance(topologyA, topologic.Topology):
            print("Topology.IsSame - Error: the input topologyA is not a valid topology. Returning None.")
            return None
        if not isinstance(topologyB, topologic.Topology):
            print("Topology.IsSame - Error: the input topologyB is not a valid topology. Returning None.")
            return None
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
        topologic.Topology
            The resulting merged Topology

        """

        from topologicpy.Cluster import Cluster
        if not isinstance(topologies, list):
            print("Topology.MergeAll - Error: the input topologies is not a valid list. Returning None.")
            return None
        
        topologyList = [t for t in topologies if isinstance(t, topologic.Topology)]
        if len(topologyList) < 1:
            print("Topology.MergeAll - Error: the input topologyList does not contain any valid topologies. Returning None.")
            return None
        return Topology.SelfMerge(Cluster.ByTopologies(topologyList))
            
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
        if not isinstance(topology, topologic.Topology):
            print("Topology.OCCTShape - Error: the input topology is not a valid topology. Returning None.")
            return None
        return topology.GetOcctShape()
    
    @staticmethod
    def Degree(topology, hostTopology):
        """
        Returns the number of immediate super topologies that use the input topology

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        hostTopology : topologic.Topology
            The input host topology to which the input topology belongs
        
        Returns
        -------
        int
            The degree of the topology (the number of immediate super topologies that use the input topology).
        
        """
        if not isinstance(topology, topologic.Topology):
            print("Topology.Degree - Error: the input topology is not a valid topology. Returning None.")
            return None
        if not isinstance(hostTopology, topologic.Topology):
            print("Topology.Degree - Error: the input hostTopology is not a valid topology. Returning None.")
            return None
        
        hostTopologyType = Topology.TypeAsString(hostTopology).lower()
        type = Topology.TypeAsString(topology).lower()
        superType = ""
        if type == "vertex" and (hostTopologyType == "cellcomplex" or hostTopologyType == "cell" or hostTopologyType == "shell"):
            superType = "face"
        elif type == "vertex" and (hostTopologyType == "wire" or hostTopologyType == "edge"):
            superType = "edge"
        elif type == "edge" and (hostTopologyType == "cellcomplex" or hostTopologyType == "cell" or hostTopologyType == "shell"):
            superType = "face"
        elif type == "face" and (hostTopologyType == "cellcomplex"):
            superType = "cell"
        superTopologies = Topology.SuperTopologies(topology, hostTopology=hostTopology, topologyType=superType)
        if not superTopologies:
            return 0
        return len(superTopologies)

    @staticmethod
    def NonPlanarFaces(topology, tolerance=0.0001):
        """
        Returns any nonplanar faces in the input topology
        
        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        Returns
        -------
        list
            The list of nonplanar faces.
        
        """
        if not isinstance(topology, topologic.Topology):
            print("Topology.NonPlanarFaces - Error: the input topology is not a valid topology. Returning None.")
            return None
        faces = Topology.SubTopologies(topology, subTopologyType="face")
        return [f for f in faces if not Topology.IsPlanar(f, tolerance=tolerance)]
    
    @staticmethod
    def OpenFaces(topology):
        """
        Returns the faces that border no cells.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        
        Returns
        -------
        list
            The list of open edges.
        
        """

        if not isinstance(topology, topologic.Topology):
            print("Topology.OpenFaces - Error: the input topology is not a valid topology. Returning None.")
            return None
        
        return [f for f in Topology.SubTopologies(topology, subTopologyType="face") if Topology.Degree(f, hostTopology=topology) < 1]
    
    @staticmethod
    def OpenEdges(topology):
        """
        Returns the edges that border only one face.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        
        Returns
        -------
        list
            The list of open edges.
        
        """

        if not isinstance(topology, topologic.Topology):
            print("Topology.OpenEdges - Error: the input topology is not a valid topology. Returning None.")
            return None
        
        return [e for e in Topology.SubTopologies(topology, subTopologyType="edge") if Topology.Degree(e, hostTopology=topology) < 2]
    
    @staticmethod
    def OpenVertices(topology):
        """
        Returns the vertices that border only one edge.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        
        Returns
        -------
        list
            The list of open edges.
        
        """

        if not isinstance(topology, topologic.Topology):
            print("Topology.OpenVertices - Error: the input topology is not a valid topology. Returning None.")
            return None
        
        return [v for v in Topology.SubTopologies(topology, subTopologyType="vertex") if Topology.Degree(v, hostTopology=topology) < 2]
    
    @staticmethod
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
            print("Topology.Orient - Error: the input topology is not a valid topology. Returning None.")
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
        world_origin = Vertex.ByCoordinates(0,0,0)

        phi = math.degrees(math.atan2(dy, dx)) # Rotation around Y-Axis
        if dist < 0.0001:
            theta = 0
        else:
            theta = math.degrees(math.acos(dz/dist)) # Rotation around Z-Axis
        returnTopology = Topology.Rotate(topology, world_origin, 0, 1, 0, theta)
        returnTopology = Topology.Rotate(returnTopology, world_origin, 0, 0, 1, phi)
        returnTopology = Topology.Place(returnTopology, world_origin, origin)
        return returnTopology

    @staticmethod
    def Place(topology, originA=None, originB=None):
        """
        Places the input topology at the specified location.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        originA : topologic.Vertex , optional
            The old location to use as the origin of the movement. If set to None, the centroid of the input topology is used. The default is None.
        originB : topologic.Vertex , optional
            The new location at which to place the topology. If set to None, the world origin (0,0,0) is used. The default is None.

        Returns
        -------
        topologic.Topology
            The placed topology.

        """
        from topologicpy.Vertex import Vertex
        if not isinstance(topology, topologic.Topology):
            return None
        if not isinstance(originA, topologic.Vertex):
            originA = Topology.Centroid(topology)
        if not isinstance(originA, topologic.Vertex):
            originA = Vertex.ByCoordinates(0,0,0)

        x = originB.X() - originA.X()
        y = originB.Y() - originA.Y()
        z = originB.Z() - originA.Z()
        newTopology = None
        try:
            newTopology = Topology.Translate(topology, x, y, z)
        except:
            print("Topology.Place - ERROR: (Topologic>TopologyUtility.Place) operation failed. Returning None.")
            newTopology = None
        return newTopology
    
    @staticmethod
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
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Cell import Cell

        if topology.Type() == topologic.Vertex.Type():
            return topology
        elif topology.Type() == topologic.Edge.Type():
            return Edge.VertexByParameter(topology, 0.5)
        elif topology.Type() == topologic.Wire.Type():
            e = Topology.Edges(topology)[0]
            return Edge.VertexByParameter(e, 0.5)
        elif topology.Type() == topologic.Face.Type():
            return Face.InternalVertex(topology, tolerance)
        elif topology.Type() == topologic.Shell.Type():
            f = Topology.Faces(topology)[0]
            return Face.InternalVertex(f, tolerance)
        elif topology.Type() == topologic.Cell.Type():
            return Cell.InternalVertex(topology, tolerance)
        elif topology.Type() == topologic.CellComplex.Type():
            c = Topology.Cells(topology)[0]
            return Cell.InternalVertex(c, tolerance)
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
                returnTopology = topologic.Cell.ByFaces(stl_final_faces, tolerance=tolerance)
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
    def RemoveCoplanarFaces(topology, planarize=False, angTolerance=0.1, tolerance=0.0001):
        """
        Removes coplanar faces in the input topology

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        planarize : bool , optional
            If set to True, the algorithm will attempt to planarize the final merged faces. Otherwise, it will triangulate any final nonplanar faces. The default is False.
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
        from topologicpy.Cell import Cell
        from topologicpy.Cluster import Cluster
        if not isinstance(topology, topologic.Topology):
            return None
        t = topology.Type()
        if (t == 1) or (t == 2) or (t == 4) or (t == 8) or (t == 128):
            return topology
        clusters = Topology.ClusterFaces(topology, tolerance=tolerance)
        faces = []
        for aCluster in clusters:
            aCluster = Topology.SelfMerge(aCluster)
            if isinstance(aCluster, topologic.Shell):
                shells = [aCluster]
            else:
                shells = Cluster.Shells(aCluster)
            tempFaces = Topology.SubTopologies(aCluster, subTopologyType="Face")
            if not shells or len(shells) < 1:
                if isinstance(aCluster, topologic.Shell):
                    shells = [aCluster]
                else:
                    temp_shell = Shell.ByFaces(tempFaces)
                    if isinstance(temp_shell, list):
                        shells = temp_shell
                    else:
                        shells = [temp_shell]
            if len(shells) > 0:
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
                eb = Topology.RemoveCollinearEdges(eb, angTolerance=angTolerance)
            except:
                pass
            finalIbList = []
            if ibList:
                for ib in ibList:
                    temp_ib = ib
                    try:
                        temp_ib = Topology.RemoveCollinearEdges(ib, angTolerance=angTolerance)
                    except:
                        pass
                    finalIbList.append(temp_ib)
            finalFaces.append(Face.ByWires(eb, finalIbList))
        faces = finalFaces
        if len(faces) == 1:
            return faces[0]
        if t == 16:
            returnTopology = topologic.Shell.ByFaces(faces, tolerance)
            if not returnTopology:
                returnTopology = topologic.Cluster.ByTopologies(faces, False)
        elif t == 32:
            returnTopology = Cell.ByFaces(faces, planarize=planarize, tolerance=tolerance)
            if not returnTopology:
                returnTopology = topologic.Shell.ByFaces(faces, tolerance)
            if not returnTopology:
                returnTopology = topologic.Cluster.ByTopologies(faces, False)
        elif t == 64:
            returnTopology = topologic.CellComplex.ByFaces(faces, tolerance, False)
            if not returnTopology:
                returnTopology = Cell.ByFaces(faces, planarize=planarize, tolerance=tolerance)
            if not returnTopology:
                returnTopology = topologic.Shell.ByFaces(faces, tolerance)
            if not returnTopology:
                returnTopology = topologic.Cluster.ByTopologies(faces, False)
        return returnTopology

    @staticmethod
    def RemoveEdges(topology, edges=[]):
        """
        Removes the input list of faces from the input topology

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        edges : list
            The input list of edges.

        Returns
        -------
        topologic.Topology
            The input topology with the input list of edges removed.

        """

        from topologicpy.Cluster import Cluster
        if not isinstance(topology, topologic.Topology):
            return None
        edges = [e for e in edges if isinstance(e, topologic.Edge)]
        if len(edges) < 1:
            return topology
        t_edges = Topology.Edges(topology)
        t_faces = Topology.Faces(topology)
        if len(t_edges) < 1:
            return topology
        if len(t_faces) > 0:
            remove_faces = []
            for t_e in t_edges:
                remove = False
                for i, e in enumerate(edges):
                    if Topology.IsSame(t_e, e):
                        remove = True
                        remove_faces += Topology.SuperTopologies(e, hostTopology=topology, topologyType="face")
                        edges = edges[:i]+ edges[i:]
                        break
            if len(remove_faces) > 0:
                return Topology.RemoveFaces(topology, remove_faces)
        else:
            remaining_edges = []
            for t_e in t_edges:
                remove = False
                for i, e in enumerate(edges):
                    if Topology.IsSame(t_e, e):
                        remove = True
                        edges = edges[:i]+ edges[i:]
                        break
                if not remove:
                    remaining_edges.append(t_e)
            if len(remaining_edges) < 1:
                return None
            elif len(remaining_edges) == 1:
                return remaining_edges[0]
            return Topology.SelfMerge(Cluster.ByTopologies(remaining_edges))

    @staticmethod
    def RemoveFaces(topology, faces=[]):
        """
        Removes the input list of faces from the input topology

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        faces : list
            The input list of faces.

        Returns
        -------
        topologic.Topology
            The input topology with the input list of faces removed.

        """

        from topologicpy.Cluster import Cluster
        if not isinstance(topology, topologic.Topology):
            return None
        faces = [f for f in faces if isinstance(f, topologic.Face)]
        if len(faces) < 1:
            return topology
        t_faces = Topology.Faces(topology)
        if len(t_faces) < 1:
            return topology
        remaining_faces = []
        for t_f in t_faces:
            remove = False
            for i, f in enumerate(faces):
                if Topology.IsSame(t_f, f):
                    remove = True
                    faces = faces[:i]+ faces[i:]
                    break
            if not remove:
                remaining_faces.append(t_f)
        if len(remaining_faces) < 1:
            return None
        elif len(remaining_faces) == 1:
            return remaining_faces[0]
        return Topology.SelfMerge(Cluster.ByTopologies(remaining_faces))
    
    @staticmethod
    def RemoveFacesBySelectors(topology, selectors=[], tolerance = 0.0001):
        """
        Removes faces that contain the input list of selectors (vertices) from the input topology

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        selectors : list
            The input list of selectors (vertices).
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Topology
            The input topology with the identified faces removed.

        """

        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster
        if not isinstance(topology, topologic.Topology):
            return None
        selectors = [v for v in selectors if isinstance(v, topologic.Vertex)]
        if len(selectors) < 1:
            return topology
        t_faces = Topology.Faces(topology)
        to_remove = []
        for t_f in t_faces:
            remove = False
            for i, v in enumerate(selectors):
                if Face.IsInside(face=t_f, vertex=v, tolerance=tolerance):
                    remove = True
                    selectors = selectors[:i]+ selectors[i:]
                    break
            if remove:
                to_remove.append(t_f)
        if len(to_remove) < 1:
            return topology
        return Topology.RemoveFaces(topology, faces = to_remove)

    @staticmethod
    def RemoveVertices(topology, vertices=[]):
        """
        Removes the input list of vertices from the input topology

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        vertices : list
            The input list of vertices.

        Returns
        -------
        topologic.Topology
            The input topology with the input list of vertices removed.

        """

        from topologicpy.Cluster import Cluster
        if not isinstance(topology, topologic.Topology):
            return None
        vertices = [v for v in vertices if isinstance(v, topologic.Vertex)]
        if len(vertices) < 1:
            return topology
        t_vertices = Topology.Vertices(topology)
        t_edges = Topology.Edges(topology)
        if len(t_vertices) < 1:
            return topology
        if len(t_edges) > 0:
            remove_edges = []
            for t_v in t_vertices:
                remove = False
                for i, v in enumerate(vertices):
                    if Topology.IsSame(t_v, v):
                        remove = True
                        remove_edges += Topology.SuperTopologies(v, hostTopology=topology, topologyType="edge")
                        vertices = vertices[:i]+ vertices[i:]
                        break
            if len(remove_edges) > 0:
                return Topology.RemoveEdges(topology, remove_edges)
        else:
            remaining_vertices = []
            for t_v in t_vertices:
                remove = False
                for i, e in enumerate(vertices):
                    if Topology.IsSame(t_v, v):
                        remove = True
                        vertices = vertices[:i]+ vertices[i:]
                        break
                if not remove:
                    remaining_vertices.append(t_v)
            if len(remaining_vertices) < 1:
                return None
            elif len(remaining_vertices) == 1:
                return remaining_vertices[0]
            return Topology.SelfMerge(Cluster.ByTopologies(remaining_vertices))
    
    @staticmethod
    def Cleanup(topology=None):
        """
        Cleans up all resources in which are managed by topologic library. Use this to manage your application's memory consumption.
        USE WITH CARE. This methods deletes dictionaries, contents, and contexts

        Parameters
        ----------
        topology : topologic.Topology , optional
            If specified the resources used by the input topology will be deleted. If not, ALL resources will be deleted.
        
        Returns
        -------
        topologic.Topology
            The input topology, but with its resources deleted or None.
        """
        if not topology == None:
            if not isinstance(topology, topologic.Topology):
                print("Topology.Cleanup - Error: The input topology parameter is not a valid topology. Returning None.")
                return None
        topologic.Topology.Cleanup(topology)
        return topology

    @staticmethod
    def ReplaceVertices(topology, verticesA=[], verticesB=[], tolerance=0.0001):
        """
        Replaces the vertices in the first input list with the vertices in the second input list and rebuilds the input topology. The two lists must be of the same length.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        verticesA : list
            The first input list of vertices.
        verticesB : list
            The second input list of vertices.
        tolerance : float, optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic.Topology
            The new topology.

        """
        if not len(verticesA) == len(verticesB):
            print("Error - Topology.ReplaceVertices: VerticesA and VerticesB must be the same length")
            return None
        from topologicpy.Vertex import Vertex
        geom = Topology.Geometry(topology)
        g_verts = geom['vertices']
        g_edges = geom['edges']
        g_faces = geom['faces']
        verts = [Topology.Vertices(Topology.ByGeometry(vertices=[g_v]))[0] for g_v in g_verts]
        new_verts = [v for v in verts]
        for i, v in enumerate(verticesA):
            n = Vertex.Index(v, verts, tolerance=tolerance)
            if not n == None:
                new_verts[n] = verticesB[i]
        new_g_verts = [[Vertex.X(v),Vertex.Y(v),Vertex.Z(v)] for v in new_verts]
        new_topology = Topology.ByGeometry(vertices=new_g_verts, edges=g_edges, faces=g_faces)
        return new_topology

    @staticmethod
    def Rotate(topology, origin=None, x=0, y=0, z=1, degree=0, angTolerance=0.001):
        """
        Rotates the input topology

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
        angTolerance : float , optional
            The angle tolerance in degrees under which no rotation is carried out. The default is 0.001 degrees.

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
        returnTopology = topology
        if abs(degree) >= angTolerance:
            try:
                returnTopology = topologic.TopologyUtility.Rotate(topology, origin, x, y, z, degree)
            except:
                print("Topology.Rotate - Error: (topologic.TopologyUtility.Rotate) operation failed. Returning None.")
                return None
        return returnTopology
    
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
            print("Topology.Scale - ERROR: (Topologic>TopologyUtility.Scale) operation failed. Returning None.")
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
        if subTopologyType.lower() == "vertex":
            t = 1
        elif subTopologyType.lower() == "edge":
            t = 2
        elif subTopologyType.lower() == "wire":
            t = 4
        elif subTopologyType.lower() == "face":
            t = 8
        elif subTopologyType.lower() == "shell":
            t = 16
        elif subTopologyType.lower() == "cell":
            t = 32
        elif subTopologyType.lower() == "cellcomplex":
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
        from topologicpy.Cluster import Cluster

        if topology.Type() != 128:
            topology = Cluster.ByTopologies([topology])
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
        return_topology = topology.SelfMerge()
        if isinstance(return_topology, topologic.CellComplex):
            cells = Topology.Cells(return_topology)
            if isinstance(cells, list):
                if len(cells) > 1:
                    topA = cells[0]
                    topB = Cluster.ByTopologies(cells[1:])
                    return_topology = Topology.Merge(topA, topB)
                else:
                    return_topology = cells[0]
        return return_topology

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
    def SharedTopologies(topologyA, topologyB):
        """
        Returns the shared topologies between the two input topologies

        Parameters
        ----------
        topologyA : topologic.Topology
            The first input topology.
        topologyB : topologic.Topology
            The second input topology.

        Returns
        -------
        dict
            A dictionary with the list of vertices, edges, wires, and faces. The keys are "vertices", "edges", "wires", and "faces".

        """

        if not isinstance(topologyA, topologic.Topology):
            print("Topology.SharedTopologies - Error: the input topologyA is not a valid topology. Returning None.")
            return None
        if not isinstance(topologyB, topologic.Topology):
            print("Topology.SharedTopologies - Error: the input topologyB is not a valid topology. Returning None.")
            return None
        vOutput = []
        eOutput = []
        wOutput = []
        fOutput = []
        _ = topologyA.SharedTopologies(topologyB, 1, vOutput)
        _ = topologyA.SharedTopologies(topologyB, 2, eOutput)
        _ = topologyA.SharedTopologies(topologyB, 4, wOutput)
        _ = topologyA.SharedTopologies(topologyB, 8, fOutput)
        return {"vertices":vOutput, "edges":eOutput, "wires":wOutput, "faces":fOutput}

    @staticmethod
    def SharedVertices(topologyA, topologyB):
        """
        Returns the shared vertices between the two input topologies

        Parameters
        ----------
        topologyA : topologic.Topology
            The first input topology.
        topologyB : topologic.Topology
            The second input topology.

        Returns
        -------
        list
            The list of shared vertices.

        """
        d = Topology.SharedTopologies(topologyA, topologyB)
        l = None
        if isinstance(d, dict):
            try:
                l = d['vertices']
            except:
                l = None
        return l
    
    
    @staticmethod
    def SharedEdges(topologyA, topologyB):
        """
        Returns the shared edges between the two input topologies

        Parameters
        ----------
        topologyA : topologic.Topology
            The first input topology.
        topologyB : topologic.Topology
            The second input topology.

        Returns
        -------
        list
            The list of shared edges.

        """
        d = Topology.SharedTopologies(topologyA, topologyB)
        l = None
        if isinstance(d, dict):
            try:
                l = d['edges']
            except:
                l = None
        return l
    
    @staticmethod
    def SharedWires(topologyA, topologyB):
        """
        Returns the shared wires between the two input topologies

        Parameters
        ----------
        topologyA : topologic.Topology
            The first input topology.
        topologyB : topologic.Topology
            The second input topology.

        Returns
        -------
        list
            The list of shared wires.

        """
        d = Topology.SharedTopologies(topologyA, topologyB)
        l = None
        if isinstance(d, dict):
            try:
                l = d['wires']
            except:
                l = None
        return l
    
    
    @staticmethod
    def SharedFaces(topologyA, topologyB):
        """
        Returns the shared faces between the two input topologies

        Parameters
        ----------
        topologyA : topologic.Topology
            The first input topology.
        topologyB : topologic.Topology
            The second input topology.

        Returns
        -------
        list
            The list of shared faces.

        """
        d = Topology.SharedTopologies(topologyA, topologyB)
        l = None
        if isinstance(d, dict):
            try:
                l = d['faces']
            except:
                l = None
        return l
    
    
    @staticmethod
    def Show(topology,
             showVertices=True, vertexSize=1.1, vertexColor="black", 
             vertexLabelKey=None, vertexGroupKey=None, vertexGroups=[], 
             vertexMinGroup=None, vertexMaxGroup=None, 
             showVertexLegend=False, vertexLegendLabel="Topology Vertices", vertexLegendRank=1, 
             vertexLegendGroup=1, 

             showEdges=True, edgeWidth=1, edgeColor="black", 
             edgeLabelKey=None, edgeGroupKey=None, edgeGroups=[], 
             edgeMinGroup=None, edgeMaxGroup=None, 
             showEdgeLegend=False, edgeLegendLabel="Topology Edges", edgeLegendRank=2, 
             edgeLegendGroup=2, 

             showFaces=True, faceOpacity=0.5, faceColor="white",
             faceLabelKey=None, faceGroupKey=None, faceGroups=[], 
             faceMinGroup=None, faceMaxGroup=None, 
             showFaceLegend=False, faceLegendLabel="Topology Faces", faceLegendRank=3,
             faceLegendGroup=3, 
             intensityKey=None,
             
             width=950, height=500,
             xAxis=False, yAxis=False, zAxis=False, axisSize=1, backgroundColor='rgba(0,0,0,0)',
             marginLeft=0, marginRight=0, marginTop=20, marginBottom=0, camera=[-1.25, -1.25, 1.25],
             target=[0, 0, 0], up=[0, 0, 1], renderer="notebook", showScale=False,
             
             cbValues=[], cbTicks=5, cbX=-0.15, cbWidth=15, cbOutlineWidth=0, cbTitle="",
             cbSubTitle="", cbUnits="", colorScale="Viridis", mantissa=4, tolerance=0.0001):
        """
            Shows the input topology on screen.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology. This must contain faces and or edges.

        showVertices : bool , optional
            If set to True the vertices will be drawn. Otherwise, they will not be drawn. The default is True.
        vertexSize : float , optional
            The desired size of the vertices. The default is 1.1.
        vertexColor : str , optional
            The desired color of the output vertices. This can be any plotly color string and may be specified as:
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is "black".
        vertexLabelKey : str , optional
            The dictionary key to use to display the vertex label. The default is None.
        vertexGroupKey : str , optional
            The dictionary key to use to display the vertex group. The default is None.
        vertexGroups : list , optional
            The list of vertex groups against which to index the color of the vertex. The default is [].
        vertexMinGroup : int or float , optional
            For numeric vertexGroups, vertexMinGroup is the desired minimum value for the scaling of colors. This should match the type of value associated with the vertexGroupKey. If set to None, it is set to the minimum value in vertexGroups. The default is None.
        edgeMaxGroup : int or float , optional
            For numeric vertexGroups, vertexMaxGroup is the desired maximum value for the scaling of colors. This should match the type of value associated with the vertexGroupKey. If set to None, it is set to the maximum value in vertexGroups. The default is None.
        showVertexLegend : bool, optional
            If set to True, the legend for the vertices of this topology is shown. Otherwise, it isn't. The default is False.
        vertexLegendLabel : str , optional
            The legend label string used to identify vertices. The default is "Topology Vertices".
        vertexLegendRank : int , optional
            The legend rank order of the vertices of this topology. The default is 1.
        vertexLegendGroup : int , optional
            The number of the vertex legend group to which the vertices of this topology belong. The default is 1.
        
        showEdges : bool , optional
            If set to True the edges will be drawn. Otherwise, they will not be drawn. The default is True.
        edgeWidth : float , optional
            The desired thickness of the output edges. The default is 1.
        edgeColor : str , optional
            The desired color of the output edges. This can be any plotly color string and may be specified as:
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is "black".
        edgeLabelKey : str , optional
            The dictionary key to use to display the edge label. The default is None.
        edgeGroupKey : str , optional
            The dictionary key to use to display the edge group. The default is None.
        edgeGroups : list , optional
            The list of edge groups against which to index the color of the edge. The default is [].
        edgeMinGroup : int or float , optional
            For numeric edgeGroups, edgeMinGroup is the desired minimum value for the scaling of colors. This should match the type of value associated with the edgeGroupKey. If set to None, it is set to the minimum value in edgeGroups. The default is None.
        edgeMaxGroup : int or float , optional
            For numeric edgeGroups, edgeMaxGroup is the desired maximum value for the scaling of colors. This should match the type of value associated with the edgeGroupKey. If set to None, it is set to the maximum value in edgeGroups. The default is None.
        showEdgeLegend : bool, optional
            If set to True, the legend for the edges of this topology is shown. Otherwise, it isn't. The default is False.
        edgeLegendLabel : str , optional
            The legend label string used to identify edges. The default is "Topology Edges".
        edgeLegendRank : int , optional
            The legend rank order of the edges of this topology. The default is 2.
        edgeLegendGroup : int , optional
            The number of the edge legend group to which the edges of this topology belong. The default is 2.
        
        showFaces : bool , optional
            If set to True the faces will be drawn. Otherwise, they will not be drawn. The default is True.
        faceOpacity : float , optional
            The desired opacity of the output faces (0=transparent, 1=opaque). The default is 0.5.
        faceColor : str , optional
            The desired color of the output faces. This can be any plotly color string and may be specified as:
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is "white".
        faceLabelKey : str , optional
            The dictionary key to use to display the face label. The default is None.
        faceGroupKey : str , optional
            The dictionary key to use to display the face group. The default is None.
        faceGroups : list , optional
            The list of face groups against which to index the color of the face. This can bhave numeric or string values. This should match the type of value associated with the faceGroupKey. The default is [].
        faceMinGroup : int or float , optional
            For numeric faceGroups, minGroup is the desired minimum value for the scaling of colors. This should match the type of value associated with the faceGroupKey. If set to None, it is set to the minimum value in faceGroups. The default is None.
        faceMaxGroup : int or float , optional
            For numeric faceGroups, maxGroup is the desired maximum value for the scaling of colors. This should match the type of value associated with the faceGroupKey. If set to None, it is set to the maximum value in faceGroups. The default is None.
        showFaceLegend : bool, optional
            If set to True, the legend for the faces of this topology is shown. Otherwise, it isn't. The default is False.
        faceLegendLabel : str , optional
            The legend label string used to idenitfy edges. The default is "Topology Faces".
        faceLegendRank : int , optional
            The legend rank order of the faces of this topology. The default is 3.
        faceLegendGroup : int , optional
            The number of the face legend group to which the faces of this topology belong. The default is 3.
        width : int , optional
            The width in pixels of the figure. The default value is 950.
        height : int , optional
            The height in pixels of the figure. The default value is 950.
        xAxis : bool , optional
            If set to True the x axis is drawn. Otherwise it is not drawn. The default is False.
        yAxis : bool , optional
            If set to True the y axis is drawn. Otherwise it is not drawn. The default is False.
        zAxis : bool , optional
            If set to True the z axis is drawn. Otherwise it is not drawn. The default is False.
        backgroundColor : str , optional
            The desired color of the background. This can be any plotly color string and may be specified as:
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is "rgba(0,0,0,0)".
        marginLeft : int , optional
            The size in pixels of the left margin. The default value is 0.
        marginRight : int , optional
            The size in pixels of the right margin. The default value is 0.
        marginTop : int , optional
            The size in pixels of the top margin. The default value is 20.
        marginBottom : int , optional
            The size in pixels of the bottom margin. The default value is 0.
        camera : list , optional
            The desired location of the camera). The default is [0,0,0].
        center : list , optional
            The desired center (camera target). The default is [0,0,0].
        up : list , optional
            The desired up vector. The default is [0,0,1].
        renderer : str , optional
            The desired renderer. See Plotly.Renderers(). The default is "notebook".
        intensityKey : str , optional
            If not None, the dictionary of each vertex is searched for the value associated with the intensity key. This value is then used to color-code the vertex based on the colorScale. The default is None.
        showScale : bool , optional
            If set to True, the colorbar is shown. The default is False.
        cbValues : list , optional
            The input list of values to use for the colorbar. The default is [].
        cbTicks : int , optional
            The number of ticks to use on the colorbar. The default is 5.
        cbX : float , optional
            The x location of the colorbar. The default is -0.15.
        cbWidth : int , optional
            The width in pixels of the colorbar. The default is 15
        cbOutlineWidth : int , optional
            The width in pixels of the outline of the colorbar. The default is 0.
        cbTitle : str , optional
            The title of the colorbar. The default is "".
        cbSubTitle : str , optional
            The subtitle of the colorbar. The default is "".
        cbUnits: str , optional
            The units used in the colorbar. The default is ""
        colorScale : str , optional
            The desired type of plotly color scales to use (e.g. "viridis", "plasma"). The default is "viridis". For a full list of names, see https://plotly.com/python/builtin-colorscales/.
        mantissa : int , optional
            The desired length of the mantissa for the values listed on the colorbar. The default is 4.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        None

        """
        from topologicpy.Plotly import Plotly
        if not isinstance(topology, topologic.Topology):
            print("Topology.Show - Error: the input topology is not a valid topology. Returning None.")
            return None
        data = Plotly.DataByTopology(topology=topology,
                       showVertices=showVertices, vertexSize=vertexSize, vertexColor=vertexColor, 
                       vertexLabelKey=vertexLabelKey, vertexGroupKey=vertexGroupKey, vertexGroups=vertexGroups, 
                       vertexMinGroup=vertexMinGroup, vertexMaxGroup=vertexMaxGroup, 
                       showVertexLegend=showVertexLegend, vertexLegendLabel=vertexLegendLabel, vertexLegendRank=vertexLegendRank,
                       vertexLegendGroup=vertexLegendGroup,
                       showEdges=showEdges, edgeWidth=edgeWidth, edgeColor=edgeColor, 
                       edgeLabelKey=edgeLabelKey, edgeGroupKey=edgeGroupKey, edgeGroups=edgeGroups, 
                       edgeMinGroup=edgeMinGroup, edgeMaxGroup=edgeMaxGroup, 
                       showEdgeLegend=showEdgeLegend, edgeLegendLabel=edgeLegendLabel, edgeLegendRank=edgeLegendRank, 
                       edgeLegendGroup=edgeLegendGroup,
                       showFaces=showFaces, faceOpacity=faceOpacity, faceColor=faceColor,
                       faceLabelKey=faceLabelKey, faceGroupKey=faceGroupKey, faceGroups=faceGroups, 
                       faceMinGroup=faceMinGroup, faceMaxGroup=faceMaxGroup, 
                       showFaceLegend=showFaceLegend, faceLegendLabel=faceLegendLabel, faceLegendRank=faceLegendRank,
                       faceLegendGroup=faceLegendGroup, 
                       intensityKey=intensityKey, colorScale=colorScale, tolerance=tolerance)
        figure = Plotly.FigureByData(data=data, width=width, height=height, xAxis=xAxis, yAxis=yAxis, zAxis=zAxis, axisSize=axisSize, backgroundColor=backgroundColor, marginLeft=marginLeft, marginRight=marginRight, marginTop=marginTop, marginBottom=marginBottom)
        if showScale:
            figure = Plotly.AddColorBar(figure, values=cbValues, nTicks=cbTicks, xPosition=cbX, width=cbWidth, outlineWidth=cbOutlineWidth, title=cbTitle, subTitle=cbSubTitle, units=cbUnits, colorScale=colorScale, mantissa=mantissa)
        Plotly.Show(figure=figure, renderer=renderer, camera=camera, target=target, up=up)

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
    def Spin(topology, origin=None, triangulate=True, direction=[0,0,1], degree=360, sides=16,
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
        direction : list , optional
            The vector representing the direction of the spin axis. The default is [0,0,1].
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
            print("Topology.Spin - Error: the input topology is not a valid topology. Returning None.")
            return None
        if not isinstance(origin, topologic.Vertex):
            print("Topology.Spin - Error: the input origin is not a valid vertex. Returning None.")
            return None
        topologies = []
        unit_degree = degree / float(sides)
        for i in range(sides+1):
            tempTopology = Topology.Rotate(topology, origin, direction[0], direction[1], direction[2], unit_degree*i)
            if tempTopology:
                topologies.append(tempTopology)
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
        if not returnTopology:
            return topologic.Cluster.ByTopologies(topologies)
        if returnTopology.Type() == topologic.Shell.Type():
            try:
                new_t = topologic.Cell.ByShell(returnTopology)
                if new_t:
                    returnTopology = new_t
            except:
                pass
        return returnTopology
    
    @staticmethod
    def Taper(topology, origin=None, ratioRange=[0,1], triangulate=False, tolerance=0.0001):
        """
        Tapers the input topology. This method tapers the input geometry along its Z-axis based on the ratio range input.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        origin : topologic.Vertex , optional
            The desired origin for tapering. If not specified, the centroid of the input topology is used. The tapering will use the X, Y coordinates of the specified origin, but will use the Z of the point being tapered. The default is None.
        ratioRange : list , optional
            The desired ratio range. This will specify a linear range from bottom to top for tapering the vertices. 0 means no tapering, and 1 means maximum (inward) tapering. Negative numbers mean that tapering will be outwards.
        triangulate : bool , optional
            If set to true, the input topology is triangulated before tapering. Otherwise, it will not be traingulated. The default is False.
        tolerance : float , optional
            The desired tolerance. Vertices will not be moved if the calculated distance is at or less than this tolerance.

        Returns
        -------
        topologic.Topology
            The tapered topology.

        """
        from topologicpy.Vertex import Vertex
        ratioRange = [min(1,ratioRange[0]), min(1,ratioRange[1])]
        if ratioRange == [0,0]:
            return topology
        if ratioRange == [1,1]:
            print("Topology.Taper - Error: Degenerate result. Returning original topology.")
            return topology
        if triangulate == True:
            topology = Topology.Triangulate(topology)
        if origin == None:
            origin = Topology.Centroid(topology)
        vertices = Topology.Vertices(topology)
        zList = [Vertex.Z(v) for v in vertices]
        minZ = min(zList)
        maxZ = max(zList)
        new_vertices = []
        for v in vertices:
            ht = (Vertex.Z(v)-minZ)/(maxZ - minZ)
            rt = ratioRange[0] + ht*(ratioRange[1] - ratioRange[0])
            new_origin = Vertex.ByCoordinates(Vertex.X(origin), Vertex.Y(origin), Vertex.Z(v))
            new_dist = Vertex.Distance(new_origin, v)*rt
            c_a = Vertex.Coordinates(new_origin)
            c_b = Vertex.Coordinates(v)
            new_dir = [(c_a[0]-c_b[0]), (c_a[1]-c_b[1]), 0]
            if abs(new_dist) > tolerance:
                new_v = Topology.TranslateByDirectionDistance(v, direction=new_dir, distance=new_dist)
            else:
                new_v = v
            new_vertices.append(new_v)
        return_topology = Topology.ReplaceVertices(topology, vertices, new_vertices)
        return return_topology
    
    @staticmethod
    def Twist(topology, origin=None, degreeRange=[45,90], triangulate=False):
        """
        Twists the input topology. This method twists the input geometry along its Z-axis based on the degree range input.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.
        origin : topologic.Vertex , optional
            The desired origin for tapering. If not specified, the centroid of the input topology is used. The tapering will use the X, Y coordinates of the specified origin, but will use the Z of the point being tapered. The default is None.
        degreeRange : list , optional
            The desired angle range in degrees. This will specify a linear range from bottom to top for twisting the vertices. positive numbers mean a clockwise rotation.
        triangulate : bool , optional
            If set to true, the input topology is triangulated before tapering. Otherwise, it will not be traingulated. The default is False.
        
        Returns
        -------
        topologic.Topology
            The twisted topology.

        """
        from topologicpy.Vertex import Vertex

        if degreeRange == [0,0]:
            return topology
        if triangulate == True:
            topology = Topology.Triangulate(topology)
        if origin == None:
            origin = Topology.Centroid(topology)
            
        vertices = Topology.Vertices(topology)
        zList = [Vertex.Z(v) for v in vertices]
        minZ = min(zList)
        maxZ = max(zList)
        h = maxZ - minZ
        new_vertices = []
        for v in vertices:
            ht = (Vertex.Z(v)-minZ)/(maxZ - minZ)
            new_rot = degreeRange[0] + ht*(degreeRange[1] - degreeRange[0])
            orig = Vertex.ByCoordinates(Vertex.X(origin), Vertex.Y(origin), Vertex.Z(v))
            new_vertices.append(Topology.Rotate(v, origin=orig, degree=new_rot))
        return_topology = Topology.ReplaceVertices(topology, vertices, new_vertices)
        return_topology = Topology.Fix(return_topology, topologyType=Topology.TypeAsString(topology))
        return return_topology
    
    @staticmethod
    def Vertices(topology):
        """
        Returns the vertices of the input topology.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.

        Returns
        -------
        list
            The list of vertices.

        """
        return Topology.SubTopologies(topology=topology, subTopologyType="vertex")
    
    @staticmethod
    def Edges(topology):
        """
        Returns the edges of the input topology.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.

        Returns
        -------
        list
            The list of edges.

        """
        if isinstance(topology, topologic.Edge) or isinstance(topology, topologic.Vertex):
            return []
        return Topology.SubTopologies(topology=topology, subTopologyType="edge")
    
    @staticmethod
    def Wires(topology):
        """
        Returns the wires of the input topology.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.

        Returns
        -------
        list
            The list of wires.

        """
        if isinstance(topology, topologic.Wire) or isinstance(topology, topologic.Edge) or isinstance(topology, topologic.Vertex):
            return []
        return Topology.SubTopologies(topology=topology, subTopologyType="wire")
    
    @staticmethod
    def Faces(topology):
        """
        Returns the faces of the input topology.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.

        Returns
        -------
        list
            The list of faces.

        """
        if isinstance(topology, topologic.Face) or isinstance(topology, topologic.Wire) or isinstance(topology, topologic.Edge) or isinstance(topology, topologic.Vertex):
            return []
        return Topology.SubTopologies(topology=topology, subTopologyType="face")
    
    @staticmethod
    def Shells(topology):
        """
        Returns the shells of the input topology.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.

        Returns
        -------
        list
            The list of shells.

        """
        if isinstance(topology, topologic.Shell) or isinstance(topology, topologic.Face) or isinstance(topology, topologic.Wire) or isinstance(topology, topologic.Edge) or isinstance(topology, topologic.Vertex):
            return []
        return Topology.SubTopologies(topology=topology, subTopologyType="shell")
    
    @staticmethod
    def Cells(topology):
        """
        Returns the cells of the input topology.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.

        Returns
        -------
        list
            The list of cells.

        """
        if isinstance(topology, topologic.Cell) or isinstance(topology, topologic.Shell) or isinstance(topology, topologic.Face) or isinstance(topology, topologic.Wire) or isinstance(topology, topologic.Edge) or isinstance(topology, topologic.Vertex):
            return []
        return Topology.SubTopologies(topology=topology, subTopologyType="cell")
    
    @staticmethod
    def CellComplexes(topology):
        """
        Returns the cellcomplexes of the input topology.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.

        Returns
        -------
        list
            The list of cellcomplexes.

        """
        if isinstance(topology, topologic.CellComplex) or isinstance(topology, topologic.Cell) or isinstance(topology, topologic.Shell) or isinstance(topology, topologic.Face) or isinstance(topology, topologic.Wire) or isinstance(topology, topologic.Edge) or isinstance(topology, topologic.Vertex):
            return []
        return Topology.SubTopologies(topology=topology, subTopologyType="cellcomplex")
    
    @staticmethod
    def Clusters(topology):
        """
        Returns the clusters of the input topology.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology.

        Returns
        -------
        list
            The list of clusters.

        """
        if not isinstance(topology, topologic.Cluster):
            return []
        return Topology.SubTopologies(topology=topology, subTopologyType="cluster")
    
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
            print("Topology.SubTopologies - Error: the input topology is not a valid topology. Returning None.")
            return None
        if Topology.TypeAsString(topology).lower() == subTopologyType.lower():
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
        if not subTopologies:
            return [] # Make sure to return an empty list instead of None
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
            print("Topology.SuperTopologies - Error: the input topology is not a valid topology. Returning None.")
            return None
        if not isinstance(hostTopology, topologic.Topology):
            print("Topology.SuperTopologies - Error: the input hostTopology is not a valid topology. Returning None.")
            return None

        superTopologies = []

        if not topologyType:
            typeID = 2*Topology.TypeID(topology)
        else:
            typeID = Topology.TypeID(topologyType)
        if topology.Type() >= typeID:
            print("Topology.SuperTopologies - Error: The input topologyType is not a valid type for a super topology of the input topology. Returning None.")
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
            print("Topology.SuperTopologies - Error: The input topologyType is not a valid type for a super topology of the input topology. Returning None.")
            return None
        if not superTopologies:
            return [] # Make sure you return an empty list instead of None
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
            print("Topology.TransferDictionaries - Error: The input sources is not a valid list. Returning None.")
            return None
        if not isinstance(sinks, list):
            print("Topology.TransferDictionaries - Error: The input sinks is not a valid list. Returning None.")
            return None
        sources = [x for x in sources if isinstance(x, topologic.Topology)]
        sinks = [x for x in sinks if isinstance(x, topologic.Topology)]
        if len(sources) < 1:
            print("Topology.TransferDictionaries - Error: The input sources does not contain any valid topologies. Returning None.")
            return None
        if len(sinks) < 1:
            print("Topology.TransferDictionaries - Error: The input sinks does not contain any valid topologies. Returning None.")
            return None
        for sink in sinks:
            sinkKeys = []
            sinkValues = []
            iv = Topology.RelevantSelector(sink, tolerance)
            j = 1
            for source in sources:
                flag = False
                if isinstance(source, topologic.Vertex):
                    flag = Topology.IsInside(sink, source, tolerance)
                else:
                    flag = Topology.IsInside(source, iv, tolerance)
                if flag:
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
        from topologicpy.Vertex import Vertex
        from topologicpy.Dictionary import Dictionary
        if not isinstance(topology, topologic.Topology):
            print("Topology.TransferDictionariesBySelectors - Error: The input topology is not a valid topology. Returning None.")
            return None
        if not isinstance(selectors, list):
            print("Topology.TransferDictionariesBySelectors - Error: The input selectors is not a valid list. Returning None.")
            return None
        selectors_tmp = [x for x in selectors if isinstance(x, topologic.Vertex)]
        if len(selectors_tmp) < 1:
            print("Topology.TransferDictionariesBySelectors - Error: The input selectors does not contain any valid topologies. Returning None.")
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
            print("Topology.Translate - Error: The input topology is not a valid topology. Returning None.")
            return None
        try:
            return topologic.TopologyUtility.Translate(topology, x, y, z)
        except:
            return topology
    
    @staticmethod
    def TranslateByDirectionDistance(topology, direction=[0,0,0], distance=0):
        """
        Translates (moves) the input topology along the input direction by the specified distance.

        Parameters
        ----------
        topology : topologic.topology
            The input topology.
        direction : list , optional
            The direction vector in which the topology should be moved. The default is [0,0,0]
        distance : float , optional
            The distance by which the toplogy should be moved. The default is 0.

        Returns
        -------
        topologic.Topology
            The translated topology.

        """
        from topologicpy.Vector import Vector
        if not isinstance(topology, topologic.Topology):
            print("Topology.TranslateByDirectionDistance - Error: The input topology is not a valid topology. Returning None.")
            return None
        v = Vector.SetMagnitude(direction, distance)
        return Topology.Translate(topology, v[0], v[1], v[2])

    
    @staticmethod
    def Triangulate(topology, transferDictionaries = False, tolerance=0.0001):
        """
        Triangulates the input topology.

        Parameters
        ----------
        topology : topologic.Topology
            The input topologgy.
        transferDictionaries : bool , optional
            If set to True, the dictionaries of the faces in the input topology will be transferred to the created triangular faces. The default is False.
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
        from topologicpy.Cluster import Cluster
        from topologicpy.Dictionary import Dictionary

        if not isinstance(topology, topologic.Topology):
            print("Topology.Triangulate - Error: The input is not a valid topology. Returning None.")
            return None
        t = topology.Type()
        if (t == 1) or (t == 2) or (t == 4):
            return topology
        elif t == 128:
            temp_topologies = []
            cellComplexes = Topology.SubTopologies(topology, subTopologyType="cellcomplex") or []
            for cc in cellComplexes:
                temp_topologies.append(Topology.Triangulate(cc, transferDictionaries=transferDictionaries, tolerance=tolerance))
            cells = Cluster.FreeCells(topology) or []
            for c in cells:
                temp_topologies.append(Topology.Triangulate(c, transferDictionaries=transferDictionaries, tolerance=tolerance))
            shells = Cluster.FreeShells(topology) or []
            for s in shells:
                temp_topologies.append(Topology.Triangulate(s, transferDictionaries=transferDictionaries, tolerance=tolerance))
            if len(temp_topologies) > 0:
                return Cluster.ByTopologies(temp_topologies)
            else:
                return topology
        topologyFaces = []
        _ = topology.Faces(None, topologyFaces)
        faceTriangles = []
        selectors = []
        for aFace in topologyFaces:
            if len(Topology.Vertices(aFace)) > 3:
                triFaces = Face.Triangulate(aFace)
            else:
                triFaces = [aFace]
            for triFace in triFaces:
                if transferDictionaries:
                    selectors.append(Topology.SetDictionary(Face.Centroid(triFace), Topology.Dictionary(aFace)))
                faceTriangles.append(triFace)
        if t == 8 or t == 16: # Face or Shell
            shell = Shell.ByFaces(faceTriangles, tolerance)
            if transferDictionaries:
                shell = Topology.TransferDictionariesBySelectors(shell, selectors, tranFaces=True)
            return shell
        elif t == 32: # Cell
            cell = Cell.ByFaces(faceTriangles, tolerance=tolerance)
            if transferDictionaries:
                cell = Topology.TransferDictionariesBySelectors(cell, selectors, tranFaces=True)
            return cell
        elif t == 64: #CellComplex
            cellComplex = CellComplex.ByFaces(faceTriangles, tolerance)
            if transferDictionaries:
                cellComplex = Topology.TransferDictionariesBySelectors(cellComplex, selectors, tranFaces=True)
            return cellComplex

    
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
            print("Topology.Type - Error: The input topology is not a valid topology. Returning None.")
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
            print("Topology.TypeAsString - Error: The input topology is not a valid topology. Returning None.")
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
            print("Topology.TypeID - Error: The input topologyType is not a valid string. Returning None.")
            return None
        topologyType = topologyType.lower()
        if not topologyType in ["vertex", "edge", "wire", "face", "shell", "cell", "cellcomplex", "cluster"]:
            print("Topology.TypeID - Error: The input topologyType is not a recognized string. Returning None.")
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