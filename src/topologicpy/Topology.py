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
import warnings
import uuid
import json
import os

import math
from collections import namedtuple
from multiprocessing import Process, Queue

# This is for View3D as not to open new browser windows
opened_urls = set()

try:
    import numpy as np
    from numpy import arctan, pi, signbit
    from numpy.linalg import norm
except:
    print("Topology - Installing required numpy library.")
    try:
        os.system("pip install numpy")
    except:
        os.system("pip install numpy --user")
    try:
        import numpy as np
        from numpy import arctan, pi, signbit
        from numpy.linalg import norm
        print("Topology - numpy library installed successfully.")
    except:
        warnings.warn("Topology - Error: Could not import numpy.")

try:
    from scipy.spatial import ConvexHull
except:
    print("Topology - Installing required scipy library.")
    try:
        os.system("pip install scipy")
    except:
        os.system("pip install scipy --user")
    try:
        from scipy.spatial import ConvexHull
        print("Topology - scipy library installed successfully.")
    except:
        warnings.warn("Topology - Error: Could not import scipy.")

try:
    from tqdm.auto import tqdm
except:
    print("Topology - Installing required tqdm library.")
    try:
        os.system("pip install tqdm")
    except:
        os.system("pip install tqdm --user")
    try:
        from tqdm.auto import tqdm
        print("Topology - tqdm library installed correctly.")
    except:
        warnings.warn("Topology - Error: Could not import tqdm.")

QueueItem = namedtuple('QueueItem', ['ID', 'sinkKeys', 'sinkValues'])
SinkItem = namedtuple('SinkItem', ['ID', 'sink_str'])

class WorkerProcessPool(object):
    """
    Create and manage a list of Worker processes. Each worker process
    transfers the dictionaries from a subset of sources to the list of sinks.
    """
    def __init__(self, num_workers, message_queue, sources, sinks, so_dicts, tolerance=0.0001):
        self.num_workers = num_workers
        self.message_queue = message_queue
        self.sources = sources
        self.sinks = sinks
        self.so_dicts = so_dicts
        self.tolerance = tolerance
        self.process_list = []

    def startProcesses(self):
        num_item_per_worker = len(self.sources) // self.num_workers
        for i in range(self.num_workers):
            if i == self.num_workers - 1:
                begin = i * num_item_per_worker
                sub_sources = self.sources[begin:]
                sub_dict = self.so_dicts[begin:]
            else:
                begin = i * num_item_per_worker
                end = begin + num_item_per_worker
                sub_sources = self.sources[begin : end]
                sub_dict = self.so_dicts[begin : end]
            wp = WorkerProcess(self.message_queue, sub_sources, self.sinks, sub_dict, self.tolerance)
            wp.start()
            self.process_list.append(wp)

    def stopProcesses(self):
        for p in self.process_list:
            p.join()
        self.process_list = []

    def join(self):
        for p in self.process_list:
            p.join()

class WorkerProcess(Process):
    """
    Transfers the dictionaries from a subset of sources to the list of sinks.
    """
    def __init__(self, message_queue, sources, sinks, so_dicts, tolerance=0.0001):
        Process.__init__(self, target=self.run)
        self.message_queue = message_queue
        self.sources = sources
        self.sinks = sinks
        self.so_dicts = so_dicts
        self.tolerance = tolerance

    def run(self):
        from topologicpy.Vertex import Vertex
        from topologicpy.Dictionary import Dictionary
        for sink_item in self.sinks:
            sink = Topology.ByBREPString(sink_item.sink_str)
            sinkKeys = []
            sinkValues = []
            iv = Topology.InternalVertex(sink, tolerance=self.tolerance)
            for j, source_str in enumerate(self.sources):
                source = Topology.ByBREPString(source_str)
                flag = False
                if Topology.IsInstance(source, "Vertex"):
                    flag = Vertex.IsInternal(source, sink, self.tolerance)
                else:
                    flag = Vertex.IsInternal(iv, source, self.tolerance)
                if flag:
                    d = Dictionary.ByPythonDictionary(self.so_dicts[j])
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
                    break
            if len(sinkKeys) > 0 and len(sinkValues) > 0:
                self.message_queue.put(QueueItem(sink_item.ID, sinkKeys, sinkValues))

class MergingProcess(Process):
    """
    Receive message from other processes and merging the result
    """
    def __init__(self, message_queue, sources, sinks, so_dicts):
        Process.__init__(self, target=self.wait_message)
        self.message_queue = message_queue
        self.sources = sources
        self.sinks = sinks
        self.so_dicts = so_dicts
        self.sinkMap = self._init_sink_map()

    def _init_sink_map(self):
        sinkMap = {}
        for sink in self.sinks:
            sinkMap[sink.ID] = QueueItem(sink.ID, [], [])
        return sinkMap

    def wait_message(self):
        while True:
            try:
                item = self.message_queue.get()
                if item is None:
                    self.message_queue.put(self.sinkMap)
                    break
                mapItem = self.sinkMap[item.ID]
                mapItem.sinkKeys.extend(item.sinkKeys)
                mapItem.sinkValues.extend(item.sinkValues)
            except Exception as e:
                print(str(e))

class Topology():
    @staticmethod
    def AddApertures(topology, apertures, exclusive=False, subTopologyType=None, tolerance=0.001):
        """
        Adds the input list of apertures to the input topology or to its subtopologies based on the input subTopologyType.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        apertures : list
            The input list of apertures.
        exclusive : bool , optional
            If set to True, one (sub)topology will accept only one aperture. Otherwise, one (sub)topology can accept multiple apertures. The default is False.
        subTopologyType : string , optional
            The subtopology type to which to add the apertures. This can be "cell", "face", "edge", or "vertex". It is case insensitive. If set to None, the apertures will be added to the input topology. The default is None.
        tolerance : float , optional
            The desired tolerance. The default is 0.001. This is larger than the usual 0.0001 as it seems to work better.

        Returns
        -------
        topologic_core.Topology
            The input topology with the apertures added to it.

        """
        from topologicpy.Dictionary import Dictionary
        
        if not Topology.IsInstance(topology, "Topology"):
            print("Topology.AddApertures - Error: The input topology parameter is not a valid topology. Returning None.")
            return None
        if not apertures:
            return topology
        if not isinstance(apertures, list):
            print("Topology.AddApertures - Error: the input apertures parameter is not a list. Returning None.")
            return None
        apertures = [x for x in apertures if Topology.IsInstance(x, "Topology")]
        if len(apertures) < 1:
            return topology
        if not subTopologyType:
            subTopologyType = "self"
        if not subTopologyType.lower() in ["self", "cell", "face", "edge", "vertex"]:
            print("Topology.AddApertures - Error: the input subtopology type parameter is not a recognized type. Returning None.")
            return None
        
        for aperture in apertures:
            d = Topology.Dictionary(aperture)
            d = Dictionary.SetValueAtKey(d, "type", "Aperture")
            aperture = Topology.SetDictionary(aperture, d)
        
        topology = Topology.AddContent(topology, apertures, subTopologyType=subTopologyType, tolerance=tolerance)
        return topology



    @staticmethod
    def AddApertures_old(topology, apertures, exclusive=False, subTopologyType=None, tolerance=0.001):
        """
        Adds the input list of apertures to the input topology or to its subtopologies based on the input subTopologyType.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        apertures : list
            The input list of apertures.
        exclusive : bool , optional
            If set to True, one (sub)topology will accept only one aperture. Otherwise, one (sub)topology can accept multiple apertures. The default is False.
        subTopologyType : string , optional
            The subtopology type to which to add the apertures. This can be "cell", "face", "edge", or "vertex". It is case insensitive. If set to None, the apertures will be added to the input topology. The default is None.
        tolerance : float , optional
            The desired tolerance. The default is 0.001. This is larger than the usual 0.0001 as it seems to work better.

        Returns
        -------
        topologic_core.Topology
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
                apCenter = Topology.InternalVertex(aperture, tolerance=tolerance)
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

        if not Topology.IsInstance(topology, "Topology"):
            print("Topology.AddApertures - Error: The input topology parameter is not a valid topology. Returning None.")
            return None
        if not apertures:
            return topology
        if not isinstance(apertures, list):
            print("Topology.AddApertures - Error: the input apertures parameter is not a list. Returning None.")
            return None
        apertures = [x for x in apertures if Topology.IsInstance(x , "Topology")]
        if len(apertures) < 1:
            return topology
        if not subTopologyType:
            subTopologyType = "self"
        if not subTopologyType.lower() in ["self", "cell", "face", "edge", "vertex"]:
            print("Topology.AddApertures - Error: the input subtopology type parameter is not a recognized type. Returning None.")
            return None
        if subTopologyType.lower() == "self":
            subTopologies = [topology]
        else:
            subTopologies = Topology.SubTopologies(topology, subTopologyType)
        processApertures(subTopologies, apertures, exclusive, tolerance=tolerance)
        return topology
    
    @staticmethod
    def AddContent(topology, contents, subTopologyType=None, tolerance=0.0001):
        """
        Adds the input list of contents to the input topology or to its subtpologies based on the input subTopologyType.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        contents : list or topologic_core.Topology
            The input list of contents (of type topologic_core.Topology). A single topology is also accepted as input.
        subTopologyType : string , optional
            The subtopology type to which to add the contents. This can be "cellcomplex", "cell", "shell", "face", "wire", "edge", or "vertex". It is case insensitive. If set to None, the contents will be added to the input topology. The default is None.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Topology
            The input topology with the contents added to it.

        """
        if not Topology.IsInstance(topology, "Topology"):
            print("Topology.AddContent - Error: the input topology parameter is not a valid topology. Returning None.")
            return None
        if not contents:
            return topology
        if not isinstance(contents, list):
            contents = [contents]
        if not isinstance(contents, list):
            print("Topology.AddContent - Error: the input contents parameter is not a list. Returning None.")
            return None
        contents = [x for x in contents if Topology.IsInstance(x, "Topology")]
        if len(contents) < 1:
            return topology
        if not subTopologyType:
            subTopologyType = "self"
        if not subTopologyType.lower() in ["self", "cellcomplex", "cell", "shell", "face", "wire", "edge", "vertex"]:
            print("Topology.AddContent - Error: the input subtopology type parameter is not a recognized type. Returning None.")
            return None
        if subTopologyType.lower() == "self":
            t = 0
        else:
            t = Topology.TypeID(subTopologyType)
        return topology.AddContents(contents, t)
    
    @staticmethod
    def AddDictionary(topology, dictionary):
        """
        Adds the input dictionary to the input topology.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        dictionary : topologic_core.Dictionary
            The input dictionary.

        Returns
        -------
        topologic_core.Topology
            The input topology with the input dictionary added to it.

        """
        from topologicpy.Dictionary import Dictionary

        if not Topology.IsInstance(topology, "Topology"):
            print("Topology.AddDictionary - Error: the input topology parameter is not a valid topology. Returning None.")
            return None
        if not Topology.IsInstance(dictionary, "Dictionary"):
            print("Topology.AddDictionary - Error: the input dictionary parameter is not a dictionary. Returning None.")
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
        Returns the topologies, as specified by the input topology type, adjacent to the input topology within the input host topology.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        hostTopology : topologic_core.Topology
            The host topology in which to search.
        topologyType : str
            The type of topology for which to search. This can be one of "vertex", "edge", "wire", "face", "shell", "cell", "cellcomplex". It is case-insensitive. If it is set to None, the type will be set to the same type as the input topology. The default is None.

        Returns
        -------
        adjacentTopologies : list
            The list of adjacent topologies.

        """
        if not Topology.IsInstance(topology, "Topology"):
            print("Topology.AdjacentTopologies - Error: the input topology parameter is not a valid topology. Returning None.")
            return None
        if not Topology.IsInstance(hostTopology, "Topology"):
            print("Topology.AdjacentTopologies - Error: the input hostTopology parameter is not a valid topology. Returning None.")
            return None
        if not topologyType:
            topologyType = Topology.TypeAsString(topology).lower()
        if not isinstance(topologyType, str):
            print("Topology.AdjacentTopologies - Error: the input topologyType parameter is not a string. Returning None.")
            return None
        if not topologyType.lower() in ["vertex", "edge", "wire", "face", "shell", "cell", "cellcomplex"]:
            print("Topology.AdjacentTopologies - Error: the input topologyType parameter is not a recognized type. Returning None.")
            return None
        adjacentTopologies = []
        error = False
        if Topology.IsInstance(topology, "Vertex"):
            if topologyType.lower() == "vertex":
                try:
                    _ = topology.AdjacentVertices(hostTopology, adjacentTopologies) # Hook to Core
                except:
                    try:
                        _ = topology.Vertices(hostTopology, adjacentTopologies) # Hook to Core
                    except:
                        error = True
            elif topologyType.lower() == "edge":
                try:
                    _ = topologic.VertexUtility.AdjacentEdges(topology, hostTopology, adjacentTopologies) # Hook to Core
                except:
                    try:
                        _ = topology.Edges(hostTopology, adjacentTopologies) # Hook to Core
                    except:
                        error = True
            elif topologyType.lower() == "wire":
                try:
                    _ = topologic.VertexUtility.AdjacentWires(topology, hostTopology, adjacentTopologies) # Hook to Core
                except:
                    try:
                        _ = topology.Wires(hostTopology, adjacentTopologies) # Hook to Core
                    except:
                        error = True
            elif topologyType.lower() == "face":
                try:
                    _ = topologic.VertexUtility.AdjacentFaces(topology, hostTopology, adjacentTopologies) # Hook to Core
                except:
                    try:
                        _ = topology.Faces(hostTopology, adjacentTopologies) # Hook to Core
                    except:
                        error = True
            elif topologyType.lower() == "shell":
                try:
                    _ = topologic.VertexUtility.AdjacentShells(topology, hostTopology, adjacentTopologies) # Hook to Core
                except:
                    try:
                        _ = topology.Shells(hostTopology, adjacentTopologies) # Hook to Core
                    except:
                        error = True
            elif topologyType.lower() == "cell":
                try:
                    _ = topologic.VertexUtility.AdjacentCells(topology, hostTopology, adjacentTopologies) # Hook to Core
                except:
                    try:
                        _ = topology.Cells(hostTopology, adjacentTopologies) # Hook to Core
                    except:
                        error = True
            elif topologyType.lower() == "cellcomplex":
                try:
                    _ = topologic.VertexUtility.AdjacentCellComplexes(topology, hostTopology, adjacentTopologies) # Hook to Core
                except:
                    try:
                        _ = topology.CellComplexes(hostTopology, adjacentTopologies) # Hook to Core
                    except:
                        error = True
        elif Topology.IsInstance(topology, "Edge"):
            if topologyType.lower() == "vertex":
                try:
                    _ = topology.Vertices(hostTopology, adjacentTopologies) # Hook to Core
                except:
                    error = True
            elif topologyType.lower() == "edge":
                try:
                    _ = topology.AdjacentEdges(hostTopology, adjacentTopologies) # Hook to Core
                except:
                    error = True
            elif topologyType.lower() == "wire":
                try:
                    _ = topologic.EdgeUtility.AdjacentWires(topology, adjacentTopologies) # Hook to Core
                except:
                    try:
                        _ = topology.Wires(hostTopology, adjacentTopologies) # Hook to Core
                    except:
                        error = True
            elif topologyType.lower() == "face":
                try:
                    _ = topologic.EdgeUtility.AdjacentFaces(topology, adjacentTopologies) # Hook to Core
                except:
                    try:
                        _ = topology.Faces(hostTopology, adjacentTopologies) # Hook to Core
                    except:
                        error = True
            elif topologyType.lower() == "shell":
                try:
                    _ = topologic.EdgeUtility.AdjacentShells(adjacentTopologies) # Hook to Core
                except:
                    try:
                        _ = topology.Shells(hostTopology, adjacentTopologies) # Hook to Core
                    except:
                        error = True
            elif topologyType.lower() == "cell":
                try:
                    _ = topologic.EdgeUtility.AdjacentCells(adjacentTopologies) # Hook to Core
                except:
                    try:
                        _ = topology.Cells(hostTopology, adjacentTopologies) # Hook to Core
                    except:
                        error = True
            elif topologyType.lower() == "cellcomplex":
                try:
                    _ = topologic.EdgeUtility.AdjacentCellComplexes(adjacentTopologies) # Hook to Core
                except:
                    try:
                        _ = topology.CellComplexes(hostTopology, adjacentTopologies) # Hook to Core
                    except:
                        error = True
        elif Topology.IsInstance(topology, "Wire"):
            if topologyType.lower() == "vertex":
                try:
                    _ = topologic.WireUtility.AdjacentVertices(topology, adjacentTopologies) # Hook to Core
                except:
                    try:
                        _ = topology.Vertices(hostTopology, adjacentTopologies) # Hook to Core
                    except:
                        error = True
            elif topologyType.lower() == "edge":
                try:
                    _ = topologic.WireUtility.AdjacentEdges(topology, adjacentTopologies) # Hook to Core
                except:
                    try:
                        _ = topology.Edges(hostTopology, adjacentTopologies) # Hook to Core
                    except:
                        error = True
            elif topologyType.lower() == "wire":
                try:
                    _ = topologic.WireUtility.AdjacentWires(topology, adjacentTopologies) # Hook to Core
                except:
                    try:
                        _ = topology.Wires(hostTopology, adjacentTopologies) # Hook to Core
                    except:
                        error = True
            elif topologyType.lower() == "face":
                try:
                    _ = topologic.WireUtility.AdjacentFaces(topology, adjacentTopologies) # Hook to Core
                except:
                    try:
                        _ = topology.Faces(hostTopology, adjacentTopologies) # Hook to Core
                    except:
                        error = True
            elif topologyType.lower() == "shell":
                try:
                    _ = topologic.WireUtility.AdjacentShells(adjacentTopologies) # Hook to Core
                except:
                    try:
                        _ = topology.Shells(hostTopology, adjacentTopologies) # Hook to Core
                    except:
                        error = True
            elif topologyType.lower() == "cell":
                try:
                    _ = topologic.WireUtility.AdjacentCells(adjacentTopologies) # Hook to Core
                except:
                    try:
                        _ = topology.Cells(hostTopology, adjacentTopologies) # Hook to Core
                    except:
                        error = True
            elif topologyType.lower() == "cellcomplex":
                try:
                    _ = topologic.WireUtility.AdjacentCellComplexes(adjacentTopologies) # Hook to Core
                except:
                    try:
                        _ = topology.CellComplexes(hostTopology, adjacentTopologies) # Hook to Core
                    except:
                        error = True
        elif Topology.IsInstance(topology, "Face"):
            if topologyType.lower() == "vertex":
                try:
                    _ = topologic.FaceUtility.AdjacentVertices(topology, adjacentTopologies) # Hook to Core
                except:
                    try:
                        _ = topology.Vertices(hostTopology, adjacentTopologies) # Hook to Core
                    except:
                        error = True
            elif topologyType.lower() == "edge":
                try:
                    _ = topologic.FaceUtility.AdjacentEdges(topology, adjacentTopologies) # Hook to Core
                except:
                    try:
                        _ = topology.Edges(hostTopology, adjacentTopologies) # Hook to Core
                    except:
                        error = True
            elif topologyType.lower() == "wire":
                try:
                    _ = topologic.FaceUtility.AdjacentWires(topology, adjacentTopologies) # Hook to Core
                except:
                    try:
                        _ = topology.Wires(hostTopology, adjacentTopologies) # Hook to Core
                    except:
                        error = True
            elif topologyType.lower() == "face":
                _ = topology.AdjacentFaces(hostTopology, adjacentTopologies) # Hook to Core
            elif topologyType.lower() == "shell":
                try:
                    _ = topologic.FaceUtility.AdjacentShells(adjacentTopologies) # Hook to Core
                except:
                    try:
                        _ = topology.Shells(hostTopology, adjacentTopologies) # Hook to Core
                    except:
                        error = True
            elif topologyType.lower() == "cell":
                try:
                    _ = topologic.FaceUtility.AdjacentCells(adjacentTopologies) # Hook to Core
                except:
                    try:
                        _ = topology.Cells(hostTopology, adjacentTopologies) # Hook to Core
                    except:
                        error = True
            elif topologyType.lower() == "cellcomplex":
                try:
                    _ = topologic.FaceUtility.AdjacentCellComplexes(adjacentTopologies) # Hook to Core
                except:
                    try:
                        _ = topology.CellComplexes(hostTopology, adjacentTopologies) # Hook to Core
                    except:
                        error = True
        elif Topology.IsInstance(topology, "Shell"):
            if topologyType.lower() == "vertex":
                try:
                    _ = topologic.ShellUtility.AdjacentVertices(topology, adjacentTopologies) # Hook to Core
                except:
                    try:
                        _ = topology.Vertices(hostTopology, adjacentTopologies) # Hook to Core
                    except:
                        error = True
            elif topologyType.lower() == "edge":
                try:
                    _ = topologic.ShellUtility.AdjacentEdges(topology, adjacentTopologies) # Hook to Core
                except:
                    try:
                        _ = topology.Edges(hostTopology, adjacentTopologies) # Hook to Core
                    except:
                        error = True
            elif topologyType.lower() == "wire":
                try:
                    _ = topologic.ShellUtility.AdjacentWires(topology, adjacentTopologies) # Hook to Core
                except:
                    try:
                        _ = topology.Wires(hostTopology, adjacentTopologies) # Hook to Core
                    except:
                        error = True
            elif topologyType.lower() == "face":
                try:
                    _ = topologic.ShellUtility.AdjacentFaces(topology, adjacentTopologies) # Hook to Core
                except:
                    try:
                        _ = topology.Faces(hostTopology, adjacentTopologies) # Hook to Core
                    except:
                        error = True
            elif topologyType.lower() == "shell":
                try:
                    _ = topologic.ShellUtility.AdjacentShells(adjacentTopologies) # Hook to Core
                except:
                    try:
                        _ = topology.Shells(hostTopology, adjacentTopologies) # Hook to Core
                    except:
                        error = True
            elif topologyType.lower() == "cell":
                try:
                    _ = topologic.ShellUtility.AdjacentCells(adjacentTopologies) # Hook to Core
                except:
                    try:
                        _ = topology.Cells(hostTopology, adjacentTopologies) # Hook to Core
                    except:
                        error = True
            elif topologyType.lower() == "cellcomplex":
                try:
                    _ = topologic.ShellUtility.AdjacentCellComplexes(adjacentTopologies) # Hook to Core
                except:
                    try:
                        _ = topology.CellComplexes(hostTopology, adjacentTopologies) # Hook to Core
                    except:
                        error = True
        elif Topology.IsInstance(topology, "Cell"):
            if topologyType.lower() == "vertex":
                try:
                    _ = topologic.CellUtility.AdjacentVertices(topology, adjacentTopologies) # Hook to Core
                except:
                    try:
                        _ = topology.Vertices(hostTopology, adjacentTopologies) # Hook to Core
                    except:
                        error = True
            elif topologyType.lower() == "edge":
                try:
                    _ = topologic.CellUtility.AdjacentEdges(topology, adjacentTopologies) # Hook to Core
                except:
                    try:
                        _ = topology.Edges(hostTopology, adjacentTopologies) # Hook to Core
                    except:
                        error = True
            elif topologyType.lower() == "wire":
                try:
                    _ = topologic.CellUtility.AdjacentWires(topology, adjacentTopologies) # Hook to Core
                except:
                    try:
                        _ = topology.Wires(hostTopology, adjacentTopologies) # Hook to Core
                    except:
                        error = True
            elif topologyType.lower() == "face":
                try:
                    _ = topologic.CellUtility.AdjacentFaces(topology, adjacentTopologies) # Hook to Core
                except:
                    try:
                        _ = topology.Faces(hostTopology, adjacentTopologies) # Hook to Core
                    except:
                        error = True
            elif topologyType.lower() == "shell":
                try:
                    _ = topologic.CellUtility.AdjacentShells(adjacentTopologies) # Hook to Core
                except:
                    try:
                        _ = topology.Shells(hostTopology, adjacentTopologies) # Hook to Core
                    except:
                        error = True
            elif topologyType.lower() == "cell":
                try:
                    _ = topology.AdjacentCells(hostTopology, adjacentTopologies) # Hook to Core
                except:
                    try:
                        _ = topology.Cells(hostTopology, adjacentTopologies) # Hook to Core
                    except:
                        error = True
            elif topologyType.lower() == "cellcomplex":
                try:
                    _ = topologic.CellUtility.AdjacentCellComplexes(adjacentTopologies) # Hook to Core
                except:
                    try:
                        _ = topology.CellComplexes(hostTopology, adjacentTopologies) # Hook to Core
                    except:
                        error = True
        elif Topology.IsInstance(topology, "CellComplex"):
            if topologyType.lower() == "vertex":
                try:
                    _ = topology.Vertices(hostTopology, adjacentTopologies) # Hook to Core
                except:
                    error = True
            elif topologyType.lower() == "edge":
                try:
                    _ = topology.Edges(hostTopology, adjacentTopologies) # Hook to Core
                except:
                    error = True
            elif topologyType.lower() == "wire":
                try:
                    _ = topology.Wires(hostTopology, adjacentTopologies) # Hook to Core
                except:
                    error = True
            elif topologyType.lower() == "face":
                try:
                    _ = topology.Faces(hostTopology, adjacentTopologies) # Hook to Core
                except:
                    error = True
            elif topologyType.lower() == "shell":
                try:
                    _ = topology.Shells(hostTopology, adjacentTopologies) # Hook to Core
                except:
                    error = True
            elif topologyType.lower() == "cell":
                try:
                    _ = topology.Cells(hostTopology, adjacentTopologies) # Hook to Core
                except:
                    error = True
            elif topologyType.lower() == "cellcomplex":
                raise Exception("Topology.AdjacentTopologies - Error: Cannot search for adjacent topologies of a CellComplex")
        elif Topology.IsInstance(topology, "Cluster"):
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
        topology : topologic_core.Topology
            The input topology.

        Returns
        -------
        str
            The analysis string.

        """
        if not Topology.IsInstance(topology, "Topology"):
            print("Topology.Analyze - Error: the input topology parameter is not a valid topology. Returning None.")
            return None
        return topologic.Topology.Analyze(topology)
    
    @staticmethod
    def Apertures(topology, subTopologyType=None):
        """
        Returns the apertures of the input topology.
        
        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        subTopologyType : string , optional
            The subtopology type from which to retrieve the apertures. This can be "cell", "face", "edge", or "vertex" or "all". It is case insensitive. If set to "all", then all apertures will be returned. If set to None, the apertures will be retrieved only from the input topology. The default is None.
        
        Returns
        -------
        list
            The list of apertures belonging to the input topology.

        """

        from topologicpy.Dictionary import Dictionary

        if not Topology.IsInstance(topology, "Topology"):
            print("Topology.Apertures - Error: the input topology parameter is not a valid topology. Returning None.")
            return None
        
        apertures = []
        subTopologies = []
        if not subTopologyType:
            _ = topology.Apertures(apertures)
            apertures = [x.Topology() for x in apertures]
            contents = Topology.Contents(topology)
            for content in contents:
                d = Topology.Dictionary(content)
                if len(Dictionary.Keys(d)) > 0:
                    type = Dictionary.ValueAtKey(d,"type")
                    if "aperture" in type.lower():
                        apertures.append(content)
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
            apertures = [x.Topology() for x in apertures]
            subTopologies = Topology.Vertices(topology)
            subTopologies += Topology.Edges(topology)
            subTopologies += Topology.Faces(topology)
            subTopologies += Topology.Cells(topology)
        else:
            print("Topology.Apertures - Error: the input subtopologyType parameter is not a recognized type. Returning None.")
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
        topology : topologic_core.Topology
            The input topology.
        subTopologyType : string , optional
            The subtopology type from which to retrieve the apertures. This can be "cell", "face", "edge", or "vertex" or "all". It is case insensitive. If set to "all", then all apertures will be returned. If set to None, the apertures will be retrieved only from the input topology. The default is None.
       
        Returns
        -------
        list
            The list of aperture topologies found in the input topology.

        """
        from topologicpy.Aperture import Aperture
        if not Topology.IsInstance(topology, "Topology"):
            print("Topology.ApertureTopologies - Error: the input topology parameter is not a valid topology. Returning None.")
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
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Shell import Shell

        if Topology.IsInstance(topologyA, "Face") and Topology.IsInstance(topologyB, "Face"):
            if Face.IsCoplanar(topologyA, topologyB):
                topologyC = Topology.Boolean(topologyA, topologyB, operation="merge", tranDict=tranDict, tolerance=tolerance)
                if Topology.IsInstance(topologyC, "Cluster"):
                    return topologyC
                elif Topology.IsInstance(topologyC, "Shell"):
                    eb_list = Shell.ExternalBoundary(topologyC)
                    if Topology.IsInstance(eb_list, "Cluster"):
                        eb_list = Topology.Wires(eb_list)
                    else:
                        eb_list = [eb_list]
                    topologyA_wire = Face.ExternalBoundary(topologyA)
                    topologyB_wire = Face.ExternalBoundary(topologyB)
                    internal_boundaries = []
                    found = False
                    for i, eb in enumerate(eb_list):
                        v = Topology.Vertices(eb)[0]
                        if found == False:
                            if Vertex.IsInternal(v, topologyA_wire) or Vertex.IsInternal(v, topologyB_wire):
                                external_boundary = eb
                                found = True
                        else:
                            internal_boundaries.append(eb)
                    return Face.ByWires(external_boundary, internal_boundaries)
        return Topology.Boolean(topologyA, topologyB, operation="union", tranDict=tranDict, tolerance=tolerance)
    
    @staticmethod
    def Difference(topologyA, topologyB, tranDict=False, tolerance=0.0001):
        """
        See Topology.Boolean().

        """
        return Topology.Boolean(topologyA=topologyA, topologyB=topologyB, operation="difference", tranDict=tranDict, tolerance=tolerance)
    
    @staticmethod
    def ExternalBoundary(topology):
        """
        Returns the external boundary of the input topology.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.

        Returns
        -------
        topologic_core.Topology
            The external boundary of the input topology.

        """
        from topologicpy.Topology import Topology
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Shell import Shell
        from topologicpy.Cell import Cell
        from topologicpy.CellComplex import CellComplex
        from topologicpy.Cluster import Cluster

        if not Topology.IsInstance(topology, "Topology"):
            print("Topology.ExternalBoundary - Error: The input topology parameter is not a valid topology. Returning None.")
            return None
        
        if Topology.IsInstance(topology, "Vertex"):
            return Vertex.ExternalBoundary(topology)
        elif Topology.IsInstance(topology, "Edge"):
            return Edge.ExternalBoundary(topology)
        elif Topology.IsInstance(topology, "Wire"):
            return Wire.ExternalBoundary(topology)
        elif Topology.IsInstance(topology, "Face"):
            return Face.ExternalBoundary(topology)
        elif Topology.IsInstance(topology, "Shell"):
            return Shell.ExternalBoundary(topology)
        elif Topology.IsInstance(topology, "Cell"):
            return Cell.ExternalBoundary(topology)
        elif Topology.IsInstance(topology, "CellComplex"):
            return CellComplex.ExternalBoundary(topology)
        elif Topology.IsInstance(topology, "Cluster"):
            eb_list = Cluster.CellComplexes(topology) + Cluster.FreeCells(topology) + Cluster.FreeShells(topology) + Cluster.FreeFaces(topology) + Cluster.FreeWires(topology) + Cluster.FreeEdges(topology) + Cluster.FreeVertices(topology)
            return_list = []
            for item in eb_list:
                return_list.append(Topology.ExternalBoundary(item))
            return Cluster.ByTopologies(return_list)
        else:
            return None
        
    @staticmethod
    def Intersect(topologyA, topologyB, tranDict=False, tolerance=0.0001):
        """
        See Topology.Boolean().

        """
        from topologicpy.Cluster import Cluster

        if topologyA == None:
            return None
        if topologyB == None:
            return None
        
        from topologicpy.Vertex import Vertex
        # Sort the two topologies by their type from lower to higher so comparison can be eased.
        if Topology.Type(topologyB) < Topology.Type(topologyA):
            temp = topologyA
            topologyA = topologyB
            topologyB = temp
        
        results = []
        if Topology.IsInstance(topologyA, "CellComplex"):
           cellsA = Topology.Cells(topologyA)
        elif Topology.IsInstance(topologyA, "Cluster"):
            cellsA = Cluster.FreeTopologies(topologyA)
        else:
            cellsA = [topologyA]
        if Topology.IsInstance(topologyB, "CellComplex"):
                cellsB = Topology.Cells(topologyB)
        elif Topology.IsInstance(topologyB, "Cluster"):
            cellsB = Cluster.FreeTopologies(topologyB)
        else:
            cellsB = [topologyB]
        cellsA_2 = []
        cellsB_2 = []
        for cellA in cellsA:
            if Topology.IsInstance(cellA, "CellComplex"):
                cellsA_2 += Topology.Cells(cellA)
            elif Topology.IsInstance(cellA, "Shell"):
                cellsA_2 += Topology.Faces(cellA)
            else:
                cellsA_2.append(cellA)
            
        for cellB in cellsB:
            if Topology.IsInstance(cellB, "CellComplex"):
                cellsB_2 += Topology.Cells(cellB)
            elif Topology.IsInstance(cellB, "Shell"):
                cellsB_2 += Topology.Faces(cellB)
            else:
                cellsB_2.append(cellB)
        
        for cellA in cellsA_2:
            for cellB in cellsB_2:
                cellC = cellA.Intersect(cellB)
                results.append(cellC)
        results = [x for x in results if x is not None]
        if len(results) == 0:
            return None
        elif len(results) == 1:
            return results[0]
        else:
            return Topology.SelfMerge(Topology.SelfMerge(Cluster.ByTopologies(results)))
    
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
        topologyA : topologic_core.Topology
            The first input topology.
        topologyB : topologic_core.Topology
            The second input topology.
        operation : str , optional
            The boolean operation. This can be one of "union", "difference", "intersect", "symdif", "merge", "slice", "impose", "imprint". It is case insensitive. The default is "union".
        tranDict : bool , optional
            If set to True the dictionaries of the operands are merged and transferred to the result. The default is False.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Topology
            the resultant topology.

        """
        from topologicpy.Dictionary import Dictionary
        if not Topology.IsInstance(topologyA, "Topology"):
            print("Topology.Boolean - Error: the input topologyA parameter is not a valid topology. Returning None.")
            return None
        if not Topology.IsInstance(topologyB, "Topology"):
            print("Topology.Boolean - Error: the input topologyB parameter is not a valid topology. Returning None.")
            return None
        if not isinstance(operation, str):
            print("Topology.Boolean - Error: the input operation parameter is not a valid string. Returning None.")
            return None
        if not operation.lower() in ["union", "difference", "intersect", "symdif", "merge", "slice", "impose", "imprint"]:
            print("Topology.Boolean - Error: the input operation parameter is not a recognized operation. Returning None.")
            return None
        if not isinstance(tranDict, bool):
            print("Topology.Boolean - Error: the input tranDict parameter is not a valid boolean. Returning None.")
            return None
        topologyC = None
        #topologyC = Topology.Intersect(topologyA, topologyB)
        #try:
        if operation.lower() == "union":
            topologyC = topologyA.Union(topologyB, False)
        elif operation.lower() == "difference":
            topologyC = topologyA.Difference(topologyB, False)
        elif operation.lower() == "intersect": #Intersect in Topologic (Core) is faulty. This is a workaround.
            #topologyC = topologyA.Intersect(topologyB, False)
            
            topologyC = Topology.Intersect(topologyA, topologyB)
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
            print("1. Topology.Boolean - Error: the boolean operation failed. Returning None.")
            return None
        #except:
            #print("2. Topology.Boolean - Error: the boolean operation failed. Returning None.")
            #return None
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

            if Topology.Type(topologyA) == Topology.TypeID("Vertex"):
                sourceVertices += [topologyA]
            elif hidimA >= Topology.TypeID("Vertex"):
                sourceVertices += Topology.Vertices(topologyA)
            if Topology.Type(topologyB) == Topology.TypeID("Vertex"):
                sourceVertices += [topologyB]
            elif hidimB >= Topology.TypeID("Vertex"):
                sourceVertices += Topology.Vertices(topologyB)
            if Topology.Type(topologyC) == Topology.TypeID("Vertex"):
                sinkVertices = [topologyC]
            elif hidimC >= Topology.TypeID("Vertex"):
                sinkVertices = Topology.Vertices(topologyC)
            if len(sourceVertices) > 0 and len(sinkVertices) > 0:
                _ = Topology.TransferDictionaries(sourceVertices, sinkVertices, tolerance=tolerance)

            if Topology.Type(topologyA) == Topology.TypeID("Edge"):
                sourceEdges += [topologyA]
            elif hidimA >= Topology.TypeID("Edge"):
                sourceEdges += Topology.Edges(topologyA)
            if Topology.Type(topologyB) == Topology.TypeID("Edge"):
                sourceEdges += [topologyB]
            elif hidimB >= Topology.TypeID("Edge"):
                sourceEdges += Topology.Edges(topologyB)
            if Topology.Type(topologyC) == Topology.TypeID("Edge"):
                sinkEdges = [topologyC]
            elif hidimC >= Topology.TypeID("Edge"):
                sinkEdges = Topology.Edges(topologyC)
            if len(sourceEdges) > 0 and len(sinkEdges) > 0:
                _ = Topology.TransferDictionaries(sourceEdges, sinkEdges, tolerance=tolerance)

            if Topology.Type(topologyA) == Topology.TypeID("Face"):
                sourceFaces += [topologyA]
            elif hidimA >= Topology.TypeID("Face"):
                sourceFaces += Topology.Faces(topologyA)
            if Topology.Type(topologyB) == Topology.TypeID("Face"):
                sourceFaces += [topologyB]
            elif hidimB >= Topology.TypeID("Face"):
                sourceFaces += Topology.Faces(topologyB)
            if Topology.Type(topologyC) == Topology.TypeID("Face"):
                sinkFaces += [topologyC]
            elif hidimC >= Topology.TypeID("Face"):
                sinkFaces += Topology.Faces(topologyC)
            if len(sourceFaces) > 0 and len(sinkFaces) > 0:
                _ = Topology.TransferDictionaries(sourceFaces, sinkFaces, tolerance=tolerance)

            if Topology.Type(topologyA) == Topology.TypeID("Cell"):
                sourceCells += [topologyA]
            elif hidimA >= Topology.TypeID("Cell"):
                sourceCells += Topology.Cells(topologyA)
            if Topology.Type(topologyB) == Topology.TypeID("Cell"):
                sourceCells += [topologyB]
            elif hidimB >= Topology.TypeID("Cell"):
                sourceCells += Topology.Cells(topologyB)
            if Topology.Type(topologyC) == Topology.TypeID("Cell"):
                sinkCells = [topologyC]
            elif hidimC >= Topology.TypeID("Cell"):
                sinkCells = Topology.Cells(topologyC)
            if len(sourceCells) > 0 and len(sinkCells) > 0:
                _ = Topology.TransferDictionaries(sourceCells, sinkCells, tolerance=tolerance)
        return topologyC

    
    @staticmethod
    def BoundingBox(topology, optimize: int = 0, axes: str ="xyz", mantissa: int = 6, tolerance: float = 0.0001):
        """
        Returns a cell representing a bounding box of the input topology. The returned cell contains a dictionary with keys "xrot", "yrot", and "zrot" that represents rotations around the X, Y, and Z axes. If applied in the order of Z, Y, X, the resulting box will become axis-aligned.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        optimize : int , optional
            If set to an integer from 1 (low optimization) to 10 (high optimization), the method will attempt to optimize the bounding box so that it reduces its surface area. The default is 0 which will result in an axis-aligned bounding box. The default is 0.
        axes : str , optional
            Sets what axes are to be used for rotating the bounding box. This can be any permutation or substring of "xyz". It is not case sensitive. The default is "xyz".
        mantissa : int , optional
            The desired length of the mantissa. The default is 6
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
        Returns
        -------
        topologic_core.Cell or topologic_core.Face
            The bounding box of the input topology.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Cell import Cell
        from topologicpy.Cluster import Cluster
        from topologicpy.Dictionary import Dictionary

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

        if not Topology.IsInstance(topology, "Topology"):
            print("Topology.BoundingBox - Error: the input topology parameter is not a valid topology. Returning None.")
            return None
        if not isinstance(axes, str):
            print("Topology.BoundingBox - Error: the input axes parameter is not a valid string. Returning None.")
            return None
        axes = axes.lower()
        x_flag = "x" in axes
        y_flag = "y" in axes
        z_flag = "z" in axes
        if not x_flag and not y_flag and not z_flag:
            print("Topology.BoundingBox - Error: the input axes parameter is not a recognized string. Returning None.")
            return None
        if Topology.IsInstance(topology, "Vertex"):
            x_min = Vertex.X(topology)
            y_min = Vertex.Y(topology)
            z_min = Vertex.Z(topology)
            dictionary = Dictionary.ByKeysValues(["xrot","yrot","zrot", "xmin", "ymin", "zmin", "xmax", "ymax", "zmax", "width", "length", "height"], [0, 0, 0, x_min, y_min, z_min, x_min, y_min, z_min, 0, 0, 0])
            box = Vertex.ByCoordinates(x_min, y_min, z_min)
            box = Topology.SetDictionary(box, dictionary)
            return box
        vertices = Topology.SubTopologies(topology, subTopologyType="vertex")
        if len(vertices) == 1: # A Cluster made of one vertex. Rare, but can happen!
            x_min = Vertex.X(vertices[0])
            y_min = Vertex.Y(vertices[0])
            z_min = Vertex.Z(vertices[0])
            dictionary = Dictionary.ByKeysValues(["xrot","yrot","zrot", "xmin", "ymin", "zmin", "xmax", "ymax", "zmax", "width", "length", "height"], [0, 0, 0, x_min, y_min, z_min, x_min, y_min, z_min, 0, 0, 0])
            box = Vertex.ByCoordinates(x_min, y_min, z_min)
            box = Topology.SetDictionary(box, dictionary)
            return box
        topology = Cluster.ByTopologies(vertices)
        boundingBox = bb(topology)
        x_min = boundingBox[0]
        y_min = boundingBox[1]
        z_min = boundingBox[2]
        x_max = boundingBox[3]
        y_max = boundingBox[4]
        z_max = boundingBox[5]
        w = abs(x_max - x_min)
        l = abs(y_max - y_min)
        h = abs(z_max - z_min)
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
            for n in range(10, 0, -1):
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
                            t = Topology.Rotate(topology, origin=origin, axis=[0, 0, 1], angle=z)
                            t = Topology.Rotate(t, origin=origin, axis=[0, 1, 0], angle=y)
                            t = Topology.Rotate(t, origin=origin, axis=[1, 0, 0], angle=x)
                            x_min, y_min, z_min, x_max, y_max, z_max = bb(t)
                            w = abs(x_max - x_min)
                            l = abs(y_max - y_min)
                            h = abs(z_max - z_min)
                            area = 2*l*w + 2*l*h + 2*w*h
                            if area < orig_area*factor:
                                best_area = area
                                best_x = x
                                best_y = y
                                best_z = z
                                best_bb = [x_min, y_min, z_min, x_max, y_max, z_max]
                                flag = True
                                break
                            if area < best_area:
                                best_area = area
                                best_x = x
                                best_y = y
                                best_z = z
                                best_bb = [x_min, y_min, z_min, x_max, y_max, z_max]
                        
        else:
            best_bb = boundingBox

        x_min, y_min, z_min, x_max, y_max, z_max = best_bb
        vb1 = Vertex.ByCoordinates(x_min, y_min, z_min)
        vb2 = Vertex.ByCoordinates(x_max, y_min, z_min)
        vb3 = Vertex.ByCoordinates(x_max, y_max, z_min)
        vb4 = Vertex.ByCoordinates(x_min, y_max, z_min)

        baseWire = Wire.ByVertices([vb1, vb2, vb3, vb4], close=True)
        baseFace = Face.ByWire(baseWire, tolerance=tolerance)
        if abs(z_max - z_min) < tolerance:
            box = baseFace
        else:
            box = Cell.ByThickenedFace(baseFace, planarize=False, thickness=abs(z_max - z_min), bothSides=False)
        box = Topology.Rotate(box, origin=origin, axis=[1, 0, 0], angle=-best_x)
        box = Topology.Rotate(box, origin=origin, axis=[0, 1, 0], angle=-best_y)
        box = Topology.Rotate(box, origin=origin, axis=[0, 0, 1], angle=-best_z)
        dictionary = Dictionary.ByKeysValues(["xrot","yrot","zrot", "xmin", "ymin", "zmin", "xmax", "ymax", "zmax", "width", "length", "height"], [best_x, best_y, best_z, x_min, y_min, z_min, x_max, y_max, z_max, (x_max - x_min), (y_max - y_min), (z_max - z_min)])
        box = Topology.SetDictionary(box, dictionary)
        return box

    @staticmethod
    def BREPString(topology, version=3):
        """
        Returns the BRep string of the input topology.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        version : int , optional
            The desired BRep version number. The default is 3.

        Returns
        -------
        str
            The BREP string.

        """
        if not Topology.IsInstance(topology, "Topology"):
            print("Topology.BREPString - Error: the input topology parameter is not a valid topology. Returning None.")
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
    def ByGeometry(vertices=[], edges=[], faces=[], topologyType: str = None, tolerance: float = 0.0001, silent: bool = False):
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
        topologyType : str , optional
            The desired highest topology type. The options are: "Vertex", "Edge", "Wire", "Face", "Shell", "Cell", "CellComplex".
            It is case insensitive. If any of these options are selected, the returned topology will only contain this type either a single topology
            or as a Cluster of these types of topologies. If set to None, a "Cluster" will be returned of vertices, edges, and/or faces. The default is None.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        topology : topologic_core.Topology
            The created topology. The topology will have a dictionary embedded in it that records the input attributes (color, id, lengthUnit, name, type)

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Shell import Shell
        from topologicpy.Cell import Cell
        from topologicpy.CellComplex import CellComplex
        from topologicpy.Cluster import Cluster

        def topologyByFaces(faces, topologyType, tolerance=0.0001):
            if len(faces) == 1:
                return faces[0]

            output = None
            if topologyType == "cell":
                c = Cell.ByFaces(faces, tolerance=tolerance)
                if Topology.IsInstance(c, "Cell"):
                    output = c
                else:
                    cc = CellComplex.ByFaces(faces, tolerance=tolerance)
                    if Topology.IsInstance(cc, "CellComplex"):
                        output = CellComplex.ExternalBoundary(cc)
            elif topologyType == "cellcomplex":
                output = CellComplex.ByFaces(faces, tolerance=tolerance, silent=silent)
                if Topology.IsInstance(output, "CellComplex"):
                    cells = Topology.Cells(output)
                    if len(cells) == 1:
                        output = cells[0]
                else:
                    output = Cluster.ByTopologies(faces)
            elif topologyType == "shell":
                output = Shell.ByFaces(faces, tolerance=tolerance)  # This can return a list
                if Topology.IsInstance(output, "Shell"):
                    return output
            elif topologyType == None or topologyType== "face":
                output = Cluster.ByTopologies(faces)

            return output

        def topologyByEdges(edges, topologyType):
            if len(edges) == 1:
                return edges[0]

            output = Cluster.ByTopologies(edges)
            if topologyType == "wire":
                output = Topology.SelfMerge(output, tolerance=tolerance)
                if Topology.IsInstance(output, "Wire"):
                    return output
                return None
            return output

        vertices = [v for v in vertices if v]
        edges = [e for e in edges if e]
        faces = [f for f in faces if f]

        if not vertices:
            return None

        topVerts = [Vertex.ByCoordinates(v[0], v[1], v[2]) for v in vertices]
        topEdges = []
        topFaces = []

        if not topologyType == None:
            topologyType = topologyType.lower()

        if topologyType == "vertex":
            if len(topVerts) == 0:
                return None
            if len(topVerts) == 1:
                return topVerts[0]
            else:
                return Cluster.ByTopologies(topVerts)
        elif topologyType == "edge":
            if len(edges) == 0:
                return None
            if len(edges) == 1 and len(vertices) >= 2:
                return Edge.ByVertices(topVerts[edges[0][0]], topVerts[edges[0][1]], tolerance=tolerance)
            else:
                topEdges = [Edge.ByVertices([topVerts[e[0]], topVerts[e[1]]], tolerance=tolerance) for e in edges]
                return Cluster.ByTopologies(topEdges)

        if topologyType == "wire" and edges:
            topEdges = [Edge.ByVertices([topVerts[e[0]], topVerts[e[1]]], tolerance=tolerance) for e in edges]
            if topEdges:
                returnTopology = topologyByEdges(topEdges, topologyType)
        elif faces:
            for aFace in faces:
                faceEdges = [Edge.ByVertices([topVerts[aFace[i]], topVerts[aFace[i + 1]]], tolerance=tolerance) for i in range(len(aFace) - 1)]
                # Connect the last vertex to the first one
                faceEdges.append(Edge.ByVertices([topVerts[aFace[-1]], topVerts[aFace[0]]], tolerance=tolerance))

                if len(faceEdges) > 2:
                    faceWire = Wire.ByEdges(faceEdges, tolerance=tolerance)
                    try:
                        topFace = Face.ByWire(faceWire, tolerance=tolerance, silent=True)
                        if Topology.IsInstance(topFace, "Face"):
                            topFaces.append(topFace)
                        elif isinstance(topFace, list):
                            topFaces.extend(topFace)
                    except:
                        pass
            if topFaces:
                returnTopology = topologyByFaces(topFaces, topologyType=topologyType, tolerance=tolerance)
        elif edges:
            topEdges = [Edge.ByVertices([topVerts[e[0]], topVerts[e[1]]], tolerance=tolerance) for e in edges]
            if topEdges:
                returnTopology = topologyByEdges(topEdges, topologyType)
        else:
            returnTopology = Cluster.ByTopologies(topVerts)
        return returnTopology






    @staticmethod
    def ByGeometry_old(vertices=[], edges=[], faces=[], color=[1.0, 1.0, 1.0, 1.0], id=None, name=None, lengthUnit="METERS", outputMode="default", tolerance=0.0001):
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
            The desired color of the object in the form of [r, g, b, a] where the components are between 0 and 1 and represent red, blue, green, and alpha (transparency) respectively. The default is [1.0, 1.0, 1.0, 1.0].
        id : str , optional
            The desired ID of the object. If set to None, an automatic uuid4 will be assigned to the object. The default is None.
        name : str , optional
            The desired name of the object. If set to None, a default name "Topologic_[topology_type]" will be assigned to the object. The default is None.
        lengthUnit : str , optional
            The length unit used for the object. The default is "METERS"
        outputMode : str , optional
            The desired output mode of the object. This can be "wire", "shell", "cell", "cellcomplex", or "default". It is case insensitive. The default is "default".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topology : topologic_core.Topology
            The created topology. The topology will have a dictionary embedded in it that records the input attributes (color, id, lengthUnit, name, type)

        """
        def topologyByFaces(faces, outputMode, tolerance=0.0001):
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
                output = Shell.ByFaces(faces, tolerance=tolerance) # This can return a list
                if Topology.IsInstance(output, "Shell"):
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
                output = Topology.SelfMerge(output, tolerance=tolerance)
                if Topology.IsInstance(output, "Wire"):
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
                e1 = Edge.ByVertices([topVerts[v1], topVerts[v2]], tolerance=tolerance)
                edges.append(e1)
            # connect the last vertex to the first one
            v1 = vertices[-1]
            v2 = vertices[0]
            e1 = Edge.ByVertices([topVerts[v1], topVerts[v2]], tolerance=tolerance)
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
        vertices = [v for v in vertices if not len(v) == 0]
        edges = [e for e in edges if not len(e) == 0]
        faces = [f for f in faces if not len(f) == 0]
        if len(vertices) > 0:
            for aVertex in vertices:
                v = Vertex.ByCoordinates(aVertex[0], aVertex[1], aVertex[2])
                topVerts.append(v)
        else:
            return None
        if (outputMode.lower == "wire") and (len(edges) > 0):
            for anEdge in edges:
                topEdge = Edge.ByVertices([topVerts[anEdge[0]], topVerts[anEdge[1]]], tolerance=tolerance)
                topEdges.append(topEdge)
            if len(topEdges) > 0:
                returnTopology = topologyByEdges(topEdges)
        elif len(faces) > 0:
            for aFace in faces:
                faceEdges = edgesByVertices(aFace, topVerts)
                if len(faceEdges) > 2:
                    faceWire = Wire.ByEdges(faceEdges, tolerance=tolerance)
                    try:
                        topFace = Face.ByWire(faceWire, tolerance=tolerance, silent=True)
                        if Topology.IsInstance(topFace, "Face"):
                            topFaces.append(topFace)
                        elif isinstance(topFace, list):
                            topFaces += topFace
                    except:
                        pass
            if len(topFaces) > 0:
                returnTopology = topologyByFaces(topFaces, outputMode=outputMode, tolerance=tolerance)
        elif len(edges) > 0:
            for anEdge in edges:
                topEdge = Edge.ByVertices([topVerts[anEdge[0]], topVerts[anEdge[1]]], tolerance=tolerance)
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
                values.append([1.0, 1.0, 1.0, 1.0])
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
    def ByBIMPath(path, guidKey: str = "guid", colorKey: str = "color", typeKey: str = "type",
                        defaultColor: list = [255,255,255,1], defaultType: str = "Structure",
                        authorKey="author", dateKey="date", mantissa: int = 6, angTolerance: float = 0.001, tolerance: float = 0.0001):
        """
        Imports topologies from the input BIM file. See https://dotbim.net/

        Parameters
        ----------
        path :str
            The path to the .bim file.
        guidKey : str , optional
            The key to use to store the the guid of the topology. The default is "guid".
        colorKey : str , optional
            The key to use to find the the color of the topology. The default is "color". If no color is found, the defaultColor parameter is used.
        typeKey : str , optional
            The key to use to find the the type of the topology. The default is "type". If no type is found, the defaultType parameter is used.
        defaultColor : list , optional
            The default color to use for the topology. The default is [255,255,255,1] which is opaque white.
        defaultType : str , optional
            The default type to use for the topology. The default is "Structure".
        authorKey : str , optional
            The key to use to store the author of the topology. The default is "author".
        dateKey : str , optional
            The key to use to store the creation date of the topology. This should be in the formate "DD.MM.YYYY". If no date is found the date of import is used.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        angTolerance : float , optional
                The angle tolerance in degrees under which no rotation is carried out. The default is 0.001 degrees.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        list
            The list of imported topologies

        """
        import json
        if not path:
            print("Topology.ByBIMPath - Error: the input path parameter is not a valid path. Returning None.")
            return None
        topologies = None
        with open(path, "r") as bim_file:
                json_string = str(bim_file.read())
                topologies = Topology.ByBIMString(string=json_string, guidKey=guidKey, colorKey=colorKey, typeKey=typeKey,
                            defaultColor=defaultColor, defaultType=defaultType,
                            authorKey=authorKey, dateKey=dateKey, mantissa=mantissa, angTolerance=angTolerance, tolerance=tolerance)
        try:
            with open(path, "r") as bim_file:
                json_string = str(bim_file.read())
                topologies = Topology.ByBIMString(string=json_string, guidKey=guidKey, colorKey=colorKey, typeKey=typeKey,
                            defaultColor=defaultColor, defaultType=defaultType,
                            authorKey=authorKey, dateKey=dateKey, mantissa=mantissa, angTolerance=angTolerance, tolerance=tolerance)
        except:
            print("Topology.ByBIMPath - Error: the BIM file is not a valid file. Returning None.")
        return topologies
    
    @staticmethod
    def ByBIMString(string, guidKey: str = "guid", colorKey: str = "color", typeKey: str = "type",
                        defaultColor: list = [255,255,255,1], defaultType: str = "Structure",
                        authorKey: str = "author", dateKey: str = "date", mantissa: int = 6, angTolerance: float = 0.001, tolerance: float = 0.0001):
        """
        Imports topologies from the input BIM file. See https://dotbim.net/

        Parameters
        ----------
        string :str
            The input dotbim str (in JSON format).
        guidKey : str , optional
            The key to use to store the the guid of the topology. The default is "guid".
        colorKey : str , optional
            The key to use to find the the color of the topology. The default is "color". If no color is found, the defaultColor parameter is used.
        typeKey : str , optional
            The key to use to find the the type of the topology. The default is "type". If no type is found, the defaultType parameter is used.
        defaultColor : list , optional
            The default color to use for the topology. The default is [255,255,255,1] which is opaque white.
        defaultType : str , optional
            The default type to use for the topology. The default is "Structure".
        authorKey : str , optional
            The key to use to store the author of the topology. The default is "author".
        dateKey : str , optional
            The key to use to store the creation date of the topology. This should be in the formate "DD.MM.YYYY". If no date is found the date of import is used.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        angTolerance : float , optional
                The angle tolerance in degrees under which no rotation is carried out. The default is 0.001 degrees.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        list
            The list of imported topologies

        """
        @staticmethod
        def convert_JSON_to_file(json_dictionary):
            import dotbimpy
            schema_version = json_dictionary["schema_version"]
            elements = json_dictionary["elements"]
            meshes = json_dictionary["meshes"]
            created_info = json_dictionary["info"]

            created_meshes = []
            for i in meshes:
                created_meshes.append(dotbimpy.Mesh(
                    mesh_id=i["mesh_id"],
                    coordinates=i["coordinates"],
                    indices=i["indices"]
                ))

            created_elements = []
            for i in elements:
                new_element = dotbimpy.Element(
                    mesh_id=i["mesh_id"],
                    vector=dotbimpy.Vector(x=i["vector"]["x"],
                                y=i["vector"]["y"],
                                z=i["vector"]["z"]),
                    rotation=dotbimpy.Rotation(qx=i["rotation"]["qx"],
                                    qy=i["rotation"]["qy"],
                                    qz=i["rotation"]["qz"],
                                    qw=i["rotation"]["qw"]),
                    info=i["info"],
                    color=dotbimpy.Color(r=i["color"]["r"],
                                g=i["color"]["g"],
                                b=i["color"]["b"],
                                a=i["color"]["a"]),
                    type=i["type"],
                    guid=i["guid"]
                )
                try:
                    new_element.face_colors = i["face_colors"]
                except KeyError as e:
                    if str(e) == "'face_colors'":
                        pass
                    else:
                        raise
                created_elements.append(new_element)

            file = dotbimpy.File(schema_version=schema_version, meshes=created_meshes, elements=created_elements, info=created_info)
            return file
        json_dictionary = json.loads(string)
        file = convert_JSON_to_file(json_dictionary)
        return Topology.ByBIMFile(file, guidKey=guidKey, colorKey=colorKey, typeKey=typeKey,
                                  defaultColor=defaultColor, defaultType=defaultType,
                                  authorKey=authorKey, dateKey=dateKey,
                                  mantissa=mantissa, angTolerance=angTolerance, tolerance=tolerance)
    @staticmethod
    def ByBIMFile(file, guidKey: str = "guid", colorKey: str = "color", typeKey: str = "type",
                        defaultColor: list = [255,255,255,1], defaultType: str = "Structure",
                        authorKey="author", dateKey="date", mantissa: int = 6, angTolerance: float = 0.001, tolerance: float = 0.0001):
        """
        Imports topologies from the input BIM (dotbimpy.file.File) file object. See https://dotbim.net/

        Parameters
        ----------
        file : dotbimpy.file.File
            The input dotbim file.
        guidKey : str , optional
            The key to use to store the the guid of the topology. The default is "guid".
        colorKey : str , optional
            The key to use to find the the color of the topology. The default is "color". If no color is found, the defaultColor parameter is used.
        typeKey : str , optional
            The key to use to find the the type of the topology. The default is "type". If no type is found, the defaultType parameter is used.
        defaultColor : list , optional
            The default color to use for the topology. The default is [255,255,255,1] which is opaque white.
        defaultType : str , optional
            The default type to use for the topology. The default is "Structure".
        authorKey : str , optional
            The key to use to store the author of the topology. The default is "author".
        dateKey : str , optional
            The key to use to store the creation date of the topology. This should be in the formate "DD.MM.YYYY". If no date is found the date of import is used.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        angTolerance : float , optional
                The angle tolerance in degrees under which no rotation is carried out. The default is 0.001 degrees.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        list
            The list of imported topologies

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        import datetime

        file_info = file.info
        elements = file.elements
        meshes = file.meshes
        final_topologies = []
        topologies = []
        id_list = []
        for mesh in meshes:
            id_list.append(mesh.mesh_id)
            coordinates = mesh.coordinates
            indices = mesh.indices
            coordinates = [coordinates[i:i + 3] for i in range(0, len(coordinates),3)]
            indices = [indices[i:i + 3] for i in range(0, len(indices),3)]
            topology = Topology.ByGeometry(vertices=coordinates, faces=indices, tolerance=tolerance)
            topologies.append(topology)
        
        for element in elements:
            element_info = element.info
            element_info[typeKey] = element.type
            element_info[colorKey] = [element.color.r, element.color.g, element.color.b, float(element.color.a)/float(255)]
            try:
                element_info[guidKey] = element.guid
            except:
                element_info[guidKey] = str(uuid.uuid4())
            try:
                element_info[authorKey] = file_info['author']
            except:
                element_info[authorKey] = "topologicpy"
            # Get the current date
            current_date = datetime.datetime.now()
            # Format the date as a string in DD.MM.YYYY format
            formatted_date = current_date.strftime("%d.%m.%Y")
            try:
                element_info[dateKey] = file_info['date']
            except:
                element_info[dateKey] = formatted_date
            d = Dictionary.ByPythonDictionary(element_info)
            mesh_id = element.mesh_id
            quat = element.rotation
            quaternion = [quat.qx, quat.qy, quat.qz, quat.qw]
            #roll, pitch, yaw = quaternion_to_euler([rot.qx, rot.qy, rot.qz, rot.qw])
            vector = element.vector
            topology = topologies[mesh_id]
            if Topology.IsInstance(topology, "Topology"):
                topology = Topology.RotateByQuaternion(topology=topology, origin=Vertex.Origin(), quaternion=quaternion, angTolerance=angTolerance, tolerance=tolerance)
                topology = Topology.Translate(topology, vector.x, vector.y, vector.z)
                topology = Topology.SetDictionary(topology, d)
                final_topologies.append(topology)
        return final_topologies
    
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
        topologic_core.Topology
            The imported topology.

        """
        topology = None
        if not file:
            print("Topology.ByBREPFile - Error: the input file parameter is not a valid file. Returning None.")
            return None
        brep_string = file.read()
        topology = Topology.ByBREPString(brep_string)
        file.close()
        return topology
    
    @staticmethod
    def ByBREPPath(path):
        """
        Imports a topology from a BREP file path.

        Parameters
        ----------
        path : str
            The path to the BREP file.

        Returns
        -------
        topologic_core.Topology
            The imported topology.

        """
        if not path:
            print("Topology.ByBREPPath - Error: the input path parameter is not a valid path. Returning None.")
            return None
        try:
            file = open(path)
        except:
            print("Topology.ByBREPPath - Error: the BREP file is not a valid file. Returning None.")
            return None
        return Topology.ByBREPFile(file)

    @staticmethod
    def ByDXFFile(file, sides: int = 16):
        """
        Imports a list of topologies from a DXF file.
        This is an experimental method with limited capabilities.

        Parameters
        ----------
        file : a DXF file object
            The DXF file object.
        sides : int , optional
            The desired number of sides of splines. The default is 16.

        Returns
        -------
        list
            The list of imported topologies.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Shell import Shell
        from topologicpy.Cell import Cell
        from topologicpy.CellComplex import CellComplex
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary


        try:
            import ezdxf
        except:
            print("Topology.ByDXFFile - Information: Installing required ezdxf library.")
            try:
                os.system("pip install ezdxf")
            except:
                os.system("pip install ezdxf --user")
            try:
                import ezdxf
                print("Topology.ByDXFFile - Information: ezdxf library installed successfully.")
            except:
                warnings.warn("Topology.ByDXFFile - Error: Could not import ezdxf library. Please install it manually. Returning None.")
                return None

        if not file:
            print("Topology.ByDXFFile - Error: the input file parameter is not a valid file. Returning None.")
            return None
        
        import ezdxf



        def get_layer_color(layers, layer_name):
            # iteration
            for layer in layers:
                if layer_name == layer.dxf.name:
                    if not layer.rgb == None:
                        r,g,b = layer.rgb
                        return [r,g,b]
            return 

        def convert_entity(entity, file, sides=36):
            entity_type = entity.dxftype()
            python_dict = entity.dxf.all_existing_dxf_attribs()
            keys = python_dict.keys()
            for key in keys:
                if python_dict[key].__class__ == ezdxf.acc.vector.Vec3:
                    python_dict[key] = list(python_dict[key])
            rgb_list = None
            try:
                rgb_list = entity.rgb   
            except:
                rgb_list = get_layer_color(file.layers, entity.dxf.layer)
            if rgb_list == None:
                rgb_list = [0,0,0]
            python_dict['color'] = rgb_list
            python_dict['type'] = entity_type
            d = Dictionary.ByPythonDictionary(python_dict)
            
            if entity_type == 'POINT':
                point = entity.dxf.location.xyz
                e = Vertex.ByCoordinates(point[0], point[1], point[2])
                e = Topology.SetDictionary(e, d)
            
            elif entity_type == 'LINE':
                sp = entity.dxf.start.xyz
                ep = entity.dxf.end.xyz
                sv = Vertex.ByCoordinates(sp[0], sp[1], sp[2])
                ev = Vertex.ByCoordinates(ep[0], ep[1], ep[2])
                e = Edge.ByVertices(sv,ev)
                e = Topology.SetDictionary(e, d)
        
            elif entity_type == 'POLYLINE':
                if entity.dxf.flags == 1:
                    closed = True
                else:
                    closed = False
                vertices = []
                for vertex in entity.vertices:
                    point = vertex.dxf.location.xyz
                    vertices.append(Vertex.ByCoordinates(point[0], point[1], point[2]))
                if entity.dxf.hasattr("closed"):
                    closed = entity.closed
                e = Wire.ByVertices(vertices, close=closed)
                e = Topology.SetDictionary(e, d)

            elif entity_type == 'LWPOLYLINE':
                vertices = []
                for point in entity.get_points():
                    vertices.append(Vertex.ByCoordinates(point[0], point[1], 0))
                if entity.dxf.hasattr("closed"):
                    close = entity.closed
                else:
                    close = False
                e = Wire.ByVertices(vertices, close=close)
                e = Topology.SetDictionary(e, d)
        
            elif entity_type == 'CIRCLE':
                center = entity.dxf.center.xyz
                radius = entity.dxf.radius
                num_points = 36  # Approximate the circle with 36 points
                vertices = []
                for i in range(sides):
                    angle = 2 * np.pi * i / num_points
                    x = center[0] + radius * np.cos(angle)
                    y = center[1] + radius * np.sin(angle)
                    z = center[2]
                    vertices.append(Vertex.ByCoordinates(x,y,z))
                e = Wire.ByVertices(vertices, close=True)
                e = Topology.SetDictionary(e, d)
            
            elif entity_type == 'ARC':
                center = entity.dxf.center.xyz
                radius = entity.dxf.radius
                start_angle = np.deg2rad(entity.dxf.start_angle)
                end_angle = np.deg2rad(entity.dxf.end_angle)
                vertices = []
                for i in range(sides+1):
                    angle = start_angle + (end_angle - start_angle) * i / (num_points - 1)
                    x = center[0] + radius * np.cos(angle)
                    y = center[1] + radius * np.sin(angle)
                    z = center[2]
                    vertices.append(Vertex.ByCoordinates(x,y,z))
                e = Wire.ByVertices(vertices, close=False)
                e = Topology.SetDictionary(e, d)

            elif entity_type == 'SPLINE':
                # draw the curve tangents as red lines:
                ct = entity.construction_tool()
                vertices = []
                for t in np.linspace(0, ct.max_t, 64):
                    point, derivative = ct.derivative(t, 1)
                    vertices.append(Vertex.ByCoordinates(list(point)))
                converted_entity = Wire.ByVertices(vertices, close=entity.closed)
                vertices = []
                for i in range(sides+1):
                    if i == 0:
                        u = 0
                    elif i == sides:
                        u = 1
                    else:
                        u = float(i)/float(sides)
                    vertices.append(Wire.VertexByParameter(converted_entity, u))

                e = Wire.ByVertices(vertices, close=entity.closed)
                e = Topology.SetDictionary(e, d)
            
            elif entity_type == 'MESH':
                vertices = [list(v) for v in entity.vertices]
                faces = [list(face) for face in entity.faces]
                converted_entity = Topology.SelfMerge(Topology.ByGeometry(vertices=vertices, faces=faces))
                # Try Cell
                temp = Cell.ByFaces(Topology.Faces(converted_entity), silent=True)
                if not Topology.IsInstance(temp, "Cell"):
                    temp = CellComplex.ByFaces(Topology.Faces(converted_entity))
                    if not Topology.IsInstance(temp, "CellComplex"):
                        temp = Shell.ByFaces(Topology.Faces(converted_entity))
                        if not Topology.IsInstance(temp, "Shell"):
                            temp = converted_entity
                e = temp
                e = Topology.SetDictionary(e, d)
            return e

        def convert_insert(entity, file, sides=16):
            block_name = entity.dxf.name
            block = file.blocks.get(block_name)
            converted_entities = []

            for block_entity in block:
                converted_entity = convert_entity(block_entity, sides=sides)
                if converted_entity is not None:
                    converted_entities.append(converted_entity)

            x, y, z = [entity.dxf.insert.x, entity.dxf.insert.y, entity.dxf.insert.z]
            return [Topology.Translate(obj, x, y, z) for obj in converted_entities]

        def convert_dxf_to_custom_types(file):
            # Read the DXF file
            msp = file.modelspace()

            # Store the converted entities
            converted_entities = []

            # Process each entity in the model space
            for entity in msp:
                entity_type = entity.dxftype()
                if entity_type in ['TEXT', 'MTEXT']:
                    continue  # Ignore TEXT and MTEXT

                if entity_type == 'INSERT':
                    converted_entities.extend(convert_insert(entity, file, sides=sides))
                else:
                    converted_entity = convert_entity(entity, file, sides=sides)
                    if converted_entity is not None:
                        converted_entities.append(converted_entity)

            return converted_entities
        converted_entities = convert_dxf_to_custom_types(file)
        return converted_entities

    @staticmethod
    def ByDXFPath(path, sides: int = 16):
        """
        Imports a list of topologies from a DXF file path.
        This is an experimental method with limited capabilities.

        Parameters
        ----------
        path : str
            The path to the DXF file.
        sides : int , optional
            The desired number of sides of splines. The default is 16.

        Returns
        -------
        list
            The list of imported topologies.

        """
        try:
            import ezdxf
        except:
            print("Topology.ByDXFPath - Information: Installing required ezdxf library.")
            try:
                os.system("pip install ezdxf")
            except:
                os.system("pip install ezdxf --user")
            try:
                import ezdxf
                print("Topology.ByDXFPath - Information: ezdxf library installed successfully.")
            except:
                warnings.warn("Topology.ByDXFPath - Error: Could not import ezdxf library. Please install it manually. Returning None.")
                return None
        if not path:
            print("Topology.ByDXFPath - Error: the input path parameter is not a valid path. Returning None.")
            return None
        try:
            file = ezdxf.readfile(path)
        except:
            file = None
        if not file:
            print("Topology.ByDXFPath - Error: the input file parameter is not a valid file. Returning None.")
            return None
        return Topology.ByDXFFile(file, sides=sides)

    @staticmethod
    def ByIFCFile(file, includeTypes=[], excludeTypes=[], transferDictionaries=False, removeCoplanarFaces=False, epsilon=0.0001, tolerance=0.0001):
        """
        Create a list of topologies by importing them from an IFC file.

        Parameters
        ----------
        file : file object
            The input IFC file.
        includeTypes : list , optional
            The list of IFC object types to include. It is case insensitive. If set to an empty list, all types are included. The default is [].
        excludeTypes : list , optional
            The list of IFC object types to exclude. It is case insensitive. If set to an empty list, no types are excluded. The default is [].
        transferDictionaries : bool , optional
            If set to True, the dictionaries from the IFC file will be transferred to the topology. Otherwise, they won't. The default is False.
        removeCoplanarFaces : bool , optional
            If set to True, coplanar faces are removed. Otherwise they are not. The default is False.
        epsilon : float , optional
                The desired epsilon (another form of tolerance) for finding if two faces are coplanar. The default is 0.0001.
            tolerance : float , optional
                The desired tolerance. The default is 0.0001.
        Returns
        -------
        list
            The created list of topologies.
        
        """

        import ifcopenshell, ifcopenshell.geom
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        import numpy as np

        def get_psets(entity):
            # Initialize the PSET dictionary for this entity
            psets = {}
            
            # Check if the entity has a GlobalId
            if not hasattr(entity, 'GlobalId'):
                raise ValueError("The provided entity does not have a GlobalId.")
            
            # Get the property sets related to this entity
            for definition in entity.IsDefinedBy:
                if definition.is_a('IfcRelDefinesByProperties'):
                    property_set = definition.RelatingPropertyDefinition
                    
                    # Check if it is a property set
                    if not property_set == None:
                        if property_set.is_a('IfcPropertySet'):
                            pset_name = "IFC_"+property_set.Name
                            
                            # Dictionary to hold individual properties
                            properties = {}
                            
                            # Iterate over the properties in the PSET
                            for prop in property_set.HasProperties:
                                if prop.is_a('IfcPropertySingleValue'):
                                    # Get the property name and value
                                    prop_name = "IFC_"+prop.Name
                                    prop_value = prop.NominalValue.wrappedValue if prop.NominalValue else None
                                    properties[prop_name] = prop_value
                            
                            # Add this PSET to the dictionary for this entity
                            psets[pset_name] = properties
            return psets
        
        def get_color_transparency_material(entity):
            import random

            # Set default Material Name and ID
            material_list = []
            # Set default transparency based on entity type or material
            default_transparency = 0.0
            
            # Check if the entity is an opening or made of glass
            is_a = entity.is_a().lower()
            if "opening" in is_a or "window" in is_a or "door" in is_a or "space" in is_a:
                default_transparency = 0.7
            elif "space" in is_a:
                default_transparency = 0.8
            
            # Check if the entity has constituent materials (e.g., glass)
            else:
                # Check for associated materials (ConstituentMaterial or direct material assignment)
                materials_checked = False
                if hasattr(entity, 'HasAssociations'):
                    for rel in entity.HasAssociations:
                        if rel.is_a('IfcRelAssociatesMaterial'):
                            material = rel.RelatingMaterial
                            if material.is_a('IfcMaterial') and 'glass' in material.Name.lower():
                                default_transparency = 0.5
                                materials_checked = True
                            elif material.is_a('IfcMaterialLayerSetUsage'):
                                material_layers = material.ForLayerSet.MaterialLayers
                                for layer in material_layers:
                                    material_list.append(layer.Material.Name)
                                    if 'glass' in layer.Material.Name.lower():
                                        default_transparency = 0.5
                                        materials_checked = True
                                        
                # Check for ConstituentMaterial if available
                if hasattr(entity, 'HasAssociations') and not materials_checked:
                    for rel in entity.HasAssociations:
                        if rel.is_a('IfcRelAssociatesMaterial'):
                            material = rel.RelatingMaterial
                            if material.is_a('IfcMaterialConstituentSet'):
                                for constituent in material.MaterialConstituents:
                                    material_list.append(constituent.Material.Name)
                                    if 'glass' in constituent.Material.Name.lower():
                                        default_transparency = 0.5
                                        materials_checked = True

                # Check if the entity has ShapeAspects with associated materials or styles
                if hasattr(entity, 'HasShapeAspects') and not materials_checked:
                    for shape_aspect in entity.HasShapeAspects:
                        if hasattr(shape_aspect, 'StyledByItem') and shape_aspect.StyledByItem:
                            for styled_item in shape_aspect.StyledByItem:
                                for style in styled_item.Styles:
                                    if style.is_a('IfcSurfaceStyle'):
                                        for surface_style in style.Styles:
                                            if surface_style.is_a('IfcSurfaceStyleRendering'):
                                                transparency = getattr(surface_style, 'Transparency', default_transparency)
                                                if transparency > 0:
                                                    default_transparency = transparency

            # Try to get the actual color and transparency if defined
            if hasattr(entity, 'Representation') and entity.Representation:
                for rep in entity.Representation.Representations:
                    for item in rep.Items:
                        if hasattr(item, 'StyledByItem') and item.StyledByItem:
                            for styled_item in item.StyledByItem:
                                if hasattr(styled_item, 'Styles'):
                                    for style in styled_item.Styles:
                                        if style.is_a('IfcSurfaceStyle'):
                                            for surface_style in style.Styles:
                                                if surface_style.is_a('IfcSurfaceStyleRendering'):
                                                    color = surface_style.SurfaceColour
                                                    transparency = getattr(surface_style, 'Transparency', default_transparency)
                                                    return (color.Red*255, color.Green*255, color.Blue*255), transparency, material_list
            
            # If no color is defined, return a consistent random color based on the entity type
            if "wall" in is_a:
                color = (175, 175, 175)
            elif "slab" in is_a:
                color = (200, 200, 200)
            elif "space" in is_a:
                color = (250, 250, 250)
            else:
                random.seed(hash(is_a))
                color = (random.random(), random.random(), random.random())
            
            return color, default_transparency, material_list
        # Create a 4x4 unity matrix
        matrix = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
        def convert_to_topology(entity, settings, transferDictionaries=False):    
            if hasattr(entity, "Representation") and entity.Representation:
                for rep in entity.Representation.Representations:
                    if rep.is_a("IfcShapeRepresentation"):
                        # Generate the geometry for this entity
                        shape = ifcopenshell.geom.create_shape(settings, entity)
                        try:
                            trans = shape.transformation
                            # Convert into a 4x4 matrix
                            matrix = [trans.matrix[i:i+4] for i in range(0, len(trans.matrix), 4)]
                        except:
                            pass
                        # Get grouped vertices and grouped faces     
                        grouped_verts = shape.geometry.verts
                        verts = [ [grouped_verts[i], grouped_verts[i + 1], grouped_verts[i + 2]] for i in range(0, len(grouped_verts), 3)]
                        grouped_edges = shape.geometry.edges
                        edges = [[grouped_edges[i], grouped_edges[i + 1]] for i in range(0, len(grouped_edges), 2)]
                        grouped_faces = shape.geometry.faces
                        faces = [ [grouped_faces[i], grouped_faces[i + 1], grouped_faces[i + 2]] for i in range(0, len(grouped_faces), 3)]
                        #shape_topology = ifc_to_topologic_geometry(verts, edges, faces)
                        #shape_topology = Topology.SelfMerge(Topology.ByGeometry(verts, edges, faces))
                        shape_topology = Topology.ByGeometry(verts, edges, faces, silent=True)
                        if not shape_topology == None:
                            if removeCoplanarFaces == True:
                                shape_topology = Topology.RemoveCoplanarFaces(shape_topology, epsilon=0.0001)
                            shape_topology = Topology.Transform(shape_topology, matrix)

                            # Store relevant information
                            if transferDictionaries == True:
                                color, transparency, material_list = get_color_transparency_material(entity)
                                entity_dict = {
                                    "TOPOLOGIC_id": str(Topology.UUID(shape_topology)),
                                    "TOPOLOGIC_name": getattr(entity, 'Name', "Untitled"),
                                    "TOPOLOGIC_type": Topology.TypeAsString(shape_topology),
                                    "TOPOLOGIC_color": color,
                                    "TOPOLOGIC_opacity": 1.0 - transparency,
                                    "IFC_global_id": getattr(entity, 'GlobalId', 0),
                                    "IFC_name": getattr(entity, 'Name', "Untitled"),
                                    "IFC_type": entity.is_a(),
                                    "IFC_material_list": material_list,
                                }
                                topology_dict = Dictionary.ByPythonDictionary(entity_dict)
                                # Get PSETs dictionary
                                pset_python_dict = get_psets(entity)
                                pset_dict = Dictionary.ByPythonDictionary(pset_python_dict)
                                topology_dict = Dictionary.ByMergedDictionaries([topology_dict, pset_dict])
                                shape_topology = Topology.SetDictionary(shape_topology, topology_dict)
                        return shape_topology
            return None

        # Main Code
        topologies = []
        settings = ifcopenshell.geom.settings()
        #settings.set("dimensionality", ifcopenshell.ifcopenshell_wrapper.SOLID)
        settings.set(settings.USE_WORLD_COORDS, True)
        products = file.by_type("IfcProduct")
        entities = []
        for product in products:
            is_a = product.is_a()
            if (is_a in includeTypes or len(includeTypes) == 0) and (not is_a in excludeTypes):
                entities.append(product)
        topologies = []
        for entity in entities:
            topologies.append(convert_to_topology(entity, settings, transferDictionaries=transferDictionaries))
        return topologies

    @staticmethod
    def _ByIFCFile_old(file, includeTypes=[], excludeTypes=[], transferDictionaries=False, removeCoplanarFaces=False):
        """
        Create a topology by importing it from an IFC file.

        Parameters
        ----------
        file : file object
            The input IFC file.
        includeTypes : list , optional
            The list of IFC object types to include. It is case insensitive. If set to an empty list, all types are included. The default is [].
        excludeTypes : list , optional
            The list of IFC object types to exclude. It is case insensitive. If set to an empty list, no types are excluded. The default is [].
        transferDictionaries : bool , optional
            If set to True, the dictionaries from the IFC file will be transferred to the topology. Otherwise, they won't. The default is False.
        removeCoplanarFaces : bool , optional
            If set to True, coplanar faces are removed. Otherwise they are not. The default is False.
        Returns
        -------
        list
            The created list of topologies.
        
        """
        import multiprocessing
        from topologicpy.Cluster import Cluster
        from topologicpy.Dictionary import Dictionary
        import uuid
        import random
        import hashlib
        import re
        import numpy as np

        try:
            import ifcopenshell
            import ifcopenshell.geom
        except:
            print("Topology.ByIFCFile - Warning: Installing required ifcopenshell library.")
            try:
                os.system("pip install ifcopenshell")
            except:
                os.system("pip install ifcopenshell --user")
            try:
                import ifcopenshell
                import ifcopenshell.geom
                print("Topology.ByIFCFile - Warning: ifcopenshell library installed correctly.")
            except:
                warnings.warn("Topology.ByIFCFile - Error: Could not import ifcopenshell. Please try to install ifcopenshell manually. Returning None.")
                return None
        if not file:
            print("Topology.ByIFCFile - Error: the input file parameter is not a valid file. Returning None.")
            return None
        
        def clean_key(string):
            # Replace any character that is not a letter, digit, or underscore with an underscore
            cleaned_string = re.sub(r'[^a-zA-Z0-9_]', '_', string)
            return cleaned_string
        
        def transform_wall_vertices(wall):

            # Relatives Placement abrufen und ausgeben
            if wall.ObjectPlacement and wall.ObjectPlacement.RelativePlacement:
                relative_placement = wall.ObjectPlacement.RelativePlacement
                if relative_placement.is_a('IFCAXIS2PLACEMENT3D'):
                    location = relative_placement.Location
                    ref_direction = relative_placement.RefDirection
                    print("Relative Placement Location:", location.Coordinates)
                    if ref_direction:
                        print("Relative Placement RefDirection:", ref_direction.DirectionRatios)
                    else:
                        print("Relative Placement RefDirection: None")

            # IFCPRODUCTDEFINITIONSHAPE der Wand abrufen
            product_definition_shape = wall.Representation
            if not product_definition_shape:
                print("Keine Repräsentation gefunden.")
                return

            # Initialisieren von Variablen für Representation Type und Layer-Infos
            representation_type = None
            diverse_representation = False
            layer_details = []

            if hasattr(product_definition_shape, 'HasShapeAspects'):
                for aspect in product_definition_shape.HasShapeAspects:
                    for representation in aspect.ShapeRepresentations:
                        if representation.is_a('IFCSHAPEREPRESENTATION'):
                            for item in representation.Items:
                                if item.is_a('IFCEXTRUDEDAREASOLID'):
                                    # Profilbeschreibung abrufen
                                    profile = item.SweptArea
                                    if profile.is_a('IFCARBITRARYCLOSEDPROFILEDEF'):
                                        if not representation_type:
                                            representation_type = "ArbitraryClosedProfil"
                                        elif representation_type != "ArbitraryClosedProfil":
                                            diverse_representation = True

                                        # Profilpunkte abrufen
                                        if hasattr(profile, 'OuterCurve') and profile.OuterCurve.is_a('IFCINDEXEDPOLYCURVE'):
                                            indexed_polycurve = profile.OuterCurve
                                            if hasattr(indexed_polycurve, 'Points') and indexed_polycurve.Points.is_a('IFCCARTESIANPOINTLIST2D'):
                                                point_list_2d = indexed_polycurve.Points
                                                points = point_list_2d.CoordList
                                                layer_info["Profilpunkte"] = points
                                    else:
                                        diverse_representation = True

                                    # Location und RefDirection abrufen
                                    if item.Position.is_a('IFCAXIS2PLACEMENT3D'):
                                        axis_placement = item.Position
                                        location = axis_placement.Location
                                        ref_direction = axis_placement.RefDirection
                                        layer_info["Location"] = location.Coordinates
                                        if ref_direction:
                                            layer_info["RefDirection"] = ref_direction.DirectionRatios
                                        else:
                                            layer_info["RefDirection"] = None

                    layer_details.append(layer_info)

            # Representation Type ausgeben
            if diverse_representation:
                representation_type = "divers"
            print("Representation Type der Wand:", representation_type)

            # Layer-Details ausgeben
            for index, layer in enumerate(layer_details):
                print(f"\nLayer {index + 1} Details:")
                print("Material:", layer.get("Material", "Nicht verfügbar"))
                print("Extrusionsstärke:", layer.get("Extrusionsstärke", "Nicht verfügbar"))
                print("Profilpunkte:", layer.get("Profilpunkte", "Nicht verfügbar"))
                print("Location:", layer.get("Location", "Nicht verfügbar"))
                print("RefDirection:", layer.get("RefDirection", "Nicht verfügbar"))






        def extract_matrix_from_placement(placement):
            """Constructs a transformation matrix from an IFC Local Placement."""
            # Initialize identity matrix
            matrix = np.identity(4)

            # Check if the placement is IfcLocalPlacement
            if placement.is_a("IfcLocalPlacement"):
                relative_placement = placement.RelativePlacement

                if relative_placement.is_a("IfcAxis2Placement3D"):
                    location = relative_placement.Location.Coordinates
                    z_dir = relative_placement.Axis.DirectionRatios if relative_placement.Axis else [0, 0, 1]
                    x_dir = relative_placement.RefDirection.DirectionRatios if relative_placement.RefDirection else [1, 0, 0]
                    
                    # Compute y direction (cross product of z and x)
                    y_dir = np.cross(z_dir, x_dir)
                    
                    # Construct the rotation matrix
                    rotation_matrix = np.array([
                        [x_dir[0], y_dir[0], z_dir[0], 0],
                        [x_dir[1], y_dir[1], z_dir[1], 0],
                        [x_dir[2], y_dir[2], z_dir[2], 0],
                        [0, 0, 0, 1]
                    ])

                    # Translation vector
                    translation_vector = np.array([
                        [1, 0, 0, location[0]],
                        [0, 1, 0, location[1]],
                        [0, 0, 1, location[2]],
                        [0, 0, 0, 1]
                    ])

                    # Combine the rotation matrix and the translation vector
                    matrix = np.dot(translation_vector, rotation_matrix)

            return matrix

        def apply_transformation(verts, matrix):
            """Applies a 4x4 transformation matrix to a list of vertices."""
            transformed_verts = []
            for vert in verts:
                print("vert:", vert)
                v = np.array([vert[0], vert[1], vert[2], 1.0])
                transformed_v = np.dot(matrix, v)
                transformed_verts.append([transformed_v[0], transformed_v[1], transformed_v[2]])
            return transformed_verts
        
        def get_entity_transformation_matrix(entity):
            """Extracts the transformation matrix from an IFC entity."""
            matrix = np.identity(4)  # Default to an identity matrix
            if hasattr(entity, "ObjectPlacement") and entity.ObjectPlacement:
                placement = entity.ObjectPlacement
                matrix = extract_matrix_from_placement(placement)
            return matrix
    
        # Function to generate a unique random color based on material ID
        def generate_color_for_material(material_id):
            # Use a hash function to get a consistent "random" seed
            hash_object = hashlib.sha1(material_id.encode())
            seed = int(hash_object.hexdigest(), 16) % (10 ** 8)
            random.seed(seed)
            # Generate a random color
            r = random.random()
            g = random.random()
            b = random.random()
            return [r, g, b]
        
        # Function to get the material IDs associated with an entity
        def get_material_ids_of_entity(entity):
            return_dict = {}
            material_names = []
            material_ids = []
            if hasattr(entity, "HasAssociations"):
                for association in entity.HasAssociations:
                    if association.is_a("IfcRelAssociatesMaterial"):
                        material = association.RelatingMaterial
                        try:
                            material_name = material.Name
                        except:
                            material_name = material.to_string()
                        if material.is_a("IfcMaterial"):
                            material_ids.append(material.id())
                            material_names.append(material_name)
                            return_dict[clean_key(material_name)] = material.id
                        elif material.is_a("IfcMaterialList"):
                            for mat in material.Materials:
                                material_ids.append(mat.id())
                                try:
                                    material_name = mat.Name
                                except:
                                    material_name = mat.to_string()
                                material_names.append(material_name)
                                return_dict[clean_key(material_name)] = mat.id
                        elif material.is_a("IfcMaterialLayerSetUsage") or material.is_a("IfcMaterialLayerSet"):
                            for layer in material.ForLayerSet.MaterialLayers:
                                material_ids.append(layer.Material.id())
                                try:
                                    material_name = layer.Name
                                except:
                                    material_name = layer.to_string()
                                material_names.append(material_name)
                                return_dict[clean_key(material_name)] = layer.Material.id()
                        elif material.is_a("IfcMaterialConstituentSet"):
                            for constituent in material.MaterialConstituents:
                                material_ids.append(constituent.Material.id())
                                try:
                                    material_name = constituent.Material.Name
                                except:
                                    material_name = constituent.Material.to_string()
                                material_names.append(material_name)
                                return_dict[clean_key(material_name)] = constituent.Material.id()
            
            return return_dict
        
        def get_wall_layers(wall, matrix=None, transferDictionaries=False):
            settings = ifcopenshell.geom.settings()
            settings.set("dimensionality", ifcopenshell.ifcopenshell_wrapper.CURVES_SURFACES_AND_SOLIDS)
            settings.set(settings.USE_WORLD_COORDS, False)

            # IFCPRODUCTDEFINITIONSHAPE der Wand abrufen
            product_definition_shape = wall.Representation
            if not product_definition_shape:
                print("Topology.ByIFCFile - Error: The object has no representation. Returning None")
                return None

            if hasattr(product_definition_shape, 'HasShapeAspects'):
                for aspect in product_definition_shape.HasShapeAspects:
                    material_name = aspect.Name
                    for representation in aspect.ShapeRepresentations:
                        print(dir(representation))
                        axis_placement = representation.Position
                        location = axis_placement.Location
                        ref_direction = axis_placement.RefDirection
                        print("Location:", location)
                        print("Direction", ref_direction)
                        aspect_matrix = get_entity_transformation_matrix(representation)
                        print("Aspect Matrix:", aspect_matrix)
                        shape = ifcopenshell.geom.create_shape(settings, representation)
                        verts = shape.verts
                        edges = shape.edges
                        faces = shape.faces
                        grouped_verts = [ [verts[i], verts[i + 1], verts[i + 2]] for i in range(0, len(verts), 3)]
                        grouped_verts = apply_transformation(grouped_verts, aspect_matrix)
                        grouped_edges = [[edges[i], edges[i + 1]] for i in range(0, len(edges), 2)]
                        grouped_faces = [ [faces[i], faces[i + 1], faces[i + 2]] for i in range(0, len(faces), 3)]
                        topology = Topology.SelfMerge(Topology.ByGeometry(grouped_verts, grouped_edges, grouped_faces))
                        #matrix = shape.transformation.matrix
                        #topology = Topology.Transform(topology, matrix)
                        d = get_material_ids_of_entity(wall)
                        material_id = d.get(clean_key(material_name), 0)
                        if transferDictionaries:
                            keys = []
                            values = []
                            try:
                                entity_name = entity.Name
                            except:
                                entity_name = entity.to_str()
                            keys.append("TOPOLOGIC_id")
                            values.append(str(uuid.uuid4()))
                            keys.append("TOPOLOGIC_name")
                            values.append(entity_name)
                            keys.append("TOPOLOGIC_type")
                            values.append(Topology.TypeAsString(topology))
                            keys.append("IFC_id")
                            values.append(str(aspect.id))
                            #keys.append("IFC_guid")
                            #values.append(str(aspect.guid))
                            #keys.append("IFC_unique_id")
                            #values.append(str(aspect.unique_id))
                            keys.append("IFC_name")
                            values.append(entity_name)
                            #keys.append("IFC_type")
                            #values.append(aspect.type)
                            keys.append("IFC_material_id")
                            values.append(material_id)
                            keys.append("IFC_material_name")
                            values.append(material_name)
                            keys.append("TOPOLOGIC_color")
                            color = generate_color_for_material(str(material_id))
                            values.append(color)
                            d = Dictionary.ByKeysValues(keys, values)
                            topology = Topology.SetDictionary(topology, d)
                
            return topology
        
        
        includeTypes = [s.lower() for s in includeTypes]
        excludeTypes = [s.lower() for s in excludeTypes]
        topologies = []
        settings = ifcopenshell.geom.settings()
        settings.set("dimensionality", ifcopenshell.ifcopenshell_wrapper.SOLID)
        settings.set(settings.USE_WORLD_COORDS, True)
        for entity in file.by_type('IfcProduct'):  # You might want to refine the types you check
                if hasattr(entity, "Representation") and entity.Representation:
                    print("Number of Representations:", len(entity.Representation.Representations))
                    for rep in entity.Representation.Representations:
                        print("Rep:", rep)
                        print(dir(rep))
                        matrix = get_entity_transformation_matrix(entity)
                        print(matrix)
                        if rep.is_a("IfcShapeRepresentation"):
                            # Generate the geometry for this entity
                            shape = ifcopenshell.geom.create_shape(settings, rep)
                            verts = shape.verts
                            edges = shape.edges
                            faces = shape.faces
                            grouped_verts = [ [verts[i], verts[i + 1], verts[i + 2]] for i in range(0, len(verts), 3)]
                            #grouped_verts = apply_transformation(grouped_verts, matrix)
                            grouped_edges = [[edges[i], edges[i + 1]] for i in range(0, len(edges), 2)]
                            grouped_faces = [ [faces[i], faces[i + 1], faces[i + 2]] for i in range(0, len(faces), 3)]
                            topology = Topology.SelfMerge(Topology.ByGeometry(grouped_verts, grouped_edges, grouped_faces))
                            if removeCoplanarFaces:
                                topology = Topology.RemoveCoplanarFaces(topology)
                            if transferDictionaries:
                                keys = []
                                values = []
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
                                material_dict = get_material_ids_of_entity(entity)
                                keys.append("IFC_materials")
                                values.append(material_dict)
                                #keys.append("IFC_material_name")
                                #values.append(material_name)
                                #keys.append("TOPOLOGIC_color")
                                #color = generate_color_for_material(str(material_ids))
                                #values.append(color)
                                d = Dictionary.ByKeysValues(keys, values)
                                topology = Topology.SetDictionary(topology, d)
                                topology = Topology.Transform(topology, matrix)
                            topologies.append(topology)
        return topologies

    @staticmethod
    def ByIFCPath(path, includeTypes=[], excludeTypes=[], transferDictionaries=False, removeCoplanarFaces=False, epsilon=0.0001, tolerance=0.0001):
        """
        Create a topology by importing it from an IFC file path.

        Parameters
        ----------
        path : str
            The path to the IFC file.
        includeTypes : list , optional
            The list of IFC object types to include. It is case insensitive. If set to an empty list, all types are included. The default is [].
        excludeTypes : list , optional
            The list of IFC object types to exclude. It is case insensitive. If set to an empty list, no types are excluded. The default is [].
        transferDictionaries : bool , optional
            If set to True, the dictionaries from the IFC file will be transferred to the topology. Otherwise, they won't. The default is False.
        removeCoplanarFaces : bool , optional
            If set to True, coplanar faces are removed. Otherwise they are not. The default is False.
        epsilon : float , optional
            The desired epsilon (another form of tolerance) for finding if two faces are coplanar. The default is 0.0001.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        Returns
        -------
        list
            The created list of topologies.
        
        """
        import ifcopenshell

        if not path:
            print("Topology.ByIFCPath - Error: the input path parameter is not a valid path. Returning None.")
            return None
        try:
            file = ifcopenshell.open(path)
        except:
            file = None
        if not file:
            print("Topology.ByIFCPath - Error: the input file parameter is not a valid file. Returning None.")
            return None
        return Topology.ByIFCFile(file, includeTypes=includeTypes, excludeTypes=excludeTypes, transferDictionaries=transferDictionaries, removeCoplanarFaces=removeCoplanarFaces, epsilon=epsilon, tolerance=tolerance)
    
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
            The list of imported topologies (Warning: the list could contain 0, 1, or many topologies, but this method will always return a list)

        """
        if not file:
            print("Topology.ByJSONFile - Error: the input file parameter is not a valid file. Returning None.")
            return None
        json_dict = json.load(file)
        return Topology.ByJSONDictionary(json_dict, tolerance=tolerance)
    
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
        import json
        if not path:
            print("Topology.ByJSONPath - Error: the input path parameter is not a valid path. Returning None.")
            return None
        with open(path) as file:
            json_dict = json.load(file)
        entities = Topology.ByJSONDictionary(json_dict, tolerance=tolerance)
        return entities
    
    @staticmethod
    def ByJSONDictionary(jsonDictionary, tolerance=0.0001):
        """
        Imports the topology from a JSON dictionary.

        Parameters
        ----------
        jsonDictionary : dict
            The input JSON dictionary.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        list
            The list of imported topologies (Warning: the list could contain 0, 1, or many topologies, but this method will always return a list)

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Shell import Shell
        from topologicpy.Cell import Cell
        from topologicpy.CellComplex import CellComplex
        from topologicpy.Dictionary import Dictionary

        # Containers for created entities
        vertices = {}
        edges = {}
        wires = {}
        faces = {}
        shells = {}
        cells = {}
        cell_complexes = {}

        vertex_apertures = []
        edge_apertures = []
        face_apertures = []
        for entity in jsonDictionary:
            entity_type = entity['type']
            entity_dict = Dictionary.ByKeysValues(keys=list(entity['dictionary'].keys()),
                                                values=list(entity['dictionary'].values()))

            parent_entity = None

            # Create basic topological entities
            if entity_type == 'Vertex':
                parent_entity = Vertex.ByCoordinates(*entity['coordinates'])
                parent_entity = Topology.SetDictionary(parent_entity, entity_dict)
                vertices[entity['uuid']] = parent_entity

            elif entity_type == 'Edge':
                vertex1 = vertices[entity['vertices'][0]]
                vertex2 = vertices[entity['vertices'][1]]
                parent_entity = Edge.ByVertices([vertex1, vertex2])
                parent_entity = Topology.SetDictionary(parent_entity, entity_dict)
                edges[entity['uuid']] = parent_entity

            elif entity_type == 'Wire':
                wire_edges = [edges[uuid] for uuid in entity['edges']]
                parent_entity = Wire.ByEdges(wire_edges)
                parent_entity = Topology.SetDictionary(parent_entity, entity_dict)
                wires[entity['uuid']] = parent_entity

            elif entity_type == 'Face':
                face_wires = [wires[uuid] for uuid in entity['wires']]
                if len(face_wires) > 1:
                    parent_entity = Face.ByWires(face_wires[0], face_wires[1:])
                else:
                    parent_entity = Face.ByWire(face_wires[0])
                parent_entity = Topology.SetDictionary(parent_entity, entity_dict)
                faces[entity['uuid']] = parent_entity

            elif entity_type == 'Shell':
                shell_faces = [faces[uuid] for uuid in entity['faces']]
                parent_entity = Shell.ByFaces(shell_faces)
                parent_entity = Topology.SetDictionary(parent_entity, entity_dict)
                shells[entity['uuid']] = parent_entity

            elif entity_type == 'Cell':
                cell_shells = [shells[uuid] for uuid in entity['shells']]
                if len(cell_shells) > 1:
                    parent_entity = Cell.ByShells(cell_shells[0], cell_shells[1:])
                else:
                    parent_entity = Cell.ByShell(cell_shells[0])
                parent_entity = Topology.SetDictionary(parent_entity, entity_dict)
                cells[entity['uuid']] = parent_entity

            elif entity_type == 'CellComplex':
                complex_cells = [cells[uuid] for uuid in entity['cells']]
                parent_entity = CellComplex.ByCells(complex_cells)
                parent_entity = Topology.SetDictionary(parent_entity, entity_dict)
                cell_complexes[entity['uuid']] = parent_entity

            # Step 3: Handle apertures within each entity
            if 'apertures' in entity:
                # Containers for created entities
                ap_vertices = {}
                ap_edges = {}
                ap_wires = {}
                ap_faces = {}
                for aperture_list in entity['apertures']:
                    types = [aperture_data['type'] for aperture_data in aperture_list]
                    save_vertex = False
                    save_edge = False
                    save_wire = False
                    save_face = False

                    if 'Face' in types:
                        save_face = True
                    elif 'Wire' in types:
                        save_wire = True
                    elif 'Edge' in types:
                        save_edge = True
                    elif 'Vertex' in types:
                        save_vertex = True

                    apertures = []
                    for aperture_data in aperture_list:
                        aperture_type = aperture_data['type']
                        aperture_dict = Dictionary.ByKeysValues(keys=list(aperture_data['dictionary'].keys()),
                                                                values=list(aperture_data['dictionary'].values()))
                        
                        if aperture_type == 'Vertex':
                            aperture_entity = Vertex.ByCoordinates(*aperture_data['coordinates'])
                            aperture_entity = Topology.SetDictionary(aperture_entity, aperture_dict)
                            ap_vertices[aperture_data['uuid']] = aperture_entity
                            if save_vertex == True:
                                apertures.append(aperture_entity)
                        
                        elif aperture_type == 'Edge':
                            vertex1 = ap_vertices[aperture_data['vertices'][0]]
                            vertex2 = ap_vertices[aperture_data['vertices'][1]]
                            aperture_entity = Edge.ByVertices([vertex1, vertex2])
                            aperture_entity = Topology.SetDictionary(aperture_entity, aperture_dict)
                            ap_edges[aperture_data['uuid']] = aperture_entity
                            if save_edge == True:
                                apertures.append(aperture_entity)

                        elif aperture_type == 'Wire':
                            wire_edges = [ap_edges[uuid] for uuid in aperture_data['edges']]
                            aperture_entity = Wire.ByEdges(wire_edges)
                            aperture_entity = Topology.SetDictionary(aperture_entity, aperture_dict)
                            ap_wires[aperture_data['uuid']] = aperture_entity
                            if save_wire == True:
                                apertures.append(aperture_entity)

                        elif aperture_type == 'Face':
                            face_wires = [ap_wires[uuid] for uuid in aperture_data['wires']]
                            if len(face_wires) > 1:
                                aperture_entity = Face.ByWires(face_wires[0], face_wires[1:])
                            else:
                                aperture_entity = Face.ByWire(face_wires[0])
                            aperture_entity = Topology.SetDictionary(aperture_entity, aperture_dict)
                            ap_faces[aperture_data['uuid']] = aperture_entity
                            if save_face == True:
                                apertures.append(aperture_entity)
                            
                    # Assign the built apertures to the parent entity
                    if len(apertures) > 0:
                        if entity_type == "Face":
                            face_apertures += apertures
                        elif entity_type == 'Edge':
                            edge_apertures += apertures
                        elif entity_type == 'Vertex':
                            vertex_apertures += apertures

                # Update the parent entity in its respective container
                if entity_type == 'Vertex':
                    vertices[entity['uuid']] = parent_entity
                elif entity_type == 'Edge':
                    edges[entity['uuid']] = parent_entity
                elif entity_type == 'Wire':
                    wires[entity['uuid']] = parent_entity
                elif entity_type == 'Face':
                    faces[entity['uuid']] = parent_entity
                elif entity_type == 'Shell':
                    shells[entity['uuid']] = parent_entity
                elif entity_type == 'Cell':
                    cells[entity['uuid']] = parent_entity
                elif entity_type == 'CellComplex':
                    cell_complexes[entity['uuid']] = parent_entity
            
            d = Topology.Dictionary(parent_entity)
            top_level = Dictionary.ValueAtKey(d, "toplevel")
        tp_vertices = list(vertices.values())
        tp_edges = list(edges.values())
        tp_wires = list(wires.values())
        tp_faces = list(faces.values())
        tp_shells = list(shells.values())
        tp_cells = list(cells.values())
        tp_cell_complexes = list(cell_complexes.values())
        everything = tp_vertices + tp_edges + tp_wires + tp_faces + tp_shells + tp_cells + tp_cell_complexes
        top_level_list = []
        for entity in everything:
            d = Topology.Dictionary(entity)
            top_level = Dictionary.ValueAtKey(d, "toplevel")
            if top_level == 1:
                if len(face_apertures) > 0:
                    entity = Topology.AddApertures(entity, face_apertures, subTopologyType="Face", tolerance=tolerance)
                if len(edge_apertures) > 0:
                    entity = Topology.AddApertures(entity, edge_apertures, subTopologyType="Edge", tolerance=0.001)
                if len(vertex_apertures) > 0:
                    entity = Topology.AddApertures(entity, vertex_apertures, subTopologyType="Vertex", tolerance=0.001)
                top_level_list.append(entity)
        return top_level_list
    
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
        list
            The list of imported topologies (Warning: the list could contain 0, 1, or many topologies, but this method will always return a list)

        """

        json_dict = json.loads(string)
        return Topology.ByJSONDictionary(json_dict, tolerance=tolerance)

    @staticmethod
    def ByOBJFile(objFile, mtlFile = None,
                defaultColor: list = [255,255,255],
                defaultOpacity: float = 1.0,
                transposeAxes: bool = True,
                removeCoplanarFaces: bool = False,
                selfMerge: bool = True,
                mantissa : int = 6,
                tolerance: float = 0.0001):
        """
        Imports a topology from an OBJ file and an associated materials file.
        This method is basic and does not support textures and vertex normals.

        Parameters
        ----------
        objFile : file object
            The OBJ file.
        mtlFile : file object , optional
            The MTL file. The default is None.
        defaultColor : list , optional
            The default color to use if none is specified in the file. The default is [255, 255, 255] (white).
        defaultOpacity : float , optional
            The default opacity to use if none is specified in the file. The default is 1.0 (fully opaque).
        transposeAxes : bool , optional
            If set to True the Z and Y axes are transposed. Otherwise, they are not. The default is True.
        removeCoplanarFaces : bool , optional
            If set to True, coplanar faces are merged. The default is True.
        selfMerge : bool , optional
            If set to True, the faces of the imported topologies will each be self-merged to create higher-dimensional objects. Otherwise, they remain a cluster of faces. The default is False.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001

        Returns
        -------
        list
            The imported topologies.

        """
        from os.path import dirname, join, exists


        def find_next_word_after_mtllib(text):
            words = text.split()
            for i, word in enumerate(words):
                if word == 'mtllib' and i + 1 < len(words):
                    return words[i + 1]
            return None

        obj_string = objFile.read()
        mtl_filename = find_next_word_after_mtllib(obj_string)
        mtl_string = None
        if mtlFile:
            mtl_string = mtlFile.read()
        return Topology.ByOBJString(obj_string, mtl_string,
                        defaultColor=defaultColor, defaultOpacity=defaultOpacity,
                        transposeAxes=transposeAxes, removeCoplanarFaces=removeCoplanarFaces,
                        selfMerge=selfMerge,
                        mantissa=mantissa, tolerance=tolerance)

    @staticmethod
    def ByOBJPath(objPath,
                  defaultColor: list = [255,255,255], defaultOpacity: float = 1.0,
                  transposeAxes: bool = True, removeCoplanarFaces: bool = False,
                  selfMerge: bool = False,
                  mantissa : int = 6, tolerance: float = 0.0001):
        """
        Imports a topology from an OBJ file path and an associated materials file.
        This method is basic and does not support textures and vertex normals.

        Parameters
        ----------
        objPath : str
            The path to the OBJ file.
        defaultColor : list , optional
            The default color to use if none is specified in the file. The default is [255, 255, 255] (white).
        defaultOpacity : float , optional
            The default opacity to use if none is specified in the file. The default is 1.0 (fully opaque).
        transposeAxes : bool , optional
            If set to True the Z and Y axes are transposed. Otherwise, they are not. The default is True.
        removeCoplanarFaces : bool , optional
            If set to True, coplanar faces are merged. The default is True.
        selfMerge : bool , optional
            If set to True, the faces of the imported topologies will each be self-merged to create higher-dimensional objects. Otherwise, they remain a cluster of faces. The default is False.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001

        Returns
        -------
        list
            The imported topologies.

        """
        from os.path import dirname, join, exists

        if not objPath:
            print("Topology.ByOBJPath - Error: the input OBJ path parameter is not a valid path. Returning None.")
            return None
        if not exists(objPath):
            print("Topology.ByOBJPath - Error: the input OBJ path does not exist. Returning None.")
            return None

        def find_next_word_after_mtllib(text):
            words = text.split()
            for i, word in enumerate(words):
                if word == 'mtllib' and i + 1 < len(words):
                    return words[i + 1]
            return None

        with open(objPath, 'r') as obj_file:
            obj_string = obj_file.read()
            mtl_filename = find_next_word_after_mtllib(obj_string)
            mtl_string = None
            if mtl_filename:
                parent_folder = dirname(objPath)
                mtl_path = join(parent_folder, mtl_filename)
                if exists(mtl_path):
                    with open(mtl_path, 'r') as mtl_file:
                        mtl_string = mtl_file.read()
        return Topology.ByOBJString(obj_string, mtl_string,
                        defaultColor=defaultColor, defaultOpacity=defaultOpacity,
                        transposeAxes=transposeAxes, removeCoplanarFaces=removeCoplanarFaces,
                        selfMerge = selfMerge,
                        mantissa=mantissa, tolerance=tolerance)

    @staticmethod
    def ByOBJString(objString: str, mtlString: str = None,
                    defaultColor: list = [255,255,255], defaultOpacity: float = 1.0,
                    transposeAxes: bool = True, removeCoplanarFaces: bool = False,
                    selfMerge: bool = False,
                    mantissa = 6, tolerance = 0.0001):
        """
        Imports a topology from  OBJ and MTL strings.

        Parameters
        ----------
        objString : str
            The string of the OBJ file.
        mtlString : str , optional
            The string of the MTL file. The default is None.
        defaultColor : list , optional
            The default color to use if none is specified in the string. The default is [255, 255, 255] (white).
        defaultOpacity : float , optional
            The default opacity to use if none is specified in the string. The default is 1.0 (fully opaque).
        transposeAxes : bool , optional
            If set to True the Z and Y axes are transposed. Otherwise, they are not. The default is True.
        removeCoplanarFaces : bool , optional
            If set to True, coplanar faces are merged. The default is True.
        selfMerge : bool , optional
            If set to True, the faces of the imported topologies will each be self-merged to create higher-dimensional objects. Otherwise, they remain a cluster of faces. The default is False.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001

        Returns
        -------
        list
            The imported topologies.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Shell import Shell
        from topologicpy.Cell import Cell
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Helper import Helper

        def load_materials(mtl_string):
            materials = {}
            if not mtl_string:
                return materials
            current_material = None
            lines = mtlString.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == 'newmtl':
                    current_material = parts[1]
                    materials[current_material] = {}
                elif current_material:
                    if parts[0] == 'Kd':  # Diffuse color
                        materials[current_material]['Kd'] = list(map(float, parts[1:4]))
                    elif parts[0] == 'Ka':  # Ambient color
                        materials[current_material]['Ka'] = list(map(float, parts[1:4]))
                    elif parts[0] == 'Ks':  # Specular color
                        materials[current_material]['Ks'] = list(map(float, parts[1:4]))
                    elif parts[0] == 'Ns':  # Specular exponent
                        materials[current_material]['Ns'] = float(parts[1])
                    elif parts[0] == 'd':  # Transparency
                        materials[current_material]['d'] = float(parts[1])
                    elif parts[0] == 'map_Kd':  # Diffuse texture map
                        materials[current_material]['map_Kd'] = parts[1]
                    # Add more properties as needed
            return materials

        materials = load_materials(mtlString)
        vertices = []
        textures = []
        normals = []
        groups = {}
        current_group = None
        current_material = None
        lines = objString.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                continue

            parts = line.split()
            if not parts:
                continue

            if parts[0] == 'v':
                vertex = list(map(float, parts[1:4]))
                vertex = [round(coord, mantissa) for coord in vertex]
                if transposeAxes == True:
                    vertex = [vertex[0], vertex[2], vertex[1]]
                vertices.append(vertex)
            elif parts[0] == 'vt':
                texture = list(map(float, parts[1:3]))
                textures.append(texture)
            elif parts[0] == 'vn':
                normal = list(map(float, parts[1:4]))
                normals.append(normal)
            elif parts[0] == 'f':
                face = []
                for part in parts[1:]:
                    indices = part.split('/')
                    vertex_index = int(indices[0]) - 1 if indices[0] else None
                    texture_index = int(indices[1]) - 1 if len(indices) > 1 and indices[1] else None
                    normal_index = int(indices[2]) - 1 if len(indices) > 2 and indices[2] else None
                    face.append((vertex_index, texture_index, normal_index))

                if current_group not in groups:
                    groups[current_group] = []
                groups[current_group].append((face, current_material))
            elif parts[0] == 'usemtl':
                current_material = parts[1]
            elif parts[0] == 'g' or parts[0] == 'o':
                current_group = parts[1] if len(parts) > 1 else None

        obj_data = {
            'vertices': vertices,
            'textures': textures,
            'normals': normals,
            'materials': materials,
            'groups': groups
        }
        groups = obj_data['groups']
        vertices = obj_data['vertices']
        groups = obj_data['groups']
        materials = obj_data['materials']
        names = list(groups.keys())
        return_topologies = []
        for i in range(len(names)):
            object_faces = []
            face_selectors = []
            object_name = names[i]
            faces = groups[object_name]
            f = faces[0] # Get object material from first face. Assume it is the material of the group
            object_color = defaultColor
            object_opacity = defaultOpacity
            object_material = None
            if len(f) >= 2:
                object_material = f[1]
                if object_material in materials.keys():
                    object_color = materials[object_material]['Kd']
                    object_color = [int(round(c*255,0)) for c in object_color]
                    object_opacity = materials[object_material]['d']
            for f in faces:
                indices = f[0]
                face_material = f[1]
                face_indices = []
                for coordinate in indices:
                    face_indices.append(coordinate[0])
                face = Topology.ByGeometry(vertices=vertices, faces=[face_indices])
                object_faces.append(face)
                if not face_material == object_material:
                    if face_material in materials.keys():
                        face_color = materials[face_material]['Kd']
                        face_color = [int(round(c*255,0)) for c in face_color]
                        face_opacity = materials[face_material]['d']
                        selector = Face.InternalVertex(face)
                        d = Dictionary.ByKeysValues(['color', 'opacity'], [face_color, face_opacity])
                        selector = Topology.SetDictionary(selector, d)
                        face_selectors.append(selector)

            topology = Cluster.ByTopologies(object_faces)
            if Topology.IsInstance(topology, "Topology"):
                if selfMerge:
                    topology = Topology.SelfMerge(topology)
                if Topology.IsInstance(topology, "Topology"):
                    if removeCoplanarFaces:
                        topology = Topology.RemoveCoplanarFaces(topology, tolerance=tolerance)
                    if Topology.IsInstance(topology, "Topology"):
                        d = Dictionary.ByKeysValues(['name', 'color', 'opacity'], [object_name, object_color, object_opacity])
                        topology = Topology.SetDictionary(topology, d)
                        if len(face_selectors) > 0:
                            topology = Topology.TransferDictionariesBySelectors(topology, selectors=face_selectors, tranFaces=True, tolerance=tolerance)
                        return_topologies.append(topology)
        return return_topologies

    @staticmethod
    def ByOCCTShape(occtShape):
        """
        Creates a topology from the input OCCT shape. See https://dev.opencascade.org/doc/overview/html/occt_user_guides__modeling_data.html.

        Parameters
        ----------
        occtShape : topologic_core.TopoDS_Shape
            The inoput OCCT Shape.

        Returns
        -------
        topologic_core.Topology
            The created topology.

        """
        return topologic.Topology.ByOcctShape(occtShape, "")
    
    @staticmethod
    def ByPDFFile(file, wires=False, faces=False, includeTypes=[], excludeTypes=[], edgeColorKey="edge_color", edgeWidthKey="edge_width", faceColorKey="face_color", faceOpacityKey="face_opacity", tolerance=0.0001, silent=False):
        """
        Import PDF file and convert its entities to topologies.

        Parameters
        ----------
        file : file
            The input PDF file
        wires : bool , optional
            If set to True, wires will be constructed when possible. The default is True.
        faces : bool , optional
            If set to True, faces will be constructed when possible. The default is True.
        includeTypes : list , optional
            A list of PDF object types to include in the returned result. The default is [] which means all PDF objects will be included.
            The possible strings to include in this list are: ["line", "curve", "rectangle", "quadrilateral"]
        excludeTypes : list , optional
            A list of PDF object types to exclude from the returned result. The default is [] which mean no PDF object type will be excluded.
            The possible strings to include in this list are: ["line", "curve", "rectangle", "quadrilateral"]
        edgeColorKey : str , optional
            The dictionary key under which to store the edge color. The default is None.
        edgeWidthKey : str , optional
            The dictionary key under which to store the edge width. The default is None.
        faceColorKey : str , optional
            The dictionary key under which to store the face color. The default is None.
        faceOpacityKey : str , optional
            The dictionary key under which to store the face opacity. The default is None.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        list
        A list of Topologic entities (edges, wires, faces, clusters) with attached dictionaries.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster
        from topologicpy.Dictionary import Dictionary
        import os
        import warnings

        def interpolate_bezier(control_points, num_points=50):
            """
            Interpolate a cubic Bezier curve given control points.

            Args:
                control_points (list): List of control points (pymupdf.Point objects).
                num_points (int): Number of points to interpolate along the curve.

            Returns:
                list: A list of interpolated points (pymupdf.Point objects).
            """
            p0, p1, p2, p3 = control_points
            points = [
                pymupdf.Point(
                    (1 - t)**3 * p0.x + 3 * (1 - t)**2 * t * p1.x + 3 * (1 - t) * t**2 * p2.x + t**3 * p3.x,
                    (1 - t)**3 * p0.y + 3 * (1 - t)**2 * t * p1.y + 3 * (1 - t) * t**2 * p2.y + t**3 * p3.y
                )
                for t in (i / num_points for i in range(num_points + 1))
            ]
            return points

        def map_types(type_list):
            type_mapping = {
                "line": "l",
                "curve": "c",
                "rect": "re",
                "quad": "qu"
            }
            return [type_mapping[key] for key in type_mapping if any(key in item for item in type_list)]

        def remove_overlap(list1, list2):
            set1, set2 = set(list1), set(list2)
            # Remove common elements from both sets
            set1 -= set2
            set2 -= set(list1)
            return list(set1), list(set2)
        
        try:
            import pymupdf  # PyMuPDF
        except:
            if not silent:
                print("Topology.ByPDFFile - Warning: Installing required PyMuPDF library.")
            try:
                os.system("pip install PyMuPDF")
            except:
                os.system("pip install PyMuPDF --user")
            try:
                import pymupdf
                if not silent:
                    print("Topology.ByPDFFile - Information: PyMUDF library installed correctly.")
            except:
                if not silent:
                    warnings.warn("Topology.ByPDFFile - Error: Could not import PyMuPDF. Please try to install PyMuPDF manually. Returning None.")
                return None
        if not file:
            if not silent:
                print("Topology.ByPDFFile - Error: Could not open the PDF file. Returning None.")
            return None
        
        if includeTypes == None:
            includeTypes = []
        if excludeTypes == None:
            excludeTypes = []
        includeTypes = [item for item in includeTypes if isinstance(item, str)]
        excludeTypes = [item for item in excludeTypes if isinstance(item, str)]
        in_types = map_types([c.lower() for c in includeTypes])
        ex_types = map_types([c.lower() for c in excludeTypes])
        in_types_1, ex_types_1 = remove_overlap(in_types, ex_types)
        if not len(in_types_1) == len(in_types) or not len(ex_types_1) == len(ex_types):
            if not silent:
                print("Topology.ByPDFFile - Warning: Ther includeTypes and excludeTypes input parameters contain overlapping elements. These have been excluded.")
        in_types = in_types_1

        topologic_entities = []
        for page_num, page in enumerate(file, 1):
            matrix = pymupdf.Matrix(1, 1)  # Identity matrix for default transformation
            paths = page.get_drawings()
            for path in paths:
                if not path.get("stroke_opacity") == 0:
                    close = path.get("closePath")
                    components = []
                    edge_color = path.get("color", [0, 0, 0]) or [0, 0, 0]
                    edge_color = [int(255 * c) for c in edge_color] # Convert stroke color to 0-255 range
                    edge_width = path.get("width", 1) or 1
                    face_color = path.get("fill", [1, 1, 1]) or [1,1,1]
                    face_color = [int(255 * c) for c in face_color]
                    face_opacity = path.get("fill_opacity", 1) or 1

                    # Create the dictionary for line width, color, fill color, and fill opacity
                    keys = [edgeWidthKey, edgeColorKey, faceColorKey, faceOpacityKey]
                    values = [
                        edge_width,
                        edge_color,  # Convert stroke color to 0-255 range
                        face_color,     # Convert fill color to 0-255 range
                        face_opacity
                    ]
                    dictionary = Dictionary.ByKeysValues(keys, values)
                    items = path.get("items") or []
                    for it, item in enumerate(items):
                        if (item[0] in in_types or len(in_types) == 0) and (not item[0] in ex_types):
                            if item[0] == "l":  # Line
                                start_point = pymupdf.Point(item[1][0], item[1][1]).transform(matrix)
                                end_point = pymupdf.Point(item[2][0], item[2][1]).transform(matrix)
                                start_vertex = Vertex.ByCoordinates(start_point.x, -start_point.y, 0)
                                end_vertex = Vertex.ByCoordinates(end_point.x, -end_point.y, 0)
                                d = Vertex.Distance(start_vertex, end_vertex)
                                if d > tolerance:
                                    edge = Edge.ByStartVertexEndVertex(start_vertex, end_vertex, tolerance=tolerance, silent=silent)
                                    if not edge == None:
                                        edge = Topology.SetDictionary(edge, dictionary)  # Set dictionary
                                        components.append(edge)
                                if it == 0:
                                    v_f_p = pymupdf.Point(item[1][0], item[1][1]).transform(matrix)
                                    very_first_vertex = Vertex.ByCoordinates(v_f_p.x, -v_f_p.y, 0)
                                if it == len(items)-1 and path.get("closePath", False) == True:
                                    v_l_p = pymupdf.Point(item[2][0], item[2][1]).transform(matrix)
                                    very_last_vertex = Vertex.ByCoordinates(v_l_p.x, -v_l_p.y, 0)
                                    edge = Edge.ByStartVertexEndVertex(very_last_vertex, very_first_vertex, tolerance=tolerance, silent=True)
                                    if not edge == None:
                                        edge = Topology.SetDictionary(edge, dictionary)  # Set dictionary
                                        components.append(edge)
                                
                            elif item[0] == "c":  # Bezier curve (approximated by segments)
                                control_points = [pymupdf.Point(p[0], p[1]).transform(matrix) for p in item[1:]]
                                bezier_points = interpolate_bezier(control_points)
                                vertices = [Vertex.ByCoordinates(pt.x, -pt.y, 0) for pt in bezier_points]
                                for i in range(len(vertices)-1):
                                    start_vertex = vertices[i]
                                    end_vertex = vertices[i+1]
                                    edge = Edge.ByStartVertexEndVertex(start_vertex, end_vertex, tolerance=tolerance, silent=False)
                                    if not edge == None:
                                        edge = Topology.SetDictionary(edge, dictionary)  # Set dictionary
                                        components.append(edge)
                            elif item[0] == "re":  # Rectangle
                                x0, y0, x1, y1 = item[1]
                                vertices = [
                                    Vertex.ByCoordinates(x0, -y0, 0),
                                    Vertex.ByCoordinates(x1, -y0, 0),
                                    Vertex.ByCoordinates(x1, -y1, 0),
                                    Vertex.ByCoordinates(x0, -y1, 0)
                                ]
                                for i in range(len(vertices)-1):
                                    start_vertex = vertices[i]
                                    end_vertex = vertices[i+1]
                                    edge = Edge.ByStartVertexEndVertex(start_vertex, end_vertex, tolerance=tolerance, silent=False)
                                    if not edge == None:
                                        edge = Topology.SetDictionary(edge, dictionary)  # Set dictionary
                                        components.append(edge)
                                    edge = Edge.ByStartVertexEndVertex(vertices[-1], vertices[0], tolerance=tolerance, silent=False)
                                    if not edge == None:
                                        edge = Topology.SetDictionary(edge, dictionary)  # Set dictionary
                                        components.append(edge)

                            elif item[0] == "qu":  # Quadrilateral
                                quad_points = [pymupdf.Point(pt[0], pt[1]).transform(matrix) for pt in item[1]]
                                vertices = [Vertex.ByCoordinates(pt.x, -pt.y, 0) for pt in quad_points]
                                for i in range(len(vertices)-1):
                                    start_vertex = vertices[i]
                                    end_vertex = vertices[i+1]
                                    edge = Edge.ByStartVertexEndVertex(start_vertex, end_vertex, tolerance=tolerance, silent=False)
                                    if not edge == None:
                                        edge = Topology.SetDictionary(edge, dictionary)  # Set dictionary
                                        components.append(edge)
                                    edge = Edge.ByStartVertexEndVertex(vertices[-1], vertices[0], tolerance=tolerance, silent=False)
                                    if not edge == None:
                                        edge = Topology.SetDictionary(edge, dictionary)  # Set dictionary
                                        components.append(edge)

                    if len(components) > 0:
                        if len(components) == 1:
                            tp_object = components[0]
                        elif len(components) > 1:
                            tp_object = Cluster.ByTopologies(components)
                            if wires == True or faces == True:
                                tp_object = Topology.SelfMerge(tp_object)
                        if faces == True:
                            if Topology.IsInstance(tp_object, "wire"):
                                if Wire.IsClosed(tp_object):
                                    tp_object = Face.ByWire(tp_object, silent=True) or tp_object
                        tp_object = Topology.SetDictionary(tp_object, dictionary)
                        edges = Topology.Edges(tp_object)
                        for edge in edges:
                            edge = Topology.SetDictionary(edge, dictionary)
                        topologic_entities.append(tp_object)

        return topologic_entities

    @staticmethod
    def ByPDFPath(path, wires=False, faces=False, includeTypes=[], excludeTypes=[], edgeColorKey="edge_color", edgeWidthKey="edge_width", faceColorKey="face_color", faceOpacityKey="face_opacity", tolerance=0.0001, silent=False):
        """
        Import PDF file and convert its entities to topologies.

        Parameters
        ----------
        path : path
            The input path to the PDF file
        wires : bool , optional
            If set to True, wires will be constructed when possible. The default is True.
        faces : bool , optional
            If set to True, faces will be constructed when possible. The default is True.
        includeTypes : list , optional
            A list of PDF object types to include in the returned result. The default is [] which means all PDF objects will be included.
            The possible strings to include in this list are: ["line", "curve", "rectangle", "quadrilateral"]
        excludeTypes : list , optional
            A list of PDF object types to exclude from the returned result. The default is [] which mean no PDF object type will be excluded.
            The possible strings to include in this list are: ["line", "curve", "rectangle", "quadrilateral"]
        edgeColorKey : str , optional
            The dictionary key under which to store the edge color. The default is None.
        edgeWidthKey : str , optional
            The dictionary key under which to store the edge width. The default is None.
        faceColorKey : str , optional
            The dictionary key under which to store the face color. The default is None.
        faceOpacityKey : str , optional
            The dictionary key under which to store the face opacity. The default is None.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        list
        A list of Topologic entities (edges, wires, faces, clusters) with attached dictionaries.

        """
        import os
        import warnings
        try:
            import pymupdf  # PyMuPDF
        except:
            if not silent:
                print("Topology.ByPDFPath - Warning: Installing required PyMuPDF library.")
            try:
                os.system("pip install PyMuPDF")
            except:
                os.system("pip install PyMuPDF --user")
            try:
                import pymupdf
                if not silent:
                    print("Topology.ByPDFPath - Information: PyMUDF library installed correctly.")
            except:
                if not silent:
                    warnings.warn("Topology.ByPDFPath - Error: Could not import PyMuPDF. Please try to install PyMuPDF manually. Returning None.")
                return None
        if not isinstance(path, str):
            if not silent:
                print("Topology.ByPDFPath - Error: the input path is not a valid path. Returning None.")
            return None
        if not os.path.exists(path):
            if not silent:
                print("Topology.ByPDFPath - Error: The specified path does not exist. Returning None.")
            return None
        pdf_file = pymupdf.open(path)
        if not pdf_file:
            if not silent:
                print("Topology.ByPDFPath - Error: Could not open the PDF file. Returning None.")
            return None

        topologies = Topology.ByPDFFile(file=pdf_file,
                            wires=wires,
                            faces=faces,
                            includeTypes = includeTypes,
                            excludeTypes = excludeTypes,
                            edgeColorKey=edgeColorKey,
                            edgeWidthKey=edgeWidthKey,
                            faceColorKey=faceColorKey,
                            faceOpacityKey=faceOpacityKey,
                            tolerance=tolerance,
                            silent=silent)
        pdf_file.close()
        return topologies

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
        topologic_core.Topology
            The created topology.

        """
        if not isinstance(string, str):
            print("Topology.ByBREPString - Error: the input string parameter is not a valid string. Returning None.")
            return None
        returnTopology = None
        try:
            returnTopology = topologic.Topology.ByString(string)
        except:
            print("Topology.ByBREPString - Error: the input string parameter is not a valid string. Returning None.")
            returnTopology = None
        return returnTopology
    
    @staticmethod
    def ByXYZFile(file, frameIdKey="id", vertexIdKey="id"):
        """
        Imports the topology from an XYZ file path. This is a very experimental method. While variants of the format exist, topologicpy reads XYZ files that conform to the following:
        An XYZ file can be made out of one or more frames. Each frame will be stored in a sepatate topologic cluster.
        First line: total number of vertices in the frame. This must be an integer. No other words or characters are allowed on this line.
        Second line: frame label. This is free text and will be stored in the dictionary of each frame (topologic_core.Cluster)
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
            The list of frames (topologic_core.Cluster).

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
            print("Topology.ByXYZFile - Error: the input file parameter is not a valid file. Returning None.")
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
        Second line: frame label. This is free text and will be stored in the dictionary of each frame (topologic_core.Cluster)
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
            The list of frames (topologic_core.Cluster).

        """
        if not path:
            print("Topology.ByXYZPath - Error: the input path parameter is not a valid path. Returning None.")
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
        topology : topologic_core.Topology
            The input topology.

        Returns
        -------
        topologic_core.Vertex
            The center of mass of the input topology.

        """
        if not Topology.IsInstance(topology, "Topology"):
            print("Topology.CenterofMass - Error: the input topology parameter is not a valid topology. Returning None.")
            return None
        return topology.CenterOfMass()
    
    @staticmethod
    def Centroid(topology):
        """
        Returns the centroid of the vertices of the input topology. See https://en.wikipedia.org/wiki/Centroid.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.

        Returns
        -------
        topologic_core.Vertex
            The centroid of the input topology.

        """
        from topologicpy.Aperture import Aperture

        if not Topology.IsInstance(topology, "Topology"):
            print("Topology.Centroid - Error: the input topology parameter is not a valid topology. Returning None.")
            return None
        if Topology.IsInstance(topology, "Aperture"):
            return Aperture.Topology(topology).Centroid()
        return topology.Centroid()

    @staticmethod
    def ClusterByKeys(topologies, *keys, silent=False):
        """
            Clusters the input list of topologies based on the input key or keys.

            Parameters
            ----------
            topologies : list
                The input list of topologies.
            keys : str or list or comma-separated str input parameters
                The key or keys in the topology's dictionary to use for clustering.
            silent : bool , optional
                If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.


            Returns
            -------
            list
                A nested list of topologies where each element is a list of topologies with the same key values.
            """
        
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Helper import Helper
        import inspect

        topologies_list = []
        if Topology.IsInstance(topologies, "Topology"):
            if not silent:
                print("Topology.ClusterByKeys - Warning: The input topologies parameter is a single topology. Returning one cluster.")
            return [[topologies]]
        if isinstance(topologies, list):
            print("topologies_list 1:", topologies_list)
            topologies_list = Helper.Flatten(topologies)
            print("topologies_list 2:", topologies_list)
            topologies_list = [x for x in topologies_list if Topology.IsInstance(x, "Topology")]
            print("topologies_list 3:", topologies_list)
            if len(topologies_list) == 0:
                if not silent:
                    print("Topology.ClusterByKeys - Error: The input topologies parameter does not contain any valid topologies. Returning None.")
                    curframe = inspect.currentframe()
                    calframe = inspect.getouterframes(curframe, 2)
                    print('caller name:', calframe[1][3])
                return None
        else:
            if not silent:
                print("Topology.ClusterByKeys - Error: The input topologies parameter is not a valid list. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
            return None
        
        keys_list = list(keys)
        keys_list = Helper.Flatten(keys_list)

        if len(keys_list) == 0:
            if not silent:
                print("Topology.ClusterByKeys - Error: The input keys parameter is an empty list. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
            return None
        keys_list = [x for x in keys_list if isinstance(x, str)]
        
        if len(keys_list) == 0:
            if not silent:
                print("Topology.ClusterByKeys - Error: The input keys parameter does not contain any valid strings. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
            return None

        clusters = []
        values = []
        for topology in topologies_list:
            d = Topology.Dictionary(topology)

            d_keys = Dictionary.Keys(d)
            if len(d_keys) > 0:
                values_list = []
                for key in keys_list:
                    v = Dictionary.ValueAtKey(d, key)
                    if not v == None:
                        values_list.append(v)
                values_list = str(values_list)
                d = Dictionary.SetValueAtKey(d, "_clustering_key_", values_list)
                topology = Topology.SetDictionary(topology, d)
                values.append(values_list)

        remaining_topologies = topologies_list
        for value in values:
            if len(remaining_topologies) == 0:
                break
            dic = Topology.Filter(remaining_topologies, topologyType="any", searchType="equal to", key="_clustering_key_", value=value)
            filtered_topologies = dic['filtered']
            final_topologies = []
            if len(filtered_topologies) > 0:
                for filtered_topology in filtered_topologies:
                    d = Topology.Dictionary(filtered_topology)
                    d = Dictionary.RemoveKey(d, "_clustering_key_")
                    keys = Dictionary.Keys(d)
                    if len(keys) > 0:
                        filtered_topology = Topology.SetDictionary(filtered_topology, d)
                    final_topologies.append(filtered_topology)
                clusters.append(final_topologies)
            remaining_topologies = dic['other']
        if len(remaining_topologies) > 0:
            clusters.append(remaining_topologies)
        return clusters

    @staticmethod
    def ClusterFaces(topology, angTolerance=2, tolerance=0.0001):
        """
        Clusters the faces of the input topology by their direction.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        angTolerance : float , optional
            The desired angular tolerance. The default is 0.1.
        tolerance : float, optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        list
            The list of clusters of faces where faces in the same cluster have the same direction.

        """
        from topologicpy.Vector import Vector
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster

        faces = Topology.SubTopologies(topology, subTopologyType="face")
        face_normals = []
        for face in faces:
            face_normals.append(Face.Normal(face))
        bins = []
        for face_normal in face_normals:
            minAngle = angTolerance * 100
            for bin in bins:
                ang = Vector.Angle(bin, face_normal)
                if ang < minAngle:
                    minAngle = ang
            if minAngle > angTolerance:
                bins.append(face_normal)
        num_bins = len(bins)
        # Convert face_normals to a numpy array for efficient computation
        face_normals_array = np.array(face_normals)

        # Compute the bounds for each bin along each dimension
        bin_bounds = [np.linspace(-1, 1, num_bins + 1) for _ in range(3)]

        # Assign each face to a bin based on its normal
        bin_indices = [np.digitize(face_normal, bounds) - 1 for face_normal, bounds in zip(face_normals_array.T, bin_bounds)]
        # Combine the indices along the three dimensions to get a single bin index for each face
        cluster_labels = bin_indices[0] * (num_bins**2) + bin_indices[1] * num_bins + bin_indices[2]
        cluster_labels = list(cluster_labels)
        bins = list(set(cluster_labels))
        clusters = []
        for bin in bins:
            clusters.append([])
        for i, face in enumerate(faces):
            ind = bins.index(cluster_labels[i])
            clusters[ind].append(face)
        final_clusters = []
        for cluster in clusters:
            final_clusters.append(Topology.SelfMerge(Cluster.ByTopologies(cluster), tolerance=tolerance))
        return final_clusters
    
    @staticmethod
    def ClusterFaces_orig(topology, angTolerance=0.1, tolerance=0.0001):
        """
        Clusters the faces of the input topology by their direction.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        angTolerance : float , optional
            The desired angular tolerance. The default is 0.1.
        tolerance : float, optional
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
                        matrix[i, j] = 1
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
                matrix[i, :] = 0
                matrix[:, i] = 0
            return matrix
        if not Topology.IsInstance(topology, "Topology"):
            print("Topology.ClusterFaces - Error: the input topology parameter is not a valid topology. Returning None.")
            return None
        faces = []
        _ = topology.Faces(None, faces)
        normals = []
        for aFace in faces:
            normals.append(Face.Normal(aFace, outputType="XYZ", mantissa=3))
        # build a matrix of similarity
        mat = buildSimilarityMatrix(normals, angTolerance)
        categories = categorizeIntoClusters(mat)
        returnList = []
        for aCategory in categories:
            tempList = []
            if len(aCategory) > 0:
                for index in aCategory:
                    tempList.append(faces[index])
                returnList.append(Topology.SelfMerge(Cluster.ByTopologies(tempList), tolerance=tolerance))
        return returnList

    @staticmethod
    def Contents(topology):
        """
        Returns the contents of the input topology

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.

        Returns
        -------
        list
            The list of contents of the input topology.

        """
        if not Topology.IsInstance(topology, "Topology"):
            print("Topology.Contents - Error: the input topology parameter is not a valid topology. Returning None.")
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
        topology : topologic_core.Topology
            The input topology.

        Returns
        -------
        list
            The list of contexts of the input topology.

        """
        if not Topology.IsInstance(topology, "Topology"):
            print("Topology.Contexts - Error: the input topology parameter is not a valid topology. Returning None.")
            return None
        contexts = []
        _ = topology.Contexts(contexts)
        return contexts

    @staticmethod
    def ConvexHull(topology, mantissa: int = 6, tolerance: float = 0.0001):
        """
        Creates a convex hull

        Parameters
        ----------
        topology : topologic_core.Topology
            The input Topology.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Topology
            The created convex hull of the input topology.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Shell import Shell
        from topologicpy.Cell import Cell
        from topologicpy.Cluster import Cluster        
        
        def convexHull3D(item, tolerance, option):
            if item:
                vertices = Topology.Vertices(item)
                pointList = []
                for v in vertices:
                    pointList.append(Vertex.Coordinates(v, mantissa=mantissa))
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
                        edges.append(Edge.ByVertices([sv, ev], tolerance=tolerance))
                    sp = hull.points[simplex[-1]]
                    ep = hull.points[simplex[0]]
                    sv = Vertex.ByCoordinates(sp[0], sp[1], sp[2])
                    ev = Vertex.ByCoordinates(ep[0], ep[1], ep[2])
                    edges.append(Edge.ByVertices([sv, ev], tolerance=tolerance))
                    faces.append(Face.ByWire(Wire.ByEdges(edges, tolerance=tolerance), tolerance=tolerance))
            try:
                c = Cell.ByFaces(faces, tolerance=tolerance)
                return c
            except:
                returnTopology = Topology.SelfMerge(Cluster.ByTopologies(faces), tolerance=tolerance)
                if Topology.Type(returnTopology) == Topology.TypeID("Shell"):
                    return Shell.ExternalBoundary(returnTopology, tolerance=tolerance)
        if not Topology.IsInstance(topology, "Topology"):
            print("Topology.ConvexHull - Error: the input topology parameter is not a valid topology. Returning None.")
            return None
        returnObject = None
        try:
            returnObject = convexHull3D(topology, tolerance, None)
        except:
            returnObject = convexHull3D(topology, tolerance, 'QJ')
        return returnObject
    
    @staticmethod
    def Copy(topology, deep=False):
        """
        Returns a copy of the input topology

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        deep : bool , optional
            If set to True, a deep copy will be performed (this is slow). Othwerwise, it will not. The default is False

        Returns
        -------
        topologic_core.Topology
            A copy of the input topology.

        """
        from topologicpy.Dictionary import Dictionary
        if not Topology.IsInstance(topology, "Topology"):
            print("Topology.Copy - Error: the input topology parameter is not a valid topology. Returning None.")
            return None
        if deep:
            return Topology.ByJSONString(Topology.JSONString([topology]))[0]
        d = Topology.Dictionary(topology)
        return_topology = Topology.ByBREPString(Topology.BREPString(topology))
        keys = Dictionary.Keys(d)
        if len(keys) > 0:
            return_topology = Topology.SetDictionary(return_topology, d)
        return return_topology
    
    @staticmethod
    def Dictionary(topology):
        """
        Returns the dictionary of the input topology

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.

        Returns
        -------
        topologic_core.Dictionary
            The dictionary of the input topology.

        """
        if not Topology.IsInstance(topology, "Topology") and not Topology.IsInstance(topology, "Graph"):
            print("Topology.Dictionary - Error: the input topology parameter is not a valid topology. Returning None.")
            return None
        return topology.GetDictionary()
    
    @staticmethod
    def Dimensionality(topology):
        """
        Returns the dimensionality of the input topology

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.

        Returns
        -------
        int
            The dimensionality of the input topology.

        """
        if not Topology.IsInstance(topology, "Topology"):
            print("Topology.Dimensionality - Error: the input topology parameter is not a valid topology. Returning None.")
            return None
        return topology.Dimensionality()
    
    @staticmethod
    def Divide(topologyA, topologyB, transferDictionary=False, addNestingDepth=False):
        """
        Divides the input topology by the input tool and places the results in the contents of the input topology.

        Parameters
        ----------
        topologyA : topologic_core.Topology
            The input topology to be divided.
        topologyB : topologic_core.Topology
            the tool used to divide the input topology.
        transferDictionary : bool , optional
            If set to True the dictionary of the input topology is transferred to the divided topologies.
        addNestingDepth : bool , optional
            If set to True the nesting depth of the division is added to the dictionaries of the divided topologies.

        Returns
        -------
        topologic_core.Topology
            The input topology with the divided topologies added to it as contents.

        """
        from topologicpy.Dictionary import Dictionary

        if not Topology.IsInstance(topologyA, "Topology"):
            print("Topology.Divide - Error: the input topologyA parameter is not a valid topology. Returning None.")
            return None
        if not Topology.IsInstance(topologyB, "Topology"):
            print("Topology.Divide - Error: the input topologyB parameter is not a valid topology. Returning None.")
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
    def Explode(topology, origin=None, scale: float = 1.25, typeFilter: str = None, axes: str = "xyz", transferDictionaries: bool = False, mantissa: int = 6, tolerance: float = 0.0001):
        """
        Explodes the input topology. See https://en.wikipedia.org/wiki/Exploded-view_drawing.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        origin : topologic_core.Vertex , optional
            The origin of the explosion. If set to None, the centroid of the input topology will be used. The default is None.
        scale : float , optional
            The scale factor of the explosion. The default is 1.25.
        typeFilter : str , optional
            The type of the subtopologies to explode. This can be any of "vertex", "edge", "face", or "cell". If set to None, a subtopology one level below the type of the input topology will be used. The default is None.
        axes : str , optional
            Sets what axes are to be used for exploding the topology. This can be any permutation or substring of "xyz". It is not case sensitive. The default is "xyz".
        transferDictionaries : bool , optional
            If set to True, the dictionaries of the original subTopologies are transferred to the exploded topologies. Otherwise, they are not. The default is False.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
        Returns
        -------
        topologic_core.Cluster
            The exploded topology.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Cluster import Cluster
        from topologicpy.Graph import Graph
        from topologicpy.Dictionary import Dictionary

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
            if Topology.IsInstance(topology, "Vertex"):
                typeFilter = "self"
            elif Topology.IsInstance(topology, "Edge"):
                typeFilter = "vertex"
            elif Topology.IsInstance(topology, "Wire"):
                typeFilter = "edge"
            elif Topology.IsInstance(topology, "Face"):
                typeFilter = "edge"
            elif Topology.IsInstance(topology, "Shell"):
                typeFilter = "face"
            elif Topology.IsInstance(topology, "Cell"):
                typeFilter = "face"
            elif Topology.IsInstance(topology, "CellComplex"):
                typeFilter = "cell"
            elif Topology.IsInstance(topology, "Cluster"):
                typeFilter = processClusterTypeFilter(topology)
            elif Topology.IsInstance(topology, "Graph"):
                typeFilter = "edge"
            return typeFilter
        
        if not Topology.IsInstance(topology, "Topology"):
            print("Topology.Explode - Error: the input topology parameter is not a valid topology. Returning None.")
            return None
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Topology.CenterOfMass(topology)
        if not typeFilter:
            typeFilter = getTypeFilter(topology)
        if not isinstance(typeFilter, str):
            print("Topology.Explode - Error: the input typeFilter parameter is not a valid string. Returning None.")
            return None
        if not isinstance(axes, str):
            print("Topology.Explode - Error: the input axes parameter is not a valid string. Returning None.")
            return None
        if Topology.IsInstance(topology, "Topology"):
            # Hack to fix a weird bug that seems to be a problem with OCCT memory handling.
            topology = Topology.ByJSONString(Topology.JSONString([topology]))[0]
        axes = axes.lower()
        x_flag = "x" in axes
        y_flag = "y" in axes
        z_flag = "z" in axes
        if not x_flag and not y_flag and not z_flag:
            print("Topology.Explode - Error: the input axes parameter is not a valid string. Returning None.")
            return None

        topologies = []
        newTopologies = []
        if Topology.IsInstance(topology, "Graph"):
            topology = Graph.Topology(topology)

        if typeFilter.lower() == "self":
            topologies = [topology]
        else:
            topologies = Topology.SubTopologies(topology, subTopologyType=typeFilter.lower())
        for aTopology in topologies:
            c = Topology.InternalVertex(aTopology, tolerance=tolerance)
            oldX = Vertex.X(c, mantissa=mantissa)
            oldY = Vertex.Y(c, mantissa=mantissa)
            oldZ = Vertex.Z(c, mantissa=mantissa)
            if x_flag:
                newX = (oldX - Vertex.X(origin, mantissa=mantissa))*scale + Vertex.X(origin, mantissa=mantissa)
            else:
                newX = oldX
            if y_flag:
                newY = (oldY - Vertex.Y(origin, mantissa=mantissa))*scale + Vertex.Y(origin, mantissa=mantissa)
            else:
                newY = oldY
            if z_flag:
                newZ = (oldZ - Vertex.Z(origin, mantissa=mantissa))*scale + Vertex.Z(origin, mantissa=mantissa)
            else:
                newZ = oldZ
            xT = newX - oldX
            yT = newY - oldY
            zT = newZ - oldZ
            newTopology = Topology.Copy(aTopology)
            newTopology = Topology.Translate(newTopology, xT, yT, zT)
            if transferDictionaries == True:
                newTopology = Topology.SetDictionary(newTopology, Topology.Dictionary(aTopology))
            newTopologies.append(newTopology)
        return Cluster.ByTopologies(newTopologies)
    
    @staticmethod
    def ExportToBIM(topologies, path : str, overwrite: bool = False, version: str = "1.0.0",
                    guidKey: str = "guid", colorKey: str = "color", typeKey: str = "type",
                    defaultColor: list = [255,255,255,1], defaultType: str = "Structure",
                    author: str = "topologicpy", date: str = None, mantissa: int = 6, tolerance: float = 0.0001):
        """
        Exports the input topology to a BIM file. See https://dotbim.net/

        Parameters
        ----------
        topologies : list or topologic_core.Topology
            The input list of topologies or a single topology. The .bim format is restricted to triangulated meshes. No wires, edges, or vertices are supported.
        path : str
            The input file path.
        overwrite : bool , optional
            If set to True the ouptut file will overwrite any pre-existing file. Otherwise, it won't. The default is False.
        version : str , optional
            The desired version number for the BIM file. The default is "1.0.0".
        guidKey : str , optional
            The key to use to find the the guid of the topology. It is case insensitive. The default is "guid". If no guid is found, one is generated automatically.
        colorKey : str , optional
            The key to use to find the the color of the topology. It is case insensitive. The default is "color". If no color is found, the defaultColor parameter is used.
        typeKey : str , optional
            The key to use to find the the type of the topology. It is case insensitive. The default is "type". If no type is found, the defaultType parameter is used.
        defaultColor : list , optional
            The default color to use for the topology. The default is [255,255,255,1] which is opaque white.
        defaultType : str , optional
            The default type to use for the topology. The default is "Structure".
        author : str , optional
            The author of the topology. The default is "topologicpy".
        date : str , optional
            The creation date of the topology. This should be in the formate "DD.MM.YYYY". The default is None which uses the date of export.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        bool
            True if the export operation is successful. False otherwise.

        """
        from topologicpy.Topology import Topology
        from topologicpy.Helper import Helper
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Cluster import Cluster
        import datetime
        from os.path import exists
        
        try:
            import dotbimpy
        except:
            print("Topology - Installing required dotbimpy library.")
            try:
                os.system("pip install dotbimpy")
            except:
                os.system("pip install dotbimpy --user")
            try:
                import dotbimpy
                print("Topology - dotbimpy library installed successfully.")
            except:
                warnings.warn("Topology - Error: Could not import dotbimpy. Please install the dotbimpy library manually. Returning None.")
                return None
        # Make sure the file extension is .brep
        ext = path[len(path)-4:len(path)]
        if ext.lower() != ".bim":
            path = path+".bim"
        if not overwrite and exists(path):
            print("Topology.ExportToBIM - Error: a file already exists at the specified path and overwrite is set to False. Returning None.")
            return None
        # Get the current date
        current_date = datetime.datetime.now()

        # Format the date as a string in DD.MM.YYYY format
        formatted_date = current_date.strftime("%d.%m.%Y")
        
        if Topology.IsInstance(topologies, "Topology"):
            topologies = [topologies]
        # Prepare input data
        final_topologies = []
        for topology in topologies:
            d = Topology.Dictionary(topology)
            topology = Topology.SelfMerge(Topology.Triangulate(topology, tolerance=tolerance), tolerance=tolerance)
            topology = Topology.SetDictionary(topology, d)
            if Topology.IsInstance(topology, "Cluster"):
                final_topologies += Cluster.CellComplexes(topology) + Cluster.FreeCells(topology, tolerance=tolerance) + Cluster.FreeShells(topology, tolerance=tolerance) + Cluster.FreeFaces(topology, tolerance=tolerance)
            elif Topology.IsInstance(topology, "CellComplex") or Topology.IsInstance(topology, "Cell") or Topology.IsInstance(topology, "Shell") or Topology.IsInstance(topology, "Face"):
                final_topologies.append(topology)
        elements = []
        meshes = []
        for i, topology in enumerate(final_topologies):
            geo_dict = Topology.Geometry(topology, mantissa=mantissa)
            coordinates = Helper.Flatten(geo_dict['vertices'])
            indices = Helper.Flatten(geo_dict['faces'])
            d = Topology.Dictionary(topology)
            color = Dictionary.ValueAtKey(d, colorKey)
            r, g, b, a = defaultColor
            if isinstance(color, list):
                if len(color) > 2:
                    r = color[0]
                    g = color[1]
                    b = color[2]
                if len(color) == 4:
                    a = color[3]
            color = dotbimpy.Color(r=r, g=g, b=b, a=int(a*255))
            guid = Dictionary.ValueAtKey(d, guidKey)
            if not guid:
                guid = str(uuid.uuid4())

            type = Dictionary.ValueAtKey(d, typeKey)
            if not type:
                type = defaultType
            info = Dictionary.PythonDictionary(d)
            info['color'] = str([r,g,b,a])
            info['guid'] = guid
            info['type'] = type
            rotation = dotbimpy.Rotation(qx=0, qy=0, qz=0, qw=1.0)
            vector = dotbimpy.Vector(x=0, y=0, z=0)
            
            # Instantiate Mesh object
            mesh = dotbimpy.Mesh(mesh_id=i, coordinates=coordinates, indices=indices)
            meshes.append(mesh)
            # Instantiate Element object
            element = dotbimpy.Element(mesh_id=i,
                            vector=vector,
                            guid=guid,
                            info=info,
                            rotation=rotation,
                            type=type,
                            color=color)
            elements.append(element)

        # File meta data
        file_info = {
            "Author": author,
            "Date": formatted_date
        }

        # Instantiate and save File object
        file = dotbimpy.File(version, meshes=meshes, elements=elements, info=file_info)
        try:
            file.save(path)
            return True
        except:
            return False

    @staticmethod
    def ExportToBREP(topology, path, overwrite=False, version=3):
        """
        Exports the input topology to a BREP file. See https://dev.opencascade.org/doc/occt-6.7.0/overview/html/occt_brep_format.html.

        Parameters
        ----------
        topology : topologic_core.Topology
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
        if not Topology.IsInstance(topology, "Topology"):
            print("Topology.ExportToBREP - Error: the input topology parameter is not a valid topology. Returning None.")
            return None
        if not isinstance(path, str):
            print("Topology.ExportToBREP - Error: the input path parameter is not a valid string. Returning None.")
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

    def ExportToDXF(topologies, path: str,  overwrite: bool = False, mantissa: int = 6):
        """
        Exports the input topology to a DXF file. See https://en.wikipedia.org/wiki/AutoCAD_DXF.
        THe DXF version is 'R2010'
        This is experimental and only geometry is exported.

        Parameters
        ----------
        topologies : list or topologic_core.Topology
            The input list of topologies. This can also be a single topologic_core.Topology.
        path : str
            The input file path.
        overwrite : bool , optional
            If set to True the ouptut file will overwrite any pre-existing file. Otherwise, it won't. The default is False.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.

        Returns
        -------
        bool
            True if the export operation is successful. False otherwise.

        """
        import os
        import warnings

        try:
            import ezdxf
        except:
            print("Topology.ExportToDXF - Information: Installing required ezdxf library.")
            try:
                os.system("pip install ezdxf")
            except:
                os.system("pip install ezdxf --user")
            try:
                import ezdxf
                print("Topology.ExportToDXF - Information: ezdxf library installed successfully.")
            except:
                warnings.warn("Topology.ExportToDXF - Error: Could not import ezdxf library. Please install it manually. Returning None.")
                return None
        
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        from os.path import exists
        if not isinstance(topologies, list):
            topologies = [topologies]
        topologies = [topology for topology in topologies if Topology.IsInstance(topology, "Topology")]
        if len(topologies) < 1:
            print("Topology.ExportToDXF - Error: The inupt list parameter topologies does not contain any valid topologies. Returning None.")
        # Make sure the file extension is .brep
        ext = path[len(path)-4:len(path)]
        if ext.lower() != ".dxf":
            path = path+".dxf"
        if not overwrite and exists(path):
            print("Topology.ExportToDXF - Error: a file already exists at the specified path and overwrite is set to False. Returning None.")
            return None
        
        def add_vertices(vertices, msp):
            for v in vertices:
                if Topology.IsInstance(v, "Vertex"):
                    msp.add_point((Vertex.X(v, mantissa=mantissa), Vertex.Y(v, mantissa=mantissa), Vertex.Z(v, mantissa=mantissa)))
        def add_edges(edges, msp):
            for e in edges:
                if Topology.IsInstance(e, "Edge"):
                    sv = Edge.StartVertex(e)
                    ev = Edge.EndVertex(e)
                    start = (Vertex.X(sv, mantissa=mantissa), Vertex.Y(sv, mantissa=mantissa), Vertex.Z(sv, mantissa=mantissa))
                    end = (Vertex.X(ev, mantissa=mantissa), Vertex.Y(ev, mantissa=mantissa), Vertex.Z(ev, mantissa=mantissa))
                    msp.add_line(start, end)
        def add_wires(wires, msp):
            for i, w in enumerate(wires):
                if Topology.IsInstance(w, "Wire"):
                    block_name = "Wire_"+str(i+1).zfill(3)
                    block = doc.blocks.new(name=block_name)
                    # Add edges to the block
                    edges = Topology.Edges(w)
                    for edge in edges:
                        sv = Edge.StartVertex(edge)
                        ev = Edge.EndVertex(edge)
                        start = (Vertex.X(sv, mantissa=mantissa), Vertex.Y(sv, mantissa=mantissa), Vertex.Z(sv, mantissa=mantissa))
                        end = (Vertex.X(ev, mantissa=mantissa), Vertex.Y(ev, mantissa=mantissa), Vertex.Z(ev, mantissa=mantissa))
                        block.add_line(start, end)
                    # Insert the block into the model space
                    msp.add_blockref(block_name, insert=(0, 0, 0))
        def add_meshes(meshes, msp):
            for m in meshes:
                data = Topology.Geometry(m)
                vertices = data['vertices']
                faces = data['faces']
                mesh = msp.add_mesh()
                mesh.dxf.subdivision_levels = 0
                with mesh.edit_data() as mesh_data:
                    mesh_data.vertices = vertices
                    mesh_data.faces = faces
        # Create a new DXF document
        doc = ezdxf.new(dxfversion='R2010')
        msp = doc.modelspace()
        i = 1
        for topology in topologies:
            if Topology.IsInstance(topology, "Vertex"):
                add_vertices([topology], msp)
            elif Topology.IsInstance(topology, "Edge"):
                add_edges([topology], msp)
            elif Topology.IsInstance(topology, "Wire"):
                add_wires([topology], msp)
            elif Topology.IsInstance(topology, "Face"):
                add_meshes([topology], msp)
            elif Topology.IsInstance(topology, "Shell"):
                add_meshes([topology], msp)
            elif Topology.IsInstance(topology, "Cell"):
                add_meshes([topology], msp)
            elif Topology.IsInstance(topology, "CellComplex"):
                add_meshes([topology], msp)
            elif Topology.IsInstance(topology, "Cluster"):
                cellComplexes = Topology.CellComplexes(topology)
                add_meshes(cellComplexes, msp)
                cells = Cluster.FreeCells(topology)
                add_meshes(cells, msp)
                shells = Cluster.FreeShells(topology)
                add_meshes(shells, msp)
                faces = Cluster.FreeFaces(topology)
                add_meshes(faces, msp)
                wires = Cluster.FreeWires(topology)
                add_wires(wires, msp)
                edges = Cluster.FreeEdges(topology)
                add_edges(edges, msp)
                vertices = Cluster.FreeVertices(topology)
                add_vertices(vertices, msp)
        
        # Save the DXF document
        status = False
        try:
            doc.saveas(path)
            status = True
        except:
            status = False
        return status

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
    def Fix(topology, topologyType: str = "CellComplex", tolerance: float = 0.0001):
        """
        Attempts to fix the input topology to matched the desired output type.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology
        topologyType : str , optional
            The desired output topology type. This must be one of "vertex", "edge", "wire", "face", "shell", "cell", "cellcomplex", "cluster". It is case insensitive. The default is "CellComplex"
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
        Returns
        -------
        topologic_core.Topology
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
            print("Topology.Fix - Error: The input topologyType parameter is not recognized. Returning original topology.")
            return topology
        if a_type == b_type:
            return topology
        if b_type == "cluster":
            topology = Topology.SelfMerge(topology, tolerance=tolerance)
            return Cluster.ByTopologies([topology])
        if b_type == "cellcomplex":
            topology = Topology.SelfMerge(topology, tolerance=tolerance)
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
            return_topology = CellComplex.ByFaces(faces, tolerance=tolerance)
            if return_topology == None:
                print("Topology.Fix - Error: Desired topologyType cannot be achieved. Returning original topology.")
                return topology
            if Topology.TypeAsString(return_topology).lower() == "cellcomplex":
                return return_topology
            print("Topology.Fix - Error: Desired topologyType cannot be achieved. Returning original topology.")
            return topology
        if b_type == "cell":
            topology = Topology.SelfMerge(topology, tolerance=tolerance)
            if Topology.TypeAsString(topology).lower() == "cell":
                return topology
            if Topology.TypeAsString(topology).lower() == "cellComplex":
                return CellComplex.ExternalBoundary(topology)
            faces = Topology.Faces(topology)
            if len(faces) < 3:
                print("Topology.Fix - Error: Desired topologyType cannot be achieved. Returning original topology.")
                return topology
            return_topology = Cell.ByFaces(faces, tolerance=tolerance)
            if return_topology == None:
                return_topology = CellComplex.ByFaces(faces, tolerance=tolerance)
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
            topology = Topology.SelfMerge(topology, tolerance=tolerance)
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
            topology = Topology.SelfMerge(topology, tolerance=tolerance)
            if Topology.TypeAsString(topology).lower() == "face":
                return topology
            wires = Topology.Wires(topology)
            if len(wires) < 1:
                print("Topology.Fix - Error: Desired topologyType cannot be achieved. Returning original topology.")
                return topology
            return_topology = Face.ByWire(wires[0], tolerance=tolerance)
            if Topology.TypeAsString(return_topology).lower() == "face":
                return return_topology
            print("Topology.Fix - Error: Desired topologyType cannot be achieved. Returning original topology.")
            return topology
        if b_type == "wire":
            topology = Topology.SelfMerge(topology, tolerance=tolerance)
            if Topology.TypeAsString(topology).lower() == "wire":
                return topology
            edges = Topology.Edges(topology)
            if len(edges) < 2:
                print("Topology.Fix - Error: Desired topologyType cannot be achieved. Returning original topology.")
                return topology
            return_topology = Wire.ByEdges(edges, tolerance=tolerance)
            if Topology.TypeAsString(return_topology).lower() == "wire":
                return return_topology
            print("Topology.Fix - Error: Desired topologyType cannot be achieved. Returning original topology.")
            return topology
        if b_type == "edge":
            topology = Topology.SelfMerge(topology, tolerance=tolerance)
            if Topology.TypeAsString(topology).lower() == "edge":
                return topology
            vertices = Topology.Vertices(topology)
            if len(vertices) < 2:
                print("Topology.Fix - Error: Desired topologyType cannot be achieved. Returning original topology.")
                return topology
            return_topology = Edge.ByVertices(vertices, tolerance=tolerance)
            if Topology.TypeAsString(return_topology).lower() == "edge":
                return return_topology
            print("Topology.Fix - Error: Desired topologyType cannot be achieved. Returning original topology.")
            return topology
        if b_type == "vertex":
            topology = Topology.SelfMerge(topology, tolerance=tolerance)
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
    def JSONString(topologies, mantissa: int = 6):
        """
        Exports the input list of topologies to a JSON string

        Parameters
        ----------
        topologies : list or topologic_core.Topology
            The input list of topologies or a single topology.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        
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
            returnDict['coordinates'] = Vertex.Coordinates(topology, mantissa=mantissa)
            returnDict['dictionary'] = Dictionary.PythonDictionary(Topology.Dictionary(topology))
            return returnDict

        def getEdge(topology, uuidKey="uuid"):
            returnDict = {}
            uuidOne = getUUID(topology, uuidKey=uuidKey)
            returnDict['type'] = "Edge"
            returnDict['uuid'] = uuidOne
            returnDict['vertices'] = [getUUID(v, uuidKey=uuidKey) for v in Topology.Vertices(topology)]
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
            returnDict = {}
            uuidOne = getUUID(topology, uuidKey=uuidKey)
            returnDict['type'] = "Face"
            returnDict['uuid'] = uuidOne
            wires = []
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
            if Topology.IsInstance(topology, "Vertex"):
                d = getVertex(topology, uuidKey=uuidKey)
            elif Topology.IsInstance(topology, "Edge"):
                d = getEdge(topology, uuidKey=uuidKey)
            elif Topology.IsInstance(topology, "Wire"):
                d = getWire(topology, uuidKey=uuidKey)
            elif Topology.IsInstance(topology, "Face"):
                d = getFace(topology, uuidKey=uuidKey)
            elif Topology.IsInstance(topology, "Shell"):
                d = getShell(topology, uuidKey=uuidKey)
            elif Topology.IsInstance(topology, "Cell"):
                d = getCell(topology, uuidKey=uuidKey)
            elif Topology.IsInstance(topology, "CellComplex"):
                d = getCellComplex(topology, uuidKey=uuidKey)
            d['dictionary']['toplevel'] = topLevel
            json_data += getSubTopologyData(topology, uuidKey=uuidKey)
            apertures = Topology.Apertures(topology)
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
                apertures = Topology.Apertures(v)
                aperture_data = []
                for ap in apertures:
                    aperture_data.append(getApertureData(ap, topLevel=False, uuidKey=uuidKey))
                d['apertures'] = aperture_data
                json_data.append(d)
            edges = Topology.Edges(topology)
            for e in edges:
                d = getEdge(e, uuidKey=uuidKey)
                d['dictionary']['toplevel'] = False
                apertures = Topology.Apertures(e)
                aperture_data = []
                for ap in apertures:
                    aperture_data.append(getApertureData(ap, topLevel=False, uuidKey=uuidKey))
                d['apertures'] = aperture_data
                json_data.append(d)
            wires = Topology.Wires(topology)
            for w in wires:
                d = getWire(w, uuidKey=uuidKey)
                d['dictionary']['toplevel'] = False
                apertures = Topology.Apertures(w)
                aperture_data = []
                for ap in apertures:
                    aperture_data.append(getApertureData(ap, topLevel=False, uuidKey=uuidKey))
                d['apertures'] = aperture_data
                json_data.append(d)
            faces = Topology.Faces(topology)
            for f in faces:
                d = getFace(f, uuidKey=uuidKey)
                d['dictionary']['toplevel'] = False
                apertures = Topology.Apertures(f)
                aperture_data = []
                for ap in apertures:
                    aperture_data.append(getApertureData(ap, topLevel=False, uuidKey=uuidKey))
                d['apertures'] = aperture_data
                json_data.append(d)
            shells = Topology.Shells(topology)
            for s in shells:
                d = getShell(s, uuidKey=uuidKey)
                d['dictionary']['toplevel'] = False
                apertures = Topology.Apertures(s)
                aperture_data = []
                for ap in apertures:
                    aperture_data.append(getApertureData(ap, topLevel=False, uuidKey=uuidKey))
                d['apertures'] = aperture_data
                json_data.append(d)
            cells = Topology.Cells(topology)
            for c in cells:
                d = getCell(c, uuidKey=uuidKey)
                d['dictionary']['toplevel'] = False
                apertures = Topology.Apertures(c)
                aperture_data = []
                for ap in apertures:
                    aperture_data.append(getApertureData(ap, topLevel=False, uuidKey=uuidKey))
                d['apertures'] = aperture_data
                json_data.append(d)
            cellComplexes = Topology.CellComplexes(topology)
            for cc in cellComplexes:
                d = getCellComplex(cc, uuidKey=uuidKey)
                d['dictionary']['toplevel'] = False
                apertures = Topology.Apertures(cc)
                aperture_data = []
                for ap in apertures:
                    aperture_data.append(getApertureData(ap, topLevel=False, uuidKey=uuidKey))
                d['apertures'] = aperture_data
                json_data.append(d)
            return json_data

        def getJSONData(topology, topLevel=False, uuidKey="uuid"):
            json_data = []
            if Topology.IsInstance(topology, "Vertex"):
                d = getVertex(topology, uuidKey=uuidKey)
            elif Topology.IsInstance(topology, "Edge"):
                d = getEdge(topology, uuidKey=uuidKey)
            elif Topology.IsInstance(topology, "Wire"):
                d = getWire(topology, uuidKey=uuidKey)
            elif Topology.IsInstance(topology, "Face"):
                d = getFace(topology, uuidKey=uuidKey)
            elif Topology.IsInstance(topology, "Shell"):
                d = getShell(topology, uuidKey=uuidKey)
            elif Topology.IsInstance(topology, "Cell"):
                d = getCell(topology, uuidKey=uuidKey)
            elif Topology.IsInstance(topology, "CellComplex"):
                d = getCellComplex(topology, uuidKey=uuidKey)
            else:
                print("Topology.JSONString - Error: Unknown topology type:", topology, ". Returning None.")
                return None
            d['dictionary']['toplevel'] = topLevel
            json_data += getSubTopologyData(topology, uuidKey=uuidKey)
            apertures = Topology.Apertures(topology)
            aperture_data = []
            for ap in apertures:
                aperture_data.append(getApertureData(ap, topLevel=False, uuidKey=uuidKey))
            d['apertures'] = aperture_data
            json_data.append(d)
            return json_data
        json_data = []
        if not isinstance(topologies, list):
            topologies = [topologies]
        topologies = [x for x in topologies if Topology.IsInstance(x, "Topology")]
        for topology in topologies:
            json_data += getJSONData(topology, topLevel=True, uuidKey="uuid")
        json_string = json.dumps(json_data, indent=4, sort_keys=False)
        return json_string
    
    @staticmethod
    def _OBJString(topology, color, vertexIndex, transposeAxes: bool = True, mode: int = 0, meshSize: float = None, mantissa: int = 6, tolerance: float = 0.0001):
        """
        Returns the Wavefront string of the input topology. This is very experimental and outputs a simple solid topology.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        color : list
            The desired color to assign to the topology
        vertexIndex : int
            The vertex index to use as the starting index.
        transposeAxes : bool , optional
            If set to True the Z and Y coordinates are transposed so that Y points "up"
        mode : int , optional
            The desired mode of meshing algorithm. Several options are available:
            0: Classic
            1: MeshAdapt
            3: Initial Mesh Only
            5: Delaunay
            6: Frontal-Delaunay
            7: BAMG
            8: Fontal-Delaunay for Quads
            9: Packing of Parallelograms
            All options other than 0 (Classic) use the gmsh library. See https://gmsh.info/doc/texinfo/gmsh.html#Mesh-options
            WARNING: The options that use gmsh can be very time consuming and can create very heavy geometry.
        meshSize : float , optional
            The desired size of the mesh when using the "mesh" option. If set to None, it will be
            calculated automatically and set to 10% of the overall size of the face.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        str
            The Wavefront OBJ string of the input topology

        """
        
        from topologicpy.Helper import Helper

        if not Topology.IsInstance(topology, "Topology"):
            print("Topology.ExportToOBJ - Error: the input topology parameter is not a valid topology. Returning None.")
            return None

        lines = []
        #version = Helper.Version()
        #lines.append("# topologicpy " + version)
        topology = Topology.Triangulate(topology, mode=mode, meshSize=meshSize, tolerance=tolerance)
        d = Topology.Geometry(topology, mantissa=mantissa)
        vertices = d['vertices']
        faces = d['faces']
        tVertices = []
        if transposeAxes:
            for v in vertices:
                tVertices.append([v[0], v[2], v[1]])
            vertices = tVertices
        for v in vertices:
            lines.append("v " + str(v[0]) + " " + str(v[1]) + " " + str(v[2]))
        for f in faces:
            line = "usemtl " + str(color) + "\nf"  # reference the material name
            for j in f:
                line = line + " " + str(j + vertexIndex)
            lines.append(line)
        finalLines = lines[0]
        for i in range(1, len(lines)):
            finalLines = finalLines + "\n" + lines[i]
        return finalLines, len(vertices)
    

    @staticmethod
    def ExportToOBJ(*topologies, path, nameKey="name", colorKey="color", opacityKey="opacity", defaultColor=[256,256,256], defaultOpacity=0.5, transposeAxes: bool = True, mode: int = 0, meshSize: float = None, overwrite: bool = False, mantissa: int = 6, tolerance: float = 0.0001):
        """
        Exports the input topology to a Wavefront OBJ file. This is very experimental and outputs a simple solid topology.

        Parameters
        ----------
        topologies : list or comma separated topologies
            The input list of topologies.
        path : str
            The input file path.
        nameKey : str , optional
            The topology dictionary key under which to find the name of the topology. The default is "name".
        colorKey : str, optional
            The topology dictionary key under which to find the color of the topology. The default is "color".
        opacityKey : str , optional
            The topology dictionary key under which to find the opacity of the topology. The default is "opacity".
        defaultColor : list , optional
            The default color to use if no color is stored in the topology dictionary. The default is [255,255, 255] (white).
        defaultOpacity : float , optional
            The default opacity to use of no opacity is stored in the topology dictionary. This must be between 0 and 1. The default is 1 (fully opaque).
        transposeAxes : bool , optional
            If set to True the Z and Y coordinates are transposed so that Y points "up"
        mode : int , optional
            The desired mode of meshing algorithm (for triangulation). Several options are available:
            0: Classic
            1: MeshAdapt
            3: Initial Mesh Only
            5: Delaunay
            6: Frontal-Delaunay
            7: BAMG
            8: Fontal-Delaunay for Quads
            9: Packing of Parallelograms
            All options other than 0 (Classic) use the gmsh library. See https://gmsh.info/doc/texinfo/gmsh.html#Mesh-options
            WARNING: The options that use gmsh can be very time consuming and can create very heavy geometry.
        meshSize : float , optional
            The desired size of the mesh when using the "mesh" option. If set to None, it will be
            calculated automatically and set to 10% of the overall size of the face.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        overwrite : bool , optional
            If set to True the ouptut file will overwrite any pre-existing file. Otherwise, it won't. The default is False.

        Returns
        -------
        bool
            True if the export operation is successful. False otherwise.

        """
        from topologicpy.Helper import Helper
        from os.path import exists

        if isinstance(topologies, tuple):
            topologies = Helper.Flatten(list(topologies))
        if isinstance(topologies, list):
            new_topologies = [d for d in topologies if Topology.IsInstance(d, "Topology")]
        if len(new_topologies) == 0:
            print("Topology.ExportToOBJ - Error: the input topologies parameter does not contain any valid topologies. Returning None.")
            return None
        if not isinstance(new_topologies, list):
            print("Topology.ExportToOBJ - Error: The input topologies parameter is not a valid list. Returning None.")
            return None

        if not overwrite and exists(path):
            print("Topology.ExportToOBJ - Error: a file already exists at the specified path and overwrite is set to False. Returning None.")
            return None
        
        # Make sure the file extension is .obj
        ext = path[len(path)-4:len(path)]
        if ext.lower() != ".obj":
            path = path+".obj"
        status = False
       
        mtl_path = path[:-4] + ".mtl"

        obj_string, mtl_string = Topology.OBJString(new_topologies,
                                                    nameKey=nameKey,
                                                    colorKey=colorKey,
                                                    opacityKey=opacityKey,
                                                    defaultColor=defaultColor,
                                                    defaultOpacity=defaultOpacity,
                                                    transposeAxes=transposeAxes,
                                                    mode=mode,
                                                    meshSize=meshSize,
                                                    mantissa=mantissa,
                                                    tolerance=tolerance)
        # Write out the material file
        with open(mtl_path, "w") as mtl_file:
            mtl_file.write(mtl_string)
        # Write out the obj file
        with open(path, "w") as obj_file:
            obj_file.write(obj_string)
        return True

    @staticmethod
    def OBJString(*topologies, nameKey="name", colorKey="color", opacityKey="opacity", defaultColor=[256,256,256], defaultOpacity=0.5, transposeAxes: bool = True, mode: int = 0, meshSize: float = None, mantissa: int = 6, tolerance: float = 0.0001):
        """
        Exports the input topology to a Wavefront OBJ file. This is very experimental and outputs a simple solid topology.

        Parameters
        ----------
        topologies : list or comma separated topologies
            The input list of topologies.
        nameKey : str , optional
            The topology dictionary key under which to find the name of the topology. The default is "name".
        colorKey : str, optional
            The topology dictionary key under which to find the color of the topology. The default is "color".
        opacityKey : str , optional
            The topology dictionary key under which to find the opacity of the topology. The default is "opacity".
        defaultColor : list , optional
            The default color to use if no color is stored in the topology dictionary. The default is [255,255, 255] (white).
        defaultOpacity : float , optional
            The default opacity to use of no opacity is stored in the topology dictionary. This must be between 0 and 1. The default is 1 (fully opaque).
        transposeAxes : bool , optional
            If set to True the Z and Y coordinates are transposed so that Y points "up"
        mode : int , optional
            The desired mode of meshing algorithm (for triangulation). Several options are available:
            0: Classic
            1: MeshAdapt
            3: Initial Mesh Only
            5: Delaunay
            6: Frontal-Delaunay
            7: BAMG
            8: Fontal-Delaunay for Quads
            9: Packing of Parallelograms
            All options other than 0 (Classic) use the gmsh library. See https://gmsh.info/doc/texinfo/gmsh.html#Mesh-options
            WARNING: The options that use gmsh can be very time consuming and can create very heavy geometry.
        meshSize : float , optional
            The desired size of the mesh when using the "mesh" option. If set to None, it will be
            calculated automatically and set to 10% of the overall size of the face.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        list
            Return the OBJ and MTL strings as a list.

        """
        from topologicpy.Helper import Helper
        from topologicpy.Dictionary import Dictionary
        import io

        obj_file = io.StringIO()
        mtl_file = io.StringIO()

        if isinstance(topologies, tuple):
            topologies = Helper.Flatten(list(topologies))
        if isinstance(topologies, list):
            new_topologies = [d for d in topologies if Topology.IsInstance(d, "Topology")]
        if len(new_topologies) == 0:
            print("Topology.OBJString - Error: the input topologies parameter does not contain any valid topologies. Returning None.")
            return None
        if not isinstance(new_topologies, list):
            print("Topology.OBJString - Error: The input dictionaries parameter is not a valid list. Returning None.")
            return None
       
        # Write out the material file
        n = max(len(str(len(topologies))), 3)
        for i in range(len(new_topologies)):
            d = Topology.Dictionary(new_topologies[i])
            name = Dictionary.ValueAtKey(d, nameKey) or "Untitled_"+str(i).zfill(n)
            color = Dictionary.ValueAtKey(d, colorKey) or defaultColor
            color = [c/255 for c in color]
            opacity = Dictionary.ValueAtKey(d, opacityKey) or defaultOpacity
            mtl_file.write("newmtl color_" + str(i).zfill(n) + "\n")
            mtl_file.write("Kd " + ' '.join(map(str, color)) + "\n")
            mtl_file.write("d " + str(opacity) + "\n")
        
        vertex_index = 1  # global vertex index counter
        obj_file.writelines("# topologicpy "+Helper.Version()+"\n")
        obj_file.writelines("mtllib example.mtl")
        for i in range(len(topologies)):
            d = Topology.Dictionary(topologies[i])
            name = Dictionary.ValueAtKey(d, nameKey) or "Untitled_"+str(i).zfill(n)
            name = name.replace(" ", "_")
            obj_file.writelines("\ng "+name+"\n")
            result = Topology._OBJString(topologies[i], "color_" + str(i).zfill(n), vertex_index, transposeAxes=transposeAxes, mode=mode,
                            meshSize=meshSize,
                            mantissa=mantissa, tolerance=tolerance)
            
            obj_file.writelines(result[0])
            vertex_index += result[1]
        obj_string = obj_file.getvalue()
        mtl_string = mtl_file.getvalue()
        obj_file.close()
        mtl_file.close()
        return obj_string, mtl_string

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
                    if v == None:
                        otherTopologies.append(aTopology)
                    else:
                        v = str(v).lower()
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
        return {"filtered": filteredTopologies, "other": otherTopologies}

    @staticmethod
    def Flatten(topology, origin=None, direction: list = [0, 0, 1], mantissa: int = 6):
        """
        Flattens the input topology such that the input origin is located at the world origin and the input topology is rotated such that the input vector is pointed in the Up direction (see Vector.Up()).

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        origin : topologic_core.Vertex , optional
            The input origin. If set to None, The object's centroid will be used to place the world origin. The default is None.
        direction : list , optional
            The input direction vector. The input topology will be rotated such that this vector is pointed in the positive Z axis.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.

        Returns
        -------
        topologic_core.Topology
            The flattened topology.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Vector import Vector
        
        if not Topology.IsInstance(topology, "Topology"):
            print("Topology.Flatten - Error: the input topology parameter is not a valid topology. Returning None.")
            return None
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Topology.Centroid(topology)
        up = Vector.Up()
        flat_topology = Topology.Translate(topology, -Vertex.X(origin, mantissa=mantissa), -Vertex.Y(origin, mantissa=mantissa), -Vertex.Z(origin, mantissa=mantissa))
        tran_mat = Vector.TransformationMatrix(direction, up)
        flat_topology = Topology.Transform(flat_topology, tran_mat)
        flat_topology = Topology.SetDictionary(flat_topology, Topology.Dictionary(topology), silent=True)
        flat_vertices = Topology.Vertices(flat_topology)
        vertices = Topology.Vertices(topology)
        flat_edges = Topology.Edges(flat_topology)
        edges = Topology.Edges(topology)
        faces = []
        flat_faces = []
        if Topology.IsInstance(topology, "Face"):
            flat_faces = Topology.Faces(flat_topology)
            faces = Topology.Faces(topology)
        elements = vertices+edges+faces
        flat_elements = flat_vertices+flat_edges+flat_faces
        for i, f, in enumerate(flat_elements):
            f = Topology.SetDictionary(f, Topology.Dictionary(elements[i]), silent=True)
        return flat_topology
    
    @staticmethod
    def Geometry(topology, mantissa: int = 6):
        """
        Returns the geometry (mesh data format) of the input topology as a dictionary of vertices, edges, and faces.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.

        Returns
        -------
        dict
            A dictionary containing the vertices, edges, and faces data. The keys found in the dictionary are "vertices", "edges", and "faces".

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face

        vertices = []
        edges = []
        faces = []
        if topology == None:
            return [None, None, None]
        topVerts = []
        if Topology.Type(topology) == Topology.TypeID("Vertex"): #input is a vertex, just add it and process it
            topVerts.append(topology)
        else:
            topVerts = Topology.Vertices(topology)
        for aVertex in topVerts:
            try:
                vertices.index(Vertex.Coordinates(aVertex, mantissa=mantissa)) # Vertex already in list
            except:
                vertices.append(Vertex.Coordinates(aVertex, mantissa=mantissa)) # Vertex not in list, add it.
        topEdges = []
        if (Topology.Type(topology) == Topology.TypeID("Edge")): #Input is an Edge, just add it and process it
            topEdges.append(topology)
        elif (Topology.Type(topology) > Topology.TypeID("Vertex")):
            topEdges = Topology.Edges(topology)
        for anEdge in topEdges:
            e = []
            sv = anEdge.StartVertex()
            ev = anEdge.EndVertex()
            try:
                svIndex = vertices.index(Vertex.Coordinates(sv, mantissa=mantissa))
            except:
                vertices.append(Vertex.Coordinates(sv, mantissa=mantissa))
                svIndex = len(vertices)-1
            try:
                evIndex = vertices.index(Vertex.Coordinates(ev, mantissa=mantissa))
            except:
                vertices.append(Vertex.Coordinates(ev, mantissa=mantissa))
                evIndex = len(vertices)-1
            e.append(svIndex)
            e.append(evIndex)
            edges.append(e)
        topFaces = []
        if (Topology.Type(topology) == Topology.TypeID("Face")): # Input is a Face, just add it and process it
            topFaces.append(topology)
        elif (Topology.Type(topology) > Topology.TypeID("Face")):
            _ = topology.Faces(None, topFaces)
        for aFace in topFaces:
            f_dir = Face.Normal(aFace)
            ib = []
            _ = aFace.InternalBoundaries(ib)
            if(len(ib) > 0):
                triFaces = Face.Triangulate(aFace)
                for aTriFace in triFaces:
                    wire = Face.ExternalBoundary(aTriFace)
                    faceVertices = Topology.Vertices(wire)
                    f = []
                    for aVertex in faceVertices:
                        try:
                            fVertexIndex = vertices.index(Vertex.Coordinates(aVertex, mantissa=mantissa))
                        except:
                            vertices.append(Vertex.Coordinates(aVertex, mantissa=mantissa))
                            fVertexIndex = len(vertices)-1
                        f.append(fVertexIndex)
                    faces.append(f)
            else:
                wire =  Face.ExternalBoundary(aFace)
                faceVertices = Topology.Vertices(wire)
                f = []
                for aVertex in faceVertices:
                    try:
                        fVertexIndex = vertices.index(Vertex.Coordinates(aVertex, mantissa=mantissa))
                    except:
                        vertices.append(Vertex.Coordinates(aVertex, mantissa=mantissa))
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
        topology : topologic_core.Topology
            The input topology.

        Returns
        -------
        int
            The highest type found in the input topology.

        """
        from topologicpy.Cluster import Cluster

        if (Topology.Type(topology) == Topology.TypeID("Cluster")):
            return Cluster.HighestType(topology)
        else:
            return Topology.Type(topology)

    @staticmethod
    def _InternalVertex(topology, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns a vertex guaranteed to be inside the input topology.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        tolerance : float , ptional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        topologic_core.Vertex
            A vertex guaranteed to be inside the input topology.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Cell import Cell
        from topologicpy.CellComplex import CellComplex
        from topologicpy.Aperture import Aperture

        if not Topology.IsInstance(topology, "Topology"):
            print("Topology.InternalVertex - Error: the input topology parameter is not a valid topology. Returning None.")
            return None
        vst = None
        top = Topology.Copy(topology)
        if Topology.IsInstance(top, "CellComplex"): #CellComplex
            tempCell = Topology.Cells(top)[0]
            vst = Cell.InternalVertex(tempCell, tolerance=tolerance, silent=silent)
        elif Topology.IsInstance(top, "Cell"): #Cell
            vst = Cell.InternalVertex(top, tolerance=tolerance, silent=silent)
        elif Topology.IsInstance(top, "Shell"): #Shell
            tempFace = Topology.Faces(top)[0]
            vst = Face.InternalVertex(tempFace, tolerance=tolerance, silent=silent)
        elif Topology.IsInstance(top, "Face"): #Face
            vst = Face.InternalVertex(top, tolerance=tolerance, silent=silent)
        elif Topology.IsInstance(top, "Wire"): #Wire
            if top.IsClosed():
                internalBoundaries = []
                try:
                    tempFace = topologic.Face.ByExternalInternalBoundaries(top, internalBoundaries)
                    vst = Face.InternalVertex(tempFace, tolerance=tolerance, silent=silent)
                except:
                    vst = Topology.Centroid(top)
            else:
                tempEdge = Topology.Edges(top)[0]
                vst = Edge.VertexByParameter(tempEdge, 0.5)
        elif Topology.IsInstance(top, "Edge"): #Edge
            vst = Edge.VertexByParameter(top, 0.5)
        elif Topology.IsInstance(top, "Vertex"): #Vertex
            vst = top
        elif Topology.IsInstance(topology, "Aperture"): #Aperture
            vst = Face.InternalVertex(Aperture.Topology(top), tolerance=tolerance, silent=silent)
        else:
            vst = Topology.Centroid(top)
        return vst

    

    @staticmethod
    def InternalVertex(topology, timeout: int = 30, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns a vertex guaranteed to be inside the input topology.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        timeout : int , optional
            The amount of seconds to wait before timing out. The default is 30 seconds.
        tolerance : float , ptional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        topologic_core.Vertex
            A vertex guaranteed to be inside the input topology.

        """
        import concurrent.futures
        import time
        # Wrapper function with timeout
        def run_with_timeout(func, topology, tolerance=0.0001, silent=False, timeout=10):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(func, topology, tolerance=tolerance, silent=silent)
                try:
                    result = future.result(timeout=timeout)  # Wait for the result with a timeout
                    return result
                except concurrent.futures.TimeoutError:
                    return None  # or try another approach here

        result = run_with_timeout(Topology._InternalVertex, topology=topology, tolerance=tolerance, silent=silent, timeout=timeout)  # Set a 10 second timeout
        if result is None:
            # Handle failure case (e.g., try a different solution)
            if not silent:
                print("Topology.InternalVertex - Warning: Operation took too long. Returning None")
            return None
        return result

    @staticmethod
    def IsInstance(topology, type: str):
        """
        Returns True if the input topology is an instance of the class specified by the input type string.
        
        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        type : string
            The topology type. This can be one of:
            "Vertex"
            "Edge"
            "Wire"
            "Face"
            "Shell"
            "Cell"
            "CellComplex"
            "Cluster"
            "Topology"
            "Graph"
            "Aperture"
            "Dictionary"
            "Context"

        Returns
        -------
        bool
            True if the input topology is an instance of the class defined by the input type string. False otherwise.
        """

        if "vertex" in type.lower():
            return isinstance(topology, topologic.Vertex)
        elif "edge" in type.lower():
            return isinstance(topology, topologic.Edge)
        elif "wire" in type.lower():
            return isinstance(topology, topologic.Wire)
        elif "face" in type.lower():
            return isinstance(topology, topologic.Face)
        elif "shell" in type.lower():
            return isinstance(topology, topologic.Shell)
        elif "cellcomplex" in type.lower(): #Hack to test for cellcomplex before cell as they share the same prefix.
            return isinstance(topology, topologic.CellComplex)
        elif "cell" in type.lower():
            return isinstance(topology, topologic.Cell)
        elif "cluster" in type.lower():
            return isinstance(topology, topologic.Cluster)
        elif "topology" in type.lower():
            return isinstance(topology, topologic.Topology)
        elif "graph" in type.lower():
            return isinstance(topology, topologic.Graph)
        elif "aperture" in type.lower():
            return isinstance(topology, topologic.Aperture)
        elif "dictionary" in type.lower():
            return isinstance(topology, topologic.Dictionary)
        elif "context" in type.lower():
            return isinstance(topology, topologic.Context)
        else:
            print("Topology.IsInstance - Error: The type input string is not a known topology type. Returning None.")
            return None

    @staticmethod
    def IsPlanar(topology, mantissa: int = 6, tolerance: float = 0.0001):
        """
        Returns True if all the vertices of the input topology are co-planar. Returns False otherwise.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        bool
            True if all the vertices of the input topology are co-planar. False otherwise.

        """
        from topologicpy.Vertex import Vertex

        def isOnPlane(v, plane, tolerance=0.0001):
            x, y, z = v
            a, b, c, d = plane
            if math.fabs(a*x + b*y + c*z + d) <= tolerance:
                return True
            return False

        def plane(v1, v2, v3):
            a1 = Vertex.X(v2, mantissa=mantissa) - Vertex.X(v1, mantissa=mantissa)
            b1 = Vertex.Y(v2, mantissa=mantissa) - Vertex.Y(v1, mantissa=mantissa)
            c1 = Vertex.Z(v2, mantissa=mantissa) - Vertex.Z(v1, mantissa=mantissa)
            a2 = Vertex.X(v3, mantissa=mantissa) - Vertex.X(v1, mantissa=mantissa)
            b2 = Vertex.Y(v3, mantissa=mantissa) - Vertex.Y(v1, mantissa=mantissa)
            c2 = Vertex.Z(v3, mantissa=mantissa) - Vertex.Z(v1, mantissa=mantissa)
            a = b1 * c2 - b2 * c1 
            b = a2 * c1 - a1 * c2 
            c = a1 * b2 - b1 * a2 
            d = (- a * Vertex.X(v1, mantissa=mantissa) - b * Vertex.Y(v1, mantissa=mantissa) - c * Vertex.Z(v1, mantissa=mantissa))
            return [a, b, c, d]

        if not Topology.IsInstance(topology, "Topology"):
            print("Topology.IsPlanar - Error: the input topology parameter is not a valid topology. Returning None.")
            return None
        vertices = Topology.Vertices(topology)

        result = True
        if len(vertices) <= 3:
            result = True
        else:
            p = plane(vertices[0], vertices[1], vertices[2])
            for i in range(len(vertices)):
                if isOnPlane([Vertex.X(vertices[i], mantissa=mantissa), Vertex.Y(vertices[i], mantissa=mantissa), Vertex.Z(vertices[i], mantissa=mantissa)], p, tolerance=tolerance) == False:
                    result = False
                    break
        return result
    
    @staticmethod
    def IsSame(topologyA, topologyB):
        """
        Returns True if the input topologies are the same topology. Returns False otherwise.

        Parameters
        ----------
        topologyA : topologic_core.Topology
            The first input topology.
        topologyB : topologic_core.Topology
            The second input topology.

        Returns
        -------
        bool
            True of the input topologies are the same topology. False otherwise.

        """
        if not Topology.IsInstance(topologyA, "Topology"):
            print("Topology.IsSame - Error: the input topologyA parameter is not a valid topology. Returning None.")
            return None
        if not Topology.IsInstance(topologyB, "Topology"):
            print("Topology.IsSame - Error: the input topologyB parameter is not a valid topology. Returning None.")
            return None
        return topologic.Topology.IsSame(topologyA, topologyB)
    
    @staticmethod
    def MergeAll(topologies, tolerance=0.0001):
        """
        Merge all the input topologies.

        Parameters
        ----------
        topologies : list
            The list of input topologies.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Topology
            The resulting merged Topology

        """
        from topologicpy.Cluster import Cluster

        if not isinstance(topologies, list):
            print("Topology.MergeAll - Error: the input topologies parameter is not a valid list. Returning None.")
            return None
        
        topologyList = [t for t in topologies if Topology.IsInstance(t, "Topology")]
        if len(topologyList) < 1:
            print("Topology.MergeAll - Error: the input topologyList does not contain any valid topologies. Returning None.")
            return None
        return Topology.SelfMerge(Cluster.ByTopologies(topologyList), tolerance=tolerance)
    
    @staticmethod
    def Move(topology, x=0, y=0, z=0):
        """
        Moves the input topology.

        Parameters
        ----------
        topology : topologic_core.topology
            The input topology.
        x : float , optional
            The x distance value. The default is 0.
        y : float , optional
            The y distance value. The default is 0.
        z : float , optional
            The z distance value. The default is 0.

        Returns
        -------
        topologic_core.Topology
            The moved topology.

        """
        return Topology.Translate(topology, x=x, y=y, z=z)
    
    @staticmethod
    def OCCTShape(topology):
        """
        Returns the occt shape of the input topology.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.

        Returns
        -------
        topologic_core.TopoDS_Shape
            The OCCT Shape.

        """
        if not Topology.IsInstance(topology, "Topology"):
            print("Topology.OCCTShape - Error: the input topology parameter is not a valid topology. Returning None.")
            return None
        return topology.GetOcctShape()
    
    @staticmethod
    def Degree(topology, hostTopology):
        """
        Returns the number of immediate super topologies that use the input topology

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        hostTopology : topologic_core.Topology
            The input host topology to which the input topology belongs
        
        Returns
        -------
        int
            The degree of the topology (the number of immediate super topologies that use the input topology).
        
        """
        if not Topology.IsInstance(topology, "Topology"):
            print("Topology.Degree - Error: the input topology parameter is not a valid topology. Returning None.")
            return None
        if not Topology.IsInstance(hostTopology, "Topology"):
            print("Topology.Degree - Error: the input hostTopology parameter is not a valid topology. Returning None.")
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
        topology : topologic_core.Topology
            The input topology.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
        Returns
        -------
        list
            The list of nonplanar faces.
        
        """
        if not Topology.IsInstance(topology, "Topology"):
            print("Topology.NonPlanarFaces - Error: the input topology parameter is not a valid topology. Returning None.")
            return None
        faces = Topology.SubTopologies(topology, subTopologyType="face")
        return [f for f in faces if not Topology.IsPlanar(f, tolerance=tolerance)]
    
    @staticmethod
    def OpenFaces(topology):
        """
        Returns the faces that border no cells.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        
        Returns
        -------
        list
            The list of open edges.
        
        """

        if not Topology.IsInstance(topology, "Topology"):
            print("Topology.OpenFaces - Error: the input topology parameter is not a valid topology. Returning None.")
            return None
        
        return [f for f in Topology.SubTopologies(topology, subTopologyType="face") if Topology.Degree(f, hostTopology=topology) < 1]
    
    @staticmethod
    def OpenEdges(topology):
        """
        Returns the edges that border only one face.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        
        Returns
        -------
        list
            The list of open edges.
        
        """

        if not Topology.IsInstance(topology, "Topology"):
            print("Topology.OpenEdges - Error: the input topology parameter is not a valid topology. Returning None.")
            return None
        
        return [e for e in Topology.SubTopologies(topology, subTopologyType="edge") if Topology.Degree(e, hostTopology=topology) < 2]
    
    @staticmethod
    def OpenVertices(topology):
        """
        Returns the vertices that border only one edge.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        
        Returns
        -------
        list
            The list of open edges.
        
        """

        if not Topology.IsInstance(topology, "Topology"):
            print("Topology.OpenVertices - Error: the input topology parameter is not a valid topology. Returning None.")
            return None
        
        return [v for v in Topology.SubTopologies(topology, subTopologyType="vertex") if Topology.Degree(v, hostTopology=topology) < 2]
    
    @staticmethod
    def Orient(topology, origin=None, dirA=[0, 0, 1], dirB=[0, 0, 1], tolerance=0.0001):
        """
        Orients the input topology such that the input such that the input dirA vector is parallel to the input dirB vector.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        origin : topologic_core.Vertex , optional
            The input origin. If set to None, The object's centroid will be used to locate the input topology. The default is None.
        dirA : list , optional
            The first input direction vector. The input topology will be rotated such that this vector is parallel to the input dirB vector. The default is [0, 0, 1].
        dirB : list , optional
            The target direction vector. The input topology will be rotated such that the input dirA vector is parallel to this vector. The default is [0, 0, 1].
        tolerance : float , optional
            The desired tolerance. The default is 0.0001

        Returns
        -------
        topologic_core.Topology
            The flattened topology.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Vector import Vector
        if not Topology.IsInstance(topology, "Topology"):
            print("Topology.Orient - Error: the input topology parameter is not a valid topology. Returning None.")
            return None
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Topology.Centroid(topology)
        return_topology = Topology.Place(topology, originA=origin, originB=Vertex.Origin())
        tran_mat = Vector.TransformationMatrix(dirA, dirB)
        return_topology = Topology.Transform(return_topology, tran_mat)
        return_topology = Topology.Place(return_topology, originA=Vertex.Origin(), originB=origin)
        return return_topology

    @staticmethod
    def Place(topology, originA=None, originB=None, mantissa: int = 6):
        """
        Places the input topology at the specified location.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        originA : topologic_core.Vertex , optional
            The old location to use as the origin of the movement. If set to None, the centroid of the input topology is used. The default is None.
        originB : topologic_core.Vertex , optional
            The new location at which to place the topology. If set to None, the world origin (0, 0, 0) is used. The default is None.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6

        Returns
        -------
        topologic_core.Topology
            The placed topology.

        """
        from topologicpy.Vertex import Vertex
        if not Topology.IsInstance(topology, "Topology"):
            return None
        if not Topology.IsInstance(originA, "Vertex"):
            originA = Topology.Centroid(topology)
        if not Topology.IsInstance(originA, "Vertex"):
            originA = Vertex.ByCoordinates(0, 0, 0)

        x = Vertex.X(originB, mantissa=mantissa) - Vertex.X(originA, mantissa=mantissa)
        y = Vertex.Y(originB, mantissa=mantissa) - Vertex.Y(originA, mantissa=mantissa)
        z = Vertex.Z(originB, mantissa=mantissa) - Vertex.Z(originA, mantissa=mantissa)
        newTopology = None
        try:
            newTopology = Topology.Translate(topology, x, y, z)
        except:
            print("Topology.Place - Error: (Topologic>TopologyUtility.Place) operation failed. Returning None.")
            newTopology = None
        return newTopology

    @staticmethod
    def RemoveCollinearEdges(topology, angTolerance: float = 0.1, tolerance: float = 0.0001, silent: bool = False):
        """
        Removes the collinear edges of the input topology

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        angTolerance : float , optional
            The desired angular tolerance. The default is 0.1.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        topologic_core.Topology
            The input topology with the collinear edges removed.

        """
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Shell import Shell
        from topologicpy.Cell import Cell
        from topologicpy.CellComplex import CellComplex
        from topologicpy.Cluster import Cluster
        import inspect
        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print("Topology.RemoveCollinearEdges - Error: The input topology parameter is not a valid topology. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
            return None
        return_topology = topology
        if Topology.IsInstance(topology, "Vertex") or Topology.IsInstance(topology, "Edge"): #Vertex or Edge or Cluster, return the original topology
            return return_topology
        elif Topology.IsInstance(topology, "Wire"):
            return_topology = Wire.RemoveCollinearEdges(topology, angTolerance=angTolerance, tolerance=tolerance, silent=silent)
            return return_topology
        elif Topology.IsInstance(topology, "Face"):
            return_topology = Face.RemoveCollinearEdges(topology, angTolerance=angTolerance, tolerance=tolerance)
            return return_topology
        elif Topology.IsInstance(topology, "Shell"):
            return_topology = Shell.RemoveCollinearEdges(topology, angTolerance=angTolerance, tolerance=tolerance)
            return return_topology
        elif Topology.IsInstance(topology, "Cell"):
            return_topology = Cell.RemoveCollinearEdges(topology, angTolerance=angTolerance, tolerance=tolerance)
            return return_topology
        elif Topology.IsInstance(topology, "CellComplex"):
            return_topology = CellComplex.RemoveCollinearEdges(topology, angTolerance=angTolerance, tolerance=tolerance)
            return return_topology
        elif Topology.IsInstance(topology, "Cluster"):
            topologies = []
            topologies += Cluster.FreeVertices(topology)
            topologies += Cluster.FreeEdges(topology)
            faces = Topology.Faces(topology)
            for face in faces:
                topologies.append(Face.RemoveCollinearEdges(face, angTolerance=angTolerance, tolerance=tolerance))
            return_topology = Topology.SelfMerge(Cluster.ByTopologies(topologies), tolerance=tolerance)
        return return_topology

    @staticmethod
    def RemoveContent(topology, contents):
        """
        Removes the input content list from the input topology

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        contentList : list
            The input list of contents.

        Returns
        -------
        topologic_core.Topology
            The input topology with the input list of contents removed.

        """
        if isinstance(contents, list) == False:
            contents = [contents]
        return topology.RemoveContents(contents)
    
    @staticmethod
    def RemoveCoplanarFaces(topology, epsilon=0.01, tolerance=0.0001):
        """
        Removes coplanar faces in the input topology

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        epsilon : float , optional
            The desired epsilon (another form of tolerance) for finding if two faces are coplanar. The default is 0.01.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Topology
            The input topology with coplanar faces merged into one face.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Shell import Shell
        from topologicpy.Cell import Cell
        from topologicpy.CellComplex import CellComplex
        from topologicpy.Cluster import Cluster

        if not Topology.IsInstance(topology, "Topology"):
            print("Topology.RemoveCoplanarFaces - Error: The input topology parameter is not a valid topologic topology. Returning None.")
            return None
        t = Topology.Type(topology)
        if (t == Topology.TypeID("Vertex")) or (t == Topology.TypeID("Edge")) or (t == Topology.TypeID("Wire")) or (t == Topology.TypeID("Face")):
            return topology

        def faces_on_same_plane(face1, face2, epsilon=1e-6):
            vertices = Topology.Vertices(face1)
            distances = []
            for v in vertices:
                distances.append(Vertex.PerpendicularDistance(v, face=face2, mantissa=6))
            d = sum(distances)/len(distances)
            return d <= epsilon

        def cluster_faces_on_planes(faces, epsilon=1e-6):

            # Create a dictionary to store bins based on plane equations
            bins = {}

            # Iterate through each face
            for i, face in enumerate(faces):
                # Check if a bin already exists for the plane equation
                found_bin = False
                for bin_face in bins.values():
                    if faces_on_same_plane(face, bin_face[0], epsilon=epsilon):
                        bin_face.append(face)
                        found_bin = True
                        break

                # If no bin is found, create a new bin
                if not found_bin:
                    bins[i] = [face]

            # Convert bins to a list of lists
            return list(bins.values())

        faces = Topology.Faces(topology)
        face_clusters = cluster_faces_on_planes(faces, epsilon=epsilon)
        final_faces = []
        for face_cluster in face_clusters:
            t = Topology.SelfMerge(Cluster.ByTopologies(face_cluster), tolerance=tolerance)
            if Topology.IsInstance(t, "Face"):
                #final_faces.append(Face.RemoveCollinearEdges(t))
                final_faces.append(t)
            elif Topology.IsInstance(t, "Shell"):
                    f = Face.ByShell(t, silent=True)
                    if Topology.IsInstance(f, "Face"):
                        final_faces.append(f)
                    else:
                        print("Topology.RemoveCoplanarFaces - Warning: Could not remove some coplanar faces. Re-adding original faces.")
                        final_faces += Shell.Faces(t)
            else: # It is a cluster
                shells = Topology.Shells(t)
                for shell in shells:
                    f = Face.ByShell(shell)
                    if Topology.IsInstance(f, "Face"):
                        final_faces.append(f)
                    else:
                        print("Topology.RemoveCoplanarFaces - Warning: Could not remove some coplanar faces. Re-adding original faces.")
                        final_faces += Shell.Faces(shell)
                if len(shells) == 0:
                    faces = Topology.Faces(t)
                    final_faces += faces
                faces = Cluster.FreeFaces(t)
                final_faces += faces
        return_topology = None
        if Topology.IsInstance(topology, "CellComplex"):
            return_topology = CellComplex.ByFaces(final_faces, tolerance=tolerance)
        elif Topology.IsInstance(topology, "Cell"):
            return_topology = Cell.ByFaces(final_faces, tolerance=tolerance)
        elif Topology.IsInstance(topology, "Shell"):
            if len(final_faces) == 1:
                return_topology = final_faces[0]
            else:
                return_topology = Shell.ByFaces(final_faces, tolerance=tolerance)
        if not Topology.IsInstance(return_topology, "Topology"):
            return_topology = Cluster.ByTopologies(final_faces)
        return return_topology

    @staticmethod
    def RemoveEdges(topology, edges=[], tolerance=0.0001):
        """
        Removes the input list of faces from the input topology

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        edges : list
            The input list of edges.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Topology
            The input topology with the input list of edges removed.

        """

        from topologicpy.Cluster import Cluster
        if not Topology.IsInstance(topology, "Topology"):
            return None
        edges = [e for e in edges if Topology.IsInstance(e, "Edge")]
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
            return Topology.SelfMerge(Cluster.ByTopologies(remaining_edges), tolerance=tolerance)

    @staticmethod
    def RemoveFaces(topology, faces=[], tolerance=0.0001):
        """
        Removes the input list of faces from the input topology

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        faces : list
            The input list of faces.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Topology
            The input topology with the input list of faces removed.

        """

        from topologicpy.Cluster import Cluster
        if not Topology.IsInstance(topology, "Topology"):
            return None
        faces = [f for f in faces if Topology.IsInstance(f, "Face")]
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
        return Topology.SelfMerge(Cluster.ByTopologies(remaining_faces), tolerance=tolerance)
    
    @staticmethod
    def RemoveFacesBySelectors(topology, selectors=[], tolerance = 0.0001):
        """
        Removes faces that contain the input list of selectors (vertices) from the input topology

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        selectors : list
            The input list of selectors (vertices).
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Topology
            The input topology with the identified faces removed.

        """
        from topologicpy.Vertex import Vertex

        if not Topology.IsInstance(topology, "Topology"):
            return None
        selectors = [v for v in selectors if Topology.IsInstance(v, "Vertex")]
        if len(selectors) < 1:
            return topology
        t_faces = Topology.Faces(topology)
        to_remove = []
        for t_f in t_faces:
            remove = False
            for i, v in enumerate(selectors):
                if Vertex.IsInternal(v, t_f, tolerance=tolerance):
                    remove = True
                    selectors = selectors[:i]+ selectors[i:]
                    break
            if remove:
                to_remove.append(t_f)
        if len(to_remove) < 1:
            return topology
        return Topology.RemoveFaces(topology, faces = to_remove)

    @staticmethod
    def RemoveVertices(topology, vertices=[], tolerance=0.0001):
        """
        Removes the input list of vertices from the input topology

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        vertices : list
            The input list of vertices.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Topology
            The input topology with the input list of vertices removed.

        """

        from topologicpy.Cluster import Cluster
        if not Topology.IsInstance(topology, "Topology"):
            return None
        vertices = [v for v in vertices if Topology.IsInstance(v, "Vertex")]
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
            return Topology.SelfMerge(Cluster.ByTopologies(remaining_vertices), tolerance=tolerance)
    
    @staticmethod
    def Cleanup(topology=None):
        """
        Cleans up all resources in which are managed by topologic library. Use this to manage your application's memory consumption.
        USE WITH CARE. This methods deletes dictionaries, contents, and contexts

        Parameters
        ----------
        topology : topologic_core.Topology , optional
            If specified the resources used by the input topology will be deleted. If not, ALL resources will be deleted.
        
        Returns
        -------
        topologic_core.Topology
            The input topology, but with its resources deleted or None.
        """
        if not topology == None:
            if not Topology.IsInstance(topology, "Topology"):
                print("Topology.Cleanup - Error: The input topology parameter is not a valid topology. Returning None.")
                return None
        topologic.Topology.Cleanup(topology)
        return topology

    @staticmethod
    def ReplaceVertices(topology, verticesA: list = [], verticesB: list = [], mantissa: int = 6, tolerance: float = 0.0001):
        """
        Replaces the vertices in the first input list with the vertices in the second input list and rebuilds the input topology. The two lists must be of the same length.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        verticesA : list
            The first input list of vertices.
        verticesB : list
            The second input list of vertices.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Topology
            The new topology.

        """
        if not len(verticesA) == len(verticesB):
            print("Topology.ReplaceVertices - Error: The input parameters verticesA and verticesB must be the same length")
            return None
        from topologicpy.Vertex import Vertex
        geom = Topology.Geometry(topology, mantissa=mantissa)
        g_verts = geom['vertices']
        g_edges = geom['edges']
        g_faces = geom['faces']
        verts = [Topology.Vertices(Topology.ByGeometry(vertices=[g_v]))[0] for g_v in g_verts]
        new_verts = [v for v in verts]
        for i, v in enumerate(verticesA):
            n = Vertex.Index(v, verts, tolerance=tolerance)
            if not n == None:
                new_verts[n] = verticesB[i]
        new_g_verts = [[Vertex.X(v, mantissa=mantissa),Vertex.Y(v, mantissa=mantissa),Vertex.Z(v, mantissa=mantissa)] for v in new_verts]
        new_topology = Topology.ByGeometry(vertices=new_g_verts, edges=g_edges, faces=g_faces)
        return new_topology

    @staticmethod
    def Rotate(topology, origin=None, axis: list = [0, 0, 1], angle: float = 0, angTolerance: float = 0.001, tolerance: float = 0.0001):
        """
        Rotates the input topology

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        origin : topologic_core.Vertex , optional
            The origin (center) of the rotation. If set to None, the world origin (0, 0, 0) is used. The default is None.
        axis : list , optional
            The vector representing the axis of rotation. The default is [0, 0, 1] which equates to the Z axis.
        angle : float , optional
            The angle of rotation in degrees. The default is 0.
        angTolerance : float , optional
            The angle tolerance in degrees under which no rotation is carried out. The default is 0.001 degrees.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Topology
            The rotated topology.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Dictionary import Dictionary

        def rotate_vertex_3d(vertex, axis, angle_degrees, origin):
            vertex = np.array(vertex)  # Vertex to be rotated
            axis = np.array(axis)    # Rotation axis (z-axis in this case)
            origin = np.array(origin)
            # Convert the angle from degrees to radians
            angle_radians = np.radians(angle_degrees)
            
            # Calculate the rotation matrix using the Rodrigues' formula
            axis = np.array(axis) / np.linalg.norm(axis)
            a = np.cos(angle_radians / 2)
            b, c, d = -axis * np.sin(angle_radians / 2)
            rotation_matrix = np.array([
                [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)],
                [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d]
            ])
            
            # Translate the vertex to the origin, apply the rotation, and then translate it back
            translated_vertex = vertex - origin
            rotated_vertex = np.dot(rotation_matrix, translated_vertex) + origin
            
            rotated_vertex = [v for v in rotated_vertex] 
            return rotated_vertex

        if not Topology.IsInstance(topology, "Topology"):
            print("Topology.Rotate - Error: The input topology parameter is not a valid topologic topology. Returning None.")
            return None
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        if not Topology.IsInstance(origin, "Vertex"):
            print("Topology.Rotate - Error: The input origin parameter is not a valid topologic vertex. Returning None.")
            return None
        return_topology = topology
        d = Topology.Dictionary(topology)
        if abs(angle) >= angTolerance:
            try:
                x, y, z = axis
                return_topology = topologic.TopologyUtility.Rotate(topology, origin, x, y, z, angle)
            except:
                print("Topology.Rotate - Warning: (topologic.TopologyUtility.Rotate) operation failed. Trying a workaround.")
                vertices = [Vertex.Coordinates(v) for v in Topology.Vertices(topology)]
                origin = Vertex.Coordinates(origin)
                rot_vertices = []
                for v in vertices:
                    rot_vertices.append(rotate_vertex_3d(v, axis, angle, origin))
                rot_vertices = [Vertex.ByCoordinates(rot_v) for rot_v in rot_vertices]
                return_topology = Topology.ReplaceVertices(topology, verticesA=Topology.Vertices(topology), verticesB=rot_vertices)
                return_topology = Topology.SelfMerge(return_topology, tolerance=tolerance)
        if len(Dictionary.Keys(d)) > 0:
                    return_topology = Topology.SetDictionary(return_topology, d)

        vertices = Topology.Vertices(topology)
        edges = Topology.Edges(topology)
        wires = Topology.Wires(topology)
        faces = Topology.Faces(topology)
        shells = Topology.Shells(topology)
        cells = Topology.Cells(topology)
        cellComplexes = Topology.CellComplexes(topology)
        
        r_vertices = Topology.Vertices(return_topology)
        r_edges = Topology.Edges(return_topology)
        r_wires = Topology.Wires(return_topology)
        r_faces = Topology.Faces(return_topology)
        r_shells = Topology.Shells(return_topology)
        r_cells = Topology.Cells(return_topology)
        r_cellComplexes = Topology.CellComplexes(return_topology)

        for i, t in enumerate(r_vertices):
            t = Topology.SetDictionary(t, Topology.Dictionary(vertices[i]), silent=True)
        for i, t in enumerate(r_edges):
            t = Topology.SetDictionary(t, Topology.Dictionary(edges[i]), silent=True)
        for i, t in enumerate(r_wires):
            t = Topology.SetDictionary(t, Topology.Dictionary(wires[i]), silent=True)
        for i, t in enumerate(r_faces):
            t = Topology.SetDictionary(t, Topology.Dictionary(faces[i]), silent=True)
        for i, t in enumerate(r_shells):
            t = Topology.SetDictionary(t, Topology.Dictionary(shells[i]), silent=True)
        for i, t in enumerate(r_cells):
            t = Topology.SetDictionary(t, Topology.Dictionary(cells[i]), silent=True)
        for i, t in enumerate(r_cellComplexes):
            t = Topology.SetDictionary(t, Topology.Dictionary(cellComplexes[i]), silent=True)
        
        return return_topology
    
    @staticmethod
    def RotateByEulerAngles(topology, origin = None, roll: float = 0, pitch: float = 0, yaw: float = 0,  angTolerance: float = 0.001, tolerance: float = 0.0001):
        """
        Rotates the input topology using Euler angles (roll, pitch, yaw). See https://en.wikipedia.org/wiki/Aircraft_principal_axes

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        origin : topologic_core.Vertex , optional
            The origin (center) of the rotation. If set to None, the world origin (0, 0, 0) is used. The default is None.
        roll : float , optional
            The rotation angle in degrees around the X-axis. The default is 0.
        pitch = float , optional
            The rotation angle in degrees around the Y-axis. The default is 0.
        yaw = float , optional
            The rotation angle in degrees around the Z-axis. The default is 0.
        angTolerance : float , optional
            The angle tolerance in degrees under which no rotation is carried out. The default is 0.001 degrees.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Topology
            The rotated topology.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Dictionary import Dictionary

        if not Topology.IsInstance(topology, "Topology"):
            print("Topology.RotateByEulerAngles - Error: The input topology parameter is not a valid topologic topology. Returning None.")
            return None
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        if not Topology.IsInstance(origin, "Vertex"):
            print("Topology.RotateByEulerAngles - Error: The input origin parameter is not a valid topologic vertex. Returning None.")
            return None
        d = Topology.Dictionary(topology)
        return_topology = Topology.Copy(topology)
        return_topology = Topology.Rotate(return_topology, origin=origin, axis=[1,0,0], angle=roll, angTolerance=angTolerance, tolerance=tolerance)
        return_topology = Topology.Rotate(return_topology, origin=origin, axis=[0,1,0], angle=pitch, angTolerance=angTolerance, tolerance=tolerance)
        return_topology = Topology.Rotate(return_topology, origin=origin, axis=[0,0,1], angle=yaw, angTolerance=angTolerance, tolerance=tolerance)
        if len(Dictionary.Keys(d)) > 0:
            return_topology = Topology.SetDictionary(return_topology, d)
        return return_topology
    
    @staticmethod
    def RotateByQuaternion(topology, origin=None, quaternion: list = [0,0,0,1], angTolerance: float = 0.001, tolerance: float = 0.0001):
        """
        Rotates the input topology using Quaternion rotations. See https://en.wikipedia.org/wiki/Quaternion
        
        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        origin : topologic_core.Vertex , optional
            The origin (center) of the rotation. If set to None, the world origin (0, 0, 0) is used. The default is None.
        quaternion : list or numpy array of size 4
            The input Quaternion list. It should be in the form [x, y, z, w].
        angTolerance : float , optional
            The angle tolerance in degrees under which no rotation is carried out. The default is 0.001 degrees.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Topology
            The rotated topology.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Dictionary import Dictionary

        def quaternion_to_euler(quaternion):
            """
            Convert a quaternion into Euler angles (roll, pitch, yaw)
            Roll is rotation around x-axis, Pitch is rotation around y-axis, and Yaw is rotation around z-axis.
            Quaternion should be in the form [x, y, z, w]

            Args:
            quaternion: list or numpy array of size 4

            Returns:
            A list of Euler angles in degrees [roll, pitch, yaw]
            """
            import numpy as np
            x, y, z, w = quaternion

            # Roll (x-axis rotation)
            sinr_cosp = 2 * (w * x + y * z)
            cosr_cosp = 1 - 2 * (x * x + y * y)
            roll = np.arctan2(sinr_cosp, cosr_cosp)

            # Pitch (y-axis rotation)
            sinp = 2 * (w * y - z * x)
            if abs(sinp) >= 1:
                pitch = np.sign(sinp) * np.pi / 2  # use 90 degrees if out of range
            else:
                pitch = np.arcsin(sinp)

            # Yaw (z-axis rotation)
            siny_cosp = 2 * (w * z + x * y)
            cosy_cosp = 1 - 2 * (y * y + z * z)
            yaw = np.arctan2(siny_cosp, cosy_cosp)

            # Convert radians to degrees
            roll = np.degrees(roll)
            pitch = np.degrees(pitch)
            yaw = np.degrees(yaw)
            return [roll, pitch, yaw]
        
        if not Topology.IsInstance(topology, "Topology"):
            print("Topology.RotateByQuaternion - Error: The input topology parameter is not a valid topologic topology. Returning None.", topology)
            return None
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        if not Topology.IsInstance(origin, "Vertex"):
            print("Topology.RotateByQuaternion - Error: The input origin parameter is not a valid topologic vertex. Returning None.")
            return None
        roll, pitch, yaw = quaternion_to_euler(quaternion)
        d = Topology.Dictionary(topology)
        return_topology = Topology.RotateByEulerAngles(topology=topology, origin=origin, roll=roll, pitch=pitch, yaw=yaw,  angTolerance=angTolerance, tolerance=tolerance)
        if len(Dictionary.Keys(d)) > 0:
            return_topology = Topology.SetDictionary(return_topology, d)
        return return_topology
    
    @staticmethod
    def Scale(topology, origin=None, x=1, y=1, z=1):
        """
        Scales the input topology

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        origin : topologic_core.Vertex , optional
            The origin (center) of the scaling. If set to None, the world origin (0, 0, 0) is used. The default is None.
        x : float , optional
            The 'x' component of the scaling factor. The default is 1.
        y : float , optional
            The 'y' component of the scaling factor. The default is 1.
        z : float , optional
            The 'z' component of the scaling factor. The default is 1..

        Returns
        -------
        topologic_core.Topology
            The scaled topology.

        """
        
        from topologicpy.Vertex import Vertex

        if not Topology.IsInstance(topology, "Topology"):
            return None
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        if not Topology.IsInstance(origin, "Vertex"):
            return None
        return_topology = None
        try:
            return_topology = topologic.TopologyUtility.Scale(topology, origin, x, y, z)
        except:
            print("Topology.Scale - ERROR: (Topologic>TopologyUtility.Scale) operation failed. Returning None.")
            return_topology = None
        
        vertices = Topology.Vertices(topology)
        edges = Topology.Edges(topology)
        wires = Topology.Wires(topology)
        faces = Topology.Faces(topology)
        shells = Topology.Shells(topology)
        cells = Topology.Cells(topology)
        cellComplexes = Topology.CellComplexes(topology)
        
        r_vertices = Topology.Vertices(return_topology)
        r_edges = Topology.Edges(return_topology)
        r_wires = Topology.Wires(return_topology)
        r_faces = Topology.Faces(return_topology)
        r_shells = Topology.Shells(return_topology)
        r_cells = Topology.Cells(return_topology)
        r_cellComplexes = Topology.CellComplexes(return_topology)

        for i, t in enumerate(r_vertices):
            t = Topology.SetDictionary(t, Topology.Dictionary(vertices[i]), silent=True)
        for i, t in enumerate(r_edges):
            t = Topology.SetDictionary(t, Topology.Dictionary(edges[i]), silent=True)
        for i, t in enumerate(r_wires):
            t = Topology.SetDictionary(t, Topology.Dictionary(wires[i]), silent=True)
        for i, t in enumerate(r_faces):
            t = Topology.SetDictionary(t, Topology.Dictionary(faces[i]), silent=True)
        for i, t in enumerate(r_shells):
            t = Topology.SetDictionary(t, Topology.Dictionary(shells[i]), silent=True)
        for i, t in enumerate(r_cells):
            t = Topology.SetDictionary(t, Topology.Dictionary(cells[i]), silent=True)
        for i, t in enumerate(r_cellComplexes):
            t = Topology.SetDictionary(t, Topology.Dictionary(cellComplexes[i]), silent=True)
        return return_topology

    
    @staticmethod
    def SelectSubTopology(topology, selector, subTopologyType="vertex"):
        """
        Returns the subtopology within the input topology based on the input selector and the subTopologyType.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        selector : topologic_core.Vertex
            A vertex located on the desired subtopology.
        subTopologyType : str , optional.
            The desired subtopology type. This can be of "vertex", "edge", "wire", "face", "shell", "cell", or "cellcomplex". It is case insensitive. The default is "vertex".

        Returns
        -------
        topologic_core.Topology
            The selected subtopology.

        """
        if not Topology.IsInstance(topology, "Topology"):
            return None
        if not Topology.IsInstance(selector, "Vertex"):
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
    def SelfMerge(topology, transferDictionaries: bool = False, tolerance: float = 0.0001):
        """
        Self merges the input topology to return the most logical topology type given the input data.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001

        Returns
        -------
        topologic_core.Topology
            The self-merged topology.

        """
        from topologicpy.Cluster import Cluster

        if not Topology.IsInstance(topology, "Topology"):
            return None #return Silently
        if not Topology.Type(topology) == Topology.TypeID("Cluster"):
            topology = Cluster.ByTopologies([topology])
        resultingTopologies = []
        topCC = Topology.CellComplexes(topology)
        topCells = Topology.Cells(topology)
        topShells = Topology.Shells(topology)
        topFaces = Topology.Faces(topology)
        topWires = Topology.Wires(topology)
        topEdges = Topology.Edges(topology)
        topVertices = Topology.Vertices(topology)
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
        try:
            return_topology = topology.SelfMerge()
        except:
            return_topology = None
        if Topology.IsInstance(return_topology, "CellComplex"):
            cells = Topology.Cells(return_topology)
            if isinstance(cells, list):
                if len(cells) > 1:
                    topA = cells[0]
                    topB = Cluster.ByTopologies(cells[1:])
                    return_topology = Topology.Merge(topA, topB, tolerance=tolerance)
                else:
                    return_topology = cells[0]
        return return_topology

    @staticmethod
    def SetDictionary(topology, dictionary, silent=False):
        """
        Sets the input topology's dictionary to the input dictionary

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        dictionary : topologic_core.Dictionary
            The input dictionary.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        topologic_core.Topology
            The input topology with the input dictionary set in it.

        """
        from topologicpy.Dictionary import Dictionary
        import inspect

        if not Topology.IsInstance(topology, "Topology") and not Topology.IsInstance(topology, "Graph"):
            if not silent:
                print("Topology.SetDictionary - Error: the input topology parameter is not a valid topology or graph. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
            return None
        if isinstance(dictionary, dict):
            dictionary = Dictionary.ByPythonDictionary(dictionary)
        if not Topology.IsInstance(dictionary, "Dictionary"):
            if not silent:
                print("Topology.SetDictionary - Warning: the input dictionary parameter is not a valid dictionary. Returning original input.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
            return topology
        if len(dictionary.Keys()) < 1:
            if not silent:
                print("Topology.SetDictionary - Warning: the input dictionary parameter is empty. Returning original input.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
            return topology
        _ = topology.SetDictionary(dictionary)
        return topology
    
    @staticmethod
    def SetSnapshot(topology, snapshot=None, timestamp=None, key="timestamp", silent=False):
        from topologicpy.Dictionary import Dictionary
        from datetime import datetime
        def is_valid_timestamp(timestamp):
            if isinstance(timestamp, datetime):
                return True
            elif isinstance(timestamp, str):
                try:
                    # Split the timestamp string into date and time parts
                    date_part, time_part = timestamp.split(' ')
                    # Parse the date part
                    date_obj = datetime.strptime(date_part, '%Y-%m-%d')
                    # Split the time part into hours, minutes, and seconds
                    hours, minutes, seconds = map(float, time_part.split(':'))
                    # Check if seconds are within valid range
                    if seconds < 0 or seconds >= 60:
                        return False
                    # Create a datetime object with the parsed date and time parts
                    datetime_obj = datetime(date_obj.year, date_obj.month, date_obj.day, int(hours), int(minutes), int(seconds))
                    return True
                except ValueError:
                    return False
            else:
                return False

        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print("Topology.SetSnapshot - Error: The input topology parameter is not a valid topology. Returning None.")
                return None
        if not Topology.IsInstance(snapshot, "Topology"):
            snapshot = Topology.Copy(topology)
        if not Topology.IsInstance(snapshot, "Topology"):
            if not silent:
                print("Topology.SetSnapshot - Error: The input snapshot parameter is not a valid topology. Returning None.")
                return None
        if timestamp == None:
            timestamp = datetime.now()
        if not is_valid_timestamp(timestamp):
            if not silent:
                print("Topology.SetSnapshot - Error: The input timestamp parameter is not a valid timestamp. Returning None.")
                return None
        
        d = Topology.Dictionary(snapshot)
        d = Dictionary.SetValueAtKey(d, key, str(timestamp))
        snapshot = Topology.SetDictionary(snapshot, d)
        topology = Topology.AddContent(topology, snapshot)
        return topology
    
    @staticmethod
    def SharedTopologies(topologyA, topologyB):
        """
        Returns the shared topologies between the two input topologies

        Parameters
        ----------
        topologyA : topologic_core.Topology
            The first input topology.
        topologyB : topologic_core.Topology
            The second input topology.

        Returns
        -------
        dict
            A dictionary with the list of vertices, edges, wires, and faces. The keys are "vertices", "edges", "wires", and "faces".

        """
        if not Topology.IsInstance(topologyA, "Topology"):
            print("Topology.SharedTopologies - Error: the input topologyA parameter is not a valid topology. Returning None.")
            return None
        if not Topology.IsInstance(topologyB, "Topology"):
            print("Topology.SharedTopologies - Error: the input topologyB parameter is not a valid topology. Returning None.")
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
        topologyA : topologic_core.Topology
            The first input topology.
        topologyB : topologic_core.Topology
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
        topologyA : topologic_core.Topology
            The first input topology.
        topologyB : topologic_core.Topology
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
        topologyA : topologic_core.Topology
            The first input topology.
        topologyB : topologic_core.Topology
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
        topologyA : topologic_core.Topology
            The first input topology.
        topologyB : topologic_core.Topology
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
    def Show(*topologies,
             nameKey = "name",
             opacityKey = "opacity",
             showVertices=True,
             vertexSize=None,
             vertexSizeKey = None,
             vertexColor="black",
             vertexColorKey = None,
             vertexLabelKey=None,
             showVertexLabel= False,
             vertexGroupKey=None,
             vertexGroups=[], 
             vertexMinGroup=None,
             vertexMaxGroup=None, 
             showVertexLegend=False,
             vertexLegendLabel="Vertices",

             showEdges=True,
             edgeWidth=None,
             edgeWidthKey = None,
             edgeColor=None,
             edgeColorKey = None,
             edgeLabelKey=None,
             showEdgeLabel = False,
             edgeGroupKey=None,
             edgeGroups=[], 
             edgeMinGroup=None,
             edgeMaxGroup=None, 
             showEdgeLegend=False,
             edgeLegendLabel="Edges",

             showFaces=True,
             faceOpacity=0.5,
             faceOpacityKey=None,
             faceColor="#FAFAFA",
             faceColorKey = None,
             faceLabelKey=None,
             faceGroupKey=None,
             faceGroups=[], 
             faceMinGroup=None,
             faceMaxGroup=None, 
             showFaceLegend=False,
             faceLegendLabel="Faces",
             intensityKey=None,
             intensities=[],
             
             width=950,
             height=500,
             xAxis=False,
             yAxis=False,
             zAxis=False,
             axisSize=1, 
             backgroundColor='rgba(0,0,0,0)',
             marginLeft=0,
             marginRight=0,
             marginTop=20,
             marginBottom=0,
             camera=[-1.25, -1.25, 1.25],
             center=[0, 0, 0],
             up=[0, 0, 1],
             projection="perspective",
             renderer="notebook",
             showScale=False,
             
             cbValues=[],
             cbTicks=5,
             cbX=-0.15,
             cbWidth=15,
             cbOutlineWidth=0,
             cbTitle="",
             cbSubTitle="",
             cbUnits="",
             colorScale="Viridis",
             
             sagitta = 0,
             absolute = False,
             sides = 8,
             angle = 0,
             mantissa=6,
             tolerance=0.0001,
             silent=False):
        """
            Shows the input topology on screen.

        Parameters
        ----------
        topologies : topologic_core.Topology or list
            The input topology. This must contain faces and or edges. If the input is a list, a cluster is first created
        opacityKey : str , optional
            The key under which to find the opacity of the topology. The default is "opacity".
        showVertices : bool , optional
            If set to True the vertices will be drawn. Otherwise, they will not be drawn. The default is True.
        vertexSize : float , optional
            The desired size of the vertices. The default is 1.1.
        vertexSizeKey : str , optional
            The key under which to find the size of the vertex. The default is None.
        vertexColor : str , optional
            The desired color of the output vertices. This can be any plotly color string and may be specified as:
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is "black".
        vertexColorKey : str , optional
            The key under which to find the color of the vertex. The default is None.
        vertexLabelKey : str , optional
            The dictionary key to use to display the vertex label. The default is None.
        showVertexLabels : bool , optional
            If set to True, the vertex labels are shown permenantely on screen. Otherwise, they are not. The default is False.
        vertexGroupKey : str , optional
            The dictionary key to use to display the vertex group. The default is None.
        vertexGroups : list , optional
            The list of vertex groups against which to index the color of the vertex. The default is [].
        vertexMinGroup : int or float , optional
            For numeric vertexGroups, vertexMinGroup is the desired minimum value for the scaling of colors. This should match the type of value associated with the vertexGroupKey. If set to None, it is set to the minimum value in vertexGroups. The default is None.
        vertexMaxGroup : int or float , optional
            For numeric vertexGroups, vertexMaxGroup is the desired maximum value for the scaling of colors. This should match the type of value associated with the vertexGroupKey. If set to None, it is set to the maximum value in vertexGroups. The default is None.
        showVertexLegend : bool, optional
            If set to True, the legend for the vertices of this topology is shown. Otherwise, it isn't. The default is False.
        vertexLegendLabel : str , optional
            The legend label string used to identify vertices. The default is "Topology Vertices".
        
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
        edgeColorKey : str , optional
            The key under which to find the color of the edge. The default is None.
        edgeWidthKey : str , optional
            The key under which to find the width of the edge. The default is None.
        edgeLabelKey : str , optional
            The dictionary key to use to display the edge label. The default is None.
        showEdgeLabels : bool , optional
            If set to True, the edge labels are shown permenantely on screen. Otherwise, they are not. The default is False.
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
        
        showFaces : bool , optional
            If set to True the faces will be drawn. Otherwise, they will not be drawn. The default is True.
        faceOpacity : float , optional
            The desired opacity of the output faces (0=transparent, 1=opaque). The default is 0.5.
        faceOpacityKey : str , optional
            The key under which to find the opacity of the face. The default is None.
        faceColor : str , optional
            The desired color of the output faces. This can be any plotly color string and may be specified as:
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is "#FAFAFA".
        faceColorKey : str , optional
            The key under which to find the color of the face. The default is None.
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
            The desired location of the camera). The default is [-1.25, -1.25, 1.25].
        center : list , optional
            The desired center (camera target). The default is [0, 0, 0].
        up : list , optional
            The desired up vector. The default is [0, 0, 1].
        projection : str , optional
            The desired type of projection. The options are "orthographic" or "perspective". It is case insensitive. The default is "perspective"
        renderer : str , optional
            The desired renderer. See Plotly.Renderers(). If set to None, the code will attempt to discover the most suitable renderer. The default is None.
        intensityKey : str , optional
            If not None, the dictionary of each vertex is searched for the value associated with the intensity key. This value is then used to color-code the vertex based on the colorScale. The default is None.
        intensities : list , optional
            The list of intensities against which to index the intensity of the vertex. The default is [].
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
            The desired length of the mantissa for the values listed on the colorbar. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        None

        """

        from topologicpy.Dictionary import Dictionary
        from topologicpy.Plotly import Plotly
        from topologicpy.Helper import Helper
        
        if isinstance(topologies, tuple):
            topologies = Helper.Flatten(list(topologies))
        if isinstance(topologies, list):
            new_topologies = [t for t in topologies if Topology.IsInstance(t, "Topology") or Topology.IsInstance(t, "Graph")]
        if len(new_topologies) == 0:
            if not silent:
                print("Topology.Show - Error: the input topologies parameter does not contain any valid topology. Returning None.")
            return None
        
        if camera[0] == 0 and camera[1] == 0 and up == [0,0,1]:
            up = [0,1,0] #default to positive Y axis being up if looking down or up at the XY plane
        data = []
        topology_counter = 0
        offset = 1
        if showEdges == True:
            offset += 1
        if showFaces == True:
            offset +=2
        temp_graphs = [g for g in new_topologies if Topology.IsInstance(g, "Graph")]
        graph_counter = len(new_topologies)*offset - len(temp_graphs)*offset
        for i, topology in enumerate(new_topologies):
            d = Topology.Dictionary(topology)
            if Topology.IsInstance(topology, "Graph"):
                name = Dictionary.ValueAtKey(d, nameKey) or "Untitled Graph"
                if vertexSize == None:
                    vSize = 10
                else:
                    vSize = vertexSize
                if edgeWidth == None:
                    eWidth = 1
                else:
                    eWidth = edgeWidth
                if edgeColor == None:
                    eColor = "red"
                else:
                    eColor = edgeColor
                vll = name+" ("+vertexLegendLabel+")"
                ell = name+" ("+edgeLegendLabel+")"
                
                data += Plotly.DataByGraph(topology,
                                           sagitta=sagitta,
                                           absolute=absolute,
                                           sides=sides,
                                           angle=angle,
                                           vertexColor=vertexColor,
                                           vertexColorKey=vertexColorKey,
                                           vertexSize=vSize,
                                           vertexSizeKey=vertexSizeKey,
                                           vertexLabelKey=vertexLabelKey,
                                           vertexGroupKey=vertexGroupKey,
                                           vertexGroups=vertexGroups,
                                           vertexMinGroup=vertexMinGroup,
                                           vertexMaxGroup=vertexMaxGroup,
                                           showVertices=showVertices,
                                           showVertexLabel=showVertexLabel,
                                           showVertexLegend=showVertexLegend,
                                           vertexLegendLabel= str(i+1)+": "+vll,
                                           vertexLegendRank= (graph_counter+1),
                                           vertexLegendGroup= (graph_counter+1),
                                           edgeColor=eColor,
                                           edgeColorKey=edgeColorKey,
                                           edgeWidth=eWidth,
                                           edgeWidthKey=edgeWidthKey,
                                           edgeLabelKey=edgeLabelKey,
                                           edgeGroupKey=edgeGroupKey,
                                           edgeGroups=edgeGroups,
                                           edgeMinGroup=edgeMinGroup,
                                           edgeMaxGroup=edgeMaxGroup,
                                           showEdges=showEdges,
                                           showEdgeLabel=showEdgeLabel,
                                           showEdgeLegend=showEdgeLegend,
                                           edgeLegendLabel = str(i+1)+": "+ell,
                                           edgeLegendRank= (graph_counter+2),
                                           edgeLegendGroup=(graph_counter+2),
                                           colorScale=colorScale,
                                           silent=silent)
                graph_counter += offset
            else:
                name = Dictionary.ValueAtKey(d, nameKey) or "Untitled"
                if vertexSize == None:
                    vSize = 1.1
                else:
                    vSize = vertexSize
                if edgeWidth == None:
                    eWidth = 1
                else:
                    eWidth = edgeWidth
                if edgeColor == None:
                    eColor = "black"
                else:
                    eColor = edgeColor
                if not d == None:
                    faceOpacity = Dictionary.ValueAtKey(d, opacityKey) or faceOpacity
                data += Plotly.DataByTopology(topology=topology,
                                              showVertices=showVertices,
                                              vertexSize=vSize,
                                              vertexSizeKey=vertexSizeKey,
                                              vertexColor=vertexColor,
                                              vertexColorKey=vertexColorKey,
                                              vertexLabelKey=vertexLabelKey,
                                              showVertexLabel=showVertexLabel,
                                              vertexGroupKey=vertexGroupKey,
                                              vertexGroups=vertexGroups,
                                              vertexMinGroup=vertexMinGroup,
                                              vertexMaxGroup=vertexMaxGroup,
                                              showVertexLegend=showVertexLegend,
                                              vertexLegendLabel=str(i+1)+": "+name+" ("+vertexLegendLabel+")",
                                              vertexLegendRank=topology_counter+1,
                                              vertexLegendGroup=topology_counter+1,
                                              showEdges=showEdges,
                                              edgeWidth=eWidth,
                                              edgeWidthKey=edgeWidthKey,
                                              edgeColor=eColor,
                                              edgeColorKey=edgeColorKey,
                                              edgeLabelKey=edgeLabelKey,
                                              showEdgeLabel=showEdgeLabel,
                                              edgeGroupKey=edgeGroupKey,
                                              edgeGroups=edgeGroups,
                                              edgeMinGroup=edgeMinGroup,
                                              edgeMaxGroup=edgeMaxGroup,
                                              showEdgeLegend=showEdgeLegend,
                                              edgeLegendLabel=str(i+1)+": "+name+" ("+edgeLegendLabel+")",
                                              edgeLegendRank=topology_counter+2,
                                              edgeLegendGroup=topology_counter+2,
                                              showFaces=showFaces,
                                              faceOpacity=faceOpacity,
                                              faceOpacityKey=faceOpacityKey,
                                              faceColor=faceColor,
                                              faceColorKey=faceColorKey,
                                              faceLabelKey=faceLabelKey,
                                              faceGroupKey=faceGroupKey,
                                              faceGroups=faceGroups,
                                              faceMinGroup=faceMinGroup,
                                              faceMaxGroup=faceMaxGroup,
                                              showFaceLegend=showFaceLegend,
                                              faceLegendLabel=str(i+1)+": "+name+" ("+faceLegendLabel+")",
                                              faceLegendRank=topology_counter+3,
                                              faceLegendGroup=topology_counter+3,
                                              intensityKey=intensityKey,
                                              intensities=intensities,
                                              colorScale=colorScale,
                                              mantissa=mantissa,
                                              tolerance=tolerance)
                topology_counter += offset
        figure = Plotly.FigureByData(data=data, width=width, height=height,
                                     xAxis=xAxis, yAxis=yAxis, zAxis=zAxis, axisSize=axisSize,
                                     backgroundColor=backgroundColor,
                                     marginLeft=marginLeft, marginRight=marginRight,
                                     marginTop=marginTop, marginBottom=marginBottom,
                                     tolerance=tolerance)
        if showScale:
            figure = Plotly.AddColorBar(figure, values=cbValues, nTicks=cbTicks, xPosition=cbX, width=cbWidth, outlineWidth=cbOutlineWidth, title=cbTitle, subTitle=cbSubTitle, units=cbUnits, colorScale=colorScale, mantissa=mantissa)
        Plotly.Show(figure=figure, renderer=renderer, camera=camera, center=center, up=up, projection=projection)

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
        from topologicpy.Vertex import Vertex

        usedTopologies = []
        sortedTopologies = []
        unsortedTopologies = []
        for i in range(len(topologies)):
            usedTopologies.append(0)
        
        for i in range(len(selectors)):
            found = False
            for j in range(len(topologies)):
                if usedTopologies[j] == 0:
                    if Vertex.IsInternal( selectors[i], topologies[j], tolerance=tolerance):
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
    def Snapshots(topology, key="timestamp", start=None, end=None, silent=False):
        from topologicpy.Dictionary import Dictionary
        from datetime import datetime
        def is_valid_timestamp(timestamp):
            if isinstance(timestamp, datetime):
                return True
            elif isinstance(timestamp, str):
                try:
                    # Split the timestamp string into date and time parts
                    date_part, time_part = timestamp.split(' ')
                    # Parse the date part
                    date_obj = datetime.strptime(date_part, '%Y-%m-%d')
                    # Split the time part into hours, minutes, and seconds
                    hours, minutes, seconds = map(float, time_part.split(':'))
                    # Check if seconds are within valid range
                    if seconds < 0 or seconds >= 60:
                        return False
                    # Create a datetime object with the parsed date and time parts
                    return datetime(date_obj.year, date_obj.month, date_obj.day, int(hours), int(minutes), int(seconds))
                except ValueError:
                    return False
            else:
                return False
        
        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print("Topology.Snapshots - Error: The input topology parameter is not a valid topology. Returning None.")
                return None
        if start == None:
            start = datetime.datetime(year=1900, month=1, day=1) # Set the start date to a date in the distant past
        if end == None:
            end = datetime.now() # Set the end date to the present.
        if not is_valid_timestamp(start):
            if not silent:
                print("Topology.Snapshots - Error: The input start parameter is not a valid timestamp. Returning None.")
                return None
        if not is_valid_timestamp(end):
            if not silent:
                print("Topology.Snapshots - Error: The input end parameter is not a valid timestamp. Returning None.")
                return None
        contents = Topology.Contents(topology)    
        snapshots = []
        for content in contents:
            d = Topology.Dictionary(content)
            timestamp = Dictionary.ValueAtKey(d, key)
            timestamp = is_valid_timestamp(timestamp)
            if not timestamp == False:
                if start <= timestamp <= end:
                    snapshots.append(content)
        return snapshots

    @staticmethod
    def Spin(topology, origin=None, triangulate: bool = True, direction: list = [0, 0, 1], angle: float = 360, sides: int = 16,
                     tolerance: float = 0.0001, silent: bool = False):
        """
        Spins the input topology around an axis to create a new topology.See https://en.wikipedia.org/wiki/Solid_of_revolution.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        origin : topologic_core.Vertex
            The origin (center) of the spin.
        triangulate : bool , optional
            If set to True, the result will be triangulated. The default is True.
        direction : list , optional
            The vector representing the direction of the spin axis. The default is [0, 0, 1].
        angle : float , optional
            The angle in degrees for the spin. The default is 360.
        sides : int , optional
            The desired number of sides. The default is 16.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Topology
            The spun topology.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Shell import Shell
        from topologicpy.Cell import Cell
        from topologicpy.CellComplex import CellComplex
        from topologicpy.Cluster import Cluster

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print("Topology.Spin - Error: the input topology parameter is not a valid topology. Returning None.")
                return None
        if not Topology.IsInstance(origin, "Vertex"):
            if not silent:
                print("Topology.Spin - Error: the input origin parameter is not a valid vertex. Returning None.")
                return None
        topologies = []
        unit_degree = angle / float(sides)
        for i in range(sides+1):
            tempTopology = Topology.Rotate(topology, origin=origin, axis=direction, angle=unit_degree*i)
            if tempTopology:
                topologies.append(tempTopology)
        returnTopology = None
        if Topology.Type(topology) == Topology.TypeID("Vertex"):
            returnTopology = Wire.ByVertices(topologies, False)
        elif Topology.Type(topology) == Topology.TypeID("Edge"):
            try:
                returnTopology = Shell.ByWires(topologies,triangulate=triangulate, tolerance=tolerance, silent=True)
            except:
                try:
                    returnTopology = Cluster.ByTopologies(topologies)
                except:
                    returnTopology = None
        elif Topology.Type(topology) == Topology.TypeID("Wire"):
            if topology.IsClosed():
                #returnTopology = Cell.ByWires(topologies, triangulate=triangulate, tolerance=tolerance, silent=True)
                try:
                    returnTopology = Cell.ByWires(topologies, triangulate=triangulate, tolerance=tolerance, silent=True)
                    returnTopology = Cell.ExternalBoundary(returnTopology)
                    returnTopology = Cell.ByShell(returnTopology)
                except:
                    try:
                        returnTopology = CellComplex.ByWires(topologies, tolerance=tolerance)
                        try:
                            returnTopology = CellComplex.ExternalBoundary(returnTopology)
                        except:
                            pass
                    except:
                        try:
                            returnTopology = Shell.ByWires(topologies, triangulate=triangulate, tolerance=tolerance, silent=True)
                        except:
                            try:
                                returnTopology = Cluster.ByTopologies(topologies)
                            except:
                                returnTopology = None
            else:
                Shell.ByWires(topologies, triangulate=triangulate, tolerance=tolerance, silent=True)
                try:
                    returnTopology = Shell.ByWires(topologies, triangulate=triangulate, tolerance=tolerance, silent=True)
                except:
                    try:
                        returnTopology = Cluster.ByTopologies(topologies)
                    except:
                        returnTopology = None
        elif Topology.IsInstance(topology, "Face"):
            external_wires = []
            for t in topologies:
                external_wires.append(topologic.Face.ExternalBoundary(t))
            try:
                returnTopology = CellComplex.ByWires(external_wires, tolerance=tolerance)
            except:
                try:
                    returnTopology = Shell.ByWires(external_wires, triangulate=triangulate, tolerance=tolerance, silent=True)
                except:
                    try:
                        returnTopology = Cluster.ByTopologies(topologies)
                    except:
                        returnTopology = None
        else:
            returnTopology = Topology.SelfMerge(Cluster.ByTopologies(topologies), tolerance=tolerance)
        if not returnTopology:
            return Cluster.ByTopologies(topologies)
        if Topology.Type(returnTopology) == Topology.TypeID("Shell"):
            try:
                new_t = Cell.ByShell(returnTopology)
                if new_t:
                    returnTopology = new_t
            except:
                pass
        return returnTopology
    
    @staticmethod
    def Taper(topology, origin=None, ratioRange: list = [0, 1], triangulate: bool = False, mantissa: int = 6, tolerance: float = 0.0001):
        """
        Tapers the input topology. This method tapers the input geometry along its Z-axis based on the ratio range input.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        origin : topologic_core.Vertex , optional
            The desired origin for tapering. If not specified, the centroid of the input topology is used. The tapering will use the X, Y coordinates of the specified origin, but will use the Z of the point being tapered. The default is None.
        ratioRange : list , optional
            The desired ratio range. This will specify a linear range from bottom to top for tapering the vertices. 0 means no tapering, and 1 means maximum (inward) tapering. Negative numbers mean that tapering will be outwards.
        triangulate : bool , optional
            If set to true, the input topology is triangulated before tapering. Otherwise, it will not be traingulated. The default is False.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. Vertices will not be moved if the calculated distance is at or less than this tolerance.

        Returns
        -------
        topologic_core.Topology
            The tapered topology.

        """
        from topologicpy.Vertex import Vertex

        ratioRange = [min(1,ratioRange[0]), min(1,ratioRange[1])]
        if ratioRange == [0, 0]:
            return topology
        if ratioRange == [1, 1]:
            print("Topology.Taper - Error: Degenerate result. Returning original topology.")
            return topology
        if triangulate == True:
            topology = Topology.Triangulate(topology)
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Topology.Centroid(topology)
        vertices = Topology.Vertices(topology)
        zList = [Vertex.Z(v, mantissa=mantissa) for v in vertices]
        z_min = min(zList)
        maxZ = max(zList)
        new_vertices = []
        for v in vertices:
            ht = (Vertex.Z(v)-z_min)/(maxZ - z_min)
            rt = ratioRange[0] + ht*(ratioRange[1] - ratioRange[0])
            new_origin = Vertex.ByCoordinates(Vertex.X(origin, mantissa=mantissa), Vertex.Y(origin, mantissa=mantissa), Vertex.Z(v, mantissa=mantissa))
            new_dist = Vertex.Distance(new_origin, v, mantissa=mantissa)*rt
            c_a = Vertex.Coordinates(new_origin, mantissa=mantissa)
            c_b = Vertex.Coordinates(v, mantissa=mantissa)
            new_dir = [(c_a[0]-c_b[0]), (c_a[1]-c_b[1]), 0]
            if abs(new_dist) > tolerance:
                new_v = Topology.TranslateByDirectionDistance(v, direction=new_dir, distance=new_dist)
            else:
                new_v = v
            new_vertices.append(new_v)
        return_topology = Topology.ReplaceVertices(topology, vertices, new_vertices)
        return return_topology
    
    @staticmethod
    def Twist(topology, origin=None, angleRange: list = [45, 90], triangulate: bool = False, mantissa: int = 6):
        """
        Twists the input topology. This method twists the input geometry along its Z-axis based on the degree range input.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        origin : topologic_core.Vertex , optional
            The desired origin for tapering. If not specified, the centroid of the input topology is used. The tapering will use the X, Y coordinates of the specified origin, but will use the Z of the point being tapered. The default is None.
        angleRange : list , optional
            The desired angle range in degrees. This will specify a linear range from bottom to top for twisting the vertices. positive numbers mean a clockwise rotation.
        triangulate : bool , optional
            If set to true, the input topology is triangulated before tapering. Otherwise, it will not be traingulated. The default is False.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        
        Returns
        -------
        topologic_core.Topology
            The twisted topology.

        """
        from topologicpy.Vertex import Vertex

        if angleRange == [0, 0]:
            return topology
        if triangulate == True:
            topology = Topology.Triangulate(topology)
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Topology.Centroid(topology)
            
        vertices = Topology.Vertices(topology)
        zList = [Vertex.Z(v, mantissa=mantissa) for v in vertices]
        z_min = min(zList)
        maxZ = max(zList)
        h = maxZ - z_min
        new_vertices = []
        for v in vertices:
            ht = (Vertex.Z(v)-z_min)/(maxZ - z_min)
            new_rot = angleRange[0] + ht*(angleRange[1] - angleRange[0])
            orig = Vertex.ByCoordinates(Vertex.X(origin, mantissa=mantissa), Vertex.Y(origin, mantissa=mantissa), Vertex.Z(v, mantissa=mantissa))
            new_vertices.append(Topology.Rotate(v, origin=orig, axis=[0, 0, 1], angle=new_rot))
        return_topology = Topology.ReplaceVertices(topology, vertices, new_vertices)
        return_topology = Topology.Fix(return_topology, topologyType=Topology.TypeAsString(topology))
        return return_topology
    
    @staticmethod
    def Unflatten(topology, origin=None, direction=[0, 0, 1]):
        """
        Unflattens the input topology such that the world origin is translated to the input origin and the input topology is rotated such that the Up direction (see Vector.Up()) is aligned with the input vector.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        origin : topologic_core.Vertex , optional
            The input origin. If set to None, The object's centroid will be used to translate the world origin. The default is None.
        vector : list , optional
            The input direction vector. The input topology will be rotated such that this vector is pointed in the positive Z axis.

        Returns
        -------
        topologic_core.Topology
            The flattened topology.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Vector import Vector

        if not Topology.IsInstance(topology, "Topology"):
            print("Topology.Unflatten - Error: the input topology parameter is not a valid topology. Returning None.")
            return None
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.Origin()
        up = Vector.Up()
        tran_mat = Vector.TransformationMatrix(up, direction)
        unflat_topology = Topology.Transform(topology, tran_mat)
        unflat_topology = Topology.Translate(unflat_topology, Vertex.X(origin), Vertex.Y(origin), Vertex.Z(origin))

        unflat_topology = Topology.SetDictionary(unflat_topology, Topology.Dictionary(topology), silent=True)
        unflat_vertices = Topology.Vertices(unflat_topology)
        vertices = Topology.Vertices(topology)
        unflat_edges = Topology.Edges(unflat_topology)
        edges = Topology.Edges(topology)
        faces = []
        unflat_faces = []
        if Topology.IsInstance(topology, "Face"):
            unflat_faces = Topology.Faces(unflat_topology)
            faces = Topology.Faces(topology)
        elements = vertices+edges+faces
        unflat_elements = unflat_vertices+unflat_edges+unflat_faces
        for i, f, in enumerate(unflat_elements):
            f = Topology.SetDictionary(f, Topology.Dictionary(elements[i]), silent=True)
        return unflat_topology
    
    @staticmethod
    def Vertices(topology):
        """
        Returns the vertices of the input topology.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.

        Returns
        -------
        list
            The list of vertices.

        """
        from topologicpy.Graph import Graph
        
        
        if Topology.IsInstance(topology, "Vertex"):
            return []
        if Topology.IsInstance(topology, "Graph"):
            return Graph.Vertices(topology)
        if topology == None:
            return None
        return Topology.SubTopologies(topology=topology, subTopologyType="vertex")
    
    @staticmethod
    def Edges(topology):
        """
        Returns the edges of the input topology.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.

        Returns
        -------
        list
            The list of edges.

        """
        from topologicpy.Graph import Graph
        if Topology.IsInstance(topology, "Edge") or Topology.IsInstance(topology, "Vertex"):
            return []
        if Topology.IsInstance(topology, "Graph"):
            return Graph.Edges(topology)
        return Topology.SubTopologies(topology=topology, subTopologyType="edge")
    
    @staticmethod
    def Wires(topology):
        """
        Returns the wires of the input topology.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.

        Returns
        -------
        list
            The list of wires.

        """
        if Topology.IsInstance(topology, "Wire") or Topology.IsInstance(topology, "Edge") or Topology.IsInstance(topology, "Vertex"):
            return []
        return Topology.SubTopologies(topology=topology, subTopologyType="wire")
    
    @staticmethod
    def Faces(topology):
        """
        Returns the faces of the input topology.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.

        Returns
        -------
        list
            The list of faces.

        """
        if Topology.IsInstance(topology, "Face") or Topology.IsInstance(topology, "Wire") or Topology.IsInstance(topology, "Edge") or Topology.IsInstance(topology, "Vertex"):
            return []
        return Topology.SubTopologies(topology=topology, subTopologyType="face")
    
    @staticmethod
    def Shells(topology):
        """
        Returns the shells of the input topology.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.

        Returns
        -------
        list
            The list of shells.

        """
        if Topology.IsInstance(topology, "Shell") or Topology.IsInstance(topology, "Face") or Topology.IsInstance(topology, "Wire") or Topology.IsInstance(topology, "Edge") or Topology.IsInstance(topology, "Vertex"):
            return []
        return Topology.SubTopologies(topology=topology, subTopologyType="shell")
    
    @staticmethod
    def Cells(topology):
        """
        Returns the cells of the input topology.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.

        Returns
        -------
        list
            The list of cells.

        """
        if Topology.IsInstance(topology, "Cell") or Topology.IsInstance(topology, "Shell") or Topology.IsInstance(topology, "Face") or Topology.IsInstance(topology, "Wire") or Topology.IsInstance(topology, "Edge") or Topology.IsInstance(topology, "Vertex"):
            return []
        return Topology.SubTopologies(topology=topology, subTopologyType="cell")
    
    @staticmethod
    def CellComplexes(topology):
        """
        Returns the cellcomplexes of the input topology.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.

        Returns
        -------
        list
            The list of cellcomplexes.

        """
        if Topology.IsInstance(topology, "CellComplex") or Topology.IsInstance(topology, "Cell") or Topology.IsInstance(topology, "Shell") or Topology.IsInstance(topology, "Face") or Topology.IsInstance(topology, "Wire") or Topology.IsInstance(topology, "Edge") or Topology.IsInstance(topology, "Vertex"):
            return []
        return Topology.SubTopologies(topology=topology, subTopologyType="cellcomplex")
    
    @staticmethod
    def Clusters(topology):
        """
        Returns the clusters of the input topology.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.

        Returns
        -------
        list
            The list of clusters.

        """
        if not Topology.IsInstance(topology, "Cluster"):
            return []
        return Topology.SubTopologies(topology=topology, subTopologyType="cluster")
    
    @staticmethod
    def SubTopologies(topology, subTopologyType="vertex"):
        """
        Returns the subtopologies of the input topology as specified by the subTopologyType input string.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        subTopologyType : str , optional
            The requested subtopology type. This can be one of "vertex", "edge", "wire", "face", "shell", "cell", "cellcomplex", "cluster". It is case insensitive. The default is "vertex".

        Returns
        -------
        list
            The list of subtopologies.

        """
        from topologicpy.Face import Face

        if not Topology.IsInstance(topology, "Topology"):
            print("Topology.SubTopologies - Error: the input topology parameter is not a valid topology. Returning None.")
            return None
        if Topology.TypeAsString(topology).lower() == subTopologyType.lower():
            return [topology]
        
        subTopologies = []

        # Spcecial case for faces to return vertices in CW/CCW order.
        if Topology.IsInstance(topology, "face") and (subTopologyType.lower() == "vertex" or subTopologyType.lower() == "edge"):
            wires = Face.Wires(topology)
            for wire in wires:
                subTopologies += Topology.SubTopologies(wire, subTopologyType=subTopologyType)
        else:
            if subTopologyType.lower() == "vertex":
                _ = topology.Vertices(None, subTopologies) # Hook to Core
            elif subTopologyType.lower() == "edge":
                _ = topology.Edges(None, subTopologies) # Hook to Core
            elif subTopologyType.lower() == "wire":
                _ = topology.Wires(None, subTopologies) # Hook to Core
            elif subTopologyType.lower() == "face":
                _ = topology.Faces(None, subTopologies) # Hook to Core
            elif subTopologyType.lower() == "shell":
                _ = topology.Shells(None, subTopologies) # Hook to Core
            elif subTopologyType.lower() == "cell":
                _ = topology.Cells(None, subTopologies) # Hook to Core
            elif subTopologyType.lower() == "cellcomplex":
                _ = topology.CellComplexes(None, subTopologies) # Hook to Core
            elif subTopologyType.lower() == "cluster":
                _ = topology.Clusters(None, subTopologies) # Hook to Core
            elif subTopologyType.lower() == "aperture":
                _ = topology.Apertures(None, subTopologies) # Hook to Core
            if not subTopologies:
                return [] # Make sure to return an empty list instead of None
        return subTopologies

    
    @staticmethod
    def SuperTopologies(topology, hostTopology, topologyType: str = None) -> list:
        """
        Returns the supertopologies connected to the input topology.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        hostTopology : topologic_core.Topology
            The host to topology in which to search for ther supertopologies.
        topologyType : str , optional
            The topology type to search for. This can be any of "edge", "wire", "face", "shell", "cell", "cellcomplex", "cluster". It is case insensitive. If set to None, the immediate supertopology type is searched for. The default is None.

        Returns
        -------
        list
            The list of supertopologies connected to the input topology.

        """
        if not Topology.IsInstance(topology, "Topology"):
            print("Topology.SuperTopologies - Error: the input topology parameter is not a valid topology. Returning None.")
            return None
        if not Topology.IsInstance(hostTopology, "Topology"):
            print("Topology.SuperTopologies - Error: the input hostTopology parameter is not a valid topology. Returning None.")
            return None

        superTopologies = []

        if topologyType == None:
            typeID = 2*Topology.TypeID(topology)
        else:
            typeID = Topology.TypeID(topologyType)
        if Topology.Type(topology) >= typeID:
            print("Topology.SuperTopologies - Error: The input topologyType parameter is not a valid type for a super topology of the input topology. Returning None.")
            return None #The user has asked for a topology type lower than the input topology
        elif typeID == Topology.TypeID("Edge"):
            topology.Edges(hostTopology, superTopologies)
        elif typeID == Topology.TypeID("Wire"):
            topology.Wires(hostTopology, superTopologies)
        elif typeID == Topology.TypeID("Face"):
            topology.Faces(hostTopology, superTopologies)
        elif typeID == Topology.TypeID("Shell"):
            topology.Shells(hostTopology, superTopologies)
        elif typeID == Topology.TypeID("Cell"):
            topology.Cells(hostTopology, superTopologies)
        elif typeID == Topology.TypeID("CellComplex"):
            topology.CellComplexes(hostTopology, superTopologies)
        elif typeID == Topology.TypeID("Cluster"):
            topology.Cluster(hostTopology, superTopologies)
        else:
            print("Topology.SuperTopologies - Error: The input topologyType parameter is not a valid type for a super topology of the input topology. Returning None.")
            return None
        if not superTopologies:
            return [] # Make sure you return an empty list instead of None
        return superTopologies
    
    @staticmethod
    def TransferDictionaries(sources, sinks, tolerance=0.0001, numWorkers=None):
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
        numWorkers : int, optional
            Number of workers run in parallel to process. The default is None which sets the number to twice the number of CPU cores.

        Returns
        -------
        dict
            Returns a dictionary with the lists of sources and sinks. The keys are "sinks" and "sources".

        """
        from topologicpy.Dictionary import Dictionary
        if not isinstance(sources, list):
            print("Topology.TransferDictionaries - Error: The input sources parameter is not a valid list. Returning None.")
            return None
        if not isinstance(sinks, list):
            print("Topology.TransferDictionaries - Error: The input sinks parameter is not a valid list. Returning None.")
            return None
        if numWorkers == None:
            import multiprocessing
            numWorkers = multiprocessing.cpu_count()*2
        sources = [x for x in sources if Topology.IsInstance(x, "Topology")]
        sinks = [x for x in sinks if Topology.IsInstance(x, "Topology")]
        so_dicts = [Dictionary.PythonDictionary(Topology.Dictionary(s)) for s in sources]
        if len(sources) < 1:
            print("Topology.TransferDictionaries - Error: The input sources does not contain any valid topologies. Returning None.")
            return None
        if len(sinks) < 1:
            print("Topology.TransferDictionaries - Error: The input sinks does not contain any valid topologies. Returning None.")
            return None

        queue = Queue()
        sources_str = [Topology.BREPString(s) for s in sources]
        sink_items = [SinkItem(id(s), Topology.BREPString(s)) for s in sinks]
        mergingProcess = MergingProcess(queue, sources_str, sink_items, so_dicts)
        mergingProcess.start()

        workerProcessPool = WorkerProcessPool(numWorkers, queue, sources_str, sink_items, so_dicts, tolerance=tolerance)
        workerProcessPool.startProcesses()
        workerProcessPool.join()

        queue.put_nowait(None)
        sinkMap = queue.get()
        mergingProcess.join()

        for i, sink in enumerate(sink_items):
            mapItem = sinkMap[sink.ID]
            newDict = Dictionary.ByKeysValues(mapItem.sinkKeys, mapItem.sinkValues)
            _ = sinks[i].SetDictionary(newDict)
        return {"sources": sources, "sinks": sinks}

    
    @staticmethod
    def TransferDictionariesBySelectors(topology, selectors, tranVertices=False, tranEdges=False, tranFaces=False, tranCells=False, tolerance=0.0001, numWorkers=None):
        """
        Transfers the dictionaries of the list of selectors to the subtopologies of the input topology based on the input parameters.

        Parameters
        ----------
        topology : topologic_core.Topology
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
        numWorkers : int , optional
            Number of workers run in parallel to process. If you set it to 1, no parallel processing will take place.
            The default is None which causes the algorithm to use twice the number of cpu cores in the host computer.
        Returns
        -------
        Topology
            The input topology with the dictionaries transferred to its subtopologies.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Cluster import Cluster
        from topologicpy.Plotly import Plotly


        def transfer_dictionaries_by_selectors(object, selectors, tranVertices=False, tranEdges=False, tranFaces=False, tranCells=False, tolerance=0.0001):
            if tranVertices == True:
                vertices = Topology.Vertices(object)
                for vertex in vertices:
                    for selector in selectors:
                        d = Vertex.Distance(selector, vertex)
                        if d < tolerance:
                            vertex = Topology.SetDictionary(vertex, Topology.Dictionary(selector), silent=True)
                            break
            if tranEdges == True:
                edges = Topology.Edges(object)
                for selector in selectors:
                    for edge in edges:
                        d = Vertex.Distance(selector, edge)
                        if d < tolerance:

                            edge = Topology.SetDictionary(edge, Topology.Dictionary(selector), silent=True)
                            break
            if tranFaces == True:
                faces = Topology.Faces(object)
                for face in faces:
                    for selector in selectors:
                        d = Vertex.Distance(selector, face)
                        if d < tolerance:
                            face = Topology.SetDictionary(face, Topology.Dictionary(selector), silent=True)
                            break
            if tranCells == True:
                cells = Topology.Cells(object)
                for cell in cells:
                    for selector in selectors:
                        if Vertex.IsInternal(selector, cell):
                            cell = Topology.SetDictionary(cell, Topology.Dictionary(selector), silent=True)
                            break
            return object
        
        if not Topology.IsInstance(topology, "Topology"):
            print("Topology.TransferDictionariesBySelectors - Error: The input topology parameter is not a valid topology. Returning None.")
            return None
        if not isinstance(selectors, list):
            print("Topology.TransferDictionariesBySelectors - Error: The input selectors parameter is not a valid list. Returning None.")
            return None
        if numWorkers == None:
            import multiprocessing
            numWorkers = multiprocessing.cpu_count()*2
        selectors_tmp = [x for x in selectors if Topology.IsInstance(x, "Vertex")]
        if len(selectors_tmp) < 1:
            print("Topology.TransferDictionariesBySelectors - Error: The input selectors do not contain any valid topologies. Returning None.")
            return None
        
        if numWorkers == 1:
            return transfer_dictionaries_by_selectors(topology, selectors, tranVertices=tranVertices, tranEdges=tranEdges, tranFaces=tranFaces, tranCells=tranCells, tolerance=tolerance)
        sinkEdges = []
        sinkFaces = []
        sinkCells = []
        hidimSink = Topology.HighestType(topology)
        if tranVertices == True:
            sinkVertices = []
            if Topology.Type(topology) == Topology.TypeID("Vertex"):
                sinkVertices.append(topology)
            elif hidimSink >= Topology.TypeID("Vertex"):
                sinkVertices = Topology.Vertices(topology)
            _ = Topology.TransferDictionaries(selectors, sinkVertices, tolerance=tolerance, numWorkers=numWorkers)
        if tranEdges == True:
            sinkEdges = []
            if Topology.Type(topology) == Topology.TypeID("Edge"):
                sinkEdges.append(topology)
            elif hidimSink >= Topology.TypeID("Edge"):
                topology.Edges(None, sinkEdges)
                _ = Topology.TransferDictionaries(selectors, sinkEdges, tolerance=tolerance, numWorkers=numWorkers)
        if tranFaces == True:
            sinkFaces = []
            if Topology.Type(topology) == Topology.TypeID("Face"):
                sinkFaces.append(topology)
            elif hidimSink >= Topology.TypeID("Face"):
                topology.Faces(None, sinkFaces)
            _ = Topology.TransferDictionaries(selectors, sinkFaces, tolerance=tolerance, numWorkers=numWorkers)
        if tranCells == True:
            sinkCells = []
            if Topology.IsInstance(topology, "Cell"):
                sinkCells = [topology]
            elif hidimSink >= Topology.TypeID("Cell"):
                sinkCells = Topology.Cells(topology)
            _ = Topology.TransferDictionaries(selectors, sinkCells, tolerance=tolerance, numWorkers=numWorkers)
        return topology

    
    @staticmethod
    def Transform(topology, matrix):
        """
        Transforms the input topology by the input 4X4 transformation matrix.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        matrix : list
            The input 4x4 transformation matrix.

        Returns
        -------
        topologic_core.Topology
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

        
        kRotation11 = matrix[0][0]
        kRotation12 = matrix[0][1]
        kRotation13 = matrix[0][2]
        kRotation21 = matrix[1][0]
        kRotation22 = matrix[1][1]
        kRotation23 = matrix[1][2]
        kRotation31 = matrix[2][0]
        kRotation32 = matrix[2][1]
        kRotation33 = matrix[2][2]
        kTranslationX = matrix[3][0]
        kTranslationY = matrix[3][1]
        kTranslationZ = matrix[3][2]

        return_topology = topologic.TopologyUtility.Transform(topology, kTranslationX, kTranslationY, kTranslationZ, kRotation11, kRotation12, kRotation13, kRotation21, kRotation22, kRotation23, kRotation31, kRotation32, kRotation33)
        
        vertices = Topology.Vertices(topology)
        edges = Topology.Edges(topology)
        wires = Topology.Wires(topology)
        faces = Topology.Faces(topology)
        shells = Topology.Shells(topology)
        cells = Topology.Cells(topology)
        cellComplexes = Topology.CellComplexes(topology)

        r_vertices = Topology.Vertices(return_topology)
        r_edges = Topology.Edges(return_topology)
        r_wires = Topology.Wires(return_topology)
        r_faces = Topology.Faces(return_topology)
        r_shells = Topology.Shells(return_topology)
        r_cells = Topology.Cells(return_topology)
        r_cellComplexes = Topology.CellComplexes(return_topology)

        for i, t in enumerate(r_vertices):
            t = Topology.SetDictionary(t, Topology.Dictionary(vertices[i]), silent=True)
        for i, t in enumerate(r_edges):
            t = Topology.SetDictionary(t, Topology.Dictionary(edges[i]), silent=True)
        for i, t in enumerate(r_wires):
            t = Topology.SetDictionary(t, Topology.Dictionary(wires[i]), silent=True)
        for i, t in enumerate(r_faces):
            t = Topology.SetDictionary(t, Topology.Dictionary(faces[i]), silent=True)
        for i, t in enumerate(r_shells):
            t = Topology.SetDictionary(t, Topology.Dictionary(shells[i]), silent=True)
        for i, t in enumerate(r_cells):
            t = Topology.SetDictionary(t, Topology.Dictionary(cells[i]), silent=True)
        for i, t in enumerate(r_cellComplexes):
            t = Topology.SetDictionary(t, Topology.Dictionary(cellComplexes[i]), silent=True)
        
        return_topology = Topology.SetDictionary(return_topology, Topology.Dictionary(topology), silent=True)
        return return_topology
    
    @staticmethod
    def Translate(topology, x=0, y=0, z=0):
        """
        Translates (moves) the input topology.

        Parameters
        ----------
        topology : topologic_core.topology
            The input topology.
        x : float , optional
            The x translation value. The default is 0.
        y : float , optional
            The y translation value. The default is 0.
        z : float , optional
            The z translation value. The default is 0.

        Returns
        -------
        topologic_core.Topology
            The translated topology.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Dictionary import Dictionary

        if not Topology.IsInstance(topology, "Topology"):
            print("Topology.Translate - Error: The input topology parameter is not a valid topology. Returning None.")
            return None
        
        if Topology.IsInstance(topology, "vertex"):
            old_x, old_y, old_z = Vertex.Coordinates(topology)
            return_topology = Vertex.ByCoordinates(old_x+x, old_y+y, old_z+z)
            return_topology = Topology.SetDictionary(return_topology, Topology.Dictionary(topology), silent=True)
            return return_topology
        vertices = Topology.Vertices(topology)
        edges = Topology.Edges(topology)
        wires = Topology.Wires(topology)
        faces = Topology.Faces(topology)
        shells = Topology.Shells(topology)
        cells = Topology.Cells(topology)
        cellComplexes = Topology.CellComplexes(topology)        

        try:
            return_topology = topologic.TopologyUtility.Translate(topology, x, y, z)
        except:
            return_topology = topology
        
        r_vertices = Topology.Vertices(return_topology)
        r_edges = Topology.Edges(return_topology)
        r_wires = Topology.Wires(return_topology)
        r_faces = Topology.Faces(return_topology)
        r_shells = Topology.Shells(return_topology)
        r_cells = Topology.Cells(return_topology)
        r_cellComplexes = Topology.CellComplexes(return_topology)
        for i, t in enumerate(r_vertices):
            t = Topology.SetDictionary(t, Topology.Dictionary(vertices[i]), silent=True)
        for i, t in enumerate(r_edges):
            t = Topology.SetDictionary(t, Topology.Dictionary(edges[i]), silent=True)
        for i, t in enumerate(r_wires):
            t = Topology.SetDictionary(t, Topology.Dictionary(wires[i]), silent=True)
        for i, t in enumerate(r_faces):
            t = Topology.SetDictionary(t, Topology.Dictionary(faces[i]), silent=True)
        for i, t in enumerate(r_shells):
            t = Topology.SetDictionary(t, Topology.Dictionary(shells[i]), silent=True)
        for i, t in enumerate(r_cells):
            t = Topology.SetDictionary(t, Topology.Dictionary(cells[i]), silent=True)
        for i, t in enumerate(r_cellComplexes):
            t = Topology.SetDictionary(t, Topology.Dictionary(cellComplexes[i]), silent=True)
        
        return_topology = Topology.SetDictionary(return_topology, Topology.Dictionary(topology), silent=True)
        return return_topology
    
    @staticmethod
    def TranslateByDirectionDistance(topology, direction: list = [0, 0, 0], distance: float = 0):
        """
        Translates (moves) the input topology along the input direction by the specified distance.

        Parameters
        ----------
        topology : topologic_core.topology
            The input topology.
        direction : list , optional
            The direction vector in which the topology should be moved. The default is [0, 0, 0]
        distance : float , optional
            The distance by which the toplogy should be moved. The default is 0.

        Returns
        -------
        topologic_core.Topology
            The translated topology.

        """
        from topologicpy.Vector import Vector
        if not Topology.IsInstance(topology, "Topology"):
            print("Topology.TranslateByDirectionDistance - Error: The input topology parameter is not a valid topology. Returning None.")
            return None
        v = Vector.SetMagnitude(direction, distance)
        return Topology.Translate(topology, v[0], v[1], v[2])

    
    @staticmethod
    def Triangulate(topology, transferDictionaries: bool = False, mode: int = 0, meshSize: float = None, tolerance: float = 0.0001):
        """
        Triangulates the input topology.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topologgy.
        transferDictionaries : bool , optional
            If set to True, the dictionaries of the faces in the input topology will be transferred to the created triangular faces. The default is False.
        mode : int , optional
            The desired mode of meshing algorithm. Several options are available:
            0: Classic
            1: MeshAdapt
            3: Initial Mesh Only
            5: Delaunay
            6: Frontal-Delaunay
            7: BAMG
            8: Fontal-Delaunay for Quads
            9: Packing of Parallelograms
            All options other than 0 (Classic) use the gmsh library. See https://gmsh.info/doc/texinfo/gmsh.html#Mesh-options
            WARNING: The options that use gmsh can be very time consuming and can create very heavy geometry.
        meshSize : float , optional
            The desired size of the mesh when using the "mesh" option. If set to None, it will be
            calculated automatically and set to 10% of the overall size of the face.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Topology
            The triangulated topology.

        """
        from topologicpy.Face import Face
        from topologicpy.Shell import Shell
        from topologicpy.Cell import Cell
        from topologicpy.CellComplex import CellComplex
        from topologicpy.Cluster import Cluster

        if not Topology.IsInstance(topology, "Topology"):
            print("Topology.Triangulate - Error: The input parameter is not a valid topology. Returning None.")
            return None
        t = Topology.Type(topology)
        if (t == Topology.TypeID("Vertex")) or (t == Topology.TypeID("Edge")) or (t == Topology.TypeID("Wire")):
            return topology
        elif t == Topology.TypeID("Cluster"):
            temp_topologies = []
            cellComplexes = Topology.SubTopologies(topology, subTopologyType="cellcomplex") or []
            for cc in cellComplexes:
                temp_topologies.append(Topology.Triangulate(cc, transferDictionaries=transferDictionaries, mode=mode, meshSize=meshSize, tolerance=tolerance))
            cells = Cluster.FreeCells(topology, tolerance=tolerance) or []
            for c in cells:
                temp_topologies.append(Topology.Triangulate(c, transferDictionaries=transferDictionaries, mode=mode, meshSize=meshSize, tolerance=tolerance))
            shells = Cluster.FreeShells(topology, tolerance=tolerance) or []
            for s in shells:
                temp_topologies.append(Topology.Triangulate(s, transferDictionaries=transferDictionaries, mode=mode, meshSize=meshSize, tolerance=tolerance))
            faces = Cluster.FreeFaces(topology, tolerance=tolerance) or []
            for f in faces:
                temp_topologies.append(Topology.Triangulate(f, transferDictionaries=transferDictionaries, mode=mode, meshSize=meshSize, tolerance=tolerance))
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
                triFaces = Face.Triangulate(aFace, mode=mode, meshSize=meshSize, tolerance=tolerance)
            else:
                triFaces = [aFace]
            for triFace in triFaces:
                if transferDictionaries:
                    selectors.append(Topology.SetDictionary(Face.Centroid(triFace), Topology.Dictionary(aFace)))
                faceTriangles.append(triFace)
        return_topology = None
        if t == Topology.TypeID("Face") or t == Topology.TypeID("Shell"): # Face or Shell
            return_topology = Shell.ByFaces(faceTriangles, tolerance=tolerance)
            if transferDictionaries and not return_topology == None:
                return_topology = Topology.TransferDictionariesBySelectors(return_topology, selectors, tranFaces=True, tolerance=tolerance)
        elif t == Topology.TypeID("Cell"): # Cell
            return_topology = Cell.ByFaces(faceTriangles, tolerance=tolerance)
            if transferDictionaries and not return_topology == None:
                return_topology = Topology.TransferDictionariesBySelectors(return_topology, selectors, tranFaces=True, tolerance=tolerance)
        elif t == Topology.TypeID("CellComplex"): #CellComplex
            return_topology = CellComplex.ByFaces(faceTriangles, tolerance=tolerance)
            if transferDictionaries and not return_topology == None:
                return_topology = Topology.TransferDictionariesBySelectors(return_topology, selectors, tranFaces=True, tolerance=tolerance)
        
        if return_topology == None:
            return_topology = Topology.SelfMerge(Cluster.ByTopologies(faceTriangles), tolerance=tolerance)
            if transferDictionaries == True:
                return_topology = Topology.TransferDictionariesBySelectors(return_topology, selectors, tranFaces=True, tolerance=tolerance)
        
        return return_topology

    
    @staticmethod
    def Type(topology):
        """
        Returns the type of the input topology.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.

        Returns
        -------
        int
            The type of the input topology.

        """
        return topology.Type()
    
    @staticmethod
    def TypeAsString(topology, silent=False):
        """
        Returns the type of the input topology as a string.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        silent : bool , optional
            If set to True, no warnings or errors will be printed. The default is False.

        Returns
        -------
        str
            The type of the topology as a string.

        """
        if Topology.IsInstance(topology, "Graph"):
            return "Graph"
        elif Topology.IsInstance(topology, "Topology"):
            return topology.GetTypeAsString() # Hook to Core
        if not silent:
            print("Topology.TypeAsString - Error: The input topology parameter is not a valid topology or graph. Returning None.")
        return None
    
    @staticmethod
    def TypeID(name : str = None) -> int:
        """
        Returns the type id of the input name string.

        Parameters
        ----------
        name : str , optional
            The input class name string. This could be one of:
                "vertex",
                "edge",
                "wire",
                "face",
                "shell",
                "cell",
                "cellcomplex",
                "cluster",
                "aperture",
                "context",
                "dictionary",
                "graph",
                "topology"
            
            It is case insensitive. The default is None.

        Returns
        -------
        int
            The type id of the input topologyType string.

        """
        if not isinstance(name, str):
            print("Topology.TypeID - Error: The input topologyType parameter is not a valid string. Returning None.")
            return None
        name = name.lower()
        if not name in ["vertex", "edge", "wire",
                                "face", "shell", "cell",
                                "cellcomplex", "cluster", "aperture",
                                "context", "dictionary", "graph", "topology"]:
            print("Topology.TypeID - Error: The input name parameter is not a recognized string. Returning None.")
            return None
        typeID = None
        if name == "vertex":
            typeID = 1
        elif name == "edge":
            typeID = 2
        elif name == "wire":
            typeID = 4
        elif name == "face":
            typeID = 8
        elif name == "shell":
            typeID = 16
        elif name == "cell":
            typeID = 32
        elif name == "cellcomplex":
            typeID = 64
        elif name == "cluster":
            typeID = 128
        elif name == "aperture":
            typeID = 256
        elif name == "context":
            typeID = 512
        elif name == "dictionary":
            typeID = 1024
        elif name == "graph":
            typeID = 2048
        elif name == "topology":
            typeID = 4096
        return typeID
    
    @staticmethod
    def UUID(topology, namespace="topologicpy"):
        """
        Generate a UUID v5 based on the provided content and a fixed namespace.
        
        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology
        namespace : str , optional
            The base namescape to use for generating the UUID

        Returns
        -------
        UUID
            The uuid of the input topology.

        """
        import uuid
        from topologicpy.Dictionary import Dictionary

        predefined_namespace_dns = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')
        namespace_uuid = uuid.uuid5(predefined_namespace_dns, namespace)
        cellComplexes = Topology.CellComplexes(topology)
        cells = Topology.Cells(topology)
        shells = Topology.Shells(topology)
        faces = Topology.Faces(topology)
        wires = Topology.Wires(topology)
        edges = Topology.Edges(topology)
        vertices = Topology.Vertices(topology)
        apertures = Topology.Apertures(topology, subTopologyType="all")
        subTopologies = cellComplexes+cells+shells+faces+wires+edges+vertices+apertures
        dictionaries = [Dictionary.PythonDictionary(Topology.Dictionary(topology))]
        dictionaries += [Dictionary.PythonDictionary(Topology.Dictionary(s)) for s in subTopologies]
        dict_str = str(dictionaries)
        top_geom = Topology.Geometry(topology, mantissa=6)
        verts_str = str(top_geom['vertices'])
        edges_str = str(top_geom['edges'])
        faces_str = str(top_geom['faces'])
        geo_str = verts_str+edges_str+faces_str
        final_str = geo_str+dict_str
        uuid_str = uuid.uuid5(namespace_uuid, final_str)
        return str(uuid_str)
    
    @staticmethod
    def View3D(*topologies, uuid = None, nameKey="name", colorKey="color", opacityKey="opacity", defaultColor=[256,256,256], defaultOpacity=0.5, transposeAxes: bool = True, mode: int = 0, meshSize: float = None, overwrite: bool = False, mantissa: int = 6, tolerance: float = 0.0001):
        """
        Sends the input topologies to 3dviewer.net. The topologies must be 3D meshes.

        Parameters
        ----------
        topologies : list or comma separated topologies
            The input list of topologies.
        uuid : UUID , optional
            The UUID v5 to use to identify these topologies. The default is a UUID based on the topologies themselves.
        nameKey : str , optional
            The topology dictionary key under which to find the name of the topology. The default is "name".
        colorKey : str, optional
            The topology dictionary key under which to find the color of the topology. The default is "color".
        opacityKey : str , optional
            The topology dictionary key under which to find the opacity of the topology. The default is "opacity".
        defaultColor : list , optional
            The default color to use if no color is stored in the topology dictionary. The default is [255,255, 255] (white).
        defaultOpacity : float , optional
            The default opacity to use of no opacity is stored in the topology dictionary. This must be between 0 and 1. The default is 1 (fully opaque).
        transposeAxes : bool , optional
            If set to True the Z and Y coordinates are transposed so that Y points "up"
        mode : int , optional
            The desired mode of meshing algorithm (for triangulation). Several options are available:
            0: Classic
            1: MeshAdapt
            3: Initial Mesh Only
            5: Delaunay
            6: Frontal-Delaunay
            7: BAMG
            8: Fontal-Delaunay for Quads
            9: Packing of Parallelograms
            All options other than 0 (Classic) use the gmsh library. See https://gmsh.info/doc/texinfo/gmsh.html#Mesh-options
            WARNING: The options that use gmsh can be very time consuming and can create very heavy geometry.
        meshSize : float , optional
            The desired size of the mesh when using the "mesh" option. If set to None, it will be
            calculated automatically and set to 10% of the overall size of the face.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        overwrite : bool , optional
            If set to True the ouptut file will overwrite any pre-existing file. Otherwise, it won't. The default is False.

        Returns
        -------
        bool
            True if the export operation is successful. False otherwise.

        """
        from topologicpy.Helper import Helper
        from topologicpy.Cluster import Cluster
        import requests
        import webbrowser

        if isinstance(topologies, tuple):
            topologies = Helper.Flatten(list(topologies))
        if isinstance(topologies, list):
            new_topologies = [d for d in topologies if Topology.IsInstance(d, "Topology")]
        if len(new_topologies) == 0:
            print("Topology.View3D - Error: the input topologies parameter does not contain any valid topologies. Returning None.")
            return None
        if not isinstance(new_topologies, list):
            print("Topology.View3D - Error: The input topologies parameter is not a valid list. Returning None.")
            return None

        if uuid == None:
            cluster = Cluster.ByTopologies(new_topologies)
            uuid = Topology.UUID(cluster)
        obj_string, mtl_string = Topology.OBJString(new_topologies,
                                                    nameKey=nameKey,
                                                    colorKey=colorKey,
                                                    opacityKey=opacityKey,
                                                    defaultColor=defaultColor,
                                                    defaultOpacity=defaultOpacity,
                                                    transposeAxes=transposeAxes,
                                                    mode=mode,
                                                    meshSize=meshSize,
                                                    mantissa=mantissa,
                                                    tolerance=tolerance)
        

        file_contents = {}
        file_contents['example.obj'] = obj_string
        file_contents['example.mtl'] = mtl_string

        try:
            response = requests.post('https://3dviewer.deno.dev/upload/'+str(uuid), files=file_contents)
            if response.status_code != 200:
                print(f'Failed to upload file(s): {response.status_code} {response.reason}')
            # Open the web page in the default web browser
            # URL of the web page you want to open
            url = "https://3dviewer.deno.dev/#channel="+str(uuid)
            if not url in opened_urls:
                opened_urls.add(url)
                webbrowser.open(url)
        except requests.exceptions.RequestException as e:
            print(f'Error uploading file(s): {e}')
        return True