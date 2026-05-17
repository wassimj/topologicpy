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

from __future__ import annotations

from topologicpy.Core import Core
import math
import os
import warnings

try:
    import numpy as np
except:
    print("Face - Installing required numpy library.")
    try:
        os.system("pip install numpy")
    except:
        os.system("pip install numpy --user")
    try:
        import numpy as np
        print("Face - numpy library installed correctly.")
    except:
        warnings.warn("Face - Error: Could not import numpy.")

class Face():
    @staticmethod
    def AddInternalBoundaries(face, wires: list):
        """
        Adds internal boundaries (closed wires) to the input face. Internal boundaries are considered holes in the input face.

        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        wires : list
            The input list of internal boundaries (closed wires).

        Returns
        -------
        topologic_core.Face
            The created face with internal boundaries added to it.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(face, "Face"):
            print("Face.AddInternalBoundaries - Error: The input face parameter is not a valid topologic face. Returning None.")
            return None
        if not isinstance(wires, list):
            print("Face.AddInternalBoundaries - Warning: The input wires parameter is not a valid list. Returning the input face.")
            return face
        wireList = [w for w in wires if Topology.IsInstance(w, "Wire")]
        if len(wireList) < 1:
            print("Face.AddInternalBoundaries - Warning: The input wires parameter does not contain any valid wires. Returning the input face.")
            return face
        faceeb = Face.ExternalBoundary(face)
        faceibList = Face.InternalBoundaries(face)
        for wire in wires:
            faceibList.append(wire)
        return Face.ByWires(faceeb, faceibList)

    @staticmethod
    def AddInternalBoundariesCluster(face, cluster):
        """
        Adds the input cluster of internal boundaries (closed wires) to the input face. Internal boundaries are considered holes in the input face.

        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        cluster : topologic_core.Cluster
            The input cluster of internal boundaries (topologic wires).

        Returns
        -------
        topologic_core.Face
            The created face with internal boundaries added to it.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(face, "Face"):
            print("Face.AddInternalBoundariesCluster - Warning: The input cluster parameter is not a valid cluster. Returning None.")
            return None
        if not cluster:
            return face
        if not Topology.IsInstance(cluster, "Cluster"):
            return face
        wires = Topology.Wires(cluster)
        return Face.AddInternalBoundaries(face, wires)
    
    @staticmethod
    def Angle(faceA, faceB, mantissa: int = 6) -> float:
        """
        Returns the angle in degrees between the two input faces.

        Parameters
        ----------
        faceA : topologic_core.Face
            The first input face.
        faceB : topologic_core.Face
            The second input face.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.

        Returns
        -------
        float
            The angle in degrees between the two input faces.

        """
        from topologicpy.Vector import Vector
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(faceA, "Face"):
            print("Face.Angle - Warning: The input faceA parameter is not a valid topologic face. Returning None.")
            return None
        if not Topology.IsInstance(faceB, "Face"):
            print("Face.Angle - Warning: The input faceB parameter is not a valid topologic face. Returning None.")
            return None
        dirA = Face.Normal(faceA, outputType="xyz", mantissa=3)
        dirB = Face.Normal(faceB, outputType="xyz", mantissa=3)
        if dirA == None or dirB == None:
            print("Face.Angle - Error: Could not compute the angle between the two input faces. Returning None.")
            return None
        return round((Vector.Angle(dirA, dirB)), mantissa)

    @staticmethod
    def Area(face, mantissa: int = 6) -> float:
        """
        Returns the area of the input face.

        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.

        Returns
        -------
        float
            The area of the input face.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(face, "Face"):
            print("Face.Area - Warning: The input face parameter is not a valid topologic face. Returning None.")
            return None
        area = None
        try:
            area = round(Core.FaceUtility.Area(face), mantissa)
        except:
            area = None
        return area

    @staticmethod
    def BoundingRectangle(topology, optimize: int = 0, tolerance: float = 0.0001):
        """
        Returns a face representing a bounding rectangle of the input topology. The returned face contains a dictionary with key "zrot" that represents rotations around the Z axis. If applied the resulting face will become axis-aligned.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology.
        optimize : int , optional
            If set to an integer from 1 (low optimization) to 10 (high optimization), the method will attempt to optimize the bounding rectangle so that it reduces its surface area. Default is 0 which will result in an axis-aligned bounding rectangle. Default is 0.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        
        Returns
        -------
        topologic_core.Face
            The bounding rectangle of the input topology.

        """
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology

        br_wire = Wire.BoundingRectangle(topology=topology, optimize=optimize, tolerance=tolerance)
        if not Topology.IsInstance(br_wire, "Wire"):
            print("Face.BoundingRectangle - Warning: Could not create base wire. Returning None.")
            return None
        br_face = Face.ByWire(br_wire)
        if not Topology.IsInstance(br_face, "Face"):
            print("Face.BoundingRectangle - Warning: Could not create face from base wire. Returning None.")
            return None
        br_face = Topology.SetDictionary(br_face, Topology.Dictionary(br_wire))
        return br_face
    
    @staticmethod
    def ByEdges(edges: list, tolerance : float = 0.0001, silent: bool = False):
        """
        Creates a face from the input list of edges.

        Parameters
        ----------
        edges : list
            The input list of edges.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        face : topologic_core.Face
            The created face.

        """
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology

        if not isinstance(edges, list):
            if not silent:
                print("Face.ByEdges - Error: The input edges parameter is not a valid list. Returning None.")
            return None
        edges = [e for e in edges if Topology.IsInstance(e, "Edge")]
        if len(edges) < 1:
            if not silent:
                print("Face.ByEdges - Error: The input edges parameter does not contain any valid edges. Returning None.")
            return None
        wire = Wire.ByEdges(edges, tolerance=tolerance)
        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Face.ByEdges - Error: Could not create the required wire. Returning None.")
            return None
        face = Face.ByWire(wire, tolerance=tolerance)
        if not Topology.IsInstance(face, "Face"):
            if not silent:
                print("Face.ByEdges - Error: Could not create face from base wire. Returning None.")
            return None
        return face

    @staticmethod
    def ByEdgesCluster(cluster, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a face from the input cluster of edges.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input cluster of edges.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        face : topologic_core.Face
            The created face.

        """
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Cluster"):
            if not silent:
                print("Face.ByEdgesCluster - Error: The input cluster parameter is not a valid topologic cluster. Returning None.")
            return None
        edges = Cluster.Edges(cluster)
        if len(edges) < 1:
            if not silent:
                print("Face.ByEdgesCluster - Error: The input cluster parameter does not contain any valid edges. Returning None.")
            return None
        face = Face.ByEdges(edges, tolerance=tolerance)
        if not Topology.IsInstance(face, "Face"):
            if not silent:
                print("Face.ByEdgesCluster - Error: Could not create face from edges. Returning None.")
            return None
        return face

    @staticmethod
    def ByOffset(face, offset: float = 1.0, offsetKey: str = "offset", stepOffsetA: float = 0, stepOffsetB: float = 0, stepOffsetKeyA: str = "stepOffsetA", stepOffsetKeyB: str = "stepOffsetB", reverse: bool = False, bisectors: bool = False, transferDictionaries: bool = False, epsilon: float = 0.01, tolerance: float = 0.0001,  silent: bool = False, numWorkers: int = None):
        """
        Creates an offset face from the input face. A positive offset value results in an offset to the interior of an anti-clockwise face.

        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        offset : float , optional
            The desired offset distance. Default is 1.0.
        offsetKey : str , optional
            The edge dictionary key under which to find the offset value. If a value cannot be found, the offset input parameter value is used instead. Default is "offset".
        stepOffsetA : float , optional
            The amount to offset along the previous edge when transitioning between parallel edges with different offsets. Default is 0.
        stepOffsetB : float , optional
            The amount to offset along the next edge when transitioning between parallel edges with different offsets. Default is 0.
        stepOffsetKeyA : str , optional
            The vertex dictionary key under which to find the step offset A value. If a value cannot be found, the stepOffsetA input parameter value is used instead. Default is "stepOffsetA".
        stepOffsetKeyB : str , optional
            The vertex dictionary key under which to find the step offset B value. If a value cannot be found, the stepOffsetB input parameter value is used instead. Default is "stepOffsetB".
        bisectors : bool , optional
            If set to True, The bisectors (seams) edges will be included in the returned wire. This will result in the returned shape to be a shell rather than a face. Default is False.
        reverse : bool , optional
            If set to True, the direction of offsets is reversed. Otherwise, it is not. Default is False.
        transferDictionaries : bool , optional
            If set to True, the dictionaries of the original wire, its edges, and its vertices are transfered to the new wire. Otherwise, they are not. Default is False.
        epsilon : float , optional
            The desired epsilon (another form of tolerance for shortest edge to remove). Default is 0.01. (This is set to a larger number as it was found to work better)
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        numWorkers : int , optional
            Number of workers run in parallel to process. If you set it to 1, no parallel processing will take place.
            The default is None which causes the algorithm to use twice the number of cpu cores in the host computer.
        
        Returns
        -------
        topologic_core.Face or topologic_core.Shell
            The created face or shell.

        """
        from topologicpy.Wire import Wire
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Helper import Helper

        if not Topology.IsInstance(face, "Face"):
            if not silent:
                print("Face.ByOffset - Warning: The input face parameter is not a valid face. Returning None.")
            return None
        
        if abs(Face.Normal(face)[2] + 1) <= tolerance:
            reverse = not(reverse)
        eb = Face.Wire(face)
        
        internal_boundaries = Face.InternalBoundaries(face)
        offset_external_boundary = Wire.ByOffset(eb,
                                                 offset=offset,
                                                 offsetKey=offsetKey,
                                                 stepOffsetA=stepOffsetA,
                                                 stepOffsetB=stepOffsetB,
                                                 stepOffsetKeyA=stepOffsetKeyA,
                                                 stepOffsetKeyB=stepOffsetKeyB,
                                                 reverse=reverse,
                                                 bisectors=bisectors,
                                                 transferDictionaries=transferDictionaries,
                                                 epsilon=epsilon,
                                                 tolerance=tolerance,
                                                 silent=silent,
                                                 numWorkers=numWorkers)
        offset_internal_boundaries = []
        for internal_boundary in internal_boundaries:
            offset_internal_boundary = Wire.ByOffset(internal_boundary,
                                                    offset=offset,
                                                    offsetKey=offsetKey,
                                                    stepOffsetA=stepOffsetA,
                                                    stepOffsetB=stepOffsetB,
                                                    stepOffsetKeyA=stepOffsetKeyA,
                                                    stepOffsetKeyB=stepOffsetKeyB,
                                                    reverse=reverse,
                                                    bisectors=bisectors,
                                                    transferDictionaries=transferDictionaries,
                                                    epsilon=epsilon,
                                                    tolerance=tolerance,
                                                    silent=silent,
                                                    numWorkers=numWorkers)
            offset_internal_boundaries.append(offset_internal_boundary)
        
        if bisectors == True:
            return_face = Face.ByOffset(face,
                                    offset=offset,
                                    offsetKey=offsetKey,
                                    stepOffsetA=stepOffsetA,
                                    stepOffsetB=stepOffsetB,
                                    stepOffsetKeyA=stepOffsetKeyA,
                                    stepOffsetKeyB=stepOffsetKeyB,
                                    reverse=reverse,
                                    bisectors=False,
                                    transferDictionaries=transferDictionaries,
                                    epsilon=epsilon,
                                    tolerance=tolerance,
                                    silent=silent,
                                    numWorkers=numWorkers)
            all_edges = Topology.Edges(offset_external_boundary)+[Topology.Edges(ib, silent=True) for ib in offset_internal_boundaries]
            all_edges += Topology.Edges(face, silent=True)
            all_edges = Helper.Flatten(all_edges)
            all_edges_cluster = Cluster.ByTopologies(all_edges)
            if reverse == True:
                return_face = Topology.Slice(return_face, all_edges_cluster)
            else:
                return_face = Topology.Slice(face, all_edges_cluster)
            if not Topology.IsInstance(return_face, "Shell"):
                if not silent:
                    print("Face.ByOffset - Warning: Could not create shell by slicing. Returning None.")
                return None
            return return_face
        return_face = Face.ByWires(offset_external_boundary, offset_internal_boundaries, tolerance=tolerance)
        if not Topology.IsInstance(return_face, "Face"):
            if not silent:
                print("Face.ByOffset - Warning: Could not create face from wires. Returning None.")
            return None
        return return_face

    @staticmethod  
    def ByOffsetArea(face,
                    area,
                    offsetKey="offset",
                    minOffsetKey="minOffset",
                    maxOffsetKey="maxOffset",
                    defaultMinOffset=0,
                    defaultMaxOffset=1,
                    maxIterations = 1,
                    tolerance=0.0001,
                    silent = False,
                    numWorkers = None):
        """
        Creates an offset face from the input face based on the input area.

        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        area : float
            The desired area of the created face.
        offsetKey : str , optional
            The edge dictionary key under which to store the offset value. Default is "offset".
        minOffsetKey : str , optional
            The edge dictionary key under which to find the desired minimum edge offset value. If a value cannot be found, the defaultMinOffset input parameter value is used instead. Default is "minOffset".
        maxOffsetKey : str , optional
            The edge dictionary key under which to find the desired maximum edge offset value. If a value cannot be found, the defaultMaxOffset input parameter value is used instead. Default is "maxOffset".
        defaultMinOffset : float , optional
            The desired minimum edge offset distance. Default is 0.
        defaultMaxOffset : float , optional
            The desired maximum edge offset distance. Default is 1.
        maxIterations: int , optional
            The desired maximum number of iterations to attempt to converge on a solution. Default is 1.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        numWorkers : int , optional
            Number of workers run in parallel to process. If you set it to 1, no parallel processing will take place.
            The default is None which causes the algorithm to use twice the number of cpu cores in the host computer.
        
        Returns
        -------
        topologic_core.Face
            The created face.

        """
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        import numpy as np
        from scipy.optimize import minimize

        def compute_offset_amounts(face,
                                area,
                                offsetKey="offset",
                                minOffsetKey="minOffset",
                                maxOffsetKey="maxOffset",
                                defaultMinOffset=0,
                                defaultMaxOffset=1,
                                maxIterations = 1,
                                tolerance=0.0001):
            
            initial_offsets = []
            bounds = []
            for edge in edges:
                d = Topology.Dictionary(edge)
                minOffset = Dictionary.ValueAtKey(d, minOffsetKey) or defaultMinOffset
                maxOffset = Dictionary.ValueAtKey(d, maxOffsetKey) or defaultMaxOffset
                # Initial guess: small negative offsets to shrink the polygon, within the constraints
                initial_offsets.append((minOffset + maxOffset) / 2)
                # Bounds based on the constraints for each edge
                bounds.append((minOffset, maxOffset))

            # Convert initial_offsets to np.array for efficiency
            initial_offsets = np.array(initial_offsets)
            iteration_count = [0]  # List to act as a mutable counter

            def objective_function(offsets):
                for i, edge in enumerate(edges):
                    d = Topology.Dictionary(edge)
                    d = Dictionary.SetValueAtKey(d, offsetKey, offsets[i])
                    edge = Topology.SetDictionary(edge, d)
                
                # Offset the wire
                new_face = Face.ByOffset(face, offsetKey=offsetKey, silent=silent, numWorkers=numWorkers)
                # Check for an illegal wire. In that case, return a very large loss value.
                if not Topology.IsInstance(new_face, "Face"):
                    return (float("inf"))
                # Calculate the area of the new wire/face
                new_area = Face.Area(new_face)
                
                # The objective is the difference between the target hole area and the actual hole area
                # We want this difference to be as close to 0 as possible
                loss = (new_area - area) ** 2
                # If the loss is less than the tolerance, accept the result and return a loss of 0.
                if loss <= tolerance:
                    return 0
                # Otherwise, return the actual loss value.
                return loss 
            
            # Callback function to track and display iteration number
            def iteration_callback(xk):
                iteration_count[0] += 1  # Increment the counter
                if not silent:
                    print(f"Face.ByOffsetArea - Information: Iteration {iteration_count[0]}")
            
            # Use scipy optimization/minimize to find the correct offsets, respecting the min/max bounds
            result = minimize(objective_function,
                            initial_offsets,
                            method = "Powell",
                            bounds=bounds,
                            options={ 'maxiter': maxIterations},
                            callback=iteration_callback
                            )

            # Return the offsets
            return result.x
        
        if not Topology.IsInstance(face, "Face"):
            if not silent:
                print("Face.OffsetByArea - Error: The input face parameter is not a valid face. Returning None.")
            return None
        
        edges = Topology.Edges(face)
        # Compute the offset amounts
        offsets = compute_offset_amounts(face,
                                area = area,
                                offsetKey = offsetKey,
                                minOffsetKey = minOffsetKey,
                                maxOffsetKey = maxOffsetKey,
                                defaultMinOffset = defaultMinOffset,
                                defaultMaxOffset = defaultMaxOffset,
                                maxIterations = maxIterations,
                                tolerance = tolerance)
        # Set the edge dictionaries correctly according to the specified offsetKey
        for i, edge in enumerate(edges):
            d = Topology.Dictionary(edge)
            d = Dictionary.SetValueAtKey(d, offsetKey, offsets[i])
            edge = Topology.SetDictionary(edge, d)
                
        # Offset the face
        return_face = Face.ByOffset(face, offsetKey=offsetKey, silent=silent, numWorkers=numWorkers)
        if not Topology.IsInstance(face, "Face"):
            if not silent:
                print("Face.OffsetByArea - Error: Could not create the offset face. Returning None.")
            return None
        return return_face

    @staticmethod
    def ByShell(shell, origin= None, angTolerance: float = 0.1, tolerance: float = 0.0001, silent=False):
        """
        Creates a face by merging the faces of the input shell.

        Parameters
        ----------
        shell : topologic_core.Shell
            The input shell.
        angTolerance : float , optional
            The desired angular tolerance. Default is 0.1.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        

        Returns
        -------
        topologic_core.Face
            The created face.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Shell import Shell
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Helper import Helper
        
        def planarizeList(wireList):
            returnList = []
            for aWire in wireList:
                returnList.append(Wire.Planarize(aWire))
            return returnList
        
        if not Topology.IsInstance(shell, "Shell"):
            print("Face.ByShell - Error: The input shell parameter is not a valid toplogic shell. Returning None.")
            return None
        
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Topology.Centroid(shell)
        if not Topology.IsInstance(origin, "Vertex"):
            print("Face.ByShell - Error: The input origin parameter is not a valid topologic vertex. Returning None.")
            return None
        
        # Try the simple method first
        face = None
        ext_boundary = Wire.RemoveCollinearEdges(Shell.ExternalBoundary(shell), tolerance=tolerance, silent=silent)
        if Topology.IsInstance(ext_boundary, "Wire"):
            face = Face.ByWire(ext_boundary, silent=silent)
        elif Topology.IsInstance(ext_boundary, "Cluster"):
            wires = Topology.Wires(ext_boundary)
            faces = [Face.ByWire(w, silent=silent) for w in wires]
            areas = [Face.Area(f) for f in faces]
            wires = Helper.Sort(wires, areas, reverseFlags=[True])
            face = Face.ByWires(wires[0], wires[1:], silent=silent)

        if Topology.IsInstance(face, "Face"):
            return face
        world_origin = Vertex.Origin()
        planar_shell = Shell.Planarize(shell)
        normal = Face.Normal(Topology.Faces(planar_shell)[0])
        planar_shell = Topology.Flatten(planar_shell, origin=origin, direction=normal)
        vertices = Topology.Vertices(planar_shell)
        new_vertices = []
        for v in vertices:
            x, y, z = Vertex.Coordinates(v)
            new_v = Vertex.ByCoordinates(x,y,0)
            new_vertices.append(new_v)
        planar_shell = Topology.SelfMerge(Topology.ReplaceVertices(planar_shell, verticesA=vertices, verticesB=new_vertices), tolerance=tolerance)
        ext_boundary = Shell.ExternalBoundary(planar_shell, tolerance=tolerance)
        ext_boundary = Topology.RemoveCollinearEdges(ext_boundary, angTolerance=angTolerance, tolerance=tolerance, silent=silent)
        if not Topology.IsInstance(ext_boundary, "Topology"):
            print("Face.ByShell - Error: Could not derive the external boundary of the input shell parameter. Returning None.")
            return None

        if Topology.IsInstance(ext_boundary, "Wire"):
            if not Topology.IsPlanar(ext_boundary, tolerance=tolerance):
                ext_boundary = Wire.Planarize(ext_boundary, origin=origin, tolerance=tolerance)
            ext_boundary = Topology.RemoveCollinearEdges(ext_boundary, angTolerance=angTolerance, tolerance=tolerance, silent=silent)
            try:
                face = Face.ByWire(ext_boundary, tolerance=tolerance, silent=silent)
                face = Topology.Unflatten(face, origin=origin, direction=normal)
                return face
            except:
                print("Face.ByShell - Error: The operation failed. Returning None.")
                return None
        elif Topology.IsInstance(ext_boundary, "Cluster"): # The shell has holes.
            wires = Topology.Wires(ext_boundary)
            faces = []
            areas = []
            for wire in wires:
                aFace = Face.ByWire(wire, tolerance=tolerance)
                if not Topology.IsInstance(aFace, "Face"):
                    print("Face.ByShell - Error: The operation failed. Returning None.")
                    return None
                anArea = abs(Face.Area(aFace))
                faces.append(aFace)
                areas.append(anArea)
            max_index = areas.index(max(areas))
            ext_boundary = faces[max_index]
            int_boundaries = list(set(faces) - set([ext_boundary]))
            int_wires = []
            for int_boundary in int_boundaries:
                temp_wires = Topology.Wires(int_boundary)
                int_wires.append(Topology.RemoveCollinearEdges(temp_wires[0], angTolerance=angTolerance, tolerance=tolerance, silent=silent))
                #int_wires.append(temp_wires[0])

            temp_wires = Topology.Wires(ext_boundary)
            ext_wire = Topology.RemoveCollinearEdges(temp_wires[0], angTolerance=angTolerance, tolerance=tolerance, silent=silent)
            #ext_wire = temp_wires[0]
            face = Face.ByWires(ext_wire, int_wires)
            face = Topology.Unflatten(face, origin=origin, direction=normal)
            return face
        else:
            return None
    
    @staticmethod
    def ByThickenedWire(wire, offsetA: float = 1.0, offsetB: float = 0.0, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a face by thickening the input wire. This method assumes the wire is manifold and planar.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire to be thickened.
        offsetA : float , optional
            The desired offset to the exterior of the wire. Default is 1.0.
        offsetB : float , optional
            The desired offset to the interior of the wire. Default is 0.0.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Face
            The created face.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Vector import Vector
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "Wire"):
            print("Face.ByThickenedWire - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        if not Wire.IsManifold(wire):
            print("Face.ByThickenedWire - Error: The input wire parameter is not a manifold wire. Returning None.")
            return None
        three_vertices = Topology.Vertices(wire)[0:3]
        temp_w = Wire.ByVertices(three_vertices, close=True, tolerance=tolerance, silent=silent)
        flat_face = Face.ByWire(temp_w, tolerance=tolerance)
        origin = Vertex.Origin()
        normal = Face.Normal(flat_face)
        flat_wire = Topology.Flatten(wire, origin=origin, direction=normal)
        outside_wire = Wire.ByOffset(flat_wire, offset=abs(offsetA)*-1, bisectors = False, tolerance=tolerance)
        inside_wire = Wire.ByOffset(flat_wire, offset=abs(offsetB), bisectors = False, tolerance=tolerance)
        inside_wire = Wire.Reverse(inside_wire)
        if not Wire.IsClosed(flat_wire):
            sv = Topology.Vertices(flat_wire)[0]
            ev = Topology.Vertices(flat_wire)[-1]
            edges = Topology.Edges(flat_wire)
            first_edge = Topology.SuperTopologies(sv, flat_wire, topologyType="edge")[0]
            first_normal = Edge.Normal(first_edge)
            last_edge = Topology.SuperTopologies(ev, flat_wire, topologyType="edge")[0]
            last_normal = Edge.Normal(last_edge)
            sv1 = Topology.TranslateByDirectionDistance(sv, first_normal, abs(offsetB))
            sv2 = Topology.TranslateByDirectionDistance(sv, Vector.Reverse(first_normal), abs(offsetA))
            ev1 = Topology.TranslateByDirectionDistance(ev, last_normal, abs(offsetB))
            ev2 = Topology.TranslateByDirectionDistance(ev, Vector.Reverse(last_normal), abs(offsetA))
            out_vertices = Topology.Vertices(outside_wire)[1:-1]
            in_vertices = Topology.Vertices(inside_wire)[1:-1]
            vertices = [sv2] + out_vertices + [ev2,ev1] + in_vertices + [sv1]
            return_face = Face.ByWire(Wire.ByVertices(vertices, close=True, tolerance=tolerance, silent=silent))
        else:
            return_face = Face.ByWires(outside_wire, [inside_wire])
        return_face = Topology.Unflatten(return_face, origin=origin, direction=normal)
        return return_face
    
    @staticmethod
    def ByVertices(vertices: list, tolerance: float = 0.0001, silent: bool = False):
        
        """
        Creates a face from the input list of vertices.

        Parameters
        ----------
        vertices : list
            The input list of vertices.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Face
            The created face.

        """
        from topologicpy.Topology import Topology
        from topologicpy.Wire import Wire

        if not isinstance(vertices, list):
            if not silent:
                print("Face.ByVertices - Error: The input vertices parameter is not a valid list. Returning None.")
            return None
        vertexList = [x for x in vertices if Topology.IsInstance(x, "Vertex")]
        if len(vertexList) < 3:
            if not silent:
                print("Face.ByVertices - Error: The input vertices parameter does not contain at least three valid vertices. Returning None.")
            return None
        
        w = Wire.ByVertices(vertexList, close=True, tolerance=tolerance, silent=silent)
        if not Topology.IsInstance(w, "Wire"):
            if not silent:
                print("Face.ByVertices - Error: Could not create the base wire. Returning None.")
            return None
        if not Wire.IsClosed(w):
            if not silent:
                print("Face.ByVertices - Error: Could not create a closed base wire. Returning None.")
            return None
        f = Face.ByWire(w, tolerance=tolerance, silent=silent)
        if not Topology.IsInstance(f, "Face"):
            if not silent:
                print("Face.ByVertices - Error: Could not create the face. Returning None.")
            return None
        return f

    @staticmethod
    def ByVerticesCluster(cluster, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a face from the input cluster of vertices.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input cluster of vertices.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Face
            The created face.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Cluster"):
            if not silent:
                print("Face.ByVertices - Error: The input cluster parameter is not a valid cluster. Returning None.")
            return None
        vertices = Topology.Vertices(cluster)
        if len(vertices) < 3:
            if not silent:
                print("Face.ByVertices - Error: The input cluster parameter does not contain at least three valid vertices. Returning None.")
            return None
        return Face.ByVertices(vertices, tolerance=tolerance)

    # @staticmethod
    # def ByWire(wire, tolerance: float = 0.0001, silent: bool = False):
    #     """
    #     Creates a face from the input closed wire.

    #     Parameters
    #     ----------
    #     wire : topologic_core.Wire
    #         The input wire.
    #     tolerance : float , optional
    #         The desired tolerance. Default is 0.0001.
    #     silent : bool , optional
    #         If set to True, error and warning messages are suppressed. Default is False.

    #     Returns
    #     -------
    #     topologic_core.Face or list
    #         The created face. If the wire is non-planar, the method will attempt to triangulate the wire and return a list of faces.

    #     """
    #     from topologicpy.Vertex import Vertex
    #     from topologicpy.Wire import Wire
    #     from topologicpy.Shell import Shell
    #     from topologicpy.Cluster import Cluster
    #     from topologicpy.Topology import Topology
    #     from topologicpy.Dictionary import Dictionary
    #     import inspect

    #     def triangulateWire(wire):
    #         wire = Topology.RemoveCollinearEdges(wire, angTolerance=0.1, tolerance=tolerance, silent=silent)
    #         vertices = Topology.Vertices(wire)
    #         shell = Shell.Delaunay(vertices)
    #         if Topology.IsInstance(shell, "Topology"):
    #             return Topology.Faces(shell)
    #         else:
    #             return []
    #     if not Topology.IsInstance(wire, "Wire"):
    #         if not silent:
    #             print("Face.ByWire - Error: The input wire parameter is not a valid topologic wire. Returning None.")
    #             curframe = inspect.currentframe()
    #             calframe = inspect.getouterframes(curframe, 2)
    #             print('caller name:', calframe[1][3])
    #         return None
    #     if not Wire.IsClosed(wire):
    #         wire = Wire.Close(wire, tolerance=tolerance, silent=silent)
    #         if wire is None or Wire.IsClosed(wire) == False:
    #             if not silent:
    #                 print("Face.ByWire - Error: The input wire parameter is not a closed topologic wire. Returning None.")
    #                 curframe = inspect.currentframe()
    #                 calframe = inspect.getouterframes(curframe, 2)
    #                 print('caller name:', calframe[1][3])
    #             return None
        
    #     edges = Wire.Edges(wire)
    #     wire = Topology.SelfMerge(Cluster.ByTopologies(edges), tolerance=tolerance)
    #     vertices = Topology.Vertices(wire)
    #     fList = []
    #     if Topology.IsInstance(wire, "Wire"):
    #         try:
    #             fList = Core.Face.ByExternalBoundary(wire)
    #         except:
    #             if not silent:
    #                 print("Face.ByWire - Warning: Could not create face by external boundary. Trying other methods.")
    #             if len(vertices) > 3:
    #                 fList = triangulateWire(wire)
    #             else:
    #                 fList = []
        
    #     if not isinstance(fList, list):
    #         fList = [fList]

    #     returnList = []
    #     for f in fList:
    #         if Face.Area(f) < 0:
    #             wire = Face.ExternalBoundary(f)
    #             wire = Wire.Invert(wire)
    #             try:
    #                 f = Core.Face.ByExternalBoundary(wire)
    #                 returnList.append(f)
    #             except:
    #                 pass
    #         else:
    #             returnList.append(f)
    #     if len(returnList) == 0:
    #         if not silent:
    #             print("Face.ByWire - Error: Could not build a face from the input wire parameter. Returning None.")
    #         return None
    #     elif len(returnList) == 1:
    #         return returnList[0]
    #     else:
    #         if not silent:
    #             print("Face.ByWire - Warning: Could not build a single face from the input wire parameter. Returning a list of faces.")
    #         return returnList

    @staticmethod
    def ByWire(wire, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a face from the input closed wire.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Face or list
            The created face. If the wire is non-planar, the method will attempt to
            triangulate the wire and return a list of faces.
        """

        try:
            from topologicpy.Core import Core
        except:
            import topologic

        from topologicpy.Wire import Wire
        from topologicpy.Shell import Shell
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        def _msg(text):
            if not silent:
                print(text)

        def _as_list(obj):
            if obj is None:
                return []
            if isinstance(obj, list):
                return obj
            return [obj]

        def _valid_faces(faces):
            return [f for f in _as_list(faces) if Topology.IsInstance(f, "Face")]

        def _face_by_external_boundary(a_wire):
            try:
                return Core.Face.ByExternalBoundary(a_wire)
            except Exception:
                return None

        def _merged_wire(a_wire):
            """
            Expensive cleanup path. Only used after direct face creation fails.
            """
            try:
                edges = Wire.Edges(a_wire)
                if not edges:
                    return None
                merged = Topology.SelfMerge(
                    Cluster.ByTopologies(edges),
                    tolerance=tolerance
                )
                if Topology.IsInstance(merged, "Wire"):
                    return merged
            except Exception:
                pass
            return None

        def _triangulate_wire(a_wire):
            """
            Fallback for wires that cannot be converted into a single face,
            typically because they are non-planar or geometrically problematic.
            """
            try:
                clean_wire = Topology.RemoveCollinearEdges(
                    a_wire,
                    angTolerance=0.1,
                    tolerance=tolerance,
                    silent=silent
                )
                vertices = Topology.Vertices(clean_wire)
                if len(vertices) < 3:
                    return []

                shell = Shell.Delaunay(vertices)
                if Topology.IsInstance(shell, "Topology"):
                    return _valid_faces(Topology.Faces(shell))
            except Exception:
                pass

            return []

        def _fix_orientation(face):
            """
            Preserve the original method's behaviour: if Face.Area reports a negative
            area, rebuild the face from an inverted external boundary.
            """
            try:
                if Face.Area(face) < 0:
                    boundary = Face.ExternalBoundary(face)
                    inverted = Wire.Invert(boundary)
                    rebuilt = _face_by_external_boundary(inverted)
                    if Topology.IsInstance(rebuilt, "Face"):
                        return rebuilt
            except Exception:
                pass

            return face

        # -------------------------------------------------------------------------
        # Validate input
        # -------------------------------------------------------------------------
        if not Topology.IsInstance(wire, "Wire"):
            _msg("Face.ByWire - Error: The input wire parameter is not a valid topologic wire. Returning None.")
            return None

        # -------------------------------------------------------------------------
        # Ensure closure
        # -------------------------------------------------------------------------
        if not Wire.IsClosed(wire):
            wire = Wire.Close(wire, tolerance=tolerance, silent=silent)

            if wire is None or not Wire.IsClosed(wire):
                _msg("Face.ByWire - Error: The input wire parameter is not a closed topologic wire. Returning None.")
                return None

        # -------------------------------------------------------------------------
        # Fast path: try direct core construction first.
        # This avoids expensive edge extraction, clustering, self-merge, and vertex
        # extraction for ordinary valid planar wires.
        # -------------------------------------------------------------------------
        faces = _valid_faces(_face_by_external_boundary(wire))

        # -------------------------------------------------------------------------
        # Cleanup path: if direct construction failed, try self-merging the wire once.
        # -------------------------------------------------------------------------
        if not faces:
            _msg("Face.ByWire - Warning: Could not create face by external boundary. Trying cleaned wire.")

            cleaned_wire = _merged_wire(wire)
            if cleaned_wire is not None:
                faces = _valid_faces(_face_by_external_boundary(cleaned_wire))
                wire = cleaned_wire

        # -------------------------------------------------------------------------
        # Fallback path: triangulate.
        # -------------------------------------------------------------------------
        if not faces:
            faces = _triangulate_wire(wire)

        # -------------------------------------------------------------------------
        # Orientation correction.
        # -------------------------------------------------------------------------
        faces = [_fix_orientation(f) for f in faces]
        faces = _valid_faces(faces)

        # -------------------------------------------------------------------------
        # Return result.
        # -------------------------------------------------------------------------
        if len(faces) == 0:
            _msg("Face.ByWire - Error: Could not build a face from the input wire parameter. Returning None.")
            return None

        if len(faces) == 1:
            return faces[0]

        _msg("Face.ByWire - Warning: Could not build a single face from the input wire parameter. Returning a list of faces.")
        return faces

    @staticmethod
    def ByWires(externalBoundary, internalBoundaries: list = [], tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a face from the input external boundary (closed wire) and the input list of internal boundaries (closed wires).

        Parameters
        ----------
        externalBoundary : topologic_core.Wire
            The input external boundary.
        internalBoundaries : list , optional
            The input list of internal boundaries (closed wires). Default is an empty list.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Face
            The created face.

        """
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology
        import inspect

        if not Topology.IsInstance(externalBoundary, "Wire"):
            if not silent:
                print("Face.ByWires - Error: The input externalBoundary parameter is not a valid topologic wire. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
            return None
        if not Wire.IsClosed(externalBoundary):
            if not silent:
                print("Face.ByWires - Error: The input externalBoundary parameter is not a closed topologic wire. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
            return None
        if not isinstance(internalBoundaries, list):
            if not silent:
                print("Face.ByWires - Error: The input internalBoundaries parameter is not a list. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
            return None
        eb_face= Face.ByWire(externalBoundary)
        eb_area = Face.Area(eb_face)
        # Make sure all internal wires are actually inside the external wire.
        ibList = []
        for ib in internalBoundaries:
            if not (Topology.IsInstance(ib, "Wire") and Wire.IsClosed(ib)):
                if not silent:
                    print("Face.ByWires - Warning: One of the internal wires is not a valid closed wire. Ignoring.")
                    curframe = inspect.currentframe()
                    calframe = inspect.getouterframes(curframe, 2)
                    print('caller name:', calframe[1][3])
                continue
            ib_face = Face.ByWire(ib)
            ib_area = Face.Area(ib_face)
            if ib_area >= eb_area:
                if not silent:
                    print("Face.ByWires - Warning: One of the iinternal wires has an area greater than that of the external wire. Ignoring.")
                    curframe = inspect.currentframe()
                    calframe = inspect.getouterframes(curframe, 2)
                    print('caller name:', calframe[1][3])
                continue
            sp = Topology.SpatialRelationship(ib_face, eb_face)
            if not sp.lower() == "within":
                if not silent:
                    print("Face.ByWires - Warning: One of the internal wires is not within the external wires. Ignoring.")
                    curframe = inspect.currentframe()
                    calframe = inspect.getouterframes(curframe, 2)
                    print('caller name:', calframe[1][3])
                continue
            ibList.append(ib)
        face = None
        try:
            face = Core.Face.ByExternalInternalBoundaries(externalBoundary, ibList, tolerance)
        except:
            if not silent:
                print("Face.ByWires - Error: The operation failed. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
            face = None
        return face

    @staticmethod
    def ByWiresCluster(externalBoundary, internalBoundariesCluster = None, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a face from the input external boundary (closed wire) and the input cluster of internal boundaries (closed wires).

        Parameters
        ----------
        externalBoundary topologic_core.Wire
            The input external boundary (closed wire).
        internalBoundariesCluster : topologic_core.Cluster
            The input cluster of internal boundaries (closed wires). Default is None.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
        Returns
        -------
        topologic_core.Face
            The created face.

        """
        from topologicpy.Wire import Wire
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(externalBoundary, "Wire"):
            if not silent:
                print("Face.ByWiresCluster - Error: The input externalBoundary parameter is not a valid topologic wire. Returning None.")
            return None
        if not Wire.IsClosed(externalBoundary):
            if not silent:
                print("Face.ByWiresCluster - Error: The input externalBoundary parameter is not a closed topologic wire. Returning None.")
            return None
        if not internalBoundariesCluster:
            internalBoundaries = []
        elif not Topology.IsInstance(internalBoundariesCluster, "Cluster"):
            if not silent:
                print("Face.ByWiresCluster - Error: The input internalBoundariesCluster parameter is not a valid topologic cluster. Returning None.")
            return None
        else:
            internalBoundaries = Cluster.Wires(internalBoundariesCluster)
        return Face.ByWires(externalBoundary, internalBoundaries, tolerance=tolerance, silent=silent)

    @staticmethod
    def CHS(origin= None, radius: float = 0.5, thickness: float = 0.25, sides: int = 16, direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a circular hollow section (CHS).

        Parameters
        ----------
        origin : topologic_core.Vertex, optional
            The location of the origin of the CHS. Default is None which results in the CHS being placed at (0, 0, 0).
        radius : float , optional
            The outer radius of the CHS. Default is 0.5.
        thickness : float , optional
            The thickness of the CHS. Default is 0.25.
        direction : list , optional
            The vector representing the up direction of the CHS. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the CHS. This can be "center", "lowerleft", "upperleft", "lowerright", "upperright". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Face
            The created face.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology
        
        if thickness >= radius:
            if not silent:
                print("Face.SHS - Error: The thickness value is larger than or equal to the outer radius value. Returning None.")
            return None
        if origin == None:
            origin = Vertex.Origin()
        
        outer_wire = Wire.Circle(origin=Vertex.Origin(), radius=radius, sides=sides, direction=[0,0,1], placement="center", tolerance=tolerance)
        inner_wire = Wire.Circle(origin=Vertex.Origin(), radius=radius-thickness, sides=sides, direction=[0,0,1], placement="center", tolerance=tolerance)
        return_face = Face.ByWires(outer_wire, [inner_wire])
        if not Topology.IsInstance(return_face, "face"):
            if not silent:
                print("Face.CHS - Error: Could not create the face for the CHS. Returning None.")
            return None
        
        xOffset = 0
        yOffset = 0
        zOffset = 0
        if placement.lower() == "lowerleft":
            xOffset = radius
            yOffset = radius
        elif placement.lower() == "upperleft":
            xOffset = radius
            yOffset = -radius
        elif placement.lower() == "lowerright":
            xOffset = -radius
            yOffset = radius
        elif placement.lower() == "upperright":
            xOffset = -radius
            yOffset = -radius
        return_face = Topology.Translate(return_face, x=xOffset, y=yOffset, z=zOffset)
        return_face = Topology.Place(return_face, originA=Vertex.Origin(), originB=origin)
        if direction != [0, 0, 1]:
            return_face = Topology.Orient(return_face, origin=origin, dirA=[0, 0, 1], dirB=direction)
        return return_face
    
    @staticmethod
    def Circle(origin= None, radius: float = 0.5, sides: int = 16, fromAngle: float = 0.0, toAngle: float = 360.0, direction: list = [0, 0, 1],
                   placement: str = "center", tolerance: float = 0.0001):
        """
        Creates a circle.

        Parameters
        ----------
        origin : topologic_core.Vertex, optional
            The location of the origin of the circle. Default is None which results in the circle being placed at (0, 0, 0).
        radius : float , optional
            The radius of the circle. Default is 1.
        sides : int , optional
            The number of sides of the circle. Default is 16.
        fromAngle : float , optional
            The angle in degrees from which to start creating the arc of the circle. Default is 0.
        toAngle : float , optional
            The angle in degrees at which to end creating the arc of the circle. Default is 360.
        direction : list , optional
            The vector representing the up direction of the circle. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the circle. This can be "center", "lowerleft", "upperleft", "lowerright", or "upperright". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        topologic_core.Face
            The created circle.

        """
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology

        wire = Wire.Circle(origin=origin, radius=radius, sides=sides, fromAngle=fromAngle, toAngle=toAngle, close=True, direction=direction, placement=placement, tolerance=tolerance)
        if not Topology.IsInstance(wire, "Wire"):
            return None
        return Face.ByWire(wire, tolerance=tolerance)

    @staticmethod
    def Compactness(face, mantissa: int = 6) -> float:
        """
        Returns the compactness measure of the input face. See https://en.wikipedia.org/wiki/Compactness_measure_of_a_shape

        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.

        Returns
        -------
        float
            The compactness measure of the input face.

        """
        from topologicpy.Topology import Topology
        from topologicpy.Edge import Edge

        exb = Face.ExternalBoundary(face)
        edges = Topology.Edges(exb)
        perimeter = 0.0
        for anEdge in edges:
            perimeter = perimeter + abs(Edge.Length(anEdge))
        area = abs(Face.Area(face))
        compactness  = 0
        #From https://en.wikipedia.org/wiki/Compactness_measure_of_a_shape

        if area <= 0:
            return None
        if perimeter <= 0:
            return None
        compactness = (math.pi*(2*math.sqrt(area/math.pi)))/perimeter
        return round(compactness, mantissa)

    @staticmethod
    def CompassAngle(face, north: list = None, mantissa: int = 6, tolerance: float = 0.0001) -> float:
        """
        Returns the horizontal compass angle in degrees between the normal vector of the input face and the input vector. The angle is measured in counter-clockwise fashion. Only the first two elements of the vectors are considered.

        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        north : list , optional
            The second vector representing the north direction. Default is the positive YAxis ([0,1,0]).
        mantissa : int, optional
            The length of the desired mantissa. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        float
            The horizontal compass angle in degrees between the direction of the face and the second input vector.

        """
        from topologicpy.Vector import Vector
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(face, "Face"):
            return None
        if not north:
            north = Vector.North()
        dirA = Face.Normal(face, outputType="xyz", mantissa=mantissa)
        return Vector.CompassAngle(vectorA=dirA, vectorB=north, mantissa=mantissa, tolerance=tolerance)

    @staticmethod
    def CrossShape(origin=None,
            width=1,
            length=1,
            a=0.25,
            b=0.25,
            c=None,
            d=None,
            flipHorizontal = False,
            flipVertical = False,
            direction=[0,0,1],
            placement="center",
            tolerance=0.0001,
            silent=False):
        """
        Creates a Cross-shape.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the T-shape. Default is None which results in the Cross-shape being placed at (0, 0, 0).
        width : float , optional
            The overall width of the Cross-shape. Default is 1.0.
        length : float , optional
            The overall length of the Cross-shape. Default is 1.0.
        a : float , optional
            The hortizontal thickness of the vertical arm of the Cross-shape. Default is 0.25.
        b : float , optional
            The vertical thickness of the horizontal arm of the Cross-shape. Default is 0.25.
        c : float , optional
            The distance of the vertical symmetry axis measured from the left side of the Cross-shape. Default is None which results in the Cross-shape being symmetrical on the Y-axis.
        d : float , optional
            The distance of the horizontal symmetry axis measured from the bottom side of the Cross-shape. Default is None which results in the Cross-shape being symmetrical on the X-axis.
        direction : list , optional
            The vector representing the up direction of the Cross-shape. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the Cross-shape. This can be "center", "lowerleft", "upperleft", "lowerright", "upperright". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Face
            The created Cross-shape.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology

        if not isinstance(width, int) and not isinstance(width, float):
            if not silent:
                print("Face.CrossShape - Error: The width input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(length, int) and not isinstance(length, float):
            if not silent:
                print("Face.CrossShape - Error: The length input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(a, int) and not isinstance(a, float):
            if not silent:
                print("Face.CrossShape - Error: The a input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(b, int) and not isinstance(b, float):
            if not silent:
                print("Face.CrossShape - Error: The b input parameter is not a valid number. Returning None.")
            return None
        if c == None:
            c = width/2
        if d == None:
            d = length/2
        if not isinstance(c, int) and not isinstance(c, float):
            if not silent:
                print("Face.CrossShape - Error: The c input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(d, int) and not isinstance(d, float):
            if not silent:
                print("Face.CrossShape - Error: The d input parameter is not a valid number. Returning None.")
        if width <= tolerance:
            if not silent:
                print("Face.CrossShape - Error: The width input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if length <= tolerance:
            if not silent:
                print("Face.CrossShape - Error: The length input parameter must be a positive number  greater than the tolerance input parameter. Returning None.")
            return None
        if a <= tolerance:
            if not silent:
                print("Face.CrossShape - Error: The a input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if b <= tolerance:
            if not silent:
                print("Face.CrossShape - Error: The b input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if c <= tolerance:
            if not silent:
                print("Face.CrossShape - Error: The c input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if d <= tolerance:
            if not silent:
                print("Face.CrossShape - Error: The d input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if a >= (width - tolerance*2):
            if not silent:
                print("Face.CrossShape - Error: The a input parameter must be less than the width input parameter. Returning None.")
            return None
        if b >= (length - tolerance*2):
            if not silent:
                print("Face.CrossShape - Error: The b input parameter must be less than the length input parameter. Returning None.")
            return None
        if c <= (tolerance + a/2):
            if not silent:
                print("Face.CrossShape - Error: The c input parameter must be more than half the a input parameter. Returning None.")
            return None
        if d <= (tolerance + b/2):
            if not silent:
                print("Face.CrossShape - Error: The c input parameter must be more than half the b input parameter. Returning None.")
            return None
        if c >= (width - tolerance - a/2):
            if not silent:
                print("Face.CrossShape - Error: The c input parameter must be less than the width minus half the a input parameter. Returning None.")
            return None
        if d >= (length - tolerance - b/2):
            if not silent:
                print("Face.CrossShape - Error: The c input parameter must be less than the width minus half the b input parameter. Returning None.")
            return None
        if origin == None:
            origin = Vertex.Origin()
        if not Topology.IsInstance(origin, "vertex"):
            if not silent:
                print("Face.CrossShape - Error: The origin input parameter is not a valid topologic vertex. Returning None.")
            return None
        if not isinstance(direction, list):
            if not silent:
                print("Face.CrossShape - Error: The direction input parameter is not a valid list. Returning None.")
            return None
        if not len(direction) == 3:
            if not silent:
                print("Face.CrossShape - Error: The direction input parameter is not a valid vector. Returning None.")
            return None
        cross_shape_wire = Wire.CrossShape(origin=origin,
                                   width=width,
                                   length=length,
                                   a=a,
                                   b=b,
                                   c=c,
                                   d=d,
                                   flipHorizontal=flipHorizontal,
                                   flipVertical=flipVertical,
                                   direction=direction,
                                   placement=placement,
                                   tolerance=tolerance,
                                   silent=silent)
        return Face.ByWire(cross_shape_wire, tolerance=tolerance, silent=silent)

    @staticmethod
    def CShape(origin=None,
            width=1,
            length=1,
            a=0.25,
            b=0.25,
            c=0.25,
            flipHorizontal = False,
            flipVertical = False,
            direction=[0,0,1],
            placement="center",
            tolerance=0.0001,
            silent=False):
        """
        Creates a C-shape.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the C-shape. Default is None which results in the C-shape being placed at (0, 0, 0).
        width : float , optional
            The overall width of the C-shape. Default is 1.0.
        length : float , optional
            The overall length of the C-shape. Default is 1.0.
        a : float , optional
            The hortizontal thickness of the vertical arm of the C-shape. Default is 0.25.
        b : float , optional
            The vertical thickness of the bottom horizontal arm of the C-shape. Default is 0.25.
        c : float , optional
            The vertical thickness of the top horizontal arm of the C-shape. Default is 0.25.
        direction : list , optional
            The vector representing the up direction of the C-shape. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the C-shape. This can be "center", "lowerleft", "upperleft", "lowerright", "upperright". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Face
            The created C-shape.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology

        if not isinstance(width, int) and not isinstance(width, float):
            if not silent:
                print("Face.CShape - Error: The width input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(length, int) and not isinstance(length, float):
            if not silent:
                print("Face.CShape - Error: The length input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(a, int) and not isinstance(a, float):
            if not silent:
                print("Face.CShape - Error: The a input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(b, int) and not isinstance(b, float):
            if not silent:
                print("Face.CShape - Error: The b input parameter is not a valid number. Returning None.")
            return None
        if width <= tolerance:
            if not silent:
                print("Face.CShape - Error: The width input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if length <= tolerance:
            if not silent:
                print("Face.CShape - Error: The length input parameter must be a positive number  greater than the tolerance input parameter. Returning None.")
            return None
        if a <= tolerance:
            if not silent:
                print("Face.CShape - Error: The a input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if b <= tolerance:
            if not silent:
                print("Face.CShape - Error: The b input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if a >= (width - tolerance):
            if not silent:
                print("Face.CShape - Error: The a input parameter must be less than the width input parameter. Returning None.")
            return None
        if b+c >= (length - tolerance):
            if not silent:
                print("Face.CShape - Error: The b and c input parameters must add to less than the length input parameter. Returning None.")
            return None
        if origin == None:
            origin = Vertex.Origin()
        if not Topology.IsInstance(origin, "vertex"):
            if not silent:
                print("Face.CShape - Error: The origin input parameter is not a valid topologic vertex. Returning None.")
            return None
        if not isinstance(direction, list):
            if not silent:
                print("Face.CShape - Error: The direction input parameter is not a valid list. Returning None.")
            return None
        if not len(direction) == 3:
            if not silent:
                print("Face.CShape - Error: The direction input parameter is not a valid vector. Returning None.")
            return None
        c_shape_wire = Wire.CShape(origin=origin,
                                   width=width,
                                   length=length,
                                   a=a,
                                   b=b,
                                   c=c,
                                   flipHorizontal=flipHorizontal,
                                   flipVertical=flipVertical,
                                   direction=direction,
                                   placement=placement,
                                   tolerance=tolerance,
                                   silent=silent)
        return Face.ByWire(c_shape_wire, tolerance=tolerance, silent=silent)

    @staticmethod
    def Edges(face) -> list:
        """
        Returns the edges of the input face.

        Parameters
        ----------
        face : topologic_core.Face
            The input face.

        Returns
        -------
        list
            The list of edges.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(face, "Face"):
            return None
        edges = []
        # _ = face.Edges(None, edges) # H to Core
        try:
            _ = Core.InstanceCall(face, "Edges", None, edges)
        except Exception:
            edges = None
        return edges

    @staticmethod
    def Einstein(origin= None, radius: float = 0.5, direction: list = [0, 0, 1],
                 placement: str = "center", tolerance: float = 0.0001):
        """
        Creates an aperiodic monotile, also called an 'einstein' tile (meaning one tile in German, not the name of the famous physicist). See https://arxiv.org/abs/2303.10798

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the tile. Default is None which results in the tiles first vertex being placed at (0, 0, 0).
        radius : float , optional
            The radius of the hexagon determining the size of the tile. Default is 0.5.
        direction : list , optional
            The vector representing the up direction of the ellipse. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the hexagon determining the location of the tile. This can be "center", or "lowerleft". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        
        Returns
        -------
            topologic_core.Face
                The created Einstein tile.

        """
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology

        wire = Wire.Einstein(origin=origin, radius=radius, direction=direction, placement=placement)
        if not Topology.IsInstance(wire, "Wire"):
            print("Face.Einstein - Error: Could not create base wire for the Einstein tile. Returning None.")
            return None
        f = Face.ByWire(wire, tolerance=tolerance)
        f = Topology.Orient(f, dirA=Face.Normal(f), dirB=direction)
        return f
    
    @staticmethod
    def Ellipse(origin= None, inputMode: int = 1, width: float = 2.0, length: float = 1.0, focalLength: float = 0.866025, eccentricity: float = 0.866025, majorAxisLength: float = 1.0, minorAxisLength: float = 0.5, sides: float = 32, fromAngle: float = 0.0, toAngle: float = 360.0, close: bool = True, direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001):
        """
        Creates an ellipse and returns all its geometry and parameters.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the ellipse. Default is None which results in the ellipse being placed at (0, 0, 0).
        inputMode : int , optional
            The method by which the ellipse is defined. Default is 1.
            Based on the inputMode value, only the following inputs will be considered. The options are:
            1. Width and Length (considered inputs: width, length)
            2. Focal Length and Eccentricity (considered inputs: focalLength, eccentricity)
            3. Focal Length and Minor Axis Length (considered inputs: focalLength, minorAxisLength)
            4. Major Axis Length and Minor Axis Length (considered input: majorAxisLength, minorAxisLength)
        width : float , optional
            The width of the ellipse. Default is 2.0. This is considered if the inputMode is 1.
        length : float , optional
            The length of the ellipse. Default is 1.0. This is considered if the inputMode is 1.
        focalLength : float , optional
            The focal length of the ellipse. Default is 0.866025. This is considered if the inputMode is 2 or 3.
        eccentricity : float , optional
            The eccentricity of the ellipse. Default is 0.866025. This is considered if the inputMode is 2.
        majorAxisLength : float , optional
            The length of the major axis of the ellipse. Default is 1.0. This is considered if the inputMode is 4.
        minorAxisLength : float , optional
            The length of the minor axis of the ellipse. Default is 0.5. This is considered if the inputMode is 3 or 4.
        sides : int , optional
            The number of sides of the ellipse. Default is 32.
        fromAngle : float , optional
            The angle in degrees from which to start creating the arc of the ellipse. Default is 0.
        toAngle : float , optional
            The angle in degrees at which to end creating the arc of the ellipse. Default is 360.
        close : bool , optional
            If set to True, arcs will be closed by connecting the last vertex to the first vertex. Otherwise, they will be left open.
        direction : list , optional
            The vector representing the up direction of the ellipse. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the ellipse. This can be "center", or "lowerleft". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        topologic_core.Face
            The created ellipse

        """
        from topologicpy.Wire import Wire
        w = Wire.Ellipse(origin=origin, inputMode=inputMode, width=width, length=length,
                         focalLength=focalLength, eccentricity=eccentricity,
                         majorAxisLength=majorAxisLength, minorAxisLength=minorAxisLength,
                         sides=sides, fromAngle=fromAngle, toAngle=toAngle,
                         close=close, direction=direction,
                         placement=placement, tolerance=tolerance)
        return Face.ByWire(w)
    
    @staticmethod
    def ExteriorAngles(face, includeInternalBoundaries=False, mantissa: int = 6) -> list:
        """
        Returns the exterior angles of the input face in degrees. The face must be planar.
        
        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        includeInternalBoundaries : bool , optional
            If set to True and if the face has internal boundaries (holes), the returned list will be a nested list where the first list is the list
            of exterior angles of the external boundary and the second list will contain lists of the exterior angles of each of the
            internal boundaries (holes). For example: [[270,270,270,270], [[270,270,270,270],[300,300,300]]]. If not, the returned list will be
            a simple list of interior angles of the external boundary. For example: [270,270,270,270]. Please note that that the interior angles of the
            internal boundaries are considered to be those interior to the original face. Thus, they are exterior to the internal boundary.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        Returns
        -------
        list
            The list of exterior angles.
        """
        
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(face, "Face"):
            print("Face.ExteriorAngles - Error: The input face parameter is not a valid face. Returning None.")
            return None
        eb = Face.ExternalBoundary(face)
        return_list = Wire.ExteriorAngles(eb, mantissa=mantissa)
        if includeInternalBoundaries:
            internal_boundaries = Face.InternalBoundaries(face)
            ib_i_a_list = []
            if len(internal_boundaries) > 0:
                for ib in internal_boundaries:
                    ib_interior_angles = Wire.InteriorAngles(ib, mantissa=mantissa)
                    ib_i_a_list.append(ib_interior_angles)
            if len(ib_i_a_list) > 0:
                return_list = [return_list]+[ib_i_a_list]
        return return_list

    @staticmethod
    def ExternalBoundary(face, tolerance: float = 0.0001, silent=False):
        """
        Returns the external boundary (closed wire) of the input face.

        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Wire
            The external boundary of the input face.

        """
        from topologicpy.Vector import Vector
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(face, "face"):
            if not silent:
                print("Face.ExternalBoundary - Error: The input face parameter is not a topologic face. Returning None.")
            return None
        # eb = face.ExternalBoundary() # H to Core
        eb = Core.InstanceCall(face, "ExternalBoundary")
        return eb
    
    @staticmethod
    def FacingToward(face, direction: list = [0,0,-1], asVertex: bool = False, mantissa: int = 6, tolerance: float = 0.0001) -> bool:
        """
        Returns True if the input face is facing toward the input direction.

        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        direction : list , optional
            The input direction. Default is [0,0,-1].
        asVertex : bool , optional
            If set to True, the direction is treated as an actual vertex in 3D space. Default is False.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        bool
            True if the face is facing toward the direction. False otherwise.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Vector import Vector

        faceNormal = Face.Normal(face, mantissa=mantissa)
        faceCenter = Face.VertexByParameters(face,0.5,0.5)
        cList = [Vertex.X(faceCenter, mantissa=mantissa), Vertex.Y(faceCenter, mantissa=mantissa), Vertex.Z(faceCenter, mantissa=mantissa)]
        if asVertex:
            dV = [direction[0]-cList[0], direction[1]-cList[1], direction[2]-cList[2]]
        else:
            dV = direction
        uV = Vector.Normalize(dV)
        dot = sum([i*j for (i, j) in zip(uV, faceNormal)])
        if dot <= tolerance:
            return False
        return True

    @staticmethod
    def Fillet(face, radius: float = 0, sides: int = 16, radiusKey: str = None, tolerance: float = 0.0001, silent: bool = False):
        """
        Fillets (rounds) the interior and exterior corners of the input face given the input radius. See https://en.wikipedia.org/wiki/Fillet_(mechanics)

        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        radius : float , optional
            The desired radius of the fillet. Default is 0.
        sides : int , optional
            The number of sides (segments) of the fillet. Default is 16.
        radiusKey : str , optional
            If specified, the dictionary of the vertices will be queried for this key to specify the desired fillet radius. Default is None.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Face
            The filleted face.

        """
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(face, "Face"):
            if not silent:
                print("Face.Fillet - Error: The input face parameter is not a valid face. Returning None.")
            return None
        
        eb = Topology.Copy(Face.ExternalBoundary(face))
        ib_list = Face.InternalBoundaries(face)
        ib_list = [Topology.Copy(ib) for ib in ib_list]
        f_vertices = Topology.Vertices(face)
        if isinstance(radiusKey, str):
            eb = Topology.TransferDictionariesBySelectors(eb, selectors=f_vertices, tranVertices=True)
        eb = Wire.Fillet(eb, radius=radius, sides=sides, radiusKey=radiusKey, tolerance=tolerance, silent=True)
        if not Topology.IsInstance(eb, "Wire"):
            if not silent:
                print("Face.Fillet - Error: The operation failed. Returning None.")
            return None
        ib_wires = []
        for ib in ib_list:
            ib = Wire.ByVertices(Topology.Vertices(ib), close=True, tolerance=tolerance, silent=silent)
            ib = Wire.Reverse(ib)
            if isinstance(radiusKey, str):
                ib = Topology.TransferDictionariesBySelectors(ib, selectors=f_vertices, tranVertices=True)
            
            ib_wire = Wire.Fillet(ib, radius=radius, sides=sides, radiusKey=radiusKey, tolerance=tolerance, silent=True)
            if Topology.IsInstance(ib, "Wire"):
                ib_wires.append(ib_wire)
            else:
                if not silent:
                    print("Face.Fillet - Error: The operation for one of the interior boundaries failed. Skipping.")
        return Face.ByWires(eb, ib_wires)

    @staticmethod
    def Harmonize(face, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns a harmonized version of the input face such that the *u* and *v* origins are always in the upperleft corner.

        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Face
            The harmonized face.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        if not Topology.IsInstance(face, "Face"):
            if not silent:
                print("Face.Harmonize - Error: The input face parameter is not a valid face. Returning None.")
            return None
        normal = Face.Normal(face)
        origin = Topology.Centroid(face)
        flatFace = Topology.Flatten(face, origin=origin, direction=normal)
        world_origin = Vertex.Origin()
        vertices = Topology.Vertices(Face.ExternalBoundary(flatFace))
        harmonizedEB = Wire.ByVertices(vertices, close=True, tolerance=tolerance, silent=silent)
        internalBoundaries = Face.InternalBoundaries(flatFace)
        harmonizedIB = []
        for ib in internalBoundaries:
            ibVertices = Topology.Vertices(ib)
            harmonizedIB.append(Wire.ByVertices(ibVertices, close=True, tolerance=tolerance, silent=silent))
        harmonizedFace = Face.ByWires(harmonizedEB, harmonizedIB, tolerance=tolerance)
        harmonizedFace = Topology.Unflatten(harmonizedFace, origin=origin, direction=normal)
        return harmonizedFace

    @staticmethod
    def InteriorAngles(face, includeInternalBoundaries: bool = False, mantissa: int = 6) -> list:
        """
        Returns the interior angles of the input face in degrees. The face must be planar.
        
        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        includeInternalBoundaries : bool , optional
            If set to True and if the face has internal boundaries (holes), the returned list will be a nested list where the first list is the list
            of interior angles of the external boundary and the second list will contain lists of the interior angles of each of the
            internal boundaries (holes). For example: [[90,90,90,90], [[90,90,90,90],[60,60,60]]]. If not, the returned list will be
            a simple list of interior angles of the external boundary. For example: [90,90,90,90]. Please note that that the interior angles of the
            internal boundaries are considered to be those interior to the original face. Thus, they are exterior to the internal boundary.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        Returns
        -------
        list
            The list of interior angles.
        """
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(face, "Face"):
            print("Face.InteriorAngles - Error: The input face parameter is not a valid face. Returning None.")
            return None
        eb = Face.ExternalBoundary(face)
        return_list = Wire.InteriorAngles(eb, mantissa=mantissa)
        if includeInternalBoundaries:
            internal_boundaries = Face.InternalBoundaries(face)
            ib_i_a_list = []
            if len(internal_boundaries) > 0:
                for ib in internal_boundaries:
                    ib_interior_angles = Wire.ExteriorAngles(ib, mantissa=mantissa)
                    ib_i_a_list.append(ib_interior_angles)
            if len(ib_i_a_list) > 0:
                return_list = [return_list]+[ib_i_a_list]
        return return_list
    
    @staticmethod
    def InternalBoundaries(face) -> list:
        """
        Returns the internal boundaries (closed wires) of the input face.

        Parameters
        ----------
        face : topologic_core.Face
            The input face.

        Returns
        -------
        list
            The list of internal boundaries (closed wires).

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(face, "Face"):
            return None
        wires = []
        # _ = face.InternalBoundaries(wires) # H to Core
        try:
            _ = Core.InstanceCall(face, "InternalBoundaries", wires)
        except:
            wires = None
        return wires

    @staticmethod
    def InternalVertex(face, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a vertex guaranteed to be inside the input face.

        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Vertex
            The created vertex.

        """

        import math
        from topologicpy.Vertex import Vertex
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        def _warn(message):
            if not silent:
                print("Face.InternalVertex - Warning:", message)

        def _is_internal(v):
            if v is None:
                return False
            try:
                if Topology.Intersect(v, face, tolerance=tolerance):
                    return not Topology.Intersect(v, Cluster.ByTopologies(Topology.Edges(face)), tolerance=tolerance)
            except TypeError:
                return Vertex.IsInternal(v, face)

        def _coords(v):
            return [
                Vertex.X(v, mantissa=12),
                Vertex.Y(v, mantissa=12),
                Vertex.Z(v, mantissa=12),
            ]

        def _vertex(c):
            return Vertex.ByCoordinates(c[0], c[1], c[2])

        def _sub(a, b):
            return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]

        def _add(a, b):
            return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]

        def _mul(a, s):
            return [a[0] * s, a[1] * s, a[2] * s]

        def _dot(a, b):
            return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

        def _cross(a, b):
            return [
                a[1]*b[2] - a[2]*b[1],
                a[2]*b[0] - a[0]*b[2],
                a[0]*b[1] - a[1]*b[0],
            ]

        def _length(a):
            return math.sqrt(_dot(a, a))

        def _distance(a, b):
            return _length(_sub(a, b))

        def _normalize(a):
            length = _length(a)
            if length <= tolerance:
                return None
            return [a[0] / length, a[1] / length, a[2] / length]

        def _deduplicate_coords(coords):
            unique = []
            for c in coords:
                duplicate = False
                for u in unique:
                    if _distance(c, u) <= tolerance:
                        duplicate = True
                        break
                if not duplicate:
                    unique.append(c)
            return unique

        def _distance_to_boundary(v, boundary):
            try:
                return Vertex.Distance(v, boundary)
            except Exception:
                return 0

        if not Topology.IsInstance(face, "Face"):
            _warn("The input face is not a valid topologic face. Returning None.")
            return None

        # 1. Try the face centroid first.
        try:
            centroid = Topology.VerticesCentroid(face, mantissa=17)
            if _is_internal(centroid):
                return centroid
        except Exception:
            centroid = None

        # 2. Try verified centroids of triangulated sub-faces.
        candidates = []

        try:
            shell = Topology.Triangulate(face)
            tri_faces = Topology.Faces(shell)

            for tri_face in tri_faces:
                try:
                    tri_centroid = Topology.VerticesCentroid(tri_face, mantissa=17)
                    # if _is_internal(tri_centroid): no need to check, a triangle centroid is intrenal unless the triangle is degenerate
                    candidates.append(tri_centroid)
                except Exception:
                    continue
        except Exception:
            tri_faces = []


        if candidates:
            boundary = Cluster.ByTopologies(Topology.Edges(face))
            # Creating a list of distance, candidate tuples for optimal sorting
            distanceCandidates = [(_distance_to_boundary(c, boundary), c) for c in candidates]
            try:
                distanceCandidates.sort(
                    reverse=True
                )
            except Exception:
                pass
            return distanceCandidates[0][1] # returning the candidate from the first tuple

        # 3. Fallback: construct a robust local coordinate system.
        try:
            vertices = Topology.Vertices(face)
        except Exception:
            vertices = []

        if not vertices or len(vertices) < 3:
            _warn("Could not extract enough vertices from the input face. Returning None.")
            return None

        coords = _deduplicate_coords([_coords(v) for v in vertices])

        if len(coords) < 3:
            _warn("The input face has fewer than three unique vertices. Returning None.")
            return None

        # Use the two furthest-apart vertices as the first local axis.
        max_dist = -1
        origin = None
        axis_point = None

        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                d = _distance(coords[i], coords[j])
                if d > max_dist:
                    max_dist = d
                    origin = coords[i]
                    axis_point = coords[j]

        if origin is None or axis_point is None or max_dist <= tolerance:
            _warn("Could not construct a local coordinate system for the face. Returning None.")
            return None

        e1 = _normalize(_sub(axis_point, origin))

        if e1 is None:
            _warn("Could not construct a valid first local axis. Returning None.")
            return None

        # Find the point furthest from the first axis to define a stable normal.
        normal = None
        max_area = -1

        for c in coords:
            v = _sub(c, origin)
            n = _cross(e1, v)
            area = _length(n)

            if area > max_area:
                max_area = area
                normal = n

        normal = _normalize(normal)

        if normal is None:
            _warn("Could not determine a valid face normal. Returning None.")
            return None

        e2 = _normalize(_cross(normal, e1))

        if e2 is None:
            _warn("Could not construct a valid second local axis. Returning None.")
            return None

        # Project face vertices to local 2D coordinates.
        uv_coords = []

        for c in coords:
            v = _sub(c, origin)
            uv_coords.append([_dot(v, e1), _dot(v, e2)])

        min_u = min(uv[0] for uv in uv_coords)
        max_u = max(uv[0] for uv in uv_coords)
        min_v = min(uv[1] for uv in uv_coords)
        max_v = max(uv[1] for uv in uv_coords)

        if abs(max_u - min_u) <= tolerance or abs(max_v - min_v) <= tolerance:
            _warn("The face has a degenerate local bounding box. Returning None.")
            return None

        try:
            boundary = Cluster.ByTopologies(Topology.Edges(face))
        except Exception:
            boundary = None

        best_vertex = None
        best_distance = -1

        # 4. Sample the face plane. Return only a point verified as internal.
        for resolution in [3, 5, 9, 17, 33, 65, 129]:
            du = (max_u - min_u) / resolution
            dv = (max_v - min_v) / resolution

            for i in range(resolution):
                u = min_u + (i + 0.5) * du

                for j in range(resolution):
                    v = min_v + (j + 0.5) * dv

                    p = _add(origin, _add(_mul(e1, u), _mul(e2, v)))
                    candidate = _vertex(p)

                    if not _is_internal(candidate):
                        continue

                    if boundary:
                        d = _distance_to_boundary(candidate, boundary)
                    else:
                        d = 0

                    if d > best_distance:
                        best_vertex = candidate
                        best_distance = d

            if best_vertex:
                return best_vertex

        _warn("Could not find an internal vertex for the input face. Returning None.")
        return None

    @staticmethod
    def Invert(face, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a face that is an inverse (mirror) of the input face.

        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Face
            The inverted face.

        """
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(face, "Face"):
            return None
        eb = Face.ExternalBoundary(face)
        vertices = Topology.Vertices(eb)
        vertices.reverse()
        inverted_wire = Wire.ByVertices(vertices, close=Wire.IsClosed(eb), tolerance=tolerance, silent=silent)
        internal_boundaries = Face.InternalBoundaries(face)
        if not internal_boundaries:
            inverted_face = Face.ByWire(inverted_wire, tolerance=tolerance, silent=silent)
        else:
            inverted_face = Face.ByWires(inverted_wire, internal_boundaries, tolerance=tolerance, silent=silent)
        return inverted_face
    
    @staticmethod
    def IsConvex(face, mantissa: int = 6, silent: bool = False) -> bool:
        """
        Returns True if the input face is convex. Returns False otherwise.

        Parameters
        ----------
        face : topologic_core.Face
            The  input face.
        mantissa : int , optional
            The length of the desired mantissa. Default is 6.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        Returns
        -------
        bool
            True if the nput face is convex. False otherwise.

        """
        from topologicpy.Topology import Topology
        
        if not Topology.IsInstance(face, "face"):
            if not silent:
                print("Face.IsConvex - Error: The input face parameter is not a valid topologic face. Returning None.")
            return None
        eb = Face.ExternalBoundary(face)
        eb = Face.ByWire(eb)
        return all(Face.InteriorAngles(eb)) < 180

    @staticmethod
    def IsCoplanar(faceA, faceB, mantissa: int = 6, tolerance: float = 0.0001) -> bool:
        """
        Returns True if the two input faces are coplanar. Returns False otherwise.

        Parameters
        ----------
        faceA : topologic_core.Face
            The first input face.
        faceB : topologic_core.Face
            The second input face
        mantissa : int , optional
            The length of the desired mantissa. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        bool
            True if the two input faces are coplanar. False otherwise.

        """
        from topologicpy.Topology import Topology
        import warnings

        try:
            import numpy as np
        except:
            print("Face.IsCoplanar - Information: Installing required numpy library.")
            try:
                os.system("pip install numpy")
            except:
                os.system("pip install numpy --user")
            try:
                import numpy as np
                print("Face.IsCoplanar - Information: numpy library installed successfully.")
            except:
                warnings.warn("Face.IsCoplanar - Error:: Could not import numpy. Please install the numpy library manually. Returning None.")
                return None

        if not Topology.IsInstance(faceA, "Face"):
            print("Face.IsCoplanar - Error: The input faceA parameter is not a valid topologic face. Returning None.")
            return None
        if not Topology.IsInstance(faceB, "Face"):
            print("Face.IsCoplanar - Error: The input faceB parameter is not a valid topologic face. Returning None.")
            return None

        def normalize_plane_coefficients(plane):
            norm = np.linalg.norm(plane[:3])  # Normalize using the first three coefficients (a, b, c)
            if norm == 0:
                return plane
            return [coef / norm for coef in plane]

        def are_planes_coplanar(plane1, plane2, tolerance=0.0001):
            normalized_plane1 = normalize_plane_coefficients(plane1)
            normalized_plane2 = normalize_plane_coefficients(plane2)
            return np.allclose(normalized_plane1, normalized_plane2, atol=tolerance)
        
        eq_a = Face.PlaneEquation(faceA, mantissa=mantissa)
        plane_a = [eq_a['a'], eq_a['b'], eq_a['c'], eq_a['d']]
        plane_a = normalize_plane_coefficients(plane_a)
        eq_b = Face.PlaneEquation(faceB, mantissa=mantissa)
        plane_b = [eq_b['a'], eq_b['b'], eq_b['c'], eq_b['d']]
        plane_b = normalize_plane_coefficients(plane_b)
        return are_planes_coplanar(plane_a, plane_b, tolerance=tolerance)

    @staticmethod
    def IShape(origin=None,
            width=1,
            length=1,
            a=0.25,
            b=0.25,
            c=0.25,
            flipHorizontal = False,
            flipVertical = False,
            direction=[0,0,1],
            placement="center",
            tolerance=0.0001,
            silent=False):
        """
        Creates an I-shape.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the I-shape. Default is None which results in the I-shape being placed at (0, 0, 0).
        width : float , optional
            The overall width of the I-shape. Default is 1.0.
        length : float , optional
            The overall length of the I-shape. Default is 1.0.
        a : float , optional
            The hortizontal thickness of the central vertical arm of the I-shape. Default is 0.25.
        b : float , optional
            The vertical thickness of the bottom horizontal arm of the I-shape. Default is 0.25.
        c : float , optional
            The vertical thickness of the top horizontal arm of the I-shape. Default is 0.25.
        flipHorizontal : bool , optional
            if set to True, the shape is flipped horizontally. Default is False.
        flipVertical : bool , optional
            if set to True, the shape is flipped vertically. Default is False.
        direction : list , optional
            The vector representing the up direction of the I-shape. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the I-shape. This can be "center", "lowerleft", "upperleft", "lowerright", "upperright". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Face
            The created I-shape.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology

        if not isinstance(width, int) and not isinstance(width, float):
            if not silent:
                print("Face.IShape - Error: The width input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(length, int) and not isinstance(length, float):
            if not silent:
                print("Face.IShape - Error: The length input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(a, int) and not isinstance(a, float):
            if not silent:
                print("Face.IShape - Error: The a input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(b, int) and not isinstance(b, float):
            if not silent:
                print("Face.IShape - Error: The b input parameter is not a valid number. Returning None.")
            return None
        if width <= tolerance:
            if not silent:
                print("Face.IShape - Error: The width input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if length <= tolerance:
            if not silent:
                print("Face.IShape - Error: The length input parameter must be a positive number  greater than the tolerance input parameter. Returning None.")
            return None
        if a <= tolerance:
            if not silent:
                print("Face.IShape - Error: The a input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if b <= tolerance:
            if not silent:
                print("Face.IShape - Error: The b input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if a >= (width - tolerance):
            if not silent:
                print("Face.IShape - Error: The a input parameter must be less than the width input parameter. Returning None.")
            return None
        if b+c >= (length - tolerance):
            if not silent:
                print("Face.IShape - Error: The b and c input parameters must add to less than the length input parameter. Returning None.")
            return None
        if origin == None:
            origin = Vertex.Origin()
        if not Topology.IsInstance(origin, "vertex"):
            if not silent:
                print("Face.IShape - Error: The origin input parameter is not a valid topologic vertex. Returning None.")
            return None
        if not isinstance(direction, list):
            if not silent:
                print("Face.IShape - Error: The direction input parameter is not a valid list. Returning None.")
            return None
        if not len(direction) == 3:
            if not silent:
                print("Face.IShape - Error: The direction input parameter is not a valid vector. Returning None.")
            return None
        i_shape_wire = Wire.IShape(origin=origin,
                                   width=width,
                                   length=length,
                                   a=a,
                                   b=b,
                                   c=c,
                                   flipHorizontal=flipHorizontal,
                                   flipVertical=flipVertical,
                                   direction=direction,
                                   placement=placement,
                                   tolerance=tolerance,
                                   silent=silent)
        return Face.ByWire(i_shape_wire, tolerance=tolerance, silent=silent)

    # @staticmethod
    # def Isovist(face, vertex, obstacles: list = [], direction: list = [0,1,0], fov: float = 360, transferDictionaries: bool = False, metrics: bool = False, triangles: bool = False, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
    #     """
    #     Returns the face representing the isovist projection from the input viewpoint.
    #     This method assumes all input is in 2D. Z coordinates are ignored.
    #     This method and the metrics are largely derived from isovists.org. Even if not explicitly listed, please assume that all credit
    #     goes to the authors of that website and its associated software.

    #     Parameters
    #     ----------
    #     face : topologic_core.Face
    #         The face representing the boundary of the isovist.
    #     vertex : topologic_core.Vertex
    #         The vertex representing the location of the viewpoint of the isovist.
    #     obstacles : list , optional
    #         A list of wires representing the obstacles within the face. All obstacles are assumed to be within the
    #         boundary of the face. Default is [].
    #     direction : list, optional
    #         The vector representing the direction (in the XY plane) in which the observer is facing. The Z component is ignored.
    #         The direction follows the Vector.CompassAngle convention where [0,1,0] (North) is considered to be
    #         in the positive Y direction, [1,0,0] (East) is considered to be in the positive X-direction.
    #         Angles are measured in a clockwise fashion. Default is [0,1,0] (North).
    #     fov : float , optional
    #         The horizontal field of view (fov) angle in degrees. See https://en.wikipedia.org/wiki/Field_of_view.
    #         The acceptable range is 1 to 360. Default is 360.
    #     transferDictionaries : bool , optional
    #         If set to True, the dictionaries of the encountered edges will be transfered to the isovist edges. Default is False.
    #     metrics : bool , optional
    #         If set to True, the following metrics are calculated and stored in the dictionary of the returned isovist. The keys of the values are:
    #         - viewpoint : list , the x , y , z coordinates of the location of the viewpoint.
    #         - direction : list , the direction of the view.
    #         - fov : int, Field of view angle.
    #         - area : float , the area of the isovist.
    #         - perimeter : float , the perimeter length of the isovist
    #         - compactness : float , how closely the shape of the isovist approximates a circle (the most compact geometric shape).
    #         - d_max : float, Maximum Visibility Distance. the length of the longest straight line that can be seen from the viewpoint.
    #         - d_min : float, Minimum Visibility Distance. the length of the shortest straight line that can be seen from the viewpoint.
    #         - d_avg : float, Average Visibility Distance. the length of the average straight line that can be seen from the viewpoint.
    #         - v_max : list, Furthest Point measures the x , y , z coordinates of the furthest visible point from the viewpoint.
    #         - v_min : list, Closest Point measures the x , y , z coordinates of the closest visible point from the viewpoint.
    #         - centroid: list, Centroid measures the x, y, z coordinates of the centroid of the isovist face.
    #         - v_d :  list, Visibility Distribution quantifies the angular distribution (in degrees) of visible points across the isovist.
    #                 This metric can tell you whether the visibility from a point is more spread out or concentrated in a certain direction. A uniform visibility distribution indicates a more balanced visual field, while a skewed distribution suggests that the observer's line of sight is constrained in certain directions.
    #         - v_density : float, Viewpoint Density which refers to the number of visible points per unit area within the isovist.
    #         - symmetry : float, Symmetry measures how balanced or symmetrical the isovist is around the point of observation.
    #         - d_f : float, Fractal Dimension measures the complexity of the isovist's boundary.
    #         - e_c : float , Edge Complexity measures how complex the edges of the isovist boundary are.
    #         - theta : float, Mean Visual Field Angle measures the average angular extent of the visible area from the observation point.
    #         - occlusivity: float, the proportion of edges of an isovist that are not physically defined.
    #         - drift: float, the distance from the observation point to the centroid of its isovist.
    #         - closed_perimeter: float, the total length of non-occluded edges of the isovist.
    #         - average_radial: float, "the mean view length of all space visible from a location." (from isovists.org)
    #         - variance: float, "the mean of the square of deviation between all radial lengths and average radial length of an isovist (Benedikt, 1979)." (from isovists.org)
    #         - skewness: float, "the mean of the cube of deviation between all radial lengths and average radial length of an isovist (Benediky, 1979)." (from isovists.org) 
    #     triangles : bool , optional
    #         If set to True, the subtended triangles of the isovist are created and stored as contents of the returned isovist face. Default is False.
    #     mantissa : int , optional
    #         The number of decimal places to round the result to. Default is 6.
    #     tolerance : float , optional:
    #         The desired tolerance. Default is 0.0001.
    #     silent : bool , optional
    #         If set to True, error and warning messages are suppressed. Default is False.

    #     Returns
    #     -------
    #     topologic_core.Face
    #         The face representing the isovist projection from the input viewpoint.

    #     """
    #     from topologicpy.Vertex import Vertex
    #     from topologicpy.Edge import Edge
    #     from topologicpy.Wire import Wire
    #     from topologicpy.Face import Face
    #     from topologicpy.Shell import Shell
    #     from topologicpy.Cluster import Cluster
    #     from topologicpy.Topology import Topology
    #     from topologicpy.Vector import Vector
    #     from topologicpy.Dictionary import Dictionary
    #     from topologicpy.Helper import Helper

    #     import math
    #     import numpy as np

    #     def calculate_angle(viewpoint, vertex):
    #         # Calculate the angle between the viewpoint and the vertex in the 2D plane
    #         # Viewpoint is (x, y, z), and vertex is (x, y)
    #         dx = vertex[0] - viewpoint[0]
    #         dy = vertex[1] - viewpoint[1]
            
    #         # Return the angle in radians using the arctangent of the y/x difference
    #         return math.degrees(math.atan2(dy, dx))
        
    #     def visibility_distribution(viewpoint, isovist_vertices):
    #         angles = []
    #         # Calculate the angle of each vertex with respect to the viewpoint
    #         for vertex in isovist_vertices:
    #             angle = calculate_angle(viewpoint, vertex)
    #             angles.append(angle)
    #         # Sort the angles to analyze the distribution
    #         angles = np.sort(angles)
    #         return list(angles)

    #     def isovist_symmetry(viewpoint, isovist_vertices):
    #         """
    #         Calculates the symmetry of the isovist polygon.
            
    #         Parameters:
    #         - viewpoint: a tuple (x, y) of the viewpoint.
    #         - isovist_vertices: a list of tuples, each representing a vertex (x, y) of the isovist polygon.
            
    #         Returns:
    #         - symmetry value: A measure of the symmetry of the isovist.
    #         """
    #         angles = [calculate_angle(viewpoint, vertex) for vertex in isovist_vertices]
    #         angles.sort()

    #         # Calculate angular deviations from the mean direction
    #         mean_angle = np.mean(angles)
    #         angular_deviation = np.std(angles)
    #         symmetry = angular_deviation / mean_angle if mean_angle != 0 else 0
    #         return float(symmetry)
        
    #     # Fractal Dimension (D_f) using Box-counting
    #     def fractal_dimension(isovist_vertices):
    #         """
    #         Calculates the fractal dimension of the isovist boundary using box-counting.
            
    #         Parameters:
    #         - isovist_vertices: a list of tuples, each representing a vertex (x, y) of the isovist polygon.
            
    #         Returns:
    #         - fractal dimension: A measure of the boundary's complexity.
    #         """
    #         # Convert isovist vertices into a boundary path (x, y coordinates)
    #         boundary_points = np.array(isovist_vertices)

    #         # Box-counting approach
    #         sizes = np.logspace(0, 2, 10)  # Varying box sizes (log scale)
    #         sizes[sizes == 0] = 1e-10  # Replace zero counts with a small value
    #         counts = []
            
    #         for size in sizes:
    #             count = 0
    #             for i in range(len(boundary_points)):
    #                 if np.abs(boundary_points[i][0] - boundary_points[(i+1)%len(boundary_points)][0]) > size or \
    #                 np.abs(boundary_points[i][1] - boundary_points[(i+1)%len(boundary_points)][1]) > size:
    #                     count += 1
    #             counts.append(count)

    #         # To avoid log(0), add a small constant to counts
    #         counts = np.array(counts)
    #         counts = np.where(counts == 0, 1e-10, counts)
    #         # Linear regression of log(count) vs log(size) to estimate fractal dimension
    #         log_sizes = np.log(sizes)
    #         log_counts = np.log(counts)
            
    #         # Perform linear regression (log-log scale)
    #         slope, _ = np.polyfit(log_sizes, log_counts, 1)
            
    #         return slope

    #     # Edge Complexity (E_C)
    #     def edge_complexity(isovist_vertices):
    #         """
    #         Calculates the edge complexity of the isovist boundary.
            
    #         Parameters:
    #         - isovist_vertices: a list of tuples, each representing a vertex (x, y) of the isovist polygon.
            
    #         Returns:
    #         - edge complexity: A measure of the complexity of the boundary.
    #         """
    #         angles = []
    #         for i in range(len(isovist_vertices)):
    #             p1 = isovist_vertices[i]
    #             p2 = isovist_vertices[(i + 1) % len(isovist_vertices)]
    #             p3 = isovist_vertices[(i + 2) % len(isovist_vertices)]

    #             # Calculate the angle between each consecutive edge
    #             angle = np.arctan2(p3[1] - p2[1], p3[0] - p2[0]) - np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
    #             angles.append(np.abs(angle))

    #         # Complexity is the number of abrupt angle changes
    #         complexity = np.sum(np.array(angles) > np.pi / 4)  # e.g., large changes in angles
    #         return float(complexity)

    #     # Mean Visual Field Angle (θ)
    #     def mean_visual_field_angle(viewpoint, isovist_vertices):
    #         """
    #         Calculates the mean visual field angle from the viewpoint to the isovist vertices.
            
    #         Parameters:
    #         - viewpoint: a tuple (x, y) of the viewpoint.
    #         - isovist_vertices: a list of tuples, each representing a vertex (x, y) of the isovist polygon.
            
    #         Returns:
    #         - mean visual field angle in degrees.
    #         """
    #         angles = [calculate_angle(viewpoint, vertex) for vertex in isovist_vertices]
    #         # Return the average angle
    #         return np.mean(angles)

    #     def vertex_part_of_face(vertex, face, tolerance=0.0001):
    #         vertices = Topology.Vertices(face)
    #         for v in vertices:
    #             if Vertex.Distance(vertex, v) <= tolerance:
    #                 return True
    #         return False

    #     if not Topology.IsInstance(face, "Face"):
    #         print("Face.Isovist - Error: The input boundary parameter is not a valid Face. Returning None")
    #         return None
    #     if not Topology.IsInstance(vertex, "Vertex"):
    #         print("Face.Isovist - Error: The input viewPoint parameter is not a valid Vertex. Returning None")
    #         return None
    #     if fov < 1 or fov > 360:
    #         print("Face.Isovist - Error: The input fov parameter is outside the acceptable range of 0 to 360 degrees. Returning None")
    #         return None
    #     if isinstance(obstacles, list):
    #         obstacles = [obs for obs in obstacles if Topology.IsInstance(obs, "Wire")]
    #     else:
    #         obstacles = []
        
    #     def closest_distance_vertex(vertex, edge, mantissa):
    #         point = Vertex.Coordinates(vertex, mantissa=mantissa)
    #         line_start = Vertex.Coordinates(Edge.StartVertex(edge), mantissa=mantissa)
    #         line_end = Vertex.Coordinates(Edge.EndVertex(edge), mantissa=mantissa)

    #         # Convert input points to NumPy arrays for vector operations
    #         point = np.array(point)
    #         line_start = np.array(line_start)
    #         line_end = np.array(line_end)
            
    #         # Calculate the direction vector of the edge
    #         line_direction = line_end - line_start
            
    #         # Vector from the edge's starting point to the point
    #         point_to_start = point - line_start
            
    #         # Calculate the parameter 't' where the projection of the point onto the edge occurs
    #         if np.dot(line_direction, line_direction) == 0:
    #             t = 0
    #         else:
    #             t = np.dot(point_to_start, line_direction) / np.dot(line_direction, line_direction)
            
    #         # Check if 't' is outside the range [0, 1], and if so, calculate distance to closest endpoint
    #         if t < 0:
    #             t = 0
    #         elif t > 1:
    #             t = 1
            
    #         # Calculate the closest point on the edge to the given point
    #         closest_point = line_start + t * line_direction
            
    #         # Calculate the distance between the closest point and the given point
    #         distance = np.linalg.norm(point - closest_point)
            
    #         return float(distance), Vertex.ByCoordinates(list(closest_point))
        
        
        
    #     def compute_average_radial_variance_skewness(vertex, edges, mantissa=6):
    #         from math import atan2, pi, sqrt, pow

    #         def subtended_angle(vertex, edge, mantissa=6):
    #             """Compute the angle subtended by the edge at point V."""
    #             v = Vertex.Coordinates(vertex, mantissa=mantissa)
    #             start = Vertex.Coordinates(Edge.StartVertex(edge), mantissa=mantissa)
    #             end = Vertex.Coordinates(Edge.EndVertex(edge), mantissa=mantissa)
    #             # Calculate the angles of the start and end vertices relative to V
    #             angle_start = atan2(start[1] - v[1], start[0] - v[0])
    #             angle_end = atan2(end[1] - v[1], end[0] - v[0])
    #             # Ensure the angle is in the range [0, 2*pi]
    #             angle_start = angle_start if angle_start >= 0 else angle_start + 2 * pi
    #             angle_end = angle_end if angle_end >= 0 else angle_end + 2 * pi
    #             # Compute the difference and handle wrapping around 2*pi
    #             angle_diff = abs(angle_end - angle_start)
    #             return min(angle_diff, 2 * pi - angle_diff)
        
    #         total_weighted_distance = 0
    #         total_angle_weight = 0
    #         total_weighted_squared_deviation = 0
    #         total_weighted_cubed_deviation = 0
    #         distances = []
    #         angles = []
    #         for edge in edges:
    #             # Calculate the distance between V and the edge
    #             distance = Vertex.Distance(vertex, edge, mantissa=mantissa)
    #             distances.append(distance)

    #             # Calculate the subtended angle for the edge
    #             angle = subtended_angle(vertex, edge, mantissa=mantissa)
    #             angles.append(angle)

    #             # Weight the distance by the subtended angle
    #             total_weighted_distance += distance * angle
    #             total_angle_weight += angle

    #         # Compute the Average Radial value
    #         if total_angle_weight == 0:
    #             average_radial = 0  # Avoid division by zero
    #         else:
    #             average_radial = round(total_weighted_distance / total_angle_weight, mantissa)

    #         # Compute Variance
    #         for i, edge in enumerate(edges):
    #             # Calculate the distance between V and the edge
    #             distance = distances[i]
    #             # Calculate the subtended angle for the edge
    #             angle = angles[i]

    #             # Calculate the deviation squared from the average radial
    #             deviation_squared = (distance - average_radial) ** 2
    #             # Calculate the deviation cubed from the average radial
    #             deviation_cubed = (distance - average_radial) ** 3
    #             # Weight the squared deviation by the subtended angle
    #             total_weighted_squared_deviation += deviation_squared * angle
    #             total_weighted_cubed_deviation += deviation_cubed * angle

    #         # Compute the Variance value
    #         if total_angle_weight == 0:
    #             variance = 0  # Avoid division by zero
    #         else:
    #             variance = round(sqrt(total_weighted_squared_deviation / total_angle_weight), mantissa)
            
    #         # Compute the Skewness value
    #         if total_angle_weight == 0:
    #             skewness = 0  # Avoid division by zero
    #         else:
    #             skewness = round(pow(total_weighted_cubed_deviation / total_angle_weight, 1/3), mantissa)

    #         return average_radial, variance, skewness
        
    #     # Main Code
    #     origin = Topology.Centroid(face)
    #     normal = Face.Normal(face)
    #     flat_face = Topology.Flatten(face, origin=origin, direction=normal)
    #     flat_vertex = Topology.Flatten(vertex, origin=origin, direction=normal)
    #     flat_obstacles = [Topology.Flatten(obstacle, origin=origin, direction=normal) for obstacle in obstacles]

    #     eb = Face.ExternalBoundary(flat_face)
    #     vertices = Topology.Vertices(eb)
    #     coords = [Vertex.Coordinates(v, outputType="xy") for v in vertices]
    #     new_vertices = [Vertex.ByCoordinates(coord) for coord in coords]
    #     eb = Wire.ByVertices(new_vertices, close=True, tolerance=tolerance, silent=silent)

    #     ib_list = Face.InternalBoundaries(flat_face)
    #     new_ib_list = []
    #     for ib in ib_list:
    #         vertices = Topology.Vertices(ib)
    #         coords = [Vertex.Coordinates(v, outputType="xy") for v in vertices]
    #         new_vertices = [Vertex.ByCoordinates(coord) for coord in coords]
    #         new_ib_list.append(Wire.ByVertices(new_vertices, close=True, tolerance=tolerance, silent=silent))

    #     flat_face = Face.ByWires(eb, new_ib_list)
    #     for obs in flat_obstacles:
    #         flat_face = Topology.Difference(flat_face, Face.ByWire(obs))
        
    #     # Check that the viewpoint is inside the face
    #     # if not Vertex.IsInternal(flat_vertex, flat_face):
    #     #     print("Face.Isovist - Error: The viewpoint is not inside the face. Returning None.")
    #     #     return None
    #     targets = Topology.Vertices(flat_face)
    #     distances = []
    #     for target in targets:
    #         distances.append(Vertex.Distance(flat_vertex, target))
    #     distances.sort()
    #     max_d = distances[-1]*1.05
    #     edges = []
    #     for target in targets:
    #         if Vertex.Distance(flat_vertex, target) > tolerance:
    #             e = Edge.ByVertices([flat_vertex, target], tolerance=tolerance, silent=True)
    #             e = Edge.SetLength(e, length=max_d, bothSides=False, tolerance=tolerance)
    #             edges.append(e)
    #     shell = Topology.Slice(flat_face, Cluster.ByTopologies(edges))
    #     faces = Topology.Faces(shell)

    #     final_faces = []
    #     for f in faces:
    #         if vertex_part_of_face(flat_vertex, f, tolerance=0.001):
    #             final_faces.append(f)

    #     if len(final_faces) < 1:
    #         if not silent:
    #             print("Face.Isovist - Error: Could not find visible slice faces. Returning None.")
    #         return None


    #     def _face_by_perimeter_edges(faces, tolerance=0.0001, silent=False):
    #         """
    #         Returns a single face by extracting the perimeter edges from a set of
    #         coplanar connected faces.

    #         Internal shared edges occur twice and are discarded. Boundary edges occur
    #         once and are used to construct the perimeter wire.
    #         """
    #         from topologicpy.Vertex import Vertex
    #         from topologicpy.Edge import Edge
    #         from topologicpy.Wire import Wire
    #         from topologicpy.Face import Face
    #         from topologicpy.Cluster import Cluster
    #         from topologicpy.Topology import Topology

    #         if not isinstance(faces, list) or len(faces) < 1:
    #             return None

    #         all_edges = []
    #         for f in faces:
    #             all_edges += Topology.Edges(f)

    #         if len(all_edges) < 3:
    #             return None

    #         all_vertices = []
    #         for e in all_edges:
    #             all_vertices.append(Edge.StartVertex(e))
    #             all_vertices.append(Edge.EndVertex(e))

    #         fused_vertices = Vertex.Fuse(all_vertices, tolerance=tolerance)

    #         edge_records = {}

    #         for e in all_edges:
    #             sv = Edge.StartVertex(e)
    #             ev = Edge.EndVertex(e)

    #             sv_index = Vertex.Index(sv, fused_vertices, tolerance=tolerance)
    #             ev_index = Vertex.Index(ev, fused_vertices, tolerance=tolerance)

    #             if sv_index is None or ev_index is None:
    #                 continue

    #             if sv_index == ev_index:
    #                 continue

    #             key = tuple(sorted([sv_index, ev_index]))

    #             if key not in edge_records:
    #                 edge_records[key] = {
    #                     "count": 1,
    #                     "sv_index": sv_index,
    #                     "ev_index": ev_index
    #                 }
    #             else:
    #                 edge_records[key]["count"] += 1

    #         perimeter_edges = []

    #         for record in edge_records.values():
    #             if record["count"] == 1:
    #                 sv = fused_vertices[record["sv_index"]]
    #                 ev = fused_vertices[record["ev_index"]]

    #                 if Vertex.Distance(sv, ev) > tolerance:
    #                     perimeter_edges.append(
    #                         Edge.ByVertices([sv, ev], tolerance=tolerance, silent=True)
    #                     )

    #         if len(perimeter_edges) < 3:
    #             return None

    #         wire = Wire.ByEdges(perimeter_edges, tolerance=tolerance, silent=silent)

    #         if not Topology.IsInstance(wire, "Wire"):
    #             wire = Topology.SelfMerge(Cluster.ByTopologies(perimeter_edges))

    #         if not Topology.IsInstance(wire, "Wire"):
    #             return None

    #         return_face = Face.ByWire(wire)

    #         if not Topology.IsInstance(return_face, "Face"):
    #             return None

    #         return return_face


    #     return_face = _face_by_perimeter_edges(
    #         final_faces,
    #         tolerance=tolerance,
    #         silent=silent
    #     )

    #     if not Topology.IsInstance(return_face, "Face"):
    #         if not silent:
    #             print("Face.Isovist - Error: Could not create isovist from perimeter edges. Returning None.")
    #         return None

        

    #     if metrics == True:
    #         vertices = Topology.Vertices(return_face)
    #         # 1 Viewpoint
    #         viewpoint = Vertex.Coordinates(vertex, mantissa=mantissa)
    #         # 2 Direction
    #         # direction is given
    #         # 3 Field of View (FOV)
    #         # fov is given
    #         # 4 Area
    #         area = round(abs(Face.Area(return_face)), mantissa)
    #         # 5 Perimeter
    #         perimeter = round(Wire.Length(Face.Wires(return_face)[0]), mantissa)
    #         # 6 Compactness
    #         compactness = round(Face.Compactness(return_face), mantissa)
    #         # 7 Maximum Distance (d_max)
    #         # 8 Minimum Distance (d_min)
    #         # 9 Average Distance (d_avg)
    #         # 10 Furthest Visible Vertex (v_max)
    #         # 11 Closest Visible Vertex (v_min)
    #         d_max = round(Vertex.Distance(vertex, vertices[0]), mantissa)
    #         d_min = round(Vertex.Distance(vertex, vertices[0]), mantissa)
    #         furthest_vertex = vertices[0]
    #         closest_vertex = vertices[0]
    #         coords = []
    #         distances = []
    #         for v in vertices:
    #             coords.append(Vertex.Coordinates(v, mantissa=mantissa))
    #             dis = Vertex.Distance(vertex, v, mantissa=mantissa)
    #             distances.append(dis)
    #             if dis > d_max:
    #                 d_max  = dis
    #                 furthest_vertex = v
    #         distances = []
    #         edges = Topology.Edges(Cluster.ByTopologies([face]+obstacles))
    #         for edge in edges:
    #             dis, c_v = closest_distance_vertex(vertex, edge, mantissa=mantissa)
    #             if dis < d_min and Vertex.IsPeripheral(c_v, return_face):
    #                 d_min = dis
    #                 closest_vertex = c_v
            
    #         # 12 Average Visible Distance
    #         if len(distances) > 0:
    #             d_avg = sum(distances)/float(len(distances))
    #         else:
    #             d_avg = 0
            
    #         # 10 Furthest Visible Vertex (v_max)
    #         v_max = Vertex.Coordinates(furthest_vertex, mantissa=mantissa)
    #         # 11 Closest Visible Vertex (v_min)
    #         v_min = Vertex.Coordinates(closest_vertex, mantissa=mantissa)
    #         # 12 Centroid of Isovist (centroid)
    #         centroid = Vertex.Coordinates(Topology.Centroid(return_face), mantissa=mantissa)

    #         # 13 Visibility Distribution (v_d)
    #         v_d = visibility_distribution(viewpoint, coords)
    #         v_d = [round(x) for x in v_d]
    #         # 14 Viewpoint density
    #         if abs(Face.Area(return_face)) > 0:
    #             v_density = round(float(len(vertices)) / abs(Face.Area(return_face)), mantissa)
    #         else:
    #             v_density = 0
    #         # 15 Isovist Symmetry
    #         symmetry = round(isovist_symmetry(viewpoint, coords), mantissa)
    #         # 16 Fractal Dimension
    #         d_f = round(fractal_dimension(coords), mantissa)
    #         # 17 Edge Complexity
    #         e_c = round(edge_complexity(coords), mantissa)
    #         # 18 Mean Visual Field Angle
    #         theta = round(mean_visual_field_angle(viewpoint, coords), mantissa)
    #         # 19 Occlusivity
    #         occ_length = 0
    #         edges = Topology.Edges(return_face)
    #         for edge in edges:
    #             d = Topology.Dictionary(edge)
    #             if Dictionary.ValueAtKey(d, "occlusive") == True:
    #                 occ_length += Edge.Length(edge)
    #         if perimeter > 0:
    #             occlusivity = round(occ_length/perimeter, mantissa)
    #         else:
    #             occlusivity = round(0.0, 6)
            
    #         # 20 Drift
    #         drift = Vertex.Distance(vertex, Topology.Centroid(return_face), mantissa=mantissa)

    #         # 21 Closed Perimeter
    #         closed_perimeter = round(perimeter - occ_length, mantissa)

    #         # 22/23/24 Average Radial, Variance, and Skewness
    #         average_radial, variance, skewness = compute_average_radial_variance_skewness(vertex, edges, mantissa=6)

    #         keys = ["viewpoint",
    #                 "direction",
    #                 "fov",
    #                 "area",
    #                 "perimeter",
    #                 "compactness",
    #                 "d_max",
    #                 "d_min",
    #                 "d_avg",
    #                 "v_max",
    #                 "v_min",
    #                 "centroid",
    #                 "v_d",
    #                 "v_density",
    #                 "symmetry",
    #                 "d_f",
    #                 "e_c",
    #                 "theta",
    #                 "occlusivity",
    #                 "drift",
    #                 "closed_perimeter",
    #                 "average_radial",
    #                 "variance",
    #                 "skewness"]
            
    #         values = [viewpoint,
    #                   direction,
    #                   fov,
    #                   area,
    #                   perimeter,
    #                   compactness,
    #                   d_max,
    #                   d_min,
    #                   d_avg,
    #                   v_max,
    #                   v_min,
    #                   centroid,
    #                   v_d,
    #                   v_density,
    #                   symmetry,
    #                   d_f,
    #                   e_c,
    #                   theta,
    #                   occlusivity,
    #                   drift,
    #                   closed_perimeter,
    #                   average_radial,
    #                   variance,
    #                   skewness]
    #         d = Dictionary.ByKeysValues(keys, values)
    #         return_face = Topology.SetDictionary(return_face, d)
    #     if triangles:
    #         triangle_list = []
    #         edges = Topology.Edges(return_face)
    #         for edge in edges:
    #             d = Topology.Dictionary(edge)
    #             if Vertex.Distance(Edge.StartVertex(edge), v) > tolerance:
    #                 e1 = Edge.ByVertices([Edge.StartVertex(edge), v], tolerance=tolerance, silent=True)
    #                 if Vertex.Distance(Edge.EndVertex(edge), v) > tolerance:
    #                     e2 = Edge.ByVertices([Edge.EndVertex(edge), v], tolerance=tolerance, silent=True)
    #                     triangle = Topology.SelfMerge(Cluster.ByTopologies(edge, e1, e2))
    #                     if Topology.IsInstance(triangle, "wire"):
    #                         if Wire.IsClosed(triangle):
    #                             triangle = Face.ByWire(triangle, silent=True)
    #                             if Topology.IsInstance(triangle, "face"):
    #                                 if transferDictionaries == True:
    #                                     triangle = Topology.SetDictionary(triangle, d)
    #                                     tri_edges = Topology.Edges(triangle)
    #                                     for tri_edge in tri_edges:
    #                                         tri_edge = Topology.SetDictionary(tri_edge, d)
    #                                 triangle_list.append(triangle)
    #         if len(triangle_list) > 0:
    #             return_face = Topology.AddContent(return_face, triangle_list)
    #     return return_face




    @staticmethod
    def Isovist(face, vertex, obstacles: list = [], direction: list = [0,1,0], fov: float = 360, transferDictionaries: bool = False, metrics: bool = False, triangles: bool = False, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns the face representing the isovist projection from the input viewpoint.

        This implementation uses a fast 2D visibility algorithm:
        1. The input face, holes, and obstacles are flattened to the XY plane.
        2. The navigable/free-space polygon is constructed using Shapely.
        3. Critical-angle rays are cast from the viewpoint to all polygon vertices.
        4. The nearest boundary intersection along each ray is used to construct the visible polygon.
        5. The resulting Shapely polygon is converted back to a TopologicPy Face.
        6. Metrics and dictionary transfer are preserved.

        This method assumes all visibility computation is carried out in 2D. The input
        topology may be non-horizontal because it is flattened before computation and
        unflattened before return.

        Parameters
        ----------
        face : topologic_core.Face
            The face representing the boundary of the isovist.
        vertex : topologic_core.Vertex
            The vertex representing the viewpoint.
        obstacles : list , optional
            A list of wires representing obstacles within the face. Default is [].
        direction : list, optional
            The vector representing the direction of view in the XY plane. Default is [0,1,0].
        fov : float , optional
            The horizontal field of view angle in degrees. Acceptable range is 1 to 360. Default is 360.
        transferDictionaries : bool , optional
            If True, dictionaries of encountered physical edges are transferred to matching isovist edges.
        metrics : bool , optional
            If True, isovist metrics are calculated and stored in the dictionary of the returned face.
        triangles : bool , optional
            If True, radial triangles from the viewpoint to isovist edges are stored as contents.
        mantissa : int , optional
            Number of decimal places to round results to. Default is 6.
        tolerance : float , optional
            Desired tolerance. Default is 0.0001.
        silent : bool , optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Face
            The face representing the isovist projection from the input viewpoint.
        """

        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Helper import Helper

        import math
        import numpy as np

        try:
            from shapely.geometry import Point, LineString, Polygon, MultiPolygon, GeometryCollection
            from shapely.ops import unary_union
            from shapely.validation import make_valid
        except Exception:
            if not silent:
                print("Face.Isovist - Error: This method requires the shapely package. Please install it using: pip install shapely. Returning None.")
            return None

        # -------------------------------------------------------------------------
        # Basic validation
        # -------------------------------------------------------------------------

        if not Topology.IsInstance(face, "Face"):
            if not silent:
                print("Face.Isovist - Error: The input face parameter is not a valid Face. Returning None.")
            return None

        if not Topology.IsInstance(vertex, "Vertex"):
            if not silent:
                print("Face.Isovist - Error: The input vertex parameter is not a valid Vertex. Returning None.")
            return None

        try:
            fov = float(fov)
        except Exception:
            if not silent:
                print("Face.Isovist - Error: The input fov parameter is not a valid number. Returning None.")
            return None

        if fov < 1 or fov > 360:
            if not silent:
                print("Face.Isovist - Error: The input fov parameter is outside the acceptable range of 1 to 360 degrees. Returning None.")
            return None

        if isinstance(obstacles, list):
            obstacles = [obs for obs in obstacles if Topology.IsInstance(obs, "Wire")]
        else:
            obstacles = []

        # -------------------------------------------------------------------------
        # Helper functions
        # -------------------------------------------------------------------------

        def _safe_dictionary_value(d, key, default_value=None):
            try:
                return Dictionary.ValueAtKey(d, key, default_value)
            except Exception:
                try:
                    value = Dictionary.ValueAtKey(d, key)
                    return default_value if value is None else value
                except Exception:
                    return default_value

        def _xy(vertex_obj):
            c = Vertex.Coordinates(vertex_obj, outputType="xy", mantissa=mantissa)
            return (float(c[0]), float(c[1]))

        def _clean_ring(coords, close=False):
            cleaned = []
            for p in coords:
                if p is None:
                    continue
                x = float(p[0])
                y = float(p[1])
                if len(cleaned) == 0:
                    cleaned.append((x, y))
                else:
                    px, py = cleaned[-1]
                    if math.hypot(x - px, y - py) > tolerance:
                        cleaned.append((x, y))

            if len(cleaned) > 1:
                x0, y0 = cleaned[0]
                x1, y1 = cleaned[-1]
                if math.hypot(x1 - x0, y1 - y0) <= tolerance:
                    cleaned = cleaned[:-1]

            if close and len(cleaned) > 2:
                cleaned.append(cleaned[0])

            return cleaned

        def _wire_to_coords(wire):
            vertices = Topology.Vertices(wire)
            coords = [_xy(v) for v in vertices]
            return _clean_ring(coords, close=False)

        def _wire_to_polygon(wire):
            coords = _wire_to_coords(wire)
            if len(coords) < 3:
                return None
            try:
                poly = Polygon(coords)
                if not poly.is_valid:
                    poly = make_valid(poly)
                if poly.is_empty:
                    return None
                if isinstance(poly, MultiPolygon):
                    poly = max(list(poly.geoms), key=lambda p: p.area)
                if not isinstance(poly, Polygon):
                    return None
                if poly.area <= tolerance:
                    return None
                return poly
            except Exception:
                return None

        def _face_to_polygon(face_obj):
            eb = Face.ExternalBoundary(face_obj)
            exterior = _wire_to_coords(eb)

            if len(exterior) < 3:
                return None

            holes = []
            try:
                internal_boundaries = Face.InternalBoundaries(face_obj)
            except Exception:
                internal_boundaries = []

            for ib in internal_boundaries:
                hole = _wire_to_coords(ib)
                if len(hole) >= 3:
                    holes.append(hole)

            try:
                poly = Polygon(exterior, holes)
                if not poly.is_valid:
                    poly = make_valid(poly)
                if poly.is_empty:
                    return None
                if isinstance(poly, MultiPolygon):
                    poly = max(list(poly.geoms), key=lambda p: p.area)
                if not isinstance(poly, Polygon):
                    return None
                if poly.area <= tolerance:
                    return None
                return poly
            except Exception:
                return None

        def _select_component_containing_point(geom, pt):
            if geom is None or geom.is_empty:
                return None

            if isinstance(geom, Polygon):
                if geom.covers(pt) or geom.distance(pt) <= tolerance:
                    return geom
                return None

            if isinstance(geom, MultiPolygon):
                candidates = []
                for g in geom.geoms:
                    if g.covers(pt) or g.distance(pt) <= tolerance:
                        candidates.append(g)
                if len(candidates) > 0:
                    return max(candidates, key=lambda p: p.area)
                return None

            if isinstance(geom, GeometryCollection):
                polys = []
                for g in geom.geoms:
                    if isinstance(g, Polygon):
                        polys.append(g)
                    elif isinstance(g, MultiPolygon):
                        polys += list(g.geoms)
                candidates = [p for p in polys if p.covers(pt) or p.distance(pt) <= tolerance]
                if len(candidates) > 0:
                    return max(candidates, key=lambda p: p.area)
                return None

            return None

        def _normalise_angle_pi(angle):
            return (angle + math.pi) % (2.0 * math.pi) - math.pi

        def _line_points_from_geometry(geom):
            points = []

            if geom is None or geom.is_empty:
                return points

            geom_type = geom.geom_type

            if geom_type == "Point":
                points.append((float(geom.x), float(geom.y)))

            elif geom_type == "MultiPoint":
                for p in geom.geoms:
                    points.append((float(p.x), float(p.y)))

            elif geom_type in ["LineString", "LinearRing"]:
                coords = list(geom.coords)
                if len(coords) > 0:
                    points.append((float(coords[0][0]), float(coords[0][1])))
                    points.append((float(coords[-1][0]), float(coords[-1][1])))

            elif geom_type == "MultiLineString":
                for line in geom.geoms:
                    coords = list(line.coords)
                    if len(coords) > 0:
                        points.append((float(coords[0][0]), float(coords[0][1])))
                        points.append((float(coords[-1][0]), float(coords[-1][1])))

            elif geom_type == "GeometryCollection":
                for g in geom.geoms:
                    points += _line_points_from_geometry(g)

            return points

        def _nearest_ray_boundary_intersection(poly, origin_xy, angle, ray_length):
            ox, oy = origin_xy
            dx = math.cos(angle)
            dy = math.sin(angle)
            far = (ox + dx * ray_length, oy + dy * ray_length)

            ray = LineString([origin_xy, far])

            try:
                intersection = poly.boundary.intersection(ray)
            except Exception:
                return None

            points = _line_points_from_geometry(intersection)

            best_point = None
            best_distance = None

            for px, py in points:
                vx = px - ox
                vy = py - oy
                along = vx * dx + vy * dy

                if along <= tolerance:
                    continue

                perpendicular = abs(vx * dy - vy * dx)
                if perpendicular > max(tolerance * 10.0, 1e-7):
                    continue

                distance = math.hypot(vx, vy)

                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_point = (px, py)

            return best_point

        def _polygon_vertices(poly):
            verts = []
            if poly is None or poly.is_empty:
                return verts

            if isinstance(poly, Polygon):
                verts += list(poly.exterior.coords)[:-1]
                for interior in poly.interiors:
                    verts += list(interior.coords)[:-1]

            elif isinstance(poly, MultiPolygon):
                for p in poly.geoms:
                    verts += _polygon_vertices(p)

            return [(float(x), float(y)) for x, y in verts]

        def _deduplicate_angles(angles, angular_tolerance=1e-10):
            if len(angles) == 0:
                return []

            angles = sorted(angles)
            result = [angles[0]]

            for a in angles[1:]:
                if abs(a - result[-1]) > angular_tolerance:
                    result.append(a)

            return result

        def _deduplicate_consecutive_points(coords):
            if len(coords) == 0:
                return []

            result = [coords[0]]

            for p in coords[1:]:
                if math.hypot(p[0] - result[-1][0], p[1] - result[-1][1]) > tolerance:
                    result.append(p)

            if len(result) > 2:
                if math.hypot(result[0][0] - result[-1][0], result[0][1] - result[-1][1]) <= tolerance:
                    result = result[:-1]

            return result

        def _shapely_polygon_to_face(poly):
            if poly is None or poly.is_empty:
                return None

            if isinstance(poly, MultiPolygon):
                poly = max(list(poly.geoms), key=lambda p: p.area)

            if not isinstance(poly, Polygon):
                return None

            if poly.area <= tolerance:
                return None

            exterior = _clean_ring(list(poly.exterior.coords), close=False)
            if len(exterior) < 3:
                return None

            exterior_vertices = [Vertex.ByCoordinates([x, y, 0]) for x, y in exterior]
            eb = Wire.ByVertices(exterior_vertices, close=True, tolerance=tolerance, silent=True)

            if not Topology.IsInstance(eb, "Wire"):
                return None

            internal_wires = []
            for interior in poly.interiors:
                hole = _clean_ring(list(interior.coords), close=False)
                if len(hole) >= 3:
                    hole_vertices = [Vertex.ByCoordinates([x, y, 0]) for x, y in hole]
                    ib = Wire.ByVertices(hole_vertices, close=True, tolerance=tolerance, silent=True)
                    if Topology.IsInstance(ib, "Wire"):
                        internal_wires.append(ib)

            if len(internal_wires) > 0:
                result = Face.ByWires(eb, internal_wires)
            else:
                result = Face.ByWire(eb, silent=True)

            if not Topology.IsInstance(result, "Face"):
                return None

            return result

        def _edge_to_linestring(edge):
            sv = Edge.StartVertex(edge)
            ev = Edge.EndVertex(edge)
            p1 = _xy(sv)
            p2 = _xy(ev)
            if math.hypot(p2[0] - p1[0], p2[1] - p1[1]) <= tolerance:
                return None
            return LineString([p1, p2])

        def _edge_overlap_length(edge_a, edge_b):
            line_a = _edge_to_linestring(edge_a)
            line_b = _edge_to_linestring(edge_b)

            if line_a is None or line_b is None:
                return 0.0

            if line_a.distance(line_b) > tolerance:
                return 0.0

            try:
                inter = line_a.intersection(line_b)
            except Exception:
                return 0.0

            if inter.is_empty:
                return 0.0

            try:
                return float(inter.length)
            except Exception:
                return 0.0

        def _tag_and_transfer_edge_dictionaries(isovist_face, physical_edges):
            isovist_edges = Topology.Edges(isovist_face)

            for i_edge in isovist_edges:
                d_i = Topology.Dictionary(i_edge)
                d_i = Dictionary.SetValueAtKey(d_i, "occlusive", True)

                best_edge = None
                best_overlap = 0.0

                for p_edge in physical_edges:
                    overlap = _edge_overlap_length(i_edge, p_edge)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_edge = p_edge

                if best_edge is not None and best_overlap > tolerance:
                    d_i = Dictionary.SetValueAtKey(d_i, "occlusive", False)

                    if transferDictionaries:
                        d_j = Topology.Dictionary(best_edge)
                        d_i = Dictionary.ByMergedDictionaries([d_i, d_j])

                Topology.SetDictionary(i_edge, d_i)

            return isovist_face

        def calculate_angle(viewpoint_coords, vertex_coords):
            dx = vertex_coords[0] - viewpoint_coords[0]
            dy = vertex_coords[1] - viewpoint_coords[1]
            return math.degrees(math.atan2(dy, dx))

        def visibility_distribution(viewpoint_coords, isovist_vertices):
            angles = []
            for v in isovist_vertices:
                angles.append(calculate_angle(viewpoint_coords, v))
            angles = np.sort(angles)
            return list(angles)

        def isovist_symmetry(viewpoint_coords, isovist_vertices):
            if len(isovist_vertices) < 1:
                return 0.0

            angles = [calculate_angle(viewpoint_coords, v) for v in isovist_vertices]
            angles.sort()
            mean_angle = float(np.mean(angles))
            angular_deviation = float(np.std(angles))

            if abs(mean_angle) <= 1e-12:
                return 0.0

            return float(angular_deviation / mean_angle)

        def fractal_dimension(isovist_vertices):
            if len(isovist_vertices) < 3:
                return 0.0

            boundary_points = np.array(isovist_vertices)

            sizes = np.logspace(0, 2, 10)
            counts = []

            for size in sizes:
                count = 0
                for i in range(len(boundary_points)):
                    p1 = boundary_points[i]
                    p2 = boundary_points[(i + 1) % len(boundary_points)]

                    if abs(p1[0] - p2[0]) > size or abs(p1[1] - p2[1]) > size:
                        count += 1

                counts.append(count)

            counts = np.array(counts)
            counts = np.where(counts == 0, 1e-10, counts)

            log_sizes = np.log(sizes)
            log_counts = np.log(counts)

            try:
                slope, _ = np.polyfit(log_sizes, log_counts, 1)
                return float(slope)
            except Exception:
                return 0.0

        def edge_complexity(isovist_vertices):
            if len(isovist_vertices) < 3:
                return 0.0

            angles = []

            for i in range(len(isovist_vertices)):
                p1 = isovist_vertices[i]
                p2 = isovist_vertices[(i + 1) % len(isovist_vertices)]
                p3 = isovist_vertices[(i + 2) % len(isovist_vertices)]

                a1 = np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
                a2 = np.arctan2(p3[1] - p2[1], p3[0] - p2[0])
                angle = abs(a2 - a1)
                angle = min(angle, 2.0 * np.pi - angle)
                angles.append(angle)

            complexity = np.sum(np.array(angles) > np.pi / 4.0)
            return float(complexity)

        def mean_visual_field_angle(viewpoint_coords, isovist_vertices):
            if len(isovist_vertices) < 1:
                return 0.0

            angles = [calculate_angle(viewpoint_coords, v) for v in isovist_vertices]
            return float(np.mean(angles))

        def closest_distance_vertex_to_edge(vertex_obj, edge_obj):
            point = np.array(Vertex.Coordinates(vertex_obj, mantissa=mantissa), dtype=float)
            line_start = np.array(Vertex.Coordinates(Edge.StartVertex(edge_obj), mantissa=mantissa), dtype=float)
            line_end = np.array(Vertex.Coordinates(Edge.EndVertex(edge_obj), mantissa=mantissa), dtype=float)

            line_direction = line_end - line_start
            denominator = float(np.dot(line_direction, line_direction))

            if abs(denominator) <= 1e-12:
                closest_point = line_start
            else:
                t = float(np.dot(point - line_start, line_direction) / denominator)
                t = max(0.0, min(1.0, t))
                closest_point = line_start + t * line_direction

            distance = float(np.linalg.norm(point - closest_point))
            return distance, Vertex.ByCoordinates(list(closest_point))

        def compute_average_radial_variance_skewness(vertex_obj, edges, mantissa=6):
            from math import atan2, pi, sqrt

            def signed_cuberoot(x):
                if x >= 0:
                    return x ** (1.0 / 3.0)
                return -((-x) ** (1.0 / 3.0))

            def subtended_angle(vertex_obj, edge_obj):
                v = Vertex.Coordinates(vertex_obj, mantissa=mantissa)
                start = Vertex.Coordinates(Edge.StartVertex(edge_obj), mantissa=mantissa)
                end = Vertex.Coordinates(Edge.EndVertex(edge_obj), mantissa=mantissa)

                angle_start = atan2(start[1] - v[1], start[0] - v[0])
                angle_end = atan2(end[1] - v[1], end[0] - v[0])

                angle_start = angle_start if angle_start >= 0 else angle_start + 2.0 * pi
                angle_end = angle_end if angle_end >= 0 else angle_end + 2.0 * pi

                angle_diff = abs(angle_end - angle_start)
                return min(angle_diff, 2.0 * pi - angle_diff)

            if len(edges) < 1:
                return 0.0, 0.0, 0.0

            total_weighted_distance = 0.0
            total_angle_weight = 0.0
            distances = []
            angles = []

            for edge_obj in edges:
                distance = Vertex.Distance(vertex_obj, edge_obj, mantissa=mantissa)
                angle = subtended_angle(vertex_obj, edge_obj)

                distances.append(distance)
                angles.append(angle)

                total_weighted_distance += distance * angle
                total_angle_weight += angle

            if total_angle_weight <= 1e-12:
                return 0.0, 0.0, 0.0

            average_radial = total_weighted_distance / total_angle_weight

            total_weighted_squared_deviation = 0.0
            total_weighted_cubed_deviation = 0.0

            for i in range(len(edges)):
                deviation = distances[i] - average_radial
                angle = angles[i]

                total_weighted_squared_deviation += deviation * deviation * angle
                total_weighted_cubed_deviation += deviation * deviation * deviation * angle

            variance = sqrt(total_weighted_squared_deviation / total_angle_weight)
            skewness = signed_cuberoot(total_weighted_cubed_deviation / total_angle_weight)

            return round(average_radial, mantissa), round(variance, mantissa), round(skewness, mantissa)

        # -------------------------------------------------------------------------
        # Flatten input
        # -------------------------------------------------------------------------

        origin = Topology.Centroid(face)
        normal = Face.Normal(face)

        flat_face = Topology.Flatten(face, origin=origin, direction=normal)
        flat_vertex = Topology.Flatten(vertex, origin=origin, direction=normal)
        flat_obstacles = [Topology.Flatten(obstacle, origin=origin, direction=normal) for obstacle in obstacles]

        viewpoint_xy = _xy(flat_vertex)
        viewpoint_point = Point(viewpoint_xy)

        # -------------------------------------------------------------------------
        # Build Shapely free-space polygon
        # -------------------------------------------------------------------------

        boundary_polygon = _face_to_polygon(flat_face)

        if boundary_polygon is None:
            if not silent:
                print("Face.Isovist - Error: Could not convert input face to a valid 2D polygon. Returning None.")
            return None

        obstacle_polygons = []
        for obs in flat_obstacles:
            obs_poly = _wire_to_polygon(obs)
            if obs_poly is not None:
                obstacle_polygons.append(obs_poly)

        try:
            free_space = boundary_polygon

            if len(obstacle_polygons) > 0:
                obstacle_union = unary_union(obstacle_polygons)
                if not obstacle_union.is_valid:
                    obstacle_union = make_valid(obstacle_union)
                free_space = free_space.difference(obstacle_union)

            if not free_space.is_valid:
                free_space = make_valid(free_space)

        except Exception:
            if not silent:
                print("Face.Isovist - Error: Could not construct free-space polygon. Returning None.")
            return None

        free_polygon = _select_component_containing_point(free_space, viewpoint_point)

        if free_polygon is None:
            if not silent:
                print("Face.Isovist - Error: The viewpoint is not inside the navigable face after obstacle subtraction. Returning None.")
            return None

        # -------------------------------------------------------------------------
        # Determine ray length and angular range
        # -------------------------------------------------------------------------

        minx, miny, maxx, maxy = free_polygon.bounds

        corner_distances = [
            math.hypot(minx - viewpoint_xy[0], miny - viewpoint_xy[1]),
            math.hypot(minx - viewpoint_xy[0], maxy - viewpoint_xy[1]),
            math.hypot(maxx - viewpoint_xy[0], miny - viewpoint_xy[1]),
            math.hypot(maxx - viewpoint_xy[0], maxy - viewpoint_xy[1])
        ]

        ray_length = max(corner_distances) * 2.0 + 10.0 * tolerance

        if ray_length <= tolerance:
            if not silent:
                print("Face.Isovist - Error: Could not determine a valid ray length. Returning None.")
            return None

        dx = float(direction[0]) if len(direction) > 0 else 0.0
        dy = float(direction[1]) if len(direction) > 1 else 1.0

        if abs(dx) <= 1e-12 and abs(dy) <= 1e-12:
            dx = 0.0
            dy = 1.0

        facing_angle = math.atan2(dy, dx)
        half_fov = math.radians(fov) * 0.5
        full_view = abs(fov - 360.0) <= 1e-9

        angular_epsilon = max(1e-9, min(1e-5, tolerance / max(ray_length, tolerance)))

        # -------------------------------------------------------------------------
        # Build critical-angle rays
        # -------------------------------------------------------------------------

        candidate_angles = []

        if not full_view:
            candidate_angles.append(facing_angle - half_fov)
            candidate_angles.append(facing_angle + half_fov)

        for vx, vy in _polygon_vertices(free_polygon):
            if math.hypot(vx - viewpoint_xy[0], vy - viewpoint_xy[1]) <= tolerance:
                continue

            angle = math.atan2(vy - viewpoint_xy[1], vx - viewpoint_xy[0])

            if full_view:
                candidate_angles.append(angle - angular_epsilon)
                candidate_angles.append(angle)
                candidate_angles.append(angle + angular_epsilon)
            else:
                relative_angle = _normalise_angle_pi(angle - facing_angle)
                if relative_angle >= -half_fov - angular_epsilon and relative_angle <= half_fov + angular_epsilon:
                    unwrapped_angle = facing_angle + relative_angle
                    candidate_angles.append(unwrapped_angle - angular_epsilon)
                    candidate_angles.append(unwrapped_angle)
                    candidate_angles.append(unwrapped_angle + angular_epsilon)

        if full_view:
            candidate_angles = [_normalise_angle_pi(a) for a in candidate_angles]
            candidate_angles = _deduplicate_angles(candidate_angles)
        else:
            filtered_angles = []
            for a in candidate_angles:
                relative_angle = a - facing_angle
                if relative_angle < -half_fov:
                    a = facing_angle - half_fov
                elif relative_angle > half_fov:
                    a = facing_angle + half_fov
                filtered_angles.append(a)
            candidate_angles = _deduplicate_angles(filtered_angles)

        if len(candidate_angles) < 2:
            if not silent:
                print("Face.Isovist - Error: Could not create enough visibility rays. Returning None.")
            return None

        # -------------------------------------------------------------------------
        # Cast rays
        # -------------------------------------------------------------------------

        hit_records = []

        for angle in candidate_angles:
            hit = _nearest_ray_boundary_intersection(free_polygon, viewpoint_xy, angle, ray_length)
            if hit is not None:
                distance = math.hypot(hit[0] - viewpoint_xy[0], hit[1] - viewpoint_xy[1])
                if distance > tolerance:
                    hit_records.append((angle, hit, distance))

        if len(hit_records) < 2:
            if not silent:
                print("Face.Isovist - Error: Could not find enough ray intersections. Returning None.")
            return None

        hit_records.sort(key=lambda item: item[0])
        hit_points = _deduplicate_consecutive_points([item[1] for item in hit_records])
        ray_distances = [item[2] for item in hit_records]

        if full_view:
            polygon_coords = hit_points
        else:
            polygon_coords = [viewpoint_xy] + hit_points

        polygon_coords = _deduplicate_consecutive_points(polygon_coords)

        if len(polygon_coords) < 3:
            if not silent:
                print("Face.Isovist - Error: The visible polygon has fewer than three vertices. Returning None.")
            return None

        try:
            visible_polygon = Polygon(polygon_coords)

            if not visible_polygon.is_valid:
                visible_polygon = make_valid(visible_polygon)

            visible_polygon = visible_polygon.intersection(free_polygon)

            if not visible_polygon.is_valid:
                visible_polygon = make_valid(visible_polygon)

        except Exception:
            if not silent:
                print("Face.Isovist - Error: Could not construct visible polygon. Returning None.")
            return None

        if isinstance(visible_polygon, MultiPolygon):
            selected = _select_component_containing_point(visible_polygon, viewpoint_point)
            if selected is None:
                selected = max(list(visible_polygon.geoms), key=lambda p: p.area)
            visible_polygon = selected

        elif isinstance(visible_polygon, GeometryCollection):
            selected = _select_component_containing_point(visible_polygon, viewpoint_point)
            if selected is None:
                polys = [g for g in visible_polygon.geoms if isinstance(g, Polygon)]
                if len(polys) > 0:
                    selected = max(polys, key=lambda p: p.area)
            visible_polygon = selected

        if visible_polygon is None or visible_polygon.is_empty or not isinstance(visible_polygon, Polygon):
            if not silent:
                print("Face.Isovist - Error: The visible polygon is invalid or empty. Returning None.")
            return None

        if visible_polygon.area <= tolerance:
            if not silent:
                print("Face.Isovist - Error: The visible polygon has near-zero area. Returning None.")
            return None

        # -------------------------------------------------------------------------
        # Convert Shapely polygon back to TopologicPy face
        # -------------------------------------------------------------------------

        return_face = _shapely_polygon_to_face(visible_polygon)

        if not Topology.IsInstance(return_face, "Face"):
            if not silent:
                print("Face.Isovist - Error: Could not convert visible polygon to a TopologicPy Face. Returning None.")
            return None

        return_face = Topology.RemoveCollinearEdges(return_face, angTolerance=0.1, tolerance=tolerance, silent=True)

        if not Topology.IsInstance(return_face, "Face"):
            return_face = _shapely_polygon_to_face(visible_polygon)

        if not Topology.IsInstance(return_face, "Face"):
            if not silent:
                print("Face.Isovist - Error: Could not create final isovist face. Returning None.")
            return None

        # -------------------------------------------------------------------------
        # Edge dictionary transfer and occlusive tagging
        # -------------------------------------------------------------------------

        if transferDictionaries or metrics:
            physical_edges = []

            physical_edges += Topology.Edges(Face.ExternalBoundary(flat_face))

            try:
                for ib in Face.InternalBoundaries(flat_face):
                    physical_edges += Topology.Edges(ib)
            except Exception:
                pass

            for obs in flat_obstacles:
                physical_edges += Topology.Edges(obs)

            return_face = _tag_and_transfer_edge_dictionaries(return_face, physical_edges)

        # -------------------------------------------------------------------------
        # Metrics
        # -------------------------------------------------------------------------

        metric_dictionary = None

        if metrics:
            isovist_vertices = Topology.Vertices(return_face)
            isovist_edges = Topology.Edges(return_face)

            viewpoint = Vertex.Coordinates(flat_vertex, mantissa=mantissa)
            area = round(abs(Face.Area(return_face)), mantissa)

            try:
                perimeter = round(Wire.Length(Face.Wires(return_face)[0]), mantissa)
            except Exception:
                perimeter = round(float(visible_polygon.length), mantissa)

            try:
                compactness = round(Face.Compactness(return_face), mantissa)
            except Exception:
                if perimeter > 0:
                    compactness = round((4.0 * math.pi * area) / (perimeter * perimeter), mantissa)
                else:
                    compactness = 0.0

            coords = []
            vertex_distances = []

            for v in isovist_vertices:
                vc = Vertex.Coordinates(v, mantissa=mantissa)
                coords.append(vc)
                vertex_distances.append(Vertex.Distance(flat_vertex, v, mantissa=mantissa))

            if len(vertex_distances) > 0:
                d_max = round(max(vertex_distances), mantissa)
                max_index = vertex_distances.index(max(vertex_distances))
                furthest_vertex = isovist_vertices[max_index]
            else:
                d_max = 0.0
                furthest_vertex = flat_vertex

            d_min = None
            closest_vertex = None

            for edge in isovist_edges:
                dis, c_v = closest_distance_vertex_to_edge(flat_vertex, edge)
                if dis <= tolerance:
                    continue
                if d_min is None or dis < d_min:
                    d_min = dis
                    closest_vertex = c_v

            if d_min is None:
                d_min = 0.0
                closest_vertex = flat_vertex

            d_min = round(d_min, mantissa)

            if len(ray_distances) > 0:
                d_avg = round(float(sum(ray_distances)) / float(len(ray_distances)), mantissa)
            elif len(vertex_distances) > 0:
                d_avg = round(float(sum(vertex_distances)) / float(len(vertex_distances)), mantissa)
            else:
                d_avg = 0.0

            v_max = Vertex.Coordinates(furthest_vertex, mantissa=mantissa)
            v_min = Vertex.Coordinates(closest_vertex, mantissa=mantissa)

            centroid = Vertex.Coordinates(Topology.Centroid(return_face), mantissa=mantissa)

            v_d = visibility_distribution(viewpoint, coords)
            v_d = [round(x) for x in v_d]

            if area > 0:
                v_density = round(float(len(isovist_vertices)) / area, mantissa)
            else:
                v_density = 0.0

            symmetry = round(isovist_symmetry(viewpoint, coords), mantissa)
            d_f = round(fractal_dimension(coords), mantissa)
            e_c = round(edge_complexity(coords), mantissa)
            theta = round(mean_visual_field_angle(viewpoint, coords), mantissa)

            occ_length = 0.0

            for edge in isovist_edges:
                d = Topology.Dictionary(edge)
                if _safe_dictionary_value(d, "occlusive", False) == True:
                    occ_length += Edge.Length(edge)

            if perimeter > 0:
                occlusivity = round(occ_length / perimeter, mantissa)
            else:
                occlusivity = 0.0

            drift = round(Vertex.Distance(flat_vertex, Topology.Centroid(return_face), mantissa=mantissa), mantissa)
            closed_perimeter = round(perimeter - occ_length, mantissa)

            average_radial, variance, skewness = compute_average_radial_variance_skewness(flat_vertex, isovist_edges, mantissa=mantissa)

            keys = [
                "viewpoint",
                "direction",
                "fov",
                "area",
                "perimeter",
                "compactness",
                "d_max",
                "d_min",
                "d_avg",
                "v_max",
                "v_min",
                "centroid",
                "v_d",
                "v_density",
                "symmetry",
                "d_f",
                "e_c",
                "theta",
                "occlusivity",
                "drift",
                "closed_perimeter",
                "average_radial",
                "variance",
                "skewness"
            ]

            values = [
                viewpoint,
                direction,
                fov,
                area,
                perimeter,
                compactness,
                d_max,
                d_min,
                d_avg,
                v_max,
                v_min,
                centroid,
                v_d,
                v_density,
                symmetry,
                d_f,
                e_c,
                theta,
                occlusivity,
                drift,
                closed_perimeter,
                average_radial,
                variance,
                skewness
            ]

            metric_dictionary = Dictionary.ByKeysValues(keys, values)
            return_face = Topology.SetDictionary(return_face, metric_dictionary)

        # -------------------------------------------------------------------------
        # Optional radial triangles
        # -------------------------------------------------------------------------

        if triangles:
            triangle_list = []
            isovist_edges = Topology.Edges(return_face)

            for edge in isovist_edges:
                sv = Edge.StartVertex(edge)
                ev = Edge.EndVertex(edge)

                if Vertex.Distance(sv, flat_vertex) <= tolerance:
                    continue

                if Vertex.Distance(ev, flat_vertex) <= tolerance:
                    continue

                e1 = Edge.ByVertices([sv, flat_vertex], tolerance=tolerance, silent=True)
                e2 = Edge.ByVertices([ev, flat_vertex], tolerance=tolerance, silent=True)

                if not Topology.IsInstance(e1, "Edge") or not Topology.IsInstance(e2, "Edge"):
                    continue

                tri_wire = Topology.SelfMerge(Cluster.ByTopologies([edge, e1, e2]))

                if Topology.IsInstance(tri_wire, "Wire") and Wire.IsClosed(tri_wire):
                    tri_face = Face.ByWire(tri_wire, silent=True)

                    if Topology.IsInstance(tri_face, "Face"):
                        if transferDictionaries:
                            d = Topology.Dictionary(edge)
                            tri_face = Topology.SetDictionary(tri_face, d)
                            for tri_edge in Topology.Edges(tri_face):
                                Topology.SetDictionary(tri_edge, d)

                        triangle_list.append(tri_face)

            if len(triangle_list) > 0:
                return_face = Topology.AddContent(return_face, triangle_list)

        # -------------------------------------------------------------------------
        # Unflatten and reattach metric dictionary if needed
        # -------------------------------------------------------------------------

        return_face = Topology.Unflatten(return_face, origin=origin, direction=normal)

        if metrics and metric_dictionary is not None:
            return_face = Topology.SetDictionary(return_face, metric_dictionary)

        return return_face






    @staticmethod
    def LShape(origin=None,
            width=1,
            length=1,
            a=0.25,
            b=0.25,
            flipHorizontal = False,
            flipVertical = False,
            direction=[0,0,1],
            placement="center",
            tolerance=0.0001,
            silent=False):
        """
        Creates an L-shape.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the L-shape. Default is None which results in the L-shape being placed at (0, 0, 0).
        width : float , optional
            The overall width of the L-shape. Default is 1.0.
        length : float , optional
            The overall length of the L-shape. Default is 1.0.
        a : float , optional
            The hortizontal thickness of the vertical arm of the L-shape. Default is 0.25.
        b : float , optional
            The vertical thickness of the horizontal arm of the L-shape. Default is 0.25.
        direction : list , optional
            The vector representing the up direction of the L-shape. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the L-shape. This can be "center", "lowerleft", "upperleft", "lowerright", "upperright". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Face
            The created L-shape.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology

        if not isinstance(width, int) and not isinstance(width, float):
            if not silent:
                print("Face.LShape - Error: The width input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(length, int) and not isinstance(length, float):
            if not silent:
                print("Face.LShape - Error: The length input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(a, int) and not isinstance(a, float):
            if not silent:
                print("Face.LShape - Error: The a input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(b, int) and not isinstance(b, float):
            if not silent:
                print("Face.LShape - Error: The b input parameter is not a valid number. Returning None.")
            return None
        if width <= tolerance:
            if not silent:
                print("Face.LShape - Error: The width input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if length <= tolerance:
            if not silent:
                print("Face.LShape - Error: The length input parameter must be a positive number  greater than the tolerance input parameter. Returning None.")
            return None
        if a <= tolerance:
            if not silent:
                print("Face.LShape - Error: The a input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if b <= tolerance:
            if not silent:
                print("Face.LShape - Error: The b input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if a >= (width - tolerance):
            if not silent:
                print("Face.LShape - Error: The a input parameter must be less than the width input parameter. Returning None.")
            return None
        if b >= (length - tolerance):
            if not silent:
                print("Face.LShape - Error: The b input parameter must be less than the length input parameter. Returning None.")
            return None
        if origin == None:
            origin = Vertex.Origin()
        if not Topology.IsInstance(origin, "vertex"):
            if not silent:
                print("Face.LShape - Error: The origin input parameter is not a valid topologic vertex. Returning None.")
            return None
        if not isinstance(direction, list):
            if not silent:
                print("Face.LShape - Error: The direction input parameter is not a valid list. Returning None.")
            return None
        if not len(direction) == 3:
            if not silent:
                print("Face.LShape - Error: The direction input parameter is not a valid vector. Returning None.")
            return None
        l_shape_wire = Wire.LShape(origin=origin,
                                   width=width,
                                   length=length,
                                   a=a,
                                   b=b,
                                   flipHorizontal=flipHorizontal,
                                   flipVertical=flipVertical,
                                   direction=direction,
                                   placement=placement,
                                   tolerance=tolerance,
                                   silent=silent)
        return Face.ByWire(l_shape_wire, tolerance=tolerance, silent=silent)

    @staticmethod
    def MedialAxis(face, resolution: int = 0, externalVertices: bool = False, internalVertices: bool = False, toLeavesOnly: bool = False, angTolerance: float = 0.1, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns a wire representing an approximation of the medial axis of the input topology. See https://en.wikipedia.org/wiki/Medial_axis.

        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        resolution : int , optional
            The desired resolution of the solution (range is 0: standard resolution to 10: high resolution). This determines the density of the sampling along each edge. Default is 0.
        externalVertices : bool , optional
            If set to True, the external vertices of the face will be connected to the nearest vertex on the medial axis. Default is False.
        internalVertices : bool , optional
            If set to True, the internal vertices of the face will be connected to the nearest vertex on the medial axis. Default is False.
        toLeavesOnly : bool , optional
            If set to True, the vertices of the face will be connected to the nearest vertex on the medial axis only if this vertex is a leaf (end point). Otherwise, it will connect to any nearest vertex. Default is False.
        angTolerance : float , optional
            The desired angular tolerance in degrees for removing collinear edges. Default is 0.1.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
        Returns
        -------
        topologic_core.Wire
            The medial axis of the input face.
        """

        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Shell import Shell
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Face import Face

        def _warn(msg):
            if not silent:
                print("Face.MedialAxis - Warning:", msg)

        def touchesEdge(vertex, edges, tolerance=0.0001):
            if not Topology.IsInstance(vertex, "Vertex"):
                return False

            for edge in edges:
                try:
                    u = Edge.ParameterAtVertex(edge, vertex, mantissa=6)
                except:
                    continue

                if u is None:
                    continue

                if tolerance < u < 1.0 - tolerance:
                    return True

            return False

        def _edges(topology):
            if topology is None:
                return []

            try:
                edges = Topology.Edges(topology)
                if isinstance(edges, list):
                    return [e for e in edges if Topology.IsInstance(e, "Edge")]
            except:
                pass

            if Topology.IsInstance(topology, "Edge"):
                return [topology]

            return []

        def _vertices(topology):
            if topology is None:
                return []

            try:
                vertices = Topology.Vertices(topology)
                if isinstance(vertices, list):
                    return [v for v in vertices if Topology.IsInstance(v, "Vertex")]
            except:
                pass

            if Topology.IsInstance(topology, "Vertex"):
                return [topology]

            return []

        def _safe_edge(v1, v2):
            if not Topology.IsInstance(v1, "Vertex"):
                return None
            if not Topology.IsInstance(v2, "Vertex"):
                return None

            try:
                if Vertex.Distance(v1, v2) <= tolerance:
                    return None
            except:
                pass

            try:
                return Edge.ByVertices(v1, v2, tolerance=tolerance, silent=True)
            except:
                try:
                    return Edge.ByVertices([v1, v2], tolerance=tolerance, silent=True)
                except:
                    return None

        if not Topology.IsInstance(face, "Face"):
            if not silent:
                print("Face.MedialAxis - Error: The input face parameter is not a valid face. Returning None.")
            return None

        # Flatten the input face
        origin = Topology.Centroid(face)
        normal = Face.Normal(face)
        flatFace = Topology.Flatten(face, origin=origin, direction=normal)

        if not Topology.IsInstance(flatFace, "Face"):
            if not silent:
                print("Face.MedialAxis - Error: Could not flatten the input face. Returning None.")
            return None

        faceEdges = Face.Edges(flatFace)

        if not isinstance(faceEdges, list) or len(faceEdges) == 0:
            if not silent:
                print("Face.MedialAxis - Error: Could not retrieve the edges of the input face. Returning None.")
            return None

        # Convert user resolution 0..10 into sampling step.
        resolution = 10 - resolution
        resolution = min(max(resolution, 1), 10)

        vertices = []

        for e in faceEdges:
            for n in range(resolution, 100, resolution):
                v = Edge.VertexByParameter(e, n * 0.01)
                if Topology.IsInstance(v, "Vertex"):
                    vertices.append(v)

        if len(vertices) < 4:
            _warn("Not enough sampled vertices to compute a Voronoi diagram. Returning None.")
            return None

        voronoi = Shell.Voronoi(vertices=vertices, face=flatFace)

        if voronoi is None:
            _warn("Could not compute the Voronoi diagram. Returning None.")
            return None

        voronoiEdges = _edges(voronoi)

        if len(voronoiEdges) == 0:
            _warn("The Voronoi diagram contains no valid edges. Returning None.")
            return None

        medialAxisEdges = []

        for e in voronoiEdges:
            sv = Edge.StartVertex(e)
            ev = Edge.EndVertex(e)

            svTouchesEdge = touchesEdge(sv, faceEdges, tolerance=tolerance)
            evTouchesEdge = touchesEdge(ev, faceEdges, tolerance=tolerance)

            if not svTouchesEdge and not evTouchesEdge:
                medialAxisEdges.append(e)

        if len(medialAxisEdges) == 0:
            _warn("No medial-axis edges were found. Returning None.")
            return None

        extBoundary = Face.ExternalBoundary(flatFace)
        extVertices = _vertices(extBoundary)

        intVertices = []

        try:
            intBoundaries = Face.InternalBoundaries(flatFace)
        except:
            intBoundaries = []

        if isinstance(intBoundaries, list):
            for ib in intBoundaries:
                intVertices += _vertices(ib)

        theVertices = []

        if internalVertices:
            theVertices += intVertices

        if externalVertices:
            theVertices += extVertices

        temp = Cluster.ByTopologies(medialAxisEdges, silent=True)
        tempWire = Topology.SelfMerge(temp, tolerance=tolerance)

        if tempWire is None:
            _warn("Could not merge the medial-axis edges. Returning None.")
            return None

        if Topology.IsInstance(tempWire, "Wire") and angTolerance > 0:
            tempWire = Wire.RemoveCollinearEdges(tempWire, angTolerance=angTolerance, tolerance=tolerance, silent=silent)

        medialAxisEdges = _edges(tempWire)

        if len(medialAxisEdges) == 0:
            _warn("The merged medial axis contains no valid edges. Returning None.")
            return None

        if len(theVertices) > 0:
            for v in theVertices:
                try:
                    nv = Vertex.NearestVertex(v, tempWire, useKDTree=False)
                except:
                    nv = None

                if not Topology.IsInstance(nv, "Vertex"):
                    continue

                if toLeavesOnly:
                    try:
                        adjVertices = Topology.AdjacentTopologies(nv, tempWire)
                    except:
                        adjVertices = []

                    if len(adjVertices) < 2:
                        edge = _safe_edge(nv, v)
                        if edge:
                            medialAxisEdges.append(edge)
                else:
                    edge = _safe_edge(nv, v)
                    if edge:
                        medialAxisEdges.append(edge)

        medialAxis = Topology.SelfMerge(
            Cluster.ByTopologies(medialAxisEdges, silent=True),
            tolerance=tolerance
        )

        if medialAxis is None:
            _warn("Could not create final medial axis. Returning None.")
            return None

        if Topology.IsInstance(medialAxis, "Wire") and angTolerance > 0:
            medialAxis = Topology.RemoveCollinearEdges(
                medialAxis,
                angTolerance=angTolerance,
                tolerance=tolerance,
                silent=silent
            )

        medialAxis = Topology.Unflatten(medialAxis, origin=origin, direction=normal)

        return medialAxis

    @staticmethod
    def Normal(face, outputType="xyz", mantissa=6):
        """
        Returns the normal vector to the input face. A normal vector of a face is a vector perpendicular to it.

        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        outputType : string , optional
            The string defining the desired output. This can be any subset or permutation of "xyz". It is case insensitive. Default is "xyz".
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.

        Returns
        -------
        list
            The normal vector to the input face.

        """
        from topologicpy.Topology import Topology
        from topologicpy.Vertex import Vertex
        import os
        import warnings

        try:
            import numpy as np
        except:
            print("Face.Normal - Warning: Installing required numpy library.")
            try:
                os.system("pip install numpy")
            except:
                os.system("pip install numpy --user")
            try:
                import numpy as np
                print("Face.Normal - Warning: numpy library installed correctly.")
            except:
                warnings.warn("Face.Normal - Error: Could not import numpy. Please try to install numpy manually. Returning None.")
                return None

        if not Topology.IsInstance(face, "Face"):
            print("Face.Normal - Error: The input face parameter is not a valid face. Returning None.")
            return None
        
        return_normal = None
        try:
            return_normal = list(Core.FaceUtility.NormalAtParameters(face, 0.5, 0.5))
        except:
            vertices = Topology.Vertices(face)+Topology.Centroid(face)
            #v1 = Face.VertexByParameters(face, u=0, v=0)
            #v2 = Face.VertexByParameters(face, u=1, v=0)
            #v3 = Face.VertexByParameters(face, u=1, v=1)
            #vertices = [v1, v2, v3]
            vertices = [Vertex.Coordinates(v, mantissa=mantissa) for v in vertices]
            
            if len(vertices) < 3:
                print("Face.Normal - Error: At least three vertices are required to define a plane. Returning None.")
                return None
            
            # Convert vertices to numpy array for easier manipulation
            vertices = np.array(vertices)
            
            # Try to find two non-collinear edge vectors
            vec1 = None
            vec2 = None
            for i in range(1, len(vertices)):
                for j in range(i + 1, len(vertices)):
                    temp_vec1 = vertices[i] - vertices[0]
                    temp_vec2 = vertices[j] - vertices[0]
                    cross_product = np.cross(temp_vec1, temp_vec2)
                    if np.linalg.norm(cross_product) > 1e-6:  # Check if the cross product is not near zero
                        vec1 = temp_vec1
                        vec2 = temp_vec2
                        break
                if vec1 is not None and vec2 is not None:
                    break
            
            if vec1 is None or vec2 is None:
                print("Face.Normal - Error: The given vertices do not form a valid plane (all vertices might be collinear). Returning None.")
                return None
            
            # Calculate the cross product of the two edge vectors
            normal = np.cross(vec1, vec2)

            # Normalize the normal vector
            normal_length = np.linalg.norm(normal)
            if normal_length == 0:
                print("Face.Normal - Error: The given vertices do not form a valid plane (cross product resulted in a zero vector). Returning None.")
                return None
            
            normal = normal / normal_length
            normal = normal.tolist()
            normal = [round(x, mantissa) for x in normal]
            return_normal = []
            outputType = list(outputType.lower())
            for axis in outputType:
                if axis == "x":
                    return_normal.append(normal[0])
                elif axis == "y":
                    return_normal.append(normal[1])
                elif axis == "z":
                    return_normal.append(normal[2])
        return return_normal
    
    @staticmethod
    def NormalEdge(face, length: float = 1.0, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns the normal vector to the input face as an edge with the desired input length. A normal vector of a face is a vector perpendicular to it.

        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        length : float , optional
            The desired length of the normal edge. Default is 1.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Edge
            The created normal edge to the input face. This is computed at the approximate center of the face.

        """
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(face, "Face"):
            if not silent:
                print("Face.NormalEdge - Error: The input face parameter is not a valid face. Retuning None.")
            return None
        if length <= tolerance:
            if not silent:
                print("Face.NormalEdge - Error: The input length parameter is less than or equal to the input tolerance. Retuning None.")
            return None
        iv = Face.InternalVertex(face)
        u, v = Face.VertexParameters(face, iv)
        vec = Face.Normal(face)
        ev = Topology.TranslateByDirectionDistance(iv, vec, length)
        return Edge.ByVertices([iv, ev], tolerance=tolerance, silent=silent)

    @staticmethod
    def NorthArrow(origin= None, radius: float = 0.5, sides: int = 16, direction: list = [0, 0, 1], northAngle: float = 0.0,
                   placement: str = "center", tolerance: float = 0.0001):
        """
        Creates a north arrow.

        Parameters
        ----------
        origin : topologic_core.Vertex, optional
            The location of the origin of the circle. Default is None which results in the circle being placed at (0, 0, 0).
        radius : float , optional
            The radius of the circle. Default is 1.
        sides : int , optional
            The number of sides of the circle. Default is 16.
        direction : list , optional
            The vector representing the up direction of the circle. Default is [0, 0, 1].
        northAngle : float , optional
            The angular offset in degrees from the positive Y axis direction. The angle is measured in a counter-clockwise fashion where 0 is positive Y, 90 is negative X, 180 is negative Y, and 270 is positive X.
        placement : str , optional
            The description of the placement of the origin of the circle. This can be "center", "lowerleft", "upperleft", "lowerright", or "upperright". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        topologic_core.Face
            The created circle.

        """
        from topologicpy.Topology import Topology
        from topologicpy.Vertex import Vertex
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.Origin()
        
        c = Face.Circle(origin=origin, radius=radius, sides=sides, direction=[0, 0, 1], placement="center", tolerance=tolerance)
        r = Face.Rectangle(origin=origin, width=radius*0.01,length=radius*1.2, placement="lowerleft")
        r = Topology.Translate(r, -0.005*radius,0,0)
        arrow = Topology.Difference(c, r, tolerance=tolerance)
        arrow = Topology.Rotate(arrow, origin=Vertex.Origin(), axis=[0, 0, 1], angle=northAngle)
        if placement.lower() == "lowerleft":
            arrow = Topology.Translate(arrow, radius, radius, 0)
        elif placement.lower() == "upperleft":
            arrow = Topology.Translate(arrow, radius, -radius, 0)
        elif placement.lower() == "lowerright":
            arrow = Topology.Translate(arrow, -radius, radius, 0)
        elif placement.lower() == "upperright":
            arrow = Topology.Translate(arrow, -radius, -radius, 0)
        arrow = Topology.Place(arrow, originA=Vertex.Origin(), originB=origin)
        arrow = Topology.Orient(arrow, origin=origin, dirA=[0,0,1], dirB=direction)
        return arrow
    
    @staticmethod
    def PlaneEquation(face, mantissa: int = 6) -> dict:
        """
        Returns the a, b, c, d coefficients of the plane equation of the input face. The input face is assumed to be planar.
        
        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        
        Returns
        -------
        dict
            The dictionary containing the coefficients of the plane equation. The keys of the dictionary are: ["a", "b", "c", "d"].

        """
        from topologicpy.Topology import Topology
        from topologicpy.Vertex import Vertex
        import random
        import time

        if not Topology.IsInstance(face, "Face"):
            print("Face.PlaneEquation - Error: The input face is not a valid topologic face. Returning None.")
            return None
        vertices = Topology.Vertices(face)
        if len(vertices) < 3:
            print("Face.PlaneEquation - Error: The input face has less than 3 vertices. Returning None.")
            return None
        return Vertex.PlaneEquation(vertices, mantissa=mantissa)
    
    @staticmethod
    def Planarize(face, origin= None,
                  tolerance: float = 0.0001):
        """
        Planarizes the input face such that its center of mass is located at the input origin and its normal is pointed in the input direction.

        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        origin : topologic_core.Vertex , optional
            The desired vertex to use as the origin of the plane to project the face unto. If set to None, the centroidof the input face is used. Default is None.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        
        Returns
        -------
        topologic_core.Face
            The planarized face.

        """

        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(face, "Face"):
            return None
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Topology.Centroid(face)
        eb = Face.ExternalBoundary(face)
        plan_eb = Wire.Planarize(eb, origin=origin)
        ib_list = Face.InternalBoundaries(face)
        plan_ib_list = []
        for ib in ib_list:
            plan_ib_list.append(Wire.Planarize(ib, origin=origin))
        plan_face = Face.ByWires(plan_eb, plan_ib_list)
        return plan_face

    @staticmethod
    def Project(faceA, faceB, direction : list = None,
                mantissa: int = 6, tolerance: float = 0.0001):
        """
        Creates a projection of the first input face unto the second input face.

        Parameters
        ----------
        faceA : topologic_core.Face
            The face to be projected.
        faceB : topologic_core.Face
            The face unto which the first input face will be projected.
        direction : list, optional
            The vector direction of the projection. If None, the reverse vector of the receiving face normal will be used. Default is None.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        topologic_core.Face
            The projected Face.

        """
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology

        if not faceA:
            return None
        if not Topology.IsInstance(faceA, "Face"):
            return None
        if not faceB:
            return None
        if not Topology.IsInstance(faceB, "Face"):
            return None

        eb = Face.ExternalBoundary(faceA)
        ib_list = Face.InternalBoundaries(faceA)
        p_eb = Wire.Project(wire=eb, face = faceB, direction=direction, mantissa=mantissa, tolerance=tolerance)
        p_ib_list = []
        for ib in ib_list:
            temp_ib = Wire.Project(wire=ib, face = faceB, direction=direction, mantissa=mantissa, tolerance=tolerance)
            if temp_ib:
                p_ib_list.append(temp_ib)
        return Face.ByWires(p_eb, p_ib_list, tolerance=tolerance)

    @staticmethod
    def RectangleByPlaneEquation(origin= None, width: float = 1.0, length: float = 1.0, placement: str = "center", equation: dict = None, tolerance: float = 0.0001):
        from topologicpy.Vertex import Vertex
        # Extract coefficients of the plane equation
        a = equation['a']
        b = equation['b']
        c = equation['c']
        d = equation['d']

        # Calculate the normal vector of the plane
        direction = np.array([a, b, c], dtype=float)
        direction /= np.linalg.norm(direction)
        direction = [x for x in direction]

        return Face.Rectangle(origin=origin, width=width, length=length, direction = direction, placement=placement, tolerance=tolerance)

    @staticmethod
    def Rectangle(origin= None, width: float = 1.0, length: float = 1.0, direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001, silent: bool = True):
        """
        Creates a rectangle.

        Parameters
        ----------
        origin : topologic_core.Vertex, optional
            The location of the origin of the rectangle. Default is None which results in the rectangle being placed at (0, 0, 0).
        width : float , optional
            The width of the rectangle. Default is 1.0.
        length : float , optional
            The length of the rectangle. Default is 1.0.
        direction : list , optional
            The vector representing the up direction of the rectangle. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the rectangle. This can be "center", "lowerleft", "upperleft", "lowerright", "upperright". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Face
            The created face.

        """
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology
        
        wire = Wire.Rectangle(origin=origin, width=width, length=length, direction=direction, placement=placement, tolerance=tolerance, silent=silent)
        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Face.Rectangle - Error: Could not create the base wire for the rectangle. Returning None.")
            return None
        return Face.ByWire(wire, tolerance=tolerance)
    
    @staticmethod
    def RemoveCollinearEdges(face, angTolerance: float = 0.1, tolerance: float = 0.0001, silent: bool = False):
        """
        Removes any collinear edges in the input face.

        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        angTolerance : float , optional
            The desired angular tolerance. Default is 0.1.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Face
            The created face without any collinear edges.

        """
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology
        import inspect
        
        if not Topology.IsInstance(face, "Face"):
            if not silent:
                print("Face.RemoveCollinearEdges - Error: The input face parameter is not a valid face. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
            return None
        eb = Wire.RemoveCollinearEdges(Face.Wire(face), angTolerance=angTolerance, tolerance=tolerance, silent=silent)
        ib = [Wire.RemoveCollinearEdges(w, angTolerance=angTolerance, tolerance=tolerance, silent=silent) for w in Face.InternalBoundaries(face)]
        return Face.ByWires(eb, ib, silent=silent)
    
    @staticmethod
    def RHS(origin= None, width: float = 1.0, length: float = 1.0, thickness: float = 0.25, outerFillet: float = 0.0, innerFillet: float = 0.0, sides: int = 16, direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a rectangluar hollow section (RHS).

        Parameters
        ----------
        origin : topologic_core.Vertex, optional
            The location of the origin of the RHS. Default is None which results in the RHS being placed at (0, 0, 0).
        width : float , optional
            The width of the RHS. Default is 1.0.
        length : float , optional
            The length of the RHS. Default is 1.0.
        thickness : float , optional
            The thickness of the RHS. Default is 0.25.
        outerFillet : float , optional
            The outer fillet multiplication factor based on the thickness (e.g. 1t). Default is 0.
        innerFillet : float , optional
            The inner fillet multiplication factor based on the thickness (e.g. 1.5t). Default is 0.
        sides : int , optional
            The desired number of sides of the fillets. Default is 16.
        direction : list , optional
            The vector representing the up direction of the RHS. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the RHS. This can be "center", "lowerleft", "upperleft", "lowerright", "upperright". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Face
            The created face.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology

        if 2*thickness >= width:
            if not silent:
                print("Face.RHS - Error: Twice the thickness value is larger than or equal to the width value. Returning None.")
            return None
        if 2*thickness >= width:
            if not silent:
                print("Face.RHS - Error: Twice the thickness value is larger than or equal to the length value. Returning None.")
            return None
        outer_dimension = min(width, length)
        fillet_dimension = 2*outerFillet*thickness
        if  fillet_dimension > outer_dimension:
            if not silent:
                print("Face.RHS = Error: The outer fillet radius input value is too large given the desired dimensions of the RHS. Returning None.")
            return None
        inner_dimension = min(width, length) - 2*thickness
        fillet_dimension = 2*innerFillet*thickness
        if fillet_dimension > inner_dimension:
            if not silent:
                print("Face.RHS = Error: The inner fillet radius input value is too large given the desired dimensions of the RHS. Returning None.")
            return None
        if origin == None:
            origin = Vertex.Origin()
        
        outer_wire = Wire.Rectangle(origin=Vertex.Origin(), width=width, length=length, direction=[0,0,1], placement="center", tolerance=tolerance, silent=silent)
        inner_wire = Wire.Rectangle(origin=Vertex.Origin(), width=width-thickness*2, length=length-thickness*2, direction=[0,0,1], placement="center", tolerance=tolerance, silent=silent)
        if outerFillet > 0:
           outer_wire = Wire.Fillet(outer_wire, radius=outerFillet*thickness, sides=sides, silent=silent)
        if innerFillet > 0:
           inner_wire = Wire.Fillet(inner_wire, radius=innerFillet*thickness, sides=sides, silent=silent) 
        return_face = Face.ByWires(outer_wire, [inner_wire], silent=silent)
        if not Topology.IsInstance(return_face, "face"):
            if not silent:
                print("Face.RHS - Error: Could not create the face for the RHS. Returning None.")
            return None
        
        xOffset = 0
        yOffset = 0
        zOffset = 0
        if placement.lower() == "lowerleft":
            xOffset = width*0.5
            yOffset = length*0.5
        elif placement.lower() == "upperleft":
            xOffset = width*0.5
            yOffset = -length*0.5
        elif placement.lower() == "lowerright":
            xOffset = -width*0.5
            yOffset = length*0.5
        elif placement.lower() == "upperright":
            xOffset = -width*0.5
            yOffset = -length*0.5
        return_face = Topology.Translate(return_face, x=xOffset, y=yOffset, z=zOffset)
        return_face = Topology.Place(return_face, originA=Vertex.Origin(), originB=origin)
        if direction != [0, 0, 1]:
            return_face = Topology.Orient(return_face, origin=origin, dirA=[0, 0, 1], dirB=direction)
        return return_face
    
    @staticmethod
    def Ring(origin= None, radius: float = 0.5, thickness: float = 0.25, sides: int = 16, direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a circular ring. This is an alias method for creating a circular hollow section (CHS).

        Parameters
        ----------
        origin : topologic_core.Vertex, optional
            The location of the origin of the ring. Default is None which results in the ring being placed at (0, 0, 0).
        radius : float , optional
            The outer radius of the ring. Default is 0.5.
        thickness : float , optional
            The thickness of the ring. Default is 0.25.
        direction : list , optional
            The vector representing the up direction of the ring. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the ring. This can be "center", "lowerleft", "upperleft", "lowerright", "upperright". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Face
            The created face.

        """
        
        if thickness >= radius:
            if not silent:
                print("Face.Ring - Error: The thickness value is larger than or equal to the outer radius value. Returning None.")
            return None
        return Face.CHS(origin=origin,
                        radius=radius,
                        thickness=thickness,
                        sides=sides,
                        direction=direction,
                        placement=placement,
                        tolerance=tolerance,
                        silent=silent)
    @staticmethod
    def SHS(origin= None, size: float = 1.0, thickness: float = 0.25, outerFillet: float = 0.0, innerFillet: float = 0.0, sides: int = 16, direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a square hollow section (SHS).

        Parameters
        ----------
        origin : topologic_core.Vertex, optional
            The location of the origin of the SHS. Default is None which results in the SHS being placed at (0, 0, 0).
        size : float , optional
            The outer size of the SHS. Default is 1.0.
        thickness : float , optional
            The thickness of the SHS. Default is 0.25.
        outerFillet : float , optional
            The outer fillet multiplication factor based on the thickness (e.g. 1t). Default is 0.
        innerFillet : float , optional
            The inner fillet multiplication factor based on the thickness (e.g. 1.5t). Default is 0.
        sides : int , optional
            The desired number of sides of the fillets. Default is 16.
        direction : list , optional
            The vector representing the up direction of the SHS. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the SHS. This can be "center", "lowerleft", "upperleft", "lowerright", "upperright". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Face
            The created face.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology
        
        if 2*thickness >= size:
            if not silent:
                print("Face.SHS - Error: Twice the thickness value is larger than or equal to the outer size value. Returning None.")
            return None
        fillet_dimension = 2*outerFillet*thickness
        if  fillet_dimension > size:
            if not silent:
                print("Face.RHS = Error: The outer fillet radius input value is too large given the desired dimensions of the RHS. Returning None.")
            return None
        inner_dimension = size - 2*thickness
        fillet_dimension = 2*innerFillet*thickness
        if fillet_dimension > inner_dimension:
            if not silent:
                print("Face.RHS = Error: The inner fillet radius input value is too large given the desired dimensions of the RHS. Returning None.")
            return None
        if origin == None:
            origin = Vertex.Origin()
        
        return Face.RHS(origin = origin, width = size, length = size, thickness = thickness, outerFillet = outerFillet, innerFillet = innerFillet, sides = sides, direction = direction, placement = placement, tolerance = tolerance, silent = silent)
    
    @staticmethod
    def Simplify(face, method='douglas-peucker', tolerance=0.0001, silent=False):
        """
        Simplifies the input wire edges based on the selected algorithm: Douglas-Peucker or Visvalingam–Whyatt.
        
        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        method : str, optional
            The simplification method to use: 'douglas-peucker' or 'visvalingam-whyatt' or 'reumann-witkam'.
            The default is 'douglas-peucker'.
        tolerance : float , optional
            The desired tolerance.
            If using the douglas-peucker method, edge lengths shorter than this amount will be removed.
            If using the visvalingam-whyatt method, triangulare areas less than is amount will be removed.
            If using the Reumann-Witkam method, the tolerance specifies the maximum perpendicular distance allowed
            between any point and the current line segment; points falling within this distance are discarded.
            The default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
            
        Returns
        -------
        topologic_core.Face
            The simplified face.
        
        """
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(face, "Face"):
            if not silent:
                print("Face.Simplify - Error: The input face parameter is not a valid face. Returning None.")
            return None
        
        eb = Face.ExternalBoundary(face)
        eb = Wire.Simplify(eb, method=method, tolerance=tolerance, silent=silent)
        ibList = Face.InternalBoundaries(face)
        ibList = [Wire.Simplify(ib, method=method, tolerance=tolerance, silent=silent) for ib in ibList]
        return_face = Face.ByWires(eb, ibList, tolerance=tolerance, silent=silent)
        if not Topology.IsInstance(return_face, "Face"):
            if not silent:
                print("Face.Simplify - Error: Could not simplify the face. Returning the original input face.")
            return face
        return return_face
    
    @staticmethod
    def Skeleton(face, boundary: bool = True, tolerance: float = 0.001):
        """
            Creates a straight skeleton. This method is contributed by 高熙鹏 xipeng gao <gaoxipeng1998@gmail.com>
            This algorithm depends on the polyskel code which is included in the library. Polyskel code is found at: https://github.com/Botffy/polyskel

        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        boundary : bool , optional
            If set to True the original boundary is returned as part of the roof. Otherwise it is not. Default is True.
        tolerance : float , optional
            The desired tolerance. Default is 0.001. (This is set to a larger number than the usual 0.0001 as it was found to work better)

        Returns
        -------
        topologic_core.Wire
            The created straight skeleton.

        """
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(face, "Face"):
            print("Face.Skeleton - Error: The input face is not a valid topologic face. Returning None.")
            return None
        return Wire.Skeleton(face, boundary=boundary, tolerance=tolerance)
    
    @staticmethod
    def Square(origin= None, size: float = 1.0, direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001):
        """
        Creates a square.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the square. Default is None which results in the square being placed at (0, 0, 0).
        size : float , optional
            The size of the square. Default is 1.0.
        direction : list , optional
            The vector representing the up direction of the square. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the square. This can be "center", "lowerleft", "upperleft", "lowerright", or "upperright". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        topologic_core.Face
            The created square.

        """
        return Face.Rectangle(origin=origin, width=size, length=size, direction=direction, placement=placement, tolerance=tolerance)
    

    @staticmethod
    def Squircle(origin = None, radius: float = 0.5, sides: int = 121, a: float = 2.0, b: float = 2.0, direction: list = [0, 0, 1], placement: str = "center", angTolerance: float = 0.1, tolerance: float = 0.0001):
        """
        Creates a Squircle which is a hybrid between a circle and a square. See https://en.wikipedia.org/wiki/Squircle

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the squircle. Default is None which results in the squircle being placed at (0, 0, 0).
        radius : float , optional
            The desired radius of the squircle. Default is 0.5.
        sides : int , optional
            The desired number of sides of the squircle. Default is 121.
        a : float , optional
            The "a" factor affects the x position of the points to interpolate between a circle and a square.
            A value of 1 will create a circle. Higher values will create a more square-like shape. Default is 2.0.
        b : float , optional
            The "b" factor affects the y position of the points to interpolate between a circle and a square.
            A value of 1 will create a circle. Higher values will create a more square-like shape. Default is 2.0.
        direction : list , optional
            The vector representing the up direction of the circle. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the circle. This can be "center", "lowerleft", "upperleft", "lowerright", or "upperright". It is case insensitive. Default is "center".
        angTolerance : float , optional
            The desired angular tolerance. Default is 0.1.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        topologic_core.Face
            The created squircle.
        """
        from topologicpy.Wire import Wire
        wire = Wire.Squircle(origin = origin, radius= radius, sides = sides, a = a, b = b, direction = direction, placement = placement, angTolerance = angTolerance, tolerance = tolerance)
        return Face.ByWire(wire)
    
    @staticmethod
    def Star(origin= None, radiusA: float = 0.5, radiusB: float = 0.2, rays: int = 8, direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001):
        """
        Creates a star.

        Parameters
        ----------
        origin : topologic_core.Vertex, optional
            The location of the origin of the star. Default is None which results in the star being placed at (0, 0, 0).
        radiusA : float , optional
            The outer radius of the star. Default is 1.0.
        radiusB : float , optional
            The outer radius of the star. Default is 0.4.
        rays : int , optional
            The number of star rays. Default is 5.
        direction : list , optional
            The vector representing the up direction of the star. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the star. This can be "center", "lowerleft", "upperleft", "lowerright", or "upperright". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        topologic_core.Face
            The created face.

        """
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology

        wire = Wire.Star(origin=origin, radiusA=radiusA, radiusB=radiusB, rays=rays, direction=direction, placement=placement, tolerance=tolerance)
        if not Topology.IsInstance(wire, "Wire"):
            print("Face.Rectangle - Error: Could not create the base wire for the star. Returning None.")
            return None
        return Face.ByWire(wire, tolerance=tolerance)


    @staticmethod
    def ThirdVertex(face, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns a third vertex on the input face to enable rotation matrix creation.

        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Face
            The created face.

        """
        from topologicpy.Vector import Vector
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(face, "Face"):
            if not silent:
                print("Face.ThirdVertex - Error: The input face parameter is not a valid face. Returning None.")
            return None
        # Retrieve all vertices of the face
        vertices = Face.Vertices(face)
        centroid = Topology.Centroid(face)
        normal = Face.Normal(face)
        for vertex in vertices:
            # Skip the centroid itself
            if Vertex.Distance(centroid, vertex) <= tolerance:
                continue
            
            # Vector from the centroid to the current vertex
            vector_to_vertex = Vector.ByVertices(centroid, vertex)
            vector_to_vertex_normalized = Vector.Normalize(vector_to_vertex)


            # Check if the vector_to_vertex is collinear with the normal direction
            if Vector.IsCollinear(vector_to_vertex_normalized, normal, tolerance):
                continue

            # If not collinear, return this vertex
            return vertex

        # No valid third vertex found
        if not silent:
                print("Face.ThirdVertex - Warning: No valid third vertex could be found. Returning None.")
        return None

    @staticmethod
    def Trapezoid(origin= None, widthA: float = 1.0, widthB: float = 0.75, offsetA: float = 0.0, offsetB: float = 0.0, length: float = 1.0, direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001):
        """
        Creates a trapezoid.

        Parameters
        ----------
        origin : topologic_core.Vertex, optional
            The location of the origin of the trapezoid. Default is None which results in the trapezoid being placed at (0, 0, 0).
        widthA : float , optional
            The width of the bottom edge of the trapezoid. Default is 1.0.
        widthB : float , optional
            The width of the top edge of the trapezoid. Default is 0.75.
        offsetA : float , optional
            The offset of the bottom edge of the trapezoid. Default is 0.0.
        offsetB : float , optional
            The offset of the top edge of the trapezoid. Default is 0.0.
        length : float , optional
            The length of the trapezoid. Default is 1.0.
        direction : list , optional
            The vector representing the up direction of the trapezoid. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the trapezoid. This can be "center", or "lowerleft". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        topologic_core.Face
            The created trapezoid.

        """
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology

        wire = Wire.Trapezoid(origin=origin, widthA=widthA, widthB=widthB, offsetA=offsetA, offsetB=offsetB, length=length, direction=direction, placement=placement, tolerance=tolerance)
        if not Topology.IsInstance(wire, "Wire"):
            print("Face.Rectangle - Error: Could not create the base wire for the trapezoid. Returning None.")
            return None
        return Face.ByWire(wire, tolerance=tolerance)

    @staticmethod
    def Triangulate(face, mode: int = 0, meshSize: float = None, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False) -> list:
        """
        Triangulates the input face and returns a list of faces.

        Parameters
        ----------
        face : topologic_core.Face
            The input face.
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
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The list of triangles of the input face.

        """
        from topologicpy.Vector import Vector
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology

        # This function was contributed by Yidan Xue.
        def generate_gmsh(face, mode="mesh", meshSize = None, tolerance = 0.0001):
            """
            Creates a gmsh of triangular meshes from the input face.

            Parameters
            ----------
            face : topologic_core.Face
                The input face.
            meshSize : float , optional
                The desired mesh size.
            tolerance : float , optional
                The desired tolerance. Default is 0.0001.
            
            Returns
            -------
            topologic_core.Shell
                The shell of triangular meshes.

            """
            import os
            import warnings
            try:
                import numpy as np
            except:
                print("Face.Triangulate - Warning: Installing required numpy library.")
                try:
                    os.system("pip install numpy")
                except:
                    os.system("pip install numpy --user")
                try:
                    import numpy as np
                    print("Face.Triangulate - Warning: numpy library installed correctly.")
                except:
                    warnings.warn("Face.Triangulate - Error: Could not import numpy. Please try to install numpy manually. Returning None.")
                    return None
            try:
                import gmsh
            except:
                print("Face.Triangulate - Warning: Installing required gmsh library.")
                try:
                    os.system("pip install gmsh")
                except:
                    os.system("pip install gmsh --user")
                try:
                    import gmsh
                    print("Face.Triangulate - Warning: gmsh library installed correctly.")
                except:
                    warnings.warn("Face.Triangulate - Error: Could not import gmsh. Please try to install gmsh manually. Returning None.")
                    return None
            
            from topologicpy.Vertex import Vertex
            from topologicpy.Wire import Wire
            from topologicpy.Topology import Topology

            if not Topology.IsInstance(face, "Face"):
                if not silent:
                    print("Face.Triangulate - Error: The input face parameter is not a valid face. Returning None.")
                return None
            if not meshSize:
                bounding_face = Face.BoundingRectangle(face)
                bounding_face_vertices = Topology.Vertices(bounding_face)
                bounding_face_vertices_x = [Vertex.X(i, mantissa=mantissa) for i in bounding_face_vertices]
                bounding_face_vertices_y = [Vertex.Y(i, mantissa=mantissa) for i in bounding_face_vertices]
                width = max(bounding_face_vertices_x)-min(bounding_face_vertices_x)
                length = max(bounding_face_vertices_y)-min(bounding_face_vertices_y)
                meshSize = max([width,length])//10
            
            gmsh.initialize()
            face_external_boundary = Face.ExternalBoundary(face)
            external_vertices = Topology.Vertices(face_external_boundary)
            external_vertex_number = len(external_vertices)
            for i in range(external_vertex_number):
                gmsh.model.geo.addPoint(Vertex.X(external_vertices[i], mantissa=mantissa), Vertex.Y(external_vertices[i], mantissa=mantissa), Vertex.Z(external_vertices[i], mantissa=mantissa), meshSize, i+1)
            for i in range(external_vertex_number):
                if i < external_vertex_number-1:
                    gmsh.model.geo.addLine(i+1, i+2, i+1)
                else:
                    gmsh.model.geo.addLine(i+1, 1, i+1)
            gmsh.model.geo.addCurveLoop([i+1 for i in range(external_vertex_number)], 1)
            current_vertex_number = external_vertex_number
            current_edge_number = external_vertex_number
            current_wire_number = 1

            face_internal_boundaries = Face.InternalBoundaries(face)
            if face_internal_boundaries:
                internal_face_number = len(face_internal_boundaries)
                for i in range(internal_face_number):
                    face_internal_boundary = face_internal_boundaries[i]
                    internal_vertices = Topology.Vertices(face_internal_boundary)
                    internal_vertex_number = len(internal_vertices)
                    for j in range(internal_vertex_number):
                        gmsh.model.geo.addPoint(Vertex.X(internal_vertices[j]), Vertex.Y(internal_vertices[j], mantissa=mantissa), Vertex.Z(internal_vertices[j], mantissa=mantissa), meshSize, current_vertex_number+j+1)
                    for j in range(internal_vertex_number):
                        if j < internal_vertex_number-1:
                            gmsh.model.geo.addLine(current_vertex_number+j+1, current_vertex_number+j+2, current_edge_number+j+1)
                        else:
                            gmsh.model.geo.addLine(current_vertex_number+j+1, current_vertex_number+1, current_edge_number+j+1)
                    gmsh.model.geo.addCurveLoop([current_edge_number+i+1 for i in range(internal_vertex_number)], current_wire_number+1)
                    current_vertex_number = current_vertex_number+internal_vertex_number
                    current_edge_number = current_edge_number+internal_vertex_number
                    current_wire_number = current_wire_number+1

            gmsh.model.geo.addPlaneSurface([i+1 for i in range(current_wire_number)])
            gmsh.model.geo.synchronize()
            if mode not in [1,3,5,6,7,8,9]:
                mode = 6
            gmsh.option.setNumber("Mesh.Algorithm", mode)
            gmsh.model.mesh.generate(2)         # For a 2D mesh
            nodeTags, nodeCoords, nodeParams = gmsh.model.mesh.getNodes(-1, -1)
            elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(-1, -1)
            gmsh.finalize()
            
            vertex_number = len(nodeTags)
            vertices = []
            for i in range(vertex_number):
                vertices.append(Vertex.ByCoordinates(nodeCoords[3*i],nodeCoords[3*i+1],nodeCoords[3*i+2]))

            faces = []
            for n in range(len(elemTypes)):
                vn = elemTypes[n]+1
                et = elemTags[n]
                ent = elemNodeTags[n]
                if vn==3:
                    for i in range(len(et)):
                        face_vertices = []
                        for j in range(vn):
                            face_vertices.append(vertices[np.where(nodeTags==ent[i*vn+j])[0][0]])
                        faces.append(Face.ByVertices(face_vertices))
            return faces

        if not Topology.IsInstance(face, "Face"):
            if not silent:
                print("Face.Triangulate - Error: The input face parameter is not a valid face. Returning None.")
            return None
        vertices = Topology.Vertices(face)
        if len(vertices) == 3: # Already a triangle
            return [face]
        origin = Topology.Centroid(face)
        normal = Face.Normal(face, mantissa=mantissa)
        flatFace = Topology.Flatten(face, origin=origin, direction=normal)

        if mode == 0:
            shell_faces = []
            for i in range(0,5,1):
                try:
                    _ = Core.FaceUtility.Triangulate(flatFace, float(i)*0.1, shell_faces)
                    break
                except:
                    continue
        else:
            shell_faces = generate_gmsh(flatFace, mode = mode, meshSize = meshSize, tolerance = tolerance)
            
        if len(shell_faces) < 1:
            return []
        finalFaces = []
        for f in shell_faces:
            f = Topology.Unflatten(f, origin=origin, direction=normal)
            if Face.Angle(face, f, mantissa=mantissa) > 90:
                wire = Face.ExternalBoundary(f)
                wire = Wire.Invert(wire)
                f = Face.ByWire(wire)
                if Topology.IsInstance(f, "Face"):
                    finalFaces.append(f)
            else:
                if Topology.IsInstance(f, "face"):
                    finalFaces.append(f)
        face_normal = Face.Normal(face)
        return_faces = []
        for ff in finalFaces:
            normal = Face.Normal(ff)
            if abs(Vector.Angle(normal, face_normal)) > 2:
                return_faces.append(Face.Invert(ff))
            else:
                return_faces.append(ff)
        return return_faces

    @staticmethod
    def TrimByWire(face, wire, reverse: bool = False):
        """
        Trims the input face by the input wire.

        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        wire : topologic_core.Wire
            The input wire.
        reverse : bool , optional
            If set to True, the effect of the trim will be reversed. Default is False.

        Returns
        -------
        topologic_core.Face
            The resulting trimmed face.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(face, "Face"):
            return None
        if not Topology.IsInstance(wire, "Wire"):
            return face
        trimmed_face = Core.FaceUtility.TrimByWire(face, wire, False)
        if reverse:
            trimmed_face = Topology.Difference(face, trimmed_face)
        return trimmed_face
    
    @staticmethod
    def TShape(origin=None,
            width=1,
            length=1,
            a=0.25,
            b=0.25,
            flipHorizontal = False,
            flipVertical = False,
            direction=[0,0,1],
            placement="center",
            tolerance=0.0001,
            silent=False):
        """
        Creates a T-shape.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the T-shape. Default is None which results in the T-shape being placed at (0, 0, 0).
        width : float , optional
            The overall width of the T-shape. Default is 1.0.
        length : float , optional
            The overall length of the T-shape. Default is 1.0.
        a : float , optional
            The hortizontal thickness of the vertical arm of the T-shape. Default is 0.25.
        b : float , optional
            The vertical thickness of the horizontal arm of the T-shape. Default is 0.25.
        direction : list , optional
            The vector representing the up direction of the T-shape. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the T-shape. This can be "center", "lowerleft", "upperleft", "lowerright", "upperright". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Face
            The created T-shape.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology

        if not isinstance(width, int) and not isinstance(width, float):
            if not silent:
                print("Face.TShape - Error: The width input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(length, int) and not isinstance(length, float):
            if not silent:
                print("Face.TShape - Error: The length input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(a, int) and not isinstance(a, float):
            if not silent:
                print("Face.TShape - Error: The a input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(b, int) and not isinstance(b, float):
            if not silent:
                print("Face.LShape - Error: The b input parameter is not a valid number. Returning None.")
            return None
        if width <= tolerance:
            if not silent:
                print("Face.TShape - Error: The width input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if length <= tolerance:
            if not silent:
                print("Face.TShape - Error: The length input parameter must be a positive number  greater than the tolerance input parameter. Returning None.")
            return None
        if a <= tolerance:
            if not silent:
                print("Face.TShape - Error: The a input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if b <= tolerance:
            if not silent:
                print("Face.TShape - Error: The b input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if a >= (width - tolerance):
            if not silent:
                print("Face.TShape - Error: The a input parameter must be less than the width input parameter. Returning None.")
            return None
        if b >= (length - tolerance):
            if not silent:
                print("Face.TShape - Error: The b input parameter must be less than the length input parameter. Returning None.")
            return None
        if origin == None:
            origin = Vertex.Origin()
        if not Topology.IsInstance(origin, "vertex"):
            if not silent:
                print("Face.TShape - Error: The origin input parameter is not a valid topologic vertex. Returning None.")
            return None
        if not isinstance(direction, list):
            if not silent:
                print("Face.TShape - Error: The direction input parameter is not a valid list. Returning None.")
            return None
        if not len(direction) == 3:
            if not silent:
                print("Face.TShape - Error: The direction input parameter is not a valid vector. Returning None.")
            return None
        t_shape_wire = Wire.TShape(origin=origin,
                                   width=width,
                                   length=length,
                                   a=a,
                                   b=b,
                                   flipHorizontal=flipHorizontal,
                                   flipVertical=flipVertical,
                                   direction=direction,
                                   placement=placement,
                                   tolerance=tolerance,
                                   silent=silent)
        return Face.ByWire(t_shape_wire, tolerance=tolerance, silent=silent)

    @staticmethod
    def VertexByParameters(face, u: float = 0.5, v: float = 0.5):
        """
        Creates a vertex at the *u* and *v* parameters of the input face.

        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        u : float , optional
            The *u* parameter of the input face. Default is 0.5.
        v : float , optional
            The *v* parameter of the input face. Default is 0.5.

        Returns
        -------
        vertex : topologic vertex
            The created vertex.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(face, "Face"):
            return None
        return Core.FaceUtility.VertexAtParameters(face, u, v)
    @staticmethod
    def VertexParameters(face, vertex, outputType: str = "uv", mantissa: int = 6) -> list:
        """
        Returns the *u* and *v* parameters of the input face at the location of the input vertex.

        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        vertex : topologic_core.Vertex
            The input vertex.
        outputType : string , optional
            The string defining the desired output. This can be any subset or permutation of "uv". It is case insensitive. Default is "uv".
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.

        Returns
        -------
        list
            The list of *u* and/or *v* as specified by the outputType input.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(face, "Face"):
            return None
        if not Topology.IsInstance(vertex, "Vertex"):
            return None
        params = Core.FaceUtility.ParametersAtVertex(face, vertex)
        u = round(params[0], mantissa)
        v = round(params[1], mantissa)
        outputType = list(outputType.lower())
        returnResult = []
        for param in outputType:
            if param == "u":
                returnResult.append(u)
            elif param == "v":
                returnResult.append(v)
        return returnResult

    @staticmethod
    def Vertices(face) -> list:
        """
        Returns the vertices of the input face.

        Parameters
        ----------
        face : topologic_core.Face
            The input face.

        Returns
        -------
        list
            The list of vertices.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(face, "Face"):
            return None
        vertices = []
        # _ = face.Vertices(None, vertices) # H to Core
        try:
            _ = Core.InstanceCall(face, "Vertices", None, vertices)
        except Exception:
            vertices = None
        return vertices
    
    @staticmethod
    def Wire(face):
        """
        Returns the external boundary (closed wire) of the input face.

        Parameters
        ----------
        face : topologic_core.Face
            The input face.

        Returns
        -------
        topologic_core.Wire
            The external boundary of the input face.

        """
        return Face.ExternalBoundary(face)
    
    @staticmethod
    def Wires(face) -> list:
        """
        Returns the wires of the input face.

        Parameters
        ----------
        face : topologic_core.Face
            The input face.

        Returns
        -------
        list
            The list of wires.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(face, "Face"):
            return None
        wires = []
        # _ = face.Wires(None, wires) # H to Core
        try:
            _ = Core.InstanceCall(face, "Wires", None, wires)
        except:
            wires = None
        return wires
