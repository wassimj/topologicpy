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
            The desired length of the mantissa. The default is 6.

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
            The desired length of the mantissa. The default is 6.

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
            area = round(topologic.FaceUtility.Area(face), mantissa) # Hook to Core
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
            If set to an integer from 1 (low optimization) to 10 (high optimization), the method will attempt to optimize the bounding rectangle so that it reduces its surface area. The default is 0 which will result in an axis-aligned bounding rectangle. The default is 0.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
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
    def ByEdges(edges: list, tolerance : float = 0.0001):
        """
        Creates a face from the input list of edges.

        Parameters
        ----------
        edges : list
            The input list of edges.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        face : topologic_core.Face
            The created face.

        """
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology

        if not isinstance(edges, list):
            print("Face.ByEdges - Error: The input edges parameter is not a valid list. Returning None.")
            return None
        edges = [e for e in edges if Topology.IsInstance(e, "Edge")]
        if len(edges) < 1:
            print("Face.ByEdges - Error: The input edges parameter does not contain any valid edges. Returning None.")
            return None
        wire = Wire.ByEdges(edges, tolerance=tolerance)
        if not Topology.IsInstance(wire, "Wire"):
            print("Face.ByEdges - Error: Could not create the required wire. Returning None.")
            return None
        face = Face.ByWire(wire, tolerance=tolerance)
        if not Topology.IsInstance(face, "Face"):
            print("Face.ByEdges - Warning: Could not create face from base wire. Returning None.")
            return None
        return face

    @staticmethod
    def ByEdgesCluster(cluster, tolerance: float = 0.0001):
        """
        Creates a face from the input cluster of edges.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input cluster of edges.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        face : topologic_core.Face
            The created face.

        """
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Cluster"):
            print("Face.ByEdgesCluster - Warning: The input cluster parameter is not a valid topologic cluster. Returning None.")
            return None
        edges = Cluster.Edges(cluster)
        if len(edges) < 1:
            print("Face.ByEdgesCluster - Warning: The input cluster parameter does not contain any valid edges. Returning None.")
            return None
        face = Face.ByEdges(edges, tolerance=tolerance)
        if not Topology.IsInstance(face, "Face"):
            print("Face.ByEdgesCluster - Warning: Could not create face from edges. Returning None.")
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
            The desired offset distance. The default is 1.0.
        offsetKey : str , optional
            The edge dictionary key under which to find the offset value. If a value cannot be found, the offset input parameter value is used instead. The default is "offset".
        stepOffsetA : float , optional
            The amount to offset along the previous edge when transitioning between parallel edges with different offsets. The default is 0.
        stepOffsetB : float , optional
            The amount to offset along the next edge when transitioning between parallel edges with different offsets. The default is 0.
        stepOffsetKeyA : str , optional
            The vertex dictionary key under which to find the step offset A value. If a value cannot be found, the stepOffsetA input parameter value is used instead. The default is "stepOffsetA".
        stepOffsetKeyB : str , optional
            The vertex dictionary key under which to find the step offset B value. If a value cannot be found, the stepOffsetB input parameter value is used instead. The default is "stepOffsetB".
        bisectors : bool , optional
            If set to True, The bisectors (seams) edges will be included in the returned wire. This will result in the returned shape to be a shell rather than a face. The default is False.
        reverse : bool , optional
            If set to True, the direction of offsets is reversed. Otherwise, it is not. The default is False.
        transferDictionaries : bool , optional
            If set to True, the dictionaries of the original wire, its edges, and its vertices are transfered to the new wire. Otherwise, they are not. The default is False.
        epsilon : float , optional
            The desired epsilon (another form of tolerance for shortest edge to remove). The default is 0.01. (This is set to a larger number as it was found to work better)
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.
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
        
        if abs(Face.Normal(face)[2] + 1) < tolerance:
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
            all_edges = Topology.Edges(offset_external_boundary)+[Topology.Edges(ib) for ib in offset_internal_boundaries]
            all_edges += Topology.Edges(face)
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
            The edge dictionary key under which to store the offset value. The default is "offset".
        minOffsetKey : str , optional
            The edge dictionary key under which to find the desired minimum edge offset value. If a value cannot be found, the defaultMinOffset input parameter value is used instead. The default is "minOffset".
        maxOffsetKey : str , optional
            The edge dictionary key under which to find the desired maximum edge offset value. If a value cannot be found, the defaultMaxOffset input parameter value is used instead. The default is "maxOffset".
        defaultMinOffset : float , optional
            The desired minimum edge offset distance. The default is 0.
        defaultMaxOffset : float , optional
            The desired maximum edge offset distance. The default is 1.
        maxIterations: int , optional
            The desired maximum number of iterations to attempt to converge on a solution. The default is 1.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.
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
                if loss < tolerance:
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
            The desired angular tolerance. The default is 0.1.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

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
        ext_boundary = Wire.RemoveCollinearEdges(Shell.ExternalBoundary(shell))
        #ext_boundary = Shell.ExternalBoundary(shell)
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
        ext_boundary = Topology.RemoveCollinearEdges(ext_boundary, angTolerance)
        if not Topology.IsInstance(ext_boundary, "Topology"):
            print("Face.ByShell - Error: Could not derive the external boundary of the input shell parameter. Returning None.")
            return None

        if Topology.IsInstance(ext_boundary, "Wire"):
            if not Topology.IsPlanar(ext_boundary, tolerance=tolerance):
                ext_boundary = Wire.Planarize(ext_boundary, origin=origin, tolerance=tolerance)
            ext_boundary = Topology.RemoveCollinearEdges(ext_boundary, angTolerance)
            try:
                face = Face.ByWire(ext_boundary)
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
                int_wires.append(Topology.RemoveCollinearEdges(temp_wires[0], angTolerance))
                #int_wires.append(temp_wires[0])

            temp_wires = Topology.Wires(ext_boundary)
            ext_wire = Topology.RemoveCollinearEdges(temp_wires[0], angTolerance)
            #ext_wire = temp_wires[0]
            face = Face.ByWires(ext_wire, int_wires)
            face = Topology.Unflatten(face, origin=origin, direction=normal)
            return face
        else:
            return None
    
    @staticmethod
    def ByThickenedWire(wire, offsetA: float = 1.0, offsetB: float = 0.0, tolerance: float = 0.0001):
        """
        Creates a face by thickening the input wire. This method assumes the wire is manifold and planar.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire to be thickened.
        offsetA : float , optional
            The desired offset to the exterior of the wire. The default is 1.0.
        offsetB : float , optional
            The desired offset to the interior of the wire. The default is 0.0.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

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
        temp_w = Wire.ByVertices(three_vertices, close=True)
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
            return_face = Face.ByWire(Wire.ByVertices(vertices))
        else:
            return_face = Face.ByWires(outside_wire, [inside_wire])
        return_face = Topology.Unflatten(return_face, origin=origin, direction=normal)
        return return_face
    
    @staticmethod
    def ByVertices(vertices: list, tolerance: float = 0.0001):
        
        """
        Creates a face from the input list of vertices.

        Parameters
        ----------
        vertices : list
            The input list of vertices.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Face
            The created face.

        """
        from topologicpy.Topology import Topology
        from topologicpy.Wire import Wire

        if not isinstance(vertices, list):
            return None
        vertexList = [x for x in vertices if Topology.IsInstance(x, "Vertex")]
        if len(vertexList) < 3:
            return None
        
        w = Wire.ByVertices(vertexList, tolerance=tolerance)
        f = Face.ByWire(w, tolerance=tolerance)
        return f

    @staticmethod
    def ByVerticesCluster(cluster, tolerance: float = 0.0001):
        """
        Creates a face from the input cluster of vertices.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input cluster of vertices.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Face
            The created face.

        """
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Cluster"):
            return None
        vertices = Topology.Vertices(cluster)
        return Face.ByVertices(vertices, tolerance=tolerance)

    @staticmethod
    def ByWire(wire, tolerance: float = 0.0001, silent=False):
        """
        Creates a face from the input closed wire.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        topologic_core.Face or list
            The created face. If the wire is non-planar, the method will attempt to triangulate the wire and return a list of faces.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Shell import Shell
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        import inspect

        def triangulateWire(wire):
            wire = Topology.RemoveCollinearEdges(wire)
            vertices = Topology.Vertices(wire)
            shell = Shell.Delaunay(vertices)
            if Topology.IsInstance(shell, "Topology"):
                return Topology.Faces(shell)
            else:
                return []
        if not Topology.IsInstance(wire, "Wire"):
            if not silent:
                print("Face.ByWire - Error: The input wire parameter is not a valid topologic wire. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
            return None
        if not Wire.IsClosed(wire):
            if not silent:
                print("Face.ByWire - Error: The input wire parameter is not a closed topologic wire. Returning None.")
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                print('caller name:', calframe[1][3])
            return None
        
        edges = Wire.Edges(wire)
        wire = Topology.SelfMerge(Cluster.ByTopologies(edges), tolerance=tolerance)
        vertices = Topology.Vertices(wire)
        fList = []
        if Topology.IsInstance(wire, "Wire"):
            try:
                fList = topologic.Face.ByExternalBoundary(wire) # Hook to Core
            except:
                if not silent:
                    print("Face.ByWire - Warning: Could not create face by external boundary. Trying other methods.")
                if len(vertices) > 3:
                    fList = triangulateWire(wire)
                else:
                    fList = []
        
        if not isinstance(fList, list):
            fList = [fList]

        returnList = []
        for f in fList:
            if Face.Area(f) < 0:
                wire = Face.ExternalBoundary(f)
                wire = Wire.Invert(wire)
                try:
                    f = topologic.Face.ByExternalBoundary(wire)  # Hook to Core
                    returnList.append(f)
                except:
                    pass
            else:
                returnList.append(f)
        if len(returnList) == 0:
            if not silent:
                print("Face.ByWire - Error: Could not build a face from the input wire parameter. Returning None.")
            return None
        elif len(returnList) == 1:
            return returnList[0]
        else:
            if not silent:
                print("Face.ByWire - Warning: Could not build a single face from the input wire parameter. Returning a list of faces.")
            return returnList

    @staticmethod
    def ByWires(externalBoundary, internalBoundaries: list = [], tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a face from the input external boundary (closed wire) and the input list of internal boundaries (closed wires).

        Parameters
        ----------
        externalBoundary : topologic_core.Wire
            The input external boundary.
        internalBoundaries : list , optional
            The input list of internal boundaries (closed wires). The default is an empty list.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        topologic_core.Face
            The created face.

        """
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(externalBoundary, "Wire"):
            if not silent:
                print("Face.ByWires - Error: The input externalBoundary parameter is not a valid topologic wire. Returning None.")
            return None
        if not Wire.IsClosed(externalBoundary):
            if not silent:
                print("Face.ByWires - Error: The input externalBoundary parameter is not a closed topologic wire. Returning None.")
            return None
        ibList = [x for x in internalBoundaries if Topology.IsInstance(x, "Wire") and Wire.IsClosed(x)]
        face = None
        try:
            face = topologic.Face.ByExternalInternalBoundaries(externalBoundary, ibList, tolerance) # Hook to Core
        except:
            if not silent:
                print("Face.ByWires - Error: The operation failed. Returning None.")
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
            The input cluster of internal boundaries (closed wires). The default is None.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.
        
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
    def NorthArrow(origin= None, radius: float = 0.5, sides: int = 16, direction: list = [0, 0, 1], northAngle: float = 0.0,
                   placement: str = "center", tolerance: float = 0.0001):
        """
        Creates a north arrow.

        Parameters
        ----------
        origin : topologic_core.Vertex, optional
            The location of the origin of the circle. The default is None which results in the circle being placed at (0, 0, 0).
        radius : float , optional
            The radius of the circle. The default is 1.
        sides : int , optional
            The number of sides of the circle. The default is 16.
        direction : list , optional
            The vector representing the up direction of the circle. The default is [0, 0, 1].
        northAngle : float , optional
            The angular offset in degrees from the positive Y axis direction. The angle is measured in a counter-clockwise fashion where 0 is positive Y, 90 is negative X, 180 is negative Y, and 270 is positive X.
        placement : str , optional
            The description of the placement of the origin of the circle. This can be "center", "lowerleft", "upperleft", "lowerright", or "upperright". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

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
    def Circle(origin= None, radius: float = 0.5, sides: int = 16, fromAngle: float = 0.0, toAngle: float = 360.0, direction: list = [0, 0, 1],
                   placement: str = "center", tolerance: float = 0.0001):
        """
        Creates a circle.

        Parameters
        ----------
        origin : topologic_core.Vertex, optional
            The location of the origin of the circle. The default is None which results in the circle being placed at (0, 0, 0).
        radius : float , optional
            The radius of the circle. The default is 1.
        sides : int , optional
            The number of sides of the circle. The default is 16.
        fromAngle : float , optional
            The angle in degrees from which to start creating the arc of the circle. The default is 0.
        toAngle : float , optional
            The angle in degrees at which to end creating the arc of the circle. The default is 360.
        direction : list , optional
            The vector representing the up direction of the circle. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the circle. This can be "center", "lowerleft", "upperleft", "lowerright", or "upperright". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

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
            The desired length of the mantissa. The default is 6.

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
            The second vector representing the north direction. The default is the positive YAxis ([0,1,0]).
        mantissa : int, optional
            The length of the desired mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

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
        _ = face.Edges(None, edges) # Hook to Core
        return edges

    @staticmethod
    def Einstein(origin= None, radius: float = 0.5, direction: list = [0, 0, 1],
                 placement: str = "center", tolerance: float = 0.0001):
        """
        Creates an aperiodic monotile, also called an 'einstein' tile (meaning one tile in German, not the name of the famous physicist). See https://arxiv.org/abs/2303.10798

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the tile. The default is None which results in the tiles first vertex being placed at (0, 0, 0).
        radius : float , optional
            The radius of the hexagon determining the size of the tile. The default is 0.5.
        direction : list , optional
            The vector representing the up direction of the ellipse. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the hexagon determining the location of the tile. This can be "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
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
            The location of the origin of the ellipse. The default is None which results in the ellipse being placed at (0, 0, 0).
        inputMode : int , optional
            The method by which the ellipse is defined. The default is 1.
            Based on the inputMode value, only the following inputs will be considered. The options are:
            1. Width and Length (considered inputs: width, length)
            2. Focal Length and Eccentricity (considered inputs: focalLength, eccentricity)
            3. Focal Length and Minor Axis Length (considered inputs: focalLength, minorAxisLength)
            4. Major Axis Length and Minor Axis Length (considered input: majorAxisLength, minorAxisLength)
        width : float , optional
            The width of the ellipse. The default is 2.0. This is considered if the inputMode is 1.
        length : float , optional
            The length of the ellipse. The default is 1.0. This is considered if the inputMode is 1.
        focalLength : float , optional
            The focal length of the ellipse. The default is 0.866025. This is considered if the inputMode is 2 or 3.
        eccentricity : float , optional
            The eccentricity of the ellipse. The default is 0.866025. This is considered if the inputMode is 2.
        majorAxisLength : float , optional
            The length of the major axis of the ellipse. The default is 1.0. This is considered if the inputMode is 4.
        minorAxisLength : float , optional
            The length of the minor axis of the ellipse. The default is 0.5. This is considered if the inputMode is 3 or 4.
        sides : int , optional
            The number of sides of the ellipse. The default is 32.
        fromAngle : float , optional
            The angle in degrees from which to start creating the arc of the ellipse. The default is 0.
        toAngle : float , optional
            The angle in degrees at which to end creating the arc of the ellipse. The default is 360.
        close : bool , optional
            If set to True, arcs will be closed by connecting the last vertex to the first vertex. Otherwise, they will be left open.
        direction : list , optional
            The vector representing the up direction of the ellipse. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the ellipse. This can be "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

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
            The desired length of the mantissa. The default is 6.
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
    def ExternalBoundary(face, silent=False):
        """
        Returns the external boundary (closed wire) of the input face.

        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

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
        eb = face.ExternalBoundary() # Hook to Core
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
            The input direction. The default is [0,0,-1].
        asVertex : bool , optional
            If set to True, the direction is treated as an actual vertex in 3D space. The default is False.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

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
        if dot < tolerance:
            return False
        return True

    @staticmethod
    def Fillet(face, radius: float = 0, radiusKey: str = None, tolerance: float = 0.0001, silent: bool = False):
        """
        Fillets (rounds) the interior and exterior corners of the input face given the input radius. See https://en.wikipedia.org/wiki/Fillet_(mechanics)

        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        radius : float
            The desired radius of the fillet.
        radiusKey : str , optional
            If specified, the dictionary of the vertices will be queried for this key to specify the desired fillet radius. The default is None.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

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
        eb = Wire.Fillet(eb, radius=radius, radiusKey=radiusKey, tolerance=tolerance)
        if not Topology.IsInstance(eb, "Wire"):
            if not silent:
                print("Face.Fillet - Error: The operation failed. Returning None.")
            return None
        ib_wires = []
        for ib in ib_list:
            ib = Wire.ByVertices(Topology.Vertices(ib))
            ib = Wire.Reverse(ib)
            if isinstance(radiusKey, str):
                ib = Topology.TransferDictionariesBySelectors(ib, selectors=f_vertices, tranVertices=True)
            
            ib_wire = Wire.Fillet(ib, radius=radius, radiusKey=radiusKey, tolerance=tolerance, silent=silent)
            if Topology.IsInstance(ib, "Wire"):
                ib_wires.append(ib_wire)
            else:
                if not silent:
                    print("Face.Fillet - Error: The operation for one of the interior boundaries failed. Skipping.")
        return Face.ByWires(eb, ib_wires)

    @staticmethod
    def Harmonize(face, tolerance: float = 0.0001):
        """
        Returns a harmonized version of the input face such that the *u* and *v* origins are always in the upperleft corner.

        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

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
            print("Face.Harmonize - Error: The input face parameter is not a valid face. Returning None.")
            return None
        normal = Face.Normal(face)
        origin = Topology.Centroid(face)
        flatFace = Topology.Flatten(face, origin=origin, direction=normal)
        world_origin = Vertex.Origin()
        vertices = Topology.Vertices(Face.ExternalBoundary(flatFace))
        harmonizedEB = Wire.ByVertices(vertices)
        internalBoundaries = Face.InternalBoundaries(flatFace)
        harmonizedIB = []
        for ib in internalBoundaries:
            ibVertices = Topology.Vertices(ib)
            harmonizedIB.append(Wire.ByVertices(ibVertices))
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
            The desired length of the mantissa. The default is 6.
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
        _ = face.InternalBoundaries(wires) # Hook to Core
        return list(wires)

    @staticmethod
    def InternalVertex(face, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a vertex guaranteed to be inside the input face.

        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

        Returns
        -------
        topologic_core.Vertex
            The created vertex.

        """
        def get_uv_radially():
            """
            Generate the points of a grid with a given size n, sorted radially from the center to the periphery.
            n should be an odd number, ensuring that there's a center point (0, 0).
            
            Args:
                n (int): The size of the grid. It should be odd for a clear center point.
                
            Returns:
                list: A list of tuples (x, y) sorted by radial distance from the center (0, 0).
            """
            import math

            points = []
            n = 100
            # Iterate over the grid, ranging from -n//2 to n//2
            for x in range(-n//2, n//2 + 1):
                for y in range(-n//2, n//2 + 1):
                    points.append((x, y))
            
            # Sort points by their Euclidean distance from the center (0, 0)
            points.sort(key=lambda point: math.sqrt(point[0]**2 + point[1]**2))
            return_points = []
            for p in points:
                new_p = ((p[0]+50)*0.01, (p[1]+50)*0.01)
                return_points.append(new_p)
            return return_points
        
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(face, "Face"):
            return None
        vert = Topology.Centroid(face)
        if Vertex.IsInternal(vert, face, tolerance=tolerance):
            return vert
        uv_list = get_uv_radially()
        for uv in uv_list:
            u, v = uv
            vert = Face.VertexByParameters(face, u, v)
            if Vertex.IsInternal(vert, face, tolerance=tolerance):
                return vert
        if not silent:
            print("Face.InternalVertex - Warning: Could not find an internal vertex. Returning the first vertex of the face.")
        vert = Topology.Vertices(face)[0]
        #v = topologic.FaceUtility.InternalVertex(face, tolerance) # Hook to Core
        return vert

    @staticmethod
    def Invert(face, tolerance: float = 0.0001):
        """
        Creates a face that is an inverse (mirror) of the input face.

        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

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
        inverted_wire = Wire.ByVertices(vertices)
        internal_boundaries = Face.InternalBoundaries(face)
        if not internal_boundaries:
            inverted_face = Face.ByWire(inverted_wire, tolerance=tolerance)
        else:
            inverted_face = Face.ByWires(inverted_wire, internal_boundaries, tolerance=tolerance)
        return inverted_face

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
            The length of the desired mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

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
    def Isovist(face, vertex, obstacles: list = [], direction: list = [0,1,0], fov: float = 360, transferDictionaries: bool = False, metrics: bool = False, triangles: bool = False, mantissa: int = 6, tolerance: float = 0.0001):
        """
        Returns the face representing the isovist projection from the input viewpoint.
        This method assumes all input is in 2D. Z coordinates are ignored.
        This method and the metrics are largely derived from isovists.org. Even if not explicitly listed, please assume that all credit
        goes to the authors of that website and its associated software.

        Parameters
        ----------
        face : topologic_core.Face
            The face representing the boundary of the isovist.
        vertex : topologic_core.Vertex
            The vertex representing the location of the viewpoint of the isovist.
        obstacles : list , optional
            A list of wires representing the obstacles within the face. All obstacles are assumed to be within the
            boundary of the face. The default is [].
        direction : list, optional
            The vector representing the direction (in the XY plane) in which the observer is facing. The Z component is ignored.
            The direction follows the Vector.CompassAngle convention where [0,1,0] (North) is considered to be
            in the positive Y direction, [1,0,0] (East) is considered to be in the positive X-direction.
            Angles are measured in a clockwise fashion. The default is [0,1,0] (North).
        fov : float , optional
            The horizontal field of view (fov) angle in degrees. See https://en.wikipedia.org/wiki/Field_of_view.
            The acceptable range is 1 to 360. The default is 360.
        transferDictionaries : bool , optional
            If set to True, the dictionaries of the encountered edges will be transfered to the isovist edges. The default is False.
        metrics : bool , optional
            If set to True, the following metrics are calculated and stored in the dictionary of the returned isovist. The keys of the values are:
            - viewpoint : list , the x , y , z coordinates of the location of the viewpoint.
            - direction : list , the direction of the view.
            - fov : int, Field of view angle.
            - area : float , the area of the isovist.
            - perimeter : float , the perimeter length of the isovist
            - compactness : float , how closely the shape of the isovist approximates a circle (the most compact geometric shape).
            - d_max : float, Maximum Visibility Distance. the length of the longest straight line that can be seen from the viewpoint.
            - d_min : float, Minimum Visibility Distance. the length of the shortest straight line that can be seen from the viewpoint.
            - d_avg : float, Average Visibility Distance. the length of the average straight line that can be seen from the viewpoint.
            - v_max : list, Furthest Point measures the x , y , z coordinates of the furthest visible point from the viewpoint.
            - v_min : list, Closest Point measures the x , y , z coordinates of the closest visible point from the viewpoint.
            - centroid: list, Centroid measures the x, y, z coordinates of the centroid of the isovist face.
            - v_d :  list, Visibility Distribution quantifies the angular distribution (in degrees) of visible points across the isovist.
                    This metric can tell you whether the visibility from a point is more spread out or concentrated in a certain direction. A uniform visibility distribution indicates a more balanced visual field, while a skewed distribution suggests that the observer's line of sight is constrained in certain directions.
            - v_density : float, Viewpoint Density which refers to the number of visible points per unit area within the isovist.
            - symmetry : float, Symmetry measures how balanced or symmetrical the isovist is around the point of observation.
            - d_f : float, Fractal Dimension measures the complexity of the isovist's boundary.
            - e_c : float , Edge Complexity measures how complex the edges of the isovist boundary are.
            - theta : float, Mean Visual Field Angle measures the average angular extent of the visible area from the observation point.
            - occlusivity: float, the proportion of edges of an isovist that are not physically defined.
            - drift: float, the distance from the observation point to the centroid of its isovist.
            - closed_perimeter: float, the total length of non-occluded edges of the isovist.
            - average_radial: float, "the mean view length of all space visible from a location." (from isovists.org)
            - variance: float, "the mean of the square of deviation between all radial lengths and average radial length of an isovist (Benedikt, 1979)." (from isovists.org)
            - skewness: float, "the mean of the cube of deviation between all radial lengths and average radial length of an isovist (Benediky, 1979)." (from isovists.org) 
        triangles : bool , optional
            If set to True, the subtended triangles of the isovist are created and stored as contents of the returned isovist face. The default is False.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional:
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Face
            The face representing the isovist projection from the input viewpoint.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Shell import Shell
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Vector import Vector
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Helper import Helper

        import math
        import numpy as np

        def calculate_angle(viewpoint, vertex):
            # Calculate the angle between the viewpoint and the vertex in the 2D plane
            # Viewpoint is (x, y, z), and vertex is (x, y)
            dx = vertex[0] - viewpoint[0]
            dy = vertex[1] - viewpoint[1]
            
            # Return the angle in radians using the arctangent of the y/x difference
            return math.degrees(math.atan2(dy, dx))
        
        def visibility_distribution(viewpoint, isovist_vertices):
            angles = []
            # Calculate the angle of each vertex with respect to the viewpoint
            for vertex in isovist_vertices:
                angle = calculate_angle(viewpoint, vertex)
                angles.append(angle)
            # Sort the angles to analyze the distribution
            angles = np.sort(angles)
            return list(angles)

        def isovist_symmetry(viewpoint, isovist_vertices):
            """
            Calculates the symmetry of the isovist polygon.
            
            Parameters:
            - viewpoint: a tuple (x, y) of the viewpoint.
            - isovist_vertices: a list of tuples, each representing a vertex (x, y) of the isovist polygon.
            
            Returns:
            - symmetry value: A measure of the symmetry of the isovist.
            """
            angles = [calculate_angle(viewpoint, vertex) for vertex in isovist_vertices]
            angles.sort()

            # Calculate angular deviations from the mean direction
            mean_angle = np.mean(angles)
            angular_deviation = np.std(angles)
            symmetry = angular_deviation / mean_angle if mean_angle != 0 else 0
            return float(symmetry)
        
        # Fractal Dimension (D_f) using Box-counting
        def fractal_dimension(isovist_vertices):
            """
            Calculates the fractal dimension of the isovist boundary using box-counting.
            
            Parameters:
            - isovist_vertices: a list of tuples, each representing a vertex (x, y) of the isovist polygon.
            
            Returns:
            - fractal dimension: A measure of the boundary's complexity.
            """
            # Convert isovist vertices into a boundary path (x, y coordinates)
            boundary_points = np.array(isovist_vertices)

            # Box-counting approach
            sizes = np.logspace(0, 2, 10)  # Varying box sizes (log scale)
            sizes[sizes == 0] = 1e-10  # Replace zero counts with a small value
            counts = []
            
            for size in sizes:
                count = 0
                for i in range(len(boundary_points)):
                    if np.abs(boundary_points[i][0] - boundary_points[(i+1)%len(boundary_points)][0]) > size or \
                    np.abs(boundary_points[i][1] - boundary_points[(i+1)%len(boundary_points)][1]) > size:
                        count += 1
                counts.append(count)

            # To avoid log(0), add a small constant to counts
            counts = np.array(counts)
            counts = np.where(counts == 0, 1e-10, counts)
            # Linear regression of log(count) vs log(size) to estimate fractal dimension
            log_sizes = np.log(sizes)
            log_counts = np.log(counts)
            
            # Perform linear regression (log-log scale)
            slope, _ = np.polyfit(log_sizes, log_counts, 1)
            
            return slope

        # Edge Complexity (E_C)
        def edge_complexity(isovist_vertices):
            """
            Calculates the edge complexity of the isovist boundary.
            
            Parameters:
            - isovist_vertices: a list of tuples, each representing a vertex (x, y) of the isovist polygon.
            
            Returns:
            - edge complexity: A measure of the complexity of the boundary.
            """
            angles = []
            for i in range(len(isovist_vertices)):
                p1 = isovist_vertices[i]
                p2 = isovist_vertices[(i + 1) % len(isovist_vertices)]
                p3 = isovist_vertices[(i + 2) % len(isovist_vertices)]

                # Calculate the angle between each consecutive edge
                angle = np.arctan2(p3[1] - p2[1], p3[0] - p2[0]) - np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
                angles.append(np.abs(angle))

            # Complexity is the number of abrupt angle changes
            complexity = np.sum(np.array(angles) > np.pi / 4)  # e.g., large changes in angles
            return float(complexity)

        # Mean Visual Field Angle (θ)
        def mean_visual_field_angle(viewpoint, isovist_vertices):
            """
            Calculates the mean visual field angle from the viewpoint to the isovist vertices.
            
            Parameters:
            - viewpoint: a tuple (x, y) of the viewpoint.
            - isovist_vertices: a list of tuples, each representing a vertex (x, y) of the isovist polygon.
            
            Returns:
            - mean visual field angle in degrees.
            """
            angles = [calculate_angle(viewpoint, vertex) for vertex in isovist_vertices]
            # Return the average angle
            return np.mean(angles)

        def vertex_part_of_face(vertex, face, tolerance=0.0001):
            vertices = Topology.Vertices(face)
            for v in vertices:
                if Vertex.Distance(vertex, v) < tolerance:
                    return True
            return False

        if not Topology.IsInstance(face, "Face"):
            print("Face.Isovist - Error: The input boundary parameter is not a valid Face. Returning None")
            return None
        if not Topology.IsInstance(vertex, "Vertex"):
            print("Face.Isovist - Error: The input viewPoint parameter is not a valid Vertex. Returning None")
            return None
        if fov < 1 or fov > 360:
            print("Face.Isovist - Error: The input fov parameter is outside the acceptable range of 0 to 360 degrees. Returning None")
            return None
        if isinstance(obstacles, list):
            obstacles = [obs for obs in obstacles if Topology.IsInstance(obs, "Wire")]
        else:
            obstacles = []
        
        def closest_distance_vertex(vertex, edge, mantissa):
            point = Vertex.Coordinates(vertex, mantissa=mantissa)
            line_start = Vertex.Coordinates(Edge.StartVertex(edge), mantissa=mantissa)
            line_end = Vertex.Coordinates(Edge.EndVertex(edge), mantissa=mantissa)

            # Convert input points to NumPy arrays for vector operations
            point = np.array(point)
            line_start = np.array(line_start)
            line_end = np.array(line_end)
            
            # Calculate the direction vector of the edge
            line_direction = line_end - line_start
            
            # Vector from the edge's starting point to the point
            point_to_start = point - line_start
            
            # Calculate the parameter 't' where the projection of the point onto the edge occurs
            if np.dot(line_direction, line_direction) == 0:
                t = 0
            else:
                t = np.dot(point_to_start, line_direction) / np.dot(line_direction, line_direction)
            
            # Check if 't' is outside the range [0, 1], and if so, calculate distance to closest endpoint
            if t < 0:
                t = 0
            elif t > 1:
                t = 1
            
            # Calculate the closest point on the edge to the given point
            closest_point = line_start + t * line_direction
            
            # Calculate the distance between the closest point and the given point
            distance = np.linalg.norm(point - closest_point)
            
            return float(distance), Vertex.ByCoordinates(list(closest_point))
        
        
        
        def compute_average_radial_variance_skewness(vertex, edges, mantissa=6):
            from math import atan2, pi, sqrt, pow

            def subtended_angle(vertex, edge, mantissa=6):
                """Compute the angle subtended by the edge at point V."""
                v = Vertex.Coordinates(vertex, mantissa=mantissa)
                start = Vertex.Coordinates(Edge.StartVertex(edge), mantissa=mantissa)
                end = Vertex.Coordinates(Edge.EndVertex(edge), mantissa=mantissa)
                # Calculate the angles of the start and end vertices relative to V
                angle_start = atan2(start[1] - v[1], start[0] - v[0])
                angle_end = atan2(end[1] - v[1], end[0] - v[0])
                # Ensure the angle is in the range [0, 2*pi]
                angle_start = angle_start if angle_start >= 0 else angle_start + 2 * pi
                angle_end = angle_end if angle_end >= 0 else angle_end + 2 * pi
                # Compute the difference and handle wrapping around 2*pi
                angle_diff = abs(angle_end - angle_start)
                return min(angle_diff, 2 * pi - angle_diff)
        
            total_weighted_distance = 0
            total_angle_weight = 0
            total_weighted_squared_deviation = 0
            total_weighted_cubed_deviation = 0
            distances = []
            angles = []
            for edge in edges:
                # Calculate the distance between V and the edge
                distance = Vertex.Distance(vertex, edge, mantissa=mantissa)
                distances.append(distance)

                # Calculate the subtended angle for the edge
                angle = subtended_angle(vertex, edge, mantissa=mantissa)
                angles.append(angle)

                # Weight the distance by the subtended angle
                total_weighted_distance += distance * angle
                total_angle_weight += angle

            # Compute the Average Radial value
            if total_angle_weight == 0:
                average_radial = 0  # Avoid division by zero
            else:
                average_radial = round(total_weighted_distance / total_angle_weight, mantissa)

            # Compute Variance
            for i, edge in enumerate(edges):
                # Calculate the distance between V and the edge
                distance = distances[i]
                # Calculate the subtended angle for the edge
                angle = angles[i]

                # Calculate the deviation squared from the average radial
                deviation_squared = (distance - average_radial) ** 2
                # Calculate the deviation cubed from the average radial
                deviation_cubed = (distance - average_radial) ** 3
                # Weight the squared deviation by the subtended angle
                total_weighted_squared_deviation += deviation_squared * angle
                total_weighted_cubed_deviation += deviation_cubed * angle

            # Compute the Variance value
            if total_angle_weight == 0:
                variance = 0  # Avoid division by zero
            else:
                variance = round(sqrt(total_weighted_squared_deviation / total_angle_weight), mantissa)
            
            # Compute the Skewness value
            if total_angle_weight == 0:
                skewness = 0  # Avoid division by zero
            else:
                skewness = round(pow(total_weighted_cubed_deviation / total_angle_weight, 1/3), mantissa)

            return average_radial, variance, skewness
        
        # Main Code
        origin = Topology.Centroid(face)
        normal = Face.Normal(face)
        flat_face = Topology.Flatten(face, origin=origin, direction=normal)
        flat_vertex = Topology.Flatten(vertex, origin=origin, direction=normal)
        flat_obstacles = [Topology.Flatten(obstacle, origin=origin, direction=normal) for obstacle in obstacles]

        eb = Face.ExternalBoundary(flat_face)
        vertices = Topology.Vertices(eb)
        coords = [Vertex.Coordinates(v, outputType="xy") for v in vertices]
        new_vertices = [Vertex.ByCoordinates(coord) for coord in coords]
        eb = Wire.ByVertices(new_vertices, close=True)

        ib_list = Face.InternalBoundaries(flat_face)
        new_ib_list = []
        for ib in ib_list:
            vertices = Topology.Vertices(ib)
            coords = [Vertex.Coordinates(v, outputType="xy") for v in vertices]
            new_vertices = [Vertex.ByCoordinates(coord) for coord in coords]
            new_ib_list.append(Wire.ByVertices(new_vertices, close=True))

        flat_face = Face.ByWires(eb, new_ib_list)
        for obs in flat_obstacles:
            flat_face = Topology.Difference(flat_face, Face.ByWire(obs))
        
        # Check that the viewpoint is inside the face
        if not Vertex.IsInternal(flat_vertex, flat_face):
            print("Face.Isovist - Error: The viewpoint is not inside the face. Returning None.")
            return None
        targets = Topology.Vertices(flat_face)
        distances = []
        for target in targets:
            distances.append(Vertex.Distance(flat_vertex, target))
        distances.sort()
        max_d = distances[-1]*1.05
        edges = []
        for target in targets:
            if Vertex.Distance(flat_vertex, target) > tolerance:
                e = Edge.ByVertices(flat_vertex, target, silent=True)
                e = Edge.SetLength(e, length=max_d, bothSides=False, tolerance=tolerance)
                edges.append(e)
        shell = Topology.Slice(flat_face, Cluster.ByTopologies(edges))
        faces = Topology.Faces(shell)
        final_faces = []
        for f in faces:
            if vertex_part_of_face(flat_vertex, f, tolerance=0.001):
                final_faces.append(f)
        shell = Shell.ByFaces(final_faces)
        return_face = Topology.RemoveCoplanarFaces(shell, epsilon=0.1)

        if not Topology.IsInstance(return_face, "face"):
            temp = Shell.ExternalBoundary(shell)
            if Topology.IsInstance(temp, "Wire"):
                return_face = Face.ByWire(temp)
            elif Topology.IsInstance(temp, "Cluster"):
                edges = Topology.Edges(temp)
                vertices = Topology.Vertices(temp)
                vertices = Vertex.Fuse(vertices, tolerance=0.01)
                new_edges = []
                for edge in edges:
                    sv = vertices[Vertex.Index(Edge.StartVertex(edge), vertices, tolerance=0.01)]
                    ev = vertices[Vertex.Index(Edge.EndVertex(edge), vertices, tolerance=0.01)]
                    if Vertex.Distance(sv, ev) > tolerance:
                        new_edges.append(Edge.ByVertices([sv,ev]))
                w = Wire.ByEdges(new_edges, tolerance=0.01)
                return_face = Face.ByWire(w)
            if not Topology.IsInstance(return_face, "Face"):
                print("Face.Isovist - Error: Could not create isovist. Returning None.")
                return None 
        compAngle = 0
        if fov == 360:
            pie = Face.Circle(origin= flat_vertex, radius=max_d*1.2, sides=16)
        else:
            compAngle = Vector.CompassAngle(Vector.North(), direction, mantissa=mantissa, tolerance=tolerance)
            fromAngle =  compAngle - fov*0.5 
            toAngle = compAngle + fov*0.5
            c = Wire.Circle(origin= flat_vertex, radius=max_d*1.2, sides=int(16*(fov/360)), fromAngle=fromAngle, toAngle=toAngle, close = False)
            e1 = Edge.ByVertices(Wire.StartVertex(c), flat_vertex, silent=True)
            e2 = Edge.ByVertices(Wire.EndVertex(c), flat_vertex, silent=True)
            edges = Topology.Edges(c) + [e1,e2]
            pie = Face.ByWire(Topology.SelfMerge(Cluster.ByTopologies(edges)))
        return_face = Topology.Intersect(pie, return_face)
        if not Topology.IsInstance(return_face, "face"):
            return_face = Topology.SelfMerge(return_face)
        if return_face == None:
            print("Face.Isovist - Error: Could not create isovist. Returning None.")
            return None
        simpler_face = Face.RemoveCollinearEdges(return_face)
        if Topology.IsInstance(simpler_face, "face"):
            if transferDictionaries == True or metrics == True:
                j_edges = [Topology.Edges(t) for t in obstacles]
                j_edges = Helper.Flatten(j_edges)
                j_edges += Topology.Edges(face)
                i_edges = Topology.Edges(simpler_face)
                used = [0 for _ in range(len(j_edges))]
                for i, i_edge in enumerate(i_edges):
                    d_i = Topology.Dictionary(i_edge)
                    d_i = Dictionary.SetValueAtKey(d_i, "occlusive", True)
                    i_edge = Topology.SetDictionary(i_edge, d_i)
                    for j, j_edge in enumerate(j_edges):
                        if used[j] == 0:
                            if Edge.IsCollinear(i_edge, j_edge):
                                d_i = Dictionary.SetValueAtKey(d_i, "occlusive", False)
                                i_edge = Topology.SetDictionary(i_edge, d_i)
                                d_j = Topology.Dictionary(j_edge)
                                d_result = Dictionary.ByMergedDictionaries([d_i, d_j])
                                if transferDictionaries == True:
                                    i_edge = Topology.SetDictionary(i_edge, d_result)
                                used[j] == 1
            
            return_face = Topology.Unflatten(simpler_face, origin=origin, direction=normal)
        else:
            return_face = Topology.Unflatten(return_face, origin=origin, direction=normal)
        if metrics == True:
            vertices = Topology.Vertices(return_face)
            # 1 Viewpoint
            viewpoint = Vertex.Coordinates(vertex, mantissa=mantissa)
            # 2 Direction
            # direction is given
            # 3 Field of View (FOV)
            # fov is given
            # 4 Area
            area = round(abs(Face.Area(return_face)), mantissa)
            # 5 Perimeter
            perimeter = round(Wire.Length(Face.Wires(return_face)[0]), mantissa)
            # 6 Compactness
            compactness = round(Face.Compactness(return_face), mantissa)
            # 7 Maximum Distance (d_max)
            # 8 Minimum Distance (d_min)
            # 9 Average Distance (d_avg)
            # 10 Furthest Visible Vertex (v_max)
            # 11 Closest Visible Vertex (v_min)
            d_max = round(Vertex.Distance(vertex, vertices[0]), mantissa)
            d_min = round(Vertex.Distance(vertex, vertices[0]), mantissa)
            furthest_vertex = vertices[0]
            closest_vertex = vertices[0]
            coords = []
            distances = []
            for v in vertices:
                coords.append(Vertex.Coordinates(v, mantissa=mantissa))
                dis = Vertex.Distance(vertex, v, mantissa=mantissa)
                distances.append(dis)
                if dis > d_max:
                    d_max  = dis
                    furthest_vertex = v
            distances = []
            edges = Topology.Edges(Cluster.ByTopologies([face]+obstacles))
            for edge in edges:
                dis, c_v = closest_distance_vertex(vertex, edge, mantissa=mantissa)
                if dis < d_min and Vertex.IsPeripheral(c_v, return_face):
                    d_min = dis
                    closest_vertex = c_v
            
            # 12 Average Visible Distance
            if len(distances) > 0:
                d_avg = sum(distances)/float(len(distances))
            else:
                d_avg = 0
            
            # 10 Furthest Visible Vertex (v_max)
            v_max = Vertex.Coordinates(furthest_vertex, mantissa=mantissa)
            # 11 Closest Visible Vertex (v_min)
            v_min = Vertex.Coordinates(closest_vertex, mantissa=mantissa)
            # 12 Centroid of Isovist (centroid)
            centroid = Vertex.Coordinates(Topology.Centroid(return_face), mantissa=mantissa)

            # 13 Visibility Distribution (v_d)
            v_d = visibility_distribution(viewpoint, coords)
            v_d = [round(x) for x in v_d]
            # 14 Viewpoint density
            if abs(Face.Area(return_face)) > 0:
                v_density = round(float(len(vertices)) / abs(Face.Area(return_face)), mantissa)
            else:
                v_density = 0
            # 15 Isovist Symmetry
            symmetry = round(isovist_symmetry(viewpoint, coords), mantissa)
            # 16 Fractal Dimension
            d_f = round(fractal_dimension(coords), mantissa)
            # 17 Edge Complexity
            e_c = round(edge_complexity(coords), mantissa)
            # 18 Mean Visual Field Angle
            theta = round(mean_visual_field_angle(viewpoint, coords), mantissa)
            # 19 Occlusivity
            occ_length = 0
            edges = Topology.Edges(return_face)
            for edge in edges:
                d = Topology.Dictionary(edge)
                if Dictionary.ValueAtKey(d, "occlusive") == True:
                    occ_length += Edge.Length(edge)
            if perimeter > 0:
                occlusivity = round(occ_length/perimeter, mantissa)
            else:
                occlusivity = round(0.0, 6)
            
            # 20 Drift
            drift = Vertex.Distance(vertex, Topology.Centroid(return_face), mantissa=mantissa)

            # 21 Closed Perimeter
            closed_perimeter = round(perimeter - occ_length, mantissa)

            # 22/23/24 Average Radial, Variance, and Skewness
            average_radial, variance, skewness = compute_average_radial_variance_skewness(vertex, edges, mantissa=6)

            keys = ["viewpoint",
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
                    "skewness"]
            
            values = [viewpoint,
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
                      skewness]
            d = Dictionary.ByKeysValues(keys, values)
            return_face = Topology.SetDictionary(return_face, d)
        if triangles:
            triangle_list = []
            edges = Topology.Edges(return_face)
            for edge in edges:
                d = Topology.Dictionary(edge)
                if Vertex.Distance(Edge.StartVertex(edge), v) > 0.0001:
                    e1 = Edge.ByVertices(Edge.StartVertex(edge), v)
                    if Vertex.Distance(Edge.EndVertex(edge), v) > 0.0001:
                        e2 = Edge.ByVertices(Edge.EndVertex(edge), v)
                        triangle = Topology.SelfMerge(Cluster.ByTopologies(edge, e1, e2))
                        if Topology.IsInstance(triangle, "wire"):
                            if Wire.IsClosed(triangle):
                                triangle = Face.ByWire(triangle, silent=True)
                                if Topology.IsInstance(triangle, "face"):
                                    if transferDictionaries == True:
                                        triangle = Topology.SetDictionary(triangle, d)
                                        tri_edges = Topology.Edges(triangle)
                                        for tri_edge in tri_edges:
                                            tri_edge = Topology.SetDictionary(tri_edge, d)
                                    triangle_list.append(triangle)
            if len(triangle_list) > 0:
                return_face = Topology.AddContent(return_face, triangle_list)
        return return_face

    @staticmethod
    def MedialAxis(face, resolution: int = 0, externalVertices: bool = False, internalVertices: bool = False, toLeavesOnly: bool = False, angTolerance: float = 0.1, tolerance: float = 0.0001):
        """
        Returns a wire representing an approximation of the medial axis of the input topology. See https://en.wikipedia.org/wiki/Medial_axis.

        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        resolution : int , optional
            The desired resolution of the solution (range is 0: standard resolution to 10: high resolution). This determines the density of the sampling along each edge. The default is 0.
        externalVertices : bool , optional
            If set to True, the external vertices of the face will be connected to the nearest vertex on the medial axis. The default is False.
        internalVertices : bool , optional
            If set to True, the internal vertices of the face will be connected to the nearest vertex on the medial axis. The default is False.
        toLeavesOnly : bool , optional
            If set to True, the vertices of the face will be connected to the nearest vertex on the medial axis only if this vertex is a leaf (end point). Otherwise, it will connect to any nearest vertex. The default is False.
        angTolerance : float , optional
            The desired angular tolerance in degrees for removing collinear edges. The default is 0.1.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
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
        from topologicpy.Dictionary import Dictionary

        def touchesEdge(vertex,edges, tolerance=0.0001):
            if not Topology.IsInstance(vertex, "Vertex"):
                return False
            for edge in edges:
                u = Edge.ParameterAtVertex(edge, vertex, mantissa=6)
                if not u:
                    continue
                if 0<u<1:
                    return True
            return False

        # Flatten the input face
        origin = Topology.Centroid(face)
        normal = Face.Normal(face)
        flatFace = Topology.Flatten(face, origin=origin, direction=normal)

        # Create a Vertex at the world's origin (0, 0, 0)
        world_origin = Vertex.Origin()

        faceEdges = Face.Edges(flatFace)
        vertices = []
        resolution = 10 - resolution
        resolution = min(max(resolution, 1), 10)
        for e in faceEdges:
            for n in range(resolution, 100, resolution):
                vertices.append(Edge.VertexByParameter(e,n*0.01))
        
        voronoi = Shell.Voronoi(vertices=vertices, face=flatFace)
        voronoiEdges = Shell.Edges(voronoi)

        medialAxisEdges = []
        for e in voronoiEdges:
            sv = Edge.StartVertex(e)
            ev = Edge.EndVertex(e)
            svTouchesEdge = touchesEdge(sv, faceEdges, tolerance=tolerance)
            evTouchesEdge = touchesEdge(ev, faceEdges, tolerance=tolerance)
            if not svTouchesEdge and not evTouchesEdge:
                medialAxisEdges.append(e)

        extBoundary = Face.ExternalBoundary(flatFace)
        extVertices = Topology.Vertices(extBoundary)

        intBoundaries = Face.InternalBoundaries(flatFace)
        intVertices = []
        for ib in intBoundaries:
            intVertices = intVertices+Topology.Vertices(ib)
        
        theVertices = []
        if internalVertices:
            theVertices = theVertices+intVertices
        if externalVertices:
            theVertices = theVertices+extVertices

        tempWire = Topology.SelfMerge(Cluster.ByTopologies(medialAxisEdges), tolerance=tolerance)
        if Topology.IsInstance(tempWire, "Wire") and angTolerance > 0:
            tempWire = Wire.RemoveCollinearEdges(tempWire, angTolerance=angTolerance)
        medialAxisEdges = Wire.Edges(tempWire)
        for v in theVertices:
            nv = Vertex.NearestVertex(v, tempWire, useKDTree=False)

            if Topology.IsInstance(nv, "Vertex"):
                if toLeavesOnly:
                    adjVertices = Topology.AdjacentTopologies(nv, tempWire)
                    if len(adjVertices) < 2:
                        medialAxisEdges.append(Edge.ByVertices([nv, v], tolerance=tolerance))
                else:
                    medialAxisEdges.append(Edge.ByVertices([nv, v], tolerance=tolerance))
        medialAxis = Topology.SelfMerge(Cluster.ByTopologies(medialAxisEdges), tolerance=tolerance)
        if Topology.IsInstance(medialAxis, "Wire") and angTolerance > 0:
            medialAxis = Topology.RemoveCollinearEdges(medialAxis, angTolerance=angTolerance)
        medialAxis = Topology.Unflatten(medialAxis, origin=origin,direction=normal)
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
            The string defining the desired output. This can be any subset or permutation of "xyz". It is case insensitive. The default is "xyz".
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.

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
            return_normal = list(topologic.FaceUtility.NormalAtParameters(face, 0.5, 0.5))
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
            The desired length of the normal edge. The default is 1.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        silent : bool , optional
            If set to True, no error and warning messages are printed. Otherwise, they are. The default is False.

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
        if length < tolerance:
            if not silent:
                print("Face.NormalEdge - Error: The input length parameter is less than the input tolerance. Retuning None.")
            return None
        iv = Face.InternalVertex(face)
        u, v = Face.VertexParameters(face, iv)
        vec = Face.Normal(face)
        ev = Topology.TranslateByDirectionDistance(iv, vec, length)
        return Edge.ByVertices([iv, ev], tolerance=tolerance, silent=silent)

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
            The desired vertex to use as the origin of the plane to project the face unto. If set to None, the centroidof the input face is used. The default is None.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
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
            The vector direction of the projection. If None, the reverse vector of the receiving face normal will be used. The default is None.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

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
    def Rectangle(origin= None, width: float = 1.0, length: float = 1.0, direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001):
        """
        Creates a rectangle.

        Parameters
        ----------
        origin : topologic_core.Vertex, optional
            The location of the origin of the rectangle. The default is None which results in the rectangle being placed at (0, 0, 0).
        width : float , optional
            The width of the rectangle. The default is 1.0.
        length : float , optional
            The length of the rectangle. The default is 1.0.
        direction : list , optional
            The vector representing the up direction of the rectangle. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the rectangle. This can be "center", "lowerleft", "upperleft", "lowerright", "upperright". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Face
            The created face.

        """
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology
        
        wire = Wire.Rectangle(origin=origin, width=width, length=length, direction=direction, placement=placement, tolerance=tolerance)
        if not Topology.IsInstance(wire, "Wire"):
            print("Face.Rectangle - Error: Could not create the base wire for the rectangle. Returning None.")
            return None
        return Face.ByWire(wire, tolerance=tolerance)
    
    @staticmethod
    def RemoveCollinearEdges(face, angTolerance: float = 0.1, tolerance: float = 0.0001):
        """
        Removes any collinear edges in the input face.

        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        angTolerance : float , optional
            The desired angular tolerance. The default is 0.1.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        topologic_core.Face
            The created face without any collinear edges.

        """
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology
        import inspect
        
        if not Topology.IsInstance(face, "Face"):
            print("Face.RemoveCollinearEdges - Error: The input face parameter is not a valid face. Returning None.")
            curframe = inspect.currentframe()
            calframe = inspect.getouterframes(curframe, 2)
            print('caller name:', calframe[1][3])
            return None
        eb = Wire.RemoveCollinearEdges(Face.Wire(face), angTolerance=angTolerance, tolerance=tolerance)
        ib = [Wire.RemoveCollinearEdges(w, angTolerance=angTolerance, tolerance=tolerance) for w in Face.InternalBoundaries(face)]
        return Face.ByWires(eb, ib)
    
    @staticmethod
    def Simplify(face, tolerance=0.0001):
        """
        Simplifies the input face edges based on the Douglas Peucker algorthim. See https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm
        Part of this code was contributed by gaoxipeng. See https://github.com/wassimj/topologicpy/issues/35
        
        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001. Edges shorter than this length will be removed.

        Returns
        -------
        topologic_core.Face
            The simplified face.
        """
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(face, "Face"):
            print("Face.Simplify - Error: The input face parameter is not a valid face. Returning None.")
            return None
        
        eb = Face.ExternalBoundary(face)
        eb = Wire.Simplify(eb, tolerance=tolerance)
        ibList = Face.InternalBoundaries(face)
        ibList = [Wire.Simplify(ib) for ib in ibList]
        return Face.ByWires(eb, ibList)
    
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
            If set to True the original boundary is returned as part of the roof. Otherwise it is not. The default is True.
        tolerance : float , optional
            The desired tolerance. The default is 0.001. (This is set to a larger number than the usual 0.0001 as it was found to work better)

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
            The location of the origin of the square. The default is None which results in the square being placed at (0, 0, 0).
        size : float , optional
            The size of the square. The default is 1.0.
        direction : list , optional
            The vector representing the up direction of the square. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the square. This can be "center", "lowerleft", "upperleft", "lowerright", or "upperright". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

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
            The location of the origin of the squircle. The default is None which results in the squircle being placed at (0, 0, 0).
        radius : float , optional
            The desired radius of the squircle. The default is 0.5.
        sides : int , optional
            The desired number of sides of the squircle. The default is 121.
        a : float , optional
            The "a" factor affects the x position of the points to interpolate between a circle and a square.
            A value of 1 will create a circle. Higher values will create a more square-like shape. The default is 2.0.
        b : float , optional
            The "b" factor affects the y position of the points to interpolate between a circle and a square.
            A value of 1 will create a circle. Higher values will create a more square-like shape. The default is 2.0.
        direction : list , optional
            The vector representing the up direction of the circle. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the circle. This can be "center", "lowerleft", "upperleft", "lowerright", or "upperright". It is case insensitive. The default is "center".
        angTolerance : float , optional
            The desired angular tolerance. The default is 0.1.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

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
            The location of the origin of the star. The default is None which results in the star being placed at (0, 0, 0).
        radiusA : float , optional
            The outer radius of the star. The default is 1.0.
        radiusB : float , optional
            The outer radius of the star. The default is 0.4.
        rays : int , optional
            The number of star rays. The default is 5.
        direction : list , optional
            The vector representing the up direction of the star. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the star. This can be "center", "lowerleft", "upperleft", "lowerright", or "upperright". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

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
    def Trapezoid(origin= None, widthA: float = 1.0, widthB: float = 0.75, offsetA: float = 0.0, offsetB: float = 0.0, length: float = 1.0, direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001):
        """
        Creates a trapezoid.

        Parameters
        ----------
        origin : topologic_core.Vertex, optional
            The location of the origin of the trapezoid. The default is None which results in the trapezoid being placed at (0, 0, 0).
        widthA : float , optional
            The width of the bottom edge of the trapezoid. The default is 1.0.
        widthB : float , optional
            The width of the top edge of the trapezoid. The default is 0.75.
        offsetA : float , optional
            The offset of the bottom edge of the trapezoid. The default is 0.0.
        offsetB : float , optional
            The offset of the top edge of the trapezoid. The default is 0.0.
        length : float , optional
            The length of the trapezoid. The default is 1.0.
        direction : list , optional
            The vector representing the up direction of the trapezoid. The default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the trapezoid. This can be "center", or "lowerleft". It is case insensitive. The default is "center".
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

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
    def Triangulate(face, mode: int = 0, meshSize: float = None, mantissa: int = 6, tolerance: float = 0.0001) -> list:
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
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

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
                The desired tolerance. The default is 0.0001.
            
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
                    _ = topologic.FaceUtility.Triangulate(flatFace, float(i)*0.1, shell_faces) # Hook to Core
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
            If set to True, the effect of the trim will be reversed. The default is False.

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
        trimmed_face = topologic.FaceUtility.TrimByWire(face, wire, False) # Hook to Core
        if reverse:
            trimmed_face = face.Difference(trimmed_face)
        return trimmed_face
    
    @staticmethod
    def VertexByParameters(face, u: float = 0.5, v: float = 0.5):
        """
        Creates a vertex at the *u* and *v* parameters of the input face.

        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        u : float , optional
            The *u* parameter of the input face. The default is 0.5.
        v : float , optional
            The *v* parameter of the input face. The default is 0.5.

        Returns
        -------
        vertex : topologic vertex
            The created vertex.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(face, "Face"):
            return None
        return topologic.FaceUtility.VertexAtParameters(face, u, v) # Hook to Core
    
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
            The string defining the desired output. This can be any subset or permutation of "uv". It is case insensitive. The default is "uv".
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.

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
        params = topologic.FaceUtility.ParametersAtVertex(face, vertex) # Hook to Core
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
        _ = face.Vertices(None, vertices) # Hook to Core
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
        _ = face.Wires(None, wires) # Hook to Core
        return wires
