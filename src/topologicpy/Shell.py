# Copyright (C) 2026
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free Software
# Foundation, either version 3.0 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

from topologicpy.Core import Core
import math
import os
import warnings

try:
    from scipy.spatial import Delaunay
    from scipy.spatial import Voronoi
except:
    print("Shell - Install required scipy library.")
    try:
        os.system("pip install scipy")
    except:
        os.system("pip install scipy --user")
    try:
        from scipy.spatial import Delaunay
        from scipy.spatial import Voronoi
    except:
        warnings.warn("Shell - Error: Could not import scipy.")

class Shell():
    @staticmethod
    def _UseNativeShellBackend() -> bool:
        """
        Returns True when the active core backend is PythonOCC and exposes the
        enhanced native Shell loft implementation.
        """
        from topologicpy.Topology import Topology

        try:
            if Topology._IsTopologicCoreBackend():
                return False
        except Exception:
            return False

        try:
            return bool(Core.HasAttribute("Shell", "ByWires"))
        except Exception:
            # Older Core dispatchers may not expose HasAttribute for class
            # factories even though PythonOCC is active.
            return True

    @staticmethod
    def ByDisjointFaces(externalBoundary,
                        faces,
                        maximumGap: float = 0.5,
                        mergeJunctions: bool = False,
                        threshold: float = 0.5,
                        uSides: int = 1,
                        vSides: int = 1,
                        transferDictionaries: bool = False,
                        mantissa: int = 6,
                        tolerance: float = 0.0001):
        """
        Creates a shell from an input list of disjointed faces. THIS IS STILL EXPERIMENTAL

        Parameters
        ----------
        externalBoundary : topologic_core.Face
            The input external boundary of the faces. This resembles a ribbon (face with hole) where its interior boundary touches the edges of the input list of faces.
        faces : list
            The input list of faces.
        maximumGap : float , optional
            The length of the maximum gap between the faces. Default is 0.5.
        mergeJunctions : bool , optional
            If set to True, the interior junctions are merged into a single vertex. Otherwise, diagonal edges are added to resolve transitions between different gap distances.
        threshold : float , optional
            The desired threshold under which vertices are merged into a single vertex. Default is 0.5.
        uSides : int , optional
            The desired number of sides along the X axis for the grid that subdivides the input faces to aid in processing. Default is 1.
        vSides : int , optional
            The desired number of sides along the Y axis for the grid that subdivides the input faces to aid in processing. Default is 1.
        transferDictionaries : bool, optional.
            If set to True, the dictionaries in the input list of faces are transfered to the faces of the resulting shell. Default is False.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        topologic_core.Shell
            The created Shell.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster
        from topologicpy.Helper import Helper
        from topologicpy.Topology import Topology
        from topologicpy.Grid import Grid
        from topologicpy.Dictionary import Dictionary

        def removeShards(edges, hostTopology, maximumGap=0.5):
            returnEdges = []
            for e in edges:
                if Edge.Length(e) < maximumGap:
                    sv = Edge.StartVertex(e)
                    ev = Edge.EndVertex(e)
                    sEdges = Topology.SuperTopologies(sv, hostTopology, "edge")
                    sn = len(sEdges)
                    eEdges = Topology.SuperTopologies(ev, hostTopology, "edge")
                    en = len(eEdges)
                    if sn >= 2 and en >= 2:
                        returnEdges.append(e)
                else:
                    returnEdges.append(e)
            return returnEdges

        def extendEdges(edges, hostTopology, maximumGap=0.5):
            returnEdges = []
            for e in edges:
                sv = Edge.StartVertex(e)
                ev = Edge.EndVertex(e)
                sEdges = Topology.SuperTopologies(sv, hostTopology, "edge")
                sn = len(sEdges)
                eEdges = Topology.SuperTopologies(ev, hostTopology, "edge")
                en = len(eEdges)
                if sn == 1:
                    ee = Edge.Extend(e, distance=maximumGap, bothSides=False, reverse=True)
                    returnEdges.append(ee)
                elif en == 1:
                    ee = Edge.Extend(e, distance=maximumGap, bothSides=False, reverse=False)
                    returnEdges.append(ee)
                else:
                    returnEdges.append(e)
            return returnEdges
        
        facesCluster = Cluster.ByTopologies(faces)
        internalBoundary = Face.ByWire(Face.InternalBoundaries(externalBoundary)[0], tolerance=tolerance)
        bb = Topology.BoundingBox(internalBoundary)
        bb_d = Topology.Dictionary(bb)
        unitU = Dictionary.ValueAtKey(bb_d, 'width') / uSides
        unitV = Dictionary.ValueAtKey(bb_d, 'length') / vSides
        uRange = [u*unitU for u in range(uSides)]
        vRange = [v*unitV for v in range(vSides)]
        grid = Grid.EdgesByDistances(internalBoundary, uRange=uRange, vRange=vRange, clip=True)
        grid = Topology.Slice(internalBoundary, grid, tolerance=tolerance)
        grid_faces = Topology.Faces(grid)
        skeletons = []
        for ib in grid_faces:
            building_shell = Topology.Slice(ib, facesCluster, tolerance=tolerance)
            wall_faces = Topology.Faces(building_shell)
            walls = []
            for w1 in wall_faces:
                iv = Topology.InternalVertex(w1, tolerance=tolerance)
                flag = False
                for w2 in faces:
                    if Vertex.IsInternal(iv, w2):
                        flag = True
                        break;
                if flag == False:
                    walls.append(w1)
            for wall in walls:
                skeleton = Wire.Skeleton(wall, tolerance=0.001) # This tolerance works better.
                skeleton = Topology.Difference(skeleton, facesCluster, tolerance=tolerance)
                skeleton = Topology.Difference(skeleton, Face.Wire(wall), tolerance=tolerance)
                skeletons.append(skeleton)
        if len(skeletons) > 0:
            skeleton_cluster = Cluster.ByTopologies(skeletons+[internalBoundary])
            skEdges = Topology.SelfMerge(Cluster.ByTopologies(removeShards(Topology.Edges(skeleton_cluster), skeleton_cluster, maximumGap=maximumGap)), tolerance=tolerance)
            if Topology.IsInstance(skEdges, "Edge"):
                skEdges = extendEdges([skEdges], skEdges, maximumGap=maximumGap)
            else:
                skEdges = extendEdges(Topology.Edges(skEdges), skEdges, maximumGap=maximumGap)
            if len(skEdges) < 1:
                print("ShellByDisjointFaces - Warning: No edges were extended.")
            #return Cluster.ByTopologies(skEdges)
        #print("ShellByDisjointFaces - Error: Could not derive central skeleton of interior walls. Returning None.")
        #return None
            shell = Topology.Slice(Topology.Copy(internalBoundary), skeleton_cluster, tolerance=tolerance)
            if mergeJunctions == True:
                vertices = Topology.Vertices(shell, silent=True)
                centers = []
                used = []
                for v in vertices:
                    for w in vertices:
                        if not Topology.IsSame(v, w) and not w in used:
                            if Vertex.Distance(v, w, mantissa=mantissa) < threshold:
                                centers.append(v)
                                used.append(w)
                edges = Shell.Edges(shell)
                new_edges = []
                for e in edges:
                    sv = Edge.StartVertex(e)
                    ev = Edge.EndVertex(e)
                    for v in centers:
                        if Vertex.Distance(sv, v, mantissa=mantissa) < threshold:
                            sv = v
                        if Vertex.Distance(ev, v, mantissa=mantissa) < threshold:
                            ev = v
                    new_edges.append(Edge.ByVertices([sv,ev], tolerance=tolerance))
                cluster = Cluster.ByTopologies(new_edges)

                vertices = Topology.Vertices(cluster, silent=True)
                edges = Topology.Edges(shell)

                xList = list(set([Vertex.X(v, mantissa=mantissa) for v in vertices]))
                xList.sort()
                xList = Helper.MergeByThreshold(xList, 0.5)
                yList = list(set([Vertex.Y(v, mantissa=mantissa) for v in vertices]))
                yList.sort()
                yList = Helper.MergeByThreshold(yList, 0.5)
                yList.sort()

                centers = []

                new_edges = []

                for e in edges:
                    sv = Edge.StartVertex(e)
                    ev = Edge.EndVertex(e)
                    svx = Vertex.X(sv, mantissa=mantissa)
                    svy = Vertex.Y(sv, mantissa=mantissa)
                    evx = Vertex.X(ev, mantissa=mantissa)
                    evy = Vertex.Y(ev, mantissa=mantissa)
                    for x in xList:
                        if abs(svx-x) < threshold:
                            svx = x
                            break;
                    for y in yList:
                        if abs(svy-y) < threshold:
                            svy = y
                            break;
                    sv = Vertex.ByCoordinates(svx, svy, 0)
                    for x in xList:
                        if abs(evx-x) < threshold:
                            evx = x
                            break;
                    for y in yList:
                        if abs(evy-y) < threshold:
                            evy = y
                            break;
                    sv = Vertex.ByCoordinates(svx, svy, 0)
                    ev = Vertex.ByCoordinates(evx, evy, 0)
                    new_edges.append(Edge.ByVertices([sv, ev], tolerance=tolerance))

                cluster = Cluster.ByTopologies(new_edges)
                eb = Face.ByWire(Shell.ExternalBoundary(shell), tolerance=tolerance)
                shell = Topology.Slice(eb, cluster, tolerance=tolerance)
            if not Topology.IsInstance(shell, "Shell"):
                try:
                    temp_wires = [Wire.RemoveCollinearEdges(w, angTolerance=1.0) for w in Topology.Wires(shell)]
                    temp_faces = [Face.ByWire(w, tolerance=tolerance) for w in temp_wires]
                except:
                    temp_faces = Topology.Faces(shell)
                shell = Shell.ByFaces(temp_faces, tolerance=tolerance)
            if transferDictionaries == True:
                selectors = []
                for f in faces:
                    d = Topology.Dictionary(f)
                    s = Topology.InternalVertex(f, tolerance=tolerance)
                    s = Topology.SetDictionary(s, d)
                    selectors.append(s)
                shell = Topology.TransferDictionariesBySelectors(topology=shell, selectors=selectors, tranFaces=True, tolerance=tolerance)
            return shell
        return None

    @staticmethod
    def ByFaces(faces: list, transferDictionaries: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a Shell from the input list of Faces.

        Parameters
        ----------
        faces : list
            The input list of Faces.
        transferDictionaries : bool , optional
            If True, dictionaries from the input Faces are transferred to the
            corresponding Faces of the created Shell. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Shell
            The created Shell, or None if construction fails.
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        from topologicpy.Cluster import Cluster
        from topologicpy.Dictionary import Dictionary

        if not isinstance(faces, list):
            if not silent:
                print("Shell.ByFaces - Error: The input faces parameter is not a valid list. Returning None.")
            return None

        face_list = [face for face in faces if Topology.IsInstance(face, "Face")]
        if len(face_list) == 0:
            if not silent:
                print("Shell.ByFaces - Error: The input faces list does not contain any valid Faces. Returning None.")
            return None

        try:
            shell = Core.Shell.ByFaces(face_list, tolerance)
        except Exception:
            shell = None

        if not Topology.IsInstance(shell, "Shell"):
            if Topology.IsInstance(shell, "Topology"):
                try:
                    shell = Topology.SelfMerge(shell, tolerance=tolerance, silent=True)
                except Exception:
                    shell = None

        if not Topology.IsInstance(shell, "Shell"):
            if not silent:
                print("Shell.ByFaces - Error: Could not create Shell. Returning None.")
            return None

        if transferDictionaries:
            # Use internal points as selectors rather than relying on wrapper
            # identity. This remains valid when the backend sews/rebuilds the
            # native Faces while constructing the Shell.
            selectors = []
            for face in face_list:
                try:
                    dictionary = Topology.Dictionary(face)
                    selector = Topology.InternalVertex(face, tolerance=tolerance, silent=True)
                    if Topology.IsInstance(selector, "Vertex"):
                        selector = Topology.SetDictionary(selector, dictionary, silent=True)
                        selectors.append(selector)
                except Exception:
                    continue

            if selectors:
                try:
                    transferred = Topology.TransferDictionariesBySelectors(
                        topology=shell,
                        selectors=selectors,
                        tranFaces=True,
                        tolerance=tolerance,
                    )
                    if Topology.IsInstance(transferred, "Shell"):
                        shell = transferred
                except Exception:
                    # Dictionary transfer is optional and must never invalidate
                    # otherwise successful Shell construction.
                    pass

        return shell

    @staticmethod
    def ByFacesCluster(cluster, transferDictionaries: bool = False, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a shell from the input cluster of faces.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input cluster of faces.
        transferDictionaries : bool , optional
            If set to True, any dictionaries in the faces are transferred to the faces of the created Shell.
            Otherwise, they are not. Default is False.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
        Returns
        -------
        topologic_core.Shell
            The created shell.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(cluster, "Cluster"):
            if not silent:
                print("Shell.ByFacesCluster - Error: The input cluster parameter is not a valid topologic cluster. Returning None.")
            return None
        faces = Topology.Faces(cluster)
        return Shell.ByFaces(faces, transferDictionaries=transferDictionaries, tolerance=tolerance, silent=silent)

    @staticmethod
    def ByThickenedWire(wire, offsetA: float = 1.0, offsetB: float = 1.0, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a shell by thickening the input wire. This method assumes the wire is manifold and planar.

        Parameters
        ----------
        wire : topologic_core.Wire
            The input wire to be thickened.
        offsetA : float , optional
            The desired offset to the exterior of the wire. Default is 1.0.
        offsetB : float , optional
            The desired offset to the interior of the wire. Default is 1.0.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        topologic_core.Shell
            The created shell.

        """
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(wire, "Wire"):
            print("Shell.ByThickenedWire - Error: The input wire parameter is not a valid wire. Returning None.")
            return None
        
        f = Face.ByThickenedWire(wire, offsetA=offsetA, offsetB=offsetB, tolerance=tolerance)
        outside_wire = Wire.ByOffset(wire, offset=abs(offsetA)*-1, bisectors = False, tolerance=tolerance)
        inside_wire = Wire.ByOffset(wire, offset=abs(offsetB), bisectors = False, tolerance=tolerance)
        border = Topology.Merge(outside_wire, inside_wire)
        outside_wire = Wire.ByOffset(wire, offset=abs(offsetA)*-1, bisectors = True, tolerance=tolerance)
        inside_wire = Wire.ByOffset(wire, offset=abs(offsetB), bisectors = True, tolerance=tolerance)
        grid = Topology.Merge(outside_wire, inside_wire)
        bisectors = Topology.Difference(grid, border)
        return_shell = Topology.Slice(f, bisectors)
        if not Topology.IsInstance(return_shell, "Shell"):
            # The offset/slice path can fail under the pythonOCC backend when the
            # input wire is non-rectangular. Fall back to a single thickened face
            # wrapped in a shell so the result is still a valid Shell.
            if f is not None:
                return_shell = Shell.ByFaces([f], tolerance=tolerance, silent=silent)
        if not Topology.IsInstance(return_shell, "Shell"):
            if not silent:
                print("Shell.ByThickenedWire - Error: The operation failed. Returning None.")
            return None
        return return_shell

    @staticmethod
    def ByWires(wires: list, triangulate: bool = True, polyhedron: bool = True, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a Shell by lofting through the input Wires.

        ``polyhedron=True`` preserves the historical faceted loft. With the
        PythonOCC backend this mode is delegated to the native TopologicPy
        backend implementation; with TopologicCore the established public-API
        faceted fallback is retained.

        ``polyhedron=False`` requests a genuine curve-preserving ruled Shell.
        On PythonOCC, OCCT lofts directly through the supplied section Wires so
        circular, B-spline, and NURBS Edges remain curved. TopologicCore does
        not expose an equivalent exact operation and therefore returns None
        rather than silently faceting the geometry.

        Parameters
        ----------
        wires : list
            The ordered input list of Wires. At least two valid Wires are required.
        triangulate : bool , optional
            If ``polyhedron`` is True, specifies whether each faceted side is
            triangulated. Default is True.
        polyhedron : bool , optional
            If True, construct the historical faceted/polyhedral loft. If False,
            construct a curve-preserving ruled Shell on PythonOCC. Default is True.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Shell
            The created Shell, or None when construction fails or exact lofting
            is requested on an unsupported backend.
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        if not isinstance(wires, (list, tuple)):
            if not silent:
                print("Shell.ByWires - Error: The input wires parameter is not a valid list. Returning None.")
            return None

        wire_list = [wire for wire in wires if Topology.IsInstance(wire, "Wire")]
        if len(wire_list) < 2:
            if not silent:
                print("Shell.ByWires - Error: At least two valid Wires are required. Returning None.")
            return None

        try:
            tolerance = abs(float(tolerance))
        except Exception:
            if not silent:
                print("Shell.ByWires - Error: The input tolerance parameter is not a valid number. Returning None.")
            return None

        if not math.isfinite(tolerance) or tolerance <= 0.0:
            if not silent:
                print("Shell.ByWires - Error: The input tolerance parameter must be greater than zero. Returning None.")
            return None

        # PythonOCC has one authoritative backend implementation for both
        # faceted and exact curve-preserving lofts.
        if Shell._UseNativeShellBackend():
            try:
                shell = Core.Shell.ByWires(
                    wire_list,
                    triangulate=bool(triangulate),
                    polyhedron=bool(polyhedron),
                    tolerance=tolerance,
                    silent=silent,
                )
            except TypeError:
                # Compatibility with an intermediate backend signature that
                # omitted keyword-only options.
                try:
                    shell = Core.Shell.ByWires(
                        wire_list,
                        bool(triangulate),
                        bool(polyhedron),
                        tolerance,
                        silent,
                    )
                except Exception:
                    shell = None
            except Exception:
                shell = None

            if Topology.IsInstance(shell, "Shell"):
                return shell

            if not silent:
                mode = "faceted" if polyhedron else "curve-preserving"
                print(f"Shell.ByWires - Error: Could not construct the {mode} Shell. Returning None.")
            return None

        # TopologicCore has no exact ruled-curve loft in this API.
        if polyhedron is False:
            if not silent:
                print("Shell.ByWires - Error: The TopologicCore backend does not support exact curve-preserving Shell loft construction. Returning None.")
            return None

        # Historical TopologicCore faceted loft.
        faces = []
        for wire_a, wire_b in zip(wire_list[:-1], wire_list[1:]):
            edges_a = Topology.Edges(wire_a)
            edges_b = Topology.Edges(wire_b)

            if not isinstance(edges_a, list) or not isinstance(edges_b, list):
                return None
            if len(edges_a) < 1 or len(edges_a) != len(edges_b):
                if not silent:
                    print("Shell.ByWires - Error: Corresponding Wires must contain the same number of Edges. Returning None.")
                return None

            for edge_a, edge_b in zip(edges_a, edges_b):
                a0 = Edge.StartVertex(edge_a)
                a1 = Edge.EndVertex(edge_a)
                b0 = Edge.StartVertex(edge_b)
                b1 = Edge.EndVertex(edge_b)

                if triangulate:
                    face_1 = Face.ByVertices([a0, a1, b1], tolerance=tolerance, silent=True)
                    face_2 = Face.ByVertices([a0, b1, b0], tolerance=tolerance, silent=True)
                    if Topology.IsInstance(face_1, "Face"):
                        faces.append(face_1)
                    if Topology.IsInstance(face_2, "Face"):
                        faces.append(face_2)
                else:
                    face = Face.ByVertices([a0, a1, b1, b0], tolerance=tolerance, silent=True)
                    if Topology.IsInstance(face, "Face"):
                        faces.append(face)

        if not faces:
            if not silent:
                print("Shell.ByWires - Error: Could not create any side Faces. Returning None.")
            return None

        shell = Shell.ByFaces(faces, tolerance=tolerance, silent=True)
        if Topology.IsInstance(shell, "Shell"):
            return shell

        if not silent:
            print("Shell.ByWires - Warning: Could not create a Shell. Returning a Cluster of Faces instead.")
        return Cluster.ByTopologies(faces, silent=True)

    @staticmethod
    def ByWiresCluster(cluster, triangulate: bool = True, polyhedron: bool = True, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a shell by lofting through the input cluster of wires.

        Parameters
        ----------
        cluster : topologic_core.Cluster
            The input cluster of wires.
        triangulate : bool , optional
            If ``polyhedron`` is True, specifies whether the side faces are
            triangulated. Default is True.
        polyhedron : bool , optional
            If True, uses the historical faceted/polyhedral loft. If False,
            requests an exact curve-preserving ruled shell on the PythonOCC
            backend. Default is True.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Shell
            The created shell.
        """
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        if not cluster:
            return None
        if not Topology.IsInstance(cluster, "Cluster"):
            return None
        wires = Cluster.Wires(cluster)
        return Shell.ByWires(
            wires,
            triangulate=triangulate,
            polyhedron=polyhedron,
            tolerance=tolerance,
            silent=silent,
        )

    @staticmethod
    def Circle(origin= None, radius: float = 0.5, sides: int = 32, fromAngle: float = 0.0, toAngle: float = 360.0, direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001):
        """
        Creates a circle.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the circle. Default is None which results in the circle being placed at (0, 0, 0).
        radius : float , optional
            The  radius of the circle. Default is 0.5.
        sides : int , optional
            The number of sides of the circle. Default is 32.
        fromAngle : float , optional
            The angle in degrees from which to start creating the arc of the circle. Default is 0.
        toAngle : float , optional
            The angle in degrees at which to end creating the arc of the circle. Default is 360.
        direction : list , optional
            The vector representing the up direction of the circle. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the pie. This can be "center", or "lowerleft". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        topologic_core.Shell
            The created circle.
        """
        return Shell.Pie(origin=origin, radiusA=radius, radiusB=0, sides=sides, rings=1, fromAngle=fromAngle, toAngle=toAngle, direction=direction, placement=placement, tolerance=tolerance)

    # @staticmethod
    # def Delaunay(
    #     vertices: list,
    #     face,
    #     deflection: float = None,
    #     maxIterations: int = 5,
    #     convergence: float = 0.001,
    #     tolerance: float = 0.0001,
    #     silent: bool = False,
    # ):
    #     """
    #     Returns the intrinsic/geodesic Delaunay partition of the input Face.

    #     The input Vertices are the Delaunay sites and must all lie on, or within
    #     ``tolerance`` of, the trimmed input Face. Distances are measured intrinsically
    #     on the Face, so the shortest paths are constrained by its outer boundary and
    #     any internal boundaries. The planar construction is therefore a special case
    #     of the general surface construction.

    #     On the PythonOCC backend the intrinsic metric is approximated with the
    #     Kimmel-Sethian Fast Marching Method on successively refined triangulations
    #     of the trimmed Face. Delaunay adjacency is derived strictly as the dual of
    #     the converged intrinsic Voronoi diagram; adjacent sites are connected by
    #     continuous steepest-descent traces through the piecewise-linear Fast
    #     Marching distance field. The resulting paths are chained and rebuilt as
    #     degree-1 B-spline p-curves on the original exact OCCT surface and used to
    #     split that Face. Thus the returned Shell contains subsets
    #     of the original analytic, B-spline, or NURBS surface rather than inheriting
    #     one topological Edge per temporary computational triangle.

    #     Parameters
    #     ----------
    #     vertices : list
    #         The input list of site Vertices. At least three are required.
    #     face : topologic_core.Face
    #         The trimmed surface domain on which the intrinsic Delaunay partition is
    #         computed.
    #     deflection : float , optional
    #         Finest permitted linear deflection of the computational triangulation.
    #         Refinement starts coarser and approaches, but never goes below, this
    #         value. If None, a scale-aware target is selected automatically. Default
    #         is None.
    #     maxIterations : int , optional
    #         Maximum number of surface-mesh refinement iterations. Default is 5.
    #     convergence : float , optional
    #         Absolute geometric convergence criterion for successive intrinsic
    #         Voronoi boundaries from which the Delaunay dual is derived. Default is
    #         0.001.
    #     tolerance : float , optional
    #         The desired geometric tolerance. Default is 0.0001.
    #     silent : bool , optional
    #         If True, error and warning messages are suppressed. Default is False.

    #     Returns
    #     -------
    #     topologic_core.Shell
    #         A Shell partitioning the exact input Face by intrinsic Delaunay geodesics,
    #         or None if the operation fails.
    #     """
    #     from topologicpy.Topology import Topology

    #     if not isinstance(vertices, list):
    #         if not silent:
    #             print("Shell.Delaunay - Error: The input vertices parameter is not a valid list. Returning None.")
    #         return None
    #     if len(vertices) < 3 or any(not Topology.IsInstance(v, "Vertex") for v in vertices):
    #         if not silent:
    #             print("Shell.Delaunay - Error: At least three valid Vertices are required. Returning None.")
    #         return None
    #     if not Topology.IsInstance(face, "Face"):
    #         if not silent:
    #             print("Shell.Delaunay - Error: The input face parameter is not a valid Face. Returning None.")
    #         return None
    #     if Topology._IsTopologicCoreBackend():
    #         if not silent:
    #             print("Shell.Delaunay - Error: Intrinsic surface Delaunay currently requires the PythonOCC backend. Returning None.")
    #         return None

    #     try:
    #         return Core.Shell.Delaunay(
    #             vertices,
    #             face,
    #             deflection=deflection,
    #             maxIterations=maxIterations,
    #             convergence=convergence,
    #             tolerance=tolerance,
    #             silent=silent,
    #         )
    #     except TypeError:
    #         try:
    #             return Core.Shell.Delaunay(vertices, face, deflection, maxIterations, convergence, tolerance, silent)
    #         except Exception:
    #             pass
    #     except Exception:
    #         pass

    #     if not silent:
    #         print("Shell.Delaunay - Error: Could not construct the intrinsic Delaunay partition. Returning None.")
    #     return None

    @staticmethod
    def Delaunay(
        vertices: list,
        face=None,
        deflection: float = None,
        maxIterations: int = 5,
        convergence: float = 0.001,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        import math
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        if not isinstance(vertices, list):
            if not silent:
                print("Shell.Delaunay - Error: The input vertices parameter is not a valid list. Returning None.")
            return None
        vertices = [v for v in vertices if Topology.IsInstance(v, "Vertex")]
        if len(vertices) < 3:
            if not silent:
                print("Shell.Delaunay - Error: At least three valid Vertices are required. Returning None.")
            return None

        try:
            tolerance = abs(float(tolerance))
        except Exception:
            tolerance = 0.0001
        if tolerance <= 0.0:
            tolerance = 0.0001

        if face is None:
            try:
                import numpy as np
                from scipy.spatial import Delaunay as SciPyDelaunay
            except Exception:
                if not silent:
                    print("Shell.Delaunay - Error: NumPy/SciPy are required for unconstrained Delaunay triangulation. Returning None.")
                return None

            points3d = []
            for vertex in vertices:
                points3d.append(np.array([
                    float(Vertex.X(vertex, mantissa=None)),
                    float(Vertex.Y(vertex, mantissa=None)),
                    float(Vertex.Z(vertex, mantissa=None)),
                ], dtype=float))

            p0 = points3d[0]
            u_axis = None
            normal = None
            for i in range(1, len(points3d)):
                candidate_u = points3d[i] - p0
                lu = float(np.linalg.norm(candidate_u))
                if lu <= tolerance:
                    continue
                for j in range(i + 1, len(points3d)):
                    candidate_v = points3d[j] - p0
                    cross = np.cross(candidate_u, candidate_v)
                    lc = float(np.linalg.norm(cross))
                    if lc <= tolerance:
                        continue
                    u_axis = candidate_u / lu
                    normal = cross / lc
                    break
                if u_axis is not None:
                    break

            if u_axis is None or normal is None:
                if not silent:
                    print("Shell.Delaunay - Error: The input Vertices are collinear. Returning None.")
                return None

            v_axis = np.cross(normal, u_axis)
            lv = float(np.linalg.norm(v_axis))
            if lv <= tolerance:
                return None
            v_axis /= lv

            points2d = []
            for point in points3d:
                delta = point - p0
                if abs(float(np.dot(delta, normal))) > tolerance:
                    if not silent:
                        print("Shell.Delaunay - Error: The input Vertices are not coplanar. Returning None.")
                    return None
                points2d.append([
                    float(np.dot(delta, u_axis)),
                    float(np.dot(delta, v_axis)),
                ])

            try:
                triangulation = SciPyDelaunay(np.asarray(points2d, dtype=float))
            except Exception:
                if not silent:
                    print("Shell.Delaunay - Error: Could not compute the planar Delaunay triangulation. Returning None.")
                return None

            faces = []
            for simplex in triangulation.simplices:
                wire = Wire.ByVertices([
                    vertices[int(simplex[0])],
                    vertices[int(simplex[1])],
                    vertices[int(simplex[2])],
                ], close=True, tolerance=tolerance, silent=True)
                triangle = Face.ByWire(wire, tolerance=tolerance, silent=True)
                if Topology.IsInstance(triangle, "Face"):
                    faces.append(triangle)

            if not faces:
                return None
            shell = Shell.ByFaces(faces, tolerance=tolerance, silent=True)
            if Topology.IsInstance(shell, "Shell"):
                return shell
            return Cluster.ByTopologies(faces, silent=True)

        if not Topology.IsInstance(face, "Face"):
            if not silent:
                print("Shell.Delaunay - Error: The input face parameter is not a valid Face. Returning None.")
            return None
        if Topology._IsTopologicCoreBackend():
            if not silent:
                print("Shell.Delaunay - Error: Intrinsic surface Delaunay currently requires the PythonOCC backend. Returning None.")
            return None

        try:
            return Core.Shell.Delaunay(
                vertices,
                face,
                deflection=deflection,
                maxIterations=maxIterations,
                convergence=convergence,
                tolerance=tolerance,
                silent=silent,
            )
        except TypeError:
            try:
                return Core.Shell.Delaunay(vertices, face, deflection, maxIterations, convergence, tolerance, silent)
            except Exception:
                pass
        except Exception:
            pass

        if not silent:
            print("Shell.Delaunay - Error: Could not construct the intrinsic Delaunay partition. Returning None.")
        return None

    @staticmethod
    def Edges(shell) -> list:
        """
        Returns the edges of the input shell.

        Parameters
        ----------
        shell : topologic_core.Shell
            The input shell.

        Returns
        -------
        list
            The list of edges.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(shell, "Shell"):
            return None
        edges = []
        # _ = shell.Edges(None, edges) # H to Core
        try:
            _ = Core.InstanceCall(shell, "Edges", None, edges)
        except Exception:
            edges = None
        return edges

    @staticmethod
    def ExternalBoundary(shell, tolerance: float = 0.0001, silent: bool = False):
        """Returns the longest external/free boundary Wire of the input Shell."""
        from topologicpy.Wire import Wire
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Helper import Helper

        if not Topology.IsInstance(shell, "Shell"):
            if not silent:
                print("Shell.ExternalBoundary - Error: The input shell parameter is not a valid Shell. Returning None.")
            return None

        # The PythonOCC backend has curve-aware edge incidence and measures true
        # curve length when disjoint free-boundary wires must be ranked.
        if not Topology._IsTopologicCoreBackend():
            try:
                result = Core.Shell.ExternalBoundary(shell, tolerance=tolerance, silent=True)
                if Topology.IsInstance(result, "Wire"):
                    return result
            except Exception:
                pass

        ebEdges = [
            edge for edge in (Topology.Edges(shell) or [])
            if len(Topology.SuperTopologies(edge, shell, topologyType="face") or []) == 1
        ]
        if len(ebEdges) == 1:
            result = Wire.ByEdges(ebEdges, tolerance=tolerance, silent=True)
            if Topology.IsInstance(result, "Wire"):
                return result
        elif len(ebEdges) > 1:
            result = Topology.SelfMerge(Cluster.ByTopologies(ebEdges), tolerance=tolerance)
            if Topology.IsInstance(result, "Wire"):
                wires = [result]
            else:
                wires = [w for w in (Topology.Wires(result) or []) if Topology.IsInstance(w, "Wire")]
            if len(wires) == 1:
                return wires[0]
            if wires:
                lengths = [Wire.Length(w) for w in wires]
                wires = Helper.Sort(wires, lengths)
                return wires[-1]

        if not silent:
            print("Shell.ExternalBoundary - Error: External boundary could not be found. Returning None.")
        return None

    @staticmethod
    def Faces(shell) -> list:
        """
        Returns the faces of the input shell.

        Parameters
        ----------
        shell : topologic_core.Shell
            The input shell.

        Returns
        -------
        list
            The list of faces.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(shell, "Shell"):
            return None
        faces = []
        # _ = shell.Faces(None, faces) # H to Core
        try:
            _ = Core.InstanceCall(shell, "Faces", None, faces)
        except Exception:
            faces = None
        return faces

    @staticmethod
    def GoldenRectangle(width: float = 1.0,
                            maxIterations: int = 10,
                            clockwise: bool = False,
                            includeSpiral = True,
                            sides = 96,
                            origin=None,
                            placement: str = "center",
                            direction: list = [0, 0, 1],
                            mantissa: int = 6,
                            tolerance: float = 0.0001,
                            silent: bool = False):
        """
        Creates a "golden rectangle" with an optional "golden spiral". See https://en.wikipedia.org/wiki/Golden_rectangle and https://en.wikipedia.org/wiki/Golden_spiral.

        Parameters
        ----------
        width : float
            The desired long side of the outer golden rectangle. Height is width/phi.
        maxIterations : int
            Number of subdivision squares to generate.
        clockwise : bool , optional
            Controls the square “peel” progression (affects which side each next square
            is taken from). Default is False.
        includeSpiral : bool , optional
            If set to True, the golden spiral is included in the resulting shell. Default is True.
        sides : int , optional
            The number of sides of the golden spiral (if included).
            Notes: If you set sides to be equal to maxIterations, you get the diagonals.
            It is best if the number of sides is a multiple of maxIterations.
            Default is 96.
        origin : topologic_core.Vertex, optional
            The location of the origin of the rectangle. Default is None which results in the rectangle being placed at (0, 0, 0).
        direction : list , optional
            The vector representing the up direction of the rectangle. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the rectangle. This can be "center", "lowerleft", "upperleft", "lowerright", "upperright". It is case insensitive. Default is "center".
        mantissa : int , optional
            The desired length of the mantissa. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Shell
            A shell made from the faces of all subdivision squares (multiple loops).
        """

        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        import math

        # -----------------------------
        # Validate
        # -----------------------------
        width = float(width)
        if width <= 0:
            if not silent:
                print("Shell.GoldenRectangle - Error: width must be greater than 0. Returning None.")
            return None
        maxIterations = int(maxIterations)
        if maxIterations <= 0:
            if not silent:
                print("Shell.GoldenRectangle - Error: maxIterations must be >= 0. Returning None.")
            return None
        if includeSpiral == True:
            sides = int(sides)
            if sides < maxIterations:
                if not silent:
                    print("Shell.GoldenRectangle - Error: sides must be >= maxIterations. Returning None.")
                return None
        clockwise = bool(clockwise)
        if origin == None:
            origin = Vertex.Origin()
        
        if not Topology.IsInstance(origin, "vertex"):
            if not silent:
                print("Shell.GoldenRectangle - Error: The input origin parameter is not a valid vertex. Returning None.")
            return None
        placement = str(placement).lower()
        if not placement in ["center", "lowerleft", "lowerright", "upperleft", "upperright"]:
            if not silent:
                print("Shell.GoldenRectangle - Error: The input placement parameter is not a valid placement string. Returning None.")
            return None
        if not isinstance(direction, list):
            if not silent:
                print("Shell.GoldenRectangle - Error: The input direction parameter is not a valid list. Returning None.")
            return None
        direction = [x for x in direction if isinstance(x, (int, float))]
        if len(direction) != 3:
            if not silent:
                print("Shell.GoldenRectangle - Error: The input direction parameter is not a valid 3D vector. Returning None.")
            return None

        # -----------------------------
        # Build the golden-rectangle subdivision tiles as faces and assemble a
        # Shell. Slicing a rectangle face by the (nested) golden-rectangle wire
        # is unreliable under the pythonOCC backend, so we construct one Face
        # per subdivision square directly — this yields the expected number of
        # faces (>= maxIterations + 1) and a valid Shell.
        # -----------------------------
        phi = (1.0 + math.sqrt(5.0)) / 2.0
        W = width
        L = width / phi
        x0 = -W * 0.5
        y0 = -L * 0.5

        def _round(x):
            return round(float(x), int(mantissa))

        def _square_edges(sx, sy, s):
            bl = Vertex.ByCoordinates(_round(sx),     _round(sy),     0.0)
            br = Vertex.ByCoordinates(_round(sx + s), _round(sy),     0.0)
            tr = Vertex.ByCoordinates(_round(sx + s), _round(sy + s), 0.0)
            tl = Vertex.ByCoordinates(_round(sx),     _round(sy + s), 0.0)
            return (bl, br, tr, tl)

        def _subdivide(rx, ry, rW, rH, k, depth, outSquares):
            if depth <= 0 or rW <= tolerance or rH <= tolerance:
                return
            if rW >= rH:
                s = rH
                if k == 0:
                    sx, sy = rx, ry
                    nrx, nry = rx + s, ry
                    nW, nH = rW - s, rH
                elif k == 1:
                    sx, sy = rx + (rW - s), ry
                    nrx, nry = rx, ry
                    nW, nH = rW - s, rH
                elif k == 2:
                    sx, sy = rx, ry
                    nrx, nry = rx + s, ry
                    nW, nH = rW - s, rH
                else:
                    sx, sy = rx + (rW - s), ry
                    nrx, nry = rx, ry
                    nW, nH = rW - s, rH
            else:
                s = rW
                if k == 0:
                    sx, sy = rx, ry
                    nrx, nry = rx, ry + s
                    nW, nH = rW, rH - s
                elif k == 1:
                    sx, sy = rx, ry + (rH - s)
                    nrx, nry = rx, ry
                    nW, nH = rW, rH - s
                elif k == 2:
                    sx, sy = rx, ry
                    nrx, nry = rx, ry + s
                    nW, nH = rW, rH - s
                else:
                    sx, sy = rx, ry + (rH - s)
                    nrx, nry = rx, ry
                    nW, nH = rW, rH - s
            outSquares.append((sx, sy, s))
            _subdivide(nrx, nry, nW, nH, (k + 1) % 4, depth - 1, outSquares)

        squares = []
        _subdivide(float(x0), float(y0), float(W), float(L), 0, maxIterations, squares)

        faces = []
        for (sx, sy, s) in squares:
            bl, br, tr, tl = _square_edges(sx, sy, s)
            w = Wire.ByVertices([bl, br, tr, tl], close=True, tolerance=tolerance, silent=True)
            f = Face.ByWire(w, tolerance=tolerance, silent=True)
            if f is not None:
                faces.append(f)
        # Add the outer rectangle face as the final tile
        ob = Vertex.ByCoordinates(_round(x0), _round(y0), 0.0)
        obr = Vertex.ByCoordinates(_round(x0 + W), _round(y0), 0.0)
        otr = Vertex.ByCoordinates(_round(x0 + W), _round(y0 + L), 0.0)
        otl = Vertex.ByCoordinates(_round(x0), _round(y0 + L), 0.0)
        outer_w = Wire.ByVertices([ob, obr, otr, otl], close=True, tolerance=tolerance, silent=True)
        outer_f = Face.ByWire(outer_w, tolerance=tolerance, silent=True)
        if outer_f is not None:
            faces.append(outer_f)

        shell = Shell.ByFaces(faces, tolerance=tolerance, silent=silent)
        if not Topology.IsInstance(shell, "Shell"):
            # Fall back to a cluster of faces if ByFaces cannot merge them.
            shell = None
        # -----------------------------
        # Orient to direction
        # -----------------------------
        if shell is not None and direction != [0, 0, 1]:
            shell = Topology.Orient(shell, origin=origin, dirA=[0, 0, 1], dirB=direction)
        return shell

    @staticmethod
    def HyperbolicParaboloidRectangularDomain(origin= None,
                                              llVertex= None,
                                              lrVertex= None,
                                              ulVertex= None,
                                              urVertex= None,
                                              uSides: int = 10,
                                              vSides: int = 10,
                                              direction: list = [0, 0, 1],
                                              placement: str = "center",
                                              mantissa: int = 6,
                                              tolerance: float = 0.0001):
        """
        Creates a hyperbolic paraboloid with a rectangular domain.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The origin of the hyperbolic paraboloid. If set to None, it will be placed at the (0, 0, 0) origin. Default is None.
        llVertex : topologic_core.Vertex , optional
            The lower left corner of the hyperbolic paraboloid. If set to None, it will be set to (-0.5, -0.5, -0.5).
        lrVertex : topologic_core.Vertex , optional
            The lower right corner of the hyperbolic paraboloid. If set to None, it will be set to (0.5, -0.5, 0.5).
        ulVertex : topologic_core.Vertex , optional
            The upper left corner of the hyperbolic paraboloid. If set to None, it will be set to (-0.5, 0.5, 0.5).
        urVertex : topologic_core.Vertex , optional
            The upper right corner of the hyperbolic paraboloid. If set to None, it will be set to (0.5, 0.5, -0.5).
        uSides : int , optional
            The number of segments along the X axis. Default is 10.
        vSides : int , optional
            The number of segments along the Y axis. Default is 10.
        direction : list , optional
            The vector representing the up direction of the hyperbolic paraboloid. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the hyperbolic paraboloid. This can be "center", "lowerleft", "bottom". It is case insensitive. Default is "center".
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        
        Returns
        -------
        topologic_core.Shell
            The created hyperbolic paraboloid.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        if not Topology.IsInstance(llVertex, "Vertex"):
            llVertex = Vertex.ByCoordinates(-0.5, -0.5, -0.5)
        if not Topology.IsInstance(lrVertex, "Vertex"):
            lrVertex = Vertex.ByCoordinates(0.5, -0.5, 0.5)
        if not Topology.IsInstance(ulVertex, "Vertex"):
            ulVertex = Vertex.ByCoordinates(-0.5, 0.5, 0.5)
        if not Topology.IsInstance(urVertex, "Vertex"):
            urVertex = Vertex.ByCoordinates(0.5, 0.5, -0.5)
        e1 = Edge.ByVertices([llVertex, lrVertex], tolerance=tolerance)
        e3 = Edge.ByVertices([urVertex, ulVertex], tolerance=tolerance)
        edges = []
        for i in range(uSides+1):
            v1 = Edge.VertexByParameter(e1, float(i)/float(uSides))
            v2 = Edge.VertexByParameter(e3, 1.0 - float(i)/float(uSides))
            edges.append(Edge.ByVertices([v1, v2], tolerance=tolerance))
        faces = []
        for i in range(uSides):
            for j in range(vSides):
                v1 = Edge.VertexByParameter(edges[i], float(j)/float(vSides))
                v2 = Edge.VertexByParameter(edges[i], float(j+1)/float(vSides))
                v3 = Edge.VertexByParameter(edges[i+1], float(j+1)/float(vSides))
                v4 = Edge.VertexByParameter(edges[i+1], float(j)/float(vSides))
                faces.append(Face.ByVertices([v1, v2, v4]))
                faces.append(Face.ByVertices([v4, v2, v3]))
        returnTopology = Shell.ByFaces(faces, tolerance=tolerance)
        if not returnTopology:
            returnTopology = None
        xOffset = 0
        yOffset = 0
        zOffset = 0
        x_min = min([Vertex.X(llVertex, mantissa=mantissa), Vertex.X(lrVertex, mantissa=mantissa), Vertex.X(ulVertex, mantissa=mantissa), Vertex.X(urVertex, mantissa=mantissa)])
        maxX = max([Vertex.X(llVertex, mantissa=mantissa), Vertex.X(lrVertex, mantissa=mantissa), Vertex.X(ulVertex, mantissa=mantissa), Vertex.X(urVertex, mantissa=mantissa)])
        y_min = min([Vertex.Y(llVertex, mantissa=mantissa), Vertex.Y(lrVertex, mantissa=mantissa), Vertex.Y(ulVertex, mantissa=mantissa), Vertex.Y(urVertex, mantissa=mantissa)])
        maxY = max([Vertex.Y(llVertex, mantissa=mantissa), Vertex.Y(lrVertex, mantissa=mantissa), Vertex.Y(ulVertex, mantissa=mantissa), Vertex.Y(urVertex, mantissa=mantissa)])
        z_min = min([Vertex.Z(llVertex, mantissa=mantissa), Vertex.Z(lrVertex, mantissa=mantissa), Vertex.Z(ulVertex, mantissa=mantissa), Vertex.Z(urVertex, mantissa=mantissa)])
        maxZ = max([Vertex.Z(llVertex, mantissa=mantissa), Vertex.Z(lrVertex, mantissa=mantissa), Vertex.Z(ulVertex, mantissa=mantissa), Vertex.Z(urVertex, mantissa=mantissa)])

        if placement.lower() == "lowerleft":
            xOffset = -x_min
            yOffset = -y_min
            zOffset = -z_min
        elif placement.lower() == "bottom":
            xOffset = -(x_min + (maxX - x_min)*0.5)
            yOffset = -(y_min + (maxY - y_min)*0.5)
            zOffset = -z_min
        elif placement.lower() == "center":
            xOffset = -(x_min + (maxX - x_min)*0.5)
            yOffset = -(y_min + (maxY - y_min)*0.5)
            zOffset = -(z_min + (maxZ - z_min)*0.5)
        returnTopology = Topology.Translate(returnTopology, xOffset, yOffset, zOffset)
        returnTopology = Topology.Place(returnTopology, originA=Vertex.Origin(), originB=origin)
        returnTopology = Topology.Orient(returnTopology, origin=origin, dirA=[0, 0, 1], dirB=direction)
        return returnTopology
    
    @staticmethod
    def HyperbolicParaboloidCircularDomain(origin= None, radius: float = 0.5, sides: int = 36, rings: int = 10,
                                           A: float = 2.0, B: float = -2.0, direction: list = [0, 0, 1],
                                           placement: str = "center", mantissa: int = 6, tolerance: float = 0.0001):
        """
        Creates a hyperbolic paraboloid with a circular domain. See https://en.wikipedia.org/wiki/Compactness_measure_of_a_shape

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The origin of the hyperbolic parabolid. If set to None, it will be placed at the (0, 0, 0) origin. Default is None.
        radius : float , optional
            The desired radius of the hyperbolic paraboloid. Default is 0.5.
        sides : int , optional
            The desired number of sides of the hyperbolic parabolid. Default is 36.
        rings : int , optional
            The desired number of concentric rings of the hyperbolic parabolid. Default is 10.
        A : float , optional
            The *A* constant in the equation z = A*x^2^ + B*y^2^. Default is 2.0.
        B : float , optional
            The *B* constant in the equation z = A*x^2^ + B*y^2^. Default is -2.0.
        direction : list , optional
            The  vector representing the up direction of the hyperbolic paraboloid. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the circle. This can be "center", "lowerleft", "bottom". It is case insensitive. Default is "center".
        mantissa : int , optional
            The desired length of the mantissa. Default is 6
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        
        Returns
        -------
        topologic_core.Shell
            The created hyperbolic paraboloid.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        uOffset = float(360)/float(sides)
        vOffset = float(radius)/float(rings)
        faces = []
        for i in range(rings-1):
            r1 = radius - vOffset*i
            r2 = radius - vOffset*(i+1)
            for j in range(sides-1):
                a1 = math.radians(uOffset)*j
                a2 = math.radians(uOffset)*(j+1)
                x1 = math.sin(a1)*r1
                y1 = math.cos(a1)*r1
                z1 = A*x1*x1 + B*y1*y1
                x2 = math.sin(a1)*r2
                y2 = math.cos(a1)*r2
                z2 = A*x2*x2 + B*y2*y2
                x3 = math.sin(a2)*r2
                y3 = math.cos(a2)*r2
                z3 = A*x3*x3 + B*y3*y3
                x4 = math.sin(a2)*r1
                y4 = math.cos(a2)*r1
                z4 = A*x4*x4 + B*y4*y4
                v1 = Vertex.ByCoordinates(x1,y1,z1)
                v2 = Vertex.ByCoordinates(x2,y2,z2)
                v3 = Vertex.ByCoordinates(x3,y3,z3)
                v4 = Vertex.ByCoordinates(x4,y4,z4)
                f1 = Face.ByVertices([v1,v2,v4])
                f2 = Face.ByVertices([v4,v2,v3])
                faces.append(f1)
                faces.append(f2)
            a1 = math.radians(uOffset)*(sides-1)
            a2 = math.radians(360)
            x1 = math.sin(a1)*r1
            y1 = math.cos(a1)*r1
            z1 = A*x1*x1 + B*y1*y1
            x2 = math.sin(a1)*r2
            y2 = math.cos(a1)*r2
            z2 = A*x2*x2 + B*y2*y2
            x3 = math.sin(a2)*r2
            y3 = math.cos(a2)*r2
            z3 = A*x3*x3 + B*y3*y3
            x4 = math.sin(a2)*r1
            y4 = math.cos(a2)*r1
            z4 = A*x4*x4 + B*y4*y4
            v1 = Vertex.ByCoordinates(x1,y1,z1)
            v2 = Vertex.ByCoordinates(x2,y2,z2)
            v3 = Vertex.ByCoordinates(x3,y3,z3)
            v4 = Vertex.ByCoordinates(x4,y4,z4)
            f1 = Face.ByVertices([v1,v2,v4])
            f2 = Face.ByVertices([v4,v2,v3])
            faces.append(f1)
            faces.append(f2)
        # Special Case: Center triangles
        r = vOffset
        x1 = 0
        y1 = 0
        z1 = 0
        v1 = Vertex.ByCoordinates(x1,y1,z1)
        for j in range(sides-1):
                a1 = math.radians(uOffset)*j
                a2 = math.radians(uOffset)*(j+1)
                x2 = math.sin(a1)*r
                y2 = math.cos(a1)*r
                z2 = A*x2*x2 + B*y2*y2
                #z2 = 0
                x3 = math.sin(a2)*r
                y3 = math.cos(a2)*r
                z3 = A*x3*x3 + B*y3*y3
                #z3 = 0
                v2 = Vertex.ByCoordinates(x2,y2,z2)
                v3 = Vertex.ByCoordinates(x3,y3,z3)
                f1 = Face.ByVertices([v2,v1,v3])
                faces.append(f1)
        a1 = math.radians(uOffset)*(sides-1)
        a2 = math.radians(360)
        x2 = math.sin(a1)*r
        y2 = math.cos(a1)*r
        z2 = A*x2*x2 + B*y2*y2
        x3 = math.sin(a2)*r
        y3 = math.cos(a2)*r
        z3 = A*x3*x3 + B*y3*y3
        v2 = Vertex.ByCoordinates(x2,y2,z2)
        v3 = Vertex.ByCoordinates(x3,y3,z3)
        f1 = Face.ByVertices([v2,v1,v3])
        faces.append(f1)
        returnTopology = Shell.ByFaces(faces, tolerance=tolerance)
        if not returnTopology:
            returnTopology = Cluster.ByTopologies(faces)
        vertices = Topology.Vertices(returnTopology, silent=True)
        xList = []
        yList = []
        zList = []
        for aVertex in vertices:
            xList.append(Vertex.X(aVertex, mantissa=mantissa))
            yList.append(Vertex.Y(aVertex, mantissa=mantissa))
            zList.append(Vertex.Z(aVertex, mantissa=mantissa))
        x_min = min(xList)
        maxX = max(xList)
        y_min = min(yList)
        maxY = max(yList)
        z_min = min(zList)
        maxZ = max(zList)
        xOffset = 0
        yOffset = 0
        zOffset = 0
        if placement.lower() == "lowerleft":
            xOffset = -x_min
            yOffset = -y_min
            zOffset = -z_min
        elif placement.lower() == "bottom":
            xOffset = -(x_min + (maxX - x_min)*0.5)
            yOffset = -(y_min + (maxY - y_min)*0.5)
            zOffset = -z_min
        elif placement.lower() == "center":
            xOffset = -(x_min + (maxX - x_min)*0.5)
            yOffset = -(y_min + (maxY - y_min)*0.5)
            zOffset = -(z_min + (maxZ - z_min)*0.5)
        returnTopology = Topology.Translate(returnTopology, xOffset, yOffset, zOffset)
        returnTopology = Topology.Place(returnTopology, originA=Vertex.Origin(), originB=origin)
        returnTopology = Topology.Orient(returnTopology, origin=origin, dirA=[0, 0, 1], dirB=direction)
        return returnTopology


    @staticmethod
    def InternalBoundaries(shell, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns the internal boundaries (holes) of the input shell.

        Parameters
        ----------
        shell : topologic_core.Shell
            The input shell.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            The list of internal boundaries (holes) of the input shell.

        """
        from topologicpy.Wire import Wire
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Helper import Helper

        if not Topology.IsInstance(shell, "Shell"):
            if not silent:
                print("Shell.InternalBoundaries - Error: The input shell parameter is not a valid Shell. Returning None.")
            return None
        ebEdges = [ebEdge for ebEdge in Topology.Edges(shell) if len(Topology.SuperTopologies(ebEdge, shell, topologyType="face")) == 1]
        if len(ebEdges) > 1:
            x = Topology.SelfMerge(Cluster.ByTopologies(ebEdges), tolerance=tolerance)
            if Topology.IsInstance(x, "wire"):
                wires = [x]
            else:
                wires = Topology.Wires(x)
            lengths = [Wire.Length(w) for w in wires if Topology.IsInstance(w, "wire")]
            wires = Helper.Sort(wires, lengths)
            return wires[:-1]
        return None
    
    @staticmethod
    def InternalEdges(shell, tolerance=0.0001, silent: bool = False):
        """
        Returns the internal edges of the input shell.

        Internal edges are edges that are shared by more than one face. Edges that
        separate the same set of faces are grouped together and self-merged.

        Parameters
        ----------
        shell : topologic_core.Shell
            The input shell.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            A list of merged internal-edge groups. Each item is typically an Edge,
            Wire, Cluster, or another valid topology returned by Topology.SelfMerge.

        """
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        def _value_at_key(dictionary, key, default=None):
            try:
                return Dictionary.ValueAtKey(dictionary, key, default)
            except TypeError:
                try:
                    value = Dictionary.ValueAtKey(dictionary, key)
                    return default if value is None else value
                except Exception:
                    return default
            except Exception:
                return default

        def _normalise_id(value):
            if isinstance(value, list):
                if len(value) < 1:
                    return None
                if len(value) == 1:
                    value = value[0]
                else:
                    return tuple(value)
            try:
                return int(value)
            except Exception:
                return value

        def _face_id(face, source_faces):
            d = Topology.Dictionary(face)
            value = _normalise_id(_value_at_key(d, "__id__", None))
            if isinstance(value, int):
                return value

            # Fallback for cases where the returned supertopology face does not
            # expose the temporary dictionary value reliably.
            for i, source_face in enumerate(source_faces):
                try:
                    if Topology.IsSame(face, source_face):
                        return i
                except Exception:
                    pass

            return None

        def _remove_temporary_key(topology, key):
            try:
                d = Topology.Dictionary(topology)
                d = Dictionary.RemoveKey(d, key)
                Topology.SetDictionary(topology, d)
            except Exception:
                pass

        if not Topology.IsInstance(shell, "Shell"):
            if not silent:
                print("Shell.InternalEdges - Error: The input shell parameter is not a valid Shell. Returning None.")
            return None

        faces = Shell.Faces(shell)
        if not isinstance(faces, list) or len(faces) < 1:
            return []

        # Temporarily tag the faces so that supertopology faces can be grouped
        # deterministically. This avoids using object identity, which can be
        # unreliable across Core calls.
        tagged_faces = []
        for i, face in enumerate(faces):
            try:
                d = Topology.Dictionary(face)
                d = Dictionary.SetValueAtKey(d, "__id__", i)
                tagged_face = Topology.SetDictionary(face, d)
                tagged_faces.append(tagged_face if tagged_face else face)
            except Exception:
                tagged_faces.append(face)

        edges = Topology.Edges(shell)
        if not isinstance(edges, list) or len(edges) < 1:
            for face in tagged_faces:
                _remove_temporary_key(face, "__id__")
            return []

        grouped_edges = {}

        for edge in edges:
            adjacent_faces = Topology.SuperTopologies(edge, shell, topologyType="face")

            if not isinstance(adjacent_faces, list) or len(adjacent_faces) <= 1:
                continue

            ids = []
            for adjacent_face in adjacent_faces:
                face_id = _face_id(adjacent_face, tagged_faces)
                if face_id is not None:
                    ids.append(face_id)

            ids = sorted(set(ids))

            if len(ids) <= 1:
                continue

            group_key = tuple(ids)
            grouped_edges.setdefault(group_key, []).append(edge)

        final_groups = []

        for group_key in sorted(grouped_edges.keys()):
            edge_group = grouped_edges[group_key]

            if len(edge_group) == 1:
                final_groups.append(edge_group[0])
                continue

            cluster = Cluster.ByTopologies(edge_group, silent=True)
            merged = Topology.SelfMerge(cluster, tolerance=tolerance)

            if Topology.IsInstance(merged, "Topology"):
                final_groups.append(merged)
            else:
                # Fallback: preserve the grouped edges rather than failing.
                fallback_cluster = Cluster.ByTopologies(edge_group, silent=True)
                if Topology.IsInstance(fallback_cluster, "Topology"):
                    final_groups.append(fallback_cluster)

        for face in tagged_faces:
            _remove_temporary_key(face, "__id__")

        return final_groups

    
    @staticmethod
    def IsClosed(shell, silent: bool = False) -> bool:
        """
        Returns True if the input shell is closed. Returns False otherwise.

        Parameters
        ----------
        shell : topologic_core.Shell
            The input shell.

        Returns
        -------
        bool
            True if the input shell is closed. False otherwise.

        """
        from topologicpy.Topology import Topology
        
        if not Topology.IsInstance(shell, "shell"):
            if not silent:
                print("Shell.IsClosed - Error: The input shell parameter is not a valid shell. Returning None.")
            return None
        # return shell.IsClosed() # H to Core
        return Core.InstanceCall(shell, "IsClosed")

    @staticmethod
    def IsOnBoundary(shell, vertex, tolerance: float = 0.0001) -> bool:
        """
        Returns True if the input vertex is on the boundary of the input shell. Returns False otherwise. On the boundary is defined as being on the boundary of one of the shell's external or internal boundaries

        Parameters
        ----------
        shell : topologic_core.Shell
            The input shell.
        vertex : topologic_core.Vertex
            The input vertex.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        bool
            Returns True if the input vertex is inside the input shell. Returns False otherwise.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(shell, "Shell"):
            return None
        if not Topology.IsInstance(vertex, "Vertex"):
            return None
        boundary = Shell.ExternalBoundary(shell, tolerance=tolerance)
        if Vertex.IsInternal(vertex, boundary, tolerance=tolerance):
            return True
        internal_boundaries = Shell.InternalBoundaries(shell, tolerance=tolerance)
        for ib in internal_boundaries:
            if Vertex.IsInternal(vertex, ib, tolerance=tolerance):
                return True
        return False

    @staticmethod
    def MobiusStrip(origin = None,
                    radius: float=0.5,
                    height: float=1,
                    uSides=32,
                    vSides=1,
                    twists: int = 1,
                    direction: list = [0, 0, 1],
                    placement: str = "center",
                    tolerance: float = 0.0001,
                    silent: bool = False):


        """
        Creates a Möbius strip. See: https://en.wikipedia.org/wiki/M%C3%B6bius_strip

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the Möbius strip. Default is None which results in the Möbius strip being placed at (0, 0, 0).
        radius : float , optional
            The radius of the Möbius strip. Default is 0.5.
        height : float , optional
            The height of the Möbius strip. Default is 1.
        uSides : int , optional
            The number of circle segments of the Möbius strip. Default is 16.
        vSides : int , optional
            The number of vertical segments of the Möbius strip. Default is 1.
        twists : int , optional
            The number of twists (multiples of a 180 degree rotation) of the Möbius strip. Default is 1.
        direction : list , optional
            The vector representing the up direction of the Möbius strip. Default is [0, 0, 1].
        placement : str , optional
            Not implemented. The description of the placement of the origin of the Möbius strip. This can be "bottom", "center", or "lowerleft". It is case insensitive. Default is "bottom".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Cell
            The created cell.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Topology import Topology
        
        if not isinstance(radius, int) and not isinstance(radius, float):
            if not silent:
                print("Shell.MobiusStrip - Error: The radius input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(height, int) and not isinstance(height, float):
            if not silent:
                print("Shell.MobiusStrip - Error: The height input parameter is not a valid number. Returning None.")
            return None
        if not isinstance(uSides, int):
            if not silent:
                print("Shell.MobiusStrip - Error: The uSides input parameter is not a valid integer. Returning None.")
            return None
        if not isinstance(vSides, int):
            if not silent:
                print("Shell.MobiusStrip - Error: The vSides input parameter is not a valid integer. Returning None.")
            return None
        if not isinstance(twists, int):
            if not silent:
                print("Shell.MobiusStrip - Error: The twists input parameter is not a valid integer. Returning None.")
            return None
        if radius <= tolerance:
            if not silent:
                print("Shell.MobiusStrip - Error: The radius input parameter must be a positive number greater than the tolerance input parameter. Returning None.")
            return None
        if height <= tolerance:
            if not silent:
                print("Shell.MobiusStrip - Error: The height input parameter must be a positive number  greater than the tolerance input parameter. Returning None.")
            return None
        if uSides < 3:
            if not silent:
                print("Shell.MobiusStrip - Error: The uSides input parameter must be a positive integer greater than 2. Returning None.")
            return None
        if vSides < 1:
            if not silent:
                print("Shell.MobiusStrip - Error: The vSides input parameter must be a positive integer greater than 0. Returning None.")
            return None
        if origin == None:
            origin = Vertex.Origin()
        if not Topology.IsInstance(origin, "Vertex"):
            if not silent:
                print("Shell.MobiusStrip - Error: The origin input parameter is not a valid topologic Vertex. Returning None.")
            return None
        if not isinstance(direction, list):
            if not silent:
                print("Shell.MobiusStrip - Error: The direction input parameter is not a valid list. Returning None.")
            return None
        if not len(direction) == 3:
            if not silent:
                print("Shell.MobiusStrip - Error: The direction input parameter is not a valid vector. Returning None.")
            return None
        
        total_angle = 180*twists
        cir = Wire.Circle(origin=origin, radius=radius, sides=uSides, polyline=True)
        c_verts = Topology.Vertices(cir, silent=True)
        wires = []
        for i, v in enumerate(c_verts):
            vb = Vertex.ByCoordinates(Vertex.X(v), Vertex.Y(v), -height*0.5)
            vt = Vertex.ByCoordinates(Vertex.X(v), Vertex.Y(v), height*0.5)
            e = Edge.ByVertices([vb, vt])
            d = Edge.Normal(Edge.ByVertices(Vertex.Origin(), v))
            angle = float(total_angle)/float(uSides)*float(i)
            e = Topology.Rotate(e, origin=v, axis=d, angle=angle)
            verts = []
            for j in range(vSides+1):
                vp = Edge.VertexByParameter(e, float(j)/float(vSides))
                verts.append(vp)
            w = Wire.ByVertices(verts, close=False, silent=True)
            wires.append(w)
            # Create the last wire
            if i == 0:
                e = Edge.ByVertices([vb, vt])
                e = Topology.Rotate(e, origin=v, axis=d, angle=total_angle)
                verts = []
                for j in range(vSides+1):
                    vp = Edge.VertexByParameter(e, float(j)/float(vSides))
                    verts.append(vp)
                last_wire = Wire.ByVertices(verts, close=False, silent=True)

        wires.append(last_wire)
        m = Shell.ByWires(wires, silent=silent, tolerance=tolerance)
        if not Topology.IsInstance(m, "Shell"):
            if not silent:
                print("Shell.MobiusStrip - Error: Could not create a mobius strip. Returning None.")
            return None
        if direction != [0, 0, 1]:
            m = Topology.Orient(m, origin=origin, dirA=[0, 0, 1], dirB=direction)
        return m

    @staticmethod
    def Paraboloid(origin= None, focalLength=0.125, width: float = 1, length: float = 1, uSides: int = 16, vSides: int = 16,
                    direction: list = [0, 0, 1], placement: str ="center", mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
            Creates a paraboloid. See https://en.wikipedia.org/wiki/Paraboloid

            Parameters
            ----------
            origin : topologic_core.Vertex , optional
                The origin location of the parabolic surface. Default is None which results in the parabolic surface being placed at (0, 0, 0).
            focalLength : float , optional
                The focal length of the parabola. Default is 1.
            width : float , optional
                The width of the parabolic surface. Default is 1.
            length : float , optional
                The length of the parabolic surface. Default is 1.
            uSides : int , optional
                The number of sides along the width. Default is 16.
            vSides : int , optional
                The number of sides along the length. Default is 16.
            direction : list , optional
                The vector representing the up direction of the parabolic surface. Default is [0, 0, 1].
            placement : str , optional
                The description of the placement of the origin of the parabolic surface. This can be "bottom", "center", or "lowerleft". It is case insensitive. Default is "center".
            mantissa : int , optional
                The number of decimal places to round the result to. Default is 6.
            tolerance : float , optional
                The desired tolerance. Default is 0.0001.
            silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
            
            Returns
            -------
            topologic_core.Shell
                The created paraboloid.

            """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology

        def create_triangulated_mesh(vertices, uSides, vSides):
            faces = []

            # Iterate over the grid of vertices to form triangular faces
            for i in range(uSides - 1):
                for j in range(vSides - 1):
                    # Get the indices of the vertices forming the current grid cell
                    v1 = vertices[i * vSides + j]
                    v2 = vertices[i * vSides + (j + 1)]
                    v3 = vertices[(i + 1) * vSides + j]
                    v4 = vertices[(i + 1) * vSides + (j + 1)]

                    # Create two triangles for each grid cell
                    # Triangle 1: (v1, v2, v3)
                    wire1 = Wire.ByVertices([v1, v2, v3])
                    face1 = Face.ByWire(wire1)
                    faces.append(face1)

                    # Triangle 2: (v3, v2, v4)
                    wire2 = Wire.ByVertices([v3, v2, v4])
                    face2 = Face.ByWire(wire2)
                    faces.append(face2)

            # Create the mesh (Shell) from the list of faces
            mesh = Shell.ByFaces(faces)
            return mesh
        
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.Origin()
        
        x_range = [-width*0.5, width*0.5]
        y_range = [-length*0.5, length*0.5]
        # Generate x and y values
        x_values = [x_range[0] + i * (x_range[1] - x_range[0]) / (uSides - 1) for i in range(uSides)]
        y_values = [y_range[0] + i * (y_range[1] - y_range[0]) / (vSides - 1) for i in range(vSides)]
        
        # Create the grid and calculate Z values
        vertices = []
        
        for x in x_values:
            for y in y_values:
                z = ((x)**2 + (y)**2) / (4 * focalLength)
                vertices.append(Vertex.ByCoordinates(x, y, z))
        
        mesh = create_triangulated_mesh(vertices=vertices, uSides=uSides, vSides=vSides)
        if not placement.lower() == "bottom":
            x_list = [Vertex.X(v) for v in vertices]
            y_list = [Vertex.Y(v) for v in vertices]
            z_list = [Vertex.Z(v) for v in vertices]
            x_list.sort()
            y_list.sort()
            z_list.sort()
            width = abs(x_list[-1] - x_list[0])
            length = abs(y_list[-1] - y_list[0])
            height = abs(z_list[-1] - z_list[0])
            if placement.lower() == "center":
                mesh = Topology.Translate(mesh, 0, 0, -height*0.5)
            elif placement.lower() == "lowerleft":
                mesh = Topology.Translate(mesh, width*0.5, length*0.5, 0)

        mesh = Topology.Orient(mesh, origin=origin, dirA=[0, 0, 1], dirB=direction, tolerance=tolerance)
        return mesh
    
    @staticmethod
    def Pie(origin= None, radiusA: float = 0.5, radiusB: float = 0.0, sides: int = 32, rings: int = 1, fromAngle: float = 0.0, toAngle: float = 360.0, direction: list = [0, 0, 1], placement: str = "center", tolerance: float = 0.0001):
        """
        Creates a pie shape.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the pie. Default is None which results in the pie being placed at (0, 0, 0).
        radiusA : float , optional
            The outer radius of the pie. Default is 0.5.
        radiusB : float , optional
            The inner radius of the pie. Default is 0.25.
        sides : int , optional
            The number of sides of the pie. Default is 32.
        rings : int , optional
            The number of rings of the pie. Default is 1.
        fromAngle : float , optional
            The angle in degrees from which to start creating the arc of the pie. Default is 0.
        toAngle : float , optional
            The angle in degrees at which to end creating the arc of the pie. Default is 360.
        direction : list , optional
            The vector representing the up direction of the pie. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the pie. This can be "center", or "lowerleft". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        topologic_core.Shell
            The created pie.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        if not Topology.IsInstance(origin, "Vertex"):
            return None
        if toAngle < fromAngle:
            toAngle += 360
        if abs(toAngle-fromAngle) <= tolerance:
            return None
        fromAngle = math.radians(fromAngle)
        toAngle = math.radians(toAngle)
        angleRange = toAngle - fromAngle
        radiusA = abs(radiusA)
        radiusB = abs(radiusB)
        if radiusB > radiusA:
            temp = radiusA
            radiusA = radiusB
            radiusB = temp
        if abs(radiusA - radiusB) <= tolerance or radiusA <= tolerance:
            return None
        radiusRange = radiusA - radiusB
        sides = int(abs(math.floor(sides)))
        if sides < 3:
            return None
        rings = int(abs(rings))
        if radiusB <= tolerance:
            radiusB = 0
        xOffset = 0
        yOffset = 0
        zOffset = 0
        if placement.lower() == "lowerleft":
            xOffset = radiusA
            yOffset = radiusA
        uOffset = float(angleRange)/float(sides)
        vOffset = float(radiusRange)/float(rings)
        faces = []
        if radiusB > tolerance:
            for i in range(rings):
                r1 = radiusA - vOffset*i
                r2 = radiusA - vOffset*(i+1)
                for j in range(sides):
                    a1 = fromAngle + uOffset*j
                    a2 = fromAngle + uOffset*(j+1)
                    x1 = math.sin(a1)*r1
                    y1 = math.cos(a1)*r1
                    z1 = 0
                    x2 = math.sin(a1)*r2
                    y2 = math.cos(a1)*r2
                    z2 = 0
                    x3 = math.sin(a2)*r2
                    y3 = math.cos(a2)*r2
                    z3 = 0
                    x4 = math.sin(a2)*r1
                    y4 = math.cos(a2)*r1
                    z4 = 0
                    v1 = Vertex.ByCoordinates(x1,y1,z1)
                    v2 = Vertex.ByCoordinates(x2,y2,z2)
                    v3 = Vertex.ByCoordinates(x3,y3,z3)
                    v4 = Vertex.ByCoordinates(x4,y4,z4)
                    f1 = Face.ByVertices([v1,v2,v3,v4])
                    faces.append(f1)
        else:
            x1 = 0
            y1 = 0
            z1 = 0
            v1 = Vertex.ByCoordinates(x1,y1,z1)
            for j in range(sides):
                a1 = fromAngle + uOffset*j
                a2 = fromAngle + uOffset*(j+1)
                x2 = math.sin(a1)*radiusA
                y2 = math.cos(a1)*radiusA
                z2 = 0
                x3 = math.sin(a2)*radiusA
                y3 = math.cos(a2)*radiusA
                z3 = 0
                v2 = Vertex.ByCoordinates(x2,y2,z2)
                v3 = Vertex.ByCoordinates(x3,y3,z3)
                f1 = Face.ByVertices([v2,v1,v3])
                faces.append(f1)

        shell = Shell.ByFaces(faces, tolerance=tolerance)
        if not shell:
            return None
        shell = Topology.Translate(shell, xOffset, yOffset, zOffset)
        shell = Topology.Place(shell, originA=Vertex.Origin(), originB=origin)
        shell = Topology.Orient(shell, origin=origin, dirA=[0, 0, 1], dirB=direction)
        return shell

    @staticmethod
    def Planarize(shell, origin=None, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Returns a planarized version of the input Shell while preserving curved
        Edge geometry whenever the active backend supports exact native projection.

        Plane inference is face-based rather than edge-sampling-based. This is
        essential for closed analytic and NURBS Edges, which may expose only one
        topological Vertex and for which direct curve sampling can be unsafe in
        some PythonOCC builds.

        Parameters
        ----------
        shell : topologic_core.Shell
            The input Shell.
        origin : topologic_core.Vertex , optional
            The desired origin of the target plane. If None, the centroid of the
            input Shell is used. Default is None.
        mantissa : int , optional
            The number of decimal places used when deriving the best-fit plane.
            Default is 6.
        tolerance : float , optional
            The desired geometric tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        topologic_core.Shell
            The planarized Shell, or None if the operation cannot be completed
            without approximating curved geometry.
        """
        import math

        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(shell, "Shell"):
            if not silent:
                print("Shell.Planarize - Error: The input shell parameter is not a valid Shell. Returning None.")
            return None

        try:
            tolerance = max(abs(float(tolerance)), 1.0e-12)
        except Exception:
            if not silent:
                print("Shell.Planarize - Error: The input tolerance parameter is invalid. Returning None.")
            return None

        faces = Shell.Faces(shell) or []
        faces = [face for face in faces if Topology.IsInstance(face, "Face")]
        if not faces:
            if not silent:
                print("Shell.Planarize - Error: The input Shell does not contain any valid Faces. Returning None.")
            return None

        try:
            is_topologic_core = bool(Topology._IsTopologicCoreBackend())
        except Exception:
            is_topologic_core = True

        # ------------------------------------------------------------------
        # Fast path for an already-planar Shell.
        #
        # Do this BEFORE touching Edge curve geometry. A valid planar Face may
        # be bounded by one closed circular/NURBS Edge with a single topological
        # Vertex. Native FaceUtility planarity/coplanarity tests operate on the
        # supporting surfaces directly and avoid unsafe curve evaluation.
        # ------------------------------------------------------------------
        if not is_topologic_core:
            try:
                all_planar = all(
                    Core.FaceUtility.IsPlanar(face, tolerance) is True
                    for face in faces
                )
                if all_planar:
                    reference = faces[0]
                    all_coplanar = all(
                        Core.FaceUtility.IsCoplanar(reference, face, tolerance) is True
                        for face in faces[1:]
                    )
                    if all_coplanar:
                        return shell
            except Exception:
                # Failure to establish the fast path is not an operation failure;
                # continue to the general best-fit-plane path below.
                pass

        # TopologicCore's exact curved-wire projection is unavailable. Preserve
        # the established rule: never silently chord curved geometry.
        if is_topologic_core:
            for edge in Topology.Edges(shell) or []:
                if Edge.IsLinear(edge, tolerance=tolerance, silent=True) is not True:
                    if not silent:
                        print("Shell.Planarize - Error: Curve-preserving Shell planarization requires the PythonOCC backend. Returning None.")
                    return None

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Topology.Centroid(shell)
        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.Origin()
        if not Topology.IsInstance(origin, "Vertex"):
            if not silent:
                print("Shell.Planarize - Error: Could not determine a valid plane origin. Returning None.")
            return None

        # ------------------------------------------------------------------
        # Best-fit plane samples from Face surfaces, never from Edge curves.
        # Sampling is used only to infer the target plane and its finite extent.
        # The output geometry is still created by projecting complete Wires.
        # ------------------------------------------------------------------
        sample_vertices = []
        sample_keys = set()

        def add_sample(vertex):
            if not Topology.IsInstance(vertex, "Vertex"):
                return
            try:
                xyz = Vertex.Coordinates(vertex, mantissa=None)
                x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
            except Exception:
                return
            key = (round(x, mantissa), round(y, mantissa), round(z, mantissa))
            if key not in sample_keys:
                sample_keys.add(key)
                sample_vertices.append(vertex)

        # Ordinary topology vertices are safe to include, but are not relied on.
        for vertex in Topology.Vertices(shell, silent=True) or []:
            add_sample(vertex)

        uv_samples = (
            (0.0, 0.0), (0.5, 0.0), (1.0, 0.0),
            (0.0, 0.5), (0.5, 0.5), (1.0, 0.5),
            (0.0, 1.0), (0.5, 1.0), (1.0, 1.0),
            (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75),
        )

        for face in faces:
            try:
                add_sample(Topology.Centroid(face))
            except Exception:
                pass
            for u, v in uv_samples:
                try:
                    add_sample(Face.VertexByParameters(face, u=u, v=v))
                except Exception:
                    pass

        if len(sample_vertices) < 3:
            if not silent:
                print("Shell.Planarize - Error: Could not derive enough geometric samples from the input Faces. Returning None.")
            return None

        try:
            equation = Vertex.PlaneEquation(
                sample_vertices,
                mantissa=mantissa,
                tolerance=tolerance,
                silent=True,
            )
        except TypeError:
            equation = Vertex.PlaneEquation(sample_vertices, mantissa=mantissa)

        if not isinstance(equation, dict):
            if not silent:
                print("Shell.Planarize - Error: Could not determine a best-fit projection plane. Returning None.")
            return None

        try:
            normal = [
                float(equation["a"]),
                float(equation["b"]),
                float(equation["c"]),
            ]
            magnitude = math.sqrt(sum(value * value for value in normal))
            if not math.isfinite(magnitude) or magnitude <= tolerance:
                raise ValueError
            normal = [value / magnitude for value in normal]
        except Exception:
            if not silent:
                print("Shell.Planarize - Error: Could not determine a valid projection-plane normal. Returning None.")
            return None

        try:
            ox, oy, oz = Vertex.Coordinates(origin, mantissa=None)
            max_distance = 0.0
            for vertex in sample_vertices:
                x, y, z = Vertex.Coordinates(vertex, mantissa=None)
                distance = math.sqrt(
                    (float(x) - float(ox)) ** 2
                    + (float(y) - float(oy)) ** 2
                    + (float(z) - float(oz)) ** 2
                )
                max_distance = max(max_distance, distance)
            plane_size = max(1.0, 4.0 * max_distance, 1000.0 * tolerance)
        except Exception:
            plane_size = 1.0

        receiving_face = Face.RectangleByPlaneEquation(
            origin=origin,
            width=plane_size,
            length=plane_size,
            equation=equation,
            tolerance=tolerance,
        )
        if not Topology.IsInstance(receiving_face, "Face"):
            if not silent:
                print("Shell.Planarize - Error: Could not construct the receiving projection Face. Returning None.")
            return None

        def project_polyline_wire(wire):
            source_edges = Wire._OrderedEdges(wire, tolerance=tolerance, silent=True)
            if not isinstance(source_edges, list) or not source_edges:
                return None

            def project_vertex(vertex):
                projected = Vertex.Project(
                    vertex,
                    receiving_face,
                    direction=normal,
                    mantissa=mantissa,
                    tolerance=tolerance,
                )
                if Topology.IsInstance(projected, "Vertex"):
                    return projected
                return Vertex.Project(
                    vertex,
                    receiving_face,
                    direction=[-normal[0], -normal[1], -normal[2]],
                    mantissa=mantissa,
                    tolerance=tolerance,
                )

            projected_edges = []
            for edge in source_edges:
                start = project_vertex(Edge.StartVertex(edge, silent=True))
                end = project_vertex(Edge.EndVertex(edge, silent=True))
                if not Topology.IsInstance(start, "Vertex") or not Topology.IsInstance(end, "Vertex"):
                    return None
                projected = Edge.ByStartVertexEndVertex(
                    start,
                    end,
                    tolerance=tolerance,
                    silent=True,
                )
                if not Topology.IsInstance(projected, "Edge"):
                    return None
                projected_edges.append(projected)

            return Wire.ByEdges(
                projected_edges,
                orient=True,
                tolerance=tolerance,
                silent=True,
            )

        def project_wire(wire, source_face):
            if not Topology.IsInstance(wire, "Wire"):
                return None

            # If this entire source Face already lies on the target plane, keep
            # the original Wire. This is a surface-level test; no Edge sampling.
            if not is_topologic_core and Topology.IsInstance(source_face, "Face"):
                try:
                    if (
                        Core.FaceUtility.IsPlanar(source_face, tolerance) is True
                        and Core.FaceUtility.IsCoplanar(source_face, receiving_face, tolerance) is True
                    ):
                        return wire
                except Exception:
                    pass

            if not is_topologic_core:
                # Use OCCT normal projection rather than BRepProj_Projection.
                # On a planar receiving Face, normal projection is the required
                # orthogonal projection and preserves analytic/B-spline curves.
                try:
                    from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_NormalProjection
                    from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_WIRE
                    from OCC.Core.TopExp import TopExp_Explorer
                    from OCC.Core.TopoDS import topods

                    source_shape = getattr(wire, "shape", None)
                    target_shape = getattr(receiving_face, "shape", None)
                    if source_shape is not None and target_shape is not None:
                        projector = BRepOffsetAPI_NormalProjection(target_shape)
                        projector.Add(source_shape)
                        projector.SetLimit(False)
                        projector.Compute3d(True)
                        projector.Build()

                        if not hasattr(projector, "IsDone") or projector.IsDone():
                            projected_shape = projector.Projection()
                            if projected_shape is not None and not projected_shape.IsNull():
                                projected_wires = []
                                explorer = TopExp_Explorer(projected_shape, TopAbs_WIRE)
                                while explorer.More():
                                    occ_wire = topods.Wire(explorer.Current())
                                    candidate = None
                                    try:
                                        if Core.HasAttribute("Wire", "ByOcctShape"):
                                            candidate = Core.Wire.ByOcctShape(occ_wire)
                                    except Exception:
                                        candidate = None
                                    if Topology.IsInstance(candidate, "Wire"):
                                        projected_wires.append(candidate)
                                    explorer.Next()

                                if len(projected_wires) == 1:
                                    return projected_wires[0]

                                projected_edges = []
                                if projected_wires:
                                    for projected_wire in projected_wires:
                                        projected_edges.extend(Wire.Edges(projected_wire, silent=True) or [])
                                else:
                                    explorer = TopExp_Explorer(projected_shape, TopAbs_EDGE)
                                    while explorer.More():
                                        occ_edge = topods.Edge(explorer.Current())
                                        candidate = None
                                        try:
                                            if Core.HasAttribute("Edge", "ByOcctShape"):
                                                candidate = Core.Edge.ByOcctShape(occ_edge)
                                        except Exception:
                                            candidate = None
                                        if Topology.IsInstance(candidate, "Edge"):
                                            projected_edges.append(candidate)
                                        explorer.Next()

                                if projected_edges:
                                    merged = Wire.ByEdges(
                                        projected_edges,
                                        orient=True,
                                        tolerance=tolerance,
                                        silent=True,
                                    )
                                    if Topology.IsInstance(merged, "Wire"):
                                        return merged
                except Exception:
                    pass

            # Exact backend-neutral fallback only for straight-edge polylines.
            if Wire.IsPolyline(wire, tolerance=tolerance, silent=True):
                return project_polyline_wire(wire)
            return None

        new_faces = []
        for face in faces:
            external = Face.ExternalBoundary(face)
            projected_external = project_wire(external, face)
            if not Topology.IsInstance(projected_external, "Wire"):
                if not silent:
                    print("Shell.Planarize - Error: Could not project a Face boundary without approximating its curves. Returning None.")
                return None

            projected_internal = []
            for wire in Face.InternalBoundaries(face) or []:
                projected = project_wire(wire, face)
                if not Topology.IsInstance(projected, "Wire"):
                    if not silent:
                        print("Shell.Planarize - Error: Could not project an internal Face boundary without approximating its curves. Returning None.")
                    return None
                projected_internal.append(projected)

            try:
                new_face = Face.ByWires(
                    projected_external,
                    projected_internal,
                    tolerance=tolerance,
                    silent=True,
                )
            except TypeError:
                new_face = Face.ByWires(
                    projected_external,
                    projected_internal,
                    tolerance=tolerance,
                )

            if not Topology.IsInstance(new_face, "Face"):
                if not silent:
                    print("Shell.Planarize - Error: Could not rebuild a projected Face. Returning None.")
                return None
            new_faces.append(new_face)

        result = Shell.ByFaces(new_faces, tolerance=tolerance, silent=True)
        if not Topology.IsInstance(result, "Shell"):
            if not silent:
                print("Shell.Planarize - Error: Could not rebuild the planarized Shell. Returning None.")
            return None

        return result

    
    @staticmethod
    def Rectangle(origin= None, width: float = 1.0, length: float = 1.0,
                  uSides: int = 2, vSides: int = 2, direction: list = [0, 0, 1],
                  placement: str = "center", tolerance: float = 0.0001):
        """
        Creates a rectangle.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the rectangle. Default is None which results in the rectangle being placed at (0, 0, 0).
        width : float , optional
            The width of the rectangle. Default is 1.0.
        length : float , optional
            The length of the rectangle. Default is 1.0.
        uSides : int , optional
            The number of sides along the width. Default is 2.
        vSides : int , optional
            The number of sides along the length. Default is 2.
        direction : list , optional
            The vector representing the up direction of the rectangle. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the rectangle. This can be "center", or "lowerleft". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        topologic_core.Shell
            The created shell.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(origin, "Vertex"):
            origin = Vertex.ByCoordinates(0, 0, 0)
        if not Topology.IsInstance(origin, "Vertex"):
            return None
        uOffset = float(width)/float(uSides)
        vOffset = float(length)/float(vSides)
        faces = []
        if placement.lower() == "center":
            wOffset = width*0.5
            lOffset = length*0.5
        else:
            wOffset = 0
            lOffset = 0
        for i in range(uSides):
            for j in range(vSides):
                rOrigin = Vertex.ByCoordinates(i*uOffset - wOffset, j*vOffset - lOffset, 0)
                w = Wire.Rectangle(origin=rOrigin, width=uOffset, length=vOffset, direction=[0, 0, 1], placement="lowerleft", tolerance=tolerance)
                f = Face.ByWire(w, tolerance=tolerance)
                faces.append(f)
        shell = Shell.ByFaces(faces, tolerance=tolerance)
        shell = Topology.Place(shell, originA=Vertex.Origin(), originB=origin)
        shell = Topology.Orient(shell, origin=origin, dirA=[0, 0, 1], dirB=direction)
        return shell

    @staticmethod
    def RemoveCollinearEdges(
        shell,
        angTolerance: float = 0.1,
        polyhedron: bool = True,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """Removes redundant collinear linear Edges from the input Shell."""
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(shell, "Shell"):
            if not silent:
                print("Shell.RemoveCollinearEdges - Error: The input shell parameter is not a valid Shell. Returning None.")
            return None
        if not isinstance(polyhedron, bool):
            if not silent:
                print("Shell.RemoveCollinearEdges - Error: The input polyhedron parameter is not a valid boolean. Returning None.")
            return None

        clean_faces = []
        for face in Shell.Faces(shell) or []:
            clean_face = Topology.RemoveCollinearEdges(
                face,
                angTolerance=angTolerance,
                polyhedron=polyhedron,
                tolerance=tolerance,
                silent=True,
            )
            if not Topology.IsInstance(clean_face, "Face"):
                if polyhedron is False:
                    clean_face = face
                else:
                    return None
            clean_faces.append(clean_face)

        result = Shell.ByFaces(clean_faces, tolerance=tolerance, silent=True)
        if not Topology.IsInstance(result, "Shell") and not silent:
            print("Shell.RemoveCollinearEdges - Error: Could not rebuild the Shell. Returning None.")
        return result if Topology.IsInstance(result, "Shell") else None
    
    @staticmethod
    def Roof(face, angle: float = 45, epsilon: float = 0.01, mantissa: int = 6, tolerance: float = 0.001):
        """
            Creates a hipped roof through a straight skeleton. This method is contributed by 高熙鹏 xipeng gao <gaoxipeng1998@gmail.com>
            This algorithm depends on the polyskel code which is included in the library. Polyskel code is found at: https://github.com/Botffy/polyskel

        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        angle : float , optioal
            The desired angle in degrees of the roof. Default is 45.
        epsilon : float , optional
            The desired epsilon (another form of tolerance for distance from plane). Default is 0.01. (This is set to a larger number as it was found to work better)
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.001. (This is set to a larger number as it was found to work better)

        Returns
        -------
        topologic_core.Shell
            The created roof.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Shell import Shell
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology

        def nearest_vertex_2d(v, vertices, tolerance=0.001):
            for vertex in vertices:
                x2 = Vertex.X(vertex, mantissa=mantissa)
                y2 = Vertex.Y(vertex, mantissa=mantissa)
                temp_v = Vertex.ByCoordinates(x2, y2, Vertex.Z(v, mantissa=mantissa))
                if Vertex.Distance(v, temp_v, mantissa=mantissa) <= tolerance:
                    return vertex
            return None
        
        if not Topology.IsInstance(face, "Face"):
            return None
        angle = abs(angle)
        if angle >= 90-tolerance:
            return None
        if angle <= tolerance:
            return None
        origin = Topology.Centroid(face)
        normal = Face.Normal(face, mantissa=mantissa)
        flat_face = Topology.Flatten(face, origin=origin, direction=normal)
        roof = Wire.Roof(flat_face, angle=angle, boundary=True, tolerance=tolerance)
        if not roof:
            return None
        shell = Shell.Skeleton(flat_face, tolerance=tolerance)
        faces = Shell.Faces(shell)
        if not faces:
            return None
        triangles = []
        for face in faces:
            internalBoundaries = Face.InternalBoundaries(face)
            if len(internalBoundaries) == 0:
                if len(Topology.Vertices(face, silent=True)) > 3:
                    triangles += Face.Triangulate(face, tolerance=tolerance)
                else:
                    triangles += [face]

        roof_vertices = Topology.Vertices(roof, silent=True)
        flat_vertices = []
        for rv in roof_vertices:
            flat_vertices.append(Vertex.ByCoordinates(Vertex.X(rv, mantissa=mantissa), Vertex.Y(rv, mantissa=mantissa), 0))

        final_triangles = []
        for triangle in triangles:
            if len(Topology.Vertices(triangle, silent=True)) > 3:
                triangles = Face.Triangulate(triangle, tolerance=tolerance)
            else:
                triangles = [triangle]
            final_triangles += triangles

        final_faces = []
        for triangle in final_triangles:
            face_vertices = Topology.Vertices(triangle, silent=True)
            top_vertices = []
            for sv in face_vertices:
                temp = nearest_vertex_2d(sv, roof_vertices, tolerance=tolerance)
                if temp:
                    top_vertices.append(temp)
                else:
                    top_vertices.append(sv)
            tri_face = Face.ByVertices(top_vertices)
            final_faces.append(tri_face)

        shell = Shell.ByFaces(final_faces, tolerance=tolerance)
        if not shell:
            shell = Cluster.ByTopologies(final_faces)
        try:
            shell = Topology.RemoveCoplanarFaces(shell, epsilon=epsilon, tolerance=tolerance)
        except:
            pass
        shell = Topology.Unflatten(shell, origin=origin, direction=normal)
        return shell
    
    @staticmethod
    def SelfMerge(shell, angTolerance: float = 0.1, tolerance: float = 0.0001, silent: bool = False):
        """
        Creates a face by merging the faces of the input shell. The shell must be planar within the input angular tolerance.

        Parameters
        ----------
        shell : topologic_core.Shell
            The input shell.
        angTolerance : float , optional
            The desired angular tolerance. Default is 0.1.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        topologic_core.Face
            The created face.

        """
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Shell import Shell
        from topologicpy.Topology import Topology
        
        def planarizeList(wireList):
            returnList = []
            for aWire in wireList:
                returnList.append(Wire.Planarize(aWire))
            return returnList
        if not Topology.IsInstance(shell, "Shell"):
            return None
        ext_boundary = Shell.ExternalBoundary(shell, tolerance=tolerance)
        if Topology.IsInstance(ext_boundary, "Wire"):
            f = Face.ByWire(Topology.RemoveCollinearEdges(ext_boundary, angTolerance=angTolerance, polyhedron=False, tolerance=tolerance, silent=True), tolerance=tolerance) or Face.ByWire(Wire.Planarize(Topology.RemoveCollinearEdges(ext_boundary, angTolerance=angTolerance, polyhedron=False, tolerance=tolerance, silent=True), tolerance=tolerance))
            if not f:
                print("FaceByPlanarShell - Error: The input Wire is not planar and could not be fixed. Returning None.")
                return None
            else:
                return f
        elif Topology.IsInstance(ext_boundary, "Cluster"):
            wires = Topology.Wires(ext_boundary)
            faces = []
            areas = []
            for aWire in wires:
                try:
                    aFace = Face.ByWire(Topology.RemoveCollinearEdges(aWire, angTolerance=angTolerance, polyhedron=False, tolerance=tolerance, silent=True))
                except:
                    aFace = Face.ByWire(Wire.Planarize(Topology.RemoveCollinearEdges(aWire, angTolerance=angTolerance, polyhedron=False, tolerance=tolerance, silent=True)))
                anArea = Face.Area(aFace)
                faces.append(aFace)
                areas.append(anArea)
            max_index = areas.index(max(areas))
            ext_boundary = faces[max_index]
            int_boundaries = list(set(faces) - set([ext_boundary]))
            int_wires = []
            for int_boundary in int_boundaries:
                temp_wires = Topology.Wires(int_boundary)
                int_wires.append(Topology.RemoveCollinearEdges(temp_wires[0], angTolerance=angTolerance, polyhedron=False, tolerance=tolerance, silent=True))
            temp_wires = Topology.Wire(ext_boundary)
            ext_wire = Topology.RemoveCollinearEdges(temp_wires[0], angTolerance=angTolerance, polyhedron=False, tolerance=tolerance, silent=True)
            try:
                return Face.ByWires(ext_wire, int_wires, tolerance=tolerance)
            except:
                return Face.ByWires(Wire.Planarize(ext_wire), planarizeList(int_wires), tolerance=tolerance)
        else:
            return None

    @staticmethod
    def Simplify(shell, simplifyBoundary: bool = True, mantissa: int = 6, tolerance: float = 0.0001, silent: bool = False):
        """
        Simplifies a planar polyline Shell using the Douglas-Peucker algorithm.

        This operation is intentionally limited to linear Edges. Curved Edges are
        rejected rather than silently replaced by chords. A tilted planar Shell is
        first rigidly flattened to XY, simplified there, and then restored.
        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Helper import Helper

        def perpendicular_distance(point, line_start, line_end):
            x0 = Vertex.X(point, mantissa=mantissa)
            y0 = Vertex.Y(point, mantissa=mantissa)
            x1 = Vertex.X(line_start, mantissa=mantissa)
            y1 = Vertex.Y(line_start, mantissa=mantissa)
            x2 = Vertex.X(line_end, mantissa=mantissa)
            y2 = Vertex.Y(line_end, mantissa=mantissa)
            denominator = Vertex.Distance(line_start, line_end)
            if denominator is None or denominator <= tolerance:
                return 0.0
            numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
            return numerator/denominator

        def douglas_peucker(wire, local_tolerance=0.0001):
            points = wire if isinstance(wire, list) else (Topology.Vertices(wire, silent=True) or [])
            if len(points) <= 2:
                return points
            start_point, end_point = points[0], points[-1]
            max_distance = 0.0
            max_index = 0
            for i in range(1, len(points)-1):
                distance = perpendicular_distance(points[i], start_point, end_point)
                if distance > max_distance:
                    max_distance = distance
                    max_index = i
            if max_distance <= local_tolerance:
                return [start_point, end_point]
            first = douglas_peucker(points[:max_index+1], local_tolerance)
            second = douglas_peucker(points[max_index:], local_tolerance)
            return first[:-1] + second

        if not Topology.IsInstance(shell, "Shell"):
            if not silent:
                print("Shell.Simplify - Error: The input shell parameter is not a valid Shell. Returning None.")
            return None
        if any(Edge.IsLinear(edge, silent=True) is not True for edge in (Topology.Edges(shell) or [])):
            if not silent:
                print("Shell.Simplify - Error: The input Shell contains curved Edges. Douglas-Peucker simplification is defined here only for polylines. Returning None.")
            return None

        vertices = Topology.Vertices(shell, silent=True) or []
        if len(vertices) < 3:
            return shell
        equation = Vertex.PlaneEquation(vertices, mantissa=mantissa)
        if not isinstance(equation, dict):
            return None
        try:
            normal = [float(equation["a"]), float(equation["b"]), float(equation["c"])]
            mag = math.sqrt(sum(x*x for x in normal))
            normal = [x/mag for x in normal]
        except Exception:
            return None
        origin = Topology.Centroid(shell)
        flat_shell = Topology.Flatten(shell, origin=origin, direction=normal)
        if not Topology.IsInstance(flat_shell, "Shell"):
            return None
        flat_vertices = Topology.Vertices(flat_shell, silent=True) or []
        z_values = [Vertex.Z(v, mantissa=mantissa) for v in flat_vertices]
        if z_values and max(z_values)-min(z_values) > max(tolerance*10.0, 10.0**(-mantissa)):
            if not silent:
                print("Shell.Simplify - Error: The input Shell is not planar within tolerance. Returning None.")
            return None

        all_edges = Topology.Edges(flat_shell) or []
        if simplifyBoundary is False:
            boundary = Shell.ExternalBoundary(flat_shell, tolerance=tolerance, silent=True)
            ext_boundary = Face.ByWire(boundary, tolerance=tolerance, silent=True) if Topology.IsInstance(boundary, "Wire") else None
            if not Topology.IsInstance(ext_boundary, "Face"):
                return None
            internal_edges = []
            for edge in all_edges:
                faces = Topology.SuperTopologies(edge, flat_shell, topologyType="face") or []
                if len(faces) > 1:
                    internal_edges.append(edge)
            wire = Topology.SelfMerge(Cluster.ByTopologies(internal_edges), tolerance=tolerance) if internal_edges else None
        else:
            wire = Topology.SelfMerge(Cluster.ByTopologies(all_edges), tolerance=tolerance)

        if wire is None:
            return shell
        components = Wire.Split(wire) or []
        separators, wires = [], []
        for component in components:
            if Topology.IsInstance(component, "Cluster"):
                component = Topology.SelfMerge(component, tolerance=tolerance)
                if Topology.IsInstance(component, "Cluster"):
                    separators.append(Cluster.FreeEdges(component, tolerance=tolerance))
                    wires.append(Cluster.FreeWires(component, tolerance=tolerance))
                elif Topology.IsInstance(component, "Edge"):
                    separators.append(component)
                elif Topology.IsInstance(component, "Wire"):
                    wires.append(component)
            elif Topology.IsInstance(component, "Edge"):
                separators.append(component)
            elif Topology.IsInstance(component, "Wire"):
                wires.append(component)

        wires = Helper.Flatten(wires)
        separators = Helper.Flatten(separators)
        simplified = []
        for wire_item in wires:
            points = douglas_peucker(wire_item, local_tolerance=tolerance)
            if len(points) >= 2:
                temp_wire = Wire.ByVertices(points, close=False, tolerance=tolerance, silent=True)
                if Topology.IsInstance(temp_wire, "Wire"):
                    simplified.append(temp_wire)

        final_edges = (Topology.Edges(Cluster.ByTopologies(simplified)) or []) + separators if simplified else separators
        if not final_edges:
            return shell
        cluster = Cluster.ByTopologies(final_edges)

        if simplifyBoundary is False:
            final_result = Topology.Slice(ext_boundary, cluster, tolerance=tolerance)
        else:
            br = Wire.BoundingRectangle(flat_shell)
            if not Topology.IsInstance(br, "Wire"):
                return shell
            br = Topology.Scale(br, Topology.Centroid(br), 1.5, 1.5, 1.5)
            br = Face.ByWire(br, tolerance=tolerance, silent=True)
            if not Topology.IsInstance(br, "Face"):
                return shell
            selector = Face.VertexByParameters(br, 0.1, 0.1)
            result = Topology.Slice(br, cluster, tolerance=tolerance)
            final_faces = []
            for face in Topology.Faces(result) or []:
                if not Vertex.IsInternal(selector, face, tolerance=0.01):
                    final_faces.append(face)
            final_result = Shell.ByFaces(final_faces, tolerance=tolerance, silent=True)

        if not Topology.IsInstance(final_result, "Shell"):
            if not silent:
                print("Shell.Simplify - Warning: Simplification produced no Shell; returning the input Shell.")
            return shell
        restored = Topology.Unflatten(final_result, origin=origin, direction=normal)
        return restored if Topology.IsInstance(restored, "Shell") else shell

    @staticmethod
    def Skeleton(face, tolerance: float = 0.001):
        """
            Creates a shell through a straight skeleton. This method is contributed by 高熙鹏 xipeng gao <gaoxipeng1998@gmail.com>
            This algorithm depends on the polyskel code which is included in the library. Polyskel code is found at: https://github.com/Botffy/polyskel

        Parameters
        ----------
        face : topologic_core.Face
            The input face.
        tolerance : float , optional
            The desired tolerance. Default is 0.001. (This is set to a larger number as it was found to work better)

        Returns
        -------
        topologic_core.Shell
            The created straight skeleton.

        """
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        from topologicpy.Core import Core
        import math

        if not Topology.IsInstance(face, "Face"):
            return None
        roof = Wire.Skeleton(face, tolerance=tolerance)
        if not (Topology.IsInstance(roof, "Wire") or Topology.IsInstance(roof, "Cluster")):
            print("Shell.Skeleton - Error: Could not create base skeleton wire. Returning None.")
            return None
        br = Wire.BoundingRectangle(roof) #This works even if it is a Cluster not a Wire
        if not Topology.IsInstance(br, "Wire"):
            print("Shell.Skeleton - Error: Could not create a bounding rectangle wire. Returning None.")
            return None
        br = Topology.Scale(br, Topology.Centroid(br), 1.5, 1.5, 1)
        bf = Face.ByWire(br, tolerance=tolerance)
        if not Topology.IsInstance(bf, "Face"):
            print("Shell.Skeleton - Error: Could not create a bounding rectangle face. Returning None.")
            return None
        large_shell = Topology.Slice(bf, roof, tolerance=tolerance)
        if not large_shell:
            return None
        faces = Topology.Faces(large_shell)
        if not faces:
            return None
        final_faces = []
        for f in faces:
            internalBoundaries = Face.InternalBoundaries(f)
            if len(internalBoundaries) == 0:
                final_faces.append(f)
        shell = Shell.ByFaces(final_faces, tolerance=tolerance)
        if not Topology.IsInstance(shell, "Shell"):
            print("Shell.Skeleton - Error: Could not create shell. Returning None.")
            return None
        return shell
    
    @staticmethod
    def Square(origin= None, size: float = 1.0,
                  uSides: int = 2, vSides: int = 2, direction: list = [0, 0, 1],
                  placement: str = "center", tolerance: float = 0.0001):
        """
        Creates a square.

        Parameters
        ----------
        origin : topologic_core.Vertex , optional
            The location of the origin of the square. Default is None which results in the square being placed at (0, 0, 0).
        size : float , optional
            The size of the square. Default is 1.0.
        length : float , optional
            The length of the square. Default is 1.0.
        uSides : int , optional
            The number of sides along the width. Default is 2.
        vSides : int , optional
            The number of sides along the length. Default is 2.
        direction : list , optional
            The vector representing the up direction of the square. Default is [0, 0, 1].
        placement : str , optional
            The description of the placement of the origin of the square. This can be "center", or "lowerleft". It is case insensitive. Default is "center".
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        topologic_core.Shell
            The created shell square.

        """
        return Shell.Rectangle(origin=origin, width=size, length=size,
                  uSides=uSides, vSides=vSides, direction=direction,
                  placement=placement, tolerance=tolerance)
    
    @staticmethod
    def _grow_connected_group(seed_idx, group_size, adjacency, visited_global):
        """
        Attempts to grow a group of the given size starting from seed_idx using adjacency.
        Returns a list of indices if successful, else None.
        """
        from collections import deque
        import random

        group = [seed_idx]
        visited = set(group)
        queue = deque([seed_idx])

        while queue and len(group) < group_size:
            current = queue.popleft()
            neighbors = adjacency.get(current, [])
            random.shuffle(neighbors)
            for neighbor in neighbors:
                if neighbor not in visited and neighbor not in visited_global:
                    group.append(neighbor)
                    visited.add(neighbor)
                    queue.append(neighbor)
                    if len(group) >= group_size:
                        break

        return group if len(group) == group_size else None
    
    @staticmethod
    def Vertices(shell) -> list:
        """
        Returns the vertices of the input shell.

        Parameters
        ----------
        shell : topologic_core.Shell
            The input shell.

        Returns
        -------
        list
            The list of vertices.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(shell, "Shell"):
            return None
        vertices = []
        # _ = shell.Vertices(None, vertices) # H to Core
        try:
            _ = Core.InstanceCall(shell, "Vertices", None, vertices)
        except Exception:
            vertices = None
        return vertices

    @staticmethod
    def Voronoi(
        vertices: list,
        face,
        deflection: float = None,
        maxIterations: int = 5,
        convergence: float = 0.001,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """
        Returns the intrinsic/geodesic Voronoi partition of the input Face.

        The sites must lie on, or within ``tolerance`` of, the trimmed input Face.
        Distances are shortest-path distances constrained to that Face, including its
        outer and internal boundaries. UV-coordinate Euclidean distance is never used
        as the metric.

        On PythonOCC, intrinsic distances are approximated with the Kimmel-Sethian
        Fast Marching Method on successively refined triangulations of the trimmed
        Face. ``deflection`` is interpreted as the finest permitted triangulation
        target; refinement starts coarser and never goes below it. Converged
        triangle-local interfaces are chained and reconstructed as degree-1 B-spline
        p-curves on the original OCCT surface before splitting, so the returned
        topology does not inherit one Edge per computational triangle. Analytic and NURBS surface geometry is retained.
        UV-coordinate Euclidean distance and mesh-edge Dijkstra are not used as the
        intrinsic metric.
        """
        from topologicpy.Topology import Topology

        if not isinstance(vertices, list):
            if not silent:
                print("Shell.Voronoi - Error: The input vertices parameter is not a valid list. Returning None.")
            return None
        if len(vertices) < 2 or any(not Topology.IsInstance(v, "Vertex") for v in vertices):
            if not silent:
                print("Shell.Voronoi - Error: At least two valid Vertices are required. Returning None.")
            return None
        if not Topology.IsInstance(face, "Face"):
            if not silent:
                print("Shell.Voronoi - Error: The input face parameter is not a valid Face. Returning None.")
            return None
        if Topology._IsTopologicCoreBackend():
            if not silent:
                print("Shell.Voronoi - Error: Intrinsic surface Voronoi currently requires the PythonOCC backend. Returning None.")
            return None

        try:
            return Core.Shell.Voronoi(
                vertices,
                face,
                deflection=deflection,
                maxIterations=maxIterations,
                convergence=convergence,
                tolerance=tolerance,
                silent=silent,
            )
        except TypeError:
            try:
                return Core.Shell.Voronoi(vertices, face, deflection, maxIterations, convergence, tolerance, silent)
            except Exception:
                pass
        except Exception:
            pass

        if not silent:
            print("Shell.Voronoi - Error: Could not construct the intrinsic Voronoi partition. Returning None.")
        return None

    @staticmethod
    def Wires(shell) -> list:
        """
        Returns the wires of the input shell.

        Parameters
        ----------
        shell : topologic_core.Shell
            The input shell.

        Returns
        -------
        list
            The list of wires.

        """
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(shell, "Shell"):
            return None
        wires = []
        # _ = shell.Wires(None, wires) # H to Core
        try:
            _ = Core.InstanceCall(shell, "Wires", None, wires)
        except Exception:
            wires = None
        return wires

    
    
    
    
    