# Copyright (C) 2026
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free Software
# Foundation, either version 3.0 of the License, or (at your option) any later
# version.

from __future__ import annotations

import math
import os
import shutil
from typing import Any, Optional

from topologicpy.Core import Core


class _Imported3DM(list):
    """List carrying the lossless Rhino document used to create its members.

    The public result remains an ordinary ``list`` for API compatibility.  The
    private provenance lets an unchanged import/export round trip retain Rhino
    document data which has no Topologic equivalent (views, materials, render
    settings, plug-in data, and object attributes).
    """

    def __init__(self, values=(), *, file=None, source_path=None, complete=False):
        super().__init__(values)
        self._rhino_file = file
        self._rhino_source_path = source_path
        self._rhino_complete = bool(complete)
        self._rhino_member_ids = tuple(id(value) for value in self)

    def is_unchanged(self):
        return self._rhino_complete and self._rhino_member_ids == tuple(
            id(value) for value in self
        )


class Rhino:
    """Utilities for importing and exporting Rhino ``.3dm`` files."""

    @staticmethod
    def _Rhino3dm(silent: bool = False):
        try:
            import rhino3dm
            return rhino3dm
        except ImportError:
            if not silent:
                print(
                    "Rhino - Error: The optional rhino3dm package is required. "
                    "Install it with 'pip install rhino3dm'."
                )
            return None

    @staticmethod
    def _ExpandedKnots(knots) -> list:
        """Convert an openNURBS knot vector to the expanded OCCT convention."""
        values = [float(value) for value in knots]
        if not values:
            return []
        # openNURBS omits one copy of each end knot. OCCT's expanded form has
        # pole_count + degree + 1 entries and includes both missing copies.
        return [values[0]] + values + [values[-1]]

    @staticmethod
    def _PointCoordinates(point) -> tuple:
        """Return Euclidean coordinates from an openNURBS homogeneous point."""
        weight = float(getattr(point, "W", 1.0))
        if abs(weight) <= 1.0e-15:
            raise ValueError("A Rhino control point has a zero weight.")
        return (
            float(point.X) / weight,
            float(point.Y) / weight,
            float(point.Z) / weight,
        )

    @staticmethod
    def _NurbsCurve(curve, tolerance: float = 0.0001, silent: bool = False):
        from topologicpy.Edge import Edge
        from topologicpy.Vertex import Vertex

        try:
            nurbs = curve.ToNurbsCurve()
        except Exception:
            nurbs = None
        if nurbs is None:
            if not silent:
                print("Rhino - Warning: A Rhino curve could not be converted to NURBS.")
            return None

        try:
            points = list(nurbs.Points)
            vertices = [Vertex.ByCoordinates(*Rhino._PointCoordinates(point)) for point in points]
            weights = [float(point.W) for point in points]
            knots = Rhino._ExpandedKnots(nurbs.Knots)
            degree = int(nurbs.Degree)

            # A periodic openNURBS curve stores repeated trailing poles and a
            # periodic knot sequence. Passing the complete representation to
            # OCCT as a non-periodic B-spline reproduces the identical finite
            # curve, including its closed seam, without refitting it.
            edge = Edge.ByNurbsParameters(
                controlPoints=vertices,
                weights=weights,
                knots=knots,
                isRational=bool(nurbs.IsRational),
                isPeriodic=False,
                degree=degree,
                tolerance=tolerance,
                silent=True,
            )
        except Exception:
            edge = None

        if edge is None and not silent:
            print("Rhino - Warning: An exact Rhino NURBS curve could not be created.")
        return edge

    @staticmethod
    def _SurfaceParameters(surface, silent: bool = False) -> Optional[dict]:
        try:
            nurbs = surface.ToNurbsSurface()
        except Exception:
            nurbs = None
        if nurbs is None:
            if not silent:
                print("Rhino - Warning: A Rhino surface could not be converted to NURBS.")
            return None

        try:
            count_u = int(nurbs.Points.CountU)
            count_v = int(nurbs.Points.CountV)
            coordinates = []
            weights = []
            for u in range(count_u):
                coordinate_row = []
                weight_row = []
                for v in range(count_v):
                    point = nurbs.Points[u, v]
                    coordinate_row.append(Rhino._PointCoordinates(point))
                    weight_row.append(float(point.W))
                coordinates.append(coordinate_row)
                weights.append(weight_row)
            return {
                "coordinates": coordinates,
                "weights": weights,
                "uKnots": Rhino._ExpandedKnots(nurbs.KnotsU),
                "vKnots": Rhino._ExpandedKnots(nurbs.KnotsV),
                "isRational": bool(nurbs.IsRational),
                # Closed Rhino surfaces commonly store duplicated seam poles;
                # the complete finite representation is therefore passed to
                # OCCT without requesting a second periodic conversion.
                "isUPeriodic": False,
                "isVPeriodic": False,
                "rhinoIsUPeriodic": bool(nurbs.IsPeriodic(0)),
                "rhinoIsVPeriodic": bool(nurbs.IsPeriodic(1)),
                "uDegree": int(nurbs.Degree(0)),
                "vDegree": int(nurbs.Degree(1)),
            }
        except Exception:
            if not silent:
                print("Rhino - Warning: Invalid Rhino NURBS surface parameters.")
            return None

    @staticmethod
    def _NurbsSurface(surface, tolerance: float = 0.0001, silent: bool = False):
        from topologicpy.Face import Face
        from topologicpy.Vertex import Vertex

        parameters = Rhino._SurfaceParameters(surface, silent=silent)
        if parameters is None:
            return None
        control_points = [
            [Vertex.ByCoordinates(*coordinates) for coordinates in row]
            for row in parameters["coordinates"]
        ]
        return Face.ByNurbsParameters(
            controlPoints=control_points,
            weights=parameters["weights"],
            uKnots=parameters["uKnots"],
            vKnots=parameters["vKnots"],
            isRational=parameters["isRational"],
            isUPeriodic=parameters["isUPeriodic"],
            isVPeriodic=parameters["isVPeriodic"],
            uDegree=parameters["uDegree"],
            vDegree=parameters["vDegree"],
            tolerance=tolerance,
            silent=silent,
        )

    @staticmethod
    def _WireByLoop(loop, edges: list, tolerance: float = 0.0001):
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire

        oriented_edges = []
        for trim in loop.Trims:
            index = int(trim.EdgeIndex)
            if index < 0 or index >= len(edges) or edges[index] is None:
                # A negative edge index denotes a singular trim. It collapses
                # to one vertex and does not contribute a three-dimensional
                # edge to the boundary wire.
                continue
            edge = edges[index]
            if bool(trim.IsReversed):
                reversed_edge = Edge.Reverse(edge, tolerance=tolerance, silent=True)
                if reversed_edge is not None:
                    edge = reversed_edge
            oriented_edges.append(edge)
        if not oriented_edges:
            return None
        return Wire.ByEdges(oriented_edges, tolerance=tolerance, silent=True)

    @staticmethod
    def _Brep(brep, tolerance: float = 0.0001, silent: bool = False):
        from topologicpy.Cell import Cell
        from topologicpy.Face import Face
        from topologicpy.Shell import Shell
        from topologicpy.Topology import Topology
        from topologicpy.Vertex import Vertex

        rhino3dm = Rhino._Rhino3dm(silent=True)

        # A closed Rhino B-rep made entirely from straight edges is already an
        # exact polyhedron. Reconstruct it directly from Rhino's shared vertex
        # and face indices. Creating six independent NURBS support faces and
        # sewing them can leave edges without usable p-curves, producing a Cell
        # that exists topologically but cannot be tessellated by OCCT.
        if bool(brep.IsSolid):
            try:
                linear_edges = all(
                    bool(edge.IsLinear(tolerance))
                    for edge in brep.Edges
                )
                simple_faces = all(
                    len(face.Loops) == 1
                    and face.OuterLoop is not None
                    for face in brep.Faces
                )
            except Exception:
                linear_edges = False
                simple_faces = False

            if linear_edges and simple_faces:
                try:
                    vertices = [
                        [
                            float(vertex.Location.X),
                            float(vertex.Location.Y),
                            float(vertex.Location.Z),
                        ]
                        for vertex in brep.Vertices
                    ]
                    face_indices = []
                    for rhino_face in brep.Faces:
                        indices = [
                            int(trim.StartVertexIndex)
                            for trim in rhino_face.OuterLoop.Trims
                            if int(trim.StartVertexIndex) >= 0
                        ]
                        if len(indices) < 3:
                            raise ValueError("A planar Rhino face has fewer than three vertices.")
                        face_indices.append(indices)
                    cell = Topology.ByGeometry(
                        vertices=vertices,
                        faces=face_indices,
                        topologyType="Cell",
                        tolerance=tolerance,
                        silent=True,
                    )
                except Exception:
                    cell = None
                if Topology.IsInstance(cell, "Cell"):
                    return cell

        edges = [Rhino._NurbsCurve(edge, tolerance=tolerance, silent=True) for edge in brep.Edges]
        faces = []

        for rhino_face in brep.Faces:
            outer = None
            holes = []
            for loop in rhino_face.Loops:
                wire = Rhino._WireByLoop(loop, edges, tolerance=tolerance)
                if wire is None:
                    continue
                if rhino3dm is not None and loop.LoopType == rhino3dm.BrepLoopType.Outer:
                    outer = wire
                else:
                    holes.append(wire)
            if outer is None:
                if not silent:
                    print("Rhino - Warning: A Rhino B-rep face has no usable outer loop.")
                continue

            parameters = Rhino._SurfaceParameters(rhino_face.UnderlyingSurface(), silent=True)
            face = None
            try:
                is_planar = bool(rhino_face.IsPlanar())
            except Exception:
                is_planar = False

            # A planar trimmed face must be constructed directly from its outer
            # and inner wires. Routing it through the general support-surface
            # constructor can cause OCCT to classify the circular hole as the
            # positive region when the Rhino face is orientation-reversed.
            if is_planar and holes:
                face = Face.ByWires(outer, holes, tolerance=tolerance, silent=True)

            if (
                not Topology.IsInstance(face, "Face")
                and parameters is not None
                and Core.HasAttribute("Face", "ByNurbsParametersAndWires")
            ):
                control_points = [
                    [Vertex.ByCoordinates(*coordinates) for coordinates in row]
                    for row in parameters["coordinates"]
                ]
                try:
                    face = Core.Face.ByNurbsParametersAndWires(
                        control_points,
                        parameters["weights"],
                        parameters["uKnots"],
                        parameters["vKnots"],
                        parameters["isRational"],
                        parameters["isUPeriodic"],
                        parameters["isVPeriodic"],
                        parameters["uDegree"],
                        parameters["vDegree"],
                        outer,
                        holes,
                        False,
                        tolerance,
                    )
                except Exception:
                    face = None

            # The fallback is exact for planar Rhino faces and allows planar
            # B-reps to import through TopologicCore as well as PythonOCC.
            if not Topology.IsInstance(face, "Face"):
                face = Face.ByWires(outer, holes, tolerance=tolerance, silent=True)
            if Topology.IsInstance(face, "Face") and bool(rhino_face.OrientationIsReversed):
                inverted = Face.Invert(face, tolerance=tolerance, silent=True)
                if Topology.IsInstance(inverted, "Face"):
                    face = inverted
            if Topology.IsInstance(face, "Face"):
                faces.append(face)

        if len(faces) == 1:
            return faces[0]
        if not faces:
            return None
        if bool(brep.IsSolid):
            cell = Cell.ByFaces(faces, tolerance=tolerance, silent=True)
            if Topology.IsInstance(cell, "Cell"):
                return cell
        shell = Shell.ByFaces(faces, tolerance=tolerance, silent=True)
        if Topology.IsInstance(shell, "Shell"):
            return shell
        return None

    @staticmethod
    def _Dictionary(file, file_object, geometry) -> dict:
        attributes = file_object.Attributes
        layer_index = int(getattr(attributes, "LayerIndex", -1))
        layer = file.Layers[layer_index] if 0 <= layer_index < len(file.Layers) else None
        result = {
            "name": str(getattr(attributes, "Name", "") or ""),
            "object_id": str(getattr(attributes, "Id", "") or ""),
            "layer": str(getattr(layer, "Name", "") or ""),
            "layer_index": layer_index,
            "rhino_type": geometry.__class__.__name__,
        }
        color = getattr(layer, "Color", None)
        if color is not None:
            try:
                if isinstance(color, (list, tuple)):
                    result["color"] = [int(value) for value in color[:4]]
                else:
                    result["color"] = [int(color.R), int(color.G), int(color.B), int(color.A)]
            except Exception:
                pass
        try:
            for key, value in attributes.GetUserStrings():
                result[str(key)] = str(value)
        except Exception:
            pass
        return result

    @staticmethod
    def _Geometry(geometry, tolerance: float = 0.0001, silent: bool = False):
        rhino3dm = Rhino._Rhino3dm(silent=True)
        if rhino3dm is None:
            return None
        from topologicpy.Vertex import Vertex

        if isinstance(geometry, rhino3dm.Point):
            location = geometry.Location
            return Vertex.ByCoordinates(float(location.X), float(location.Y), float(location.Z))
        if isinstance(geometry, rhino3dm.Brep):
            return Rhino._Brep(geometry, tolerance=tolerance, silent=silent)
        if isinstance(geometry, rhino3dm.Surface):
            return Rhino._NurbsSurface(geometry, tolerance=tolerance, silent=silent)
        if isinstance(geometry, rhino3dm.Curve):
            return Rhino._NurbsCurve(geometry, tolerance=tolerance, silent=silent)
        if not silent:
            print(f"Rhino - Warning: Unsupported Rhino geometry type '{geometry.__class__.__name__}'.")
        return None

    @staticmethod
    def _ObjectAttributes(topology, rhino3dm):
        """Create Rhino attributes from a Topologic dictionary."""
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology

        attributes = rhino3dm.ObjectAttributes()
        try:
            values = Dictionary.PythonDictionary(Topology.Dictionary(topology)) or {}
        except Exception:
            values = {}
        name = values.pop("name", "")
        if name is not None:
            attributes.Name = str(name)
        ignored = {"object_id", "layer", "layer_index", "rhino_type", "color"}
        for key, value in values.items():
            if key not in ignored and value is not None:
                try:
                    attributes.SetUserString(str(key), str(value))
                except Exception:
                    pass
        return attributes

    @staticmethod
    def _Mesh(topology, rhino3dm, tolerance=0.0001):
        """Return a Rhino mesh for face-bearing topology."""
        from topologicpy.Topology import Topology

        data = Topology.Geometry(
            topology, triangulate=True, mantissa=12, tolerance=tolerance
        )
        if not isinstance(data, dict) or not data.get("faces"):
            return None
        mesh = rhino3dm.Mesh()
        for point in data.get("vertices", []):
            mesh.Vertices.Add(float(point[0]), float(point[1]), float(point[2]))
        for face in data.get("faces", []):
            if len(face) == 3:
                mesh.Faces.AddFace(int(face[0]), int(face[1]), int(face[2]))
            elif len(face) == 4:
                mesh.Faces.AddFace(
                    int(face[0]), int(face[1]), int(face[2]), int(face[3])
                )
            else:
                # Geometry(triangulate=True) should already return triangles,
                # but retain a safe fan fallback for alternate backends.
                for index in range(1, len(face) - 1):
                    mesh.Faces.AddFace(int(face[0]), int(face[index]), int(face[index + 1]))
        try:
            mesh.Normals.ComputeNormals()
            mesh.Compact()
        except Exception:
            pass
        return mesh if mesh.IsValid else None

    @staticmethod
    def _AddTopology(file, topology, tolerance=0.0001, silent=False):
        """Add one topology to a File3dm, returning whether it was supported."""
        rhino3dm = Rhino._Rhino3dm(silent=silent)
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology
        from topologicpy.Vertex import Vertex

        attributes = Rhino._ObjectAttributes(topology, rhino3dm)
        try:
            if Topology.IsInstance(topology, "Vertex"):
                point = rhino3dm.Point3d(
                    Vertex.X(topology, mantissa=12),
                    Vertex.Y(topology, mantissa=12),
                    Vertex.Z(topology, mantissa=12),
                )
                file.Objects.AddPoint(point, attributes)
                return True
            if Topology.IsInstance(topology, "Edge"):
                start = Edge.StartVertex(topology)
                end = Edge.EndVertex(topology)
                start_point = rhino3dm.Point3d(
                    *Vertex.Coordinates(start, mantissa=12)
                )
                end_point = rhino3dm.Point3d(*Vertex.Coordinates(end, mantissa=12))
                file.Objects.AddLine(start_point, end_point, attributes)
                return True
            if Topology.IsInstance(topology, "Wire"):
                points = [
                    rhino3dm.Point3d(*Vertex.Coordinates(v, mantissa=12))
                    for v in Topology.Vertices(topology, silent=True)
                ]
                if len(points) >= 2:
                    file.Objects.AddPolyline(points, attributes)
                    return True
            if Topology.IsInstance(topology, "Face") or Topology.IsInstance(
                topology, "Shell"
            ) or Topology.IsInstance(topology, "Cell") or Topology.IsInstance(
                topology, "CellComplex"
            ):
                mesh = Rhino._Mesh(topology, rhino3dm, tolerance=tolerance)
                if mesh is not None:
                    file.Objects.AddMesh(mesh, attributes)
                    return True
        except Exception:
            pass
        if not silent:
            print(
                "Rhino.ExportTo3DM - Warning: A topology could not be exported."
            )
        return False

    @staticmethod
    def ExportTo3DM(
        topologies,
        path: str,
        overwrite: bool = False,
        version: int = 8,
        tolerance: float = 0.0001,
        silent: bool = False,
    ) -> bool:
        """Export Topologic geometry to a Rhino ``.3dm`` file.

        An unchanged list returned by :meth:`By3DMPath` is exported losslessly,
        including all geometry, attributes, layers, units, tolerances, views,
        materials, and document data. This is the exact round-trip path. Other
        Topologic input is exported as native points/lines/polylines and Rhino
        meshes; this explicit mesh conversion does not claim NURBS fidelity.
        """
        rhino3dm = Rhino._Rhino3dm(silent=silent)
        if rhino3dm is None:
            return False
        if not isinstance(path, (str, os.PathLike)):
            if not silent:
                print("Rhino.ExportTo3DM - Error: The input path is invalid.")
            return False
        output_path = os.fspath(path)
        if not output_path.lower().endswith(".3dm"):
            output_path += ".3dm"
        if os.path.exists(output_path) and not overwrite:
            if not silent:
                print(
                    "Rhino.ExportTo3DM - Error: A file already exists and "
                    "overwrite is False. Returning False."
                )
            return False

        # The source bytes are the only way to retain document sections which
        # have no Topologic representation. Copy them for an unchanged complete
        # import, yielding a byte-for-byte round trip.
        if isinstance(topologies, _Imported3DM) and topologies.is_unchanged():
            source = topologies._rhino_source_path
            try:
                if source and os.path.isfile(source):
                    if os.path.abspath(source) != os.path.abspath(output_path):
                        shutil.copyfile(source, output_path)
                    return True
                if topologies._rhino_file is not None:
                    return bool(topologies._rhino_file.Write(output_path, int(version)))
            except Exception:
                if not silent:
                    print("Rhino.ExportTo3DM - Error: Could not write the source document.")
                return False

        from topologicpy.Topology import Topology

        if Topology.IsInstance(topologies, "Topology"):
            topologies = [topologies]
        elif not isinstance(topologies, (list, tuple)):
            if not silent:
                print("Rhino.ExportTo3DM - Error: No valid topologies were supplied.")
            return False
        valid = [item for item in topologies if Topology.IsInstance(item, "Topology")]
        if not valid:
            if not silent:
                print("Rhino.ExportTo3DM - Error: No valid topologies were supplied.")
            return False

        file = rhino3dm.File3dm()
        try:
            file.Settings.ModelAbsoluteTolerance = abs(float(tolerance))
        except Exception:
            pass
        added = 0
        for topology in valid:
            members = (
                Topology.SubTopologies(topology, subTopologyType="CellComplex")
                + Topology.SubTopologies(topology, subTopologyType="Cell")
                + Topology.SubTopologies(topology, subTopologyType="Shell")
                + Topology.SubTopologies(topology, subTopologyType="Face")
                + Topology.SubTopologies(topology, subTopologyType="Wire")
                + Topology.SubTopologies(topology, subTopologyType="Edge")
                + Topology.SubTopologies(topology, subTopologyType="Vertex")
                if Topology.IsInstance(topology, "Cluster")
                else [topology]
            )
            for member in members:
                added += int(Rhino._AddTopology(file, member, tolerance, silent))
        if not added:
            return False
        try:
            return bool(file.Write(output_path, int(version)))
        except Exception:
            if not silent:
                print("Rhino.ExportTo3DM - Error: Could not write the file.")
            return False

    @staticmethod
    def By3DMFile(
        file,
        objectNames: list = None,
        layerNames: list = None,
        transferDictionaries: bool = True,
        tolerance: float = None,
        silent: bool = False,
    ) -> list:
        """Import exact supported geometry from a ``rhino3dm.File3dm`` object.

        Curves and surfaces are transferred through their rational B-spline
        control points, weights, degrees and knots. B-rep faces are reconstructed
        from their exact support surfaces and ordered three-dimensional boundary
        loops. Geometry is never tessellated as an implicit fallback.
        """
        rhino3dm = Rhino._Rhino3dm(silent=silent)
        if rhino3dm is None or not isinstance(file, rhino3dm.File3dm):
            if not silent:
                print("Rhino.By3DMFile - Error: The input is not a valid File3dm object.")
            return []

        try:
            document_tolerance = float(file.Settings.ModelAbsoluteTolerance)
        except Exception:
            document_tolerance = 0.0001
        if tolerance is None:
            tolerance = document_tolerance
        try:
            tolerance = abs(float(tolerance))
        except Exception:
            tolerance = document_tolerance
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            tolerance = document_tolerance if document_tolerance > 0.0 else 0.0001

        object_filter = {str(value) for value in (objectNames or [])}
        layer_filter = {str(value) for value in (layerNames or [])}
        result = []
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology

        for file_object in file.Objects:
            attributes = file_object.Attributes
            name = str(getattr(attributes, "Name", "") or "")
            layer_index = int(getattr(attributes, "LayerIndex", -1))
            layer = file.Layers[layer_index] if 0 <= layer_index < len(file.Layers) else None
            layer_name = str(getattr(layer, "Name", "") or "")
            if object_filter and name not in object_filter:
                continue
            if layer_filter and layer_name not in layer_filter:
                continue
            topology = Rhino._Geometry(file_object.Geometry, tolerance=tolerance, silent=silent)
            if not Topology.IsInstance(topology, "Topology"):
                continue
            if transferDictionaries:
                dictionary = Dictionary.ByPythonDictionary(
                    Rhino._Dictionary(file, file_object, file_object.Geometry)
                )
                topology = Topology.SetDictionary(topology, dictionary)
            result.append(topology)
        # "Complete" describes the requested document selection, not how many
        # objects the active Topologic backend can materialize. Unsupported
        # Rhino objects still remain in the retained source document and must
        # survive an unchanged round trip.
        complete = not object_filter and not layer_filter
        return _Imported3DM(result, file=file, complete=complete)

    @staticmethod
    def By3DMPath(
        path: str,
        objectNames: list = None,
        layerNames: list = None,
        transferDictionaries: bool = True,
        tolerance: float = None,
        silent: bool = False,
    ) -> list:
        """Import exact supported geometry from a Rhino ``.3dm`` file path."""
        rhino3dm = Rhino._Rhino3dm(silent=silent)
        if rhino3dm is None:
            return []
        if not isinstance(path, (str, os.PathLike)) or not os.path.isfile(path):
            if not silent:
                print("Rhino.By3DMPath - Error: The input path is not a valid file.")
            return []
        try:
            file = rhino3dm.File3dm.Read(str(path))
        except Exception:
            file = None
        if file is None:
            if not silent:
                print("Rhino.By3DMPath - Error: The .3dm file could not be read.")
            return []
        result = Rhino.By3DMFile(
            file,
            objectNames=objectNames,
            layerNames=layerNames,
            transferDictionaries=transferDictionaries,
            tolerance=tolerance,
            silent=silent,
        )
        if isinstance(result, _Imported3DM):
            result._rhino_source_path = os.path.abspath(os.fspath(path))
        return result
