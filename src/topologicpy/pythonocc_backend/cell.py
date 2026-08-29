from __future__ import annotations

import math
from dataclasses import dataclass

from .occ_utils import make_occ_cell
from .shell import Shell
from .topology import (
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_SHELL,
    TopAbs_VERTEX,
    Topology,
    _downward_wrappers,
    _is_null_shape,
)
from .vertex import Vertex
from .wire import Wire


@dataclass(eq=False)
class Cell(Topology):
    def __init__(
        self,
        shape=None,
        dictionary=None,
        contents=None,
        contexts=None,
        apertures=None,
        shells=None,
    ):
        super().__init__(
            shape=shape,
            dictionary=dictionary,
            contents=contents,
            contexts=contexts,
            apertures=apertures,
        )
        self.shells = list(shells) if shells else []

    @staticmethod
    def ByShell(shell, tolerance: float = 0.0001, silent: bool = False):
        """Create a Cell from a closed Shell."""
        if not Topology.IsInstance(shell, "Shell"):
            if not silent:
                print(
                    "Cell.ByShell - Error: The input shell parameter is not a valid "
                    "topologic shell. Returning None."
                )
            return None

        occ_cell = make_occ_cell(shell)
        if occ_cell is None or _is_null_shape(occ_cell):
            if not silent:
                print(
                    "Cell.ByShell - Error: Could not create an OpenCascade solid. "
                    "Returning None."
                )
            return None

        return Cell(shape=occ_cell, shells=[shell])

    @staticmethod
    def ByFaces(
        faces,
        planarize: bool = False,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """Create a Cell from a collection of Faces."""
        # ``planarize`` is intentionally retained for backend-call compatibility.
        # It is not applied here: the PythonOCC backend preserves the supplied
        # Face geometry and delegates sewing to Shell.ByFaces.
        _ = planarize

        shell = Shell.ByFaces(faces, tolerance=tolerance, silent=silent)
        if not Topology.IsInstance(shell, "Shell"):
            if not silent:
                print(
                    "Cell.ByFaces - Error: Could not create a shell from the input "
                    "faces. Returning None."
                )
            return None

        return Cell.ByShell(shell, tolerance=tolerance, silent=silent)

    @staticmethod
    def ByWires(wires, tolerance: float = 0.0001):
        """
        Create a curve-preserving Cell by lofting through closed section Wires.

        The section curves are passed directly to OCCT. Corresponding Wires must
        contain the same number of Edges; the backend does not resample or rebuild
        mismatched sections.

        Parameters
        ----------
        wires : list
            Ordered closed section Wires.
        tolerance : float , optional
            Geometric tolerance. Default is 0.0001.

        Returns
        -------
        Cell
            The resulting OCCT solid, or None if construction fails.
        """
        if not isinstance(wires, (list, tuple)):
            return None

        wire_list = [wire for wire in wires if isinstance(wire, Wire)]
        if len(wire_list) < 2:
            return None

        try:
            tolerance = abs(float(tolerance))
        except Exception:
            return None

        if not math.isfinite(tolerance) or tolerance <= 0.0:
            return None

        try:
            from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_ThruSections
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopoDS import topods
        except Exception:
            return None

        occ_wires = []
        expected_edge_count = None

        for wire in wire_list:
            shape = getattr(wire, "shape", None)
            if _is_null_shape(shape):
                return None

            try:
                occ_wire = topods.Wire(shape)
            except Exception:
                return None

            if _is_null_shape(occ_wire):
                return None

            try:
                explorer = TopExp_Explorer(occ_wire, TopAbs_EDGE)
                edge_count = 0
                while explorer.More():
                    edge_count += 1
                    explorer.Next()
            except Exception:
                return None

            if edge_count < 1:
                return None

            if expected_edge_count is None:
                expected_edge_count = edge_count
            elif edge_count != expected_edge_count:
                return None

            occ_wires.append(occ_wire)

        try:
            loft = BRepOffsetAPI_ThruSections(
                True,   # isSolid
                True,   # ruled
                tolerance,
            )
            loft.CheckCompatibility(False)

            for occ_wire in occ_wires:
                loft.AddWire(occ_wire)

            loft.Build()
            if not loft.IsDone():
                return None

            shape = loft.Shape()
        except Exception:
            return None

        if _is_null_shape(shape):
            return None

        try:
            result = Topology.ByOcctShape(shape)
        except Exception:
            return None

        return result if isinstance(result, Cell) else None

    @staticmethod
    def ByBox(
        width: float = 1.0,
        length: float = 1.0,
        height: float = 1.0,
        origin=None,
        direction=None,
        placement: str = "center",
        tolerance: float = 0.0001,
    ):
        """Create an OCCT box Cell and optionally orient it to a direction."""
        try:
            width = float(width)
            length = float(length)
            height = float(height)
            tolerance = abs(float(tolerance))
        except Exception:
            return None

        if (
            not all(math.isfinite(value) for value in (width, length, height, tolerance))
            or width <= tolerance
            or length <= tolerance
            or height <= tolerance
            or tolerance <= 0.0
        ):
            return None

        try:
            from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
            from OCC.Core.gp import gp_Pnt
        except Exception:
            return None

        placement = str(placement or "center").lower()
        x_offset = y_offset = z_offset = 0.0

        if placement == "center":
            x_offset = -0.5 * width
            y_offset = -0.5 * length
            z_offset = -0.5 * height
        elif placement == "bottom":
            x_offset = -0.5 * width
            y_offset = -0.5 * length

        ox = float(origin.x) if isinstance(origin, Vertex) else 0.0
        oy = float(origin.y) if isinstance(origin, Vertex) else 0.0
        oz = float(origin.z) if isinstance(origin, Vertex) else 0.0

        try:
            shape = BRepPrimAPI_MakeBox(
                gp_Pnt(ox + x_offset, oy + y_offset, oz + z_offset),
                width,
                length,
                height,
            ).Shape()
        except Exception:
            return None

        if _is_null_shape(shape):
            return None

        result = Topology.ByOcctShape(shape)
        if not isinstance(result, Cell):
            return None

        if direction is None:
            return result

        try:
            direction = [float(direction[0]), float(direction[1]), float(direction[2])]
        except Exception:
            return result

        if direction == [0.0, 0.0, 1.0]:
            return result

        origin_vertex = origin if isinstance(origin, Vertex) else Vertex.ByCoordinates(ox, oy, oz)
        oriented = Cell._orient_to_direction(result, origin_vertex, direction)
        return oriented if isinstance(oriented, Cell) else result

    @staticmethod
    def _orient_to_direction(cell, origin, direction):
        """Rotate a Cell from the +Z axis to the supplied direction."""
        if not isinstance(cell, Cell) or not isinstance(origin, Vertex):
            return None

        try:
            target = [float(direction[0]), float(direction[1]), float(direction[2])]
        except Exception:
            return None

        target_norm = math.sqrt(sum(value * value for value in target))
        if not math.isfinite(target_norm) or target_norm <= 1.0e-12:
            return None

        target = [value / target_norm for value in target]
        source = [0.0, 0.0, 1.0]

        cross = [
            source[1] * target[2] - source[2] * target[1],
            source[2] * target[0] - source[0] * target[2],
            source[0] * target[1] - source[1] * target[0],
        ]
        sin_angle = math.sqrt(sum(value * value for value in cross))
        cos_angle = sum(source[i] * target[i] for i in range(3))

        if sin_angle <= 1.0e-12:
            if cos_angle > 0.0:
                return cell
            axis = [1.0, 0.0, 0.0]
            angle = 180.0
        else:
            axis = [value / sin_angle for value in cross]
            angle = math.degrees(math.atan2(sin_angle, cos_angle))

        try:
            return cell.Rotate(origin, axis[0], axis[1], axis[2], angle)
        except Exception:
            return None

    @staticmethod
    def InternalVertex(cell, tolerance: float = 0.0001):
        """Return a Vertex strictly inside the Cell when one can be found."""
        return CellUtility.InternalVertex(cell, tolerance=tolerance)

    def Shells(self, hostTopology=None, shells=None):
        if not _is_null_shape(getattr(self, "shape", None)):
            result = _downward_wrappers(self, TopAbs_SHELL)
        else:
            result = list(getattr(self, "shells", []) or [])

        if shells is not None:
            shells.extend(result)
            return 0
        return result

    def Faces(self, hostTopology=None, faces=None):
        if not _is_null_shape(getattr(self, "shape", None)):
            result = _downward_wrappers(self, TopAbs_FACE)
        else:
            result = []
            for shell in getattr(self, "shells", []) or []:
                result.extend(shell.Faces())

        if faces is not None:
            faces.extend(result)
            return 0
        return result

    def Edges(self, hostTopology=None, edges=None):
        if not _is_null_shape(getattr(self, "shape", None)):
            result = _downward_wrappers(self, TopAbs_EDGE)
        else:
            result = []
            for shell in getattr(self, "shells", []) or []:
                result.extend(shell.Edges())

        if edges is not None:
            edges.extend(result)
            return 0
        return result

    def Vertices(self, hostTopology=None, vertices=None):
        if not _is_null_shape(getattr(self, "shape", None)):
            result = _downward_wrappers(self, TopAbs_VERTEX)
        else:
            result = []
            for shell in getattr(self, "shells", []) or []:
                result.extend(shell.Vertices())

        if vertices is not None:
            vertices.extend(result)
            return 0
        return result

    def Cells(self, hostTopology=None, cells=None):
        result = [self]
        if cells is not None:
            cells.extend(result)
            return 0
        return result


class CellUtility:
    @staticmethod
    def Volume(cell):
        """Return the OCCT volume of a Cell."""
        if not isinstance(cell, Cell):
            return None

        shape = getattr(cell, "shape", None)
        if _is_null_shape(shape):
            return None

        try:
            from OCC.Core.BRepGProp import brepgprop
            from OCC.Core.GProp import GProp_GProps

            properties = GProp_GProps()
            brepgprop.VolumeProperties(shape, properties)
            volume = float(properties.Mass())
        except Exception:
            return None

        return volume if math.isfinite(volume) else None

    @staticmethod
    def Contains(cell, vertex, tolerance: float = 0.0001):
        """
        Classify a Vertex against a Cell.

        Returns 0 for inside, 1 for on the boundary, and 2 for outside or an
        invalid classification.
        """
        if not isinstance(cell, Cell) or not isinstance(vertex, Vertex):
            return 2

        shape = getattr(cell, "shape", None)
        if _is_null_shape(shape):
            return 2

        try:
            tolerance = abs(float(tolerance))
        except Exception:
            return 2

        try:
            from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
            from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON
            from OCC.Core.gp import gp_Pnt

            classifier = BRepClass3d_SolidClassifier(shape)
            classifier.Perform(
                gp_Pnt(float(vertex.x), float(vertex.y), float(vertex.z)),
                tolerance,
            )
            state = classifier.State()
        except Exception:
            return 2

        if state == TopAbs_IN:
            return 0
        if state == TopAbs_ON:
            return 1
        return 2

    @staticmethod
    def InternalVertex(cell, tolerance: float = 0.0001):
        """Return a Vertex strictly inside a Cell when one can be found."""
        if not isinstance(cell, Cell):
            return None

        shape = getattr(cell, "shape", None)
        if _is_null_shape(shape):
            return None

        center = Topology.CenterOfMass(cell)
        if isinstance(center, Vertex) and CellUtility.Contains(cell, center, tolerance) == 0:
            return center

        try:
            from OCC.Core.Bnd import Bnd_Box
            from OCC.Core.BRepBndLib import brepbndlib

            box = Bnd_Box()
            brepbndlib.Add(shape, box)
            xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
        except Exception:
            return center if isinstance(center, Vertex) else None

        steps = 6
        for i in range(1, steps):
            x = xmin + (xmax - xmin) * i / steps
            for j in range(1, steps):
                y = ymin + (ymax - ymin) * j / steps
                for k in range(1, steps):
                    z = zmin + (zmax - zmin) * k / steps
                    candidate = Vertex.ByCoordinates(x, y, z)
                    if CellUtility.Contains(cell, candidate, tolerance) == 0:
                        return candidate

        return center if isinstance(center, Vertex) else None

    @staticmethod
    def AdjacentVertices(topology, hostTopology, output):
        if topology is None:
            return 1
        return topology.Vertices(hostTopology, output)

    @staticmethod
    def AdjacentEdges(topology, hostTopology, output):
        if topology is None:
            return 1
        return topology.Edges(hostTopology, output)

    @staticmethod
    def AdjacentWires(topology, hostTopology, output):
        if topology is None:
            return 1
        return topology.Wires(hostTopology, output)

    @staticmethod
    def AdjacentFaces(topology, hostTopology, output):
        if topology is None:
            return 1
        return topology.Faces(hostTopology, output)

    @staticmethod
    def AdjacentShells(topology, hostTopology, output):
        if topology is None:
            return 1
        return topology.Shells(hostTopology, output)

    @staticmethod
    def AdjacentCells(topology, hostTopology, output):
        if topology is None:
            return 1
        return topology.Cells(hostTopology, output)

    @staticmethod
    def AdjacentCellComplexes(topology, hostTopology, output):
        if topology is None:
            return 1
        return topology.CellComplexes(hostTopology, output)
