from __future__ import annotations

from dataclasses import dataclass
from .topology import Topology
from .shell import Shell
from .face import Face, FaceUtility
from .wire import Wire
from .edge import Edge
from .vertex import Vertex
from .occ_utils import make_occ_cell
from .helpers import edge_key, vertex_key, dedupe_vertices_by_distance


def _dedupe_vertices(vertices, tolerance: float = 0.0001):
    """
    Dedupe Vertex wrappers by distance, not OCCT shape-hash: faces built by separate
    Face.ByVertices calls never share OCCT vertex sub-shapes even where coincident, so
    hash-dedup leaves duplicates at shared corners.
    """
    return dedupe_vertices_by_distance((v for v in vertices if isinstance(v, Vertex)), tolerance)


@dataclass(eq=False)
class Cell(Topology):
    def __init__(self, shape=None, dictionary=None, contents=None, contexts=None, apertures=None, shells=None):
        super().__init__(shape=shape, dictionary=dictionary, contents=contents, contexts=contexts, apertures=apertures)
        self.shells = list(shells) if shells else []

    @staticmethod
    def _faces_form_closed_shell(faces, tolerance=0.0001):
        edge_counts = {}
        for face in faces:
            external = getattr(face, "external", None)
            if not isinstance(external, Wire):
                continue
            for edge in getattr(external, "edges", []) or []:
                if not isinstance(edge, Edge):
                    continue
                key = edge_key(edge, tolerance)
                edge_counts[key] = edge_counts.get(key, 0) + 1
        return bool(edge_counts) and all(count == 2 for count in edge_counts.values())

    @staticmethod
    def ByShell(shell, tolerance: float = 0.0001, silent: bool = False):
        if not Topology.IsInstance(shell, "Shell"):
            if not silent:
                print("Cell.ByShell - Error: The input shell parameter is not a valid topologic shell. Returning None.")
            return None
        occ_cell = make_occ_cell(shell)
        if occ_cell is None:
            if not silent:
                print("Cell.ByShell - Error: Could not create an OpenCascade solid. Returning None.")
            return None
        return Cell(shape=occ_cell, shells=[shell])

    @staticmethod
    def ByFaces(faces, planarize: bool = False, tolerance: float = 0.0001, silent: bool = False):
        shell = Shell.ByFaces(faces, tolerance=tolerance, silent=silent)
        if not Topology.IsInstance(shell, "Shell"):
            if not silent:
                print("Cell.ByFaces - Error: Could not create a shell from the input faces. Returning None.")
            return None
        return Cell.ByShell(shell, tolerance=tolerance, silent=silent)

    def Shells(self, hostTopology=None, shells=None):
        result = list(getattr(self, "shells", []) or [])
        if shells is not None:
            shells.extend(result)
            return 0
        return result

    def Faces(self, hostTopology=None, faces=None):
        result = []
        for shell in self.shells:
            result.extend(shell.Faces())
        if faces is not None:
            faces.extend(result)
            return 0
        return result

    def Edges(self, hostTopology=None, edges=None):
        result = []
        seen_keys = set()
        for face in self.Faces():
            for edge in FaceUtility.Edges(face) or []:
                if not isinstance(edge, Edge):
                    continue
                key = edge_key(edge)
                if key not in seen_keys:
                    seen_keys.add(key)
                    result.append(edge)
        if edges is not None:
            edges.extend(result)
            return 0
        return result

    def Vertices(self, hostTopology=None, vertices=None):
        result = []
        for edge in self.Edges():
            result.extend([edge.start, edge.end])
        result = _dedupe_vertices([v for v in result if isinstance(v, Vertex)])
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


def _cell_by_box(width: float = 1.0, length: float = 1.0, height: float = 1.0,
                  origin=None, direction=None, placement: str = "center", tolerance: float = 0.0001):
    """
    Builds an axis-aligned box Cell directly via BRepPrimAPI_MakeBox, then
    orients/places it. This mirrors the algorithm-layer Cell.Box contract
    (width/length/height + origin + placement), but the algorithm-layer
    Cell.Box (src/topologicpy/Cell.py) actually delegates to Cell.Prism, so
    this backend-level Cell.ByBox only matters to callers that go through
    Core.Cell.ByBox directly.
    """
    try:
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
        from OCC.Core.gp import gp_Pnt
    except Exception:
        return None

    xOffset = yOffset = zOffset = 0.0
    placement = (placement or "center").lower()
    if placement == "center":
        xOffset = -width * 0.5
        yOffset = -length * 0.5
        zOffset = -height * 0.5
    elif placement == "bottom":
        xOffset = -width * 0.5
        yOffset = -length * 0.5

    ox = origin.x if isinstance(origin, Vertex) else 0.0
    oy = origin.y if isinstance(origin, Vertex) else 0.0
    oz = origin.z if isinstance(origin, Vertex) else 0.0

    try:
        occ_box = BRepPrimAPI_MakeBox(
            gp_Pnt(ox + xOffset, oy + yOffset, oz + zOffset),
            float(width), float(length), float(height),
        ).Shape()
    except Exception:
        return None

    result = Topology.ByOcctShape(occ_box)
    if result is None or direction in (None, [0, 0, 1], (0, 0, 1)):
        return result

    # Reorient from the default [0, 0, 1] up-direction to the requested one.
    try:
        from .topology import Topology as _T
        origin_vertex = origin if isinstance(origin, Vertex) else Vertex.ByCoordinates(ox, oy, oz)
        oriented = _reorient(result, origin_vertex, [0, 0, 1], direction, tolerance)
        if oriented is not None:
            return oriented
    except Exception:
        pass
    return result


def _reorient(topology, origin, dirA, dirB, tolerance=0.0001):
    """
    Small local re-implementation of the rotate-to-align-directions step used
    by Topology.Orient (src/topologicpy/Topology.py), for backend-level
    primitive constructors (Cell.ByBox) that need it without depending on the
    algorithm layer.
    """
    import math

    def _normalize(v):
        n = math.sqrt(sum(c * c for c in v))
        if n == 0:
            return [0.0, 0.0, 1.0]
        return [c / n for c in v]

    a = _normalize(dirA)
    b = _normalize(dirB)
    cross = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    sin_a = math.sqrt(sum(c * c for c in cross))
    cos_a = sum(a[i] * b[i] for i in range(3))
    if sin_a < 1e-12:
        if cos_a > 0:
            return topology
        # 180 degree flip: pick any axis perpendicular to a.
        axis = [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0]
        cross = [a[1] * axis[2] - a[2] * axis[1], a[2] * axis[0] - a[0] * axis[2], a[0] * axis[1] - a[1] * axis[0]]
        cross = _normalize(cross)
        angle = 180.0
    else:
        cross = [c / sin_a for c in cross]
        angle = math.degrees(math.atan2(sin_a, cos_a))
    return topology.Rotate(origin, cross[0], cross[1], cross[2], angle)


Cell.ByBox = staticmethod(_cell_by_box)


def _cell_by_wires(wires, close: bool = False, tolerance: float = 0.0001, silent: bool = False):
    """
    Loft profiles and cap the ends: side faces via Shell.ByWires, bottom cap
    Face.ByWire(wires[0]), top cap Face.ByWire(wires[-1]). The algorithm-layer
    Cell.ByWires is self-contained and never reaches Core; this only serves
    direct Core.Cell.ByWires callers.
    """
    if not isinstance(wires, list):
        if not silent:
            print("Cell.ByWires - Error: The input wires parameter is not a valid list. Returning None.")
        return None
    wire_list = [w for w in wires if isinstance(w, Wire)]
    if len(wire_list) < 2:
        if not silent:
            print("Cell.ByWires - Error: At least two valid wires are required. Returning None.")
        return None
    if close:
        wire_list = wire_list + [wire_list[0]]

    side_shell = Shell.ByWires(wire_list, tolerance=tolerance, silent=silent)
    if side_shell is None:
        if not silent:
            print("Cell.ByWires - Error: Could not create the side faces. Returning None.")
        return None

    faces = list(getattr(side_shell, "faces", []) or [])
    if not close:
        bottom_cap = Face.ByWire(wire_list[0])
        if bottom_cap is not None:
            faces.append(bottom_cap)
        top_cap = Face.ByWire(wire_list[-1])
        if top_cap is not None:
            faces.append(top_cap)

    return Cell.ByFaces(faces, tolerance=tolerance, silent=silent)


Cell.ByWires = staticmethod(_cell_by_wires)


def _cell_internal_vertex(cell, tolerance: float = 0.0001):
    return CellUtility.InternalVertex(cell, tolerance=tolerance)


Cell.InternalVertex = _cell_internal_vertex


class CellUtility:
    @staticmethod
    def Volume(cell):
        """
        Returns the volume of the input Cell via OCCT's volume properties
        (same brepgprop.VolumeProperties call already used successfully by
        Topology.CenterOfMass in topology.py for Cell/CellComplex/Cluster
        shapes).
        """
        if not isinstance(cell, Cell):
            return None
        shape = getattr(cell, "shape", None)
        if shape is None or (hasattr(shape, "IsNull") and shape.IsNull()):
            return None
        try:
            from OCC.Core.GProp import GProp_GProps
            from OCC.Core.BRepGProp import brepgprop
            props = GProp_GProps()
            brepgprop.VolumeProperties(shape, props)
            return props.Mass()
        except Exception:
            return None

    @staticmethod
    def Contains(cell, vertex, tolerance: float = 0.0001):
        """
        Classify a vertex against the cell via BRepClass3d_SolidClassifier
        (0=inside, 1=on-boundary, 2=outside). Cells are boundary-representation
        shells, so the classifier returns ON for any point on/within the shell;
        the algorithm layer's Cell.ContainmentStatus compensates with 8 offset
        vertices + majority vote.
        """
        if not isinstance(cell, Cell) or not isinstance(vertex, Vertex):
            return 2
        shape = getattr(cell, "shape", None)
        if shape is None or (hasattr(shape, "IsNull") and shape.IsNull()):
            return 2
        try:
            from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
            from OCC.Core.gp import gp_Pnt
            from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON, TopAbs_OUT
            classifier = BRepClass3d_SolidClassifier(shape)
            classifier.Perform(gp_Pnt(float(vertex.x), float(vertex.y), float(vertex.z)), float(tolerance))
            state = classifier.State()
            if state == TopAbs_IN:
                return 0
            if state == TopAbs_ON:
                return 1
            return 2
        except Exception:
            return 2

    @staticmethod
    def InternalVertex(cell, tolerance: float = 0.0001):
        """
        Vertex strictly inside the Cell: try CenterOfMass, confirm via
        CellUtility.Contains; for non-convex cells whose centroid is outside or on
        the boundary, sample a small grid inside the bounding box until one tests
        strictly inside.
        """
        if not isinstance(cell, Cell):
            return None
        shape = getattr(cell, "shape", None)
        if shape is None or (hasattr(shape, "IsNull") and shape.IsNull()):
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
            for j in range(1, steps):
                for k in range(1, steps):
                    x = xmin + (xmax - xmin) * i / steps
                    y = ymin + (ymax - ymin) * j / steps
                    z = zmin + (zmax - zmin) * k / steps
                    candidate = Vertex.ByCoordinates(x, y, z)
                    if CellUtility.Contains(cell, candidate, tolerance) == 0:
                        return candidate

        return center if isinstance(center, Vertex) else None

# ---------------------------------------------------------------------------
# Explicit unsupported Cell API
# ---------------------------------------------------------------------------
from .helpers import not_implemented as _not_implemented


def _cell_not_implemented(name, return_value=None):
    def _method(*args, **kwargs):
        return _not_implemented(f"Cell.{name}", return_value)
    return _method


def _cell_utility_not_implemented(name, return_value=None):
    def _method(*args, **kwargs):
        return _not_implemented(f"CellUtility.{name}", return_value)
    return _method


def _make_adjacent(method_name):
    """Return a staticmethod that delegates to topology.method(hostTopology, output)."""
    @staticmethod
    def _impl(topology, hostTopology, output):
        if topology is None:
            return 1
        return getattr(topology, method_name)(hostTopology, output)
    return _impl

CellUtility.AdjacentVertices = _make_adjacent("Vertices")
CellUtility.AdjacentEdges = _make_adjacent("Edges")
CellUtility.AdjacentWires = _make_adjacent("Wires")
CellUtility.AdjacentFaces = _make_adjacent("Faces")
CellUtility.AdjacentShells = _make_adjacent("Shells")
CellUtility.AdjacentCells = _make_adjacent("Cells")
CellUtility.AdjacentCellComplexes = _make_adjacent("CellComplexes")

# Cell.ByBox, Cell.ByWires, Cell.InternalVertex, CellUtility.Volume,
# CellUtility.Contains, and CellUtility.InternalVertex are implemented above
# -- do not clobber them here.
