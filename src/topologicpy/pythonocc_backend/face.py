from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import math
from .topology import Topology
from .wire import Wire
from .vertex import Vertex
from .edge import Edge
from .occ_utils import make_occ_face
from .helpers import unique_by_uuid


@dataclass(eq=False)
class Face(Topology):
    external: Optional[Wire] = None
    internals: list = field(default_factory=list)

    @staticmethod
    def ByExternalBoundary(wire):
        if not isinstance(wire, Wire):
            return None
        if not wire.IsClosed():
            return None
        return Face(shape=make_occ_face(wire), external=wire, internals=[])

    @staticmethod
    def ByWire(wire):
        return Face.ByExternalBoundary(wire)

    @staticmethod
    def ByWires(externalBoundary, internalBoundaries=None):
        face = Face.ByExternalBoundary(externalBoundary)
        if face is None:
            return None
        face.internals = [w for w in (internalBoundaries or []) if isinstance(w, Wire)]
        return face

    @staticmethod
    def ByVertices(vertices):
        wire = Wire.ByVertices(vertices, close=True)
        if wire is None:
            return None
        return Face.ByWire(wire)

    def ExternalBoundary(self):
        return self.external

    def InternalBoundaries(self, wires=None):
        result = list(getattr(self, "internals", []) or [])
        if wires is not None:
            wires.extend(result)
            return 0
        return result

    def Edges(self, hostTopology=None, edges=None):
        result = []
        if isinstance(self.external, Wire):
            result.extend(self.external.Edges())
        for wire in getattr(self, "internals", []) or []:
            if isinstance(wire, Wire):
                result.extend(wire.Edges())
        result = unique_by_uuid(result)
        if edges is not None:
            edges.extend(result)
            return 0
        return result

    def Vertices(self, hostTopology=None, vertices=None):
        result = []
        for edge in self.Edges():
            result.extend([edge.start, edge.end])
        result = unique_by_uuid([v for v in result if isinstance(v, Vertex)])
        if vertices is not None:
            vertices.extend(result)
            return 0
        return result

    def Wires(self, hostTopology=None, wires=None):
        result = []
        if isinstance(self.external, Wire):
            result.append(self.external)
        result.extend([w for w in getattr(self, "internals", []) or [] if isinstance(w, Wire)])
        if wires is not None:
            wires.extend(result)
            return 0
        return result

    def Faces(self, hostTopology=None, faces=None):
        result = [self]
        if faces is not None:
            faces.extend(result)
            return 0
        return result


class FaceUtility:
    @staticmethod
    def Area(face):
        if not isinstance(face, Face):
            return None
        vertices = face.Vertices()
        if len(vertices) < 3:
            return 0.0
        nx = ny = nz = 0.0
        for i, v in enumerate(vertices):
            w = vertices[(i + 1) % len(vertices)]
            nx += (v.y - w.y) * (v.z + w.z)
            ny += (v.z - w.z) * (v.x + w.x)
            nz += (v.x - w.x) * (v.y + w.y)
        area = 0.5 * math.sqrt(nx * nx + ny * ny + nz * nz)
        for wire in getattr(face, "internals", []) or []:
            if isinstance(wire, Wire):
                tmp_face = Face.ByWire(wire)
                if tmp_face is not None:
                    area -= FaceUtility.Area(tmp_face) or 0.0
        return abs(area)

    @staticmethod
    def NormalAtParameters(face, u=0.5, v=0.5):
        if not isinstance(face, Face):
            return None
        vertices = face.Vertices()
        if len(vertices) < 3:
            return None
        a, b, c = vertices[0], vertices[1], vertices[2]
        ux, uy, uz = b.x - a.x, b.y - a.y, b.z - a.z
        vx, vy, vz = c.x - a.x, c.y - a.y, c.z - a.z
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx
        length = math.sqrt(nx * nx + ny * ny + nz * nz)
        if length == 0:
            return [0, 0, 1]
        return [nx / length, ny / length, nz / length]

    @staticmethod
    def Edges(face):
        if isinstance(face, Face):
            return face.Edges()
        return []

# ---------------------------------------------------------------------------
# Explicit unsupported Face API
# ---------------------------------------------------------------------------
from .helpers import not_implemented as _not_implemented


def _face_not_implemented(name, return_value=None):
    def _method(*args, **kwargs):
        return _not_implemented(f"Face.{name}", return_value)
    return _method


def _face_utility_not_implemented(name, return_value=None):
    def _method(*args, **kwargs):
        return _not_implemented(f"FaceUtility.{name}", return_value)
    return _method


# Hole-bearing OCC faces are not yet built. Face.ByWire and Face.ByVertices are implemented.
Face.ByWires = staticmethod(_face_not_implemented("ByWires"))
Face.ByExternalInternalBoundaries = staticmethod(_face_not_implemented("ByExternalInternalBoundaries"))
Face.InternalVertex = _face_not_implemented("InternalVertex")
FaceUtility.InternalVertex = staticmethod(_face_utility_not_implemented("InternalVertex"))
FaceUtility.VertexAtParameters = staticmethod(_face_utility_not_implemented("VertexAtParameters"))
FaceUtility.ParametersAtVertex = staticmethod(_face_utility_not_implemented("ParametersAtVertex"))
FaceUtility.IsInside = staticmethod(_face_utility_not_implemented("IsInside", False))
FaceUtility.Triangulate = staticmethod(_face_utility_not_implemented("Triangulate", []))
