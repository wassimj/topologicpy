from __future__ import annotations

from dataclasses import dataclass
from .topology import Topology
from .face import Face, FaceUtility
from .edge import Edge
from .vertex import Vertex
from .occ_utils import make_occ_shell
from .helpers import unique_by_uuid


@dataclass(eq=False)
class Shell(Topology):
    def __init__(self, shape=None, dictionary=None, contents=None, contexts=None, apertures=None, faces=None):
        super().__init__(shape=shape, dictionary=dictionary, contents=contents, contexts=contexts, apertures=apertures)
        self.faces = list(faces) if faces else []

    @staticmethod
    def ByFaces(faces, tolerance: float = 0.0001, silent: bool = False):
        if faces is None:
            if not silent:
                print("Shell.ByFaces - Error: The input faces parameter is None. Returning None.")
            return None
        if not isinstance(faces, list):
            faces = [faces]
        valid_faces = [face for face in faces if Topology.IsInstance(face, "Face")]
        if len(valid_faces) == 0:
            if not silent:
                print("Shell.ByFaces - Error: The input faces list does not contain any valid faces. Returning None.")
            return None
        occ_shell = make_occ_shell(valid_faces)
        if occ_shell is None:
            if not silent:
                print("Shell.ByFaces - Error: Could not create an OpenCascade shell. Returning None.")
            return None
        return Shell(shape=occ_shell, faces=valid_faces)

    def Faces(self, hostTopology=None, faces=None):
        result = list(getattr(self, "faces", []) or [])
        if faces is not None:
            faces.extend(result)
            return 0
        return result

    def Edges(self, hostTopology=None, edges=None):
        result = []
        for face in getattr(self, "faces", []) or []:
            if isinstance(face, Face):
                result.extend(face.Edges())
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

    def Shells(self, hostTopology=None, shells=None):
        result = [self]
        if shells is not None:
            shells.extend(result)
            return 0
        return result


class ShellUtility:
    @staticmethod
    def Area(shell):
        if not isinstance(shell, Shell):
            return None
        return sum(FaceUtility.Area(f) or 0.0 for f in shell.faces)

# ---------------------------------------------------------------------------
# Explicit unsupported Shell API
# ---------------------------------------------------------------------------
from .helpers import not_implemented as _not_implemented


def _shell_not_implemented(name, return_value=None):
    def _method(*args, **kwargs):
        return _not_implemented(f"Shell.{name}", return_value)
    return _method


def _shell_utility_not_implemented(name, return_value=None):
    def _method(*args, **kwargs):
        return _not_implemented(f"ShellUtility.{name}", return_value)
    return _method


Shell.ByWires = staticmethod(_shell_not_implemented("ByWires"))
Shell.ExternalBoundary = _shell_not_implemented("ExternalBoundary")
ShellUtility.ExternalBoundary = staticmethod(_shell_utility_not_implemented("ExternalBoundary"))
ShellUtility.InternalBoundaries = staticmethod(_shell_utility_not_implemented("InternalBoundaries", []))
