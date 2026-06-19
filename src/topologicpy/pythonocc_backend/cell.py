from __future__ import annotations

from dataclasses import dataclass
from .topology import Topology
from .shell import Shell
from .face import FaceUtility
from .wire import Wire
from .edge import Edge
from .vertex import Vertex
from .occ_utils import make_occ_cell
from .helpers import edge_key, unique_by_uuid


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
        for face in self.Faces():
            result.extend(FaceUtility.Edges(face) or [])
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

    def Cells(self, hostTopology=None, cells=None):
        result = [self]
        if cells is not None:
            cells.extend(result)
            return 0
        return result


class CellUtility:
    @staticmethod
    def Volume(cell):
        return None

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


Cell.ByBox = staticmethod(_cell_not_implemented("ByBox"))
Cell.ByWires = staticmethod(_cell_not_implemented("ByWires"))
Cell.InternalVertex = _cell_not_implemented("InternalVertex")
CellUtility.Volume = staticmethod(_cell_utility_not_implemented("Volume"))
CellUtility.InternalVertex = staticmethod(_cell_utility_not_implemented("InternalVertex"))
CellUtility.Contains = staticmethod(_cell_utility_not_implemented("Contains", False))
