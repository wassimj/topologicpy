from __future__ import annotations

from dataclasses import dataclass, field
from .topology import Topology
from .cell import Cell
from .helpers import unique_by_uuid


@dataclass(eq=False)
class CellComplex(Topology):
    cells: list = field(default_factory=list)

    @staticmethod
    def ByCells(cells, tolerance=0.0001):
        cells = [c for c in (cells or []) if isinstance(c, Cell)]
        if not cells:
            return None
        return CellComplex(shape=None, cells=cells)

    @staticmethod
    def ByFaces(faces, tolerance=0.0001):
        cell = Cell.ByFaces(faces, tolerance=tolerance)
        if cell is None:
            return None
        return CellComplex.ByCells([cell], tolerance)

    def Cells(self, hostTopology=None, cells=None):
        result = list(getattr(self, "cells", []) or [])
        if cells is not None:
            cells.extend(result)
            return 0
        return result

    def Shells(self, hostTopology=None, shells=None):
        result = []
        for cell in self.Cells():
            result.extend(cell.Shells())
        result = unique_by_uuid(result)
        if shells is not None:
            shells.extend(result)
            return 0
        return result

    def Faces(self, hostTopology=None, faces=None):
        result = []
        for cell in self.Cells():
            result.extend(cell.Faces())
        result = unique_by_uuid(result)
        if faces is not None:
            faces.extend(result)
            return 0
        return result

    def Edges(self, hostTopology=None, edges=None):
        result = []
        for cell in self.Cells():
            result.extend(cell.Edges())
        result = unique_by_uuid(result)
        if edges is not None:
            edges.extend(result)
            return 0
        return result

    def Vertices(self, hostTopology=None, vertices=None):
        result = []
        for cell in self.Cells():
            result.extend(cell.Vertices())
        result = unique_by_uuid(result)
        if vertices is not None:
            vertices.extend(result)
            return 0
        return result

    def CellComplexes(self, hostTopology=None, cellComplexes=None):
        result = [self]
        if cellComplexes is not None:
            cellComplexes.extend(result)
            return 0
        return result

# ---------------------------------------------------------------------------
# Explicit unsupported CellComplex API
# ---------------------------------------------------------------------------
from .helpers import not_implemented as _not_implemented


def _cell_complex_not_implemented(name, return_value=None):
    def _method(*args, **kwargs):
        return _not_implemented(f"CellComplex.{name}", return_value)
    return _method


CellComplex.ByCellsCluster = staticmethod(_cell_complex_not_implemented("ByCellsCluster"))
CellComplex.ExternalBoundary = _cell_complex_not_implemented("ExternalBoundary")
CellComplex.NonManifoldFaces = _cell_complex_not_implemented("NonManifoldFaces", [])
