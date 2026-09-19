# Copyright (C) 2026
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3.0 of the License, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with this program. If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import math

from .edge import Edge
from .face import Face
from .topology import _is_null_shape, _shape_from_topology


class Grid:
    """Backend-native grid geometry primitives."""

    @staticmethod
    def IsoCurve(face, axis: str, parameter: float):
        """Create an exact normalized constant-U or constant-V surface Edge."""
        if not isinstance(face, Face):
            return None

        axis = str(axis).lower()
        if axis not in ("u", "v"):
            return None

        try:
            parameter = float(parameter)
        except Exception:
            return None

        if not math.isfinite(parameter):
            return None

        parameter = min(1.0, max(0.0, parameter))
        shape = _shape_from_topology(face)
        if _is_null_shape(shape):
            return None

        try:
            from OCC.Core.BRep import BRep_Tool
            from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge

            adaptor = BRepAdaptor_Surface(shape, True)
            u_min = float(adaptor.FirstUParameter())
            u_max = float(adaptor.LastUParameter())
            v_min = float(adaptor.FirstVParameter())
            v_max = float(adaptor.LastVParameter())

            bounds = (u_min, u_max, v_min, v_max)
            if not all(math.isfinite(value) for value in bounds):
                return None

            try:
                surface = BRep_Tool.Surface(shape)
            except Exception:
                surface = BRep_Tool.Surface_s(shape)

            if surface is None:
                return None

            if axis == "u":
                value = u_min + parameter * (u_max - u_min)
                maker = BRepBuilderAPI_MakeEdge(
                    surface.UIso(value),
                    v_min,
                    v_max,
                )
            else:
                value = v_min + parameter * (v_max - v_min)
                maker = BRepBuilderAPI_MakeEdge(
                    surface.VIso(value),
                    u_min,
                    u_max,
                )

            if hasattr(maker, "IsDone") and not maker.IsDone():
                return None

            return Edge.ByOcctShape(maker.Edge())
        except Exception:
            return None
