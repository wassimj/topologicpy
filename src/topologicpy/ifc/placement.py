# Copyright (C) 2026
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

# src/topologicpy/ifc/placement.py

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import ifcopenshell


class IFCPlacementWriter:
    def __init__(self, cfg):
        self.cfg = cfg

    def assign_placement(self, f: ifcopenshell.file, ctx, ifc_el, spatial, el: Dict[str, Any]) -> None:
        """
        Assign an IfcLocalPlacement to `ifc_el` relative to the storey placement.

        This implementation does NOT use ifcopenshell.api geometry placement usecases,
        because some IfcOpenShell builds don't include them.

        If el["transform"] is provided, it should be a 4x4 nested list:
            [[r11,r12,r13,tx],
             [r21,r22,r23,ty],
             [r31,r32,r33,tz],
             [0,  0,  0,  1 ]]
        and is interpreted as a LOCAL transform relative to the storey.
        """
        # Ensure spatial placements exist (spatial.py should already create these,
        # but we keep this resilient).
        self._ensure_spatial_placements(f, spatial)

        # Local transform (optional)
        m = el.get("transform", None)
        axis2 = self._axis2placement3d_from_matrix(f, m) if m else self._axis2placement3d_identity(f)

        # Element placement relative to storey
        ifc_el.ObjectPlacement = f.createIfcLocalPlacement(spatial.storey.ObjectPlacement, axis2)

    # ------------------------
    # Placement constructors
    # ------------------------
    def _axis2placement3d_identity(self, f: ifcopenshell.file):
        origin = f.createIfcCartesianPoint((0.0, 0.0, 0.0))
        zdir = f.createIfcDirection((0.0, 0.0, 1.0))
        xdir = f.createIfcDirection((1.0, 0.0, 0.0))
        return f.createIfcAxis2Placement3D(origin, zdir, xdir)

    def _axis2placement3d_from_matrix(self, f: ifcopenshell.file, m: List[List[float]]):
        """
        Build IfcAxis2Placement3D from a 4x4 matrix.

        - Location = (tx, ty, tz)
        - Axis (local Z) = third column (r13,r23,r33) if matrix is column-major for basis,
          but most 4x4 transforms here are row-major; we interpret as:
             X axis = first row basis? No: In common IFC/graphics, basis vectors are columns.
          To avoid ambiguity, we take:
             X axis = (m[0][0], m[1][0], m[2][0])  (column 0)
             Z axis = (m[0][2], m[1][2], m[2][2])  (column 2)
          Translation = (m[0][3], m[1][3], m[2][3])

        If your TopologicPy matrices are row-basis instead of column-basis,
        swap accordingly (but keep this consistent across exporter).
        """
        try:
            tx, ty, tz = float(m[0][3]), float(m[1][3]), float(m[2][3])

            # Basis vectors as COLUMNS
            x = (float(m[0][0]), float(m[1][0]), float(m[2][0]))
            z = (float(m[0][2]), float(m[1][2]), float(m[2][2]))

            origin = f.createIfcCartesianPoint((tx, ty, tz))
            zdir = self._safe_dir(f, z, fallback=(0.0, 0.0, 1.0))
            xdir = self._safe_dir(f, x, fallback=(1.0, 0.0, 0.0))
            return f.createIfcAxis2Placement3D(origin, zdir, xdir)
        except Exception:
            # Fail safe: identity
            if not self.cfg.silent:
                print("Warning: invalid transform matrix for placement; using identity.")
            return self._axis2placement3d_identity(f)

    def _safe_dir(self, f: ifcopenshell.file, v: Tuple[float, float, float], fallback=(1.0, 0.0, 0.0)):
        # Minimal normalization with safety
        x, y, z = v
        n2 = x * x + y * y + z * z
        if n2 <= 1e-18:
            x, y, z = fallback
            n2 = x * x + y * y + z * z
        inv = (n2 ** 0.5)
        return f.createIfcDirection((x / inv, y / inv, z / inv))

    def _ensure_spatial_placements(self, f: ifcopenshell.file, spatial) -> None:
        """
        Make sure site/building/storey have ObjectPlacement so relative placements work.
        """
        if spatial.site.ObjectPlacement is None:
            spatial.site.ObjectPlacement = f.createIfcLocalPlacement(
                None,
                self._axis2placement3d_identity(f),
            )
        if spatial.building.ObjectPlacement is None:
            spatial.building.ObjectPlacement = f.createIfcLocalPlacement(
                spatial.site.ObjectPlacement,
                self._axis2placement3d_identity(f),
            )
        if spatial.storey.ObjectPlacement is None:
            spatial.storey.ObjectPlacement = f.createIfcLocalPlacement(
                spatial.building.ObjectPlacement,
                self._axis2placement3d_identity(f),
            )