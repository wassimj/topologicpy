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

# src/topologicpy/ifc/geometry.py

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import ifcopenshell
import ifcopenshell.api


class IFCGeometryBuilder:
    def __init__(self, cfg):
        self.cfg = cfg

    def assign_representation(self, f: ifcopenshell.file, ctx, ifc_el, el: Dict[str, Any]) -> None:
        """
        Strategy:
        1) If use_sweeps and element provides profile+height -> SweptSolid
        2) Else if use_tessellation_fallback and element provides mesh -> Tessellation
        3) Else leave Representation unset
        """

        # 1) Sweep
        if self.cfg.use_sweeps:
            rep = self._try_swept_representation(f, ctx, ifc_el, el)
            if rep is not None:
                ifc_el.Representation = rep
                return

        # 2) Tessellation fallback (mesh-driven; no topology required)
        if self.cfg.use_tessellation_fallback:
            rep = self._tessellate_representation(f, ctx, ifc_el, el)
            if rep is not None:
                ifc_el.Representation = rep
                return

    def _try_swept_representation(self, f, ctx, ifc_el, el: Dict[str, Any]):
        """
        Minimal placeholder:
        Expect:
          - el["profile"]: list of (x,y) closed polyline
          - el["height"]: float
        """
        profile = el.get("profile", None)
        height = el.get("height", None)
        if not profile or height is None:
            return None

        # Convert 2D polyline to IfcArbitraryClosedProfileDef
        pts = [f.createIfcCartesianPoint((float(x), float(y))) for (x, y) in profile]
        poly = f.createIfcPolyline(pts + [pts[0]])
        prof = f.createIfcArbitraryClosedProfileDef("AREA", None, poly)

        # Extrude along +Z
        direction = f.createIfcDirection((0.0, 0.0, 1.0))
        origin = f.createIfcCartesianPoint((0.0, 0.0, 0.0))
        axis2 = f.createIfcAxis2Placement3D(origin, None, None)
        solid = f.createIfcExtrudedAreaSolid(prof, axis2, direction, float(height))

        shape_rep = f.createIfcShapeRepresentation(
            ctx.body_context,
            "Body",
            "SweptSolid",
            [solid]
        )
        prod_shape = f.createIfcProductDefinitionShape(None, None, [shape_rep])
        return prod_shape

    def _tessellate_representation(self, f, ctx, ifc_el, el: Dict[str, Any]):
        """
        CAD Assistant-friendly: write a triangle mesh as IfcFacetedBrep (BRep),
        not IfcTriangulatedFaceSet.

        Expect:
        el["mesh"] = {"vertices":[(x,y,z),...], "faces":[(i,j,k),...]}  # 0-based indices
        """
        mesh = el.get("mesh", None)
        if not mesh:
            return None

        verts = mesh.get("vertices", [])
        faces = mesh.get("faces", [])
        if not verts or not faces:
            return None

        n = len(verts)

        # Validate indices (and auto-fix 1-based faces if detected)
        try:
            min_idx = min(min(int(i) for i in tri) for tri in faces)
            max_idx = max(max(int(i) for i in tri) for tri in faces)
        except Exception:
            return None

        # If faces look 1-based and fit, convert to 0-based
        if min_idx >= 1 and max_idx <= n:
            faces = [(a - 1, b - 1, c - 1) for (a, b, c) in faces]
            min_idx = min(min(tri) for tri in faces)
            max_idx = max(max(tri) for tri in faces)

        if min_idx < 0 or max_idx >= n:
            raise ValueError(f"Invalid mesh indices: faces reference [{min_idx}..{max_idx}] but vertices={n}.")

        # Create reusable IfcCartesianPoint entities
        pts = [f.createIfcCartesianPoint(tuple(map(float, v))) for v in verts]

        # Build faces: each triangle -> IfcFace(IfcFaceOuterBound(IfcPolyLoop([...]), True))
        ifc_faces = []
        for (a, b, c) in faces:
            loop = f.createIfcPolyLoop([pts[a], pts[b], pts[c]])
            outer = f.createIfcFaceOuterBound(loop, True)
            ifc_face = f.createIfcFace([outer])
            ifc_faces.append(ifc_face)

        shell = f.createIfcClosedShell(ifc_faces)
        brep = f.createIfcFacetedBrep(shell)

        shape_rep = f.createIfcShapeRepresentation(
            ctx.body_context,
            "Body",
            "Brep",
            [brep]
        )
        return f.createIfcProductDefinitionShape(None, None, [shape_rep])