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

# src/topologicpy/ifc/semantics.py

from __future__ import annotations

from typing import Any, Dict, Optional

import ifcopenshell
import ifcopenshell.api


class IFCSemanticsWriter:
    def __init__(self, cfg):
        self.cfg = cfg

    def assign_semantics(self, f: ifcopenshell.file, ctx, ifc_el, el: Dict[str, Any]) -> None:
        """
        Attach a TopologicPy property set from el["dictionary"] (if present).

        Expects:
          el["dictionary"] -> dict[str, Any]
        """
        d = el.get("dictionary", None)
        if not isinstance(d, dict) or len(d) == 0:
            return

        # Create a pset attached to the product
        pset = self._run_usecase(
            "pset.add_pset",
            f,
            product=ifc_el,
            name="Pset_TopologicPy",
        )
        if pset is None:
            return

        # Stamp OwnerHistory on the Pset (IfcPropertySet is IfcRoot)
        self._assign_owner_history(pset, ctx)

        # Convert values to plain Python scalars (IfcOpenShell will map types)
        # Keep it simple and deterministic.
        props: Dict[str, Any] = {}
        for k, v in d.items():
            if k is None:
                continue
            key = str(k)
            # Skip complex objects; keep RV-friendly values only
            if isinstance(v, (str, int, float, bool)) or v is None:
                props[key] = v
            else:
                # if you want, you can stringify, but for now skip non-scalars
                props[key] = str(v)

        # IMPORTANT: properties must be a dict (NOT "{...}" which is a set)
        self._run_usecase(
            "pset.edit_pset",
            f,
            pset=pset,
            properties=props,
        )

        # Stamp OwnerHistory on the relationship linking product<->pset
        self._stamp_rel_defines_by_properties(f, ctx, ifc_el, pset)

    # ----------------------------
    # Internals
    # ----------------------------
    def _run_usecase(self, usecase: str, f: ifcopenshell.file, **settings) -> Any:
        try:
            return ifcopenshell.api.run(usecase, f, **settings)
        except Exception as e:
            if not getattr(self.cfg, "silent", False):
                print(f"Warning: ifcopenshell.api.run('{usecase}') failed: {e}")
            return None

    def _assign_owner_history(self, obj: Any, ctx) -> None:
        oh = getattr(ctx, "owner_history", None)
        if oh is None or obj is None:
            return
        if hasattr(obj, "OwnerHistory") and obj.OwnerHistory is None:
            try:
                obj.OwnerHistory = oh
            except Exception:
                if not getattr(self.cfg, "silent", False):
                    print("Warning: Could not assign OwnerHistory on an entity.")

    def _stamp_rel_defines_by_properties(self, f: ifcopenshell.file, ctx, product, pset) -> None:
        oh = getattr(ctx, "owner_history", None)
        if oh is None:
            return

        # Build-independent stamping: find any rel that links product to pset
        try:
            rels = f.by_type("IfcRelDefinesByProperties")
        except Exception:
            return

        for rel in rels:
            try:
                if rel.RelatingPropertyDefinition == pset and product in (rel.RelatedObjects or []):
                    if hasattr(rel, "OwnerHistory") and rel.OwnerHistory is None:
                        rel.OwnerHistory = oh
            except Exception:
                continue