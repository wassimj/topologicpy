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

# src/topologicpy/ifc/types.py
#
# Minimal, deterministic typing for IFC4 RV:
# - Create IfcWallType (and others) once per "type key"
# - Assign occurrences via IfcRelDefinesByType
# - Reuse ctx.owner_history everywhere
#
# No reliance on ifcopenshell.api.type.* (varies by build); uses direct entity creation.

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import ifcopenshell
import ifcopenshell.guid


class IFCTypesWriter:
    def __init__(self, cfg):
        self.cfg = cfg
        # cache: (ifc_type_class, type_name) -> IfcTypeObject
        self._cache: Dict[Tuple[str, str], Any] = {}

    def assign_type(self, f: ifcopenshell.file, ctx, ifc_el: Any, el: Dict[str, Any]) -> None:
        """
        Assign an IfcTypeObject to the occurrence if applicable.

        Element dict conventions (optional):
          - el["type_name"] : str  (e.g. "Basic Wall 200mm")
          - el["type_id"]   : str  (stable key; if absent, type_name used)
          - el["predefined_type"] : str (e.g. "NOTDEFINED") for IFC enum if supported
        """
        if ifc_el is None or not hasattr(ifc_el, "is_a"):
            return

        occ_class = ifc_el.is_a()
        type_class = self._occurrence_to_type_class(occ_class)
        if type_class is None:
            return

        # Determine a type name/key
        type_name = el.get("type_name")
        if not type_name:
            # default: one type per IFC class
            type_name = f"{occ_class} Type"

        # Optional: stable cache key separate from display name
        type_id = el.get("type_id") or type_name

        type_obj = self._get_or_create_type(f, ctx, type_class, type_id, type_name, el)

        # Create relationship: IfcRelDefinesByType
        # Avoid duplicates: if already typed, do nothing.
        if getattr(ifc_el, "IsTypedBy", None):
            try:
                for rel in ifc_el.IsTypedBy:
                    if rel and getattr(rel, "RelatingType", None) == type_obj:
                        return
            except Exception:
                pass

        rel = f.createIfcRelDefinesByType(
            ifcopenshell.guid.new(),
            getattr(ctx, "owner_history", None),
            None,
            None,
            [ifc_el],
            type_obj,
        )

        # Stamp OwnerHistory (some builds may not set it in createIfcRelDefinesByType)
        self._assign_owner_history(rel, ctx)

    # ----------------------------
    # Internals
    # ----------------------------
    def _occurrence_to_type_class(self, occ_class: str) -> Optional[str]:
        """
        Map an Ifc* occurrence class to a corresponding Ifc*Type class.
        Extend this as you add more element categories.
        """
        mapping = {
            "IfcWall": "IfcWallType",
            "IfcDoor": "IfcDoorType",
            "IfcWindow": "IfcWindowType",
            "IfcSlab": "IfcSlabType",
            "IfcColumn": "IfcColumnType",
            "IfcBeam": "IfcBeamType",
        }
        return mapping.get(occ_class)

    def _get_or_create_type(
        self,
        f: ifcopenshell.file,
        ctx,
        type_class: str,
        type_id: str,
        type_name: str,
        el: Dict[str, Any],
    ) -> Any:
        key = (type_class, str(type_id))
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        oh = getattr(ctx, "owner_history", None)

        # Minimal IFC4 TypeObject creation. Use enum NOTDEFINED where possible.
        predefined = el.get("predefined_type", "NOTDEFINED")

        if type_class == "IfcWallType":
            # IFC4: IfcWallType(GlobalId, OwnerHistory, Name, Description, ApplicableOccurrence,
            #                   HasPropertySets, RepresentationMaps, Tag, ElementType, PredefinedType)
            type_obj = f.createIfcWallType(
                ifcopenshell.guid.new(),
                oh,
                str(type_name),
                None,
                None,
                None,
                None,
                None,
                str(type_name),   # ElementType (string label)
                predefined,        # IfcWallTypeEnum (string)
            )
        elif type_class == "IfcDoorType":
            type_obj = f.createIfcDoorType(
                ifcopenshell.guid.new(), oh, str(type_name), None, None, None, None, None, str(type_name), predefined
            )
        elif type_class == "IfcWindowType":
            type_obj = f.createIfcWindowType(
                ifcopenshell.guid.new(), oh, str(type_name), None, None, None, None, None, str(type_name), predefined
            )
        elif type_class == "IfcSlabType":
            type_obj = f.createIfcSlabType(
                ifcopenshell.guid.new(), oh, str(type_name), None, None, None, None, None, str(type_name), predefined
            )
        elif type_class == "IfcColumnType":
            type_obj = f.createIfcColumnType(
                ifcopenshell.guid.new(), oh, str(type_name), None, None, None, None, None, str(type_name), predefined
            )
        elif type_class == "IfcBeamType":
            type_obj = f.createIfcBeamType(
                ifcopenshell.guid.new(), oh, str(type_name), None, None, None, None, None, str(type_name), predefined
            )
        else:
            # Fallback: try generic creation if you later extend mapping
            type_obj = ifcopenshell.api.run("root.create_entity", f, ifc_class=type_class, name=str(type_name))
            self._assign_owner_history(type_obj, ctx)

        # Ensure OwnerHistory present (in case any branch didn't set)
        self._assign_owner_history(type_obj, ctx)

        self._cache[key] = type_obj
        return type_obj

    def _assign_owner_history(self, obj: Any, ctx) -> None:
        oh = getattr(ctx, "owner_history", None)
        if oh is None or obj is None:
            return
        if hasattr(obj, "OwnerHistory") and obj.OwnerHistory is None:
            try:
                obj.OwnerHistory = oh
            except Exception:
                if not getattr(self.cfg, "silent", False):
                    print("Warning: Could not assign OwnerHistory on a type/relationship entity.")