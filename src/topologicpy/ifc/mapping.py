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

# src/topologicpy/ifc/mapping.py

from __future__ import annotations

from typing import Any, Dict, Optional

import ifcopenshell
import ifcopenshell.api

from .guid import GUIDStrategy


class IFCElementMapper:
    """
    Minimal mapper: expects each element dict to declare its IFC class.
    You will likely replace/extend this with TopologicPy BIM-class logic.
    """

    RV_ALLOWLIST = {
        "IfcWall",
        "IfcSlab",
        "IfcBeam",
        "IfcColumn",
        "IfcDoor",
        "IfcWindow",
        "IfcOpeningElement",
        "IfcSpace",
        "IfcBuildingElementProxy",  # last resort
    }

    def __init__(self, cfg):
        self.cfg = cfg
        self.guid = GUIDStrategy(cfg)

    def map_ifc_class(self, el: Dict[str, Any]) -> Optional[str]:
        ifc_class = el.get("ifc_class", None)
        if not ifc_class:
            return None
        if ifc_class not in self.RV_ALLOWLIST:
            # Fail fast? For starter: skip.
            return None
        return ifc_class

    def create_ifc_element(self, f: ifcopenshell.file, ctx, ifc_class: str, el: Dict[str, Any]):
        name = el.get("name", ifc_class)
        tag = el.get("tag", None)
        predefined_type = el.get("predefined_type", None)

        # Deterministic GUID (recommended)
        guid = self.guid.guid_for_element(el)

        ifc_el = ifcopenshell.api.run(
            "root.create_entity", f,
            ifc_class=ifc_class,
            name=name
        )

        # Assign owner history to the element
        if ctx.owner_history is not None and hasattr(ifc_el, "OwnerHistory"):
            ifc_el.OwnerHistory = ctx.owner_history

        # Set GlobalId deterministically if possible
        if guid:
            ifc_el.GlobalId = guid

        if tag is not None and hasattr(ifc_el, "Tag"):
            ifc_el.Tag = str(tag)

        # PredefinedType if schema supports it on the class
        if predefined_type and hasattr(ifc_el, "PredefinedType"):
            try:
                ifc_el.PredefinedType = predefined_type
            except Exception:
                # ignore invalid enum in skeleton
                pass

        return ifc_el