# Copyright (C) 2026
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free Software
# Foundation, either version 3.0 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along with
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

    ONTOLOGY_TO_IFC = {
        "top:Wall": "IfcWall",
        "top:CurtainWall": "IfcWall",
        "top:Slab": "IfcSlab",
        "top:Roof": "IfcSlab",
        "top:Beam": "IfcBeam",
        "top:Column": "IfcColumn",
        "top:Door": "IfcDoor",
        "top:Window": "IfcWindow",
        "top:Opening": "IfcOpeningElement",
        "top:Aperture": "IfcOpeningElement",
        "top:Space": "IfcSpace",
        "top:Room": "IfcSpace",
        "top:Element": "IfcBuildingElementProxy",
        "top:Equipment": "IfcBuildingElementProxy",
        "top:Furniture": "IfcBuildingElementProxy",
    }

    def __init__(self, cfg):
        self.cfg = cfg
        self.guid = GUIDStrategy(cfg)

    @staticmethod
    def _nested_dictionary(el: Dict[str, Any]) -> Dict[str, Any]:
        d = el.get("dictionary", None) if isinstance(el, dict) else None
        return d if isinstance(d, dict) else {}

    @staticmethod
    def _normalise_ifc_class(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if text == "":
            return None
        key = text.replace("_", "").upper()
        by_upper = {c.upper(): c for c in IFCElementMapper.RV_ALLOWLIST}
        if key in by_upper:
            return by_upper[key]
        if text.upper().startswith("IFC"):
            return "Ifc" + key[3:].lower().title().replace("_", "")
        return text

    def ifc_class_by_ontology_class(self, ontology_class: Any) -> Optional[str]:
        if ontology_class is None:
            return None
        text = str(ontology_class).strip()
        if text.startswith("http://w3id.org/topologicpy#"):
            text = "top:" + text.rsplit("#", 1)[-1]
        return self.ONTOLOGY_TO_IFC.get(text, None)

    def map_ifc_class(self, el: Dict[str, Any]) -> Optional[str]:
        dictionary = self._nested_dictionary(el)

        # Explicit IFC class wins. Accept both current _005 keys and older IFC_*
        # dictionary keys to remain backward-compatible with imported data.
        ifc_class = (
            el.get("ifc_class", None) or el.get("IfcClass", None) or el.get("IFC_class", None) or
            el.get("ifc_type", None) or el.get("IFC_type", None) or
            dictionary.get("ifc_class", None) or dictionary.get("IfcClass", None) or dictionary.get("IFC_class", None) or
            dictionary.get("ifc_type", None) or dictionary.get("IFC_type", None)
        )
        ifc_class = self._normalise_ifc_class(ifc_class)

        # If no explicit IFC class exists, infer it from the canonical
        # TopologicPy ontology class. Brick classes are not used here because
        # they are external semantic refinements, not IFC entity classes.
        if not ifc_class:
            ontology_class = (
                el.get("ontology_class", None) or el.get("topologic_class", None) or
                dictionary.get("ontology_class", None) or dictionary.get("topologic_class", None)
            )
            ifc_class = self.ifc_class_by_ontology_class(ontology_class)

        if not ifc_class:
            return None
        if ifc_class not in self.RV_ALLOWLIST:
            return None
        return ifc_class

    def create_ifc_element(self, f: ifcopenshell.file, ctx, ifc_class: str, el: Dict[str, Any]):
        dictionary = self._nested_dictionary(el)
        name = el.get("name", None) or dictionary.get("name", None) or dictionary.get("label", None) or ifc_class
        tag = el.get("tag", None) or dictionary.get("tag", None)
        predefined_type = el.get("predefined_type", None) or dictionary.get("predefined_type", None) or dictionary.get("ifc_predefined_type", None)

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