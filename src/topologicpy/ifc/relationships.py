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

# src/topologicpy/ifc/relationships.py

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional

import ifcopenshell
import ifcopenshell.api


class IFCRelationshipWriter:
    def __init__(self, cfg):
        self.cfg = cfg

    # ----------------------------
    # Public API used by exporter
    # ----------------------------
    def contain_in_storey(self, f: ifcopenshell.file, ctx, ifc_el, spatial) -> None:
        """
        Contain a product in a spatial structure (typically storey).
        """
        rel = self._run_usecase(
            "spatial.assign_container",
            f,
            products=[ifc_el],
            relating_structure=spatial.storey,
        )
        self._assign_owner_history(rel, ctx)

    def postprocess_relationships(
        self,
        f: ifcopenshell.file,
        ctx,
        spatial,
        created: List[Tuple[Dict[str, Any], Any]],
    ) -> None:
        """
        Build relationships that require both sides already created.

        Conventions expected in element records:
          - Openings:
              el["ifc_class"] == "IfcOpeningElement"
              el["host_id"] = "<id of host element>"
          - Doors/Windows:
              el["ifc_class"] in {"IfcDoor","IfcWindow"}
              el["opening_id"] = "<id of opening element>"
        """
        by_id: Dict[str, Any] = {}
        for el, ifc_el in created:
            _id = el.get("id")
            if _id:
                by_id[_id] = ifc_el

        # 1) Host -> Opening (IfcRelVoidsElement)
        for el, ifc_el in created:
            if not ifc_el or not ifc_el.is_a("IfcOpeningElement"):
                continue

            host_id = el.get("host_id")
            host_ifc = by_id.get(host_id) if host_id else None
            if host_ifc is None:
                continue

            rel = self._run_usecase(
                "feature.add_opening",
                f,
                opening=ifc_el,
                element=host_ifc,
            )
            self._assign_owner_history(rel, ctx)

        # 2) Opening -> Filling (IfcRelFillsElement)
        for el, ifc_el in created:
            if not ifc_el or not (ifc_el.is_a("IfcDoor") or ifc_el.is_a("IfcWindow")):
                continue

            opening_id = el.get("opening_id")
            opening_ifc = by_id.get(opening_id) if opening_id else None
            if opening_ifc is None:
                continue

            rel = self._run_usecase(
                "feature.add_filling",
                f,
                opening=opening_ifc,
                element=ifc_el,
            )
            self._assign_owner_history(rel, ctx)

    # ----------------------------
    # Internals
    # ----------------------------
    def _run_usecase(self, usecase: str, f: ifcopenshell.file, **settings) -> Any:
        """
        Run an IfcOpenShell API usecase robustly.
        Some usecases return:
          - the created relationship entity
          - a list of entities
          - None
        We return whatever comes back.
        """
        try:
            return ifcopenshell.api.run(usecase, f, **settings)
        except Exception as e:
            if not getattr(self.cfg, "silent", False):
                print(f"Warning: ifcopenshell.api.run('{usecase}') failed: {e}")
            return None

    def _assign_owner_history(self, obj: Any, ctx) -> None:
        """
        Assign ctx.owner_history to:
          - a single relationship/entity
          - or a list/tuple of entities
        Safely no-ops if not applicable.
        """
        owner_history = getattr(ctx, "owner_history", None)
        if owner_history is None or obj is None:
            return

        if isinstance(obj, (list, tuple)):
            for item in obj:
                self._assign_owner_history(item, ctx)
            return

        # Only IfcRoot has OwnerHistory; relationships are IfcRelationship -> IfcRoot
        if hasattr(obj, "OwnerHistory"):
            try:
                obj.OwnerHistory = owner_history
            except Exception:
                # Keep exporter resilient; don't fail export over metadata.
                if not getattr(self.cfg, "silent", False):
                    print("Warning: Could not assign OwnerHistory on a relationship/entity.")