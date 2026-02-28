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

# src/topologicpy/ifc/spatial.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import ifcopenshell
import ifcopenshell.api


@dataclass
class IFCSpatial:
    site: Any
    building: Any
    storey: Any


class IFCSpatialBuilder:
    def __init__(self, cfg):
        self.cfg = cfg

    def build(self, f: ifcopenshell.file, ctx) -> IFCSpatial:
        site = ifcopenshell.api.run(
            "root.create_entity", f,
            ifc_class="IfcSite",
            name=self.cfg.site_name
        )
        building = ifcopenshell.api.run(
            "root.create_entity", f,
            ifc_class="IfcBuilding",
            name=self.cfg.building_name
        )
        storey = ifcopenshell.api.run(
            "root.create_entity", f,
            ifc_class="IfcBuildingStorey",
            name=self.cfg.storey_name
        )

        # Assign owner history
        if ctx.owner_history is not None:
            for obj in (site, building, storey):
                if hasattr(obj, "OwnerHistory"):
                    obj.OwnerHistory = ctx.owner_history

        # Aggregation: Project → Site → Building → Storey
        ifcopenshell.api.run("aggregate.assign_object", f, relating_object=ctx.project, products=[site])
        ifcopenshell.api.run("aggregate.assign_object", f, relating_object=site, products=[building])
        ifcopenshell.api.run("aggregate.assign_object", f, relating_object=building, products=[storey])

        # Stamp OwnerHistory on the IfcRelAggregates created above (IfcOpenShell may return None)
        if ctx.owner_history is not None:
            # Project -> Site
            for rel in f.by_type("IfcRelAggregates"):
                if rel.RelatingObject == ctx.project and site in (rel.RelatedObjects or []):
                    rel.OwnerHistory = ctx.owner_history

            # Site -> Building
            for rel in f.by_type("IfcRelAggregates"):
                if rel.RelatingObject == site and building in (rel.RelatedObjects or []):
                    rel.OwnerHistory = ctx.owner_history

            # Building -> Storey
            for rel in f.by_type("IfcRelAggregates"):
                if rel.RelatingObject == building and storey in (rel.RelatedObjects or []):
                    rel.OwnerHistory = ctx.owner_history

        # IMPORTANT: Ensure placements exist without relying on geometry.add_local_placement
        # Site at world origin
        site.ObjectPlacement = f.createIfcLocalPlacement(None, self._axis2placement3d_identity(f))
        # Building relative to site
        building.ObjectPlacement = f.createIfcLocalPlacement(site.ObjectPlacement, self._axis2placement3d_identity(f))
        # Storey relative to building
        storey.ObjectPlacement = f.createIfcLocalPlacement(building.ObjectPlacement, self._axis2placement3d_identity(f))

        return IFCSpatial(site=site, building=building, storey=storey)

    def _axis2placement3d_identity(self, f: ifcopenshell.file):
        origin = f.createIfcCartesianPoint((0.0, 0.0, 0.0))
        zdir = f.createIfcDirection((0.0, 0.0, 1.0))
        xdir = f.createIfcDirection((1.0, 0.0, 0.0))
        return f.createIfcAxis2Placement3D(origin, zdir, xdir)