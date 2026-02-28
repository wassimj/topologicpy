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

# src/topologicpy/ifc/exporter.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import ifcopenshell

from .context import IFCContextBuilder
from .spatial import IFCSpatialBuilder
from .mapping import IFCElementMapper
from .placement import IFCPlacementWriter
from .geometry import IFCGeometryBuilder
from .relationships import IFCRelationshipWriter
from .semantics import IFCSemanticsWriter
from .validation import IFCValidator
from .types import IFCTypesWriter


from topologicpy.Helper import Helper

@dataclass(frozen=True)
class IFCExportConfig:
    schema: str = "IFC4"
    mvd: str = "RV1.2"
    project_name: str = "TopologicPy Project"
    site_name: str = "Default Site"
    building_name: str = "Default Building"
    storey_name: str = "Level 0"
    units: Dict[str, str] = None  # {"length": "...", "angle": "...", ...}
    use_sweeps: bool = True
    use_tessellation_fallback: bool = True
    transfer_dictionaries: bool = True
    validate: bool = True
    silent: bool = False

    def __post_init__(self):
        if self.units is None:
            object.__setattr__(self, "units", {
                "length": "METRE",
                "area": "SQUARE_METRE",
                "volume": "CUBIC_METRE",
            })


class IFCReferenceViewExporter:
    """
    IFC4 Reference View exporter (starter skeleton).

    The exporter expects you to provide a model that can be iterated to yield "elements".
    In this skeleton, element extraction is kept minimal: provide your own adapter later.
    """

    def __init__(self, config: IFCExportConfig):
        self.cfg = config
        # staged writers/builders
        self.context_builder = IFCContextBuilder(self.cfg)
        self.spatial_builder = IFCSpatialBuilder(self.cfg)
        self.mapper = IFCElementMapper(self.cfg)
        self.placement_writer = IFCPlacementWriter(self.cfg)
        self.geometry_builder = IFCGeometryBuilder(self.cfg)
        self.semantics_writer = IFCSemanticsWriter(self.cfg)
        self.types_writer = IFCTypesWriter(self.cfg)
        self.relationship_writer = IFCRelationshipWriter(self.cfg)
        self.validator = IFCValidator(self.cfg)

    def export(self, model: Any, path: str) -> None:
        """
        - builds IFC4 RV structure + elements (same as before)
        - sets header metadata correctly (no char-splitting in FILE_NAME)
        - avoids conflicting/duplicate header assignments
        """
        # --- Versions (deterministic; no network) ---
        try:
            from topologicpy.Helper import Helper
            tp_version = Helper.Version(check=False)
        except Exception:
            tp_version = "unknown"

        try:
            import ifcopenshell
            engine_version = getattr(ifcopenshell, "__version__", "unknown")
        except Exception:
            # If this fails, we still proceed (but file creation will likely fail anyway)
            engine_version = "unknown"
            import ifcopenshell  # may raise; let it raise

        f = ifcopenshell.file(schema=self.cfg.schema)

        # 1) contexts (project, units, geometric contexts, owner history)
        ctx = self.context_builder.build(f)

        # 2) spatial structure (site/building/storey)
        spatial = self.spatial_builder.build(f, ctx)

        # 3) extract elements
        elements = self._extract_elements(model)

        # 4) create IFC elements
        created = []
        for el in elements:
            ifc_class = self.mapper.map_ifc_class(el)
            if ifc_class is None:
                continue

            ifc_el = self.mapper.create_ifc_element(f, ctx, ifc_class, el)

            # placement
            self.placement_writer.assign_placement(f, ctx, ifc_el, spatial, el)

            # geometry
            self.geometry_builder.assign_representation(f, ctx, ifc_el, el)

            # semantics
            self.semantics_writer.assign_semantics(f, ctx, ifc_el, el)

            # types
            self.types_writer.assign_type(f, ctx, ifc_el, el)

            # containment
            self.relationship_writer.contain_in_storey(f, ctx, ifc_el, spatial)

            created.append((el, ifc_el))

        # 5) element relationships requiring both sides (voids/fills etc.)
        self.relationship_writer.postprocess_relationships(f, ctx, spatial, created)

        # 6) Set Reference View in header
        f.header.file_description.description = ("ViewDefinition [ReferenceView]",)
        f.header.file_description.implementation_level = "2;1"

        # 7) FILE_NAME metadata (IMPORTANT: author/organization MUST be tuple-of-strings)
        f.header.file_name.author = ("TopologicPy",)
        f.header.file_name.organization = ("Syntopy.io",)

        # Brand TopologicPy while still acknowledging the engine
        f.header.file_name.originating_system = f"TopologicPy {tp_version}"
        f.header.file_name.preprocessor_version = f"TopologicPy {tp_version} (IfcOpenShell {engine_version})"

        # 8) write
        f.write(path)

        # 9) validate (optional)
        if getattr(self.cfg, "validate", False):
            self.validator.validate(path)

    # -------------------------
    # Minimal element extraction
    # -------------------------
    def _extract_elements(self, model: Any) -> List[Dict[str, Any]]:
        """
        Convert input `model` into a list of element records.

        Expected element record fields (starter):
          - "ifc_class": str (e.g., "IfcWall", "IfcSlab", "IfcDoor", ...)
          - "name": str
          - "tag": str
          - "predefined_type": str (optional)
          - "topology": Topologic topology object (Face/Shell/Cell/Cluster)
          - "transform": 4x4 list (optional)
          - "host_id": str (for doors/windows/openings) (optional)
          - "opening_id": str (optional)
          - "id": stable id string (optional but recommended)
          - "dictionary": Topologic dictionary or python dict (optional)
        """
        # You will likely replace this with a Graph adapter.
        if model is None:
            return []

        if isinstance(model, list):
            return model

        if isinstance(model, dict) and "elements" in model:
            return list(model["elements"])

        # Fallback: treat as single element record
        return [model]