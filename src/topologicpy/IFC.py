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

# src/topologicpy/IFC.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from topologicpy.ifc.exporter import IFCReferenceViewExporter, IFCExportConfig


class IFC:
    """
    TopologicPy → IFC export façade.

    Notes
    -----
    - This module intentionally targets IFC4 Reference View workflows.
    - The public API is a stable façade; internal modules can evolve.
    """

    @staticmethod
    def ExportReferenceView(
        model: Any,
        path: str,
        schema: str = "IFC4",
        mvd: str = "RV1.2",
        project_name: str = "TopologicPy Project",
        site_name: str = "Default Site",
        building_name: str = "Default Building",
        storey_name: str = "Level 0",
        length_unit: str = "METRE",
        angle_unit: str = "RADIAN",
        area_unit: str = "SQUARE_METRE",
        volume_unit: str = "CUBIC_METRE",
        use_sweeps: bool = True,
        use_tessellation_fallback: bool = True,
        transfer_dictionaries: bool = True,
        validate: bool = True,
        silent: bool = False,
    ) -> str:
        """
        Export a TopologicPy-authored model to an IFC4 Reference View-compatible IFC.

        Parameters
        ----------
        model : object
            A TopologicPy structure representing a BIM model. For this starter skeleton,
            this is expected to be either:
            - a TopologicPy Graph that contains "element nodes" with dictionaries, or
            - a list/dict of element records you pass through your own adapter.
        path : str
            Output IFC file path.
        schema : str, optional
            IFC schema identifier (default "IFC4").
        mvd : str, optional
            Model View Definition label (default "RV1.2"). Used for intent/metadata.
        project_name, site_name, building_name, storey_name : str
            Spatial structure names.
        length_unit, angle_unit, area_unit, volume_unit : str
            IFC units.
        use_sweeps : bool
            Attempt swept solid representations where possible.
        use_tessellation_fallback : bool
            If sweeps fail / are unavailable, tessellate geometry.
        transfer_dictionaries : bool
            Transfer Topologic dictionaries into IFC properties (simple mapping).
        validate : bool
            Run basic IFC validation hooks after writing.
        silent : bool
            If True, suppress warnings.

        Returns
        -------
        str
            The output path.
        """
        cfg = IFCExportConfig(
            schema=schema,
            mvd=mvd,
            project_name=project_name,
            site_name=site_name,
            building_name=building_name,
            storey_name=storey_name,
            units={
                "length": length_unit,
                "angle": angle_unit,
                "area": area_unit,
                "volume": volume_unit,
            },
            use_sweeps=use_sweeps,
            use_tessellation_fallback=use_tessellation_fallback,
            transfer_dictionaries=transfer_dictionaries,
            validate=validate,
            silent=silent,
        )

        exporter = IFCReferenceViewExporter(cfg)
        exporter.export(model=model, path=path)
        return path