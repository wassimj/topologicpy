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

# src/topologicpy/ifc/validation.py

from __future__ import annotations

from typing import Optional

class IFCValidator:
    def __init__(self, cfg):
        self.cfg = cfg

    def validate(self, path: str) -> None:
        """
        Starter validation hook.

        Keep it simple here: you can later integrate IfcOpenShell validation / IDS checks
        as a separate, testable pipeline.
        """
        if self.cfg.silent:
            return
        # Placeholder: you can integrate ifcopenshell.validate or IfcTester later.
        # For now we do nothing to keep this skeleton dependency-light.
        return