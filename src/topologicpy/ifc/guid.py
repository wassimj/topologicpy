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

# src/topologicpy/ifc/guid.py

from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional

import ifcopenshell.guid


class GUIDStrategy:
    """
    Deterministic GlobalId generator using a stable element identifier.

    Priority:
      1) el["id"]
      2) hash of (ifc_class, name, tag)
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def guid_for_element(self, el: Dict[str, Any]) -> Optional[str]:
        seed = el.get("id", None)
        if seed is None:
            seed = f'{el.get("ifc_class","")}|{el.get("name","")}|{el.get("tag","")}'
        if not seed:
            return None

        digest = hashlib.sha1(str(seed).encode("utf-8")).hexdigest()[:32]
        # ifcopenshell.guid.compress expects 32 hex chars
        return ifcopenshell.guid.compress(digest)