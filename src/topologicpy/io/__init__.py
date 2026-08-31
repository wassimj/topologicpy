# Copyright (C) 2026
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3.0 of the License, or (at your option)
# any later version.

"""TopologicPy serialization and exchange codecs.

This package deliberately separates native persistence from CAD and mesh
exchange. Codecs register themselves by filename extension. The first native
codec is ``.tpy``; STEP/3DM/OBJ/etc. can be added without growing Topology.py.
"""

from __future__ import annotations

import os
from typing import Dict, Type

_CODECS: Dict[str, type] = {}


def register_codec(extension: str, codec: type) -> None:
    """Register a codec class for a filename extension."""
    extension = str(extension).strip().lower()
    if not extension.startswith("."):
        extension = "." + extension
    _CODECS[extension] = codec


def codec_for_path(path):
    """Return the registered codec class for *path*, or None."""
    try:
        extension = os.path.splitext(os.fspath(path))[1].lower()
    except Exception:
        return None
    if not extension:
        extension = ".tpy"
    return _CODECS.get(extension)


from .tpy import TPYCodec  # noqa: E402

register_codec(".tpy", TPYCodec)

__all__ = ["TPYCodec", "codec_for_path", "register_codec"]
