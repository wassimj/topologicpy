# Copyright (C) 2026
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3.0 of the License, or (at your option)
# any later version.

"""TopologicPy persistence and neutral CAD interchange codecs."""

from __future__ import annotations

import os
from typing import Dict


_CODECS: Dict[str, type] = {}


def register_codec(extension: str, codec: type) -> None:
    """Register a codec class for a filename extension."""
    extension = str(
        extension
    ).strip().lower()

    if not extension.startswith("."):
        extension = "." + extension

    _CODECS[
        extension
    ] = codec


def codec_for_path(path):
    """Return the registered codec class for *path*, or None."""
    try:
        extension = os.path.splitext(
            os.fspath(path)
        )[1].lower()
    except Exception:
        return None

    if not extension:
        return None

    return _CODECS.get(
        extension
    )


from .step import STEPCodec  # noqa: E402
from .tpy import TPYCodec  # noqa: E402

register_codec(
    ".step",
    STEPCodec,
)
register_codec(
    ".stp",
    STEPCodec,
)
register_codec(
    ".tpy",
    TPYCodec,
)

__all__ = [
    "STEPCodec",
    "TPYCodec",
    "codec_for_path",
    "register_codec",
]
