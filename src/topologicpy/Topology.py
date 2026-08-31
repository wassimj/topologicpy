# Copyright (C) 2026
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3.0 of the License, or (at your option)
# any later version.

"""Public Topology router."""

from topologicpy.Core import Core

_backend_name = Core.Backend().__class__.__name__.lower()

if "pythonocc" in _backend_name:
    from topologicpy.TopologyPythonOCC import Topology
else:
    from topologicpy.TopologyTopologicCore import Topology

__all__ = ["Topology"]
