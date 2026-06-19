from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from .topology import Topology


@dataclass(eq=False)
class Aperture(Topology):
    topology: Optional[Topology] = None

    @staticmethod
    def ByTopologyContext(topology, context):
        return Aperture(shape=None, topology=topology)

# ---------------------------------------------------------------------------
# Explicit unsupported Aperture API
# ---------------------------------------------------------------------------
from .helpers import not_implemented as _not_implemented


def _aperture_not_implemented(name, return_value=None):
    def _method(*args, **kwargs):
        return _not_implemented(f"Aperture.{name}", return_value)
    return _method


# This placeholder does not yet attach apertures to hosts or contexts in the way
# topologic_core does, so expose it as unsupported for now.
Aperture.ByTopologyContext = staticmethod(_aperture_not_implemented("ByTopologyContext"))
