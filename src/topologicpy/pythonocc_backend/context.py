from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from .topology import Topology


@dataclass(eq=False)
class Context(Topology):
    topology: Optional[Topology] = None
    x: float = 0
    y: float = 0
    z: float = 0

    @staticmethod
    def ByTopologyParameters(topology, u=0.5, v=0.5, w=0.5):
        return Context(shape=None, topology=topology, x=u, y=v, z=w)

# ---------------------------------------------------------------------------
# Explicit unsupported Context API
# ---------------------------------------------------------------------------
from .helpers import not_implemented as _not_implemented


def _context_not_implemented(name, return_value=None):
    def _method(*args, **kwargs):
        return _not_implemented(f"Context.{name}", return_value)
    return _method


# This placeholder does not yet implement true topologic_core context semantics.
Context.ByTopologyParameters = staticmethod(_context_not_implemented("ByTopologyParameters"))
