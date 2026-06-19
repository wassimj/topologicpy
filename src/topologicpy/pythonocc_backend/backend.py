from __future__ import annotations

from .aperture import Aperture
from .attributes import DoubleAttribute, IntAttribute, ListAttribute, StringAttribute
from .cell import Cell, CellUtility
from .cell_complex import CellComplex
from .cluster import Cluster
from .context import Context
from .dictionary import Dictionary
from .edge import Edge, EdgeUtility
from .face import Face, FaceUtility
from .graph import Graph, GraphUtility
from .shell import Shell, ShellUtility
from .topology import Topology, TopologyUtility
from .vertex import Vertex, VertexUtility
from .wire import Wire, WireUtility


def backend_name():
    return "PythonOCCBackend"


def namespaces():
    return [
        "Aperture", "Cell", "CellUtility", "CellComplex", "Cluster", "Context", "Dictionary",
        "DoubleAttribute", "Edge", "EdgeUtility", "Face", "FaceUtility",
        "Graph", "GraphUtility", "IntAttribute", "ListAttribute", "Shell",
        "ShellUtility", "StringAttribute", "Topology", "TopologyUtility",
        "Vertex", "VertexUtility", "Wire", "WireUtility",
    ]


_NAMESPACE_MAP = {
    "Aperture": Aperture,
    "Cell": Cell,
    "CellUtility": CellUtility,
    "CellComplex": CellComplex,
    "Cluster": Cluster,
    "Context": Context,
    "Dictionary": Dictionary,
    "DoubleAttribute": DoubleAttribute,
    "Edge": Edge,
    "EdgeUtility": EdgeUtility,
    "Face": Face,
    "FaceUtility": FaceUtility,
    "Graph": Graph,
    "GraphUtility": GraphUtility,
    "IntAttribute": IntAttribute,
    "ListAttribute": ListAttribute,
    "Shell": Shell,
    "ShellUtility": ShellUtility,
    "StringAttribute": StringAttribute,
    "Topology": Topology,
    "TopologyUtility": TopologyUtility,
    "Vertex": Vertex,
    "VertexUtility": VertexUtility,
    "Wire": Wire,
    "WireUtility": WireUtility,
}


class PythonOCCBackend:
    Aperture = Aperture
    Cell = Cell
    CellUtility = CellUtility
    CellComplex = CellComplex
    Cluster = Cluster
    Context = Context
    Dictionary = Dictionary
    DoubleAttribute = DoubleAttribute
    Edge = Edge
    EdgeUtility = EdgeUtility
    Face = Face
    FaceUtility = FaceUtility
    Graph = Graph
    GraphUtility = GraphUtility
    IntAttribute = IntAttribute
    ListAttribute = ListAttribute
    Shell = Shell
    ShellUtility = ShellUtility
    StringAttribute = StringAttribute
    Topology = Topology
    TopologyUtility = TopologyUtility
    Vertex = Vertex
    VertexUtility = VertexUtility
    Wire = Wire
    WireUtility = WireUtility

    @staticmethod
    def Name():
        return "PythonOCCBackend"

    @staticmethod
    def BackendName():
        return "PythonOCCBackend"

    @staticmethod
    def backend_name():
        return "PythonOCCBackend"

    @staticmethod
    def Namespaces():
        return namespaces()

    @staticmethod
    def namespaces():
        return namespaces()

    @staticmethod
    def Namespace(name):
        return _NAMESPACE_MAP.get(name, None)

# ---------------------------------------------------------------------------
# Compatibility call helper
# ---------------------------------------------------------------------------

def _instance_call(obj, method_name, *args):
    if obj is None:
        raise ValueError("Core.InstanceCall - Error: obj cannot be None.")
    if method_name is None:
        raise ValueError("Core.InstanceCall - Error: method_name cannot be None.")
    method = getattr(obj, method_name, None)
    if method is None or not callable(method):
        raise AttributeError(f"Core.InstanceCall - Error: {obj.__class__.__name__}.{method_name} is not available.")
    return method(*args)


PythonOCCBackend.InstanceCall = staticmethod(_instance_call)
