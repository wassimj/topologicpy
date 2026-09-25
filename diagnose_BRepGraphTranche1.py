"""Low-level diagnostic for TopologicPy BRepGraph Tranche 1 on pythonocc-core 8.x."""

from OCC.Core.BRepGraph import BRepGraph_ChildExplorer, brepgraph
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox

from topologicpy.pythonocc_backend._brepgraph import BRepGraphIndex


def nk(node):
    if node is None:
        return None
    try:
        return (str(node.NodeKind), int(node.Index), bool(node.IsValid()))
    except Exception as exc:
        return (type(node).__name__, repr(node), f"ERROR: {exc}")


box = BRepPrimAPI_MakeBox(10.0, 20.0, 30.0).Shape()
graph = brepgraph()
add = graph.Shapes().Add(box)
root = add.TopologyRoot

print("add.IsOk:", add.IsOk())
print("root:", nk(root))
print("graph counts:")
print("  nodes   :", graph.Topo().Gen().NbNodes())
print("  solids  :", graph.Topo().Solids().Nb())
print("  shells  :", graph.Topo().Shells().Nb())
print("  faces   :", graph.Topo().Faces().Nb())
print("  wires   :", graph.Topo().Wires().Nb())
print("  edges   :", graph.Topo().Edges().Nb())
print("  vertices:", graph.Topo().Vertices().Nb())

print("\nChildExplorer rows:")
explorer = BRepGraph_ChildExplorer(graph, root)
count = 0
while explorer.More():
    depth = int(explorer.Depth())
    try:
        current = explorer.NodeAt(depth - 2) if depth >= 2 else None
    except Exception as exc:
        current = f"ERROR: {exc}"
    try:
        parent = explorer.CurrentParent()
    except Exception as exc:
        parent = f"ERROR: {exc}"
    print(f"  {count:03d} depth={depth} current={nk(current) if not isinstance(current, str) else current} parent={nk(parent) if not isinstance(parent, str) else parent}")
    count += 1
    if count >= 40:
        print("  ... truncated after 40 rows")
        break
    explorer.Next()

print("\nTopologicPy BRepGraphIndex:")
index = BRepGraphIndex(box)
print("  valid:", index.valid)
print("  incidence ready:", index._ensure_incidence())
print("  recorded nodes:", len(index._nodes_by_key))
print("  parent buckets:", len(index._children_by_parent))
print("  child buckets:", len(index._parents_by_child))

from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
for name, kind in (("faces", TopAbs_FACE), ("edges", TopAbs_EDGE), ("vertices", TopAbs_VERTEX)):
    result = index.subshapes(box, kind)
    print(f"  {name}:", None if result is None else len(result))
