"""Independent contracts for the 4 x 8 cube Boolean regression matrix.

Face expectations reproduce the corrected reviewer notebook. Coordinates label
and evaluate known planar faces only; they never create production provenance.
Geometric measures are analytical (intersection volume = 0.4 ** 3).
"""
from itertools import product
from topologicpy.Cell import Cell
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex

TOLERANCE = 1e-6
REFERENCE_EPS = 1e-8
STATES = {
    "disjoint": ((2, 0, 0), 1),
    "touching": ((1, 0, 0), 1),
    "intersecting": ((0.6, 0.6, 0.6), 1),
    "contained": ((0.25, 0.25, 0.25), 0.5),
}
OPERATIONS = ("Union", "Difference A-B", "Difference B-A", "Intersect",
              "XOR", "Merge", "Impose", "Imprint")
CASES = tuple(product(STATES, OPERATIONS))
CASE_IDS = tuple(state + "-" + operation.replace(" ", "_") for state, operation in CASES)
SIDES = ("x-", "x+", "y-", "y+", "z-", "z+")
A = {"A:" + side for side in SIDES}
B = {"B:" + side for side in SIDES}
A_PLUS = {"A:" + side for side in ("x+", "y+", "z+")}
B_MINUS = {"B:" + side for side in ("x-", "y-", "z-")}
EXPECTED_FACES = {
    "disjoint": {
        "Union": A | B, "Difference A-B": A, "Difference B-A": B,
        "Intersect": set(), "XOR": A | B, "Merge": A | B,
        "Impose": A | B, "Imprint": A,
    },
    "touching": {
        "Union": (A | B) - {"A:x+", "B:x-"},
        "Difference A-B": A | {"B:x-"},
        "Difference B-A": B | {"A:x+"},
        "Intersect": {"A:x+", "B:x-"}, "XOR": A | B,
        "Merge": A | B, "Impose": A | B, "Imprint": A | {"B:x-"},
    },
    "intersecting": {
        "Union": A | B, "Difference A-B": A | B_MINUS,
        "Difference B-A": B | A_PLUS, "Intersect": A_PLUS | B_MINUS,
        "XOR": A | B, "Merge": A | B, "Impose": A | B,
        "Imprint": A | B_MINUS,
    },
    "contained": {
        "Union": A, "Difference A-B": A | B, "Difference B-A": set(),
        "Intersect": B, "XOR": A | B, "Merge": A | B,
        "Impose": A | B, "Imprint": A | B,
    },
}
# (exact root type, dimension, analytical total measure).
# Cluster denotes disconnected pieces. Connected Merge/Impose partitions
# are CellComplexes; Imprint retains A only.
EXPECTED_GEOMETRY = {
    "disjoint": {
        "Union": ("Cluster", 3, 2), "Difference A-B": ("Cell", 3, 1),
        "Difference B-A": ("Cell", 3, 1), "Intersect": (None, None, None),
        "XOR": ("Cluster", 3, 2), "Merge": ("Cluster", 3, 2),
        "Impose": ("Cluster", 3, 2), "Imprint": ("Cell", 3, 1),
    },
    "touching": {
        "Union": ("Cell", 3, 2), "Difference A-B": ("Cell", 3, 1),
        "Difference B-A": ("Cell", 3, 1), "Intersect": ("Face", 2, 1),
        "XOR": ("Cluster", 3, 2), "Merge": ("CellComplex", 3, 2),
        "Impose": ("CellComplex", 3, 2), "Imprint": ("Cell", 3, 1),
    },
    "intersecting": {
        "Union": ("Cell", 3, 1.936), "Difference A-B": ("Cell", 3, 0.936),
        "Difference B-A": ("Cell", 3, 0.936), "Intersect": ("Cell", 3, 0.064),
        "XOR": ("Cluster", 3, 1.872), "Merge": ("CellComplex", 3, 1.936),
        "Impose": ("CellComplex", 3, 1.936), "Imprint": ("CellComplex", 3, 1),
    },
    "contained": {
        "Union": ("Cell", 3, 1), "Difference A-B": ("Cell", 3, 0.875),
        "Difference B-A": (None, None, None), "Intersect": ("Cell", 3, 0.125),
        "XOR": ("Cell", 3, 0.875), "Merge": ("CellComplex", 3, 1),
        "Impose": ("CellComplex", 3, 1), "Imprint": ("CellComplex", 3, 1),
    },
}

def same(a, b):
    return a is not None and b is not None and bool(Topology.IsSame(a, b, silent=True))

def unique(items):
    answer = []
    for item in items:
        if not any(same(item, old) for old in answer):
            answer.append(item)
    return answer

def members(root, kind):
    if root is None:
        return []
    if Topology.IsInstance(root, kind):
        return [root]
    getter = {"Cell": "Cells", "Face": "Faces", "Edge": "Edges", "Vertex": "Vertices"}[kind]
    return unique(getattr(Topology, getter)(root, silent=True) or [])

def cube(origin=(0, 0, 0), size=1):
    return Cell.Prism(origin=Vertex.ByCoordinates(*origin), width=size, length=size,
                      height=size, placement="lowerleft", tolerance=TOLERANCE, silent=True)

def make_inputs(state):
    origin, size = STATES[state]
    return cube(), cube(origin, size)

def run_operation(a, b, operation, return_provenance=False):
    left, right = (b, a) if operation == "Difference B-A" else (a, b)
    name = "Difference" if operation.startswith("Difference") else operation
    return getattr(Topology, name)(left, right, tranDict=False,
                                  tolerance=TOLERANCE, silent=True,
                                  returnProvenance=return_provenance)

def face_extent(face):
    points = [Vertex.Coordinates(v, mantissa=12) for v in Topology.Vertices(face, silent=True)]
    assert points, "Reference face has no vertices"
    lo = [min(p[i] for p in points) for i in range(3)]
    hi = [max(p[i] for p in points) for i in range(3)]
    axes = [i for i in range(3) if abs(hi[i] - lo[i]) <= REFERENCE_EPS]
    assert len(axes) == 1, "Reference requires a nondegenerate axis-aligned planar face"
    axis = axes[0]
    uv = [i for i in range(3) if i != axis]
    return {"axis": axis, "plane": lo[axis], "lo": [lo[i] for i in uv], "hi": [hi[i] for i in uv]}

def plane_overlap(a, b):
    return (a["axis"] == b["axis"] and abs(a["plane"] - b["plane"]) <= REFERENCE_EPS
            and all(min(a["hi"][i], b["hi"][i]) - max(a["lo"][i], b["lo"][i]) > REFERENCE_EPS
                    for i in (0, 1)))

def labelled_faces(a, b, state):
    origin, size = STATES[state]
    labelled, extents = {}, {}
    for prefix, root, start, length in (("A", a, (0, 0, 0), 1), ("B", b, origin, size)):
        for face in members(root, "Face"):
            extent = face_extent(face)
            axis = extent["axis"]
            if abs(extent["plane"] - start[axis]) <= REFERENCE_EPS:
                side = "-"
            else:
                assert abs(extent["plane"] - (start[axis] + length)) <= REFERENCE_EPS
                side = "+"
            label = prefix + ":" + "xyz"[axis] + side
            assert label not in labelled
            labelled[label], extents[label] = face, extent
    assert set(labelled) == A | B
    return labelled, extents

def identity_label(entity, mapping):
    matches = [label for label, obj in mapping.items() if same(entity, obj)]
    assert len(matches) == 1, "Provenance endpoint is missing or ambiguous in actual faces"
    return matches[0]
