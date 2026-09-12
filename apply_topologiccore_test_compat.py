#!/usr/bin/env python3
"""
Apply ONLY the TopologicCore capability adjustments needed by the current
Curve/NURBS reintegration tests.

IMPORTANT:
    Run this only after restoring these files to the repository's current
    pre-tranche versions:

        tests/test_Face.py
        tests/test_Wire.py
        tests/test_Shell.py
        tests/test_Face_CurveWireSupport.py
        tests/test_Topology_BREPString.py

This script deliberately refuses to operate on the stale whole-file test
snapshots from the previous ChatGPT tranche.
"""

from pathlib import Path
import re
import sys

ROOT = Path.cwd()
TESTS = ROOT / "tests"

FILES = {
    "face": TESTS / "test_Face.py",
    "wire": TESTS / "test_Wire.py",
    "shell": TESTS / "test_Shell.py",
    "curve": TESTS / "test_Face_CurveWireSupport.py",
    "brep": TESTS / "test_Topology_BREPString.py",
}

for key, path in FILES.items():
    if not path.exists():
        raise SystemExit(f"ERROR: {path} does not exist. Run from the repository root.")

# ---------------------------------------------------------------------------
# Guard against the stale whole-file snapshots from the previous tranche.
# Those snapshots are NOT a valid base because they pre-date fixes already
# demonstrated by the 1470-pass PythonOCC run.
# ---------------------------------------------------------------------------
stale_fingerprints = {
    FILES["face"]: [
        "Face.Compactness currently does not validate invalid face input before iterating edges.",
        "Face.IsConvex currently appears to return True for concave faces because it compares all(...) to 180.",
    ],
    FILES["shell"]: [
        "Shell.IsClosed currently does not validate invalid input before forwarding to Core.InstanceCall.",
        "Shell.IsOnBoundary iterates over None when a shell has no internal boundaries and the vertex is not on the external boundary.",
    ],
    FILES["wire"]: [
        "Wire.Project computes the default direction as -1 * list, which produces an invalid empty list.",
        "assert len(representation) == 4",
    ],
}

bad = []
for path, fingerprints in stale_fingerprints.items():
    text = path.read_text(encoding="utf-8")
    found = [fp for fp in fingerprints if fp in text]
    if found:
        bad.append((path, found))

if bad:
    print("ERROR: stale replacement test files are still present.")
    print("Restore the five test files to their pre-tranche/Git versions first.")
    for path, found in bad:
        print(f"  {path}:")
        for item in found:
            print(f"    - {item}")
    sys.exit(2)


def write(path, text):
    path.write_text(text, encoding="utf-8")
    print(f"patched: {path}")


def add_kwarg_to_call(text, function_name, anchor, kwarg="polyline=True"):
    """
    Add kwarg to the first call to function_name occurring in the named test
    function. The call may span multiple lines.
    """
    func_match = re.search(
        rf"(?ms)^def {re.escape(anchor)}\([^)]*\):\n(?P<body>.*?)(?=^def |\Z)",
        text,
    )
    if not func_match:
        raise RuntimeError(f"Could not locate test function: {anchor}")

    block = func_match.group(0)
    start = block.find(function_name + "(")
    if start < 0:
        raise RuntimeError(f"Could not locate {function_name}(...) inside {anchor}")

    # Find matching closing parenthesis.
    i = start + len(function_name)
    depth = 0
    end = None
    for j in range(i, len(block)):
        c = block[j]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                end = j
                break
    if end is None:
        raise RuntimeError(f"Could not parse {function_name}(...) inside {anchor}")

    call = block[start:end+1]
    if kwarg.split("=")[0] in call:
        return text

    new_call = call[:-1].rstrip()
    if new_call.endswith("("):
        new_call += kwarg + ")"
    else:
        new_call += ", " + kwarg + ")"

    new_block = block[:start] + new_call + block[end+1:]
    return text[:func_match.start()] + new_block + text[func_match.end():]


def add_skip_decorator(text, test_name, reason):
    marker = f"def {test_name}("
    if marker not in text:
        raise RuntimeError(f"Could not locate {test_name}")
    before = text[:text.index(marker)]
    tail = before[-300:]
    if "Requires exact PythonOCC/OCCT" in tail or "raw OCCT DBRep_DrawableShape" in tail:
        return text
    decorator = (
        "@pytest.mark.skipif(\n"
        "    Topology._IsTopologicCoreBackend(),\n"
        f'    reason="{reason}",\n'
        ")\n"
    )
    return text.replace(marker, decorator + marker, 1)


# ---------------------------------------------------------------------------
# test_Face.py
# Generic constructor coverage should request the historical polygonal ellipse
# explicitly. Do not alter any other Face tests.
# ---------------------------------------------------------------------------
path = FILES["face"]
text = path.read_text(encoding="utf-8")

m = re.search(
    r"Face\.Ellipse\((?P<args>[^)]*width\s*=\s*4[^)]*length\s*=\s*2[^)]*)\)",
    text,
)
if m and "polyline" not in m.group(0):
    replacement = m.group(0)[:-1].rstrip() + ", polyline=True)"
    text = text[:m.start()] + replacement + text[m.end():]

write(path, text)


# ---------------------------------------------------------------------------
# test_Wire.py
# These generic constructor tests are asking only for a valid Wire / closure,
# not exact-curve semantics. Explicitly request the polygonal mode.
# ---------------------------------------------------------------------------
path = FILES["wire"]
text = path.read_text(encoding="utf-8")

# Scope each edit to the intended test to avoid changing exact-curve tests.
def patch_all_calls_in_function(text, test_name, call_name):
    fm = re.search(
        rf"(?ms)^def {re.escape(test_name)}\([^)]*\):\n(?P<body>.*?)(?=^def |\Z)",
        text,
    )
    if not fm:
        raise RuntimeError(f"Could not locate {test_name}")
    block = fm.group(0)

    pos = 0
    rebuilt = ""
    changed = False
    while True:
        idx = block.find(call_name + "(", pos)
        if idx < 0:
            rebuilt += block[pos:]
            break
        rebuilt += block[pos:idx]

        open_i = idx + len(call_name)
        depth = 0
        end = None
        for j in range(open_i, len(block)):
            c = block[j]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end is None:
            raise RuntimeError(f"Could not parse {call_name} in {test_name}")

        call = block[idx:end+1]
        if "polyline=" not in call:
            call = call[:-1].rstrip() + ", polyline=True)"
            changed = True
        rebuilt += call
        pos = end + 1

    if not changed:
        return text
    return text[:fm.start()] + rebuilt + text[fm.end():]

for call in ("Wire.Circle", "Wire.Arc", "Wire.ArcByEdge"):
    text = patch_all_calls_in_function(
        text,
        "test_circle_and_arc_constructors_create_expected_wire_types",
        call,
    )

text = patch_all_calls_in_function(
    text,
    "test_arc_respects_close_parameter",
    "Wire.Arc",
)
text = patch_all_calls_in_function(
    text,
    "test_shape_constructors_return_closed_wires",
    "Wire.Squircle",
)
for call in ("Wire.Spiral", "Wire.GoldenSpiral"):
    text = patch_all_calls_in_function(
        text,
        "test_spiral_and_golden_spiral_create_open_or_nonzero_wires",
        call,
    )

write(path, text)


# ---------------------------------------------------------------------------
# test_Shell.py
# Keep Delaunay coverage on both backends. Only positive intrinsic Voronoi is
# PythonOCC-only. Invalid-input Voronoi checks remain on both backends.
# ---------------------------------------------------------------------------
path = FILES["shell"]
text = path.read_text(encoding="utf-8")

fm = re.search(
    r"(?ms)^def test_delaunay_and_voronoi_return_partition_topologies\([^)]*\):\n(?P<body>.*?)(?=^def |\Z)",
    text,
)
if not fm:
    raise RuntimeError("Could not locate Shell Delaunay/Voronoi test.")

block = fm.group(0)
if "if not Topology._IsTopologicCoreBackend():" not in block:
    old = re.search(
        r"(?ms)(?P<indent>    )voronoi = Shell\.Voronoi\((?P<call>.*?)\)\n\n"
        r"    _assert_topology\(delaunay\)\n"
        r"    _assert_topology\(voronoi\)\n"
        r"    assert len\(Topology\.Faces\(delaunay, silent=True\)\) > 0\n"
        r"    assert len\(Topology\.Faces\(voronoi, silent=True\)\) > 0",
        block,
    )
    if not old:
        raise RuntimeError(
            "Could not locate the expected positive Voronoi assertion block. "
            "Inspect test_Shell.py manually."
        )

    call_text = old.group("call")
    replacement = (
        "    _assert_topology(delaunay)\n"
        "    assert len(Topology.Faces(delaunay, silent=True)) > 0\n\n"
        "    if not Topology._IsTopologicCoreBackend():\n"
        f"        voronoi = Shell.Voronoi({call_text})\n"
        "        _assert_topology(voronoi)\n"
        "        assert len(Topology.Faces(voronoi, silent=True)) > 0"
    )
    block = block[:old.start()] + replacement + block[old.end():]
    text = text[:fm.start()] + block + text[fm.end():]

write(path, text)


# ---------------------------------------------------------------------------
# test_Face_CurveWireSupport.py
# Gate only the tests that explicitly require exact OCCT curve/surface support.
# ---------------------------------------------------------------------------
path = FILES["curve"]
text = path.read_text(encoding="utf-8")

for test_name in (
    "test_face_circle_exact_mode_preserves_curved_boundary",
    "test_face_ellipse_exact_mode_preserves_rational_curves",
    "test_face_bywires_preserves_curved_outer_and_inner_boundaries",
    "test_pythonocc_surface_evaluation_respects_face_location",
    "test_pythonocc_normaledge_uses_local_surface_normal",
):
    text = add_skip_decorator(
        text,
        test_name,
        "Requires exact PythonOCC/OCCT curve or surface support.",
    )

write(path, text)


# ---------------------------------------------------------------------------
# test_Topology_BREPString.py
# The DBRep_DrawableShape header is specifically OCCT serialization.
# ---------------------------------------------------------------------------
path = FILES["brep"]
text = path.read_text(encoding="utf-8")

if "import pytest" not in text:
    text = "import pytest\n\n" + text

for test_name in (
    "test_brep_string_is_raw_occt_brep",
    "test_export_to_brep_writes_raw_occt_brep",
):
    text = add_skip_decorator(
        text,
        test_name,
        "This test requires raw OCCT DBRep_DrawableShape serialization.",
    )

write(path, text)

print("\nTargeted TopologicCore test compatibility patch complete.")
