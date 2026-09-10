import math
import os

import pytest

from topologicpy.Cell import Cell
from topologicpy.Edge import Edge
from topologicpy.Face import Face
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex

BACKEND = os.environ.get(
    "TOPOLOGICPY_CORE_BACKEND",
    "",
).lower()

IS_PYTHONOCC = "pythonocc" in BACKEND


def test_step_entry_points_validate_without_throwing(tmp_path):
    bad = tmp_path / "bad.xyz"

    assert (
        Topology.Save(
            None,
            bad,
            silent=True,
        )
        is False
    )

    assert (
        Topology.Load(
            bad,
            silent=True,
        )
        is None
    )

    assert (
        Topology.ExportToSTEP(
            None,
            tmp_path / "bad.step",
            silent=True,
        )
        is False
    )

    assert (
        Topology.BySTEPPath(
            tmp_path / "missing.step",
            silent=True,
        )
        is None
    )


def test_generic_save_rejects_unregistered_extension(tmp_path):
    face = Face.Rectangle(
        width=2.0,
        length=3.0,
        silent=True,
    )

    assert (
        Topology.Save(
            face,
            tmp_path / "face.xyz",
            silent=True,
        )
        is False
    )


@pytest.mark.skipif(
    not IS_PYTHONOCC,
    reason="STEP BRep exchange is PythonOCC-specific.",
)
def test_step_roundtrip_preserves_exact_arc_geometry(tmp_path):
    edge = Edge.Arc(
        radius=3.0,
        fromAngle=15.0,
        toAngle=145.0,
        silent=True,
    )

    assert Topology.IsInstance(
        edge,
        "Edge",
    )

    expected_length = Edge.Length(
        edge,
        mantissa=None,
        silent=True,
    )

    path = tmp_path / "arc.step"

    assert Topology.ExportToSTEP(
        edge,
        path,
        overwrite=True,
        silent=True,
    )

    result = Topology.BySTEPPath(
        path,
        silent=True,
    )

    assert Topology.IsInstance(
        result,
        "Edge",
    )

    assert (
        Edge.IsLinear(
            result,
            silent=True,
        )
        is False
    )

    assert math.isclose(
        Edge.Length(
            result,
            mantissa=None,
            silent=True,
        ),
        expected_length,
        rel_tol=1.0e-7,
        abs_tol=1.0e-7,
    )


@pytest.mark.skipif(
    not IS_PYTHONOCC,
    reason="STEP BRep exchange is PythonOCC-specific.",
)
def test_step_roundtrip_preserves_nurbs_surface(tmp_path):
    # Exact rational quarter-cylinder patch.
    w = 1.0 / math.sqrt(2.0)

    control_points = [
        [
            Vertex.ByCoordinates(1.0, 0.0, 0.0),
            Vertex.ByCoordinates(1.0, 1.0, 0.0),
            Vertex.ByCoordinates(0.0, 1.0, 0.0),
        ],
        [
            Vertex.ByCoordinates(1.0, 0.0, 2.0),
            Vertex.ByCoordinates(1.0, 1.0, 2.0),
            Vertex.ByCoordinates(0.0, 1.0, 2.0),
        ],
    ]

    weights = [
        [1.0, w, 1.0],
        [1.0, w, 1.0],
    ]

    face = Face.ByNurbsParameters(
        controlPoints=control_points,
        weights=weights,
        uKnots=[0.0, 0.0, 1.0, 1.0],
        vKnots=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        isRational=True,
        uDegree=1,
        vDegree=2,
        silent=True,
    )

    assert Topology.IsInstance(
        face,
        "Face",
    )

    assert (
        Face.IsPlanar(
            face,
            silent=True,
        )
        is False
    )

    expected_area = Face.Area(
        face,
        mantissa=None,
        silent=True,
    )

    path = tmp_path / "surface.step"

    assert Topology.ExportToSTEP(
        face,
        path,
        overwrite=True,
        silent=True,
    )

    result = Topology.BySTEPPath(
        path,
        silent=True,
    )

    assert Topology.IsInstance(
        result,
        "Face",
    )

    assert (
        Face.IsPlanar(
            result,
            silent=True,
        )
        is False
    )

    assert math.isclose(
        Face.Area(
            result,
            mantissa=None,
            silent=True,
        ),
        expected_area,
        rel_tol=1.0e-6,
        abs_tol=1.0e-6,
    )


@pytest.mark.skipif(
    not IS_PYTHONOCC,
    reason="STEP BRep exchange is PythonOCC-specific.",
)
def test_generic_save_load_routes_step_codec(tmp_path):
    cell = Cell.Cylinder(
        radius=1.25,
        height=3.0,
        uSides=24,
        vSides=1,
        polyhedron=False,
        silent=True,
    )

    assert Topology.IsInstance(
        cell,
        "Cell",
    )

    path = tmp_path / "cylinder.stp"

    assert Topology.Save(
        cell,
        path,
        overwrite=True,
        silent=True,
    )

    result = Topology.Load(
        path,
        silent=True,
    )

    assert Topology.IsInstance(
        result,
        "Cell",
    )

    curved_faces = [
        face
        for face in (
            Topology.Faces(
                result,
                silent=True,
            )
            or []
        )
        if not Face.IsPlanar(
            face,
            silent=True,
        )
    ]

    assert len(curved_faces) >= 1


@pytest.mark.skipif(
    not IS_PYTHONOCC,
    reason="STEP BRep exchange is PythonOCC-specific.",
)
def test_step_overwrite_contract(tmp_path):
    face = Face.Rectangle(
        width=2.0,
        length=3.0,
        silent=True,
    )

    path = tmp_path / "overwrite.step"

    assert Topology.ExportToSTEP(
        face,
        path,
        overwrite=False,
        silent=True,
    )

    assert (
        Topology.ExportToSTEP(
            face,
            path,
            overwrite=False,
            silent=True,
        )
        is False
    )

    assert Topology.ExportToSTEP(
        face,
        path,
        overwrite=True,
        silent=True,
    )
