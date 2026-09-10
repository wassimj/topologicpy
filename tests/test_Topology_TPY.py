import json
import math
import os
import zipfile

import pytest

from topologicpy.Aperture import Aperture
from topologicpy.Cell import Cell
from topologicpy.Cluster import Cluster
from topologicpy.Context import Context
from topologicpy.Dictionary import Dictionary
from topologicpy.Edge import Edge
from topologicpy.Face import Face
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex


BACKEND = os.environ.get(
    "TOPOLOGICPY_CORE_BACKEND",
    "",
).lower()

IS_PYTHONOCC = (
    "pythonocc" in BACKEND
)


def _set_dict(topology, values):
    dictionary = Dictionary.ByPythonDictionary(
        values,
        silent=True,
    )

    return Topology.SetDictionary(
        topology,
        dictionary,
        silent=True,
    )


def _py_dict(topology):
    dictionary = Topology.Dictionary(
        topology,
        silent=True,
    )

    return Dictionary.PythonDictionary(
        dictionary,
        silent=True,
    ) or {}


def test_tpy_parent_dictionary_roundtrip(tmp_path):
    face = Face.Rectangle(
        width=4.0,
        length=3.0,
        silent=True,
    )

    face = _set_dict(
        face,
        {
            "name": "Room A",
            "number": 17,
            "active": True,
            "values": [1, 2.5, "x"],
        },
    )

    # Capture TopologicPy's actual stored dictionary representation before
    # persistence. Some backends represent Python bool values as integer 0/1.
    expected_dictionary = _py_dict(
        face
    )

    path = tmp_path / "parent.tpy"

    assert Topology.ExportToTPY(
        face,
        path,
        overwrite=True,
        silent=True,
    )

    result = Topology.ByTPYPath(
        path,
        silent=True,
    )

    assert Topology.IsInstance(
        result,
        "Face",
    )

    dictionary = _py_dict(
        result
    )

    assert dictionary == expected_dictionary
    assert dictionary["name"] == "Room A"
    assert dictionary["number"] == 17
    assert dictionary["values"] == [1, 2.5, "x"]


def test_generic_save_load_routes_tpy_codec(tmp_path):
    face = Face.Rectangle(
        width=2.0,
        length=2.0,
        silent=True,
    )

    face = _set_dict(
        face,
        {
            "id": "generic-save",
        },
    )

    path = tmp_path / "generic.tpy"

    assert Topology.Save(
        face,
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
        "Face",
    )

    assert (
        _py_dict(
            result
        ).get("id")
        == "generic-save"
    )


def test_tpy_subtopology_dictionary_roundtrip(tmp_path):
    cell = Cell.Box(
        width=4.0,
        length=3.0,
        height=2.0,
        silent=True,
    )

    faces = Topology.Faces(
        cell,
        silent=True,
    ) or []

    edges = Topology.Edges(
        cell,
        silent=True,
    ) or []

    assert len(faces) >= 1
    assert len(edges) >= 1

    _set_dict(
        faces[0],
        {
            "saved_face": "face-0",
        },
    )

    _set_dict(
        edges[0],
        {
            "saved_edge": "edge-0",
        },
    )

    path = tmp_path / "subtopologies.tpy"

    assert Topology.ExportToTPY(
        cell,
        path,
        overwrite=True,
        silent=True,
    )

    result = Topology.ByTPYPath(
        path,
        silent=True,
    )

    assert Topology.IsInstance(
        result,
        "Cell",
    )

    loaded_faces = Topology.Faces(
        result,
        silent=True,
    ) or []

    loaded_edges = Topology.Edges(
        result,
        silent=True,
    ) or []

    assert any(
        _py_dict(face).get(
            "saved_face"
        )
        == "face-0"
        for face in loaded_faces
    )

    assert any(
        _py_dict(edge).get(
            "saved_edge"
        )
        == "edge-0"
        for edge in loaded_edges
    )


def test_tpy_content_relationship_roundtrip(tmp_path):
    host = Face.Rectangle(
        width=6.0,
        length=4.0,
        silent=True,
    )

    content = Edge.ByVertices(
        [
            Vertex.ByCoordinates(
                -1.0,
                0.0,
                0.0,
            ),
            Vertex.ByCoordinates(
                1.0,
                0.0,
                0.0,
            ),
        ],
        silent=True,
    )

    content = _set_dict(
        content,
        {
            "role": "content",
        },
    )

    host = Topology.AddContent(
        host,
        content,
        subTopologyType="self",
        silent=True,
    )

    contents = Topology.Contents(
        host,
        silent=True,
    ) or []

    assert len(contents) == 1

    path = tmp_path / "contents.tpy"

    assert Topology.ExportToTPY(
        host,
        path,
        overwrite=True,
        silent=True,
    )

    result = Topology.ByTPYPath(
        path,
        silent=True,
    )

    loaded_contents = Topology.Contents(
        result,
        silent=True,
    ) or []

    assert len(loaded_contents) == 1

    assert (
        _py_dict(
            loaded_contents[0]
        ).get("role")
        == "content"
    )

    contexts = Topology.Contexts(
        loaded_contents[0],
        silent=True,
    ) or []

    assert len(contexts) >= 1


def test_tpy_aperture_relationship_roundtrip(tmp_path):
    host = Face.Rectangle(
        width=6.0,
        length=4.0,
        silent=True,
    )

    aperture_topology = Face.Rectangle(
        origin=Vertex.ByCoordinates(
            0.0,
            0.0,
            0.0,
        ),
        width=1.0,
        length=1.0,
        silent=True,
    )

    aperture_topology = _set_dict(
        aperture_topology,
        {
            "role": "aperture",
        },
    )

    context = Context.ByTopologyParameters(
        host,
        u=0.25,
        v=0.75,
        w=0.5,
    )

    aperture = Aperture.ByTopologyContext(
        aperture_topology,
        context,
    )

    assert aperture is not None

    assert len(
        Topology.Apertures(
            host,
            silent=True,
        )
        or []
    ) >= 1

    path = tmp_path / "aperture.tpy"

    assert Topology.ExportToTPY(
        host,
        path,
        overwrite=True,
        silent=True,
    )

    result = Topology.ByTPYPath(
        path,
        silent=True,
    )

    apertures = Topology.Apertures(
        result,
        silent=True,
    ) or []

    assert len(apertures) >= 1

    assert any(
        _py_dict(aperture).get(
            "role"
        )
        == "aperture"
        for aperture in apertures
    )


@pytest.mark.pythonocc_only
def test_tpy_exact_arc_roundtrip(tmp_path):
    arc = Edge.Arc(
        radius=3.0,
        fromAngle=20.0,
        toAngle=160.0,
        silent=True,
    )

    expected_length = Edge.Length(
        arc,
        mantissa=None,
        silent=True,
    )

    path = tmp_path / "arc.tpy"

    assert Topology.ExportToTPY(
        arc,
        path,
        overwrite=True,
        silent=True,
    )

    result = Topology.ByTPYPath(
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
        rel_tol=1.0e-9,
        abs_tol=1.0e-9,
    )


@pytest.mark.pythonocc_only
def test_tpy_exact_nurbs_face_roundtrip(tmp_path):
    w = 1.0 / math.sqrt(2.0)

    control_points = [
        [
            Vertex.ByCoordinates(
                1.0,
                0.0,
                0.0,
            ),
            Vertex.ByCoordinates(
                1.0,
                1.0,
                0.0,
            ),
            Vertex.ByCoordinates(
                0.0,
                1.0,
                0.0,
            ),
        ],
        [
            Vertex.ByCoordinates(
                1.0,
                0.0,
                2.0,
            ),
            Vertex.ByCoordinates(
                1.0,
                1.0,
                2.0,
            ),
            Vertex.ByCoordinates(
                0.0,
                1.0,
                2.0,
            ),
        ],
    ]

    face = Face.ByNurbsParameters(
        controlPoints=control_points,
        weights=[
            [1.0, w, 1.0],
            [1.0, w, 1.0],
        ],
        uKnots=[
            0.0,
            0.0,
            1.0,
            1.0,
        ],
        vKnots=[
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
        ],
        isRational=True,
        uDegree=1,
        vDegree=2,
        silent=True,
    )

    expected_area = Face.Area(
        face,
        mantissa=None,
        silent=True,
    )

    path = tmp_path / "nurbs.tpy"

    assert Topology.ExportToTPY(
        face,
        path,
        overwrite=True,
        silent=True,
    )

    result = Topology.ByTPYPath(
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
        rel_tol=1.0e-9,
        abs_tol=1.0e-9,
    )


def test_tpy_shapeless_cluster_roundtrip(tmp_path):
    edge_a = Edge.ByVertices(
        [
            Vertex.ByCoordinates(
                0,
                0,
                0,
            ),
            Vertex.ByCoordinates(
                1,
                0,
                0,
            ),
        ],
        silent=True,
    )

    edge_b = Edge.ByVertices(
        [
            Vertex.ByCoordinates(
                0,
                1,
                0,
            ),
            Vertex.ByCoordinates(
                1,
                1,
                0,
            ),
        ],
        silent=True,
    )

    cluster = Cluster.ByTopologies(
        [
            edge_a,
            edge_b,
        ],
        silent=True,
    )

    assert Topology.IsInstance(
        cluster,
        "Cluster",
    )

    path = tmp_path / "cluster.tpy"

    assert Topology.ExportToTPY(
        cluster,
        path,
        overwrite=True,
        silent=True,
    )

    result = Topology.ByTPYPath(
        path,
        silent=True,
    )

    assert Topology.IsInstance(
        result,
        "Cluster",
    )

    assert len(
        Topology.Edges(
            result,
            silent=True,
        )
        or []
    ) == 2


def test_tpy_overwrite_contract(tmp_path):
    face = Face.Rectangle(
        width=2.0,
        length=2.0,
        silent=True,
    )

    path = tmp_path / "overwrite.tpy"

    assert Topology.ExportToTPY(
        face,
        path,
        overwrite=False,
        silent=True,
    )

    assert (
        Topology.ExportToTPY(
            face,
            path,
            overwrite=False,
            silent=True,
        )
        is False
    )

    assert Topology.ExportToTPY(
        face,
        path,
        overwrite=True,
        silent=True,
    )


def test_tpy_corrupt_checksum_is_rejected(tmp_path):
    face = Face.Rectangle(
        width=2.0,
        length=2.0,
        silent=True,
    )

    good = tmp_path / "good.tpy"
    bad = tmp_path / "bad.tpy"

    assert Topology.ExportToTPY(
        face,
        good,
        overwrite=True,
        silent=True,
    )

    with zipfile.ZipFile(
        good,
        "r",
    ) as source:
        manifest = json.loads(
            source.read(
                "manifest.json"
            ).decode("utf-8")
        )

        geometry_path = (
            manifest["objects"][0][
                "geometry"
            ]["path"]
        )

        members = {
            name: source.read(name)
            for name in source.namelist()
        }

    members[
        geometry_path
    ] = (
        members[geometry_path]
        + b"\n# corrupted\n"
    )

    with zipfile.ZipFile(
        bad,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as target:
        for name, data in members.items():
            target.writestr(
                name,
                data,
            )

    assert (
        Topology.ByTPYPath(
            bad,
            silent=True,
        )
        is None
    )
