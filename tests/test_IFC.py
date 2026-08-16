"""Best-effort tests for topologicpy.IFC.

These tests exercise the pure-Python parser and fast topology importer paths
without requiring IfcOpenShell, TopologicCore, or a live IFC geometry kernel.
"""

import importlib
import sys
import types

import pytest


@pytest.fixture(autouse=True)
def _suppress_expected_topologicpy_output(capfd):
    """Keep expected TopologicPy diagnostic prints out of normal pytest output."""
    capfd.readouterr()
    yield
    capfd.readouterr()


@pytest.fixture
def ifc_module():
    return importlib.import_module("topologicpy.IFC")


BOUNDING_BOX_STEP = """ISO-10303-21;
DATA;
#1=IFCCARTESIANPOINT((0.,0.,0.));
#2=IFCBOUNDINGBOX(#1,2.,3.,4.);
#3=IFCSHAPEREPRESENTATION($,'Body','BoundingBox',(#2));
#4=IFCPRODUCTDEFINITIONSHAPE($,$,(#3));
#5=IFCWALL('gid-wall',$,'Wall A',$,$,$,#4,$);
#6=IFCOPENINGELEMENT('gid-open',$,'Opening',$,$,$,#4,$);
ENDSEC;
END-ISO-10303-21;
"""


EXTRUSION_STEP = """ISO-10303-21;
DATA;
#1=IFCCARTESIANPOINT((0.,0.,0.));
#2=IFCDIRECTION((0.,0.,1.));
#3=IFCDIRECTION((1.,0.,0.));
#4=IFCAXIS2PLACEMENT3D(#1,#2,#3);
#5=IFCRECTANGLEPROFILEDEF(.AREA.,'Profile',$,2.,3.);
#6=IFCEXTRUDEDAREASOLID(#5,#4,#2,4.);
#7=IFCSHAPEREPRESENTATION($,'Body','SweptSolid',(#6));
#8=IFCPRODUCTDEFINITIONSHAPE($,$,(#7));
#9=IFCWALL('gid-extrude',$,'Extruded Wall',$,$,$,#8,$);
ENDSEC;
END-ISO-10303-21;
"""


def _install_fake_topology(monkeypatch):
    """Install a small fake topologicpy.Topology module for geometry creation."""

    class FakeTopology:
        created = []

        @staticmethod
        def ByGeometry(vertices=None, edges=None, faces=None, topologyType=None, tolerance=0.0001, silent=False):
            result = {
                "kind": "Topology",
                "vertices": list(vertices or []),
                "edges": list(edges or []),
                "faces": list(faces or []),
                "topologyType": topologyType,
                "tolerance": tolerance,
            }
            FakeTopology.created.append(result)
            return result

        @staticmethod
        def IsInstance(topology, type_name):
            return isinstance(topology, dict) and topology.get("kind") == "Topology"

        @staticmethod
        def SetDictionary(topology, dictionary, silent=True):
            topology = dict(topology)
            topology["dictionary"] = dictionary
            return topology

    module = types.ModuleType("topologicpy.Topology")
    module.Topology = FakeTopology
    monkeypatch.setitem(sys.modules, "topologicpy.Topology", module)
    return FakeTopology


def _install_fake_dictionary(monkeypatch):
    """Install a small fake topologicpy.Dictionary module."""

    class FakeDictionary:
        @staticmethod
        def ByKeysValues(keys, values):
            return dict(zip(keys, values))

        @staticmethod
        def ByKeyValue(key, value):
            return {key: value}

        @staticmethod
        def SetValueAtKey(dictionary, key, value):
            out = dict(dictionary or {})
            out[key] = value
            return out

    module = types.ModuleType("topologicpy.Dictionary")
    module.Dictionary = FakeDictionary
    monkeypatch.setitem(sys.modules, "topologicpy.Dictionary", module)
    return FakeDictionary


def test_fast_entity_and_dynamic_ifc_entity_properties(ifc_module):
    entity = ifc_module.IFCFastEntity(
        42,
        "IFCWALL",
        ["gid-42", None, "Wall Name", "Description", None, None, None, "Tag"],
    )
    wrapper = ifc_module.IFCEntity(
        entity,
        properties={
            "properties": {"Pset_WallCommon": {"FireRating": "2HR"}},
            "quantities": {"BaseQuantities": {"Length": 12.5}},
            "materials": ["Concrete"],
        },
    )

    assert entity.is_a("IfcWall") is True
    assert entity.is_a("IfcSlab") is False
    assert wrapper.IFCId == 42
    assert wrapper.IFCKey == "#42"
    assert wrapper.IFCType == "IfcWall"
    assert wrapper.GlobalId == "gid-42"
    assert wrapper.Name == "Wall Name"
    assert wrapper.Attribute("FireRating") == "2HR"
    assert wrapper.Attribute("Pset_WallCommon.FireRating") == "2HR"
    assert wrapper.Attribute("Length") == 12.5
    assert wrapper.Attribute("Missing", default="fallback") == "fallback"
    assert wrapper.ToDict()["materials"] == ["Concrete"]




def test_parse_text_entities_and_ifc_wrapper_lookup(ifc_module):
    entities = ifc_module.IFCFastTopology.ParseText(BOUNDING_BOX_STEP, silent=True)

    assert sorted(entities) == [1, 2, 3, 4, 5, 6]
    assert entities[5].type == "IFCWALL"
    assert entities[5].args[0] == "gid-wall"
    assert ifc_module.IFC.Entities(entities, silent=True) is entities
    assert ifc_module.IFC.Entities(BOUNDING_BOX_STEP, silent=True)[5].type == "IFCWALL"

    obj_by_id = ifc_module.IFC.Object(entities, ifcId=5, silent=True)
    obj_by_key = ifc_module.IFC.Object(entities, ifcKey="#5", silent=True)
    obj_by_gid = ifc_module.IFC.Object(entities, globalId="gid-wall", silent=True)

    assert obj_by_id.Name == "Wall A"
    assert obj_by_key.IFCId == 5
    assert obj_by_gid.IFCType == "IfcWall"
    assert ifc_module.IFC.Object(entities, ifcId=999, silent=True) is None


def test_parse_path_summary_and_file_like_sources(tmp_path, ifc_module):
    path = tmp_path / "minimal.ifc"
    path.write_text(BOUNDING_BOX_STEP, encoding="utf-8")

    parsed = ifc_module.IFCFastTopology.Parse(str(path), silent=True)
    summary = ifc_module.IFCFastTopology.SummaryByPath(str(path), silent=True)

    assert parsed[5].type == "IFCWALL"
    assert summary["entity_count"] == 6
    assert summary["product_count"] == 1
    assert summary["product_type_counts"] == {"IFCWALL": 1}
    assert ifc_module.IFC.FileByPath(str(path), silent=True)[5].args[2] == "Wall A"
    assert ifc_module.IFC.EntitiesByPath(str(path), silent=True)[5].type == "IFCWALL"

    class FileLike:
        def to_string(self):
            return BOUNDING_BOX_STEP

    assert ifc_module.IFC.Entities(FileLike(), silent=True)[5].type == "IFCWALL"


def test_product_selection_excludes_openings_unless_explicitly_requested(ifc_module):
    entities = ifc_module.IFCFastTopology.ParseText(BOUNDING_BOX_STEP, silent=True)

    products = ifc_module.IFCFastTopology.ProductsByEntities(entities)
    assert [p.type for p in products] == ["IFCWALL"]

    openings = ifc_module.IFCFastTopology.ProductsByEntities(entities, includeTypes=["IfcOpeningElement"])
    assert [p.type for p in openings] == ["IFCOPENINGELEMENT"]

    no_walls = ifc_module.IFCFastTopology.ProductsByEntities(entities, excludeTypes=["ifcwall"])
    assert no_walls == []


def test_bounding_box_mesh_and_merged_product_mesh(ifc_module):
    entities = ifc_module.IFCFastTopology.ParseText(BOUNDING_BOX_STEP, silent=True)
    wall = entities[5]

    mesh = ifc_module.IFCFastTopology.MeshDataByProduct(wall, entities, scale=2.0)
    assert len(mesh["vertices"]) == 8
    assert len(mesh["faces"]) == 6
    assert mesh["vertices"][1] == [4.0, 0.0, 0.0]
    assert mesh["vertices"][7] == [0.0, 6.0, 8.0]

    merged = ifc_module.IFCFastTopology.MeshDataByProducts([wall], entities, dictionaryMode="none", ontology=False, silent=True)
    assert len(merged["vertices"]) == 8
    assert merged["product_ranges"][0]["global_id"] == "gid-wall"
    assert merged["product_ranges"][0]["face_end"] == 6


def test_extruded_rectangle_mesh(ifc_module):
    entities = ifc_module.IFCFastTopology.ParseText(EXTRUSION_STEP, silent=True)
    wall = entities[9]

    mesh = ifc_module.IFCFastTopology.MeshDataByProduct(wall, entities, scale=1.0)

    assert len(mesh["vertices"]) == 8
    assert len(mesh["faces"]) == 6
    assert [-1.0, -1.5, 0.0] in mesh["vertices"]
    assert [1.0, 1.5, 4.0] in mesh["vertices"]


def test_topologies_by_entities_uses_fake_topology(monkeypatch, ifc_module):
    fake_topology = _install_fake_topology(monkeypatch)
    entities = ifc_module.IFCFastTopology.ParseText(BOUNDING_BOX_STEP, silent=True)

    topologies = ifc_module.IFCFastTopology.TopologiesByEntities(
        entities,
        dictionaryMode="none",
        ontology=False,
        tolerance=0.25,
        silent=True,
    )

    assert len(topologies) == 1
    assert topologies[0]["kind"] == "Topology"
    assert len(topologies[0]["vertices"]) == 8
    assert len(topologies[0]["faces"]) == 6
    assert topologies[0]["tolerance"] == 0.25
    assert fake_topology.created[0]["topologyType"] is None


def test_topologies_by_file_and_path_wrapper_dispatch(tmp_path, monkeypatch, ifc_module):
    _install_fake_topology(monkeypatch)
    path = tmp_path / "box.ifc"
    path.write_text(BOUNDING_BOX_STEP, encoding="utf-8")

    by_path = ifc_module.IFC.TopologiesByPath(str(path), dictionaryMode="none", ontology=False, silent=True)
    by_file_path = ifc_module.IFC.TopologiesByFile(str(path), dictionaryMode="none", ontology=False, silent=True)
    by_file_dict = ifc_module.IFC.TopologiesByFile(
        ifc_module.IFC.Entities(str(path), silent=True),
        dictionaryMode="none",
        ontology=False,
        silent=True,
    )

    assert len(by_path) == 1
    assert len(by_file_path) == 1
    assert len(by_file_dict) == 1
    assert ifc_module.IFC.TopologiesByPath(str(path / "missing.ifc"), silent=True) is None






