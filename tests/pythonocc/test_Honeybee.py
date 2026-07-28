# Auto-generated PythonOCC backend parity test
# Copied from test_Honeybee.py

import os
os.environ["TOPOLOGICPY_CORE_BACKEND"] = "pythonocc"

"""Best-effort tests for topologicpy.Honeybee.

The production module has optional, heavyweight Ladybug Tools dependencies. These
unit tests inject small fake modules before importing topologicpy.Honeybee so the
core wrapper logic can be tested deterministically without installing Honeybee,
Honeybee Energy, Honeybee Radiance, Ladybug, or Ladybug Geometry.
"""

from __future__ import annotations

import importlib
import json
import math
import sys
import types

import pytest


@pytest.fixture(autouse=True)
def _suppress_expected_topologicpy_output(capfd):
    """Keep expected TopologicPy diagnostic prints out of normal pytest output."""
    capfd.readouterr()
    yield
    capfd.readouterr()


def _new_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__package__ = name.rpartition(".")[0]
    return module


def _install_external_honeybee_fakes(monkeypatch):
    """Install minimal fake Ladybug Tools modules used at Honeybee.py import time."""
    # ------------------------------------------------------------------
    # honeybee core
    # ------------------------------------------------------------------
    honeybee = _new_module("honeybee")
    honeybee.__path__ = []
    facetype = _new_module("honeybee.facetype")
    facetype.get_type_from_normal = lambda normal, roof_angle=30, floor_angle=150: "Wall"

    class FakeHBFace:
        def __init__(self, identifier, geometry):
            self.identifier = identifier
            self.geometry = geometry
            self.apertures = []
            self.type = None

        def add_aperture(self, aperture):
            self.apertures.append(aperture)

        def apertures_by_ratio(self, ratio, tolerance=0.01):
            self.aperture_ratio = ratio

    class FakeHBRoom:
        def __init__(self, identifier, faces, tolerance=0.01, angle_tolerance=1):
            self.identifier = identifier
            self.faces = list(faces)
            self.story = None
            self.properties = types.SimpleNamespace(
                energy=types.SimpleNamespace(
                    program_type=None,
                    construction_set=None,
                    setpoint=None,
                    service_hot_water=None,
                    add_default_ideal_air=lambda: setattr(self, "ideal_air_added", True),
                )
            )

        def generate_grid(self, *args, **kwargs):
            return {"mesh": self.identifier}

        @staticmethod
        def solve_adjacency(rooms, tolerance=0.01):
            for room in rooms:
                room.adjacency_solved = True

    class FakeHBModel:
        def __init__(self, identifier, rooms=None, orphaned_shades=None):
            self.identifier = identifier
            self.rooms = list(rooms or [])
            self.orphaned_shades = list(orphaned_shades or [])
            self.properties = types.SimpleNamespace(
                radiance=types.SimpleNamespace(
                    sensor_grids=[],
                    add_sensor_grids=lambda grids: setattr(self.properties.radiance, "sensor_grids", list(grids)),
                )
            )

        def to_dict(self):
            return {
                "type": "Model",
                "identifier": self.identifier,
                "room_count": len(self.rooms),
                "shade_count": len(self.orphaned_shades),
            }

    class FakeHBShade:
        def __init__(self, identifier, geometry):
            self.identifier = identifier
            self.geometry = geometry

    class FakeHBAperture(FakeHBShade):
        pass

    class FakeHBDoor(FakeHBShade):
        pass

    face_mod = _new_module("honeybee.face")
    face_mod.Face = FakeHBFace
    model_mod = _new_module("honeybee.model")
    model_mod.Model = FakeHBModel
    room_mod = _new_module("honeybee.room")
    room_mod.Room = FakeHBRoom
    shade_mod = _new_module("honeybee.shade")
    shade_mod.Shade = FakeHBShade
    aperture_mod = _new_module("honeybee.aperture")
    aperture_mod.Aperture = FakeHBAperture
    door_mod = _new_module("honeybee.door")
    door_mod.Door = FakeHBDoor

    honeybee.facetype = facetype
    for name, module in {
        "honeybee": honeybee,
        "honeybee.facetype": facetype,
        "honeybee.face": face_mod,
        "honeybee.model": model_mod,
        "honeybee.room": room_mod,
        "honeybee.shade": shade_mod,
        "honeybee.aperture": aperture_mod,
        "honeybee.door": door_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    # ------------------------------------------------------------------
    # honeybee-energy
    # ------------------------------------------------------------------
    hb_energy = _new_module("honeybee_energy")
    hb_energy.__path__ = []
    hb_energy_lib = _new_module("honeybee_energy.lib")
    hb_energy_lib.__path__ = []
    constructionsets = _new_module("honeybee_energy.lib.constructionsets")
    constructionsets.CONSTRUCTION_SETS = ["Default Generic Construction Set", "High Performance"]
    constructionsets.construction_set_by_identifier = lambda identifier: {
        "kind": "construction_set",
        "identifier": identifier,
    }
    programtypes = _new_module("honeybee_energy.lib.programtypes")
    programtypes.PROGRAM_TYPES = ["Generic Office Program", "Residential"]
    programtypes.office_program = {"kind": "program", "identifier": "Office"}
    programtypes.program_type_by_identifier = lambda identifier: {
        "kind": "program_type",
        "identifier": identifier,
    }
    scheduletypelimits = _new_module("honeybee_energy.lib.scheduletypelimits")
    scheduletypelimits.temperature = "temperature"
    scheduletypelimits.humidity = "humidity"
    scheduletypelimits.fractional = "fractional"

    ruleset_mod = _new_module("honeybee_energy.schedule.ruleset")

    class FakeScheduleRuleset:
        def __init__(self, identifier, day_schedule=None, *args):
            self.identifier = identifier
            self.day_schedule = day_schedule

        @staticmethod
        def from_constant_value(identifier, value, schedule_type):
            return {"identifier": identifier, "value": value, "type": schedule_type}

    ruleset_mod.ScheduleRuleset = FakeScheduleRuleset
    day_mod = _new_module("honeybee_energy.schedule.day")

    class FakeScheduleDay:
        def __init__(self, identifier, values=None, times=None):
            self.identifier = identifier
            self.values = list(values or [])
            self.times = list(times or [])

    day_mod.ScheduleDay = FakeScheduleDay
    setpoint_mod = _new_module("honeybee_energy.load.setpoint")

    class FakeSetpoint:
        def __init__(self, identifier, *args):
            self.identifier = identifier
            self.args = args

    setpoint_mod.Setpoint = FakeSetpoint
    hotwater_mod = _new_module("honeybee_energy.load.hotwater")

    class FakeServiceHotWater:
        def __init__(self, identifier, flow_per_area, schedule):
            self.identifier = identifier
            self.flow_per_area = flow_per_area
            self.schedule = schedule

    hotwater_mod.ServiceHotWater = FakeServiceHotWater

    for name, module in {
        "honeybee_energy": hb_energy,
        "honeybee_energy.lib": hb_energy_lib,
        "honeybee_energy.lib.constructionsets": constructionsets,
        "honeybee_energy.lib.programtypes": programtypes,
        "honeybee_energy.lib.scheduletypelimits": scheduletypelimits,
        "honeybee_energy.schedule": _new_module("honeybee_energy.schedule"),
        "honeybee_energy.schedule.ruleset": ruleset_mod,
        "honeybee_energy.schedule.day": day_mod,
        "honeybee_energy.load": _new_module("honeybee_energy.load"),
        "honeybee_energy.load.setpoint": setpoint_mod,
        "honeybee_energy.load.hotwater": hotwater_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    # ------------------------------------------------------------------
    # honeybee-radiance, ladybug, ladybug-geometry
    # ------------------------------------------------------------------
    sensorgrid_mod = _new_module("honeybee_radiance.sensorgrid")

    class FakeSensorGrid:
        @staticmethod
        def from_mesh3d(identifier, mesh):
            return {"identifier": identifier, "mesh": mesh}

    sensorgrid_mod.SensorGrid = FakeSensorGrid
    monkeypatch.setitem(sys.modules, "honeybee_radiance", _new_module("honeybee_radiance"))
    monkeypatch.setitem(sys.modules, "honeybee_radiance.sensorgrid", sensorgrid_mod)

    ladybug_dt = _new_module("ladybug.dt")

    class FakeTime:
        def __init__(self, hour=0, minute=0):
            self.hour = hour
            self.minute = minute

    ladybug_dt.Time = FakeTime
    monkeypatch.setitem(sys.modules, "ladybug", _new_module("ladybug"))
    monkeypatch.setitem(sys.modules, "ladybug.dt", ladybug_dt)

    face3d_mod = _new_module("ladybug_geometry.geometry3d.face")

    class FakeFace3D:
        def __init__(self, boundary, holes=None, plane=None):
            self.boundary = list(boundary)
            self.holes = holes
            self.plane = plane

    pointvector_mod = _new_module("ladybug_geometry.geometry3d.pointvector")

    class FakePoint3D:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z

    class FakeVector3D(FakePoint3D):
        pass

    face3d_mod.Face3D = FakeFace3D
    pointvector_mod.Point3D = FakePoint3D
    pointvector_mod.Vector3D = FakeVector3D
    monkeypatch.setitem(sys.modules, "ladybug_geometry", _new_module("ladybug_geometry"))
    monkeypatch.setitem(sys.modules, "ladybug_geometry.geometry3d", _new_module("ladybug_geometry.geometry3d"))
    monkeypatch.setitem(sys.modules, "ladybug_geometry.geometry3d.face", face3d_mod)
    monkeypatch.setitem(sys.modules, "ladybug_geometry.geometry3d.pointvector", pointvector_mod)


def _install_topologicpy_geometry_fakes(monkeypatch):
    """Install minimal topologicpy submodules used by the HBJSON import path."""

    def make_topology(type_name, **data):
        obj = {"_type": type_name.lower()}
        obj.update(data)
        return obj

    vertex_mod = _new_module("topologicpy.Vertex")

    class FakeVertex:
        @staticmethod
        def ByCoordinates(x, y, z):
            return make_topology("vertex", coords=(float(x), float(y), float(z)))

        @staticmethod
        def X(vertex, mantissa=None):
            value = vertex["coords"][0]
            return round(value, mantissa) if mantissa is not None else value

        @staticmethod
        def Y(vertex, mantissa=None):
            value = vertex["coords"][1]
            return round(value, mantissa) if mantissa is not None else value

        @staticmethod
        def Z(vertex, mantissa=None):
            value = vertex["coords"][2]
            return round(value, mantissa) if mantissa is not None else value

        @staticmethod
        def Distance(a, b):
            return math.sqrt(sum((a["coords"][i] - b["coords"][i]) ** 2 for i in range(3)))

    vertex_mod.Vertex = FakeVertex

    edge_mod = _new_module("topologicpy.Edge")

    class FakeEdge:
        @staticmethod
        def ByVertices(*args, tolerance=0.0001, silent=False):
            if len(args) == 1 and isinstance(args[0], (list, tuple)):
                vertices = list(args[0])
            else:
                vertices = list(args[:2])
            if len(vertices) < 2:
                return None
            return make_topology("edge", vertices=vertices[:2])

    edge_mod.Edge = FakeEdge

    wire_mod = _new_module("topologicpy.Wire")

    class FakeWire:
        @staticmethod
        def ByEdges(edges, tolerance=0.0001, silent=False):
            edges = list(edges or [])
            return make_topology("wire", edges=edges) if edges else None

    wire_mod.Wire = FakeWire

    face_mod = _new_module("topologicpy.Face")

    class FakeFace:
        @staticmethod
        def ByWire(wire, tolerance=0.0001, silent=False):
            return make_topology("face", wire=wire) if wire else None

        @staticmethod
        def ExternalBoundary(face):
            return face.get("wire")

        @staticmethod
        def Normal(face, mantissa=6):
            return [0, 0, 1]

    face_mod.Face = FakeFace

    shell_mod = _new_module("topologicpy.Shell")

    class FakeShell:
        @staticmethod
        def ByFaces(faces, tolerance=0.001, silent=True):
            faces = list(faces or [])
            return make_topology("shell", faces=faces) if faces else None

    shell_mod.Shell = FakeShell

    cell_mod = _new_module("topologicpy.Cell")

    class FakeCell:
        @staticmethod
        def ByFaces(faces, tolerance=0.001, silent=True):
            faces = list(faces or [])
            return make_topology("cell", faces=faces) if faces else None

    cell_mod.Cell = FakeCell

    cluster_mod = _new_module("topologicpy.Cluster")

    class FakeCluster:
        @staticmethod
        def ByTopologies(topologies, silent=True):
            return make_topology("cluster", topologies=list(topologies or []))

    cluster_mod.Cluster = FakeCluster

    topology_mod = _new_module("topologicpy.Topology")

    class FakeTopology:
        @staticmethod
        def IsInstance(obj, type_name):
            if not isinstance(obj, dict) or "_type" not in obj:
                return False
            requested = str(type_name).lower()
            if requested == "topology":
                return True
            return obj.get("_type") == requested

        @staticmethod
        def SetDictionary(topology, dictionary):
            if isinstance(topology, dict):
                topology["dictionary"] = dict(dictionary or {})
            return topology

        @staticmethod
        def Dictionary(topology):
            return dict(topology.get("dictionary", {})) if isinstance(topology, dict) else {}

        @staticmethod
        def SelfMerge(cluster, tolerance=0.0001):
            edges = list(cluster.get("topologies", [])) if isinstance(cluster, dict) else []
            return make_topology("wire", edges=edges) if edges else None

        @staticmethod
        def Difference(host, tools):
            host["difference_called"] = True
            return host

        @staticmethod
        def Edges(topology):
            if isinstance(topology, dict):
                return list(topology.get("edges", []))
            return []

        @staticmethod
        def Vertices(topology, silent=False):
            if not isinstance(topology, dict):
                return []
            if topology.get("_type") == "wire":
                vertices = []
                for edge in topology.get("edges", []):
                    vertices.extend(edge.get("vertices", []))
                return vertices
            if topology.get("_type") == "edge":
                return list(topology.get("vertices", []))
            return list(topology.get("vertices", []))

        @staticmethod
        def InternalVertex(face):
            return make_topology("vertex", coords=(0.0, 0.0, 0.0))

        @staticmethod
        def TransferDictionariesBySelectors(cell, selectors, tranFaces=True, numWorkers=1):
            cell["selectors"] = list(selectors or [])
            return cell

        @staticmethod
        def Cells(topology):
            return []

        @staticmethod
        def Faces(topology):
            return []

        @staticmethod
        def SubTopologies(topology, subTopologyType="face"):
            return []

        @staticmethod
        def Apertures(topology):
            return []

        @staticmethod
        def CenterOfMass(topology):
            return make_topology("vertex", coords=(0.0, 0.0, 0.0))

    topology_mod.Topology = FakeTopology

    dictionary_mod = _new_module("topologicpy.Dictionary")

    class FakeDictionary:
        @staticmethod
        def ByKeysValues(keys, values):
            return dict(zip(keys or [], values or []))

        @staticmethod
        def Keys(dictionary):
            return list((dictionary or {}).keys()) if isinstance(dictionary, dict) else []

        @staticmethod
        def ValueAtKey(dictionary, key, default=None, silent=False):
            return (dictionary or {}).get(key, default) if isinstance(dictionary, dict) else default

        @staticmethod
        def SetValueAtKey(dictionary, key, value):
            out = dict(dictionary or {})
            out[key] = value
            return out

    dictionary_mod.Dictionary = FakeDictionary

    helper_mod = _new_module("topologicpy.Helper")

    class FakeHelper:
        @staticmethod
        def Flatten(value):
            if not isinstance(value, list):
                return [value]
            out = []
            for item in value:
                out.extend(FakeHelper.Flatten(item))
            return out

    helper_mod.Helper = FakeHelper

    aperture_mod = _new_module("topologicpy.Aperture")

    class FakeAperture:
        pass

    aperture_mod.Aperture = FakeAperture

    for name, module in {
        "topologicpy.Vertex": vertex_mod,
        "topologicpy.Edge": edge_mod,
        "topologicpy.Wire": wire_mod,
        "topologicpy.Face": face_mod,
        "topologicpy.Shell": shell_mod,
        "topologicpy.Cell": cell_mod,
        "topologicpy.Cluster": cluster_mod,
        "topologicpy.Topology": topology_mod,
        "topologicpy.Dictionary": dictionary_mod,
        "topologicpy.Helper": helper_mod,
        "topologicpy.Aperture": aperture_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def _load_honeybee(monkeypatch):
    _install_external_honeybee_fakes(monkeypatch)
    monkeypatch.delitem(sys.modules, "topologicpy.Honeybee", raising=False)
    return importlib.import_module("topologicpy.Honeybee")


def _square(identifier="face", z=0.0):
    return {
        "identifier": identifier,
        "geometry": {
            "boundary": [[0, 0, z], [1, 0, z], [1, 1, z], [0, 1, z]],
        },
    }


def test_import_uses_injected_dependencies_without_pip_install(monkeypatch):
    calls = []
    monkeypatch.setattr("os.system", lambda command: calls.append(command))

    module = _load_honeybee(monkeypatch)

    assert hasattr(module.Honeybee, "ByHBJSONDictionary")
    assert calls == []


def test_lookup_wrappers_string_and_export_to_hbjson(monkeypatch, tmp_path):
    module = _load_honeybee(monkeypatch)
    Honeybee = module.Honeybee

    assert Honeybee.ConstructionSetByIdentifier("Default Generic Construction Set") == {
        "kind": "construction_set",
        "identifier": "Default Generic Construction Set",
    }
    construction_sets, construction_ids = Honeybee.ConstructionSets()
    assert construction_ids == ["Default Generic Construction Set", "High Performance"]
    assert construction_sets[0]["identifier"] == "Default Generic Construction Set"

    assert Honeybee.ProgramTypeByIdentifier("Generic Office Program") == {
        "kind": "program_type",
        "identifier": "Generic Office Program",
    }
    program_types, program_ids = Honeybee.ProgramTypes()
    assert program_ids == ["Generic Office Program", "Residential"]
    assert program_types[1]["identifier"] == "Residential"

    model = module.HBModel("Exported", rooms=[object()], orphaned_shades=[object()])
    assert Honeybee.String(model) == {
        "type": "Model",
        "identifier": "Exported",
        "room_count": 1,
        "shade_count": 1,
    }

    out_base = tmp_path / "model_without_extension"
    assert Honeybee.ExportToHBJSON(model, str(out_base), overwrite=True) is True
    written_path = tmp_path / "model_without_extension.hbjson"
    assert written_path.exists()
    assert json.loads(written_path.read_text())["identifier"] == "Exported"
    assert Honeybee.ExportToHBJSON(model, str(written_path), overwrite=False) is None


def test_by_hbjson_dictionary_rejects_invalid_input_and_returns_standard_shape(monkeypatch):
    module = _load_honeybee(monkeypatch)
    _install_topologicpy_geometry_fakes(monkeypatch)
    Honeybee = module.Honeybee

    assert Honeybee.ByHBJSONDictionary(None, silent=True) is None

    result = Honeybee.ByHBJSONDictionary(
        {"properties": {"radiance": {"grid": True}, "energy": {"loads": True}}},
        silent=True,
    )

    assert sorted(result.keys()) == sorted([
        "rooms",
        "faces",
        "shades",
        "apertures",
        "doors",
        "orphanedRooms",
        "orphanedFaces",
        "orphanedShades",
        "orphanedApertures",
        "orphanedDoors",
        "properties",
    ])
    assert result["rooms"] == []
    assert result["properties"] == {"radiance": {"grid": True}, "energy": {"loads": True}}


def test_by_hbjson_string_and_path_delegate_flags(monkeypatch, tmp_path):
    module = _load_honeybee(monkeypatch)
    Honeybee = module.Honeybee
    calls = []

    def fake_by_dictionary(dictionary, **kwargs):
        calls.append((dictionary, kwargs))
        return {"ok": True, "dictionary": dictionary, "kwargs": kwargs}

    monkeypatch.setattr(Honeybee, "ByHBJSONDictionary", staticmethod(fake_by_dictionary))

    assert Honeybee.ByHBJSONString(123, silent=True) is None
    payload = {"rooms": [], "properties": {}}
    result = Honeybee.ByHBJSONString(
        json.dumps(payload),
        includeRooms=False,
        includeDoors=False,
        tolerance=0.25,
        silent=True,
    )
    assert result["ok"] is True
    assert calls[-1][0] == payload
    assert calls[-1][1]["includeRooms"] is False
    assert calls[-1][1]["includeDoors"] is False
    assert calls[-1][1]["tolerance"] == 0.25

    assert Honeybee.ByHBJSONPath("", silent=True) is None
    path = tmp_path / "model.hbjson"
    path.write_text(json.dumps(payload))
    path_result = Honeybee.ByHBJSONPath(str(path), includeFaces=False, silent=True)
    assert path_result["ok"] is True
    assert calls[-1][1]["includeFaces"] is False


def test_hbjson_dictionary_parses_orphaned_geometry_best_effort(monkeypatch):
    module = _load_honeybee(monkeypatch)
    _install_topologicpy_geometry_fakes(monkeypatch)
    Honeybee = module.Honeybee

    hbjson = {
        "orphaned_faces": [_square("of-1")],
        "faces": [_square("top-face")],
        "orphaned_shades": [_square("os-1")],
        "context_geometry": {"shades": [_square("ctx-1")]},
        "shades": [_square("shade-top")],
        "orphaned_apertures": [_square("oa-1")],
        "orphaned_doors": [_square("od-1")],
    }

    result = Honeybee.ByHBJSONDictionary(hbjson, silent=True)

    assert len(result["orphanedFaces"]) == 2
    assert len(result["orphanedShades"]) == 3
    assert len(result["orphanedApertures"]) == 1
    assert len(result["orphanedDoors"]) == 1
    assert result["orphanedFaces"][0]["dictionary"]["identifier"] == "of-1"
    assert result["orphanedApertures"][0]["dictionary"]["identifier"] == "oa-1"


def test_hbjson_room_faces_apertures_doors_and_shades_best_effort(monkeypatch):
    module = _load_honeybee(monkeypatch)
    _install_topologicpy_geometry_fakes(monkeypatch)
    Honeybee = module.Honeybee

    room_face = _square("host-face")
    room_face["apertures"] = [_square("window-1")]
    room_face["doors"] = [_square("door-1")]
    hbjson = {
        "rooms": [{
            "identifier": "room-1",
            "faces": [room_face],
            "shades": [_square("room-shade")],
        }]
    }

    result = Honeybee.ByHBJSONDictionary(hbjson, silent=True)

    assert len(result["rooms"]) == 1
    assert result["rooms"][0]["_type"] == "cell"
    assert result["rooms"][0]["dictionary"]["identifier"] == "room-1"
    assert len(result["faces"]) == 1
    assert len(result["apertures"]) == 1
    assert len(result["doors"]) == 1
    assert result["faces"][0]["dictionary"]["identifier"] == "host-face"
    assert result["apertures"][0]["dictionary"]["identifier"] == "window-1"
    assert result["doors"][0]["dictionary"]["identifier"] == "door-1"


def test_model_by_topology_rejects_non_topology_input(monkeypatch):
    module = _load_honeybee(monkeypatch)
    _install_topologicpy_geometry_fakes(monkeypatch)

    assert module.Honeybee.ModelByTopology(None) is None
