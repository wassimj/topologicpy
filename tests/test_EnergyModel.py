"""Unit tests for topologicpy.EnergyModel.

These tests avoid requiring a real OpenStudio installation. OpenStudio-dependent
paths are exercised through lightweight stand-ins and monkeypatching.
"""

import importlib
import inspect
import os
import subprocess
import types
import warnings

import pytest

from topologicpy.EnergyModel import EnergyModel


@pytest.fixture(autouse=True)
def _suppress_expected_topologicpy_output(capfd):
    """Keep expected TopologicPy diagnostic prints out of normal pytest output."""
    capfd.readouterr()
    yield
    capfd.readouterr()


class FakeOptional:
    def __init__(self, value=None, initialized=True):
        self._value = value
        self._initialized = initialized

    def is_initialized(self):
        return self._initialized

    def isNull(self):
        return not self._initialized

    def get(self):
        return self._value


class FakeNamed:
    def __init__(self, name):
        self._name = name

    def name(self):
        return FakeOptional(self._name)


class FakeColor:
    def __init__(self, red=1, green=2, blue=3):
        self._red = red
        self._green = green
        self._blue = blue

    def renderingRedValue(self):
        return self._red

    def renderingGreenValue(self):
        return self._green

    def renderingBlueValue(self):
        return self._blue


class FakeSpaceType(FakeNamed):
    def __init__(self, name, color=None, color_initialized=True):
        super().__init__(name)
        self._color = color
        self._color_initialized = color_initialized

    def renderingColor(self):
        return FakeOptional(self._color, self._color_initialized)


class FakeSQL:
    def execAndReturnVectorOfString(self, query):
        if "ColumnName" in query:
            return FakeOptional(["Heating", "Cooling", "Heating"])
        if "ReportName" in query and "WHERE" not in query:
            return FakeOptional(["Annual", "Annual", "HVAC"])
        if "RowName" in query:
            return FakeOptional(["Zone A", "Zone B", "Zone A"])
        if "TableName" in query:
            return FakeOptional(["Summary", "Summary", "Loads"])
        return FakeOptional([])

    def execAndReturnFirstString(self, query):
        if "Units" in query:
            return FakeOptional("W")
        return FakeOptional(None, initialized=False)

    def execAndReturnFirstDouble(self, query):
        if "Zone A" in query:
            return FakeOptional(123.5)
        if "Zone B" in query:
            return FakeOptional(456.5)
        return FakeOptional(None, initialized=False)


class FakeModelWithSQL:
    def __init__(self, sql=None, initialized=True):
        self._sql = sql if sql is not None else FakeSQL()
        self._initialized = initialized

    def sqlFile(self):
        return FakeOptional(self._sql, initialized=self._initialized)


class FakeGbXMLTranslator:
    def modelToGbXML(self, model, path):
        model["gbxml_path"] = path
        return True

    def modelToGbXMLString(self, model):
        return "<gbXML/>"


class FakeOpenStudioForExport:
    gbxml = types.SimpleNamespace(GbXMLForwardTranslator=FakeGbXMLTranslator)
    openstudioutilitiescore = types.SimpleNamespace(toPath=lambda p: "OSPATH:" + str(p))


class FakeWorkflow:
    def __init__(self):
        self.seed_file = None
        self.weather_file = None
        self.saved_as = None

    def setSeedFile(self, path):
        self.seed_file = path

    def setWeatherFile(self, path):
        self.weather_file = path

    def saveAs(self, path):
        self.saved_as = path


class FakeRunModel:
    def __init__(self):
        self.workflow = FakeWorkflow()
        self.saved = []
        self.sql_file = None

    def getBuilding(self):
        return types.SimpleNamespace(name=lambda: FakeOptional("RunBuilding"))

    def save(self, path, overwrite):
        self.saved.append((path, overwrite))
        return True

    def workflowJSON(self):
        return self.workflow

    def setSqlFile(self, sql_file):
        self.sql_file = sql_file


class FakeOpenStudioForRun:
    openstudioutilitiescore = types.SimpleNamespace(toPath=lambda p: str(p))

    @staticmethod
    def SqlFile(path):
        return {"sql_path": path}


def _fake_openstudio_for_osm(model_value="MODEL"):
    class FakeVersionTranslator:
        def loadModel(self, path):
            return FakeOptional(str(model_value) + ":" + str(path))

    return types.SimpleNamespace(
        osversion=types.SimpleNamespace(VersionTranslator=FakeVersionTranslator),
        openstudioutilitiescore=types.SimpleNamespace(toPath=lambda p: "OSPATH:" + str(p)),
    )












def test_sql_query_helpers_deduplicate_and_preserve_order():
    model = FakeModelWithSQL()
    assert EnergyModel.ColumnNames(model, "Annual", "Summary") == ["Heating", "Cooling"]
    assert EnergyModel.ReportNames(model) == ["Annual", "HVAC"]
    assert EnergyModel.RowNames(model, "Annual", "Summary") == ["Zone A", "Zone B"]
    assert EnergyModel.TableNames(model, "Annual") == ["Summary", "Loads"]
    assert EnergyModel.Units(model, "Annual", "Summary", "Heating") == "W"
    assert EnergyModel.Query(model, rowNames=["Zone A", "Zone B", "Missing"]) == [123.5, 456.5, None]
    assert EnergyModel.Query(model, rowNames="Zone A") == [123.5]


def test_default_construction_and_schedule_sets_return_sets_and_names():
    class FakeSetModel:
        def getDefaultConstructionSets(self):
            return [FakeNamed("C1"), FakeNamed("C2")]

        def getDefaultScheduleSets(self):
            return [FakeNamed("S1")]

    construction_sets, construction_names = EnergyModel.DefaultConstructionSets(FakeSetModel())
    schedule_sets, schedule_names = EnergyModel.DefaultScheduleSets(FakeSetModel())
    assert len(construction_sets) == 2
    assert construction_names == ["C1", "C2"]
    assert len(schedule_sets) == 1
    assert schedule_names == ["S1"]




def test_by_osm_path_invalid_path_and_stubbed_success(monkeypatch, tmp_path):
    fake_openstudio = _fake_openstudio_for_osm()
    monkeypatch.setattr(EnergyModel, "_ImportOpenStudio", staticmethod(lambda *args, **kwargs: fake_openstudio))

    assert EnergyModel.ByOSMPath(str(tmp_path / "missing.osm")) is None

    osm_path = tmp_path / "model.osm"
    osm_path.write_text("OSM")
    assert EnergyModel.ByOSMPath(str(osm_path)) == "MODEL:OSPATH:" + str(osm_path)


def test_export_to_gbxml_handles_extension_overwrite_and_translator(monkeypatch, tmp_path):
    monkeypatch.setattr(EnergyModel, "_ImportOpenStudio", staticmethod(lambda *args, **kwargs: FakeOpenStudioForExport))

    existing = tmp_path / "existing.xml"
    existing.write_text("old")
    assert EnergyModel.ExportToGBXML({}, str(existing), overwrite=False) is None

    model = {}
    out_path = tmp_path / "export_target"
    assert EnergyModel.ExportToGBXML(model, str(out_path), overwrite=True) is True
    assert str(model["gbxml_path"]).endswith("export_target.xml")


def test_export_to_osm_handles_extension_overwrite_and_model_save(monkeypatch, tmp_path):
    monkeypatch.setattr(EnergyModel, "_ImportOpenStudio", staticmethod(lambda *args, **kwargs: types.SimpleNamespace()))

    existing = tmp_path / "existing.osm"
    existing.write_text("old")
    assert EnergyModel.ExportToOSM(types.SimpleNamespace(save=lambda *a: True), str(existing), overwrite=False) is None

    calls = []

    class SaveModel:
        def save(self, path, overwrite):
            calls.append((path, overwrite))
            return True

    out_path = tmp_path / "model_without_ext"
    assert EnergyModel.ExportToOSM(SaveModel(), str(out_path), overwrite=True) is True
    assert calls[0][0].endswith("model_without_ext.osm")
    assert calls[0][1] is True


def test_gbxml_string_uses_translator_when_openstudio_is_available(monkeypatch):
    monkeypatch.setattr(EnergyModel, "_ImportOpenStudio", staticmethod(lambda *args, **kwargs: FakeOpenStudioForExport))
    assert EnergyModel.GBXMLString({}) == "<gbXML/>"


def test_openstudio_dependent_methods_return_none_when_openstudio_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(EnergyModel, "_ImportOpenStudio", staticmethod(lambda *args, **kwargs: None))
    assert EnergyModel.ByOSMPath(str(tmp_path / "x.osm")) is None
    assert EnergyModel.ExportToGBXML({}, str(tmp_path / "x.xml"), overwrite=True) is None
    assert EnergyModel.ExportToOSM(types.SimpleNamespace(save=lambda *a: True), str(tmp_path / "x.osm"), overwrite=True) is None
    assert EnergyModel.GBXMLString({}) is None
    assert EnergyModel.Run(object(), weatherFilePath=str(tmp_path / "missing.epw")) is None
    assert EnergyModel.Version(check=False, silent=True) is None


def test_run_returns_none_for_missing_weather_or_missing_binary(monkeypatch, tmp_path):
    monkeypatch.setattr(EnergyModel, "_ImportOpenStudio", staticmethod(lambda *args, **kwargs: FakeOpenStudioForRun))
    monkeypatch.setattr("shutil.which", lambda exe: None)

    model = FakeRunModel()
    with pytest.warns(UserWarning, match="Weather file not found"):
        assert EnergyModel.Run(model, weatherFilePath=str(tmp_path / "missing.epw"), outputFolder=str(tmp_path)) is None

    weather = tmp_path / "weather.epw"
    weather.write_text("weather")
    with pytest.warns(UserWarning, match="Could not find the OpenStudio executable"):
        assert EnergyModel.Run(model, weatherFilePath=str(weather), outputFolder=str(tmp_path)) is None


def test_run_creates_workflow_invokes_subprocess_and_attaches_sql(monkeypatch, tmp_path):
    monkeypatch.setattr(EnergyModel, "_ImportOpenStudio", staticmethod(lambda *args, **kwargs: FakeOpenStudioForRun))
    monkeypatch.setattr("shutil.which", lambda exe: "/usr/bin/openstudio")

    run_calls = []

    def fake_run(cmd, capture_output=True, text=True):
        run_calls.append((cmd, capture_output, text))
        osw_path = cmd[-1]
        output_folder = os.path.dirname(osw_path)
        sql_folder = os.path.join(output_folder, "run")
        os.makedirs(sql_folder, exist_ok=True)
        with open(os.path.join(sql_folder, "eplusout.sql"), "w", encoding="utf-8") as f:
            f.write("sql")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    weather = tmp_path / "weather.epw"
    weather.write_text("weather")
    model = FakeRunModel()
    result = EnergyModel.Run(model, weatherFilePath=str(weather), outputFolder=str(tmp_path))

    assert result is model
    assert run_calls
    assert run_calls[0][0][0] == "/usr/bin/openstudio"
    assert run_calls[0][0][1:3] == ["run", "-w"]
    assert model.saved
    assert model.workflow.seed_file.endswith("RunBuilding.osm")
    assert model.workflow.weather_file == str(weather)
    assert model.workflow.saved_as.endswith("RunBuilding.osw")
    assert model.sql_file["sql_path"].endswith(os.path.join("run", "eplusout.sql"))


def test_run_returns_none_when_subprocess_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(EnergyModel, "_ImportOpenStudio", staticmethod(lambda *args, **kwargs: FakeOpenStudioForRun))
    monkeypatch.setattr("shutil.which", lambda exe: "/usr/bin/openstudio")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: types.SimpleNamespace(returncode=1, stdout="", stderr="bad"))

    weather = tmp_path / "weather.epw"
    weather.write_text("weather")
    with pytest.warns(UserWarning, match="OpenStudio simulation failed"):
        assert EnergyModel.Run(FakeRunModel(), weatherFilePath=str(weather), outputFolder=str(tmp_path)) is None


def test_topologies_none_returns_empty_payload():
    payload = EnergyModel.Topologies(None)
    assert payload == {"cells": [], "apertures": [], "shadingFaces": []}


def test_version_returns_openstudio_version_without_pypi_check(monkeypatch):
    fake_openstudio = types.SimpleNamespace(openStudioVersion=lambda: "3.9.0")
    monkeypatch.setattr(EnergyModel, "_ImportOpenStudio", staticmethod(lambda *args, **kwargs: fake_openstudio))
    assert EnergyModel.Version(check=False, silent=True) == "3.9.0"
