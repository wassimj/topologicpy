"""Dependency failures must never mutate the running Python environment."""
import ast
import builtins
import importlib
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def block_import(monkeypatch, package):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == package or name.startswith(package + "."):
            raise ModuleNotFoundError(f"Blocked optional dependency: {name}", name=name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


@pytest.fixture(autouse=True)
def forbid_installation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Library code attempted to launch a shell or subprocess")
    monkeypatch.setattr("os.system", forbidden)
    monkeypatch.setattr("subprocess.run", forbidden)
    monkeypatch.setattr("subprocess.Popen", forbidden)


@pytest.mark.parametrize("module,package", [
    ("Sun", "ephem"), ("Honeybee", "honeybee"),
    ("Kuzu", "kuzu"), ("LadybugDB", "real_ladybug"),
])
def test_optional_module_import_does_not_require_dependency(monkeypatch, module, package):
    block_import(monkeypatch, package)
    importlib.reload(importlib.import_module("topologicpy." + module))


@pytest.mark.parametrize("method,args", [
    ("Azimuth", (51.5, 0, datetime(2026, 6, 21))),
    ("Altitude", (51.5, 0, datetime(2026, 6, 21))),
    ("Sunrise", (51.5, 0, datetime(2026, 6, 21))),
    ("Sunset", (51.5, 0, datetime(2026, 6, 21))),
    ("WinterSolstice", (51.5, 2026)), ("SummerSolstice", (51.5, 2026)),
    ("SpringEquinox", (51.5, 2026)), ("AutumnEquinox", (51.5, 2026)),
])
def test_sun_missing_ephem_raises_install_hint(monkeypatch, method, args):
    from topologicpy.Sun import Sun
    block_import(monkeypatch, "ephem")
    with pytest.raises(ImportError, match=r"python -m pip install.*topologicpy\[sun\]") as error:
        getattr(Sun, method)(*args)
    assert isinstance(error.value.__cause__, ModuleNotFoundError)


@pytest.mark.parametrize("module,package,extra", [
    ("Kuzu", "kuzu", "kuzu"), ("LadybugDB", "real_ladybug", "ladybugdb"),
])
@pytest.mark.parametrize("method", ["Database", "Manager", "Connection"])
def test_database_dependency_error_is_not_swallowed(monkeypatch, module, package, extra, method):
    cls = getattr(importlib.import_module("topologicpy." + module), module)
    block_import(monkeypatch, package)
    if module == "LadybugDB":
        block_import(monkeypatch, "ladybug")
    with pytest.raises(ImportError, match=rf"topologicpy\[{extra}\]"):
        getattr(cls, method)(None, silent=True)


def test_honeybee_missing_dependency_raises_hint(monkeypatch):
    from topologicpy.Honeybee import Honeybee
    block_import(monkeypatch, "honeybee_energy")
    with pytest.raises(ImportError, match=r"topologicpy\[honeybee\]"):
        Honeybee.ConstructionSets()


def test_graph_missing_dependency_raises_hint(monkeypatch):
    from topologicpy.Graph import Graph
    block_import(monkeypatch, "networkx")
    with pytest.raises(ImportError, match=r"topologicpy\[graph\]"):
        Graph.NetworkXGraph(None, silent=True)


def test_required_dependency_has_install_hint(monkeypatch):
    from topologicpy.Color import Color
    block_import(monkeypatch, "webcolors")
    with pytest.raises(ImportError, match=r"python -m pip install.*webcolors"):
        Color._import_webcolors("test")


def test_package_has_no_automatic_installation_commands():
    for path in (ROOT / "src/topologicpy").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Install hints are fine; invoking an installer is not.
                name = ast.unparse(node.func)
                assert name != "os.system", str(path)
                if name.startswith("subprocess."):
                    assert not any(isinstance(n, ast.Constant) and isinstance(n.value, str)
                                   and ("pip install" in n.value or n.value in ("pip", "ensurepip"))
                                   for n in ast.walk(node)), str(path)


def test_optional_extras_are_declared():
    try:
        import tomllib
    except ImportError:
        tomllib = pytest.importorskip("tomli")
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    for extra in ("sun", "graph", "honeybee", "kuzu", "ladybugdb", "ifc"):
        assert extras[extra]
    assert "ephem" in extras["sun"]
    assert "honeybee-core" in extras["honeybee"]
    assert "ladybug-core" in extras["honeybee"]
