# Copyright (C) 2026
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3.0 of the License, or (at your option)
# any later version.

"""STEP CAD/BREP exchange codec.

STEP is an interchange representation, not TopologicPy's native persistence
format.  This codec therefore preserves OCCT BREP/analytic/NURBS geometry as
far as the STEP standard and OCCT translator permit, but it does not promise
byte-identical BREP, stable subtopology identity, or TopologicPy
Content/Context/dictionary roundtripping.  Use ``.tpy`` for those guarantees.

TopologicPy-authored STEP files carry a standard ISO 10303-21 comment with the
original root topology type.  The hint is semantically ignored by other STEP
processors and is used only to recover translator-ambiguous distinctions such
as standalone Face versus one-face Shell on a direct TopologicPy roundtrip.

No tessellation is performed by this codec.  STEP writing uses
``STEPControl_AsIs`` and explicitly disables tessellated STEP entities.
"""

from __future__ import annotations

import math
import os
import re
import tempfile
import threading
from typing import Any


_SCHEMAS = {"AP203", "AP214CD", "AP214DIS", "AP214IS", "AP242DIS"}
_UNITS = {"INCH", "MM", "FT", "MI", "M", "KM", "MIL", "UM", "CM", "UIN"}
_ASSEMBLY_MODES = {"off": 0, "on": 1, "auto": 2}
_INTERFACE_LOCK = threading.RLock()
_ROOT_TYPE_MARKER = "TOPOLOGICPY_STEP_ROOT_TYPE"
_ROOT_TYPES = {"VERTEX", "EDGE", "WIRE", "FACE", "SHELL", "CELL", "CELLCOMPLEX", "CLUSTER"}


class STEPError(RuntimeError):
    """Raised when a STEP import/export operation cannot be completed."""


def _normalise_schema(schema: str) -> str:
    if not isinstance(schema, str):
        raise STEPError("The STEP schema must be a string.")
    value = schema.strip().upper()
    aliases = {
        "203": "AP203",
        "214CD": "AP214CD",
        "214DIS": "AP214DIS",
        "214IS": "AP214IS",
        "242": "AP242DIS",
        "242DIS": "AP242DIS",
    }
    value = aliases.get(value, value)
    if value not in _SCHEMAS:
        raise STEPError(
            "Unsupported STEP schema. Expected one of: "
            + ", ".join(sorted(_SCHEMAS))
            + "."
        )
    return value


def _normalise_unit(unit: str) -> str:
    if not isinstance(unit, str):
        raise STEPError("The STEP unit must be a string.")
    value = unit.strip().upper()
    aliases = {
        "MILLIMETER": "MM",
        "MILLIMETERS": "MM",
        "MILLIMETRE": "MM",
        "MILLIMETRES": "MM",
        "CENTIMETER": "CM",
        "CENTIMETERS": "CM",
        "CENTIMETRE": "CM",
        "CENTIMETRES": "CM",
        "METER": "M",
        "METERS": "M",
        "METRE": "M",
        "METRES": "M",
        "MICROMETER": "UM",
        "MICROMETERS": "UM",
        "MICROMETRE": "UM",
        "MICROMETRES": "UM",
        "IN": "INCH",
        "INCHES": "INCH",
        "FOOT": "FT",
        "FEET": "FT",
    }
    value = aliases.get(value, value)
    if value not in _UNITS:
        raise STEPError(
            "Unsupported STEP unit. Expected one of: "
            + ", ".join(sorted(_UNITS))
            + "."
        )
    return value


def _normalise_assembly(assembly) -> int:
    if isinstance(assembly, bool):
        return 1 if assembly else 0
    if isinstance(assembly, int) and assembly in (0, 1, 2):
        return int(assembly)
    if isinstance(assembly, str):
        value = assembly.strip().lower()
        if value in _ASSEMBLY_MODES:
            return _ASSEMBLY_MODES[value]
    raise STEPError("The assembly parameter must be False, True, or 'auto'.")


def _normalise_tolerance(tolerance):
    if tolerance is None:
        return None
    try:
        value = float(tolerance)
    except Exception as exc:
        raise STEPError("The tolerance parameter must be a positive number or None.") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise STEPError("The tolerance parameter must be a positive finite number.")
    return value


def _output_path(path) -> str:
    try:
        value = os.fspath(path)
    except Exception as exc:
        raise STEPError("The output path is not valid.") from exc
    root, extension = os.path.splitext(value)
    if not extension:
        return value + ".step"
    if extension.lower() not in (".step", ".stp"):
        raise STEPError("A STEP output path must end in .step or .stp.")
    return value


def _input_path(path) -> str:
    try:
        value = os.fspath(path)
    except Exception as exc:
        raise STEPError("The input path is not valid.") from exc
    if os.path.isfile(value):
        return value
    root, extension = os.path.splitext(value)
    if not extension:
        for suffix in (".step", ".stp"):
            candidate = value + suffix
            if os.path.isfile(candidate):
                return candidate
    raise STEPError("The STEP input file does not exist.")



def _topology_root_type(topology) -> str | None:
    """Return a stable TopologicPy root topology type name for STEP hinting."""
    try:
        from topologicpy.Topology import Topology
        value = Topology.TypeAsString(topology, silent=True)
    except Exception:
        value = None
    if not isinstance(value, str):
        return None
    value = value.replace(" ", "").strip().upper()
    return value if value in _ROOT_TYPES else None


def _inject_root_type_hint(path: str, root_type: str | None) -> None:
    """Insert a standard Part 21 comment carrying the TopologicPy root type.

    ISO 10303-21 comments are semantically ignored by STEP processors, so this
    hint does not alter the CAD model.  It exists only to resolve inherently
    ambiguous direct roundtrips such as TopoDS_Face ->
    SHELL_BASED_SURFACE_MODEL -> TopoDS_Shell.
    """
    if root_type not in _ROOT_TYPES:
        return
    with open(path, "rb") as stream:
        data = stream.read()
    token = b"ISO-10303-21;"
    index = data.find(token)
    if index < 0:
        raise STEPError("OCCT produced a STEP file without the ISO-10303-21 header.")
    marker = f"/* {_ROOT_TYPE_MARKER}={root_type} */".encode("ascii")
    if marker in data:
        return
    position = index + len(token)
    data = data[:position] + b"\n" + marker + b"\n" + data[position:]
    with open(path, "wb") as stream:
        stream.write(data)


def _read_root_type_hint(path: str) -> str | None:
    """Read a TopologicPy root-type hint without relying on the STEP parser."""
    try:
        with open(path, "rb") as stream:
            data = stream.read(65536)
    except Exception:
        return None
    match = re.search(
        rb"/\*\s*TOPOLOGICPY_STEP_ROOT_TYPE\s*=\s*([A-Z]+)\s*\*/",
        data.upper(),
    )
    if match is None:
        return None
    try:
        value = match.group(1).decode("ascii")
    except Exception:
        return None
    return value if value in _ROOT_TYPES else None


def _restore_root_type_hint(shape, root_type: str | None):
    """Restore only STEP topology distinctions known to be translator-ambiguous.

    OCCT maps a standalone Face and a Shell through STEP's
    ShellBasedSurfaceModel representation.  A TopologicPy-authored root-type
    hint lets us distinguish those two direct roundtrips.  Unhinted third-party
    STEP remains conservative and is never collapsed merely because a Shell has
    one Face.
    """
    if shape is None or root_type not in _ROOT_TYPES:
        return shape
    try:
        from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SHELL, TopAbs_WIRE
        shape_type = shape.ShapeType()
    except Exception:
        return shape

    if root_type == "FACE" and shape_type == TopAbs_SHELL:
        children = _direct_children(shape)
        faces = []
        for child in children:
            try:
                if child.ShapeType() == TopAbs_FACE:
                    faces.append(child)
            except Exception:
                pass
        if len(children) == 1 and len(faces) == 1:
            return faces[0]

    # Some STEP translators package a single standalone edge as a one-edge Wire.
    # This is the same class of ambiguity and is safe to resolve only when the
    # TopologicPy-authored hint explicitly says the source root was an Edge.
    if root_type == "EDGE" and shape_type == TopAbs_WIRE:
        children = _direct_children(shape)
        edges = []
        for child in children:
            try:
                if child.ShapeType() == TopAbs_EDGE:
                    edges.append(child)
            except Exception:
                pass
        if len(children) == 1 and len(edges) == 1:
            return edges[0]

    return shape


def _snapshot_interface(Interface_Static):
    """Capture only the process-global translator settings changed by this codec."""
    return {
        "write.step.schema": ("c", Interface_Static.CVal("write.step.schema")),
        "write.step.unit": ("c", Interface_Static.CVal("write.step.unit")),
        "xstep.cascade.unit": ("c", Interface_Static.CVal("xstep.cascade.unit")),
        "write.step.assembly": ("i", Interface_Static.IVal("write.step.assembly")),
        "write.step.tessellated": ("i", Interface_Static.IVal("write.step.tessellated")),
        "write.surfacecurve.mode": ("i", Interface_Static.IVal("write.surfacecurve.mode")),
    }


def _restore_interface(Interface_Static, state) -> None:
    for name, (kind, value) in state.items():
        if kind == "c":
            Interface_Static.SetCVal(name, value)
        else:
            Interface_Static.SetIVal(name, int(value))


def _direct_children(shape):
    """Return the direct OCCT children of ``shape`` without recursive exploration."""
    try:
        from OCC.Core.TopoDS import TopoDS_Iterator
    except Exception:
        return []

    children = []
    try:
        iterator = TopoDS_Iterator(shape)
        while iterator.More():
            child = iterator.Value()
            if child is not None and not child.IsNull():
                children.append(child)
            iterator.Next()
    except Exception:
        return []
    return children


def _normalise_transport_shape(shape):
    """Remove STEP/OCCT transport compounds that contain exactly one child.

    STEP product and representation translation can introduce TopoDS_Compound
    packaging around a single logical BREP result.  Such a wrapper is not a
    meaningful TopologicPy Cluster.  Only compounds with exactly one *direct*
    child are unwrapped, recursively.  Shells, CompSolids and compounds with
    multiple children are deliberately preserved because they can represent
    genuine CAD topology or assemblies.
    """
    try:
        from OCC.Core.TopAbs import TopAbs_COMPOUND
    except Exception:
        return shape

    current = shape
    seen = 0
    while current is not None and seen < 64:
        seen += 1
        try:
            if current.IsNull() or current.ShapeType() != TopAbs_COMPOUND:
                break
        except Exception:
            break
        children = _direct_children(current)
        if len(children) != 1:
            break
        current = children[0]
    return current



def _cell_volume_sum(topology) -> float | None:
    """Return the summed volume of all Cells contained in ``topology``.

    STEP can reconstruct a hinted CellComplex root as a Compound/Cluster of
    independent Solids.  Summing the constituent Cell volumes provides a stable
    geometric invariant across that transport representation and the rebuilt
    CellComplex.
    """
    try:
        from topologicpy.Cell import Cell
        from topologicpy.Topology import Topology

        cells = Topology.Cells(topology, silent=True) or []
        if Topology.IsInstance(topology, "Cell"):
            cells = [topology]
        if not cells:
            return None

        total = 0.0
        for cell in cells:
            value = Cell.Volume(cell, mantissa=None, silent=True)
            if value is None:
                return None
            value = float(value)
            if not math.isfinite(value):
                return None
            total += value
        return total
    except Exception:
        return None


def _restore_cellcomplex_root_type(topology, root_type: str | None, silent: bool = False):
    """Reconstruct a TopologicPy-authored CellComplex from imported STEP solids.

    OCCT/STEP can preserve all constituent Solids of a CompSolid while returning
    them inside a Compound.  The TopologicPy root-type hint gives us provenance
    that the authored root was a CellComplex, but reconstruction is accepted only
    when structural and geometric invariants validate the result.  Unhinted STEP
    is never promoted.
    """
    if root_type != "CELLCOMPLEX" or topology is None:
        return topology

    try:
        from topologicpy.CellComplex import CellComplex
        from topologicpy.Topology import Topology
    except Exception:
        return topology

    if Topology.IsInstance(topology, "CellComplex"):
        return topology
    if not Topology.IsInstance(topology, "Cluster"):
        return topology

    source_cells = Topology.Cells(topology, silent=True) or []
    if len(source_cells) < 2:
        if not silent:
            print(
                "STEP.Load - Warning: The STEP root was authored as a CellComplex, "
                "but fewer than two Cells were recovered. Returning the imported "
                "topology unchanged."
            )
        return topology

    source_volume = _cell_volume_sum(topology)
    if source_volume is None:
        if not silent:
            print(
                "STEP.Load - Warning: Could not validate imported Cell volumes for "
                "CellComplex reconstruction. Returning the imported Cluster."
            )
        return topology

    try:
        rebuilt = CellComplex.ByCells(
            source_cells,
            transferDictionaries=False,
            tolerance=0.0001,
            silent=True,
        )
    except Exception:
        rebuilt = None

    if not Topology.IsInstance(rebuilt, "CellComplex"):
        if not silent:
            print(
                "STEP.Load - Warning: STEP preserved the constituent Cells, but they "
                "could not be reconstructed as a CellComplex. Returning the imported "
                "Cluster."
            )
        return topology

    rebuilt_cells = Topology.Cells(rebuilt, silent=True) or []
    if len(rebuilt_cells) != len(source_cells):
        if not silent:
            print(
                "STEP.Load - Warning: CellComplex reconstruction changed the Cell "
                "count. Returning the imported Cluster."
            )
        return topology

    rebuilt_volume = _cell_volume_sum(rebuilt)
    if rebuilt_volume is None or not math.isclose(
        rebuilt_volume,
        source_volume,
        rel_tol=1.0e-7,
        abs_tol=1.0e-9 * max(1.0, abs(source_volume)),
    ):
        if not silent:
            print(
                "STEP.Load - Warning: CellComplex reconstruction did not preserve "
                "total Cell volume. Returning the imported Cluster."
            )
        return topology

    try:
        internal_faces = CellComplex.InternalFaces(rebuilt) or []
    except Exception:
        internal_faces = []
    if len(rebuilt_cells) > 1 and len(internal_faces) < 1:
        if not silent:
            print(
                "STEP.Load - Warning: Reconstructed Cells do not share an internal "
                "Face. Returning the imported Cluster."
            )
        return topology

    return rebuilt

def _wrap_shape(shape):
    """Wrap an imported OCCT shape while preserving native container kind."""
    from topologicpy.Core import Core
    from topologicpy.Topology import Topology

    try:
        from OCC.Core.TopAbs import TopAbs_COMPOUND, TopAbs_COMPSOLID
        shape_type = shape.ShapeType()
    except Exception:
        shape_type = None

    if shape_type is not None:
        try:
            if shape_type == TopAbs_COMPOUND:
                result = Core.Call("Cluster", "ByOcctShape", shape)
                if Topology.IsInstance(result, "Cluster"):
                    return result
            if shape_type == TopAbs_COMPSOLID:
                result = Core.Call("CellComplex", "ByOcctShape", shape)
                if Topology.IsInstance(result, "CellComplex"):
                    return result
        except Exception:
            pass

    return Topology.ByOCCTShape(shape, silent=True)


class STEPCodec:
    """Codec for neutral STEP BREP exchange using the PythonOCC/OCCT translator."""

    extensions = (".step", ".stp")
    preserves_brep = True
    preserves_curves = True
    preserves_surfaces = True
    preserves_topologic_metadata = False
    preserves_semantic_identity = False
    tessellated = False
    geometry_fidelity = "brep_exchange"
    preserves_topologicpy_root_type_hint = True

    @staticmethod
    def save(
        topology,
        path,
        overwrite: bool = False,
        silent: bool = False,
        schema: str = "AP242DIS",
        unit: str = "MM",
        assembly="auto",
        tolerance=None,
    ) -> bool:
        """Export ``topology`` to STEP without tessellating its BREP geometry."""
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print("STEP.Save - Error: The input topology is not valid. Returning False.")
            return False

        try:
            path = _output_path(path)
            schema = _normalise_schema(schema)
            unit = _normalise_unit(unit)
            assembly_mode = _normalise_assembly(assembly)
            tolerance = _normalise_tolerance(tolerance)
        except Exception as error:
            if not silent:
                print(f"STEP.Save - Error: {error} Returning False.")
            return False

        if os.path.exists(path) and not overwrite:
            if not silent:
                print("STEP.Save - Error: The output file already exists. Returning False.")
            return False

        root_type = _topology_root_type(topology)
        shape = Topology.OCCTShape(topology, silent=True)
        if shape is None:
            if not silent:
                print("STEP.Save - Error: Could not retrieve the OCCT BREP. Returning False.")
            return False
        try:
            if shape.IsNull():
                raise STEPError("The OCCT BREP is null.")
        except AttributeError:
            if not silent:
                print("STEP.Save - Error: The topology does not expose a PythonOCC shape. Returning False.")
            return False

        temporary_path = None
        try:
            from OCC.Core.IFSelect import IFSelect_RetDone
            from OCC.Core.Interface import Interface_Static
            from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer

            directory = os.path.dirname(os.path.abspath(path))
            os.makedirs(directory, exist_ok=True)
            fd, temporary_path = tempfile.mkstemp(
                prefix=".topologicpy_",
                suffix=".step.tmp",
                dir=directory,
            )
            os.close(fd)
            os.remove(temporary_path)

            with _INTERFACE_LOCK:
                # Constructing the STEP controller initializes OCCT's STEP
                # Interface_Static resources before we inspect or change them.
                writer = STEPControl_Writer()
                state = _snapshot_interface(Interface_Static)
                try:
                    # ``unit`` declares the physical unit represented by one
                    # TopologicPy/OCCT coordinate unit. Keep CASCADE's internal
                    # and STEP output units aligned so numerical scale is not
                    # silently reinterpreted.
                    Interface_Static.SetCVal("xstep.cascade.unit", unit)
                    Interface_Static.SetCVal("write.step.unit", unit)
                    Interface_Static.SetCVal("write.step.schema", schema)
                    Interface_Static.SetIVal("write.step.assembly", assembly_mode)
                    Interface_Static.SetIVal("write.step.tessellated", 0)
                    Interface_Static.SetIVal("write.surfacecurve.mode", 1)
                    # OCCT documents that a new model is required after changing
                    # write.step.schema.
                    writer.Model(True)
                    if tolerance is not None:
                        writer.SetTolerance(tolerance)
                    transfer_status = writer.Transfer(shape, STEPControl_AsIs)
                    if transfer_status != IFSelect_RetDone:
                        raise STEPError("OCCT could not translate the BREP to STEP.")
                    write_status = writer.Write(temporary_path)
                    if write_status != IFSelect_RetDone:
                        raise STEPError("OCCT could not write the STEP file.")
                finally:
                    _restore_interface(Interface_Static, state)

            if not os.path.isfile(temporary_path) or os.path.getsize(temporary_path) <= 0:
                raise STEPError("OCCT reported success but no STEP file was produced.")
            _inject_root_type_hint(temporary_path, root_type)
            os.replace(temporary_path, path)
            temporary_path = None
            return True
        except Exception as error:
            if not silent:
                print(f"STEP.Save - Error: {error} Returning False.")
            return False
        finally:
            if temporary_path and os.path.exists(temporary_path):
                try:
                    os.remove(temporary_path)
                except Exception:
                    pass

    @staticmethod
    def load(path, silent: bool = False, unit: str = "MM"):
        """Import STEP BREP geometry into TopologicPy.

        TopologicPy dictionaries, Content, Aperture and Context semantics are not
        reconstructed from basic STEP.  Multiple STEP roots are returned as a
        Cluster.
        """
        try:
            path = _input_path(path)
            unit = _normalise_unit(unit)
            root_type_hint = _read_root_type_hint(path)
        except Exception as error:
            if not silent:
                print(f"STEP.Load - Error: {error} Returning None.")
            return None

        try:
            from OCC.Core.IFSelect import IFSelect_RetDone
            from OCC.Core.Interface import Interface_Static
            from OCC.Core.STEPControl import STEPControl_Reader

            with _INTERFACE_LOCK:
                # Reader construction initializes the STEP translator resources.
                reader = STEPControl_Reader()
                previous_unit = Interface_Static.CVal("xstep.cascade.unit")
                try:
                    Interface_Static.SetCVal("xstep.cascade.unit", unit)
                    read_status = reader.ReadFile(path)
                    if read_status != IFSelect_RetDone:
                        raise STEPError("OCCT could not read the STEP file.")
                    if not reader.TransferRoots():
                        raise STEPError("OCCT could not transfer STEP roots to BREP.")
                    count = int(reader.NbShapes())
                    if count < 1:
                        raise STEPError("The STEP file contains no transferable shapes.")

                    # OCCT's OneShape() is the canonical aggregate result of the
                    # transfer: it returns the sole result directly or a Compound
                    # when several results were produced.  STEP product/representation
                    # translation can additionally wrap one logical result in one or
                    # more single-child Compounds, so remove only those transport
                    # wrappers.
                    shape = reader.OneShape()
                    if shape is None or shape.IsNull():
                        raise STEPError("OCCT produced a null STEP transfer result.")
                    shape = _normalise_transport_shape(shape)
                    shape = _restore_root_type_hint(shape, root_type_hint)
                    shapes = [shape]
                finally:
                    Interface_Static.SetCVal("xstep.cascade.unit", previous_unit)

            topologies = []
            for shape in shapes:
                topology = _wrap_shape(shape)
                if topology is None:
                    raise STEPError("An imported STEP root could not be wrapped as TopologicPy topology.")
                topologies.append(topology)

            if len(topologies) == 1:
                return _restore_cellcomplex_root_type(
                    topologies[0],
                    root_type_hint,
                    silent=silent,
                )

            from topologicpy.Cluster import Cluster
            result = Cluster.ByTopologies(topologies, silent=True)
            if result is None:
                raise STEPError("Multiple STEP roots could not be assembled into a Cluster.")
            return result
        except Exception as error:
            if not silent:
                print(f"STEP.Load - Error: {error} Returning None.")
            return None


__all__ = ["STEPCodec", "STEPError"]
