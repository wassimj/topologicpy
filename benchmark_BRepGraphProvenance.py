#!/usr/bin/env python3
"""Benchmark Tranche 3 exact provenance against selector-style geometric remapping.

The benchmark intentionally compares the metadata-mapping stage on the same
rigid transform.  The legacy path emulates the common pre-Tranche-3 pattern:
compute a selector/centroid for every source Face and search all result Faces
for the nearest match.  The Tranche-3 path uses OCCT ModifiedShape lineage and
BRepGraph result membership.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

os.environ.setdefault("TOPOLOGICPY_CORE_BACKEND", "pythonocc")

from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.gp import gp_Trsf, gp_Vec

from topologicpy.CellComplex import CellComplex
from topologicpy.Dictionary import Dictionary
from topologicpy.Topology import Topology
from topologicpy.pythonocc_backend.attribute_manager import AttributeManager
from topologicpy.pythonocc_backend._provenance import transfer_by_modifier


def _unique_faces(shape):
    result = []
    buckets = {}
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = explorer.Current()
        try:
            key = hash(face)
        except Exception:
            key = None
        duplicate = False
        if key is not None:
            bucket = buckets.setdefault(key, [])
            for existing in bucket:
                if face.IsSame(existing):
                    duplicate = True
                    break
            if not duplicate:
                bucket.append(face)
        else:
            for existing in result:
                if face.IsSame(existing):
                    duplicate = True
                    break
        if not duplicate:
            result.append(face)
        explorer.Next()
    return result


def _centroid(face):
    props = GProp_GProps()
    brepgprop.SurfaceProperties(face, props)
    p = props.CentreOfMass()
    return float(p.X()), float(p.Y()), float(p.Z())


def _set_dict(topology, values):
    return Topology.SetDictionary(
        topology,
        Dictionary.ByPythonDictionary(values),
        silent=True,
    )


def _dict_to_python(dictionary):
    if dictionary is None:
        return {}
    if isinstance(dictionary, dict):
        return dict(dictionary)
    try:
        value = Dictionary.PythonDictionary(dictionary)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _prepare_model(u_sides=6, v_sides=6, w_sides=2):
    cc = CellComplex.Prism(
        width=6,
        length=6,
        height=3,
        uSides=u_sides,
        vSides=v_sides,
        wSides=w_sides,
        tolerance=0.0001,
    )
    if not Topology.IsInstance(cc, "CellComplex"):
        raise RuntimeError("Could not construct benchmark CellComplex.")

    _set_dict(cc, {"model": "provenance-benchmark"})
    faces = Topology.Faces(cc, silent=True) or []
    for index, face in enumerate(faces):
        _set_dict(face, {"face_id": index, "zone": index % 7})
    return cc, faces


def _exact_once(source_shape, root_dictionary, trsf):
    maker = BRepBuilderAPI_Transform(source_shape, trsf, True)
    if not maker.IsDone():
        raise RuntimeError("BRepBuilderAPI_Transform failed.")
    target_shape = maker.Shape()
    report = transfer_by_modifier(
        source_shape,
        target_shape,
        maker,
        root_dictionary=root_dictionary,
        operation="Benchmark.Transform",
    )
    return target_shape, report


def _native_only_once(source_shape, trsf):
    """Transform only: lower bound used to quantify provenance overhead."""
    maker = BRepBuilderAPI_Transform(source_shape, trsf, True)
    if not maker.IsDone():
        raise RuntimeError("BRepBuilderAPI_Transform failed.")
    return maker.Shape()


def _legacy_selector_once(source_shape, trsf):
    """Centroid-nearest emulation of selector-style dictionary remapping."""
    maker = BRepBuilderAPI_Transform(source_shape, trsf, True)
    if not maker.IsDone():
        raise RuntimeError("BRepBuilderAPI_Transform failed.")
    target_shape = maker.Shape()

    manager = AttributeManager.GetInstance()
    source_faces = _unique_faces(source_shape)
    target_faces = _unique_faces(target_shape)
    target_centers = [_centroid(face) for face in target_faces]

    dx, dy, dz = 11.25, -7.5, 4.0
    transferred = 0
    for source_face in source_faces:
        if not manager.HasDictionary(source_face):
            continue
        sx, sy, sz = _centroid(source_face)
        expected = (sx + dx, sy + dy, sz + dz)
        best_index = None
        best_d2 = None
        for index, center in enumerate(target_centers):
            d2 = (
                (expected[0] - center[0]) ** 2
                + (expected[1] - center[1]) ** 2
                + (expected[2] - center[2]) ** 2
            )
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best_index = index
        if best_index is not None:
            manager.SetDictionary(
                target_faces[best_index],
                manager.GetDictionary(source_face),
            )
            transferred += 1
    return target_shape, transferred


def _face_dictionary_ids(shape):
    manager = AttributeManager.GetInstance()
    values = []
    for face in _unique_faces(shape):
        if not manager.HasDictionary(face):
            continue
        dictionary = _dict_to_python(manager.GetDictionary(face))
        if "face_id" in dictionary:
            values.append(int(dictionary["face_id"]))
    return sorted(values)


def _time_samples(fn, samples, repeats):
    values = []
    for _ in range(samples):
        start = time.perf_counter()
        for _ in range(repeats):
            fn()
        values.append(time.perf_counter() - start)
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--u", type=int, default=6)
    parser.add_argument("--v", type=int, default=6)
    parser.add_argument("--w", type=int, default=2)
    args = parser.parse_args()

    model, source_faces = _prepare_model(args.u, args.v, args.w)
    source_shape = getattr(model, "shape", None)
    if source_shape is None:
        raise RuntimeError("Benchmark model has no native shape.")

    manager = AttributeManager.GetInstance()
    root_dictionary = manager.GetDictionary(source_shape) if manager.HasDictionary(source_shape) else {}

    trsf = gp_Trsf()
    trsf.SetTranslation(gp_Vec(11.25, -7.5, 4.0))

    # Logic verification before timing.
    exact_shape, report = _exact_once(source_shape, root_dictionary, trsf)
    legacy_shape, legacy_count = _legacy_selector_once(source_shape, trsf)
    expected_ids = []
    for face in _unique_faces(source_shape):
        if not manager.HasDictionary(face):
            continue
        dictionary = _dict_to_python(manager.GetDictionary(face))
        if "face_id" in dictionary:
            expected_ids.append(int(dictionary["face_id"]))
    expected_ids.sort()
    exact_ids = _face_dictionary_ids(exact_shape)
    legacy_ids = _face_dictionary_ids(legacy_shape)

    if exact_ids != expected_ids:
        raise RuntimeError(
            f"Exact provenance logic mismatch: expected {len(expected_ids)} face dictionaries, "
            f"got {len(exact_ids)}."
        )
    if legacy_ids != expected_ids:
        raise RuntimeError(
            f"Legacy selector emulation mismatch: expected {len(expected_ids)} face dictionaries, "
            f"got {len(legacy_ids)}."
        )

    native_samples = _time_samples(
        lambda: _native_only_once(source_shape, trsf),
        args.samples,
        args.repeats,
    )
    exact_samples = _time_samples(
        lambda: _exact_once(source_shape, root_dictionary, trsf),
        args.samples,
        args.repeats,
    )
    legacy_samples = _time_samples(
        lambda: _legacy_selector_once(source_shape, trsf),
        args.samples,
        args.repeats,
    )

    native_sample_median = statistics.median(native_samples)
    exact_sample_median = statistics.median(exact_samples)
    legacy_sample_median = statistics.median(legacy_samples)

    repeats = max(1, int(args.repeats))
    native_median = native_sample_median / repeats
    exact_median = exact_sample_median / repeats
    legacy_median = legacy_sample_median / repeats

    exact_increment = max(0.0, exact_median - native_median)
    legacy_increment = max(0.0, legacy_median - native_median)
    total_speedup = legacy_median / exact_median if exact_median > 0 else float("inf")
    mapping_speedup = (
        legacy_increment / exact_increment
        if exact_increment > 0
        else float("inf")
    )
    overhead = exact_median / native_median if native_median > 0 else float("inf")

    print(f"Source Faces:          {len(source_faces)}")
    print(f"Mapped Face dicts:     {len(expected_ids)}")
    print(f"BRepGraph indexed:     {report.used_brepgraph_index}")
    print(f"Native-only / call:    {native_median:.6f} s")
    print(f"Exact-history / call:  {exact_median:.6f} s")
    print(f"Selector-style / call: {legacy_median:.6f} s")
    print(f"Exact mapping cost:    {exact_increment:.6f} s")
    print(f"Selector mapping cost: {legacy_increment:.6f} s")
    print(f"Provenance overhead:   {overhead:.2f}x native-only")
    print(f"Total speed-up:        {total_speedup:.2f}x vs selector-style")
    print(f"Mapping speed-up:      {mapping_speedup:.2f}x exact vs selector")
    print(f"Samples:               {args.samples}; repeats/sample: {args.repeats}")
    print(f"Legacy mapped:         {legacy_count}")


if __name__ == "__main__":
    main()
