#!/usr/bin/env python3
"""Benchmark Tranche-2 CellComplex face-incidence classification.

The workload deliberately targets the old quadratic NonManifoldFaces scan.
It constructs one 4 x 4 x 2 prism (32 Cells, 64 internal Faces), warms both
paths, checks semantic equivalence, and reports median query time.
"""
from __future__ import annotations

import argparse
import os
import statistics
import time

os.environ.setdefault("TOPOLOGICPY_CORE_BACKEND", "pythonocc")

from topologicpy.CellComplex import CellComplex
from topologicpy.Topology import Topology


def _shape_set(items):
    return [getattr(item, "shape", None) for item in items or []]


def _same_shape_sets(a, b):
    left = _shape_set(a)
    right = _shape_set(b)
    if len(left) != len(right):
        return False
    remaining = list(right)
    for shape in left:
        for i, candidate in enumerate(remaining):
            try:
                same = bool(shape.IsSame(candidate))
            except Exception:
                same = False
            if same:
                remaining.pop(i)
                break
        else:
            return False
    return not remaining


def _query(model, repeats: int):
    checksum = 0
    for _ in range(repeats):
        faces = CellComplex.NonManifoldFaces(model)
        checksum += len(faces or [])
    return checksum


def _timed(model, repeats: int):
    start = time.perf_counter()
    checksum = _query(model, repeats)
    return time.perf_counter() - start, checksum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    model = CellComplex.Prism(
        width=4.0,
        length=4.0,
        height=2.0,
        uSides=4,
        vSides=4,
        wSides=2,
        tolerance=0.0001,
    )
    if not Topology.IsInstance(model, "CellComplex"):
        raise RuntimeError("Could not construct benchmark CellComplex.")

    old = os.environ.pop("TOPOLOGICPY_DISABLE_BREPGRAPH", None)
    try:
        graph_faces = CellComplex.NonManifoldFaces(model)
        os.environ["TOPOLOGICPY_DISABLE_BREPGRAPH"] = "1"
        legacy_faces = CellComplex.NonManifoldFaces(model)
        if not _same_shape_sets(graph_faces, legacy_faces):
            raise RuntimeError(
                f"Graph/legacy mismatch: {len(graph_faces or [])} vs {len(legacy_faces or [])} internal Faces."
            )

        # Warm both paths before sampling.
        os.environ.pop("TOPOLOGICPY_DISABLE_BREPGRAPH", None)
        _query(model, 2)
        os.environ["TOPOLOGICPY_DISABLE_BREPGRAPH"] = "1"
        _query(model, 2)

        graph_times = []
        legacy_times = []
        graph_checksum = legacy_checksum = None

        for _ in range(args.samples):
            os.environ.pop("TOPOLOGICPY_DISABLE_BREPGRAPH", None)
            elapsed, checksum = _timed(model, args.repeats)
            graph_times.append(elapsed)
            graph_checksum = checksum

            os.environ["TOPOLOGICPY_DISABLE_BREPGRAPH"] = "1"
            elapsed, checksum = _timed(model, args.repeats)
            legacy_times.append(elapsed)
            legacy_checksum = checksum

        if graph_checksum != legacy_checksum:
            raise RuntimeError(
                f"Checksum mismatch: BRepGraph={graph_checksum}, legacy={legacy_checksum}."
            )

        graph_median = statistics.median(graph_times)
        legacy_median = statistics.median(legacy_times)
        speedup = legacy_median / graph_median if graph_median > 0 else float("inf")

        print(f"BRepGraph median: {graph_median:.6f} s")
        print(f"Legacy median:    {legacy_median:.6f} s")
        print(f"Speed-up:         {speedup:.2f}x")
        print(f"Internal Faces:   {len(graph_faces)}")
        print(f"Samples:          {args.samples}; repeats/sample: {args.repeats}")
    finally:
        if old is None:
            os.environ.pop("TOPOLOGICPY_DISABLE_BREPGRAPH", None)
        else:
            os.environ["TOPOLOGICPY_DISABLE_BREPGRAPH"] = old


if __name__ == "__main__":
    main()
