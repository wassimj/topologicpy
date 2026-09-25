#!/usr/bin/env python3
"""A/B benchmark for TopologicPy BRepGraph Tranche 1.

Run from the TopologicPy repository root after applying the tranche:

    python benchmark_BRepGraphQueries.py --repeats 100

The script compares the normal OCCT 8 BRepGraph path with the preserved legacy
TopExp/ancestor-map path by toggling TOPOLOGICPY_DISABLE_BREPGRAPH at runtime.
"""
from __future__ import annotations

import argparse
import os
import statistics
import time

os.environ.setdefault("TOPOLOGICPY_CORE_BACKEND", "pythonocc")

from topologicpy.CellComplex import CellComplex
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex


def build_model():
    host = CellComplex.Octahedron(
        origin=Vertex.Origin(),
        radius=1.0,
        direction=[0.0, 0.0, 1.0],
        placement="center",
        tolerance=0.0001,
    )
    if not Topology.IsInstance(host, "CellComplex"):
        raise RuntimeError("Could not build the benchmark CellComplex.")
    return host


def workload(host, repeats: int):
    vertices = Topology.Vertices(host, silent=True) or []
    edges = Topology.Edges(host, silent=True) or []
    faces = Topology.Faces(host, silent=True) or []
    cells = Topology.Cells(host, silent=True) or []

    checksum = 0
    for _ in range(repeats):
        # Descendant traversal.
        checksum += len(Topology.Faces(host, silent=True) or [])
        checksum += len(Topology.Edges(host, silent=True) or [])

        # Reverse incidence.
        for vertex in vertices:
            checksum += len(
                Topology.SuperTopologies(
                    vertex, hostTopology=host, topologyType="edge", silent=True
                )
                or []
            )
        for edge in edges:
            checksum += len(
                Topology.SuperTopologies(
                    edge, hostTopology=host, topologyType="face", silent=True
                )
                or []
            )
        for face in faces:
            checksum += len(
                Topology.SuperTopologies(
                    face, hostTopology=host, topologyType="cell", silent=True
                )
                or []
            )

        # Same-dimensional adjacency.
        for face in faces:
            checksum += len(
                Topology.AdjacentTopologies(
                    face, hostTopology=host, topologyType="face", silent=True
                )
                or []
            )

        # Shared topology.
        if len(cells) >= 2:
            shared = Topology.SharedTopologies(cells[0], cells[1], silent=True) or {}
            checksum += sum(len(shared.get(key, []) or []) for key in ("vertices", "edges", "wires", "faces"))
    return checksum


def timed(host, repeats: int, disable: bool, samples: int):
    if disable:
        os.environ["TOPOLOGICPY_DISABLE_BREPGRAPH"] = "1"
    else:
        os.environ.pop("TOPOLOGICPY_DISABLE_BREPGRAPH", None)

    # Warm-up. For BRepGraph this also constructs and caches the host index.
    expected = workload(host, 1)
    times = []
    for _ in range(samples):
        start = time.perf_counter()
        checksum = workload(host, repeats)
        elapsed = time.perf_counter() - start
        if checksum != expected * repeats:
            raise RuntimeError("Benchmark paths returned different query counts.")
        times.append(elapsed)
    return times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    host = build_model()
    graph_times = timed(host, args.repeats, disable=False, samples=args.samples)
    legacy_times = timed(host, args.repeats, disable=True, samples=args.samples)

    graph_med = statistics.median(graph_times)
    legacy_med = statistics.median(legacy_times)
    speedup = legacy_med / graph_med if graph_med > 0.0 else float("inf")

    print(f"BRepGraph median: {graph_med:.6f} s")
    print(f"Legacy median:    {legacy_med:.6f} s")
    print(f"Speed-up:         {speedup:.2f}x")
    print(f"Samples:          {args.samples}; repeats/sample: {args.repeats}")


if __name__ == "__main__":
    main()
