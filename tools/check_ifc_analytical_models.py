"""Offline real-model checks for the self-contained IFC analytical API.

Usage: python tools/check_ifc_analytical_models.py --fixtures PATH --output PATH
The CSV manifest records source URLs and hashes; input files are not downloaded
automatically. Each conversion runs in its own process with a timeout.
"""
from pathlib import Path
import argparse
import builtins
import csv
import hashlib
import json
import subprocess
import sys
import time


def worker(source, directory):
    original = builtins.__import__
    def guard(name, *args, **kwargs):
        if name == "bim2bem" or name.startswith("bim2bem.") or name == "ifcopenshell" or name.startswith("ifcopenshell."):
            raise AssertionError("Release reconstruction imported an external engine")
        return original(name, *args, **kwargs)
    builtins.__import__ = guard
    from topologicpy.IFC import IFC
    from topologicpy.Core import Core
    directory.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    def progress(stage, **details):
        (directory/"progress.json").write_text(json.dumps({"stage": stage, "seconds": round(time.perf_counter()-start, 3), **details}))
    result = IFC.AnalyticalModel(str(source), progress=progress)
    if result.valid:
        IFC.AnalyticalModelSave(result, directory)
    report = result.report()
    report["statistics"].update(elapsed_seconds=round(time.perf_counter()-start, 3), backend=type(Core.Backend()).__name__)
    (directory/"report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    raise SystemExit(0 if result.valid else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1]/"tests/data/ifc_analytical_real_models.csv")
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.output)
    if not args.fixtures:
        parser.error("--fixtures is required")
    outcomes = []
    args.output.mkdir(parents=True, exist_ok=True)
    for entry in csv.DictReader(args.manifest.open(encoding="utf-8")):
        path = args.fixtures/(entry["id"]+".ifc")
        folder = args.output/entry["id"]
        folder.mkdir(parents=True, exist_ok=True)
        outcome = {"id": entry["id"], "passed": False, "source_agreement_only": True}
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            outcome["status"] = "missing_or_changed_fixture"
        else:
            start = time.perf_counter()
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(path.resolve()), "--output", str(folder.resolve())]
            try:
                with (folder/"process.log").open("w", encoding="utf-8") as log:
                    process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout, check=False)
                report = json.loads((folder/"report.json").read_text()) if (folder/"report.json").is_file() else {}
                f, s = report.get("fidelity", {}), report.get("statistics", {})
                valid = process.returncode == 0 and report.get("valid") and s.get("native_roundtrip_verified")
                valid = valid and f.get("source_space_coverage", 0) >= .8 and f.get("source_area_coverage", 0) >= .8 and f.get("native_cell_coverage", 0) >= .95
                valid = valid and all(b["statistics"].get("topology_type") == "CellComplex" for b in report.get("buildings", []))
                if entry["id"] == "bimdata-kit-institute":
                    valid = valid and f.get("exterior_fit_applied") and f.get("roof_source_fit_rmse_m", 1) < .03
                outcome.update(passed=bool(valid), status="passed" if valid else "failed", seconds=round(time.perf_counter()-start, 3),
                    cells=s.get("rooms"), footprint_iou=f.get("weighted_footprint_iou"), roof_rmse_m=f.get("roof_source_fit_rmse_m"),
                    report=str(folder/"report.json"), backend=s.get("backend"))
            except subprocess.TimeoutExpired:
                outcome["status"] = "timeout"
        (folder/"outcome.json").write_text(json.dumps(outcome, indent=2)+"\n")
        outcomes.append(outcome)
        print(json.dumps(outcome), flush=True)
    (args.output/"summary.json").write_text(json.dumps(outcomes, indent=2)+"\n")
    raise SystemExit(0 if all(o["passed"] for o in outcomes) else 1)


if __name__ == "__main__":
    main()
