#!/usr/bin/env python3
"""Copy the canonical root ontology to runtime and GitHub Pages artefacts."""
from pathlib import Path
import shutil
root = Path.cwd()
src = root / "ontology" / "topologicpy.ttl"
if not src.exists(): raise SystemExit(f"Missing canonical ontology: {src}")
for dst in (root / "src/topologicpy/ontology/topologicpy.ttl", root / "docs/ontology/topologicpy.ttl"):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print("Synced", dst)
