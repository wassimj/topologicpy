# Beta: PythonOCC Backend in TopologicPy v0.9.60

TopologicPy v0.9.61 includes a new **PythonOCC/OpenCASCADE backend**, currently considered **beta**.

For the most predictable setup, Python 3.11 and `pythonocc-core 7.9.3` are recommended. `pythonocc-core` is distributed through conda-forge.

## 1. Create a Dedicated Conda Environment

```bash
conda create -n topologicpy-occ -c conda-forge python=3.11 pythonocc-core=7.9.3 pip -y
conda activate topologicpy-occ
```

## 2. Install TopologicPy

```bash
pip install topologicpy==0.9.60
```

TopologicPy itself is installed normally through `pip`.

## 3. Select the PythonOCC Backend

On **Windows PowerShell**:

```powershell
$env:TOPOLOGICPY_CORE_BACKEND="pythonocc"
```

On **macOS/Linux**:

```bash
export TOPOLOGICPY_CORE_BACKEND=pythonocc
```

Set this **before starting Python or Jupyter**.

## 4. Verify the Active Backend

```python
from topologicpy.Core import Core

print(Core.Backend().__class__.__name__)
```

You should see:

```text
PythonOCCBackend
```

You can then use TopologicPy normally:

```python
from topologicpy.Cell import Cell

cell = Cell.Prism()
```

To return explicitly to the legacy backend, set `TOPOLOGICPY_CORE_BACKEND` to `topologic_core`.

> **Beta note:** The PythonOCC backend now passes the TopologicPy test suite alongside `topologic_core`, but it should still be regarded as beta while it receives wider real-world testing. Bug reports and reproducible examples are very welcome.
