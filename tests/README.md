# TopologicPy shared public API test suite

- One canonical set of public API tests.
- No PythonOCC-specific test tree.
- No backend parity tests.
- No direct tests of underscore-prefixed TopologicPy implementation helpers.
- Each backend runs in its own pytest process.
- The selected backend is restored after every test so tests of `Core.SetBackend` cannot leak state.
- Frozen TopologicCore exceptions live only in `backend_exceptions.json`.

Run all available backends:

```bash
python tests/run_backends.py -n auto
```

Run one backend:

```bash
python -m pytest tests --backend pythonocc -n auto
python -m pytest tests --backend topologic_core -n auto
```
