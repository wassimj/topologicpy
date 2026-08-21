# TopologicPy shared public API test suite

This is one backend-neutral public API test suite for TopologicPy.

## Backend policy

- The tests do **not** require `topologic_core` specifically.
- The tests do **not** require PythonOCC specifically.
- Plain `pytest` automatically selects an installed backend.
- Automatic selection prefers `pythonocc-core`, then falls back to `topologic_core`.
- If only one backend is installed, only that backend is used.
- If both are installed, plain `pytest` uses PythonOCC; `run_backends.py` runs the same suite once against each installed backend in separate processes.
- If a backend is requested explicitly but is not installed, pytest reports a clear configuration error rather than silently testing a different engine.
- Backend-specific known defects are isolated in `backend_exceptions.json` rather than embedded in ordinary public API tests.

## Normal CI invocation

If the CI environment contains one backend, this is sufficient:

```bash
python -m pytest tests -n auto
```

The installed engine is detected automatically.

To run the shared suite against **every backend that is actually installed**:

```bash
python tests/run_backends.py -n auto
```

A missing optional backend is reported as unavailable and skipped.

## Explicit backend runs

Use these only when CI intentionally provisions that backend:

```bash
python -m pytest tests --backend pythonocc -n auto
python -m pytest tests --backend topologic_core -n auto
```

or:

```bash
python tests/run_backends.py --only pythonocc -n auto
python tests/run_backends.py --only topologic_core -n auto
```

## Design rules

- One canonical set of public API tests.
- No PythonOCC-specific test tree.
- No TopologicCore-specific test tree.
- No backend parity tests in the normal public API suite.
- No direct tests of underscore-prefixed TopologicPy implementation helpers.
- Public API tests should assert TopologicPy behaviour and topology semantics, not concrete backend object types.
- Tests that modify `Core.SetBackend` are isolated; the selected session backend is restored after every test.
- Frozen backend-specific exceptions live only in `backend_exceptions.json`.
