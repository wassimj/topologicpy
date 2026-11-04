# TPY format version 2

TPY is a ZIP archive with a UTF-8 `manifest.json` and exact native BREP payloads
under `geometry/`. Writers now use `formatVersion: 2`; readers accept versions
1 and 2. Older readers must be upgraded before opening v2 files. This source
change does not itself publish a new PyPI package.

## Public API

`Topology.Save(value, path)` and `Topology.Load(path)` accept native topologies,
TGraphs, and mixed list/tuple/dictionary roots. `Topology.ExportToTPY` uses the
same codec. `TGraph.Save`, `Load`, `ExportToTPY` and `ByTPYPath` are conveniences.
STEP export still requires a native topology.

```python
Topology.Save({"building": building, "graph": graph}, "project.tpy")
project = Topology.Load("project.tpy")
graph = project["graph"]
```

## Graph semantics

A TGraph object record stores graph direction defaults, loop/parallel-edge
policies, graph dictionary, revision, and complete vertex/edge slot arrays.
Inactive slots remain in place. Every edge retains source, target, its own
direction and active flag. Per-element dictionaries, representations,
originators and parent originators are included. Adjacency indexes are rebuilt
on load; compilation and analysis caches are deliberately not serialized.

Graph and native topology references use object IDs, optional native subshape
locators and orientation. Native objects are stored in a common containing
BREP when needed, preserving shared subshapes across graph originators. A
reference cache preserves aliases. Graph references may be cyclic.

## Manifest records and values

Native records retain geometry path/checksum, dictionary-bearing subtopologies,
Contents, Apertures and Context parameters. `TGraph` records have an encoded
`state`; `Collection` records have an encoded `value`. References use
`__tpy_type__: reference` with `object`, optional `locator` and `orientation`.
Object IDs are archive-local and have no meaning outside that archive.

All dictionary values use a tagged mapping containing encoded key/value pairs.
This preserves non-string keys and distinguishes literal dictionaries from
codec tags. Supported values include primitive values, nonfinite floats,
bytes, tuples, sets, frozensets, complex numbers, dates/times, Decimal, Fraction,
UUID, and NumPy arrays/scalars with explicit dtype and shape. Structured NumPy
dtypes, arbitrary Python objects, and cycles in ordinary Python containers
are rejected explicitly. No pickle, executable code or repr reconstruction
is used. User-defined Python classes are not claimed to be losslessly portable.

## Validation and compatibility

Readers validate format/version, object IDs, references, native checksums and
graph slot/end-point policies. The Syntopy adapter additionally re-exports and
compares metadata and reference semantics before accepting a transport result.
BREP bytes remain the authority for curves and surfaces; viewer tessellation
is only a display approximation. TPY v1 remains readable. Re-saving writes v2.
