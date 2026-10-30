# IDS information delivery assessment

Install the optional XML dependencies with `pip install "topologicpy[ids]"`.
For this checkout use `pip install -e ".[ids]"`.
Reading and IDS assessment do not require IfcOpenShell, IfcTester or a geometry
backend. The existing `ifc` extra remains available for optional IFC export.

```python
from topologicpy.IFC import IFC
from topologicpy.IDS import IDS

model = IFC.ModelByPath("delivery.ifc", silent=True)
requirements = IDS.ByPath("requirements.ids")
result = IDS.Validate(model, requirements)
# Equivalent convenience API:
result = IDS.ValidateFiles("delivery.ifc", "requirements.ids")

print(result["status"])  # pass or fail
for finding in result["findings"]:
    print(finding["global_id"], finding["expected"], finding["actual"])
```

`ByPath` returns a schema-validated IDS 1.0 XML root. `Validate` accepts that
document and a TopologicPy IFCModel, returning a dictionary with `status`,
`specifications`, `findings`, `warnings` and `summary`. Findings retain IDS
identifiers, IFC GlobalIds and STEP IDs, expected and observed information,
and IDS correction instructions. `ValidateFiles` combines both input steps.
Malformed IDS and unsupported IFC schemas raise `ValueError`; errors must not
be interpreted as compliance passes. XML entity declarations are rejected and
IDS schema validation uses bundled resources without network access.

All six IDS facets are supported: entity, attribute, property, classification,
material and partOf. Evaluation includes applicability, cardinality, restrictions,
type inheritance and occurrence overrides, relationship traversal and common
project/explicit unit conversions. Schema metadata covers IFC2X3 TC1, IFC4 ADD2
TC1 and IFC4X3 ADD2. An IFC4X3 alias denotes ADD2; release candidates and other
unrecognized variants are rejected. Differences in IDS version metadata are
reported as warnings while checks use the actual model schema.

The assessment covers the supplied IDS. It is not a full IFC schema audit,
design/regulatory check, or buildingSMART certification. Reference-valued
properties cannot satisfy a required property check. The result is independent
of TopologicCheck's web interface, temporary uploads, PDF/Excel exports and
hosting, which remain application responsibilities.

## Verification

The unchanged buildingSMART test corpus is pinned to commit
`870f9c4e6e8f414e737b4d84ca1aa9b46fc6c8f3`. Its 334 IFC/IDS pairs are stored with
their original SHA-256 manifest in `tests/ids/buildingSMART-corpus.zip`; copyright
and license are in `tests/ids/CORPUS-LICENSE.md`. The corpus covers IFC2X3 and IFC4;
IFC4X3 has a separate smoke test. Expected invalid cases must fail assessment or
be explicitly rejected, and processing errors never count as matches.

Run the pure-Python checks without selecting a geometry backend:

```shell
python -m pytest -o addopts= --noconftest tests/ids
```

The existing `tests/test_IFC.py` tests also check geometry integration and need
the usual TopologicPy dependencies and a topology backend.

Packaged schema sources, hashes and attribution are recorded in
`src/topologicpy/ids_schemas/PROVENANCE.json`. The reader preserves logical
UNKNOWN as `IFCLogicalUnknown`, distinct from null and the literal text UNKNOWN,
and decodes STEP Unicode strings. IFC's optional exporter is imported only
when `ExportReferenceView` is called.
