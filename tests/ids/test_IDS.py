"""Pure-Python IDS/IFC regressions; no geometry backend or IFC SDK needed.

Run with: pytest -o addopts= --noconftest tests/ids
The pinned upstream fixture archive is unmodified; see CORPUS-LICENSE.md.
"""
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import zipfile

import pytest

from topologicpy.IDS import IDS
from topologicpy.IFC import IFC, IFCLogicalUnknown

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / 'buildingSMART-corpus.zip'
with zipfile.ZipFile(ARCHIVE) as archive:
    CASES = sorted(name for name in archive.namelist() if name.endswith('.ids'))


@pytest.fixture(scope='session')
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp('official-ids')
    with zipfile.ZipFile(ARCHIVE) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        assert manifest['upstream_commit'] == '870f9c4e6e8f414e737b4d84ca1aa9b46fc6c8f3'
        assert len(manifest['files']) == 668 and len(CASES) == 334
        for name, expected in manifest['files'].items():
            relative = PurePosixPath(name)
            assert not relative.is_absolute() and '..' not in relative.parts
            content = archive.read(name)
            assert hashlib.sha256(content).hexdigest() == expected
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    return root


@pytest.mark.parametrize('case', CASES)
def test_official_ids_case(corpus, case):
    path = corpus / case
    expected = path.name.split('-', 1)[0]
    try:
        result = IDS.ValidateFiles(path.with_suffix('.ifc'), path)
        actual = result['status']
        assert result['summary']['findings'] == len(result['findings'])
    except ValueError as error:
        assert 'IDS is invalid or unsupported' in str(error)
        actual = 'rejected_ids'
    assert actual == expected if expected != 'invalid' else actual in {'fail', 'rejected_ids'}


def test_document_and_model_api_agree(corpus):
    case = corpus / CASES[0]
    document = IDS.ByPath(case)
    model = IFC.ModelByPath(str(case.with_suffix('.ifc')), silent=True)
    assert IDS.Validate(model, document) == IDS.ValidateFiles(case.with_suffix('.ifc'), case)


def test_reader_decodes_step_and_preserves_logical_unknown(tmp_path):
    path = tmp_path / 'unicode.ifc'
    path.write_text(r"""ISO-10303-21;
HEADER;FILE_SCHEMA(('IFC4'));ENDSEC;
DATA;
#1=IFCPROPERTYSINGLEVALUE('Unicode',$,IFCLABEL('\X2\00E9\X0\'),$);
#2=IFCPROPERTYSINGLEVALUE('Unicode4',$,IFCLABEL('\X4\0001F600\X0\'),$);
#3=IFCPROPERTYSINGLEVALUE('Logical',$,IFCLOGICAL(.U.),$);
#4=IFCPROPERTYSINGLEVALUE('Text',$,IFCLABEL('UNKNOWN'),$);
#5=IFCPROPERTYSINGLEVALUE('Missing',$,$,$);
ENDSEC;END-ISO-10303-21;""", encoding='utf-8')
    model = IFC.ModelByPath(str(path), silent=True)
    assert model.entities[1].args[2][2] == ['é']
    assert model.entities[2].args[2][2] == ['😀']
    assert isinstance(model.entities[3].args[2][2][0], IFCLogicalUnknown)
    assert type(model.entities[4].args[2][2][0]) is str
    assert model.entities[5].args[2] is None


def test_invalid_xml_and_external_entities_are_rejected(tmp_path):
    path = tmp_path / 'unsafe.ids'
    for content in ['<broken>', '<!DOCTYPE ids [<!ENTITY x SYSTEM "file:///etc/passwd">]><ids xmlns="http://standards.buildingsmart.org/IDS">&x;</ids>']:
        path.write_text(content)
        with pytest.raises(ValueError, match='IDS is invalid or unsupported'):
            IDS.ByPath(path)


def test_reading_and_ids_never_import_legacy_sdks():
    # Separate interpreter proves absence of even an attempted optional SDK import.
    code = '''import importlib.abc, sys
attempts = []
class BlockLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'ifcopenshell', 'ifctester'}:
            attempts.append(fullname)
            raise ImportError(fullname)
sys.meta_path.insert(0, BlockLegacy())
from topologicpy.IDS import IDS
from topologicpy.IFC import IFC
assert not attempts, attempts
assert not any(name.split('.')[0] in {'ifcopenshell','ifctester'} for name in sys.modules)
'''
    subprocess.run([sys.executable, '-c', code], check=True)


def test_optional_exporter_fails_cleanly_without_sdk(tmp_path):
    if importlib.util.find_spec('ifcopenshell') is not None:
        pytest.skip('Optional exporter SDK is available in this environment')
    assert IFC.ExportReferenceView(None, str(tmp_path / 'unused.ifc'), silent=True) is None
    assert not (tmp_path / 'unused.ifc').exists()


def test_ifc43_assessment_uses_final_schema_metadata(tmp_path):
    model_path = tmp_path / 'road.ifc'
    model_path.write_text("ISO-10303-21;HEADER;FILE_SCHEMA(('IFC4X3_ADD2'));ENDSEC;DATA;#1=IFCROAD('road-guid',$,'Road',$,$,$,$,$,.ROAD.);ENDSEC;END-ISO-10303-21;")
    ids_path = tmp_path / 'road.ids'
    ids_path.write_text('<ids xmlns="http://standards.buildingsmart.org/IDS"><info><title>Road test</title></info><specifications><specification name="Road name" ifcVersion="IFC4X3_ADD2"><applicability minOccurs="1" maxOccurs="unbounded"><entity><name><simpleValue>IFCROAD</simpleValue></name></entity></applicability><requirements><attribute><name><simpleValue>Name</simpleValue></name></attribute></requirements></specification></specifications></ids>')
    result = IDS.ValidateFiles(model_path, ids_path)
    assert result['status'] == 'pass' and result['summary']['checks'] == 1
    assert result['warnings'] == []


def test_unrecognized_ifc43_variant_is_not_treated_as_add2(tmp_path):
    model_path = tmp_path / 'rc.ifc'
    model_path.write_text("ISO-10303-21;HEADER;FILE_SCHEMA(('IFC4X3_RC1'));ENDSEC;DATA;#1=IFCROAD('road-guid',$,'Road',$,$,$,$,$,.ROAD.);ENDSEC;END-ISO-10303-21;")
    model = IFC.ModelByPath(str(model_path), silent=True)
    with pytest.raises(ValueError, match='IFC schema is not supported'):
        IDS.Validate(model, None)


def test_escaped_step_backslash_does_not_start_unicode_directive(tmp_path):
    path = tmp_path / 'literal.ifc'
    path.write_text(r"""ISO-10303-21;HEADER;FILE_SCHEMA(('IFC4'));ENDSEC;DATA;
#1=IFCPROPERTYSINGLEVALUE('Literal',$,IFCLABEL('\\X2\\00E9\\X0\\'),$);
ENDSEC;END-ISO-10303-21;""")
    model = IFC.ModelByPath(str(path), silent=True)
    assert model.entities[1].args[2][2] == ['\\X2\\00E9\\X0\\']
    model.sourcePath = None
    output = tmp_path / 'literal-roundtrip.ifc'
    assert IFC.Export(model, str(output), silent=True) == str(output)
    reparsed = IFC.ModelByPath(str(output), silent=True)
    assert reparsed.entities[1].args == model.entities[1].args


def test_logical_unknown_roundtrips_without_becoming_text(tmp_path):
    path = tmp_path / 'unknown.ifc'
    path.write_text("ISO-10303-21;HEADER;FILE_SCHEMA(('IFC4'));ENDSEC;DATA;#1=IFCPROPERTYSINGLEVALUE('Logical',$,IFCLOGICAL(.U.),$);#2=IFCPROPERTYSINGLEVALUE('Literal',$,IFCLABEL('UNKNOWN'),$);ENDSEC;END-ISO-10303-21;")
    model = IFC.ModelByPath(str(path), silent=True)
    model.sourcePath = None
    output = tmp_path / 'unknown-roundtrip.ifc'
    assert IFC.Export(model, str(output), silent=True) == str(output)
    reparsed = IFC.ModelByPath(str(output), silent=True)
    assert isinstance(reparsed.entities[1].args[2][2][0], IFCLogicalUnknown)
    assert type(reparsed.entities[2].args[2][2][0]) is str
