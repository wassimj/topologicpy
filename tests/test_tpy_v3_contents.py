"""TPY v3: apertures are tagged contents, not separate persistent objects."""
import json
import zipfile

import pytest

from topologicpy.Aperture import Aperture
from topologicpy.Context import Context
from topologicpy.Core import Core
from topologicpy.Dictionary import Dictionary
from topologicpy.Face import Face
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex
from topologicpy.io import tpy


def _tag(topology, value):
    return Topology.SetDictionary(topology, Dictionary.ByPythonDictionary({'type': value, 'marker': 'keep'}))


def _model():
    host = Face.Rectangle(width=4, length=4)
    opening = _tag(Face.Rectangle(width=1, length=1), 'Aperture')
    context = Context.ByTopologyParameters(host, u=0.25, v=0.75, w=0.5)
    assert Aperture.ByTopologyContext(opening, context) is not None
    ordinary = Vertex.ByCoordinates(1, 1, 0)
    assert Topology.AddContent(host, [ordinary], subTopologyType='self') is not None
    return host, opening, context


def test_aperture_is_content_and_keeps_geometric_type():
    host, opening, context = _model()
    assert Topology.IsInstance(opening, 'Face')
    assert Topology.IsInstance(opening, 'Aperture')
    assert Topology.TypeAsString(opening) == 'Face'
    assert Aperture.Topology(opening) is opening
    assert Aperture.ByTopologyContext(opening, context) is opening
    assert len(Topology.Contents(host)) == 2
    assert len(Topology.Apertures(host)) == 1
    assert len(Topology.Apertures(host, subTopologyType='all')) == 1
    assert Topology.IsSame(Topology.ApertureTopologies(host)[0], opening)
    assert len(Topology.Contexts(opening)) == 1
    assert Topology.InternalVertex(opening) is not None


@pytest.mark.parametrize('value, expected', [('aperture', True), ('APERTURE', True), ('not_aperture', False), (42, False), (None, False)])
def test_exact_semantic_aperture_tag(value, expected):
    face = _tag(Face.Rectangle(), value)
    assert Topology.IsInstance(face, 'Aperture') == expected
    assert (Aperture.Topology(face) is face) == expected


@pytest.mark.parametrize('contents, apertures, count', [(True, True, 2), (True, False, 1), (False, True, 1), (False, False, 0)])
def test_v3_relationships_and_independent_filters(tmp_path, contents, apertures, count):
    host, _, _ = _model()
    path = tmp_path / 'model.tpy'
    assert Topology.ExportToTPY(host, path, includeContents=contents, includeApertures=apertures)
    with zipfile.ZipFile(path) as archive:
        raw = archive.read('manifest.json')
        manifest = json.loads(raw)
    assert manifest['formatVersion'] == 3
    assert raw == json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    assert len(manifest['relations']) == count
    assert len(manifest['objects']) == count + 1
    assert all(r['kind'] == 'content' for r in manifest['relations'])
    restored = Topology.ByTPYPath(path)
    assert restored is not None
    assert len(Topology.Contents(restored)) == count
    found = Topology.Apertures(restored)
    assert len(found) == int(apertures)
    if apertures:
        assert Dictionary.ValueAtKey(Topology.Dictionary(found[0]), 'marker') == 'keep'
        assert len(Topology.Contexts(found[0])) == 1
        ctx = Topology.Contexts(found[0])[0]
        assert Topology.IsSame(Context.Topology(ctx), restored)
        assert (ctx.x, ctx.y, ctx.z) == pytest.approx((0.25, 0.75, 0.5))


def test_remove_content_removes_aperture_and_reverse_context():
    host, opening, _ = _model()
    assert Topology.RemoveContent(host, [opening]) is not None
    assert len(Topology.Contents(host)) == 1
    assert Topology.Apertures(host) == []
    assert Topology.Contexts(opening) == []


def test_manifest_limits_writer_and_reader(tmp_path, monkeypatch):
    host, _, _ = _model()
    path = tmp_path / 'valid.tpy'
    assert Topology.ExportToTPY(host, path)
    with zipfile.ZipFile(path) as archive:
        size = archive.getinfo('manifest.json').file_size
    assert tpy._MAX_MANIFEST_BYTES == 256 * 1024 * 1024
    monkeypatch.setattr(tpy, '_MAX_MANIFEST_BYTES', size - 1)
    # Reject oversized ZIP metadata before opening/decompressing the member.
    def forbidden(*args, **kwargs):
        pytest.fail('Oversized manifest was opened')
    with monkeypatch.context() as patch:
        patch.setattr(zipfile.ZipFile, 'open', forbidden)
        assert tpy.TPYCodec.load(path, silent=True) is None
    rejected = tmp_path / 'rejected.tpy'
    assert not tpy.TPYCodec.save(host, rejected, silent=True)
    assert not rejected.exists()
