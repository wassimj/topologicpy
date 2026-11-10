import pytest
from topologicpy.io import tpy
from topologicpy.Topology import Topology


class Shape:
    def __init__(self, identity, brep, signature):
        self.identity, self.brep, self.signature = identity, brep, signature


@pytest.fixture
def native(monkeypatch):
    root=Shape('root','root','root')
    first=Shape('one','coincident','same')
    alias=Shape('one','coincident','same')
    second=Shape('two','coincident','same')
    third=Shape('three','third','unique')
    calls={'enumerate':0,'brep':0,'signature':0}
    def sub(*args,**kwargs):calls['enumerate']+=1;return [root,first,alias,second,third]
    def brep(item):calls['brep']+=1;return item.brep
    def signature(item):calls['signature']+=1;return {'hash':item.signature}
    monkeypatch.setattr(Topology,'SubTopologies',sub)
    monkeypatch.setattr(Topology,'IsInstance',lambda item,kind:isinstance(item,Shape))
    monkeypatch.setattr(tpy,'_same_topology',lambda a,b:a.identity==b.identity)
    monkeypatch.setattr(tpy,'_candidate_brep_hash',brep)
    monkeypatch.setattr(tpy,'_signature',signature)
    monkeypatch.setattr(tpy.TPYCodec,'_IdentityBucket',lambda item:hash(item.identity))
    return root,first,second,third,calls


def test_locator_cache_preserves_native_occurrence_and_coincident_ambiguity(native):
    root,first,second,third,calls=native;cache={}
    locator={'type':'face','index':1,'brepSha256':'coincident','signature':{'hash':'same'}}
    assert tpy._resolve_locator(root,locator,cache) is second
    locator['index']=0
    assert tpy._resolve_locator(root,locator,cache) is first
    locator['index']=99
    assert tpy._resolve_locator(root,locator,cache) is None
    assert calls=={'enumerate':1,'brep':3,'signature':0}


def test_signature_fallback_cached_once_and_new_load_gets_fresh_candidates(native):
    root,first,second,third,calls=native;cache={}
    locator={'type':'face','index':-1,'brepSha256':'old-writer-hash','signature':{'hash':'unique'}}
    assert tpy._resolve_locator(root,locator,cache) is third
    assert tpy._resolve_locator(root,locator,cache) is third
    assert calls=={'enumerate':1,'brep':3,'signature':3}
    assert tpy._resolve_locator(root,locator,{}) is third
    assert calls['enumerate']==2


def test_index_only_and_invalid_locators_keep_existing_behaviour(native):
    root,first,second,third,calls=native;cache={}
    assert tpy._resolve_locator(root,{'type':'face','index':2},cache) is third
    assert tpy._resolve_locator(root,{'type':'face','index':True},cache) is None
    assert tpy._resolve_locator(root,{'type':'unknown','index':0},cache) is None
    assert tpy._resolve_locator(root,None,cache) is root
    assert calls['enumerate']==1


def test_hash_collisions_and_unavailable_hashes_use_exact_identity(native,monkeypatch):
    root,first,second,third,calls=native
    for bucket in (lambda item:0,lambda item:None):
        monkeypatch.setattr(tpy.TPYCodec,'_IdentityBucket',bucket)
        values=tpy.TPYCodec._UniqueNative([first,first,second,third],root)
        assert values==[first,second,third]


def test_verified_metadata_reads_detect_changes_and_unexpected_dictionaries(tmp_path):
    import json,zipfile
    from topologicpy.Cell import Cell
    from topologicpy.Dictionary import Dictionary
    cell=Cell.Prism(width=2,length=3,height=4)
    cell=Topology.SetDictionary(cell,Dictionary.ByPythonDictionary({'name':'cache test','values':[1,2,3]}))
    path=tmp_path/'metadata.tpy'
    assert Topology.Save(cell,path,overwrite=True,silent=True)
    with zipfile.ZipFile(path) as archive:manifest=json.loads(archive.read('manifest.json'))
    assert manifest['formatVersion'] in (1,2)
    loaded={}
    restored=tpy.TPYCodec.load(path,silent=True,_loaded=loaded)
    assert restored is not None
    snapshot=tpy.TPYCodec._ReadMetadataManifest(manifest,loaded)
    fingerprint=lambda value:tpy.TPYCodec._MetadataFingerprint(value,include_geometry_signatures=manifest['formatVersion']==2)
    assert fingerprint(snapshot)==fingerprint(manifest)
    edge=Topology.Edges(restored)[0]
    Topology.SetDictionary(edge,Dictionary.ByPythonDictionary({'unexpected':True}))
    assert fingerprint(tpy.TPYCodec._ReadMetadataManifest(manifest,loaded))!=fingerprint(manifest)
    with pytest.raises(ValueError,match='relationship'):
        tpy.TPYCodec._ReadMetadataManifest(dict(manifest,relations=[{}]),loaded)


def test_occt_hash_bucket_preserves_reversed_identity_and_independent_coincidence():
    from topologicpy.Cell import Cell
    from topologicpy.Vertex import Vertex
    first=Cell.Prism(width=2,length=3,height=4)
    native=Topology.OCCTShape(first,silent=True)
    if native is None or tpy.TPYCodec._IdentityBucket(first) is None:pytest.skip('Requires the OCCT backend')
    reversed_shape=Topology.ByOCCTShape(native.Reversed(),silent=True)
    assert tpy._same_topology(first,reversed_shape)
    assert tpy.TPYCodec._IdentityBucket(first)==tpy.TPYCodec._IdentityBucket(reversed_shape)
    other=Cell.Prism(width=2,length=3,height=4)
    assert not tpy._same_topology(first,other)
    assert len(tpy.TPYCodec._UniqueNative([first,reversed_shape,other],Vertex.ByCoordinates(99,99,99)))==2
