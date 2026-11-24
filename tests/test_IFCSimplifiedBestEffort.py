"""Best-effort simplified reconstruction preserves rooms and valid components."""
from pathlib import Path
from copy import deepcopy
import os
import pytest
from shapely.geometry import box,shape,mapping
from shapely.ops import unary_union
from topologicpy.IFC import IFC
from topologicpy.Topology import Topology
from topologicpy.Cell import Cell

pytestmark=pytest.mark.pythonocc_only
SETTINGS=dict(minimum_clear_width=.6,minimum_clear_height=1.8,simplification=.1,drafting_grid=.05,tolerance=1e-5,role_overrides={})

def record(identity,polygon):
    return dict(id=identity,source=str(identity),type='IFCSPACE',footprint=polygon,z0=0.,z1=3.,name='Office '+str(identity),long_name='',building=1,storey=2)

def rooms():
    return [dict(id='a',source_space_id='1',footprint=mapping(box(0,0,4,4)),bottom=0.,top=3.,storey=2,name='A'),dict(id='b',source_space_id='2',footprint=mapping(box(4.2,0,8.2,4)),bottom=0.,top=3.,storey=2,name='B')]

def test_retry_assigns_small_gaps_without_losing_room_labels():
    result=IFC._Simplified_grid_partition(rooms(),SETTINGS,.1,bridge_gaps=True)
    polygons=[shape(r['footprint']) for r in result]
    assert {r['source_space_id'] for r in result}=={'1','2'}
    assert unary_union(polygons).geom_type=='Polygon'
    assert sum(p.area for p in polygons)==pytest.approx(unary_union(polygons).area)
    assert sum(r['gap_fill_area_m2'] for r in result)>0

def test_retry_preserves_a_large_narrow_courtyard_even_on_a_coarse_grid():
    polygon=box(0,0,8,8).difference(box(3.9,2,4.1,6))
    source=dict(rooms()[0],footprint=mapping(polygon))
    result=IFC._Simplified_grid_partition([source],SETTINGS,.4,bridge_gaps=True)
    actual=unary_union([shape(r['footprint']) for r in result])
    assert actual.intersection(box(3.9,2,4.1,6)).area<1e-8

def test_disconnected_valid_rooms_are_returned_as_verified_components(tmp_path):
    result=IFC._Simplified_building([record(1,box(0,0,4,4)),record(2,box(10,0,14,4))],{},SETTINGS)
    assert result.valid
    assert len(result.spaces)==2
    assert {r['source_space_id'] for r in result.spaces}=={'1','2'}
    assert not result.statistics['connected_partition']
    assert result.statistics['connected_components']==2
    assert len(result.cell_complexes)==2
    assert all(Topology.IsInstance(c,'CellComplex') for c in result.cell_complexes)
    assert sum(Cell.Volume(c,mantissa=8) for c in Topology.Cells(result.topology))==pytest.approx(96)
    assert any(d.code=='SIMPLIFIED_DISCONNECTED_COMPONENTS_RETAINED' for d in result.diagnostics)
    path=tmp_path/'components.tpy'
    assert Topology.ExportToTPY(result.topology,str(path),overwrite=True,silent=True)
    restored=Topology.ByTPYPath(str(path),silent=True)
    assert len(IFC._Analytical_model_Result(topology=restored).cell_complexes)==2
    from topologicpy.Dictionary import Dictionary
    components=IFC._Analytical_model_Result(topology=restored).cell_complexes
    assert {Dictionary.ValueAtKey(Topology.Dictionary(c),'source_space_id') for part in components for c in Topology.Cells(part)}=={'1','2'}
    assert IFC._Analytical_persistence_geometry_signature(restored)==IFC._Analytical_persistence_geometry_signature(result.topology)

def test_connected_retry_is_preferred_over_the_saved_disconnected_candidate(monkeypatch):
    def connect(source,settings,grid,bridge_gaps=False):
        output=deepcopy(source)
        output[0]['footprint']=mapping(box(0,0,4,4))
        output[1]['footprint']=mapping(box(4,0,8,4))
        return output
    monkeypatch.setattr(IFC,'_Simplified_grid_partition',connect)
    result=IFC._Simplified_building([record(1,box(0,0,4,4)),record(2,box(10,0,14,4))],{},SETTINGS)
    assert result.valid and Topology.IsInstance(result.topology,'CellComplex')
    assert result.statistics['connected_partition']
    assert result.statistics['reconstruction_attempt']==2
    assert not any(d.code=='SIMPLIFIED_DISCONNECTED_COMPONENTS_RETAINED' for d in result.diagnostics)


def test_sigma_office_preserves_every_source_space_and_roundtrips(tmp_path):
    source=Path(os.environ.get('TOPOLOGICPY_BIMDATA',Path(__file__).parent/'data'/'bimdata'))/'M25_Sigma_Office.ifc'
    if not source.is_file():pytest.skip('Sigma Office integration fixture is not available')
    model=IFC.Model(str(source),silent=True)
    source_ids={str(e.args[0]) for e in model.entities.values() if e.type=='IFCSPACE'}
    result=IFC.AnalyticalModel_Simplified(str(source))
    assert result.valid,result.report()
    assert Topology.IsInstance(result.topology,'CellComplex')
    assert result.statistics['geometry_verified']
    assert result.fidelity['coarsened_partition']
    assert result.fidelity['inter_room_gap_fill_area_m2']>0
    shafts={str(e.args[0]) for e in model.entities.values() if e.type=='IFCSPACE' and len(e.args)>7 and str(e.args[7]).upper()=='SHAFT'}
    assert len(shafts)==2
    assert {r['source_space_id'] for r in result.spaces}==source_ids-shafts
    assert set(result.fidelity['excluded_source_spaces'])==shafts
    assert len(result.spaces)==len(source_ids)==150
    assert all(IFC._Simplified_occupiable(shape(r['footprint']),r['top']-r['bottom'],SETTINGS) for r in result.spaces)
    path=tmp_path/'Sigma.tpy'
    assert Topology.ExportToTPY(result.topology,str(path),overwrite=True,silent=True)
    restored=Topology.ByTPYPath(str(path),silent=True)
    assert IFC._Analytical_persistence_geometry_signature(restored)==IFC._Analytical_persistence_geometry_signature(result.topology)


def test_best_disconnected_candidate_prefers_more_shared_connectivity(monkeypatch):
    def improve(source,settings,grid,bridge_gaps=False):
        output=deepcopy(source)
        for item in output:
            if item['source_space_id']=='2':item['footprint']=mapping(box(4,0,8,4))
        return output
    monkeypatch.setattr(IFC,'_Simplified_grid_partition',improve)
    result=IFC._Simplified_building([record(1,box(0,0,4,4)),record(2,box(10,0,14,4)),record(3,box(20,0,24,4))],{},SETTINGS)
    assert result.valid and len(result.spaces)==3
    assert result.statistics['connected_components']==2
    assert result.statistics['reconstruction_attempt']==2
    assert {r['source_space_id'] for r in result.spaces}=={'1','2','3'}
