"""gbXML consumers require non-collinear initial polygon vertices."""
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
import pytest
from topologicpy.EnergyModel import EnergyModel

def xml(points):
    loop=''.join('<CartesianPoint>'+''.join(f'<Coordinate>{c}</Coordinate>' for c in p)+'</CartesianPoint>' for p in points)
    return '<gbXML xmlns="http://www.gbxml.org/schema"><Surface id="wall"><AdjacentSpaceId spaceIdRef="room"/><PlanarGeometry><PolyLoop>'+loop+'</PolyLoop></PlanarGeometry></Surface></gbXML>'

def coordinates(text):
    return [tuple(float(c.text) for c in p) for p in ET.fromstring(text).iter('{http://www.gbxml.org/schema}CartesianPoint')]

@pytest.mark.parametrize('points',[
    [(0,0,0),(1,0,0),(2,0,0),(2,0,1),(2,0,2),(0,0,2),(0,0,0)],
    [(0,0,0),(1,0,1),(2,0,2),(2,1,2),(1,1,1),(1,2,1),(0,2,0)],
])
def test_redundant_vertices_removed_without_changing_surface(points):
    from shapely.geometry import Polygon
    result=EnergyModel._CleanGBXMLPolyLoops(xml(points))
    cleaned=coordinates(result)
    # These examples project bijectively onto x/z or x/y.
    axes=(0,2) if all(p[1]==0 for p in points) else (0,1)
    before=Polygon([(p[axes[0]],p[axes[1]]) for p in points])
    after=Polygon([(p[axes[0]],p[axes[1]]) for p in cleaned])
    assert before.equals(after) and before.exterior.is_ccw==after.exterior.is_ccw
    a,b,c=cleaned[:3];u=[b[i]-a[i] for i in range(3)];v=[c[i]-a[i] for i in range(3)]
    assert sum((u[i]*v[j]-u[j]*v[i])**2 for i,j in [(0,1),(1,2),(2,0)])>0
    assert 'spaceIdRef="room"' in result and 'id="wall"' in result
    assert EnergyModel._CleanGBXMLPolyLoops(result)==result

def test_real_bend_and_opening_metadata_are_preserved():
    text=xml([(0,0,0),(1,.001,0),(2,0,0),(2,2,0),(0,2,0)])
    assert set(coordinates(EnergyModel._CleanGBXMLPolyLoops(text)))==set(coordinates(text))

def test_rounded_sloping_roof_starts_at_stable_corner():
    import numpy as np
    points=[(42,2,9.676881),(42,2.15,9.748414),(42,2.89,10.101306),
            (25,2.89,10.101306),(25,2.15,9.748414),(25,2,9.676881)]
    cleaned=EnergyModel._CleanGBXMLPolyLoops(xml(points))
    p=np.array(coordinates(cleaned))
    assert set(map(tuple,p))==set(points)  # Keep the rounded intermediate bends.
    normal=np.cross(p[1]-p[0],p[2]-p[0]);normal/=np.linalg.norm(normal)
    assert max(abs((p-p[0])@normal))<2e-6
    assert EnergyModel._CleanGBXMLPolyLoops(cleaned)==cleaned

def test_file_and_string_exports_use_same_cleanup(monkeypatch,tmp_path):
    text=xml([(0,0,0),(1,0,0),(2,0,0),(2,2,0),(0,2,0)])
    class Translator:
        def modelToGbXML(self,model,path):Path(path).write_text(text,encoding='utf-8');return True
        def modelToGbXMLString(self,model):return text
    monkeypatch.setattr(EnergyModel,'_ImportOpenStudio',staticmethod(lambda *a:SimpleNamespace(gbxml=SimpleNamespace(GbXMLForwardTranslator=Translator))))
    monkeypatch.setattr(EnergyModel,'_OSPath',staticmethod(lambda path,studio:path))
    path=tmp_path/'test.xml'
    assert EnergyModel.ExportToGBXML(object(),str(path)) is True
    assert path.read_text()==EnergyModel.GBXMLString(object())
    assert len(coordinates(path.read_text()))==4
