from datetime import datetime, timedelta, timezone
import math
import numpy as np
import pytest
from topologicpy.Sun import Sun
from topologicpy.BVH import BVH
from topologicpy.Vertex import Vertex
from topologicpy.Face import Face
from topologicpy.Wire import Wire
from topologicpy.Topology import Topology
from topologicpy.Cell import Cell

UTC=timezone.utc
D=datetime(2024,6,21,12,tzinfo=UTC)
TRI=[[[0,0,1],[2,0,1],[0,2,1]]]
def empty():return BVH.ByTriangles([])
def record(direction,above=True,date=D):return {'toSun':direction,'aboveHorizon':above,'timestamp':date}

@pytest.mark.parametrize('lat,lon',[(51.48,-3.18),(-33.87,151.21),(35.68,139.69),(34.05,-118.24),(69.65,18.96)])
@pytest.mark.parametrize('month',[3,6,9,12])
def test_positions_against_pvlib(lat,lon,month):
    pytest.importorskip("ephem")
    pytest.importorskip("pvlib")
    d=D.replace(month=month)
    a=Sun.PositionData(lat,lon,d,pressure=0)
    b=Sun.PositionData(lat,lon,d,pressure=0,engine='pvlib')
    assert abs((a['azimuth']-b['azimuth']+180)%360-180)<0.003
    assert a['geometricElevation']==pytest.approx(b['geometricElevation'],abs=0.003)
    assert np.linalg.norm(a['toSun'])==pytest.approx(1)

def test_positions_instants_and_conventions():
    pytest.importorskip("ephem")
    a=Sun.PositionData(51,-3,D)
    b=Sun.PositionData(51,-3,D.astimezone(timezone(timedelta(hours=9))))
    assert a==b
    assert Sun.PositionData(51,-3,D.replace(tzinfo=None))==a
    assert Sun.Vector(51,-3,D)==a['fromSun']
    assert Sun.Vector(51,-3,D,reverse=True)==a['toSun']
    c=Sun.PositionData(51,-3,D,north=90)
    x,y,z=a['toSun']
    assert c['toSun']==pytest.approx([y,-x,z])
    assert Sun.PositionsByDates(0,0,[])==[]

@pytest.mark.parametrize('lat,lon',[(-91,0),(91,0),(0,181),(math.nan,0),(0,math.inf)])
def test_invalid_coordinates(lat,lon):
    pytest.importorskip("ephem")
    with pytest.raises(ValueError):Sun.PositionData(lat,lon,D)

def test_dst_naive_rejected_and_fold_respected():
    pytest.importorskip("ephem")
    for d in [datetime(2024,3,31,1,30),datetime(2024,10,27,1,30)]:
        with pytest.raises(ValueError):Sun.PositionData(51,0,d,timeZone='Europe/London')
    from zoneinfo import ZoneInfo
    d=datetime(2024,10,27,1,30,tzinfo=ZoneInfo('Europe/London'))
    assert Sun._Date(d)!=Sun._Date(d.replace(fold=1))

@pytest.mark.parametrize('lat,lon,zone',[(35.68,139.69,'Asia/Tokyo'),(-33.87,151.21,'Australia/Sydney'),(34.05,-118.24,'America/Los_Angeles')])
def test_events_same_civil_day(lat,lon,zone):
    pytest.importorskip("ephem")
    r=Sun.EventsByDate(lat,lon,D,timeZone=zone)
    assert r['sunrise'].date().isoformat()==r['date']==r['sunset'].date().isoformat()
    hours=sum((b.astimezone(UTC)-a.astimezone(UTC)).total_seconds()/3600 for a,b in r['daylightIntervals'])
    assert 8<hours<17
    assert r['sunrise']<r['sunset']

def test_utc_event_pair_and_polar_day():
    pytest.importorskip("ephem")
    r=Sun.EventsByDate(35.68,139.69,D)
    assert r['sunset']<r['sunrise']
    assert len(r['daylightIntervals'])==2
    a=Sun.EventsByDate(69.65,18.96,D)
    b=Sun.EventsByDate(69.65,18.96,D.replace(month=12))
    assert a['status']=='polar_day' and len(a['daylightIntervals'])==1
    assert b['status']=='polar_night' and b['daylightIntervals']==[]
    assert Sun.Sunrise(69.65,18.96,D) is None

def test_dst_day_duration():
    pytest.importorskip("ephem")
    r=Sun.EventsByDate(69.65,18.96,datetime(2024,3,31),timeZone='Europe/London')
    assert (r['end'].astimezone(UTC)-r['start'].astimezone(UTC)).total_seconds()==23*3600

def test_sampling_year_endpoints_and_validation(monkeypatch):
    pytest.importorskip("ephem")
    monkeypatch.setattr(Sun,'Vertex',staticmethod(lambda lat,lon,date,*a:date))
    a=Sun.VerticesByDate(0,0,D,D,D+timedelta(minutes=65),interval=60)
    assert a==[D,D+timedelta(hours=1),D+timedelta(minutes=65)]
    a=Sun.VerticesByHour(0,0,12,startDay=364,interval=5,year=2024)
    assert a[-1].day==31 and a[-1].year==2024 and len(a)==2
    for interval in [0,-1,float('nan')]:
        with pytest.raises(ValueError):Sun.VerticesByDate(0,0,D,interval=interval)
    with pytest.raises(ValueError):Sun.VerticesByHour(0,0,D)
    with pytest.raises(ValueError):Sun.VerticesByHour(0,0,12,startDay=0)
    assert Sun.VerticesByDate(69.65,18.96,D.replace(month=12))==[]

def test_seasons_and_drawing_endpoints():
    pytest.importorskip("ephem")
    assert Sun.SummerSolstice(-33,2024).month==12
    assert Sun.WinterSolstice(-33,2024).month==6
    path=Sun.PathByDate(51,-3,D,D,D+timedelta(hours=2),sides=4)
    vs=Wire.Vertices(path)
    assert len(vs)==5
    expected=Sun.Position(51,-3,D+timedelta(hours=2))
    assert Vertex.Coordinates(vs[-1])==pytest.approx(expected,abs=1e-5)

def test_actual_hits_and_bounds():
    scene=BVH.ByTriangles(TRI,sourceIds=[13],faceIds=[42])
    hits=BVH.RayHits(scene,[[.25,.25,0],[1.5,1.5,0]],[0,0,10])
    assert hits[0]['distance']==pytest.approx(1)
    assert hits[0]['sourceId']==13 and hits[0]['faceId']==42
    assert hits[1] is None
    assert not BVH.IsOccluded(scene,[.25,.25,0],[0,0,1],maxDistance=.5)[0]
    assert BVH.IsOccluded(scene,[.25,.25,0],[0,0,1],maxDistance=1)[0]
    assert not BVH.IsOccluded(scene,[.25,.25,0],[0,0,-1])[0]

def test_nearest_hit_and_two_sided():
    tris=TRI+[[[0,0,2],[0,2,2],[2,0,2]]]
    scene=BVH.ByTriangles(tris,maxLeafSize=1)
    assert BVH.RayHits(scene,[.25,.25,0],[0,0,1])[0]['primitiveId']==0
    assert BVH.RayHits(scene,[.25,.25,3],[0,0,-1])[0]['primitiveId']==1

def test_self_hit_and_other_host_surface():
    scene=BVH.ByTriangles(TRI+[[[0,0,2],[2,0,2],[0,2,2]]],sourceIds=[0,0])
    assert BVH.RayHits(scene,[.25,.25,1],[0,0,1])[0]['distance']==pytest.approx(1)
    assert not BVH.IsOccluded(BVH.ByTriangles(TRI),[.25,.25,1],[0,0,1])[0]

def test_boundaries_coplanar_and_translation():
    scene=BVH.ByTriangles(TRI)
    origins=np.array([[1,1,0],[0,0,0],[2.1,0,0],[.25,.25,1]])
    directions=np.array([[0,0,1],[0,0,1],[0,0,1],[1,0,0]])
    assert BVH.IsOccluded(scene,origins,directions).tolist()==[True,True,False,False]
    shift=np.array([1e6,-3e6,4e6])
    scene2=BVH.ByTriangles(np.array(TRI)+shift)
    assert np.array_equal(BVH.IsOccluded(scene,origins,directions),BVH.IsOccluded(scene2,origins+shift,directions))

def test_hole_and_open_surface_backend():
    f=Face.ByWires(Wire.Rectangle(width=4,length=4),[Wire.Rectangle(width=2,length=2)])
    f=Topology.Translate(f,0,0,1)
    scene=BVH.PrepareScene([f])
    assert BVH.IsOccluded(scene,[[0,0,0],[1.5,0,0]],[0,0,1]).tolist()==[False,True]
    assert BVH.IsOccluded(scene,[[0,0,2],[1.5,0,2]],[0,0,-1]).tolist()==[False,True]

def test_prism_shadow_and_facade():
    # Cube at x,y=0..2 and z=1..3. Upward rays hit its roof footprint.
    cube=Cell.Prism(origin=Vertex.ByCoordinates(1,1,2),width=2,length=2,height=2)
    scene=BVH.PrepareScene([cube])
    pts=[[x,y,0] for x in [-1,.5,1.5,3] for y in [-1,.5,1.5,3]]
    expected=[0<=x<=2 and 0<=y<=2 for x,y,z in pts]
    assert BVH.IsOccluded(scene,pts,[0,0,1]).tolist()==expected
    assert BVH.IsOccluded(scene,[[-1,1,2],[-1,3,2]],[1,0,0]).tolist()==[True,False]

def test_chunking_matches_and_cancellation():
    rng=np.random.default_rng(78)
    origins=np.column_stack([rng.uniform(-1,3,(200,2)),np.zeros(200)])
    scene=BVH.ByTriangles(TRI,maxLeafSize=1)
    a=BVH.IsOccluded(scene,origins,[0,0,1],batchSize=1)
    assert np.array_equal(a,BVH.IsOccluded(scene,origins,[0,0,1],batchSize=200))
    with pytest.raises(InterruptedError):BVH.IsOccluded(scene,origins,[0,0,1],cancelled=lambda:True)

@pytest.mark.parametrize('origins,directions',[([0,0,0],[0,0,0]),([math.nan,0,0],[0,0,1]),([[0,0,0],[0,0,0]],[[0,0,1]]*3)])
def test_invalid_rays(origins,directions):
    with pytest.raises(ValueError):BVH.IsOccluded(empty(),origins,directions)

def test_visibility_states_and_integration():
    scene=BVH.ByTriangles(TRI)
    positions=[record([0,0,1]),record([1,0,0]),record([0,0,-1],False)]
    result=Sun.Visibility([[.25,.25,0],[3,3,0]],[0,0,1],positions,scene,batchSize=2)
    assert result['states'].tolist()==[[1,2,3],[0,2,3]]
    a=Sun.SolarAccess([[.25,.25,0],[3,3,0]],[0,0,1],positions,scene,[1800,900,900],areas=[1,3])
    assert a['sunlightHours'].tolist()==[0,.5]
    assert a['occludedHours'].tolist()==[.5,0]
    assert a['areaWeightedSunlightHours']==pytest.approx(.375)

def test_time_edges_and_limits():
    p=[record([0,0,1]),record([0,0,1],date=D+timedelta(hours=1))]
    r=Sun.SolarAccess([0,0,0],[0,0,1],p,empty(),timeEdges=[D,D+timedelta(hours=1),D+timedelta(minutes=65)])
    assert r['sunlightHours'][0]==pytest.approx(65/60)
    with pytest.raises(ValueError):Sun.SolarAccess([0,0,0],[0,0,1],p,empty(),[3600,-1])
    with pytest.raises(ValueError):Sun.Visibility([0,0,0],[0,0,1],p,empty(),maxPairs=1)
    with pytest.raises(InterruptedError):Sun.Visibility([0,0,0],[0,0,1],p,empty(),cancelled=lambda:True)

def test_sky_hemisphere_and_shadow():
    result=Sun.SkyVisibility([[0,0,0]]*3,[[0,0,1],[1,0,0],[0,0,-1]],empty(),azimuthDivisions=72,altitudeDivisions=24)
    assert result['skyViewFactor']==pytest.approx([1,.5,0],abs=.003)
    assert result['solidAngles'].sum()==pytest.approx(2*math.pi)
    canopy=Face.Rectangle(origin=Vertex.ByCoordinates(0,0,1),width=100,length=100)
    blocked=Sun.SkyVisibility([0,0,0],[0,0,1],BVH.PrepareScene([canopy]))
    assert blocked['skyViewFactor'][0]<.01

def test_empty_and_bad_scenes():
    assert BVH.RayHits(empty(),[0,0,0],[0,0,1])==[None]
    assert Sun.Visibility([0,0,0],[0,0,1],[],empty())['states'].shape==(1,0)
    with pytest.raises(ValueError):BVH.ByTriangles([[[0,0,0]]*3])
    with pytest.raises(ValueError):BVH.ByTriangles(TRI,sourceIds=[1.5])
    with pytest.raises(ValueError):BVH.PrepareScene([None])
    with pytest.raises(ValueError):BVH.IsOccluded(empty(),[0,0,0],[0,0,1],minDistance=-1)

def test_random_oblique_rays_against_planar_projection():
    f=Face.ByVertices([Vertex.ByCoordinates(0,0,3),Vertex.ByCoordinates(5,0,3),Vertex.ByCoordinates(0,5,3)])
    scene=BVH.PrepareScene([f])
    rng=np.random.default_rng(117)
    pts=np.column_stack([rng.uniform(-2,6,(150,2)),np.zeros(150)])
    d=[.2,-.3,1]
    expected=[]
    for xyz in pts:
        p=Vertex.Project(Vertex.ByCoordinates(xyz.tolist()),f,direction=d)
        expected.append(bool(p is not None and Vertex.IsInternal(p,f)))
    assert BVH.IsOccluded(scene,pts,d).tolist()==expected

def test_exact_analytic_oblique_wall_shadow():
    # Wall x=0, y=-5..5, z=0..4; sun at 45 degrees toward +X.
    wall=Face.ByVertices([Vertex.ByCoordinates(0,-5,0),Vertex.ByCoordinates(0,5,0),
        Vertex.ByCoordinates(0,5,4),Vertex.ByCoordinates(0,-5,4)])
    scene=BVH.PrepareScene([wall])
    points=[[-5,0,0],[-3,0,0],[-1,0,0],[1,0,0],[-3,6,0]]
    assert BVH.IsOccluded(scene,points,[1,0,1]).tolist()==[False,True,True,False,False]

def test_curved_tessellation_convergence():
    cylinder=Cell.Cylinder(origin=Vertex.ByCoordinates(0,0,1),radius=1,height=2)
    a=BVH.PrepareScene([cylinder],linearDeflection=.1,angularDeflection=15)
    b=BVH.PrepareScene([cylinder],linearDeflection=.005,angularDeflection=2)
    assert b['metadata']['triangles']>=a['metadata']['triangles']
    pts=[[x,y,-1] for x,y in [(0,0),(.5,.5),(2,0),(-2,0)]]
    assert BVH.IsOccluded(a,pts,[0,0,1]).tolist()==[True,True,False,False]
    assert np.array_equal(BVH.IsOccluded(a,pts,[0,0,1]),BVH.IsOccluded(b,pts,[0,0,1]))

def test_sky_wall_convergence_and_scene_cancel():
    a=Sun.SkyVisibility([0,0,0],[1,0,0],empty(),azimuthDivisions=12,altitudeDivisions=4)
    b=Sun.SkyVisibility([0,0,0],[1,0,0],empty(),azimuthDivisions=96,altitudeDivisions=32)
    assert abs(b['skyViewFactor'][0]-.5)<abs(a['skyViewFactor'][0]-.5)
    with pytest.raises(InterruptedError):BVH.PrepareScene([Face.Rectangle()],cancelled=lambda:True)

def test_diagram_explicit_year():
    pytest.importorskip("ephem")
    r=Sun.Diagram(51.48,-3.18,minuteInterval=120,dayInterval=60,uSides=12,vSides=12,year=2024,shell=True)
    assert len(r['date_paths'])==3
    assert Topology.IsInstance(r['shell'],'Shell')

def test_paths_preserve_metadata():
    pytest.importorskip("ephem")
    from topologicpy.Dictionary import Dictionary
    path=Sun.PathByHour(51,-3,12,startDay=1,endDay=30,year=2024,sides=3)
    d=Topology.Dictionary(path)
    assert Dictionary.ValueAtKey(d,'year')==2024
    assert Dictionary.ValueAtKey(d,'type')=='hour'
    with pytest.raises(ValueError):Sun.PathByHour(51,-3,12,sides=1,close=True)

def test_direction_magnitude_does_not_change_hits():
    scene=BVH.ByTriangles(TRI)
    assert BVH.IsOccluded(scene,[.25,.25,0],[0,0,1e300])[0]
    assert BVH.IsOccluded(scene,[.25,.25,0],[0,0,1e-300])[0]
