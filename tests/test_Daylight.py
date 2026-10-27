import math
import numpy as np
import pytest
from topologicpy.Daylight import Daylight
from topologicpy.Cell import Cell
from topologicpy.Face import Face
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex

def polygon(points):
    return Face.ByVertices([Vertex.ByCoordinates(*p) for p in points])

def room(window=True):
    c = Cell.Prism(Vertex.ByCoordinates(0, 0, 0), width=4, length=4, height=3, placement='lowerleft')
    w = polygon([(1, 0, 1), (3, 0, 1), (3, 0, 2.5), (1, 0, 2.5)])
    if window:
        c = Topology.AddApertures(c, [w], subTopologyType='face')
    return Daylight.PrepareRoom(c)

@pytest.fixture(scope='module')
def model():
    r = room()
    grid = Daylight.SensorGrid(Daylight.WorkingPlane(r['cell']), spacing=2)
    before = Daylight.Waldram(r, Daylight.PrepareScene([]), grid=grid, resolution='low')
    obstruction = polygon([(-5, -1, 0), (9, -1, 0), (9, -1, 9), (-5, -1, 9)])
    after = Daylight.Waldram(r, Daylight.PrepareScene([obstruction]), grid=grid, resolution='low')
    return (r, grid, before, after)

@pytest.mark.parametrize('sky', ['uniform', 'cie_overcast'])
def test_external_sky_normalization(sky):
    scene = Daylight.PrepareScene([])
    values = Daylight.SkyComponent([[0, 0, 0]] * 3, [[0, 0, 1], [0, 0, -1], [0, 1, 0]], scene, sky=sky, resolution='low')['values']
    assert values[0] == pytest.approx(100, abs=1e-10)
    assert values[1] == 0
    assert values[2] == pytest.approx(50 if sky == 'uniform' else 39.62, abs=0.12)

def test_apertures_are_real_holes(model):
    r, grid, before, after = model
    assert len(r['apertures']) == 1
    assert sum((Face.Area(f) for f in r['opaqueFaces'])) == pytest.approx(Cell.Area(r['cell']) - 3)
    assert max(before['values']) > 0
    assert max(after['values']) == 0
    assert all((y <= x for x, y in zip(before['values'], after['values'])))

def test_grid_area_and_internal_sensors(model):
    r, g, _, _ = model
    assert g['area'] == pytest.approx(16)
    assert len(g['vertices']) == len(g['areas'])
    assert all((Vertex.IsInternal(v, r['cell']) for v in g['vertices']))

def test_independent_rectangle_integral(model):
    from scipy.integrate import dblquad
    _, g, b, _ = model
    for p, actual in zip(g['coordinates'], b['values']):
        x, y, z = p
        expected = dblquad(lambda zz, xx: 100 * y * max(zz - z, 0) / (math.pi * ((xx - x) ** 2 + y * y + (zz - z) ** 2) ** 2), 1, 3, lambda _: 1, lambda _: 2.5)[0]
        assert actual == pytest.approx(expected, rel=0.004, abs=0.002)

def test_comparison_and_contours(model):
    _, g, b, a = model
    comparison = Daylight.Compare(b, a)
    assert comparison['percentagePointChange'] < 0
    assert comparison['areaGained'] == 0
    assert Daylight.Compare(b, b)['percentagePointChange'] == 0
    contour = Daylight.Contours(b)[0]
    assert contour['areaAboveThreshold'] == pytest.approx(b['summary']['areaAboveThreshold'])
    assert sum((Face.Area(f) for f in contour['aboveFaces'] + contour['belowFaces'])) == pytest.approx(g['area'])
    assert contour['boundaryWires']

def test_diagram_contribution_consistency(model):
    _, _, b, _ = model
    d = Daylight.WaldramDiagram(b, 0, tolerance=1e-08)
    assert d['skyFactor'] == pytest.approx(b['values'][0])
    assert sum((p['contribution'] for p in d['patches'])) == pytest.approx(d['skyFactor'], rel=1e-08)
    assert d['omittedPatchCount'] == 0
    assert all((Topology.IsInstance(p['face'], 'Face') for p in d['patches']))

def test_no_apertures():
    r = room(False)
    a = Daylight.Waldram(r, Daylight.PrepareScene([]), spacing=4, resolution='low')
    assert a['values'] and (not any(a['values']))
    assert a['warnings']

@pytest.mark.parametrize('spacing', [0, -1, float('nan')])
def test_bad_grid_spacing(spacing):
    with pytest.raises(ValueError):
        Daylight.SensorGrid([Face.Rectangle()], spacing=spacing)

def test_incompatible_and_cancelled(model):
    r, g, b, _ = model
    with pytest.raises(ValueError):
        Daylight.Compare(b, dict(b, gridId='different'))
    with pytest.raises(InterruptedError):
        Daylight.Waldram(r, Daylight.PrepareScene([]), grid=g, cancelled=lambda: True)
    with pytest.raises(ValueError):
        Daylight.WorkingPlane(r['cell'], height=4)
    with pytest.raises(ValueError):
        Daylight.SkyComponent([[0, 0, 0]], [0, 0, 0], Daylight.PrepareScene([]))

def test_partial_obstruction_and_rotation(model):
    r, g, b, _ = model
    wall = polygon([(0, -1, 0), (2, -1, 0), (2, -1, 6), (0, -1, 6)])
    a = Daylight.Waldram(r, Daylight.PrepareScene([wall]), grid=g, resolution='low')
    assert any((y < x for x, y in zip(b['values'], a['values'])))
    assert all((y <= x + 1e-12 for x, y in zip(b['values'], a['values'])))
    rotated = Topology.Rotate(r['cell'], origin=Vertex.Origin(), axis=[0, 0, 1], angle=65)
    apertures = [Topology.Rotate(w, origin=Vertex.Origin(), axis=[0, 0, 1], angle=65) for w in r['apertures']]
    rotated = Topology.AddApertures(rotated, apertures, subTopologyType='face')
    rr = Daylight.PrepareRoom(rotated)
    rg = dict(g, vertices=[Topology.Rotate(v, origin=Vertex.Origin(), axis=[0, 0, 1], angle=65) for v in g['vertices']])
    rg['coordinates'] = [Vertex.Coordinates(v) for v in rg['vertices']]
    rb = Daylight.Waldram(rr, Daylight.PrepareScene([]), grid=rg, resolution='low')
    assert rb['values'] == pytest.approx(b['values'], abs=0.02)

def test_duplicate_and_overlapping_apertures():
    c = room(False)['cell']
    w1 = polygon([(1, 0, 1), (2.5, 0, 1), (2.5, 0, 2.5), (1, 0, 2.5)])
    w2 = polygon([(2, 0, 1), (3, 0, 1), (3, 0, 2.5), (2, 0, 2.5)])
    c = Topology.AddApertures(c, [w1, w2, w1], subTopologyType='face')
    r = Daylight.PrepareRoom(c)
    assert sum((Face.Area(w) for w in r['apertures'])) == pytest.approx(3)
    a = Daylight.Waldram(r, Daylight.PrepareScene([]), spacing=2, resolution='low')
    baseline = Daylight.Waldram(room(), Daylight.PrepareScene([]), grid=a['grid'], resolution='low')
    assert a['values'] == pytest.approx(baseline['values'], abs=0.02)

def test_inclined_window_and_face_winding():
    bottom = polygon([(0, 0, 0), (4, 0, 0), (4, 4, 0), (0, 4, 0)])
    top = polygon([(0, 1, 3), (4, 1, 3), (4, 4, 3), (0, 4, 3)])
    left = polygon([(0, 0, 0), (0, 4, 0), (0, 4, 3), (0, 1, 3)])
    right = polygon([(4, 0, 0), (4, 1, 3), (4, 4, 3), (4, 4, 0)])
    back = polygon([(0, 4, 0), (4, 4, 0), (4, 4, 3), (0, 4, 3)])
    front = polygon([(0, 0, 0), (0, 1, 3), (4, 1, 3), (4, 0, 0)])
    cell = Cell.ByFaces([bottom, top, left, right, back, front])
    points = [(1, 1 / 3, 1), (3, 1 / 3, 1), (3, 5 / 6, 2.5), (1, 5 / 6, 2.5)]
    results = []
    for order in (points, list(reversed(points))):
        c = Topology.Copy(cell)
        c = Topology.AddApertures(c, [polygon(order)], subTopologyType='face')
        r = Daylight.PrepareRoom(c)
        a = Daylight.Waldram(r, Daylight.PrepareScene([]), spacing=2, resolution='medium', checkConvergence=False)
        assert max(a['values']) > 0
        results.append(a['values'])
    assert results[0] == pytest.approx(results[1], abs=0.02)

def test_invalid_aperture():
    c = room(False)['cell']
    w = polygon([(1, 0.05, 1), (3, 0.05, 1), (3, 0.05, 2.5), (1, 0.05, 2.5)])
    c = Topology.AddApertures(c, [w])
    with pytest.raises(ValueError, match='coplanar host'):
        Daylight.PrepareRoom(c)

def test_spatial_convergence(model):
    r, _, coarse, _ = model
    fine = Daylight.Waldram(r, Daylight.PrepareScene([]), spacing=1, resolution='low')
    report = Daylight.Convergence(coarse, fine)
    assert report['sameGrid'] is False
    assert report['maxPointChange'] is None
    assert report['fineSensorCount'] > report['coarseSensorCount']

def test_concave_and_holed_grid():
    from topologicpy.Wire import Wire
    external = Face.ExternalBoundary(polygon([(0, 0, 1), (4, 0, 1), (4, 4, 1), (2, 4, 1), (2, 2, 1), (0, 2, 1)]))
    hole = Face.ExternalBoundary(polygon([(2.5, 2.5, 1), (3.5, 2.5, 1), (3.5, 3.5, 1), (2.5, 3.5, 1)]))
    f = Face.ByWires(external, [hole])
    g = Daylight.SensorGrid(f, spacing=0.7, rotation=17)
    assert g['area'] == pytest.approx(11, abs=1e-06)
    assert all((Vertex.IsInternal(v, f) for v in g['vertices']))
    with pytest.raises(ValueError):
        Daylight.SensorGrid([f, f])

@pytest.mark.parametrize('normal', [[0, 0, 0], [float('nan'), 0, 1]])
def test_invalid_normals(normal):
    with pytest.raises(ValueError):
        Daylight.SkyComponent([[0, 0, 0]], normal, Daylight.PrepareScene([]))

def test_contour_mixed_cells(model):
    _, g, b, _ = model
    vals = [0 if i % 2 else 1 for i in range(len(g['cells']))]
    mixed = dict(b, values=vals)
    contour = Daylight.Contours(mixed)[0]
    assert sum((Face.Area(f) for f in contour['aboveFaces'])) == pytest.approx(sum((a for a, v in zip(g['areas'], vals) if v >= 0.2)))
    assert contour['belowFaces']
    zero = Daylight.Contours(dict(b, values=[0] * len(vals)))[0]
    assert not zero['aboveFaces'] and (not zero['boundaryWires'])

def test_sky_blocker_and_cancel():
    roof = polygon([(-100, -100, 1), (100, -100, 1), (100, 100, 1), (-100, 100, 1)])
    r = Daylight.SkyFactor([[0, 0, 0]], Daylight.PrepareScene([roof]), resolution='low')
    assert r['values'][0] < 0.1
    with pytest.raises(InterruptedError):
        Daylight.SkyFactor([[0, 0, 0]], Daylight.PrepareScene([]), cancelled=lambda: True)

def test_portable_export(model):
    import json
    _, _, b, _ = model
    data = Daylight.ResultData(b, includeGeometry=True)
    restored = json.loads(json.dumps(data, allow_nan=False))
    assert restored['values'] == b['values']
    assert '_diagram' not in restored
    assert len(restored['grid']['cellBREPs']) == len(b['grid']['cells'])
    for brep in restored['grid']['cellBREPs']:
        assert Topology.IsInstance(Topology.ByBREPString(brep), 'Face')

def test_multiple_windows(model):
    r, g, baseline, _ = model
    c = room(False)['cell']
    windows = [polygon([(1, 0, 1), (2, 0, 1), (2, 0, 2.5), (1, 0, 2.5)]), polygon([(2, 0, 1), (3, 0, 1), (3, 0, 2.5), (2, 0, 2.5)])]
    c = Topology.AddApertures(c, windows, subTopologyType='face')
    actual = Daylight.Waldram(Daylight.PrepareRoom(c), Daylight.PrepareScene([]), grid=g, resolution='low')
    assert actual['values'] == pytest.approx(baseline['values'], abs=0.02)

def test_reproducible_grid_id(model):
    r, g, _, _ = model
    again = Daylight.SensorGrid(Daylight.WorkingPlane(r['cell']), spacing=2)
    assert again['gridId'] == g['gridId']

def test_different_domain_rejected(model):
    _, _, b, _ = model
    shift = polygon([(5, 0, 0.85), (9, 0, 0.85), (9, 4, 0.85), (5, 4, 0.85)])
    g = Daylight.SensorGrid(shift, 2)
    shifted = dict(b, grid=g, gridId=g['gridId'])
    with pytest.raises(ValueError, match='domain shapes'):
        Daylight.Convergence(b, shifted)

def test_room_identity_survives_native_preparation(model):
    r, grid, original, _ = model
    # On TopologicCore, sectioning/meshing adds representation data to the
    # shared room shape. Its physical boundaries and apertures remain unchanged.
    Daylight.PrepareScene([r])
    Daylight.WorkingPlane(r['cell'], height=1.1)
    repeated = Daylight.Waldram(r, Daylight.PrepareScene([]), grid=grid, resolution='low')
    assert repeated['roomId'] == original['roomId']
    assert Daylight.Compare(original, repeated)['percentagePointChange'] == 0


def test_scene_identity_ignores_serialization_and_face_order():
    points = [(0, 0, 1), (2, 0, 1), (2, 2, 1), (0, 2, 1)]
    a = polygon(points)
    b = polygon([(x, y, z+2) for x, y, z in points])
    first = Daylight.PrepareScene([a, b])
    Topology.Tessellate(a, silent=True)
    assert Daylight.PrepareScene([b, a])['sceneId'] == first['sceneId']
    reordered = polygon(list(reversed(points[1:] + points[:1])))
    assert Daylight.PrepareScene([reordered, b])['sceneId'] == first['sceneId']
    moved = Topology.Translate(b, z=.1)
    assert Daylight.PrepareScene([a, moved])['sceneId'] != first['sceneId']


def test_window_changes_still_reject_comparison(model):
    _, grid, original, _ = model
    cell = room(False)['cell']
    small_window = polygon([(1.5, 0, 1), (2.5, 0, 1), (2.5, 0, 2), (1.5, 0, 2)])
    cell = Topology.AddApertures(cell, [small_window], subTopologyType='face')
    altered = Daylight.Waldram(Daylight.PrepareRoom(cell), Daylight.PrepareScene([]), grid=grid, resolution='low')
    assert altered['roomId'] != original['roomId']
    with pytest.raises(ValueError, match='Incompatible'):
        Daylight.Compare(original, altered)
    with pytest.raises(ValueError, match='Different room'):
        Daylight.Convergence(original, altered)


def test_scene_identity_preserves_aperture_holes():
    outer = Face.ExternalBoundary(polygon([(0,0,1), (3,0,1), (3,3,1), (0,3,1)]))
    small_hole = Face.ExternalBoundary(polygon([(1,1,1), (2,1,1), (2,2,1), (1,2,1)]))
    large_hole = Face.ExternalBoundary(polygon([(.5,.5,1), (2.5,.5,1), (2.5,2.5,1), (.5,2.5,1)]))
    a = Daylight.PrepareScene([Face.ByWires(outer, [small_hole])])
    b = Daylight.PrepareScene([Face.ByWires(outer, [large_hole])])
    assert a['sceneId'] != b['sceneId']
