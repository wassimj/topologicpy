# Copyright (C) 2026
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free Software
# Foundation, either version 3.0 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

"""Geometry-first, multi-level navigation models and graph construction.

Navigation coordinates are model units with world Z as up. IFC extraction
normalizes coordinates to metres. Planar faces may slope but must have a
nonzero XY projection. No spatial links are inferred across horizontal gaps.
"""
from dataclasses import dataclass, field
import copy
import math
import json

@dataclass
class NavigationModel:
    """Editable navigation geometry, independent of any application.

    faces and obstacles contain Topologic Faces. Face dictionaries retain
    storey/source identifiers. connectors contain the descriptors produced by
    Stair, Ramp and Lift. Storeys are dictionaries with id/name/elevation.
    Rebuilding a graph reads current geometry; parsing is never repeated.
    """
    faces: list = field(default_factory=list)
    obstacles: list = field(default_factory=list)
    connectors: list = field(default_factory=list)
    storeys: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    _ifc: dict = field(default_factory=dict, repr=False)

class Navigation:
    """Static methods for constructing and querying navigation models."""

    @staticmethod
    def _Metadata(topology):
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        d = Topology.Dictionary(topology)
        return Dictionary.PythonDictionary(d) if d else {}

    @staticmethod
    def _Face(record):
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        def wire(points):
            clean = []
            for point in points:
                if not clean or math.dist(point, clean[-1]) > 1e-07:
                    clean.append(point)
            if len(clean) > 1 and math.dist(clean[0], clean[-1]) <= 1e-07:
                clean.pop()
            return Wire.ByVertices([Vertex.ByCoordinates(*p) for p in clean], close=True, tolerance=1e-07, silent=True)
        cleaned = Navigation._Shape(record).buffer(0).simplify(1e-06, preserve_topology=True)
        if cleaned.geom_type != 'Polygon' or cleaned.is_empty:
            raise ValueError('Expected one nonempty planar polygon.')
        geometry = Navigation._Surface(cleaned, record['plane'], record['storey'])
        face = Face.ByWires(wire(geometry['outer']), [wire(r) for r in geometry.get('holes', [])], tolerance=1e-07, silent=True)
        if face is None:
            raise ValueError('Cannot construct a planar navigation face.')
        data = {k: v for k, v in record.items() if k in ('id', 'storey', 'source', 'sourceIDs', 'kind') and v is not None}
        return Topology.SetDictionary(face, Dictionary.ByPythonDictionary(data)) if data else face

    @staticmethod
    def _Record(face, storey=None, source=None, tolerance=1e-05):
        import numpy as np
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        from topologicpy.Wire import Wire
        from topologicpy.Vertex import Vertex
        if not Topology.IsInstance(face, 'Face'):
            raise TypeError('Expected a Topologic Face.')
        from topologicpy.Edge import Edge

        def ring(w):
            segments = []
            for edge in Topology.Edges(w):
                if not Edge.IsLinear(edge, tolerance=tolerance, silent=True):
                    raise ValueError('Navigation requires linear face boundaries; tessellate curves first.')
                segments.append([Vertex.Coordinates(Edge.StartVertex(edge)), Vertex.Coordinates(Edge.EndVertex(edge))])
            if not segments:
                raise ValueError('Face boundary is empty.')
            points = list(segments.pop(0))
            while segments:
                candidates = [(math.dist(points[-1], point), i, end) for i, segment in enumerate(segments) for end, point in enumerate(segment)]
                distance, i, end = min(candidates)
                if distance > tolerance:
                    raise ValueError('Face boundary is disconnected.')
                segment = segments.pop(i)
                points.append(segment[1 - end])
            if math.dist(points[0], points[-1]) > tolerance:
                raise ValueError('Face boundary is open.')
            return points[:-1]
        outer = ring(Face.ExternalBoundary(face))
        points = np.asarray(outer, dtype=float)
        local = points - points[0]
        matrix = np.column_stack([local[:, :2], np.ones(len(points))])
        plane, _, rank, _ = np.linalg.lstsq(matrix, local[:, 2], rcond=None)
        if rank < 3 or np.max(np.abs(matrix @ plane - local[:, 2])) > tolerance:
            raise ValueError('Navigation requires planar faces with a nonzero XY projection.')
        plane[2] += points[0, 2] - plane[0] * points[0, 0] - plane[1] * points[0, 1]
        data = Navigation._Metadata(face)
        return {'outer': outer, 'holes': [ring(w) for w in Face.InternalBoundaries(face)], 'plane': plane.tolist(), 'storey': str(storey if storey is not None else data.get('storey', '0')), 'source': source if source is not None else data.get('source'), 'kind': data.get('kind', 'floor')}

    @staticmethod
    def ByFaces(faces, obstacles=None, connectors=None, storeys=None, storeyIDs=None, tolerance=1e-05):
        """Create a model from planar horizontal/sloping Faces.

        Obstacles are planar footprints at walking elevation. Solid IFC
        obstacles are sectioned by IFC.NavigationModel. storeyIDs, when given,
        must parallel faces. Otherwise face dictionary storey values are used,
        falling back to rounded face elevations. Units are unchanged.
        """
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        faces = list(faces)
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError('tolerance must be positive and finite.')
        if not faces:
            raise ValueError('At least one walkable face is required.')
        if storeyIDs is not None and len(storeyIDs) != len(faces):
            raise ValueError('storeyIDs must parallel faces.')
        result = []
        levels = {}
        for i, face in enumerate(faces):
            r = Navigation._Record(face, tolerance=tolerance)
            z = min((p[2] for p in r['outer']))
            data = Navigation._Metadata(face)
            sid = str(storeyIDs[i]) if storeyIDs is not None else str(data.get('storey', f'z:{z:.5f}'))
            levels[sid] = min(z, levels.get(sid, z))
            data.update(storey=sid, source=data.get('source', str(i)))
            result.append(Topology.SetDictionary(face, Dictionary.ByPythonDictionary(data)))
        assigned = copy.deepcopy(storeys) if storeys is not None else [{'id': k, 'name': f'Level {z:g}', 'elevation': z} for k, z in sorted(levels.items(), key=lambda q: q[1])]
        for level in assigned:
            level['id'] = str(level['id'])
        if not set(levels).issubset({s['id'] for s in assigned}):
            raise ValueError('Every face storey must exist in storeys.')
        assigned_obstacles = []
        for obstacle in obstacles or []:
            record = Navigation._Record(obstacle, tolerance=tolerance)
            metadata = Navigation._Metadata(obstacle)
            if 'storey' not in metadata:
                metadata['storey'] = min(assigned, key=lambda level: abs(level['elevation'] - record['plane'][2]))['id']
                obstacle = Topology.SetDictionary(obstacle, Dictionary.ByPythonDictionary(metadata))
            assigned_obstacles.append(obstacle)
        return NavigationModel(result, assigned_obstacles, list(connectors or []), assigned, {'unit': 'model'})

    @staticmethod
    def ByIFCPath(path, eyeHeight=1.6, maxSlope=45.0, obstacleTypes=None, excludeTypes=None):
        """Parse an IFC path once and extract a metre-based model."""
        from topologicpy.IFC import IFC
        from topologicpy.IFC import IFC
        model = IFC.ModelByPath(str(path), silent=True)
        if model is None:
            raise ValueError('Could not parse IFC file.')
        return IFC.NavigationModel(model, eyeHeight=eyeHeight, maxSlope=maxSlope, obstacleTypes=obstacleTypes, excludeTypes=excludeTypes)

    @staticmethod
    def _ByIFC(model, eyeHeight=1.6, maxSlope=45.0, obstacleTypes=None, excludeTypes=None):
        from topologicpy.IFC import IFC
        if not math.isfinite(eyeHeight) or eyeHeight < 0:
            raise ValueError('eyeHeight must be nonnegative and finite.')
        if not 0 <= maxSlope < 90:
            raise ValueError('maxSlope must be in [0,90).')
        data = IFC._NavigationExtract(model, eyeHeight=eyeHeight, maxSlope=maxSlope, obstacleTypes=obstacleTypes, excludeTypes=excludeTypes)
        from shapely import make_valid, set_precision
        omitted = []

        def convert(record):
            result = []
            for poly in Navigation._Polygons(set_precision(make_valid(Navigation._Shape(record)).buffer(0), 1e-05)):
                try:
                    result.append(Navigation._Face({**record, **Navigation._Surface(poly, record['plane'], record['storey'], record.get('kind', 'floor'), record.get('source'))}))
                except ValueError:
                    omitted.append(record.get('source'))
            return result
        connectors = []
        for c in data['connectors']:
            if c['type'] == 'ramp':
                connectors.append(Navigation.Ramp([f for s in c['surfaces'] for f in convert(s)], storey=c['storey'], source=c.get('source'), group=c['group']))
            else:
                connectors.append(Navigation.Stair(convert(c), storey=c['storey'], source=c.get('source'), group=c['group'], kind=c['kind']))
        from shapely.geometry import box
        for lift in data['lifts']:
            lo, hi = lift['bounds']
            shaft = box(lo[0], lo[1], hi[0], hi[1])
            doors = []
            cabins = []
            served = []
            for door in data['doors']:
                dl, dh = door['bounds']
                opening = box(dl[0], dl[1], dh[0], dh[1])
                if shaft.distance(opening) > 0.3:
                    continue
                candidates = [f for f in data['_walkable_faces'] if f['storey'] == door['storey'] and abs(f['plane'][2] - dl[2]) <= 0.35]
                if not candidates:
                    continue
                floor = min(candidates, key=lambda f: Navigation._Shape(f).distance(opening))
                dp = opening.centroid
                a, b, c = floor['plane']
                z = a * dp.x + b * dp.y + c
                doors.append([dp.x, dp.y, z])
                cabins.append([shaft.centroid.x, shaft.centroid.y, z])
                served.append(door['storey'])
            if len(doors) >= 2:
                connectors.append(Navigation.Lift(doors, cabins, served, source=lift['source']))
        result = NavigationModel([f for s in data['_walkable_faces'] for f in convert(s)], [f for s in data['obstacles'] for f in convert(s)], connectors, data['storeys'], {'unit': 'm', 'reader': 'TopologicPy.IFC', 'eyeHeight': eyeHeight, 'omittedSurfaces': omitted}, {k: data[k] for k in ('_obstacle_meshes', '_void_meshes')})
        result._ifc['_obstacle_faces'] = tuple((id(f) for f in result.obstacles))
        return result

    @staticmethod
    def Stair(stair, storey=None, source=None, group=None, kind='tread'):
        """Describe horizontal tread/landing Faces of a Stair topology or list.

        Extraction keeps horizontal upward surfaces from solids; explicitly
        supplied Faces are treated as walking faces regardless of winding.
        kind='landing' describes an explicit landing.
        """
        faces = Navigation._WalkingFaces(stair, horizontal=True)
        return {'type': 'stair', 'faces': faces, 'storey': storey, 'source': source, 'group': group, 'kind': kind}

    @staticmethod
    def Ramp(ramp=None, spacing=None, spacingRatio=0.15, centreline=None, storey=None, source=None, group=None):
        """Describe ramp top Faces, or an explicit ordered centreline.

        spacing is a positive absolute distance and overrides spacingRatio.
        Ratios refer to 3D centreline length. Surface extraction is best effort;
        an explicit centreline bypasses automatic elevation-slice inference.
        """
        if spacing is not None and (not math.isfinite(spacing) or spacing <= 0):
            raise ValueError('spacing must be positive and finite.')
        if not math.isfinite(spacingRatio) or not 0 < spacingRatio <= 1:
            raise ValueError('spacingRatio must be in (0,1].')
        return {'type': 'ramp', 'faces': Navigation._WalkingFaces(ramp) if ramp is not None else [], 'centreline': Navigation._Points(centreline) if centreline is not None else None, 'spacing': spacing, 'spacingRatio': spacingRatio, 'storey': storey, 'source': source, 'group': group}

    @staticmethod
    def Lift(doors, cabinCentres, storeys=None, source=None):
        """Describe a lift with one door and cabin point per served level.

        Inputs are equal-length ordered lists of XYZ coordinates or Vertices.
        Cabin points are joined in elevation order; each door joins its cabin.
        Only door points attach to floor grids. Lifts have no colour tiles.
        """
        doors = Navigation._Points(doors)
        cabins = Navigation._Points(cabinCentres)
        if len(doors) != len(cabins) or len(doors) < 2:
            raise ValueError('Supply matching door/cabin lists for at least two levels.')
        if storeys is not None and len(storeys) != len(doors):
            raise ValueError('storeys must parallel doors.')
        if any((abs(a[2] - b[2]) > 1e-05 for a, b in zip(doors, cabins))):
            raise ValueError('Each door and cabin must share an elevation.')
        return {'type': 'lift', 'doors': doors, 'cabins': cabins, 'group': source, 'storeys': list(storeys) if storeys is not None else None, 'source': source}

    @staticmethod
    def _Points(points):
        from topologicpy.Topology import Topology
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        if Topology.IsInstance(points, 'Wire'):
            from topologicpy.Edge import Edge
            segments = []
            for edge in Topology.Edges(points):
                if not Edge.IsLinear(edge, silent=True):
                    raise ValueError('Tessellate curved centreline Wires first.')
                segments.append([Vertex.Coordinates(Edge.StartVertex(edge)), Vertex.Coordinates(Edge.EndVertex(edge))])
            if not segments:
                raise ValueError('Centreline Wire is empty.')
            endpoints = [p for pair in segments for p in pair]
            starts = [p for p in endpoints if sum((math.dist(p, q) < 1e-07 for q in endpoints)) == 1]
            ordered = [starts[0] if starts else segments[0][0]]
            while segments:
                distance, i, end = min(((math.dist(ordered[-1], p), i, j) for i, pair in enumerate(segments) for j, p in enumerate(pair)))
                if distance > 1e-05:
                    raise ValueError('Centreline Wire is disconnected or branched.')
                ordered.append(segments.pop(i)[1 - end])
            points = ordered
        result = [Vertex.Coordinates(p) if Topology.IsInstance(p, 'Vertex') else list(p) for p in points]
        if any((len(p) != 3 or not all((math.isfinite(x) for x in p)) for p in result)):
            raise ValueError('Expected finite XYZ points.')
        return result

    @staticmethod
    def _WalkingFaces(topology, horizontal=False):
        from topologicpy.Topology import Topology
        from topologicpy.Face import Face
        explicit = isinstance(topology, (list, tuple)) or Topology.IsInstance(topology, 'Face')
        faces = list(topology) if isinstance(topology, (list, tuple)) else [topology] if Topology.IsInstance(topology, 'Face') else Topology.Faces(topology)
        result = []
        for face in faces or []:
            try:
                r = Navigation._Record(face)
            except ValueError:
                continue
            if horizontal and math.hypot(*r['plane'][:2]) > 0.01:
                continue
            if not explicit and Face.Normal(face)[2] <= 0:
                continue
            result.append(face)
        return result

    @staticmethod
    def _Data(model):
        if not isinstance(model, NavigationModel):
            raise TypeError('Expected NavigationModel.')
        faces = [Navigation._Record(f) for f in model.faces]
        data = {'name': 'Navigation', 'faces': faces, 'storeys': copy.deepcopy(model.storeys), 'obstacles': [Navigation._Record(f) for f in model.obstacles], 'connectors': [], 'explicit_connectors': []}
        for index, c in enumerate(model.connectors):
            if c['type'] == 'lift':
                n = len(c['doors'])
                order = sorted(range(n), key=lambda i: c['cabins'][i][2])
                data['explicit_connectors'].append({'kind': 'lift', 'points': c['doors'] + c['cabins'], 'edges': [(i, n + i) for i in range(n)] + [(n + a, n + b) for a, b in zip(order, order[1:])], 'attachments': list(range(n)), 'storeys': (c['storeys'] or [None] * n) * 2, 'source': c.get('source'), 'kinds': ['lift_door'] * n + ['lift'] * n, 'edgeKinds': ['lift_door'] * n + ['lift'] * (n - 1)})
                continue
            group = str(c.get('group') if c.get('group') is not None else f'connector:{index}')
            records = [Navigation._Record(f, c.get('storey'), c.get('source')) for f in c['faces']]
            if c['type'] == 'stair':
                data['connectors'].extend(({**r, 'type': 'stair', 'group': group, 'kind': 'landing' if r.get('kind') == 'landing' else c['kind']} for r in records))
            elif c.get('centreline') is not None:
                points = Navigation._SampleLine(c['centreline'], c.get('spacing'), c['spacingRatio'])
                data['explicit_connectors'].append({'kind': 'ramp', 'points': points, 'edges': [(i, i + 1) for i in range(len(points) - 1)], 'attachments': [0, len(points) - 1], 'storeys': [c.get('storey')] * len(points), 'group': group, 'source': c.get('source')})
            elif records:
                data['connectors'].append({'type': 'ramp', 'group': group, 'source': c.get('source'), 'storey': records[0]['storey'], 'surfaces': records, 'spacing': c['spacing'], 'spacingRatio': c['spacingRatio']})
        return data

    @staticmethod
    def _SampleLine(points, spacing, ratio):
        if len(points) < 2:
            raise ValueError('A centreline requires at least two points.')
        distances = [math.dist(a, b) for a, b in zip(points, points[1:])]
        total = sum(distances)
        if total <= 0:
            raise ValueError('Centreline length must be positive.')
        step = spacing if spacing is not None else total * ratio
        cumulative = [0.0]
        for d in distances:
            cumulative.append(cumulative[-1] + d)
        targets = sorted(set(cumulative + [min(i * step, total) for i in range(1, math.ceil(total / step))]))
        result = []
        i = 0
        for t in targets:
            while i < len(distances) - 1 and cumulative[i + 1] <= t:
                i += 1
            f = (t - cumulative[i]) / distances[i] if distances[i] else 0
            result.append([points[i][k] + f * (points[i + 1][k] - points[i][k]) for k in range(3)])
        return result

    @staticmethod
    def Graph(model, spacing=1.0, maxStepHeight=0.17, maxHorizontalGap=0.0, storey=None, mode='navigation', origin=None, direction=None, tolerance=1e-05, graphType='graph', maxStairRiser=0.35):
        """Return a standard Topologic graph with persistent tile associations.

        mode is 'navigation' (adjacent grid cells) or 'visibility'. Grid spacing
        is measured in XY, projected onto each surface. origin and direction
        define a shared XY frame. Horizontal gap bridging is unsupported;
        maxHorizontalGap must be zero. Storey restriction removes other levels
        before routing. Vertex dictionaries include navigationID/storey/kind/
        source/tileIDs; edges include Length/kind. Units match the model.
        """
        from topologicpy.IFC import IFC
        from topologicpy.Graph import Graph
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        if not math.isfinite(spacing) or spacing <= 0:
            raise ValueError('spacing must be positive and finite.')
        if not math.isfinite(maxStairRiser) or maxStairRiser <= 0:
            raise ValueError('maxStairRiser must be positive and finite.')
        if not math.isfinite(maxStepHeight) or maxStepHeight < 0:
            raise ValueError('maxStepHeight must be nonnegative and finite.')
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError('tolerance must be positive and finite.')
        if maxHorizontalGap != 0:
            raise ValueError('Horizontal gap bridging is unsupported; use explicit connectors.')
        if graphType not in ('graph', 'tgraph'):
            raise ValueError('graphType must be graph or tgraph.')
        if mode not in ('navigation', 'visibility'):
            raise ValueError('mode must be navigation or visibility.')
        origin, direction = Navigation._Frame(origin, direction)
        data = Navigation._Data(model)
        from shapely.ops import unary_union
        floors = []
        sections = {}

        def solid_sections(z):
            key = round(z, 5)
            if key not in sections:
                cuts = []
                for mesh in model._ifc['_obstacle_meshes']:
                    voids = unary_union([IFC._NavigationSection(v, z) for v in model._ifc.get('_void_meshes', {}).get(mesh['source_id'], [])])
                    cuts.append(IFC._NavigationSection(mesh, z).difference(voids))
                sections[key] = unary_union(cuts)
            return sections[key]
        for f in data['faces']:
            if Navigation._UseSolidObstacles(model):
                zs = [p[2] for p in f['outer']]
                blocked = unary_union([solid_sections(z + 0.1) for z in (min(zs), (min(zs) + max(zs)) / 2, max(zs))])
            else:
                blocked = unary_union([Navigation._Shape(o) for o in data['obstacles'] if o['storey'] == f['storey']])
            floors.extend((Navigation._Surface(p, f['plane'], f['storey'], source=f.get('source')) for p in Navigation._Polygons(Navigation._Shape(f).difference(blocked))))
        from collections import defaultdict
        groups = defaultdict(list)
        for f in floors:
            groups[f['storey'], tuple((round(v, 6) for v in f['plane']))].append(f)
        merged = []
        for (sid, plane), parts in groups.items():
            sources = sorted({str(f['source']) for f in parts if f.get('source') is not None})
            for poly in Navigation._Polygons(unary_union([Navigation._Shape(f) for f in parts])):
                merged.append({**Navigation._Surface(poly, plane, sid, source=sources[0] if sources else None), 'sourceIDs': sources})
        data['faces'] = merged
        entry, _ = Navigation._Build(data, spacing=spacing, storey=str(storey) if storey is not None else None, mode=mode, maxStepHeight=maxStepHeight, tolerance=tolerance, origin=origin, direction=direction, maxStairRiser=maxStairRiser)
        entry = Navigation._UniquePoints(entry, tolerance)
        tile_records = []
        tile_ids = {}
        for i, tile in enumerate(entry['tiles']):
            ident = f'tile:{i}'
            tile_ids.setdefault(tile['node'], []).append(ident)
            tile_records.append({**tile, 'id': ident, 'sourceIDs': entry['nodes'][tile['node']].get('sourceIDs', [])})
        vertex_data = []
        for i, n in enumerate(entry['nodes']):
            vertex_data.append({'navigationID': i, 'storey': n['storey'], 'storeyIDs': n.get('storeyIDs', [n['storey']]), 'kind': n['kind'], 'kinds': n.get('kinds', [n['kind']]), 'source': str(n.get('source') or ''), 'group': str(n.get('group') or ''), 'sourceIDs': n.get('sourceIDs', []), 'tileIDs': tile_ids.get(i, [])})
        edge_data = [{'Length': e['length'], 'kind': e['kind']} for e in entry['edges']]
        graph_metadata = {'navigationTiles': json.dumps(tile_records), 'navigationUnit': model.metadata.get('unit', 'model')}
        if graphType == 'tgraph':
            from topologicpy.TGraph import TGraph
            graph = TGraph()
            for point, metadata in zip([n['point'] for n in entry['nodes']], vertex_data):
                graph.AddVertex(dictionary={**metadata, **dict(zip(('x', 'y', 'z'), point))}, tolerance=tolerance, silent=True)
            for e, metadata in zip(entry['edges'], edge_data):
                graph.AddEdge(e['src'], e['dst'], directed=False, dictionary=metadata, silent=True)
            return graph.SetDictionary(graph_metadata)
        graph = Graph.ByMeshData([n['point'] for n in entry['nodes']], [[e['src'], e['dst']] for e in entry['edges']], vertexDictionaries=vertex_data, edgeDictionaries=edge_data, ontology=False, tolerance=tolerance)
        if graph is None:
            raise ValueError('Could not construct navigation graph.')
        return Graph.SetDictionary(graph, Dictionary.ByPythonDictionary(graph_metadata))

    @staticmethod
    def _UniquePoints(entry, tolerance):
        from itertools import product
        nodes = []
        buckets = {}
        indices = {}
        for old, node in enumerate(entry['nodes']):
            point = node['point']
            key = tuple((math.floor(v / tolerance) for v in point))
            candidates = [i for offset in product((-1, 0, 1), repeat=3) for i in buckets.get(tuple((k + d for k, d in zip(key, offset))), [])]
            existing = next((i for i in candidates if math.dist(nodes[i]['point'], point) <= tolerance), None)
            if existing is None:
                existing = len(nodes)
                nodes.append({**node, 'kinds': [node['kind']], 'storeyIDs': [node['storey']]})
                buckets.setdefault(key, []).append(existing)
            else:
                for field, values in [('kinds', [node['kind']]), ('storeyIDs', [node['storey']]), ('sourceIDs', node.get('sourceIDs', []))]:
                    nodes[existing][field] = list(dict.fromkeys(nodes[existing].get(field, []) + values))
            indices[old] = existing
        edges = []
        seen = set()
        for edge in entry['edges']:
            a, b = (indices[edge['src']], indices[edge['dst']])
            key = tuple(sorted((a, b)))
            if a == b or key in seen:
                continue
            seen.add(key)
            edges.append({**edge, 'src': a, 'dst': b, 'length': math.dist(nodes[a]['point'], nodes[b]['point'])})
        return {**entry, 'nodes': nodes, 'edges': edges, 'tiles': [{**tile, 'node': indices[tile['node']]} for tile in entry['tiles']]}

    @staticmethod
    def _Frame(origin, direction):
        origin = list(origin)[:2] if origin is not None else [0, 0]
        direction = list(direction)[:2] if direction is not None else [1, 0]
        if len(origin) != 2 or len(direction) != 2 or (not all((math.isfinite(v) for v in origin + direction))):
            raise ValueError('Expected finite XY origin and direction.')
        length = math.hypot(*direction)
        if length == 0:
            raise ValueError('direction must have nonzero XY length.')
        return (origin, [v / length for v in direction])

    @staticmethod
    def Tiles(model, graph, storey=None):
        """Return Topologic tile Faces with navigationID and tileID metadata.

        Associations are captured on the graph at construction time, so results
        remain stable if the source model is later edited. Lift nodes have none.
        """
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from topologicpy.TGraph import TGraph
        d = TGraph.Dictionary(graph) if isinstance(graph, TGraph) else Navigation._Metadata(graph)
        result = []
        for record in json.loads(d.get('navigationTiles', '[]')):
            if storey is not None and str(record['storey']) != str(storey):
                continue
            face = Navigation._Face(record)
            metadata = Navigation._Metadata(face)
            metadata.update(navigationID=record['node'], tileID=record['id'])
            result.append(Topology.SetDictionary(face, Dictionary.ByPythonDictionary(metadata)))
        return result

    @staticmethod
    def _UseSolidObstacles(model):
        return '_obstacle_meshes' in model._ifc and tuple((id(f) for f in model.obstacles)) == model._ifc.get('_obstacle_faces')

    @staticmethod
    def IsovistFaces(model, eyeHeight=None, storey=None):
        """Return horizontal Faces at each storey/walking elevation + eyeHeight.

        IFC solid obstacles are resectioned at the requested eye height.
        Generic planar obstacles are treated as vertical footprints.
        """
        from topologicpy.IFC import IFC
        from collections import defaultdict
        from shapely.ops import unary_union
        if eyeHeight is None:
            eyeHeight = model.metadata.get('eyeHeight', 1.6)
        if not math.isfinite(eyeHeight) or eyeHeight < 0:
            raise ValueError('eyeHeight must be nonnegative and finite.')
        data = Navigation._Data(model)
        levels = defaultdict(list)
        surfaces = data['faces'] + [c for c in data['connectors'] if c.get('kind') == 'landing']
        for f in surfaces:
            if storey is not None and f['storey'] != str(storey):
                continue
            p = Navigation._Shape(f).representative_point()
            z = round(Navigation._Height(f, p.x, p.y), 5)
            levels[f['storey'], z].append(Navigation._Shape(f))
        result = []
        for (sid, z), parts in levels.items():
            meshes = model._ifc.get('_obstacle_meshes') if Navigation._UseSolidObstacles(model) else None
            if meshes is not None:
                blocked = []
                for mesh in meshes:
                    voids = unary_union([IFC._NavigationSection(v, z + eyeHeight) for v in model._ifc.get('_void_meshes', {}).get(mesh['source_id'], [])])
                    blocked.append(IFC._NavigationSection(mesh, z + eyeHeight).difference(voids))
            else:
                blocked = [Navigation._Shape(o) for o in data['obstacles'] if o['storey'] == sid]
            area = unary_union(parts).difference(unary_union(blocked))
            result.extend((Navigation._Face(Navigation._Surface(p, [0, 0, z + eyeHeight], sid, 'isovist')) for p in Navigation._Polygons(area)))
        return result
    _NavigationTolerance = 1e-05

    @staticmethod
    def _Polygons(g):
        if g.is_empty:
            return []
        if g.geom_type == 'Polygon':
            return [g] if g.area > Navigation._NavigationTolerance * Navigation._NavigationTolerance else []
        return [p for part in getattr(g, 'geoms', []) for p in Navigation._Polygons(part)]

    @staticmethod
    def _Surface(poly, plane, storey, kind='floor', source=None):
        a, b, c = plane
        xyz = lambda ring: [[float(x), float(y), float(a * x + b * y + c)] for x, y in list(ring.coords)[:-1]]
        return {'outer': xyz(poly.exterior), 'holes': [xyz(r) for r in poly.interiors], 'plane': list(plane), 'storey': storey, 'kind': kind, 'source': source}

    @staticmethod
    def _Shape(s):
        from shapely.geometry import Polygon
        return Polygon([p[:2] for p in s['outer']], [[p[:2] for p in h] for h in s.get('holes', [])])

    @staticmethod
    def _Height(s, x, y):
        a, b, c = s.get('plane', [0, 0, s.get('elevation', 0)])
        return a * x + b * y + c

    @staticmethod
    def _CanonicalModel(model):
        import json
        data = model.model_dump() if hasattr(model, 'model_dump') else model
        data = json.loads(json.dumps(data))
        for s in data['faces']:
            z = s.get('elevation', 0)
            s['plane'] = s.get('plane') or [0, 0, z]
            s['outer'] = [p if len(p) == 3 else [*p, z] for p in s['outer']]
            s['holes'] = [[p if len(p) == 3 else [*p, z] for p in r] for r in s.get('holes', [])]
        if not data.get('storeys'):
            levels = sorted(set((round(s.get('elevation', 0), 4) for s in data['faces'])))
            data['storeys'] = [{'id': str(i), 'name': f'Level {z:g} m', 'elevation': z} for i, z in enumerate(levels)]
            for s in data['faces']:
                s['storey'] = str(levels.index(round(s.get('elevation', 0), 4)))
        return data

    @staticmethod
    def _RampSamples(connector, spacing=None, spacingRatio=0.15):
        """Infer an elevation-monotonic centreline and sample by distance or ratio."""
        import math
        import numpy as np
        from shapely.geometry import Polygon, LineString
        from shapely.ops import unary_union
        parts = connector['surfaces']
        zs = [p[2] for s in parts for p in s['outer']]
        lo, hi = (min(zs), max(zs))
        points = []
        landings = []
        if hi - lo < Navigation._NavigationTolerance:
            return []
        for s in parts:
            if abs(s['plane'][0]) + abs(s['plane'][1]) < Navigation._NavigationTolerance:
                p = Navigation._Shape(s).representative_point()
                landings.append(([p.x, p.y, Navigation._Height(s, p.x, p.y)], s))
        for z in np.linspace(lo + Navigation._NavigationTolerance, hi - Navigation._NavigationTolerance, 101):
            if any((abs(z - p[2]) < Navigation._NavigationTolerance for p, _ in landings)):
                continue
            lines = []
            for s in parts:
                if abs(s['plane'][0]) + abs(s['plane'][1]) < Navigation._NavigationTolerance:
                    continue
                rings = [s['outer'], *s.get('holes', [])]
                hits = []
                for ring in rings:
                    for p, q in zip(ring, ring[1:] + ring[:1]):
                        if p[2] <= z < q[2] or q[2] <= z < p[2]:
                            t = (z - p[2]) / (q[2] - p[2])
                            hits.append([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])])
                if len(hits) >= 2:
                    line = LineString(hits[:2]).intersection(Navigation._Shape(s))
                    if not line.is_empty:
                        lines.append(line)
            if not lines:
                continue
            joined = unary_union(lines)
            from shapely.ops import linemerge
            if joined.geom_type == 'MultiLineString':
                joined = linemerge(joined)
            candidates = [joined] if joined.geom_type == 'LineString' else [g for g in getattr(joined, 'geoms', []) if g.geom_type == 'LineString']
            if not candidates:
                continue
            if len(candidates) > 1:
                continue
            mid = candidates[0].interpolate(0.5, normalized=True)
            points.append([mid.x, mid.y, float(z)])
        if len(points) < 2:
            return []
        raw = [(p, None) for p in points] + landings
        raw.sort(key=lambda item: item[0][2])
        points = [p for p, _ in raw]
        points[0][2] = lo
        points[-1][2] = hi
        lengths = [math.dist(a, b) for a, b in zip(points, points[1:])]
        total = sum(lengths)
        if total < Navigation._NavigationTolerance:
            return []
        cumulative = np.r_[0, np.cumsum(lengths)]
        step = spacing if spacing is not None else spacingRatio * total
        targets = sorted(set([0, total, *np.arange(step, total, step), *[float(cumulative[i]) for i, (_, s) in enumerate(raw) if s is not None]]))
        sampled = []
        for t in targets:
            i = min(len(lengths) - 1, max(0, int(np.searchsorted(cumulative, t, side='right') - 1)))
            f = (t - cumulative[i]) / lengths[i] if lengths[i] > Navigation._NavigationTolerance else 0
            p = [points[i][k] + f * (points[i + 1][k] - points[i][k]) for k in range(3)]
            sampled.append(p)
        result = []
        for i, p in enumerate(sampled):
            low = lo if i == 0 else (sampled[i - 1][2] + p[2]) / 2
            high = hi if i == len(sampled) - 1 else (p[2] + sampled[i + 1][2]) / 2
            tiles = []
            for s in parts:
                poly = Navigation._Shape(s)
                a, b, c = s['plane']
                if abs(a) + abs(b) < Navigation._NavigationTolerance:
                    if abs(p[2] - c) < Navigation._NavigationTolerance:
                        tiles.append(s)
                    continue
                n = np.array([a, b])
                nn = float(np.linalg.norm(n))
                n /= nn
                u = np.array([-n[1], n[0]])
                radius = max(100, math.hypot(*[poly.bounds[j + 2] - poly.bounds[j] for j in range(2)]) * 4)
                centre = np.array(poly.centroid.coords[0])
                along = float(np.dot(centre, u))
                q0 = (low - c) / nn
                q1 = (high - c) / nn
                clip = Polygon([n * q0 + u * (along - radius), n * q1 + u * (along - radius), n * q1 + u * (along + radius), n * q0 + u * (along + radius)])
                tiles.extend((Navigation._Surface(q, s['plane'], s['storey'], 'ramp', s.get('source')) for q in Navigation._Polygons(poly.intersection(clip))))
            result.append((p, tiles))
        return result

    @staticmethod
    def _Build(model, spacing=4, mode='navigation', storey=None, maxStepHeight=0.17, tolerance=1e-05, origin=(0, 0), direction=(1, 0), maxStairRiser=0.35):
        import math
        import numpy as np
        from shapely.geometry import Polygon, Point, LineString, box
        from shapely.ops import unary_union
        from shapely.strtree import STRtree
        MAX_RISE = maxStepHeight
        EPS = tolerance
        m = Navigation._CanonicalModel(model)
        nodes = []
        edges = []
        seen = set()
        tiles = []
        footprints = []
        planes = []
        floors = [s for s in m['faces'] if storey is None or s['storey'] == storey]
        floor_union = {sid: unary_union([Navigation._Shape(s) for s in floors if s['storey'] == sid]) for sid in {s['storey'] for s in floors}}
        obstacle_by_level = {}

        def obstacle(sid, z):
            k = (sid, round(z, 3))
            if k not in obstacle_by_level:
                obstacle_by_level[k] = unary_union([Navigation._Shape(s) for s in m.get('obstacles', []) if s['storey'] == sid and abs(s['plane'][2] - z) <= MAX_RISE + EPS])
            return obstacle_by_level[k]

        def node(p, sid, kind='floor', display=None, poly=None, plane=None, source=None, group=None, sourceIDs=None):
            i = len(nodes)
            nodes.append({'point': list(map(float, p)), 'storey': sid, 'kind': kind, 'source': source, 'group': group, 'sourceIDs': sourceIDs or ([str(source)] if source is not None else [])})
            footprints.append(poly)
            planes.append(plane)
            for s in display or []:
                tiles.append({'node': i, **s})
            return i

        def edge(a, b, kind='floor'):
            if a == b:
                return
            k = tuple(sorted((a, b)))
            if k in seen:
                return
            d = math.dist(nodes[a]['point'], nodes[b]['point'])
            if d < EPS:
                return
            seen.add(k)
            edges.append({'src': a, 'dst': b, 'length': d, 'kind': kind})
        for s in floors:
            poly = Navigation._Shape(s)
            from shapely.affinity import affine_transform
            ux, uy = direction
            ox, oy = origin
            local = affine_transform(poly, [ux, uy, -uy, ux, -ux * ox - uy * oy, uy * ox - ux * oy])
            x0, y0, x1, y1 = local.bounds
            if (x1 - x0) * (y1 - y0) / spacing ** 2 > 200000:
                raise ValueError('Grid is too dense. Increase spacing.')
            for x in np.arange(math.floor(x0 / spacing) * spacing, x1, spacing):
                for y in np.arange(math.floor(y0 / spacing) * spacing, y1, spacing):
                    for local_tile in Navigation._Polygons(local.intersection(box(x, y, x + spacing, y + spacing))):
                        tile = affine_transform(local_tile, [ux, -uy, uy, ux, ox, oy])
                        pt = tile.centroid
                        if not tile.covers(pt):
                            pt = tile.representative_point()
                        node([pt.x, pt.y, Navigation._Height(s, pt.x, pt.y)], s['storey'], display=[Navigation._Surface(tile, s['plane'], s['storey'])], poly=tile, plane=s['plane'], source=s.get('source'), sourceIDs=s.get('sourceIDs'))
                        if len(nodes) > 10000:
                            raise ValueError('More than 10000 navigation points. Increase spacing.')
        floor_count = len(nodes)
        if not floor_count:
            raise ValueError('No walkable floor geometry was found.')
        tree = STRtree(footprints)
        for i, p in enumerate(footprints[:floor_count]):
            for j in tree.query(p.buffer(EPS)):
                j = int(j)
                if j <= i or nodes[i]['storey'] != nodes[j]['storey']:
                    continue
                q = footprints[j]
                common = p.boundary.intersection(q.boundary.buffer(EPS * 0.1))
                if common.length < EPS:
                    continue
                at = common.interpolate(0.5, normalized=True)
                a, b = (planes[i], planes[j])
                if abs((a[0] - b[0]) * at.x + (a[1] - b[1]) * at.y + a[2] - b[2]) > MAX_RISE + EPS:
                    continue
                segment = LineString([nodes[i]['point'][:2], nodes[j]['point'][:2]])
                if p.union(q).buffer(EPS * 0.1).covers(segment):
                    edge(i, j)
        if mode == 'visibility':
            for i in range(floor_count):
                p = nodes[i]['point']
                sid = nodes[i]['storey']
                for j in range(i + 1, floor_count):
                    q = nodes[j]['point']
                    if nodes[j]['storey'] != sid:
                        continue
                    line = LineString([p[:2], q[:2]])
                    if line.length < EPS:
                        continue
                    if not floor_union[sid].buffer(EPS * 0.1).covers(line):
                        continue
                    traversed = [int(k) for k in tree.query(line) if footprints[int(k)].intersection(line).length > EPS]
                    if not traversed:
                        continue
                    valid = True
                    intervals = []
                    for k in traversed:
                        part = footprints[k].intersection(line)
                        for geom in [part] if part.geom_type == 'LineString' else getattr(part, 'geoms', []):
                            if geom.geom_type != 'LineString':
                                continue
                            ends = [Point(geom.coords[0]), Point(geom.coords[-1])]
                            ts = sorted((line.project(e) / line.length for e in ends))
                            intervals.append((*ts, k))
                    intervals.sort()
                    for left, right in zip(intervals, intervals[1:]):
                        t = (left[1] + right[0]) / 2
                        r = line.interpolate(t, normalized=True)
                        a, b = (planes[left[2]], planes[right[2]])
                        if abs((a[0] - b[0]) * r.x + (a[1] - b[1]) * r.y + a[2] - b[2]) > MAX_RISE + EPS:
                            valid = False
                            break
                    if valid:
                        edge(i, j, 'visibility')
        connector_ids = []
        levels = sorted(m['storeys'], key=lambda s: s['elevation'])

        def level_for(p, preferred):
            near = [s for s in levels if s['elevation'] <= p[2] + MAX_RISE]
            return max(near, key=lambda s: s['elevation'])['id'] if near else preferred
        for c in m.get('connectors', []):
            if c['type'] == 'stair':
                poly = Navigation._Shape(c)
                pt = poly.centroid
                if not poly.covers(pt):
                    pt = poly.representative_point()
                p = [pt.x, pt.y, Navigation._Height(c, pt.x, pt.y)]
                sid = level_for(p, c['storey'])
                if storey is not None and sid != storey:
                    continue
                i = node(p, sid, c['kind'], [c], poly, c['plane'], c.get('source'), c['group'])
                connector_ids.append(i)
            else:
                previous = None
                for p, display in Navigation._RampSamples(c, c.get('spacing'), c.get('spacingRatio', 0.15)):
                    sid = level_for(p, c['storey'])
                    if storey is not None and sid != storey:
                        previous = None
                        continue
                    poly = unary_union([Navigation._Shape(s) for s in c['surfaces']])
                    i = node(p, sid, 'ramp', display, poly, None, c.get('source'), c['group'])
                    connector_ids.append(i)
                    if previous is not None:
                        line = LineString([nodes[previous]['point'][:2], p[:2]])
                        if poly.buffer(EPS * 2).covers(line):
                            edge(previous, i, 'ramp')
                    previous = i
        stairs = [i for i in connector_ids if nodes[i]['kind'] != 'ramp']
        if stairs:
            stair_tree = STRtree([footprints[i] for i in stairs])
            for local, i in enumerate(stairs):
                for k in stair_tree.query(footprints[i].buffer(EPS)):
                    j = stairs[int(k)]
                    if j <= i or nodes[i]['group'] != nodes[j]['group']:
                        continue
                    if abs(nodes[i]['point'][2] - nodes[j]['point'][2]) > maxStairRiser + EPS:
                        continue
                    if footprints[i].distance(footprints[j]) > EPS:
                        continue
                    if footprints[i].boundary.intersection(footprints[j].boundary.buffer(EPS)).length < EPS * 10:
                        continue
                    if footprints[i].union(footprints[j]).buffer(EPS).covers(LineString([nodes[i]['point'][:2], nodes[j]['point'][:2]])):
                        edge(i, j, 'stair')

        def attach(i, extra=None):
            p = nodes[i]['point']
            sid = nodes[i]['storey']
            support = unary_union([floor_union.get(sid, Polygon()), footprints[i] if footprints[i] is not None else Polygon(), extra if extra is not None else Polygon()])
            candidates = sorted((j for j in range(floor_count) if nodes[j]['storey'] == sid and abs(nodes[j]['point'][2] - p[2]) <= MAX_RISE + EPS), key=lambda j: math.dist(nodes[j]['point'], p))
            for j in candidates:
                line = LineString([p[:2], nodes[j]['point'][:2]])
                if not support.buffer(EPS).covers(line):
                    continue
                blocked = obstacle(sid, p[2])
                if not blocked.is_empty and line.intersection(blocked).length > EPS:
                    continue
                edge(i, j, 'attachment')
                return
        for i in connector_ids:
            if nodes[i]['kind'] != 'ramp':
                continue
            for j in stairs:
                if nodes[i]['group'] != nodes[j]['group']:
                    continue
                if abs(nodes[i]['point'][2] - nodes[j]['point'][2]) > MAX_RISE + EPS:
                    continue
                if footprints[i].distance(footprints[j]) > EPS:
                    continue
                line = LineString([nodes[i]['point'][:2], nodes[j]['point'][:2]])
                if footprints[i].union(footprints[j]).buffer(EPS).covers(line):
                    edge(i, j, 'ramp_landing')
        for i in connector_ids:
            attach(i)
        for c in m.get('explicit_connectors', []):
            ids = []
            for k, p in enumerate(c['points']):
                sid = c.get('storeys', [None] * len(c['points']))[k]
                if sid is None:
                    sid = level_for(p, levels[0]['id'])
                sid = str(sid)
                if storey is not None and sid != storey:
                    ids.append(None)
                    continue
                i = node(p, sid, c.get('kinds', [c['kind']] * len(c['points']))[k], source=c.get('source'), group=c.get('group'))
                ids.append(i)
                if k in c.get('attachments', []):
                    attach(i)
            for k, (a, b) in enumerate(c['edges']):
                if ids[a] is not None and ids[b] is not None:
                    edge(ids[a], ids[b], c.get('edgeKinds', [c['kind']] * len(c['edges']))[k])
        entry = {'model': m, 'nodes': nodes, 'edges': edges, 'tiles': tiles, 'floor_count': floor_count}
        return (entry, False)
