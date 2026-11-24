"""Conservative shared-network regularisation for simplified IFC floor bands."""

import math
from collections import defaultdict


def slice_floor_network(polygons, settings, domain=None):
    """The 2D network slice used by both the TPY command and IFC reconstruction.

    Returns the hole-free shared polygons and the native sliced Shell. A
    component that cannot be sliced consistently returns None.
    """
    from shapely.ops import unary_union

    if domain is None:
        domain = unary_union(polygons)
    if domain.geom_type != 'Polygon' or domain.interiors:
        return None
    return _regularize_component(polygons, domain, settings, return_shell=True)


def regularize(rooms, settings, protected_voids=()):
    """Return rooms whose shared 2D network is straightened before extrusion.

    Every change is proposed on connected edges. Invalid or unsupported floor
    components retain their original footprints. There is no proximity merge.
    """
    from shapely.geometry import Polygon, shape, mapping
    from shapely.ops import unary_union
    from shapely.errors import GEOSException

    grouped = defaultdict(list)
    level_tolerance = max(settings['tolerance'], 1e-5)
    for room in rooms:
        grouped[(round(room['bottom']/level_tolerance),
                 round(room['top']/level_tolerance), room.get('storey'),
                 bool(room.get('roof_space')))].append(room)
    output = []
    for members in grouped.values():
        polygons = [shape(room['footprint']) for room in members]
        if any(p.geom_type != 'Polygon' or not p.is_valid for p in polygons):
            output.extend(members)
            continue
        try:
            low, high = members[0]['bottom'], members[0]['top']
            protected = [footprint for bottom, top, footprint in protected_voids
                         if bottom < high - settings['tolerance'] and
                         top > low + settings['tolerance']]
            polygons = _close_enclosed_wall_gaps(polygons, settings, protected)
        except (ValueError, GEOSException):
            output.extend(members)
            continue
        # Rectify the entire floor from one weighted set of long wall lines.
        # This is deliberately all-or-nothing: if a protected void, exterior
        # area budget, native slice, or polygon coverage fails, retain the
        # established conservative network path below.
        try:
            from topologicpy._OrthogonalFloorNetwork import rectify_floor
            revised, report = rectify_floor(
                polygons, wall=settings.get('network_wall_thickness', .35),
                protected_voids=protected)
            common = unary_union(revised)
            if (len(revised) and common.geom_type == 'Polygon' and
                    not common.interiors and
                    slice_floor_network(revised, dict(settings,
                        network_contract=False,
                        network_straighten_exterior_jogs=False), common) is not None):
                counts=defaultdict(int)
                for polygon, owner in zip(revised, report['output_owners']):
                    source_room=members[owner]
                    counts[owner]+=1
                    room_id=source_room['id'] if counts[owner]==1 else (
                        f"{source_room['id']}:rectified{counts[owner]}")
                    output.append(dict(source_room,id=room_id,
                        bottom=low,top=high,
                        footprint=mapping(polygon),area=polygon.area))
                continue
        except (ImportError, ValueError, RuntimeError, GEOSException, ArithmeticError,
                TypeError, AttributeError, IndexError):
            pass
        if any(p.interiors for p in polygons):
            # A source room can legitimately enclose another room or a large
            # explicit courtyard. Do not convert it into overlapping cells.
            output.extend({**room, 'footprint': mapping(p), 'area': p.area}
                          for room, p in zip(members, polygons))
            continue
        try:
            domain = unary_union(polygons)
        except GEOSException:
            output.extend(members)
            continue
        components = [domain] if domain.geom_type == 'Polygon' else list(getattr(domain, 'geoms', []))
        if any(p.geom_type != 'Polygon' or p.interiors for p in components):
            output.extend({**room, 'footprint': mapping(p), 'area': p.area}
                          for room, p in zip(members, polygons))
            continue
        if len(members) < 2:
            output.extend({**room, 'footprint': mapping(p), 'area': p.area}
                          for room, p in zip(members, polygons))
            continue
        assigned = set()
        replacements = {}
        for component in components:
            indices = [i for i, polygon in enumerate(polygons)
                       if polygon.intersection(component).area > settings['tolerance'] ** 2]
            if any(i in assigned or not component.covers(polygons[i].representative_point())
                   for i in indices):
                replacements = None
                break
            assigned.update(indices)
            if len(indices) < 2:
                continue
            try:
                slice_result = slice_floor_network([polygons[i] for i in indices],
                                                   settings, component)
            except (ValueError, RuntimeError, GEOSException, ArithmeticError, TypeError,
                    AttributeError, IndexError):
                slice_result = None
            if slice_result is None:
                continue
            revised, _ = slice_result
            replacements.update(zip(indices, revised))
        if replacements is None:
            output.extend({**room, 'footprint': mapping(p), 'area': p.area}
                          for room, p in zip(members, polygons))
        else:
            for i, room in enumerate(members):
                polygon = replacements.get(i, polygons[i])
                output.append({**room, 'footprint': mapping(polygon), 'area': polygon.area})
    return output


def _close_enclosed_wall_gaps(polygons, settings, protected=()):
    """Fill only enclosed, wall-width holes, across every floor including roofs."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    wall = settings.get('network_wall_thickness', 0.35)
    tolerance = settings['tolerance']
    polygons = list(polygons)
    for _ in range(2):
        domain = unary_union(polygons)
        holes = [Polygon(ring) for component in
                 ([domain] if domain.geom_type == 'Polygon' else getattr(domain, 'geoms', ()))
                 if component.geom_type == 'Polygon' for ring in component.interiors]
        holes.extend(Polygon(ring) for polygon in polygons for ring in polygon.interiors)
        changed = False
        for hole in holes:
            if hole.is_empty or not hole.is_valid or hole.area <= tolerance**2:
                continue
            if any(void.intersection(hole).area > tolerance**2 for void in protected):
                continue
            # A room occupying this hole must remain its own cell.
            if any(p.intersection(hole).area > tolerance**2 for p in polygons):
                continue
            narrow = (hole.area < .1 or
                      (hole.minimum_clearance <= wall * 1.25 and
                       hole.area <= wall * hole.length / 2))
            if not narrow:
                continue
            ranked = sorted(((p.boundary.intersection(hole.boundary).length, i)
                             for i, p in enumerate(polygons)), reverse=True)
            if not ranked or ranked[0][0] <= tolerance:
                continue
            index = ranked[0][1]
            candidate = polygons[index].union(hole)
            if candidate.geom_type != 'Polygon' or not candidate.is_valid:
                continue
            polygons[index] = candidate
            changed = True
        if not changed:
            break
    return polygons


def _regularize_component(polygons, domain, settings, return_shell=False):
    from shapely import coverage_is_valid
    from shapely.geometry import LineString, Point, Polygon
    from shapely.ops import unary_union, polygonize
    from topologicpy.Cluster import Cluster
    from topologicpy.Core import Core
    from topologicpy.IFC import IFC
    from topologicpy.Topology import Topology
    from topologicpy.Vertex import Vertex
    from topologicpy.Face import Face
    from topologicpy.Wire import Wire

    if not coverage_is_valid(polygons):
        return None
    angular_tolerance = settings.get('network_angular_tolerance', 2.5)
    wall = settings.get('network_wall_thickness', 0.35)
    reach = wall * settings.get('network_junction_reach_factor', 1.25)
    if not 0 <= angular_tolerance < 45 or wall <= 0:
        raise ValueError('Invalid network direction or wall-thickness setting')
    def length(a, b):
        return math.dist(a, b)
    def direction(a, b, c, d):
        first = (b[0]-a[0], b[1]-a[1])
        second = (d[0]-c[0], d[1]-c[1])
        product = math.hypot(*first) * math.hypot(*second)
        if product < 1e-12:
            return 180.
        cosine = min(1., abs((first[0]*second[0]+first[1]*second[1])/product))
        return math.degrees(math.acos(cosine))
    def intersection(a, b, c, d):
        u = (b[0]-a[0], b[1]-a[1])
        v = (d[0]-c[0], d[1]-c[1])
        delta = (c[0]-a[0], c[1]-a[1])
        determinant = u[0]*v[1]-u[1]*v[0]
        if abs(determinant) < 1e-9:
            return None
        t = (delta[0]*v[1]-delta[1]*v[0])/determinant
        return (a[0]+t*u[0], a[1]+t*u[1])

    linework = unary_union([domain.boundary]+[p.boundary for p in polygons])
    paths = [linework] if linework.geom_type == 'LineString' else list(linework.geoms)
    segments = set()
    for line in paths:
        for a, b in zip(line.coords, list(line.coords)[1:]):
            if a != b:
                segments.add(tuple(sorted((tuple(a), tuple(b)))))
    neighbors = defaultdict(set)
    for a, b in segments:
        neighbors[a].add(b)
        neighbors[b].add(a)
    parent = {v: v for v in neighbors}
    boundary = list(domain.exterior.coords)[:-1]
    fixed = {p for p in neighbors if Point(p).distance(domain.boundary) < 1e-7}
    moved_outer = set()

    def root(v):
        if v not in parent:
            parent[v] = v
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v
    def valid_moves(moves):
        for a, b in segments:
            u, v = root(a), root(b)
            if u not in moves and v not in moves:
                continue
            if u in moves and v in moves and moves[u] == moves[v]:
                continue
            x, y = moves.get(u, u), moves.get(v, v)
            if x == y or direction(a, b, x, y) > angular_tolerance + 1e-8:
                return False
        return True
    def run(v, neighbor):
        visited = set()
        stack = [(v, neighbor)]
        total = 0.
        while stack:
            a, b = stack.pop()
            edge = tuple(sorted((a, b)))
            if edge in visited:
                continue
            visited.add(edge)
            total += length(a, b)
            for following in neighbors[b]:
                if (tuple(sorted((b, following))) not in visited and
                        direction(v, neighbor, b, following) <= angular_tolerance):
                    stack.append((b, following))
        return total
    def support(v):
        return max((run(v, n) for n in neighbors.get(v, ())), default=0.)

    # Find the longest supported exterior run spanning each short jog. Its
    # straight line, including both flanking vertices, controls the new point.
    for a, b in sorted(segments, key=lambda edge: length(*edge)):
        u, v = root(a), root(b)
        if not settings.get('network_straighten_exterior_jogs', True):
            break
        if (u == v or u not in fixed or v not in fixed or length(u, v) > reach or
                LineString((u, v)).distance(domain.boundary) > 1e-7):
            continue
        flank_u = [n for n in neighbors.get(u, set())-{v}
                   if LineString((u, n)).distance(domain.boundary) < 1e-7 and
                   length(u, n) > 1.]
        flank_v = [n for n in neighbors.get(v, set())-{u}
                   if LineString((v, n)).distance(domain.boundary) < 1e-7 and
                   length(v, n) > 1.]
        options = []
        for left in flank_u:
            for right in flank_v:
                if direction(u, left, v, right) > 12.:
                    continue
                # A real transverse partition supports this junction.
                for endpoint in (u, v):
                    for branch in neighbors.get(endpoint, set())-{u, v}:
                        if direction(left, right, endpoint, branch) < 20.:
                            continue
                        point = intersection(left, right, endpoint, branch)
                        if (point is None or max(length(u, point), length(v, point)) > reach or
                                direction(left, right, u, left) > angular_tolerance or
                                direction(left, right, v, right) > angular_tolerance):
                            continue
                        if valid_moves({u: point, v: point}):
                            options.append((length(u, left)+length(v, right)+run(endpoint, branch), point))
        if options:
            _, point = max(options)
            parent[point] = point
            parent[u] = parent[v] = point
            fixed.add(point)
            moved_outer.update((u, v))

    # For interior jogs, use the intersection of existing supported lines.
    for a, b in sorted(segments, key=lambda edge: length(*edge)):
        u, v = root(a), root(b)
        if not settings.get('network_contract', True):
            break
        if u == v or u in fixed or v in fixed or length(u, v) > reach:
            continue
        options = []
        for other_u in neighbors.get(u, set())-{v}:
            for other_v in neighbors.get(v, set())-{u}:
                if (direction(u, v, u, other_u) <= angular_tolerance or
                        direction(u, v, v, other_v) <= angular_tolerance or
                        direction(u, other_u, v, other_v) < 20.):
                    continue
                point = intersection(u, other_u, v, other_v)
                if (point is not None and max(length(u, point),length(v, point)) <= reach and
                        valid_moves({u: point, v: point})):
                    options.append((run(u, other_u)+run(v, other_v),point))
        if options:
            _, point = max(options)
            parent[point] = point
            parent[u] = parent[v] = point

    # Contract only actual shared-network edges. The longer straight run wins;
    # fixed exterior vertices cannot be contracted by this pass.
    for a, b in sorted(segments, key=lambda edge: length(*edge)):
        u, v = root(a), root(b)
        if not settings.get('network_contract', True):
            break
        if u == v or length(u, v) > wall:
            continue
        choices = []
        for winner, loser in ((u, v), (v, u)):
            if loser not in fixed and valid_moves({loser: winner}):
                choices.append((winner in fixed, support(winner), winner, loser))
        if choices:
            _, _, winner, loser = max(choices)
            parent[loser] = winner

    outer_points = [root(p) for p in boundary]
    outer_points = [p for i, p in enumerate(outer_points) if p != outer_points[i-1]]
    outer = Polygon(outer_points)
    flank_length = sum(length(p, n) for p in moved_outer for n in neighbors[p]
                       if n not in moved_outer and
                       LineString((p, n)).distance(domain.boundary) < 1e-7)
    if (not outer.is_valid or outer.is_empty or
            outer.symmetric_difference(domain).area >
            (reach*flank_length/2 if moved_outer else 1e-8) or
            outer.boundary.hausdorff_distance(domain.boundary) >
            (reach+1e-8 if moved_outer else 1e-8)):
        return None

    proposals = []
    for polygon in polygons:
        ring = [root(tuple(p)) for p in list(polygon.exterior.coords)[:-1]]
        ring = [p for i, p in enumerate(ring) if p != ring[i-1]]
        if len(ring) < 3:
            return None
        proposed = Polygon(ring)
        if not proposed.is_valid or proposed.is_empty:
            return None
        proposals.append(proposed)
    network = unary_union([outer.boundary]+[p.boundary for p in proposals])
    atoms = [p for p in polygonize(network)
             if p.area > settings['tolerance']**2 and outer.covers(p.representative_point())]
    if not atoms or outer.symmetric_difference(unary_union(atoms)).area > 1e-6:
        return None
    owned = defaultdict(list)
    for atom in atoms:
        overlaps = [atom.intersection(p).area for p in proposals]
        owner = (max(range(len(polygons)), key=lambda i: overlaps[i]) if max(overlaps)>atom.area*.05
                 else max(range(len(polygons)), key=lambda i:
                          atom.boundary.intersection(proposals[i].boundary).length))
        owned[owner].append(atom)
    if len(owned) != len(polygons):
        return None
    regions = [unary_union(owned[i]) for i in range(len(polygons))]
    if any(p.geom_type != 'Polygon' or p.interiors or not p.is_valid for p in regions):
        return None

    # Slice one common exterior Face with closed shared partition wires.
    # Native slicing can insert T vertices: clean collinear degree-two points
    # from the WHOLE resulting network after converting its faces to polygons.
    def wire(p):
        return IFC._Analytical_safe_wire([(x, y, 0.) for x, y in
                                          list(p.exterior.coords)[:-1]], tolerance=1e-7)
    exterior_wire = wire(outer)
    if exterior_wire is None:
        return None
    face = Core.Face.ByExternalInternalBoundaries(exterior_wire, [], 1e-7)
    tool = Cluster.ByTopologies([wire(p) for p in regions])
    if face is None or tool is None:
        return None
    sliced = Topology.Slice(face, tool, tolerance=1e-5, silent=True)
    if not sliced or not Topology.IsInstance(sliced, 'Shell'):
        return None
    def xy(w):
        return [tuple(Vertex.Coordinates(v)[:2]) for v in Wire.Vertices(w)]
    pieces = [Polygon(xy(Face.ExternalBoundary(f)),
                      [xy(w) for w in Face.InternalBoundaries(f)])
              for f in Topology.Faces(sliced)]
    if (not pieces or not coverage_is_valid(pieces) or
            any(not p.is_valid or p.interiors for p in pieces) or
            unary_union(pieces).symmetric_difference(outer).area > 1e-6):
        return None
    divided = defaultdict(list)
    for piece in pieces:
        scores = [piece.intersection(region).area for region in regions]
        divided[max(range(len(regions)), key=lambda i: scores[i])].append(piece)
    if len(divided) != len(regions):
        return None
    result = [unary_union(divided[i]) for i in range(len(regions))]
    graph = defaultdict(set)
    edges = unary_union([p.boundary for p in result])
    for path in ([edges] if edges.geom_type == 'LineString' else edges.geoms):
        for a, b in zip(path.coords, list(path.coords)[1:]):
            if a != b:
                graph[a].add(b)
                graph[b].add(a)
    redundant = set()
    for vertex, connected in graph.items():
        if len(connected) == 2:
            a, b = tuple(connected)
            cross = ((a[0]-vertex[0])*(b[1]-vertex[1])-
                     (a[1]-vertex[1])*(b[0]-vertex[0]))
            if abs(cross) <= 1e-8*length(a,vertex)*length(b,vertex):
                redundant.add(vertex)
    cleaned = [Polygon([p for p in list(region.exterior.coords)[:-1] if p not in redundant])
               for region in result]
    if (any(not p.is_valid or p.is_empty or p.interiors for p in cleaned) or
            not coverage_is_valid(cleaned) or
            unary_union(cleaned).symmetric_difference(outer).area > 1e-6):
        return None
    return (cleaned, sliced) if return_shell else cleaned
