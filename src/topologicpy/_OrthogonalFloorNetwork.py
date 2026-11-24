"""Reconstruct a floor as a shared, axis-aligned network of rectangular atoms.

All boundary decisions are made on one planar grid, not separately per room.
Long original lines and exterior lines vote for the grid coordinates. Near
parallel source lines within a wall thickness become one supported grid line.
"""

from collections import defaultdict, deque
from math import hypot

from shapely import affinity, coverage_is_valid
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
from shapely.prepared import prep


def _parts(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == 'Polygon':
        return [geometry]
    return [p for g in getattr(geometry, 'geoms', ()) for p in _parts(g)]


def _building_axis(polygons):
    import math
    votes=[]
    for polygon in polygons:
        for ring in [polygon.exterior, *polygon.interiors]:
            for (x,y),(u,v) in zip(ring.coords,list(ring.coords)[1:]):
                distance=hypot(u-x,v-y)
                if distance>=.75:
                    votes.append((math.atan2(v-y,u-x),distance))
    if not votes:
        raise ValueError('Floor has no supported straight direction')
    return math.degrees(math.atan2(sum(w*math.sin(4*a) for a,w in votes),
                                  sum(w*math.cos(4*a) for a,w in votes))/4)


def _axis_lines(regions,domain,wall):
    votes=[[],[]]
    for polygon,exterior in [(p,False) for p in regions]+[(domain,True)]:
        for part in _parts(polygon):
            for ring in [part.exterior,*part.interiors]:
                for (x,y),(u,v) in zip(ring.coords,list(ring.coords)[1:]):
                    length=hypot(u-x,v-y)
                    if length<wall*.75:
                        continue
                    # Ignore short angled artefacts; no oblique partitions
                    # are constructed from them.
                    if abs(u-x)<=length*.0524:
                        votes[0].append(((x+u)/2,length*(3 if exterior else 1)))
                    elif abs(v-y)<=length*.0524:
                        votes[1].append(((y+v)/2,length*(3 if exterior else 1)))
    # Original exterior extrema constrain the envelope on every axis.
    for part in _parts(domain):
        minx,miny,maxx,maxy=part.bounds
        votes[0].extend(((minx,100.),(maxx,100.)))
        votes[1].extend(((miny,100.),(maxy,100.)))

    def cluster(items):
        groups=[]
        for coordinate,weight in sorted(items):
            if groups and coordinate-groups[-1][-1][0]<=wall and coordinate-groups[-1][0][0]<=wall*1.15:
                groups[-1].append((coordinate,weight))
            else:
                groups.append([(coordinate,weight)])
        anchors=[]
        for group in groups:
            total=sum(weight for _,weight in group)
            count=0.
            for coordinate,weight in group:
                count+=weight
                if count>=total/2:
                    anchors.append((coordinate,total))
                    break
        # After the first pass, representative lines may still be wall-close
        # when their original clusters touched at the support limit. Prefer
        # the stronger straight run, never a midpoint or a diagonal detour.
        while len(anchors)>2:
            pairs=[(anchors[j+1][0]-anchors[j][0],j)
                   for j in range(len(anchors)-1)
                   if anchors[j+1][0]-anchors[j][0]<=wall]
            if not pairs:break
            _,j=min(pairs)
            selected=max(anchors[j:j+2],key=lambda entry:entry[1])
            anchors[j:j+2]=[(selected[0],anchors[j][1]+anchors[j+1][1])]
        return sorted(set(round(value,7) for value,_ in anchors))
    return cluster(votes[0]),cluster(votes[1])


def _neighbors(cells,lines_x,lines_y):
    adjacency=defaultdict(list)
    for x,y in cells:
        for u,v,wall in ((x-1,y,lines_y[y+1]-lines_y[y]),
                         (x+1,y,lines_y[y+1]-lines_y[y]),
                         (x,y-1,lines_x[x+1]-lines_x[x]),
                         (x,y+1,lines_x[x+1]-lines_x[x])):
            if (u,v) in cells:
                adjacency[x,y].append(((u,v),wall))
    return adjacency


def _components(cells,adjacency):
    outstanding=set(cells)
    found=[]
    while outstanding:
        seed=outstanding.pop()
        component={seed}
        queue=deque([seed])
        while queue:
            current=queue.popleft()
            for neighbor,_ in adjacency[current]:
                if neighbor in outstanding:
                    outstanding.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        found.append(component)
    return found


def rectify_floor(polygons,wall=.35,minimum_fragment_area=.9,
                  maximum_domain_change=.025,minimum_usable_width=.9,
                  protected_voids=()):
    """Return hole-free, edge-matched planar regions and an audit dictionary.

    The floor footprint is inferred independently from input room polygons.
    Changes exceeding maximum_domain_change raise ValueError. Grid regions
    are split where needed to avoid face holes. Input need not have room names.
    """
    if not polygons or any(p.geom_type!='Polygon' or not p.is_valid for p in polygons):
        raise ValueError('Expected valid planar source polygons')
    angle=_building_axis(polygons)
    source=[affinity.rotate(p,-angle,origin=(0,0)) for p in polygons]
    total=unary_union(source)
    # Join only gaps reachable within half a wall thickness. The exterior
    # change budget below prevents this from closing a real courtyard.
    joined=total.buffer(wall/2,join_style=2).buffer(-wall/2,join_style=2)
    if joined.geom_type!='Polygon':
        raise ValueError('Prepare independent disconnected buildings separately')
    domain=Polygon(joined.exterior)
    protected=list(protected_voids)
    if any(void.intersection(domain.difference(total)).area>1e-5 for void in protected):
        raise ValueError('Floor includes a protected shaft or courtyard')
    prepared_change=domain.symmetric_difference(total).area
    if prepared_change>max(1.,total.area*maximum_domain_change):
        raise ValueError(f'Input floor has unresolved gaps or holes ({prepared_change:.3f} square metres)')
    xs,ys=_axis_lines(source,domain,wall)
    if len(xs)<2 or len(ys)<2:
        raise ValueError('Insufficient building-axis lines')
    prepared=prep(domain)
    tiles={}
    for x in range(len(xs)-1):
        for y in range(len(ys)-1):
            tile=box(xs[x],ys[y],xs[x+1],ys[y+1])
            if prepared.covers(tile.representative_point()):
                tiles[x,y]=tile
    if not tiles:
        raise ValueError('No cells in floor network')
    grid_domain=unary_union(list(tiles.values()))
    if any(void.intersection(grid_domain.difference(total)).area>1e-5 for void in protected):
        raise ValueError('Grid would cover a protected shaft or courtyard')
    if len(_parts(grid_domain))!=1 or grid_domain.interiors:
        raise ValueError('Grid would split the exterior or create a hole')
    change=domain.symmetric_difference(grid_domain).area
    if change+prepared_change>max(1.,domain.area*maximum_domain_change):
        raise ValueError(f'Grid changes the exterior by {change+prepared_change:.3f} square metres')

    neighbors=_neighbors(tiles,xs,ys)
    owners={}
    affinity_scores={}
    for key,tile in tiles.items():
        overlaps=[tile.intersection(region).area for region in source]
        owner=max(range(len(source)),key=lambda i:overlaps[i])
        owners[key]=owner
        affinity_scores[key]=overlaps

    # Small detached fragments and thin tendrils cross their longest shared
    # boundary. The outer envelope is independent of room ownership.
    reassigned=0
    for iteration in range(5):
        changed=0
        by_owner=defaultdict(set)
        for key,owner in owners.items():
            by_owner[owner].add(key)
        for owner,keys in by_owner.items():
            components=sorted(_components(keys,neighbors),key=lambda c:-sum(tiles[k].area for k in c))
            for component in components[1:]:
                if sum(tiles[k].area for k in component)>max(4.,minimum_fragment_area*4):
                    continue
                contact=defaultdict(float)
                for key in component:
                    for other,length in neighbors[key]:
                        if other not in component:
                            contact[owners[other]]+=length
                if contact:
                    target=max((v,k) for k,v in contact.items())[1]
                    for key in component:owners[key]=target
                    reassigned+=len(component)
                    changed+=len(component)
        # Each cell of a one-cell-wide tongue must have stronger contact with
        # its recipient than with the retained owner region. Retain entire
        # source rooms when their grid core is small but connected.
        for key,owner in list(owners.items()):
            same=sum(length for neighbor,length in neighbors[key] if owners[neighbor]==owner)
            contact=defaultdict(float)
            for neighbor,length in neighbors[key]:
                if owners[neighbor]!=owner:
                    contact[owners[neighbor]]+=length
            if not contact:continue
            target_length,target=max((length,other) for other,length in contact.items())
            if (tiles[key].area<minimum_fragment_area and
                    target_length>max(same,wall*1.5) and
                    affinity_scores[key][owner]<tiles[key].area*.75 and
                    sum(1 for v in owners.values() if v==owner)>1):
                owners[key]=target
                changed+=1
                reassigned+=1
        if not changed:break

    # A continuous room-width ribbon can survive all fragment tests because
    # its source was itself a narrow polygon. Transfer the whole ribbon to
    # its longest adjacent wall, retaining the exact floor envelope.
    ribbon_tiles=0
    for owner in sorted(set(owners.values())):
        keys={key for key,value in owners.items() if value==owner}
        if not keys or len(keys)==len(owners):continue
        bounds=unary_union([tiles[key] for key in keys]).bounds
        width=min(bounds[2]-bounds[0],bounds[3]-bounds[1])
        length=max(bounds[2]-bounds[0],bounds[3]-bounds[1])
        if width>=minimum_usable_width or length<3*width:
            continue
        contacts=defaultdict(float)
        for key in keys:
            for adjacent,shared in neighbors[key]:
                if owners[adjacent]!=owner:
                    contacts[owners[adjacent]]+=shared
        if contacts:
            target=max((shared,other) for other,shared in contacts.items())[1]
            for key in keys:owners[key]=target
            ribbon_tiles+=len(keys)

    # Dissolve the atoms against common coordinates. Splitting an annular
    # owner's disconnected components avoids publishing holed faces.
    by_owner=defaultdict(list)
    for key,owner in owners.items():
        by_owner[owner].append(tiles[key])
    regions=[]
    region_owners=[]
    for owner,blocks in by_owner.items():
        for part in _parts(unary_union(blocks)):
            if part.area>1e-7:
                regions.append(part)
                region_owners.append(owner)
    if any(p.interiors for p in regions):
        # A ring surrounding an independent room is split along grid lines
        # instead of creating a holed room face.
        from shapely.ops import split
        from shapely.geometry import LineString
        clean=[]
        clean_owners=[]
        for region,region_owner in zip(regions,region_owners):
            pending=[region]
            while pending:
                candidate=pending.pop()
                if not candidate.interiors:
                    clean.append(candidate)
                    clean_owners.append(region_owner)
                    continue
                hole=Polygon(candidate.interiors[0])
                mid=hole.representative_point()
                cuts=sorted([(abs(x-mid.x), 'x',x) for x in xs if hole.bounds[0]<x<hole.bounds[2]]+
                            [(abs(y-mid.y),'y',y) for y in ys if hole.bounds[1]<y<hole.bounds[3]])
                minx,miny,maxx,maxy=candidate.bounds
                for _,axis,coordinate in cuts:
                    line=(LineString([(coordinate,miny-1),(coordinate,maxy+1)]) if axis=='x'
                          else LineString([(minx-1,coordinate),(maxx+1,coordinate)]))
                    divided=_parts(split(candidate,line))
                    if len(divided)>=2 and sum(len(p.interiors) for p in divided)<len(candidate.interiors):
                        pending.extend(divided)
                        break
                else:
                    raise ValueError('Cannot split an enclosed room without a face hole')
        regions=clean
        region_owners=clean_owners
    if not coverage_is_valid(regions):
        raise ValueError('Rectilinear network does not share exact edges')
    if unary_union(regions).symmetric_difference(grid_domain).area>1e-6:
        raise ValueError('Rectilinear network left a gap')
    result=[affinity.rotate(p,angle,origin=(0,0)) for p in regions]
    if not coverage_is_valid(result):
        raise ValueError('Rotation broke shared boundaries')
    report={'axis_degrees':angle,'input_regions':len(polygons),
            'rectangular_atoms':len(tiles),'output_regions':len(result),
            'domain_area_m2':round(domain.area,3),
            'exterior_change_m2':round(change+prepared_change,3),
            'fragment_tiles_reassigned':reassigned,
            'ribbon_tiles_reassigned':ribbon_tiles,
            'axis_lines':(len(xs),len(ys)),
            'output_owners':region_owners}
    return result,report
