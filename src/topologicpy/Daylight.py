# Copyright (C) 2026 Wassim Jabi
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Direct daylight geometry and preliminary Waldram studies.

Factors are percentages of unobstructed external horizontal illuminance.
Lengths use model units. No reflections, glazing losses or legal classification.
"""
import hashlib
import math

class Daylight:
    """Static direct-sky API; prepared records are immutable-by-convention snapshots."""

    @staticmethod
    def _Positive(value, name, zero=False):
        if isinstance(value, bool) or not math.isfinite(float(value)) or (value < 0 if zero else value <= 0):
            raise ValueError(name + ' must be finite and ' + ('nonnegative' if zero else 'positive'))
        return float(value)

    @staticmethod
    def _XYZ(vertices):
        import numpy as np
        from topologicpy.Vertex import Vertex
        from topologicpy.Topology import Topology
        a = np.asarray([Vertex.Coordinates(v) if Topology.IsInstance(v, 'Vertex') else v for v in vertices], dtype=float)
        if a.ndim != 2 or a.shape[1] != 3 or (not len(a)) or (not np.isfinite(a).all()):
            raise ValueError('Expected nonempty finite vertices/XYZ coordinates')
        return a

    @staticmethod
    def _Faces(topology):
        from topologicpy.Topology import Topology
        if topology is None:
            return []
        return [topology] if Topology.IsInstance(topology, 'Face') else Topology.Faces(topology, silent=True) or []

    @staticmethod
    def _GeometryKey(topology):
        """Canonical boundary identity for the supported planar polygon geometry.

        BREP text is unsuitable: native preparation may change p-curves or mesh
        caches without changing the boundary. Ignore ring starts/winding, hole
        order and Face enumeration. Round coordinates to nine decimal places
        (model units); keep aperture geometry separate from the room envelope.
        """
        from topologicpy.Face import Face
        from topologicpy.Wire import Wire
        from topologicpy.Vertex import Vertex

        def ring_key(wire):
            points = tuple(tuple(0.0 if round(float(x), 9) == 0 else round(float(x), 9) for x in Vertex.Coordinates(v))
                           for v in Wire.Vertices(wire))
            if not points:
                raise ValueError('Cannot fingerprint an empty boundary')
            candidates = []
            for sequence in (points, tuple(reversed(points))):
                smallest = min(sequence)
                candidates.extend(sequence[i:] + sequence[:i]
                                  for i, point in enumerate(sequence) if point == smallest)
            return min(candidates)

        return tuple(sorted((ring_key(Face.ExternalBoundary(face)),
                             tuple(sorted(ring_key(w) for w in Face.InternalBoundaries(face) or [])))
                            for face in Daylight._Faces(topology)))

    @staticmethod
    def _Resolution(resolution):
        levels = ('low', 'medium', 'high', 'very_high')
        if resolution not in levels:
            raise ValueError('Unknown resolution')
        return levels.index(resolution)

    @staticmethod
    def _Cancelled(cancelled):
        if cancelled is not None and cancelled():
            raise InterruptedError('Daylight calculation cancelled')

    @staticmethod
    def _PlanarPolygon(face, tolerance):
        import numpy as np
        from topologicpy.Topology import Topology
        from topologicpy.Face import Face
        from topologicpy.Edge import Edge
        if not Topology.IsInstance(face, 'Face'):
            raise ValueError('Expected a Face')
        xyz = Daylight._XYZ(Topology.Vertices(face))
        normal = np.asarray(Face.Normal(face, mantissa=12))
        if not Face.IsPlanar(face, tolerance=tolerance) or np.max(np.abs((xyz - xyz[0]) @ normal)) > tolerance:
            raise ValueError('Nonplanar Face unsupported')
        for e in Topology.Edges(face):
            ends = Daylight._XYZ(Topology.Vertices(e))
            if len(ends) != 2 or abs(Edge.Length(e, mantissa=12) - float(np.linalg.norm(ends[1] - ends[0]))) > tolerance:
                raise ValueError('Only straight polygon boundaries supported')

    @staticmethod
    def PrepareRoom(cell, *, tolerance=0.0001):
        """Validate Cell/windows and cut actual analytical openings.

        Reads Faces attached by Topology.AddApertures to Cell or its Faces.
        Windows must be contained in one coplanar host. Overlaps are unioned.
        Only planar straight-edged geometry supported. Does not mutate input.
        """
        from topologicpy.Topology import Topology
        from topologicpy.Face import Face
        tol = Daylight._Positive(tolerance, 'tolerance')
        if not Topology.IsInstance(cell, 'Cell'):
            raise ValueError('Expected a Cell')
        walls = Topology.Faces(cell)
        for f in walls:
            Daylight._PlanarPolygon(f, tol)
        groups = [[] for _ in walls]
        for w in Topology.Apertures(cell, subTopologyType='all', silent=True) or []:
            Daylight._PlanarPolygon(w, tol)
            area = Face.Area(w, mantissa=12)
            candidates = []
            for i, wall in enumerate(walls):
                if not Face.IsCoplanar(w, wall, tolerance=tol):
                    continue
                common = Topology.Intersect(w, wall, tolerance=tol, silent=True)
                covered = sum((Face.Area(f, mantissa=12) for f in Daylight._Faces(common)))
                if abs(covered - area) <= max(tol * tol, area * 1e-07):
                    candidates.append(i)
            if len(candidates) != 1:
                raise ValueError('Aperture needs exactly one containing coplanar host')
            groups[candidates[0]].append(w)
        opaque, apertures = ([], [])
        for wall, group in zip(walls, groups):
            merged = None
            for w in group:
                merged = w if merged is None else Topology.Union(merged, w, tolerance=tol, silent=True)
                if merged is None:
                    raise ValueError('Aperture union failed')
            if merged is None:
                opaque.append(wall)
                continue
            apertures.extend(Daylight._Faces(merged))
            parts = Daylight._Faces(Topology.Difference(wall, merged, tolerance=tol, silent=True))
            expected = Face.Area(wall, mantissa=12) - sum((Face.Area(f, mantissa=12) for f in Daylight._Faces(merged)))
            actual = sum((Face.Area(f, mantissa=12) for f in parts))
            if abs(actual - expected) > max(tol * tol * 10, abs(expected) * 1e-06):
                raise ValueError('Aperture cut failed area check')
            opaque.extend(parts)
        xyz = Daylight._XYZ(Topology.Vertices(cell))
        return dict(
            kind='DaylightRoom',
            cell=cell,
            opaqueFaces=opaque,
            apertures=apertures,
            tolerance=tol,
            bounds=[xyz.min(axis=0).tolist(), xyz.max(axis=0).tolist()],
            warnings=[] if apertures else ['Room has no attached apertures']
        )

    @staticmethod
    def WorkingPlane(cell, height=0.85, *, referenceElevation=None, tolerance=0.0001):
        """Horizontal room section Faces. Default reference is lowest vertex Z.

        Set referenceElevation explicitly for split levels/sloping floors.
        Plane must be strictly inside the room height.
        """
        from topologicpy.Topology import Topology
        from topologicpy.Face import Face
        from topologicpy.Vertex import Vertex
        tol = Daylight._Positive(tolerance, 'tolerance')
        height = Daylight._Positive(height, 'height', zero=True)
        if not Topology.IsInstance(cell, 'Cell'):
            raise ValueError('Expected a Cell')
        xyz = Daylight._XYZ(Topology.Vertices(cell))
        lo, hi = (xyz.min(axis=0), xyz.max(axis=0))
        z = (lo[2] if referenceElevation is None else float(referenceElevation)) + height
        if not math.isfinite(z) or not lo[2] + tol < z < hi[2] - tol:
            raise ValueError('Working plane outside room height')
        plane = Face.Rectangle(Vertex.ByCoordinates((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, z), width=float(hi[0] - lo[0] + 2), length=float(hi[1] - lo[1] + 2))
        faces = Daylight._Faces(Topology.Intersect(cell, plane, tolerance=tol, silent=True))
        if not faces:
            raise ValueError('Room section failed')
        return faces

    @staticmethod
    def SensorGrid(faces, spacing=0.3, *, origin=None, rotation=0, tolerance=0.0001):
        """Clipped horizontal grid; sensors lie inside their real area-weighted cells.

        Source Faces must be coplanar and nonoverlapping. IDs fingerprint geometry.
        """
        from topologicpy.Grid import Grid
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        import numpy as np
        tol = Daylight._Positive(tolerance, 'tolerance')
        spacing = Daylight._Positive(spacing, 'spacing')
        faces = [faces] if Topology.IsInstance(faces, 'Face') else list(faces)
        if not faces:
            raise ValueError('Empty grid domain')
        cells, vertices, areas = ([], [], [])
        elevation = None
        for i, f in enumerate(faces):
            Daylight._PlanarPolygon(f, tol)
            xyz = Daylight._XYZ(Topology.Vertices(f))
            if np.ptp(xyz[:, 2]) > tol or (elevation is not None and abs(xyz[0, 2] - elevation) > tol):
                raise ValueError('Grid domain must share a horizontal plane')
            elevation = xyz[0, 2]
            for prev in faces[:i]:
                common = Topology.Intersect(f, prev, tolerance=tol, silent=True)
                if sum((Face.Area(x, mantissa=12) for x in Daylight._Faces(common))) > tol * tol:
                    raise ValueError('Grid domains overlap')
            parts = Grid.OnFace(f, spacing=spacing, gridType='rectangular', output='faces', origin=origin, rotation=rotation, tolerance=tol, silent=True)
            if not parts:
                raise ValueError('Grid partition failed')
            for part in parts:
                area = Face.Area(part, mantissa=12)
                v = Face.InternalVertex(part, tolerance=tol, silent=True)
                if area <= 0 or not Topology.IsInstance(v, 'Vertex'):
                    raise ValueError('Invalid grid cell')
                cells.append(part)
                vertices.append(v)
                areas.append(area)
        expected = sum((Face.Area(f, mantissa=12) for f in faces))
        if abs(sum(areas) - expected) > max(tol * tol * 10, expected * 1e-06):
            raise ValueError('Grid area not conserved')
        payload = repr((Daylight._XYZ(vertices).round(9).tolist(), [round(a, 12) for a in areas], [Daylight._GeometryKey(f) for f in cells]))
        return dict(
            kind='DaylightGrid',
            faces=faces,
            cells=cells,
            vertices=vertices,
            coordinates=Daylight._XYZ(vertices).tolist(),
            areas=areas,
            area=sum(areas),
            gridId=hashlib.sha256(payload.encode()).hexdigest(),
            spacing=spacing,
            rotation=rotation,
            tolerance=tol
        )

    @staticmethod
    def PrepareScene(topologies, *, tolerance=0.0001, labels=None):
        """Prepare opaque two-sided geometry. Prepared rooms retain window holes.

        Raw Cells remain opaque. Rebuild snapshot after edits. Display properties
        have no analytical effect. Curved geometry is rejected in this version.
        """
        from topologicpy.BVH import BVH
        from topologicpy.Topology import Topology
        tol = Daylight._Positive(tolerance, 'tolerance')
        if isinstance(topologies, dict) or Topology.IsInstance(topologies, 'Topology'):
            topologies = [topologies]
        topologies = list(topologies)
        if labels is None: labels = ['Existing']*len(topologies)
        if len(labels)!=len(topologies) or any(not isinstance(label,str) for label in labels):
            raise ValueError('One string label per scene topology is required')
        faces = []; face_labels = []
        for t,label in zip(topologies,labels):
            start_count = len(faces)
            if isinstance(t, dict) and t.get('kind') == 'DaylightRoom':
                faces.extend(t['opaqueFaces'])
            elif Topology.IsInstance(t, 'Topology'):
                items = Daylight._Faces(t)
                if not items:
                    raise ValueError('Scene topology contains no opaque Faces')
                faces.extend(items)
            else:
                raise ValueError('Expected topologies or prepared room records')
            face_labels.extend([label]*(len(faces)-start_count))
        for f in faces:
            Daylight._PlanarPolygon(f, tol)
        bvh = BVH.PrepareScene(faces, tolerance=tol)
        identity = hashlib.sha256(repr(sorted(Daylight._GeometryKey(f) for f in faces)).encode()).hexdigest()
        return dict(kind='DaylightScene', bvh=bvh, sceneId=identity, tolerance=tol, faceCount=len(faces), faceLabels=face_labels, _context=list(zip(topologies,labels)))

    @staticmethod
    def SkyComponent(vertices, normals, scene, *, sky='uniform', resolution='medium', cancelled=None):
        """Direct sky component (%) for arbitrary normals, not total daylight factor.

        Uniform or CIE overcast relative luminance (1+2*sin(altitude))/3.
        Denominator: unobstructed external horizontal illuminance under same sky.
        Whole-sky midpoint quadrature may miss narrow windows; use Waldram there.
        """
        import numpy as np
        from topologicpy.Sun import Sun
        if sky not in ('uniform', 'cie_overcast'):
            raise ValueError('Unknown sky')
        level = Daylight._Resolution(resolution)
        xyz = Daylight._XYZ(vertices)
        normals = np.asarray(normals, dtype=float)
        if normals.shape == (3,):
            normals = np.tile(normals, (len(xyz), 1))
        if normals.shape != xyz.shape or not np.isfinite(normals).all() or any(np.linalg.norm(normals, axis=1) == 0):
            raise ValueError('Expected nonzero finite normals')
        normals = normals / np.linalg.norm(normals, axis=1)[:, None]
        result = Sun.SkyVisibility(xyz, normals, scene['bvh'], azimuthDivisions=72 * 2 ** level, altitudeDivisions=24 * 2 ** level, cancelled=cancelled)
        mu = np.asarray([p['toSun'][2] for p in result['patches']])
        lum = np.ones_like(mu) if sky == 'uniform' else (1 + 2 * mu) / 3
        denom = float(np.sum(mu * lum * np.asarray(result['solidAngles'])))
        values = 100 * np.sum((result['states'] == 0) * result['projectedWeights'] * math.pi * lum, axis=1) / denom
        return dict(
            kind='DaylightSkyComponent',
            values=values.tolist(),
            unit='percent',
            sky=sky,
            resolution=resolution,
            sceneId=scene['sceneId'],
            coordinates=xyz.tolist(),
            method='hemisphere-midpoint',
            denominator='unobstructed external horizontal illuminance',
            warnings=['Direct sky only; refine angular resolution']
        )

    @staticmethod
    def SkyFactor(vertices, scene, *, resolution='medium', cancelled=None):
        """Horizontal uniform-sky component (%). See SkyComponent limitations."""
        return Daylight.SkyComponent(vertices, [0, 0, 1], scene, resolution=resolution, cancelled=cancelled)

    @staticmethod
    def _Triangles(faces, depth, cancelled=None):
        import numpy as np
        from topologicpy.Topology import Topology
        triangles = []
        for face in faces:
            Daylight._Cancelled(cancelled)
            mesh = Topology.Tessellate(face, weld=False, mantissa=12, silent=True)
            if not mesh or not mesh.get('faces'):
                raise ValueError('Aperture tessellation failed')
            xyz = np.asarray(mesh['vertices'], dtype=float)
            for ids in mesh['faces']:
                if len(ids) != 3:
                    raise ValueError('Nontriangular tessellation')
                triangles.append(xyz[ids])
        a = np.asarray(triangles, dtype=float).reshape(-1, 3, 3)
        for _ in range(depth):
            Daylight._Cancelled(cancelled)
            if len(a) * 4 > 500000:
                raise ValueError('Aperture quadrature exceeds 500000 triangles')
            x, y, z = (a[:, 0], a[:, 1], a[:, 2])
            xy, yz, zx = ((x + y) / 2, (y + z) / 2, (z + x) / 2)
            a = np.concatenate([np.stack(t, axis=1) for t in ((x, xy, zx), (xy, y, yz), (zx, yz, z), (xy, yz, zx))])
        return a

    @staticmethod
    def _Integrate(vertices, triangles, scene, cancelled=None, retain=False):
        import numpy as np
        from topologicpy.BVH import BVH
        xyz = Daylight._XYZ(vertices)
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        areas = np.linalg.norm(cross, axis=1) / 2
        normals = cross / np.maximum(2 * areas[:, None], 1e-300)
        centres = triangles.mean(axis=1)
        aperture_scene = BVH.ByTriangles(triangles, tolerance=scene['tolerance'])
        values, details = ([], [])
        for p in xyz:
            Daylight._Cancelled(cancelled)
            delta = centres - p
            r = np.linalg.norm(delta, axis=1)
            if any(r <= scene['tolerance']):
                raise ValueError('Sensor too close to aperture')
            d = delta / np.maximum(r[:, None], 1e-300)
            weights = 100 * areas * np.abs(np.sum(normals * d, axis=1)) * np.maximum(d[:, 2], 0) / (math.pi * np.maximum(r * r, 1e-300))
            blocked = BVH.IsOccluded(scene['bvh'], p, d, minDistance=scene['tolerance'], cancelled=cancelled) if len(d) else np.zeros(0, dtype=bool)
            if len(d):
                first = BVH.RayHits(aperture_scene, p, d, minDistance=scene['tolerance'], cancelled=cancelled)
                duplicate = np.asarray([hit is not None and hit['distance'] < distance - scene['tolerance'] for hit, distance in zip(first, r)])
                blocked |= duplicate
            weights[blocked] = 0
            value = float(weights.sum())
            if value > 100 + 1e-8:
                raise ValueError('Sky factor exceeds 100%; refine resolution or inspect geometry')
            values.append(value)
            if retain:
                details.append(dict(contributions=weights.tolist(), visible=(~blocked & (d[:, 2] > 0)).tolist()))
        return (values, details)

    @staticmethod
    def _Summary(values, grid, threshold):
        if len(values) != len(grid['areas']) or not values:
            raise ValueError('Result length does not match grid areas')
        if any(not math.isfinite(float(v)) or v < 0 for v in values):
            raise ValueError('Sky factors must be finite and nonnegative')
        if any(not math.isfinite(float(a)) or a <= 0 for a in grid['areas']):
            raise ValueError('Grid areas must be finite and positive')
        area = sum(grid['areas'])
        lit = sum((a for a, v in zip(grid['areas'], values) if v >= threshold))
        return dict(
            area=area,
            areaAboveThreshold=lit,
            percentageAboveThreshold=100 * lit / area,
            meanSkyFactor=sum((a * v for a, v in zip(grid['areas'], values))) / area
        )

    @staticmethod
    def Waldram(room, scene, *, grid=None, height=0.85, spacing=0.3, threshold=0.2, resolution='medium', checkConvergence=True, cancelled=None):
        """Aperture-focused uniform-sky horizontal assessment (%).

        Includes room opaque faces automatically. Do not also supply its uncut
        Cell. Midpoint quadrature subdivides each aperture triangle. Optional
        convergence checks one finer angular level and RETURNS the finer result.
        Spatial refinement is separate. Threshold 0.2 means 0.2 percent.
        No glazing/frame losses, reflections, sunlight or legal conclusions.
        """
        from topologicpy.Topology import Topology
        from topologicpy.BVH import BVH
        from topologicpy.Vertex import Vertex
        import numpy as np
        if not isinstance(room, dict) or room.get('kind') != 'DaylightRoom':
            room = Daylight.PrepareRoom(room)
        threshold = Daylight._Positive(threshold, 'threshold', zero=True)
        level = Daylight._Resolution(resolution)
        tol = room['tolerance']
        if grid is None:
            grid = Daylight.SensorGrid(Daylight.WorkingPlane(room['cell'], height, tolerance=tol), spacing, tolerance=tol)
        for v in grid['vertices']:
            if not Vertex.IsInternal(v, room['cell'], tolerance=tol):
                raise ValueError('Grid sensor outside room')
        local = BVH.PrepareScene(room['opaqueFaces'], tolerance=tol)
        evaluation = dict(
            scene,
            bvh=BVH.ByTriangles(np.concatenate([scene['bvh']['triangles'] + scene['bvh']['offset'], local['triangles'] + local['offset']]), sourceIds=np.concatenate([scene['bvh']['sourceIds'],local['sourceIds']+scene['faceCount']]), tolerance=tol),
            faceLabels=scene.get('faceLabels',['Existing']*scene['faceCount'])+['Room']*len(room['opaqueFaces'])
        )
        triangles = Daylight._Triangles(room['apertures'], level + 2, cancelled)
        values, _ = Daylight._Integrate(grid['vertices'], triangles, evaluation, cancelled)
        result = dict(
            kind='DaylightWaldram',
            methodVersion='1',
            method='aperture-centroid-quadrature',
            unit='percent',
            sky='uniform',
            threshold=threshold,
            resolution=resolution,
            grid=grid,
            gridId=grid['gridId'],
            values=values,
            sceneId=scene['sceneId'],
            roomId=hashlib.sha256(repr((Daylight._GeometryKey(room['cell']), sorted(Daylight._GeometryKey(w) for w in room['apertures']))).encode()).hexdigest(),
            assumptions=['Direct uniform sky', 'No glazing or frame losses', 'No reflected light'],
            warnings=list(room['warnings']),
            areaMethod='piecewise-constant clipped cells',
            summary=Daylight._Summary(values, grid, threshold),
            convergence=None
        )
        if checkConvergence:
            fine_triangles = Daylight._Triangles(room['apertures'], level + 3, cancelled)
            fine_values, _ = Daylight._Integrate(grid['vertices'], fine_triangles, evaluation, cancelled)
            fine = dict(result, values=fine_values, summary=Daylight._Summary(fine_values, grid, threshold))
            result['convergence'] = Daylight.Convergence(result, fine, threshold=threshold)
            result['values'] = fine_values
            result['summary'] = fine['summary']
            triangles = fine_triangles
            result['warnings'].append('Angular sensitivity is not an error bound; also refine spatial grid')
        result['denominator'] = 'unobstructed external horizontal illuminance'
        result['angularSubdivisionDepth'] = level + 3 if checkConvergence else level + 2
        result['apertureTriangleCount'] = len(triangles)
        result['_diagram'] = dict(triangles=triangles, scene=evaluation, apertures=room['apertures'], cell=room['cell'])
        return result

    @staticmethod
    def Compare(before, after, *, threshold=0.2, referencePercentage=50.0):
        """Same-grid numeric comparison, including gains, losses and reference crossing."""
        threshold = Daylight._Positive(threshold, 'threshold', zero=True)
        if not math.isfinite(referencePercentage) or not 0 <= referencePercentage <= 100:
            raise ValueError('Invalid reference percentage')
        keys = ('gridId', 'unit', 'sky', 'method', 'methodVersion', 'resolution', 'areaMethod', 'roomId')
        if any((before.get(k) != after.get(k) for k in keys)):
            raise ValueError('Incompatible grids or method settings')
        if before.get('angularSubdivisionDepth') != after.get('angularSubdivisionDepth') or bool(before.get('convergence')) != bool(after.get('convergence')):
            raise ValueError('Incompatible refinement settings')
        if len(before['values']) != len(after['values']) or len(before['values']) != len(before['grid']['areas']):
            raise ValueError('Invalid result length')
        a, b = (Daylight._Summary(x['values'], x['grid'], threshold) for x in (before, after))
        lost = gained = 0
        for area, x, y in zip(before['grid']['areas'], before['values'], after['values']):
            if x >= threshold > y:
                lost += area
            if y >= threshold > x:
                gained += area
        return dict(
            before=a,
            after=b,
            threshold=threshold,
            percentagePointChange=b['percentageAboveThreshold'] - a['percentageAboveThreshold'],
            areaLost=lost,
            areaGained=gained,
            differences=[y - x for x, y in zip(before['values'], after['values'])],
            referencePercentage=referencePercentage,
            referenceCrossed=(a['percentageAboveThreshold'] >= referencePercentage) != (b['percentageAboveThreshold'] >= referencePercentage),
            convergence=dict(before=before.get('convergence'), after=after.get('convergence'))
        )

    @staticmethod
    def Convergence(coarse, fine, *, threshold=0.2):
        """Sensitivity on same domain/scene; different spatial grids allowed.

        Reports changes, never a certified error bound or automatic convergence.
        """
        from topologicpy.Topology import Topology
        from topologicpy.Face import Face
        threshold = Daylight._Positive(threshold, 'threshold', zero=True)
        for key in ('unit', 'sky', 'roomId', 'sceneId', 'method'):
            if coarse.get(key) != fine.get(key):
                raise ValueError('Different room, scene or method')
        ca, fa = (coarse['grid']['area'], fine['grid']['area'])
        if abs(ca - fa) > max(ca, fa) * 1e-06:
            raise ValueError('Different domain areas')
        same = coarse['gridId'] == fine['gridId']
        if not same:
            for source, target in ((coarse, fine), (fine, coarse)):
                for f in source['grid']['faces']:
                    remainder = f
                    for g in target['grid']['faces']:
                        if remainder is not None:
                            remainder = Topology.Difference(remainder, g, tolerance=source['grid']['tolerance'], silent=True)
                    if sum((Face.Area(x, mantissa=12) for x in Daylight._Faces(remainder))) > ca * 1e-06:
                        raise ValueError('Different domain shapes')
        a, b = (Daylight._Summary(x['values'], x['grid'], threshold) for x in (coarse, fine))
        return dict(
            percentagePointChange=b['percentageAboveThreshold'] - a['percentageAboveThreshold'],
            meanSkyFactorChange=b['meanSkyFactor'] - a['meanSkyFactor'],
            maxPointChange=max((abs(x - y) for x, y in zip(coarse['values'], fine['values']))) if same else None,
            sameGrid=same,
            coarseSensorCount=len(coarse['values']),
            fineSensorCount=len(fine['values']),
            interpretation='Sensitivity only; not an error bound'
        )

    @staticmethod
    def Contours(assessment, levels=None, *, tolerance=0.0001):
        """Piecewise-constant cell-region Faces/Wires, NOT interpolated isolines.

        Native unions remove internal cell edges. Wires include domain boundary.
        Region area agrees with the sampled assessment estimate.
        """
        from topologicpy.Topology import Topology
        from topologicpy.Face import Face
        tol = Daylight._Positive(tolerance, 'tolerance')
        levels = [assessment['threshold']] if levels is None else list(levels)
        output = []
        for level in levels:
            Daylight._Positive(level, 'level', zero=True)
            regions = []
            for select in (True, False):
                merged = None
                for f, value in zip(assessment['grid']['cells'], assessment['values']):
                    if (value >= level) == select:
                        merged = f if merged is None else Topology.Union(merged, f, tolerance=tol, silent=True)
                        if merged is None:
                            raise ValueError('Contour union failed')
                regions.append(Daylight._Faces(merged))
            above, below = regions
            expected = Daylight._Summary(assessment['values'], assessment['grid'], level)['areaAboveThreshold']
            if abs(sum((Face.Area(f, mantissa=12) for f in above)) - expected) > max(tol * tol * 10, assessment['grid']['area'] * 1e-06):
                raise ValueError('Contour area mismatch')
            wires = []
            for f in above:
                wires.append(Face.ExternalBoundary(f))
                wires.extend(Face.InternalBoundaries(f) or [])
            output.append(dict(
                level=level,
                aboveFaces=above,
                belowFaces=below,
                boundaryWires=wires,
                areaAboveThreshold=expected,
                method='piecewise-constant cell regions',
                includesDomainBoundary=True)
            )
        return output

    @staticmethod
    def WaldramDiagram(assessment, sensorIndex, *, origin=None, direction=None, scale=1.0, tolerance=0.0001,
                       calibrated=False, azimuthRange=None, angularStep=.5, gridStep=15,
                       projectionTolerance=1e-4, cancelled=None):
        """Sampled sheet Faces: x=azimuth radians, y=sin(altitude)^2/2.

        Midpoint visibility and straight projected edges are approximations.
        Numeric contributions equal assessment quadrature, not sheet-face areas.
        Horizon-straddling patches are omitted from geometry and flagged.
        """
        import numpy as np
        from topologicpy.Face import Face
        from topologicpy.Vertex import Vertex
        # Legacy native Face sheets remain the default for compatibility.
        # calibrated=True returns portable chart data in degrees and sin² altitude.
        if calibrated:
            data=Daylight._CalibratedWaldram(assessment,sensorIndex,direction,azimuthRange,angularStep,gridStep,projectionTolerance,cancelled)
            data.pop('_angular')
            return data
        tol = Daylight._Positive(tolerance, 'tolerance')
        scale = Daylight._Positive(scale, 'scale')
        if isinstance(sensorIndex, bool) or not isinstance(sensorIndex, int) or (not 0 <= sensorIndex < len(assessment['values'])):
            raise ValueError('Invalid sensor index')
        if '_diagram' not in assessment:
            raise ValueError('Diagram cache missing; recalculate')
        o = Daylight._XYZ([[0, 0, 0] if origin is None else origin])[0]
        d = Daylight._XYZ([[0, 1, 0] if direction is None else direction])[0]
        if not any(d[:2]):
            raise ValueError('Direction must have horizontal component')
        offset = math.atan2(d[0], d[1])
        p = assessment['grid']['coordinates'][sensorIndex]
        triangles = assessment['_diagram']['triangles']
        values, details = Daylight._Integrate([p], triangles, assessment['_diagram']['scene'], retain=True)
        patches = []
        omitted = 0
        for tri, visible, weight in zip(triangles, details[0]['visible'], details[0]['contributions']):
            rays = tri - p
            lengths = np.linalg.norm(rays, axis=1)
            if any(lengths <= tol):
                omitted += 1
                continue
            mu = rays[:, 2] / lengths
            if any(mu < 0):
                omitted += 1
                continue
            az = np.unwrap(np.arctan2(rays[:, 0], rays[:, 1]) - offset)
            az -= math.floor((float(az.mean()) + math.pi) / (2 * math.pi)) * 2 * math.pi
            coords = [o + scale * np.array([a, m * m / 2, 0]) for a, m in zip(az, mu)]
            face = Face.ByVertices([Vertex.ByCoordinates(*q) for q in coords], tolerance=tol, silent=True)
            if face is not None:
                patches.append(dict(face=face, visible=visible, contribution=weight))
            else:
                omitted += 1
        return dict(
            patches=patches,
            skyFactor=values[0],
            unit='percent',
            sensorIndex=sensorIndex,
            omittedPatchCount=omitted,
            projection='azimuth / half sine-squared altitude',
            warnings=['Sampled visibility; straight projected edges; seam may extend beyond sheet bounds']
        )

    @staticmethod
    def _WaldramProjection(point, sensor, forward):
        import numpy as np
        q = np.asarray(point, dtype=float)-sensor
        horizontal = math.hypot(q[0], q[1])
        azimuth = math.degrees(math.atan2(q[0]*forward[1]-q[1]*forward[0], q[0]*forward[0]+q[1]*forward[1]))
        altitude = math.degrees(math.atan2(q[2], horizontal))
        return [azimuth, math.sin(math.radians(altitude))**2]

    @staticmethod
    def _WaldramRuns(categories, azimuths, heights):
        """Coalesce horizontal sampled cells, retaining their exact projected area."""
        regions = {}
        for j, row in enumerate(categories):
            start = 0
            while start < len(row):
                value = str(row[start]); end = start+1
                while end < len(row) and row[end] == row[start]: end += 1
                if value != 'outside':
                    regions.setdefault(value, []).append([float(azimuths[start]), float(azimuths[end]),
                                                         float(heights[j]), float(heights[j+1])])
                start = end
        return [{'category': key, 'rectangles': value} for key, value in regions.items()]

    @staticmethod
    def _WaldramOutline(aperture, sensor, forward, bounds, error):
        """Adaptive curved projection of straight aperture edges, clipped at horizon/seam."""
        import numpy as np
        from topologicpy.Face import Face
        from topologicpy.Wire import Wire
        from topologicpy.Vertex import Vertex
        segments = []
        def projected(p): return Daylight._WaldramProjection(p, sensor, forward)
        def sample(a, b, depth=0):
            pa, pb = projected(a), projected(b)
            while pb[0]-pa[0] > 180: pb[0] -= 360
            while pb[0]-pa[0] < -180: pb[0] += 360
            middle = (a+b)/2; pm = projected(middle)
            while pm[0]-(pa[0]+pb[0])/2 > 180: pm[0] -= 360
            while pm[0]-(pa[0]+pb[0])/2 < -180: pm[0] += 360
            deviation = max(abs(pm[0]-(pa[0]+pb[0])/2)/360, abs(pm[1]-(pa[1]+pb[1])/2))
            if depth < 14 and deviation > error:
                sample(a,middle,depth+1); sample(middle,b,depth+1)
            else:
                for shift in (-360,0,360):
                    x0,x1 = pa[0]+shift,pb[0]+shift
                    lo,hi = 0.,1.
                    for origin,delta,minimum,maximum in ((x0,x1-x0,bounds[0],bounds[1]),(pa[1],pb[1]-pa[1],0,1)):
                        if abs(delta)<1e-15:
                            if not minimum <= origin <= maximum: lo,hi=1,0
                        else:
                            t0,t1=(minimum-origin)/delta,(maximum-origin)/delta
                            lo=max(lo,min(t0,t1));hi=min(hi,max(t0,t1))
                    if hi>lo:
                        segments.append([[x0+(x1-x0)*t,pa[1]+(pb[1]-pa[1])*t] for t in (lo,hi)])
        for wire in [Face.ExternalBoundary(aperture), *(Face.InternalBoundaries(aperture) or [])]:
            vertices = Daylight._XYZ(Wire.Vertices(wire))
            for a,b in zip(vertices,np.roll(vertices,-1,axis=0)):
                a,b=a.copy(),b.copy()
                if a[2]<sensor[2] and b[2]<sensor[2]: continue
                if (a[2]<sensor[2]) != (b[2]<sensor[2]):
                    p=a+(b-a)*(sensor[2]-a[2])/(b[2]-a[2])
                    if a[2]<sensor[2]: a=p
                    else: b=p
                sample(a,b)
        return segments

    @staticmethod
    def _WaldramPlan(cache):
        """Native floor footprints and window edges for a world-XY context inset."""
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        from topologicpy.Wire import Wire
        from topologicpy.Vertex import Vertex
        def footprint(topology):
            if isinstance(topology,dict):topology=topology['cell']
            faces=Daylight._Faces(topology)
            candidates=[f for f in faces if abs(Face.Normal(f)[2])>.999]
            if not candidates:return []
            low=min(Vertex.Z(Topology.Centroid(f)) for f in candidates)
            return [[[float(p[0]),float(p[1])] for p in Daylight._XYZ(Wire.Vertices(Face.ExternalBoundary(f)))]
                    for f in candidates if abs(Vertex.Z(Topology.Centroid(f))-low)<cache['scene']['tolerance']]
        scene=[]
        for topology,label in cache['scene'].get('_context',[]):
            scene.extend(dict(role=label,polygon=p) for p in footprint(topology))
        return dict(scene=scene,room=footprint(cache['cell']),
                    windows=[Daylight._XYZ(Topology.Vertices(f))[:,:2].tolist() for f in cache['apertures']],
                    axes='world XY',unit='model units')

    @staticmethod
    def _CalibratedWaldram(assessment, sensorIndex, direction=None, azimuthRange=None,
                           angularStep=.5, gridStep=15, projectionTolerance=1e-4,
                           cancelled=None):
        import numpy as np
        from topologicpy.BVH import BVH
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology
        Daylight._Positive(angularStep,'angularStep')
        Daylight._Positive(gridStep,'gridStep')
        Daylight._Positive(projectionTolerance,'projectionTolerance')
        if angularStep > 10 or gridStep > 45: raise ValueError('Angular/grid steps are too coarse')
        if isinstance(sensorIndex,bool) or not isinstance(sensorIndex,int) or not 0 <= sensorIndex < len(assessment['values']):
            raise ValueError('Invalid sensor index')
        cache=assessment.get('_diagram')
        if not cache or 'apertures' not in cache:
            raise ValueError('Recalculate assessment to populate calibrated diagram geometry')
        p=np.asarray(assessment['grid']['coordinates'][sensorIndex],dtype=float)
        apertures=cache['apertures']
        if direction is None:
            forward=np.array([0.,1.,0.])
            if apertures:
                n=np.asarray(Face.Normal(apertures[0],mantissa=12),dtype=float)
                centre=Daylight._XYZ([Topology.Centroid(apertures[0])])[0]
                if np.dot(n,centre-p)<0:n=-n
                if np.linalg.norm(n[:2])>1e-8:forward=np.array([n[0],n[1],0.])/np.linalg.norm(n[:2])
        else:
            forward=Daylight._XYZ([direction])[0]
            if np.linalg.norm(forward[:2])<1e-12: raise ValueError('Direction must have a horizontal component')
            forward=np.array([forward[0],forward[1],0.])/np.linalg.norm(forward[:2])
        if azimuthRange is None:
            wide=any(abs(Face.Normal(f)[2])>.99 for f in apertures)
            for f in apertures:
                wide |= any(abs(Daylight._WaldramProjection(v,p,forward)[0])>90+1e-7
                            for v in Daylight._XYZ(Topology.Vertices(f)))
            azimuthRange=(-180,180) if wide else (-90,90)
        if len(azimuthRange)!=2 or not all(math.isfinite(float(v)) for v in azimuthRange):
            raise ValueError('Invalid azimuth range')
        amin,amax=map(float,azimuthRange)
        if not -180<=amin<amax<=180:raise ValueError('Azimuth range must be within -180..180 degrees')
        nx=int(math.ceil((amax-amin)/angularStep));ny=int(math.ceil(90/angularStep))
        if nx*ny>600000:raise ValueError('Diagram exceeds 600000 angular cells; increase angularStep')
        az=np.linspace(amin,amax,nx+1);alt=np.linspace(0,90,ny+1)
        y=np.sin(np.radians(alt))**2
        mid_alt=np.arcsin(np.sqrt((y[:-1]+y[1:])/2))
        mid_az=np.radians((az[:-1]+az[1:])/2)
        a,h=np.meshgrid(mid_az,mid_alt)
        right=np.array([forward[1],-forward[0],0.])
        directions=(np.cos(h)[...,None]*(np.cos(a)[...,None]*forward+np.sin(a)[...,None]*right)
                    +np.sin(h)[...,None]*np.array([0,0,1])).reshape(-1,3)
        Daylight._Cancelled(cancelled)
        aperture_scene=BVH.ByTriangles(cache['triangles'],tolerance=cache['scene']['tolerance'])
        aperture_hits=BVH.RayHits(aperture_scene,p,directions,cancelled=cancelled)
        mask=np.array([hit is not None for hit in aperture_hits])
        active=np.flatnonzero(mask)
        visible=np.zeros(nx*ny,dtype=bool)
        categories=np.full(nx*ny,'outside',dtype=object)
        categories[active]='Existing'
        if len(active):
            hits=BVH.RayHits(cache['scene']['bvh'],p,directions[active],cancelled=cancelled)
            labels=cache['scene'].get('faceLabels',[])
            for index,hit in zip(active,hits):
                if hit is None:visible[index]=True;categories[index]='visible'
                else:
                    source=hit['sourceId']
                    categories[index]=labels[source] if 0<=source<len(labels) else 'Existing'
        weights=np.diff(y)[:,None]*np.diff(az)[None,:]*100/360
        visible=visible.reshape(ny,nx);mask=mask.reshape(ny,nx)
        factor=float(np.sum(weights*visible))
        graticule=dict(azimuth=[],altitude=[],droop=[])
        for angle in np.arange(math.ceil(amin/gridStep)*gridStep,amax+1e-7,gridStep):
            graticule['azimuth'].append(dict(angle=float(angle),points=[[float(angle),0],[float(angle),1]]))
        for angle in np.arange(0,90+1e-7,gridStep):
            yy=math.sin(math.radians(float(angle)))**2
            graticule['altitude'].append(dict(angle=float(angle),points=[[amin,yy],[amax,yy]]))
        if amin>=-90 and amax<=90:
            xx=np.linspace(amin,amax,361)
            for angle in np.arange(gridStep,90,gridStep):
                hh=np.arctan(math.tan(math.radians(float(angle)))*np.cos(np.radians(xx)))
                graticule['droop'].append(dict(angle=float(angle),points=np.column_stack([xx,np.sin(hh)**2]).tolist()))
        outlines=[dict(windowIndex=i,segments=Daylight._WaldramOutline(f,p,forward,(amin,amax),projectionTolerance))
                  for i,f in enumerate(apertures)]
        target=float(assessment['values'][sensorIndex])
        return dict(kind='CalibratedWaldram',sensorIndex=sensorIndex,sensor=p.tolist(),unit='percent',
                    skyFactor=target,diagramSkyFactor=factor,
                    frame=dict(azimuthRange=[amin,amax],altitudeRange=[0,90],forward=forward.tolist(),
                               bearingDegrees=math.degrees(math.atan2(forward[0],forward[1]))%360,
                               projection='uniform-sky horizontal; y = sin²(altitude)',
                               fullFrameSkyFactor=(amax-amin)*100/360),
                    graticule=graticule,apertureOutlines=outlines,plan=Daylight._WaldramPlan(cache),
                    regions=Daylight._WaldramRuns(categories.reshape(ny,nx),az,y),
                    angularStep=[float(az[1]-az[0]),float(alt[1]-alt[0])],
                    angularCellCount=nx*ny,projectionTolerance=projectionTolerance,
                    reconciliation=dict(assessmentSkyFactor=target,diagramSkyFactor=factor,differencePP=factor-target),
                    warnings=['Angularly sampled visibility; curved aperture edges are adaptively approximated'],
                    _angular=dict(azimuth=az,heights=y,visible=visible,window=mask,weights=weights))

    @staticmethod
    def WaldramDiagramComparison(before, after, sensorIndex, *, direction=None,
                                  azimuthRange=None, angularStep=.5, gridStep=15,
                                  projectionTolerance=1e-4, reconciliationTolerance=.05,
                                  maxRefinements=1, cancelled=None):
        """Calibrated paired uniform-sky charts and loss/gain regions.

        Uses one angular frame/grid for both states. Equal projected chart areas
        represent equal horizontal sky-factor contributions. Reconciliation
        compares an independent angular integration to the assessment, in
        percentage points; it is reported, never forced to match. This is a
        sampled numerical diagram, not an exact legal/certified silhouette.
        """
        import numpy as np
        Daylight.Compare(before,after)
        Daylight._Positive(reconciliationTolerance,'reconciliationTolerance')
        if isinstance(maxRefinements,bool) or not isinstance(maxRefinements,int) or not 0<=maxRefinements<=3:
            raise ValueError('maxRefinements must be an integer in 0..3')
        step=angularStep
        for refinement in range(maxRefinements+1):
            b=Daylight._CalibratedWaldram(before,sensorIndex,direction,azimuthRange,step,gridStep,projectionTolerance,cancelled)
            frame=b['frame']
            a=Daylight._CalibratedWaldram(after,sensorIndex,frame['forward'],frame['azimuthRange'],step,gridStep,projectionTolerance,cancelled)
            agreement=all(abs(d['reconciliation']['differencePP'])<=reconciliationTolerance for d in (b,a))
            nx=math.ceil((frame['azimuthRange'][1]-frame['azimuthRange'][0])/(step/2));ny=math.ceil(90/(step/2))
            if agreement or refinement==maxRefinements or nx*ny>600000:break
            step/=2
        bg,ag=b['_angular'],a['_angular']
        if not np.array_equal(bg['window'],ag['window']):raise ValueError('Before/After window masks differ')
        categories=np.full(bg['visible'].shape,'outside',dtype=object)
        categories[bg['window']]='blocked'
        categories[bg['visible']&ag['visible']]='retained'
        categories[bg['visible']&~ag['visible']]='lost'
        categories[~bg['visible']&ag['visible']]='gained'
        changes={name:float(np.sum(bg['weights']*(categories==name))) for name in ('lost','gained','retained')}
        regions=Daylight._WaldramRuns(categories,bg['azimuth'],bg['heights'])
        for d in (b,a):
            d.pop('_angular')
            d['reconciliation'].update(tolerancePP=reconciliationTolerance,withinTolerance=abs(d['reconciliation']['differencePP'])<=reconciliationTolerance)
        return dict(kind='WaldramDiagramComparison',before=b,after=a,
                    change=dict(regions=regions,**changes,diagramChangePP=a['diagramSkyFactor']-b['diagramSkyFactor'],
                                assessmentChangePP=a['skyFactor']-b['skyFactor']),
                    withinTolerance=agreement,refinements=refinement,
                    warnings=[] if agreement else ['Diagram/assessment discrepancy exceeds tolerance; refine both angular and room sampling'])

    @staticmethod
    def ResultData(assessment, *, includeGeometry=False):
        """Portable JSON-compatible assessment, excluding live scene/diagram cache.

        Optional geometry uses TopologicPy BREP strings. Recalculate to recreate
        diagram caches; this export is not a prepared-scene persistence format.
        """
        import copy
        from topologicpy.Topology import Topology
        result = {k: copy.deepcopy(v) for k, v in assessment.items() if k not in ('grid', '_diagram')}
        grid = assessment['grid']
        result['grid'] = {k: copy.deepcopy(v) for k, v in grid.items() if k not in ('faces', 'cells', 'vertices')}
        if includeGeometry:
            result['grid']['faceBREPs'] = [Topology.BREPString(f) for f in grid['faces']]
            result['grid']['cellBREPs'] = [Topology.BREPString(f) for f in grid['cells']]
        return result
