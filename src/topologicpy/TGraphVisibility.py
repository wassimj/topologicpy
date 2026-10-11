# Copyright (C) 2026
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free Software
# Foundation, either version 3.0 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUTv
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

"""Optional single-core visibility predicates for TGraph.VisibilityGraph.

Imported only on request. No fast-math or parallel execution is used, so
the tolerance, endpoint and corner rules match the Python predicates.
"""
import numpy as np

from numba import njit

@njit(cache=True, fastmath=False)
def _dist2_xy(ax, ay, bx, by, tol, tol2, allowBoundaryPaths):
    dx = ax - bx
    dy = ay - by
    return dx * dx + dy * dy

@njit(cache=True, fastmath=False)
def _same_xy(ax, ay, bx, by, tol, tol2, allowBoundaryPaths):
    return _dist2_xy(ax, ay, bx, by, tol, tol2, allowBoundaryPaths) <= tol2

@njit(cache=True, fastmath=False)
def _orient(ax, ay, bx, by, cx, cy, tol, tol2, allowBoundaryPaths):
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

@njit(cache=True, fastmath=False)
def _on_segment(ax, ay, bx, by, px, py, tol, tol2, allowBoundaryPaths):
    if abs(_orient(ax, ay, bx, by, px, py, tol, tol2, allowBoundaryPaths)) > tol:
        return False
    return min(ax, bx) - tol <= px <= max(ax, bx) + tol and min(ay, by) - tol <= py <= max(ay, by) + tol

@njit(cache=True, fastmath=False)
def _bbox_overlap(a_minx, a_miny, a_maxx, a_maxy, b_minx, b_miny, b_maxx, b_maxy, tol, tol2, allowBoundaryPaths):
    return not (a_maxx < b_minx - tol or b_maxx < a_minx - tol or a_maxy < b_miny - tol or (b_maxy < a_miny - tol))

@njit(cache=True, fastmath=False)
def _collinear_overlap_length(ax, ay, bx, by, cx, cy, dx, dy, tol, tol2, allowBoundaryPaths):
    if abs(bx - ax) >= abs(by - ay):
        a0, a1 = sorted((ax, bx))
        c0, c1 = sorted((cx, dx))
    else:
        a0, a1 = sorted((ay, by))
        c0, c1 = sorted((cy, dy))
    return min(a1, c1) - max(a0, c0)

@njit(cache=True, fastmath=False)
def _segments_block_with_bbox(ax, ay, bx, by, candidate_bbox, blocker, tol, tol2, allowBoundaryPaths):
    cx, cy, dx, dy, minx, miny, maxx, maxy = blocker
    a_minx, a_miny, a_maxx, a_maxy = candidate_bbox
    if not _bbox_overlap(a_minx, a_miny, a_maxx, a_maxy, minx, miny, maxx, maxy, tol, tol2, allowBoundaryPaths):
        return False
    if allowBoundaryPaths:
        if _same_xy(ax, ay, cx, cy, tol, tol2, allowBoundaryPaths) and _same_xy(bx, by, dx, dy, tol, tol2, allowBoundaryPaths) or (_same_xy(ax, ay, dx, dy, tol, tol2, allowBoundaryPaths) and _same_xy(bx, by, cx, cy, tol, tol2, allowBoundaryPaths)):
            return False
    o1 = _orient(ax, ay, bx, by, cx, cy, tol, tol2, allowBoundaryPaths)
    o2 = _orient(ax, ay, bx, by, dx, dy, tol, tol2, allowBoundaryPaths)
    o3 = _orient(cx, cy, dx, dy, ax, ay, tol, tol2, allowBoundaryPaths)
    o4 = _orient(cx, cy, dx, dy, bx, by, tol, tol2, allowBoundaryPaths)
    if (o1 > tol and o2 < -tol or (o1 < -tol and o2 > tol)) and (o3 > tol and o4 < -tol or (o3 < -tol and o4 > tol)):
        return True
    if abs(o1) <= tol and abs(o2) <= tol and (abs(o3) <= tol) and (abs(o4) <= tol):
        overlap = _collinear_overlap_length(ax, ay, bx, by, cx, cy, dx, dy, tol, tol2, allowBoundaryPaths)
        if overlap <= tol:
            return False
        return not allowBoundaryPaths
    if abs(o3) <= tol and _on_segment(cx, cy, dx, dy, ax, ay, tol, tol2, allowBoundaryPaths):
        return False
    if abs(o4) <= tol and _on_segment(cx, cy, dx, dy, bx, by, tol, tol2, allowBoundaryPaths):
        return False
    if abs(o1) <= tol and _on_segment(ax, ay, bx, by, cx, cy, tol, tol2, allowBoundaryPaths):
        if _same_xy(cx, cy, ax, ay, tol, tol2, allowBoundaryPaths) or _same_xy(cx, cy, bx, by, tol, tol2, allowBoundaryPaths):
            return False
        return True
    if abs(o2) <= tol and _on_segment(ax, ay, bx, by, dx, dy, tol, tol2, allowBoundaryPaths):
        if _same_xy(dx, dy, ax, ay, tol, tol2, allowBoundaryPaths) or _same_xy(dx, dy, bx, by, tol, tol2, allowBoundaryPaths):
            return False
        return True
    return False


@njit(cache=True, fastmath=False)
def visible_source(i, targets, xs, ys, blockers, face_blockers, tol, tol2, allowBoundaryPaths, trim, limit):
    result=np.zeros(len(targets),dtype=np.bool_)
    ax=xs[i]; ay=ys[i]
    for k in range(len(targets)):
        j=targets[k]; bx=xs[j]; by=ys[j]
        d2=_dist2_xy(ax,ay,bx,by,tol,tol2,allowBoundaryPaths)
        if d2<=tol2 or (limit>=0 and d2>limit): continue
        bbox=(min(ax,bx),min(ay,by),max(ax,bx),max(ay,by))
        blocked=False
        for b in blockers:
            if _segments_block_with_bbox(ax,ay,bx,by,bbox,b,tol,tol2,allowBoundaryPaths):
                blocked=True; break
        if blocked: continue
        inside=True
        if trim:
            for fraction in (0.25,0.5,0.75):
                px=ax+fraction*(bx-ax); py=ay+fraction*(by-ay)
                parity=False; boundary=False
                for b in face_blockers:
                    cx,cy,dx,dy=b[0],b[1],b[2],b[3]
                    if _on_segment(cx,cy,dx,dy,px,py,tol,tol2,allowBoundaryPaths):
                        boundary=True; break
                    if (cy>py)!=(dy>py):
                        intersection=cx+(py-cy)*(dx-cx)/((dy-cy) or 1e-300)
                        if intersection>px+tol: parity=not parity
                if not (boundary or parity):
                    inside=False; break
        result[k]=inside
    return result
