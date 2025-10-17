# Copyright (C) 2025
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations
import os
import warnings

try:
    import honeybee.facetype
    from honeybee.face import Face as HBFace
    from honeybee.model import Model as HBModel
    from honeybee.room import Room as HBRoom
    from honeybee.shade import Shade as HBShade
    from honeybee.aperture import Aperture as HBAperture
    from honeybee.door import Door as HBDoor
except:
    print("Honeybee - Installing required honeybee library.")
    try:
        os.system("pip install honeybee")
    except:
        os.system("pip install honeybee --user")
    try:
        import honeybee.facetype
        from honeybee.face import Face as HBFace
        from honeybee.model import Model as HBModel
        from honeybee.room import Room as HBRoom
        from honeybee.shade import Shade as HBShade
        from honeybee.aperture import Aperture as HBAperture
        from honeybee.door import Door as HBDoor
    except:
        warnings.warn("Honeybee - ERROR: Could not import honeybee")

try:
    import honeybee_energy.lib.constructionsets as constr_set_lib
    import honeybee_energy.lib.programtypes as prog_type_lib
    import honeybee_energy.lib.scheduletypelimits as schedule_types
    from honeybee_energy.schedule.ruleset import ScheduleRuleset
    from honeybee_energy.schedule.day import ScheduleDay
    from honeybee_energy.load.setpoint import Setpoint
    from honeybee_energy.load.hotwater import  ServiceHotWater
except:
    print("Honeybee - Installing required honeybee-energy library.")
    try:
        os.system("pip install -U honeybee-energy[standards]")
    except:
        os.system("pip install -U honeybee-energy[standards] --user")
    try:
        import honeybee_energy.lib.constructionsets as constr_set_lib
        import honeybee_energy.lib.programtypes as prog_type_lib
        import honeybee_energy.lib.scheduletypelimits as schedule_types
        from honeybee_energy.schedule.ruleset import ScheduleRuleset
        from honeybee_energy.schedule.day import ScheduleDay
        from honeybee_energy.load.setpoint import Setpoint
        from honeybee_energy.load.hotwater import  ServiceHotWater
    except:
        warnings.warn("Honeybee - Error: Could not import honeybee-energy")

try:
    from honeybee_radiance.sensorgrid import SensorGrid
except:
    print("Honeybee - Installing required honeybee-radiance library.")
    try:
        os.system("pip install -U honeybee-radiance")
    except:
        os.system("pip install -U honeybee-radiance --user")
    try:
        from honeybee_radiance.sensorgrid import SensorGrid
    except:
        warnings.warn("Honeybee - Error: Could not import honeybee-radiance")

try:
    from ladybug.dt import Time
except:
    print("Honeybee - Installing required ladybug library.")
    try:
        os.system("pip install -U ladybug")
    except:
        os.system("pip install -U ladybug --user")
    try:
        from ladybug.dt import Time
    except:
        warnings.warn("Honeybee - Error: Could not import ladybug")

try:
    from ladybug_geometry.geometry3d.face import Face3D
    from ladybug_geometry.geometry3d.pointvector import Point3D, Vector3D
except:
    print("Honeybee - Installing required ladybug-geometry library.")
    try:
        os.system("pip install -U ladybug-geometry")
    except:
        os.system("pip install -U ladybug-geometry --user")
    try:
        from ladybug_geometry.geometry3d.face import Face3D
        from ladybug_geometry.geometry3d.pointvector import Point3D, Vector3D
    except:
        warnings.warn("Honeybee - Error: Could not import ladybug-geometry")

import json
import topologic_core as topologic

class Honeybee:
    @staticmethod
    def ByHBJSONDictionary(
        dictionary,
        includeRooms: bool = True,
        includeFaces: bool = True,
        includeShades: bool = True,
        includeApertures: bool = True,
        includeDoors: bool = True,
        includeOrphanedRooms: bool = True,
        includeOrphanedFaces: bool = True,
        includeOrphanedShades: bool = True,
        includeOrphanedApertures: bool = True,
        includeOrphanedDoors: bool = True,
        tolerance: float = 0.0001,
        silent: bool = False):
        """
        Import an HBJSON model from a python dictionary and return a python dictionary. See: https://github.com/ladybug-tools/honeybee-schema/wiki/1.1-Model-Schema
        
        Parameters
        ----------
        dictionary : dict
            The HBJSON model as a Python dictionary (e.g., loaded via ``json.load``).
        includeRooms : bool, optional
            If True, parse rooms and attempt to create one ``Cell`` per room. Default is True.
        includeFaces : bool, optional
            If True, include top-level planar faces found outside rooms (e.g., at root "faces"). Default is True.
        includeShades : bool, optional
            If True, include context/standalone shades (e.g., ``context_geometry.shades``). Default is True.
        includeApertures : bool, optional
            If True, include **room** apertures (e.g., windows) as separate ``Face`` objects (not cut from hosts). Default is True.
        includeDoors : bool, optional
            If True, include **room** doors as separate ``Face`` objects (not cut from hosts). Default is True.
        includeOrphanedRooms : bool, optional
            If True, include the topology of the room when a room fails to close as a ``Cell``. This may be a ``Shell`` or a ``Cluster``. Default is True.
        includeOrphanedFaces : bool, optional
            If True, include planar faces listed at the HBJSON root (e.g., "faces"). Default is True.
        includeOrphanedShades : bool, optional
            If True, include shades listed at the HBJSON root (e.g., "orphaned_shades"). Default is True.
        includeOrphanedApertures : bool, optional
            If True, include apertures listed at the HBJSON root (e.g., "orphaned_apertures"). Default is True.
        includeOrphanedDoors : bool, optional
            If True, include doors listed at the HBJSON root (e.g., "orphaned_doors"). Default is True.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
        Returns
        -------
        dict
            The created cluster of vertices, edges, faces, and cells.
            - 'rooms': list of Cells (one per successfully closed room)
            - 'faces': list of Faces (all faces that make up the rooms)
            - 'shades': list of Faces (all shade faces)
            - 'apertures': list of Faces (all apertures, never cut from hosts)
            - 'doors': list of Faces (all doors, never cut from hosts)
            - 'orphanedRooms': list of Topologies (context/top-level topologies (e.g. Shells or Clustser) that failed to form a Cell)
            - 'orphanedFaces': list of Faces (context/top-level faces + host faces of rooms that failed to form a Cell)
            - 'orphanedShades': list of Faces (context/top-level shade faces that failed to have a parent cell)
            - 'orphanedApertures': list of Faces (apertures that failed to have a parent face)
            - 'orphanedDoors': list of Faces (doors that failed to have a parent face)
            - 'properties': hierarchical dict copied verbatim from HBJSON['properties']
        """

        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Shell import Shell
        from topologicpy.Cell import Cell
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Helper import Helper
        from typing import Any, Dict, List, Optional, Tuple

        if not isinstance(dictionary, dict):
            if not silent:
                print("Honeybee.ByHBJSONDictionary - Error: The input dictionary parameter is not a valid python dictionary. Returning None.")
            return None

        # ---------------------- helpers ----------------------
        def _close(points: List[List[float]]) -> List[List[float]]:
            if not points:
                return points
            p0, pN = points[0], points[-1]
            if (abs(p0[0]-pN[0]) > tolerance or
                abs(p0[1]-pN[1]) > tolerance or
                abs(p0[2]-pN[2]) > tolerance):
                return points + [p0]
            return points

        def _V(p: List[float]) -> Vertex:
            return Vertex.ByCoordinates(float(p[0]), float(p[1]), float(p[2]))

        # Tolerance-filtered wire (your spec)
        def _wire(points: List[List[float]], tolerance: float = 1e-6) -> Wire:
            pts = _close(points)
            verts = [_V(x) for x in pts]
            edges = [
                Edge.ByVertices(verts[i], verts[i+1], tolerance=tolerance, silent=True)
                for i in range(len(verts)-1)
                if Vertex.Distance(verts[i], verts[i+1]) > tolerance
            ]
            w = None
            try:
                w = Wire.ByEdges(edges, tolerance=tolerance, silent=True)
            except:
                w = Topology.SelfMerge(Cluster.ByTopologies(edges), tolerance=tolerance)
            if w == None:
                if not silent:
                    print("Honeybee.ByHBSJONDictionary - Error: Could not build wire. Returning None.")
            return w

        def _face_from_boundary(boundary: List[List[float]]) -> Face:
            w = _wire(boundary, tolerance)
            if w:
                f = Face.ByWire(w, tolerance=tolerance, silent=True)
                if not f:
                    if not silent:
                        print("Honeybee.ByHBSJONDictionary - Error: Could not build face. Returning the wire")
                    return w
                return f
            if not silent:
                print("Honeybee.ByHBSJONDictionary - Error: Could not build face. Returning None")
            return None

        def _attach_all(top: Topology, full_py_dict: Dict[str, Any]) -> Topology:
            # Attach the entire available dict (no filtering)
            try:
                keys = list(full_py_dict.keys())
                values = [full_py_dict[k] for k in keys]
                d = Dictionary.ByKeysValues(keys, values)
                return Topology.SetDictionary(top, d)
            except Exception:
                return top  # be robust to non-serializable values

        def _build_host_face_cut_holes(fobj: Dict[str, Any]) -> Optional[Face]:
            """
            Build host face from outer boundary and cut ONLY explicit 'holes' (NOT apertures/doors).
            Attach the full fobj dict.
            """
            geom = fobj.get("geometry") or {}
            boundary = geom.get("boundary") or fobj.get("boundary")
            holes = geom.get("holes") or fobj.get("holes") or []

            if not boundary or len(boundary) < 3:
                return None

            hosts = _face_from_boundary(boundary)
            if Topology.IsInstance(hosts, "face") or Topology.IsInstance(hosts, "wire"):
                hosts = [hosts]

            for host in hosts:
                # Subtract explicit hole loops (if any)
                hole_faces: List[Face] = []
                for h in holes:
                    if h and len(h) >= 3:
                        hole_faces.append(_face_from_boundary(h))

                if hole_faces:
                    hole_cluster = Cluster.ByTopologies(hole_faces)
                    try:
                        host = Topology.Difference(host, hole_cluster)
                    except Exception as e:
                        if not silent:
                            name = fobj.get("identifier") or fobj.get("name") or "unnamed"
                            print(f"HBJSON Import: Hole cutting failed on face '{name}'. Keeping uncut. Error: {e}")
                _attach_all(host, fobj)
            return hosts

        def _aperture_faces_from(fobj: Dict[str, Any], kind: str) -> List[Face]:
            """
            Build separate faces for apertures/doors on a host (DO NOT cut from host).
            'kind' ∈ {'apertures','doors'}. Attach full dict for each.
            """
            out: List[Face] = []
            ap_list = fobj.get(kind) or []
            for ap in ap_list:
                g = ap.get("geometry") or {}
                boundary = g.get("boundary") or ap.get("boundary")
                if not boundary or len(boundary) < 3:
                    continue
                f = _face_from_boundary(boundary)
                out.append(_attach_all(f, ap))
            return out

        def _orphaned_aperture_faces(ap_list: List[Dict[str, Any]]) -> List[Face]:
            out: List[Face] = []
            for ap in ap_list or []:
                g = ap.get("geometry") or {}
                boundary = g.get("boundary") or ap.get("boundary")
                if not boundary or len(boundary) < 3:
                    continue
                f = _face_from_boundary(boundary)
                out.append(_attach_all(f, ap))
            return out

        def _room_to_cell_and_apertures(room: Dict[str, Any]) -> Tuple[Optional[Cell], List[Face], List[Face]]:
            """
            Build host faces (cut 'holes' only) and aperture faces for a room.
            Return (cell_or_none, host_faces, aperture_faces).
            """
            hb_faces = room.get("faces") or room.get("Faces") or []
            rm_faces: List[Face] = []
            sh_faces = room.get("shades") or room.get("Shades") or []
            ap_faces: List[Face] = []
            dr_faces: List[Face] = []


            for fobj in hb_faces:
                hosts = _build_host_face_cut_holes(fobj)
                if hosts:
                    rm_faces.extend(hosts)
                ap_faces.extend(_aperture_faces_from(fobj, "apertures"))
                dr_faces.extend(_aperture_faces_from(fobj, "doors"))
            
            # Room Shades
            for sh in sh_faces:
                shades = _build_host_face_cut_holes(sh)
                if shades:
                    sh_faces.extend(shades)
            # Try to make a Cell. If it fails, we DO NOT return a Shell/Cluster in rooms;
            # instead we will salvage host faces into the 'faces' bucket.
            if rm_faces:
                selectors = []
                for rm_face in rm_faces:
                    s = Topology.InternalVertex(rm_face)
                    face_d = Topology.Dictionary(rm_face)
                    s = Topology.SetDictionary(s, face_d)
                    selectors.append(s)
                cell = Cell.ByFaces(Helper.Flatten(rm_faces), tolerance=0.001, silent=True)
                if Topology.IsInstance(cell, "cell"):
                    cell = _attach_all(cell, room)  # attach full room dict
                else:
                    cell = Shell.ByFaces(Helper.Flatten(rm_faces), tolerance=0.001, silent=True)
                    if not cell:
                        cell = Cluster.ByTopologies(Helper.Flatten(rm_faces), silent=True)
                if Topology.IsInstance(cell, "topology"):
                    cell = _attach_all(cell, room)  # attach full room dict
                
                if Topology.IsInstance(cell, "Topology"):
                    cell = Topology.TransferDictionariesBySelectors(cell, selectors,tranFaces=True, numWorkers=1)
                return cell, rm_faces, sh_faces, ap_faces, dr_faces
            # No host faces -> no cell
            return None, [], sh_faces, ap_faces, dr_faces

        rooms: List[Cell] = []
        faces: List[Face] = []
        shades: List[Face] = []
        apertures: List[Face] = []
        doors: List[Face] = []
        orphaned_rooms: List[Cell] = []
        orphaned_faces: List[Face] = []
        orphaned_shades: List[Face] = []
        orphaned_apertures: List[Face] = []
        orphaned_doors: List[Face] = []

        # Rooms → Cells (when possible) + collect apertures. If a Cell cannot be made,
        # the room goes to the orphaned_rooms list.
        for room in (dictionary.get("rooms") or dictionary.get("Rooms") or []):
            cell, host_faces, sh_faces, ap_faces, dr_faces = _room_to_cell_and_apertures(room)

            if includeRooms and Topology.IsInstance(cell, "cell"):
                rooms.append(cell)
            elif includeOrphanedRooms and Topology.IsInstance(cell, "topology"):
                orphaned_rooms.append(cell)
            if cell:
                if includeFaces and host_faces:
                    faces.extend(host_faces)
                if includeShades and sh_faces:
                    shades.extend(sh_faces)
                if includeApertures and ap_faces:
                    apertures.extend(ap_faces)
                if includeDoors and dr_faces:
                    doors.extend(dr_faces)

        # Explicit orphaned faces → 'orphaned_faces'
        if includeOrphanedFaces:
            explicit_orphaned_faces = dictionary.get("orphaned_faces") or dictionary.get("OrphanedFaces") or []
            for f in explicit_orphaned_faces:
                hf = _build_host_face_cut_holes(f)
                if hf:
                    orphaned_faces.extend(hf)
            # Some files also place planar surfaces at top-level 'faces'
            for fobj in (dictionary.get("faces") or dictionary.get("Faces") or []):
                hf = _build_host_face_cut_holes(fobj)
                if hf:
                    orphaned_faces.extend(hf)
        
        # Explicit orphaned shades (and/or context shades)
        if includeOrphanedShades:
            explicit_orphaned_shades = dictionary.get("orphaned_shades") or dictionary.get("OrphanedShades") or []
            for s in explicit_orphaned_shades:
                hf = _build_host_face_cut_holes(s)
                if hf:
                    orphaned_shades.extend(hf)

            ctx = dictionary.get("context_geometry") or dictionary.get("Context") or {}
            shade_list = []
            if isinstance(ctx, dict):
                shade_list = ctx.get("shades") or ctx.get("Shades") or []
            elif isinstance(ctx, list):
                shade_list = ctx
            for s in shade_list:
                hf = _build_host_face_cut_holes(s)
                if hf:
                    orphaned_shades.extend(hf)
            # Some files might also place planar shade surfaces at top-level 'shades'
            for fobj in (dictionary.get("shades") or dictionary.get("Shades") or []):
                hf = _build_host_face_cut_holes(fobj)
                if hf:
                    orphaned_shades.extend(hf)

        # Explicit orphaned apertures → 'orphaned_apertures'
        if includeOrphanedApertures:
            orphaned_ap_list = dictionary.get("orphaned_apertures") or dictionary.get("OrphanedApertures") or []
            if orphaned_ap_list:
                orphaned_apertures.extend(_orphaned_aperture_faces(orphaned_ap_list))
        
         # Explicit orphaned doors → 'orphaned_doors'
        if includeOrphanedDoors:
            orphaned_dr_list = dictionary.get("orphaned_doors") or dictionary.get("OrphanedDoors") or []
            if orphaned_dr_list:
                orphaned_doors.extend(_orphaned_aperture_faces(orphaned_dr_list)) #You can use the same function as apertures.

        # Properties → hierarchical dict verbatim
        props_root = dictionary.get("properties") or dictionary.get("Properties") or {}
        properties = {
            "radiance": props_root.get("radiance") or props_root.get("Radiance") or {},
            "energy": props_root.get("energy") or props_root.get("Energy") or {},
        }

        return {
            "rooms": rooms,
            "faces": faces,
            "shades": shades,
            "apertures": apertures,
            "doors": doors,
            "orphanedRooms": orphaned_rooms,
            "orphanedFaces": orphaned_faces,
            "orphanedShades": orphaned_shades,
            "orphanedApertures": orphaned_apertures,
            "orphanedDoors": orphaned_doors,
            "properties": properties
        }
    
    @staticmethod
    def ByHBJSONPath(
        path: str,
        includeRooms: bool = True,
        includeFaces: bool = True,
        includeShades: bool = True,
        includeApertures: bool = True,
        includeDoors: bool = True,
        includeOrphanedRooms: bool = True,
        includeOrphanedFaces: bool = True,
        includeOrphanedShades: bool = True,
        includeOrphanedApertures: bool = True,
        includeOrphanedDoors: bool = True,
        tolerance: float = 0.0001,
        silent: bool = False):
        """
        Import an HBJSON model from a file path and return a python dictionary. See: https://github.com/ladybug-tools/honeybee-schema/wiki/1.1-Model-Schema
        
        Parameters
        ----------
        dictionary : dict
            The HBJSON model as a Python dictionary (e.g., loaded via ``json.load``).
        includeRooms : bool, optional
            If True, parse rooms and attempt to create one ``Cell`` per room. Default is True.
        includeFaces : bool, optional
            If True, include top-level planar faces found outside rooms (e.g., at root "faces"). Default is True.
        includeShades : bool, optional
            If True, include context/standalone shades (e.g., ``context_geometry.shades``). Default is True.
        includeApertures : bool, optional
            If True, include **room** apertures (e.g., windows) as separate ``Face`` objects (not cut from hosts). Default is True.
        includeDoors : bool, optional
            If True, include **room** doors as separate ``Face`` objects (not cut from hosts). Default is True.
        includeOrphanedRooms : bool, optional
            If True, include the topology of the room when a room fails to close as a ``Cell``. This may be a ``Shell`` or a ``Cluster``. Default is True.
        includeOrphanedFaces : bool, optional
            If True, include planar faces listed at the HBJSON root (e.g., "faces"). Default is True.
        includeOrphanedShades : bool, optional
            If True, include shades listed at the HBJSON root (e.g., "orphaned_shades"). Default is True.
        includeOrphanedApertures : bool, optional
            If True, include apertures listed at the HBJSON root (e.g., "orphaned_apertures"). Default is True.
        includeOrphanedDoors : bool, optional
            If True, include doors listed at the HBJSON root (e.g., "orphaned_doors"). Default is True.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
        Returns
        -------
        dict
            The created cluster of vertices, edges, faces, and cells.
            - 'rooms': list of Cells (one per successfully closed room)
            - 'faces': list of Faces (all faces that make up the rooms)
            - 'shades': list of Faces (all shade faces)
            - 'apertures': list of Faces (all apertures, never cut from hosts)
            - 'doors': list of Faces (all doors, never cut from hosts)
            - 'orphanedRooms': list of Topologies (context/top-level topologies (e.g. Shells or Clustser) that failed to form a Cell)
            - 'orphanedFaces': list of Faces (context/top-level faces + host faces of rooms that failed to form a Cell)
            - 'orphanedShades': list of Faces (context/top-level shade faces that failed to have a parent cell)
            - 'orphanedApertures': list of Faces (apertures that failed to have a parent face)
            - 'orphanedDoors': list of Faces (doors that failed to have a parent face)
            - 'properties': hierarchical dict copied verbatim from HBJSON['properties']
        """

        import json
        if not path:
            if not silent:
                print("Honeybee.ByHBJSONPath - Error: the input path parameter is not a valid path. Returning None.")
            return None
        with open(path) as file:
            try:
                hbjson_dict = json.load(file)
            except:
                if not silent:
                    print("Honeybee.ByHBJSONPath - Error: Could not open the HBJSON file. Returning None.")
                    return None
        return Honeybee.ByHBJSONDictionary(hbjson_dict,
                                             includeRooms = includeRooms,
                                             includeFaces = includeFaces,
                                             includeShades = includeShades,
                                             includeApertures = includeApertures,
                                             includeDoors = includeDoors,
                                             includeOrphanedRooms = includeOrphanedRooms,
                                             includeOrphanedFaces = includeOrphanedFaces,
                                             includeOrphanedShades = includeOrphanedShades,
                                             includeOrphanedApertures = includeOrphanedApertures,
                                             includeOrphanedDoors = includeOrphanedDoors,
                                             tolerance = tolerance,
                                             silent = silent)
    
    @staticmethod
    def ByHBJSONString(
        string,
        includeRooms: bool = True,
        includeFaces: bool = True,
        includeShades: bool = True,
        includeApertures: bool = True,
        includeDoors: bool = True,
        includeOrphanedRooms: bool = True,
        includeOrphanedFaces: bool = True,
        includeOrphanedShades: bool = True,
        includeOrphanedApertures: bool = True,
        includeOrphanedDoors: bool = True,
        tolerance: float = 0.0001,
        silent: bool = False):
        """
        Import an HBJSON model from a file path and return a python dictionary. See: https://github.com/ladybug-tools/honeybee-schema/wiki/1.1-Model-Schema
        
        Parameters
        ----------
        string : str
            The HBJSON model as a string.
        includeRooms : bool, optional
            If True, parse rooms and attempt to create one ``Cell`` per room. Default is True.
        includeFaces : bool, optional
            If True, include top-level planar faces found outside rooms (e.g., at root "faces"). Default is True.
        includeShades : bool, optional
            If True, include context/standalone shades (e.g., ``context_geometry.shades``). Default is True.
        includeApertures : bool, optional
            If True, include **room** apertures (e.g., windows) as separate ``Face`` objects (not cut from hosts). Default is True.
        includeDoors : bool, optional
            If True, include **room** doors as separate ``Face`` objects (not cut from hosts). Default is True.
        includeOrphanedRooms : bool, optional
            If True, include the topology of the room when a room fails to close as a ``Cell``. This may be a ``Shell`` or a ``Cluster``. Default is True.
        includeOrphanedFaces : bool, optional
            If True, include planar faces listed at the HBJSON root (e.g., "faces"). Default is True.
        includeOrphanedShades : bool, optional
            If True, include shades listed at the HBJSON root (e.g., "orphaned_shades"). Default is True.
        includeOrphanedApertures : bool, optional
            If True, include apertures listed at the HBJSON root (e.g., "orphaned_apertures"). Default is True.
        includeOrphanedDoors : bool, optional
            If True, include doors listed at the HBJSON root (e.g., "orphaned_doors"). Default is True.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
        Returns
        -------
        dict
            The created cluster of vertices, edges, faces, and cells.
            - 'rooms': list of Cells (one per successfully closed room)
            - 'faces': list of Faces (all faces that make up the rooms)
            - 'shades': list of Faces (all shade faces)
            - 'apertures': list of Faces (all apertures, never cut from hosts)
            - 'doors': list of Faces (all doors, never cut from hosts)
            - 'orphanedRooms': list of Topologies (context/top-level topologies (e.g. Shells or Clustser) that failed to form a Cell)
            - 'orphanedFaces': list of Faces (context/top-level faces + host faces of rooms that failed to form a Cell)
            - 'orphanedShades': list of Faces (context/top-level shade faces that failed to have a parent cell)
            - 'orphanedApertures': list of Faces (apertures that failed to have a parent face)
            - 'orphanedDoors': list of Faces (doors that failed to have a parent face)
            - 'properties': hierarchical dict copied verbatim from HBJSON['properties']
        """

        if not isinstance(string, str):
            if not silent:
                print("Honeybee.ByHBJSONString - Error: The input string parameter is not a valid string. Returning None.")
            return None
        hbjson_dict = json.loads(string)
        if not isinstance(hbjson_dict, dict):
            if not silent:
                print("Honeybee.ByHBJSONString - Error: Could not convert the input string into a valid HBJSON dictionary. Returning None.")
            return None
        return Honeybee.ByHBJSONDictionary(hbjson_dict,
                                           includeRooms = includeRooms,
                                           includeFaces = includeFaces,
                                           includeShades = includeShades,
                                           includeApertures = includeApertures,
                                           includeDoors = includeDoors,
                                           includeOrphanedRooms = includeOrphanedRooms,
                                           includeOrphanedFaces = includeOrphanedFaces,
                                           includeOrphanedShades = includeOrphanedShades,
                                           includeOrphanedApertures = includeOrphanedApertures,
                                           includeOrphanedDoors = includeOrphanedDoors,
                                           tolerance = tolerance,
                                           silent = silent)
    
    @staticmethod
    def ConstructionSetByIdentifier(id):
        """
        Returns the built-in construction set by the input identifying string.

        Parameters
        ----------
        id : str
            The construction set identifier.

        Returns
        -------
        HBConstructionSet
            The found built-in construction set.

        """
        return constr_set_lib.construction_set_by_identifier(id)
    
    @staticmethod
    def ConstructionSets():
        """
        Returns the list of built-in construction sets

        Returns
        -------
        list
            The list of built-in construction sets.

        """
        constrSets = []
        constrIdentifiers = list(constr_set_lib.CONSTRUCTION_SETS)
        for constrIdentifier in constrIdentifiers: 
            constrSets.append(constr_set_lib.construction_set_by_identifier(constrIdentifier))
        return [constrSets, constrIdentifiers]
    
    @staticmethod
    def ExportToHBJSON(model, path, overwrite=False):
        """
        Exports the input HB Model to a file.

        Parameters
        ----------
        model : HBModel
            The input HB Model.
        path : str
            The location of the output file.
        overwrite : bool , optional
            If set to True this method overwrites any existing file. Otherwise, it won't. Default is False.

        Returns
        -------
        bool
            Returns True if the operation is successful. Returns False otherwise.

        """
        from os.path import exists

        # Make sure the file extension is .hbjson
        ext = path[len(path)-7:len(path)]
        if ext.lower() != ".hbjson":
            path = path+".hbjson"
        
        if not overwrite and exists(path):
            print("Honeybee.ExportToHBJSON - Error: a file already exists at the specified path and overwrite is set to False. Returning None.")
            return None
        f = None
        try:
            if overwrite == True:
                f = open(path, "w")
            else:
                f = open(path, "x") # Try to create a new File
        except:
            print("Honeybee.ExportToHBJSON - Error: Could not create a new file at the following location: "+path+". Returning None.")
            return None
        if (f):
            json.dump(model.to_dict(), f, indent=4)
            f.close()    
            return True
        return False
    
    @staticmethod
    def ModelByTopology(tpBuilding,
                tpShadingFacesCluster = None,
                buildingName: str = "Generic_Building",
                defaultProgramIdentifier: str = "Generic Office Program",
                defaultConstructionSetIdentifier: str = "Default Generic Construction Set",
                coolingSetpoint: float = 25.0,
                heatingSetpoint: float = 20.0,
                humidifyingSetpoint: float = 30.0,
                dehumidifyingSetpoint: float = 55.0,
                roomNameKey: str = "TOPOLOGIC_name",
                roomTypeKey: str = "TOPOLOGIC_type",
                apertureTypeKey: str = "TOPOLOGIC_type",
                addSensorGrid: bool = False,
                mantissa: int = 6):
        """
        Creates an HB Model from the input Topology.

        Parameters
        ----------
        tpBuilding : topologic_core.CellComplex or topologic_core.Cell
            The input building topology.
        tpShadingFaceCluster : topologic_core.Cluster , optional
            The input cluster for shading faces. Default is None.
        buildingName : str , optional
            The desired name of the building. Default is "Generic_Building".
        defaultProgramIdentifier: str , optional
            The desired default program identifier. Default is "Generic Office Program".
        defaultConstructionSetIdentifier: str , optional
            The desired default construction set identifier. Default is "Default Generic Construction Set".
        coolingSetpoint : float , optional
            The desired HVAC cooling set point in degrees Celsius. Default is 25.
        heatingSetpoint : float , optional
            The desired HVAC heating set point in degrees Celsius. Default is 20.
        humidifyingSetpoint : float , optional
            The desired HVAC humidifying set point in percentage. Default is 55.
        roomNameKey : str , optional
            The dictionary key under which the room name is stored. Default is "TOPOLOGIC_name".
        roomTypeKey : str , optional
            The dictionary key under which the room type is stored. Default is "TOPOLOGIC_type".
        apertureTypeKey : str , optional
            The dictionary key under which the aperture type is stored. Default is "TOPOLOGIC_type".
        addSensorGrid : bool , optional
            If set to True a sensor grid is add to horizontal surfaces. Default is False.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        
        Returns
        -------
        HBModel
            The created HB Model

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Cell import Cell
        from topologicpy.Aperture import Aperture
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        def cellFloor(cell):
            faces = Topology.Faces(cell)
            c = [Vertex.Z(Topology.CenterOfMass(x)) for x in faces]
            return round(min(c),2)

        def floorLevels(cells, min_difference):
            floors = [cellFloor(x) for x in cells]
            floors = list(set(floors)) #create a unique list
            floors.sort()
            returnList = []
            for aCell in cells:
                for floorNumber, aFloor in enumerate(floors):
                    if abs(cellFloor(aCell) - aFloor) > min_difference:
                        continue
                    returnList.append("Floor"+str(floorNumber).zfill(2))
                    break
            return returnList

        def getKeyName(d, keyName):
            if not d == None:
                keys = Dictionary.Keys(d)
            else:
                keys = []
            for key in keys:
                if key.lower() == keyName.lower():
                    return key
            return None

        def createUniqueName(name, nameList, number):
            if not (name in nameList):
                return name
            elif not ((name+"_"+str(number)) in nameList):
                return name+"_"+str(number)
            else:
                return createUniqueName(name,nameList, number+1)
        
        if not Topology.IsInstance(tpBuilding, "Topology"):
            return None
        rooms = []
        tpCells = Topology.Cells(tpBuilding)
        # Sort cells by Z Levels
        tpCells.sort(key=lambda c: cellFloor(c), reverse=False)
        fl = floorLevels(tpCells, 2)
        spaceNames = []
        sensorGrids = []
        for spaceNumber, tpCell in enumerate(tpCells):
            tpDictionary = Topology.Dictionary(tpCell)
            tpCellName = "Untitled Space"
            tpCellStory = fl[0]
            tpCellProgramIdentifier = None
            tpCellConstructionSetIdentifier = None
            tpCellConditioned = True
            if tpDictionary:
                keyName = getKeyName(tpDictionary, 'Story')
                tpCellStory = Dictionary.ValueAtKey(tpDictionary, keyName, silent=True) or fl[spaceNumber]
                if tpCellStory:
                    tpCellStory = tpCellStory.replace(" ","_")
                else:
                    tpCellStory = "Untitled_Floor"
                if roomNameKey:
                    keyName = getKeyName(tpDictionary, roomNameKey)
                else:
                    keyName = getKeyName(tpDictionary, 'Name')
                tpCellName = Dictionary.ValueAtKey(tpDictionary,keyName, silent=True) or createUniqueName(tpCellStory+"_SPACE_"+(str(spaceNumber+1)), spaceNames, 1)
                if roomTypeKey:
                    keyName = getKeyName(tpDictionary, roomTypeKey)
                try:
                    tpCellProgramIdentifier = Dictionary.ValueAtKey(tpDictionary, keyName, silent=True)
                    if tpCellProgramIdentifier:
                        program = prog_type_lib.program_type_by_identifier(tpCellProgramIdentifier)
                    elif defaultProgramIdentifier:
                        program = prog_type_lib.program_type_by_identifier(defaultProgramIdentifier)
                except:
                    program = prog_type_lib.office_program #Default Office Program as a last resort
                keyName = getKeyName(tpDictionary, 'construction_set')
                try:
                    tpCellConstructionSetIdentifier = Dictionary.ValueAtKey(tpDictionary, keyName, silent=True)
                    if tpCellConstructionSetIdentifier:
                        constr_set = constr_set_lib.construction_set_by_identifier(tpCellConstructionSetIdentifier)
                    elif defaultConstructionSetIdentifier:
                        constr_set = constr_set_lib.construction_set_by_identifier(defaultConstructionSetIdentifier)
                except:
                    constr_set = constr_set_lib.construction_set_by_identifier("Default Generic Construction Set")
            else:
                tpCellStory = fl[spaceNumber]
                tpCellName = str(tpCellStory)+"_SPACE_"+(str(spaceNumber+1))
                program = prog_type_lib.office_program
                constr_set = constr_set_lib.construction_set_by_identifier("Default Generic Construction Set")
            spaceNames.append(tpCellName)

            tpCellFaces = Topology.Faces(tpCell)
            if tpCellFaces:
                hbRoomFaces = []
                for tpFaceNumber, tpCellFace in enumerate(tpCellFaces):
                    tpCellFaceNormal = Face.Normal(tpCellFace, mantissa=mantissa)
                    hbRoomFacePoints = []
                    tpFaceVertices = Topology.Vertices(Face.ExternalBoundary(tpCellFace))
                    for tpVertex in tpFaceVertices:
                        hbRoomFacePoints.append(Point3D(Vertex.X(tpVertex, mantissa=mantissa), Vertex.Y(tpVertex, mantissa=mantissa), Vertex.Z(tpVertex, mantissa=mantissa)))
                    hbRoomFace = HBFace(tpCellName+'_Face_'+str(tpFaceNumber+1), Face3D(hbRoomFacePoints))
                    tpFaceApertures = []
                    tpFaceApertures = Topology.Apertures(tpCellFace)
                    if tpFaceApertures:
                        for tpFaceApertureNumber, apertureTopology in enumerate(tpFaceApertures):
                            tpFaceApertureDictionary = Topology.Dictionary(apertureTopology)
                            if tpFaceApertureDictionary:
                                apertureKeyName = getKeyName(tpFaceApertureDictionary, apertureTypeKey)
                                tpFaceApertureType = Dictionary.ValueAtKey(tpFaceApertureDictionary,apertureKeyName, silent=True)
                            hbFaceAperturePoints = []
                            tpFaceApertureVertices = []
                            tpFaceApertureVertices = Topology.Vertices(Face.ExternalBoundary(apertureTopology), silent=True)
                            for tpFaceApertureVertex in tpFaceApertureVertices:
                                hbFaceAperturePoints.append(Point3D(Vertex.X(tpFaceApertureVertex, mantissa=mantissa), Vertex.Y(tpFaceApertureVertex, mantissa=mantissa), Vertex.Z(tpFaceApertureVertex, mantissa=mantissa)))
                            if(tpFaceApertureType):
                                if ("door" in tpFaceApertureType.lower()):
                                    hbFaceAperture = HBDoor(tpCellName+'_Face_'+str(tpFaceNumber+1)+'_Door_'+str(tpFaceApertureNumber), Face3D(hbFaceAperturePoints))
                                else:
                                    hbFaceAperture = HBAperture(tpCellName+'_Face_'+str(tpFaceNumber+1)+'_Window_'+str(tpFaceApertureNumber), Face3D(hbFaceAperturePoints))
                            else:
                                hbFaceAperture = HBAperture(tpCellName+'_Face_'+str(tpFaceNumber+1)+'_Window_'+str(tpFaceApertureNumber), Face3D(hbFaceAperturePoints))
                            hbRoomFace.add_aperture(hbFaceAperture)
                    else:
                        tpFaceDictionary = Topology.Dictionary(tpCellFace)
                        if (abs(tpCellFaceNormal[2]) < 1e-6) and tpFaceDictionary: #It is a mostly vertical wall and has a dictionary
                            apertureRatio = Dictionary.ValueAtKey(tpFaceDictionary,'apertureRatio', silent=True)
                            if apertureRatio:
                                hbRoomFace.apertures_by_ratio(apertureRatio, tolerance=0.01)
                    fType = honeybee.facetype.get_type_from_normal(Vector3D(tpCellFaceNormal[0],tpCellFaceNormal[1],tpCellFaceNormal[2]), roof_angle=30, floor_angle=150)
                    hbRoomFace.type = fType
                    hbRoomFaces.append(hbRoomFace)
                room = HBRoom(tpCellName, hbRoomFaces, 0.01, 1)
                if addSensorGrid:
                    floor_mesh = room.generate_grid(0.5, 0.5, 1)
                    sensorGrids.append(SensorGrid.from_mesh3d(tpCellName+"_SG", floor_mesh))
                heat_setpt = ScheduleRuleset.from_constant_value('Room Heating', heatingSetpoint, schedule_types.temperature)
                cool_setpt = ScheduleRuleset.from_constant_value('Room Cooling', coolingSetpoint, schedule_types.temperature)
                humidify_setpt = ScheduleRuleset.from_constant_value('Room Humidifying', humidifyingSetpoint, schedule_types.humidity)
                dehumidify_setpt = ScheduleRuleset.from_constant_value('Room Dehumidifying', dehumidifyingSetpoint, schedule_types.humidity)
                setpoint = Setpoint('Room Setpoint', heat_setpt, cool_setpt, humidify_setpt, dehumidify_setpt)
                simple_office = ScheduleDay('Simple Weekday', [0, 1, 0], [Time(0, 0), Time(9, 0), Time(17, 0)]) #Todo: Remove hardwired scheduleday
                schedule = ScheduleRuleset('Office Water Use', simple_office, None, schedule_types.fractional) #Todo: Remove hardwired schedule
                shw = ServiceHotWater('Office Hot Water', 0.1, schedule) #Todo: Remove hardwired schedule hot water
                room.properties.energy.program_type = program
                room.properties.energy.construction_set = constr_set
                room.properties.energy.add_default_ideal_air() #Ideal Air Exchange
                room.properties.energy.setpoint = setpoint #Heating/Cooling/Humidifying/Dehumidifying
                room.properties.energy.service_hot_water = shw #Service Hot Water
                if tpCellStory:
                    room.story = tpCellStory
                rooms.append(room)
        HBRoom.solve_adjacency(rooms, 0.01)

        hbShades = []
        if(tpShadingFacesCluster):
            hbShades = []
            tpShadingFaces = Topology.SubTopologies(tpShadingFacesCluster, subTopologyType="face")
            for faceIndex, tpShadingFace in enumerate(tpShadingFaces):
                faceVertices = []
                faceVertices = Topology.Vertices(Face.ExternalBoundary(tpShadingFace), silent=True)
                facePoints = []
                for aVertex in faceVertices:
                    facePoints.append(Point3D(Vertex.X(aVertex, mantissa=mantissa), Vertex.Y(aVertex, mantissa=mantissa), Vertex.Z(aVertex, mantissa=mantissa)))
                hbShadingFace = Face3D(facePoints, None, [])
                hbShade = HBShade("SHADINGSURFACE_" + str(faceIndex+1), hbShadingFace)
                hbShades.append(hbShade)
        model = HBModel(buildingName, rooms, orphaned_shades=hbShades)
        if addSensorGrid:
            model.properties.radiance.sensor_grids = []
            model.properties.radiance.add_sensor_grids(sensorGrids)
        return model
    
    @staticmethod
    def ProgramTypeByIdentifier(id):
        """
        Returns the program type by the input identifying string.

        Parameters
        ----------
        id : str
            The identifiying string.

        Returns
        -------
        HBProgram
            The found built-in program.

        """
        return prog_type_lib.program_type_by_identifier(id)
    
    @staticmethod
    def ProgramTypes():
        """
        Returns the list of available built-in program types.

        Returns
        -------
        list
            The list of available built-in program types.

        """
        progTypes = []
        progIdentifiers = list(prog_type_lib.PROGRAM_TYPES)
        for progIdentifier in progIdentifiers: 
            progTypes.append(prog_type_lib.program_type_by_identifier(progIdentifier))
        return [progTypes, progIdentifiers]
    
    @staticmethod
    def String(model):
        """
        Returns the string representation of the input model.

        Parameters
        ----------
        model : HBModel
            The input HB Model.

        Returns
        -------
        dict
            A dictionary representing the input HB Model.

        """
        return model.to_dict()