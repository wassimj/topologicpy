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

from __future__ import annotations
import os
import warnings
import json

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
        try:
            import honeybee_energy.lib.constructionsets as constr_set_lib
        except ImportError as exc:
            raise ImportError('Honeybee requires Honeybee/Ladybug packages. Install it in this environment with: python -m pip install "topologicpy[honeybee]"') from exc
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
        try:
            import honeybee_energy.lib.constructionsets as constr_set_lib
        except ImportError as exc:
            raise ImportError('Honeybee requires Honeybee/Ladybug packages. Install it in this environment with: python -m pip install "topologicpy[honeybee]"') from exc
        constrSets = []
        constrIdentifiers = list(constr_set_lib.CONSTRUCTION_SETS)
        for constrIdentifier in constrIdentifiers: 
            constrSets.append(constr_set_lib.construction_set_by_identifier(constrIdentifier))
        return [constrSets, constrIdentifiers]
    
    @staticmethod
    def ExportToGBXML(model, path, overwrite=False, writeMapping=True):
        """Write and validate direct Honeybee gbXML, optionally with an ID-only JSON map.

        Returns a validation report. Existing files are protected by default.
        """
        import tempfile
        from pathlib import Path
        target = Path(path).absolute()
        if target.suffix.lower() != '.xml':
            target = Path(str(target) + '.xml') if target.suffix.lower() != '.gbxml' else target
        mapping = target.with_suffix('.json')
        destinations = [target] + ([mapping] if writeMapping else [])
        if any(p.is_symlink() for p in destinations):
            raise ValueError('Cannot overwrite symbolic links.')
        if not overwrite and any(p.exists() for p in destinations):
            raise FileExistsError('gbXML or mapping already exists.')
        with tempfile.TemporaryDirectory(prefix='hb-gbxml-', dir=target.parent) as folder:
            staged = Path(folder) / target.name
            report = _export_direct_gbxml(model, staged)
            sources = [staged] + ([staged.with_suffix('.json')] if writeMapping else [])
            backups = {p: p.read_bytes() if p.exists() else None for p in destinations}
            committed = []
            try:
                for source, destination in zip(sources, destinations):
                    if overwrite:
                        os.replace(source, destination)
                    else:
                        os.link(source, destination)
                    committed.append(destination)
            except BaseException:
                for destination in reversed(committed):
                    if backups[destination] is None:
                        destination.unlink(missing_ok=True)
                    else:
                        restore = Path(folder) / (destination.name + '.restore')
                        restore.write_bytes(backups[destination])
                        os.replace(restore, destination)
                raise
        return report

    @staticmethod
    def ValidateGBXML(path, model=None):
        """Validate the delivered XML geometry independently of OpenStudio."""
        return validate_gbxml(path, model)

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
    def _FitSubFaceGeometry(geometry, host, tolerance):
        """Fit boundary-touching sub-faces with recorded, bounded corrections."""
        import math
        from ladybug_geometry.geometry3d.pointvector import Vector3D
        if geometry.normal.dot(host.normal) < 0:
            geometry = geometry.flip()
        from shapely.geometry import Polygon
        def project(points):
            return [(p.x,p.y) for p in (host.plane.xyz_to_xy(v) for v in points)]
        outer=Polygon(project(host.boundary),[project(h) for h in host.holes] if host.holes else [])
        aperture=Polygon(project(geometry.boundary))
        clearance=max(2e-7,min(tolerance*.2,1e-5))
        def robustly_inside(candidate):
            polygon=Polygon(project(candidate.boundary))
            return (outer.covers(polygon) and polygon.boundary.distance(outer.boundary)>=clearance*.5
                    and host.is_sub_face(candidate,tolerance,math.radians(1)))
        if robustly_inside(geometry):
            return geometry, None
        centre, target = geometry.centroid, host.centroid
        direction = Vector3D(target.x-centre.x, target.y-centre.y, target.z-centre.z)
        direction = direction - host.normal * direction.dot(host.normal)
        if direction.magnitude > 1e-12:
            direction = direction.normalize()
            for multiplier in (2, 4, 8, 16, 32, 64):
                distance = tolerance*multiplier
                if distance > 0.001:
                    break
                vector = direction*distance
                candidate = geometry.move(vector)
                if robustly_inside(candidate):
                    return candidate, {'method': 'area_preserving_translation', 'reason': 'opening touches host boundary',
                        'translation_m': [vector.x, vector.y, vector.z], 'distance_m': distance,
                        'original_area_m2': geometry.area, 'export_area_m2': candidate.area}
        # A sub-face can touch opposing host edges, so no rigid translation
        # can put it strictly inside. Apply only a micron-scale, audited inset
        # when the original polygon is already covered by its coplanar host.
        from shapely.geometry import Polygon
        from ladybug_geometry.geometry2d.pointvector import Point2D
        from ladybug_geometry.geometry3d.face import Face3D
        def project(points):
            return [(p.x,p.y) for p in (host.plane.xyz_to_xy(v) for v in points)]
        outer=Polygon(project(host.boundary),[project(h) for h in host.holes] if host.holes else [])
        aperture=Polygon(project(geometry.boundary))
        if host.plane.is_coplanar_tolerance(geometry.plane,tolerance,math.radians(1)) and aperture.difference(outer.buffer(tolerance)).area <= 1e-10:
            distance=max(2e-7,min(tolerance*.2,1e-5))
            fitted=aperture.intersection(outer.buffer(-distance,join_style=2))
            if fitted.geom_type=='Polygon' and not fitted.is_empty and not fitted.interiors:
                loss=geometry.area-fitted.area
                if 0 <= loss <= max(1e-9,geometry.area*.001):
                    candidate=Face3D(tuple(host.plane.xy_to_xyz(Point2D(x,y)) for x,y in list(fitted.exterior.coords)[:-1]))
                    if candidate.normal.dot(host.normal)<0:candidate=candidate.flip()
                    if host.is_sub_face(candidate,tolerance,math.radians(1)):
                        return candidate,{'method':'bounded_boundary_inset','distance_m':distance,
                            'original_area_m2':geometry.area,'export_area_m2':candidate.area,
                            'area_change_m2':candidate.area-geometry.area,'relative_area_change':abs(candidate.area-geometry.area)/geometry.area}
        raise ValueError('Honeybee.ModelByTopology - Opening cannot fit its host within the bounded export correction budget.')

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
                mantissa: int = 6,
                tolerance: float = 0.00001,
                idKey: str = "tpy_id",
                nameKey: str = "TOPOLOGIC_name"):
        """
        Creates an HB Model from the input Topology.

        Parameters
        ----------
        tpBuilding : CellComplex or Cell
            The input building topology.
        tpShadingFaceCluster : Cluster , optional
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
        tolerance : float , optional
            Geometry and adjacency tolerance in metres. Default is 0.00001.
            It must resolve the smallest retained analytical features.
        
        Returns
        -------
        HBModel
            The created HB Model

        """
        try:
            import honeybee.facetype
            from honeybee.face import Face as HBFace
            from honeybee.model import Model as HBModel
            from honeybee.room import Room as HBRoom
            from honeybee.shade import Shade as HBShade
            from honeybee.aperture import Aperture as HBAperture
            from honeybee.door import Door as HBDoor
            import honeybee_energy.lib.constructionsets as constr_set_lib
            import honeybee_energy.lib.programtypes as prog_type_lib
            import honeybee_energy.lib.scheduletypelimits as schedule_types
            from honeybee_energy.schedule.ruleset import ScheduleRuleset
            from honeybee_energy.schedule.day import ScheduleDay
            from honeybee_energy.load.setpoint import Setpoint
            from honeybee_energy.load.hotwater import ServiceHotWater
            from honeybee_radiance.sensorgrid import SensorGrid
            from ladybug.dt import Time
            from ladybug_geometry.geometry3d.face import Face3D
            from ladybug_geometry.geometry3d.pointvector import Point3D, Vector3D
        except ImportError as exc:
            raise ImportError('Honeybee requires Honeybee/Ladybug packages. Install it in this environment with: python -m pip install "topologicpy[honeybee]"') from exc
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire
        from topologicpy.Face import Face
        from topologicpy.Cell import Cell
        from topologicpy.Aperture import Aperture
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from topologicpy.EnergyModel import EnergyModel
        import math
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError('Honeybee.ModelByTopology - tolerance must be positive and finite.')

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
        if roomNameKey == "TOPOLOGIC_name": roomNameKey = nameKey
        # Keep native topology intact while generating valid Honeybee identifiers.
        from honeybee.typing import clean_string
        rooms = []
        tpCells = [tpBuilding] if Topology.IsInstance(tpBuilding, "Cell") else Topology.Cells(tpBuilding)
        lowest_level = min(Vertex.Z(v, mantissa=mantissa) for v in Topology.Vertices(tpBuilding))
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
            displayName = str(tpCellName)
            try:
                identifier = clean_string(displayName[:60])
            except AssertionError:
                identifier = "Space"
            tpCellName = createUniqueName(identifier, spaceNames, 1)
            spaceNames.append(tpCellName)

            tpCellFaces = Topology.Faces(tpCell)
            export_normals = EnergyModel._ExportFaceNormals(tpCellFaces) if tpCellFaces else []
            if tpCellFaces:
                hbRoomFaces = []
                native_face_sources = []
                for tpFaceNumber, tpCellFace in enumerate(tpCellFaces):
                    tpCellFaceNormal = export_normals[tpFaceNumber]
                    hbRoomFacePoints = []
                    tpFaceVertices = Wire.Vertices(Face.ExternalBoundary(tpCellFace))
                    for tpVertex in tpFaceVertices:
                        hbRoomFacePoints.append(Point3D(Vertex.X(tpVertex, mantissa=max(12, mantissa)), Vertex.Y(tpVertex, mantissa=max(12, mantissa)), Vertex.Z(tpVertex, mantissa=max(12, mantissa))))
                    hb_holes = [[Point3D(*Vertex.Coordinates(v,mantissa=max(12, mantissa))) for v in Wire.Vertices(w)] for w in Face.InternalBoundaries(tpCellFace) or []]
                    hb_geometry = Face3D(hbRoomFacePoints,holes=hb_holes or None)
                    if hb_geometry.normal.dot(Vector3D(*tpCellFaceNormal))<0:hb_geometry=hb_geometry.flip()
                    hbRoomFace = HBFace(tpCellName+'_Face_'+str(tpFaceNumber+1), hb_geometry)
                    from honeybee.boundarycondition import boundary_conditions
                    hbRoomFace.boundary_condition = boundary_conditions.outdoors
                    voids = []
                    seen_voids = set()
                    tpFaceApertures = Topology.Apertures(tpCellFace)
                    if tpFaceApertures:
                        from honeybee.boundarycondition import boundary_conditions
                        hbRoomFace.boundary_condition = boundary_conditions.outdoors
                        for tpFaceApertureNumber, apertureTopology in enumerate(tpFaceApertures):
                            tpFaceApertureType = None
                            tpFaceApertureDictionary = Topology.Dictionary(apertureTopology)
                            if tpFaceApertureDictionary:
                                apertureKeyName = getKeyName(tpFaceApertureDictionary, apertureTypeKey)
                                tpFaceApertureType = Dictionary.ValueAtKey(tpFaceApertureDictionary,apertureKeyName, silent=True) or Dictionary.ValueAtKey(tpFaceApertureDictionary, 'kind', silent=True)
                            hbFaceAperturePoints = []
                            tpFaceApertureVertices = []
                            tpFaceApertureVertices = Wire.Vertices(Face.ExternalBoundary(apertureTopology))
                            for tpFaceApertureVertex in tpFaceApertureVertices:
                                hbFaceAperturePoints.append(Point3D(Vertex.X(tpFaceApertureVertex, mantissa=max(12, mantissa)), Vertex.Y(tpFaceApertureVertex, mantissa=max(12, mantissa)), Vertex.Z(tpFaceApertureVertex, mantissa=max(12, mantissa))))
                            aperture_data=tpFaceApertureDictionary if isinstance(tpFaceApertureDictionary,dict) else Dictionary.PythonDictionary(tpFaceApertureDictionary) or {}
                            if EnergyModel._IsUnfilledOpening(aperture_data.get('kind') or tpFaceApertureType):
                                points = [tuple(p) for p in hbFaceAperturePoints]
                                key = tuple(sorted(points))
                                if key not in seen_voids:
                                    seen_voids.add(key); voids.append((points, aperture_data))
                                continue
                            projected, projection = EnergyModel._ProjectOpeningToHost(
                                [tuple(p) for p in hbFaceAperturePoints],
                                [tuple(p) for p in hbRoomFace.geometry.boundary], tolerance)
                            # Use the actual Ladybug host plane as the final reference.
                            planar_points = [hbRoomFace.geometry.plane.xy_to_xyz(
                                hbRoomFace.geometry.plane.xyz_to_xy(Point3D(*p))) for p in projected]
                            aperture_geometry, correction = Honeybee._FitSubFaceGeometry(Face3D(planar_points), hbRoomFace.geometry, tolerance)
                            if projection:
                                correction = ({**correction, 'host_plane_projection': projection}
                                              if correction else projection)
                            if(tpFaceApertureType):
                                if ("door" in tpFaceApertureType.lower()):
                                    hbFaceAperture = HBDoor(tpCellName+'_Face_'+str(tpFaceNumber+1)+'_Door_'+str(tpFaceApertureNumber), aperture_geometry)
                                else:
                                    hbFaceAperture = HBAperture(tpCellName+'_Face_'+str(tpFaceNumber+1)+'_Window_'+str(tpFaceApertureNumber), aperture_geometry)
                            else:
                                hbFaceAperture = HBAperture(tpCellName+'_Face_'+str(tpFaceNumber+1)+'_Window_'+str(tpFaceApertureNumber), aperture_geometry)
                            hbFaceAperture.user_data = {**EnergyModel._ExportMetadata(aperture_data, idKey, nameKey), 'bim2bem_opening_id':aperture_data.get('opening_id')}
                            if hbFaceAperture.user_data.get('tpy_name'): hbFaceAperture.display_name = hbFaceAperture.user_data['tpy_name']
                            if correction:
                                hbFaceAperture.user_data['topologicpy_geometry_correction'] = correction
                            duplicate=next((existing for existing in list(hbRoomFace.apertures)+list(hbRoomFace.doors)
                                if type(existing) is type(hbFaceAperture) and existing.geometry.is_geometrically_equivalent(hbFaceAperture.geometry,tolerance)),None)
                            if duplicate is not None:
                                data=duplicate.user_data or {}
                                ids=set(data.get('coincident_source_opening_ids',[]))
                                ids.update(i for i in [data.get('bim2bem_opening_id'),hbFaceAperture.user_data.get('bim2bem_opening_id')] if i)
                                data['coincident_source_opening_ids']=sorted(ids)
                                record=data.setdefault('topologicpy_geometry_correction',{'method':'coincident_source_opening_merge','original_area_m2':duplicate.area,'export_area_m2':duplicate.area})
                                record['coincident_source_opening_ids']=sorted(ids)
                                incoming = hbFaceAperture.user_data or {}
                                refs=set(data.get('tpy_ids',[])) | set(incoming.get('tpy_ids',[]))
                                refs.update(i for i in (data.get('tpy_id'),incoming.get('tpy_id')) if i)
                                if not data.get('tpy_id') and incoming.get('tpy_id'):
                                    data['tpy_id'] = incoming['tpy_id']
                                if not data.get('tpy_name') and incoming.get('tpy_name'):
                                    data['tpy_name'] = incoming['tpy_name']
                                if len(refs)>1: data['tpy_ids']=sorted(refs)
                                duplicate.user_data=data
                                if data.get('tpy_name'): duplicate.display_name = data['tpy_name']
                                continue
                            if isinstance(hbFaceAperture, HBDoor):
                                hbRoomFace.add_door(hbFaceAperture)
                            else:
                                hbRoomFace.add_aperture(hbFaceAperture)
                    else:
                        tpFaceDictionary = Topology.Dictionary(tpCellFace)
                        if (abs(tpCellFaceNormal[2]) < 1e-6) and tpFaceDictionary: #It is a mostly vertical wall and has a dictionary
                            apertureRatio = Dictionary.ValueAtKey(tpFaceDictionary,'apertureRatio', silent=True)
                            if apertureRatio:
                                hbRoomFace.apertures_by_ratio(apertureRatio, tolerance=0.01)
                    fType = honeybee.facetype.get_type_from_normal(Vector3D(tpCellFaceNormal[0],tpCellFaceNormal[1],tpCellFaceNormal[2]), roof_angle=30, floor_angle=150)
                    hbRoomFace.type = fType
                    if voids:
                        patches = EnergyModel._ExportAirBoundaryPatches(
                            [tuple(p) for p in hb_geometry.boundary],
                            [[tuple(p) for p in h] for h in hb_geometry.holes or []],
                            voids, tolerance=max(1e-7,min(tolerance,1e-5)))
                        new_faces = []
                        for i, (ring, air_data) in enumerate(patches):
                            geo = Face3D([Point3D(*p) for p in ring])
                            if geo.normal.dot(Vector3D(*tpCellFaceNormal)) < 0: geo = geo.flip()
                            face = HBFace(hbRoomFace.identifier+'_Patch_'+str(i+1), geo, fType)
                            if air_data is not None:
                                face.user_data = {**EnergyModel._ExportMetadata(air_data, idKey, nameKey), 'topologicpy_opening_kind':'void',
                                    'bim2bem_opening_id':air_data.get('opening_id')}
                            new_faces.append(face)
                        for sub in list(hbRoomFace.apertures)+list(hbRoomFace.doors):
                            i = EnergyModel._OpeningPatchIndex([tuple(p) for p in sub.geometry.boundary], patches)
                            copy = sub.duplicate()
                            (new_faces[i].add_door if isinstance(copy,HBDoor) else new_faces[i].add_aperture)(copy)
                        hbRoomFaces.extend(new_faces)
                        native_face_sources.extend([tpCellFace]*len(new_faces))
                    else:
                        hbRoomFaces.append(hbRoomFace)
                        native_face_sources.append(tpCellFace)
                room = HBRoom(tpCellName, hbRoomFaces, tolerance, 1)
                for hb_face, native_face in zip(room.faces, native_face_sources):
                    hb_face.type = honeybee.facetype.get_type_from_normal(hb_face.normal, roof_angle=30, floor_angle=150)
                    d = Topology.Dictionary(native_face)
                    d = d if isinstance(d, dict) else Dictionary.PythonDictionary(d) or {}
                    levels = [Vertex.Z(v, mantissa=mantissa) for v in Topology.Vertices(native_face)]
                    condition = EnergyModel._ExteriorBoundaryCondition(d, lowest_level, levels, hb_face.normal.z, tolerance, bool(hb_face.apertures or hb_face.doors or (hb_face.user_data or {}).get('topologicpy_opening_kind')))
                    hb_face.boundary_condition = boundary_conditions.ground if condition == 'Ground' else boundary_conditions.outdoors
                    record = d.get('analytical_record', {})
                    if isinstance(record, str):
                        try:
                            record = json.loads(record)
                        except (ValueError, TypeError):
                            record = {}
                    exposure = record.get('exposure', d.get('exposure', 'unknown'))
                    existing_data = hb_face.user_data or {}
                    native_metadata = EnergyModel._ExportMetadata(d, idKey, nameKey)
                    parent_id = native_metadata.get('tpy_id')
                    for sub in list(hb_face.apertures)+list(hb_face.doors):
                        if parent_id:
                            sub.user_data = {**(sub.user_data or {}), 'tpy_parent_id':parent_id}
                    source_data = ({'tpy_parent_id': parent_id}
                                   if existing_data.get('topologicpy_opening_kind') == 'void' else native_metadata)
                    display = existing_data.get('tpy_name') or source_data.get('tpy_name')
                    if display: hb_face.display_name = display
                    hb_face.user_data = {**source_data, **existing_data, 'topologicpy_boundary_id': d.get('bim2bem_boundary_id'),
                        'source_exposure': exposure,
                        'boundary_condition_basis': 'explicit_exposure' if exposure in {'ground', 'external_earth', 'earth', 'outdoors', 'external', 'air'} else 'basal_floor_assumption' if condition == 'Ground' else 'outdoor_assumption'}
                room.user_data={**EnergyModel._ExportMetadata(EnergyModel._ExportDictionary(tpCell), idKey, nameKey), 'bim2bem_space_id':Dictionary.ValueAtKey(tpDictionary,'bim2bem_space_id',silent=True)}
                room.display_name = room.user_data.get('tpy_name') or displayName
                if addSensorGrid:
                    floor_mesh = room.generate_grid(0.5, 0.5, 1)
                    sensorGrids.append(SensorGrid.from_mesh3d(tpCellName+"_SG", floor_mesh))
                heat_setpt = ScheduleRuleset.from_constant_value('Room_Heating', heatingSetpoint, schedule_types.temperature)
                cool_setpt = ScheduleRuleset.from_constant_value('Room_Cooling', coolingSetpoint, schedule_types.temperature)
                humidify_setpt = ScheduleRuleset.from_constant_value('Room_Humidifying', humidifyingSetpoint, schedule_types.humidity)
                dehumidify_setpt = ScheduleRuleset.from_constant_value('Room_Dehumidifying', dehumidifyingSetpoint, schedule_types.humidity)
                setpoint = Setpoint('Room_Setpoint', heat_setpt, cool_setpt, humidify_setpt, dehumidify_setpt)
                simple_office = ScheduleDay('Simple_Weekday', [0, 1, 0], [Time(0, 0), Time(9, 0), Time(17, 0)]) #Todo: Remove hardwired scheduleday
                schedule = ScheduleRuleset('Office_Water_Use', simple_office, None, schedule_types.fractional) #Todo: Remove hardwired schedule
                shw = ServiceHotWater('Office_Hot_Water', 0.1, schedule) #Todo: Remove hardwired schedule hot water
                room.properties.energy.program_type = program
                room.properties.energy.construction_set = constr_set
                room.properties.energy.add_default_ideal_air() #Ideal Air Exchange
                room.properties.energy.setpoint = setpoint #Heating/Cooling/Humidifying/Dehumidifying
                room.properties.energy.service_hot_water = shw #Service Hot Water
                if tpCellStory:
                    room.story = tpCellStory
                rooms.append(room)
        HBRoom.solve_adjacency(rooms, tolerance)
        air_faces = {face.identifier:face for room in rooms for face in room.faces
                     if (face.user_data or {}).get('topologicpy_opening_kind') == 'void'}
        air_construction = None
        if air_faces:
            from honeybee_energy.construction.air import AirBoundaryConstruction
            air_construction = AirBoundaryConstruction('TopologicPy_Unfilled_Opening', air_mixing_per_area=0.)
        void_approximations = []
        for room in rooms:
            for face in room.faces:
                if face.identifier in air_faces:
                    bc = face.boundary_condition
                    other = air_faces.get(bc.boundary_condition_objects[0]) if bc.name == 'Surface' else None
                    paired = (other is not None and other is not face and
                              other.boundary_condition.name == 'Surface' and
                              other.boundary_condition.boundary_condition_objects[0] == face.identifier)
                    if paired:
                        face.type = honeybee.facetype.AirBoundary()
                        face.properties.energy.construction = air_construction
                        face.user_data['topologicpy_void_treatment'] = 'paired_air_boundary'
                    else:
                        # Preserve the opening patch and its source ID, but do not
                        # invent glazing, a door, or an exterior air construction.
                        face.type = honeybee.facetype.get_type_from_normal(face.normal,roof_angle=30,floor_angle=150)
                        face.user_data['topologicpy_void_treatment'] = 'opaque_approximation'
                        void_approximations.append(EnergyModel._VoidApproximation(face.user_data, face.identifier, bc.name, face.area))
                else:
                    face.type = honeybee.facetype.get_type_from_normal(face.normal,roof_angle=30,floor_angle=150)
                if face.boundary_condition.name == 'Surface':
                    face.user_data['boundary_condition_basis'] = 'shared_geometry'
        if any((f.user_data or {}).get('topologicpy_void_treatment') == 'paired_air_boundary' for f in air_faces.values()):
            warnings.warn('Paired interior voids use air boundaries; specify inter-zone airflow before simulation.', RuntimeWarning)
        if void_approximations:
            warnings.warn(EnergyModel._VoidApproximationWarning(void_approximations), RuntimeWarning)

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
                hbShade.user_data = EnergyModel._ExportMetadata(EnergyModel._ExportDictionary(tpShadingFace), idKey, nameKey)
                if hbShade.user_data.get('tpy_name'): hbShade.display_name = hbShade.user_data['tpy_name']
                hbShades.append(hbShade)
        model = HBModel(buildingName, rooms, orphaned_shades=hbShades)
        model.user_data = EnergyModel._ExportMetadata(EnergyModel._ExportDictionary(tpBuilding), idKey, nameKey)
        if void_approximations: model.user_data['topologicpy_void_approximations'] = void_approximations
        if (model.user_data or {}).get('tpy_name'): model.display_name = model.user_data['tpy_name']
        model.tolerance = tolerance
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
        try:
            import honeybee_energy.lib.programtypes as prog_type_lib
        except ImportError as exc:
            raise ImportError('Honeybee requires Honeybee/Ladybug packages. Install it in this environment with: python -m pip install "topologicpy[honeybee]"') from exc
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
        try:
            import honeybee_energy.lib.programtypes as prog_type_lib
        except ImportError as exc:
            raise ImportError('Honeybee requires Honeybee/Ladybug packages. Install it in this environment with: python -m pip install "topologicpy[honeybee]"') from exc
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


"""Direct Honeybee gbXML export and validation of the delivered XML geometry.

The XML reader deliberately does not use OpenStudio, geometry repair, nearest
host assignment or face intersection. Invalid geometry remains visible to checks.
"""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

NS = {'g': 'http://www.gbxml.org/schema'}


class HoneybeeValidationError(ValueError):
    def __init__(self, label, errors):
        self.validation = dict(passed=False, validator='Honeybee',
                               error_count=len(errors), errors=errors)
        details = '; '.join(str(e.get('message', e)) for e in errors[:3])
        super().__init__(f'{label}: {len(errors)} Honeybee validation errors: {details}')


def validate_model(model, label, geometry_only=False):
    if geometry_only:
        # The XML adapter imports geometry and adjacency, not thermal resources.
        # Do not validate invented Honeybee default construction assignments.
        tol, angle = model.tolerance, model.angle_tolerance
        checks = [model.check_all_duplicate_identifiers(False, True),
            model.check_planar(tol, False, True), model.check_self_intersecting(tol, False, True),
            model.check_degenerate_rooms(tol, False, True), model.check_sub_faces_valid(tol, angle, False, True),
            model.check_sub_faces_overlapping(tol, False, True), model.check_upside_down_faces(angle, False, True),
            model.check_rooms_solid(tol, False, True), model.check_room_volume_collisions(tol, False, True),
            model.check_missing_adjacencies(False, True), model.check_matching_adjacent_areas(tol, False, True),
            model.check_all_air_boundaries_adjacent(False, True)]
        errors = [e for group in checks for e in group]
    else: errors = model.check_all(raise_exception=False, detailed=True)
    if errors:
        raise HoneybeeValidationError(label, errors)
    return dict(passed=True, validator='Honeybee', error_count=0, errors=[],
                spaces=len(model.rooms), surfaces=len(model.faces),
                openings=len(model.apertures)+len(model.doors),
                volume_m3=sum(r.volume for r in model.rooms))


def validate_hbjson(path, expected_rooms=None):
    from honeybee.model import Model
    model = Model.from_hbjson(str(path))
    if expected_rooms is not None and len(model.rooms) != expected_rooms:
        raise ValueError('HBJSON reopening changed the room count.')
    result = validate_model(model, 'HBJSON')
    result['reopened'] = True
    return model, result


def read_geometry(path, tolerance=1e-5):
    """Reconstruct every XML Surface/Opening in Honeybee without SDK translation."""
    from honeybee.model import Model
    from honeybee.room import Room
    from honeybee.face import Face
    from honeybee.aperture import Aperture
    from honeybee.door import Door
    from honeybee.facetype import Wall, Floor, RoofCeiling, AirBoundary
    from honeybee.boundarycondition import Outdoors, Ground
    from ladybug_geometry.geometry3d import Face3D, Point3D
    from lxml import etree
    parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False)
    root = etree.parse(str(path), parser).getroot()
    if root.tag != '{http://www.gbxml.org/schema}gbXML':
        raise ValueError('Not a gbXML document.')
    scale = {'Meters': 1., 'Feet': .3048, 'Millimeters': .001,
             'Centimeters': .01, 'Inches': .0254}.get(root.get('lengthUnit'))
    if scale is None: raise ValueError('Unsupported gbXML length unit.')
    identities = [e.get('id') for e in root.iter() if e.get('id')]
    if len(identities) != len(set(identities)):
        from collections import Counter
        duplicates = [k for k,v in Counter(identities).items() if v > 1]
        raise ValueError('gbXML contains duplicate IDs: '+', '.join(duplicates[:8]))
    spaces = root.findall('.//g:Building/g:Space', NS)
    rooms = {s.get('id'): [] for s in spaces}
    if not rooms: raise ValueError('gbXML contains no spaces.')

    def identity(value):
        return 'xml_'+hashlib.sha256(value.encode()).hexdigest()[:32]

    def geometry(element):
        loops = element.findall('g:PlanarGeometry/g:PolyLoop', NS)
        if not loops: raise ValueError(f"Missing PlanarGeometry: {element.get('id')}")
        rings = []
        for loop in loops:
            ring = []
            for point in loop.findall('g:CartesianPoint', NS):
                xyz = [float(c.text)*scale for c in point.findall('g:Coordinate', NS)]
                if len(xyz) != 3: raise ValueError('gbXML point must have three coordinates.')
                ring.append(Point3D(*xyz))
            rings.append(ring)
        return Face3D(rings[0], holes=rings[1:] or None)

    pairs = []; shades = []; surface_count = opening_count = 0
    for surface in root.findall('g:Campus/g:Surface', NS):
        sid = surface.get('id'); kind = surface.get('surfaceType', '')
        refs = [a.get('spaceIdRef') for a in surface.findall('g:AdjacentSpaceId', NS)]
        if len(refs) != len(set(refs)) or len(refs) > 2 or any(r not in rooms for r in refs):
            raise ValueError(f'Invalid space references on {sid}.')
        geo = geometry(surface)
        if not refs:
            if kind != 'Shade': raise ValueError(f'Unhosted gbXML surface: {sid}')
            from honeybee.shade import Shade
            shade=Shade(identity(sid),geo,is_detached=True);shade.display_name=sid
            shades.append(shade);continue
        surface_count += 1; copies = []
        openings = surface.findall('g:Opening', NS); opening_count += len(openings)
        for side, ref in enumerate(refs):
            face_type = Floor() if kind in ('InteriorFloor','SlabOnGrade','UndergroundSlab','RaisedFloor','ExposedFloor') else RoofCeiling() if kind in ('Roof','Ceiling','UndergroundCeiling') else AirBoundary() if kind == 'Air' else Wall()
            bc = Ground() if kind in ('SlabOnGrade','UndergroundSlab','UndergroundWall','UndergroundCeiling') else Outdoors()
            face = Face(identity(sid+'|'+ref), geo if side == 0 else geo.flip(), face_type, bc)
            face.display_name = sid; face.user_data = {'gbxml_surface_id':sid}
            for opening in openings:
                oid = opening.get('id'); opening_geo = geometry(opening)
                if side: opening_geo = opening_geo.flip()
                name = identity(oid+'|'+ref)
                kind_o = opening.get('openingType', '')
                child = Door(name, opening_geo, is_glass=kind_o == 'SlidingDoor') if 'Door' in kind_o else Aperture(name, opening_geo)
                child.display_name = oid; child.user_data = {'gbxml_opening_id':oid}
                (face.add_door if isinstance(child, Door) else face.add_aperture)(child)
            rooms[ref].append(face); copies.append(face)
        if len(copies) == 2: pairs.append(copies)
    hb_rooms = []
    for ref, faces in rooms.items():
        if not faces: raise ValueError(f'gbXML space has no surfaces: {ref}')
        # Preserve XML winding. Room(tolerance>0) rewrites face normals while
        # joining the shell; on skinny concave faces this can invert a valid
        # exported face and conceal the distinction between input and repair.
        # Honeybee's solid checks below still use the model's nonzero tolerance.
        room = Room(identity(ref), faces)
        for face in room.faces:
            if isinstance(face.type, (Floor, RoofCeiling)):
                face.type = RoofCeiling() if face.geometry.normal.z > 0 else Floor()
        room.display_name = ref; room.user_data = {'gbxml_space_id':ref}; hb_rooms.append(room)
    for left, right in pairs: left.set_adjacency(right, tolerance)
    model = Model('gbxml_reopened', hb_rooms, orphaned_shades=shades,
                  units='Meters', tolerance=tolerance, angle_tolerance=1.)
    return model, dict(xml_surfaces=surface_count, xml_openings=opening_count,
                       schema_version=root.get('version'))


def check_crossing_faces(model):
    """Detect opaque faces covering another face's window within the same room."""
    import numpy as np
    from shapely.geometry import Polygon
    from shapely.strtree import STRtree
    from shapely.geometry import box
    tol = model.tolerance
    for room in model.rooms:
        faces = list(room.faces)
        boxes = [box(min(p.x for p in f.geometry.boundary)-tol,
                     min(p.y for p in f.geometry.boundary)-tol,
                     max(p.x for p in f.geometry.boundary)+tol,
                     max(p.y for p in f.geometry.boundary)+tol) for f in faces]
        tree = STRtree(boxes)
        for i, face in enumerate(faces):
            if not face.has_sub_faces: continue
            g = face.geometry; origin = np.array(tuple(g.plane.o)); normal = np.array(tuple(g.normal))
            axes = np.array([tuple(g.plane.x), tuple(g.plane.y)])
            def project(points): return (np.array([tuple(p) for p in points])-origin)@axes.T
            apertures = [(a, Polygon(project(a.geometry.boundary))) for a in list(face.apertures)+list(face.doors)]
            for j in tree.query(boxes[i]):
                if i == j: continue
                other = faces[j]; pts = np.array([tuple(p) for p in other.geometry.boundary])
                if np.max(np.abs((pts-origin)@normal)) > tol: continue
                opaque = Polygon(project(other.geometry.boundary), [project(h) for h in other.geometry.holes or []])
                for sub in list(other.apertures)+list(other.doors): opaque = opaque.difference(Polygon(project(sub.geometry.boundary)))
                for opening, polygon in apertures:
                    if polygon.intersection(opaque).area > max(1e-7, polygon.area*1e-6):
                        raise ValueError(f'Opaque surface {other.display_name} crosses opening {opening.display_name} hosted by {face.display_name} in {room.display_name}.')


def validate_gbxml(path, expected=None):
    model, result = read_geometry(path, expected.tolerance if expected else 1e-5)
    result.update(validate_model(model, 'gbXML', geometry_only=True))
    result['scope'] = 'Serialized geometry, room closure, openings and adjacencies; thermal resources not imported'
    check_crossing_faces(model)
    result.update(reopened=True, reader='Direct XML to Honeybee; no OpenStudio', crossing_openings=0)
    if expected is not None:
        # ElementTree expands the default namespace on read. The identity
        # helper also operates on Honeybee's unqualified in-memory XML tree.
        identity_root = ET.parse(path).getroot()
        for element in identity_root.iter():
            if isinstance(element.tag, str): element.tag = element.tag.rsplit('}', 1)[-1]
        _validate_gbxml_source_identity(identity_root, expected)
        result['source_identity_preserved'] = True
        if len(model.rooms) != len(expected.rooms): raise ValueError('gbXML changed the room count.')
        from honeybee.typing import clean_xml_tag_string
        actual=[dict(id=s.display_name,vertices=[list(p) for p in s.geometry.boundary],holes=[[list(p) for p in h] for h in s.geometry.holes or []]) for s in model.orphaned_shades]
        source=[dict(id=clean_xml_tag_string(s.identifier),vertices=[list(p) for p in s.geometry.boundary],holes=[[list(p) for p in h] for h in s.geometry.holes or []]) for s in expected.orphaned_shades]
        result['exterior_shading']=_gbxml_check_shading(actual,source)
        from honeybee.typing import clean_xml_tag_string
        expected_rooms = {clean_xml_tag_string(r.identifier):r for r in expected.rooms}
        for room in model.rooms:
            source = expected_rooms.get(room.user_data['gbxml_space_id'])
            if source is None: raise ValueError('gbXML changed Honeybee room identities.')
        if len(model.apertures)+len(model.doors) != len(expected.apertures)+len(expected.doors):
            raise ValueError('gbXML changed the opening count.')
        import numpy as np
        from shapely.geometry import Polygon
        faces = {clean_xml_tag_string(f.identifier):f for f in expected.faces}
        openings = {clean_xml_tag_string(a.identifier):a for a in list(expected.apertures)+list(expected.doors)}
        def same_geometry(actual, source):
            ring = [tuple(p) for p in source.geometry.boundary]
            _, centre, axes = _gbxml_polygon(ring)
            def polygon(g):
                points = np.array([tuple(p) for p in g.boundary])
                if np.max(np.abs((points-centre)@axes[-1])) > expected.tolerance:
                    raise ValueError('gbXML moved a surface or opening out of its source plane.')
                return Polygon((points-centre)@axes[:2].T,
                    [(np.array([tuple(p) for p in h])-centre)@axes[:2].T for h in g.holes or []])
            left, right = polygon(actual.geometry), polygon(source.geometry)
            if left.symmetric_difference(right).area > max(1e-8, right.area*1e-6):
                raise ValueError(f'gbXML changed source geometry: {actual.display_name}')
        def geometry_volume(room):
            # Use the delivered face loops, including holes, for both sides of
            # the comparison. Honeybee's cached Polyface volume can differ when
            # edge merging changes tessellation of a finely subdivided room.
            total = 0.
            for face in room.faces:
                points = np.array([tuple(p) for p in face.geometry.boundary])
                _, centre, axes = _gbxml_polygon(points)
                polygon = Polygon((points-centre)@axes[:2].T,
                    [(np.array([tuple(p) for p in h])-centre)@axes[:2].T for h in face.geometry.holes or []])
                centroid = centre+np.array(polygon.centroid.coords[0])@axes[:2]
                normal = np.cross(axes[0],axes[1])
                if normal@np.array(tuple(face.geometry.normal)) < 0: normal = -normal
                total += polygon.area*(centroid@normal)/3
            return abs(total)
        for face in model.faces:
            source = faces.get(face.user_data['gbxml_surface_id'])
            if source is None: raise ValueError('gbXML lost a source surface identity.')
            same_geometry(face, source)
            if (source.type.name == 'AirBoundary') != (face.type.name == 'AirBoundary'):
                raise ValueError('gbXML changed an air-boundary surface classification.')
            if source.type.name == 'AirBoundary' and face.boundary_condition.name != 'Surface':
                raise ValueError('gbXML lost an air-boundary adjacency.')
            for opening in list(face.apertures)+list(face.doors):
                source = openings.get(opening.user_data['gbxml_opening_id'])
                if source is None: raise ValueError('gbXML lost a source opening identity.')
                same_geometry(opening, source)
        verified_volume = 0.
        for room in model.rooms:
            source = expected_rooms[room.user_data['gbxml_space_id']]
            before, after = geometry_volume(source), geometry_volume(room)
            if abs(before-after) > max(.005, before*1e-4):
                raise ValueError(f'gbXML changed source room volume: {room.display_name}')
            verified_volume += after
        result['face_loop_volume_m3'] = verified_volume
        result['source_honeybee_volume_m3'] = sum(r.volume for r in expected.rooms)
        result['source_geometry_preserved'] = True
    return result


def provenance_mapping(root, model):
    """Map emitted XML IDs to TPY references, including both room sides."""
    from honeybee.typing import clean_xml_tag_string as clean
    def index(objects):
        result = {}
        for obj in objects:
            key = clean(obj.identifier)
            if key in result: raise ValueError('Ambiguous cleaned XML identifier: '+key)
            result[key] = obj
        return result
    rooms = index(model.rooms)
    faces = index(model.faces)
    openings = index(list(model.apertures)+list(model.doors))
    shades = index(model.orphaned_shades)
    def entry(obj):
        parent = getattr(obj, 'parent', None)
        return {'honeybee_id':obj.identifier, 'parent_id':getattr(parent,'identifier',None),
                **{k:v for k,v in (obj.user_data or {}).items() if k in ('tpy_id','tpy_parent_id','tpy_ids','tpy_name','tpy_building_id','tpy_cellcomplex_id','topologicpy_opening_kind','topologicpy_void_treatment','bim2bem_opening_id')}}
    def sides(obj, lookup):
        result = [entry(obj)]
        bc = getattr(obj, 'boundary_condition', None)
        if getattr(bc, 'name', None) == 'Surface':
            refs = bc.boundary_condition_objects
            other = lookup.get(clean(refs[0]))
            if other is None: raise ValueError('Missing opposite-side provenance: '+obj.identifier)
            if other.identifier != obj.identifier: result.append(entry(other))
        return result
    result = {'schema_version':2,
              'buildings':{}, 'spaces':{}, 'surfaces':{}, 'openings':{},
              'void_approximations': (model.user_data or {}).get('topologicpy_void_approximations', [])}
    for b in root.findall('Campus/Building'):
        result['buildings'][b.get('id')] = {**{k:v for k,v in (model.user_data or {}).items() if k in ('tpy_id','tpy_name','tpy_ids')}}
        for sp in b.findall('Space'):
            sid=sp.get('id'); result['spaces'][sid]={**entry(rooms[sid]), 'cad_object_ids':[v.text for v in sp.findall('CADObjectId') if v.text]}
    for surface in root.findall('Campus/Surface'):
        sid=surface.get('id')
        obj=shades.get(sid) if surface.get('surfaceType')=='Shade' else faces.get(sid)
        if obj is None: raise ValueError('No source provenance object for XML surface '+str(sid))
        result['surfaces'][sid]={'cad_object_id':surface.findtext('CADObjectId'),
                                  'cad_object_ids':[v.text for v in surface.findall('CADObjectId') if v.text],
                                  'sources':sides(obj, faces)}
        for child in surface.findall('Opening'):
            oid=child.get('id'); result['openings'][oid]={'parent_surface_id':sid,
                'cad_object_id':child.findtext('CADObjectId'), 'cad_object_ids':[v.text for v in child.findall('CADObjectId') if v.text], 'sources':sides(openings[oid], openings)}
    return result


def _source_reference_ids(data):
    """Return only explicit analytical IDs; use parent only for generated geometry."""
    primary = data.get('tpy_id') or data.get('tpy_parent_id')
    refs = {str(primary)} if primary else set()
    extra = data.get('tpy_ids') or []
    if isinstance(extra, str):
        try: extra = json.loads(extra)
        except (ValueError, TypeError): extra = [extra]
    if not isinstance(extra, (list, tuple, set)): extra = [extra]
    refs.update(str(value) for value in extra if value is not None and str(value).strip())
    return refs


def _apply_gbxml_source_identity(root, model):
    """Preserve both sides of shared boundaries in the delivered XML itself."""
    mapping = provenance_mapping(root, model)
    def assign(el, sources):
        ids = set().union(*(_source_reference_ids(s) for s in sources))
        for child in list(el.findall('CADObjectId')): el.remove(child)
        primary = sources[0].get('tpy_id') or sources[0].get('tpy_parent_id')
        ordered = ([str(primary)] if primary else []) + sorted(ids - ({str(primary)} if primary else set()))
        for uid in ordered: ET.SubElement(el, 'CADObjectId').text = uid
        name = sources[0].get('tpy_name')
        if name:
            node = el.find('Name')
            if node is None: node = ET.SubElement(el, 'Name')
            node.text = str(name)
    for building in root.findall('Campus/Building'):
        data = mapping['buildings'][building.get('id')]
        ids = _source_reference_ids(data)
        if ids:
            desc = building.find('Description')
            if desc is None: desc = ET.SubElement(building, 'Description')
            text = 'TopologicPy source IDs: ' + json.dumps(sorted(ids), ensure_ascii=False)
            desc.text = '\n'.join(v for v in (desc.text, text) if v)
        if data.get('tpy_name'):
            node = building.find('Name')
            if node is None: node = ET.SubElement(building, 'Name')
            node.text = str(data['tpy_name'])
        for space in building.findall('Space'):
            assign(space, [mapping['spaces'][space.get('id')]])
    for surface in root.findall('Campus/Surface'):
        assign(surface, mapping['surfaces'][surface.get('id')]['sources'])
        for opening in surface.findall('Opening'):
            assign(opening, mapping['openings'][opening.get('id')]['sources'])
    return provenance_mapping(root, model)


def _validate_gbxml_source_identity(root, model):
    """Check the serialized CAD references against the original Honeybee data."""
    mapping = provenance_mapping(root, model)
    identified = lambda objects: {obj.identifier for obj in objects
                                 if _source_reference_ids(obj.user_data or {}) or (obj.user_data or {}).get('tpy_name')}
    expected = {
        'spaces': identified(model.rooms),
        'surfaces': identified(list(model.faces) + list(model.orphaned_shades)),
        'openings': identified(list(model.apertures) + list(model.doors))}
    found = {'spaces': {r['honeybee_id'] for r in mapping['spaces'].values()}}
    for section in ('surfaces', 'openings'):
        found[section] = {s['honeybee_id'] for entry in mapping[section].values() for s in entry['sources']}
    for section in expected:
        if expected[section] - found[section]:
            raise ValueError('gbXML omitted identified ' + section + ': ' + ', '.join(sorted(expected[section] - found[section])[:8]))
    for xml_id, entry in mapping['spaces'].items():
        if set(entry['cad_object_ids']) != _source_reference_ids(entry):
            raise ValueError('gbXML changed room source identity: ' + xml_id)
    for section in ('surfaces', 'openings'):
        for xml_id, entry in mapping[section].items():
            refs = set().union(*(_source_reference_ids(s) for s in entry['sources']))
            if set(entry['cad_object_ids']) != refs:
                raise ValueError('gbXML changed ' + section + ' source identity: ' + xml_id)
    def check_name(el, data):
        name = data.get('tpy_name')
        if name and el.findtext('Name') != str(name):
            raise ValueError('gbXML changed source name: ' + el.get('id', ''))
    for building in root.findall('Campus/Building'):
        check_name(building, mapping['buildings'][building.get('id')])
        for space in building.findall('Space'):
            check_name(space, mapping['spaces'][space.get('id')])
        refs = _source_reference_ids(mapping['buildings'][building.get('id')])
        if refs:
            description = building.findtext('Description') or ''
            marker = 'TopologicPy source IDs: ' + json.dumps(sorted(refs), ensure_ascii=False)
            if marker not in description:
                raise ValueError('gbXML changed building source identity: ' + building.get('id'))
    for surface in root.findall('Campus/Surface'):
        check_name(surface, mapping['surfaces'][surface.get('id')]['sources'][0])
        for opening in surface.findall('Opening'):
            check_name(opening, mapping['openings'][opening.get('id')]['sources'][0])
    return mapping


def _export_direct_gbxml(model, path):
    """Use Honeybee's native ElementTree writer, never its OpenStudio writer."""
    from honeybee_energy.writer import model_to_gbxml_element
    validate_model(model, 'gbXML source')
    check_crossing_faces(model)
    # The upstream writer uses the room identifier as the default Zone ID,
    # duplicating the Space ID in the global XML ID namespace. Set explicit,
    # deterministic zone identities on a copy before calling the writer.
    export_model = model.duplicate()
    # The writer derives decimal precision from this tolerance. Five decimals
    # would round away the audited 2-micrometre aperture clearance. Preserve
    # coordinates to nine decimals; validation still uses the source tolerance.
    export_model.tolerance = min(model.tolerance, 1e-12)
    for room in export_model.rooms:
        room.zone = 'zone_'+hashlib.sha256(str(room.zone or room.identifier).encode()).hexdigest()[:32]
    root = model_to_gbxml_element(export_model, triangulate_subfaces=False,
        triangulate_non_planar=False, explicit_holes=True, gbxml_schema_version='7.03',
        program_name='TopologicPy / Honeybee direct Python writer', program_version='0.1.0')
    # Honeybee uses empty synthetic Spaces to group standalone shades. They
    # are not thermal rooms; represent exterior shades directly at Campus.
    for building in root.findall('Campus/Building'):
        for space in list(building.findall('Space')):
            if space.get('id') in {'Detached_Shades','Attached_Shades'}:
                building.remove(space)
    for surface in root.findall('Campus/Surface'):
        if surface.get('surfaceType')=='Shade':
            for ref in list(surface.findall('AdjacentSpaceId')):surface.remove(ref)
    from honeybee.typing import clean_xml_tag_string
    shade_sources={clean_xml_tag_string(s.identifier):s for s in export_model.orphaned_shades}
    transparent=[sid for sid,s in shade_sources.items() if (s.user_data or {}).get('glazing_geometry')]
    opaque=root.find("Construction[@id='Shading_Surface_Without_Construction']")
    if opaque is not None:
        for wavelength in ('Solar','Visible'):
            ET.SubElement(opaque,'Transmittance',unit='Fraction',type=wavelength,surfaceType='Both').text='0'
    if transparent:
        construction=ET.SubElement(root,'Construction',id='TopologicPy_Transparent_Context')
        ET.SubElement(construction,'Name').text='Source glazing context: no additional opaque shading'
        for wavelength in ('Solar','Visible'):
            ET.SubElement(construction,'Transmittance',unit='Fraction',type=wavelength,surfaceType='Both').text='1'
        for surface in root.findall('Campus/Surface'):
            if surface.get('id') in transparent:surface.set('constructionIdRef','TopologicPy_Transparent_Context')
    # Glaze/Gap are inline definitions repeated in each window construction,
    # not shared references. Their XML IDs must also be globally unique.
    for window in root.findall('WindowType'):
        for index, layer in enumerate(window):
            if layer.tag in ('Glaze','Gap') and layer.get('id'):
                layer.set('id','layer_'+hashlib.sha256(
                    (window.get('id')+'|'+str(index)+'|'+layer.get('id')).encode()).hexdigest()[:32])
    provenance = _apply_gbxml_source_identity(root, model)
    _validate_gbxml_source_identity(root, model)
    ET.indent(root)
    path = Path(path)
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)
    mapping_path = path.with_suffix('.json')
    try:
        result = validate_gbxml(path, model)
        provenance['gbxml_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        mapping_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
        result['provenance_mapping'] = {k:len(provenance[k]) for k in ('buildings','spaces','surfaces','openings')}
    except Exception:
        path.unlink(missing_ok=True);mapping_path.unlink(missing_ok=True);raise
    result['writer'] = 'honeybee_energy.writer.model_to_gbxml_element'
    return result


def _gbxml_polygon(points,display_precision=False):
    import numpy as np
    from shapely.geometry import Polygon
    points = np.asarray(points, dtype=float)
    if len(points) < 3 or not np.isfinite(points).all():
        raise ValueError('Surface has invalid coordinates.')
    centre = points.mean(0)
    _, _, axes = np.linalg.svd(points-centre)
    if np.max(np.abs((points-centre)@axes[-1])) > .001:
        raise ValueError('Surface is non-planar by more than 1 mm.')
    polygon = Polygon((points-centre)@axes[:2].T)
    if not polygon.is_valid or polygon.area < (1e-8 if display_precision else 1e-12):
        raise ValueError('Surface loop is self-crossing or degenerate.')
    reduced = points.astype(np.float32).astype(float)
    reduced_polygon = Polygon((reduced-centre)@axes[:2].T)
    if display_precision and (not reduced_polygon.is_valid or reduced_polygon.area < 1e-8):
        raise ValueError('Surface collapses at display precision.')
    return polygon, centre, axes


def _gbxml_check_shading(actual,expected,tolerance=1e-7,allow_split=False):
    import numpy as np
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from topologicpy.IFC import _AnalyticalExportPlane as Plane
    """Compare delivered panel geometry with the shared source collection."""
    if allow_split:
        if len({r['id'] for r in actual})!=len(actual):raise ValueError('Export duplicated shading identities.')
        grouped={r['id']:[] for r in expected}
        for record in actual:
            identity=record['id'];parent=identity.rsplit('_panel',1)[0] if identity not in grouped else identity
            if parent not in grouped:raise ValueError('Export lost shading identities.')
            grouped[parent].append(record)
        for source in expected:
            parts=grouped[source['id']]
            if not parts:raise ValueError('Export lost shading surface '+source['id'])
            plane=Plane.polygon(source['vertices'],tolerance)
            shapes=[]
            for part in parts:
                if np.max(np.abs(np.asarray(part['vertices'])@plane.normal-plane.offset))>tolerance:raise ValueError('Export moved shading out of plane.')
                shapes.append(Polygon(plane.project(part['vertices']),[plane.project(h) for h in part.get('holes',[])]))
            original=Polygon(plane.project(source['vertices']),[plane.project(h) for h in source.get('holes',[])])
            combined=unary_union(shapes);limit=max(1e-8,original.area*1e-7)
            if combined.symmetric_difference(original).area>limit or sum(p.area for p in shapes)-combined.area>limit:raise ValueError('Export changed split shading geometry: '+source['id'])
        return dict(passed=True,surfaces=len(expected),delivered_panels=len(actual),source_geometry_preserved=True)
    if len(actual)!=len(expected):raise ValueError(f'Export changed exterior shading panel count: {len(actual)} delivered, {len(expected)} expected.')
    for a,e in zip(sorted(actual,key=lambda r:r['id']),sorted(expected,key=lambda r:r['id'])):
        if a['id']!=e['id']:raise ValueError('Export lost shading identities.')
        plane=Plane.polygon(e['vertices'],tolerance);p=np.asarray(a['vertices'])
        if np.max(np.abs(p@plane.normal-plane.offset))>tolerance:raise ValueError('Export moved shading out of plane.')
        left=Polygon(plane.project(a['vertices']),[plane.project(h) for h in a.get('holes',[])])
        right=Polygon(plane.project(e['vertices']),[plane.project(h) for h in e.get('holes',[])])
        difference=left.symmetric_difference(right).area
        if difference>max(1e-8,right.area*1e-7):raise ValueError(f"Export changed exterior shading geometry: {e['id']}; difference={difference}; expected area={right.area}; delivered area={left.area}.")
    return dict(passed=True,surfaces=len(expected),source_geometry_preserved=True)

