# Copyright (C) 2026
# Wassim Jabi <wassim.jabi@gmail.com>
#
# Pure-Python IFC topology importer for TopologicPy.
# This module intentionally avoids IfcOpenShell for geometry extraction.
# It parses IFC STEP text directly and converts supported IFC geometry
# constructs to TopologicPy topologies via Topology.ByGeometry.
#
# It is a benchmark / experimental importer, not a complete IFC geometric kernel.
# Supported constructs include common IFC2X3/IFC4 LOD100-LOD200 geometry:
# - IfcExtrudedAreaSolid from arbitrary closed/void profiles, rectangles, circles
# - IfcShellBasedSurfaceModel / IfcFaceBasedSurfaceModel
# - IfcFacetedBrep / IfcClosedShell / IfcOpenShell / IfcFace / IfcPolyLoop
# - IfcBoundingBox
# - IfcPolyline / IfcIndexedPolyCurve
# - IfcMappedItem / IfcRepresentationMap
# - Approximate IfcBooleanResult / IfcBooleanClippingResult by first operand
# - IfcLocalPlacement and Axis2Placement transforms

from __future__ import annotations

try:
    from topologicpy.ifc.exporter import IFCReferenceViewExporter, IFCExportConfig
except Exception:
    IFCReferenceViewExporter = None
    IFCExportConfig = None

import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

Ref = Tuple[str, int]
Call = Tuple[str, str, list]


@dataclass(slots=True)
class IFCFastEntity:
    id: int
    type: str
    args: list

    def is_a(self, name: Optional[str] = None):
        if name is None:
            return self.type
        return self.type.lower() == str(name).lower()

    def __repr__(self):
        return f"#{self.id}={self.type}(...)"


class IFCFastTopology:
    """
    Experimental pure-Python IFC importer for TopologicPy topologies.

    The goal is to benchmark a direct STEP-text parsing route against
    IFC.TopologiesByFile, which currently uses IfcOpenShell triangulation.

    This class is intentionally conservative: unsupported advanced geometry is
    skipped or approximated rather than evaluated through a full CSG/BREP kernel.
    """

    _REC_HEAD_RE = re.compile(rb"^\s*#\s*(\d+)\s*=\s*([A-Z][A-Z0-9_]*)\s*\(", re.I)

    _PRODUCT_TYPES = {
        "IFCACTUATOR", "IFCAIRTERMINAL", "IFCAIRTERMINALBOX", "IFCAIRTOAIRHEATRECOVERY",
        "IFCALARM", "IFCANNOTATION", "IFCBEAM", "IFCBEAMSTANDARDCASE", "IFCBOILER",
        "IFCBUILDING", "IFCBUILDINGELEMENTPART", "IFCBUILDINGELEMENTPROXY", "IFCBUILDINGSTOREY",
        "IFCBURNER", "IFCCHILLER", "IFCCHIMNEY", "IFCCIVILELEMENT", "IFCCOLUMN",
        "IFCCOLUMNSTANDARDCASE", "IFCCOMPRESSOR", "IFCCONDENSER", "IFCCONTROLLER",
        "IFCCOOLEDBEAM", "IFCCOOLINGTOWER", "IFCCOVERING", "IFCCURTAINWALL", "IFCDAMPER",
        "IFCDISCRETEACCESSORY", "IFCDISTRIBUTIONCHAMBERELEMENT", "IFCDISTRIBUTIONCONTROLELEMENT",
        "IFCDISTRIBUTIONELEMENT", "IFCDISTRIBUTIONFLOWELEMENT", "IFCDOOR", "IFCDOORSTANDARDCASE",
        "IFCELECTRICAPPLIANCE", "IFCELECTRICDISTRIBUTIONBOARD", "IFCELECTRICFLOWSTORAGEDEVICE",
        "IFCELECTRICGENERATOR", "IFCELECTRICMOTOR", "IFCELECTRICTIMECONTROL", "IFCELEMENTASSEMBLY",
        "IFCENERGYCONVERSIONDEVICE", "IFCEVAPORATIVECOOLER", "IFCEVAPORATOR", "IFCFAN",
        "IFCFASTENER", "IFCFILTER", "IFCFIRESUPPRESSIONTERMINAL", "IFCFLOWCONTROLLER",
        "IFCFLOWFITTING", "IFCFLOWINSTRUMENT", "IFCFLOWMOVINGDEVICE", "IFCFLOWSEGMENT",
        "IFCFLOWSTORAGEDEVICE", "IFCFLOWTERMINAL", "IFCFLOWTREATMENTDEVICE", "IFCFOOTING",
        "IFCFURNISHINGELEMENT", "IFCFURNITURE", "IFCGEOGRAPHICELEMENT", "IFCGRID", "IFCGRIDAXIS",
        "IFCHEATEXCHANGER", "IFCHUMIDIFIER", "IFCINTERCEPTOR", "IFCJUNCTIONBOX", "IFCLAMP",
        "IFCLIGHTFIXTURE", "IFCMECHANICALFASTENER", "IFCMEMBER", "IFCMEMBERSTANDARDCASE",
        "IFCMOTORCONNECTION", "IFCOPENINGELEMENT", "IFCPILE", "IFCPIPEFITTING", "IFCPIPESEGMENT",
        "IFCPLATE", "IFCPLATESTANDARDCASE", "IFCPROJECT", "IFCPROJECTIONELEMENT", "IFCPROTECTIVEDEVICE",
        "IFCPROTECTIVEDEVICETRIPPINGUNIT", "IFCPUMP", "IFCRAILING", "IFCRAMP", "IFCRAMPFLIGHT",
        "IFCREINFORCINGBAR", "IFCREINFORCINGMESH", "IFCROOF", "IFCSANITARYTERMINAL", "IFCSHADINGDEVICE",
        "IFCSITE", "IFCSLAB", "IFCSLABELEMENTEDCASE", "IFCSLABSTANDARDCASE", "IFCSPACE",
        "IFCSPATIALELEMENT", "IFCSPATIALSTRUCTUREELEMENT", "IFCSTACKTERMINAL", "IFCSTAIR",
        "IFCSTAIRFLIGHT", "IFCSWITCHINGDEVICE", "IFCSYSTEMFURNITUREELEMENT", "IFCTANK",
        "IFCTRANSPORTELEMENT", "IFCTUBEBUNDLE", "IFCUNITARYCONTROLELEMENT", "IFCUNITARYEQUIPMENT",
        "IFCVALVE", "IFCVIRTUALELEMENT", "IFCWALL", "IFCWALLELEMENTEDCASE", "IFCWALLSTANDARDCASE",
        "IFCWASTETERMINAL", "IFCWINDOW", "IFCWINDOWSTANDARDCASE",
    }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def TopologiesByPath(path: str,
                         includeTypes: list = [],
                         excludeTypes: list = [],
                         dictionaryMode: str = "basic",
                         clean: bool = False,
                         epsilon: float = 0.01,
                         angTolerance: float = 0.1,
                         tolerance: float = 0.0001,
                         circleSides: int = 24,
                         topologyType: str = None,
                         scale: float = 1.0,
                         silent: bool = False) -> list:
        """
        Imports IFC product geometry from a path into TopologicPy topologies.

        This method is intentionally API-compatible in spirit with
        IFC.TopologiesByPath / IFC.TopologiesByFile, but it does not use
        IfcOpenShell for geometry extraction.
        """
        if not path or not os.path.exists(path):
            if not silent:
                print("IFCFastTopology.TopologiesByPath - Error: The input path is invalid. Returning None.")
            return None
        entities = IFCFastTopology.Parse(path, silent=silent)
        return IFCFastTopology.TopologiesByEntities(
            entities,
            includeTypes=includeTypes,
            excludeTypes=excludeTypes,
            dictionaryMode=dictionaryMode,
            clean=clean,
            epsilon=epsilon,
            angTolerance=angTolerance,
            tolerance=tolerance,
            circleSides=circleSides,
            topologyType=topologyType,
            scale=scale,
            silent=silent,
        )

    @staticmethod
    def TopologiesByFile(file,
                         includeTypes: list = [],
                         excludeTypes: list = [],
                         dictionaryMode: str = "basic",
                         clean: bool = False,
                         epsilon: float = 0.01,
                         angTolerance: float = 0.1,
                         tolerance: float = 0.0001,
                         circleSides: int = 24,
                         topologyType: str = None,
                         scale: float = 1.0,
                         silent: bool = False) -> list:
        """
        Imports IFC product geometry into TopologicPy topologies.

        Parameters
        ----------
        file : str or ifcopenshell.file-like object
            Prefer passing a path string. If an ifcopenshell file-like object is
            supplied, this method attempts to serialise it to STEP text.
        """
        if isinstance(file, str):
            return IFCFastTopology.TopologiesByPath(
                file,
                includeTypes=includeTypes,
                excludeTypes=excludeTypes,
                dictionaryMode=dictionaryMode,
                clean=clean,
                epsilon=epsilon,
                angTolerance=angTolerance,
                tolerance=tolerance,
                circleSides=circleSides,
                topologyType=topologyType,
                scale=scale,
                silent=silent,
            )

        text = IFCFastTopology._file_to_step_text(file)
        if not text:
            if not silent:
                print("IFCFastTopology.TopologiesByFile - Error: Could not obtain STEP text from input file. Returning None.")
            return None
        entities = IFCFastTopology.ParseText(text, silent=silent)
        return IFCFastTopology.TopologiesByEntities(
            entities,
            includeTypes=includeTypes,
            excludeTypes=excludeTypes,
            dictionaryMode=dictionaryMode,
            clean=clean,
            epsilon=epsilon,
            angTolerance=angTolerance,
            tolerance=tolerance,
            circleSides=circleSides,
            topologyType=topologyType,
            scale=scale,
            silent=silent,
        )

    @staticmethod
    def MeshDataByPath(path: str,
                       includeTypes: list = [],
                       excludeTypes: list = [],
                       dictionaryMode: str = "basic",
                       circleSides: int = 24,
                       scale: float = 1.0,
                       silent: bool = False) -> dict:
        entities = IFCFastTopology.Parse(path, silent=silent)
        products = IFCFastTopology.ProductsByEntities(entities, includeTypes, excludeTypes)
        return IFCFastTopology.MeshDataByProducts(products, entities, dictionaryMode, circleSides, scale, silent=silent)

    @staticmethod
    def TopologiesByEntities(entities: Dict[int, IFCFastEntity],
                             includeTypes: list = [],
                             excludeTypes: list = [],
                             dictionaryMode: str = "basic",
                             clean: bool = False,
                             epsilon: float = 0.01,
                             angTolerance: float = 0.1,
                             tolerance: float = 0.0001,
                             circleSides: int = 24,
                             topologyType: str = None,
                             scale: float = 1.0,
                             silent: bool = False) -> list:
        from topologicpy.Topology import Topology

        start = time.time()
        products = IFCFastTopology.ProductsByEntities(entities, includeTypes, excludeTypes)
        topologies = []
        converted = 0
        skipped = 0

        for product in products:
            mesh = IFCFastTopology.MeshDataByProduct(product, entities, circleSides=circleSides, scale=scale)
            if not mesh or not mesh.get("vertices"):
                skipped += 1
                continue
            try:
                topology = Topology.ByGeometry(
                    vertices=mesh.get("vertices") or [],
                    edges=mesh.get("edges") or [],
                    faces=mesh.get("faces") or [],
                    topologyType=topologyType,
                    tolerance=tolerance,
                    silent=True,
                )
            except TypeError:
                try:
                    topology = Topology.ByGeometry(
                        vertices=mesh.get("vertices") or [],
                        edges=mesh.get("edges") or [],
                        faces=mesh.get("faces") or [],
                        tolerance=tolerance,
                    )
                except Exception as e:
                    if not silent:
                        print(f"IFCFastTopology.TopologiesByEntities - Warning: Could not create topology for #{product.id} {product.type}: {e}")
                    skipped += 1
                    continue
            except Exception as e:
                if not silent:
                    print(f"IFCFastTopology.TopologiesByEntities - Warning: Could not create topology for #{product.id} {product.type}: {e}")
                skipped += 1
                continue

            if topology is None or not Topology.IsInstance(topology, "Topology"):
                skipped += 1
                continue

            if clean:
                topology = IFCFastTopology._clean_topology(topology, epsilon, angTolerance, tolerance, silent)

            d = IFCFastTopology._entity_dictionary(product, dictionaryMode=dictionaryMode)
            if d is not None:
                try:
                    topology = Topology.SetDictionary(topology, d, silent=True)
                except TypeError:
                    try:
                        topology = Topology.SetDictionary(topology, d)
                    except Exception:
                        pass
                except Exception:
                    pass
            topologies.append(topology)
            converted += 1

        if not silent:
            print(
                f"IFCFastTopology.TopologiesByEntities - Created {converted} topologies; "
                f"skipped {skipped} products in {time.time() - start:.3f}s."
            )
        return topologies

    @staticmethod
    def SummaryByPath(path: str, silent: bool = False) -> dict:
        entities = IFCFastTopology.Parse(path, silent=silent)
        products = IFCFastTopology.ProductsByEntities(entities)
        type_counts = {}
        product_counts = {}
        rep_counts = {}
        item_counts = {}
        for e in entities.values():
            type_counts[e.type] = type_counts.get(e.type, 0) + 1
            if e.type == "IFCSHAPEREPRESENTATION":
                rep_id = IFCFastTopology._as_string(e.args[1] if len(e.args) > 1 else None)
                rep_type = IFCFastTopology._as_string(e.args[2] if len(e.args) > 2 else None)
                rep_counts[(rep_id, rep_type)] = rep_counts.get((rep_id, rep_type), 0) + 1
                for ref in IFCFastTopology._refs_in_value(e.args[3] if len(e.args) > 3 else None):
                    item = IFCFastTopology._entity_from_ref(ref, entities)
                    if item is not None:
                        item_counts[item.type] = item_counts.get(item.type, 0) + 1
        for p in products:
            product_counts[p.type] = product_counts.get(p.type, 0) + 1
        return {
            "entity_count": len(entities),
            "product_count": len(products),
            "entity_type_counts": dict(sorted(type_counts.items())),
            "product_type_counts": dict(sorted(product_counts.items())),
            "shape_representation_counts": {str(k): v for k, v in sorted(rep_counts.items(), key=lambda kv: str(kv[0]))},
            "representation_item_counts": dict(sorted(item_counts.items())),
        }

    # ------------------------------------------------------------------
    # STEP parsing
    # ------------------------------------------------------------------

    @staticmethod
    def Parse(path: str, silent: bool = False) -> Dict[int, IFCFastEntity]:
        start = time.time()
        with open(path, "rb") as f:
            data = f.read()
        entities = IFCFastTopology.ParseBytes(data, silent=True)
        if not silent:
            print(f"IFCFastTopology.Parse - Parsed {len(entities)} entities in {time.time() - start:.3f}s.")
        return entities

    @staticmethod
    def ParseText(text: str, silent: bool = False) -> Dict[int, IFCFastEntity]:
        data = text.encode("utf-8", errors="ignore")
        return IFCFastTopology.ParseBytes(data, silent=silent)

    @staticmethod
    def ParseBytes(data: bytes, silent: bool = False) -> Dict[int, IFCFastEntity]:
        entities: Dict[int, IFCFastEntity] = {}
        start = time.time()
        for stmt in IFCFastTopology._iter_step_statements(data):
            m = IFCFastTopology._REC_HEAD_RE.match(stmt)
            if not m:
                continue
            step_id = int(m.group(1))
            entity_type = m.group(2).decode("ascii", errors="ignore").upper()
            p0 = stmt.find(b"(", m.end() - 1)
            p1 = stmt.rfind(b")")
            if p0 < 0 or p1 <= p0:
                continue
            raw_args = stmt[p0 + 1:p1].decode("utf-8", errors="ignore")
            try:
                args = _STEPArgParser(raw_args).parse_list_content()
                entities[step_id] = IFCFastEntity(step_id, entity_type, args)
            except Exception as e:
                if not silent:
                    print(f"IFCFastTopology.ParseBytes - Warning: Could not parse #{step_id}={entity_type}: {e}")
        if not silent:
            print(f"IFCFastTopology.ParseBytes - Parsed {len(entities)} entities in {time.time() - start:.3f}s.")
        return entities

    @staticmethod
    def _iter_step_statements(data: bytes) -> Iterator[bytes]:
        """
        Fast IFC STEP statement splitter.

        IFC exchange files are statement-oriented and each entity assignment ends
        with a semicolon. This intentionally uses a direct byte split for speed.
        It is substantially faster and avoids pathological behaviour caused by
        malformed or non-standard apostrophe escaping in large STEP strings.
        Semicolons inside string literals are theoretically possible but rare in
        authoring-tool IFC exports; if encountered, only that individual record
        is likely to fail parsing and will be skipped.
        """
        for stmt in data.split(b";"):
            stmt = stmt.strip()
            if stmt.startswith(b"#") or stmt.startswith(b"\r#") or stmt.startswith(b"\n#"):
                yield stmt

    @staticmethod
    def _file_to_step_text(file: Any) -> Optional[str]:
        if file is None:
            return None
        # IfcOpenShell usually supports to_string() on the file object.
        for attr in ("to_string",):
            fn = getattr(file, attr, None)
            if callable(fn):
                try:
                    s = fn()
                    if isinstance(s, bytes):
                        return s.decode("utf-8", errors="ignore")
                    if isinstance(s, str) and "ISO-10303-21" in s:
                        return s
                except Exception:
                    pass
        try:
            s = str(file)
            if "ISO-10303-21" in s:
                return s
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Product selection / mesh extraction
    # ------------------------------------------------------------------

    @staticmethod
    def ProductsByEntities(entities: Dict[int, IFCFastEntity], includeTypes: list = [], excludeTypes: list = []) -> list:
        include = IFCFastTopology._normalise_type_set(includeTypes)
        exclude = IFCFastTopology._normalise_type_set(excludeTypes)
        products = []
        feature_subtraction_types = {"IFCOPENINGELEMENT", "IFCFEATUREELEMENTSUBTRACTION", "IFCVOIDINGFEATURE"}
        for e in entities.values():
            if include and e.type not in include:
                continue
            if exclude and e.type in exclude:
                continue
            # Match IFC.TopologiesByFile/IfcOpenShell behaviour more closely: do not
            # import subtraction features such as IfcOpeningElement as standalone
            # visible products unless the caller explicitly requests them.
            if not include and e.type in feature_subtraction_types:
                continue
            if IFCFastTopology._is_product_like(e):
                products.append(e)
        products.sort(key=lambda ent: ent.id)
        return products

    @staticmethod
    def MeshDataByProducts(products: list,
                           entities: Dict[int, IFCFastEntity],
                           dictionaryMode: str = "basic",
                           circleSides: int = 24,
                           scale: float = 1.0,
                           silent: bool = False) -> dict:
        vertices, edges, faces, vertex_dictionaries, product_ranges = [], [], [], [], []
        for product in products:
            mesh = IFCFastTopology.MeshDataByProduct(product, entities, circleSides=circleSides, scale=scale)
            if not mesh or not mesh.get("vertices"):
                continue
            d = IFCFastTopology._entity_dictionary(product, dictionaryMode)
            v0, e0, f0 = len(vertices), len(edges), len(faces)
            vertices.extend(mesh.get("vertices") or [])
            vertex_dictionaries.extend([d] * len(mesh.get("vertices") or []))
            edges.extend([[a + v0, b + v0] for a, b in (mesh.get("edges") or [])])
            faces.extend([[i + v0 for i in face] for face in (mesh.get("faces") or [])])
            product_ranges.append({
                "id": product.id,
                "type": product.type,
                "global_id": IFCFastTopology._root_attr(product, 0),
                "name": IFCFastTopology._root_attr(product, 2),
                "start": v0,
                "end": len(vertices),
                "edge_start": e0,
                "edge_end": len(edges),
                "face_start": f0,
                "face_end": len(faces),
            })
        if not silent:
            print(
                f"IFCFastTopology.MeshDataByProducts - Extracted {len(product_ranges)} products, "
                f"{len(vertices)} vertices, {len(edges)} edges, {len(faces)} faces."
            )
        return {"vertices": vertices, "edges": edges, "faces": faces, "vertex_dictionaries": vertex_dictionaries, "product_ranges": product_ranges}

    @staticmethod
    def MeshDataByProduct(product: IFCFastEntity,
                          entities: Dict[int, IFCFastEntity],
                          circleSides: int = 24,
                          scale: float = 1.0) -> Optional[dict]:
        placement = IFCFastTopology._product_placement_matrix(product, entities)
        rep_ref = product.args[6] if len(product.args) > 6 else None
        rep = IFCFastTopology._entity_from_ref(rep_ref, entities)
        if rep is None or rep.type != "IFCPRODUCTDEFINITIONSHAPE":
            return None

        # IFC products often carry several shape representations simultaneously,
        # for example Body + Axis + BoundingBox. IFC.TopologiesByFile/IfcOpenShell
        # effectively imports the visible/body geometry. Merging every
        # representation into one TopologicPy object creates apparent placement
        # errors: walls get their axis curves, windows get their bounding boxes,
        # and panes may appear duplicated or offset.
        shape_reps = []
        for sr_ref in IFCFastTopology._refs_in_value(rep.args[2] if len(rep.args) > 2 else None):
            sr = IFCFastTopology._entity_from_ref(sr_ref, entities)
            if sr is None or sr.type != "IFCSHAPEREPRESENTATION":
                continue
            identifier = str(sr.args[1]).upper() if len(sr.args) > 1 and sr.args[1] not in (None, "*") else ""
            rep_type = str(sr.args[2]).upper() if len(sr.args) > 2 and sr.args[2] not in (None, "*") else ""
            shape_reps.append((identifier, rep_type, sr))

        if not shape_reps:
            return None

        body_reps = [sr for ident, rtype, sr in shape_reps if ident == "BODY"]
        if body_reps:
            selected_reps = body_reps
        else:
            non_bbox = [(ident, rtype, sr) for ident, rtype, sr in shape_reps if rtype != "BOUNDINGBOX" and ident != "BOX"]
            # Prefer footprint/curve geometry for virtual elements and other
            # products that genuinely have no Body representation. Avoid Axis
            # unless it is the only available representation.
            non_axis = [sr for ident, rtype, sr in non_bbox if ident != "AXIS"]
            selected_reps = non_axis if non_axis else [sr for ident, rtype, sr in non_bbox]
            if not selected_reps:
                # Last resort: explicit IFC bounding box. This is useful for very
                # coarse models, but it must not be merged with body geometry.
                selected_reps = [sr for ident, rtype, sr in shape_reps if rtype == "BOUNDINGBOX" or ident == "BOX"]

        meshes = []
        for sr in selected_reps:
            for item_ref in IFCFastTopology._refs_in_value(sr.args[3] if len(sr.args) > 3 else None):
                item = IFCFastTopology._entity_from_ref(item_ref, entities)
                if item is None:
                    continue
                m = IFCFastTopology._mesh_from_item(item, entities, placement, circleSides=circleSides, scale=scale)
                if m is not None and m.get("vertices"):
                    meshes.append(m)
        if not meshes:
            return None
        return IFCFastTopology._merge_meshes(meshes, dedupe=True)

    @staticmethod
    def _mesh_from_item(item: IFCFastEntity,
                        entities: Dict[int, IFCFastEntity],
                        matrix: list,
                        circleSides: int = 24,
                        scale: float = 1.0) -> Optional[dict]:
        t = item.type
        if t == "IFCEXTRUDEDAREASOLID":
            return IFCFastTopology._extruded_area_solid_mesh(item, entities, matrix, circleSides, scale)
        if t == "IFCPOLYLINE":
            pts = IFCFastTopology._polyline_points(item, entities)
            if not pts:
                return None
            pts = [IFCFastTopology._transform_point(p, matrix, scale) for p in pts]
            return {"vertices": pts, "edges": [[i, i + 1] for i in range(len(pts) - 1)], "faces": []}
        if t == "IFCINDEXEDPOLYCURVE":
            pts, eds = IFCFastTopology._indexed_polycurve_points_edges(item, entities)
            if not pts:
                return None
            pts = [IFCFastTopology._transform_point(p, matrix, scale) for p in pts]
            return {"vertices": pts, "edges": eds, "faces": []}
        if t == "IFCBOUNDINGBOX":
            m = IFCFastTopology._bounding_box_mesh(item, entities)
            if m is None:
                return None
            m["vertices"] = [IFCFastTopology._transform_point(p, matrix, scale) for p in m["vertices"]]
            return m
        if t in ("IFCSHELLBASEDSURFACEMODEL", "IFCFACEBASEDSURFACEMODEL"):
            return IFCFastTopology._surface_model_mesh(item, entities, matrix, scale)
        if t in ("IFCFACETEDBREP", "IFCADVANCEDBREP"):
            shell = IFCFastTopology._entity_from_ref(item.args[0] if item.args else None, entities)
            return IFCFastTopology._shell_mesh(shell, entities, matrix, scale) if shell is not None else None
        if t in ("IFCCLOSEDSHELL", "IFCOPENSHELL", "IFCCONNECTEDFACESET"):
            return IFCFastTopology._shell_mesh(item, entities, matrix, scale)
        if t == "IFCMAPPEDITEM":
            return IFCFastTopology._mapped_item_mesh(item, entities, matrix, circleSides, scale)
        if t in ("IFCBOOLEANRESULT", "IFCBOOLEANCLIPPINGRESULT"):
            # Approximation: keep first operand. This is fast but does not apply openings/cuts.
            first = IFCFastTopology._entity_from_ref(item.args[1] if len(item.args) > 1 else None, entities)
            if first is not None:
                return IFCFastTopology._mesh_from_item(first, entities, matrix, circleSides, scale)
        return None

    @staticmethod
    def _surface_model_mesh(item: IFCFastEntity, entities: Dict[int, IFCFastEntity], matrix: list, scale: float) -> Optional[dict]:
        meshes = []
        for ref in IFCFastTopology._refs_in_value(item.args[0] if item.args else None):
            shell = IFCFastTopology._entity_from_ref(ref, entities)
            if shell is not None:
                if shell.type in ("IFCCLOSEDSHELL", "IFCOPENSHELL", "IFCCONNECTEDFACESET"):
                    m = IFCFastTopology._shell_mesh(shell, entities, matrix, scale)
                    if m is not None:
                        meshes.append(m)
                elif shell.type == "IFCFACE":
                    m = IFCFastTopology._face_mesh(shell, entities, matrix, scale)
                    if m is not None:
                        meshes.append(m)
        return IFCFastTopology._merge_meshes(meshes, dedupe=True) if meshes else None

    @staticmethod
    def _shell_mesh(shell: IFCFastEntity, entities: Dict[int, IFCFastEntity], matrix: list, scale: float) -> Optional[dict]:
        if shell is None or not shell.args:
            return None
        vertices: list = []
        faces: list = []
        key_to_index: dict = {}
        face_refs = IFCFastTopology._refs_in_value(shell.args[0])
        for face_ref in face_refs:
            face = IFCFastTopology._entity_from_ref(face_ref, entities)
            if face is None or face.type != "IFCFACE":
                continue
            loops = IFCFastTopology._face_loops(face, entities)
            if not loops:
                continue
            # Topology.ByGeometry only accepts a face as one index loop here.
            # Use the outer loop. Inner bounds/openings are ignored in this fast path.
            loop = loops[0]
            if len(loop) < 3:
                continue
            idxs = []
            for p in loop:
                wp = IFCFastTopology._transform_point(p, matrix, scale)
                key = IFCFastTopology._point_key(wp)
                idx = key_to_index.get(key)
                if idx is None:
                    idx = len(vertices)
                    key_to_index[key] = idx
                    vertices.append(wp)
                idxs.append(idx)
            if len(idxs) > 1 and idxs[0] == idxs[-1]:
                idxs = idxs[:-1]
            if len(set(idxs)) >= 3:
                faces.append(idxs)
        if not vertices:
            return None
        return {"vertices": vertices, "edges": [], "faces": faces}

    @staticmethod
    def _face_mesh(face: IFCFastEntity, entities: Dict[int, IFCFastEntity], matrix: list, scale: float) -> Optional[dict]:
        loops = IFCFastTopology._face_loops(face, entities)
        if not loops:
            return None
        pts = [IFCFastTopology._transform_point(p, matrix, scale) for p in loops[0]]
        if len(pts) > 1 and IFCFastTopology._points_close(pts[0], pts[-1]):
            pts = pts[:-1]
        return {"vertices": pts, "edges": [], "faces": [list(range(len(pts)))] if len(pts) >= 3 else []}

    @staticmethod
    def _face_loops(face: IFCFastEntity, entities: Dict[int, IFCFastEntity]) -> list:
        loops = []
        for bound_ref in IFCFastTopology._refs_in_value(face.args[0] if face.args else None):
            bound = IFCFastTopology._entity_from_ref(bound_ref, entities)
            if bound is None or bound.type not in ("IFCFACEOUTERBOUND", "IFCFACEBOUND"):
                continue
            loop_ent = IFCFastTopology._entity_from_ref(bound.args[0] if bound.args else None, entities)
            if loop_ent is None:
                continue
            pts = []
            if loop_ent.type == "IFCPOLYLOOP":
                for p_ref in IFCFastTopology._refs_in_value(loop_ent.args[0] if loop_ent.args else None):
                    p = IFCFastTopology._cartesian_point(p_ref, entities)
                    if p is not None:
                        pts.append(p)
            elif loop_ent.type == "IFCEDGELOOP":
                # Minimal edge-loop support. Curved edges are approximated by endpoints.
                for edge_ref in IFCFastTopology._refs_in_value(loop_ent.args[0] if loop_ent.args else None):
                    edge = IFCFastTopology._entity_from_ref(edge_ref, entities)
                    pts.extend(IFCFastTopology._edge_points(edge, entities))
            if len(pts) >= 3:
                loops.append(pts)
        # Ensure outer loop first if present.
        return loops

    @staticmethod
    def _edge_points(edge: Optional[IFCFastEntity], entities: Dict[int, IFCFastEntity]) -> list:
        if edge is None:
            return []
        if edge.type == "IFCORIENTEDEDGE":
            edge_element = IFCFastTopology._entity_from_ref(edge.args[3] if len(edge.args) > 3 else None, entities)
            pts = IFCFastTopology._edge_points(edge_element, entities)
            orientation = edge.args[4] if len(edge.args) > 4 else True
            return pts if orientation is not False else list(reversed(pts))
        if edge.type in ("IFCEDGE", "IFCEDGECURVE"):
            a = IFCFastTopology._vertex_point(edge.args[0] if len(edge.args) > 0 else None, entities)
            b = IFCFastTopology._vertex_point(edge.args[1] if len(edge.args) > 1 else None, entities)
            return [p for p in [a, b] if p is not None]
        return []

    @staticmethod
    def _vertex_point(v: Any, entities: Dict[int, IFCFastEntity]) -> Optional[list]:
        vertex = IFCFastTopology._entity_from_ref(v, entities)
        if vertex is None:
            return None
        if vertex.type == "IFCVERTEXPOINT":
            return IFCFastTopology._cartesian_point(vertex.args[0] if vertex.args else None, entities)
        return None

    @staticmethod
    def _mapped_item_mesh(item: IFCFastEntity, entities: Dict[int, IFCFastEntity], matrix: list, circleSides: int, scale: float) -> Optional[dict]:
        """
        Returns mesh data for an IfcMappedItem.

        The correct transform chain is:

            product/local placement * mapping target * inverse(mapping source origin)

        The previous version used:

            product/local placement * mapping target * mapping source origin

        That is wrong when the representation map has a non-identity
        MappingOrigin. It typically shows up first in repeated components such
        as window panes, mullions, curtain-wall parts, and partition modules.
        """
        mapping_source = IFCFastTopology._entity_from_ref(item.args[0] if len(item.args) > 0 else None, entities)
        operator = IFCFastTopology._entity_from_ref(item.args[1] if len(item.args) > 1 else None, entities)
        if mapping_source is None or mapping_source.type != "IFCREPRESENTATIONMAP":
            return None

        origin = IFCFastTopology._entity_from_ref(mapping_source.args[0] if len(mapping_source.args) > 0 else None, entities)
        origin_matrix = IFCFastTopology._axis2placement_matrix(origin, entities)
        origin_inverse = IFCFastTopology._inverse_rigid_matrix(origin_matrix)

        source_rep = IFCFastTopology._entity_from_ref(mapping_source.args[1] if len(mapping_source.args) > 1 else None, entities)
        op_matrix = IFCFastTopology._mapped_item_operator_matrix(operator, entities)

        new_matrix = IFCFastTopology._matmul(matrix, IFCFastTopology._matmul(op_matrix, origin_inverse))

        meshes = []
        if source_rep is not None and source_rep.type == "IFCSHAPEREPRESENTATION":
            for ref in IFCFastTopology._refs_in_value(source_rep.args[3] if len(source_rep.args) > 3 else None):
                child = IFCFastTopology._entity_from_ref(ref, entities)
                if child is not None:
                    m = IFCFastTopology._mesh_from_item(child, entities, new_matrix, circleSides, scale)
                    if m is not None:
                        meshes.append(m)
        return IFCFastTopology._merge_meshes(meshes, dedupe=True) if meshes else None

    # ------------------------------------------------------------------
    # Swept solids / profiles
    # ------------------------------------------------------------------

    @staticmethod
    def _extruded_area_solid_mesh(solid: IFCFastEntity, entities: Dict[int, IFCFastEntity], product_matrix: list, circleSides: int, scale: float) -> Optional[dict]:
        if len(solid.args) < 4:
            return None
        profile = IFCFastTopology._entity_from_ref(solid.args[0], entities)
        position = IFCFastTopology._axis2placement_matrix(IFCFastTopology._entity_from_ref(solid.args[1], entities), entities)
        direction = IFCFastTopology._direction(solid.args[2], entities, default=[0.0, 0.0, 1.0])
        depth = IFCFastTopology._as_float(solid.args[3], 0.0)
        if profile is None or abs(depth) <= 1e-12:
            return None
        loops = IFCFastTopology._profile_loops(profile, entities, circleSides)
        if not loops:
            return None
        outer = loops[0]
        if len(outer) > 1 and IFCFastTopology._points_close(outer[0], outer[-1]):
            outer = outer[:-1]
        if len(outer) < 3:
            return None
        direction = IFCFastTopology._normalize(IFCFastTopology._normalise_point(direction))
        if IFCFastTopology._norm(direction) <= 1e-12:
            direction = [0.0, 0.0, 1.0]
        top_offset = [direction[0] * depth, direction[1] * depth, direction[2] * depth]
        n = len(outer)
        local_vertices = [IFCFastTopology._normalise_point(p) for p in outer]
        local_vertices += [[p[0] + top_offset[0], p[1] + top_offset[1], p[2] + top_offset[2]] for p in local_vertices[:n]]
        matrix = IFCFastTopology._matmul(product_matrix, position)
        vertices = [IFCFastTopology._transform_point(p, matrix, scale) for p in local_vertices]
        faces = [list(range(n)), list(range(2 * n - 1, n - 1, -1))]
        for i in range(n):
            j = (i + 1) % n
            faces.append([i, j, j + n, i + n])
        return {"vertices": vertices, "edges": [], "faces": faces}

    @staticmethod
    def _profile_loops(profile: IFCFastEntity, entities: Dict[int, IFCFastEntity], circleSides: int) -> list:
        t = profile.type
        if t == "IFCARBITRARYCLOSEDPROFILEDEF":
            curve = IFCFastTopology._entity_from_ref(profile.args[2] if len(profile.args) > 2 else None, entities)
            pts = IFCFastTopology._curve_points_local(curve, entities)
            return [pts] if pts else []
        if t == "IFCARBITRARYPROFILEDEFWITHVOIDS":
            outer_curve = IFCFastTopology._entity_from_ref(profile.args[2] if len(profile.args) > 2 else None, entities)
            pts = IFCFastTopology._curve_points_local(outer_curve, entities)
            # Inner voids are ignored in this fast topology path.
            return [pts] if pts else []
        if t == "IFCRECTANGLEPROFILEDEF":
            xdim = IFCFastTopology._as_float(profile.args[3] if len(profile.args) > 3 else 0.0, 0.0)
            ydim = IFCFastTopology._as_float(profile.args[4] if len(profile.args) > 4 else 0.0, 0.0)
            hx, hy = xdim * 0.5, ydim * 0.5
            pts = [[-hx, -hy, 0.0], [hx, -hy, 0.0], [hx, hy, 0.0], [-hx, hy, 0.0]]
            pos = IFCFastTopology._axis2placement_matrix(IFCFastTopology._entity_from_ref(profile.args[2] if len(profile.args) > 2 else None, entities), entities)
            return [[IFCFastTopology._transform_point(p, pos) for p in pts]]
        if t == "IFCCIRCLEPROFILEDEF":
            radius = IFCFastTopology._as_float(profile.args[3] if len(profile.args) > 3 else 0.0, 0.0)
            if radius <= 0:
                return []
            sides = max(8, int(circleSides or 24))
            pts = [[radius * math.cos(2 * math.pi * i / sides), radius * math.sin(2 * math.pi * i / sides), 0.0] for i in range(sides)]
            pos = IFCFastTopology._axis2placement_matrix(IFCFastTopology._entity_from_ref(profile.args[2] if len(profile.args) > 2 else None, entities), entities)
            return [[IFCFastTopology._transform_point(p, pos) for p in pts]]
        return []

    @staticmethod
    def _curve_points_local(curve: Optional[IFCFastEntity], entities: Dict[int, IFCFastEntity]) -> list:
        if curve is None:
            return []
        if curve.type == "IFCPOLYLINE":
            return IFCFastTopology._polyline_points(curve, entities)
        if curve.type == "IFCINDEXEDPOLYCURVE":
            pts, _ = IFCFastTopology._indexed_polycurve_points_edges(curve, entities)
            return pts
        return []

    @staticmethod
    def _polyline_points(polyline: IFCFastEntity, entities: Dict[int, IFCFastEntity]) -> list:
        pts = []
        for r in IFCFastTopology._refs_in_value(polyline.args[0] if polyline.args else None):
            p = IFCFastTopology._cartesian_point(r, entities)
            if p is not None:
                pts.append(p)
        return pts

    @staticmethod
    def _indexed_polycurve_points_edges(curve: IFCFastEntity, entities: Dict[int, IFCFastEntity]) -> tuple[list, list]:
        point_list = IFCFastTopology._entity_from_ref(curve.args[0] if curve.args else None, entities)
        if point_list is None:
            return [], []
        pts = []
        coords = point_list.args[0] if point_list.args else []
        if isinstance(coords, list):
            for c in coords:
                if isinstance(c, list):
                    pts.append(IFCFastTopology._normalise_point(c))
        edges = [[i, i + 1] for i in range(len(pts) - 1)]
        segments = curve.args[1] if len(curve.args) > 1 else None
        if isinstance(segments, list) and segments:
            edges = []
            for seg in segments:
                if isinstance(seg, tuple) and len(seg) == 3 and seg[0] == "CALL" and seg[1] == "IFCLINEINDEX":
                    idxs = [int(x) - 1 for x in seg[2][0] if isinstance(x, (int, float))]
                    for a, b in zip(idxs[:-1], idxs[1:]):
                        if 0 <= a < len(pts) and 0 <= b < len(pts):
                            edges.append([a, b])
        return pts, edges

    @staticmethod
    def _bounding_box_mesh(box: IFCFastEntity, entities: Dict[int, IFCFastEntity]) -> Optional[dict]:
        corner = IFCFastTopology._cartesian_point(box.args[0] if len(box.args) > 0 else None, entities)
        if corner is None or len(box.args) < 4:
            return None
        xdim = IFCFastTopology._as_float(box.args[1], 0.0)
        ydim = IFCFastTopology._as_float(box.args[2], 0.0)
        zdim = IFCFastTopology._as_float(box.args[3], 0.0)
        x, y, z = IFCFastTopology._normalise_point(corner)
        verts = [
            [x, y, z], [x + xdim, y, z], [x + xdim, y + ydim, z], [x, y + ydim, z],
            [x, y, z + zdim], [x + xdim, y, z + zdim], [x + xdim, y + ydim, z + zdim], [x, y + ydim, z + zdim],
        ]
        faces = [[0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]]
        return {"vertices": verts, "edges": [], "faces": faces}

    # ------------------------------------------------------------------
    # Placement / transforms
    # ------------------------------------------------------------------

    @staticmethod
    def _product_placement_matrix(product: IFCFastEntity, entities: Dict[int, IFCFastEntity]) -> list:
        placement = IFCFastTopology._entity_from_ref(product.args[5] if len(product.args) > 5 else None, entities)
        return IFCFastTopology._local_placement_matrix(placement, entities)

    @staticmethod
    def _local_placement_matrix(placement: Optional[IFCFastEntity], entities: Dict[int, IFCFastEntity]) -> list:
        if placement is None or placement.type != "IFCLOCALPLACEMENT":
            return IFCFastTopology._identity()
        parent = IFCFastTopology._entity_from_ref(placement.args[0] if len(placement.args) > 0 else None, entities)
        rel = IFCFastTopology._entity_from_ref(placement.args[1] if len(placement.args) > 1 else None, entities)
        parent_matrix = IFCFastTopology._local_placement_matrix(parent, entities) if parent is not None else IFCFastTopology._identity()
        rel_matrix = IFCFastTopology._axis2placement_matrix(rel, entities)
        return IFCFastTopology._matmul(parent_matrix, rel_matrix)

    @staticmethod
    def _axis2placement_matrix(axis: Optional[IFCFastEntity], entities: Dict[int, IFCFastEntity]) -> list:
        if axis is None:
            return IFCFastTopology._identity()
        if axis.type == "IFCAXIS2PLACEMENT3D":
            loc = IFCFastTopology._cartesian_point(axis.args[0] if len(axis.args) > 0 else None, entities) or [0.0, 0.0, 0.0]
            z = IFCFastTopology._direction(axis.args[1] if len(axis.args) > 1 else None, entities, default=[0.0, 0.0, 1.0])
            x = IFCFastTopology._direction(axis.args[2] if len(axis.args) > 2 else None, entities, default=[1.0, 0.0, 0.0])
            z = IFCFastTopology._normalize(IFCFastTopology._normalise_point(z))
            x = IFCFastTopology._normalize(IFCFastTopology._normalise_point(x))
            y = IFCFastTopology._normalize(IFCFastTopology._cross(z, x))
            x = IFCFastTopology._normalize(IFCFastTopology._cross(y, z))
            loc = IFCFastTopology._normalise_point(loc)
            return [[x[0], y[0], z[0], loc[0]], [x[1], y[1], z[1], loc[1]], [x[2], y[2], z[2], loc[2]], [0.0, 0.0, 0.0, 1.0]]
        if axis.type == "IFCAXIS2PLACEMENT2D":
            loc = IFCFastTopology._cartesian_point(axis.args[0] if len(axis.args) > 0 else None, entities) or [0.0, 0.0, 0.0]
            x = IFCFastTopology._direction(axis.args[1] if len(axis.args) > 1 else None, entities, default=[1.0, 0.0, 0.0])
            x = IFCFastTopology._normalize(IFCFastTopology._normalise_point(x))
            y = [-x[1], x[0], 0.0]
            loc = IFCFastTopology._normalise_point(loc)
            return [[x[0], y[0], 0.0, loc[0]], [x[1], y[1], 0.0, loc[1]], [0.0, 0.0, 1.0, loc[2]], [0.0, 0.0, 0.0, 1.0]]
        return IFCFastTopology._identity()

    @staticmethod
    def _mapped_item_operator_matrix(operator: Optional[IFCFastEntity], entities: Dict[int, IFCFastEntity]) -> list:
        if operator is None:
            return IFCFastTopology._identity()

        if operator.type in ("IFCCARTESIANTRANSFORMATIONOPERATOR3D", "IFCCARTESIANTRANSFORMATIONOPERATOR3DNONUNIFORM"):
            raw_axis1 = IFCFastTopology._direction(operator.args[0] if len(operator.args) > 0 else None, entities, default=[1.0, 0.0, 0.0])
            raw_axis2 = IFCFastTopology._direction(operator.args[1] if len(operator.args) > 1 else None, entities, default=None)
            origin = IFCFastTopology._cartesian_point(operator.args[2] if len(operator.args) > 2 else None, entities) or [0.0, 0.0, 0.0]
            s1 = IFCFastTopology._as_float(operator.args[3] if len(operator.args) > 3 else 1.0, 1.0)
            raw_axis3 = IFCFastTopology._direction(operator.args[4] if len(operator.args) > 4 else None, entities, default=None)
            s2 = IFCFastTopology._as_float(operator.args[5] if len(operator.args) > 5 else s1, s1)
            s3 = IFCFastTopology._as_float(operator.args[6] if len(operator.args) > 6 else s1, s1)

            x = IFCFastTopology._normalize(IFCFastTopology._normalise_point(raw_axis1 or [1.0, 0.0, 0.0]))
            if raw_axis2 is not None:
                y = IFCFastTopology._normalize(IFCFastTopology._normalise_point(raw_axis2))
            elif raw_axis3 is not None:
                z_tmp = IFCFastTopology._normalize(IFCFastTopology._normalise_point(raw_axis3))
                y = IFCFastTopology._normalize(IFCFastTopology._cross(z_tmp, x))
            else:
                y = [0.0, 1.0, 0.0]

            if raw_axis3 is not None:
                z = IFCFastTopology._normalize(IFCFastTopology._normalise_point(raw_axis3))
            else:
                z = IFCFastTopology._normalize(IFCFastTopology._cross(x, y))

            y = IFCFastTopology._normalize(IFCFastTopology._cross(z, x))
            x = IFCFastTopology._normalize(IFCFastTopology._cross(y, z))

            o = IFCFastTopology._normalise_point(origin)
            return [
                [x[0] * s1, y[0] * s2, z[0] * s3, o[0]],
                [x[1] * s1, y[1] * s2, z[1] * s3, o[1]],
                [x[2] * s1, y[2] * s2, z[2] * s3, o[2]],
                [0.0, 0.0, 0.0, 1.0],
            ]

        if operator.type == "IFCCARTESIANTRANSFORMATIONOPERATOR2D":
            raw_axis1 = IFCFastTopology._direction(operator.args[0] if len(operator.args) > 0 else None, entities, default=[1.0, 0.0, 0.0])
            raw_axis2 = IFCFastTopology._direction(operator.args[1] if len(operator.args) > 1 else None, entities, default=None)
            origin = IFCFastTopology._cartesian_point(operator.args[2] if len(operator.args) > 2 else None, entities) or [0.0, 0.0, 0.0]
            s1 = IFCFastTopology._as_float(operator.args[3] if len(operator.args) > 3 else 1.0, 1.0)
            x = IFCFastTopology._normalize(IFCFastTopology._normalise_point(raw_axis1 or [1.0, 0.0, 0.0]))
            if raw_axis2 is None:
                y = [-x[1], x[0], 0.0]
            else:
                y = IFCFastTopology._normalize(IFCFastTopology._normalise_point(raw_axis2))
            o = IFCFastTopology._normalise_point(origin)
            return [
                [x[0] * s1, y[0] * s1, 0.0, o[0]],
                [x[1] * s1, y[1] * s1, 0.0, o[1]],
                [0.0, 0.0, s1, o[2]],
                [0.0, 0.0, 0.0, 1.0],
            ]

        return IFCFastTopology._identity()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_topology(topology, epsilon, angTolerance, tolerance, silent):
        from topologicpy.Topology import Topology
        try:
            t_id = Topology.Type(topology)
            clean_faces = t_id > 8 or (t_id == 128 and len(Topology.Faces(topology)) >= 2)
            clean_edges = t_id > 2 or (t_id == 128 and len(Topology.Wires(topology)) > 0)
            if clean_faces:
                temp = Topology.RemoveCoplanarFaces(topology, epsilon=epsilon, tolerance=tolerance, silent=silent)
                if Topology.IsInstance(temp, "Topology"):
                    topology = temp
            if clean_edges:
                temp = Topology.RemoveCollinearEdges(topology, angTolerance=angTolerance, tolerance=tolerance, silent=silent)
                if Topology.IsInstance(temp, "Topology"):
                    topology = temp
        except Exception:
            pass
        return topology

    @staticmethod
    def _normalise_type_set(types: Optional[list]) -> set:
        if not types:
            return set()
        return {str(t).strip().upper() for t in types if t is not None and str(t).strip()}

    @staticmethod
    def _is_product_like(e: IFCFastEntity) -> bool:
        if e.type in IFCFastTopology._PRODUCT_TYPES:
            return True
        # Fallback heuristic for IfcProduct descendants: ObjectPlacement and Representation slots.
        return e.type.startswith("IFC") and not e.type.startswith("IFCREL") and len(e.args) >= 7 and IFCFastTopology._is_ref(e.args[5])

    @staticmethod
    def _root_attr(entity: IFCFastEntity, index: int) -> Any:
        if entity is None or index >= len(entity.args):
            return None
        v = entity.args[index]
        if isinstance(v, str):
            return v
        return None if v in (None, "*") else v

    @staticmethod
    def _entity_dictionary(entity: IFCFastEntity, dictionaryMode: str = "basic"):
        mode = (dictionaryMode or "none").strip().lower()
        if mode in ("none", "off", "false", "no"):
            return None
        try:
            from topologicpy.Dictionary import Dictionary
        except Exception:
            return None
        keys = ["IFC_id", "IFC_key", "IFC_type", "IFC_global_id", "IFC_name"]
        values = [entity.id, f"#{entity.id}", entity.type.lower(), IFCFastTopology._root_attr(entity, 0), IFCFastTopology._root_attr(entity, 2)]
        if mode in ("all", "full"):
            keys.append("IFC_raw_args")
            values.append(str(entity.args))
        try:
            return Dictionary.ByKeysValues(keys, values)
        except Exception:
            return None

    @staticmethod
    def _is_ref(v: Any) -> bool:
        return isinstance(v, tuple) and len(v) == 2 and v[0] == "REF"

    @staticmethod
    def _entity_from_ref(v: Any, entities: Dict[int, IFCFastEntity]) -> Optional[IFCFastEntity]:
        return entities.get(v[1]) if IFCFastTopology._is_ref(v) else None

    @staticmethod
    def _refs_in_value(v: Any) -> list:
        if IFCFastTopology._is_ref(v):
            return [v]
        out = []
        if isinstance(v, list):
            for item in v:
                out.extend(IFCFastTopology._refs_in_value(item))
        return out

    @staticmethod
    def _cartesian_point(v: Any, entities: Dict[int, IFCFastEntity]) -> Optional[list]:
        e = IFCFastTopology._entity_from_ref(v, entities)
        if e is None or e.type != "IFCCARTESIANPOINT" or not e.args:
            return None
        coords = e.args[0]
        return IFCFastTopology._normalise_point(coords) if isinstance(coords, list) else None

    @staticmethod
    def _direction(v: Any, entities: Dict[int, IFCFastEntity], default: Optional[list] = None) -> list:
        e = IFCFastTopology._entity_from_ref(v, entities)
        if e is not None and e.type == "IFCDIRECTION" and e.args and isinstance(e.args[0], list):
            return IFCFastTopology._normalise_point(e.args[0])
        if isinstance(v, list):
            return IFCFastTopology._normalise_point(v)
        return list(default) if default is not None else [0.0, 0.0, 1.0]

    @staticmethod
    def _normalise_point(coords: Sequence[Any]) -> list:
        vals = [IFCFastTopology._as_float(x, 0.0) for x in coords]
        if len(vals) == 2:
            vals.append(0.0)
        while len(vals) < 3:
            vals.append(0.0)
        return vals[:3]

    @staticmethod
    def _as_float(v: Any, default: float = 0.0) -> float:
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, tuple) and len(v) == 3 and v[0] == "CALL":
            return IFCFastTopology._as_float(v[2][0], default) if v[2] else default
        try:
            return float(v)
        except Exception:
            return float(default)

    @staticmethod
    def _as_string(v: Any) -> Optional[str]:
        return v if isinstance(v, str) else None

    @staticmethod
    def _point_key(p: Sequence[float], mantissa: int = 7) -> tuple:
        return (round(float(p[0]), mantissa), round(float(p[1]), mantissa), round(float(p[2]), mantissa))

    @staticmethod
    def _merge_meshes(meshes: list, dedupe: bool = False) -> Optional[dict]:
        if not meshes:
            return None
        vertices, edges, faces = [], [], []
        key_to_new = {}
        for mesh in meshes:
            local_map = []
            for p in mesh.get("vertices") or []:
                if dedupe:
                    key = IFCFastTopology._point_key(p)
                    idx = key_to_new.get(key)
                    if idx is None:
                        idx = len(vertices)
                        key_to_new[key] = idx
                        vertices.append(p)
                    local_map.append(idx)
                else:
                    local_map.append(len(vertices))
                    vertices.append(p)
            for a, b in mesh.get("edges") or []:
                if 0 <= a < len(local_map) and 0 <= b < len(local_map) and local_map[a] != local_map[b]:
                    edges.append([local_map[a], local_map[b]])
            for face in mesh.get("faces") or []:
                nf = [local_map[i] for i in face if 0 <= i < len(local_map)]
                if len(set(nf)) >= 3:
                    faces.append(nf)
        return {"vertices": vertices, "edges": edges, "faces": faces}

    @staticmethod
    def _identity() -> list:
        return [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]

    @staticmethod
    def _matmul(a: list, b: list) -> list:
        return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]

    @staticmethod
    def _inverse_rigid_matrix(m: list) -> list:
        """
        Inverts a 4x4 matrix whose upper-left 3x3 block is a rotation-like
        orthonormal frame. This is used for IfcRepresentationMap.MappingOrigin.
        """
        try:
            r00, r01, r02 = float(m[0][0]), float(m[0][1]), float(m[0][2])
            r10, r11, r12 = float(m[1][0]), float(m[1][1]), float(m[1][2])
            r20, r21, r22 = float(m[2][0]), float(m[2][1]), float(m[2][2])
            tx, ty, tz = float(m[0][3]), float(m[1][3]), float(m[2][3])

            itx = -(r00 * tx + r10 * ty + r20 * tz)
            ity = -(r01 * tx + r11 * ty + r21 * tz)
            itz = -(r02 * tx + r12 * ty + r22 * tz)

            return [
                [r00, r10, r20, itx],
                [r01, r11, r21, ity],
                [r02, r12, r22, itz],
                [0.0, 0.0, 0.0, 1.0],
            ]
        except Exception:
            return IFCFastTopology._identity()

    @staticmethod
    def _transform_point(p: Sequence[float], m: list, scale: float = 1.0) -> list:
        x, y, z = IFCFastTopology._normalise_point(p)
        return [
            scale * (m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3]),
            scale * (m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3]),
            scale * (m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3]),
        ]

    @staticmethod
    def _dot(a, b) -> float:
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

    @staticmethod
    def _cross(a, b) -> list:
        return [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]

    @staticmethod
    def _norm(a) -> float:
        return math.sqrt(IFCFastTopology._dot(a, a))

    @staticmethod
    def _normalize(a) -> list:
        n = IFCFastTopology._norm(a)
        return [0.0, 0.0, 0.0] if n <= 1e-12 else [a[0]/n, a[1]/n, a[2]/n]

    @staticmethod
    def _points_close(a, b, tol: float = 1e-7) -> bool:
        return IFCFastTopology._norm([a[0]-b[0], a[1]-b[1], a[2]-b[2]]) <= tol


class _STEPArgParser:
    __slots__ = ("text", "i", "n")

    def __init__(self, text: str):
        self.text = text
        self.i = 0
        self.n = len(text)

    def parse_list_content(self) -> list:
        values = []
        while True:
            self._skip_ws()
            if self.i >= self.n:
                break
            values.append(self._parse_value())
            self._skip_ws()
            if self.i < self.n and self.text[self.i] == ',':
                self.i += 1
                continue
            break
        return values

    def _parse_value(self) -> Any:
        self._skip_ws()
        if self.i >= self.n:
            return None
        c = self.text[self.i]
        if c == '$':
            self.i += 1
            return None
        if c == '*':
            self.i += 1
            return '*'
        if c == '#':
            return self._parse_ref()
        if c == "'":
            return self._parse_string()
        if c == '(':
            return self._parse_list()
        if c == '.':
            return self._parse_enum_or_bool()
        if c.isalpha() or c == '_':
            return self._parse_call_or_word()
        return self._parse_number_or_word()

    def _parse_ref(self) -> Ref:
        self.i += 1
        start = self.i
        while self.i < self.n and self.text[self.i].isdigit():
            self.i += 1
        return ("REF", int(self.text[start:self.i]))

    def _parse_string(self) -> str:
        self.i += 1
        chars = []
        while self.i < self.n:
            c = self.text[self.i]
            if c == "'":
                if self.i + 1 < self.n and self.text[self.i + 1] == "'":
                    chars.append("'")
                    self.i += 2
                    continue
                self.i += 1
                break
            chars.append(c)
            self.i += 1
        return ''.join(chars)

    def _parse_list(self) -> list:
        self.i += 1
        values = []
        while True:
            self._skip_ws()
            if self.i < self.n and self.text[self.i] == ')':
                self.i += 1
                break
            values.append(self._parse_value())
            self._skip_ws()
            if self.i < self.n and self.text[self.i] == ',':
                self.i += 1
                continue
            if self.i < self.n and self.text[self.i] == ')':
                self.i += 1
                break
            if self.i >= self.n:
                break
        return values

    def _parse_enum_or_bool(self):
        self.i += 1
        start = self.i
        while self.i < self.n and self.text[self.i] != '.':
            self.i += 1
        token = self.text[start:self.i].upper()
        if self.i < self.n and self.text[self.i] == '.':
            self.i += 1
        if token == 'T':
            return True
        if token == 'F':
            return False
        if token == 'U':
            return None
        return token

    def _parse_call_or_word(self):
        start = self.i
        while self.i < self.n and (self.text[self.i].isalnum() or self.text[self.i] in ['_', '-']):
            self.i += 1
        word = self.text[start:self.i].upper()
        self._skip_ws()
        if self.i < self.n and self.text[self.i] == '(':
            args = self._parse_list()
            return ("CALL", word, args)
        return word

    def _parse_number_or_word(self):
        start = self.i
        while self.i < self.n and self.text[self.i] not in ',)':
            self.i += 1
        token = self.text[start:self.i].strip()
        if not token:
            return None
        try:
            if any(ch in token for ch in '.Ee'):
                return float(token)
            return int(token)
        except Exception:
            return token

    def _skip_ws(self):
        while self.i < self.n and self.text[self.i].isspace():
            self.i += 1


class IFC:
    @staticmethod
    def TopologiesByFile(
        file,
        includeTypes: list = [],
        excludeTypes: list = [],
        dictionaryMode: str = "basic",
        clean: bool = False,
        epsilon: float = 0.01,
        angTolerance: float = 0.1,
        tolerance: float = 0.0001,
        silent: bool = False,
    ) -> list[Any]:
        """Import an IFC file into a list of TopologicPy topologies using the pure-Python STEP-text importer."""
        if isinstance(file, dict):
            return IFCFastTopology.TopologiesByEntities(
                file, includeTypes=includeTypes, excludeTypes=excludeTypes, dictionaryMode=dictionaryMode,
                clean=clean, epsilon=epsilon, angTolerance=angTolerance, tolerance=tolerance, silent=silent)
        return IFCFastTopology.TopologiesByFile(
            file, includeTypes=includeTypes, excludeTypes=excludeTypes, dictionaryMode=dictionaryMode,
            clean=clean, epsilon=epsilon, angTolerance=angTolerance, tolerance=tolerance, silent=silent)

    @staticmethod
    def TopologiesByPath(path: str,
                  includeTypes: list = [],
                  excludeTypes: list = [],
                  dictionaryMode: str = "basic",
                  clean: bool = False,
                  epsilon: float = 0.0001,
                  angTolerance: float = 0.1,
                  tolerance: float = 0.0001,
                  silent: bool = False) -> list[Any]:
        """Imports the topologies from an IFC file path using the pure-Python STEP-text importer."""
        if not path or not isinstance(path, str) or not os.path.exists(path):
            if not silent:
                print("IFC.TopologiesByPath - Error: the input path parameter is not a valid path. Returning None.")
            return None
        return IFCFastTopology.TopologiesByPath(
            path, includeTypes=includeTypes, excludeTypes=excludeTypes, dictionaryMode=dictionaryMode,
            clean=clean, epsilon=epsilon, angTolerance=angTolerance, tolerance=tolerance, silent=silent)

    @staticmethod
    def FileByPath(path: str, silent: bool = False):
        """Returns the parsed IFC STEP entity dictionary found at the input path."""
        if not path or not isinstance(path, str) or not os.path.exists(path):
            if not silent:
                print("IFC.FileByPath - Error: Could not open the IFC file. Returning None.")
            return None
        return IFCFastTopology.Parse(path, silent=silent)

    @staticmethod
    def ElementTypes(file, includeCounts: bool = False, silent: bool = False):
        """Returns the IFC element/entity types found in the input IFC file, sorted alphabetically."""
        entities = IFC._entities_from_input(file, silent=silent)
        if entities is None:
            if not silent:
                print("IFC.ElementTypes - Error: Could not read the input IFC file. Returning None.")
            return None
        type_counts = {}
        for element in entities.values():
            element_type = IFC._display_type(element.type)
            type_counts[element_type] = type_counts.get(element_type, 0) + 1
        if includeCounts:
            return [{"type": key, "count": type_counts[key]} for key in sorted(type_counts.keys(), key=lambda x: x.lower())]
        return sorted(type_counts.keys(), key=lambda x: x.lower())

    @staticmethod
    def RelationshipTypes(file, includeCounts: bool = False, silent: bool = False):
        """Returns the IFC relationship entity types found in the input IFC file, sorted alphabetically."""
        entities = IFC._entities_from_input(file, silent=silent)
        if entities is None:
            if not silent:
                print("IFC.RelationshipTypes - Error: Could not read the input IFC file. Returning None.")
            return None
        type_counts = {}
        for element in entities.values():
            element_type = element.is_a()
            if element_type.lower().startswith("ifcrel"):
                type_counts[element_type] = type_counts.get(element_type, 0) + 1
        if includeCounts:
            return [{"type": key, "count": type_counts[key]} for key in sorted(type_counts.keys(), key=lambda x: x.lower())]
        return sorted(type_counts.keys(), key=lambda x: x.lower())

    @staticmethod
    def RelationshipTypesByGlobalId(file, globalId: str, includeCounts: bool = False, includeRelationshipIds: bool = False, silent: bool = False):
        """Returns the IFC relationship entity types associated with the IFC element identified by the input GlobalId."""
        if not isinstance(globalId, str) or len(globalId.strip()) < 1:
            if not silent:
                print("IFC.RelationshipTypesByGlobalId - Error: The input globalId parameter is not a valid string. Returning None.")
            return None
        entities = IFC._entities_from_input(file, silent=silent)
        if entities is None:
            if not silent:
                print("IFC.RelationshipTypesByGlobalId - Error: Could not read the input IFC file. Returning None.")
            return None
        target = None
        gid = globalId.strip()
        for entity in entities.values():
            if IFCFastTopology._root_attr(entity, 0) == gid:
                target = entity
                break
        if target is None:
            if not silent:
                print(f"IFC.RelationshipTypesByGlobalId - Error: Could not find an IFC element with GlobalId '{globalId}'. Returning None.")
            return None
        target_ref = ("REF", target.id)
        relationship_data = {}
        for entity in entities.values():
            rel_type = IFC._display_type(entity.type)
            if not rel_type.startswith("IfcRel") or not IFC._value_contains_ref(entity.args, target_ref):
                continue
            relationship_data.setdefault(rel_type, {"count": 0, "globalIds": []})
            relationship_data[rel_type]["count"] += 1
            if includeRelationshipIds:
                rel_gid = IFCFastTopology._root_attr(entity, 0)
                if rel_gid:
                    relationship_data[rel_type]["globalIds"].append(rel_gid)
        sorted_types = sorted(relationship_data.keys(), key=lambda x: x.lower())
        if includeRelationshipIds:
            return [{"type": rel_type, "count": relationship_data[rel_type]["count"], "globalIds": sorted(relationship_data[rel_type]["globalIds"], key=lambda x: x.lower())} for rel_type in sorted_types]
        if includeCounts:
            return [{"type": rel_type, "count": relationship_data[rel_type]["count"]} for rel_type in sorted_types]
        return sorted_types

    @staticmethod
    def ExportReferenceView(
        model: Any, path: str, schema: str = "IFC4", mvd: str = "RV1.2",
        project_name: str = "TopologicPy Project", site_name: str = "Default Site",
        building_name: str = "Default Building", storey_name: str = "Level 0",
        length_unit: str = "METRE", angle_unit: str = "RADIAN", area_unit: str = "SQUARE_METRE", volume_unit: str = "CUBIC_METRE",
        use_sweeps: bool = True, use_tessellation_fallback: bool = True, transfer_dictionaries: bool = True, validate: bool = True, silent: bool = False) -> str:
        """Export a TopologicPy-authored model to an IFC4 Reference View-compatible IFC."""
        if IFCExportConfig is None or IFCReferenceViewExporter is None:
            if not silent:
                print("IFC.ExportReferenceView - Error: Could not import topologicpy.ifc.exporter. Returning None.")
            return None
        cfg = IFCExportConfig(
            schema=schema, mvd=mvd, project_name=project_name, site_name=site_name, building_name=building_name, storey_name=storey_name,
            units={"length": length_unit, "angle": angle_unit, "area": area_unit, "volume": volume_unit},
            use_sweeps=use_sweeps, use_tessellation_fallback=use_tessellation_fallback, transfer_dictionaries=transfer_dictionaries, validate=validate, silent=silent)
        exporter = IFCReferenceViewExporter(cfg)
        exporter.export(model=model, path=path)
        return path

    @staticmethod
    def SummaryByPath(path: str, silent: bool = False) -> dict:
        """Returns a compact summary of the IFC file using the pure-Python parser."""
        if not path or not isinstance(path, str) or not os.path.exists(path):
            if not silent:
                print("IFC.SummaryByPath - Error: the input path parameter is not a valid path. Returning None.")
            return None
        return IFCFastTopology.SummaryByPath(path, silent=silent)

    @staticmethod
    def MeshDataByPath(path: str, includeTypes: list = [], excludeTypes: list = [], dictionaryMode: str = "basic", silent: bool = False) -> dict:
        """Returns extracted mesh data using the pure-Python parser."""
        if not path or not isinstance(path, str) or not os.path.exists(path):
            if not silent:
                print("IFC.MeshDataByPath - Error: the input path parameter is not a valid path. Returning None.")
            return None
        return IFCFastTopology.MeshDataByPath(path, includeTypes=includeTypes, excludeTypes=excludeTypes, dictionaryMode=dictionaryMode, silent=silent)

    @staticmethod
    def _entities_from_input(file, silent: bool = False):
        if isinstance(file, dict):
            return file
        if isinstance(file, str):
            if os.path.exists(file):
                return IFCFastTopology.Parse(file, silent=silent)
            if "ISO-10303-21" in file:
                return IFCFastTopology.ParseText(file, silent=silent)
            return None
        text = IFCFastTopology._file_to_step_text(file)
        if text:
            return IFCFastTopology.ParseText(text, silent=silent)
        return None

    @staticmethod
    def _display_type(entity_type: str) -> str:
        s = str(entity_type or "").upper()
        if not s.startswith("IFC"):
            return s
        # Keep conventional IFC spelling for the prefix, while preserving the rest of the token as supplied.
        return "Ifc" + s[3:]

    @staticmethod
    def _value_contains_ref(value, ref) -> bool:
        if value == ref:
            return True
        if isinstance(value, list):
            return any(IFC._value_contains_ref(v, ref) for v in value)
        if isinstance(value, tuple):
            return any(IFC._value_contains_ref(v, ref) for v in value)
        return False
