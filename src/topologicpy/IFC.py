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

class IFCEntity:
    """
    Lightweight dynamic wrapper around an IFCFastEntity.

    This class exposes IFC root attributes, extracted properties,
    quantities, classifications, materials, and type information
    through Python attribute access.

    Example
    -------
    wall = IFC.Object(file, globalId)

    print(wall.GlobalId)
    print(wall.Name)
    print(wall.FireRating)
    print(wall.Materials)
    """

    _ROOT_ATTRS = {
        "GlobalId": 0,
        "OwnerHistory": 1,
        "Name": 2,
        "Description": 3,
    }

    def __init__(self, entity, entities=None, properties=None):
        self._entity = entity
        self._entities = entities or {}
        self._properties = properties or {}

    def __repr__(self):
        gid = self.GlobalId or ""
        return f"<{self.IFCType} #{self.IFCId} {gid}>"

    @property
    def IFCId(self):
        return self._entity.id if self._entity is not None else None

    @property
    def IFCKey(self):
        return f"#{self._entity.id}" if self._entity is not None else None

    @property
    def IFCType(self):
        if self._entity is None:
            return None

        t = self._entity.type

        if str(t).upper().startswith("IFC"):
            return "Ifc" + str(t)[3:].lower().title().replace("_", "")

        return t

    @property
    def Entity(self):
        """
        Returns the wrapped IFCFastEntity.
        """
        return self._entity

    @property
    def Properties(self):
        return self._properties.get("properties", {})

    @property
    def TypeProperties(self):
        return self._properties.get("type_properties", {})

    @property
    def Quantities(self):
        return self._properties.get("quantities", {})

    @property
    def TypeQuantities(self):
        return self._properties.get("type_quantities", {})

    @property
    def Classifications(self):
        return self._properties.get("classifications", [])

    @property
    def Materials(self):
        return self._properties.get("materials", [])

    @property
    def Type(self):
        return self._properties.get("type", None)

    def is_a(self, name=None):
        if self._entity is None:
            return None if name is None else False

        return self._entity.is_a(name)

    def Attribute(self, name, default=None):
        """
        Returns an attribute/property safely.
        """

        try:
            return getattr(self, name)
        except Exception:
            return default

    def ToDict(self):
        """
        Returns a serializable dictionary representation.
        """

        return {
            "ifc_id": self.IFCId,
            "ifc_key": self.IFCKey,
            "ifc_type": self.IFCType,
            "global_id": self.GlobalId,
            "name": self.Name,
            "description": self.Description,
            "properties": self.Properties,
            "type_properties": self.TypeProperties,
            "quantities": self.Quantities,
            "type_quantities": self.TypeQuantities,
            "classifications": self.Classifications,
            "materials": self.Materials,
            "type": self.Type,
        }

    def _normalise_name(self, value):
        if value is None:
            return ""

        return "".join(
            ch for ch in str(value).strip().lower()
            if ch.isalnum()
        )

    def _lookup_in_grouped_dict(self, grouped, name):
        """
        Searches grouped property dictionaries.

        Supports:
        - FireRating
        - Pset_WallCommon.FireRating
        """

        target = self._normalise_name(name)

        if not isinstance(grouped, dict):
            return None

        # Direct property search.
        for group_name, values in grouped.items():
            if not isinstance(values, dict):
                continue

            for key, value in values.items():
                if self._normalise_name(key) == target:
                    return value

        # Qualified search.
        if "." in str(name):
            group_part, key_part = str(name).split(".", 1)

            group_target = self._normalise_name(group_part)
            key_target = self._normalise_name(key_part)

            for group_name, values in grouped.items():
                if self._normalise_name(group_name) != group_target:
                    continue

                if not isinstance(values, dict):
                    continue

                for key, value in values.items():
                    if self._normalise_name(key) == key_target:
                        return value

        return None

    def __getattr__(self, name):
        """
        Dynamic IFC-style attribute lookup.

        Lookup order:
        1. IFC root attributes
        2. instance properties
        3. type properties
        4. instance quantities
        5. type quantities
        """

        # IFC root attributes.
        if name in IFCEntity._ROOT_ATTRS:
            index = IFCEntity._ROOT_ATTRS[name]

            try:
                return IFCFastTopology._root_attr(self._entity, index)
            except Exception:
                return None

        # Instance properties.
        value = self._lookup_in_grouped_dict(
            self.Properties,
            name
        )

        if value is not None:
            return value

        # Type properties.
        value = self._lookup_in_grouped_dict(
            self.TypeProperties,
            name
        )

        if value is not None:
            return value

        # Instance quantities.
        value = self._lookup_in_grouped_dict(
            self.Quantities,
            name
        )

        if value is not None:
            return value

        # Type quantities.
        value = self._lookup_in_grouped_dict(
            self.TypeQuantities,
            name
        )

        if value is not None:
            return value

        raise AttributeError(
            f"{self.IFCType} has no attribute '{name}'"
        )

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
        metadata_cache = IFCFastTopology._entity_metadata_cache(entities, dictionaryMode=dictionaryMode)
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

            d = IFCFastTopology._entity_dictionary(product, dictionaryMode=dictionaryMode, metadataCache=metadata_cache)
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
        metadata_cache = IFCFastTopology._entity_metadata_cache(entities, dictionaryMode=dictionaryMode)
        for product in products:
            mesh = IFCFastTopology.MeshDataByProduct(product, entities, circleSides=circleSides, scale=scale)
            if not mesh or not mesh.get("vertices"):
                continue
            d = IFCFastTopology._entity_dictionary(product, dictionaryMode, metadataCache=metadata_cache)
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
    def _dictionary_key(value) -> str:
        """
        Returns a safe, flat dictionary key component.
        """
        if value in [None, "", "*"]:
            return ""
        value = str(value).strip()
        value = re.sub(r"[^0-9A-Za-z_]+", "_", value)
        value = re.sub(r"_+", "_", value).strip("_")
        return value

    @staticmethod
    def _dictionary_value(value, entities: Dict[int, IFCFastEntity] = None):
        """
        Converts parsed IFC STEP values to TopologicPy Dictionary-friendly values.

        The STEP parser represents references as ("REF", id) and typed IFC values
        as ("CALL", type, args). For dictionary storage we keep scalar values as
        scalars, convert references to "#id", unwrap one-argument typed values,
        and stringify complex nested values.
        """
        if value in [None, "*", "$"]:
            return ""

        if isinstance(value, (str, int, float, bool)):
            return value

        if IFCFastTopology._is_ref(value):
            return f"#{value[1]}"

        if isinstance(value, tuple):
            if len(value) == 3 and value[0] == "CALL":
                call_name = str(value[1])
                args = value[2] or []
                if len(args) == 1:
                    return IFCFastTopology._dictionary_value(args[0], entities=entities)
                clean_args = [IFCFastTopology._dictionary_value(v, entities=entities) for v in args]
                return f"{call_name}({', '.join(str(v) for v in clean_args)})"
            return str(value)

        if isinstance(value, (list, set)):
            clean_values = [IFCFastTopology._dictionary_value(v, entities=entities) for v in value]
            if all(isinstance(v, (str, int, float, bool)) for v in clean_values):
                return ", ".join(str(v) for v in clean_values if str(v) != "")
            return str(clean_values)

        try:
            return str(value)
        except Exception:
            return ""

    @staticmethod
    def _entity_metadata_cache(entities: Dict[int, IFCFastEntity], dictionaryMode: str = "basic") -> dict:
        """
        Builds a flat metadata cache for IFC entities.

        The cache is intentionally built once per import. It captures relationship-
        based semantic data that cannot be discovered from an entity alone:
        property sets, type property sets, quantities, type quantities, materials,
        classifications, and type assignments.
        """
        mode = (dictionaryMode or "none").strip().lower()
        if mode not in ("all", "full"):
            return {}

        if entities is None or not isinstance(entities, dict):
            return {}

        def _normalise_type(value):
            if value is None:
                return ""
            return str(value).strip().upper()

        def _refs(value):
            try:
                return IFCFastTopology._refs_in_value(value)
            except Exception:
                return []

        def _entity_from_ref(ref):
            try:
                return IFCFastTopology._entity_from_ref(ref, entities)
            except Exception:
                return None

        def _root_attr(entity, index):
            try:
                return IFCFastTopology._root_attr(entity, index)
            except Exception:
                return None

        def _name(entity, fallback):
            value = _root_attr(entity, 2)
            if value in [None, "", "*"]:
                return fallback
            return str(value)

        def _clean_key(*parts):
            clean = [IFCFastTopology._dictionary_key(p) for p in parts]
            clean = [p for p in clean if p]
            return "_".join(clean)

        def _clean_value(value):
            return IFCFastTopology._dictionary_value(value, entities=entities)

        def _property_name(prop):
            if prop is None:
                return ""
            try:
                return str(prop.args[0]) if len(prop.args) > 0 and prop.args[0] not in [None, "", "*"] else ""
            except Exception:
                return ""

        def _property_value(prop):
            if prop is None:
                return ""
            ptype = _normalise_type(prop.type)
            args = prop.args or []

            if ptype == "IFCPROPERTYSINGLEVALUE":
                return _clean_value(args[2] if len(args) > 2 else "")
            if ptype == "IFCPROPERTYENUMERATEDVALUE":
                return _clean_value(args[2] if len(args) > 2 else "")
            if ptype == "IFCPROPERTYLISTVALUE":
                return _clean_value(args[2] if len(args) > 2 else "")
            if ptype == "IFCPROPERTYBOUNDEDVALUE":
                upper = _clean_value(args[2] if len(args) > 2 else "")
                lower = _clean_value(args[3] if len(args) > 3 else "")
                setpoint = _clean_value(args[5] if len(args) > 5 else "")
                return f"LowerBoundValue={lower}; UpperBoundValue={upper}; SetPointValue={setpoint}"
            if ptype == "IFCPROPERTYTABLEVALUE":
                defining = _clean_value(args[2] if len(args) > 2 else "")
                defined = _clean_value(args[3] if len(args) > 3 else "")
                return f"DefiningValues={defining}; DefinedValues={defined}"
            if ptype == "IFCCOMPLEXPROPERTY":
                values = []
                for child_ref in _refs(args[3] if len(args) > 3 else None):
                    child = _entity_from_ref(child_ref)
                    child_name = _property_name(child)
                    if child_name:
                        values.append(f"{child_name}={_property_value(child)}")
                return "; ".join(values)
            return _clean_value(args)

        def _quantity_name(q):
            if q is None:
                return ""
            try:
                return str(q.args[0]) if len(q.args) > 0 and q.args[0] not in [None, "", "*"] else ""
            except Exception:
                return ""

        def _quantity_value(q):
            if q is None:
                return ""
            qtype = _normalise_type(q.type)
            args = q.args or []
            value_slots = {
                "IFCQUANTITYLENGTH": 3,
                "IFCQUANTITYAREA": 3,
                "IFCQUANTITYVOLUME": 3,
                "IFCQUANTITYCOUNT": 3,
                "IFCQUANTITYWEIGHT": 3,
                "IFCQUANTITYTIME": 3,
            }
            slot = value_slots.get(qtype, 3)
            return _clean_value(args[slot] if len(args) > slot else "")

        def _material_name(material):
            if material is None:
                return None
            mtype = _normalise_type(material.type)
            args = material.args or []
            if mtype == "IFCMATERIAL":
                return _clean_value(args[0] if len(args) > 0 else "")
            if mtype in ("IFCMATERIALLIST", "IFCMATERIALLAYERSET", "IFCMATERIALPROFILESET", "IFCMATERIALCONSTITUENTSET"):
                names = []
                for ref in _refs(args[0] if len(args) > 0 else None):
                    child = _entity_from_ref(ref)
                    child_name = _material_name(child)
                    if child_name:
                        names.append(str(child_name))
                return ", ".join(names) if names else _clean_value(args)
            if mtype in ("IFCMATERIALLAYER", "IFCMATERIALPROFILE", "IFCMATERIALCONSTITUENT"):
                for arg in args:
                    for ref in _refs(arg):
                        child = _entity_from_ref(ref)
                        if child is not None and _normalise_type(child.type).startswith("IFCMATERIAL"):
                            child_name = _material_name(child)
                            if child_name:
                                return child_name
                return _clean_value(args)
            return _clean_value(args)

        def _classification_label(classification):
            if classification is None:
                return None
            args = classification.args or []
            ctype = _normalise_type(classification.type)
            # IfcClassificationReference commonly has Location/Identification/Name.
            candidates = []
            for i in range(min(len(args), 4)):
                v = _clean_value(args[i])
                if v not in [None, ""]:
                    candidates.append(str(v))
            if not candidates:
                return ctype
            return " | ".join(candidates)

        property_sets = {}
        quantity_sets = {}

        for entity in entities.values():
            etype = _normalise_type(entity.type)
            args = entity.args or []

            if etype == "IFCPROPERTYSET":
                pset_name = _name(entity, f"IfcPropertySet_{entity.id}")
                values = {}
                for prop_ref in _refs(args[4] if len(args) > 4 else None):
                    prop = _entity_from_ref(prop_ref)
                    pname = _property_name(prop)
                    if pname:
                        values[pname] = _property_value(prop)
                property_sets[entity.id] = {"name": pset_name, "values": values}

            elif etype == "IFCELEMENTQUANTITY":
                qset_name = _name(entity, f"IfcElementQuantity_{entity.id}")
                values = {}
                for q_ref in _refs(args[5] if len(args) > 5 else None):
                    q = _entity_from_ref(q_ref)
                    qname = _quantity_name(q)
                    if qname:
                        values[qname] = _quantity_value(q)
                quantity_sets[entity.id] = {"name": qset_name, "values": values}

        cache = {}

        def _ensure(entity):
            if entity is None:
                return None
            eid = getattr(entity, "id", None)
            if eid is None:
                return None
            if eid not in cache:
                cache[eid] = {}
            return cache[eid]

        def _add_group(record, prefix, group_name, values):
            if record is None or not isinstance(values, dict):
                return
            for key, value in values.items():
                flat_key = _clean_key(prefix, group_name, key)
                if flat_key:
                    record[flat_key] = value

        # Instance property and quantity assignments.
        for rel in entities.values():
            if _normalise_type(rel.type) != "IFCRELDEFINESBYPROPERTIES" or len(rel.args) <= 5:
                continue
            related_refs = _refs(rel.args[4])
            definition_refs = _refs(rel.args[5])
            for definition_ref in definition_refs:
                definition = _entity_from_ref(definition_ref)
                if definition is None:
                    continue
                pset = property_sets.get(definition.id)
                qset = quantity_sets.get(definition.id)
                for related_ref in related_refs:
                    related = _entity_from_ref(related_ref)
                    record = _ensure(related)
                    if pset is not None:
                        _add_group(record, "IFC_pset", pset["name"], pset["values"])
                    if qset is not None:
                        _add_group(record, "IFC_qto", qset["name"], qset["values"])

        # Type property and quantity assignments.
        type_payload = {}

        def _type_payload(type_entity):
            if type_entity is None:
                return None
            if type_entity.id in type_payload:
                return type_payload[type_entity.id]
            payload = {
                "ifc_id": type_entity.id,
                "ifc_key": f"#{type_entity.id}",
                "ifc_type": str(type_entity.type).lower(),
                "global_id": _root_attr(type_entity, 0),
                "name": _root_attr(type_entity, 2),
                "properties": {},
                "quantities": {},
            }
            for slot in (5, 6):
                if len(type_entity.args) <= slot:
                    continue
                for definition_ref in _refs(type_entity.args[slot]):
                    definition = _entity_from_ref(definition_ref)
                    if definition is None:
                        continue
                    pset = property_sets.get(definition.id)
                    qset = quantity_sets.get(definition.id)
                    if pset is not None:
                        payload["properties"][pset["name"]] = dict(pset["values"])
                    if qset is not None:
                        payload["quantities"][qset["name"]] = dict(qset["values"])
            type_payload[type_entity.id] = payload
            return payload

        for rel in entities.values():
            if _normalise_type(rel.type) != "IFCRELDEFINESBYTYPE" or len(rel.args) <= 5:
                continue
            for type_ref in _refs(rel.args[5]):
                type_entity = _entity_from_ref(type_ref)
                payload = _type_payload(type_entity)
                if payload is None:
                    continue
                for related_ref in _refs(rel.args[4]):
                    related = _entity_from_ref(related_ref)
                    record = _ensure(related)
                    if record is None:
                        continue
                    record["IFC_type_object_id"] = payload["ifc_id"]
                    record["IFC_type_object_key"] = payload["ifc_key"]
                    record["IFC_type_object_type"] = payload["ifc_type"]
                    record["IFC_type_object_global_id"] = _clean_value(payload["global_id"])
                    record["IFC_type_object_name"] = _clean_value(payload["name"])
                    for group_name, values in payload["properties"].items():
                        _add_group(record, "IFC_type_pset", group_name, values)
                    for group_name, values in payload["quantities"].items():
                        _add_group(record, "IFC_type_qto", group_name, values)

        # Materials.
        for rel in entities.values():
            if _normalise_type(rel.type) != "IFCRELASSOCIATESMATERIAL" or len(rel.args) <= 5:
                continue
            names = []
            for material_ref in _refs(rel.args[5]):
                material = _entity_from_ref(material_ref)
                name = _material_name(material)
                if name:
                    names.append(str(name))
            if not names:
                continue
            material_value = ", ".join(dict.fromkeys(names))
            for related_ref in _refs(rel.args[4]):
                related = _entity_from_ref(related_ref)
                record = _ensure(related)
                if record is not None:
                    record["IFC_materials"] = material_value

        # Classifications.
        for rel in entities.values():
            if _normalise_type(rel.type) != "IFCRELASSOCIATESCLASSIFICATION" or len(rel.args) <= 5:
                continue
            labels = []
            for classification_ref in _refs(rel.args[5]):
                classification = _entity_from_ref(classification_ref)
                label = _classification_label(classification)
                if label:
                    labels.append(str(label))
            if not labels:
                continue
            classification_value = ", ".join(dict.fromkeys(labels))
            for related_ref in _refs(rel.args[4]):
                related = _entity_from_ref(related_ref)
                record = _ensure(related)
                if record is not None:
                    record["IFC_classifications"] = classification_value

        return cache

    @staticmethod
    def _entity_dictionary(entity: IFCFastEntity,
                           dictionaryMode: str = "basic",
                           metadataCache: dict = None):
        """
        Returns a TopologicPy Dictionary for an IFCFastEntity.

        In "basic" mode this preserves the previous compact behaviour. In
        "all" or "full" mode it stores all positional IFC arguments as
        individually retrievable keys, adds common semantic aliases, and merges
        relationship-derived metadata from metadataCache when supplied.
        """
        mode = (dictionaryMode or "none").strip().lower()
        if mode in ("none", "off", "false", "no"):
            return None
        try:
            from topologicpy.Dictionary import Dictionary
        except Exception:
            return None
        if entity is None:
            return None

        def _value(value):
            return IFCFastTopology._dictionary_value(value)

        def _add(keys, values, key, value):
            if key is None or key == "":
                return
            if key in keys:
                return
            keys.append(key)
            values.append(_value(value))

        def _add_arg_alias(keys, values, key, index):
            try:
                value = entity.args[index] if index < len(entity.args) else ""
            except Exception:
                value = ""
            _add(keys, values, key, value)

        ifc_type = str(getattr(entity, "type", "")).strip().upper()
        ifc_type_lower = ifc_type.lower()

        keys = []
        values = []

        _add(keys, values, "IFC_id", getattr(entity, "id", ""))
        _add(keys, values, "IFC_key", f"#{getattr(entity, 'id', '')}")
        _add(keys, values, "IFC_type", ifc_type_lower)
        _add(keys, values, "IFC_type_upper", ifc_type)
        _add(keys, values, "IFC_global_id", IFCFastTopology._root_attr(entity, 0))
        _add(keys, values, "IFC_name", IFCFastTopology._root_attr(entity, 2))

        name = IFCFastTopology._root_attr(entity, 2)
        gid = IFCFastTopology._root_attr(entity, 0)
        label = name if name not in [None, "", "*"] else gid
        if label in [None, "", "*"]:
            label = f"{ifc_type_lower}_{getattr(entity, 'id', '')}"
        _add(keys, values, "label", label)
        _add(keys, values, "type", ifc_type_lower)
        _add(keys, values, "category", ifc_type_lower)

        if mode not in ("all", "full"):
            try:
                return Dictionary.ByKeysValues(keys, values)
            except Exception:
                return None

        # Store all IFC positional arguments as separate retrievable keys.
        try:
            for i, arg in enumerate(entity.args):
                _add(keys, values, f"IFC_arg_{i}", arg)
        except Exception:
            pass

        # Common IfcRoot / IfcObject / IfcProduct aliases.
        _add_arg_alias(keys, values, "IFC_owner_history", 1)
        _add_arg_alias(keys, values, "IFC_description", 3)
        _add_arg_alias(keys, values, "IFC_object_type", 4)
        _add_arg_alias(keys, values, "IFC_object_placement", 5)
        _add_arg_alias(keys, values, "IFC_representation", 6)
        _add_arg_alias(keys, values, "IFC_tag", 7)

        # Product aliases.
        if ifc_type in IFCFastTopology._PRODUCT_TYPES or IFCFastTopology._is_product_like(entity):
            _add_arg_alias(keys, values, "IFC_product_object_placement", 5)
            _add_arg_alias(keys, values, "IFC_product_representation", 6)

        # Common element aliases.
        if any(token in ifc_type for token in ["WALL", "DOOR", "WINDOW", "SLAB", "COLUMN", "BEAM", "MEMBER", "OPENING", "BUILDINGELEMENT"]):
            _add_arg_alias(keys, values, "IFC_element_tag", 7)

        if ifc_type == "IFCSPACE":
            _add_arg_alias(keys, values, "IFC_long_name", 7)
            _add_arg_alias(keys, values, "IFC_composition_type", 8)
            _add_arg_alias(keys, values, "IFC_interior_or_exterior_space", 9)
            _add_arg_alias(keys, values, "IFC_elevation_with_flooring", 10)

        if ifc_type == "IFCOPENINGELEMENT":
            _add_arg_alias(keys, values, "IFC_opening_tag", 7)
            _add_arg_alias(keys, values, "IFC_opening_predefined_type", 8)

        if ifc_type in ("IFCDOOR", "IFCDOORSTANDARDCASE"):
            _add_arg_alias(keys, values, "IFC_door_tag", 7)
            _add_arg_alias(keys, values, "IFC_overall_height", 8)
            _add_arg_alias(keys, values, "IFC_overall_width", 9)
            _add_arg_alias(keys, values, "IFC_door_predefined_type", 10)
            _add_arg_alias(keys, values, "IFC_door_operation_type", 11)
            _add_arg_alias(keys, values, "IFC_user_defined_operation_type", 12)

        if ifc_type in ("IFCWINDOW", "IFCWINDOWSTANDARDCASE"):
            _add_arg_alias(keys, values, "IFC_window_tag", 7)
            _add_arg_alias(keys, values, "IFC_overall_height", 8)
            _add_arg_alias(keys, values, "IFC_overall_width", 9)
            _add_arg_alias(keys, values, "IFC_window_predefined_type", 10)
            _add_arg_alias(keys, values, "IFC_partitioning_type", 11)
            _add_arg_alias(keys, values, "IFC_user_defined_partitioning_type", 12)

        if ifc_type in ("IFCWALL", "IFCWALLSTANDARDCASE", "IFCWALLELEMENTEDCASE"):
            _add_arg_alias(keys, values, "IFC_wall_tag", 7)
            _add_arg_alias(keys, values, "IFC_wall_predefined_type", 8)

        if ifc_type == "IFCSLAB":
            _add_arg_alias(keys, values, "IFC_slab_tag", 7)
            _add_arg_alias(keys, values, "IFC_slab_predefined_type", 8)

        if ifc_type in ("IFCRELSPACEBOUNDARY", "IFCRELSPACEBOUNDARY1STLEVEL", "IFCRELSPACEBOUNDARY2NDLEVEL"):
            _add_arg_alias(keys, values, "IFC_relating_space", 4)
            _add_arg_alias(keys, values, "IFC_related_building_element", 5)
            _add_arg_alias(keys, values, "IFC_connection_geometry", 6)
            _add_arg_alias(keys, values, "IFC_physical_or_virtual_boundary", 7)
            _add_arg_alias(keys, values, "IFC_internal_or_external_boundary", 8)
            _add_arg_alias(keys, values, "IFC_parent_boundary", 9)
            _add_arg_alias(keys, values, "IFC_corresponding_boundary", 10)

        if ifc_type == "IFCRELFILLSELEMENT":
            _add_arg_alias(keys, values, "IFC_relating_opening_element", 4)
            _add_arg_alias(keys, values, "IFC_related_building_element", 5)

        if ifc_type == "IFCRELVOIDSELEMENT":
            _add_arg_alias(keys, values, "IFC_relating_building_element", 4)
            _add_arg_alias(keys, values, "IFC_related_opening_element", 5)

        if ifc_type == "IFCRELCONTAINEDINSPATIALSTRUCTURE":
            _add_arg_alias(keys, values, "IFC_related_elements", 4)
            _add_arg_alias(keys, values, "IFC_relating_structure", 5)

        if ifc_type == "IFCRELAGGREGATES":
            _add_arg_alias(keys, values, "IFC_relating_object", 4)
            _add_arg_alias(keys, values, "IFC_related_objects", 5)

        if ifc_type == "IFCRELDEFINESBYPROPERTIES":
            _add_arg_alias(keys, values, "IFC_related_objects", 4)
            _add_arg_alias(keys, values, "IFC_relating_property_definition", 5)

        if ifc_type == "IFCRELDEFINESBYTYPE":
            _add_arg_alias(keys, values, "IFC_related_objects", 4)
            _add_arg_alias(keys, values, "IFC_relating_type", 5)

        if ifc_type == "IFCPRODUCTDEFINITIONSHAPE":
            _add_arg_alias(keys, values, "IFC_shape_name", 0)
            _add_arg_alias(keys, values, "IFC_shape_description", 1)
            _add_arg_alias(keys, values, "IFC_representations", 2)

        if ifc_type == "IFCSHAPEREPRESENTATION":
            _add_arg_alias(keys, values, "IFC_context_of_items", 0)
            _add_arg_alias(keys, values, "IFC_representation_identifier", 1)
            _add_arg_alias(keys, values, "IFC_representation_type", 2)
            _add_arg_alias(keys, values, "IFC_items", 3)

        # Merge relationship-derived psets, type psets, qtos, materials, classifications.
        if metadataCache is not None:
            try:
                extra = metadataCache.get(entity.id, {})
            except Exception:
                extra = {}
            if isinstance(extra, dict):
                for key, value in extra.items():
                    _add(keys, values, key, value)

        # Keep the compact raw argument string for debugging only. Retrieval should use IFC_arg_N or aliases.
        try:
            _add(keys, values, "IFC_raw_args", str(entity.args))
        except Exception:
            pass

        try:
            return Dictionary.ByKeysValues(keys, values)
        except Exception:
            d = None
            for k, v in zip(keys, values):
                try:
                    if d is None:
                        d = Dictionary.ByKeyValue(k, v)
                    else:
                        d = Dictionary.SetValueAtKey(d, k, v)
                except Exception:
                    pass
            return d

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
    def Entities(file, silent: bool = False):
        """
        Returns the parsed IFC STEP entity dictionary from the input IFC file.

        This method accepts:
        - a previously parsed IFC entity dictionary,
        - a path to an IFC file,
        - a STEP text string,
        - or an ifcopenshell file-like object that can be serialised to STEP text.

        Parameters
        ----------
        file : dict, str, or ifcopenshell.file-like object
            The input IFC source.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        dict
            A dictionary of parsed IFC entities keyed by STEP id. Each value is an
            IFCFastEntity object. Returns None if parsing fails.
        """

        entities = IFC._entities_from_input(file, silent=silent)

        if entities is None:
            if not silent:
                print("IFC.Entities - Error: Could not read or parse the input IFC file. Returning None.")
            return None

        if not isinstance(entities, dict):
            if not silent:
                print("IFC.Entities - Error: Parsed IFC entities are not stored in a valid dictionary. Returning None.")
            return None

        return entities
    
    @staticmethod
    def EntitiesByPath(path: str, silent: bool = False):
        """
        Returns the parsed IFC STEP entity dictionary from the IFC file at the input path.

        Parameters
        ----------
        path : str
            The input IFC file path.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        dict
            A dictionary of parsed IFC entities keyed by STEP id. Each value is an
            IFCFastEntity object. Returns None if parsing fails.
        """

        return IFC.Entities(path, silent=silent)


    @staticmethod
    def Object(file,
               globalId: str = None,
               ifcId: int = None,
               ifcKey: str = None,
               silent: bool = False):
        """
        Returns a dynamic IFCEntity wrapper for one IFC entity.

        Parameters
        ----------
        file : dict, str, or ifcopenshell.file-like object
            The input IFC source.
        globalId : str , optional
            The GlobalId of the IFC entity to retrieve.
        ifcId : int , optional
            The STEP id of the IFC entity to retrieve.
        ifcKey : str , optional
            The STEP key of the IFC entity to retrieve, e.g. "#123".
        silent : bool , optional
            If True, suppresses error and warning messages. Default is False.

        Returns
        -------
        IFCEntity
            A dynamic IFC entity wrapper. Returns None if the entity is not found.
        """

        try:
            entities = IFC.Entities(file, silent=silent)
        except Exception:
            entities = IFC._entities_from_input(file, silent=silent)

        if entities is None or not isinstance(entities, dict):
            if not silent:
                print("IFC.Object - Error: Could not read or parse the input IFC file. Returning None.")
            return None

        target = None

        if ifcId is not None:
            try:
                target = entities.get(int(ifcId))
            except Exception:
                target = None

        if target is None and ifcKey is not None:
            try:
                key = str(ifcKey).strip()
                if key.startswith("#"):
                    key = key[1:]
                target = entities.get(int(key))
            except Exception:
                target = None

        if target is None and isinstance(globalId, str) and len(globalId.strip()) > 0:
            gid = globalId.strip()
            for entity in entities.values():
                if IFCFastTopology._root_attr(entity, 0) == gid:
                    target = entity
                    break

        if target is None:
            if not silent:
                print("IFC.Object - Error: Could not find the requested IFC entity. Returning None.")
            return None

        props = IFC.Properties(file, includeEmpty=True, silent=silent)
        prop_record = {}

        if isinstance(props, dict):
            records = props.get("entities", {})
            gid = IFCFastTopology._root_attr(target, 0)
            possible_keys = [
                gid,
                str(target.id),
                f"#{target.id}",
            ]

            for key in possible_keys:
                if key in records:
                    prop_record = records[key]
                    break

        return IFCEntity(target, entities=entities, properties=prop_record)

    @staticmethod
    def Objects(file,
                includeTypes: list = None,
                excludeTypes: list = None,
                includeEmpty: bool = True,
                silent: bool = False):
        """
        Returns dynamic IFCEntity wrappers for IFC entities in a file.

        Parameters
        ----------
        file : dict, str, or ifcopenshell.file-like object
            The input IFC source.
        includeTypes : list , optional
            IFC entity types to include. Default is None.
        excludeTypes : list , optional
            IFC entity types to exclude. Default is None.
        includeEmpty : bool , optional
            If True, include entities with no extracted property payload.
            Default is True.
        silent : bool , optional
            If True, suppresses error and warning messages. Default is False.

        Returns
        -------
        list
            A list of IFCEntity objects.
        """

        def _normalise_type(value):
            if value is None:
                return None
            value = str(value).strip().upper()
            return value if value else None

        def _normalise_type_set(values):
            if not values:
                return set()
            return set([_normalise_type(v) for v in values if _normalise_type(v)])

        try:
            entities = IFC.Entities(file, silent=silent)
        except Exception:
            entities = IFC._entities_from_input(file, silent=silent)

        if entities is None or not isinstance(entities, dict):
            if not silent:
                print("IFC.Objects - Error: Could not read or parse the input IFC file. Returning None.")
            return None

        include_set = _normalise_type_set(includeTypes)
        exclude_set = _normalise_type_set(excludeTypes)

        props = IFC.Properties(
            file,
            includeTypes=includeTypes,
            excludeTypes=excludeTypes,
            includeEmpty=includeEmpty,
            silent=silent,
        )

        records = {}
        if isinstance(props, dict):
            records = props.get("entities", {})

        objects = []

        for entity in sorted(entities.values(), key=lambda e: e.id):
            etype = _normalise_type(entity.type)

            if include_set and etype not in include_set:
                continue

            if exclude_set and etype in exclude_set:
                continue

            if not include_set:
                try:
                    if not IFCFastTopology._is_product_like(entity):
                        continue
                except Exception:
                    if etype is None or etype.startswith("IFCREL"):
                        continue

            gid = IFCFastTopology._root_attr(entity, 0)
            possible_keys = [
                gid,
                str(entity.id),
                f"#{entity.id}",
            ]

            record = {}
            for key in possible_keys:
                if key in records:
                    record = records[key]
                    break

            if not includeEmpty and not record:
                continue

            objects.append(IFCEntity(entity, entities=entities, properties=record))

        return objects

    @staticmethod
    def ObjectsByType(file,
                      ifcType: str,
                      includeEmpty: bool = True,
                      silent: bool = False):
        """
        Returns dynamic IFCEntity wrappers for entities of one IFC type.

        Example
        -------
        walls = IFC.ObjectsByType(file, "IfcWall")
        print(walls[0].FireRating)
        """

        if not isinstance(ifcType, str) or len(ifcType.strip()) < 1:
            if not silent:
                print("IFC.ObjectsByType - Error: The ifcType parameter is not a valid string. Returning None.")
            return None

        t = ifcType.strip().upper()
        if not t.startswith("IFC"):
            t = "IFC" + t.upper()

        return IFC.Objects(
            file,
            includeTypes=[t],
            includeEmpty=includeEmpty,
            silent=silent,
        )

    @staticmethod
    def TopologiesByEntities(entities,
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
                            silent: bool = False):
        """
        Imports IFC product geometry from an already parsed IFC entity dictionary.

        This avoids reparsing the IFC file when the same parsed entities are also
        used for graph extraction, metadata queries, or other IFC workflows.
        """

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
    def Triples(file,
                includeRels: list = None,
                excludeRels: list = None,
                subjectKey: str = "global_id",
                objectKey: str = "global_id",
                includeMetadata: bool = True,
                silent: bool = False):
        """
        Returns IFC relationship triples from the input IFC file.

        Each IfcRel* entity is converted into one or more explicit triples in
        the form:

        {
            "subject": ...,
            "predicate": ...,
            "object": ...
        }

        If includeMetadata is True, each triple also includes IFC ids, STEP keys,
        entity types, entity names, and relationship metadata.

        Parameters
        ----------
        file : dict, str, or ifcopenshell.file-like object
            The input IFC source.
        includeRels : list , optional
            A list of IFC relationship types to include. If None, all supported
            relationship types are included. Default is None.
        excludeRels : list , optional
            A list of IFC relationship types to exclude. Default is None.
        subjectKey : str , optional
            Identifier to use for the subject. Options are "global_id", "id",
            or "key". Default is "global_id".
        objectKey : str , optional
            Identifier to use for the object. Options are "global_id", "id",
            or "key". Default is "global_id".
        includeMetadata : bool , optional
            If set to True, additional metadata is included in each triple.
            Default is True.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        list
            A list of dictionaries representing IFC relationship triples.
        """

        def _normalise_type(value):
            if value is None:
                return None
            value = str(value).strip()
            if not value:
                return None
            return value.upper()

        def _normalise_type_set(values):
            if not values:
                return set()
            return set([_normalise_type(v) for v in values if _normalise_type(v)])

        def _display_type(ifc_type):
            if ifc_type is None:
                return None
            ifc_type = str(ifc_type).strip()
            if ifc_type.upper().startswith("IFC"):
                return "Ifc" + ifc_type[3:].lower().title().replace("_", "")
            return ifc_type

        def _entity_id(entity, key="global_id"):
            if entity is None:
                return None

            key = (key or "global_id").strip().lower()

            if key in ["id", "ifc_id", "step_id"]:
                return entity.id

            if key in ["key", "ifc_key", "step_key"]:
                return f"#{entity.id}"

            # Default: use GlobalId where possible, otherwise STEP key.
            gid = IFCFastTopology._root_attr(entity, 0)
            if gid not in [None, "", "*"]:
                return gid

            return f"#{entity.id}"

        def _entity_name(entity):
            try:
                return IFCFastTopology._root_attr(entity, 2)
            except Exception:
                return None

        def _refs(value):
            try:
                return IFCFastTopology._refs_in_value(value)
            except Exception:
                return []

        def _entity_from_ref(ref):
            try:
                return IFCFastTopology._entity_from_ref(ref, entities)
            except Exception:
                return None

        def _make_triple(subject_entity, predicate, object_entity, rel_entity):
            if subject_entity is None or object_entity is None:
                return None

            triple = {
                "subject": _entity_id(subject_entity, subjectKey),
                "predicate": predicate,
                "object": _entity_id(object_entity, objectKey),
            }

            if includeMetadata:
                triple.update({
                    "subject_id": subject_entity.id,
                    "subject_key": f"#{subject_entity.id}",
                    "subject_type": _display_type(subject_entity.type),
                    "subject_name": _entity_name(subject_entity),

                    "object_id": object_entity.id,
                    "object_key": f"#{object_entity.id}",
                    "object_type": _display_type(object_entity.type),
                    "object_name": _entity_name(object_entity),

                    "relationship_id": rel_entity.id,
                    "relationship_key": f"#{rel_entity.id}",
                    "relationship_type": _display_type(rel_entity.type),
                    "relationship_global_id": IFCFastTopology._root_attr(rel_entity, 0),
                    "relationship_name": IFCFastTopology._root_attr(rel_entity, 2),
                })

            return triple

        # --------------------------------------------------------------
        # Parse once
        # --------------------------------------------------------------

        try:
            entities = IFC.Entities(file, silent=silent)
        except Exception:
            entities = IFC._entities_from_input(file, silent=silent)

        if entities is None or not isinstance(entities, dict):
            if not silent:
                print("IFC.Triples - Error: Could not read or parse the input IFC file. Returning None.")
            return None

        include_rel_set = _normalise_type_set(includeRels)
        exclude_rel_set = _normalise_type_set(excludeRels)

        # --------------------------------------------------------------
        # Relationship mapping
        #
        # Each entry maps:
        # relationship type -> (subject argument index, object argument index)
        #
        # Lists are automatically expanded, so one IfcRel can produce many
        # triples.
        # --------------------------------------------------------------

        relationship_map = {
            "IFCRELAGGREGATES": (4, 5),                         # RelatingObject -> RelatedObjects
            "IFCRELNESTS": (4, 5),                              # RelatingObject -> RelatedObjects
            "IFCRELCONTAINEDINSPATIALSTRUCTURE": (5, 4),        # RelatingStructure -> RelatedElements
            "IFCRELASSIGNSTOGROUP": (5, 4),                     # RelatingGroup -> RelatedObjects

            "IFCRELDEFINESBYPROPERTIES": (5, 4),                # RelatingPropertyDefinition -> RelatedObjects
            "IFCRELDEFINESBYTYPE": (5, 4),                      # RelatingType -> RelatedObjects

            "IFCRELASSOCIATESMATERIAL": (5, 4),                 # RelatingMaterial -> RelatedObjects
            "IFCRELASSOCIATESCLASSIFICATION": (5, 4),           # RelatingClassification -> RelatedObjects
            "IFCRELASSOCIATESDOCUMENT": (5, 4),                 # RelatingDocument -> RelatedObjects
            "IFCRELASSOCIATESAPPROVAL": (5, 4),                 # RelatingApproval -> RelatedObjects
            "IFCRELASSOCIATESCONSTRAINT": (5, 4),               # RelatingConstraint -> RelatedObjects

            "IFCRELVOIDSELEMENT": (5, 4),                       # RelatedOpeningElement -> RelatingBuildingElement
            "IFCRELFILLSELEMENT": (5, 4),                       # RelatedBuildingElement -> RelatingOpeningElement

            "IFCRELSPACEBOUNDARY": (4, 5),                      # RelatingSpace -> RelatedBuildingElement
            "IFCRELSPACEBOUNDARY1STLEVEL": (4, 5),
            "IFCRELSPACEBOUNDARY2NDLEVEL": (4, 5),

            "IFCRELCONNECTSELEMENTS": (4, 5),                   # RelatingElement -> RelatedElement
            "IFCRELCONNECTSPATH ELEMENTS": (4, 5),
            "IFCRELCONNECTSPORTTOELEMENT": (4, 5),              # RelatingPort -> RelatedElement
            "IFCRELCONNECTSPORTS": (4, 5),                      # RelatingPort -> RelatedPort
            "IFCRELCONNECTSSTRUCTURALMEMBER": (4, 5),           # RelatingStructuralMember -> RelatedStructuralConnection
        }

        # Typo-safe duplicate for correct IFC spelling.
        relationship_map["IFCRELCONNECTSPATHELEMENTS"] = (4, 5)

        triples = []

        for rel in entities.values():
            rel_type = _normalise_type(rel.type)

            if rel_type is None or not rel_type.startswith("IFCREL"):
                continue

            if include_rel_set and rel_type not in include_rel_set:
                continue

            if exclude_rel_set and rel_type in exclude_rel_set:
                continue

            predicate = _display_type(rel_type)

            # ----------------------------------------------------------
            # Preferred path: known IFC relationship signatures.
            # ----------------------------------------------------------

            if rel_type in relationship_map:
                subject_index, object_index = relationship_map[rel_type]

                if subject_index >= len(rel.args) or object_index >= len(rel.args):
                    continue

                subject_refs = _refs(rel.args[subject_index])
                object_refs = _refs(rel.args[object_index])

                for subject_ref in subject_refs:
                    subject_entity = _entity_from_ref(subject_ref)

                    for object_ref in object_refs:
                        object_entity = _entity_from_ref(object_ref)
                        triple = _make_triple(subject_entity, predicate, object_entity, rel)

                        if triple is not None:
                            triples.append(triple)

                continue

            # ----------------------------------------------------------
            # Fallback path:
            # For unsupported IfcRel* entities, inspect arguments after
            # the Root attributes and connect the first referenced entity
            # to all subsequent referenced entities.
            # ----------------------------------------------------------

            rel_refs = []
            for arg in rel.args[4:]:
                rel_refs.extend(_refs(arg))

            if len(rel_refs) < 2:
                continue

            subject_entity = _entity_from_ref(rel_refs[0])

            for object_ref in rel_refs[1:]:
                object_entity = _entity_from_ref(object_ref)
                triple = _make_triple(subject_entity, predicate, object_entity, rel)

                if triple is not None:
                    triples.append(triple)

        return triples

    @staticmethod
    def Properties(file,
                   includeTypes: list = None,
                   excludeTypes: list = None,
                   keyMode: str = "global_id",
                   includeEmpty: bool = False,
                   silent: bool = False):
        """
        Extracts semantic properties, quantities, classifications, and materials
        from an IFC file.

        Parameters
        ----------
        file : dict, str, or ifcopenshell.file-like object
            The input IFC source.
        includeTypes : list , optional
            IFC entity types to include. If None, all product-like entities are
            included. Default is None.
        excludeTypes : list , optional
            IFC entity types to exclude. Default is None.
        keyMode : str , optional
            Entity key mode. Options are "global_id", "id", or "key".
            Default is "global_id".
        includeEmpty : bool , optional
            If True, entities with no extracted semantic payload are included.
            Default is False.
        silent : bool , optional
            If True, suppresses error and warning messages. Default is False.

        Returns
        -------
        dict
            A dictionary with two main keys:

            {
                "entities": {
                    "<entity_key>": {
                        "ifc_id": int,
                        "ifc_key": "#123",
                        "ifc_type": "IfcWall",
                        "global_id": "...",
                        "name": "...",
                        "properties": {...},
                        "type_properties": {...},
                        "quantities": {...},
                        "classifications": [...],
                        "materials": [...],
                        "type": {...}
                    }
                },
                "summary": {...}
            }
        """

        def _normalise_type(value):
            if value is None:
                return None
            value = str(value).strip().upper()
            return value if value else None

        def _normalise_type_set(values):
            if not values:
                return set()
            return set([_normalise_type(v) for v in values if _normalise_type(v)])

        def _display_type(ifc_type):
            if ifc_type is None:
                return None
            ifc_type = str(ifc_type).strip()
            if ifc_type.upper().startswith("IFC"):
                return "Ifc" + ifc_type[3:].lower().title().replace("_", "")
            return ifc_type

        def _root_attr(entity, index):
            try:
                return IFCFastTopology._root_attr(entity, index)
            except Exception:
                return None

        def _refs(value):
            try:
                return IFCFastTopology._refs_in_value(value)
            except Exception:
                return []

        def _entity_from_ref(ref):
            try:
                return IFCFastTopology._entity_from_ref(ref, entities)
            except Exception:
                return None

        def _entity_key(entity):
            if entity is None:
                return None

            mode = (keyMode or "global_id").strip().lower()

            if mode in ["id", "ifc_id", "step_id"]:
                return str(entity.id)

            if mode in ["key", "ifc_key", "step_key"]:
                return f"#{entity.id}"

            gid = _root_attr(entity, 0)
            if gid not in [None, "", "*"]:
                return str(gid)

            return f"#{entity.id}"

        def _entity_name(entity):
            value = _root_attr(entity, 2)
            return value if value not in [None, "", "*"] else None

        def _unwrap(value):
            """
            Converts STEP parser values into practical Python scalar/list forms.
            Handles IFC measure wrapper calls such as IFCLABEL('A') or
            IFCAREAMEASURE(12.3).
            """
            if value in [None, "*", "$"]:
                return None

            if isinstance(value, tuple):
                if len(value) == 3 and value[0] == "CALL":
                    args = value[2] or []
                    if len(args) == 1:
                        return _unwrap(args[0])
                    return [_unwrap(v) for v in args]
                if len(value) == 2 and value[0] == "REF":
                    ref_entity = _entity_from_ref(value)
                    if ref_entity is None:
                        return f"#{value[1]}"
                    return _entity_key(ref_entity)

            if isinstance(value, list):
                return [_unwrap(v) for v in value]

            return value

        def _property_name(entity):
            if entity is None or len(entity.args) < 1:
                return None
            name = entity.args[0]
            return str(name) if name not in [None, "", "*", "$"] else None

        def _property_value(prop):
            """
            Extracts values from common IfcProperty* entities.
            """
            if prop is None:
                return None

            ptype = _normalise_type(prop.type)

            # IfcPropertySingleValue(Name, Description, NominalValue, Unit)
            if ptype == "IFCPROPERTYSINGLEVALUE":
                return _unwrap(prop.args[2]) if len(prop.args) > 2 else None

            # IfcPropertyEnumeratedValue(Name, Description, EnumerationValues, EnumerationReference)
            if ptype == "IFCPROPERTYENUMERATEDVALUE":
                return _unwrap(prop.args[2]) if len(prop.args) > 2 else None

            # IfcPropertyListValue(Name, Description, ListValues, Unit)
            if ptype == "IFCPROPERTYLISTVALUE":
                return _unwrap(prop.args[2]) if len(prop.args) > 2 else None

            # IfcPropertyBoundedValue(Name, Description, UpperBoundValue, LowerBoundValue, SetPointValue, Unit)
            if ptype == "IFCPROPERTYBOUNDEDVALUE":
                return {
                    "upper": _unwrap(prop.args[2]) if len(prop.args) > 2 else None,
                    "lower": _unwrap(prop.args[3]) if len(prop.args) > 3 else None,
                    "set_point": _unwrap(prop.args[4]) if len(prop.args) > 4 else None,
                }

            # IfcPropertyTableValue(Name, Description, DefiningValues, DefinedValues, ...)
            if ptype == "IFCPROPERTYTABLEVALUE":
                return {
                    "defining": _unwrap(prop.args[2]) if len(prop.args) > 2 else None,
                    "defined": _unwrap(prop.args[3]) if len(prop.args) > 3 else None,
                }

            # IfcComplexProperty(Name, Description, UsageName, HasProperties)
            if ptype == "IFCCOMPLEXPROPERTY":
                nested = {}
                if len(prop.args) > 3:
                    for ref in _refs(prop.args[3]):
                        child = _entity_from_ref(ref)
                        child_name = _property_name(child)
                        if child_name:
                            nested[child_name] = _property_value(child)
                return nested

            return _unwrap(prop.args)

        def _quantity_name(entity):
            if entity is None or len(entity.args) < 1:
                return None
            name = entity.args[0]
            return str(name) if name not in [None, "", "*", "$"] else None

        def _quantity_value(quantity):
            """
            Extracts the numerical value from common IfcPhysicalSimpleQuantity
            subtypes. IFC simple quantities normally store the value at args[3].
            """
            if quantity is None:
                return None

            qtype = _normalise_type(quantity.type)

            if qtype in [
                "IFCQUANTITYLENGTH",
                "IFCQUANTITYAREA",
                "IFCQUANTITYVOLUME",
                "IFCQUANTITYCOUNT",
                "IFCQUANTITYWEIGHT",
                "IFCQUANTITYTIME",
            ]:
                return _unwrap(quantity.args[3]) if len(quantity.args) > 3 else None

            # IfcQuantityNumber is present in some IFC versions / exports.
            if qtype == "IFCQUANTITYNUMBER":
                return _unwrap(quantity.args[3]) if len(quantity.args) > 3 else None

            return _unwrap(quantity.args)

        def _merge_named_value(target, group_name, name, value):
            if group_name not in target:
                target[group_name] = {}
            target[group_name][name] = value

        def _classification_info(classification_entity):
            if classification_entity is None:
                return None

            ctype = _normalise_type(classification_entity.type)

            data = {
                "ifc_id": classification_entity.id,
                "ifc_key": f"#{classification_entity.id}",
                "ifc_type": _display_type(classification_entity.type),
            }

            # IfcClassificationReference(Location, Identification/ItemReference, Name, ReferencedSource, ...)
            if ctype == "IFCCLASSIFICATIONREFERENCE":
                data.update({
                    "location": _unwrap(classification_entity.args[0]) if len(classification_entity.args) > 0 else None,
                    "identification": _unwrap(classification_entity.args[1]) if len(classification_entity.args) > 1 else None,
                    "name": _unwrap(classification_entity.args[2]) if len(classification_entity.args) > 2 else None,
                })
                if len(classification_entity.args) > 3:
                    source = _entity_from_ref(classification_entity.args[3])
                    if source is not None:
                        data["source"] = _classification_info(source)
                return data

            # IfcClassification(Source, Edition, EditionDate, Name, ...)
            if ctype == "IFCCLASSIFICATION":
                data.update({
                    "source": _unwrap(classification_entity.args[0]) if len(classification_entity.args) > 0 else None,
                    "edition": _unwrap(classification_entity.args[1]) if len(classification_entity.args) > 1 else None,
                    "name": _unwrap(classification_entity.args[3]) if len(classification_entity.args) > 3 else None,
                })
                return data

            data["value"] = _unwrap(classification_entity.args)
            return data

        def _material_info(material_entity):
            if material_entity is None:
                return None

            mtype = _normalise_type(material_entity.type)

            data = {
                "ifc_id": material_entity.id,
                "ifc_key": f"#{material_entity.id}",
                "ifc_type": _display_type(material_entity.type),
            }

            if mtype == "IFCMATERIAL":
                data.update({
                    "name": _unwrap(material_entity.args[0]) if len(material_entity.args) > 0 else None,
                    "description": _unwrap(material_entity.args[1]) if len(material_entity.args) > 1 else None,
                    "category": _unwrap(material_entity.args[2]) if len(material_entity.args) > 2 else None,
                })
                return data

            if mtype == "IFCMATERIALLAYER":
                mat = _entity_from_ref(material_entity.args[0]) if len(material_entity.args) > 0 else None
                data.update({
                    "material": _material_info(mat),
                    "thickness": _unwrap(material_entity.args[1]) if len(material_entity.args) > 1 else None,
                    "is_ventilated": _unwrap(material_entity.args[2]) if len(material_entity.args) > 2 else None,
                    "name": _unwrap(material_entity.args[3]) if len(material_entity.args) > 3 else None,
                })
                return data

            if mtype == "IFCMATERIALLAYERSET":
                layers = []
                if len(material_entity.args) > 0:
                    for ref in _refs(material_entity.args[0]):
                        layer = _entity_from_ref(ref)
                        if layer is not None:
                            layers.append(_material_info(layer))
                data.update({
                    "layers": layers,
                    "name": _unwrap(material_entity.args[1]) if len(material_entity.args) > 1 else None,
                    "description": _unwrap(material_entity.args[2]) if len(material_entity.args) > 2 else None,
                })
                return data

            if mtype == "IFCMATERIALLAYERSETUSAGE":
                layer_set = _entity_from_ref(material_entity.args[0]) if len(material_entity.args) > 0 else None
                data.update({
                    "layer_set": _material_info(layer_set),
                    "layer_set_direction": _unwrap(material_entity.args[1]) if len(material_entity.args) > 1 else None,
                    "direction_sense": _unwrap(material_entity.args[2]) if len(material_entity.args) > 2 else None,
                    "offset_from_reference_line": _unwrap(material_entity.args[3]) if len(material_entity.args) > 3 else None,
                })
                return data

            if mtype == "IFCMATERIALLIST":
                mats = []
                if len(material_entity.args) > 0:
                    for ref in _refs(material_entity.args[0]):
                        mat = _entity_from_ref(ref)
                        if mat is not None:
                            mats.append(_material_info(mat))
                data["materials"] = mats
                return data

            data["value"] = _unwrap(material_entity.args)
            return data

        def _is_product_like(entity):
            try:
                return IFCFastTopology._is_product_like(entity)
            except Exception:
                if entity is None:
                    return False
                t = _normalise_type(entity.type)
                return t is not None and t.startswith("IFC") and not t.startswith("IFCREL")

        def _empty_record(entity):
            return {
                "ifc_id": entity.id,
                "ifc_key": f"#{entity.id}",
                "ifc_type": _display_type(entity.type),
                "global_id": _root_attr(entity, 0),
                "name": _entity_name(entity),
                "properties": {},
                "type_properties": {},
                "quantities": {},
                "type_quantities": {},
                "classifications": [],
                "materials": [],
                "type": None,
            }

        # --------------------------------------------------------------
        # Parse IFC once.
        # --------------------------------------------------------------

        try:
            entities = IFC.Entities(file, silent=silent)
        except Exception:
            try:
                entities = IFC._entities_from_input(file, silent=silent)
            except Exception:
                entities = None

        if entities is None or not isinstance(entities, dict):
            if not silent:
                print("IFC.Properties - Error: Could not read or parse the input IFC file. Returning None.")
            return None

        include_type_set = _normalise_type_set(includeTypes)
        exclude_type_set = _normalise_type_set(excludeTypes)

        records = {}

        def _ensure_entity_record(entity):
            if entity is None:
                return None

            etype = _normalise_type(entity.type)

            if include_type_set and etype not in include_type_set:
                return None
            if exclude_type_set and etype in exclude_type_set:
                return None

            key = _entity_key(entity)
            if key is None:
                return None

            if key not in records:
                records[key] = _empty_record(entity)

            return records[key]

        # Initialise records for product-like entities.
        for entity in entities.values():
            if _is_product_like(entity):
                _ensure_entity_record(entity)

        # --------------------------------------------------------------
        # Extract IfcPropertySet and IfcElementQuantity content.
        # --------------------------------------------------------------

        property_sets = {}
        quantity_sets = {}

        for entity in entities.values():
            etype = _normalise_type(entity.type)

            # IfcPropertySet(GlobalId, OwnerHistory, Name, Description, HasProperties)
            if etype == "IFCPROPERTYSET":
                pset_name = _root_attr(entity, 2) or f"IfcPropertySet_{entity.id}"
                values = {}

                if len(entity.args) > 4:
                    for prop_ref in _refs(entity.args[4]):
                        prop = _entity_from_ref(prop_ref)
                        pname = _property_name(prop)

                        if pname:
                            values[pname] = _property_value(prop)

                property_sets[entity.id] = {
                    "name": pset_name,
                    "values": values,
                    "ifc_id": entity.id,
                    "ifc_key": f"#{entity.id}",
                }

            # IfcElementQuantity(GlobalId, OwnerHistory, Name, Description, MethodOfMeasurement, Quantities)
            elif etype == "IFCELEMENTQUANTITY":
                qset_name = _root_attr(entity, 2) or f"IfcElementQuantity_{entity.id}"
                values = {}

                if len(entity.args) > 5:
                    for q_ref in _refs(entity.args[5]):
                        q = _entity_from_ref(q_ref)
                        qname = _quantity_name(q)

                        if qname:
                            values[qname] = _quantity_value(q)

                quantity_sets[entity.id] = {
                    "name": qset_name,
                    "values": values,
                    "ifc_id": entity.id,
                    "ifc_key": f"#{entity.id}",
                }

        # --------------------------------------------------------------
        # IfcRelDefinesByProperties:
        #   args[4] = RelatedObjects
        #   args[5] = RelatingPropertyDefinition
        # --------------------------------------------------------------

        for rel in entities.values():
            if _normalise_type(rel.type) != "IFCRELDEFINESBYPROPERTIES":
                continue
            if len(rel.args) <= 5:
                continue

            related_refs = _refs(rel.args[4])
            definition_refs = _refs(rel.args[5])

            for definition_ref in definition_refs:
                definition = _entity_from_ref(definition_ref)
                if definition is None:
                    continue

                pset = property_sets.get(definition.id)
                qset = quantity_sets.get(definition.id)

                for related_ref in related_refs:
                    related = _entity_from_ref(related_ref)
                    record = _ensure_entity_record(related)

                    if record is None:
                        continue

                    if pset is not None:
                        record["properties"][pset["name"]] = dict(pset["values"])

                    if qset is not None:
                        record["quantities"][qset["name"]] = dict(qset["values"])

        # --------------------------------------------------------------
        # IfcRelDefinesByType:
        #   args[4] = RelatedObjects
        #   args[5] = RelatingType
        #
        # Also collect psets/qsets attached to type objects directly through
        # HasPropertySets, usually args[5] on IfcTypeObject derivatives.
        # --------------------------------------------------------------

        type_payload = {}

        def _type_payload(type_entity):
            if type_entity is None:
                return None

            key = _entity_key(type_entity)
            if key in type_payload:
                return type_payload[key]

            payload = {
                "ifc_id": type_entity.id,
                "ifc_key": f"#{type_entity.id}",
                "ifc_type": _display_type(type_entity.type),
                "global_id": _root_attr(type_entity, 0),
                "name": _entity_name(type_entity),
                "properties": {},
                "quantities": {},
            }

            # IFC TypeObject: HasPropertySets is commonly at args[5].
            possible_slots = [5, 6]
            for slot in possible_slots:
                if len(type_entity.args) <= slot:
                    continue
                for ref in _refs(type_entity.args[slot]):
                    definition = _entity_from_ref(ref)
                    if definition is None:
                        continue

                    pset = property_sets.get(definition.id)
                    qset = quantity_sets.get(definition.id)

                    if pset is not None:
                        payload["properties"][pset["name"]] = dict(pset["values"])

                    if qset is not None:
                        payload["quantities"][qset["name"]] = dict(qset["values"])

            type_payload[key] = payload
            return payload

        for rel in entities.values():
            if _normalise_type(rel.type) != "IFCRELDEFINESBYTYPE":
                continue
            if len(rel.args) <= 5:
                continue

            related_refs = _refs(rel.args[4])
            type_refs = _refs(rel.args[5])

            for type_ref in type_refs:
                type_entity = _entity_from_ref(type_ref)
                payload = _type_payload(type_entity)

                if payload is None:
                    continue

                for related_ref in related_refs:
                    related = _entity_from_ref(related_ref)
                    record = _ensure_entity_record(related)

                    if record is None:
                        continue

                    record["type"] = {
                        "ifc_id": payload["ifc_id"],
                        "ifc_key": payload["ifc_key"],
                        "ifc_type": payload["ifc_type"],
                        "global_id": payload["global_id"],
                        "name": payload["name"],
                    }

                    for pset_name, pset_values in payload["properties"].items():
                        record["type_properties"][pset_name] = dict(pset_values)

                    for qset_name, qset_values in payload["quantities"].items():
                        record["type_quantities"][qset_name] = dict(qset_values)

        # --------------------------------------------------------------
        # IfcRelAssociatesClassification:
        #   args[4] = RelatedObjects
        #   args[5] = RelatingClassification
        # --------------------------------------------------------------

        for rel in entities.values():
            if _normalise_type(rel.type) != "IFCRELASSOCIATESCLASSIFICATION":
                continue
            if len(rel.args) <= 5:
                continue

            classification_refs = _refs(rel.args[5])

            for classification_ref in classification_refs:
                classification = _entity_from_ref(classification_ref)
                info = _classification_info(classification)

                if info is None:
                    continue

                for related_ref in _refs(rel.args[4]):
                    related = _entity_from_ref(related_ref)
                    record = _ensure_entity_record(related)

                    if record is not None:
                        record["classifications"].append(info)

        # --------------------------------------------------------------
        # IfcRelAssociatesMaterial:
        #   args[4] = RelatedObjects
        #   args[5] = RelatingMaterial
        # --------------------------------------------------------------

        for rel in entities.values():
            if _normalise_type(rel.type) != "IFCRELASSOCIATESMATERIAL":
                continue
            if len(rel.args) <= 5:
                continue

            material_refs = _refs(rel.args[5])

            for material_ref in material_refs:
                material = _entity_from_ref(material_ref)
                info = _material_info(material)

                if info is None:
                    continue

                for related_ref in _refs(rel.args[4]):
                    related = _entity_from_ref(related_ref)
                    record = _ensure_entity_record(related)

                    if record is not None:
                        record["materials"].append(info)

        # --------------------------------------------------------------
        # Remove empty records unless requested.
        # --------------------------------------------------------------

        if not includeEmpty:
            filtered = {}

            for key, record in records.items():
                has_payload = bool(
                    record["properties"] or
                    record["type_properties"] or
                    record["quantities"] or
                    record["type_quantities"] or
                    record["classifications"] or
                    record["materials"] or
                    record["type"]
                )

                if has_payload:
                    filtered[key] = record

            records = filtered

        summary = {
            "entity_count": len(records),
            "property_entity_count": sum(1 for r in records.values() if r["properties"]),
            "type_property_entity_count": sum(1 for r in records.values() if r["type_properties"]),
            "quantity_entity_count": sum(1 for r in records.values() if r["quantities"]),
            "type_quantity_entity_count": sum(1 for r in records.values() if r["type_quantities"]),
            "classification_entity_count": sum(1 for r in records.values() if r["classifications"]),
            "material_entity_count": sum(1 for r in records.values() if r["materials"]),
            "property_set_count": len(property_sets),
            "quantity_set_count": len(quantity_sets),
        }

        return {
            "entities": records,
            "summary": summary,
        }

    @staticmethod
    def AccessGraph(file,
                            importMode: str = "topology",
                            useInternalVertex: bool = False,
                            connectingElementTypes: list = None,
                            useFillingElements: bool = True,
                            includeIsolatedSpaces: bool = True,
                            viaConnectingElements: bool = False,
                            dictionaryMode: str = "basic",
                            bidirectional: bool = True,
                            tolerance: float = 0.0001,
                            silent: bool = False):
        """
        Creates a TopologicPy Graph representing space adjacency from an IFC file.

        The returned graph contains one vertex per IfcSpace. Edges are created
        between spaces that share a boundary-related connector element.

        IMPORTANT
        ---------
        Door/opening/window-like connector elements are local connectors and can
        safely create adjacency by grouping the spaces that reference the same
        connector. Wall-like connector elements are not local connectors. A long
        wall may be referenced by several space boundaries, so this method does
        not create all pairwise adjacencies among spaces that reference the same
        wall. For IfcWall, IfcWallStandardCase, and IfcWallElementedCase, adjacency
        is created only from IfcRelSpaceBoundary2ndLevel.CorrespondingBoundary.

        Parameters
        ----------
        file : dict, str, or ifcopenshell.file-like object
            The input IFC source.
        importMode : str , optional
            Controls graph vertex placement. If set to "topology", vertices are
            placed using a deterministic circular graph layout. If set to
            "geometry", vertices are placed at the centre of their parent IFC
            product representation when such representation data can be evaluated.
            Objects without usable geometry fall back to deterministic placement.
            Options are "topology" and "geometry". Default is "topology".
        useInternalVertex : bool , optional
            In geometry mode, if set to True, uses an internal vertex to represent
            the subtopology. Otherwise, uses its centroid. Default is False.
        connectingElementTypes : list , optional
            IFC element types that are allowed to act as adjacency connectors.
            If set to None, the default is ["IFCDOOR", "IFCDOORSTANDARDCASE",
            "IFCOPENINGELEMENT"]. Wall types are supported, but they are handled
            through IfcRelSpaceBoundary2ndLevel.CorrespondingBoundary rather than
            simple shared-wall grouping. Default is None.
        useFillingElements : bool , optional
            If set to True, IfcOpeningElement references are resolved through
            IfcRelFillsElement to their filling elements, usually doors/windows.
            The opening is still retained and preferred as the via-vertex entity
            when viaConnectingElements is True. Default is True.
        includeIsolatedSpaces : bool , optional
            If set to True, all IfcSpace entities are included as vertices even
            when they have no detected adjacency. If False, only spaces involved
            in at least one adjacency edge are included. Default is True.
        viaConnectingElements : bool , optional
            If set to False, adjacency edges are created directly between spaces.
            If set to True, each adjacency is routed through a vertex representing
            the connecting element, preferably the IfcOpeningElement when present,
            otherwise the resolved connector element. For walls, a separate
            connecting vertex is created per confirmed corresponding-boundary
            pair to avoid turning long walls into artificial graph hubs. Default
            is False.
        dictionaryMode : str , optional
            The dictionary mode to use for entity dictionaries. Currently supports
            the same basic entity dictionary behaviour as the IFC parser.
            Default is "basic".
        bidirectional : bool , optional
            If set to True, edge dictionaries record the edge as bidirectional.
            Default is True.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        silent : bool , optional
            If set to True, error and warning messages are suppressed.
            Default is False.

        Returns
        -------
        topologic_core.Graph
            A TopologicPy graph whose vertices are IfcSpace entities and whose
            edges represent inferred space adjacency through shared or
            corresponding boundary elements. Returns None if the IFC file cannot
            be parsed.
        """

        import math
        from itertools import combinations

        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Graph import Graph
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        importMode = str(importMode).strip().lower() if importMode is not None else "topology"
        if importMode not in ["topology", "geometry"]:
            if not silent:
                print("IFC.AccessGraph - Error: The importMode parameter must be either 'topology' or 'geometry'. Returning None.")
            return None

        wall_connector_types = {
            "IFCWALL",
            "IFCWALLSTANDARDCASE",
            "IFCWALLELEMENTEDCASE",
        }

        boundary_types = {
            "IFCRELSPACEBOUNDARY",
            "IFCRELSPACEBOUNDARY1STLEVEL",
            "IFCRELSPACEBOUNDARY2NDLEVEL",
        }

        def _normalise_type(value):
            if value is None:
                return None
            value = str(value).strip().upper()
            return value if value else None

        def _normalise_type_set(values):
            if not values:
                return set()
            return set([_normalise_type(v) for v in values if _normalise_type(v)])

        def _display_type(ifc_type):
            if ifc_type is None:
                return None
            ifc_type = str(ifc_type).strip()
            if ifc_type.upper().startswith("IFC"):
                return "Ifc" + ifc_type[3:].lower().title().replace("_", "")
            return ifc_type

        def _root_attr(entity, index):
            try:
                return IFCFastTopology._root_attr(entity, index)
            except Exception:
                return None

        def _refs(value):
            try:
                return IFCFastTopology._refs_in_value(value)
            except Exception:
                return []

        def _entity_from_ref(ref):
            try:
                return IFCFastTopology._entity_from_ref(ref, entities)
            except Exception:
                return None

        def _entity_key(entity):
            if entity is None:
                return None
            gid = _root_attr(entity, 0)
            if gid not in [None, "", "*"]:
                return str(gid)
            return f"#{entity.id}"

        def _entity_name(entity):
            try:
                name = _root_attr(entity, 2)
                return name if name not in [None, "", "*"] else None
            except Exception:
                return None

        def _is_wall_like(entity):
            return entity is not None and _normalise_type(entity.type) in wall_connector_types

        def _is_allowed(raw_element, connector, opening):
            if not allowed_connector_types:
                return True
            for entity in [raw_element, connector, opening]:
                if entity is not None and _normalise_type(entity.type) in allowed_connector_types:
                    return True
            return False

        def _safe_dictionary_by_keys_values(keys, values):
            clean_keys = []
            clean_values = []
            for k, v in zip(keys, values):
                if k is None:
                    continue
                if v is None:
                    v = ""
                if isinstance(v, (list, tuple, set)):
                    v = ", ".join([str(x) for x in v])
                clean_keys.append(k)
                clean_values.append(v)
            try:
                return Dictionary.ByKeysValues(clean_keys, clean_values)
            except Exception:
                d = None
                for k, v in zip(clean_keys, clean_values):
                    try:
                        if d is None:
                            d = Dictionary.ByKeyValue(k, v)
                        else:
                            d = Dictionary.SetValueAtKey(d, k, v)
                    except Exception:
                        pass
                return d

        def _set_values(dictionary, keys, values):
            try:
                return Dictionary.SetValuesAtKeys(dictionary, keys, values)
            except Exception:
                d = dictionary
                for k, v in zip(keys, values):
                    try:
                        d = Dictionary.SetValueAtKey(d, k, v)
                    except Exception:
                        pass
                return d

        def _entity_dictionary(entity, index, role="ifc_entity"):
            if entity is None:
                return _safe_dictionary_by_keys_values(
                    ["index", "label", "type", "vertex_role"],
                    [index, f"Entity_{index}", role, role],
                )

            try:
                d = IFCFastTopology._entity_dictionary(
                    entity,
                    dictionaryMode=dictionaryMode,
                    metadataCache=metadata_cache,
                )
            except Exception:
                d = None

            if d is None:
                gid = _root_attr(entity, 0)
                name = _entity_name(entity)
                key = _entity_key(entity)
                label = name if name not in [None, ""] else (key or f"Entity_{index}")
                d = _safe_dictionary_by_keys_values(
                    [
                        "index",
                        "IFC_id",
                        "IFC_key",
                        "IFC_type",
                        "IFC_global_id",
                        "IFC_name",
                        "label",
                        "type",
                        "vertex_role",
                    ],
                    [
                        index,
                        entity.id,
                        f"#{entity.id}",
                        _display_type(entity.type),
                        gid if gid not in [None, "", "*"] else "",
                        name if name not in [None, "", "*"] else "",
                        label,
                        role,
                        role,
                    ],
                )
            else:
                d = _set_values(
                    d,
                    ["index", "vertex_role"],
                    [index, role],
                )

            return d

        def _space_dictionary(space_entity, index):
            return _entity_dictionary(space_entity, index, role="space")

        def _connecting_element_dictionary(entity,
                                           index,
                                           connector=None,
                                           opening=None,
                                           boundary_ids=None,
                                           source=""):
            d = _entity_dictionary(entity, index, role="connecting_element")

            connector_gid = _root_attr(connector, 0) if connector is not None else None
            connector_name = _entity_name(connector) if connector is not None else None
            opening_gid = _root_attr(opening, 0) if opening is not None else None
            opening_name = _entity_name(opening) if opening is not None else None

            d = _set_values(
                d,
                [
                    "connector_source",
                    "connector_id",
                    "connector_key",
                    "connector_global_id",
                    "connector_type",
                    "connector_name",
                    "opening_id",
                    "opening_key",
                    "opening_global_id",
                    "opening_type",
                    "opening_name",
                    "space_boundary_ids",
                ],
                [
                    source,
                    connector.id if connector is not None else "",
                    _entity_key(connector) or "",
                    connector_gid if connector_gid not in [None, "", "*"] else "",
                    _display_type(connector.type) if connector is not None else "",
                    connector_name if connector_name not in [None, "", "*"] else "",
                    opening.id if opening is not None else "",
                    _entity_key(opening) or "",
                    opening_gid if opening_gid not in [None, "", "*"] else "",
                    _display_type(opening.type) if opening is not None else "",
                    opening_name if opening_name not in [None, "", "*"] else "",
                    sorted(list(boundary_ids or [])),
                ],
            )
            return d

        def _edge_dictionary(space_a, space_b, connector, boundary_ids, index_a, index_b, source=""):
            connector_key = _entity_key(connector)
            connector_gid = _root_attr(connector, 0)
            connector_name = _entity_name(connector)

            return _safe_dictionary_by_keys_values(
                [
                    "src",
                    "dst",
                    "IFC_type",
                    "relationship_type",
                    "predicate",
                    "edge_role",
                    "connector_source",
                    "connector_id",
                    "connector_key",
                    "connector_global_id",
                    "connector_type",
                    "connector_name",
                    "space_a",
                    "space_b",
                    "space_a_name",
                    "space_b_name",
                    "space_a_global_id",
                    "space_b_global_id",
                    "space_boundary_ids",
                    "bidirectional",
                    "weight",
                ],
                [
                    index_a,
                    index_b,
                    "IfcRelSpaceAdjacency",
                    "IfcRelSpaceAdjacency",
                    "adjacent_to",
                    "space_to_space",
                    source,
                    connector.id if connector is not None else "",
                    connector_key or "",
                    connector_gid if connector_gid not in [None, "", "*"] else "",
                    _display_type(connector.type) if connector is not None else "",
                    connector_name if connector_name not in [None, "", "*"] else "",
                    _entity_key(space_a) or "",
                    _entity_key(space_b) or "",
                    _entity_name(space_a) or "",
                    _entity_name(space_b) or "",
                    _root_attr(space_a, 0) or "",
                    _root_attr(space_b, 0) or "",
                    sorted(list(boundary_ids or [])),
                    bool(bidirectional),
                    1.0,
                ],
            )

        def _routed_edge_dictionary(src_index,
                                    dst_index,
                                    space_entity,
                                    connecting_entity,
                                    connector,
                                    boundary_ids,
                                    source=""):
            connector_gid = _root_attr(connector, 0) if connector is not None else None
            connecting_gid = _root_attr(connecting_entity, 0) if connecting_entity is not None else None

            return _safe_dictionary_by_keys_values(
                [
                    "src",
                    "dst",
                    "IFC_type",
                    "relationship_type",
                    "predicate",
                    "edge_role",
                    "connector_source",
                    "space_id",
                    "space_key",
                    "space_global_id",
                    "space_name",
                    "connecting_element_id",
                    "connecting_element_key",
                    "connecting_element_global_id",
                    "connecting_element_type",
                    "connector_id",
                    "connector_key",
                    "connector_global_id",
                    "connector_type",
                    "space_boundary_ids",
                    "bidirectional",
                    "weight",
                ],
                [
                    src_index,
                    dst_index,
                    "IfcRelSpaceBoundaryAdjacency",
                    "IfcRelSpaceBoundaryAdjacency",
                    "connects_to",
                    "space_to_connecting_element",
                    source,
                    space_entity.id if space_entity is not None else "",
                    _entity_key(space_entity) or "",
                    _root_attr(space_entity, 0) or "",
                    _entity_name(space_entity) or "",
                    connecting_entity.id if connecting_entity is not None else "",
                    _entity_key(connecting_entity) or "",
                    connecting_gid if connecting_gid not in [None, "", "*"] else "",
                    _display_type(connecting_entity.type) if connecting_entity is not None else "",
                    connector.id if connector is not None else "",
                    _entity_key(connector) or "",
                    connector_gid if connector_gid not in [None, "", "*"] else "",
                    _display_type(connector.type) if connector is not None else "",
                    sorted(list(boundary_ids or [])),
                    bool(bidirectional),
                    1.0,
                ],
            )

        def _bbox_centre_by_fast_mesh(entity):
            if entity is None:
                return None

            try:
                bbox = IFCFastTopology.BoundingBoxDataByProduct(entity, entities)
                if bbox is not None and bbox.get("center") is not None:
                    c = bbox.get("center")
                    return (float(c[0]), float(c[1]), float(c[2]))
            except Exception:
                pass

            try:
                mesh = IFCFastTopology.MeshDataByProduct(entity, entities)
            except Exception:
                mesh = None

            if mesh is None:
                return None

            vertices = mesh.get("vertices", None)
            if not vertices:
                return None

            xs = []
            ys = []
            zs = []
            for v in vertices:
                if v is None or len(v) < 3:
                    continue
                try:
                    xs.append(float(v[0]))
                    ys.append(float(v[1]))
                    zs.append(float(v[2]))
                except Exception:
                    pass

            if not xs:
                return None

            return (
                0.5 * (min(xs) + max(xs)),
                0.5 * (min(ys) + max(ys)),
                0.5 * (min(zs) + max(zs)),
            )

        def _make_vertex(x, y, z, dictionary):
            try:
                vertex = Vertex.ByCoordinates(float(x), float(y), float(z))
            except Exception:
                return None

            try:
                vertex = Topology.SetDictionary(vertex, dictionary, silent=True)
            except TypeError:
                try:
                    vertex = Topology.SetDictionary(vertex, dictionary)
                except Exception:
                    pass
            except Exception:
                pass

            return vertex

        def _make_edge(vertex_a, vertex_b, dictionary):
            try:
                edge = Edge.ByVertices([vertex_a, vertex_b], tolerance=tolerance, silent=True)
            except TypeError:
                try:
                    edge = Edge.ByVertices(vertex_a, vertex_b)
                except Exception:
                    edge = None
            except Exception:
                edge = None

            if edge is None:
                return None

            try:
                edge = Topology.SetDictionary(edge, dictionary, silent=True)
            except TypeError:
                try:
                    edge = Topology.SetDictionary(edge, dictionary)
                except Exception:
                    pass
            except Exception:
                pass

            return edge

        def _add_pair(pair_to_data,
                      space_a,
                      space_b,
                      connector,
                      connecting_entity,
                      opening,
                      boundary_ids,
                      source):
            key_a = _entity_key(space_a)
            key_b = _entity_key(space_b)

            if key_a is None or key_b is None or key_a == key_b:
                return

            pair_key = tuple(sorted([key_a, key_b]))

            if pair_key not in pair_to_data:
                pair_to_data[pair_key] = {
                    "space_a": space_a,
                    "space_b": space_b,
                    "connectors": [],
                    "connecting_entities": [],
                    "openings": [],
                    "boundary_ids": set(),
                    "sources": set(),
                }

            if connector is not None:
                pair_to_data[pair_key]["connectors"].append(connector)
            if connecting_entity is not None:
                pair_to_data[pair_key]["connecting_entities"].append(connecting_entity)
            if opening is not None:
                pair_to_data[pair_key]["openings"].append(opening)
            pair_to_data[pair_key]["boundary_ids"].update(set(boundary_ids or []))
            pair_to_data[pair_key]["sources"].add(source)
        
        def _vertex_xyz(vertex):
            try:
                return [
                    float(Vertex.X(vertex)),
                    float(Vertex.Y(vertex)),
                    float(Vertex.Z(vertex)),
                ]
            except Exception:
                return None

        def _topology_by_fast_mesh(entity):
            try:
                from topologicpy.Topology import Topology

                mesh = IFCFastTopology.MeshDataByProduct(
                    entity,
                    entities,
                    circleSides=24,
                    scale=1.0,
                )

                if mesh is None or not mesh.get("vertices"):
                    return None

                try:
                    topology = Topology.ByGeometry(
                        vertices=mesh.get("vertices") or [],
                        edges=mesh.get("edges") or [],
                        faces=mesh.get("faces") or [],
                        tolerance=tolerance,
                        silent=True,
                    )
                except TypeError:
                    topology = Topology.ByGeometry(
                        vertices=mesh.get("vertices") or [],
                        edges=mesh.get("edges") or [],
                        faces=mesh.get("faces") or [],
                        tolerance=tolerance,
                    )

                if topology is None:
                    return None

                return topology
            except Exception:
                return None


        def _bbox_data_by_fast_mesh(entity):
            try:
                mesh = IFCFastTopology.MeshDataByProduct(
                    entity,
                    entities,
                    circleSides=24,
                    scale=1.0,
                )
            except Exception:
                return None

            if mesh is None:
                return None

            vertices = mesh.get("vertices", None)

            if not vertices:
                return None

            xs = []
            ys = []
            zs = []

            for v in vertices:
                if v is None or len(v) < 3:
                    continue
                try:
                    xs.append(float(v[0]))
                    ys.append(float(v[1]))
                    zs.append(float(v[2]))
                except Exception:
                    continue

            if not xs:
                return None

            x_min = min(xs)
            y_min = min(ys)
            z_min = min(zs)

            x_max = max(xs)
            y_max = max(ys)
            z_max = max(zs)

            return {
                "min": [x_min, y_min, z_min],
                "max": [x_max, y_max, z_max],
                "center": [
                    0.5 * (x_min + x_max),
                    0.5 * (y_min + y_max),
                    0.5 * (z_min + z_max),
                ],
            }


        def _is_internal_xyz(x, y, z, topology):
            if topology is None:
                return False

            try:
                candidate = Vertex.ByCoordinates(float(x), float(y), float(z))
            except Exception:
                return False

            try:
                return bool(Vertex.IsInternal(candidate, topology, tolerance=tolerance))
            except TypeError:
                try:
                    return bool(Vertex.IsInternal(candidate, topology))
                except Exception:
                    return False
            except Exception:
                return False


        def _internal_point_by_space_geometry(space):
            """
            Returns a point guaranteed, as far as TopologicPy can test, to be inside
            the space geometry.

            The returned tuple is:

                (x, y, z, placement_label)

            Returns None if no valid internal point can be found.
            """

            bbox = _bbox_data_by_fast_mesh(space)

            if bbox is None:
                return None

            topology = _topology_by_fast_mesh(space)
            if topology is None:
                return None
            if Topology.IsInstance(topology, "cluster"):
                topology = Topology.SelfMerge(topology)
            if Topology.IsInstance(topology, "cluster"):
                cells = Topology.Cells(topology)
                if len(cells) > 0:
                    iv = Topology.InternalVertex(cells[0])
                    return Vertex.X(iv), Vertex.Y(iv), Vertex.Z(iv)
                faces = Topology.Faces(topology)
                if len(faces) > 0:
                    iv = Topology.InternalVertex(faces[0])
                    return Vertex.X(iv), Vertex.Y(iv), Vertex.Z(iv)
                edges = Topology.Edges(topology)
                if len(edges) > 0:
                    iv = Topology.InternalVertex(edges[0])
                    return Vertex.X(iv), Vertex.Y(iv), Vertex.Z(iv)
                vertices = Topology.Vertices(topology)
                if len(vertices) > 0:
                    iv = vertices[0]
                    return Vertex.X(iv), Vertex.Y(iv), Vertex.Z(iv)

            iv = Topology.InternalVertex(topology)
            return Vertex.X(iv), Vertex.Y(iv), Vertex.Z(iv)

        # --------------------------------------------------------------
        # Parse IFC once.
        # --------------------------------------------------------------

        try:
            entities = IFC.Entities(file, silent=silent)
        except Exception:
            entities = IFC._entities_from_input(file, silent=silent)

        if entities is None or not isinstance(entities, dict):
            if not silent:
                print("IFC.AccessGraph - Error: Could not read or parse the input IFC file. Returning None.")
            return None

        metadata_cache = IFCFastTopology._entity_metadata_cache(entities, dictionaryMode=dictionaryMode)

        if connectingElementTypes is None:
            connectingElementTypes = ["IFCDOOR", "IFCDOORSTANDARDCASE", "IFCOPENINGELEMENT"]

        allowed_connector_types = _normalise_type_set(connectingElementTypes)

        # --------------------------------------------------------------
        # Collect spaces.
        # --------------------------------------------------------------

        spaces = []
        for entity in entities.values():
            if _normalise_type(entity.type) == "IFCSPACE":
                spaces.append(entity)

        spaces.sort(key=lambda e: e.id)

        if len(spaces) < 1:
            if not silent:
                print("IFC.AccessGraph - Warning: No IfcSpace entities were found. Returning None.")
            return None

        # --------------------------------------------------------------
        # Resolve openings and fillings.
        # IfcRelFillsElement args:
        #   4 = RelatingOpeningElement
        #   5 = RelatedBuildingElement
        # --------------------------------------------------------------

        opening_to_filling = {}
        filling_to_opening = {}

        if useFillingElements:
            for rel in entities.values():
                if _normalise_type(rel.type) != "IFCRELFILLSELEMENT":
                    continue
                if len(rel.args) <= 5:
                    continue

                opening_refs = _refs(rel.args[4])
                filling_refs = _refs(rel.args[5])

                for opening_ref in opening_refs:
                    opening = _entity_from_ref(opening_ref)
                    if opening is None:
                        continue

                    for filling_ref in filling_refs:
                        filling = _entity_from_ref(filling_ref)
                        if filling is not None:
                            opening_to_filling[opening.id] = filling
                            filling_to_opening[filling.id] = opening

        # --------------------------------------------------------------
        # Collect boundary records.
        # Local connectors are grouped by connector entity.
        # Wall-like connectors are NOT grouped into pairwise combinations.
        # They are handled later using CorrespondingBoundary.
        #
        # IfcRelSpaceBoundary args:
        #   4  = RelatingSpace
        #   5  = RelatedBuildingElement
        #   10 = CorrespondingBoundary for IfcRelSpaceBoundary2ndLevel
        # --------------------------------------------------------------

        boundary_records = {}
        connector_to_spaces = {}
        connector_to_boundary_ids = {}

        for rel in entities.values():
            rel_type = _normalise_type(rel.type)

            if rel_type not in boundary_types:
                continue
            if len(rel.args) <= 5:
                continue

            space_refs = _refs(rel.args[4])
            element_refs = _refs(rel.args[5])

            if len(space_refs) < 1 or len(element_refs) < 1:
                continue

            for space_ref in space_refs:
                space_entity = _entity_from_ref(space_ref)

                if space_entity is None or _normalise_type(space_entity.type) != "IFCSPACE":
                    continue

                for element_ref in element_refs:
                    raw_element = _entity_from_ref(element_ref)

                    if raw_element is None:
                        continue

                    raw_type = _normalise_type(raw_element.type)
                    opening_entity = raw_element if raw_type == "IFCOPENINGELEMENT" else filling_to_opening.get(raw_element.id, None)
                    connector_entity = raw_element

                    if useFillingElements and raw_type == "IFCOPENINGELEMENT":
                        connector_entity = opening_to_filling.get(raw_element.id, raw_element)

                    if not _is_allowed(raw_element, connector_entity, opening_entity):
                        continue

                    is_wall = _is_wall_like(raw_element) or _is_wall_like(connector_entity)
                    connector_key = _entity_key(connector_entity)

                    if connector_key is None:
                        continue

                    rec = {
                        "rel": rel,
                        "rel_type": rel_type,
                        "space": space_entity,
                        "raw_element": raw_element,
                        "connector": connector_entity,
                        "opening": opening_entity,
                        "connecting_entity": opening_entity if opening_entity is not None else connector_entity,
                        "is_wall": is_wall,
                    }
                    boundary_records[rel.id] = rec

                    if is_wall:
                        continue

                    if connector_key not in connector_to_spaces:
                        connector_to_spaces[connector_key] = {
                            "connector": connector_entity,
                            "opening": opening_entity,
                            "connecting_entity": rec["connecting_entity"],
                            "spaces": {},
                        }

                    connector_to_spaces[connector_key]["spaces"][_entity_key(space_entity)] = space_entity
                    connector_to_boundary_ids.setdefault(connector_key, set()).add(rel.id)

        # --------------------------------------------------------------
        # Build adjacency pairs.
        # 1. Local connectors: pairwise grouping is acceptable.
        # 2. Walls: only CorrespondingBoundary pairs are accepted.
        # --------------------------------------------------------------

        pair_to_data = {}

        # 1. Local connectors such as doors/openings/windows.
        for connector_key, data in connector_to_spaces.items():
            connector = data.get("connector")
            opening = data.get("opening")
            connecting_entity = data.get("connecting_entity")
            connector_spaces = list(data.get("spaces", {}).values())

            if len(connector_spaces) < 2:
                continue

            connector_spaces.sort(key=lambda e: e.id)

            for space_a, space_b in combinations(connector_spaces, 2):
                _add_pair(
                    pair_to_data,
                    space_a,
                    space_b,
                    connector,
                    connecting_entity,
                    opening,
                    connector_to_boundary_ids.get(connector_key, set()),
                    "shared_local_connector",
                )

        # 2. Wall-like connectors through IfcRelSpaceBoundary2ndLevel.CorrespondingBoundary.
        for rec in list(boundary_records.values()):
            if not rec.get("is_wall"):
                continue

            rel = rec.get("rel")
            if rel is None or _normalise_type(rel.type) != "IFCRELSPACEBOUNDARY2NDLEVEL":
                continue
            if len(rel.args) <= 10:
                continue

            corresponding_refs = _refs(rel.args[10])
            if not corresponding_refs:
                continue

            for corresponding_ref in corresponding_refs:
                corresponding_rel = _entity_from_ref(corresponding_ref)
                if corresponding_rel is None:
                    continue

                corr_rec = boundary_records.get(corresponding_rel.id)
                if corr_rec is None or not corr_rec.get("is_wall"):
                    continue

                space_a = rec.get("space")
                space_b = corr_rec.get("space")

                if space_a is None or space_b is None:
                    continue

                raw_a = rec.get("raw_element")
                raw_b = corr_rec.get("raw_element")

                # Be conservative: corresponding wall boundaries should refer to
                # the same wall/building element. If not, skip the pair rather
                # than risk false adjacency.
                if _entity_key(raw_a) != _entity_key(raw_b):
                    continue

                connector = rec.get("connector") or raw_a
                connecting_entity = rec.get("connecting_entity") or connector
                opening = rec.get("opening")
                boundary_ids = {rel.id, corresponding_rel.id}

                _add_pair(
                    pair_to_data,
                    space_a,
                    space_b,
                    connector,
                    connecting_entity,
                    opening,
                    boundary_ids,
                    "corresponding_wall_boundary",
                )

        # --------------------------------------------------------------
        # Decide which spaces become vertices.
        # --------------------------------------------------------------

        if includeIsolatedSpaces:
            graph_spaces = spaces
        else:
            used_space_keys = set()
            for pair_key in pair_to_data.keys():
                used_space_keys.update(pair_key)
            graph_spaces = [space for space in spaces if _entity_key(space) in used_space_keys]

        graph_spaces.sort(key=lambda e: e.id)

        # --------------------------------------------------------------
        # Create IfcSpace vertices.
        # --------------------------------------------------------------

        vertices = []
        edges = []
        space_key_to_index = {}
        space_key_to_xyz = {}

        n = max(len(graph_spaces), 1)
        radius = max(1.0, float(n) / (2.0 * math.pi))

        for i, space in enumerate(graph_spaces):
            angle = (2.0 * math.pi * float(i)) / float(n)
            fallback_x = radius * math.cos(angle)
            fallback_y = radius * math.sin(angle)
            fallback_z = 0.0

            x = fallback_x
            y = fallback_y
            z = fallback_z
            vertex_placement = "topology_layout"

            if importMode == "geometry":
                if useInternalVertex == True:
                    centre = _internal_point_by_space_geometry(space)

                    if centre is not None:
                        x, y, z = centre
                        vertex_placement = "geometry_internal_vertex"
                    else:
                        centre = _bbox_centre_by_fast_mesh(space)

                        if centre is not None:
                            x, y, z = centre
                            vertex_placement = "geometry_bbox_centre_unverified"
                        else:
                            vertex_placement = "topology_layout_fallback"
                else:
                    centre = _bbox_centre_by_fast_mesh(space)
                    if centre is not None:
                        x, y, z = centre
                        vertex_placement = "geometry_bbox_centre_unverified"
                    else:
                        vertex_placement = "topology_layout_fallback"


            d = _space_dictionary(space, len(vertices))
            d = _set_values(
                d,
                ["import_mode", "vertex_placement", "x", "y", "z"],
                [importMode, vertex_placement, float(x), float(y), float(z)],
            )

            vertex = _make_vertex(x, y, z, d)
            if vertex is None:
                if not silent:
                    print(f"IFC.AccessGraph - Warning: Could not create vertex for IfcSpace #{space.id}. Skipping.")
                continue

            key = _entity_key(space)
            space_key_to_index[key] = len(vertices)
            space_key_to_xyz[key] = (float(x), float(y), float(z))
            vertices.append(vertex)

        # --------------------------------------------------------------
        # Create edges, optionally routed through connecting elements.
        # --------------------------------------------------------------

        connecting_key_to_index = {}
        connecting_key_to_xyz = {}

        for pair_key, data in pair_to_data.items():
            key_a, key_b = pair_key

            if key_a not in space_key_to_index or key_b not in space_key_to_index:
                continue

            index_a = space_key_to_index[key_a]
            index_b = space_key_to_index[key_b]

            if index_a == index_b:
                continue

            vertex_a = vertices[index_a]
            vertex_b = vertices[index_b]

            connectors = data.get("connectors", [])
            connecting_entities = data.get("connecting_entities", [])
            openings = data.get("openings", [])
            connector = connectors[0] if connectors else None
            opening = openings[0] if openings else None
            connecting_entity = connecting_entities[0] if connecting_entities else (opening if opening is not None else connector)
            boundary_ids = data.get("boundary_ids", set())
            source = ", ".join(sorted(list(data.get("sources", set()))))

            if not viaConnectingElements:
                d = _edge_dictionary(
                    data.get("space_a"),
                    data.get("space_b"),
                    connector,
                    boundary_ids,
                    index_a,
                    index_b,
                    source=source,
                )

                if len(connectors) > 1:
                    connector_keys = [_entity_key(c) for c in connectors if c is not None]
                    connector_types = [_display_type(c.type) for c in connectors if c is not None]
                    d = _set_values(
                        d,
                        ["connector_count", "connector_keys", "connector_types"],
                        [len(connectors), connector_keys, connector_types],
                    )

                edge = _make_edge(vertex_a, vertex_b, d)
                if edge is not None:
                    edges.append(edge)
                continue

            # For wall corresponding-boundary pairs, do NOT reuse one wall vertex
            # across all pairs. Doing so would create a high-degree wall hub and
            # would again imply implausible cross-wall connectivity. For local
            # connectors, reuse the door/opening/window vertex.
            if "corresponding_wall_boundary" in data.get("sources", set()):
                connecting_key = "boundary_pair::" + "::".join([str(x) for x in sorted(list(boundary_ids))])
            else:
                connecting_key = _entity_key(connecting_entity) or _entity_key(connector) or ("connector::" + "::".join([str(x) for x in sorted(list(boundary_ids))]))

            if connecting_key not in connecting_key_to_index:
                
                if importMode == "geometry":
                    if useInternalVertex == True:
                        centre = _internal_point_by_space_geometry(connecting_entity)

                        if centre is not None:
                            x, y, z = centre
                            vertex_placement = "geometry_internal_vertex"
                        else:
                            centre = _bbox_centre_by_fast_mesh(connecting_entity)

                            if centre is not None:
                                x, y, z = centre
                                vertex_placement = "geometry_bbox_centre_unverified"
                            else:
                                vertex_placement = "topology_layout_fallback"
                    else:
                        centre = _bbox_centre_by_fast_mesh(connecting_entity)
                        if centre is not None:
                            x, y, z = centre
                            vertex_placement = "geometry_bbox_centre_unverified"
                        else:
                            vertex_placement = "topology_layout_fallback"
                if centre is not None:
                    cx, cy, cz = centre
                    vertex_placement = "geometry_bbox_centre"
                else:
                    ax, ay, az = space_key_to_xyz.get(key_a, (0.0, 0.0, 0.0))
                    bx, by, bz = space_key_to_xyz.get(key_b, (0.0, 0.0, 0.0))
                    cx = 0.5 * (ax + bx)
                    cy = 0.5 * (ay + by)
                    cz = 0.5 * (az + bz)
                    vertex_placement = "incident_space_midpoint" if importMode == "geometry" else "topology_layout_midpoint"

                d = _connecting_element_dictionary(
                    connecting_entity,
                    len(vertices),
                    connector=connector,
                    opening=opening,
                    boundary_ids=boundary_ids,
                    source=source,
                )
                d = _set_values(
                    d,
                    ["import_mode", "vertex_placement", "x", "y", "z"],
                    [importMode, vertex_placement, float(cx), float(cy), float(cz)],
                )

                connecting_vertex = _make_vertex(cx, cy, cz, d)
                if connecting_vertex is None:
                    continue

                connecting_key_to_index[connecting_key] = len(vertices)
                connecting_key_to_xyz[connecting_key] = (float(cx), float(cy), float(cz))
                vertices.append(connecting_vertex)

            index_c = connecting_key_to_index[connecting_key]
            vertex_c = vertices[index_c]

            d_ac = _routed_edge_dictionary(
                index_a,
                index_c,
                data.get("space_a"),
                connecting_entity,
                connector,
                boundary_ids,
                source=source,
            )
            edge_ac = _make_edge(vertex_a, vertex_c, d_ac)
            if edge_ac is not None:
                edges.append(edge_ac)

            d_cb = _routed_edge_dictionary(
                index_c,
                index_b,
                data.get("space_b"),
                connecting_entity,
                connector,
                boundary_ids,
                source=source,
            )
            edge_cb = _make_edge(vertex_c, vertex_b, d_cb)
            if edge_cb is not None:
                edges.append(edge_cb)

        try:
            graph = Graph.ByVerticesEdges(vertices, edges, index=True)
        except Exception as e:
            if not silent:
                print(f"IFC.AccessGraph - Error: Could not create graph. {e} Returning None.")
            return None

        return graph


    @staticmethod
    def DoorSpaceCardinalityReport(file,
                                   connectorTypes: list = None,
                                   includeOpenings: bool = True,
                                   includeWindows: bool = False,
                                   includeZeroSpaceConnectors: bool = True,
                                   highCardinalityThreshold: int = 2,
                                   silent: bool = False):
        """
        Returns a diagnostic report of door/opening-to-space cardinalities in an IFC file.

        The method inspects IfcRelSpaceBoundary relationships to determine how many
        IfcSpace entities are associated with each connector element. It also uses
        IfcRelFillsElement to resolve IfcOpeningElement entities to their filling
        elements, usually IfcDoor or IfcWindow.

        A correctly exported door connecting two rooms is normally expected to have
        a space_count of 2. Doors with one related space may indicate an external
        door, incomplete space boundaries, or an export issue. Doors with more than
        two related spaces are usually suspicious and are flagged as high_cardinality.

        Parameters
        ----------
        file : dict, str, or ifcopenshell.file-like object
            The input IFC source.
        connectorTypes : list , optional
            IFC connector types to include. If None, defaults to IfcDoor and,
            optionally, IfcOpeningElement and IfcWindow depending on the other
            parameters. Default is None.
        includeOpenings : bool , optional
            If set to True, unresolved IfcOpeningElement entities are included in
            the report. Default is True.
        includeWindows : bool , optional
            If set to True, IfcWindow and IfcWindowStandardCase are included.
            Default is False.
        includeZeroSpaceConnectors : bool , optional
            If set to True, connector elements found through IfcRelFillsElement are
            included even if no related IfcSpace was found. Default is True.
        highCardinalityThreshold : int , optional
            Cardinalities above this number are labelled high_cardinality.
            Default is 2.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.

        Returns
        -------
        list
            A list of dictionaries. Each dictionary reports one connector element and
            the IfcSpace entities associated with it.
        """

        def _normalise_type(value):
            if value is None:
                return None
            value = str(value).strip().upper()
            return value if value else None

        def _normalise_type_set(values):
            if not values:
                return set()
            return set([_normalise_type(v) for v in values if _normalise_type(v)])

        def _display_type(ifc_type):
            if ifc_type is None:
                return None
            ifc_type = str(ifc_type).strip()
            if ifc_type.upper().startswith("IFC"):
                return "Ifc" + ifc_type[3:].lower().title().replace("_", "")
            return ifc_type

        def _root_attr(entity, index):
            try:
                return IFCFastTopology._root_attr(entity, index)
            except Exception:
                return None

        def _refs(value):
            try:
                return IFCFastTopology._refs_in_value(value)
            except Exception:
                return []

        def _entity_from_ref(ref):
            try:
                return IFCFastTopology._entity_from_ref(ref, entities)
            except Exception:
                return None

        def _entity_key(entity):
            if entity is None:
                return None
            gid = _root_attr(entity, 0)
            if gid not in [None, "", "*"]:
                return str(gid)
            return f"#{entity.id}"

        def _entity_name(entity):
            name = _root_attr(entity, 2)
            return name if name not in [None, "", "*"] else None

        def _entity_row(entity):
            return {
                "id": entity.id if entity is not None else None,
                "key": f"#{entity.id}" if entity is not None else None,
                "global_id": _root_attr(entity, 0) if entity is not None else None,
                "type": _display_type(entity.type) if entity is not None else None,
                "name": _entity_name(entity) if entity is not None else None,
            }

        def _status(space_count):
            if space_count <= 0:
                return "no_space_boundary"
            if space_count == 1:
                return "single_space"
            if space_count <= highCardinalityThreshold:
                return "ok"
            return "high_cardinality"

        # --------------------------------------------------------------
        # Parse IFC once.
        # --------------------------------------------------------------

        try:
            entities = IFC.Entities(file, silent=silent)
        except Exception:
            try:
                entities = IFC._entities_from_input(file, silent=silent)
            except Exception:
                entities = None

        if entities is None or not isinstance(entities, dict):
            if not silent:
                print("IFC.DoorSpaceCardinalityReport - Error: Could not read or parse the input IFC file. Returning None.")
            return None

        # --------------------------------------------------------------
        # Decide connector types.
        # --------------------------------------------------------------

        if connectorTypes is None:
            connectorTypes = ["IFCDOOR", "IFCDOORSTANDARDCASE"]
            if includeOpenings:
                connectorTypes += ["IFCOPENINGELEMENT"]
            if includeWindows:
                connectorTypes += ["IFCWINDOW", "IFCWINDOWSTANDARDCASE"]

        connector_type_set = _normalise_type_set(connectorTypes)

        boundary_types = {
            "IFCRELSPACEBOUNDARY",
            "IFCRELSPACEBOUNDARY1STLEVEL",
            "IFCRELSPACEBOUNDARY2NDLEVEL",
        }

        # --------------------------------------------------------------
        # Resolve opening <-> filling element relationships.
        #
        # IfcRelFillsElement:
        #   args[4] = RelatingOpeningElement
        #   args[5] = RelatedBuildingElement
        # --------------------------------------------------------------

        opening_to_fillings = {}
        filling_to_openings = {}

        for rel in entities.values():
            if _normalise_type(rel.type) != "IFCRELFILLSELEMENT":
                continue
            if len(rel.args) <= 5:
                continue

            opening_refs = _refs(rel.args[4])
            filling_refs = _refs(rel.args[5])

            for opening_ref in opening_refs:
                opening = _entity_from_ref(opening_ref)
                if opening is None:
                    continue

                for filling_ref in filling_refs:
                    filling = _entity_from_ref(filling_ref)
                    if filling is None:
                        continue

                    opening_to_fillings.setdefault(opening.id, [])
                    filling_to_openings.setdefault(filling.id, [])

                    if filling.id not in [f.id for f in opening_to_fillings[opening.id]]:
                        opening_to_fillings[opening.id].append(filling)

                    if opening.id not in [o.id for o in filling_to_openings[filling.id]]:
                        filling_to_openings[filling.id].append(opening)

        # --------------------------------------------------------------
        # Initialise connector records from entities and fill relationships.
        # --------------------------------------------------------------

        connector_records = {}

        def _ensure_record(connector):
            if connector is None:
                return None

            connector_type = _normalise_type(connector.type)

            if connector_type_set and connector_type not in connector_type_set:
                return None

            key = _entity_key(connector)
            if key is None:
                return None

            if key not in connector_records:
                connector_records[key] = {
                    "connector": connector,
                    "spaces": {},
                    "space_boundary_ids": set(),
                    "space_boundary_keys": set(),
                    "opening_elements": {},
                    "filling_elements": {},
                    "raw_boundary_elements": {},
                }

            return connector_records[key]

        for entity in entities.values():
            entity_type = _normalise_type(entity.type)
            if connector_type_set and entity_type in connector_type_set:
                _ensure_record(entity)

        for opening_id, fillings in opening_to_fillings.items():
            opening = entities.get(opening_id)
            if opening is not None:
                opening_record = _ensure_record(opening)
                if opening_record is not None:
                    for filling in fillings:
                        opening_record["filling_elements"][_entity_key(filling)] = filling

            for filling in fillings:
                filling_record = _ensure_record(filling)
                if filling_record is not None and opening is not None:
                    filling_record["opening_elements"][_entity_key(opening)] = opening

        # --------------------------------------------------------------
        # Traverse IfcRelSpaceBoundary.
        #
        # IfcRelSpaceBoundary:
        #   args[4] = RelatingSpace
        #   args[5] = RelatedBuildingElement
        # --------------------------------------------------------------

        for rel in entities.values():
            rel_type = _normalise_type(rel.type)

            if rel_type not in boundary_types:
                continue
            if len(rel.args) <= 5:
                continue

            space_refs = _refs(rel.args[4])
            element_refs = _refs(rel.args[5])

            if len(space_refs) < 1 or len(element_refs) < 1:
                continue

            spaces = []
            for space_ref in space_refs:
                space = _entity_from_ref(space_ref)
                if space is not None and _normalise_type(space.type) == "IFCSPACE":
                    spaces.append(space)

            if len(spaces) < 1:
                continue

            for element_ref in element_refs:
                element = _entity_from_ref(element_ref)
                if element is None:
                    continue

                # The raw boundary element may be an opening, door, wall, etc.
                possible_connectors = []

                if _normalise_type(element.type) == "IFCOPENINGELEMENT":
                    fillings = opening_to_fillings.get(element.id, [])
                    if fillings:
                        possible_connectors.extend(fillings)
                    if includeOpenings:
                        possible_connectors.append(element)
                else:
                    possible_connectors.append(element)

                    # If this element itself has related openings, keep them as metadata.
                    for opening in filling_to_openings.get(element.id, []):
                        if includeOpenings:
                            possible_connectors.append(opening)

                for connector in possible_connectors:
                    record = _ensure_record(connector)
                    if record is None:
                        continue

                    record["raw_boundary_elements"][_entity_key(element)] = element
                    record["space_boundary_ids"].add(rel.id)
                    record["space_boundary_keys"].add(f"#{rel.id}")

                    if _normalise_type(element.type) == "IFCOPENINGELEMENT" and connector.id != element.id:
                        record["opening_elements"][_entity_key(element)] = element

                    if _normalise_type(connector.type) == "IFCOPENINGELEMENT":
                        for filling in opening_to_fillings.get(connector.id, []):
                            record["filling_elements"][_entity_key(filling)] = filling

                    for space in spaces:
                        record["spaces"][_entity_key(space)] = space

        # --------------------------------------------------------------
        # Build report rows.
        # --------------------------------------------------------------

        rows = []

        for connector_key in sorted(connector_records.keys()):
            record = connector_records[connector_key]
            connector = record["connector"]

            spaces = list(record["spaces"].values())
            spaces.sort(key=lambda e: e.id)

            if len(spaces) == 0 and not includeZeroSpaceConnectors:
                continue

            openings = list(record["opening_elements"].values())
            openings.sort(key=lambda e: e.id)

            fillings = list(record["filling_elements"].values())
            fillings.sort(key=lambda e: e.id)

            raw_boundary_elements = list(record["raw_boundary_elements"].values())
            raw_boundary_elements.sort(key=lambda e: e.id)

            space_count = len(spaces)

            row = {
                "connector_id": connector.id,
                "connector_key": f"#{connector.id}",
                "connector_global_id": _root_attr(connector, 0),
                "connector_type": _display_type(connector.type),
                "connector_name": _entity_name(connector),

                "space_count": space_count,
                "status": _status(space_count),

                "space_ids": [s.id for s in spaces],
                "space_keys": [f"#{s.id}" for s in spaces],
                "space_global_ids": [_root_attr(s, 0) for s in spaces],
                "space_names": [_entity_name(s) for s in spaces],
                "spaces": [_entity_row(s) for s in spaces],

                "opening_ids": [o.id for o in openings],
                "opening_keys": [f"#{o.id}" for o in openings],
                "opening_global_ids": [_root_attr(o, 0) for o in openings],
                "opening_names": [_entity_name(o) for o in openings],

                "filling_ids": [f.id for f in fillings],
                "filling_keys": [f"#{f.id}" for f in fillings],
                "filling_global_ids": [_root_attr(f, 0) for f in fillings],
                "filling_names": [_entity_name(f) for f in fillings],

                "raw_boundary_element_ids": [e.id for e in raw_boundary_elements],
                "raw_boundary_element_keys": [f"#{e.id}" for e in raw_boundary_elements],
                "raw_boundary_element_types": [_display_type(e.type) for e in raw_boundary_elements],

                "space_boundary_count": len(record["space_boundary_ids"]),
                "space_boundary_ids": sorted(list(record["space_boundary_ids"])),
                "space_boundary_keys": sorted(list(record["space_boundary_keys"])),
            }

            rows.append(row)

        return rows


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
