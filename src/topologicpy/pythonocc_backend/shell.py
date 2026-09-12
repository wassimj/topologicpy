from __future__ import annotations

import types
from dataclasses import dataclass
from .topology import (
    Topology,
    _is_null_shape,
    _downward_wrappers,
    TopAbs_VERTEX,
    TopAbs_EDGE,
    TopAbs_FACE,
    )
from .face import Face, FaceUtility
from .edge import Edge, EdgeUtility
from .vertex import Vertex
from .occ_utils import make_occ_shell
from .helpers import unique_by_uuid, vertex_key


@dataclass(eq=False)
class Shell(Topology):
    def __init__(self, shape=None, dictionary=None, contents=None, contexts=None, apertures=None, faces=None):
        super().__init__(shape=shape, dictionary=dictionary, contents=contents, contexts=contexts, apertures=apertures)
        self.faces = list(faces) if faces else []

    @staticmethod
    def ByFaces(faces, tolerance: float = 0.0001, silent: bool = False):
        if faces is None:
            if not silent:
                print("Shell.ByFaces - Error: The input faces parameter is None. Returning None.")
            return None
        if not isinstance(faces, list):
            faces = [faces]
        valid_faces = [face for face in faces if Topology.IsInstance(face, "Face")]
        if len(valid_faces) == 0:
            if not silent:
                print("Shell.ByFaces - Error: The input faces list does not contain any valid faces. Returning None.")
            return None
        occ_shell = make_occ_shell(valid_faces)
        if occ_shell is None:
            if not silent:
                print("Shell.ByFaces - Error: Could not create an OpenCascade shell. Returning None.")
            return None
        # Re-derive the Face wrappers from the sewn occ_shell rather than
        # keeping the original, independently-built valid_faces list.
        # make_occ_shell welds coincident vertices/edges across face
        # boundaries via BRepBuilderAPI_Sewing, but that welding is only
        # useful if the Shell's own Faces()/Edges() (which iterate
        # self.faces, not self.shape) actually see the welded topology. Two
        # faces coming from separately-computed boolean fragments (e.g. each
        # face-pair of Topology.Intersect's per-face decomposition) are
        # geometrically coincident along their shared boundary but were
        # never the same OCCT edge/vertex until sewn -- keeping the
        # pre-sewing faces here silently discarded that welding and doubled
        # edge counts along every such seam.
        from .topology import _iter_occ_subshapes, TopAbs_FACE
        sewn_faces = [Topology.ByOcctShape(f) for f in _iter_occ_subshapes(occ_shell, TopAbs_FACE)]
        sewn_faces = [f for f in sewn_faces if isinstance(f, Face)]
        if len(sewn_faces) == len(valid_faces):
            valid_faces = sewn_faces
        shell = Shell(shape=occ_shell, faces=valid_faces)
        Shell._patch_edge_face_membership(shell, valid_faces, tolerance=tolerance)
        return shell

    @staticmethod
    def ByWires(
        wires,
        triangulate: bool = True,
        polyhedron: bool = True,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """
        Creates a Shell by lofting through ordered profile Wires.

        Two construction modes are supported:

        * ``polyhedron=True`` preserves the historical faceted behaviour. Each
          corresponding pair of section Edges is connected by planar triangles
          or quadrilateral Faces using only the Edge endpoints.
        * ``polyhedron=False`` uses OpenCascade's native
          ``BRepOffsetAPI_ThruSections`` ruled loft. Curved section Edges remain
          genuine curves and the generated side Faces remain genuine ruled
          surfaces; no tessellation is introduced.

        In both modes, all valid section Wires must contain the same number of
        Edges so that correspondence between sections is explicit.

        Parameters
        ----------
        wires : list
            The ordered profile Wires. At least two valid Wires are required.
        triangulate : bool , optional
            If ``polyhedron`` is True, triangulate each faceted side quad into
            two triangular Faces. If False, create one quadrilateral Face.
            Ignored when ``polyhedron`` is False. Default is True.
        polyhedron : bool , optional
            If True, create the historical planar/faceted loft. If False, create
            a curve-preserving native OCCT ruled Shell. Default is True.
        tolerance : float , optional
            The desired geometric tolerance. Default is 0.0001.
        silent : bool , optional
            If True, suppress error and warning messages. Default is False.

        Returns
        -------
        Shell
            The created Shell, or None if construction fails.
        """
        import math
        from .wire import Wire

        if not isinstance(wires, (list, tuple)):
            if not silent:
                print("Shell.ByWires - Error: The input wires parameter is not a valid list. Returning None.")
            return None

        wire_list = [wire for wire in wires if isinstance(wire, Wire)]
        if len(wire_list) < 2:
            if not silent:
                print("Shell.ByWires - Error: At least two valid wires are required. Returning None.")
            return None

        try:
            tolerance = abs(float(tolerance))
        except Exception:
            if not silent:
                print("Shell.ByWires - Error: The input tolerance parameter is not a valid number. Returning None.")
            return None

        if not math.isfinite(tolerance) or tolerance <= 0.0:
            if not silent:
                print("Shell.ByWires - Error: The input tolerance parameter must be greater than zero. Returning None.")
            return None

        triangulate = bool(triangulate)
        polyhedron = bool(polyhedron)

        section_edges = []
        edge_count = None
        for wire in wire_list:
            edges = wire.Edges() or []
            edges = [edge for edge in edges if isinstance(edge, Edge)]
            if not edges:
                if not silent:
                    print("Shell.ByWires - Error: One or more input wires contain no valid edges. Returning None.")
                return None
            if edge_count is None:
                edge_count = len(edges)
            elif len(edges) != edge_count:
                if not silent:
                    print("Shell.ByWires - Error: Corresponding wires must contain the same number of edges. Returning None.")
                return None
            section_edges.append(edges)

        # ------------------------------------------------------------------
        # Curve-preserving native ruled loft.
        # ------------------------------------------------------------------
        if not polyhedron:
            try:
                from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_ThruSections
                from OCC.Core.TopoDS import topods
            except Exception:
                if not silent:
                    print("Shell.ByWires - Error: Could not import the required OpenCascade loft classes. Returning None.")
                return None

            occ_wires = []
            for wire in wire_list:
                shape = getattr(wire, "shape", None)
                if shape is None:
                    return None
                try:
                    if shape.IsNull():
                        return None
                    occ_wire = topods.Wire(shape)
                    if occ_wire.IsNull():
                        return None
                except Exception:
                    return None
                occ_wires.append(occ_wire)

            try:
                loft = BRepOffsetAPI_ThruSections(
                    False,      # isSolid
                    True,       # ruled
                    tolerance,
                )
                # Section edge correspondence is already explicit. Prevent OCCT
                # from splitting sections while searching for compatibility.
                loft.CheckCompatibility(False)
                for occ_wire in occ_wires:
                    loft.AddWire(occ_wire)
                loft.Build()
                if hasattr(loft, "IsDone") and not loft.IsDone():
                    return None
                shape = loft.Shape()
            except Exception:
                return None

            if shape is None:
                return None
            try:
                if shape.IsNull():
                    return None
            except Exception:
                return None

            result = Topology.ByOcctShape(shape)
            if not isinstance(result, Shell):
                if not silent:
                    print("Shell.ByWires - Error: OpenCascade did not produce a valid Shell. Returning None.")
                return None
            return result

        # ------------------------------------------------------------------
        # Historical faceted/polyhedral loft.
        # ------------------------------------------------------------------
        def _endpoints_are_distinct(edge):
            start = getattr(edge, "start", None)
            end = getattr(edge, "end", None)
            if not isinstance(start, Vertex) or not isinstance(end, Vertex):
                return False
            dx = float(end.x) - float(start.x)
            dy = float(end.y) - float(start.y)
            dz = float(end.z) - float(start.z)
            return math.sqrt(dx * dx + dy * dy + dz * dz) > tolerance

        for edges in section_edges:
            if any(not _endpoints_are_distinct(edge) for edge in edges):
                if not silent:
                    print(
                        "Shell.ByWires - Error: Faceted lofting requires section edges "
                        "with distinct start and end vertices. Increase the section Wire "
                        "segmentation or set polyhedron=False. Returning None."
                    )
                return None

        faces = []
        for edges_a, edges_b in zip(section_edges[:-1], section_edges[1:]):
            for edge_a, edge_b in zip(edges_a, edges_b):
                if triangulate:
                    tri1 = Face.ByVertices([
                        edge_a.start,
                        edge_a.end,
                        edge_b.end,
                    ])
                    tri2 = Face.ByVertices([
                        edge_a.start,
                        edge_b.end,
                        edge_b.start,
                    ])
                    if tri1 is not None:
                        faces.append(tri1)
                    if tri2 is not None:
                        faces.append(tri2)
                else:
                    quad = Face.ByVertices([
                        edge_a.start,
                        edge_a.end,
                        edge_b.end,
                        edge_b.start,
                    ])
                    if quad is not None:
                        faces.append(quad)

        if not faces:
            if not silent:
                print("Shell.ByWires - Error: Could not create any side faces. Returning None.")
            return None

        return Shell.ByFaces(
            faces,
            tolerance=tolerance,
            silent=silent,
        )



    @staticmethod
    def _IntrinsicGeodesicPartition(
        vertices,
        face,
        mode="voronoi",
        deflection=None,
        maxIterations: int = 5,
        convergence: float = 0.001,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """
        Shared intrinsic Voronoi/Delaunay engine for one trimmed OCCT Face.

        Intrinsic distance is approximated with triangulated Fast Marching on
        successively refined triangulations of the *trimmed* Face. The temporary triangulation
        is only a numerical substrate: converged partition curves are rebuilt as
        p-curves on the original supporting surface and used to split the exact
        input Face.

        The numerical solve uses NumPy and a local triangulated Fast Marching
        implementation. It does not depend on SciPy, pygeodesic, or any native
        geodesic/linear-solver extension.
        """
        import math
        import os
        import numpy as np

        debug = str(os.environ.get("TOPOLOGICPY_INTRINSIC_DEBUG", "")).strip().lower() in (
            "1", "true", "yes", "on"
        )

        def debug_print(message):
            if debug:
                print(f"Shell.{str(mode).capitalize()} [intrinsic] - {message}", flush=True)

        debug_print("entry")
        debug_print(
            f"engine=v5-delaunay-domain-supports source={__file__} "
            f"FaceUtility.IsInside={getattr(FaceUtility.IsInside, '__module__', '?')}"
        )
        try:
            from OCC.Core.BRep import BRep_Tool
            from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
            from OCC.Core.BOPAlgo import BOPAlgo_Splitter
            from OCC.Core.BRepTopAdaptor import BRepTopAdaptor_FClass2d
            from OCC.Core.GCE2d import GCE2d_MakeSegment
            from OCC.Core.Geom2d import Geom2d_BSplineCurve
            from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
            from OCC.Core.TColgp import TColgp_Array1OfPnt2d
            from OCC.Core.TColStd import TColStd_Array1OfInteger, TColStd_Array1OfReal
            from OCC.Core.gp import gp_Pnt, gp_Pnt2d
            from OCC.Core.TopAbs import TopAbs_REVERSED, TopAbs_IN, TopAbs_ON
            from OCC.Core.TopLoc import TopLoc_Location
            from OCC.Core.TopoDS import topods
            from .topology import _iter_occ_subshapes
            debug_print("OCCT imports complete")
        except Exception:
            if not silent:
                print(
                    "Shell intrinsic geodesics - Error: Required OCCT classes "
                    "are unavailable. Returning None."
                )
            return None

        mode = str(mode).lower()
        if mode not in ("voronoi", "delaunay"):
            return None
        if not isinstance(face, Face):
            return None
        if not isinstance(vertices, list):
            return None
        required = 2 if mode == "voronoi" else 3
        if len(vertices) < required or any(not isinstance(v, Vertex) for v in vertices):
            return None

        try:
            tolerance = abs(float(tolerance))
            convergence = abs(float(convergence))
            maxIterations = int(maxIterations)
        except Exception:
            return None
        if tolerance <= 0.0 or convergence <= 0.0 or maxIterations < 1:
            return None

        debug_print("acquiring OCCT Face and supporting surface")
        occ_face = getattr(face, "shape", None)
        if _is_null_shape(occ_face):
            return None
        try:
            occ_face = topods.Face(occ_face)
            surface = BRep_Tool.Surface(occ_face)
        except Exception:
            return None
        if surface is None:
            return None
        debug_print("OCCT Face and supporting surface acquired")

        # ------------------------------------------------------------------
        # Validate sites against the actual trimmed Face. Project each site only
        # once onto the supporting surface, then classify that UV point against
        # the trimmed Face. This avoids the former duplicate IsInside + projection
        # sequence and does not invoke BRepExtrema.
        # ------------------------------------------------------------------
        debug_print("site validation: start")
        site_points = []
        site_uv = []

        try:
            classifier = BRepTopAdaptor_FClass2d(occ_face, tolerance)
        except Exception:
            classifier = None

        if classifier is None:
            if not silent:
                print(
                    f"Shell.{mode.capitalize()} - Error: Could not create the "
                    "trimmed-Face classifier. Returning None."
                )
            return None

        for i, vertex in enumerate(vertices):
            debug_print(f"site {i}: projection/classification start")
            try:
                point = gp_Pnt(float(vertex.x), float(vertex.y), float(vertex.z))
                projector = GeomAPI_ProjectPointOnSurf(point, surface)
                if projector.NbPoints() < 1:
                    raise RuntimeError("no projection")
                distance = float(projector.LowerDistance())
                if not math.isfinite(distance) or distance > tolerance:
                    raise RuntimeError("site is off surface")
                u, v = projector.LowerDistanceParameters()
                u = float(u)
                v = float(v)
                state = classifier.Perform(gp_Pnt2d(u, v))
                if state not in (TopAbs_IN, TopAbs_ON):
                    raise RuntimeError("site is outside trimmed face")

                # Keep the user's point when it is already within tolerance. It is
                # the site of the intrinsic problem; UV is used only to insert it
                # into the computational triangulation and later rebuild cutters.
                site_points.append(
                    (float(vertex.x), float(vertex.y), float(vertex.z))
                )
                site_uv.append((u, v))
                debug_print(f"site {i}: projection/classification complete")
            except Exception:
                if not silent:
                    print(
                        f"Shell.{mode.capitalize()} - Error: Vertex at index {i} "
                        "is not on the trimmed input Face within tolerance. "
                        "Returning None."
                    )
                return None

        debug_print("site validation: complete")

        for i in range(len(site_points)):
            for j in range(i + 1, len(site_points)):
                if math.dist(site_points[i], site_points[j]) <= tolerance:
                    if not silent:
                        print(
                            f"Shell.{mode.capitalize()} - Error: Site Vertices "
                            f"{i} and {j} coincide within tolerance. Returning None."
                        )
                    return None


        # ------------------------------------------------------------------
        # Exact planar-convex special case.
        #
        # On a convex planar polygon with no holes, intrinsic geodesic distance
        # is exactly ordinary Euclidean distance in the plane. Use that invariant
        # directly rather than approximating it with the heat method. This gives
        # an exact sanity path for rectangles and other convex polygonal Faces.
        # ------------------------------------------------------------------
        def exact_planar_convex_partition():
            """
            Exact Euclidean Voronoi/Delaunay on conservative planar-convex domains.

            The original OCCT Face is always retained as the geometric domain. This
            preserves exact straight, circular, and elliptical boundary curves. The
            special case is deliberately conservative: it accepts convex polygonal
            Faces and exact circular/elliptical Faces, including conics segmented
            into multiple exact trimmed Edges. Other curved planar trims fall back
            to the general intrinsic solver.
            """
            try:
                from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
                from OCC.Core.GeomAbs import (
                    GeomAbs_Circle,
                    GeomAbs_Ellipse,
                    GeomAbs_Plane,
                )
            except Exception:
                return False, None

            # The Euclidean shortcut is valid only on a planar Face without holes.
            try:
                adaptor = BRepAdaptor_Surface(occ_face, True)
                if adaptor.GetType() != GeomAbs_Plane:
                    return False, None
            except Exception:
                return False, None

            try:
                if len(face.InternalBoundaries() or []) != 0:
                    return False, None
            except Exception:
                return False, None

            try:
                boundary_edges = face.Edges() or []
            except Exception:
                boundary_edges = []
            boundary_edges = [edge for edge in boundary_edges if isinstance(edge, Edge)]
            if not boundary_edges:
                return False, None

            # ------------------------------------------------------------------
            # Conservative convex-domain recognition.
            # ------------------------------------------------------------------
            all_linear = all(
                EdgeUtility.IsLinear(edge, tolerance=tolerance)
                for edge in boundary_edges
            )
            exact_conic = False

            if not all_linear:
                # TopologicPy can represent circles/ellipses either as native
                # OCCT conics or as exact rational quadratic NURBS segments.
                # Prefer exact OCCT conic recognition first. If that is not
                # available, identify an exact NURBS ellipse by fitting the
                # sampled boundary points to one ellipse-type conic. Sampling
                # is used ONLY for recognition; the original exact Face remains
                # the geometric domain used by the splitter.
                curve_adaptors = []
                try:
                    for boundary_edge in boundary_edges:
                        edge_shape = getattr(boundary_edge, "shape", None)
                        if _is_null_shape(edge_shape):
                            curve_adaptors = []
                            break
                        curve_adaptors.append(BRepAdaptor_Curve(edge_shape))
                except Exception:
                    curve_adaptors = []

                # ----------------------------------------------------------
                # 1. Native OCCT circle / ellipse.
                # ----------------------------------------------------------
                if curve_adaptors:
                    try:
                        curve_types = [item.GetType() for item in curve_adaptors]
                        if (
                            curve_types
                            and all(item == curve_types[0] for item in curve_types)
                            and curve_types[0] in (GeomAbs_Circle, GeomAbs_Ellipse)
                        ):
                            reference_type = curve_types[0]
                            geometric_tol = max(float(tolerance), 1.0e-9)

                            def point_tuple(point):
                                return np.array(
                                    [float(point.X()), float(point.Y()), float(point.Z())],
                                    dtype=float,
                                )

                            def direction_tuple(direction):
                                value = np.array(
                                    [
                                        float(direction.X()),
                                        float(direction.Y()),
                                        float(direction.Z()),
                                    ],
                                    dtype=float,
                                )
                                length = float(np.linalg.norm(value))
                                return value / length if length > 1.0e-15 else value

                            if reference_type == GeomAbs_Circle:
                                reference = curve_adaptors[0].Circle()
                                reference_center = point_tuple(reference.Location())
                                reference_radius = float(reference.Radius())
                                exact_conic = all(
                                    float(
                                        np.linalg.norm(
                                            point_tuple(item.Circle().Location())
                                            - reference_center
                                        )
                                    ) <= geometric_tol
                                    and abs(float(item.Circle().Radius()) - reference_radius)
                                    <= geometric_tol
                                    for item in curve_adaptors[1:]
                                )
                            else:
                                reference = curve_adaptors[0].Ellipse()
                                reference_center = point_tuple(reference.Location())
                                reference_major = float(reference.MajorRadius())
                                reference_minor = float(reference.MinorRadius())
                                reference_x = direction_tuple(
                                    reference.Position().XDirection()
                                )
                                exact_conic = True
                                for item in curve_adaptors[1:]:
                                    conic = item.Ellipse()
                                    conic_x = direction_tuple(
                                        conic.Position().XDirection()
                                    )
                                    if (
                                        float(
                                            np.linalg.norm(
                                                point_tuple(conic.Location())
                                                - reference_center
                                            )
                                        ) > geometric_tol
                                        or abs(float(conic.MajorRadius()) - reference_major)
                                        > geometric_tol
                                        or abs(float(conic.MinorRadius()) - reference_minor)
                                        > geometric_tol
                                        or abs(float(np.dot(conic_x, reference_x)))
                                        < 1.0 - 1.0e-9
                                    ):
                                        exact_conic = False
                                        break
                    except Exception:
                        exact_conic = False

                # ----------------------------------------------------------
                # 2. Exact rational quadratic NURBS ellipse.
                #
                # Wire.Ellipse can deliberately encode an ellipse as exact
                # rational quadratic NURBS Edges. BRepAdaptor_Curve therefore
                # reports a B-spline rather than GeomAbs_Ellipse. Recognise
                # that representation by checking whether many independently
                # sampled boundary points satisfy one ellipse-type conic to
                # numerical precision.
                # ----------------------------------------------------------
                if not exact_conic and curve_adaptors:
                    try:
                        sample_uvs = []
                        samples_per_edge = 9
                        for curve_adaptor in curve_adaptors:
                            first = float(curve_adaptor.FirstParameter())
                            last = float(curve_adaptor.LastParameter())
                            if (
                                not math.isfinite(first)
                                or not math.isfinite(last)
                                or abs(last - first) <= 1.0e-15
                            ):
                                sample_uvs = []
                                break

                            for sample_index in range(samples_per_edge):
                                parameter = first + (last - first) * (
                                    float(sample_index)
                                    / float(samples_per_edge - 1)
                                )
                                point = curve_adaptor.Value(parameter)
                                projection = GeomAPI_ProjectPointOnSurf(
                                    point, surface
                                )
                                if projection.NbPoints() < 1:
                                    sample_uvs = []
                                    break
                                u, v = projection.LowerDistanceParameters()
                                uv = np.array([float(u), float(v)], dtype=float)
                                if not np.all(np.isfinite(uv)):
                                    sample_uvs = []
                                    break
                                sample_uvs.append(uv)
                            if not sample_uvs:
                                break

                        if len(sample_uvs) >= 6:
                            points_2d = np.asarray(sample_uvs, dtype=float)
                            centre = np.mean(points_2d, axis=0)
                            centred = points_2d - centre
                            scale = max(
                                float(np.ptp(centred[:, 0])),
                                float(np.ptp(centred[:, 1])),
                                1.0e-12,
                            )
                            x = centred[:, 0] / scale
                            y = centred[:, 1] / scale

                            design = np.column_stack(
                                (
                                    x * x,
                                    x * y,
                                    y * y,
                                    x,
                                    y,
                                    np.ones_like(x),
                                )
                            )
                            _, _, vh = np.linalg.svd(
                                design,
                                full_matrices=False,
                            )
                            coefficients = vh[-1]
                            coefficient_norm = float(
                                np.linalg.norm(coefficients)
                            )
                            if coefficient_norm > 1.0e-15:
                                coefficients = coefficients / coefficient_norm
                                residual = float(
                                    np.linalg.norm(design @ coefficients)
                                    / math.sqrt(float(design.shape[0]))
                                )
                                a, b, c = (
                                    float(coefficients[0]),
                                    float(coefficients[1]),
                                    float(coefficients[2]),
                                )
                                quadratic = np.array(
                                    [[a, 0.5 * b], [0.5 * b, c]],
                                    dtype=float,
                                )
                                eigenvalues = np.linalg.eigvalsh(quadratic)
                                ellipse_type = (
                                    float(eigenvalues[0] * eigenvalues[1]) > 0.0
                                    and abs(float(eigenvalues[0])) > 1.0e-10
                                    and abs(float(eigenvalues[1])) > 1.0e-10
                                )
                                fit_tolerance = max(
                                    1.0e-10,
                                    min(1.0e-7, abs(float(tolerance)) * 1.0e-3),
                                )
                                exact_conic = bool(
                                    ellipse_type
                                    and math.isfinite(residual)
                                    and residual <= fit_tolerance
                                )
                                if exact_conic:
                                    debug_print(
                                        "exact planar-convex path: "
                                        f"NURBS ellipse fit residual={residual:.3e}"
                                    )
                    except Exception:
                        exact_conic = False

                if not exact_conic:
                    return False, None

            # Surface UV coordinates on a Geom_Plane are orthonormal Euclidean
            # coordinates, so they can be used directly for exact bisectors.
            sites_2d = [np.asarray(uv, dtype=float) for uv in site_uv]

            if all_linear:
                try:
                    raw_vertices = face.Vertices() or []
                except Exception:
                    raw_vertices = []
                if len(raw_vertices) < 3:
                    return False, None

                boundary_2d = []
                for vertex in raw_vertices:
                    try:
                        point = gp_Pnt(
                            float(vertex.x),
                            float(vertex.y),
                            float(vertex.z),
                        )
                        projection = GeomAPI_ProjectPointOnSurf(point, surface)
                        if projection.NbPoints() < 1:
                            return False, None
                        u, v = projection.LowerDistanceParameters()
                        uv = np.array([float(u), float(v)], dtype=float)
                    except Exception:
                        return False, None
                    if not any(
                        float(np.linalg.norm(uv - existing)) <= tolerance
                        for existing in boundary_2d
                    ):
                        boundary_2d.append(uv)

                if len(boundary_2d) < 3:
                    return False, None

                center_2d = np.mean(np.asarray(boundary_2d), axis=0)
                boundary_2d.sort(
                    key=lambda point: math.atan2(
                        float(point[1] - center_2d[1]),
                        float(point[0] - center_2d[0]),
                    )
                )

                # Convexity is essential: only then is intrinsic distance equal to
                # unconstrained Euclidean distance everywhere in the Face.
                turn_sign = 0
                for index in range(len(boundary_2d)):
                    a = boundary_2d[index]
                    b = boundary_2d[(index + 1) % len(boundary_2d)]
                    c = boundary_2d[(index + 2) % len(boundary_2d)]
                    ab = b - a
                    bc = c - b
                    cross = float(ab[0] * bc[1] - ab[1] * bc[0])
                    if abs(cross) <= tolerance * tolerance:
                        continue
                    sign = 1 if cross > 0.0 else -1
                    if turn_sign == 0:
                        turn_sign = sign
                    elif sign != turn_sign:
                        return False, None
                if turn_sign == 0:
                    return False, None

            debug_print(
                "exact planar-convex path: "
                + ("circle/ellipse" if exact_conic else "linear polygon")
            )

            # ------------------------------------------------------------------
            # Exact UV bounds of the trimmed planar Face. These bounds are used
            # only to make finite cutter Edges long enough to represent complete
            # bisector lines across the domain; they do not approximate the Face.
            # ------------------------------------------------------------------
            def planar_uv_bounds():
                try:
                    from OCC.Core.BRepTools import breptools
                    values = breptools.UVBounds(occ_face)
                    if values is not None and len(values) == 4:
                        return tuple(float(value) for value in values)
                except Exception:
                    pass
                try:
                    from OCC.Core.BRepTools import BRepTools
                    for name in ("UVBounds_s", "UVBounds"):
                        fn = getattr(BRepTools, name, None)
                        if callable(fn):
                            values = fn(occ_face)
                            if values is not None and len(values) == 4:
                                return tuple(float(value) for value in values)
                except Exception:
                    pass
                return None

            bounds = planar_uv_bounds()
            if bounds is None:
                return True, None
            umin, umax, vmin, vmax = bounds
            if (
                not all(math.isfinite(value) for value in bounds)
                or umax <= umin
                or vmax <= vmin
            ):
                return True, None

            uv_center = np.array(
                [0.5 * (umin + umax), 0.5 * (vmin + vmax)],
                dtype=float,
            )
            uv_diagonal = math.hypot(umax - umin, vmax - vmin)
            if not math.isfinite(uv_diagonal) or uv_diagonal <= 1.0e-15:
                return True, None

            def build_curves3d(edge_shape):
                """Ensure a planar p-curve cutter also has a usable 3D curve."""
                if edge_shape is None:
                    return None
                try:
                    from OCC.Core.BRepLib import breplib
                    fn = getattr(breplib, "BuildCurves3d", None)
                    if callable(fn):
                        fn(edge_shape)
                        return edge_shape
                except Exception:
                    pass
                try:
                    from OCC.Core.BRepLib import BRepLib
                    for name in ("BuildCurves3d_s", "BuildCurves3d"):
                        fn = getattr(BRepLib, name, None)
                        if callable(fn):
                            fn(edge_shape)
                            return edge_shape
                except Exception:
                    pass
                return edge_shape

            def planar_pcurve_edge(uv0, uv1):
                uv0 = np.asarray(uv0, dtype=float)
                uv1 = np.asarray(uv1, dtype=float)
                if float(np.linalg.norm(uv1 - uv0)) <= 1.0e-14:
                    return None
                try:
                    curve = GCE2d_MakeSegment(
                        gp_Pnt2d(float(uv0[0]), float(uv0[1])),
                        gp_Pnt2d(float(uv1[0]), float(uv1[1])),
                    ).Value()
                    maker = BRepBuilderAPI_MakeEdge(curve, surface)
                    if hasattr(maker, "IsDone") and not maker.IsDone():
                        return None
                    return build_curves3d(maker.Edge())
                except Exception:
                    return None

            def bisector_edge(site_i, site_j):
                """Return a finite p-curve Edge spanning the complete bisector line."""
                site_i = np.asarray(site_i, dtype=float)
                site_j = np.asarray(site_j, dtype=float)
                normal_2d = site_j - site_i
                norm = float(np.linalg.norm(normal_2d))
                if not math.isfinite(norm) or norm <= max(tolerance, 1.0e-12):
                    return None

                offset = 0.5 * (
                    float(np.dot(site_j, site_j))
                    - float(np.dot(site_i, site_i))
                )
                line_point = normal_2d * (offset / float(np.dot(normal_2d, normal_2d)))
                tangent = np.array(
                    [-normal_2d[1], normal_2d[0]],
                    dtype=float,
                ) / norm

                # Cover the complete Face bounding rectangle even when the closest
                # point of the infinite line lies well outside that rectangle.
                half_length = (
                    float(np.linalg.norm(line_point - uv_center))
                    + 4.0 * uv_diagonal
                    + 1.0
                )
                return planar_pcurve_edge(
                    line_point - tangent * half_length,
                    line_point + tangent * half_length,
                )

            def split_face_with_tools(tool_edges):
                tool_edges = [edge for edge in tool_edges if edge is not None]
                if not tool_edges:
                    return [occ_face]
                try:
                    splitter = BOPAlgo_Splitter()
                    splitter.AddArgument(occ_face)
                    for edge_shape in tool_edges:
                        splitter.AddTool(edge_shape)
                    splitter.SetFuzzyValue(float(tolerance))
                    splitter.Perform()
                    if hasattr(splitter, "HasErrors") and splitter.HasErrors():
                        return None
                    split_shape = splitter.Shape()
                except Exception:
                    return None
                if _is_null_shape(split_shape):
                    return None
                split_faces = [
                    topods.Face(subface)
                    for subface in _iter_occ_subshapes(split_shape, TopAbs_FACE)
                ]
                return split_faces or None

            def face_containing_uv(candidate_faces, uv):
                """Select the split Face containing a known site UV coordinate."""
                p2d = gp_Pnt2d(float(uv[0]), float(uv[1]))
                for candidate in candidate_faces or []:
                    try:
                        sub_classifier = BRepTopAdaptor_FClass2d(
                            topods.Face(candidate),
                            tolerance,
                        )
                        state = sub_classifier.Perform(p2d)
                        if state in (TopAbs_IN, TopAbs_ON):
                            return topods.Face(candidate)
                    except Exception:
                        continue
                return None

            def assemble_faces(result_faces, sew=True):
                wrapped_faces = []
                for face_shape in result_faces or []:
                    try:
                        wrapped = Face.ByOcctShape(topods.Face(face_shape))
                    except Exception:
                        wrapped = None
                    if isinstance(wrapped, Face):
                        wrapped_faces.append(wrapped)
                if not wrapped_faces:
                    return None

                # Voronoi cells are produced by independent per-site splits, so
                # sewing is required to recover one shared OCCT Edge per common
                # bisector. Delaunay faces come from one splitter and already share
                # topology, but sewing is harmless and keeps one consistent path.
                return Shell.ByFaces(
                    wrapped_faces,
                    tolerance=tolerance,
                    silent=True,
                ) if sew else None

            if mode == "voronoi":
                result_faces = []
                for i, site_i in enumerate(sites_2d):
                    tools = []
                    for j, site_j in enumerate(sites_2d):
                        if i == j:
                            continue
                        cutter = bisector_edge(site_i, site_j)
                        if cutter is None:
                            return True, None
                        tools.append(cutter)

                    split_faces = split_face_with_tools(tools)
                    if not split_faces:
                        return True, None
                    cell_face = face_containing_uv(split_faces, site_i)
                    if cell_face is None:
                        return True, None
                    result_faces.append(cell_face)

                result = assemble_faces(result_faces, sew=True)
                if result is None:
                    return True, None
                debug_print(
                    f"exact planar-convex Voronoi: {len(result_faces)} cells"
                )
                return True, result

            # --------------------------------------------------------------
            # Exact Euclidean Delaunay partition of the input Face.
            #
            # Preserve TopologicPy's historical semantics: when a Face is
            # supplied, its boundary participates in the triangulation. The
            # user sites therefore form the interior generators while auxiliary
            # boundary support points anchor the triangulation to the domain.
            # The resulting straight Delaunay edges are used only as cutters of
            # the ORIGINAL exact Face, so circular/elliptical outer boundaries
            # remain exact OCCT conics rather than polygonal approximations.
            # --------------------------------------------------------------
            if len(sites_2d) < 3:
                return True, None

            def orient2d(a, b, c):
                return float(
                    (b[0] - a[0]) * (c[1] - a[1])
                    - (b[1] - a[1]) * (c[0] - a[0])
                )

            def project_xyz_to_uv(point):
                try:
                    pnt = gp_Pnt(
                        float(point[0]),
                        float(point[1]),
                        float(point[2]),
                    )
                    projection = GeomAPI_ProjectPointOnSurf(pnt, surface)
                    if projection.NbPoints() < 1:
                        return None
                    u, v = projection.LowerDistanceParameters()
                    return np.array([float(u), float(v)], dtype=float)
                except Exception:
                    return None

            def append_unique_uv(items, uv, eps=None):
                if uv is None:
                    return False
                uv = np.asarray(uv, dtype=float)
                if uv.shape[0] < 2 or np.any(~np.isfinite(uv[:2])):
                    return False
                uv = uv[:2]
                if eps is None:
                    eps = max(float(tolerance), 1.0e-9)
                for existing in items:
                    if float(np.linalg.norm(uv - existing)) <= eps:
                        return False
                items.append(uv)
                return True

            # Start with the user-supplied sites. These remain the semantic
            # generators; all later points are auxiliary domain supports.
            delaunay_points_2d = []
            for uv in sites_2d:
                append_unique_uv(delaunay_points_2d, uv)
            n_user_sites = len(delaunay_points_2d)
            if n_user_sites < 3:
                return True, None

            # Add actual topological boundary vertices first. This mirrors the
            # historical Shell.Delaunay behaviour for polygonal Faces and for
            # conics already segmented into several exact arcs.
            try:
                boundary_vertices = face.Vertices() or []
            except Exception:
                boundary_vertices = []

            boundary_uv = []
            for vertex in boundary_vertices:
                try:
                    uv = project_xyz_to_uv((vertex.x, vertex.y, vertex.z))
                except Exception:
                    uv = None
                append_unique_uv(boundary_uv, uv)

            # Prefer the Face's existing topological boundary vertices exactly as
            # supplied. This is important for exact conics segmented into several
            # curved Edges: e.g. a circle made from four arcs should contribute
            # exactly its four existing boundary vertices, not an arbitrary dense
            # sampling of the circumference.
            #
            # Only a single-edge (or otherwise under-constrained) closed conic has
            # too few distinct boundary vertices to anchor a domain-covering
            # triangulation. In that exceptional case, introduce auxiliary samples.
            # These samples are ONLY triangulation supports; the final Shell is
            # still cut from the original exact Face and preserves its exact conic.
            if exact_conic and len(boundary_uv) < 3:
                target_samples = 32
                edge_count = max(1, len(boundary_edges))
                samples_per_edge = max(
                    2,
                    int(math.ceil(float(target_samples) / float(edge_count))),
                )
                for boundary_edge in boundary_edges:
                    edge_shape = getattr(boundary_edge, "shape", None)
                    if _is_null_shape(edge_shape):
                        continue
                    try:
                        curve_adaptor = BRepAdaptor_Curve(edge_shape)
                        first = float(curve_adaptor.FirstParameter())
                        last = float(curve_adaptor.LastParameter())
                    except Exception:
                        continue
                    if not (math.isfinite(first) and math.isfinite(last)):
                        continue
                    span = last - first
                    if abs(span) <= 1.0e-15:
                        continue
                    # Exclude the terminal sample: for a closed conic it is the
                    # next Edge's start (or the same seam point for one Edge).
                    for sample_index in range(samples_per_edge):
                        parameter = first + span * (
                            float(sample_index) / float(samples_per_edge)
                        )
                        try:
                            point = curve_adaptor.Value(float(parameter))
                            uv = project_xyz_to_uv(
                                (float(point.X()), float(point.Y()), float(point.Z()))
                            )
                        except Exception:
                            uv = None
                        append_unique_uv(boundary_uv, uv)

            for uv in boundary_uv:
                append_unique_uv(delaunay_points_2d, uv)

            # Need at least three non-collinear points in the augmented set.
            non_collinear = False
            for i in range(len(delaunay_points_2d) - 2):
                for j in range(i + 1, len(delaunay_points_2d) - 1):
                    for k in range(j + 1, len(delaunay_points_2d)):
                        if abs(
                            orient2d(
                                delaunay_points_2d[i],
                                delaunay_points_2d[j],
                                delaunay_points_2d[k],
                            )
                        ) > tolerance * tolerance:
                            non_collinear = True
                            break
                    if non_collinear:
                        break
                if non_collinear:
                    break
            if not non_collinear:
                if not silent:
                    print(
                        "Shell.Delaunay - Error: The input sites and boundary "
                        "supports are collinear. Returning None."
                    )
                return True, None

            points_2d = [
                np.asarray(point, dtype=float)
                for point in delaunay_points_2d
            ]
            min_xy = np.min(np.asarray(points_2d), axis=0)
            max_xy = np.max(np.asarray(points_2d), axis=0)
            span = max(
                float(max_xy[0] - min_xy[0]),
                float(max_xy[1] - min_xy[1]),
                1.0,
            )
            center = 0.5 * (min_xy + max_xy)
            super_points = [
                center + np.array([-20.0 * span, -10.0 * span]),
                center + np.array([0.0, 20.0 * span]),
                center + np.array([20.0 * span, -10.0 * span]),
            ]
            work_points = points_2d + super_points
            n_points = len(points_2d)
            super_indices = (n_points, n_points + 1, n_points + 2)
            triangles_work = [super_indices]

            def incircle_contains(triangle, point):
                ia, ib, ic = triangle
                a = work_points[ia] - point
                b = work_points[ib] - point
                c = work_points[ic] - point
                determinant = (
                    float(np.dot(a, a)) * (b[0] * c[1] - b[1] * c[0])
                    - float(np.dot(b, b)) * (a[0] * c[1] - a[1] * c[0])
                    + float(np.dot(c, c)) * (a[0] * b[1] - a[1] * b[0])
                )
                orientation = orient2d(
                    work_points[ia],
                    work_points[ib],
                    work_points[ic],
                )
                eps = max(tolerance * tolerance, 1.0e-14)
                return (
                    determinant > eps
                    if orientation > 0.0
                    else determinant < -eps
                )

            for point_index in range(n_points):
                point = work_points[point_index]
                bad = [
                    triangle
                    for triangle in triangles_work
                    if incircle_contains(triangle, point)
                ]
                edge_counts = {}
                for triangle in bad:
                    for a, b in (
                        (triangle[0], triangle[1]),
                        (triangle[1], triangle[2]),
                        (triangle[2], triangle[0]),
                    ):
                        key = (a, b) if a < b else (b, a)
                        edge_counts[key] = edge_counts.get(key, 0) + 1
                triangles_work = [
                    triangle
                    for triangle in triangles_work
                    if triangle not in bad
                ]
                cavity_edges = [
                    edge
                    for edge, count in edge_counts.items()
                    if count == 1
                ]
                for a, b in cavity_edges:
                    triangle = (a, b, point_index)
                    if orient2d(
                        work_points[triangle[0]],
                        work_points[triangle[1]],
                        work_points[triangle[2]],
                    ) < 0.0:
                        triangle = (
                            triangle[1],
                            triangle[0],
                            triangle[2],
                        )
                    triangles_work.append(triangle)

            delaunay_triangles = []
            seen_triangles = set()
            for triangle in triangles_work:
                if any(index >= n_points for index in triangle):
                    continue
                if abs(
                    orient2d(
                        work_points[triangle[0]],
                        work_points[triangle[1]],
                        work_points[triangle[2]],
                    )
                ) <= tolerance * tolerance:
                    continue
                key = tuple(sorted(int(index) for index in triangle))
                if key in seen_triangles:
                    continue
                seen_triangles.add(key)
                delaunay_triangles.append(tuple(int(index) for index in triangle))

            if not delaunay_triangles:
                return True, None

            delaunay_pairs = set()
            for triangle in delaunay_triangles:
                for a, b in (
                    (triangle[0], triangle[1]),
                    (triangle[1], triangle[2]),
                    (triangle[2], triangle[0]),
                ):
                    delaunay_pairs.add((a, b) if a < b else (b, a))

            cutter_edges = []
            for a, b in sorted(delaunay_pairs):
                cutter = planar_pcurve_edge(
                    delaunay_points_2d[a],
                    delaunay_points_2d[b],
                )
                if cutter is None:
                    return True, None
                cutter_edges.append(cutter)

            split_faces = split_face_with_tools(cutter_edges)
            if not split_faces:
                return True, None
            result = assemble_faces(split_faces, sew=True)
            if result is None:
                return True, None
            debug_print(
                f"exact planar-convex Delaunay: {n_user_sites} user sites + "
                f"{len(delaunay_points_2d) - n_user_sites} boundary supports; "
                f"{len(delaunay_pairs)} edges / {len(split_faces)} faces"
            )
            return True, result

        planar_applicable, planar_result = exact_planar_convex_partition()
        if planar_applicable:
            debug_print("exact planar-convex path")
            return planar_result

        # ------------------------------------------------------------------
        # Generic intrinsic Delaunay must triangulate the complete trimmed
        # domain, not merely connect the user sites in a floating interior
        # network.  Mirror the exact planar path by augmenting the intrinsic
        # generators with the Face's existing boundary vertices.  These are
        # auxiliary domain supports only; the caller's input vertices remain
        # the semantic sites.
        #
        # Adjacent support vertices already have the exact host boundary Edge
        # between them, so record those pairs and do not reconstruct a second
        # interior geodesic for that Delaunay adjacency.
        # ------------------------------------------------------------------
        n_user_sites = len(site_points)
        delaunay_boundary_pairs = set()

        if mode == "delaunay":
            def _append_delaunay_support(point_xyz, uv):
                point_xyz = tuple(float(value) for value in point_xyz)
                uv = (float(uv[0]), float(uv[1]))
                for index, existing in enumerate(site_points):
                    try:
                        if math.dist(existing, point_xyz) <= tolerance:
                            return index
                    except Exception:
                        continue
                site_points.append(point_xyz)
                site_uv.append(uv)
                return len(site_points) - 1

            def _project_support_xyz(point_xyz):
                try:
                    point = gp_Pnt(
                        float(point_xyz[0]),
                        float(point_xyz[1]),
                        float(point_xyz[2]),
                    )
                    projector = GeomAPI_ProjectPointOnSurf(point, surface)
                    if projector.NbPoints() < 1:
                        return None
                    if float(projector.LowerDistance()) > max(tolerance, 1.0e-8):
                        return None
                    u, v = projector.LowerDistanceParameters()
                    state = classifier.Perform(gp_Pnt2d(float(u), float(v)))
                    if state not in (TopAbs_IN, TopAbs_ON):
                        return None
                    nearest = projector.NearestPoint()
                    return (
                        (float(nearest.X()), float(nearest.Y()), float(nearest.Z())),
                        (float(u), float(v)),
                    )
                except Exception:
                    return None

            def _support_index_for_vertex(vertex):
                if not isinstance(vertex, Vertex):
                    return None
                xyz = (float(vertex.x), float(vertex.y), float(vertex.z))
                projected = _project_support_xyz(xyz)
                if projected is None:
                    return None
                point_xyz, uv = projected
                return _append_delaunay_support(point_xyz, uv)

            try:
                boundary_vertices = face.Vertices() or []
            except Exception:
                boundary_vertices = []

            for boundary_vertex in boundary_vertices:
                _support_index_for_vertex(boundary_vertex)

            try:
                boundary_edges_for_support = face.Edges() or []
            except Exception:
                boundary_edges_for_support = []

            # Record the host's existing boundary segments.  Their geometry is
            # already exact and must remain the Delaunay domain boundary.
            for boundary_edge in boundary_edges_for_support:
                if not isinstance(boundary_edge, Edge):
                    continue
                try:
                    index_a = _support_index_for_vertex(boundary_edge.start)
                    index_b = _support_index_for_vertex(boundary_edge.end)
                except Exception:
                    index_a = None
                    index_b = None
                if (
                    index_a is not None
                    and index_b is not None
                    and index_a != index_b
                ):
                    delaunay_boundary_pairs.add(
                        (index_a, index_b) if index_a < index_b else (index_b, index_a)
                    )

            # A single-edge or otherwise under-constrained boundary may expose
            # fewer than three distinct topological vertices.  Add conservative
            # auxiliary samples along the exact boundary Edges so the intrinsic
            # Delaunay complex can still cover the full domain.  Consecutive
            # samples are recorded as host-boundary pairs and are not rebuilt as
            # interior cutters.
            boundary_support_indices = {
                index
                for pair in delaunay_boundary_pairs
                for index in pair
                if index >= n_user_sites
            }
            if len(boundary_support_indices) < 3:
                try:
                    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
                except Exception:
                    BRepAdaptor_Curve = None

                if BRepAdaptor_Curve is not None and boundary_edges_for_support:
                    target_samples = 16
                    edge_count = max(1, len(boundary_edges_for_support))
                    samples_per_edge = max(
                        3,
                        int(math.ceil(float(target_samples) / float(edge_count))),
                    )

                    for boundary_edge in boundary_edges_for_support:
                        if not isinstance(boundary_edge, Edge):
                            continue
                        edge_shape = getattr(boundary_edge, "shape", None)
                        if _is_null_shape(edge_shape):
                            continue
                        try:
                            adaptor = BRepAdaptor_Curve(edge_shape)
                            first = float(adaptor.FirstParameter())
                            last = float(adaptor.LastParameter())
                        except Exception:
                            continue
                        if not (math.isfinite(first) and math.isfinite(last)):
                            continue
                        span = last - first
                        if abs(span) <= 1.0e-15:
                            continue

                        edge_supports = []
                        for sample_index in range(samples_per_edge + 1):
                            parameter = first + span * (
                                float(sample_index) / float(samples_per_edge)
                            )
                            try:
                                point = adaptor.Value(float(parameter))
                                xyz = (
                                    float(point.X()),
                                    float(point.Y()),
                                    float(point.Z()),
                                )
                            except Exception:
                                continue
                            projected = _project_support_xyz(xyz)
                            if projected is None:
                                continue
                            point_xyz, uv = projected
                            support_index = _append_delaunay_support(point_xyz, uv)
                            if not edge_supports or support_index != edge_supports[-1]:
                                edge_supports.append(support_index)

                        for index_a, index_b in zip(edge_supports[:-1], edge_supports[1:]):
                            if index_a == index_b:
                                continue
                            delaunay_boundary_pairs.add(
                                (index_a, index_b)
                                if index_a < index_b
                                else (index_b, index_a)
                            )

            n_boundary_supports = max(0, len(site_points) - n_user_sites)
            debug_print(
                f"generic Delaunay domain augmentation: {n_user_sites} user sites + "
                f"{n_boundary_supports} boundary supports; "
                f"{len(delaunay_boundary_pairs)} host-boundary pairs"
            )

            if len(site_points) < 3:
                return None

        try:
            if deflection is None:
                target_deflection = float(
                    Topology._MeshDeflectionNative(occ_face, tolerance)
                )
            else:
                target_deflection = abs(float(deflection))
            if not math.isfinite(target_deflection) or target_deflection <= 0.0:
                raise ValueError

            # ``deflection`` is the finest permitted geometric approximation.
            # At most two coarser levels are used before the target. Very coarse
            # OCCT meshes are not useful for an intrinsic metric and can create
            # severe diagonal bias before the final refinement is reached.
            coarse_steps = min(max(0, maxIterations - 1), 2)
            current_deflection = target_deflection * (2.0 ** coarse_steps)
        except Exception:
            return None

        def _uv_bounds_native():
            """Return raw OCCT UV bounds without assuming a specific binding API."""
            try:
                from OCC.Core.BRepTools import breptools
                values = breptools.UVBounds(occ_face)
                if values is not None and len(values) == 4:
                    return tuple(float(value) for value in values)
            except Exception:
                pass
            try:
                from OCC.Core.BRepTools import BRepTools
                for name in ("UVBounds_s", "UVBounds"):
                    fn = getattr(BRepTools, name, None)
                    if callable(fn):
                        values = fn(occ_face)
                        if values is not None and len(values) == 4:
                            return tuple(float(value) for value in values)
            except Exception:
                pass
            return None

        def _natural_rectangular_uv_domain():
            """
            Detect an untrimmed natural rectangular parametric Face.

            For this case we deliberately avoid OCCT's unconstrained triangle
            diagonal choices. A structured UV mesh can be made exactly symmetric
            under u/v reflections and diagonal interchange when the surface and
            sites have those symmetries.
            """
            bounds = _uv_bounds_native()
            if bounds is None:
                return None
            umin, umax, vmin, vmax = bounds
            if (
                not all(math.isfinite(value) for value in bounds)
                or umax <= umin
                or vmax <= vmin
            ):
                return None

            try:
                if len(face.InternalBoundaries() or []) != 0:
                    return None
            except Exception:
                return None

            # A natural rectangular Face must classify interior samples as IN/ON
            # and points sampled on each of the four parametric limits as ON.
            try:
                for fu, fv in ((0.5, 0.5), (0.25, 0.25), (0.75, 0.75)):
                    u = umin + fu * (umax - umin)
                    v = vmin + fv * (vmax - vmin)
                    state = classifier.Perform(gp_Pnt2d(float(u), float(v)))
                    if state not in (TopAbs_IN, TopAbs_ON):
                        return None

                boundary_samples = []
                for t in (0.2, 0.5, 0.8):
                    u = umin + t * (umax - umin)
                    v = vmin + t * (vmax - vmin)
                    boundary_samples.extend(
                        [
                            (u, vmin),
                            (u, vmax),
                            (umin, v),
                            (umax, v),
                        ]
                    )
                for u, v in boundary_samples:
                    state = classifier.Perform(gp_Pnt2d(float(u), float(v)))
                    if state != TopAbs_ON:
                        return None
            except Exception:
                return None

            return bounds

        structured_bounds = _natural_rectangular_uv_domain()

        def _surface_xyz(u, v):
            p = surface.Value(float(u), float(v))
            return np.array(
                [float(p.X()), float(p.Y()), float(p.Z())],
                dtype=np.float64,
            )

        def _structured_target_divisions(bounds, allowed_error):
            """
            Choose a uniform UV resolution from geometric chordal error.

            The returned resolution is uniform in both parameter directions.
            This is intentional: an otherwise symmetric surface must not inherit
            a preferred triangulation diagonal from the mesher.
            """
            umin, umax, vmin, vmax = bounds
            divisions = 1
            max_divisions = 32
            permitted = max(float(allowed_error), float(tolerance), 1.0e-8)

            while True:
                max_error = 0.0
                for iu in range(divisions):
                    ua = umin + (umax - umin) * iu / divisions
                    ub = umin + (umax - umin) * (iu + 1) / divisions
                    um = 0.5 * (ua + ub)
                    for iv in range(divisions):
                        va = vmin + (vmax - vmin) * iv / divisions
                        vb = vmin + (vmax - vmin) * (iv + 1) / divisions
                        vm = 0.5 * (va + vb)

                        p00 = _surface_xyz(ua, va)
                        p10 = _surface_xyz(ub, va)
                        p11 = _surface_xyz(ub, vb)
                        p01 = _surface_xyz(ua, vb)
                        pc = _surface_xyz(um, vm)
                        pu0 = _surface_xyz(um, va)
                        pu1 = _surface_xyz(um, vb)
                        pv0 = _surface_xyz(ua, vm)
                        pv1 = _surface_xyz(ub, vm)

                        errors = (
                            np.linalg.norm(pc - 0.25 * (p00 + p10 + p11 + p01)),
                            np.linalg.norm(pu0 - 0.5 * (p00 + p10)),
                            np.linalg.norm(pu1 - 0.5 * (p01 + p11)),
                            np.linalg.norm(pv0 - 0.5 * (p00 + p01)),
                            np.linalg.norm(pv1 - 0.5 * (p10 + p11)),
                        )
                        max_error = max(max_error, *(float(value) for value in errors))

                if max_error <= permitted or divisions >= max_divisions:
                    return max(1, int(divisions))
                divisions *= 2

        structured_target_divisions = None
        if structured_bounds is not None:
            try:
                structured_target_divisions = _structured_target_divisions(
                    structured_bounds,
                    target_deflection,
                )
                debug_print(
                    "mesh mode=structured-uv-d4 "
                    f"target divisions={structured_target_divisions}"
                )
            except Exception:
                structured_bounds = None
                structured_target_divisions = None

        def _dedupe_axis(values, span):
            values = sorted(float(value) for value in values if math.isfinite(float(value)))
            if not values:
                return []
            eps = max(abs(float(span)) * 1.0e-12, 1.0e-12)
            result = [values[0]]
            for value in values[1:]:
                if abs(value - result[-1]) > eps:
                    result.append(value)
            return result

        def _structured_mesh(divisions):
            """
            Build a reflection/diagonal-symmetric computational mesh.

            Every parametric rectangle is split through its centre into four
            triangles, never by selecting one of its two diagonals. All site U
            and V coordinates are inserted as grid lines, so symmetric sites are
            existing mesh vertices rather than order-dependent triangle splits.
            """
            if structured_bounds is None:
                return None
            umin, umax, vmin, vmax = structured_bounds
            divisions = max(1, int(divisions))

            u_values = [
                umin + (umax - umin) * i / divisions
                for i in range(divisions + 1)
            ]
            v_values = [
                vmin + (vmax - vmin) * i / divisions
                for i in range(divisions + 1)
            ]
            u_values.extend(float(uv[0]) for uv in site_uv)
            v_values.extend(float(uv[1]) for uv in site_uv)
            u_values = _dedupe_axis(u_values, umax - umin)
            v_values = _dedupe_axis(v_values, vmax - vmin)

            points = []
            uvs = []
            corner_indices = {}

            for iu, u in enumerate(u_values):
                for iv, v in enumerate(v_values):
                    p = _surface_xyz(u, v)
                    corner_indices[(iu, iv)] = len(points)
                    points.append(p.tolist())
                    uvs.append([float(u), float(v)])

            triangles = []
            reversed_face = occ_face.Orientation() == TopAbs_REVERSED

            def add_triangle(a, b, c):
                if reversed_face:
                    b, c = c, b
                triangles.append([int(a), int(b), int(c)])

            for iu in range(len(u_values) - 1):
                ua = u_values[iu]
                ub = u_values[iu + 1]
                for iv in range(len(v_values) - 1):
                    va = v_values[iv]
                    vb = v_values[iv + 1]

                    c00 = corner_indices[(iu, iv)]
                    c10 = corner_indices[(iu + 1, iv)]
                    c11 = corner_indices[(iu + 1, iv + 1)]
                    c01 = corner_indices[(iu, iv + 1)]

                    um = 0.5 * (ua + ub)
                    vm = 0.5 * (va + vb)
                    pc = _surface_xyz(um, vm)
                    center = len(points)
                    points.append(pc.tolist())
                    uvs.append([float(um), float(vm)])

                    add_triangle(c00, c10, center)
                    add_triangle(c10, c11, center)
                    add_triangle(c11, c01, center)
                    add_triangle(c01, c00, center)

            return points, uvs, triangles

        # ------------------------------------------------------------------
        # Computational triangulation of the trimmed Face.
        # ------------------------------------------------------------------
        def triangulation_at(defl):
            # For natural rectangular parametric Faces, use the structured mesh
            # above. Scale its uniform resolution consistently with the current
            # refinement level while never exceeding the target resolution.
            if structured_bounds is not None and structured_target_divisions is not None:
                ratio = max(1.0, float(defl) / float(target_deflection))
                divisions = max(
                    1,
                    int(math.ceil(float(structured_target_divisions) / ratio)),
                )
                debug_print(
                    f"tessellation: structured-uv-d4 divisions={divisions}"
                )
                return _structured_mesh(divisions)

            # General trimmed-Face fallback: retain OCCT triangulation. Curved
            # boundaries and holes require a conforming trim-aware mesh.
            try:
                debug_print("tessellation: creating serial BRepMesh_IncrementalMesh")
                mesher = BRepMesh_IncrementalMesh(
                    occ_face,
                    float(defl),
                    False,
                    0.5,
                    False,
                )
                try:
                    mesher.Perform()
                except Exception:
                    pass
                debug_print("tessellation: mesher returned")
                if hasattr(mesher, "IsDone") and not mesher.IsDone():
                    return None

                location = TopLoc_Location()
                debug_print("tessellation: retrieving Poly_Triangulation")
                triangulation = BRep_Tool.Triangulation(occ_face, location)
                debug_print("tessellation: Poly_Triangulation retrieved")
                if triangulation is None or triangulation.NbTriangles() < 1:
                    return None

                transform = location.Transformation()
                identity = location.IsIdentity()
                points = []
                uvs = []
                has_uv = (
                    bool(triangulation.HasUVNodes())
                    if hasattr(triangulation, "HasUVNodes")
                    else False
                )

                for index in range(1, triangulation.NbNodes() + 1):
                    q = triangulation.Node(index)
                    p = gp_Pnt(q.X(), q.Y(), q.Z())
                    if not identity:
                        p.Transform(transform)
                    points.append([float(p.X()), float(p.Y()), float(p.Z())])

                    if has_uv:
                        uv = triangulation.UVNode(index)
                        uvs.append([float(uv.X()), float(uv.Y())])
                    else:
                        projection = GeomAPI_ProjectPointOnSurf(p, surface)
                        if projection.NbPoints() < 1:
                            return None
                        u, v = projection.LowerDistanceParameters()
                        uvs.append([float(u), float(v)])

                triangles = []
                reversed_face = occ_face.Orientation() == TopAbs_REVERSED
                for index in range(1, triangulation.NbTriangles() + 1):
                    a, b, c = triangulation.Triangle(index).Get()
                    if reversed_face:
                        b, c = c, b
                    triangles.append([int(a - 1), int(b - 1), int(c - 1)])

                return points, uvs, triangles
            except Exception:
                return None

        def barycentric_2d(point, a, b, c):
            px, py = point
            ax, ay = a
            bx, by = b
            cx, cy = c
            denominator = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
            if abs(denominator) <= 1.0e-18:
                return None
            l0 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / denominator
            l1 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / denominator
            return l0, l1, 1.0 - l0 - l1

        def insert_site(points, uvs, triangles, p, uv):
            for index, q in enumerate(points):
                if math.dist(p, q) <= tolerance:
                    return index

            hit = None
            bary = None
            eps = 1.0e-9
            for triangle_index, triangle in enumerate(triangles):
                coordinates = barycentric_2d(
                    uv,
                    uvs[triangle[0]],
                    uvs[triangle[1]],
                    uvs[triangle[2]],
                )
                if (
                    coordinates is not None
                    and min(coordinates) >= -eps
                    and max(coordinates) <= 1.0 + eps
                ):
                    hit = triangle_index
                    bary = coordinates
                    break

            if hit is None:
                return None

            new_index = len(points)
            points.append(list(p))
            uvs.append(list(uv))
            original = triangles[hit]
            near_zero = [k for k, value in enumerate(bary) if abs(value) <= 1.0e-8]

            if not near_zero:
                a, b, c = original
                triangles[hit : hit + 1] = [
                    [a, b, new_index],
                    [b, c, new_index],
                    [c, a, new_index],
                ]
                return new_index

            if len(near_zero) >= 2:
                return original[max(range(3), key=lambda k: bary[k])]

            # Site lies on an existing mesh edge: split every incident triangle.
            k = near_zero[0]
            e0 = original[(k + 1) % 3]
            e1 = original[(k + 2) % 3]
            affected = [
                index
                for index, triangle in enumerate(triangles)
                if e0 in triangle and e1 in triangle
            ]
            replacements = []
            for index in affected:
                triangle = triangles[index]
                third = next(x for x in triangle if x not in (e0, e1))
                pos0 = triangle.index(e0)
                pos1 = triangle.index(e1)
                if (pos0 + 1) % 3 == pos1:
                    new_triangles = [
                        [e0, new_index, third],
                        [new_index, e1, third],
                    ]
                else:
                    new_triangles = [
                        [e1, new_index, third],
                        [new_index, e0, third],
                    ]
                replacements.append((index, new_triangles))

            for index, new_triangles in sorted(replacements, reverse=True):
                triangles[index : index + 1] = new_triangles
            return new_index

        # ------------------------------------------------------------------
        # Intrinsic Fast Marching Method (FMM) on the computational triangle
        # mesh.
        #
        # This is a continuous triangle update, not graph/edge Dijkstra. Each
        # tentative value uses the upwind solution of |grad T| = 1 across a
        # triangle when both opposite vertices are already accepted. If the
        # resulting characteristic does not enter through the accepted edge,
        # the local causal fallback is the appropriate one-edge update.
        #
        # Distances are therefore non-negative and the generating source is
        # exactly zero by construction. Refining the computational surface mesh
        # improves the approximation to the smooth-surface intrinsic metric.
        # ------------------------------------------------------------------
        def fast_marching_distance(points, triangles, source_index):
            import heapq

            points = np.asarray(points, dtype=np.float64)
            triangles = np.asarray(triangles, dtype=np.int64)
            n_vertices = int(points.shape[0])
            source_index = int(source_index)

            if (
                source_index < 0
                or source_index >= n_vertices
                or triangles.ndim != 2
                or triangles.shape[1] != 3
            ):
                return None

            FAR = 0
            TRIAL = 1
            ACCEPTED = 2

            state = np.zeros(n_vertices, dtype=np.int8)
            distance = np.full(n_vertices, np.inf, dtype=np.float64)
            incident_triangles = [[] for _ in range(n_vertices)]
            neighbours = [set() for _ in range(n_vertices)]

            for triangle_index, triangle in enumerate(triangles):
                try:
                    i, j, k = [int(value) for value in triangle]
                except Exception:
                    return None
                if (
                    min(i, j, k) < 0
                    or max(i, j, k) >= n_vertices
                    or len({i, j, k}) < 3
                ):
                    return None

                pi, pj, pk = points[i], points[j], points[k]
                double_area = float(np.linalg.norm(np.cross(pj - pi, pk - pi)))
                if not math.isfinite(double_area) or double_area <= 1.0e-15:
                    return None

                incident_triangles[i].append(triangle_index)
                incident_triangles[j].append(triangle_index)
                incident_triangles[k].append(triangle_index)
                neighbours[i].update((j, k))
                neighbours[j].update((i, k))
                neighbours[k].update((i, j))

            geometric_epsilon = max(float(tolerance) * 1.0e-6, 1.0e-12)
            value_epsilon = max(float(tolerance) * 1.0e-8, 1.0e-12)

            def one_triangle_candidate(vertex_index, triangle):
                vertex_index = int(vertex_index)
                other = [
                    int(value)
                    for value in triangle
                    if int(value) != vertex_index
                ]
                if len(other) != 2:
                    return float("inf")

                accepted = [
                    index
                    for index in other
                    if state[index] == ACCEPTED
                    and math.isfinite(float(distance[index]))
                ]

                candidate = float("inf")
                point_c = points[vertex_index]

                # One-sided causal updates are always admissible.
                for index_a in accepted:
                    length = float(np.linalg.norm(point_c - points[index_a]))
                    if math.isfinite(length) and length > geometric_epsilon:
                        candidate = min(
                            candidate,
                            float(distance[index_a]) + length,
                        )

                if len(accepted) != 2:
                    return candidate

                index_a, index_b = accepted
                point_a = points[index_a]
                point_b = points[index_b]

                edge_ab = point_b - point_a
                length_ab = float(np.linalg.norm(edge_ab))
                if not math.isfinite(length_ab) or length_ab <= geometric_epsilon:
                    return candidate

                axis_x = edge_ab / length_ab
                vector_ac = point_c - point_a
                coordinate_x = float(np.dot(vector_ac, axis_x))
                perpendicular = vector_ac - coordinate_x * axis_x
                coordinate_y = float(np.linalg.norm(perpendicular))
                if not math.isfinite(coordinate_y) or coordinate_y <= geometric_epsilon:
                    return candidate

                value_a = float(distance[index_a])
                value_b = float(distance[index_b])
                gradient_x = (value_b - value_a) / length_ab

                # No real unit-gradient solution crosses edge AB if the tangential
                # derivative already has magnitude >= 1.
                if not math.isfinite(gradient_x) or abs(gradient_x) >= 1.0:
                    return candidate

                gradient_y_squared = 1.0 - gradient_x * gradient_x
                if gradient_y_squared <= 0.0:
                    return candidate
                gradient_y = math.sqrt(gradient_y_squared)

                two_point = (
                    value_a
                    + gradient_x * coordinate_x
                    + gradient_y * coordinate_y
                )

                # The back-traced characteristic from C must intersect the
                # accepted edge AB itself. Otherwise this two-point update is
                # non-causal for this triangle.
                foot = (
                    coordinate_x
                    - coordinate_y * gradient_x / gradient_y
                )
                foot_tolerance = max(
                    geometric_epsilon,
                    length_ab * 1.0e-10,
                )

                if (
                    math.isfinite(two_point)
                    and -foot_tolerance <= foot <= length_ab + foot_tolerance
                    and two_point >= max(value_a, value_b) - value_epsilon
                ):
                    candidate = min(candidate, two_point)

                return candidate

            def update_vertex(vertex_index, heap):
                vertex_index = int(vertex_index)
                if state[vertex_index] == ACCEPTED:
                    return

                best = float(distance[vertex_index])
                for triangle_index in incident_triangles[vertex_index]:
                    candidate = one_triangle_candidate(
                        vertex_index,
                        triangles[triangle_index],
                    )
                    if candidate < best:
                        best = candidate

                if not math.isfinite(best):
                    return

                if best < float(distance[vertex_index]) - value_epsilon:
                    distance[vertex_index] = best
                    state[vertex_index] = TRIAL
                    heapq.heappush(heap, (best, vertex_index))
                elif state[vertex_index] == FAR:
                    # The first finite candidate can differ from infinity without
                    # needing a numerical comparison against it.
                    distance[vertex_index] = best
                    state[vertex_index] = TRIAL
                    heapq.heappush(heap, (best, vertex_index))

            distance[source_index] = 0.0
            state[source_index] = ACCEPTED
            heap = []

            for neighbour in neighbours[source_index]:
                update_vertex(neighbour, heap)

            accepted_count = 1
            while heap:
                trial_value, vertex_index = heapq.heappop(heap)
                vertex_index = int(vertex_index)

                if state[vertex_index] == ACCEPTED:
                    continue
                if trial_value > float(distance[vertex_index]) + value_epsilon:
                    continue

                state[vertex_index] = ACCEPTED
                accepted_count += 1

                touched = set()
                for triangle_index in incident_triangles[vertex_index]:
                    touched.update(
                        int(value)
                        for value in triangles[triangle_index]
                    )
                for candidate_vertex in touched:
                    if state[candidate_vertex] != ACCEPTED:
                        update_vertex(candidate_vertex, heap)

            if accepted_count != n_vertices or np.any(~np.isfinite(distance)):
                return None

            # Numerical guard only; the marching construction itself is
            # non-negative and fixes the source exactly at zero.
            tiny_negative = (distance < 0.0) & (distance >= -value_epsilon)
            distance[tiny_negative] = 0.0
            if np.any(distance < 0.0):
                return None
            distance[source_index] = 0.0
            return distance

        def voronoi_segments(points, uvs, triangles, fields):
            """
            Extract the exact lower-envelope interfaces of the piecewise-linear
            distance fields on each computational triangle.

            For every site pair i,j, the equality d_i=d_j is a line in barycentric
            coordinates on a triangle. Its segment inside the triangle is clipped
            further by d_i<=d_k for every other site k. The surviving interval is
            therefore the actual Voronoi boundary of the PL fields; no inference
            from vertex labels and no centroid fallback are used.
            """
            points = np.asarray(points, dtype=float)
            uvs = np.asarray(uvs, dtype=float)
            triangles = np.asarray(triangles, dtype=np.int64)
            fields = np.asarray(fields, dtype=float)

            n_sites = int(fields.shape[0])
            segments = []
            clouds = {}
            equality_eps = max(1.0e-12, tolerance * 1.0e-6)

            bary_vertices = (
                np.array([1.0, 0.0, 0.0], dtype=float),
                np.array([0.0, 1.0, 0.0], dtype=float),
                np.array([0.0, 0.0, 1.0], dtype=float),
            )
            local_edges = ((0, 1), (1, 2), (2, 0))

            def add_unique(items, bary):
                for existing in items:
                    if float(np.linalg.norm(existing - bary)) <= 1.0e-10:
                        return
                items.append(bary)

            for triangle in triangles:
                tri = np.asarray(triangle, dtype=np.int64)
                tri_points = points[tri]
                tri_uvs = uvs[tri]

                for i in range(n_sites - 1):
                    values_i = fields[i, tri]
                    for j in range(i + 1, n_sites):
                        difference = values_i - fields[j, tri]

                        # Coincident affine fields over the complete triangle are
                        # a metric degeneracy, not a finite Voronoi interface.
                        if float(np.max(np.abs(difference))) <= equality_eps:
                            continue

                        intersections = []

                        # Equality at a triangle vertex.
                        for local_index in range(3):
                            if abs(float(difference[local_index])) <= equality_eps:
                                add_unique(
                                    intersections,
                                    bary_vertices[local_index].copy(),
                                )

                        # Equality crossing a triangle edge.
                        for a_local, b_local in local_edges:
                            fa = float(difference[a_local])
                            fb = float(difference[b_local])
                            if (
                                fa > equality_eps and fb > equality_eps
                            ) or (
                                fa < -equality_eps and fb < -equality_eps
                            ):
                                continue
                            denominator = fa - fb
                            if abs(denominator) <= 1.0e-18:
                                continue
                            parameter = fa / denominator
                            if parameter < -1.0e-10 or parameter > 1.0 + 1.0e-10:
                                continue
                            parameter = max(0.0, min(1.0, parameter))
                            bary = (
                                (1.0 - parameter) * bary_vertices[a_local]
                                + parameter * bary_vertices[b_local]
                            )
                            add_unique(intersections, bary)

                        if len(intersections) < 2:
                            continue

                        # In vertex-degenerate cases there can be >2 candidates.
                        # Use the farthest pair, which spans the equality segment.
                        if len(intersections) > 2:
                            best_pair = None
                            best_distance = -1.0
                            for a_index in range(len(intersections) - 1):
                                for b_index in range(a_index + 1, len(intersections)):
                                    pa = (
                                        intersections[a_index][0] * tri_points[0]
                                        + intersections[a_index][1] * tri_points[1]
                                        + intersections[a_index][2] * tri_points[2]
                                    )
                                    pb = (
                                        intersections[b_index][0] * tri_points[0]
                                        + intersections[b_index][1] * tri_points[1]
                                        + intersections[b_index][2] * tri_points[2]
                                    )
                                    distance = float(np.linalg.norm(pb - pa))
                                    if distance > best_distance:
                                        best_distance = distance
                                        best_pair = (
                                            intersections[a_index],
                                            intersections[b_index],
                                        )
                            if best_pair is None:
                                continue
                            bary0, bary1 = best_pair
                        else:
                            bary0, bary1 = intersections[0], intersections[1]

                        # Restrict the pairwise equality segment to the lower
                        # envelope: d_i=d_j must also be <= every other d_k.
                        t_min = 0.0
                        t_max = 1.0
                        valid = True

                        for k in range(n_sites):
                            if k == i or k == j:
                                continue

                            diff_ik = fields[i, tri] - fields[k, tri]
                            h0 = float(np.dot(bary0, diff_ik))
                            h1 = float(np.dot(bary1, diff_ik))

                            inside0 = h0 <= equality_eps
                            inside1 = h1 <= equality_eps

                            if inside0 and inside1:
                                continue
                            if not inside0 and not inside1:
                                valid = False
                                break

                            denominator = h0 - h1
                            if abs(denominator) <= 1.0e-18:
                                valid = False
                                break

                            crossing_t = h0 / denominator
                            crossing_t = max(0.0, min(1.0, crossing_t))

                            if not inside0:
                                t_min = max(t_min, crossing_t)
                            else:
                                t_max = min(t_max, crossing_t)

                            if t_max - t_min <= 1.0e-12:
                                valid = False
                                break

                        if not valid:
                            continue

                        clipped0 = bary0 + t_min * (bary1 - bary0)
                        clipped1 = bary0 + t_max * (bary1 - bary0)

                        point0 = (
                            clipped0[0] * tri_points[0]
                            + clipped0[1] * tri_points[1]
                            + clipped0[2] * tri_points[2]
                        )
                        point1 = (
                            clipped1[0] * tri_points[0]
                            + clipped1[1] * tri_points[1]
                            + clipped1[2] * tri_points[2]
                        )

                        if float(np.linalg.norm(point1 - point0)) <= tolerance:
                            continue

                        uv0 = (
                            clipped0[0] * tri_uvs[0]
                            + clipped0[1] * tri_uvs[1]
                            + clipped0[2] * tri_uvs[2]
                        )
                        uv1 = (
                            clipped1[0] * tri_uvs[0]
                            + clipped1[1] * tri_uvs[1]
                            + clipped1[2] * tri_uvs[2]
                        )

                        pair = (i, j)
                        segments.append(
                            (
                                pair,
                                np.asarray(uv0, dtype=float),
                                np.asarray(uv1, dtype=float),
                                np.asarray(point0, dtype=float),
                                np.asarray(point1, dtype=float),
                            )
                        )
                        clouds.setdefault(pair, []).extend(
                            [
                                np.asarray(point0, dtype=float),
                                np.asarray(point1, dtype=float),
                            ]
                        )

            return segments, clouds

        def hausdorff(a, b):
            if not a or not b:
                return float("inf")
            array_a = np.asarray(a, dtype=float)
            array_b = np.asarray(b, dtype=float)
            distance_a = max(
                float(np.min(np.linalg.norm(array_b - point, axis=1)))
                for point in array_a
            )
            distance_b = max(
                float(np.min(np.linalg.norm(array_a - point, axis=1)))
                for point in array_b
            )
            return max(distance_a, distance_b)

        # ------------------------------------------------------------------
        # Continuous steepest-descent tracing of a piecewise-linear Fast Marching
        # distance field. This is used only after Voronoi adjacency establishes
        # which site pairs are Delaunay neighbours. No graph shortest path is
        # used.
        # ------------------------------------------------------------------
        def trace_distance_path(
            points,
            uvs,
            triangles,
            field,
            source_index,
            target_index,
        ):
            points = np.asarray(points, dtype=float)
            uvs = np.asarray(uvs, dtype=float)
            triangles = np.asarray(triangles, dtype=np.int64)
            source_index = int(source_index)
            target_index = int(target_index)

            vertex_triangles = [[] for _ in range(len(points))]
            vertex_neighbours = [set() for _ in range(len(points))]
            edge_triangles = {}
            triangle_geometry = []

            for triangle_index, triangle in enumerate(triangles):
                i, j, k = [int(value) for value in triangle]
                for vertex_index in (i, j, k):
                    vertex_triangles[vertex_index].append(triangle_index)
                for a, b in ((i, j), (j, k), (k, i)):
                    edge_key = tuple(sorted((a, b)))
                    edge_triangles.setdefault(edge_key, []).append(triangle_index)
                    vertex_neighbours[a].add(b)
                    vertex_neighbours[b].add(a)

                pi, pj, pk = points[i], points[j], points[k]
                normal = np.cross(pj - pi, pk - pi)
                normal_squared = float(np.dot(normal, normal))
                if normal_squared <= 1.0e-30:
                    triangle_geometry.append(None)
                    continue

                grad_i = np.cross(normal, pk - pj) / normal_squared
                grad_j = np.cross(normal, pi - pk) / normal_squared
                grad_k = np.cross(normal, pj - pi) / normal_squared
                gradient = (
                    field[i] * grad_i
                    + field[j] * grad_j
                    + field[k] * grad_k
                )
                triangle_geometry.append((grad_i, grad_j, grad_k, gradient))

            source_uv = uvs[source_index]
            target_uv = uvs[target_index]
            if source_index == target_index:
                return [target_uv.copy()]

            def barycentric_3d(point, triangle):
                i, j, k = [int(value) for value in triangle]
                a, b, c = points[i], points[j], points[k]
                v0 = b - a
                v1 = c - a
                v2 = point - a
                d00 = float(np.dot(v0, v0))
                d01 = float(np.dot(v0, v1))
                d11 = float(np.dot(v1, v1))
                d20 = float(np.dot(v2, v0))
                d21 = float(np.dot(v2, v1))
                denominator = d00 * d11 - d01 * d01
                if abs(denominator) <= 1.0e-30:
                    return None
                b1 = (d11 * d20 - d01 * d21) / denominator
                b2 = (d00 * d21 - d01 * d20) / denominator
                return np.array([1.0 - b1 - b2, b1, b2], dtype=float)

            def entering_triangle(triangle_index, point):
                geometry = triangle_geometry[triangle_index]
                if geometry is None:
                    return None
                triangle = triangles[triangle_index]
                bary = barycentric_3d(point, triangle)
                if bary is None:
                    return None

                direction = -np.asarray(geometry[3], dtype=float)
                magnitude = float(np.linalg.norm(direction))
                if magnitude <= 1.0e-15:
                    return None
                direction /= magnitude

                derivatives = np.array(
                    [
                        float(np.dot(geometry[0], direction)),
                        float(np.dot(geometry[1], direction)),
                        float(np.dot(geometry[2], direction)),
                    ],
                    dtype=float,
                )

                zero_indices = np.where(bary <= 1.0e-8)[0]
                if any(derivatives[index] < -1.0e-9 for index in zero_indices):
                    return None

                positive_steps = []
                for local_index in range(3):
                    derivative = derivatives[local_index]
                    if derivative < -1.0e-12:
                        step = -bary[local_index] / derivative
                        if step > 1.0e-10 and math.isfinite(step):
                            positive_steps.append(step)
                if not positive_steps:
                    return None

                return magnitude, direction, bary, derivatives

            def choose_from_vertex(vertex_index):
                point = points[vertex_index]
                candidates = []
                for triangle_index in vertex_triangles[vertex_index]:
                    entry = entering_triangle(triangle_index, point)
                    if entry is not None:
                        candidates.append((entry[0], triangle_index))
                if candidates:
                    return "triangle", max(candidates, key=lambda item: item[0])[1]

                lower_neighbours = [
                    neighbour
                    for neighbour in vertex_neighbours[vertex_index]
                    if field[neighbour] < field[vertex_index] - 1.0e-12
                ]
                if lower_neighbours:
                    return "vertex", min(
                        lower_neighbours,
                        key=lambda index: field[index],
                    )
                return None

            current_point = points[target_index].copy()
            state = ("vertex", target_index)
            path_uv = [target_uv.copy()]
            visited = set()
            max_steps = max(100, 10 * len(triangles))

            for _ in range(max_steps):
                if state[0] == "vertex":
                    vertex_index = int(state[1])
                    if vertex_index == source_index:
                        if np.linalg.norm(path_uv[-1] - source_uv) > 1.0e-12:
                            path_uv.append(source_uv.copy())
                        return path_uv

                    next_state = choose_from_vertex(vertex_index)
                    if next_state is None:
                        return None
                    if next_state[0] == "vertex":
                        next_vertex = int(next_state[1])
                        current_point = points[next_vertex].copy()
                        if np.linalg.norm(path_uv[-1] - uvs[next_vertex]) > 1.0e-12:
                            path_uv.append(uvs[next_vertex].copy())
                        state = ("vertex", next_vertex)
                        continue
                    state = next_state

                triangle_index = int(state[1])
                triangle = triangles[triangle_index]
                if source_index in triangle:
                    if np.linalg.norm(path_uv[-1] - source_uv) > 1.0e-12:
                        path_uv.append(source_uv.copy())
                    return path_uv

                entry = entering_triangle(triangle_index, current_point)
                if entry is None:
                    return None
                _, direction, bary, derivatives = entry

                candidates = []
                for local_index in range(3):
                    derivative = derivatives[local_index]
                    if derivative < -1.0e-12:
                        step = -bary[local_index] / derivative
                        if step > 1.0e-10 and math.isfinite(step):
                            candidates.append((step, local_index))
                if not candidates:
                    return None

                step, vanished = min(candidates, key=lambda item: item[0])
                next_point = current_point + step * direction
                next_bary = bary + step * derivatives
                next_bary[np.abs(next_bary) < 1.0e-10] = 0.0
                next_bary = np.maximum(next_bary, 0.0)
                total = float(np.sum(next_bary))
                if total <= 1.0e-15:
                    return None
                next_bary /= total

                i, j, k = [int(value) for value in triangle]
                next_uv = (
                    next_bary[0] * uvs[i]
                    + next_bary[1] * uvs[j]
                    + next_bary[2] * uvs[k]
                )
                if np.linalg.norm(path_uv[-1] - next_uv) > 1.0e-12:
                    path_uv.append(next_uv.copy())

                edge_vertices = [
                    int(triangle[index])
                    for index in range(3)
                    if index != vanished
                ]
                edge_key = tuple(sorted(edge_vertices))
                current_point = next_point

                # Across a PL distance field, gradients are discontinuous at a
                # mesh edge. Prefer a neighbouring triangle whose descent vector
                # points into its interior. If neither side admits an inward
                # descent direction, the steepest path follows the edge ridge
                # toward its lower-distance endpoint.
                adjacent_candidates = []
                for neighbour_triangle in edge_triangles.get(edge_key, []):
                    neighbour_entry = entering_triangle(
                        neighbour_triangle,
                        current_point,
                    )
                    if neighbour_entry is not None:
                        adjacent_candidates.append(
                            (neighbour_entry[0], neighbour_triangle)
                        )

                if adjacent_candidates:
                    next_triangle = max(
                        adjacent_candidates,
                        key=lambda item: item[0],
                    )[1]
                    marker = (
                        round(float(current_point[0]), 10),
                        round(float(current_point[1]), 10),
                        round(float(current_point[2]), 10),
                        int(next_triangle),
                    )
                    if marker in visited:
                        return None
                    visited.add(marker)
                    state = ("triangle", next_triangle)
                    continue

                current_value = float(
                    next_bary[0] * field[i]
                    + next_bary[1] * field[j]
                    + next_bary[2] * field[k]
                )
                lower_vertex = min(edge_vertices, key=lambda index: field[index])
                if field[lower_vertex] >= current_value - 1.0e-10:
                    return None
                current_point = points[lower_vertex].copy()
                if np.linalg.norm(path_uv[-1] - uvs[lower_vertex]) > 1.0e-12:
                    path_uv.append(uvs[lower_vertex].copy())
                state = ("vertex", lower_vertex)

            return None

        final = None
        previous_clouds = None
        previous_adjacency = None
        converged = False

        for iteration_index in range(maxIterations):
            mesh_kind = (
                "structured-uv-d4"
                if structured_bounds is not None
                else "serial-occt"
            )
            debug_print(
                f"iteration {iteration_index + 1}/{maxIterations}: {mesh_kind} tessellation "
                f"(deflection={current_deflection:g})"
            )
            mesh = triangulation_at(current_deflection)
            debug_print(f"iteration {iteration_index + 1}: tessellation complete")
            if mesh is None:
                return None
            points, uvs, triangles = mesh
            debug_print(
                f"iteration {iteration_index + 1}: computational mesh has "
                f"{len(points)} vertices / {len(triangles)} triangles"
            )

            source_indices = []
            for point, uv in zip(site_points, site_uv):
                source_index = insert_site(points, uvs, triangles, point, uv)
                if source_index is None:
                    if not silent:
                        print(
                            "Shell intrinsic geodesics - Error: Could not insert "
                            "a site into the computational surface mesh. Returning None."
                        )
                    return None
                source_indices.append(source_index)

            fields = []
            debug_print(
                f"iteration {iteration_index + 1}: solving intrinsic Fast Marching distance fields"
            )
            for source_index in source_indices:
                field = fast_marching_distance(
                    points,
                    triangles,
                    source_index,
                )
                if field is None:
                    if not silent:
                        print(
                            "Shell intrinsic geodesics - Error: Fast Marching "
                            "distance propagation failed. Returning None."
                        )
                    return None
                fields.append(field)
            fields = np.vstack(fields)
            debug_print(
                f"iteration {iteration_index + 1}: Fast Marching distance fields complete"
            )

            # A geodesic distance field must attain its minimum at its own
            # generating site. Check this before extracting any Voronoi boundary.
            # This invariant distinguishes a numerical heat-solver failure from a
            # later curve-reconstruction/splitting failure.
            ownership_eps = max(float(tolerance) * 0.1, 1.0e-8)
            for site_index, source_index in enumerate(source_indices):
                values = np.asarray(fields[:, int(source_index)], dtype=float)
                own_value = float(values[site_index])
                best_index = int(np.argmin(values))
                best_value = float(values[best_index])
                if best_index != site_index and best_value < own_value - ownership_eps:
                    if not silent:
                        print(
                            f"Shell.{mode.capitalize()} - Error: Intrinsic distance "
                            f"field violates site ownership at site {site_index}; "
                            f"site {best_index} has a smaller distance there. "
                            "Returning None."
                        )
                    return None

            segments, clouds = voronoi_segments(points, uvs, triangles, fields)
            debug_print(
                f"iteration {iteration_index + 1}: Voronoi interfaces complete "
                f"({len(segments)} segments)"
            )
            adjacency = set(clouds.keys())
            final = (
                np.asarray(points, dtype=np.float64),
                np.asarray(triangles, dtype=np.int64),
                np.asarray(uvs, dtype=np.float64),
                source_indices,
                fields,
                segments,
                clouds,
            )

            if previous_clouds is not None and adjacency == previous_adjacency:
                displacement = 0.0
                for pair in adjacency:
                    displacement = max(
                        displacement,
                        hausdorff(previous_clouds[pair], clouds[pair]),
                    )
                if displacement <= convergence:
                    converged = True
                    break

            previous_clouds = clouds
            previous_adjacency = adjacency

            # Refine toward, but never beyond, the user-requested target.
            if current_deflection <= target_deflection * (1.0 + 1.0e-12):
                break
            next_deflection = max(target_deflection, current_deflection * 0.5)
            if next_deflection >= current_deflection * (1.0 - 1.0e-12):
                break
            current_deflection = next_deflection

        if final is None:
            return None
        if not converged and maxIterations > 1 and not silent:
            print(
                f"Shell.{mode.capitalize()} - Warning: The intrinsic partition "
                "did not meet the requested convergence criterion before "
                "refinement stopped; returning the finest computed result."
            )

        (
            points_array,
            triangles_array,
            uvs_array,
            source_indices,
            fields,
            segments,
            clouds,
        ) = final

        def _build_curves3d(edge):
            """Ensure a surface p-curve Edge also carries a usable 3D curve."""
            if edge is None:
                return None
            built = False
            try:
                from OCC.Core.BRepLib import breplib
                fn = getattr(breplib, "BuildCurves3d", None)
                if callable(fn):
                    fn(edge)
                    built = True
            except Exception:
                pass
            if not built:
                try:
                    from OCC.Core.BRepLib import BRepLib
                    for name in ("BuildCurves3d_s", "BuildCurves3d"):
                        fn = getattr(BRepLib, name, None)
                        if callable(fn):
                            fn(edge)
                            built = True
                            break
                except Exception:
                    pass
            return edge

        def pcurve_edge(uv0, uv1):
            """Create one straight p-curve Edge on the exact supporting surface."""
            if math.hypot(
                float(uv1[0] - uv0[0]),
                float(uv1[1] - uv0[1]),
            ) <= 1.0e-14:
                return None
            try:
                curve = GCE2d_MakeSegment(
                    gp_Pnt2d(float(uv0[0]), float(uv0[1])),
                    gp_Pnt2d(float(uv1[0]), float(uv1[1])),
                ).Value()
                maker = BRepBuilderAPI_MakeEdge(curve, surface)
                if hasattr(maker, "IsDone") and not maker.IsDone():
                    return None
                return _build_curves3d(maker.Edge())
            except Exception:
                return None

        def pcurve_polyline_bspline(uv_points):
            """
            Build one degree-1 2D B-spline through an ordered UV chain and lift
            it onto the exact supporting surface as a single OCCT Edge.

            Degree 1 is intentional. The intrinsic computation yields a
            piecewise-linear approximation on the computational triangulation.
            A higher-degree interpolating spline can overshoot between samples,
            cross a generating site or another bisector, and therefore change the
            Voronoi topology. A degree-1 B-spline preserves the computed chain
            exactly while still collapsing the complete chain to one topological
            Edge.
            """
            clean = []
            for uv in uv_points or []:
                uv = np.asarray(uv, dtype=float)
                if uv.shape[0] < 2 or np.any(~np.isfinite(uv[:2])):
                    continue
                uv = uv[:2]
                if not clean or float(np.linalg.norm(uv - clean[-1])) > 1.0e-12:
                    clean.append(uv)
            if len(clean) < 2:
                return None
            if len(clean) == 2:
                return pcurve_edge(clean[0], clean[1])

            try:
                count = len(clean)
                poles = TColgp_Array1OfPnt2d(1, count)
                for index, uv in enumerate(clean, start=1):
                    poles.SetValue(
                        index,
                        gp_Pnt2d(float(uv[0]), float(uv[1])),
                    )

                # Degree-1 open clamped B-spline. One distinct knot per pole;
                # end multiplicities are degree+1 (=2), interior multiplicities
                # are 1. Chord-length parameters improve numerical conditioning
                # without changing the piecewise-linear image of the curve.
                parameters = [0.0]
                for a, b in zip(clean[:-1], clean[1:]):
                    step = float(np.linalg.norm(b - a))
                    parameters.append(parameters[-1] + max(step, 1.0e-12))
                total = parameters[-1]
                if total <= 1.0e-15:
                    return None
                parameters = [value / total for value in parameters]

                knots = TColStd_Array1OfReal(1, count)
                multiplicities = TColStd_Array1OfInteger(1, count)
                for index, value in enumerate(parameters, start=1):
                    knots.SetValue(index, float(value))
                    multiplicities.SetValue(
                        index,
                        2 if index in (1, count) else 1,
                    )

                curve = Geom2d_BSplineCurve(
                    poles,
                    knots,
                    multiplicities,
                    1,
                    False,
                )
                maker = BRepBuilderAPI_MakeEdge(curve, surface)
                if hasattr(maker, "IsDone") and not maker.IsDone():
                    return None
                return _build_curves3d(maker.Edge())
            except Exception:
                return None

        def _ordered_voronoi_chains(raw_segments):
            """
            Join triangle-local Voronoi pieces into maximal continuous chains.
            Each returned item is ``(site_pair, [uv...])``. Endpoint matching is
            performed in 3D so UV anisotropy or periodic parameter scales do not
            affect connectivity.
            """
            grouped = {}
            for pair, uv0, uv1, point0, point1 in raw_segments:
                grouped.setdefault(tuple(pair), []).append(
                    [
                        np.asarray(uv0, dtype=float),
                        np.asarray(uv1, dtype=float),
                        np.asarray(point0, dtype=float),
                        np.asarray(point1, dtype=float),
                    ]
                )

            join_tol = max(float(tolerance) * 2.0, 1.0e-8)
            chains = []

            def close3(a, b):
                return float(np.linalg.norm(a - b)) <= join_tol

            for pair, pieces in grouped.items():
                remaining = list(pieces)
                while remaining:
                    uv0, uv1, p0, p1 = remaining.pop()
                    chain_uv = [uv0, uv1]
                    chain_points = [p0, p1]

                    changed = True
                    while changed and remaining:
                        changed = False
                        for index, (a_uv, b_uv, a_p, b_p) in enumerate(remaining):
                            if close3(chain_points[-1], a_p):
                                chain_uv.append(b_uv)
                                chain_points.append(b_p)
                            elif close3(chain_points[-1], b_p):
                                chain_uv.append(a_uv)
                                chain_points.append(a_p)
                            elif close3(chain_points[0], b_p):
                                chain_uv.insert(0, a_uv)
                                chain_points.insert(0, a_p)
                            elif close3(chain_points[0], a_p):
                                chain_uv.insert(0, b_uv)
                                chain_points.insert(0, b_p)
                            else:
                                continue
                            remaining.pop(index)
                            changed = True
                            break

                    # Remove duplicate consecutive points introduced at shared
                    # triangle edges before interpolation.
                    clean_uv = [chain_uv[0]]
                    for uv in chain_uv[1:]:
                        if float(np.linalg.norm(uv - clean_uv[-1])) > 1.0e-12:
                            clean_uv.append(uv)
                    if len(clean_uv) >= 2:
                        chains.append((pair, clean_uv))

            return chains

        debug_print("building topology-safe OCCT cutter curves from converged interfaces")
        cutter_edges = []
        if mode == "voronoi":
            chains = _ordered_voronoi_chains(segments)
            debug_print(
                f"reconstruction: {len(segments)} triangle segments -> "
                f"{len(chains)} continuous Voronoi chains"
            )
            for _, uv_chain in chains:
                edge = pcurve_polyline_bspline(uv_chain)
                if edge is None:
                    # Conservative fallback: preserve the partition even if one
                    # unusual chain cannot be interpolated by OCCT.
                    for uv0, uv1 in zip(uv_chain[:-1], uv_chain[1:]):
                        fallback = pcurve_edge(uv0, uv1)
                        if fallback is not None:
                            cutter_edges.append(fallback)
                else:
                    cutter_edges.append(edge)
        else:
            # Delaunay is strictly the dual of the intrinsic Voronoi diagram.
            # Each traced intrinsic path is reconstructed as one topology-safe degree-1 p-curve
            # Edge rather than one Edge per crossed computational triangle.
            for i, j in sorted(clouds.keys()):
                pair = (i, j) if i < j else (j, i)
                if pair in delaunay_boundary_pairs:
                    continue
                path_uv = trace_distance_path(
                    points_array,
                    uvs_array,
                    triangles_array,
                    fields[i],
                    source_indices[i],
                    source_indices[j],
                )
                if path_uv is None:
                    reverse_path = trace_distance_path(
                        points_array,
                        uvs_array,
                        triangles_array,
                        fields[j],
                        source_indices[j],
                        source_indices[i],
                    )
                    if reverse_path is not None:
                        path_uv = list(reversed(reverse_path))
                if path_uv is None or len(path_uv) < 2:
                    if not silent:
                        print(
                            f"Shell.Delaunay - Error: Could not trace the intrinsic "
                            f"geodesic between sites {i} and {j}. Returning None."
                        )
                    return None

                edge = pcurve_polyline_bspline(path_uv)
                if edge is None:
                    for uv0, uv1 in zip(path_uv[:-1], path_uv[1:]):
                        fallback = pcurve_edge(
                            np.asarray(uv0, dtype=float),
                            np.asarray(uv1, dtype=float),
                        )
                        if fallback is not None:
                            cutter_edges.append(fallback)
                else:
                    cutter_edges.append(edge)

        debug_print(f"cutter construction complete ({len(cutter_edges)} edges)")
        if not cutter_edges:
            return Shell.ByFaces([face], tolerance=tolerance, silent=True)

        debug_print("OCCT split: start")
        try:
            splitter = BOPAlgo_Splitter()
            splitter.AddArgument(occ_face)
            for edge in cutter_edges:
                splitter.AddTool(edge)
            splitter.SetFuzzyValue(float(tolerance))
            splitter.Perform()
            if hasattr(splitter, "HasErrors") and splitter.HasErrors():
                return None
            result_shape = splitter.Shape()
        except Exception:
            return None
        debug_print("OCCT split: complete")

        if _is_null_shape(result_shape):
            return None

        debug_print("wrapping split faces")
        result_faces = []
        for occ_subface in _iter_occ_subshapes(result_shape, TopAbs_FACE):
            wrapped = Face.ByOcctShape(occ_subface)
            if isinstance(wrapped, Face):
                result_faces.append(wrapped)
        if not result_faces:
            return None

        # Reconstruction invariant: every generating site must belong to exactly
        # one returned Voronoi Face.  Classify the already-projected site UVs
        # directly against each split Face with OCCT's native 2D classifier.
        # FaceUtility.IsInside is intentionally not used here: on trimmed NURBS
        # split Faces it can return false positives/negatives even when the OCCT
        # UV classifier is unambiguous.
        if mode == "voronoi":
            invalid_sites = []
            for site_index, uv in enumerate(site_uv):
                owners = []
                for face_index, result_face in enumerate(result_faces):
                    try:
                        face_shape = getattr(result_face, "shape", None)
                        if _is_null_shape(face_shape):
                            continue
                        face_classifier = BRepTopAdaptor_FClass2d(
                            topods.Face(face_shape),
                            max(float(tolerance), 1.0e-8),
                        )
                        state = face_classifier.Perform(
                            gp_Pnt2d(float(uv[0]), float(uv[1]))
                        )
                        if state in (TopAbs_IN, TopAbs_ON):
                            owners.append(face_index)
                    except Exception:
                        pass
                debug_print(
                    f"pre-assembly ownership site {site_index}: "
                    f"faces={owners}"
                )
                if len(owners) != 1:
                    invalid_sites.append((site_index, owners))
            if invalid_sites:
                if not silent:
                    print(
                        "Shell.Voronoi - Error: Reconstructed partition does not "
                        f"give each generating site exactly one Face: "
                        f"{invalid_sites}. Returning None."
                    )
                return None

        # The split Faces can contain geometrically coincident but topologically
        # distinct Edge objects (and, in particular, one cutter segment on one
        # Face may correspond to several subsegments on its neighbour).  Merely
        # inserting those Faces into a TopoDS_Shell leaves one-sided interior
        # seams.  Shell.ByFaces uses OCCT sewing, then re-derives the Face wrappers
        # from the sewn Shell so shared boundaries acquire common OCCT Edge
        # identity.  On exact NURBS partitions this preserves the original
        # supporting surface and the splitter-generated p-curves.
        result = Shell.ByFaces(
            result_faces,
            tolerance=tolerance,
            silent=True,
        )
        if not isinstance(result, Shell):
            if not silent:
                print(
                    f"Shell.{mode.capitalize()} - Error: Could not sew the split "
                    "Faces into a coherent Shell. Returning None."
                )
            return None

        # Final-shell invariant.  Query Faces back through the sewn Shell shape,
        # because this is exactly what downstream TopologicPy calls will see.
        final_faces = result.Faces() or []
        if len(final_faces) != len(result_faces):
            if not silent:
                print(
                    f"Shell.{mode.capitalize()} - Error: Final Shell changed the "
                    "split Face count during sewing. Returning None."
                )
            return None

        if mode == "voronoi":
            invalid_sites = []
            for site_index, uv in enumerate(site_uv):
                owners = []
                for face_index, final_face in enumerate(final_faces):
                    try:
                        face_shape = getattr(final_face, "shape", None)
                        if _is_null_shape(face_shape):
                            continue
                        face_classifier = BRepTopAdaptor_FClass2d(
                            topods.Face(face_shape),
                            max(float(tolerance), 1.0e-8),
                        )
                        state = face_classifier.Perform(
                            gp_Pnt2d(float(uv[0]), float(uv[1]))
                        )
                        if state in (TopAbs_IN, TopAbs_ON):
                            owners.append(face_index)
                    except Exception:
                        pass
                debug_print(
                    f"final-shell ownership site {site_index}: "
                    f"faces={owners}"
                )
                if len(owners) != 1:
                    invalid_sites.append((site_index, owners))
            if invalid_sites:
                if not silent:
                    print(
                        "Shell.Voronoi - Error: Final assembled Shell does not "
                        f"give each generating site exactly one Face: "
                        f"{invalid_sites}. Returning None."
                    )
                return None

        debug_print("complete")
        return result

    @staticmethod
    def Voronoi(
        vertices,
        face,
        deflection=None,
        maxIterations: int = 5,
        convergence: float = 0.001,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        return Shell._IntrinsicGeodesicPartition(
            vertices, face, mode="voronoi", deflection=deflection,
            maxIterations=maxIterations, convergence=convergence,
            tolerance=tolerance, silent=silent,
        )

    @staticmethod
    def Delaunay(
        vertices,
        face,
        deflection=None,
        maxIterations: int = 5,
        convergence: float = 0.001,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        return Shell._IntrinsicGeodesicPartition(
            vertices, face, mode="delaunay", deflection=deflection,
            maxIterations=maxIterations, convergence=convergence,
            tolerance=tolerance, silent=silent,
        )


    @staticmethod
    def _EdgesSame(edgeA, edgeB, tolerance: float = 0.0001):
        """Curve-aware Edge equivalence, preferring OCCT topological identity."""
        if not isinstance(edgeA, Edge) or not isinstance(edgeB, Edge):
            return False
        shapeA=getattr(edgeA,'shape',None); shapeB=getattr(edgeB,'shape',None)
        if not _is_null_shape(shapeA) and not _is_null_shape(shapeB):
            try:
                if shapeA.IsSame(shapeB):
                    return True
            except Exception:
                pass
        samplesA=[]; samplesB=[]
        for parameter in (0.0,0.25,0.5,0.75,1.0):
            try:
                a=EdgeUtility.PointAtParameter(edgeA,parameter)
                b=EdgeUtility.PointAtParameter(edgeB,parameter)
                samplesA.append((float(a.x),float(a.y),float(a.z)))
                samplesB.append((float(b.x),float(b.y),float(b.z)))
            except Exception:
                return False
        tol2=float(tolerance)*float(tolerance)
        def close(a,b):
            return (a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2 <= tol2
        return (
            all(close(a,b) for a,b in zip(samplesA,samplesB)) or
            all(close(a,b) for a,b in zip(samplesA,reversed(samplesB)))
        )

    @staticmethod
    def _IncidencePairs(incidence, edge, tolerance: float = 0.0001):
        for group in incidence or []:
            if Shell._EdgesSame(edge,group.get('edge'),tolerance=tolerance):
                return group.get('pairs',[])
        return []

    @staticmethod
    def _edge_face_incidence(faces, tolerance: float = 0.0001):
        """Group identical curve Edges and record their owning Faces."""
        groups=[]
        for face in faces:
            if not isinstance(face,Face):
                continue
            for edge in face.Edges() or []:
                if not isinstance(edge,Edge):
                    continue
                group=None
                for candidate in groups:
                    if Shell._EdgesSame(edge,candidate['edge'],tolerance=tolerance):
                        group=candidate
                        break
                if group is None:
                    group={'edge':edge,'pairs':[]}
                    groups.append(group)
                group['pairs'].append((face,edge))
        return groups

    @staticmethod
    def _patch_edge_face_membership(shell, faces, tolerance: float = 0.0001):
        """Attach curve-aware per-Shell owning-Face information to extracted Edges."""
        incidence=Shell._edge_face_incidence(faces,tolerance=tolerance)
        seen=[]
        for face in faces:
            if not isinstance(face,Face):
                continue
            for edge in face.Edges() or []:
                if not isinstance(edge,Edge):
                    continue
                if any(Shell._EdgesSame(edge,e,tolerance=tolerance) for e in seen):
                    continue
                seen.append(edge)
                owning_faces=unique_by_uuid([f for f,_ in Shell._IncidencePairs(incidence,edge,tolerance=tolerance)])
                by_host=getattr(edge,'_shell_faces_by_host',None)
                if by_host is None:
                    by_host={}; edge._shell_faces_by_host=by_host
                by_host[shell._uuid]=owning_faces
                if not getattr(edge,'_shell_faces_patched',False):
                    edge._shell_faces_patched=True
                    def _edge_faces(self,hostTopology=None,output=None):
                        host_map=getattr(self,'_shell_faces_by_host',None) or {}
                        if hostTopology is not None:
                            host_key=getattr(hostTopology,'_uuid',None)
                            if host_key is not None and host_key in host_map:
                                result=list(host_map[host_key])
                            else:
                                result=Topology.SuperTopologies(self,hostTopology,'Face') or []
                        elif host_map:
                            result=list(next(reversed(list(host_map.values()))))
                        else:
                            result=[]
                        if output is not None:
                            output.extend(result); return 0
                        return result
                    edge.Faces=types.MethodType(_edge_faces,edge)

    def Faces(self, hostTopology=None, faces=None):
        if not _is_null_shape(getattr(self, "shape", None)):
            result = _downward_wrappers(
                self,
                TopAbs_FACE
            )
        else:
            result = list(getattr(self, "faces", []) or [])

        if faces is not None:
            faces.extend(result)
            return 0

        return result


    def Edges(self, hostTopology=None, edges=None):
        if not _is_null_shape(getattr(self, "shape", None)):
            result = _downward_wrappers(
                self,
                TopAbs_EDGE
            )
        else:
            result = []

            for face in getattr(self, "faces", []) or []:
                if isinstance(face, Face):
                    result.extend(face.Edges())

        # Retain the existing backend ordering behaviour.
        result = Shell._boundary_first_ordering(
            result,
            host=self
        )

        if edges is not None:
            edges.extend(result)
            return 0

        return result


    def Vertices(self, hostTopology=None, vertices=None):
        if not _is_null_shape(getattr(self, "shape", None)):
            result = _downward_wrappers(
                self,
                TopAbs_VERTEX
            )
        else:
            result = []

            for edge in self.Edges():
                result.extend(
                    [edge.start, edge.end]
                )

        if vertices is not None:
            vertices.extend(result)
            return 0

        return result

    @staticmethod
    def _boundary_first_ordering(edges, host=None, tolerance: float = 0.0001):
        """Order free boundary chains first without conflating distinct curved Edges."""
        from .wire import Wire
        edges=[e for e in edges if isinstance(e,Edge)]
        incidence=Shell._edge_face_incidence(host.Faces() or [],tolerance=tolerance) if isinstance(host,Shell) else None
        boundary_edges=[]; other_edges=[]
        for edge in edges:
            if incidence is not None:
                owning=Shell._IncidencePairs(incidence,edge,tolerance=tolerance)
            else:
                faces_method=getattr(edge,'Faces',None)
                owning=faces_method(host) if callable(faces_method) else None
            if isinstance(owning,list) and len(owning)==1:
                if not any(Shell._EdgesSame(edge,e,tolerance=tolerance) for e in boundary_edges):
                    boundary_edges.append(edge)
            else:
                other_edges.append(edge)
        if not boundary_edges:
            return edges
        remaining=list(boundary_edges); ordered_boundary=[]
        while remaining:
            chain=Wire._order_edges(remaining,tolerance=tolerance)
            if chain is not None:
                ordered_boundary.extend(chain); break
            component=[remaining[0]]; rest=remaining[1:]; changed=True
            while changed:
                changed=False
                comp_vertices=[]
                for e in component:
                    if isinstance(e.start,Vertex): comp_vertices.append(vertex_key(e.start,tolerance))
                    if isinstance(e.end,Vertex): comp_vertices.append(vertex_key(e.end,tolerance))
                for i,candidate in enumerate(rest):
                    keys=(vertex_key(candidate.start,tolerance),vertex_key(candidate.end,tolerance))
                    if keys[0] in comp_vertices or keys[1] in comp_vertices:
                        component.append(candidate); rest.pop(i); changed=True; break
            sub_order=Wire._order_edges(component,tolerance=tolerance)
            ordered_boundary.extend(sub_order if sub_order is not None else component)
            remaining=rest
        return ordered_boundary+other_edges

    def Shells(self, hostTopology=None, shells=None):
        result = [self]
        if shells is not None:
            shells.extend(result)
            return 0
        return result

    def IsClosed(self, tolerance: float = 0.0001):
        """
        A shell is closed when it has no free (boundary) edges, i.e. every
        edge is shared by exactly two of the shell's faces.
        """
        faces = self.Faces() or []
        if not faces:
            return False
        return len(Shell._boundary_edges(faces, tolerance=tolerance, min_count=1, max_count=1)) == 0

    @staticmethod
    def _boundary_edges(faces, tolerance: float = 0.0001, min_count=None, max_count=None):
        """Return representative Edges whose curve-aware Face incidence is in range."""
        incidence = Shell._edge_face_incidence(faces, tolerance=tolerance)
        result=[]
        for group in incidence:
            pairs=group['pairs']
            count=len(pairs)
            if min_count is not None and count < min_count:
                continue
            if max_count is not None and count > max_count:
                continue
            if pairs:
                result.append(pairs[0][1])
        return result

    @staticmethod
    def _merge_boundary_edges(edges, tolerance: float = 0.0001):
        """
        Stitch boundary edges into a Wire (or Cluster of Wires for disjoint chains). Prefer
        Wire.ByEdges (keeps the walk-ordered .edges the naive IsClosed relies on) over
        _merge_edges_into_wires; fall back for non-simple chains.
        """
        from .wire import Wire

        edges = [e for e in edges if isinstance(e, Edge)]
        if not edges:
            return None
        ordered = Wire._order_edges(edges, tolerance=tolerance)
        if ordered is not None:
            wire = Wire.ByEdges(ordered, tolerance=tolerance)
            if wire is not None:
                return wire
        return Topology._merge_edges_into_wires(edges, tolerance=tolerance)

    @staticmethod
    def ExternalBoundary(shell, tolerance: float = 0.0001, silent: bool = False):
        """Return the longest free/boundary Wire of an open Shell."""
        if not isinstance(shell,Shell):
            if not silent:
                print("Shell.ExternalBoundary - Error: The input shell parameter is not a valid Shell. Returning None.")
            return None
        faces=shell.Faces() or []
        boundary_edges=Shell._boundary_edges(faces,tolerance=tolerance,min_count=1,max_count=1)
        if not boundary_edges:
            if not silent:
                print("Shell.ExternalBoundary - Error: External boundary could not be found. Returning None.")
            return None
        merged=Shell._merge_boundary_edges(boundary_edges,tolerance=tolerance)
        if merged is None:
            return None
        if Topology.IsInstance(merged,'Wire'):
            return merged
        wires=[w for w in getattr(merged,'topologies',[]) or [] if Topology.IsInstance(w,'Wire')]
        if not wires:
            try:
                wires=Topology.Wires(merged) or []
            except Exception:
                wires=[]
        if not wires:
            return None

        def wire_length(wire):
            total=0.0
            try:
                for edge in wire.Edges() or []:
                    value=EdgeUtility.Length(edge,tolerance=tolerance)
                    if value is not None:
                        total += abs(float(value))
            except Exception:
                return 0.0
            return total

        return max(wires,key=wire_length)

    def Slice(self, otherTopology, transferDictionary: bool = False):
        """
        Slice this Shell's faces by a cutting tool, keeping self's material, and reassemble
        the surviving sub-faces into a single Shell (unlike the generic _partition_by which
        wraps the raw result as a Cluster). Cell.Prism depends on returning a Shell.
        """
        from .topology import (
            _collect_boolean_operand_shapes,
            _postprocess_boolean_result,
            _merge_backend_dictionaries,
            _is_null_shape,
            _iter_occ_subshapes,
        )
        try:
            from OCC.Core.TopTools import TopTools_ListOfShape
            from OCC.Core.BOPAlgo import BOPAlgo_CellsBuilder
            from OCC.Core.TopAbs import TopAbs_FACE
        except Exception:
            return None
        if BOPAlgo_CellsBuilder is None:
            return None

        shapes_a = _collect_boolean_operand_shapes(self)
        shapes_b = _collect_boolean_operand_shapes(otherTopology)
        if not shapes_a or not shapes_b:
            return None

        try:
            builder = BOPAlgo_CellsBuilder()
            for shape in shapes_a:
                builder.AddArgument(shape)
            for shape in shapes_b:
                builder.AddArgument(shape)
            builder.Perform()
            if hasattr(builder, "HasErrors") and builder.HasErrors():
                return None

            empty_avoid = TopTools_ListOfShape()
            for shape in shapes_a:
                to_take = TopTools_ListOfShape()
                to_take.Append(shape)
                builder.AddToResult(to_take, empty_avoid)

            builder.MakeContainers()
            result_shape = builder.Shape()
        except Exception:
            return None

        if _is_null_shape(result_shape):
            return None
        result_shape = _postprocess_boolean_result(result_shape)

        result_dictionary = {}
        if transferDictionary:
            result_dictionary = _merge_backend_dictionaries(
                Topology.GetDictionary(self), Topology.GetDictionary(otherTopology)
            )

        result_faces = []
        for occ_face in _iter_occ_subshapes(result_shape, TopAbs_FACE):
            f = Face.ByOcctShape(occ_face)
            if f is not None:
                result_faces.append(f)

        if result_faces:
            new_shell = Shell.ByFaces(result_faces, silent=True)
            if new_shell is not None:
                new_shell.dictionary = result_dictionary
                return new_shell

        # Fall back to the generic wrap (e.g. a genuinely disjoint result).
        return Topology.ByOcctShape(result_shape, dictionary=result_dictionary)

    def Divide(self, otherTopology, transferDictionary: bool = False):
        return self.Slice(otherTopology, transferDictionary=transferDictionary)

    # Impose and Imprint intentionally do NOT alias Slice here (unlike
    # Divide): Impose has its own distinct semantics (keep self's exclusive
    # material AND otherTopology's whole, unsplit material -- see
    # Topology.Impose), and Imprint is already handled correctly by the base
    # Topology._split_by_tool. Aliasing both to Slice used to shadow those
    # base-class implementations for every Shell operand.


class ShellUtility:
    @staticmethod
    def Area(shell):
        if not isinstance(shell, Shell):
            return None
        return sum(
        FaceUtility.Area(f) or 0.0
        for f in shell.Faces()
        )

    @staticmethod
    def ExternalBoundary(shell, tolerance: float = 0.0001):
        return Shell.ExternalBoundary(shell, tolerance=tolerance, silent=True)

    @staticmethod
    def InternalBoundaries(shell, tolerance: float = 0.0001):
        """
        Returns the internal (non-manifold, shared-by-2-faces) boundary wires
        of the shell -- i.e. every wire made of edges that are NOT part of the
        shell's single external boundary. For a simple open shell each such
        internal edge is shared by exactly two faces.
        """
        if not isinstance(shell, Shell):
            return []
        faces = shell.Faces() or []
        internal_edges = Shell._boundary_edges(faces, tolerance=tolerance, min_count=2, max_count=None)
        if not internal_edges:
            return []
        merged = Shell._merge_boundary_edges(internal_edges, tolerance=tolerance)
        if merged is None:
            return []
        if Topology.IsInstance(merged, "Wire"):
            return [merged]
        return [w for w in getattr(merged, "topologies", []) or [] if Topology.IsInstance(w, "Wire")]











def _make_adjacent(method_name):
    """Return a staticmethod that delegates to topology.method(hostTopology, output)."""
    @staticmethod
    def _impl(topology, hostTopology, output):
        if topology is None:
            return 1
        return getattr(topology, method_name)(hostTopology, output)
    return _impl

ShellUtility.AdjacentVertices = _make_adjacent("Vertices")
ShellUtility.AdjacentEdges = _make_adjacent("Edges")
ShellUtility.AdjacentWires = _make_adjacent("Wires")
ShellUtility.AdjacentFaces = _make_adjacent("Faces")
ShellUtility.AdjacentShells = _make_adjacent("Shells")
ShellUtility.AdjacentCells = _make_adjacent("Cells")
ShellUtility.AdjacentCellComplexes = _make_adjacent("CellComplexes")

# Shell.ExternalBoundary, Shell.Slice/Divide/Impose/Imprint, ShellUtility.ExternalBoundary,
# and ShellUtility.InternalBoundaries are implemented above -- do not clobber them here.
