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

try:
    import plotly
    import plotly.graph_objects as go
    import plotly.offline as ofl
except Exception:
    warnings.warn("Plotly - Error: Could not import plotly. Please install plotly manually.")
    plotly = None
    go = None
    ofl = None

class Plotly:
    @staticmethod
    def _plotly_available(silent: bool = True):
        """Returns True if Plotly was imported successfully."""
        if plotly is not None and go is not None:
            return True
        if not silent:
            print("Plotly - Error: Plotly is not available. Please install plotly manually. Returning None.")
        return False

    @staticmethod
    def _color_to_hex(value, default="rgba(0,0,0,0)"):
        """Returns a Plotly-compatible colour while preserving alpha-bearing strings."""
        if value is None:
            value = default
        # Plotly already accepts CSS/named/hex/rgb/rgba/hsl/hsla strings. Returning
        # them unchanged is important because Color.AnyToHex intentionally drops alpha.
        if isinstance(value, str) and value.strip():
            return value.strip()
        try:
            from topologicpy.Color import Color
            result = Color.AnyToHex(value, silent=True)
            return result if isinstance(result, str) and result else default
        except Exception:
            return default


    @staticmethod
    def _dictionary_value(dictionary, key, default=None):
        """Returns a value from a Python or TopologicPy Dictionary."""
        if dictionary is None or key is None:
            return default
        if isinstance(dictionary, dict):
            value = dictionary.get(key, default)
            return default if value is None else value
        try:
            from topologicpy.Dictionary import Dictionary
            try:
                value = Dictionary.ValueAtKey(dictionary, key=key, defaultValue=default)
            except TypeError:
                value = Dictionary.ValueAtKey(dictionary, key, default)
            return default if value is None else value
        except Exception:
            return default

    @staticmethod
    def _dictionary_python(dictionary):
        """Converts a TopologicPy Dictionary to a plain Python dictionary."""
        if dictionary is None:
            return {}
        if isinstance(dictionary, dict):
            return dict(dictionary)
        try:
            from topologicpy.Dictionary import Dictionary
            result = Dictionary.PythonDictionary(dictionary)
            return dict(result) if isinstance(result, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _format_hover(dictionary=None, label=None, fallback="", labelKey=None, excludeKeys=None):
        """Creates concise, HTML-safe hover text from a topology dictionary.

        The heading is shown once. The key used as the heading and visual-style keys
        supplied in ``excludeKeys`` are omitted from the body, preventing the common
        duplicate-label and implementation-detail hover output.
        """
        import html

        data = Plotly._dictionary_python(dictionary)
        excluded = {str(k) for k in (excludeKeys or []) if k is not None}
        if labelKey is not None:
            excluded.add(str(labelKey))

        heading = label if label not in (None, "") else fallback
        lines = []
        if heading not in (None, ""):
            lines.append("<b>" + html.escape(str(heading)) + "</b>")

        def format_value(value):
            if isinstance(value, dict):
                return "; ".join(f"{k}: {v}" for k, v in value.items())
            if isinstance(value, (list, tuple, set)):
                return ", ".join(str(v) for v in value)
            return str(value)

        for key in sorted(data.keys(), key=lambda value: str(value).lower()):
            try:
                key_text = str(key)
                if key_text.startswith("_") or key_text in excluded:
                    continue
                value = data[key]
                if value is None or value == "":
                    continue
                lines.append(
                    "<b>" + html.escape(key_text) + ":</b> " +
                    html.escape(format_value(value))
                )
            except Exception:
                continue

        if lines:
            return "<br>".join(lines)
        return html.escape(str(fallback or ""))

    @staticmethod
    def _vertex_coordinates(vertex, mantissa=6):
        """Returns [x, y, z] for a TopologicPy Vertex."""
        from topologicpy.Vertex import Vertex
        try:
            values = Vertex.Coordinates(vertex, mantissa=mantissa)
        except Exception:
            try:
                values = Vertex.Coordinates(vertex)
            except Exception:
                return None
        if not isinstance(values, (list, tuple)) or len(values) < 3:
            return None
        try:
            return [float(values[0]), float(values[1]), float(values[2])]
        except Exception:
            return None

    @staticmethod
    def _edge_render_points(edge, samples=32, mantissa=6, tolerance=0.0001):
        """
        Samples a TopologicPy Edge for rendering without changing its geometry.

        Linear Edges yield two points. Curved analytic/Bezier/NURBS Edges yield
        ``samples + 1`` points evaluated on the original curve. Degenerate
        zero-length Edges are ignored.

        Parameters
        ----------
        edge : topologicpy.Edge
            The input Edge.
        samples : int , optional
            The number of sampling intervals for curved Edges. Default is 32.
        mantissa : int , optional
            The number of decimal places used for output coordinates. Default is 6.
        tolerance : float , optional
            The geometric tolerance. Default is 0.0001.

        Returns
        -------
        list
            The sampled XYZ coordinates.
        """
        from topologicpy.Edge import Edge
        from topologicpy.Topology import Topology

        try:
            samples = max(4, int(samples))
        except Exception:
            samples = 32

        try:
            tolerance = max(abs(float(tolerance)), 1.0e-12)
        except Exception:
            tolerance = 0.0001

        if not Topology.IsInstance(edge, "Edge"):
            return []

        # OCCT solids such as spheres and cones can contain degenerate
        # topological Edges at singularities. These have no useful renderable
        # 3D curve and should simply be omitted.
        try:
            length = Edge.Length(
                edge,
                mantissa=None,
                tolerance=tolerance,
                silent=True,
            )
        except TypeError:
            try:
                length = Edge.Length(edge)
            except Exception:
                length = None
        except Exception:
            length = None

        if length is not None:
            try:
                if abs(float(length)) <= tolerance:
                    return []
            except Exception:
                pass

        try:
            is_linear = Edge.IsLinear(
                edge,
                tolerance=tolerance,
                silent=True,
            )
        except TypeError:
            try:
                is_linear = Edge.IsLinear(edge)
            except Exception:
                is_linear = None
        except Exception:
            is_linear = None

        parameters = (
            [0.0, 1.0]
            if is_linear is True
            else [i / float(samples) for i in range(samples + 1)]
        )

        points = []

        for parameter in parameters:
            try:
                vertex = Edge.VertexByParameter(
                    edge,
                    parameter,
                    tolerance=tolerance,
                    silent=True,
                )
            except Exception:
                vertex = None

            point = Plotly._vertex_coordinates(
                vertex,
                mantissa=mantissa,
            )

            if point is None:
                continue

            if points:
                dx = point[0] - points[-1][0]
                dy = point[1] - points[-1][1]
                dz = point[2] - points[-1][2]

                if dx * dx + dy * dy + dz * dz <= tolerance * tolerance:
                    continue

            points.append(point)

        if len(points) >= 2:
            return points

        # Defensive endpoint fallback.
        try:
            a = Plotly._vertex_coordinates(
                Edge.StartVertex(edge, silent=True),
                mantissa=mantissa,
            )
            b = Plotly._vertex_coordinates(
                Edge.EndVertex(edge, silent=True),
                mantissa=mantissa,
            )

            if a is None or b is None:
                return []

            dx = b[0] - a[0]
            dy = b[1] - a[1]
            dz = b[2] - a[2]

            if dx * dx + dy * dy + dz * dz <= tolerance * tolerance:
                return []

            return [a, b]

        except Exception:
            return []

    @staticmethod
    def _polyline_midpoint(points):
        """Returns the arc-length midpoint of a rendered polyline."""
        import math
        if not isinstance(points, list) or len(points) < 1:
            return None
        if len(points) == 1:
            return list(points[0])
        lengths = []
        total = 0.0
        for a, b in zip(points[:-1], points[1:]):
            length = math.sqrt(
                (b[0]-a[0])**2 +
                (b[1]-a[1])**2 +
                (b[2]-a[2])**2
            )
            lengths.append(length)
            total += length
        if total <= 1.0e-15:
            return list(points[len(points)//2])
        target = total * 0.5
        cumulative = 0.0
        for index, length in enumerate(lengths):
            if cumulative + length >= target and length > 1.0e-15:
                t = (target - cumulative) / length
                a = points[index]
                b = points[index + 1]
                return [a[j] * (1.0 - t) + b[j] * t for j in range(3)]
            cumulative += length
        return list(points[-1])

    @staticmethod
    def _face_render_mesh(face, quality="medium", mantissa=6, tolerance=0.0001):
        """Returns a Plotly-ready triangular render mesh for a Face.

        The preferred path is ``Topology.Tessellate`` so curved and NURBS faces are
        rendered from their actual supporting geometry. Polygonal tessellation cells
        with more than three indices are triangulated as a fan instead of silently
        discarding all indices after the first three.
        """
        from topologicpy.Face import Face
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(face, "Face"):
            return None

        try:
            tolerance = max(abs(float(tolerance)), 1.0e-12)
        except Exception:
            tolerance = 0.0001

        try:
            mesh = Topology.Tessellate(
                face, quality=quality, weld=True, remesh=True,
                mantissa=mantissa, silent=True,
            )
        except TypeError:
            try:
                mesh = Topology.Tessellate(face, quality=quality, silent=True)
            except Exception:
                mesh = None
        except Exception:
            mesh = None

        if isinstance(mesh, dict):
            raw_vertices = mesh.get("vertices", mesh.get("verts", []))
            raw_faces = mesh.get("faces", mesh.get("tris", []))
            if isinstance(raw_vertices, (list, tuple)) and isinstance(raw_faces, (list, tuple)) and raw_vertices and raw_faces:
                vertices = []
                valid = True
                for vertex in raw_vertices:
                    try:
                        vertices.append([float(vertex[0]), float(vertex[1]), float(vertex[2])])
                    except Exception:
                        valid = False
                        break
                if valid:
                    triangles = []
                    n_vertices = len(vertices)
                    for polygon in raw_faces:
                        if not isinstance(polygon, (list, tuple)) or len(polygon) < 3:
                            continue
                        try:
                            indices = [int(i) for i in polygon]
                        except Exception:
                            continue
                        if any(i < 0 or i >= n_vertices for i in indices):
                            continue
                        a = indices[0]
                        for k in range(1, len(indices) - 1):
                            b, c = indices[k], indices[k + 1]
                            if a != b and b != c and c != a:
                                triangles.append([a, b, c])
                    if triangles:
                        return {"vertices": vertices, "faces": triangles}

        try:
            triangles = Face.Triangulate(face, tolerance=tolerance, silent=True)
        except TypeError:
            try:
                triangles = Face.Triangulate(face, tolerance=tolerance)
            except Exception:
                triangles = None
        except Exception:
            triangles = None

        if not isinstance(triangles, list):
            return None

        vertices, faces = [], []
        for triangle in triangles:
            try:
                tv = Topology.Vertices(triangle, silent=True) or []
            except TypeError:
                try:
                    tv = Topology.Vertices(triangle) or []
                except Exception:
                    tv = []
            except Exception:
                tv = []
            if len(tv) < 3:
                continue
            coords = [Plotly._vertex_coordinates(v, mantissa=mantissa) for v in tv]
            coords = [c for c in coords if c is not None]
            if len(coords) < 3:
                continue
            base = len(vertices)
            vertices.extend(coords)
            for k in range(1, len(coords) - 1):
                faces.append([base, base + k, base + k + 1])
        return {"vertices": vertices, "faces": faces} if faces else None

    @staticmethod
    def AddColorBar(figure, values=None, nTicks=5, xPosition=-0.15, width=15,
                    outlineWidth=0, title="", subTitle="", units="",
                    colorScale="viridis", mantissa: int = 6):
        """Adds an independent scalar colour bar to a Plotly figure."""
        import math

        if not Plotly._plotly_available(silent=True) or not isinstance(figure, go.Figure):
            return None
        if values is None:
            return figure
        if isinstance(values, (int, float)):
            values = [values]
        try:
            iterator = list(values)
        except Exception:
            return figure

        clean_values = []
        for value in iterator:
            try:
                number = float(value)
                if math.isfinite(number):
                    clean_values.append(number)
            except Exception:
                continue
        if not clean_values:
            return figure

        try:
            nTicks = max(2, int(nTicks))
        except Exception:
            nTicks = 5
        try:
            mantissa = max(0, int(mantissa))
        except Exception:
            mantissa = 6

        minValue = min(clean_values)
        maxValue = max(clean_values)
        if maxValue == minValue:
            tickvals = [round(minValue, mantissa)]
        else:
            step = (maxValue - minValue) / float(nTicks - 1)
            tickvals = [round(minValue + i * step, mantissa) for i in range(nTicks)]
            tickvals[-1] = round(maxValue, mantissa)

        title_parts = []
        if title:
            title_parts.append("<b>" + str(title) + "</b>")
        if subTitle:
            title_parts.append(str(subTitle))
        if units:
            title_parts.append("Units: " + str(units))

        figure.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
            marker=dict(
                size=0,
                color=[minValue],
                colorscale=Plotly.ColorScale(colorScale),
                cmin=minValue,
                cmax=maxValue if maxValue != minValue else minValue + 1.0,
                opacity=0,
                showscale=True,
                colorbar=dict(
                    x=float(xPosition),
                    title=dict(text="<br>".join(title_parts)),
                    ticks="outside",
                    tickvals=tickvals,
                    ticktext=[str(x) for x in tickvals],
                    tickmode="array",
                    thickness=max(1, int(width)),
                    outlinewidth=max(0, int(outlineWidth)),
                ),
            ),
        ))
        return figure
    
    @staticmethod
    def Colors():
        """
        Returns the list of named CSS colors that plotly can use.

        Returns
        -------
        list
            The list of named CSS colors.
        """
        return ["aliceblue","antiquewhite","aqua",
                "aquamarine","azure","beige",
                "bisque","black","blanchedalmond",
                "blue","blueviolet","brown",
                "burlywood","cadetblue",
                "chartreuse","chocolate",
                "coral","cornflowerblue","cornsilk",
                "crimson","cyan","darkblue",
                "darkcyan","darkgoldenrod","darkgray",
                "darkgrey","darkgreen","darkkhaki",
                "darkmagenta","darkolivegreen","darkorange",
                "darkorchid","darkred","darksalmon",
                "darkseagreen","darkslateblue","darkslategray",
                "darkslategrey","darkturquoise","darkviolet",
                "deeppink","deepskyblue","dimgray",
                "dimgrey","dodgerblue","firebrick",
                "floralwhite","forestgreen","fuchsia",
                "gainsboro","ghostwhite","gold",
                "goldenrod","gray","grey",
                "green","greenyellow","honeydew",
                "hotpink","indianred","indigo",
                "ivory","khaki","lavender",
                "lavenderblush","lawngreen","lemonchiffon",
                "lightblue","lightcoral","lightcyan",
                "lightgoldenrodyellow","lightgray","lightgrey",
                "lightgreen","lightpink","lightsalmon",
                "lightseagreen","lightskyblue","lightslategray",
                "lightslategrey","lightsteelblue","lightyellow",
                "lime","limegreen","linen",
                "magenta","maroon","mediumaquamarine",
                "mediumblue","mediumorchid","mediumpurple",
                "mediumseagreen","mediumslateblue","mediumspringgreen",
                "mediumturquoise","mediumvioletred","midnightblue",
                "mintcream","mistyrose","moccasin",
                "navajowhite","navy","oldlace",
                "olive","olivedrab","orange",
                "orangered","orchid","palegoldenrod",
                "palegreen","paleturquoise","palevioletred",
                "papayawhip","peachpuff","peru",
                "pink","plum","powderblue",
                "purple","red","rosybrown",
                "royalblue","rebeccapurple","saddlebrown",
                "salmon","sandybrown","seagreen",
                "seashell","sienna","silver",
                "skyblue","slateblue","slategray",
                "slategrey","snow","springgreen",
                "steelblue","tan","teal",
                "thistle","tomato","turquoise",
                "violet","wheat","white",
                "whitesmoke","yellow","yellowgreen"]

    @staticmethod
    def ColorScale(colorScale: str = "viridis"):
        """Returns a Plotly colorscale or a TopologicPy colour-blind-friendly scale."""
        protanopia_colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
        deuteranopia_colors = ["#377EB8", "#FF7F00", "#4DAF4A", "#F781BF", "#A65628", "#984EA3", "#999999"]
        tritanopia_colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

        def create_colorscale(colors):
            if len(colors) == 1:
                return [[0.0, colors[0]], [1.0, colors[0]]]
            return [[i / float(len(colors) - 1), color] for i, color in enumerate(colors)]

        if colorScale is None:
            return "viridis"
        if isinstance(colorScale, (list, tuple)):
            return list(colorScale)
        name = str(colorScale).strip()
        lower = name.lower()
        if "prota" in lower:
            return create_colorscale(protanopia_colors)
        if "deutera" in lower:
            return create_colorscale(deuteranopia_colors)
        if "trita" in lower:
            return create_colorscale(tritanopia_colors)
        return name or "viridis"
    
    @staticmethod
    def DataByGraph(graph,
                    sagitta: float = 0,
                    absolute: bool = False,
                    sides: int = 8,
                    angle: float = 0,
                    directed: bool = False,
                    arrowSize: int = 0.1,
                    arrowSizeKey: str = None,
                    vertexColor: str = "black",
                    vertexColorKey: str = None,
                    vertexSize: float = 10,
                    vertexSizeKey: str = None,
                    vertexLabelKey: str = None,
                    vertexBorderColor: str = "black",
                    vertexBorderWidth: float = 0,
                    vertexBorderColorKey: str = None,
                    vertexBorderWidthKey: float = None,
                    vertexGroupKey: str = None,
                    vertexGroups: list = None,
                    vertexMinGroup=None,
                    vertexMaxGroup=None,
                    showVertices: bool = True,
                    showVertexLabel: bool = False,
                    vertexLabelFontSize: int = 5,
                    showVertexLegend: bool = False,
                    vertexLegendLabel="Graph Vertices",
                    vertexLegendRank=4,
                    vertexLegendGroup=4,
                    edgeColor: str = "red",
                    edgeColorKey: str = None,
                    edgeWidth: float = 1,
                    edgeWidthKey: str = None,
                    edgeDash: bool = False,
                    edgeDashKey: str = None,
                    edgeLabelKey: str = None,
                    edgeGroupKey: str = None,
                    edgeGroups: list = None,
                    edgeMinGroup=None,
                    edgeMaxGroup=None,
                    showEdges: bool = True,
                    showEdgeLabel: bool = False,
                    showEdgeLegend: bool = False,
                    edgeLegendLabel="Graph Edges",
                    edgeLegendRank=5,
                    edgeLegendGroup=5,
                    colorScale: str = "viridis",
                    mantissa: int = 6,
                    silent: bool = False):
        """Creates Plotly data from a Graph, preserving curved Graph Edges."""
        from topologicpy.Edge import Edge
        from topologicpy.Graph import Graph
        from topologicpy.Topology import Topology
        from topologicpy.Vertex import Vertex
        from topologicpy.Wire import Wire

        if not Topology.IsInstance(graph, "Graph"):
            if not silent:
                print("Plotly.DataByGraph - Error: The input graph is not a valid Graph. Returning None.")
            return None

        data = []
        tolerance = 0.0001
        curve_samples = max(16, int(sides) * 4) if isinstance(sides, (int, float)) else 32
        if showEdges:
            render_vertices = []
            render_edges = []
            edge_dicts = []
            graph_edges = Graph.Edges(graph) or []

            for edge in graph_edges:
                try:
                    dictionary = Topology.Dictionary(edge, silent=True)
                except Exception:
                    dictionary = Topology.Dictionary(edge)

                points = []
                if sagitta and float(sagitta) != 0.0:
                    try:
                        arc = Wire.ArcByEdge(edge, sagitta=sagitta, absolute=absolute, sides=max(4, int(sides)), close=False, silent=True)
                    except Exception:
                        arc = None
                    if Topology.IsInstance(arc, "Wire"):
                        if angle:
                            try:
                                arc = Topology.Rotate(arc, origin=Topology.Centroid(edge), axis=Edge.Direction(edge), angle=angle, silent=True)
                            except Exception:
                                pass
                        arc_edges = Topology.Edges(arc, silent=True) or []
                        for arc_edge in arc_edges:
                            segment = Plotly._edge_render_points(arc_edge, samples=curve_samples, mantissa=mantissa, tolerance=tolerance)
                            if points and segment and points[-1] == segment[0]:
                                segment = segment[1:]
                            points.extend(segment)
                if not points:
                    points = Plotly._edge_render_points(edge, samples=curve_samples, mantissa=mantissa, tolerance=tolerance)
                if len(points) < 2:
                    continue
                base = len(render_vertices)
                render_vertices.extend(points)
                render_edges.append(list(range(base, base + len(points))))
                edge_dicts.append(dictionary)

            data.extend(Plotly.edgeData(
                render_vertices, render_edges, dictionaries=edge_dicts,
                color=edgeColor, colorKey=edgeColorKey, width=edgeWidth, widthKey=edgeWidthKey,
                dash=edgeDash, dashKey=edgeDashKey, directed=directed,
                arrowSize=arrowSize, arrowSizeKey=arrowSizeKey,
                labelKey=edgeLabelKey, showEdgeLabel=showEdgeLabel,
                groupKey=edgeGroupKey, minGroup=edgeMinGroup, maxGroup=edgeMaxGroup,
                groups=edgeGroups, legendLabel=edgeLegendLabel,
                legendGroup=edgeLegendGroup, legendRank=edgeLegendRank,
                showLegend=showEdgeLegend, colorScale=colorScale,
            ))

        if showVertices:
            graph_vertices = Graph.Vertices(graph) or []
            coordinates = []
            dictionaries = []
            for vertex in graph_vertices:
                point = Plotly._vertex_coordinates(vertex, mantissa=mantissa)
                if point is None:
                    continue
                coordinates.append(point)
                try:
                    dictionaries.append(Topology.Dictionary(vertex, silent=True))
                except Exception:
                    dictionaries.append(Topology.Dictionary(vertex))
            data.extend(Plotly.vertexData(
                coordinates, dictionaries=dictionaries,
                color=vertexColor, colorKey=vertexColorKey,
                size=vertexSize, sizeKey=vertexSizeKey,
                borderColor=vertexBorderColor, borderWidth=vertexBorderWidth,
                borderColorKey=vertexBorderColorKey, borderWidthKey=vertexBorderWidthKey,
                labelKey=vertexLabelKey, showVertexLabel=showVertexLabel,
                vertexLabelFontSize=vertexLabelFontSize,
                groupKey=vertexGroupKey, minGroup=vertexMinGroup, maxGroup=vertexMaxGroup,
                groups=vertexGroups, legendLabel=vertexLegendLabel,
                legendGroup=vertexLegendGroup, legendRank=vertexLegendRank,
                showLegend=showVertexLegend, colorScale=colorScale,
            ))
        return data

    @staticmethod
    def DataByTGraph(
        graph,
        sagitta: float = 0,
        absolute: bool = False,
        sides: int = 16,
        angle: float = 0,
        directed: bool = None,
        showBidirectionalArrows: bool = True,
        arrowSize: float = 0.15,
        arrowSizeKey: str = None,
        vertexColor: str = "black",
        vertexColorKey: str = "color",
        vertexSize: float = 10,
        vertexSizeKey: str = "size",
        vertexShape: str = "circle",
        vertexShapeKey: str = None,
        vertexLabelKey: str = None,
        vertexBorderColor: str = "black",
        vertexBorderWidth: float = 0,
        vertexBorderColorKey: str = None,
        vertexBorderWidthKey: str = None,
        vertexGroupKey: str = None,
        vertexGroups: list = None,
        vertexMinGroup=None,
        vertexMaxGroup=None,
        showVertices: bool = True,
        showVertexLabel: bool = False,
        vertexLabelFontSize: int = 10,
        showVertexLegend: bool = False,
        vertexLegendLabel: str = "TGraph Vertices",
        vertexLegendRank: int = 4,
        vertexLegendGroup: int = 4,
        edgeColor: str = "red",
        edgeColorKey: str = "color",
        edgeWidth: float = 2,
        edgeWidthKey: str = "width",
        edgeDash: bool = False,
        edgeDashKey: str = None,
        edgeLabelKey: str = None,
        edgeGroupKey: str = None,
        edgeGroups: list = None,
        edgeMinGroup=None,
        edgeMaxGroup=None,
        showEdges: bool = True,
        showEdgeLabel: bool = False,
        edgeLabelFontSize: int = 10,
        showEdgeLegend: bool = False,
        edgeLegendLabel: str = "TGraph Edges",
        edgeLegendRank: int = 5,
        edgeLegendGroup: int = 5,
        colorScale: str = "viridis",
        selfLoopMode: str = "circle",
        selfLoopRadius: float = 0.25,
        selfLoopMajorRadius: float = None,
        selfLoopMinorRadius: float = None,
        selfLoopSides: int = 48,
        selfLoopNormal: list = None,
        selfLoopVertexSize: float = 0,
        splitVertexTracesByStyle: bool = False,
        mantissa: int = 6,
        tolerance: float = 0.0001,
        silent: bool = False,
    ):
        """
        Creates Plotly traces from a ``topologicpy.TGraph``.

        The implementation is optimised for large graphs. Instead of creating
        one ``Scatter3d`` trace per edge, edges are bucketed by visual style and
        rendered in aggregated traces. Labels and cone arrowheads are emitted as
        separate traces only when requested. Self-loops are supported without
        conversion to topologic_core geometry.

        Parameters
        ----------
        graph : topologicpy.TGraph
            The input TGraph.
        sagitta : float, optional
            If non-zero, non-loop edges are drawn as quadratic arcs. If
            ``absolute`` is False, sagitta is interpreted as a ratio of chord
            length. If True, it is interpreted as model units.
        directed : bool or None, optional
            If None, each edge's directed flag is used. If True or False, the
            input value overrides edge-level directionality.
        splitVertexTracesByStyle : bool, optional
            If True, vertices are split by marker symbol and border style. This
            enables per-style borders but creates more traces.

        Returns
        -------
        list or None
            A list of Plotly graph objects, or None if the input is invalid.
        """
        import math
        import plotly.graph_objs as go
        try:
            from topologicpy.TGraph import TGraph
        except Exception:
            try:
                from TGraph import TGraph
            except Exception:
                TGraph = None

        if TGraph is None or not isinstance(graph, TGraph):
            if not silent:
                print("Plotly.DataByTGraph - Error: The input graph is not a valid TGraph. Returning None.")
            return None

        vertexGroups = list(vertexGroups) if vertexGroups is not None else []
        edgeGroups = list(edgeGroups) if edgeGroups is not None else []
        data = []

        # _005 semantic-key compatibility. These aliases are for visual lookup only;
        # RDF serialization remains the responsibility of Ontology/TGraph.
        _KEY_ALIASES = {
            "hasX": "x", "hasY": "y", "hasZ": "z",
            "hasLength": "length", "hasArea": "area", "hasVolume": "volume",
            "hasMantissa": "mantissa", "hasUnit": "unit", "hasWeight": "weight",
            "hasFeature": "feature", "hasFeatureVector": "feature_vector",
            "src": "srcId", "dst": "dstId",
            "IFC_global_id": "ifc_guid", "IFC_id": "ifc_step_id",
            "IFC_key": "ifc_step_key", "IFC_name": "ifc_name", "IFC_type": "ifc_type",
            "ifcGUID": "ifc_guid", "ifcClass": "ifc_class", "ifcName": "ifc_name",
            "ifcType": "ifc_type", "ifcStepId": "ifc_step_id", "ifcStepKey": "ifc_step_key",
            "ontologyClass": "ontology_class", "ontologyURI": "ontology_uri",
            "generatedBy": "generated_by", "generatedByMethod": "generated_by",
            "derivedFrom": "derived_from", "createdAt": "created_at", "modifiedAt": "modified_at",
        }
        _INTERNAL_KEYS = {
            "active", "directed", "color", "colour", "dictionary_mode", "dictionaryMode",
            "import_mode", "importMode", "ontology_predicate", "ontologyPredicate",
            "inverse_predicate", "inversePredicate", "ifc_relationship", "ifcRelationship",
            "relationship_predicate", "relationshipPredicate",
        }

        def _value(d, key, default=None):
            if key is None or not isinstance(d, dict):
                return default
            if key in d:
                return d.get(key, default)
            alias = _KEY_ALIASES.get(str(key))
            if alias is not None and alias in d:
                return d.get(alias, default)
            # Common snake/camel fallback for IFC/ontology keys.
            alt = str(key).replace("IFC_", "ifc_")
            if alt in d:
                return d.get(alt, default)
            return default

        def _format_hover_dict(d, fallback="", label=None, label_key=None, exclude=None):
            excluded = set(_INTERNAL_KEYS)
            excluded.update(str(k) for k in (exclude or []) if k is not None)
            return Plotly._format_hover(
                d, label=label, fallback=fallback, labelKey=label_key,
                excludeKeys=excluded,
            )

        def _number(value, default):
            try:
                if value is None:
                    return default
                return float(value)
            except Exception:
                return default

        def _bool(value, default=False):
            if value is None:
                return default
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            if isinstance(value, str):
                return value.strip().lower() in ("true", "1", "yes", "y", "t")
            return default

        def _label(d, key, default=""):
            if key is None or not isinstance(d, dict):
                return default
            value = _value(d, key, default)
            return "" if value is None else str(value)

        def _unit(vector, default=None):
            if default is None:
                default = [0.0, 0.0, 1.0]
            try:
                x, y, z = float(vector[0]), float(vector[1]), float(vector[2])
            except Exception:
                x, y, z = float(default[0]), float(default[1]), float(default[2])
            length = math.sqrt(x*x + y*y + z*z)
            if length <= 1e-12:
                return [float(default[0]), float(default[1]), float(default[2])]
            return [x/length, y/length, z/length]

        def _cross(a, b):
            return [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]

        def _dot(a, b):
            return float(a[0]*b[0] + a[1]*b[1] + a[2]*b[2])

        def _frame_from_normal(normal=None):
            n = _unit(normal if normal is not None else [0, 0, 1], [0, 0, 1])
            ref = [1, 0, 0] if abs(_dot(n, [1, 0, 0])) <= 0.9 else [0, 1, 0]
            u = _unit(_cross(n, ref), [1, 0, 0])
            v = _unit(_cross(n, u), [0, 1, 0])
            return u, v, n

        def _distance(a, b):
            return math.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2 + (b[2]-a[2])**2)

        def _plotly_symbol(symbol):
            if symbol is None:
                return "circle"
            table = {
                "sphere": "circle", "dot": "circle", "box": "square",
                "triangle": "diamond", "triangle-up": "diamond",
                "triangle-down": "diamond", "star": "diamond",
            }
            s = str(symbol).lower().strip()
            return table.get(s, s)

        def _sample_line(a, b, n=2):
            n = max(2, int(n))
            return [[a[0]*(1-t)+b[0]*t, a[1]*(1-t)+b[1]*t, a[2]*(1-t)+b[2]*t]
                    for t in [i / float(n - 1) for i in range(n)]]

        def _dashed_xyz(points):
            x, y, z = [], [], []
            for i in range(len(points) - 1):
                if i % 2 == 0:
                    a, b = points[i], points[i + 1]
                    x.extend([a[0], b[0], None]); y.extend([a[1], b[1], None]); z.extend([a[2], b[2], None])
            return x, y, z

        def _solid_xyz(points):
            x, y, z = [], [], []
            for p in points:
                x.append(p[0]); y.append(p[1]); z.append(p[2])
            x.append(None); y.append(None); z.append(None)
            return x, y, z

        def _colour_from_group(value, groups, minGroup=None, maxGroup=None, default="black"):
            if value is None:
                return default
            palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                       "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
            try:
                import plotly.colors as pc
                numeric_value = float(value)
                numeric_groups = []
                for item in groups or []:
                    try:
                        numeric_groups.append(float(item))
                    except Exception:
                        pass
                mn = float(minGroup) if minGroup is not None else (min(numeric_groups) if numeric_groups else 0.0)
                mx = float(maxGroup) if maxGroup is not None else (max(numeric_groups) if numeric_groups else 1.0)
                t = 0.0 if abs(mx - mn) <= 1e-12 else (numeric_value - mn) / (mx - mn)
                return pc.sample_colorscale(Plotly.ColorScale(colorScale), [max(0.0, min(1.0, t))])[0]
            except Exception:
                pass
            if groups and value in groups:
                return palette[groups.index(value) % len(palette)]
            try:
                return palette[abs(hash(value)) % len(palette)]
            except Exception:
                return default

        vertex_records = TGraph.Vertices(graph, asTopologic=False, active=True) or []
        edge_records = TGraph.Edges(graph, asTopologic=False, active=True) or []

        if vertexGroupKey is not None and not vertexGroups:
            for record in vertex_records:
                d = record.get("dictionary", {}) if isinstance(record, dict) else {}
                value = _value(d, vertexGroupKey, None)
                if value is not None and value not in vertexGroups:
                    vertexGroups.append(value)
        if edgeGroupKey is not None and not edgeGroups:
            for record in edge_records:
                d = record.get("dictionary", {}) if isinstance(record, dict) else {}
                value = _value(d, edgeGroupKey, None)
                if value is not None and value not in edgeGroups:
                    edgeGroups.append(value)

        coords_by_index = {}
        fallback_n = max(1, len(vertex_records))
        for i, record in enumerate(vertex_records):
            idx = record.get("index")
            coords = TGraph.Coordinates(graph, idx, default=None)
            if coords is None:
                a = 2.0 * math.pi * float(i) / float(fallback_n)
                coords = [math.cos(a), math.sin(a), 0.0]
            coords_by_index[idx] = [round(float(coords[0]), mantissa), round(float(coords[1]), mantissa), round(float(coords[2]), mantissa)]

        def _self_loop_points(anchor, edge_dict):
            d = edge_dict if isinstance(edge_dict, dict) else {}
            mode = str(d.get("self_loop_mode", d.get("mode", selfLoopMode))).lower()
            if mode in ("selfloop", "self_loop", "loop"):
                mode = str(d.get("shape", selfLoopMode)).lower()
            if mode not in ("circle", "ellipse"):
                mode = selfLoopMode
            radius = _number(d.get("self_loop_radius", d.get("radius", None)), selfLoopRadius)
            major = _number(d.get("self_loop_major_radius", d.get("major_radius", d.get("majorRadius", None))), selfLoopMajorRadius if selfLoopMajorRadius is not None else radius)
            minor = _number(d.get("self_loop_minor_radius", d.get("minor_radius", d.get("minorRadius", None))), selfLoopMinorRadius if selfLoopMinorRadius is not None else radius * 0.65)
            if mode == "circle":
                major = radius; minor = radius
            loop_sides = max(8, int(_number(d.get("self_loop_sides", d.get("sides", None)), selfLoopSides)))
            normal = d.get("self_loop_normal", d.get("normal", selfLoopNormal))
            u, v, _ = _frame_from_normal(normal)
            centre = [anchor[0] + major*u[0], anchor[1] + major*u[1], anchor[2] + major*u[2]]
            pts = []
            for k in range(loop_sides + 1):
                a = math.pi + 2.0 * math.pi * float(k) / float(loop_sides)
                ca, sa = math.cos(a), math.sin(a)
                pts.append([centre[0] + major*ca*u[0] + minor*sa*v[0],
                            centre[1] + major*ca*u[1] + minor*sa*v[1],
                            centre[2] + major*ca*u[2] + minor*sa*v[2]])
            return pts

        def _arc_points(a, b, edge_dict):
            d = edge_dict if isinstance(edge_dict, dict) else {}
            edge_sagitta = _number(d.get("sagitta", sagitta), 0.0)
            if abs(edge_sagitta) <= tolerance:
                return _sample_line(a, b, max(2, sides if edgeDash else 2))
            length = _distance(a, b)
            if length <= tolerance:
                return [a, b]
            actual_sagitta = edge_sagitta if absolute else edge_sagitta * length
            tangent = _unit([b[0]-a[0], b[1]-a[1], b[2]-a[2]], [1, 0, 0])
            normal = _unit(d.get("normal", d.get("arc_normal", [0, 0, 1])), [0, 0, 1])
            perp = _cross(normal, tangent)
            if math.sqrt(sum(x*x for x in perp)) <= tolerance:
                _, perp, _ = _frame_from_normal(normal)
            perp = _unit(perp, [0, 1, 0])
            mid = [(a[0]+b[0])*0.5 + actual_sagitta*perp[0],
                   (a[1]+b[1])*0.5 + actual_sagitta*perp[1],
                   (a[2]+b[2])*0.5 + actual_sagitta*perp[2]]
            arc_sides = max(4, int(sides))
            pts = []
            for k in range(arc_sides + 1):
                t = float(k) / float(arc_sides); omt = 1.0 - t
                pts.append([omt*omt*a[0] + 2*omt*t*mid[0] + t*t*b[0],
                            omt*omt*a[1] + 2*omt*t*mid[1] + t*t*b[1],
                            omt*omt*a[2] + 2*omt*t*mid[2] + t*t*b[2]])
            if abs(float(angle or 0)) > 1e-12:
                theta = math.radians(float(angle)); ca, sa = math.cos(theta), math.sin(theta)
                origin = [(a[0]+b[0])*0.5, (a[1]+b[1])*0.5, (a[2]+b[2])*0.5]
                axis = tangent
                rotated = []
                for p in pts:
                    x = [p[0]-origin[0], p[1]-origin[1], p[2]-origin[2]]
                    cr = _cross(axis, x); dt = _dot(axis, x)
                    r = [x[0]*ca + cr[0]*sa + axis[0]*dt*(1-ca),
                         x[1]*ca + cr[1]*sa + axis[1]*dt*(1-ca),
                         x[2]*ca + cr[2]*sa + axis[2]*dt*(1-ca)]
                    rotated.append([origin[0]+r[0], origin[1]+r[1], origin[2]+r[2]])
                pts = rotated
            return pts

        def _edge_points(record):
            src, dst = record.get("src"), record.get("dst")
            d = record.get("dictionary", {}) if isinstance(record, dict) else {}
            a, b = coords_by_index.get(src), coords_by_index.get(dst)
            if a is None or b is None:
                return []
            return _self_loop_points(a, d) if src == dst else _arc_points(a, b, d)

        def _direction(points, reverse=False):
            if len(points) < 2:
                return [1, 0, 0]
            a, b = (points[1], points[0]) if reverse else (points[-2], points[-1])
            return _unit([b[0]-a[0], b[1]-a[1], b[2]-a[2]], [1, 0, 0])

        if showEdges:
            edge_buckets = {}
            arrow_buckets = {}
            self_loop_markers = {"x": [], "y": [], "z": [], "color": [], "hover": []}
            label_x, label_y, label_z, label_text, label_color = [], [], [], [], []
            for record in edge_records:
                d = record.get("dictionary", {}) if isinstance(record, dict) else {}
                points = _edge_points(record)
                if len(points) < 2:
                    continue
                this_color = _value(d, edgeColorKey, None)
                if this_color is None and edgeGroupKey is not None:
                    this_color = _colour_from_group(_value(d, edgeGroupKey, None), edgeGroups, edgeMinGroup, edgeMaxGroup, default=edgeColor)
                if this_color is None:
                    this_color = edgeColor
                this_width = _number(_value(d, edgeWidthKey, None), edgeWidth)
                this_dash = _bool(_value(d, edgeDashKey, None), edgeDash)
                key = (str(this_color), float(this_width), bool(this_dash))
                bucket = edge_buckets.setdefault(key, {"x": [], "y": [], "z": [], "text": []})
                if this_dash:
                    x, y, z = _dashed_xyz(points if len(points) > 2 else _sample_line(points[0], points[1], max(8, int(sides) * 2)))
                else:
                    x, y, z = _solid_xyz(points)
                bucket["x"].extend(x); bucket["y"].extend(y); bucket["z"].extend(z)
                edge_label = _label(d, edgeLabelKey, "")
                edge_fallback = f"Edge {record.get('index', '')}"
                edge_hover = _format_hover_dict(
                    d, fallback=edge_fallback, label=edge_label or edge_fallback,
                    label_key=edgeLabelKey,
                    exclude=[edgeColorKey, edgeWidthKey, edgeDashKey, arrowSizeKey, edgeGroupKey],
                )
                bucket["text"].extend([edge_hover] * len(x))
                if record.get("src") == record.get("dst") and float(selfLoopVertexSize or 0) > 0:
                    anchor = coords_by_index.get(record.get("src"))
                    if anchor is not None:
                        self_loop_markers["x"].append(anchor[0]); self_loop_markers["y"].append(anchor[1]); self_loop_markers["z"].append(anchor[2])
                        self_loop_markers["color"].append(this_color); self_loop_markers["hover"].append(edge_hover)
                if showEdgeLabel and edgeLabelKey is not None and edge_label:
                    mp = points[len(points)//2]
                    label_x.append(mp[0]); label_y.append(mp[1]); label_z.append(mp[2]); label_text.append(edge_label); label_color.append(this_color)
                edge_is_directed = bool(record.get("directed", getattr(graph, "_directed", False)))
                draw_directed = edge_is_directed if directed is None else bool(directed)
                draw_both = (not edge_is_directed) and bool(directed) and bool(showBidirectionalArrows)
                if draw_directed or draw_both:
                    this_arrow_size = _number(_value(d, arrowSizeKey, None), arrowSize)
                    akey = (str(this_color), float(this_arrow_size))
                    ab = arrow_buckets.setdefault(akey, {"x": [], "y": [], "z": [], "u": [], "v": [], "w": []})
                    end = points[-1]; vec = _direction(points, reverse=False)
                    ab["x"].append(end[0]); ab["y"].append(end[1]); ab["z"].append(end[2]); ab["u"].append(vec[0]); ab["v"].append(vec[1]); ab["w"].append(vec[2])
                    if draw_both:
                        start = points[0]; vec = _direction(points, reverse=True)
                        ab["x"].append(start[0]); ab["y"].append(start[1]); ab["z"].append(start[2]); ab["u"].append(vec[0]); ab["v"].append(vec[1]); ab["w"].append(vec[2])
            first = True
            for (this_color, this_width, _), bucket in edge_buckets.items():
                data.append(go.Scatter3d(x=bucket["x"], y=bucket["y"], z=bucket["z"], mode="lines",
                                         line=dict(color=this_color, width=this_width), name=edgeLegendLabel,
                                         legendgroup=str(edgeLegendGroup), legendrank=edgeLegendRank,
                                         showlegend=bool(showEdgeLegend and first),
                                         hovertext=bucket["text"],
                                         hovertemplate="%{hovertext}<extra></extra>",
                                         hoverlabel=dict(align="left", namelength=-1)))
                first = False
            if showEdgeLabel and label_text:
                data.append(go.Scatter3d(x=label_x, y=label_y, z=label_z, mode="text", text=label_text,
                                         textfont=dict(size=max(1, int(edgeLabelFontSize))),
                                         showlegend=False, hoverinfo="skip"))
            for (this_color, this_arrow_size), bucket in arrow_buckets.items():
                data.append(go.Cone(x=bucket["x"], y=bucket["y"], z=bucket["z"],
                                    u=bucket["u"], v=bucket["v"], w=bucket["w"],
                                    sizemode="absolute", sizeref=this_arrow_size, anchor="tip",
                                    showscale=False, colorscale=[[0, this_color], [1, this_color]],
                                    showlegend=False, hoverinfo="skip"))
            if self_loop_markers["x"]:
                data.append(go.Scatter3d(
                    x=self_loop_markers["x"], y=self_loop_markers["y"], z=self_loop_markers["z"],
                    mode="markers", marker=dict(size=max(0.1, float(selfLoopVertexSize)), color=self_loop_markers["color"]),
                    hovertext=self_loop_markers["hover"], hovertemplate="%{hovertext}<extra></extra>",
                    hoverlabel=dict(align="left", namelength=-1), showlegend=False,
                ))

        if showVertices and vertex_records:
            vertex_items = []
            for record in vertex_records:
                idx = record.get("index")
                d = record.get("dictionary", {}) if isinstance(record, dict) else {}
                c = coords_by_index.get(idx)
                if c is None:
                    continue
                this_color = _value(d, vertexColorKey, None)
                if this_color is None and vertexGroupKey is not None:
                    this_color = _colour_from_group(_value(d, vertexGroupKey, None), vertexGroups, vertexMinGroup, vertexMaxGroup, default=vertexColor)
                if this_color is None:
                    this_color = vertexColor
                vertex_items.append({
                    "x": c[0], "y": c[1], "z": c[2],
                    "label": _label(d, vertexLabelKey, str(idx) if vertexLabelKey is None else ""),
                    "hover": _format_hover_dict(
                        d, fallback=f"Vertex {idx}",
                        label=_label(d, vertexLabelKey, f"Vertex {idx}"),
                        label_key=vertexLabelKey,
                        exclude=[vertexColorKey, vertexSizeKey, vertexShapeKey, vertexBorderColorKey, vertexBorderWidthKey, vertexGroupKey],
                    ),
                    "color": this_color,
                    "size": _number(_value(d, vertexSizeKey, None), vertexSize),
                    "symbol": _plotly_symbol(_value(d, vertexShapeKey, vertexShape)),
                    "border_color": _value(d, vertexBorderColorKey, vertexBorderColor) if splitVertexTracesByStyle else vertexBorderColor,
                    "border_width": _number(_value(d, vertexBorderWidthKey, None), vertexBorderWidth) if splitVertexTracesByStyle else vertexBorderWidth,
                })
            if vertex_items and not splitVertexTracesByStyle:
                data.append(go.Scatter3d(
                    x=[i["x"] for i in vertex_items], y=[i["y"] for i in vertex_items], z=[i["z"] for i in vertex_items],
                    mode="markers+text" if showVertexLabel else "markers",
                    marker=dict(size=[i["size"] for i in vertex_items], color=[i["color"] for i in vertex_items],
                                symbol=[i["symbol"] for i in vertex_items], line=dict(color=vertexBorderColor, width=vertexBorderWidth)),
                    text=[i["label"] for i in vertex_items] if showVertexLabel else None,
                    textfont=dict(size=vertexLabelFontSize), hovertext=[i["hover"] for i in vertex_items],
                    hovertemplate="%{hovertext}<extra></extra>", hoverlabel=dict(align="left", namelength=-1),
                    name=vertexLegendLabel, legendgroup=str(vertexLegendGroup), legendrank=vertexLegendRank,
                    showlegend=showVertexLegend))
            elif vertex_items:
                buckets = {}
                for item in vertex_items:
                    buckets.setdefault((item["symbol"], item["border_color"], float(item["border_width"])), []).append(item)
                first = True
                for (symbol, border_color, border_width), items in buckets.items():
                    data.append(go.Scatter3d(
                        x=[i["x"] for i in items], y=[i["y"] for i in items], z=[i["z"] for i in items],
                        mode="markers+text" if showVertexLabel else "markers",
                        marker=dict(size=[i["size"] for i in items], color=[i["color"] for i in items], symbol=symbol,
                                    line=dict(color=border_color, width=border_width)),
                        text=[i["label"] for i in items] if showVertexLabel else None,
                        textfont=dict(size=vertexLabelFontSize), hovertext=[i["hover"] for i in items],
                        hovertemplate="%{hovertext}<extra></extra>", hoverlabel=dict(align="left", namelength=-1),
                        name=vertexLegendLabel, legendgroup=str(vertexLegendGroup), legendrank=vertexLegendRank,
                        showlegend=bool(showVertexLegend and first)))
                    first = False

        return data


    @staticmethod
    def DataByProofGraph(
        proofGraphData=None,
        result=None,
        triple=None,
        graph=None,
        layout: str = "tree",
        useExistingCoordinates: bool = True,
        showNodes: bool = True,
        showEdges: bool = True,
        showNodeLabel: bool = True,
        showEdgeLabel: bool = False,
        nodeLabelKey: str = "label",
        edgeLabelKey: str = "label",
        nodeTypeKey: str = "type",
        edgeTypeKey: str = "type",
        nodeSize: float = 12,
        ruleNodeSize: float = 15,
        factNodeSize: float = 12,
        literalNodeSize: float = 9,
        edgeWidth: float = 3,
        levelSpacing: float = 1.8,
        nodeSpacing: float = 1.35,
        radialRadiusStep: float = 1.5,
        forceIterations: int = 120,
        forceScale: float = 1.0,
        showLegend: bool = True,
        hover: bool = True,
        mantissa: int = 6,
        silent: bool = False,
    ):
        """
        Creates Plotly traces for a proof graph.

        This method is designed for the explainable reasoning workflow in
        ``topologicpy.Reasoner``. It accepts a proof-graph data dictionary,
        a proof ``TGraph``, or an inference ``result`` and ``triple`` from
        which proof-graph data can be requested.

        The expected dictionary format is intentionally permissive:

        ``{"nodes": [...], "edges": [...]}``

        where each node may contain ``id``, ``label``, ``type``/``kind``,
        ``depth``/``level``, ``x``, ``y``, and ``z``; and each edge may contain
        ``source``/``src``/``from``, ``target``/``dst``/``to``, ``label``, and
        ``type``/``role``. This loose contract allows the method to consume
        proof data from ``Reasoner``, ``KnowledgeGraph``, or user-authored
        dictionaries.

        Parameters
        ----------
        proofGraphData : dict or topologicpy.TGraph , optional
            The proof graph data or proof TGraph. Default is None.
        result : object , optional
            An inference result returned by Reasoner.Infer or
            TGraph.InferOntology. Used only when proofGraphData is None.
        triple : tuple or list , optional
            The inferred triple to explain. Used only when proofGraphData is
            None.
        graph : object , optional
            Optional source TGraph/RDF graph to pass to Reasoner when deriving
            proofGraphData. Default is None.
        layout : str , optional
            The layout to use. Options include "tree", "radial", and "force".
            Default is "tree".
        useExistingCoordinates : bool , optional
            If True, existing node coordinates in proofGraphData are used when
            available. Default is True.
        showNodes, showEdges : bool , optional
            Controls visibility of proof graph nodes and edges.
        showNodeLabel, showEdgeLabel : bool , optional
            Controls permanent text labels.
        nodeLabelKey, edgeLabelKey : str , optional
            Dictionary keys used for node and edge labels.
        nodeTypeKey, edgeTypeKey : str , optional
            Dictionary keys used for node and edge type classification.
        nodeSize, ruleNodeSize, factNodeSize, literalNodeSize : float , optional
            Marker sizes.
        edgeWidth : float , optional
            Line width for proof graph edges.
        levelSpacing, nodeSpacing, radialRadiusStep : float , optional
            Layout spacing controls.
        forceIterations : int , optional
            Number of iterations for the force-directed layout.
        forceScale : float , optional
            Scale factor for force-directed layout coordinates.
        showLegend : bool , optional
            If True, show node/edge legends.
        hover : bool , optional
            If True, attach dictionary metadata to hover text.
        mantissa : int , optional
            Number of decimal places to round coordinates.
        silent : bool , optional
            If True, suppress warnings.

        Returns
        -------
        list or None
            A list of Plotly traces, or None if proof graph data cannot be
            derived.
        """
        import math
        import plotly.graph_objs as go

        def _is_tgraph(obj):
            try:
                from topologicpy.TGraph import TGraph
                return isinstance(obj, TGraph)
            except Exception:
                try:
                    from TGraph import TGraph
                    return isinstance(obj, TGraph)
                except Exception:
                    return False

        def _call_reasoner_for_data():
            if proofGraphData is not None:
                return proofGraphData
            # Prefer the new Reasoner plotting-neutral proof data path.
            try:
                from topologicpy.Reasoner import Reasoner
            except Exception:
                try:
                    from Reasoner import Reasoner
                except Exception:
                    Reasoner = None
            if Reasoner is not None and hasattr(Reasoner, "ProofGraphData"):
                # Reasoner_005 uses resultOrGraph as the first argument. Older
                # drafts accepted result= as a loose keyword, but that keyword was
                # ignored by the plotting-neutral proof-data function and could
                # silently produce an empty/not-found proof graph.
                call_variants = []
                if result is not None:
                    call_variants.extend([
                        (result, dict(triple=triple, layout=layout)),
                        (result, dict(triple=triple)),
                    ])
                if graph is not None:
                    call_variants.extend([
                        (graph, dict(triple=triple, layout=layout)),
                        (graph, dict(triple=triple)),
                    ])
                call_variants.extend([
                    (None, dict(triple=triple, layout=layout)),
                    (None, dict(triple=triple)),
                ])
                for first_arg, kwargs in call_variants:
                    try:
                        if first_arg is None:
                            candidate = Reasoner.ProofGraphData(**kwargs)
                        else:
                            candidate = Reasoner.ProofGraphData(first_arg, **kwargs)
                        if isinstance(candidate, dict):
                            return candidate
                    except TypeError:
                        continue
                    except Exception:
                        continue
            # Some result containers may expose their own proof graph data.
            if result is not None:
                for attr in ["ProofGraphData", "proofGraphData", "proof_graph_data", "ProofData", "proofData"]:
                    obj = getattr(result, attr, None)
                    if callable(obj):
                        try:
                            return obj(triple=triple, layout=layout)
                        except TypeError:
                            try:
                                return obj(triple)
                            except Exception:
                                pass
                        except Exception:
                            pass
                    elif obj is not None:
                        return obj
            return None

        data = _call_reasoner_for_data()

        if data is None:
            if not silent:
                print("Plotly.DataByProofGraph - Error: Could not derive proof graph data. Returning None.")
            return None

        if _is_tgraph(data):
            return Plotly.DataByTGraph(
                data,
                directed=True,
                vertexColorKey="color",
                vertexSizeKey="size",
                vertexShapeKey="shape",
                vertexLabelKey=nodeLabelKey,
                showVertexLabel=showNodeLabel,
                edgeColorKey="color",
                edgeWidthKey="width",
                edgeLabelKey=edgeLabelKey,
                showEdgeLabel=showEdgeLabel,
                edgeWidth=edgeWidth,
                vertexSize=nodeSize,
                showVertexLegend=showLegend,
                showEdgeLegend=showLegend,
                silent=silent,
            )

        if not isinstance(data, dict):
            if not silent:
                print("Plotly.DataByProofGraph - Error: The proof graph data must be a dictionary or TGraph. Returning None.")
            return None

        raw_nodes = data.get("nodes", data.get("vertices", []))
        raw_edges = data.get("edges", data.get("links", data.get("relationships", [])))

        if not isinstance(raw_nodes, list):
            raw_nodes = []
        if not isinstance(raw_edges, list):
            raw_edges = []

        nodes = []
        node_index = {}

        def _as_dict_node(item, i):
            if isinstance(item, dict):
                d = dict(item)
            else:
                d = {"id": str(item), "label": str(item)}
            node_id = d.get("id", d.get("key", d.get("uri", d.get("name", i))))
            node_id = str(node_id)
            d["id"] = node_id
            d.setdefault("label", d.get(nodeLabelKey, node_id))
            ntype = d.get(nodeTypeKey, d.get("kind", d.get("category", d.get("role", "node"))))
            d[nodeTypeKey] = str(ntype if ntype is not None else "node")
            return d

        for i, item in enumerate(raw_nodes):
            d = _as_dict_node(item, i)
            if d["id"] in node_index:
                continue
            node_index[d["id"]] = len(nodes)
            nodes.append(d)

        def _edge_source(edge):
            if not isinstance(edge, dict):
                if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                    return str(edge[0])
                return None
            return edge.get("source", edge.get("src", edge.get("srcId", edge.get("from", edge.get("subject", edge.get("s"))))))

        def _edge_target(edge):
            if not isinstance(edge, dict):
                if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                    return str(edge[1])
                return None
            return edge.get("target", edge.get("dst", edge.get("dstId", edge.get("to", edge.get("object", edge.get("o"))))))

        edges = []
        for i, item in enumerate(raw_edges):
            if isinstance(item, dict):
                e = dict(item)
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                e = {"source": item[0], "target": item[1]}
                if len(item) >= 3:
                    e["label"] = item[2]
            else:
                continue
            s = _edge_source(e)
            t = _edge_target(e)
            if s is None or t is None:
                continue
            s = str(s)
            t = str(t)
            for node_id in [s, t]:
                if node_id not in node_index:
                    node_index[node_id] = len(nodes)
                    nodes.append({"id": node_id, "label": node_id, nodeTypeKey: "node"})
            e["source"] = s
            e["target"] = t
            e.setdefault("label", e.get(edgeLabelKey, e.get("role", e.get("predicate", ""))))
            e.setdefault(edgeTypeKey, e.get("kind", e.get("role", "edge")))
            edges.append(e)

        if not nodes:
            if not silent:
                print("Plotly.DataByProofGraph - Error: The proof graph contains no nodes. Returning None.")
            return None

        def _num(value, default=0.0):
            try:
                if value is None:
                    return float(default)
                return float(value)
            except Exception:
                return float(default)

        def _type(value):
            return str(value if value is not None else "node").strip().lower()

        def _display_label(d, key, fallback):
            value = d.get(key, d.get("label", fallback))
            if value is None:
                value = fallback
            return str(value)

        _INTERNAL_PROOF_KEYS = {
            "active", "directed", "color", "colour", "dictionary_mode", "dictionaryMode",
            "import_mode", "importMode", "ontology_predicate", "ontologyPredicate",
            "inverse_predicate", "inversePredicate", "ifc_relationship", "ifcRelationship",
            "relationship_predicate", "relationshipPredicate",
        }

        def _hover(d, label=None, label_key=None, fallback=""):
            if not hover:
                return ""
            return Plotly._format_hover(
                d, label=label, fallback=fallback, labelKey=label_key,
                excludeKeys=_INTERNAL_PROOF_KEYS,
            )

        node_palette = {
            "conclusion": "#D55E00",
            "target": "#D55E00",
            "inferred": "#0072B2",
            "inferred_fact": "#0072B2",
            "fact": "#0072B2",
            "asserted": "#009E73",
            "asserted_fact": "#009E73",
            "premise": "#56B4E9",
            "rule": "#CC79A7",
            "inference_rule": "#CC79A7",
            "class": "#E69F00",
            "ontology_class": "#E69F00",
            "property": "#999999",
            "predicate": "#999999",
            "literal": "#666666",
            "node": "#7f7f7f",
        }

        edge_palette = {
            "derived_by": "#CC79A7",
            "derived-by": "#CC79A7",
            "supports": "#0072B2",
            "premise": "#56B4E9",
            "uses": "#56B4E9",
            "subclass": "#E69F00",
            "subclassof": "#E69F00",
            "subproperty": "#999999",
            "edge": "#999999",
        }

        def _node_color(d):
            explicit = d.get("color", d.get("colour", None))
            if explicit:
                return str(explicit)
            t = _type(d.get(nodeTypeKey, d.get("kind", d.get("category", "node"))))
            if t in node_palette:
                return node_palette[t]
            if "rule" in t:
                return node_palette["rule"]
            if "assert" in t:
                return node_palette["asserted"]
            if "infer" in t or "fact" in t:
                return node_palette["inferred"]
            if "literal" in t:
                return node_palette["literal"]
            return node_palette["node"]

        def _edge_color(e):
            explicit = e.get("color", e.get("colour", None))
            if explicit:
                return str(explicit)
            t = _type(e.get(edgeTypeKey, e.get("kind", e.get("role", "edge"))))
            compact = t.replace("_", "").replace("-", "").replace(" ", "")
            if t in edge_palette:
                return edge_palette[t]
            if compact in edge_palette:
                return edge_palette[compact]
            if "premise" in compact or "support" in compact:
                return edge_palette["premise"]
            if "derive" in compact:
                return edge_palette["derived_by"]
            return edge_palette["edge"]

        def _node_symbol(d):
            explicit = d.get("symbol", d.get("shape", None))
            if explicit is not None:
                s = str(explicit).lower()
                if s in ["circle", "square", "diamond", "cross", "x", "circle-open", "square-open", "diamond-open"]:
                    return s
            t = _type(d.get(nodeTypeKey, d.get("kind", d.get("category", "node"))))
            if "rule" in t:
                return "diamond"
            if "literal" in t:
                return "square"
            if "class" in t:
                return "square"
            if "assert" in t:
                return "circle-open"
            return "circle"

        def _node_size(d):
            explicit = d.get("size", None)
            if explicit is not None:
                return max(1.0, _num(explicit, nodeSize))
            t = _type(d.get(nodeTypeKey, d.get("kind", d.get("category", "node"))))
            if "rule" in t:
                return ruleNodeSize
            if "literal" in t:
                return literalNodeSize
            if "fact" in t or "infer" in t or "assert" in t:
                return factNodeSize
            return nodeSize

        def _has_coordinates():
            if not useExistingCoordinates:
                return False
            for d in nodes:
                if not all(k in d for k in ["x", "y", "z"]):
                    return False
                try:
                    float(d["x"]); float(d["y"]); float(d["z"])
                except Exception:
                    return False
            return True

        coords = {}

        if _has_coordinates():
            for d in nodes:
                coords[d["id"]] = [round(_num(d.get("x")), mantissa), round(_num(d.get("y")), mantissa), round(_num(d.get("z")), mantissa)]
        else:
            layout_l = str(layout or "tree").strip().lower()

            # Establish levels. Prefer explicit depth/level; otherwise derive
            # them from directed edge incidence.
            levels = {}
            for d in nodes:
                val = d.get("level", d.get("depth", None))
                if val is not None:
                    try:
                        levels[d["id"]] = int(float(val))
                    except Exception:
                        pass

            if len(levels) < len(nodes):
                indeg = {d["id"]: 0 for d in nodes}
                out = {d["id"]: [] for d in nodes}
                for e in edges:
                    s, t = e["source"], e["target"]
                    out.setdefault(s, []).append(t)
                    indeg[t] = indeg.get(t, 0) + 1
                    indeg.setdefault(s, 0)
                roots = [node_id for node_id, deg in indeg.items() if deg == 0]
                if not roots:
                    roots = [nodes[0]["id"]]
                queue = [(r, 0) for r in roots]
                seen = set()
                while queue:
                    node_id, lev = queue.pop(0)
                    if node_id in seen:
                        continue
                    seen.add(node_id)
                    if node_id not in levels:
                        levels[node_id] = lev
                    for nb in out.get(node_id, []):
                        queue.append((nb, lev + 1))
                for d in nodes:
                    levels.setdefault(d["id"], 0)

            if "rad" in layout_l:
                by_level = {}
                for d in nodes:
                    by_level.setdefault(levels.get(d["id"], 0), []).append(d["id"])
                for lev, ids in by_level.items():
                    radius = max(0.1, (lev + 1) * radialRadiusStep)
                    count = max(1, len(ids))
                    for i, node_id in enumerate(ids):
                        angle = 2.0 * math.pi * float(i) / float(count)
                        coords[node_id] = [
                            round(radius * math.cos(angle), mantissa),
                            round(radius * math.sin(angle), mantissa),
                            round(-0.15 * lev, mantissa),
                        ]
            elif "force" in layout_l or "spring" in layout_l:
                n = len(nodes)
                index = {d["id"]: i for i, d in enumerate(nodes)}
                pos = {}
                for i, d in enumerate(nodes):
                    a = 2.0 * math.pi * float(i) / float(max(1, n))
                    pos[d["id"]] = [math.cos(a) * forceScale, math.sin(a) * forceScale, 0.0]
                edge_pairs = [(e["source"], e["target"]) for e in edges]
                k = math.sqrt(1.0 / float(max(1, n))) * forceScale
                iterations = max(1, int(forceIterations))
                for it in range(iterations):
                    disp = {d["id"]: [0.0, 0.0, 0.0] for d in nodes}
                    temperature = forceScale * (1.0 - float(it) / float(iterations + 1))
                    ids = [d["id"] for d in nodes]
                    for i in range(n):
                        vi = ids[i]
                        for j in range(i + 1, n):
                            vj = ids[j]
                            dx = pos[vi][0] - pos[vj][0]
                            dy = pos[vi][1] - pos[vj][1]
                            dist = max(1e-6, math.sqrt(dx * dx + dy * dy))
                            force = (k * k) / dist
                            ux, uy = dx / dist, dy / dist
                            disp[vi][0] += ux * force
                            disp[vi][1] += uy * force
                            disp[vj][0] -= ux * force
                            disp[vj][1] -= uy * force
                    for s, t in edge_pairs:
                        if s not in pos or t not in pos:
                            continue
                        dx = pos[s][0] - pos[t][0]
                        dy = pos[s][1] - pos[t][1]
                        dist = max(1e-6, math.sqrt(dx * dx + dy * dy))
                        force = (dist * dist) / max(k, 1e-6)
                        ux, uy = dx / dist, dy / dist
                        disp[s][0] -= ux * force
                        disp[s][1] -= uy * force
                        disp[t][0] += ux * force
                        disp[t][1] += uy * force
                    for node_id in pos:
                        dx, dy = disp[node_id][0], disp[node_id][1]
                        dist = max(1e-6, math.sqrt(dx * dx + dy * dy))
                        step = min(dist, temperature)
                        pos[node_id][0] += (dx / dist) * step
                        pos[node_id][1] += (dy / dist) * step
                for d in nodes:
                    lev = levels.get(d["id"], 0)
                    coords[d["id"]] = [round(pos[d["id"]][0], mantissa), round(pos[d["id"]][1], mantissa), round(-0.05 * lev, mantissa)]
            else:
                # Layered tree layout. Edges usually flow from premises to
                # conclusion; deeper levels are placed lower on the y-axis.
                by_level = {}
                for d in nodes:
                    by_level.setdefault(levels.get(d["id"], 0), []).append(d["id"])
                for lev in sorted(by_level.keys()):
                    ids = by_level[lev]
                    count = len(ids)
                    for i, node_id in enumerate(ids):
                        x = (float(i) - float(count - 1) * 0.5) * nodeSpacing
                        y = -float(lev) * levelSpacing
                        coords[node_id] = [round(x, mantissa), round(y, mantissa), 0.0]

        traces = []

        if showEdges and edges:
            edge_buckets = {}
            label_x, label_y, label_z, label_text = [], [], [], []
            for e in edges:
                s = e.get("source")
                t = e.get("target")
                if s not in coords or t not in coords:
                    continue
                c1 = coords[s]
                c2 = coords[t]
                this_color = _edge_color(e)
                this_width = _num(e.get("width", edgeWidth), edgeWidth)
                key = (this_color, this_width)
                bucket = edge_buckets.setdefault(key, {"x": [], "y": [], "z": [], "text": []})
                bucket["x"].extend([c1[0], c2[0], None])
                bucket["y"].extend([c1[1], c2[1], None])
                bucket["z"].extend([c1[2], c2[2], None])
                elabel = _display_label(e, edgeLabelKey, e.get("role", ""))
                ehover = _hover(e, label=elabel or "Proof dependency", label_key=edgeLabelKey, fallback="Proof dependency")
                bucket["text"].extend([ehover, ehover, None])
                if showEdgeLabel and elabel:
                    label_x.append((c1[0] + c2[0]) * 0.5)
                    label_y.append((c1[1] + c2[1]) * 0.5)
                    label_z.append((c1[2] + c2[2]) * 0.5)
                    label_text.append(elabel)

            first = True
            for (this_color, this_width), bucket in edge_buckets.items():
                traces.append(go.Scatter3d(
                    x=bucket["x"], y=bucket["y"], z=bucket["z"],
                    mode="lines",
                    line=dict(color=this_color, width=this_width),
                    hoverinfo="skip" if not hover else None,
                    hovertext=bucket["text"],
                    hovertemplate="%{hovertext}<extra></extra>" if hover else None,
                    hoverlabel=dict(align="left", namelength=-1),
                    name="Proof dependencies",
                    legendgroup="proof_edges",
                    showlegend=bool(showLegend and first),
                ))
                first = False

            if showEdgeLabel and label_text:
                traces.append(go.Scatter3d(
                    x=label_x, y=label_y, z=label_z,
                    mode="text",
                    text=label_text,
                    textfont=dict(size=10),
                    hoverinfo="skip",
                    showlegend=False,
                ))

        if showNodes:
            buckets = {}
            for d in nodes:
                node_id = d["id"]
                if node_id not in coords:
                    continue
                ntype = _type(d.get(nodeTypeKey, d.get("kind", d.get("category", "node"))))
                color = _node_color(d)
                symbol = _node_symbol(d)
                buckets.setdefault((ntype, color, symbol), []).append(d)

            for (ntype, color, symbol), items in buckets.items():
                xs, ys, zs, labels, hovers, sizes = [], [], [], [], [], []
                for d in items:
                    c = coords[d["id"]]
                    xs.append(c[0]); ys.append(c[1]); zs.append(c[2])
                    this_label = _display_label(d, nodeLabelKey, d["id"])
                    labels.append(this_label)
                    hovers.append(_hover(d, label=this_label, label_key=nodeLabelKey, fallback=str(d["id"])))
                    sizes.append(_node_size(d))
                traces.append(go.Scatter3d(
                    x=xs, y=ys, z=zs,
                    mode="markers+text" if showNodeLabel else "markers",
                    marker=dict(
                        size=sizes,
                        color=color,
                        symbol=symbol,
                        line=dict(color="rgba(0,0,0,0.35)", width=1),
                    ),
                    text=labels if showNodeLabel else None,
                    textposition="top center",
                    textfont=dict(size=10),
                    hoverinfo="skip" if not hover else None,
                    hovertext=hovers,
                    hovertemplate="%{hovertext}<extra></extra>" if hover else None,
                    hoverlabel=dict(align="left", namelength=-1),
                    name=ntype.replace("_", " ").title(),
                    legendgroup="proof_nodes_" + ntype,
                    showlegend=showLegend,
                ))

        return traces

    @staticmethod
    def FigureByProofGraph(
        proofGraphData=None,
        result=None,
        triple=None,
        graph=None,
        layout: str = "tree",
        title: str = "Proof Graph",
        width: int = 950,
        height: int = 700,
        backgroundColor="rgba(0,0,0,0)",
        showNodeLabel: bool = True,
        showEdgeLabel: bool = False,
        showLegend: bool = True,
        xAxis: bool = False,
        yAxis: bool = False,
        zAxis: bool = False,
        marginLeft: int = 0,
        marginRight: int = 0,
        marginTop: int = 40,
        marginBottom: int = 0,
        silent: bool = False,
        **kwargs,
    ):
        """
        Creates a Plotly figure for a proof graph.

        Parameters
        ----------
        proofGraphData : dict or topologicpy.TGraph , optional
            The proof graph data or proof TGraph.
        result : object , optional
            An inference result returned by Reasoner.Infer or TGraph.InferOntology.
        triple : tuple or list , optional
            The inferred triple to explain.
        graph : object , optional
            Optional source graph passed to Reasoner when deriving proof data.
        layout : str , optional
            Layout type: "tree", "radial", or "force".
        title : str , optional
            Figure title.
        width, height : int , optional
            Figure size in pixels.
        backgroundColor : str or list , optional
            Figure background colour.
        showNodeLabel, showEdgeLabel, showLegend : bool , optional
            Display controls.
        xAxis, yAxis, zAxis : bool , optional
            Axis visibility controls.
        marginLeft, marginRight, marginTop, marginBottom : int , optional
            Figure margins.
        silent : bool , optional
            If True, suppress warnings.
        **kwargs : dict
            Additional arguments passed to Plotly.DataByProofGraph.

        Returns
        -------
        plotly.graph_objects.Figure or None
            The resulting figure.
        """
        try:
            from topologicpy.Color import Color
            background = Color.AnyToHex(backgroundColor)
        except Exception:
            background = backgroundColor

        data = Plotly.DataByProofGraph(
            proofGraphData=proofGraphData,
            result=result,
            triple=triple,
            graph=graph,
            layout=layout,
            showNodeLabel=showNodeLabel,
            showEdgeLabel=showEdgeLabel,
            showLegend=showLegend,
            silent=silent,
            **kwargs,
        )

        if data is None:
            return None

        # Prefer the established Plotly.py figure pathway so proof graph
        # visualisation behaves like the rest of TopologicPy's Plotly output.
        # Fall back to a direct go.Figure when Topologic geometry helpers are
        # unavailable, for example in a lightweight reasoning-only environment.
        try:
            figure = Plotly.FigureByData(
                data=data,
                width=width,
                height=height,
                xAxis=xAxis,
                yAxis=yAxis,
                zAxis=zAxis,
                axisSize=1,
                backgroundColor=background,
                marginLeft=marginLeft,
                marginRight=marginRight,
                marginTop=marginTop,
                marginBottom=marginBottom,
            )
        except Exception:
            figure = None

        if figure is None:
            figure = go.Figure(data=data)
            figure.update_layout(
                width=width,
                height=height,
                scene=dict(
                    xaxis=dict(visible=bool(xAxis)),
                    yaxis=dict(visible=bool(yAxis)),
                    zaxis=dict(visible=bool(zAxis)),
                    aspectmode="data",
                ),
                paper_bgcolor=background,
                plot_bgcolor=background,
                margin=dict(l=marginLeft, r=marginRight, t=marginTop, b=marginBottom),
            )

        figure.update_layout(title=title, showlegend=showLegend)
        figure.update_xaxes(showgrid=False, zeroline=False, visible=False)
        figure.update_yaxes(showgrid=False, zeroline=False, visible=False)
        return figure

    @staticmethod
    def ProofGraphHTML(
        proofGraphData=None,
        result=None,
        triple=None,
        graph=None,
        path: str = "proof_graph.html",
        layout: str = "tree",
        title: str = "Proof Graph",
        includePlotlyJS: str = "cdn",
        autoOpen: bool = False,
        silent: bool = False,
        **kwargs,
    ):
        """
        Exports a proof graph visualisation to an HTML file.

        Parameters
        ----------
        proofGraphData : dict or topologicpy.TGraph , optional
            The proof graph data or proof TGraph.
        result : object , optional
            An inference result returned by Reasoner.Infer or TGraph.InferOntology.
        triple : tuple or list , optional
            The inferred triple to explain.
        graph : object , optional
            Optional source graph passed to Reasoner when deriving proof data.
        path : str , optional
            The output HTML path. Default is "proof_graph.html".
        layout : str , optional
            Layout type: "tree", "radial", or "force".
        title : str , optional
            Figure title.
        includePlotlyJS : str or bool , optional
            Passed to plotly.io.write_html. Common values are "cdn", True, or
            False. Default is "cdn".
        autoOpen : bool , optional
            If True, open the HTML file after writing it.
        silent : bool , optional
            If True, suppress warnings.
        **kwargs : dict
            Additional arguments passed to Plotly.FigureByProofGraph.

        Returns
        -------
        str or None
            The output path if successful; otherwise None.
        """
        if not isinstance(path, str) or len(path.strip()) == 0:
            if not silent:
                print("Plotly.ProofGraphHTML - Error: The input path is invalid. Returning None.")
            return None

        figure = Plotly.FigureByProofGraph(
            proofGraphData=proofGraphData,
            result=result,
            triple=triple,
            graph=graph,
            layout=layout,
            title=title,
            silent=silent,
            **kwargs,
        )

        if figure is None:
            return None

        try:
            import plotly.io as pio
            pio.write_html(figure, file=path, include_plotlyjs=includePlotlyJS, auto_open=autoOpen, full_html=True)
            return path
        except Exception as exc:
            if not silent:
                print("Plotly.ProofGraphHTML - Error: Could not write the HTML file. Returning None.")
                print("Error:", exc)
            return None


    @staticmethod
    def vertexData(vertices, dictionaries=None, color="black", colorKey=None,
                   size=1.1, sizeKey=None, borderColor="black", borderWidth=0,
                   borderColorKey=None, borderWidthKey=None, labelKey=None,
                   showVertexLabel=False, vertexLabelFontSize=5, groupKey=None,
                   minGroup=None, maxGroup=None, groups=None,
                   legendLabel="Topology Vertices", legendGroup=1, legendRank=1,
                   showLegend=True, colorScale="Viridis"):
        """Creates Plotly vertex traces with source-correct labels, styles, and hover text."""
        from topologicpy.Color import Color

        if not vertices:
            return []
        dictionaries = list(dictionaries) if dictionaries is not None else []
        groups = list(groups) if groups is not None else []

        def as_float(value, default):
            try:
                number = float(value)
                return number if number == number else default
            except Exception:
                return default

        def as_hex(value, default="black"):
            return Plotly._color_to_hex(value, Plotly._color_to_hex(default, "black"))

        observed_groups = []
        if groupKey is not None:
            for d in dictionaries:
                value = Plotly._dictionary_value(d, groupKey, None)
                if value is not None and value not in observed_groups:
                    observed_groups.append(value)
        domain = groups if groups else observed_groups
        numeric_domain = []
        numeric = bool(domain)
        for item in domain:
            try:
                numeric_domain.append(float(item))
            except Exception:
                numeric = False
                break

        def group_color(value, default):
            if value is None:
                return default
            if numeric:
                try:
                    number = float(value)
                    lo = float(minGroup) if minGroup is not None else min(numeric_domain)
                    hi = float(maxGroup) if maxGroup is not None else max(numeric_domain)
                    if hi < lo:
                        lo, hi = hi, lo
                    if abs(hi - lo) <= 1.0e-15:
                        mapped = Color.ByValueInRange(0.5, minValue=0.0, maxValue=1.0, colorScale=colorScale)
                    else:
                        mapped = Color.ByValueInRange(max(lo, min(hi, number)), minValue=lo, maxValue=hi, colorScale=colorScale)
                    return as_hex(mapped, default)
                except Exception:
                    return default
            categorical = domain
            if categorical and value in categorical:
                index = categorical.index(value)
                hi = max(len(categorical) - 1, 1)
                return as_hex(Color.ByValueInRange(index, minValue=0, maxValue=hi, colorScale=colorScale), default)
            return default

        x, y, z = [], [], []
        sizes, labels, hovertexts, colors = [], [], [], []
        border_colors, border_sizes, border_widths = [], [], []
        default_color = as_hex(color)
        default_border = as_hex(borderColor)
        default_size = max(as_float(size, 1.1), 0.1)
        default_border_width = max(as_float(borderWidth, 0.0), 0.0)
        digits = len(str(max(1, len(vertices))))
        exclude = [colorKey, sizeKey, borderColorKey, borderWidthKey, groupKey]

        for index, vertex in enumerate(vertices):
            try:
                point = [float(vertex[0]), float(vertex[1]), float(vertex[2])]
            except Exception:
                continue
            d = dictionaries[index] if index < len(dictionaries) else None
            fallback = "Vertex " + str(index + 1).zfill(digits)
            label_value = Plotly._dictionary_value(d, labelKey, None) if labelKey else None
            label = fallback if label_value in (None, "") else str(label_value)
            hover = Plotly._format_hover(d, label=label, fallback=fallback, labelKey=labelKey, excludeKeys=exclude)

            this_size = max(as_float(Plotly._dictionary_value(d, sizeKey, default_size), default_size), 0.1) if sizeKey else default_size
            this_color = as_hex(Plotly._dictionary_value(d, colorKey, default_color), default_color) if colorKey else default_color
            if groupKey is not None:
                this_color = group_color(Plotly._dictionary_value(d, groupKey, None), this_color)
            this_border = as_hex(Plotly._dictionary_value(d, borderColorKey, default_border), default_border) if borderColorKey else default_border
            this_border_width = max(as_float(Plotly._dictionary_value(d, borderWidthKey, default_border_width), default_border_width), 0.0) if borderWidthKey else default_border_width

            x.append(point[0]); y.append(point[1]); z.append(point[2])
            labels.append(label); hovertexts.append(hover); sizes.append(this_size); colors.append(this_color)
            border_colors.append(this_border); border_widths.append(this_border_width)
            border_sizes.append(this_size + this_border_width * 2.0 if this_border_width > 0 else 0.0)

        if not x:
            return []

        traces = []
        if any(w > 0 for w in border_widths):
            traces.append(go.Scatter3d(
                x=x, y=y, z=z, mode="markers",
                marker=dict(color=border_colors, size=border_sizes, symbol="circle", opacity=1, line=dict(width=0), sizemode="diameter"),
                name=legendLabel, showlegend=False, hoverinfo="skip",
                legendgroup=str(legendGroup), legendrank=legendRank,
            ))

        traces.append(go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers+text" if showVertexLabel else "markers",
            marker=dict(color=colors, size=sizes, symbol="circle", opacity=1, line=dict(width=0), sizemode="diameter"),
            name=legendLabel, showlegend=bool(showLegend),
            legendgroup=str(legendGroup), legendrank=legendRank,
            text=labels if showVertexLabel else None,
            textfont=dict(size=max(1, int(vertexLabelFontSize))),
            hovertext=hovertexts,
            hovertemplate="%{hovertext}<extra></extra>",
            hoverlabel=dict(align="left", namelength=-1),
        ))
        return traces

    @staticmethod
    def edgeData(vertices, edges, dictionaries=None, color="black", colorKey=None,
                 width=1, widthKey=None, dash=False, dashKey=None, directed=False,
                 arrowSize=0.1, arrowSizeKey=None, labelKey=None,
                 showEdgeLabel=False, groupKey=None, minGroup=None, maxGroup=None,
                 groups=None, legendLabel="Topology Edges", legendGroup=2,
                 legendRank=2, showLegend=True, colorScale="Viridis"):
        """Creates Plotly line traces from indexed straight or sampled curved edges."""
        import math
        from topologicpy.Color import Color

        if vertices is None or edges is None:
            return []
        groups = list(groups) if groups is not None else []
        dictionaries = list(dictionaries) if dictionaries is not None else []

        def as_float(value, default):
            try:
                number = float(value)
                return number if math.isfinite(number) else default
            except Exception:
                return default

        def as_bool(value, default=False):
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            if isinstance(value, str):
                return value.strip().lower() in ("true", "1", "yes", "y", "t", "on")
            return default

        def as_hex(value, default="black"):
            return Plotly._color_to_hex(value, Plotly._color_to_hex(default, "black"))

        observed_groups = []
        if groupKey is not None:
            for d in dictionaries:
                value = Plotly._dictionary_value(d, groupKey, None)
                if value is not None and value not in observed_groups:
                    observed_groups.append(value)
        domain = groups if groups else observed_groups
        numeric_domain = []
        numeric = bool(domain)
        for item in domain:
            try:
                numeric_domain.append(float(item))
            except Exception:
                numeric = False
                break

        def group_color(value, default):
            if value is None:
                return default
            if numeric:
                try:
                    number = float(value)
                    lo = float(minGroup) if minGroup is not None else min(numeric_domain)
                    hi = float(maxGroup) if maxGroup is not None else max(numeric_domain)
                    if hi < lo:
                        lo, hi = hi, lo
                    if abs(hi - lo) <= 1.0e-15:
                        mapped = Color.ByValueInRange(0.5, minValue=0.0, maxValue=1.0, colorScale=colorScale)
                    else:
                        mapped = Color.ByValueInRange(max(lo, min(hi, number)), minValue=lo, maxValue=hi, colorScale=colorScale)
                    return as_hex(mapped, default)
                except Exception:
                    return default
            if domain and value in domain:
                index = domain.index(value)
                hi = max(len(domain)-1, 1)
                return as_hex(Color.ByValueInRange(index, minValue=0, maxValue=hi, colorScale=colorScale), default)
            return default

        def direction(points):
            for a, b in zip(reversed(points[:-1]), reversed(points[1:])):
                u, v, w = b[0]-a[0], b[1]-a[1], b[2]-a[2]
                length = math.sqrt(u*u + v*v + w*w)
                if length > 1.0e-12:
                    return (u/length, v/length, w/length)
            return None

        buckets, arrow_buckets = {}, {}
        label_x, label_y, label_z, label_text, label_hover = [], [], [], [], []
        default_color = as_hex(color)
        digits = len(str(max(1, len(edges))))
        exclude = [colorKey, widthKey, dashKey, arrowSizeKey, groupKey]

        for index, edge in enumerate(edges):
            if not isinstance(edge, (list, tuple)) or len(edge) < 2:
                continue
            points = []
            for vertex_index in edge:
                try:
                    p = vertices[int(vertex_index)]
                    p = [float(p[0]), float(p[1]), float(p[2])]
                except Exception:
                    continue
                if not points or p != points[-1]:
                    points.append(p)
            if len(points) < 2:
                continue

            d = dictionaries[index] if index < len(dictionaries) else None
            fallback = "Edge " + str(index + 1).zfill(digits)
            label_value = Plotly._dictionary_value(d, labelKey, None) if labelKey else None
            label = fallback if label_value in (None, "") else str(label_value)
            hover = Plotly._format_hover(d, label=label, fallback=fallback, labelKey=labelKey, excludeKeys=exclude)

            this_color = as_hex(Plotly._dictionary_value(d, colorKey, default_color), default_color) if colorKey else default_color
            if groupKey is not None:
                this_color = group_color(Plotly._dictionary_value(d, groupKey, None), this_color)
            this_width = max(0.1, as_float(Plotly._dictionary_value(d, widthKey, width), width) if widthKey else as_float(width, 1.0))
            this_dash = as_bool(Plotly._dictionary_value(d, dashKey, dash), dash) if dashKey else bool(dash)
            this_arrow_size = max(0.0, as_float(Plotly._dictionary_value(d, arrowSizeKey, arrowSize), arrowSize) if arrowSizeKey else as_float(arrowSize, 0.1))

            key = (this_color, this_width, this_dash)
            bucket = buckets.setdefault(key, {"x": [], "y": [], "z": [], "hover": []})
            bucket["x"].extend([p[0] for p in points] + [None])
            bucket["y"].extend([p[1] for p in points] + [None])
            bucket["z"].extend([p[2] for p in points] + [None])
            bucket["hover"].extend([hover] * len(points) + [None])

            midpoint = Plotly._polyline_midpoint(points)
            if showEdgeLabel and midpoint is not None:
                label_x.append(midpoint[0]); label_y.append(midpoint[1]); label_z.append(midpoint[2])
                label_text.append(label); label_hover.append(hover)

            if directed and this_arrow_size > 0:
                vector = direction(points)
                if vector is not None:
                    end = points[-1]
                    akey = (this_color, this_arrow_size)
                    ab = arrow_buckets.setdefault(akey, {"x": [], "y": [], "z": [], "u": [], "v": [], "w": []})
                    ab["x"].append(end[0]); ab["y"].append(end[1]); ab["z"].append(end[2])
                    ab["u"].append(vector[0]); ab["v"].append(vector[1]); ab["w"].append(vector[2])

        traces = []
        first = True
        for (this_color, this_width, this_dash), bucket in buckets.items():
            traces.append(go.Scatter3d(
                x=bucket["x"], y=bucket["y"], z=bucket["z"],
                mode="lines+markers" if this_dash else "lines",
                line=dict(color=this_color, width=this_width, dash="dash" if this_dash else "solid"),
                marker=dict(
                    color=this_color,
                    size=max(1.0, float(this_width)),
                    opacity=1 if this_dash else 0,
                ),
                name=legendLabel, showlegend=bool(showLegend and first),
                legendgroup=str(legendGroup), legendrank=legendRank,
                hovertext=bucket["hover"], hovertemplate="%{hovertext}<extra></extra>",
                hoverlabel=dict(align="left", namelength=-1), connectgaps=False,
            ))
            first = False

        if showEdgeLabel and label_text:
            traces.append(go.Scatter3d(
                x=label_x, y=label_y, z=label_z, mode="text", text=label_text,
                textfont=dict(size=10), hovertext=label_hover,
                hovertemplate="%{hovertext}<extra></extra>", hoverlabel=dict(align="left", namelength=-1),
                showlegend=False,
            ))

        for (this_color, this_arrow_size), bucket in arrow_buckets.items():
            traces.append(go.Cone(
                x=bucket["x"], y=bucket["y"], z=bucket["z"],
                u=bucket["u"], v=bucket["v"], w=bucket["w"],
                sizemode="absolute", sizeref=this_arrow_size, anchor="tip",
                showscale=False, colorscale=[[0, this_color], [1, this_color]],
                showlegend=False, hoverinfo="skip",
            ))
        return traces

    @staticmethod
    def DataByTopology(topology,
                       showVertices=True,
                       vertexSize=2.8,
                       vertexSizeKey=None,
                       vertexColor="black",
                       vertexColorKey=None,
                       vertexLabelKey=None,
                       vertexBorderColor="black",
                       vertexBorderWidth=0,
                       vertexBorderColorKey=None,
                       vertexBorderWidthKey=None,
                       showVertexLabel=False,
                       vertexLabelFontSize=5,
                       vertexGroupKey=None,
                       vertexGroups=None,
                       vertexMinGroup=None,
                       vertexMaxGroup=None,
                       showVertexLegend=False,
                       vertexLegendLabel="Topology Vertices",
                       vertexLegendRank=1,
                       vertexLegendGroup=1,
                       directed=False,
                       arrowSize=0.1,
                       arrowSizeKey=None,
                       showEdges=True,
                       edgeWidth=1,
                       edgeWidthKey=None,
                       edgeColor="black",
                       edgeColorKey=None,
                       edgeDash=False,
                       edgeDashKey=None,
                       edgeLabelKey=None,
                       showEdgeLabel=False,
                       edgeGroupKey=None,
                       edgeGroups=None,
                       edgeMinGroup=None,
                       edgeMaxGroup=None,
                       showEdgeLegend=False,
                       edgeLegendLabel="Topology Edges",
                       edgeLegendRank=2,
                       edgeLegendGroup=2,
                       showFaces=True,
                       faceOpacity=0.5,
                       faceOpacityKey=None,
                       faceColor="#FAFAFA",
                       faceColorKey=None,
                       faceLabelKey=None,
                       faceGroupKey=None,
                       faceGroups=None,
                       faceMinGroup=None,
                       faceMaxGroup=None,
                       showFaceLegend=False,
                       faceLegendLabel="Topology Faces",
                       faceLegendRank=3,
                       faceLegendGroup=3,
                       intensityKey=None,
                       intensities=None,
                       material="default",
                       materialKey=None,
                       flatShading=False,
                       ambient=None,
                       ambientKey=None,
                       diffuse=None,
                       diffuseKey=None,
                       specular=None,
                       specularKey=None,
                       roughness=None,
                       roughnessKey=None,
                       colorScale="viridis",
                       mantissa=6,
                       tolerance=0.0001,
                       silent=False):
        """Creates Plotly data from a Topologic topology.

        Curved edges are sampled for display and faces are tessellated from their
        actual geometry. Visual dictionary keys are resolved on each source
        sub-topology, with the root topology dictionary acting only as a fallback.
        """
        import math
        from topologicpy.Color import Color
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print("Plotly.DataByTopology - Error: The input is not a valid topology. Returning None.")
            return None
        if not Plotly._plotly_available(silent=silent):
            return None

        vertexGroups = list(vertexGroups) if vertexGroups is not None else []
        edgeGroups = list(edgeGroups) if edgeGroups is not None else []
        faceGroups = list(faceGroups) if faceGroups is not None else []
        intensities = list(intensities) if intensities is not None else []
        curve_samples = 32
        face_quality = "medium"

        try:
            tolerance = max(abs(float(tolerance)), 1.0e-12)
        except Exception:
            tolerance = 0.0001

        def get(kind):
            singular = {"Vertices": "Vertex", "Edges": "Edge", "Faces": "Face"}.get(kind, kind[:-1] if kind.endswith("s") else kind)
            if Topology.IsInstance(topology, singular):
                return [topology]
            method = getattr(Topology, kind, None)
            if method is None:
                return []
            try:
                result = method(topology, silent=True)
            except TypeError:
                try:
                    result = method(topology)
                except Exception:
                    result = []
            except Exception:
                result = []
            return result if isinstance(result, list) else []

        def dictionary(tp):
            try:
                return Topology.Dictionary(tp, silent=True)
            except TypeError:
                try:
                    return Topology.Dictionary(tp)
                except Exception:
                    return None
            except Exception:
                return None

        def finite_number(value, default=None, lo=None, hi=None):
            try:
                number = float(value)
                if not math.isfinite(number):
                    return default
                if lo is not None:
                    number = max(float(lo), number)
                if hi is not None:
                    number = min(float(hi), number)
                return number
            except Exception:
                return default

        root_dictionary = dictionary(topology)
        data = []

        if showVertices:
            vertex_coordinates, vertex_dictionaries = [], []
            for vertex in get("Vertices"):
                point = Plotly._vertex_coordinates(vertex, mantissa=mantissa)
                if point is None:
                    continue
                vertex_coordinates.append(point)
                vertex_dictionaries.append(dictionary(vertex))
            data.extend(Plotly.vertexData(
                vertex_coordinates, dictionaries=vertex_dictionaries,
                color=vertexColor, colorKey=vertexColorKey,
                size=vertexSize, sizeKey=vertexSizeKey,
                borderColor=vertexBorderColor, borderWidth=vertexBorderWidth,
                borderColorKey=vertexBorderColorKey, borderWidthKey=vertexBorderWidthKey,
                labelKey=vertexLabelKey, showVertexLabel=showVertexLabel,
                vertexLabelFontSize=vertexLabelFontSize,
                groupKey=vertexGroupKey, minGroup=vertexMinGroup, maxGroup=vertexMaxGroup,
                groups=vertexGroups, legendLabel=vertexLegendLabel,
                legendGroup=vertexLegendGroup, legendRank=vertexLegendRank,
                showLegend=showVertexLegend, colorScale=colorScale,
            ))

        if showEdges:
            render_vertices, render_edges, edge_dictionaries = [], [], []
            for edge in get("Edges"):
                points = Plotly._edge_render_points(edge, samples=curve_samples, mantissa=mantissa, tolerance=tolerance)
                if len(points) < 2:
                    continue
                base = len(render_vertices)
                render_vertices.extend(points)
                render_edges.append(list(range(base, base + len(points))))
                edge_dictionaries.append(dictionary(edge))
            data.extend(Plotly.edgeData(
                render_vertices, render_edges, dictionaries=edge_dictionaries,
                color=edgeColor, colorKey=edgeColorKey,
                width=edgeWidth, widthKey=edgeWidthKey,
                dash=edgeDash, dashKey=edgeDashKey,
                directed=directed, arrowSize=arrowSize, arrowSizeKey=arrowSizeKey,
                labelKey=edgeLabelKey, showEdgeLabel=showEdgeLabel,
                groupKey=edgeGroupKey, minGroup=edgeMinGroup, maxGroup=edgeMaxGroup,
                groups=edgeGroups, legendLabel=edgeLegendLabel,
                legendGroup=edgeLegendGroup, legendRank=edgeLegendRank,
                showLegend=showEdgeLegend, colorScale=colorScale,
            ))

        if not showFaces:
            return data

        source_faces = get("Faces")
        if not source_faces:
            return data
        face_dicts = [dictionary(face) for face in source_faces]

        source_intensity_samples = []
        if intensityKey is not None:
            for vertex in get("Vertices"):
                point = Plotly._vertex_coordinates(vertex, mantissa=mantissa)
                value = finite_number(Plotly._dictionary_value(dictionary(vertex), intensityKey, None), None)
                if point is not None and value is not None:
                    source_intensity_samples.append((point, value))

        observed_groups = []
        if faceGroupKey is not None:
            for d in face_dicts:
                value = Plotly._dictionary_value(d, faceGroupKey, None)
                if value is not None and value not in observed_groups:
                    observed_groups.append(value)
        group_domain = faceGroups if faceGroups else observed_groups
        numeric_domain = []
        numeric_groups = bool(group_domain)
        for value in group_domain:
            try:
                numeric_domain.append(float(value))
            except Exception:
                numeric_groups = False
                break

        try:
            default_face_color = Color.AnyToHex(faceColor)
        except Exception:
            default_face_color = "#FAFAFA"

        def group_color(group, default):
            if group is None:
                return default
            try:
                if numeric_groups:
                    value = float(group)
                    lo = float(faceMinGroup) if faceMinGroup is not None else min(numeric_domain)
                    hi = float(faceMaxGroup) if faceMaxGroup is not None else max(numeric_domain)
                    if hi < lo:
                        lo, hi = hi, lo
                    if abs(hi - lo) <= 1.0e-15:
                        mapped = Color.ByValueInRange(0.5, minValue=0.0, maxValue=1.0, colorScale=colorScale)
                    else:
                        mapped = Color.ByValueInRange(max(lo, min(hi, value)), minValue=lo, maxValue=hi, colorScale=colorScale)
                    return Plotly._color_to_hex(mapped, default)
                if group_domain and group in group_domain:
                    index = group_domain.index(group)
                    mapped = Color.ByValueInRange(index, minValue=0, maxValue=max(1, len(group_domain)-1), colorScale=colorScale)
                    return Plotly._color_to_hex(mapped, default)
            except Exception:
                pass
            return default

        def inherited_value(face_dictionary, key, fallback):
            if key is None:
                return fallback
            value = Plotly._dictionary_value(face_dictionary, key, None)
            if value is None:
                value = Plotly._dictionary_value(root_dictionary, key, None)
            return fallback if value is None else value

        presets = {
            "chalk": (1.0, 0.4, 0.0, 1.0),
            "concrete": (0.85, 0.75, 0.05, 0.9),
            "eggshell": (0.65, 0.85, 0.25, 0.45),
            "glossy": (0.5, 0.9, 0.6, 0.1),
            "matte": (0.9, 0.7, 0.0, 1.0),
            "metallic": (0.3, 0.8, 0.9, 0.2),
            "plastic": (0.6, 0.9, 0.2, 0.4),
        }

        def face_style(face_dictionary):
            this_color = default_face_color
            if faceColorKey is not None:
                candidate = inherited_value(face_dictionary, faceColorKey, None)
                if candidate is not None:
                    try:
                        this_color = Plotly._color_to_hex(candidate, this_color)
                    except Exception:
                        pass
            if faceGroupKey is not None:
                this_color = group_color(Plotly._dictionary_value(face_dictionary, faceGroupKey, None), this_color)

            this_opacity = finite_number(inherited_value(face_dictionary, faceOpacityKey, faceOpacity), finite_number(faceOpacity, 0.5), 0.0, 1.0)

            material_name = str(inherited_value(face_dictionary, materialKey, material) or "default").lower()
            base = presets.get(material_name)
            lighting = {"facenormalsepsilon": 0}
            if base is not None:
                lighting.update(ambient=base[0], diffuse=base[1], specular=base[2], roughness=base[3])

            parameters = (
                ("ambient", ambient, ambientKey, 0.0, 1.0),
                ("diffuse", diffuse, diffuseKey, 0.0, 1.0),
                ("specular", specular, specularKey, 0.0, 2.0),
                ("roughness", roughness, roughnessKey, 0.0, 1.0),
            )
            for name, explicit, key, lo, hi in parameters:
                candidate = inherited_value(face_dictionary, key, explicit)
                number = finite_number(candidate, None, lo, hi)
                if number is not None:
                    lighting[name] = number

            lighting_key = tuple(sorted((k, float(v)) for k, v in lighting.items()))
            return this_color, this_opacity, lighting, lighting_key

        scale_values = []
        for value in intensities:
            number = finite_number(value, None)
            if number is not None:
                scale_values.append(number)
        if not scale_values:
            scale_values = [item[1] for item in source_intensity_samples]
        cmin = min(scale_values) if scale_values else None
        cmax = max(scale_values) if scale_values else None
        if cmin is not None and cmax == cmin:
            cmax = cmin + 1.0

        buckets = {}
        face_digits = len(str(max(1, len(source_faces))))
        hover_exclude = [faceColorKey, faceOpacityKey, faceGroupKey, materialKey, ambientKey, diffuseKey, specularKey, roughnessKey]

        for face_index, (face, face_dictionary) in enumerate(zip(source_faces, face_dicts)):
            fallback = "Face " + str(face_index + 1).zfill(face_digits)
            label_value = Plotly._dictionary_value(face_dictionary, faceLabelKey, None) if faceLabelKey else None
            label = fallback if label_value in (None, "") else str(label_value)
            hover = Plotly._format_hover(face_dictionary, label=label, fallback=fallback, labelKey=faceLabelKey, excludeKeys=hover_exclude)
            render_mesh = Plotly._face_render_mesh(face, quality=face_quality, mantissa=mantissa, tolerance=tolerance)
            if not render_mesh:
                if not silent:
                    print(f"Plotly.DataByTopology - Warning: Could not tessellate {fallback}. Skipping face.")
                continue
            local_vertices = render_mesh.get("vertices", [])
            local_faces = render_mesh.get("faces", [])
            this_color, this_opacity, lighting, lighting_key = face_style(face_dictionary)
            bucket_key = (round(float(this_opacity), 12), lighting_key)
            bucket = buckets.setdefault(bucket_key, {
                "vertices": [], "faces": [], "facecolors": [], "hover": [], "intensity": [],
                "opacity": this_opacity, "lighting": lighting,
            })

            for triangle in local_faces:
                try:
                    coords = [local_vertices[int(triangle[q])] for q in range(3)]
                    coords = [[float(p[0]), float(p[1]), float(p[2])] for p in coords]
                except Exception:
                    continue
                base = len(bucket["vertices"])
                bucket["vertices"].extend(coords)
                bucket["faces"].append([base, base + 1, base + 2])
                bucket["facecolors"].append(this_color)
                bucket["hover"].extend([hover, hover, hover])

                if source_intensity_samples:
                    for point in coords:
                        nearest = min(source_intensity_samples, key=lambda item: (
                            (point[0]-item[0][0])**2 + (point[1]-item[0][1])**2 + (point[2]-item[0][2])**2
                        ))
                        bucket["intensity"].append(nearest[1])

        first = True
        for bucket in buckets.values():
            if not bucket["faces"]:
                continue
            vertices = bucket["vertices"]
            faces = bucket["faces"]
            intensity_values = bucket["intensity"] if source_intensity_samples else None
            data.append(go.Mesh3d(
                x=[p[0] for p in vertices], y=[p[1] for p in vertices], z=[p[2] for p in vertices],
                i=[f[0] for f in faces], j=[f[1] for f in faces], k=[f[2] for f in faces],
                name=faceLegendLabel,
                showlegend=bool(showFaceLegend and first),
                legendgroup=str(faceLegendGroup), legendrank=faceLegendRank,
                color=default_face_color,
                facecolor=None if intensity_values is not None else bucket["facecolors"],
                intensity=intensity_values,
                intensitymode="vertex" if intensity_values is not None else None,
                colorscale=Plotly.ColorScale(colorScale), cmin=cmin, cmax=cmax,
                opacity=bucket["opacity"], flatshading=bool(flatShading), lighting=bucket["lighting"],
                hovertext=bucket["hover"], hovertemplate="%{hovertext}<extra></extra>",
                hoverlabel=dict(align="left", namelength=-1), showscale=False,
            ))
            first = False

        return data
    
    @staticmethod
    def FigureByConfusionMatrix(matrix,
            categories=None,
            minValue=None,
            maxValue=None,
            title="Confusion Matrix",
            xTitle = "Actual Categories",
            yTitle = "Predicted Categories",
            width=950,
            height=500,
            showScale = True,
            colorScale='viridis',
            colorSamples=10,
            backgroundColor='rgba(0,0,0,0)',
            marginLeft=0,
            marginRight=0,
            marginTop=40,
            marginBottom=0,
            baseFontSize = 16,
            tickFontSize = 14,
            titleFontSize = 22,
            axisTitleFontSize = 16,
            annotationFontSize = 18,
            grayScale = False):
        """
        Returns a Plotly Figure of the input confusion matrix. Actual categories are displayed on the X-Axis,
        Predicted categories are displayed on the Y-Axis.

        Parameters
        ----------
        matrix : list or numpy.array
            The matrix to display.
        categories : list
            The list of categories to use on the X and Y axes.
        minValue : float , optional
            The desired minimum value to use for the color scale. If set to None, the minimum value found in the input matrix will be used.
        maxValue : float , optional
            The desired maximum value to use for the color scale. If set to None, the maximum value found in the input matrix will be used.
        title : str , optional
            The desired title to display. Default is "Confusion Matrix".
        xTitle : str , optional
            The desired X-axis title to display. Default is "Actual Categories".
        yTitle : str , optional
            The desired Y-axis title to display. Default is "Predicted Categories".
        width : int , optional
            The desired width of the figure. Default is 950.
        height : int , optional
            The desired height of the figure. Default is 500.
        showScale : bool , optional
            If set to True, a color scale is shown on the right side of the figure. Default is True.
        colorScale : str , optional
            The desired type of plotly color scales to use (e.g. "Viridis", "Plasma"). Default is "Viridis".
        colorSamples : int , optional
            The number of discrete color samples to use for displaying the data. Default is 10.
        backgroundColor : list or str , optional
            The desired background color (see docstring above). Default is transparent.
        marginLeft, marginRight, marginTop, marginBottom : int , optional
            Plot margins in pixels.
        baseFontSize : int , optional
            The base font size. Default is 16.
        tickFontSize : int , optional
            The tick font size. Default is 14.
        titleFontSize : int , optional
            The title font size. Default is 22.
        axisTitleFontSize : int , optional
            The axis title font size. Default is 16.
        annotationFontSize : int , optional
            The annotation font size. Default is 18.
        grayScale : bool , optional
            If set to True, the figure is rendered in grayscale. Default is False.

        Returns
        -------
        plotly.graph_objects.Figure
            The created plotly figure.
        """
        import warnings
        import numpy as np

        # Local imports (TopologicPy style)
        from topologicpy.Color import Color

        # --- Validate matrix
        if not isinstance(matrix, (list, np.ndarray)):
            warnings.warn("Plotly.FigureByConfusionMatrix - Error: The input matrix is not a list or numpy array. Returning None.")
            return None

        try:
            m = np.asarray(matrix, dtype=float)
        except Exception:
            warnings.warn("Plotly.FigureByConfusionMatrix - Error: The matrix must contain numeric values. Returning None.")
            return None
        if m.ndim != 2 or m.shape[0] == 0 or m.shape[1] == 0:
            warnings.warn("Plotly.FigureByConfusionMatrix - Error: The input matrix must be a non-empty 2D matrix. Returning None.")
            return None

        n_rows, n_cols = int(m.shape[0]), int(m.shape[1])

        # --- Defensive categories handling (avoid mutable-default pitfalls + mismatches)
        cats = list(categories) if categories is not None else []
        if len(cats) == 0:
            # Default category names if none provided
            cats = [str(i) for i in range(max(n_rows, n_cols))]
        else:
            cats = [str(c) for c in cats]

        # Confusion matrices should be square; if not, handle gracefully.
        # Make sure we have at least max(n_rows, n_cols) labels.
        needed = max(n_rows, n_cols)
        if len(cats) < needed:
            cats = cats + [str(i) for i in range(len(cats), needed)]
        elif len(cats) > needed:
            cats = cats[:needed]

        # --- Derive min/max if needed
        finite_values = m[np.isfinite(m)]
        if minValue is None:
            minValue = float(np.min(finite_values)) if finite_values.size else 0.0
        if maxValue is None:
            maxValue = float(np.max(finite_values)) if finite_values.size else 1.0
        if maxValue < minValue:
            minValue, maxValue = maxValue, minValue

        # --- Build the figure using existing robust matrix plotter
        figure = Plotly.FigureByMatrix(
            m.tolist(),
            xCategories=cats[:n_cols],
            minValue=minValue,
            maxValue=maxValue,
            title=title,
            xTitle=xTitle,
            yTitle=yTitle,
            width=width,
            height=height,
            showScale=showScale,
            colorScale=Plotly.ColorScale(colorScale),
            colorSamples=colorSamples,
            backgroundColor=Plotly._color_to_hex(backgroundColor),
            marginLeft=marginLeft,
            marginRight=marginRight,
            marginTop=marginTop,
            marginBottom=marginBottom,
            baseFontSize = baseFontSize,
            tickFontSize = tickFontSize,
            titleFontSize = titleFontSize,
            axisTitleFontSize = axisTitleFontSize,
            annotationFontSize = annotationFontSize,
            grayscale = grayScale
        )

        # --- Enforce correct y-axis order (confusion matrix convention)
        figure.update_layout(yaxis={"autorange": "reversed"})

        # ------------------------------------------------------------------
        # Improve size + clarity (ticks, titles, annotations, colorbar)
        # ------------------------------------------------------------------
        # Global font sizing
        base_font = baseFontSize
        tick_font = tickFontSize
        title_font = titleFontSize
        axis_title_font = axisTitleFontSize
        annot_font = annotationFontSize

        # If many categories, rotate x tick labels for readability
        rotate_x = 0
        if len(cats) >= 8:
            rotate_x = 45
        if len(cats) >= 16:
            rotate_x = 60

        figure.update_layout(
            template="plotly_white",
            font=dict(size=base_font),
            title=dict(font=dict(size=title_font)),
        )

        figure.update_xaxes(
            tickfont=dict(size=tick_font),
            title_font=dict(size=axis_title_font),
            tickangle=rotate_x,
            tickmode="array",
            tickvals=list(range(n_cols)),
            ticktext=cats[:n_cols]
        )
        figure.update_yaxes(
            tickfont=dict(size=tick_font),
            title_font=dict(size=axis_title_font),
            tickmode="array",
            tickvals=list(range(n_rows)),
            ticktext=cats[:n_rows]
        )

        # Colorbar text sizing (if a heatmap with a colorbar exists)
        if getattr(figure, "data", None):
            for tr in figure.data:
                if hasattr(tr, "colorbar") and tr.colorbar is not None:
                    tr.colorbar.tickfont = dict(size=tick_font)
                    tr.colorbar.title = dict(font=dict(size=axis_title_font))

        # Increase annotation font size if FigureByMatrix generated annotations
        if hasattr(figure.layout, "annotations") and figure.layout.annotations:
            new_anns = []
            for a in figure.layout.annotations:
                a = a.to_plotly_json() if hasattr(a, "to_plotly_json") else dict(a)
                a_font = a.get("font", {}) or {}
                a_font["size"] = max(int(a_font.get("size", annot_font)), annot_font)
                a["font"] = a_font
                new_anns.append(a)
            figure.update_layout(annotations=new_anns)

        return figure
    
    @staticmethod
    def FigureByMatrix(matrix,
            xCategories=None,
            yCategories=None,
            minValue=None,
            maxValue=None,
            title="Matrix",
            xTitle="X Axis",
            yTitle="Y Axis",
            width=950,
            height=950,
            showScale=False,
            colorScale="gray",
            colorSamples=10,
            backgroundColor="rgba(0,0,0,0)",
            marginLeft=0,
            marginRight=0,
            marginTop=40,
            marginBottom=0,
            baseFontSize=16,
            tickFontSize=14,
            titleFontSize=22,
            axisTitleFontSize=16,
            annotationFontSize=18,
            grayscale=False,          # <-- grayscaleinput flag (used below)
            mantissa: int = 6):
        """
        Returns a Plotly Figure of the input matrix.

        Notes
        -----
        - Plots matrix values as provided (no implicit normalization).
        - If `grayscale` is True, the figure becomes publication-friendly:
        white background, black axis lines/ticks, grayscale colorscale, and
        annotation contrast tuned for grayscale.
        """
        import os
        import warnings

        import plotly.graph_objects as go
        import plotly.express as px
        from topologicpy.Color import Color

        try:
            import numpy as np
        except Exception:
            warnings.warn("Plotly.FigureByMatrix - Error: Could not import numpy. Please install numpy manually. Returning None.")
            return None

        if not isinstance(matrix, (list, np.ndarray)):
            warnings.warn("Plotly.FigureByMatrix - Error: The input matrix is not a list or numpy array. Returning None.")
            return None

        try:
            m = np.asarray(matrix, dtype=float)
        except Exception:
            warnings.warn("Plotly.FigureByMatrix - Error: The matrix must contain numeric values. Returning None.")
            return None
        if m.ndim != 2 or m.shape[0] == 0 or m.shape[1] == 0:
            warnings.warn("Plotly.FigureByMatrix - Error: The input matrix must be a non-empty 2D matrix. Returning None.")
            return None

        n_rows, n_cols = int(m.shape[0]), int(m.shape[1])

        # -----------------------------
        # Categories (safe + strings)
        # -----------------------------
        xCats = list(xCategories) if xCategories is not None else []
        yCats = list(yCategories) if yCategories is not None else []

        if len(xCats) == 0:
            xCats = [str(i) for i in range(n_cols)]
        else:
            xCats = [str(x) for x in xCats]
            if len(xCats) < n_cols:
                xCats += [str(i) for i in range(len(xCats), n_cols)]
            elif len(xCats) > n_cols:
                xCats = xCats[:n_cols]

        if len(yCats) == 0:
            yCats = [str(i) for i in range(n_rows)]
        else:
            yCats = [str(y) for y in yCats]
            if len(yCats) < n_rows:
                yCats += [str(i) for i in range(len(yCats), n_rows)]
            elif len(yCats) > n_rows:
                yCats = yCats[:n_rows]

        # -----------------------------
        # Min/Max (None-safe; allow 0)
        # -----------------------------
        finite_values = m[np.isfinite(m)]
        if minValue is None:
            minValue = float(np.min(finite_values)) if finite_values.size else 0.0
        else:
            try:
                minValue = float(minValue)
            except Exception:
                minValue = float(np.min(finite_values)) if finite_values.size else 0.0
        if maxValue is None:
            maxValue = float(np.max(finite_values)) if finite_values.size else 1.0
        else:
            try:
                maxValue = float(maxValue)
            except Exception:
                maxValue = float(np.max(finite_values)) if finite_values.size else 1.0
        if maxValue < minValue:
            minValue, maxValue = maxValue, minValue

        denom = (maxValue - minValue) if abs(maxValue - minValue) > 1.0e-15 else 1.0

        # -----------------------------
        # Grayscale "publication" mode
        # -----------------------------
        grayscaleInput = bool(grayscale)  # <-- this is the grayscaleinput flag the user asked for

        # For publication: prefer white paper, black text, black axis lines.
        if grayscaleInput:
            # Ignore provided backgroundColor for publication-friendly output
            paper_bg = "#FFFFFF"
            plot_bg = "#FFFFFF"
            template_name = "plotly_white"

            # Force a true grayscale scale. (Low=white, High=black)
            # Keep it continuous but discretized via `colorSamples` for consistent legend steps.
            forced_color_scale = "Greys"
            forced_samples = max(int(colorSamples), 2)
        else:
            paper_bg = Plotly._color_to_hex(backgroundColor)
            plot_bg = Plotly._color_to_hex(backgroundColor)
            template_name = "plotly_white"
            forced_color_scale = None
            forced_samples = None

        # -----------------------------
        # Build discrete colorscale
        # -----------------------------
        scale_name = forced_color_scale if grayscaleInput else colorScale
        base_scale = Plotly.ColorScale(scale_name)

        if isinstance(base_scale, str):
            nS = forced_samples if grayscaleInput else max(int(colorSamples), 2)
            samples = [i / max(nS - 1, 1) for i in range(nS)]
            cols = px.colors.sample_colorscale(base_scale, samples)
            colorscale = [[samples[i], cols[i]] for i in range(len(samples))]
        else:
            colorscale = base_scale

        # -----------------------------
        # Helpers: hex/rgb parse + interpolation + luminance
        # -----------------------------
        def _parse_rgb(s):
            s = str(s).strip()
            if s.startswith("#"):
                h = s.lstrip("#")
                if len(h) == 3:
                    h = "".join([c + c for c in h])
                r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
                return (r, g, b)
            if s.startswith("rgb"):
                inside = s[s.find("(") + 1:s.find(")")]
                parts = [p.strip() for p in inside.split(",")]
                r = int(float(parts[0])); g = int(float(parts[1])); b = int(float(parts[2]))
                return (r, g, b)
            hx = Color.AnyToHex(s)
            return _parse_rgb(hx)

        scale_pos = [float(p) for p, _ in colorscale]
        scale_rgb = [_parse_rgb(c) for _, c in colorscale]

        def _interp_color(t):
            if t <= scale_pos[0]:
                return scale_rgb[0]
            if t >= scale_pos[-1]:
                return scale_rgb[-1]
            for k in range(len(scale_pos) - 1):
                a, b = scale_pos[k], scale_pos[k + 1]
                if a <= t <= b:
                    u = 0.0 if b == a else (t - a) / (b - a)
                    r0, g0, b0 = scale_rgb[k]
                    r1, g1, b1 = scale_rgb[k + 1]
                    r = r0 + (r1 - r0) * u
                    g = g0 + (g1 - g0) * u
                    bb = b0 + (b1 - b0) * u
                    return (r, g, bb)
            return scale_rgb[-1]

        def _rel_luminance(rgb):
            # WCAG relative luminance from sRGB
            def f(c):
                c = float(c) / 255.0
                return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
            r, g, b = rgb
            return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)

        # -----------------------------
        # Annotations: robust contrast
        # -----------------------------
        annotations = []
        annot_font_size = annotationFontSize

        # In grayscale mode, use slightly more conservative threshold and subtler backgrounds.
        lum_threshold = 0.60 if grayscaleInput else 0.55
        bg_light = "rgba(255,255,255,0.25)" if grayscaleInput else "rgba(255,255,255,0.35)"
        bg_dark  = "rgba(0,0,0,0.25)"       if grayscaleInput else "rgba(0,0,0,0.35)"

        for i in range(n_rows):
            for j in range(n_cols):
                val = m[i, j]
                t = float((val - minValue) / denom) if np.isfinite(val) else 0.5
                t = max(0.0, min(1.0, t))
                rgb = _interp_color(t)
                lum = _rel_luminance(rgb)

                font_color = "black" if lum >= lum_threshold else "white"

                # Publication mode: keep annotation backgrounds subtle (or none if you prefer)
                # Here we keep a slight translucent pad for readability on mid-gray cells.
                bg = bg_light if font_color == "black" else bg_dark

                if np.isfinite(val) and float(val).is_integer():
                    txt = str(int(val))
                else:
                    txt = str(round(float(val), int(mantissa))) if np.isfinite(val) else "nan"

                annotations.append(
                    dict(
                        x=j, y=i,
                        text=txt,
                        showarrow=False,
                        xref="x", yref="y",
                        font=dict(color=font_color, size=annot_font_size),
                        bgcolor=bg,
                        opacity=1.0
                    )
                )

        # -----------------------------
        # Heatmap
        # -----------------------------
        data = go.Heatmap(
            z=m,
            x=list(range(n_cols)),
            y=list(range(n_rows)),
            zmin=minValue,
            zmax=maxValue,
            showscale=bool(showScale),
            colorscale=colorscale,
            colorbar=dict(
                tickfont=dict(size=14, color=("black" if grayscaleInput else None)),
                title=dict(font=dict(size=15, color=("black" if grayscaleInput else None))),
                outlinecolor=("black" if grayscaleInput else None),
                outlinewidth=(1 if grayscaleInput else None)
            )
        )

        # -----------------------------
        # Layout + axes
        # -----------------------------
        rotate_x = 0
        if n_cols >= 8:
            rotate_x = 45
        if n_cols >= 16:
            rotate_x = 60

        fig = go.Figure(data=data)
        fig.update_layout(
            width=width,
            height=height,
            title=dict(text=title, font=dict(size=titleFontSize, color=("black" if grayscaleInput else None))),
            paper_bgcolor=paper_bg,
            plot_bgcolor=plot_bg,
            margin=dict(l=marginLeft, r=marginRight, t=marginTop, b=marginBottom),
            template=template_name,
            annotations=annotations,
            font=dict(size=baseFontSize, color=("black" if grayscaleInput else None))
        )

        # Axis styling for publication grayscale output
        axis_common = dict(
            showline=True if grayscaleInput else False,
            linecolor="black" if grayscaleInput else None,
            linewidth=1 if grayscaleInput else None,
            mirror=True if grayscaleInput else False,
            ticks="outside" if grayscaleInput else None,
            tickcolor="black" if grayscaleInput else None,
            ticklen=6 if grayscaleInput else None,
            tickwidth=1 if grayscaleInput else None,
            showgrid=True,
            gridcolor="rgba(0,0,0,0.12)" if grayscaleInput else "rgba(0,0,0,0.08)",
            zeroline=False
        )

        fig.update_xaxes(
            title=dict(text=xTitle, font=dict(size=axisTitleFontSize, color=("black" if grayscaleInput else None))),
            tickmode="array",
            tickvals=list(range(n_cols)),
            ticktext=xCats,
            tickangle=rotate_x,
            tickfont=dict(size=tickFontSize, color=("black" if grayscaleInput else None)),
            **axis_common
        )
        fig.update_yaxes(
            title=dict(text=yTitle, font=dict(size=axisTitleFontSize, color=("black" if grayscaleInput else None))),
            tickmode="array",
            tickvals=list(range(n_rows)),
            ticktext=yCats,
            tickfont=dict(size=tickFontSize, color=("black" if grayscaleInput else None)),
            autorange="reversed",
            **axis_common
        )

        return fig
    
    @staticmethod
    def FigureByCorrelation(actual, predicted,
                            title="Correlation between Actual and Predicted Values",
                            xTitle="Actual Values", yTitle="Predicted Values",
                            showIdentity=True, showBestFit=True,
                            dotSize=6, dotColor="blue", lineColor="red",
                            width=800, height=600, theme="default",
                            backgroundColor="rgba(0,0,0,0)",
                            marginLeft=0, marginRight=0, marginTop=40, marginBottom=0):
        """Creates a parity/correlation plot for paired actual and predicted values."""
        import numpy as np
        import plotly.graph_objects as go

        if actual is None or predicted is None:
            return None
        try:
            x = np.asarray(actual, dtype=float).reshape(-1)
            y = np.asarray(predicted, dtype=float).reshape(-1)
        except Exception:
            return None
        if len(x) == 0 or len(x) != len(y):
            return None
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if len(x) == 0:
            return None

        mae = float(np.mean(np.abs(y - x)))
        rmse = float(np.sqrt(np.mean((y - x) ** 2)))
        ss_res = float(np.sum((x - y) ** 2))
        ss_tot = float(np.sum((x - np.mean(x)) ** 2))
        r2 = float("nan") if ss_tot <= 1.0e-15 else 1.0 - ss_res / ss_tot

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers", name="Predictions",
            marker=dict(size=max(1, int(dotSize)), color=dotColor, opacity=0.8),
            hovertemplate=f"{xTitle}: %{{x}}<br>{yTitle}: %{{y}}<extra></extra>",
        ))

        mn = float(min(np.min(x), np.min(y)))
        mx = float(max(np.max(x), np.max(y)))
        if mn == mx:
            pad = max(abs(mn) * 0.05, 0.5)
            mn, mx = mn - pad, mx + pad

        if showIdentity:
            fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines", name="Identity (y=x)", line=dict(color="black", dash="dash"), hoverinfo="skip"))

        if showBestFit and len(x) >= 2 and float(np.ptp(x)) > 1.0e-15:
            a, b = np.polyfit(x, y, 1)
            fig.add_trace(go.Scatter(x=[mn, mx], y=[a * mn + b, a * mx + b], mode="lines", name=f"Best fit (y={a:.3g}x+{b:.3g})", line=dict(color=lineColor), hoverinfo="skip"))

        metrics = f"MAE={mae:.4g}, RMSE={rmse:.4g}"
        if np.isfinite(r2):
            metrics += f", R²={r2:.4g}"
        theme_name = str(theme or "default").lower()
        template = {"default": "plotly_white", "light": "plotly_white", "dark": "plotly_dark"}.get(theme_name, "plotly_white")
        background = Plotly._color_to_hex(backgroundColor)
        fig.update_layout(
            title=f"{title} — {metrics}", xaxis_title=xTitle, yaxis_title=yTitle,
            width=width, height=height, template=template,
            paper_bgcolor=background, plot_bgcolor=background,
            margin=dict(l=marginLeft, r=marginRight, t=marginTop, b=marginBottom),
            hoverlabel=dict(align="left", namelength=-1),
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        return fig

    @staticmethod
    def FigureByDataFrame(dataFrame, labels=None, width=950, height=500,
                          title="Untitled", xTitle="X Axis", xSpacing=1,
                          yTitle="Y Axis", ySpacing=1.0, useMarkers=False,
                          chartType="Line", backgroundColor="rgba(0,0,0,0)",
                          gridColor="lightgray", marginLeft=0, marginRight=0,
                          marginTop=40, marginBottom=0):
        """Returns a Plotly figure from a pandas-compatible dataframe."""
        import plotly.express as px

        if dataFrame is None or not hasattr(dataFrame, "columns"):
            raise TypeError("Plotly.FigureByDataFrame - Error: The input dataFrame parameter is not a valid dataframe.")
        columns = list(dataFrame.columns)
        labels = list(labels) if labels is not None else columns
        if not labels:
            labels = columns
        if len(labels) < 2:
            return None
        if any(label not in columns for label in labels):
            return None

        chart = str(chartType or "line").lower()
        if chart == "line":
            figure = px.line(dataFrame, x=labels[0], y=labels[1:], title=title, markers=bool(useMarkers))
        elif chart == "bar":
            figure = px.bar(dataFrame, x=labels[0], y=labels[1:], title=title)
        elif chart == "scatter":
            figure = px.scatter(dataFrame, x=labels[0], y=labels[1:], title=title)
        else:
            return None

        figure.update_layout(
            width=width, height=height, title=title,
            xaxis=dict(title=xTitle, dtick=xSpacing, gridcolor=Plotly._color_to_hex(gridColor, gridColor)),
            yaxis=dict(title=yTitle, dtick=ySpacing, gridcolor=Plotly._color_to_hex(gridColor, gridColor)),
            paper_bgcolor=Plotly._color_to_hex(backgroundColor),
            plot_bgcolor=Plotly._color_to_hex(backgroundColor),
            margin=dict(l=marginLeft, r=marginRight, t=marginTop, b=marginBottom),
            hoverlabel=dict(align="left", namelength=-1),
        )
        return figure


    @staticmethod
    def FigureByData(data, width=950, height=500,
                     xAxis=False, yAxis=False, zAxis=False,
                     axisSize=1, backgroundColor="rgba(0,0,0,0)",
                     marginLeft=0, marginRight=0,
                     marginTop=20, marginBottom=0,
                     tolerance=0.0001):
        """Creates a Plotly figure from a list of traces."""
        if not Plotly._plotly_available(silent=True) or not isinstance(data, list):
            return None

        traces = list(data)
        if xAxis or yAxis or zAxis:
            try:
                from topologicpy.Vertex import Vertex
                from topologicpy.Edge import Edge
                v0 = Vertex.ByCoordinates(0, 0, 0)
                v1 = Vertex.ByCoordinates(axisSize, 0, 0)
                v2 = Vertex.ByCoordinates(0, axisSize, 0)
                v3 = Vertex.ByCoordinates(0, 0, axisSize)
                if xAxis:
                    axis_data = Plotly.DataByTopology(Edge.ByVertices([v0, v1], tolerance=tolerance), edgeColor="red", edgeWidth=6, showFaces=False, showEdges=True, showVertices=False, edgeLegendLabel="X-Axis") or []
                    traces.extend(axis_data)
                if yAxis:
                    axis_data = Plotly.DataByTopology(Edge.ByVertices([v0, v2], tolerance=tolerance), edgeColor="green", edgeWidth=6, showFaces=False, showEdges=True, showVertices=False, edgeLegendLabel="Y-Axis") or []
                    traces.extend(axis_data)
                if zAxis:
                    axis_data = Plotly.DataByTopology(Edge.ByVertices([v0, v3], tolerance=tolerance), edgeColor="blue", edgeWidth=6, showFaces=False, showEdges=True, showVertices=False, edgeLegendLabel="Z-Axis") or []
                    traces.extend(axis_data)
            except Exception:
                pass

        figure = go.Figure(data=traces)
        figure.update_layout(
            width=width, height=height, showlegend=True,
            scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode="data"),
            paper_bgcolor=Plotly._color_to_hex(backgroundColor),
            plot_bgcolor=Plotly._color_to_hex(backgroundColor),
            margin=dict(l=marginLeft, r=marginRight, t=marginTop, b=marginBottom),
            hoverlabel=dict(align="left", namelength=-1),
        )
        figure.update_xaxes(showgrid=False, zeroline=False, visible=False)
        figure.update_yaxes(showgrid=False, zeroline=False, visible=False)
        return figure

    @staticmethod
    def FigureByJSONFile(file):
        """Imports a Plotly figure from an open JSON file object or path-like object."""
        if not Plotly._plotly_available(silent=True) or file is None:
            return None
        try:
            return plotly.io.read_json(file, output_type="Figure", skip_invalid=False, engine=None)
        except Exception:
            return None
    
    @staticmethod
    def FigureByJSONPath(path):
        """Imports a Plotly figure from a JSON file path."""
        if not Plotly._plotly_available(silent=True) or not isinstance(path, (str, os.PathLike)):
            return None
        try:
            return plotly.io.read_json(path, output_type="Figure", skip_invalid=False, engine=None)
        except Exception:
            print("Plotly.FigureByJSONPath - Error: The JSON path is not a valid Plotly JSON file. Returning None.")
            return None

    @staticmethod
    def FigureByPieChart(data, values, names):
        """
        Creates a plotly pie chart figure.

        Parameters
        ----------
        data : list
            The input list of plotly data.
        values : list
            The input list of values.
        names : list
            The input list of names.
        """

        import plotly.express as px

        try:
            import pandas as pd
        except Exception:
            warnings.warn("Plotly.FigureByPieChart - Error: Could not import pandas. Please install pandas manually. Returning None.")
            return None

        try:
            if hasattr(data, "columns"):
                df = data
            elif isinstance(data, dict):
                df = pd.DataFrame(data)
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                if not data:
                    return None
                df = pd.DataFrame(data)
            fig = px.pie(df, values=values, names=names)
            return fig
        except Exception as exc:
            warnings.warn(f"Plotly.FigureByPieChart - Error: {exc}. Returning None.")
            return None
    
    @staticmethod
    def FigureByTopology(topology,
                         showVertices=True, vertexSize=1.1, vertexColor="black",
                         vertexLabelKey=None, vertexGroupKey=None, vertexGroups=None,
                         vertexMinGroup=None, vertexMaxGroup=None,
                         showVertexLegend=False, vertexLegendLabel="Topology Vertices", vertexLegendRank=1,
                         vertexLegendGroup=1,
                         showEdges=True, edgeWidth=1, edgeColor="black",
                         edgeLabelKey=None, edgeGroupKey=None, edgeGroups=None,
                         edgeMinGroup=None, edgeMaxGroup=None,
                         showEdgeLegend=False, edgeLegendLabel="Topology Edges", edgeLegendRank=2,
                         edgeLegendGroup=2,
                         showFaces=True, faceOpacity=0.5, faceColor="#FAFAFA",
                         faceLabelKey=None, faceGroupKey=None, faceGroups=None,
                         faceMinGroup=None, faceMaxGroup=None,
                         showFaceLegend=False, faceLegendLabel="Topology Faces", faceLegendRank=3,
                         faceLegendGroup=3, intensityKey=None,
                         width=950, height=500,
                         xAxis=False, yAxis=False, zAxis=False, axisSize=1,
                         backgroundColor="rgba(0,0,0,0)",
                         marginLeft=0, marginRight=0, marginTop=20, marginBottom=0, showScale=False,
                         cbValues=None, cbTicks=5, cbX=-0.15, cbWidth=15, cbOutlineWidth=0, cbTitle="",
                         cbSubTitle="", cbUnits="", colorScale="viridis", mantissa=6, tolerance=0.0001,
                         # Extended styling arguments are appended to preserve the
                         # positional API of earlier TopologicPy releases.
                         vertexSizeKey=None, vertexColorKey=None,
                         vertexBorderColor="black", vertexBorderWidth=0,
                         vertexBorderColorKey=None, vertexBorderWidthKey=None,
                         showVertexLabel=False, vertexLabelFontSize=5,
                         directed=False, arrowSize=0.1, arrowSizeKey=None,
                         edgeWidthKey=None, edgeColorKey=None, edgeDash=False, edgeDashKey=None,
                         showEdgeLabel=False,
                         faceOpacityKey=None, faceColorKey=None, intensities=None,
                         material="default", materialKey=None, flatShading=False,
                         ambient=None, ambientKey=None, diffuse=None, diffuseKey=None,
                         specular=None, specularKey=None, roughness=None, roughnessKey=None,
                         silent=False):
        """Creates a Plotly figure from a Topologic topology."""
        from topologicpy.Topology import Topology

        if not Topology.IsInstance(topology, "Topology"):
            if not silent:
                print("Plotly.FigureByTopology - Error: The input topology is not valid. Returning None.")
            return None

        data = Plotly.DataByTopology(
            topology=topology,
            showVertices=showVertices, vertexSize=vertexSize, vertexSizeKey=vertexSizeKey,
            vertexColor=vertexColor, vertexColorKey=vertexColorKey, vertexLabelKey=vertexLabelKey,
            vertexBorderColor=vertexBorderColor, vertexBorderWidth=vertexBorderWidth,
            vertexBorderColorKey=vertexBorderColorKey, vertexBorderWidthKey=vertexBorderWidthKey,
            showVertexLabel=showVertexLabel, vertexLabelFontSize=vertexLabelFontSize,
            vertexGroupKey=vertexGroupKey, vertexGroups=vertexGroups,
            vertexMinGroup=vertexMinGroup, vertexMaxGroup=vertexMaxGroup,
            showVertexLegend=showVertexLegend, vertexLegendLabel=vertexLegendLabel,
            vertexLegendRank=vertexLegendRank, vertexLegendGroup=vertexLegendGroup,
            directed=directed, arrowSize=arrowSize, arrowSizeKey=arrowSizeKey,
            showEdges=showEdges, edgeWidth=edgeWidth, edgeWidthKey=edgeWidthKey,
            edgeColor=edgeColor, edgeColorKey=edgeColorKey, edgeDash=edgeDash,
            edgeDashKey=edgeDashKey, edgeLabelKey=edgeLabelKey, showEdgeLabel=showEdgeLabel,
            edgeGroupKey=edgeGroupKey, edgeGroups=edgeGroups,
            edgeMinGroup=edgeMinGroup, edgeMaxGroup=edgeMaxGroup,
            showEdgeLegend=showEdgeLegend, edgeLegendLabel=edgeLegendLabel,
            edgeLegendRank=edgeLegendRank, edgeLegendGroup=edgeLegendGroup,
            showFaces=showFaces, faceOpacity=faceOpacity, faceOpacityKey=faceOpacityKey,
            faceColor=faceColor, faceColorKey=faceColorKey, faceLabelKey=faceLabelKey,
            faceGroupKey=faceGroupKey, faceGroups=faceGroups,
            faceMinGroup=faceMinGroup, faceMaxGroup=faceMaxGroup,
            showFaceLegend=showFaceLegend, faceLegendLabel=faceLegendLabel,
            faceLegendRank=faceLegendRank, faceLegendGroup=faceLegendGroup,
            intensityKey=intensityKey, intensities=intensities,
            material=material, materialKey=materialKey, flatShading=flatShading,
            ambient=ambient, ambientKey=ambientKey, diffuse=diffuse, diffuseKey=diffuseKey,
            specular=specular, specularKey=specularKey, roughness=roughness, roughnessKey=roughnessKey,
            colorScale=colorScale, mantissa=mantissa, tolerance=tolerance, silent=silent,
        )
        if data is None:
            return None
        figure = Plotly.FigureByData(
            data=data, width=width, height=height,
            xAxis=xAxis, yAxis=yAxis, zAxis=zAxis, axisSize=axisSize,
            backgroundColor=backgroundColor,
            marginLeft=marginLeft, marginRight=marginRight,
            marginTop=marginTop, marginBottom=marginBottom,
            tolerance=tolerance,
        )
        if figure is not None and showScale:
            values = cbValues if cbValues is not None else intensities
            figure = Plotly.AddColorBar(
                figure, values=values, nTicks=cbTicks, xPosition=cbX,
                width=cbWidth, outlineWidth=cbOutlineWidth,
                title=cbTitle, subTitle=cbSubTitle, units=cbUnits,
                colorScale=colorScale, mantissa=mantissa,
            )
        return figure
    
    @staticmethod
    def FigureExportToJSON(figure, path, overwrite=False):
        """Exports a Plotly figure to JSON."""
        if not Plotly._plotly_available(silent=False) or not isinstance(figure, go.Figure):
            return None
        if not isinstance(path, (str, os.PathLike)):
            return None
        path = os.fspath(path)
        if not path.lower().endswith(".json"):
            path += ".json"
        if not overwrite and os.path.exists(path):
            print("Plotly.FigureExportToJSON - Error: A file already exists at this location and overwrite is False. Returning None.")
            return None
        try:
            plotly.io.write_json(figure, path, validate=True, pretty=False, remove_uids=True, engine=None)
            return True
        except Exception as exc:
            print(f"Plotly.FigureExportToJSON - Error: {exc}. Returning None.")
            return None

    @staticmethod
    def FigureExportToPDF(figure, path, width=1920, height=1200, overwrite=False):
        """Exports a Plotly figure to PDF."""
        if not Plotly._plotly_available(silent=False) or not isinstance(figure, go.Figure):
            return None
        if not isinstance(path, (str, os.PathLike)):
            return None
        path = os.fspath(path)
        if not path.lower().endswith(".pdf"):
            path += ".pdf"
        if not overwrite and os.path.exists(path):
            print("Plotly.FigureExportToPDF - Error: A file already exists at this location and overwrite is False. Returning None.")
            return None
        try:
            plotly.io.write_image(figure, path, format="pdf", scale=1, width=int(width), height=int(height), validate=True)
            return True
        except Exception as exc:
            print(f"Plotly.FigureExportToPDF - Error: {exc}. Returning None.")
            return None
    
    @staticmethod
    def FigureExportToPNG(figure, path, width=1920, height=1200, overwrite=False):
        """Exports a Plotly figure to PNG."""
        if not Plotly._plotly_available(silent=False) or not isinstance(figure, go.Figure):
            return None
        if not isinstance(path, (str, os.PathLike)):
            return None
        path = os.fspath(path)
        if not path.lower().endswith(".png"):
            path += ".png"
        if not overwrite and os.path.exists(path):
            print("Plotly.FigureExportToPNG - Error: A file already exists at this location and overwrite is False. Returning None.")
            return None
        try:
            plotly.io.write_image(figure, path, format="png", scale=1, width=int(width), height=int(height), validate=True)
            return True
        except Exception as exc:
            print(f"Plotly.FigureExportToPNG - Error: {exc}. Returning None.")
            return None
    
    @staticmethod
    def FigureExportToSVG(figure, path, width=1920, height=1200, overwrite=False):
        """Exports a Plotly figure to SVG."""
        if not Plotly._plotly_available(silent=False) or not isinstance(figure, go.Figure):
            return None
        if not isinstance(path, (str, os.PathLike)):
            return None
        path = os.fspath(path)
        if not path.lower().endswith(".svg"):
            path += ".svg"
        if not overwrite and os.path.exists(path):
            print("Plotly.FigureExportToSVG - Error: A file already exists at this location and overwrite is False. Returning None.")
            return None
        try:
            plotly.io.write_image(figure, path, format="svg", scale=1, width=int(width), height=int(height), validate=True)
            return True
        except Exception as exc:
            print(f"Plotly.FigureExportToSVG - Error: {exc}. Returning None.")
            return None
    
    @staticmethod
    def SetCamera(figure, camera=None, center=None, up=None, projection="perspective"):
        """Sets the 3D scene camera of a Plotly figure."""
        if not Plotly._plotly_available(silent=True) or not isinstance(figure, go.Figure):
            return None

        def vector(value, default):
            if isinstance(value, (list, tuple)) and len(value) >= 3:
                try:
                    return [float(value[0]), float(value[1]), float(value[2])]
                except Exception:
                    pass
            return list(default)

        eye = vector(camera, [-1.25, -1.25, 1.25])
        target = vector(center, [0.0, 0.0, 0.0])
        up_vector = vector(up, [0.0, 0.0, 1.0])
        projection_name = "orthographic" if "ortho" in str(projection or "perspective").lower() else "perspective"
        figure.update_layout(scene_camera=dict(
            eye=dict(x=eye[0], y=eye[1], z=eye[2]),
            center=dict(x=target[0], y=target[1], z=target[2]),
            up=dict(x=up_vector[0], y=up_vector[1], z=up_vector[2]),
            projection=dict(type=projection_name),
        ))
        return figure

    @staticmethod
    def Show(figure, camera=None, center=None, up=None, renderer=None, projection=None):
        """Displays a Plotly figure without silently overwriting its existing camera.

        Camera components are changed only when the corresponding arguments are
        explicitly supplied. This preserves camera settings applied with
        ``figure.update_layout`` or ``Plotly.SetCamera``.
        """
        if not Plotly._plotly_available(silent=False):
            return None
        if not isinstance(figure, go.Figure):
            print("Plotly.Show - Error: The input is not a Plotly figure. Returning None.")
            return None

        if any(value is not None for value in (camera, center, up, projection)):
            existing = figure.layout.scene.camera.to_plotly_json() if figure.layout.scene.camera else {}

            def vector(value, current, default):
                if value is None:
                    raw = current or {}
                    return [raw.get("x", default[0]), raw.get("y", default[1]), raw.get("z", default[2])]
                if isinstance(value, (list, tuple)) and len(value) >= 3:
                    try:
                        return [float(value[0]), float(value[1]), float(value[2])]
                    except Exception:
                        pass
                return list(default)

            eye = vector(camera, existing.get("eye"), [-1.25, -1.25, 1.25])
            target = vector(center, existing.get("center"), [0.0, 0.0, 0.0])
            up_vector = vector(up, existing.get("up"), [0.0, 0.0, 1.0])
            current_projection = (existing.get("projection") or {}).get("type", "perspective")
            projection_name = current_projection if projection is None else ("orthographic" if "ortho" in str(projection).lower() else "perspective")
            figure.update_layout(scene_camera=dict(
                eye=dict(x=eye[0], y=eye[1], z=eye[2]),
                center=dict(x=target[0], y=target[1], z=target[2]),
                up=dict(x=up_vector[0], y=up_vector[1], z=up_vector[2]),
                projection=dict(type=projection_name),
            ))

        renderer = Plotly.Renderer() if renderer is None else str(renderer).lower()
        if renderer == "offline":
            if ofl is None:
                return None
            ofl.plot(figure)
            return None

        available = Plotly.Renderers()
        if renderer not in available:
            print("Plotly.Show - Error: The input renderer is not available. Returning None.")
            return None
        figure.show(renderer=renderer)
        return None

    @staticmethod
    def Renderer():
        """
        Return the renderer most suitable for the environment in which the script is running.

        Parameters
        ----------

        Returns
        -------
        str
            The most suitable renderer type for the environment in which the script is running.
            Currently, this is limited to:
            - "vscode" if running in Visual Studio Code
            - "colab" if running in Google Colab
            - "iframe" if running in jupyter notebook or jupyterlab
            - "browser" if running in anything else
        """
        import sys
        import os
        
        if 'VSCODE_PID' in os.environ:
            return 'vscode'
        elif "google.colab" in sys.modules:
            return "colab"
        elif "ipykernel" in sys.modules:
            return "iframe" #works for jupyter notebook and jupyterlab
        else:
            return "browser"

    @staticmethod
    def Renderers():
        """Returns the Plotly renderers available in the current installation."""
        if not Plotly._plotly_available(silent=True):
            return []
        try:
            names = list(plotly.io.renderers)
        except Exception:
            names = []
        if "offline" not in names:
            names.append("offline")
        return names

    @staticmethod
    def ExportToImage(figure, path, format="png", width=1920, height=1080):
        """Exports a Plotly figure to a static image using the current Kaleido pathway."""
        if not Plotly._plotly_available(silent=True) or not isinstance(figure, go.Figure):
            return None
        if not isinstance(path, (str, os.PathLike)):
            return None
        fmt = str(format or "png").lower()
        if fmt not in ["jpg", "jpeg", "pdf", "png", "svg", "webp"]:
            return None
        path = os.fspath(path)
        try:
            plotly.io.write_image(figure, path, format=fmt, width=int(width), height=int(height), validate=True)
            return True
        except Exception:
            return False
