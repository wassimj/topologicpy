# Copyright (C) 2026
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
    import plotly
    import plotly.graph_objects as go
    import plotly.offline as ofl
except:
    print("Plotly - Installing required plotly library.")
    try:
        os.system("pip install plotly")
    except:
        os.system("pip install plotly --user")
    try:
        import plotly
        import plotly.graph_objects as go
        import plotly.offline as ofl
    except:
        warnings.warn("Plotly - Error: Could not import plotly.")

class Plotly:
    @staticmethod
    def AddColorBar(figure, values=None, nTicks=5, xPosition=-0.15, width=15,
                    outlineWidth=0, title="", subTitle="", units="",
                    colorScale="viridis", mantissa: int = 6):
        """
        Adds a scalar colour bar to a Plotly figure without adding visible data.

        Parameters
        ----------
        figure : plotly.graph_objs._figure.Figure
            The input Plotly figure.
        values : list, optional
            Numeric values used to derive the colour-bar range. If omitted or
            empty, the method returns the input figure unchanged.
        nTicks : int, optional
            Number of tick labels to draw. Values below 2 are clamped to 2.
        xPosition : float, optional
            Horizontal colour-bar position in Plotly paper coordinates.
        width : int, optional
            Colour-bar thickness in pixels.
        outlineWidth : int, optional
            Colour-bar outline width in pixels.
        title, subTitle, units : str, optional
            Text displayed above the colour bar.
        colorScale : str, optional
            Plotly colour scale name or one of TopologicPy's colour-blind
            friendly aliases: protanopia, deuteranopia, tritanopia.
        mantissa : int, optional
            Number of decimal places used for tick labels.

        Returns
        -------
        plotly.graph_objs._figure.Figure or None
            The updated figure, or None if the input is not a Plotly figure.
        """
        if not isinstance(figure, plotly.graph_objs._figure.Figure):
            return None

        if values is None:
            return figure

        clean_values = []
        for value in values:
            try:
                clean_values.append(float(value))
            except Exception:
                pass
        if not clean_values:
            return figure

        nTicks = max(2, int(nTicks or 2))
        minValue = min(clean_values)
        maxValue = max(clean_values)
        if maxValue == minValue:
            tickvals = [round(minValue, mantissa)]
        else:
            step = (maxValue - minValue) / float(nTicks - 1)
            tickvals = [round(minValue + i * step, mantissa) for i in range(nTicks)]
            tickvals[-1] = round(maxValue, mantissa)
        ticktext = [str(x) for x in tickvals]

        title_parts = []
        if title:
            title_parts.append("<b>" + str(title) + "</b>")
        if subTitle:
            title_parts.append(str(subTitle))
        if units:
            title_parts.append("Units: " + str(units))
        colorbar_title = "<br>".join(title_parts)

        # Use a marker with transparent colour and no visible size. This is the
        # lightest reliable way to attach an independent colour bar to a figure.
        colorbar_trace = go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            showlegend=False,
            hoverinfo="skip",
            marker=dict(
                size=0,
                colorscale=Plotly.ColorScale(colorScale),
                cmin=minValue,
                cmax=maxValue,
                color=[minValue],
                opacity=0,
                colorbar=dict(
                    x=xPosition,
                    title=colorbar_title,
                    ticks="outside",
                    tickvals=tickvals,
                    ticktext=ticktext,
                    tickmode="array",
                    thickness=width,
                    outlinewidth=outlineWidth,
                ),
            ),
        )
        figure.add_trace(colorbar_trace)
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
        
        # Colors recommended by various sources for color-blind-friendly palettes
        protanopia_colors = [
            "#E69F00", # orange
            "#56B4E9", # sky blue
            "#009E73", # bluish green
            "#F0E442", # yellow
            "#0072B2", # blue
            "#D55E00", # vermillion
            "#CC79A7", # reddish purple
        ]

        deuteranopia_colors = [
            "#377EB8", # blue
            "#FF7F00", # orange
            "#4DAF4A", # green
            "#F781BF", # pink
            "#A65628", # brown
            "#984EA3", # purple
            "#999999", # grey
        ]

        tritanopia_colors = [
            "#E69F00", # orange
            "#56B4E9", # sky blue
            "#009E73", # bluish green
            "#F0E442", # yellow
            "#0072B2", # blue
            "#D55E00", # vermillion
            "#CC79A7", # reddish purple
        ]

        # Create colorscales for Plotly
        def create_colorscale(colors):
            colorscale = []
            num_colors = len(colors)
            for i, color in enumerate(colors):
                position = i / (num_colors - 1)
                colorscale.append((position, color))
            return colorscale
        
        if "prota" in colorScale.lower():
            return create_colorscale(protanopia_colors)
        elif "deutera" in colorScale.lower():
            return create_colorscale(deuteranopia_colors)
        elif "trita"in colorScale.lower():
            return create_colorscale(tritanopia_colors)
        else:
            return colorScale
    
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
                    vertexGroups: list = [],
                    vertexMinGroup = None,
                    vertexMaxGroup = None,
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
                    edgeGroups: list = [],
                    edgeMinGroup = None,
                    edgeMaxGroup = None,
                    showEdges: bool = True,
                    showEdgeLabel: bool = False,
                    showEdgeLegend: bool = False,
                    edgeLegendLabel="Graph Edges",
                    edgeLegendRank=5, 
                    edgeLegendGroup=5,
                    colorScale: str = "viridis",
                    mantissa: int = 6,
                    silent: bool = False):
        """
        Creates plotly vertex and edge data from the input graph.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        sagitta : float , optional
            The length of the sagitta. In mathematics, the sagitta is the line connecting the center of a chord to the apex (or highest point) of the arc subtended by that chord. Default is 0 which means a straight edge is drawn instead of an arc. Default is 0.
        absolute : bool , optional
            If set to True, the sagitta length is treated as an absolute value. Otherwise, it is treated as a ratio based on the length of the edge. Default is False.
            For example, if the length of the edge is 10, the sagitta is set to 0.5, and absolute is set to False, the sagitta length will be 5. Default is True.
        sides : int , optional
            The number of sides of the arc. Default is 8.
        directed : bool , optional
            If set to True, arrowheads are drawn to show direction. Default is False.
        arrowSize : int, optional
            The desired size of arrowheads for directed graphs. Default is 0.1.
        arrowSizeKey: str , optional
            The edge dictionary key under which to find the arrowhead size. Default is None.
        vertexColor : str , optional
            The desired color of the output vertices. This can be any plotly color string and may be specified as:
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is "black".
        vertexColorKey : str , optional
            The dictionary key under which to find the vertex color. Default is None.
        vertexSize : float , optional
            The desired size of the vertices. Default is 6.
        vertexLabelKey : str , optional
            The dictionary key to use to display the vertex label. Default is None.
        vertexGroupKey : str , optional
            The dictionary key to use to display the vertex group. Default is None.
        vertexGroups : list , optional
            The list of vertex groups against which to index the color of the vertex. Default is [].
        vertexMinGroup : int or float , optional
            For numeric vertexGroups, vertexMinGroup is the desired minimum value for the scaling of colors. This should match the type of value associated with the vertexGroupKey. If set to None, it is set to the minimum value in vertexGroups. Default is None.
        vertexMaxGroup : int or float , optional
            For numeric vertexGroups, vertexMaxGroup is the desired maximum value for the scaling of colors. This should match the type of value associated with the vertexGroupKey. If set to None, it is set to the maximum value in vertexGroups. Default is None.
        showVertices : bool , optional
            If set to True the vertices will be drawn. Otherwise, they will not be drawn. Default is True.
        showVertexLabel : bool , optional
            If set to True, the vertex labels are shown permenantely on screen. Otherwise, they are not. Default is False.
        showVertexLegend : bool , optional
            If set to True the vertex legend will be drawn. Otherwise, it will not be drawn. Default is False.
        vertexLegendLabel : str , optional
            The legend label string used to identify vertices. Default is "Topology Vertices".
        vertexLegendRank : int , optional
            The legend rank order of the vertices of this topology. Default is 1.
        vertexLegendGroup : int , optional
            The number of the vertex legend group to which the vertices of this topology belong. Default is 1.
        edgeColor : str , optional
            The desired color of the output edges. This can be any plotly color string and may be specified as:
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is "black".
        edgeColorKey : str , optional
            The dictionary key under which to find the edge color. Default is None.
        edgeWidth : float , optional
            The desired thickness of the output edges. Default is 1.
        edgeWidthKey : str , optional
            The dictionary key under which to find the edge width. Default is None.
        edgeLabelKey : str , optional
            The dictionary key to use to display the edge label. Default is None.
        edgeDash : bool , optional
            If set to True, the edges are drawn as dashed lines. Default is False.
        edgeDashKey : str , optional
            The key under which to find the boolean flag to draw edges as dashed lines. Default is None.
        edgeGroupKey : str , optional
            The dictionary key to use to display the edge group. Default is None.
        edgeGroups : list , optional
            The list of groups to use for indexing the color of edges. Default is None.
        showEdges : bool , optional
            If set to True the edges will be drawn. Otherwise, they will not be drawn. Default is True.
        showEdgeLabels : bool , optional
            If set to True, the edge labels are shown permenantely on screen. Otherwise, they are not. Default is False.
        showEdgeLegend : bool , optional
            If set to True the edge legend will be drawn. Otherwise, it will not be drawn. Default is False.
        colorScale : str , optional
            The desired type of plotly color scales to use (e.g. "Viridis", "Plasma"). Default is "Viridis". For a full list of names, see https://plotly.com/python/builtin-colorscales/.
        silent : bool , optional
            If set to True, error and warning messages are suppressed. Default is False.
        
        Returns
        -------
        list
            The vertex and edge data list.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
        from topologicpy.Cluster import Cluster
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology
        from topologicpy.Graph import Graph
        from topologicpy.Color import Color
        import plotly.graph_objs as go

        if not Topology.IsInstance(graph, "Graph"):
            return None
        data = []
        
        if showEdges:
            e_dictionaries = []
            edges = Graph.Edges(graph)
            new_edges = []
            # if sagitta > 0:
            for edge in edges:
                d = Topology.Dictionary(edge)
                if sagitta > 0:
                    arc = Wire.ArcByEdge(edge, sagitta=sagitta, absolute=absolute, sides=sides, close=False, silent=silent)
                    if Topology.IsInstance(arc, "Wire"):
                        if not angle == 0:
                            direc = Edge.Direction(edge)
                            arc = Topology.Rotate(arc, origin=Topology.Centroid(edge), axis=direc, angle=angle)
                        arc_edges = Topology.Edges(arc)
                        for arc_edge in arc_edges:
                            arc_edge = Topology.SetDictionary(arc_edge, d, silent=True)
                            new_edges.append(arc_edge)
                            e_dictionaries.append(d)
                    else:
                        new_edges.append(edge)
                        e_dictionaries.append(d)

                else:
                    new_edges = edges
                    e_dictionaries.append(d)
            if len(new_edges) > 0:
                e_cluster = Cluster.ByTopologies(new_edges)
                geo = Topology.Geometry(e_cluster, mantissa=mantissa)
                vertices = geo['vertices']
                edges = geo['edges']
                data.extend(Plotly.edgeData(vertices, edges, dictionaries=e_dictionaries, color=edgeColor, colorKey=edgeColorKey, width=edgeWidth, widthKey=edgeWidthKey, dash=edgeDash, dashKey=edgeDashKey, directed=directed, arrowSize=arrowSize, arrowSizeKey=arrowSizeKey, labelKey=edgeLabelKey, showEdgeLabel=showEdgeLabel, groupKey=edgeGroupKey, minGroup=edgeMinGroup, maxGroup=edgeMaxGroup, groups=edgeGroups, legendLabel=edgeLegendLabel, legendGroup=edgeLegendGroup, legendRank=edgeLegendRank, showLegend=showEdgeLegend, colorScale=colorScale))        
        
        if showVertices:
            vertices = Graph.Vertices(graph)
            v_dictionaries = [Topology.Dictionary(v) for v in vertices]
            e_cluster = Cluster.ByTopologies(vertices)
            geo = Topology.Geometry(e_cluster, mantissa=mantissa)
            vertices = geo['vertices']
            if len(vertices) > 0:
                v_data = Plotly.vertexData(vertices,
                                   dictionaries=v_dictionaries,
                                   color=vertexColor,
                                   colorKey=vertexColorKey,
                                   size=vertexSize,
                                   sizeKey=vertexSizeKey,
                                   borderColor=vertexBorderColor,
                                   borderWidth=vertexBorderWidth,
                                   borderColorKey=vertexBorderColorKey,
                                   borderWidthKey=vertexBorderWidthKey,
                                   labelKey=vertexLabelKey,
                                   showVertexLabel=showVertexLabel,
                                   vertexLabelFontSize=vertexLabelFontSize,
                                   groupKey=vertexGroupKey,
                                   minGroup=vertexMinGroup,
                                   maxGroup=vertexMaxGroup,
                                   groups=vertexGroups,
                                   legendLabel=vertexLegendLabel,
                                   legendGroup=vertexLegendGroup,
                                   legendRank=vertexLegendRank,
                                   showLegend=showVertexLegend,
                                   colorScale=colorScale)
                data += v_data
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
        vertexColorKey: str = None,
        vertexSize: float = 10,
        vertexSizeKey: str = None,
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
        edgeColorKey: str = None,
        edgeWidth: float = 2,
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
        from topologicpy.TGraph import TGraph

        if not isinstance(graph, TGraph):
            if not silent:
                print("Plotly.DataByTGraph - Error: The input graph is not a valid TGraph. Returning None.")
            return None

        vertexGroups = list(vertexGroups) if vertexGroups is not None else []
        edgeGroups = list(edgeGroups) if edgeGroups is not None else []
        data = []

        def _value(d, key, default=None):
            if key is None or not isinstance(d, dict):
                return default
            return d.get(key, default)

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
            value = d.get(key, default)
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

        vertex_records = TGraph.Vertices(graph, asTopologic=False, activeOnly=True) or []
        edge_records = TGraph.Edges(graph, asTopologic=False, activeOnly=True) or []

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
                bucket["text"].extend([edge_label] * len(x))
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
                                         showlegend=bool(showEdgeLegend and first), hoverinfo="text",
                                         hovertext=bucket["text"]))
                first = False
            if showEdgeLabel and label_text:
                data.append(go.Scatter3d(x=label_x, y=label_y, z=label_z, mode="text", text=label_text,
                                         textfont=dict(size=edgeLabelFontSize), showlegend=False, hoverinfo="skip"))
            for (this_color, this_arrow_size), bucket in arrow_buckets.items():
                data.append(go.Cone(x=bucket["x"], y=bucket["y"], z=bucket["z"],
                                    u=bucket["u"], v=bucket["v"], w=bucket["w"],
                                    sizemode="absolute", sizeref=this_arrow_size, anchor="tip",
                                    showscale=False, colorscale=[[0, this_color], [1, this_color]],
                                    showlegend=False, hoverinfo="skip"))

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
                    "hover": "<br>".join([f"{k}: {v}" for k, v in d.items()]) if d else str(idx),
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
                    textfont=dict(size=vertexLabelFontSize), hoverinfo="text", hovertext=[i["hover"] for i in vertex_items],
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
                        textfont=dict(size=vertexLabelFontSize), hoverinfo="text", hovertext=[i["hover"] for i in items],
                        name=vertexLegendLabel, legendgroup=str(vertexLegendGroup), legendrank=vertexLegendRank,
                        showlegend=bool(showVertexLegend and first)))
                    first = False

        return data

    @staticmethod
    def vertexData(vertices,
                   dictionaries=None,
                   color="black",
                   colorKey=None,
                   size=1.1,
                   sizeKey=None,
                   borderColor="black",
                   borderWidth=0,
                   borderColorKey=None,
                   borderWidthKey=None,
                   labelKey=None,
                   showVertexLabel=False,
                   vertexLabelFontSize=5,
                   groupKey=None,
                   minGroup=None,
                   maxGroup=None,
                   groups=None,
                   legendLabel="Topology Vertices",
                   legendGroup=1,
                   legendRank=1,
                   showLegend=True,
                   colorScale="Viridis"):
        """
        Creates efficient Plotly Scatter3d traces for vertices.

        The method keeps the legacy API but avoids shared mutable defaults,
        handles missing dictionaries safely, and guards against group values not
        present in the supplied group list. When vertex borders are requested,
        an underlay marker trace is emitted before the main marker trace.
        """
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Color import Color

        if not vertices:
            return []
        dictionaries = list(dictionaries) if dictionaries is not None else []
        groups = list(groups) if groups is not None else []

        def _value(d, key, default=None):
            if key is None or not d:
                return default
            try:
                return Dictionary.ValueAtKey(d, key=key, defaultValue=default)
            except TypeError:
                return Dictionary.ValueAtKey(d, key, default)
            except Exception:
                return default

        def _float(value, default):
            try:
                return float(value)
            except Exception:
                return default

        def _hex(value, default="black"):
            try:
                return Color.AnyToHex(value)
            except Exception:
                try:
                    return Color.AnyToHex(default)
                except Exception:
                    return default

        def _group_color(value, default):
            if value is None:
                return default
            try:
                numeric_value = float(value)
                numeric_groups = []
                for g in groups:
                    try:
                        numeric_groups.append(float(g))
                    except Exception:
                        pass
                mn = float(minGroup) if minGroup is not None else (min(numeric_groups) if numeric_groups else 0.0)
                mx = float(maxGroup) if maxGroup is not None else (max(numeric_groups) if numeric_groups else 1.0)
                numeric_value = max(mn, min(mx, numeric_value))
                return _hex(Color.ByValueInRange(numeric_value, minValue=mn, maxValue=mx, colorScale=colorScale), default)
            except Exception:
                pass
            if groups and value in groups:
                mn = 0 if minGroup is None else minGroup
                mx = max(len(groups) - 1, 1) if maxGroup is None else maxGroup
                return _hex(Color.ByValueInRange(groups.index(value), minValue=mn, maxValue=mx, colorScale=colorScale), default)
            return default

        x, y, z = [], [], []
        sizes, labels, colors = [], [], []
        border_colors, border_sizes = [], []
        default_color = _hex(color)
        default_border_color = _hex(borderColor)
        default_size = max(_float(size, 1.1), 0.1)
        default_border_width = max(_float(borderWidth, 0), 0)
        n = len(str(len(vertices)))

        for i, v in enumerate(vertices):
            try:
                x.append(v[0]); y.append(v[1]); z.append(v[2])
            except Exception:
                continue
            d = dictionaries[i] if i < len(dictionaries) else None
            label = str(_value(d, labelKey, "Vertex_" + str(i + 1).zfill(n))) if labelKey else "Vertex_" + str(i + 1).zfill(n)
            this_size = max(_float(_value(d, sizeKey, default_size), default_size), 0.1)
            this_color = _hex(_value(d, colorKey, default_color), default_color) if colorKey else default_color
            if groupKey is not None:
                this_color = _group_color(_value(d, groupKey, None), this_color)
            this_border_color = _hex(_value(d, borderColorKey, default_border_color), default_border_color) if borderColorKey else default_border_color
            this_border_width = max(_float(_value(d, borderWidthKey, default_border_width), default_border_width), 0)
            labels.append(label)
            sizes.append(this_size)
            colors.append(this_color)
            border_colors.append(this_border_color)
            border_sizes.append(this_size + this_border_width * 2 if this_border_width > 0 else 0)

        if not x:
            return []

        mode = "markers+text" if showVertexLabel else "markers"
        traces = []
        if borderWidth > 0 or borderWidthKey:
            traces.append(go.Scatter3d(
                x=x, y=y, z=z,
                name=legendLabel,
                showlegend=False,
                marker=dict(color=border_colors, size=border_sizes, symbol="circle", opacity=1, line=dict(width=0), sizemode="diameter"),
                mode="markers",
                hoverinfo="skip",
                legendgroup=str(legendGroup),
                legendrank=legendRank,
            ))

        traces.append(go.Scatter3d(
            x=x, y=y, z=z,
            name=legendLabel,
            showlegend=showLegend,
            marker=dict(color=colors, size=sizes, symbol="circle", opacity=1, line=dict(width=0), sizemode="diameter"),
            mode=mode,
            customdata=labels,
            legendgroup=str(legendGroup),
            legendrank=legendRank,
            text=labels if showVertexLabel else None,
            textfont=dict(size=vertexLabelFontSize),
            hoverinfo="text",
            hovertext=labels,
            hovertemplate=["Click " + str(label) for label in labels],
        ))
        return traces

    @staticmethod
    def edgeData(vertices, edges, dictionaries=None, color="black", colorKey=None,
                 width=1, widthKey=None, dash=False, dashKey=None, directed=False,
                 arrowSize=0.1, arrowSizeKey=None, labelKey=None,
                 showEdgeLabel=False, groupKey=None, minGroup=None, maxGroup=None,
                 groups=None, legendLabel="Topology Edges", legendGroup=2,
                 legendRank=2, showLegend=True, colorScale="Viridis"):
        """
        Creates efficient Plotly Scatter3d traces for topology or graph edges.

        Edges are grouped by visual style (colour, width, dash state, and arrow
        size) so large graphs are drawn with far fewer traces. This greatly
        improves browser-side rendering speed compared with one trace per edge.

        Parameters are intentionally compatible with the legacy implementation.
        """
        import math
        from topologicpy.Color import Color
        from topologicpy.Dictionary import Dictionary

        if vertices is None or edges is None:
            return []
        groups = list(groups) if groups is not None else []
        dictionaries = list(dictionaries) if dictionaries is not None else []

        def _value(d, key, default=None):
            if key is None or not d:
                return default
            try:
                return Dictionary.ValueAtKey(d, key=key, defaultValue=default)
            except TypeError:
                return Dictionary.ValueAtKey(d, key, default)
            except Exception:
                return default

        def _float(value, default):
            try:
                return float(value)
            except Exception:
                return default

        def _bool(value, default=False):
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            if isinstance(value, str):
                return value.strip().lower() in ("true", "1", "yes", "y", "t")
            return default

        def _hex(value, default="black"):
            try:
                return Color.AnyToHex(value)
            except Exception:
                try:
                    return Color.AnyToHex(default)
                except Exception:
                    return default

        def _group_color(value, default):
            if value is None:
                return default
            try:
                numeric_value = float(value)
                numeric_groups = []
                for item in groups:
                    try:
                        numeric_groups.append(float(item))
                    except Exception:
                        pass
                mn = float(minGroup) if minGroup is not None else (min(numeric_groups) if numeric_groups else 0.0)
                mx = float(maxGroup) if maxGroup is not None else (max(numeric_groups) if numeric_groups else 1.0)
                numeric_value = max(mn, min(mx, numeric_value))
                return _hex(Color.ByValueInRange(numeric_value, minValue=mn, maxValue=mx, colorScale=colorScale), default)
            except Exception:
                pass
            if groups and value in groups:
                mn = 0 if minGroup is None else minGroup
                mx = max(len(groups) - 1, 1) if maxGroup is None else maxGroup
                return _hex(Color.ByValueInRange(groups.index(value), minValue=mn, maxValue=mx, colorScale=colorScale), default)
            return default

        def _dash_xyz(points):
            x, y, z = [], [], []
            for a, b in zip(points[:-1], points[1:]):
                x.extend([a[0], b[0], None])
                y.extend([a[1], b[1], None])
                z.extend([a[2], b[2], None])
            return x, y, z

        def _unit(a, b):
            u = b[0] - a[0]
            v = b[1] - a[1]
            w = b[2] - a[2]
            length = math.sqrt(u*u + v*v + w*w)
            if length <= 1e-12:
                return None
            return (u / length, v / length, w / length)

        trace_buckets = {}
        arrow_buckets = {}
        label_x, label_y, label_z, label_text = [], [], [], []

        default_color = _hex(color)
        for index, edge in enumerate(edges):
            try:
                sv = vertices[edge[0]]
                ev = vertices[edge[1]]
            except Exception:
                continue

            d = dictionaries[index] if index < len(dictionaries) else None
            this_color = _hex(_value(d, colorKey, default_color), default_color) if colorKey else default_color
            if groupKey is not None:
                this_color = _group_color(_value(d, groupKey, None), this_color)
            this_width = _float(_value(d, widthKey, width), width) if widthKey else _float(width, 1.0)
            this_dash = _bool(_value(d, dashKey, dash), dash) if dashKey else bool(dash)
            this_arrow_size = _float(_value(d, arrowSizeKey, arrowSize), arrowSize) if arrowSizeKey else _float(arrowSize, 0.1)

            key = (this_color, this_width, this_dash)
            bucket = trace_buckets.setdefault(key, {"x": [], "y": [], "z": [], "text": []})
            bucket["x"].extend([sv[0], ev[0], None])
            bucket["y"].extend([sv[1], ev[1], None])
            bucket["z"].extend([sv[2], ev[2], None])
            edge_label = str(_value(d, labelKey, "")) if labelKey else ""
            bucket["text"].extend([edge_label, edge_label, None])

            if showEdgeLabel and edge_label:
                label_x.append((sv[0] + ev[0]) * 0.5)
                label_y.append((sv[1] + ev[1]) * 0.5)
                label_z.append((sv[2] + ev[2]) * 0.5)
                label_text.append(edge_label)

            if directed:
                direction = _unit(sv, ev)
                if direction:
                    akey = (this_color, this_arrow_size)
                    ab = arrow_buckets.setdefault(akey, {"x": [], "y": [], "z": [], "u": [], "v": [], "w": []})
                    ab["x"].append(ev[0]); ab["y"].append(ev[1]); ab["z"].append(ev[2])
                    ab["u"].append(direction[0]); ab["v"].append(direction[1]); ab["w"].append(direction[2])

        traces = []
        first_trace = True
        for (this_color, this_width, this_dash), bucket in trace_buckets.items():
            line = dict(color=this_color, width=this_width)
            # Scatter3d does not reliably support line.dash across Plotly
            # versions, so dashed lines are represented as dotted markers plus
            # line segments where possible.
            mode = "lines+markers" if this_dash else "lines"
            marker = dict(size=max(1, this_width * 0.25), color=this_color) if this_dash else None
            traces.append(go.Scatter3d(
                x=bucket["x"], y=bucket["y"], z=bucket["z"],
                mode=mode,
                line=line,
                marker=marker,
                name=legendLabel,
                showlegend=bool(showLegend and first_trace),
                legendgroup=str(legendGroup),
                legendrank=legendRank,
                hoverinfo="text",
                hovertext=bucket["text"],
            ))
            first_trace = False

        if showEdgeLabel and label_text:
            traces.append(go.Scatter3d(
                x=label_x, y=label_y, z=label_z,
                mode="text",
                text=label_text,
                showlegend=False,
                hoverinfo="skip",
            ))

        for (this_color, this_arrow_size), bucket in arrow_buckets.items():
            traces.append(go.Cone(
                x=bucket["x"], y=bucket["y"], z=bucket["z"],
                u=bucket["u"], v=bucket["v"], w=bucket["w"],
                sizemode="absolute",
                sizeref=this_arrow_size,
                anchor="tip",
                showscale=False,
                colorscale=[[0, this_color], [1, this_color]],
                showlegend=False,
                hoverinfo="skip",
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
                       vertexBorderColor: str = "black",
                       vertexBorderWidth: float = 0,
                       vertexBorderColorKey: str = None,
                       vertexBorderWidthKey: float = None,
                       showVertexLabel=False,
                       vertexLabelFontSize = 5,
                       vertexGroupKey=None,
                       vertexGroups=[], 
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
                       edgeGroups=[], 
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
                       faceGroups=[], 
                       faceMinGroup=None,
                       faceMaxGroup=None, 
                       showFaceLegend=False,
                       faceLegendLabel="Topology Faces",
                       faceLegendRank=3,
                       faceLegendGroup=3, 
                       intensityKey=None,
                       intensities=[],
                       material = "default",
                       materialKey=None,
                       flatShading = True,
                       ambient = None,
                       ambientKey=None,
                       diffuse = None,
                       diffuseKey=None,
                       specular = None,
                       specularKey=None,
                       roughness = None,
                       roughnessKey=None,
                       colorScale="viridis",
                       mantissa=6,
                       tolerance=0.0001,
                       silent=False):
        """
        Creates plotly face, edge, and vertex data.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology. This must contain faces and or edges.

        showVertices : bool , optional
            If set to True the vertices will be drawn. Otherwise, they will not be drawn. Default is True.
        vertexSize : float , optional
            The desired size of the output vertices. Default is 1.1.
        vertexSizeKey : str , optional
            The dictionary key under which to find the vertex size.The default is None.
        vertexColor : str , optional
            The desired color of the output vertices. This can be any plotly color string and may be specified as:
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is "black".
        vertexColorKey : str , optional
            The dictionary key under which to find the vertex color.The default is None.
        vertexBorderWidth : float , optional
            The desired width of the border of the output vertices. Default is 0.
        vertexBorderColor : str , optional
            The desired color of the border of the output vertices. This can be any plotly color string and may be specified as:
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is "black".
        vertexLabelKey : str , optional
            The dictionary key to use to display the vertex label. Default is None.
        vertexLabelFontSize : int , optional
            The font size to use for vertex labels. Default is 5.
        vertexGroupKey : str , optional
            The dictionary key to use to display the vertex group. Default is None.
        vertexGroups : list , optional
            The list of vertex groups against which to index the color of the vertex. Default is [].
        vertexMinGroup : int or float , optional
            For numeric vertexGroups, vertexMinGroup is the desired minimum value for the scaling of colors. This should match the type of value associated with the vertexGroupKey. If set to None, it is set to the minimum value in vertexGroups. Default is None.
        vertexMaxGroup : int or float , optional
            For numeric vertexGroups, vertexMaxGroup is the desired maximum value for the scaling of colors. This should match the type of value associated with the vertexGroupKey. If set to None, it is set to the maximum value in vertexGroups. Default is None.
        showVertexLegend : bool, optional
            If set to True, the legend for the vertices of this topology is shown. Otherwise, it isn't. Default is False.
        vertexLegendLabel : str , optional
            The legend label string used to identify vertices. Default is "Topology Vertices".
        vertexLegendRank : int , optional
            The legend rank order of the vertices of this topology. Default is 1.
        vertexLegendGroup : int , optional
            The number of the vertex legend group to which the vertices of this topology belong. Default is 1.
        directed : bool , optional
            If set to True, arrowheads are drawn to show direction. Default is False.
        arrowSize : int, optional
            The desired size of arrowheads for directed graphs. Default is 0.1.
        arrowSizeKey: str , optional
            The edge dictionary key under which to find the arrowhead size. Default is None.
        showEdges : bool , optional
            If set to True the edges will be drawn. Otherwise, they will not be drawn. Default is True.
        edgeWidth : float , optional
            The desired thickness of the output edges. Default is 1.
        edgeWidthKey : str , optional
            The dictionary key under which to find the edge width.The default is None.
        edgeColor : str , optional
            The desired color of the output edges. This can be any plotly color string and may be specified as:
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is "black".
        edgeColorKey : str , optional
            The dictionary key under which to find the edge color.The default is None.
        edgeDash : bool , optional
            If set to True, the edges are drawn as dashed lines. Default is False.
        edgeDashKey : str , optional
            The key under which to find the boolean flag to draw edges as dashed lines. Default is None.
        edgeLabelKey : str , optional
            The dictionary key to use to display the edge label. Default is None.
        edgeGroupKey : str , optional
            The dictionary key to use to display the edge group. Default is None.
        edgeGroups : list , optional
            The list of edge groups against which to index the color of the edge. Default is [].
        edgeMinGroup : int or float , optional
            For numeric edgeGroups, edgeMinGroup is the desired minimum value for the scaling of colors. This should match the type of value associated with the edgeGroupKey. If set to None, it is set to the minimum value in edgeGroups. Default is None.
        edgeMaxGroup : int or float , optional
            For numeric edgeGroups, edgeMaxGroup is the desired maximum value for the scaling of colors. This should match the type of value associated with the edgeGroupKey. If set to None, it is set to the maximum value in edgeGroups. Default is None.
        showEdgeLegend : bool, optional
            If set to True, the legend for the edges of this topology is shown. Otherwise, it isn't. Default is False.
        edgeLegendLabel : str , optional
            The legend label string used to identify edges. Default is "Topology Edges".
        edgeLegendRank : int , optional
            The legend rank order of the edges of this topology. Default is 2.
        edgeLegendGroup : int , optional
            The number of the edge legend group to which the edges of this topology belong. Default is 2.
        showFaces : bool , optional
            If set to True the faces will be drawn. Otherwise, they will not be drawn. Default is True.
        faceOpacity : float , optional
            The desired opacity of the output faces (0=transparent, 1=opaque). Default is 0.5.
        faceOpacityKey : str , optional
            The dictionary key under which to find the face opacity.The default is None.
        faceColor : str , optional
            The desired color of the output faces. This can be any plotly color string and may be specified as:
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is "#FAFAFA".
        faceColorKey : str , optional
            The dictionary key under which to find the face color.The default is None.
        faceLabelKey : str , optional
            The dictionary key to use to display the face label. Default is None.
        faceGroupKey : str , optional
            The dictionary key to use to display the face group. Default is None.
        faceGroups : list , optional
            The list of face groups against which to index the color of the face. This can bhave numeric or string values. This should match the type of value associated with the faceGroupKey. Default is [].
        faceMinGroup : int or float , optional
            For numeric faceGroups, minGroup is the desired minimum value for the scaling of colors. This should match the type of value associated with the faceGroupKey. If set to None, it is set to the minimum value in faceGroups. Default is None.
        faceMaxGroup : int or float , optional
            For numeric faceGroups, maxGroup is the desired maximum value for the scaling of colors. This should match the type of value associated with the faceGroupKey. If set to None, it is set to the maximum value in faceGroups. Default is None.
        showFaceLegend : bool, optional
            If set to True, the legend for the faces of this topology is shown. Otherwise, it isn't. Default is False.
        faceLegendLabel : str , optional
            The legend label string used to idenitfy edges. Default is "Topology Faces".
        faceLegendRank : int , optional
            The legend rank order of the faces of this topology. Default is 3.
        faceLegendGroup : int , optional
            The number of the face legend group to which the faces of this topology belong. Default is 3.
        intensityKey : str, optional
            If not None, the dictionary of each vertex is searched for the value associated with the intensity key. This value is then used to color-code the vertex based on the colorScale. Default is None.
        intensities : list , optional
            The list of intensities against which to index the intensity of the vertex. Default is [].
        material : str , optional
            The type of object material. Supported pre-built materials are:
            Preset     Ambient  Diffuse  Specular  Roughness  Description
            --------------------------------------------------------------
            chalk        1.0      0.4       0.0        1.0     Very soft shading, low contrast
            concrete     0.85     0.75      0.05       0.9     Highly matte, micro-rough surface, minimal specular reflection
            eggshell     0.65     0.85      0.25       0.45    Slight sheen, soft highlights without gloss
            glossy       0.5      0.9       0.6        0.1     Highly polished appearance
            matte        0.9      0.7       0.0        1.0     Flat, non-reflective surfaces
            metallic     0.3      0.8       0.9        0.2     Strong, sharp reflections
            plastic      0.6      0.9       0.2        0.4     Soft highlights, good shape readability
            default      N/A      N/A       N/A        N/A     Flat shading is applied.
            Default is plastic.
        materialKey : str , optional
            The dictionary key under which the material string is stored. Default is None.
        flatShading : bool , optional
            If set to True, the model is rendered with flat shading with no clear light source. Default is True.
        ambient : float , optional
            Controls the strength of ambient light applied uniformly to the surface.
            Higher values reduce shading contrast by increasing overall brightness.
            Typical range is [0, 1]. This over-rides the material pre-sets. Default is 0.6.
        ambientKey : str , optional
            The dictionary key under which the ambient value (float) is stored. Default is None.
        diffuse : float , optional
            Controls the strength of diffuse (Lambertian) lighting based on the angle
            between the light direction and the surface normal.
            Higher values enhance shape perception through shading.
            Typical range is [0, 1]. This over-rides the material pre-sets. Default is None.
        diffuseKey : str , optional
            The dictionary key under which the diffuse value (float) is stored. Default is None.
        specular : float , optional
            Controls the intensity of specular (mirror-like) highlights on the surface.
            Higher values produce sharper and brighter highlights, giving a glossy appearance.
            Typical range is [0, 1]. This over-rides the material pre-sets. Default is None.
        specularKey : str , optional
            The dictionary key under which the specular value (float) is stored. Default is None.
        roughness : float , optional
            Controls the spread of specular highlights on the surface.
            Lower values result in sharp, concentrated highlights (smooth surfaces),
            while higher values produce broader, softer highlights (rough surfaces).
            Typical range is [0, 1]. This over-rides the material pre-sets. Default is None.
        roughnessKey : str , optional
            The dictionary key under which the roughness value (float) is stored. Default is None.
        colorScale : str , optional
            The desired type of plotly color scales to use (e.g. "Viridis", "Plasma"). Default is "Viridis". For a full list of names, see https://plotly.com/python/builtin-colorscales/.
        mantissa : int , optional
            The number of decimal places to round the result to. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        
        Returns
        -------
        list
            The vertex, edge, and face data list.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Face import Face
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Color import Color
        from topologicpy.Helper import Helper
        from time import time
        
        materials = {
            "chalk": {"ambient":1.0, "diffuse":0.4, "specular":0.0, "roughness":1.0},
            "concrete": {"ambient":0.85, "diffuse":0.75, "specular":0.05, "roughness":0.9},
            "eggshell": {"ambient":0.65, "diffuse":0.85, "specular":0.25, "roughness":0.45},
            "glossy": {"ambient":0.5, "diffuse":0.9, "specular":0.6, "roughness":0.1},
            "matte": {"ambient":0.9, "diffuse":0.7, "specular":0.0, "roughness":1.0},
            "metallic": {"ambient":0.3, "diffuse":0.8, "specular":0.9, "roughness":0.2},
            "plastic": {"ambient":0.6, "diffuse":0.9, "specular":0.2, "roughness":0.4},
            "default": {"ambient":None, "diffuse":None, "specular":None, "roughness":None}
        }
        def closest_index(input_value, values):
            return int(min(range(len(values)), key=lambda i: abs(values[i] - input_value)))


        def faceData(vertices, faces, dictionaries=None,
                     color="#FAFAFA",
                     colorKey=None,
                     opacity=0.5,
                     opacityKey=None,
                     ambient=0.6,
                     diffuse=0.9,
                     specular=0.2,
                     roughness=0.4,
                     labelKey=None, groupKey=None,
                     minGroup=None, maxGroup=None, groups=[], legendLabel="Topology Faces",
                     legendGroup=3, legendRank=3, showLegend=True, intensities=None, colorScale="viridis"):
            x = []
            y = []
            z = []
            for v in vertices:
                x.append(v[0])
                y.append(v[1])
                z.append(v[2])
            i = []
            j = []
            k = []
            labels = []
            groupList = []
            label = ""
            group = ""
            color = Color.AnyToHex(color)
            if colorKey or labelKey or groupKey:
                if groups:
                    if len(groups) > 0:
                        if type(groups[0]) == int or type(groups[0]) == float:
                            if not minGroup:
                                minGroup = min(groups)
                            if not maxGroup:
                                maxGroup = max(groups)
                        else:
                            minGroup = 0
                            maxGroup = len(groups) - 1
                else:
                    minGroup = 0
                    maxGroup = 1
                n = len(str(len(faces)))
                for m, f in enumerate(faces):
                    i.append(f[0])
                    j.append(f[1])
                    k.append(f[2])
                    label = ""
                    group = None
                    groupList.append(Color.AnyToHex(color)) # Store a default color for that face
                    labels.append("Face_"+str(m+1).zfill(n))
                    if len(dictionaries) > 0:
                        d = dictionaries[m]
                        if d:
                            if not colorKey == None:
                                d_color = Dictionary.ValueAtKey(d, key=colorKey) or color
                                groupList[m] = Color.AnyToHex(d_color) #Replace the default color by the dictionary color.
                            if not labelKey == None:
                                label = Dictionary.ValueAtKey(d, key=labelKey)
                                if not label == None:
                                    labels[m] = str(label) # Replace the default label with the dictionary label
                            if not groupKey == None:
                                group = Dictionary.ValueAtKey(d, key=groupKey) or None
                        
                        if group == None:
                            pass # do nothing because the default color will be used.
                        elif type(group) == int or type(group) == float:
                            if group < minGroup:
                                group = minGroup
                            if group > maxGroup:
                                group = maxGroup
                            f_color = Color.ByValueInRange(group, minValue=minGroup, maxValue=maxGroup, colorScale=colorScale)
                            groupList[m] = Color.AnyToHex(f_color) # Replace the default color by the group value.
                        else:
                            f_color = Color.ByValueInRange(groups.index(group), minValue=minGroup, maxValue=maxGroup, colorScale=colorScale)
                            groupList[m] = Color.AnyToHex(f_color)
            else:
                for f in faces:
                    i.append(f[0])
                    j.append(f[1])
                    k.append(f[2])

            if len(groupList) == 0:
                groupList = None
            if len(labels) == 0:
                labels = ""
            if material == "default":
                lighting = {"facenormalsepsilon": 0}
            else:
                lighting = dict(ambient=ambient, diffuse=diffuse, specular=specular, roughness=roughness)
            fData = go.Mesh3d(
                    x = x,
                    y = y,
                    z = z,
                    i = i,
                    j = j,
                    k = k,
                    name = legendLabel,
                    showlegend = showLegend,
                    legendgroup = legendGroup,
                    legendrank = legendRank,
                    color = color,
                    facecolor = groupList,
                    colorscale = Plotly.ColorScale(colorScale),
                    cmin = 0,
                    cmax = 1,
                    intensity = intensities,
                    opacity = opacity,
                    hoverinfo = 'text',
                    text = labels,
                    hovertext = labels,
                    showscale = False,
                    flatshading = flatShading,
                    lighting = lighting
                )
            return fData

        if not Topology.IsInstance(topology, "Topology"):
            return None
    
        intensityList = []
        alt_intensities = []
        data = []
        v_list = []
        
        if not isinstance(colorScale, str):
            colorScale = "viridis"
        if Topology.Type(topology) == Topology.TypeID("Vertex"):
            tp_vertices = [topology]
        else:
            tp_vertices = Topology.Vertices(topology, silent=True)
        
        if isinstance(intensities, list):
            if len(intensities) == 0:
                intensities = None
    
        if not (tp_vertices == None or tp_vertices == []):
            vertices = []
            v_dictionaries = []
            intensityList = []
            
            if intensityKey:
                for i, tp_v in enumerate(tp_vertices):
                    vertices.append([Vertex.X(tp_v, mantissa=mantissa), Vertex.Y(tp_v, mantissa=mantissa), Vertex.Z(tp_v, mantissa=mantissa)])
                    d = Topology.Dictionary(tp_v)
                    if d:
                        v = Dictionary.ValueAtKey(d, key=intensityKey)
                        if not v == None:
                            alt_intensities.append(v)
                            v_list.append(v)
                        else:
                            alt_intensities.append(0)
                            v_list.append(0)
                    else:
                        alt_intensities.append(0)
                        v_list.append(0)
                alt_intensities = list(set(alt_intensities))
                alt_intensities.sort()
                if isinstance(intensities, list):
                    if len(intensities) > 0:
                        alt_intensities = intensities
                min_i = min(alt_intensities)
                max_i = max(alt_intensities)
                for i, tp_v in enumerate(tp_vertices):
                    v = v_list[i]      
                    ci = closest_index(v_list[i], alt_intensities)
                    value = (intensities[ci] if isinstance(intensities, list) and len(intensities) > ci else alt_intensities[ci])
                    if (max_i - min_i) == 0:
                        value = 0
                    else:
                        value = (value - min_i)/(max_i - min_i)
                    intensityList.append(value)
            if all(x == 0 for x in intensityList):
                intensityList = None
            if showVertices:
                if len(vertices) == 0:
                    for i, tp_v in enumerate(tp_vertices):
                        if vertexColorKey or vertexSizeKey or vertexBorderColorKey or vertexBorderWidthKey or vertexLabelKey or vertexGroupKey:
                            d = Topology.Dictionary(tp_v)
                            v_dictionaries.append(d)
                        vertices.append([Vertex.X(tp_v, mantissa=mantissa), Vertex.Y(tp_v, mantissa=mantissa), Vertex.Z(tp_v, mantissa=mantissa)])
                data.extend(Plotly.vertexData(vertices,
                                              dictionaries=v_dictionaries,
                                              color=vertexColor,
                                              colorKey=vertexColorKey,
                                              size=vertexSize,
                                              sizeKey=vertexSizeKey,
                                              borderColor=vertexBorderColor,
                                              borderWidth=vertexBorderWidth,
                                              borderColorKey=vertexBorderColorKey,
                                              borderWidthKey=vertexBorderWidthKey,
                                              labelKey=vertexLabelKey,
                                              showVertexLabel=showVertexLabel,
                                              vertexLabelFontSize=vertexLabelFontSize,
                                              groupKey=vertexGroupKey,
                                              minGroup=vertexMinGroup,
                                              maxGroup=vertexMaxGroup,
                                              groups=vertexGroups,
                                              legendLabel=vertexLegendLabel,
                                              legendGroup=vertexLegendGroup,
                                              legendRank=vertexLegendRank,
                                              showLegend=showVertexLegend,
                                              colorScale=colorScale))
            
        if showEdges and Topology.Type(topology) > Topology.TypeID("Vertex"):
            if Topology.Type(topology) == Topology.TypeID("Edge"):
                tp_edges = [topology]
            else:
                tp_edges = Topology.Edges(topology)
            if not (tp_edges == None or tp_edges == []):
                e_dictionaries = []
                if edgeColorKey or edgeWidthKey or edgeLabelKey or edgeGroupKey:
                    for tp_edge in tp_edges:
                        e_dictionaries.append(Topology.Dictionary(tp_edge))
                        
                e_cluster = Cluster.ByTopologies(tp_edges)
                geo = Topology.Geometry(e_cluster, mantissa=mantissa)
                vertices = geo['vertices']
                edges = geo['edges']
                if len(edges) > 0:
                    data.extend(Plotly.edgeData(vertices, edges, dictionaries=e_dictionaries, color=edgeColor, colorKey=edgeColorKey, width=edgeWidth, widthKey=edgeWidthKey, dash=edgeDash, dashKey=edgeDashKey, directed=directed, arrowSize=arrowSize, arrowSizeKey=arrowSizeKey, labelKey=edgeLabelKey, showEdgeLabel=showEdgeLabel, groupKey=edgeGroupKey, minGroup=edgeMinGroup, maxGroup=edgeMaxGroup, groups=edgeGroups, legendLabel=edgeLegendLabel, legendGroup=edgeLegendGroup, legendRank=edgeLegendRank, showLegend=showEdgeLegend, colorScale=colorScale))
        
        if showFaces and Topology.Type(topology) >= Topology.TypeID("Face"):
            d = Topology.Dictionary(topology)
            if not faceColorKey == None:
                faceColor = Dictionary.ValueAtKey(d, faceColorKey, faceColor)
            if not faceOpacityKey == None:
                d_opacity = Dictionary.ValueAtKey(d, key=faceOpacityKey)
                if not d_opacity == None:
                    if 0 <= d_opacity <= 1:
                        faceOpacity = d_opacity

            if not materialKey == None:
                d_material = Dictionary.ValueAtKey(d, key=materialKey)
                if not d_material == None and isinstance(d_material, str):
                    if d_material.lower() in list(materials.keys()):
                        material = d_material
            if not material == None and isinstance(material, str):
                material = material.lower()
            if not material in list(materials.keys()):
                material = "plastic"
            if not ambientKey == None:
                d_ambient = Dictionary.ValueAtKey(d, key=ambientKey)
                if not d_ambient == None:
                    if 0 <= d_ambient <= 1:
                        ambient = d_ambient
            if not diffuseKey == None:
                d_diffuse = Dictionary.ValueAtKey(d, key=diffuseKey)
                if not d_diffuse == None:
                    if 0 <= d_diffuse <= 1:
                        diffuse = d_diffuse
            if not specularKey == None:
                d_specular = Dictionary.ValueAtKey(d, key=specularKey)
                if not d_specular == None:
                    if 0 <= d_specular <= 1:
                        specular = d_specular
            if not roughnessKey == None:
                d_roughness = Dictionary.ValueAtKey(d, key=roughnessKey)
                if not d_roughness == None:
                    if 0 <= d_roughness <= 1:
                        roughness = d_roughness
            if ambient == None:
                ambient = materials[material]['ambient']
            if diffuse == None:
                diffuse = materials[material]['diffuse']
            if specular == None:
                specular = materials[material]['specular']
            if roughness == None:
                roughness = materials[material]['roughness']
            if Topology.IsInstance(topology, "Face"):
                tp_faces = [topology]
            else:
                tp_faces = Topology.Faces(topology)
            if not(tp_faces == None or tp_faces == []):
                f_dictionaries = []
                all_triangles = []
                for tp_face in tp_faces:
                    triangles = Face.Triangulate(tp_face, tolerance=tolerance, silent=silent)
                    if isinstance(triangles, list):
                        for tri in triangles:
                            d = Topology.Dictionary(tp_face)
                            f_dictionaries.append(d)
                            if d:
                                tri = Topology.SetDictionary(tri, d, silent=True)
                            all_triangles.append(tri)
                if len(all_triangles) > 0:
                    f_cluster = Cluster.ByTopologies(all_triangles)
                    geo = Topology.Geometry(f_cluster, mantissa=mantissa)
                    vertices = geo['vertices']
                    faces = geo['faces']
                    if len(faces) > 0:
                        data.append(faceData(vertices, faces, dictionaries=f_dictionaries, color=faceColor, colorKey=faceColorKey, opacity=faceOpacity, opacityKey=faceOpacityKey,
                                             ambient=ambient, diffuse=diffuse, specular=specular, roughness=roughness,
                                             labelKey=faceLabelKey, groupKey=faceGroupKey, minGroup=faceMinGroup, maxGroup=faceMaxGroup, groups=faceGroups, legendLabel=faceLegendLabel, legendGroup=faceLegendGroup, legendRank=faceLegendRank, showLegend=showFaceLegend, intensities=intensityList, colorScale=colorScale))
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

        # Ensure Plotly class is accessible in this scope
        # (This method lives inside topologicpy.Plotly.Plotly)
        try:
            Plotly  # noqa: B018
        except NameError:
            # Fallback import if called from elsewhere
            from topologicpy.Plotly import Plotly as Plotly  # type: ignore

        # --- Validate matrix
        if not isinstance(matrix, (list, np.ndarray)):
            warnings.warn("Plotly.FigureByConfusionMatrix - Error: The input matrix is not a list or numpy array. Returning None.")
            return None

        m = np.array(matrix)
        if m.ndim != 2:
            warnings.warn("Plotly.FigureByConfusionMatrix - Error: The input matrix is not 2D. Returning None.")
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
        if minValue is None:
            minValue = float(np.nanmin(m)) if m.size else 0.0
        if maxValue is None:
            maxValue = float(np.nanmax(m)) if m.size else 1.0

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
            backgroundColor=Color.AnyToHex(backgroundColor),
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
            print("Plotly.FigureByMatrix - Installing required numpy library.")
            try:
                os.system("pip install numpy")
            except Exception:
                os.system("pip install numpy --user")
            try:
                import numpy as np
            except Exception:
                warnings.warn("Plotly.FigureByMatrix - Error: Could not import numpy. Please install numpy manually. Returning None.")
                return None

        try:
            Plotly  # noqa: B018
        except NameError:
            from topologicpy.Plotly import Plotly as Plotly  # type: ignore

        if not isinstance(matrix, (list, np.ndarray)):
            warnings.warn("Plotly.FigureByMatrix - Error: The input matrix is not a list or numpy array. Returning None.")
            return None

        m = np.array(matrix)
        if m.ndim != 2:
            warnings.warn("Plotly.FigureByMatrix - Error: The input matrix is not 2D. Returning None.")
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
        if minValue is None:
            minValue = float(np.nanmin(m)) if m.size else 0.0
        if maxValue is None:
            maxValue = float(np.nanmax(m)) if m.size else 1.0

        denom = (maxValue - minValue) if (maxValue - minValue) != 0 else 1.0

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
            paper_bg = Color.AnyToHex(backgroundColor)
            plot_bg = Color.AnyToHex(backgroundColor)
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
                t = float((val - minValue) / denom)
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
    def FigureByCorrelation(actual,
                            predicted,
                            title="Correlation between Actual and Predicted Values",
                            xTitle="Actual Values",
                            yTitle="Predicted Values",
                            showIdentity=True,
                            showBestFit=True,
                            dotSize=6,
                            dotColor="blue",
                            lineColor="red",
                            width=800,
                            height=600,
                            theme='default',
                            backgroundColor='rgba(0,0,0,0)',
                            marginLeft=0,
                            marginRight=0,
                            marginTop=40,
                            marginBottom=0,
                            ):
        """
        Returns a Plotly Figure showing the correlation between the input actual and predicted values. Actual values are displayed on the X-Axis, Predicted values are displayed on the Y-Axis.

        Parameters
        ----------
        actual : list
            The actual values to display.
        predicted : list
            The predicted values to display.
        title : str , optional
            The desired title to display. Default is "Correlation between Actual and Predicted Values".
        xTitle : str , optional
            The desired X-axis title to display. Default is "Actual Values".
        yTitle : str , optional
            The desired Y-axis title to display. Default is "Predicted Values".
        showIdentity : bool, optional
            If set to true, shows the 45 degree line.
        showBestFit : bool, optional
            If set to True, draws the best fit line through the data.
        dotSize : int, optional
            The marker size
        dotColor : str , optional
            The desired color of the dots. This can be any plotly color string and may be specified as:
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is 'blue'.
        lineColor : str , optional
            The desired color of the best fit line. This can be any plotly color string and may be specified as:
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is 'red'.
        width : int , optional
            The desired width of the figure. Default is 800.
        height : int , optional
            The desired height of the figure. Default is 600.
        theme : str , optional
            The plotly color scheme to use. The options are "dark", "light", "default". Default is "default".
        backgroundColor : list or str , optional
            The desired background color. This can be any color list or plotly color string and may be specified as:
            - An rgb list (e.g. [255,0,0])
            - A cmyk list (e.g. [0.5, 0, 0.25, 0.2])
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is 'rgba(0,0,0,0)' (transparent).
        marginLeft : int , optional
            The desired left margin in pixels. Default is 0.
        marginRight : int , optional
            The desired right margin in pixels. Default is 0.
        marginTop : int , optional
            The desired top margin in pixels. Default is 40.
        marginBottom : int , optional
            The desired bottom margin in pixels. Default is 0.
        
        Returns
        -------
        plotly.Figure
            The created plotly figure.
        
        """

        import numpy as np
        import plotly.graph_objects as go

        # --- Safety
        if actual is None or predicted is None:
            return None

        x = np.array(actual).reshape(-1).astype(float)
        y = np.array(predicted).reshape(-1).astype(float)

        if len(x) == 0:
            return None

        # --- Metrics
        eps = 1e-12
        mae = float(np.mean(np.abs(y - x)))
        rmse = float(np.sqrt(np.mean((y - x) ** 2)))
        ss_res = float(np.sum((x - y) ** 2))
        ss_tot = float(np.sum((x - np.mean(x)) ** 2))
        r2 = 1.0 - ss_res / (ss_tot + eps)

        # --- Figure
        fig = go.Figure()

        # Scatter
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="Predictions",
            marker=dict(
                size=int(dotSize),
                color=dotColor,
                opacity=0.8
            )
        ))

        mn = float(min(np.min(x), np.min(y)))
        mx = float(max(np.max(x), np.max(y)))

        # Identity line
        if showIdentity:
            fig.add_trace(go.Scatter(
                x=[mn, mx],
                y=[mn, mx],
                mode="lines",
                name="Identity (y=x)",
                line=dict(color="black", dash="dash"),
                hoverinfo="skip"
            ))

        # Best-fit line
        if showBestFit and len(x) >= 2:
            a, b = np.polyfit(x, y, 1)
            fig.add_trace(go.Scatter(
                x=[mn, mx],
                y=[a * mn + b, a * mx + b],
                mode="lines",
                name=f"Best fit (y={a:.3g}x+{b:.3g})",
                line=dict(color=lineColor),
                hoverinfo="skip"
            ))

        # --- Layout
        fig.update_layout(
            title=f"{title} — MAE={mae:.4g}, RMSE={rmse:.4g}, R²={r2:.4g}",
            xaxis_title=xTitle,
            yaxis_title=yTitle,
            width=width,
            height=height,
            template="plotly_white" if theme == "default" else f"plotly_{theme}",
            paper_bgcolor=backgroundColor,
            plot_bgcolor=backgroundColor,
            margin=dict(
                l=marginLeft,
                r=marginRight,
                t=marginTop,
                b=marginBottom
            )
        )

        # --- Enforce square axes (important for parity plots)
        fig.update_xaxes(scaleanchor="y", scaleratio=1)
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

        return fig

    @staticmethod
    def FigureByDataFrame(dataFrame,
             labels=[],
             width=950,
             height=500,
             title="Untitled",
             xTitle="X Axis",
             xSpacing=1,
             yTitle="Y Axis",
             ySpacing=1.0,
             useMarkers=False,
             chartType="Line",
             backgroundColor='rgba(0,0,0,0)',
             gridColor = 'lightgray',
             marginLeft=0,
             marginRight=0,
             marginTop=40,
             marginBottom=0):
        
        """
        Returns a Plotly Figure of the input dataframe

        Parameters
        ----------
        df : pandas.df
            The pandas dataframe to display.
        data_labels : list
            The labels to use for the data.
        width : int , optional
            The desired width of the figure. Default is 950.
        height : int , optional
            The desired height of the figure. Default is 500.
        title : str , optional
            The chart title. Default is "Training and Testing Results".
        xTitle : str , optional
            The X-axis title. Default is "Epochs".
        xSpacing : float , optional
            The X-axis spacing. Default is 1.0.
        yTitle : str , optional
            The Y-axis title. Default is "Accuracy and Loss".
        ySpacing : float , optional
            The Y-axis spacing. Default is 0.1.
        useMarkers : bool , optional
            If set to True, markers will be displayed. Default is False.
        chartType : str , optional
            The desired type of chart. The options are "Line", "Bar", or "Scatter". It is case insensitive. Default is "Line".
        backgroundColor : list or str , optional
            The desired background color. This can be any color list or plotly color string and may be specified as:
            - An rgb list (e.g. [255,0,0])
            - A cmyk list (e.g. [0.5, 0, 0.25, 0.2])
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is 'rgba(0,0,0,0)' (transparent).
        grid : str , optional
            The desired background color. This can be any plotly color string and may be specified as:
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is 'lightgray'
        marginLeft : int , optional
            The desired left margin in pixels. Default is 0.
        marginRight : int , optional
            The desired right margin in pixels. Default is 0.
        marginTop : int , optional
            The desired top margin in pixels. Default is 40.
        marginBottom : int , optional
            The desired bottom margin in pixels. Default is 0.

        Returns
        -------
        None.

        """
        import plotly.express as px
        from topologicpy.Color import Color
        
        if chartType.lower() == "line":
            figure = px.line(dataFrame, x=labels[0], y=labels[1:], title=title, markers=useMarkers)
        elif chartType.lower() == "bar":
            figure = px.bar(dataFrame, x=labels[0], y=labels[1:], title=title)
        elif chartType.lower() == "scatter":
            figure = px.scatter(dataFrame, x=labels[0], y=labels[1:], title=title)
        else:
            raise NotImplementedError
        
        layout = {
            "width": width,
            "height": height,
            "title": title,
            "xaxis": {"title": xTitle, "dtick": xSpacing, 'gridcolor': gridColor},
            "yaxis": {"title": yTitle, "dtick": ySpacing, 'gridcolor': gridColor},
            "paper_bgcolor": Color.AnyToHex(backgroundColor),
            "plot_bgcolor": Color.AnyToHex(backgroundColor),
            "margin":dict(l=marginLeft, r=marginRight, t=marginTop, b=marginBottom)
        }
        figure.update_layout(layout)
        return figure


    @staticmethod
    def FigureByData(data, width=950, height=500,
                     xAxis=False, yAxis=False, zAxis=False,
                     axisSize=1, backgroundColor='rgba(0,0,0,0)',
                     marginLeft=0, marginRight=0,
                     marginTop=20, marginBottom=0,
                     tolerance = 0.0001):
        """
        Creates a plotly figure.

        Parameters
        ----------
        data : list
            The input list of plotly data.
        width : int , optional
            The width in pixels of the figure. The default value is 950.
        height : int , optional
            The height in pixels of the figure. The default value is 950.
        xAxis : bool , optional
            If set to True the x axis is drawn. Otherwise it is not drawn. Default is False.
        yAxis : bool , optional
            If set to True the y axis is drawn. Otherwise it is not drawn. Default is False.
        zAxis : bool , optional
            If set to True the z axis is drawn. Otherwise it is not drawn. Default is False.
        axisSize : float , optional
            The size of the X, Y, Z, axes. Default is 1.
        backgroundColor : list or str , optional
            The desired background color. This can be any color list or plotly color string and may be specified as:
            - An rgb list (e.g. [255,0,0])
            - A cmyk list (e.g. [0.5, 0, 0.25, 0.2])
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is 'rgba(0,0,0,0)' (transparent).
        marginLeft : int , optional
            The size in pixels of the left margin. The default value is 0.
        marginRight : int , optional
            The size in pixels of the right margin. The default value is 0.
        marginTop : int , optional
            The size in pixels of the top margin. The default value is 20.
        marginBottom : int , optional
            The size in pixels of the bottom margin. The default value is 0.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.
        
        Returns
        -------
        plotly.graph_objs._figure.Figure
            The created plotly figure.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Color import Color
        if not isinstance(data, list):
            return None

        v0 = Vertex.ByCoordinates(0, 0, 0)
        v1 = Vertex.ByCoordinates(axisSize,0,0)
        v2 = Vertex.ByCoordinates(0,axisSize,0)
        v3 = Vertex.ByCoordinates(0,0,axisSize)

        if xAxis:
            xEdge = Edge.ByVertices([v0,v1], tolerance=tolerance)
            xData = Plotly.DataByTopology(xEdge, edgeColor="red", edgeWidth=6, showFaces=False, showEdges=True, showVertices=False, edgeLegendLabel="X-Axis")
            data = data + xData
        if yAxis:
            yEdge = Edge.ByVertices([v0,v2], tolerance=tolerance)
            yData = Plotly.DataByTopology(yEdge, edgeColor="green", edgeWidth=6, showFaces=False, showEdges=True, showVertices=False, edgeLegendLabel="Y-Axis")
            data = data + yData
        if zAxis:
            zEdge = Edge.ByVertices([v0,v3], tolerance=tolerance)
            zData = Plotly.DataByTopology(zEdge, edgeColor="blue", edgeWidth=6, showFaces=False, showEdges=True, showVertices=False, edgeLegendLabel="Z-Axis")
            data = data + zData

        figure = go.Figure(data=data)
        figure.update_layout(
            width=width,
            height=height,
            showlegend=True,
            scene = dict(
                xaxis = dict(visible=False),
                yaxis = dict(visible=False),
                zaxis =dict(visible=False),
                ),
            scene_aspectmode='data',
            paper_bgcolor= Color.AnyToHex(backgroundColor),
            plot_bgcolor= Color.AnyToHex(backgroundColor),
            margin=dict(l=marginLeft, r=marginRight, t=marginTop, b=marginBottom),
            )
        figure.update_xaxes(showgrid=False, zeroline=False, visible=False)
        figure.update_yaxes(showgrid=False, zeroline=False, visible=False)
        return figure

    @staticmethod
    def FigureByJSONFile(file):
        """
        Imports a plotly figure from a JSON file.

        Parameters
        ----------
        file : file object
            The JSON file.

        Returns
        -------
        plotly.graph_objs._figure.Figure
            The imported figure.

        """
        figure = None
        if not file:
            return None
        figure = plotly.io.read_json(file, output_type='Figure', skip_invalid=True, engine=None)
        file.close()
        return figure
    
    @staticmethod
    def FigureByJSONPath(path):
        """
        Imports a plotly figure from a JSON file path.

        Parameters
        ----------
        path : str
            The path to the BRep file.

        Returns
        -------
        plotly.graph_objs._figure.Figure
            The imported figure.

        """
        if not path:
            return None
        try:
            file = open(path)
        except:
            print("Plotly.FigureByJSONPath - Error: the JSON file is not a valid file. Returning None.")
            return None
        return Plotly.FigureByJSONFile(file)

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
        except:
            print("Plotly.FigureByPieChart - Installing required pandas library.")
            try:
                os.system("pip install pandas")
            except:
                os.system("pip install pandas --user")
            try:
                import pandas as pd
            except:
                warnings.warn("Plotly.FigureByPieChart - Error: Could not import pandas. Please install the pandas library manually. Returning None")
                return None
            
        dlist = list(map(list, zip(*data)))
        df = pd.DataFrame(dlist, columns=data['names'])
        fig = px.pie(df, values=values, names=names)
        return fig
    
    @staticmethod
    def FigureByTopology(topology,
             showVertices=True, vertexSize=1.1, vertexColor="black", 
             vertexLabelKey=None, vertexGroupKey=None, vertexGroups=[], 
             vertexMinGroup=None, vertexMaxGroup=None, 
             showVertexLegend=False, vertexLegendLabel="Topology Vertices", vertexLegendRank=1, 
             vertexLegendGroup=1, 

             showEdges=True, edgeWidth=1, edgeColor="black", 
             edgeLabelKey=None, edgeGroupKey=None, edgeGroups=[], 
             edgeMinGroup=None, edgeMaxGroup=None, 
             showEdgeLegend=False, edgeLegendLabel="Topology Edges", edgeLegendRank=2, 
             edgeLegendGroup=2, 

             showFaces=True, faceOpacity=0.5, faceColor="#FAFAFA",
             faceLabelKey=None, faceGroupKey=None, faceGroups=[], 
             faceMinGroup=None, faceMaxGroup=None, 
             showFaceLegend=False, faceLegendLabel="Topology Faces", faceLegendRank=3,
             faceLegendGroup=3, 
             intensityKey=None,
             
             width=950, height=500,
             xAxis=False, yAxis=False, zAxis=False, axisSize=1, backgroundColor='rgba(0,0,0,0)',
             marginLeft=0, marginRight=0, marginTop=20, marginBottom=0, showScale=False,
             
             cbValues=[], cbTicks=5, cbX=-0.15, cbWidth=15, cbOutlineWidth=0, cbTitle="",
             cbSubTitle="", cbUnits="", colorScale="viridis", mantissa=6, tolerance=0.0001):
        """
        Creates a figure from the input topology.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology. This must contain faces and or edges.

        showVertices : bool , optional
            If set to True the vertices will be drawn. Otherwise, they will not be drawn. Default is True.
        vertexSize : float , optional
            The desired size of the vertices. Default is 1.1.
        vertexColor : str , optional
            The desired color of the output vertices. This can be any plotly color string and may be specified as:
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is "black".
        vertexLabelKey : str , optional
            The dictionary key to use to display the vertex label. Default is None.
        vertexGroupKey : str , optional
            The dictionary key to use to display the vertex group. Default is None.
        vertexGroups : list , optional
            The list of vertex groups against which to index the color of the vertex. Default is [].
        vertexMinGroup : int or float , optional
            For numeric vertexGroups, vertexMinGroup is the desired minimum value for the scaling of colors. This should match the type of value associated with the vertexGroupKey. If set to None, it is set to the minimum value in vertexGroups. Default is None.
        edgeMaxGroup : int or float , optional
            For numeric vertexGroups, vertexMaxGroup is the desired maximum value for the scaling of colors. This should match the type of value associated with the vertexGroupKey. If set to None, it is set to the maximum value in vertexGroups. Default is None.
        showVertexLegend : bool, optional
            If set to True, the legend for the vertices of this topology is shown. Otherwise, it isn't. Default is False.
        vertexLegendLabel : str , optional
            The legend label string used to identify vertices. Default is "Topology Vertices".
        vertexLegendRank : int , optional
            The legend rank order of the vertices of this topology. Default is 1.
        vertexLegendGroup : int , optional
            The number of the vertex legend group to which the vertices of this topology belong. Default is 1.
        
        showEdges : bool , optional
            If set to True the edges will be drawn. Otherwise, they will not be drawn. Default is True.
        edgeWidth : float , optional
            The desired thickness of the output edges. Default is 1.
        edgeColor : str , optional
            The desired color of the output edges. This can be any plotly color string and may be specified as:
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is "black".
        edgeLabelKey : str , optional
            The dictionary key to use to display the edge label. Default is None.
        edgeGroupKey : str , optional
            The dictionary key to use to display the edge group. Default is None.
        edgeGroups : list , optional
            The list of edge groups against which to index the color of the edge. Default is [].
        edgeMinGroup : int or float , optional
            For numeric edgeGroups, edgeMinGroup is the desired minimum value for the scaling of colors. This should match the type of value associated with the edgeGroupKey. If set to None, it is set to the minimum value in edgeGroups. Default is None.
        edgeMaxGroup : int or float , optional
            For numeric edgeGroups, edgeMaxGroup is the desired maximum value for the scaling of colors. This should match the type of value associated with the edgeGroupKey. If set to None, it is set to the maximum value in edgeGroups. Default is None.
        showEdgeLegend : bool, optional
            If set to True, the legend for the edges of this topology is shown. Otherwise, it isn't. Default is False.
        edgeLegendLabel : str , optional
            The legend label string used to identify edges. Default is "Topology Edges".
        edgeLegendRank : int , optional
            The legend rank order of the edges of this topology. Default is 2.
        edgeLegendGroup : int , optional
            The number of the edge legend group to which the edges of this topology belong. Default is 2.
        
        showFaces : bool , optional
            If set to True the faces will be drawn. Otherwise, they will not be drawn. Default is True.
        faceOpacity : float , optional
            The desired opacity of the output faces (0=transparent, 1=opaque). Default is 0.5.
        faceColor : str , optional
            The desired color of the output faces. This can be any plotly color string and may be specified as:
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is "#FAFAFA".
        faceLabelKey : str , optional
            The dictionary key to use to display the face label. Default is None.
        faceGroupKey : str , optional
            The dictionary key to use to display the face group. Default is None.
        faceGroups : list , optional
            The list of face groups against which to index the color of the face. This can bhave numeric or string values. This should match the type of value associated with the faceGroupKey. Default is [].
        faceMinGroup : int or float , optional
            For numeric faceGroups, minGroup is the desired minimum value for the scaling of colors. This should match the type of value associated with the faceGroupKey. If set to None, it is set to the minimum value in faceGroups. Default is None.
        faceMaxGroup : int or float , optional
            For numeric faceGroups, maxGroup is the desired maximum value for the scaling of colors. This should match the type of value associated with the faceGroupKey. If set to None, it is set to the maximum value in faceGroups. Default is None.
        showFaceLegend : bool, optional
            If set to True, the legend for the faces of this topology is shown. Otherwise, it isn't. Default is False.
        faceLegendLabel : str , optional
            The legend label string used to idenitfy edges. Default is "Topology Faces".
        faceLegendRank : int , optional
            The legend rank order of the faces of this topology. Default is 3.
        faceLegendGroup : int , optional
            The number of the face legend group to which the faces of this topology belong. Default is 3.
        width : int , optional
            The width in pixels of the figure. The default value is 950.
        height : int , optional
            The height in pixels of the figure. The default value is 950.
        xAxis : bool , optional
            If set to True the x axis is drawn. Otherwise it is not drawn. Default is False.
        yAxis : bool , optional
            If set to True the y axis is drawn. Otherwise it is not drawn. Default is False.
        zAxis : bool , optional
            If set to True the z axis is drawn. Otherwise it is not drawn. Default is False.
        backgroundColor : list or str , optional
            The desired background color. This can be any color list or plotly color string and may be specified as:
            - An rgb list (e.g. [255,0,0])
            - A cmyk list (e.g. [0.5, 0, 0.25, 0.2])
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is 'rgba(0,0,0,0)' (transparent).
        marginLeft : int , optional
            The size in pixels of the left margin. The default value is 0.
        marginRight : int , optional
            The size in pixels of the right margin. The default value is 0.
        marginTop : int , optional
            The size in pixels of the top margin. The default value is 20.
        marginBottom : int , optional
            The size in pixels of the bottom margin. The default value is 0.
        camera : list , optional
            The desired location of the camera). Default is [-1.25, -1.25, 1.25].
        center : list , optional
            The desired center (camera target). Default is [0, 0, 0].
        up : list , optional
            The desired up vector. Default is [0, 0, 1].
        renderer : str , optional
            The desired renderer. See Plotly.Renderers(). Default is "notebook".
        intensityKey : str , optional
            If not None, the dictionary of each vertex is searched for the value associated with the intensity key. This value is then used to color-code the vertex based on the colorScale. Default is None.
        showScale : bool , optional
            If set to True, the colorbar is shown. Default is False.
        cbValues : list , optional
            The input list of values to use for the colorbar. Default is [].
        cbTicks : int , optional
            The number of ticks to use on the colorbar. Default is 5.
        cbX : float , optional
            The x location of the colorbar. Default is -0.15.
        cbWidth : int , optional
            The width in pixels of the colorbar. Default is 15
        cbOutlineWidth : int , optional
            The width in pixels of the outline of the colorbar. Default is 0.
        cbTitle : str , optional
            The title of the colorbar. Default is "".
        cbSubTitle : str , optional
            The subtitle of the colorbar. Default is "".
        cbUnits: str , optional
            The units used in the colorbar. Default is ""
        colorScale : str , optional
            The desired type of plotly color scales to use (e.g. "viridis", "plasma"). Default is "viridis". For a full list of names, see https://plotly.com/python/builtin-colorscales/.
        mantissa : int , optional
            The desired length of the mantissa for the values listed on the colorbar. Default is 6.
        tolerance : float , optional
            The desired tolerance. Default is 0.0001.

        Returns
        -------
        Plotly figure

        """
        from topologicpy.Topology import Topology
        from topologicpy.Color import Color

        if not Topology.IsInstance(topology, "Topology"):
            print("Plotly.FigureByTopology - Error: the input topology is not a valid topology. Returning None.")
            return None
        data = Plotly.DataByTopology(topology=topology,
                       showVertices=showVertices, vertexSize=vertexSize, vertexColor=vertexColor, 
                       vertexLabelKey=vertexLabelKey, vertexGroupKey=vertexGroupKey, vertexGroups=vertexGroups, 
                       vertexMinGroup=vertexMinGroup, vertexMaxGroup=vertexMaxGroup, 
                       showVertexLegend=showVertexLegend, vertexLegendLabel=vertexLegendLabel, vertexLegendRank=vertexLegendRank,
                       vertexLegendGroup=vertexLegendGroup,
                       showEdges=showEdges, edgeWidth=edgeWidth, edgeColor=edgeColor, 
                       edgeLabelKey=edgeLabelKey, edgeGroupKey=edgeGroupKey, edgeGroups=edgeGroups, 
                       edgeMinGroup=edgeMinGroup, edgeMaxGroup=edgeMaxGroup, 
                       showEdgeLegend=showEdgeLegend, edgeLegendLabel=edgeLegendLabel, edgeLegendRank=edgeLegendRank, 
                       edgeLegendGroup=edgeLegendGroup,
                       showFaces=showFaces, faceOpacity=faceOpacity, faceColor=faceColor,
                       faceLabelKey=faceLabelKey, faceGroupKey=faceGroupKey, faceGroups=faceGroups, 
                       faceMinGroup=faceMinGroup, faceMaxGroup=faceMaxGroup, 
                       showFaceLegend=showFaceLegend, faceLegendLabel=faceLegendLabel, faceLegendRank=faceLegendRank,
                       faceLegendGroup=faceLegendGroup, 
                       intensityKey=intensityKey, colorScale=colorScale, tolerance=tolerance)
        figure = Plotly.FigureByData(data=data, width=width, height=height,
                                     xAxis=xAxis, yAxis=yAxis, zAxis=zAxis, axisSize=axisSize,
                                     backgroundColor=Color.AnyToHex(backgroundColor),
                                     marginLeft=marginLeft, marginRight=marginRight,
                                     marginTop=marginTop, marginBottom=marginBottom,
                                     tolerance=tolerance)
        if showScale:
            figure = Plotly.AddColorBar(figure, values=cbValues, nTicks=cbTicks, xPosition=cbX, width=cbWidth, outlineWidth=cbOutlineWidth, title=cbTitle, subTitle=cbSubTitle, units=cbUnits, colorScale=colorScale, mantissa=mantissa)
        return figure
    
    @staticmethod
    def FigureExportToJSON(figure, path, overwrite=False):
        """
        Exports the input plotly figure to a JSON file.

        Parameters
        ----------
        figure : plotly.graph_objs._figure.Figure
            The input plotly figure.
        path : str
            The input file path.
        overwrite : bool , optional
            If set to True the ouptut file will overwrite any pre-existing file. Otherwise, it won't.

        Returns
        -------
        bool
            True if the export operation is successful. False otherwise.

        """
        if not isinstance(figure, plotly.graph_objs._figure.Figure):
            print("Plotly.FigureExportToJSON - Error: The input figure is not a plolty figure. Returning None.")
            return None
        if not isinstance(path, str):
            print("Plotly.FigureExportToJSON - Error: The input path is not a string. Returning None.")
            return None
        # Make sure the file extension is .json
        ext = path[len(path)-5:len(path)]
        if ext.lower() != ".json":
            path = path+".json"
        f = None
        try:
            if overwrite == True:
                f = open(path, "w")
            else:
                f = open(path, "x") # Try to create a new File
        except:
           print("Plotly.FigureExportToJSON - Error: Could not create a new file at the following location: "+path+". Returning None.")
           return None
        if (f):
            plotly.io.write_json(figure, f, validate=True, pretty=False, remove_uids=True, engine=None)
            f.close()    
            return True
        if f:
            try:
                f.close()
            except:
                pass
        return False

    @staticmethod
    def FigureExportToPDF(figure, path, width=1920, height=1200, overwrite=False):
        """
        Exports the input plotly figure to a PDF file.

        Parameters
        ----------
        figure : plotly.graph_objs._figure.Figure
            The input plotly figure.
        path : str
            The input file path.
        width : int, optional
            The width of the exported image in pixels. Default is 1920.
        height : int , optional
            The height of the exported image in pixels. Default is 1200.
        overwrite : bool , optional
            If set to True the ouptut file will overwrite any pre-existing file. Otherwise, it won't.

        Returns
        -------
        bool
            True if the export operation is successful. False otherwise.

        """
        import os
        if not isinstance(figure, plotly.graph_objs._figure.Figure):
            print("Plotly.FigureExportToPDF - Error: The input figure is not a plolty figure. Returning None.")
            return None
        if not isinstance(path, str):
            print("Plotly.FigureExportToPDF - Error: The input path is not a string. Returning None.")
            return None
        # Make sure the file extension is .pdf
        ext = path[len(path)-4:len(path)]
        if ext.lower() != ".pdf":
            path = path+".pdf"
        
        if overwrite == False and os.path.exists(path):
            print("Plotly.FigureExportToPDF - Error: A file already exists at this location and overwrite is set to False. Returning None.")
            return None

        plotly.io.write_image(figure, path, format='pdf', scale=1, width=width, height=height, validate=True, engine='auto')  
        return True
    
    @staticmethod
    def FigureExportToPNG(figure, path, width=1920, height=1200, overwrite=False):
        """
        Exports the input plotly figure to a PNG file.

        Parameters
        ----------
        figure : plotly.graph_objs._figure.Figure
            The input plotly figure.
        path : str
            The input file path.
        width : int, optional
            The width of the exported image in pixels. Default is 1920.
        height : int , optional
            The height of the exported image in pixels. Default is 1200.
        overwrite : bool , optional
            If set to True the ouptut file will overwrite any pre-existing file. Otherwise, it won't.

        Returns
        -------
        bool
            True if the export operation is successful. False otherwise.

        """
        import os
        if not isinstance(figure, plotly.graph_objs._figure.Figure):
            print("Plotly.FigureExportToPNG - Error: The input figure is not a plolty figure. Returning None.")
            return None
        if not isinstance(path, str):
            print("Plotly.FigureExportToPNG - Error: The input path is not a string. Returning None.")
            return None
        # Make sure the file extension is .png
        ext = path[len(path)-4:len(path)]
        if ext.lower() != ".png":
            path = path+".png"
        
        if overwrite == False and os.path.exists(path):
            print("Plotly.FigureExportToPNG - Error: A file already exists at this location and overwrite is set to False. Returning None.")
            return None

        plotly.io.write_image(figure, path, format='png', scale=1, width=width, height=height, validate=True, engine='auto')  
        return True
    
    @staticmethod
    def FigureExportToSVG(figure, path, width=1920, height=1200, overwrite=False):
        """
        Exports the input plotly figure to a SVG file.

        Parameters
        ----------
        figure : plotly.graph_objs._figure.Figure
            The input plotly figure.
        path : str
            The input file path.
        width : int, optional
            The width of the exported image in pixels. Default is 1920.
        height : int , optional
            The height of the exported image in pixels. Default is 1200.
        overwrite : bool , optional
            If set to True the ouptut file will overwrite any pre-existing file. Otherwise, it won't.

        Returns
        -------
        bool
            True if the export operation is successful. False otherwise.

        """
        import os
        if not isinstance(figure, plotly.graph_objs._figure.Figure):
            print("Plotly.FigureExportToSVG - Error: The input figure is not a plolty figure. Returning None.")
            return None
        if not isinstance(path, str):
            print("Plotly.FigureExportToSVG - Error: The input path is not a string. Returning None.")
            return None
        # Make sure the file extension is .svg
        ext = path[len(path)-4:len(path)]
        if ext.lower() != ".svg":
            path = path+".svg"
        
        if overwrite == False and os.path.exists(path):
            print("Plotly.FigureExportToSVG - Error: A file already exists at this location and overwrite is set to False. Returning None.")
            return None

        plotly.io.write_image(figure, path, format='svg', scale=1, width=width, height=height, validate=True, engine='auto')  
        return True
    
    @staticmethod
    def SetCamera(figure, camera=None, center=None, up=None, projection="perspective"):
        """
        Sets the camera for the input figure.

        Parameters
        ----------
        figure : plotly.graph_objs._figure.Figure
            The input plotly figure.
        camera : list , optional
            The desired location of the camera. Default is [-1.25, -1.25, 1.25].
        center : list , optional
            The desired center (camera target). Default is [0, 0, 0].
        up : list , optional
            The desired up vector. Default is [0, 0, 1].
        projection : str , optional
            The desired type of projection. The options are "orthographic" or "perspective". It is case insensitive. Default is "perspective"
        
        Returns
        -------
        plotly.graph_objs._figure.Figure
            The updated figure

        """
        if not isinstance(camera, list):
            camera = [-1.25, -1.25, 1.25]
        if not isinstance(center, list):
            center = [0, 0, 0]
        if not isinstance(up, list):
            up = [0, 0, 1]
        projection = projection.lower()
        if projection in "orthographic":
            projection = "orthographic"
        else:
            projection = "perspective"
        scene_camera = dict(
        up=dict(x=up[0], y=up[1], z=up[2]),
        eye=dict(x=camera[0], y=camera[1], z=camera[2]),
        center=dict(x=center[0], y=center[1], z=center[2]),
        projection=dict(type=projection)
        )
        figure.update_layout(scene_camera=scene_camera)
        return figure

    @staticmethod
    def Show(figure, camera=None, center=None, up=None, renderer=None, projection="perspective"):
        """
        Shows the input figure.

        Parameters
        ----------
        figure : plotly.graph_objs._figure.Figure
            The input plotly figure.
        camera : list , optional
            The desired location of the camera. Default is [0, 0, 0].
        center : list , optional
            The desired center (camera target). Default is [0, 0, 0].
        up : list , optional
            The desired up vector. Default is [0, 0, 1].
        renderer : str , optional
            The desired renderer. See Plotly.Renderers(). If set to None, the code will attempt to discover the most suitable renderer. Default is None.
        projection : str, optional
            The desired type of projection. The options are "orthographic" or "perspective". It is case insensitive. Default is "perspective"

        
        Returns
        -------
        None
            
        """

        if figure == None:
            print("Plotly.Show - Error: The input is NULL. Returning None.")
            return None
        if not isinstance(camera, list):
            camera = [-1.25, -1.25, 1.25]
        if not isinstance(center, list):
            center = [0, 0, 0]
        if not isinstance(up, list):
            up = [0, 0, 1]
        if not isinstance(figure, plotly.graph_objs._figure.Figure):
            print("Plotly.Show - Error: The input is not a figure. Returning None.")
            return None
        if renderer == None:
            renderer = Plotly.Renderer()
        if not renderer.lower() in Plotly.Renderers():
            print("Plotly.Show - Error: The input renderer is not in the approved list of renderers. Returning None.")
            return None
        # Set up camera projection
        if "ortho" in projection.lower():
            camera_settings = dict(eye=dict(x=camera[0], y=camera[1], z=camera[2]),
                                center=dict(x=center[0], y=center[1], z=center[2]),
                                up=dict(x=up[0], y=up[1], z=up[2]),
                                projection=dict(type="orthographic"))
        else:
            camera_settings = dict(eye=dict(x=camera[0], y=camera[1], z=camera[2]),
                                center=dict(x=center[0], y=center[1], z=center[2]),
                                up=dict(x=up[0], y=up[1], z=up[2]),
                                projection=dict(type="perspective"))

        figure.update_layout(
            scene_camera = camera_settings,
            scene=dict(aspectmode="data"),
            autosize=True,
            margin=dict(l=40, r=40, t=40, b=40)
            )
        if renderer.lower() == "offline":
            ofl.plot(figure)
        else:
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
        """
        Returns a list of the available plotly renderers.

        Parameters
        ----------
        
        Returns
        -------
        list
            The list of the available plotly renderers.

        """
        return ['plotly_mimetype', 'jupyterlab', 'nteract', 'vscode',
         'notebook', 'notebook_connected', 'kaggle', 'azure', 'colab',
         'cocalc', 'databricks', 'json', 'png', 'jpeg', 'jpg', 'svg',
         'pdf', 'browser', 'firefox', 'chrome', 'chromium', 'iframe',
         'iframe_connected', 'sphinx_gallery', 'sphinx_gallery_png', 'offline']

    @staticmethod
    def ExportToImage(figure, path, format="png", width="1920", height="1080"):
        """
        Exports the plotly figure to an image.

        Parameters
        ----------
        figure : plotly.graph_objs._figure.Figure
            The input plotly figure.
        path : str
            The image file path.
        format : str , optional
            The desired format. This can be any of "jpg", "jpeg", "pdf", "png", "svg", or "webp". It is case insensitive. Default is "png". 
        width : int , optional
            The width in pixels of the figure. The default value is 1920.
        height : int , optional
            The height in pixels of the figure. The default value is 1080.
        
        Returns
        -------
        bool
            True if the image was exported sucessfully. False otherwise.

        """
        if not isinstance(figure, plotly.graph_objs._figure.Figure):
            return None
        if not isinstance(path, str):
            return None
        if not format.lower() in ["jpg", "jpeg", "pdf", "png", "svg", "webp"]:
            return None
        returnStatus = False
        try:
            plotly.io.write_image(figure, path, format=format.lower(), scale=None, width=width, height=height, validate=True, engine='auto')
            returnStatus = True
        except:
            returnStatus = False
        return returnStatus
