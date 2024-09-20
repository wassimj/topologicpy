# Copyright (C) 2024
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

import topologic_core as topologic

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
    def AddColorBar(figure, values=[], nTicks=5, xPosition=-0.15, width=15, outlineWidth=0, title="", subTitle="", units="", colorScale="viridis", mantissa: int = 6):
        """
        Adds a color bar to the input figure

        Parameters
        ----------
        figure : plotly.graph_objs._figure.Figure
            The input plotly figure.
        values : list , optional
            The input list of values to use for the color bar. The default is [].
        nTicks : int , optional
            The number of ticks to use on the color bar. The default is 5.
        xPosition : float , optional
            The x location of the color bar. The default is -0.15.
        width : int , optional
            The width in pixels of the color bar. The default is 15
        outlineWidth : int , optional
            The width in pixels of the outline of the color bar. The default is 0.
        title : str , optional
            The title of the color bar. The default is "".
        subTitle : str , optional
            The subtitle of the color bar. The default is "".
        units: str , optional
            The units used in the color bar. The default is ""
        colorScale : str , optional
            The desired type of plotly color scales to use (e.g. "viridis", "plasma"). The default is "viridis". For a full list of names, see https://plotly.com/python/builtin-colorscales/.
            In addition to these, three color-blind friendly scales are included. These are "protanopia", "deuteranopia", and "tritanopia" for red, green, and blue colorblindness respectively.
        mantissa : int , optional
            The desired length of the mantissa for the values listed on the color bar. The default is 6.
        Returns
        -------
        plotly.graph_objs._figure.Figure
            The input figure with the color bar added.

        """
        if not isinstance(figure, plotly.graph_objs._figure.Figure):
            return None
        if units:
            units = "Units: "+units
        minValue = min(values)
        maxValue = max(values)
        step = (maxValue - minValue)/float(nTicks-1)
        r = [round(minValue+i*step, mantissa) for i in range(nTicks)]
        r[-1] = round(maxValue, mantissa)
                # Define the minimum and maximum range of the colorbar
        rs = [str(x) for x in r]

        # Define the colorbar as a trace with no data, x or y coordinates
        colorbar_trace = go.Scatter(
            x=[0],
            y=[0],
            mode="markers",
            showlegend=False,
            marker=dict(
                size=0,
                colorscale=Plotly.ColorScale(colorScale), # choose the colorscale
                cmin=minValue,
                cmax=maxValue,
                color=['rgba(0,0,0,0)'],
                colorbar=dict(
                    x=xPosition,
                    title="<b>"+title+"</b><br>"+subTitle+"<br>"+units, # title of the colorbar
                    ticks="outside", # position of the ticks
                    tickvals=r, # values of the ticks
                    ticktext=rs, # text of the ticks
                    tickmode="array",
                    thickness=width,
                    outlinewidth=outlineWidth,

                )
            )
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
                "green"," greenyellow","honeydew",
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
    def DataByDGL(data, labels):
        """
        Returns a data frame from the DGL data.

        Parameters
        ----------
        data : list
            The data to display.
        labels : list
            The labels to use for the data. The data with the labels in this list will be extracted and used in the returned dataFrame.

        Returns
        -------
        pd.DataFrame
            A pandas dataFrame

        """

        try:
            import pandas as pd
        except:
            print("Plotly - Installing required pandas library.")
            try:
                os.system("pip install pandas")
            except:
                os.system("pip install pandas --user")
            try:
                import pandas as pd
            except:
                warnings.warn("Plotly.DataByDGL - Error: Could not import pandas. Please install the pandas library manually. Returning None")
                return None
        
        if isinstance(data[labels[0]][0], int):
            xAxis_list = list(range(1, data[labels[0]][0]+1))
        else:
            xAxis_list = data[labels[0]][0]
        plot_data = [xAxis_list]
        for i in range(1,len(labels)):
            plot_data.append(data[labels[i]][0][:len(xAxis_list)])

        dlist = list(map(list, zip(*plot_data)))
        df = pd.DataFrame(dlist, columns=labels)
        return df

    @staticmethod
    def DataByGraph(graph,
                    vertexColor: str = "black",
                    vertexSize: float = 6,
                    vertexLabelKey: str = None,
                    vertexGroupKey: str = None,
                    vertexGroups: list = [],
                    showVertices: bool = True,
                    showVertexLabels: bool = False,
                    showVertexLegend: bool = False,
                    edgeColor: str = "black",
                    edgeWidth: float = 1,
                    edgeLabelKey: str = None,
                    edgeGroupKey: str = None,
                    edgeGroups: list = [],
                    showEdges: bool = True,
                    showEdgeLabels: bool = False,
                    showEdgeLegend: bool = False,
                    colorScale: str = "viridis",
                    mantissa: int = 6):
        """
        Creates plotly vertex and edge data from the input graph.

        Parameters
        ----------
        graph : topologic_core.Graph
            The input graph.
        vertexColor : str , optional
            The desired color of the output vertices. This can be any plotly color string and may be specified as:
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is "black".
        vertexSize : float , optional
            The desired size of the vertices. The default is 6.
        vertexLabelKey : str , optional
            The dictionary key to use to display the vertex label. The default is None.
        vertexGroupKey : str , optional
            The dictionary key to use to display the vertex group. The default is None.
        vertexGroups : list , optional
            The list of vertex groups against which to index the color of the vertex. The default is [].
        showVertices : bool , optional
            If set to True the vertices will be drawn. Otherwise, they will not be drawn. The default is True.
        showVertexLabels : bool , optional
            If set to True, the vertex labels are shown permenantely on screen. Otherwise, they are not. The default is False.
        showVertexLegend : bool , optional
            If set to True the vertex legend will be drawn. Otherwise, it will not be drawn. The default is False.
        edgeColor : str , optional
            The desired color of the output edges. This can be any plotly color string and may be specified as:
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is "black".
        edgeWidth : float , optional
            The desired thickness of the output edges. The default is 1.
        edgeLabelKey : str , optional
            The dictionary key to use to display the edge label. The default is None.
        edgeGroupKey : str , optional
            The dictionary key to use to display the edge group. The default is None.
        edgeGroups : list , optional
            The list of groups to use for indexing the color of edges. The default is None.
        showEdges : bool , optional
            If set to True the edges will be drawn. Otherwise, they will not be drawn. The default is True.
        showEdgeLabels : bool , optional
            If set to True, the edge labels are shown permenantely on screen. Otherwise, they are not. The default is False.
        showEdgeLegend : bool , optional
            If set to True the edge legend will be drawn. Otherwise, it will not be drawn. The default is False.
        colorScale : str , optional
            The desired type of plotly color scales to use (e.g. "Viridis", "Plasma"). The default is "Viridis". For a full list of names, see https://plotly.com/python/builtin-colorscales/.
        Returns
        -------
        list
            The vertex and edge data list.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology
        from topologicpy.Graph import Graph
        import plotly.graph_objs as go

        if not Topology.IsInstance(graph, "Graph"):
            return None
        v_labels = []
        v_groupList = []
        data = []
        if showVertices:
            vertices = Graph.Vertices(graph)
            if vertexLabelKey or vertexGroupKey:
                for v in vertices:
                    Xn=[Vertex.X(v, mantissa=mantissa) for v in vertices] # x-coordinates of nodes
                    Yn=[Vertex.Y(v, mantissa=mantissa) for v in vertices] # y-coordinates of nodes
                    Zn=[Vertex.Z(v, mantissa=mantissa) for v in vertices] # x-coordinates of nodes
                    v_label = ""
                    v_group = ""
                    d = Topology.Dictionary(v)
                    if d:
                        if vertexLabelKey:
                            v_label = str(Dictionary.ValueAtKey(d, key=vertexLabelKey))
                            if v_label == None:
                                v_label = ""
                        if vertexGroupKey:
                            v_group = Dictionary.ValueAtKey(d, key=vertexGroupKey)
                            if v_group == None:
                                v_group = ""
                    try:
                        v_groupList.append(vertexGroups.index(v_group))
                    except:
                        v_groupList.append(len(vertexGroups))
                    v_labels.append(v_label)
            else:
                for v in vertices:
                    Xn=[Vertex.X(v, mantissa=mantissa) for v in vertices] # x-coordinates of nodes
                    Yn=[Vertex.Y(v, mantissa=mantissa) for v in vertices] # y-coordinates of nodes
                    Zn=[Vertex.Z(v, mantissa=mantissa) for v in vertices] # x-coordinates of nodes
            if len(v_labels) < 1:
                v_labels = ""
            if showVertexLabels == True:
                mode = "markers+text"
            else:
                mode = "markers"
            normalized_categories = vertexColor # Start with the default vertexColor
            if len(vertexGroups) > 0:
                # Normalize categories to a range between 0 and 1 for the color scale
                min_category = 0
                max_category = max(len(vertexGroups), 1)
                normalized_categories = [(cat - min_category) / (max_category - min_category) for cat in v_groupList]
            v_trace=go.Scatter3d(x=Xn,
                y=Yn,
                z=Zn,
                mode=mode,
                name='Graph Vertices',
                legendgroup=4,
                legendrank=4,
                showlegend=showVertexLegend,
                marker=dict(symbol='circle',
                                size=vertexSize,
                                color=normalized_categories,
                                colorscale=Plotly.ColorScale(colorScale),
                                cmin = 0,
                                cmax=1,
                                line=dict(color=edgeColor, width=0.5)
                                ),
                text=v_labels,
                hoverinfo='text'
                )
            data.append(v_trace)
        
        if showEdges:
            Xe=[]
            Ye=[]
            Ze=[]
            e_labels = []
            e_groupList = []
            edges = Graph.Edges(graph)
                
            if edgeLabelKey or edgeGroupKey:
                for e in edges:
                    sv = Edge.StartVertex(e)
                    ev = Edge.EndVertex(e)
                    Xe+=[Vertex.X(sv, mantissa=mantissa), Vertex.X(ev, mantissa=mantissa), None] # x-coordinates of edge ends
                    Ye+=[Vertex.Y(sv, mantissa=mantissa), Vertex.Y(ev, mantissa=mantissa), None] # y-coordinates of edge ends
                    Ze+=[Vertex.Z(sv, mantissa=mantissa), Vertex.Z(ev, mantissa=mantissa), None] # z-coordinates of edge ends
                    e_label = ""
                    e_group = ""
                    d = Topology.Dictionary(e)
                    if d:
                        if not edgeLabelKey == None:
                            e_label = str(Dictionary.ValueAtKey(d, key=edgeLabelKey)) or ""
                        if not edgeGroupKey == None:
                            e_group = str(Dictionary.ValueAtKey(d, key=edgeGroupKey)) or ""
                    try:
                        e_groupList.append(edgeGroups.index(e_group))
                    except:
                        e_groupList.append(len(edgeGroups))
                    if not e_label == "" and not e_group == "":
                        e_label = e_label+" ("+e_group+")"
                    e_labels.append(e_label)
            else:
                for e in edges:
                    sv = Edge.StartVertex(e)
                    ev = Edge.EndVertex(e)
                    Xe+=[Vertex.X(sv, mantissa=mantissa), Vertex.X(ev, mantissa=mantissa), None] # x-coordinates of edge ends
                    Ye+=[Vertex.Y(sv, mantissa=mantissa), Vertex.Y(ev, mantissa=mantissa), None] # y-coordinates of edge ends
                    Ze+=[Vertex.Z(sv, mantissa=mantissa), Vertex.Z(ev, mantissa=mantissa), None] # z-coordinates of edge ends

            if len(list(set(e_groupList))) < 2:
                e_groupList = edgeColor
            if len(e_labels) < 1:
                e_labels = ""
            
            if showEdgeLabels == True:
                mode = "lines+text"
            else:
                mode = "lines"
            e_trace=go.Scatter3d(x=Xe,
                                 y=Ye,
                                 z=Ze,
                                 mode=mode,
                                 name='Graph Edges',
                                 legendgroup=5,
                                 legendrank=5,
                                 showlegend=showEdgeLegend,
                                 line=dict(color=e_groupList, width=edgeWidth),
                                 text=e_labels,
                                 hoverinfo='text'
                                )
            data.append(e_trace)

        return data

    @staticmethod
    def DataByTopology(topology,
                       showVertices=True, vertexSize=1.1, vertexColor="black", 
                       vertexLabelKey=None, showVertexLabel=False, vertexGroupKey=None, vertexGroups=[], 
                       vertexMinGroup=None, vertexMaxGroup=None, 
                       showVertexLegend=False, vertexLegendLabel="Topology Vertices", vertexLegendRank=1,
                       vertexLegendGroup=1,
                       showEdges=True, edgeWidth=1, edgeColor="black", 
                       edgeLabelKey=None, showEdgeLabel=False, edgeGroupKey=None, edgeGroups=[], 
                       edgeMinGroup=None, edgeMaxGroup=None, 
                       showEdgeLegend=False, edgeLegendLabel="Topology Edges", edgeLegendRank=2, 
                       edgeLegendGroup=2,
                       showFaces=True, faceOpacity=0.5, faceColor="#FAFAFA",
                       faceLabelKey=None, faceGroupKey=None, faceGroups=[], 
                       faceMinGroup=None, faceMaxGroup=None, 
                       showFaceLegend=False, faceLegendLabel="Topology Faces", faceLegendRank=3,
                       faceLegendGroup=3, 
                       intensityKey=None, intensities=[], colorScale="viridis", mantissa=6, tolerance=0.0001):
        """
        Creates plotly face, edge, and vertex data.

        Parameters
        ----------
        topology : topologic_core.Topology
            The input topology. This must contain faces and or edges.

        showVertices : bool , optional
            If set to True the vertices will be drawn. Otherwise, they will not be drawn. The default is True.
        vertexSize : float , optional
            The desired size of the vertices. The default is 1.1.
        vertexColor : str , optional
            The desired color of the output vertices. This can be any plotly color string and may be specified as:
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is "black".
        vertexLabelKey : str , optional
            The dictionary key to use to display the vertex label. The default is None.
        vertexGroupKey : str , optional
            The dictionary key to use to display the vertex group. The default is None.
        vertexGroups : list , optional
            The list of vertex groups against which to index the color of the vertex. The default is [].
        vertexMinGroup : int or float , optional
            For numeric vertexGroups, vertexMinGroup is the desired minimum value for the scaling of colors. This should match the type of value associated with the vertexGroupKey. If set to None, it is set to the minimum value in vertexGroups. The default is None.
        vertexMaxGroup : int or float , optional
            For numeric vertexGroups, vertexMaxGroup is the desired maximum value for the scaling of colors. This should match the type of value associated with the vertexGroupKey. If set to None, it is set to the maximum value in vertexGroups. The default is None.
        showVertexLegend : bool, optional
            If set to True, the legend for the vertices of this topology is shown. Otherwise, it isn't. The default is False.
        vertexLegendLabel : str , optional
            The legend label string used to identify vertices. The default is "Topology Vertices".
        vertexLegendRank : int , optional
            The legend rank order of the vertices of this topology. The default is 1.
        vertexLegendGroup : int , optional
            The number of the vertex legend group to which the vertices of this topology belong. The default is 1.
        
        showEdges : bool , optional
            If set to True the edges will be drawn. Otherwise, they will not be drawn. The default is True.
        edgeWidth : float , optional
            The desired thickness of the output edges. The default is 1.
        edgeColor : str , optional
            The desired color of the output edges. This can be any plotly color string and may be specified as:
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is "black".
        edgeLabelKey : str , optional
            The dictionary key to use to display the edge label. The default is None.
        edgeGroupKey : str , optional
            The dictionary key to use to display the edge group. The default is None.
        edgeGroups : list , optional
            The list of edge groups against which to index the color of the edge. The default is [].
        edgeMinGroup : int or float , optional
            For numeric edgeGroups, edgeMinGroup is the desired minimum value for the scaling of colors. This should match the type of value associated with the edgeGroupKey. If set to None, it is set to the minimum value in edgeGroups. The default is None.
        edgeMaxGroup : int or float , optional
            For numeric edgeGroups, edgeMaxGroup is the desired maximum value for the scaling of colors. This should match the type of value associated with the edgeGroupKey. If set to None, it is set to the maximum value in edgeGroups. The default is None.
        showEdgeLegend : bool, optional
            If set to True, the legend for the edges of this topology is shown. Otherwise, it isn't. The default is False.
        edgeLegendLabel : str , optional
            The legend label string used to identify edges. The default is "Topology Edges".
        edgeLegendRank : int , optional
            The legend rank order of the edges of this topology. The default is 2.
        edgeLegendGroup : int , optional
            The number of the edge legend group to which the edges of this topology belong. The default is 2.
        
        showFaces : bool , optional
            If set to True the faces will be drawn. Otherwise, they will not be drawn. The default is True.
        faceOpacity : float , optional
            The desired opacity of the output faces (0=transparent, 1=opaque). The default is 0.5.
        faceColor : str , optional
            The desired color of the output faces. This can be any plotly color string and may be specified as:
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is "#FAFAFA".
        faceLabelKey : str , optional
            The dictionary key to use to display the face label. The default is None.
        faceGroupKey : str , optional
            The dictionary key to use to display the face group. The default is None.
        faceGroups : list , optional
            The list of face groups against which to index the color of the face. This can bhave numeric or string values. This should match the type of value associated with the faceGroupKey. The default is [].
        faceMinGroup : int or float , optional
            For numeric faceGroups, minGroup is the desired minimum value for the scaling of colors. This should match the type of value associated with the faceGroupKey. If set to None, it is set to the minimum value in faceGroups. The default is None.
        faceMaxGroup : int or float , optional
            For numeric faceGroups, maxGroup is the desired maximum value for the scaling of colors. This should match the type of value associated with the faceGroupKey. If set to None, it is set to the maximum value in faceGroups. The default is None.
        showFaceLegend : bool, optional
            If set to True, the legend for the faces of this topology is shown. Otherwise, it isn't. The default is False.
        faceLegendLabel : str , optional
            The legend label string used to idenitfy edges. The default is "Topology Faces".
        faceLegendRank : int , optional
            The legend rank order of the faces of this topology. The default is 3.
        faceLegendGroup : int , optional
            The number of the face legend group to which the faces of this topology belong. The default is 3.
        intensityKey: str, optional
            If not None, the dictionary of each vertex is searched for the value associated with the intensity key. This value is then used to color-code the vertex based on the colorScale. The default is None.
        intensities : list , optional
            The list of intensities against which to index the intensity of the vertex. The default is [].
        colorScale : str , optional
            The desired type of plotly color scales to use (e.g. "Viridis", "Plasma"). The default is "Viridis". For a full list of names, see https://plotly.com/python/builtin-colorscales/.
        mantissa : int , optional
            The desired length of the mantissa. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
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
        from time import time
        
        def closest_index(input_value, values):
            return int(min(range(len(values)), key=lambda i: abs(values[i] - input_value)))

        def vertexData(vertices, dictionaries=[], color="black", size=1.1, labelKey=None, showVertexLabel = False, groupKey=None, minGroup=None, maxGroup=None, groups=[], legendLabel="Topology Vertices", legendGroup=1, legendRank=1, showLegend=True, colorScale="Viridis"):
            x = []
            y = []
            z = []
            labels = []
            groupList = []
            label = ""
            group = None
            if labelKey or groupKey:
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
                for m, v in enumerate(vertices):
                    x.append(v[0])
                    y.append(v[1])
                    z.append(v[2])
                    label = ""
                    group = None
                    if len(dictionaries) > 0:
                        d = dictionaries[m]
                        if d:
                            if not labelKey == None:
                                label = str(Dictionary.ValueAtKey(d, key=labelKey)) or ""
                            if not groupKey == None:
                                group = Dictionary.ValueAtKey(d, key=groupKey) or ""
                        try:
                            if group == "":
                                color = 'white'
                            elif type(group) == int or type(group) == float:
                                if group < minGroup:
                                    group = minGroup
                                if group > maxGroup:
                                    group = maxGroup
                                color = Color.ByValueInRange(group, minValue=minGroup, maxValue=maxGroup, colorScale=colorScale)
                            else:
                                color = Color.ByValueInRange(groups.index(group), minValue=minGroup, maxValue=maxGroup, colorScale=colorScale)
                            color = "rgb("+str(color[0])+","+str(color[1])+","+str(color[2])+")"
                            groupList.append(color)
                        except:
                            groupList.append(len(groups))
                        labels.append(label)
            else:
                for v in vertices:
                    x.append(v[0])
                    y.append(v[1])
                    z.append(v[2])
            
            if len(list(set(groupList))) < 2:
                groupList = color
            if len(labels) < 1:
                labels = ""
            if showVertexLabel == True:
                mode = "markers+text"
            else:
                mode = "markers"
            vData= go.Scatter3d(x=x,
                                y=y,
                                z=z,
                                name=legendLabel,
                                showlegend=showLegend,
                                marker=dict(color=groupList,  size=vertexSize),
                                mode=mode,
                                legendgroup=legendGroup,
                                legendrank=legendRank,
                                text=labels,
                                hoverinfo='text',
                                hovertext=labels
                                )
            return vData

        def edgeData(vertices, edges, dictionaries=None, color="black", width=1, labelKey=None, showEdgeLabel = False, groupKey=None, minGroup=None, maxGroup=None, groups=[], legendLabel="Topology Edges", legendGroup=2, legendRank=2, showLegend=True, colorScale="Viridis"):
            x = []
            y = []
            z = []
            labels = []
            groupList = []
            label = ""
            group = ""
            if labelKey or groupKey:
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
                for m, e in enumerate(edges):
                    sv = vertices[e[0]]
                    ev = vertices[e[1]]
                    x+=[sv[0], ev[0], None] # x-coordinates of edge ends
                    y+=[sv[1], ev[1], None] # y-coordinates of edge ends
                    z+=[sv[2], ev[2], None] # z-coordinates of edge ends
                    label = ""
                    group = None
                    if len(dictionaries) > 0:
                        d = dictionaries[m]
                        if d:
                            if not labelKey == None:
                                label = str(Dictionary.ValueAtKey(d, key=labelKey)) or ""
                            if not groupKey == None:
                                group = Dictionary.ValueAtKey(d, key=groupKey) or ""
                        try:
                            if group == "":
                                color = 'white'
                            if type(group) == int or type(group) == float:
                                if group < minGroup:
                                    group = minGroup
                                if group > maxGroup:
                                    group = maxGroup
                                color = Color.ByValueInRange(group, minValue=minGroup, maxValue=maxGroup, colorScale=colorScale)
                            else:
                                color = Color.ByValueInRange(groups.index(group), minValue=minGroup, maxValue=maxGroup, colorScale=colorScale)
                            color = "rgb("+str(color[0])+","+str(color[1])+","+str(color[2])+")"
                            groupList.append(color)
                        except:
                            groupList.append(len(groups))
                        labels.append(label)
            else:
                for e in edges:
                    sv = vertices[e[0]]
                    ev = vertices[e[1]]
                    x+=[sv[0], ev[0], None] # x-coordinates of edge ends
                    y+=[sv[1], ev[1], None] # y-coordinates of edge ends
                    z+=[sv[2], ev[2], None] # z-coordinates of edge ends
                
            if len(list(set(groupList))) < 2:
                    groupList = color
            if len(labels) < 1:
                labels = ""
            if showEdgeLabel == True:
                mode = "lines+text"
            else:
                mode = "lines"
            eData = go.Scatter3d(x=x,
                                y=y,
                                z=z,
                                name=legendLabel,
                                showlegend=showLegend,
                                marker_size=0,
                                mode=mode,
                                line=dict(color=groupList, width=edgeWidth),
                                legendgroup=legendGroup,
                                legendrank=legendRank,
                                text=labels,
                                hoverinfo='text')
            return eData


        def faceData(vertices, faces, dictionaries=None, color="#FAFAFA",
                     opacity=0.5, labelKey=None, groupKey=None,
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
            if labelKey or groupKey:
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
                for m, f in enumerate(faces):
                    i.append(f[0])
                    j.append(f[1])
                    k.append(f[2])
                    label = ""
                    group = ""
                    if len(dictionaries) > 0:
                        d = dictionaries[m]
                        if d:
                            if not labelKey == None:
                                label = str(Dictionary.ValueAtKey(d, key=labelKey)) or ""
                            if not groupKey == None:
                                group = Dictionary.ValueAtKey(d, key=groupKey) or None
                        try:
                            if group == "":
                                color = 'white'
                            elif type(group) == int or type(group) == float:
                                if group < minGroup:
                                    group = minGroup
                                if group > maxGroup:
                                    group = maxGroup
                                color = Color.ByValueInRange(group, minValue=minGroup, maxValue=maxGroup, colorScale=colorScale)
                            else:
                                color = Color.ByValueInRange(groups.index(group), minValue=minGroup, maxValue=maxGroup, colorScale=colorScale)
                            color = "rgb("+str(color[0])+","+str(color[1])+","+str(color[2])+")"
                            groupList.append(color)
                        except:
                            groupList.append(len(groups))
                        labels.append(label)
            else:
                for f in faces:
                    i.append(f[0])
                    j.append(f[1])
                    k.append(f[2])
                
            if len(list(set(groupList))) < 2:
                groupList = None
            if len(labels) < 1:
                labels = ""
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
                    flatshading = True,
                    showscale = False,
                    lighting = {"facenormalsepsilon": 0},
                )
            return fData

        if not Topology.IsInstance(topology, "Topology"):
            return None
    
        intensityList = []
        alt_intensities = []
        data = []
        v_list = []
        
        if Topology.Type(topology) == Topology.TypeID("Vertex"):
            tp_vertices = [topology]
        else:
            tp_vertices = Topology.Vertices(topology)
        
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
                    value = intensities[ci]
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
                        if vertexLabelKey or vertexGroupKey:
                            d = Topology.Dictionary(tp_v)
                            v_dictionaries.append(d)
                        vertices.append([Vertex.X(tp_v, mantissa=mantissa), Vertex.Y(tp_v, mantissa=mantissa), Vertex.Z(tp_v, mantissa=mantissa)])
                data.append(vertexData(vertices, dictionaries=v_dictionaries, color=vertexColor, size=vertexSize, labelKey=vertexLabelKey, showVertexLabel=showVertexLabel, groupKey=vertexGroupKey, minGroup=vertexMinGroup, maxGroup=vertexMaxGroup, groups=vertexGroups, legendLabel=vertexLegendLabel, legendGroup=vertexLegendGroup, legendRank=vertexLegendRank, showLegend=showVertexLegend, colorScale=colorScale))
            
        if showEdges and Topology.Type(topology) > Topology.TypeID("Vertex"):
            if Topology.Type(topology) == Topology.TypeID("Edge"):
                tp_edges = [topology]
            else:
                tp_edges = Topology.Edges(topology)
            if not (tp_edges == None or tp_edges == []):
                e_dictionaries = []
                if edgeLabelKey or edgeGroupKey:
                    for tp_edge in tp_edges:
                        e_dictionaries.append(Topology.Dictionary(tp_edge))
                e_cluster = Cluster.ByTopologies(tp_edges)
                geo = Topology.Geometry(e_cluster, mantissa=mantissa)
                vertices = geo['vertices']
                edges = geo['edges']
                data.append(edgeData(vertices, edges, dictionaries=e_dictionaries, color=edgeColor, width=edgeWidth, labelKey=edgeLabelKey, showEdgeLabel=showEdgeLabel, groupKey=edgeGroupKey, minGroup=edgeMinGroup, maxGroup=edgeMaxGroup, groups=edgeGroups, legendLabel=edgeLegendLabel, legendGroup=edgeLegendGroup, legendRank=edgeLegendRank, showLegend=showEdgeLegend, colorScale=colorScale))
        
        if showFaces and Topology.Type(topology) >= Topology.TypeID("Face"):
            if Topology.IsInstance(topology, "Face"):
                tp_faces = [topology]
            else:
                tp_faces = Topology.Faces(topology)
            if not(tp_faces == None or tp_faces == []):
                f_dictionaries = []
                all_triangles = []
                for tp_face in tp_faces:
                    triangles = Face.Triangulate(tp_face, tolerance=tolerance)
                    if isinstance(triangles, list):
                        for tri in triangles:
                            if faceLabelKey or faceGroupKey:
                                d = Topology.Dictionary(tp_face)
                                f_dictionaries.append(d)
                                if d:
                                    _ = Topology.SetDictionary(tri, d)
                            all_triangles.append(tri)
                if len(all_triangles) > 0:
                    f_cluster = Cluster.ByTopologies(all_triangles)
                    geo = Topology.Geometry(f_cluster, mantissa=mantissa)
                    vertices = geo['vertices']
                    faces = geo['faces']
                    data.append(faceData(vertices, faces, dictionaries=f_dictionaries, color=faceColor, opacity=faceOpacity, labelKey=faceLabelKey, groupKey=faceGroupKey, minGroup=faceMinGroup, maxGroup=faceMaxGroup, groups=faceGroups, legendLabel=faceLegendLabel, legendGroup=faceLegendGroup, legendRank=faceLegendRank, showLegend=showFaceLegend, intensities=intensityList, colorScale=colorScale))
        return data

    @staticmethod
    def FigureByConfusionMatrix(matrix,
             categories=[],
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
             marginBottom=0):
        """
        Returns a Plotly Figure of the input confusion matrix. Actual categories are displayed on the X-Axis, Predicted categories are displayed on the Y-Axis.

        Parameters
        ----------
        matrix : list or numpy.array
            The matrix to display.
        categories : list
            The list of categories to use on the X and Y axes.
        minValue : float , optional
            The desired minimum value to use for the color scale. If set to None, the minmum value found in the input matrix will be used. The default is None.
        maxValue : float , optional
            The desired maximum value to use for the color scale. If set to None, the maximum value found in the input matrix will be used. The default is None.
        title : str , optional
            The desired title to display. The default is "Confusion Matrix".
        xTitle : str , optional
            The desired X-axis title to display. The default is "Actual Categories".
        yTitle : str , optional
            The desired Y-axis title to display. The default is "Predicted Categories".
        width : int , optional
            The desired width of the figure. The default is 950.
        height : int , optional
            The desired height of the figure. The default is 500.
        showScale : bool , optional
            If set to True, a color scale is shown on the right side of the figure. The default is True.
        colorScale : str , optional
            The desired type of plotly color scales to use (e.g. "Viridis", "Plasma"). The default is "Viridis". For a full list of names, see https://plotly.com/python/builtin-colorscales/.
        colorSamples : int , optional
            The number of discrete color samples to use for displaying the data. The default is 10.
        backgroundColor : str , optional
            The desired background color. This can be any plotly color string and may be specified as:
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is 'rgba(0,0,0,0)' (transparent).
        marginLeft : int , optional
            The desired left margin in pixels. The default is 0.
        marginRight : int , optional
            The desired right margin in pixels. The default is 0.
        marginTop : int , optional
            The desired top margin in pixels. The default is 40.
        marginBottom : int , optional
            The desired bottom margin in pixels. The default is 0.
        
        Returns
        -------
        plotly.Figure
            The created plotly figure.
        """
        try:
            import numpy as np
        except:
            print("Plotly.FigureByConfusionMatrix - Installing required numpy library.")
            try:
                os.system("pip install numpy")
            except:
                os.system("pip install numpy --user")
            try:
                import numpy as np
            except:
                warnings.warn("Plotly.FigureByConfusionMatrix - Error: Could not import numpy. Please install numpy manually. Returning None.")
                return None
        
        if not isinstance(matrix, list) and not isinstance(matrix, np.ndarray):
            print("Plotly.FigureByConfusionMatrix - Error: The input matrix is not of the correct type. Returning None.")
            return None
        figure = Plotly.FigureByMatrix(matrix,
             xCategories=categories,
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
             backgroundColor=backgroundColor,
             marginLeft=marginLeft,
             marginRight=marginRight,
             marginTop=marginTop,
             marginBottom=marginBottom)
        layout = {
            "yaxis": {"autorange": "reversed"},
        }
        figure.update_layout(layout)
        return figure
    
    @staticmethod
    def FigureByMatrix(matrix,
             xCategories=[],
             yCategories=[],
             minValue=None,
             maxValue=None,
             title="Matrix",
             xTitle = "X Axis",
             yTitle = "Y Axis",
             width=950,
             height=950,
             showScale = False,
             colorScale='gray',
             colorSamples=10,
             backgroundColor='rgba(0,0,0,0)',
             marginLeft=0,
             marginRight=0,
             marginTop=40,
             marginBottom=0,
             mantissa: int = 6):
        """
        Returns a Plotly Figure of the input matrix.

        Parameters
        ----------
        matrix : list or numpy.array
            The matrix to display.
        categories : list
            The list of categories to use on the X and Y axes.
        minValue : float , optional
            The desired minimum value to use for the color scale. If set to None, the minmum value found in the input matrix will be used. The default is None.
        maxValue : float , optional
            The desired maximum value to use for the color scale. If set to None, the maximum value found in the input matrix will be used. The default is None.
        title : str , optional
            The desired title to display. The default is "Confusion Matrix".
        xTitle : str , optional
            The desired X-axis title to display. The default is "Actual".
        yTitle : str , optional
            The desired Y-axis title to display. The default is "Predicted".
        width : int , optional
            The desired width of the figure. The default is 950.
        height : int , optional
            The desired height of the figure. The default is 500.
        showScale : bool , optional
            If set to True, a color scale is shown on the right side of the figure. The default is True.
        colorScale : str , optional
            The desired type of plotly color scales to use (e.g. "Viridis", "Plasma"). The default is "Viridis". For a full list of names, see https://plotly.com/python/builtin-colorscales/.
        colorSamples : int , optional
            The number of discrete color samples to use for displaying the data. The default is 10.
        backgroundColor : str , optional
            The desired background color. This can be any plotly color string and may be specified as:
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is 'rgba(0,0,0,0)' (transparent).
        marginLeft : int , optional
            The desired left margin in pixels. The default is 0.
        marginRight : int , optional
            The desired right margin in pixels. The default is 0.
        marginTop : int , optional
            The desired top margin in pixels. The default is 40.
        marginBottom : int , optional
            The desired bottom margin in pixels. The default is 0.
        mantissa : int , optional
            The desired number of digits of the mantissa. The default is 6.

        """
        #import plotly.figure_factory as ff
        import plotly.graph_objects as go
        import plotly.express as px

        try:
            import numpy as np
        except:
            print("Plotly.FigureByMatrix - Installing required numpy library.")
            try:
                os.system("pip install numpy")
            except:
                os.system("pip install numpy --user")
            try:
                import numpy as np
            except:
                warnings.warn("Plotly.FigureByMatrix - Error: Could not import numpy. Please install numpy manually. Returning None.")
                return None

        if not isinstance(matrix, list) and not isinstance(matrix, np.ndarray):
            print("Plotly.FigureByMatrix - Error: The input matrix is not of the correct type. Returning None.")
            return None

        annotations = []

        if isinstance(matrix, list):
            matrix = np.array(matrix)
        colors = px.colors.sample_colorscale(Plotly.ColorScale(colorScale), [n/(colorSamples -1) for n in range(colorSamples)])

        if not xCategories:
            xCategories = [x for x in range(len(matrix[0]))]
        if not yCategories:
            yCategories = [y for y in range(len(matrix))]
        
        if not maxValue or not minValue:
            max_values = []
            min_values = []
            for i in range(len(matrix)):
                row = matrix[i]
                max_values.append(max(row))
                min_values.append(min(row))
                for j, value in enumerate(row):
                    annotations.append(
                        {
                            "x": xCategories[j],
                            "y": yCategories[i],
                            "font": {"color": "black"},
                            "bgcolor": "white",
                            "opacity": 0.5,
                            "text": str(round(value, mantissa)), 
                            "xref": "x1",
                            "yref": "y1",
                            "showarrow": False
                        }
                    )
            if not minValue:
                minValueB = min(min_values)
            if not maxValue:
                maxValue = max(max_values)
        else:
            for i in range(len(matrix)):
                row = matrix[i]
                for j, value in enumerate(row):
                    annotations.append(
                        {
                            "x": xCategories[j],
                            "y": yCategories[i],
                            "font": {"color": "black"},
                            "bgcolor": "white",
                            "opacity": 0.5,
                            "text": str(round(value,mantissa)),
                            "xref": "x1",
                            "yref": "y1",
                            "showarrow": False
                        }
                    )
        new_matrix = []
        for i in range(len(matrix)):
            row = matrix[i]
            new_row = []
            maxRow = sum(row)
            for j in range(len(row)):
                if maxRow == 0:
                    new_row.append(round(0, mantissa))
                else:
                    new_row.append(round(float(row[j])/float(maxRow), mantissa))
            new_matrix.append(new_row)
        data = go.Heatmap(z=new_matrix, y=yCategories, x=xCategories, zmin=minValue, zmax=maxValue, showscale=showScale, colorscale=colors)
        
        layout = {
            "width": width,
            "height": height,
            "title": title,
            "xaxis": {"title": xTitle},
            "yaxis": {"title": yTitle, "autorange": "reversed"},
            "annotations": annotations,
            "paper_bgcolor": backgroundColor,
            "plot_bgcolor": backgroundColor,
            "margin":dict(l=marginLeft, r=marginRight, t=marginTop, b=marginBottom)
        }
        fig = go.Figure(data=data, layout=layout)
        fig.update_xaxes( tickvals=xCategories)
        fig.update_yaxes( tickvals=yCategories)
        return fig
    
    @staticmethod
    def FigureByCorrelation(actual,
                        predicted,
                        title="Correlation between Actual and Predicted Values",
                        xTitle = "Actual Values",
                        yTitle = "Predicted Values",
                        dotColor = "blue",
                        lineColor = "red",
                        width=800,
                        height=600,
                        theme='default',
                        backgroundColor='rgba(0,0,0,0)',
                        marginLeft=0,
                        marginRight=0,
                        marginTop=40,
                        marginBottom=0):
        """
        Returns a Plotly Figure showing the correlation between the input actual and predicted values. Actual values are displayed on the X-Axis, Predicted values are displayed on the Y-Axis.

        Parameters
        ----------
        actual : list
            The actual values to display.
        predicted : list
            The predicted values to display.
        title : str , optional
            The desired title to display. The default is "Correlation between Actual and Predicted Values".
        xTitle : str , optional
            The desired X-axis title to display. The default is "Actual Values".
        yTitle : str , optional
            The desired Y-axis title to display. The default is "Predicted Values".
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
            The desired width of the figure. The default is 800.
        height : int , optional
            The desired height of the figure. The default is 600.
        theme : str , optional
            The plotly color scheme to use. The options are "dark", "light", "default". The default is "default".
        backgroundColor : str , optional
            The desired background color. This can be any plotly color string and may be specified as:
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is 'rgba(0,0,0,0)' (transparent).
        marginLeft : int , optional
            The desired left margin in pixels. The default is 0.
        marginRight : int , optional
            The desired right margin in pixels. The default is 0.
        marginTop : int , optional
            The desired top margin in pixels. The default is 40.
        marginBottom : int , optional
            The desired bottom margin in pixels. The default is 0.
        
        Returns
        -------
        plotly.Figure
            The created plotly figure.
        
        """

        import numpy as np
        import plotly.graph_objs as go
        from sklearn.linear_model import LinearRegression
        import plotly.io as pio
        actual_values = np.array(actual)
        predicted_values = np.array(predicted)

        # Validate the theme input
        if theme == 'light':
            pio.templates.default = "plotly_white"
            backgroundColor='white'
        elif theme == 'dark':
            pio.templates.default = "plotly_dark"
            backgroundColor='black'
        else:
            pio.templates.default = None  # Use default Plotly theme
        
        # Calculate the best-fit line using linear regression
        regressor = LinearRegression()
        regressor.fit(np.array(actual_values).reshape(-1, 1), np.array(predicted_values))
        line = regressor.predict(np.array(actual_values).reshape(-1, 1))

        # Determine the range and tick step
        combined_values = np.concatenate([actual_values, predicted_values])
        min_value = np.min(combined_values)
        max_value = np.max(combined_values)
        margin = 0.1 * (max_value - min_value)
        tick_range = [min_value - margin, max_value + margin]
        tick_step = (max_value - min_value) / 10  # Adjust as needed for a different number of ticks

        # Create the scatter plot for actual vs predicted values
        scatter_trace = go.Scatter(
            x=actual_values,
            y=predicted_values,
            mode='markers',
            name='Actual vs. Predicted',
            marker=dict(color=dotColor)
        )

        # Create the line of best fit
        line_trace = go.Scatter(
            x=actual_values,
            y=line,
            mode='lines',
            name='Best Fit Line',
            line=dict(color=lineColor)
        )

        # Create the 45-degree line
        line_45_trace = go.Scatter(
            x=tick_range,
            y=tick_range,
            mode='lines',
            name='45-Degree Line',
        line=dict(color='green', dash='dash')
        )

        # Combine the traces into a single figure
        layout = {
            "title": title,
            "width": width,
            "height": height,
            "xaxis": {"title": xTitle, "range":tick_range, "dtick":tick_step},
            "yaxis": {"title": yTitle, "range":tick_range, "dtick":tick_step},
            "showlegend": True,
            "paper_bgcolor": backgroundColor,
            "plot_bgcolor": backgroundColor,
            "margin":dict(l=marginLeft, r=marginRight, t=marginTop, b=marginBottom)
        }

        fig = go.Figure(data=[scatter_trace, line_trace, line_45_trace], layout=layout)

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
            The desired width of the figure. The default is 950.
        height : int , optional
            The desired height of the figure. The default is 500.
        title : str , optional
            The chart title. The default is "Training and Testing Results".
        xTitle : str , optional
            The X-axis title. The default is "Epochs".
        xSpacing : float , optional
            The X-axis spacing. The default is 1.0.
        yTitle : str , optional
            The Y-axis title. The default is "Accuracy and Loss".
        ySpacing : float , optional
            The Y-axis spacing. The default is 0.1.
        useMarkers : bool , optional
            If set to True, markers will be displayed. The default is False.
        chartType : str , optional
            The desired type of chart. The options are "Line", "Bar", or "Scatter". It is case insensitive. The default is "Line".
        backgroundColor : str , optional
            The desired background color. This can be any plotly color string and may be specified as:
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
            The desired left margin in pixels. The default is 0.
        marginRight : int , optional
            The desired right margin in pixels. The default is 0.
        marginTop : int , optional
            The desired top margin in pixels. The default is 40.
        marginBottom : int , optional
            The desired bottom margin in pixels. The default is 0.

        Returns
        -------
        None.

        """
        import plotly.express as px
        
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
            "paper_bgcolor": backgroundColor,
            "plot_bgcolor": backgroundColor,
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
            If set to True the x axis is drawn. Otherwise it is not drawn. The default is False.
        yAxis : bool , optional
            If set to True the y axis is drawn. Otherwise it is not drawn. The default is False.
        zAxis : bool , optional
            If set to True the z axis is drawn. Otherwise it is not drawn. The default is False.
        axisSize : float , optional
            The size of the X, Y, Z, axes. The default is 1.
        backgroundColor : str , optional
            The desired color of the background. This can be any plotly color string and may be specified as:
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is "rgba(0,0,0,0)".
        marginLeft : int , optional
            The size in pixels of the left margin. The default value is 0.
        marginRight : int , optional
            The size in pixels of the right margin. The default value is 0.
        marginTop : int , optional
            The size in pixels of the top margin. The default value is 20.
        marginBottom : int , optional
            The size in pixels of the bottom margin. The default value is 0.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.
        
        Returns
        -------
        plotly.graph_objs._figure.Figure
            The created plotly figure.

        """
        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Wire import Wire
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
            paper_bgcolor=backgroundColor,
            plot_bgcolor=backgroundColor,
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
            If set to True the vertices will be drawn. Otherwise, they will not be drawn. The default is True.
        vertexSize : float , optional
            The desired size of the vertices. The default is 1.1.
        vertexColor : str , optional
            The desired color of the output vertices. This can be any plotly color string and may be specified as:
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is "black".
        vertexLabelKey : str , optional
            The dictionary key to use to display the vertex label. The default is None.
        vertexGroupKey : str , optional
            The dictionary key to use to display the vertex group. The default is None.
        vertexGroups : list , optional
            The list of vertex groups against which to index the color of the vertex. The default is [].
        vertexMinGroup : int or float , optional
            For numeric vertexGroups, vertexMinGroup is the desired minimum value for the scaling of colors. This should match the type of value associated with the vertexGroupKey. If set to None, it is set to the minimum value in vertexGroups. The default is None.
        edgeMaxGroup : int or float , optional
            For numeric vertexGroups, vertexMaxGroup is the desired maximum value for the scaling of colors. This should match the type of value associated with the vertexGroupKey. If set to None, it is set to the maximum value in vertexGroups. The default is None.
        showVertexLegend : bool, optional
            If set to True, the legend for the vertices of this topology is shown. Otherwise, it isn't. The default is False.
        vertexLegendLabel : str , optional
            The legend label string used to identify vertices. The default is "Topology Vertices".
        vertexLegendRank : int , optional
            The legend rank order of the vertices of this topology. The default is 1.
        vertexLegendGroup : int , optional
            The number of the vertex legend group to which the vertices of this topology belong. The default is 1.
        
        showEdges : bool , optional
            If set to True the edges will be drawn. Otherwise, they will not be drawn. The default is True.
        edgeWidth : float , optional
            The desired thickness of the output edges. The default is 1.
        edgeColor : str , optional
            The desired color of the output edges. This can be any plotly color string and may be specified as:
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is "black".
        edgeLabelKey : str , optional
            The dictionary key to use to display the edge label. The default is None.
        edgeGroupKey : str , optional
            The dictionary key to use to display the edge group. The default is None.
        edgeGroups : list , optional
            The list of edge groups against which to index the color of the edge. The default is [].
        edgeMinGroup : int or float , optional
            For numeric edgeGroups, edgeMinGroup is the desired minimum value for the scaling of colors. This should match the type of value associated with the edgeGroupKey. If set to None, it is set to the minimum value in edgeGroups. The default is None.
        edgeMaxGroup : int or float , optional
            For numeric edgeGroups, edgeMaxGroup is the desired maximum value for the scaling of colors. This should match the type of value associated with the edgeGroupKey. If set to None, it is set to the maximum value in edgeGroups. The default is None.
        showEdgeLegend : bool, optional
            If set to True, the legend for the edges of this topology is shown. Otherwise, it isn't. The default is False.
        edgeLegendLabel : str , optional
            The legend label string used to identify edges. The default is "Topology Edges".
        edgeLegendRank : int , optional
            The legend rank order of the edges of this topology. The default is 2.
        edgeLegendGroup : int , optional
            The number of the edge legend group to which the edges of this topology belong. The default is 2.
        
        showFaces : bool , optional
            If set to True the faces will be drawn. Otherwise, they will not be drawn. The default is True.
        faceOpacity : float , optional
            The desired opacity of the output faces (0=transparent, 1=opaque). The default is 0.5.
        faceColor : str , optional
            The desired color of the output faces. This can be any plotly color string and may be specified as:
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is "#FAFAFA".
        faceLabelKey : str , optional
            The dictionary key to use to display the face label. The default is None.
        faceGroupKey : str , optional
            The dictionary key to use to display the face group. The default is None.
        faceGroups : list , optional
            The list of face groups against which to index the color of the face. This can bhave numeric or string values. This should match the type of value associated with the faceGroupKey. The default is [].
        faceMinGroup : int or float , optional
            For numeric faceGroups, minGroup is the desired minimum value for the scaling of colors. This should match the type of value associated with the faceGroupKey. If set to None, it is set to the minimum value in faceGroups. The default is None.
        faceMaxGroup : int or float , optional
            For numeric faceGroups, maxGroup is the desired maximum value for the scaling of colors. This should match the type of value associated with the faceGroupKey. If set to None, it is set to the maximum value in faceGroups. The default is None.
        showFaceLegend : bool, optional
            If set to True, the legend for the faces of this topology is shown. Otherwise, it isn't. The default is False.
        faceLegendLabel : str , optional
            The legend label string used to idenitfy edges. The default is "Topology Faces".
        faceLegendRank : int , optional
            The legend rank order of the faces of this topology. The default is 3.
        faceLegendGroup : int , optional
            The number of the face legend group to which the faces of this topology belong. The default is 3.
        width : int , optional
            The width in pixels of the figure. The default value is 950.
        height : int , optional
            The height in pixels of the figure. The default value is 950.
        xAxis : bool , optional
            If set to True the x axis is drawn. Otherwise it is not drawn. The default is False.
        yAxis : bool , optional
            If set to True the y axis is drawn. Otherwise it is not drawn. The default is False.
        zAxis : bool , optional
            If set to True the z axis is drawn. Otherwise it is not drawn. The default is False.
        backgroundColor : str , optional
            The desired color of the background. This can be any plotly color string and may be specified as:
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is "rgba(0,0,0,0)".
        marginLeft : int , optional
            The size in pixels of the left margin. The default value is 0.
        marginRight : int , optional
            The size in pixels of the right margin. The default value is 0.
        marginTop : int , optional
            The size in pixels of the top margin. The default value is 20.
        marginBottom : int , optional
            The size in pixels of the bottom margin. The default value is 0.
        camera : list , optional
            The desired location of the camera). The default is [-1.25, -1.25, 1.25].
        center : list , optional
            The desired center (camera target). The default is [0, 0, 0].
        up : list , optional
            The desired up vector. The default is [0, 0, 1].
        renderer : str , optional
            The desired renderer. See Plotly.Renderers(). The default is "notebook".
        intensityKey : str , optional
            If not None, the dictionary of each vertex is searched for the value associated with the intensity key. This value is then used to color-code the vertex based on the colorScale. The default is None.
        showScale : bool , optional
            If set to True, the colorbar is shown. The default is False.
        cbValues : list , optional
            The input list of values to use for the colorbar. The default is [].
        cbTicks : int , optional
            The number of ticks to use on the colorbar. The default is 5.
        cbX : float , optional
            The x location of the colorbar. The default is -0.15.
        cbWidth : int , optional
            The width in pixels of the colorbar. The default is 15
        cbOutlineWidth : int , optional
            The width in pixels of the outline of the colorbar. The default is 0.
        cbTitle : str , optional
            The title of the colorbar. The default is "".
        cbSubTitle : str , optional
            The subtitle of the colorbar. The default is "".
        cbUnits: str , optional
            The units used in the colorbar. The default is ""
        colorScale : str , optional
            The desired type of plotly color scales to use (e.g. "viridis", "plasma"). The default is "viridis". For a full list of names, see https://plotly.com/python/builtin-colorscales/.
        mantissa : int , optional
            The desired length of the mantissa for the values listed on the colorbar. The default is 6.
        tolerance : float , optional
            The desired tolerance. The default is 0.0001.

        Returns
        -------
        Plotly figure

        """
        from topologicpy.Topology import Topology

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
                                     backgroundColor=backgroundColor,
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
            The width of the exported image in pixels. The default is 1920.
        height : int , optional
            The height of the exported image in pixels. The default is 1200.
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
            The width of the exported image in pixels. The default is 1920.
        height : int , optional
            The height of the exported image in pixels. The default is 1200.
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
            The width of the exported image in pixels. The default is 1920.
        height : int , optional
            The height of the exported image in pixels. The default is 1200.
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
    def SetCamera(figure, camera=[-1.25, -1.25, 1.25], center=[0, 0, 0], up=[0, 0, 1], projection="perspective"):
        """
        Sets the camera for the input figure.

        Parameters
        ----------
        figure : plotly.graph_objs._figure.Figure
            The input plotly figure.
        camera : list , optional
            The desired location of the camera. The default is [-1.25, -1.25, 1.25].
        center : list , optional
            The desired center (camera target). The default is [0, 0, 0].
        up : list , optional
            The desired up vector. The default is [0, 0, 1].
        projection : str , optional
            The desired type of projection. The options are "orthographic" or "perspective". It is case insensitive. The default is "perspective"
        
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
    def Show(figure, camera=[-1.25, -1.25, 1.25], center=[0, 0, 0], up=[0, 0, 1], renderer=None, projection="perspective"):
        """
        Shows the input figure.

        Parameters
        ----------
        figure : plotly.graph_objs._figure.Figure
            The input plotly figure.
        camera : list , optional
            The desired location of the camera. The default is [0, 0, 0].
        center : list , optional
            The desired center (camera target). The default is [0, 0, 0].
        up : list , optional
            The desired up vector. The default is [0, 0, 1].
        renderer : str , optional
            The desired renderer. See Plotly.Renderers(). If set to None, the code will attempt to discover the most suitable renderer. The default is None.
        projection : str, optional
            The desired type of projection. The options are "orthographic" or "perspective". It is case insensitive. The default is "perspective"

        
        Returns
        -------
        None
            
        """
        if figure == None:
            print("Plotly.Show - Error: The input is NULL. Returning None.")
            return None
        if not isinstance(figure, plotly.graph_objs._figure.Figure):
            print("Plotly.Show - Error: The input is not a figure. Returning None.")
            return None
        if renderer == None:
            renderer = Plotly.Renderer()
        if not renderer.lower() in Plotly.Renderers():
            print("Plotly.Show - Error: The input renderer is not in the approved list of renderers. Returning None.")
            return None
        if not camera == None and not center == None and not up == None:
            figure = Plotly.SetCamera(figure, camera=camera, center=center, up=up, projection=projection)
        figure.update_layout(
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
            The desired format. This can be any of "jpg", "jpeg", "pdf", "png", "svg", or "webp". It is case insensitive. The default is "png". 
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
