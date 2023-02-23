import topologicpy
import topologic
import plotly
import plotly.graph_objects as go
from topologicpy.Vertex import Vertex
from topologicpy.Edge import Edge
from topologicpy.Wire import Wire
from topologicpy.Face import Face
from topologicpy.Cell import Cell
from topologicpy.CellComplex import CellComplex
from topologicpy.Cluster import Cluster
from topologicpy.Topology import Topology

class Plotly:
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
    def DataByGraph(graph, vertexColor="white", vertexSize=6, vertexLabelKey=None, vertexGroupKey=None, vertexGroups=[], showVertices=True, edgeColor="black", edgeWidth=1, edgeLabelKey=None, edgeGroupKey=None, edgeGroups=[], showEdges=True):

        from topologicpy.Vertex import Vertex
        from topologicpy.Edge import Edge
        from topologicpy.Dictionary import Dictionary
        from topologicpy.Topology import Topology
        from topologicpy.Graph import Graph
        import plotly.graph_objs as go

        if not isinstance(graph, topologic.Graph):
            return None
        
        v_labels = []
        v_groupList = []
        data = []
        if showVertices:
            vertices = Graph.Vertices(graph)
            if vertexLabelKey or vertexGroupKey:
                for v in vertices:
                    d = Topology.Dictionary(v)
                    if d:
                        try:
                            v_label = str(Dictionary.ValueAtKey(d, key=vertexLabelKey)) or ""
                        except:
                            v_label = ""
                        try:
                            v_group = str(Dictionary.ValueAtKey(d, key=vertexGroupKey)) or ""
                        except:
                            v_group = ""
                        try:
                            v_groupList.append(vertexGroups.index(v_group))
                        except:
                            v_groupList.append(len(vertexGroups))
                        if not v_label == "" and not v_group == "":
                            v_label = v_label+" ("+v_group+")"
                    v_labels.append(v_label)
            if len(list(set(v_groupList))) < 2:
                v_groupList = vertexColor
            Xn=[Vertex.X(v) for v in vertices] # x-coordinates of nodes
            Yn=[Vertex.Y(v) for v in vertices] # y-coordinates of nodes
            Zn=[Vertex.Z(v) for v in vertices] # x-coordinates of nodes
            v_trace=go.Scatter3d(x=Xn,
                y=Yn,
                z=Zn,
                mode='markers',
                name='Graph Nodes',
                legendgroup=4,
                legendrank=4,
                marker=dict(symbol='circle',
                                size=vertexSize,
                                color=v_groupList,
                                colorscale='Viridis',
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
            for e in edges:
                sv = Edge.StartVertex(e)
                ev = Edge.EndVertex(e)
                if edgeLabelKey or edgeGroupKey:
                    for e in edges:
                        d = Topology.Dictionary(e)
                        if d:
                            e_label = str(Dictionary.ValueAtKey(d, key=edgeLabelKey)) or ""
                            e_group = str(Dictionary.ValueAtKey(d, key=edgeGroupKey)) or ""
                            try:
                                e_groupList.append(edgeGroups.index(e_group))
                            except:
                                e_groupList.append(len(edgeGroups))
                            if not e_label == "" and not e_group == "":
                                e_label = e_label+" ("+e_group+")"
                        e_labels.append(e_label)
                Xe+=[Vertex.X(sv),Vertex.X(ev), None] # x-coordinates of edge ends
                Ye+=[Vertex.Y(sv),Vertex.Y(ev), None] # y-coordinates of edge ends
                Ze+=[Vertex.Z(sv),Vertex.Z(ev), None] # z-coordinates of edge ends
            if len(list(set(e_groupList))) < 2:
                    e_groupList = edgeColor
            e_trace=go.Scatter3d(x=Xe,
                        y=Ye,
                        z=Ze,
                        mode='lines',
                        name='Graph Edges',
                        legendgroup=5,
                        legendrank=5,
                        line=dict(color=e_groupList, width=edgeWidth),
                        text=e_labels,
                        hoverinfo='text'
                        )
            data.append(e_trace)

        return data

    @staticmethod
    def DataByTopology(topology, vertexLabelKey=None, vertexGroupKey=None, edgeLabelKey=None, edgeGroupKey=None, faceLabelKey=None, faceGroupKey=None, vertexGroups=[], edgeGroups=[], faceGroups=[], faceColor="white", faceOpacity=0.5, edgeColor="black", edgeWidth=1, vertexColor="black", vertexSize=1.1, showFaces=True, showEdges=True, showVertices=True):
        """
        Creates plotly face, wire, and vertex data.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology. This must contain faces and or wires.
        vertexLabelKey : str , optional
            The dictionary key to use to display the vertex label. The default is None.
        vertexGroupKey : str , optional
            The dictionary key to use to display the vertex group. The default is None.
        edgeLabelKey : str , optional
            The dictionary key to use to display the edge label. The default is None.
        edgeGroupKey : str , optional
            The dictionary key to use to display the edge group. The default is None.
        faceLabelKey : str , optional
            The dictionary key to use to display the face label. The default is None.
        faceGroupKey : str , optional
            The dictionary key to use to display the face group. The default is None.
        faceColor : str , optional
            The desired color of the output faces. This can be any plotly color string and may be specified as:
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is "white".
        faceOpacity : float , optional
            The desired opacity of the output faces (0=transparent, 1=opaque). The default is 0.5.
        edgeColor : str , optional
            The desired color of the output wires (edges). This can be any plotly color string and may be specified as:
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is "black".
        edgeWidth : float , optional
            The desired thickness of the output wires (edges). The default is 1.
        vertexColor : str , optional
            The desired color of the output vertices. This can be any plotly color string and may be specified as:
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is "black".
        vertexSize : float , optional
            The desired size of the vertices. The default is 1.1.
        showFaces : bool , optional
            If set to True the faces will be drawn. Otherwise, they will not be drawn. The default is True.
        showEdges : bool , optional
            If set to True the wires (edges) will be drawn. Otherwise, they will not be drawn. The default is True.
        showVertices : bool , optional
            If set to True the vertices will be drawn. Otherwise, they will not be drawn. The default is True.

        Returns
        -------
        list
            The vertex, wire, and face data list.

        """
        from topologicpy.Topology import Topology
        from topologicpy.Dictionary import Dictionary

        def vertexData(topology, color="black", size=1.1, labelKey=None, groupKey=None, groups=[]):
            if isinstance(topology, topologic.Vertex):
                vertices = [topology]
            else:
                vertices = Topology.SubTopologies(topology, "vertex")
            x = []
            y = []
            z = []
            v_labels = []
            for v in vertices:
                x.append(v.X())
                y.append(v.Y())
                z.append(v.Z())
                d = Topology.Dictionary(v)
                if d:
                    v_label = Dictionary.ValueAtKey(d, key=labelKey) or ""
                    v_labels.append(v_label)
            return go.Scatter3d(x=x,
                                y=y,
                                z=z,
                                name='Topology Vertices',
                                showlegend=True,
                                marker=dict(color=color,  size=size),
                                mode='markers',
                                legendgroup=1,
                                legendrank=1,
                                text=v_labels)

        def edgeData(topology, color="black", width=1, labelKey=None, groupKey=None, groups=[]):
            x = []
            y = []
            z = []
            if isinstance(topology, topologic.Edge):
                edges = [topology]
            else:
                edges = Topology.SubTopologies(topology, "edge")
            for edge in edges:
                vertices = Edge.Vertices(edge)
                for v in vertices:
                    x.append(v.X())
                    y.append(v.Y())
                    z.append(v.Z())
                x.append(None)
                y.append(None)
                z.append(None)

            return go.Scatter3d(x=x,
                                y=y,
                                z=z,
                                name="Topology Edges",
                                showlegend=True,
                                marker_size=0,
                                mode="lines",
                                line=dict(color=color, width=width),
                                legendgroup=2,
                                legendrank=2,
                                hoverinfo='text')

        def faceData(topology, color="white", opacity=0.5, labelKey=None, groupKey=None, groups=[]):
            if isinstance(topology, topologic.Cluster):
                faces = Cluster.Faces(topology)
                if len(faces) > 0:
                    triangulated_faces = []
                    for face in faces:
                        triangulated_faces.append(Topology.Triangulate(face, 0.0001))
                    topology = Cluster.ByTopologies(triangulated_faces)
            else:
                topology = Topology.Triangulate(topology, 0.0001)
            tp_vertices = []
            _ = topology.Vertices(None, tp_vertices)
            x = []
            y = []
            z = []
            vertices = []
            intensities = []
            for tp_v in tp_vertices:
                vertices.append([tp_v.X(), tp_v.Y(), tp_v.Z()])
                x.append(tp_v.X())
                y.append(tp_v.Y())
                z.append(tp_v.Z())
                intensities.append(0)
            faces = []
            tp_faces = Topology.SubTopologies(topology, "face")
            for tp_f in tp_faces:
                f_vertices = Face.Vertices(tp_f)
                f = []
                for f_v in f_vertices:
                    f.append(vertices.index([f_v.X(), f_v.Y(), f_v.Z()]))
                faces.append(f)

            i = []
            j = []
            k = []
            for f in faces:
                i.append(f[0])
                j.append(f[1])
                k.append(f[2])

            return go.Mesh3d(
                    x=x,
                    y=y,
                    z=z,
                    # i, j and k give the vertices of triangles
                    # here we represent the 4 triangles of the tetrahedron surface
                    i=i,
                    j=j,
                    k=k,
                    name='Topology Faces',
                    showscale=False,
                    showlegend = True,
                    legendgroup=3,
                    legendrank=3,
                    color = color,
                    opacity = opacity,
                    flatshading = True,
                    lighting = {"facenormalsepsilon": 0},
                )
        from topologicpy.Wire import Wire
        from topologicpy.Cluster import Cluster
        from topologicpy.Topology import Topology
        if not isinstance(topology, topologic.Topology):
            return None
        data = []
        if showVertices:
            data.append(vertexData(topology, color=vertexColor, size=vertexSize, labelKey=vertexLabelKey, groupKey=vertexGroupKey, groups=vertexGroups))
        if showEdges and topology.Type() > topologic.Vertex.Type():
            data.append(edgeData(topology, color=edgeColor, width=edgeWidth, labelKey=edgeLabelKey, groupKey=edgeGroupKey, groups=edgeGroups))
        if showFaces and topology.Type() >= topologic.Face.Type():
            data.append(faceData(topology, color=faceColor, opacity=faceOpacity, labelKey=faceLabelKey, groupKey=faceGroupKey, groups=faceGroups))
        return data

    @staticmethod
    def FigureByConfusionMatrix(matrix,
             categories=[],
             minValue=None,
             maxValue=None,
             title="Confusion Matrix",
             xTitle = "Actual",
             yTitle = "Predicted",
             width=950,
             height=500,
             showscale = True,
             colorscale='Viridis',
             backgroundColor='rgba(0,0,0,0)',
             marginLeft=0,
             marginRight=0,
             marginTop=40,
             marginBottom=0):
        """
        Returns a Plotly Figure of the input matrix

        Parameters
        ----------
        matrix : list
            The matrix to display.
        categories : list
            The list of categories to use on the X and Y axes.

        """
        #import plotly.figure_factory as ff
        import plotly.graph_objects as go

        annotations = []
       
        if not minValue:
            minValue = 0
        if not maxValue:
            max_values = []
            for i, row in enumerate(matrix):
                max_values.append(max(row))
                for j, value in enumerate(row):
                    annotations.append(
                        {
                            "x": categories[i],
                            "y": categories[j],
                            "font": {"color": "white"},
                            "text": str(value),
                            "xref": "x1",
                            "yref": "y1",
                            "showarrow": False
                        }
                    )
            maxValue = max(max_values)
        else:
            for i, row in enumerate(matrix):
                for j, value in enumerate(row):
                    annotations.append(
                        {
                            "x": categories[i],
                            "y": categories[j],
                            "font": {"color": "white"},
                            "text": str(value),
                            "xref": "x1",
                            "yref": "y1",
                            "showarrow": False
                        }
                    )
        data = go.Heatmap(z=matrix, y=categories, x=categories, zmin=minValue, zmax=maxValue, showscale=showscale, colorscale=colorscale)
        
        layout = {
            "width": width,
            "height": height,
            "title": title,
            "xaxis": {"title": xTitle},
            "yaxis": {"title": yTitle},
            "annotations": annotations,
            "paper_bgcolor": backgroundColor,
            "plot_bgcolor": backgroundColor,
            "margin":dict(l=marginLeft, r=marginRight, t=marginTop, b=marginBottom),
        }
        fig = go.Figure(data=data, layout=layout)
        return fig
        
    @staticmethod
    def FigureByDataFrame(df,
             labels=[],
             title="Untitled",
             x_title="X Axis",
             x_spacing=1.0,
             y_title="Y Axis",
             y_spacing=1,
             use_markers=False,
             chart_type="Line"):
        """
        Returns a Plotly Figure of the input dataframe

        Parameters
        ----------
        df : pandas.df
            The p[andas dataframe to display.
        data_labels : list
            The labels to use for the data.
        title : str , optional
            The chart title. The default is "Untitled".
        x_title : str , optional
            The X-axis title. The default is "Epochs".
        x_spacing : float , optional
            The X-axis spacing. The default is 1.0.
        y_title : str , optional
            The Y-axis title. The default is "Accuracy and Loss".
        y_spacing : float , optional
            THe Y-axis spacing. The default is 0.1.
        use_markers : bool , optional
            If set to True, markers will be displayed. The default is False.
        chart_type : str , optional
            The desired type of chart. The options are "Line", "Bar", or "Scatter". It is case insensitive. The default is "Line".
        renderer : str , optional
            The desired plotly renderer. The default is "notebook".

        Returns
        -------
        None.

        """
        import plotly.express as px
        if chart_type.lower() == "line":
            fig = px.line(df, x=labels[0], y=labels[1:], title=title, markers=use_markers)
        elif chart_type.lower() == "bar":
            fig = px.bar(df, x=labels[0], y=labels[1:], title=title)
        elif chart_type.lower() == "scatter":
            fig = px.scatter(df, x=labels[0], y=labels[1:], title=title)
        else:
            raise NotImplementedError
        fig.layout.xaxis.title=x_title
        fig.layout.xaxis.dtick=x_spacing
        fig.layout.yaxis.title=y_title
        fig.layout.yaxis.dtick= y_spacing
        return fig

    @staticmethod
    def FigureByData(data, color=None, width=950, height=500, xAxis=False, yAxis=False, zAxis=False, backgroundColor='rgba(0,0,0,0)', marginLeft=0, marginRight=0, marginTop=20, marginBottom=0):
        """
        Creates plotly figure.

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

        v0 = Vertex.ByCoordinates(0,0,0)
        v1 = Vertex.ByCoordinates(1,0,0)
        v2 = Vertex.ByCoordinates(0,1,0)
        v3 = Vertex.ByCoordinates(0,0,1)

        if xAxis:
            xEdge = Edge.ByVertices([v0,v1])
            xWire = Wire.ByEdges([xEdge])
            xData = Plotly.DataByTopology(xWire, wireColor="red", wireWidth=6, drawFaces=False, drawWires=True, drawVertices=False)
            data = data + xData
        if yAxis:
            yEdge = Edge.ByVertices([v0,v2])
            yWire = Wire.ByEdges([yEdge])
            yData = Plotly.DataByTopology(yWire, wireColor="green", wireWidth=6, drawFaces=False, drawWires=True, drawVertices=False)
            data = data + yData
        if zAxis:
            zEdge = Edge.ByVertices([v0,v3])
            zWire = Wire.ByEdges([zEdge])
            zData = Plotly.DataByTopology(zWire, wireColor="blue", wireWidth=6, drawFaces=False, drawWires=True, drawVertices=False)
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
        return figure

    @staticmethod
    def PieChartByData(data, values, names, renderer="notebook"):
        import pandas as pd
        import plotly.express as px
        dlist = list(map(list, zip(*data)))
        df = pd.DataFrame(dlist, columns=data['names'])
        fig = px.pie(df, values=values, names=names)
        fig.show(renderer=renderer)
    
    @staticmethod
    def SetCamera(figure, camera=[1.25, 1.25, 1.25], target=[0, 0, 0], up=[0, 0, 1]):
        """
        Sets the camera for the input figure.

        Parameters
        ----------
        figure : plotly.graph_objs._figure.Figure
            The input plotly figure.
        camera : list , optional
            The desired location of the camera). The default is [0,0,0].
        center : list , optional
            The desired center (camera target). The default is [0,0,0].
        up : list , optional
            The desired up vector. The default is [0,0,1].
        
        Returns
        -------
        plotly.graph_objs._figure.Figure
            The updated figure

        """
        if not isinstance(camera, list):
            return None
        if not isinstance(target, list):
            return None
        if not isinstance(up, list):
            return None
        scene_camera = dict(
        up=dict(x=up[0], y=up[1], z=up[2]),
        eye=dict(x=camera[0], y=camera[1], z=camera[2]),
        center=dict(x=target[0], y=target[1], z=target[2])
        )
        figure.update_layout(scene_camera=scene_camera)
        return figure

    @staticmethod
    def Show(figure, renderer="notebook", camera=[1.25, 1.25, 1.25], target=[0, 0, 0], up=[0, 0, 1]):
        """
        Shows the input figure.

        Parameters
        ----------
        figure : plotly.graph_objs._figure.Figure
            The input plotly figure.
        renderer : str , optional
            The desired rendered. See Plotly.Renderers(). The default is "notebook".
        
        Returns
        -------
        None
            
        """
        if not isinstance(figure, plotly.graph_objs._figure.Figure):
            return None
        if not renderer.lower() in Plotly.Renderers():
            return None
        figure = Plotly.SetCamera(figure, camera=camera, target=target, up=up)
        figure.show(renderer=renderer)
        return None

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
         'iframe_connected', 'sphinx_gallery', 'sphinx_gallery_png']

    @staticmethod
    def ExportToImage(figure, filePath, format="png", width="1920", height="1080"):
        """
        Exports the plotly figure to an image.

        Parameters
        ----------
        figure : plotly.graph_objs._figure.Figure
            The input plotly figure.
        filePath : str
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
        if not isinstance(filePath, str):
            return None
        if not format.lower() in ["jpg", "jpeg", "pdf", "png", "svg", "webp"]:
            return None
        returnStatus = False
        try:
            plotly.io.write_image(figure, filePath, format=format.lower(), scale=None, width=width, height=height, validate=True, engine='auto')
            returnStatus = True
        except:
            returnStatus = False
        return returnStatus

