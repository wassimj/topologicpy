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
    def DataByTopology(topology, faceColor="white", faceOpacity=0.5, wireColor="black", wireWidth=1, vertexColor="black", vertexSize=1.1, drawFaces=True, drawWires=True, drawVertices=True):
        """
        Creates plotly face, wire, and vertex data.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology. This must contain faces and or wires.
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
        wireColor : str , optional
            The desired color of the output wires (edges). This can be any plotly color string and may be specified as:
            - A hex string (e.g. '#ff0000')
            - An rgb/rgba string (e.g. 'rgb(255,0,0)')
            - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color.
            The default is "black".
        wireWidth : float , optional
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
        drawFaces : bool , optional
            If set to True the faces will be drawn. Otherwise, they will not be drawn. The default is True.
        drawWires : bool , optional
            If set to True the wires (edges) will be drawn. Otherwise, they will not be drawn. The default is True.
        drawVertices : bool , optional
            If set to True the vertices will be drawn. Otherwise, they will not be drawn. The default is True.

        Returns
        -------
        list
            The vertex, wire, and face data list.

        """
        def vertexData(topology, color="black", size=1.1):
            if isinstance(topology, topologic.Vertex):
                vertices = [topology]
            else:
                vertices = Topology.SubTopologies(topology, "vertex")
            x = []
            y = []
            z = []
            for v in vertices:
                x.append(v.X())
                y.append(v.Y())
                z.append(v.Z())

            return go.Scatter3d(x=x, y=y, z=z, showlegend=False, marker=dict(
                            color=color,  
                            size=size),
                            mode='markers')

        def wireData(topology, color="black", width=1):
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

            return go.Scatter3d(x=x, y=y, z=z, showlegend=False, marker_size=0, mode="lines",
                                     line=dict(
                                         color=color,
                                         width=width
                                     ))

        def faceData(topology, color="lightblue", opacity=0.5):
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
                    name='y',
                    showscale=False,
                    showlegend = False,
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
        if drawVertices:
            data.append(vertexData(topology, color=vertexColor, size=vertexSize))
        if drawWires and topology.Type() > topologic.Vertex.Type():
            data.append(wireData(topology, color=wireColor, width=wireWidth))
        if drawFaces and topology.Type() >= topologic.Face.Type():
            data.append(faceData(topology, color=faceColor, opacity=faceOpacity))
        return data

    @staticmethod
    def FigureByData(data, width=950, height=500, xAxis=False, yAxis=False, zAxis=False, backgroundColor='rgba(0,0,0,0)', marginLeft=0, marginRight=0, marginTop=0, marginBottom=0):
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
            The size in pixels of the top margin. The default value is 0.
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
    def Show(figure, renderer="browser"):
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

