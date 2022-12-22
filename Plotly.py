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
    def DataByTopology(topology, faceColor="lightblue", faceOpacity=0.5, wireColor="black", wireWidth=1, vertexColor="black", vertexSize=1.1, drawFaces=True, drawWires=True, drawVertices=True):
        """
        Creates plotly face, wire, and vertex data.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology. This must contain faces and or wires.
        faceColor : str , optional
            The desired color of the output faces. This can be any plotly color string. The default is "lightblue".
        faceOpacity : float , optional
            The desired opacity of the output faces (0=transparent, 1=opaque). The default is 0.5.
        wireColor : str , optional
            The desired color of the output wires (edges). This can be any plotly color string. The default is "black".
        wireWidth : float , optional
            The desired thickness of the output wires (edges). The default is 1.
        vertexColor : str , optional
            The desired color of the output vertices. This can be any plotly color string. The default is "black".
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
                wires = [Wire.ByEdges(topology)]
            elif isinstance(topology, topologic.Wire):
                wires = [topology]
            else:
                wires = Topology.SubTopologies(topology, "wire")
            for w in wires:
                edges = Wire.Edges(w)
                for edge in edges:
                    vertices = Edge.Vertices(edge)
                    for v in vertices:
                        x.append(v.X())
                        y.append(v.Y())
                        z.append(v.Z())
                    #x.append(vertices[0].X())
                    #y.append(vertices[0].Y())
                    #z.append(vertices[0].Z())
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
                cells = Cluster.Cells(topology)
                triangulated_cells = []
                for cell in cells:
                    triangulated_cells.append(Topology.Triangulate(cell, 0.0001))
                topology = Cluster.ByTopologies(triangulated_cells)
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
    def FigureByData(data, width=950, height=500, xAxis=False, yAxis=False, zAxis=False, paperBackgroundColor='lightgrey', plotBackgroundColor='lightgrey', marginLeft=2, marginRight=2, marginTop=2, marginBottom=2):
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
        paperBackgroundColor : str , optional
            The desired color of the paper background. This can be any plotly color string. The default is "lightgrey".
        paperBackgroundColor : str , optional
            The desired color of the plot background. This can be any plotly color string. The default is "lightgrey".
        marginLeft : int , optional
            The size in pixels of the left margin. The default value is 2.
        marginRight : int , optional
            The size in pixels of the right margin. The default value is 2.
        marginTop : int , optional
            The size in pixels of the top margin. The default value is 2.
        marginBottom : int , optional
            The size in pixels of the bottom margin. The default value is 2.
        
        Returns
        -------
        plotly.graph_objs._figure.Figure
            The created plotly figure.

        """
        if not isinstance(data, list):
            return None
        figure = go.Figure(data=data)
        figure.update_layout(
            width=width,
            height=height,
            scene = dict(
                xaxis = dict(visible=xAxis),
                yaxis = dict(visible=yAxis),
                zaxis =dict(visible=zAxis),
                ),
            paper_bgcolor=paperBackgroundColor,
            plot_bgcolor=plotBackgroundColor,
            margin=dict(l=marginLeft, r=marginRight, t=marginTop, b=marginBottom),
            )
        return figure

    @staticmethod
    def Show(figure, renderer="browser"):
        if not isinstance(figure, plotly.graph_objs._figure.Figure):
            return None
        if not renderer.lower() in ['plotly_mimetype', 'jupyterlab', 'nteract', 'vscode',
         'notebook', 'notebook_connected', 'kaggle', 'azure', 'colab',
         'cocalc', 'databricks', 'json', 'png', 'jpeg', 'jpg', 'svg',
         'pdf', 'browser', 'firefox', 'chrome', 'chromium', 'iframe',
         'iframe_connected', 'sphinx_gallery', 'sphinx_gallery_png']:
            return None
        figure.show(renderer=renderer)
        return None

    @staticmethod
    def Renderers():
        return ['plotly_mimetype', 'jupyterlab', 'nteract', 'vscode',
         'notebook', 'notebook_connected', 'kaggle', 'azure', 'colab',
         'cocalc', 'databricks', 'json', 'png', 'jpeg', 'jpg', 'svg',
         'pdf', 'browser', 'firefox', 'chrome', 'chromium', 'iframe',
         'iframe_connected', 'sphinx_gallery', 'sphinx_gallery_png']