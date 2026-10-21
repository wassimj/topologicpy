"""Production graph features must work with the retired modules unavailable."""
import ast
import importlib
import sys
from pathlib import Path

import pytest


@pytest.fixture
def no_legacy_graph(monkeypatch):
    for name in ('topologicpy.Graph', 'topologicpy.pythonocc_backend.graph'):
        monkeypatch.setitem(sys.modules, name, None)


def test_source_has_no_legacy_graph_consumers():
    root = Path(__file__).resolve().parents[1] / 'src/topologicpy'
    retired = {root / 'Graph.py', root / 'pythonocc_backend/graph.py'}
    for path in root.rglob('*.py'):
        if path in retired:
            continue
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        external_graph = any(isinstance(n, ast.ImportFrom) and n.module == 'rdflib'
                             and any(a.name == 'Graph' and not a.asname for a in n.names)
                             for n in ast.walk(tree))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module not in ('Graph', 'topologicpy.Graph'), str(path)
                assert not (node.level and node.module == 'graph'), str(path)
            if isinstance(node, ast.Call):
                assert external_graph or not (isinstance(node.func, ast.Name) and node.func.id == 'Graph'), str(path)
                if isinstance(node.func, ast.Attribute):
                    assert not (isinstance(node.func.value, ast.Name) and node.func.value.id == 'Graph'), str(path)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'Core':
                assert node.attr not in ('Graph','GraphUtility'), str(path)


def test_backend_does_not_register_native_graph(no_legacy_graph):
    from topologicpy.Core import Core
    backend = Core.Backend()
    assert not hasattr(backend, 'Graph')
    assert not hasattr(backend, 'GraphUtility')


def test_navigation_always_constructs_tgraph(no_legacy_graph):
    from topologicpy.TGraph import TGraph
    from topologicpy.Navigation import Navigation
    from topologicpy.Face import Face
    from topologicpy.Vertex import Vertex
    from topologicpy.Topology import Topology
    face = Face.ByVertices([Vertex.ByCoordinates(x,y,0) for x,y in [(0,0),(2,0),(2,2),(0,2)]])
    model = Navigation.ByFaces([face])
    graph = Navigation.Graph(model)
    assert isinstance(graph, TGraph)
    assert TGraph.Order(graph) == 4
    assert TGraph.Size(graph) == 4
    assert TGraph.ShortestPath(graph, 0, 3)
    assert len(Navigation.Tiles(model, graph)) == 4
    assert Topology.Geometry(graph)['vertices'] == TGraph.MeshData(graph)['vertices']


def test_plotly_public_graph_entry_uses_tgraph(no_legacy_graph):
    from topologicpy.TGraph import TGraph
    from topologicpy.Plotly import Plotly
    graph = TGraph.ByMeshData([[0,0,0],[1,0,0]], [[0,1]], ontology=False)
    assert Plotly.DataByGraph(graph, showVertexLabel=True)


class RowsManager:
    def ensure_schema(self):
        pass

    def exec(self, query, parameters, **kwargs):
        if 'MATCH (g:Graph)' in query:
            return [{'id':'g', 'label':'Example', 'props':'{"custom":42}'}]
        if 'MATCH (v:Vertex)' in query:
            return [{'id':name, 'label':name, 'x':0, 'y':0, 'z':0, 'props':'{}'} for name in ('a','b')]
        if 'MATCH (a:Vertex)' in query:
            return [{'a_id':'a','b_id':'b','label':'connect','props':'{"cost":3}'}]
        return []


@pytest.mark.parametrize('module', ['Kuzu','LadybugDB'])
@pytest.mark.parametrize('legacy_selector', [None,False,True])
def test_database_retrieval_keeps_distinct_coincident_nodes(no_legacy_graph, module, legacy_selector):
    from topologicpy.TGraph import TGraph
    provider = getattr(importlib.import_module('topologicpy.'+module), module)
    result = provider.GraphByID(RowsManager(), 'g', ontology=False,
                                **({'asTGraph':legacy_selector} if legacy_selector is not None else {}))
    assert isinstance(result, TGraph)
    assert TGraph.Order(result) == 2
    assert TGraph.Size(result) == 1
    assert [v['dictionary']['id'] for v in TGraph.Vertices(result)] == ['a','b']
    assert TGraph.Edges(result)[0]['dictionary']['cost'] == 3
    assert TGraph.Dictionary(result)['custom'] == 42


def test_gql_materialization_uses_tgraph_even_with_legacy_selector(no_legacy_graph):
    from topologicpy.TGraph import TGraph
    from topologicpy.gql.Executor import Executor
    graph = Executor._graph_by_vertices_edges([{'id':'a'},{'id':'b'}],
            [{'src':0,'dst':1}], ontology=False, asTGraph=False)
    assert isinstance(graph, TGraph)
    assert TGraph.Order(graph) == 2
    assert TGraph.Size(graph) == 1


@pytest.mark.pythonocc_only
@pytest.mark.parametrize('reverse', [False,True])
def test_plotting_keeps_stored_curves_and_incidence_direction(no_legacy_graph,reverse):
    from topologicpy.TGraph import TGraph
    from topologicpy.Plotly import Plotly
    from topologicpy.Edge import Edge
    from topologicpy.Vertex import Vertex
    arc = Edge.Arc(radius=1, fromAngle=0, toAngle=180, silent=True)
    assert arc is not None
    a,b = Edge.StartVertex(arc),Edge.EndVertex(arc)
    graph = TGraph.ByVerticesEdges([a,b],[],ontology=False)
    graph.AddEdge(1 if reverse else 0,0 if reverse else 1,representation=arc)
    traces = Plotly.DataByGraph(graph,showVertices=False)
    trace = next(t for t in traces if t.type=='scatter3d' and t.mode=='lines')
    coords = [(x,y,z) for x,y,z in zip(trace.x,trace.y,trace.z) if x is not None]
    assert len(coords)>3
    assert max(p[1] for p in coords)>.9
    assert coords[0] == pytest.approx(TGraph.Coordinates(graph,1 if reverse else 0))
    assert coords[-1] == pytest.approx(TGraph.Coordinates(graph,0 if reverse else 1))


def test_non_geometric_representation_remains_renderable(no_legacy_graph,monkeypatch):
    from topologicpy.TGraph import TGraph
    from topologicpy.Plotly import Plotly
    graph = TGraph.ByMeshData([[0,0,0],[1,0,0]],[],ontology=False)
    graph.AddEdge(0,1,representation=object())
    def unavailable(*args,**kwargs):
        raise ImportError('Geometry kernel unavailable')
    monkeypatch.setattr(Plotly,'_edge_render_points',staticmethod(unavailable))
    assert Plotly.DataByTGraph(graph)


def test_notebooks_have_no_legacy_graph_consumers():
    import json
    root = Path(__file__).resolve().parents[1] / "notebooks"
    for path in root.rglob("*.ipynb"):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            assert "topologicpy.Graph" not in source, str(path)
            # Ignore IPython-only commands when parsing notebook Python.
            source = "\n".join("" if line.lstrip().startswith(("%", "!")) else line
                               for line in source.splitlines())
            try:
                tree = ast.parse(source)
            except SyntaxError:
                # Some unrelated historical notebooks contain incomplete scratch cells.
                assert "Graph." not in source, str(path)
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    assert not (node.module == "topologicpy" and
                                any(alias.name == "Graph" for alias in node.names)), str(path)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    assert not (isinstance(node.func.value, ast.Name) and
                                node.func.value.id == "Graph"), str(path)


@pytest.mark.parametrize("native_geometry", [False, True])
def test_capacity_notebook_routes_with_tgraph_records(no_legacy_graph, native_geometry):
    import json
    from topologicpy.TGraph import TGraph
    from topologicpy.Vertex import Vertex
    from topologicpy.Edge import Edge
    from topologicpy.Wire import Wire
    from topologicpy.Topology import Topology
    from topologicpy.Dictionary import Dictionary
    import random
    import math
    path = Path(__file__).resolve().parents[1] / "notebooks/Shortest_Paths_With_Capacities.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = dict(TGraph=TGraph, Vertex=Vertex, Edge=Edge, Wire=Wire,
                     Topology=Topology, Dictionary=Dictionary, random=random.Random(0), math=math)
    helper = next("".join(c["source"]) for c in notebook["cells"]
                  if c["cell_type"] == "code" and "def _path_wire(" in "".join(c["source"]))
    exec(compile(helper, str(path), "exec"), namespace)
    vertices = [Vertex.ByCoordinates(x, 0, 0) for x in range(3)]
    edges = [Edge.ByVertices(vertices[i:i+2]) for i in range(2)]
    graph = (TGraph.ByVerticesEdges(vertices, edges, ontology=False) if native_geometry else
             TGraph.ByMeshData([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[0, 1], [1, 2]], ontology=False))
    namespace.update(g=graph, vertices=TGraph.Vertices(graph), edges=namespace["_edge_records"](graph),
                     sv=Vertex.ByCoordinates(0, 0, 0), ev=Vertex.ByCoordinates(2, 0, 0))
    source = next("".join(c["source"]) for c in notebook["cells"]
                  if c["cell_type"] == "code" and "def edge_filter(" in "".join(c["source"]))
    route = source[source.index("# Centers"):source.index("for i, s_p in enumerate(sp_list):")]
    exec(compile(route, str(path), "exec"), namespace)
    assert 2 <= len(namespace["sp_list"]) <= 5
    assert all(Topology.IsInstance(wire, "Wire") for wire in namespace["sp_list"])
    assert any(e["dictionary"]["capacity"] == 0 for e in TGraph.Edges(graph))
