"""Native persistence contracts, including stable slots and geometry identity."""
import json, math, tempfile, unittest, zipfile
from datetime import datetime, timezone
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from uuid import UUID
from topologicpy.TGraph import TGraph
from topologicpy.Topology import Topology
from topologicpy.Cell import Cell
from topologicpy.CellComplex import CellComplex
from topologicpy.Edge import Edge
from topologicpy.Vertex import Vertex
from topologicpy.Wire import Wire
from topologicpy.io.tpy import TPYCodec

class GraphArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'model.tpy'

    def roundtrip(self,value):
        self.assertTrue(Topology.Save(value,self.path,overwrite=True,silent=False))
        result=Topology.Load(self.path,silent=False)
        self.assertIsNotNone(result)
        return result

    def graph(self):
        graph=TGraph(directed=True,allowSelfLoops=True,allowParallelEdges=True,dictionary={'project':'test'})
        for i in range(4):graph.AddVertex(dictionary={'x':i,'y':0,'z':0,'name':str(i)})
        graph.AddEdge(0,1,dictionary={'weight':1})
        graph.AddEdge(0,1,dictionary={'weight':2})
        graph.AddEdge(1,1,directed=False)
        graph.AddEdge(2,3)
        graph.RemoveEdge(0);graph.RemoveVertex(3)
        return graph

    def test_stable_inactive_slots_parallel_edges_loops_and_flags(self):
        graph=self.graph();restored=self.roundtrip(graph)
        self.assertEqual(restored._vertices,graph._vertices)
        self.assertEqual(restored._edges,graph._edges)
        self.assertEqual(restored._dictionary,graph._dictionary)
        self.assertEqual(restored._out_edges,graph._out_edges)
        self.assertEqual(restored._in_edges,graph._in_edges)
        self.assertEqual(restored._incident_edges,graph._incident_edges)
        self.assertEqual(restored._edge_lookup,graph._edge_lookup)
        self.assertEqual(restored.AddVertex(),4)
        self.assertEqual(restored.AddEdge(0,1),4)

    def test_empty_and_undirected_graphs(self):
        self.assertEqual(TGraph.Order(self.roundtrip(TGraph())),0)
        graph=TGraph.ByEdgeIndexPairs(order=3,edgeIndexPairs=[[0,1],[1,2]],ontology=False)
        restored=self.roundtrip(graph)
        self.assertFalse(restored._directed)
        self.assertEqual(TGraph.ShortestPath(restored,0,2),TGraph.ShortestPath(graph,0,2))

    def test_native_curves_and_shared_originator_subshape(self):
        model=Cell.Prism(width=2,length=3,height=4)
        face=Topology.Faces(model)[0]
        curve=Wire.Circle(radius=3,polyline=False)
        graph=TGraph()
        graph.AddVertex(dictionary={'name':'a'},representation=curve,originator=face)
        graph.AddVertex(dictionary={'name':'b'},representation=curve,originator=face)
        restored=self.roundtrip({'model':model,'graph':graph,'curve':curve,'face':face})
        h=restored['graph'];a,b=h._vertices
        self.assertIs(a['representation'],b['representation'])
        self.assertIs(a['representation'],restored['curve'])
        self.assertIs(a['originator'],b['originator'])
        self.assertIs(a['originator'],restored['face'])
        self.assertTrue(any(Topology.IsSame(a['originator'],f) for f in Topology.Faces(restored['model'])))
        self.assertAlmostEqual(Wire.Length(restored['curve'],mantissa=12),6*math.pi,places=9)
        self.assertEqual(Cell.Volume(restored['model']),24)

    def test_graph_only_archive_preserves_shared_faces_between_originators(self):
        building=CellComplex.Prism(width=8,length=4,height=3,uSides=2,vSides=1,wSides=1)
        cells=Topology.Cells(building)
        self.assertEqual(len(Topology.SharedFaces(cells[0],cells[1])),1)
        graph=TGraph()
        for cell in cells:graph.AddVertex(originator=cell)
        restored=self.roundtrip(graph)
        origins=[vertex['originator'] for vertex in restored._vertices]
        self.assertEqual(len(Topology.SharedFaces(origins[0],origins[1])),1)

    def test_orientations_and_coincident_independent_vertices(self):
        edge=Edge.ByVertices([Vertex.ByCoordinates(0,0,0),Vertex.ByCoordinates(3,0,0)])
        reverse=Topology.ByOCCTShape(Topology.OCCTShape(edge).Reversed())
        graph=TGraph()
        graph.AddVertex(representation=edge);graph.AddVertex(representation=reverse)
        graph.AddVertex(representation=Vertex.ByCoordinates(1,2,3))
        graph.AddVertex(representation=Vertex.ByCoordinates(1,2,3))
        restored=self.roundtrip(graph)
        representations=[record['representation'] for record in restored._vertices]
        self.assertEqual(Vertex.Coordinates(Edge.StartVertex(representations[0])),[0,0,0])
        self.assertEqual(Vertex.Coordinates(Edge.StartVertex(representations[1])),[3,0,0])
        self.assertFalse(Topology.IsSame(representations[2],representations[3]))

    def test_mixed_collections_and_graph_reference_cycles(self):
        a,b=TGraph(),TGraph()
        a._dictionary['other']=b;b._dictionary['other']=a
        restored=self.roundtrip((a,{'alias':a,'second':b},Vertex.ByCoordinates(1,2,3)))
        self.assertIsInstance(restored,tuple)
        self.assertIs(restored[0],restored[1]['alias'])
        self.assertIs(restored[0]._dictionary['other'],restored[1]['second'])
        self.assertIs(restored[1]['second']._dictionary['other'],restored[0])

    def test_rich_metadata_and_reserved_tag_literals(self):
        import numpy as np
        graph=TGraph(dictionary={'bytes':b'\x00\xff','tuple':(1,True),'set':{1,2},'frozen':frozenset({3}),
            'decimal':Decimal('1.234500'),'fraction':Fraction(2,3),'uuid':UUID(int=123),
            'time':datetime(2026,11,4,12,30,tzinfo=timezone.utc),'nan':float('nan'),'inf':float('inf'),
            'complex':complex(2,3),'array':np.arange(6,dtype=np.int16).reshape(2,3),
            'scalar':np.float64(1.25),'literal':{'__tpy_type__':'reference','object':'not-an-object'},
            'keys':{(1,2):'tuple-key',3:'int-key'}})
        result=self.roundtrip(graph)._dictionary
        for key in ('bytes','tuple','set','frozen','decimal','fraction','uuid','time','complex','literal','keys'):
            self.assertEqual(result[key],graph._dictionary[key],key)
        self.assertTrue(math.isnan(result['nan']));self.assertEqual(result['inf'],float('inf'))
        np.testing.assert_array_equal(result['array'],graph._dictionary['array'])
        self.assertEqual(result['array'].dtype,np.dtype('int16'))
        self.assertIs(type(result['scalar']),np.float64)

    def test_compiled_and_analysis_caches_are_rebuilt(self):
        graph=TGraph.ByEdgeIndexPairs(order=3,edgeIndexPairs=[[0,1],[1,2]],ontology=False)
        graph._compiled=object();graph._analysis_context=object()
        restored=self.roundtrip(graph)
        self.assertIsNone(restored._compiled);self.assertIsNone(restored._analysis_context)
        self.assertEqual(TGraph.Degree(restored,1),2)

    def test_public_graph_convenience_methods(self):
        graph=self.graph()
        self.assertTrue(TGraph.Save(graph,self.path))
        self.assertIsInstance(TGraph.Load(self.path),TGraph)
        self.assertIsInstance(TGraph.ByTPYPath(self.path),TGraph)
        self.assertTrue(TGraph.ExportToTPY(graph,self.path,overwrite=True))
        self.assertTrue(Topology.ExportToTPY(graph,self.path,overwrite=True))
        self.assertFalse(Topology.Save(graph,Path(self.temp.name)/'graph.step',silent=True))

    def test_native_topology_and_version_one_compatibility(self):
        model=Cell.Prism(width=2,length=3,height=4)
        restored=self.roundtrip(model);self.assertEqual(Cell.Volume(restored),24)
        with zipfile.ZipFile(self.path) as archive:
            contents={name:archive.read(name) for name in archive.namelist()}
        manifest=json.loads(contents['manifest.json']);self.assertEqual(manifest['formatVersion'],2)
        manifest['formatVersion']=1
        contents['manifest.json']=json.dumps(manifest).encode()
        with zipfile.ZipFile(self.path,'w') as archive:
            for name,data in contents.items():archive.writestr(name,data)
        self.assertEqual(Cell.Volume(Topology.Load(self.path)),24)

    def test_original_version_one_fixture(self):
        fixture=Path(__file__).parent/'fixtures/tpy_v1_rooms.tpy'
        with zipfile.ZipFile(fixture) as archive:
            self.assertEqual(json.loads(archive.read('manifest.json'))['formatVersion'],1)
        restored=Topology.Load(fixture)
        self.assertIsNotNone(restored)
        self.assertEqual(len(Topology.Cells(restored)),4)

    def mutate(self,change):
        self.roundtrip(self.graph())
        with zipfile.ZipFile(self.path) as archive:contents={name:archive.read(name) for name in archive.namelist()}
        manifest=json.loads(contents['manifest.json']);change(manifest)
        contents['manifest.json']=json.dumps(manifest).encode()
        with zipfile.ZipFile(self.path,'w') as archive:
            for name,data in contents.items():archive.writestr(name,data)
        self.assertIsNone(Topology.Load(self.path,silent=True))

    def test_invalid_versions_ids_references_and_graph_slots_rejected(self):
        self.mutate(lambda m:m.update(formatVersion=999))
        self.mutate(lambda m:m['objects'].append(m['objects'][0]))
        self.mutate(lambda m:m['objects'][0].update(state={'__tpy_type__':'reference','object':'missing'}))
        def bad_index(m):
            state=TPYCodec._DecodeValue(m['objects'][0]['state'])
            state['edges'][0]['src']=10000
            m['objects'][0]['state']=TPYCodec._EncodeValue(state)
        self.mutate(bad_index)

    def test_unsupported_values_and_python_container_cycles_fail_explicitly(self):
        graph=TGraph(dictionary={'unsupported':object()})
        self.assertFalse(TPYCodec.save(graph,self.path,silent=True))
        self.assertFalse(self.path.exists())
        values=[];values.append(values)
        self.assertFalse(TPYCodec.save(values,self.path,silent=True))
        self.assertFalse(self.path.exists())

if __name__=='__main__':unittest.main()
