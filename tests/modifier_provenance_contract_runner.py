"""Execute real public methods and Provenance with a controlled backend double.

These checks verify return shape, forwarding, failure/no-op behaviour and capture
scope. They do NOT verify OCCT geometry or native modifier correspondence.
"""
import ast, contextlib, importlib.util, math, sys, types, unittest
from pathlib import Path

repo=Path(__file__).resolve().parents[1]
package=types.ModuleType('topologicpy'); package.__path__=[]
sys.modules['topologicpy']=package
class Shape:
    def __init__(self,kind='Cell',point=(0,0,0),children=None):
        self.kind=kind; self.point=point; self.children=children or []
    def Copy(self): return perform(self,'Copy')
    def DeepCopy(self): return perform(self,'Copy')

state={'sink':None,'calls':0,'legacy':False,'emit':True,'children_only':False,'flags':[]}
@contextlib.contextmanager
def capture(operation_node,role_nodes,stage,sink):
    previous=state['sink']; state['sink']=sink
    try: yield
    finally: state['sink']=previous

def perform(source,operation,flag=True):
    state['calls']+=1; state['flags'].append(flag)
    target=Shape(source.kind,children=[Shape(c.kind) for c in source.children])
    if state['sink'] is not None and state['emit']:
        pairs=list(zip(source.children,target.children))
        if not state['children_only']: pairs.insert(0,(source,target))
        for a,b in pairs:
            state['sink'].append({'source':a,'result':b,'sourceType':a.kind,'resultType':b.kind,
                                  'sourceRole':'source','operation':operation,'relation':'modified'})
    return target

lineage=types.ModuleType('topologicpy.pythonocc_backend._csg_lineage')
lineage.capture=capture; lineage.materialise_record=lambda r:dict(r)
sys.modules[lineage.__name__]=lineage
class Utility:
    @staticmethod
    def Translate(t,x,y,z,flag): return perform(t,'Transform',flag)
    @staticmethod
    def Rotate(t,origin,x,y,z,angle,flag): return perform(t,'Transform',flag)
    @staticmethod
    def Scale(t,origin,x,y,z,flag): return perform(t,'GTransform',flag)
class FakeCore:
    TopologyUtility=Utility
    @staticmethod
    def InstanceCall(t,name): return getattr(t,name)()

vertex=types.ModuleType('topologicpy.Vertex')
class FakeVertex:
    @staticmethod
    def ByCoordinates(x,y,z): return Shape('Vertex',(x,y,z))
    @staticmethod
    def Coordinates(v,mantissa=12): return list(v.point)
vertex.Vertex=FakeVertex; sys.modules[vertex.__name__]=vertex
dictionary=types.ModuleType('topologicpy.Dictionary'); dictionary.Dictionary=object
sys.modules[dictionary.__name__]=dictionary

tree=ast.parse((repo/'src/topologicpy/Topology.py').read_text(encoding='utf-8-sig'))
klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Topology')
wanted={'_WithModifierProvenance','Copy','Translate','Rotate','Scale'}
klass.body=[n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name in wanted]
namespace={'Core':FakeCore}
exec(compile(ast.Module(body=[klass],type_ignores=[]),'actual-public-methods','exec'),namespace)
Topology=namespace['Topology']
Topology.IsInstance=staticmethod(lambda o,k: isinstance(o,Shape) and (k=='Topology' or o.kind==k))
Topology.IsSame=staticmethod(lambda a,b,**kw:a is b)
Topology.TypeAsString=staticmethod(lambda o:o.kind)
Topology._IsTopologicCoreBackend=staticmethod(lambda:state['legacy'])
for name,kind in [('Vertices','Vertex'),('Edges','Edge'),('Wires','Wire'),('Faces','Face'),
                  ('Shells','Shell'),('Cells','Cell'),('CellComplexes','CellComplex')]:
    setattr(Topology,name,staticmethod(lambda o,silent=True,kind=kind:
                                      ([o] if o.kind==kind else [])+[c for c in o.children if c.kind==kind]))
module=types.ModuleType('topologicpy.Topology'); module.Topology=Topology
sys.modules[module.__name__]=module
spec=importlib.util.spec_from_file_location('topologicpy.Provenance',repo/'src/topologicpy/Provenance.py')
provmodule=importlib.util.module_from_spec(spec); sys.modules[spec.name]=provmodule; spec.loader.exec_module(provmodule)
Provenance=provmodule.Provenance

class ContractChecks(unittest.TestCase):
    def setUp(self):
        state.update(sink=None,calls=0,legacy=False,emit=True,children_only=False,flags=[])
        self.source=Shape(children=[Shape('Face'),Shape('Vertex')])

    def test_all_public_operation_names_and_single_execution(self):
        for name in ['Translate','Scale','Rotate','Copy']:
            with self.subTest(operation=name):
                state['calls']=0
                kwargs={'angle':25} if name=='Rotate' else {'x':2} if name in ['Scale','Translate'] else {}
                result,p=getattr(Topology,name)(self.source,**kwargs,returnProvenance=True)
                self.assertEqual(state['calls'],1)
                self.assertTrue(p.supported)
                self.assertIs(p.result,result)
                self.assertIs(p.sources['self'],self.source)
                self.assertEqual({r['operation'] for r in p.History()},{name})

    def test_semantic_records_and_queries_retain_public_operands(self):
        for name in ['Translate', 'Scale', 'Rotate', 'Copy']:
            with self.subTest(operation=name):
                kwargs = {'angle':25} if name=='Rotate' else {'x':2} if name in ['Scale','Translate'] else {}
                result,p = getattr(Topology,name)(self.source,**kwargs,returnProvenance=True)
                self.assertEqual(len(p.Records()),len(p.History()))
                self.assertEqual({r['sourceRole'] for r in p.Records()},{'self'})
                self.assertEqual(len(p.Origins(result)),1)
                self.assertIs(p.Origins(result)[0]['source'],self.source)
                self.assertEqual(len(p.Descendants(self.source)),1)
                self.assertIs(p.Descendants(self.source)[0]['result'],result)

    def test_composed_semantic_queries_trace_to_original(self):
        a,p1 = Topology.Translate(self.source,x=1,returnProvenance=True)
        b,p2 = Topology.Scale(a,x=2,returnProvenance=True)
        c,p3 = Topology.Rotate(b,angle=25,returnProvenance=True)
        d,p4 = Topology.Copy(c,returnProvenance=True)
        p = Provenance.Compose(p1,p2,p3,p4)
        self.assertEqual(len(p.Origins(d)),1)
        self.assertIs(p.Origins(d)[0]['source'],self.source)
        self.assertEqual(len(p.Descendants(self.source)),1)
        self.assertIs(p.Descendants(self.source)[0]['result'],d)

    def test_noop_semantic_queries_are_not_empty(self):
        result,p = Topology.Rotate(self.source,angle=0,returnProvenance=True,silent=True)
        self.assertTrue(p.Records())
        self.assertEqual({r['relation'] for r in p.Records()},{'unchanged'})
        self.assertIs(p.Origins(result)[0]['source'],self.source)

    def test_transfer_false_is_forwarded_without_suppressing_capture(self):
        for name in ['Translate','Scale','Rotate']:
            with self.subTest(operation=name):
                kwargs={'angle':25} if name=='Rotate' else {'x':2}
                result,p=getattr(Topology,name)(self.source,**kwargs,transferDictionaries=False,returnProvenance=True)
                self.assertFalse(state['flags'][-1]); self.assertTrue(p.supported)

    def test_default_return_is_unchanged(self):
        for name in ['Translate','Scale','Rotate','Copy']:
            self.assertIsInstance(getattr(Topology,name)(self.source),Shape)

    def test_shallow_and_deep_copy_are_independent(self):
        for deep in [False,True]:
            result,p=Topology.Copy(self.source,deep=deep,returnProvenance=True)
            self.assertIsNot(result,self.source); self.assertTrue(p.supported)
            self.assertEqual(p.metadata['parameters']['deep'],deep)

    def test_noop_rotation_has_unchanged_subtopologies(self):
        result,p=Topology.Rotate(self.source,angle=0,returnProvenance=True)
        self.assertIs(result,self.source); self.assertEqual(state['calls'],0)
        self.assertTrue(p.supported)
        self.assertEqual({r['relation'] for r in p.History()},{'unchanged'})
        self.assertEqual(len(p.History()),3)

    def test_empty_capture_is_explicit_and_not_replayed(self):
        state['emit']=False
        result,p=Topology.Translate(self.source,x=1,returnProvenance=True)
        self.assertIsInstance(result,Shape); self.assertFalse(p.supported)
        self.assertEqual(p.metadata['reason'],'native_history_empty'); self.assertEqual(state['calls'],1)

    def test_unsupported_backend_runs_callback_once(self):
        state['legacy']=True
        calls=[]
        def apply(): calls.append(1); return Shape()
        result,p=Topology._WithModifierProvenance(self.source,'Translate',apply)
        self.assertFalse(p.supported); self.assertEqual(calls,[1])
        self.assertEqual(p.metadata['reason'],'backend_history_unavailable')

    def test_invalid_inputs_return_unsupported_pairs(self):
        for name in ['Translate','Scale','Rotate','Copy']:
            result,p=getattr(Topology,name)(None,returnProvenance=True,silent=True)
            self.assertIsNone(result); self.assertFalse(p.supported)
            self.assertEqual(p.metadata['reason'],'operation_failed')

    def test_singular_nonfinite_and_nonnumeric_scale_are_rejected(self):
        for axis in ['x','y','z']:
            for factor in [0,float('inf'),float('nan'),'invalid']:
                with self.subTest(axis=axis,factor=factor):
                    result,p=Topology.Scale(self.source,**{axis:factor},returnProvenance=True,silent=True)
                    self.assertIsNone(result); self.assertFalse(p.supported)
        self.assertEqual(state['calls'],0)

    def test_negative_and_nonuniform_scale_are_forwarded(self):
        result,p=Topology.Scale(self.source,x=-2,y=.5,z=3,returnProvenance=True)
        self.assertTrue(p.supported); self.assertEqual(p.metadata['parameters']['x'],-2)

    def test_origin_is_recorded_as_coordinates(self):
        result,p=Topology.Scale(self.source,origin=Shape('Vertex',(3,4,5)),x=2,returnProvenance=True)
        self.assertEqual(p.metadata['parameters']['origin'],[3,4,5])

    def test_container_root_evidence_is_labelled(self):
        state['children_only']=True; self.source.kind='Cluster'
        result,p=Topology.Translate(self.source,x=1,returnProvenance=True)
        roots=[r for r in p.History() if r['source'] is self.source and r['result'] is result]
        self.assertEqual(len(roots),1); self.assertEqual(roots[0]['captureMechanism'],'operation-root')

if __name__=='__main__': unittest.main(verbosity=2)
