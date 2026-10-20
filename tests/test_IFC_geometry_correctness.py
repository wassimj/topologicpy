import pytest
from topologicpy.IFC import IFCFastTopology as F, IFCFastEntity as E
from topologicpy.Topology import Topology
from topologicpy.Cell import Cell
from topologicpy.Face import Face

R = lambda i: ('REF', i)
I = F._identity()

def volume(mesh):
    if not mesh['vertices']:
        return 0
    solid = Topology.ByGeometry(vertices=mesh['vertices'], faces=mesh['faces'], topologyType='cell', silent=True)
    assert solid is not None
    return sum(Cell.Volume(c) for c in Topology.Cells(solid)) if not Topology.IsInstance(solid, 'Cell') else Cell.Volume(solid)

def test_boolean_operations_and_empty_results():
    entities = {1:E(1,'IFCCARTESIANPOINT',[[0,0,0]]), 2:E(2,'IFCCARTESIANPOINT',[[1,0,0]]),
                3:E(3,'IFCBOUNDINGBOX',[R(1),2,2,2]),4:E(4,'IFCBOUNDINGBOX',[R(2),2,2,2])}
    for op, expected in [('DIFFERENCE',4),('INTERSECTION',4),('UNION',12)]:
        mesh = F._mesh_from_item(E(5,'IFCBOOLEANRESULT',[op,R(3),R(4)]),entities,I)
        assert volume(mesh) == pytest.approx(expected)
    assert volume(F._mesh_from_item(E(5,'IFCBOOLEANRESULT',['DIFFERENCE',R(3),R(3)]),entities,I)) == 0

def test_ordered_indexed_profile_and_void_extrusion():
    e={1:E(1,'IFCCARTESIANPOINTLIST2D',[[[0,0],[4,4],[4,0],[0,4]]]),
       2:E(2,'IFCINDEXEDPOLYCURVE',[R(1),[('CALL','IFCLINEINDEX',[[1,3,2,4,1]])],False]),
       3:E(3,'IFCCARTESIANPOINTLIST2D',[[[1,1],[3,1],[3,3],[1,3],[1,1]]]),
       4:E(4,'IFCINDEXEDPOLYCURVE',[R(3),None,False]),
       5:E(5,'IFCARBITRARYPROFILEDEFWITHVOIDS',['AREA',None,R(2),[R(4)]]),
       6:E(6,'IFCDIRECTION',[[0,0,1]])}
    mesh=F._mesh_from_item(E(7,'IFCEXTRUDEDAREASOLID',[R(5),None,R(6),2]),e,I)
    assert volume(mesh) == pytest.approx(24)

def test_face_hole_outer_order_and_orientation():
    e={}
    for i,p in enumerate([[0,0,0],[4,0,0],[4,4,0],[0,4,0],[1,1,0],[3,1,0],[3,3,0],[1,3,0]],1):
        e[i]=E(i,'IFCCARTESIANPOINT',[p])
    e.update({9:E(9,'IFCPOLYLOOP',[[R(i) for i in range(1,5)]]),10:E(10,'IFCPOLYLOOP',[[R(i) for i in range(5,9)]]),
              11:E(11,'IFCFACEOUTERBOUND',[R(9),False]),12:E(12,'IFCFACEBOUND',[R(10),True])})
    face=E(13,'IFCFACE',[[R(12),R(11)]])
    assert F._face_loops(face,e)[0][0] == [0,4,0]
    mesh=F._face_mesh(face,e,I,1)
    topo=Topology.ByGeometry(vertices=mesh['vertices'],faces=mesh['faces'],silent=True)
    assert sum(Face.Area(f) for f in Topology.Faces(topo)) == pytest.approx(12)
    e[14]=E(14,'IFCPLANE',[None])
    e[15]=E(15,'IFCADVANCEDFACE',[[R(11)],R(14),True])
    assert F._shell_mesh(E(16,'IFCCLOSEDSHELL',[[R(15)]]),e,I,1)['faces']

def test_oriented_edge_uses_schema_attribute_positions():
    e={1:E(1,'IFCCARTESIANPOINT',[[0,0,0]]),2:E(2,'IFCCARTESIANPOINT',[[2,0,0]]),
       3:E(3,'IFCVERTEXPOINT',[R(1)]),4:E(4,'IFCVERTEXPOINT',[R(2)]),5:E(5,'IFCEDGE',[R(3),R(4)])}
    assert F._edge_points(E(6,'IFCORIENTEDEDGE',['*','*',R(5),False]),e)==[[2,0,0],[0,0,0]]

def test_opening_uses_world_placement_and_is_subtracted():
    e={1:E(1,'IFCCARTESIANPOINT',[[0,0,0]]),2:E(2,'IFCCARTESIANPOINT',[[1,0,0]]),
       3:E(3,'IFCBOUNDINGBOX',[R(1),4,4,4]),4:E(4,'IFCBOUNDINGBOX',[R(2),2,4,4]),
       5:E(5,'IFCSHAPEREPRESENTATION',[None,'Body','BoundingBox',[R(3)]]),
       6:E(6,'IFCSHAPEREPRESENTATION',[None,'Body','BoundingBox',[R(4)]]),
       7:E(7,'IFCPRODUCTDEFINITIONSHAPE',[None,None,[R(5)]]),8:E(8,'IFCPRODUCTDEFINITIONSHAPE',[None,None,[R(6)]]),
       9:E(9,'IFCWALL',[None,None,None,None,None,None,R(7)]),
       10:E(10,'IFCOPENINGELEMENT',[None,None,None,None,None,None,R(8)]),
       11:E(11,'IFCRELVOIDSELEMENT',[None,None,None,None,R(9),R(10)])}
    assert volume(F.MeshDataByProduct(e[9],e,scale=1)) == pytest.approx(32)

@pytest.mark.parametrize('agreement,expected_z',[(True,(1,2)),(False,(0,1))])
def test_half_space_clipping_agreement_and_translation(agreement,expected_z):
    e={1:E(1,'IFCCARTESIANPOINT',[[0,0,0]]),2:E(2,'IFCBOUNDINGBOX',[R(1),2,2,2]),
       3:E(3,'IFCCARTESIANPOINT',[[0,0,1]]),4:E(4,'IFCAXIS2PLACEMENT3D',[R(3),None,None]),
       5:E(5,'IFCPLANE',[R(4)]),6:E(6,'IFCHALFSPACESOLID',[R(5),agreement])}
    mesh=F._mesh_from_item(E(7,'IFCBOOLEANCLIPPINGRESULT',['DIFFERENCE',R(2),R(6)]),e,I)
    assert volume(mesh)==pytest.approx(4)
    assert (min(p[2] for p in mesh['vertices']),max(p[2] for p in mesh['vertices']))==pytest.approx(expected_z)

def test_indexed_arc_is_sampled_through_middle_point():
    e={1:E(1,'IFCCARTESIANPOINTLIST2D',[[[1,0],[0,1],[-1,0]]])}
    curve=E(2,'IFCINDEXEDPOLYCURVE',[R(1),[('CALL','IFCARCINDEX',[[1,2,3]])],False])
    points=F._curve_points_local(curve,e)
    assert len(points)>3
    assert points[0]==[1,0,0] and points[-1]==[-1,0,0]
    assert max(p[1] for p in points)==pytest.approx(1)
    assert all(p[0]**2+p[1]**2==pytest.approx(1) for p in points)

def test_unsupported_advanced_surface_is_not_flattened():
    e={1:E(1,'IFCCYLINDRICALSURFACE',[None,1])}
    with pytest.raises(ValueError,match='Curved IfcAdvancedFace'):
        F._face_mesh(E(2,'IFCADVANCEDFACE',[[],R(1),True]),e,I,1)

def test_unsupported_boolean_operand_is_not_ignored():
    e={1:E(1,'IFCCARTESIANPOINT',[[0,0,0]]),2:E(2,'IFCBOUNDINGBOX',[R(1),2,2,2]),3:E(3,'IFCUNSUPPORTED',[])}
    with pytest.raises(ValueError,match='Unsupported IFC Boolean operand'):
        F._mesh_from_item(E(4,'IFCBOOLEANRESULT',['DIFFERENCE',R(2),R(3)]),e,I)

@pytest.mark.parametrize('operation,expected', [('UNION',16),('INTERSECTION',0),('DIFFERENCE',8)])
def test_disjoint_solid_booleans(operation, expected):
    e={1:E(1,'IFCCARTESIANPOINT',[[0,0,0]]),2:E(2,'IFCCARTESIANPOINT',[[5,0,0]]),
       3:E(3,'IFCBOUNDINGBOX',[R(1),2,2,2]),4:E(4,'IFCBOUNDINGBOX',[R(2),2,2,2])}
    assert volume(F._mesh_from_item(E(5,'IFCBOOLEANRESULT',[operation,R(3),R(4)]),e,I)) == pytest.approx(expected)

def test_native_boolean_failure_does_not_become_an_empty_success(monkeypatch):
    if Topology._IsTopologicCoreBackend():
        pytest.skip('PythonOCC status distinction')
    from topologicpy.pythonocc_backend.topology import Topology as NativeTopology
    monkeypatch.setattr(NativeTopology, '_binary_boolean', lambda *args,**kwargs: None)
    e={1:E(1,'IFCCARTESIANPOINT',[[0,0,0]])}
    box=F._mesh_from_item(E(2,'IFCBOUNDINGBOX',[R(1),2,2,2]),e,I)
    with pytest.raises(ValueError,match='Native IFC Boolean operation failed'):
        F._boolean_mesh(box,box,'DIFFERENCE')

def test_public_mesh_import_applies_scaled_nested_opening_placements():
    from topologicpy.IFC import IFC, IFCModel
    e={1:E(1,'IFCCARTESIANPOINT',[[0,0,0]]),2:E(2,'IFCBOUNDINGBOX',[R(1),4000,4000,4000]),
       3:E(3,'IFCBOUNDINGBOX',[R(1),2000,4000,4000]),
       4:E(4,'IFCSHAPEREPRESENTATION',[None,'Body','BoundingBox',[R(2)]]),
       5:E(5,'IFCSHAPEREPRESENTATION',[None,'Body','BoundingBox',[R(3)]]),
       6:E(6,'IFCPRODUCTDEFINITIONSHAPE',[None,None,[R(4)]]),7:E(7,'IFCPRODUCTDEFINITIONSHAPE',[None,None,[R(5)]]),
       8:E(8,'IFCCARTESIANPOINT',[[10000,0,0]]),9:E(9,'IFCCARTESIANPOINT',[[1000,0,0]]),
       10:E(10,'IFCAXIS2PLACEMENT3D',[R(8),None,None]),11:E(11,'IFCAXIS2PLACEMENT3D',[R(9),None,None]),
       12:E(12,'IFCLOCALPLACEMENT',[None,R(10)]),13:E(13,'IFCLOCALPLACEMENT',[R(12),R(11)]),
       14:E(14,'IFCWALL',['wall',None,'Wall',None,None,R(12),R(6)]),
       15:E(15,'IFCOPENINGELEMENT',['opening',None,'Opening',None,None,R(13),R(7)]),
       16:E(16,'IFCRELVOIDSELEMENT',[None,None,None,None,R(14),R(15)])}
    model=IFCModel(entities=e)
    mesh=IFC.MeshData(model,includeTypes=['IFCWALL'],scale=.001,ontology=False,silent=True)
    assert volume(mesh)==pytest.approx(32)
    assert min(p[0] for p in mesh['vertices'])==pytest.approx(10)
    assert max(p[0] for p in mesh['vertices'])==pytest.approx(14)
    assert len(mesh['product_ranges'])==1
    assert IFC.MeshData(model,includeTypes=['IFCWALL'],scale=.001,ontology=False,silent=True)['vertices']==mesh['vertices']

def test_face_without_explicit_outer_bound_uses_enclosing_loop():
    e={}
    for i,p in enumerate([[0,0,0],[4,0,0],[4,4,0],[0,4,0],[1,1,0],[3,1,0],[3,3,0],[1,3,0]],1):
        e[i]=E(i,'IFCCARTESIANPOINT',[p])
    e.update({9:E(9,'IFCPOLYLOOP',[[R(i) for i in range(1,5)]]),10:E(10,'IFCPOLYLOOP',[[R(i) for i in range(5,9)]]),
              11:E(11,'IFCFACEBOUND',[R(9),True]),12:E(12,'IFCFACEBOUND',[R(10),True])})
    mesh=F._face_mesh(E(13,'IFCFACE',[[R(12),R(11)]]),e,I,1)
    topo=Topology.ByGeometry(vertices=mesh['vertices'],faces=mesh['faces'],silent=True)
    assert sum(Face.Area(f) for f in Topology.Faces(topo)) == pytest.approx(12)

@pytest.mark.parametrize('kind',['IFCBOXEDHALFSPACE','IFCPOLYGONALBOUNDEDHALFSPACE'])
def test_bounded_half_space_limits_the_cut(kind):
    e={1:E(1,'IFCCARTESIANPOINT',[[0,0,0]]),2:E(2,'IFCBOUNDINGBOX',[R(1),4,4,4]),
       3:E(3,'IFCCARTESIANPOINT',[[0,0,2]]),4:E(4,'IFCAXIS2PLACEMENT3D',[R(3),None,None]),
       5:E(5,'IFCPLANE',[R(4)]),6:E(6,'IFCBOUNDINGBOX',[R(1),2,4,4]),
       7:E(7,'IFCCARTESIANPOINTLIST2D',[[[0,0],[2,0],[2,4],[0,4],[0,0]]]),
       8:E(8,'IFCINDEXEDPOLYCURVE',[R(7),None,False])}
    half=E(9,kind,[R(5),False,R(6)] if kind=='IFCBOXEDHALFSPACE' else [R(5),False,None,R(8)])
    e[9]=half
    mesh=F._mesh_from_item(E(10,'IFCBOOLEANCLIPPINGRESULT',['DIFFERENCE',R(2),R(9)]),e,I)
    assert volume(mesh)==pytest.approx(48)
