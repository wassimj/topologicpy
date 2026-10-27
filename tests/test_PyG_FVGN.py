"""Independent finite-volume identities and FVGN end-to-end regressions."""
import copy
import numpy as np
import pytest
torch=pytest.importorskip('torch')
pytest.importorskip('torch_geometric')
from topologicpy.PyG import PyG
from test_PyG import cpu_determinism


def mesh():
 return np.array([[0.,0.],[1.,0.],[2.,0.],[0.,1.],[1.,1.],[2.,1.]]),np.array([[0,1,4],[0,4,3],[1,2,5],[1,5,4]])


def sequences(frames=5):
 p,c=mesh();m=PyG.BuildFVGNMesh(p,c);nf=len(m['owner']);values=[]
 for i,split in enumerate(['train','train','valid','test']):
  v=np.zeros((frames,len(c),2));v[:,:,0]=(.12+.02*i)*(.97**np.arange(frames))[:,None]
  v[:,:,1]=np.array([.03,-.02,.01,-.01])[None,:]*(.95**np.arange(frames))[:,None]
  values.append(dict(positions=p,cells=c,velocity=v,times=np.arange(frames)*.01,split=split,
                     density=1.2,kinematic_viscosity=.01,graph_features={'inlet':.12+.02*i}))
 return values


def runner(tmp_path,**kwargs):
 options=dict(formulation="physical",latent_dim=16,processor_blocks=2)
 options.update(kwargs.pop('model_options',{}))
 return PyG.ByFVGNSequences(str(tmp_path/'fvgn'),sequences(),use_gpu=False,epochs=3,lr=.002,
       ontology=False,model_options=options,**kwargs)


def test_mesh_geometry_normals_and_winding():
 p,c=mesh();m=PyG.BuildFVGNMesh(p,c)
 assert m['area'].sum()==pytest.approx(2.)
 torch.testing.assert_close(m['normal'].norm(dim=1),torch.ones(len(m['owner'])))
 # The surface vector sum of each closed triangle must vanish.
 total=torch.zeros((len(c),2));vector=m['normal']*m['length'][:,None]
 total.index_add_(0,m['owner'],vector);valid=m['neighbour']>=0
 total.index_add_(0,m['neighbour'][valid],-vector[valid])
 torch.testing.assert_close(total,torch.zeros_like(total),atol=1e-7,rtol=0)
 other=PyG.BuildFVGNMesh(p,c[:,::-1].copy())
 for key in ('normal','area','owner','neighbour','face_vertices'):torch.testing.assert_close(m[key],other[key])


def test_uniform_flow_and_pressure_have_zero_divergence_and_momentum():
 m=PyG.BuildFVGNMesh(*mesh());nf=len(m['owner'])
 result=PyG.FVGNIntegrate(m,torch.tensor([.7,-.2]).expand(nf,2),torch.full((nf,),3.1),torch.zeros(nf,2,2),.01,density=1.2)
 torch.testing.assert_close(result['increment'],torch.zeros(4,2),atol=1e-7,rtol=0)
 torch.testing.assert_close(result['divergence'],torch.zeros(4),atol=2e-7,rtol=0)


def test_internal_flux_conservation_and_autograd():
 m=PyG.BuildFVGNMesh(*mesh());nf=len(m['owner']);inside=m['neighbour']>=0
 velocity=torch.randn(nf,2,requires_grad=True);pressure=torch.zeros(nf,requires_grad=True)
 # Suppress all exterior fluxes; remaining shared-face contributions cancel.
 u=velocity*inside[:,None]
 result=PyG.FVGNIntegrate(m,u,pressure,torch.zeros(nf,2,2),.03)
 total=(result['increment']*m['area'][:,None]).sum(0)
 torch.testing.assert_close(total,torch.zeros(2),atol=1e-7,rtol=0)
 result['increment'].square().sum().backward()
 assert velocity.grad is not None and torch.isfinite(velocity.grad).all() and velocity.grad.abs().sum()>0


def test_pressure_and_viscous_flux_signs():
 p=np.array([[0.,0.],[1.,0.],[0.,1.]]);m=PyG.BuildFVGNMesh(p,np.array([[0,1,2]]));nf=len(m['owner'])
 # p=x -> grad(p)=(1,0); acceleration must be -grad(p)/rho.
 pressure=m['midpoint'][:,0]
 result=PyG.FVGNIntegrate(m,torch.zeros(nf,2),pressure,torch.zeros(nf,2,2),.2,density=2.)
 torch.testing.assert_close(result['increment'],torch.tensor([[-.1,0.]]),atol=1e-7,rtol=0)
 # grad(u_x)=(x,0) -> Laplacian(u_x)=1, hence delta u_x=nu*dt.
 grad=torch.zeros(nf,2,2);grad[:,0,0]=m['midpoint'][:,0]
 result=PyG.FVGNIntegrate(m,torch.zeros(nf,2),torch.zeros(nf),grad,.2,viscosity=.3)
 torch.testing.assert_close(result['increment'],torch.tensor([[.06,0.]]),atol=1e-7,rtol=0)


@pytest.mark.parametrize('kind',['degenerate','duplicate','indices','3d','nonmanifold','labels'])
def test_invalid_meshes_rejected(kind):
 p,c=mesh();tags=None
 if kind=='degenerate':p[4]=p[0]
 if kind=='duplicate':c=np.r_[c,c[:1]]
 if kind=='indices':c[0,0]=99
 if kind=='3d':p=np.c_[p,np.arange(len(p))]
 if kind=='nonmanifold':
  p=np.r_[p,[[.5,.2]]];c=np.r_[c,[[0,4,6]]]
 if kind=='labels':tags=np.zeros(len(PyG.BuildFVGNMesh(p,c)['owner']),dtype=int)
 with pytest.raises(ValueError):PyG.BuildFVGNMesh(p,c,tags)


def test_training_face_boundary_graph_features_and_rollout(tmp_path):
 r=runner(tmp_path);assert r.config.model_name=='fvgn'
 assert r._Loader(r.train_set,batch_size=12).batch_size==1
 before=r.PredictFVGN(0);assert before['increment'].shape==(4,2)
 wall=r.data_list[0].fv_mesh['face_type']==1
 np.testing.assert_array_equal(before['face_velocity'][wall],0.)
 history=r.Train();assert np.isfinite(history['train_loss']).all()
 assert history['train_loss'][-1]<history['train_loss'][0]
 data=next(iter(r._Loader([r.train_set[0]])))
 result=r.model(data);result['prediction'].square().mean().backward()
 assert all(p.grad is not None for p in r.model.processors[0].parameters())
 other=data.clone();other.u=other.u+2
 assert not torch.allclose(r.model(data)['prediction'],r.model(other)['prediction'])
 other=data.clone();other.edge_attr=other.edge_attr+2
 assert not torch.allclose(r.model(data)['prediction'],r.model(other)['prediction'])
 out=r.Rollout(12,30)
 assert out['velocity'].shape==(31,4,2) and np.isfinite(out['velocity']).all()
 assert out['face_pressure'].shape[0]==30 and out['divergence'].shape==(30,4)
 assert np.isfinite(list(r.Test().values())).all()


def test_checkpoint_resume_and_reload_are_exact(tmp_path):
 a=runner(tmp_path,model_options={'noise_std':.01})
 a.Train(epochs=3)
 b=PyG.ByCFDDataset(str(tmp_path/'fvgn'));b.Train(epochs=1)
 file=str(tmp_path/'partial.pt');b.SaveModel(file)
 c=PyG.ByCFDDataset(str(tmp_path/'fvgn'));c.LoadModel(file,resume_training=True);c.Train(epochs=2,resume=True)
 for key,value in a.model.state_dict().items():torch.testing.assert_close(value,c.model.state_dict()[key],atol=0,rtol=0)
 for key,value in a.PredictFVGN(0).items():np.testing.assert_array_equal(value,c.PredictFVGN(0)[key])


def test_vertex_data_conversion_and_inlet_conditions(tmp_path):
 p,c=mesh();m=PyG.BuildFVGNMesh(p,c);tags=m['face_type'].numpy().copy()
 tags[(m['midpoint'][:,0].numpy()==0)]=2
 tags[(m['midpoint'][:,0].numpy()==2)]=3
 nf=len(tags);values=[]
 for split in ['train','valid','test']:
  v=np.zeros((3,len(p),2));v[:,:,0]=.2
  f=np.zeros((3,nf,2));f[:,:,0]=.2;f[:,tags==1]=0;f[1:,tags==2,0]=.4
  values.append(dict(positions=p,cells=c,velocity=v,pressure=np.zeros((3,len(p))),face_types=tags,face_velocity=f,split=split))
 r=PyG.ByFVGNSequences(str(tmp_path/'vertex'),values,fieldLocation='vertex',use_gpu=False,epochs=1,
      model_options=dict(formulation="physical",latent_dim=8,processor_blocks=1,pressure_loss_weight=.2,continuity_loss_weight=.01))
 r.Train()
 out=r.PredictFVGN(1);np.testing.assert_allclose(out['face_velocity'][tags==2,0],.4)
 # Rollout takes caller-known future boundary values, not archived interior targets.
 before=r.Rollout(0,2)['velocity']
 schedule=np.broadcast_to(f[0],(2,nf,2)).copy();schedule[:,tags==2,0]=1.
 after=r.Rollout(0,2,faceBoundaryValues=schedule)['velocity']
 assert not np.allclose(before,after)
 data=r.data_list[0].clone();data.fv_target_velocity.fill_(1000.)
 r.model.eval()
 with torch.no_grad():torch.testing.assert_close(r.model(r.data_list[0])['prediction'],r.model(data)['prediction'])


@pytest.mark.parametrize('options',[{'latent_dim':0},{'processor_blocks':False},{'unknown':1},{'pressure_scale':0},{'noise_std':-1},{'momentum_loss_weight':0}])
def test_bad_options_rejected_before_export(tmp_path,options):
 with pytest.raises(ValueError):PyG.ByFVGNSequences(str(tmp_path/'bad'),sequences(),model_options=dict(formulation='physical',**options))
 assert not (tmp_path/'bad'/'graphs.csv').exists()


def test_missing_pressure_and_inlet_data_rejected(tmp_path):
 with pytest.raises(ValueError,match='pressure'):runner(tmp_path,model_options={'pressure_loss_weight':1.})
 seq=sequences();m=PyG.BuildFVGNMesh(*mesh());tags=m['face_type'].numpy().copy();tags[0]=2
 seq[0]['face_types']=tags
 with pytest.raises(ValueError,match='inlet'):PyG.ByFVGNSequences(str(tmp_path/'inlet'),seq)

def test_cell_renumbering_is_equivariant(tmp_path):
 r=runner(tmp_path);r.model.eval();d=r.data_list[0].clone()
 permutation=torch.tensor([3,1,0,2]);inverse=permutation.argsort()
 changed=d.clone();changed.x=d.x[permutation];changed.pos=d.pos[permutation]
 changed.edge_index=inverse[d.edge_index]
 changed.fv_mesh=PyG.BuildFVGNMesh(d.fv_mesh['vertices'],d.fv_mesh['cells'][permutation],d.fv_mesh['face_type'])
 with torch.no_grad():
  first=r.model(d)['prediction'];second=r.model(changed)['prediction']
 torch.testing.assert_close(first[permutation],second,atol=1e-5,rtol=1e-5)


def test_future_boundary_input_and_unused_reference_fields(tmp_path):
 seq=sequences();m=PyG.BuildFVGNMesh(*mesh());nf=len(m['owner']);tags=m['face_type'].numpy().copy();tags[0]=2
 for s in seq:
  s['face_types']=tags
  face=np.zeros((5,nf,2));face[:,0,0]=np.arange(5)*.1;s['face_velocity']=face
 r=PyG.ByFVGNSequences(str(tmp_path/'dynamic'),seq,use_gpu=False,ontology=False,
       model_options=dict(formulation="physical",latent_dim=8,processor_blocks=1))
 assert r.data_list[0].fv_face_velocity[0,0]==pytest.approx(.1)
 r.model.eval();d=r.data_list[0].clone();changed=d.clone()
 changed.fv_target_velocity.fill_(1000.)
 internal=changed.fv_mesh['face_type']==0
 changed.fv_face_velocity[internal]=1000.
 with torch.no_grad():torch.testing.assert_close(r.model(d)['prediction'],r.model(changed)['prediction'])
 # Both directed adjacency feature rows must contribute to their physical face.
 ids=d.fv_edge_map[d.fv_edge_map[:,0]>=0][0]
 with torch.no_grad():base=r.model(d)['prediction']
 for idx in ids:
  other=d.clone();other.edge_attr[idx]+=3
  with torch.no_grad():assert not torch.allclose(base,r.model(other)['prediction'])


def test_changed_physics_prevents_training_resume(tmp_path):
 r=runner(tmp_path);r.Train(epochs=1);file=str(tmp_path/'model.pt');r.SaveModel(file)
 archive=tmp_path/'fvgn'/'fvgn-sequence-0.npz'
 with np.load(archive,allow_pickle=False) as z:fields={k:z[k] for k in z.files}
 fields['density']=np.asarray(2.5)
 np.savez_compressed(archive,**fields)
 changed=PyG.ByCFDDataset(str(tmp_path/'fvgn'))
 with pytest.raises(ValueError,match='different dataset'):changed.LoadModel(file,resume_training=True)


def test_rollout_boundary_shapes_and_nonfinite_diagnostics(tmp_path):
 r=runner(tmp_path);nf=len(r.data_list[0].fv_mesh['owner'])
 with pytest.raises(ValueError,match='faceBoundaryValues'):r.Rollout(0,2,faceBoundaryValues=np.zeros((1,nf,2)))
 with pytest.raises(ValueError):r.Rollout(0,2,timeStep=.5)
 with torch.no_grad():r.model.decoder[-1].weight.fill_(float('nan'))
 with pytest.raises(FloatingPointError,match='step 1'):r.Rollout(0,2)


@pytest.mark.parametrize('dt,rho,nu',[(0,1,.1),(.1,0,.1),(.1,1,-.1),(float('nan'),1,.1)])
def test_invalid_physical_parameters(dt,rho,nu):
 m=PyG.BuildFVGNMesh(*mesh());nf=len(m['owner'])
 with pytest.raises(ValueError):PyG.FVGNIntegrate(m,torch.zeros(nf,2),torch.zeros(nf),torch.zeros(nf,2,2),dt,rho,nu)

def test_single_cell_mesh_has_no_adjacency_edges(tmp_path):
 p=np.array([[0.,0.],[1.,0.],[0.,1.]]);cells=np.array([[0,1,2]])
 seq=[dict(positions=p,cells=cells,velocity=np.zeros((3,1,2)),split=split) for split in ('train','valid','test')]
 r=PyG.ByFVGNSequences(str(tmp_path/'single'),seq,use_gpu=False,epochs=1,
       model_options=dict(formulation="physical",latent_dim=8,processor_blocks=1))
 assert r.data_list[0].num_edges==0
 r.Train();assert np.isfinite(r.Rollout(0,3)['velocity']).all()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA is unavailable')
def test_cuda_training_prediction_and_rollout(tmp_path):
 r=PyG.ByFVGNSequences(str(tmp_path/'gpu'),sequences(),use_gpu=True,epochs=1,
       model_options=dict(formulation="physical",latent_dim=8,processor_blocks=1))
 assert r.device.type=='cuda'
 assert np.isfinite(r.Train()['train_loss']).all()
 assert np.isfinite(r.PredictFVGN(0)['increment']).all()
 assert np.isfinite(r.Rollout(0,5)['velocity']).all()

def test_geometry_changes_require_reexport(tmp_path):
 r=runner(tmp_path)
 with pytest.raises(ValueError,match='Re-export FVGN'):r.SetGraphData(0,pos=r.data_list[0].pos+1)
