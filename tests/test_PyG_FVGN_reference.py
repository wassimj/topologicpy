"""Numerical fixtures from the pinned, unmodified FVGN author implementation.

Tests do not import or install author code or torch-scatter. Provenance and exact
reference commit are stored beside the compact golden fixture in tests/data.
"""
from pathlib import Path
from types import SimpleNamespace
import copy,json,hashlib
import numpy as np
import pytest
torch=pytest.importorskip('torch')
pytest.importorskip('torch_geometric')
from torch_geometric.data import Data
from topologicpy.PyG import PyG
from test_PyG import cpu_determinism


@pytest.fixture
def reference():
    path=Path(__file__).parent/'data/fvgn-author-golden.npz'
    metadata=json.loads(path.with_suffix('.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==metadata['fixture_sha256']
    with np.load(path,allow_pickle=False) as source:arrays={k:source[k] for k in source.files}
    options=dict(formulation='reference',latent_dim=8,processor_blocks=2,random_edge_flip=False,
                 noise_std=0.,normalizer_warmup_passes=0,pressure_loss_weight=1.,continuity_loss_weight=0.)
    context=dict(options=options,node_dim=6,edge_dim=4,graph_dim=0,output_dim=2,
                 velocity_indices=[0,1],normalization={})
    model=PyG._FVGNModel(context)
    model.load_state_dict({k[7:]:torch.from_numpy(v.copy()) for k,v in arrays.items() if k.startswith('state__')})
    data=Data(**{k[7:]:torch.from_numpy(v.copy()) for k,v in arrays.items() if k.startswith('input__')})
    data.fv_mesh={k[6:]:torch.from_numpy(v.copy()) for k,v in arrays.items() if k.startswith('mesh__')}
    runner=object.__new__(PyG);runner.config=SimpleNamespace(model_options=options);runner.model=model
    return runner,data,arrays


def test_author_forward_loss_and_backward_fixture(reference):
    runner,data,expected=reference
    model=runner.model;model.train();result=model(data);runner._fvgn_output=result
    for name in ('prediction','increment','face_velocity','face_pressure','normalized_compensation',
                 '_face_normalized','_delta_normalized','_target_delta','_target_face','_continuity'):
        np.testing.assert_allclose(result[name].detach(),expected['expected__'+name],atol=2e-6,rtol=2e-5,err_msg=name)
    loss=runner._FVGNObjective(data,result['prediction'],data.train_mask)
    np.testing.assert_allclose(loss.detach(),expected['expected__loss'],atol=1e-6,rtol=1e-6)
    loss.backward()
    for name,parameter in model.named_parameters():
        key='gradient__'+name
        if key in expected:
            assert parameter.grad is not None,name
            np.testing.assert_allclose(parameter.grad,expected[key],atol=2e-6,rtol=2e-4,err_msg=name)


def test_pressure_detachment_matches_reference(reference):
    runner,data,_=reference;result=runner.model(data)
    momentum=(result['_delta_normalized']-result['_target_delta']).square().mean()
    grad=torch.autograd.grad(momentum,result['_face_normalized'],retain_graph=True)[0]
    torch.testing.assert_close(grad[:,2],torch.zeros_like(grad[:,2]),atol=0,rtol=0)
    assert grad[:,:2].abs().sum()>0 and grad[:,3:].abs().sum()>0
    pressure=(result['_face_normalized'][:,2]-result['_target_face'][:,2]).square().mean()
    pressure_grad=torch.autograd.grad(pressure,result['_face_normalized'])[0]
    assert pressure_grad[:,2].abs().sum()>0


def test_reference_statistics_are_frozen_and_targets_do_not_drive_inference(reference):
    runner,data,_=reference;model=runner.model;model.eval()
    before={k:v.clone() for k,v in model.state_dict().items()}
    baseline=model(data)['prediction']
    changed=data.clone();changed.y.fill_(1000.);changed.fv_target_velocity.fill_(-1000.);changed.fv_target_pressure.fill_(5000.)
    torch.testing.assert_close(model(changed)['prediction'],baseline,atol=0,rtol=0)
    for key,value in model.state_dict().items():torch.testing.assert_close(value,before[key],atol=0,rtol=0)


def reference_sequences():
    p=np.array([[0.,0.],[1.,0.],[2.,0.],[0.,1.],[1.,1.],[2.,1.]])
    c=np.array([[0,1,4],[0,4,3],[1,2,5],[1,5,4]])
    seq=[]
    for i,split in enumerate(['train','train','valid','test']):
        v=np.zeros((4,4,2));v[:,:,0]=(.2+.01*i)*(.99**np.arange(4))[:,None]
        seq.append(dict(positions=p,cells=c,velocity=v,pressure=np.zeros((4,4)),split=split))
    return seq


def test_reference_training_warmup_exact_resume_and_reload(tmp_path):
    args=dict(model_options=dict(latent_dim=8,processor_blocks=2),use_gpu=False,epochs=1,lr=.001)
    a=PyG.ByFVGNSequences(str(tmp_path/'a'),reference_sequences(),**args)
    assert a.config.model_options['formulation']=='reference'
    assert not a.model.warmup_complete
    a.Train(epochs=3);assert a.model.warmup_complete
    assert np.isfinite(a.history['train_loss']).all()
    b=PyG.ByCFDDataset(str(tmp_path/'a'));b.Train(epochs=1)
    b.SaveModel(str(tmp_path/'partial.pt'))
    c=PyG.ByCFDDataset(str(tmp_path/'a'));c.LoadModel(str(tmp_path/'partial.pt'),resume_training=True)
    c.Train(epochs=2,resume=True)
    for key,value in a.model.state_dict().items():torch.testing.assert_close(value,c.model.state_dict()[key],atol=0,rtol=0)
    first=a.PredictFVGN(0);second=c.PredictFVGN(0)
    for key in first:np.testing.assert_array_equal(first[key],second[key])
    # Legacy model names without a formulation are explicitly migrated, never
    # loaded into a numerically different updater with the same state keys.
    physical=PyG.ByFVGNSequences(str(tmp_path/'physical'),reference_sequences(),use_gpu=False,
        model_options=dict(formulation='physical',latent_dim=8,processor_blocks=1))
    physical.SaveModel(str(tmp_path/'old.pt'))
    state=torch.load(tmp_path/'old.pt',weights_only=True)
    state['config_fields']['model_options'].pop('formulation')
    torch.save(state,tmp_path/'old.pt')
    physical.LoadModel(str(tmp_path/'old.pt'))
    assert physical.config.model_options['formulation']=='physical'


def test_vertex_interpolation_matches_author_not_centroid_mean(tmp_path):
    seq=reference_sequences();p=seq[0]['positions'];cells=seq[0]['cells']
    for s in seq:
        s['velocity']=np.stack([np.c_[p[:,0]**2,p[:,1]]*(1+.01*t) for t in range(4)])
        s['pressure']=np.broadcast_to(np.arange(len(p),dtype=np.float32)**2,(4,len(p))).copy()
    r=PyG.ByFVGNSequences(str(tmp_path/'weighted'),seq,fieldLocation='vertex',use_gpu=False,
        model_options=dict(latent_dim=8,processor_blocks=1))
    m=r.data_list[0].fv_mesh
    expected=(seq[0]['velocity'][0,m['cells']]*m['vertex_weights'].numpy()[:,:,None]).sum(1)
    np.testing.assert_allclose(r._MSACurrentVelocity(r.data_list[0]),expected,atol=1e-6)
    assert not np.allclose(expected,seq[0]['velocity'][0,cells].mean(1))
    face=m['face_vertices'].numpy()
    pressure=seq[0]['pressure'][1,face].mean(1)
    np.testing.assert_allclose(r.data_list[0].fv_target_pressure,pressure,atol=1e-6)
    internal=m['face_type'].numpy()==0
    face_velocity=seq[0]['velocity'][1,face].mean(1)
    np.testing.assert_allclose(r.data_list[0].fv_target_velocity[internal],face_velocity[internal],atol=1e-6)


def test_reference_additional_features_and_loss_requirements(tmp_path):
    seq=reference_sequences()
    for s in seq:s['graph_features']={'speed':.2};s['extra_node_features']=np.arange(8).reshape(4,2)*.1
    r=PyG.ByFVGNSequences(str(tmp_path/'extra'),seq,use_gpu=False,model_options=dict(latent_dim=8,processor_blocks=1))
    # Add two custom adjacency fields beyond the adapter's four geometry columns.
    import pandas as pd
    path=tmp_path/'extra'/'edges.csv';edges=pd.read_csv(path);edges['feat_4']=.3;edges['feat_5']=.5;edges.to_csv(path,index=False)
    r=PyG.ByCFDDataset(str(tmp_path/'extra'));r.model.eval()
    d=r.data_list[0].clone();base=r.model(d)['prediction']
    modified=d.clone();modified.u+=1
    assert not torch.allclose(base,r.model(modified)['prediction'])
    modified=d.clone();modified.x[:,6:]+=1
    assert not torch.allclose(base,r.model(modified)['prediction'])
    modified=d.clone();modified.edge_attr[:,4:]+=1
    assert not torch.allclose(base,r.model(modified)['prediction'])
    missing=reference_sequences()
    for s in missing:s.pop('pressure')
    with pytest.raises(ValueError,match='pressure'):PyG.ByFVGNSequences(str(tmp_path/'missing'),missing)


@pytest.mark.parametrize('options',[{'latent_dim':7},{'normalizer_warmup_passes':-1},{'normalizer_accumulations':1},{'formulation':'other'},{'random_edge_flip':1},{'pressure_scale':1.}])
def test_reference_option_validation(tmp_path,options):
    with pytest.raises(ValueError):PyG.ByFVGNSequences(str(tmp_path/'bad'),reference_sequences(),model_options=options)

def test_reference_unlabelled_inference_needs_no_future_pressure(tmp_path):
    seq=reference_sequences()
    r=PyG.ByFVGNSequences(str(tmp_path/'train'),seq,use_gpu=False,epochs=1,
        model_options=dict(latent_dim=8,processor_blocks=1))
    r.Train();r.SaveModel(str(tmp_path/'reference.pt'))
    case=copy.deepcopy(seq[0]);case.pop('pressure');case['split']='test'
    case['velocity']=np.repeat(case['velocity'][:1],2,axis=0)
    infer=PyG.ByFVGNSequences(str(tmp_path/'new-case'),[case],inference=True,use_gpu=False,
        model_options=dict(latent_dim=8,processor_blocks=1))
    infer.LoadModel(str(tmp_path/'reference.pt'))
    assert not hasattr(infer.data_list[0],'fv_target_pressure')
    expected=r.PredictFVGN(0);actual=infer.PredictFVGN(0)
    for key in expected:np.testing.assert_array_equal(expected[key],actual[key])
    with pytest.raises(ValueError):infer.Train()
