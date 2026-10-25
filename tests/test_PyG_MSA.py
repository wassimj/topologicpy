"""MSA-GNN model, sparse hierarchy, CFD ingestion and rollout regressions."""
from __future__ import annotations
import copy
import json
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from topologicpy.PyG import PyG
from test_PyG import cpu_determinism


def grid(nx=4, ny=3):
    positions = np.asarray([[x,y,0.] for y in range(ny) for x in range(nx)],dtype=float)
    pairs = []
    for y in range(ny):
        for x in range(nx):
            a = y*nx+x
            if x+1<nx: pairs.extend([(a,a+1),(a+1,a)])
            if y+1<ny: pairs.extend([(a,a+nx),(a+nx,a)])
    return positions,np.asarray(pairs,dtype=np.int64).T


def trajectories(count=5, frames=7, dimensions=2):
    pos,index = grid()
    values = []
    for seed in range(count):
        tags = np.zeros(len(pos),dtype=int)
        tags[pos[:,0]==0] = 2
        velocity = np.zeros((frames,len(pos),dimensions))
        velocity[0,:,0] = .1 + .03*seed + .02*pos[:,0]
        velocity[:,:,1] = .01*pos[:,1]
        for step in range(frames-1):
            velocity[step+1] = .94*velocity[step]
            velocity[step+1,tags==2,0] = .2 + .03*seed
        values.append({"positions":pos.copy(),"edge_index":index.copy(),"velocity":velocity,
                       "times":np.arange(frames)*.1,"boundary_type":tags,
                       "weights":np.ones(len(pos)),"graph_features":{"inlet_speed":.2+.03*seed}})
    return values


def make_model(tmp_path, **kwargs):
    options = dict(latent_dim=16,processor_blocks=2,attention_heads=2,dropout=0.,n_clusters=3,max_coarse_nodes=8)
    options.update(kwargs.pop("model_options",{}))
    return PyG.ByCFDSequences(str(tmp_path / "cfd"),trajectories(),model_options=options,
                             split=(.6,.2,.2),epochs=2,lr=.003,use_gpu=False,**kwargs)


def test_sparse_hierarchy_is_deterministic_connected_and_permutation_equivariant():
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    pos,index = grid(8,5)
    a = PyG.BuildMSAHierarchy(pos,index,nClusters=6,maxClusters=8)
    b = PyG.BuildMSAHierarchy(pos,index,nClusters=6,maxClusters=8)
    for key in a: torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
    adjacency = coo_matrix((np.ones(index.shape[1]),index),shape=(len(pos),len(pos))).tocsr()
    for group in range(6):
        members = np.flatnonzero(a["cluster_index"].numpy()==group)
        assert connected_components(adjacency[members][:,members],directed=False)[0] == 1
    permutation = np.random.default_rng(9).permutation(len(pos))
    inverse = np.argsort(permutation)
    c = PyG.BuildMSAHierarchy(pos[permutation],inverse[index],nClusters=6,maxClusters=8)
    torch.testing.assert_close(a["cluster_index"][permutation],c["cluster_index"])


def test_hierarchy_respects_disconnected_regions_and_budget():
    pos,index = grid()
    positions = np.r_[pos,pos+np.asarray([.01,0,0])]
    edges = np.concatenate([index,index+len(pos)],axis=1)
    h = PyG.BuildMSAHierarchy(positions,edges,nClusters=4,maxClusters=8)
    assert set(h["cluster_index"][:len(pos)].tolist()).isdisjoint(h["cluster_index"][len(pos):].tolist())
    with pytest.raises(ValueError,match="disconnected"): PyG.BuildMSAHierarchy(positions,edges,nClusters=1)


@pytest.mark.parametrize("aggregation",["gru","mean"])
def test_msa_training_prediction_and_attention_gradients(tmp_path, aggregation):
    model = make_model(tmp_path,model_options={"aggregation":aggregation})
    assert model.config.model_name == "msa_gnn"
    assert len(model._msa_hierarchy_cache) == 1
    assert model._Loader(model.train_set,batch_size=8).batch_size == 1
    loss = model.Train()["train_loss"]
    assert np.isfinite(loss).all()
    assert np.isfinite(list(model.Test().values())).all()
    result = model.Predict(return_embeddings=True,return_logits=True)
    assert result["pred"][0].shape == (12,2)
    assert result["emb"][0].shape == (12,16)
    data = next(iter(model._Loader([model.data_list[0]]))).to(model.device)
    model.model.train(); model.optimizer.zero_grad()
    prediction,_ = model._Forward(data)
    model._loss_from_logits(prediction,data.y,"regression").backward()
    grad = model.model.processors[0].attention.in_proj_weight.grad
    assert grad is not None and grad.abs().sum() > 0


def test_checkpoint_resume_and_physical_rollout(tmp_path):
    model = make_model(tmp_path)
    model.Train()
    gid = model._identities[0]["graph_id"]
    original = copy.deepcopy(model.data_list[0].x)
    before = model.Rollout(gid,steps=4)
    assert before["velocity"].shape == (5,12,2)
    np.testing.assert_allclose(before["times"],[0,.1,.2,.3,.4],atol=1e-6)
    np.testing.assert_array_equal(before["velocity"][1:,before["boundary_mask"]],
                                  np.broadcast_to(before["velocity"][0,before["boundary_mask"]],(4,int(before["boundary_mask"].sum()),2)))
    torch.testing.assert_close(original,model.data_list[0].x)
    file = str(tmp_path / "msa.pt")
    model.SaveModel(file)
    restored = PyG.ByCFDDataset(str(tmp_path / "cfd"),use_gpu=False)
    restored.LoadModel(file,resume_training=True)
    after = restored.Rollout(gid,steps=4)
    np.testing.assert_array_equal(before["velocity"],after["velocity"])
    restored.Train(epochs=1,resume=True)
    assert restored._epochs_completed == 3


def test_rollout_dynamic_boundaries_conditions_and_reference_metrics(tmp_path):
    model = make_model(tmp_path)
    model.Train()
    gid = model._identities[0]["graph_id"]
    boundary = np.ones((3,12,2))*.7
    result = model.Rollout(gid,3,boundaryValues=boundary,graphFeatures=np.asarray([[.2],[.3],[.4]]))
    np.testing.assert_allclose(result["velocity"][1:,result["boundary_mask"]],.7,atol=1e-6)
    repeated = model.Rollout(gid,3,boundaryValues=boundary,graphFeatures=np.asarray([[.2],[.3],[.4]]),reference=result["velocity"])
    assert repeated["metrics"]["rmse"] == [0.,0.,0.]
    with pytest.raises(ValueError,match="interval"): model.Rollout(gid,2,timeStep=.2)
    with pytest.raises(ValueError): model.Rollout(gid,0)
    with pytest.raises(ValueError): model.Rollout(gid,3,boundaryValues=np.zeros((2,12,2)))


def test_multistep_training_and_noise_target_correction(tmp_path):
    model = make_model(tmp_path,model_options={"rollout_steps":3,"noise_std":.02})
    data = next(iter(model._Loader([model.train_set[0]])))
    truth_next = model._MSACurrentVelocity(data)+model._PhysicalTensor(data.y)
    perturbed = model._MSANoise(data)
    corrected = model._MSACurrentVelocity(perturbed)+model._PhysicalTensor(perturbed.y)
    torch.testing.assert_close(truth_next,corrected,atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(model._MSACurrentVelocity(perturbed)[perturbed.boundary_mask],
                               model._MSACurrentVelocity(data)[data.boundary_mask])
    assert np.isfinite(model.Train()["train_loss"]).all()


@pytest.mark.parametrize("options",[{"attention":False},{"multiscale":False}])
def test_ablation_modes_run(tmp_path,options):
    model = make_model(tmp_path,model_options=options)
    assert np.isfinite(model.Train()["train_loss"]).all()
    assert np.isfinite(model.Rollout(0,2)["velocity"]).all()


def test_npz_adapter_and_safe_overwrite(tmp_path):
    seq = trajectories(count=1)[0]
    seq["graph_features"] = np.ones((7,1))
    file = tmp_path / "flow.npz"
    np.savez(file,**seq)
    imported = PyG.ReadCFDSequence(str(file))
    np.testing.assert_array_equal(imported["velocity"],seq["velocity"])
    model = PyG.ByCFDSequences(str(tmp_path / "data"),[str(file)],model_options={"latent_dim":8,"processor_blocks":1,"attention_heads":2},epochs=1,use_gpu=False)
    assert np.isfinite(model.Train()["train_loss"]).all()
    marker = tmp_path / "data" / "keep.txt"; marker.write_text("keep")
    with pytest.raises(FileExistsError): PyG.ByCFDSequences(str(tmp_path / "data"),[str(file)])
    PyG.ByCFDSequences(str(tmp_path / "data"),[str(file)],overwrite=True,model_name="default",use_gpu=False)
    assert marker.read_text() == "keep"


def test_vtu_point_and_cell_import(tmp_path):
    import meshio
    points = np.asarray([[0.,0,0],[1,0,0],[0,1,0],[1,1,0]])
    cells = [("triangle",np.asarray([[0,1,2],[1,3,2]]))]
    files = []
    for i in range(3):
        path = tmp_path / f"flow{i}.vtu"
        meshio.write(path,meshio.Mesh(points,cells,point_data={"U":np.ones((4,2))*i,"type":np.asarray([2,0,1,3])},
                     cell_data={"U":[np.ones((2,2))*i],"volume":[np.asarray([.5,.5])]}))
        files.append(str(path))
    point = PyG.ReadCFDSequence(files,boundaryKey="type")
    assert point["velocity"].shape == (3,4,2)
    assert point["boundary_type"].tolist() == [2,0,1,3]
    cell = PyG.ReadCFDSequence(files,location="cell",weightKey="volume")
    assert cell["positions"].shape == (2,3)
    assert cell["edge_index"].shape == (2,2)
    assert cell["velocity"].shape == (3,2,2)
    np.testing.assert_allclose(cell["weights"],[.5,.5])
    meshio.write(tmp_path / "changed.vtu",meshio.Mesh(points+1,cells,point_data={"U":np.zeros((4,2))}))
    with pytest.raises(ValueError,match="mesh/order changed"): PyG.ReadCFDSequence([files[0],str(tmp_path / "changed.vtu")])


@pytest.mark.parametrize("options",[{"latent_dim":7,"attention_heads":2},{"processor_blocks":0},{"aggregation":"bad"},{"noise_std":-1}])
def test_invalid_model_options(tmp_path,options):
    with pytest.raises(ValueError): make_model(tmp_path,model_options=options)


def test_msa_learns_and_three_component_fields_run(tmp_path):
    model=make_model(tmp_path,model_options={"aggregation":"mean"})
    losses=model.Train(epochs=8)["train_loss"]
    assert losses[-1] < .8*losses[0]
    seq=trajectories(count=3,dimensions=3)
    other=PyG.ByCFDSequences(str(tmp_path/'three_d'),seq,epochs=1,use_gpu=False,
                            model_options={"latent_dim":8,"processor_blocks":1,"attention_heads":2,"max_coarse_nodes":8})
    assert np.isfinite(other.Train()["train_loss"]).all()
    assert other.Rollout(0,2)["velocity"].shape == (3,12,3)


def test_noise_multistep_resume_matches_uninterrupted(tmp_path):
    model=make_model(tmp_path,model_options={"rollout_steps":2,"noise_std":.01,"dropout":.1})
    model.Train(epochs=3)
    other=PyG.ByCFDDataset(str(tmp_path/'cfd'))
    other.Train(epochs=1)
    file=str(tmp_path/'partial.pt'); other.SaveModel(file)
    resumed=PyG.ByCFDDataset(str(tmp_path/'cfd'))
    resumed.LoadModel(file,resume_training=True); resumed.Train(epochs=2,resume=True)
    for key,value in model.model.state_dict().items():
        torch.testing.assert_close(value,resumed.model.state_dict()[key],atol=0,rtol=0)


def test_model_permutation_equivariance_with_canonical_gru_order(tmp_path):
    model=make_model(tmp_path)
    model.model.eval()
    data=model.data_list[0].clone()
    permutation=torch.tensor(np.random.default_rng(9).permutation(data.num_nodes))
    inverse=permutation.argsort()
    shuffled=data.clone()
    shuffled.x=data.x[permutation]; shuffled.pos=data.pos[permutation]
    shuffled.edge_index=inverse[data.edge_index]
    shuffled.hierarchy=PyG.BuildMSAHierarchy(shuffled.pos,shuffled.edge_index,nClusters=3,maxClusters=8)
    with torch.no_grad():
        original=model.model(data)["prediction"]
        changed=model.model(shuffled)["prediction"]
    torch.testing.assert_close(original[permutation],changed,atol=1e-5,rtol=1e-5)


def test_bad_hierarchy_mesh_and_schema_are_rejected(tmp_path):
    seq=trajectories(count=1)[0]
    bad=copy.deepcopy(seq); bad['times'][2]=bad['times'][1]
    with pytest.raises(ValueError): PyG.ByCFDSequences(str(tmp_path/'bad'),[bad])
    bad=copy.deepcopy(seq); bad['graph_features']={'different_name':1.}
    with pytest.raises(ValueError,match='schemas'): PyG.ByCFDSequences(str(tmp_path/'bad_schema'),[seq,bad])
    model=make_model(tmp_path)
    data=model.data_list[0].clone(); data.hierarchy=copy.deepcopy(data.hierarchy)
    data.hierarchy['member_index'][:]=0
    with pytest.raises(ValueError): model._ValidateMSAHierarchy(data)
    data=model.data_list[0].clone()
    data.pos=data.pos[:,:1]
    with pytest.raises(ValueError): PyG.BuildMSAHierarchy(data.pos,data.edge_index)


def test_hierarchy_settings_update_rebuilds_cache(tmp_path):
    model=make_model(tmp_path)
    assert len(model.data_list[0].hierarchy['coarse_pos']) == 3
    opts=dict(model.config.model_options); opts['n_clusters']=4
    model.SetHyperparameters(model_options=opts)
    assert len(model.data_list[0].hierarchy['coarse_pos']) == 4


def test_packed_gru_matches_individual_group_evaluation(tmp_path):
    runner=make_model(tmp_path)
    model=runner.model; model.eval()
    h=runner.data_list[0].hierarchy
    nodes=torch.randn(12,16)
    with torch.no_grad():
        packed=model._aggregate(nodes,h)
        expected=[]
        ptr=h['group_ptr'].tolist()
        for i in range(len(ptr)-1):
            sequence=nodes[h['member_index'][ptr[i]:ptr[i+1]]][None]
            _,state=model.group_gru(sequence)
            expected.append(state[0,0])
        expected=model.group_projection(torch.stack(expected))
    torch.testing.assert_close(packed,expected,atol=1e-6,rtol=1e-5)


def test_vtu_hexahedral_cell_adjacency(tmp_path):
    import meshio
    points=np.asarray([[x,y,z] for z in (0.,1.) for y in (0.,1.) for x in (0.,1.,2.)])
    cells=np.asarray([[0,1,4,3,6,7,10,9],[1,2,5,4,7,8,11,10]])
    files=[]
    for i in range(2):
        file=tmp_path/f'hex{i}.vtu'
        meshio.write(file,meshio.Mesh(points,[('hexahedron',cells)],cell_data={'U':[np.ones((2,3))*i]}))
        files.append(str(file))
    data=PyG.ReadCFDSequence(files,location='cell')
    np.testing.assert_allclose(data['positions'],[[.5,.5,.5],[1.5,.5,.5]])
    assert data['edge_index'].tolist() == [[0,1],[1,0]]
    assert data['velocity'].shape == (2,2,3)


def test_external_hierarchy_validation_is_transactional(tmp_path):
    model=make_model(tmp_path)
    original=model.data_list[0].hierarchy
    bad=copy.deepcopy(original); bad['member_index'][:]=0
    with pytest.raises(ValueError): model.SetGraphData(0,hierarchy=bad)
    assert model.data_list[0].hierarchy is original
    model.SetGraphData(0,hierarchy=copy.deepcopy(original))
    assert np.isfinite(model.Rollout(0,1)['velocity']).all()


def test_export_does_not_clobber_unrelated_temp_file_and_resume_detects_edits(tmp_path):
    model=make_model(tmp_path)
    marker=tmp_path/'cfd'/'graphs.csv.tmp'; marker.write_text('keep')
    PyG.ByCFDSequences(str(tmp_path/'cfd'),trajectories(),overwrite=True,model_name='default',use_gpu=False)
    assert marker.read_text() == 'keep'
    model.Train(epochs=1)
    model.SetGraphData(0,node_weight=torch.ones(12)*2)
    file=str(tmp_path/'changed.pt'); model.SaveModel(file)
    other=PyG.ByCFDDataset(str(tmp_path/'cfd'))
    with pytest.raises(ValueError,match='different dataset'): other.LoadModel(file,resume_training=True)
