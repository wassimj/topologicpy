"""Official CFD record transport, split isolation and supervision regressions."""
import copy,json,struct
import numpy as np
import pytest
pytest.importorskip('torch_geometric')
from topologicpy.PyG import PyG
from test_PyG_MSA import trajectories
from test_PyG import cpu_determinism

def varint(v):
 out=bytearray()
 while v>127:out.append((v&127)|128);v>>=7
 out.append(v);return bytes(out)
def field(n,v):return varint(n*8+2)+varint(len(v))+v

def record():
 arrays={'cells':np.array([[[0,1,2],[1,3,2]]],dtype='<i4'),
         'mesh_pos':np.array([[[0,0],[1,0],[0,1],[1,1]]],dtype='<f4'),
         'node_type':np.array([[[0],[4],[5],[6]]],dtype='<i4'),
         'velocity':np.arange(24,dtype='<f4').reshape(3,4,2),
         'pressure':np.arange(12,dtype='<f4').reshape(3,4,1)}
 meta={'dt':.01,'features':{k:{'dtype':str(v.dtype),'shape':[len(v),-1,v.shape[-1]]} for k,v in arrays.items()}}
 payload=field(1,b''.join(field(1,field(1,k.encode())+field(2,field(1,field(1,v.tobytes())))) for k,v in arrays.items()))
 length=struct.pack('<Q',len(payload))
 return length+struct.pack('<I',PyG._TFCRC(length))+payload+struct.pack('<I',PyG._TFCRC(payload)),meta

def test_record_reader_geometry_labels_and_crc(tmp_path):
 data,meta=record();p=tmp_path/'data.tfrecord';p.write_bytes(data*2)
 values=list(PyG.ReadMeshGraphNets(p,meta))
 assert len(values)==2
 s=values[0];assert s['velocity'].shape==(3,4,2)
 assert 'pressure' not in s
 pressure=next(PyG.ReadMeshGraphNets(p,meta,includePressure=True))['pressure']
 np.testing.assert_array_equal(pressure,np.arange(12).reshape(3,4,1))
 assert s['boundary_type'].tolist()==[0,2,3,1]
 assert s['supervision_mask'].tolist()==[True,False,True,False]
 assert s['edge_index'].shape==(2,10)
 assert s['weights'].sum()==pytest.approx(1.)
 assert s['times'].tolist()==[0.,.01,.02]
 assert len(list(PyG.ReadMeshGraphNets(p,meta,limit=1)))==1
 corrupted=bytearray(data);corrupted[-7]^=1;p.write_bytes(corrupted)
 with pytest.raises(ValueError,match='CRC'):list(PyG.ReadMeshGraphNets(p,meta))
 p.write_bytes(data[:-2])
 with pytest.raises(ValueError,match='Truncated'):list(PyG.ReadMeshGraphNets(p,meta))

def test_crc_known_vector_and_malformed_protobuf():
 # CRC32C('123456789') = e3069283, masked by the TFRecord rotation/offset.
 crc=0xe3069283
 assert PyG._TFCRC(b'123456789')==(((crc>>15)|(crc<<17))+0xa282ead8)&0xffffffff
 with pytest.raises(ValueError):list(PyG._TFFields(b'\x0a\x10abc'))
 with pytest.raises(ValueError):PyG._TFVarint(b'\x80',0)

def test_bounded_download_ranges_and_cache(tmp_path,monkeypatch):
 from urllib import request
 data,meta=record();calls=[]
 class Response:
  def __init__(self,body,status=200,headers=None):self.body=body;self.status=status;self.headers=headers or {}
  def __enter__(self):return self
  def __exit__(self,*args):pass
  def read(self,n=-1):return self.body if n<0 else self.body[:n]
 def urlopen(req,timeout):
  if isinstance(req,str):return Response(json.dumps(meta).encode())
  span=req.get_header('Range');calls.append(span)
  start,end=map(int,span.removeprefix('bytes=').split('-'))
  return Response((data*3)[start:end+1],206,{'Content-Range':f'bytes {start}-{end}/{len(data)*3}'})
 monkeypatch.setattr(request,'urlopen',urlopen)
 result=PyG.DownloadMeshGraphNets(tmp_path,counts={'train':2})
 assert len(calls)==4
 assert len(list(PyG.ReadMeshGraphNets(result['splits']['train'],meta)))==2
 manifest=json.loads((tmp_path/'train.first-2.tfrecord.json').read_text())
 assert manifest['record_indices']==[0,1] and manifest['full_split'] is False
 PyG.DownloadMeshGraphNets(tmp_path,counts={'train':2});assert len(calls)==4

def test_official_splits_masks_and_checkpoint(tmp_path):
 seqs=trajectories(count=3,frames=3)
 for i,(s,split) in enumerate(zip(seqs,['train','valid','test'])):
  s.update(split=split,simulation_id=f'{split}:0',frame_offset=201,
           supervision_mask=np.arange(len(s['positions']))%2==0)
 model=PyG.ByCFDSequences(tmp_path/'dataset',seqs,use_gpu=False,epochs=1,
      model_options=dict(latent_dim=8,processor_blocks=1,attention_heads=2,n_clusters=3,max_coarse_nodes=8))
 assert len(model.train_set)==len(model.val_set)==len(model.test_set)==2
 assert set(model.graph_df.frame_index)=={201,202}
 for split in ['train','val','test']:
  for d in model._DataForSplit(split):
   assert int(getattr(d,split+'_mask').sum())==6
   assert getattr(d,split+'_mask').equal(d.supervision_mask)
 model.Train();model.SaveModel(str(tmp_path/'model.pt'));model.LoadModel(str(tmp_path/'model.pt'))
 with pytest.raises(ValueError,match='split'):model.CrossValidate()
 # A simulation appearing in two official splits is invalid.
 seqs[1]['simulation_id']=seqs[0]['simulation_id']
 with pytest.raises(ValueError):PyG.ByCFDSequences(tmp_path/'bad',seqs,use_gpu=False)



def test_edge_identity_does_not_round_large_integer_ids(tmp_path):
 import pandas as pd
 seqs=trajectories(count=3,frames=2)
 for s,split in zip(seqs,['train','valid','test']):s['split']=split
 folder=tmp_path/'large-ids'
 PyG.ByCFDSequences(folder,seqs,use_gpu=False,ontology=False,
      model_options=dict(latent_dim=8,processor_blocks=1,attention_heads=2,n_clusters=3,max_coarse_nodes=8))
 nodes=pd.read_csv(folder/'nodes.csv');edges=pd.read_csv(folder/'edges.csv')
 offset=2**54
 nodes.node_id+=offset;edges.src_id+=offset;edges.dst_id+=offset
 nodes.to_csv(folder/'nodes.csv',index=False);edges.to_csv(folder/'edges.csv',index=False)
 model=PyG.ByCFDDataset(str(folder),use_gpu=False)
 assert model._identities[0]['node_ids']==list(range(offset,offset+12))
 assert all(isinstance(a,int) and isinstance(b,int) for a,b in model._identities[0]['edge_endpoints'])
 assert set(model._identities[0]['edge_endpoints'])=={(int(a),int(b)) for a,b in zip(edges[edges.graph_id==0].src_id,edges[edges.graph_id==0].dst_id)}
