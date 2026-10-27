"""Bounded, official-split MeshGraphNets cylinder-flow benchmark (not full reproduction)."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from topologicpy.PyG import PyG

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--epochs',type=int,default=12);p.add_argument('--steps',type=int,default=200)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
 files=PyG.DownloadMeshGraphNets(a.data,counts={'train':8,'valid':2,'test':2})
 raw={k:list(PyG.ReadMeshGraphNets(v,files['meta'])) for k,v in files['splits'].items()}
 windows=[]
 for split,seqs in raw.items():
  starts=(1,201,401) if split=='train' else ((1,201) if split=='valid' else (1,))
  for i,s in enumerate(seqs):
   for start in starts:
    w=dict(s);w.update(split=split,simulation_id=f'{split}:{i}',frame_offset=start,
                         velocity=s['velocity'][start:start+17],times=s['times'][start:start+17])
    windows.append(w)
 options=dict(latent_dim=64,processor_blocks=2,attention_heads=2,max_coarse_nodes=24,
              coarsening_ratio=.02,aggregation='gru',dropout=0.,noise_std=0.,rollout_steps=1,
              velocity_feature_columns=['feat_0','feat_1'],time_steps=[.01])
 t=time.perf_counter();print('Building dataset',flush=True)
 kwargs=dict(model_options=options,epochs=1,lr=.001,use_gpu=True,random_state=42,
             gradient_clip_norm=1.,early_stopping=False,ontology=False)
 if (a.output/'dataset'/'msa_manifest.json').exists():
  model=PyG.ByCFDDataset(str(a.output/'dataset'),**kwargs)
 else:
  model=PyG.ByCFDSequences(str(a.output/'dataset'),windows,overwrite=True,**kwargs)
 print('Built',model.device,len(model.train_set),len(model.val_set),len(model.test_set),time.perf_counter()-t,flush=True)
 def graph_id(split,i=0):
  df=model.graph_df
  row=df[(df.simulation_id==f'{split}:{i}')&(df.frame_index==1)].iloc[0]
  return int(row.graph_id)
 progress=[];best=float('inf');start=time.perf_counter()
 for epoch in range(a.epochs):
  t=time.perf_counter();h=model.Train(epochs=1,resume=epoch>0)
  ref=raw['valid'][0]['velocity'][1:22]
  pred=model.Rollout(graph_id('valid'),20)['velocity']
  score=float(np.mean((pred[1:]-ref[1:])**2))
  entry=dict(epoch=epoch+1,train_loss=float(h['train_loss'][-1]),val_loss=float(h['val_loss'][-1]),
             validation_rollout20_mse=score,seconds=time.perf_counter()-t)
  progress.append(entry);print(json.dumps(entry),flush=True)
  (a.output/'progress.json').write_text(json.dumps(progress,indent=2))
  if np.isfinite(score) and score<best:
   best=score;model.SaveModel(str(a.output/'best.pt'));best_epoch=epoch+1
 training_seconds=time.perf_counter()-start
 model.LoadModel(str(a.output/'best.pt'))
 results=[]
 for i,s in enumerate(raw['test']):
  ref=s['velocity'][1:a.steps+2];t=time.perf_counter()
  pred=model.Rollout(graph_id('test',i),a.steps)['velocity'];elapsed=time.perf_counter()-t
  persistence=np.broadcast_to(ref[0],ref.shape)
  mse=np.mean((pred-ref)**2,axis=(1,2));base=np.mean((persistence-ref)**2,axis=(1,2))
  horizons={str(h):dict(msa_mse=float(mse[1:h+1].mean()),persistence_mse=float(base[1:h+1].mean()))
            for h in (1,10,20,50,100,200) if h<=a.steps}
  result=dict(trajectory=i,nodes=len(s['positions']),edges=s['edge_index'].shape[1],horizons=horizons,
              final_rmse=float(np.sqrt(mse[-1])),persistence_final_rmse=float(np.sqrt(base[-1])),
              rollout_seconds=elapsed,all_finite=bool(np.isfinite(pred).all()),
              boundary_max_error=float(np.max(np.abs(pred[:,np.isin(s['boundary_type'],[1,2])]-ref[0,np.isin(s['boundary_type'],[1,2])]))))
  results.append(result);print(json.dumps(result),flush=True)
  np.savez_compressed(a.output/f'test-{i}-rollout.npz',prediction=pred,reference=ref,positions=s['positions'],
                       cells=s['cells'],mse=mse,persistence_mse=base,times=s['times'][1:a.steps+2])
  if i==0:
   from msa_gnn_airflow import write_preview
   write_preview(a.output/'rollout.html',s['positions'],ref[::10],pred[::10],s['times'][1:a.steps+2:10])
   text=(a.output/'rollout.html').read_text(encoding='utf-8').replace('Synthetic reference','CFD reference').replace('Synthetic flow demonstration — not CFD validation','Official cylinder-flow subset: CFD reference and MSA-GNN')
   (a.output/'rollout.html').write_text(text,encoding='utf-8')
 summary=dict(dataset='Official MeshGraphNets cylinder_flow prefix subset',counts={k:len(v) for k,v in raw.items()},
    windows={k:sum(w['split']==k for w in windows) for k in raw},window_transitions=16,
    training_frames=[1,201,401],validation_frames=[1,201],test_initial_frame=1,dt=.01,
    options=options,epochs=a.epochs,best_epoch=best_epoch,checkpoint_selection='validation trajectory 0 cumulative 20-step MSE',
    device=str(model.device),gpu=torch.cuda.get_device_name(0),torch=torch.__version__,seed=42,
    training_seconds=training_seconds,history=progress,test=results,
    provenance={k:json.loads(Path(str(v)+'.json').read_text()) for k,v in files['splits'].items()},
    metric='Mean squared velocity component error over all nodes and predicted frames 1 through horizon, excluding initial state',
    limitations=['8/2/2 first trajectories, not random or full splits','Three short training windows per trajectory',
                 'Compact model and short training; not paper-scale accuracy reproduction','Pressure is not predicted'])
 (a.output/'metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
 print('DONE',flush=True)
if __name__=='__main__':main()


