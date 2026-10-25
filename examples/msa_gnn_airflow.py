"""Runnable MSA-GNN pipeline demonstration on SYNTHETIC, non-CFD flow fields.

python examples/msa_gnn_airflow.py --output demo_output --epochs 8 --compare

For real CFD, replace synthetic_sequences with ReadCFDSequence/ByCFDPaths or
fixed-mesh dictionaries. The synthetic dynamics are not Navier-Stokes validation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from topologicpy.PyG import PyG


def synthetic_sequences(count=6, frames=10, nx=8, ny=5):
    positions = np.asarray([[x,y,0.] for y in range(ny) for x in range(nx)],dtype=float)
    pairs = []
    for y in range(ny):
        for x in range(nx):
            i=y*nx+x
            if x+1<nx: pairs.extend([(i,i+1),(i+1,i)])
            if y+1<ny: pairs.extend([(i,i+nx),(i+nx,i)])
    edge_index=np.asarray(pairs,dtype=np.int64).T
    src,dst=edge_index
    degree=np.bincount(dst,minlength=len(positions)).clip(1)
    tags=np.zeros(len(positions),dtype=int)
    tags[(positions[:,1]==0)|(positions[:,1]==ny-1)]=1
    tags[(positions[:,0]==0)&(tags==0)]=2
    tags[(positions[:,0]==nx-1)&(tags==0)]=3
    sequences=[]
    for simulation in range(count):
        inlet=.2+.025*simulation+.02*np.sin(np.arange(frames)*.3)
        velocity=np.zeros((frames,len(positions),2))
        velocity[0,:,0]=(.08+.01*simulation)*(1-positions[:,0]/nx)
        velocity[0,:,1]=.03*np.sin(np.pi*positions[:,0]/(nx-1))*np.sin(np.pi*positions[:,1]/(ny-1))
        velocity[0,tags==1]=0
        velocity[0,tags==2,0]=inlet[0]
        for step in range(frames-1):
            mean=np.zeros_like(velocity[step]); np.add.at(mean,dst,velocity[step,src])
            velocity[step+1]=.9*velocity[step]+.08*mean/degree[:,None]
            velocity[step+1,tags==1]=0
            velocity[step+1,tags==2,0]=inlet[step+1]
            velocity[step+1,tags==2,1]=0
        sequences.append(dict(positions=positions.copy(),edge_index=edge_index.copy(),velocity=velocity,
                              boundary_type=tags.copy(),weights=np.ones(len(positions)),times=np.arange(frames)*.1,
                              graph_features=inlet[:,None]))
    return sequences


def write_preview(path,positions,reference,prediction,times):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    maximum=float(max(np.linalg.norm(reference,axis=-1).max(),np.linalg.norm(prediction,axis=-1).max()))
    def trace(values,name):
        return go.Scatter(x=positions[:,0],y=positions[:,1],mode='markers',name=name,
                          marker=dict(size=12,color=np.linalg.norm(values,axis=-1),coloraxis='coloraxis'))
    fig=make_subplots(rows=1,cols=2,subplot_titles=['Synthetic reference','MSA-GNN rollout'])
    fig.add_trace(trace(reference[0],'Reference'),row=1,col=1)
    fig.add_trace(trace(prediction[0],'MSA-GNN'),row=1,col=2)
    fig.frames=[go.Frame(name=str(i),data=[trace(reference[i],'Reference'),trace(prediction[i],'MSA-GNN')]) for i in range(len(times))]
    fig.update_layout(title='Synthetic flow demonstration — not CFD validation',
                      coloraxis=dict(colorscale='Viridis',cmin=0,cmax=maximum,colorbar=dict(title='Speed (m/s)')),
                      sliders=[dict(steps=[dict(method='animate',args=[[str(i)],dict(mode='immediate',frame=dict(duration=0,redraw=True))],label=f'{t:.1f}s') for i,t in enumerate(times)])])
    fig.update_yaxes(scaleanchor='x',scaleratio=1,row=1,col=1)
    fig.update_yaxes(scaleanchor='x2',scaleratio=1,row=1,col=2)
    fig.write_html(str(path),include_plotlyjs=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=8)
    parser.add_argument('--compare',action='store_true')
    parser.add_argument('--training-rollout-steps',type=int,default=1)
    args=parser.parse_args()
    import torch
    torch.set_num_threads(1)
    if args.compare and args.training_rollout_steps != 1:
        parser.error('--compare uses shared one-step training; run multi-step experiments separately')
    args.output.mkdir(parents=True,exist_ok=True)
    sequences=synthetic_sequences()
    options=dict(latent_dim=32,processor_blocks=2,attention_heads=2,max_coarse_nodes=16,
                 coarsening_ratio=.15,aggregation='gru',dropout=.05,noise_std=0.,rollout_steps=args.training_rollout_steps)
    model=PyG.ByCFDSequences(str(args.output/'dataset'),sequences,overwrite=True,model_options=options,
                            epochs=args.epochs,lr=.003,use_gpu=False,random_state=42)
    candidates=[('msa_gnn',model)]
    if args.compare:
        candidates.insert(0,('feature_gnn',PyG.ByCFDDataset(str(args.output/'dataset'),model_name='default',
                          hidden_dims=(32,32),dropout=.05,epochs=args.epochs,lr=.003,use_gpu=False)))
        no_attention=dict(model.config.model_options); no_attention['attention']=False
        candidates.insert(1,('multiscale_without_attention',PyG.ByCFDDataset(str(args.output/'dataset'),
                          model_options=no_attention,epochs=args.epochs,lr=.003,use_gpu=False)))
    summary={'data':'synthetic discrete diffusion-like dynamics; not CFD validation','models':{}}
    for name,runner in candidates:
        start=perf_counter(); history=runner.Train(); training_time=perf_counter()-start
        first=runner.test_set[0]
        index=int(first.topologicpy_graph_index)
        row=runner.graph_df.iloc[index]
        simulation,frame=int(row.simulation_id),int(row.frame_index)
        sequence=sequences[simulation]
        steps=min(8,len(sequence['velocity'])-frame-1)
        reference=sequence['velocity'][frame:frame+steps+1]
        start=perf_counter()
        result=runner.Rollout(runner._identities[index]['graph_id'],steps,
                              boundaryValues=reference[1:],graphFeatures=sequence['graph_features'][frame:frame+steps],reference=reference)
        elapsed=perf_counter()-start
        summary['models'][name]=dict(increment_test_metrics=runner.Test(),initial_training_loss=history['train_loss'][0],
                                    final_training_loss=history['train_loss'][-1],training_seconds=training_time,
                                    rollout_seconds=elapsed,rollout_final_rmse=result['metrics']['rmse'][-1],
                                    rollout_rmse=result['metrics']['rmse'])
        if name=='msa_gnn':
            runner.SaveModel(str(args.output/'msa_demo.pt'))
            np.savez_compressed(args.output/'rollout.npz',velocity=result['velocity'],reference=reference,
                                positions=sequence['positions'],times=result['times'],boundary_mask=result['boundary_mask'])
            write_preview(args.output/'rollout.html',sequence['positions'],reference,result['velocity'],result['times'])
    (args.output/'metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
