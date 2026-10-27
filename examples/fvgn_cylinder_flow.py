"""Small FVGN verification on official MeshGraphNets cylinder-flow data.

python examples/fvgn_cylinder_flow.py --data cylinder-cache --output fvgn-results
Point-based CFD fields are linearly interpolated to triangles and faces; this is
an implementation check, not a reproduction of FVGN's full published benchmark.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from topologicpy.PyG import PyG


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=12)
    parser.add_argument('--steps',type=int,default=100)
    parser.add_argument('--latent-dim',type=int,default=32)
    parser.add_argument('--processor-blocks',type=int,default=2)
    parser.add_argument('--cpu',action='store_true')
    parser.add_argument('--formulation',choices=['reference','physical'],default='reference')
    parser.add_argument('--pressure',action='store_true',help='Supervise physical face pressure as well as velocity')
    args=parser.parse_args()
    if args.formulation=='reference': args.pressure=True
    torch.set_num_threads(2)
    args.output.mkdir(parents=True,exist_ok=True)
    records={};files={}
    for split,count in [('train',2),('valid',1),('test',1)]:
        candidates=sorted(args.data.glob(f'{split}.first-*.tfrecord'))
        candidates=[p for p in candidates if int(p.name.split('first-')[1].split('.')[0])>=count]
        if candidates:
            path=candidates[0]
        else:
            path=Path(PyG.DownloadMeshGraphNets(str(args.data),{split:count})['splits'][split])
        files[split]=str(path.resolve())
        records[split]=list(PyG.ReadMeshGraphNets(path,args.data/'meta.json',limit=count,includePressure=args.pressure))
    sequences=[]
    for split,values in records.items():
        for i,s in enumerate(values):
            mesh=PyG.BuildFVGNMesh(s['positions'],s['cells'])
            faces=mesh['face_vertices'].numpy()
            tags=mesh['face_type'].numpy().copy()
            types=s['original_node_type'][faces]
            exterior=mesh['neighbour'].numpy()<0
            tags[exterior & (types==4).all(1)]=2
            tags[exterior & (types==5).all(1)]=3
            # Match the original converter's inlet/outlet corner treatment.
            midpoint=mesh['midpoint'].numpy()
            tags[exterior & np.isin(types,(4,6)).all(1) & (types==4).any(1) & np.isclose(midpoint[:,0],midpoint[:,0].min(),rtol=0,atol=1e-4)]=2
            tags[exterior & np.isin(types,(5,6)).all(1) & (types==5).any(1) & np.isclose(midpoint[:,0],midpoint[:,0].max(),rtol=0,atol=1e-4)]=3
            v=s['velocity'][1:10]
            face_v=v[:,faces].mean(2)
            # Prescribed wall faces are stationary, including inlet/wall corners.
            face_v[:,tags==1]=0
            extra={}
            if args.pressure:extra['face_pressure']=s['pressure'][1:10,faces,0].mean(2)
            sequences.append(dict(positions=s['positions'],cells=s['cells'],velocity=v,
                face_velocity=face_v,face_types=tags,times=s['times'][1:10],
                split=split,simulation_id=f'{split}:{i}',frame_offset=1,
                density=1.,kinematic_viscosity=.001,**extra))
    options=dict(formulation=args.formulation,latent_dim=args.latent_dim,processor_blocks=args.processor_blocks,
                 continuity_loss_weight=0. if args.formulation=='reference' else 1.,noise_std=.02 if args.formulation=='reference' else .001)
    if args.pressure:
        training_pressure=np.concatenate([v['face_pressure'].reshape(-1) for v in sequences if v['split']=='train'])
        options.update(pressure_loss_weight=1.)
        if args.formulation=='physical':options['pressure_scale']=max(float(training_pressure.std()),1e-3)
    start=perf_counter()
    model=PyG.ByFVGNSequences(str(args.output/'dataset'),sequences,fieldLocation='vertex',overwrite=True,
            model_options=options,epochs=1,lr=.001,gradient_clip_norm=1.,ontology=False,
            early_stopping=False,use_gpu=not args.cpu,random_state=42)
    print('Dataset ready',len(model.train_set),len(model.val_set),len(model.test_set),str(model.device),flush=True)
    history=[];best=float('inf');best_epoch=None
    for epoch in range(args.epochs):
        h=model.Train(epochs=1,resume=epoch>0)
        row=dict(epoch=epoch+1,train_loss=float(h['train_loss'][-1]),val_loss=float(h['val_loss'][-1]))
        history.append(row);print(json.dumps(row),flush=True)
        (args.output/'progress.json').write_text(json.dumps(history,indent=2))
        if row['val_loss']<best:
            best=row['val_loss'];best_epoch=epoch+1;model.SaveModel(str(args.output/'best.pt'))
    training_seconds=perf_counter()-start
    model.LoadModel(str(args.output/'best.pt'))
    graph=int(model.graph_df[(model.graph_df.simulation_id=='test:0')&(model.graph_df.frame_index==1)].iloc[0].graph_id)
    fields=model.PredictFVGN(graph)
    source=records['test'][0]
    mesh=model.test_set[0].fv_mesh
    cells=mesh['cells'].cpu().numpy()
    requested=args.steps;failures=[];result=None
    for horizon in sorted({requested,*[h for h in (1,5,10,20,50) if h<requested]},reverse=True):
        if args.formulation=='reference':
            reference=(source['velocity'][1:horizon+2,cells]*mesh['vertex_weights'].cpu().numpy()[None,:,:,None]).sum(2)
        else:
            reference=source['velocity'][1:horizon+2,cells].mean(2)
        begin=perf_counter()
        try:
            result=model.Rollout(graph,horizon,reference=reference)
            elapsed=perf_counter()-begin
            break
        except FloatingPointError as error:
            failures.append(dict(steps=horizon,error=str(error)));print(failures[-1],flush=True)
    if result is None:raise FloatingPointError('FVGN failed even a one-step rollout')
    pred=result['velocity'];error=((pred-reference)**2).mean((1,2))
    persistence=((reference-reference[0])**2).mean((1,2))
    np.savez_compressed(args.output/'rollout.npz',prediction=pred,reference=reference,
                        positions=source['positions'],cells=cells,times=result['times'],
                        face_pressure=result['face_pressure'],divergence=result['divergence'],
                        mse=error,persistence_mse=persistence)
    model.SaveModel(str(args.output/'verified.pt'))
    model.LoadModel(str(args.output/'verified.pt'))
    reloaded=model.PredictFVGN(graph)
    delta=max(float(np.max(np.abs(fields[k]-reloaded[k]))) for k in fields)
    # CUDA index_add reductions are not bit-identical. CPU exact reload is
    # separately covered by the regression suite; record GPU differences here.
    reload_close=all(np.allclose(fields[k],reloaded[k],atol=1e-4,rtol=1e-4) for k in fields)
    print('Reload max absolute difference',delta,'within tolerance',reload_close,flush=True)
    summary=dict(dataset='Official MeshGraphNets cylinder_flow prefix subset',used_trajectories={'train':2,'valid':1,'test':1},
                 source_files=files,training_transition_frames=list(range(1,9)),interpolation='squared-distance vertex-to-cell weights; endpoint means on faces' if args.formulation=='reference' else 'vertex U to triangle means; face U to endpoint means',
                 target_location='cell',options=model.config.model_options,device=str(model.device),seed=42,
                 best_epoch=best_epoch,history=history,training_and_preparation_seconds=training_seconds,
                 test_teacher_forced=model.Test(),test_cells=len(cells),test_faces=len(fields['face_pressure']),
                 requested_rollout_steps=requested,rollout_failures=failures,rollout_steps=len(pred)-1,rollout_seconds=elapsed,finite=bool(np.isfinite(pred).all()),
                 cumulative_mse=float(error[1:].mean()),persistence_cumulative_mse=float(persistence[1:].mean()),
                 final_rmse=float(np.sqrt(error[-1])),reload_max_error=delta,reload_close=reload_close,
                 limitation='Small interpolated velocity dataset with assumed rho=1, nu=.001; no pressure accuracy or published benchmark reproduction claim. Finite outputs alone do not establish stability or accuracy.')
    (args.output/'metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
