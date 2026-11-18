"""Read-only event-aligned additive ball prediction decomposition."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2
from quadsmash.rl.supermimic_ball_residual_v2 import BallResidualV2
from quadsmash.rl.supermimic_action_ball_v1 import action_ball_for_schema
from quadsmash.rl.supermimic_ball_residual_v1 import ball_geometry_features

parser=argparse.ArgumentParser();parser.add_argument('--model-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
if args.output.exists():raise FileExistsError(args.output)
torch.set_num_threads(2)
root=Path('/home/autodrive/LZQW/experiments/supermimic_admission256_20261118_s5lbit6k')
package=torch.load('/home/autodrive/LZQW/experiments/supermimic_frozen_closure_20261117_223243/packed/data.pt',map_location='cpu',weights_only=True);d=package['tensors']
models={name:torch.load(args.model_root/file,map_location='cpu',weights_only=True) for name,file in [('world','world.pt'),('ball','ball.pt'),('sweep','action_ball_candidate.pt')]}
world=PhysicalWorldV2(package['source_manifest']['task_contract']['asset']['sha256']);world.load_state_dict(models['world']['model'],strict=True)
ball=BallResidualV2();ball.load_state_dict(models['ball']['model'],strict=True)
sweep=action_ball_for_schema(models['sweep']['schema']);sweep.load_state_dict(models['sweep']['model'],strict=True)
suite=Path(package['source_manifest']['task_contract']['suite']['path']);cases=json.loads(suite.read_text())['cases']
launch=torch.tensor([c['launch']['position_m']+c['launch']['velocity_mps'] for c in cases])
split=json.loads((root/'fixed_split.json').read_text())
edge=(d['next_event_state'][...,1]>d['event_state'][...,1])|(d['next_event_state'][...,6]>d['event_state'][...,6])
results={}
with torch.no_grad():
    for group in ('train_lanes','validation_lanes'):
        rows={}
        for offset in (-1,0,1):
            pairs=[(t+offset,l) for l in split[group] for t in edge[:,l].nonzero().flatten().tolist()
                if 0<=t+offset<len(edge) and d['episode'][t,l]==d['episode'][t+offset,l]]
            t,l=torch.tensor(pairs).unbind(1);state=d['state'][t,l];present=d['present'][t,l]
            prior,_=world.prior(state,present,d['elapsed'][t,l],launch[l])
            physical,_,_=world(state,d['targets'][t,l],d['previous_action'][t,l],d['feet'][t,l],present,d['elapsed'][t,l],launch[l])
            features=ball_geometry_features(state,world.geometry)
            probability,responses=ball.components(features);legacy=(probability[:,1:,None]*responses).sum(1)*present[:,None]
            kwargs={'racket_probability':probability[:,1]} if getattr(sweep,'requires_racket_probability',False) else {}
            action,details=sweep(state,physical,prior[:,49:],world.geometry,present,details=True,**kwargs)
            truth=d['next_state'][t,l,49:];native_residual=truth-prior[:,49:]
            total=prior[:,49:]+legacy+action
            def rms(x):return float(x.square().mean().sqrt())
            rows[str(offset)]=dict(count=len(t),velocity_rmse=rms(total[:,3:]-truth[:,3:]),
                without_action_velocity_rmse=rms(prior[:,52:]+legacy[:,3:]-truth[:,3:]),
                legacy_correction_velocity_rms=rms(legacy[:,3:]),action_correction_velocity_rms=rms(action[:,3:]),
                native_impulse_velocity_rms=rms(native_residual[:,3:]),
                racket_probability_mean=float(probability[:,1].mean()),free_probability_mean=float(probability[:,0].mean()),
                geometric_soft_support_mean=float(details['support'].mean()),
                samples=[dict(lane=int(l[i]),tick=int(t[i]),native_residual=native_residual[i].tolist(),
                    legacy_correction=legacy[i].tolist(),action_correction=action[i].tolist(),mode_probabilities=probability[i].tolist(),
                    support=float(details['support'][i])) for i in range(len(t))])
        results[group]=rows
manifest={file:hashlib.sha256((args.model_root/file).read_bytes()).hexdigest() for file in ('world.pt','ball.pt','action_ball_candidate.pt')}
args.output.write_text(json.dumps(dict(scope=__doc__,models=manifest,results=results,actor_updates=0,model_updates=0,
    limitation='Component magnitudes are not a causal error decomposition; no alternative model or changed forward was evaluated.'),indent=2)+'\n')
print(json.dumps({g:{t:{k:v for k,v in row.items() if k!='samples'} for t,row in times.items()} for g,times in results.items()},indent=2))
