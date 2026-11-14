#!/usr/bin/env python3
"""Compare full physical-state errors with unchanged ball model and controls."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
import torch
from quadsmash.rl.supermimic_ball_sequence_v1 import ball_transition
from quadsmash.rl.supermimic_ball_geometry_preview_v1 import BallGeometryPreviewV1,GeometrySupportedBallV1
from quadsmash.rl.supermimic_ball_residual_v2 import BallResidualV2
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2
from scripts.fit_supermimic_ball_residual_v2 import verified
from scripts.fit_supermimic_physical_world import model_args,write_json


@torch.no_grad()
def audit(base,ball,data,pairs,launch):
    t,l=pairs.T;state,_,_,feet,present,elapsed,lb=model_args(data,pairs,launch)
    initial_phase=data['observation'][t,l,770:775].argmax(-1)
    result={}
    for k in range(64):
        state,feet,present=ball_transition(base,ball,state,data['final_joint_targets'][t+k,l],data['observation'][t+k,l,-18:],feet,present,elapsed,lb)
        elapsed=elapsed+.01
        if k+1 not in (1,8,16,32,64):continue
        truth=data['next_state'][t+k,l];err=(state-truth).abs()
        # q and -q denote the same orientation; report angle in physical units.
        angle=(state[:,3:7]*truth[:,3:7]).sum(-1).abs().clamp(0,1).acos()*360/torch.pi
        def stats(mask):
            n=int(mask.sum())
            if not n:return {'count':0}
            out={'count':n,'root_orientation_mean_deg':float(angle[mask].mean()),
                'foot_accuracy':float(((feet[mask]>=.5)==data['next_foot_contact'][t+k,l][mask]).float().mean())}
            for name,a,b in (('root_position_mae_m',0,3),('root_velocity_mae_m_s',7,10),('root_omega_mae_rad_s',10,13),
                             ('joint_mae_rad',13,31),('joint_velocity_mae_rad_s',31,49),('ball_position_mae_m',49,52)):
                out[name]=float(err[mask,a:b].mean())
            return out
        result[str(k+1)]={'all':stats(torch.ones(len(t),device=t.device,dtype=torch.bool)),
            'by_initial_phase':{str(i):stats(initial_phase==i) for i in range(5)}}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if not (a.run/'completed.json').exists():raise ValueError('Fit must have completed')
    torch.set_num_threads(2);begin=time.perf_counter();inputs=json.loads((a.run/'inputs.json').read_text())
    old=verified(inputs['physical_checkpoint'],inputs['physical_checkpoint_sha256']);old_ckpt=torch.load(old,map_location='cpu',weights_only=True)
    new_ckpt=torch.load(a.run/'world.pt',map_location='cpu',weights_only=True)
    if new_ckpt['source_checkpoint_sha256']!=inputs['physical_checkpoint_sha256']:raise ValueError('Wrong lineage')
    bp=verified(inputs['ball_checkpoint'],inputs['ball_checkpoint_sha256']);bc=torch.load(bp,map_location='cpu',weights_only=True)
    ball=BallResidualV2().cuda();ball.load_state_dict(bc['model'],strict=True);preview=BallGeometryPreviewV1().cuda()
    if preview.geometry_hashes!=inputs['geometry_hashes']:raise ValueError('Changed geometry')
    model=GeometrySupportedBallV1(ball,preview).eval().requires_grad_(False)
    data=torch.load(verified(inputs['dataset'],inputs['dataset_sha256']),map_location='cpu',weights_only=True)['tensors']
    data={k:v.cuda() for k,v in data.items()}
    suite=old_ckpt['source_manifest']['task_contract']['suite'];cases=json.loads(verified(suite['path'],suite['sha256']).read_text())['cases']
    launch=torch.tensor([x['launch']['position_m']+x['launch']['velocity_mps'] for x in cases],device='cuda')
    pairs=torch.tensor(inputs['split']['validation_selected'],device='cuda');results={}
    for label,ckpt in (('source',old_ckpt),('candidate',new_ckpt)):
        base=PhysicalWorldV2(ckpt['source_manifest']['task_contract']['asset']['sha256']).cuda()
        base.load_state_dict(ckpt['model'],strict=True);base.eval().requires_grad_(False)
        results[label]=audit(base,model,data,pairs,launch)
    write_json(a.output,dict(scope='Fixed held-out-window diagnostics, no native evaluation or parameter update',
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        candidate_sha256=hashlib.sha256((a.run/'world.pt').read_bytes()).hexdigest(),optimizer_updates=0,
        seconds=time.perf_counter()-begin,results=results))
    print(json.dumps({label:r['64']['all'] for label,r in results.items()},indent=2))


if __name__=='__main__':main()
