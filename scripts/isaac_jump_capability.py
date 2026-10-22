#!/usr/bin/env python3
"""Ground-init bounded-action search; engineering evidence, never a PPO policy."""
import argparse,csv,json,sys,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--num-envs',type=int,default=256)
p.add_argument('--seed',type=int,default=711)
AppLauncher.add_app_launcher_args(p);args=p.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
app=AppLauncher(args).app
import torch
from quadsmash.tasks.stand.jump_env import JumpEnv,JumpEnvCfg
cfg=JumpEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.sim.device=args.device or 'cuda:0';cfg.seed=args.seed
env=None
try:
    env=JumpEnv(cfg)
    with torch.inference_mode():
        env.reset();n=env.num_envs;device=env.device
        g=torch.Generator(device=device).manual_seed(args.seed)
        params=torch.rand((n,8),generator=g,device=device)
        # Every lane is a real ground reset with identical audited hardware.
        crouch_time=.10+.25*params[:,0]
        push_time=.06+.20*params[:,1]
        crouch_thigh=.35+.65*params[:,2]
        crouch_calf=.4+.6*params[:,3]
        extend_thigh=.25+.75*params[:,4]
        extend_calf=.4+.6*params[:,5]
        front_bias=(params[:,6]-.5)*.6
        tuck_time=.05+.20*params[:,7]
        trace=[]
        for k in range(400):
            t=k*.01;action=torch.zeros((n,18),device=device)
            crouch=(t>=.8)&(t<.8+crouch_time)
            push=(t>=.8+crouch_time)&(t<.8+crouch_time+push_time)
            tuck=(t>=.8+crouch_time+push_time)&(t<.8+crouch_time+push_time+tuck_time)
            action[:,4:8]=torch.where(crouch,crouch_thigh,torch.where(push,-extend_thigh,torch.where(tuck,.3,0.)))[:,None]
            action[:,8:12]=torch.where(crouch,-crouch_calf,torch.where(push,extend_calf,torch.where(tuck,-.6,0.)))[:,None]
            action[:,4:6] += (push*front_bias)[:,None]
            action[:,6:8] -= (push*front_bias)[:,None]
            action[0]=0. # Negative control, standing throughout.
            env.step(action.clamp(-1.,1.))
            if k%5==0:
                trace.append({'t':t,'max_com_rise':float(env.max_com_rise.max()),
                    'max_continuous_air_s':float(env.airborne_time.max()),'max_body_force_n':float(env.maximum_body_ground_force.max()),
                    'qualified_flights':int(env.flight_qualified.sum()),'successes':int(env.jump_success.sum())})
        rows=env.episode_records[:]
        parameters=torch.stack((crouch_time,push_time,crouch_thigh,crouch_calf,extend_thigh,extend_calf,front_bias,tuck_time),1).cpu().tolist()
        for row in rows:row['parameters']=json.dumps(parameters[row['env_id']])
        with (args.output/'episodes.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=sorted(rows[0]));w.writeheader();w.writerows(rows)
        report={'kind':'bounded action capability diagnostic, not trained policy',
            'sample_count':len(rows),'total_mass_kg':float(env.body_masses[0].sum()),
            'maximum_com_rise_m':max(r['total_com_rise_m'] for r in rows),
            'maximum_continuous_air_s':max(r['airborne_s'] for r in rows),
            'qualified_jump_landed_count':sum(r['qualified_jump_landed'] for r in rows),
            'success_count':sum(r['success'] for r in rows),'negative_control':next(r for r in rows if r['env_id']==0),
            'env_steps':400*n,'best':sorted(rows,key=lambda r:(r['success'],r['qualified_jump_landed'],r['total_com_rise_m']),reverse=True)[:10]}
        (args.output/'trace.json').write_text(json.dumps(trace,indent=2))
        (args.output/'metrics.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)
except BaseException:
    (args.output/'error.txt').write_text(traceback.format_exc());raise
finally:
    if env is not None:env.close()
    app.close(skip_cleanup=True)
