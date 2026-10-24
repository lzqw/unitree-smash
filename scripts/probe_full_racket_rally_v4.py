#!/usr/bin/env python3
"""Actual court-scale shuttle rollout of a learned full-racket controller."""
import argparse,hashlib,json,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--count',type=int,default=16)
AppLauncher.add_app_launcher_args(p);args=p.parse_args();app=AppLauncher(args).app
import numpy as np
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaacsim.core.utils.stage import get_current_stage
from quadsmash.tasks.stand.full_racket_env_v4 import full_racket_cfg,FullRacketEnvV4
from quadsmash.tasks.stand.visual_geometry import export_robot_geometry
from quadsmash.control.full_racket_rally_controller_v4 import FullRacketRallyControllerV4


def main():
    args.output.mkdir(parents=True,exist_ok=True)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    protocol=json.loads((ROOT/'configs/g07_full_racket_eval_v4_r3.json').read_text())
    assert checkpoint['asset_sha256']==protocol['asset']['sha256']
    assert 2<=args.count<=128 and args.count%2==0
    suite=protocol['training_suite'];path=ROOT/suite['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==suite['sha256']
    all_cases=json.loads(path.read_text())['cases']
    indices=list(range(args.count//2))+list(range(64,64+args.count//2))
    cases=[all_cases[i] for i in indices]
    cfg=full_racket_cfg(args.count);cfg.seed=720601
    env=FullRacketEnvV4(cfg,cases=cases);wrapper=RslRlVecEnvWrapper(env,clip_actions=None)
    controller=FullRacketRallyControllerV4(checkpoint,args.count,env.device)
    try:
        observation,_=wrapper.reset();controller.reset()
        export_robot_geometry(get_current_stage(),env.robot,args.output)
        default=env.robot.data.default_joint_pos[0,env.joint_ids]
        limits=env.robot.data.joint_pos_limits[0,env.joint_ids]
        observations,actions,poses,traces,masks=[],[],[],[],[]
        finished=torch.zeros(args.count,dtype=torch.bool,device=env.device)
        with torch.inference_mode():
            for tick in range(800):
                action=controller.action(observation,default,limits)
                observations.append(observation['policy'].cpu().numpy().copy())
                actions.append(action.cpu().numpy().copy())
                masks.append((~finished).cpu().numpy().copy())
                if tick%2==0:
                    data=env.robot.data
                    poses.append(torch.cat((data.body_link_pos_w[:2]-env.scene.env_origins[:2,None],data.body_link_quat_w[:2]),-1).cpu().numpy())
                    traces.append(dict(time_s=tick*.01,ball_position_m=(env.shuttle.data.root_pos_w[:2]-env.scene.env_origins[:2]).cpu().tolist(),
                        ball_velocity_mps=env.shuttle.data.root_lin_vel_w[:2].cpu().tolist(),
                        active=env.rally.in_play[:2].cpu().tolist(),contact=env.rally.had_contact[:2].cpu().tolist(),
                        airborne=env.observed_airborne[:2].cpu().tolist()))
                observation,_,done,_=wrapper.step(action);finished|=done.bool()
        first={}
        for row in env.rally_records:first.setdefault(row['env_id'],row)
        assert len(first)==args.count
        rows=[dict(first[i],case_id=cases[i]['case_id'],source_case_index=indices[i]) for i in range(args.count)]
        np.save(args.output/'public_observations.npy',np.stack(observations))
        np.save(args.output/'public_actions.npy',np.stack(actions))
        np.save(args.output/'valid_first_episode_mask.npy',np.stack(masks))
        np.save(args.output/'link_poses_first2.npy',np.stack(poses))
        (args.output/'trajectory_first2.json').write_text(json.dumps(traces)+'\n')
        (args.output/'episodes.json').write_text(json.dumps(rows,indent=2)+'\n')
        (args.output/'contacts.json').write_text(json.dumps(env.contact_records,indent=2)+'\n')
        (args.output/'causal_plans.json').write_text(json.dumps(controller.logs,indent=2)+'\n')
        air=sum(r['airborne_string_hit'] and r['legal_initial_contact'] for r in rows)
        result=dict(kind='Native ball development probe; not formal400 acceptance',episodes=args.count,
            complete=sum(r['complete_smash'] for r in rows),airborne_contacts=air,
            contacts=sum(r['had_contact'] for r in rows),crossed_net=sum(r['crossed_net'] for r in rows),
            landed_target=sum(r['landed_target'] for r in rows),recovery=sum(r['recovery_success'] for r in rows),
            numerical_failures=sum(r['numerical_failures'] for r in rows),
            fault_counts={name:sum(r[name] for r in rows) for name in env.rally.flags if 'fault' in name or name=='engineering_body_crossing'},
            policy_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),asset_sha256=protocol['asset']['sha256'],
            source_training_suite=suite,ground_initialization=True,reset_perturbations=False,
            physics_dt_s=cfg.sim.dt,formal_acceptance=False,old_passes_inherited=False)
        (args.output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
        (args.output/'completed.json').write_text(json.dumps(dict(env_steps=800*args.count))+'\n')
        print(json.dumps(result),flush=True)
    finally:env.close()


try:main()
except BaseException:
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'error.txt').write_text(traceback.format_exc());raise
finally:app.close(skip_cleanup=True)
