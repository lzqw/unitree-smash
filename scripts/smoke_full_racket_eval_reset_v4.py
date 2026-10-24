#!/usr/bin/env python3
"""Native perturbed ground/partial reset check on TRAIN inputs, no formal score."""
import argparse,hashlib,json,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
AppLauncher.add_app_launcher_args(p);args=p.parse_args();app=AppLauncher(args).app
import numpy as np
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.tasks.stand.full_racket_env_v4 import full_racket_cfg
from quadsmash.tasks.stand.full_racket_eval_env_v4 import FrozenPanelEnv


def main():
    args.output.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((ROOT/'configs/g07_full_racket_eval_v4_r3.json').read_text())
    source=ROOT/protocol['training_suite']['path']
    assert hashlib.sha256(source.read_bytes()).hexdigest()==protocol['training_suite']['sha256']
    cases=json.loads(source.read_text())['cases'][:8]
    generator=np.random.default_rng(protocol['initialization']['reset_seeds']['B'])
    bounds=protocol['initialization']['perturbations']['B']
    # Draw exactly the complete formal noise sequence, use eight samples
    # with training balls only. No formal case or policy outcome is read.
    noise={name:generator.uniform(-bounds[name],bounds[name],shape)[:8] for name,shape in
        [('joint_position_rad',(200,18)),('joint_velocity_rad_s',(200,18)),('root_xy_m',(200,2)),('root_yaw_rad',(200,))]}
    env=FrozenPanelEnv(full_racket_cfg(8),cases=cases,reset_noise=noise)
    wrapped=RslRlVecEnvWrapper(env,clip_actions=None)
    try:
        observation,_=wrapped.reset();ids=torch.arange(8,device=env.device)
        data=env.robot.data;default=data.default_joint_pos[:,env.joint_ids]
        limits=data.joint_pos_limits[:,env.joint_ids]
        desired_q=(default+torch.tensor(noise['joint_position_rad'],dtype=torch.float32,device=env.device)).clamp(limits[...,0],limits[...,1])
        desired_qd=torch.tensor(noise['joint_velocity_rad_s'],dtype=torch.float32,device=env.device)
        desired_xy=torch.tensor(noise['root_xy_m'],dtype=torch.float32,device=env.device)
        angle=torch.tensor(noise['root_yaw_rad'],dtype=torch.float32,device=env.device)/2
        desired_quat=torch.stack((angle.cos(),torch.zeros_like(angle),torch.zeros_like(angle),angle.sin()),1)
        root=data.root_state_w.clone();actual_local=root[:,:3]-env.scene.env_origins
        errors=dict(joint_position_rad=float((data.joint_pos[:,env.joint_ids]-desired_q).abs().max()),
            joint_velocity_rad_s=float((data.joint_vel[:,env.joint_ids]-desired_qd).abs().max()),
            root_xy_m=float((actual_local[:,:2]-desired_xy).abs().max()),root_z_m=float((actual_local[:,2]-.323).abs().max()),
            quaternion=float((root[:,3:7]-desired_quat).abs().max()))
        assert all(v<1e-5 for v in errors.values()),errors
        assert not env.robot.is_fixed_base and torch.isfinite(observation['policy']).all()
        assert torch.equal(observation['policy'][:,9:27],data.joint_pos[:,env.joint_ids]-default)
        assert torch.equal(observation['policy'][:,27:45],data.joint_vel[:,env.joint_ids])
        assert not env.launch_done.any() and not env.ball_observation_enabled.any()
        initial=[row for row in env.initial_states if row['env_id']<8]
        for _ in range(6):
            targets=2*(desired_q-limits[...,0])/(limits[...,1]-limits[...,0])-1
            observation,_,_,_=wrapped.step(targets)
        partial=torch.tensor([1,5],device=env.device)
        preserved=torch.tensor([0,2,3,4,6,7],device=env.device)
        before_root=env.robot.data.root_state_w[preserved].clone();before_q=env.robot.data.joint_pos[preserved].clone()
        env._reset_idx(partial)
        after_root=env.robot.data.root_state_w[preserved];after_q=env.robot.data.joint_pos[preserved]
        isolated_root=float((after_root-before_root).abs().max());isolated_q=float((after_q-before_q).abs().max())
        assert isolated_root<1e-5 and isolated_q<1e-6
        assert env.policy_history.frames[partial].count_nonzero()==0
        assert env.last_action[partial].count_nonzero()==0
        assert torch.allclose(env.robot.data.joint_pos[partial][:,env.joint_ids],desired_q[partial],atol=1e-6,rtol=0)
        assert not env.launch_done.any()
        result=dict(kind='Native perturbed/partial reset smoke only; no strategy evaluation',
            status='PASSED',environments=8,control_ticks=6,physics_dt_s=env.physics_dt,formal_cases_read=False,
            noise_panel='B',noise_seed=protocol['initialization']['reset_seeds']['B'],initial_errors=errors,
            unaffected_root_error=isolated_root,unaffected_joint_error=isolated_q,
            initial_states=initial,asset_sha256=protocol['asset']['sha256'],formal_acceptance=False)
        (args.output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
        (args.output/'completed.json').write_text(json.dumps(dict(env_steps=8*6))+'\n')
        print(json.dumps(result),flush=True)
    finally:env.close()


try:main()
except BaseException:
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'error.txt').write_text(traceback.format_exc());raise
finally:app.close(skip_cleanup=True)
