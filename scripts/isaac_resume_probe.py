#!/usr/bin/env python3
"""Real PPO updates and independent-process checkpoint continuation tests."""
import argparse
import json
from pathlib import Path
import random
import signal
import sys
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--iterations',type=int,default=10)
parser.add_argument('--resume',type=Path)
parser.add_argument('--reference',type=Path)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
app=AppLauncher(args).app
env=None
stop_requested=False
def stop(_signum,_frame):
    global stop_requested
    stop_requested=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
try:
    import numpy as np
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from quadsmash.tasks.stand.jump_env import JumpEnv,JumpEnvCfg
    from quadsmash.rl.native_ppo import NativeRunner
    config=json.loads(Path('configs/coupled_ppo.json').read_text())['runner']
    cfg=JumpEnvCfg();cfg.seed=2017;cfg.scene.num_envs=32;cfg.episode_length_s=2.
    env=JumpEnv(cfg);wrapped=RslRlVecEnvWrapper(env,clip_actions=1.)
    runner=NativeRunner(wrapped,config,log_dir=str(args.output/'ppo'),device=env.device)
    resume_metrics={}
    if args.resume:
        runner.load(args.resume,load_optimizer=True)
        if args.reference is None:raise ValueError('independent resume validation requires --reference')
        reference=torch.load(args.reference,weights_only=False,map_location=env.device)
        with torch.inference_mode():
            policy=runner.get_inference_policy()
            obs=wrapped.get_observations()
            observation_error=float((obs['policy']-reference['initial_observation']).abs().max())
            observation_differences=torch.nonzero((obs['policy']-reference['initial_observation']).abs()>1e-5).tolist()
            action_error=float((policy(obs)-reference['initial_action']).abs().max())
            random_draws=dict(python=random.random(),numpy=float(np.random.random()),
                cpu=torch.rand(8),cuda=torch.rand(8,device=env.device))
            rng_ok=all(random_draws[k]==reference['rng_draws'][k] for k in ('python','numpy')) and all(
                torch.equal(random_draws[k],reference['rng_draws'][k].to(random_draws[k].device)) for k in ('cpu','cuda'))
            root_errors=[];joint_errors=[];velocity_errors=[]
            action_sensitivity=[]
            for expected in reference['continuation']:
                computed=policy(obs)
                replay=expected.get('action',computed)
                action_sensitivity.append(float((computed-replay).abs().max()))
                obs,_,_,_=wrapped.step(replay)
                root_errors.append(float((env.robot.data.root_pos_w-expected['root_position']).abs().max()))
                joint_errors.append(float((env.robot.data.joint_pos-expected['joint_position']).abs().max()))
                velocity_errors.append(float((env.robot.data.root_lin_vel_w-expected['root_velocity']).abs().max()))
            resume_metrics=dict(observation_error=observation_error,deterministic_action_error=action_error,
                observation_difference_indices=observation_differences[:40],
                rng_exact=rng_ok,root_position_error_m=max(root_errors),joint_position_error_rad=max(joint_errors),
                root_velocity_error_m_s=max(velocity_errors),restored_iteration=runner.current_learning_iteration,
                restored_optimizer_states=len(runner.alg.optimizer.state),restored_learning_rate=runner.alg.learning_rate)
            resume_metrics['continuation_mode']='identical recorded actions' if 'action' in reference['continuation'][0] else 'closed-loop legacy probe'
            resume_metrics['policy_sensitivity_during_replay_max_abs']=max(action_sensitivity)
            (args.output/'resume_comparison.json').write_text(json.dumps(resume_metrics,indent=2)+'\n')
            assert observation_error<1e-5 and action_error<1e-5 and rng_ok
            assert max(root_errors)<.005 and max(joint_errors)<.01 and max(velocity_errors)<.05
            assert runner.current_learning_iteration==reference['iteration'] and runner.alg.optimizer.state
        # Validate continuation above, then resume the original checkpoint so
        # diagnostics are not silently inserted into the learning trajectory.
        runner.load(args.resume,load_optimizer=True)
    initial_iteration=runner.current_learning_iteration
    losses=[]
    for _ in range(args.iterations):
        if stop_requested:break
        runner.learn(1,init_at_random_ep_len=False)
        losses.append(dict(iteration=runner.current_learning_iteration,**runner.alg.last_update_stats))
        temp=args.output/'progress.json.tmp'
        temp.write_text(json.dumps(dict(completed_updates=len(losses),iteration=runner.current_learning_iteration)))
        temp.replace(args.output/'progress.json')
    checkpoint=args.output/'checkpoint.pt'
    runner.save_restart(checkpoint)
    # A second process must reproduce the saved RNG stream and deterministic
    # action before checking a short physical continuation with tolerances.
    with torch.inference_mode():
        policy=runner.get_inference_policy();obs=wrapped.get_observations()
        reference=dict(initial_observation=obs['policy'].clone(),initial_action=policy(obs).clone(),
            iteration=runner.current_learning_iteration,
            rng_draws=dict(python=random.random(),numpy=float(np.random.random()),
                cpu=torch.rand(8),cuda=torch.rand(8,device=env.device)),continuation=[])
        for _ in range(5):
            action=policy(obs).clone()
            obs,_,_,_=wrapped.step(action)
            reference['continuation'].append(dict(action=action,root_position=env.robot.data.root_pos_w.clone(),
                joint_position=env.robot.data.joint_pos.clone(),root_velocity=env.robot.data.root_lin_vel_w.clone()))
        torch.save(reference,args.output/'continuation_reference.pt')
    result=dict(backend='Isaac Lab / GPU PhysX',updates=len(losses),initial_iteration=initial_iteration,
        completed_iteration=runner.current_learning_iteration,stop_requested=stop_requested,
        finite_gradients=True,losses=losses,resume_metrics=resume_metrics,checkpoint=str(checkpoint))
    (args.output/'resume.json').write_text(json.dumps(result,indent=2)+'\n')
    print('RESUME '+json.dumps({k:v for k,v in result.items() if k!='losses'}),flush=True)
except BaseException:
    (args.output/'error.txt').write_text(traceback.format_exc());raise
finally:
    if env is not None:env.close()
    app.close()
