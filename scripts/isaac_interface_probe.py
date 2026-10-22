#!/usr/bin/env python3
"""Native G03 observation, partial reset and final-state probe."""
import argparse
import json
from pathlib import Path
import sys
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--steps',type=int,default=200)
parser.add_argument('--stage',choices=('jump','virtual','ball'),default='jump')
parser.add_argument('--num-envs',type=int,default=32)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
app=AppLauncher(args).app
env=None
try:
    import torch
    from quadsmash.tasks.stand.jump_env import JumpEnv,JumpEnvCfg
    from quadsmash.tasks.stand.virtual_env import VirtualHitEnv,VirtualHitEnvCfg
    from quadsmash.tasks.stand.ball_env import BallEnv,BallEnvCfg
    from quadsmash.tasks.stand.observation_contract import OBSERVATION_SIZE
    cfg_class,env_class={'jump':(JumpEnvCfg,JumpEnv),'virtual':(VirtualHitEnvCfg,VirtualHitEnv),'ball':(BallEnvCfg,BallEnv)}[args.stage]
    cfg=cfg_class();cfg.seed=1977;cfg.scene.num_envs=args.num_envs;cfg.episode_length_s=.10
    env=env_class(cfg)
    with torch.inference_mode():
        obs,_=env.reset()
        assert obs['policy'].shape==(args.num_envs,OBSERVATION_SIZE)
        assert (obs['policy'][:,67]==(1. if args.stage=='ball' else 0.)).all()
        zero=torch.zeros((args.num_envs,18),device=env.device)
        for _ in range(3):env.step(zero)
        physical_before=env.robot.data.root_state_w[1:].clone()
        history_before=env.policy_history.frames[1:].clone()
        obs_before=env._get_observations()['policy'][1:].clone()
        env._reset_idx(torch.tensor([0],device=env.device))
        obs_after=env._get_observations()['policy'][1:]
        isolation=bool(torch.equal(physical_before,env.robot.data.root_state_w[1:]) and
            torch.equal(history_before,env.policy_history.frames[1:]) and torch.equal(obs_before,obs_after))
        timeout_count=0
        largest_final_reset_difference=0.
        for _ in range(args.steps):
            obs,reward,term,trunc,extras=env.step(zero)
            assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
            mask=term|trunc
            assert torch.equal(extras['final_observation_valid'],mask)
            if mask.any():
                final=extras['final_observation']['policy'][mask]
                reset=obs['policy'][mask]
                assert (final[:,-19]>.0).all() and (reset[:,-19]==0).all()
                assert reset[:,:6].abs().max()<1e-6
                largest_final_reset_difference=max(largest_final_reset_difference,float((final-reset).abs().max()))
            timeout_count+=int((trunc&~term).sum())
        result=dict(backend='Isaac Lab / GPU PhysX',stage=args.stage,shape=list(obs['policy'].shape),
            partial_reset_isolation_ok=isolation,timeout_count=timeout_count,
            final_reset_observation_max_difference=largest_final_reset_difference,
            steps=args.steps,numerical_failures=0)
        assert isolation and timeout_count>0 and largest_final_reset_difference>.01
        (args.output/'interface.json').write_text(json.dumps(result,indent=2)+'\n')
        print('INTERFACE '+json.dumps(result),flush=True)
except BaseException:
    (args.output/'error.txt').write_text(traceback.format_exc())
    raise
finally:
    if env is not None:env.close()
    app.close()
