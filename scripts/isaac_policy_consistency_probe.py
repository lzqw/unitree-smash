#!/usr/bin/env python3
"""Compare zero, deterministic, stochastic and perturbed ground rollouts."""
import argparse,csv,json,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
app=AppLauncher(a).app
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.rl.native_ppo import NativeRunner
from quadsmash.tasks.stand.jump_env import JumpEnv,JumpEnvCfg
env=None
try:
    cfg=JumpEnvCfg();cfg.scene.num_envs=128;cfg.seed=911;cfg.sim.device=a.device or 'cuda:0'
    env=JumpEnv(cfg);wrapped=RslRlVecEnvWrapper(env,clip_actions=None)
    config=json.loads(a.config.read_text());runner=NativeRunner(wrapped,config['runner'],log_dir=str(a.output/'ppo'),device=env.device)
    runner.load(str(a.checkpoint.resolve()),load_optimizer=False);policy=runner.get_inference_policy(device=env.device)
    with torch.inference_mode():
        wrapped.reset();ids=torch.arange(96,128,device=env.device)
        q=env.robot.data.default_joint_pos[ids]+.01*(2*torch.rand((32,18),device=env.device)-1)
        env.robot.write_joint_state_to_sim(q,torch.zeros_like(q),env_ids=ids)
        env.sim.forward();env.policy_history.reset(torch.arange(128,device=env.device))
        obs=wrapped.get_observations()
        normalized=runner.alg.policy.actor_obs_normalizer(runner.alg.policy.get_actor_obs(obs))
        raw=policy(obs)
        initial={'raw_observation_nominal_vs_perturbed_max':float((obs['policy'][96:]-obs['policy'][32:64]).abs().max()),
            'normalized_observation_nominal_vs_perturbed_max':float((normalized[96:]-normalized[32:64]).abs().max()),
            'nominal_action_mean':raw[32:64].mean(0).cpu().tolist(),'perturbed_action_mean':raw[96:].mean(0).cpu().tolist(),
            'nominal_normalized_max':float(normalized[32:64].abs().max()),'perturbed_normalized_max':float(normalized[96:].abs().max())}
        names=['zero_nominal','deterministic_nominal','stochastic_nominal','deterministic_joint_perturbation']
        for step in range(400):
            action=policy(obs).clone();sampled=runner.alg.policy.act(obs)
            action[:32]=0.;action[64:96]=sampled[64:96]
            obs,_,_,_=wrapped.step(action)
        first={}
        for row in env.episode_records:first.setdefault(row['env_id'],row)
        rows=[dict(group=names[i//32],**r) for i,r in sorted(first.items())]
        results={}
        for name in names:
            r=[x for x in rows if x['group']==name]
            results[name]={'episodes':len(r),'successes':sum(x['success'] for x in r),
                'body_landings':sum(x['body_landing'] for x in r),'max_com_rise_m':max(x['total_com_rise_m'] for x in r),
                'max_airborne_s':max(x['airborne_s'] for x in r),'mean_max_joint_speed':sum(x['max_joint_speed_rad_s'] for x in r)/len(r)}
        with (a.output/'episodes.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=sorted(rows[0]));w.writeheader();w.writerows(rows)
        report={'initial':initial,'groups':results,'env_steps':51200}
        (a.output/'metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
except BaseException:
    (a.output/'error.txt').write_text(traceback.format_exc());raise
finally:
    if env is not None:env.close()
    app.close(skip_cleanup=True)
