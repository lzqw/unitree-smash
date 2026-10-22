#!/usr/bin/env python3
"""Two frozen 200-episode ground panels, evaluated in an independent process."""
import argparse,csv,hashlib,json,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--config',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--num-envs',type=int,default=32)
p.add_argument('--episodes',type=int,default=400)
p.add_argument('--suite',default='dev_jump_v2')
p.add_argument('--reset-mode',choices=['ground'],default='ground')
p.add_argument('--stage',default='jump');p.add_argument('--seed',type=int,default=1907)
AppLauncher.add_app_launcher_args(p);args=p.parse_args()
if args.episodes!=400:p.error('the frozen G04 protocol requires exactly 2 x 200 episodes')
args.output.mkdir(parents=True,exist_ok=True)
app=AppLauncher(args).app
import numpy as np
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.rl.native_ppo import NativeRunner
from quadsmash.tasks.stand.jump_env import JumpEnv,JumpEnvCfg,PROTOCOL

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_csv(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=sorted(rows[0]));w.writeheader();w.writerows(rows)

env=None
try:
    cfg=JumpEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.sim.device=args.device or 'cuda:0';cfg.seed=1907
    config=json.loads(args.config.read_text());cfg.episode_length_s=config['episode_seconds']
    env=JumpEnv(cfg);wrapped=RslRlVecEnvWrapper(env,clip_actions=None)
    runner=NativeRunner(wrapped,config['runner'],log_dir=str(args.output/'ppo'),device=env.device)
    runner.load(str(args.checkpoint.resolve()),load_optimizer=False)
    policy=runner.get_inference_policy(device=env.device)
    rows=[];initial_rows=[];trace_rows=[];numerical_failures=0;env_steps=0
    with torch.inference_mode():
        for panel in PROTOCOL['panels']:
            rng=np.random.Generator(np.random.PCG64(panel['seed']))
            count=panel['episodes']
            dq=rng.uniform(-panel['joint_offset_uniform_rad'],panel['joint_offset_uniform_rad'],(count,18))
            qd=rng.uniform(-panel['joint_velocity_uniform_rad_s'],panel['joint_velocity_uniform_rad_s'],(count,18))
            yaw=rng.uniform(-panel['yaw_uniform_rad'],panel['yaw_uniform_rad'],count)
            xy=rng.uniform(-panel['xy_uniform_m'],panel['xy_uniform_m'],(count,2))
            for start in range(0,count,env.num_envs):
                size=min(env.num_envs,count-start);ids=torch.arange(size,device=env.device)
                env._episode_has_stepped.zero_();env._metrics_started=False
                wrapped.reset();env.episode_records.clear()
                root=env.robot.data.root_state_w[ids].clone()
                root[:,:2]+=torch.tensor(xy[start:start+size],device=env.device,dtype=torch.float32)
                angles=torch.tensor(yaw[start:start+size],device=env.device,dtype=torch.float32)
                root[:,3]=torch.cos(angles/2);root[:,4:6]=0.;root[:,6]=torch.sin(angles/2)
                root[:,7:]=0.
                q=env.robot.data.default_joint_pos[ids].clone()
                v=torch.zeros_like(q)
                q[:,env.joint_ids]+=torch.tensor(dq[start:start+size],device=env.device,dtype=torch.float32)
                v[:,env.joint_ids]=torch.tensor(qd[start:start+size],device=env.device,dtype=torch.float32)
                env.robot.write_root_state_to_sim(root,env_ids=ids)
                env.robot.write_joint_state_to_sim(q,v,env_ids=ids)
                env.sim.forward();env.policy_history.reset(torch.arange(env.num_envs,device=env.device))
                obs=wrapped.get_observations();ground={}
                for i in range(size):
                    ground[i]=bool(torch.isfinite(root[i]).all() and abs(float(root[i,2])-.323)<1e-5)
                    initial_rows.append({'panel':panel['id'],'scenario':start+i,'ground_init':ground[i],
                        'root_state':json.dumps(root[i].cpu().tolist()),'joint_position':json.dumps(q[i].cpu().tolist()),
                        'joint_velocity':json.dumps(v[i].cpu().tolist())})
                seen=set();cursor=0
                for tick in range(env.max_episode_length):
                    action=policy(obs)
                    numerical_failures+=int(not torch.isfinite(action).all())
                    obs,reward,done,_=wrapped.step(action);env_steps+=env.num_envs
                    numerical_failures+=int(not torch.isfinite(obs['policy']).all())+int(not torch.isfinite(reward).all())
                    # Fixed scenario IDs, chosen before outcomes; no cherry-picking.
                    if start==0 and tick%5==0:
                        for i in range(min(2,size)):
                            if i not in seen and not bool(done[i]):
                                trace_rows.append({'panel':panel['id'],'scenario':i,'time_s':(tick+1)*env.step_dt,
                                    'com_height_m':float(env.com_position[i,2]),'com_vertical_velocity_m_s':float(env.com_velocity[i,2]),
                                    'com_rise_m':float(env.max_com_rise[i]),'flight_continuous_s':float(env.flight_clear_time[i]),
                                    'recovery_continuous_s':float(env.recovery_run[i]),
                                    'supported_feet':int(env.observed_foot_contact[i].sum()),
                                    'horizontal_speed_m_s':float(env.robot.data.root_link_lin_vel_w[i,:2].norm()),
                                    'angular_speed_rad_s':float(env.robot.data.root_ang_vel_w[i].norm()),
                                    'maximum_joint_speed_rad_s':float(env.robot.data.joint_vel[i].abs().max()),
                                    'body_ground_seen':bool(env.body_ground_seen[i]),
                                    'raw_action':json.dumps(action[i].cpu().tolist())})
                    for record in env.episode_records[cursor:]:
                        i=record['env_id']
                        if i<size and i not in seen:
                            rows.append({'panel':panel['id'],'scenario':start+i,'ground_init':ground[i],**record})
                            seen.add(i)
                    cursor=len(env.episode_records)
                    if len(seen)==size:break
                if len(seen)!=size:raise RuntimeError('missing completed evaluation episodes')
            panel_rows=[r for r in rows if r['panel']==panel['id']]
            print(json.dumps({'panel':panel['id'],'episodes':len(panel_rows),'successes':sum(r['success'] for r in panel_rows)}),flush=True)
    panels={p['id']:sum(r['success'] and r['ground_init'] for r in rows if r['panel']==p['id'])/p['episodes'] for p in PROTOCOL['panels']}
    metrics={'sample_count':len(rows),'ground_init_fraction':sum(r['ground_init'] for r in rows)/len(rows),
        'jump_recovery_success_rate':min(panels.values()),'panel_success_rates':panels,'numerical_failures':numerical_failures,
        'env_steps':env_steps,'backend':'Isaac Lab / PhysX','task':'jump_recover_ground_v1',
        'checkpoint_sha256':digest(args.checkpoint),'protocol_sha256':digest(ROOT/'configs/evaluation_protocol_v2.json'),
        'qualified_jump_landed_rate':sum(r['qualified_jump_landed'] for r in rows)/len(rows),
        'qualified_flight_seen_rate':sum(r['qualified_flight_seen'] for r in rows)/len(rows),
        'body_landing_rate':sum(r['body_landing'] for r in rows)/len(rows),
        'maximum_com_rise_m':max(r['total_com_rise_m'] for r in rows)}
    write_csv(args.output/'episodes.csv',rows);write_csv(args.output/'initial_states.csv',initial_rows)
    if trace_rows:write_csv(args.output/'fixed_scenario_trace.csv',trace_rows)
    manifest={'protocol':PROTOCOL,'suite_requested':args.suite,'suite_executed':'dev_jump_v2_a+dev_jump_v2_b',
        'checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':digest(args.checkpoint),
        'config':str(args.config.resolve()),'config_sha256':digest(args.config),'metrics_source':'episodes.csv','initial_states_source':'initial_states.csv',
        'num_envs':env.num_envs,'evaluation_policy':'deterministic learned actor inference; no assisting controller'}
    (args.output/'eval_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (args.output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics),flush=True)
except BaseException:
    (args.output/'error.txt').write_text(traceback.format_exc());raise
finally:
    if env is not None:env.close()
    app.close(skip_cleanup=True)
