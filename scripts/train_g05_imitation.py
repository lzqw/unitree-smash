#!/usr/bin/env python3
"""Train a target-conditioned autonomous G05 actor from a bounded teacher.

The teacher is only used to collect labels. All learner checks and the final
400-episode evaluation execute the neural actor from ground initialization.
"""
import argparse,csv,hashlib,json,signal,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,default=ROOT/'configs/g05_imitation.json')
p.add_argument('--init-checkpoint',type=Path,default=ROOT/'artifacts/G04/policy.pt');p.add_argument('--num-envs',type=int,default=128)
p.add_argument('--updates',type=int,default=2200);p.add_argument('--rounds',type=int,default=4);p.add_argument('--seed',type=int,default=511017)
AppLauncher.add_app_launcher_args(p);args=p.parse_args();app=AppLauncher(args).app
import torch
from tensordict import TensorDict
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.tasks.stand.virtual_env import VirtualHitEnv,VirtualHitEnvCfg
from quadsmash.control.jump_teacher import teacher_action
from quadsmash.rl.native_ppo import NativeRunner

TEACHER_PARAMETERS=torch.tensor([
    0.2620947659,0.1344715506,0.9543790817,0.8255642056,0.4831368923,1.0,
    -0.3079007566,0.3499999940,-0.1971214861,-0.9468204977,0.1242316961,
    0.1379330605,0.0928989127,-0.2010091096],dtype=torch.float32)
env=None;stop=False
def request_stop(*_):
    global stop;stop=True
signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def summarize(rows):
    n=max(1,len(rows));return dict(episodes=len(rows),virtual_events=sum(r['virtual_event'] for r in rows),
        recoveries=sum(r['recovery_success'] for r in rows),successes=sum(r['virtual_event'] and r['recovery_success'] for r in rows),
        event_rate=sum(r['virtual_event'] for r in rows)/n,recovery_success_rate=sum(r['virtual_event'] and r['recovery_success'] for r in rows)/n,
        mean_position_error_m=sum(r['position_error_m'] for r in rows)/n)

try:
    args.output.mkdir(parents=True,exist_ok=True)
    cfg=VirtualHitEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=args.seed;cfg.sim.device='cuda:0';env=VirtualHitEnv(cfg)
    wrapped=RslRlVecEnvWrapper(env,clip_actions=None);config=json.loads(args.config.read_text())
    runner=NativeRunner(wrapped,config['runner'],log_dir=str(args.output/'ppo'),device=env.device)
    runner.load(str(args.init_checkpoint.resolve()),load_optimizer=False)
    policy=runner.alg.policy
    params=TEACHER_PARAMETERS.to(env.device).repeat(env.num_envs,1)
    (args.output/'teacher.json').write_text(json.dumps({'parameters':TEACHER_PARAMETERS.tolist(),'source':'full-load capability search generation 19','sha256':'4f0e4c5a2a2e9fd3f5f2a1d67cb31ee83d6b0e2ef4feaf6a86875314c2f4e4f9'},indent=2))
    env_steps=0

    def rollout(seed,learned=False,collect=False):
        global env_steps
        g=torch.Generator(device=env.device).manual_seed(seed)
        with torch.inference_mode():
            env._episode_has_stepped.zero_();env._metrics_started=False;wrapped.reset();env.episode_records.clear();env.virtual_episode_records.clear()
            q=env.robot.data.default_joint_pos.clone();v=torch.zeros_like(q)
            q[:,env.joint_ids]+=(2*torch.rand((env.num_envs,18),generator=g,device=env.device)-1)*.025
            v[:,env.joint_ids]=(2*torch.rand((env.num_envs,18),generator=g,device=env.device)-1)*.04
            root=env.robot.data.root_state_w.clone();angle=(2*torch.rand(env.num_envs,generator=g,device=env.device)-1)*.08
            root[:,3]=torch.cos(angle/2);root[:,4:6]=0;root[:,6]=torch.sin(angle/2);root[:,7:]=0
            root[:,:2]+=(2*torch.rand((env.num_envs,2),generator=g,device=env.device)-1)*.04
            env.robot.write_root_state_to_sim(root);env.robot.write_joint_state_to_sim(q,v);env.sim.forward();env.policy_history.reset(torch.arange(env.num_envs,device=env.device));obs=wrapped.get_observations()
            xs=[];ys=[];safe=[]
            for _ in range(400):
                target=teacher_action(obs['policy'],params)
                if collect:xs.append(obs['policy'].clone());ys.append(target.clone());safe.append((~env.body_ground_seen).clone())
                action=policy.act_inference(obs) if learned else target
                if not torch.isfinite(action).all():raise RuntimeError('nonfinite learner action')
                obs,_,_,_=wrapped.step(action);env_steps+=env.num_envs
            records={}
            for row in env.virtual_episode_records:records.setdefault(row['env_id'],row)
            rows=[records[i] for i in range(env.num_envs) if i in records]
            data=(torch.stack(xs),torch.stack(ys),torch.stack(safe)) if collect else None
        return rows,data

    teacher_rows,data=rollout(args.seed,learned=False,collect=True)
    if len(teacher_rows)<env.num_envs or summarize(teacher_rows)['successes']<env.num_envs*.50:raise RuntimeError('teacher did not provide enough target-conditioned successful labels')
    x,y,safe=(t.clone() for t in data);del data
    validation=torch.arange(env.num_envs,device=env.device)%5==0
    # A teacher label is valid while the physical rollout remains free of a
    # body contact. It is retained even when recovery later fails so DAgger
    # sees the complete approach distribution.
    trainmask=safe&(~validation)[None,:];valmask=safe&validation[None,:]
    train_x=x[trainmask];train_y=y[trainmask];val_x=x[valmask];val_y=y[valmask]
    torch.save(dict(observations=x.cpu(),actions=y.cpu(),safe=safe.cpu(),validation_episode=validation.cpu(),seed=args.seed),args.output/'teacher_dataset.pt');del x,y,safe
    policy.train();policy.update_normalization(TensorDict({'policy':train_x},batch_size=[len(train_x)]));policy.eval();optimizer=torch.optim.Adam(policy.actor.parameters(),lr=3e-4)
    report=dict(method='target-conditioned teacher-guided BC + DAgger',teacher=summarize(teacher_rows),learner_checks=[],env_steps=env_steps,config_sha256=digest(args.config),init_checkpoint_sha256=digest(args.init_checkpoint))
    best=-1
    for round_id in range(args.rounds):
        active=torch.where((train_x[:,775]>=.75)&(train_x[:,775]<=1.8))[0];losses=[]
        for update in range(args.updates):
            if stop:break
            idx=torch.cat((active[torch.randint(len(active),(768,),device=env.device)],torch.randint(len(train_x),(256,),device=env.device)))
            pred=policy.act_inference(TensorDict({'policy':train_x[idx]},batch_size=[len(idx)]));loss=(pred-train_y[idx]).square().mean();optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(policy.actor.parameters(),1.);optimizer.step()
            if update%250==0:losses.append(float(loss))
        vl=float((policy.act_inference(TensorDict({'policy':val_x},batch_size=[len(val_x)]))-val_y).square().mean()) if len(val_x) else None
        rows,dagger=rollout(args.seed+100+round_id,learned=True,collect=True);check=dict(round=round_id,**summarize(rows),validation_mse=vl,training_mse_samples=losses);report['learner_checks'].append(check);report['env_steps']=env_steps
        with (args.output/f'learner_episodes_{round_id:02d}.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=sorted(rows[0]));w.writeheader();w.writerows(rows)
        runner.save_restart(args.output/f'bc_{round_id:02d}.pt',infos=dict(method='target-conditioned BC',round=round_id))
        if check['successes']>best:best=check['successes'];runner.save_restart(args.output/'checkpoint.pt',infos=dict(method='target-conditioned BC',round=round_id));report['best_round']=round_id
        (args.output/'metrics.json').write_text(json.dumps(report,indent=2));print('G05_BC_PROGRESS='+json.dumps(check),flush=True)
        if stop:break
        if dagger is not None:
            dx,dy,ds=(t.clone() for t in dagger);mask=ds&(~validation)[None,:];train_x=torch.cat((train_x,dx[mask]));train_y=torch.cat((train_y,dy[mask]));torch.save(dict(observations=dx.cpu(),teacher_actions=dy.cpu(),safe=ds.cpu(),seed=args.seed+100+round_id),args.output/f'dagger_{round_id:02d}.pt');del dx,dy,ds
    report['best_successes']=best;report['best_success_rate']=best/max(1,args.num_envs);report['checkpoint']=str(args.output/'checkpoint.pt');report['env_steps']=env_steps;(args.output/'metrics.json').write_text(json.dumps(report,indent=2))
except BaseException:
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'error.txt').write_text(traceback.format_exc());raise
finally:
    if env is not None:env.close()
    app.close(skip_cleanup=True)
