#!/usr/bin/env python3
"""Train a high-arm G07 actor from a force-free real-PhysX teacher."""
import argparse, csv, hashlib, json, signal, sys, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True); p.add_argument('--config',type=Path,default=ROOT/'configs/g07_high_arm.json'); p.add_argument('--init-checkpoint',type=Path,default=ROOT/'artifacts/G06/policy.pt'); p.add_argument('--num-envs',type=int,default=64); p.add_argument('--updates',type=int,default=600); p.add_argument('--rounds',type=int,default=2); p.add_argument('--seed',type=int,default=613060); p.add_argument('--physics-dt',type=float,default=.00025); AppLauncher.add_app_launcher_args(p); a=p.parse_args(); app=AppLauncher(a).app
import torch
from tensordict import TensorDict
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.tasks.stand.ball_env import BallEnv,BallEnvCfg
from quadsmash.control.jump_teacher import teacher_action
from quadsmash.rl.native_ppo import NativeRunner

PARAMS=torch.tensor([.2620947659,.1344715506,.9543790817,.8255642056,.4831368923,1.,-.3079007566,.349999994,-.1971214861,-.9468204977,.1242316961,.25,.15,-.20],dtype=torch.float32)
Q_PRE=torch.tensor([.13985,1.47213,-2.90887,1.29097,-1.17635,-.01722],dtype=torch.float32)
Q_LATE=torch.tensor([-.675925,1.060654,-1.455027,.369168,-.298615,-.248977],dtype=torch.float32)
stop=False; env=None
def sig(*_):
 global stop; stop=True
signal.signal(signal.SIGTERM,sig);signal.signal(signal.SIGINT,sig)
def digest(path):
 import hashlib
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def summarize(rows):
 n=max(1,len(rows)); contact=sum(bool(r['contact']) for r in rows); air=sum(bool(r['airborne_contact']) for r in rows); attack=sum(bool(r['contact'] and r['outgoing_speed_mps']>=8 and r['outgoing_vz_mps']<=-1) for r in rows)
 return {'episodes':len(rows),'contacts':contact,'airborne_contacts':air,'attacks':attack,'recoveries':sum(bool(r['recovery_success']) for r in rows),'crossed_net':sum(bool(r['crossed_net']) for r in rows),'contact_rate':contact/n,'airborne_contact_rate':air/n,'attack_rate':attack/n}
try:
 a.output.mkdir(parents=True,exist_ok=True); cfg=BallEnvCfg(); cfg.scene.num_envs=a.num_envs; cfg.seed=a.seed; cfg.sim.device='cuda:0'; cfg.sim.dt=a.physics_dt; cfg.decimation=round(.01/a.physics_dt); cfg.sim.render_interval=cfg.decimation; cfg.episode_length_s=4.; cfg.arm_action_scale=2.; cfg.include_net=True; cfg.shuttle_start=(8.18,0.,1.15); cfg.shuttle_velocity=(-6.6,0.,6.82); cfg.event_goal_position=(.3,0.,1.25); cfg.event_goal_normal=(0.,0.,1.); cfg.event_goal_velocity=(0.,0.,0.)
 env=BallEnv(cfg); wrapped=RslRlVecEnvWrapper(env,clip_actions=None); config=json.loads(a.config.read_text()); runner=NativeRunner(wrapped,config['runner'],log_dir=str(a.output/'ppo'),device=env.device); runner.load(str(a.init_checkpoint.resolve()),load_optimizer=False); policy=runner.alg.policy
 default=env.robot.data.default_joint_pos[0,env.joint_ids[12:]].clone(); pre=((Q_PRE.to(env.device)-default)/2.).view(1,6); late=((Q_LATE.to(env.device)-default)/2.).view(1,6); params=PARAMS.to(env.device).repeat(a.num_envs,1); env_steps=0
 def target_action(obs):
  out=teacher_action(obs['policy'],params); t=obs['policy'][:,775]; arm=torch.zeros((a.num_envs,6),device=env.device); active=(t>=.70)&(t<1.42); arm[active]=torch.where((t[active]<1.16)[:,None],pre,late); out[:,12:]=arm; return out
 def rollout(learned=False,collect=False):
  global env_steps; env._episode_has_stepped.zero_(); env._metrics_started=False; env.episode_records.clear(); env.ball_episode_records.clear(); obs,_=wrapped.reset(); xs=[];ys=[];safe=[]
  with torch.inference_mode():
   for _ in range(400):
    y=target_action(obs)
    if collect: xs.append(obs['policy'].clone());ys.append(y.clone());safe.append((~env.body_ground_seen).clone())
    act=policy.act_inference(obs) if learned else y
    if not torch.isfinite(act).all(): raise RuntimeError('nonfinite action')
    obs,_,_,_=wrapped.step(act); env_steps+=a.num_envs
  rows=list(env.ball_episode_records); data=(torch.stack(xs),torch.stack(ys),torch.stack(safe)) if collect else None; return rows,data
 teacher_rows,data=rollout(collect=True)
 with (a.output/'teacher_episodes.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=sorted(teacher_rows[0]));w.writeheader();w.writerows(teacher_rows)
 x,y,safe=(v.clone() for v in data); val=(torch.arange(a.num_envs,device=env.device)%5==0); tm=safe&(~val)[None,:]; vm=safe&val[None,:]; train_x,train_y=x[tm],y[tm]; val_x,val_y=x[vm],y[vm]; torch.save({'observations':x.cpu(),'actions':y.cpu(),'safe':safe.cpu(),'validation_episode':val.cpu(),'seed':a.seed},a.output/'teacher_dataset.pt'); del x,y,safe,data
 policy.train(); policy.update_normalization(TensorDict({'policy':train_x},batch_size=[len(train_x)])); policy.eval(); opt=torch.optim.Adam(policy.actor.parameters(),lr=3e-4)
 report={'method':'real-PhysX high-arm teacher BC + DAgger','physics_dt_s':a.physics_dt,'teacher':summarize(teacher_rows),'learner_checks':[],'env_steps':env_steps,'config_sha256':digest(a.config),'init_checkpoint_sha256':digest(a.init_checkpoint)}; best=-1
 for rnd in range(a.rounds):
  active=torch.where((train_x[:,775]>=.65)&(train_x[:,775]<=1.8))[0]; losses=[]
  for u in range(a.updates):
   if stop: break
   n1=min(768,len(active)); idx=torch.cat((active[torch.randint(len(active),(n1,),device=env.device)],torch.randint(len(train_x),(min(256,len(train_x)),),device=env.device))); pred=policy.act_inference(TensorDict({'policy':train_x[idx]},batch_size=[len(idx)])); loss=(pred-train_y[idx]).square().mean(); opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(policy.actor.parameters(),1.);opt.step()
   if u%100==0: losses.append(float(loss))
  vmse=float((policy.act_inference(TensorDict({'policy':val_x},batch_size=[len(val_x)]))-val_y).square().mean()) if len(val_x) else None; rows,dagger=rollout(learned=True,collect=True); check=dict(round=rnd,**summarize(rows),validation_mse=vmse,training_mse_samples=losses); report['learner_checks'].append(check); report['env_steps']=env_steps
  with (a.output/f'learner_episodes_{rnd:02d}.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=sorted(rows[0]));w.writeheader();w.writerows(rows)
  runner.save_restart(a.output/f'bc_{rnd:02d}.pt',infos={'method':'real-PhysX high-arm BC','round':rnd})
  score=check['attacks']
  if score>best: best=score; runner.save_restart(a.output/'checkpoint.pt',infos={'method':'real-PhysX high-arm BC','round':rnd});report['best_round']=rnd
  (a.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n'); print('G07_HIGH_ARM_PROGRESS='+json.dumps(check,sort_keys=True),flush=True)
  if dagger is not None:
   dx,dy,ds=(v.clone() for v in dagger); mask=ds&(~val)[None,:]; train_x=torch.cat((train_x,dx[mask]));train_y=torch.cat((train_y,dy[mask]));torch.save({'observations':dx.cpu(),'teacher_actions':dy.cpu(),'safe':ds.cpu(),'seed':a.seed+100+rnd},a.output/f'dagger_{rnd:02d}.pt');del dx,dy,ds
 report['best_attacks']=best;report['best_attack_rate']=best/max(1,a.num_envs);report['checkpoint']=str(a.output/'checkpoint.pt');report['env_steps']=env_steps;(a.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
except BaseException:
 a.output.mkdir(parents=True,exist_ok=True);(a.output/'error.txt').write_text(traceback.format_exc());raise
finally:
 if env is not None: env.close()
 app.close(skip_cleanup=True)
