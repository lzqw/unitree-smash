#!/usr/bin/env python3
"""Fast full-load dynamic reach curriculum before costly shuttle rollouts.

No ball or collision success is claimed. The near-net start is an explicit
optional curriculum, not the canonical G07 ground evaluation distribution.
"""
import argparse,json,signal,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--num-envs',type=int,default=128);p.add_argument('--generations',type=int,default=12);p.add_argument('--seed',type=int,default=714101);p.add_argument('--root-x',type=float,default=0.);p.add_argument('--initial',type=Path);p.add_argument('--settle-default',action='store_true');p.add_argument('--landing-feedback',action='store_true');p.add_argument('--reposition',action='store_true');p.add_argument('--fold-arm',action='store_true');p.add_argument('--forward-seed',action='store_true');p.add_argument('--global-fraction',type=float,default=0.);AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.tasks.stand.jump_env import JumpEnv,JumpEnvCfg
from quadsmash.physics.native import quat_rotate

from quadsmash.control.dynamic_reach_teacher import LOW,HIGH,INITIAL
stopping=False
def stop(*_):
 global stopping;stopping=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)

def main():
 a.output.mkdir(parents=True,exist_ok=True);cfg=JumpEnvCfg();cfg.scene.num_envs=a.num_envs;cfg.seed=a.seed;cfg.sim.device='cuda:0';cfg.robot.init_state.pos=(a.root_x,0.,.323);cfg.sim.dt=.002;cfg.decimation=5;cfg.sim.render_interval=5;cfg.episode_length_s=4.
 class ReachEnv(JumpEnv):
  def _pre_physics_step(self,actions):
   self._record_action(actions);self.targets.copy_(self.robot.data.default_joint_pos);lim=self.robot.data.joint_pos_limits[:,self.joint_ids];self.targets[:,self.joint_ids]=lim[:,:,0]+.5*(self.last_action+1)*(lim[:,:,1]-lim[:,:,0])
 env=ReachEnv(cfg);w=RslRlVecEnvWrapper(env,clip_actions=None);device=env.device;low=torch.tensor(LOW,device=device);high=torch.tensor(HIGH,device=device)
 if a.landing_feedback:
  low=torch.cat((low,torch.tensor([-.5]*6+[1.1,0.,-.8,-.8,0.],device=device)));high=torch.cat((high,torch.tensor([.5]*6+[1.7,.8,.8,.8,.8],device=device)))
 if a.settle_default:
  if not a.landing_feedback:raise ValueError('settle-default requires landing-feedback')
  low=torch.cat((low,torch.tensor([2.05],device=device)));high=torch.cat((high,torch.tensor([2.65],device=device)))
 mean=torch.tensor(INITIAL+([0.]*6+[1.42,.25,0.,.1,.15] if a.landing_feedback else [])+([2.3] if a.settle_default else []),device=device).clamp(low,high);sigma=(high-low)*.18;g=torch.Generator(device=device).manual_seed(a.seed)
 if a.initial:
  prior=torch.load(a.initial,map_location=device,weights_only=False);seed_params=prior['parameters'].to(device);mean[:len(seed_params)]=seed_params;mean=mean.clamp(low,high);sigma=(high-low)*.10
 if a.forward_seed:
  mean[5]=1.10;mean[6]=1.10;mean[8]=0.;sigma=(high-low)*.18
 limits=env.robot.data.joint_pos_limits[0,env.joint_ids];default=env.robot.data.default_joint_pos[0,env.joint_ids];geometry=json.loads((ROOT/'reports/G07/feasibility_v3_20261023/geometry.json').read_text());arm=torch.tensor(geometry['poses']['effective_face_normal_x_ge_0.9']['arm_q_rad'],device=device)
 if a.fold_arm:arm=default[12:].clone()
 target=torch.tensor([1.4,0.,1.65],device=device);target_normal=torch.tensor([.96936,0.,.24565],device=device)
 from quadsmash.control.dynamic_reach_teacher import dynamic_reach_action
 def actions(obs,parameters):return dynamic_reach_action(obs,parameters,default,limits,arm)
 history=[];best_score=-float('inf')
 try:
  for generation in range(a.generations):
   parameters=(mean[None,:]+sigma[None,:]*torch.randn((a.num_envs,len(low)),generator=g,device=device)).clamp(low,high);parameters[0]=mean
   global_count=min(a.num_envs-1,round(a.num_envs*a.global_fraction))
   if global_count:parameters[-global_count:]=low[None,:]+(high-low)[None,:]*torch.rand((global_count,len(low)),generator=g,device=device)
   env._episode_has_stepped.zero_();env._metrics_started=False;env.episode_records.clear();obs,_=w.reset();finished=torch.zeros(a.num_envs,dtype=torch.bool,device=device);best=torch.full((a.num_envs,),-100.,device=device);position=torch.zeros((a.num_envs,3),device=device);normal=position.clone();root=position.clone();time=torch.zeros(a.num_envs,device=device);apex=torch.zeros(a.num_envs,device=device);air=torch.zeros(a.num_envs,dtype=torch.bool,device=device);final_root=torch.zeros((a.num_envs,3),device=device);final_velocity=final_root.clone();final_tilt=torch.zeros(a.num_envs,device=device)
   with torch.inference_mode():
    for _ in range(400):
     t=obs['policy'][:,775];d=env.robot.data;active=~finished;final_root[active]=(d.root_pos_w-env.scene.env_origins)[active];final_velocity[active]=d.root_link_lin_vel_w[active];final_tilt[active]=d.projected_gravity_b[active,:2].norm(dim=-1);tool=d.body_pos_w[:,env.racket_body_id]-env.scene.env_origins;quat=d.body_quat_w[:,env.racket_body_id];n=quat_rotate(quat,torch.tensor([1.,0.,0.],device=device).expand(a.num_envs,-1));up_y=quat_rotate(quat,torch.tensor([0.,1.,0.],device=device).expand(a.num_envs,-1));up_z=quat_rotate(quat,torch.tensor([0.,0.,1.],device=device).expand(a.num_envs,-1));point=tool+.0325*n+.095*torch.sign(up_y[:,2:])*up_y+.095*torch.sign(up_z[:,2:])*up_z
     candidate=(~finished)&(t>=1.)&(t<=1.85);score=-5*(point-target).norm(dim=-1)-(n-target_normal).norm(dim=-1)+.5*env.observed_airborne.float()-.4*env.body_ground_seen.float();improved=candidate&(score>best);best[improved]=score[improved];position[improved]=point[improved];normal[improved]=n[improved];root[improved]=(d.root_pos_w-env.scene.env_origins)[improved];time[improved]=t[improved];air[improved]=env.observed_airborne[improved];apex=torch.where(~finished,torch.maximum(apex,d.root_pos_w[:,2]),apex)
     obs,_,done,_=w.step(actions(obs,parameters));finished|=done.bool()
   first={}
   for row in env.episode_records:first.setdefault(row['env_id'],row)
   recovery=torch.tensor([bool(first[i]['success']) for i in range(a.num_envs)],device=device);scores=best+.15*recovery.float()
   if a.reposition:
    safe=torch.tensor([not first[i]['body_landing'] for i in range(a.num_envs)],device=device);quiet=torch.tensor([first[i]['recovery_s'] for i in range(a.num_envs)],device=device)
    movement_recovery=torch.tensor([first[i]['stable_recovery'] for i in range(a.num_envs)],device=device)&safe&(final_root[:,2]>.2)&(final_tilt<.26)&(final_velocity[:,:2].norm(dim=-1)<.3)
    scores=20*movement_recovery.float()+3*quiet.clamp(max=1.)-4*(final_root[:,0]-1.).abs()-3*final_root[:,1].abs()-final_tilt-.5*final_velocity.norm(dim=-1)-4*(~safe).float()
   elite=scores.topk(max(8,a.num_envs//8)).indices;mean=.25*mean+.75*parameters[elite].mean(0);sigma=(.25*sigma+.75*parameters[elite].std(0)).clamp(min=(high-low)*.025);top=int(scores.argmax())
   result=dict(movement_recovery=bool(movement_recovery[top]) if a.reposition else None,objective='ground reposition' if a.reposition else 'dynamic effective face reach',settle_default=a.settle_default,landing_feedback=a.landing_feedback,folded_arm=a.fold_arm,root_final_m=final_root[top].cpu().tolist(),root_final_velocity_mps=final_velocity[top].cpu().tolist(),generation=generation,score=float(scores[top]),parameters=parameters[top].cpu().tolist(),virtual_ball_center_m=position[top].cpu().tolist(),face_normal_world=normal[top].cpu().tolist(),root_at_event_m=root[top].cpu().tolist(),event_time_s=float(time[top]),feet_airborne=bool(air[top]),recovery=bool(recovery[top]),root_apex_m=float(apex[top]),maximum_root_apex_m=float(apex.max()),root_start_x_m=a.root_x,kind='no-ball dynamic reach curriculum; not a smash demonstration or policy result')
   history.append(result);torch.save(dict(parameters=parameters.cpu(),scores=scores.cpu(),best_virtual_positions=position.cpu(),root_at_event=root.cpu(),root_apex=apex.cpu(),event_time=time.cpu(),feet_airborne=air.cpu(),recovery=recovery.cpu(),root_final=final_root.cpu(),root_final_velocity=final_velocity.cpu(),mean=mean.cpu(),sigma=sigma.cpu(),rng=g.get_state().cpu(),jump_episodes=first),a.output/f'population_{generation:03d}.pt')
   if float(scores[top])>best_score:best_score=float(scores[top]);torch.save(dict(parameters=parameters[top].cpu(),arm_pose=arm.cpu(),result=result),a.output/'best_candidate.pt')
   (a.output/'metrics.json').write_text(json.dumps(dict(history=history,best=max(history,key=lambda r:r['score']),env_steps=(generation+1)*a.num_envs*400),indent=2)+'\n');print(json.dumps(result),flush=True)
   if stopping:break
  if len(history)==a.generations:(a.output/'completed.json').write_text(json.dumps(dict(completed_generations=len(history),requested_generations=a.generations))+'\n')
 finally:env.close()

try:main()
except BaseException:
 a.output.mkdir(parents=True,exist_ok=True);error=traceback.format_exc();(a.output/'error.txt').write_text(error);print(error,file=sys.stderr,flush=True);raise
finally:app.close(skip_cleanup=True)
