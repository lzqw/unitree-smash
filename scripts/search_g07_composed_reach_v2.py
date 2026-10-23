#!/usr/bin/env python3
"""Canonical learned walking followed by all-body strike reach, before shuttle rollouts.

No ball or collision success is claimed. The near-net start is an explicit
optional curriculum, not the canonical G07 ground evaluation distribution.
"""
import argparse,json,signal,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--num-envs',type=int,default=128);p.add_argument('--generations',type=int,default=12);p.add_argument('--seed',type=int,default=714101);p.add_argument('--root-x',type=float,default=0.);p.add_argument('--initial',type=Path);p.add_argument('--deep-crouch',action='store_true');p.add_argument('--vertical-seed',action='store_true');p.add_argument('--flight-objective',action='store_true');p.add_argument('--seed-bank',type=Path);p.add_argument('--resume-population',type=Path);p.add_argument('--walk-candidate',type=Path,required=True);p.add_argument('--pose-seed',type=Path,required=True);p.add_argument('--walk-seconds',type=float,default=3.4);p.add_argument('--waypoint-x',type=float,default=1.03);p.add_argument('--settle-default',action='store_true');p.add_argument('--landing-feedback',action='store_true');p.add_argument('--reposition',action='store_true');p.add_argument('--fold-arm',action='store_true');p.add_argument('--forward-seed',action='store_true');p.add_argument('--global-fraction',type=float,default=0.);AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.tasks.stand.jump_env import JumpEnvCfg
from quadsmash.tasks.stand.loaded_walk_env import LoadedWalkEnv
from quadsmash.control.go2_walk_transfer import Go2WalkTransfer
from quadsmash.control.composed_reach_teacher import composed_reach_action
from quadsmash.physics.native import quat_rotate

from quadsmash.control.dynamic_reach_teacher import LOW,HIGH,INITIAL
stopping=False
def stop(*_):
 global stopping;stopping=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)

def main():
 a.output.mkdir(parents=True,exist_ok=True);cfg=JumpEnvCfg();cfg.scene.num_envs=a.num_envs;cfg.seed=a.seed;cfg.sim.device='cuda:0';cfg.robot.init_state.pos=(a.root_x,0.,.323);cfg.sim.dt=.002;cfg.decimation=5;cfg.sim.render_interval=5;cfg.episode_length_s=a.walk_seconds+4.;cfg.event_goal_enabled=True;cfg.event_goal_position=(a.waypoint_x,0.,.3);cfg.event_goal_time_s=a.walk_seconds+1.485;cfg.event_goal_normal=(.866025404,0.,.5)
 class ReachEnv(LoadedWalkEnv):
  def _pre_physics_step(self,actions):
   self._record_action(actions);self.targets.copy_(self.robot.data.default_joint_pos);lim=self.robot.data.joint_pos_limits[:,self.joint_ids];self.targets[:,self.joint_ids]=lim[:,:,0]+.5*(self.last_action+1)*(lim[:,:,1]-lim[:,:,0])
 env=ReachEnv(cfg);w=RslRlVecEnvWrapper(env,clip_actions=None);device=env.device;low=torch.tensor(LOW,device=device);high=torch.tensor(HIGH,device=device)
 if a.deep_crouch:low[4]=-1.20;high[3]=1.15
 if a.landing_feedback:
  low=torch.cat((low,torch.tensor([-.5]*6+[1.1,0.,-.8,-.8,0.],device=device)));high=torch.cat((high,torch.tensor([.5]*6+[1.7,.8,.8,.8,.8],device=device)))
 if a.settle_default:
  if not a.landing_feedback:raise ValueError('settle-default requires landing-feedback')
  low=torch.cat((low,torch.tensor([2.05],device=device)));high=torch.cat((high,torch.tensor([2.65],device=device)))
 if not a.landing_feedback or a.settle_default:raise ValueError('v2 requires landing-feedback and no settle-default')
 low=torch.cat((low,torch.tensor([.65,.20],device=device)));high=torch.cat((high,torch.tensor([1.25,.65],device=device)))
 mean=torch.tensor(INITIAL+[0.]*6+[1.42,.25,0.,.1,.15,.95,.50],device=device).clamp(low,high);sigma=(high-low)*.18;g=torch.Generator(device=device).manual_seed(a.seed)
 if a.initial:
  prior=torch.load(a.initial,map_location=device,weights_only=False);seed_params=prior['parameters'].to(device);mean[:min(len(seed_params),len(mean))]=seed_params[:len(mean)];mean=mean.clamp(low,high);sigma=(high-low)*.10
 if a.vertical_seed:
  mean[5]=.42;mean[6]=.42;mean[8]=0.;mean[17:21]=0.;sigma=(high-low)*.15
  if a.deep_crouch:mean[3]=.85;mean[4]=-1.0
 if a.forward_seed:
  mean[5]=1.10;mean[6]=1.10;mean[8]=0.;sigma=(high-low)*.18
 limits=env.robot.data.joint_pos_limits[0,env.joint_ids];default=env.robot.data.default_joint_pos[0,env.joint_ids];geometry=json.loads((ROOT/'reports/G07/feasibility_v3_20261023/geometry.json').read_text());arm=torch.tensor(geometry['poses']['effective_face_normal_x_ge_0.9']['arm_q_rad'],device=device)
 arm=torch.load(a.pose_seed,map_location=device,weights_only=False)['arm_pose'].to(device);mean[21:27]=0.
 if len(mean)>=34:mean[27:33]=0.;mean[33]=1.42
 walk=Go2WalkTransfer(ROOT/'configs/g07_go2_walk_transfer.json',a.num_envs,device);walk_seed=torch.load(a.walk_candidate,map_location=device,weights_only=False);walk_params=walk_seed['parameters'].to(device).repeat(a.num_envs,1);walk_ticks=round(a.walk_seconds/.01);ticks=round(cfg.episode_length_s/.01)
 initial_arm=default[12:].clone()
 if len(walk_seed['parameters'])==15:initial_arm=walk_seed['parameters'][8:14].clone()
 else:initial_arm[1]=walk_seed['parameters'][8];initial_arm[2]=walk_seed['parameters'][9];initial_arm[4]=walk_seed['parameters'][11]
 target=torch.tensor([1.40,0.,1.65],device=device);target_normal=torch.tensor([.866025404,0.,.5],device=device)
 if a.resume_population:
  resume=torch.load(a.resume_population,map_location=device,weights_only=False);mean=resume['mean'].to(device);sigma=resume['sigma'].to(device);g.set_state(resume['rng'].cpu())
 from quadsmash.control.dynamic_reach_teacher import dynamic_reach_action
 def actions(obs,parameters):return composed_reach_action(obs,parameters,default,limits,arm,initial_arm)
 history=[];best_score=-float('inf');seed_bank=torch.load(a.seed_bank,map_location=device,weights_only=False)['parameters'] if a.seed_bank else None
 try:
  for generation in range(a.generations):
   parameters=(mean[None,:]+sigma[None,:]*torch.randn((a.num_envs,len(low)),generator=g,device=device)).clamp(low,high);parameters[0]=mean
   global_count=min(a.num_envs-1,round(a.num_envs*a.global_fraction))
   if global_count:parameters[-global_count:]=low[None,:]+(high-low)[None,:]*torch.rand((global_count,len(low)),generator=g,device=device)
   if generation==0 and seed_bank is not None:
    count=min(len(seed_bank),a.num_envs-1);parameters[1:count+1,:21]=seed_bank[:count,:21]
   env._episode_has_stepped.zero_();env._metrics_started=False;env.episode_records.clear();env.net_episode_records.clear();env.goal_position_w.copy_(env.scene.env_origins+torch.tensor([a.waypoint_x,0.,.3],device=device));walk.reset();obs,_=w.reset();finished=torch.zeros(a.num_envs,dtype=torch.bool,device=device);best=torch.full((a.num_envs,),-100.,device=device);position=torch.zeros((a.num_envs,3),device=device);normal=position.clone();root=position.clone();time=torch.zeros(a.num_envs,device=device);apex=torch.zeros(a.num_envs,device=device);air=torch.zeros(a.num_envs,dtype=torch.bool,device=device);predicted_velocity=torch.zeros((a.num_envs,3),device=device);predicted_margin=torch.zeros(a.num_envs,device=device);predicted_landing=torch.zeros((a.num_envs,2),device=device);prefix_net=torch.zeros_like(air);face_uv=torch.zeros((a.num_envs,2),device=device);final_root=torch.zeros((a.num_envs,3),device=device);final_velocity=final_root.clone();final_tilt=torch.zeros(a.num_envs,device=device)
   with torch.inference_mode():
    for tick in range(ticks):
     t=obs['policy'][:,775]-a.walk_seconds;d=env.robot.data;active=~finished;final_root[active]=(d.root_pos_w-env.scene.env_origins)[active];final_velocity[active]=d.root_link_lin_vel_w[active];final_tilt[active]=d.projected_gravity_b[active,:2].norm(dim=-1);tool=d.body_pos_w[:,env.racket_body_id]-env.scene.env_origins;quat=d.body_quat_w[:,env.racket_body_id];n=quat_rotate(quat,torch.tensor([1.,0.,0.],device=device).expand(a.num_envs,-1));up_y=quat_rotate(quat,torch.tensor([0.,1.,0.],device=device).expand(a.num_envs,-1));up_z=quat_rotate(quat,torch.tensor([0.,0.,1.],device=device).expand(a.num_envs,-1));delta=target[None,:]-tool-.0325*n;u=(delta*up_y).sum(1).clamp(-.095,.095);v=(delta*up_z).sum(1).clamp(-.095,.095);point=tool+.0325*n+u[:,None]*up_y+v[:,None]*up_z
     candidate=(~finished)&(t>=1.)&(t<=1.85);score=-5*(point-target).norm(dim=-1)-(n-target_normal).norm(dim=-1)+.5*env.observed_airborne.float()-10*env.body_ground_seen.float()-10*env.net_robot_seen.float()
     if a.flight_objective:
      tool_velocity=d.body_link_lin_vel_w[:,env.racket_body_id]+torch.linalg.cross(d.body_link_ang_vel_w[:,env.racket_body_id],point-tool)
      incoming=torch.tensor([-3.1,0.,-17.1],device=device)[None,:]
      outgoing=incoming-1.72*((incoming-tool_velocity)*n).sum(1,keepdim=True)*n
      landing_time=(outgoing[:,2]+torch.sqrt(outgoing[:,2].square()+19.62*(point[:,2]-.025).clamp_min(0)))/9.81
      landing=point[:,:2]+landing_time[:,None]*outgoing[:,:2]
      net_time=(1.535-point[:,0])/outgoing[:,0].clamp_min(.1)
      margin=point[:,2]+outgoing[:,2]*net_time-4.905*net_time.square()-.025-1.524
      violation=(8-outgoing.norm(dim=1)).clamp_min(0)/8+(outgoing[:,2]+1).clamp_min(0)/4+(-outgoing[:,0]).clamp_min(0)/8
      violation+=(.02-margin).clamp_min(0)*5+(2-landing[:,0]).clamp_min(0)+(landing[:,0]-3.8).clamp_min(0)+(landing[:,1].abs()-.8).clamp_min(0)
      score=-5*(point[:,:2]-target[:2]).norm(dim=1)-2*violation+.5*env.observed_airborne.float()-10*env.body_ground_seen.float()-10*env.net_robot_seen.float()
     improved=candidate&(score>best);prefix_net[improved]=env.net_robot_seen[improved]
     if a.flight_objective:predicted_velocity[improved]=outgoing[improved];predicted_margin[improved]=margin[improved];predicted_landing[improved]=landing[improved]
     best[improved]=score[improved];position[improved]=point[improved];normal[improved]=n[improved];root[improved]=(d.root_pos_w-env.scene.env_origins)[improved];time[improved]=t[improved];face_uv[improved]=torch.stack((u,v),1)[improved];air[improved]=env.observed_airborne[improved];apex=torch.where(~finished,torch.maximum(apex,d.root_pos_w[:,2]),apex)
     if tick<walk_ticks:
      act=walk.action(obs,walk_params,default,limits)
     else:
      if tick==walk_ticks:env.goal_position_w.copy_(env.scene.env_origins+target)
      local_obs={'policy':obs['policy'].clone()};local_obs['policy'][:,775]-=a.walk_seconds;act=actions(local_obs,parameters)
     obs,_,done,_=w.step(act);finished|=done.bool()
   first={}
   for row in env.episode_records:first.setdefault(row['env_id'],row)
   recovery=torch.tensor([bool(first[i]['success']) for i in range(a.num_envs)],device=device)
   first_net={}
   for row in env.net_episode_records:first_net.setdefault(row['env_id'],row)
   if len(first_net)!=a.num_envs:raise RuntimeError('Incomplete robot/net force telemetry')
   net_bad=torch.tensor([first_net[i]['robot_net_contact'] for i in range(a.num_envs)],device=device);scores=best+.05*recovery.float()-.5*net_bad.float()
   if a.reposition:
    safe=torch.tensor([not first[i]['body_landing'] for i in range(a.num_envs)],device=device);quiet=torch.tensor([first[i]['recovery_s'] for i in range(a.num_envs)],device=device)
    movement_recovery=torch.tensor([first[i]['stable_recovery'] for i in range(a.num_envs)],device=device)&safe&(final_root[:,2]>.2)&(final_tilt<.26)&(final_velocity[:,:2].norm(dim=-1)<.3)
    scores=20*movement_recovery.float()+3*quiet.clamp(max=1.)-4*(final_root[:,0]-1.).abs()-3*final_root[:,1].abs()-final_tilt-.5*final_velocity.norm(dim=-1)-4*(~safe).float()
   elite=scores.topk(max(8,a.num_envs//8)).indices;mean=.25*mean+.75*parameters[elite].mean(0);sigma=(.25*sigma+.75*parameters[elite].std(0)).clamp(min=(high-low)*.025);top=int(scores.argmax())
   result=dict(search_prediction_kind='ideal restitution and ballistic no-drag shaping only; must verify real flight' if a.flight_objective else None,predicted_outgoing_velocity_mps=predicted_velocity[top].cpu().tolist() if a.flight_objective else None,predicted_net_margin_m=float(predicted_margin[top]) if a.flight_objective else None,predicted_landing_xy_m=predicted_landing[top].cpu().tolist() if a.flight_objective else None,prefix_robot_net_contact=bool(prefix_net[top]),parameter_schema='composed-reach-v2-40',net_geometry_version=env.net_geometry_version,robot_net_contact=bool(net_bad[top]),maximum_robot_net_force_n=first_net[top]['maximum_robot_net_force_n'],first_robot_net_contact_time_s=first_net[top]['first_robot_net_contact_time_s'],first_robot_net_contact_body=first_net[top]['first_robot_net_contact_body'],net_contact_lanes=int(net_bad.sum()),effective_face_uv_m=face_uv[top].cpu().tolist(),movement_recovery=bool(movement_recovery[top]) if a.reposition else None,objective='ground reposition' if a.reposition else 'dynamic effective face reach',settle_default=a.settle_default,landing_feedback=a.landing_feedback,folded_arm=a.fold_arm,root_final_m=final_root[top].cpu().tolist(),root_final_velocity_mps=final_velocity[top].cpu().tolist(),generation=generation,score=float(scores[top]),parameters=parameters[top].cpu().tolist(),virtual_ball_center_m=position[top].cpu().tolist(),face_normal_world=normal[top].cpu().tolist(),root_at_event_m=root[top].cpu().tolist(),event_time_s=float(time[top])+a.walk_seconds,local_event_time_s=float(time[top]),feet_airborne=bool(air[top]),recovery=bool(recovery[top]),root_apex_m=float(apex[top]),maximum_root_apex_m=float(apex.max()),root_start_x_m=a.root_x,walk_seconds=a.walk_seconds,kind='canonical learned walking plus no-ball strike reach; not a real smash or final policy result')
   history.append(result);torch.save(dict(parameters=parameters.cpu(),scores=scores.cpu(),best_virtual_positions=position.cpu(),root_at_event=root.cpu(),root_apex=apex.cpu(),event_time=time.cpu(),feet_airborne=air.cpu(),recovery=recovery.cpu(),root_final=final_root.cpu(),root_final_velocity=final_velocity.cpu(),mean=mean.cpu(),sigma=sigma.cpu(),rng=g.get_state().cpu(),jump_episodes=first,net_episodes=first_net,face_normals=normal.cpu(),face_uv=face_uv.cpu()),a.output/f'population_{generation:03d}.pt')
   if float(scores[top])>best_score:best_score=float(scores[top]);torch.save(dict(parameters=parameters[top].cpu(),arm_pose=arm.cpu(),walk_parameters=walk_seed['parameters'].cpu(),walk_seconds=a.walk_seconds,walk_waypoint_x=a.waypoint_x,arm_initial_pose=initial_arm.cpu(),parameter_schema='composed-reach-v2-40',result=result),a.output/'best_candidate.pt')
   (a.output/'metrics.json').write_text(json.dumps(dict(history=history,best=max(history,key=lambda r:r['score']),env_steps=(generation+1)*a.num_envs*ticks),indent=2)+'\n');print(json.dumps(result),flush=True)
   if stopping:break
  if len(history)==a.generations:(a.output/'completed.json').write_text(json.dumps(dict(completed_generations=len(history),requested_generations=a.generations))+'\n')
 finally:env.close()

try:main()
except BaseException:
 a.output.mkdir(parents=True,exist_ok=True);error=traceback.format_exc();(a.output/'error.txt').write_text(error);print(error,file=sys.stderr,flush=True);raise
finally:app.close(skip_cleanup=True)
