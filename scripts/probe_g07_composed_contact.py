#!/usr/bin/env python3
"""Canonical walking, dynamic strike and recovery against a real initial serve."""
import argparse,hashlib,json,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--strike-policy',type=Path);p.add_argument('--recovery-scan',action='store_true');p.add_argument('--num-envs',type=int,default=32);p.add_argument('--population',type=Path);p.add_argument('--strong-serve-grid',action='store_true');p.add_argument('--population-candidates',type=int,default=8);p.add_argument('--launches',type=Path);p.add_argument('--launch-index',type=int,default=0);p.add_argument('--seed',type=int,default=715801);AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
import numpy as np
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.control.go2_walk_transfer import Go2WalkTransfer
from quadsmash.control.composed_reach_teacher import composed_reach_action
from quadsmash.control.smash_search import add_flight_search_diagnostics,search_candidate_key
from quadsmash.physics.initial_launch import solve_initial_launches
from quadsmash.tasks.stand.ball_env import BallEnvCfg
from quadsmash.tasks.stand.smash_env_v3 import FullJointSmashEnv
from quadsmash.tasks.stand.net_contact_monitor import NetContactMonitor,add_net_contact_sensors

def main():
 a.output.mkdir(parents=True,exist_ok=True);candidate=torch.load(a.candidate,map_location='cpu',weights_only=False)
 if candidate['parameter_schema'] not in ('composed-reach-v2-40','composed-reach-v3-41','composed-reach-v4-44'):raise ValueError('Wrong composed controller schema')
 inputs=[a.candidate]+([a.strike_policy] if a.strike_policy else [])+([a.population] if a.population else [])+([a.launches] if a.launches else [])
 (a.output/'input_artifacts.json').write_text(json.dumps({str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},indent=2)+'\n')
 record=candidate['result'];point=record['virtual_ball_center_m'];event_time=record['event_time_s'];n=a.num_envs
 population=None;population_lanes=None;parameter_rows=candidate['parameters'].repeat(n,1)
 points=np.tile(point,(n,1));event_times=np.full(n,event_time)
 if a.population:
  if a.launches:raise ValueError('Population search and frozen launch are separate probe modes')
  population=torch.load(a.population,map_location='cpu',weights_only=False)
  ranks=population['scores'].argsort(descending=True).tolist()
  free=[i for i in ranks if bool(population['feet_airborne'][i]) and not bool(population['prefix_robot_net'][i])]
  if not free:raise ValueError('No unassisted airborne population candidates')
  count=min(a.population_candidates,len(free),n)
  # Each selected controller sees several physically distinct initial serves.
  population_lanes=[free[i%count] for i in range(n)]
  parameter_rows=population['parameters'][population_lanes]
  points=population['best_virtual_positions'][population_lanes].numpy()
  event_times=population['event_time'][population_lanes].numpy()+candidate['walk_seconds']
 if a.launches:
  panel=json.loads(a.launches.read_text());selected=panel[a.launch_index];launches=[dict(selected,env_id=i) for i in range(n)]
 else:
  vx=[-12.,-20.,-30.,-45.,-70.,-90.,-120.,-150.] if a.strong_serve_grid else [-8.,-10.,-12.,-16.,-20.,-30.,-45.,-70.];offset=[-.015,-.005,.005,.015]
  if population is not None:
   grid=[-20.,-45.,-90.,-150.] if a.strong_serve_grid else [-12.,-20.,-30.,-45.]
   velocities=[grid[(i//count)%4] for i in range(n)]
   launch_times=event_times+np.array([[-.005,.005][(i//(4*count))%2] if n>=8*count else 0. for i in range(n)])
   launches=solve_initial_launches(points,launch_times,velocities)
  else:
   times=np.array([event_time+offset[(i//8)%4] for i in range(n)])
   launches=solve_initial_launches(points,times,[vx[i%8] for i in range(n)])
 (a.output/'launches.json').write_text(json.dumps(launches,indent=2)+'\n')
 starts=[r['shuttle_start'] for r in launches];velocities=[r['shuttle_velocity'] for r in launches]
 cfg=BallEnvCfg();cfg.scene.num_envs=n;cfg.seed=a.seed;cfg.sim.device='cuda:0';cfg.sim.dt=.00025;cfg.decimation=40;cfg.sim.render_interval=40;cfg.episode_length_s=candidate['walk_seconds']+4.;cfg.include_net=True;cfg.robot.init_state.pos=(0.,0.,.323);cfg.event_goal_position=(candidate['walk_waypoint_x'],0.,.3);cfg.event_goal_normal=tuple(float(v) for v in record['face_normal_world']);cfg.event_goal_time_s=event_time
 class ProbeEnv(NetContactMonitor,FullJointSmashEnv):
  def _setup_scene(self):
   super()._setup_scene();add_net_contact_sensors(self)
  def _write_shuttle_reset(self,ids):
   state=self.shuttle.data.default_root_state[ids].clone();state[:,:3]=self.scene.env_origins[ids]+torch.tensor(starts,device=self.device,dtype=torch.float32)[ids];state[:,7:10]=torch.tensor(velocities,device=self.device,dtype=torch.float32)[ids];self.shuttle.write_root_state_to_sim(state,env_ids=ids)
   if self.contact_events is not None:self.previous_ball_position[ids]=state[:,:3];self.previous_ball_velocity[ids]=state[:,7:10];self.previous_drag[ids]=0.
 env=ProbeEnv(cfg);w=RslRlVecEnvWrapper(env,clip_actions=None);device=env.device;walk=Go2WalkTransfer(ROOT/'configs/g07_go2_walk_transfer.json',n,device);walk_parameters=candidate['walk_parameters'].to(device).repeat(n,1);parameters=parameter_rows.to(device);arm=candidate['arm_pose'].to(device);initial_arm=candidate['arm_initial_pose'].to(device);default=env.robot.data.default_joint_pos[0,env.joint_ids];limits=env.robot.data.joint_pos_limits[0,env.joint_ids];walk_seconds=candidate['walk_seconds'];walk_ticks=round(walk_seconds/.01);ticks=round(cfg.episode_length_s/.01);obs,_=w.reset();trajectories=[[] for _ in range(n)];observations=[];actions=[];poses=[]
 learned_strike=None;strike_checkpoint=None;strike_policy_source=None
 if a.strike_policy or 'strike_policy_checkpoint' in candidate:
  from quadsmash.control.piecewise_affine import PiecewiseAffinePolicy
  strike_checkpoint=torch.load(a.strike_policy,map_location='cpu',weights_only=False) if a.strike_policy else candidate['strike_policy_checkpoint']
  strike_policy_source=str(a.strike_policy) if a.strike_policy else 'embedded learned PWA checkpoint'
  learned_strike=PiecewiseAffinePolicy(strike_checkpoint).to(device).eval()
 standing=None
 if 'recovery_parameters' in candidate:
  from quadsmash.control.go2_standing_transfer import Go2StandingTransfer
  standing=Go2StandingTransfer(ROOT/'configs/g07_go2_walk_transfer.json',n,device)
  standing_params=candidate['recovery_parameters'].to(device).repeat(n,1)
  standing_arm=candidate['recovery_arm_target'].to(device)
  standing_start=torch.full((n,),candidate['recovery_start_s'],device=device)
  if a.recovery_scan:
   generator=torch.Generator(device=device).manual_seed(a.seed+9)
   noise=torch.randn((n,5),device=device,generator=generator)*torch.tensor([.03,.03,.04,.08,0.],device=device)
   noise[:min(n,8)]=0.
   gain_scales=torch.tensor([1.,.25,.4,.6,.8,1.25,1.5,2.],device=device)
   standing_params[:,4]*=gain_scales[(torch.arange(n,device=device)//8)%8]
   standing_params=(standing_params+noise).clamp(torch.tensor([-.2,-.3,-.35,-.8,.1],device=device),torch.tensor([.5,.3,.6,.8,1.4],device=device))
   offsets=torch.tensor([-.10,-.075,-.05,-.025,0.,.025,.05,.075],device=device)
   standing_start=(standing_start+offsets[torch.arange(n,device=device)%8]).clamp(min=1.42,max=1.95)
 elif a.recovery_scan:raise ValueError('Recovery scan needs a saved standing controller')
 try:
  with torch.inference_mode():
   for tick in range(ticks):
    if tick<walk_ticks:action=walk.action(obs,walk_parameters,default,limits)
    else:
     if tick==walk_ticks:env.goal_position_w.copy_(env.scene.env_origins+torch.tensor(points,device=device,dtype=torch.float32))
     local={'policy':obs['policy'].clone()};local['policy'][:,775]-=walk_seconds;action=learned_strike(local['policy']) if learned_strike is not None else composed_reach_action(local,parameters,default,limits,arm,initial_arm)
     if standing is not None:
      active=local['policy'][:,775]>=standing_start
      standing_action=standing.action(obs,standing_params,default,limits,standing_arm,active)
      action=torch.where(active[:,None],standing_action,action)
    observations.append(obs['policy'].cpu());actions.append(action.cpu())
    if tick%2==0:
     position=env.robot.data.body_link_pos_w-env.scene.env_origins[:,None,:]
     poses.append(torch.cat((position,env.robot.data.body_link_quat_w),-1).cpu().numpy())
     ball=(env.shuttle.data.root_pos_w-env.scene.env_origins).cpu().tolist();tool=(env.robot.data.body_pos_w[:,env.racket_body_id]-env.scene.env_origins).cpu().tolist();body=(env.robot.data.root_pos_w-env.scene.env_origins).cpu().tolist();velocity=env.shuttle.data.root_lin_vel_w.cpu().tolist();contact=env.contact_seen.cpu().tolist();airborne=env.observed_airborne.cpu().tolist()
     for i in range(n):trajectories[i].append(dict(time_s=tick*.01,ball_xyz=ball[i],ball_v=velocity[i],tool_xyz=tool[i],root_xyz=body[i],contact=contact[i],airborne=airborne[i]))
    obs,_,_,_=w.step(action)
  first={};net={}
  for row in env.smash_episode_records:first.setdefault(row['env_id'],row)
  for row in env.net_episode_records:net.setdefault(row['env_id'],row)
  if len(first)!=n or len(net)!=n:raise RuntimeError('Incomplete canonical physical probe')
  rows=[dict(first[i],**{k:v for k,v in net[i].items() if k!='env_id'},launch=launches[i],recovery_parameters=standing_params[i].cpu().tolist() if standing is not None else None,recovery_start_s=float(standing_start[i]) if standing is not None else None,parameter_population_lane=population_lanes[i] if population_lanes else None) for i in range(n)];add_flight_search_diagnostics(rows);best=max(rows,key=lambda r:(not r['robot_net_contact'],*search_candidate_key(r)));index=best['env_id']
  metrics=dict(kind='canonical native walking plus searched strike teacher and physical initial serve',frozen_launch_panel=str(a.launches),launch_case=a.launch_index if a.launches else None,net_contact_sensor_version='one-source-per-environment-v2',selected_video_lane=index,controller=('learned PWA strike' if learned_strike is not None else 'searched strike')+(' plus frozen learned standing' if standing is not None else ''),strike_policy=strike_policy_source,teacher_prefix_steps=0,recovery_parameter_scan=a.recovery_scan,population_source=str(a.population),formal_policy_acceptance=False,root_start_x_m=0.,net_geometry_version=cfg.net_geometry_version,episodes=n,airborne_contact=sum(r['airborne_contact'] for r in rows),effective_face_contact=sum(r['effective_face_contact'] for r in rows),ordered_net=sum(r['ordered_crossed_net'] for r in rows),ordered_landing=sum(r['ordered_landed_target'] for r in rows),complete=sum(r['complete_smash'] and not r['robot_net_contact'] for r in rows),recovery=sum(r['recovery_success'] for r in rows),robot_net_contact=sum(r['robot_net_contact'] for r in rows),numerical_failures=sum(r['numerical_failures'] for r in rows),best=best,env_steps=n*ticks)
  np.save(a.output/'link_poses.npy',np.stack(poses)[:,index]);(a.output/'body_names.json').write_text(json.dumps(env.robot.body_names)+'\n')
  selected=dict(candidate);selected['parameters']=parameter_rows[index].cpu();selected['result']=dict(record)
  selected['result'].update(virtual_ball_center_m=points[index].tolist(),event_time_s=float(event_times[index]))
  if population is not None:
   lane=population_lanes[index]
   selected['result'].update(face_normal_world=population['face_normals'][lane].tolist(),
    root_at_event_m=population['root_at_event'][lane].tolist(),local_event_time_s=float(population['event_time'][lane]),
    effective_face_uv_m=population['face_uv'][lane].tolist(),source_population_lane=lane,source_population=str(a.population),
    predicted_outgoing_velocity_mps=population['predicted_outgoing_velocity'][lane].tolist(),
    predicted_net_margin_m=float(population['predicted_net_margin'][lane]),
    predicted_landing_xy_m=population['predicted_landing_xy'][lane].tolist())
  if standing is not None:
   selected['recovery_parameters']=standing_params[index].cpu();selected['recovery_start_s']=float(standing_start[index])
  if strike_checkpoint is not None:
   selected['strike_policy_checkpoint']=strike_checkpoint
   selected['controller_system']='frozen learned walking + learned PWA strike + frozen learned standing' if standing is not None else 'frozen learned walking + learned PWA strike'
   selected['policy_schema']='hybrid-learned-smash-v1'
  selected['selection_physical_episode']=best
  torch.save(selected,a.output/'selected_teacher.pt')
  if strike_checkpoint is not None:torch.save(selected,a.output/'selected_policy.pt')
  (a.output/'episodes.json').write_text(json.dumps(rows,indent=2)+'\n');(a.output/'jump_episodes.json').write_text(json.dumps(env.episode_records,indent=2)+'\n');(a.output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n');(a.output/'trajectories.json').write_text(json.dumps(trajectories)+'\n');(a.output/'trajectory.json').write_text(json.dumps(trajectories[index])+'\n');torch.save(dict(observations=torch.stack(observations),actions=torch.stack(actions),episodes=rows,controller=candidate,homogeneous_controller=population is None and not a.recovery_scan,controller_parameters_per_episode=parameter_rows.cpu()),a.output/'teacher_dataset.pt');(a.output/'completed.json').write_text(json.dumps(dict(episodes=n,env_steps=n*ticks))+'\n');print(json.dumps(metrics),flush=True)
 finally:env.close()

try:main()
except BaseException:
 a.output.mkdir(parents=True,exist_ok=True);error=traceback.format_exc();(a.output/'error.txt').write_text(error);print(error,file=sys.stderr,flush=True);raise
finally:app.close(skip_cleanup=True)
