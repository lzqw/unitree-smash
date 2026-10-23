#!/usr/bin/env python3
"""Real shuttle/net probe of a retained dynamic-reach controller.

The optional near-net ground start is a labeled curriculum; results cannot
replace canonical-ground formal G07 evaluation.
"""
import argparse,json,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--num-envs',type=int,default=32);p.add_argument('--physics-dt',type=float,default=.00025);AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
import numpy as np
import torch
from scipy.optimize import root
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.physics.shuttlecock import ShuttlecockModel
from quadsmash.control.dynamic_reach_teacher import dynamic_reach_action
from quadsmash.control.smash_search import add_flight_search_diagnostics,search_candidate_key
from quadsmash.tasks.stand.ball_env import BallEnvCfg
from quadsmash.tasks.stand.smash_env_v3 import FullJointSmashEnv

def main():
 a.output.mkdir(parents=True,exist_ok=True);candidate=torch.load(a.candidate,map_location='cpu',weights_only=False);record=candidate['result'];point=np.array(record['virtual_ball_center_m']);base_time=record['event_time_s'];model=ShuttlecockModel();starts=[];velocities=[];panel=[]
 for i in range(a.num_envs):
  vx=-10.-2.*(i%8);time=base_time+[-.045,-.015,.015,.045][(i//8)%4]
  def arrival(u):
   pos=np.array([u[0],point[1],1.15]);vel=np.array([vx,0.,u[1]]);steps=round(time/.002);dt=time/steps
   for _ in range(steps):pos,vel=model.integrate(pos,vel,dt)
   return pos
  solution=root(lambda u:(arrival(u)-point)[[0,2]],[point[0]+abs(vx)*time*.85,8.])
  if not solution.success or np.linalg.norm(arrival(solution.x)-point)>1e-5:raise RuntimeError('Incoming launch solve failed')
  starts.append([float(solution.x[0]),float(point[1]),1.15]);velocities.append([vx,0.,float(solution.x[1])]);panel.append(dict(env_id=i,arrival_time_s=time,shuttle_start=starts[-1],shuttle_velocity=velocities[-1]))
 cfg=BallEnvCfg();cfg.scene.num_envs=a.num_envs;cfg.seed=714201;cfg.sim.device='cuda:0';cfg.sim.dt=a.physics_dt;cfg.decimation=round(.01/a.physics_dt);cfg.sim.render_interval=cfg.decimation;cfg.episode_length_s=4.;cfg.include_net=True;cfg.robot.init_state.pos=(record['root_start_x_m'],0.,.323);cfg.event_goal_position=tuple(float(v) for v in point);cfg.event_goal_normal=tuple(record['face_normal_world']);cfg.event_goal_time_s=base_time
 class ProbeEnv(FullJointSmashEnv):
  def _write_shuttle_reset(self,ids):
   state=self.shuttle.data.default_root_state[ids].clone();state[:,:3]=self.scene.env_origins[ids]+torch.tensor(starts,device=self.device)[ids];state[:,7:10]=torch.tensor(velocities,device=self.device)[ids];self.shuttle.write_root_state_to_sim(state,env_ids=ids)
   if self.contact_events is not None:self.previous_ball_position[ids]=state[:,:3];self.previous_ball_velocity[ids]=state[:,7:10];self.previous_drag[ids]=0.
 env=ProbeEnv(cfg);w=RslRlVecEnvWrapper(env,clip_actions=None);device=env.device;parameters=candidate['parameters'].to(device).repeat(a.num_envs,1);arm=candidate['arm_pose'].to(device);default=env.robot.data.default_joint_pos[0,env.joint_ids];limits=env.robot.data.joint_pos_limits[0,env.joint_ids];obs,_=w.reset();trajectories=[[] for _ in range(a.num_envs)]
 try:
  with torch.inference_mode():
   for tick in range(400):
    ball=env.shuttle.data.root_pos_w-env.scene.env_origins;tool=env.robot.data.body_pos_w[:,env.racket_body_id]-env.scene.env_origins;velocity=env.shuttle.data.root_lin_vel_w;body=env.robot.data.root_pos_w-env.scene.env_origins
    if tick%2==0:
     for i in range(a.num_envs):trajectories[i].append(dict(time_s=tick*.01,ball_xyz=ball[i].cpu().tolist(),ball_v=velocity[i].cpu().tolist(),tool_xyz=tool[i].cpu().tolist(),root_xyz=body[i].cpu().tolist(),airborne=bool(env.observed_airborne[i]),contact=bool(env.contact_seen[i]),landed=bool(env.ordered_smash.ground_seen[i])))
    obs,_,_,_=w.step(dynamic_reach_action(obs,parameters,default,limits,arm))
  first={}
  for row in env.smash_episode_records:first.setdefault(row['env_id'],row)
  rows=[dict(first[i],launch=panel[i]) for i in range(a.num_envs)];add_flight_search_diagnostics(rows);best=max(rows,key=search_candidate_key)
  metrics=dict(kind='real PhysX curriculum probe; near-net ground start; not autonomous-policy acceptance',candidate=str(a.candidate),root_start_x_m=record['root_start_x_m'],episodes=len(rows),contact=sum(r['contact_count']>0 for r in rows),airborne_contact=sum(r['airborne_contact'] for r in rows),ordered_net=sum(r['ordered_crossed_net'] for r in rows),ordered_landing=sum(r['ordered_landed_target'] for r in rows),complete=sum(r['complete_smash'] for r in rows),recovery=sum(r['recovery_success'] for r in rows),numerical_failures=sum(r['numerical_failures'] for r in rows),best=best)
  (a.output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n');(a.output/'episodes.json').write_text(json.dumps(rows,indent=2)+'\n');(a.output/'launches.json').write_text(json.dumps(panel,indent=2)+'\n');(a.output/'trajectories.json').write_text(json.dumps(trajectories));(a.output/'completed.json').write_text(json.dumps(dict(episodes=len(rows),env_steps=400*a.num_envs))+'\n');print(json.dumps(metrics),flush=True)
 finally:env.close()

try:main()
except BaseException:
 a.output.mkdir(parents=True,exist_ok=True);error=traceback.format_exc();(a.output/'error.txt').write_text(error);print(error,file=sys.stderr,flush=True);raise
finally:app.close(skip_cleanup=True)
