#!/usr/bin/env python3
"""Search actual loaded walking as a bridge from canonical ground to interception."""
import argparse,json,signal,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--num-envs',type=int,default=128);p.add_argument('--generations',type=int,default=16);p.add_argument('--seed',type=int,default=714401);p.add_argument('--seconds',type=float,default=4.);p.add_argument('--initial',type=Path);p.add_argument('--balanced',action='store_true');AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.tasks.stand.jump_env import JumpEnv,JumpEnvCfg
from quadsmash.control.loaded_walk_teacher import loaded_walk_action,LOW,HIGH,INITIAL
stopping=False
def stop(*_):
 global stopping;stopping=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)

def main():
 a.output.mkdir(parents=True,exist_ok=True);cfg=JumpEnvCfg();cfg.scene.num_envs=a.num_envs;cfg.seed=a.seed;cfg.sim.device='cuda:0';cfg.episode_length_s=a.seconds;cfg.event_goal_enabled=True;cfg.event_goal_position=(1.0,0.,.3);cfg.event_goal_time_s=2.7
 class WalkEnv(JumpEnv):
  def _pre_physics_step(self,actions):
   self._record_action(actions);self.targets.copy_(self.robot.data.default_joint_pos);lim=self.robot.data.joint_pos_limits[:,self.joint_ids];self.targets[:,self.joint_ids]=lim[:,:,0]+.5*(self.last_action+1)*(lim[:,:,1]-lim[:,:,0])
 env=WalkEnv(cfg);w=RslRlVecEnvWrapper(env,clip_actions=None);d=env.device;lo=torch.tensor(LOW,device=d);hi=torch.tensor(HIGH,device=d);mean=torch.tensor(INITIAL,device=d);sigma=(hi-lo)*.20;g=torch.Generator(device=d).manual_seed(a.seed);default=env.robot.data.default_joint_pos[0,env.joint_ids];limits=env.robot.data.joint_pos_limits[0,env.joint_ids];history=[];best=-float('inf');ticks=round(a.seconds/.01)
 if a.balanced:
  lo[-1]=3.;hi[-1]=a.seconds-1.4;lo[6]=-.06;hi[6]=.06;mean[6]=0.;mean[-1]=a.seconds-1.5;sigma=(hi-lo)*.20
 if a.initial:
  prior=torch.load(a.initial,map_location=d,weights_only=False);mean=prior['parameters'].to(d).clamp(lo,hi);sigma=(hi-lo)*.15
  if a.balanced:mean[6]=0.;mean[-1]=a.seconds-1.5
 try:
  for generation in range(a.generations):
   params=(mean[None,:]+sigma[None,:]*torch.randn((a.num_envs,len(lo)),generator=g,device=d)).clamp(lo,hi);params[0]=mean;env._episode_has_stepped.zero_();env._metrics_started=False;env.episode_records.clear();obs,_=w.reset();finished=torch.zeros(a.num_envs,dtype=torch.bool,device=d);last=torch.zeros((a.num_envs,3),device=d);vel=last.clone();tilt=torch.zeros(a.num_envs,device=d);bad=finished.clone()
   with torch.inference_mode():
    for _ in range(ticks):
     active=~finished;last[active]=(env.robot.data.root_pos_w-env.scene.env_origins)[active];vel[active]=env.robot.data.root_link_lin_vel_w[active];tilt[active]=env.robot.data.projected_gravity_b[active,:2].norm(dim=-1);bad|=env.body_ground_seen;obs,_,done,_=w.step(loaded_walk_action(obs,params,default,limits,balanced=a.balanced));finished|=done.bool()
   valid=(~bad)&(last[:,2]>.2)&(tilt<.26);scores=100*valid.float()+5*last[:,0].clamp(max=1.)-5*(1-last[:,0]).abs()-4*last[:,1].abs()-3*tilt-vel.norm(dim=-1);elite=scores.topk(max(8,a.num_envs//8)).indices;mean=.25*mean+.75*params[elite].mean(0);sigma=(.25*sigma+.75*params[elite].std(0)).clamp(min=(hi-lo)*.02);i=int(scores.argmax());row=dict(generation=generation,score=float(scores[i]),parameters=params[i].cpu().tolist(),root_final_m=last[i].cpu().tolist(),root_velocity_mps=vel[i].cpu().tolist(),upright_no_body_contact=bool(valid[i]),valid_lanes=int(valid.sum()),balanced=a.balanced,seconds=a.seconds,kind='physical full-load walking search, not a learned smash policy')
   if float(scores[i])>best:best=float(scores[i]);torch.save(dict(parameters=params[i].cpu(),result=row),a.output/'best_candidate.pt')
   history.append(row);(a.output/'metrics.json').write_text(json.dumps(dict(history=history,best=max(history,key=lambda r:r['score']),env_steps=(generation+1)*ticks*a.num_envs),indent=2)+'\n');torch.save(dict(parameters=params.cpu(),root_final=last.cpu(),valid=valid.cpu(),scores=scores.cpu()),a.output/f'population_{generation:03d}.pt');print(json.dumps(row),flush=True)
   if stopping:break
  if len(history)==a.generations:(a.output/'completed.json').write_text(json.dumps(dict(completed_generations=len(history)))+'\n')
 finally:env.close()

try:main()
except BaseException:
 a.output.mkdir(parents=True,exist_ok=True);error=traceback.format_exc();(a.output/'error.txt').write_text(error);print(error,file=sys.stderr,flush=True);raise
finally:app.close(skip_cleanup=True)
