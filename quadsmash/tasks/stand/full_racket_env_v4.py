"""Full racket, suspended regulation net, court-scale rally and legal events."""
from pathlib import Path
import itertools
import os
import numpy as np
import torch
from pxr import Usd,UsdPhysics
from isaaclab.sensors import ContactSensor,ContactSensorCfg
from isaacsim.core.utils.stage import get_current_stage

from .ball_env import BallEnv,BallEnvCfg,PHYSICS
from .coupled_env import ROOT
from .visual_geometry import collision_proxy_corners
from quadsmash.physics.native import quat_rotate
from quadsmash.physics.full_racket_v4 import load_racket,parts,string_prism_gap,swept_string_prism,string_polygon,string_strips,apply_neutral_arm_reset,RacketPlaneSupport
from quadsmash.physics.court_v4 import load_court,VERSION,top_height
from quadsmash.physics.racket_plane_geometry_v4 import RacketPlaneGeometry
from quadsmash.physics.rally_rules_v4 import RallyRulesV4


def full_racket_cfg(num_envs=16,physics_dt=.0000625):
    cfg=BallEnvCfg();racket,court=load_racket(),load_court()
    # The isolated SuperMimic worktree contains the learner code, while the
    # frozen G07 execution root owns the measured USD/configuration assets.
    # Keep asset resolution on the explicit execution root used by the task
    # contract instead of silently resolving through the learner worktree.
    asset_root = Path(os.environ.get('QUADSMASH_G07_EXECUTION_ROOT', ROOT)).expanduser().resolve()
    cfg.robot.spawn.usd_path=str(asset_root/racket['output_asset'])
    cfg.scene.num_envs,cfg.scene.env_spacing=num_envs,court['environment_spacing_m']
    cfg.ground_size_m=(4096.,4096.,.1)
    cfg.sim.device='cuda:0';cfg.sim.dt=physics_dt
    cfg.decimation=cfg.sim.render_interval=round(.01/physics_dt)
    cfg.episode_length_s=8.;cfg.include_net=True
    cfg.net_geometry_version=VERSION
    cfg.net_height_m,cfg.net_side_height_m=court['net_center_top_m'],court['net_side_top_m']
    cfg.net_profile_segments=court['net_profile_segments']
    cfg.shuttle_start=(6.,0.,.025);cfg.shuttle_velocity=(0.,0.,0.)
    cfg.event_goal_time_s=2.75;cfg.event_goal_position=(1.10,0.,1.80)
    cfg.event_goal_normal=(1.,0.,0.)
    cfg.sweep_audit_iterations=4096
    return apply_neutral_arm_reset(cfg)


class FullRacketEnvV4(BallEnv):
    action_family='absolute-joint-limits-full-racket-v4'
    def __init__(self,cfg,*,cases=None,contact_flight_recovery=None,**kwargs):
        self.contact_recovery=None
        if contact_flight_recovery not in (None,'contact-flight-recovery-v1'):
            raise ValueError('Unknown explicit recovery association version')
        self.racket_config,self.court_config=load_racket(),load_court()
        self.rally=None;self._v4_ready=False;self.rally_records=[];self.contact_records=[]
        self.ball_observation_enabled=torch.zeros(cfg.scene.num_envs,dtype=torch.bool,device=cfg.sim.device)
        self.cases=cases
        super().__init__(cfg,**kwargs)
        n,device=self.num_envs,self.device
        self.rally=RallyRulesV4(n,device,self.court_config,self.cfg.shuttle_radius_m)
        self.launch_done=torch.zeros(n,dtype=torch.bool,device=device)
        self.launch_count=torch.zeros(n,dtype=torch.long,device=device)
        self.launch_frame=torch.full((n,),-1,dtype=torch.long,device=device)
        self.launch_signals_ready=torch.zeros(n,dtype=torch.bool,device=device)
        self.episode_index=torch.full((n,),-1,dtype=torch.long,device=device)
        self.flange_id=self.robot.find_bodies('link6')[0][0]
        ids,points,paths=collision_proxy_corners(get_current_stage(),self.robot)
        self.safety_body_ids=torch.tensor(ids,device=device)
        self.safety_points=torch.tensor(points,dtype=torch.float32,device=device)[None]
        self.safety_proxy_paths=paths
        clamp=next(part for part in parts(self.racket_config) if part['name']=='Clamp')
        self.clamp_center=torch.tensor(clamp['center'],dtype=torch.float32,device=device)
        self.clamp_half=torch.tensor(clamp['size']/2,dtype=torch.float32,device=device)
        racket_points=[]
        for part in parts(self.racket_config):
            if part['name']=='Clamp' or not part['collision']:continue
            if part['kind']=='string_prism':
                half=np.array([self.racket_config['assumptions']['string_thickness_m']/2,
                               *self.racket_config['assumptions']['strings_semiaxes_yz_m']])
            elif part['kind'] in ('cylinder','capsule'):
                half=np.abs(part['axis'])*part['length']/2+part['radius']
            else:half=part['size']/2
            racket_points.extend(part['center']+np.array(list(itertools.product((-1,1),repeat=3)))*half)
        self.racket_points=torch.tensor(np.array(racket_points),dtype=torch.float32,device=device)[None]
        self.racket_plane_support=RacketPlaneSupport(self.racket_config,device)
        self.racket_plane_geometry=RacketPlaneGeometry(self.racket_config,self.court_config)
        self.racket_plane_records=[];self.racket_plane_query_calls=0
        self.racket_plane_gpu_broadphase_calls=0
        self.racket_plane_gpu_broadphase_candidates=0
        self.first_tool_normal=torch.zeros((n,3),device=device)
        self.first_tool_velocity=torch.zeros((n,3),device=device)
        self.hit_velocity_error=torch.zeros(n,device=device)
        self.first_string_point=torch.full((n,3),float('nan'),device=device)
        self.first_string_time=torch.full((n,),float('nan'),device=device)
        self._jump_metric_stride=max(1,round(.001/self.physics_dt))
        if contact_flight_recovery is not None:
            from .contact_flight_recovery_v5 import ContactFlightRecoveryV5
            from .jump_env import PROTOCOL
            self.contact_recovery=ContactFlightRecoveryV5(n,device,PROTOCOL)
        self._v4_ready=True

    def _consume_jump_frame(self,frame):
        previous=getattr(self,'_jump_physics_frame',-1)
        metric_previous=getattr(self,'_jump_metric_previous_frame',frame-1)
        super()._consume_jump_frame(frame)
        if self.contact_recovery is not None and frame>previous and frame!=0:
            self.contact_recovery.metric(self,self.stable_recovery,(frame-metric_previous)*self.physics_dt)

    @property
    def rally_recovery_success(self):
        if self.contact_recovery is None:return self.jump_success
        return self.contact_recovery.success(self)&(self.robot.data.root_pos_w[:,2]>=.12)

    def _racket_over_net_geometry(self,tool,quaternion,fully_own):
        """Use live GPU poses, not stale DirectGPU CPU scene-query poses."""
        over=self.rally.followthrough_crossed&(~self.rally.followthrough_finished)&(~fully_own)
        candidates=self.rally.in_play&(~fully_own)&(~over)
        broad=self.racket_plane_geometry.gpu_broadphase_over_net(
            tool-self.scene.env_origins,quaternion)
        self.racket_plane_gpu_broadphase_calls+=1
        self.racket_plane_gpu_broadphase_candidates+=int((candidates&broad).sum().item())
        ids=torch.where(candidates&broad)[0].cpu().tolist()
        if not ids:return over
        positions=(tool[ids]-self.scene.env_origins[ids]).cpu().numpy()
        rotations=quaternion[ids].cpu().numpy()
        for index,i in enumerate(ids):
            witness=self.racket_plane_geometry.first_above_net_overlap(positions[index],rotations[index])
            self.racket_plane_query_calls+=1
            if witness:
                over[i]=True
                self.racket_plane_records.append(dict(env_id=i,episode_index=int(self.episode_index[i]),
                    time_s=float(self.ball_clock[i]),**witness))
        return over

    def _setup_scene(self):
        super()._setup_scene()
        stage=get_current_stage()
        paths=[str(p.GetPath()).replace('/env_0/','/env_.*/')
               for p in Usd.PrimRange(stage.GetPrimAtPath('/World/envs/env_0/Robot'),Usd.TraverseInstanceProxies())
               if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        self.net_robot_body_paths=paths
        self.scene.sensors['inverse_net']=ContactSensor(ContactSensorCfg(
            prim_path='/World/envs/env_.*/Net',filter_prim_paths_expr=[*paths,'/World/envs/env_.*/Shuttle'],
            force_threshold=.001,track_contact_points=True,max_contact_data_count_per_prim=128))

    def _tool_gap(self,position,center,quaternion):
        return string_prism_gap(position,center,quaternion,self.racket_config,self.cfg.shuttle_radius_m)

    def _tool_sweep(self,start,end,center_start,center_end,quat_start,quat_end):
        return swept_string_prism(start,end,center_start,center_end,quat_start,quat_end,
                                  self.racket_config,self.cfg.shuttle_radius_m,self.cfg.sweep_audit_iterations)

    def _sweep_geometry_spec(self):
        return dict(geometry_version=self.racket_config['version'],
                    kind=self.racket_config.get('string_collision_representation','32gon-prism'),
                    strips=string_strips(self.racket_config))

    @torch.inference_mode()
    def initialize_rally_launch(self,ids,position=None,velocity=None):
        state=self.shuttle.data.root_state_w[ids].clone()
        if position is None:
            rows=[self.cases[int(i)]['launch'] for i in ids.cpu().tolist()]
            position=torch.tensor([row['position_m'] for row in rows],device=self.device)
            velocity=torch.tensor([row['velocity_mps'] for row in rows],device=self.device)
        state[:,:3]=position+self.scene.env_origins[ids]
        state[:,3:7]=torch.tensor([1.,0,0,0],device=self.device)
        state[:,7:10]=velocity;state[:,10:]=0.
        self.shuttle.write_root_state_to_sim(state,env_ids=ids)
        self.contact_events.reset(ids)
        for name in ('contact_seen','airborne_contact_seen','body_contact_seen','crossed_net','landed_target','landed'):
            getattr(self,name)[ids]=False
        self.previous_ball_position[ids]=state[:,:3];self.previous_ball_velocity[ids]=velocity
        self.previous_drag[ids]=0.;self.sweep_ready[ids]=False;self.pending_sweep_age[ids]=-1
        self.launch_done[ids]=self.ball_observation_enabled[ids]=True
        self.launch_count[ids]+=1
        self.launch_frame[ids]=self._sim_step_counter
        self.launch_signals_ready[ids]=False
        self.rally.launch(ids,self.ball_clock)
        # Contact buffers still describe the last preparation substep. Do not
        # treat that old floor contact as a landing of the newly launched ball.
        self.rally.in_play[ids]=False

    def _pre_physics_step(self,actions):
        self.action_delta.copy_((actions.clamp(-1,1)-self.last_action).square().sum(1))
        self._record_action(actions)
        self.action_clip_count+=(actions.abs()>1).sum(1)
        limits=self.robot.data.joint_pos_limits[:,self.joint_ids]
        targets=limits[...,0]+.5*(self.last_action+1)*(limits[...,1]-limits[...,0])
        self.targets.copy_(self.robot.data.default_joint_pos);self.targets[:,self.joint_ids]=targets
        self._contact_bonus.zero_()
        if self._v4_ready and self.cases is not None:
            clock=self.episode_length_buf.float()*self.step_dt
            ids=torch.where((~self.launch_done)&(clock>=self.court_config['rally_preparation_seconds']-1e-6))[0]
            if len(ids):self.initialize_rally_launch(ids)

    def _consume_physics_frame(self,frame):
        if self.contact_events is None or frame<=self._last_physics_frame:return
        previous=self.previous_ball_position.clone();counts=self.contact_events.count.clone()
        super()._consume_physics_frame(frame)
        if not self._v4_ready or frame==0:return
        fresh=self.launch_done&(~self.launch_signals_ready)&(frame>self.launch_frame)
        self.rally.in_play[fresh]=True;self.launch_signals_ready[fresh]=True
        d=self.robot.data;tool=d.body_link_pos_w[:,self.racket_body_id];q=d.body_link_quat_w[:,self.racket_body_id]
        forces=self.shuttle_contact.data.force_matrix_w[:,0]
        net=self.scene.sensors['inverse_net'].data.force_matrix_w[:,0]
        racket_force,ground_force=forces[:,0],forces[:,1]
        touching=racket_force.norm(dim=-1)>PHYSICS['contact_force_threshold_n']
        point=self.shuttle_contact.data.contact_pos_w[:,0,0]
        local=quat_rotate(torch.cat((q[:,:1],-q[:,1:]),-1),torch.nan_to_num(point)-tool)
        polygon=torch.as_tensor(string_polygon(self.racket_config),dtype=local.dtype,device=self.device)
        edges=polygon.roll(-1,0)-polygon
        delta=local[:,None,1:]-polygon
        inside=(edges[:,0]*delta[...,1]-edges[:,1]*delta[...,0]>=0).all(-1)
        string=inside&(local[:,0].abs()<=self.racket_config['assumptions']['string_thickness_m']/2+.001)
        a=self.racket_config['assumptions'];elliptic=((local[:,1]/(a['head_outer_width_m']/2))**2+(local[:,2]/(a['head_outer_length_m']/2))**2)
        rim=(elliptic>.75)&(elliptic<1.2)&(local[:,0].abs()<.006)&(~string)
        clamp=((local-self.clamp_center).abs()<=self.clamp_half+.001).all(-1)
        residual=self.shuttle_contact.data.net_forces_w[:,0]-racket_force-ground_force+net[:,-1]
        body_touch=(residual.norm(dim=-1)>.001)|(touching&clamp)
        safety=quat_rotate(d.body_link_quat_w[:,self.safety_body_ids,None,:],self.safety_points)+d.body_link_pos_w[:,self.safety_body_ids,None,:]-self.scene.env_origins[:,None,None,:]
        above=safety[...,2]>=top_height(safety[...,1],self.court_config)
        beyond=safety[...,0]>self.court_config['net_plane_x_m']
        body_over=(above&beyond).any((1,2));engineering=beyond.any((1,2))
        racket_maximum_x=self.racket_plane_support.maximum_x(tool-self.scene.env_origins,q)
        racket_fully_own=racket_maximum_x<=self.court_config['net_plane_x_m']
        racket_over=self._racket_over_net_geometry(tool,q,racket_fully_own)
        normal=quat_rotate(q,torch.tensor([1.,0,0],device=self.device).expand(self.num_envs,-1))
        head_velocity=d.body_link_lin_vel_w[:,self.racket_body_id]
        events=(self.contact_events.count>counts)&self.rally.in_play
        first=events&(~self.rally.had_contact)
        first_strings=touching&string&self.rally.in_play&torch.isnan(self.first_string_time)
        self.first_string_point[first_strings]=(point-self.scene.env_origins)[first_strings]
        self.first_string_time[first_strings]=self.ball_clock[first_strings]
        self.first_tool_normal[first]=normal[first];self.first_tool_velocity[first]=head_velocity[first]
        same_pair=((self.rally.frame_hit&string)|(self.rally.string_hit&rim))
        same_stroke=(same_pair&((self.ball_clock-self.rally.contact_time)<.04)&
                     ((normal*self.first_tool_normal).sum(-1)>np.cos(np.deg2rad(15)))&
                     ((head_velocity*self.first_tool_velocity).sum(-1)>=0))
        separated=(~touching)&(self._tool_gap(self.shuttle.data.root_pos_w,tool,q)>PHYSICS['separation_hysteresis_m'])
        recovery_active=self.rally.in_play.clone() if self.contact_recovery is not None else None
        self.rally.advance(previous_position=previous-self.scene.env_origins,
            position=self.shuttle.data.root_pos_w-self.scene.env_origins,velocity=self.shuttle.data.root_lin_vel_w,
            time=self.ball_clock,dt=self.physics_dt,racket_contact_event=self.contact_events.count>counts,
            racket_touching=touching,separated=separated,contact_point=point-self.scene.env_origins,
            string_contact=string,frame_contact=rim,airborne=self.observed_airborne,
            ground_touch=ground_force.norm(dim=-1)>.001,robot_net_touch=net[:,:-1].norm(dim=-1).sum(-1)>.1,
            body_ball_touch=body_touch,shuttle_net_touch=net[:,-1].norm(dim=-1)>.001,
            racket_over_net=racket_over,body_over_net=body_over,engineering_body_crossing=engineering,
            racket_fully_own_side=racket_fully_own,
            same_stroke_frame_string=same_stroke)
        record_events=events.any()
        if self.contact_recovery is not None:
            recovery_bind=(recovery_active&touching&string&self.observed_airborne&self.rally.legal_initial_contact&(~self.contact_recovery.bound))
            record_events|=recovery_bind.any()
        if bool(record_events):
            if self.contact_recovery is not None:
                self.contact_recovery.contact(self,legal_air_strings_touching=recovery_bind,
                    time=self.ball_clock,rally_contact_time=self.rally.contact_time)
            contact_velocity=head_velocity+torch.linalg.cross(d.body_link_ang_vel_w[:,self.racket_body_id],torch.nan_to_num(point)-tool)
            expected=d.body_link_lin_vel_w[:,self.flange_id]+torch.linalg.cross(d.body_link_ang_vel_w[:,self.flange_id],torch.nan_to_num(point)-d.body_link_pos_w[:,self.flange_id])
            self.hit_velocity_error[first]=(contact_velocity-expected).norm(dim=-1)[first]
            for i in torch.where(events)[0].cpu().tolist():
                self.contact_records.append(dict(env_id=i,episode_index=int(self.episode_index[i]),
                    event_index=int(self.rally.contact_count[i]),first_racket_contact=bool(first[i]),
                    same_stroke_frame_string=bool(same_stroke[i]),time_s=float(self.ball_clock[i]),
                    point_m=(point[i]-self.scene.env_origins[i]).cpu().tolist(),string_contact=bool(string[i]),
                    frame_contact=bool(rim[i]),fixture_contact=bool(clamp[i]),
                    point_velocity_mps=contact_velocity[i].cpu().tolist(),face_normal=normal[i].cpu().tolist(),
                    body_link_poses_local=torch.cat((d.body_link_pos_w[i]-self.scene.env_origins[i],
                        d.body_link_quat_w[i]),-1).cpu().tolist(),
                    ball_position_m=(self.shuttle.data.root_pos_w[i]-self.scene.env_origins[i]).cpu().tolist(),
                    ball_velocity_mps=self.shuttle.data.root_lin_vel_w[i].cpu().tolist(),
                    root_link_position_m=(d.root_link_pos_w[i]-self.scene.env_origins[i]).cpu().tolist(),
                    root_com_position_m=(d.body_com_pos_w[i,0]-self.scene.env_origins[i]).cpu().tolist(),
                    root_com_linear_velocity_mps=d.root_com_lin_vel_w[i].cpu().tolist(),
                    root_angular_velocity_rad_s=d.root_com_ang_vel_w[i].cpu().tolist(),
                    flange_position_m=(d.body_link_pos_w[i,self.flange_id]-self.scene.env_origins[i]).cpu().tolist(),
                    flange_linear_velocity_mps=d.body_link_lin_vel_w[i,self.flange_id].cpu().tolist(),
                    flange_angular_velocity_rad_s=d.body_link_ang_vel_w[i,self.flange_id].cpu().tolist(),
                    joint_position_rad=d.joint_pos[i,self.joint_ids].cpu().tolist(),
                    joint_velocity_rad_s=d.joint_vel[i,self.joint_ids].cpu().tolist(),
                    commanded_joint_position_rad=self.targets[i,self.joint_ids].cpu().tolist(),
                    legal_initial_contact=bool(self.rally.legal_initial_contact[i]),
                    fault_flags={name:bool(getattr(self.rally,name)[i]) for name in self.rally.flags
                                 if name.endswith('_fault') or name=='engineering_body_crossing'}))

    def _reset_idx(self,ids):
        if self._v4_ready:
            complete=ids[self._episode_has_stepped[ids]]
            for i in complete.cpu().tolist():
                row={name:bool(getattr(self.rally,name)[i]) for name in self.rally.flags}
                row.update(env_id=i,episode_index=int(self.episode_index[i]),
                    # Preserve the immutable evaluation-case identity through
                    # the asynchronous reset record.  Frozen return
                    # monitoring compares this identity across refreshes.
                    case_id=(self.cases[i].get('case_id', f'ENV{i}')
                             if self.cases is not None else f'ENV{i}'),
                    episode_timeout=bool(self.reset_time_outs[i]),
                    episode_duration_s=float(self.episode_length_buf[i] * self.step_dt),
                    complete_smash=bool(self.rally.complete(self.rally_recovery_success)[i]),
                    complete5=bool(self.rally.complete5(self.rally_recovery_success)[i]),
                    recovery_success=bool(self.rally_recovery_success[i]),contact_count=int(self.rally.contact_count[i]),
                    contact_point_m=self.rally.contact_point[i].cpu().tolist(),
                    outgoing_velocity_mps=self.rally.outgoing_velocity[i].cpu().tolist(),
                    first_landing_m=self.rally.landing_point[i].cpu().tolist(),
                    first_landing_time_s=float(self.rally.landing_time[i]),
                    crossing_point_m=self.rally.crossing_point[i].cpu().tolist(),crossing_time_s=float(self.rally.crossing_time[i]),
                    release_time_s=float(self.rally.release_time[i]),
                    first_string_contact_point_m=self.first_string_point[i].cpu().tolist(),first_string_time_s=float(self.first_string_time[i]),
                    contact_time_s=float(self.rally.contact_time[i]),net_margin_m=float(self.rally.net_margin[i]),
                    numerical_failures=int((self.sweep_misses+self.sweep_unresolved)[i]),
                    racket_plane_native_events=[record for record in self.racket_plane_records
                        if record['env_id']==i and record['episode_index']==int(self.episode_index[i])],
                    active_rally_launch_count=int(self.launch_count[i]),hit_velocity_error_mps=float(self.hit_velocity_error[i]))
                if self.contact_recovery is not None:
                    row['contact_flight_recovery']=self.contact_recovery.record(self,i,self.robot.data.root_pos_w[i,2])
                self.rally_records.append(row)
            self.rally.reset(ids);self.launch_done[ids]=False;self.launch_count[ids]=0
            self.launch_signals_ready[ids]=False;self.launch_frame[ids]=-1
            self.ball_observation_enabled[ids]=False
            self.episode_index[complete]+=1
            self.episode_index[ids]=self.episode_index[ids].clamp_min(0)
            self.hit_velocity_error[ids]=0.
            self.first_string_point[ids]=float('nan');self.first_string_time[ids]=float('nan')
        super()._reset_idx(ids)
        if self.contact_recovery is not None:self.contact_recovery.reset(ids)
