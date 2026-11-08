"""Read-only native foot friction/anchors and net contact telemetry.

Preserves the original all-body ground observer and every control command.
Normal and friction forces are recorded separately. Ragged contact buffers
are reduced before a later API call can reuse their count/index storage.
"""
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import torch
from scripts.native_pure_control_observation_v1 import NativePureControlObservationV1


def reduce_pairs(values,counts,starts,maximum_points=8):
    """Reduce ragged vectors; counts beyond the explicit cap are not certified."""
    counts=counts.reshape(-1).to(torch.int64);starts=starts.reshape(-1).to(torch.int64)
    slots=torch.arange(maximum_points,device=values.device)
    valid=slots[None,:]<counts[:,None]
    index=(starts[:,None]+slots).clamp(0,len(values)-1)
    selected=torch.where(valid[:,:,None],values[index],torch.zeros((),device=values.device,dtype=values.dtype))
    return selected.sum(1)


class NativeLandingContactObserverV2(NativePureControlObservationV1):
    def __init__(self,env,output):
        super().__init__(env,output)
        self.feet=['FL_foot','FR_foot','RL_foot','RR_foot'];paths=[]
        for name in self.feet:
            found=[p for p in self.view.sensor_paths if '/env_0/' in p and p.rsplit('/',1)[-1]==name]
            if len(found)!=1:raise ValueError('Unique actual foot body required')
            paths.append(found[0].replace('/env_0/','/env_*/'))
        self.foot_view=env.feet_contact._physics_sim_view.create_rigid_contact_view(paths,
            filter_patterns=[['/World/Ground/geometry/mesh'] for _ in paths],max_contact_data_count=8192)
        mapping={(int(re.search(r'/env_(\d+)/',p)[1]),p.rsplit('/',1)[-1]):i for i,p in enumerate(self.foot_view.sensor_paths)}
        expected=[(lane,name) for lane in range(128) for name in self.feet]
        if set(mapping)!=set(expected) or self.foot_view.filter_count!=1:raise ValueError('Incomplete ground foot view')
        self.foot_order=torch.tensor([mapping[k] for k in expected],device=env.device)
        self.foot_ids=[self.names.index(n) for n in self.feet]
        self.normal_impulse=torch.zeros((800,128,4,3),device=env.device)
        self.friction_impulse=torch.zeros_like(self.normal_impulse)
        self.maximum_points=torch.zeros((),dtype=torch.int64,device=env.device)
        self.normal_view_difference=torch.zeros((),device=env.device)
        self.net_paths=list(env.net_robot_body_paths)
        self.net_peak=torch.zeros((800,128,len(self.net_paths)),device=env.device)
        self.net_first=torch.full(self.net_peak.shape,-1,dtype=torch.int32,device=env.device)
        # Compact per-millisecond contact and actual foot kinematics. Root
        # and joint states are available at control rate in the command log.
        self.wave={k:torch.empty((8000,128,4,n),device=env.device) for k,n in
            [('normal_force',3),('friction_force',3),('normal_point',3),('friction_point',3),('foot_pose',7),('foot_twist',6)]}
        self.wave_count={k:torch.empty((8000,128,4),dtype=torch.int32,device=env.device) for k in ['normal','friction']}
        base_observed=self.wrapper;self.contact_seen=-1;self.contact_samples=0
        def observed(frame):
            result=base_observed(frame)
            if frame<=self.contact_seen:return result
            self.contact_seen=frame
            if not 0<frame<=128000:return result
            frictions,points,counts,starts=self.foot_view.get_friction_data(env.physics_dt)
            friction=reduce_pairs(frictions,counts,starts)[self.foot_order].reshape(128,4,3)
            friction_points=reduce_pairs(points,counts,starts)[self.foot_order].reshape(128,4,3)
            friction_count=counts.reshape(-1).to(torch.int32)[self.foot_order].reshape(128,4).clone()
            self.maximum_points=torch.maximum(self.maximum_points,counts.to(torch.int64).max())
            normal,points,normals,distance,counts,starts=self.foot_view.get_contact_data(env.physics_dt)
            vectors=normal.reshape(-1,1)*normals
            normal_force=reduce_pairs(vectors,counts,starts)[self.foot_order].reshape(128,4,3)
            normal_points=reduce_pairs(points,counts,starts)[self.foot_order].reshape(128,4,3)
            normal_count=counts.reshape(-1).to(torch.int32)[self.foot_order].reshape(128,4).clone()
            self.maximum_points=torch.maximum(self.maximum_points,counts.to(torch.int64).max())
            interval=(frame-1)//160
            self.normal_impulse[interval]+=normal_force*env.physics_dt
            self.friction_impulse[interval]+=friction*env.physics_dt
            self.normal_view_difference=torch.maximum(self.normal_view_difference,
                (normal_force-self.current[:,self.foot_ids]).abs().max())
            net=env.scene.sensors['inverse_net'].data.force_matrix_w[:,0,:-1].norm(dim=-1)
            if net.shape!=self.net_peak[interval].shape:raise ValueError('Native net body filter ordering changed')
            self.net_peak[interval]=torch.maximum(self.net_peak[interval],net)
            self.net_first[interval]=torch.where((net>.1)&(self.net_first[interval]<0),frame,self.net_first[interval])
            if frame%16==0:
                k=self.contact_samples;self.contact_samples+=1
                self.wave['normal_force'][k]=normal_force;self.wave['friction_force'][k]=friction
                self.wave['normal_point'][k]=normal_points/normal_count.clamp(min=1)[:,:,None]
                self.wave['friction_point'][k]=friction_points/friction_count.clamp(min=1)[:,:,None]
                self.wave_count['normal'][k]=normal_count;self.wave_count['friction'][k]=friction_count
                d=env.robot.data
                self.wave['foot_pose'][k]=torch.cat((d.body_link_pos_w[:,self.foot_ids],d.body_link_quat_w[:,self.foot_ids]),-1)
                self.wave['foot_twist'][k]=torch.cat((d.body_link_lin_vel_w[:,self.foot_ids],d.body_link_ang_vel_w[:,self.foot_ids]),-1)
            return result
        self.wrapper=observed;env._observe_physics_frame=observed

    def save(self):
        parent=super().save()
        result=dict(complete=self.contact_seen==128000 and self.contact_samples==8000,
            maximum_points_per_pair=int(self.maximum_points.item()),point_capacity_per_pair=8,
            maximum_normal_view_difference_n=float(self.normal_view_difference.item()),
            normal_and_friction_separate=True,physics_dt_s=self.env.physics_dt,sampling_period_s=.001,
            foot_order=self.feet,net_body_filter_paths=self.net_paths,
            commands_changed=False,physics_parameters_changed=False,friction_points_are_arithmetic_patch_means=True)
        path=self.output/'landing_contact_detail.npz';tmp=path.with_suffix('.tmp')
        arrays={k:v[:self.contact_samples].cpu().numpy() for k,v in self.wave.items()}
        arrays.update({k+'_count':v[:self.contact_samples].cpu().numpy() for k,v in self.wave_count.items()})
        arrays.update(normal_impulse=self.normal_impulse.cpu().numpy(),friction_impulse=self.friction_impulse.cpu().numpy(),
            net_peak=self.net_peak.cpu().numpy(),net_first_frame_over_point1n=self.net_first.cpu().numpy())
        if any(not np.isfinite(v).all() for v in arrays.values()):raise ValueError('Nonfinite contact telemetry')
        with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
        tmp.replace(path);result['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        (self.output/'landing_contact_detail.json').write_text(json.dumps(result,indent=2)+'\n')
        if result['maximum_points_per_pair']>8:raise ValueError('Contact telemetry exceeded admitted point capacity')
        return dict(parent,contact_detail=result)
