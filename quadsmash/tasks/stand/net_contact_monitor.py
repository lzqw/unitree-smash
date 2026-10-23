"""Separate robot/net force telemetry; original jump/ground metrics stay intact."""
import torch
from isaaclab.sensors import ContactSensor,ContactSensorCfg
import omni.usd
from pxr import Usd,UsdPhysics


def add_net_contact_sensors(env):
    # A filtered ContactSensor must have ONE source body per environment.
    # Grouping all Go2/PiPER/tool bodies makes PhysX expect N*B net targets,
    # while the actual net pattern has N entries, silently producing zeros.
    stage=omni.usd.get_context().get_stage()
    robot=stage.GetPrimAtPath('/World/envs/env_0/Robot')
    env._net_sensor_names=[]
    env._net_sensor_body_paths=[]
    for prim in Usd.PrimRange(robot,Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):continue
        name=f'net_body_{len(env._net_sensor_names):02d}'
        path=str(prim.GetPath()).replace('/env_0/','/env_.*/')
        cfg=ContactSensorCfg(prim_path=path,
            filter_prim_paths_expr=['/World/envs/env_.*/Net/geometry/mesh'],force_threshold=.1)
        env.scene.sensors[name]=ContactSensor(cfg)
        env._net_sensor_names.append(name)
        env._net_sensor_body_paths.append(path)
        if prim.GetName()=='Racket':env._net_racket_sensor_name=name
    if not env._net_sensor_names:raise RuntimeError('No robot bodies for net contact monitoring')


class NetContactMonitor:
    def __init__(self,*args,**kwargs):
        self._net_force_ready=False
        self.net_episode_records=[]
        super().__init__(*args,**kwargs)
        self.net_robot_seen=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.net_robot_max_force=torch.zeros(self.num_envs,device=self.device)
        self.net_first_contact_time=torch.full((self.num_envs,),float('nan'),device=self.device)
        self.net_first_contact_body=torch.full((self.num_envs,),-1,dtype=torch.long,device=self.device)
        self._net_force_ready=True

    def _consume_jump_frame(self,frame):
        previous=getattr(self,'_jump_physics_frame',-1)
        super()._consume_jump_frame(frame)
        if self._net_force_ready and frame>previous:
            per_body=torch.stack([self.scene.sensors[name].data.force_matrix_w.norm(dim=-1).sum((1,2))
                                  for name in self._net_sensor_names])
            force=per_body.sum(0)
            first=(force>.1)&(~self.net_robot_seen)
            self.net_first_contact_time[first]=self.jump_clock[first]
            self.net_first_contact_body[first]=per_body[:,first].argmax(0)
            self.net_robot_seen|=force>.1
            self.net_robot_max_force.copy_(torch.maximum(self.net_robot_max_force,force))

    def _reset_idx(self,ids):
        if self._net_force_ready:
            complete=ids[self._episode_has_stepped[ids]]
            for i in complete.cpu().tolist():
                self.net_episode_records.append(dict(env_id=i,
                    net_contact_sensor_version='one-source-per-environment-v2',
                    first_robot_net_contact_time_s=(float(self.net_first_contact_time[i])
                        if bool(self.net_robot_seen[i]) else None),
                    first_robot_net_contact_body=(self._net_sensor_body_paths[int(self.net_first_contact_body[i])]
                        if int(self.net_first_contact_body[i])>=0 else None),
                    robot_net_contact=bool(self.net_robot_seen[i]),
                    maximum_robot_net_force_n=float(self.net_robot_max_force[i])))
            self.net_robot_seen[ids]=False
            self.net_robot_max_force[ids]=0.
            self.net_first_contact_time[ids]=float('nan')
            self.net_first_contact_body[ids]=-1
        super()._reset_idx(ids)
