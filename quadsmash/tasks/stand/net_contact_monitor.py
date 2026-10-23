"""Separate robot/net force telemetry; original jump/ground metrics stay intact."""
import torch
from isaaclab.sensors import ContactSensor,ContactSensorCfg


def add_net_contact_sensors(env):
    patterns={
        'net_go2':'/World/envs/env_.*/Robot/Go2/.*',
        'net_arm':'/World/envs/env_.*/Robot/Piper/.*',
        'net_tool':'/World/envs/env_.*/Robot/(MountPlate|Racket)',
    }
    for name,path in patterns.items():
        cfg=ContactSensorCfg(prim_path=path,
            filter_prim_paths_expr=['/World/envs/env_.*/Net/geometry/mesh'],force_threshold=.1)
        env.scene.sensors[name]=ContactSensor(cfg)


class NetContactMonitor:
    def __init__(self,*args,**kwargs):
        self._net_force_ready=False
        self.net_episode_records=[]
        super().__init__(*args,**kwargs)
        self.net_robot_seen=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.net_robot_max_force=torch.zeros(self.num_envs,device=self.device)
        self._net_force_ready=True

    def _consume_jump_frame(self,frame):
        previous=getattr(self,'_jump_physics_frame',-1)
        super()._consume_jump_frame(frame)
        if self._net_force_ready and frame>previous:
            force=sum(self.scene.sensors[name].data.force_matrix_w.norm(dim=-1).sum((1,2))
                      for name in ('net_go2','net_arm','net_tool'))
            self.net_robot_seen|=force>.1
            self.net_robot_max_force.copy_(torch.maximum(self.net_robot_max_force,force))

    def _reset_idx(self,ids):
        if self._net_force_ready:
            complete=ids[self._episode_has_stepped[ids]]
            for i in complete.cpu().tolist():
                self.net_episode_records.append(dict(env_id=i,
                    robot_net_contact=bool(self.net_robot_seen[i]),
                    maximum_robot_net_force_n=float(self.net_robot_max_force[i])))
            self.net_robot_seen[ids]=False
            self.net_robot_max_force[ids]=0.
        super()._reset_idx(ids)
