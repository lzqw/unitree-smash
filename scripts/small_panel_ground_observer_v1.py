"""Read-only same body/ground force input for a small development panel."""
import re
import torch


class SmallPanelGroundObserver:
    def __init__(self,env):
        from pxr import Usd,UsdPhysics
        from omni.usd import get_context
        self.env=env;self.names=list(env.robot.body_names)
        root=get_context().get_stage().GetPrimAtPath('/World/envs/env_0/Robot')
        paths={p.GetName():str(p.GetPath()) for p in Usd.PrimRange(root,Usd.TraverseInstanceProxies()) if p.HasAPI(UsdPhysics.RigidBodyAPI)}
        patterns=[paths[n].replace('/env_0/','/env_*/') for n in self.names]
        self.view=env.feet_contact._physics_sim_view.create_rigid_contact_view(patterns,
            filter_patterns=[['/World/Ground/geometry/mesh'] for _ in patterns],max_contact_data_count=131072)
        mapping={(int(re.search(r'/env_(\d+)/',p)[1]),p.rsplit('/',1)[-1]):i for i,p in enumerate(self.view.sensor_paths)}
        expected=[(lane,n) for lane in range(env.num_envs) for n in self.names]
        if set(mapping)!=set(expected) or self.view.filter_count!=1:raise ValueError('Incomplete ground pair view')
        self.order=torch.tensor([mapping[k] for k in expected],device=env.device)
    def read(self):
        force=self.view.get_contact_force_matrix(self.env.physics_dt)
        expected=(self.env.num_envs*len(self.names),1,3)
        if tuple(force.shape)!=expected:raise ValueError('Unexpected native ground matrix')
        return force[self.order,0].reshape(self.env.num_envs,len(self.names),3)
