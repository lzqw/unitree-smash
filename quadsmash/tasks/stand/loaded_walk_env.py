"""Whole loaded robot with the original static net, for no-ball movement search."""
import isaaclab.sim as sim_utils
import omni.usd
from .jump_env import JumpEnv
from .net_contact_monitor import NetContactMonitor,add_net_contact_sensors


class LoadedWalkEnv(NetContactMonitor,JumpEnv):
    action_family='absolute-joint-limits-v1'
    net_geometry_version='environment-local-net-v2'

    def _setup_scene(self):
        super()._setup_scene()
        stage=omni.usd.get_context().get_stage()
        net=sim_utils.CuboidCfg(size=(.02,2.,1.524),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(restitution=.1))
        for i in range(self.num_envs):
            path=f'/World/envs/env_{i}/Net'
            if not stage.GetPrimAtPath(path).IsValid():
                net.func(path,net,translation=(1.5,0.,.762))
        add_net_contact_sensors(self)

    def _pre_physics_step(self,actions):
        self.action_delta.copy_((actions.clamp(-1,1)-self.last_action).square().sum(1))
        self._record_action(actions)
        limits=self.robot.data.joint_pos_limits[:,self.joint_ids]
        self.targets.copy_(self.robot.data.default_joint_pos)
        self.targets[:,self.joint_ids]=limits[:,:,0]+.5*(self.last_action+1)*(limits[:,:,1]-limits[:,:,0])
