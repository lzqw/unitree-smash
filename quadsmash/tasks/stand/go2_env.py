"""Native Isaac Lab diagnostic for P0; 12 leg joints, no arm or shuttle.

The robot is free floating and all states come from PhysX. This task validates
the runtime and PPO IO cycle, not loaded jumping or badminton performance.
"""
from __future__ import annotations

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

JOINT_ORDER = tuple(f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR") for joint in ("hip", "thigh", "calf"))


@configclass
class Go2DiagnosticCfg(DirectRLEnvCfg):
    decimation = 5
    episode_length_s = 4.0
    # This diagnostic is explicitly finite horizon. No timeout bootstrapping.
    is_finite_horizon = True
    action_space = 12
    observation_space = 45
    state_space = 0
    sim = sim_utils.SimulationCfg(dt=0.002, render_interval=5)
    scene = InteractiveSceneCfg(num_envs=32, env_spacing=3.0, replicate_physics=True)
    robot = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contacts = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/.*", history_length=3)
    action_scale = 0.1


class Go2DiagnosticEnv(DirectRLEnv):
    cfg: Go2DiagnosticCfg

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.joint_ids, found = self.robot.find_joints(list(JOINT_ORDER), preserve_order=True)
        if tuple(found) != JOINT_ORDER:
            raise ValueError(f"Unexpected Go2 joints: {found}")
        self.actions = torch.zeros((self.num_envs, 12), device=self.device)
        self.targets = self.robot.data.default_joint_pos.clone()

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.contacts = ContactSensor(self.cfg.contacts)
        # Procedural ground avoids a network dependency for the diagnostic.
        ground = sim_utils.CuboidCfg(
            size=(200.0, 200.0, 0.1), collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.8),
        )
        ground.func("/World/Ground", ground, translation=(0.0, 0.0, -0.05))
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/Ground"])
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contacts"] = self.contacts

    def _pre_physics_step(self, actions):
        if actions.shape != (self.num_envs, 12) or not torch.isfinite(actions).all():
            raise ValueError("Expected finite [num_envs, 12] actions")
        self.actions = actions.clamp(-1.0, 1.0)
        self.targets = self.robot.data.default_joint_pos.clone()
        self.targets[:, self.joint_ids] += self.cfg.action_scale * self.actions
        limits = self.robot.data.soft_joint_pos_limits
        self.targets = torch.maximum(torch.minimum(self.targets, limits[..., 1]), limits[..., 0])

    def _apply_action(self):
        self.robot.set_joint_position_target(self.targets)

    def _get_observations(self):
        d = self.robot.data
        observation = torch.cat((d.root_lin_vel_b, d.root_ang_vel_b, d.projected_gravity_b,
                                 (d.joint_pos - d.default_joint_pos)[:, self.joint_ids],
                                 d.joint_vel[:, self.joint_ids], self.actions), dim=-1)
        return {"policy": observation}

    def _get_rewards(self):
        d = self.robot.data
        tilt = torch.sum(d.projected_gravity_b[:, :2].square(), dim=1)
        pose_error = torch.sum((d.joint_pos - d.default_joint_pos).square(), dim=1)
        # A stand diagnostic only: these costs are not appropriate for jumping.
        return (torch.exp(-4 * tilt - pose_error) - 0.0001 * d.applied_torque.square().sum(1)) * self.step_dt

    def _get_dones(self):
        d = self.robot.data
        fallen = (d.root_pos_w[:, 2] < 0.15) | (d.projected_gravity_b[:, 2] > -0.5)
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        return fallen, timeout

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        d = self.robot.data
        root = d.default_root_state[env_ids].clone()
        root[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(d.default_joint_pos[env_ids], d.default_joint_vel[env_ids], env_ids=env_ids)
        if hasattr(self, "actions"):
            self.actions[env_ids] = 0
            self.targets[env_ids] = d.default_joint_pos[env_ids]
