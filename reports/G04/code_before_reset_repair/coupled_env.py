"""Native Isaac Lab 18-DOF Go2+PiPER standing/recovery environment."""
from __future__ import annotations

from pathlib import Path

import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass


ROOT = Path(__file__).resolve().parents[3]
COUPLED_USD = ROOT / "assets/quadsmash_go2_piper.usd"
LEG_JOINTS = (
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
)
ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 7))
EXPECTED_JOINTS = LEG_JOINTS + ARM_JOINTS


def default_joint_positions() -> dict[str, float]:
    return {
        "FL_hip_joint": 0.1, "FR_hip_joint": -0.1,
        "RL_hip_joint": 0.1, "RR_hip_joint": -0.1,
        "FL_thigh_joint": 0.8, "FR_thigh_joint": 0.8,
        "RL_thigh_joint": 1.0, "RR_thigh_joint": 1.0,
        "FL_calf_joint": -1.5, "FR_calf_joint": -1.5,
        "RL_calf_joint": -1.5, "RR_calf_joint": -1.5,
        "joint1": 0.0, "joint2": 1.0, "joint3": -1.0,
        "joint4": 0.0, "joint5": 0.0, "joint6": 0.0,
    }


@configclass
class CoupledEnvCfg(DirectRLEnvCfg):
    decimation = 5
    episode_length_s = 4.0
    is_finite_horizon = False
    action_space = 18
    observation_space = 48
    state_space = 0
    sim = sim_utils.SimulationCfg(dt=0.002, render_interval=5)
    scene = InteractiveSceneCfg(num_envs=32, env_spacing=3.0, replicate_physics=True)
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        articulation_root_prim_path="/Go2/base",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(COUPLED_USD),
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                fix_root_link=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.4),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos=default_joint_positions(),
            joint_vel={".*": 0.0},
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                effort_limit_sim=23.5,
                velocity_limit_sim=30.0,
                stiffness=25.0,
                damping=0.5,
            ),
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-6]"],
                effort_limit_sim=100.0,
                velocity_limit_sim=5.0,
                stiffness=80.0,
                damping=4.0,
            ),
        },
    )


class CoupledEnv(DirectRLEnv):
    cfg: CoupledEnvCfg

    def __init__(self, cfg: CoupledEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.joint_ids, found = self.robot.find_joints(list(EXPECTED_JOINTS), preserve_order=True)
        if tuple(found) != EXPECTED_JOINTS:
            raise RuntimeError(f"unexpected active joints: {found}")
        self.joint_mapping_ok = True
        self.targets = self.robot.data.default_joint_pos.clone()

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        ground = sim_utils.CuboidCfg(
            size=(200.0, 200.0, 0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.8),
        )
        ground.func("/World/Ground", ground, translation=(0.0, 0.0, -0.05))
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/Ground"])
        self.scene.articulations["robot"] = self.robot

    def _pre_physics_step(self, actions: torch.Tensor):
        if actions.shape != (self.num_envs, 18) or not torch.isfinite(actions).all():
            raise ValueError("expected finite [num_envs, 18] actions")
        self.targets = self.robot.data.default_joint_pos.clone()
        self.targets[:, self.joint_ids] += 0.1 * actions.clamp(-1.0, 1.0)

    def _apply_action(self):
        self.robot.set_joint_position_target(self.targets)

    def _get_observations(self):
        data = self.robot.data
        obs = torch.cat(
            (
                data.root_lin_vel_b,
                data.root_ang_vel_b,
                data.projected_gravity_b,
                (data.joint_pos - data.default_joint_pos)[:, self.joint_ids],
                data.joint_vel[:, self.joint_ids],
                torch.zeros((self.num_envs, 3), device=self.device),
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self):
        data = self.robot.data
        tilt = torch.sum(data.projected_gravity_b[:, :2].square(), dim=1)
        pose_error = torch.sum((data.joint_pos - data.default_joint_pos).square(), dim=1)
        return (torch.exp(-4.0 * tilt - pose_error) - 0.0001 * data.applied_torque.square().sum(1)) * self.step_dt

    def _get_dones(self):
        data = self.robot.data
        fallen = (data.root_pos_w[:, 2] < 0.15) | (data.projected_gravity_b[:, 2] > -0.5)
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        return fallen, timeout

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self.targets[env_ids] = self.robot.data.default_joint_pos[env_ids]
