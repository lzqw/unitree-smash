"""Native Isaac Lab 18-DOF Go2+PiPER standing/recovery environment."""
from __future__ import annotations

from pathlib import Path

import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass
from .observation_contract import OBSERVATION_SIZE, PolicyHistory, observe_native


ROOT = Path(__file__).resolve().parents[3]
COUPLED_USD = ROOT / "assets/quadsmash_standard_v2.usd"
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
    observation_space = OBSERVATION_SIZE
    state_space = 0
    sim = sim_utils.SimulationCfg(dt=0.002, render_interval=5)
    scene = InteractiveSceneCfg(num_envs=32, env_spacing=3.0, replicate_physics=True)
    event_goal_enabled = False
    event_goal_time_s = 1.25
    event_goal_position = (0.23, 0.0, 1.18)
    event_goal_normal = (1.0, 0.0, 0.0)
    event_goal_velocity = (0.0, 0.0, 0.0)
    feet_contact = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/Go2/.*_foot",
        track_air_time=True, force_threshold=1.0)
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        articulation_root_prim_path="/Go2/base",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(COUPLED_USD),
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                fix_root_link=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.323),
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
        self.racket_body_id = self.robot.find_bodies("Racket")[0][0]
        self.foot_ids = self.robot.find_bodies(".*_foot")[0]
        self.feet_contact = self.scene.sensors["feet_contact"]
        self.last_action = torch.zeros((self.num_envs,18),device=self.device)
        self.action_before_clip = torch.zeros_like(self.last_action)
        self.policy_history = PolicyHistory(self.num_envs,self.device)
        self.observed_foot_contact = torch.zeros((self.num_envs,4),dtype=torch.bool,device=self.device)
        self.observed_airborne = torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.observed_airborne_seen = torch.zeros_like(self.observed_airborne)
        self._observed_air_time = torch.zeros(self.num_envs,device=self.device)
        self._observation_physics_frame = -1
        self.goal_position_w = self.scene.env_origins + torch.tensor(self.cfg.event_goal_position,device=self.device)
        self.goal_normal_w = torch.tensor(self.cfg.event_goal_normal,device=self.device).expand(self.num_envs,-1).clone()
        self.goal_velocity_w = torch.tensor(self.cfg.event_goal_velocity,device=self.device).expand(self.num_envs,-1).clone()
        self.goal_time_s = torch.full((self.num_envs,),self.cfg.event_goal_time_s,device=self.device)
        self._inside_step = False
        self.final_observation = torch.zeros((self.num_envs,OBSERVATION_SIZE),device=self.device)
        self.final_observation_valid = torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)

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
        self.scene.sensors["feet_contact"] = ContactSensor(self.cfg.feet_contact)

    def _record_action(self,actions):
        if actions.shape != (self.num_envs,18) or not torch.isfinite(actions).all():
            raise ValueError("expected finite [num_envs,18] actions")
        self.action_before_clip.copy_(actions)
        self.last_action.copy_(actions.clamp(-1.,1.))

    def _pre_physics_step(self, actions: torch.Tensor):
        self._record_action(actions)
        if actions.shape != (self.num_envs, 18) or not torch.isfinite(actions).all():
            raise ValueError("expected finite [num_envs, 18] actions")
        self.targets.copy_(self.robot.data.default_joint_pos)
        self.targets[:, self.joint_ids] += 0.1 * actions.clamp(-1.0, 1.0)

    def _apply_action(self):
        self._observe_physics_frame(self._sim_step_counter-1)
        self.robot.set_joint_position_target(self.targets)

    def _observe_physics_frame(self,frame):
        if frame<=self._observation_physics_frame:
            return
        self._observation_physics_frame=frame
        if frame==0:
            return
        force=self.feet_contact.data.net_forces_w.norm(dim=-1)
        # Frozen G01 calibration: force hysteresis and 20 ms clear interval.
        self.observed_foot_contact.copy_(torch.where(self.observed_foot_contact,force>.5,force>1.))
        clear=(~self.observed_foot_contact.any(1)) & (self.robot.data.body_pos_w[:,self.foot_ids,2].min(1).values>.035)
        self._observed_air_time.copy_(torch.where(clear,self._observed_air_time+self.physics_dt,0.))
        self.observed_airborne.copy_(self._observed_air_time>=.02)
        self.observed_airborne_seen |= self.observed_airborne

    def step(self,actions):
        self.final_observation_valid.zero_()
        self._inside_step=True
        try:
            result=super().step(actions)
            self.extras["final_observation"]={"policy":self.final_observation}
            self.extras["final_observation_valid"]=self.final_observation_valid
            self.extras["terminated"]=self.reset_terminated
            return result
        finally:
            self._inside_step=False

    def _get_observations(self):
        self._observe_physics_frame(self._sim_step_counter)
        return observe_native(self)

    def _get_rewards(self):
        data = self.robot.data
        tilt = torch.sum(data.projected_gravity_b[:, :2].square(), dim=1)
        pose_error = torch.sum((data.joint_pos - data.default_joint_pos).square(), dim=1)
        return (torch.exp(-4.0 * tilt - pose_error) - 0.0001 * data.applied_torque.square().sum(1)) * self.step_dt

    def _get_dones(self):
        data = self.robot.data
        fallen = (data.root_pos_w[:, 2] < 0.15) | (data.projected_gravity_b[:, 2] > -0.5)
        timeout = self.episode_length_buf >= self.max_episode_length
        return fallen, timeout

    @torch.inference_mode()
    def _reset_idx(self, env_ids):
        if self._inside_step and env_ids.numel():
            final=self._get_observations()["policy"]
            self.final_observation[env_ids]=final[env_ids]
            self.final_observation_valid[env_ids]=True
        super()._reset_idx(env_ids)
        # DirectRLEnv/InteractiveScene.reset clear buffers, not physical state.
        # Write only the requested environments, including their grid origins.
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)
        self.robot.write_joint_state_to_sim(
            self.robot.data.default_joint_pos[env_ids],
            self.robot.data.default_joint_vel[env_ids],
            env_ids=env_ids,
        )
        self.targets[env_ids] = self.robot.data.default_joint_pos[env_ids]
        self.robot.set_joint_position_target(self.targets[env_ids], env_ids=env_ids)
        # Refresh link kinematics without advancing physical time. Tool
        # observations after an automatic reset must describe the reset pose.
        self.sim.forward()
        self.policy_history.reset(env_ids)
        self.last_action[env_ids]=0.
        self.action_before_clip[env_ids]=0.
        self.observed_foot_contact[env_ids]=False
        self.observed_airborne[env_ids]=False
        self.observed_airborne_seen[env_ids]=False
        self._observed_air_time[env_ids]=0.
