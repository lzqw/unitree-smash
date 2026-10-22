#!/usr/bin/env python3
"""Run the real Go2+PiPER coupled articulation in Isaac Lab.

This is the first asset gate for the task pack.  The source USD contains one
PhysX articulation with the 12 Go2 leg joints and the six PiPER arm joints.
The PiPER gripper joints are deliberately removed from the task asset so the
policy/action contract remains 18-dimensional.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--episode-seconds", type=float, default=10.0)
parser.add_argument("--output", type=Path, default=Path("reports/G01"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 1 or args.episodes < 1 or args.episode_seconds <= 0:
    parser.error("num-envs, episodes and episode-seconds must be positive")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaacsim.core.utils.stage import get_current_stage
from pxr import Sdf, UsdPhysics


ROOT = Path(__file__).resolve().parents[1]
COUPLED_USD = ROOT / "assets/quadsmash_go2_piper.usd"
PIPER_USD = ROOT / "assets/vendor/piper_isaac_sim/USD/piper_v1.usd"
GO2_USD = Path("/home/autodrive/Isaac/unitree_model/Go2/usd/go2.usd")
PIPER_REPO = ROOT / "assets/vendor/piper_isaac_sim"
RACKET_MANIFEST = ROOT / "assets/rackets/standard_racket_v1.json"

LEG_JOINTS = (
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
)
ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 7))
EXPECTED_JOINTS = LEG_JOINTS + ARM_JOINTS


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _joint_positions() -> dict[str, float]:
    values = {
        "FL_hip_joint": 0.1, "FR_hip_joint": -0.1,
        "RL_hip_joint": 0.1, "RR_hip_joint": -0.1,
        "FL_thigh_joint": 0.8, "FR_thigh_joint": 0.8,
        "RL_thigh_joint": 1.0, "RR_thigh_joint": 1.0,
        "FL_calf_joint": -1.5, "FR_calf_joint": -1.5,
        "RL_calf_joint": -1.5, "RR_calf_joint": -1.5,
        "joint1": 0.0, "joint2": 1.0, "joint3": -1.0,
        "joint4": 0.0, "joint5": 0.0, "joint6": 0.0,
    }
    return values


@configclass
class CoupledCfg(DirectRLEnvCfg):
    decimation = 5
    episode_length_s = args.episode_seconds
    is_finite_horizon = True
    action_space = 18
    observation_space = 48
    state_space = 0
    sim = sim_utils.SimulationCfg(dt=0.002, render_interval=5)
    scene = InteractiveSceneCfg(num_envs=args.num_envs, env_spacing=3.0, replicate_physics=True)
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
            joint_pos=_joint_positions(),
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
    cfg: CoupledCfg

    def __init__(self, cfg: CoupledCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.joint_ids, found = self.robot.find_joints(list(EXPECTED_JOINTS), preserve_order=True)
        if tuple(found) != EXPECTED_JOINTS:
            raise RuntimeError(f"unexpected active joints: {found}")
        # ``joint_names`` is PhysX's native order; ``find_joints(...,
        # preserve_order=True)`` is the policy contract order used below.
        self.joint_mapping_ok = tuple(found) == EXPECTED_JOINTS
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
        self.targets[:, self.joint_ids] += 0.05 * actions.clamp(-1.0, 1.0)

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


def main() -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = CoupledCfg()
    cfg.sim.device = args.device or "cuda:0"
    env = CoupledEnv(cfg)
    try:
        stage = get_current_stage()
        mount_path = Sdf.Path("/World/envs/env_0/Robot/MountJoint")
        mount = UsdPhysics.FixedJoint.Get(stage, mount_path)
        body0 = [str(value) for value in mount.GetBody0Rel().GetTargets()]
        body1 = [str(value) for value in mount.GetBody1Rel().GetTargets()]
        mount_ok = (
            mount.GetPrim().IsValid()
            and len(body0) == 1
            and len(body1) == 1
            and stage.GetPrimAtPath(body0[0]).IsValid()
            and stage.GetPrimAtPath(body1[0]).IsValid()
        )
        racket_mount_path = Sdf.Path("/World/envs/env_0/Robot/RacketMount")
        racket_mount = UsdPhysics.FixedJoint.Get(stage, racket_mount_path)
        racket_body0 = [str(value) for value in racket_mount.GetBody0Rel().GetTargets()]
        racket_body1 = [str(value) for value in racket_mount.GetBody1Rel().GetTargets()]
        racket_mount_ok = (
            racket_mount.GetPrim().IsValid()
            and len(racket_body0) == 1
            and len(racket_body1) == 1
            and stage.GetPrimAtPath(racket_body0[0]).IsValid()
            and stage.GetPrimAtPath(racket_body1[0]).IsValid()
            and stage.GetPrimAtPath("/World/envs/env_0/Robot/Racket/Blade").IsValid()
        )

        obs, _ = env.reset()
        action = torch.zeros((env.num_envs, 18), device=env.device)
        completed = 0
        successes = 0
        numerical_failures = 0
        total_steps = 0
        start = time.perf_counter()
        with torch.inference_mode():
            while completed < args.episodes:
                obs, reward, terminated, truncated, _ = env.step(action)
                total_steps += env.num_envs
                data = env.robot.data
                finite = torch.isfinite(obs["policy"]).all(dim=1) & torch.isfinite(reward)
                numerical_failures += int((~finite).sum())
                done = terminated | truncated
                if done.any():
                    valid_end = (
                        (data.root_pos_w[:, 2] > 0.15)
                        & (data.projected_gravity_b[:, 2] <= -0.5)
                    )
                    episode_success = truncated & (~terminated) & valid_end & finite
                    successes += int(episode_success.sum())
                    completed += int(done.sum())
        torch.cuda.synchronize() if str(env.device).startswith("cuda") else None
        elapsed = time.perf_counter() - start
        metrics = {
            "joint_mapping_ok": bool(env.joint_mapping_ok),
            "coupled_dynamics_ok": bool(mount_ok and racket_mount_ok),
            "racket_mount_ok": bool(racket_mount_ok),
            "asset_manifest_ok": all(path.is_file() for path in (COUPLED_USD, PIPER_USD, GO2_USD, RACKET_MANIFEST)),
            "stand_success_rate": successes / completed if completed else 0.0,
            "sample_count": completed,
            "stand_episode_seconds": float(args.episode_seconds),
            "numerical_failures": numerical_failures,
            "backend": "Isaac Lab / PhysX",
            "num_envs": env.num_envs,
            "env_steps": total_steps,
            "elapsed_s": elapsed,
            "env_steps_per_second": total_steps / elapsed if elapsed else 0.0,
            "active_joint_count": env.robot.num_joints,
        }
        manifest = {
            "backend": "Isaac Lab 2.3.0 / Isaac Sim 5.1.0 / PhysX",
            "assumptions": {"path": str((ROOT / "assets/assumptions.md").relative_to(ROOT)), "sha256": sha256(ROOT / "assets/assumptions.md")},
            "combined_usd": {"path": str(COUPLED_USD.relative_to(ROOT)), "sha256": sha256(COUPLED_USD)},
            "go2_usd": {"path": str(GO2_USD), "sha256": sha256(GO2_USD)},
            "piper_usd": {"path": str(PIPER_USD.relative_to(ROOT)), "sha256": sha256(PIPER_USD)},
            "piper_source_commit": (
                __import__("subprocess").check_output(
                    ["git", "-C", str(PIPER_REPO), "rev-parse", "HEAD"], text=True
                ).strip()
            ),
            "active_joints": list(EXPECTED_JOINTS),
            "mount_joint": {"path": str(mount_path), "body0": body0, "body1": body1},
            "racket": {
                "manifest": str(RACKET_MANIFEST.relative_to(ROOT)),
                "mount_path": str(racket_mount_path),
                "body0": racket_body0,
                "body1": racket_body1,
                "blade_path": "/World/envs/env_0/Robot/Racket/Blade",
            },
            "note": "The two PiPER gripper prismatic joints are excluded from the 18-DOF task asset; standard_racket_v1 is a coupled rigid payload with parameterized mass.",
        }
        (args.output / "asset_tests.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        (args.output / "assets_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print("QUADSMASH_ASSET_RESULT=" + json.dumps(metrics, sort_keys=True))
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
