#!/usr/bin/env python3
"""Calibrate feet and audit each actuator plus sampled standard-arm poses."""
import argparse
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--asset", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--self-collisions", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
launcher = AppLauncher(args)
app = launcher.app
import torch
from quadsmash.tasks.stand.jump_env import JumpEnv, JumpEnvCfg

cfg = JumpEnvCfg()
cfg.seed = 1907
cfg.scene.num_envs = 36
cfg.episode_length_s = 10.0
cfg.robot.spawn.usd_path = str(args.asset.resolve())
cfg.robot.spawn.articulation_props.enabled_self_collisions = args.self_collisions
cfg.robot.init_state.pos = (0.0, 0.0, 0.323)
cfg.sim.device = args.device or "cuda:0"
env = JumpEnv(cfg)
try:
    with torch.inference_mode():
        env.reset()
        zero = torch.zeros((36, 18), device=env.device)
        feet = env.robot.find_bodies(".*_foot")[0]
        force_min, force_max, feet_z_min, feet_z_max = 1e9, 0.0, 1e9, -1e9
        max_leg_effort, max_arm_effort = 0.0, 0.0
        for step in range(500):
            env.step(zero)
            if step >= 200:
                force = env.feet_contact.data.net_forces_w.norm(dim=-1)
                force_min, force_max = min(force_min, force.min().item()), max(force_max, force.max().item())
                heights = env.robot.data.body_pos_w[:, feet, 2]
                feet_z_min, feet_z_max = min(feet_z_min, heights.min().item()), max(feet_z_max, heights.max().item())
                torque = env.robot.data.applied_torque[:, env.joint_ids].abs()
                max_leg_effort = max(max_leg_effort, torque[:, :12].max().item())
                max_arm_effort = max(max_arm_effort, torque[:, 12:].max().item())
        # Each actuated lane has its own simultaneous unactuated control lane.
        env.robot.data.default_root_state[:, 2] = 5.0
        env.reset()
        maximum_airborne_force = 0.0
        for step in range(50):
            action = zero.clone()
            action[torch.arange(1, 36, 2, device=env.device), torch.arange(18, device=env.device)] = 0.5
            env.step(action)
            if step >= 2:
                maximum_airborne_force = max(maximum_airborne_force, env.feet_contact.data.net_forces_w.norm(dim=-1).max().item())
        actual = env.robot.data.joint_pos[:, env.joint_ids]
        difference = actual[1::2] - actual[0::2]
        intended = difference.diag()
        joint_rows = [{"joint": env.robot.joint_names[joint_id], "paired_position_response_rad": intended[i].item(),
                       "commanded_offset_rad": 0.125 if i < 12 else 0.06,
                       "positive_response": bool(intended[i] > 0.001)} for i, joint_id in enumerate(env.joint_ids)]
        # Reset-only sampled poses measure geometric coverage under actual
        # inertia/actuator limits; they are not PPO training initializations.
        env.robot.data.default_root_state[:, 2] = 0.323
        nominal = env.robot.data.default_joint_pos.clone()
        limits = env.robot.data.joint_pos_limits[:, env.joint_ids[12:]]
        fractions = torch.rand((36, 6), device=env.device)
        sampled = limits[..., 0] + (0.15 + 0.7 * fractions) * (limits[..., 1] - limits[..., 0])
        env.robot.data.default_joint_pos[:, env.joint_ids[12:]] = sampled
        env.reset()
        racket_id = env.robot.find_bodies("Racket")[0][0]
        max_tool_speed = torch.zeros(36, device=env.device)
        resets = torch.zeros(36, dtype=torch.bool, device=env.device)
        for _ in range(150):
            _, _, term, trunc, _ = env.step(zero)
            max_tool_speed = torch.maximum(max_tool_speed, env.robot.data.body_com_lin_vel_w[:, racket_id].norm(dim=-1))
            resets |= term | trunc
        relative = env.robot.data.body_pos_w[:, racket_id] - env.robot.data.root_pos_w
        poses = [{"sample": i, "arm_joint_positions_rad": sampled[i].tolist(),
                  "observed_racket_relative_position_m": relative[i].tolist(),
                  "observed_max_world_tool_speed_m_s": max_tool_speed[i].item(),
                  "premature_reset": bool(resets[i]),
                  "final_base_tilt_deg": torch.rad2deg(torch.acos((-env.robot.data.projected_gravity_b[i, 2]).clamp(-1, 1))).item()} for i in range(36)]
        env.robot.data.default_joint_pos.copy_(nominal)
    result = {"backend": "Isaac Lab / PhysX", "asset": str(args.asset.resolve()),
              "self_collisions_enabled": args.self_collisions,
              "ground_contact_force_range_n": [force_min, force_max],
              "ground_foot_center_height_range_m": [feet_z_min, feet_z_max],
              "maximum_airborne_contact_force_n": maximum_airborne_force,
              "proposed_contact_on_n": 1.0, "proposed_contact_off_n": 0.5,
              "proposed_airborne_foot_center_min_m": 0.035,
              "proposed_hysteresis_s": 0.02,
              "zero_action_static_max_leg_torque_nm": max_leg_effort,
              "zero_action_static_max_arm_torque_nm": max_arm_effort,
              "body_names": env.robot.body_names,
              "body_com_local_m": env.robot.data.body_com_pos_b[0].tolist(),
              "body_inertia_kg_m2": env.robot.root_physx_view.get_inertias()[0].tolist(),
              "joint_names_native": env.robot.joint_names,
              "joint_position_limits_rad": env.robot.data.joint_pos_limits[0].tolist(),
              "joint_velocity_limits_rad_s": env.robot.data.joint_vel_limits[0].tolist(),
              "joint_effort_limits_nm": env.robot.data.joint_effort_limits[0].tolist(),
              "joint_response_tests": joint_rows, "all_18_joints_respond": all(r["positive_response"] for r in joint_rows),
              "sampled_workspace": poses,
              "workspace_caveat": "36 interior-limit poses only, not an exhaustive reachable set. World speed includes passive motion; not a smash capability measurement. See self_collisions_enabled for contact setting."}
    (args.output / "calibration.json").write_text(json.dumps(result, indent=2) + "\n")
    print("ASSET_CALIBRATION=" + json.dumps({k:v for k,v in result.items() if k not in ("joint_response_tests","sampled_workspace")}), flush=True)
except BaseException:
    error = traceback.format_exc()
    (args.output / "error.txt").write_text(error)
    print(error, file=sys.stderr, flush=True)
    raise
finally:
    env.close()
    app.close()
