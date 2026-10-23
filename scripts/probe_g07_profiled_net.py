#!/usr/bin/env python3
"""Native profile-clearance and deliberately colliding robot positive controls."""
import argparse
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a).app

import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from pxr import UsdGeom
from isaacsim.core.utils.stage import get_current_stage
from quadsmash.physics.net_geometry import PROFILE_VERSION
from quadsmash.tasks.stand.ball_env import BallEnvCfg
from quadsmash.tasks.stand.smash_env_v3 import FullJointSmashEnv
from quadsmash.tasks.stand.net_contact_monitor import NetContactMonitor, add_net_contact_sensors


def main():
    a.output.mkdir(parents=True, exist_ok=True)
    cfg = BallEnvCfg()
    cfg.scene.num_envs = 20
    cfg.seed = 717701
    cfg.sim.device = 'cuda:0'
    cfg.sim.dt, cfg.decimation, cfg.sim.render_interval = .00025, 40, 40
    cfg.episode_length_s = .2
    cfg.include_net = True
    cfg.net_geometry_version, cfg.net_side_height_m = PROFILE_VERSION, 1.55

    class ProbeEnv(NetContactMonitor, FullJointSmashEnv):
        def _setup_scene(self):
            super()._setup_scene()
            add_net_contact_sensors(self)

        def _write_shuttle_reset(self, ids):
            state = self.shuttle.data.default_root_state[ids].clone()
            starts = torch.tensor([[1.3, 0., 1.], [1.3, .9, 1.565],
                                   [1.3, 0., 1.565], [1.3, .9, 1.60]], device=self.device)
            state[:, :3] = self.scene.env_origins[ids]+starts[ids % 4]
            state[:, 7:10] = torch.tensor([5., 0., 0.], device=self.device)
            self.shuttle.write_root_state_to_sim(state, env_ids=ids)
            if self.contact_events is not None:
                self.previous_ball_position[ids] = state[:, :3]
                self.previous_ball_velocity[ids] = state[:, 7:10]
                self.previous_drag[ids] = 0.

    env = ProbeEnv(cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
    try:
        wrapped.reset()
        positive = torch.arange(16, 20, device=env.device)
        with torch.inference_mode():
            root = env.robot.data.root_state_w[positive].clone()
            root[:, 0] = env.scene.env_origins[positive, 0]+1.35
            env.robot.write_root_state_to_sim(root, env_ids=positive)
            env.sim.forward()
        limits = env.robot.data.joint_pos_limits[:, env.joint_ids]
        default = env.robot.data.default_joint_pos[:, env.joint_ids]
        action = (2*(default-limits[:, :, 0])/(limits[:, :, 1]-limits[:, :, 0])-1).clamp(-1, 1)
        trace, geometry = [], []
        stage = get_current_stage()
        cache = UsdGeom.XformCache()
        for i in range(20):
            mesh = UsdGeom.Mesh.Get(stage, f'/World/envs/env_{i}/Net/geometry/mesh')
            transform = cache.GetLocalToWorldTransform(mesh.GetPrim())
            from pxr import Gf
            points = [transform.Transform(Gf.Vec3d(*v)) for v in mesh.GetPointsAttr().Get()]
            origin = env.scene.env_origins[i].cpu().tolist()
            points = [[float(v[j])-origin[j] for j in range(3)] for v in points]
            geometry.append(dict(env_id=i, x_bounds_m=[min(v[0] for v in points), max(v[0] for v in points)],
                                 top_center_m=max(v[2] for v in points if abs(v[1]) < 1e-6),
                                 top_sides_m=max(v[2] for v in points)))
        with torch.inference_mode():
            for tick in range(20):
                trace.append(dict(time_s=tick*.01, position=(env.shuttle.data.root_pos_w-env.scene.env_origins).cpu().tolist(),
                                  velocity=env.shuttle.data.root_lin_vel_w.cpu().tolist()))
                wrapped.step(action)
        positions = torch.tensor([r['position'] for r in trace])
        velocities = torch.tensor([r['velocity'] for r in trace])
        crossed = (positions[:, :16, 0] > 1.54).any(0)
        reflected = ((positions[:, :16, 0] > 1.40) & (velocities[:, :16, 0] < -.1)).any(0)
        expected = torch.tensor([False, False, True, True]*4)
        correct = (crossed == expected) & (expected | reflected)
        records = {r['env_id']: r for r in env.net_episode_records}
        positives = [records[i] for i in range(16, 20)]
        geometry_ok = all(abs(g['top_center_m']-1.524) < 1e-5 and abs(g['top_sides_m']-1.55) < 1e-5
                          and abs(g['x_bounds_m'][0]-1.49) < 1e-5 for g in geometry)
        result = dict(kind='native diagnostic only; positive lanes deliberately start colliding',
                      net_geometry_version=PROFILE_VERSION, geometry=geometry, geometry_correct=geometry_ok,
                      ball_cases_correct=int(correct.sum()), ball_cases=16, expected_crossing=expected.tolist(),
                      observed_crossing=crossed.tolist(), reflected=reflected.tolist(),
                      robot_net_positive_detected=sum(r['robot_net_contact'] for r in positives),
                      robot_net_positive_cases=4, positive_rows=positives, env_steps=400)
        result['checks_passed'] = bool(correct.all()) and geometry_ok and result['robot_net_positive_detected'] == 4
        (a.output/'metrics.json').write_text(json.dumps(result, indent=2)+'\n')
        (a.output/'trace.json').write_text(json.dumps(trace)+'\n')
        (a.output/'completed.json').write_text(json.dumps(dict(checks_passed=result['checks_passed']))+'\n')
        print(json.dumps(result), flush=True)
    finally:
        env.close()


try:
    main()
except BaseException:
    a.output.mkdir(parents=True, exist_ok=True)
    error = traceback.format_exc()
    (a.output/'error.txt').write_text(error)
    print(error, file=sys.stderr, flush=True)
    raise
finally:
    app.close(skip_cleanup=True)
