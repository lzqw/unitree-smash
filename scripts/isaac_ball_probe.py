#!/usr/bin/env python3
"""Short native production-environment collision probe, never a learning gate."""
import argparse
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
app = AppLauncher(args).app
env = None
try:
    import torch
    from quadsmash.tasks.stand.ball_env import BallEnv, BallEnvCfg
    from quadsmash.physics.native import quat_rotate
    cfg = BallEnvCfg()
    cfg.scene.num_envs = 4
    cfg.episode_length_s = 10.0
    cfg.robot.init_state.pos = (0.0, 0.0, 5.0)
    cfg.sim.device = args.device or "cuda:0"
    env = BallEnv(cfg)
    with torch.inference_mode():
        env.reset()
        zero = torch.zeros((4, 18), device=env.device)
        env.step(zero)
        # A reset-only diagnostic launch at the measured tool pose. Lanes
        # compare a face impact, fast impact, geometric miss and free fall.
        tool = env.robot.data.body_pos_w[:, env.racket_body_id].clone()
        quat = env.robot.data.body_quat_w[:, env.racket_body_id].clone()
        normal = quat_rotate(quat, torch.tensor([[1., 0., 0.]], device=env.device).expand(4, -1))
        tangent = quat_rotate(quat, torch.tensor([[0., 1., 0.]], device=env.device).expand(4, -1))
        state = env.shuttle.data.default_root_state.clone()
        state[:, :3] = tool + 0.25 * normal
        state[2, :3] += 0.3*tangent[2]
        state[3, :3] += 0.7*tangent[3]
        state[:, 7:10] = -normal * torch.tensor([10., 120., 20., 0.], device=env.device)[:, None]
        env.shuttle.write_root_state_to_sim(state)
        env.previous_ball_position.copy_(state[:, :3])
        env.previous_ball_velocity.copy_(state[:, 7:10])
        rows = []
        for i in range(40):
            env.step(zero)
            rows.append({"step": i, "position": env.shuttle.data.root_pos_w.tolist(),
                         "velocity": env.shuttle.data.root_lin_vel_w.tolist(),
                         "count": env.contact_events.count.tolist(),
                         "force": env.shuttle_contact.data.force_matrix_w.tolist(),
                         "outgoing_speed": env.outgoing_speed.tolist()})
        result = {"backend": "Isaac Lab / PhysX", "contact_count": env.contact_events.count.tolist(),
                  "force_shape": list(env.shuttle_contact.data.force_matrix_w.shape),
                  "contact_point_shape": list(env.shuttle_contact.data.contact_pos_w.shape),
                  "rows": rows}
        (args.output / "probe.json").write_text(json.dumps(result, indent=2)+"\n")
        print("BALL_PROBE " + json.dumps({k:v for k,v in result.items() if k != "rows"}), flush=True)
except BaseException:
    (args.output / "error.txt").write_text(traceback.format_exc())
    raise
finally:
    if env is not None:
        env.close()
    app.close()
