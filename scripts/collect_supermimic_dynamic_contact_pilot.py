#!/usr/bin/env python3
"""Short controlled contact trials on the task's native articulated robot and court."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def collect(args):
    import torch
    from quadsmash.physics.native import drag_force, quat_rotate
    from quadsmash.rl.supermimic_native_substep_trace_v1 import NativeSubstepTraceV1
    from quadsmash.tasks.stand.full_racket_env_v4 import FullRacketEnvV4, full_racket_cfg
    from quadsmash.rl.supermimic_physical_capture_v1 import tensor_tree_copy

    teacher = torch.load(args.teacher, weights_only=True, map_location='cpu')
    initial = teacher['raw']['start']['before']['physical']
    targets = teacher['raw']['final_joint_targets']
    cfg = full_racket_cfg(args.lanes)
    cfg.sim.device = args.device
    cfg.seed = args.seed
    env = FullRacketEnvV4(cfg)
    try:
        for batch in range(args.batches):
            seed = args.seed + batch
            rng = torch.Generator().manual_seed(seed)
            env.reset()
            device = env.device
            lanes = torch.arange(args.lanes, device=device)
            # Distinct seeds select new launch parameters around a measured swing.
            family = seed % (len(initial['joint_position']) // 8)
            parent = family * 8 + torch.randint(8, (args.lanes,), generator=rng)
            def sample(low, high):
                return (torch.rand(args.lanes, generator=rng) * (high - low) + low).to(device)
            def source(key):
                return initial[key][parent].to(device)
            data = env.robot.data
            origin = env.scene.env_origins
            pose = torch.cat((source('root_position_w') + origin, source('root_quaternion_w')), -1)
            source_origins = teacher['physics']['actual_origins_m']
            pose[:, :3] -= torch.as_tensor(source_origins, device=device)[parent.to(device)]
            # A free airborne root avoids introducing foot impulses during these 10 ms trials.
            pose[:, 2] += .65
            root_w = source('root_angular_velocity_w')
            root_v = source('root_link_velocity_w') + torch.linalg.cross(
                root_w, quat_rotate(pose[:, 3:7], data.body_com_pos_b[:, 0]))
            root_v += torch.stack([sample(-2, 2) for _ in range(3)], -1)
            root_w += torch.stack([sample(-7, 7) for _ in range(3)], -1)
            quiet = lanes % 12 == 0
            fast_spin = lanes % 12 == 3
            root_v[quiet] *= .025
            root_w[quiet] *= .025
            root_w[fast_spin, 2] += 22.
            joint_velocity = source('joint_velocity')
            joint_velocity[quiet] *= .025
            joint_targets = targets[parent].to(device)
            joint_targets[quiet] = source('joint_position')[quiet]
            env.robot.write_root_pose_to_sim(pose)
            env.robot.write_root_velocity_to_sim(torch.cat((root_v, root_w), -1))
            env.robot.write_joint_state_to_sim(source('joint_position'), joint_velocity,
                                              joint_ids=env.joint_ids)
            env.robot.set_joint_position_target(joint_targets, joint_ids=env.joint_ids)
            ball = env.shuttle.data.default_root_state.clone()
            ball[:, :3] = origin + origin.new_tensor([0., 0., 5.])
            env.shuttle.write_root_state_to_sim(ball)
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(env.physics_dt)
            rid = env.racket_body_id
            q = data.body_link_quat_w[:, rid]
            side = torch.where(lanes % 2 == 0, 1., -1.)
            mode = lanes % 8
            # Spread launch targets over the 5x5 face, frame, edge and just beyond it.
            y = (lanes % 5 - 2) * .030 + sample(-.008, .008)
            z = ((lanes // 5) % 5 - 2) * .038 + sample(-.010, .010)
            angle = sample(-3.14159, 3.14159)
            y = torch.where(mode == 0, sample(-.009, .009), y)
            z = torch.where(mode == 0, sample(-.012, .012), z)
            ring = mode >= 4
            scale = torch.where(mode == 7, sample(1.29, 1.5), sample(.90, 1.08))
            y = torch.where(ring, .10 * scale * angle.cos(), y)
            z = torch.where(ring, .125 * scale * angle.sin(), z)
            target = torch.stack((side * .0256, y, z), -1)
            flight = sample(.0002, .0094)
            relative = torch.stack((-side * sample(.5, 18), sample(-10, 10), sample(-10, 10)), -1)
            relative[lanes % 8 == 1, 1:] *= .025
            grazing = torch.ones_like(lanes, dtype=torch.bool) if args.grazing_only else lanes % 8 == 5
            target[grazing, 1] = sample(-.045, .045)[grazing]
            target[grazing, 2] = sample(-.05, .05)[grazing]
            relative[grazing, 0] = (-side * sample(.25, 2.0))[grazing]
            relative[grazing, 1] = sample(10, 22)[grazing]
            offset = quat_rotate(q, target - flight[:, None] * relative)
            ball[:, :3] = data.body_link_pos_w[:, rid] + offset
            ball[:, 7:10] = (data.body_link_lin_vel_w[:, rid] + quat_rotate(q, relative)
                             + torch.linalg.cross(data.body_link_ang_vel_w[:, rid], offset))
            # Local ground/net negatives challenge object attribution.
            ground, net = lanes % 32 == 29, lanes % 32 == 30
            if args.grazing_only:
                ground.zero_()
                net.zero_()
            speed = sample(3, 14)
            ball[ground, :3] = origin[ground] + torch.stack((
                sample(2, 3), sample(-1, 1), .025 + flight * speed), -1)[ground]
            ball[ground, 7:10] = 0.
            ball[ground, 9] = -speed[ground]
            plane = env.court_config["net_plane_x_m"]
            ball[net, :3] = origin[net] + torch.stack((
                plane - .027 - flight * speed, sample(-1, 1), sample(.9, 1.45)), -1)[net]
            ball[net, 7:10] = 0.
            ball[net, 7] = speed[net]
            env.shuttle.write_root_state_to_sim(ball)
            env.sim.forward()
            trace = NativeSubstepTraceV1(env, start_frame=env.sim.current_time_step_index)
            for _ in range(cfg.decimation):
                force = drag_force(env.shuttle.data.root_lin_vel_w, cfg.shuttle_mass_kg,
                                   cfg.drag_k_per_m, cfg.wind_m_s)[:, None]
                env.shuttle.set_external_force_and_torque(force, torch.zeros_like(force), is_global=True)
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(env.physics_dt)
                trace.observe_frame(env.sim.current_time_step_index)
            archive = tensor_tree_copy(trace.finish(), cpu=True)
            series = archive['tensors']
            if not torch.isfinite(series['shuttle_state_w']).all() or not archive['manifold_frames']:
                raise ValueError('Invalid fixture trace or missing native manifold stream')
            contact = series['racket_force_w'].norm(dim=-1) > .001
            if contact[0].any():
                raise ValueError('Stale racket contact at fixture initialization')
            directory = args.output / f'spec_{seed}'
            directory.mkdir()
            temp = directory / 'trace.tmp'
            torch.save(dict(schema='native-controlled-contact-trial-v1', seed=seed, teacher=args.teacher.name,
                            parent_lane=parent, parent_family=family, raw={'substep_trace': archive}), temp)
            temp.replace(directory / 'trace.pt')
            print(json.dumps(dict(seed=seed, contact_lanes=int(contact.any(0).sum()),
                                  intervals=int((contact[1:] & ~contact[:-1]).sum()),
                                  lanes=args.lanes)), flush=True)
    finally:
        env.close()


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher', type=Path, required=True)
    parser.add_argument('--execution-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--grazing-only', action='store_true', help='Target secondary impacts with tangential passes across the string bed')
    parser.add_argument('--batches', type=int, default=1)
    parser.add_argument('--lanes', type=int, default=128)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if min(args.batches, args.lanes) <= 0 or not args.teacher.is_file():
        parser.error('Positive trial dimensions and an existing native teacher are required')
    os.environ['QUADSMASH_G07_EXECUTION_ROOT'] = str(args.execution_root.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'provenance.json').write_text(json.dumps(dict(
        teacher_sha256=hashlib.sha256(args.teacher.read_bytes()).hexdigest(),
        collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        seed=args.seed, batches=args.batches, lanes=args.lanes, scope='controlled native contact trials',
        args=sys.argv), indent=2))
    app = AppLauncher(args).app
    try:
        import torch
        with torch.inference_mode():
            collect(args)
    finally:
        app.close(skip_cleanup=True)


if __name__ == '__main__':
    main()
