#!/usr/bin/env python3
"""Compare eight shared preparation proposals in1024 native ground episodes."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--proposals', type=Path, required=True)
parser.add_argument('--seed', type=int, default=750501)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaacsim.core.utils.stage import get_current_stage
from quadsmash.control.full_racket_prepose_branch_v4 import GroupedPreposeControllerV4, SCHEMA
from quadsmash.control.full_racket_body_search_v4 import summarize_candidate
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases
from quadsmash.physics.full_racket_v4 import string_prism_gap
from quadsmash.tasks.stand.full_racket_train_env_v4 import RallyTrainingEnvV4
from quadsmash.tasks.stand.full_racket_env_v4 import full_racket_cfg
from quadsmash.tasks.stand.visual_geometry import export_robot_geometry
from quadsmash.usd_dependencies_v4 import audit_composed_asset


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def main():
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output/'INPUTS.json').exists(), 'Preserve every prior attempt'
    proposal = json.loads(args.proposals.read_text())
    source_path, evaluation = Path(proposal['source_policy']), Path(proposal['source_evaluation'])
    assert sha(source_path) == proposal['source_policy_sha256']
    assert all(sha(evaluation/name) == digest for name, digest in proposal['source_evaluation_sha256'].items())
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    candidates = proposal['candidates']; groups = len(candidates)
    assert groups == 8 and candidates[0]['overrides'] == {}
    assert [candidate['candidate_id'] for candidate in candidates] == list(range(groups))
    protocol_path = ROOT/'configs/g07_full_racket_eval_v4_r3.json'
    protocol = json.loads(protocol_path.read_text()); protocol['_root'] = str(ROOT)
    assert sha(ROOT/protocol['asset']['path']) == protocol['asset']['sha256'] == source['asset_sha256']
    assert all(sha(ROOT/name) == digest for name, digest in protocol['physical_configs'].items())
    cases, indices = training_cases(ROOT, protocol)
    assert len(cases) == 128 and indices == list(range(128))
    case_count, count = 128, groups*128
    cfg = full_racket_cfg(count, protocol['physics_dt_s']); cfg.seed = args.seed
    cfg.episode_length_s = protocol['episode_seconds']
    assert abs(cfg.sim.dt*cfg.decimation-protocol['policy_dt_s']) < 1e-12
    env = RallyTrainingEnvV4(cfg, cases=cases*groups, continue_after_miss=True)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    try:
        assert not env.robot.is_fixed_base
        controller = GroupedPreposeControllerV4(source, candidates, case_count, env.device)
        atomic(args.output/'asset_dependencies.json', audit_composed_asset(ROOT, cfg.robot.spawn.usd_path))
        export_robot_geometry(get_current_stage(), env.robot, args.output)
        observation, _ = wrapped.reset(); controller.reset()
        default = env.robot.data.default_joint_pos[0, env.joint_ids]
        limits = env.robot.data.joint_pos_limits[0, env.joint_ids]
        start_ids = env.episode_index.detach().clone(); data = env.robot.data
        roots = torch.cat((data.root_pos_w-env.scene.env_origins, data.root_quat_w,
                           data.root_lin_vel_w, data.root_ang_vel_w), -1).cpu().numpy()
        assert abs(roots[:, :3]-protocol['initialization']['root_position_m']).max() < 1e-5
        assert abs(roots[:, 3:7]-[1., 0., 0., 0.]).max() < 1e-5 and abs(roots[:, 7:]).max() < 1e-5
        assert data.joint_vel[:, env.joint_ids].abs().max() < 1e-5 and (observation['policy'][:, 775] == 0).all()
        atomic(args.output/'initial_ground_states.json', dict(episode_ids=start_ids.cpu().tolist(),
            root_state_local=roots.tolist(), joint_position_rad=data.joint_pos[:, env.joint_ids].cpu().tolist(),
            joint_velocity_rad_s=data.joint_vel[:, env.joint_ids].cpu().tolist(), all_clocks_zero=True))
        stream_ids = torch.tensor([128*group+i for group in range(groups) for i in range(2)], device=env.device)
        paths = sorted(ROOT.glob('quadsmash/**/*.py'))+[Path(__file__)]
        atomic(args.output/'INPUTS.json', dict(source_policy_sha256=sha(source_path), source_evaluation=str(evaluation),
            proposal_path=str(args.proposals), proposal_sha256=sha(args.proposals), candidates=candidates,
            protocol_sha256=sha(protocol_path), source_hashes={str(path.relative_to(ROOT)):sha(path) for path in paths},
            training_indices=indices, num_envs=count, candidate_count=groups, seed=args.seed, reset_noise='none',
            default_joint_position_rad=default.cpu().tolist(), joint_limits_rad=limits.cpu().tolist(),
            source_actors_and_normalizers_unchanged=True,
            episode_identity='Actual native global env_id and generation-start episode_index retained.',
            physical_initialization='Original ground0 for all groups; walking offsets are autonomous command goals.',
            formal_cases_read=False, formal_acceptance=False))
        finished = torch.zeros(count, dtype=torch.bool, device=env.device)
        closest = torch.full((count,), 10., device=env.device)
        poses, balls, clocks, valid, public, actions, public_valid = [], [], [], [], [], [], []
        rally_start, jump_start, contact_start = len(env.rally_records), len(env.episode_records), len(env.contact_records)
        ticks = round(protocol['episode_seconds']/protocol['policy_dt_s'])
        with torch.inference_mode():
            for tick in range(ticks):
                obs, data = observation['policy'], env.robot.data
                active = (~finished) & env.launch_done & env.rally.in_play
                gap = string_prism_gap(env.shuttle.data.root_pos_w, data.body_link_pos_w[:, env.racket_body_id],
                    data.body_link_quat_w[:, env.racket_body_id], env.racket_config, cfg.shuttle_radius_m).clamp_min(0.)
                closest = torch.where(active, torch.minimum(closest, gap), closest)
                action = controller.action(observation, default, limits)
                assert action.shape == (count, 18) and torch.isfinite(action).all() and (abs(action) <= 1).all()
                public.append(obs[stream_ids].cpu().numpy().copy()); actions.append(action[stream_ids].cpu().numpy().copy())
                public_valid.append((~finished[stream_ids]).cpu().numpy().copy())
                if tick % 2 == 0:
                    poses.append(torch.cat((data.body_link_pos_w-env.scene.env_origins[:, None], data.body_link_quat_w), -1).cpu().numpy())
                    balls.append((env.shuttle.data.root_pos_w-env.scene.env_origins).cpu().numpy())
                    clocks.append(obs[:, 775].cpu().numpy().copy()); valid.append((~finished).cpu().numpy().copy())
                observation, _, done, _ = wrapped.step(action)
                finished |= done.bool()
                ids = torch.where(done.bool())[0]
                if len(ids):
                    controller.reset(ids)
                if (tick+1) % 100 == 0 or stopping or bool(finished.all()):
                    progress = dict(control_ticks=tick+1, first_episodes_complete=int(finished.sum()),
                        actual_control_env_steps=(tick+1)*count, formal_acceptance=False)
                    atomic(args.output/'progress.json', progress)
                    atomic(args.output/'rallies.partial.json', env.rally_records[rally_start:])
                    print(json.dumps(progress), flush=True)
                if stopping or bool(finished.all()):
                    break
        np.savez_compressed(args.output/'all_first_episodes.npz', link_poses=np.stack(poses),
            ball_position_m=np.stack(balls), episode_clock_s=np.stack(clocks), valid_first_episode=np.stack(valid))
        np.savez_compressed(args.output/'public_and_actions.npz', observations=np.stack(public), actions=np.stack(actions),
            valid_first_episode=np.stack(public_valid), native_env_ids=stream_ids.cpu().numpy())
        rallies, jumps = env.rally_records[rally_start:], env.episode_records[jump_start:]
        atomic(args.output/'rallies.json', rallies); atomic(args.output/'jumps.json', jumps)
        atomic(args.output/'contacts.json', env.contact_records[contact_start:]); atomic(args.output/'causal_plans.json', controller.logs)
        if stopping and not bool(finished.all()):
            atomic(args.output/'interrupted.json', dict(actual_control_env_steps=(tick+1)*count,
                first_episodes_complete=int(finished.sum()), full_native_resume_claimed=False, formal_acceptance=False))
            return
        assert len(rallies) == len(jumps) and bool(finished.all())
        first = {}
        for ordinal, (rally, jump) in enumerate(zip(rallies, jumps)):
            lane = rally['env_id']; assert jump['env_id'] == lane
            if rally['episode_index'] == int(start_ids[lane]):
                assert lane not in first
                first[lane] = dict(rally, jump=jump, candidate_id=lane//128, source_training_index=lane % 128,
                    case_id=cases[lane % 128]['case_id'], ground_init=True, closest_string_gap_m=float(closest[lane]),
                    logger_ordinal=ordinal)
        assert set(first) == set(range(count))
        rows = [first[lane] for lane in range(count)]
        summaries = [summarize_candidate(rows[group*128:(group+1)*128], protocol) for group in range(groups)]
        atomic(args.output/'episodes.json', rows)
        atomic(args.output/'candidate_summaries.json', summaries)
        assert all(summary['numerical_failures'] == 0 for summary in summaries), 'Preserve numerical faults; no auto retry'
        ranked = sorted(range(groups), key=lambda group: tuple(summaries[group]['rank']), reverse=True)
        best = ranked[0]
        atomic(args.output/'metrics.json', dict(summaries=summaries, ranked_candidate_indices=ranked,
            best_candidate_id=best, actual_control_env_steps=(tick+1)*count, formal_cases_read=False, formal_acceptance=False))
        policy = dict(schema=SCHEMA, source_public_policy=source, source_policy_sha256=sha(source_path),
            parameters=source['parameters'].clone(), asset_sha256=source['asset_sha256'],
            runtime_overrides=candidates[best]['overrides'], native_execution=True, performance_inherited=False,
            actual_native_search=dict(output=str(args.output), candidate_id=best, summary=summaries[best],
                                     proposal_sha256=sha(args.proposals)), formal_acceptance=False)
        temporary = args.output/'best_candidate.pt.tmp'; torch.save(policy, temporary)
        temporary.replace(args.output/'best_candidate.pt')
        atomic(args.output/'completed.json', dict(env_steps=(tick+1)*count))
        print(json.dumps(dict(best=best, summaries=summaries)), flush=True)
    finally:
        env.close()


try:
    main()
except BaseException:
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'error.txt').write_text(traceback.format_exc())
    raise
finally:
    app.close(skip_cleanup=True)
