"""Execute public air-only countermotion proposals, then permit an independent cold replay.

Actual runtime imports come from the unchanged recovery-v5 worktree. Search
uses three sequential ground resets of all original128 TRAIN cases. No formal
panel or geometry is changed. Native episode identifiers are never rewritten.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import signal
import sys
import traceback

from source_support import verify_prior_source

MAIN = Path(__file__).resolve().parents[3]
ROOT = MAIN.parent / 'unitree_smash_contact_recovery'
HERE = Path(__file__).resolve().parent
PROPOSALS = MAIN / 'reports/G07_full_racket_v5/public_air_countermotion_20261027'
sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher
from scripts.origin0_launcher_parser_v4 import add_origin0_launcher_args

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--source-evaluation', type=Path, required=True)
parser.add_argument('--precision-receipt', type=Path, required=True)
parser.add_argument('--mode', choices=['search', 'cold'], required=True)
parser.add_argument('--seed', type=int, default=724201)
add_origin0_launcher_args(parser, AppLauncher)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaacsim.core.utils.stage import get_current_stage
from quadsmash.control.full_racket_body_search_v4 import summarize_candidate
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases, training_noise, summarize_training
from quadsmash.evaluation_v4 import FAULTS, independent_episode
from quadsmash.physics.full_racket_v4 import string_prism_gap
from quadsmash.rl.contact_recovery_source_v5 import snapshot_execution_sources, verify_recovery_source
from quadsmash.tasks.stand.contact_recovery_origin0_v5 import contact_recovery_origin0_cfg
from quadsmash.tasks.stand.full_racket_eval_env_v4 import FrozenPanelEnv
from quadsmash.tasks.stand.visual_geometry import export_robot_geometry
from quadsmash.usd_dependencies_v4 import audit_composed_asset

SCHEMA = 'g07-public-air-countermotion-v5'
RECORDED = [0, 29, 69, 100, 112, 123]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def module():
    spec = importlib.util.spec_from_file_location('native_air_countermotion_controller', PROPOSALS / 'native_factory.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def generation_summary(rows, cases, indices, start_ids, protocol):
    assert len(rows) == len(cases) == len(indices) == len(start_ids) == 128
    for lane, (row, case) in enumerate(zip(rows, cases)):
        assert row['env_id'] == lane and row['episode_index'] == start_ids[lane]
        assert row['case_id'] == case['case_id'] and row['source_training_index'] == indices[lane]
        assert row['ground_init'] and row['active_rally_launch_count'] <= 1
    checked = [independent_episode(row, protocol) for row in rows]
    result = summarize_candidate(rows, protocol)
    result.update(airborne_contact_rate=result['airborne_contacts'] / 128,
        complete_success_rate=result['complete'] / 128,
        conditional_recovery_rate=(result['conditional_recovery_count'] / result['airborne_contacts']
                                   if result['airborne_contacts'] else 0.),
        actual_racket_contacts=sum(row['had_contact'] for row in rows),
        crossed_net=sum(row['crossed_net'] for row in rows),
        landed_legal_court=sum(row['landed_legal_court'] for row in rows),
        landed_target=sum(row['landed_target'] for row in rows),
        fault_counts={name: sum(row[name] for row in rows) for name in FAULTS},
        native_generation_start_episode_ids=start_ids,
        original128_case_order_preserved=True, formal_cases_read=False, formal_eligible=False)
    assert result['complete'] == sum(row['complete'] for row in checked)
    return result


def main():
    assert not args.output.exists() or not (args.output / 'INPUTS.json').exists()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = ROOT / 'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'
    protocol = read(protocol_path)
    protocol['_root'] = str(ROOT)
    cases, indices = training_cases(ROOT, protocol)
    assert indices == list(range(128)) and args.seed == 724201
    checkpoint = torch.load(args.source, map_location='cpu', weights_only=False)
    proposal = read(PROPOSALS / 'PROPOSAL.json')
    cpu = read(PROPOSALS / 'CPU_CONTROLLER_CHECK.json')
    factory_check = read(PROPOSALS / 'NATIVE_FACTORY_CPU_CHECK.json')
    assert factory_check['factory_sha256'] == sha(PROPOSALS / 'native_factory.py')
    assert len(cpu['results']) == len(proposal['candidates']) == 3
    assert cpu['results'][0]['maximum_source_action_difference'] == 0.
    assert cpu['results'][0]['maximum_original_GPU_action_difference'] <= 5e-4
    assert factory_check['actual_baseline_CPU_vs_GPU_max_error'] <= 5e-4
    assert all(row['public_inputs_and_source_weights_unchanged'] for row in cpu['results'])
    if args.mode == 'search':
        source_check = verify_prior_source(args.source, checkpoint, args.source_evaluation, ROOT, sha(protocol_path))
        candidates = proposal['candidates']
        assert candidates == module().candidates()
        assert [c['maximum_velocity_change_rad_s'] for c in candidates] == [r['maximum_velocity_change_rad_s'] for r in cpu['results']]
        base = checkpoint
    else:
        assert checkpoint['schema'] == SCHEMA and checkpoint['native_execution']
        assert checkpoint['protocol_sha256'] == sha(protocol_path)
        assert checkpoint['factory_sha256'] == sha(PROPOSALS / 'native_factory.py')
        selected = checkpoint['actual_native_search']
        search = Path(selected['output'])
        assert read(search / 'job.json')['status'] == 'SUCCEEDED'
        search_metrics = read(search / 'metrics.json')
        assert len(search_metrics['history']) == 3
        assert 0 < read(search / 'completed.json')['env_steps'] == search_metrics['actual_control_env_steps'] <= 307200
        assert sha(args.source) == sha(search / 'best_candidate.pt')
        candidates = [dict(candidate_id=selected['candidate_id'], name='independent_cold_selected_air_countermotion',
                           **checkpoint['air_countermotion'])]
        base = checkpoint['source_public_policy']
        assert torch.equal(checkpoint['parameters'], base['parameters'])
        source_check = dict(search=str(search), selected=selected, policy_sha256=sha(args.source),
                            performance_inherited=False, formal_acceptance=False)
    assert base['asset_sha256'] == protocol['asset']['sha256']
    assert sha(ROOT / protocol['asset']['path']) == protocol['asset']['sha256']
    assert all(sha(ROOT / name) == digest for name, digest in protocol['physical_configs'].items())
    snapshot = snapshot_execution_sources(ROOT, args.output)
    extra = [Path(__file__), PROPOSALS / 'native_factory.py', PROPOSALS / 'PROPOSAL.json',
             PROPOSALS / 'CPU_CONTROLLER_CHECK.json', HERE / 'run.py', HERE / 'PLAN.json', HERE / 'PRE_NATIVE_CPU_CHECK.json',
             HERE / 'source_support.py', HERE / 'SELECTED_SOURCE_CPU_CHECK.json',
             HERE.parent / 'early_tangent_intercept_20261027/early_lateral_controller.py',
             HERE.parent / 'origin0_measured_swing_velocity_20261027/swing_controller.py',
             HERE.parent / 'windowed_swing_20261027/window_controller.py',
             HERE.parent / 'recovery_motion_prior_20261027/SEED_RECEIPT.json',
             HERE.parent / 'recovery_motion_prior_20261027/recorded_recovery_prior.npz',
             HERE.parent / 'recovery_prior_native_20261027/prior_controller.py',
             PROPOSALS / 'countermotion.py', PROPOSALS / 'NATIVE_FACTORY_CPU_CHECK.json',
             HERE.parent / 'centroidal_recovery_model_20261027/known_model.py',
             HERE.parent / 'centroidal_recovery_model_20261027/MODEL.json',
             HERE.parent / 'centroidal_recovery_model_20261027/ACTUAL_MODEL_CHECK.json']
    extra_hashes = {str(path): sha(path) for path in extra}
    atomic(args.output / 'extra_execution_source_hashes.json', extra_hashes)
    for path in extra:
        directory = args.output / 'extra_execution_sources'
        directory.mkdir(exist_ok=True)
        shutil.copyfile(path, directory / path.name)
    noise = training_noise(128, dict(joint_position_rad=0., joint_velocity_rad_s=0., root_xy_m=0., root_yaw_rad=0.), args.seed)
    cfg = contact_recovery_origin0_cfg(128, protocol, args.precision_receipt)
    cfg.seed = args.seed
    cfg.episode_length_s = protocol['episode_seconds']
    assert abs(cfg.sim.dt * cfg.decimation - protocol['policy_dt_s']) < 1e-12
    env = FrozenPanelEnv(cfg, cases=cases, reset_noise=noise,
        contact_flight_recovery=protocol['recovery_association']['schema'])
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        assert not env.robot.is_fixed_base
        atomic(args.output / 'asset_dependencies.json', audit_composed_asset(ROOT, cfg.robot.spawn.usd_path))
        export_robot_geometry(get_current_stage(), env.robot, args.output)
        default = env.robot.data.default_joint_pos[0, env.joint_ids]
        limits = env.robot.data.joint_pos_limits[0, env.joint_ids]
        atomic(args.output / 'INPUTS.json', dict(mode=args.mode, source_policy_sha256=sha(args.source),
            protocol_sha256=sha(protocol_path), source_check=source_check, **snapshot,
            proposal_sha256=sha(PROPOSALS / 'PROPOSAL.json'), candidates=candidates,
            extra_execution_source_hashes=extra_hashes,
            training_indices=indices, seed=args.seed, reset_noise='none', origin_spacing_m=0.,
            origin_precision_receipt_sha256=sha(args.precision_receipt),
            default_joint_position_rad=default.cpu().tolist(), joint_limits_rad=limits.cpu().tolist(),
            public_training_indices=RECORDED, source_actors_and_normalizers_preserved=True,
            physical_rules_target_and_thresholds_unchanged=True,
            formal_cases_read=False, formal_acceptance=False))
        ticks = round(protocol['episode_seconds'] / protocol['policy_dt_s'])
        assert ticks == 800
        history, best_rank = [], None
        factory = module()
        model_record = read(HERE.parent / 'centroidal_recovery_model_20261027/MODEL.json')
        if args.mode == 'cold':
            assert checkpoint['known_model_sha256'] == sha(HERE.parent / 'centroidal_recovery_model_20261027/MODEL.json')
            assert checkpoint['known_model'] == model_record
            assert checkpoint['counter_core_sha256'] == sha(PROPOSALS / 'countermotion.py')
        for candidate in candidates:
            candidate_id = candidate['candidate_id']
            suffix = f'{candidate_id:03d}'
            prior_rallies, prior_jumps = len(env.rally_records), len(env.episode_records)
            observation, _ = wrapped.reset()
            controller = factory.make_controller(base, 128, env.device, candidate['maximum_velocity_change_rad_s'])
            controller.reset()
            start_ids = env.episode_index.cpu().tolist()
            atomic(args.output / f'reset_transition_rallies_{suffix}.json', env.rally_records[prior_rallies:])
            atomic(args.output / f'reset_transition_jumps_{suffix}.json', env.episode_records[prior_jumps:])
            data = env.robot.data
            initial = dict(episode_ids=start_ids, actual_environment_origins_m=env.scene.env_origins.cpu().tolist(),
                root_state_local=torch.cat((data.root_pos_w - env.scene.env_origins, data.root_quat_w,
                    data.root_lin_vel_w, data.root_ang_vel_w), -1).cpu().tolist(),
                joint_position_rad=data.joint_pos[:, env.joint_ids].cpu().tolist(),
                joint_velocity_rad_s=data.joint_vel[:, env.joint_ids].cpu().tolist(),
                all_clocks_zero=bool((observation['policy'][:, 775] == 0).all()))
            roots = np.asarray(initial['root_state_local'])
            assert not env.scene.env_origins.any() and initial['all_clocks_zero']
            assert np.max(abs(roots[:, :3] - protocol['initialization']['root_position_m'])) < 1e-5
            assert np.max(abs(roots[:, 3:7] - [1, 0, 0, 0])) < 1e-5
            assert np.max(abs(roots[:, 7:])) < 1e-5
            assert np.max(abs(np.asarray(initial['joint_velocity_rad_s']))) < 1e-5
            atomic(args.output / f'initial_ground_states_{suffix}.json', initial)
            finished = torch.zeros(128, dtype=torch.bool, device=env.device)
            closest = torch.full((128,), 10., device=env.device)
            poses, balls, clocks, valid, public, actions, public_valid = [], [], [], [], [], [], []
            rally_start, jump_start, contact_start = len(env.rally_records), len(env.episode_records), len(env.contact_records)
            with torch.inference_mode():
                for tick in range(ticks):
                    obs, data = observation['policy'], env.robot.data
                    active = (~finished) & env.launch_done & env.rally.in_play
                    gap = string_prism_gap(env.shuttle.data.root_pos_w,
                        data.body_link_pos_w[:, env.racket_body_id], data.body_link_quat_w[:, env.racket_body_id],
                        env.racket_config, cfg.shuttle_radius_m).clamp_min(0.)
                    closest = torch.where(active, torch.minimum(closest, gap), closest)
                    action = controller.action(observation, default, limits)
                    assert action.shape == (128, 18) and torch.isfinite(action).all() and (abs(action) <= 1).all()
                    public.append(obs[RECORDED].cpu().numpy().copy())
                    actions.append(action[RECORDED].cpu().numpy().copy())
                    public_valid.append((~finished[RECORDED]).cpu().numpy().copy())
                    if tick % 2 == 0:
                        pose = torch.cat((data.body_link_pos_w - env.scene.env_origins[:, None], data.body_link_quat_w), -1).cpu().numpy()
                        ball = (env.shuttle.data.root_pos_w - env.scene.env_origins).cpu().numpy()
                        clock = obs[:, 775].cpu().numpy().copy()
                        assert all(np.isfinite(value).all() for value in [pose, ball, clock])
                        poses.append(pose)
                        balls.append(ball)
                        clocks.append(clock)
                        valid.append((~finished).cpu().numpy().copy())
                    observation, _, done, _ = wrapped.step(action)
                    finished |= done.bool()
                    ids = torch.where(done.bool())[0]
                    if len(ids):
                        controller.reset(ids)
                    if (tick + 1) % 100 == 0 or stopping or bool(finished.all()):
                        progress = dict(candidate_id=candidate_id, candidate_name=candidate['name'],
                            control_ticks=tick + 1, first_episodes_complete=int(finished.sum()),
                            completed_candidates=len(history), actual_control_env_steps=sum(h['env_steps'] for h in history) + (tick + 1) * 128,
                            formal_acceptance=False)
                        atomic(args.output / 'progress.json', progress)
                        print(json.dumps(progress), flush=True)
                    if stopping or bool(finished.all()):
                        break
            np.savez_compressed(args.output / f'all_first_episodes_{suffix}.npz', link_poses=np.stack(poses),
                ball_position_m=np.stack(balls), episode_clock_s=np.stack(clocks), valid_first_episode=np.stack(valid))
            np.savez_compressed(args.output / f'public_and_actions_{suffix}.npz', observations=np.stack(public),
                actions=np.stack(actions), valid_first_episode=np.stack(public_valid), training_indices=np.array(RECORDED))
            rallies, jumps = env.rally_records[rally_start:], env.episode_records[jump_start:]
            atomic(args.output / f'rallies_{suffix}.json', rallies)
            atomic(args.output / f'jumps_{suffix}.json', jumps)
            atomic(args.output / f'contacts_{suffix}.json', env.contact_records[contact_start:])
            atomic(args.output / f'causal_plans_{suffix}.json', controller.logs)
            if stopping and not bool(finished.all()):
                atomic(args.output / 'interrupted.json', progress)
                return
            assert len(rallies) == len(jumps)
            first = {}
            for ordinal, (rally, jump) in enumerate(zip(rallies, jumps)):
                assert rally['env_id'] == jump['env_id']
                lane = rally['env_id']
                if rally['episode_index'] == start_ids[lane]:
                    assert lane not in first
                    first[lane] = dict(rally, jump=jump, case_id=cases[lane]['case_id'],
                        ground_init=True, source_training_index=lane, candidate_id=candidate_id,
                        closest_string_gap_m=float(closest[lane]), logger_ordinal=ordinal)
            assert len(first) == 128 and bool(finished.all())
            rows = [first[i] for i in range(128)]
            summary = generation_summary(rows, cases, indices, start_ids, protocol)
            if args.mode == 'cold':
                assert start_ids == [0] * 128
                cold = summarize_training(rows, cases, indices, protocol)
                assert all(summary[key] == cold[key] for key in ['complete', 'airborne_contacts', 'conditional_recovery_count', 'numerical_failures'])
                atomic(args.output / 'episodes.json', rows)
            atomic(args.output / f'episodes_{suffix}.json', rows)
            entry = dict(candidate_id=candidate_id, candidate=candidate, summary=summary,
                control_ticks=tick + 1, env_steps=(tick + 1) * 128)
            history.append(entry)
            if summary['numerical_failures']:
                atomic(args.output / 'numerical_failure.json', entry)
                raise RuntimeError('Actual numerical failure; preserve without automatic retry')
            if args.mode == 'search':
                policy = dict(schema=SCHEMA, source_public_policy=base, source_policy_sha256=sha(args.source),
                    parameters=base['parameters'].clone(), asset_sha256=base['asset_sha256'],
                    air_countermotion=dict(maximum_velocity_change_rad_s=candidate['maximum_velocity_change_rad_s']),
                    known_model=model_record,
                    known_model_sha256=sha(HERE.parent / 'centroidal_recovery_model_20261027/MODEL.json'),
                    counter_core_sha256=sha(PROPOSALS / 'countermotion.py'), protocol_sha256=sha(protocol_path),
                    factory_sha256=sha(PROPOSALS / 'native_factory.py'), native_execution=True,
                    performance_inherited=False, formal_acceptance=False,
                    actual_native_search=dict(output=str(args.output), candidate_id=candidate_id, original128_summary=summary))
                path = args.output / f'candidate_{suffix}.pt'
                assert not path.exists()
                torch.save(policy, path)
                rank = tuple(summary['rank'])
                if best_rank is None or rank > best_rank:
                    best_rank = rank
                    shutil.copyfile(path, args.output / 'best_candidate.pt')
            atomic(args.output / 'metrics.json', dict(history=history, best_rank=best_rank,
                actual_control_env_steps=sum(h['env_steps'] for h in history),
                source_actor_weights_and_normalizers_unchanged=True, formal_cases_read=False, formal_acceptance=False))
            print(json.dumps(entry), flush=True)
            if stopping:
                return
        atomic(args.output / 'completed.json', dict(env_steps=sum(h['env_steps'] for h in history)))
    finally:
        env.close()


try:
    main()
except BaseException:
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'error.txt').write_text(traceback.format_exc())
    raise
finally:
    app.close(skip_cleanup=True)
