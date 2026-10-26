#!/usr/bin/env python3
"""Measure eight residual-layer gain ablations under the admitted origin0 layout.

Each candidate gets an ordinary ground reset in one128-lane native environment.
Native episode identities are preserved across these sequential comparisons.
The source weights, physics, rules and all success thresholds stay unchanged.
"""
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
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--source-evaluation', type=Path, required=True)
parser.add_argument('--proposals', type=Path, required=True)
parser.add_argument('--precision-receipt', type=Path, required=True)
parser.add_argument('--seed', type=int, default=750801)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaacsim.core.utils.stage import get_current_stage
from quadsmash.control.full_racket_residual_gain_v4 import ResidualGainControllerV4, SCHEMA, gain_candidates
from quadsmash.control.full_racket_body_search_v4 import summarize_candidate
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases, summarize_training
from quadsmash.physics.full_racket_v4 import string_prism_gap
from quadsmash.tasks.stand.full_racket_train_env_v4 import RallyTrainingEnvV4
from quadsmash.tasks.stand.full_racket_origin0_v4 import full_racket_origin0_cfg
from quadsmash.tasks.stand.visual_geometry import export_robot_geometry
from quadsmash.usd_dependencies_v4 import audit_composed_asset


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def save_torch(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def main():
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'INPUTS.json').exists():
        raise ValueError('Require a fresh output; preserve all prior attempts')
    protocol_path = ROOT/'configs/g07_full_racket_eval_v4_r3_origin0.json'
    protocol = json.loads(protocol_path.read_text())
    protocol['_root'] = str(ROOT)
    assert sha(ROOT/protocol['asset']['path']) == protocol['asset']['sha256']
    assert all(sha(ROOT/path) == digest for path,digest in protocol['physical_configs'].items())
    cases, indices = training_cases(ROOT, protocol)
    assert indices == list(range(128)) and len(cases) == 128
    read = lambda name: json.loads((args.source_evaluation/name).read_text())
    source_inputs, source_metrics, job = read('INPUTS.json'), read('metrics.json'), read('job.json')
    assert job['status'] == 'SUCCEEDED' and 0 < read('completed.json')['env_steps'] <= job['charged_env_steps']
    assert source_inputs['policy_sha256'] == source_metrics['policy_sha256'] == sha(args.source)
    assert source_inputs['protocol_sha256'] == source_metrics['protocol_sha256'] == sha(protocol_path)
    assert source_inputs['training_indices'] == indices and source_metrics['noise_level'] == 'none'
    assert source_inputs['origin_spacing_m'] == 0.
    source_contract = json.loads(protocol_path.read_text())
    source_contract['_root'] = str(ROOT)
    source_summary = summarize_training(read('episodes.json'), cases, indices, source_contract)
    assert all(source_metrics[key] == value for key,value in source_summary.items())
    assert source_summary['numerical_failures'] == 0
    source = torch.load(args.source, map_location='cpu', weights_only=False)
    assert source['asset_sha256'] == protocol['asset']['sha256'] and source.get('native_execution')
    proposal = json.loads(args.proposals.read_text())
    assert proposal['source_policy_sha256'] == sha(args.source)
    assert all(sha(args.source_evaluation/name) == digest for name,digest in proposal['source_evaluation_sha256'].items())
    candidates = proposal['candidates']
    assert candidates == gain_candidates()
    count = 128
    cfg = full_racket_origin0_cfg(count, protocol, args.precision_receipt)
    cfg.seed = args.seed
    cfg.episode_length_s = protocol['episode_seconds']
    assert abs(cfg.sim.dt*cfg.decimation-protocol['policy_dt_s']) < 1e-12
    env = RallyTrainingEnvV4(cfg, cases=cases, continue_after_miss=True)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        assert not env.robot.is_fixed_base
        atomic(args.output/'asset_dependencies.json', audit_composed_asset(ROOT, cfg.robot.spawn.usd_path))
        export_robot_geometry(get_current_stage(), env.robot, args.output)
        default = env.robot.data.default_joint_pos[0, env.joint_ids]
        limits = env.robot.data.joint_pos_limits[0, env.joint_ids]
        paths = sorted(ROOT.glob('quadsmash/**/*.py')) + [Path(__file__)]
        atomic(args.output/'INPUTS.json', dict(source_policy_sha256=sha(args.source),
            source_evaluation=str(args.source_evaluation), source_original_summary=source_summary,
            source_evaluation_sha256={name:sha(args.source_evaluation/name) for name in
                ('INPUTS.json','metrics.json','episodes.json','job.json','completed.json')},
            protocol_sha256=sha(protocol_path),
            source_hashes={str(p.relative_to(ROOT)):sha(p) for p in paths},
            training_indices=indices, candidate_residual_gain_definitions=candidates, num_envs=count, seed=args.seed,
            origin_precision_receipt_sha256=sha(args.precision_receipt),origin_spacing_m=0.,
            proposal_sha256=sha(args.proposals),source_performance_inherited=False,
            reset_noise='none', comparison='Eight sequential128-lane ground resets, all original TRAIN128 per candidate, admitted co-located independent origins.',
            original_source_actors_and_normalizers_preserved=True,
            changed_algorithm='Shared per-layer leg/arm bounded residual gains on the admitted numerical origin layout.',
            unchanged='Original physical collision geometry, mass, hardware limits, all128 cases and success criteria; source policy preserved without inherited performance.',
            episode_identity='Native generation-start episode_index is preserved; never renumbered.',
            default_joint_position_rad=default.cpu().tolist(), joint_limits_rad=limits.cpu().tolist(),
            formal_cases_read=False, formal_acceptance=False))
        ticks = round(protocol['episode_seconds']/protocol['policy_dt_s'])
        history, best, best_rank = [], None, None
        for candidate_id, candidate in enumerate(candidates):
            prior_rallies, prior_jumps = len(env.rally_records), len(env.episode_records)
            observation, _ = wrapped.reset()
            controller = ResidualGainControllerV4(source, candidate['gains'], count, env.device)
            controller.reset()
            suffix = f'{candidate_id:03d}'
            atomic(args.output/f'reset_transition_rallies_{suffix}.json', env.rally_records[prior_rallies:])
            atomic(args.output/f'reset_transition_jumps_{suffix}.json', env.episode_records[prior_jumps:])
            start_ids = env.episode_index.detach().clone()
            data = env.robot.data
            initial = dict(episode_ids=start_ids.cpu().tolist(),
                actual_environment_origins_m=env.scene.env_origins.cpu().tolist(),
                root_state_local=torch.cat((data.root_pos_w-env.scene.env_origins,data.root_quat_w,
                    data.root_lin_vel_w,data.root_ang_vel_w),-1).cpu().tolist(),
                joint_position_rad=data.joint_pos[:,env.joint_ids].cpu().tolist(),
                joint_velocity_rad_s=data.joint_vel[:,env.joint_ids].cpu().tolist(),
                all_clocks_zero=bool((observation['policy'][:,775] == 0).all()))
            roots = np.asarray(initial['root_state_local'])
            assert not env.scene.env_origins.any(), 'Require actual co-located independent origins'
            assert roots.shape == (128,13) and np.isfinite(roots).all()
            assert np.max(abs(roots[:,:3]-protocol['initialization']['root_position_m'])) < 1e-5
            assert np.max(abs(roots[:,3:7]-[1,0,0,0])) < 1e-5 and np.max(abs(roots[:,7:])) < 1e-5
            assert np.max(abs(np.asarray(initial['joint_velocity_rad_s']))) < 1e-5 and initial['all_clocks_zero']
            atomic(args.output/f'initial_ground_states_{suffix}.json', initial)
            finished = torch.zeros(count, dtype=torch.bool, device=env.device)
            closest = torch.full((count,), 10., device=env.device)
            poses, balls, clocks, valid, public, actions, public_valid = [], [], [], [], [], [], []
            rally_start, jump_start, contact_start = len(env.rally_records),len(env.episode_records),len(env.contact_records)
            with torch.inference_mode():
                for tick in range(ticks):
                    obs, data = observation['policy'], env.robot.data
                    active = (~finished) & env.launch_done & env.rally.in_play
                    gap = string_prism_gap(env.shuttle.data.root_pos_w,
                        data.body_link_pos_w[:,env.racket_body_id],data.body_link_quat_w[:,env.racket_body_id],
                        env.racket_config,cfg.shuttle_radius_m).clamp_min(0.)
                    closest = torch.where(active, torch.minimum(closest,gap),closest)
                    action = controller.action(observation,default,limits)
                    assert action.shape == (count,18) and torch.isfinite(action).all() and (abs(action) <= 1).all()
                    public.append(obs[:2].cpu().numpy().copy())
                    actions.append(action[:2].cpu().numpy().copy())
                    public_valid.append((~finished[:2]).cpu().numpy().copy())
                    if tick % 2 == 0:
                        poses.append(torch.cat((data.body_link_pos_w-env.scene.env_origins[:,None],
                                              data.body_link_quat_w),-1).cpu().numpy())
                        balls.append((env.shuttle.data.root_pos_w-env.scene.env_origins).cpu().numpy())
                        clocks.append(obs[:,775].cpu().numpy().copy())
                        valid.append((~finished).cpu().numpy().copy())
                    observation, _, done, _ = wrapped.step(action)
                    finished |= done.bool()
                    reset_ids = torch.where(done.bool())[0]
                    if len(reset_ids):
                        controller.reset(reset_ids)
                    if (tick+1) % 100 == 0 or stopping or bool(finished.all()):
                        progress = dict(candidate_id=candidate_id,candidate_name=candidate['name'],control_ticks=tick+1,
                            first_episodes_complete=int(finished.sum()),
                            actual_control_env_steps=sum(e['env_steps'] for e in history)+(tick+1)*count,
                            formal_acceptance=False)
                        atomic(args.output/'progress.json',progress)
                        atomic(args.output/'rallies.partial.json',env.rally_records[rally_start:])
                        print(json.dumps(progress),flush=True)
                    if stopping or bool(finished.all()):
                        break
            np.savez_compressed(args.output/f'all_first_episodes_{suffix}.npz', link_poses=np.stack(poses),
                ball_position_m=np.stack(balls),episode_clock_s=np.stack(clocks),valid_first_episode=np.stack(valid))
            np.savez_compressed(args.output/f'public_and_actions_{suffix}.npz', observations=np.stack(public),
                actions=np.stack(actions),valid_first_episode=np.stack(public_valid))
            rallies, jumps = env.rally_records[rally_start:],env.episode_records[jump_start:]
            atomic(args.output/f'rallies_{suffix}.json',rallies)
            atomic(args.output/f'jumps_{suffix}.json',jumps)
            atomic(args.output/f'contacts_{suffix}.json',env.contact_records[contact_start:])
            atomic(args.output/f'causal_plans_{suffix}.json',controller.logs)
            if stopping and not bool(finished.all()):
                atomic(args.output/'interrupted.json',dict(candidate_id=candidate_id,
                    actual_control_env_steps=sum(e['env_steps'] for e in history)+(tick+1)*count,
                    resume='Start this interrupted candidate from ordinary ground; no physical-state resume claimed.',
                    formal_acceptance=False))
                return
            assert len(rallies) == len(jumps), 'Preserve and investigate original logger mismatch'
            first = {}
            for ordinal,(rally,jump) in enumerate(zip(rallies,jumps)):
                assert rally['env_id'] == jump['env_id']
                lane = rally['env_id']
                if rally['episode_index'] == int(start_ids[lane]):
                    assert lane not in first
                    first[lane] = dict(rally,jump=jump,case_id=cases[lane]['case_id'],
                        ground_init=True,source_training_index=lane,candidate_id=candidate_id,
                        closest_string_gap_m=float(closest[lane]),logger_ordinal=ordinal)
            assert len(first) == count and bool(finished.all())
            rows = [first[i] for i in range(count)]
            summary = summarize_candidate(rows,protocol)
            atomic(args.output/f'episodes_{suffix}.json',rows)
            entry = dict(candidate_id=candidate_id,candidate=candidate,summary=summary,
                control_ticks=tick+1,env_steps=(tick+1)*count)
            history.append(entry)
            if summary['numerical_failures']:
                atomic(args.output/'numerical_failure.json',entry)
                raise RuntimeError('Preserve actual numerical fault; no automatic candidate retry')
            rank = tuple(summary['rank'])
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best = dict(schema=SCHEMA,source_public_policy=source,source_policy_sha256=sha(args.source),
                    parameters=source['parameters'].clone(),asset_sha256=source['asset_sha256'],residual_gains_outer_to_inner=candidate['gains'],
                    numerical_execution=protocol['numerical_execution'],protocol_sha256=sha(protocol_path),
                    origin_precision_receipt_sha256=sha(args.precision_receipt),
                    native_execution=True,performance_inherited=False,formal_acceptance=False,
                    actual_native_search=dict(output=str(args.output),candidate_id=candidate_id,
                        original128_summary=summary),kind='Measured residual-layer/group gain ablations under admitted origin0 layout')
                save_torch(args.output/'best_candidate.pt',best)
            atomic(args.output/'metrics.json',dict(history=history,best_rank=best_rank,
                actual_control_env_steps=sum(e['env_steps'] for e in history),
                source_actor_weights_and_normalizers_unchanged=True,formal_cases_read=False,formal_acceptance=False))
            print(json.dumps(entry),flush=True)
            if stopping:
                return
        atomic(args.output/'completed.json',dict(env_steps=sum(e['env_steps'] for e in history)))
    finally:
        env.close()


try:
    main()
except BaseException:
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'error.txt').write_text(traceback.format_exc())
    raise
finally:
    app.close(skip_cleanup=True)
