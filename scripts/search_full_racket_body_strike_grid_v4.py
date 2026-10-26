#!/usr/bin/env python3
"""Execute the declared fixed16-candidate TRAIN strike grid around measured body4."""
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
parser.add_argument('--comparison', type=Path, required=True)
parser.add_argument('--generations', type=int, choices=(1,), default=1)
parser.add_argument('--seed', type=int, default=750101)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaacsim.core.utils.stage import get_current_stage
from quadsmash.control.full_racket_fixed_actor_body_v4 import FixedActorBodyControllerV4, SCHEMA
from quadsmash.control.full_racket_body_search_v4 import summarize_candidate
from quadsmash.control.full_racket_body_strike_grid_v4 import (
    CANDIDATE_COUNT, COORDINATES, population_and_labels)
from scripts.initialize_centered_policy_v4 import same_tree
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases
from quadsmash.tasks.stand.full_racket_train_env_v4 import RallyTrainingEnvV4
from quadsmash.tasks.stand.full_racket_env_v4 import full_racket_cfg
from quadsmash.tasks.stand.visual_geometry import export_robot_geometry
from quadsmash.physics.full_racket_v4 import string_prism_gap
from quadsmash.usd_dependencies_v4 import audit_composed_asset


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def atomic(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n'); temp.replace(path)
def save_torch(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temp); temp.replace(path)


def main():
    if args.generations != 1: raise ValueError('One declared fixed grid only')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'INPUTS.json').exists(): raise ValueError('Require a new native output')
    protocol_path = ROOT / 'configs/g07_full_racket_eval_v4_r3.json'
    protocol = json.loads(protocol_path.read_text()); protocol['_root'] = str(ROOT)
    assert sha(ROOT / protocol['asset']['path']) == protocol['asset']['sha256']
    assert all(sha(ROOT / name) == value for name, value in protocol['physical_configs'].items())
    cases, indices = training_cases(ROOT, protocol)
    assert indices == list(range(128)) and len(cases) == 128
    body = torch.load(args.source, map_location='cpu', weights_only=False)
    comparison = torch.load(args.comparison, map_location='cpu', weights_only=False)
    assert body['schema'] == comparison['schema'] == SCHEMA
    assert body['native_execution'] and comparison['native_execution']
    assert body['asset_sha256'] == comparison['asset_sha256'] == protocol['asset']['sha256']
    assert same_tree(body['source_public_policy'], comparison['source_public_policy'])
    source = body['source_public_policy']
    reference = body['parameters'].clone()
    population, proposal_labels = population_and_labels(reference, comparison['parameters'])
    count, case_count = CANDIDATE_COUNT * 128, 128
    cfg = full_racket_cfg(count, protocol['physics_dt_s']); cfg.seed = args.seed
    cfg.episode_length_s = protocol['episode_seconds']
    assert abs(cfg.sim.dt * cfg.decimation - protocol['policy_dt_s']) < 1e-12
    env = RallyTrainingEnvV4(cfg, cases=cases * CANDIDATE_COUNT, continue_after_miss=True)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    try:
        assert not env.robot.is_fixed_base
        controller = FixedActorBodyControllerV4(source, reference, count, env.device)
        atomic(args.output / 'asset_dependencies.json', audit_composed_asset(ROOT, cfg.robot.spawn.usd_path))
        export_robot_geometry(get_current_stage(), env.robot, args.output)
        default = env.robot.data.default_joint_pos[0, env.joint_ids]
        limits = env.robot.data.joint_pos_limits[0, env.joint_ids]
        ticks = round(protocol['episode_seconds'] / protocol['policy_dt_s'])
        reference = reference.to(env.device); population = population.to(env.device)
        stream_ids = torch.tensor([128*c+i for c in range(CANDIDATE_COUNT) for i in range(2)], device=env.device)
        sources = sorted(ROOT.glob('quadsmash/**/*.py')) + [Path(__file__)]
        atomic(args.output / 'INPUTS.json', dict(source_policy_sha256=sha(args.source),
            protocol_sha256=sha(protocol_path), source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources},
            source_actor_weights_frozen=True, both_original_residual_actors_preserved=True,
            training_suite=protocol['training_suite'], training_indices=indices,
            candidate_count=CANDIDATE_COUNT, num_envs=count,
            warm_start_body_policy_sha256=sha(args.source), comparison_body_policy_sha256=sha(args.comparison),
            nested_source_policy_sha256=body['source_policy_sha256'], proposal_labels=proposal_labels, requested_generations=args.generations, seed=args.seed,
            initial_population=population.cpu().tolist(), search_coordinates=COORDINATES,
            selection='Numerical validity, actual complete count, actual attack/cross/target count, declared partial TRAIN score; ties preserve earlier candidate.',
            reset_noise='none', controller_inputs='Original public794 only; parameters fixed across all128 cases within each candidate.',
            episode_identity='Actual generation-start native episode IDs retained; never renumbered to episode0.',
            formal_cases_read=False, formal_acceptance=False))
        history = []; best_rank = None; best = None
        for generation in range(args.generations):
            # Ordinary environment reset; all initial physical states stay on
            # the ground. Do not alter native episode IDs to simplify labels.
            previous_rallies, previous_jumps = len(env.rally_records), len(env.episode_records)
            observation, _ = wrapped.reset(); controller.reset()
            atomic(args.output / f'reset_transition_rallies_{generation:03d}.json',
                env.rally_records[previous_rallies:])
            atomic(args.output / f'reset_transition_jumps_{generation:03d}.json',
                env.episode_records[previous_jumps:])
            controller.set_parameters(population.repeat_interleave(case_count, dim=0))
            start_ids = env.episode_index.detach().clone()
            state = env.robot.data
            initial = dict(episode_ids=start_ids.cpu().tolist(),
                root_state_local=torch.cat((state.root_pos_w-env.scene.env_origins,
                    state.root_quat_w, state.root_lin_vel_w, state.root_ang_vel_w), -1).cpu().tolist(),
                joint_position_rad=state.joint_pos[:, env.joint_ids].cpu().tolist(),
                joint_velocity_rad_s=state.joint_vel[:, env.joint_ids].cpu().tolist(),
                all_clocks_zero=bool((observation['policy'][:,775] == 0).all()))
            assert initial['all_clocks_zero'] and (start_ids >= 0).all()
            expected_root = torch.tensor(protocol['initialization']['root_position_m'],device=env.device)
            assert ((state.root_pos_w-env.scene.env_origins)-expected_root).abs().max() < 1e-5
            assert (state.root_quat_w-torch.tensor([1.,0.,0.,0.],device=env.device)).abs().max() < 1e-5
            assert state.root_lin_vel_w.abs().max() < 1e-5 and state.root_ang_vel_w.abs().max() < 1e-5
            assert state.joint_vel[:,env.joint_ids].abs().max() < 1e-5
            atomic(args.output / f'initial_ground_states_{generation:03d}.json', initial)
            save_torch(args.output / f'generation_start_{generation:03d}.pt', dict(
                population=population.cpu(), proposal_labels=proposal_labels,
                native_episode_ids=start_ids.cpu(), new_generation_restarts_from_ground=True,
                full_native_restart=False, actual_completed_generations=generation))
            finished = torch.zeros(count, dtype=torch.bool, device=env.device)
            closest = torch.full((count,), 10., device=env.device)
            poses, balls, clocks, valid, public, actions, public_valid = [], [], [], [], [], [], []
            rally_start, jump_start, contact_start = len(env.rally_records), len(env.episode_records), len(env.contact_records)
            with torch.inference_mode():
                for tick in range(ticks):
                    obs = observation['policy']; data = env.robot.data
                    active = (~finished) & env.launch_done & env.rally.in_play
                    gap = string_prism_gap(env.shuttle.data.root_pos_w,
                        data.body_link_pos_w[:,env.racket_body_id],data.body_link_quat_w[:,env.racket_body_id],
                        env.racket_config,cfg.shuttle_radius_m).clamp_min(0.)
                    closest = torch.where(active, torch.minimum(closest, gap), closest)
                    action = controller.action(observation, default, limits)
                    assert action.shape == (count,18) and torch.isfinite(action).all() and (action.abs() <= 1).all()
                    public.append(obs[stream_ids].cpu().numpy().copy())
                    actions.append(action[stream_ids].cpu().numpy().copy())
                    public_valid.append((~finished[stream_ids]).cpu().numpy().copy())
                    if tick % 2 == 0:
                        poses.append(torch.cat((data.body_link_pos_w-env.scene.env_origins[:,None],data.body_link_quat_w),-1).cpu().numpy())
                        balls.append((env.shuttle.data.root_pos_w-env.scene.env_origins).cpu().numpy())
                        clocks.append(obs[:,775].cpu().numpy().copy()); valid.append((~finished).cpu().numpy().copy())
                    observation, _, done, _ = wrapped.step(action)
                    finished |= done.bool()
                    # Ended lanes are irrelevant to this generation's first
                    # episodes, but their automatic resets remain ordinary.
                    ids = torch.where(done.bool())[0]
                    if len(ids): controller.reset(ids)
                    if (tick+1) % 100 == 0 or stopping or bool(finished.all()):
                        atomic(args.output / 'progress.json',dict(generation=generation,control_ticks=tick+1,
                            first_episodes_complete=int(finished.sum()), requested_episodes=count,
                            actual_control_env_steps=(generation*ticks+tick+1)*count,formal_acceptance=False))
                        atomic(args.output / 'rallies.partial.json',env.rally_records[rally_start:])
                        print(json.dumps(dict(generation=generation,tick=tick+1,first_complete=int(finished.sum()))),flush=True)
                    if stopping or bool(finished.all()): break
            suffix = f'{generation:03d}'
            np.savez_compressed(args.output / f'all_first_episodes_{suffix}.npz',link_poses=np.stack(poses),
                ball_position_m=np.stack(balls),episode_clock_s=np.stack(clocks),valid_first_episode=np.stack(valid))
            np.savez_compressed(args.output / f'public_and_actions_{suffix}.npz',observations=np.stack(public),
                actions=np.stack(actions),valid_first_episode=np.stack(public_valid),native_env_ids=stream_ids.cpu().numpy())
            atomic(args.output / f'rallies_{suffix}.json',env.rally_records[rally_start:])
            atomic(args.output / f'jumps_{suffix}.json',env.episode_records[jump_start:])
            atomic(args.output / f'contacts_{suffix}.json',env.contact_records[contact_start:])
            atomic(args.output / f'causal_plans_{suffix}.json',controller.logs)
            if stopping and not bool(finished.all()):
                atomic(args.output / 'interrupted.json',dict(generation=generation,control_ticks=tick+1,
                    replay_from=f'generation_start_{suffix}.pt',actual_partial_env_steps=(tick+1)*count,
                    full_native_resume_claimed=False,formal_acceptance=False)); return
            # The existing jump logger has no episode_index field. Its
            # synchronous reset order matches the rally logger. Keep both
            # original streams and verify/record their ordinal association.
            rally_raw, jump_raw = env.rally_records[rally_start:], env.episode_records[jump_start:]
            assert len(rally_raw) == len(jump_raw)
            first, jumps, links = {}, {}, []
            for ordinal, (row, jump) in enumerate(zip(rally_raw,jump_raw)):
                lane = row['env_id']; assert jump['env_id'] == lane
                links.append(dict(raw_record_ordinal=ordinal,env_id=lane,
                    actual_native_episode_index=row['episode_index'],
                    jump_linkage='Synchronous native reset record order; jump logger itself has no episode ID.'))
                if row['episode_index'] == int(start_ids[lane]):
                    assert lane not in first
                    first[lane], jumps[lane] = row, jump
            atomic(args.output / f'raw_episode_links_{suffix}.json',links)
            assert set(first) == set(jumps) == set(range(count))
            rows = [dict(first[i],jump=jumps[i],candidate_id=i//case_count,
                case_id=cases[i%case_count]['case_id'],source_training_index=i%case_count,ground_init=True,
                closest_string_gap_m=float(closest[i])) for i in range(count)]
            summaries = [summarize_candidate(rows[i*case_count:(i+1)*case_count],protocol) for i in range(CANDIDATE_COUNT)]
            ranked = sorted(range(CANDIDATE_COUNT),key=lambda i:tuple(summaries[i]['rank']),reverse=True)
            assert summaries[ranked[0]]['numerical_failures'] == 0
            top = ranked[0]
            if best_rank is None or tuple(summaries[top]['rank']) > best_rank:
                best_rank = tuple(summaries[top]['rank'])
                best = dict(schema=SCHEMA,source_public_policy=source,source_policy_sha256=body['source_policy_sha256'],
                    warm_start_body_policy_sha256=sha(args.source),comparison_body_policy_sha256=sha(args.comparison),
                    parameters=population[top].cpu().clone(),asset_sha256=source['asset_sha256'],
                    native_execution=True,performance_inherited=False,formal_acceptance=False,
                    actual_native_search=dict(generation=generation,candidate_id=top,summary=summaries[top],
                        episode_indices=start_ids[top*case_count:(top+1)*case_count].cpu().tolist(),
                        training_indices=indices,seed=args.seed),kind='Fixed learned actors plus remeasured body strike-grid parameters')
            atomic(args.output / f'episodes_{suffix}.json', rows)
            entry = dict(generation=generation,control_ticks=tick+1,env_steps=(tick+1)*count,
                summaries=summaries,ranked_candidate_indices=ranked,best_this_generation=top)
            history.append(entry)
            save_torch(args.output / f'population_{suffix}.pt',dict(parameters=population.cpu(),summaries=summaries,
                actual_episode_ids=start_ids.cpu(),ranked_candidate_indices=ranked))
            save_torch(args.output / 'best_candidate.pt', best)
            atomic(args.output / 'metrics.json',dict(history=history,best_rank=best_rank,
                actual_control_env_steps=sum(x['env_steps'] for x in history),formal_acceptance=False,formal_cases_read=False))
            print(json.dumps(entry),flush=True)
            if stopping: return
        atomic(args.output / 'completed.json',dict(env_steps=sum(x['env_steps'] for x in history)))
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
