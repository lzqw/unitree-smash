"""Freeze the actual completed cold TRAIN128 and all four airborne contacts."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
MAIN = ROOT.parent / 'unitree_smash'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(MAIN))
import numpy as np
import torch
from quadsmash.control.full_racket_prepose_branch_v4 import make_rally_controller
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases, summarize_training
from quadsmash.evaluation_v4 import independent_episode


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert Path.cwd() == MAIN and not (HERE / 'PLAN.json').exists()
    reports = MAIN / 'reports/G07_full_racket_v4'
    run = reports / 'prepose_origin0_8_best_fixed_train128_01'
    search = reports / 'prepose_origin0_8_sequential_n128_01'
    parent = reports / 'prepose_origin0_after_precision_queue02'
    held = read(reports / 'origin0_residual_after_gain_body_queue01/PLAN.json')['source_sha256']
    assert all(sha(Path(p)) == s for p, s in held.items())
    assert not Path('/proc/1884350').exists()
    assert read(parent / 'state.json')['state'] == 'COMPLETED_PREPOSE_ORIGIN0_SEARCH_AND_COLD128'
    policy = search / 'best_candidate.pt'
    protocol_path = MAIN / 'configs/g07_full_racket_eval_v4_r3_origin0.json'
    protocol = read(protocol_path)
    protocol['_root'] = str(MAIN)
    cases, indices = training_cases(MAIN, protocol)
    rows, inputs = read(run / 'episodes.json'), read(run / 'INPUTS.json')
    summary = summarize_training(rows, cases, indices, protocol)
    assert summary == read(parent / 'COLD_TRAIN128_RESULT.json')
    assert all(read(run / 'metrics.json')[k] == v for k, v in summary.items())
    assert summary['episodes'] == 128 and summary['complete'] == summary['conditional_recovery_count'] == 0
    assert summary['airborne_contacts'] == 4 and summary['numerical_failures'] == 0
    assert sha(policy) == inputs['policy_sha256'] == '8a3aab4ab96fdbcdad7d9d33fc2f6de057f6178cb8fc6c8508b7dcb90fc5fc15'
    assert sha(protocol_path) == inputs['protocol_sha256']
    assert inputs['origin_spacing_m'] == 0 and inputs['training_indices'] == list(range(128))
    assert inputs['reset_noise_seed'] == 724201 and not any(inputs['reset_noise_amplitudes'].values())
    assert not np.asarray(read(run / 'origin_layout.json')['actual_environment_origins_m']).any()
    initial = read(run / 'initial_states.json')
    assert len(initial) == 128
    for i, state in enumerate(initial):
        assert state['env_id'] == i
        assert np.max(abs(np.asarray(state['root_state_local'][:3]) - protocol['initialization']['root_position_m'])) < 1e-5
    job = read(run / 'job.json')
    assert job['status'] == 'SUCCEEDED' and job['exit_code'] == 0 and not job['timed_out']
    assert read(run / 'completed.json')['env_steps'] == job['charged_env_steps'] == 102400
    selected = [i for i, row in enumerate(rows) if independent_episode(row, protocol)['airborne_contact']]
    assert selected == [69, 100, 112, 123]
    selected.append(0)  # Explicit miss control, in addition to every actual airborne contact.
    assert not rows[0]['had_contact']
    source_files = [p for p in sorted(run.iterdir()) if p.is_file() and '.partial.' not in p.name]
    source_files += [policy, protocol_path, MAIN / 'configs/court_net_v4.json',
                     MAIN / protocol['training_suite']['path'], search / 'INPUTS.json',
                     parent / 'COLD_TRAIN128_RESULT.json', parent / 'SEARCH_RECEIPT.json', Path(__file__)]
    source_files += [MAIN / name for name in read(run / 'source_hashes.json')]
    assert sha(ROOT / 'configs/court_net_v4.json') == sha(MAIN / 'configs/court_net_v4.json')
    plan = dict(created_at=datetime.now(timezone.utc).isoformat(), selected_TRAIN_indices=selected,
        selection='Every actual airborne string contact in the completed frozen cold128, followed by original TRAIN0 miss control. Posthoc selection; not a random sample.',
        source_sha256={str(p): sha(p) for p in source_files}, summary=summary,
        previous_goal_turn='no_progress_for_training_goal: answered the user video-location question; this turn verifies the completed cold result and actual successor, then creates current evidence.',
        scope='Completed independent cold TRAIN128; no old search scores inherited and no formal A/B cases read.',
        formal_cases_read=False, formal_acceptance=False)
    (HERE / 'PLAN.json').write_text(json.dumps(plan, indent=2) + '\n')
    snapshot = HERE / 'snapshot'
    snapshot.mkdir()
    native = snapshot / run.name
    native.mkdir()
    for path in source_files:
        if path.parent == run:
            shutil.copy2(path, native / path.name)
    shutil.copy2(policy, HERE / 'displayed_frozen_policy.pt')
    checkpoint = torch.load(policy, map_location='cpu', weights_only=False)
    native_search_inputs = read(search / 'INPUTS.json')
    default, limits = [torch.tensor(native_search_inputs[key], dtype=torch.float32) for key in
                       ['default_joint_position_rad', 'joint_limits_rad']]
    controller = make_rally_controller(checkpoint, 2, 'cpu')
    with np.load(run / 'public_and_actions_first2.npz') as arrays:
        public, actions, valid = [arrays[k].copy() for k in ['observations', 'actions', 'valid_first_episode']]
    maximum, previous = 0., None
    with torch.inference_mode():
        for obs, actual, keep in zip(public, actions, valid):
            observation = torch.from_numpy(obs)
            if previous is not None:
                reset = torch.where(observation[:, 775] < previous - 1e-5)[0]
                if len(reset):
                    controller.reset(reset)
            command = controller.action({'policy': observation}, default, limits).numpy()
            if keep.any():
                maximum = max(maximum, float(abs(command[keep] - actual[keep]).max()))
            previous = observation[:, 775].clone()
    assert valid.sum() == 1600 and maximum <= 5e-4
    events = read(run / 'contacts.json')
    film_cases, diagnoses = [], []
    with np.load(run / 'all_first_episodes.npz') as arrays:
        for case_id, lane in enumerate(selected):
            row = rows[lane]
            ids = np.flatnonzero(arrays['valid_first_episode'][:, lane])
            assert np.array_equal(ids, np.arange(len(ids))) and len(ids) == 400
            clock = arrays['episode_clock_s'][ids, lane]
            assert abs(clock[0]) < 1e-6 and np.max(abs(np.diff(clock) - .02)) < 2e-6
            poses = arrays['link_poses'][ids, lane]
            balls = arrays['ball_position_m'][ids, lane]
            np.savez_compressed(snapshot / f'case{case_id}.npz', link_poses=poses,
                ball_position_m=balls, episode_clock_s=clock, valid_first_episode=np.ones(len(ids), bool),
                native_episode_index=np.full(len(ids), row['episode_index']))
            contacts = [e for e in events if e['env_id'] == lane and e['episode_index'] == row['episode_index']]
            contact = contacts[0] if contacts else None
            if lane != 0:
                assert len(contacts) == 1 and contact['first_racket_contact'] and contact['string_contact']
                assert contact['time_s'] == row['contact_time_s'] and row['jump']['body_landing']
                assert not row['jump']['stable_recovery']
                speed = float(np.linalg.norm(row['outgoing_velocity_mps']))
                assert speed < 8
                title = '空中拍线触球｜球被网挡，身体触地' if lane == 100 else (
                    '空中拍线触球｜命中目标，身体触地' if row['landed_target'] else
                    '空中拍线触球｜场内偏出目标，身体触地')
                status = f'出球{speed:.2f} m/s低于8；身体触地，完整任务失败'
                limitation = '独立冷128：4空中、0完整、0合格恢复；不计作成功扣杀'
            else:
                speed = None
                title, status = '未触球对照｜保留原TRAIN0', '没有实际球拍接触；完整任务失败'
                limitation = '不只展示四次触球；同一策略原128中其余124例未获合格空中触球'
            last = dict(time_s=float(clock[-1]), body_link_positions_m=poses[-1, :, :3].tolist(),
                body_link_quaternions_wxyz=poses[-1, :, 3:].tolist(), ball_position_m=balls[-1].tolist())
            film_cases.append(dict(id=case_id, run=run.name, env_id=lane, episode_index=row['episode_index'],
                source_training_index=lane, row=row, contact=contact, title=title,
                subtitle=f'TRAIN{lane}｜独立冻结冷评估｜完整地面起始回合',
                terminal_status=status, limitation=limitation, last_recorded_state=last,
                selected_source_frames=ids.tolist()))
            diagnoses.append(dict(TRAIN_index=lane, original_episode_index=row['episode_index'],
                actual_airborne_string_contact=bool(row['airborne_string_hit']), outgoing_speed_mps=speed,
                crossed_net=row['crossed_net'], landed_legal_court=row['landed_legal_court'],
                landed_target=row['landed_target'], net_blocked_fault=row['net_blocked_fault'],
                body_landing=row['jump']['body_landing'], stable_recovery=row['jump']['stable_recovery'],
                qualified_jump_landed=row['jump']['qualified_jump_landed'], complete=False))
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), cases=film_cases,
        opening_title='最新冷评估｜四次空中拍线触球',
        opening_lines=['固定同一策略、原128来球：4空中、0完整、0合格恢复。',
            '播放全部四次空中触球，再保留一个未触球对照。',
            '完整地面起始回合、侧视与俯视、真实接触及首次落点。'],
        closing_title='已找到空中接触，尚未完成扣杀恢复',
        closing_lines=['四次出球均低于原8m/s门槛，四例均身体触地。',
            '两例首次落在原目标区；另有一例被网挡、一例场内偏出目标。',
            '继续优化全身起跳与恢复；原球网、球拍、目标和门槛保持。'],
        evaluation_footer='独立固定TRAIN128：4空中 / 0完整 / 0合格恢复 / 数值故障0｜非正式A/B',
        original_source_hashes=plan['source_sha256'], completed_cold_summary=summary,
        caveats=['Posthoc selected all four cold airborne contacts plus one miss control, not a random sample.',
            'All128 cold rows are retained; no search or old-origin16 performance inherited.',
            'Motion is actual50Hz Native data, exact contact is actual substep data. No interpolation or new physics.',
            'Final freeze is explicitly labelled the last actual50Hz sample, not an exact terminal substep.',
            'Four cold contacts all have body landing; the search TRAIN100 stable flag does not transfer to this independent cold episode.'],
        new_native_execution=False, formal_cases_read=False, formal_acceptance=False)
    (snapshot / 'cases.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    assert all(sha(Path(p)) == s for p, s in held.items())
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), cold_summary=summary,
        all128_original_rows_independently_checked=True, actual_ground_initial_states_checked=128,
        selected_TRAIN_indices=selected, selected_original_episode_indices=[0] * 5,
        actual_valid_public_rows=int(valid.sum()), CPU_vs_actual_GPU_max_action_error=maximum,
        original_action_tolerance=5e-4, selected_episode_diagnoses=diagnoses,
        source_policy_sha256=sha(policy), held482_source_hashes_unchanged=True,
        new_native_execution=False, formal_cases_read=False, formal_acceptance=False)
    (HERE / 'NATIVE_EPISODE_CHECK.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(selected=selected, max_action_error=maximum, summary=summary)), flush=True)


if __name__ == '__main__':
    main()
