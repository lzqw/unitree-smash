"""Prepare actual frozen residual15 cold episodes; require completed evidence first."""
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
from quadsmash.control.full_racket_origin0_residual_v4 import make_rally_controller
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases, summarize_training
from quadsmash.evaluation_v4 import independent_episode


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert Path.cwd() == MAIN and not (HERE / 'PLAN.json').exists()
    reports = MAIN / 'reports/G07_full_racket_v4'
    run = reports / 'origin0_residual_fresh15_fixed_train128_01'
    parent = reports / 'origin0_residual_after_gain_body_queue01'
    policy = reports / 'origin0_residual_fresh_n128_updates15_01/policy.pt'
    assert not Path('/proc/1896271').exists()
    assert read(parent / 'state.json')['state'] == 'COMPLETED_ORIGIN0_RESIDUAL15_AND_COLD128'
    held = read(reports / 'recovery_sequence_observer_20261027/QUEUE_PLAN.json')['source_sha256']
    assert all(sha(Path(p)) == s for p, s in held.items())
    protocol_path = MAIN / 'configs/g07_full_racket_eval_v4_r3_origin0.json'
    protocol = read(protocol_path)
    protocol['_root'] = str(MAIN)
    cases, indices = training_cases(MAIN, protocol)
    rows, inputs = read(run / 'episodes.json'), read(run / 'INPUTS.json')
    summary = summarize_training(rows, cases, indices, protocol)
    assert summary == read(parent / 'COLD_TRAIN128_RESULT.json')
    assert all(read(run / 'metrics.json')[k] == v for k, v in summary.items())
    assert summary['episodes'] == 128 and summary['numerical_failures'] == 0
    assert inputs['policy_sha256'] == sha(policy) == '21b7670cabbe72587cdb5b8270f6add4159733282f07bc93ee0217c0bb1b0b8c'
    assert inputs['protocol_sha256'] == sha(protocol_path) and inputs['training_indices'] == list(range(128))
    assert inputs['reset_noise_seed'] == 724201 and not any(inputs['reset_noise_amplitudes'].values())
    assert not np.asarray(read(run / 'origin_layout.json')['actual_environment_origins_m']).any()
    initial = read(run / 'initial_states.json')
    assert len(initial) == 128
    for lane, state in enumerate(initial):
        assert state['env_id'] == lane
        assert np.max(abs(np.asarray(state['root_state_local'][:3]) - protocol['initialization']['root_position_m'])) < 1e-5
    job = read(run / 'job.json')
    assert job['status'] == 'SUCCEEDED' and job['exit_code'] == 0 and not job['timed_out']
    assert read(run / 'completed.json')['env_steps'] <= job['charged_env_steps'] == 102400
    checked = [independent_episode(row, protocol) for row in rows]
    air = [i for i, value in enumerate(checked) if value['airborne_contact']]
    racket_contacts = [i for i, row in enumerate(rows) if row['had_contact']]
    selected = list(dict.fromkeys(air[:8] + racket_contacts[:8] + [69, 112, 0]))
    failures = [i for i, value in enumerate(checked) if not value['complete']]
    if failures and all(checked[i]['complete'] for i in selected):
        selected.append(failures[0])
    source_files = [p for p in sorted(run.iterdir()) if p.is_file() and '.partial.' not in p.name]
    source_files += [policy, protocol_path, MAIN / 'configs/court_net_v4.json',
                     MAIN / protocol['training_suite']['path'], parent / 'COLD_TRAIN128_RESULT.json',
                     parent / 'COLD_CPU_COMMAND_CHECK.json', Path(__file__)]
    source_files += [MAIN / name for name in read(run / 'source_hashes.json')]
    assert sha(ROOT / 'configs/court_net_v4.json') == sha(MAIN / 'configs/court_net_v4.json')
    hardware = read(reports / 'gain_body8_sequential_n128_01/INPUTS.json')
    default, limits = [torch.tensor(hardware[k], dtype=torch.float32) for k in
                       ['default_joint_position_rad', 'joint_limits_rad']]
    assert np.max(abs(np.asarray(initial[0]['joint_position_rad']) - default.numpy())) < 1e-5
    controller = make_rally_controller(torch.load(policy, map_location='cpu', weights_only=False), 2, 'cpu')
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
    assert maximum <= 5e-4 and valid.sum() > 0
    source_files.append(reports / 'gain_body8_sequential_n128_01/INPUTS.json')
    plan = dict(created_at=datetime.now(timezone.utc).isoformat(), selected_TRAIN_indices=selected,
        fixed_reference_indices_declared_before_completed_result=[69, 112, 0],
        selection='Up to first8 actual airborne and first8 actual racket contacts in original TRAIN order, plus fixed69/112/0 references and first failure if otherwise missing; posthoc contact selection, not a random sample. All128 metrics retained.',
        all_actual_airborne_indices=air, summary=summary,
        source_sha256={str(p): sha(p) for p in source_files}, new_native_execution=False,
        formal_cases_read=False, formal_acceptance=False)
    (HERE / 'PLAN.json').write_text(json.dumps(plan, indent=2) + '\n')
    snapshot = HERE / 'snapshot'
    native = snapshot / run.name
    native.mkdir(parents=True)
    for path in source_files:
        if path.parent == run:
            shutil.copy2(path, native / path.name)
    shutil.copy2(policy, HERE / 'displayed_frozen_policy.pt')
    events = read(run / 'contacts.json')
    film_cases, diagnoses = [], []
    with np.load(run / 'all_first_episodes.npz') as arrays:
        for case_id, lane in enumerate(selected):
            row, value = rows[lane], checked[lane]
            ids = np.flatnonzero(arrays['valid_first_episode'][:, lane])
            assert len(ids) > 0 and np.array_equal(ids, np.arange(len(ids)))
            clock = arrays['episode_clock_s'][ids, lane]
            assert abs(clock[0]) < 1e-6
            if len(ids) > 1:
                assert np.max(abs(np.diff(clock) - .02)) < 2e-6
            poses, balls = arrays['link_poses'][ids, lane], arrays['ball_position_m'][ids, lane]
            assert np.isfinite(poses).all() and np.isfinite(balls).all()
            np.savez_compressed(snapshot / f'case{case_id}.npz', link_poses=poses, ball_position_m=balls,
                episode_clock_s=clock, valid_first_episode=np.ones(len(ids), bool),
                native_episode_index=np.full(len(ids), row['episode_index']))
            matches = [e for e in events if e['env_id'] == lane and e['episode_index'] == row['episode_index']
                       and e['time_s'] == row['contact_time_s']]
            contact = matches[0] if matches else None
            assert not row['had_contact'] or contact is not None
            speed = float(np.linalg.norm(row['outgoing_velocity_mps'])) if row['had_contact'] else None
            title = ('完整成功' if value['complete'] else '空中拍线接触，完整任务未通过') if value['airborne_contact'] else (
                '拍框接触，未获空中拍线资格' if row['frame_hit'] else
                '拍线接触，未获合格空中资格' if row['had_contact'] else '未触球对照')
            outcomes = []
            if speed is not None:
                outcomes.append(f'出球{speed:.2f}m/s')
                if speed < 8.:
                    outcomes.append('低于原8m/s门槛')
            if row['jump']['body_landing']:
                outcomes.append('身体触地')
            elif value['airborne_contact'] and not value['conditional_recovery']:
                outcomes.append('恢复未通过原判定')
            if value['fault']:
                outcomes.append('有规则违规')
            if row['ground_seen']:
                outcomes.append('首落原目标' if row['landed_target'] else '未命中原目标')
            elif row['had_contact']:
                outcomes.append('首次落点未记录')
            last = dict(time_s=float(clock[-1]), body_link_positions_m=poses[-1, :, :3].tolist(),
                body_link_quaternions_wxyz=poses[-1, :, 3:].tolist(), ball_position_m=balls[-1].tolist())
            film_cases.append(dict(id=case_id, run=run.name, env_id=lane, episode_index=row['episode_index'],
                source_training_index=lane, row=row, contact=contact, title=title,
                subtitle=f'TRAIN{lane}｜新残差15｜本轮独立冻结评估',
                terminal_status='；'.join(outcomes) or '本回合无合格触球',
                limitation='实际末次50Hz采样；任务结论使用本轮全部128例原判定',
                last_recorded_state=last, selected_source_frames=ids.tolist()))
            diagnoses.append(dict(TRAIN_index=lane, actual_airborne_contact=value['airborne_contact'],
                qualified_recovery=value['conditional_recovery'], complete=value['complete'],
                outgoing_speed_mps=speed, body_landing=row['jump']['body_landing'],
                last_recorded_time_s=float(clock[-1]), selected_samples=len(ids)))
    stats = f"{summary['airborne_contacts']}空中 / {summary['complete']}完整 / {summary['conditional_recovery_count']}合格恢复 / NF{summary['numerical_failures']}"
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), cases=film_cases,
        opening_title='最新效果｜残差15独立冷评估',
        opening_lines=[f'同一冻结策略、原128来球：{stats}。',
            '播放实际球拍接触及固定TRAIN69、112、0对照。', '完整地面起始回合，侧视与俯视，真实接触和首次落点。'],
        closing_title='以独立冻结评估衡量当前效果',
        closing_lines=[f'原128实测：{stats}。', '训练中的7次空中触球不移用为本轮冻结成绩。',
            '原场地、完整球拍、目标区和50/70/85门槛保持。'],
        evaluation_footer=f'残差15独立TRAIN128：{stats}｜非正式A/B',
        original_source_hashes=plan['source_sha256'], completed_cold_summary=summary,
        caveats=[plan['selection'], 'Actual50Hz Native poses/ball and exact original contact events; no interpolation or new physics.',
            'Episode durations may end early under original rule/root termination; all recorded first-episode frames are retained.',
            'Final freeze is labelled the last actual50Hz sample, not an exact terminal physics state.',
            'Original recovery scoring retained; the earlier stochastic TRAIN69 flag observation is not transplanted here.'],
        new_native_execution=False, formal_cases_read=False, formal_acceptance=False)
    (snapshot / 'cases.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    assert all(sha(Path(p)) == s for p, s in held.items())
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), cold_summary=summary,
        all128_original_rows_independently_checked=True, selected_TRAIN_indices=selected,
        actual_valid_public_rows=int(valid.sum()), CPU_vs_actual_GPU_max_action_error=maximum,
        original_action_tolerance=5e-4, selected_episode_diagnoses=diagnoses,
        held529_source_hashes_unchanged=True, new_native_execution=False, formal_acceptance=False)
    (HERE / 'NATIVE_EPISODE_CHECK.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(selected=selected, max_action_error=maximum, cold_summary=summary)), flush=True)


if __name__ == '__main__':
    main()
