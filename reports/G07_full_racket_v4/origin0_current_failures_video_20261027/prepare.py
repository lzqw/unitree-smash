"""Freeze actual origin0 episodes: frame/fall, recovered miss and net fault."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
MAIN = ROOT.parent/'unitree_smash'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(MAIN))
import numpy as np
import torch
from quadsmash.control.full_racket_body_search_v4 import summarize_candidate
from quadsmash.control.full_racket_prepose_branch_v4 import PreposeBranchControllerV4, SCHEMA
from quadsmash.evaluation_v4 import independent_episode


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert Path.cwd() == MAIN and not (HERE/'PLAN.json').exists()
    run = MAIN/'reports/G07_full_racket_v4/prepose_origin0_8_sequential_n128_01'
    owner = MAIN/'reports/G07_full_racket_v4/gain_body_after_prepose_queue01/PLAN.json'
    held = read(owner)['source_sha256']
    assert all(sha(Path(path)) == digest for path, digest in held.items())
    protocol_path = MAIN/'configs/g07_full_racket_eval_v4_r3_origin0.json'
    protocol = read(protocol_path); protocol['_root'] = str(MAIN)
    history, inputs = read(run/'metrics.json')['history'], read(run/'INPUTS.json')
    source_path = MAIN/'reports/G07_full_racket_v4/contact_aim7_sequential_n128_01/best_candidate.pt'
    assert sha(source_path) == inputs['source_policy_sha256']
    assert sha(ROOT/'configs/court_net_v4.json') == sha(MAIN/'configs/court_net_v4.json')
    entries, summaries, all_rows, initial = {}, {}, {}, {}
    sources = [source_path, protocol_path, MAIN/'configs/court_net_v4.json', Path(__file__)]
    for candidate in [0, 1]:
        suffix = f'{candidate:03d}'
        entry = next(row for row in history if row['candidate_id'] == candidate)
        rows = read(run/f'episodes_{suffix}.json')
        summary = summarize_candidate(rows, protocol)
        assert summary == entry['summary'] and summary['complete'] == summary['airborne_contacts'] == summary['numerical_failures'] == 0
        entries[candidate], summaries[candidate], all_rows[candidate] = entry, summary, rows
        initial[candidate] = read(run/f'initial_ground_states_{suffix}.json')
        assert len(rows) == 128 and not np.asarray(initial[candidate]['actual_environment_origins_m']).any()
        for lane, row in enumerate(rows):
            assert row['env_id'] == row['jump']['env_id'] == lane and row['candidate_id'] == candidate
            assert row['episode_index'] == initial[candidate]['episode_ids'][lane] and row['ground_init']
            assert row['source_training_index'] == lane
        sources.extend(run/f'{name}_{suffix}.{extension}' for name, extension in
            [('episodes', 'json'), ('contacts', 'json'), ('all_first_episodes', 'npz'),
             ('public_and_actions', 'npz'), ('initial_ground_states', 'json'), ('causal_plans', 'json')])
    sources.extend(run/name for name in ['geometry.json', 'geometry.npz', 'INPUTS.json',
        'source_snapshot.zip', 'source_hashes.json', 'dirty.patch', 'asset_dependencies.json'])
    selected = [(0, 19), (0, 72), (1, 1)]
    (HERE/'PLAN.json').write_text(json.dumps(dict(created_at=datetime.now(timezone.utc).isoformat(),
        selected_candidates_and_TRAIN_lanes=selected,
        selection='Posthoc origin0 reference frame/fall TRAIN19, stable-recovery miss TRAIN72, and first net-fault TRAIN1 of measured-pose candidate1. Three independent actual episodes.',
        completed_candidate_entries=entries, source_sha256={str(path):sha(path) for path in sources},
        scope='Two completed candidate TRAIN128 samples from the still-running eight-candidate job; no final cold score or new physical run.',
        formal_cases_read=False, formal_acceptance=False), indent=2)+'\n')
    snapshot = HERE/'snapshot'; snapshot.mkdir()
    native = snapshot/run.name; native.mkdir()
    for path in sources:
        if path.parent == run:
            shutil.copy2(path, native/path.name)
    (HERE/'ACTUAL_PRODUCER_SNAPSHOT.json').write_text(json.dumps(dict(
        metrics=read(run/'metrics.json'), progress=read(run/'progress.json'),
        job=read(run/'job.json') if (run/'job.json').exists() else dict(receipt_pending_until_guard_exit=True),
        whole_course_finished=False, cold_confirmation_finished=False), indent=2)+'\n')
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    default, limits = [torch.tensor(inputs[key], dtype=torch.float32) for key in
                       ['default_joint_position_rad', 'joint_limits_rad']]
    replay_receipts = []
    for candidate in [0, 1]:
        with np.load(run/f'public_and_actions_{candidate:03d}.npz') as arrays:
            public, actions, valid = [arrays[key].copy() for key in ['observations', 'actions', 'valid_first_episode']]
        controller = PreposeBranchControllerV4(source, entries[candidate]['candidate']['overrides'], 2, 'cpu')
        maximum, previous = 0., None
        with torch.inference_mode():
            for obs, actual, keep in zip(public, actions, valid):
                observation = torch.from_numpy(obs)
                if previous is not None:
                    reset = torch.where(observation[:, 775] < previous-1e-5)[0]
                    if len(reset):
                        controller.reset(reset)
                command = controller.action({'policy': observation}, default, limits).numpy()
                if keep.any():
                    maximum = max(maximum, float(abs(command[keep]-actual[keep]).max()))
                previous = observation[:, 775].clone()
        assert maximum <= 5e-4
        replay_receipts.append(dict(candidate_id=candidate, actual_valid_public_rows=int(valid.sum()),
            CPU_vs_actual_GPU_max_action_error=maximum, original_action_tolerance=5e-4))
        torch.save(dict(schema=SCHEMA, source_public_policy=source, source_policy_sha256=sha(source_path),
            parameters=source['parameters'].clone(), asset_sha256=source['asset_sha256'],
            runtime_overrides=entries[candidate]['candidate']['overrides'], native_execution=True,
            performance_inherited=False, formal_acceptance=False, numerical_execution=protocol['numerical_execution'],
            protocol_sha256=sha(protocol_path), actual_native_search=dict(output=str(run), candidate_id=candidate,
            original128_summary=summaries[candidate])), HERE/f'displayed_candidate_{candidate:03d}.pt')
    cases = []
    for case_id, (candidate, lane) in enumerate(selected):
        row = all_rows[candidate][lane]; checked = independent_episode(row, protocol)
        assert not checked['complete'] and not checked['airborne_contact'] and row['numerical_failures'] == 0
        with np.load(run/f'all_first_episodes_{candidate:03d}.npz') as arrays:
            ids = np.flatnonzero(arrays['valid_first_episode'][:, lane])
            assert np.array_equal(ids, np.arange(len(ids)))
            clock = arrays['episode_clock_s'][ids, lane]
            assert abs(clock[0]) < 1e-6 and abs(np.diff(clock)-.02).max() < 2e-6
            np.savez_compressed(snapshot/f'case{case_id}.npz', link_poses=arrays['link_poses'][ids, lane],
                ball_position_m=arrays['ball_position_m'][ids, lane], episode_clock_s=clock,
                valid_first_episode=np.ones(len(ids), dtype=bool), native_episode_index=np.full(len(ids), row['episode_index']))
        events = [event for event in read(run/f'contacts_{candidate:03d}.json')
                  if event['env_id'] == lane and event['episode_index'] == row['episode_index']]
        event = events[0] if events else None
        if case_id == 0:
            assert row['frame_hit'] and row['jump']['body_landing'] and not row['jump']['stable_recovery']
            title, status = '新布局原样策略｜拍框误击与身体触地', '拍框接触，身体触地；合格空中拍线与恢复均失败'
            limitation = '原128例：0完整、0合格空中拍线；这段不是成功回合'
        elif case_id == 1:
            assert not row['had_contact'] and row['jump']['stable_recovery']
            title, status = '新布局原样策略｜恢复站稳，但漏球', '机器人稳定恢复，没有实际球拍接触；完整任务失败'
            limitation = '正常播放从地面到8秒；站稳不计作接触后的合格恢复'
        else:
            assert row['robot_net_fault']
            title, status = '更换引拍姿态｜触网违规', '实际机器人或球拍触网，回合失效；没有合格空中拍线'
            limitation = f'实际终止{row["terminal_native_state"]["time_s"]:.2f}秒；不隐藏改引拍后的退化'
        cases.append(dict(id=case_id, run=run.name, env_id=lane, episode_index=row['episode_index'],
            source_training_index=lane, candidate_id=candidate, row=row, contact=event, title=title,
            subtitle=f'TRAIN{lane}｜候选{candidate}｜原生回合{row["episode_index"]}｜数值修正后的真实执行',
            terminal_status=status, limitation=limitation, selected_source_frames=ids.tolist()))
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), cases=cases,
        opening_title='当前新布局实录｜击球与恢复仍未完成',
        opening_lines=['原样与更换引拍两个已结束候选，各128例均0完整。',
            '保留拍框摔落、站稳漏球和实际触网三个失败。',
            '全部侧视、俯视和慢放来自真实50Hz全身记录。'],
        closing_title='需要重新适配全身动作',
        closing_lines=['两候选数值故障0，但没有合格空中拍线接触。',
            '原球网、完整球拍、目标区与硬件限制全部保留。',
            '当前8项比较继续；之后比较起跳、走位与恢复时序。'],
        evaluation_footer='origin0新布局｜两个完成候选各128例0完整｜整轮及冷评估待结束｜非正式A/B',
        original_source_hashes={str(path):sha(path) for path in sources},
        completed_candidate_summaries=summaries,
        caveats=['Posthoc selected actual failures, not a random sample. Each actual native episode identity is preserved.',
            'Snapshot covers two completed128-case candidates from an ongoing eight-candidate producer. No whole-course or cold outcome inferred.',
            'Native50Hz robot/ball motion and original exact contacts/terminal events. No interpolation, generated motion or state assignment.',
            'Old16m-layout3/128 and old successful footage are not inherited or relabelled as new-layout performance.'],
        new_native_execution=False, formal_cases_read=False, formal_acceptance=False)
    (snapshot/'cases.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    assert all(sha(Path(path)) == digest for path, digest in held.items())
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), completed_candidate_summaries=summaries,
        selected_candidates_and_TRAIN_lanes=selected, original_native_episode_indices=[case['episode_index'] for case in cases],
        all256_original_candidate_rows_independently_checked=True, actual_public_replays=replay_receipts,
        held471_inputs_unchanged=True, new_native_execution=False, formal_cases_read=False, formal_acceptance=False)
    (HERE/'NATIVE_EPISODE_CHECK.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
