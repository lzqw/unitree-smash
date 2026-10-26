"""Freeze finished Aim candidate2: a new complete case and two honest failures."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
MAIN = ROOT.parent/'unitree_smash'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(MAIN))
from quadsmash.evaluation_v4 import independent_episode, FAULTS
from quadsmash.control.full_racket_body_search_v4 import summarize_candidate
from quadsmash.control.full_racket_contact_aim_v4 import ContactAimControllerV4, SCHEMA


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert Path.cwd() == MAIN
    assert not (HERE/'PLAN.json').exists()
    run = MAIN/'reports/G07_full_racket_v4/contact_aim7_sequential_n128_01'
    source_path = MAIN/'reports/G07_full_racket_v4/body_strike_grid16_n2048_01/best_candidate.pt'
    old_run = MAIN/'reports/G07_full_racket_v4/body_strike_grid16_fixed_train128_01'
    protocol = read(MAIN/'configs/g07_full_racket_eval_v4_r3.json'); protocol['_root'] = str(MAIN)
    owner = read(MAIN/'reports/G07_full_racket_v4/contact_aim_after_body48_queue01/ADMISSION_DECISION.json')
    assert all(sha(Path(path)) == digest for path, digest in owner['source_sha256'].items())
    history = read(run/'metrics.json')['history']; entry = next(item for item in history if item['candidate_id'] == 2)
    assert entry['aim_uv_m'] == [0., .06] and entry['control_ticks'] == 800 and entry['env_steps'] == 102400
    rows = read(run/'episodes_002.json'); old_rows = read(old_run/'episodes.json')
    summary = summarize_candidate(rows, protocol); assert summary == entry['summary']
    assert summary['complete'] == 5 and summary['airborne_contacts'] == 60
    assert summary['conditional_recovery_count'] == 28 and summary['numerical_failures'] == 0
    complete = [row['env_id'] for row in rows if independent_episode(row, protocol)['complete']]
    new_complete = [row['env_id'] for row, old in zip(rows, old_rows)
                    if independent_episode(row, protocol)['complete'] and not independent_episode(old, protocol)['complete']]
    selected = [min(new_complete), 13, 0]
    assert selected == [12, 13, 0]
    sources = [run/name for name in ['episodes_002.json','contacts_002.json','all_first_episodes_002.npz',
        'public_and_actions_002.npz','initial_ground_states_002.json','causal_plans_002.json','geometry.json','geometry.npz','INPUTS.json']]
    sources += [source_path, old_run/'episodes.json', MAIN/'configs/g07_full_racket_eval_v4_r3.json', MAIN/'configs/court_net_v4.json']
    assert sha(ROOT/'configs/court_net_v4.json') == sha(MAIN/'configs/court_net_v4.json')
    (HERE/'PLAN.json').write_text(json.dumps(dict(created_at=datetime.now(timezone.utc).isoformat(),
        candidate_id=2, aim_uv_m=[0., .06], selected_TRAIN_indices=selected,
        selection='Posthoc first newly complete TRAIN case, old successful TRAIN13 now failed, and fixed TRAIN0 frame contact.',
        source_sha256={str(path):sha(path) for path in sources},
        complete_TRAIN_indices=complete, new_complete_TRAIN_indices=new_complete,
        scope='This candidate finished all128 real first episodes. Overall seven-candidate job and cold confirmation are still pending.',
        new_native_execution=False, formal_cases_read=False, formal_acceptance=False), indent=2)+'\n')
    snapshot = HERE/'snapshot'; snapshot.mkdir()
    native = snapshot/run.name; native.mkdir()
    for path in sources:
        if path.parent == run:
            shutil.copy2(path, native/path.name)
    (HERE/'ACTUAL_PARTIAL_JOB_SNAPSHOT.json').write_text(json.dumps(dict(job=read(run/'job.json'),
        progress=read(run/'progress.json'), metrics=read(run/'metrics.json')), indent=2)+'\n')
    initial = read(run/'initial_ground_states_002.json'); contacts = read(run/'contacts_002.json')
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    assert sha(source_path) == read(run/'INPUTS.json')['source_policy_sha256']
    with np.load(run/'public_and_actions_002.npz') as arrays:
        public, actions, valid = [arrays[key].copy() for key in ('observations','actions','valid_first_episode')]
    inputs = read(run/'INPUTS.json')
    default, limits = [torch.tensor(inputs[key], dtype=torch.float32) for key in
                       ['default_joint_position_rad','joint_limits_rad']]
    controller = ContactAimControllerV4(source, [0., .06], 2, 'cpu')
    maximum = 0.; previous = None
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
    torch.save(dict(schema=SCHEMA, source_public_policy=source, source_policy_sha256=sha(source_path),
        parameters=source['parameters'].clone(), asset_sha256=source['asset_sha256'], aim_uv_m=[0., .06],
        native_execution=True, performance_inherited=False, formal_acceptance=False,
        actual_native_search=dict(output=str(run), candidate_id=2, original128_summary=summary)), HERE/'displayed_candidate.pt')
    cases = []
    with np.load(run/'all_first_episodes_002.npz') as arrays:
        for i, lane in enumerate(selected):
            row = rows[lane]; checked = independent_episode(row, protocol)
            assert row['env_id'] == lane and row['candidate_id'] == 2 and row['ground_init']
            assert row['episode_index'] == initial['episode_ids'][lane] and row['numerical_failures'] == 0
            keep = arrays['valid_first_episode'][:, lane]
            ids = np.flatnonzero(keep)
            assert np.array_equal(ids, np.arange(len(ids)))
            clock = arrays['episode_clock_s'][ids, lane]
            assert abs(clock[0]) < 1e-6 and np.max(abs(np.diff(clock)-.02)) < 2e-6
            np.savez_compressed(snapshot/f'case{i}.npz', link_poses=arrays['link_poses'][ids, lane],
                ball_position_m=arrays['ball_position_m'][ids, lane], episode_clock_s=clock,
                valid_first_episode=np.ones(len(ids), dtype=bool), native_episode_index=np.full(len(ids), row['episode_index']))
            events = [event for event in contacts if event['env_id'] == lane and event['episode_index'] == row['episode_index']]
            event = next((event for event in events if event['string_contact']), events[0] if events else None)
            speed = float(np.linalg.norm(row['outgoing_velocity_mps']))
            if i == 0:
                assert checked['complete'] and not independent_episode(old_rows[lane], protocol)['complete']
                title = '新搜索成功例｜同回合完整扣杀成立'
                status = '空中拍线、攻击出球、原目标首落与合格恢复全部通过'
                limitation = f"出球{speed:.2f}米/秒；连续稳定{row['jump']['recovery_s']:.2f}秒；NF0"
            elif i == 1:
                assert row['landed_target'] and checked['airborne_contact'] and row['jump']['body_landing'] and not checked['complete']
                title = '失败对照｜击球命中，但身体触地'
                status = '出球和目标落点通过，恢复失败；完整任务失败'
                limitation = f"原教师TRAIN13曾成功；本候选同一来球失败，终止{row['terminal_native_state']['time_s']:.2f}秒"
            else:
                assert row['frame_hit'] and not checked['complete']
                title = '失败对照｜拍框误击'
                status = '发生拍框接触，没有合格空中拍线击球；完整任务失败'
                limitation = '拍框出球速度不计为空中拍线扣杀成功；保留完整机器人回合'
            cases.append(dict(id=i, run=run.name, env_id=lane, episode_index=row['episode_index'],
                source_training_index=lane, candidate_id=2, row=row, contact=event, title=title,
                subtitle=f'TRAIN{lane}｜参考点[0,0.06]米｜已完成候选的实际原生回合',
                terminal_status=status, limitation=limitation, selected_source_frames=ids.tolist()))
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), cases=cases,
        opening_title='新瞄准点搜索｜原128例出现5次完整成功',
        opening_lines=['展示一个新成功例，再保留恢复失败与拍框误击。',
            '该候选128例：5完整、60空中、28合格恢复，数值故障0。',
            '整体7点搜索与单独冷启动验证仍未结束。'],
        closing_title='搜索有改进，也有原成功例退化',
        closing_lines=['5/128是本次搜索成绩，尚未用冷启动确认。',
            '原2×2米目标、球拍和硬件限制全部保留。',
            '正式A/B各200与50/70/85门槛仍未通过。'],
        evaluation_footer='参考点[0,0.06]已完成候选｜5/128搜索成绩｜冷启动待验证｜非正式A/B',
        original_source_hashes={str(path):sha(path) for path in sources},
        caveats=['Posthoc selected TRAIN12 complete, formerly complete TRAIN13 now failed, and fixed TRAIN0 frame contact; not a random sample.',
            'All128 actual rows for this candidate are independently checked. The overall producer job was still running at snapshot time.',
            'Real50Hz body/ball poses, exact contact and terminal metadata. No interpolation or generated movement.',
            'Each native episode_index is preserved; three different episodes are never spliced into one successful episode.'],
        candidate_original128_summary=summary, new_native_execution=False, formal_cases_read=False, formal_acceptance=False)
    (snapshot/'cases.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), candidate_original128_summary=summary,
        displayed_indices=selected, original_native_episode_indices=[case['episode_index'] for case in cases],
        actual_valid_public_rows=int(valid.sum()), CPU_vs_actual_GPU_max_action_error=maximum,
        original_action_tolerance=5e-4, all128_original_rows_independently_checked=True,
        all_current_owner419_inputs_unchanged=True, new_native_execution=False, formal_acceptance=False)
    assert all(sha(Path(path)) == digest for path, digest in owner['source_sha256'].items())
    (HERE/'NATIVE_EPISODE_CHECK.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
