"""Snapshot actual saved ground-start TRAIN episodes, including nonzero IDs.

Default requires completed native receipts. --allow-running permits explicitly
labelled closed episodes of a verified live native job, never population scores.
This program does not execute policies or generate/interpolate motion.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from quadsmash.evaluation_v4 import independent_episode
from quadsmash.rl.episode_trace_v4 import unpack_public, SCHEMA


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same(left, right):
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def parse_key(value):
    lane, episode = map(int, value.split(':'))
    if lane < 0 or episode < 0:
        raise ValueError('Nonnegative actual lane and episode IDs required')
    return lane, episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episode', type=parse_key, action='append', required=True)
    parser.add_argument('--allow-running', action='store_true')
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    assert run.parent == ROOT/'reports/G07_full_racket_v4'
    assert not output.exists() and len(set(args.episode)) == len(args.episode)
    protocol_path = ROOT/'configs/g07_full_racket_eval_v4_r3.json'
    protocol = read(protocol_path); protocol['_root'] = str(ROOT)
    manifest_path = run/'trajectory_traces/manifest.json'
    manifest_bytes = manifest_path.read_bytes(); recorded = json.loads(manifest_bytes)
    assert recorded['schema'] == SCHEMA and recorded['enabled'] and recorded['disabled_error'] is None
    assert recorded['native_execution'] and recorded['provenance']['native_source_confirmed']
    assert recorded['provenance']['protocol_sha256'] == sha(protocol_path)
    assert recorded['provenance']['asset_sha256'] == protocol['asset']['sha256']
    names = recorded['body_names']; assert names == read(run/'geometry.json')['body_names']
    tool = names.index('Racket')
    inputs = read(run/'INPUTS.json')
    assert inputs['contract']['protocol_sha256'] == sha(protocol_path)
    live = None
    complete_job = (run/'job.json').exists() and read(run/'job.json')['status'] == 'SUCCEEDED'
    if complete_job:
        job = read(run/'job.json')
        assert job['exit_code'] == 0 and job['charged_env_steps'] == read(run/'completed.json')['env_steps'] > 0
        raw_path = run/'episodes.json'
    else:
        assert args.allow_running, 'Wait for completed native receipts or explicitly request a running-job snapshot'
        active = [entry for entry in read(ROOT/'status/GPU_LEDGER.json')['runs'] if entry['status'] == 'RUNNING']
        assert len(active) == 1 and Path(active[0]['output']) == run
        live = {name:active[0][name] for name in ['pid','child_pid']}
        for pid in live.values():
            os.kill(pid, 0)
            assert str(run).encode() in Path('/proc',str(pid),'cmdline').read_bytes()
        raw_path = run/'episodes.partial.json'
    raw_bytes = raw_path.read_bytes(); raw = json.loads(raw_bytes)
    indexed = {(row['env_id'],row['episode_index']):row for row in raw}
    assert len(indexed) == len(raw)
    selected = []
    # Verify every requested episode before creating any partial snapshot.
    for key in args.episode:
        assert key in indexed, 'Wait for the actual producer to publish the closed raw row: '+str(key)
        entry = next(value for value in recorded['saved_episodes'] if (value['env_id'],value['episode_index']) == key)
        source = run/'trajectory_traces'/entry['data_file']
        metadata = source.with_suffix('.json'); meta = read(metadata); row = meta['raw_episode']
        assert meta['native_execution'] and same(row,indexed[key]) and row['ground_init']
        assert (row['env_id'],row['episode_index']) == key
        assert row['source_training_index'] == entry['source_training_index'] == meta['source_training_index']
        assert sha(source) == meta['data_sha256'] == entry['data_sha256']
        assert meta['full_public_reconstruction_bitwise_verified']
        outcome = independent_episode(row,protocol)
        contacts = meta['actual_contact_events']
        assert len(contacts) == row['contact_count']
        assert all((event['env_id'],event['episode_index']) == key for event in contacts)
        if complete_job:
            original = [event for event in read(run/'contacts.json') if (event['env_id'],event['episode_index']) == key]
            assert same(original,contacts)
        event = next((event for event in contacts if event['first_racket_contact']),None)
        if row['had_contact']:
            assert event is not None and abs(event['time_s']-row['contact_time_s']) < 1e-7
        with np.load(source) as stored:
            data = {name:stored[name] for name in stored.files}
        public = unpack_public(data['packed_public']); length = len(public)
        assert length == meta['samples'] == entry['samples']
        assert hashlib.sha256(public.tobytes()).hexdigest() == meta['full_public794_sha256'] == entry['full_public794_sha256']
        assert np.array_equal(data['episode_clock_s'],public[:,775])
        assert abs(public[0,775]) < 1e-6 and np.max(abs(np.diff(public[:,775])-.01)) < 2e-6
        assert abs(float(public[-1,775])+.01-row['terminal_native_state']['time_s']) < 2e-5
        poses = data['link_poses']; assert poses.shape == (length,len(names),7) and np.isfinite(poses).all()
        assert np.isfinite(public).all() and np.isfinite(data['ball_position_m']).all()
        inverse = Rotation.from_quat(poses[:,0,3:][:,[1,2,3,0]]).inv()
        local_tool = inverse.apply(poses[:,tool,:3]-poses[:,0,:3])
        tool_error = float(np.max(np.linalg.norm(local_tool-public[:,45:48],axis=1)))
        normal = Rotation.from_quat(poses[:,tool,3:][:,[1,2,3,0]]).apply(np.tile([1.,0.,0.],(length,1)))
        normal_error = float(np.max(np.linalg.norm(inverse.apply(normal)-public[:,48:51],axis=1)))
        present = public[:,67] > .5
        ball_error = float(np.max(np.linalg.norm(inverse.apply(data['ball_position_m']-poses[:,0,:3])[present]-public[present,61:64],axis=1))) if present.any() else 0.
        assert tool_error <= 2e-4 and ball_error <= 2e-4 and normal_error <= 5e-5
        actual_iterations = np.unique(data['learner_iteration']).tolist()
        assert actual_iterations == meta['actor_iterations_encountered'] == entry['actor_iterations_encountered']
        selected.append((entry,source,metadata,meta,row,outcome,event,data,
            dict(tool_public_pose_max_error_m=tool_error,normal_public_pose_max_error=normal_error,
                 ball_public_pose_max_error_m=ball_error)))
    output.mkdir(parents=True)
    destination = output/run.name; destination.mkdir()
    hashes = {}
    for name in ['geometry.json','geometry.npz','asset_dependencies.json','INPUTS.json','initial_ground_states.json',
                 'source_hashes.json','source_snapshot.zip','dirty.patch']:
        source = run/name; target = destination/name
        shutil.copy2(source,target); assert sha(source) == sha(target)
        hashes[str(source.relative_to(ROOT))] = sha(source)
    # Preserve the exact live producer publication rather than a later version.
    (destination/'observed_episodes.json').write_bytes(raw_bytes)
    (destination/'observed_trace_manifest.json').write_bytes(manifest_bytes)
    cases = []
    for index,(entry,source,metadata,meta,row,outcome,event,data,checks) in enumerate(selected):
        ids = np.arange(0,len(data['episode_clock_s']),2)
        clock = data['episode_clock_s'][ids]
        assert len(clock) > 1 and np.max(abs(np.diff(clock)-.02)) < 2e-6
        np.savez_compressed(output/f'case{index}.npz',link_poses=data['link_poses'][ids],
            ball_position_m=data['ball_position_m'][ids],episode_clock_s=clock,
            valid_first_episode=np.ones(len(ids),dtype=bool))
        # valid_first_episode is the legacy renderer mask for this selected
        # complete ground-start trace. Its actual original episode ID is kept.
        for path in [source,metadata]:
            shutil.copy2(path,destination/path.name); assert sha(path) == sha(destination/path.name)
            hashes[str(path.relative_to(ROOT))] = sha(path)
        velocity = np.asarray(row['outgoing_velocity_mps']); landing = np.asarray(row['first_landing_m'])
        outgoing = f'出球{np.linalg.norm(velocity):.3f}m/s、vz{velocity[2]:+.2f}' if np.isfinite(velocity).all() else '未记录有效出球'
        landing_text = f'首落[{landing[0]:.3f},{landing[1]:.3f}]m' if np.isfinite(landing).all() else '终止前未观察到首次落点'
        kind = '完整成功' if outcome['complete'] else '空中触球但未完整成功' if outcome['airborne_contact'] else '漏球或无合格空中触球'
        cases.append(dict(id=index,run=run.name,source_type='actual_saved_changing_policy_training_trace',
            env_id=row['env_id'],episode_index=row['episode_index'],source_training_index=row['source_training_index'],
            title=f'真实训练回合｜TRAIN{row["source_training_index"]}｜{kind}',
            subtitle=f'LANE{row["env_id"]}、EP{row["episode_index"]}｜从地面执行｜变化中策略，非冻结成绩',
            terminal_status=('本训练回合完整扣杀成立' if outcome['complete'] else '本训练回合完整扣杀失败'),
            limitation=f'{outgoing}；{landing_text}；稳定窗口{row["jump"]["recovery_s"]:.2f}s',
            row=row,contact=event,selected_source_frames=ids.tolist(),source_sampling_hz=100,display_sampling_hz=50,
            actor_iterations_encountered=entry['actor_iterations_encountered'],whole_source_job_complete=complete_job,
            recorder_selection_reasons=entry['selection_reasons'],independent_outcome=outcome,trace_checks=checks,
            renderer_mask_semantics='All samples belong to this selected actual ground-start episode, with original nonzero ID preserved.'))
    count = sum(case['independent_outcome']['complete'] for case in cases)
    has_failure = count < len(cases)
    hashes[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    result = dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='Actual closed native TRAIN episodes. Explicit selections occur after seeing outcomes; no frozen-policy or population score.',
        cases=cases,opening_title='真实训练动作｜完整回合与失败对照',
        opening_lines=['同一完整球拍与原球网，从地面实际执行。',f'本视频选择{len(cases)}例，其中{count}例完整成功；选择发生在观察结果之后。',
            '策略、归一化与探索噪声随训练变化，不能代替冻结评测。'],
        closing_title='单回合可行性｜稳定复现仍待评测',
        closing_lines=['完整成功须同时通过原攻击、落点、恢复和规则判定。',
            '接近速度或落点边界的成功余量小，不代表稳定能力。','最终仍需同一冻结策略通过原A/B各200回合。'],
        evaluation_footer=f'实际变化中策略训练｜本视频{count}/{len(cases)}例完整｜非成功率估计、非正式A/B',
        original_source_hashes=hashes,observed_raw_publication_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        observed_trace_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),source_job_complete=complete_job,
        live_native_job_verified=live,includes_failure=has_failure,
        caveats=['Actual100Hz whole-body poses/actions reduced to every second recorded sample for50Hz; no interpolation or generated motion.',
            'Actual original lane, episode and TRAIN IDs are preserved; canonical resume does not imply episode0.',
            'Recorder selection reasons are retained. Selection for this video is explicitly after outcomes, with failure examples when selected.',
            'Weights, normalization and sampled exploration vary within training; no single frozen policy performance is inferred.',
            'During a live job, raw episode publications can lag immutable saved traces. Missing rows must wait, never be fabricated.',
            'Exact terminal poses and exact first-contact events are native records. First-landing markers use original event locations with a nearby50Hz body pose.'],
        formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    (output/'cases.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(snapshot=str(output),keys=args.episode,complete_selected=count,includes_failure=has_failure,
        whole_job_complete=complete_job,live_native_job_verified=live,frames=[len(case['selected_source_frames'])for case in cases]),ensure_ascii=False))


if __name__ == '__main__':
    main()
