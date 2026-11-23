"""Export contiguous post-contact body trajectories from the verified native clip index."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from quadsmash.rl.supermimic_archive_storage_v1 import read_archive_rows, require_archive_space


STATE_FIELDS = (
    'elapsed_tick', 'history', 'history_tick', 'body_position_w', 'body_quaternion_w',
    'body_linear_velocity_w', 'body_angular_velocity_w', 'applied_torque', 'foot_net_force_w',
    'shuttle_state_w', 'event_state',
)
ACTION_FIELDS = ('requested_action', 'base_action', 'final_action', 'final_joint_targets', 'reward',
                 'terminated', 'timeout')
MIXER_STATE = {
    'arm_target': 'controller/mixer/arm_rate/target',
    'arm_clock': 'controller/mixer/arm_rate/clock',
    'arm_initialized': 'controller/mixer/arm_rate/initialized',
    'last_requested': 'controller/mixer/last_requested',
    'last_actual': 'controller/mixer/last_actual',
}


def lane_tensor(tensor, lane, lanes):
    if tensor.ndim == 0 or tensor.shape[0] != lanes:
        raise ValueError('Expected an archived per-lane tensor')
    return tensor[lane].clone()


def lane_state(snapshot, lane, lanes):
    state = {key: lane_tensor(snapshot[key], lane, lanes) for key in STATE_FIELDS}
    state.update({component: {key: lane_tensor(value, lane, lanes) for key, value in snapshot[component].items()}
                  for component in ('physical', 'task', 'recovery')})
    return state


def capture_step(row, lane, lanes, *, frozen):
    step = {key: lane_tensor(row[key], lane, lanes) for key in ACTION_FIELDS}
    step['arm_identification_active'] = (torch.tensor(False) if frozen else
                                         lane_tensor(row['arm_identification_active'], lane, lanes))
    for side in ('before', 'after'):
        step[side] = lane_state(row[side], lane, lanes)
        memory = row['executor_'+side]['tensors']
        step['mixer_'+side] = {key: lane_tensor(memory[path], lane, lanes) for key, path in MIXER_STATE.items()}
    return step


def stack_steps(steps):
    return {key: stack_steps([step[key] for step in steps]) if isinstance(value, dict)
            else torch.stack([step[key] for step in steps]) for key, value in steps[0].items()}


def export_run(registered, scratch):
    from index_supermimic_recovery_tails import validate_frozen_source

    run = Path(registered['training_run'])
    archive = Path(registered['archive'])
    kind = registered.get('source_kind', 'training')
    if kind not in ('training', 'frozen_natural'):
        raise ValueError(f'Unrecognized recovery source kind {kind}')
    frozen = kind == 'frozen_natural'
    raw = (archive/'manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != registered['manifest_sha256']:
        raise ValueError(f'Archive manifest changed after indexing {run}')
    manifest = json.loads(raw)
    episode_bytes = (run/('episodes_summary.json' if frozen else 'native_episodes.jsonl')).read_bytes()
    if hashlib.sha256(episode_bytes).hexdigest() != registered['episode_records_sha256']:
        raise ValueError(f'Native episode records changed after indexing {run}')
    episodes = json.loads(episode_bytes) if frozen else [json.loads(line) for line in episode_bytes.splitlines()]
    if frozen:
        validate_frozen_source(run, manifest, episodes)
    episodes = {(row['env_id'], row['episode_index']): row for row in episodes}
    config_bytes = (run/('startup.json' if frozen else 'config.json')).read_bytes()
    reserved = set((manifest if frozen else json.loads(config_bytes))['wm_holdout_lanes'])
    lanes = manifest['num_envs']
    clips = registered['clips']
    timelines = [[] for _ in clips]
    by_chunk = {}
    receipts = {entry['file']: entry for entry in manifest['chunks']}
    for i, clip in enumerate(clips):
        native = episodes[clip['env_id'], clip['episode_index']]
        if (clip['case_id'] != native['case_id']
                or clip['arm_intervention'] != (False if frozen else native['arm_identification_episode'])
                or (frozen and clip['env_id'] in reserved)
                or clip['ball_completed'] != native['ball_side_complete_except_recovery']
                or clip['physical_recovery_success'] != native['physical_recovery_success']
                or not native['approach_ball_terminal']['recovery_tail_observed'] or native['numerical_failures']):
            raise ValueError(f'Native clip labels changed for {clip["case_id"]}')
        for chunk in clip['chunks']:
            if receipts[chunk['file']]['sha256'] != chunk['sha256']:
                raise ValueError(f'Clip chunk signature changed for {clip["case_id"]}')
            by_chunk.setdefault(chunk['file'], []).append(i)

    for receipt in manifest['chunks']:
        name = receipt['file']
        if name not in by_chunk:
            continue
        with read_archive_rows(archive, receipt, scratch=scratch) as rows:
            for i in by_chunk[name]:
                clip = clips[i]
                lane = clip['env_id']
                start, stop = clip['contact'], clip['terminal']
                first = start['row_offset'] if start['file'] == name else 0
                last = stop['row_offset']+1 if stop['file'] == name else len(rows)
                for offset in range(first, last):
                    row = rows[offset]
                    tick = row['before']['control_tick'] if frozen else row['native_tick']
                    final = tick == stop['native_tick']
                    if (tick != start['native_tick'] + len(timelines[i])
                            or tick > stop['native_tick']
                            or int(row['before']['episode_index'][lane]) != clip['episode_index']
                            or int(row['after']['episode_index'][lane]) != clip['episode_index']
                            or bool(row['terminated'][lane] or row['timeout'][lane]) != final):
                        raise ValueError(f'Discontinuous or reset trajectory for {clip["case_id"]}')
                    timelines[i].append(capture_step(row, lane, lanes, frozen=frozen))

    exported = []
    for clip, steps in zip(clips, timelines, strict=True):
        if len(steps) != clip['control_steps'] or not steps:
            raise ValueError(f'Missing transitions for {clip["case_id"]}')
        trajectory = stack_steps(steps)
        q_before = trajectory['before']['physical']['joint_position']
        q_after = trajectory['after']['physical']['joint_position']
        if not torch.equal(q_before[1:], q_after[:-1]) or not torch.isfinite(trajectory['final_joint_targets']).all():
            raise ValueError(f'Joint state or execution history is invalid for {clip["case_id"]}')
        before, after = trajectory['before']['task'], trajectory['after']['task']
        if bool(before['had_contact'][0]) or not bool(after['had_contact'][0]):
            raise ValueError(f'First contact evidence changed for {clip["case_id"]}')
        release = clip['release']
        if release is not None:
            offset = release['native_tick'] - clip['contact']['native_tick']
            if not 0 <= offset < len(steps) or before['released'][offset] or not after['released'][offset]:
                raise ValueError(f'Release evidence changed for {clip["case_id"]}')
        exported.append(dict(
            source=clip, wm_holdout_lane=clip['env_id'] in reserved,
            control_dt_s=manifest['control_dt_s'], trajectory=trajectory,
        ))
    return dict(source=registered['training_run'], source_kind=kind,
                source_manifest_sha256=registered['manifest_sha256'],
                source_config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                native_episodes_sha256=registered['episode_records_sha256'],
                joint_names=manifest['joint_names'], body_names=manifest['body_names'],
                task_contract=manifest['task_contract'], clips=exported)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    index_bytes = args.index.read_bytes()
    index = json.loads(index_bytes)
    if index['schema'] != 'supermimic-stage2-recovery-tail-index-v1':
        raise ValueError('Expected the verified native recovery tail index')
    runs = [export_run(run, args.scratch) for run in index['runs']]
    if sum(len(run['clips']) for run in runs) != index['clips']:
        raise ValueError('Exported clip count differs from the verified index')
    report = dict(
        schema='supermimic-stage2-recovery-trajectories-v1',
        source_index_sha256=hashlib.sha256(index_bytes).hexdigest(),
        scope='Measured body, actions and mixer histories for Recovery supervision. Complete execution snapshots remain in the source archives.',
        runs=runs,
    )
    # Reserve space for the complete serialized payload and the usual recovery checkpoint margin.
    def tensor_bytes(record):
        if isinstance(record, torch.Tensor):
            return record.numel()*record.element_size()
        if isinstance(record, dict):
            return sum(map(tensor_bytes, record.values()))
        if isinstance(record, list):
            return sum(map(tensor_bytes, record))
        return 0

    require_archive_space(args.output.parent, tensor_bytes(report)+4*1024**2)
    staged = args.output.with_suffix('.pending')
    with staged.open('xb') as stream:
        torch.save(report, stream)
    staged.replace(args.output)
    print(f'Exported {index["clips"]} recovery trajectories to {args.output}', flush=True)


if __name__ == '__main__':
    main()
