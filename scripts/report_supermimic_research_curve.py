"""Publish measured research-arm curves, with inherited interactions removed."""
import argparse
import json
import time
from pathlib import Path

from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter


def arm_origin(registration):
    import torch
    from run_supermimic_research_scale import digest
    cmd = registration['command']
    path = Path(registration.get('initial_checkpoint', cmd[cmd.index('--resume-training') + 1]))
    if digest(path) != registration['initial']['checkpoint_sha256']:
        raise ValueError('Initial training checkpoint changed')
    ck = torch.load(path, mmap=True, map_location='cpu', weights_only=False)
    return ck['global_counters']['accepted'], ck['native_ticks']


def axes(row, origin, lanes):
    retained = row['accepted_id'] - origin[0]
    steps = (row['native_ticks'] - origin[1]) * lanes
    if retained < 0 or steps < 0 or row['native_transitions'] != row['native_ticks'] * lanes:
        raise ValueError('Invalid research-arm counters')
    return dict(retained_updates=retained, new_native_env_steps=steps,
                training_wall_seconds=int(row['research_elapsed_s']))


def scalars(row):
    head = row['block']
    metrics = {f'{head}/{key}': row[key] for key in (
        'objective_before', 'objective_after', 'final_command_max_change', 'task_gradient_norm',
        'tracking_gradient_norm', 'effective_gate_lanes', 'critic_loss', 'contact_aux_weight',
        'native_collect_seconds', 'bptt_seconds', 'world_update_seconds') if key in row}
    metrics[head+'/accepted'] = int(row['update_accepted'])
    metrics['gpu/peak_allocated_GiB'] = row['peak_gpu_bytes'] / 2**30
    guard = row.get('fixed_native_forward')
    if guard:
        for group in ('body', 'flight', 'contact'):
            checks = guard[group]
            for item in checks.get('comparisons', checks.get('checks', [])):
                if item.get('candidate') is not None:
                    path = '/'.join(map(str, (group, item.get('horizon', 'event'), item['metric'])))
                    metrics['world/'+path] = item['candidate']
    fits = [r['contact_supervision'] for r in row.get('world_update_reports', []) if r.get('contact_supervision')]
    if fits:
        metrics['world/contact_supervision_batches'] = len(fits)
        metrics['world/contact_optimizer_steps'] = sum(int(r['updated']) for r in fits)
        metrics['world/sampled_contact_events'] = sum(r['measured_events'] for r in fits)
    return metrics


def read_new(path, offset):
    if not path.exists():
        return [], offset
    rows = []
    with path.open('rb') as stream:
        stream.seek(offset)
        for line in stream:
            if not line.endswith(b'\n'):
                break
            rows.append(json.loads(line))
            offset += len(line)
    return rows, offset


def main():
    import psutil
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm-dir', type=Path, required=True)
    parser.add_argument('--tensorboard', type=Path, required=True)
    parser.add_argument('--follow-pid', type=int)
    args = parser.parse_args()
    owner = psutil.Process(args.follow_pid) if args.follow_pid else None
    arm = args.arm_dir.resolve()
    reg = json.loads((arm/'registration.json').read_text())
    manifest_path = arm/'train/native_transitions/manifest.json'
    while owner and not manifest_path.exists():
        if not owner.is_running() or owner.status() == psutil.STATUS_ZOMBIE:
            return
        time.sleep(5)
    origin = arm_origin(reg)
    manifest = json.loads(manifest_path.read_text())
    writer = EventFileWriter(str(args.tensorboard))
    from scripts.report_supermimic_native_contact_counts import NativeContactCounts
    contacts = NativeContactCounts(arm)
    offset = 0
    heads = dict(reg.get('previous_heads', {'recovery': 0, 'approach': 0}))
    attempts = reg.get('previous_proposals', 0)
    try:
        while True:
            live = owner and owner.is_running() and owner.status() != psutil.STATUS_ZOMBIE
            contacts.update(writer)
            rows, offset = read_new(arm/'train/training_history.jsonl', offset)
            for row in rows:
                pos = axes(row, origin, manifest['num_envs'])
                metrics = scalars(row)
                heads[row['block']] += int(row['update_accepted'])
                attempts += 1
                if sum(heads.values()) != pos['retained_updates']:
                    raise ValueError('Recorded proposals do not account for every retained step')
                if row['update_accepted'] and pos['retained_updates'] == 16:
                    from scripts.report_supermimic_first_pair_budget import write_budget
                    budget = write_budget(arm)
                    if budget is not None:
                        metrics['timing/first_pair_wall_seconds'] = budget['first_pair_wall_seconds']
                        metrics['timing/estimated_remaining_hours_after_first_pair'] = budget['estimated_remaining_hours_after_first_pair']
                for axis, step in pos.items():
                    values = [Summary.Value(tag=axis+'/'+k, simple_value=v) for k,v in metrics.items()]
                    writer.add_event(Event(wall_time=time.time(), step=step, summary=Summary(value=values)))
            if rows:
                writer.flush()
                status = dict(arm=reg['arm'], **pos, accepted=dict(heads), proposals=attempts,
                              inherited_native_ticks=origin[1], inherited_updates=origin[0],
                              note='Training wall time starts after initial replay reconstruction.')
                tmp = arm/'learning_curve.json.tmp'
                tmp.write_text(json.dumps(status, indent=2)+'\n')
                tmp.replace(arm/'learning_curve.json')
            if not live:
                break
            time.sleep(10)
    finally:
        writer.close()


if __name__ == '__main__':
    main()
