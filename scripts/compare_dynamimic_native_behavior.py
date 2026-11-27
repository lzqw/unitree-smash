"""Compare native recovery episodes from two behavior policies on matching launches."""
import argparse
import json
from pathlib import Path

import torch


def load(root):
    directory = root/'trajectories'
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest['schema'] != 'dynamimic-native-recovery-v1':
        raise ValueError('Unknown native recovery trajectory schema')
    keys = ('live', 'active', 'gate', 'before', 'request', 'final_joint_targets', 'done',
            'body', 'net', 'strict', 'relaxed')
    chunks = [torch.load(directory/name, weights_only=True, mmap=True) for name in manifest['chunks']]
    data = {key: torch.cat([chunk[key] for chunk in chunks]) for key in keys}
    if not ((data['done'] & data['live']).sum(0) == 1).all():
        raise ValueError('Expected one complete native episode per case')
    return manifest, data


def first_tick(mask):
    return torch.where(mask.any(0), mask.int().argmax(0), mask.shape[0])


def compare(before, after):
    for filename in ('actuator_contract.json', 'task_contract.json', 'initial_states.json'):
        if json.loads((before/filename).read_text()) != json.loads((after/filename).read_text()):
            raise ValueError(f'Native conditions differ in {filename}')
    old, a = load(before)
    new, b = load(after)
    for key in ('cases', 'origins', 'dt', 'lane_approach', 'companion_sha256'):
        if old[key] != new[key]:
            raise ValueError(f'Behavior archives differ in {key}')
    n = min(len(a['live']), len(b['live']))
    live = a['live'][:n] & b['live'][:n]
    first_gate = torch.minimum(first_tick(a['gate'][:n] & live), first_tick(b['gate'][:n] & live))
    prefix = live & (torch.arange(n)[:, None] < first_gate)
    eligible = (a['active'][:n] & b['active'][:n] & live).any(0)
    records = []
    for lane, case in enumerate(old['cases']):
        row = dict(case_id=case['case_id'], approach=old['lane_approach'][lane],
                   held=lane % 16 < 2, paired_active=bool(eligible[lane]))
        for key in ('before', 'request', 'final_joint_targets'):
            delta = (a[key][:n, lane] - b[key][:n, lane])[prefix[:, lane]]
            row['prefix_max_abs_'+key] = float(delta.abs().max()) if delta.numel() else None
        for key in ('body','strict','relaxed','net'):
            row[key] = [bool((data[key][:,lane] & data['live'][:,lane]).any()) for data in (a,b)]
        records.append(row)
    return dict(baseline=old['checkpoint_sha256'], candidate=new['checkpoint_sha256'], cases=records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('before', 'after', 'output'):
        parser.add_argument('--'+key, required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(compare(args.before, args.after), indent=2)+'\n')
