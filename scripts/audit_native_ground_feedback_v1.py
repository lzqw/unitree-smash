"""Audit continuous ground-only feedback without assuming an MPC solve schedule.

Separates contact observation, light load and loaded contact. It checks the
actual observer/command clocks and retains every disagreement with the old
net-force flag. No disagreement is automatically labelled a failure cause.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.native_ground_support_state_v1 import ground_contact_update


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    if a.output.exists():
        raise ValueError('Fresh audit required')
    read = lambda p: json.loads(Path(p).read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    job = read(a.run/'job.json'); meta = read(a.run/'controller_ground_observation.json')
    observer = ROOT/'scripts/native_ground_support_state_v1.py'
    archive = a.run/'controller_ground_observation.npz'
    if (job['status'] != 'SUCCEEDED' or job['exit_code'] != 0 or not meta['complete'] or
            sha(archive) != meta['data_sha256'] or sha(observer) != meta['source_sha256']):
        raise ValueError('Completed unchanged native ground telemetry required')
    with np.load(archive, allow_pickle=False) as z:
        d = {k: z[k].copy() for k in z.files}
    if not np.array_equal(d['tick'], np.arange(800)) or any(not np.isfinite(x).all() for x in d.values()):
        raise ValueError('Observer clock or finite data invalid')
    contacts = np.zeros(4, bool); episode = None
    for i in range(800):
        if episode != d['episode_index'][i]:
            contacts[:] = False
        if not np.array_equal(contacts, d['prior_contacts'][i]):
            raise ValueError('Missing reset or control-rate history')
        contacts, modes = ground_contact_update(contacts, d['ground_normal_force_world_n'][i])
        if not np.array_equal(contacts, d['ground_contacts'][i]) or any(
                not np.array_equal(v, d[k][i]) for k, v in modes.items()):
            raise ValueError('Observer cannot be reproduced')
        episode = d['episode_index'][i]
    execution = read(a.run/'sequence_native_result.json')
    # Older MPC and new timed-reference adapters use different record keys.
    records = execution['records']
    ticks = np.array([r['tick'] if 'tick' in r else r['control_tick'] for r in records], dtype=int)
    with np.load(a.run/'sequence_native_trace.npz', allow_pickle=False) as z:
        trace = {k: z[k].copy() for k in z.files}
    if (meta['case'] != execution['case'] or len(ticks) != execution['applied_steps'] or
            not np.array_equal(ticks, np.arange(280, 280+len(ticks))) or
            trace['q'].shape != (len(ticks), 25) or np.any(d['episode_index'][ticks] != 0)):
        raise ValueError('Controller lifetime differs from actual first episode')
    if (not np.array_equal(trace['contacts'], d['ground_contacts'][ticks]) or
            not np.array_equal(trace['foot_force_world_n'], d['ground_normal_force_world_n'][ticks])):
        raise ValueError('Feedback input differs from independent ground observation')
    first = d['episode_index'] == 0
    changed = first & np.any(d['original_contacts'] != d['ground_contacts'], axis=1)
    differences = []
    for tick in np.flatnonzero(changed):
        i = int(tick)-280
        row = dict(tick=int(tick), time_s=float(tick*.01),
            original_force_flag=d['original_contacts'][tick].tolist(),
            ground_force_flag=d['ground_contacts'][tick].tolist(),
            ground_normal_force_world_n=d['ground_normal_force_world_n'][tick].tolist(),
            original_net_normal_world_n=d['original_net_normal_world_n'][tick].tolist(),
            light=d['light'][tick].tolist(), loaded=d['loaded'][tick].tolist(),
            establishing=d['establishing'][tick].tolist(), feedback_active=0 <= i < len(ticks))
        if row['feedback_active']:
            row.update(requested_target_rad=trace['requested_target_rad'][i].tolist(),
                actual_state_q=trace['q'][i].tolist(), actual_state_v=trace['v'][i].tolist())
        differences.append(row)
    paths = [Path(__file__), observer, a.run/'job.json', a.run/'controller_ground_observation.json',
        archive, a.run/'sequence_native_result.json', a.run/'sequence_native_trace.npz']
    result = dict(scope=__doc__, passed=True, case=meta['case'], control_records=800,
        feedback_steps=len(ticks), contact_change_ticks=d['tick'][changed].tolist(),
        light_contact_foot_samples=int(d['light'][first].sum()),
        establishing_foot_samples=int(d['establishing'][first].sum()),
        ground_contact_transitions=int(np.sum((d['ground_contacts'] != d['prior_contacts'])[first])),
        force_source='ground only', five_newton_exclusion_used=False,
        task_scoring_changed=False, student_observation_changed=False,
        independent_geometric_contact_evidence=False, disagreements_prove_cause=False,
        contact_disagreements=differences, valid_expert_labels=False,
        source_sha256={str(p.resolve()): sha(p) for p in paths})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    tmp=a.output.with_suffix('.tmp'); tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(a.output)
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','contact_disagreements']}))


if __name__ == '__main__':
    main()
