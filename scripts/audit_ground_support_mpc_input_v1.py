"""Verify controller contact inputs against independent native ground records."""
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Fresh audit path required')
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    meta_path = a.run / 'controller_ground_observation.json'
    meta = json.loads(meta_path.read_text())
    data_path = a.run / 'controller_ground_observation.npz'
    if not meta['complete'] or sha(data_path) != meta['data_sha256']:
        raise ValueError('Incomplete ground observation or changed data')
    with np.load(data_path, allow_pickle=False) as z:
        d = {k: z[k].copy() for k in z.files}
    if not np.array_equal(d['tick'], np.arange(800)) or any(not np.isfinite(v).all() for v in d.values()):
        raise ValueError('Clock or finite-state violation')
    contacts = np.zeros(4, bool); episode = None
    for i in range(800):
        if episode != d['episode_index'][i]:
            contacts[:] = False
        if not np.array_equal(contacts, d['prior_contacts'][i]):
            raise ValueError('Observer history differs')
        contacts, modes = ground_contact_update(contacts, d['ground_normal_force_world_n'][i])
        if not np.array_equal(contacts, d['ground_contacts'][i]) or any(
                not np.array_equal(v, d[k][i]) for k, v in modes.items()):
            raise ValueError('Ground state update differs')
        episode = d['episode_index'][i]
    with np.load(a.run / 'sequence_native_trace.npz', allow_pickle=False) as z:
        trace = {k: z[k].copy() for k in z.files}
    if trace['q'].shape != (520, 25) or not np.array_equal(trace['contacts'], d['ground_contacts'][280:]):
        raise ValueError('Feedback did not receive the independently filtered ground contacts')
    if not np.array_equal(trace['foot_force_world_n'], d['ground_normal_force_world_n'][280:]):
        raise ValueError('Feedback still received net/self-contact forces')
    planning_path = a.run / 'foot_space_current_state_planning.json'
    planning = json.loads(planning_path.read_text())
    calls = planning['calls']
    if [r['control_tick'] for r in calls] != [400, 550, 700, 750]:
        raise ValueError('Declared full-remainder replanning schedule was not executed')
    for call in calls:
        tick = call['control_tick']; q = call['query']
        if not np.array_equal(q['contacts'], d['ground_contacts'][tick]):
            raise ValueError('Planning used a different contact source')
        if not call['accepted'] or not call['independent_replay_exact'] or len(call['prediction']['trajectory']) != 800-tick:
            raise ValueError('Missing independently replayed full remainder')
    first = d['episode_index'] == 0
    changed = np.any(d['original_contacts'] != d['ground_contacts'], axis=1) & first
    paths = [Path(__file__), ROOT/'scripts/native_ground_support_state_v1.py', meta_path,
        data_path, planning_path, a.run/'sequence_native_trace.npz']
    result = dict(passed=True, case=meta['case'], control_records=800, feedback_steps=520,
        planning_calls=len(calls), contact_change_ticks=d['tick'][changed].tolist(),
        light_contact_foot_samples=int(d['light'][first].sum()),
        establishing_foot_samples=int(d['establishing'][first].sum()),
        force_source='ground only', five_newton_gate_used=False,
        task_scoring_changed=False, student_observation_changed=False,
        full_feedback_retained=True, native_success_asserted=False, valid_expert_labels=False,
        source_sha256={str(x.resolve()):sha(x) for x in paths})
    temp=a.output.with_suffix('.tmp'); temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(a.output)
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}))


if __name__ == '__main__':
    main()
