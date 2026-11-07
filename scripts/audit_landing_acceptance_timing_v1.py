#!/usr/bin/env python3
"""Compare model acceptance with recorded nonfoot-event chronology.

Offline diagnostic only. An accepted receding-horizon plan is not a certificate
of task success; later replanning means even an event inside its horizon is not
by itself a matched open-loop prediction error.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def audit(source):
    names = ['episodes.json', 'landing_mpc_records.json',
             'first_nonfoot_contact_details.json', 'contact_telemetry_metadata.json',
             'contact_telemetry_summary.json', 'fixed_policy_traces/manifest.json']
    data = {n: json.loads((source / n).read_text()) for n in names}
    episodes = data['episodes.json']
    if len(episodes) != 128 or any(e['source_training_index'] != i or
            e['env_id'] != i or e['episode_index'] != 0 for i, e in enumerate(episodes)):
        raise ValueError('Require original ordered TRAIN128 first episodes')
    dt = float(data['contact_telemetry_metadata.json']['physics_dt_s'])
    if abs(data['contact_telemetry_summary.json']['last_physics_frame'] * dt - 8.) > 1e-8:
        raise ValueError('Require ground-start eight-second physics-frame clock')
    lengths = {e['env_id']: e['samples'] for e in
               data['fixed_policy_traces/manifest.json']['episodes']}
    first = {}
    last_frame = -1
    for event in data['first_nonfoot_contact_details.json']:
        frame = event['physics_frame']
        if frame <= last_frame:
            raise ValueError('Nonmonotonic event archive')
        last_frame = frame
        sensors = event['sensors']
        if len(sensors) != 3:
            raise ValueError('Require original three nonfoot sensor groups')
        selected = []
        for lane in range(128):
            magnitude = sum(math.sqrt(sum(x*x for x in force))
                for s in sensors for body in s['force_matrix_world_n'][lane]
                for force in body)
            if lane not in first and frame * dt < lengths[lane] * .01 and magnitude > 1.:
                selected.append(lane)
        if selected != event['first_event_env_ids']:
            raise ValueError('Raw forces do not reproduce first-event selection')
        for lane in selected:
            first[lane] = frame * dt
    cases = []
    records = data['landing_mpc_records.json']
    for lane in sorted({r['case'] for r in records}):
        rows = [r for r in records if r['case'] == lane]
        if any(r['episode_index'] != 0 or abs(r['clock_s'] - .01*r['control_tick']) > 1e-6
               for r in rows):
            raise ValueError('Unsupported episode clock')
        accepted = [r for r in rows if r['accepted']]
        event_time = first.get(lane)
        before = [r for r in accepted if event_time is not None and
                  .01*r['control_tick'] < event_time]
        after = [r for r in accepted if event_time is not None and
                 .01*r['control_tick'] >= event_time]
        recovery = episodes[lane]['contact_flight_recovery']
        if bool(event_time is not None) != bool(recovery['body_landing']):
            raise ValueError('First-event evidence and task body-contact flag disagree')
        cases.append(dict(case=lane, records=len(rows), accepted=len(accepted),
            accepted_applied=sum(bool(r['applied_first_step']) for r in accepted),
            first_accepted_s=.01*accepted[0]['control_tick'] if accepted else None,
            last_accepted_s=.01*accepted[-1]['control_tick'] if accepted else None,
            first_observed_nonfoot_s=event_time,
            accepted_before_nonfoot=len(before), accepted_after_nonfoot=len(after),
            accepted_without_nonfoot_event=len(accepted) if event_time is None else 0,
            accepted_before_event_within_400ms=sum(event_time-.01*r['control_tick'] <= .4
                                                   for r in before),
            recovery_success=recovery['success'], stable_window_s=recovery['recovery_s'],
            terminal_root_height_m=recovery['terminal_root_height_m'],
            terminated=episodes[lane]['jump']['terminated']))
    totals = {k: sum(c[k] for c in cases) for k in ['records', 'accepted', 'accepted_applied',
        'accepted_before_nonfoot', 'accepted_after_nonfoot', 'accepted_without_nonfoot_event',
        'accepted_before_event_within_400ms']}
    assert totals['accepted'] == sum(totals[k] for k in
        ['accepted_before_nonfoot', 'accepted_after_nonfoot', 'accepted_without_nonfoot_event'])
    return dict(source=str(source.resolve()), source_sha256={n: hashlib.sha256(
        (source/n).read_bytes()).hexdigest() for n in names}, totals=totals, cases=cases,
        scope=__doc__, new_native_execution=False,
        event_semantics='First sampled original nonfoot force sum >1 N; not exact collision onset',
        first_event_validation='Raw sensor groups reproduce all first-event selections',
        historical_failure_can_be_undone=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Preserve existing evidence')
    result = audit(a.source)
    result['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result['totals']))


if __name__ == '__main__':
    main()
