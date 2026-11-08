"""Locate measured landing failures relative to reference and support transitions.

Read-only analysis of completed native episodes. Positive sampled clearance or
an instantaneous stance trigger is never classified as successful landing.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def rotation(quaternion):
    x, y, z, w = quaternion / np.linalg.norm(quaternion)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['run', 'audit', 'library', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--trace-cases', nargs='+', type=int,
        default=[12, 30, 50, 58, 61, 64, 118, 123])
    args = parser.parse_args()
    read = lambda p: json.loads(p.read_text())
    if args.output.exists() or read(args.run/'job.json')['status'] != 'SUCCEEDED':
        raise ValueError('Fresh output and completed native run required')
    audit = read(args.audit)
    if not audit['passed_dispatch_integrity']:
        raise ValueError('Verified final command ownership required')
    evidence = {r['case']: r for r in audit['cases']}
    meta = read(args.run/'pure_control_ground_forces.json')
    if not meta['complete'] or not meta['interval_aggregates_complete']:
        raise ValueError('Complete native ground-force recording required')
    refs = {r['source_key']: r for r in read(args.library)['references']}
    with np.load(args.run/'pure_control_commands.npz') as z:
        commands = {key: z[key].copy() for key in ['tick', 'lane', 'q', 'v', 'ground_force']}
    with np.load(args.run/'pure_control_ground_forces.npz') as z:
        first_frames = z['interval_first_frame_over1n'].copy()
    rows = [json.loads(line) for line in (args.run/'pure_control_diagnostics.jsonl').read_text().splitlines()]
    if len(rows) != len(commands['lane']):
        raise ValueError('Command and diagnostic lengths differ')
    for i, row in enumerate(rows):
        if (row['tick'], row['lane']) != (commands['tick'][i], commands['lane'][i]):
            raise ValueError('Misaligned native observations')
    head = [meta['body_names'].index(n) for n in ['Head_lower', 'Head_upper']]
    other = [i for i,n in enumerate(meta['body_names']) if not n.endswith('_foot') and i not in head]
    cases = []
    for lane in sorted(set(commands['lane'].tolist())):
        ids = np.flatnonzero(commands['lane'] == lane)
        start, end = int(commands['tick'][ids[0]]), int(commands['tick'][ids[-1]])+1
        source = refs.get(rows[int(ids[0])]['diagnostic'].get('offline_reference_source_key'))
        ref_end = start + len(source['trajectory']) if source is not None else None

        def sample(i):
            i = int(i)
            row = rows[i]
            diag, phase = row['diagnostic'], row['phase']
            r = rotation(commands['q'][i, 3:7])
            return dict(tick=int(commands['tick'][i]), time_s=int(commands['tick'][i])*.01,
                phase=phase['phase'], command_source=diag['source'],
                reference_active=diag.get('reference_active', False),
                terminal_stance_active=diag.get('terminal_stance_active', False),
                terminal_candidate_conditions=diag.get('terminal_candidate_conditions', {}),
                root_height_m=float(commands['q'][i, 2]),
                roll_pitch_rad=[float(np.arctan2(r[2,1], r[2,2])), float(np.arcsin(np.clip(-r[2,0],-1,1)))],
                root_velocity_world_m_s=(r @ commands['v'][i,:3]).tolist(),
                angular_speed_rad_s=float(np.linalg.norm(commands['v'][i,3:6])),
                contact_mask=phase['ground_contacts'],
                measured_foot_force_world_n=commands['ground_force'][i].tolist(),
                sampled_stable_s=phase['control_sample_stable_duration_s'])

        def first_contact(indices):
            frames = first_frames[start:end,lane][:,indices]
            hits = frames[frames >= 0]
            if not len(hits):
                return None
            frame = int(hits.min())
            at = np.argwhere(frames == frame)[0]
            tick = start + int(at[0])
            j = np.searchsorted(commands['tick'][ids], tick, side='right')-1
            return dict(body=meta['body_names'][indices[int(at[1])]], physics_frame=frame,
                time_s=frame*meta['physics_rate_aggregation_s'],
                seconds_after_reference_end=(frame*meta['physics_rate_aggregation_s']-ref_end*.01 if ref_end is not None else None),
                preceding_control_observation=sample(ids[max(0,j)]))

        transitions = []
        previous = None
        stance = None
        for i in ids:
            row = rows[int(i)]
            phase = row['phase']['phase']
            if phase != previous:
                transitions.append(dict(tick=row['tick'], phase=phase))
                previous = phase
            if stance is None and row['diagnostic'].get('terminal_stance_active'):
                stance = sample(i)
        physical = evidence[int(lane)]
        tail = [i for i in ids if ref_end is not None and commands['tick'][i] >= ref_end]
        cases.append(dict(case=int(lane), takeover_tick=start, last_control_tick=end-1,
            reference_source_case=source['source_case'] if source else None,
            nominal_reference_end_tick=ref_end, first_head_after_takeover=first_contact(head),
            first_other_nonfoot_after_takeover=first_contact(other),
            reference_end_observation=sample(tail[0]) if tail else None,
            terminal_stance_entry=stance, phase_transitions=transitions,
            after_reference_command_sources=dict(Counter(rows[int(i)]['diagnostic']['source'] for i in tail)),
            no_recorded_nonfoot_ground_force=physical['no_recorded_nonfoot_ground_force'],
            longest_sampled_stable_s=physical['longest_control_sample_stable_s'],
            final_sampled_stable_s=physical['final_control_sample_stable_s'],
            trace=[sample(i) for j,i in enumerate(ids) if j%25 == 0 or j==len(ids)-1]
                if lane in args.trace_cases else []))
    summary = dict(owned_cases=len(cases), historical_complete5=audit['complete5'],
        first_head_over1n_cases=sum(r['first_head_after_takeover'] is not None for r in cases),
        first_other_over1n_cases=sum(r['first_other_nonfoot_after_takeover'] is not None for r in cases),
        terminal_stance_entered_cases=[r['case'] for r in cases if r['terminal_stance_entry'] is not None],
        clean_and_final_sampled_stable_08s_through_tick799=[r['case'] for r in cases
            if r['no_recorded_nonfoot_ground_force'] and r['final_sampled_stable_s'] >= .8
            and r['last_control_tick'] == 799],
        sampled_stability_is_not_formal_acceptance=True, training=False, valid_expert_labels=False)
    paths = [Path(__file__), args.audit, args.library, args.run/'job.json',
        args.run/'pure_control_commands.npz', args.run/'pure_control_diagnostics.jsonl',
        args.run/'pure_control_ground_forces.npz', args.run/'pure_control_ground_forces.json']
    output = dict(summary=summary, cases=cases,
        contact_threshold_note='1 N dates diagnostic events; any recorded positive nonfoot ground force prevents the clean flag.',
        source_sha256={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix('.tmp')
    temp.write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    temp.replace(args.output)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
