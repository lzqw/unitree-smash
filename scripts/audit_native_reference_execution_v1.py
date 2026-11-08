"""Compare declared model-reference selection with completed native execution.

Separates missing reference coverage, predictive command rejection and actual
tracking divergence. All statistics are retrospective; no case-ID control.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['run', 'audit', 'library', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    read = lambda p: json.loads(Path(p).read_text())
    if args.output.exists() or read(args.run/'job.json')['status'] != 'SUCCEEDED':
        raise ValueError('Fresh audit of completed native run required')
    refs = {r['source_key']: r for r in read(args.library)['references']}
    evidence = {r['case']: r for r in read(args.audit)['cases']}
    with np.load(args.run/'pure_control_commands.npz') as archive:
        commands = {k: archive[k].copy() for k in archive.files}
    diagnostics = [json.loads(line) for line in (args.run/'pure_control_diagnostics.jsonl').read_text().splitlines()]
    if len(diagnostics) != len(commands['lane']):
        raise ValueError('Command and diagnostic lengths differ')
    model = LoadedPinocchioModelV1()
    output = []
    for lane in sorted(set(commands['lane'].tolist())):
        ids = np.flatnonzero(commands['lane'] == lane)
        first = diagnostics[int(ids[0])]['diagnostic']
        source = refs.get(first.get('offline_reference_source_key'))
        trace = []
        rejected = []
        clock_mismatches = []
        for i in ids:
            i = int(i)
            row = diagnostics[i]
            if row['lane'] != lane:
                raise ValueError('Command/diagnostic lane mismatch')
            diag = row['diagnostic']
            elapsed = float(commands['time_s'][i]-commands['time_s'][ids[0]])
            k = int(commands['tick'][i]-commands['tick'][ids[0]])
            if source is not None:
                observed = diag.get('reference', {}).get('reference_index') if diag.get('reference_active') else None
                expected = k if k < len(source['trajectory']) else None
                if observed != expected:
                    clock_mismatches.append(dict(control_tick_since_release=k, elapsed_s=elapsed,
                        observed_reference_index=observed, expected_reference_index=expected))
            if diag.get('no_feasible_predictive_candidate'):
                rejected.append(dict(elapsed_s=elapsed, control_tick=int(commands['tick'][i]),
                    candidates=diag.get('predictive_candidates', []), executed_source=diag['source']))
            if source is None or k >= len(source['trajectory']):
                continue
            desired = source['query'] if k == 0 else source['trajectory'][k-1]
            actual_q = commands['q'][i]
            desired_q = np.asarray(desired['q'])
            dot = float(np.dot(actual_q[3:7], desired_q[3:7]))
            trace.append(dict(elapsed_s=elapsed, command_source=diag['source'],
                root_position_error_m=float(np.linalg.norm(actual_q[:3]-desired_q[:3])),
                root_rotation_error_rad=float(2*np.arccos(np.clip(abs(dot), 0., 1.))),
                maximum_joint_error_rad=float(np.max(abs(actual_q[model.q_indices]-desired_q[model.q_indices]))),
                maximum_joint_speed_error_rad_s=float(np.max(abs(commands['v'][i,model.v_indices]-np.asarray(desired['v'])[model.v_indices])))))
        physical = evidence[int(lane)]
        output.append(dict(case=int(lane), reference_source_case=None if source is None else source['source_case'],
            initial_state_match_score=first.get('initial_state_match_score'),
            reference_samples=len(trace),
            reference_clock_mismatch_steps=len(clock_mismatches),
            first_reference_clock_mismatch=clock_mismatches[0] if clock_mismatches else None,
            command_sources=dict(Counter(diagnostics[int(i)]['diagnostic']['source'] for i in ids)),
            reference_command_sources=dict(Counter(r['command_source'] for r in trace)),
            first_no_feasible_predictive_candidate=rejected[0] if rejected else None,
            no_feasible_predictive_candidate_steps=len(rejected),
            reference_tracking_samples=[r for j,r in enumerate(trace) if j%10 == 0 or j==len(trace)-1],
            max_reference_root_error_m=max((r['root_position_error_m'] for r in trace),default=None),
            max_reference_joint_error_rad=max((r['maximum_joint_error_rad'] for r in trace),default=None),
            head_after_takeover=physical['head_after_takeover'],
            other_nonfoot_after_takeover=physical['other_nonfoot_after_takeover'],
            no_recorded_nonfoot_ground_force=physical['no_recorded_nonfoot_ground_force'],
            final_sampled_stable_s=physical['final_control_sample_stable_s']))
    summary = dict(owned_cases=len(output), matched_reference_cases=sum(r['reference_source_case'] is not None for r in output),
        same_source_matches=sum(r['reference_source_case']==r['case'] for r in output),
        different_source_matches=sum(r['reference_source_case'] is not None and r['reference_source_case']!=r['case'] for r in output),
        reference_with_predictive_rejections=sum(r['no_feasible_predictive_candidate_steps']>0 for r in output),
        actual_reference_clock_mismatch_steps=sum(r['reference_clock_mismatch_steps'] for r in output),
        clean_and_final_sampled_stable_08s=[r['case'] for r in output if r['no_recorded_nonfoot_ground_force'] and r['final_sampled_stable_s']>=.8],
        strict_acceptance_not_automatically_inferred=True)
    sources = [Path(__file__), args.library, args.audit, args.run/'pure_control_commands.npz',
               args.run/'pure_control_diagnostics.jsonl', args.run/'job.json']
    result = dict(summary=summary,cases=output,
        source_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
