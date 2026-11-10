#!/usr/bin/env python3
"""Read saved native evidence only; never runs a controller or simulator."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.audit_sequence_repair_run import (
    case_details, finite, load_run, sha, task_sets,
)
from scripts.preview_recovery_execution_v1 import preview_recovery_execution


def command_audit(run, limits):
    rows = run['rows']
    with np.load(run['root'] / 'pure_control_commands.npz') as z:
        c = {k: z[k] for k in z.files}
    with np.load(run['root'] / 'sequence_dispatch.npz') as z:
        normalized = z['final'][c['tick'], c['lane']]
    physical = limits[:, 0] + .5 * (normalized + 1) * np.diff(limits).ravel()
    errors = []
    missing = []
    first = []
    previous_errors = []
    branch_errors = []
    branch_history = {}
    last = {}
    for i, row in enumerate(rows):
        assert int(c['tick'][i]) == row['tick'] and int(c['lane'][i]) == row['lane']
        d = row['diagnostic']
        preview = d.get('execution_preview')
        if preview is None:
            missing.append(i)
        else:
            errors.append(float(np.max(abs(np.asarray(preview['predicted_final_target']) - c['final'][i]))))
            # Preview must use the real previous command and preserve the request.
            assert np.array_equal(preview['previous_final_target'], c['previous'][i])
            assert np.array_equal(preview['requested_target'], c['requested'][i])
        key = (int(c['lane'][i]), int(c['episode'][i]))
        if key in last and c['tick'][i] == c['tick'][last[key]] + 1:
            previous_errors.append(float(np.max(abs(c['previous'][i] - c['final'][last[key]]))))
        last[key] = i
        # Reconstruct the used portion of each accepted plan with its own
        # predicted command history. Do not refresh from actual history on
        # cached-plan ticks: this also detects accumulated float32 drift.
        if d.get('replanned') or not d['solver_accepted'] or key not in branch_history:
            branch_history[key] = c['previous'][i].copy()
        mapped, _ = preview_recovery_execution(limits, c['requested'][i], branch_history[key])
        branch_errors.append(float(np.max(abs(mapped-c['final'][i]))))
        branch_history[key] = mapped.copy()
        p = d.get('plan', {})
        if d.get('replanned') and p.get('sequence_repaired'):
            first.append(dict(lane=row['lane'], tick=row['tick'],
                request_delta_rad=p.get('first_request_max_delta_rad'),
                final_delta_rad=p.get('first_final_max_delta_rad')))
    actual = float(np.max(abs(physical - c['final'])))
    preview_max = max(errors, default=None)
    history_max = max(previous_errors, default=None)
    branch_max = max(branch_errors, default=None)
    first_pass = all(x['request_delta_rad'] == 0 and x['final_delta_rad'] == 0 for x in first)
    return dict(
        checked_rows=len(rows), missing_preview_rows=missing,
        max_request_vs_final_delta_rad=float(np.max(abs(c['requested']-c['final']))),
        limit_active_rows=int(np.sum(np.max(abs(c['requested']-c['final']),axis=1)>2e-6)),
        max_prediction_preview_vs_native_arbiter_delta_rad=preview_max,
        max_final_vs_recorded_dispatch_roundtrip_delta_rad=actual,
        max_actual_history_vs_previous_final_delta_rad=history_max,
        max_reconstructed_plan_branch_final_vs_actual_delta_rad=branch_max,
        unchanged_precision_tolerance_rad=2e-6,
        repaired_first_targets=first,
        repair_first_target_checks_pass=first_pass,
        all_pass=(not missing and preview_max is not None and preview_max <= 2e-6
                  and actual <= 2e-6 and history_max is not None and history_max <= 2e-6
                  and branch_max is not None and branch_max <= 2e-6
                  and first_pass),
        physical_execution_check='Unchanged verify_applied checked post-step actuator targets at <=2e-6 throughout native run; no overwrite exception.',
        limitation='Command mapping agreement does not prove predicted mechanical response equals PhysX response.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--historical57', type=Path, required=True)
    p.add_argument('--comparison', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    candidate = load_run(args.candidate)
    baseline = load_run(args.baseline)
    historical = load_run(args.historical57)
    comp = json.loads(args.comparison.read_text())
    config_path = Path(__file__).resolve().parents[1] / 'reports/pure_control_state_driven_long_mpc_qp_continuity_repair_native_preparation_20261110_01/controller.json'
    limits = np.array(json.loads(config_path.read_text())['interface']['limits'])
    check = command_audit(candidate, limits)
    before = task_sets(historical)
    after = task_sets(candidate)
    focus = {20, 23, 26, 83} | (set(before['complete5']) ^ set(after['complete5']))
    history_compare = {k:dict(retained=sorted(set(before[k]) & set(after[k])),
                            lost=sorted(set(before[k])-set(after[k])),
                            gained=sorted(set(after[k])-set(before[k]))) for k in before}
    a = json.loads((args.baseline/'INPUTS.json').read_text())
    b = json.loads((args.candidate/'INPUTS.json').read_text())
    same = {k:a[k] == b.get(k) for k in a if k != 'pure_control_configuration'}
    prefix = comp['prefix']
    prefix_summary = {}
    for key in ['takeover_tick_equal','contact_time_s_equal','release_time_s_equal',
                'first_string_time_s_equal','outgoing_velocity_mps_equal']:
        prefix_summary[key] = dict(equal=sum(x[key] for x in prefix.values()), total=128,
            different=[int(i) for i,x in prefix.items() if not x[key]])
    for key in ['prefix_max_command_delta','takeover_q_max_delta','takeover_v_max_delta',
                'takeover_previous_max_delta']:
        prefix_summary[key] = max(x.get(key,0.) for x in prefix.values())
    result = dict(
        label='Sequence repair and execution-interface-consistent prediction; combined correction, not single-factor ablation',
        official_baseline='9cb7983: 52/74', historical57_role='Diagnostic only; original source and result contain known defects',
        command_consistency=check,
        input_contract_equal_except_controller=same,
        changed_input_fields=[k for k,v in same.items() if not v],
        prefix_summary=prefix_summary,
        historical57_comparison=history_compare,
        historical57_task_sets=before,
        focus_vs_historical57={'historical57':case_details(historical,focus),
                               'candidate':case_details(candidate,focus)},
        baseline_vs_candidate=comp['seven_metrics'],
        native_evaluations_this_authorization=1,
        additional_seed_or_baseline_run=False,
        complete_evidence_sha256={f:sha(args.candidate/f) for f in
             ['completed.json','metrics.json','episodes.json','pure_control_commands.npz',
              'pure_control_diagnostics.jsonl','sequence_dispatch.npz',
              'pure_control_ground_forces.npz','INPUTS.json']})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(finite(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(command_all_pass=check['all_pass'],
                         changed_input_fields=result['changed_input_fields'],
                         history57_recovery=history_compare['recovery']),indent=2))


if __name__ == '__main__':
    main()
