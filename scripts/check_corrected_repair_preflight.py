#!/usr/bin/env python3
"""CPU-only replay of a recorded command boundary; never launches Isaac."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pinocchio as pin

WORKTREE = Path(__file__).resolve().parents[1]
# Match the already declared evaluator's import resolution, without changing it.
MAIN = Path('/home/autodrive/LZQW/unitree_smash_contact_recovery')
sys.path.insert(0, str(WORKTREE))
sys.path.insert(0, str(MAIN))
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.loaded_pinocchio_model_v1 import PublicState
from scripts.pure_control_state_driven_long_mpc_v1_qp_continuity_repair import PureControlStateDrivenLongMPCV1QPContinuityRepair as Candidate
from scripts.pure_control_state_driven_long_mpc_v1_qp_continuity import PureControlStateDrivenLongMPCV1QPContinuity as Baseline
from scripts.event_recovery_manager_v1 import RecoveryCommandArbiterV1


def main():
    outdir = WORKTREE / 'reports/execution_consistent_repair_preflight_20261110'
    outdir.mkdir(parents=True, exist_ok=True)
    root = MAIN / 'reports/pure_control_state_driven_long_mpc_qp_continuity_all128_20261109_01/native'
    config = json.loads((WORKTREE / 'reports/pure_control_state_driven_long_mpc_qp_continuity_repair_native_preparation_20261110_01/controller.json').read_text())
    nominal = np.array(json.loads((root / 'native_command_inputs.json').read_text())['default_joint_position_rad'])
    rows = [json.loads(line) for line in (root / 'pure_control_diagnostics.jsonl').open()]
    with np.load(root / 'pure_control_commands.npz') as data:
        index = int(np.flatnonzero((data['lane'] == 71) & (data['tick'] == 264))[0])
        record = {key: data[key][index].copy() for key in data.files}
    row = rows[index]
    candidate = Candidate(PositionPDInterface(**config['interface']), nominal)
    baseline = Baseline(PositionPDInterface(**config['interface']), nominal)
    q, v = record['q'], record['v']
    rotation = pin.Quaternion(q[3:7]).matrix()
    state = PublicState(q.copy(), v.copy(), rotation, q[:3].copy(),
                        q[candidate.model.q_indices].copy(), v[candidate.model.v_indices].copy(),
                        np.array(row['phase']['ground_contacts']), 0.)
    state.previous_command_target = record['previous'].copy()
    state.measured_force_world = record['ground_force'].copy()
    plans = []
    solve = candidate.solve

    def logged_solve(current, reference, **kwargs):
        plan = solve(current, reference, **kwargs)
        plans.append(plan)
        return plan

    candidate.solve = logged_solve
    requested, diagnostic = candidate.command(copy.deepcopy(state), row['phase']['phase'], float(record['time_s']))
    baseline_requested, _ = baseline.command(copy.deepcopy(state), row['phase']['phase'], float(record['time_s']))
    final, _ = RecoveryCommandArbiterV1(candidate.interface.limits).select(
        None, requested, state.previous_command_target, owns_recovery=True)
    terminal = nominal.copy()
    terminal[12:] = candidate._base_reference(state)[12:]
    qp = np.asarray(plans[0]['joint_position_target'])
    first = 31/32 * qp + 1/32 * terminal
    requested_final_delta = float(np.max(abs(final-requested)))
    predicted_final = np.asarray(diagnostic['execution_preview']['predicted_final_target'])
    mapping_delta = float(np.max(abs(final-predicted_final)))

    previous = candidate.interface.previous_command.copy()
    phase, elapsed = candidate.phase, candidate.elapsed
    cursor, tick = candidate._plan_cursor, candidate._tick
    worker1 = candidate._make_repair_worker()
    worker2 = candidate._make_repair_worker()
    independent = (worker1.interface is not candidate.interface
                   and worker1.model.data is not candidate.model.data
                   and worker1.geometry is not candidate.geometry
                   and worker1.interface is not worker2.interface
                   and worker1.model.data is not worker2.model.data)
    worker1.interface.bind_previous_command(nominal)
    worker1.phase = 'CPU_TEST_ONLY'
    worker1.elapsed += 1.
    unchanged = (np.array_equal(candidate.interface.previous_command, previous)
                 and (candidate.phase, candidate.elapsed, candidate._plan_cursor, candidate._tick)
                 == (phase, elapsed, cursor, tick))
    source = WORKTREE / 'scripts/pure_control_state_driven_long_mpc_v1_qp_continuity_repair.py'
    result = dict(
        scope='CPU-only replay of saved TRAIN71 tick264; case selected for diagnostics, not runtime control',
        native_evaluation_started=False,
        candidate_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        normal_path_matches_baseline=bool(np.array_equal(requested, baseline_requested)),
        original_plan_passed=bool(diagnostic['plan']['accepted']),
        repair_used=bool(diagnostic['plan'].get('sequence_repaired')),
        workers_independent=independent, live_history_and_clock_unchanged=unchanged,
        original_qp_target=qp.tolist(), original_rollout_first_target=first.tolist(),
        corrected_requested_target=requested.tolist(), final_mapped_target=final.tolist(),
        predicted_final_target=predicted_final.tolist(),
        previous_actual_target=state.previous_command_target.tolist(),
        requested_vs_original_first_delta_rad=float(np.max(abs(requested-first))),
        requested_vs_final_delta_rad=requested_final_delta,
        predicted_vs_final_delta_rad=mapping_delta,
        prediction_execution_consistent=mapping_delta <= 2e-6,
        overall_pass=independent and unchanged and mapping_delta <= 2e-6,
        failure_reason='Existing arbiter clips arm target to previous +/- 0.1rad after prediction' if mapping_delta > 2e-6 else None)
    (outdir / 'preflight_result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if not isinstance(v,list)}, indent=2))
    return 0 if result['overall_pass'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
