#!/usr/bin/env python3
"""Locate constraint-family conflicts on previously rejected recorded states.

Diagnostic LP family removal is never an executable controller. Separately
test a free-arm QP retaining every physical and command constraint. No native
actions, physics changes, or success claims are produced by this script.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy import sparse
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quadsmash.rl.episode_trace_v4 import unpack_public
from scripts.landing_active_set_interval_mpc_v1 import LandingActiveSetIntervalMPCV1
from scripts.landing_support_mpc_v2 import LandingMPCSettingsV2
from scripts.whole_body_support_qp_v1 import PositionPDInterface, SupportSettings
from scripts.dense_active_set_qp_v1 import DenseActiveSetQP
import scripts.interval_command_support_qp_v1 as interval_module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def feasibility(matrix, mask):
    A, low, high = matrix['A'][mask], matrix['l'][mask], matrix['u'][mask]
    eq = np.isfinite(low) & np.isfinite(high) & (low == high)
    upper, lower = np.isfinite(high) & ~eq, np.isfinite(low) & ~eq
    result = linprog(np.zeros(A.shape[1]),
        A_ub=sparse.vstack((A[upper], -A[lower])), b_ub=np.r_[high[upper], -low[lower]],
        A_eq=A[eq], b_eq=high[eq], bounds=[(None, None)] * A.shape[1], method='highs')
    report = dict(status=int(result.status), message=result.message)
    if result.success:
        response = A @ result.x
        report['scaled_violation'] = float(max(0., np.max(low-response), np.max(response-high)))
        if report['scaled_violation'] > 1e-6:
            raise ValueError('LP feasibility has excessive independent residual')
    elif result.status != 2:
        raise ValueError('LP did not establish feasible or infeasible: ' + result.message)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--convergence-audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Preserve prior evidence')
    read = lambda name: json.loads((a.run/name).read_text())
    if read('job.json')['status'] != 'SUCCEEDED':
        raise ValueError('Completed native source required')
    prior = json.loads(a.convergence_audit.read_text())
    # Historical code can differ; recorded observations and command history
    # must match the earlier convergence audit exactly.
    paths = []
    for name, expected in prior['source_sha256'].items():
        path = Path(name)
        if path.resolve().is_relative_to(a.run.resolve()):
            if sha(path) != expected:
                raise ValueError('Recorded source changed: ' + name)
            paths.append(path)
    selected = {(row['case'], row['tick']) for row in prior['records']
                if row['linear_feasibility']['status'] == 2}
    native = read('native_command_inputs.json')
    default = np.asarray(native['default_joint_position_rad'])
    interface = PositionPDInterface(np.asarray(native['joint_limits_rad']),
        np.r_[np.full(12,25.),np.full(6,80.)], np.r_[np.full(12,.5),np.full(6,4.)],
        np.r_[np.full(12,23.5),np.full(6,100.)], np.r_[np.full(12,30.),np.full(6,5.)], .65)
    cfg = SupportSettings(solver_eps=1e-6,residual_tolerance=3e-4,
        swing_foot_weight=1.,swing_foot_kp_s2=60.,swing_foot_kd_s=12.,swing_foot_clearance_m=.005,
        nonfoot_clearance_weight=5.,nonfoot_clearance_m=.05,nonfoot_clearance_kp_s2=60.,nonfoot_clearance_kd_s=15.)
    settings = LandingMPCSettingsV2(horizon_steps=40,minimum_support_feet=1,
        one_foot_max_com_distance_m=.20,landing_leg_posture_blend=.15,
        landing_arm_posture_blend=0.,executor_replay=True,predict_touchdown_contacts=True,contact_mode_fallback=True)
    mpc = LandingActiveSetIntervalMPCV1(interface,qp_settings=cfg,settings=settings,arm_target_rate_rad_s=10.)
    mpc.set_posture_target(default)
    logs = read('landing_mpc_records.json'); public = {}
    for case in sorted({row['case'] for row in logs}):
        path = a.run/'fixed_policy_traces'/f'case_{case:03d}.npz'
        with np.load(path) as z:
            public[case] = unpack_public(z['packed_public'])
        paths.append(path)
    captured = {}; labels = []
    original_constraints = interval_module.interval_constraints

    def capture_constraints(*args, **kwargs):
        result = original_constraints(*args, **kwargs)
        labels[:] = [kind.rsplit('_',1)[0] if kind.startswith(('normal_force_', 'friction_'))
                     else kind for stage,kind,index in result['labels']]
        return result

    class CaptureSolver(DenseActiveSetQP):
        def setup(self, **kwargs):
            captured.clear()
            captured.update({k:kwargs[k].copy() for k in ('P','q','A','l','u')})
            return super().setup(**kwargs)

    mpc.qp._make_solver = CaptureSolver
    interval_module.interval_constraints = capture_constraints
    output = []
    try:
        for row in logs:
            case,tick = row['case'],row['control_tick']
            if row['episode_index'] != 0:
                raise ValueError('Only first real episodes are supported')
            state = mpc.qp.model.state_from_public(public[case][tick],default)
            state.contacts = np.asarray(row['support_contacts'],dtype=bool)
            state.measured_force_world = np.asarray(row['measured_foot_force_world'])
            state.previous_command_target = np.asarray(row['previous_dispatched_target'])
            reference = mpc.admit_command_target(state,np.asarray(row['requested_student_joint_reference']))
            mpc.observe(state,reference,lane=case,control_tick=tick)
            if (case,tick) not in selected:
                continue
            before = mpc._prediction_reference[case]
            effective = mpc._posture_reference(state,reference)
            mpc.qp.restore_reference_state(before)
            baseline = mpc.qp.solve(state,effective)
            matrix = captured.copy()
            family = np.asarray(labels + ['native_target_bounds']*18 + ['preserve_arm']*6 + ['root_height']*16)
            if len(family) != len(matrix['l']):
                raise ValueError('Constraint-family mapping no longer matches formulation')
            all_rows = np.ones(len(family),dtype=bool)
            original = feasibility(matrix,all_rows)
            if original['status'] != 2 or baseline['accepted']:
                raise ValueError('Prior infeasibility did not reproduce')
            groups = sorted(set(family))
            without = {str(group):feasibility(matrix,family != group) for group in groups}
            # Deterministic inclusion-minimal conflict in groups, not a unique
            # row-level IIS and not a physical impossibility proof.
            conflict = list(groups)
            for group in groups:
                trial = [x for x in conflict if x != group]
                if trial and feasibility(matrix,np.isin(family,trial))['status'] == 2:
                    conflict = trial
            mpc.qp.restore_reference_state(before)
            mpc.qp.preserve_arm_command = False
            try:
                free = mpc.qp.solve(state,effective)
            finally:
                mpc.qp.preserve_arm_command = True
            result = dict(case=case,tick=tick,old_native_applied=row['applied_first_step'],
                contacts=state.contacts.tolist(),root_height_m=float(state.q[2]),
                measured_arm_speed_rad_s=state.joint_velocity[12:].tolist(),
                joint_position_rad=state.joint_position.tolist(),
                joint_velocity_rad_s=state.joint_velocity.tolist(),
                joint_order=list(mpc.qp.model.joint_order),
                joint_limits_rad=interface.limits.tolist(),
                baseline_status=baseline.get('solver_status'),linear_feasibility=original,
                diagnostic_family_removal=without,inclusion_minimal_conflicting_families=conflict,
                free_arm={k:free.get(k) for k in ['accepted','solver_status','rejection','residuals']})
            if free.get('accepted'):
                result['free_arm']['arm_change_from_reference_max_rad'] = float(np.max(abs(free['joint_position_target'][12:]-effective[12:])))
                result['free_arm']['actual_command_not_executed'] = True
            output.append(result)
            print(json.dumps(result),flush=True)
    finally:
        interval_module.interval_constraints = original_constraints
    if len(output) != len(selected):
        raise ValueError('Missing selected states')
    paths += [Path(__file__),a.convergence_audit] + [ROOT/'scripts'/name for name in
        ['interval_command_support_qp_v1.py','condensed_implicit_drive_v1.py','landing_active_set_interval_mpc_v1.py',
         'landing_interval_command_mpc_v1.py','dense_active_set_qp_v1.py','implicit_support_prediction_v1.py']]
    summary = dict(samples=len(output),free_arm_physically_verified=sum(x['free_arm']['accepted'] for x in output),
        diagnostic_family_removal_feasible={group:sum(x['diagnostic_family_removal'][group]['status']==0 for x in output)
            for group in output[0]['diagnostic_family_removal']})
    report = dict(scope=__doc__,native_execution=False,source=str(a.run.resolve()),summary=summary,records=output,
        conflict_scope='One inclusion-minimal conflict among constraint families; frozen contact and unsaturated drive model only.',
        source_sha256={str(path.resolve()):sha(path) for path in set(paths)})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
