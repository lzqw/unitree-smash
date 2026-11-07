#!/usr/bin/env python3
"""Check MPC lane memory isolation on recorded states, without new physics.

Public-state reconstruction is used identically in every ordering. This tests
controller memory isolation, not native prediction accuracy or recovery gains.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from quadsmash.rl.episode_trace_v4 import unpack_public
from scripts.landing_command_history_mpc_v1 import LandingCommandHistoryMPCV1
from scripts.landing_support_mpc_v2 import LandingMPCSettingsV2
from scripts.whole_body_support_qp_v1 import PositionPDInterface, SupportSettings


def compare(a, b):
    """Exact structure/discrete decisions; numeric maximum absolute error."""
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        return max((compare(a[k], b[k]) for k in a), default=0.)
    if isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        return max((compare(x, y) for x, y in zip(a, b)), default=0.)
    if isinstance(a, np.ndarray):
        assert a.shape == b.shape
        if a.dtype == bool:
            assert np.array_equal(a, b)
            return 0.
        assert np.isfinite(a).all() and np.isfinite(b).all()
        return float(np.max(abs(a-b))) if a.size else 0.
    if isinstance(a, (float, np.floating)):
        if not np.isfinite(a):
            assert a == b
            return 0.
        return abs(float(a)-float(b))
    assert a == b
    return 0.


def stripped(value):
    if isinstance(value, dict):
        return {k: stripped(v) for k, v in value.items()
                if k not in ('wall_runtime_s', 'runtime_ms')}
    if isinstance(value, list):
        return [stripped(v) for v in value]
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Preserve prior evidence')
    read = lambda n: json.loads((args.run/n).read_text())
    assert read('job.json')['status'] == 'SUCCEEDED'
    native = read('native_command_inputs.json')
    inputs = read('INPUTS.json')
    assert inputs['landing_mpc_command_history_interface']
    assert inputs['landing_mpc_continuous_reference']
    assert not any(inputs[k] for k in ('landing_mpc_free_arm',
        'landing_mpc_load_aware_support', 'landing_mpc_implicit_command_qp',
        'landing_mpc_interval_command_qp', 'landing_mpc_full_command_damper'))
    limits = np.asarray(native['joint_limits_rad'])
    default = np.asarray(native['default_joint_position_rad'])
    records = read('landing_mpc_records.json')
    # Two formerly successful cases, a rescued case, and the sole surviving
    # assisted success. Case selection is diagnostic only, never an online gate.
    cases = [10, 58, 61, 94]
    samples, source_files = {}, []
    for case in cases:
        rows = [r for r in records if r['case'] == case and r['episode_index'] == 0]
        start = next(i for i, r in enumerate(rows) if r.get('applied_first_step'))
        rows = rows[start:start+12]
        assert len(rows) == 12
        path = args.run/'fixed_policy_traces'/f'case_{case:03d}.npz'
        source_files.append(path)
        with np.load(path) as z:
            public = unpack_public(z['packed_public'])
        samples[case] = [(r, public[r['control_tick']]) for r in rows]

    def factory():
        interface = PositionPDInterface(limits,
            np.r_[np.full(12,25.),np.full(6,80.)],
            np.r_[np.full(12,.5),np.full(6,4.)],
            np.r_[np.full(12,23.5),np.full(6,100.)],
            np.r_[np.full(12,30.),np.full(6,5.)], .65)
        c = LandingCommandHistoryMPCV1(interface, arm_target_rate_rad_s=10.,
            settings=LandingMPCSettingsV2(horizon_steps=inputs['landing_mpc_horizon_steps'],
                minimum_support_feet=1, one_foot_max_com_distance_m=.20,
                landing_leg_posture_blend=inputs['landing_mpc_leg_posture_blend'],
                predict_touchdown_contacts=inputs['landing_mpc_predict_touchdown_contacts'],
                contact_mode_fallback=inputs['landing_mpc_contact_mode_fallback'],
                executor_replay=inputs['landing_mpc_executor_replay']),
            qp_settings=SupportSettings(solver_eps=1e-6, residual_tolerance=3e-4,
                swing_foot_weight=inputs['landing_mpc_swing_foot_weight'],
                swing_foot_kp_s2=inputs['landing_mpc_swing_foot_kp'],
                swing_foot_kd_s=inputs['landing_mpc_swing_foot_kd'],
                swing_foot_clearance_m=inputs['landing_mpc_swing_foot_clearance'],
                nonfoot_clearance_weight=inputs['landing_mpc_nonfoot_clearance_weight'],
                nonfoot_clearance_m=inputs['landing_mpc_nonfoot_clearance_m'],
                nonfoot_clearance_kp_s2=inputs['landing_mpc_nonfoot_clearance_kp'],
                nonfoot_clearance_kd_s=inputs['landing_mpc_nonfoot_clearance_kd']))
        c.set_posture_target(default)
        return c

    def execute(c, case, index):
        row, public = samples[case][index]
        state = c.qp.model.state_from_public(public, default)
        state.contacts = np.asarray(row['support_contacts'], dtype=bool)
        state.measured_force_world = np.asarray(row['measured_foot_force_world'])
        state.previous_command_target = np.asarray(row['previous_dispatched_target'])
        reference = np.asarray(row['requested_student_joint_reference'])
        tick = row['control_tick']
        c.observe(state, reference, lane=case, control_tick=tick)
        before = copy.deepcopy(c._live_reference[case])
        result = c.solve(state, reference, force=True, lane=case, control_tick=tick)
        assert compare(before, c._live_reference[case]) == 0.
        # A duplicate observe must not advance the critical-damping reference.
        c.observe(state, reference, lane=case, control_tick=tick)
        assert compare(before, c._live_reference[case]) == 0.
        return dict(plan=stripped(result), memory=copy.deepcopy(c._live_reference[case]))

    independent = {case: factory() for case in cases}
    expected = {}
    for index in range(12):
        if index == 6:
            independent[cases[0]].reset(cases[0])
        for case in cases:
            expected[(case,index)] = execute(independent[case],case,index)
    summaries = []
    for name in ['forward', 'reverse', 'alternating']:
        c = factory()
        maximum = 0.
        for index in range(12):
            if index == 6:
                c.reset(cases[0])
            order = list(reversed(cases)) if name == 'reverse' or (
                name == 'alternating' and index % 2) else cases
            for case in order:
                maximum = max(maximum, compare(expected[(case,index)], execute(c,case,index)))
        assert maximum <= 1e-10, (name, maximum)
        summaries.append(dict(order=name, comparisons=48, max_absolute_numeric_difference=maximum))
    source_files += [args.run/n for n in ('job.json', 'INPUTS.json', 'native_command_inputs.json',
                                         'landing_mpc_records.json')]
    source_files += [Path(__file__)] + [ROOT/'scripts'/n for n in (
        'landing_command_history_mpc_v1.py','landing_support_mpc_v3.py',
        'command_history_support_qp_v1.py','continuous_support_qp_v1.py',
        'landing_support_mpc_v2.py','landing_support_mpc_v1.py',
        'reduced_whole_body_support_qp_v1.py','whole_body_support_qp_v1.py')]
    result = dict(passed=True, scope=__doc__, cases=cases, states=48,
        solver_calls=192, comparisons=summaries, controller_contract=factory().contract(),
        single_lane_reset_at_sample=6,
        speculative_memory_rollback_checked=True, repeated_tick_idempotence_checked=True,
        new_native_execution=False, recovered_cases_proven=0,
        source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'source_sha256'}))


if __name__ == '__main__':
    main()
