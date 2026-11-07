#!/usr/bin/env python3
"""Actual applied-command diagnostic: frozen versus refreshed local dynamics.

Both predictions retain the observed initial contact set for 10 ms. Refreshing
M, h, J and Jdot is not a collision, friction/slip or contact-transition model.
No controller or native physical parameter is changed by this CPU audit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pinocchio as pin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.implicit_support_prediction_v1 import implicit_velocity_step, predict_implicit_support
from scripts.landing_command_history_mpc_v1 import LandingCommandHistoryMPCV1
from scripts.landing_support_mpc_v2 import LandingMPCSettingsV2
from scripts.whole_body_support_qp_v1 import PositionPDInterface, SupportSettings
from quadsmash.rl.episode_trace_v4 import unpack_public


def refreshed(model, interface, state, target):
    q, v = state.q.copy(), state.v.copy()
    feet = np.flatnonzero(state.contacts)
    selection = np.zeros((24, 18)); selection[model.v_indices, np.arange(18)] = 1.
    dt = .01 / 16
    violations = dict(normal_force_n=0., friction_n=0., velocity_rad_s=0., dynamics_residual=0.)
    for _ in range(16):
        snap = model.snapshot(q, v)
        jac = snap['foot_jacobian'][feet].reshape(-1, 24)
        jdot_v = (snap['foot_jacobian_dot'][feet] @ v).reshape(-1)
        step = implicit_velocity_step(snap['mass_matrix'], v, snap['nonlinear'], selection,
            interface.kp, interface.kd, target-q[model.q_indices], interface.effort,
            dt, jac, dt*(-jac@v/.06-jdot_v))
        v = step['velocity']; q = pin.integrate(model.model, q, dt*v)
        force = step['force'].reshape(-1, 3)
        violations['dynamics_residual'] = max(violations['dynamics_residual'], step['residual'])
        violations['velocity_rad_s'] = max(violations['velocity_rad_s'], float(np.max(abs(v[model.v_indices])-interface.velocity)))
        if len(feet):
            violations['normal_force_n'] = max(violations['normal_force_n'], float(np.max(-force[:, 2])))
            violations['friction_n'] = max(violations['friction_n'], float(np.max(np.linalg.norm(force[:, :2], axis=1)-interface.friction*force[:, 2])))
    return q, v, violations


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise ValueError('Preserve previous evidence')
    read = lambda name: json.loads((a.run/name).read_text())
    if read('job.json')['status'] != 'SUCCEEDED': raise ValueError('Completed native input required')
    native = read('native_command_inputs.json'); default = np.asarray(native['default_joint_position_rad'])
    interface = PositionPDInterface(np.asarray(native['joint_limits_rad']),
        np.r_[np.full(12,25.),np.full(6,80.)],np.r_[np.full(12,.5),np.full(6,4.)],
        np.r_[np.full(12,23.5),np.full(6,100.)],np.r_[np.full(12,30.),np.full(6,5.)],.65)
    mpc = LandingCommandHistoryMPCV1(interface, qp_settings=SupportSettings(),
        settings=LandingMPCSettingsV2(executor_replay=True), arm_target_rate_rad_s=10.)
    model = mpc.qp.model; cache = {}; rows = []; sources = [Path(__file__)]
    for r in read('landing_mpc_records.json'):
        if r.get('episode_index') != 0 or r['control_tick'] % 20 or not r['applied_first_step']: continue
        case, tick = r['case'], r['control_tick']
        if case not in cache:
            path = a.run/'fixed_policy_traces'/f'case_{case:03d}.npz'; sources.append(path)
            with np.load(path, allow_pickle=False) as z: cache[case] = unpack_public(z['packed_public'])
        public = cache[case]
        if tick+1 >= len(public): continue
        if abs(float(public[tick+1,775]-public[tick,775])-.01)>2e-6: raise ValueError('Not adjacent control ticks')
        state = model.state_from_public(public[tick], default)
        actual = model.state_from_public(public[tick+1], default)
        state.contacts = np.asarray(r['support_contacts'], dtype=bool)
        target = np.asarray(r['final_dispatched_joint_target_rad'])
        row = dict(case=case, tick=tick,
            force_flag_changed=not np.array_equal(public[tick,57:61]>.5,public[tick+1,57:61]>.5), models={})
        for name in ['frozen', 'refreshed']:
            try:
                if name == 'frozen':
                    out = predict_implicit_support(model, interface, state, target, substeps=16)
                    q, v, diagnostics = out['state'].q, out['state'].v, out['diagnostics']
                else: q, v, diagnostics = refreshed(model, interface, state, target)
                row['models'][name] = dict(ok=True,
                    leg_position_rad=float(np.max(abs(q[model.q_indices[:12]]-actual.joint_position[:12]))),
                    arm_position_rad=float(np.max(abs(q[model.q_indices[12:]]-actual.joint_position[12:]))),
                    base_linear_velocity_mps=float(np.linalg.norm(v[:3]-actual.v[:3])),
                    base_angular_velocity_rad_s=float(np.linalg.norm(v[3:6]-actual.v[3:6])), diagnostics=diagnostics)
            except (ValueError, np.linalg.LinAlgError) as error:
                row['models'][name] = dict(ok=False, error=repr(error))
        rows.append(row)
    summary = {}
    for name in ['frozen', 'refreshed']:
        valid = [r['models'][name] for r in rows if r['models'][name]['ok']]
        summary[name] = dict(successful=len(valid), failed=len(rows)-len(valid), p95={
            k:float(np.quantile([v[k] for v in valid], .95)) for k in
            ['leg_position_rad','arm_position_rad','base_linear_velocity_mps','base_angular_velocity_rad_s']})
    sources += [a.run/n for n in ['job.json','native_command_inputs.json','landing_mpc_records.json']]
    sources += [ROOT/'scripts'/n for n in ['implicit_support_prediction_v1.py','loaded_pinocchio_model_v1.py']]
    result = dict(scope=__doc__, source=str(a.run), native_new_execution=False, samples=len(rows),
        summary=summary, records=rows, source_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(samples=len(rows), summary=summary)))


if __name__ == '__main__': main()
