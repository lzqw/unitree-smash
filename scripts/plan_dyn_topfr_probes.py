#!/usr/bin/env python3
"""Plan Dynamic Delayed-Intercept probes for unsolved FRAME_HIT / NO_CONTACT cases.

Exploits the exact airborne kinematics discovered in forensics: on top-rim FRAME_HITs
(loc_z > +85mm), the arm is near vertical reach saturation (+40mm), whereas intercepting
20-35ms later closes loc_z by ~110mm (ball drops -3.2m/s while rising torso gains +0.48m/s).
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from plan_jacobian_probes import (
    compute_arm_jacobians, solve_dls, solve_dynamic_intercept_ik, clip_action_vec, quat_wxyz_to_rot
)
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nominal-dirs', type=Path, nargs='+', required=True)
    parser.add_argument('--nominal-specs', type=Path, nargs='+', required=True)
    parser.add_argument('--solved-probe-dirs', type=Path, nargs='+', default=[])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-cases', type=int, default=64)
    args = parser.parse_args()

    solved = set()
    for pdir in args.solved_probe_dirs:
        if not (pdir / 'probe_summary.json').is_file():
            continue
        ps = json.loads((pdir / 'probe_summary.json').read_text())
        for r in ps['rows']:
            if r['success'] and not r['robot_net']:
                solved.add(r['case_id'])

    candidates = []
    for ndir, nspec_p in zip(args.nominal_dirs, args.nominal_specs, strict=True):
        ns = json.loads((ndir / 'probe_summary.json').read_text())
        nspec = json.loads(nspec_p.read_text())
        cms = {}
        for c in json.loads((ndir / 'native_contact_motion.json').read_text()):
            if c['env_id'] not in cms:
                cms[c['env_id']] = c
        for i, (row, c_obj) in enumerate(zip(ns['rows'], nspec['cases'], strict=True)):
            cid = row['case_id']
            if row['success'] or cid in solved:
                continue
            if not (row['frame'] or not row['string']):
                continue
            cm = cms.get(i)
            loc_z = 0.10
            if cm is not None and all(k in cm for k in ('point_m', 'racket_center_m', 'racket_quaternion_w')):
                R = quat_wxyz_to_rot(cm['racket_quaternion_w'])
                loc_z = float((R.T @ (np.array(cm['point_m']) - np.array(cm['racket_center_m'])))[2])
            prio = 0 if (row['frame'] and loc_z > 0.060) else (1 if row['frame'] else 2)
            candidates.append((prio, -abs(loc_z), ndir, i, row, c_obj, loc_z))

    candidates.sort(key=lambda x: (x[0], x[1]))
    selected = candidates[:args.max_cases]
    print(f'Selected {len(selected)}/{len(candidates)} unsolved FRAME_HIT/NO_CONTACT cases for Dynamic Delayed-Intercept probing')

    pw = PhysicalWorldV2('c2264fcece90b042c7c6a1cac7d516af83972729d0402947e3a519041e184cf2')
    geom = pw.geometry
    rid = geom.ids['Racket']

    tape_cache = {}
    out_cases, out_offsets, out_ramps, out_peaks, out_widths, branch_meta = [], [], [], [], [], []
    for prio, _, ndir, i, row, c_obj, loc_z in selected:
        key = str(ndir)
        if key not in tape_cache:
            blob = torch.load(ndir / 'probe_tape.pt', map_location='cpu', weights_only=True)
            tape_cache[key] = (blob['tape'], int(blob['start_capture_tick']))
        tape, start_tick = tape_cache[key]
        ep0 = tape['episode'][:, i] == 0
        st = tape['state'][ep0, i].float()
        hc = tape['had_contact'][ep0, i].numpy()
        c_idx = max(0, int(np.nonzero(hc)[0][0]) - 1) if hc.any() else len(st) // 2
        peak_tick = start_tick + c_idx
        v_in = np.array(c_obj['prediction_only']['incoming_velocity_mps'], dtype=np.float64)

        Jp, Jn, p_r, R_r = compute_arm_jacobians(geom, rid, st[c_idx])
        normal = R_r[:, 0].copy()
        if np.dot(normal, v_in) > 0:
            normal = -normal
        da_fwd = solve_dls(np.vstack([Jp, 0.25 * Jn]), np.concatenate([0.10 * normal, np.zeros(3)]), reg=0.02) / 0.4
        da_fwd = da_fwd / max(float(np.abs(da_fwd).max()), 1e-6)

        # 4 Dynamic Delayed-Intercept branches spanning dt=+18..+38ms & normal pitch/speed trade-off!
        da1, dt1 = solve_dynamic_intercept_ik(
            geom, rid, st[c_idx], v_in, target_yz=(0.0, 0.0), target_nyz=(-0.03, 0.04),
            dt_init=0.020, dt_bounds=(0.008, 0.042), qdot_scale=0.30, dq_limit=0.26)
        da2, dt2 = solve_dynamic_intercept_ik(
            geom, rid, st[c_idx], v_in, target_yz=(0.0, -0.010), target_nyz=(-0.03, 0.00),
            dt_init=0.026, dt_bounds=(0.012, 0.048), qdot_scale=0.45, dq_limit=0.30)
        da3, dt3 = solve_dynamic_intercept_ik(
            geom, rid, st[c_idx], v_in, target_yz=(0.0, -0.015), target_nyz=(-0.03, -0.04),
            dt_init=0.032, dt_bounds=(0.016, 0.055), qdot_scale=0.55, dq_limit=0.32)
        da4, dt4 = solve_dynamic_intercept_ik(
            geom, rid, st[c_idx], v_in, target_yz=(0.0, 0.005), target_nyz=(-0.02, 0.06),
            dt_init=0.024, dt_bounds=(0.010, 0.050), qdot_scale=0.00, dq_limit=0.28)

        branches = [
            (clip_action_vec(1.00 * da1, 0.56), [0.0]*18, peak_tick + dt1, 58, 'dyn_delay1_0rmp'),
            (clip_action_vec(1.00 * da2, 0.60), clip_action_vec(0.18 * da_fwd, 0.24), peak_tick + dt2, 60, 'dyn_delay2_spd0.18'),
            (clip_action_vec(1.05 * da3, 0.64), clip_action_vec(0.24 * da_fwd, 0.28), peak_tick + dt3, 62, 'dyn_delay3_spd0.24'),
            (clip_action_vec(1.00 * da4, 0.58), clip_action_vec(0.14 * da_fwd, 0.20), peak_tick + dt4, 58, 'dyn_delay4_q0'),
        ]
        kind = 'TOP_FR' if prio == 0 else ('OTHER_FR' if prio == 1 else 'NO_CONTACT')
        for b_idx, (off, rmp, pt, wt, tag) in enumerate(branches):
            out_cases.append(c_obj)
            out_offsets.append(off)
            out_ramps.append(rmp)
            out_peaks.append(float(pt))
            out_widths.append(float(wt))
            branch_meta.append(dict(
                case_id=row['case_id'], failure_kind=kind, loc_z_mm=round(loc_z * 1000, 1),
                branch=b_idx, tag=tag, peak_tick=int(pt),
            ))

    spec = dict(
        mode='dyn_delayed_intercept',
        cases=out_cases,
        offsets=out_offsets,
        ramp_offsets=out_ramps,
        peak_ticks=out_peaks,
        width_ticks=out_widths,
        branch_meta=branch_meta,
        gate_mode='approach_gate',
        record_tape=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(spec, indent=2) + '\n')
    print(f'Wrote {len(out_cases)}-lane Dynamic Delayed-Intercept spec -> {args.output}')


if __name__ == '__main__':
    main()
