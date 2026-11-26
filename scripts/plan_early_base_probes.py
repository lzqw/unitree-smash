#!/usr/bin/env python3
"""Plan Pre-Takeoff Jump-Boost + Early-Arm (550ms) probes on remaining unsolved failures.

Tests four physical mechanisms inaccessible to the frozen-base approach_gate (230ms):
  Branch 0: Pure pre-takeoff jump boost (thigh -0.10, calf +0.10 at pushoff) + 0 arm
  Branch 1: Strong jump boost (thigh -0.16, calf +0.16) + 550ms early arm IK
  Branch 2: Pure 550ms early arm wind-up (no arm_gate wait) + per-joint calibrated FK IK + whip
  Branch 3: Jump boost (-0.13/+0.13) + 550ms early arm IK + stronger whip (+0.24)
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

from plan_jacobian_probes import compute_arm_jacobians, solve_dls, solve_nonlinear_ik, solve_dynamic_intercept_ik, quat_wxyz_to_rot
from quadsmash.rl.supermimic_physical_dataset_v1 import decode_physical
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2

# Empirically calibrated per-joint PhysX PD tracking gains (dq_rad / da_offset)
ALPHA_JOINT = np.array([0.26, 0.45, 0.50, 0.34, 0.55, 0.65], dtype=np.float64)


def clip_arm(da6, limit=0.52):
    m = float(np.abs(da6).max())
    if m > limit:
        da6 = da6 * (limit / m)
    out = [0.0] * 18
    out[12:18] = da6.tolist()
    return out


def make_leg_jump(boost: float, lat_hip: float = 0.0):
    vec = [0.0] * 18
    # Go2 pushoff: thigh negative extends leg, calf positive extends leg
    vec[0] = lat_hip; vec[1] = lat_hip; vec[2] = lat_hip; vec[3] = lat_hip
    vec[4] = -boost;  vec[5] = -boost;  vec[6] = -boost;  vec[7] = -boost
    vec[8] = +boost;  vec[9] = +boost;  vec[10] = +boost; vec[11] = +boost
    return vec


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
        if (pdir / 'probe_summary.json').is_file():
            for r in json.loads((pdir / 'probe_summary.json').read_text())['rows']:
                if r['success'] and not r['robot_net'] and not r['low_speed']:
                    solved.add(r['case_id'])

    unsolved = []
    marginal_top = []
    for ndir, nspec_p in zip(args.nominal_dirs, args.nominal_specs, strict=True):
        ns = json.loads((ndir / 'probe_summary.json').read_text())
        nspec = json.loads(nspec_p.read_text())
        cms = {c['env_id']: c for c in reversed(json.loads((ndir / 'native_contact_motion.json').read_text()))}
        for i, (row, c_obj) in enumerate(zip(ns['rows'], nspec['cases'], strict=True)):
            cid = row['case_id']
            cm = cms.get(i)
            loc_z = 0.08
            if cm and all(k in cm for k in ('point_m', 'racket_center_m', 'racket_quaternion_w')):
                R = quat_wxyz_to_rot(cm['racket_quaternion_w'])
                loc_z = float((R.T @ (np.array(cm['point_m']) - np.array(cm['racket_center_m'])))[2])
            if not row['success'] and cid not in solved:
                prio = 0 if row['frame'] else (1 if row['low_speed'] else 2)
                unsolved.append((prio, -loc_z, ndir, i, row, c_obj, loc_z))
            elif row['success'] and loc_z > 0.065:
                marginal_top.append((3, -loc_z, ndir, i, row, c_obj, loc_z))

    unsolved.sort(key=lambda x: (x[0], x[1]))
    marginal_top.sort(key=lambda x: x[1])
    selected = (unsolved + marginal_top)[:args.max_cases]
    print(f'Selected {len(selected)} cases ({min(len(unsolved), args.max_cases)} unsolved + {max(0, len(selected)-len(unsolved))} high-rim marginals)')

    pw = PhysicalWorldV2('c2264fcece90b042c7c6a1cac7d516af83972729d0402947e3a519041e184cf2')
    geom = pw.geometry
    rid = geom.ids['Racket']
    tape_cache = {}

    out_cases, out_offs, out_rmps, out_pts, out_wts = [], [], [], [], []
    out_loffs, out_lpts, out_lwts, branch_meta = [], [], [], []

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

        # Find exact pushoff tick (vz first >= 0.8 m/s)
        dec_all = decode_physical(st)
        vz = dec_all['root_link_velocity_w'][:, 2].numpy()
        po_idxs = np.nonzero(vz >= 0.8)[0]
        tk_tick = float(start_tick + (int(po_idxs[0]) + 1 if len(po_idxs) else max(35, c_idx - 28)))

        v_in = np.array(c_obj['prediction_only']['incoming_velocity_mps'], dtype=np.float64)
        iy = float(c_obj['prediction_only']['intercept_position_m'][1])
        Jp, Jn, p_r, R_r = compute_arm_jacobians(geom, rid, st[c_idx])
        normal = R_r[:, 0].copy()
        if np.dot(normal, v_in) > 0:
            normal = -normal
        da_fwd = solve_dls(np.vstack([Jp, 0.25 * Jn]), np.concatenate([0.10 * normal, np.zeros(3)]), reg=0.02) / ALPHA_JOINT
        da_fwd = da_fwd / max(float(np.abs(da_fwd).max()), 1e-6)

        # Solve nonlinear FK IK calibrated with exact per-joint PhysX PD gains!
        da_s_raw, dt_s = solve_nonlinear_ik(
            geom, rid, st[c_idx], v_in, target_yz=(0.0, 0.0), target_nyz=(-0.03, 0.01),
            w_yz=140.0, w_n=5.0, w_dq=0.16, steps=35, dq_limit=0.30)
        da_s = (da_s_raw * 0.4) / ALPHA_JOINT

        da_d_raw, dt_d = solve_dynamic_intercept_ik(
            geom, rid, st[c_idx], v_in, target_yz=(0.0, 0.0), target_nyz=(-0.03, -0.02),
            dt_init=0.020, dt_bounds=(0.005, 0.042), qdot_scale=0.35, dq_limit=0.28)
        da_d = (da_d_raw * 0.4) / ALPHA_JOINT

        # If high-rim (loc_z > 0.05) or miss, jump higher; if low ball (loc_z < 0.0), gentler jump boost
        jb0 = 0.11 if loc_z > 0.04 else 0.05
        jb1 = 0.16 if loc_z > 0.04 else 0.08
        jb3 = 0.13 if loc_z > 0.04 else 0.06
        lat = float(np.clip(-0.08 * iy / 0.15, -0.08, 0.08))  # subtle lateral hip lean

        branches = [
            ([0.0]*18, [0.0]*18, peak_tick, 65.0, make_leg_jump(jb0, 0.0), tk_tick, 12.0, 'jump_only_mild'),
            (clip_arm(0.85 * da_s, 0.48), clip_arm(0.14 * da_fwd, 0.20), peak_tick + dt_s, 72.0, make_leg_jump(jb1, 0.0), tk_tick, 14.0, 'jump_strong+early_arm'),
            (clip_arm(1.00 * da_s, 0.54), clip_arm(0.20 * da_fwd, 0.26), peak_tick + dt_s, 78.0, [0.0]*18, tk_tick, 12.0, 'early_arm_550ms'),
            (clip_arm(0.95 * da_d, 0.54), clip_arm(0.24 * da_fwd, 0.30), peak_tick + dt_d, 76.0, make_leg_jump(jb3, lat), tk_tick, 14.0, 'jump+dyn_early_whip'),
        ]
        kind = 'FRAME' if row['frame'] else ('LOW_SPD' if row['low_speed'] else ('SUCC_RIM' if row['success'] else 'OTHER'))
        for b_idx, (off, rmp, pt, wt, loff, lpt, lwt, tag) in enumerate(branches):
            out_cases.append(c_obj)
            out_offs.append(off); out_rmps.append(rmp); out_pts.append(float(pt)); out_wts.append(float(wt))
            out_loffs.append(loff); out_lpts.append(float(lpt)); out_lwts.append(float(lwt))
            branch_meta.append(dict(case_id=row['case_id'], failure_kind=kind, loc_z_mm=round(loc_z*1000, 1),
                                    branch=b_idx, tag=tag, peak_tick=int(pt), leg_peak_tick=int(lpt)))

    spec = dict(
        mode='early_base_probe',
        cases=out_cases,
        offsets=out_offs,
        ramp_offsets=out_rmps,
        peak_ticks=out_pts,
        width_ticks=out_wts,
        leg_offsets=out_loffs,
        leg_peak_ticks=out_lpts,
        leg_width_ticks=out_lwts,
        branch_meta=branch_meta,
        gate_mode='present_precontact',
        record_tape=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(spec, indent=2) + '\n')
    print(f'Wrote {len(out_cases)}-lane Pre-Takeoff + Early-Arm spec -> {args.output}')


if __name__ == '__main__':
    main()
