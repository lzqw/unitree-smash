#!/usr/bin/env python3
"""Plan Pullback +30..+45ms Delayed-Intercept probes for physically unreachable high-z / +y top-rim failures.

Physical discovery:
  On hard Panel B failures (iz >= 1.84m or iy >= +0.14m), the baseline policy intercepts too early
  at x ~ 1.15m where ball-to-root distance is 1.31-1.35m (> 1.18m arm reach -> guaranteed top-rim FRAME_HIT).
  Waiting 28-45ms (pulling the racket back in -x to ~0.88-0.96m) lets the shuttle drop 100-140mm
  while the jumping torso rises +20mm, shrinking ball-to-root distance to 1.08-1.12m (7-20mm from sweet spot!).
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from plan_jacobian_probes import clip_action_vec, solve_dls
from quadsmash.rl.supermimic_physical_dataset_v1 import decode_physical
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2

ASSET_SHA = 'c2264fcece90b042c7c6a1cac7d516af83972729d0402947e3a519041e184cf2'
ALPHA_J = np.array([0.32, 0.44, 0.48, 0.35, 0.52, 0.60], dtype=np.float64)


def solve_delayed_pullback_ik(geom, rid, state_row, *, dt_s: float, target_nyz=(-0.03, -0.06),
                              z_bias_m: float = 0.0, dq_limit: float = 0.62, steps: int = 75):
    """Exact 3D FK IK placing racket center at predicted ball(t_c + dt_s) using ballistic root+ball motion."""
    d0 = decode_physical(state_row.float().unsqueeze(0))
    q0 = d0['joint_position'][0, 12:18].clone()
    bp0 = d0['ball_position_w'][0].float()
    bv0 = d0['ball_velocity_w'][0].float()
    rp0 = d0['root_position_w'][0].float()
    rv0 = d0['root_link_velocity_w'][0].float()
    rq0 = d0['root_quaternion_w']
    # Account for shuttle aerodynamic drag (~k_d * |v| * v, k_d ~ 0.22 m^-1) over 30-45ms + gravity!
    bp_k = bp0.clone()
    bv_k = bv0.clone()
    n_sub = max(1, int(round(dt_s / 0.005)))
    h = dt_s / n_sub
    grav = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float32)
    for _ in range(n_sub):
        spd = bv_k.norm().clamp_min(1e-4)
        acc = grav - 0.22 * spd * bv_k
        bp_k = bp_k + h * bv_k + 0.5 * h * h * acc
        bv_k = bv_k + h * acc
    bp_k[2] = bp_k[2] + z_bias_m
    rp_k = rp0 + rv0 * dt_s + 0.5 * grav * (dt_s * dt_s)

    tnyz = torch.tensor(target_nyz, dtype=torch.float32)
    dq = torch.zeros(6, requires_grad=True)
    opt = torch.optim.Adam([dq], lr=0.07)
    for _ in range(steps):
        opt.zero_grad()
        q = d0['joint_position'].clone()
        q[0, 12:18] = q0 + dq
        fk = geom.forward(q, d0['joint_velocity'], rp_k[None], rq0,
                          d0['root_link_velocity_w'], d0['root_angular_velocity_w'])
        tp = fk['position'][0, rid].float()
        R_r = fk['rotation'][0, rid].float()
        n = R_r[:, 0]
        n_pos = n if float(n[0]) >= 0 else -n
        # Penalize 3D position error strongly (guarantees pullback in -x AND sweet spot in y/z!)
        loss = (260.0 * (tp - bp_k).square().sum()
                + 8.0 * (n_pos[1:3] - tnyz).square().sum()
                + 0.10 * dq.square().sum())
        loss.backward()
        opt.step()
        with torch.no_grad():
            dq.clamp_(-dq_limit, dq_limit)

    # Compute forward-whip Jacobian at the NEW delayed pose q0 + dq!
    q_sol = (q0 + dq.detach()).clone()

    def fn_p(qa):
        q = d0['joint_position'].clone()
        q[0, 12:18] = qa
        fk = geom.forward(q, d0['joint_velocity'], rp_k[None], rq0,
                          d0['root_link_velocity_w'], d0['root_angular_velocity_w'])
        return fk['position'][0, rid].float()

    def fn_n(qa):
        q = d0['joint_position'].clone()
        q[0, 12:18] = qa
        fk = geom.forward(q, d0['joint_velocity'], rp_k[None], rq0,
                          d0['root_link_velocity_w'], d0['root_angular_velocity_w'])
        return fk['rotation'][0, rid, :, 0].float()

    Jp = torch.autograd.functional.jacobian(fn_p, q_sol).numpy().astype(np.float64)
    Jn = torch.autograd.functional.jacobian(fn_n, q_sol).numpy().astype(np.float64)
    with torch.no_grad():
        q = d0['joint_position'].clone()
        q[0, 12:18] = q_sol
        fk = geom.forward(q, d0['joint_velocity'], rp_k[None], rq0,
                          d0['root_link_velocity_w'], d0['root_angular_velocity_w'])
        n_sol = fk['rotation'][0, rid, :, 0].numpy().astype(np.float64)
        if n_sol[0] < 0:
            n_sol = -n_sol
        err_mm = float((fk['position'][0, rid].float() - bp_k).norm() * 1000.0)

    da_fwd = solve_dls(np.vstack([Jp, 0.30 * Jn]), np.concatenate([0.10 * n_sol, np.zeros(3)]), reg=0.025) / 0.4
    da_fwd = da_fwd / max(float(np.abs(da_fwd).max()), 1e-6)
    da_pos = dq.detach().numpy().astype(np.float64) / ALPHA_J
    return da_pos, da_fwd, err_mm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nominal-dirs', type=Path, nargs='+', required=True)
    parser.add_argument('--nominal-specs', type=Path, nargs='+', required=True)
    parser.add_argument('--solved-probe-dirs', type=Path, nargs='*', default=[])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-cases', type=int, default=64)
    args = parser.parse_args()

    solved = set()
    for pd in args.solved_probe_dirs:
        if (pd / 'probe_summary.json').exists():
            for r in json.loads((pd / 'probe_summary.json').read_text())['rows']:
                if r['success'] and r['string'] and not r['robot_net'] and not r['low_speed']:
                    solved.add(r['case_id'])

    # Gather all unsolved failures across nominal dirs, prioritizing high iz and wide +y!
    cands = []
    tapes = {}
    for ndir, nspec_p in zip(args.nominal_dirs, args.nominal_specs, strict=True):
        ps = json.loads((ndir / 'probe_summary.json').read_text())['rows']
        sp = json.loads(nspec_p.read_text())['cases']
        for i, (r, c) in enumerate(zip(ps, sp, strict=True)):
            cid = r['case_id']
            if r['success'] or cid in solved:
                continue
            ip = c['prediction_only']['intercept_position_m']
            iz, iy = float(ip[2]), float(ip[1])
            kind = 'FRAME' if (r['frame'] and not r['string']) else ('LOW_SPD' if r['low_speed'] else ('BNET' if r['ball_net'] else ('OOB' if r['string'] else 'MISS')))
            # Priority: unsolved FRAME/MISS on high-iz / wide-+y first, then other unsolved failures
            is_pb = 0 if cid.startswith('B-') else 1
            is_fr = 0 if kind in ('FRAME', 'MISS') else 1
            hard_score = -(iz - 1.80) * 3.0 - max(0.0, iy) * 1.5 - abs(iy) * 0.5
            cands.append((is_pb, is_fr, hard_score, str(ndir), i, c, kind))

    cands.sort()
    selected = cands[:args.max_cases]
    by_k = {}
    for x in selected:
        by_k[x[6]] = by_k.get(x[6], 0) + 1
    hz84 = sum(1 for x in selected if x[5]['prediction_only']['intercept_position_m'][2] >= 1.84)
    py14 = sum(1 for x in selected if x[5]['prediction_only']['intercept_position_m'][1] >= 0.14)
    print(f'Selected {len(selected)}/{len(cands)} unsolved failures (iz>=1.84: {hz84}, iy>=+0.14: {py14}, by_kind={by_k})')

    geom = PhysicalWorldV2(ASSET_SHA).geometry
    rid = geom.ids['Racket']

    out_cases, out_offsets, out_ramps, out_peaks, out_widths, branch_meta = [], [], [], [], [], []
    errs = []
    for _, _, _, ndir_s, lane, c_obj, kind in selected:
        if ndir_s not in tapes:
            blob = torch.load(Path(ndir_s) / 'probe_tape.pt', map_location='cpu', weights_only=True)
            tapes[ndir_s] = (blob['tape'], int(blob['start_capture_tick']))
        tape, start_tick = tapes[ndir_s]
        ep0 = tape['episode'][:, lane] == 0
        st = tape['state'][ep0, lane].float()
        hc = tape['had_contact'][ep0, lane].numpy()
        pres = tape['present'][ep0, lane].numpy()
        if hc.any():
            rel_idx = int(np.nonzero(hc)[0][0])
            c_idx = max(0, rel_idx - 1)
            t_c = start_tick + rel_idx
        else:
            d_all = decode_physical(st)
            with torch.no_grad():
                fk = geom.forward(d_all['joint_position'], d_all['joint_velocity'],
                                  d_all['root_position_w'], d_all['root_quaternion_w'],
                                  d_all['root_link_velocity_w'], d_all['root_angular_velocity_w'])
                dists = np.linalg.norm(d_all['ball_position_w'].numpy() - fk['position'][:, rid].numpy(), axis=-1)
                dists[~pres] = 9.9
                c_idx = int(np.argmin(dists))
                t_c = start_tick + c_idx

        # 4 Pullback Delayed-Intercept branches (+26ms, +34ms, +42ms, +34ms high-speed)
        cfgs = [
            ('pb_26ms_flat',  0.026, (-0.03, -0.05),  0.000, 0.18, 46.0, 0.52),
            ('pb_34ms_flat',  0.034, (-0.03, -0.07),  0.000, 0.22, 50.0, 0.56),
            ('pb_42ms_smash', 0.042, (-0.03, -0.09),  0.000, 0.26, 54.0, 0.58),
            ('pb_34ms_spd',   0.034, (-0.03, -0.03), -0.015, 0.32, 50.0, 0.56),
        ]
        for b_i, (tag, dt_s, nyz, zb, spd_g, wt, off_lim) in enumerate(cfgs):
            da_pos, da_fwd, err_mm = solve_delayed_pullback_ik(
                geom, rid, st[c_idx], dt_s=dt_s, target_nyz=nyz, z_bias_m=zb)
            errs.append(err_mm)
            pt = float(t_c + int(round(dt_s * 100.0)))
            # Subtract slight forward ramp component from position offset so total offset+ramp lands on sweet spot at pt!
            off18 = clip_action_vec(da_pos - 0.35 * spd_g * da_fwd, off_lim)
            rmp18 = clip_action_vec(spd_g * da_fwd, min(0.36, spd_g + 0.04))
            out_cases.append(c_obj)
            out_offsets.append(off18)
            out_ramps.append(rmp18)
            out_peaks.append(pt)
            out_widths.append(wt)
            branch_meta.append(dict(
                case_id=c_obj['case_id'], failure_kind=kind, branch=b_i, tag=tag,
                peak_tick=int(round(pt)), ik_err_mm=round(err_mm, 1),
            ))

    print(f'Pullback Delayed-Intercept IK 3D error: med={np.median(errs):.1f}mm, q75={np.quantile(errs, 0.75):.1f}mm')
    spec = dict(
        mode='pullback_delayed_probe',
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
    print(f'Wrote {len(out_cases)}-lane pullback delayed-intercept spec -> {args.output}')


if __name__ == '__main__':
    main()
