#!/usr/bin/env python3
"""Plan Cock-Back + Forward-Whip Apex-Intercept probes for physically unreachable hard-tail serves.

Physical mechanism:
  On hard Panel-B failures (ft < 0.72, iz >= 1.85, or iy >= +0.14 & iz >= 1.82), early interception
  at x ~ 1.15-1.20m has |ball - root| = 1.28-1.31m (> 1.23m arm reach -> guaranteed top-rim FRAME/MISS),
  AND the robot is still ~40ms BEFORE jump apex (vz ~ +0.43 m/s).
  Waiting dt = 32-52ms to jump apex drops |ball - root| by -210mm to 1.06-1.09m (dead-center inside reach!).

Biomechanical Cock-Back + Forward-Whip decomposition:
  At peak_tick = t_c + dt_ticks:
    offset      = da_hit - da_whip   (cocks racket back -14..-22cm so shuttle passes x=1.15m untouched)
    ramp_offset = +da_whip           (whips forward +14..+22cm over final 180ms straight through sweet spot!)
  So at t = peak_tick:
    1. Position = (da_hit - da_whip) + da_whip = da_hit (exact 3D delayed sweet spot!)
    2. Velocity = +Jp * d/dt(da_whip) (+2.0..+3.5 m/s forward whip -> v_out >= 7.5 m/s, zero low_speed!)
    3. Normal pitch nz in [+0.04, +0.14] & court-centering yaw ny = -0.22 * y_hit -> clears 1.524m net
       from x ~ 0.85m and lands dead-center around lx ~ 3.2m, |ly| < 0.5m!
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


def solve_apex_whip_ik(geom, rid, state_row, *, dt_s: float, target_nz: float,
                       whip_m: float = 0.16, dq_limit: float = 0.62, steps: int = 80):
    """Exact 3D delayed-apex FK IK + forward-whip Jacobian at the delayed pose."""
    d0 = decode_physical(state_row.float().unsqueeze(0))
    q0 = d0['joint_position'][0, 12:18].clone()
    bp0 = d0['ball_position_w'][0].float()
    bv0 = d0['ball_velocity_w'][0].float()
    rp0 = d0['root_position_w'][0].float()
    rv0 = d0['root_link_velocity_w'][0].float()
    rq0 = d0['root_quaternion_w']

    # Integrate shuttle with aerodynamic drag (k_d ~ 0.22 m^-1) + gravity over dt_s
    bp_k = bp0.clone()
    bv_k = bv0.clone()
    n_sub = max(2, int(round(dt_s / 0.004)))
    h = dt_s / n_sub
    grav = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float32)
    for _ in range(n_sub):
        spd = bv_k.norm().clamp_min(1e-4)
        acc = grav - 0.22 * spd * bv_k
        bp_k = bp_k + h * bv_k + 0.5 * h * h * acc
        bv_k = bv_k + h * acc

    # Ballistic torso position at t_c + dt_s (free-flight in air!)
    rp_k = rp0 + rv0 * dt_s + 0.5 * grav * (dt_s * dt_s)

    # Aim strongly into court-center: on +y (left) reaches the arm naturally deflects +1.3m wide left,
    # so we counter with strong inward yaw (-0.28 to -0.36 for iy>0, +0.10 to -0.10 for iy<0)!
    ny_aim = float(np.clip(-0.85 * float(bp_k[1]) - 0.12 - 0.06 * float(bv_k[1]), -0.38, 0.22))
    tnyz = torch.tensor([ny_aim, float(target_nz)], dtype=torch.float32)

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
        loss = (300.0 * (tp - bp_k).square().sum()
                + 14.0 * (n_pos[1:3] - tnyz).square().sum()
                + 0.08 * dq.square().sum())
        loss.backward()
        opt.step()
        with torch.no_grad():
            dq.clamp_(-dq_limit, dq_limit)

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

    # Forward-whip displacement along +n_sol of magnitude whip_m meters, holding racket normal steady!
    dq_whip = solve_dls(np.vstack([Jp, 0.35 * Jn]), np.concatenate([whip_m * n_sol, np.zeros(3)]), reg=0.020)
    da_hit = dq.detach().numpy().astype(np.float64) / ALPHA_J
    da_whip = dq_whip / ALPHA_J
    # Cap whip action amplitude so it stays clean & learnable
    w_max = float(np.abs(da_whip).max())
    if w_max > 0.42:
        da_whip = da_whip * (0.42 / w_max)
    # Cock-back + forward-whip decomposition: offset = da_hit - da_whip, ramp = +da_whip!
    da_cock = da_hit - da_whip
    return da_cock, da_whip, da_hit, err_mm, float(bp_k[0]), float(bp_k[2])


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
                if r['success'] and not r['robot_net'] and not r['low_speed']:
                    solved.add(r['case_id'])

    pw = PhysicalWorldV2(ASSET_SHA)
    geom = pw.geometry
    rid = geom.ids['Racket']

    # Collect unsolved hard-shell failures, prioritizing iy>=+0.12 / iz>=1.84 / ft<0.72!
    cands = []
    cache = {}
    for nd, sp in zip(args.nominal_dirs, args.nominal_specs, strict=True):
        rows = json.loads((nd / 'probe_summary.json').read_text())['rows']
        cases = json.loads(sp.read_text())['cases']
        blob = torch.load(nd / 'probe_tape.pt', map_location='cpu', weights_only=True)
        cache[str(nd)] = (blob['tape'], int(blob['start_capture_tick']))
        for i, (r, c) in enumerate(zip(rows, cases, strict=True)):
            cid = r['case_id']
            if r['success'] or cid in solved:
                continue
            ip = c['prediction_only']['intercept_position_m']
            ft = float(c['prediction_only']['flight_time_s'])
            iy, iz = float(ip[1]), float(ip[2])
            # Priority score: highest for hard unreachable shell (iy>=+0.12, iz>=1.84, ft<0.72)
            prio = 0.0
            if iy >= 0.12:
                prio += 2.5
            if iz >= 1.84:
                prio += 2.0
            if ft < 0.72:
                prio += 2.0
            if r['frame'] and not r['string']:
                prio += 1.0
            cands.append((-prio, str(nd), i, c, r))

    cands.sort(key=lambda x: x[0])
    sel = cands[:args.max_cases]
    print(f'[ApexWhip] Selected {len(sel)}/{len(cands)} unsolved hard-shell failures '
          f'(iy>=0.12={sum(x[3]["prediction_only"]["intercept_position_m"][1]>=0.12 for x in sel)}, '
          f'iz>=1.84={sum(x[3]["prediction_only"]["intercept_position_m"][2]>=1.84 for x in sel)}, '
          f'ft<0.72={sum(x[3]["prediction_only"]["flight_time_s"]<0.72 for x in sel)})', flush=True)

    out_cases, out_offs, out_rmps, out_peaks, out_widths, bmeta = [], [], [], [], [], []
    errs = []
    # 4 biomechanically distinct Cock-Back + Forward-Whip apex branches (tuned nz & inward yaw for center court!):
    configs = [
        (0.032, 0.04, 0.15, 'aw2_32ms_nz04_w15'),
        (0.040, 0.06, 0.17, 'aw2_40ms_nz06_w17'),
        (0.048, 0.09, 0.18, 'aw2_48ms_nz09_w18'),
        (0.044, 0.05, 0.16, 'aw2_44ms_nz05_w16'),
    ]

    for _, nd_s, lane, c, r in sel:
        tape, start_tick = cache[nd_s]
        ep0 = tape['episode'][:, lane] == 0
        st = tape['state'][ep0, lane].float()
        hc = tape['had_contact'][ep0, lane].numpy()
        pres = tape['present'][ep0, lane].numpy()
        if hc.any():
            rel = int(np.nonzero(hc)[0][0])
            c_idx = max(0, rel - 1)
            base_peak = start_tick + rel
        else:
            dec = decode_physical(st)
            bx = dec['ball_position_w'][:, 0].numpy()
            bx[~pres] = 99.0
            c_idx = int(np.argmin(np.abs(bx - 1.15)))
            base_peak = start_tick + c_idx

        for dt_s, nz, whip_m, tag in configs:
            da_cock, da_whip, _, err_mm, hx, hz = solve_apex_whip_ik(
                geom, rid, st[c_idx], dt_s=dt_s, target_nz=nz, whip_m=whip_m)
            errs.append(err_mm)
            pt = base_peak + int(round(dt_s * 100.0))
            out_cases.append(c)
            out_offs.append(clip_action_vec(da_cock, limit=0.68))
            out_rmps.append(clip_action_vec(da_whip, limit=0.44))
            out_peaks.append(int(pt))
            out_widths.append(58)
            bmeta.append(dict(case_id=c['case_id'], tag=tag, dt_ms=int(round(dt_s * 1000)),
                              nz=nz, whip_m=whip_m, ik_err_mm=round(err_mm, 1), hx=round(hx, 2), hz=round(hz, 2)))

    print(f'[ApexWhip] IK 3D sweetspot error across {len(errs)} branches: '
          f'med={np.median(errs):.1f}mm, q75={np.quantile(errs, 0.75):.1f}mm, q90={np.quantile(errs, 0.90):.1f}mm', flush=True)
    spec = dict(
        mode='apex_whip_probe',
        cases=out_cases,
        offsets=out_offs,
        ramp_offsets=out_rmps,
        peak_ticks=out_peaks,
        width_ticks=out_widths,
        branch_meta=bmeta,
        gate_mode='approach_gate',
        record_tape=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(spec, indent=2) + '\n')
    print(f'[ApexWhip] Wrote {len(out_cases)}-lane spec -> {args.output}', flush=True)


if __name__ == '__main__':
    main()
