#!/usr/bin/env python3
"""Plan Pass-2 closed-loop Nonlinear-FK refinements on Pass-1 unsolved string-hit cases.

Takes cases from Pass-1 probe runs that ALREADY achieved qualified string contact
(fixing FRAME_HIT/NO_CONTACT) yet failed post-contact (LONG OOB, LOW_SPD, BALL_NET, WIDE),
and adds 4 targeted second-order normal/speed refinements on top of the exact Pass-1
(offset, ramp, peak_tick) while holding contact centered on the string bed.
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

from plan_jacobian_probes import compute_arm_jacobians, solve_dls, solve_nonlinear_ik, clip_action_vec
from quadsmash.rl.supermimic_physical_dataset_v1 import decode_physical
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2


def compute_shaped_reward(row, min_dist_m: float = 0.15) -> float:
    succ = 1.0 if row['success'] else 0.0
    qstr = 1.0 if row['string'] else 0.0
    rnet = 1.0 if row['robot_net'] else 0.0
    sm = row['string_margin_m']
    sm_t = max(-1.5, min(1.0, sm / 0.040)) if (sm is not None and math.isfinite(sm)) else -1.5
    spm = row['speed_margin_mps']
    spm_t = max(-1.5, min(1.0, spm / 0.55)) if (qstr > 0 and spm is not None and math.isfinite(spm)) else -0.5
    lm = row['landing_margin_m']
    lm_t = max(-1.5, min(1.0, lm / 0.35)) if (qstr > 0 and lm is not None and math.isfinite(lm)) else -1.0
    nm = row['net_margin_m']
    nm_t = max(-1.0, min(1.0, nm / 0.12)) if (qstr > 0 and nm is not None and math.isfinite(nm)) else -0.5
    had_c = (row['string'] or row['frame'] or (sm is not None))
    dist_pen = 0.0 if had_c else -2.0 * max(0.0, min(0.40, min_dist_m - 0.10))
    return float(
        3.50 * succ + 1.20 * qstr + 0.70 * sm_t + 0.55 * spm_t + 0.70 * lm_t + 0.35 * nm_t + dist_pen - 6.00 * rnet
    )


def add_clipped(base18, delta6, limit=0.58):
    out = np.array(base18, dtype=np.float64).copy()
    out[12:18] += delta6
    m = float(np.abs(out[12:18]).max())
    if m > limit:
        out[12:18] *= (limit / m)
    return out.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nominal-dirs', type=Path, nargs='+', required=True)
    parser.add_argument('--nominal-specs', type=Path, nargs='+', required=True)
    parser.add_argument('--probe-dirs', type=Path, nargs='+', required=True)
    parser.add_argument('--probe-specs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-cases', type=int, default=64)
    args = parser.parse_args()

    # Load nominal summaries & case specs
    nom_rows = {}
    case_objs = {}
    for ndir, nspec_p in zip(args.nominal_dirs, args.nominal_specs, strict=True):
        ns = json.loads((ndir / 'probe_summary.json').read_text())
        nspec = json.loads(nspec_p.read_text())
        for r, c in zip(ns['rows'], nspec['cases'], strict=True):
            nom_rows[r['case_id']] = r
            case_objs[r['case_id']] = c

    # Group all Pass-1 probe branches by case_id
    by_case = {}
    probe_cache = {}
    for pdir, pspec_p in zip(args.probe_dirs, args.probe_specs, strict=True):
        ps = json.loads((pdir / 'probe_summary.json').read_text())
        pspec = json.loads(pspec_p.read_text())
        eps = {r['env_id']: r for r in map(json.loads, (pdir / 'episodes.jsonl').read_text().splitlines())
               if r['episode_index'] == 0}
        probe_cache[str(pdir)] = dict(pspec=pspec, eps=eps, tape=None)
        for idx, (row, meta) in enumerate(zip(ps['rows'], ps['branch_meta'], strict=True)):
            by_case.setdefault(meta['case_id'], []).append((str(pdir), idx, row, meta))

    # Select unsolved nominal failures: Priority 1 = Pass-1 string-hit cases; Priority 2 = closest Pass-1 frame-hit cases
    str_candidates = []
    frame_candidates = []
    for cid, items in by_case.items():
        if nom_rows[cid]['success']:
            continue
        if any(row['success'] and not row['robot_net'] for _, _, row, _ in items):
            continue
        str_items = [(compute_shaped_reward(row, 0.15), pdir, idx, row, meta)
                     for pdir, idx, row, meta in items if row['string'] and not row['robot_net']]
        if str_items:
            best = max(str_items, key=lambda x: x[0])
            str_candidates.append((best[0], 'STRING', cid, best[1], best[2], best[3], best[4]))
        else:
            fr_items = [(compute_shaped_reward(row, 0.15), pdir, idx, row, meta)
                        for pdir, idx, row, meta in items if row['frame'] and not row['robot_net']]
            if fr_items:
                best = max(fr_items, key=lambda x: x[0])
                frame_candidates.append((best[0], 'FRAME', cid, best[1], best[2], best[3], best[4]))

    str_candidates.sort(key=lambda x: -x[0])
    frame_candidates.sort(key=lambda x: -x[0])
    selected = (str_candidates + frame_candidates)[:args.max_cases]
    n_str_sel = sum(1 for x in selected if x[1] == 'STRING')
    n_fr_sel = sum(1 for x in selected if x[1] == 'FRAME')
    print(f'Selected {len(selected)}/{len(str_candidates)+len(frame_candidates)} unsolved cases '
          f'({n_str_sel} Pass-1 string + {n_fr_sel} Pass-1 closest-frame) for 4-branch Pass-2 refinement')

    asset_sha = 'c2264fcece90b042c7c6a1cac7d516af83972729d0402947e3a519041e184cf2'
    pw = PhysicalWorldV2(asset_sha)
    geom = pw.geometry
    rid = geom.ids['Racket']

    out_cases, out_offsets, out_ramps, out_peaks, out_widths, branch_meta = [], [], [], [], [], []
    counts = {}
    for _, pool_kind, cid, pdir_s, p_idx, row, meta in selected:
        cache = probe_cache[pdir_s]
        if cache['tape'] is None:
            blob = torch.load(Path(pdir_s) / 'probe_tape.pt', map_location='cpu', weights_only=True)
            cache['tape'] = blob['tape']
            cache['start_tick'] = int(blob['start_capture_tick'])
        tape = cache['tape']
        pspec = cache['pspec']
        e = cache['eps'][p_idx]

        off1 = pspec['offsets'][p_idx]
        rmp1 = pspec['ramp_offsets'][p_idx] if pspec.get('ramp_offsets') else [0.0] * 18
        pt1 = float(pspec['peak_ticks'][p_idx])
        wt1 = float(pspec['width_ticks'][p_idx])
        c_obj = case_objs[cid]
        v_in = np.array(c_obj['prediction_only']['incoming_velocity_mps'], dtype=np.float64)

        ep0 = tape['episode'][:, p_idx] == 0
        st = tape['state'][ep0, p_idx].float()
        hc = tape['had_contact'][ep0, p_idx].numpy()
        c_idx = max(0, int(np.nonzero(hc)[0][0]) - 1) if hc.any() else len(st) // 2

        Jp, Jn, p_r, R_r = compute_arm_jacobians(geom, rid, st[c_idx])
        normal = R_r[:, 0].copy()
        if np.dot(normal, v_in) > 0:
            normal = -normal
        n0_pos = R_r[:, 0].copy()
        if n0_pos[0] < 0:
            n0_pos = -n0_pos
        da_fwd = solve_dls(np.vstack([Jp, 0.25 * Jn]), np.concatenate([0.10 * normal, np.zeros(3)]), reg=0.02) / 0.4
        da_fwd = da_fwd / max(float(np.abs(da_fwd).max()), 1e-6)

        land = e.get('first_landing_m') or [3.0, 0.0, 0.0]
        lx = land[0] if land[0] is not None and math.isfinite(land[0]) else 3.0
        ly = land[1] if land[1] is not None and math.isfinite(land[1]) else 0.0
        spd = 4.5 + row['speed_margin_mps'] if row['speed_margin_mps'] is not None else 4.0

        if pool_kind == 'FRAME':
            subkind = 'P2_FRAME'
            da_f1, dt1 = solve_nonlinear_ik(
                geom, rid, st[c_idx], v_in, target_yz=(0.0, -0.015), target_nyz=(-0.03, -0.02),
                w_yz=130.0, w_n=6.0, w_dq=0.22, steps=35, dq_limit=0.30)
            da_f2, dt2 = solve_nonlinear_ik(
                geom, rid, st[c_idx], v_in, target_yz=(0.0, -0.025), target_nyz=(-0.03, -0.08),
                w_yz=150.0, w_n=8.0, w_dq=0.14, steps=35, dq_limit=0.36)
            branches = [
                (add_clipped(off1, 0.90 * da_f1, 0.60), [0.0] * 18, pt1 + dt1, wt1, 'p2_fr_0rmp'),
                (add_clipped(off1, 1.05 * da_f1, 0.60), add_clipped(rmp1, 0.14 * da_fwd, 0.36), pt1 + dt1, wt1, 'p2_fr_spd0.14'),
                (add_clipped(off1, 1.00 * da_f2, 0.62), [0.0] * 18, pt1 + dt2, wt1, 'p2_fr2_0rmp'),
                (add_clipped(off1, 1.15 * da_f2, 0.64), add_clipped(rmp1, 0.22 * da_fwd, 0.40), pt1 + dt2, wt1, 'p2_fr2_spd0.22'),
            ]
        elif row['low_speed']:
            subkind = 'P2_LOW_SPD'
            dn_y = -0.10 * float(np.clip(ly, -1.2, 1.2))
            dn_z = 0.06 if lx < 3.1 else -0.04
            tny = float(np.clip(n0_pos[1] + dn_y, -0.25, 0.25))
            tnz = float(np.clip(n0_pos[2] + dn_z, -0.10, 0.30))
            da_aim, _ = solve_nonlinear_ik(
                geom, rid, st[c_idx], v_in, target_yz=(0.0, 0.0), target_nyz=(tny, tnz),
                w_yz=100.0, w_n=12.0, w_dq=0.35, steps=30, dq_limit=0.18)
            branches = [
                (off1, add_clipped(rmp1, 0.14 * da_fwd, 0.36), pt1, wt1, 'p2_spd+0.14'),
                (off1, add_clipped(rmp1, 0.24 * da_fwd, 0.40), pt1, wt1, 'p2_spd+0.24'),
                (add_clipped(off1, 0.9 * da_aim, 0.58), add_clipped(rmp1, 0.26 * da_fwd, 0.40), pt1, wt1, 'p2_aim+spd0.26'),
                (add_clipped(off1, 1.1 * da_aim, 0.58), add_clipped(rmp1, 0.34 * da_fwd, 0.44), pt1, wt1, 'p2_aim+spd0.34'),
            ]
        elif (not row['ball_net']) and lx > 3.95:
            subkind = 'P2_LONG_OOB'
            dn_y = -0.16 * float(np.clip(ly, -1.4, 1.4)) if abs(ly) > 0.55 else -0.06 * float(np.clip(ly, -1.0, 1.0))
            dn_z = -float(np.clip((lx - 3.10) / 2.57, 0.14, 0.36))
            tny1 = float(np.clip(n0_pos[1] + dn_y, -0.28, 0.28))
            tnz1 = float(np.clip(n0_pos[2] + dn_z, -0.24, 0.22))
            tny2 = float(np.clip(n0_pos[1] + 1.4 * dn_y, -0.34, 0.34))
            tnz2 = float(np.clip(n0_pos[2] + 1.45 * dn_z, -0.32, 0.22))
            da_d1, dt1 = solve_nonlinear_ik(
                geom, rid, st[c_idx], v_in, target_yz=(0.0, 0.0), target_nyz=(tny1, tnz1),
                w_yz=100.0, w_n=18.0, w_dq=0.22, steps=35, dq_limit=0.26)
            da_d2, dt2 = solve_nonlinear_ik(
                geom, rid, st[c_idx], v_in, target_yz=(0.0, 0.0), target_nyz=(tny2, tnz2),
                w_yz=100.0, w_n=22.0, w_dq=0.16, steps=35, dq_limit=0.32)
            branches = [
                (add_clipped(off1, 1.0 * da_d1, 0.68), [0.0] * 18, pt1 + dt1, wt1, 'p2_long_down_0rmp'),
                (add_clipped(off1, 1.0 * da_d1, 0.68), (0.5 * np.array(rmp1)).tolist(), pt1 + dt1, wt1, 'p2_long_down_halfrmp'),
                (add_clipped(off1, 1.05 * da_d2, 0.68), (0.5 * np.array(rmp1)).tolist(), pt1 + dt2, wt1, 'p2_long_down2_halfrmp'),
                (add_clipped(off1, 1.05 * da_d2, 0.68), rmp1, pt1 + dt2, wt1, 'p2_long_down2_fullrmp'),
            ]
        else:
            subkind = 'P2_BNET_OR_WIDE'
            is_short = row['ball_net'] or lx < 2.3
            dn_y = -0.18 * float(np.clip(ly, -1.5, 1.5)) if abs(ly) > 0.65 else -0.06 * float(np.clip(ly, -1.0, 1.0))
            dn_z = 0.14 if is_short else (0.04 if lx < 3.0 else -0.06)
            tny1 = float(np.clip(n0_pos[1] + dn_y, -0.28, 0.28))
            tnz1 = float(np.clip(n0_pos[2] + dn_z, -0.12, 0.36))
            tny2 = float(np.clip(n0_pos[1] + 1.6 * dn_y, -0.34, 0.34))
            tnz2 = float(np.clip(n0_pos[2] + 1.6 * dn_z, -0.15, 0.42))
            da_a1, dt1 = solve_nonlinear_ik(
                geom, rid, st[c_idx], v_in, target_yz=(0.0, 0.0), target_nyz=(tny1, tnz1),
                w_yz=100.0, w_n=18.0, w_dq=0.25, steps=35, dq_limit=0.24)
            da_a2, dt2 = solve_nonlinear_ik(
                geom, rid, st[c_idx], v_in, target_yz=(0.0, 0.0), target_nyz=(tny2, tnz2),
                w_yz=100.0, w_n=22.0, w_dq=0.18, steps=35, dq_limit=0.30)
            branches = [
                (add_clipped(off1, 1.0 * da_a1, 0.58), rmp1, pt1 + dt1, wt1, 'p2_aim_1.0x'),
                (add_clipped(off1, 1.1 * da_a2, 0.58), rmp1, pt1 + dt2, wt1, 'p2_aim_1.6x'),
                (add_clipped(off1, 1.0 * da_a1, 0.58), add_clipped(rmp1, 0.14 * da_fwd, 0.36), pt1 + dt1, wt1, 'p2_aim+spd0.14'),
                (add_clipped(off1, 1.1 * da_a2, 0.58), add_clipped(rmp1, 0.22 * da_fwd, 0.40), pt1 + dt2, wt1, 'p2_aim+spd0.22'),
            ]

        counts[subkind] = counts.get(subkind, 0) + 1
        for b_idx, (off, rmp, pt, wt, tag) in enumerate(branches):
            out_cases.append(c_obj)
            out_offsets.append(off)
            out_ramps.append(rmp)
            out_peaks.append(float(pt))
            out_widths.append(float(wt))
            branch_meta.append(dict(
                case_id=cid, failure_kind=subkind, orig_failure_kind=meta['failure_kind'],
                pass1_tag=meta['tag'], pass1_dir=pdir_s, branch=b_idx, tag=tag, peak_tick=int(pt),
            ))

    print('  By Pass-2 subkind:', counts)
    spec = dict(
        mode='pass2_refinement',
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
    print(f'Wrote {len(out_cases)}-lane Pass-2 refinement spec -> {args.output}')


if __name__ == '__main__':
    main()
