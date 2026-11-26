#!/usr/bin/env python3
"""Generate strictly disjoint training serves and plan exact FK-Jacobian-guided probes."""
import argparse
import gzip
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from freeze_full_racket_protocol_v4 import incoming_cases
from quadsmash.physics.court_v4 import load_court, top_height
from quadsmash.physics.full_racket_v4 import load_racket, string_polygon
from quadsmash.rl.supermimic_physical_dataset_v1 import decode_physical
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2


def quat_wxyz_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def reserved_case_ids(execution_root: Path):
    ids = set()
    p01 = execution_root / 'reports/G07_full_racket_v4/protocol_frozen01'
    for name in ('training_cases.json', 'A_cases.json', 'B_cases.json'):
        p = p01 / name
        if p.is_file():
            d = json.loads(p.read_text())
            rows = d.get('cases', d) if isinstance(d, dict) else d
            ids.update(r['case_id'] for r in rows)
    rs = ROOT / 'reports/supermimic_research_scale_20261121'
    for name in ('fixed128.json.gz', 'independent512.json.gz'):
        p = rs / name
        if p.is_file():
            d = json.loads(gzip.decompress(p.read_bytes()))
            ids.update(r['case_id'] for r in d['cases'])
    return ids


def generate_training_spec(count_a: int, count_b: int, seed_a: int, seed_b: int,
                           execution_root: Path, output: Path):
    reserved = reserved_case_ids(execution_root)
    cases = incoming_cases(count_a, seed_a, 'A') + incoming_cases(count_b, seed_b, 'B')
    overlap = {c['case_id'] for c in cases} & reserved
    if overlap:
        raise ValueError(f'Generated training cases overlap reserved eval panels: {sorted(overlap)[:5]}')
    spec = dict(
        mode='nominal',
        seed_a=seed_a, seed_b=seed_b, count_a=count_a, count_b=count_b,
        cases=cases,
        offsets=None,
        record_tape=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(spec, indent=2) + '\n')
    print(f'Generated {len(cases)} strictly disjoint training cases ({count_a} A + {count_b} B) -> {output}')


def interior_string_grid(margin_m=0.028):
    poly = np.array(string_polygon(load_racket()), dtype=np.float64)
    edges = np.roll(poly, -1, axis=0) - poly
    inward = np.stack((-edges[:, 1], edges[:, 0]), axis=-1)
    inward /= np.linalg.norm(inward, axis=-1, keepdims=True)
    offsets = (poly * inward).sum(-1)
    ys = np.linspace(-0.075, 0.075, 13)
    zs = np.linspace(-0.095, 0.095, 15)
    pts = []
    for y in ys:
        for z in zs:
            m = (inward[:, 0] * y + inward[:, 1] * z - offsets).min()
            if m >= margin_m:
                pts.append((y, z, m))
    return np.array(pts, dtype=np.float64)


def compute_arm_jacobians(geom, racket_id, state_row):
    d0 = decode_physical(state_row.float().unsqueeze(0))
    q0 = d0['joint_position'][0, 12:18].clone()

    def fn_p(q_arm):
        q = d0['joint_position'].clone()
        q[0, 12:18] = q_arm
        fk = geom.forward(q, d0['joint_velocity'], d0['root_position_w'], d0['root_quaternion_w'],
                          d0['root_link_velocity_w'], d0['root_angular_velocity_w'])
        return fk['position'][0, racket_id].float()

    def fn_n(q_arm):
        q = d0['joint_position'].clone()
        q[0, 12:18] = q_arm
        fk = geom.forward(q, d0['joint_velocity'], d0['root_position_w'], d0['root_quaternion_w'],
                          d0['root_link_velocity_w'], d0['root_angular_velocity_w'])
        return fk['rotation'][0, racket_id, :, 0].float()

    Jp = torch.autograd.functional.jacobian(fn_p, q0).numpy().astype(np.float64)
    Jn = torch.autograd.functional.jacobian(fn_n, q0).numpy().astype(np.float64)
    with torch.no_grad():
        q = d0['joint_position']
        fk = geom.forward(q, d0['joint_velocity'], d0['root_position_w'], d0['root_quaternion_w'],
                          d0['root_link_velocity_w'], d0['root_angular_velocity_w'])
        p_r = fk['position'][0, racket_id].numpy().astype(np.float64)
        R_r = fk['rotation'][0, racket_id].numpy().astype(np.float64)
    return Jp, Jn, p_r, R_r


def solve_dls(A, b, reg=0.015):
    return np.linalg.solve(A.T @ A + reg * np.eye(A.shape[1]), A.T @ b)


def solve_nonlinear_ik(geom, racket_id, state_row, v_in_np, *,
                       target_yz=(0.0, 0.0), target_nyz=None,
                       w_yz=120.0, w_n=4.0, w_dq=0.25, steps=35, lr=0.06, dq_limit=0.34):
    """Exact differentiable nonlinear FK IK through PhysicalWorldV2.geometry.forward.

    Simultaneously places the exact oblique ray-plane intersection at target_yz on the
    string bed AND steers the racket normal toward target_nyz (court center aim) with
    minimum joint displacement dq.
    """
    d0 = decode_physical(state_row.float().unsqueeze(0))
    q0 = d0['joint_position'][0, 12:18].clone()
    bp = d0['ball_position_w'][0].float()
    v_in = torch.tensor(v_in_np, dtype=torch.float32)
    tyz = torch.tensor(target_yz, dtype=torch.float32)

    with torch.no_grad():
        fk0 = geom.forward(d0['joint_position'], d0['joint_velocity'], d0['root_position_w'],
                           d0['root_quaternion_w'], d0['root_link_velocity_w'], d0['root_angular_velocity_w'])
        p_r0 = fk0['position'][0, racket_id].float()
        n0 = fk0['rotation'][0, racket_id, :, 0].float()
        n0_in = -n0 if float(torch.dot(n0, v_in)) > 0 else n0
        d0_nv = float(torch.dot(v_in, n0_in))
        t_plane_0 = float(torch.dot(p_r0 - bp, n0_in)) / (d0_nv if abs(d0_nv) >= 0.5 else -6.0)
        if float(n0[0]) < 0:
            n0 = -n0
    tnyz = n0[1:3].clone() if target_nyz is None else torch.tensor(target_nyz, dtype=torch.float32)

    dq = torch.zeros(6, requires_grad=True)
    opt = torch.optim.Adam([dq], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        q = d0['joint_position'].clone()
        q[0, 12:18] = q0 + dq
        fk = geom.forward(q, d0['joint_velocity'], d0['root_position_w'], d0['root_quaternion_w'],
                          d0['root_link_velocity_w'], d0['root_angular_velocity_w'])
        p_r = fk['position'][0, racket_id].float()
        R_r = fk['rotation'][0, racket_id].float()
        n = R_r[:, 0]
        if float(torch.dot(n, v_in)) > 0:
            n = -n
        denom = torch.dot(v_in, n)
        denom = torch.where(denom.abs() < 0.5, torch.tensor(-6.0), denom)
        t_plane = torch.dot(p_r - bp, n) / denom
        p_hit = bp + t_plane * v_in
        loc = R_r.T @ (p_hit - p_r)
        n_pos = n if float(n[0]) >= 0 else -n
        loss = (w_yz * (loc[1:3] - tyz).square().sum()
                + w_n * (n_pos[1:3] - tnyz).square().sum()
                + w_dq * dq.square().sum())
        loss.backward()
        opt.step()
        with torch.no_grad():
            dq.clamp_(-dq_limit, dq_limit)

    with torch.no_grad():
        q = d0['joint_position'].clone()
        q[0, 12:18] = q0 + dq
        fk = geom.forward(q, d0['joint_velocity'], d0['root_position_w'], d0['root_quaternion_w'],
                          d0['root_link_velocity_w'], d0['root_angular_velocity_w'])
        p_r = fk['position'][0, racket_id].float()
        R_r = fk['rotation'][0, racket_id].float()
        n = R_r[:, 0]
        if float(torch.dot(n, v_in)) > 0:
            n = -n
        denom = float(torch.dot(v_in, n))
        t_plane = float(torch.dot(p_r - bp, n)) / (denom if abs(denom) >= 0.5 else -6.0)
        dt_ticks = int(round(np.clip((t_plane - t_plane_0) * 100.0, -4.0, 6.0)))
    return (dq.detach().numpy().astype(np.float64) / 0.4), dt_ticks


def solve_dynamic_intercept_ik(geom, racket_id, state_row, v_in_np, *,
                               target_yz=(0.0, 0.0), target_nyz=(-0.04, 0.04),
                               dt_init=0.024, dt_bounds=(0.005, 0.055), qdot_scale=0.45,
                               w_pos=180.0, w_n=6.0, w_dq=0.22, steps=45, lr=0.05, dq_limit=0.30):
    """Exact dynamic delayed-intercept FK IK accounting for rising torso (vr) & falling shuttle (v_in).

    On high top-rim FRAME_HITs (loc_z > +85mm), the arm is near vertical reach saturation (+40mm max),
    whereas intercepting ~20-35ms later closes loc_z by ~110mm (ball drops -3.2m/s while torso rises +0.48m/s).
    """
    d0 = decode_physical(state_row.float().unsqueeze(0))
    q0 = d0['joint_position'][0, 12:18].clone()
    qd0 = d0['joint_velocity'][0, 12:18].clone()
    bp = d0['ball_position_w'][0].float()
    vr = d0['root_link_velocity_w'][0].float()
    v_in = torch.tensor(v_in_np, dtype=torch.float32)
    tyz = torch.tensor(target_yz, dtype=torch.float32)
    tnyz = torch.tensor(target_nyz, dtype=torch.float32)

    dq = torch.zeros(6, requires_grad=True)
    dt = torch.tensor(float(dt_init), requires_grad=True)
    opt = torch.optim.Adam([dq, dt], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        q = d0['joint_position'].clone()
        q[0, 12:18] = q0 + qdot_scale * dt * qd0 + dq
        fk = geom.forward(q, d0['joint_velocity'], d0['root_position_w'], d0['root_quaternion_w'],
                          d0['root_link_velocity_w'], d0['root_angular_velocity_w'])
        p_r = fk['position'][0, racket_id].float() + dt * vr
        R_r = fk['rotation'][0, racket_id].float()
        p_b = bp + dt * v_in
        loc = R_r.T @ (p_b - p_r)  # want loc[0]=0 (on plane), loc[1:3]=target_yz (string sweet spot!)
        n = R_r[:, 0]
        n_pos = n if float(n[0]) >= 0 else -n
        loss = (w_pos * (loc[0].square() + (loc[1:3] - tyz).square().sum())
                + w_n * (n_pos[1:3] - tnyz).square().sum()
                + w_dq * dq.square().sum())
        loss.backward()
        opt.step()
        with torch.no_grad():
            dq.clamp_(-dq_limit, dq_limit)
            dt.clamp_(float(dt_bounds[0]), float(dt_bounds[1]))

    dt_ticks = int(round(np.clip(float(dt) * 100.0, -3.0, 6.0)))
    alpha_j = np.array([0.32, 0.44, 0.48, 0.35, 0.52, 0.60], dtype=np.float64)
    return (dq.detach().numpy().astype(np.float64) / alpha_j), dt_ticks


def clip_action_vec(da6, limit=0.35):
    m = float(np.abs(da6).max())
    if m > limit:
        da6 = da6 * (limit / m)
    out = [0.0] * 18
    out[12:18] = da6.tolist()
    return out


def plan_probes(nominal_dir: Path, nominal_spec_path: Path, output: Path,
                max_cases: int = 32, offset_cases: int = 0, branches_per_case: int = 4, include_marginal: bool = True):
    nom_spec = json.loads(nominal_spec_path.read_text())
    cases = nom_spec['cases']
    eps = {r['env_id']: r for r in map(json.loads, (nominal_dir / 'episodes.jsonl').read_text().splitlines())
           if r['episode_index'] == 0}
    summary = json.loads((nominal_dir / 'probe_summary.json').read_text())
    rows_by_idx = summary['rows']
    contacts = {}
    for c in json.loads((nominal_dir / 'native_contact_motion.json').read_text()):
        if c['env_id'] not in contacts:
            contacts[c['env_id']] = c

    tape_blob = torch.load(nominal_dir / 'probe_tape.pt', map_location='cpu', weights_only=True)
    tape = tape_blob['tape']
    start_tick = int(tape_blob['start_capture_tick'])

    # Load exact FK geometry
    asset_sha = 'c2264fcece90b042c7c6a1cac7d516af83972729d0402947e3a519041e184cf2'
    pw = PhysicalWorldV2(asset_sha)
    geom = pw.geometry
    racket_id = geom.ids['Racket']
    grid = interior_string_grid(margin_m=0.028)

    # Prioritize failing cases: 0=FRAME_HIT, 1=STR_FAIL(LOW_SPD/BNET/OOB), 2=NO_CONTACT(near-miss first), 3=MARGINAL_SUCCESS
    candidates = []
    for i, c_obj in enumerate(cases):
        r = rows_by_idx[i]
        e = eps[i]
        if r['success']:
            if include_marginal:
                land = e.get('first_landing_m') or [3.0, 0.0, 0.0]
                lx = land[0] if land[0] is not None and math.isfinite(land[0]) else 3.0
                ly = land[1] if land[1] is not None and math.isfinite(land[1]) else 0.0
                spm = r['speed_margin_mps'] if r['speed_margin_mps'] is not None else 1.0
                sm = r['string_margin_m'] if r['string_margin_m'] is not None else 0.05
                border_gap = min(4.0 - lx, lx - 2.0, 1.0 - abs(ly))
                if spm < 0.35 or sm < 0.015 or border_gap < 0.32:
                    candidates.append((3, min(border_gap, spm, sm * 20.0), i, 'MARGINAL'))
            continue
        if r['frame'] and not r['string']:
            sm = abs(r['string_margin_m']) if r['string_margin_m'] is not None else 0.05
            candidates.append((0, sm, i, 'FRAME_HIT'))
        elif r['string']:
            kind = 'LOW_SPD' if r['low_speed'] else ('BALL_NET' if r['ball_net'] else 'OOB')
            candidates.append((1, 0.0, i, kind))
        else:
            # NO_CONTACT: compute min ball-to-racket distance from tape
            ep0 = tape['episode'][:, i] == 0
            st = tape['state'][ep0, i].float()
            pres = tape['present'][ep0, i]
            d_all = decode_physical(st)
            with torch.no_grad():
                fk = geom.forward(d_all['joint_position'], d_all['joint_velocity'],
                                  d_all['root_position_w'], d_all['root_quaternion_w'],
                                  d_all['root_link_velocity_w'], d_all['root_angular_velocity_w'])
                r_pos = fk['position'][:, racket_id].numpy()
                b_pos = d_all['ball_position_w'].numpy()
                dists = np.linalg.norm(b_pos - r_pos, axis=-1)
                dists[~pres.numpy()] = 9.9
                min_d = float(dists.min())
            candidates.append((2, min_d, i, 'NO_CONTACT'))

    candidates.sort()
    selected = candidates[offset_cases:offset_cases + max_cases]
    print(f'Selected {len(selected)}/{len(candidates)} training cases for {branches_per_case}-branch Jacobian probing:')
    counts = {}
    for _, _, _, k in selected:
        counts[k] = counts.get(k, 0) + 1
    print('  By failure type:', counts)

    out_cases, out_offsets, out_ramps, out_peaks, out_widths, branch_meta = [], [], [], [], [], []
    for _, metric, i, kind in selected:
        c_obj = cases[i]
        e = eps[i]
        ep0 = tape['episode'][:, i] == 0
        st = tape['state'][ep0, i].float()
        el = tape['elapsed'][ep0, i].numpy()
        hc = tape['had_contact'][ep0, i].numpy()
        pres = tape['present'][ep0, i].numpy()
        v_in = np.array(c_obj['prediction_only']['incoming_velocity_mps'], dtype=np.float64)
        pred_t = 2.0 + float(c_obj['prediction_only']['flight_time_s'])
        pred_tick = int(round(pred_t * 100))

        # Determine target contact/closest tick
        if hc.any():
            rel_idx = int(np.nonzero(hc)[0][0])
            c_idx = max(0, rel_idx - 1)
            peak_tick = start_tick + rel_idx
        else:
            d_all = decode_physical(st)
            with torch.no_grad():
                fk = geom.forward(d_all['joint_position'], d_all['joint_velocity'],
                                  d_all['root_position_w'], d_all['root_quaternion_w'],
                                  d_all['root_link_velocity_w'], d_all['root_angular_velocity_w'])
                r_pos = fk['position'][:, racket_id].numpy()
                b_pos = d_all['ball_position_w'].numpy()
                dists = np.linalg.norm(b_pos - r_pos, axis=-1)
                dists[~pres] = 9.9
                c_idx = int(np.argmin(dists))
                peak_tick = start_tick + c_idx

        Jp, Jn, p_r, R_r = compute_arm_jacobians(geom, racket_id, st[c_idx])
        normal = R_r[:, 0]
        if np.dot(normal, v_in) > 0:
            normal = -normal
        ey, ez = R_r[:, 1], R_r[:, 2]

        # Oblique-ray contact Jacobian: maps dq_arm -> delta (local_y, local_z) of contact on racket face!
        nv = float(np.dot(normal, v_in))
        if abs(nv) < 0.5:
            nv = -6.0
        P_oblique = np.outer(v_in, normal) / nv - np.eye(3)
        J_contact = np.vstack([ey, ez]) @ P_oblique @ Jp  # [2, 6]

        # Unit forward-velocity direction in arm joint space (aligns racket velocity with +normal)
        da_fwd = solve_dls(np.vstack([Jp, 0.25 * Jn]), np.concatenate([0.10 * normal, np.zeros(3)]), reg=0.02) / 0.4
        da_fwd = da_fwd / max(float(np.abs(da_fwd).max()), 1e-6)

        # Nominal racket normal pointing into opponent (+x) court
        n0_pos = R_r[:, 0].copy()
        if n0_pos[0] < 0:
            n0_pos = -n0_pos

        branches = []  # list of (offset_18, ramp_18, peak_tick, width_tick, tag)
        if kind == 'FRAME_HIT':
            cm = contacts.get(i)
            loc_z = 0.0
            if cm is not None:
                R_c = quat_wxyz_to_rot(cm['racket_quaternion_w'])
                local = R_c.T @ (np.array(cm['point_m']) - np.array(cm['racket_center_m']))
                loc_z = float(local[2])
            # Exact nonlinear FK IK with flat/downward smash normal aim (prevents x>4.0m long OOB!)
            tz_mild = float(np.clip(loc_z, -0.020, 0.020))
            da_gentle, dt_g = solve_nonlinear_ik(
                geom, racket_id, st[c_idx], v_in,
                target_yz=(0.0, tz_mild), target_nyz=None,
                w_yz=110.0, w_n=4.0, w_dq=0.40, steps=30, dq_limit=0.22)
            da_smash, dt_s = solve_nonlinear_ik(
                geom, racket_id, st[c_idx], v_in,
                target_yz=(0.0, 0.0), target_nyz=(-0.03, -0.02),
                w_yz=140.0, w_n=5.0, w_dq=0.18, steps=35, dq_limit=0.30)
            da_flat, dt_f = solve_nonlinear_ik(
                geom, racket_id, st[c_idx], v_in,
                target_yz=(0.0, 0.0), target_nyz=(-0.03, -0.08),
                w_yz=140.0, w_n=6.0, w_dq=0.16, steps=35, dq_limit=0.32)
            da_deep, dt_d = solve_nonlinear_ik(
                geom, racket_id, st[c_idx], v_in,
                target_yz=(0.0, 0.0), target_nyz=(-0.03, -0.04),
                w_yz=160.0, w_n=4.5, w_dq=0.07, steps=40, dq_limit=0.36)
            if loc_z > 0.060:
                # Top-rim hit: combine 2 pitch-safe static IK branches + 2 pitch-safe Dynamic Delayed-Intercept IK branches!
                da_dyn1, dt_d1 = solve_dynamic_intercept_ik(
                    geom, racket_id, st[c_idx], v_in,
                    target_yz=(0.0, 0.0), target_nyz=(-0.03, -0.03),
                    dt_init=0.022, dt_bounds=(0.008, 0.045), qdot_scale=0.35, dq_limit=0.26)
                da_dyn2, dt_d2 = solve_dynamic_intercept_ik(
                    geom, racket_id, st[c_idx], v_in,
                    target_yz=(0.0, -0.010), target_nyz=(-0.03, -0.09),
                    dt_init=0.030, dt_bounds=(0.012, 0.055), qdot_scale=0.50, dq_limit=0.30)
                branches = [
                    (clip_action_vec(1.00 * da_gentle, 0.34), [0.0]*18, peak_tick + dt_g, 55, 'frame_nl_gentle0'),
                    (clip_action_vec(1.05 * da_smash, 0.44), [0.0]*18, peak_tick + dt_s, 55, 'frame_nl_smash0'),
                    (clip_action_vec(1.00 * da_dyn1, 0.56), [0.0]*18, peak_tick + dt_d1, 58, 'frame_dyn_delay1_0rmp'),
                    (clip_action_vec(1.05 * da_dyn2, 0.62), clip_action_vec(0.10 * da_fwd, 0.14), peak_tick + dt_d2, 62, 'frame_dyn_delay2_flat'),
                ]
            else:
                branches = [
                    (clip_action_vec(1.00 * da_gentle, 0.34), [0.0]*18, peak_tick + dt_g, 55, 'frame_nl_gentle0'),
                    (clip_action_vec(1.05 * da_smash, 0.44), [0.0]*18, peak_tick + dt_s, 55, 'frame_nl_smash0'),
                    (clip_action_vec(1.10 * da_flat, 0.48), clip_action_vec(0.10 * da_fwd, 0.14), peak_tick + dt_f, 55, 'frame_nl_flat_ctrl'),
                    (clip_action_vec(1.15 * da_deep, 0.56), clip_action_vec(0.14 * da_fwd, 0.18), peak_tick + dt_d, 60, 'frame_nl_deep'),
                ]
        elif kind == 'NO_CONTACT':
            da_m1, dt_1 = solve_nonlinear_ik(
                geom, racket_id, st[c_idx], v_in,
                target_yz=(0.0, 0.015), target_nyz=(-0.04, 0.00),
                w_yz=120.0, w_n=5.0, w_dq=0.25, steps=35, dq_limit=0.30)
            da_m2, dt_2 = solve_nonlinear_ik(
                geom, racket_id, st[c_idx], v_in,
                target_yz=(0.0, 0.0), target_nyz=(-0.04, -0.04),
                w_yz=140.0, w_n=6.0, w_dq=0.10, steps=40, dq_limit=0.36)
            da_m3, dt_3 = solve_nonlinear_ik(
                geom, racket_id, st[c_idx], v_in,
                target_yz=(0.0, 0.0), target_nyz=(-0.04, -0.07),
                w_yz=160.0, w_n=6.0, w_dq=0.05, steps=45, dq_limit=0.40)
            da_m4, dt_4 = solve_nonlinear_ik(
                geom, racket_id, st[c_idx], v_in,
                target_yz=(0.0, 0.0), target_nyz=(-0.04, 0.04),
                w_yz=140.0, w_n=5.0, w_dq=0.08, steps=40, dq_limit=0.36)
            branches = [
                (clip_action_vec(0.95 * da_m1, 0.44), clip_action_vec(0.14 * da_fwd, 0.18), peak_tick + dt_1, 60, 'miss_nl_mild+spd'),
                (clip_action_vec(1.10 * da_m2, 0.54), clip_action_vec(0.20 * da_fwd, 0.24), peak_tick + dt_2, 60, 'miss_nl_mid_smash'),
                (clip_action_vec(1.20 * da_m3, 0.62), clip_action_vec(0.24 * da_fwd, 0.28), peak_tick + dt_3, 65, 'miss_nl_deep_smash'),
                (clip_action_vec(1.15 * da_m4, 0.56), clip_action_vec(0.18 * da_fwd, 0.22), peak_tick + dt_4, 65, 'miss_nl_deep_lift'),
            ]
        elif kind == 'LOW_SPD':
            # Sweet-spot centering + strong forward-velocity ramps + delayed-intercept for high-z low-speed hits
            land = e.get('first_landing_m') or [3.0, 0.0, 0.0]
            lx = land[0] if land[0] is not None and math.isfinite(land[0]) else 3.0
            ly = land[1] if land[1] is not None and math.isfinite(land[1]) else 0.0
            tny = float(np.clip(n0_pos[1] - 0.10 * np.clip(ly, -1.2, 1.2), -0.20, 0.20))
            tnz = float(np.clip(n0_pos[2] + (0.04 if lx < 3.0 else -0.05), -0.08, 0.24))
            da_aim, dt_ls = solve_nonlinear_ik(
                geom, racket_id, st[c_idx], v_in,
                target_yz=(0.0, 0.0), target_nyz=(tny, tnz),
                w_yz=120.0, w_n=10.0, w_dq=0.22, steps=35, dq_limit=0.26)
            da_dyn_ls, dt_dls = solve_dynamic_intercept_ik(
                geom, racket_id, st[c_idx], v_in,
                target_yz=(0.0, 0.0), target_nyz=(tny, tnz),
                dt_init=0.015, dt_bounds=(0.004, 0.032), qdot_scale=0.40, dq_limit=0.24)
            branches = [
                ([0.0]*18, clip_action_vec(0.20 * da_fwd, 0.24), peak_tick, 50, 'spd_ramp_0.20'),
                (clip_action_vec(1.0 * da_aim, 0.30), clip_action_vec(0.26 * da_fwd, 0.30), peak_tick + dt_ls, 52, 'spd_aim+0.26'),
                (clip_action_vec(1.0 * da_aim, 0.32), clip_action_vec(0.36 * da_fwd, 0.40), peak_tick + dt_ls, 52, 'spd_aim+0.36'),
                (clip_action_vec(1.0 * da_dyn_ls, 0.36), clip_action_vec(0.28 * da_fwd, 0.32), peak_tick + dt_dls, 55, 'spd_dyn+0.28'),
            ]
        elif kind in ('BALL_NET', 'OOB', 'MARGINAL'):
            # Exact target box is x in [2.0, 4.0], y in [-1.0, +1.0], center (3.0, 0.0)!
            land = e.get('first_landing_m') or [3.0, 0.0, 0.0]
            lx = land[0] if land[0] is not None and math.isfinite(land[0]) else 3.0
            ly = land[1] if land[1] is not None and math.isfinite(land[1]) else 0.0
            is_long = ((kind == 'OOB') and (lx > 3.85)) or ((kind == 'MARGINAL') and (lx > 3.65))
            is_short = (kind == 'BALL_NET') or (lx < 2.50)
            dn_y = -0.18 * float(np.clip(ly, -1.5, 1.5)) if abs(ly) > 0.65 else -0.08 * float(np.clip(ly, -1.0, 1.0))
            if is_short:
                dn_z = 0.14
            elif is_long:
                dn_z = -float(np.clip((lx - 3.05) / 2.57, 0.12, 0.34))
            else:
                dn_z = 0.04 if lx < 3.0 else -0.05
            tny1 = float(np.clip(n0_pos[1] + dn_y, -0.25, 0.25))
            tnz1 = float(np.clip(n0_pos[2] + dn_z, -0.24, 0.35))
            tny2 = float(np.clip(n0_pos[1] + 1.65 * dn_y, -0.32, 0.32))
            tnz2 = float(np.clip(n0_pos[2] + 1.50 * dn_z, -0.32, 0.42))
            da_aim1, dt_a1 = solve_nonlinear_ik(
                geom, racket_id, st[c_idx], v_in,
                target_yz=(0.0, 0.0), target_nyz=(tny1, tnz1),
                w_yz=80.0, w_n=18.0, w_dq=0.25, steps=35, dq_limit=0.28)
            da_aim2, dt_a2 = solve_nonlinear_ik(
                geom, racket_id, st[c_idx], v_in,
                target_yz=(0.0, 0.0), target_nyz=(tny2, tnz2),
                w_yz=80.0, w_n=22.0, w_dq=0.18, steps=35, dq_limit=0.36)
            if is_long:
                branches = [
                    (clip_action_vec(1.0 * da_aim1, 0.38), [0.0]*18, peak_tick + dt_a1, 50, f'{kind.lower()}_down_1.0x'),
                    (clip_action_vec(1.15 * da_aim2, 0.46), [0.0]*18, peak_tick + dt_a2, 50, f'{kind.lower()}_down_1.5x'),
                    (clip_action_vec(1.05 * da_aim2, 0.44), clip_action_vec(0.10 * da_fwd, 0.14), peak_tick + dt_a2, 50, f'{kind.lower()}_down+spd'),
                    (clip_action_vec(1.0 * da_aim1, 0.38), clip_action_vec(-0.06 * da_fwd, 0.10), peak_tick + dt_a1, 50, f'{kind.lower()}_down_soft'),
                ]
            else:
                branches = [
                    (clip_action_vec(1.0 * da_aim1, 0.34), [0.0]*18, peak_tick + dt_a1, 50, f'{kind.lower()}_aim_1.0x'),
                    (clip_action_vec(1.2 * da_aim2, 0.44), [0.0]*18, peak_tick + dt_a2, 50, f'{kind.lower()}_aim_1.6x'),
                    (clip_action_vec(1.0 * da_aim1, 0.34), clip_action_vec(0.16 * da_fwd, 0.20), peak_tick + dt_a1, 50, f'{kind.lower()}_aim+0.16'),
                    (clip_action_vec(1.1 * da_aim2, 0.42), clip_action_vec(0.24 * da_fwd, 0.28), peak_tick + dt_a2, 50, f'{kind.lower()}_aim+0.24'),
                ]

        for b_idx, (off, rmp, pt, wt, tag) in enumerate(branches[:branches_per_case]):
            out_cases.append(c_obj)
            out_offsets.append(off)
            out_ramps.append(rmp)
            out_peaks.append(float(pt))
            out_widths.append(float(wt))
            branch_meta.append(dict(
                nominal_index=i, case_id=c_obj['case_id'], failure_kind=kind,
                branch=b_idx, tag=tag, nominal_metric=float(metric), peak_tick=int(pt),
            ))

    spec = dict(
        mode='jacobian_probe',
        nominal_dir=str(nominal_dir),
        cases=out_cases,
        offsets=out_offsets,
        ramp_offsets=out_ramps,
        peak_ticks=out_peaks,
        width_ticks=out_widths,
        branch_meta=branch_meta,
        gate_mode='approach_gate',
        record_tape=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(spec, indent=2) + '\n')
    print(f'Wrote {len(out_cases)}-lane Jacobian probe spec -> {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)

    g = sub.add_parser('generate')
    g.add_argument('--count-a', type=int, default=64)
    g.add_argument('--count-b', type=int, default=64)
    g.add_argument('--seed-a', type=int, required=True)
    g.add_argument('--seed-b', type=int, required=True)
    g.add_argument('--g07-execution-root', type=Path, required=True)
    g.add_argument('--output', type=Path, required=True)

    p = sub.add_parser('plan')
    p.add_argument('--nominal-dir', type=Path, required=True)
    p.add_argument('--nominal-spec', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-cases', type=int, default=32)
    p.add_argument('--offset-cases', type=int, default=0)
    p.add_argument('--branches-per-case', type=int, default=4)
    p.add_argument('--no-marginal', action='store_true')

    args = parser.parse_args()
    if args.mode == 'generate':
        generate_training_spec(args.count_a, args.count_b, args.seed_a, args.seed_b,
                               args.g07_execution_root, args.output)
    else:
        plan_probes(args.nominal_dir, args.nominal_spec, args.output,
                    max_cases=args.max_cases, offset_cases=args.offset_cases,
                    branches_per_case=args.branches_per_case,
                    include_marginal=not args.no_marginal)


if __name__ == '__main__':
    main()
