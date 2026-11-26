#!/usr/bin/env python3
"""Unimodal Smooth-Manifold, Perturbation-Robust Distillation for SuperMimic actor.approach.

Upgrades over distill_native_approach.py:
1. Perturbation-robustness weighting: CEM teachers tagged rob_cnt==4 get 1.50x weight,
   rob_cnt==3 get 1.20x, while unverified legacy probe teachers get 0.85x and must pass
   strict continuous margins (spm, lm, nm, lx).
2. Unimodal Smooth-Manifold Outlier Rejection: computes leave-one-out RBF-predicted arm
   correction in serve space (iy, iz, ft, iv_x, iv_y, iv_z) for every Panel-B teacher,
   downweighting/dropping multimodal outliers whose action direction conflicts with neighbors
   (eliminates MSE averaging into FRAME_HIT!).
3. Clean Panel-A Protection WITHOUT L_inf hinge wall or boundary cliff:
   - Never overrides Panel-A successes.
   - Separate soft Panel-A retention multiplier (--pa-ret-mult) with zero Panel-A obs noise.
   - Recomputes exact target EarlyArmApproachGate(arm_prep) dynamically and preserves
     anchor_a's exact 0.25s Panel-A requests (so switching arm_prep e.g. 0.25s -> 0.38s
     keeps Panel A bit-aligned while unlocking +130ms causal arm prep on hard Panel B!).
4. Trains ONLY APPROACH_NAMES (actor.base, actor.recovery, normalizer verified bit-exact!),
   with K-fold cross-validated Model Soup (n_soup=5) and full audit JSON output.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from distill_native_approach import compute_delta_schedule, _load_lx_nm_map
from quadsmash.rl.supermimic_v1 import load_ppo_initialized_actor
from quadsmash.rl.supermimic_physical_loop_loading_v1 import APPROACH_NAMES
from quadsmash.rl.supermimic_policy_execution_v1 import contract as policy_contract, POSTHIT


def extract_frames_with_gate(tape, lane: int, *, min_gate: float = 0.02,
                             override_req_delta=None, arm_prep_gate_fn=None):
    ep0 = tape['episode'][:, lane] == 0
    hc = tape['had_contact'][:, lane]
    pres = tape['present'][:, lane]
    obs_all = tape['observation'][:, lane].float()
    tape_gate = tape['gate'][:, lane].float()
    gate = arm_prep_gate_fn(obs_all) if arm_prep_gate_fn is not None else tape_gate
    gate_max = gate.amax(-1)
    valid = ep0 & ~hc & pres & (gate_max > min_gate)
    idxs = valid.nonzero().flatten()
    if len(idxs) == 0:
        return None
    hc_idxs = (ep0 & hc).nonzero().flatten()
    c_tick = int(hc_idxs[0]) if len(hc_idxs) else int(idxs[-1]) + 1
    dist_to_c = (c_tick - idxs).clamp_min(0)
    weights = torch.where(dist_to_c <= 15, 4.5, torch.where(dist_to_c <= 30, 2.0, 1.0)).float()
    req = tape['request'][idxs, lane].float()
    if override_req_delta is not None:
        req = (req + gate[idxs] * override_req_delta[idxs]).clamp(-1.0, 1.0)
    return dict(
        obs=obs_all[idxs],
        req=req,
        gate=gate[idxs],
        weight=weights,
        dist_to_c=dist_to_c,
    )


def compute_contact_arm_delta(ep, base_actor, device):
    """Mean 6-D arm delta (req - base_actor(obs)) over the last 15 pre-contact ticks."""
    with torch.no_grad():
        obs = ep['obs'].to(device)
        mod = base_actor.actor
        nobs = mod.normalizer(obs)
        base_req = (mod.base(nobs) + base_actor.approach_action_gate(obs) * mod.approach(nobs) * mod.approach_scale).clamp(-1.0, 1.0)
        d = (ep['req'].to(device) - base_req)[:, 12:18]
        m = (ep['dist_to_c'].to(device) <= 15) & (ep['gate'][:, 12:18].to(device).amax(-1) > 0.05)
        if not m.any():
            m = ep['gate'][:, 12:18].to(device).amax(-1) > 0.05
        return d[m].mean(0).cpu().numpy() if m.any() else np.zeros(6, dtype=np.float64)


def load_spec_map(spec_paths):
    mp = {}
    for sp in spec_paths:
        p = Path(sp)
        if not p.exists():
            continue
        data = json.loads(p.read_text())
        for c in data.get('cases', []):
            if isinstance(c, dict) and 'case_id' in c and 'prediction_only' in c:
                mp[c['case_id']] = c
    return mp


def apply_smooth_manifold_filter(teachers, base_actor, spec_map, device, *,
                                 bw: float = 1.15, tau: float = 0.28,
                                 max_resid_probe: float = 0.52, max_resid_cem: float = 0.85):
    """Leave-one-out smooth RBF manifold consistency weighting & multimodal outlier rejection."""
    pb_entries = [t for t in teachers if not t['case_id'].startswith('A-') and t['case_id'] in spec_map]
    if len(pb_entries) < 8:
        print(f'[Manifold] Only {len(pb_entries)} Panel-B teachers with specs; skipping RBF filter.')
        return teachers, dict(checked=len(pb_entries), dropped=0)

    X, D, auth = [], [], []
    for t in pb_entries:
        c = spec_map[t['case_id']]
        ip = c['prediction_only']['intercept_position_m']
        iv = c['prediction_only']['incoming_velocity_mps']
        ft = float(c['prediction_only']['flight_time_s'])
        X.append([float(ip[1]), float(ip[2]), ft, float(iv[0]), float(iv[1]), float(iv[2])])
        D.append(compute_contact_arm_delta(t['teacher_ep'], base_actor, device))
        rc = t.get('rob_cnt', 0)
        auth.append(2.5 if rc == 4 else (1.8 if rc == 3 else 0.85))
    X = np.array(X, dtype=np.float64)
    D = np.array(D, dtype=np.float64)
    auth = np.array(auth, dtype=np.float64)
    mu_x = X.mean(0); sd_x = X.std(0) + 1e-6
    Xn = (X - mu_x) / sd_x

    resids = np.zeros(len(pb_entries), dtype=np.float64)
    dropped_cids = set()
    kept_resids = []
    for i, t in enumerate(pb_entries):
        d2 = ((Xn - Xn[i:i + 1]) ** 2).sum(axis=1)
        w = np.exp(-0.5 * d2 / (bw ** 2)) * auth
        w[i] = 0.0
        if w.sum() < 1e-6:
            resids[i] = 0.0
            continue
        d_hat = (w[:, None] * D).sum(0) / w.sum()
        r = float(np.linalg.norm(D[i] - d_hat))
        resids[i] = r
        rc = t.get('rob_cnt', 0)
        max_r = max_resid_cem if rc >= 3 else max_resid_probe
        if r > max_r:
            dropped_cids.add(t['case_id'])
        else:
            # Smooth Cauchy consistency weight
            cw = 1.0 / (1.0 + (r / tau) ** 2)
            t['teacher_ep']['weight'] = t['teacher_ep']['weight'] * float(cw)
            if t['nom_paired_ep'] is not None:
                t['nom_paired_ep']['weight'] = t['nom_paired_ep']['weight'] * float(cw)
            t['manifold_resid'] = r
            kept_resids.append(r)

    filtered = [t for t in teachers if t['case_id'] not in dropped_cids]
    info = dict(
        checked=len(pb_entries),
        dropped=len(dropped_cids),
        raw_med_resid=float(np.median(resids)),
        raw_p90_resid=float(np.percentile(resids, 90)),
        kept_med_resid=float(np.median(kept_resids)) if kept_resids else 0.0,
    )
    print(f"[Manifold] Checked {info['checked']} PB teachers | dropped={info['dropped']} multimodal outliers "
          f"| med_resid={info['raw_med_resid']:.3f} -> kept_med={info['kept_med_resid']:.3f} (p90_raw={info['raw_p90_resid']:.3f})",
          flush=True)
    return filtered, info


def compute_smooth_box_alpha(iy: float, iz: float, ft: float,
                             y_in: float = 0.105, dy: float = 0.025,
                             z_mid: float = 1.800, z_half: float = 0.045, dz: float = 0.018,
                             ft_mid: float = 0.750, ft_half: float = 0.065, dft: float = 0.022) -> float:
    uy = max(0.0, abs(iy) - y_in) / dy
    uz = max(0.0, abs(iz - z_mid) - z_half) / dz
    ut = max(0.0, abs(ft - ft_mid) - ft_half) / dft
    u = min(1.0, max(uy, uz, ut))
    return float(u * u * (3.0 - 2.0 * u))


def pack_tensors(entries, is_teacher: bool, base_actor, target_actor, device,
                 mode: str = 'both', pa_ret_mult: float = 2.8, pb_ret_mult: float = 1.0,
                 anchor_a_actor=None, recompute_pa_from_anchor: bool = False,
                 spec_map=None, smooth_anchor_box: bool = False):
    obs_l, req_l, gate_l, w_l, is_pa_l, alpha_l = [], [], [], [], [], []
    for e in entries:
        if is_teacher:
            eps = []
            if mode in ('both', 'teacher'):
                eps.append(e['teacher_ep'])
            if mode in ('both', 'nominal') and e['nom_paired_ep'] is not None:
                eps.append(e['nom_paired_ep'])
            for ep in eps:
                obs_l.append(ep['obs'])
                req_l.append(ep['req'])
                gate_l.append(ep['gate'])
                w_l.append(ep['weight'])
                is_pa_l.append(torch.full((len(ep['obs']),), 1.0 if e['case_id'].startswith('A-') else 0.0))
                alpha_l.append(torch.ones((len(ep['obs']),), dtype=torch.float32))
        else:
            obs_l.append(e['obs'])
            req_l.append(e['req'])
            gate_l.append(e['gate'])
            is_a = e['case_id'].startswith('A-')
            m = pa_ret_mult if is_a else pb_ret_mult
            w_l.append(e['weight'] * e['arm_weight'] * m)
            is_pa_l.append(torch.full((len(e['obs']),), 1.0 if is_a else 0.0))
            if is_a:
                a_val = 0.0
            elif smooth_anchor_box and spec_map is not None and e['case_id'] in spec_map:
                c = spec_map[e['case_id']]
                ip = c['prediction_only']['intercept_position_m']
                ft = float(c['prediction_only']['flight_time_s'])
                a_val = compute_smooth_box_alpha(float(ip[1]), float(ip[2]), ft)
            else:
                a_val = 1.0
            alpha_l.append(torch.full((len(e['obs']),), a_val, dtype=torch.float32))
    if not obs_l:
        return None
    obs = torch.cat(obs_l, dim=0).to(device)
    req = torch.cat(req_l, dim=0).to(device)
    w = torch.cat(w_l, dim=0).to(device)
    is_pa = torch.cat(is_pa_l, dim=0).to(device) > 0.5
    alpha = torch.cat(alpha_l, dim=0).to(device)
    with torch.no_grad():
        mod = target_actor.actor
        norm_obs = mod.normalizer(obs)
        base_out = mod.base(norm_obs)
        agate = target_actor.approach_action_gate(obs)
        if recompute_pa_from_anchor and (not is_teacher) and (anchor_a_actor is not None):
            # Smoothly anchor Panel A AND inner-box Panel B to anchor_a (eliminates identical-coord conflict!)
            blend_m = is_pa | (alpha < 0.999)
            if blend_m.any():
                mod_a = anchor_a_actor.actor
                req_a = (mod_a.base(norm_obs[blend_m])
                         + anchor_a_actor.approach_action_gate(obs[blend_m]) * mod_a.approach(norm_obs[blend_m]) * mod_a.approach_scale).clamp(-1.0, 1.0)
                a_b = alpha[blend_m][:, None]
                req = req.clone()
                req[blend_m] = (1.0 - a_b) * req_a + a_b * req[blend_m]
        base_pol = (base_out + base_actor.approach_action_gate(obs) * base_actor.actor.approach(norm_obs) * base_actor.actor.approach_scale).clamp(-1.0, 1.0)
    return dict(norm=norm_obs, base=base_out, agate=agate, req=req, base_pol=base_pol, w=w, is_pa=is_pa)


@torch.no_grad()
def eval_split(actor, pack):
    if pack is None:
        return {}
    mod = actor.actor
    pred = (pack['base'] + pack['agate'] * mod.approach(pack['norm']) * mod.approach_scale).clamp(-1.0, 1.0)
    m_arm = pack['agate'][:, 12:18] > 0.05
    m_leg = pack['agate'][:, :12] > 0.05
    d_arm = (pred - pack['req'])[:, 12:18][m_arm]
    d_leg = (pred - pack['req'])[:, :12][m_leg]
    out = dict(
        rms_arm=float(d_arm.square().mean().sqrt()) if len(d_arm) else 0.0,
        q95_arm=float(d_arm.abs().quantile(0.95)) if len(d_arm) else 0.0,
        max_arm=float(d_arm.abs().max()) if len(d_arm) else 0.0,
        rms_leg=float(d_leg.square().mean().sqrt()) if len(d_leg) else 0.0,
    )
    u = (pred - pack['base_pol'])[:, 12:18][m_arm]
    v = (pack['req'] - pack['base_pol'])[:, 12:18][m_arm]
    if len(u) and v.norm() > 1e-6:
        out['cos_delta'] = float((u * v).sum() / (u.norm() * v.norm() + 1e-8))
        out['mag_ratio'] = float(u.norm() / (v.norm() + 1e-8))
    if pack['is_pa'].any():
        pm = pack['is_pa'][:, None] & m_arm
        dp = (pred - pack['req'])[:, 12:18][pm]
        if len(dp):
            out['pa_rms'] = float(dp.square().mean().sqrt())
            out['pa_q95'] = float(dp.abs().quantile(0.95))
            out['pa_max'] = float(dp.abs().max())
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-checkpoint', type=Path, required=True, help='Starting checkpoint (e.g. final_best.pt or H8d)')
    p.add_argument('--anchor-a-checkpoint', type=Path, default=None, help='Panel-A anchor checkpoint (defaults to base)')
    p.add_argument('--arm-prep-seconds', type=float, default=None, help='Target arm_preparation_seconds (defaults to base)')
    p.add_argument('--nominal-dirs', type=Path, nargs='+', required=True)
    p.add_argument('--retention-exclude-dirs', type=Path, nargs='*', default=[])
    p.add_argument('--pa-only-retention-dirs', type=Path, nargs='*', default=[],
                   help='Nominal dirs from which ONLY Panel-A retentions are kept (prevents off-policy FB PB retentions fighting H8d!)')
    p.add_argument('--probe-dirs', type=Path, nargs='+', required=True)
    p.add_argument('--probe-specs', type=Path, nargs='+', required=True)
    p.add_argument('--case-specs', type=Path, nargs='+', required=True, help='Spec JSONs containing prediction_only metadata')
    p.add_argument('--forbid-specs', type=Path, nargs='*', default=[], help='Val/test specs that MUST have zero case_id overlap')
    p.add_argument('--output', type=Path, required=True)
    # Teacher quality & manifold filters
    p.add_argument('--min-spm', type=float, default=0.22)
    p.add_argument('--min-lm', type=float, default=0.12)
    p.add_argument('--min-nm', type=float, default=0.018)
    p.add_argument('--max-lx', type=float, default=3.70)
    p.add_argument('--max-teacher-delta', type=float, default=0.72)
    p.add_argument('--nominal-pair-weight', type=float, default=0.85)
    p.add_argument('--manifold-bw', type=float, default=1.15)
    p.add_argument('--manifold-tau', type=float, default=0.28)
    p.add_argument('--max-resid-probe', type=float, default=0.52)
    p.add_argument('--max-resid-cem', type=float, default=0.85)
    p.add_argument('--high-z-mult', type=float, default=1.45)
    p.add_argument('--wide-y-mult', type=float, default=1.45)
    # Loss weights (ZERO L_inf hinge wall by default!)
    p.add_argument('--retention-weight', type=float, default=2.6)
    p.add_argument('--pa-ret-mult', type=float, default=2.8)
    p.add_argument('--pb-ret-mult', type=float, default=1.0)
    p.add_argument('--pa-hinge-weight', type=float, default=0.0)
    p.add_argument('--pa-hinge-margin', type=float, default=0.025)
    p.add_argument('--prox-weight', type=float, default=0.004)
    p.add_argument('--ball-tool-focus', type=float, default=2.0)
    p.add_argument('--pb-obs-noise', type=float, default=0.010)
    p.add_argument('--epochs', type=int, default=260)
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--lr', type=float, default=4.5e-4)
    p.add_argument('--n-soup', type=int, default=5)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    ck_base = torch.load(args.base_checkpoint, map_location='cpu', weights_only=False)
    ck_a = torch.load(args.anchor_a_checkpoint, map_location='cpu', weights_only=False) if args.anchor_a_checkpoint else ck_base
    src_path = Path(ck_base['metadata']['init_policy_checkpoint']['path'])
    base_arm_prep = ck_base['metadata'].get('policy_execution', {}).get('arm_preparation_seconds', 0.25)
    anchor_a_prep = ck_a['metadata'].get('policy_execution', {}).get('arm_preparation_seconds', 0.25)
    target_arm_prep = args.arm_prep_seconds if args.arm_prep_seconds is not None else base_arm_prep

    def make_actor(ck, ap):
        a, _ = load_ppo_initialized_actor(src_path, device)
        a.load_state_dict(ck['actor'], strict=True)
        if ap is not None:
            a.configure_approach_gate(ap)
        a.eval()
        return a

    base_actor = make_actor(ck_base, base_arm_prep)
    anchor_a_actor = make_actor(ck_a, anchor_a_prep)
    target_gate_actor = make_actor(ck_base, target_arm_prep)
    cpu_gate_actor, _ = load_ppo_initialized_actor(src_path, torch.device('cpu'))
    cpu_gate_actor.load_state_dict(ck_base['actor'], strict=True)
    cpu_gate_actor.configure_approach_gate(target_arm_prep)
    cpu_gate_actor.eval()

    def target_gate_cpu(obs_cpu):
        with torch.no_grad():
            return cpu_gate_actor.approach_action_gate(obs_cpu)

    # Leakage guard
    forbidden = set()
    for fsp in args.forbid_specs:
        if Path(fsp).exists():
            d = json.loads(Path(fsp).read_text())
            for c in d.get('cases', d if isinstance(d, list) else []):
                if isinstance(c, dict) and 'case_id' in c:
                    forbidden.add(c['case_id'])

    spec_map = load_spec_map(list(args.case_specs) + list(args.probe_specs))
    exclude_set = {Path(d).resolve() for d in args.retention_exclude_dirs}
    pa_only_ret_set = {Path(d).resolve() for d in args.pa_only_retention_dirs}

    # 1. Load nominal tapes & retentions
    retentions = {}
    nom_tapes = {}
    nom_ok_rows = {}
    for ndir in args.nominal_dirs:
        ndir = Path(ndir)
        skip_ret = ndir.resolve() in exclude_set
        pa_only_ret = ndir.resolve() in pa_only_ret_set
        summary = json.loads((ndir / 'probe_summary.json').read_text())
        lx_nm = _load_lx_nm_map(ndir)
        blob = torch.load(ndir / 'probe_tape.pt', map_location='cpu', weights_only=True)
        tape = blob['tape']
        start_tick = int(blob['start_capture_tick'])
        for i, row in enumerate(summary['rows']):
            cid = row['case_id']
            if cid in forbidden:
                raise RuntimeError(f'LEAKAGE DETECTED: {cid} in {ndir}!')
            lx, nm = lx_nm.get(i, (None, None))
            row = dict(row, lx=lx, nm=nm)
            sm = row['string_margin_m'] or 0.0
            spm = row['speed_margin_mps'] or 0.0
            lm = row.get('landing_margin_m') or 0.0
            score = float(row['margin_score']) + 0.5 * min(spm, 1.0) + 10.0 * min(max(sm, 0.0), 0.05)
            nom_tapes[cid] = (tape, i, start_tick, row, score)
            if row['success'] and not row['robot_net'] and not row['low_speed']:
                nom_ok_rows[cid] = row
                if skip_ret or (pa_only_ret and not cid.startswith('A-')):
                    continue
                ep = extract_frames_with_gate(tape, i, arm_prep_gate_fn=target_gate_cpu)
                if ep is not None:
                    sm_w = max(0.30, min(1.0, (sm - 0.008) / 0.025))
                    sp_w = max(0.35, min(1.0, spm / 0.35))
                    lm_w = max(0.35, min(1.0, lm / 0.22))
                    lx_w = max(0.25, min(1.0, (3.95 - lx) / 0.35)) if lx is not None else 1.0
                    ep['arm_weight'] = torch.full_like(ep['weight'], sm_w * sp_w * lm_w * lx_w)
                    prev = retentions.get(cid)
                    if prev is None or score > prev['score']:
                        retentions[cid] = dict(case_id=cid, score=score, row=row, source=str(ndir), **ep)

    # 2. Load probe + CEM teachers with strict margin & robustness priority!
    teachers = {}
    for pdir, pspec_path in zip(args.probe_dirs, args.probe_specs, strict=True):
        pdir = Path(pdir)
        pspec = json.loads(Path(pspec_path).read_text())
        summary = json.loads((pdir / 'probe_summary.json').read_text())
        lx_nm = _load_lx_nm_map(pdir)
        blob = torch.load(pdir / 'probe_tape.pt', map_location='cpu', weights_only=True)
        tape = blob['tape']
        meta = summary.get('branch_meta') or [None] * len(summary['rows'])
        cases = pspec.get('cases') or [None] * len(summary['rows'])
        offsets = pspec['offsets']
        ramps = pspec.get('ramp_offsets') or [None] * len(offsets)
        peaks = pspec['peak_ticks']
        widths = pspec['width_ticks']

        for i, row in enumerate(summary['rows']):
            cid = row['case_id']
            if cid in forbidden:
                raise RuntimeError(f'LEAKAGE DETECTED: {cid} in {pdir}!')
            if not (row['success'] and row['string'] and not row['robot_net']
                    and not row['ball_net'] and not row['low_speed']):
                continue
            # NEVER override a Panel-A success!
            if cid.startswith('A-') and cid in nom_ok_rows:
                continue
            lx, nm = lx_nm.get(i, (None, None))
            sm = row['string_margin_m'] or 0.0
            spm = row['speed_margin_mps'] or 0.0
            lm = row.get('landing_margin_m') or 0.0
            m_info = meta[i] if isinstance(meta[i], dict) else {}
            rob_cnt = int(m_info.get('rob_cnt', 0))
            is_cem = (rob_cnt >= 3) or (m_info.get('tag') == 'cem_robust')
            # Strict margin gate (slightly gentler on 4/4-verified CEM teachers!)
            min_spm = args.min_spm * (0.75 if rob_cnt >= 4 else 1.0)
            min_lm = args.min_lm * (0.80 if rob_cnt >= 4 else 1.0)
            min_nm = args.min_nm * (0.75 if rob_cnt >= 4 else 1.0)
            if spm < min_spm or lm < min_lm or (nm is not None and nm < min_nm) or (lx is not None and lx > args.max_lx):
                continue
            off = np.array(offsets[i], dtype=np.float64)
            rmp = np.array(ramps[i], dtype=np.float64) if ramps[i] is not None else np.zeros(18)
            eff_max = float(np.abs(off + rmp).max())
            eff_l2 = float(np.linalg.norm(off) + np.linalg.norm(rmp))
            if eff_max > args.max_teacher_delta:
                continue

            ep_tea = extract_frames_with_gate(tape, i, arm_prep_gate_fn=target_gate_cpu)
            if ep_tea is None:
                continue
            c = cases[i] if (cases[i] is not None and 'prediction_only' in cases[i]) else spec_map.get(cid)
            iy = float(c['prediction_only']['intercept_position_m'][1]) if c else 0.0
            iz = float(c['prediction_only']['intercept_position_m'][2]) if c else 1.78
            rob_w = 1.50 if rob_cnt == 4 else (1.20 if rob_cnt == 3 else 0.85)
            z_w = 1.0 + (args.high_z_mult - 1.0) * min(1.5, max(0.0, (iz - 1.80) / 0.06))
            y_w = 1.0 + (args.wide_y_mult - 1.0) * min(1.5, max(0.0, (abs(iy) - 0.10) / 0.06))
            lx_w = 1.0 if (lx is None or lx <= 3.55) else max(0.55, 1.0 - 1.5 * (lx - 3.55))
            total_w = rob_w * z_w * y_w * lx_w
            ep_tea['weight'] = ep_tea['weight'] * total_w

            ep_nom = None
            if cid in nom_tapes and args.nominal_pair_weight > 0 and (not is_cem or abs(target_arm_prep - base_arm_prep) < 1e-4):
                ntape, nlane, nstart, _, _ = nom_tapes[cid]
                delta_sched = compute_delta_schedule(
                    ntape['observation'].shape[0], nstart, offsets[i], ramps[i], peaks[i], widths[i])
                ep_nom = extract_frames_with_gate(ntape, nlane, override_req_delta=delta_sched, arm_prep_gate_fn=target_gate_cpu)
                if ep_nom is not None:
                    ep_nom['weight'] = ep_nom['weight'] * args.nominal_pair_weight * total_w

            # Rank: CEM robust >=3/4 ALWAYS beats unverified probe for same case!
            rank = (1 if rob_cnt >= 3 else 0, rob_cnt, -0.5 * eff_l2 + 0.35 * min(spm, 1.2) + 0.8 * min(lm, 0.45) + 3.0 * min(sm, 0.04))
            prev = teachers.get(cid)
            if prev is None or rank > prev['rank']:
                teachers[cid] = dict(
                    case_id=cid, rank=rank, rob_cnt=rob_cnt, is_cem=is_cem,
                    eff_l2=eff_l2, row=row, source=str(pdir),
                    teacher_ep=ep_tea, nom_paired_ep=ep_nom,
                )
                retentions.pop(cid, None)

    teacher_list = list(teachers.values())
    ret_list = list(retentions.values())
    n_cem = sum(1 for t in teacher_list if t['rob_cnt'] >= 3)
    n_pa_ret = sum(1 for r in ret_list if r['case_id'].startswith('A-'))
    n_pb_ret = len(ret_list) - n_pa_ret
    print(f'[SmoothDistill] Loaded {len(teacher_list)} teachers (CEM_robust={n_cem}, probe={len(teacher_list)-n_cem}) '
          f'| {len(ret_list)} retentions (PA={n_pa_ret}, PB={n_pb_ret}) | target_arm_prep={target_arm_prep:.2f}s', flush=True)

    # 3. Unimodal Smooth-Manifold Outlier Rejection
    teacher_list, manifold_info = apply_smooth_manifold_filter(
        teacher_list, base_actor, spec_map, device,
        bw=args.manifold_bw, tau=args.manifold_tau,
        max_resid_probe=args.max_resid_probe, max_resid_cem=args.max_resid_cem)

    # Column-wise proximal mask (free ball/tool + arm pos/vel columns!)
    w0_col_mult = torch.full((794,), float(args.ball_tool_focus), dtype=torch.float32, device=device)
    for f in range(11):
        b = f * 69
        w0_col_mult[b + 0:b + 9] = 0.70    # root vel/grav
        w0_col_mult[b + 21:b + 27] = 0.35  # arm joint pos!
        w0_col_mult[b + 39:b + 45] = 0.35  # arm joint vel!
        w0_col_mult[b + 45:b + 57] = 0.30  # tool pose/vel!
        w0_col_mult[b + 61:b + 67] = 0.08  # ball pos/vel!
    w0_col_mult[759:776] = 0.35

    # K-fold cross-validated Model Soup!
    rng = np.random.default_rng(args.seed)
    t_perm = rng.permutation(len(teacher_list))
    r_perm = rng.permutation(len(ret_list))
    t_folds = np.array_split(t_perm, max(1, args.n_soup))
    r_folds = np.array_split(r_perm, max(1, args.n_soup))

    init_state = {k: v.detach().cpu().clone() for k, v in ck_base['actor'].items()}
    anchor_cpu = {k: init_state[k].clone() for k in APPROACH_NAMES}
    recomp_pa = (args.anchor_a_checkpoint is not None) or (abs(target_arm_prep - anchor_a_prep) > 1e-4)

    member_states = []
    fold_stats = []

    for f_idx in range(args.n_soup):
        torch.manual_seed(args.seed + 1009 * f_idx)
        np.random.seed(args.seed + 1009 * f_idx)
        val_t_idx = set(t_folds[f_idx].tolist()) if args.n_soup > 1 else set(t_perm[:max(1, int(0.15 * len(t_perm)))].tolist())
        val_r_idx = set(r_folds[f_idx].tolist()) if args.n_soup > 1 else set(r_perm[:max(1, int(0.15 * len(r_perm)))].tolist())

        trn_t = [teacher_list[i] for i in range(len(teacher_list)) if i not in val_t_idx]
        val_t = [teacher_list[i] for i in range(len(teacher_list)) if i in val_t_idx]
        trn_r = [ret_list[i] for i in range(len(ret_list)) if i not in val_r_idx]
        val_r = [ret_list[i] for i in range(len(ret_list)) if i in val_r_idx]

        p_trn_t = pack_tensors(trn_t, True, base_actor, target_gate_actor, device, mode='both')
        p_val_t = pack_tensors(val_t, True, base_actor, target_gate_actor, device, mode='teacher')
        p_trn_r = pack_tensors(trn_r, False, base_actor, target_gate_actor, device,
                               pa_ret_mult=args.pa_ret_mult, pb_ret_mult=args.pb_ret_mult,
                               anchor_a_actor=anchor_a_actor, recompute_pa_from_anchor=recomp_pa,
                               spec_map=spec_map, smooth_anchor_box=(args.anchor_a_checkpoint is not None))
        p_val_r = pack_tensors(val_r, False, base_actor, target_gate_actor, device,
                               pa_ret_mult=args.pa_ret_mult, pb_ret_mult=args.pb_ret_mult,
                               anchor_a_actor=anchor_a_actor, recompute_pa_from_anchor=recomp_pa,
                               spec_map=spec_map, smooth_anchor_box=(args.anchor_a_checkpoint is not None))

        student = make_actor(ck_base, target_arm_prep)
        for name, param in student.named_parameters():
            param.requires_grad_(name in APPROACH_NAMES)
        trainable = [(n, param) for n, param in student.named_parameters() if param.requires_grad]
        anchor_dev = {n: anchor_cpu[n].to(device) for n, _ in trainable}
        opt = torch.optim.AdamW([param for _, param in trainable], lr=args.lr, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.04)

        mod = student.actor
        Nt = len(p_trn_t['norm'])
        Nr = len(p_trn_r['norm'])
        best_sc = float('inf')
        best_st = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
        best_ep = 0

        for ep in range(1, args.epochs + 1):
            student.train()
            pidx_r = torch.randperm(Nr, device=device)
            n_steps = max(1, (Nr + args.batch_size - 1) // args.batch_size)
            for s in range(n_steps):
                opt.zero_grad(set_to_none=True)
                it = torch.randint(0, Nt, (min(Nt, args.batch_size),), device=device)
                ir = pidx_r[s * args.batch_size:(s + 1) * args.batch_size]
                if len(ir) < 64:
                    ir = pidx_r[:min(Nr, args.batch_size)]

                # Teacher loss (mild PB obs noise only!)
                inp_t = p_trn_t['norm'][it]
                if args.pb_obs_noise > 0:
                    inp_t = inp_t + torch.randn_like(inp_t) * args.pb_obs_noise
                pred_t = (p_trn_t['base'][it] + p_trn_t['agate'][it] * mod.approach(inp_t) * mod.approach_scale).clamp(-1.0, 1.0)
                gm_t = (p_trn_t['agate'][it] > 0.02).float()
                jw = torch.ones_like(gm_t); jw[:, :12] = 2.0; jw[:, 12:18] = 3.5
                mw_t = gm_t * jw
                sq_t = ((pred_t - p_trn_t['req'][it]).square() * mw_t).sum(-1) / mw_t.sum(-1).clamp_min(1.0)
                loss_t = (sq_t * p_trn_t['w'][it]).sum() / p_trn_t['w'][it].sum().clamp_min(1.0)

                # Retention loss (ZERO obs noise on Panel A!)
                inp_r = p_trn_r['norm'][ir]
                if args.pb_obs_noise > 0:
                    ns_r = torch.where(p_trn_r['is_pa'][ir], 0.0, args.pb_obs_noise * 0.5)[:, None]
                    inp_r = inp_r + torch.randn_like(inp_r) * ns_r
                pred_r = (p_trn_r['base'][ir] + p_trn_r['agate'][ir] * mod.approach(inp_r) * mod.approach_scale).clamp(-1.0, 1.0)
                gm_r = (p_trn_r['agate'][ir] > 0.02).float()
                jw_r = torch.ones_like(gm_r); jw_r[:, :12] = 2.5; jw_r[:, 12:18] = 3.5
                mw_r = gm_r * jw_r
                sq_r = ((pred_r - p_trn_r['req'][ir]).square() * mw_r).sum(-1) / mw_r.sum(-1).clamp_min(1.0)
                loss_r = (sq_r * p_trn_r['w'][ir]).sum() / p_trn_r['w'][ir].sum().clamp_min(1.0)

                loss = loss_t + args.retention_weight * loss_r

                if args.pa_hinge_weight > 0 and p_trn_r['is_pa'][ir].any():
                    pa_m = p_trn_r['is_pa'][ir]
                    pm = (p_trn_r['agate'][ir][pa_m, 12:18] > 0.02).float()
                    exc = F.relu((pred_r[pa_m, 12:18] - p_trn_r['req'][ir][pa_m, 12:18]).abs() - args.pa_hinge_margin)
                    loss = loss + args.pa_hinge_weight * (exc.square() * pm).sum() / pm.sum().clamp_min(1.0)

                if args.prox_weight > 0:
                    pt = []
                    for name, param in trainable:
                        if name == 'actor.approach.0.weight':
                            pt.append(((param - anchor_dev[name]).square() * w0_col_mult[None, :]).mean())
                        else:
                            pt.append(0.25 * (param - anchor_dev[name]).square().mean())
                    loss = loss + args.prox_weight * (sum(pt) / len(pt))

                loss.backward()
                torch.nn.utils.clip_grad_norm_([param for _, param in trainable], 2.0)
                opt.step()
            sched.step()

            if ep % 10 == 0 or ep == args.epochs:
                student.eval()
                mt = eval_split(student, p_val_t)
                mr = eval_split(student, p_val_r)
                sc = (mt['rms_arm'] ** 2) + 2.2 * (mr.get('pa_rms', mr['rms_arm']) ** 2) + 0.8 * (mr['rms_arm'] ** 2)
                if sc < best_sc:
                    best_sc = sc
                    best_st = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
                    best_ep = ep

        student.load_state_dict({k: v.to(device) for k, v in best_st.items()}, strict=True)
        student.eval()
        mt = eval_split(student, p_val_t)
        mr = eval_split(student, p_val_r)
        fold_stats.append(dict(fold=f_idx, best_ep=best_ep, val_tea=mt, val_ret=mr))
        member_states.append(best_st)
        print(f"  [fold {f_idx}] best@{best_ep:3d} | val_tea rms={mt['rms_arm']:.4f} cos={mt.get('cos_delta',0):+.3f} mag={mt.get('mag_ratio',0):.3f} "
              f"| val_ret rms={mr['rms_arm']:.4f} PA_rms={mr.get('pa_rms',0):.4f} PA_q95={mr.get('pa_q95',0):.4f} PA_max={mr.get('pa_max',0):.3f}",
              flush=True)

    # Model Soup strictly over APPROACH_NAMES; verify frozen keys bit-exact!
    soup_actor = copy.deepcopy(init_state)
    for k in APPROACH_NAMES:
        soup_actor[k] = sum(st[k].float() for st in member_states) / len(member_states)
        soup_actor[k] = soup_actor[k].to(init_state[k].dtype)
    for k, v in init_state.items():
        if k not in APPROACH_NAMES and torch.is_tensor(v):
            assert torch.equal(soup_actor[k], v), f'Frozen parameter changed: {k}'

    out_ckpt = copy.deepcopy(ck_base)
    out_ckpt['actor'] = soup_actor
    out_ckpt['metadata']['policy_execution'] = policy_contract(POSTHIT, arm_preparation_seconds=float(target_arm_prep))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ckpt, args.output)

    meta_out = dict(
        base=str(args.base_checkpoint),
        anchor_a=str(args.anchor_a_checkpoint or args.base_checkpoint),
        arm_prep_seconds=float(target_arm_prep),
        teachers=len(teacher_list),
        cem_robust_teachers=sum(1 for t in teacher_list if t['rob_cnt'] >= 3),
        retentions=len(ret_list),
        pa_retentions=n_pa_ret,
        pb_retentions=n_pb_ret,
        manifold=manifold_info,
        fold_stats=fold_stats,
        mean_val_tea_cos=float(np.mean([f['val_tea'].get('cos_delta', 0.0) for f in fold_stats])),
        mean_val_tea_mag=float(np.mean([f['val_tea'].get('mag_ratio', 0.0) for f in fold_stats])),
        mean_pa_rms=float(np.mean([f['val_ret'].get('pa_rms', 0.0) for f in fold_stats])),
        mean_pa_q95=float(np.mean([f['val_ret'].get('pa_q95', 0.0) for f in fold_stats])),
    )
    args.output.with_suffix('.smooth.json').write_text(json.dumps(meta_out, indent=2) + '\n')
    print(f'[SmoothDistill] Saved {args.output} | mean_cos={meta_out["mean_val_tea_cos"]:+.3f} '
          f'mean_mag={meta_out["mean_val_tea_mag"]:.3f} mean_PA_q95={meta_out["mean_pa_q95"]:.4f}', flush=True)


if __name__ == '__main__':
    main()
