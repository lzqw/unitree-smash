#!/usr/bin/env python3
"""Closed-loop DAgger distillation with causal-shortcut elimination and tight 18-DoF retention.

Key improvements over naive teacher BC:
1. Causal-shortcut elimination: supervises BOTH the nominal observation obs_nom ->
   (base_action(obs_nom) + gate(obs_nom) * delta_teacher) AND the closed-loop teacher
   observation obs_teacher -> req_teacher, preventing the MLP from cheating by copying
   perturbed action history.
2. Minimum-effort teacher selection: among all verified safe-margin winning branches
   for a case, prefers the smallest-amplitude smooth correction so the learned policy
   surface stays smooth and generalizes cleanly.
3. Strict quadratic + outlier-aware 18-DoF retention anchor on ALL already-successful
   training cases so existing successes stay within millimeter tolerance.
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
from quadsmash.rl.supermimic_v1 import load_ppo_initialized_actor
from quadsmash.rl.supermimic_physical_loop_loading_v1 import APPROACH_NAMES


def compute_delta_schedule(num_ticks: int, start_tick: int, offset, ramp_offset, peak_tick: float, width_tick: float):
    ticks = torch.arange(start_tick, start_tick + num_ticks, dtype=torch.float32)
    rise = ((ticks - (peak_tick - width_tick)) / max(width_tick, 1.0)).clamp(0.0, 1.0)
    fall = ((ticks - peak_tick) / 16.0).clamp(0.0, 1.0)
    env_pos = torch.where(ticks <= peak_tick, torch.sin(0.5 * math.pi * rise).square(),
                          torch.cos(0.5 * math.pi * fall).square())
    off = torch.tensor(offset, dtype=torch.float32)
    delta = env_pos[:, None] * off[None, :]
    if ramp_offset is not None:
        rmp = torch.tensor(ramp_offset, dtype=torch.float32)
        ramp = ((ticks - (peak_tick - 18.0)) / 18.0).clamp(0.0, 1.0) * (ticks <= peak_tick + 4.0).float()
        delta = delta + ramp[:, None] * rmp[None, :]
    return delta  # [T, 18]


def extract_episode_frames(tape, lane: int, *, min_gate: float = 0.05, last_ticks_boost: int = 25,
                           override_req_delta=None):
    ep0 = tape['episode'][:, lane] == 0
    hc = tape['had_contact'][:, lane]
    pres = tape['present'][:, lane]
    gate = tape['gate'][:, lane].float()  # [T, 18]
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
        # Nominal obs + exact teacher open-loop delta gated by nominal gate -> eliminates action-history shortcut!
        req = (req + gate[idxs] * override_req_delta[idxs]).clamp(-1.0, 1.0)
    return dict(
        obs=tape['observation'][idxs, lane].float(),
        req=req,
        gate=gate[idxs],
        weight=weights,
    )


def _load_lx_nm_map(run_dir: Path):
    ep_path = run_dir / 'episodes.jsonl'
    lx_map = {}
    if not ep_path.exists():
        return lx_map
    for line in ep_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        eid = int(r['env_id'])
        lm = r.get('first_landing_m')
        lx = float(lm[0]) if lm and math.isfinite(lm[0]) else None
        nm = r.get('net_margin_m')
        nm_f = float(nm) if nm is not None and math.isfinite(nm) else None
        lx_map[eid] = (lx, nm_f)
    return lx_map


def load_datasets(nominal_dirs, probe_dirs, probe_specs, *,
                  max_teacher_delta: float = 0.55, nominal_pair_weight: float = 1.0,
                  round_retention_decay: float = 1.0, min_speed_margin: float = 0.0,
                  max_teacher_lx: float = 4.0, target_lx: float = 3.20,
                  min_net_margin: float = 0.0, min_landing_margin: float = 0.0,
                  retention_exclude_dirs=(),
                  fail_teacher_mult: float = 1.0, high_z_mult: float = 1.0,
                  wide_y_mult: float = 1.0, no_pa_override: bool = False):
    teachers = {}
    retentions = {}
    nom_tapes = {}  # case_id -> (tape, lane, start_tick, row, score)
    nom_ok_rows = {}  # case_id -> row (tracks nominal successes even for retention_exclude_dirs)
    exclude_set = {Path(d).resolve() for d in retention_exclude_dirs}

    n_rounds = len(nominal_dirs)
    for r_idx, ndir in enumerate(nominal_dirs):
        ndir = Path(ndir)
        skip_ret = ndir.resolve() in exclude_set
        round_w = round_retention_decay ** (n_rounds - 1 - r_idx)
        summary = json.loads((ndir / 'probe_summary.json').read_text())
        lx_nm = _load_lx_nm_map(ndir)
        blob = torch.load(ndir / 'probe_tape.pt', map_location='cpu', weights_only=True)
        tape = blob['tape']
        start_tick = int(blob['start_capture_tick'])
        for i, row in enumerate(summary['rows']):
            cid = row['case_id']
            lx, nm = lx_nm.get(i, (None, None))
            row = dict(row, lx=lx, nm=nm)
            sm = row['string_margin_m'] if row['string_margin_m'] is not None else 0.0
            spm = row['speed_margin_mps'] if row['speed_margin_mps'] is not None else 0.0
            lm = row.get('landing_margin_m') if row.get('landing_margin_m') is not None else 0.0
            score = float(row['margin_score']) + 0.5 * min(spm, 1.0) + 10.0 * min(max(sm, 0.0), 0.05)
            nom_tapes[cid] = (tape, i, start_tick, row, score)
            if row['success'] and not row['robot_net'] and not row['low_speed']:
                nom_ok_rows[cid] = row
                if skip_ret:
                    continue
                ep = extract_episode_frames(tape, i)
                if ep is not None:
                    # Margin-weighted arm retention: full anchor on high-margin centered successes,
                    # softer on razor-edge string/speed/backline (>3.65m) successes so they don't fight teachers.
                    sm_w = max(0.25, min(1.0, (sm - 0.010) / 0.025))
                    sp_w = max(0.30, min(1.0, spm / 0.35))
                    lm_w = max(0.30, min(1.0, lm / 0.25))
                    lx_w = max(0.20, min(1.0, (3.95 - lx) / 0.35)) if lx is not None else 1.0
                    ep['arm_weight'] = torch.full_like(ep['weight'], sm_w * sp_w * lm_w * lx_w * round_w)
                    ep['is_pa'] = torch.full_like(ep['weight'], 1.0 if cid.startswith('A-') else 0.0)
                    prev = retentions.get(cid)
                    if prev is None or score > prev['score']:
                        retentions[cid] = dict(case_id=cid, score=score, row=row, source=str(ndir), **ep)

    for pdir, pspec_path in zip(probe_dirs, probe_specs, strict=True):
        pdir = Path(pdir)
        pspec = json.loads(Path(pspec_path).read_text())
        summary = json.loads((pdir / 'probe_summary.json').read_text())
        lx_nm = _load_lx_nm_map(pdir)
        blob = torch.load(pdir / 'probe_tape.pt', map_location='cpu', weights_only=True)
        tape = blob['tape']
        start_tick = int(blob['start_capture_tick'])
        meta = summary.get('branch_meta') or [None] * len(summary['rows'])
        cases = pspec.get('cases') or [None] * len(summary['rows'])
        offsets = pspec['offsets']
        ramps = pspec.get('ramp_offsets') or [None] * len(offsets)
        peaks = pspec['peak_ticks']
        widths = pspec['width_ticks']

        for i, row in enumerate(summary['rows']):
            cid = row['case_id']
            if not (row['success'] and row['string'] and not row['robot_net']
                    and not row['ball_net'] and not row['low_speed']):
                continue
            lx, nm = lx_nm.get(i, (None, None))
            row = dict(row, lx=lx, nm=nm)
            if lx is not None and lx > max_teacher_lx:
                continue
            if nm is not None and nm < min_net_margin:
                continue
            off = np.array(offsets[i], dtype=np.float64)
            rmp = np.array(ramps[i], dtype=np.float64) if ramps[i] is not None else np.zeros(18)
            eff_max = float(np.abs(off + rmp).max())
            eff_l2 = float(np.linalg.norm(off) + np.linalg.norm(rmp))
            if eff_max > max_teacher_delta:
                continue
            sm = row['string_margin_m'] if row['string_margin_m'] is not None else 0.0
            spm = row['speed_margin_mps'] if row['speed_margin_mps'] is not None else 0.0
            if spm < min_speed_margin:
                continue
            lm = row.get('landing_margin_m') if row.get('landing_margin_m') is not None else 0.0
            if lm < min_landing_margin:
                continue
            # Quality tier: strongly prefer crisp court-centered safe-margin solutions
            lx_ok2 = (lx is None) or (2.50 <= lx <= 3.55)
            lx_ok1 = (lx is None) or (2.30 <= lx <= 3.70)
            tier = 2 if (spm >= 0.30 and sm >= 0.012 and lm >= 0.15 and lx_ok2) else (1 if (spm >= 0.15 and sm >= 0.006 and lm >= 0.05 and lx_ok1) else 0)
            lx_pen = (0.75 * max(0.0, lx - 3.40) + 0.30 * abs(lx - target_lx)) if lx is not None else 0.0
            sel_rank = (tier, -eff_l2 - lx_pen + 0.18 * min(spm, 1.2) + 2.5 * min(sm, 0.04) + 0.40 * min(lm, 0.40))
            was_nom_ok = cid in nom_ok_rows
            if was_nom_ok:
                if no_pa_override and cid.startswith('A-'):
                    continue
                old_r = nom_ok_rows[cid]
                old_lm = old_r.get('landing_margin_m') if old_r.get('landing_margin_m') is not None else 0.0
                old_lx = old_r.get('lx')
                # Override an already-successful nominal case if probe meaningfully centers lx or widens margin
                centered_long = (old_lx is not None and lx is not None and old_lx > 3.60 and lx <= 3.48 and spm >= 0.25 and (nm is None or nm >= 0.035))
                improved = centered_long or (row['margin_score'] >= old_r['margin_score'] + 0.06) or (lm >= old_lm + 0.12 and spm >= 0.20)
                if not (improved and eff_max <= 0.45):
                    continue

            ep_teacher = extract_episode_frames(tape, i)
            if ep_teacher is None:
                continue
            iz = 1.75
            iy = 0.0
            if cases[i] is not None and 'prediction_only' in cases[i]:
                iz = float(cases[i]['prediction_only']['intercept_position_m'][2])
                iy = float(cases[i]['prediction_only']['intercept_position_m'][1])
            # Soft court-centering + genuine-failure rescue + high-z + wide-y weighting
            lx_soft_w = 1.0 if (lx is None or lx <= 3.60) else max(0.55, 1.0 - 1.2 * (lx - 3.60))
            fail_w = 1.0 if was_nom_ok else float(fail_teacher_mult)
            z_w = 1.0 + (float(high_z_mult) - 1.0) * min(1.5, max(0.0, (iz - 1.80) / 0.06))
            y_w = 1.0 + (float(wide_y_mult) - 1.0) * min(1.5, max(0.0, (abs(iy) - 0.10) / 0.06))
            total_tw = lx_soft_w * fail_w * z_w * y_w
            ep_teacher['weight'] = ep_teacher['weight'] * total_tw

            # Build paired nominal-obs -> (nom_req + nom_gate * delta) frames to kill action-history shortcut!
            ep_nom_paired = None
            if cid in nom_tapes and nominal_pair_weight > 0:
                ntape, nlane, nstart, _, _ = nom_tapes[cid]
                delta_sched = compute_delta_schedule(
                    ntape['observation'].shape[0], nstart, offsets[i], ramps[i], peaks[i], widths[i])
                ep_nom_paired = extract_episode_frames(ntape, nlane, override_req_delta=delta_sched)
                if ep_nom_paired is not None:
                    ep_nom_paired['weight'] = ep_nom_paired['weight'] * nominal_pair_weight * total_tw

            prev = teachers.get(cid)
            if prev is None or sel_rank > prev['sel_rank']:
                teachers[cid] = dict(
                    case_id=cid, sel_rank=sel_rank, eff_max=eff_max, eff_l2=eff_l2,
                    row=row, meta=meta[i], source=str(pdir), iz=iz, was_nom_ok=was_nom_ok,
                    teacher_ep=ep_teacher, nom_paired_ep=ep_nom_paired,
                )
                retentions.pop(cid, None)

    return list(teachers.values()), list(retentions.values())


def concat_teacher_episodes(entries, device, *, mode='both'):
    obs_l, req_l, gate_l, w_l = [], [], [], []
    for e in entries:
        if mode in ('both', 'teacher') and e['teacher_ep'] is not None:
            ep = e['teacher_ep']
            obs_l.append(ep['obs']); req_l.append(ep['req']); gate_l.append(ep['gate']); w_l.append(ep['weight'])
        if mode in ('both', 'nominal') and e['nom_paired_ep'] is not None:
            ep = e['nom_paired_ep']
            obs_l.append(ep['obs']); req_l.append(ep['req']); gate_l.append(ep['gate']); w_l.append(ep['weight'])
    if not obs_l:
        return None
    return dict(
        obs=torch.cat(obs_l, dim=0).to(device),
        req=torch.cat(req_l, dim=0).to(device),
        gate=torch.cat(gate_l, dim=0).to(device),
        weight=torch.cat(w_l, dim=0).to(device),
    )


def concat_episodes(episodes, device):
    if not episodes:
        return None
    out = dict(
        obs=torch.cat([e['obs'] for e in episodes], dim=0).to(device),
        req=torch.cat([e['req'] for e in episodes], dim=0).to(device),
        gate=torch.cat([e['gate'] for e in episodes], dim=0).to(device),
        weight=torch.cat([e['weight'] for e in episodes], dim=0).to(device),
    )
    if 'arm_weight' in episodes[0]:
        out['arm_weight'] = torch.cat([e['arm_weight'] for e in episodes], dim=0).to(device)
    if 'is_pa' in episodes[0]:
        out['is_pa'] = torch.cat([e['is_pa'] for e in episodes], dim=0).to(device)
    return out


def active_loss(pred, target, gate, weight, *, arm_only: bool, quadratic: bool = False, beta: float = 0.015,
                arm_weight=None):
    sl = slice(12, 18) if arm_only else slice(0, 18)
    p = pred[:, sl]
    t = target[:, sl]
    mask = (gate[:, sl] > 0.02).float()
    if arm_weight is not None and not arm_only:
        # Keep full 1.0 weight on legs 0:12, apply margin-based scaling on arm 12:18
        joint_w = torch.ones_like(mask)
        joint_w[:, 12:18] = arm_weight[:, None]
        mask = mask * joint_w
    if quadratic:
        sq = (p - t).square() * mask
        per_frame = sq.sum(-1) / mask.sum(-1).clamp_min(1.0)
        return (per_frame * weight).sum() / weight.sum().clamp_min(1.0)
    err = F.smooth_l1_loss(p, t, beta=beta, reduction='none') * mask
    per_frame = err.sum(-1) / mask.sum(-1).clamp_min(1.0)
    return (per_frame * weight).sum() / weight.sum().clamp_min(1.0)


@torch.no_grad()
def eval_metrics(actor, batch, *, arm_only: bool, base_actor=None):
    if batch is None:
        return dict(mse=0.0, rms_arm=0.0, rms_leg=0.0, max_arm=0.0, cos_delta=0.0, mag_ratio=0.0)
    no_rec = torch.zeros(len(batch['obs']), dtype=torch.bool, device=batch['obs'].device)
    pred = actor.forward_with_recovery_gate(batch['obs'], no_rec, sample=False)[0].clamp(-1.0, 1.0)
    aw = batch.get('arm_weight') if not arm_only else None
    mse = float(active_loss(pred, batch['req'], batch['gate'], batch['weight'], arm_only=arm_only, quadratic=True, arm_weight=aw))
    diff = pred - batch['req']
    m_arm = batch['gate'][:, 12:18] > 0.05
    m_leg = batch['gate'][:, 0:12] > 0.05
    rms_arm = float(diff[:, 12:18][m_arm].square().mean().sqrt()) if m_arm.any() else 0.0
    max_arm = float(diff[:, 12:18][m_arm].abs().max()) if m_arm.any() else 0.0
    rms_leg = float(diff[:, 0:12][m_leg].square().mean().sqrt()) if m_leg.any() else 0.0
    cos_d, mag_r = 0.0, 0.0
    if base_actor is not None:
        base_pred = base_actor.forward_with_recovery_gate(batch['obs'], no_rec, sample=False)[0].clamp(-1.0, 1.0)
        d_true = (batch['req'] - base_pred)[:, 12:18][m_arm]
        d_pred = (pred - base_pred)[:, 12:18][m_arm]
        if d_true.norm() > 1e-4 and d_pred.norm() > 1e-6:
            cos_d = float(F.cosine_similarity(d_pred, d_true, dim=0))
            mag_r = float(d_pred.norm() / d_true.norm())
    return dict(mse=mse, rms_arm=rms_arm, rms_leg=rms_leg, max_arm=max_arm, cos_delta=cos_d, mag_ratio=mag_r)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-checkpoint', type=Path, required=True)
    parser.add_argument('--nominal-dirs', type=Path, nargs='+', required=True)
    parser.add_argument('--probe-dirs', type=Path, nargs='+', required=True)
    parser.add_argument('--probe-specs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--teacher-weight', type=float, default=1.0)
    parser.add_argument('--teacher-leg-weight', type=float, default=1.0)
    parser.add_argument('--retention-weight', type=float, default=12.0)
    parser.add_argument('--round-retention-decay', type=float, default=1.0)
    parser.add_argument('--panel-a-anchor-dirs', type=Path, nargs='+', default=[],
                        help='Nominal dirs whose first 128 (Panel A) lanes anchor directly to --panel-a-anchor-ckpt')
    parser.add_argument('--panel-a-anchor-ckpt', type=Path, default=None)
    parser.add_argument('--panel-a-ret-mult', type=float, default=2.5)
    parser.add_argument('--proximal-weight', type=float, default=0.10)
    parser.add_argument('--l0-prox-mult', type=float, default=1.0)
    parser.add_argument('--ball-tool-focus', type=float, default=1.0,
                        help='Extra proximal penalty multiplier on non-ball/tool/timing columns of W0 (e.g. 8.0)')
    parser.add_argument('--ball-prox-mult', type=float, default=0.5,
                        help='Proximal multiplier on ball pos/vel columns of W0 (default 0.5; lower e.g. 0.15 increases ball responsiveness)')
    parser.add_argument('--tool-prox-mult', type=float, default=1.0,
                        help='Proximal multiplier on tool & phase/clock columns of W0 (default 1.0)')
    parser.add_argument('--head-prox-mult', type=float, default=1.0,
                        help='Proximal multiplier on hidden/output approach layers .0.bias, .2.*, .4.* (default 1.0; e.g. 0.25)')
    parser.add_argument('--retention-exclude-dirs', type=Path, nargs='*', default=[],
                        help='Nominal dirs used only for nom_tapes pairing, excluded from retention targets')
    parser.add_argument('--fail-teacher-mult', type=float, default=1.0,
                        help='Extra weight multiplier for genuine FAIL->OK teachers vs OK->centered teachers')
    parser.add_argument('--high-z-mult', type=float, default=1.0,
                        help='Extra weight multiplier for high-altitude (iz>=1.82m) teachers')
    parser.add_argument('--wide-y-mult', type=float, default=1.0,
                        help='Extra weight multiplier for wide-lateral (|iy|>=0.12m) teachers')
    parser.add_argument('--no-pa-override', action='store_true',
                        help='Never override already-successful Panel A cases with centering teachers')
    parser.add_argument('--panel-b-teachers-only', action='store_true',
                        help='Restrict teachers strictly to Panel B (B-*) so Panel A stays 100% locked to base policy')
    parser.add_argument('--panel-b-retentions-only', action='store_true',
                        help='Restrict retentions strictly to Panel B (B-*) for unconstrained Panel-B specialist training')
    parser.add_argument('--panel-a-hinge-margin', type=float, default=0.020,
                        help='Max allowed arm action drift on Panel A before hard L_inf quadratic hinge barrier activates')
    parser.add_argument('--panel-a-hinge-weight', type=float, default=0.0,
                        help='Weight of Panel-A hinge barrier (e.g. 25.0-60.0 to hold PA_ok q95 < 0.03)')
    parser.add_argument('--closed-loop-teacher-dir', type=Path, default=None,
                        help='Optional closed-loop verified rollout dir (e.g. r5_bptt_b5_256) for fail->ok & court-centering teachers')
    parser.add_argument('--closed-loop-teacher-ckpt', type=Path, default=None,
                        help='Checkpoint that generated --closed-loop-teacher-dir (for exact obs_nom pairing)')
    parser.add_argument('--closed-loop-nom-dir', type=Path, default=None,
                        help='Matching nominal dir (e.g. r5_nominal_256) for --closed-loop-teacher-dir')
    parser.add_argument('--min-speed-margin', type=float, default=0.0)
    parser.add_argument('--max-teacher-lx', type=float, default=4.0,
                        help='Reject teacher trajectories landing further than this x [m] (e.g. 3.65)')
    parser.add_argument('--target-lx', type=float, default=3.20,
                        help='Preferred landing x [m] for court-centering teacher ranking')
    parser.add_argument('--min-net-margin', type=float, default=0.0,
                        help='Minimum net clearance margin [m] for teacher trajectories')
    parser.add_argument('--min-landing-margin', type=float, default=0.0,
                        help='Minimum court landing margin [m] (guards both |ly|<=1-m and lx<=4-m)')
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--obs-noise', type=float, default=0.0)
    parser.add_argument('--norm-obs-noise', type=float, default=0.0)
    parser.add_argument('--n-soup', type=int, default=1)
    parser.add_argument('--max-teacher-delta', type=float, default=0.55)
    parser.add_argument('--nominal-pair-weight', type=float, default=4.5)
    parser.add_argument('--val-fraction', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(args.base_checkpoint, map_location='cpu', weights_only=False)
    source_path = Path(ckpt['metadata']['init_policy_checkpoint']['path'])
    actor, _ = load_ppo_initialized_actor(source_path, device)
    actor.load_state_dict(ckpt['actor'], strict=True)
    base_actor, _ = load_ppo_initialized_actor(source_path, device)
    base_actor.load_state_dict(ckpt['actor'], strict=True)
    arm_prep = ckpt['metadata'].get('policy_execution', {}).get('arm_preparation_seconds', 0.25)
    if arm_prep is not None:
        actor.configure_approach_gate(arm_prep)
        base_actor.configure_approach_gate(arm_prep)
    base_actor.eval()
    for p in base_actor.parameters():
        p.requires_grad_(False)

    for name, p in actor.named_parameters():
        p.requires_grad_(name in APPROACH_NAMES)
    trainable = [(name, p) for name, p in actor.named_parameters() if p.requires_grad]
    anchor_params = {name: p.detach().clone() for name, p in trainable}
    init_full_state = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}

    teachers, retentions = load_datasets(
        args.nominal_dirs, args.probe_dirs, args.probe_specs,
        max_teacher_delta=args.max_teacher_delta, nominal_pair_weight=args.nominal_pair_weight,
        round_retention_decay=args.round_retention_decay, min_speed_margin=args.min_speed_margin,
        max_teacher_lx=args.max_teacher_lx, target_lx=args.target_lx,
        min_net_margin=args.min_net_margin, min_landing_margin=args.min_landing_margin,
        retention_exclude_dirs=args.retention_exclude_dirs,
        fail_teacher_mult=args.fail_teacher_mult, high_z_mult=args.high_z_mult,
        wide_y_mult=args.wide_y_mult, no_pa_override=args.no_pa_override)

    # Optional closed-loop real-PhysX verified teachers (e.g. B5_tight fail->ok & lx>3.45 -> <=3.45 centering)
    if args.closed_loop_teacher_dir and args.closed_loop_teacher_ckpt and args.closed_loop_nom_dir:
        cl_ckpt = torch.load(args.closed_loop_teacher_ckpt, map_location='cpu', weights_only=False)
        cl_actor, _ = load_ppo_initialized_actor(source_path, device)
        cl_actor.load_state_dict(cl_ckpt['actor'], strict=True)
        if arm_prep is not None:
            cl_actor.configure_approach_gate(arm_prep)
        cl_actor.eval()
        cl_sum = json.loads((args.closed_loop_teacher_dir / 'probe_summary.json').read_text())['rows']
        cl_lx = _load_lx_nm_map(args.closed_loop_teacher_dir)
        cl_tape = torch.load(args.closed_loop_teacher_dir / 'probe_tape.pt', map_location='cpu', weights_only=True)['tape']
        nm_sum = json.loads((args.closed_loop_nom_dir / 'probe_summary.json').read_text())['rows']
        nm_lx = _load_lx_nm_map(args.closed_loop_nom_dir)
        nm_tape = torch.load(args.closed_loop_nom_dir / 'probe_tape.pt', map_location='cpu', weights_only=True)['tape']
        tea_by_cid = {t['case_id']: t for t in teachers}
        ret_by_cid = {r['case_id']: r for r in retentions}
        n_cl_f, n_cl_c = 0, 0
        for i, (rn, rb) in enumerate(zip(nm_sum, cl_sum, strict=True)):
            cid = rb['case_id']
            if cid in tea_by_cid:
                continue
            if not (rb['success'] and rb['string'] and not rb['robot_net'] and not rb['low_speed']):
                continue
            lx_n, _ = nm_lx.get(i, (None, None))
            lx_b, nm_b = cl_lx.get(i, (None, None))
            if nm_b is not None and nm_b < 0.025:
                continue
            is_fail_ok = not rn['success']
            is_ctr = ((not (args.no_pa_override and cid.startswith('A-')))
                      and rn['success'] and lx_n is not None and lx_b is not None
                      and lx_n > 3.45 and lx_b <= lx_n - 0.15 and lx_b <= 3.48)
            if not (is_fail_ok or is_ctr):
                continue
            ep_t = extract_episode_frames(cl_tape, i)
            ep_n = extract_episode_frames(nm_tape, i)
            if ep_t is None or ep_n is None:
                continue
            with torch.no_grad():
                og = ep_n['obs'].to(device)
                nr = torch.zeros((len(og), 1), dtype=torch.bool, device=device)
                ep_n['req'] = cl_actor._gated_distribution(og, nr).loc.clamp(-1.0, 1.0).cpu()
            tw = float(args.fail_teacher_mult) if is_fail_ok else 1.25
            ep_t['weight'] = ep_t['weight'] * tw
            ep_n['weight'] = ep_n['weight'] * args.nominal_pair_weight * tw
            rb_row = dict(rb, lx=lx_b, nm=nm_b)
            tea_by_cid[cid] = dict(
                case_id=cid, sel_rank=(2, 0.0), eff_max=0.25, eff_l2=0.25,
                row=rb_row, meta=None, source=f'cl:{args.closed_loop_teacher_dir.name}',
                iz=1.80, was_nom_ok=not is_fail_ok, teacher_ep=ep_t, nom_paired_ep=ep_n,
            )
            ret_by_cid.pop(cid, None)
            if is_fail_ok:
                n_cl_f += 1
            else:
                n_cl_c += 1
        teachers = list(tea_by_cid.values())
        retentions = list(ret_by_cid.values())
        print(f'Injected {n_cl_f + n_cl_c} closed-loop verified teachers from {args.closed_loop_teacher_dir.name} '
              f'(fail->ok={n_cl_f}, lx-centered={n_cl_c})')

    if args.panel_b_teachers_only:
        before_n = len(teachers)
        teachers = [t for t in teachers if not t['case_id'].startswith('A-')]
        print(f'Filtered --panel-b-teachers-only: {before_n} -> {len(teachers)} pure Panel-B teachers')
    if args.panel_b_retentions_only:
        before_r = len(retentions)
        retentions = [r for r in retentions if not r['case_id'].startswith('A-')]
        print(f'Filtered --panel-b-retentions-only: {before_r} -> {len(retentions)} pure Panel-B retentions')

    # Optional Panel-A Fortress: anchor ONLY verified centered Panel A successes to panel_a_anchor_ckpt!
    if args.panel_a_anchor_dirs:
        pa_ckpt_path = args.panel_a_anchor_ckpt or args.base_checkpoint
        pa_ckpt = torch.load(pa_ckpt_path, map_location='cpu', weights_only=False)
        pa_actor, _ = load_ppo_initialized_actor(source_path, device)
        pa_actor.load_state_dict(pa_ckpt['actor'], strict=True)
        if arm_prep is not None:
            pa_actor.configure_approach_gate(arm_prep)
        pa_actor.eval()
        ret_by_cid = {r['case_id']: r for r in retentions}
        tea_cids = {t['case_id'] for t in teachers}
        n_pa_added = 0
        for adir in args.panel_a_anchor_dirs:
            summary = json.loads((adir / 'probe_summary.json').read_text())
            lx_nm = _load_lx_nm_map(adir)
            blob = torch.load(adir / 'probe_tape.pt', map_location='cpu', weights_only=True)
            tape = blob['tape']
            for i, row in enumerate(summary['rows']):
                cid = row['case_id']
                if not cid.startswith('A-'):
                    continue
                if not (row['success'] and not row['robot_net'] and not row['low_speed']):
                    continue
                if cid in tea_cids:
                    continue
                ep = extract_episode_frames(tape, i)
                if ep is None:
                    continue
                with torch.no_grad():
                    og = ep['obs'].to(device)
                    nr = torch.zeros((len(og), 1), dtype=torch.bool, device=device)
                    ep['req'] = pa_actor._gated_distribution(og, nr).loc.clamp(-1.0, 1.0).cpu()
                ep['weight'] = ep['weight'] * args.panel_a_ret_mult
                ep['arm_weight'] = torch.ones_like(ep['weight'])
                ep['is_pa'] = torch.ones_like(ep['weight'])
                ret_by_cid[cid] = dict(case_id=cid, score=10.0, row=row, source=f'pa_anchor:{adir.name}', **ep)
                n_pa_added += 1
        retentions = list(ret_by_cid.values())
        print(f'Injected {n_pa_added} verified centered Panel-A Fortress anchors from {pa_ckpt_path.name} (mult={args.panel_a_ret_mult}x)')
    # Build per-column proximal weight mask for actor.approach.0.weight (128 x 794):
    # Keep low penalty on ball(61:67), tool(45:57), phase/time(759:776); apply ball_tool_focus on joint/action history!
    w0_col_mult = torch.full((794,), float(args.ball_tool_focus), dtype=torch.float32, device=device)
    for f in range(11):
        b = f * 69
        w0_col_mult[b + 0:b + 9] = 1.5 * float(args.tool_prox_mult)  # root lin/ang vel & gravity
        w0_col_mult[b + 45:b + 57] = float(args.tool_prox_mult)      # tool pos/normal/vel
        w0_col_mult[b + 61:b + 67] = float(args.ball_prox_mult)      # ball pos/vel (HIGHEST priority!)
    w0_col_mult[759:776] = float(args.tool_prox_mult)                # command/phase/clock (776:794 last_action penalized!)
    effs = [e['eff_max'] for e in teachers]
    n_fail_t = sum(1 for e in teachers if not e.get('was_nom_ok', False))
    n_hz82 = sum(1 for e in teachers if e.get('iz', 0.0) >= 1.82)
    n_hz86 = sum(1 for e in teachers if e.get('iz', 0.0) >= 1.86)
    n_pa_t = sum(1 for e in teachers if e['case_id'].startswith('A-'))
    n_pa_r = sum(1 for e in retentions if e['case_id'].startswith('A-'))
    t_lxs = [e['row']['lx'] for e in teachers if e['row'].get('lx') is not None]
    print(f'Loaded {len(teachers)} unique teachers (fail->ok={n_fail_t}, ok->ctr={len(teachers)-n_fail_t}, '
          f'A={n_pa_t}, B={len(teachers)-n_pa_t}, iz>=1.82={n_hz82}, iz>=1.86={n_hz86}, '
          f'eff_max med={np.median(effs):.3f}, lx med={np.median(t_lxs):.2f}) '
          f'and {len(retentions)} unique retentions (A={n_pa_r}, B={len(retentions)-n_pa_r})')

    @torch.no_grad()
    def precompute_frozen(batch):
        if batch is None:
            return None
        mod = base_actor.actor
        norm_obs = mod.normalizer(batch['obs'])
        base_out = mod.base(norm_obs)
        agate = base_actor.approach_action_gate(batch['obs'])
        batch['norm_obs'] = norm_obs
        batch['base_out'] = base_out
        batch['agate'] = agate
        return batch

    def forward_fast(mod, norm_obs, base_out, agate, noise_std=0.0):
        inp = norm_obs + torch.randn_like(norm_obs) * noise_std if noise_std > 0.0 else norm_obs
        return (base_out + agate * mod.approach(inp) * mod.approach_scale).clamp(-1.0, 1.0)

    def train_one_member(member_idx: int, t_trn_eps, t_val_eps, r_trn_eps, r_val_eps):
        seed_m = args.seed + 1009 * member_idx
        torch.manual_seed(seed_m)
        np.random.seed(seed_m)
        actor.load_state_dict(init_full_state, strict=True)
        actor.to(device)
        trainable_m = [(name, p) for name, p in actor.named_parameters() if name in APPROACH_NAMES]
        for name, p in actor.named_parameters():
            p.requires_grad_(name in APPROACH_NAMES)

        t_trn = precompute_frozen(concat_teacher_episodes(t_trn_eps, device, mode='both'))
        t_val_nom = concat_teacher_episodes(t_val_eps, device, mode='nominal')
        t_val_tea = concat_teacher_episodes(t_val_eps, device, mode='teacher')
        t_trn_nom = concat_teacher_episodes(t_trn_eps, device, mode='nominal')
        r_trn = precompute_frozen(concat_episodes(r_trn_eps, device))
        r_val = concat_episodes(r_val_eps, device)

        if member_idx == 0:
            print(f'Frames: teacher_train(both)={len(t_trn["obs"])}, ret_train={len(r_trn["obs"])}, '
                  f'val_nom={len(t_val_nom["obs"])}, val_ret={len(r_val["obs"])}')

        if args.weight_decay > 0:
            opt = torch.optim.AdamW([p for _, p in trainable_m], lr=args.lr, weight_decay=args.weight_decay)
        else:
            opt = torch.optim.Adam([p for _, p in trainable_m], lr=args.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.05)

        best_score = float('inf')
        best_state = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
        best_epoch = 0
        n_t = len(t_trn['obs'])
        n_r = len(r_trn['obs'])
        steps_per_epoch = max(1, math.ceil(max(n_t, n_r) / args.batch_size))
        mod = actor.actor

        for epoch in range(1, args.epochs + 1):
            actor.train()
            for _ in range(steps_per_epoch):
                opt.zero_grad(set_to_none=True)
                t_idx = torch.randint(0, n_t, (min(args.batch_size, n_t),), device=device)
                pred_t = forward_fast(mod, t_trn['norm_obs'][t_idx], t_trn['base_out'][t_idx],
                                      t_trn['agate'][t_idx], noise_std=args.norm_obs_noise)
                loss_t = active_loss(pred_t, t_trn['req'][t_idx], t_trn['gate'][t_idx],
                                     t_trn['weight'][t_idx], arm_only=True, quadratic=True)
                if args.teacher_leg_weight > 0:
                    mask_leg = (t_trn['gate'][t_idx, :12] > 0.02).float()
                    sq_leg = (pred_t[:, :12] - t_trn['req'][t_idx, :12]).square() * mask_leg
                    pf_leg = sq_leg.sum(-1) / mask_leg.sum(-1).clamp_min(1.0)
                    loss_t_leg = (pf_leg * t_trn['weight'][t_idx]).sum() / t_trn['weight'][t_idx].sum().clamp_min(1.0)
                    loss_t = loss_t + args.teacher_leg_weight * loss_t_leg

                r_idx = torch.randint(0, n_r, (min(args.batch_size, n_r),), device=device)
                pred_r = forward_fast(mod, r_trn['norm_obs'][r_idx], r_trn['base_out'][r_idx],
                                      r_trn['agate'][r_idx], noise_std=args.norm_obs_noise * 0.5)
                loss_r = active_loss(pred_r, r_trn['req'][r_idx], r_trn['gate'][r_idx],
                                     r_trn['weight'][r_idx], arm_only=False, quadratic=True,
                                     arm_weight=r_trn.get('arm_weight')[r_idx] if 'arm_weight' in r_trn else None)
                if args.panel_a_hinge_weight > 0.0 and 'is_pa' in r_trn:
                    pa_m = r_trn['is_pa'][r_idx] > 0.5
                    if pa_m.any():
                        pa_mask = (r_trn['gate'][r_idx][pa_m, 12:18] > 0.02).float()
                        pa_exc = F.relu((pred_r[pa_m, 12:18] - r_trn['req'][r_idx][pa_m, 12:18]).abs() - args.panel_a_hinge_margin)
                        loss_pa_h = (pa_exc.square() * pa_mask).sum() / pa_mask.sum().clamp_min(1.0)
                        loss_r = loss_r + args.panel_a_hinge_weight * loss_pa_h

                prox_terms = []
                for name, p in trainable_m:
                    if name == 'actor.approach.0.weight' and args.ball_tool_focus != 1.0:
                        prox_terms.append(args.l0_prox_mult * ((p - anchor_params[name]).square() * w0_col_mult[None, :]).mean())
                    else:
                        mult = args.l0_prox_mult if name == 'actor.approach.0.weight' else args.head_prox_mult
                        prox_terms.append(mult * (p - anchor_params[name]).square().mean())
                prox = sum(prox_terms) / len(prox_terms)
                loss = args.teacher_weight * loss_t + args.retention_weight * loss_r + args.proximal_weight * prox
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for _, p in trainable_m], 2.0)
                opt.step()
            sched.step()

            if epoch % 5 == 0 or epoch == args.epochs:
                actor.eval()
                mn = eval_metrics(actor, t_val_nom, arm_only=True, base_actor=base_actor)
                mr = eval_metrics(actor, r_val, arm_only=False)
                score = args.teacher_weight * mn['mse'] + args.retention_weight * mr['mse']
                if score < best_score:
                    best_score = score
                    best_state = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
                    best_epoch = epoch
                if (epoch % 100 == 0 or epoch == args.epochs) and member_idx == 0:
                    print(f'  [m{member_idx}] ep={epoch:3d} val_NOM_rms={mn["rms_arm"]:.4f} cos={mn["cos_delta"]:+.3f} '
                          f'mag={mn["mag_ratio"]:.2f} | val_RET_arm={mr["rms_arm"]:.4f}(max={mr["max_arm"]:.3f}) '
                          f'leg={mr["rms_leg"]:.4f} score={score:.6f} (best@{best_epoch})')
        return best_state, best_epoch, (t_trn_nom, t_val_nom, t_val_tea, r_trn, r_val)

    def make_splits(eps, k_folds: int):
        perm = np.random.permutation(len(eps))
        if k_folds <= 1:
            n_val = max(1, int(round(len(eps) * args.val_fraction))) if len(eps) >= 6 else 0
            return [([eps[i] for i in perm[n_val:]], [eps[i] for i in perm[:n_val]])]
        folds = np.array_split(perm, k_folds)
        splits = []
        for f in range(k_folds):
            val_idx = set(folds[f].tolist())
            trn_e = [eps[i] for i in perm if i not in val_idx]
            val_e = [eps[i] for i in perm if i in val_idx]
            splits.append((trn_e, val_e))
        return splits

    t_splits = make_splits(teachers, args.n_soup)
    r_splits = make_splits(retentions, args.n_soup)

    member_states = []
    member_epochs = []
    eval_pack = None
    for m_idx in range(args.n_soup):
        st_m, ep_m, pack_m = train_one_member(m_idx, t_splits[m_idx][0], t_splits[m_idx][1],
                                              r_splits[m_idx][0], r_splits[m_idx][1])
        member_states.append(st_m)
        member_epochs.append(ep_m)
        if m_idx == 0:
            eval_pack = pack_m

    if args.n_soup > 1:
        best_state = {k: v.clone() for k, v in init_full_state.items()}
        for name in APPROACH_NAMES:
            best_state[name] = sum(s[name].float() for s in member_states) / len(member_states)
            best_state[name] = best_state[name].to(init_full_state[name].dtype)
        best_epoch = int(round(float(np.mean(member_epochs))))
    else:
        best_state = member_states[0]
        best_epoch = member_epochs[0]

    t_trn_nom, t_val_nom, t_val_tea, r_trn, r_val = eval_pack
    actor.load_state_dict(best_state, strict=True)
    actor.eval()
    stats = dict(
        trn_nom=eval_metrics(actor, t_trn_nom, arm_only=True, base_actor=base_actor),
        val_nom=eval_metrics(actor, t_val_nom, arm_only=True, base_actor=base_actor),
        val_tea=eval_metrics(actor, t_val_tea, arm_only=True, base_actor=base_actor),
        ret_trn=eval_metrics(actor, r_trn, arm_only=False),
        ret_val=eval_metrics(actor, r_val, arm_only=False),
    )
    print(f'FINAL (soup={args.n_soup}, best@{best_epoch}):', json.dumps(stats, indent=2))
    param_deltas = torch.cat([(best_state[name].float() - anchor_params[name].cpu().float()).flatten()
                              for name, _ in trainable])
    print(f'Param delta vs base: rms={float(param_deltas.square().mean().sqrt()):.6f}, max={float(param_deltas.abs().max()):.6f}')

    for k, v in ckpt['actor'].items():
        if k not in APPROACH_NAMES and torch.is_tensor(v):
            assert torch.equal(v.cpu(), best_state[k].cpu()), f'Frozen param changed: {k}'

    out_ckpt = {k: v for k, v in ckpt.items()
                if k in ('algorithm', 'adapter_schema', 'actor', 'metadata', 'config',
                         'actor_updates', 'world_updates', 'critic_updates')}
    out_ckpt['actor'] = best_state
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ckpt, args.output)
    args.output.with_suffix('.distill.json').write_text(json.dumps(dict(
        base=str(args.base_checkpoint), teachers=len(teachers), retentions=len(retentions),
        best_epoch=best_epoch, stats=stats,
        param_delta_rms=float(param_deltas.square().mean().sqrt()),
        param_delta_max=float(param_deltas.abs().max()),
    ), indent=2) + '\n')
    print(f'Saved -> {args.output}')


if __name__ == '__main__':
    main()
