#!/usr/bin/env python3
"""Upgraded Closed-Loop DAgger + Offline Distance-Margin DPO + Center-Biased Retention Distillation.

Key upgrades over distill_native_approach.py:
1. Seamlessly incorporates Pass-1 AND Pass-2 refined teachers mapped back to their exact
   unperturbed nominal observations (zero action-history shortcut).
2. Center-biased retention softening: reads actual contact local_z (native_contact_motion.json)
   and first_landing_m (episodes.jsonl) for every nominal success, softening arm retention on
   high-rim (local_z > +35mm), deep-backline (lx > 3.50m), wide (|ly| > 0.60m), or slow (<4.85m/s)
   nominal successes so teachers & DPO pairs can pull the entire distribution into the sweet spot.
3. Scale-invariant Offline Distance-Margin DPO Preference Loss across ALL partial-credit and
   negative-contrast probe pairs (R+ - R- >= dpo_min_gap) on unperturbed nominal observations,
   extracting rich directional gradient even from unsolved FRAME_HIT / LOW_SPD / LONG_OOB cases.
4. Optional --unlock-base mode to jointly fine-tune actor.base (256->128->64->18) + actor.approach
   (128->64->18) for 3.2x representational capacity while strictly locking ALL off-window frames
   (standing, run-up, takeoff, post-contact recovery) via an explicit off-window anchor.
"""
import argparse
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


def quat_wxyz_to_rot(q):
    w, x, y, z = [float(v) for v in q]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def compute_shaped_reward(row) -> float:
    succ = 1.0 if row['success'] else 0.0
    qstr = 1.0 if row['string'] else 0.0
    rnet = 1.0 if row['robot_net'] else 0.0
    sm = row.get('string_margin_m')
    sm_t = max(-1.5, min(1.0, sm / 0.040)) if (sm is not None and math.isfinite(sm)) else -1.5
    spm = row.get('speed_margin_mps')
    spm_t = max(-1.5, min(1.0, spm / 0.55)) if (qstr > 0 and spm is not None and math.isfinite(spm)) else -0.5
    lm = row.get('landing_margin_m')
    lm_t = max(-1.5, min(1.0, lm / 0.35)) if (qstr > 0 and lm is not None and math.isfinite(lm)) else -1.0
    nm = row.get('net_margin_m')
    nm_t = max(-1.0, min(1.0, nm / 0.12)) if (qstr > 0 and nm is not None and math.isfinite(nm)) else -0.5
    had_c = (row.get('string') or row.get('frame') or (sm is not None))
    dist_pen = 0.0 if had_c else -0.60
    return float(
        3.50 * succ + 1.20 * qstr + 0.70 * sm_t + 0.55 * spm_t + 0.70 * lm_t + 0.35 * nm_t + dist_pen - 6.00 * rnet
    )


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


def extract_episode_frames(tape, lane: int, *, min_gate: float = 0.05, override_req_delta=None):
    ep0 = tape['episode'][:, lane] == 0
    hc = tape['had_contact'][:, lane]
    pres = tape['present'][:, lane]
    gate = tape['gate'][:, lane].float()
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
        obs=tape['observation'][idxs, lane].float(),
        req=req,
        gate=gate[idxs],
        weight=weights,
        idxs=idxs,
    )


def extract_off_window_frames(tape, lane: int, *, stride: int = 3):
    """Extract standing/run-up/takeoff/recovery frames where approach gate is inactive or contact occurred."""
    ep0 = tape['episode'][:, lane] == 0
    hc = tape['had_contact'][:, lane]
    gate_max = tape['gate'][:, lane].float().amax(-1)
    off_mask = ep0 & (hc | (gate_max <= 0.05))
    idxs = off_mask.nonzero().flatten()
    if len(idxs) == 0:
        return None
    idxs = idxs[::stride]
    return dict(
        obs=tape['observation'][idxs, lane].float(),
        hc=hc[idxs],
    )


def load_datasets(nominal_dirs, probe_dirs, probe_specs, *,
                  max_teacher_delta: float = 0.62, nominal_pair_weight: float = 4.5,
                  round_retention_decay: float = 0.50, teacher_round_decay: float = 1.0,
                  center_bias_retention: bool = True,
                  dpo_min_gap: float = 1.0, dpo_max_pairs_per_case: int = 2,
                  dpo_round_decay: float = 0.70, dpo_success_only: bool = True,
                  collect_off_window: bool = False):
    teachers = {}
    retentions = {}
    nom_tapes = {}  # case_id -> (tape, lane, start_tick, row, score, r_idx, round_w_dpo, round_w_tea)
    case_branches = {}  # case_id -> list of dict(...)
    off_window_list = []

    n_rounds = len(nominal_dirs)
    for r_idx, ndir in enumerate(nominal_dirs):
        ndir = Path(ndir)
        round_w = round_retention_decay ** (n_rounds - 1 - r_idx)
        round_w_tea = teacher_round_decay ** (n_rounds - 1 - r_idx)
        round_w_dpo = dpo_round_decay ** (n_rounds - 1 - r_idx)
        summary = json.loads((ndir / 'probe_summary.json').read_text())
        blob = torch.load(ndir / 'probe_tape.pt', map_location='cpu', weights_only=True)
        tape = blob['tape']
        start_tick = int(blob['start_capture_tick'])

        # Load exact contact local_z and first_landing_m for center-biased retention
        contacts = {}
        cm_path = ndir / 'native_contact_motion.json'
        if cm_path.is_file():
            for c in json.loads(cm_path.read_text()):
                if c['env_id'] not in contacts:
                    contacts[c['env_id']] = c
        eps_map = {}
        ep_path = ndir / 'episodes.jsonl'
        if ep_path.is_file():
            for line in ep_path.read_text().splitlines():
                r_ep = json.loads(line)
                if r_ep.get('episode_index') == 0:
                    eps_map[r_ep['env_id']] = r_ep

        for i, row in enumerate(summary['rows']):
            cid = row['case_id']
            sm = row['string_margin_m'] if row['string_margin_m'] is not None else 0.0
            spm = row['speed_margin_mps'] if row['speed_margin_mps'] is not None else 0.0
            lm = row.get('landing_margin_m') if row.get('landing_margin_m') is not None else 0.0
            score = float(row['margin_score']) + 0.5 * min(spm, 1.0) + 10.0 * min(max(sm, 0.0), 0.05)
            nom_tapes[cid] = (tape, i, start_tick, row, score, r_idx, round_w_dpo, round_w_tea)
            case_branches.setdefault(cid, []).append(dict(
                reward=compute_shaped_reward(row),
                row=row,
                offset=None,
                ramp=None,
                peak=None,
                width=None,
                is_nom=True,
                eff_max=0.0,
            ))

            if collect_off_window and (i % 2 == 0):
                ow = extract_off_window_frames(tape, i, stride=4)
                if ow is not None:
                    off_window_list.append(ow)

            if row['success'] and not row['robot_net'] and not row['low_speed']:
                ep = extract_episode_frames(tape, i)
                if ep is not None:
                    sm_w = max(0.45, min(1.0, (sm - 0.010) / 0.025))
                    sp_w = max(0.50, min(1.0, spm / 0.35))
                    lm_w = max(0.50, min(1.0, lm / 0.25))
                    lz_w, lx_w, ly_w = 1.0, 1.0, 1.0
                    if center_bias_retention:
                        cm = contacts.get(i)
                        if cm is not None and all(k in cm for k in ('point_m', 'racket_center_m', 'racket_quaternion_w')):
                            R_c = quat_wxyz_to_rot(cm['racket_quaternion_w'])
                            loc = R_c.T @ (np.array(cm['point_m']) - np.array(cm['racket_center_m']))
                            loc_z = float(loc[2])
                            if loc_z > 0.050:
                                lz_w = max(0.55, 1.0 - (loc_z - 0.050) / 0.080)
                        ep_rec = eps_map.get(i)
                        if ep_rec is not None and ep_rec.get('first_landing_m'):
                            lx, ly = float(ep_rec['first_landing_m'][0]), float(ep_rec['first_landing_m'][1])
                            if lx > 3.65:
                                lx_w = max(0.55, 1.0 - (lx - 3.65) / 0.50)
                            if abs(ly) > 0.70:
                                ly_w = max(0.60, 1.0 - (abs(ly) - 0.70) / 0.40)
                    comb_w = max(0.38, sm_w * sp_w * lm_w * lz_w * lx_w * ly_w) * round_w
                    ep['arm_weight'] = torch.full_like(ep['weight'], comb_w)
                    ep.pop('idxs', None)
                    prev = retentions.get(cid)
                    if prev is None or score > prev['score']:
                        retentions[cid] = dict(case_id=cid, score=score, row=row, source=str(ndir), **ep)

    for pdir, pspec_path in zip(probe_dirs, probe_specs, strict=True):
        pdir = Path(pdir)
        pspec = json.loads(Path(pspec_path).read_text())
        summary = json.loads((pdir / 'probe_summary.json').read_text())
        blob = torch.load(pdir / 'probe_tape.pt', map_location='cpu', weights_only=True)
        tape = blob['tape']
        meta = summary.get('branch_meta') or [None] * len(summary['rows'])
        offsets = pspec['offsets']
        ramps = pspec.get('ramp_offsets') or [None] * len(offsets)
        peaks = pspec['peak_ticks']
        widths = pspec['width_ticks']

        for i, row in enumerate(summary['rows']):
            cid = row['case_id']
            if cid not in nom_tapes:
                continue
            off = np.array(offsets[i], dtype=np.float64)
            rmp = np.array(ramps[i], dtype=np.float64) if ramps[i] is not None else np.zeros(18)
            eff_max = float(np.abs(off + rmp).max())
            eff_l2 = float(np.linalg.norm(off) + np.linalg.norm(rmp))

            # Record valid probe branch for Offline DPO preference pairs
            if eff_max <= max_teacher_delta + 0.08:
                case_branches.setdefault(cid, []).append(dict(
                    reward=compute_shaped_reward(row),
                    row=row,
                    offset=offsets[i],
                    ramp=ramps[i],
                    peak=peaks[i],
                    width=widths[i],
                    is_nom=False,
                    eff_max=eff_max,
                ))

            if not (row['success'] and row['string'] and not row['robot_net']
                    and not row['ball_net'] and not row['low_speed']):
                continue
            if eff_max > max_teacher_delta:
                continue
            sm = row['string_margin_m'] if row['string_margin_m'] is not None else 0.0
            spm = row['speed_margin_mps'] if row['speed_margin_mps'] is not None else 0.0
            lm = row.get('landing_margin_m') if row.get('landing_margin_m') is not None else 0.0
            tier = 2 if (spm >= 0.30 and sm >= 0.012 and lm >= 0.15) else (1 if (spm >= 0.15 and sm >= 0.006 and lm >= 0.05) else 0)
            sel_rank = (tier, -eff_l2 + 0.18 * min(spm, 1.2) + 2.5 * min(sm, 0.04) + 0.40 * min(lm, 0.40))
            if cid in retentions:
                old_r = retentions[cid]['row']
                old_lm = old_r.get('landing_margin_m') if old_r.get('landing_margin_m') is not None else 0.0
                improved = (row['margin_score'] >= old_r['margin_score'] + 0.06) or (lm >= old_lm + 0.12 and spm >= 0.20)
                if not (improved and eff_max <= 0.48):
                    continue

            ntape, nlane, nstart, _, _, _, _, round_w_tea = nom_tapes[cid]
            ep_teacher = extract_episode_frames(tape, i)
            if ep_teacher is None:
                continue
            ep_teacher['weight'] = ep_teacher['weight'] * round_w_tea
            ep_teacher.pop('idxs', None)

            ep_nom_paired = None
            if nominal_pair_weight > 0:
                delta_sched = compute_delta_schedule(
                    ntape['observation'].shape[0], nstart, offsets[i], ramps[i], peaks[i], widths[i])
                ep_nom_paired = extract_episode_frames(ntape, nlane, override_req_delta=delta_sched)
                if ep_nom_paired is not None:
                    ep_nom_paired['weight'] = ep_nom_paired['weight'] * nominal_pair_weight * round_w_tea
                    ep_nom_paired.pop('idxs', None)

            prev = teachers.get(cid)
            if prev is None or sel_rank > prev['sel_rank']:
                teachers[cid] = dict(
                    case_id=cid, sel_rank=sel_rank, eff_max=eff_max, eff_l2=eff_l2,
                    row=row, meta=meta[i], source=str(pdir),
                    teacher_ep=ep_teacher, nom_paired_ep=ep_nom_paired,
                )
                retentions.pop(cid, None)

    # Build Offline DPO preference pairs on strictly unperturbed nominal observations!
    dpo_episodes = []
    n_dpo_pairs = 0
    n_dpo_unsolved = 0
    for cid, branches in case_branches.items():
        if cid not in nom_tapes or len(branches) < 2:
            continue
        if dpo_success_only:
            valid_wins = [b for b in branches if b['row'].get('success') and not b['row'].get('robot_net') and not b['row'].get('low_speed')]
        else:
            valid_wins = [b for b in branches if not b['row'].get('robot_net', False)]
        if not valid_wins:
            continue
        best_b = max(valid_wins, key=lambda b: (b['reward'], -b['eff_max']))
        losers = sorted(
            [b for b in branches if (best_b['reward'] - b['reward']) >= dpo_min_gap],
            key=lambda b: b['reward']
        )[:dpo_max_pairs_per_case]
        if not losers:
            continue

        ntape, nlane, nstart, _, _, _, round_w_dpo, _ = nom_tapes[cid]
        T = ntape['observation'].shape[0]
        base_ep = extract_episode_frames(ntape, nlane)
        if base_ep is None:
            continue
        idxs = base_ep['idxs']
        obs = base_ep['obs']
        req_nom = base_ep['req']
        gate = base_ep['gate']
        base_w = base_ep['weight']

        if best_b['is_nom']:
            a_pos = req_nom
        else:
            dp = compute_delta_schedule(T, nstart, best_b['offset'], best_b['ramp'], best_b['peak'], best_b['width'])
            a_pos = (req_nom + gate * dp[idxs]).clamp(-1.0, 1.0)

        is_unsolved = (cid not in teachers) and (not best_b['row'].get('success', False))
        # Extra boost for unsolved partial-credit pairs (no BC teacher covers them!)
        case_mult = 1.35 if is_unsolved else 0.70

        ep_obs, ep_apos, ep_aneg, ep_w = [], [], [], []
        for lb in losers:
            if lb['is_nom']:
                a_neg = req_nom
            else:
                dn = compute_delta_schedule(T, nstart, lb['offset'], lb['ramp'], lb['peak'], lb['width'])
                a_neg = (req_nom + gate * dn[idxs]).clamp(-1.0, 1.0)
            sep_arm = (a_pos[:, 12:18] - a_neg[:, 12:18]).square().mean(-1).sqrt()
            active = (gate[:, 12:18].amax(-1) > 0.05) & (sep_arm > 0.015)
            if not active.any():
                continue
            gap = float(best_b['reward'] - lb['reward'])
            pair_w = min(2.0, gap / 2.5) * case_mult * round_w_dpo
            ep_obs.append(obs[active])
            ep_apos.append(a_pos[active])
            ep_aneg.append(a_neg[active])
            ep_w.append(base_w[active] * pair_w)
            n_dpo_pairs += 1
            if is_unsolved:
                n_dpo_unsolved += 1

        if ep_obs:
            dpo_episodes.append(dict(
                case_id=cid,
                obs=torch.cat(ep_obs, dim=0),
                a_pos=torch.cat(ep_apos, dim=0),
                a_neg=torch.cat(ep_aneg, dim=0),
                weight=torch.cat(ep_w, dim=0),
            ))

    print(f'Built {n_dpo_pairs} Offline DPO preference pairs across {len(dpo_episodes)} cases '
          f'({n_dpo_unsolved} partial-credit pairs on unsolved failures)')
    return list(teachers.values()), list(retentions.values()), dpo_episodes, off_window_list


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
    return out


def concat_dpo(episodes, device):
    if not episodes:
        return None
    return dict(
        obs=torch.cat([e['obs'] for e in episodes], dim=0).to(device),
        a_pos=torch.cat([e['a_pos'] for e in episodes], dim=0).to(device),
        a_neg=torch.cat([e['a_neg'] for e in episodes], dim=0).to(device),
        weight=torch.cat([e['weight'] for e in episodes], dim=0).to(device),
    )


def active_loss(pred, target, gate, weight, *, arm_only: bool, arm_weight=None, full_legs: bool = False):
    sl = slice(12, 18) if arm_only else slice(0, 18)
    p = pred[:, sl]
    t = target[:, sl]
    mask = (gate[:, sl] > 0.02).float()
    if full_legs and not arm_only:
        mask[:, :12] = 1.0
    if arm_weight is not None and not arm_only:
        joint_w = torch.ones_like(mask)
        joint_w[:, 12:18] = arm_weight[:, None]
        mask = mask * joint_w
    sq = (p - t).square() * mask
    per_frame = sq.sum(-1) / mask.sum(-1).clamp_min(1.0)
    return (per_frame * weight).sum() / weight.sum().clamp_min(1.0)


def dpo_distance_loss(pred, a_pos, a_neg, weight, beta: float = 18.0):
    p = pred[:, 12:18]
    ap = a_pos[:, 12:18]
    an = a_neg[:, 12:18]
    d_pos = (p - ap).square().mean(-1)
    d_neg = (p - an).square().mean(-1)
    sep = (ap - an).square().mean(-1).sqrt() + 0.02
    margin = (d_neg - d_pos) / (2.0 * sep)
    loss_vec = F.softplus(-beta * margin)
    return (loss_vec * weight).sum() / weight.sum().clamp_min(1.0), float((margin > 0).float().mean())


@torch.no_grad()
def eval_metrics(actor, batch, *, arm_only: bool, base_actor=None):
    if batch is None:
        return dict(mse=0.0, rms_arm=0.0, rms_leg=0.0, max_arm=0.0, cos_delta=0.0, mag_ratio=0.0)
    no_rec = torch.zeros(len(batch['obs']), dtype=torch.bool, device=batch['obs'].device)
    pred = actor.forward_with_recovery_gate(batch['obs'], no_rec, sample=False)[0].clamp(-1.0, 1.0)
    aw = batch.get('arm_weight') if not arm_only else None
    mse = float(active_loss(pred, batch['req'], batch['gate'], batch['weight'], arm_only=arm_only, arm_weight=aw))
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
    parser.add_argument('--epochs', type=int, default=260)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--teacher-weight', type=float, default=1.0)
    parser.add_argument('--teacher-leg-weight', type=float, default=1.0)
    parser.add_argument('--retention-weight', type=float, default=3.2)
    parser.add_argument('--round-retention-decay', type=float, default=0.50)
    parser.add_argument('--teacher-round-decay', type=float, default=1.0)
    parser.add_argument('--center-bias-retention', action='store_true', default=True)
    parser.add_argument('--no-center-bias-retention', dest='center_bias_retention', action='store_false')
    parser.add_argument('--dpo-weight', type=float, default=0.008)
    parser.add_argument('--dpo-beta', type=float, default=18.0)
    parser.add_argument('--dpo-min-gap', type=float, default=1.0)
    parser.add_argument('--dpo-round-decay', type=float, default=0.75)
    parser.add_argument('--dpo-success-only', action='store_true', default=True)
    parser.add_argument('--allow-partial-dpo', dest='dpo_success_only', action='store_false')
    parser.add_argument('--unlock-base', action='store_true', default=False)
    parser.add_argument('--base-lr-mult', type=float, default=0.35)
    parser.add_argument('--base-prox-weight', type=float, default=0.35)
    parser.add_argument('--off-window-weight', type=float, default=35.0)
    parser.add_argument('--proximal-weight', type=float, default=0.04)
    parser.add_argument('--l0-prox-mult', type=float, default=1.0)
    parser.add_argument('--norm-obs-noise', type=float, default=0.025)
    parser.add_argument('--n-soup', type=int, default=5)
    parser.add_argument('--max-teacher-delta', type=float, default=0.62)
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

    base_names = {k for k in ckpt['actor'].keys() if k.startswith('actor.base.')}
    trainable_names = (set(APPROACH_NAMES) | base_names) if args.unlock_base else set(APPROACH_NAMES)
    anchor_params = {name: p.detach().clone() for name, p in actor.named_parameters() if name in trainable_names}
    init_full_state = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}

    teachers, retentions, dpo_episodes, off_window_list = load_datasets(
        args.nominal_dirs, args.probe_dirs, args.probe_specs,
        max_teacher_delta=args.max_teacher_delta, nominal_pair_weight=args.nominal_pair_weight,
        round_retention_decay=args.round_retention_decay, teacher_round_decay=args.teacher_round_decay,
        center_bias_retention=args.center_bias_retention,
        dpo_min_gap=args.dpo_min_gap, dpo_round_decay=args.dpo_round_decay,
        dpo_success_only=args.dpo_success_only,
        collect_off_window=args.unlock_base,
    )
    effs = [e['eff_max'] for e in teachers]
    print(f'Loaded {len(teachers)} unique min-effort teachers (eff_max med={np.median(effs):.3f}, max={max(effs):.3f}), '
          f'{len(retentions)} retention eps, {len(dpo_episodes)} DPO eps, unlock_base={args.unlock_base}')

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

    off_win_batch = None
    if args.unlock_base and off_window_list:
        ow_obs = torch.cat([x['obs'] for x in off_window_list], dim=0).to(device)
        with torch.no_grad():
            ow_norm = base_actor.actor.normalizer(ow_obs)
            ow_base = base_actor.actor.base(ow_norm)
        off_win_batch = dict(norm_obs=ow_norm, base_out=ow_base)
        print(f'Built off-window base anchor: {len(ow_norm)} frames (standing/run-up/takeoff/recovery)')

    def forward_fast(mod, norm_obs, base_out, agate, noise_std=0.0):
        inp = norm_obs + torch.randn_like(norm_obs) * noise_std if noise_std > 0.0 else norm_obs
        b_out = mod.base(inp) if args.unlock_base else base_out
        return (b_out + agate * mod.approach(inp) * mod.approach_scale).clamp(-1.0, 1.0)

    def train_one_member(member_idx: int, t_trn_eps, t_val_eps, r_trn_eps, r_val_eps, d_trn_eps, d_val_eps):
        seed_m = args.seed + 1009 * member_idx
        torch.manual_seed(seed_m)
        np.random.seed(seed_m)
        actor.load_state_dict(init_full_state, strict=True)
        actor.to(device)
        for name, p in actor.named_parameters():
            p.requires_grad_(name in trainable_names)
        app_params = [(n, p) for n, p in actor.named_parameters() if n in APPROACH_NAMES]
        bas_params = [(n, p) for n, p in actor.named_parameters() if n in base_names] if args.unlock_base else []
        all_trainable = app_params + bas_params

        t_trn = precompute_frozen(concat_teacher_episodes(t_trn_eps, device, mode='both'))
        t_val_nom = concat_teacher_episodes(t_val_eps, device, mode='nominal')
        t_val_tea = concat_teacher_episodes(t_val_eps, device, mode='teacher')
        t_trn_nom = concat_teacher_episodes(t_trn_eps, device, mode='nominal')
        r_trn = precompute_frozen(concat_episodes(r_trn_eps, device))
        r_val = concat_episodes(r_val_eps, device)
        d_trn = precompute_frozen(concat_dpo(d_trn_eps, device)) if (args.dpo_weight > 0 and d_trn_eps) else None
        d_val = precompute_frozen(concat_dpo(d_val_eps, device)) if (args.dpo_weight > 0 and d_val_eps) else None

        if member_idx == 0:
            n_d = len(d_trn['obs']) if d_trn is not None else 0
            print(f'Frames: teacher(both)={len(t_trn["obs"])}, ret={len(r_trn["obs"])}, dpo={n_d}, '
                  f'val_nom={len(t_val_nom["obs"])}, val_ret={len(r_val["obs"])}')

        param_groups = [{'params': [p for _, p in app_params], 'lr': args.lr}]
        if bas_params:
            param_groups.append({'params': [p for _, p in bas_params], 'lr': args.lr * args.base_lr_mult})
        opt = torch.optim.Adam(param_groups)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.05)

        best_score = float('inf')
        best_state = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
        best_epoch = 0
        n_t = len(t_trn['obs'])
        n_r = len(r_trn['obs'])
        n_d = len(d_trn['obs']) if d_trn is not None else 0
        n_ow = len(off_win_batch['norm_obs']) if off_win_batch is not None else 0
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
                                     t_trn['weight'][t_idx], arm_only=True)
                if args.teacher_leg_weight > 0:
                    if args.unlock_base:
                        mask_leg = torch.ones_like(t_trn['gate'][t_idx, :12])
                    else:
                        mask_leg = (t_trn['gate'][t_idx, :12] > 0.02).float()
                    sq_leg = (pred_t[:, :12] - t_trn['req'][t_idx, :12]).square() * mask_leg
                    pf_leg = sq_leg.sum(-1) / mask_leg.sum(-1).clamp_min(1.0)
                    loss_t_leg = (pf_leg * t_trn['weight'][t_idx]).sum() / t_trn['weight'][t_idx].sum().clamp_min(1.0)
                    loss_t = loss_t + (args.teacher_leg_weight * (4.0 if args.unlock_base else 1.0)) * loss_t_leg

                r_idx = torch.randint(0, n_r, (min(args.batch_size, n_r),), device=device)
                pred_r = forward_fast(mod, r_trn['norm_obs'][r_idx], r_trn['base_out'][r_idx],
                                      r_trn['agate'][r_idx], noise_std=args.norm_obs_noise * 0.5)
                loss_r = active_loss(pred_r, r_trn['req'][r_idx], r_trn['gate'][r_idx],
                                     r_trn['weight'][r_idx], arm_only=False,
                                     arm_weight=r_trn.get('arm_weight')[r_idx] if 'arm_weight' in r_trn else None,
                                     full_legs=args.unlock_base)

                loss = args.teacher_weight * loss_t + args.retention_weight * loss_r

                if d_trn is not None and args.dpo_weight > 0:
                    d_idx = torch.randint(0, n_d, (min(args.batch_size, n_d),), device=device)
                    pred_d = forward_fast(mod, d_trn['norm_obs'][d_idx], d_trn['base_out'][d_idx],
                                          d_trn['agate'][d_idx], noise_std=args.norm_obs_noise * 0.5)
                    loss_d, _ = dpo_distance_loss(pred_d, d_trn['a_pos'][d_idx], d_trn['a_neg'][d_idx],
                                                  d_trn['weight'][d_idx], beta=args.dpo_beta)
                    loss = loss + args.dpo_weight * loss_d

                if off_win_batch is not None and args.off_window_weight > 0:
                    ow_idx = torch.randint(0, n_ow, (min(args.batch_size, n_ow),), device=device)
                    ow_pred = mod.base(off_win_batch['norm_obs'][ow_idx])
                    loss_ow = (ow_pred - off_win_batch['base_out'][ow_idx]).square().mean()
                    loss = loss + args.off_window_weight * loss_ow

                prox_app = sum((args.l0_prox_mult if n == 'actor.approach.0.weight' else 1.0)
                               * (p - anchor_params[n]).square().mean() for n, p in app_params) / len(app_params)
                loss = loss + args.proximal_weight * prox_app
                if bas_params:
                    prox_bas = sum((p - anchor_params[n]).square().mean() for n, p in bas_params) / len(bas_params)
                    loss = loss + args.base_prox_weight * prox_bas

                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for _, p in all_trainable], 2.0)
                opt.step()
            sched.step()

            if epoch % 5 == 0 or epoch == args.epochs:
                actor.eval()
                mn = eval_metrics(actor, t_val_nom, arm_only=True, base_actor=base_actor)
                mr = eval_metrics(actor, r_val, arm_only=False)
                score = args.teacher_weight * mn['mse'] + args.retention_weight * mr['mse']
                dpo_val_s, dpo_acc = 0.0, 0.0
                if d_val is not None and args.dpo_weight > 0:
                    with torch.no_grad():
                        pd = forward_fast(mod, d_val['norm_obs'], d_val['base_out'], d_val['agate'], noise_std=0.0)
                        ld, dpo_acc = dpo_distance_loss(pd, d_val['a_pos'], d_val['a_neg'], d_val['weight'], beta=args.dpo_beta)
                        dpo_val_s = float(ld)
                        score += args.dpo_weight * dpo_val_s
                ow_rms = 0.0
                if off_win_batch is not None:
                    with torch.no_grad():
                        ow_p = mod.base(off_win_batch['norm_obs'][:2048])
                        ow_mse = float((ow_p - off_win_batch['base_out'][:2048]).square().mean())
                        ow_rms = math.sqrt(ow_mse)
                        score += args.off_window_weight * ow_mse
                if score < best_score:
                    best_score = score
                    best_state = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
                    best_epoch = epoch
                if (epoch % 65 == 0 or epoch == args.epochs) and member_idx == 0:
                    print(f'  [m{member_idx}] ep={epoch:3d} val_NOM={mn["rms_arm"]:.4f} cos={mn["cos_delta"]:+.3f} '
                          f'mag={mn["mag_ratio"]:.2f} | RET_arm={mr["rms_arm"]:.4f} leg={mr["rms_leg"]:.4f} '
                          f'| DPO={dpo_val_s:.3f}(acc={dpo_acc:.2f}) ow={ow_rms:.4f} best@{best_epoch}')
        return best_state, best_epoch, (t_trn_nom, t_val_nom, t_val_tea, r_trn, r_val, d_val)

    def make_splits(eps, k_folds: int):
        if not eps:
            return [([], []) for _ in range(max(1, k_folds))]
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
    d_splits = make_splits(dpo_episodes, args.n_soup)

    member_states, member_epochs = [], []
    eval_pack = None
    for m_idx in range(args.n_soup):
        st_m, ep_m, pack_m = train_one_member(
            m_idx, t_splits[m_idx][0], t_splits[m_idx][1],
            r_splits[m_idx][0], r_splits[m_idx][1],
            d_splits[m_idx][0], d_splits[m_idx][1],
        )
        member_states.append(st_m)
        member_epochs.append(ep_m)
        if m_idx == 0:
            eval_pack = pack_m

    if args.n_soup > 1:
        best_state = {k: v.clone() for k, v in init_full_state.items()}
        for name in trainable_names:
            best_state[name] = sum(s[name].float() for s in member_states) / len(member_states)
            best_state[name] = best_state[name].to(init_full_state[name].dtype)
        best_epoch = int(round(float(np.mean(member_epochs))))
    else:
        best_state = member_states[0]
        best_epoch = member_epochs[0]

    t_trn_nom, t_val_nom, t_val_tea, r_trn, r_val, d_val = eval_pack
    actor.load_state_dict(best_state, strict=True)
    actor.eval()
    stats = dict(
        trn_nom=eval_metrics(actor, t_trn_nom, arm_only=True, base_actor=base_actor),
        val_nom=eval_metrics(actor, t_val_nom, arm_only=True, base_actor=base_actor),
        val_tea=eval_metrics(actor, t_val_tea, arm_only=True, base_actor=base_actor),
        ret_trn=eval_metrics(actor, r_trn, arm_only=False),
        ret_val=eval_metrics(actor, r_val, arm_only=False),
    )
    if d_val is not None:
        with torch.no_grad():
            pd = forward_fast(actor.actor, d_val['norm_obs'], d_val['base_out'], d_val['agate'], noise_std=0.0)
            ld, acc = dpo_distance_loss(pd, d_val['a_pos'], d_val['a_neg'], d_val['weight'], beta=args.dpo_beta)
            stats['dpo_val'] = dict(loss=float(ld), acc=float(acc))
    if off_win_batch is not None:
        with torch.no_grad():
            ow_p = actor.actor.base(off_win_batch['norm_obs'])
            stats['off_window_rms'] = float((ow_p - off_win_batch['base_out']).square().mean().sqrt())
            stats['off_window_max'] = float((ow_p - off_win_batch['base_out']).abs().max())

    print(f'FINAL (soup={args.n_soup}, unlock_base={args.unlock_base}, best@{best_epoch}):', json.dumps(stats, indent=2))
    param_deltas = torch.cat([(best_state[name].float() - anchor_params[name].cpu().float()).flatten()
                              for name in sorted(trainable_names)])
    print(f'Param delta vs base: rms={float(param_deltas.square().mean().sqrt()):.6f}, max={float(param_deltas.abs().max()):.6f}')

    for k, v in ckpt['actor'].items():
        if k not in trainable_names and torch.is_tensor(v):
            assert torch.equal(v.cpu(), best_state[k].cpu()), f'Frozen param changed: {k}'

    out_ckpt = {k: v for k, v in ckpt.items()
                if k in ('algorithm', 'adapter_schema', 'actor', 'metadata', 'config',
                         'actor_updates', 'world_updates', 'critic_updates')}
    out_ckpt['actor'] = best_state
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ckpt, args.output)
    args.output.with_suffix('.distill.json').write_text(json.dumps(dict(
        base=str(args.base_checkpoint), teachers=len(teachers), retentions=len(retentions),
        dpo_cases=len(dpo_episodes), unlock_base=args.unlock_base,
        dpo_weight=args.dpo_weight, retention_weight=args.retention_weight,
        center_bias_retention=args.center_bias_retention,
        best_epoch=best_epoch, stats=stats,
        param_delta_rms=float(param_deltas.square().mean().sqrt()),
        param_delta_max=float(param_deltas.abs().max()),
    ), indent=2) + '\n')
    print(f'Saved -> {args.output}')


if __name__ == '__main__':
    main()
