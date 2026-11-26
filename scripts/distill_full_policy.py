#!/usr/bin/env python3
"""Panel-A-Protected Joint actor.base + actor.approach Distillation with Pre-Takeoff/Early-Arm Support.

Key design principles:
1. Panel-A Fortress Anchor: uses ALL nominal Panel A episodes across r1..r6 (~680 serves)
   evaluated through ref_actor (final_best.pt) across the full pres & ~hc window (ticks 201..contact),
   eliminating stale-checkpoint clash and keeping Panel A locked at 91-97% (ret_A_rms < 0.020).
2. Strict Standing (~pres) & Recovery (had_contact) Anchor: locks actor.base on 25,000+ off-window
   frames at 60x weight so standing balance and 0-robot-net recovery remain bit-tight.
3. Full-Window Teacher Fusion: seamlessly ingests BOTH:
   - approach_gate teachers (Pass-1, Pass-2, Dynamic-IK on r5+r6), AND
   - present_precontact teachers (Pre-Takeoff Jump-Boost + 550ms Early-Arm on r5+r6), AND
   - optional external on-policy Panel B successes (e.g. d3's 82/128 Panel B wins on r7_nominal_256!).
4. 5-Fold Model Soup across actor.approach + actor.base (with optional --freeze-base ablation).
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


def compute_delta_schedule(num_ticks: int, start_tick: int, offset, ramp_offset, peak_tick: float, width_tick: float,
                           leg_offset=None, leg_peak_tick=None, leg_width_tick=None):
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
    if leg_offset is not None and leg_peak_tick is not None and leg_width_tick is not None:
        lrise = ((ticks - (leg_peak_tick - leg_width_tick)) / max(leg_width_tick, 1.0)).clamp(0.0, 1.0)
        lfall = ((ticks - leg_peak_tick) / 8.0).clamp(0.0, 1.0)
        lenv = torch.where(ticks <= leg_peak_tick, torch.sin(0.5 * math.pi * lrise).square(),
                           torch.cos(0.5 * math.pi * lfall).square())
        loff = torch.tensor(leg_offset, dtype=torch.float32)
        delta = delta + lenv[:, None] * loff[None, :]
    return delta  # [T, 18]


def extract_frames(tape, lane: int, *, gate_mode: str = 'approach_gate', min_gate: float = 0.05,
                   override_delta=None, arm_only_mask: bool = False):
    ep0 = tape['episode'][:, lane] == 0
    hc = tape['had_contact'][:, lane]
    pres = tape['present'][:, lane]
    gate = tape['gate'][:, lane].float()
    if gate_mode == 'present_precontact':
        valid = ep0 & ~hc & pres
    else:
        valid = ep0 & ~hc & pres & (gate.amax(-1) > min_gate)
    idxs = valid.nonzero().flatten()
    if len(idxs) == 0:
        return None
    hc_idxs = (ep0 & hc).nonzero().flatten()
    c_tick = int(hc_idxs[0]) if len(hc_idxs) else int(idxs[-1]) + 1
    dist_to_c = (c_tick - idxs).clamp_min(0)
    if gate_mode == 'present_precontact':
        # High weight near contact (<=20t) and during jump pushoff (20..40t)
        weights = torch.where(dist_to_c <= 18, 4.0, torch.where(dist_to_c <= 42, 2.2, 1.0)).float()
    else:
        weights = torch.where(dist_to_c <= 15, 4.5, torch.where(dist_to_c <= 30, 2.0, 1.0)).float()
    req = tape['request'][idxs, lane].float()
    if override_delta is not None:
        if gate_mode == 'approach_gate':
            req = (req + gate[idxs] * override_delta[idxs]).clamp(-1.0, 1.0)
        else:
            req = (req + override_delta[idxs]).clamp(-1.0, 1.0)
    # Build per-joint active mask [N, 18]
    if gate_mode == 'present_precontact':
        jmask = torch.ones_like(req)
        if arm_only_mask:
            jmask[:, :12] = 0.0
    else:
        jmask = (gate[idxs] > 0.02).float()
        if arm_only_mask:
            jmask[:, :12] = 0.0
    return dict(
        obs=tape['observation'][idxs, lane].float(),
        req=req,
        jmask=jmask,
        weight=weights,
    )


def extract_off_window(tape, lane: int, *, stride: int = 3):
    ep0 = tape['episode'][:, lane] == 0
    hc = tape['had_contact'][:, lane]
    pres = tape['present'][:, lane]
    off = ep0 & (hc | (~pres))
    idxs = off.nonzero().flatten()
    if len(idxs) == 0:
        return None
    return tape['observation'][idxs[::stride], lane].float()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-checkpoint', type=Path, required=True)
    parser.add_argument('--onpolicy-nominal-dirs', type=Path, nargs='+', required=True,
                        help='Nominal dirs rolled out on base-checkpoint (e.g. r5, r6)')
    parser.add_argument('--panel-a-anchor-dirs', type=Path, nargs='+', default=[],
                        help='Extra nominal dirs (e.g. r1..r4) whose Panel A obs anchor to base-checkpoint(obs)')
    parser.add_argument('--extra-panel-b-teacher-dirs', type=Path, nargs='+', default=[],
                        help='Optional on-policy dirs (e.g. r7_nominal on d3) whose Panel B successes become teachers')
    parser.add_argument('--probe-dirs', type=Path, nargs='+', required=True)
    parser.add_argument('--probe-specs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--unlock-base', action='store_true', default=True)
    parser.add_argument('--freeze-base', dest='unlock_base', action='store_false')
    parser.add_argument('--epochs', type=int, default=220)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1.6e-4)
    parser.add_argument('--base-lr-mult', type=float, default=0.35)
    parser.add_argument('--retention-weight', type=float, default=5.0)
    parser.add_argument('--panel-a-ret-mult', type=float, default=2.8)
    parser.add_argument('--off-window-weight', type=float, default=60.0)
    parser.add_argument('--proximal-weight', type=float, default=0.04)
    parser.add_argument('--base-prox-weight', type=float, default=0.30)
    parser.add_argument('--norm-obs-noise', type=float, default=0.02)
    parser.add_argument('--max-teacher-delta', type=float, default=0.62)
    parser.add_argument('--nominal-pair-weight', type=float, default=4.0)
    parser.add_argument('--n-soup', type=int, default=5)
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
    ref_actor, _ = load_ppo_initialized_actor(source_path, device)
    ref_actor.load_state_dict(ckpt['actor'], strict=True)
    arm_prep = ckpt['metadata'].get('policy_execution', {}).get('arm_preparation_seconds', 0.25)
    if arm_prep is not None:
        actor.configure_approach_gate(arm_prep)
        ref_actor.configure_approach_gate(arm_prep)
    ref_actor.eval()
    for p in ref_actor.parameters():
        p.requires_grad_(False)

    base_names = {k for k in ckpt['actor'].keys() if k.startswith('actor.base.')}
    trainable_names = (set(APPROACH_NAMES) | base_names) if args.unlock_base else set(APPROACH_NAMES)
    anchor_params = {name: p.detach().clone() for name, p in actor.named_parameters() if name in trainable_names}
    init_full_state = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}

    # 1. Load on-policy nominal dirs (r5, r6 rolled out on base-checkpoint)
    nom_tapes = {}
    retentions = {}
    off_win_obs_list = []
    full_window = args.unlock_base

    for ndir in args.onpolicy_nominal_dirs:
        summary = json.loads((ndir / 'probe_summary.json').read_text())
        blob = torch.load(ndir / 'probe_tape.pt', map_location='cpu', weights_only=True)
        tape = blob['tape']
        start_tick = int(blob['start_capture_tick'])
        for i, row in enumerate(summary['rows']):
            cid = row['case_id']
            is_panel_a = (i < 128)
            nom_tapes[cid] = (tape, i, start_tick, row, is_panel_a)
            if args.unlock_base and (i % 2 == 0):
                ow = extract_off_window(tape, i, stride=3)
                if ow is not None:
                    off_win_obs_list.append(ow)
            if row['success'] and not row['robot_net'] and not row['low_speed']:
                gmode = 'present_precontact' if full_window else 'approach_gate'
                ep = extract_frames(tape, i, gate_mode=gmode)
                if ep is not None:
                    sm = row['string_margin_m'] or 0.0
                    spm = row['speed_margin_mps'] or 0.0
                    lm = row.get('landing_margin_m') or 0.0
                    if is_panel_a:
                        # Panel A Fortress: strong uniform retention on ALL Panel A successes!
                        w_mult = args.panel_a_ret_mult
                        aw = 1.0
                    else:
                        w_mult = 1.0
                        sm_w = max(0.45, min(1.0, (sm - 0.010) / 0.025))
                        sp_w = max(0.50, min(1.0, spm / 0.35))
                        lm_w = max(0.50, min(1.0, lm / 0.25))
                        aw = max(0.40, sm_w * sp_w * lm_w)
                    ep['weight'] = ep['weight'] * w_mult
                    j_scale = torch.ones_like(ep['jmask'])
                    j_scale[:, 12:18] = aw
                    ep['jmask'] = ep['jmask'] * j_scale
                    retentions[cid] = dict(case_id=cid, is_panel_a=is_panel_a, row=row, **ep)

    # 2. Panel-A Fortress Expansion: anchor ALL Panel A obs from r1..r4 directly to ref_actor(obs)!
    n_extra_a = 0
    for ndir in args.panel_a_anchor_dirs:
        summary = json.loads((ndir / 'probe_summary.json').read_text())
        blob = torch.load(ndir / 'probe_tape.pt', map_location='cpu', weights_only=True)
        tape = blob['tape']
        for i, row in enumerate(summary['rows'][:128]):  # Panel A ONLY (first 128 lanes)
            cid = row['case_id']
            if cid in retentions:
                continue
            if args.unlock_base and (i % 4 == 0):
                ow = extract_off_window(tape, i, stride=4)
                if ow is not None:
                    off_win_obs_list.append(ow)
            gmode = 'present_precontact' if full_window else 'approach_gate'
            ep = extract_frames(tape, i, gate_mode=gmode)
            if ep is None:
                continue
            # Compute EXACT ref_actor (final_best.pt) target on these Panel A observations!
            with torch.no_grad():
                o_gpu = ep['obs'].to(device)
                no_rec = torch.zeros((len(o_gpu), 1), dtype=torch.bool, device=device)
                ep['req'] = ref_actor._gated_distribution(o_gpu, no_rec).loc.clamp(-1.0, 1.0).cpu()
            ep['weight'] = ep['weight'] * (args.panel_a_ret_mult * 0.85)
            retentions[cid] = dict(case_id=cid, is_panel_a=True, row=row, **ep)
            n_extra_a += 1

    # 3. Load probe teachers (BOTH approach_gate AND present_precontact!)
    teachers = {}
    n_pre_teachers = 0
    for pdir, pspec_path in zip(args.probe_dirs, args.probe_specs, strict=True):
        pspec = json.loads(Path(pspec_path).read_text())
        summary = json.loads((pdir / 'probe_summary.json').read_text())
        blob = torch.load(pdir / 'probe_tape.pt', map_location='cpu', weights_only=True)
        tape = blob['tape']
        gmode = pspec.get('gate_mode', 'approach_gate')
        if gmode != 'approach_gate' and not args.unlock_base:
            continue
        offsets = pspec['offsets']
        ramps = pspec.get('ramp_offsets') or [None] * len(offsets)
        peaks = pspec['peak_ticks']
        widths = pspec['width_ticks']
        loffs = pspec.get('leg_offsets') or [None] * len(offsets)
        lpts = pspec.get('leg_peak_ticks') or [None] * len(offsets)
        lwts = pspec.get('leg_width_ticks') or [None] * len(offsets)
        meta = summary.get('branch_meta') or [None] * len(summary['rows'])

        for i, row in enumerate(summary['rows']):
            cid = row['case_id']
            if cid not in nom_tapes:
                continue
            if not (row['success'] and row['string'] and not row['robot_net']
                    and not row['ball_net'] and not row['low_speed']):
                continue
            off = np.array(offsets[i], dtype=np.float64)
            rmp = np.array(ramps[i], dtype=np.float64) if ramps[i] is not None else np.zeros(18)
            loff = np.array(loffs[i], dtype=np.float64) if loffs[i] is not None else np.zeros(18)
            eff_max = float(max(np.abs(off + rmp).max(), np.abs(loff).max()))
            eff_l2 = float(np.linalg.norm(off) + np.linalg.norm(rmp) + np.linalg.norm(loff))
            if eff_max > args.max_teacher_delta:
                continue

            ntape, nlane, nstart, nrow, is_pa = nom_tapes[cid]
            # NEVER overwrite a solid Panel A success unless it was actually failing!
            if is_pa and nrow['success']:
                continue
            if cid in retentions and not nrow['success']:
                pass
            elif cid in retentions:
                old_r = retentions[cid]['row']
                old_lm = old_r.get('landing_margin_m') or 0.0
                lm = row.get('landing_margin_m') or 0.0
                spm = row['speed_margin_mps'] or 0.0
                if not ((row['margin_score'] >= old_r['margin_score'] + 0.08) or (lm >= old_lm + 0.14 and spm >= 0.25)):
                    continue

            sm = row['string_margin_m'] or 0.0
            spm = row['speed_margin_mps'] or 0.0
            lm = row.get('landing_margin_m') or 0.0
            tier = 2 if (spm >= 0.30 and sm >= 0.012 and lm >= 0.15) else (1 if (spm >= 0.15 and sm >= 0.006 and lm >= 0.05) else 0)
            sel_rank = (tier, -eff_l2 + 0.18 * min(spm, 1.2) + 2.5 * min(sm, 0.04) + 0.40 * min(lm, 0.40))

            # If leg_offset is non-zero, supervise all 18 DoF; otherwise arm + active legs anchored to ref!
            has_leg_delta = (np.abs(loff[:12]).max() > 1e-4)
            ep_tea = extract_frames(tape, i, gate_mode=gmode)
            if ep_tea is None:
                continue
            if not has_leg_delta:
                # Keep legs supervised to nominal/ref so arm teachers don't perturb legs
                ep_tea['jmask'][:, :12] = 1.0 if full_window else (ep_tea['jmask'][:, :12])

            ep_nom = None
            if args.nominal_pair_weight > 0:
                delta_sched = compute_delta_schedule(
                    ntape['observation'].shape[0], nstart, offsets[i], ramps[i], peaks[i], widths[i],
                    leg_offset=loffs[i], leg_peak_tick=lpts[i], leg_width_tick=lwts[i])
                ep_nom = extract_frames(ntape, nlane, gate_mode=gmode, override_delta=delta_sched)
                if ep_nom is not None:
                    if not has_leg_delta and full_window:
                        ep_nom['jmask'][:, :12] = 1.0
                    ep_nom['weight'] = ep_nom['weight'] * args.nominal_pair_weight

            prev = teachers.get(cid)
            if prev is None or sel_rank > prev['sel_rank']:
                teachers[cid] = dict(
                    case_id=cid, sel_rank=sel_rank, eff_max=eff_max, eff_l2=eff_l2,
                    gmode=gmode, has_leg=has_leg_delta, is_panel_a=is_pa,
                    teacher_ep=ep_tea, nom_paired_ep=ep_nom,
                )
                retentions.pop(cid, None)

    # 4. Optional external Panel B verified complete-return teachers (e.g. d3's 82/128 Panel B wins on r7!)
    n_ext_b = 0
    for edir in args.extra_panel_b_teacher_dirs:
        summary = json.loads((edir / 'probe_summary.json').read_text())
        blob = torch.load(edir / 'probe_tape.pt', map_location='cpu', weights_only=True)
        tape = blob['tape']
        for i, row in enumerate(summary['rows'][128:], start=128):  # Panel B ONLY
            cid = row['case_id']
            if cid in teachers or cid in retentions:
                continue
            if not (row['success'] and row['string'] and not row['robot_net']
                    and not row['ball_net'] and not row['low_speed']):
                continue
            gmode = 'present_precontact' if full_window else 'approach_gate'
            ep_tea = extract_frames(tape, i, gate_mode=gmode)
            if ep_tea is None:
                continue
            ep_tea['weight'] = ep_tea['weight'] * 1.4
            teachers[cid] = dict(
                case_id=cid, sel_rank=(2, 0.0), eff_max=0.10, eff_l2=0.10,
                gmode=gmode, has_leg=False, is_panel_a=False,
                teacher_ep=ep_tea, nom_paired_ep=None,
            )
            n_ext_b += 1

    teacher_list = list(teachers.values())
    ret_list = list(retentions.values())
    n_pre_teachers = sum(1 for t in teacher_list if t['gmode'] == 'present_precontact' and t['nom_paired_ep'] is not None)
    n_ret_a = sum(1 for r in ret_list if r['is_panel_a'])
    n_ret_b = len(ret_list) - n_ret_a
    print(f'Dataset: teachers={len(teacher_list)} (early_base={n_pre_teachers}, ext_r7_B={n_ext_b}) | '
          f'retentions={len(ret_list)} (Panel_A={n_ret_a} [+{n_extra_a} ref-anchored], Panel_B={n_ret_b}) | '
          f'unlock_base={args.unlock_base}')

    @torch.no_grad()
    def precompute_batch(eps, is_teacher=False, mode='both'):
        obs_l, req_l, jm_l, w_l = [], [], [], []
        for e in eps:
            if is_teacher:
                if mode in ('both', 'teacher') and e['teacher_ep'] is not None:
                    ep = e['teacher_ep']
                    obs_l.append(ep['obs']); req_l.append(ep['req']); jm_l.append(ep['jmask']); w_l.append(ep['weight'])
                if mode in ('both', 'nominal') and e['nom_paired_ep'] is not None:
                    ep = e['nom_paired_ep']
                    obs_l.append(ep['obs']); req_l.append(ep['req']); jm_l.append(ep['jmask']); w_l.append(ep['weight'])
            else:
                obs_l.append(e['obs']); req_l.append(e['req']); jm_l.append(e['jmask']); w_l.append(e['weight'])
        if not obs_l:
            return None
        obs = torch.cat(obs_l, dim=0).to(device)
        mod = ref_actor.actor
        norm_obs = mod.normalizer(obs)
        base_out = mod.base(norm_obs)
        agate = ref_actor.approach_action_gate(obs)
        ref_req = (base_out + agate * mod.approach(norm_obs) * mod.approach_scale).clamp(-1.0, 1.0)
        return dict(
            obs=obs, norm_obs=norm_obs, base_out=base_out, agate=agate, ref_req=ref_req,
            req=torch.cat(req_l, dim=0).to(device),
            jmask=torch.cat(jm_l, dim=0).to(device),
            weight=torch.cat(w_l, dim=0).to(device),
        )

    off_win_batch = None
    if args.unlock_base and off_win_obs_list:
        ow_obs = torch.cat(off_win_obs_list, dim=0).to(device)
        with torch.no_grad():
            ow_norm = ref_actor.actor.normalizer(ow_obs)
            ow_base = ref_actor.actor.base(ow_norm)
        off_win_batch = dict(norm_obs=ow_norm, base_out=ow_base)
        print(f'Off-window (~pres | had_contact) anchor: {len(ow_norm)} frames')

    def forward_fast(mod, norm_obs, base_out, agate, noise_std=0.0):
        inp = norm_obs + torch.randn_like(norm_obs) * noise_std if noise_std > 0.0 else norm_obs
        b_out = mod.base(inp) if args.unlock_base else base_out
        return (b_out + agate * mod.approach(inp) * mod.approach_scale).clamp(-1.0, 1.0)

    def masked_mse(pred, target, jmask, weight):
        sq = (pred - target).square() * jmask
        pf = sq.sum(-1) / jmask.sum(-1).clamp_min(1.0)
        return (pf * weight).sum() / weight.sum().clamp_min(1.0)

    @torch.no_grad()
    def eval_pack_metrics(batch):
        if batch is None:
            return dict(mse=0.0, rms_arm=0.0, rms_leg=0.0, cos_arm=0.0, mag_arm=0.0)
        pred = forward_fast(actor.actor, batch['norm_obs'], batch['base_out'], batch['agate'], noise_std=0.0)
        mse = float(masked_mse(pred, batch['req'], batch['jmask'], batch['weight']))
        diff = pred - batch['req']
        ma = batch['jmask'][:, 12:18] > 0.05
        ml = batch['jmask'][:, :12] > 0.05
        rms_a = float(diff[:, 12:18][ma].square().mean().sqrt()) if ma.any() else 0.0
        rms_l = float(diff[:, :12][ml].square().mean().sqrt()) if ml.any() else 0.0
        dt = (batch['req'] - batch['ref_req'])[:, 12:18][ma]
        dp = (pred - batch['ref_req'])[:, 12:18][ma]
        cos_a = float(F.cosine_similarity(dp, dt, dim=0)) if (dt.norm() > 1e-4 and dp.norm() > 1e-6) else 0.0
        mag_a = float(dp.norm() / dt.norm()) if (dt.norm() > 1e-4 and dp.norm() > 1e-6) else 0.0
        return dict(mse=mse, rms_arm=rms_a, rms_leg=rms_l, cos_arm=cos_a, mag_arm=mag_a)

    def make_splits(eps, k_folds: int):
        perm = np.random.permutation(len(eps))
        if k_folds <= 1:
            nv = max(1, int(round(len(eps) * 0.15)))
            return [([eps[i] for i in perm[nv:]], [eps[i] for i in perm[:nv]])]
        folds = np.array_split(perm, k_folds)
        return [([eps[i] for i in perm if i not in set(folds[f].tolist())],
                 [eps[i] for i in perm if i in set(folds[f].tolist())]) for f in range(k_folds)]

    t_splits = make_splits(teacher_list, args.n_soup)
    r_splits = make_splits(ret_list, args.n_soup)

    member_states, member_epochs = [], []
    eval_batches = None
    for m_idx in range(args.n_soup):
        seed_m = args.seed + 1009 * m_idx
        torch.manual_seed(seed_m); np.random.seed(seed_m)
        actor.load_state_dict(init_full_state, strict=True); actor.to(device)
        for name, p in actor.named_parameters():
            p.requires_grad_(name in trainable_names)
        app_p = [(n, p) for n, p in actor.named_parameters() if n in APPROACH_NAMES]
        bas_p = [(n, p) for n, p in actor.named_parameters() if n in base_names] if args.unlock_base else []
        all_p = app_p + bas_p

        t_trn = precompute_batch(t_splits[m_idx][0], is_teacher=True, mode='both')
        t_val = precompute_batch(t_splits[m_idx][1], is_teacher=True, mode='both')
        r_trn = precompute_batch(r_splits[m_idx][0], is_teacher=False)
        r_val_a = precompute_batch([e for e in r_splits[m_idx][1] if e['is_panel_a']], is_teacher=False)
        r_val_b = precompute_batch([e for e in r_splits[m_idx][1] if not e['is_panel_a']], is_teacher=False)
        r_val = precompute_batch(r_splits[m_idx][1], is_teacher=False)

        if m_idx == 0:
            eval_batches = (t_val, r_val_a, r_val_b)
            print(f'Frames: teacher={len(t_trn["obs"])}, ret={len(r_trn["obs"])}, val_t={len(t_val["obs"])}, val_retA={len(r_val_a["obs"]) if r_val_a else 0}, val_retB={len(r_val_b["obs"]) if r_val_b else 0}')

        pg = [{'params': [p for _, p in app_p], 'lr': args.lr}]
        if bas_p:
            pg.append({'params': [p for _, p in bas_p], 'lr': args.lr * args.base_lr_mult})
        opt = torch.optim.Adam(pg)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.05)

        best_score = float('inf')
        best_st = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
        best_ep = 0
        n_t, n_r = len(t_trn['obs']), len(r_trn['obs'])
        n_ow = len(off_win_batch['norm_obs']) if off_win_batch is not None else 0
        steps = max(1, math.ceil(max(n_t, n_r) / args.batch_size))
        mod = actor.actor

        for ep in range(1, args.epochs + 1):
            actor.train()
            for _ in range(steps):
                opt.zero_grad(set_to_none=True)
                ti = torch.randint(0, n_t, (min(args.batch_size, n_t),), device=device)
                pt = forward_fast(mod, t_trn['norm_obs'][ti], t_trn['base_out'][ti], t_trn['agate'][ti], args.norm_obs_noise)
                lt = masked_mse(pt, t_trn['req'][ti], t_trn['jmask'][ti], t_trn['weight'][ti])

                ri = torch.randint(0, n_r, (min(args.batch_size, n_r),), device=device)
                pr = forward_fast(mod, r_trn['norm_obs'][ri], r_trn['base_out'][ri], r_trn['agate'][ri], args.norm_obs_noise * 0.5)
                lr_loss = masked_mse(pr, r_trn['req'][ri], r_trn['jmask'][ri], r_trn['weight'][ri])

                loss = lt + args.retention_weight * lr_loss
                if off_win_batch is not None and args.off_window_weight > 0:
                    oi = torch.randint(0, n_ow, (min(args.batch_size, n_ow),), device=device)
                    loss = loss + args.off_window_weight * (mod.base(off_win_batch['norm_obs'][oi]) - off_win_batch['base_out'][oi]).square().mean()

                prox_a = sum((p - anchor_params[n]).square().mean() for n, p in app_p) / len(app_p)
                loss = loss + args.proximal_weight * prox_a
                if bas_p:
                    prox_b = sum((p - anchor_params[n]).square().mean() for n, p in bas_p) / len(bas_p)
                    loss = loss + args.base_prox_weight * prox_b
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for _, p in all_p], 2.0)
                opt.step()
            sched.step()

            if ep % 5 == 0 or ep == args.epochs:
                actor.eval()
                mt = eval_pack_metrics(t_val)
                mr = eval_pack_metrics(r_val)
                sc = mt['mse'] + args.retention_weight * mr['mse']
                ow_r = 0.0
                if off_win_batch is not None:
                    with torch.no_grad():
                        ow_m = float((mod.base(off_win_batch['norm_obs'][:2048]) - off_win_batch['base_out'][:2048]).square().mean())
                        ow_r = math.sqrt(ow_m)
                        sc += args.off_window_weight * ow_m
                if sc < best_score:
                    best_score = sc
                    best_st = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
                    best_ep = ep
                if (ep % 70 == 0 or ep == args.epochs) and m_idx == 0:
                    mra = eval_pack_metrics(r_val_a)
                    print(f'  [m0] ep={ep:3d} T_rms={mt["rms_arm"]:.4f} cos={mt["cos_arm"]:+.3f} mag={mt["mag_arm"]:.2f} | '
                          f'retA={mra["rms_arm"]:.4f}(leg={mra["rms_leg"]:.4f}) ow={ow_r:.4f} best@{best_ep}')
        member_states.append(best_st)
        member_epochs.append(best_ep)

    if args.n_soup > 1:
        final_st = {k: v.clone() for k, v in init_full_state.items()}
        for name in trainable_names:
            final_st[name] = (sum(s[name].float() for s in member_states) / len(member_states)).to(init_full_state[name].dtype)
        best_ep = int(round(float(np.mean(member_epochs))))
    else:
        final_st = member_states[0]
        best_ep = member_epochs[0]

    actor.load_state_dict(final_st, strict=True); actor.eval()
    t_val, r_val_a, r_val_b = eval_batches
    stats = dict(
        val_tea=eval_pack_metrics(t_val),
        ret_val_A=eval_pack_metrics(r_val_a),
        ret_val_B=eval_pack_metrics(r_val_b),
    )
    if off_win_batch is not None:
        with torch.no_grad():
            ow_d = actor.actor.base(off_win_batch['norm_obs']) - off_win_batch['base_out']
            stats['off_window_rms'] = float(ow_d.square().mean().sqrt())
            stats['off_window_max'] = float(ow_d.abs().max())
    print(f'FINAL (soup={args.n_soup}, unlock_base={args.unlock_base}, best@{best_ep}):', json.dumps(stats, indent=2))
    for k, v in ckpt['actor'].items():
        if k not in trainable_names and torch.is_tensor(v):
            assert torch.equal(v.cpu(), final_st[k].cpu()), f'Frozen param changed: {k}'

    out_ckpt = {k: v for k, v in ckpt.items()
                if k in ('algorithm', 'adapter_schema', 'actor', 'metadata', 'config',
                         'actor_updates', 'world_updates', 'critic_updates')}
    out_ckpt['actor'] = final_st
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ckpt, args.output)
    args.output.with_suffix('.distill.json').write_text(json.dumps(dict(
        base=str(args.base_checkpoint), teachers=len(teacher_list), retentions=len(ret_list),
        ret_A=n_ret_a, ret_B=n_ret_b, early_base_teachers=n_pre_teachers, ext_r7_B=n_ext_b,
        unlock_base=args.unlock_base, best_epoch=best_ep, stats=stats,
    ), indent=2) + '\n')
    print(f'Saved -> {args.output}')


if __name__ == '__main__':
    main()
