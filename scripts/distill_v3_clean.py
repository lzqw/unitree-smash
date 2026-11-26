#!/usr/bin/env python3
"""Clean Unified-Manifold Distillation (Fresh-from-A40 or Warm-Start + Arm-Only base.6[12:18]).

Fixes every single subtle bottleneck uncovered in P21..P25 forensics:
1. Unified Ref-Manifold Retention: all ~980 nominal successes across r1..r6 (Panel A & B)
   can be labeled with ref_checkpoint (final_best.pt) outputs so r1..r4 NEVER clash with
   r5..r6 even when training fresh from A40_safe_rec.pt!
2. Speed-Safe Teacher Filtering (--min-speed-margin 0.20): eliminates borderline 4.51 m/s
   and negative-ramp slowdown teachers that caused low_speed=6 regressions in F1..F4.
3. Zero-Leg-Drift Base Arm Unlock (--unlock-base-arm6): freezes actor.base.0/2/4 AND
   actor.base.6[0:12] completely (0.0000000 leg drift during standing, run-up, jump takeoff,
   flight, and recovery!), unlocking ONLY actor.base.6[12:18] (390 arm weights) for 550ms
   early-arm wind-up alongside full actor.approach (111k params)!
4. No premature epoch-5 early stopping (--min-epoch 50) + 5-fold Model Soup.
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


def extract_frames(tape, lane: int, *, gate_mode: str = 'approach_gate', min_gate: float = 0.05, override_delta=None):
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
    weights = torch.where(dist_to_c <= 15, 4.5, torch.where(dist_to_c <= 30, 2.0, 1.0)).float()
    req = tape['request'][idxs, lane].float()
    if override_delta is not None:
        if gate_mode == 'approach_gate':
            req = (req + gate[idxs] * override_delta[idxs]).clamp(-1.0, 1.0)
        else:
            req = (req + override_delta[idxs]).clamp(-1.0, 1.0)
    jmask = torch.ones_like(req) if gate_mode == 'present_precontact' else (gate[idxs] > 0.02).float()
    return dict(obs=tape['observation'][idxs, lane].float(), req=req, jmask=jmask, weight=weights)


def extract_off_window_arm(tape, lane: int, *, stride: int = 4):
    ep0 = tape['episode'][:, lane] == 0
    hc = tape['had_contact'][:, lane]
    pres = tape['present'][:, lane]
    idxs = (ep0 & (hc | (~pres))).nonzero().flatten()
    if len(idxs) == 0:
        return None
    return tape['observation'][idxs[::stride], lane].float()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--init-checkpoint', type=Path, required=True, help='Starting weights (A40_safe_rec or final_best)')
    parser.add_argument('--ref-checkpoint', type=Path, required=True, help='Unified reference policy for relabeling r1..r4 retentions (final_best)')
    parser.add_argument('--nominal-dirs', type=Path, nargs='+', required=True)
    parser.add_argument('--probe-dirs', type=Path, nargs='+', required=True)
    parser.add_argument('--probe-specs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--unlock-base-arm6', action='store_true', default=False,
                        help='Unlock ONLY actor.base.6[12:18] (0.000000 leg drift, 550ms early arm!)')
    parser.add_argument('--relabel-old-retentions', action='store_true', default=True)
    parser.add_argument('--epochs', type=int, default=280)
    parser.add_argument('--min-epoch', type=int, default=55)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=2.0e-4)
    parser.add_argument('--retention-weight', type=float, default=3.5)
    parser.add_argument('--panel-a-ret-mult', type=float, default=1.6)
    parser.add_argument('--round-retention-decay', type=float, default=0.70)
    parser.add_argument('--proximal-weight', type=float, default=0.02)
    parser.add_argument('--off-window-weight', type=float, default=50.0)
    parser.add_argument('--min-speed-margin', type=float, default=0.20, help='Require outgoing speed >= 4.5 + margin (m/s)')
    parser.add_argument('--max-teacher-delta', type=float, default=0.52)
    parser.add_argument('--nominal-pair-weight', type=float, default=4.5)
    parser.add_argument('--norm-obs-noise', type=float, default=0.02)
    parser.add_argument('--n-soup', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    init_ckpt = torch.load(args.init_checkpoint, map_location='cpu', weights_only=False)
    ref_ckpt = torch.load(args.ref_checkpoint, map_location='cpu', weights_only=False)
    source_path = Path(ref_ckpt['metadata']['init_policy_checkpoint']['path'])

    actor, _ = load_ppo_initialized_actor(source_path, device)
    actor.load_state_dict(init_ckpt['actor'], strict=True)
    ref_actor, _ = load_ppo_initialized_actor(source_path, device)
    ref_actor.load_state_dict(ref_ckpt['actor'], strict=True)
    arm_prep = ref_ckpt['metadata'].get('policy_execution', {}).get('arm_preparation_seconds', 0.25)
    if arm_prep is not None:
        actor.configure_approach_gate(arm_prep)
        ref_actor.configure_approach_gate(arm_prep)
    ref_actor.eval()
    for p in ref_actor.parameters():
        p.requires_grad_(False)

    base_arm6_names = {'actor.base.6.weight', 'actor.base.6.bias'}
    trainable_names = (set(APPROACH_NAMES) | base_arm6_names) if args.unlock_base_arm6 else set(APPROACH_NAMES)
    init_full_state = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
    anchor_params = {n: p.detach().clone() for n, p in actor.named_parameters() if n in trainable_names}

    # Precompute frozen base.0..4 hidden feature extractor if unlock_base_arm6!
    @torch.no_grad()
    def extract_base_h64(norm_obs):
        x = norm_obs
        for cname, layer in ref_actor.actor.base.named_children():
            if cname == '6':
                break
            x = layer(x)
        return x  # [N, 64]

    # 1. Load nominal dirs (r1..r6)
    nom_tapes = {}
    retentions = {}
    off_win_list = []
    n_rounds = len(args.nominal_dirs)
    gmode_ret = 'present_precontact' if args.unlock_base_arm6 else 'approach_gate'

    for r_idx, ndir in enumerate(args.nominal_dirs):
        round_w = args.round_retention_decay ** (n_rounds - 1 - r_idx)
        is_recent_onpol = (r_idx >= n_rounds - 2)  # r5, r6 rolled out on final_best!
        summary = json.loads((ndir / 'probe_summary.json').read_text())
        blob = torch.load(ndir / 'probe_tape.pt', map_location='cpu', weights_only=True)
        tape = blob['tape']
        start_tick = int(blob['start_capture_tick'])
        for i, row in enumerate(summary['rows']):
            cid = row['case_id']
            is_pa = (i < 128)
            nom_tapes[cid] = (tape, i, start_tick, row, is_pa, round_w)
            if args.unlock_base_arm6 and (i % 3 == 0):
                ow = extract_off_window_arm(tape, i, stride=4)
                if ow is not None:
                    off_win_list.append(ow)
            if row['success'] and not row['robot_net'] and not row['low_speed']:
                ep = extract_frames(tape, i, gate_mode=gmode_ret)
                if ep is None:
                    continue
                if args.relabel_old_retentions and not is_recent_onpol:
                    # Relabel older rounds' nominal successes with exact final_best.pt output -> 0 stale clash!
                    with torch.no_grad():
                        og = ep['obs'].to(device)
                        nr = torch.zeros((len(og), 1), dtype=torch.bool, device=device)
                        ep['req'] = ref_actor._gated_distribution(og, nr).loc.clamp(-1.0, 1.0).cpu()
                sm = row['string_margin_m'] or 0.0
                spm = row['speed_margin_mps'] or 0.0
                lm = row.get('landing_margin_m') or 0.0
                sm_w = max(0.35, min(1.0, (sm - 0.010) / 0.025))
                sp_w = max(0.40, min(1.0, spm / 0.35))
                lm_w = max(0.40, min(1.0, lm / 0.25))
                aw = 1.0 if is_pa else max(0.35, sm_w * sp_w * lm_w)
                ep['weight'] = ep['weight'] * round_w * (args.panel_a_ret_mult if is_pa else 1.0)
                jscale = torch.ones_like(ep['jmask'])
                jscale[:, 12:18] = aw
                ep['jmask'] = ep['jmask'] * jscale
                retentions[cid] = dict(case_id=cid, is_panel_a=is_pa, row=row, **ep)

    # 2. Load speed-safe verified complete-return teachers across all probe dirs!
    teachers = {}
    n_filtered_slow = 0
    for pdir, pspec_path in zip(args.probe_dirs, args.probe_specs, strict=True):
        pspec = json.loads(Path(pspec_path).read_text())
        summary = json.loads((pdir / 'probe_summary.json').read_text())
        blob = torch.load(pdir / 'probe_tape.pt', map_location='cpu', weights_only=True)
        tape = blob['tape']
        gmode = pspec.get('gate_mode', 'approach_gate')
        if gmode != 'approach_gate' and not args.unlock_base_arm6:
            continue
        offsets = pspec['offsets']
        ramps = pspec.get('ramp_offsets') or [None] * len(offsets)
        peaks = pspec['peak_ticks']
        widths = pspec['width_ticks']
        loffs = pspec.get('leg_offsets') or [None] * len(offsets)

        for i, row in enumerate(summary['rows']):
            cid = row['case_id']
            if cid not in nom_tapes:
                continue
            if not (row['success'] and row['string'] and not row['robot_net']
                    and not row['ball_net'] and not row['low_speed']):
                continue
            # Skip leg-perturbed branches since legs 0:12 are strictly frozen!
            if loffs[i] is not None and float(np.abs(np.array(loffs[i])[:12]).max()) > 1e-4:
                continue
            spm = row['speed_margin_mps'] if row['speed_margin_mps'] is not None else 0.0
            if spm < args.min_speed_margin:
                n_filtered_slow += 1
                continue
            off = np.array(offsets[i], dtype=np.float64)
            rmp = np.array(ramps[i], dtype=np.float64) if ramps[i] is not None else np.zeros(18)
            eff_max = float(np.abs(off + rmp).max())
            eff_l2 = float(np.linalg.norm(off) + np.linalg.norm(rmp))
            if eff_max > args.max_teacher_delta:
                continue

            ntape, nlane, nstart, nrow, is_pa, round_w = nom_tapes[cid]
            if cid in retentions:
                old_r = retentions[cid]['row']
                old_lm = old_r.get('landing_margin_m') or 0.0
                lm = row.get('landing_margin_m') or 0.0
                if is_pa or not (((row['margin_score'] >= old_r['margin_score'] + 0.08) or (lm >= old_lm + 0.14)) and eff_max <= 0.42):
                    continue

            sm = row['string_margin_m'] or 0.0
            lm = row.get('landing_margin_m') or 0.0
            tier = 2 if (spm >= 0.40 and sm >= 0.012 and lm >= 0.15) else (1 if (spm >= 0.25 and sm >= 0.008 and lm >= 0.08) else 0)
            sel_rank = (tier, -eff_l2 + 0.22 * min(spm, 1.4) + 2.5 * min(sm, 0.04) + 0.40 * min(lm, 0.40))

            ep_tea = extract_frames(tape, i, gate_mode=gmode)
            if ep_tea is None:
                continue
            ep_nom = None
            if args.nominal_pair_weight > 0:
                dsched = compute_delta_schedule(ntape['observation'].shape[0], nstart, offsets[i], ramps[i], peaks[i], widths[i])
                ep_nom = extract_frames(ntape, nlane, gate_mode=gmode, override_delta=dsched)
                if ep_nom is not None:
                    ep_nom['weight'] = ep_nom['weight'] * args.nominal_pair_weight

            prev = teachers.get(cid)
            if prev is None or sel_rank > prev['sel_rank']:
                teachers[cid] = dict(
                    case_id=cid, sel_rank=sel_rank, eff_max=eff_max, eff_l2=eff_l2, gmode=gmode,
                    teacher_ep=ep_tea, nom_paired_ep=ep_nom,
                )
                retentions.pop(cid, None)

    teacher_list = list(teachers.values())
    ret_list = list(retentions.values())
    effs = [t['eff_max'] for t in teacher_list]
    print(f'Loaded {len(teacher_list)} speed-safe teachers (eff_max med={np.median(effs):.3f}, filtered_slow={n_filtered_slow}), '
          f'{len(ret_list)} unified retentions, unlock_base_arm6={args.unlock_base_arm6}')

    @torch.no_grad()
    def build_batch(eps, is_teacher=False, mode='both'):
        ol, rl, jl, wl = [], [], [], []
        for e in eps:
            if is_teacher:
                if mode in ('both', 'teacher') and e['teacher_ep'] is not None:
                    ep = e['teacher_ep']; ol.append(ep['obs']); rl.append(ep['req']); jl.append(ep['jmask']); wl.append(ep['weight'])
                if mode in ('both', 'nominal') and e['nom_paired_ep'] is not None:
                    ep = e['nom_paired_ep']; ol.append(ep['obs']); rl.append(ep['req']); jl.append(ep['jmask']); wl.append(ep['weight'])
            else:
                ol.append(e['obs']); rl.append(e['req']); jl.append(e['jmask']); wl.append(e['weight'])
        if not ol:
            return None
        obs = torch.cat(ol, dim=0).to(device)
        mod = ref_actor.actor
        norm_obs = mod.normalizer(obs)
        base_out = mod.base(norm_obs)
        base_h64 = extract_base_h64(norm_obs) if args.unlock_base_arm6 else None
        agate = ref_actor.approach_action_gate(obs)
        ref_req = (base_out + agate * mod.approach(norm_obs) * mod.approach_scale).clamp(-1.0, 1.0)
        return dict(
            obs=obs, norm_obs=norm_obs, base_out=base_out, base_h64=base_h64, agate=agate, ref_req=ref_req,
            req=torch.cat(rl, dim=0).to(device), jmask=torch.cat(jl, dim=0).to(device), weight=torch.cat(wl, dim=0).to(device),
        )

    ow_batch = None
    if args.unlock_base_arm6 and off_win_list:
        ow_obs = torch.cat(off_win_list, dim=0).to(device)
        with torch.no_grad():
            ow_norm = ref_actor.actor.normalizer(ow_obs)
            ow_h64 = extract_base_h64(ow_norm)
            ow_arm = ref_actor.actor.base(ow_norm)[:, 12:18]
        ow_batch = dict(h64=ow_h64, arm=ow_arm)
        print(f'Off-window arm6 anchor: {len(ow_h64)} frames')

    def forward_fast(mod, batch, idx=None, noise_std=0.0):
        nobs = batch['norm_obs'][idx] if idx is not None else batch['norm_obs']
        ag = batch['agate'][idx] if idx is not None else batch['agate']
        inp = nobs + torch.randn_like(nobs) * noise_std if noise_std > 0.0 else nobs
        if args.unlock_base_arm6:
            h64 = batch['base_h64'][idx] if idx is not None else batch['base_h64']
            pmap = dict(actor.named_parameters())
            b_out = F.linear(h64, pmap['actor.base.6.weight'], pmap['actor.base.6.bias'])
        else:
            b_out = batch['base_out'][idx] if idx is not None else batch['base_out']
        return (b_out + ag * mod.approach(inp) * mod.approach_scale).clamp(-1.0, 1.0)

    def masked_mse(pred, target, jmask, weight, arm_only=False):
        if arm_only:
            p, t, m = pred[:, 12:18], target[:, 12:18], jmask[:, 12:18]
        else:
            p, t, m = pred, target, jmask
        sq = (p - t).square() * m
        pf = sq.sum(-1) / m.sum(-1).clamp_min(1.0)
        return (pf * weight).sum() / weight.sum().clamp_min(1.0)

    @torch.no_grad()
    def eval_batch(batch, arm_only=False):
        if batch is None:
            return dict(mse=0.0, rms_arm=0.0, rms_leg=0.0, cos_arm=0.0, mag_arm=0.0)
        pred = forward_fast(actor.actor, batch, None, 0.0)
        mse = float(masked_mse(pred, batch['req'], batch['jmask'], batch['weight'], arm_only=arm_only))
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

    def make_splits(eps, k_folds):
        perm = np.random.permutation(len(eps))
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
        # Register EXACT zero-grad hook on leg rows 0:12 of actor.base.6!
        hooks = []
        if args.unlock_base_arm6:
            def zero_leg_rows(grad):
                g = grad.clone()
                g[:12] = 0.0
                return g
            for nm in ('actor.base.6.weight', 'actor.base.6.bias'):
                param = dict(actor.named_parameters())[nm]
                hooks.append(param.register_hook(zero_leg_rows))

        trainable = [(n, p) for n, p in actor.named_parameters() if p.requires_grad]
        t_trn = build_batch(t_splits[m_idx][0], is_teacher=True, mode='both')
        t_val_nom = build_batch(t_splits[m_idx][1], is_teacher=True, mode='nominal')
        r_trn = build_batch(r_splits[m_idx][0], is_teacher=False)
        r_val = build_batch(r_splits[m_idx][1], is_teacher=False)
        r_val_a = build_batch([e for e in r_splits[m_idx][1] if e['is_panel_a']], is_teacher=False)
        if m_idx == 0:
            eval_batches = (t_val_nom, r_val, r_val_a)
            print(f'Frames: teacher={len(t_trn["obs"])}, ret={len(r_trn["obs"])}, val_nom={len(t_val_nom["obs"])}, val_ret={len(r_val["obs"])}')

        opt = torch.optim.Adam([p for _, p in trainable], lr=args.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.05)

        best_score = float('inf')
        best_st = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
        best_ep = 0
        n_t, n_r = len(t_trn['obs']), len(r_trn['obs'])
        n_ow = len(ow_batch['h64']) if ow_batch is not None else 0
        steps = max(1, math.ceil(max(n_t, n_r) / args.batch_size))
        mod = actor.actor

        for ep in range(1, args.epochs + 1):
            actor.train()
            for _ in range(steps):
                opt.zero_grad(set_to_none=True)
                ti = torch.randint(0, n_t, (min(args.batch_size, n_t),), device=device)
                pt = forward_fast(mod, t_trn, ti, args.norm_obs_noise)
                lt = masked_mse(pt, t_trn['req'][ti], t_trn['jmask'][ti], t_trn['weight'][ti], arm_only=False)

                ri = torch.randint(0, n_r, (min(args.batch_size, n_r),), device=device)
                pr = forward_fast(mod, r_trn, ri, args.norm_obs_noise * 0.5)
                lr_l = masked_mse(pr, r_trn['req'][ri], r_trn['jmask'][ri], r_trn['weight'][ri], arm_only=False)

                loss = lt + args.retention_weight * lr_l
                if ow_batch is not None:
                    oi = torch.randint(0, n_ow, (min(args.batch_size, n_ow),), device=device)
                    pmap = dict(actor.named_parameters())
                    ow_p = F.linear(ow_batch['h64'][oi], pmap['actor.base.6.weight'][12:18], pmap['actor.base.6.bias'][12:18])
                    loss = loss + args.off_window_weight * (ow_p - ow_batch['arm'][oi]).square().mean()
                if args.proximal_weight > 0:
                    prox = sum((p - anchor_params[n]).square().mean() for n, p in trainable) / len(trainable)
                    loss = loss + args.proximal_weight * prox
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 2.0)
                opt.step()
            sched.step()

            if ep % 5 == 0 or ep == args.epochs:
                actor.eval()
                mt = eval_batch(t_val_nom, arm_only=True)
                mr = eval_batch(r_val, arm_only=False)
                sc = mt['mse'] + args.retention_weight * mr['mse']
                if ep >= args.min_epoch and sc < best_score:
                    best_score = sc
                    best_st = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
                    best_ep = ep
                if (ep % 70 == 0 or ep == args.epochs) and m_idx == 0:
                    mra = eval_batch(r_val_a, arm_only=False)
                    print(f'  [m0] ep={ep:3d} val_NOM={mt["rms_arm"]:.4f} cos={mt["cos_arm"]:+.3f} mag={mt["mag_arm"]:.2f} | '
                          f'ret={mr["rms_arm"]:.4f} retA={mra["rms_arm"]:.4f} leg={mr["rms_leg"]:.4f} best@{best_ep}')

        for h in hooks:
            h.remove()
        member_states.append(best_st)
        member_epochs.append(best_ep)

    final_st = {k: v.clone() for k, v in init_full_state.items()}
    for name in trainable_names:
        final_st[name] = (sum(s[name].float() for s in member_states) / len(member_states)).to(init_full_state[name].dtype)
    best_ep = int(round(float(np.mean(member_epochs))))

    # STRICT BIT-EXACT LEG INVARIANCE ASSERTION!
    if args.unlock_base_arm6:
        assert torch.allclose(final_st['actor.base.6.weight'][:12], init_ckpt['actor']['actor.base.6.weight'][:12], atol=1e-6), 'base.6 leg weights changed!'
        assert torch.allclose(final_st['actor.base.6.bias'][:12], init_ckpt['actor']['actor.base.6.bias'][:12], atol=1e-6), 'base.6 leg bias changed!'
        final_st['actor.base.6.weight'][:12].copy_(init_ckpt['actor']['actor.base.6.weight'][:12])
        final_st['actor.base.6.bias'][:12].copy_(init_ckpt['actor']['actor.base.6.bias'][:12])
        assert torch.equal(final_st['actor.base.6.weight'][:12], init_ckpt['actor']['actor.base.6.weight'][:12])
        assert torch.equal(final_st['actor.base.6.bias'][:12], init_ckpt['actor']['actor.base.6.bias'][:12])
    for k, v in init_ckpt['actor'].items():
        if k not in trainable_names and torch.is_tensor(v):
            assert torch.equal(v.cpu(), final_st[k].cpu()), f'Frozen param changed: {k}'

    actor.load_state_dict(final_st, strict=True); actor.eval()
    t_val_nom, r_val, r_val_a = eval_batches
    stats = dict(
        val_nom=eval_batch(t_val_nom, arm_only=True),
        ret_val=eval_batch(r_val, arm_only=False),
        ret_val_A=eval_batch(r_val_a, arm_only=False),
    )
    if ow_batch is not None:
        with torch.no_grad():
            pmap = dict(actor.named_parameters())
            ow_p = F.linear(ow_batch['h64'], pmap['actor.base.6.weight'][12:18], pmap['actor.base.6.bias'][12:18])
            stats['off_window_arm_rms'] = float((ow_p - ow_batch['arm']).square().mean().sqrt())
    print(f'FINAL (soup={args.n_soup}, base_arm6={args.unlock_base_arm6}, best@{best_ep}):', json.dumps(stats, indent=2))

    out_ckpt = {k: v for k, v in ref_ckpt.items()
                if k in ('algorithm', 'adapter_schema', 'actor', 'metadata', 'config',
                         'actor_updates', 'world_updates', 'critic_updates')}
    out_ckpt['actor'] = final_st
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ckpt, args.output)
    args.output.with_suffix('.distill.json').write_text(json.dumps(dict(
        init=str(args.init_checkpoint), ref=str(args.ref_checkpoint),
        teachers=len(teacher_list), retentions=len(ret_list),
        unlock_base_arm6=args.unlock_base_arm6, best_epoch=best_ep, stats=stats,
    ), indent=2) + '\n')
    print(f'Saved -> {args.output}')


if __name__ == '__main__':
    main()
