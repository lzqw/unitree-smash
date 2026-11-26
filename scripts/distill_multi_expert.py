#!/usr/bin/env python3
"""Smooth Geometric Multi-Expert Distillation into a single standard actor.approach MLP.

Fuses:
  - Expert A (e.g. final_best.pt): inside the Panel-A interior box
      (|iy| <= 0.100m, iz in [1.760, 1.840]m, ft in [0.690, 0.810]s) -> alpha = 0.0
  - Expert B (e.g. H8d or unconstrained Panel-B specialist): outside the Panel-A box -> alpha = 1.0
  - Smooth C^1 smoothstep transition over a small geometric buffer shell (dy=18mm, dz=12mm, dft=15ms).

Strictly trains ONLY APPROACH_NAMES (actor.base & actor.recovery verified bit-identical to Expert A).
Zero test-set usage; zero runtime routing code; produces a 100% standard checkpoint.
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
sys.path.insert(0, str(ROOT / 'scripts'))

from distill_native_approach import extract_episode_frames
from quadsmash.rl.supermimic_v1 import load_ppo_initialized_actor
from quadsmash.rl.supermimic_physical_loop_loading_v1 import APPROACH_NAMES


def compute_box_alpha(iy: float, iz: float, ft: float,
                      y_in: float = 0.100, dy: float = 0.018,
                      z_mid: float = 1.800, z_half: float = 0.040, dz: float = 0.012,
                      ft_mid: float = 0.750, ft_half: float = 0.060, dft: float = 0.015) -> float:
    uy = max(0.0, abs(iy) - y_in) / dy
    uz = max(0.0, abs(iz - z_mid) - z_half) / dz
    ut = max(0.0, abs(ft - ft_mid) - ft_half) / dft
    u = min(1.0, max(uy, uz, ut))
    return float(u * u * (3.0 - 2.0 * u))


def build_spec_map(spec_paths):
    mp = {}
    for sp in spec_paths:
        data = json.loads(Path(sp).read_text())
        for c in data.get('cases', []):
            if 'case_id' in c and 'prediction_only' in c:
                mp[c['case_id']] = c
    return mp


def load_episodes(dirs, spec_map, *, keep_all_b: bool = True, pa_only_ok: bool = True,
                  y_in: float = 0.100, dy: float = 0.018,
                  z_half: float = 0.040, dz: float = 0.012,
                  ft_half: float = 0.060, dft: float = 0.015):
    eps = []
    seen = set()
    for d in dirs:
        d = Path(d)
        rows = json.loads((d / 'probe_summary.json').read_text())['rows']
        tape = torch.load(d / 'probe_tape.pt', map_location='cpu', weights_only=True)['tape']
        for i, row in enumerate(rows):
            cid = row['case_id']
            key = (d.name, cid, i)
            if cid in seen:
                # Allow a second trajectory of the same case from a different policy rollout (DAgger coverage)
                if (d.name, cid) in seen:
                    continue
            c = spec_map.get(cid)
            if c is None:
                continue
            is_a = cid.startswith('A-')
            if is_a and pa_only_ok and not (row['success'] and not row['robot_net'] and not row['low_speed']):
                continue
            if (not is_a) and (not keep_all_b) and not (row['success'] and not row['robot_net']):
                continue
            ep = extract_episode_frames(tape, i)
            if ep is None:
                continue
            seen.add((d.name, cid))
            ip = c['prediction_only']['intercept_position_m']
            ft = float(c['prediction_only']['flight_time_s'])
            alpha = 0.0 if is_a else compute_box_alpha(
                float(ip[1]), float(ip[2]), ft,
                y_in=y_in, dy=dy, z_half=z_half, dz=dz, ft_half=ft_half, dft=dft)
            # Extra emphasis on hard outer-shell Panel B (iz>=1.82 or |iy|>=0.14)
            hard_b = (not is_a) and (abs(float(ip[1])) >= 0.14 or float(ip[2]) >= 1.82)
            eps.append(dict(
                case_id=cid, source=d.name, is_a=is_a, alpha=alpha, hard_b=hard_b,
                obs=ep['obs'], gate=ep['gate'], weight=ep['weight'],
            ))
    return eps


def pack_split(eps, exp_a, exp_b, device, pa_mult: float, pb_mult: float, hard_b_mult: float):
    obs = torch.cat([e['obs'] for e in eps], dim=0).to(device)
    gate = torch.cat([e['gate'] for e in eps], dim=0).to(device)
    w_l, alpha_l, is_a_l = [], [], []
    for e in eps:
        n = len(e['obs'])
        m = pa_mult if e['is_a'] else (pb_mult * (hard_b_mult if e['hard_b'] else 1.0))
        w_l.append(e['weight'] * m)
        alpha_l.append(torch.full((n,), e['alpha'], dtype=torch.float32))
        is_a_l.append(torch.full((n,), e['is_a'], dtype=torch.bool))
    w = torch.cat(w_l, dim=0).to(device)
    alpha = torch.cat(alpha_l, dim=0).to(device)
    is_a = torch.cat(is_a_l, dim=0).to(device)
    with torch.no_grad():
        mod_a = exp_a.actor
        mod_b = exp_b.actor
        norm_obs = mod_a.normalizer(obs)
        base_out = mod_a.base(norm_obs)
        agate = exp_a.approach_action_gate(obs)
        req_a = (base_out + agate * mod_a.approach(norm_obs) * mod_a.approach_scale).clamp(-1.0, 1.0)
        req_b = (base_out + agate * mod_b.approach(norm_obs) * mod_b.approach_scale).clamp(-1.0, 1.0)
        tgt = (1.0 - alpha[:, None]) * req_a + alpha[:, None] * req_b
    return dict(norm=norm_obs, base=base_out, agate=agate, gate=gate, w=w, alpha=alpha,
                is_a=is_a, req_a=req_a, req_b=req_b, tgt=tgt)


@torch.no_grad()
def evaluate_pack(actor, pack):
    mod = actor.actor
    pred = (pack['base'] + pack['agate'] * mod.approach(pack['norm']) * mod.approach_scale).clamp(-1.0, 1.0)
    m_arm = pack['gate'][:, 12:18] > 0.05
    m_leg = pack['gate'][:, :12] > 0.05
    out = {}
    pa = pack['is_a']
    if pa.any():
        d_pa = (pred - pack['req_a'])[:, 12:18][pa][m_arm[pa]]
        d_pa_l = (pred - pack['req_a'])[:, :12][pa][m_leg[pa]]
        out['pa_rms'] = float(d_pa.square().mean().sqrt())
        out['pa_q95'] = float(d_pa.abs().quantile(0.95))
        out['pa_max'] = float(d_pa.abs().max())
        out['pa_leg'] = float(d_pa_l.square().mean().sqrt()) if len(d_pa_l) else 0.0
    bo = (~pack['is_a']) & (pack['alpha'] >= 0.85)
    if bo.any():
        d_bo = (pred - pack['req_b'])[:, 12:18][bo][m_arm[bo]]
        u = (pred - pack['req_a'])[:, 12:18][bo][m_arm[bo]]
        v = (pack['req_b'] - pack['req_a'])[:, 12:18][bo][m_arm[bo]]
        out['bo_rms'] = float(d_bo.square().mean().sqrt())
        out['bo_q95'] = float(d_bo.abs().quantile(0.95))
        out['bo_cos'] = float((u * v).sum() / (u.norm() * v.norm() + 1e-8))
        out['bo_mag'] = float(u.norm() / (v.norm() + 1e-8))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expert-a', type=Path, required=True, help='Panel-A interior expert (e.g. final_best.pt)')
    parser.add_argument('--expert-b', type=Path, required=True, help='Panel-B exterior expert (e.g. H8d or B-specialist)')
    parser.add_argument('--data-dirs', type=Path, nargs='+', required=True)
    parser.add_argument('--spec-files', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pa-mult', type=float, default=3.2)
    parser.add_argument('--pb-mult', type=float, default=1.8)
    parser.add_argument('--hard-b-mult', type=float, default=1.35)
    parser.add_argument('--pa-hinge-margin', type=float, default=0.018)
    parser.add_argument('--pa-hinge-weight', type=float, default=18.0)
    parser.add_argument('--box-y-in', type=float, default=0.100)
    parser.add_argument('--box-dy', type=float, default=0.018)
    parser.add_argument('--box-z-half', type=float, default=0.040)
    parser.add_argument('--box-dz', type=float, default=0.012)
    parser.add_argument('--box-ft-half', type=float, default=0.060)
    parser.add_argument('--box-dft', type=float, default=0.015)
    parser.add_argument('--prox-weight', type=float, default=0.015)
    parser.add_argument('--ball-tool-focus', type=float, default=6.0)
    parser.add_argument('--norm-obs-noise', type=float, default=0.018)
    parser.add_argument('--epochs', type=int, default=220)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=4.5e-4)
    parser.add_argument('--n-soup', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1101)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    ck_a = torch.load(args.expert_a, map_location='cpu', weights_only=False)
    ck_b = torch.load(args.expert_b, map_location='cpu', weights_only=False)
    src = Path(ck_a['metadata']['init_policy_checkpoint']['path'])
    arm_prep = ck_a['metadata'].get('policy_execution', {}).get('arm_preparation_seconds', 0.25)

    def make_actor(ck):
        a, _ = load_ppo_initialized_actor(src, device)
        a.load_state_dict(ck['actor'], strict=True)
        if arm_prep is not None:
            a.configure_approach_gate(arm_prep)
        a.eval()
        return a

    exp_a = make_actor(ck_a)
    exp_b = make_actor(ck_b)
    for k, v in ck_a['actor'].items():
        if k not in APPROACH_NAMES and torch.is_tensor(v):
            assert torch.equal(v, ck_b['actor'][k]), f'Frozen mismatch between experts on {k}'

    spec_map = build_spec_map(args.spec_files)
    eps = load_episodes(
        args.data_dirs, spec_map,
        y_in=args.box_y_in, dy=args.box_dy,
        z_half=args.box_z_half, dz=args.box_dz,
        ft_half=args.box_ft_half, dft=args.box_dft)
    n_a = sum(e['is_a'] for e in eps)
    n_b = len(eps) - n_a
    n_bo = sum((not e['is_a']) and e['alpha'] >= 0.85 for e in eps)
    print(f'[MultiExpert] Loaded {len(eps)} episodes (PA={n_a}, PB={n_b}, PB_out={n_bo}) from {len(args.data_dirs)} dirs')

    # Column-wise proximal mask (focus changes on ball/tool/phase columns!)
    w0_col_mult = torch.full((794,), float(args.ball_tool_focus), dtype=torch.float32, device=device)
    for f in range(11):
        b = f * 69
        w0_col_mult[b + 0:b + 9] = 0.75
        w0_col_mult[b + 45:b + 57] = 0.40
        w0_col_mult[b + 61:b + 67] = 0.08
    w0_col_mult[759:776] = 0.40

    # Group by case_id for leakage-free K-fold splits
    by_cid = {}
    for e in eps:
        by_cid.setdefault(e['case_id'], []).append(e)
    cids = sorted(by_cid.keys())
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(cids))
    folds = np.array_split(perm, max(1, args.n_soup))

    init_state = {k: v.detach().cpu().clone() for k, v in ck_a['actor'].items()}
    anchor_cpu = {k: init_state[k].clone() for k in APPROACH_NAMES}
    member_states = []
    oos_metrics = []

    for m_idx in range(args.n_soup):
        torch.manual_seed(args.seed + 1009 * m_idx)
        np.random.seed(args.seed + 1009 * m_idx)
        if args.n_soup > 1:
            val_set = {cids[i] for i in folds[m_idx]}
            trn_eps = [e for c in cids if c not in val_set for e in by_cid[c]]
            val_eps = [e for c in cids if c in val_set for e in by_cid[c]]
        else:
            n_val = max(1, int(0.15 * len(cids)))
            val_set = {cids[i] for i in perm[:n_val]}
            trn_eps = [e for c in cids if c not in val_set for e in by_cid[c]]
            val_eps = [e for c in cids if c in val_set for e in by_cid[c]]

        trn = pack_split(trn_eps, exp_a, exp_b, device, args.pa_mult, args.pb_mult, args.hard_b_mult)
        val = pack_split(val_eps, exp_a, exp_b, device, args.pa_mult, args.pb_mult, args.hard_b_mult)

        student = make_actor(ck_a)
        for name, p in student.named_parameters():
            p.requires_grad_(name in APPROACH_NAMES)
        trainable = [(n, p) for n, p in student.named_parameters() if p.requires_grad]
        anchor_dev = {n: anchor_cpu[n].to(device) for n, _ in trainable}
        opt = torch.optim.Adam([p for _, p in trainable], lr=args.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.04)

        N = len(trn['norm'])
        mod = student.actor
        best_sc = float('inf')
        best_st = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
        best_ep = 0

        for ep in range(1, args.epochs + 1):
            student.train()
            pidx = torch.randperm(N, device=device)
            for s in range(0, N, args.batch_size):
                idx = pidx[s:s + args.batch_size]
                opt.zero_grad(set_to_none=True)
                inp = trn['norm'][idx]
                if args.norm_obs_noise > 0:
                    ns = torch.where(trn['is_a'][idx], args.norm_obs_noise * 0.5, args.norm_obs_noise)[:, None]
                    inp = inp + torch.randn_like(inp) * ns
                pred = (trn['base'][idx] + trn['agate'][idx] * mod.approach(inp) * mod.approach_scale).clamp(-1.0, 1.0)
                gm = (trn['gate'][idx] > 0.02).float()
                jw = torch.ones_like(gm)
                jw[:, :12] = 2.5
                jw[:, 12:18] = 3.5
                mw = gm * jw
                sq = ((pred - trn['tgt'][idx]).square() * mw).sum(-1) / mw.sum(-1).clamp_min(1.0)
                wb = trn['w'][idx]
                loss = (sq * wb).sum() / wb.sum().clamp_min(1.0)

                # Explicit L_inf hinge strictly on Panel-A frames (independent of retention weight!)
                if args.pa_hinge_weight > 0:
                    pa_m = trn['is_a'][idx]
                    if pa_m.any():
                        pm = (trn['gate'][idx][pa_m, 12:18] > 0.02).float()
                        exc = F.relu((pred[pa_m, 12:18] - trn['req_a'][idx][pa_m, 12:18]).abs() - args.pa_hinge_margin)
                        loss = loss + args.pa_hinge_weight * (exc.square() * pm).sum() / pm.sum().clamp_min(1.0)

                if args.prox_weight > 0:
                    pt = []
                    for name, p in trainable:
                        if name == 'actor.approach.0.weight':
                            pt.append(((p - anchor_dev[name]).square() * w0_col_mult[None, :]).mean())
                        else:
                            pt.append(0.25 * (p - anchor_dev[name]).square().mean())
                    loss = loss + args.prox_weight * (sum(pt) / len(pt))

                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 2.0)
                opt.step()
            sched.step()

            if ep % 10 == 0 or ep == args.epochs:
                student.eval()
                mv = evaluate_pack(student, val)
                sc = 2.0 * (mv.get('pa_rms', 0.0) ** 2) + 1.0 * (mv.get('bo_rms', 0.0) ** 2) + 1.5 * max(0.0, mv.get('pa_q95', 0.0) - 0.024) ** 2
                if sc < best_sc:
                    best_sc = sc
                    best_st = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
                    best_ep = ep

        student.load_state_dict({k: v.to(device) for k, v in best_st.items()}, strict=True)
        student.eval()
        mv = evaluate_pack(student, val)
        oos_metrics.append(mv)
        member_states.append(best_st)
        print(f'  [fold {m_idx}] best@{best_ep:3d} OOS | PA rms={mv["pa_rms"]:.4f} q95={mv["pa_q95"]:.4f} max={mv["pa_max"]:.3f} '
              f'| B_out rms={mv["bo_rms"]:.4f} q95={mv["bo_q95"]:.4f} cos={mv["bo_cos"]:+.4f} mag={mv["bo_mag"]:.3f}', flush=True)

    final_state = {k: v.clone() for k, v in init_state.items()}
    for name in APPROACH_NAMES:
        final_state[name] = (sum(s[name].float() for s in member_states) / len(member_states)).to(init_state[name].dtype)
    for k, v in ck_a['actor'].items():
        if k not in APPROACH_NAMES and torch.is_tensor(v):
            assert torch.equal(v.cpu(), final_state[k].cpu()), f'Frozen param mutated: {k}'

    mean_oos = {k: float(np.mean([m[k] for m in oos_metrics if k in m])) for k in oos_metrics[0]}
    print('[MultiExpert] Mean K-fold OOS:', json.dumps(mean_oos, indent=2), flush=True)

    out_ckpt = {k: v for k, v in ck_a.items()
                if k in ('algorithm', 'adapter_schema', 'actor', 'metadata', 'config',
                         'actor_updates', 'world_updates', 'critic_updates')}
    out_ckpt['actor'] = final_state
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ckpt, args.output)
    args.output.with_suffix('.fusion.json').write_text(json.dumps(dict(
        expert_a=str(args.expert_a), expert_b=str(args.expert_b),
        n_episodes=len(eps), mean_oos=mean_oos,
    ), indent=2) + '\n')
    print(f'Saved -> {args.output}', flush=True)


if __name__ == '__main__':
    main()
