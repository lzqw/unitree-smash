#!/usr/bin/env python3
"""Zero-Anchored Ensemble-Dynamics Closed-Loop Differentiable Physics BPTT for Approach.

Stage 1 (`fit-surrogate`):
  Trains a 3-member ensemble of ZeroAnchoredFullBodyDelta models on ALL real PhysX
  (nominal, probe) trajectory pairs. At delta_action = 0, model error is identically 0.0mm.
  Supervises both 49D full-body state deltas (18 joints + 18 vels + root pos/quat/vel/angvel)
  and exact 3D FK racket pose/normal/velocity via FullBodyGeometryV1.

Stage 2 (`train-bptt`):
  Rolls out `actor` (frozen base/recovery, trainable APPROACH_NAMES only) in true closed loop
  over real PhysX nominal base trajectories using exact zero-anchored observation updates,
  differentiable swept ray-plane contact, calibrated impact (e=0.85, f=0.30), and vectorized
  differentiable RK4 ShuttlecockModel flight. Directly optimizes sweet-spot margin, court-centered
  landing (lx, ly), outgoing speed, net clearance, robot-net barrier, and trust-region bounds.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quadsmash.physics.full_racket_v4 import load_racket, string_polygon
from quadsmash.physics.native import quat_rotate
from quadsmash.rl.supermimic_physical_dataset_v1 import decode_physical
from quadsmash.rl.supermimic_physical_loop_loading_v1 import APPROACH_NAMES
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2
from quadsmash.rl.supermimic_v1 import load_ppo_initialized_actor
from quadsmash.tasks.stand.observation_contract import inverse_rotate, relative_velocity

ASSET_SHA = 'c2264fcece90b042c7c6a1cac7d516af83972729d0402947e3a519041e184cf2'
DRAG_SCALE = 0.5 * 1.225 * 0.12 * 0.0018 / 0.0052
SL_START = 35   # tick 235 (relative to start_capture_tick=200)
SL_END = 96     # tick 296 (61 ticks = 600ms)
SL_LEN = SL_END - SL_START


def shuttle_accel(v: torch.Tensor) -> torch.Tensor:
    spd = v.norm(dim=-1, keepdim=True)
    g = v.new_tensor([0.0, 0.0, -9.81])
    return g - DRAG_SCALE * spd * v


def shuttle_step_rk4(p: torch.Tensor, v: torch.Tensor, dt: float = 0.01):
    k1 = shuttle_accel(v)
    k2 = shuttle_accel(v + 0.5 * dt * k1)
    k3 = shuttle_accel(v + 0.5 * dt * k2)
    k4 = shuttle_accel(v + dt * k3)
    v_next = v + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    p_next = p + (dt / 6.0) * (v + 2.0 * (v + 0.5 * dt * k1) + 2.0 * (v + 0.5 * dt * k2) + (v + dt * k3))
    return p_next, v_next


def compute_free_ball_traj(tape_state: torch.Tensor, start_idx: int = 25, end_idx: int = SL_END):
    """Integrate exact uninterrupted incoming shuttle trajectory from pre-contact tick 225."""
    d0 = decode_physical(tape_state[start_idx].float())
    p = d0['ball_position_w'].clone()
    v = d0['ball_velocity_w'].clone()
    ps, vs = [p], [v]
    for _ in range(start_idx, end_idx - 1):
        p, v = shuttle_step_rk4(p, v, 0.005)
        p, v = shuttle_step_rk4(p, v, 0.005)
        ps.append(p)
        vs.append(v)
    ps = torch.stack(ps, dim=0)  # [end_idx - start_idx, N, 3]
    vs = torch.stack(vs, dim=0)
    off = SL_START - start_idx
    return ps[off:off + SL_LEN], vs[off:off + SL_LEN]


class ZeroAnchoredFullBodyDelta(nn.Module):
    def __init__(self, hidden: int = 384):
        super().__init__()
        # Input: ctx(71) + x(49) + da(18) + da_prev(18) = 156
        self.in_proj = nn.Linear(71 + 49 + 36, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 49)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _f(self, inp: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.ln1(self.in_proj(inp)))
        h = h + F.silu(self.ln2(self.fc1(h)))
        h = h + F.silu(self.fc2(h))
        return self.out(h)

    def step(self, x: torch.Tensor, u: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([ctx, x, u], dim=-1)
        z0 = torch.cat([ctx, torch.zeros_like(x), torch.zeros_like(u)], dim=-1)
        dx = self._f(inp) - self._f(z0)
        dqd = x[:, 18:36] + dx[:, 18:36]
        dq = x[:, 0:18] + 0.01 * dqd + 0.05 * dx[:, 0:18]
        drv = x[:, 43:46] + dx[:, 43:46]
        drp = x[:, 36:39] + 0.01 * drv + 0.02 * dx[:, 36:39]
        dqr = x[:, 39:43] + 0.05 * dx[:, 39:43]
        drw = x[:, 46:49] + dx[:, 46:49]
        return torch.cat([dq, dqd, drp, dqr, drv, drw], dim=-1)


def load_surrogate_pairs(base_dir: Path):
    groups = [
        (base_dir / 'P21_goal_opt/r1_nominal_256', [base_dir / 'P21_goal_opt/r1_probe_256', base_dir / 'P21_goal_opt/r1_probe_256_b'], 1),
        (base_dir / 'P21_goal_opt/r2_nominal_256', [base_dir / 'P21_goal_opt/r2_probe_256', base_dir / 'P21_goal_opt/r2_probe_256_b'], 1),
        (base_dir / 'P21_goal_opt/r3_nominal_256', [base_dir / 'P21_goal_opt/r3_probe_256', base_dir / 'P21_goal_opt/r3_probe_256_b'], 1),
        (base_dir / 'P21_goal_opt/r4_nominal_256', [base_dir / 'P21_goal_opt/r4_probe_256', base_dir / 'P21_goal_opt/r4_probe_256_b'], 1),
        (base_dir / 'P21_goal_opt/r5_nominal_256', [
            base_dir / 'P21_goal_opt/r5_probe_256', base_dir / 'P21_goal_opt/r5_probe_256_b',
            base_dir / 'P23_push80/r56_pass2_256', base_dir / 'P23_push80/r56_topfr_dyn_256',
            base_dir / 'P23_push80/r56_topfr_pass2',
        ], 1),
        # Closed-loop BPTT on-policy real PhysX pairs (exclude r56_early_base leg-jump probes!)
        (base_dir / 'P21_goal_opt/r5_nominal_256', [
            base_dir / 'P27_push85/r5_bptt_b5_256',
        ], 5),
        (base_dir / 'P21_goal_opt/r6_nominal_256', [
            base_dir / 'P21_goal_opt/r6_probe_256',
            base_dir / 'P23_push80/r56_pass2_256', base_dir / 'P23_push80/r56_topfr_dyn_256',
            base_dir / 'P23_push80/r56_topfr_pass2',
        ], 1),
        (base_dir / 'P26_balltool/r8_panelB_nominal', [
            base_dir / 'P26_balltool/r8_panelB_probe_a',
            base_dir / 'P26_balltool/r8_panelB_probe_b',
            base_dir / 'P27_push85/r8_panelB_pass2_256',
        ], 1),
    ]
    all_ctx, all_da, all_y, all_mask, all_ct, all_s0, all_tag = [], [], [], [], [], [], []
    for nom_dir, pdirs, repeat in groups:
        if not (nom_dir / 'probe_tape.pt').exists():
            continue
        nom_t = torch.load(nom_dir / 'probe_tape.pt', map_location='cpu', weights_only=True)
        nom_ids = {cid: i for i, cid in enumerate(nom_t['case_ids'])}
        nt = nom_t['tape']
        for pdir in pdirs:
            if not (pdir / 'probe_tape.pt').exists():
                continue
            prb_t = torch.load(pdir / 'probe_tape.pt', map_location='cpu', weights_only=True)
            pt = prb_t['tape']
            T = min(nt['state'].shape[0], pt['state'].shape[0], SL_END)
            if T < SL_END:
                continue
            is_bptt_dir = 'bptt' in pdir.name
            for j, cid in enumerate(prb_t['case_ids']):
                if cid not in nom_ids:
                    continue
                i = nom_ids[cid]
                hc0 = nt['had_contact'][:SL_END, i]
                hc1 = pt['had_contact'][:SL_END, j]
                c0 = int(hc0.nonzero()[0]) if hc0.any() else SL_END
                c1 = int(hc1.nonzero()[0]) if hc1.any() else SL_END
                if min(c0, c1) < 55:
                    continue
                d0 = decode_physical(nt['state'][SL_START:SL_END, i].float())
                d1 = decode_physical(pt['state'][SL_START:SL_END, j].float())
                q0r, q1r = d0['root_quaternion_w'], d1['root_quaternion_w']
                sign = ((q0r * q1r).sum(-1, keepdim=True) >= 0).float() * 2.0 - 1.0
                dqr = (q1r * sign) - q0r
                y = torch.cat([
                    d1['joint_position'] - d0['joint_position'],
                    d1['joint_velocity'] - d0['joint_velocity'],
                    d1['root_position_w'] - d0['root_position_w'],
                    dqr,
                    d1['root_link_velocity_w'] - d0['root_link_velocity_w'],
                    d1['root_angular_velocity_w'] - d0['root_angular_velocity_w'],
                ], dim=-1)
                s0 = torch.cat([
                    d0['joint_position'], d0['joint_velocity'],
                    d0['root_position_w'], d0['root_quaternion_w'],
                    d0['root_link_velocity_w'], d0['root_angular_velocity_w'],
                ], dim=-1)
                feet0 = nt['observation'][SL_START:SL_END, i, 57:61].float()
                a0 = nt['request'][SL_START:SL_END, i].float()
                a1 = pt['request'][SL_START:SL_END, j].float()
                ctx = torch.cat([
                    d0['joint_position'], d0['joint_velocity'] * 0.1,
                    d0['root_position_w'], d0['root_quaternion_w'],
                    d0['root_link_velocity_w'] * 0.2, d0['root_angular_velocity_w'] * 0.2,
                    feet0, a0,
                ], dim=-1)
                mask = torch.zeros(SL_LEN)
                n_valid = max(1, min(SL_LEN, min(c0, c1) - SL_START))
                mask[:n_valid] = 1.0
                # Hold out lanes 208..255 of bptt dir cleanly (no repeat on held-out lanes!)
                if is_bptt_dir and j >= 208:
                    tag = 2  # held-out closed-loop BPTT val!
                    n_rep = 1
                else:
                    tag = 1 if is_bptt_dir else 0
                    n_rep = repeat
                for _ in range(n_rep):
                    all_ctx.append(ctx)
                    all_da.append(a1 - a0)
                    all_y.append(y)
                    all_mask.append(mask)
                    all_ct.append(max(0, min(SL_LEN - 1, min(c0, c1) - 1 - SL_START)))
                    all_s0.append(s0)
                    all_tag.append(tag)
    return (torch.stack(all_ctx), torch.stack(all_da), torch.stack(all_y),
            torch.stack(all_mask), torch.tensor(all_ct, dtype=torch.long), torch.stack(all_s0),
            np.array(all_tag, dtype=np.int64))


def fk_from_flat49(geom, s49: torch.Tensor):
    q = s49[:, 0:18]
    qd = s49[:, 18:36]
    rp = s49[:, 36:39]
    rq = s49[:, 39:43]
    rq = rq / rq.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    rv = s49[:, 43:46]
    rw = s49[:, 46:49]
    return geom.forward(q, qd, rp, rq, rv, rw)


def cmd_fit_surrogate(args):
    device = torch.device(args.device)
    t0 = time.perf_counter()
    ctx, da, y, mask, ct, s0, tags = load_surrogate_pairs(args.base_dir)
    N = len(ctx)
    print(f'Loaded {N} pure-arm full-body PhysX delta pairs (bptt_trn={(tags==1).sum()}, bptt_val={(tags==2).sum()}) in {time.perf_counter()-t0:.1f}s')

    rng = np.random.RandomState(args.seed)
    probe_idx = np.where(tags == 0)[0]
    bptt_trn = np.where(tags == 1)[0]
    bptt_val = np.where(tags == 2)[0]
    rng.shuffle(probe_idx)
    val_idx = np.concatenate([probe_idx[:208], bptt_val])
    trn_idx = np.concatenate([probe_idx[208:], bptt_trn])
    n_val = len(val_idx)
    n_bptt_val = len(bptt_val)

    geom = PhysicalWorldV2(ASSET_SHA).geometry.to(device)
    rid = geom.ids['Racket']

    ctx_tr, da_tr, y_tr, m_tr, ct_tr, s0_tr = [z[trn_idx].to(device) for z in (ctx, da, y, mask, ct, s0)]
    ctx_va, da_va, y_va, m_va, ct_va, s0_va = [z[val_idx].to(device) for z in (ctx, da, y, mask, ct, s0)]

    w_dim = torch.tensor(
        [6.0] * 12 + [15.0] * 6 + [0.3] * 12 + [0.55] * 6 + [38.0] * 3 + [28.0] * 4 + [2.2] * 3 + [1.1] * 3,
        device=device)

    member_states = []
    val_preds_ens = []
    for m_idx in range(args.n_ensemble):
        torch.manual_seed(args.seed + 101 * m_idx)
        model = ZeroAnchoredFullBodyDelta(hidden=args.hidden).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.04)
        N_tr = len(trn_idx)

        for step in range(1, args.steps + 1):
            idx = torch.randint(0, N_tr, (args.batch_size,), device=device)
            cb, dab, yb, mb, ctb, s0b = ctx_tr[idx], da_tr[idx], y_tr[idx], m_tr[idx], ct_tr[idx], s0_tr[idx]
            x = torch.zeros(args.batch_size, 49, device=device)
            dap = torch.zeros(args.batch_size, 18, device=device)
            loss_state = 0.0
            x_at_ct = torch.zeros(args.batch_size, 49, device=device)
            b_arange = torch.arange(args.batch_size, device=device)
            for t in range(SL_LEN - 1):
                u = torch.cat([dab[:, t], dap], dim=-1)
                x = model.step(x, u, cb[:, t])
                dap = dab[:, t]
                err = ((x - yb[:, t + 1]) * w_dim).square().mean(-1)
                loss_state = loss_state + (err * mb[:, t + 1]).sum() / mb[:, t + 1].sum().clamp_min(1.0)
                hit_t = (ctb == (t + 1))
                if hit_t.any():
                    x_at_ct = torch.where(hit_t[:, None], x, x_at_ct)
            loss_state = loss_state / (SL_LEN - 1)

            # Direct 3D FK Racket Pos + Normal + Velocity loss at contact tick!
            s0_c = s0b[b_arange, ctb]
            y_c = yb[b_arange, ctb]
            fk_pred = fk_from_flat49(geom, s0_c + x_at_ct)
            with torch.no_grad():
                fk_true = fk_from_flat49(geom, s0_c + y_c)
            pos_l = (fk_pred['position'][:, rid] - fk_true['position'][:, rid]).square().sum(-1).mean()
            nrm_l = (fk_pred['rotation'][:, rid, :, 0] - fk_true['rotation'][:, rid, :, 0]).square().sum(-1).mean()
            vel_l = (fk_pred['velocity'][:, rid] - fk_true['velocity'][:, rid]).square().sum(-1).mean()
            loss = loss_state + 280.0 * pos_l + 30.0 * nrm_l + 2.5 * vel_l

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()

            if step % 150 == 0 or step == args.steps:
                print(f'  [ens {m_idx}] step={step:4d}/{args.steps} loss={float(loss):.4f} '
                      f'fk_pos_rms={math.sqrt(float(pos_l))*1000:.1f}mm fk_vel_rms={math.sqrt(float(vel_l)):.2f}m/s', flush=True)

        model.eval()
        with torch.no_grad():
            x = torch.zeros(n_val, 49, device=device)
            dap = torch.zeros(n_val, 18, device=device)
            preds = [x]
            for t in range(SL_LEN - 1):
                u = torch.cat([da_va[:, t], dap], dim=-1)
                x = model.step(x, u, ctx_va[:, t])
                dap = da_va[:, t]
                preds.append(x)
            val_preds_ens.append(torch.stack(preds, dim=1))
        member_states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

    # Evaluate 3-member ensemble on held-out val!
    with torch.no_grad():
        pred_mean = sum(val_preds_ens) / len(val_preds_ens)
        b_ar = torch.arange(n_val, device=device)
        s0_c = s0_va[b_ar, ct_va]
        y_c = y_va[b_ar, ct_va]
        p_c = pred_mean[b_ar, ct_va]
        fk0 = fk_from_flat49(geom, s0_c)
        fkt = fk_from_flat49(geom, s0_c + y_c)
        fkp = fk_from_flat49(geom, s0_c + p_c)
        pos_e = (fkp['position'][:, rid] - fkt['position'][:, rid]).norm(dim=-1).cpu().numpy()
        shift = (fk0['position'][:, rid] - fkt['position'][:, rid]).norm(dim=-1).cpu().numpy()
        ang_e = torch.acos((fkt['rotation'][:, rid, :, 0] * fkp['rotation'][:, rid, :, 0]).sum(-1).abs().clamp(max=1.0)).cpu().numpy() * 180.0 / math.pi
        vel_e = (fkp['velocity'][:, rid] - fkt['velocity'][:, rid]).norm(dim=-1).cpu().numpy()
        q_e = (p_c[:, 12:18] - y_c[:, 12:18]).abs().mean(-1).cpu().numpy()
        r2 = 1.0 - float((pos_e ** 2).sum() / ((shift ** 2).sum() + 1e-9))
        print(f'\n=== HELD-OUT {n_val}-traj {args.n_ensemble}-Member Ensemble Validation ===')
        print(f'  ALL   |dq_arm|={np.median(q_e):.4f} rad | 3D Pos={np.median(pos_e)*1000:.2f} mm (R2={r2:.4f}) | Norm={np.median(ang_e):.2f} deg | Vel={np.median(vel_e):.3f} m/s')
        if n_bptt_val > 0:
            bp_sl = slice(n_val - n_bptt_val, n_val)
            bp_r2 = 1.0 - float((pos_e[bp_sl] ** 2).sum() / ((shift[bp_sl] ** 2).sum() + 1e-9))
            bp_bias = (fkp['position'][bp_sl, rid] - fkt['position'][bp_sl, rid]).mean(0).cpu().numpy() * 1000.0
            print(f'  BPTT  (N={n_bptt_val}) 3D Pos={np.median(pos_e[bp_sl])*1000:.2f} mm (shift={np.median(shift[bp_sl])*1000:.1f} mm, R2={bp_r2:.4f}, bias={np.round(bp_bias,1)} mm) | Norm={np.median(ang_e[bp_sl]):.2f} deg | Vel={np.median(vel_e[bp_sl]):.3f} m/s')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(hidden=args.hidden, n_ensemble=args.n_ensemble, members=member_states,
                    val_racket_pos_mm=float(np.median(pos_e) * 1000.0), val_r2=r2), args.output)
    print(f'Saved ensemble surrogate -> {args.output}')


def build_frame69_from_s49(geom, rid, s49, bp, bv, feet, present):
    q = s49[:, 0:18]
    qd = s49[:, 18:36]
    rp = s49[:, 36:39]
    rq = s49[:, 39:43]
    rq = rq / rq.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    rv = s49[:, 43:46]
    rw = s49[:, 46:49]
    fk = geom.forward(q, qd, rp, rq, rv, rw)
    tp = fk['position'][:, rid]
    tn = fk['rotation'][:, rid, :, 0]
    tv = fk['velocity'][:, rid]
    tw = fk['angular_velocity'][:, rid]
    grav_w = rp.new_tensor([0.0, 0.0, -1.0]).expand_as(rp)
    frame = torch.cat([
        inverse_rotate(rq, rv),
        inverse_rotate(rq, rw),
        inverse_rotate(rq, grav_w),
        q,
        qd,
        inverse_rotate(rq, tp - rp),
        inverse_rotate(rq, tn),
        relative_velocity(rq, tp, tv, rp, rv, rw),
        inverse_rotate(rq, tw - rw),
        feet,
        inverse_rotate(rq, bp - rp) * present[:, None],
        relative_velocity(rq, bp, bv, rp, rv, rw) * present[:, None],
        present[:, None],
        torch.ones_like(present[:, None]),
    ], dim=-1)
    return frame, tp, fk['rotation'][:, rid], tv, tw


def load_bptt_episodes(nominal_dirs, geom, rid, device):
    """Load real PhysX nominal trajectories + precompute exact uninterrupted free-ball & base frames."""
    items = dict(ctx=[], a_nom=[], obs_nom=[], s0=[], bp=[], bv=[], feet=[], pres=[],
                 frame0_base=[], is_panel_a=[], nom_ok=[], c0_rel=[])
    for ndir in nominal_dirs:
        ndir = Path(ndir)
        blob = torch.load(ndir / 'probe_tape.pt', map_location='cpu', weights_only=True)
        tape = blob['tape']
        summary = json.loads((ndir / 'probe_summary.json').read_text())
        bp_free, bv_free = compute_free_ball_traj(tape['state'], start_idx=25, end_idx=SL_END)
        T_s = tape['state'][SL_START:SL_END].float()  # [61, N, 55]
        N = T_s.shape[1]
        for i in range(N):
            d0 = decode_physical(T_s[:, i])
            s0_i = torch.cat([
                d0['joint_position'], d0['joint_velocity'],
                d0['root_position_w'], d0['root_quaternion_w'],
                d0['root_link_velocity_w'], d0['root_angular_velocity_w'],
            ], dim=-1)
            obs_i = tape['observation'][SL_START:SL_END, i].float()
            a_i = tape['request'][SL_START:SL_END, i].float()
            feet_i = obs_i[:, 57:61]
            pres_i = obs_i[:, 67]
            ctx_i = torch.cat([
                d0['joint_position'], d0['joint_velocity'] * 0.1,
                d0['root_position_w'], d0['root_quaternion_w'],
                d0['root_link_velocity_w'] * 0.2, d0['root_angular_velocity_w'] * 0.2,
                feet_i, a_i,
            ], dim=-1)
            hc = tape['had_contact'][SL_START:SL_END, i]
            c0 = int(hc.nonzero()[0]) if hc.any() else SL_LEN - 1
            row = summary['rows'][i]
            items['ctx'].append(ctx_i)
            items['a_nom'].append(a_i)
            items['obs_nom'].append(obs_i)
            items['s0'].append(s0_i)
            items['bp'].append(bp_free[:, i])
            items['bv'].append(bv_free[:, i])
            items['feet'].append(feet_i)
            items['pres'].append(pres_i)
            items['is_panel_a'].append(1.0 if row['case_id'].startswith('A-') else 0.0)
            items['nom_ok'].append(1.0 if row['success'] else 0.0)
            items['c0_rel'].append(c0)

    out = {k: torch.stack(v, dim=0).to(device) if torch.is_tensor(v[0]) else torch.tensor(v, device=device)
           for k, v in items.items() if k != 'frame0_base'}
    # Precompute frame0_base in batch on GPU!
    N_tot = out['s0'].shape[0]
    with torch.no_grad():
        f_list = []
        for t in range(SL_LEN):
            f_t, _, _, _, _ = build_frame69_from_s49(
                geom, rid, out['s0'][:, t], out['bp'][:, t], out['bv'][:, t],
                out['feet'][:, t], out['pres'][:, t])
            f_list.append(f_t)
        out['frame0_base'] = torch.stack(f_list, dim=1)  # [N, 61, 69]
    return out


def simulate_shuttle_flight_diff(p0: torch.Tensor, v0: torch.Tensor, n_steps: int = 90, dt: float = 0.01):
    """Differentiable vectorized RK4 shuttle flight returning (lx, ly, net_margin_m)."""
    B = p0.shape[0]
    p, v = p0, v0
    lx = p0[:, 0].clone()
    ly = p0[:, 1].clone()
    landed = torch.zeros(B, dtype=torch.bool, device=p0.device)
    z_net = p0.new_full((B,), -1.0)
    y_net = p0.new_zeros((B,))
    crossed = torch.zeros(B, dtype=torch.bool, device=p0.device)

    for _ in range(n_steps):
        p_prev = p
        p, v = shuttle_step_rk4(p, v, dt)
        # Net crossing at x = 1.500m
        cross_now = (~crossed) & (p_prev[:, 0] < 1.500) & (p[:, 0] >= 1.500)
        if cross_now.any():
            frac = ((1.500 - p_prev[:, 0]) / (p[:, 0] - p_prev[:, 0]).clamp_min(1e-5)).clamp(0.0, 1.0)
            zn = p_prev[:, 2] + frac * (p[:, 2] - p_prev[:, 2])
            yn = p_prev[:, 1] + frac * (p[:, 1] - p_prev[:, 1])
            z_net = torch.where(cross_now, zn, z_net)
            y_net = torch.where(cross_now, yn, y_net)
            crossed = crossed | cross_now
        # Ground landing at z = 0.0m
        land_now = (~landed) & (p_prev[:, 2] > 0.0) & (p[:, 2] <= 0.0)
        if land_now.any():
            frac = (p_prev[:, 2] / (p_prev[:, 2] - p[:, 2]).clamp_min(1e-5)).clamp(0.0, 1.0)
            pl = p_prev + frac[:, None] * (p - p_prev)
            lx = torch.where(land_now, pl[:, 0], lx)
            ly = torch.where(land_now, pl[:, 1], ly)
            landed = landed | land_now
        if landed.all():
            break

    # Fallback if still airborne at end of 0.9s: linear project to z=0
    still = ~landed
    if still.any():
        t_rem = (p[:, 2].clamp_min(0.0) / (-v[:, 2]).clamp_min(0.5))
        lx = torch.where(still, p[:, 0] + v[:, 0] * t_rem, lx)
        ly = torch.where(still, p[:, 1] + v[:, 1] * t_rem, ly)

    # Exact court net height (1.524m center + gentle sag) minus 0.024m shuttle radius
    net_top = 1.524 + 0.004 * y_net.square()
    # If never crossed net (e.g. hit backwards or too low), approximate z at x=1.5 linearly
    no_cross = ~crossed
    if no_cross.any():
        z_est = p0[:, 2] + ((1.500 - p0[:, 0]) / v0[:, 0].clamp_min(0.5)) * v0[:, 2] - 0.5 * 9.81 * ((1.500 - p0[:, 0]) / v0[:, 0].clamp_min(0.5)).square()
        z_net = torch.where(no_cross, z_est, z_net)
    nm = z_net - net_top - 0.024
    return lx, ly, nm


def rollout_bptt_batch(actor, ens_models, geom, rid, poly_inward, poly_boundary, data, idx,
                       *, use_all_ens: bool = True, obs_noise: float = 0.0):
    B = len(idx)
    ctx = data['ctx'][idx]
    a_nom = data['a_nom'][idx]
    obs_nom = data['obs_nom'][idx]
    s0 = data['s0'][idx]
    bp = data['bp'][idx]
    bv = data['bv'][idx]
    feet = data['feet'][idx]
    pres = data['pres'][idx]
    f0_base = data['frame0_base'][idx]
    device = s0.device

    mod = actor.actor
    x = torch.zeros(B, 49, device=device)
    da_prev = torch.zeros(B, 18, device=device)
    hist_d = torch.zeros(B, 11, 69, device=device)

    tp_l, Rr_l, tv_l, tw_l = [], [], [], []
    da_arm_l, da_leg_l = [], []
    rx_max = s0[:, 0, 36].clone()
    tx_max = s0[:, 0, 36].clone()
    ens_var_acc = 0.0

    for t in range(SL_LEN):
        s_cur = s0[:, t] + x
        f_cur, tp, Rr, tv, tw = build_frame69_from_s49(geom, rid, s_cur, bp[:, t], bv[:, t], feet[:, t], pres[:, t])
        f_delta = f_cur - f0_base[:, t]
        hist_d = torch.cat([f_delta[:, None, :], hist_d[:, :-1, :]], dim=1)

        obs_t = obs_nom[:, t].clone()
        obs_t[:, :759] = obs_t[:, :759] + hist_d.reshape(B, 759)
        obs_t[:, 776:794] = obs_t[:, 776:794] + da_prev

        norm_o = mod.normalizer(obs_t)
        norm_app = norm_o + torch.randn_like(norm_o) * obs_noise if obs_noise > 0.0 else norm_o
        agate = actor.approach_action_gate(obs_t)
        a_sim = (mod.base(norm_o) + agate * mod.approach(norm_app) * mod.approach_scale).clamp(-1.0, 1.0)
        da = a_sim - a_nom[:, t]

        tp_l.append(tp); Rr_l.append(Rr); tv_l.append(tv); tw_l.append(tw)
        da_arm_l.append(da[:, 12:18])
        da_leg_l.append(da[:, 0:12])
        rx_max = torch.maximum(rx_max, s_cur[:, 36])
        tx_max = torch.maximum(tx_max, tp[:, 0])

        if t < SL_LEN - 1:
            u = torch.cat([da, da_prev], dim=-1)
            if use_all_ens:
                xs = [m.step(x, u, ctx[:, t]) for m in ens_models]
                x_next = sum(xs) / len(xs)
                if len(xs) > 1:
                    ens_var_acc = ens_var_acc + sum((xi[:, 12:18] - x_next[:, 12:18]).square().mean() for xi in xs) / len(xs)
                x = x_next
            else:
                x = ens_models[0].step(x, u, ctx[:, t])
            da_prev = da

    tp_all = torch.stack(tp_l, dim=1)   # [B, 61, 3]
    Rr_all = torch.stack(Rr_l, dim=1)   # [B, 61, 3, 3]
    tv_all = torch.stack(tv_l, dim=1)   # [B, 61, 3]
    tw_all = torch.stack(tw_l, dim=1)   # [B, 61, 3]

    # Differentiable swept ray-plane intersection over ticks [15..60] (t = 2.50..2.95s)
    t_lo = 15
    bp_w = bp[:, t_lo:]      # [B, W, 3]
    bv_w = bv[:, t_lo:]
    tp_w = tp_all[:, t_lo:]
    Rr_w = Rr_all[:, t_lo:]
    tv_w = tv_all[:, t_lo:]
    tw_w = tw_all[:, t_lo:]

    n_raw = Rr_w[:, :, :, 0]  # [B, W, 3]
    # Flip normal to face incoming ball (dot(n_in, bv) < 0)
    flip = ((n_raw * bv_w).sum(-1, keepdim=True) > 0.0).float() * (-2.0) + 1.0
    n_in = n_raw * flip
    # Contact occurs at shuttle radius R = 0.024m above string plane!
    d_signed = ((bp_w - tp_w) * n_in).sum(-1) - 0.024  # [B, W]

    # Find first crossing where d[k] >= 0 and d[k+1] < 0
    cross = (d_signed[:, :-1] >= 0.0) & (d_signed[:, 1:] < 0.0)
    has_cross = cross.any(dim=1)
    first_idx = cross.float().argmax(dim=1)
    closest_idx = d_signed[:, :-1].abs().argmin(dim=1)
    k_star = torch.where(has_cross, first_idx, closest_idx)  # [B]

    b_ar = torch.arange(B, device=device)
    d0 = d_signed[b_ar, k_star]
    d1 = d_signed[b_ar, k_star + 1]
    denom = (d0 - d1).clamp_min(1e-4)
    alpha = (d0 / denom).clamp(0.0, 1.0)  # [B], differentiable!
    a1 = alpha[:, None]
    a0 = 1.0 - a1

    bp_hit = a0 * bp_w[b_ar, k_star] + a1 * bp_w[b_ar, k_star + 1]
    bv_hit = a0 * bv_w[b_ar, k_star] + a1 * bv_w[b_ar, k_star + 1]
    tp_hit = a0 * tp_w[b_ar, k_star] + a1 * tp_w[b_ar, k_star + 1]
    # Use clean pre-contact velocity at k_star (never contaminated by post-impact recoil at k_star+1!)
    tv_hit = tv_w[b_ar, k_star]
    tw_hit = tw_w[b_ar, k_star]

    # Interpolate racket basis smoothly and orthonormalize
    R0 = Rr_w[b_ar, k_star]
    R1 = Rr_w[b_ar, k_star + 1]
    R_blend = a0[:, :, None] * R0 + a1[:, :, None] * R1
    ex = F.normalize(R_blend[:, :, 0], dim=-1)
    ey = F.normalize(R_blend[:, :, 1] - (R_blend[:, :, 1] * ex).sum(-1, keepdim=True) * ex, dim=-1)
    ez = torch.linalg.cross(ex, ey)

    rel_p = bp_hit - tp_hit
    loc_y = (rel_p * ey).sum(-1)
    loc_z = (rel_p * ez).sum(-1)
    loc_yz = torch.stack([loc_y, loc_z], dim=-1)  # [B, 2]

    # 32-gon signed string margin (exact min for metrics, smooth softmin for gradient)
    edge_margins = loc_yz @ poly_inward.T - poly_boundary[None, :]  # [B, 32]
    sm_exact = edge_margins.min(dim=-1).values
    sm_smooth = -0.008 * torch.logsumexp(-edge_margins / 0.008, dim=-1)

    # Calibrated pre-impulse collision including articulated racket effective-mass recoil (e=0.58, f=0.30, vpt_gain=0.85)
    n_hit = torch.where(((ex * bv_hit).sum(-1, keepdim=True) > 0.0), -ex, ex)
    v_pt = 0.85 * (tv_hit + torch.linalg.cross(tw_hit, rel_p))
    v_rel = bv_hit - v_pt
    vn = (v_rel * n_hit).sum(-1, keepdim=True).clamp_max(0.0)
    vt = v_rel - vn * n_hit
    v_out = v_pt - 0.58 * vn * n_hit + 0.70 * vt
    spd = v_out.norm(dim=-1)

    lx, ly, nm = simulate_shuttle_flight_diff(bp_hit, v_out, n_steps=90, dt=0.01)

    da_arm = torch.stack(da_arm_l, dim=1)  # [B, 61, 6]
    da_leg = torch.stack(da_leg_l, dim=1)  # [B, 61, 12]
    return dict(
        loc_y=loc_y, loc_z=loc_z, sm_exact=sm_exact, sm_smooth=sm_smooth,
        lx=lx, ly=ly, nm=nm, spd=spd, rx_max=rx_max, tx_max=tx_max,
        da_arm=da_arm, da_leg=da_leg, ens_var=ens_var_acc / max(1, SL_LEN - 1),
        tp_hit=tp_hit, tp_w=tp_w,
    )


def summarize_sim(out, is_panel_a):
    sm = out['sm_exact']
    qstr = sm >= 0.005
    ok = qstr & (out['lx'] >= 2.0) & (out['lx'] <= 4.0) & (out['ly'].abs() <= 1.0) & (out['spd'] >= 4.5) & (out['nm'] >= 0.0)
    pa = is_panel_a > 0.5
    pb = ~pa
    return dict(
        tot=int(ok.sum()), n=len(ok), pct=float(ok.float().mean() * 100.0),
        a_ok=int(ok[pa].sum()), a_n=int(pa.sum()),
        b_ok=int(ok[pb].sum()), b_n=int(pb.sum()),
        qstr=int(qstr.sum()),
        frm=int((~qstr).sum()),
        loob=int((qstr & (out['lx'] > 4.0)).sum()),
        low=int((qstr & (out['spd'] < 4.5)).sum()),
        bnet=int((qstr & (out['nm'] < 0.0)).sum()),
        woob=int((qstr & (out['ly'].abs() > 1.0)).sum()),
        lx_med=float(out['lx'][qstr].median()) if qstr.any() else 0.0,
        spd10=float(out['spd'][qstr].quantile(0.10)) if qstr.any() else 0.0,
        da_arm_rms=float(out['da_arm'].square().mean().sqrt()),
        da_leg_rms=float(out['da_leg'].square().mean().sqrt()),
    )


def cmd_train_bptt(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load 3-member frozen dynamics ensemble
    ens_blob = torch.load(args.surrogate, map_location='cpu', weights_only=True)
    ens_models = []
    for st in ens_blob['members']:
        m = ZeroAnchoredFullBodyDelta(hidden=ens_blob['hidden']).to(device)
        m.load_state_dict(st, strict=True)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        ens_models.append(m)
    print(f'Loaded {len(ens_models)}-member surrogate (val pos={ens_blob["val_racket_pos_mm"]:.2f}mm, R2={ens_blob["val_r2"]:.4f})')

    geom = PhysicalWorldV2(ASSET_SHA).geometry.to(device)
    rid = geom.ids['Racket']
    poly = torch.tensor(string_polygon(load_racket()), dtype=torch.float32, device=device)
    edge = poly.roll(-1, 0) - poly
    poly_inward = F.normalize(torch.stack((-edge[:, 1], edge[:, 0]), dim=-1), dim=-1)
    poly_boundary = (poly * poly_inward).sum(-1)

    # Load actor (ONLY APPROACH_NAMES trainable!)
    ckpt = torch.load(args.base_checkpoint, map_location='cpu', weights_only=False)
    source_path = Path(ckpt['metadata']['init_policy_checkpoint']['path'])
    actor, _ = load_ppo_initialized_actor(source_path, device)
    actor.load_state_dict(ckpt['actor'], strict=True)
    arm_prep = ckpt['metadata'].get('policy_execution', {}).get('arm_preparation_seconds', 0.25)
    if arm_prep is not None:
        actor.configure_approach_gate(arm_prep)

    for name, p in actor.named_parameters():
        p.requires_grad_(name in APPROACH_NAMES)
    trainable = [(name, p) for name, p in actor.named_parameters() if p.requires_grad]
    anchor_params = {name: p.detach().clone() for name, p in trainable}

    # Column-wise ball/tool focus proximal mask on W0 (128 x 794)
    w0_col_mult = torch.full((794,), float(args.ball_tool_focus), dtype=torch.float32, device=device)
    for f in range(11):
        b = f * 69
        w0_col_mult[b + 0:b + 9] = 1.5
        w0_col_mult[b + 45:b + 57] = 0.8
        w0_col_mult[b + 61:b + 67] = 0.25
    w0_col_mult[759:776] = 0.8

    print('Loading BPTT nominal trajectories...')
    data = load_bptt_episodes(args.nominal_dirs, geom, rid, device)
    N = data['s0'].shape[0]
    pa_all = (data['is_panel_a'] > 0.5).nonzero().flatten().cpu().numpy()
    pb_all = (data['is_panel_a'] <= 0.5).nonzero().flatten().cpu().numpy()
    rng = np.random.RandomState(args.seed)
    rng.shuffle(pa_all); rng.shuffle(pb_all)
    n_va_a = min(64, len(pa_all) // 4)
    n_va_b = min(64, len(pb_all) // 4)
    val_idx = torch.tensor(np.concatenate([pa_all[:n_va_a], pb_all[:n_va_b]]), dtype=torch.long, device=device)
    trn_idx = torch.tensor(np.concatenate([pa_all[n_va_a:], pb_all[n_va_b:]]), dtype=torch.long, device=device)
    print(f'BPTT episodes: total={N} (train={len(trn_idx)}, val={len(val_idx)} [A={n_va_a}, B={n_va_b}])')

    # Precompute frozen base-actor pre-contact action a_fb_nom so trust/PA retention is EXACTLY 0 at step 0
    # and NEVER contaminated by post-contact recovery actions (t > c0_rel) or older rollout actors (r4/r8B)!
    actor.eval()
    with torch.no_grad():
        N_all = data['obs_nom'].shape[0]
        a_fb_list = []
        nr_all = torch.zeros(N_all, dtype=torch.bool, device=device)
        for t in range(SL_LEN):
            a_t = actor.forward_with_recovery_gate(data['obs_nom'][:, t], nr_all, sample=False)[0].clamp(-1.0, 1.0)
            a_fb_list.append(a_t)
        data['a_fb_nom'] = torch.stack(a_fb_list, dim=1)  # [N, 61, 18]
        t_grid = torch.arange(SL_LEN, device=device)[None, :]  # [1, 61]
        data['pre_mask'] = (t_grid <= data['c0_rel'][:, None]).float()  # [N, 61]
        if args.pos_lock_weight > 0.0:
            tp_bh_l, tp_bw_l, ab_arm_l, ab_leg_l, bq_l = [], [], [], [], []
            for s_i in range(0, N_all, 128):
                c_idx = torch.arange(s_i, min(s_i + 128, N_all), device=device)
                bo = rollout_bptt_batch(actor, ens_models, geom, rid, poly_inward, poly_boundary, data, c_idx, use_all_ens=True)
                tp_bh_l.append(bo['tp_hit'])
                tp_bw_l.append(bo['tp_w'])
                ab_arm_l.append(bo['da_arm'] + data['a_nom'][c_idx, :, 12:18])
                ab_leg_l.append(bo['da_leg'] + data['a_nom'][c_idx, :, 0:12])
                bq_l.append(bo['sm_exact'] >= 0.005)
            data['tp_base_hit'] = torch.cat(tp_bh_l, dim=0)
            data['tp_base_w'] = torch.cat(tp_bw_l, dim=0)
            data['a_base_arm'] = torch.cat(ab_arm_l, dim=0)
            data['a_base_leg'] = torch.cat(ab_leg_l, dim=0)
            data['base_qstr'] = torch.cat(bq_l, dim=0)
            print(f'Position-Lock Polish Mode ON (w={args.pos_lock_weight}, lx_cap={args.polish_lx_cap}, '
                  f'base_qstr={int(data["base_qstr"].sum())}/{N_all})')

    # Check step-0 calibration on train & val!
    actor.eval()
    with torch.no_grad():
        v0 = rollout_bptt_batch(actor, ens_models, geom, rid, poly_inward, poly_boundary, data, val_idx, use_all_ens=True)
        s0_val = summarize_sim(v0, data['is_panel_a'][val_idx])
        print(f'STEP 0 (base) VAL SIM: {s0_val["tot"]}/{s0_val["n"]} ({s0_val["pct"]:.1f}%) '
              f'[A={s0_val["a_ok"]}/{s0_val["a_n"]} B={s0_val["b_ok"]}/{s0_val["b_n"]}] '
              f'qstr={s0_val["qstr"]} frm={s0_val["frm"]} loob={s0_val["loob"]} low={s0_val["low"]} bnet={s0_val["bnet"]} '
              f'lx={s0_val["lx_med"]:.2f} spd10={s0_val["spd10"]:.2f}')

    opt = torch.optim.Adam([p for _, p in trainable], lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.05)

    best_val_score = -1e9
    best_state = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
    best_step = 0

    for step in range(1, args.steps + 1):
        actor.train()
        perm_b = torch.randint(0, len(trn_idx), (args.batch_size,), device=device)
        b_idx = trn_idx[perm_b]
        out = rollout_bptt_batch(actor, ens_models, geom, rid, poly_inward, poly_boundary, data, b_idx,
                                 use_all_ens=True, obs_noise=args.obs_noise)

        is_pa = data['is_panel_a'][b_idx]
        nom_ok = data['nom_ok'][b_idx]
        pm = data['pre_mask'][b_idx]  # [B, 61]
        a_sim_arm = out['da_arm'] + data['a_nom'][b_idx, :, 12:18]
        a_sim_leg = out['da_leg'] + data['a_nom'][b_idx, :, 0:12]

        if args.pos_lock_weight > 0.0:
            # Position-Locked RK4-Flight Pitch Polisher: LOCK 3D racket sweet-spot within <5mm of base policy
            # so QSTR cannot drop and body surrogate stays strictly in linear regime, while exact analytic
            # collision + RK4 flight eliminates loob / low / bnet on all base string hits!
            q_mask = data['base_qstr'][b_idx].float()
            l_pos = args.pos_lock_weight * (
                (out['tp_hit'] - data['tp_base_hit'][b_idx]).square().sum(-1)
                + 0.35 * (out['tp_w'] - data['tp_base_w'][b_idx]).square().sum(-1).mean(-1)
            )
            l_str_keep = 900.0 * F.relu(0.018 - out['sm_smooth']).square() * q_mask
            l_lx = (45.0 * F.relu(out['lx'] - args.polish_lx_cap).square()
                    + 18.0 * F.relu(2.55 - out['lx']).square()
                    + 0.30 * (out['lx'] - args.polish_lx_tgt).square()) * q_mask
            l_ly = 16.0 * F.relu(out['ly'].abs() - 0.62).square() * q_mask
            l_spd = (5.0 * F.relu(args.polish_spd_min - out['spd']).square()
                     + 20.0 * F.relu(4.70 - out['spd']).square()) * q_mask
            l_nm = (160.0 * F.relu(0.09 - out['nm']).square()
                    + 500.0 * F.relu(0.03 - out['nm']).square()) * q_mask
            l_rnet = 100.0 * F.relu(out['rx_max'] - 1.06).square() + 100.0 * F.relu(out['tx_max'] - 1.38).square()
            ep_loss = (l_pos + l_str_keep + l_lx + l_ly + l_spd + l_nm + l_rnet).clamp(max=25.0)
            l_phys = ep_loss.mean()

            d_fb_arm = a_sim_arm - data['a_base_arm'][b_idx]
            d_fb_leg = a_sim_leg - data['a_base_leg'][b_idx]
            t_rel = torch.arange(SL_LEN, device=device)[None, :] - data['c0_rel'][b_idx, None]
            t_mult = torch.where(t_rel < -16, 4.0, 1.0) if args.polish_late_only else torch.ones_like(pm)
            pm_w = pm * t_mult
            da_arm_sq = (d_fb_arm.square().mean(dim=-1) * pm_w).sum(dim=1) / pm.sum(dim=1).clamp_min(1.0)
            da_leg_sq = (d_fb_leg.square().mean(dim=-1) * pm).sum(dim=1) / pm.sum(dim=1).clamp_min(1.0)
            l_trust = args.trust_weight * da_arm_sq.mean() + args.leg_weight * da_leg_sq.mean()
        else:
            # Pure margin-hinge physics loss: ~0 gradient on already-safe interior hits,
            # 100% of gradient focused on fixing FRAME/MISS, LONG_OOB, LOW_SPD, and BNET!
            l_str = (F.softplus((0.028 - out['sm_smooth']) / 0.007).square() * 0.000049 * 1400.0
                     + 4.0 * (out['loc_y'].square() + out['loc_z'].square()))
            l_lx = 36.0 * F.relu(out['lx'] - 3.52).square() + 14.0 * F.relu(2.50 - out['lx']).square() + 0.04 * (out['lx'] - 3.25).square()
            l_ly = 14.0 * F.relu(out['ly'].abs() - 0.65).square()
            l_spd = 3.0 * F.relu(4.95 - out['spd']).square() + 14.0 * F.relu(4.62 - out['spd']).square()
            l_nm = 140.0 * F.relu(0.08 - out['nm']).square() + 450.0 * F.relu(0.02 - out['nm']).square()
            l_rnet = 80.0 * F.relu(out['rx_max'] - 1.06).square() + 80.0 * F.relu(out['tx_max'] - 1.38).square()

            ep_w = torch.where(is_pa > 0.5, 1.0, 1.30)
            ep_loss = (l_str + l_lx + l_ly + l_spd + l_nm + l_rnet).clamp(max=25.0)
            l_phys = (ep_loss * ep_w).mean()

            d_fb_arm = a_sim_arm - data['a_fb_nom'][b_idx, :, 12:18]
            d_fb_leg = a_sim_leg - data['a_fb_nom'][b_idx, :, 0:12]
            pm_denom = pm.sum(dim=1).clamp_min(1.0)
            da_arm_sq = (d_fb_arm.square().mean(dim=-1) * pm).sum(dim=1) / pm_denom
            da_leg_sq = (d_fb_leg.square().mean(dim=-1) * pm).sum(dim=1) / pm_denom
            pa_ok_mask = (is_pa > 0.5) & (nom_ok > 0.5)
            l_pa_ret = da_arm_sq[pa_ok_mask].mean() if pa_ok_mask.any() else 0.0
            l_trust = args.trust_weight * da_arm_sq.mean() + args.leg_weight * da_leg_sq.mean() + args.pa_anchor_weight * l_pa_ret
        l_ens = args.ens_var_weight * out['ens_var']

        prox_terms = []
        for name, p in trainable:
            if name == 'actor.approach.0.weight' and args.ball_tool_focus != 1.0:
                prox_terms.append(((p - anchor_params[name]).square() * w0_col_mult[None, :]).mean())
            else:
                prox_terms.append((p - anchor_params[name]).square().mean())
        l_prox = args.proximal_weight * (sum(prox_terms) / len(prox_terms))

        loss = l_phys + l_trust + l_ens + l_prox
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # Keep leg head rows (0:12) of actor.approach.4 strictly frozen so approach only refines arm!
        for name, p in trainable:
            if name in ('actor.approach.4.weight', 'actor.approach.4.bias') and p.grad is not None:
                p.grad[:12].zero_()
        torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
        opt.step()
        sched.step()

        if step % 15 == 0 or step == args.steps:
            actor.eval()
            with torch.no_grad():
                vo = rollout_bptt_batch(actor, ens_models, geom, rid, poly_inward, poly_boundary, data, val_idx, use_all_ens=True)
                sv = summarize_sim(vo, data['is_panel_a'][val_idx])
                pm_v = data['pre_mask'][val_idx]
                a_v_arm = vo['da_arm'] + data['a_nom'][val_idx, :, 12:18]
                ref_arm = data['a_base_arm'][val_idx] if args.pos_lock_weight > 0.0 else data['a_fb_nom'][val_idx, :, 12:18]
                d_v_fb = a_v_arm - ref_arm
                fb_arm_rms = float(((d_v_fb.square().mean(-1) * pm_v).sum() / pm_v.sum().clamp_min(1.0)).sqrt())
                a_v_leg = vo['da_leg'] + data['a_nom'][val_idx, :, 0:12]
                ref_leg = data['a_base_leg'][val_idx] if args.pos_lock_weight > 0.0 else data['a_fb_nom'][val_idx, :, 0:12]
                d_v_leg = a_v_leg - ref_leg
                fb_leg_rms = float(((d_v_leg.square().mean(-1) * pm_v).sum() / pm_v.sum().clamp_min(1.0)).sqrt())
                tp_mm = float((vo['tp_hit'] - data['tp_base_hit'][val_idx]).norm(dim=-1).median() * 1000.0) if args.pos_lock_weight > 0.0 else 0.0
                val_score = sv['pct'] + 0.35 * (100.0 * sv['qstr'] / sv['n']) - 15.0 * max(0.0, fb_arm_rms - args.max_da_rms) - (0.6 * max(0.0, tp_mm - 7.0) if args.pos_lock_weight > 0.0 else 0.0)
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_state = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
                    best_step = step
                print(f'  step={step:3d}/{args.steps} loss={float(loss):.3f} (phys={float(l_phys):.3f}) | '
                      f'VAL SIM: {sv["tot"]:3d}/{sv["n"]} ({sv["pct"]:5.1f}%) [A={sv["a_ok"]:2d}/{sv["a_n"]} B={sv["b_ok"]:2d}/{sv["b_n"]}] '
                      f'qstr={sv["qstr"]:3d} frm={sv["frm"]:2d} loob={sv["loob"]:2d} low={sv["low"]:2d} bnet={sv["bnet"]:2d} '
                      f'lx={sv["lx_med"]:.2f} spd10={sv["spd10"]:.2f} d_arm={fb_arm_rms:.3f} tp={tp_mm:.1f}mm (best@{best_step})',
                      flush=True)

    # Verify frozen params untouched & save
    for k, v in ckpt['actor'].items():
        if k not in APPROACH_NAMES and torch.is_tensor(v):
            assert torch.equal(v.cpu(), best_state[k].cpu()), f'Frozen param changed: {k}'
    out_ckpt = {k: v for k, v in ckpt.items()
                if k in ('algorithm', 'adapter_schema', 'actor', 'metadata', 'config',
                         'actor_updates', 'world_updates', 'critic_updates')}
    out_ckpt['actor'] = best_state
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ckpt, args.output)
    pdeltas = torch.cat([(best_state[name].float() - anchor_params[name].cpu().float()).flatten() for name, _ in trainable])
    print(f'Saved BPTT policy (best@{best_step}, param_rms={float(pdeltas.square().mean().sqrt()):.5f}) -> {args.output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='cmd', required=True)

    f = sub.add_parser('fit-surrogate')
    f.add_argument('--base-dir', type=Path, required=True)
    f.add_argument('--output', type=Path, required=True)
    f.add_argument('--n-ensemble', type=int, default=3)
    f.add_argument('--hidden', type=int, default=384)
    f.add_argument('--steps', type=int, default=550)
    f.add_argument('--batch-size', type=int, default=256)
    f.add_argument('--lr', type=float, default=2.5e-3)
    f.add_argument('--seed', type=int, default=77)
    f.add_argument('--device', default='cuda:0')

    t = sub.add_parser('train-bptt')
    t.add_argument('--base-checkpoint', type=Path, required=True)
    t.add_argument('--surrogate', type=Path, required=True)
    t.add_argument('--nominal-dirs', type=Path, nargs='+', required=True)
    t.add_argument('--output', type=Path, required=True)
    t.add_argument('--steps', type=int, default=180)
    t.add_argument('--batch-size', type=int, default=128)
    t.add_argument('--lr', type=float, default=3e-4)
    t.add_argument('--trust-weight', type=float, default=12.0)
    t.add_argument('--leg-weight', type=float, default=250.0)
    t.add_argument('--pa-anchor-weight', type=float, default=25.0)
    t.add_argument('--ens-var-weight', type=float, default=40.0)
    t.add_argument('--proximal-weight', type=float, default=0.10)
    t.add_argument('--ball-tool-focus', type=float, default=10.0)
    t.add_argument('--obs-noise', type=float, default=0.015)
    t.add_argument('--max-da-rms', type=float, default=0.12)
    t.add_argument('--pos-lock-weight', type=float, default=0.0,
                   help='Enable Position-Locked RK4-Flight Pitch Polish mode (e.g. 6000.0)')
    t.add_argument('--polish-lx-cap', type=float, default=3.32)
    t.add_argument('--polish-lx-tgt', type=float, default=3.08)
    t.add_argument('--polish-spd-min', type=float, default=5.05)
    t.add_argument('--polish-late-only', action='store_true')
    t.add_argument('--seed', type=int, default=42)
    t.add_argument('--device', default='cuda:0')

    args = parser.parse_args()
    if args.cmd == 'fit-surrogate':
        cmd_fit_surrogate(args)
    else:
        cmd_train_bptt(args)


if __name__ == '__main__':
    main()
