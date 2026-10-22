#!/usr/bin/env python3
"""Target-conditioned G06 BC/DAgger on the real PhysX shuttle task.

The bounded teacher supplies labels only.  The learner and all reported
checks execute the neural actor from a ground reset in BallEnv; the shuttle
is an ordinary RigidObject integrated by PhysX.
"""
import argparse, csv, hashlib, json, signal, sys, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--config", type=Path, default=ROOT / "configs/g05_imitation.json")
parser.add_argument("--init-checkpoint", type=Path, default=ROOT / "artifacts/G05/policy.pt")
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--updates", type=int, default=1600)
parser.add_argument("--rounds", type=int, default=3)
parser.add_argument("--seed", type=int, default=611017)
parser.add_argument("--physics-dt", type=float, default=0.00025)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
from tensordict import TensorDict
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.tasks.stand.ball_env import BallEnv, BallEnvCfg
from quadsmash.control.jump_teacher import teacher_action
from quadsmash.rl.native_ppo import NativeRunner

TEACHER_PARAMETERS = torch.tensor([
    0.2620947659, 0.1344715506, 0.9543790817, 0.8255642056, 0.4831368923, 1.0,
    -0.3079007566, 0.3499999940, -0.1971214861, -0.9468204977, 0.1242316961,
    0.25, 0.15, -0.20,
], dtype=torch.float32)

stop = False
env = None


def request_stop(*_):
    global stop
    stop = True


signal.signal(signal.SIGTERM, request_stop)
signal.signal(signal.SIGINT, request_stop)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def summarize(rows):
    n = max(1, len(rows))
    return {
        "episodes": len(rows),
        "contacts": sum(bool(r["contact"]) for r in rows),
        "airborne_contacts": sum(bool(r["airborne_contact"]) for r in rows),
        "recoveries": sum(bool(r["recovery_success"]) for r in rows),
        "contact_recovery": sum(bool(r["contact"] and r["recovery_success"]) for r in rows),
        "contact_rate": sum(bool(r["contact"]) for r in rows) / n,
        "airborne_contact_rate": sum(bool(r["airborne_contact"]) for r in rows) / n,
        "contact_recovery_rate": sum(bool(r["contact"] and r["recovery_success"]) for r in rows) / n,
        "mean_ball_min_distance_m": sum(float(r["ball_min_distance_m"]) for r in rows) / n,
    }


try:
    args.output.mkdir(parents=True, exist_ok=True)
    if args.physics_dt <= 0 or round(0.01 / args.physics_dt) * args.physics_dt != 0.01:
        raise ValueError("--physics-dt must divide the 10 ms policy period")
    cfg = BallEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.sim.device = "cuda:0"
    cfg.sim.dt = args.physics_dt
    cfg.decimation = round(0.01 / args.physics_dt)
    cfg.sim.render_interval = cfg.decimation
    env = BallEnv(cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    runner = NativeRunner(wrapped, config["runner"], log_dir=str(args.output / "ppo"), device=env.device)
    runner.load(str(args.init_checkpoint.resolve()), load_optimizer=False)
    policy = runner.alg.policy
    params = TEACHER_PARAMETERS.to(env.device).repeat(env.num_envs, 1)
    (args.output / "teacher.json").write_text(json.dumps({
        "parameters": TEACHER_PARAMETERS.tolist(),
        "source": "G05 full-load capability search generation 19",
        "physics_dt_s": args.physics_dt,
    }, indent=2) + "\n", encoding="utf-8")
    env_steps = 0

    def rollout(seed, learned=False, collect=False):
        global env_steps
        env._episode_has_stepped.zero_()
        env._metrics_started = False
        env.episode_records.clear()
        env.ball_episode_records.clear()
        obs, _ = wrapped.reset()
        xs, ys, safe = [], [], []
        with torch.inference_mode():
            for _ in range(400):
                target = teacher_action(obs["policy"], params)
                if collect:
                    xs.append(obs["policy"].clone())
                    ys.append(target.clone())
                    safe.append((~env.body_ground_seen).clone())
                action = policy.act_inference(obs) if learned else target
                if not torch.isfinite(action).all():
                    raise RuntimeError("nonfinite learner action")
                obs, _, _, _ = wrapped.step(action)
                env_steps += env.num_envs
        rows = list(env.ball_episode_records)
        data = (torch.stack(xs), torch.stack(ys), torch.stack(safe)) if collect else None
        return rows, data

    teacher_rows, data = rollout(args.seed, learned=False, collect=True)
    if len(teacher_rows) < env.num_envs:
        raise RuntimeError(f"teacher did not complete all episodes: {len(teacher_rows)}/{env.num_envs}")
    with (args.output / "teacher_episodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(teacher_rows[0]))
        writer.writeheader()
        writer.writerows(teacher_rows)
    x, y, safe = (t.clone() for t in data)
    validation = torch.arange(env.num_envs, device=env.device) % 5 == 0
    trainmask = safe & (~validation)[None, :]
    valmask = safe & validation[None, :]
    train_x, train_y = x[trainmask], y[trainmask]
    val_x, val_y = x[valmask], y[valmask]
    torch.save({"observations": x.cpu(), "actions": y.cpu(), "safe": safe.cpu(),
                "validation_episode": validation.cpu(), "seed": args.seed}, args.output / "teacher_dataset.pt")
    del x, y, safe, data
    policy.train()
    policy.update_normalization(TensorDict({"policy": train_x}, batch_size=[len(train_x)]))
    policy.eval()
    optimizer = torch.optim.Adam(policy.actor.parameters(), lr=3e-4)
    report = {
        "method": "real-physx shuttle teacher-guided BC + DAgger",
        "physics_dt_s": args.physics_dt,
        "teacher": summarize(teacher_rows),
        "learner_checks": [],
        "env_steps": env_steps,
        "config_sha256": digest(args.config),
        "init_checkpoint_sha256": digest(args.init_checkpoint),
    }
    best = -1
    for round_id in range(args.rounds):
        active = torch.where((train_x[:, 775] >= 0.75) & (train_x[:, 775] <= 1.8))[0]
        losses = []
        for update in range(args.updates):
            if stop:
                break
            idx = torch.cat((active[torch.randint(len(active), (768,), device=env.device)],
                             torch.randint(len(train_x), (256,), device=env.device)))
            pred = policy.act_inference(TensorDict({"policy": train_x[idx]}, batch_size=[len(idx)]))
            loss = (pred - train_y[idx]).square().mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.actor.parameters(), 1.0)
            optimizer.step()
            if update % 250 == 0:
                losses.append(float(loss))
        val_mse = float((policy.act_inference(TensorDict({"policy": val_x}, batch_size=[len(val_x)])) - val_y).square().mean()) if len(val_x) else None
        rows, dagger = rollout(args.seed + 100 + round_id, learned=True, collect=True)
        check = dict(round=round_id, **summarize(rows), validation_mse=val_mse, training_mse_samples=losses)
        report["learner_checks"].append(check)
        report["env_steps"] = env_steps
        with (args.output / f"learner_episodes_{round_id:02d}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        runner.save_restart(args.output / f"bc_{round_id:02d}.pt", infos={"method": "real-physx shuttle BC", "round": round_id})
        if check["contact_recovery"] > best:
            best = check["contact_recovery"]
            runner.save_restart(args.output / "checkpoint.pt", infos={"method": "real-physx shuttle BC", "round": round_id})
            report["best_round"] = round_id
        (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("G06_BC_PROGRESS=" + json.dumps(check, sort_keys=True), flush=True)
        if stop:
            break
        if dagger is not None:
            dx, dy, ds = (t.clone() for t in dagger)
            mask = ds & (~validation)[None, :]
            train_x = torch.cat((train_x, dx[mask]))
            train_y = torch.cat((train_y, dy[mask]))
            torch.save({"observations": dx.cpu(), "teacher_actions": dy.cpu(), "safe": ds.cpu(),
                        "seed": args.seed + 100 + round_id}, args.output / f"dagger_{round_id:02d}.pt")
            del dx, dy, ds
    report["best_contact_recovery"] = best
    report["best_contact_recovery_rate"] = best / max(1, args.num_envs)
    report["checkpoint"] = str(args.output / "checkpoint.pt")
    report["env_steps"] = env_steps
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
except BaseException:
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    raise
finally:
    if env is not None:
        env.close()
    app.close(skip_cleanup=True)
