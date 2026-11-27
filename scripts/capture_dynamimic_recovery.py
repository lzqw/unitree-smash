"""Collect full native episodes, retaining reset-safe robot transitions and G07 start states."""
import hashlib
import os

import torch

from quadsmash.rl.supermimic_physical_capture_v1 import capture_controller, tensor_tree_copy
from quadsmash.rl.supermimic_physical_dataset_v1 import encode_physical
from quadsmash.rl.supermimic_physical_observation_v1 import capture_native_inputs
from scripts.run_supermimic_recovery_baseline import dump


def physical_state(env):
    inputs = capture_native_inputs(env)
    return encode_physical(vars(inputs))


def native_boundary(env):
    return (physical_state(env), env._sm_joint_targets.clone(), env.observed_foot_contact.clone(),
            env.landed_once.clone(), env.episode_index.clone())


@torch.no_grad()
def collect(args, env, wrapper, obs, actor, companion, cases, stop):
    from quadsmash.rl.supermimic_actuator_identification import actuator_contract
    import json

    if len({c['case_id'] for c in cases}) != 256 or env.num_envs != 256:
        raise ValueError('Expected 256 distinct training cases')
    root = args.output
    native_contract = json.loads((root/'task_contract.json').read_text())
    dump(root/'actuator_contract.json', actuator_contract(env, native_contract))
    actor.install_release_arm_reference()
    companion.install_release_arm_reference()
    companion.release_arm_reference.load_state_dict(actor.release_arm_reference.state_dict())
    companion.actor.recovery.load_state_dict(actor.actor.recovery.state_dict())
    prev = physical_state(env)
    episode = env.episode_index.clone()
    closed = torch.zeros_like(episode, dtype=torch.bool)
    old_ground = closed.clone()
    old_net = closed.clone()
    old_relaxed = closed.clone()
    old_strict = closed.clone()
    old_run = torch.zeros_like(episode, dtype=prev.dtype)
    chunks, rows = [], []
    storage = root/'trajectories'
    storage.mkdir()
    for tick in range(800):
        if stop['signal']:
            raise InterruptedError(stop['signal'])
        x = obs['policy'].clone()
        gate = env.rally.had_contact & env.rally.released
        release = env.rally.release_time.clone()
        mean = actor._gated_distribution(x, gate, release_time=release).loc
        mean[1::2] = companion._gated_distribution(x[1::2], gate[1::2], release_time=release[1::2]).loc
        action = mean.clamp(-1, 1)
        if tick % 16 == 0 and (gate & ~closed).any():
            memory = capture_controller(env._sm_residual_controller, plan_observation_cache={})
            torch.save(tensor_tree_copy(memory, cpu=True), storage/f'g07_{tick:04d}.pt')
        obs, _, done, _ = wrapper.step(action)
        after, target, feet, landed, post_episode = env.dyna_boundary
        if ((post_episode != episode) & ~closed).any():
            raise ValueError('Invalid terminal boundary capture')
        done = done.bool().reshape(-1)
        dense, stable, longest, ground, net, relaxed, strict = env.recovery_ppo_boundary
        reward = ((~ground & ~net) * (dense + .08*stable + 4*(longest-old_run).clamp_min(0)/.8
                                     + 2.5*(relaxed & ~old_relaxed) + 8*(strict & ~old_strict))
                  - 15*(ground & ~old_ground).float() - 20*(net & ~old_net).float())
        row = dict(before=prev, after=after, final_joint_targets=target, obs=x, request=action,
                   live=~closed, gate=gate, release_time=release, reward=reward,
                   done=done, feet=feet, landed=landed, body=ground, net=net, relaxed=relaxed,
                   strict=strict, active=gate & ~closed & ~old_net & ~old_ground)
        # release_time, observations and the public gate must describe the pre-step boundary.
        rows.append({key: value.detach().cpu() for key, value in row.items()})
        closed |= done
        old_run, old_ground, old_net, old_relaxed, old_strict = longest, ground, net, relaxed, strict
        prev = after
        if len(rows) == 32 or closed.all() or tick == 799:
            path = storage/f'chunk_{len(chunks):03d}.pt'
            torch.save({k: torch.stack([r[k] for r in rows]) for k in rows[0]}, path.with_suffix('.tmp'))
            os.replace(path.with_suffix('.tmp'), path)
            rows.clear()
            chunks.append(path.name)
            print(f'Native Dyna ticks {tick+1}, completed episodes {int(closed.sum())}/256', flush=True)
        if closed.all():
            break
    if not closed.all() or any(r['numerical_failures'] for r in env.rally_records):
        raise ValueError('Incomplete or numerically invalid native batch')
    dump(storage/'manifest.json', dict(schema='dynamimic-native-recovery-v1', cases=cases,
         episode_indices=episode.cpu().tolist(), origins=env.scene.env_origins.cpu().tolist(),
         lane_approach=['h8d' if i%2 else 'primary' for i in range(256)], chunks=chunks,
         checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
         companion_sha256=hashlib.sha256(args.recovery_probe_companion.read_bytes()).hexdigest(),
         ticks=tick+1, dt=float(env.step_dt)))
