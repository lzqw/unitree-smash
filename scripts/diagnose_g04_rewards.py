#!/usr/bin/env python3
"""Prepare or run a paired G04 diagnostic. Default prints a plan; no Isaac import.

Native mode is a FUTURE GPU job, to be launched only after explicit approval
through run_gpu_job.py. It never updates weights or changes acceptance.
"""
import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
GROUPS = ('zero', 'existing_expert', 'reference_mean', 'comparison_mean',
          'reference_native_noise', 'reference_noise015', 'comparison_native_noise', 'comparison_noise015')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-native', action='store_true')
    p.add_argument('--output', type=Path)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/jump_ppo_v2c.json')
    p.add_argument('--reference-checkpoint', type=Path, default=ROOT / 'runs/G04/corrected_v2c_main01/native/checkpoint.pt')
    p.add_argument('--comparison-checkpoint', type=Path, default=ROOT / 'runs/G04/corrected_v2c_continue_03/native/checkpoint.pt')
    p.add_argument('--per-group', type=int, default=16)
    args = p.parse_args()
    if args.per_group < 2 or args.per_group > 32:
        p.error('per-group must be between 2 and 32')
    plan = dict(status='PREPARED_NOT_EXECUTED', groups=GROUPS, seed=311001,
        environments=8 * args.per_group, maximum_interactions=8 * args.per_group * 400,
        perturbations=dict(joint_position_rad=.01, joint_velocity_rad_s=.02),
        scenario_source='paired independent diagnostic set; never the frozen acceptance panels',
        reference_checkpoint=str(args.reference_checkpoint), comparison_checkpoint=str(args.comparison_checkpoint),
        reward='unchanged v2c; qualified-recovery draft logged only as a shadow',
        weights_updated=False, histories='actual runtime PolicyHistory; no reconstruction from summaries')
    if not args.run_native:
        print(json.dumps(plan, indent=2)); return
    if args.output is None:
        p.error('native mode needs --output supplied by the budget guard')
    # Verify the actual parent guardian, not just a stale RUNNING file.
    import os
    ledger = json.loads((ROOT / 'status/GPU_LEDGER.json').read_text())
    guarded = any(r['status'] == 'RUNNING' and r['goal'] == 'G04'
                  and r['pid'] == os.getppid() and Path(r['output']).resolve() == args.output.resolve()
                  for r in ledger['runs'])
    if not guarded:
        p.error('native diagnostic requires the live run_gpu_job.py parent')
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device='cuda:0').app
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from quadsmash.rl.native_ppo import NativeRunner
    from quadsmash.rl.action_diagnostics import policy_std
    from quadsmash.tasks.stand.jump_env import JumpEnv, JumpEnvCfg, PROTOCOL
    from quadsmash.tasks.stand.jump_diagnostics import JumpDiagnosticRecorder
    env = recorder = None
    try:
        cfg = JumpEnvCfg(); cfg.scene.num_envs = 8 * args.per_group; cfg.seed = 311001; cfg.sim.device = 'cuda:0'
        env = JumpEnv(cfg); wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
        config = json.loads(args.config.read_text())
        policies = []
        for label, checkpoint in [('reference', args.reference_checkpoint), ('comparison', args.comparison_checkpoint)]:
            runner = NativeRunner(wrapped, config['runner'], log_dir=str(args.output / ('policy' + label)), device=env.device)
            runner.load(str(checkpoint), load_optimizer=False)
            runner.get_inference_policy(device=env.device)  # freezes normalizers via eval mode
            policies.append(runner.alg.policy)
        n = args.per_group
        generator = torch.Generator(device='cpu').manual_seed(311001)
        noise_generator = torch.Generator(device=env.device).manual_seed(311002)
        dq = (.02 * torch.rand((n, 18), generator=generator) - .01).repeat(8, 1).to(env.device)
        qd = (.04 * torch.rand((n, 18), generator=generator) - .02).repeat(8, 1).to(env.device)
        source = json.loads((ROOT / 'reports/G04/capability_all_body_v2/metrics.json').read_text())
        params = torch.tensor([json.loads(source['best'][i % len(source['best'])]['parameters']) for i in range(n)], device=env.device)
        (args.output / 'diagnostic_initial_offsets.json').write_text(json.dumps(dict(
            seed=311001, joint_offsets=dq.cpu().tolist(), joint_velocities=qd.cpu().tolist(),
            expert_parameters=params.cpu().tolist()), indent=2))
        with torch.inference_mode():
            wrapped.reset()
            q = env.robot.data.default_joint_pos.clone(); v = torch.zeros_like(q)
            q[:, env.joint_ids] += dq; v[:, env.joint_ids] = qd
            env.robot.write_joint_state_to_sim(q, v); env.sim.forward()
            env.reset_com_height.copy_((env.mass_weights * env.robot.data.body_com_pos_w[:, :, 2]).sum(1))
            env.policy_history.reset(torch.arange(env.num_envs, device=env.device))
            obs = wrapped.get_observations()
            paired_observation_error = [float((obs['policy'][i*n:(i+1)*n] - obs['policy'][:n]).abs().max()) for i in range(8)]
            (args.output / 'diagnostic_initial_states.json').write_text(json.dumps(dict(
                root_state=env.robot.data.root_state_w.cpu().tolist(), joint_position=q.cpu().tolist(),
                joint_velocity=v.cpu().tolist(), paired_observation_max_abs=paired_observation_error)))
            recorder = JumpDiagnosticRecorder(env, args.output, [name for name in GROUPS for _ in range(n)], PROTOCOL)
            env._diagnostic_observer = recorder
            for step in range(400):
                means = [policy.act_inference(obs) for policy in policies]
                stds = [policy_std(policy) for policy in policies]
                action = torch.zeros((env.num_envs, 18), device=env.device)
                noise = torch.randn((n, 18), device=env.device, generator=noise_generator)
                recorder.std.zero_(); recorder.mean.zero_(); recorder.model_std.zero_()
                for group, index, sigma in [(2, 0, None), (3, 1, None), (4, 0, 'native'),
                                            (5, 0, .15), (6, 1, 'native'), (7, 1, .15)]:
                    sl = slice(group * n, (group + 1) * n)
                    mu = means[index][sl]; sd = stds[index] if sigma == 'native' else float(sigma or 0.)
                    action[sl] = mu + sd * noise
                    recorder.mean[sl] = mu; recorder.std[sl] = sd
                    recorder.model_std[sl] = stds[index]
                t = env._phase()[n:2*n]
                ct, pt, cth, cca, eth, eca, bias, tt = params.unbind(1)
                crouch = (t >= .8) & (t < .8 + ct)
                push = (t >= .8 + ct) & (t < .8 + ct + pt)
                tuck = (t >= .8 + ct + pt) & (t < .8 + ct + pt + tt)
                action[n:2*n, 4:8] = torch.where(crouch, cth, torch.where(push, -eth, torch.where(tuck, .3, 0.)))[:, None]
                action[n:2*n, 8:12] = torch.where(crouch, -cca, torch.where(push, eca, torch.where(tuck, -.6, 0.)))[:, None]
                action[n:2*n, 4:6] += (push * bias)[:, None]; action[n:2*n, 6:8] -= (push * bias)[:, None]
                action[n:2*n].clamp_(-1., 1.); recorder.mean[n:2*n] = action[n:2*n]
                obs, _, _, _ = wrapped.step(action)
            metrics = recorder.finish()
        report = dict(plan, status='MEASURED_DIAGNOSTIC_NOT_ACCEPTANCE', **metrics,
            env_steps=env.num_envs * 400, policy_joint_order=[env.robot.joint_names[i] for i in env.joint_ids],
            paired_observation_max_abs=paired_observation_error,
            config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
            capability_sha256=hashlib.sha256((ROOT/'reports/G04/capability_all_body_v2/metrics.json').read_bytes()).hexdigest(),
            checkpoints={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.reference_checkpoint, args.comparison_checkpoint)},
            policies_evaluated_without_weight_updates=True)
        (args.output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)
    except BaseException:
        (args.output / 'error.txt').write_text(traceback.format_exc()); raise
    finally:
        if recorder is not None: recorder.close()
        if env is not None: env.close()
        app.close(skip_cleanup=True)


if __name__ == '__main__':
    main()
