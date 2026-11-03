#!/usr/bin/env python3
"""Explicit reset-persistent recovery trial over a measured fixed residual actor.

Original task thresholds, robot and actuators are retained; reset behavior changes explicitly.
Ground dimensions and first nonfoot contact events are recorded explicitly.
This is development evidence, never an inherited or formal acceptance score.
"""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys
import traceback
import io
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--persistent-recovery-trial', action='store_true', required=True,
    help='Explicitly keep declared recovery assistance effective after every controller reset')
parser.add_argument('--precision-receipt', type=Path, required=True)
parser.add_argument('--ground-xy-m', type=float, choices=(64.,4096.), required=True)
parser.add_argument('--training-indices', type=int, nargs='+', default=None)
parser.add_argument('--noise', choices=('none', 'A', 'B'), default='none')
parser.add_argument('--seed', type=int, default=724201)
parser.add_argument('--record-first-episodes',action='store_true',
    help='Losslessly retain actual policy inputs and dispatched actions for every selected first episode')
parser.add_argument('--allow-offline-finetune', action='store_true',
    help='Evaluate an actor-only trace-finetuned candidate; the result is measured here and is never inherited')
parser.add_argument('--allow-declared-controller-variant', action='store_true',
    help='Explicit diagnostic trial for a controller variant declared in checkpoint metadata; never a policy score')
assistance_options=parser.add_mutually_exclusive_group()
assistance_options.add_argument('--landing-config',type=Path,
    help='Explicit controller trial over the frozen source; performance must be measured anew')
assistance_options.add_argument('--support-config',type=Path,
    help='Explicit centroidal support trial through the original position PD; no aerial override')
assistance_options.add_argument('--disable-support',action='store_true',
    help='Explicit ablation of support stored in the trained policy; recorded as a controller trial')
from scripts.origin0_launcher_parser_v4 import add_origin0_launcher_args
add_origin0_launcher_args(parser,AppLauncher)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaacsim.core.utils.stage import get_current_stage
from quadsmash.control.full_racket_body_residual_v4 import ResidualActionMixerV4
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases, training_noise, summarize_training
from quadsmash.tasks.stand.contact_recovery_origin0_v5 import contact_recovery_origin0_cfg as full_racket_origin0_cfg
from quadsmash.rl.contact_recovery_source_v5 import snapshot_execution_sources
from quadsmash.tasks.stand.full_racket_eval_env_v4 import FrozenPanelEnv
from quadsmash.tasks.stand.visual_geometry import export_robot_geometry
from quadsmash.usd_dependencies_v4 import audit_composed_asset
from quadsmash.control.shared_target_assistance_v6 import LandingAssistedController,RECOVERY_FIELDS
from quadsmash.control.persistent_recovery_assistance_v1 import PersistentRecoveryAssistanceV1,SCHEMA as RECOVERY_EXECUTION_SCHEMA
from quadsmash.control.reactive_landing_v1 import settings as landing_settings
from quadsmash.control.centroidal_support_v1 import settings as support_settings,SupportAssistedController
from quadsmash.control.support_policy_contract_v1 import policy_support_config,add_support_controller
from scripts.ground_extent_diagnostic_v1 import observe_ground
from scripts.strike_ground_contact_observer_v1 import StrikeGroundContactObserver
from scripts.native_whole_body_admission_v1 import observe_native_whole_body
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
from scripts.ground64_training_support_v1 import GROUND

SHARED_SCHEMA='g07-public-shared-target-bias-v5'
RESIDUAL_SCHEMA='g07-shared-target-residual-ppo-v1'


def load_shared_factory():
    path=ROOT.parent/'unitree_smash/reports/G07_full_racket_v5/current_intercept_diagnosis_20261027/native_factory.py'
    spec=importlib.util.spec_from_file_location('shared_target_eval_factory',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SharedTargetResidualController:
    """Freeze the measured shared-target source and apply only exported residual actor."""
    def __init__(self, checkpoint, count, device,landing_trial=None,support_trial=None,disable_support=False):
        if checkpoint.get('schema') not in (RESIDUAL_SCHEMA,SHARED_SCHEMA):
            raise ValueError('Require the shared-target source or residual policy schema')
        has_actor=checkpoint['schema']==RESIDUAL_SCHEMA
        base=checkpoint['base_controller'] if has_actor else checkpoint
        if base.get('schema') != SHARED_SCHEMA or not base.get('native_execution'):
            raise ValueError('Require the native shared-target base controller')
        factory=load_shared_factory()
        self.base=factory.make_controller(base['source_public_policy'],count,device,
            bias_world_m=base['shared_target_bias']['bias_world_m'])
        metadata=checkpoint.get('residual_training_metadata',{})
        if not metadata.get('recovery_assistance'):
            raise ValueError('Persistent recovery trial requires explicit recovery settings')
        self.base=PersistentRecoveryAssistanceV1(self.base,metadata['recovery_assistance'])
        self.landing_config=landing_trial if landing_trial is not None else metadata.get('landing_assistance')
        self.support_config=policy_support_config(metadata,support_trial,disable_support)
        if self.support_config is not None and self.landing_config is not None:
            raise ValueError('Separate the aerial-position and grounded-support comparisons')
        if self.landing_config is not None:
            self.base=LandingAssistedController(self.base,count,device,self.landing_config,checkpoint['asset_sha256'])
        self.base=add_support_controller(self.base,count,device,self.support_config,checkpoint['asset_sha256'])
        self.actor=torch.jit.load(io.BytesIO(checkpoint['residual_actor_jit']),map_location=device).eval() if has_actor else None
        self.mixer=ResidualActionMixerV4(count,device,checkpoint['residual_joint_scale_rad'],
            float(checkpoint.get('residual_arm_target_rate_rad_s',5.)),
            float(checkpoint.get('residual_scale_limit_rad',max(checkpoint['residual_joint_scale_rad'])))) if has_actor else None

    @property
    def reference_controller(self):
        return self.base.reference_controller

    @property
    def logs(self):
        return self.base.logs

    @logs.setter
    def logs(self,value):
        self.base.logs=value

    def reset(self,ids=None):
        self.base.reset(ids)
        if self.mixer is not None:self.mixer.reset(ids)

    @torch.inference_mode()
    def action(self,observation,default,limits):
        base=self.base.action(observation,default,limits)
        if self.actor is None:return base
        residual=self.actor(observation['policy'])
        return self.mixer.action(base,residual,observation['policy'],default,limits)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'INPUTS.json').exists():
        raise ValueError('Use a fresh output; never overwrite a previous diagnostic')
    protocol_path = ROOT / 'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'
    protocol = json.loads(protocol_path.read_text())
    cases, indices = training_cases(ROOT, protocol, args.training_indices)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint.get('schema') not in (RESIDUAL_SCHEMA,SHARED_SCHEMA) or checkpoint.get('protocol_sha256') != digest(protocol_path):
        raise ValueError('Require the explicitly measured shared-target residual policy')
    offline_finetune = bool(args.allow_offline_finetune and
        checkpoint.get('residual_training_metadata', {}).get('offline_trace_finetune') and
        checkpoint.get('base_controller', {}).get('native_execution'))
    if not checkpoint.get('native_execution') and not offline_finetune:
        raise ValueError('Evaluate an actually executed policy; unexecuted proposals require explicit offline-finetune admission')
    if checkpoint['parameters'].ndim != 1 or not torch.isfinite(checkpoint['parameters']).all():
        raise ValueError('Require one fixed finite policy, not an optimizer population')
    has_actor=checkpoint['schema']==RESIDUAL_SCHEMA
    if has_actor and checkpoint['base_controller'].get('schema') != SHARED_SCHEMA:
        raise ValueError('Residual base must be the measured shared-target controller')
    if has_actor and checkpoint.get('residual_base_mode') != 'shared_target_bias':
        raise ValueError('Unexpected residual base mode')
    if checkpoint['asset_sha256'] != protocol['asset']['sha256']:
        raise ValueError('Checkpoint belongs to a different physical model')
    if digest(ROOT / protocol['asset']['path']) != protocol['asset']['sha256']:
        raise ValueError('Frozen physical asset changed')
    for name, expected in protocol['physical_configs'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen physical config changed: '+name)
    amplitudes = (dict(joint_position_rad=0., joint_velocity_rad_s=0., root_xy_m=0., root_yaw_rad=0.)
                  if args.noise == 'none' else protocol['initialization']['perturbations'][args.noise])
    count = len(cases)
    if count != 128 or args.noise != "none" or not args.record_first_episodes:
        raise ValueError("Require all original128 zero-noise ground starts and complete action recording")
    if not has_actor or ((args.landing_config or args.support_config or args.disable_support)
                         and not args.allow_declared_controller_variant):
        raise ValueError("Require the actually trained residual actor without an undeclared controller change")
    if args.ground_xy_m != 64. or checkpoint.get("residual_training_metadata",{}).get("ground_extent_diagnostic") != GROUND:
        raise ValueError("The evaluated ground must match the trained ground64 policy")
    ground_diagnostic = dict(original_configured_m=[4096.,4096.,.1], requested_m=[args.ground_xy_m,args.ground_xy_m,.1])
    landing_trial=landing_settings(json.loads(args.landing_config.read_text())) if args.landing_config else None
    support_trial=support_settings(json.loads(args.support_config.read_text())) if args.support_config else None
    if args.allow_declared_controller_variant:
        declared_landing=checkpoint.get('residual_training_metadata',{}).get('landing_assistance')
        declared_support=checkpoint.get('residual_training_metadata',{}).get('support_assistance')
        if landing_trial is not None and declared_landing != landing_trial:
            raise ValueError('Declared landing controller variant differs from checkpoint metadata')
        if support_trial is not None and declared_support != support_trial:
            raise ValueError('Declared support controller variant differs from checkpoint metadata')
    effective_support=policy_support_config(checkpoint.get('residual_training_metadata',{}),support_trial,args.disable_support)
    controller_trial=bool(args.persistent_recovery_trial)
    noise = training_noise(count, amplitudes, args.seed)
    atomic_json(args.output / 'INPUTS.json', dict(policy_sha256=digest(args.checkpoint),
        protocol_sha256=digest(protocol_path), training_suite=protocol['training_suite'],
        source_checkpoint_protocol_sha256=checkpoint['protocol_sha256'],
        execution_worktree=str(ROOT),
        origin_precision_receipt_sha256=digest(args.precision_receipt),origin_spacing_m=0.,
        training_indices=indices, reset_noise_amplitudes=amplitudes, reset_noise_seed=args.seed,
        controller_trial=controller_trial, ground_extent_diagnostic=ground_diagnostic,
        recovery_assistance_execution=RECOVERY_EXECUTION_SCHEMA,
        landing_trial=landing_trial,support_trial=support_trial,
        support_config=effective_support,support_disabled=args.disable_support,
        record_all_first_episode_actions=args.record_first_episodes,
        residual_actor_enabled=has_actor,
        offline_trace_finetune_candidate=offline_finetune,
        reset_noise={k: v.tolist() for k, v in noise.items()},
        scope='TRAIN-only numerical ground-extent diagnostic; original task scoring, no inherited performance or formal cases.',
        formal_cases_read=False, formal_acceptance=False))
    execution_snapshot=snapshot_execution_sources(ROOT,args.output)
    cfg = full_racket_origin0_cfg(count, protocol, args.precision_receipt)
    cfg.seed = args.seed
    cfg.episode_length_s = protocol['episode_seconds']
    original_ground_size = tuple(cfg.ground_size_m)
    if original_ground_size != (4096.,4096.,.1):
        raise ValueError('Unexpected source ground geometry')
    cfg.ground_size_m = (args.ground_xy_m,args.ground_xy_m,original_ground_size[2])
    if abs(cfg.sim.dt*cfg.decimation-protocol['policy_dt_s']) > 1e-12:
        raise ValueError('Control period differs from original protocol')
    env = FrozenPanelEnv(cfg, cases=cases, reset_noise=noise,
        contact_flight_recovery=protocol['recovery_association']['schema'])
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
    controller = SharedTargetResidualController(checkpoint, count, env.device,landing_trial,support_trial,args.disable_support)
    stopping = False
    def stop(_signal, _frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        if env.robot.is_fixed_base:
            raise ValueError('Require a floating robot with an actual ground reset')
        atomic_json(args.output / 'asset_dependencies.json', audit_composed_asset(ROOT, cfg.robot.spawn.usd_path))
        if controller.support_config is not None:
            from quadsmash.control.support_admission_v1 import validate_native_support_interface
            atomic_json(args.output/'native_support_interface.json',
                validate_native_support_interface(env,get_current_stage()))
        observation, _ = wrapped.reset()
        controller.reset()
        requested_recovery=checkpoint['residual_training_metadata']['recovery_assistance']
        effective_recovery={}
        reference=controller.reference_controller
        for name,value in requested_recovery.items():
            index=RECOVERY_FIELDS[name][0]
            if name == 'recovery_side_aware_roll_feedback':
                actual=float(reference.teacher.standing.side_aware_roll_feedback)
                matches=bool(actual == float(value))
            elif index is None:
                actual=reference.teacher.leg_recovery_blend.seconds
                matches=abs(actual-float(value))<1e-6
            else:
                actual=float(reference.parameters[0,index])
                matches=bool(torch.allclose(reference.parameters[:,index],
                    torch.full_like(reference.parameters[:,index],float(value)),atol=1e-6,rtol=0.))
            if not matches:raise ValueError('Recovery setting lost during actual reset: '+name)
            effective_recovery[name]=actual
        atomic_json(args.output/'recovery_parameters_after_reset.json',dict(
            schema=RECOVERY_EXECUTION_SCHEMA,requested=requested_recovery,effective=effective_recovery,
            verified_lanes=count,all_lanes_match=True,controller_trial=True,formal_acceptance=False))
        atomic_json(args.output/'ground_geometry.json',observe_ground(get_current_stage(),original_ground_size,cfg.ground_size_m))
        admission,_interface=observe_native_whole_body(env,get_current_stage(),LoadedPinocchioModelV1())
        atomic_json(args.output/'native_admission.json',admission)
        if not admission['passed']:raise ValueError('Native loaded model/actuator admission failed')
        if getattr(env,'_diagnostic_observer',None) is not None:
            raise ValueError('Preserve existing physical observer')
        contact_observer=StrikeGroundContactObserver(env,args.output)
        env._diagnostic_observer=contact_observer
        if env.scene.env_origins.any():
            raise ValueError('Require actual co-located independent origins')
        atomic_json(args.output / 'origin_layout.json', dict(actual_environment_origins_m=env.scene.env_origins.cpu().tolist(),
            origin_precision_receipt_sha256=digest(args.precision_receipt)))
        initial = {r['env_id']: r for r in env.initial_states}
        if set(initial) != set(range(count)):
            raise ValueError('Missing first ground reset states')
        for i, row in initial.items():
            position = np.array(row['root_state_local'][:3])
            expected = np.r_[noise['root_xy_m'][i], protocol['initialization']['root_position_m'][2]]
            if np.max(abs(position-expected)) > 1e-5:
                raise ValueError('Native reset origin/height differs from the requested ground initialization')
        atomic_json(args.output / 'initial_states.json', [initial[i] for i in range(count)])
        export_robot_geometry(get_current_stage(), env.robot, args.output)
        default = env.robot.data.default_joint_pos[0, env.joint_ids]
        limits = env.robot.data.joint_pos_limits[0, env.joint_ids]
        atomic_json(args.output/'native_command_inputs.json',dict(
            default_joint_position_rad=default.cpu().tolist(),joint_limits_rad=limits.cpu().tolist(),
            protocol_sha256=digest(protocol_path),policy_sha256=digest(args.checkpoint)))
        finished = torch.zeros(count, dtype=torch.bool, device=env.device)
        poses, balls, clocks, valid = [], [], [], []
        public, actions, public_valid = [], [], []
        support_records={}
        requested = round(protocol['episode_seconds']/protocol['policy_dt_s'])
        fixed_trace=None
        if args.record_first_episodes:
            from quadsmash.rl.fixed_policy_trace_v1 import FixedPolicyTraceV1
            fixed_trace=FixedPolicyTraceV1(args.output/'fixed_policy_traces',count,requested,
                dict(native_execution=True,source_kind='Fixed learned residual actor on explicitly recorded ground64',
                     ground_extent_diagnostic=ground_diagnostic,
                     policy_sha256=digest(args.checkpoint),protocol_sha256=digest(protocol_path),
                     asset_sha256=protocol['asset']['sha256'],
                     default_joint_position_rad=default.cpu().tolist(),joint_limits_rad=limits.cpu().tolist(),
                     controller_trial=controller_trial,
                     recovery_assistance_execution=RECOVERY_EXECUTION_SCHEMA,
                     support_config=controller.support_config,landing_config=controller.landing_config))
        with torch.inference_mode():
            for tick in range(requested):
                obs = observation['policy']
                action = controller.action(observation, default, limits)
                if action.shape != (count, 18) or not torch.isfinite(action).all() or (abs(action)>1).any():
                    raise ValueError('Nonfinite, invalid-width or unbounded native action')
                if fixed_trace is not None:
                    fixed_trace.append(obs.cpu().numpy(),action.cpu().numpy(),env.episode_index.cpu().numpy())
                public.append(obs[:2].cpu().numpy().copy())
                actions.append(action[:2].cpu().numpy().copy())
                public_valid.append((~finished[:2]).cpu().numpy().copy())
                if tick % 2 == 0:
                    data = env.robot.data
                    pose = torch.cat((data.body_link_pos_w-env.scene.env_origins[:, None], data.body_link_quat_w), -1).cpu().numpy()
                    ball = (env.shuttle.data.root_pos_w-env.scene.env_origins).cpu().numpy()
                    clock = obs[:, 775].cpu().numpy().copy()
                    if not all(np.isfinite(a).all() for a in [pose, ball, clock]):
                        raise ValueError('Nonfinite actual body/ball/clock stream')
                    poses.append(pose);balls.append(ball);clocks.append(clock)
                    valid.append((~finished).cpu().numpy().copy())
                    if controller.support_config is not None:
                        diagnostic=controller.base.support.last_diagnostic
                        if diagnostic:
                            selected=diagnostic['env_ids'];keep=~finished[selected]
                            values={k:v[keep] for k,v in diagnostic.items() if torch.is_tensor(v)}
                            values['clock_s']=obs[selected[keep],775]
                            values['observed_foot_net_forces_world_n']=env.feet_contact.data.net_forces_w[selected[keep]]
                            for key,value in values.items():
                                support_records.setdefault(key,[]).append(value.cpu().numpy().copy())
                observation, _, done, _ = wrapped.step(action)
                if fixed_trace is not None:fixed_trace.close(done.bool().cpu().numpy())
                finished |= done.bool()
                if (tick+1) % 100 == 0 or stopping or bool(finished.all()):
                    atomic_json(args.output / 'progress.json', dict(control_ticks=tick+1,
                        requested_ticks=requested, first_episodes_complete=int(finished.sum()),
                        training_indices=indices, formal_eligible=False, formal_acceptance=False))
                    atomic_json(args.output / 'episodes.partial.json', env.rally_records)
                    np.savez_compressed(args.output / 'all_first_episodes.partial.npz',
                        link_poses=np.stack(poses), ball_position_m=np.stack(balls),
                        episode_clock_s=np.stack(clocks), valid_first_episode=np.stack(valid))
                    print(json.dumps(dict(tick=tick+1, first_complete=int(finished.sum()))), flush=True)
                if stopping or bool(finished.all()):
                    break
        np.savez_compressed(args.output / 'all_first_episodes.npz', link_poses=np.stack(poses),
            ball_position_m=np.stack(balls), episode_clock_s=np.stack(clocks), valid_first_episode=np.stack(valid))
        np.savez_compressed(args.output / 'public_and_actions_first2.npz', observations=np.stack(public),
            actions=np.stack(actions), valid_first_episode=np.stack(public_valid))
        if support_records:
            np.savez_compressed(args.output/'support_reference_and_measured_forces.npz',
                **{key:np.concatenate(values) for key,values in support_records.items()})
        contact_observer.save()
        atomic_json(args.output / 'contacts.json', env.contact_records)
        atomic_json(args.output / 'causal_plans.json', controller.logs)
        first, jumps = {}, {}
        for row in env.rally_records:
            first.setdefault(row['env_id'], row)
        for row in env.episode_records:
            jumps.setdefault(row['env_id'], row)
        if stopping and (len(first) != count or len(jumps) != count):
            atomic_json(args.output / 'interrupted.json', dict(control_ticks=tick+1,
                actual_control_env_steps=(tick+1)*count, completed_first_episodes=len(first),
                requested_first_episodes=count, formal_acceptance=False,
                resume_semantics='Replay the same fixed policy/cases/noise from ground; no state restoration claimed.'))
            return
        if len(first) != count or len(jumps) != count:
            raise ValueError('Incomplete first-episode TRAIN diagnostic')
        rows = [dict(first[i], jump=jumps[i], case_id=cases[i]['case_id'], ground_init=True,
                     source_training_index=indices[i]) for i in range(count)]
        protocol['_root'] = str(ROOT)
        summary = summarize_training(rows, cases, indices, protocol)
        if checkpoint.get('residual_training_metadata',{}).get('attack_speed_threshold_mps') == 5.0:
            from scripts.compare_frozen_complete5_v2 import complete5_summary
            primary=complete5_summary(rows,protocol)
            summary.update(complete5=primary['complete5'],complete5_success_rate=primary['complete5_rate'],
                primary_metric='complete5',primary_speed_threshold_mps=5.0,
                complete5_case_outcomes=primary['stages'],legacy_complete_field_speed_threshold_mps=8.0)
        if fixed_trace is not None:
            from quadsmash.evaluation_v4 import independent_episode
            summary['fixed_policy_recording']=fixed_trace.save(rows,[independent_episode(row,protocol) for row in rows])
        summary.update(policy_sha256=digest(args.checkpoint), protocol_sha256=digest(protocol_path),
            noise_level=args.noise, noise_seed=args.seed, actual_control_env_steps=(tick+1)*count,
            controller_trial=controller_trial,support_disabled=args.disable_support,
            recovery_assistance_execution=RECOVERY_EXECUTION_SCHEMA,
            landing_config=controller.landing_config,support_config=controller.support_config,
            residual_actor_enabled=has_actor, ground_extent_diagnostic=ground_diagnostic)
        atomic_json(args.output / 'episodes.json', rows)
        atomic_json(args.output / 'metrics.json', summary)
        atomic_json(args.output / 'completed.json', dict(env_steps=(tick+1)*count))
        print(json.dumps(summary), flush=True)
    finally:
        env._diagnostic_observer=None
        env.close()


try:
    main()
except BaseException:
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'error.txt').write_text(traceback.format_exc())
    raise
finally:
    app.close(skip_cleanup=True)
