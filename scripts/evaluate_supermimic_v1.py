#!/usr/bin/env python3
"""Frozen TRAIN128 evaluation with source task and pre-reset terminal evidence.

This is a development regression, not the independent A/B formal acceptance.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import signal
import faulthandler
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--checkpoint', type=Path)
    source_group.add_argument('--init-policy-checkpoint', type=Path,
                              help='Zero-update native source evaluation, no world model or optimizers')
    parser.add_argument('--init-policy-recovery-gate',choices=('legacy','causal'),default='legacy',
                        help='Explicit zero-update execution contract; causal matches physical training')
    parser.add_argument('--diagnostic-grounded-miss', action='store_true',
                        help='Frozen TRAIN128 comparison: also activate Recovery after a confirmed grounded miss')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--relaxed-scoring', action='store_true')
    parser.add_argument('--official-full-episode', action='store_true',
                        help='Use the historical independent512 full-episode task after verifying the checkpoint')
    parser.add_argument('--approach-ball-only', action='store_true',
                        help='Score the complete measured ball outcome independently of robot recovery')
    parser.add_argument('--early-arm-seconds', type=float, default=None,
                        help='Zero-update early-arm execution window for the Approach-only A0 panel')
    parser.add_argument('--relaxed-survival-diagnostic', action='store_true',
                        help='Continue successful lanes for post-success survival and uncensored strict scoring')
    parser.add_argument('--recovery-probe-companion', type=Path)
    parser.add_argument('--recovery-probe-shard', type=int, choices=range(8), default=0)
    parser.add_argument('--dynamimic-capture', action='store_true')
    parser.add_argument('--dynamimic-pulse-checkpoint', type=Path,
                        help='Apply a different Recovery head for 16 native control ticks in a training archive')
    timing = parser.add_mutually_exclusive_group()
    timing.add_argument('--dynamimic-pulse-start', type=int)
    timing.add_argument('--dynamimic-pulse-timing-from', type=Path,
                        help='Target pre-fall windows using the case-matched native training archive')
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument('--dynamimic-pulse-first-support', action='store_true',
                       help='Apply the candidate on the first control decision after measured touchdown')
    phase.add_argument('--dynamimic-pulse-air-preparation', action='store_true',
                        help='Start the training pulse at the first airborne state at least 50 ms after release')
    parser.add_argument('--recovery-cem-generations', type=int, choices=range(1,21))
    parser.add_argument('--recovery-ppo-panel', type=Path)
    parser.add_argument('--recovery-ppo-iterations', type=int, default=1)
    parser.add_argument('--recovery-ppo-resume', type=Path)
    parser.add_argument('--recovery-ground-prefix', action='store_true')
    parser.add_argument('--recovery-training-probes', type=Path)
    parser.add_argument('--recovery-full-tail', action='store_true',
                        help='Full-length fixed128 recovery baseline with strict and relaxed diagnostics')
    parser.add_argument('--g07-execution-root', type=Path)
    parser.add_argument('--precision-receipt', type=Path)
    parser.add_argument('--seed', type=int, default=882072)
    parser.add_argument('--sweep-audit-iterations', type=int, default=None,
                        help='Inherit source audit by default; only an identical explicit value is accepted')
    parser.add_argument('--physics-device', default='cuda:0')
    parser.add_argument('--record-physical-transitions', action='store_true',
                        help='Read-only pre-reset physical/executor archive; no model/actor updates')
    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument('--shutdown-diagnostic-ticks', type=int, default=None,
                        help='Lifecycle-only partial run on original 128 lanes; never writes scores/completed.json')
    panels = parser.add_mutually_exclusive_group()
    panels.add_argument('--dynamimic-validation', type=Path,
                        help='Score two Recovery candidates deterministically on the same 64 validation cases')
    parser.add_argument('--dynamimic-alternate', type=Path)
    parser.add_argument('--dynamimic-companion', type=Path)
    parser.add_argument('--dynamimic-exclude-training', type=Path)
    panels.add_argument('--research-fixed128', action='store_true',
                        help='Score the predeclared research cases held out from native training')
    panels.add_argument('--independent-panel-batch', type=int, choices=range(4),
                        help='Evaluate one stage-end batch of the declared independent 512 cases')
    panels.add_argument('--approach-quick64', action='store_true',
                        help='Score 64 registered training-side cases for block monitoring')
    panels.add_argument('--approach-search-spec', type=Path,
                        help='Score two to four bounded arm-bias variants in paired PhysX lanes')
    parser.add_argument('--approach-reset-probe', action='store_true',
                        help='Compare 64 continuation ticks with a scene restart on eight training cases')
    parser.add_argument('--approach-sensitivity', type=float, default=0.,
                        help='TRAIN64 only: seeded relative RMS perturbation of Approach parameters')
    parser.add_argument('--approach-prefix-checkpoint', type=Path,
                        help='Replay a frozen TRAIN64 ground prefix before applying the candidate Approach head')
    parser.add_argument('--recovery-prefix-checkpoint', type=Path,
                        help='Diagnostic only: use incumbent requests across all lanes before switching Recovery')
    parser.add_argument('--recovery-prefix-tick', type=int)
    parser.add_argument('--approach-prefix-tick', type=int, default=235)
    parser.add_argument('--approach-verify-snapshot', action='store_true',
                        help='Validate the resident snapshot against a 64-tick uninterrupted control replay')
    parser.add_argument('--approach-hybrid-spec', type=Path)
    parser.add_argument('--native-action-probe-spec',type=Path,default=None)
    parser.add_argument('--native-action-probe-branch',choices=('nominal_a','nominal_b','minus','plus','candidate'),default=None)
    args = parser.parse_args()
    if bool(args.recovery_prefix_checkpoint) != (args.recovery_prefix_tick is not None):
        parser.error('Recovery prefix requires both checkpoint and switch tick')
    if args.recovery_prefix_checkpoint and (not args.research_fixed128 or not args.recovery_full_tail
            or not 0 < args.recovery_prefix_tick < 800 or args.dynamimic_validation
            or args.approach_prefix_checkpoint or args.early_arm_seconds is not None):
        parser.error('Recovery prefix is restricted to a full-tail fixed128 diagnostic')
    if args.official_full_episode and (args.independent_panel_batch is None or not args.checkpoint
            or args.recovery_full_tail or args.relaxed_scoring or args.relaxed_survival_diagnostic
            or args.approach_ball_only or args.early_arm_seconds is not None):
        parser.error('Official full-episode scoring requires an independent batch and a frozen checkpoint')
    dyna_flags = (args.dynamimic_validation, args.dynamimic_alternate,
                  args.dynamimic_companion, args.dynamimic_exclude_training)
    if any(dyna_flags) and (not all(dyna_flags) or not args.recovery_full_tail or
            args.recovery_training_probes or args.recovery_ppo_panel or args.approach_prefix_checkpoint or
            args.approach_sensitivity or args.early_arm_seconds is not None or args.approach_hybrid_spec):
        parser.error('Paired Dyna validation requires a separate deterministic full-tail evaluation')
    if (args.dynamimic_pulse_air_preparation or args.dynamimic_pulse_first_support) and not args.dynamimic_pulse_timing_from:
        parser.error('Phase selection requires the case-matched timing archive')
    if bool(args.dynamimic_pulse_checkpoint) != (args.dynamimic_pulse_start is not None or args.dynamimic_pulse_timing_from is not None):
        parser.error('Recovery intervention requires a checkpoint and a timing source')
    if args.dynamimic_pulse_checkpoint and (not args.dynamimic_capture or (args.dynamimic_pulse_start is not None and not 0 <= args.dynamimic_pulse_start <= 784)):
        parser.error('Recovery intervention requires native training capture and a complete 16-tick interval')
    if args.recovery_cem_generations and (not args.recovery_ppo_panel or args.dynamimic_capture or args.recovery_ppo_resume):
        parser.error('Native Recovery CEM requires a separate training panel and fresh search')
    if args.dynamimic_capture and not args.recovery_ppo_panel:
        parser.error('Dyna capture requires a mixed Recovery training panel')
    if args.recovery_ppo_panel and (args.recovery_training_probes or args.research_fixed128 or
            args.independent_panel_batch is not None or args.approach_quick64 or args.approach_hybrid_spec or
            args.record_physical_transitions or not args.recovery_full_tail):
        parser.error('Recovery PPO requires separate full-tail training cases')
    if bool(args.recovery_training_probes or args.recovery_ppo_panel) != bool(args.recovery_probe_companion):
        parser.error('Recovery training pairs require a companion Approach')
    if args.recovery_training_probes and (not args.recovery_full_tail or args.research_fixed128
            or args.independent_panel_batch is not None or args.approach_quick64
            or args.approach_hybrid_spec or args.record_physical_transitions):
        parser.error('Native recovery teachers require a separate full-tail training rollout')
    if args.recovery_full_tail:
        if (not args.checkpoint or not (args.research_fixed128 or args.recovery_training_probes or args.recovery_ppo_panel or args.dynamimic_validation or args.independent_panel_batch is not None) or args.approach_ball_only
                or args.native_action_probe_spec or args.shutdown_diagnostic_ticks
                or args.diagnostic_grounded_miss or args.approach_search_spec):
            parser.error('Recovery baseline requires a frozen checkpoint and the fixed128 panel')
        args.relaxed_scoring = args.relaxed_survival_diagnostic = True
    if args.approach_hybrid_spec and (not args.checkpoint or not args.approach_ball_only
            or not args.headless or args.approach_search_spec or args.research_fixed128
            or args.approach_quick64 or args.independent_panel_batch is not None
            or args.native_action_probe_spec or args.record_physical_transitions
            or args.shutdown_diagnostic_ticks or args.diagnostic_grounded_miss):
        parser.error('Hybrid boundary requires its own 128-lane frozen native layout')
    if args.approach_search_spec and (not args.checkpoint or not args.approach_ball_only or not args.headless
            or args.native_action_probe_spec or args.record_physical_transitions or args.shutdown_diagnostic_ticks
            or args.diagnostic_grounded_miss):
        parser.error('Native Approach search requires the frozen headless ball-only protocol')
    if not 0. <= args.approach_sensitivity <= 1e-3 or (args.approach_sensitivity and not args.approach_quick64):
        parser.error('Use a relative perturbation in [0, 0.001] on TRAIN64 only')
    if args.approach_prefix_checkpoint and (not args.approach_quick64 or args.approach_reset_probe
                                            or args.approach_sensitivity or not 0 < args.approach_prefix_tick < 800):
        parser.error('A prefix requires ordinary frozen TRAIN64 and a positive switch tick')
    if args.approach_verify_snapshot and not args.approach_prefix_checkpoint:
        parser.error('Snapshot verification requires a frozen prefix actor')
    if args.approach_reset_probe and not args.approach_quick64:
        parser.error('The restart probe requires the ball-only training-case panel')
    if args.approach_quick64 and (not args.approach_ball_only or not args.checkpoint
                                 or args.native_action_probe_spec or args.record_physical_transitions
                                 or args.shutdown_diagnostic_ticks is not None or args.diagnostic_grounded_miss):
        parser.error('Approach quick-check requires a frozen ball-only checkpoint')
    if args.early_arm_seconds is not None and not args.approach_ball_only:
        parser.error('Early-arm override requires the ball-only A0 evaluation protocol')
    if args.approach_ball_only and (args.relaxed_scoring or args.relaxed_survival_diagnostic):
        parser.error('Choose a single evaluation termination protocol')
    if (args.independent_panel_batch is not None or args.research_fixed128) and (args.native_action_probe_spec or args.record_physical_transitions
            or args.shutdown_diagnostic_ticks is not None):
        parser.error('Independent panel evaluation requires a complete score-only rollout')
    if args.diagnostic_grounded_miss and (not args.checkpoint or args.independent_panel_batch is not None
            or args.research_fixed128 or args.native_action_probe_spec or args.record_physical_transitions
            or args.shutdown_diagnostic_ticks is not None):
        parser.error('The grounded-miss diagnostic uses a learned checkpoint and a full TRAIN128 score')
    if bool(args.native_action_probe_spec)!=bool(args.native_action_probe_branch):
        parser.error('Probe spec and branch are required together')
    if args.native_action_probe_spec:
        if not args.physics_device.startswith('cuda') or not args.headless:
            parser.error('Paired native probes require headless GPU PhysX on the original128 lanes')
        if args.shutdown_diagnostic_ticks is not None or args.record_physical_transitions:
            parser.error('Probe is a separate diagnostic, not evaluation/archive mode')
        from quadsmash.rl.supermimic_native_pair_capture_v1 import validate_spec
        probe_spec=validate_spec(json.loads(args.native_action_probe_spec.read_text()))
        args.shutdown_diagnostic_ticks=probe_spec.get('terminal_tick_cap', probe_spec['tick']+1)

    if args.shutdown_diagnostic_ticks is not None and not 1 <= args.shutdown_diagnostic_ticks < 800:
        parser.error('shutdown diagnostic ticks must be in [1,799]')
    if args.checkpoint and args.init_policy_recovery_gate!='legacy':
        parser.error('Learned checkpoints must use their recorded execution contract')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    startup = dict(argv=sys.argv, cwd=os.getcwd(), script=str(Path(__file__).resolve()),
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                   dirty_diff=subprocess.check_output(['git','diff','--no-ext-diff'],cwd=ROOT,text=True),
                   executable=sys.executable, pid=os.getpid(),
                   retained_exception_audit=os.environ.get('QUADSMASH_AUDIT_RETAINED_EXCEPTIONS') == '1',
                   shutdown_diagnostic_ticks=args.shutdown_diagnostic_ticks)
    write_json(args.output/'startup.json', startup)
    from quadsmash.rl.supermimic_run_lifecycle import stage as lifecycle_stage, execute_scoped_native
    # Keep the descriptor valid through interpreter/native finalization. The OS
    # closes it at process exit; do not disable faulthandler after app.close().
    fault_fd = os.open(args.output/'native_fault.log', os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    faulthandler.enable(file=fault_fd, all_threads=True)
    lifecycle_stage(args.output, 'BEFORE_APP')
    app = None
    stop_request = {'signal': None}
    try:
        app = AppLauncher(args, fast_shutdown=False).app
        faulthandler.enable(file=fault_fd, all_threads=True)
        lifecycle_stage(args.output, 'AFTER_APP')
        def request_stop(signum, frame):
            stop_request['signal'] = signal.Signals(signum).name
        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
        # No native environment, wrapper, actor or tensor escapes this scope.
        execute_scoped_native(args.output, lambda: _run_native(args, stop_request), app)
    except BaseException as error:
        if not (args.output/'failure.json').exists():
            trace = traceback.format_exc()
            (args.output/'failure_traceback.txt').write_text(trace)
            write_json(args.output/'failure.json', dict(
                stage='APPLICATION_OR_SCOPED_CLEANUP', type=type(error).__name__, message=str(error)))
            print(trace, file=sys.stderr, flush=True)
        raise


def _run_native(args, stop_request):
    from quadsmash.rl.supermimic_run_lifecycle import stage as lifecycle_stage, close_resource
    env = None
    tick = -1
    ticks_completed = 0
    mapping = None
    stage = 'AFTER_APP'
    primary_failed = False
    try:
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaacsim.core.simulation_manager import SimulationManager
        from quadsmash.rl.supermimic_v1 import load_ppo_initialized_actor
        from quadsmash.rl.supermimic_task_contract import (
            resolve_task_contract, make_task_cfg, require_resume_contract,
            zero_panel_reset_noise, validate_sweep_contract)
        from quadsmash.rl.supermimic_terminal_scoring import (
            score_terminal_episode, relaxed_scoring_protocol, approach_ball_scoring_protocol, STRICT_SCORING)
        from quadsmash.rl.supermimic_timebase_v2 import full_range_targets
        from quadsmash.tasks.stand.supermimic_env_v1 import SuperMimicEnvV1
        from quadsmash.tasks.stand.full_racket_eval_env_v4 import FrozenPanelEnv
        from quadsmash.control.g07_shared_target_residual_v1 import make_g07_residual_controller
        from quadsmash.control.full_racket_residual_v4 import FullRacketResidualControllerV4

        class EvaluationEnv(SuperMimicEnvV1, FrozenPanelEnv):
            """Use the historical panel's ground-reset procedure without noise."""

        class RecoveryEvaluationEnv(EvaluationEnv):
            diagnostic = None

            def _consume_jump_frame(self, frame):
                if self.diagnostic is None or frame <= self._jump_physics_frame or frame == 0:
                    return super()._consume_jump_frame(frame)
                previous_time = self.jump_clock.clone()
                previous_run = self.contact_recovery.stable_run.clone()
                super()._consume_jump_frame(frame)
                self.diagnostic.observe(self, previous_time, previous_run)

            def _reset_idx(self, ids):
                if self.diagnostic is None:
                    return super()._reset_idx(ids)
                evidence = self.diagnostic.take(self, ids)
                offset = len(self.rally_records)
                super()._reset_idx(ids)
                for row in self.rally_records[offset:]:
                    row['recovery_baseline_diagnostic'] = evidence[row['env_id'], row['episode_index']]

        torch.manual_seed(args.seed)
        trained = None
        if args.checkpoint:
            trained = torch.load(args.checkpoint, map_location='cpu', weights_only=False,
                                 mmap=zipfile.is_zipfile(args.checkpoint))
            if trained.get('algorithm') != 'supermimic_v1':
                raise ValueError('Expected SuperMimic checkpoint')
            init_meta = trained['metadata']['init_policy_checkpoint']
            source_path = Path(init_meta['path'])
            execution_root = args.g07_execution_root or init_meta.get('g07_execution_root')
            if hashlib.sha256(source_path.read_bytes()).hexdigest() != init_meta['sha256']:
                raise ValueError('Initialization source changed')
        else:
            source_path, execution_root = args.init_policy_checkpoint, args.g07_execution_root
        source = torch.load(source_path, map_location='cpu', weights_only=False)
        contract, protocol, cases, receipt = resolve_task_contract(
            source, source_path, execution_root=execution_root,
            precision_receipt=args.precision_receipt, root=execution_root)
        scoring_protocol = dict(protocol, _root=str(execution_root or ROOT))
        if trained is not None:
            stored = trained.get('metadata', {}).get('task_contract', {})
            if 'relaxed_scoring' in stored:
                contract = dict(contract, relaxed_scoring=relaxed_scoring_protocol())
            if 'approach_ball_scoring' in stored:
                contract = dict(contract, approach_ball_scoring=approach_ball_scoring_protocol())
                if args.early_arm_seconds is not None:
                    raise ValueError('A trained ball-only checkpoint retains its recorded gate')
            require_resume_contract(trained, contract)
        if args.official_full_episode:
            # Validate the source training contract above, then run the original
            # strict task through native episode termination on the declared panel.
            contract.pop('approach_ball_scoring', None)
            contract.pop('relaxed_scoring', None)
        if args.recovery_full_tail:
            if contract['contact_flight_recovery'] is None:
                raise ValueError('Recovery baseline requires native contact-flight association')
            # Keep the checkpoint's identity check above. Disable only the
            # diagnostic rollout's ball-side early termination.
            contract.pop('approach_ball_scoring', None)
        if args.relaxed_scoring:
            contract = dict(contract, relaxed_scoring=relaxed_scoring_protocol())
        if args.approach_ball_only:
            contract.pop('relaxed_scoring', None)
            contract['approach_ball_scoring'] = approach_ball_scoring_protocol()
        scoring_protocol['approach_ball_scoring'] = contract.get('approach_ball_scoring')
        scoring_protocol['relaxed_scoring'] = contract.get('relaxed_scoring')
        if args.relaxed_survival_diagnostic and scoring_protocol['relaxed_scoring'] is None:
            raise ValueError('Post-success survival requires relaxed scoring')
        capture_contract = contract
        search_spec = None
        hybrid = None
        if args.approach_hybrid_spec:
            from quadsmash.rl.supermimic_approach_data_v1 import approach_quick_panel
            from quadsmash.rl.supermimic_hybrid_native import (
                FIRST_CAPTURE_TICK, native_layout, native_state, pack_tapes, perturbation_envelope)

            hybrid = json.loads(args.approach_hybrid_spec.read_text())
            _, quick_identity = approach_quick_panel(cases)
            cases = native_layout(cases, hybrid, trained['config']['wm_holdout_lanes'])
            capture_contract = dict(contract, approach_quick_panel=quick_identity)
            hybrid_frames = []

        if args.approach_search_spec:
            from quadsmash.rl.supermimic_native_approach_search import candidate_layout, validate_offsets
            search_spec = json.loads(args.approach_search_spec.read_text())
            offsets = validate_offsets(search_spec['offsets'], maximum=4)
            cases = candidate_layout(cases, search_spec['case_indices'], offsets)
            capture_contract = dict(contract, approach_native_search=dict(
                case_indices=search_spec['case_indices'], offsets=offsets))
        if args.dynamimic_validation:
            from scripts.dynamimic_recovery_validation import panel_layout
            cases = panel_layout(args.dynamimic_validation, args.dynamimic_exclude_training,
                                 contract['protocol_sha256'], args.seed)
            capture_contract = dict(contract, dynamimic_paired_validation=True)
        if args.independent_panel_batch is not None:
            from quadsmash.rl.supermimic_independent_panel import panel_batch
            cases, panel_identity = panel_batch(args.independent_panel_batch, contract['protocol_sha256'], args.seed)
            capture_contract = dict(contract, independent_evaluation=panel_identity)
        if args.research_fixed128:
            from quadsmash.rl.supermimic_independent_panel import fixed128_cases
            cases, panel_identity = fixed128_cases(contract['protocol_sha256'], args.seed)
            capture_contract = dict(contract, research_evaluation=panel_identity)
        if args.approach_quick64:
            from quadsmash.rl.supermimic_approach_data_v1 import approach_quick_panel
            cases, panel_identity = approach_quick_panel(cases)
            capture_contract = dict(contract, approach_quick_panel=panel_identity)
            if args.approach_reset_probe:
                cases = cases[:8]
        if args.native_action_probe_spec:
            from scripts.declare_supermimic_contact_training_cases import resolve_probe_cases
            spec = json.loads(args.native_action_probe_spec.read_text())
            cases, training_digest = resolve_probe_cases(spec, cases)
            if training_digest:
                capture_contract = dict(contract, contact_training_panel_sha256=training_digest,
                                        contact_training_only=True)
        write_json(args.output/'task_contract.json', capture_contract)
        teachers = None
        if args.recovery_training_probes:
            from scripts.recovery_native_teachers import NativeTeachers, training_layout
            cases = training_layout(args.recovery_training_probes, contract['protocol_sha256'], args.seed, args.recovery_probe_shard)
            teachers = NativeTeachers(args.device)
        if args.recovery_ppo_panel:
            from scripts.recovery_native_teachers import training_layout
            cases = [case for shard in range(8) for case in training_layout(
                args.recovery_ppo_panel,contract['protocol_sha256'],args.seed,shard)[:32]]
        if args.recovery_cem_generations:
            from scripts.native_recovery_cem import panel_layout, search
            cem_bank = cases
            cases = panel_layout(cem_bank, 0)
        lanes = len(cases)
        cfg = make_task_cfg(lanes, contract, protocol, receipt, execution_root=execution_root)
        cfg.seed, cfg.sim.device = args.seed, args.physics_device
        validate_sweep_contract(cfg, contract, args.sweep_audit_iterations)
        noise = zero_panel_reset_noise(len(cases), args.seed)
        stage = 'BEFORE_ENV'
        env_type = RecoveryEvaluationEnv if args.recovery_full_tail and teachers is None else EvaluationEnv
        if args.recovery_ppo_panel:
            from scripts.native_recovery_ppo import reward_state
            class RecoveryPPOEnv(EvaluationEnv):
                def _observe_physics_frame(self, frame):
                    fresh = frame > self._observation_physics_frame and frame > 0
                    super()._observe_physics_frame(frame)
                    if fresh and getattr(self, 'dyna_foot_impulse', None) is not None:
                        self.dyna_foot_impulse.add_(self.feet_contact.data.net_forces_w, alpha=self.physics_dt)
                        self.dyna_impulse_frames += 1

                def _get_dones(self):
                    result = super()._get_dones()
                    self.recovery_ppo_boundary = reward_state(self)
                    if args.dynamimic_capture:
                        from scripts.capture_dynamimic_recovery import native_boundary
                        self.dyna_boundary = native_boundary(self)
                    return result
            env_type = RecoveryPPOEnv
        env = env_type(cfg, cases=cases, reset_noise=noise,
            residual_policy_checkpoint=source, g07_execution_root=execution_root,
            contact_flight_recovery=contract['contact_flight_recovery'],
            relaxed_protocol=contract.get('relaxed_scoring'),
            approach_ball_protocol=contract.get('approach_ball_scoring'),
            relaxed_survival_diagnostic=args.relaxed_survival_diagnostic)
        stage = 'AFTER_ENV'
        wrapper = RslRlVecEnvWrapper(env, clip_actions=None)
        obs, _ = wrapper.reset()
        stage = 'AFTER_RESET'
        write_json(args.output/'initial_states.json', env.initial_states)
        if args.recovery_full_tail and teachers is None and not args.recovery_ppo_panel:
            from quadsmash.tasks.stand.jump_env import PROTOCOL
            from quadsmash.tasks.stand.recovery_baseline_diagnostics import RecoveryBaselineDiagnostics
            env.diagnostic = RecoveryBaselineDiagnostics(env, PROTOCOL['recovery'])
        actor, init_meta = load_ppo_initialized_actor(source_path, env.device)
        if trained is not None:
            actor.load_state_dict(trained['actor'], strict=True)
        actor.eval()
        from quadsmash.rl.supermimic_policy_execution_v1 import resolve_contract,policy_forward,recovery_gate,contract as execution_contract,POSTHIT,POSTHIT_OR_GROUNDED_MISS
        from quadsmash.rl.supermimic_physical_loop_loading_v1 import tensor_digest
        clean_actor_sha = tensor_digest(actor.state_dict().items())
        if args.approach_sensitivity:
            generator = torch.Generator().manual_seed(args.seed + 1)
            with torch.no_grad():
                for parameter in actor.actor.approach.parameters():
                    direction = torch.randn(parameter.shape, generator=generator).to(parameter)
                    magnitude = parameter.norm() * args.approach_sensitivity
                    parameter.add_(direction * (magnitude / direction.norm().clamp_min(1e-12)))
        actor_state_before=tensor_digest(actor.state_dict().items())
        policy_execution,execution_provenance=resolve_contract(trained)
        if args.recovery_full_tail and policy_execution['recovery_gate'] != POSTHIT:
            raise ValueError('Recovery baseline expects the causal contact-and-release gate')
        prefix_actor = None
        prefix_path = args.recovery_prefix_checkpoint or args.approach_prefix_checkpoint
        prefix_tick = args.recovery_prefix_tick if args.recovery_prefix_checkpoint else args.approach_prefix_tick
        if prefix_path:
            from quadsmash.rl.supermimic_physical_loop_loading_v1 import APPROACH_NAMES, RECOVERY_NAMES
            allowed = RECOVERY_NAMES if args.recovery_prefix_checkpoint else APPROACH_NAMES
            prefix = torch.load(prefix_path, map_location='cpu', mmap=True, weights_only=False)
            require_resume_contract(prefix, stored if args.recovery_prefix_checkpoint else contract)
            if (resolve_contract(prefix)[0] != policy_execution or prefix['actor'].keys() != trained['actor'].keys()
                    or any(not torch.equal(value, trained['actor'][name])
                           for name, value in prefix['actor'].items() if name not in allowed)):
                raise ValueError('Prefix checkpoint differs outside the selected head or execution contract')
            prefix_actor, _ = load_ppo_initialized_actor(source_path, env.device)
            prefix_actor.load_state_dict(prefix['actor'], strict=True)
            prefix_actor.eval()
            prefix_actor.configure_approach_gate(policy_execution.get('arm_preparation_seconds'))
            prefix_actor_sha = tensor_digest(prefix_actor.state_dict().items())
        if trained is None and args.init_policy_recovery_gate=='causal':
            policy_execution=execution_contract(POSTHIT);execution_provenance='explicit frozen initialization under training causal gate'
        if args.early_arm_seconds is not None:
            policy_execution = execution_contract(POSTHIT, arm_preparation_seconds=args.early_arm_seconds)
            execution_provenance = 'explicit early-arm zero-update baseline'
        if policy_execution.get('arm_preparation_seconds') is not None:
            actor.configure_approach_gate(policy_execution['arm_preparation_seconds'])
        dyna_policies = None
        if args.dynamimic_validation:
            from scripts.dynamimic_recovery_validation import policies, paired_requests
            dyna_policies = policies(args.checkpoint, args.dynamimic_alternate, args.dynamimic_companion,
                                     lambda: load_ppo_initialized_actor(source_path, env.device)[0])
        companion = None
        if teachers is not None or args.recovery_ppo_panel:
            from scripts.run_supermimic_recovery_baseline import verify
            verify([args.checkpoint, args.recovery_probe_companion], synchronize_recovery=bool(args.recovery_ppo_panel))
            companion, _ = load_ppo_initialized_actor(source_path, env.device)
            companion.load_state_dict(torch.load(args.recovery_probe_companion, map_location='cpu',
                                                  weights_only=False)['actor'])
            companion.eval()
            if policy_execution.get('arm_preparation_seconds') is not None:
                companion.configure_approach_gate(policy_execution['arm_preparation_seconds'])
        if args.recovery_ppo_panel:
            if args.recovery_cem_generations:
                from quadsmash.rl.supermimic_actuator_identification import actuator_contract
                drive = actuator_contract(env, contract)
                if drive['lane_profile'] != drive['lane_profile'][:16] * 8:
                    raise ValueError('CEM candidate and nominal actuator profiles differ')
                write_json(args.output/'actuator_contract.json', drive)
                search(args, env, wrapper, obs, actor, companion, cem_bank, scoring_protocol, stop_request)
                return
            if args.dynamimic_capture:
                from scripts.capture_dynamimic_recovery import collect
                collect(args, env, wrapper, obs, actor, companion, trained, cases, stop_request)
                return
            from scripts.native_recovery_ppo import train
            train(args, env, wrapper, obs, actor, companion, trained, cases, stop_request)
            return
        search_bias = None
        if search_spec is not None:
            if policy_execution['recovery_gate'] != POSTHIT:
                raise ValueError('Native arm search requires the declared causal Approach controller')
            search_bias = torch.zeros((lanes, 18), device=env.device)
            offsets = search_bias.new_tensor(search_spec['offsets'])
            search_bias[:, -6:] = offsets.repeat_interleave(32, dim=0)
            search_bias = search_bias * actor.actor.approach_scale
        if hybrid is not None:
            search_bias = torch.zeros((lanes, 18), device=env.device)
            search_bias[64:, -6:] = search_bias.new_tensor(hybrid['offsets']).repeat_interleave(16, dim=0)
            search_bias = search_bias * actor.actor.approach_scale
        if args.diagnostic_grounded_miss:
            if policy_execution != execution_contract(POSTHIT):
                raise ValueError('Grounded-miss comparison requires a post-hit checkpoint')
            policy_execution = execution_contract(POSTHIT_OR_GROUNDED_MISS)
            execution_provenance = 'frozen TRAIN128 grounded-miss gate comparison, same checkpoint weights'
        write_json(args.output/'policy_execution.json',dict(contract=policy_execution,provenance=execution_provenance))
        if args.record_physical_transitions:
            from quadsmash.rl.supermimic_physical_capture_v1 import PhysicalTransitionArchive
            if env._sm_residual_controller is None:
                raise ValueError('Physical capture requires the initialized residual execution contract')
            compress_archive = os.environ.get('QUADSMASH_ZSTD_ARCHIVE') == '1'
            env.physical_transition_archive = PhysicalTransitionArchive(
                args.output/'physical_transitions', metadata=dict(
                    source_checkpoint=str(source_path), source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                    task_contract=contract,policy_execution=policy_execution,actor_state_sha256=actor_state_before,policy_id='frozen-evaluation',body_names=list(env.robot.body_names),
                    control_dt_s=float(env.step_dt), num_envs=env.num_envs,
                    seed=args.seed, runtime_commit=subprocess.check_output(
                        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    wm_holdout_lanes=trained.get('config', {}).get('wm_holdout_lanes') if trained is not None else None),
                compact_plan_observations=compress_archive, compress=compress_archive)
            env.physical_transition_archive.attach_task_sequence(env)
        reference = None
        if trained is None:
            reference = (make_g07_residual_controller(source,lanes,env.device,execution_root)
                         if source['schema']=='g07-shared-target-residual-ppo-v1'
                         else FullRacketResidualControllerV4(source,lanes,env.device))
            reference.reset()
        physics = dict(device=str(SimulationManager.get_physics_sim_device()),
                       gpu_dynamics=bool(SimulationManager.is_gpu_dynamics_enabled()),
                       origin_spacing_m=cfg.scene.env_spacing,
                       actual_origins_m=env.scene.env_origins.cpu().tolist(),
                       ground_size_m=list(cfg.ground_size_m), sweep_audit_iterations=cfg.sweep_audit_iterations)
        write_json(args.output/'physics.json', physics)
        from quadsmash.rl.supermimic_actuator_identification import actuator_contract
        write_json(args.output/'actuator_contract.json', actuator_contract(env, contract))
        first, cursor, numerical = {}, 0, 0
        head_counts = torch.zeros(2, dtype=torch.long, device=env.device)
        if args.diagnostic_grounded_miss:
            gate_first_tick = torch.full((lanes,), -1, device=env.device, dtype=torch.long)
            gate_ticks = torch.zeros_like(gate_first_tick)
            first_active = torch.ones_like(gate_ticks, dtype=torch.bool)
        mapping = dict(max_final_target_error_rad=0., tolerance_rad=2e-6,
                       phase_samples={str(i):0 for i in range(5)}, phase_max_error={str(i):0. for i in range(5)},
                       checked_native_ticks=0, scope='Independent old controller versus new actor+executor on the SAME live observations; not a separate old physics rollout')
        if policy_execution['recovery_gate']==POSTHIT:
            mapping['scope']='Independent original executor and new executor with identical causal-gated requests; not legacy policy equivalence'
        probe_capture=None
        candidate_actor = None
        if args.native_action_probe_spec:
            from quadsmash.rl.supermimic_native_pair_capture_v1 import NativePairCapture
            if policy_execution['recovery_gate']!=POSTHIT:
                raise ValueError('Native training probe requires the causal execution contract')
            if args.native_action_probe_branch == 'candidate':
                # Reuse the nominal recorder for a request with no injected
                # pulse. The evaluator supplies the closed-loop policy suffix.
                probe_capture = NativePairCapture(spec, 'nominal_a')
                probe_capture.branch = 'candidate'
            else:
                probe_capture = NativePairCapture(spec, args.native_action_probe_branch)
            if spec.get('terminal_tick_cap') and scoring_protocol['approach_ball_scoring'] is None:
                raise ValueError('Native probe terminal labels require ball-only scoring')
            if probe_capture.branch == 'candidate':
                from quadsmash.rl.supermimic_physical_loop_loading_v1 import APPROACH_NAMES
                with Path(spec['candidate_checkpoint']).open('rb') as stream:
                    if hashlib.file_digest(stream, 'sha256').hexdigest() != spec['candidate_checkpoint_sha256']:
                        raise ValueError('Candidate checkpoint changed')
                    candidate_path = Path(stream.name)
                challenger = torch.load(candidate_path, map_location='cpu', mmap=True, weights_only=False)
                require_resume_contract(challenger, contract)
                if resolve_contract(challenger)[0] != policy_execution:
                    raise ValueError('Candidate execution contract differs')
                base = trained['actor']
                state = challenger['actor']
                if state.keys() != base.keys() or any(not torch.equal(state[k], base[k])
                                                      for k in base if k not in APPROACH_NAMES):
                    raise ValueError('Only Approach weights may differ in the candidate branch')
                candidate_actor, _ = load_ppo_initialized_actor(source_path, env.device)
                candidate_actor.load_state_dict(state, strict=True)
                candidate_actor.eval()
                if policy_execution.get('arm_preparation_seconds') is not None:
                    candidate_actor.configure_approach_gate(policy_execution['arm_preparation_seconds'])
                probe_capture.candidate_actor_sha256 = tensor_digest(candidate_actor.state_dict().items())
        start = time.perf_counter()
        step_wall_seconds = 0.
        phase_timing = None
        if scoring_protocol['approach_ball_scoring'] is not None:
            from quadsmash.rl.supermimic_native_phase_timing import NativePhaseTiming
            phase_timing = NativePhaseTiming(env)
        stage = 'EVALUATING'
        if args.approach_reset_probe:
            from quadsmash.rl.supermimic_native_approach_reset import (
                NativeApproachRestart, observe_restart_state, restart_comparison)
            restart = None
            replay_actions, continuation = [], []
        with torch.inference_mode():
            for tick in range(args.shutdown_diagnostic_ticks or 800):
                if stop_request['signal'] is not None:
                    raise InterruptedError(f"Evaluation stopped at control boundary: {stop_request['signal']}")
                if args.approach_verify_snapshot and tick == args.approach_prefix_tick:
                    from quadsmash.rl.supermimic_native_approach_reset import verify_resident_snapshot

                    def prefix_request(observations):
                        return policy_forward(prefix_actor, observations['policy'], env.rally,
                                              mode=policy_execution['recovery_gate'], sample=False)[0].clamp(-1., 1.)

                    obs, restart_report = verify_resident_snapshot(env, wrapper, obs, prefix_request)
                    write_json(args.output/'approach_snapshot_verification.json', restart_report)
                    if not restart_report['passed']:
                        raise RuntimeError('Resident PhysX snapshot failed continuation verification')
                before = obs['policy'] if reference is None and probe_capture is None else obs['policy'].clone()
                old_target = None
                active_actor = prefix_actor if prefix_actor is not None and tick < prefix_tick else actor
                if dyna_policies is not None:
                    action = paired_requests(dyna_policies, before, env.rally.had_contact & env.rally.released,
                                             env.rally.release_time)
                elif search_bias is None:
                    action = policy_forward(active_actor,before,env.rally,mode=policy_execution['recovery_gate'],sample=False)[0].clamp(-1.,1.)
                else:
                    latch = recovery_gate(env.rally, before)
                    offset = search_bias
                    if hybrid is not None:
                        offset = offset * perturbation_envelope(tick) * (~env.rally.had_contact[:, None])
                    action = (actor._gated_distribution(before, latch).loc
                              + actor.approach_action_gate(before) * offset).clamp(-1., 1.)
                if teachers is not None:
                    companion_action = policy_forward(companion, before, env.rally, mode=POSTHIT, sample=False)[0]
                    action[1::2] = companion_action[1::2].clamp(-1., 1.)
                    action = teachers.apply(tick, before, action, env.rally, env.episode_index)
                if args.approach_reset_probe and tick == 235:
                    restart = NativeApproachRestart.capture(env)
                if env.approach_ball_termination is not None:
                    env.record_approach_gate(active_actor.approach_action_gate(before))
                if probe_capture is not None:
                    action = probe_capture.before(tick,env,before,action)
                    if candidate_actor is not None and tick >= spec['action_tick']:
                        alternate = policy_forward(candidate_actor, before, env.rally,
                                                   mode=policy_execution['recovery_gate'], sample=False)[0].clamp(-1., 1.)
                        probe_lanes = spec['lanes']
                        action = action.clone()
                        action[probe_lanes] = alternate[probe_lanes]
                        if tick == spec['action_tick']:
                            probe_capture.request_after = action.detach().cpu().clone()
                if policy_execution['recovery_gate'] == POSTHIT_OR_GROUNDED_MISS:
                    active_gate = recovery_gate(env.rally,before,mode=POSTHIT_OR_GROUNDED_MISS)
                    head_counts[0].add_(active_gate.sum())
                    if args.diagnostic_grounded_miss:
                        extra = first_active & active_gate & ~env.rally.had_contact
                        gate_first_tick.masked_fill_(extra & (gate_first_tick < 0), tick)
                        gate_ticks.add_(extra)
                else:
                    head_counts[0].add_((env.rally.had_contact & env.rally.released).sum())
                head_counts[1].add_((actor.approach_gate(before) > .25).sum())
                if reference is not None:
                    if policy_execution['recovery_gate']==POSTHIT:
                        old_final=reference.apply_residual(obs,action,env._sm_residual_default,env._sm_residual_limits)
                    else:
                        old_final = reference.action(obs, env._sm_residual_default, env._sm_residual_limits)
                    old_target = full_range_targets(old_final, env._sm_residual_limits).clone()
                if not torch.isfinite(action).all():
                    raise FloatingPointError('Nonfinite evaluation action')
                frame = None
                if hybrid is not None and tick >= FIRST_CAPTURE_TICK:
                    frame = native_state(env)
                    frame['observation'] = before[64:].clone()
                    frame['request'] = action[64:].clone()
                    frame['gate'] = actor.approach_action_gate(before)[64:].amax(-1)
                step_start = time.perf_counter()
                obs, _, done, _ = (wrapper.step(action) if phase_timing is None else
                                   phase_timing.step(wrapper.step, action, before, tick))
                if frame is not None:
                    frame['targets'] = env._sm_joint_targets[64:].clone()
                    hybrid_frames.append(frame)
                if args.approach_reset_probe and restart is not None:
                    replay_actions.append(action.detach().clone())
                    continuation.append(observe_restart_state(env, obs))
                    if len(continuation) == 64:
                        restart.restore(env)
                        replay = []
                        for command in replay_actions:
                            obs, _, _, _ = wrapper.step(command)
                            replay.append(observe_restart_state(env, obs))
                        outcome = restart_comparison(continuation, replay)
                        outcome.update(start_control_tick=235, cases=[case['case_id'] for case in cases])
                        write_json(args.output/'approach_restart_probe.json', outcome)
                        return
                if probe_capture is not None:probe_capture.after(tick,env,done)
                ticks_completed += 1
                if not torch.isfinite(obs['policy']).all():
                    raise FloatingPointError('Nonfinite evaluation observation')
                step_wall_seconds += time.perf_counter() - step_start
                if old_target is not None:
                    errors=(env._sm_joint_targets-old_target).abs().amax(1)
                    maximum=float(errors.max())
                    mapping['max_final_target_error_rad']=max(mapping['max_final_target_error_rad'],maximum)
                    phase=before[:,770:775].argmax(1)
                    for i in range(5):
                        mask=phase==i
                        mapping['phase_samples'][str(i)]+=int(mask.sum())
                        if mask.any():
                            mapping['phase_max_error'][str(i)]=max(mapping['phase_max_error'][str(i)],float(errors[mask].max()))
                    mapping['checked_native_ticks']+=1
                    if maximum > mapping['tolerance_rad']:
                        write_json(args.output/'command_consistency.json',mapping)
                        raise RuntimeError('Independent old/new final target mismatch')
                    reset_ids = torch.where(done.bool())[0]
                    if reset_ids.numel():
                        reference.reset(reset_ids)
                for record in env.rally_records[cursor:]:
                    lane=int(record['env_id'])
                    if lane not in first:
                        measured=score_terminal_episode(record, scoring_protocol)
                        if measured['episode_index'] != 0:
                            raise RuntimeError('Missing first episode; refusing later episode substitution')
                        if args.recovery_full_tail:
                            if teachers is None:
                                measured['recovery_baseline_diagnostic'] = record['recovery_baseline_diagnostic']
                            measured['strict_run_s'] = record['contact_flight_recovery']['recovery_s']
                            measured['strict_physical_recovery'] = bool(
                                record['contact_flight_recovery']['success'] and not record['jump']['terminated'])
                            measured['robot_net_fault'] = bool(record['robot_net_fault'])
                        first[lane]=measured
                        if probe_capture is not None:
                            probe_capture.capture_terminal(measured)
                        if args.diagnostic_grounded_miss:
                            first_active[lane] = False
                        with (args.output/'episodes.jsonl').open('a') as stream:
                            stream.write(json.dumps(record)+'\n')
                    else:
                        with (args.output/'later_episodes.jsonl').open('a') as stream:
                            stream.write(json.dumps(record)+'\n')
                cursor=len(env.rally_records)
                if (tick+1)%100==0:
                    progress=dict(control_ticks=tick+1,first_episodes=len(first),elapsed_s=time.perf_counter()-start,
                                  first_episode_numerical_audit_count=sum(r['numerical_failures'] for r in first.values()),
                                  active_episode_sweep_unresolved=int(env.sweep_unresolved.sum()),
                                  active_episode_sweep_misses=int(env.sweep_misses.sum()))
                    returns=[r['native_episode_return'] for r in first.values()]
                    discounted=[r['native_episode_discounted_return'] for r in first.values()]
                    progress.update(native_episode_return_mean=sum(returns)/len(returns) if returns else None,
                        native_episode_discounted_return_mean=sum(discounted)/len(discounted) if discounted else None,
                        native_return_gamma=env._sm_native_returns.gamma)
                    write_json(args.output/'progress.json',progress)
                    write_json(args.output/'command_consistency.json',mapping)
                    print('SUPERMIMIC_EVAL_PROGRESS='+json.dumps(progress),flush=True)
                if probe_capture is not None and spec.get('terminal_tick_cap'):
                    if tick >= spec['tick'] and len(probe_capture.terminal_records) == len(spec['lanes']):
                        break
                elif scoring_protocol['approach_ball_scoring'] is not None and len(first) == lanes:
                    break
        if env.physical_transition_archive is not None:
            env.physical_transition_archive.flush()
        if probe_capture is not None:
            probe_capture.save(args.output,actor,capture_contract,policy_execution,physics,args.seed)
        if args.shutdown_diagnostic_ticks is not None:
            write_json(args.output/'command_consistency.json', mapping)
            write_json(args.output/'shutdown_diagnostic.json', dict(
                control_ticks=ticks_completed, num_envs=lanes,
                requested_ticks=args.shutdown_diagnostic_ticks,
                first_episodes=len(first), elapsed_s=time.perf_counter()-start,
                scope='Shutdown diagnostic only; no task score or formal completion',
                actor_updates=0, critic_updates=0, world_updates=0))
            lifecycle_stage(args.output, 'SHUTDOWN_DIAGNOSTIC_FINISHED', control_ticks=ticks_completed)
            print('SUPERMIMIC_SHUTDOWN_DIAGNOSTIC_FINISHED', flush=True)
            return
        if len(first) != lanes:
            raise RuntimeError(f'Only {len(first)}/{lanes} pre-reset records; no live-state backfill permitted')
        if (search_spec is not None or hybrid is not None or args.approach_quick64
                or args.research_fixed128 or args.independent_panel_batch is not None or args.dynamimic_validation):
            if any(first[i]['case_id'] != cases[i]['case_id'] for i in range(lanes)):
                raise ValueError('Terminal case differs from the declared lane assignment')
        rows=[dict(first[i],source_case_index=i,case_id=cases[i].get('case_id',i)) for i in range(lanes)]
        if args.dynamimic_validation:
            from scripts.dynamimic_recovery_validation import report
            write_json(args.output/'dyna_validation.json', report(rows, args.checkpoint, args.dynamimic_alternate,
                                                                 args.dynamimic_companion,
                                                                 json.loads((args.output/'initial_states.json').read_text())))
        if teachers is not None:
            teachers.save(args.output)
        if scoring_protocol['approach_ball_scoring'] is not None:
            contacts = [event for event in env.contact_records if event['episode_index'] == 0]
            write_json(args.output/'native_contact_motion.json', contacts)
            write_json(args.output/'native_phase_timing.json', phase_timing.report())
        from quadsmash.rl.supermimic_native_metrics import summarize_native_episodes
        native_summary = summarize_native_episodes(rows)
        n=native_summary['qualified_airborne_string_contact']
        recovery=native_summary['conditional_recovery_count']
        recovery_observed=native_summary['conditional_recovery_denominator']
        with (args.checkpoint or source_path).open('rb') as checkpoint_stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: checkpoint_stream.read(8 << 20), b''):
                digest.update(chunk)
            checkpoint_sha256 = digest.hexdigest()
        metrics=dict(policy_execution=policy_execution,policy_execution_provenance=execution_provenance,evaluation_scope='TRAIN128 development, not A/B acceptance',
            num_envs=lanes,control_ticks=ticks_completed,episode_seconds=8.,
            qualified_airborne_string_contact=n,
            ball_side_complete_except_recovery=sum(r['ball_side_complete_except_recovery'] for r in rows),
            recovery_success=recovery if recovery_observed or scoring_protocol['approach_ball_scoring'] is None else None,
            physical_recovery_success=sum(r['physical_recovery_success'] is True for r in rows)
                if any(r['physical_recovery_success'] is not None for r in rows) else None,
            scoring_schema=rows[0].get('scoring_schema', STRICT_SCORING),
            conditional_recovery_count=recovery,conditional_recovery_denominator=recovery_observed,
            recovery_given_airborne_string_contact=native_summary['recovery_given_airborne_string_contact'],
            complete5=native_summary['complete5'],
            body_ground_seen=sum(r['body_ground_seen'] for r in rows),
            no_head_contact_recovery=None,no_independent_nonfoot_contact_recovery=None,
            missing_contact_audit_reason='No independent paired head/all-body contact labels in this environment; never default to false',
            numerical_failures=sum(r['numerical_failures'] for r in rows),
            extra_mpc_interventions=0,extra_teacher_interventions=0,
            historical_base_and_recovery_assistance_preserved=contract['historical_base_and_recovery_assistance_preserved'],
            task_contract=contract,source=init_meta,actor_updated=trained is not None,
            elapsed_s=time.perf_counter()-start,physics=physics,
            native_step_and_observation_check_seconds=step_wall_seconds,
            evaluated_checkpoint_sha256=checkpoint_sha256)
        if args.diagnostic_grounded_miss:
            metrics.update(evaluation_scope='frozen TRAIN128 grounded-miss gate comparison',
                           checkpoint_policy_execution=execution_contract(POSTHIT),
                           first_episode_extra_gate_ticks=gate_ticks.tolist(),
                           first_episode_extra_gate_onset=gate_first_tick.tolist())
        if args.independent_panel_batch is not None:
            metrics.update(task_contract=capture_contract, evaluation_scope='independent512 stage-end research evaluation')
        if args.research_fixed128:
            metrics.update(task_contract=capture_contract, evaluation_scope='fixed128 held-out research development')
        if args.approach_quick64:
            metrics.update(task_contract=capture_contract, evaluation_scope='fixed train64 Approach quick-check')
        actor_state_after=tensor_digest(actor.state_dict().items())
        if actor_state_after!=actor_state_before:raise RuntimeError('Frozen evaluation mutated actor weights')
        if search_spec is not None or hybrid is not None:
            from quadsmash.physics.full_racket_v4 import string_polygon
            from quadsmash.rl.supermimic_native_approach_search import native_fitness
            from quadsmash.tasks.stand.observation_contract import inverse_rotate

            polygon = torch.tensor(string_polygon(env.racket_config))
            edge = polygon.roll(-1, 0) - polygon
            inward = torch.nn.functional.normalize(torch.stack((-edge[:, 1], edge[:, 0]), -1), dim=-1)
            boundary = (polygon * inward).sum(-1)

            def signed_margin(motion):
                if not all(k in motion for k in ('point_m', 'racket_center_m', 'racket_quaternion_w')):
                    return None
                local = inverse_rotate(polygon.new_tensor([motion['racket_quaternion_w']]),
                                       polygon.new_tensor([motion['point_m']]) - polygon.new_tensor([motion['racket_center_m']]))
                return float((local[0, 1:] @ inward.T - boundary).min()) if local.isfinite().all() else None

            raw = {row['env_id']: row for row in env.rally_records if row['episode_index'] == 0}
            if hybrid is not None:
                variants = []
                for start in range(64, 128, 16):
                    group = [raw[i] for i in range(start, start + 16)]
                    variants.append(native_fitness(group, env.court_config, signed_margin))
                quick = native_fitness([raw[i] for i in range(64)], env.court_config, signed_margin)
                launch = torch.tensor([row['launch']['position_m'] + row['launch']['velocity_mps']
                                       for row in cases[64:]])
                data = pack_tapes(hybrid_frames, variants, hybrid['case_indices'], launch, actor_state_after)
                data['flight_truth'] = dict(
                    ball_net=torch.tensor([raw[i]['shuttle_net_contact'] for i in range(64, 128)]),
                    landing=torch.tensor([raw[i]['first_landing_m'] for i in range(64, 128)]))
                data['checkpoint_sha256'] = checkpoint_sha256
                torch.save(data, args.output / 'hybrid_native.tmp')
                (args.output / 'hybrid_native.tmp').replace(args.output / 'hybrid_native.pt')
                write_json(args.output / 'hybrid_boundary.json', dict(
                    panel=quick, variants=variants, panel_sha256=quick_identity['sha256'],
                    examples=data['examples'], actor_sha256=actor_state_after,
                    native_lanes=lanes, elapsed_s=metrics['elapsed_s']))
            else:
                expected = [cases[i]['case_id'] for i in range(32)]
                variants = []
                for branch, bias in enumerate(search_spec['offsets']):
                    group = [raw[i] for i in range(branch * 32, (branch + 1) * 32)]
                    if [row['case_id'] for row in group] != expected:
                        raise ValueError('Unpaired native search outcomes')
                    variants.append(dict(offset=bias, **native_fitness(group, env.court_config, signed_margin)))
                write_json(args.output/'approach_search_results.json', dict(
                    offsets=search_spec['offsets'], case_indices=search_spec['case_indices'],
                    actor_sha256=actor_state_after, checkpoint_sha256=checkpoint_sha256, variants=variants,
                    elapsed_s=metrics['elapsed_s'], native_lanes=lanes))
        head_active_lane_steps = dict(zip(("recovery", "approach"), head_counts.tolist()))
        metrics.update(actor_state_sha256=actor_state_after,actor_parameters_unchanged=True,
            optimizer_updates=dict(actor=0,world=0,critic=0),head_active_lane_steps=head_active_lane_steps)
        if args.approach_sensitivity:
            metrics.update(approach_sensitivity=dict(relative_rms=args.approach_sensitivity,
                                                       seed=args.seed + 1, source_actor_sha256=clean_actor_sha))
        if args.approach_prefix_checkpoint:
            metrics.update(approach_native_prefix=dict(
                mode='verified_scene_restart' if args.approach_verify_snapshot else 'ground_prefix_replay',
                switch_tick=args.approach_prefix_tick,
                                                         actor_sha256=prefix_actor_sha))
        if teachers is not None:
            metrics.update(evaluation_scope='Recovery training teacher branches', extra_teacher_interventions=True)
        metrics['native_episode_metrics'] = native_summary
        if scoring_protocol['relaxed_scoring'] is not None:
            strict = [r['strict_observed'] for r in rows]
            metrics['strict_observed'] = {
                name: dict(success=sum(r[name] is True for r in strict),
                           observed=sum(r[name] is not None for r in strict))
                for name in ('complete5', 'recovery_success')
            }
            metrics['terminated_success_count'] = sum(r['terminated_success'] for r in rows)
            if args.relaxed_survival_diagnostic:
                survival = [r['relaxed_survival_one_second'] for r in rows]
                observed = sum(r['observed'] for r in survival)
                survived = sum(r['survived'] is True for r in survival)
                metrics['relaxed_survival_one_second'] = dict(
                    reached=sum(r['success_reached'] for r in survival),
                    observed=observed, survived=survived, rate=survived / observed if observed else None)
        metrics['conditional_recovery_scope'] = native_summary['conditional_recovery_scope']
        metrics['native_episode_return_mean'] = native_summary['native_episode_return_mean']
        metrics['native_episode_discounted_return_mean'] = sum(r['native_episode_discounted_return'] for r in rows)/len(rows)
        metrics['native_return_gamma'] = env._sm_native_returns.gamma
        write_json(args.output/'episodes_summary.json',rows)
        if args.recovery_prefix_checkpoint:
            metrics.update(evaluation_scope='fixed128 mixed-policy Recovery prefix diagnostic',
                           recovery_prefix_checkpoint=str(prefix_path), recovery_prefix_actor_sha=prefix_actor_sha,
                           recovery_switch_tick=prefix_tick, eligible_for_promotion=False)
        write_json(args.output/'metrics.json',metrics)
        write_json(args.output/'command_consistency.json',mapping)
        write_json(args.output/'completed.json',dict(control_ticks=ticks_completed,episodes=lanes,formal_acceptance=False))
        stage='FINISHED'
        lifecycle_stage(args.output, 'SIMULATION_FINISHED', control_ticks=ticks_completed, episodes=lanes)
        print('SUPERMIMIC_EVAL_RESULT='+json.dumps(metrics),flush=True)
    except BaseException as error:
        primary_failed = True
        text=traceback.format_exc()
        (args.output/'failure_traceback.txt').write_text(text)
        write_json(args.output/'failure.json',dict(stage=stage,ticks_completed=ticks_completed,
                   attempted_tick=tick+1, stop_requested=stop_request['signal'],
                   type=type(error).__name__,message=str(error)))
        if mapping is not None:
            write_json(args.output/'command_consistency.json',mapping)
        print(text,file=sys.stderr,flush=True)
        raise
    finally:
        close_resource(args.output, 'ENV', env, primary_failed=primary_failed)


if __name__=='__main__':
    # Isaac Sim 5.1's PhysX Python bindings can run a module-level destructor
    # after Kit has already released the framework.  That destructor calls
    # PyThreadState_Get without the GIL and aborts with
    # ``PyThreadState_Get ... tstate is NULL``.  The scoped lifecycle above has
    # already flushed artifacts and closed native resources, so bypass Python's
    # second interpreter-finalization pass on the clean path.
    os.environ['QUADSMASH_SKIP_KIT_CLOSE'] = '1'
    try:
        main()
    except BaseException:
        # The failure artifact and traceback are written by main().  Avoid
        # interpreter finalization too, since it has the same PhysX/GIL bug.
        os._exit(1)
    else:
        os._exit(0)
