#!/usr/bin/env python3
"""Explicit reset-persistent recovery trial over a measured fixed residual actor.

Original task thresholds, robot and actuators are retained; reset behavior changes explicitly.
Ground dimensions and first nonfoot contact events are recorded explicitly.
This is development evidence, never an inherited or formal acceptance score.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import signal
import sys
import traceback
import io
import importlib.util

ROOT = Path('/home/autodrive/LZQW/unitree_smash_teacher_eval')
sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--landing-blas-threads', type=int, choices=[0,1], default=0,
    help='Explicit process-local OpenBLAS limit after Isaac startup; 0 retains historical runtime')
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--pure-control-configuration', type=Path, required=True)
parser.add_argument('--teacher-only', action='store_true', required=True,
    help='Disable residual actor and all learned recovery/approach heads')
parser.add_argument('--learned-recovery-command-head', type=Path, default=None,
    help='Explicit learned native recovery head; extra MPC/VHSIP disabled, all lanes causally gated')
parser.add_argument('--learned-recovery-mpc-correction', action='store_true',
    help='Explicit learned-proposal then MPC execution collection; all128 causal lanes, not teacher-off')
parser.add_argument('--expert-bank', type=Path, default=None,
    help='Explicit case-conditioned measured expert bank; controller trial only')
parser.add_argument('--persistent-recovery-trial', action='store_true', required=True,
    help='Explicitly keep declared recovery assistance effective after every controller reset')
parser.add_argument('--precision-receipt', type=Path, required=True)
parser.add_argument('--ground-xy-m', type=float, choices=(64.,4096.), required=True)
parser.add_argument('--training-indices', type=int, nargs='+', default=None)
parser.add_argument('--noise', choices=('none', 'A', 'B'), default='none')
parser.add_argument('--seed', type=int, default=724201)
parser.add_argument('--record-first-episodes',action='store_true',
    help='Losslessly retain actual policy inputs and dispatched actions for every selected first episode')
parser.add_argument('--landing-mpc-cases', type=int, nargs='+', default=None,
    help='Optional measured landing-support MPC controller trial for selected TRAIN indices')
parser.add_argument('--landing-mpc-horizon-steps', type=int, default=10,
    help='Short rolling support-MPC horizon in 10 ms policy steps')
parser.add_argument('--landing-mpc-allow-first-step', action='store_true',
    help='Apply a feasible first QP stage only when the local risk score improves, even if the terminal gate rejects')
parser.add_argument('--landing-mpc-min-risk-score-drop', type=float, default=0.,
    help='Minimum dimensionless risk-score decrease required for a nonterminal first-step application')
parser.add_argument('--landing-mpc-min-start-risk-score', type=float, default=0.,
    help='Minimum risk score required to start a nonterminal touchdown intervention')
parser.add_argument('--landing-mpc-hold-steps', type=int, default=0,
    help='Number of subsequent policy ticks to keep a started touchdown intervention active')
parser.add_argument('--landing-mpc-hold-first-action', action='store_true',
    help='Hold the accepted first normalized action during hold-steps instead of re-solving and replacing it every tick')
parser.add_argument('--landing-mpc-executor-replay', action='store_true',
    help='Replay position-target rate limits and implicit-PD effort saturation inside MPC prediction')
parser.add_argument('--landing-mpc-command-rate-scale', type=float, default=1.0,
    help='Scale the native one-tick position-target rate allowance used by MPC (0<scale<=1)')
parser.add_argument('--landing-mpc-support-margin', action='store_true',
    help='Use the experimental support-margin viability gate (MPC v2)')
parser.add_argument('--landing-mpc-torque-overlay', action='store_true',
    help='Apply the accepted MPC leg torque as an explicit residual on top of the position-PD command')
parser.add_argument('--landing-mpc-touchdown-recovery', action='store_true',
    help='Allow one-contact touchdown recovery and keep rolling a ground-level swing-foot target')
parser.add_argument('--landing-mpc-force-post-touchdown', action='store_true',
    help='After measured post-touchdown phase, solve and apply MPC without waiting for risk thresholds')
parser.add_argument('--landing-mpc-causal-posthit-only', action='store_true',
    help='Original frozen actor with causal observed-hit/support gate on all TRAIN128; no learned head')
parser.add_argument('--landing-mpc-latched-post-touchdown-ticks', type=int, default=0,
    help='Keep the post-touchdown MPC lifecycle active for this many 100 Hz ticks after phase774 is first observed (0 disables latch)')
parser.add_argument('--landing-mpc-contact-force-threshold', type=float, default=0.,
    help='Minimum upward measured foot force for MPC support (0 keeps the existing force-norm hysteresis flag)')
parser.add_argument('--landing-mpc-contact-release-threshold', type=float, default=None,
    help='Upward-force threshold below which a latched foot begins unloading (default half engage threshold)')
parser.add_argument('--landing-mpc-contact-release-ticks', type=int, default=3,
    help='Consecutive unloading ticks before measured support is released')
parser.add_argument('--landing-mpc-mode-validation-force', type=float, default=0.,
    help='Optional upward force required for each predicted support foot during held MPC actions')
parser.add_argument('--landing-mpc-mode-validation-ticks', type=int, default=3,
    help='Consecutive mode-validation failures before a held MPC action is released')
parser.add_argument('--landing-mpc-hold-tracking-error-release-rad', type=float, default=0.,
    help='Release a held MPC target when measured joint tracking error exceeds this value (0 disables)')
parser.add_argument('--landing-mpc-predict-touchdown-contacts', action='store_true',
    help='Let MPC add an imminent swing-foot contact to later horizon stages (event approximation)')
parser.add_argument('--landing-mpc-contact-mode-fallback', action='store_true',
    help='When the current support mode is infeasible, try bounded imminent-contact modes')
parser.add_argument('--landing-mpc-contact-mode-sequence-search', action='store_true',
    help='Evaluate bounded imminent touchdown modes even when the current QP is feasible')
parser.add_argument('--landing-mpc-contact-mode-sequence-max-candidates', type=int, default=3,
    help='Maximum imminent touchdown mode candidates evaluated per horizon stage')
parser.add_argument('--landing-mpc-touchdown-mode-height-scale', type=float, default=2.5,
    help='Multiplier for the touchdown height/velocity window used by contact-mode fallback')
parser.add_argument('--landing-mpc-capture-foot-gain', type=float, default=0.,
    help='VHSIP/capture-point horizontal offset gain for inactive swing feet')
parser.add_argument('--landing-mpc-capture-foot-max-offset', type=float, default=.25,
    help='Maximum horizontal capture-point foot offset in metres')
parser.add_argument('--landing-mpc-leg-posture-blend', type=float, default=.15,
    help='Blend from the student leg posture toward the admitted landing posture')
parser.add_argument('--landing-mpc-arm-posture-blend', type=float, default=0.,
    help='Blend from the student arm posture toward the admitted landing posture')
parser.add_argument('--landing-mpc-touchdown-height-tolerance', type=float, default=.012,
    help='Predicted foot-to-support-plane height tolerance for contact-mode transition in metres')
parser.add_argument('--landing-mpc-touchdown-vertical-speed-limit', type=float, default=1.5,
    help='Maximum downward foot speed accepted for predicted contact-mode transition in m/s')
parser.add_argument('--landing-mpc-touchdown-impact-projection', action='store_true',
    help='Project predicted new-contact normal velocity through a unilateral touchdown impulse')
parser.add_argument('--landing-mpc-force-distribution-weight', type=float, default=0.,
    help='Soft equal-share normal support-force allocation weight')
parser.add_argument('--landing-mpc-force-wrench-weight', type=float, default=0.,
    help='Soft centroidal net-force/zero-moment support allocation weight')
parser.add_argument('--landing-mpc-swing-foot-weight', type=float, default=0.,
    help='Optional soft touchdown-foot task weight; zero preserves the original support QP')
parser.add_argument('--landing-mpc-swing-foot-kp', type=float, default=80.,
    help='Soft touchdown-foot proportional acceleration gain')
parser.add_argument('--landing-mpc-swing-foot-kd', type=float, default=18.,
    help='Soft touchdown-foot damping gain')
parser.add_argument('--landing-mpc-swing-foot-clearance', type=float, default=.025,
    help='Target clearance above the measured support plane in metres')
parser.add_argument('--landing-mpc-nonfoot-clearance-weight', type=float, default=0.,
    help='Soft MPC weight for keeping thighs above the support plane')
parser.add_argument('--landing-mpc-nonfoot-clearance-hard', action='store_true',
    help='Add a one-step hard lower acceleration bound for near-ground thigh links')
parser.add_argument('--landing-mpc-nonfoot-clearance-m', type=float, default=.08,
    help='Minimum predicted thigh clearance above the measured support plane in metres')
parser.add_argument('--landing-mpc-nonfoot-clearance-kp', type=float, default=120.,
    help='Thigh clearance proportional acceleration gain')
parser.add_argument('--landing-mpc-nonfoot-clearance-kd', type=float, default=25.,
    help='Thigh clearance damping gain')
parser.add_argument('--landing-mpc-continuous-reference', action='store_true',
    help='Use continuous COM/support references and preserve the supplied arm command (MPC v3)')
parser.add_argument('--landing-mpc-load-aware-support', action='store_true',
    help='Weight the continuous support reference by measured per-foot vertical load')
parser.add_argument('--landing-mpc-load-force-floor', type=float, default=2.,
    help='Minimum force weight for a foot selected by the observed-contact flag')
parser.add_argument('--landing-mpc-minimum-command-cost', action='store_true',
    help='Prefer small command changes with full-horizon feasibility fallback')
parser.add_argument('--landing-mpc-command-change-weight', type=float, default=1.,
    help='Nonnegative command-change cost weight after QP objective normalization')
parser.add_argument('--landing-mpc-full-command-damper', action='store_true',
    help='All18 interval commands with joint velocity envelope; requires free-arm and active-set interval mode')
parser.add_argument('--landing-mpc-interval-active-set', action='store_true',
    help='Use quadprog for the identical interval QP; requires interval-command mode')
parser.add_argument('--landing-mpc-interval-command-qp', action='store_true',
    help='Constrain all 16 implicit substeps within each held native command interval')
parser.add_argument('--landing-mpc-implicit-command-qp', action='store_true',
    help='Experimental matching implicit-PD QP and predictor; requires command-history mode')
parser.add_argument('--landing-mpc-command-history-interface', action='store_true',
    help='Use native command-history arm slew and physical next-state velocity limits (command-history MPC)')
parser.add_argument('--landing-mpc-free-arm', action='store_true',
    help='With continuous-reference MPC, leave arm commands as a soft cost instead of an exact PD constraint')
parser.add_argument('--reactive-vhsip-cases', type=int, nargs='+', default=None,
    help='Controller-trial TRAIN indices for the GPU-batched VHSIP reactive landing reference')
parser.add_argument('--reactive-vhsip-post-touchdown-only', action='store_true',
    help='Keep the learned/policy action through flight and start VHSIP only after measured foot touchdown')
parser.add_argument('--reactive-vhsip-capture-model', choices=('height','vhsip'), default='height',
    help='Capture model for the reactive landing trial; height preserves the prior approximation, vhsip runs the analytical VHSIP rollout')
parser.add_argument('--reactive-vhsip-settling-s', type=float, default=.7,
    help='VHSIP/landing reference settling horizon in seconds')
parser.add_argument('--reactive-vhsip-capture-gain', type=float, default=1.,
    help='Horizontal capture reference gain')
parser.add_argument('--reactive-vhsip-blend-s', type=float, default=.08,
    help='Bounded joint-reference blend time after the selected intervention stage')
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
if args.landing_blas_threads:
    import os
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.landing_blas_threads)

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
from scripts.landing_support_mpc_v1 import LandingSupportMPCV1, LandingMPCSettingsV1
from scripts.landing_support_mpc_v2 import LandingSupportMPCV2, LandingMPCSettingsV2
from scripts.landing_support_mpc_v3 import LandingSupportMPCV3
from scripts.landing_command_history_mpc_v1 import LandingCommandHistoryMPCV1
from scripts.landing_implicit_command_mpc_v1 import LandingImplicitCommandMPCV1
from scripts.landing_interval_command_mpc_v1 import LandingIntervalCommandMPCV1
from scripts.landing_active_set_interval_mpc_v1 import LandingActiveSetIntervalMPCV1
from scripts.landing_full_command_interval_mpc_v1 import LandingFullCommandIntervalMPCV1
from scripts.landing_feasible_command_cost_mpc_v1 import LandingFeasibleCommandCostMPCV1
from scripts.touchdown_contact_state_v1 import TouchdownContactSettingsV1,TouchdownContactStateV1
from scripts.whole_body_support_qp_v1 import PositionPDInterface, SupportSettings

LANDING_BLAS_RUNTIME = None
if args.landing_blas_threads:
    from scripts.landing_blas_runtime_v1 import configure_landing_blas_threads
    LANDING_BLAS_RUNTIME = configure_landing_blas_threads(args.landing_blas_threads)

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
    def __init__(self, checkpoint, count, device,landing_trial=None,support_trial=None,disable_support=False,
                 landing_mpc_cases=None, landing_mpc_horizon_steps=10, landing_mpc_allow_first_step=False,
                 landing_mpc_min_risk_score_drop=0.,
                 landing_mpc_min_start_risk_score=0., landing_mpc_hold_steps=0,
                 landing_mpc_hold_first_action=False,
                 landing_mpc_executor_replay=False,
                 landing_mpc_command_rate_scale=1.0,
                 landing_mpc_support_margin=False,
                 landing_mpc_torque_overlay=False,
                 landing_mpc_touchdown_recovery=False,
                 landing_mpc_force_post_touchdown=False,
                 landing_mpc_latched_post_touchdown_ticks=0,
                 landing_mpc_contact_force_threshold=0.,
                 landing_mpc_contact_release_threshold=None,
                 landing_mpc_contact_release_ticks=3,
                 landing_mpc_mode_validation_force=0.,
                 landing_mpc_mode_validation_ticks=3,
                 landing_mpc_hold_tracking_error_release_rad=0.,
                 landing_mpc_predict_touchdown_contacts=False,
                 landing_mpc_contact_mode_fallback=False,
                 landing_mpc_contact_mode_sequence_search=False,
                 landing_mpc_contact_mode_sequence_max_candidates=3,
                 landing_mpc_touchdown_mode_height_scale=2.5,
                 landing_mpc_capture_foot_gain=0.,
                 landing_mpc_capture_foot_max_offset=.25,
                 landing_mpc_leg_posture_blend=.15,
                 landing_mpc_arm_posture_blend=0.,
                 landing_mpc_touchdown_height_tolerance=.012,
                 landing_mpc_touchdown_vertical_speed_limit=1.5,
                 landing_mpc_touchdown_impact_projection=False,
                 landing_mpc_force_distribution_weight=0.,
                 landing_mpc_force_wrench_weight=0.,
                 landing_mpc_swing_foot_weight=0., landing_mpc_swing_foot_kp=80.,
                 landing_mpc_swing_foot_kd=18., landing_mpc_swing_foot_clearance=.025,
                 landing_mpc_nonfoot_clearance_weight=0., landing_mpc_nonfoot_clearance_m=.08,
                 landing_mpc_nonfoot_clearance_hard=False,
                 landing_mpc_nonfoot_clearance_kp=120., landing_mpc_nonfoot_clearance_kd=25.,
                 landing_mpc_continuous_reference=False,
                 landing_mpc_load_aware_support=False,
                 landing_mpc_load_force_floor=2.,
                 landing_mpc_free_arm=False,
                 landing_mpc_command_history_interface=False,
                 landing_mpc_implicit_command_qp=False,
                 landing_mpc_interval_command_qp=False,
                 landing_mpc_interval_active_set=False,
                 landing_mpc_full_command_damper=False,
                 landing_mpc_minimum_command_cost=False, landing_mpc_command_change_weight=1.,
                 reactive_vhsip_cases=None, reactive_vhsip_post_touchdown_only=False,
                 reactive_vhsip_capture_model='height', reactive_vhsip_settling_s=.7,
                 reactive_vhsip_capture_gain=1., reactive_vhsip_blend_s=.08,
                 teacher_only=False):
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
        self.teacher_only=bool(teacher_only)
        self.actor_forward_count=0
        self.optimizer_step_count=0
        self.actor=torch.jit.load(io.BytesIO(checkpoint['residual_actor_jit']),map_location=device).eval() if (has_actor and not self.teacher_only) else None
        self.mixer=ResidualActionMixerV4(count,device,checkpoint['residual_joint_scale_rad'],
            float(checkpoint.get('residual_arm_target_rate_rad_s',5.)),
            float(checkpoint.get('residual_scale_limit_rad',max(checkpoint['residual_joint_scale_rad'])))) if has_actor else None
        self.memory_conditioned=bool(metadata.get('memory_conditioned',False))
        self.first_ball_feature=torch.zeros((count,18),device=device) if self.memory_conditioned else None
        self.first_ball_seen=torch.zeros((count,),dtype=torch.bool,device=device) if self.memory_conditioned else None
        self.landing_mpc_cases=set(int(x) for x in (landing_mpc_cases or []))
        self.landing_mpc_allow_first_step=bool(landing_mpc_allow_first_step)
        self.landing_mpc_min_risk_score_drop=float(landing_mpc_min_risk_score_drop)
        if self.landing_mpc_min_risk_score_drop < 0. or not np.isfinite(self.landing_mpc_min_risk_score_drop):
            raise ValueError('landing_mpc_min_risk_score_drop must be finite and nonnegative')
        self.landing_mpc_min_start_risk_score=float(landing_mpc_min_start_risk_score)
        if self.landing_mpc_min_start_risk_score < 0. or not np.isfinite(self.landing_mpc_min_start_risk_score):
            raise ValueError('landing_mpc_min_start_risk_score must be finite and nonnegative')
        self.landing_mpc_hold_steps=int(landing_mpc_hold_steps)
        if self.landing_mpc_hold_steps < 0:
            raise ValueError('landing_mpc_hold_steps must be nonnegative')
        self.landing_mpc_hold_first_action=bool(landing_mpc_hold_first_action)
        self.landing_mpc_executor_replay=bool(landing_mpc_executor_replay)
        self.landing_mpc_command_rate_scale=float(landing_mpc_command_rate_scale)
        if (not np.isfinite(self.landing_mpc_command_rate_scale) or
                not 0. < self.landing_mpc_command_rate_scale <= 1.):
            raise ValueError('landing_mpc_command_rate_scale must be in (0,1]')
        self._landing_mpc_hold_remaining=torch.zeros((count,),dtype=torch.int32,device=device)
        self._landing_mpc_finished_first_episode=torch.zeros((count,),dtype=torch.bool,device=device)
        self._landing_mpc_held_action=torch.zeros((count,18),dtype=torch.float32,device=device)
        self._landing_mpc_last_target=np.full((count,18),np.nan,dtype=np.float64)
        self._landing_mpc_last_predicted=np.full((count,18),np.nan,dtype=np.float64)
        self._landing_mpc_prediction_tick=np.full(count,-1,dtype=np.int64)
        self.landing_mpc_support_margin=bool(landing_mpc_support_margin)
        self.landing_mpc_torque_overlay=bool(landing_mpc_torque_overlay)
        self.landing_mpc_touchdown_recovery=bool(landing_mpc_touchdown_recovery)
        self.landing_mpc_force_post_touchdown=bool(landing_mpc_force_post_touchdown)
        self.landing_mpc_latched_post_touchdown_ticks=int(landing_mpc_latched_post_touchdown_ticks)
        if self.landing_mpc_latched_post_touchdown_ticks < 0:
            raise ValueError('landing_mpc_latched_post_touchdown_ticks must be nonnegative')
        self._landing_mpc_latched_post_touchdown=torch.zeros((count,),dtype=torch.bool,device=device)
        self._landing_mpc_latched_age=torch.zeros((count,),dtype=torch.int32,device=device)
        self.landing_mpc_contact_force_threshold=float(landing_mpc_contact_force_threshold)
        if (not np.isfinite(self.landing_mpc_contact_force_threshold) or
                self.landing_mpc_contact_force_threshold < 0.):
            raise ValueError('landing_mpc_contact_force_threshold must be nonnegative')
        release_threshold = (self.landing_mpc_contact_force_threshold * .5
                             if landing_mpc_contact_release_threshold is None
                             else float(landing_mpc_contact_release_threshold))
        self.landing_mpc_contact_release_threshold=release_threshold
        self.landing_mpc_contact_release_ticks=int(landing_mpc_contact_release_ticks)
        self.landing_mpc_mode_validation_force=float(landing_mpc_mode_validation_force)
        if (not np.isfinite(self.landing_mpc_mode_validation_force) or
                self.landing_mpc_mode_validation_force < 0.):
            raise ValueError('landing_mpc_mode_validation_force must be finite and nonnegative')
        self.landing_mpc_mode_validation_ticks=int(landing_mpc_mode_validation_ticks)
        if self.landing_mpc_mode_validation_ticks < 1:
            raise ValueError('landing_mpc_mode_validation_ticks must be positive')
        self.landing_mpc_hold_tracking_error_release_rad=float(landing_mpc_hold_tracking_error_release_rad)
        if (not np.isfinite(self.landing_mpc_hold_tracking_error_release_rad) or
                self.landing_mpc_hold_tracking_error_release_rad < 0.):
            raise ValueError('landing_mpc_hold_tracking_error_release_rad must be finite and nonnegative')
        self._landing_mpc_expected_contacts=np.zeros((count,4),dtype=bool)
        self._landing_mpc_mode_mismatch=torch.zeros((count,),dtype=torch.int32,device=device)
        self._landing_mpc_contact_state_lane=[TouchdownContactStateV1(
            TouchdownContactSettingsV1(self.landing_mpc_contact_force_threshold,
                                       release_threshold,
                                       self.landing_mpc_contact_release_ticks)) for _ in range(count)]
        self.landing_mpc_predict_touchdown_contacts=bool(landing_mpc_predict_touchdown_contacts)
        self.landing_mpc_contact_mode_fallback=bool(landing_mpc_contact_mode_fallback)
        self.landing_mpc_contact_mode_sequence_search=bool(landing_mpc_contact_mode_sequence_search)
        self.landing_mpc_contact_mode_sequence_max_candidates=int(landing_mpc_contact_mode_sequence_max_candidates)
        self.landing_mpc_touchdown_mode_height_scale=float(landing_mpc_touchdown_mode_height_scale)
        self.landing_mpc_capture_foot_gain=float(landing_mpc_capture_foot_gain)
        self.landing_mpc_capture_foot_max_offset=float(landing_mpc_capture_foot_max_offset)
        self.landing_mpc_leg_posture_blend=float(landing_mpc_leg_posture_blend)
        self.landing_mpc_arm_posture_blend=float(landing_mpc_arm_posture_blend)
        self.landing_mpc_touchdown_height_tolerance=float(landing_mpc_touchdown_height_tolerance)
        self.landing_mpc_touchdown_vertical_speed_limit=float(landing_mpc_touchdown_vertical_speed_limit)
        self.landing_mpc_touchdown_impact_projection=bool(landing_mpc_touchdown_impact_projection)
        self.landing_mpc_force_distribution_weight=float(landing_mpc_force_distribution_weight)
        if (not np.isfinite(self.landing_mpc_force_distribution_weight) or
                self.landing_mpc_force_distribution_weight < 0.):
            raise ValueError('landing_mpc_force_distribution_weight must be finite and nonnegative')
        self.landing_mpc_force_wrench_weight=float(landing_mpc_force_wrench_weight)
        if (not np.isfinite(self.landing_mpc_force_wrench_weight) or
                self.landing_mpc_force_wrench_weight < 0.):
            raise ValueError('landing_mpc_force_wrench_weight must be finite and nonnegative')
        self._effort_override=torch.zeros((count,18),device=device)
        self._effort_override_mask=torch.zeros((count,),dtype=torch.bool,device=device)
        self.landing_mpc_swing_foot_weight=float(landing_mpc_swing_foot_weight)
        self.landing_mpc_swing_foot_kp=float(landing_mpc_swing_foot_kp)
        self.landing_mpc_swing_foot_kd=float(landing_mpc_swing_foot_kd)
        self.landing_mpc_swing_foot_clearance=float(landing_mpc_swing_foot_clearance)
        self.landing_mpc_nonfoot_clearance_weight=float(landing_mpc_nonfoot_clearance_weight)
        self.landing_mpc_nonfoot_clearance_m=float(landing_mpc_nonfoot_clearance_m)
        self.landing_mpc_nonfoot_clearance_hard=bool(landing_mpc_nonfoot_clearance_hard)
        self.landing_mpc_nonfoot_clearance_kp=float(landing_mpc_nonfoot_clearance_kp)
        self.landing_mpc_nonfoot_clearance_kd=float(landing_mpc_nonfoot_clearance_kd)
        self.landing_mpc_continuous_reference=bool(landing_mpc_continuous_reference)
        self.landing_mpc_load_aware_support=bool(landing_mpc_load_aware_support)
        self.landing_mpc_load_force_floor=float(landing_mpc_load_force_floor)
        if (not np.isfinite(self.landing_mpc_load_force_floor) or
                self.landing_mpc_load_force_floor < 0.):
            raise ValueError('landing_mpc_load_force_floor must be finite and nonnegative')
        self.landing_mpc_free_arm=bool(landing_mpc_free_arm)
        self.landing_mpc_command_history_interface=bool(landing_mpc_command_history_interface)
        self.landing_mpc_minimum_command_cost=bool(landing_mpc_minimum_command_cost)
        self.landing_mpc_command_change_weight=float(landing_mpc_command_change_weight)
        if (not np.isfinite(self.landing_mpc_command_change_weight) or self.landing_mpc_command_change_weight<0 or
                (self.landing_mpc_minimum_command_cost and not landing_mpc_full_command_damper)):
            raise ValueError('Minimum-command cost requires full-command interval mode and finite nonnegative weight')
        self.landing_mpc_full_command_damper=bool(landing_mpc_full_command_damper)
        if self.landing_mpc_full_command_damper and not (landing_mpc_interval_command_qp and landing_mpc_interval_active_set and landing_mpc_free_arm):
            raise ValueError("Full-command damper requires free-arm and active-set interval mode")
        self.landing_mpc_interval_active_set=bool(landing_mpc_interval_active_set)
        if self.landing_mpc_interval_active_set and not landing_mpc_interval_command_qp:
            raise ValueError("Active-set backend requires interval-command QP")
        self.landing_mpc_interval_command_qp=bool(landing_mpc_interval_command_qp)
        if self.landing_mpc_interval_command_qp and (not self.landing_mpc_command_history_interface or landing_mpc_implicit_command_qp):
            raise ValueError('Interval command QP requires history interface and excludes single-step implicit QP')
        self.landing_mpc_implicit_command_qp=bool(landing_mpc_implicit_command_qp)
        if self.landing_mpc_implicit_command_qp and not self.landing_mpc_command_history_interface:
            raise ValueError("Implicit command QP requires the command-history interface")
        self.landing_mpc_arm_target_rate=float(checkpoint.get("residual_arm_target_rate_rad_s",5.))
        if self.landing_mpc_command_history_interface and (
                not self.landing_mpc_continuous_reference or
                not (self.landing_mpc_support_margin or self.landing_mpc_touchdown_recovery) or
                not self.landing_mpc_executor_replay or
                self.landing_mpc_torque_overlay or self.landing_mpc_hold_first_action):
            raise ValueError('Command-history MPC requires continuous reference/replay and no torque overlay/held action')
        self.reactive_vhsip_cases=None if reactive_vhsip_cases is None else set(int(x) for x in reactive_vhsip_cases)
        self.reactive_vhsip_post_touchdown_only=bool(reactive_vhsip_post_touchdown_only)
        if reactive_vhsip_capture_model not in ('height','vhsip'):
            raise ValueError('Unknown reactive VHSIP capture model')
        self.reactive_vhsip_capture_model=reactive_vhsip_capture_model
        self.reactive_vhsip_settling_s=float(reactive_vhsip_settling_s)
        self.reactive_vhsip_capture_gain=float(reactive_vhsip_capture_gain)
        self.reactive_vhsip_blend_s=float(reactive_vhsip_blend_s)
        if self.reactive_vhsip_cases and self.landing_config is not None:
            raise ValueError('Separate declared landing assistance and VHSIP trial')
        self.landing_mpc_logs=[]
        # Set by the native evaluator immediately before action().  Keeping
        # this as an explicit adapter makes it impossible to accidentally use
        # the public observation as the contact-sensitive MPC state when the
        # live simulator is available.
        self._landing_mpc_native_env=None
        self.landing_mpc=None
        if self.landing_mpc_cases:
            limits=np.asarray(checkpoint.get('residual_training_metadata',{}).get('joint_limits_rad',[]),dtype=np.float64)
            if limits.shape != (18,2):
                # The exact native limits are supplied on the first action by
                # the evaluator; this placeholder is replaced in configure.
                limits=None
            self._landing_mpc_horizon_steps=int(landing_mpc_horizon_steps)
            if self._landing_mpc_horizon_steps<1 or self._landing_mpc_horizon_steps>40:
                raise ValueError('landing_mpc_horizon_steps must be in 1..40')
        self.reactive_vhsip=None
        if self.reactive_vhsip_cases:
            vhsip_config=dict(mode=('post_touchdown_buffer_vhsip' if self.reactive_vhsip_post_touchdown_only
                                    else 'prepare_and_buffer_vhsip'), blend_s=self.reactive_vhsip_blend_s,
                settling_s=self.reactive_vhsip_settling_s, landing_height_m=.3,
                capture_gain=self.reactive_vhsip_capture_gain, ik_iterations=3,
                joint_target_rate_rad_s=20., ik_damping=.025,
                capture_model=self.reactive_vhsip_capture_model)
            self.reactive_vhsip=LandingAssistedController(self.base,count,device,vhsip_config,
                checkpoint['asset_sha256'],enabled_cases=self.reactive_vhsip_cases)

    def configure_landing_mpc(self, limits, default=None):
        """Create the optional CPU MPC after native limits are admitted."""
        if not self.landing_mpc_cases:
            return
        if torch.is_tensor(limits):
            limits=limits.detach().cpu().numpy()
        limits=np.asarray(limits,dtype=np.float64)
        if limits.shape != (18,2) or not np.isfinite(limits).all():
            raise ValueError('Landing MPC requires the admitted 18x2 native limits')
        interface=PositionPDInterface(
            limits,np.r_[np.full(12,25.),np.full(6,80.)],
            np.r_[np.full(12,.5),np.full(6,4.)],
            np.r_[np.full(12,23.5),np.full(6,100.)],
            np.r_[np.full(12,30.),np.full(6,5.)],.65)
        qp_settings=SupportSettings(control_dt_s=.01,solver_eps=1e-6,residual_tolerance=3e-4,
            swing_foot_weight=self.landing_mpc_swing_foot_weight,
            swing_foot_kp_s2=self.landing_mpc_swing_foot_kp,
            swing_foot_kd_s=self.landing_mpc_swing_foot_kd,
            swing_foot_clearance_m=self.landing_mpc_swing_foot_clearance,
            nonfoot_clearance_weight=self.landing_mpc_nonfoot_clearance_weight,
            nonfoot_clearance_hard=self.landing_mpc_nonfoot_clearance_hard,
            nonfoot_clearance_m=self.landing_mpc_nonfoot_clearance_m,
            nonfoot_clearance_kp_s2=self.landing_mpc_nonfoot_clearance_kp,
            nonfoot_clearance_kd_s=self.landing_mpc_nonfoot_clearance_kd,
            force_distribution_weight=self.landing_mpc_force_distribution_weight)
        qp_settings = replace(qp_settings,
            force_wrench_weight=self.landing_mpc_force_wrench_weight)
        qp_settings = replace(qp_settings,
            swing_foot_capture_gain=self.landing_mpc_capture_foot_gain,
            swing_foot_capture_max_m=self.landing_mpc_capture_foot_max_offset)
        if self.landing_mpc_support_margin or self.landing_mpc_touchdown_recovery:
            mpc_settings = LandingMPCSettingsV2(horizon_steps=self._landing_mpc_horizon_steps,
                minimum_support_feet=1 if self.landing_mpc_touchdown_recovery else 2,
                one_foot_max_com_distance_m=.20 if self.landing_mpc_touchdown_recovery else .045,
                landing_leg_posture_blend=(self.landing_mpc_leg_posture_blend
                                           if self.landing_mpc_touchdown_recovery else .35),
                landing_arm_posture_blend=(self.landing_mpc_arm_posture_blend
                                           if self.landing_mpc_touchdown_recovery else 0.),
                predict_touchdown_contacts=self.landing_mpc_predict_touchdown_contacts,
                contact_mode_fallback=self.landing_mpc_contact_mode_fallback,
                contact_mode_sequence_search=self.landing_mpc_contact_mode_sequence_search,
                contact_mode_sequence_max_candidates=self.landing_mpc_contact_mode_sequence_max_candidates,
                touchdown_mode_height_scale=self.landing_mpc_touchdown_mode_height_scale,
                touchdown_height_tolerance_m=self.landing_mpc_touchdown_height_tolerance,
                touchdown_vertical_speed_limit_mps=self.landing_mpc_touchdown_vertical_speed_limit,
                touchdown_impact_projection=self.landing_mpc_touchdown_impact_projection)
            mpc_settings = replace(mpc_settings, executor_replay=self.landing_mpc_executor_replay)
            mpc_settings = replace(mpc_settings, command_rate_scale=self.landing_mpc_command_rate_scale)
            continuous_type=LandingCommandHistoryMPCV1 if self.landing_mpc_command_history_interface else LandingSupportMPCV3
            if getattr(self,"landing_mpc_implicit_command_qp",False):
                continuous_type=LandingImplicitCommandMPCV1
            if getattr(self,"landing_mpc_interval_command_qp",False):
                continuous_type=(LandingActiveSetIntervalMPCV1 if getattr(self,"landing_mpc_interval_active_set",False)
                                 else LandingIntervalCommandMPCV1)
            if getattr(self,"landing_mpc_full_command_damper",False):
                continuous_type=LandingFullCommandIntervalMPCV1
            if self.landing_mpc_minimum_command_cost:
                continuous_type=LandingFeasibleCommandCostMPCV1
            continuous_extra=({'arm_target_rate_rad_s':self.landing_mpc_arm_target_rate}
                              if self.landing_mpc_command_history_interface else {})
            if self.landing_mpc_minimum_command_cost:
                continuous_extra['command_change_weight']=self.landing_mpc_command_change_weight
            self.landing_mpc=(continuous_type(interface, qp_settings=qp_settings,
                settings=mpc_settings, preserve_arm_command=not self.landing_mpc_free_arm,
                load_aware_support=self.landing_mpc_load_aware_support,
                load_force_floor_n=self.landing_mpc_load_force_floor,**continuous_extra)
                if self.landing_mpc_continuous_reference else
                LandingSupportMPCV2(interface, qp_settings=qp_settings,
                settings=mpc_settings))
            if default is not None:
                if torch.is_tensor(default):
                    default=default.detach().cpu().numpy()
                self.landing_mpc.set_posture_target(default)
        else:
            self.landing_mpc=LandingSupportMPCV1(interface, qp_settings=qp_settings,
                settings=LandingMPCSettingsV1(horizon_steps=self._landing_mpc_horizon_steps,
                    executor_replay=self.landing_mpc_executor_replay))

    def landing_mpc_diagnostics(self):
        result=list(self.landing_mpc_logs);self.landing_mpc_logs=[];return result

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
        if self.reactive_vhsip is not None:self.reactive_vhsip.reset(ids)
        else:self.base.reset(ids)
        if self.mixer is not None:self.mixer.reset(ids)
        if self.memory_conditioned:
            if ids is None:
                self.first_ball_feature.zero_();self.first_ball_seen.zero_()
            else:
                self.first_ball_feature[ids].zero_();self.first_ball_seen[ids]=False
        if ids is None:
            self._landing_mpc_finished_first_episode.zero_()
            self._effort_override.zero_();self._effort_override_mask.zero_()
            self._landing_mpc_hold_remaining.zero_()
            self._landing_mpc_held_action.zero_()
            self._landing_mpc_last_target.fill(np.nan)
            self._landing_mpc_last_predicted.fill(np.nan)
            self._landing_mpc_prediction_tick.fill(-1)
            self._landing_mpc_expected_contacts[:]=False
            self._landing_mpc_mode_mismatch.zero_()
            self._landing_mpc_latched_post_touchdown.zero_()
            self._landing_mpc_latched_age.zero_()
            for contact_state in self._landing_mpc_contact_state_lane:
                contact_state.reset()
        else:
            self._landing_mpc_finished_first_episode[ids]=False
            self._effort_override[ids].zero_();self._effort_override_mask[ids]=False
            self._landing_mpc_hold_remaining[ids]=0
            self._landing_mpc_held_action[ids].zero_()
            if isinstance(ids, slice):
                selected = range(self._effort_override.shape[0])[ids]
            elif torch.is_tensor(ids):
                selected = ids.detach().cpu().reshape(-1).tolist()
            else:
                selected = np.asarray(ids).reshape(-1).tolist()
            for lane in selected:
                self._landing_mpc_contact_state_lane[int(lane)].reset()
                self._landing_mpc_last_target[int(lane),:]=np.nan
                self._landing_mpc_last_predicted[int(lane),:]=np.nan
                self._landing_mpc_prediction_tick[int(lane)]=-1
                self._landing_mpc_expected_contacts[int(lane),:]=False
                self._landing_mpc_mode_mismatch[int(lane)]=0
                self._landing_mpc_latched_post_touchdown[int(lane)]=False
                self._landing_mpc_latched_age[int(lane)]=0
        if self.landing_mpc is not None and hasattr(self.landing_mpc, 'reset'):
            if ids is None:
                self.landing_mpc.reset()
            else:
                for lane in selected:
                    self.landing_mpc.reset(lane=int(lane))

    def effort_override(self):
        """Return explicit effort residuals for the native implicit-PD actuator."""
        return self._effort_override

    def finish_landing_mpc_episodes(self, done):
        """This evaluator scores first episodes only; stop unscored MPC work."""
        newly = done.bool() & ~self._landing_mpc_finished_first_episode
        self._landing_mpc_finished_first_episode |= done.bool()
        self._landing_mpc_hold_remaining[newly] = 0
        if isinstance(self.landing_mpc, LandingSupportMPCV3):
            for lane in torch.where(newly)[0].tolist():
                self.landing_mpc.reset(lane=lane)

    def set_landing_mpc_native_state(self, env):
        """Attach the live Isaac articulation for the next MPC action."""
        self._landing_mpc_native_env=env
        if self.reactive_vhsip is not None and hasattr(self.reactive_vhsip, 'set_native_state'):
            self.reactive_vhsip.set_native_state(env)

    @torch.inference_mode()
    def action(self,observation,default,limits,*,defer_landing_mpc=False):
        self._effort_override.zero_();self._effort_override_mask.zero_()
        source=self.reactive_vhsip if self.reactive_vhsip is not None else self.base
        base=source.action(observation,default,limits)
        if self.actor is None:return base
        self.actor_forward_count += 1
        raw=observation['policy']
        if self.memory_conditioned:
            present=raw[:,67]>.5
            choose=present & (~self.first_ball_seen)
            if bool(choose.any()):
                self.first_ball_feature[choose]=torch.cat((raw[choose,61:67],raw[choose,759:770],raw[choose,775:776]),dim=1)
                self.first_ball_seen[choose]=True
            actor_input=torch.cat((raw,self.first_ball_feature),dim=1)
        else:
            actor_input=raw
        residual=self.actor(actor_input)
        action=self.mixer.action(base,residual,observation['policy'],default,limits)
        if defer_landing_mpc:return action
        return self.apply_landing_mpc(observation,default,limits,action)

    @torch.inference_mode()
    def apply_landing_mpc(self,observation,default,limits,action,*,eligible_mask=None):
        if self.landing_mpc is None:return action
        raw=observation['policy'];public_clock=raw[:,775]
        # phase774 is a measured current support/contact phase. We keep the
        # first native version explicit: no future outcome label or case ID is
        # exposed to the MPC; the selected case list only limits this trial.
        phase_touchdown=(raw[:,774]>.5)
        if eligible_mask is not None:
            if eligible_mask.shape!=phase_touchdown.shape or eligible_mask.dtype!=torch.bool:
                raise ValueError('Require causal per-lane recovery eligibility')
            phase_touchdown &= eligible_mask
        if self.landing_mpc_latched_post_touchdown_ticks > 0:
            newly_landed = phase_touchdown & (~self._landing_mpc_latched_post_touchdown)
            self._landing_mpc_latched_post_touchdown |= phase_touchdown
            self._landing_mpc_latched_age = torch.where(
                self._landing_mpc_latched_post_touchdown,
                self._landing_mpc_latched_age + 1,
                self._landing_mpc_latched_age)
            active = phase_touchdown | (
                self._landing_mpc_latched_post_touchdown &
                (self._landing_mpc_latched_age <= self.landing_mpc_latched_post_touchdown_ticks))
        else:
            newly_landed = torch.zeros_like(phase_touchdown)
            active=phase_touchdown
        active = active & ~self._landing_mpc_finished_first_episode
        if eligible_mask is not None:active &= eligible_mask
        for lane in torch.where(active)[0].tolist():
            case=int(lane)  # evaluator maps env lanes to the fixed TRAIN panel
            if case not in self.landing_mpc_cases:continue
            episode_index = (int(self._landing_mpc_native_env.episode_index[lane].item())
                             if self._landing_mpc_native_env is not None else None)
            try:
                from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
                if not hasattr(self,'_landing_mpc_model'):
                    self._landing_mpc_model=LoadedPinocchioModelV1()
                lim=limits.detach().cpu().numpy()
                dflt=default.detach().cpu().numpy()
                if self._landing_mpc_native_env is not None:
                    state=self._landing_mpc_model.state_from_native(
                        self._landing_mpc_native_env, lane, dflt,
                        self.landing_mpc_contact_force_threshold)
                    state_source='native_sim_state'
                else:
                    pub=raw[lane].detach().cpu().numpy()
                    state=self._landing_mpc_model.state_from_public(pub,dflt)
                    state_source='public_observation_reconstruction'
                if self._landing_mpc_native_env is not None:
                    geometric_contacts=getattr(state, 'geometric_contacts', state.contacts)
                    measured_force=getattr(state, 'measured_force_world', None)
                    state.contacts=self._landing_mpc_contact_state_lane[lane].update(
                        geometric_contacts, measured_force)
                joint_ref=lim[:,0]+.5*(action[lane].detach().cpu().numpy()+1.)*(lim[:,1]-lim[:,0])
                requested_student_joint_ref=joint_ref.copy()
                if self.landing_mpc_command_history_interface:
                    if self._landing_mpc_native_env is None:
                        raise ValueError('Command-history MPC requires native final-command state')
                    native=self._landing_mpc_native_env
                    state.previous_command_target=native.targets[lane,native.joint_ids].detach().cpu().numpy().copy()
                    # The same history admission applies to student fallback;
                    # failed solves must not jump back to a different slew history.
                    joint_ref=self.landing_mpc.admit_command_target(state,joint_ref)
                    action[lane]=torch.as_tensor(self.landing_mpc.qp.interface.normalized_action(joint_ref),
                                                device=action.device,dtype=action.dtype)
                control_tick = int(round(float(public_clock[lane]) / .01))
                if isinstance(self.landing_mpc, LandingSupportMPCV3):
                    self.landing_mpc.observe(state, joint_ref, lane=lane,
                                             control_tick=control_tick)
                release_for_mode_mismatch = False
                release_for_tracking = False
                mode_mismatch = False
                measured_hold_force = None
                if (self.landing_mpc_hold_first_action and
                        self._landing_mpc_hold_remaining[lane].item() > 0 and
                        self.landing_mpc_mode_validation_force > 0. and
                        self._landing_mpc_expected_contacts[lane].any() and
                        self._landing_mpc_native_env is not None and
                        hasattr(self._landing_mpc_native_env, 'feet_contact')):
                    measured_hold_force = self._landing_mpc_native_env.feet_contact.data.net_forces_w[
                        lane].detach().cpu().numpy()
                    if measured_hold_force.shape != (4, 3) or not np.isfinite(measured_hold_force).all():
                        raise ValueError('invalid native foot force during held-mode validation')
                    geometric_hold = self._landing_mpc_native_env.observed_foot_contact[
                        lane].detach().cpu().numpy().astype(bool)
                    expected = self._landing_mpc_expected_contacts[lane]
                    mode_mismatch = bool(np.any(
                        expected & (~geometric_hold | (measured_hold_force[:, 2] <
                                                       self.landing_mpc_mode_validation_force))))
                    if mode_mismatch:
                        self._landing_mpc_mode_mismatch[lane] += 1
                    else:
                        self._landing_mpc_mode_mismatch[lane] = 0
                    release_for_mode_mismatch = bool(
                        self._landing_mpc_mode_mismatch[lane].item() >=
                        self.landing_mpc_mode_validation_ticks)
                if (self.landing_mpc_hold_tracking_error_release_rad > 0. and
                        self._landing_mpc_hold_remaining[lane].item() > 0 and
                        self._landing_mpc_native_env is not None and
                        np.isfinite(self._landing_mpc_last_target[lane]).all()):
                    actual_q = self._landing_mpc_native_env.robot.data.joint_pos[
                        lane, self._landing_mpc_native_env.joint_ids].detach().cpu().numpy()
                    if actual_q.shape == (18,) and np.isfinite(actual_q).all():
                        tracking_error_now = float(np.max(np.abs(actual_q-self._landing_mpc_last_target[lane])))
                        release_for_tracking = bool(
                            tracking_error_now > self.landing_mpc_hold_tracking_error_release_rad)
                if (self.landing_mpc_hold_first_action and
                        self._landing_mpc_hold_remaining[lane].item() > 0 and
                        not release_for_mode_mismatch and not release_for_tracking):
                    action[lane]=self._landing_mpc_held_action[lane].to(action.dtype)
                    self._landing_mpc_hold_remaining[lane]-=1
                    tracking_error = None
                    prediction_error = None
                    if self._landing_mpc_native_env is not None and np.isfinite(self._landing_mpc_last_target[lane]).all():
                        actual_q = self._landing_mpc_native_env.robot.data.joint_pos[
                            lane, self._landing_mpc_native_env.joint_ids].detach().cpu().numpy()
                        tracking_error = float(np.max(np.abs(actual_q-self._landing_mpc_last_target[lane])))
                        if (control_tick == self._landing_mpc_prediction_tick[lane] and
                                np.isfinite(self._landing_mpc_last_predicted[lane]).all()):
                            prediction_error = float(np.max(np.abs(actual_q-self._landing_mpc_last_predicted[lane])))
                    self.landing_mpc_logs.append(dict(case=case,episode_index=episode_index,
                        control_tick=control_tick,clock_s=float(public_clock[lane]),
                        accepted=True,first_action_available=True,applied_first_step=True,
                        held_first_action=True,hold_active=True,solver_failure=False,
                        constraint_failure=False,risk_trigger=True,risk_reason='held_mpc_action',
                        state_source='held_mpc_action',native_execution=True,
                        expected_support_contacts=self._landing_mpc_expected_contacts[lane].tolist(),
                        mode_validation_force_n=self.landing_mpc_mode_validation_force,
                        mode_validation_mismatch=mode_mismatch,
                        mode_validation_mismatch_ticks=int(self._landing_mpc_mode_mismatch[lane].item()),
                        mode_validation_released=False,
                        tracking_error_released=False,
                        measured_hold_force_world=(measured_hold_force.tolist()
                                                   if measured_hold_force is not None else None),
                        command_tracking_error_max_rad=tracking_error,
                        prediction_error_scope="next-control-tick-only; null-if-unmatched",
                        prediction_joint_position_error_max_rad=prediction_error))
                    continue
                if release_for_mode_mismatch or release_for_tracking:
                    self._landing_mpc_hold_remaining[lane]=0
                    self._landing_mpc_mode_mismatch[lane]=0
                command_tracking_error = None
                prediction_error = None
                if (self._landing_mpc_native_env is not None and
                        np.isfinite(self._landing_mpc_last_target[lane]).all()):
                    actual_q = self._landing_mpc_native_env.robot.data.joint_pos[
                        lane, self._landing_mpc_native_env.joint_ids].detach().cpu().numpy()
                    command_tracking_error = float(np.max(np.abs(actual_q-self._landing_mpc_last_target[lane])))
                    if (control_tick == self._landing_mpc_prediction_tick[lane] and
                                np.isfinite(self._landing_mpc_last_predicted[lane]).all()):
                        prediction_error = float(np.max(np.abs(actual_q-self._landing_mpc_last_predicted[lane])))
                original_action=np.asarray(action[lane].detach().cpu().numpy(),dtype=np.float64).copy()
                # The native trial must exercise the declared risk trigger.  A
                # selected TRAIN case is only an admission filter; it is not a
                # reason to keep overwriting a stable four-foot command.
                force_post_touchdown = bool(self.landing_mpc_force_post_touchdown and
                                            self.landing_mpc_touchdown_recovery and
                                            active[lane].item())
                if isinstance(self.landing_mpc, LandingSupportMPCV3):
                    plan=self.landing_mpc.solve(state,joint_ref,force=force_post_touchdown,
                                               lane=lane,control_tick=control_tick)
                else:
                    plan=self.landing_mpc.solve(state,joint_ref,force=force_post_touchdown)
                horizon=plan.get('horizon',[])
                failed_stage=next((row for row in horizon if not row.get('accepted')),None)
                predicted_contact_additions=[
                    dict(stage=int(row.get('stage',-1)), feet=list(row.get('predicted_contact_additions',[])))
                    for row in horizon if row.get('predicted_contact_additions')]
                contact_mode_additions=[
                    dict(stage=int(row.get('stage',-1)), feet=list(row.get('contact_mode_additions',[])))
                    for row in horizon if row.get('contact_mode_additions')]
                predicted_touchdown_impacts=[
                    dict(stage=int(row.get('stage',-1)), impact=row.get('predicted_touchdown_impact'))
                    for row in horizon if row.get('predicted_touchdown_impact')]
                record=dict(case=case,episode_index=episode_index,control_tick=control_tick,
                    clock_s=float(public_clock[lane]),accepted=bool(plan['accepted']),
                    first_action_available=bool(plan.get('horizon')) and bool(plan['horizon'][0].get('accepted')),
                    solver_failure=bool(plan.get('solver_failure',False)),rejection=plan.get('rejection'),
                    constraint_failure=bool(plan.get('constraint_failure',False)),
                    risk_trigger=bool(plan.get('risk',{}).get('trigger',False)),
                    force_post_touchdown=force_post_touchdown,
                    risk_reason=plan.get('risk',{}).get('reason'),
                    first_step_improves=bool(plan.get('first_step_improves',False)),
                    initial_risk_score=plan.get('initial_risk_score'),
                    first_step_risk_score=plan.get('first_step_risk_score'),
                    command_cost_selection=plan.get('command_cost_selection'),
                    hold_active=bool(self._landing_mpc_hold_remaining[lane].item() > 0),
                    runtime_ms=float(plan.get('wall_runtime_s',0.)*1000.),terminal_risk=plan.get('terminal_risk'),
                    implicit_prediction_check_count=len(plan.get("implicit_prediction_checks",[])),
                    implicit_prediction_maxima={key:max(check[key] for check in plan["implicit_prediction_checks"])
                        for key in plan.get("implicit_prediction_checks",[{}])[0]}
                        if plan.get("implicit_prediction_checks") else {},
                    horizon_steps=len(horizon),
                    failed_stage=failed_stage.get('stage') if failed_stage is not None else None,
                    failed_solver_status=failed_stage.get('solver_status') if failed_stage is not None else None,
                    failed_solver_iterations=failed_stage.get('solver_iterations') if failed_stage is not None else None,
                    failed_contacts=failed_stage.get('contacts') if failed_stage is not None else None,
                    failed_stage_risk=failed_stage.get('risk') if failed_stage is not None else None,
                    failed_stage_residuals=failed_stage.get('residuals') if failed_stage is not None else None,
                    predicted_contact_additions=predicted_contact_additions,
                    contact_mode_additions=contact_mode_additions,
                    predicted_touchdown_impacts=predicted_touchdown_impacts,
                    state_source=state_source,
                    requested_student_joint_reference=requested_student_joint_ref.tolist(),
                    previous_dispatched_target=(state.previous_command_target.tolist()
                        if self.landing_mpc_command_history_interface else None),
                    expected_support_contacts=self._landing_mpc_expected_contacts[lane].tolist(),
                    mode_validation_force_n=self.landing_mpc_mode_validation_force,
                    mode_validation_released=release_for_mode_mismatch,
                    tracking_error_released=release_for_tracking,
                    # Legacy key retained for old readers; this is force-norm
                    # hysteresis, not independent geometric contact evidence.
                    geometric_contacts=(getattr(state, 'geometric_contacts', state.contacts).tolist()),
                    observed_force_contacts=(getattr(state, 'geometric_contacts', state.contacts).tolist()),
                    contact_observation_semantics='native-force-norm-hysteresis-1N-on-0.5N-off',
                    support_contacts=state.contacts.tolist(),
                    latched_support_contacts=state.contacts.tolist(),
                    measured_foot_force_world=(getattr(state, 'measured_force_world', None).tolist()
                                               if getattr(state, 'measured_force_world', None) is not None else None),
                    command_tracking_error_max_rad=command_tracking_error,
                    prediction_error_scope="next-control-tick-only; null-if-unmatched",
                        prediction_joint_position_error_max_rad=prediction_error,
                    native_execution=True)
                risk_drop = (float(plan['initial_risk_score']) - float(plan['first_step_risk_score'])
                    if plan.get('initial_risk_score') is not None and
                    plan.get('first_step_risk_score') is not None else None)
                first_step_sufficiently_improves = bool(
                    risk_drop is not None and risk_drop >= self.landing_mpc_min_risk_score_drop and
                    plan.get('initial_risk_score') is not None and
                    float(plan['initial_risk_score']) >= self.landing_mpc_min_start_risk_score)
                hold_active = bool(self._landing_mpc_hold_remaining[lane].item() > 0)
                forced_first_step = bool(force_post_touchdown and record['first_action_available'])
                if plan.get('accepted') or forced_first_step or (self.landing_mpc_allow_first_step and
                                            record['first_action_available'] and
                                            record['risk_trigger'] and
                                            (first_step_sufficiently_improves or hold_active)):
                    requested_target=np.asarray(plan['horizon'][0]['joint_position_target'],dtype=np.float64)
                    # The support QP predicts this same one-step target
                    # admission during executor replay.  Apply it before
                    # converting back to the native normalized absolute-
                    # position action; otherwise Isaac receives a target jump
                    # that the prediction never simulated.
                    if self._landing_mpc_native_env is not None and hasattr(self.landing_mpc, 'normalized_command_target'):
                        admitted_target=self.landing_mpc.admit_command_target(state, requested_target)
                        chosen=np.asarray(self.landing_mpc.qp.interface.normalized_action(admitted_target),
                                          dtype=np.float32)
                    else:
                        admitted_target=requested_target
                        chosen=np.asarray(plan['horizon'][0]['normalized_action'],dtype=np.float32)
                    # A torque-overlay trial keeps the student's position
                    # target and adds only the MPC torque residual.  Replacing
                    # the target first would make the residual identically
                    # zero because the QP target is itself PD-inverted.
                    if not self.landing_mpc_torque_overlay:
                        action[lane]=torch.as_tensor(chosen,device=action.device)
                    chosen_target = lim[:,0] + .5*(chosen+1.)*(lim[:,1]-lim[:,0])
                    record['requested_joint_position_target_rad']=requested_target.tolist()
                    record['admitted_joint_position_target_rad']=chosen_target.tolist()
                    record['requested_target_delta_max_rad']=float(np.max(np.abs(requested_target-state.joint_position)))
                    record['admitted_target_delta_max_rad']=float(np.max(np.abs(chosen_target-state.joint_position)))
                    self._landing_mpc_last_target[lane,:] = chosen_target
                    first_stage=plan['horizon'][0]
                    predicted_joint_position=np.asarray(
                        first_stage.get('predicted_joint_position', state.joint_position), dtype=np.float64)
                    self._landing_mpc_last_predicted[lane,:] = predicted_joint_position
                    # A one-step prediction can be compared only at its target
                    # tick, never throughout a 40-tick held-action interval.
                    self._landing_mpc_prediction_tick[lane] = (
                        control_tick + 1 if not self.landing_mpc_torque_overlay else -1)
                    record['applied_first_step']=True
                    record['forced_first_step']=forced_first_step
                    record['first_step_risk_score_drop']=risk_drop
                    record['predicted_first_joint_position_rad']=predicted_joint_position.tolist()
                    if self.landing_mpc_hold_steps:
                        self._landing_mpc_hold_remaining[lane]=self.landing_mpc_hold_steps
                        if self.landing_mpc_hold_first_action:
                            self._landing_mpc_held_action[lane]=torch.as_tensor(
                                chosen,device=action.device,dtype=self._landing_mpc_held_action.dtype)
                            self._landing_mpc_expected_contacts[lane,:] = np.asarray(
                                first_stage.get('contacts', state.contacts.tolist()), dtype=bool)
                            self._landing_mpc_mode_mismatch[lane]=0
                    record['torque_overlay_applied']=False
                    first_plan_torque = plan.get('torque')
                    if first_plan_torque is None and plan.get('horizon'):
                        first_plan_torque = plan['horizon'][0].get('torque')
                    if self.landing_mpc_torque_overlay and first_plan_torque is not None:
                        # The QP torque is the desired total leg torque.  The
                        # implicit actuator adds effort target to its position
                        # PD torque, so send only the residual.  Arm torques
                        # remain zero in the first native trial to protect the
                        # racket/contact event.
                        target=lim[:,0]+.5*(original_action+1.)*(lim[:,1]-lim[:,0])
                        q=state.joint_position;v=state.joint_velocity
                        kp=np.asarray(self.landing_mpc.qp.interface.kp,dtype=np.float64)
                        kd=np.asarray(self.landing_mpc.qp.interface.kd,dtype=np.float64)
                        total=np.asarray(first_plan_torque,dtype=np.float64)
                        residual=total-(kp*(target-q)-kd*v)
                        self._effort_override[lane,:12]=torch.as_tensor(residual[:12],device=action.device,dtype=action.dtype)
                        self._effort_override_mask[lane]=True
                        record['torque_overlay_applied']=True
                        record['torque_overlay_residual_max']=float(np.max(np.abs(residual[:12])))
                else:
                    record['applied_first_step']=False
                    record['first_step_risk_score_drop']=risk_drop
                    if hold_active:
                        self._landing_mpc_hold_remaining[lane]-=1
                if self.landing_mpc_command_history_interface:
                    final_target=lim[:,0]+.5*(action[lane].detach().cpu().numpy()+1.)*(lim[:,1]-lim[:,0])
                    arm_step=np.abs(final_target[12:]-state.previous_command_target[12:])
                    record['final_dispatched_joint_target_rad']=final_target.tolist()
                    record['arm_command_rate_rad_s']=self.landing_mpc.qp.interface.arm_rate
                    record['arm_target_step_max_rad']=float(arm_step.max())
                    record['arm_target_slew_violation_rad']=float(max(0.,
                        arm_step.max()-self.landing_mpc.qp.interface.arm_rate*self.landing_mpc.settings.control_dt_s))
                self.landing_mpc_logs.append(record)
            except Exception as error:
                self.landing_mpc_logs.append(dict(case=case,episode_index=episode_index,
                    clock_s=float(public_clock[lane]),accepted=False,
                    first_action_available=False,applied_first_step=False,solver_failure=True,
                    rejection='exception:'+repr(error),state_source=(
                        'native_sim_state' if self._landing_mpc_native_env is not None
                        else 'public_observation_reconstruction'),native_execution=True))
        return action


class CaseExpertBankController:
    """Select one measured expert after the first causal ball observation.

    The bank is deliberately held fixed for an episode. It is a controller
    selection experiment over the fixed TRAIN panel, not a formal policy score.
    All experts share the candidate's native base/controller and action mixer.
    """
    def __init__(self, candidate, manifest_path, count, device):
        self.candidate_controller = SharedTargetResidualController(candidate, count, device)
        manifest = json.loads(Path(manifest_path).read_text())
        if manifest.get('schema') != 'case-conditioned-native-expert-bank-v1':
            raise ValueError('Unexpected expert-bank manifest schema')
        self.manifest = manifest
        self.features = torch.as_tensor(
            [row['feature'] for row in manifest['cases']], dtype=torch.float32, device=device)
        self.expert_paths = [row['expert_policy'] for row in manifest['cases']]
        self.experts = []
        for path in sorted(set(x for x in self.expert_paths if x)):
            source = torch.load(path, map_location='cpu', weights_only=False)
            if source.get('schema') != RESIDUAL_SCHEMA or not source.get('native_execution'):
                raise ValueError('Expert bank contains a non-native residual policy')
            if source.get('residual_joint_scale_rad') != candidate.get('residual_joint_scale_rad'):
                raise ValueError('Expert bank action scales differ from candidate')
            if source.get('residual_training_metadata', {}).get('base_policy_sha256') != \
                    candidate.get('residual_training_metadata', {}).get('base_policy_sha256'):
                raise ValueError('Expert bank base policy differs from candidate')
            self.experts.append((path, torch.jit.load(
                io.BytesIO(source['residual_actor_jit']), map_location=device).eval()))
        path_to_index = {path: i for i, (path, _) in enumerate(self.experts)}
        self.case_expert_indices = torch.as_tensor(
            [path_to_index.get(path, -1) for path in self.expert_paths],
            dtype=torch.long, device=device)
        self.selected = torch.full((count,), -1, dtype=torch.long, device=device)
        self.selected_case = torch.full((count,), -1, dtype=torch.long, device=device)

    @property
    def reference_controller(self): return self.candidate_controller.reference_controller
    @property
    def logs(self): return self.candidate_controller.logs
    @logs.setter
    def logs(self, value): self.candidate_controller.logs = value
    @property
    def base(self): return self.candidate_controller.base
    @property
    def mixer(self): return self.candidate_controller.mixer
    @property
    def landing_config(self): return self.candidate_controller.landing_config
    @property
    def support_config(self): return self.candidate_controller.support_config

    def reset(self, ids=None):
        self.candidate_controller.reset(ids)
        if ids is None:
            self.selected.fill_(-1); self.selected_case.fill_(-1)
        else:
            self.selected[ids] = -1; self.selected_case[ids] = -1

    @torch.inference_mode()
    def action(self, observation, default, limits):
        raw = observation['policy']
        base = self.candidate_controller.base.action(observation, default, limits)
        present = raw[:, 67] > .5
        choose = (self.selected < 0) & present
        if bool(choose.any()):
            feature = torch.cat((raw[choose, 61:67], raw[choose, 759:770], raw[choose, 775:776]), dim=1)
            distance = ((feature[:, None, :] - self.features[None, :, :]) ** 2).sum(-1)
            case = distance.argmin(1)
            expert = self.case_expert_indices[case]
            self.selected[choose] = expert
            self.selected_case[choose] = case
        residual = torch.zeros_like(base)
        for expert_index, (_, actor) in enumerate(self.experts):
            mask = self.selected == expert_index
            if bool(mask.any()): residual[mask] = actor(raw[mask])
        fallback = self.selected < 0
        if bool(fallback.any()): residual[fallback] = self.candidate_controller.actor(raw[fallback])
        return self.candidate_controller.mixer.action(base, residual, raw, default, limits)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


storage_reserve=None

def main():
    global storage_reserve
    args.output.mkdir(parents=True, exist_ok=True)
    from scripts.native_output_reserve_v1 import NativeOutputReserve
    storage_reserve=NativeOutputReserve(args.output)
    if LANDING_BLAS_RUNTIME is not None:
        atomic_json(args.output / 'landing_blas_runtime.json', LANDING_BLAS_RUNTIME)
    if (args.output / 'INPUTS.json').exists():
        raise ValueError('Use a fresh output; never overwrite a previous diagnostic')
    protocol_path = ROOT / 'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'
    protocol = json.loads(protocol_path.read_text())
    cases, indices = training_cases(ROOT, protocol, args.training_indices)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint.get('schema') not in (RESIDUAL_SCHEMA,SHARED_SCHEMA) or checkpoint.get('protocol_sha256') != digest(protocol_path):
        raise ValueError('Require the explicitly measured shared-target residual policy')
    if args.learned_recovery_mpc_correction:
        if (args.learned_recovery_command_head is None or args.landing_mpc_cases!=list(range(128)) or
                not args.landing_mpc_command_history_interface or not args.landing_mpc_continuous_reference or
                args.landing_mpc_torque_overlay):
            raise ValueError('Correction collection requires learned head, all128 lanes and continuous command-history MPC without torque overlay')
    if args.landing_mpc_causal_posthit_only:
        if (args.learned_recovery_command_head or args.learned_recovery_mpc_correction or
                args.landing_mpc_cases != list(range(128)) or args.reactive_vhsip_cases is not None or
                not args.landing_mpc_command_history_interface or not args.record_first_episodes):
            raise ValueError('Causal baseline trial requires all128 command-history MPC, traces and no recovery learner/VHSIP')
    if (not args.allow_declared_controller_variant or args.learned_recovery_command_head is not None
            or args.landing_mpc_cases is not None or args.reactive_vhsip_cases is not None
            or args.expert_bank is not None or args.landing_config is not None
            or args.support_config is not None or args.disable_support):
        raise ValueError('Sequence diagnostic requires exact original student and no other new teacher')
    learned_recovery_checkpoint = None
    if args.learned_recovery_command_head is not None:
        if not args.allow_declared_controller_variant:
            raise ValueError('Learned command head requires an explicit controller variant')
        if ((args.landing_mpc_cases is not None and not args.learned_recovery_mpc_correction) or args.reactive_vhsip_cases is not None or
                args.expert_bank is not None or args.landing_config is not None or
                args.support_config is not None or args.disable_support):
            raise ValueError('Evaluate learned recovery with unchanged base and no extra teacher')
        learned_recovery_checkpoint = torch.load(args.learned_recovery_command_head,
            map_location='cpu', weights_only=False)
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
    if not has_actor and not args.teacher_only:
        raise ValueError('Teacher-only or residual actor execution must be explicit')
    if args.teacher_only and (args.learned_recovery_command_head is not None or args.learned_recovery_mpc_correction or args.expert_bank is not None):
        raise ValueError('Teacher-only forbids learned recovery heads, correction collection and expert banks')
    if (not has_actor or args.teacher_only) and ((args.landing_config or args.support_config or args.disable_support)
                         and not args.allow_declared_controller_variant):
        raise ValueError("Require the actually trained residual actor without an undeclared controller change")
    if args.expert_bank is not None and not args.allow_declared_controller_variant:
        raise ValueError('Expert-bank selection requires --allow-declared-controller-variant')
    if args.landing_mpc_cases is not None:
        if args.allow_declared_controller_variant is not True:
            raise ValueError('Landing MPC selection requires --allow-declared-controller-variant')
        if any(int(x)<0 or int(x)>=count for x in args.landing_mpc_cases):
            raise ValueError('Landing MPC cases must be fixed TRAIN indices in0..127')
        if args.expert_bank is not None:
            raise ValueError('Keep landing MPC trial separate from expert-bank selection')
    if args.reactive_vhsip_cases is not None:
        if args.allow_declared_controller_variant is not True:
            raise ValueError('Reactive VHSIP selection requires --allow-declared-controller-variant')
        if any(int(x)<0 or int(x)>=count for x in args.reactive_vhsip_cases):
            raise ValueError('Reactive VHSIP cases must be fixed TRAIN indices in0..127')
        if args.expert_bank is not None or args.landing_mpc_cases is not None:
            raise ValueError('Keep reactive VHSIP trial separate from other controller trials')
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
    controller_trial=bool(args.persistent_recovery_trial or args.expert_bank is not None or
                          args.landing_mpc_cases is not None or args.reactive_vhsip_cases is not None)
    noise = training_noise(count, amplitudes, args.seed)
    atomic_json(args.output / 'INPUTS.json', dict(policy_sha256=digest(args.checkpoint),
        protocol_sha256=digest(protocol_path), training_suite=protocol['training_suite'],
        source_checkpoint_protocol_sha256=checkpoint['protocol_sha256'],
        execution_worktree=str(ROOT),
        origin_precision_receipt_sha256=digest(args.precision_receipt),origin_spacing_m=0.,
        training_indices=indices, reset_noise_amplitudes=amplitudes, reset_noise_seed=args.seed,
        controller_trial=controller_trial, ground_extent_diagnostic=ground_diagnostic,
        learned_recovery_command_head=(str(args.learned_recovery_command_head.resolve())
            if args.learned_recovery_command_head else None),
        learned_recovery_command_head_sha256=(digest(args.learned_recovery_command_head)
            if args.learned_recovery_command_head else None),
        extra_recovery_teacher_enabled=bool(args.learned_recovery_mpc_correction or args.landing_mpc_causal_posthit_only) if (args.learned_recovery_command_head or args.landing_mpc_causal_posthit_only) else None,
        learned_recovery_mpc_correction=bool(args.learned_recovery_mpc_correction),
        landing_mpc_causal_posthit_only=bool(args.landing_mpc_causal_posthit_only),
        student_sampling_log_prob_available=False,correction_collection_is_ppo_buffer=False,
        recovery_assistance_execution=RECOVERY_EXECUTION_SCHEMA,
        landing_trial=landing_trial,support_trial=support_trial,
        support_config=effective_support,support_disabled=args.disable_support,
        record_all_first_episode_actions=args.record_first_episodes,
        residual_actor_enabled=has_actor,
        landing_mpc_cases=args.landing_mpc_cases,
        landing_mpc_horizon_steps=args.landing_mpc_horizon_steps,
        landing_mpc_allow_first_step=args.landing_mpc_allow_first_step,
        landing_mpc_min_risk_score_drop=args.landing_mpc_min_risk_score_drop,
        landing_mpc_min_start_risk_score=args.landing_mpc_min_start_risk_score,
        landing_mpc_hold_steps=args.landing_mpc_hold_steps,
        landing_mpc_hold_first_action=args.landing_mpc_hold_first_action,
        landing_mpc_executor_replay=args.landing_mpc_executor_replay,
        landing_mpc_command_rate_scale=args.landing_mpc_command_rate_scale,
        landing_mpc_support_margin=args.landing_mpc_support_margin,
        landing_mpc_torque_overlay=args.landing_mpc_torque_overlay,
        landing_mpc_touchdown_recovery=args.landing_mpc_touchdown_recovery,
        landing_mpc_force_post_touchdown=args.landing_mpc_force_post_touchdown,
        landing_mpc_latched_post_touchdown_ticks=args.landing_mpc_latched_post_touchdown_ticks,
        landing_mpc_contact_force_threshold=args.landing_mpc_contact_force_threshold,
        landing_mpc_contact_release_threshold=args.landing_mpc_contact_release_threshold,
        landing_mpc_contact_release_ticks=args.landing_mpc_contact_release_ticks,
        landing_mpc_mode_validation_force=args.landing_mpc_mode_validation_force,
        landing_mpc_mode_validation_ticks=args.landing_mpc_mode_validation_ticks,
        landing_mpc_hold_tracking_error_release_rad=args.landing_mpc_hold_tracking_error_release_rad,
        landing_mpc_predict_touchdown_contacts=args.landing_mpc_predict_touchdown_contacts,
        landing_mpc_contact_mode_fallback=args.landing_mpc_contact_mode_fallback,
        landing_mpc_contact_mode_sequence_search=args.landing_mpc_contact_mode_sequence_search,
        landing_mpc_contact_mode_sequence_max_candidates=args.landing_mpc_contact_mode_sequence_max_candidates,
        landing_mpc_touchdown_mode_height_scale=args.landing_mpc_touchdown_mode_height_scale,
        landing_mpc_capture_foot_gain=args.landing_mpc_capture_foot_gain,
        landing_mpc_capture_foot_max_offset=args.landing_mpc_capture_foot_max_offset,
        landing_mpc_leg_posture_blend=args.landing_mpc_leg_posture_blend,
        landing_mpc_arm_posture_blend=args.landing_mpc_arm_posture_blend,
        landing_mpc_touchdown_height_tolerance=args.landing_mpc_touchdown_height_tolerance,
        landing_mpc_touchdown_vertical_speed_limit=args.landing_mpc_touchdown_vertical_speed_limit,
                      landing_mpc_touchdown_impact_projection=args.landing_mpc_touchdown_impact_projection,
                      landing_mpc_force_distribution_weight=args.landing_mpc_force_distribution_weight,
                      landing_mpc_force_wrench_weight=args.landing_mpc_force_wrench_weight,
        landing_mpc_swing_foot_weight=args.landing_mpc_swing_foot_weight,
        landing_mpc_swing_foot_kp=args.landing_mpc_swing_foot_kp,
        landing_mpc_swing_foot_kd=args.landing_mpc_swing_foot_kd,
        landing_mpc_swing_foot_clearance=args.landing_mpc_swing_foot_clearance,
        landing_mpc_nonfoot_clearance_weight=args.landing_mpc_nonfoot_clearance_weight,
        landing_mpc_nonfoot_clearance_hard=args.landing_mpc_nonfoot_clearance_hard,
        landing_mpc_nonfoot_clearance_m=args.landing_mpc_nonfoot_clearance_m,
        landing_mpc_nonfoot_clearance_kp=args.landing_mpc_nonfoot_clearance_kp,
        landing_mpc_nonfoot_clearance_kd=args.landing_mpc_nonfoot_clearance_kd,
        landing_mpc_continuous_reference=args.landing_mpc_continuous_reference,
        landing_mpc_load_aware_support=args.landing_mpc_load_aware_support,
        landing_mpc_load_force_floor=args.landing_mpc_load_force_floor,
        landing_mpc_free_arm=args.landing_mpc_free_arm,
        landing_mpc_command_history_interface=args.landing_mpc_command_history_interface,
        landing_mpc_implicit_command_qp=args.landing_mpc_implicit_command_qp,
        landing_mpc_interval_command_qp=args.landing_mpc_interval_command_qp,
        landing_mpc_interval_active_set=args.landing_mpc_interval_active_set,
        landing_mpc_full_command_damper=args.landing_mpc_full_command_damper,
        landing_mpc_minimum_command_cost=args.landing_mpc_minimum_command_cost,
        landing_mpc_command_change_weight=args.landing_mpc_command_change_weight,
        reactive_vhsip_cases=args.reactive_vhsip_cases,
        reactive_vhsip_post_touchdown_only=args.reactive_vhsip_post_touchdown_only,
        reactive_vhsip_capture_model=args.reactive_vhsip_capture_model,
        reactive_vhsip_settling_s=args.reactive_vhsip_settling_s,
        reactive_vhsip_capture_gain=args.reactive_vhsip_capture_gain,
        reactive_vhsip_blend_s=args.reactive_vhsip_blend_s,
        pure_control_configuration=str(args.pure_control_configuration.resolve()),
        causal_release_full_control=True,case_time_conditioned_diagnostic=False,
        expert_bank=str(args.expert_bank.resolve()) if args.expert_bank else None,
        offline_trace_finetune_candidate=offline_finetune,
        reset_noise={k: v.tolist() for k, v in noise.items()},
        scope='TRAIN-only numerical ground-extent diagnostic; original task scoring, no inherited performance or formal cases.',
        formal_cases_read=False, formal_acceptance=False))
    execution_snapshot=snapshot_execution_sources(ROOT,args.output)
    (args.output/'sequence_evaluator.py').write_text(Path(__file__).read_text())
    atomic_json(args.output/'sequence_entry_provenance.json',dict(
        path=str(Path(__file__).resolve()),sha256=digest(Path(__file__))))
    cfg = full_racket_origin0_cfg(count, protocol, args.precision_receipt)
    # CPU engineering mode: preserve the task/controller protocol while
    # routing the IsaacLab simulation tensors away from the GPU job.
    if args.device == 'cpu':
        cfg.sim.device = 'cpu'
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
    controller = (CaseExpertBankController(checkpoint, args.expert_bank, count, env.device)
                  if args.expert_bank is not None else
                  SharedTargetResidualController(checkpoint, count, env.device,landing_trial,support_trial,args.disable_support,
                      landing_mpc_cases=args.landing_mpc_cases,
                      landing_mpc_horizon_steps=args.landing_mpc_horizon_steps,
                      landing_mpc_allow_first_step=args.landing_mpc_allow_first_step,
                      landing_mpc_min_risk_score_drop=args.landing_mpc_min_risk_score_drop,
                      landing_mpc_min_start_risk_score=args.landing_mpc_min_start_risk_score,
                      landing_mpc_hold_steps=args.landing_mpc_hold_steps,
                      landing_mpc_hold_first_action=args.landing_mpc_hold_first_action,
                      landing_mpc_executor_replay=args.landing_mpc_executor_replay,
                      landing_mpc_command_rate_scale=args.landing_mpc_command_rate_scale,
                      landing_mpc_support_margin=args.landing_mpc_support_margin,
                      landing_mpc_torque_overlay=args.landing_mpc_torque_overlay,
                      landing_mpc_touchdown_recovery=args.landing_mpc_touchdown_recovery,
                      landing_mpc_force_post_touchdown=args.landing_mpc_force_post_touchdown,
                      landing_mpc_latched_post_touchdown_ticks=args.landing_mpc_latched_post_touchdown_ticks,
                      landing_mpc_contact_force_threshold=args.landing_mpc_contact_force_threshold,
                      landing_mpc_contact_release_threshold=args.landing_mpc_contact_release_threshold,
                      landing_mpc_contact_release_ticks=args.landing_mpc_contact_release_ticks,
                      landing_mpc_mode_validation_force=args.landing_mpc_mode_validation_force,
                      landing_mpc_mode_validation_ticks=args.landing_mpc_mode_validation_ticks,
                      landing_mpc_hold_tracking_error_release_rad=args.landing_mpc_hold_tracking_error_release_rad,
        landing_mpc_predict_touchdown_contacts=args.landing_mpc_predict_touchdown_contacts,
                      landing_mpc_contact_mode_fallback=args.landing_mpc_contact_mode_fallback,
                      landing_mpc_contact_mode_sequence_search=args.landing_mpc_contact_mode_sequence_search,
                      landing_mpc_contact_mode_sequence_max_candidates=args.landing_mpc_contact_mode_sequence_max_candidates,
                      teacher_only=args.teacher_only,
                      landing_mpc_touchdown_mode_height_scale=args.landing_mpc_touchdown_mode_height_scale,
                      landing_mpc_capture_foot_gain=args.landing_mpc_capture_foot_gain,
                      landing_mpc_capture_foot_max_offset=args.landing_mpc_capture_foot_max_offset,
                      landing_mpc_leg_posture_blend=args.landing_mpc_leg_posture_blend,
                      landing_mpc_arm_posture_blend=args.landing_mpc_arm_posture_blend,
        landing_mpc_touchdown_height_tolerance=args.landing_mpc_touchdown_height_tolerance,
        landing_mpc_touchdown_vertical_speed_limit=args.landing_mpc_touchdown_vertical_speed_limit,
        landing_mpc_touchdown_impact_projection=args.landing_mpc_touchdown_impact_projection,
        landing_mpc_force_distribution_weight=args.landing_mpc_force_distribution_weight,
        landing_mpc_force_wrench_weight=args.landing_mpc_force_wrench_weight,
                      landing_mpc_swing_foot_weight=args.landing_mpc_swing_foot_weight,
                      landing_mpc_swing_foot_kp=args.landing_mpc_swing_foot_kp,
                      landing_mpc_swing_foot_kd=args.landing_mpc_swing_foot_kd,
                      landing_mpc_swing_foot_clearance=args.landing_mpc_swing_foot_clearance,
        landing_mpc_nonfoot_clearance_weight=args.landing_mpc_nonfoot_clearance_weight,
        landing_mpc_nonfoot_clearance_hard=args.landing_mpc_nonfoot_clearance_hard,
        landing_mpc_nonfoot_clearance_m=args.landing_mpc_nonfoot_clearance_m,
        landing_mpc_nonfoot_clearance_kp=args.landing_mpc_nonfoot_clearance_kp,
        landing_mpc_nonfoot_clearance_kd=args.landing_mpc_nonfoot_clearance_kd,
                      landing_mpc_continuous_reference=args.landing_mpc_continuous_reference,
                      landing_mpc_load_aware_support=args.landing_mpc_load_aware_support,
                      landing_mpc_load_force_floor=args.landing_mpc_load_force_floor,
                      landing_mpc_free_arm=args.landing_mpc_free_arm,
        landing_mpc_command_history_interface=args.landing_mpc_command_history_interface,
        landing_mpc_implicit_command_qp=args.landing_mpc_implicit_command_qp,
        landing_mpc_interval_command_qp=args.landing_mpc_interval_command_qp,
        landing_mpc_interval_active_set=args.landing_mpc_interval_active_set,
        landing_mpc_full_command_damper=args.landing_mpc_full_command_damper,
        landing_mpc_minimum_command_cost=args.landing_mpc_minimum_command_cost,
        landing_mpc_command_change_weight=args.landing_mpc_command_change_weight,
                      reactive_vhsip_cases=args.reactive_vhsip_cases,
                      reactive_vhsip_post_touchdown_only=args.reactive_vhsip_post_touchdown_only,
                      reactive_vhsip_capture_model=args.reactive_vhsip_capture_model,
                      reactive_vhsip_settling_s=args.reactive_vhsip_settling_s,
                      reactive_vhsip_capture_gain=args.reactive_vhsip_capture_gain,
                      reactive_vhsip_blend_s=args.reactive_vhsip_blend_s))
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
        from scripts.native_landing_contact_observer_v2 import NativeLandingContactObserverV2 as NativePureControlObservationV1
        from scripts.native_pure_control_state_driven_long_mpc_v1_qp_continuity_repair import NativePureControlStateDrivenLongMPCV1QPContinuityRepair as NativePureControlRecoveryV1
        pure_observer=NativePureControlObservationV1(env,args.output)
        native_replay=NativePureControlRecoveryV1(env,limits,default,
            args.pure_control_configuration,args.output,pure_observer)
        sequence_student_commands=[];sequence_final_commands=[];sequence_first_episode=[]
        learned_recovery = None
        learned_records = []
        causal_teacher_records = []
        causal_teacher_gate = None
        if args.landing_mpc_causal_posthit_only:
            from quadsmash.control.observed_posthit_gate_v1 import ObservedPostHitGateV1
            causal_teacher_gate = ObservedPostHitGateV1(count, env.device, protocol['policy_dt_s'])
        if learned_recovery_checkpoint is not None:
            from quadsmash.control.learned_recovery_command_v1 import (
                admit_native_recovery_head, NativeRecoveryCommandAdapter)
            recovery_head = admit_native_recovery_head(learned_recovery_checkpoint,
                digest(args.checkpoint), digest(protocol_path), limits, protocol['policy_dt_s'])
            learned_recovery = NativeRecoveryCommandAdapter(recovery_head, count, env.device, limits)
        if hasattr(controller,'configure_landing_mpc'):
            controller.configure_landing_mpc(limits, default)
        atomic_json(args.output/'native_command_inputs.json',dict(
            default_joint_position_rad=default.cpu().tolist(),joint_limits_rad=limits.cpu().tolist(),
            protocol_sha256=digest(protocol_path),policy_sha256=digest(args.checkpoint)))
        finished = torch.zeros(count, dtype=torch.bool, device=env.device)
        poses, balls, clocks, valid = [], [], [], []
        public, actions, public_valid = [], [], []
        support_records={}
        landing_mpc_records=[]
        requested = round(protocol['episode_seconds']/protocol['policy_dt_s'])
        fixed_trace=None
        if args.record_first_episodes:
            from quadsmash.rl.fixed_policy_trace_v1 import FixedPolicyTraceV1
            fixed_trace=FixedPolicyTraceV1(args.output/'fixed_policy_traces',count,requested,
                dict(native_execution=True,source_kind='Teacher-only programmatic base/approach/WBC/recovery on explicitly recorded ground64',
                     ground_extent_diagnostic=ground_diagnostic,
                     policy_sha256=digest(args.checkpoint),protocol_sha256=digest(protocol_path),
                     asset_sha256=protocol['asset']['sha256'],
                     default_joint_position_rad=default.cpu().tolist(),joint_limits_rad=limits.cpu().tolist(),
                     controller_trial=controller_trial,
                     teacher_only=bool(args.teacher_only),
                     recovery_assistance_execution=RECOVERY_EXECUTION_SCHEMA,
                     support_config=controller.support_config,landing_config=controller.landing_config))
        with torch.inference_mode():
            for tick in range(requested):
                storage_reserve.check()
                obs = observation['policy']
                if hasattr(controller, 'set_landing_mpc_native_state'):
                    controller.set_landing_mpc_native_state(env)
                if args.learned_recovery_mpc_correction or causal_teacher_gate is not None:
                    action = controller.action(observation,default,limits,defer_landing_mpc=True)
                else:
                    action = controller.action(observation, default, limits)
                if causal_teacher_gate is not None:
                    base_command = action.clone()
                    eligible = causal_teacher_gate.update(obs) & (~finished)
                    action = controller.apply_landing_mpc(observation,default,limits,action,eligible_mask=eligible)
                    causal_teacher_records.append(dict(active=eligible.cpu().numpy().copy(),
                        base_command=base_command.cpu().numpy().copy(),
                        final_command=action.cpu().numpy().copy(),
                        first_episode=(~finished).cpu().numpy().copy()))
                if learned_recovery is not None:
                    base_command = action.clone()
                    if args.learned_recovery_mpc_correction:
                        from quadsmash.control.teacher_corrected_recovery_command_v1 import teacher_corrected_recovery_command
                        action,pair=teacher_corrected_recovery_command(learned_recovery,obs,base_command,
                            lambda proposal,eligible:controller.apply_landing_mpc(
                                observation,default,limits,proposal,eligible_mask=eligible))
                    else:
                        action = learned_recovery.action(obs, base_command)
                    learned_records.append(dict(
                        active=learned_recovery.active.cpu().numpy().copy(),
                        base_command=base_command.cpu().numpy().copy(),
                        learned_delta=learned_recovery.last_delta.cpu().numpy().copy(),
                        final_command=action.cpu().numpy().copy(),
                        first_episode=(~finished).cpu().numpy().copy()))
                    if args.learned_recovery_mpc_correction:
                        learned_records[-1].update({key:pair[key].cpu().numpy().copy() for key in
                            ('student_command','previous_command','teacher_command_delta','teacher_changed')})
                if hasattr(controller,'landing_mpc_diagnostics'):
                    landing_mpc_records.extend(controller.landing_mpc_diagnostics())
                sequence_student_commands.append(action.cpu().numpy().copy())
                action=native_replay.action(tick,obs,action,finished)
                sequence_final_commands.append(action.cpu().numpy().copy())
                sequence_first_episode.append((~finished).cpu().numpy().copy())
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
                # Recovery owns the total position command; no old torque overlay.
                env.robot.set_joint_effort_target(torch.zeros_like(action),joint_ids=env.joint_ids)
                observation, _, done, _ = wrapped.step(action)
                native_replay.verify_applied(finished)
                if learned_recovery is not None:
                    learned_recovery.reset(done.bool())
                if hasattr(controller, 'finish_landing_mpc_episodes'):
                    controller.finish_landing_mpc_episodes(done)
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
        if causal_teacher_records:
            np.savez_compressed(args.output/'causal_teacher_commands.npz',
                **{key:np.stack([row[key] for row in causal_teacher_records]) for key in causal_teacher_records[0]})
        if learned_records:
            np.savez_compressed(args.output/'learned_recovery_commands.npz',
                **{key:np.stack([row[key] for row in learned_records]) for key in learned_records[0]})
        storage_reserve.release()
        contact_observer.save()
        np.savez_compressed(args.output/'sequence_dispatch.npz',
            student=np.stack(sequence_student_commands),final=np.stack(sequence_final_commands),
            first_episode=np.stack(sequence_first_episode))
        sequence_result=native_replay.save()
        atomic_json(args.output / 'contacts.json', env.contact_records)
        atomic_json(args.output / 'causal_plans.json', controller.logs)
        if args.landing_mpc_cases is not None:
            atomic_json(args.output / 'landing_mpc_records.json', landing_mpc_records)
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
        summary["landing_blas_runtime"] = LANDING_BLAS_RUNTIME
        summary['pure_control_recovery']=sequence_result
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
                       teacher_only=bool(args.teacher_only), actor_forward_count=int(getattr(controller,'actor_forward_count',0)),
                       optimizer_step_count=int(getattr(controller,'optimizer_step_count',0)),
                       action_contributions=dict(base_actor=False,approach_head=False,recovery_head=False),
            noise_level=args.noise, noise_seed=args.seed, actual_control_env_steps=(tick+1)*count,
            controller_trial=controller_trial,support_disabled=args.disable_support,
            recovery_assistance_execution=RECOVERY_EXECUTION_SCHEMA,
            landing_config=controller.landing_config,support_config=controller.support_config,
            residual_actor_enabled=(has_actor and not args.teacher_only), ground_extent_diagnostic=ground_diagnostic)
        if causal_teacher_gate is not None:
            summary['causal_baseline_teacher'] = dict(extra_teacher_enabled=True,
                learned_recovery_head_enabled=False,case_id_or_future_outcome_as_input=False,
                gate='observed-airborne-hit-then-support-latched-until-reset',
                active_first_episode_ticks=int(sum(r['active'].sum() for r in causal_teacher_records)),
                sustained_until_episode_end=True,autonomous_risk_exit_implemented=False)
        if learned_recovery is not None:
            summary['learned_recovery_command'] = dict(
                schema='learned-native-recovery-command-v1',
                head_sha256=digest(args.learned_recovery_command_head),
                extra_mpc_or_vhsip_teacher_enabled=bool(args.learned_recovery_mpc_correction),
                collection_is_on_policy_ppo=False,
                teacher_changed_first_episode_ticks=(int(sum((r['teacher_changed'] & r['first_episode']).sum() for r in learned_records))
                    if args.learned_recovery_mpc_correction else 0),
                base_controller_and_persistent_recovery_retained=True,
                case_id_or_future_outcome_as_input=False,
                active_first_episode_ticks=int(sum((r['active'] & r['first_episode']).sum() for r in learned_records)))
        if args.reactive_vhsip_cases is not None:
            summary['reactive_vhsip'] = dict(schema='reactive-vhsip-position-interface-v1',
                selected_cases=[int(x) for x in args.reactive_vhsip_cases],
                post_touchdown_only=bool(args.reactive_vhsip_post_touchdown_only),
                capture_model=args.reactive_vhsip_capture_model,
                settling_s=float(args.reactive_vhsip_settling_s),
                capture_gain=float(args.reactive_vhsip_capture_gain),
                blend_s=float(args.reactive_vhsip_blend_s),
                controller_trial=True, native_execution=True, frozen_policy_performance=False,
                formal_acceptance=False)
        if args.landing_mpc_cases is not None:
            summary['landing_mpc'] = dict(schema=('landing-feasible-command-cost-v1' if args.landing_mpc_minimum_command_cost else 'landing-full-command-interval-v1' if args.landing_mpc_full_command_damper else 'landing-active-set-interval-mpc-v1'
                if args.landing_mpc_interval_active_set else 'landing-interval-command-mpc-v1'
                if args.landing_mpc_interval_command_qp else 'landing-implicit-command-mpc-v1'
                if args.landing_mpc_implicit_command_qp else 'landing-command-history-mpc-v1'
                if args.landing_mpc_command_history_interface else 'landing-support-mpc-v3-continuous-touchdown-recovery'
                if args.landing_mpc_continuous_reference and args.landing_mpc_touchdown_recovery else
                'landing-support-mpc-v3-continuous-reference' if args.landing_mpc_continuous_reference else
                'landing-support-mpc-v2-touchdown-recovery'
                if args.landing_mpc_touchdown_recovery else
                'landing-support-mpc-v2-support-margin' if args.landing_mpc_support_margin else
                'landing-support-mpc-v1'),controller_trial=True,
                selected_cases=[int(x) for x in args.landing_mpc_cases],
                horizon_steps=args.landing_mpc_horizon_steps,
                allow_first_step=bool(args.landing_mpc_allow_first_step),
                min_risk_score_drop=float(args.landing_mpc_min_risk_score_drop),
                min_start_risk_score=float(args.landing_mpc_min_start_risk_score),
                hold_steps=int(args.landing_mpc_hold_steps),
                hold_first_action=bool(args.landing_mpc_hold_first_action),
                executor_replay=bool(args.landing_mpc_executor_replay),
                command_rate_scale=float(args.landing_mpc_command_rate_scale),
                support_margin=bool(args.landing_mpc_support_margin),
                torque_overlay=bool(args.landing_mpc_torque_overlay),
                touchdown_recovery=bool(args.landing_mpc_touchdown_recovery),
                force_post_touchdown=bool(args.landing_mpc_force_post_touchdown),
                latched_post_touchdown_ticks=int(args.landing_mpc_latched_post_touchdown_ticks),
                contact_force_threshold_n=float(args.landing_mpc_contact_force_threshold),
                contact_release_threshold_n=(None if args.landing_mpc_contact_release_threshold is None
                                            else float(args.landing_mpc_contact_release_threshold)),
                contact_release_ticks=int(args.landing_mpc_contact_release_ticks),
                mode_validation_force_n=float(args.landing_mpc_mode_validation_force),
                mode_validation_ticks=int(args.landing_mpc_mode_validation_ticks),
                hold_tracking_error_release_rad=float(args.landing_mpc_hold_tracking_error_release_rad),
                predict_touchdown_contacts=bool(args.landing_mpc_predict_touchdown_contacts),
                contact_mode_fallback=bool(args.landing_mpc_contact_mode_fallback),
                contact_mode_sequence_search=bool(args.landing_mpc_contact_mode_sequence_search),
                contact_mode_sequence_max_candidates=int(args.landing_mpc_contact_mode_sequence_max_candidates),
                touchdown_mode_height_scale=float(args.landing_mpc_touchdown_mode_height_scale),
                capture_foot_gain=float(args.landing_mpc_capture_foot_gain),
                capture_foot_max_offset_m=float(args.landing_mpc_capture_foot_max_offset),
                leg_posture_blend=float(args.landing_mpc_leg_posture_blend),
                arm_posture_blend=float(args.landing_mpc_arm_posture_blend),
                touchdown_height_tolerance_m=float(args.landing_mpc_touchdown_height_tolerance),
                touchdown_vertical_speed_limit_mps=float(args.landing_mpc_touchdown_vertical_speed_limit),
                touchdown_impact_projection=bool(args.landing_mpc_touchdown_impact_projection),
                force_distribution_weight=float(args.landing_mpc_force_distribution_weight),
                force_wrench_weight=float(args.landing_mpc_force_wrench_weight),
                nonfoot_clearance_weight=float(args.landing_mpc_nonfoot_clearance_weight),
                nonfoot_clearance_hard=bool(args.landing_mpc_nonfoot_clearance_hard),
                nonfoot_clearance_m=float(args.landing_mpc_nonfoot_clearance_m),
                nonfoot_clearance_kp_s2=float(args.landing_mpc_nonfoot_clearance_kp),
                nonfoot_clearance_kd_s=float(args.landing_mpc_nonfoot_clearance_kd),
                continuous_reference=bool(args.landing_mpc_continuous_reference),
                load_aware_support=bool(args.landing_mpc_load_aware_support),
                load_force_floor_n=float(args.landing_mpc_load_force_floor),
                free_arm=bool(args.landing_mpc_free_arm),
                command_history_interface=bool(args.landing_mpc_command_history_interface),
                implicit_command_qp=bool(args.landing_mpc_implicit_command_qp),
                interval_command_qp=bool(args.landing_mpc_interval_command_qp),
                interval_active_set=bool(args.landing_mpc_interval_active_set),
                full_command_damper=bool(args.landing_mpc_full_command_damper),
                minimum_command_cost=bool(args.landing_mpc_minimum_command_cost),
                command_change_weight=float(args.landing_mpc_command_change_weight),
                student_future_policy_rolled_out=False,
                records=len(landing_mpc_records),
                predicted_contact_transition_records=sum(bool(r.get('predicted_contact_additions')) for r in landing_mpc_records),
                predicted_touchdown_impact_records=sum(
                    any(item.get('impact', {}).get('applied', False)
                        for item in r.get('predicted_touchdown_impacts', []))
                    for r in landing_mpc_records),
                applied_first_steps=sum(bool(r.get('applied_first_step')) for r in landing_mpc_records),
                applied_torque_overlays=sum(bool(r.get('torque_overlay_applied')) for r in landing_mpc_records),
                solver_failures=sum(bool(r.get('solver_failure')) for r in landing_mpc_records),
                terminal_acceptances=sum(bool(r.get('accepted')) for r in landing_mpc_records),
                records_path='landing_mpc_records.json',native_execution=True,
                frozen_policy_performance=False,formal_acceptance=False)
        if args.expert_bank is not None:
            summary['expert_bank_selection'] = dict(
                manifest=str(args.expert_bank.resolve()), manifest_sha256=digest(args.expert_bank),
                controller_trial=True, runtime_selection_not_formal=True,
                selected_case_ids=controller.selected_case.cpu().tolist())
        atomic_json(args.output / 'episodes.json', rows)
        atomic_json(args.output / 'metrics.json', summary)
        atomic_json(args.output / 'completed.json', dict(env_steps=(tick+1)*count))
        print(json.dumps(summary), flush=True)
    finally:
        if 'native_replay' in locals():
            native_replay.save()
        if 'pure_observer' in locals():
            pure_observer.save()
            pure_observer.detach()
        env._diagnostic_observer=None
        env.close()


try:
    main()
except BaseException:
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'error.txt').write_text(traceback.format_exc())
    raise
finally:
    if storage_reserve is not None:
        storage_reserve.release()
    app.close(skip_cleanup=True)
