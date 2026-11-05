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
parser.add_argument('--landing-mpc-support-margin', action='store_true',
    help='Use the experimental support-margin viability gate (MPC v2)')
parser.add_argument('--landing-mpc-swing-foot-weight', type=float, default=0.,
    help='Optional soft touchdown-foot task weight; zero preserves the original support QP')
parser.add_argument('--landing-mpc-swing-foot-kp', type=float, default=80.,
    help='Soft touchdown-foot proportional acceleration gain')
parser.add_argument('--landing-mpc-swing-foot-kd', type=float, default=18.,
    help='Soft touchdown-foot damping gain')
parser.add_argument('--landing-mpc-swing-foot-clearance', type=float, default=.025,
    help='Target clearance above the measured support plane in metres')
parser.add_argument('--reactive-vhsip-cases', type=int, nargs='+', default=None,
    help='Controller-trial TRAIN indices for the GPU-batched VHSIP reactive landing reference')
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
from scripts.landing_support_mpc_v1 import LandingSupportMPCV1, LandingMPCSettingsV1
from scripts.landing_support_mpc_v2 import LandingSupportMPCV2, LandingMPCSettingsV2
from scripts.whole_body_support_qp_v1 import PositionPDInterface, SupportSettings

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
                 landing_mpc_support_margin=False,
                 landing_mpc_swing_foot_weight=0., landing_mpc_swing_foot_kp=80.,
                 landing_mpc_swing_foot_kd=18., landing_mpc_swing_foot_clearance=.025,
                 reactive_vhsip_cases=None):
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
        self.memory_conditioned=bool(metadata.get('memory_conditioned',False))
        self.first_ball_feature=torch.zeros((count,18),device=device) if self.memory_conditioned else None
        self.first_ball_seen=torch.zeros((count,),dtype=torch.bool,device=device) if self.memory_conditioned else None
        self.landing_mpc_cases=set(int(x) for x in (landing_mpc_cases or []))
        self.landing_mpc_allow_first_step=bool(landing_mpc_allow_first_step)
        self.landing_mpc_support_margin=bool(landing_mpc_support_margin)
        self.landing_mpc_swing_foot_weight=float(landing_mpc_swing_foot_weight)
        self.landing_mpc_swing_foot_kp=float(landing_mpc_swing_foot_kp)
        self.landing_mpc_swing_foot_kd=float(landing_mpc_swing_foot_kd)
        self.landing_mpc_swing_foot_clearance=float(landing_mpc_swing_foot_clearance)
        self.reactive_vhsip_cases=None if reactive_vhsip_cases is None else set(int(x) for x in reactive_vhsip_cases)
        if self.reactive_vhsip_cases and self.landing_config is not None:
            raise ValueError('Separate declared landing assistance and VHSIP trial')
        self.landing_mpc_logs=[]
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
            vhsip_config=dict(mode='prepare_and_buffer_vhsip', blend_s=.08,
                settling_s=.7, landing_height_m=.3, capture_gain=1., ik_iterations=3,
                joint_target_rate_rad_s=20., ik_damping=.025)
            self.reactive_vhsip=LandingAssistedController(self.base,count,device,vhsip_config,
                checkpoint['asset_sha256'],enabled_cases=self.reactive_vhsip_cases)

    def configure_landing_mpc(self, limits):
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
            swing_foot_clearance_m=self.landing_mpc_swing_foot_clearance)
        if self.landing_mpc_support_margin:
            self.landing_mpc=LandingSupportMPCV2(interface, qp_settings=qp_settings,
                settings=LandingMPCSettingsV2(horizon_steps=self._landing_mpc_horizon_steps))
        else:
            self.landing_mpc=LandingSupportMPCV1(interface, qp_settings=qp_settings,
                settings=LandingMPCSettingsV1(horizon_steps=self._landing_mpc_horizon_steps))

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

    @torch.inference_mode()
    def action(self,observation,default,limits):
        source=self.reactive_vhsip if self.reactive_vhsip is not None else self.base
        base=source.action(observation,default,limits)
        if self.actor is None:return base
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
        if self.landing_mpc is None:return action
        raw=observation['policy'];public_clock=raw[:,775]
        # phase774 is a measured current support/contact phase. We keep the
        # first native version explicit: no future outcome label or case ID is
        # exposed to the MPC; the selected case list only limits this trial.
        active=(raw[:,774]>.5)
        for lane in torch.where(active)[0].tolist():
            case=int(lane)  # evaluator maps env lanes to the fixed TRAIN panel
            if case not in self.landing_mpc_cases:continue
            try:
                from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
                if not hasattr(self,'_landing_mpc_model'):
                    self._landing_mpc_model=LoadedPinocchioModelV1()
                pub=raw[lane].detach().cpu().numpy()
                lim=limits.detach().cpu().numpy()
                dflt=default.detach().cpu().numpy()
                state=self._landing_mpc_model.state_from_public(pub,dflt)
                joint_ref=lim[:,0]+.5*(action[lane].detach().cpu().numpy()+1.)*(lim[:,1]-lim[:,0])
                # The native trial must exercise the declared risk trigger.  A
                # selected TRAIN case is only an admission filter; it is not a
                # reason to keep overwriting a stable four-foot command.
                plan=self.landing_mpc.solve(state,joint_ref,force=False)
                horizon=plan.get('horizon',[])
                failed_stage=next((row for row in horizon if not row.get('accepted')),None)
                record=dict(case=case,clock_s=float(public_clock[lane]),accepted=bool(plan['accepted']),
                    first_action_available=bool(plan.get('horizon')) and bool(plan['horizon'][0].get('accepted')),
                    solver_failure=bool(plan.get('solver_failure',False)),rejection=plan.get('rejection'),
                    constraint_failure=bool(plan.get('constraint_failure',False)),
                    risk_trigger=bool(plan.get('risk',{}).get('trigger',False)),
                    risk_reason=plan.get('risk',{}).get('reason'),
                    first_step_improves=bool(plan.get('first_step_improves',False)),
                    initial_risk_score=plan.get('initial_risk_score'),
                    first_step_risk_score=plan.get('first_step_risk_score'),
                    runtime_ms=float(plan.get('wall_runtime_s',0.)*1000.),terminal_risk=plan.get('terminal_risk'),
                    horizon_steps=len(horizon),
                    failed_stage=failed_stage.get('stage') if failed_stage is not None else None,
                    failed_solver_status=failed_stage.get('solver_status') if failed_stage is not None else None,
                    failed_solver_iterations=failed_stage.get('solver_iterations') if failed_stage is not None else None,
                    failed_contacts=failed_stage.get('contacts') if failed_stage is not None else None,
                    failed_stage_risk=failed_stage.get('risk') if failed_stage is not None else None,
                    failed_stage_residuals=failed_stage.get('residuals') if failed_stage is not None else None,
                    native_execution=True)
                if plan.get('accepted') or (self.landing_mpc_allow_first_step and
                                            record['first_action_available'] and record['risk_trigger'] and
                                            record['first_step_improves']):
                    chosen=np.asarray(plan['horizon'][0]['normalized_action'],dtype=np.float32)
                    action[lane]=torch.as_tensor(chosen,device=action.device)
                    record['applied_first_step']=True
                else:record['applied_first_step']=False
                self.landing_mpc_logs.append(record)
            except Exception as error:
                self.landing_mpc_logs.append(dict(case=case,clock_s=float(public_clock[lane]),accepted=False,
                    first_action_available=False,applied_first_step=False,solver_failure=True,
                    rejection='exception:'+repr(error),native_execution=True))
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
        recovery_assistance_execution=RECOVERY_EXECUTION_SCHEMA,
        landing_trial=landing_trial,support_trial=support_trial,
        support_config=effective_support,support_disabled=args.disable_support,
        record_all_first_episode_actions=args.record_first_episodes,
        residual_actor_enabled=has_actor,
        landing_mpc_cases=args.landing_mpc_cases,
        landing_mpc_horizon_steps=args.landing_mpc_horizon_steps,
        landing_mpc_allow_first_step=args.landing_mpc_allow_first_step,
        landing_mpc_support_margin=args.landing_mpc_support_margin,
        landing_mpc_swing_foot_weight=args.landing_mpc_swing_foot_weight,
        landing_mpc_swing_foot_kp=args.landing_mpc_swing_foot_kp,
        landing_mpc_swing_foot_kd=args.landing_mpc_swing_foot_kd,
        landing_mpc_swing_foot_clearance=args.landing_mpc_swing_foot_clearance,
        reactive_vhsip_cases=args.reactive_vhsip_cases,
        expert_bank=str(args.expert_bank.resolve()) if args.expert_bank else None,
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
    controller = (CaseExpertBankController(checkpoint, args.expert_bank, count, env.device)
                  if args.expert_bank is not None else
                  SharedTargetResidualController(checkpoint, count, env.device,landing_trial,support_trial,args.disable_support,
                      landing_mpc_cases=args.landing_mpc_cases,
                      landing_mpc_horizon_steps=args.landing_mpc_horizon_steps,
                      landing_mpc_allow_first_step=args.landing_mpc_allow_first_step,
                      landing_mpc_support_margin=args.landing_mpc_support_margin,
                      landing_mpc_swing_foot_weight=args.landing_mpc_swing_foot_weight,
                      landing_mpc_swing_foot_kp=args.landing_mpc_swing_foot_kp,
                      landing_mpc_swing_foot_kd=args.landing_mpc_swing_foot_kd,
                      landing_mpc_swing_foot_clearance=args.landing_mpc_swing_foot_clearance,
                      reactive_vhsip_cases=args.reactive_vhsip_cases))
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
        if hasattr(controller,'configure_landing_mpc'):
            controller.configure_landing_mpc(limits)
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
                if hasattr(controller,'landing_mpc_diagnostics'):
                    landing_mpc_records.extend(controller.landing_mpc_diagnostics())
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
        if args.reactive_vhsip_cases is not None:
            summary['reactive_vhsip'] = dict(schema='reactive-vhsip-position-interface-v1',
                selected_cases=[int(x) for x in args.reactive_vhsip_cases],
                controller_trial=True, native_execution=True, frozen_policy_performance=False,
                formal_acceptance=False)
        if args.landing_mpc_cases is not None:
            summary['landing_mpc'] = dict(schema=('landing-support-mpc-v2-support-margin'
                if args.landing_mpc_support_margin else 'landing-support-mpc-v1'),controller_trial=True,
                selected_cases=[int(x) for x in args.landing_mpc_cases],
                horizon_steps=args.landing_mpc_horizon_steps,
                allow_first_step=bool(args.landing_mpc_allow_first_step),
                support_margin=bool(args.landing_mpc_support_margin),
                records=len(landing_mpc_records),
                applied_first_steps=sum(bool(r.get('applied_first_step')) for r in landing_mpc_records),
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
