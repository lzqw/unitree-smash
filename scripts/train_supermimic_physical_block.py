#!/usr/bin/env python3
"""Long-running sequential Recovery→Approach SuperMimic training block.

Native Isaac interaction, replay, world-model imagination and actor/critic
updates are durable and restartable. No MPC or teacher intervention.
"""
import argparse,hashlib,json,os,shutil,subprocess,sys,time,traceback,signal,io
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
LEARNER_SEED = 20261114
NATIVE_TRAIN128_SEED = 882072


def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,default=str)+'\n');temp.replace(path)


def parse_quota(text, *, allowed, name):
    """Parse auditable ``key=count`` quotas without silently dropping keys."""
    result={}
    if text is None or not str(text).strip():
        return result
    for item in str(text).split(','):
        if '=' not in item:
            raise ValueError(f'{name} entries must be key=count: {item!r}')
        key,value=(part.strip() for part in item.split('=',1))
        if key not in allowed:
            raise ValueError(f'Unknown {name} key: {key}')
        value=int(value)
        if value < 0:
            raise ValueError(f'{name} values must be nonnegative')
        result[key]=value
    return result


def scale_wm_quotas(quotas, batch_size):
    """CLI quotas retain their documented128-window unit; scale ratios exactly.

    This only changes supervised model minibatch size, never native lanes or
    imagination/world temporal horizons. No implicit OOM retry/ratio change.
    """
    if type(batch_size) is not int or batch_size<128 or batch_size%128:
        raise ValueError('WM batch must be a positive multiple of128')
    if not quotas or any(type(v) is not int or v<0 for v in quotas.values()) or sum(quotas.values())!=128:
        raise ValueError('WM quota units must sum to128 before batch scaling')
    return {k:v*(batch_size//128) for k,v in quotas.items()}


def parse_lane_list(text):
    """Parse explicit whole-lane exclusions such as ``96-127,3,7``."""
    if text is None or not str(text).strip():
        return []
    lanes = set()
    for item in str(text).split(','):
        item = item.strip()
        if not item:
            continue
        if '-' in item:
            left, right = (part.strip() for part in item.split('-', 1))
            start, end = int(left), int(right)
            if start > end:
                raise ValueError(f'Invalid lane range: {item}')
            lanes.update(range(start, end + 1))
        else:
            lanes.add(int(item))
    if any(lane < 0 for lane in lanes):
        raise ValueError('Lane ids must be nonnegative')
    return sorted(lanes)


def select_launch_for_start(row, launch):
    """Select the causal launch batch for a full or compact lane start."""
    lane_id=row.get('_lane_id') if isinstance(row,dict) else None
    if lane_id is None or launch.shape[0] == len(row['before']['physical']['joint_position']):
        return launch
    lane_id=int(lane_id)
    if not 0 <= lane_id < launch.shape[0]:
        raise ValueError(f'Dynamic start lane outside launch tensor: {lane_id}')
    return launch[lane_id:lane_id+1]


def research_learner_rng(requested_seed, saved_config, *, fresh, replay, natural_flight):
    """Fork training randomness once; an ordinary continuation keeps the restored RNG cursors."""
    inherited = saved_config.get('research_training_seed')
    if requested_seed is None:
        return inherited
    if not fresh:
        if requested_seed != inherited:
            raise ValueError('Cannot change the training seed of an ongoing schedule')
        return inherited
    import torch
    # The Actor start bank samples independently of WM replay. Keep the first
    # two derived seeds compatible with earlier explicit-seed runs.
    torch.manual_seed(requested_seed)
    streams = torch.Generator().manual_seed(requested_seed)
    replay_seed, flight_seed = torch.empty(2, dtype=torch.int64).random_(generator=streams).tolist()
    start_seed = torch.empty((), dtype=torch.int64).random_(generator=streams).item()
    replay.generator.manual_seed(replay_seed)
    replay.actor_start_bank.generator.manual_seed(start_seed)
    if natural_flight is not None:
        natural_flight.generator.manual_seed(flight_seed)
    return requested_seed


def restore_world_optimizer(optimizer,state,requested_lr):
    optimizer.load_state_dict(state)
    if requested_lr is not None:
        for group in optimizer.param_groups:
            group['lr']=requested_lr
    lr=float(optimizer.param_groups[0]['lr'])
    if not 0<lr<float('inf'):
        raise ValueError('Invalid resumed WM optimizer rate')
    return lr


def main():
    from isaaclab.app import AppLauncher
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('archive','packed-data','physical-checkpoint','ball-checkpoint','task-checkpoint','safety-mapping','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--resume-training',type=Path)
    parser.add_argument('--resume-wm-replay',type=Path,help='Resume the preregistered candidate WM with its fitted Adam and sampling state')
    parser.add_argument('--resume-next-cycle',action='store_true',help='Start a new cycle only after checkpoint-matched native acceptance')
    parser.add_argument('--resume-reset-schedule',action='store_true',
                        help='Resume model/optimizer state while starting a fresh bounded short-alternating schedule')
    parser.add_argument('--resume-without-replay',action='store_true',
                        help='Bounded diagnostic resume using model/optimizer state without loading historical online chunks')
    parser.add_argument('--native-acceptance-receipt',type=Path)
    parser.add_argument('--training-profile',choices=('bounded','large','research_scale'),default='bounded')
    parser.add_argument('--research-contact-gradient',choices=('G0','G1'),default='G1')
    parser.add_argument('--extend-completed-research', action='store_true',
                        help='Append cycles after the final refresh of a completed research schedule')
    parser.add_argument('--research-training-seed', type=int,
                        help='Fork learner and sampler RNGs for an independent research training replicate')
    parser.add_argument('--shuffle-wm-quota-groups', action='store_true',
                        help='Use replay RNG to vary tied quota allocations between WM batches')
    parser.add_argument('--research-contact-weight',type=float,default=.10)
    parser.add_argument('--training-stage',choices=('wm_calibrate','shadow_policy','precontact_only','contact_detached','research_contact','full_contact'),default='full_contact')
    parser.add_argument('--contact-bptt-mode',choices=('blocked','detached','precontact','enabled'))
    parser.add_argument('--contact-validation-receipt',type=Path)
    parser.add_argument('--action-ball-checkpoint',type=Path,default=None,
                        help='Opt-in fitted swept-racket candidate; never silently replaces legacy ball response')
    parser.add_argument('--g07-execution-root',type=Path,default=None,
                        help='Explicit root containing the measured G07 execution contract and diagnostics')
    parser.add_argument('--train-action-ball-in-wm',action='store_true',
                        help='Allow WM optimizer steps to update the action-conditioned ball head')
    parser.add_argument('--resume-start-block',choices=('recovery','approach'),default='recovery',
                        help='Resume a completed Recovery checkpoint directly at Approach')
    parser.add_argument('--resume-warmup-ticks',type=int,default=0,
                        help='Optional native ticks before a resumed stream; zero preserves exact block budgets')
    parser.add_argument('--actor-update-budget',type=int,default=None,help='Legacy single-block budget')
    parser.add_argument('--toi-effective-start-lanes',type=int,default=1,
                        help='Target distinct episodes per TOI-only Approach proposal when greater than one')
    parser.add_argument('--recovery-effective-start-lanes',type=int,default=12,help='Pool distinct gated Recovery episodes across archived G07 snapshots per optimizer proposal')
    parser.add_argument('--recovery-update-budget',type=int,default=256,
                        help='Total effective updates for this head per cycle; joint mode uses --joint-update-budget')
    parser.add_argument('--approach-update-budget',type=int,default=256,
                        help='Total effective updates for this head per cycle; joint mode uses --joint-update-budget')
    parser.add_argument('--joint-update-budget',type=int,default=16)
    parser.add_argument('--update-mode',choices=('sequential','joint'),default='sequential')
    parser.add_argument('--sequential-layout',choices=('short-alternating','legacy-blocks'),default='short-alternating')
    parser.add_argument('--head-block-size',type=int,default=8)
    parser.add_argument('--refresh-mode',choices=('pair','per-head'),default='pair',
                        help='pair: R8/A8 then one combined refresh; per-head: integration diagnostic only')
    parser.add_argument('--ball-side-mode',choices=('shadow','enforce'),default='shadow')
    parser.add_argument('--recovery-proposal-budget',type=int,default=512)
    parser.add_argument('--approach-proposal-budget',type=int,default=512)
    parser.add_argument('--refresh-ticks',type=int,default=800)
    parser.add_argument('--cycles',type=int,default=1)
    parser.add_argument('--native-ticks-per-actor-update',type=int,default=16)
    parser.add_argument('--replay-capacity-ticks',type=int,default=16384)
    parser.add_argument('--tensorboard-dir',type=Path)
    parser.add_argument('--native-return-drop-tolerance',type=float,default=.10)
    parser.add_argument('--native-return-flat-tolerance',type=float,default=.05)
    parser.add_argument('--native-return-patience',type=int,default=2)
    parser.add_argument('--refresh-world-updates',type=int,default=32)
    parser.add_argument('--refresh-critic-updates',type=int,default=20)
    parser.add_argument('--reference-policy-checkpoint',type=Path)
    parser.add_argument('--bad-policy-checkpoint',type=Path)
    parser.add_argument('--video-dir',type=Path)
    parser.add_argument('--command-delta-budget',type=float,default=0.005)
    parser.add_argument('--target-delta-budget-rad',type=float,default=0.04)
    parser.add_argument('--native-reference-command-budget',type=float,default=None,
                        help='Optional cumulative final-command budget relative to the accepted native reference')
    parser.add_argument('--native-reference-target-budget-rad',type=float,default=None,
                        help='Optional cumulative joint-target budget relative to the accepted native reference')
    parser.add_argument('--world-updates-per-actor-update',type=int,default=4,
                        help='World-model optimizer steps before each actor proposal')
    parser.add_argument('--wm-separate-supervision',action='store_true',help='Fit robot, ball and task parameters to their respective native targets')
    parser.add_argument('--wm-forward-guard',type=Path,help='Fixed native launch panel and verified block-initial WM score for every online refresh')
    parser.add_argument('--world-learning-rate',type=float,help='Defaults to the restored optimizer rate, or 0.0002 for a new run')
    parser.add_argument('--wm-natural-flight-loss',choices=('huber','mse'),default='huber')
    parser.add_argument('--wm-direct-long-body-supervision',action='store_true',help='Use measured 4/16/32/64-step robot, root angular velocity and racket FK targets for the robot gradient')
    parser.add_argument('--wm-active-flight-batch',type=int,default=0,help='Extra long flight windows from eligible lanes in the latest frozen dataset')
    parser.add_argument('--wm-natural-flight-batch',type=int,default=0,help='Additional immutable training-lane 64-step native windows per shared WM update')
    parser.add_argument('--wm-calibration-updates',type=int,default=256,
                        help='GPU world-model updates for wm_calibrate stage')
    parser.add_argument('--wm-ball-loss-weight',type=float,default=4.0)
    parser.add_argument('--wm-contact-ball-loss-weight',type=float,default=8.0)
    parser.add_argument('--racket-loss-weight',type=float,default=1.0,
                        help='Weight of the FK racket position/normal/velocity auxiliary WM loss')
    parser.add_argument('--cpu-threads',type=int,default=4)
    parser.add_argument('--wm-batch-size',type=int,default=128,help='Supervised WM minibatch, multiple of128; native environments remain128')
    parser.add_argument('--wm-bootstrap-max-ticks',type=int,default=0,
                        help='Collect up to this many frozen native ticks to satisfy online quotas before the first proposal')
    parser.add_argument('--wm-action-collection',type=Path,action='append',default=[],
                        help='Completed train-only signed PhysX collections with native 16-step WM supervision')
    parser.add_argument('--wm-source-quota',
                        default='recent_online=32,older_online=16,initial_archive=32,action_identification=32,contact_event=16',
                        help='Source quotas per128 WM windows. Default targets natural62.5%%, action25%%, event12.5%%; diagnostic mode records shortages.')
    parser.add_argument('--wm-sampling-mode',choices=('diagnostic','promotion'),default='diagnostic',
                        help='diagnostic records quota fallback; promotion rejects shortfalls and fallback')
    parser.add_argument('--approach-refresh-interval',type=int,default=8)
    parser.add_argument('--wm-approach-stage-quota',default='uniform=24,approach_active=32,hit_prepare=24,contact_release=24,approach_failure=16,post_hit=8')
    parser.add_argument('--wm-contact-stage-quota',default='uniform=24,hit_prepare=32,contact_release=48,post_hit=24')
    parser.add_argument('--wm-stage-quota',default='uniform=32,pre_touchdown=24,touchdown=24,post_touchdown=24,recovery_failure=16,stable_recovery=8')
    parser.add_argument('--wm-latest-policy-windows', type=int,
                        help='Prefer this many whole windows from the latest native behavior within existing quotas')
    parser.add_argument('--wm-recent-policy-window',type=int,default=32)
    parser.add_argument('--wm-recent-tick-window',type=int,default=1024)
    parser.add_argument('--wm-holdout-lanes',default='',
                        help='Explicit whole-lane WM exclusion, for example 96-127. '
                             'Use this for independent calibration/validation splits.')
    parser.add_argument('--task-reference-bank',type=Path,default=None,
                        help='Optional success-only task-space reference bank')
    parser.add_argument('--tracking-weight',type=float,default=0.0)
    parser.add_argument('--use-substep-toi-references', action='store_true',
                        help='Use measured first-string substep targets for newly admitted Approach references')
    parser.add_argument('--intercept-plane-calibration',type=Path,
                        help='Use a training-fitted contact target plane in bounded TOI-only imagination')
    parser.add_argument('--toi-racket-weight',type=float,default=1.0,
                        help='Approach TOI-racket-only auxiliary loss weight')
    parser.add_argument('--diagnostic-critic-loss',action='store_true',
                        help='Research only: monitor the minibatch critic MSE after warmup; retain finite-value and update checks')
    parser.add_argument('--landing-and-first-contact-objectives', action='store_true',
                        help='Revised post-landing and first-racket research objectives and sampling')
    parser.add_argument('--interleaved-approach-starts',action='store_true',
                        help='Spread the four Approach start families across each 20-draw quota period')
    parser.add_argument('--restart-uphill-approach-momentum', action='store_true',
                        help='Restart the Approach Adam first moment when its proposal increases the linearized objective')
    parser.add_argument('--recovery-rollout-landing',action='store_true',
                        help='Start the post-landing Recovery objective at an imagined touchdown within the rollout')
    parser.add_argument('--recovery-prefix-and-string-margin', action='store_true',
                        help='Sample causal pre-fall starts and train a 5 mm TOI string-bed margin')
    parser.add_argument('--continuous-approach-starts',action='store_true',
                        help='Advance the four start quotas across short blocks using checkpointed Approach attempts')
    parser.add_argument('--approach-objective',choices=('task','toi_racket_only'),default='task',
                        help='Approach auxiliary objective. toi_racket_only uses event-aligned TOI racket references while ball contact gradients stay detached.')
    parser.add_argument('--paired-evaluation-baseline',type=Path,default=None,
                        help='JSON list or episodes_summary.json for the fixed 128-case pre-update paired evaluation')
    AppLauncher.add_app_launcher_args(parser);args=parser.parse_args()
    # Every training stage uses the GPU-resident actor, WM and 128-lane
    # physical loop.  Fail before AppLauncher/Isaac startup when the driver is
    # absent, instead of spending minutes constructing a simulator that can
    # only fail later on a hard-coded cuda:0 tensor allocation.
    import torch
    research = args.training_profile == 'research_scale'
    if args.research_training_seed is not None and (not research or not 0 <= args.research_training_seed < 2**63):
        parser.error('research-training-seed requires research_scale and a nonnegative signed 64-bit seed')
    if args.extend_completed_research and (not research or args.resume_training is None or
                                           args.resume_reset_schedule or args.resume_next_cycle):
        parser.error('Extending research cycles requires continuation of a saved research schedule')
    if args.diagnostic_critic_loss and not research:
        parser.error('--diagnostic-critic-loss requires research_scale')
    if research != (args.training_stage == 'research_contact'):
        parser.error('research_scale uses the non-promotable research_contact stage')
    if research and (args.sequential_layout!='short-alternating' or args.update_mode!='sequential' or
                     args.approach_objective!='toi_racket_only' or args.action_ball_checkpoint is None or
                     args.contact_bptt_mode not in (None,'enabled') or args.resume_without_replay or
                     not 0 < args.research_contact_weight <= 1):
        parser.error('Research training requires alternating TOI blocks, calibrated contact and complete replay')
    if not 1<=args.toi_effective_start_lanes<=128 or (args.toi_effective_start_lanes>1 and args.approach_objective!='toi_racket_only'):
        parser.error('TOI start pooling requires a size in [1,128] and the TOI racket objective')
    if not 1<=args.recovery_effective_start_lanes<=128:
        parser.error('Recovery effective start lanes must be in [1,128]')
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        parser.error('CUDA GPU is required for SuperMimic training; no usable CUDA device is visible')
    from quadsmash.rl.supermimic_training_stage_v1 import resolve_training_stage
    training_stage=resolve_training_stage(args.training_stage)
    args.contact_bptt_mode=args.contact_bptt_mode or training_stage.contact_bptt_mode
    if args.training_stage=='full_contact' and args.contact_bptt_mode!='enabled':
        parser.error('full_contact requires contact-bptt-mode=enabled')
    if args.wm_calibration_updates<1 or args.wm_ball_loss_weight<=0 or args.wm_contact_ball_loss_weight<1 or args.racket_loss_weight<0:
        parser.error('WM calibration/loss settings invalid')
    if args.resume_wm_replay and (not args.resume_training or not args.wm_forward_guard or args.resume_without_replay
            or args.training_profile!='bounded' or args.contact_bptt_mode!='detached'):
        parser.error('A replayed WM requires a bounded detached resume with its full replay and a native forward guard')
    args.requested_world_lr=args.world_learning_rate
    if args.world_learning_rate is None:
        args.world_learning_rate=2e-4
    if not 0<args.world_learning_rate<float('inf'): parser.error('world-learning-rate must be finite and positive')
    if args.wm_direct_long_body_supervision and not (args.wm_separate_supervision and args.wm_natural_flight_batch>0):parser.error('Direct long body supervision requires separate supervision and native anchors')
    if args.wm_active_flight_batch<0 or (args.wm_active_flight_batch and not args.wm_natural_flight_batch):parser.error('Active flight windows require natural anchors')
    if args.wm_natural_flight_batch<0: parser.error('wm-natural-flight-batch must be nonnegative')
    if args.cpu_threads<1: parser.error('cpu-threads must be positive')
    if args.actor_update_budget is not None and args.actor_update_budget < 1: parser.error('actor-update-budget must be positive')
    if args.recovery_update_budget < 0 or args.approach_update_budget < 0: parser.error('block budgets must be non-negative')
    if args.recovery_proposal_budget < 1 or args.approach_proposal_budget < 1: parser.error('proposal budgets must be positive')
    if args.cycles < 1 or args.native_ticks_per_actor_update < 1 or args.replay_capacity_ticks < 64: parser.error('long-run budgets invalid')
    if args.refresh_ticks < 0 or args.refresh_world_updates < 0 or args.refresh_critic_updates < 0 or args.resume_warmup_ticks < 0: parser.error('refresh/resume budgets invalid')
    if args.world_updates_per_actor_update < 1: parser.error('world-updates-per-actor-update must be positive')
    if args.wm_bootstrap_max_ticks < 0:
        parser.error('Native WM bootstrap budget cannot be negative')
    if args.wm_batch_size < 128 or args.wm_batch_size%128: parser.error('wm-batch-size must be a positive multiple of128')
    if args.wm_latest_policy_windows is not None and not 0 <= args.wm_latest_policy_windows <= args.wm_batch_size:
        parser.error('Latest-policy preference must fit within the WM batch')
    if args.wm_recent_policy_window < 0 or args.wm_recent_tick_window < 1: parser.error('WM freshness windows invalid')
    if args.approach_refresh_interval < 1: parser.error('approach-refresh-interval must be positive')
    if args.refresh_ticks not in (0,800): parser.error('Frozen refresh requires the original 800-tick task')
    from quadsmash.rl.supermimic_frozen_return_monitor_v1 import FrozenReturnMonitor
    try:FrozenReturnMonitor(args.native_return_drop_tolerance,args.native_return_flat_tolerance,args.native_return_patience)
    except ValueError as exc:parser.error(str(exc))
    if args.tracking_weight < 0 or args.toi_racket_weight < 0 or (args.tracking_weight > 0 and args.task_reference_bank is None): parser.error('Tracking weights and reference-bank settings are invalid')
    if args.approach_objective=='toi_racket_only' and (args.toi_racket_weight<=0 or args.training_stage=='full_contact'):
        parser.error('TOI-only Approach requires positive racket weight and a detached-contact diagnostic stage')
    if args.intercept_plane_calibration and (not args.resume_training or args.approach_objective!='toi_racket_only'
                                             or args.training_profile not in ('bounded','research_scale')):
        parser.error('A fitted contact plane requires the corresponding bounded TOI-only resume checkpoint')
    if args.approach_objective=='toi_racket_only' and args.task_reference_bank is None:
        parser.error('toi_racket_only requires a validated event-aligned task reference bank')
    if args.ball_side_mode == 'enforce': parser.error('ball-side enforce mode is not implemented; use shadow for this audit-only revision')
    if args.actor_update_budget is not None:
        args.recovery_update_budget=0;args.approach_update_budget=args.actor_update_budget
    from quadsmash.rl.supermimic_long_run_contract_v1 import validate_budgets
    try:validate_budgets(args.recovery_update_budget,args.approach_update_budget,args.recovery_proposal_budget,args.approach_proposal_budget)
    except ValueError as error:parser.error(str(error))
    if args.action_ball_checkpoint is not None and args.contact_validation_receipt is None and args.training_stage=='full_contact':
        parser.error('Action-ball candidate requires a current, calibrated native contact-validation receipt')
    if args.training_profile=='large' and args.training_stage=='full_contact' and args.contact_validation_receipt is None:parser.error('Large training requires contact-validation-receipt; model-only sensitivity is insufficient')
    if args.training_profile=='large' and args.wm_sampling_mode!='promotion':
        parser.error('Large training requires wm-sampling-mode=promotion')
    if args.training_profile=='large' and (args.cycles!=1 or args.refresh_ticks!=800 or args.update_mode!='sequential'):parser.error('Large profile uses one cycle per process, full refresh, and external frozen evaluation between cycles')
    if args.resume_next_cycle and (args.resume_training is None or args.native_acceptance_receipt is None):parser.error('New cycle requires checkpoint and matching native acceptance')
    if args.landing_and_first_contact_objectives:
        if args.training_profile != 'research_scale':
            parser.error('Revised objectives require the research training profile')
        args.continuous_approach_starts = args.shuffle_wm_quota_groups = True
        args.recovery_prefix_and_string_margin = True
    if args.resume_reset_schedule and (args.resume_training is None or args.resume_next_cycle or
                                       args.training_profile not in ('bounded','research_scale') or args.training_stage=='full_contact'):
        parser.error('Fresh schedule requires a bounded non-promotable checkpoint continuation')
    if args.resume_without_replay and (args.resume_training is None or args.training_profile=='large' or args.training_stage=='full_contact'):
        parser.error('resume-without-replay is restricted to bounded non-promotable diagnostics')
    from quadsmash.rl.supermimic_alternating_schedule_v1 import make_plan,AlternatingSchedule,refresh_world_budgets,run_schedule
    from quadsmash.rl.supermimic_training_audit_v1 import sampling_promotion_gate, rollback_audit
    alternating=None
    if args.update_mode=='sequential' and args.sequential_layout=='short-alternating' and args.training_stage!='wm_calibrate':
        if (not args.resume_reset_schedule and args.resume_start_block!='recovery') or args.actor_update_budget is not None:
            parser.error('Short alternating resumes its saved event cursor, not resume-start-block/legacy actor budget')
        if args.refresh_ticks!=800:parser.error('Short alternating requires full frozen refresh; cannot disable it')
        if args.refresh_world_updates<2 or args.refresh_critic_updates<1:
            parser.error('Combined refresh needs both WM sampling distributions and native critic calibration')
        try:
            alternating=AlternatingSchedule(make_plan(args.recovery_update_budget,args.approach_update_budget,
                args.recovery_proposal_budget,args.approach_proposal_budget,block_size=args.head_block_size,
                cycles=args.cycles,refresh_mode=args.refresh_mode,refresh_every_pairs=8 if research else 1),execution_config=dict(
                    native_ticks_per_proposal=args.native_ticks_per_actor_update,world_updates_per_proposal=args.world_updates_per_actor_update,wm_batch_size=args.wm_batch_size,
                    refresh_ticks=args.refresh_ticks,refresh_world_updates=args.refresh_world_updates,refresh_critic_updates=args.refresh_critic_updates,
                    **(dict(contact_gradient=args.research_contact_gradient,contact_weight=args.research_contact_weight) if research else {}),
                    training_profile=args.training_profile,contact_receipt_sha256=hashlib.sha256(args.contact_validation_receipt.read_bytes()).hexdigest() if args.contact_validation_receipt else None))
        except ValueError as error:parser.error(str(error))
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    try:
        args.wm_approach_stage_quota=parse_quota(args.wm_approach_stage_quota,allowed={'uniform','approach_active','hit_prepare','contact_release','approach_failure','post_hit'},name='wm-approach-stage-quota')
        args.wm_contact_stage_quota=parse_quota(args.wm_contact_stage_quota,allowed={'uniform','hit_prepare','contact_release','post_hit'},name='wm-contact-stage-quota')
        args.wm_source_quota=parse_quota(args.wm_source_quota,allowed={'recent_online','older_online','initial_archive','action_identification','contact_event'},name='wm-source-quota')
        args.wm_stage_quota=parse_quota(args.wm_stage_quota,allowed={'uniform','pre_touchdown','touchdown','post_touchdown','pre_body_ground','post_landing_support_loss','post_landing_unload','recovery_failure','stable_recovery'},name='wm-stage-quota')
        for field in ('wm_source_quota','wm_stage_quota','wm_approach_stage_quota','wm_contact_stage_quota'):
            setattr(args,field,scale_wm_quotas(getattr(args,field),args.wm_batch_size))
    except ValueError as error:
        parser.error(str(error))
    if args.tensorboard_dir is not None:
        args.tensorboard_dir=args.tensorboard_dir.resolve()
        args.tensorboard_dir.mkdir(parents=True,exist_ok=True)
    if args.video_dir is not None:
        args.video_dir=args.video_dir.resolve();args.video_dir.mkdir(parents=True,exist_ok=True)
        (args.output/'video_request.json').write_text(json.dumps({
            'directory':str(args.video_dir),'headless':bool(getattr(args,'headless',False)),
            'status':'directory prepared; native viewport capture is explicit and must be provided by the Isaac camera/recorder extension'
        },indent=2)+'\n')
    write(args.output/'startup.json',dict(argv=sys.argv,cwd=os.getcwd(),executable=sys.executable,
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        dirty_diff=subprocess.check_output(['git','diff'],text=True),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        training_stage=training_stage.manifest(),wm_ball_loss_weight=args.wm_ball_loss_weight,
        wm_contact_ball_loss_weight=args.wm_contact_ball_loss_weight,
        racket_loss_weight=args.racket_loss_weight,cpu_threads=args.cpu_threads))
    from quadsmash.rl.supermimic_run_lifecycle import execute_scoped_native,stage
    stage(args.output,'BEFORE_APP');app=AppLauncher(args,fast_shutdown=False).app
    stage(args.output,'AFTER_APP')
    args.stop_request={'signal':None}
    def request_stop(signum,frame):args.stop_request['signal']=signal.Signals(signum).name
    # Override Kit's handler: it shuts down plugins before Python can save.
    signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
    execute_scoped_native(args.output,lambda:run(args, alternating=alternating),app)


def run(args, *, alternating=None):
    research=args.training_profile=='research_scale'
    from quadsmash.rl.supermimic_run_lifecycle import stage as lifecycle_stage
    # These audits are consumed by nested WM/actor callbacks.  Import them in
    # the run scope as well as the CLI setup scope so frozen refreshes cannot
    # hit a late NameError after spending hundreds of native ticks.
    from quadsmash.rl.supermimic_training_audit_v1 import sampling_promotion_gate, rollback_audit
    lifecycle_stage(args.output, 'RUN_ENTER')
    import torch
    lifecycle_stage(args.output, 'IMPORT_TORCH')
    from torch.utils.tensorboard import SummaryWriter
    lifecycle_stage(args.output, 'IMPORT_TENSORBOARD')
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    lifecycle_stage(args.output, 'IMPORT_RSL')
    from quadsmash.rl.supermimic_physical_loop_loading_v1 import load_physical_loop,APPROACH_NAMES,RECOVERY_NAMES,tensor_digest
    lifecycle_stage(args.output, 'IMPORT_LOADING')
    from quadsmash.rl.supermimic_physical_learner_v1 import PhysicalHorizonTrainer
    lifecycle_stage(args.output, 'IMPORT_LEARNER')
    from quadsmash.rl.supermimic_physical_loop_replay_v1 import PhysicalLoopReplay,current_start,STAGE_NAMES
    from quadsmash.rl.supermimic_physical_imagination_state_v1 import PhysicalImaginationStateV1
    from quadsmash.rl.supermimic_physical_capture_v1 import PhysicalTransitionArchive,tensor_tree_copy
    from quadsmash.rl.supermimic_physical_run_guard_v1 import critic_ready,atomic_torch_save
    from quadsmash.rl.supermimic_timebase_v2 import Timebase
    from quadsmash.rl.supermimic_task_contract import resolve_task_contract,make_task_cfg,zero_panel_reset_noise
    from quadsmash.rl.supermimic_terminal_scoring import score_terminal_episode
    from quadsmash.rl.supermimic_native_metrics import NativeEpisodeStatistics, count_first_episode_body_contacts, summarize_native_episodes
    from quadsmash.rl.supermimic_run_lifecycle import close_resource,stage
    # The alternating schedule is constructed in main(), but execution happens
    # inside this function.  Import the runner here as well so the short
    # alternating path does not depend on main()'s local namespace.
    from quadsmash.rl.supermimic_alternating_schedule_v1 import run_schedule,refresh_world_budgets
    from quadsmash.tasks.stand.supermimic_env_v1 import SuperMimicEnvV1
    from quadsmash.tasks.stand.full_racket_eval_env_v4 import FrozenPanelEnv
    lifecycle_stage(args.output, 'IMPORT_ALL')
    env=trainer=archive=None;writer=None;failed=False;tick=0;phase='LOAD'
    torch.set_num_threads(args.cpu_threads);torch.manual_seed(LEARNER_SEED)
    try:
        lifecycle_stage(args.output, 'BEFORE_LOAD_PHYSICAL_LOOP')
        actor,adapter,template,launch,manifest=load_physical_loop(archive=args.archive,
            physical_checkpoint=args.physical_checkpoint,ball_checkpoint=args.ball_checkpoint,
            task_checkpoint=args.task_checkpoint,safety_mapping=args.safety_mapping,
            action_ball_checkpoint=args.action_ball_checkpoint,execution_root=args.g07_execution_root)
        action_ball_trainable=bool(args.train_action_ball_in_wm or args.training_stage!='wm_calibrate')
        if adapter.action_ball_model is not None and not action_ball_trainable:
            adapter.action_ball_model.requires_grad_(False)
        adapter.set_contact_bptt_mode(args.contact_bptt_mode)
        manifest['training_stage']=dict(name=args.training_stage,contact_bptt_mode=args.contact_bptt_mode,
                                        actor_updates_allowed=args.training_stage!='wm_calibrate',
                                        promotable=args.training_stage=='full_contact')
        manifest['action_ball_trainable_in_wm']=action_ball_trainable
        adapter.manifest['training_stage']=manifest['training_stage']
        adapter.manifest['action_ball_trainable_in_wm']=action_ball_trainable
        lifecycle_stage(args.output, 'AFTER_LOAD_PHYSICAL_LOOP')
        contact_validation=None;wm_holdout_lanes=parse_lane_list(args.wm_holdout_lanes);runtime_wm_bounds={};runtime_validation_data=None;dual_quality=None;native_value_bank=None
        if args.contact_validation_receipt is not None:
            from quadsmash.rl.supermimic_long_run_contract_v1 import verify_contact_receipt
            contact_validation=verify_contact_receipt(args.contact_validation_receipt,dict(physical=args.physical_checkpoint,ball=args.ball_checkpoint,task=args.task_checkpoint,**({'action_ball':args.action_ball_checkpoint} if args.action_ball_checkpoint else {})),manifest['task_contract'])
            write(args.output/'contact_validation_verified.json',contact_validation)
            if args.training_profile=='large' or 'dual_head_quality' in json.loads(args.contact_validation_receipt.read_text()):
                from quadsmash.rl.supermimic_dual_head_quality_v1 import quality_spec
                dual_quality=quality_spec(json.loads(args.contact_validation_receipt.read_text()))
                receipt_lanes=sorted(dual_quality['heldout_lane_ids'])
                if wm_holdout_lanes and wm_holdout_lanes != receipt_lanes:
                    raise ValueError('Explicit --wm-holdout-lanes disagrees with contact receipt lanes')
                wm_holdout_lanes=receipt_lanes
                runtime_wm_bounds=dual_quality['wm_upper_bounds']
        forward_guard=None
        if args.wm_forward_guard:
            from scripts.supermimic_online_forward_guard import OnlineForwardGuard
            if research:
                # The published panel baseline is bound to the legacy component sources.
                # Verify it with that model, then score the actual composed WM on the same windows.
                _,baseline_adapter,_,_,_=load_physical_loop(archive=args.archive,physical_checkpoint=args.physical_checkpoint,
                    ball_checkpoint=args.ball_checkpoint,task_checkpoint=args.task_checkpoint,safety_mapping=args.safety_mapping,
                    execution_root=args.g07_execution_root)
                forward_guard=OnlineForwardGuard(args.wm_forward_guard,baseline_adapter,manifest['task_contract'])
                del baseline_adapter
            else:
                forward_guard=OnlineForwardGuard(args.wm_forward_guard,adapter,manifest['task_contract'])
        write(args.output/'model_manifest.json',adapter.manifest)
        packed_sha=hashlib.sha256(args.packed_data.read_bytes()).hexdigest()
        expected=json.loads((args.packed_data.parent/'completed.json').read_text())
        if packed_sha!=expected['sha256'] or expected['first_completed']!=128:raise ValueError('Invalid initialization data')
        package=torch.load(args.packed_data,map_location='cpu',weights_only=True)
        lane_count=int(package['tensors']['episode'].shape[1])
        if any(lane >= lane_count for lane in wm_holdout_lanes):
            raise ValueError(f'WM holdout lane outside packed archive: {wm_holdout_lanes}')
        lifecycle_stage(args.output, 'LOADED_PACKED_DATA')
        runtime_validation_data=package['tensors'];runtime_validation_path=str(args.packed_data)
        from functools import partial
        from quadsmash.rl.supermimic_approach_data_v1 import recorded_approach_metadata
        labeler = partial(recorded_approach_metadata, schema=adapter.task_model.memory_schema,
                          actor=actor, geometry=adapter.physical_model.geometry)
        replay=PhysicalLoopReplay(package,adapter.task_model.memory_schema,capacity_ticks=args.replay_capacity_ticks,approach_labeler=labeler)
        from quadsmash.rl.supermimic_physical_loop_replay_v1 import ActiveFlightWindows, NaturalFlightWindows, sample_long_native
        natural_flight=NaturalFlightWindows(package['tensors'],wm_holdout_lanes) if args.wm_natural_flight_batch else None
        active_flight=None
        def sample_long_flight():
            nonlocal active_flight
            if natural_flight is None:
                return None
            if args.wm_active_flight_batch and (active_flight is None or active_flight.data is not runtime_validation_data):
                active_flight=ActiveFlightWindows(runtime_validation_data,wm_holdout_lanes,natural_flight.generator)
            window,lanes=sample_long_native(natural_flight,args.wm_natural_flight_batch,active_flight,args.wm_active_flight_batch,
                                            trainer.world_updates,launch.device)
            return window,launch[lanes]

        lifecycle_stage(args.output, 'REPLAY_READY')
        # Resolve the exact execution contract before validating any reference
        # bank.  The reference-bank check runs during startup, before the
        # training environment is constructed, so leaving this assignment
        # until the later environment setup caused a startup-only
        # UnboundLocalError on valid TOI-racket canary runs.
        from quadsmash.rl.supermimic_policy_execution_v1 import contract as execution_contract
        policy_execution = execution_contract()
        from quadsmash.rl.supermimic_training_probe_replay import import_training_collection
        action_collections = []
        used_branches = set()
        for path in args.wm_action_collection:
            receipt = import_training_collection(replay,path,manifest['task_contract'],policy_execution,wm_holdout_lanes)
            branches = {h for group in receipt['spec_branches'].values() for h in group.values()}
            if used_branches & branches:
                raise ValueError('Duplicate native action-identification data')
            used_branches.update(branches)
            action_collections.append(receipt)
        from quadsmash.rl.supermimic_training_audit_v1 import identification_window_capacity
        capacity = identification_window_capacity(action_collections, args.wm_source_quota.get('action_identification',0))
        write(args.output/'action_identification_capacity.json',capacity)
        if args.wm_sampling_mode=='promotion' and capacity['unavoidable_shortfall']:
            raise ValueError(f'Insufficient distinct signed native windows for the WM batch: {capacity}')
        reference_bank=None
        if args.task_reference_bank is not None:
            from quadsmash.rl.supermimic_task_space_reference_v1 import TaskSpaceReferenceBankV1
            bank_payload=torch.load(args.task_reference_bank,map_location='cpu',weights_only=False)
            reference_bank=TaskSpaceReferenceBankV1.from_payload(bank_payload)
            reference_bank.assert_excludes(wm_holdout_lanes)
            if not reference_bank.manifest['protocol_verified']:raise ValueError('Tracking requires independently scored complete native references')
            if reference_bank.geometry_sha256 and reference_bank.geometry_sha256 != adapter.physical_model.geometry.model_sha256:
                raise ValueError('Task reference bank geometry hash mismatch')
            execution_contract_sha256=hashlib.sha256(json.dumps(policy_execution,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            if (args.tracking_weight>0 or (args.approach_objective=='toi_racket_only' and args.toi_racket_weight>0)) and reference_bank.execution_contract_sha256 != execution_contract_sha256:
                raise ValueError('Task reference bank execution contract hash does not match current policy execution')
            reference_bank=reference_bank.to(torch.device('cuda:0'))
        lifecycle_stage(args.output, 'REFERENCE_BANK_READY', enabled=reference_bank is not None)
        writer=SummaryWriter(str(args.tensorboard_dir or (args.output/'tensorboard')),flush_secs=30)
        from quadsmash.rl.supermimic_paired_evaluation_v1 import paired_evaluation
        paired_eval_dir=args.output/'paired_evaluation'; paired_eval_dir.mkdir(exist_ok=True)
        paired_eval_snapshots={}; paired_eval_actor_snapshots={}
        paired_eval_previous_stage={}
        def paired_action_probe(snapshot):
            """Evaluate deterministic actions on fixed causal starts for RMS audits."""
            current={k:v.detach().clone() for k,v in actor.state_dict().items()}
            try:
                actor.load_state_dict(snapshot,strict=True)
                values=[]
                with torch.no_grad():
                    for probe_state in paired_probe_states:
                        values.append(trainer._actor_action(probe_state.branch(),sample=False)[0].detach().cpu())
                log_std=getattr(actor,'log_std',None)
                return dict(action=torch.cat(values,dim=0),
                            log_std=None if log_std is None else log_std.detach().clamp(-8.,1.).cpu().clone(),
                            clipped_action=True)
            except (RuntimeError, AttributeError, KeyError):
                return None
            finally:
                actor.load_state_dict(current,strict=True)
        from quadsmash.rl.supermimic_physical_start_cache_v1 import load_start_cache,training_start_mask,APPROACH_TICKS,RECOVERY_TICKS,CRITIC_TICKS
        start_cache=load_start_cache(args.archive,manifest)
        lifecycle_stage(args.output, 'START_CACHE_LOADED')
        starts=[start_cache[t] for t in APPROACH_TICKS]
        critic_starts=[start_cache[t] for t in CRITIC_TICKS]
        lifecycle_stage(args.output, 'STARTS_SELECTED')
        def as_state(row):
            # Dynamic starts keep the full G07 graph and a per-lane loss mask.
            # Keep explicit launch selection for verified legacy compact starts.
            row_launch=select_launch_for_start(row,launch)
            return PhysicalImaginationStateV1.from_record(row,executor_template=template,launch=row_launch,device='cuda:0')
        timebase=Timebase(**json.loads((ROOT/'configs/supermimic_task_horizon_v2.json').read_text()))
        if research:
            from quadsmash.rl.supermimic_action_ball_v1 import CalibratedRacketImpulseV13
            if not isinstance(adapter.action_ball_model, CalibratedRacketImpulseV13):
                raise ValueError('Research profile requires calibrated C1 and V13')
        trainer=PhysicalHorizonTrainer(actor,adapter,as_state(starts[1]),timebase,
            actor_lr=2e-4 if research else 5e-5,command_delta_budget=args.command_delta_budget,
            target_delta_budget_rad=args.target_delta_budget_rad,
            native_reference_command_budget=args.native_reference_command_budget,
            native_reference_target_budget_rad=args.native_reference_target_budget_rad,
            trainable_names=RECOVERY_NAMES + APPROACH_NAMES,block_name='joint',
            reference_bank=reference_bank,tracking_weight=args.tracking_weight,
            world_lr=args.world_learning_rate,approach_objective=args.approach_objective,
            toi_racket_weight=args.toi_racket_weight)
        if research:
            adapter.checkpoint_supervised_rollout=True
            trainer.anchor_coeff=.01
            trainer.research_court=json.loads((ROOT/'configs/court_net_v4.json').read_text())
        trainer.ball_loss_weight=float(args.wm_ball_loss_weight)
        trainer.contact_ball_loss_weight=float(args.wm_contact_ball_loss_weight)
        trainer.direct_long_body_supervision=args.wm_direct_long_body_supervision
        trainer.natural_flight_loss=args.wm_natural_flight_loss
        trainer.racket_loss_weight=float(args.racket_loss_weight)
        lifecycle_stage(args.output, 'TRAINER_READY')
        trainer.set_budget_state_library([as_state(row) for row in starts+critic_starts[:4]])
        paired_probe_states=tuple(trainer.budget_states) if args.paired_evaluation_baseline else ()
        def paired_critic_audit():
            """Evaluate the current critic against the latest native RTG bank.

            This is an audit metric.  It never supplies an Actor target and it
            remains explicitly tied to the frozen policy identity in the bank.
            """
            if native_value_bank is None or native_value_bank.get('inputs') is None:
                return None
            inputs=native_value_bank.get('inputs'); targets=native_value_bank.get('rtg')
            if inputs is None or targets is None or len(inputs)==0:
                return None
            device=next(trainer.critic.parameters()).device
            values=[]
            with torch.no_grad():
                for part in inputs.split(128):
                    values.append(trainer.critic(part.to(device)).detach().cpu().reshape(-1))
            from quadsmash.rl.supermimic_paired_evaluation_v1 import native_critic_td
            predictions=torch.cat(values)
            td=native_critic_td(predictions,targets,native_value_bank['identity'],timebase.gamma)
            error=predictions-targets.detach().cpu().reshape(-1)
            return dict(td=td,sample_count=int(error.numel()),mae=float(error.abs().mean()),
                        rmse=float(error.square().mean().sqrt()),bias=float(error.mean()),
                        policy_id=native_value_bank.get('policy_id'),
                        actor_sha256=native_value_bank.get('actor_sha256'),
                        scope='latest frozen-policy native RTG audit; current-policy trust is separate')
        def approach_active(row):
            """Require a meaningful activation of the actual exported Approach gate."""
            observation=row['observation']
            if isinstance(observation,dict):
                observation=observation['policy']
            if observation.ndim != 2 or observation.shape[-1] != 794:
                raise ValueError('Approach start observation is not the 794D public contract')
            from quadsmash.rl.supermimic_approach_data_v1 import gate_on_actor_device
            gate=gate_on_actor_device(actor,observation)
            if gate.ndim == 2: gate=gate[:,0]
            # A tiny positive gate is numerically present but does not provide
            # a useful actor gradient. Singleton/masked batches need only their
            # own active lanes, not an impossible fixed four-lane threshold.
            from quadsmash.rl.supermimic_approach_data_v1 import active_gate
            mask=training_start_mask(row,wm_holdout_lanes).to(gate.device)
            gate=gate[mask]
            return active_gate(gate) if gate.numel() else False

        approach_pool=[start_cache[t] for t in APPROACH_TICKS if approach_active(start_cache[t])]
        if not approach_pool:
            raise RuntimeError('No approach-gated initialization starts available')
        recovery_pool=[start_cache[t] for t in RECOVERY_TICKS]
        # Same-policy native RTG calibration labels from the frozen 800-tick archive.
        # This is separate from the 0.64 s imagined objective and never mixed with
        # candidate-policy monitoring.
        from quadsmash.rl.supermimic_evidence_binding_v1 import complete_native_rtg
        rtg,rtg_complete=complete_native_rtg(package['tensors'],timebase.gamma)
        replay_archives=[];replay_receipts=[];critic_coverage_blocks=0
        refresh_world_updates=0;refresh_critic_updates=0
        replay_handoff=None
        plane_calibration = None
        if args.resume_training:
            resumed=torch.load(args.resume_training,map_location='cpu',weights_only=False)
            if args.wm_latest_policy_windows is None and not args.resume_reset_schedule:
                args.wm_latest_policy_windows = resumed['config'].get('wm_latest_policy_windows', 0)
            if not args.resume_reset_schedule and resumed['config'].get('shuffle_wm_quota_groups'):
                args.shuffle_wm_quota_groups=True
            if research and not args.resume_reset_schedule and resumed['config'].get('diagnostic_critic_loss'):
                args.diagnostic_critic_loss=True
            if not args.resume_reset_schedule and resumed['config'].get('recovery_prefix_and_string_margin'):
                args.recovery_prefix_and_string_margin = True
            saved_plane = resumed['config'].get('intercept_plane_calibration')
            if args.approach_objective=='toi_racket_only' and args.training_profile in ('bounded','research_scale'):
                plane_calibration = saved_plane
            if args.intercept_plane_calibration:
                from quadsmash.rl.supermimic_long_run_contract_v1 import sha256 as file_sha
                from quadsmash.rl.supermimic_toi_racket_objective import calibrated_plane_offset
                identity = file_sha(args.intercept_plane_calibration)
                if saved_plane is None or saved_plane['sha256'] != identity:
                    if not (args.resume_reset_schedule or args.resume_next_cycle):
                        raise ValueError('A changed contact target plane requires a fresh schedule')
                    source = file_sha(args.resume_training)
                    if research and args.resume_reset_schedule:
                        composition = resumed['metadata'].get('research_contact_composition', {})
                        source = composition.get('source_checkpoint_sha256', source)
                else:
                    source = saved_plane['source_checkpoint_sha256']
                offset = calibrated_plane_offset(json.loads(args.intercept_plane_calibration.read_text()),
                                                 wm_holdout_lanes, source)
                plane_calibration = dict(sha256=identity,source_checkpoint_sha256=source,offset_m=offset)
            if saved_plane and not plane_calibration and not (args.resume_reset_schedule or args.resume_next_cycle):
                raise ValueError('An ongoing schedule must retain its fitted TOI target plane')
            if plane_calibration:
                trainer.toi_plane_offset_m = plane_calibration['offset_m']
            previous_guard=resumed.get('config',{}).get('wm_forward_guard_sha256')
            if previous_guard is not None and (forward_guard is None or forward_guard.spec_sha256!=previous_guard):
                raise ValueError('Cannot drop or replace a checkpoint-bound native forward panel')
            old_collections = resumed['metadata'].get('wm_action_collections', [])
            if any(item['spec_branches'] not in [r['spec_branches'] for r in action_collections]
                   for item in old_collections):
                raise ValueError('Resume omitted or replaced previously used native action evidence')
            # Training stage, contact-gradient mode and WM trainability are
            # deliberately mutable across a resume (for example, a detached
            # shadow canary can be resumed as TOI-racket-only).  Compare the
            # immutable physical/execution contract instead of the entire
            # runtime manifest, otherwise a valid stage transition is rejected
            # as a source mismatch before the first update.
            immutable_manifest_keys = (
                'checkpoint_paths', 'checkpoint_sha256', 'g07_execution_root',
                'task_schema', 'actor_whitelist', 'actor_heads',
                'adapter_schema', 'physical_class', 'physical_schema',
                'ball_class', 'task_class', 'event_semantics',
                'discrete_approximations',
            )
            saved_physical = resumed['metadata']['physical_imagination']
            current_physical = adapter.manifest
            saved_contract = {key: saved_physical.get(key) for key in immutable_manifest_keys}
            current_contract = {key: current_physical.get(key) for key in immutable_manifest_keys}
            if resumed['adapter_schema']!=adapter.schema or saved_contract!=current_contract:
                raise ValueError('Resume physical model/execution source mismatch')
            if not (args.resume_reset_schedule or args.resume_next_cycle):
                saved_objective=resumed['config'].get('approach_objective','task')
                if args.approach_objective!=saved_objective:
                    raise ValueError('An Approach objective change requires a fresh schedule; inherited updates cannot count toward the new objective')
            world=resumed['world']
            world_optimizer=resumed['world_optimizer']
            natural_rng=resumed.get('natural_flight_rng')
            if args.resume_wm_replay:
                recipe=dict(natural_flight_batch=args.wm_natural_flight_batch,natural_flight_loss=args.wm_natural_flight_loss,
                            separate_supervision=args.wm_separate_supervision)
                recipe['active_flight_batch']=args.wm_active_flight_batch
                recipe['direct_long_body_supervision']=args.wm_direct_long_body_supervision
                if args.requested_world_lr is not None:
                    recipe['learning_rate']=args.requested_world_lr
                world,wm_state,replay_handoff=forward_guard.replay_for_resume(args.resume_wm_replay,args.resume_training,
                                                                            resumed,wm_holdout_lanes,recipe)
                world_optimizer=wm_state['optimizer']
                natural_rng=wm_state['natural_flight_rng']
            actor.load_state_dict(resumed['actor'],strict=True)
            adapter.load_state_dict(world,strict=True)
            if forward_guard is not None:
                resume_check=forward_guard.check(adapter)
                write(args.output/'resumed_forward_validation.json',resume_check)
                if not resume_check['passed'] and not research:
                    raise ValueError('Restored online WM failed fixed native forward validation')
            if reference_bank is None and resumed.get('reference_bank') is not None:
                from quadsmash.rl.supermimic_task_space_reference_v1 import TaskSpaceReferenceBankV1
                trainer.reference_bank=TaskSpaceReferenceBankV1.from_payload(resumed['reference_bank']).to('cuda:0')
                trainer.reference_bank.assert_excludes(wm_holdout_lanes)
            for attr,key in (('accepted_snapshot','step'),('block_start_snapshot','block'),('native_accepted_snapshot','native')):
                if key in resumed.get('actor_references',{}):setattr(trainer,attr,resumed['actor_references'][key])
            trainer.critic.load_state_dict(resumed['critic'],strict=True)
            if 'actor_optimizers' in resumed:
                trainer._pending_actor_optimizer_states=dict(resumed['actor_optimizers'])
                if 'joint' in trainer._pending_actor_optimizer_states:
                    trainer._actor_optimizers['joint'].load_state_dict(trainer._pending_actor_optimizer_states.pop('joint'))
            else:
                trainer.actor_opt.load_state_dict(resumed['actor_optimizer'])
            args.world_learning_rate=trainer.world_lr=restore_world_optimizer(
                trainer.world_opt,world_optimizer,args.requested_world_lr)
            if natural_flight is not None and natural_rng is not None:
                natural_flight.generator.set_state(natural_rng.cpu())

            trainer.critic_opt.load_state_dict(resumed['critic_optimizer'])
            for name in ('actor_updates','world_updates','critic_updates','last_critic_loss'):setattr(trainer,name,resumed[name])
            tick=int(resumed['native_ticks'])
            resumed_global=dict(resumed.get('global_counters',{}))
            critic_coverage_blocks=int(resumed.get('critic_coverage_blocks',0))
            if trainer.frozen_hash()!=trainer.frozen_actor_hash:raise ValueError('Resume changed frozen actor parameters')
            if resumed['config'].get('landing_and_first_contact_objectives') and not args.landing_and_first_contact_objectives:
                raise ValueError('Resume must preserve the revised research objective')
            if args.landing_and_first_contact_objectives:
                replay.recovery_failure_prefixes = True
                replay.recovery_training_mask = torch.ones(replay.lanes, dtype=torch.bool)
                replay.recovery_training_mask[list(wm_holdout_lanes)] = False
            replay_archives=[] if args.resume_without_replay else resumed['replay_archives']
            # A direct Approach resume must retain the Recovery-generated
            # online transitions; skipping them breaks the native-data→WM
            # closed loop.  Rebuild every committed archive before restoring
            # sampler metadata.
            replay_receipts=None if args.resume_without_replay else resumed.get('replay_archive_receipts')
            if replay_receipts is not None and len(replay_receipts)!=len(replay_archives):raise ValueError('Replay receipt/archive count mismatch')
            for archive_index,directory in enumerate(replay_archives):
                replay.new_stream(source='online',policy_version=int(resumed_global.get('accepted',0)))
                directory=Path(directory)
                receipt=replay_receipts[archive_index] if replay_receipts is not None else json.loads((directory/'manifest.json').read_text())
                for item in receipt['chunks']:
                    from quadsmash.rl.supermimic_archive_storage_v1 import read_archive_rows
                    with read_archive_rows(directory,item,scratch=args.output) as rows:
                        for row in rows:
                            replay.append(row,retain_start=False,policy_version=row.get('policy_version',0))
                    del rows
            if 'actor_start_bank' in resumed:
                replay.actor_start_bank.load_state_dict(tensor_tree_copy(resumed['actor_start_bank'],cpu=True))
            elif args.resume_start_block=='approach':
                raise ValueError('Approach continuation lacks persisted dynamic actor starts; rebuild audited starts from native archives first')
            if 'dynamic_stage_starts' in resumed:
                rebuilt = {}
                if args.landing_and_first_contact_objectives:
                    for name in ('post_touchdown','touchdown','recovery_failure','stable_recovery','post_landing_support_loss'):
                        rebuilt[name] = [dict(record, _actor_lane_mask=mask, _source_lanes=replay.lanes)
                                         for record,mask in replay.restore_stages.get(name, ())]
                    rebuilt['pre_body_ground'] = replay.stage_starts.get('pre_body_ground', [])
                replay.restore_stages.clear()
                replay.stage_starts=tensor_tree_copy(resumed['dynamic_stage_starts'],cpu=True)
                replay.stage_starts.update(rebuilt)
            replay.restore_prefix.clear()
            replay.generator.set_state(resumed['replay_rng'].cpu())
            replay_state=resumed.get('replay_state',{})
            saved_meta=resumed.get('replay_metadata')
            if not args.resume_without_replay:
                replay.current_source=str(replay_state.get('current_source','online'))
                replay.current_stream=int(replay_state.get('current_stream',replay.current_stream))
                replay.current_policy_version=int(replay_state.get('current_policy_version',resumed_global.get('accepted',0)))
                replay._native_tick_counter=int(replay_state.get('native_tick_counter',replay._native_tick_counter))
                if saved_meta is not None and (replay.count!=int(replay_state.get('count',-1)) or replay.write_pos!=int(replay_state.get('write_pos',-1))):
                    raise ValueError('Replay tensor reconstruction differs from checkpoint metadata; refusing mismatched resume')
            if saved_meta is not None and not args.resume_without_replay:
                for key,value in saved_meta.get('approach_metadata',{}).items():
                    if value.shape!=replay.approach_metadata[key].shape:raise ValueError('Replay Approach metadata shape mismatch')
                    replay.approach_metadata[key].copy_(value.cpu())
                for name in ('streams','stage_ids','source_ids','policy_versions','native_ticks'):
                    if name == 'stage_ids' and args.landing_and_first_contact_objectives:
                        continue  # Reclassified from the committed replay's causal landing history.
                    value=saved_meta.get(name)
                    target=getattr(replay,name,None)
                    if value is not None and target is not None and tuple(value.shape)==tuple(target.shape):
                        target.copy_(value)
            if resumed.get('runtime_validation_archive'):
                saved_validation=resumed['runtime_validation_archive']
                from quadsmash.rl.supermimic_long_run_contract_v1 import sha256
                if sha256(saved_validation['path'])!=saved_validation['sha256']:raise ValueError('Saved validation archive changed')
                runtime_validation_path=saved_validation['path']
                runtime_validation_data=torch.load(runtime_validation_path,map_location='cpu',weights_only=True)['tensors']
            torch.set_rng_state(resumed['torch_rng'].cpu());torch.cuda.set_rng_state(resumed['cuda_rng'].cpu())
        from scripts.compare_supermimic_fixed_stages import restore_training_comparison, training_comparison_context
        baseline_file_sha = None
        if args.paired_evaluation_baseline is not None:
            baseline_file_sha = hashlib.sha256(args.paired_evaluation_baseline.read_bytes()).hexdigest()
            continuing = args.resume_training and not (args.resume_reset_schedule or args.resume_next_cycle)
            paired_eval_snapshots, paired_eval_actor_snapshots, baseline_hash = restore_training_comparison(
                args.paired_evaluation_baseline, actor.state_dict(), policy_execution, manifest['task_contract'],
                continuation=resumed if continuing else None)
            baseline_payload=paired_eval_snapshots['pre-update']
            write(paired_eval_dir/'pre-update.json',dict(schema='supermimic-paired-fixed-case-evaluation-v1',stage='pre-update',
                actor_sha256=baseline_hash,case_count=len(baseline_payload),case_ids=sorted(str(row.get('case_id')) for row in baseline_payload),
                records=baseline_payload,actor_parameter_delta={'available':False},
                scope='fixed 128-case reference supplied by user'))
        from quadsmash.rl.supermimic_sequential_progress_v1 import SequentialProgress
        schedule=SequentialProgress.for_run(resumed.get('sequential_progress') if args.resume_training else None,
                                            fresh=args.resume_reset_schedule or args.resume_next_cycle)
        if args.resume_training and not args.resume_next_cycle and not args.resume_reset_schedule:
            if not (args.extend_completed_research or args.landing_and_first_contact_objectives) and args.continuous_approach_starts != resumed['config'].get('continuous_approach_starts',False):
                raise ValueError('Resume changes the Approach start quota schedule')
            if alternating is None and schedule.complete():raise ValueError('Completed schedule requires explicit resume-next-cycle and native acceptance')
        if args.resume_training and not args.resume_next_cycle:
            saved_alternating=resumed.get('alternating_schedule')
            if alternating is not None and not args.resume_reset_schedule:
                if saved_alternating is None:raise ValueError('Legacy checkpoint has no alternating cursor; no silent schedule migration')
                if (args.landing_and_first_contact_objectives and args.research_contact_gradient == 'G0'
                        and saved_alternating['execution_config'].get('contact_gradient') == 'G1'):
                    # Fork the control only from a prefix that has never executed an auxiliary update.
                    retained = resumed['global_counters']['accepted'] - resumed['config']['research_origin']
                    handoff = resumed['metadata'].get('research_revision_handoff', {})
                    from quadsmash.rl.supermimic_long_run_contract_v1 import sha256 as checkpoint_sha
                    source_block = handoff.get('pre_auxiliary_checkpoint')
                    revised_prefix = resumed['config'].get('landing_and_first_contact_objectives')
                    verified_block = (source_block and Path(source_block).is_file()
                                      and checkpoint_sha(Path(source_block)) == handoff.get('source_checkpoint_sha256')
                                      and handoff.get('pause_actor') == handoff.get('starting_actor'))
                    if (retained > 40 or handoff.get('common_prefix_aux_updates') != 0
                            or handoff.get('starting_actor') != tensor_digest(resumed['actor'].items())
                            or (revised_prefix and not verified_block)):
                        raise ValueError('Control fork requires the common pre-auxiliary G1 prefix')
                    saved_alternating = dict(saved_alternating, execution_config=dict(
                        saved_alternating['execution_config'], contact_gradient='G0'))
                alternating.load_state_dict(saved_alternating, extend_completed=args.extend_completed_research)
                if alternating.complete:raise ValueError('Completed alternating cycle requires explicit native acceptance and resume-next-cycle')
            elif saved_alternating is not None and alternating is None and not args.resume_reset_schedule:
                raise ValueError('Cannot silently resume alternating checkpoint in legacy layout')
        if args.resume_next_cycle:
            from quadsmash.rl.supermimic_long_run_contract_v1 import verify_native_promotion
            from quadsmash.rl.supermimic_policy_execution_v1 import contract as execution_contract
            promotion=verify_native_promotion(args.native_acceptance_receipt,args.resume_training,expected_execution=execution_contract())
            trainer.native_accepted_snapshot={k:v.detach().clone() for k,v in actor.state_dict().items()}
            write(args.output/'native_reference_promotion.json',promotion)
        source_path=Path(manifest['source_checkpoint']);source=torch.load(source_path,map_location='cpu',weights_only=False)
        # Older native archives do not carry the execution-root field.  The
        # explicit CLI path is the authoritative runtime contract and keeps
        # those archives usable without weakening the task-identity checks.
        execution_root = args.g07_execution_root
        if execution_root is None:
            recorded_root = manifest.get('g07_execution_root')
            if recorded_root is None:
                recorded_root = manifest.get('init_policy_checkpoint', {}).get('g07_execution_root')
            if recorded_root is None:
                raise ValueError(
                    'Missing g07_execution_root. Pass --g07-execution-root for legacy archives.'
                )
            execution_root = Path(recorded_root)
        execution_root = Path(execution_root).resolve()
        replay.recovery_failure_prefixes = args.recovery_prefix_and_string_margin
        if args.resume_training and not args.resume_reset_schedule:
            args.recovery_rollout_landing |= resumed['config'].get('recovery_rollout_landing',False)
        if args.recovery_rollout_landing and not args.landing_and_first_contact_objectives:
            raise ValueError('Rollout landing requires the post-landing Recovery objective')
        trainer.recovery_rollout_landing = args.recovery_rollout_landing
        trainer.landing_and_first_contact_objectives = args.landing_and_first_contact_objectives
        if args.landing_and_first_contact_objectives:
            from quadsmash.physics.full_racket_v4 import load_racket, string_strips
            trainer.string_strips = len(string_strips(load_racket(execution_root/'configs/full_racket_v4_r3.json')))
        if args.recovery_prefix_and_string_margin:
            from quadsmash.rl.supermimic_toi_racket_objective import string_boundary
            trainer.string_boundary = string_boundary(execution_root/'configs/full_racket_v4_r3.json',
                                                      next(actor.parameters()))
        contract,protocol,cases,receipt=resolve_task_contract(
            source, source_path, execution_root=execution_root, root=execution_root)
        if contract!=manifest['task_contract']:raise ValueError('Native task contract changed')
        cfg=make_task_cfg(128,contract,protocol,receipt);cfg.seed=NATIVE_TRAIN128_SEED;cfg.sim.device='cuda:0'
        class TrainingEnv(SuperMimicEnvV1,FrozenPanelEnv):pass
        phase='ENV';stage(args.output,'BEFORE_ENV')
        env=TrainingEnv(cfg,cases=cases,reset_noise=zero_panel_reset_noise(128,NATIVE_TRAIN128_SEED),
            residual_policy_checkpoint=source,g07_execution_root=execution_root,
            native_return_gamma=timebase.gamma,contact_flight_recovery=contract['contact_flight_recovery'])
        wrapper=RslRlVecEnvWrapper(env,clip_actions=None);obs,_=wrapper.reset()
        if args.resume_training:
            # Environment seeding must not overwrite the resumed learner RNG.
            torch.set_rng_state(resumed['torch_rng'].cpu());torch.cuda.set_rng_state(resumed['cuda_rng'].cpu())
        training_seed = None
        if research:
            training_seed = research_learner_rng(args.research_training_seed,
                resumed['config'] if args.resume_training else {},
                fresh=not args.resume_training or args.resume_reset_schedule, replay=replay, natural_flight=natural_flight)
        stage(args.output,'AFTER_RESET')
        from quadsmash.rl.supermimic_policy_execution_v1 import policy_forward,resolve_contract
        if args.resume_training and resolve_contract(resumed)[0]!=policy_execution:
            raise ValueError('Resume would change trained recovery gate')
        def applicable_probe(head,current):
            from quadsmash.rl.supermimic_evidence_binding_v1 import probe_applicability
            from quadsmash.rl.supermimic_probe_region_v1 import assess_region
            probe=dual_quality['native_action_probes'][head]
            assessment=None
            if probe.get('validity_region') is not None:
                def preview(row):
                    state=as_state(row).branch()
                    _,final,target=trainer._preview_command(state)
                    return final,target
                assessment=assess_region(probe['validity_region'],args.contact_validation_receipt,
                                         current,adapter,preview,head)
            return probe_applicability(probe,current,assessment)
        if args.action_ball_checkpoint is not None:
            # A model-only candidate is permitted for the calibration stages, where
            # the actor is blocked or contact gradients are detached.  It must not
            # silently enter a promotable actor run without current paired-action
            # evidence for the exact actor and WM that will be updated.
            from quadsmash.rl.supermimic_evidence_binding_v1 import runtime_identity,probe_applicability
            if args.training_stage == 'full_contact':
                if dual_quality is None:
                    raise ValueError('Action-ball candidate requires calibrated dual-head physical evidence')
                version=int(resumed_global.get('accepted',0)) if args.resume_training else 0
                current=runtime_identity(actor,adapter,manifest['task_contract'],policy_execution,version)
                invalid={head:applicable_probe(head,current)
                         for head in ('recovery','approach')}
                if any(not result['current_version_valid'] for result in invalid.values()):
                    write(args.output/'action_ball_native_probe_rejected.json',invalid)
                    raise ValueError('Action-ball candidate lacks CURRENT actor/WM native paired-action evidence')
            else:
                write(args.output/'action_ball_native_probe_deferred.json',dict(
                    training_stage=args.training_stage,
                    reason='Actor contact updates are not promotable in this stage',
                    action_ball_checkpoint=str(args.action_ball_checkpoint.resolve()),
                    dual_head_quality_available=dual_quality is not None))
        behavior_key = behavior_actor_hash = None
        def native_actor_action(observation):
            nonlocal behavior_key, behavior_actor_hash
            key = (global_accepted_id, trainer.actor_updates)
            if key != behavior_key:
                behavior_actor_hash = tensor_digest(actor.state_dict().items())
                behavior_key = key
            return policy_forward(actor,observation['policy'],env.rally,sample=False)[0]
        def encode_native_value_start(row):
            from quadsmash.rl.supermimic_policy_execution_v1 import recovery_gate
            with torch.no_grad():
                state=as_state(row);rally=state.task_memory.get('rally',{})
                recovery=recovery_gate(rally,state.observation)
                approach=(actor.approach_gate(state.observation).reshape(-1)>.25)&(rally['had_contact'].reshape(-1)<.5)
                return dict(critic_input=adapter.critic_input(state).detach(),head_masks=dict(recovery=recovery,approach=approach))
        def training_lane_available(row):
            return bool(training_start_mask(row,wm_holdout_lanes).any())
        def training_state(row,*,compact=False):
            if compact and '_actor_lane_mask' in row:
                from quadsmash.rl.supermimic_compact_starts import compact_start
                lanes=training_start_mask(row,wm_holdout_lanes)
                small,small_launch=compact_start(row,lanes,select_launch_for_start(row,launch))
                return PhysicalImaginationStateV1.from_record(small,executor_template=template,launch=small_launch,
                                                               device='cuda:0',array_plan_observations=True)
            state=as_state(row)
            if wm_holdout_lanes:
                state.clock_goal_context['actor_lane_mask']=training_start_mask(row,wm_holdout_lanes).to(state.physical.device)
            return state
        @torch.no_grad()
        def toi_start_lanes(row, candidate_mask=None):
            from quadsmash.rl.supermimic_toi_racket_objective import bind_intercept_reference, compact_intercept_preview, preview_intercept_lanes
            allowed=training_start_mask(row,wm_holdout_lanes).reshape(-1).cpu()
            if candidate_mask is not None:
                allowed &= candidate_mask
            if research:
                return compact_intercept_preview(row, allowed, select_launch_for_start(row,launch), template, trainer, 'cuda:0')
            state=training_state(row)
            reference=bind_intercept_reference(state,trainer.reference_bank,adapter.physical_model.geometry,
                                               timebase.imagination_steps*timebase.control_dt_s,plane_offset_m=trainer.toi_plane_offset_m)
            if reference is None:
                return torch.zeros_like(allowed)
            return preview_intercept_lanes(state,reference,allowed.to(state.physical.device),trainer).cpu()

        from quadsmash.rl.supermimic_frozen_refresh_v1 import FrozenRefresh
        from quadsmash.rl.supermimic_approach_data_v1 import native_approach_metadata
        from quadsmash.rl.supermimic_frozen_return_monitor_v1 import FrozenReturnMonitor
        frozen_return_monitor=FrozenReturnMonitor(args.native_return_drop_tolerance,args.native_return_flat_tolerance,args.native_return_patience)
        if args.resume_training and resumed.get('frozen_return_monitor') is not None:
            frozen_return_monitor.load_state_dict(resumed['frozen_return_monitor'])
        elif args.resume_training and args.training_profile=='large':
            raise ValueError('Large resume requires saved native-return monitoring state; do not reset trend history')
        bootstrap_actor=None
        refresh_collector=None; frozen_policy_id=None; refresh_serial=int(resumed.get('refresh_serial',0)) if args.resume_training else 0
        archive_flush_rows=max(32,int(os.environ.get('QUADSMASH_ARCHIVE_FLUSH_ROWS','32')))
        archive_refresh_only=os.environ.get('QUADSMASH_ARCHIVE_REFRESH_ONLY','0')=='1'
        if archive_refresh_only:raise ValueError('ARCHIVE_REFRESH_ONLY drops live training transitions; unsupported in the data-closed-loop path')
        from quadsmash.rl.supermimic_archive_storage_v1 import require_archive_space,write_archive_chunk
        class TrainingArchive(PhysicalTransitionArchive):
            def finish_record(self,*values):
                nonlocal refresh_collector
                super().finish_record(*values)
                row=self.rows[-1]
                sampling_metadata=native_approach_metadata(row,actor,adapter.physical_model.geometry)
                row['sampling_metadata']=sampling_metadata
                row['policy_version']=global_accepted_id
                row['actor_state_sha256']=behavior_actor_hash
                row['source_id']='online'
                row['block_name']='combined_refresh' if frozen_policy_id is not None else trainer.block_name
                row['policy_id']=frozen_policy_id or f'mixed_v{global_accepted_id}'
                row['collection_mode']='frozen_combination' if frozen_policy_id is not None else 'mixed_training'
                if bootstrap_actor is not None:
                    row.update(block_name='wm_bootstrap', policy_id=bootstrap_actor,
                               collection_mode='frozen_wm_bootstrap')
                row['stream_id']=replay.current_stream
                row['native_tick']=tick
                record = replay.append(row,policy_version=global_accepted_id,native_tick=tick,
                                       approach_metadata=sampling_metadata,return_record=refresh_collector is not None)
                if refresh_collector is not None:
                    refresh_collector.append(row, record=record)
                if len(self.rows)>=archive_flush_rows:
                    if getattr(self,'pipeline',None) is None:self.flush()
                    else:self.pipeline.enqueue()

            def flush(self):
                if getattr(self,'pipeline',None) is not None:
                    self.pipeline.enqueue()
                    self.pipeline.drain()
                    self._manifest()
                    return
                if os.environ.get('QUADSMASH_ZSTD_ARCHIVE','0')=='1':
                    return self.flush_compressed()
                if self.rows:
                    name=f'chunk_{len(self.chunks):05d}.pt'
                    path=self.directory/name
                    require_archive_space(self.directory)
                    payload=io.BytesIO()
                    torch.save(dict(schema=self.schema,rows=tensor_tree_copy(self.rows,cpu=True)),payload)
                    saved=write_archive_chunk(path,payload.getbuffer())
                    self.chunks.append(dict(file=name,ticks=len(self.rows),**saved))
                    self.rows.clear()
                self._manifest()
        startup_identity=json.loads((args.output/'startup.json').read_text())
        archive=TrainingArchive(args.output/'native_transitions',metadata=dict(manifest,policy_execution=policy_execution,control_ticks=0,transitions=0,chunks=[],
            initial_archive_identity={k:manifest.get(k) for k in ('policy_id','actor_state_sha256','runtime_commit')},
            behavior_policy_scope='per_transition',policy_id='mixed-training',actor_state_sha256=None,
            runtime_commit=startup_identity['commit'],runtime_script_sha256=startup_identity['script_sha256'],
            scope='Native training transitions with per-row behavior identity',actor_update_budget=args.recovery_update_budget+args.approach_update_budget,
            replay_source_schema=['recent_online','older_online','initial_archive'],
            replay_stage_schema=[*STAGE_NAMES,'approach_active','hit_prepare','contact_release','approach_failure','post_hit']),chunk_ticks=1000000,compact_plan_observations=True)
        if research and os.environ.get('QUADSMASH_ZSTD_ARCHIVE','0')=='1':
            from quadsmash.rl.supermimic_archive_pipeline import ArchivePipeline
            archive.pipeline=ArchivePipeline(archive)
        archive.attach_task_sequence(env);env.physical_transition_archive=archive
        # The packed 800-tick archive is a separate stream from this fresh
        # native rollout.  Prevent windows from crossing the reset boundary
        # even when episode indices are reused by the fixed panel.
        replay.new_stream(source='online', policy_version=0)
        def file_sha(path):
            return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None
        reference_path=args.reference_policy_checkpoint or source_path
        if file_sha(reference_path)!=manifest['source_sha256']:
            raise ValueError('Reference policy checkpoint does not match the verified source actor')
        bad_meta=dict(path=str(args.bad_policy_checkpoint.resolve()),sha256=file_sha(args.bad_policy_checkpoint)) if args.bad_policy_checkpoint else None
        interleaved_approach_starts = bool(args.interleaved_approach_starts or args.extend_completed_research or
            (args.resume_training and resumed['config'].get('interleaved_approach_starts', False)))
        continuous_recovery_starts = bool(args.extend_completed_research or
            (args.resume_training and resumed['config'].get('continuous_recovery_starts', False)))
        trainer.restart_uphill_approach_momentum = bool(args.restart_uphill_approach_momentum or
            (args.resume_training and resumed['config'].get('restart_uphill_approach_momentum', False)))
        config=dict(restart_uphill_approach_momentum=trainer.restart_uphill_approach_momentum,interleaved_approach_starts=interleaved_approach_starts,outgoing_first_contact_semantics=('string_only_v1' if args.landing_and_first_contact_objectives else 'any_racket'),recovery_rollout_landing=args.recovery_rollout_landing,landing_and_first_contact_objectives=args.landing_and_first_contact_objectives,recovery_prefix_and_string_margin=args.recovery_prefix_and_string_margin,continuous_approach_starts=args.continuous_approach_starts,toi_effective_start_lanes=args.toi_effective_start_lanes,recovery_effective_start_lanes=args.recovery_effective_start_lanes,sequential_layout=args.sequential_layout,head_block_size=args.head_block_size,refresh_mode=args.refresh_mode,alternating_plan=alternating.plan if alternating is not None else None,dual_head_quality=dual_quality,wm_holdout_lanes=wm_holdout_lanes,runtime_wm_bounds=runtime_wm_bounds,training_profile=args.training_profile,contact_validation=contact_validation,resume_next_cycle=args.resume_next_cycle,resume_reset_schedule=args.resume_reset_schedule,resume_without_replay=args.resume_without_replay,update_mode=args.update_mode,resume_start_block=args.resume_start_block,ball_side_mode=args.ball_side_mode,cycles=args.cycles,recovery_proposal_budget=args.recovery_proposal_budget,approach_proposal_budget=args.approach_proposal_budget,refresh_ticks=args.refresh_ticks,refresh_world_updates=args.refresh_world_updates,refresh_critic_updates=args.refresh_critic_updates,replay_capacity_ticks=args.replay_capacity_ticks,actor_update_budget=(args.recovery_update_budget+args.approach_update_budget if args.update_mode=='sequential' else args.joint_update_budget),joint_update_budget=args.joint_update_budget,recovery_update_budget=args.recovery_update_budget,approach_update_budget=args.approach_update_budget,native_ticks_per_actor_update=args.native_ticks_per_actor_update,world_updates_per_actor_update=args.world_updates_per_actor_update,
            world_batch=args.wm_batch_size,imagination_batch=128,timebase=timebase.manifest(),actor_lr=trainer.actor_lr,actor_anchor=trainer.anchor_coeff,world_lr=args.world_learning_rate,critic_lr=2e-4,
            wm_forward_guard_sha256=forward_guard.spec_sha256 if forward_guard else None,wm_replay_handoff=replay_handoff,
            wm_separate_supervision=args.wm_separate_supervision,wm_direct_long_body_supervision=args.wm_direct_long_body_supervision,wm_active_flight_batch=args.wm_active_flight_batch,wm_natural_flight_batch=args.wm_natural_flight_batch,wm_natural_flight_loss=args.wm_natural_flight_loss,
            wm_action_collections=action_collections,wm_sampling_mode=args.wm_sampling_mode,
            native_return_monitor=frozen_return_monitor.config,
            policy_anchor=.05,critic_warmup_updates=200,critic_max_loss=None if args.diagnostic_critic_loss else 50.,
            shuffle_wm_quota_groups=args.shuffle_wm_quota_groups,
            continuous_recovery_starts=continuous_recovery_starts,
            diagnostic_critic_loss=args.diagnostic_critic_loss,critic_mse_reference=50.,max_critic_only_blocks=32,
            command_delta_budget=args.command_delta_budget,target_delta_budget_rad=args.target_delta_budget_rad,
            native_reference_command_budget=args.native_reference_command_budget,
            native_reference_target_budget_rad=args.native_reference_target_budget_rad,
            actor_whitelist={'joint':RECOVERY_NAMES + APPROACH_NAMES,'recovery':RECOVERY_NAMES,'approach':APPROACH_NAMES},source_sha256=manifest['source_sha256'],packed_data_sha256=packed_sha,
            initialization_native_return=expected['native_return_mean'],extra_mpc=0,
            reference_policy=dict(path=str(reference_path),sha256=file_sha(reference_path)),bad_policy=bad_meta,
            sampling='source×event stratified WM windows; actor starts are causal native event states',
            wm_batch_size=args.wm_batch_size,wm_source_quota=args.wm_source_quota,wm_stage_quota=args.wm_stage_quota,
            wm_approach_stage_quota=args.wm_approach_stage_quota,wm_contact_stage_quota=args.wm_contact_stage_quota,
            approach_sampling_thresholds=dict(gate=.25,minimum_gate_steps=4,near_distance_m=.30),approach_refresh_interval=args.approach_refresh_interval,
            wm_latest_policy_windows=args.wm_latest_policy_windows or 0,
            wm_recent_policy_window=args.wm_recent_policy_window,wm_recent_tick_window=args.wm_recent_tick_window,
            task_reference_bank=str(args.task_reference_bank.resolve()) if args.task_reference_bank else None,
            action_ball_checkpoint=str(args.action_ball_checkpoint.resolve()) if args.action_ball_checkpoint else None,
            tracking_weight=args.tracking_weight,approach_objective=args.approach_objective,toi_racket_weight=args.toi_racket_weight,
            use_substep_toi_references=args.use_substep_toi_references,intercept_plane_calibration=plane_calibration,
            paired_evaluation_baseline=str(args.paired_evaluation_baseline.resolve()) if args.paired_evaluation_baseline else None,
            actor_start_ticks=APPROACH_TICKS,critic_start_ticks=CRITIC_TICKS,critic_coverage_period=len(CRITIC_TICKS))
        metadata=dict(wm_action_collections=action_collections,policy_execution=policy_execution,init_policy_checkpoint=dict(path=str(source_path),sha256=manifest['source_sha256'],g07_execution_root=str(execution_root)),
                      task_contract=contract,physical_imagination=adapter.manifest)
        global_attempt_id=0;global_accepted_id=0;global_rejected=0;global_backtracked=0
        if args.resume_training:
            global_attempt_id=int(resumed_global.get('attempts',0));global_accepted_id=int(resumed_global.get('accepted',0))
            global_rejected=int(resumed_global.get('rejected',0));global_backtracked=int(resumed_global.get('backtracked',0))
        research_origin=(global_accepted_id if args.resume_reset_schedule else resumed.get('config',{}).get('research_origin',global_accepted_id)) if research and args.resume_training else 0
        config['research_origin']=research_origin
        if research: config['research_training_seed']=training_seed
        write(args.output/'config.json',config)
        elapsed_offset = 0.
        if research and args.resume_training and not args.resume_reset_schedule:
            elapsed_offset = resumed.get('research_elapsed_s', 0.)
            if args.extend_completed_research and 'research_elapsed_s' not in resumed:
                # The first G1 process predates checkpoint timing. Its launcher
                # measured the entire training process, including replay rebuild.
                previous_exit = args.resume_training.resolve().parent.parent/'exit.json'
                elapsed_offset = json.loads(previous_exit.read_text())['wall_seconds']
        research_started=time.monotonic()-elapsed_offset
        from quadsmash.rl.supermimic_evidence_binding_v1 import runtime_identity,probe_applicability
        loaded_identity=runtime_identity(actor,adapter,manifest['task_contract'],policy_execution,global_accepted_id)
        invocation_start=dict(actor_sha256=loaded_identity['actor_state_sha256'],accepted=global_accepted_id,
                              attempts=global_attempt_id,world=trainer.world_updates,critic=trainer.critic_updates)
        invocation_heads={head:dict(accepted=0,attempts=0) for head in ('recovery','approach')}
        def invocation_receipt():
            accepted=global_accepted_id-invocation_start['accepted']
            attempts=global_attempt_id-invocation_start['attempts']
            if sum(h['accepted'] for h in invocation_heads.values())!=accepted or sum(h['attempts'] for h in invocation_heads.values())!=attempts:
                raise RuntimeError('Process-local head counters differ from optimizer proposal counters')
            return dict(start=invocation_start,actor_end_sha256=tensor_digest(actor.state_dict().items()),
                        actor_updates=accepted,actor_attempts=attempts,world_updates=trainer.world_updates-invocation_start['world'],
                        critic_updates=trainer.critic_updates-invocation_start['critic'],heads=invocation_heads,
                        approach_objective=args.approach_objective,scope='updates executed in this process only')
        write(args.output/'loaded_runtime_identity.json',dict(identity=loaded_identity,resumed=bool(args.resume_training),
            probe_applicability={} if dual_quality is None else {h:probe_applicability(dual_quality['native_action_probes'][h],loaded_identity) for h in ('recovery','approach')}))
        from quadsmash.rl.supermimic_evidence_binding_v1 import verify_fit_provenance,loaded_fit_provenance
        initial_fit_provenance=verify_fit_provenance(json.loads(args.contact_validation_receipt.read_text()),args.contact_validation_receipt) if args.contact_validation_receipt else dict(verified=False,reason='no-fit-provenance')
        loaded_fit=loaded_fit_provenance(initial_fit_provenance,resumed if args.resume_training else None,loaded_identity['world_state_sha256'],wm_holdout_lanes)
        write(args.output/'loaded_fit_provenance.json',loaded_fit)
        def save(name):
            archive.flush()  # Checkpoint never points beyond durable replay.
            atomic_torch_save(dict(algorithm='supermimic_v1',adapter_schema=adapter.schema,actor=actor.state_dict(),
                critic=trainer.critic.state_dict(),world=adapter.state_dict(),actor_optimizer=trainer.actor_opt.state_dict(),
                actor_optimizers=dict(getattr(trainer,'_pending_actor_optimizer_states',{}),**{name:opt.state_dict() for name,opt in trainer._actor_optimizers.items()}),
                world_optimizer=trainer.world_opt.state_dict(),critic_optimizer=trainer.critic_opt.state_dict(),
                actor_updates=trainer.actor_updates,actor_attempts=trainer.actor_attempts,
                rejected_actor_updates=trainer.rejected_actor_updates,world_updates=trainer.world_updates,critic_updates=trainer.critic_updates,
                research_elapsed_s=time.monotonic()-research_started,
                native_ticks=tick,metadata=metadata,config=config,last_critic_loss=trainer.last_critic_loss,
                critic_coverage_blocks=critic_coverage_blocks,
                global_counters=dict(attempts=global_attempt_id,accepted=global_accepted_id,
                                     rejected=global_rejected,backtracked=global_backtracked),
                replay_state=dict(count=int(replay.count),write_pos=int(replay.write_pos),
                                  current_stream=int(replay.current_stream),
                                  current_source=str(replay.current_source),
                                  current_policy_version=int(replay.current_policy_version),
                                  native_tick_counter=int(replay._native_tick_counter),
                                  source_names=['initial_archive','online'],
                                  stage_names=list(STAGE_NAMES)),
                runtime_validation_archive=dict(path=runtime_validation_path,sha256=file_sha(runtime_validation_path)),
                wm_fit_provenance=dict(verified=loaded_fit['verified'],world_state_sha256=tensor_digest(adapter.state_dict().items()),
                    excluded_lanes=wm_holdout_lanes,initial_manifest_sha256=initial_fit_provenance.get('manifest_sha256')),
                alternating_schedule=alternating.state_dict() if alternating is not None else None,refresh_serial=refresh_serial,
                frozen_return_monitor=frozen_return_monitor.state_dict(),
                native_value_bank=native_value_bank,sequential_progress=schedule.state_dict(),actor_start_bank=replay.actor_start_bank.state_dict(),dynamic_stage_starts=replay.stage_starts,
                reference_bank=trainer.reference_bank.payload() if trainer.reference_bank else None,
                actor_references=dict(step=trainer.accepted_snapshot,block=trainer.block_start_snapshot,native=trainer.native_accepted_snapshot),
                replay_metadata=dict(approach_metadata=replay.approach_metadata,
                    streams=replay.streams.cpu(), stage_ids=replay.stage_ids.cpu(),
                    source_ids=replay.source_ids.cpu(), policy_versions=replay.policy_versions.cpu(),
                    native_ticks=replay.native_ticks.cpu()),
                paired_evaluation_context=training_comparison_context(paired_eval_snapshots,paired_eval_actor_snapshots,baseline_file_sha),
                replay_archive_receipts=(replay_receipts if replay_receipts is not None else [json.loads((Path(d)/'manifest.json').read_text()) for d in replay_archives])+[json.loads((archive.directory/'manifest.json').read_text())],
                replay_archives=replay_archives+[str(archive.directory)],replay_rng=replay.generator.get_state(),
                natural_flight_rng=natural_flight.generator.get_state() if natural_flight is not None else None,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),args.output/name)
        def check_stop():
            if args.stop_request['signal'] is not None:raise InterruptedError(args.stop_request['signal'])
            if frozen_return_monitor.stop_reason:raise RuntimeError('Native return stop remains active: '+frozen_return_monitor.stop_reason)
        def ensure_critic():
            maximum=config['critic_max_loss']
            for retry in range(32):
                if critic_ready(trainer.critic_updates,trainer.last_critic_loss,maximum=maximum):return
                check_stop()
                state=training_state(starts[retry%len(starts)])
                stats=trainer.imagine_update(state,update_actor=False)
                print('PHYSICAL_CRITIC='+json.dumps(stats),flush=True)
                with (args.output/'critic_history.jsonl').open('a') as f:f.write(json.dumps(stats)+'\n')
                if not args.landing_and_first_contact_objectives:
                    save('latest.pt')
            if not critic_ready(trainer.critic_updates,trainer.last_critic_loss,maximum=maximum):
                raise RuntimeError('Contextual critic quality gate did not pass within finite32-block budget')
        stage_ticks=list(CRITIC_TICKS)
        # WM calibration is deliberately actor/critic blocked.  Constructing
        # dozens of full G07 imagination states and native value labels here
        # adds several minutes and does not affect the supervised WM update.
        # Keep the audit explicit so the candidate cannot be mistaken for a
        # policy-training checkpoint.
        from quadsmash.rl.supermimic_dual_head_quality_v1 import split_native_value_rows,critic_holdout,reference_coverage
        from scripts.index_supermimic_success_archive import archive_actor_sha256
        label_actor_sha256=archive_actor_sha256(manifest)
        if args.training_stage == 'wm_calibrate':
            native_value_bank=dict(
                inputs=torch.empty((0, 0)), rtg=torch.empty((0,)), identity=[],
                head_masks={'recovery':torch.empty((0,),dtype=torch.bool),
                            'approach':torch.empty((0,),dtype=torch.bool)},
                policy_id=manifest.get('policy_id','initial_archive_unknown_policy'),
                actor_sha256=label_actor_sha256, source_sha256=packed_sha,
                policy_execution=manifest.get('policy_execution'), accepted_updates=0,
                label_actor_state=None)
            write(args.output/'native_critic_stage_audit.json',dict(
                skipped=True, reason='wm_calibrate_actor_critic_blocked',
                source='packed_loop/data.pt', horizon_steps=int(rtg.shape[0])))
        else:
            stage_states=[as_state(start_cache[t]) for t in stage_ticks]
            stage_meta=[dict(policy_id='initial_reference',episode_id=start_cache[t]['before']['episode_index'].tolist(),time_s=float(t*timebase.control_dt_s),complete_lane_mask=rtg_complete[t]) for t in stage_ticks]
            native_calibration=trainer.audit_native_rtg_stages(stage_states,[rtg[t] for t in stage_ticks],stage_meta)
            native_calibration.update(source='packed_loop/data.pt',discount_mode='physical_time',
                                      horizon_steps=int(rtg.shape[0]),native_return_scope='frozen archive label-policy episodes; execution identity checked separately')
            write(args.output/'native_critic_stage_audit.json',native_calibration)
            if args.resume_training and resumed.get('native_value_bank') is not None:
                native_value_bank=resumed['native_value_bank']
            else:
                value_x=[];value_y=[];value_ids=[];value_masks={h:[] for h in ('recovery','approach')}
                for ti in sorted(start_cache):
                    encoded=encode_native_value_start(start_cache[ti]);valid=rtg_complete[ti].cpu()
                    if not valid.any():continue
                    value_x.append(encoded['critic_input'].cpu()[valid]);value_y.append(rtg[ti].float().cpu()[valid])
                    value_ids.extend([(lane,int(package['tensors']['episode'][ti,lane]),int(start_cache[ti]['before']['elapsed_tick'][lane])) for lane in torch.where(valid)[0].tolist()])
                    for h in value_masks:value_masks[h].append(encoded['head_masks'][h].cpu()[valid])
                native_value_bank=dict(inputs=torch.cat(value_x),rtg=torch.cat(value_y),identity=value_ids,
                    head_masks={h:torch.cat(v) for h,v in value_masks.items()},policy_id=manifest.get('policy_id','initial_archive_unknown_policy'),
                    actor_sha256=label_actor_sha256,source_sha256=packed_sha,policy_execution=manifest.get('policy_execution'),accepted_updates=0,
                    label_actor_state={k:v.detach().cpu().clone() for k,v in trainer.reference.state_dict().items()} if label_actor_sha256==tensor_digest(trainer.reference.state_dict().items()) else None)
        if args.training_stage != 'wm_calibrate' and not args.resume_training:
            from quadsmash.rl.supermimic_dual_head_quality_v1 import native_calibration_rows
            initial_ids,calibration_contract=native_calibration_rows(native_value_bank,policy_execution,wm_holdout_lanes,CRITIC_TICKS,expected_actor_sha256=tensor_digest(actor.state_dict().items()))
            if len(initial_ids):
                native_calibration_fit=trainer.calibrate_critic_tensors(native_value_bank['inputs'][initial_ids],native_value_bank['rtg'][initial_ids],epochs=2)
            else:
                native_calibration_fit=dict(optimizer_updates=0,skipped=True,reason=calibration_contract['reason'])
            native_calibration_fit.update(calibration_contract,discount_mode='physical_time',horizon_steps=int(rtg.shape[0]),
                native_return_scope='matching-execution frozen-policy completed episodes only; validation lanes excluded',epoch_budget=2)
            write(args.output/'native_critic_calibration.json',native_calibration_fit)
        def check_head_quality(head,attempt):
            coverage=reference_coverage(trainer.reference_bank,head,trainer.tracking_coefficient(head))
            write(args.output/(head+'_reference_coverage.json'),coverage)
            if dual_quality is None:
                return dict(enforced=False,reference=coverage,reason='no calibrated dual-head bounds configured')
            from quadsmash.rl.supermimic_wm_validation_v1 import runtime_head_check
            from quadsmash.rl.supermimic_evidence_binding_v1 import runtime_identity,probe_applicability,verify_fit_provenance
            identity=runtime_identity(actor,adapter,manifest['task_contract'],policy_execution,global_accepted_id)
            probe=applicable_probe(head,identity)
            fit_provenance=loaded_fit
            wm=runtime_head_check(adapter,runtime_validation_data,wm_holdout_lanes,launch,head,runtime_wm_bounds[head],
                minimum_event_episodes=dual_quality['wm_min_event_episodes'][head])
            drift=None
            if native_value_bank.get('label_actor_state') is not None and trainer.budget_states:
                with torch.no_grad():
                    current=trainer._budget_snapshot();label=trainer._snapshot_from(native_value_bank['label_actor_state'])
                    deltas=[trainer._max_budget_delta(a,b) for a,b in zip(current,label)]
                drift=dict(final_normalized_command_delta=max(d[1] for d in deltas),final_joint_target_delta_rad=max(d[2] for d in deltas))
            value=critic_holdout(trainer.critic,native_value_bank,head,wm_holdout_lanes,
                current_actor_sha256=tensor_digest(actor.state_dict().items()),current_execution=policy_execution,limits=dual_quality['critic_upper_bounds'][head],
                accepted_updates=global_accepted_id,command_delta=drift,trust_limits=dual_quality.get('critic_policy_drift_bounds',{}).get(head))
            result=dict(head=head,attempt=attempt,enforced=True,wm=wm,critic=value,reference=coverage,probe_applicability=probe,fit_provenance=fit_provenance,runtime_identity=identity,
                world_updates=trainer.world_updates,critic_updates=trainer.critic_updates)
            write(args.output/'latest_dual_head_quality.json',result)
            with (args.output/'dual_head_quality.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
            if writer is not None:
                step=tick*128
                writer.add_scalar('quality/'+head+'/wm_passed',float(wm['passed']),step)
                for event,horizons in wm['errors'].items():
                    for horizon,metrics in horizons.items():
                        for name,metric in metrics.items():
                            if isinstance(metric,(int,float)):writer.add_scalar('quality/'+head+'/wm/'+event+'/'+horizon+'/'+name,metric,step)
                writer.add_scalar('quality/'+head+'/critic_passed',float(value['passed']),step)
                for key,metric in value.get('metrics',{}).items():
                    if metric is not None:writer.add_scalar('quality/'+head+'/critic_'+key,metric,step)
                writer.add_scalar('tracking/'+head+'/verified_reference_count',coverage['verified_reference_count'],step)
                writer.flush()
            if not wm['passed'] or not value['passed'] or not probe['current_version_valid'] or not fit_provenance['verified']:
                write(args.output/'revalidation_required.json',dict(head=head,attempt=attempt,probe=probe,wm=wm,value=value))
                save('quality_stopped.pt')
                raise RuntimeError(head+' model/probe provenance or independent WM/critic quality failed; actor disabled, no automatic extra fitting')
            return result
        # A already supplies complete native episodes. This warmup concerns the
        # new contextual critic, not recollecting a fresh 800 ticks each update.
        def cover_critic():
            nonlocal critic_coverage_blocks
            check_stop()
            selected=critic_coverage_blocks%len(critic_starts)
            stats=trainer.imagine_update(training_state(critic_starts[selected]),update_actor=False)
            critic_coverage_blocks+=1
            stats.update(coverage_blocks=critic_coverage_blocks,source_tick=CRITIC_TICKS[selected])
            with (args.output/'critic_coverage.jsonl').open('a') as f:f.write(json.dumps(stats)+'\n')
            print('PHYSICAL_CRITIC_COVERAGE='+json.dumps(stats),flush=True)
            if not args.landing_and_first_contact_objectives:
                save('latest.pt')
        phase='CRITIC'
        # WM calibration is an actor-blocked model-fitting stage.  It should
        # consume native windows directly instead of spending a full critic
        # coverage pass first.  Normal policy stages retain the complete
        # multi-phase critic warmup below.
        if args.training_stage != 'wm_calibrate':
            # Cover terminal and all preceding task phases before further actor
            # updates. Overlapping 0.64s windows, unchanged BPTT/GAE and
            # discount formula.
            while critic_coverage_blocks<len(critic_starts):cover_critic()
            ensure_critic()
        cursor=0;completed_returns=[];completed_discounted=[];phase='LEARNING'
        first_body_lanes = torch.zeros(replay.lanes, dtype=torch.bool)
        first_mixed_tick, first_refresh_serial = tick, refresh_serial
        episode_stats=NativeEpisodeStatistics()
        def verify_online_forward(head,attempt):
            if forward_guard is None:
                return None
            result=forward_guard.check(adapter)
            result.update(head=head,attempt=attempt,world_updates=trainer.world_updates)
            with (args.output/'online_forward_guard.jsonl').open('a') as log:
                log.write(json.dumps(result,allow_nan=False)+'\n')
            if writer is not None:
                writer.add_scalar('quality/fixed_native_forward_passed',float(result['passed']),trainer.world_updates)
            metadata['online_forward_validation']=dict(passed=result['passed'],world_sha256=result['world_sha256'],
                                                       spec_sha256=forward_guard.spec_sha256)
            if not result['passed'] and not research:
                write(args.output/'online_forward_rejected.json',result)
                raise RuntimeError('Online WM failed fixed native forward validation before downstream updates')
            return result
        def run_world_updates(iteration, count=None, head=None):
            """Fit WM with explicit source×event quotas and return audit totals."""
            head=head or trainer.block_name
            quotas=(args.wm_approach_stage_quota if head=='approach' else
                    args.wm_contact_stage_quota if head=='contact' else args.wm_stage_quota)
            # The calibration stage is a supervised diagnostic over the fixed
            # native archive.  The episode-capped integer allocator is required
            # for online mixed-policy training, but solving it 256 times over a
            # 102400-transition archive dominates the calibration wall time and
            # leaves the GPU idle.  Use the existing event-pool sampler here.
            calibration_fast_path = args.training_stage == 'wm_calibrate' and not action_collections
            sampling_source_quota = None if calibration_fast_path else args.wm_source_quota
            sampling_stage_quota = None if calibration_fast_path else quotas
            total=int(args.world_updates_per_actor_update if count is None else count)
            reports=[]; started=time.perf_counter()
            for offset in range(total):
                sampling_started=time.perf_counter()
                window,lanes,sampling=replay.window(
                    timebase.world_steps,args.wm_batch_size,iteration*total+offset,
                    source_quota=sampling_source_quota,stage_quota=sampling_stage_quota,head=head,
                    recent_policy_window=args.wm_recent_policy_window,
                    recent_tick_window=args.wm_recent_tick_window,excluded_lanes=wm_holdout_lanes,report_loss_groups=True,
                    shuffle_quota_groups=args.shuffle_wm_quota_groups,latest_policy_windows=args.wm_latest_policy_windows or 0)
                sampling['sampling_mode']=args.wm_sampling_mode
                sampling['promotion_gate']=sampling_promotion_gate(sampling,args.wm_sampling_mode)
                sampling['source_profile'] = {
                    'requested': sampling.get('source_requested', {}),
                    'actual': sampling.get('source_counts', {}),
                    'shortfall': sampling.get('source_shortfall', {}),
                    'effective_weight': sampling.get('effective_weight', {}).get('source', {}),
                }
                sampling['event_profile'] = {
                    'requested': sampling.get('stage_requested', {}),
                    'allocated': sampling.get('stage_counts', {}),
                    'actual': sampling.get('stage_actual_counts', sampling.get('stage_counts', {})),
                    'shortfall': sampling.get('stage_shortfall', {}),
                    'effective_weight': sampling.get('effective_weight', {}).get('stage', {}),
                }
                if args.wm_sampling_mode=='promotion' and not sampling['promotion_gate']['passed']:
                    write(args.output/'wm_sampling_failure.json', dict(
                        head=head, attempt=iteration, batch_in_refresh=offset, native_tick=tick,
                        world_updates_before_step=trainer.world_updates, sampling=sampling))
                    raise RuntimeError('WM promotion sampling contract failed: '+json.dumps(sampling['promotion_gate'],sort_keys=True))
                sampling_seconds=time.perf_counter()-sampling_started
                fit_started=time.perf_counter()
                stats=trainer.fit_world_window(window,launch[lanes],loss_groups=sampling.pop('loss_groups'),
                                               separate_supervision=args.wm_separate_supervision,long_native=sample_long_flight() if not research or offset%4==0 else None)
                if research and offset%2==0:
                    stats['contact_supervision']=trainer.fit_research_contact(window,launch[lanes])
                del window
                stats.update(sampling_seconds=sampling_seconds,fit_wall_seconds=time.perf_counter()-fit_started)
                if forward_guard is not None and (not research or offset+1==total):
                    stats['fixed_native_forward']=verify_online_forward(head,iteration)
                stats['sampling']=sampling; reports.append(stats)
                if calibration_fast_path and ((offset + 1) == total or (offset + 1) % 16 == 0):
                    print('WM_CALIBRATION_PROGRESS='+json.dumps(dict(
                        update=offset + 1, total=total,
                        world_loss=stats.get('world_loss'),
                        world_ball_loss=stats.get('world_ball_loss'),
                        world_action_ball_loss=stats.get('world_action_ball_loss'),
                        wall_seconds=time.perf_counter() - started)), flush=True)
                if writer is not None:
                    step=int(tick*128)
                    for key in ('world_loss','world_physical_loss','world_contact_loss','world_ball_loss','world_action_ball_loss','world_task_loss',
                                'world_physical_gradient','world_ball_gradient','world_action_ball_gradient','world_task_gradient'):
                        if key in stats: writer.add_scalar('wm/'+key.replace('world_',''),float(stats[key]),step)
                    for source,value in sampling.get('source_counts',{}).items():
                        writer.add_scalar('wm/source/'+source,float(value)/args.wm_batch_size,step)
                    for stage_name,value in sampling.get('stage_counts',{}).items():
                        writer.add_scalar('wm/stage/'+stage_name,float(value),step)
                    for stage_name,value in sampling.get('stage_shortfall',{}).items():
                        writer.add_scalar('wm/shortfall/'+stage_name,float(value),step)
                    for stage_name,value in sampling.get('actual_event_coverage',{}).items():
                        writer.add_scalar('wm/coverage/'+stage_name,float(value),step)
            aggregate={}
            for key in ('world_loss','world_gradient','world_physical_loss','world_contact_loss','world_ball_loss','world_action_ball_loss','world_task_loss',
                        'world_physical_gradient','world_ball_gradient','world_action_ball_gradient','world_task_gradient','world_seconds'):
                values=[float(item[key]) for item in reports if key in item]
                if values: aggregate[key]=sum(values)/len(values)
            aggregate.update(world_updates=len(reports),world_update_reports=reports,
                             world_update_seconds=time.perf_counter()-started)
            if total and forward_guard is not None:
                aggregate['fixed_native_forward']=reports[-1]['fixed_native_forward']
            return aggregate
        if args.wm_bootstrap_max_ticks and args.training_stage != 'wm_calibrate':
            from quadsmash.rl.supermimic_training_audit_v1 import collect_until_quota
            heads = {name:quota for name,budget,quota in (
                ('recovery',args.recovery_update_budget,args.wm_stage_quota),
                ('approach',args.approach_update_budget,args.wm_approach_stage_quota)) if budget > 0}
            bootstrap_start = tick
            bootstrap_actor = tensor_digest(actor.state_dict().items())
            def sample_bootstrap():
                reports = {}
                for name,quota in heads.items():
                    _,_,sample = replay.window(timebase.world_steps,args.wm_batch_size,0,
                        source_quota=args.wm_source_quota,stage_quota=quota,head=name,
                        recent_policy_window=args.wm_recent_policy_window,
                        recent_tick_window=args.wm_recent_tick_window,excluded_lanes=wm_holdout_lanes,report_loss_groups=True,
                        shuffle_quota_groups=args.shuffle_wm_quota_groups,latest_policy_windows=args.wm_latest_policy_windows or 0)
                    reports[name] = dict(source_counts=sample['source_counts'],stage_counts=sample['stage_counts'],
                                         gate=sampling_promotion_gate(sample,'promotion'))
                with (args.output/'wm_bootstrap.jsonl').open('a') as log:
                    log.write(json.dumps(dict(start_tick=bootstrap_start,tick=tick,actor_sha256=bootstrap_actor,
                                              reports=reports))+'\n')
                return all(report['gate']['passed'] for report in reports.values())
            def inspect_bootstrap():
                rng_state = replay.generator.get_state()
                try:
                    return sample_bootstrap()
                finally:
                    replay.generator.set_state(rng_state)
            @torch.inference_mode()
            def collect_bootstrap(count):
                nonlocal obs,tick,cursor
                for _ in range(count):
                    check_stop()
                    obs,_,_,_=wrapper.step(native_actor_action(obs))
                    tick+=1
                    if not torch.isfinite(obs['policy']).all():
                        raise FloatingPointError('Nonfinite native observation during WM bootstrap')
                    for raw in env.rally_records[cursor:]:
                        record=score_terminal_episode(raw,dict(protocol,_root=str(ROOT)))
                        replay.actor_start_bank.label_terminal(record,replay.current_stream)
                        with (args.output/'wm_bootstrap_episodes.jsonl').open('a') as log:
                            log.write(json.dumps(dict(record,bootstrap_actor_sha256=bootstrap_actor))+'\n')
                    cursor=len(env.rally_records)
                archive.flush()
            collect_until_quota(inspect_bootstrap,collect_bootstrap,args.wm_bootstrap_max_ticks)
            if tensor_digest(actor.state_dict().items()) != bootstrap_actor:
                raise RuntimeError('Actor changed during frozen native WM bootstrap')
            if tick != bootstrap_start:
                save('wm_bootstrap_ready.pt')
            bootstrap_actor = None
        if args.training_stage=='wm_calibrate':
            calibration=run_world_updates(0,count=args.wm_calibration_updates,head='contact')
            write(args.output/'wm_calibration_completed.json',dict(training_stage=args.training_stage,
                contact_bptt_mode=args.contact_bptt_mode,**calibration,actor_updates=0,
                critic_updates=trainer.critic_updates,formal_evaluation=False,
                promotion_status='candidate_unverified',
                promotion_contract='Run validate_supermimic_physical_wm.py on the fixed holdout and pass supermimic_wm_candidate_gate_v1 before actor training'))
            # This is deliberately a candidate artifact.  A model-loss
            # decrease is not sufficient to replace the historical anchor.
            save('wm_calibration_candidate.pt')
            write(args.output/'wm_calibration_promotion_required.json',dict(
                status='candidate_unverified',
                candidate=str(args.output/'wm_calibration_candidate.pt'),
                validation_required='same-window held-out event-aligned comparison',
                promotion_script='scripts/promote_supermimic_wm_candidate.py'))
            stage(args.output,'TRAINING_FINISHED',actor_updates=0,world_updates=trainer.world_updates,
                  training_stage=args.training_stage)
            return
        def run_block(block_name, names, budget, proposal_limit,cycle_index=0,*,managed_refresh=False):
            nonlocal obs,tick,cursor
            nonlocal global_attempt_id,global_accepted_id,global_rejected,global_backtracked
            sampling_history_start=len(replay.actor_start_bank.sampling_history)
            if budget<=0:return 0
            progress=schedule.block(cycle_index,block_name,budget,proposal_limit)
            if progress['status']=='done':return progress['accepted']
            if progress.get('proposal_in_flight',False):raise ValueError('Interrupted proposal: resume its pre_proposal checkpoint, not partial optimizer state')
            continuing=progress['status']!='new'
            trainer.set_trainable_names(names,block_name,reset_block=not continuing)
            if research:
                for group in trainer.actor_opt.param_groups:
                    group['lr']=trainer.actor_lr
            progress['status']='training'
            dynamic_names=('touchdown','post_touchdown','post_landing_support_loss','recovery_failure','stable_recovery') if block_name=='recovery' else ('approach_active',)
            if block_name=='recovery' and args.recovery_prefix_and_string_margin:
                dynamic_names=('pre_body_ground', *dynamic_names)
            dynamic_pool=[item for name in dynamic_names for item in replay.stage_starts.get(name,[]) if training_lane_available(item)]
            if block_name=='approach': dynamic_pool=[item for item in dynamic_pool if approach_active(item)]
            block_pool=(dynamic_pool + (recovery_pool if block_name=='recovery' else approach_pool))
            # Fixed, bounded multi-stage budget library for the entire block.
            # Actor starts refresh independently; do not move the budget test set per proposal.
            budget_rows=(recovery_pool if block_name=='recovery' else approach_pool)+critic_starts[:4]
            trainer.set_budget_state_library([as_state(row) for row in budget_rows])
            accepted=progress['accepted'];attempts=progress['attempts'];max_attempts=proposal_limit
            trainer.actor_updates=accepted;trainer.actor_attempts=attempts
            trainer.rejected_actor_updates=progress.get('rejected',attempts-accepted)
            trainer.backtracked_actor_updates=progress.get('backtracked',0)
            if managed_refresh and progress['pending_refresh']:raise ValueError('Internal refresh found in externally managed short block')
            if progress['pending_refresh']:
                schedule.finish_approach_refresh(progress,refresh_frozen_policy(block_name))
                save('latest.pt')
            while accepted<budget and attempts<max_attempts:
                check_stop()
                collect_block = not research or attempts == 0
                proposal_checkpoint=args.output/('pre_block.pt' if research else f'pre_proposal_{cycle_index}_{block_name}.pt')
                if collect_block:
                    save(proposal_checkpoint.name)
                pre_native_metrics=episode_stats.snapshot()
                attempts+=1;global_attempt_id+=1;trainer.block_attempts=attempts
                invocation_heads[block_name]['attempts']+=1
                progress['attempts']=attempts;progress['proposal_in_flight']=True
                trainer.actor_updates=accepted
                trainer.actor_attempts=attempts
                check_stop()
                if not args.landing_and_first_contact_objectives or collect_block:
                    cover_critic()
                ensure_critic()
                critic_loss_before_proposal=trainer.last_critic_loss
                if shutil.disk_usage(args.output).free<8*1024**3:raise RuntimeError('Disk guard: less than8GiB free')
                started=time.perf_counter()
                replay_tick_before_collect = replay._native_tick_counter
                with torch.inference_mode():
                    for _ in range(args.native_ticks_per_actor_update if collect_block else 0):
                        check_stop();action=native_actor_action(obs);obs,reward,done,info=wrapper.step(action);tick+=1
                        if not torch.isfinite(obs['policy']).all():raise FloatingPointError('Nonfinite native observation')
                        for raw in env.rally_records[cursor:]:
                            record=score_terminal_episode(raw,dict(protocol,_root=str(ROOT)))
                            record.update(policy_block=block_name,policy_update_at_end=accepted,
                                behavior_at_terminal=dict(policy_version=global_accepted_id,actor_sha256=behavior_actor_hash,native_tick=tick),
                                native_flight={key:raw[key] for key in ('outgoing_velocity_mps','first_landing_m',
                                    'contact_time_s','release_time_s','crossing_time_s','first_landing_time_s',
                                    'net_margin_m','shuttle_net_contact','crossed_net','landed_target') if key in raw},
                                scope='mixed-policy monitoring, not frozen score')
                            completed_returns.append(record['native_episode_return'])
                            completed_discounted.append(record['native_episode_discounted_return'])
                            episode_stats.add(record)
                            replay.actor_start_bank.label_terminal(record,replay.current_stream)
                            with (args.output/'native_episodes.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                            if writer is not None:
                                step=int(tick*128)
                                measured=episode_stats.snapshot()
                                metric_names={'qualified_airborne_string_contact_rate':'qualified_contact_rate','ball_side_complete_except_recovery_rate':'ball_side_rate','recovery_given_airborne_string_contact':'conditional_recovery_rate','complete5_rate':'complete5_rate'}
                                for key,name in metric_names.items():
                                    if measured[key] is not None:writer.add_scalar('native/'+name,measured[key],step)
                                writer.add_scalar('native/body_ground_evidence_count',episode_stats.body_observed,step)
                                if episode_stats.body_observed == episode_stats.episodes:
                                    writer.add_scalar('native/body_ground_rate',episode_stats.body_contacts/episode_stats.episodes,step)
                                writer.add_scalar('native/episode_return',float(record['native_episode_return']),step)
                                writer.add_scalar('native/return_ema',sum(completed_returns[-16:])/len(completed_returns[-16:]),step)
                        if writer is not None and cursor < len(env.rally_records):
                            # Publish one rolling observation after all simultaneous lane completions.
                            completed_window = completed_returns[-128:]
                            writer.add_scalar('native/return_mean_128', sum(completed_window)/len(completed_window), tick*128)
                            writer.add_scalar('native/return_mean_128_cases', len(completed_window), tick*128)
                            writer.flush()
                        cursor=len(env.rally_records)
                if research and collect_block:
                    # Publish the collected chunks to live native monitors before policy fitting.
                    archive.flush()
                collect_s=time.perf_counter()-started;check_stop()
                if collect_block:
                    progress['collection_reward_mean']=float(reward.mean())
                    if research and args.recovery_rollout_landing:
                        body_count = None
                        # All lanes start from the registered first episode. Only compare the
                        # uninterrupted initial stream; each first body penalty is irreversible.
                        n = replay._native_tick_counter - replay_tick_before_collect
                        if (tick - first_mixed_tick <= 800 and refresh_serial == first_refresh_serial
                                and n == args.native_ticks_per_actor_update and n <= replay.capacity):
                            indices = (torch.arange(n) + replay.write_pos - n) % replay.capacity
                            episode = replay.data['episode'][indices].reshape(n, replay.lanes)
                            body = replay.data['next_event_state'][indices, :, 5] > replay.data['event_state'][indices, :, 5]
                            body_count = count_first_episode_body_contacts(first_body_lanes, episode, body)
                        sample = dict(native_ticks=tick, accepted_id=global_accepted_id,
                                      completed_native_episodes=len(completed_returns), first_episode_body_cases=body_count)
                        with (args.output/'native_collection_history.jsonl').open('a') as stream:
                            stream.write(json.dumps(sample)+'\n')
                block_reward_mean=progress['collection_reward_mean']
                world_stats=run_world_updates(global_attempt_id,head=block_name) if collect_block else {}
                quality=check_head_quality(block_name,global_attempt_id)
                # Rebuild the start pool for every proposal.  Recovery starts
                # therefore follow fresh event-aligned native data as it
                # arrives; the verified archive remains a deterministic
                # fallback when a stage is temporarily under-populated.
                dynamic_pool=[item for name in dynamic_names for item in replay.stage_starts.get(name,[]) if training_lane_available(item)]
                if block_name=='approach': dynamic_pool=[item for item in dynamic_pool if approach_active(item)]
                pool=dynamic_pool + (recovery_pool if block_name=='recovery' else approach_pool)
                if block_name=='approach' and replay.new_starts:
                    fresh=[item for item in replay.new_starts if approach_active(item)]
                    if fresh: pool=fresh + approach_pool
                if not pool: pool=block_pool
                if block_name=='approach':
                    active_pool=[item for item in pool if approach_active(item)]
                    pool=active_pool if active_pool else approach_pool
                proposal_rows=None
                if block_name=='recovery':
                    eligible=[]
                    for item in pool:
                        lane_mask=training_start_mask(item,wm_holdout_lanes,recovery=True)
                        if lane_mask.any():
                            eligible.append((item,lane_mask))
                    if not eligible:
                        raise RuntimeError('No post-contact released training lanes in Recovery replay or archive starts')
                    original,lane_mask=eligible[(attempts-1)%len(eligible)]
                    row=dict(original,_actor_lane_mask=lane_mask)
                    start_sampling=dict(source='recovery_event_pool' if any(original is item for item in dynamic_pool)
                                        else 'verified_initial_archive',pool_size=len(pool),eligible_starts=len(eligible),
                                        effective_start_lanes=int(lane_mask.sum()))
                    if args.recovery_effective_start_lanes>1:
                        from quadsmash.rl.supermimic_recovery_batches import recovery_batch, late_recovery_batch
                        categories={name:{id(v) for v in replay.stage_starts.get(name,[])}
                                    for name in dynamic_names}
                        families={name:[] for name in (*categories,'initial_archive')}
                        for item,mask in eligible:
                            family=next((k for k,ids in categories.items() if id(item) in ids),'initial_archive')
                            families[family].append((item,mask))
                        batcher = late_recovery_batch if args.landing_and_first_contact_objectives else recovery_batch
                        draw = schedule.head_attempts('recovery') if continuous_recovery_starts else attempts
                        proposal_rows,layout=batcher(families,args.recovery_effective_start_lanes,draw)
                        start_sampling['recovery_draw_index'] = draw
                        row=proposal_rows[0]
                        if args.landing_and_first_contact_objectives:
                            count=args.recovery_effective_start_lanes
                            requested={'risk':count//2,'landing':count//4,'stable':count-count//2-count//4}
                            actual=dict(risk=0,landing=0,stable=0,initial_archive=0)
                            for record in layout:
                                stage=record['stage']
                                family=('landing' if stage in ('touchdown','post_touchdown') else
                                        'stable' if stage=='stable_recovery' else
                                        'initial_archive' if stage=='initial_archive' else 'risk')
                                actual[family]+=len(record['lane_episode_tick'])
                            start_sampling.update(requested_by_family=requested, actual_by_family=actual,
                                shortfall_by_family={k:max(0,n-actual[k]) for k,n in requested.items()})
                        start_sampling.update(source='pooled_recovery_episodes',records=layout,
                            requested_lanes=args.recovery_effective_start_lanes,
                            effective_start_lanes=sum(len(r['lane_episode_tick']) for r in layout))
                else:
                    row=pool[(attempts-1)%len(pool)]
                if block_name=='approach':
                    precheck=bank_precheck=None
                    if args.approach_objective=='toi_racket_only':
                        previews={}
                        bank_masks=replay.actor_start_bank.candidate_lanes_by_start() if research else {}
                        def precheck(start, *, bank_only=False):
                            candidates=bank_masks.get(id(start)) if bank_only else None
                            key=(id(start),candidates is not None)
                            if key not in previews:
                                previews[key]=toi_start_lanes(start,candidates)
                            return previews[key]
                        bank_precheck=partial(precheck,bank_only=True)
                    draw=schedule.head_attempts('approach') if args.continuous_approach_starts else attempts
                    sampled,start_sampling=replay.actor_start_bank.sample(draw,eligible_lanes=bank_precheck,
                        interleaved=interleaved_approach_starts)
                    start_sampling['approach_draw_index']=draw
                    if sampled is not None and training_lane_available(sampled):
                        row=sampled
                    else:
                        start_sampling['fallback']='verified_initial_archive_or_event_pool'
                        start_sampling['excluded_holdout_only_start']=sampled is not None
                        if precheck is not None:
                            offset=(draw-1)%len(pool)
                            for alternative in pool[offset:]+pool[:offset]:
                                mask=precheck(alternative)
                                if mask.any():
                                    row=dict(alternative,_actor_lane_mask=mask)
                                    start_sampling['effective_lanes']=int(mask.sum())
                                    break
                            else:
                                raise RuntimeError('No eligible training lane at its predicted TOI with live precontact gate and successful reference')
                    if args.toi_effective_start_lanes>1:
                        from quadsmash.rl.supermimic_actor_start_bank_v1 import pool_toi_starts
                        if start_sampling.get('fallback'):
                            alternatives=((alt,None) for alt in pool[offset:]+pool[:offset])
                        else:
                            alternatives=replay.actor_start_bank.same_category_starts(start_sampling['actual'],draw)
                        check_alternative=precheck if start_sampling.get('fallback') else bank_precheck
                        proposal_rows,layout=pool_toi_starts(row,alternatives,check_alternative,args.toi_effective_start_lanes,draw)
                        row=proposal_rows[0]
                        actual=sum(len(r['lane_episode_tick']) for r in layout)
                        start_sampling.update(requested_lanes=args.toi_effective_start_lanes,effective_lanes=actual,
                                              missing_episodes=args.toi_effective_start_lanes-actual,records=layout)
                state=training_state(row,compact=research);check_stop();trainer.actor_attempts=attempts-1
                actor_start=(tuple([state]+[training_state(r,compact=research) for r in proposal_rows[1:]]) if proposal_rows else state)
                if research:
                    retained=global_accepted_id-research_origin
                    trainer.contact_aux_weight=(args.research_contact_weight if block_name=='approach'
                        and args.research_contact_gradient=='G1' and retained>=32 else 0.)
                    adapter.action_ball_model.detach_residual_for_actor=True
                try:
                    stats=trainer.imagine_update(actor_start)
                finally:
                    if research:adapter.action_ball_model.detach_residual_for_actor=False
                if research:
                    stats['critic_loss_before_proposal']=critic_loss_before_proposal
                    stats['contact_aux_weight']=trainer.contact_aux_weight
                    stats['research_elapsed_s']=time.monotonic()-research_started
                stats['actor_start_sampling']=start_sampling
                stats['quality_protection']=quality
                post_native_metrics=episode_stats.snapshot()
                if stats['update_accepted']:
                    invocation_heads[block_name]['accepted']+=1
                    if not any(v>0.0 for v in stats['parameter_max_changes'].values()):
                        raise RuntimeError(f'{block_name} accepted update had zero trainable parameter change; no effective gradient')
                    accepted+=1
                    global_accepted_id+=1
                else:
                    global_rejected+=1
                backtracked=int(stats.get('update_accepted',False) and stats.get('backtrack_alpha',1.)<1.)
                global_backtracked += backtracked
                stats['pre_native_metrics'] = pre_native_metrics
                stats['post_native_metrics'] = post_native_metrics
                stats['rollback'] = rollback_audit(
                    phase=block_name,
                    update_index=accepted if stats['update_accepted'] else accepted+1,
                    proposal_checkpoint=proposal_checkpoint,
                    stats=stats,
                    pre_metrics=pre_native_metrics,
                    post_metrics=post_native_metrics,
                    replay_stream=replay.current_stream)
                progress.update(proposal_in_flight=False,attempts=attempts,accepted=accepted,rejected=progress.get('rejected',0)+int(not stats['update_accepted']),backtracked=progress.get('backtracked',0)+backtracked)
                with torch.no_grad():request=trainer._actor_action(state,sample=False)[0];preview,_,_=adapter.executor.forward(state.observation,request,state.clock_goal_context['joint_default'][0],state.clock_goal_context['limits'],state.executor_memory)
                record=dict(**stats,**{k:v for k,v in world_stats.items() if k not in stats},block=block_name,block_key=str(cycle_index),block_accepted_updates=accepted,block_rejected=progress['rejected'],block_backtracked=progress['backtracked'],attempt_id=global_attempt_id,accepted_id=global_accepted_id,global_attempts=global_attempt_id,global_accepted=global_accepted_id,global_rejected=global_rejected,global_backtracked=global_backtracked,native_ticks=tick,native_transitions=tick*128,native_collect_seconds=collect_s, native_reward_mean=block_reward_mean,completed_native_episodes=len(completed_returns),native_return_mean=sum(completed_returns)/len(completed_returns) if completed_returns else None,native_discounted_return_mean=sum(completed_discounted)/len(completed_discounted) if completed_discounted else None,raw_return_scope='completed mixed-policy monitoring; incomplete episodes excluded',peak_gpu_bytes=torch.cuda.max_memory_allocated(),model_calls=adapter.calls,window_sampling=world_stats.get('world_update_reports',[]),imagined_start_source='fresh_native' if replay.new_starts and any(pool_item is replay.new_starts[-1] for pool_item in pool) else 'event_aligned_replay')
                if block_name=='recovery':
                    stats['imagined_start_source']=start_sampling['source']
                if writer is not None:
                    step=int(tick*128)
                    writer.add_scalar('native/step_reward_mean',block_reward_mean,step)
                    writer.add_scalar('native/completed_episodes',len(completed_returns),step)
                    if block_name=='recovery' and args.recovery_prefix_and_string_margin:
                        prefix_lanes = sum(len(group['lane_episode_tick']) for group in start_sampling.get('records', [])
                                           if group['stage']=='pre_body_ground')
                        writer.add_scalar('actor/recovery_pre_body_ground_lanes', prefix_lanes, global_accepted_id)
                    writer.add_scalar('actor/imagined_return',float(stats.get('imagined_return_mean',0.0)),step)
                    if 'world_loss' in world_stats:writer.add_scalar('model/world_loss',float(world_stats['world_loss']),step)
                    writer.add_scalar('critic/loss',float(stats.get('critic_loss',0.0)),step)
                    writer.add_scalar('actor/%s_updates'%block_name,sum(row['accepted'] for key,row in schedule.cycles.items() if key.endswith('/'+block_name)),step)
                    writer.add_scalar('actor/%s_gate_active'%block_name,float(stats.get('effective_gate_lanes',0)),step)
                    writer.add_scalar('actor/effective_updates',float(global_accepted_id),step)
                    writer.add_scalar('actor/tracking_loss',float(stats.get('tracking_loss',0.0)),step)
                    for name in ('task_gradient_norm','tracking_gradient_norm','anchor_gradient_norm','task_tracking_cosine','final_command_max_change','objective_before','objective_after','tracking_active_lane_steps','tracking_effective_weight_sum','tracking_nonzero_gradient_lanes'):
                        if name in stats:writer.add_scalar('actor/'+name,float(stats[name]),step)
                    if research:
                        axes={'retained_updates':global_accepted_id-research_origin,'native_env_steps':int(tick*128),
                              'wall_seconds':int(stats['research_elapsed_s'])}
                        for axis,position in axes.items():
                            for metric in ('objective_before','objective_after','final_command_max_change',
                                           'tracking_loss','task_gradient_norm','tracking_gradient_norm','contact_aux_weight','contact_aux_to_toi_gradient_ratio'):
                                if metric in stats:writer.add_scalar(f'research/{axis}/{block_name}/{metric}',stats[metric],position)
                    writer.add_scalar('actor/accepted_updates',global_accepted_id,step)
                    writer.add_scalar('actor/rejected_updates',global_rejected,step)
                    writer.add_scalar('performance/native_collect_s',collect_s,step)
                    writer.add_scalar('performance/imagination_s',float(stats.get('bptt_seconds',0.0)),step)
                    writer.add_scalar('performance/critic_s',float(stats.get('critic_seconds',0.0)),step)
                    writer.add_scalar('performance/gpu_peak_memory',float(torch.cuda.max_memory_allocated()),step)
                    writer.flush()
                if not research:save('latest.pt')
                write(args.output/'progress.json',record)
                with (args.output/'training_history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                print('PHYSICAL_LEARNING='+json.dumps(record),flush=True)
                if stats['update_accepted'] and accepted and accepted%16==0:save(f'{cycle_index}_{block_name}_checkpoint_{accepted:04d}.pt')
                if not managed_refresh and block_name=='approach' and stats['update_accepted'] and accepted%args.approach_refresh_interval==0:
                    progress['pending_refresh']=True;save('latest.pt')
                    schedule.finish_approach_refresh(progress,refresh_frozen_policy('approach'))
                    save('latest.pt')
            write(args.output/(str(cycle_index)+'_'+block_name+'_actor_start_sampling.json'),replay.actor_start_bank.sampling_report(sampling_history_start))
            if accepted<budget:
                save(f'{cycle_index}_{block_name}_candidate_{accepted:04d}.pt')
                write(args.output/'block_incomplete.json',dict(block=block_name,target=budget,accepted=accepted,attempts=attempts,limit=max_attempts))
                raise RuntimeError(f'{block_name}: proposal budget exhausted ({accepted}/{budget} accepted); next block disabled')
            if not managed_refresh and block_name=='approach' and accepted>progress['refresh_at']:
                progress['pending_refresh']=True;save('latest.pt')
                schedule.finish_approach_refresh(progress,refresh_frozen_policy('approach'))
            if not research:save(f'{cycle_index}_{block_name}_candidate_{accepted:04d}.pt')
            progress['status']='done';save('latest.pt')
            if research and (global_accepted_id-research_origin)%32==0:
                save(f'research_{global_accepted_id-research_origin:04d}.pt')
            return accepted
        def refresh_frozen_policy(head,*,event_id=None,managed_refresh=False):
            """Bounded first-complete-episode collection; no mid-episode manual reset."""
            nonlocal obs,tick,cursor,refresh_collector,refresh_serial,frozen_policy_id,runtime_validation_data,runtime_validation_path,native_value_bank
            if args.refresh_ticks==0:return 0,0
            refresh_serial+=1
            consumer_heads=('recovery','approach') if managed_refresh else (head,)
            policy_id=f'{head}_frozen_v{global_accepted_id}_refresh{refresh_serial}'
            frozen_policy_id=policy_id
            refresh_dir=args.output/policy_id
            refresh_dir.mkdir(exist_ok=False)
            from quadsmash.rl.supermimic_physical_loop_loading_v1 import tensor_digest
            actor_hash=tensor_digest(actor.state_dict().items())
            fit_ticks=sorted(set(CRITIC_TICKS+APPROACH_TICKS+RECOVERY_TICKS))
            # Consecutive native inputs permit a one-step TD diagnostic. Only
            # the originally specified phases are eligible for calibration.
            capture_ticks=sorted(set(fit_ticks) | {t+1 for t in fit_ticks if t+1<800})
            refresh_collector=FrozenRefresh(128,800,timebase.gamma,policy_id,adapter.task_model.memory_schema,
                stage_ticks=capture_ticks,start_encoder=encode_native_value_start)
            replay.new_stream(source='online',policy_version=global_accepted_id)
            start_tick=tick
            collect_started=time.perf_counter()
            io_pipeline=getattr(archive,'pipeline',None)
            io_before=io_pipeline.timing() if io_pipeline is not None else {}
            record_cursor=len(env.rally_records)
            if archive.task_sequence is not None:
                archive.task_sequence.capture_reference_targets = trainer.reference_bank is not None
            try:
                with torch.inference_mode():
                    for _ in range(1601):
                        check_stop()
                        if _%archive_flush_rows==0:require_archive_space(args.output)
                        obs,reward,done,info=wrapper.step(native_actor_action(obs)); tick+=1
                        if not torch.isfinite(obs['policy']).all():raise FloatingPointError('Nonfinite frozen refresh')
                        if refresh_collector.complete.all():break
                rollout_seconds=time.perf_counter()-collect_started
                collector=refresh_collector
                rtg_refresh,receipt=collector.result()
            finally:
                if archive.task_sequence is not None:
                    archive.task_sequence.capture_reference_targets = False
                refresh_collector=None;frozen_policy_id=None
                primary=sys.exc_info()[1]
                try:archive.flush()
                except BaseException as cleanup_error:
                    if primary is None:raise
                    primary.add_note('Archive flush during exception cleanup also failed: '+repr(cleanup_error))
                    write(refresh_dir/'archive_flush_failure.json',dict(type=type(cleanup_error).__name__,message=str(cleanup_error),
                        original_type=type(primary).__name__,original_message=str(primary)))
            receipt['collection_timing']=dict(
                rollout_wall_seconds=rollout_seconds,through_archive_drain_seconds=time.perf_counter()-collect_started,
                native_ticks=tick-start_tick,env_steps_per_second=(tick-start_tick)*128/rollout_seconds,
                archive={k:v-io_before[k] for k,v in io_pipeline.timing().items()} if io_pipeline is not None else None)
            if actor_hash!=tensor_digest(actor.state_dict().items()):raise RuntimeError('Frozen collection changed actor')
            wanted={(lane,int(ep)) for lane,ep in enumerate(collector.ids.tolist())}
            native_records=[score_terminal_episode(r,dict(protocol,_root=str(ROOT))) for r in env.rally_records[record_cursor:]
                            if (int(r['env_id']),int(r['episode_index'])) in wanted]
            keys={(int(r['env_id']),int(r['episode_index'])) for r in native_records}
            if len(native_records)!=128 or keys!=wanted:raise RuntimeError('Frozen refresh terminal-score identities incomplete or duplicated')
            for record in native_records:
                record.update(policy_id=policy_id,actor_sha256=actor_hash,policy_execution=policy_execution,scope='frozen training refresh, not candidate promotion')
                replay.actor_start_bank.label_terminal(record,replay.current_stream)
            if paired_eval_snapshots:
                previous_stage=next(reversed(paired_eval_snapshots))
                stage_name=('post-Recovery' if head=='recovery' else
                            'post-Approach' if head=='approach' else 'post-refresh')
                paired_eval_previous_stage=dict(previous_stage=previous_stage,stage_name=stage_name,
                    actor_before=paired_eval_actor_snapshots[previous_stage],
                    actor_after={k:v.detach().cpu().clone() for k,v in actor.state_dict().items()},
                    action_before=paired_action_probe(paired_eval_actor_snapshots[previous_stage]),
                    action_after=paired_action_probe({k:v.detach().cpu().clone() for k,v in actor.state_dict().items()}))
            else:
                paired_eval_previous_stage={}
            receipt_paired=None
            stage_name=paired_eval_previous_stage.get('stage_name')
            if stage_name is not None:
                snapshot_id=f'{stage_name}_{refresh_serial:02d}'
                paired_eval_snapshots[snapshot_id]=native_records
                paired_eval_actor_snapshots[snapshot_id]=paired_eval_previous_stage['actor_after']
            write(refresh_dir/'episodes_summary.json',native_records)
            cursor=len(env.rally_records) # these records belong to the frozen stream, not mixed monitoring
            receipt.update(start_tick=start_tick,end_tick=tick,actor_sha256=actor_hash,policy_execution=policy_execution,head=head,consumer_heads=list(consumer_heads),schedule_event_id=event_id,formal_evaluation=False,
                           native_metrics=summarize_native_episodes(native_records))
            if receipt_paired is not None: receipt['paired_evaluation']=receipt_paired
            from quadsmash.rl.supermimic_evidence_binding_v1 import attach_terminal_ball_side
            packed=attach_terminal_ball_side(collector.packed_complete(),native_records)
            runtime_validation_data=packed
            atomic_torch_save(dict(tensors=packed,policy_id=policy_id,receipt=receipt),refresh_dir/'complete_episodes.pt')
            runtime_validation_path=str(refresh_dir/'complete_episodes.pt')
            native_contract_sha=hashlib.sha256(json.dumps(dict(task=manifest['task_contract'],execution=policy_execution),sort_keys=True).encode()).hexdigest()
            trend=frozen_return_monitor.observe(native_records,policy_id=policy_id,actor_sha256=actor_hash,contract_sha256=native_contract_sha)
            receipt['native_return_trend']=trend
            write(refresh_dir/'native_return_trend.json',trend)
            if writer is not None:
                writer.add_scalar('native_frozen/episode_return_mean',trend['mean'],tick*128)
                writer.add_scalar('native_frozen/decline_streak',trend['decline_streak'],tick*128)
                writer.add_scalar('native_frozen/plateau_streak',trend['flat_streak'],tick*128)
                writer.flush()
            if trend['stop_reason']:
                write(refresh_dir/'manifest.json',dict(receipt,complete_episodes=128,actor_unchanged=True,model_refresh_not_run=True))
                save('native_return_stopped.pt')
                write(args.output/'training_stopped.json',dict(reason=trend['stop_reason'],trend=trend,actor_updates=global_accepted_id,complete=False))
                raise RuntimeError('Stopped before further optimizer updates: '+trend['stop_reason'])
            wm_budgets=refresh_world_budgets(args.refresh_world_updates,consumer_heads)
            wm_fits={h:run_world_updates(global_attempt_id,n,head=h) for h,n in wm_budgets.items() if n}
            refresh_stats=dict(world_updates=sum(x['world_updates'] for x in wm_fits.values()))
            receipt['wm_refresh_by_head']=wm_fits
            receipt['shared_data_policy_id']=policy_id
            # Encode every eligible lane once, keeping only compact critic
            # inputs. Full asynchronous G07 snapshots are read one at a time.
            native_x=[];native_y=[];native_stage=[];native_identity=[];native_masks={h:[] for h in ('recovery','approach')}
            with torch.no_grad():
                for ti,row,mask in collector.calibration_rows():
                    ids=torch.where(mask)[0]
                    if 'critic_input' in row:native_x.append(row['critic_input'][mask])
                    else:
                        state=as_state(row)
                        native_x.append(adapter.critic_input(state)[mask.to(state.physical.device)].detach().cpu())
                    for h in native_masks:native_masks[h].append(row['head_masks'][h][mask].cpu())
                    native_y.append(rtg_refresh[ti,ids].float())
                    native_stage.extend([ti]*len(ids))
                    native_identity.extend([(int(i),int(collector.ids[i]),ti) for i in ids])
            if not native_x:raise RuntimeError('No complete frozen native critic labels')
            x=torch.cat(native_x);y=torch.cat(native_y);phase_ids=torch.tensor(native_stage)
            native_value_bank=dict(inputs=x,rtg=y,identity=native_identity,head_masks={h:torch.cat(v) for h,v in native_masks.items()},
                policy_id=policy_id,actor_sha256=actor_hash,policy_execution=policy_execution,source_sha256=file_sha(refresh_dir/'complete_episodes.pt'),accepted_updates=global_accepted_id,
                label_actor_state={k:v.detach().cpu().clone() for k,v in actor.state_dict().items()})
            train_mask=torch.isin(phase_ids,torch.tensor(fit_ticks))
            if wm_holdout_lanes:
                permitted,_=split_native_value_rows(native_value_bank,wm_holdout_lanes)
                train_mask &= permitted
            native_rows=torch.where(train_mask)[0]
            visited=torch.zeros(len(y),dtype=torch.bool)
            order=native_rows[torch.randperm(len(native_rows),generator=replay.generator)]
            fits=[]
            for update in range(args.refresh_critic_updates):
                start=(update*128)%len(order);ids=torch.cat((order,order))[start:start+min(128,len(order))]
                visited[ids]=True
                fit=trainer.calibrate_critic_tensors(x[ids].to('cuda:0'),y[ids].to('cuda:0'),epochs=1)
                fits.append(fit)
            receipt['critic_stage_coverage']={str(t):dict(eligible=int((phase_ids==t).sum()),used=int(((phase_ids==t)&visited).sum())) for t in sorted(set(native_stage))}
            atomic_torch_save(dict(native_value_bank,stage=phase_ids),refresh_dir/'native_critic_labels.pt')
            if wm_holdout_lanes:
                receipt['critic_holdout']={h:critic_holdout(trainer.critic,native_value_bank,h,wm_holdout_lanes,current_actor_sha256=actor_hash,current_execution=policy_execution,limits=dual_quality['critic_upper_bounds'][h] if dual_quality is not None else None) for h in ('recovery','approach')}
            if trainer.reference_bank is not None:
                from quadsmash.rl.supermimic_task_space_reference_v1 import TaskSpaceReferenceBankV1
                source_hash=hashlib.sha256((refresh_dir/'complete_episodes.pt').read_bytes()).hexdigest()
                with torch.no_grad():
                    device=next(adapter.parameters()).device
                    additions=TaskSpaceReferenceBankV1.from_packed({k:v.to(device) for k,v in packed.items()},
                        adapter.physical_model.geometry,source_sha256=source_hash,policy_id=policy_id,
                        match_distance_threshold=trainer.reference_bank.match_distance_threshold,terminal_records=native_records,
                        excluded_lanes=wm_holdout_lanes,use_substep_toi=args.use_substep_toi_references,
                        execution_contract_sha256=trainer.reference_bank.execution_contract_sha256)
                if research:
                    atomic_torch_save(additions.payload(),refresh_dir/'candidate_references.pt')
                else:
                    trainer.reference_bank.extend(additions)
                atomic_torch_save(trainer.reference_bank.payload(),refresh_dir/'reference_bank.pt')
                receipt['reference_bank']=trainer.reference_bank.manifest
            # Same post-refresh WM and same completed trajectories are checked for BOTH heads.
            # Missing calibrated bounds are explicitly unverified, never a success receipt.
            if dual_quality is not None:
                from quadsmash.rl.supermimic_wm_validation_v1 import runtime_head_check
                receipt['wm_quality']={h:runtime_head_check(adapter,packed,wm_holdout_lanes,launch,h,runtime_wm_bounds[h],
                    minimum_event_episodes=dual_quality['wm_min_event_episodes'][h]) for h in ('recovery','approach')}
            else:
                receipt['wm_quality']={h:dict(passed=None,reason='no calibrated independent validation bounds') for h in ('recovery','approach')}
            if stage_name is not None:
                # The paired report is written after the refresh has produced
                # its actual WM and Critic audit results.  Previously this was
                # a path string and omitted the value diagnostics entirely.
                critic_audit=paired_critic_audit()
                paired=paired_evaluation(
                    paired_eval_snapshots[paired_eval_previous_stage['previous_stage']],native_records,
                    stage=stage_name,
                    actor_before=paired_eval_previous_stage['actor_before'],
                    actor_after=paired_eval_previous_stage['actor_after'],
                    action_before=paired_eval_previous_stage['action_before'],
                    action_after=paired_eval_previous_stage['action_after'],
                    critic_td=critic_audit.get('td') if critic_audit else None,
                    wm_holdout=receipt.get('wm_quality'))
                paired['critic_native_rtg_fit']=receipt.get('critic_holdout') or critic_audit
                paired['comparison_from']=paired_eval_previous_stage['previous_stage']
                paired['policy_id']=policy_id
                paired['actor_sha256']=actor_hash
                write(paired_eval_dir/f'{stage_name}_{refresh_serial:02d}.json',paired)
                receipt_paired=paired
            receipt['reference_coverage']={h:reference_coverage(trainer.reference_bank,h,trainer.tracking_coefficient(h)) for h in ('recovery','approach')}
            if actor_hash!=tensor_digest(actor.state_dict().items()):raise RuntimeError('Refresh WM/critic fitting changed actor')
            receipt.update(complete_episodes=128,actor_unchanged=True,world_updates=refresh_stats['world_updates'],critic_updates=len(fits),critic_fits=fits,
                           warmup_ticks=0,partial_suffix_scope='WM only, excluded from calibration')
            write(refresh_dir/'manifest.json',receipt)
            if writer is not None:
                timing=receipt['collection_timing']
                for key in ('env_steps_per_second','through_archive_drain_seconds','rollout_wall_seconds'):
                    writer.add_scalar('native_frozen/'+key,timing[key],tick*128)
                for key,value in (timing['archive'] or {}).items():
                    writer.add_scalar('native_frozen/archive_'+key,value,tick*128)
                writer.add_scalar('native_frozen/episode_return_mean',sum(receipt['returns'])/128,tick*128)
                writer.flush()
            print('PHYSICAL_REFRESH='+json.dumps(receipt),flush=True)
            if dual_quality is not None and any(not receipt['wm_quality'][h]['passed'] or not receipt['critic_holdout'][h]['passed'] for h in ('recovery','approach')):
                raise RuntimeError('Combined frozen refresh quality failed; next short block disabled; see manifest')
            if managed_refresh:
                return dict(complete_episodes=128,actor_unchanged=True,consumer_heads=list(consumer_heads),policy_id=policy_id,
                    actor_sha256=actor_hash,manifest_path=str(refresh_dir/'manifest.json'),
                    world_updates=refresh_stats['world_updates'],critic_updates=len(fits),native_ticks=tick-start_tick)
            # Caller checkpoints the new optimizer state and completed refresh together.
            return refresh_stats['world_updates'],len(fits)
        if args.update_mode == 'joint':
            # One Adam update jointly covers both heads.  The old sequential blocks
            # remain available only as diagnostic code; this experiment must not
            # attribute the two heads to separate optimizers or acceptance steps.
            joint_pool = recovery_pool + approach_pool
            trainer.set_trainable_names(RECOVERY_NAMES + APPROACH_NAMES,'joint')
            trainer.set_budget_state_library([as_state(row) for row in joint_pool+critic_starts[:4]])
            accepted=0; attempts=0; max_attempts=max(8*args.joint_update_budget,args.joint_update_budget+8)
            while accepted < args.joint_update_budget:
                if attempts >= max_attempts: raise RuntimeError('joint block rejected too many actor proposals')
                attempts += 1; trainer.block_attempts=attempts; trainer.actor_updates=accepted; trainer.actor_attempts=attempts
                check_stop(); cover_critic(); ensure_critic()
                if shutil.disk_usage(args.output).free < 8*1024**3: raise RuntimeError('Disk guard: less than8GiB free')
                started=time.perf_counter()
                with torch.inference_mode():
                    for _ in range(args.native_ticks_per_actor_update):
                        check_stop(); action=native_actor_action(obs); obs,reward,done,info=wrapper.step(action); tick+=1
                        if not torch.isfinite(obs['policy']).all(): raise FloatingPointError('Nonfinite native observation')
                        for raw in env.rally_records[cursor:]:
                            record=score_terminal_episode(raw,dict(protocol,_root=str(ROOT))); record.update(policy_block='joint',policy_update_at_end=accepted,scope='mixed-policy monitoring, not frozen score'); completed_returns.append(record['native_episode_return']); completed_discounted.append(record['native_episode_discounted_return'])
                            with (args.output/'native_episodes.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                        cursor=len(env.rally_records)
                collect_s=time.perf_counter()-started; check_stop()
                world_stats=run_world_updates(global_attempt_id)
                row=joint_pool[accepted%len(joint_pool)]; state=as_state(row); check_stop(); stats=trainer.imagine_update(state)
                if stats['update_accepted']:
                    if not any(v>0.0 for v in stats['parameter_max_changes'].values()): raise RuntimeError('joint accepted update had zero trainable parameter change')
                    accepted += 1
                with torch.no_grad(): request=trainer._actor_action(state,sample=False)[0]; preview,_,_=adapter.executor.forward(state.observation,request,state.clock_goal_context['joint_default'][0],state.clock_goal_context['limits'],state.executor_memory)
                record=dict(**stats,**{k:v for k,v in world_stats.items() if k not in stats},block='joint',block_accepted_updates=accepted,native_ticks=tick,native_transitions=tick*128,native_collect_seconds=collect_s,native_reward_mean=float(reward.mean()),completed_native_episodes=len(completed_returns),native_return_mean=sum(completed_returns)/len(completed_returns) if completed_returns else None,native_discounted_return_mean=sum(completed_discounted)/len(completed_discounted) if completed_discounted else None,raw_return_scope='completed mixed-policy monitoring; incomplete episodes excluded',peak_gpu_bytes=torch.cuda.max_memory_allocated(),model_calls=adapter.calls,window_sampling=world_stats.get('world_update_reports',[]),imagined_start_source='initialization_archive')
                save('latest.pt'); write(args.output/'progress.json',record)
                with (args.output/'training_history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                print('PHYSICAL_LEARNING='+json.dumps(record),flush=True)
                if accepted and accepted%16==0: save(f'joint_checkpoint_{accepted:04d}.pt')
            archive.flush(); save(f'candidate_{args.joint_update_budget:04d}.pt')
            write(args.output/'training_completed.json',dict(actor_updates=args.joint_update_budget,native_ticks=tick,
                 actor_attempts=trainer.actor_attempts,rejected_actor_updates=trainer.rejected_actor_updates,
                 world_updates=trainer.world_updates,critic_updates=trainer.critic_updates,formal_evaluation=False,blocks=['joint'],trainable_names=sorted(trainer.trainable_names)))
            stage(args.output,'TRAINING_FINISHED',actor_updates=args.joint_update_budget)
            return
        elif alternating is not None:
            cycle_results=[]
            def train_short(event):
                head=event['head'];names=RECOVERY_NAMES if head=='recovery' else APPROACH_NAMES
                run_block(head,names,event['target'],event['proposal_limit'],event['block_key'],managed_refresh=True)
                row=schedule.block(event['block_key'],head,event['target'],event['proposal_limit'])
                return {k:row[k] for k in ('attempts','accepted','rejected','backtracked')}
            def refresh_pair(event):
                refresh_mode=getattr(args,'refresh_mode','pair')
                refresh_head=event.get('after_head','combined') if refresh_mode=='per-head' else 'combined'
                return refresh_frozen_policy(refresh_head,event_id=event['id'],managed_refresh=True)
            def finalize_cycle(event):
                totals=alternating.head_totals(event['cycle'])
                total=sum(x['accepted'] for x in totals.values())
                refreshes=[r['result'] for r in alternating.results if r['event']['cycle']==event['cycle'] and r['event']['kind']=='refresh']
                result=dict(cycle=event['cycle']+1,recovery_accepted=totals['recovery']['accepted'],approach_accepted=totals['approach']['accepted'],
                    head_counters=totals,native_ticks=tick,world_updates=trainer.world_updates,critic_updates=trainer.critic_updates,
                    refresh_count=len(refreshes),refresh_world_updates=sum(r['world_updates'] for r in refreshes),
                    refresh_critic_updates=sum(r['critic_updates'] for r in refreshes),formal_evaluation=False)
                save('cycle_%02d_candidate_%04d.pt' % (event['cycle']+1,total))
                write(args.output/('cycle_%02d.json' % (event['cycle']+1)),result)
                return result
            run_schedule(alternating,train=train_short,refresh=refresh_pair,finalize=finalize_cycle,checkpoint=save)
            cycle_results=[r['result'] for r in alternating.results if r['event']['kind']=='finalize']
            totals=alternating.head_totals();total=sum(x['accepted'] for x in totals.values())
            write(args.output/'training_completed.json',dict(update_mode='sequential',invocation=invocation_receipt(),sequential_layout='short-alternating',
                refresh_mode=args.refresh_mode,head_block_size=args.head_block_size,cycles=args.cycles,actor_updates=total,
                recovery_accepted=totals['recovery']['accepted'],approach_accepted=totals['approach']['accepted'],head_counters=totals,
                proposal_attempts=global_attempt_id,global_accepted=global_accepted_id,global_rejected=global_rejected,
                global_backtracked=global_backtracked,native_ticks=tick,world_updates=trainer.world_updates,
                critic_updates=trainer.critic_updates,cycle_results=cycle_results,formal_evaluation=False))
            stage(args.output,'TRAINING_FINISHED',actor_updates=total)
        else:
            # Legacy diagnostic path: recovery then approach. Each block owns
            # its trainable head and Adam state; the intermediate checkpoint is
            # written before the approach block starts.
            cycle_results=[]
            for cycle in range(args.cycles):
                if args.resume_start_block=='approach' and cycle==0:
                    recovery_accepted=0
                    refresh_world_updates,refresh_critic_updates=0,0
                else:
                    recovery_was_done=schedule.block(cycle,'recovery',args.recovery_update_budget,args.recovery_proposal_budget)['status']=='done'
                    recovery_accepted=run_block('recovery', RECOVERY_NAMES, args.recovery_update_budget, args.recovery_proposal_budget,cycle)
                    if not recovery_was_done:save('cycle_%02d_recovery_candidate_%04d.pt' % (cycle+1,recovery_accepted))
                    rp=schedule.block(cycle,'recovery',args.recovery_update_budget,args.recovery_proposal_budget)
                    if not rp.get('frozen_refresh_done',False):
                        refresh_world_updates,refresh_critic_updates=refresh_frozen_policy('recovery')
                        rp.update(frozen_refresh_done=True,refresh_world_updates=refresh_world_updates,refresh_critic_updates=refresh_critic_updates);save('latest.pt')
                    else:
                        refresh_world_updates=rp.get('refresh_world_updates',0);refresh_critic_updates=rp.get('refresh_critic_updates',0)
                # Do not mislabel an already-updated Approach actor as a Recovery checkpoint.
                if trainer.block_name=='recovery':save('cycle_%02d_recovery_refresh_candidate_%04d.pt' % (cycle+1,recovery_accepted))
                approach_accepted=run_block('approach', APPROACH_NAMES, args.approach_update_budget, args.approach_proposal_budget,cycle)
                total_accepted=recovery_accepted+approach_accepted
                archive.flush(); save('cycle_%02d_candidate_%04d.pt' % (cycle+1,total_accepted))
                cycle_results.append(dict(cycle=cycle+1,recovery_accepted=recovery_accepted,approach_accepted=approach_accepted,
                    native_ticks=tick,world_updates=trainer.world_updates,critic_updates=trainer.critic_updates,
                    **schedule.cycle_refresh_counts(cycle,refresh_world_updates,refresh_critic_updates)))
                write(args.output/('cycle_%02d.json' % (cycle+1)),cycle_results[-1])
            total_accepted=sum(x['recovery_accepted']+x['approach_accepted'] for x in cycle_results)
            write(args.output/'training_completed.json', dict(
                update_mode='sequential', invocation=invocation_receipt(), cycles=args.cycles, actor_updates=total_accepted,
                recovery_accepted=sum(x['recovery_accepted'] for x in cycle_results),
                approach_accepted=sum(x['approach_accepted'] for x in cycle_results),
                proposal_attempts=global_attempt_id, global_accepted=global_accepted_id,
                global_rejected=global_rejected, global_backtracked=global_backtracked,
                native_ticks=tick, actor_attempts=trainer.actor_attempts,
                rejected_actor_updates=trainer.rejected_actor_updates,
                world_updates=trainer.world_updates, critic_updates=trainer.critic_updates,
                cycle_results=cycle_results, formal_evaluation=False, blocks=['recovery','approach'],
                trainable_names={'recovery': sorted(RECOVERY_NAMES), 'approach': sorted(APPROACH_NAMES)}))
            stage(args.output, 'TRAINING_FINISHED', actor_updates=total_accepted)
        return
        # Legacy single-block loop retained below for provenance but unreachable.
        max_attempts=max(args.actor_update_budget*8,args.actor_update_budget+8)
        while trainer.actor_updates<args.actor_update_budget:
            if trainer.actor_attempts>=max_attempts:
                raise RuntimeError('Execution-space budget rejected too many actor updates')
            check_stop();cover_critic();ensure_critic()
            if shutil.disk_usage(args.output).free<8*1024**3:raise RuntimeError('Disk guard: less than8GiB free')
            started=time.perf_counter()
            with torch.inference_mode():
                for _ in range(4):
                    check_stop()
                    action=native_actor_action(obs)
                    obs,reward,done,info=wrapper.step(action);tick+=1
                    if not torch.isfinite(obs['policy']).all():raise FloatingPointError('Nonfinite native observation')
                    for raw in env.rally_records[cursor:]:
                        record=score_terminal_episode(raw,dict(protocol,_root=str(ROOT)))
                        record.update(policy_update_at_end=trainer.actor_updates,scope='mixed-policy monitoring, not frozen score')
                        completed_returns.append(record['native_episode_return'])
                        completed_discounted.append(record['native_episode_discounted_return'])
                        with (args.output/'native_episodes.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                    cursor=len(env.rally_records)
            collect_s=time.perf_counter()-started
            check_stop()
            window,lanes,sampling=replay.window(timebase.world_steps,128,trainer.actor_updates)
            world_stats=trainer.fit_world_window(window,launch[lanes],separate_supervision=args.wm_separate_supervision,long_native=sample_long_flight());del window
            verify_online_forward(trainer.block_name,trainer.actor_attempts+1)
            pool=replay.new_starts if replay.new_starts and trainer.actor_updates%2 else starts
            row=pool[trainer.actor_updates%len(pool)];state=as_state(row)
            check_stop()
            stats=trainer.imagine_update(state)
            if stats['update_accepted'] and not any(v>0 for v in stats['parameter_max_changes'].values()):
                raise RuntimeError('Accepted actor optimizer step made no approach parameter change')
            with torch.no_grad():
                request=trainer._actor_action(state,sample=False)[0]
                preview,_,_=adapter.executor.forward(state.observation,request,state.clock_goal_context['joint_default'][0],
                    state.clock_goal_context['limits'],state.executor_memory)
            record=dict(**stats,**{k:v for k,v in world_stats.items() if k not in stats},native_ticks=tick,
                native_transitions=tick*128,native_collect_seconds=collect_s,native_reward_mean=float(reward.mean()),
                completed_native_episodes=len(completed_returns),
                native_return_mean=sum(completed_returns)/len(completed_returns) if completed_returns else None,
                native_discounted_return_mean=sum(completed_discounted)/len(completed_discounted) if completed_discounted else None,
                raw_return_scope='completed mixed-policy monitoring; incomplete episodes excluded',
                peak_gpu_bytes=torch.cuda.max_memory_allocated(),model_calls=adapter.calls,window_sampling=sampling,
                imagined_start_source='fresh_native' if pool is replay.new_starts else 'initialization_archive')
            save('latest.pt')
            write(args.output/'progress.json',record)
            with (args.output/'training_history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print('PHYSICAL_LEARNING='+json.dumps(record),flush=True)
            if trainer.actor_updates and trainer.actor_updates%32==0:save(f'checkpoint_{trainer.actor_updates:04d}.pt')
        archive.flush();save(f'candidate_{args.actor_update_budget:04d}.pt')
        write(args.output/'training_completed.json',dict(actor_updates=trainer.actor_updates,native_ticks=tick,
             actor_attempts=trainer.actor_attempts,rejected_actor_updates=trainer.rejected_actor_updates,
             world_updates=trainer.world_updates,critic_updates=trainer.critic_updates,formal_evaluation=False))
        stage(args.output,'TRAINING_FINISHED',actor_updates=trainer.actor_updates)
    except BaseException as error:
        failed=True;(args.output/'failure_traceback.txt').write_text(traceback.format_exc())
        write(args.output/'failure.json',dict(phase=phase,native_ticks=tick,
            actor_updates=trainer.actor_updates if trainer else 0,type=type(error).__name__,message=str(error)))
        if trainer is not None and 'save' in locals():
            try:
                save('interrupted.pt')
            except BaseException as cleanup_error:
                write(args.output/'cleanup_failure.json',dict(
                    type=type(cleanup_error).__name__,message=str(cleanup_error),
                    original_type=type(error).__name__,original_message=str(error)))
        raise
    finally:
        if writer is not None:
            writer.flush(); writer.close()
        if archive is not None:
            try:
                try:
                    archive.flush()
                finally:
                    if getattr(archive,'pipeline',None) is not None:
                        archive.pipeline.close()
            except BaseException as cleanup_error:
                if failed:
                    write(args.output/'archive_cleanup_failure.json',dict(
                        type=type(cleanup_error).__name__,message=str(cleanup_error)))
                else:
                    raise
            if archive.task_sequence is not None:archive.task_sequence.close()
        close_resource(args.output,'ENV',env,primary_failed=failed)


if __name__=='__main__':
    # Isaac Sim 5.1's PhysX Python bindings can run a module-level destructor
    # after the scoped native cleanup has released Kit.  That destructor calls
    # PyThreadState_Get without the GIL and aborts during interpreter exit.
    # Training artifacts are already durable when main() returns, so bypass
    # Python's second finalization pass on the clean path.
    os.environ['QUADSMASH_SKIP_KIT_CLOSE'] = '1'
    try:
        main()
    except BaseException as exc:
        # Keep pre-Kit failures visible.  The previous hard exit suppressed
        # argparse/launcher diagnostics and made a failed calibration look
        # like a silent process disappearance.
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
        os._exit(1)
    else:
        os._exit(0)
