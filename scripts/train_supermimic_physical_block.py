#!/usr/bin/env python3
"""Long-running sequential Recovery→Approach SuperMimic training block.

Native Isaac interaction, replay, world-model imagination and actor/critic
updates are durable and restartable. No MPC or teacher intervention.
"""
import argparse,hashlib,json,os,shutil,subprocess,sys,time,traceback,signal,io
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


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


def main():
    from isaaclab.app import AppLauncher
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('archive','packed-data','physical-checkpoint','ball-checkpoint','task-checkpoint','safety-mapping','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--resume-training',type=Path)
    parser.add_argument('--resume-start-block',choices=('recovery','approach'),default='recovery',
                        help='Resume a completed Recovery checkpoint directly at Approach')
    parser.add_argument('--resume-warmup-ticks',type=int,default=0,
                        help='Optional native ticks before a resumed stream; zero preserves exact block budgets')
    parser.add_argument('--actor-update-budget',type=int,default=None,help='Legacy single-block budget')
    parser.add_argument('--recovery-update-budget',type=int,default=16,
                        help='Retained for config compatibility; joint mode uses --joint-update-budget')
    parser.add_argument('--approach-update-budget',type=int,default=16,
                        help='Retained for config compatibility; joint mode uses --joint-update-budget')
    parser.add_argument('--joint-update-budget',type=int,default=16)
    parser.add_argument('--update-mode',choices=('sequential','joint'),default='sequential')
    parser.add_argument('--ball-side-mode',choices=('shadow','enforce'),default='shadow')
    parser.add_argument('--recovery-proposal-budget',type=int,default=4)
    parser.add_argument('--approach-proposal-budget',type=int,default=4)
    parser.add_argument('--refresh-ticks',type=int,default=800)
    parser.add_argument('--cycles',type=int,default=1)
    parser.add_argument('--native-ticks-per-actor-update',type=int,default=16)
    parser.add_argument('--replay-capacity-ticks',type=int,default=16384)
    parser.add_argument('--tensorboard-dir',type=Path)
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
    parser.add_argument('--wm-source-quota',default='recent_online=64,older_online=32,initial_archive=32')
    parser.add_argument('--wm-stage-quota',default='uniform=32,pre_touchdown=24,touchdown=24,post_touchdown=24,recovery_failure=16,stable_recovery=8')
    parser.add_argument('--task-reference-bank',type=Path,default=None,
                        help='Optional success-only task-space reference bank')
    parser.add_argument('--tracking-weight',type=float,default=0.0)
    AppLauncher.add_app_launcher_args(parser);args=parser.parse_args()
    if args.actor_update_budget is not None and args.actor_update_budget < 1: parser.error('actor-update-budget must be positive')
    if args.recovery_update_budget < 0 or args.approach_update_budget < 0: parser.error('block budgets must be non-negative')
    if args.recovery_proposal_budget < 1 or args.approach_proposal_budget < 1: parser.error('proposal budgets must be positive')
    if args.cycles < 1 or args.native_ticks_per_actor_update < 1 or args.replay_capacity_ticks < 64: parser.error('long-run budgets invalid')
    if args.refresh_ticks < 0 or args.refresh_world_updates < 0 or args.refresh_critic_updates < 0 or args.resume_warmup_ticks < 0: parser.error('refresh/resume budgets invalid')
    if args.world_updates_per_actor_update < 1: parser.error('world-updates-per-actor-update must be positive')
    if args.ball_side_mode == 'enforce': parser.error('ball-side enforce mode is not implemented; use shadow for this audit-only revision')
    if args.actor_update_budget is not None:
        args.recovery_update_budget=0;args.approach_update_budget=args.actor_update_budget
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    try:
        args.wm_source_quota=parse_quota(args.wm_source_quota,allowed={'recent_online','older_online','initial_archive'},name='wm-source-quota')
        args.wm_stage_quota=parse_quota(args.wm_stage_quota,allowed={'uniform','pre_touchdown','touchdown','post_touchdown','recovery_failure','stable_recovery'},name='wm-stage-quota')
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
        dirty_diff=subprocess.check_output(['git','diff'],text=True),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    from quadsmash.rl.supermimic_run_lifecycle import execute_scoped_native,stage
    stage(args.output,'BEFORE_APP');app=AppLauncher(args,fast_shutdown=False).app
    stage(args.output,'AFTER_APP')
    args.stop_request={'signal':None}
    def request_stop(signum,frame):args.stop_request['signal']=signal.Signals(signum).name
    # Override Kit's handler: it shuts down plugins before Python can save.
    signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
    execute_scoped_native(args.output,lambda:run(args),app)


def run(args):
    from quadsmash.rl.supermimic_run_lifecycle import stage as lifecycle_stage
    lifecycle_stage(args.output, 'RUN_ENTER')
    import torch
    lifecycle_stage(args.output, 'IMPORT_TORCH')
    from torch.utils.tensorboard import SummaryWriter
    lifecycle_stage(args.output, 'IMPORT_TENSORBOARD')
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    lifecycle_stage(args.output, 'IMPORT_RSL')
    from quadsmash.rl.supermimic_physical_loop_loading_v1 import load_physical_loop,APPROACH_NAMES,RECOVERY_NAMES
    lifecycle_stage(args.output, 'IMPORT_LOADING')
    from quadsmash.rl.supermimic_physical_learner_v1 import PhysicalHorizonTrainer
    lifecycle_stage(args.output, 'IMPORT_LEARNER')
    from quadsmash.rl.supermimic_physical_loop_replay_v1 import PhysicalLoopReplay,current_start
    from quadsmash.rl.supermimic_physical_imagination_state_v1 import PhysicalImaginationStateV1
    from quadsmash.rl.supermimic_physical_capture_v1 import PhysicalTransitionArchive,tensor_tree_copy
    from quadsmash.rl.supermimic_physical_run_guard_v1 import critic_ready,atomic_torch_save
    from quadsmash.rl.supermimic_timebase_v2 import Timebase
    from quadsmash.rl.supermimic_task_contract import resolve_task_contract,make_task_cfg,zero_panel_reset_noise
    from quadsmash.rl.supermimic_terminal_scoring import score_terminal_episode
    from quadsmash.rl.supermimic_run_lifecycle import close_resource,stage
    from quadsmash.tasks.stand.supermimic_env_v1 import SuperMimicEnvV1
    from quadsmash.tasks.stand.full_racket_eval_env_v4 import FrozenPanelEnv
    lifecycle_stage(args.output, 'IMPORT_ALL')
    env=trainer=archive=None;writer=None;failed=False;tick=0;phase='LOAD'
    torch.set_num_threads(2);torch.manual_seed(20261114)
    try:
        lifecycle_stage(args.output, 'BEFORE_LOAD_PHYSICAL_LOOP')
        actor,adapter,template,launch,manifest=load_physical_loop(archive=args.archive,
            physical_checkpoint=args.physical_checkpoint,ball_checkpoint=args.ball_checkpoint,
            task_checkpoint=args.task_checkpoint,safety_mapping=args.safety_mapping)
        lifecycle_stage(args.output, 'AFTER_LOAD_PHYSICAL_LOOP')
        write(args.output/'model_manifest.json',adapter.manifest)
        packed_sha=hashlib.sha256(args.packed_data.read_bytes()).hexdigest()
        expected=json.loads((args.packed_data.parent/'completed.json').read_text())
        if packed_sha!=expected['sha256'] or expected['first_completed']!=128:raise ValueError('Invalid initialization data')
        package=torch.load(args.packed_data,map_location='cpu',weights_only=True)
        lifecycle_stage(args.output, 'LOADED_PACKED_DATA')
        replay=PhysicalLoopReplay(package,adapter.task_model.memory_schema,capacity_ticks=args.replay_capacity_ticks)
        lifecycle_stage(args.output, 'REPLAY_READY')
        reference_bank=None
        if args.task_reference_bank is not None:
            from quadsmash.rl.supermimic_task_space_reference_v1 import TaskSpaceReferenceBankV1
            bank_payload=torch.load(args.task_reference_bank,map_location='cpu',weights_only=False)
            if isinstance(bank_payload,TaskSpaceReferenceBankV1):
                reference_bank=bank_payload
            elif isinstance(bank_payload,dict) and bank_payload.get('schema')=='success-guided-task-space-reference-v1':
                reference_bank=TaskSpaceReferenceBankV1(**{key:bank_payload[key] for key in (
                    'approach_sequences','recovery_sequences','approach_keys','recovery_keys',
                    'elapsed','geometry_sha256','source_sha256')})
            else:
                raise ValueError('Task reference bank schema mismatch')
            if reference_bank.geometry_sha256 and reference_bank.geometry_sha256 != adapter.physical_model.geometry.model_sha256:
                raise ValueError('Task reference bank geometry hash mismatch')
            reference_bank=reference_bank.to(torch.device('cuda:0'))
        lifecycle_stage(args.output, 'REFERENCE_BANK_READY', enabled=reference_bank is not None)
        writer=SummaryWriter(str(args.tensorboard_dir or (args.output/'tensorboard')),flush_secs=30)
        from quadsmash.rl.supermimic_physical_start_cache_v1 import load_start_cache,APPROACH_TICKS,RECOVERY_TICKS,CRITIC_TICKS
        start_cache=load_start_cache(args.archive,manifest)
        lifecycle_stage(args.output, 'START_CACHE_LOADED')
        starts=[start_cache[t] for t in APPROACH_TICKS]
        critic_starts=[start_cache[t] for t in CRITIC_TICKS]
        lifecycle_stage(args.output, 'STARTS_SELECTED')
        def as_state(row):return PhysicalImaginationStateV1.from_record(row,executor_template=template,launch=launch,device='cuda:0')
        timebase=Timebase(**json.loads((ROOT/'configs/supermimic_task_horizon_v2.json').read_text()))
        trainer=PhysicalHorizonTrainer(actor,adapter,as_state(starts[1]),timebase,
            command_delta_budget=args.command_delta_budget,
            target_delta_budget_rad=args.target_delta_budget_rad,
            native_reference_command_budget=args.native_reference_command_budget,
            native_reference_target_budget_rad=args.native_reference_target_budget_rad,
            trainable_names=RECOVERY_NAMES + APPROACH_NAMES,block_name='joint',
            reference_bank=reference_bank,tracking_weight=args.tracking_weight)
        lifecycle_stage(args.output, 'TRAINER_READY')
        trainer.set_budget_state_library([as_state(row) for row in starts+critic_starts[:4]])
        def approach_active(row):
            """Require a meaningful activation of the actual exported Approach gate."""
            observation=row['observation']
            if isinstance(observation,dict):
                observation=observation['policy']
            if observation.ndim != 2 or observation.shape[-1] != 794:
                raise ValueError('Approach start observation is not the 794D public contract')
            gate=actor.approach_gate(observation)
            if gate.ndim == 2: gate=gate[:,0]
            # A tiny positive gate is numerically present but does not provide
            # a useful actor gradient. Keep at least four genuinely active lanes.
            return bool((gate > 0.25).sum() >= 4)

        approach_pool=[start_cache[t] for t in APPROACH_TICKS if approach_active(start_cache[t])]
        if not approach_pool:
            raise RuntimeError('No approach-gated initialization starts available')
        recovery_pool=[start_cache[t] for t in RECOVERY_TICKS]
        # Same-policy native RTG calibration labels from the frozen 800-tick archive.
        # This is separate from the 0.64 s imagined objective and never mixed with
        # candidate-policy monitoring.
        native_reward=package['tensors']['reward'].double()
        native_done=package['tensors']['terminated']|package['tensors']['timeout']
        rtg=torch.zeros_like(native_reward)
        running=torch.zeros(native_reward.shape[1],dtype=torch.float64)
        for ti in range(native_reward.shape[0]-1,-1,-1):
            running=native_reward[ti].double()+timebase.gamma*running*(~native_done[ti]).double()
            rtg[ti]=running
        del native_reward,native_done,running
        replay_archives=[];critic_coverage_blocks=0
        refresh_world_updates=0;refresh_critic_updates=0
        if args.resume_training:
            resumed=torch.load(args.resume_training,map_location='cuda:0',weights_only=False)
            if resumed['adapter_schema']!=adapter.schema or resumed['metadata']['physical_imagination']!=adapter.manifest:
                raise ValueError('Resume physical model/execution source mismatch')
            actor.load_state_dict(resumed['actor'],strict=True);adapter.load_state_dict(resumed['world'],strict=True)
            trainer.critic.load_state_dict(resumed['critic'],strict=True)
            if 'actor_optimizers' in resumed:
                trainer._pending_actor_optimizer_states=dict(resumed['actor_optimizers'])
                if 'joint' in trainer._pending_actor_optimizer_states:
                    trainer._actor_optimizers['joint'].load_state_dict(trainer._pending_actor_optimizer_states.pop('joint'))
            else:
                trainer.actor_opt.load_state_dict(resumed['actor_optimizer'])
            trainer.world_opt.load_state_dict(resumed['world_optimizer'])
            trainer.critic_opt.load_state_dict(resumed['critic_optimizer'])
            for name in ('actor_updates','world_updates','critic_updates','last_critic_loss'):setattr(trainer,name,resumed[name])
            tick=int(resumed['native_ticks'])
            resumed_global=dict(resumed.get('global_counters',{}))
            critic_coverage_blocks=int(resumed.get('critic_coverage_blocks',0))
            if trainer.frozen_hash()!=trainer.frozen_actor_hash:raise ValueError('Resume changed frozen actor parameters')
            replay_archives=resumed['replay_archives']
            # A direct Approach continuation has a valid fixed initialization
            # ring and fixed approach-gated starts.  Re-decoding every prior
            # native archive row is both redundant and very expensive; it also
            # recreates large executor trees that are not needed for this
            # block.  Keep the archive manifest for provenance, but start the
            # live ring from the verified packed initialization data.
            restore_archives = not (args.resume_start_block == 'approach')
            if not restore_archives:
                stage(args.output, 'RESUME_ARCHIVES_SKIPPED', reason='direct_approach_uses_packed_ring')
            for directory in (replay_archives if restore_archives else []):
                directory=Path(directory);receipt=json.loads((directory/'manifest.json').read_text())
                for item in receipt['chunks']:
                    path=directory/item['file']
                    if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Resume archive hash mismatch')
                    if path.suffix == '.zst':
                        proc=subprocess.run(['zstd','-q','-d','-c',str(path)],check=True,capture_output=True)
                        chunk=torch.load(io.BytesIO(proc.stdout),map_location='cpu',weights_only=True)
                    else:
                        chunk=torch.load(path,map_location='cpu',weights_only=True)
                    # Rebuild only the compact chronological ring on resume.
                    # Historical full executor/observation trees are not
                    # needed for Approach proposals and otherwise dominate
                    # memory (many archived rows × full 128-lane state).
                    for row in chunk['rows']:replay.append(row,retain_start=False)
                replay.new_stream(source='online', policy_version=int(resumed_global.get('accepted',0)))
            replay.generator.set_state(resumed['replay_rng'].cpu())
            replay_state=resumed.get('replay_state',{})
            replay.current_source=str(replay_state.get('current_source','online'))
            replay.current_policy_version=int(replay_state.get('current_policy_version',resumed_global.get('accepted',0)))
            replay._native_tick_counter=int(replay_state.get('native_tick_counter',replay._native_tick_counter))
            saved_meta=resumed.get('replay_metadata')
            if saved_meta is not None:
                for name in ('streams','stage_ids','source_ids','policy_versions','native_ticks'):
                    value=saved_meta.get(name)
                    target=getattr(replay,name,None)
                    if value is not None and target is not None and tuple(value.shape)==tuple(target.shape):
                        target.copy_(value)
            torch.set_rng_state(resumed['torch_rng'].cpu());torch.cuda.set_rng_state(resumed['cuda_rng'].cpu())
        source_path=Path(manifest['source_checkpoint']);source=torch.load(source_path,map_location='cpu',weights_only=False)
        execution_root=Path('/home/autodrive/LZQW/unitree_smash')
        contract,protocol,cases,receipt=resolve_task_contract(source,source_path,execution_root=execution_root)
        if contract!=manifest['task_contract']:raise ValueError('Native task contract changed')
        cfg=make_task_cfg(128,contract,protocol,receipt);cfg.seed=882072;cfg.sim.device='cuda:0'
        class TrainingEnv(SuperMimicEnvV1,FrozenPanelEnv):pass
        phase='ENV';stage(args.output,'BEFORE_ENV')
        env=TrainingEnv(cfg,cases=cases,reset_noise=zero_panel_reset_noise(128,882072),
            residual_policy_checkpoint=source,g07_execution_root=execution_root,
            native_return_gamma=timebase.gamma,contact_flight_recovery=contract['contact_flight_recovery'])
        wrapper=RslRlVecEnvWrapper(env,clip_actions=None);obs,_=wrapper.reset()
        if args.resume_training:
            # Environment seeding must not overwrite the resumed learner RNG.
            torch.set_rng_state(resumed['torch_rng'].cpu());torch.cuda.set_rng_state(resumed['cuda_rng'].cpu())
        stage(args.output,'AFTER_RESET')
        def native_actor_action(observation):
            # Phase4 only means foot contact.  Recovery corrections are allowed
            # only after the native rally has both contacted and released the
            # shuttle; this latch is per lane and resets with the rally.
            gate=(env.rally.had_contact & env.rally.released)
            if hasattr(actor,'forward_with_recovery_gate'):
                return actor.forward_with_recovery_gate(observation['policy'],gate,sample=False)[0]
            return actor(observation['policy'],sample=False)[0]
        refresh_records=[]
        refresh_rewards=[]; refresh_dones=[]; refresh_stage_records={}; refresh_count=0
        refresh_stage_ticks={0,96,192,288,384,480,576,672}
        archive_flush_rows=max(32,int(os.environ.get('QUADSMASH_ARCHIVE_FLUSH_ROWS','32')))
        archive_refresh_only=os.environ.get('QUADSMASH_ARCHIVE_REFRESH_ONLY','0')=='1'
        class TrainingArchive(PhysicalTransitionArchive):
            def finish_record(self,*values):
                nonlocal refresh_count
                super().finish_record(*values)
                if archive_refresh_only and not getattr(self,'capture_refresh',False):
                    # The packed initialization replay already supplies the
                    # warmup states.  In the bounded refresh audit, avoid
                    # serializing duplicate warmup/post-refresh rows; only
                    # durable the explicitly captured frozen-policy refresh.
                    self.rows.clear()
                    return
                replay.append(self.rows[-1], policy_version=global_accepted_id, native_tick=tick)
                if getattr(self,'capture_refresh',False):
                    row=self.rows[-1]
                    # Keep only compact reward/terminal arrays plus the small
                    # stage snapshot set needed for bounded critic refresh.
                    refresh_rewards.append(row['reward'].detach().cpu())
                    refresh_dones.append((row['terminated']|row['timeout']).detach().cpu())
                    if refresh_count in refresh_stage_ticks:
                        refresh_stage_records[refresh_count]=tensor_tree_copy(row)
                    refresh_count += 1
                if len(self.rows)>=archive_flush_rows:self.flush()
            def flush(self):
                if os.environ.get('QUADSMASH_ZSTD_ARCHIVE','0') != '1':
                    return super().flush()
                if self.rows:
                    name=f'chunk_{len(self.chunks):05d}.pt.zst'
                    path=self.directory/name
                    out=path.open('wb')
                    proc=subprocess.Popen(['zstd','-q','-T0','-3','-c'],stdin=subprocess.PIPE,stdout=out)
                    try:
                        # torch.save needs a seekable stream; serialize one
                        # chunk in memory, then stream its bytes through zstd.
                        payload=io.BytesIO()
                        torch.save(dict(schema=self.schema,rows=tensor_tree_copy(self.rows,cpu=True)),payload)
                        proc.stdin.write(payload.getbuffer())
                        proc.stdin.close(); rc=proc.wait()
                    except BaseException:
                        proc.kill(); proc.wait(); raise
                    finally:
                        out.close()
                    if rc != 0: raise RuntimeError(f'zstd archive compression failed: {rc}')
                    self.chunks.append(dict(file=name,ticks=len(self.rows),
                                            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                            compression='zstd'))
                    self.rows.clear()
                self._manifest()
        archive=TrainingArchive(args.output/'native_transitions',metadata=dict(manifest,control_ticks=0,transitions=0,chunks=[],
            scope='Mixed-policy training transitions, NOT frozen evaluation',actor_update_budget=args.recovery_update_budget+args.approach_update_budget,
            replay_source_schema=['recent_online','older_online','initial_archive'],
            replay_stage_schema=['uniform','pre_touchdown','touchdown','post_touchdown','recovery_failure','stable_recovery']),chunk_ticks=1000000)
        archive.capture_refresh=False
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
        config=dict(update_mode=args.update_mode,resume_start_block=args.resume_start_block,ball_side_mode=args.ball_side_mode,cycles=args.cycles,recovery_proposal_budget=args.recovery_proposal_budget,approach_proposal_budget=args.approach_proposal_budget,refresh_ticks=args.refresh_ticks,refresh_world_updates=args.refresh_world_updates,refresh_critic_updates=args.refresh_critic_updates,replay_capacity_ticks=args.replay_capacity_ticks,actor_update_budget=(args.recovery_update_budget+args.approach_update_budget if args.update_mode=='sequential' else args.joint_update_budget),joint_update_budget=args.joint_update_budget,recovery_update_budget=args.recovery_update_budget,approach_update_budget=args.approach_update_budget,native_ticks_per_actor_update=args.native_ticks_per_actor_update,world_updates_per_actor_update=args.world_updates_per_actor_update,
            world_batch=128,imagination_batch=128,timebase=timebase.manifest(),actor_lr=5e-5,world_lr=2e-4,critic_lr=2e-4,
            policy_anchor=.05,critic_warmup_updates=200,critic_max_loss=50.,max_critic_only_blocks=32,
            command_delta_budget=args.command_delta_budget,target_delta_budget_rad=args.target_delta_budget_rad,
            native_reference_command_budget=args.native_reference_command_budget,
            native_reference_target_budget_rad=args.native_reference_target_budget_rad,
            actor_whitelist={'joint':RECOVERY_NAMES + APPROACH_NAMES,'recovery':RECOVERY_NAMES,'approach':APPROACH_NAMES},source_sha256=manifest['source_sha256'],packed_data_sha256=packed_sha,
            initialization_native_return=expected['native_return_mean'],extra_mpc=0,
            reference_policy=dict(path=str(reference_path),sha256=file_sha(reference_path)),bad_policy=bad_meta,
            sampling='source×event stratified WM windows; actor starts are causal native event states',
            wm_source_quota=args.wm_source_quota,wm_stage_quota=args.wm_stage_quota,
            task_reference_bank=str(args.task_reference_bank.resolve()) if args.task_reference_bank else None,
            tracking_weight=args.tracking_weight,
            actor_start_ticks=APPROACH_TICKS,critic_start_ticks=CRITIC_TICKS,critic_coverage_period=len(CRITIC_TICKS))
        write(args.output/'config.json',config)
        metadata=dict(init_policy_checkpoint=dict(path=str(source_path),sha256=manifest['source_sha256'],g07_execution_root=str(execution_root)),
                      task_contract=contract,physical_imagination=adapter.manifest)
        global_attempt_id=0;global_accepted_id=0;global_rejected=0;global_backtracked=0
        if args.resume_training:
            global_attempt_id=int(resumed_global.get('attempts',0));global_accepted_id=int(resumed_global.get('accepted',0))
            global_rejected=int(resumed_global.get('rejected',0));global_backtracked=int(resumed_global.get('backtracked',0))
        def save(name):
            archive.flush()  # Checkpoint never points beyond durable replay.
            atomic_torch_save(dict(algorithm='supermimic_v1',adapter_schema=adapter.schema,actor=actor.state_dict(),
                critic=trainer.critic.state_dict(),world=adapter.state_dict(),actor_optimizer=trainer.actor_opt.state_dict(),
                actor_optimizers={name:opt.state_dict() for name,opt in trainer._actor_optimizers.items()},
                world_optimizer=trainer.world_opt.state_dict(),critic_optimizer=trainer.critic_opt.state_dict(),
                actor_updates=trainer.actor_updates,actor_attempts=trainer.actor_attempts,
                rejected_actor_updates=trainer.rejected_actor_updates,world_updates=trainer.world_updates,critic_updates=trainer.critic_updates,
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
                                  stage_names=['uniform','pre_touchdown','touchdown','post_touchdown','recovery_failure','stable_recovery']),
                replay_metadata=dict(
                    streams=replay.streams.cpu(), stage_ids=replay.stage_ids.cpu(),
                    source_ids=replay.source_ids.cpu(), policy_versions=replay.policy_versions.cpu(),
                    native_ticks=replay.native_ticks.cpu()),
                replay_archives=replay_archives+[str(archive.directory)],replay_rng=replay.generator.get_state(),
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),args.output/name)
        def check_stop():
            if args.stop_request['signal'] is not None:raise InterruptedError(args.stop_request['signal'])
        def ensure_critic():
            for retry in range(32):
                if critic_ready(trainer.critic_updates,trainer.last_critic_loss):return
                check_stop()
                state=as_state(starts[retry%len(starts)])
                stats=trainer.imagine_update(state,update_actor=False)
                print('PHYSICAL_CRITIC='+json.dumps(stats),flush=True)
                with (args.output/'critic_history.jsonl').open('a') as f:f.write(json.dumps(stats)+'\n')
                save('latest.pt')
            if not critic_ready(trainer.critic_updates,trainer.last_critic_loss):
                raise RuntimeError('Contextual critic quality gate did not pass within finite32-block budget')
        stage_ticks=list(CRITIC_TICKS)
        stage_states=[as_state(start_cache[t]) for t in stage_ticks]
        stage_meta=[dict(policy_id='initial_reference',episode_id='archive_first_128',time_s=float(t*timebase.control_dt_s)) for t in stage_ticks]
        native_calibration=trainer.audit_native_rtg_stages(stage_states,[rtg[t] for t in stage_ticks],stage_meta)
        native_calibration.update(source='packed_loop/data.pt',discount_mode='physical_time',
                                  horizon_steps=int(rtg.shape[0]),native_return_scope='same-policy first completed 128 episodes')
        write(args.output/'native_critic_stage_audit.json',native_calibration)
        native_calibration_fit=trainer.calibrate_critic_native([as_state(start_cache[0])],rtg[0],epochs=2)
        native_calibration_fit.update(policy_id='initial_reference',source='packed_loop/data.pt',
                                      discount_mode='physical_time',horizon_steps=int(rtg.shape[0]),
                                      native_return_scope='same-policy first completed 128 episodes')
        write(args.output/'native_critic_calibration.json',native_calibration_fit)
        # A already supplies complete native episodes. This warmup concerns the
        # new contextual critic, not recollecting a fresh 800 ticks each update.
        def cover_critic():
            nonlocal critic_coverage_blocks
            check_stop()
            selected=critic_coverage_blocks%len(critic_starts)
            stats=trainer.imagine_update(as_state(critic_starts[selected]),update_actor=False)
            critic_coverage_blocks+=1
            stats.update(coverage_blocks=critic_coverage_blocks,source_tick=CRITIC_TICKS[selected])
            with (args.output/'critic_coverage.jsonl').open('a') as f:f.write(json.dumps(stats)+'\n')
            print('PHYSICAL_CRITIC_COVERAGE='+json.dumps(stats),flush=True)
            save('latest.pt')
        phase='CRITIC'
        # Cover terminal and all preceding task phases before further actor updates.
        # Overlapping 0.64s windows, unchanged BPTT/GAE and discount formula.
        while critic_coverage_blocks<len(critic_starts):cover_critic()
        ensure_critic()
        cursor=0;completed_returns=[];completed_discounted=[];completed_records=[];phase='LEARNING'
        def run_world_updates(iteration, count=None):
            """Fit WM with explicit source×event quotas and return audit totals."""
            total=int(args.world_updates_per_actor_update if count is None else count)
            reports=[]; started=time.perf_counter()
            for offset in range(total):
                window,lanes,sampling=replay.window(
                    timebase.world_steps,128,iteration*total+offset,
                    source_quota=args.wm_source_quota,stage_quota=args.wm_stage_quota)
                stats=trainer.fit_world_window(window,launch[lanes]); del window
                stats['sampling']=sampling; reports.append(stats)
                if writer is not None:
                    step=int(tick*128)
                    for key in ('world_loss','world_physical_loss','world_contact_loss','world_ball_loss','world_task_loss',
                                'world_physical_gradient','world_ball_gradient','world_task_gradient'):
                        if key in stats: writer.add_scalar('wm/'+key.replace('world_',''),float(stats[key]),step)
                    for source,value in sampling.get('source_counts',{}).items():
                        writer.add_scalar('wm/source/'+source,float(value)/128.,step)
                    for stage_name,value in sampling.get('stage_counts',{}).items():
                        writer.add_scalar('wm/stage/'+stage_name,float(value),step)
                    for stage_name,value in sampling.get('stage_shortfall',{}).items():
                        writer.add_scalar('wm/shortfall/'+stage_name,float(value),step)
            aggregate={}
            for key in ('world_loss','world_gradient','world_physical_loss','world_contact_loss','world_ball_loss','world_task_loss',
                        'world_physical_gradient','world_ball_gradient','world_task_gradient','world_seconds'):
                values=[float(item[key]) for item in reports if key in item]
                if values: aggregate[key]=sum(values)/len(values)
            aggregate.update(world_updates=len(reports),world_update_reports=reports,
                             world_update_seconds=time.perf_counter()-started)
            return aggregate
        def run_block(block_name, names, budget, proposal_limit):
            nonlocal completed_returns,completed_discounted,completed_records,obs,tick,cursor
            nonlocal global_attempt_id,global_accepted_id,global_rejected,global_backtracked
            if budget<=0:return 0
            trainer.set_trainable_names(names,block_name)
            dynamic_names=('touchdown','post_touchdown','recovery_failure','stable_recovery') if block_name=='recovery' else ('prepare','approach','pre_touchdown')
            dynamic_pool=[item for name in dynamic_names for item in replay.stage_starts.get(name,[])]
            if block_name=='approach': dynamic_pool=[item for item in dynamic_pool if approach_active(item)]
            block_pool=(dynamic_pool + (recovery_pool if block_name=='recovery' else approach_pool))
            trainer.set_budget_state_library([as_state(row) for row in block_pool+critic_starts[:4]])
            accepted=0;attempts=0;max_attempts=proposal_limit
            while accepted<budget and attempts<max_attempts:
                attempts+=1;global_attempt_id+=1;trainer.block_attempts=attempts
                trainer.actor_updates=accepted
                trainer.actor_attempts=attempts
                check_stop();cover_critic();ensure_critic()
                if shutil.disk_usage(args.output).free<8*1024**3:raise RuntimeError('Disk guard: less than8GiB free')
                started=time.perf_counter()
                with torch.inference_mode():
                    for _ in range(args.native_ticks_per_actor_update):
                        check_stop();action=native_actor_action(obs);obs,reward,done,info=wrapper.step(action);tick+=1
                        if not torch.isfinite(obs['policy']).all():raise FloatingPointError('Nonfinite native observation')
                        for raw in env.rally_records[cursor:]:
                            record=score_terminal_episode(raw,dict(protocol,_root=str(ROOT)));record.update(policy_block=block_name,policy_update_at_end=accepted,scope='mixed-policy monitoring, not frozen score');completed_returns.append(record['native_episode_return']);completed_discounted.append(record['native_episode_discounted_return']);completed_records.append(record)
                            with (args.output/'native_episodes.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                            if writer is not None:
                                step=int(tick*128); total=len(completed_records)
                                metric_names={'qualified_airborne_string_contact':'qualified_contact_rate','ball_side_complete_except_recovery':'ball_side_rate','recovery_success':'conditional_recovery_rate','complete5':'complete5_rate','body_ground_seen':'body_ground_rate'}
                                for key,name in metric_names.items():
                                    writer.add_scalar('native/'+name,sum(bool(x.get(key,False)) for x in completed_records)/total,step)
                                writer.add_scalar('native/episode_return',float(record['native_episode_return']),step)
                                writer.add_scalar('native/return_ema',sum(completed_returns[-16:])/len(completed_returns[-16:]),step)
                                writer.flush()
                        cursor=len(env.rally_records)
                collect_s=time.perf_counter()-started;check_stop()
                world_stats=run_world_updates(global_attempt_id)
                # Rebuild the start pool for every proposal.  Recovery starts
                # therefore follow fresh event-aligned native data as it
                # arrives; the verified archive remains a deterministic
                # fallback when a stage is temporarily under-populated.
                dynamic_pool=[item for name in dynamic_names for item in replay.stage_starts.get(name,[])]
                if block_name=='approach': dynamic_pool=[item for item in dynamic_pool if approach_active(item)]
                pool=dynamic_pool + (recovery_pool if block_name=='recovery' else approach_pool)
                if block_name=='approach' and replay.new_starts:
                    fresh=[item for item in replay.new_starts if approach_active(item)]
                    if fresh: pool=fresh + approach_pool
                if not pool: pool=block_pool
                trainer.set_budget_state_library([as_state(row) for row in pool+critic_starts[:4]])
                if block_name=='approach':
                    active_pool=[item for item in pool if approach_active(item)]
                    pool=active_pool if active_pool else approach_pool
                row=pool[accepted%len(pool)];state=as_state(row);check_stop();stats=trainer.imagine_update(state)
                if stats['update_accepted']:
                    if not any(v>0.0 for v in stats['parameter_max_changes'].values()):
                        raise RuntimeError(f'{block_name} accepted update had zero trainable parameter change; no effective gradient')
                    accepted+=1
                    global_accepted_id+=1
                else:
                    global_rejected+=1
                global_backtracked += int(stats.get('backtracked',False))
                with torch.no_grad():request=actor(state.observation,sample=False)[0];preview,_,_=adapter.executor.forward(state.observation,request,state.clock_goal_context['joint_default'][0],state.clock_goal_context['limits'],state.executor_memory)
                record=dict(**stats,**{k:v for k,v in world_stats.items() if k not in stats},block=block_name,block_accepted_updates=accepted,attempt_id=global_attempt_id,accepted_id=global_accepted_id,global_attempts=global_attempt_id,global_accepted=global_accepted_id,global_rejected=global_rejected,global_backtracked=global_backtracked,native_ticks=tick,native_transitions=tick*128,native_collect_seconds=collect_s, native_reward_mean=float(reward.mean()),completed_native_episodes=len(completed_returns),native_return_mean=sum(completed_returns)/len(completed_returns) if completed_returns else None,native_discounted_return_mean=sum(completed_discounted)/len(completed_discounted) if completed_discounted else None,raw_return_scope='completed mixed-policy monitoring; incomplete episodes excluded',peak_gpu_bytes=torch.cuda.max_memory_allocated(),model_calls=adapter.calls,window_sampling=world_stats.get('world_update_reports',[]),imagined_start_source='fresh_native' if replay.new_starts and any(pool_item is replay.new_starts[-1] for pool_item in pool) else 'event_aligned_replay')
                if writer is not None:
                    step=int(tick*128)
                    writer.add_scalar('native/step_reward_mean',float(reward.mean()),step)
                    writer.add_scalar('native/completed_episodes',len(completed_returns),step)
                    if completed_returns: writer.add_scalar('native/return_mean_128',sum(completed_returns)/len(completed_returns),step)
                    writer.add_scalar('actor/imagined_return',float(stats.get('imagined_return_mean',0.0)),step)
                    writer.add_scalar('model/world_loss',float(world_stats.get('world_loss',0.0)),step)
                    writer.add_scalar('critic/loss',float(stats.get('critic_loss',0.0)),step)
                    writer.add_scalar('actor/%s_updates'%block_name,accepted,step)
                    writer.add_scalar('actor/%s_gate_active'%block_name,float(stats.get('effective_gate_lanes',0)),step)
                    writer.add_scalar('actor/effective_updates',float(stats.get('actor_updates',0)),step)
                    writer.add_scalar('actor/tracking_loss',float(stats.get('tracking_loss',0.0)),step)
                    writer.add_scalar('actor/accepted_updates',global_accepted_id,step)
                    writer.add_scalar('actor/rejected_updates',global_rejected,step)
                    writer.add_scalar('performance/native_collect_s',collect_s,step)
                    writer.add_scalar('performance/imagination_s',float(stats.get('bptt_seconds',0.0)),step)
                    writer.add_scalar('performance/critic_s',float(stats.get('critic_seconds',0.0)),step)
                    writer.add_scalar('performance/gpu_peak_memory',float(torch.cuda.max_memory_allocated()),step)
                    writer.flush()
                save('latest.pt');write(args.output/'progress.json',record)
                with (args.output/'training_history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                print('PHYSICAL_LEARNING='+json.dumps(record),flush=True)
                if accepted and accepted%16==0:save(f'{block_name}_checkpoint_{accepted:04d}.pt')
            save(f'{block_name}_candidate_{accepted:04d}.pt')
            return accepted
        def refresh_frozen_recovery():
            """Collect one reset-separated full native block before Approach."""
            nonlocal obs,tick,refresh_count,refresh_world_updates,refresh_critic_updates
            if args.refresh_ticks == 0:
                return 0,0
            archive.capture_refresh=False
            refresh_records.clear()
            refresh_rewards.clear(); refresh_dones.clear(); refresh_stage_records.clear(); refresh_count=0
            start_tick=tick
            # Finish the partially sampled episode through the native timeout.
            # Manual reset before an 800-tick terminal record is rejected by the
            # task's pre-reset evidence contract. The environment performs the
            # reset itself at the boundary, after which this stream starts.
            warmup_ticks=int(args.resume_warmup_ticks)
            with torch.inference_mode():
                for _ in range(warmup_ticks):
                    check_stop()
                    action=native_actor_action(obs)
                    obs,reward,done,info=wrapper.step(action)
                    tick+=1
                    if not torch.isfinite(obs['policy']).all():
                        raise FloatingPointError('Nonfinite observation during recovery boundary warmup')
            replay.new_stream(source='online', policy_version=global_accepted_id)
            refresh_records.clear()
            refresh_rewards.clear(); refresh_dones.clear(); refresh_stage_records.clear(); refresh_count=0
            archive.capture_refresh=True
            with torch.inference_mode():
                for _ in range(args.refresh_ticks):
                    check_stop()
                    action=native_actor_action(obs)
                    obs,reward,done,info=wrapper.step(action)
                    tick+=1
                    if not torch.isfinite(obs['policy']).all():
                        raise FloatingPointError('Nonfinite native observation during recovery refresh')
            archive.capture_refresh=False
            archive.flush()
            if args.refresh_ticks and refresh_count!=args.refresh_ticks:
                raise RuntimeError(f'Expected {args.refresh_ticks} refresh rows, got {refresh_count}')
            # Same frozen-policy labels, separated by episode and stage. These
            # labels are used only for bounded critic calibration, never actor input.
            rewards=torch.stack(refresh_rewards).double()
            dones=torch.stack(refresh_dones)
            rtg_refresh=torch.zeros_like(rewards)
            running=torch.zeros(rewards.shape[1],dtype=torch.float64)
            for ti in range(args.refresh_ticks-1,-1,-1):
                running=rewards[ti]+timebase.gamma*running*(~dones[ti]).double()
                rtg_refresh[ti]=running
            stage_ticks_refresh=[0,96,192,288,384,480,576,672]
            refresh_world=0
            if args.refresh_world_updates:
                refresh_stats=run_world_updates(global_attempt_id, args.refresh_world_updates)
                refresh_world=int(refresh_stats['world_updates'])
            refresh_critic=0
            refresh_critic_losses=[]
            for update in range(args.refresh_critic_updates):
                ti=stage_ticks_refresh[update%len(stage_ticks_refresh)]
                # Refresh critic values against the completed frozen-policy
                # native return-to-go.  Do not call imagine_update here: that
                # would train the critic on a model rollout and silently erase
                # the policy/episode-aligned native labels just computed above.
                state=as_state(refresh_stage_records[ti])
                fit=trainer.calibrate_critic_native(
                    [state], rtg_refresh[ti].to(state.physical.device), epochs=1)
                refresh_critic += int(fit['native_critic_calibration_updates'])
                refresh_critic_losses.append(float(fit['native_critic_calibration_loss']))
            write(args.output/'refresh_manifest.json',dict(
                policy_id='recovery_frozen_refresh',start_tick=start_tick,end_tick=tick,
                ticks=refresh_count,transitions=refresh_count*128,
                world_updates=refresh_world,critic_updates=refresh_critic,
                critic_update_mode='native_rtg_calibration',
                critic_calibration_losses=refresh_critic_losses,
                stage_ticks=stage_ticks_refresh,rtg_scope='same frozen recovery policy',
                replay_count=int(replay.count),replay_capacity=int(replay.capacity),warmup_ticks=warmup_ticks,
                missing_fields=[]))
            print('PHYSICAL_REFRESH='+json.dumps(dict(ticks=refresh_count,transitions=refresh_count*128,world_updates=refresh_world,critic_updates=refresh_critic)),flush=True)
            refresh_world_updates,refresh_critic_updates=refresh_world,refresh_critic
            return refresh_world,refresh_critic
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
                with torch.no_grad(): request=actor(state.observation,sample=False)[0]; preview,_,_=adapter.executor.forward(state.observation,request,state.clock_goal_context['joint_default'][0],state.clock_goal_context['limits'],state.executor_memory)
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
        else:
            # Default production path: recovery then approach. Each block owns
            # its trainable head and Adam state; the intermediate checkpoint is
            # written before the approach block starts.
            cycle_results=[]
            for cycle in range(args.cycles):
                if args.resume_start_block=='approach' and cycle==0:
                    recovery_accepted=0
                    refresh_world_updates,refresh_critic_updates=0,0
                else:
                    recovery_accepted=run_block('recovery', RECOVERY_NAMES, min(args.recovery_update_budget,args.recovery_proposal_budget), args.recovery_proposal_budget)
                    save('cycle_%02d_recovery_candidate_%04d.pt' % (cycle+1,recovery_accepted))
                    refresh_world_updates,refresh_critic_updates=refresh_frozen_recovery()
                save('cycle_%02d_recovery_refresh_candidate_%04d.pt' % (cycle+1,recovery_accepted))
                approach_accepted=run_block('approach', APPROACH_NAMES, min(args.approach_update_budget,args.approach_proposal_budget), args.approach_proposal_budget)
                total_accepted=recovery_accepted+approach_accepted
                archive.flush(); save('cycle_%02d_candidate_%04d.pt' % (cycle+1,total_accepted))
                cycle_results.append(dict(cycle=cycle+1,recovery_accepted=recovery_accepted,approach_accepted=approach_accepted,
                    native_ticks=tick,world_updates=trainer.world_updates,critic_updates=trainer.critic_updates,
                    refresh_world_updates=refresh_world_updates,refresh_critic_updates=refresh_critic_updates))
                write(args.output/'cycle_%02d.json' % (cycle+1),cycle_results[-1])
            total_accepted=sum(x['recovery_accepted']+x['approach_accepted'] for x in cycle_results)
            write(args.output/'training_completed.json', dict(
                update_mode='sequential', cycles=args.cycles, actor_updates=total_accepted,
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
            world_stats=trainer.fit_world_window(window,launch[lanes]);del window
            pool=replay.new_starts if replay.new_starts and trainer.actor_updates%2 else starts
            row=pool[trainer.actor_updates%len(pool)];state=as_state(row)
            check_stop()
            stats=trainer.imagine_update(state)
            if stats['update_accepted'] and not any(v>0 for v in stats['parameter_max_changes'].values()):
                raise RuntimeError('Accepted actor optimizer step made no approach parameter change')
            with torch.no_grad():
                request=actor(state.observation,sample=False)[0]
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
        if trainer is not None and 'save' in locals():save('interrupted.pt')
        raise
    finally:
        if writer is not None:
            writer.flush(); writer.close()
        if archive is not None:
            archive.flush()
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
    except BaseException:
        os._exit(1)
    else:
        os._exit(0)
