#!/usr/bin/env python3
"""One bounded 256-actor-update physical SuperMimic integration block.

No automatic evaluation, parameter scan, reference trajectory or online MPC.
Fresh native interaction continues across learner updates; no 800-tick wait.
"""
import argparse,hashlib,json,os,shutil,subprocess,sys,time,traceback,signal
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,default=str)+'\n');temp.replace(path)


def main():
    from isaaclab.app import AppLauncher
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('archive','packed-data','physical-checkpoint','ball-checkpoint','task-checkpoint','safety-mapping','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--resume-training',type=Path)
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
    parser.add_argument('--refresh-world-updates',type=int,default=32)
    parser.add_argument('--refresh-critic-updates',type=int,default=20)
    parser.add_argument('--reference-policy-checkpoint',type=Path)
    parser.add_argument('--bad-policy-checkpoint',type=Path)
    parser.add_argument('--video-dir',type=Path)
    parser.add_argument('--command-delta-budget',type=float,default=0.005)
    parser.add_argument('--target-delta-budget-rad',type=float,default=0.04)
    AppLauncher.add_app_launcher_args(parser);args=parser.parse_args()
    if args.actor_update_budget is not None and args.actor_update_budget < 1: parser.error('actor-update-budget must be positive')
    if args.recovery_update_budget < 0 or args.approach_update_budget < 0: parser.error('block budgets must be non-negative')
    if args.recovery_proposal_budget < 1 or args.approach_proposal_budget < 1: parser.error('proposal budgets must be positive')
    if args.refresh_ticks < 1 or args.refresh_world_updates < 0 or args.refresh_critic_updates < 0: parser.error('refresh budgets invalid')
    if args.ball_side_mode == 'enforce': parser.error('ball-side enforce mode is not implemented; use shadow for this audit-only revision')
    if args.actor_update_budget is not None:
        args.recovery_update_budget=0;args.approach_update_budget=args.actor_update_budget
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
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
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from quadsmash.rl.supermimic_physical_loop_loading_v1 import load_physical_loop,APPROACH_NAMES,RECOVERY_NAMES
    from quadsmash.rl.supermimic_physical_learner_v1 import PhysicalHorizonTrainer
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
    env=trainer=archive=None;failed=False;tick=0;phase='LOAD'
    torch.set_num_threads(2);torch.manual_seed(20261114)
    try:
        actor,adapter,template,launch,manifest=load_physical_loop(archive=args.archive,
            physical_checkpoint=args.physical_checkpoint,ball_checkpoint=args.ball_checkpoint,
            task_checkpoint=args.task_checkpoint,safety_mapping=args.safety_mapping)
        write(args.output/'model_manifest.json',adapter.manifest)
        packed_sha=hashlib.sha256(args.packed_data.read_bytes()).hexdigest()
        expected=json.loads((args.packed_data.parent/'completed.json').read_text())
        if packed_sha!=expected['sha256'] or expected['first_completed']!=128:raise ValueError('Invalid initialization data')
        package=torch.load(args.packed_data,map_location='cpu',weights_only=True)
        replay=PhysicalLoopReplay(package,adapter.task_model.memory_schema)
        from quadsmash.rl.supermimic_physical_start_cache_v1 import load_start_cache,APPROACH_TICKS,CRITIC_TICKS
        start_cache=load_start_cache(args.archive,manifest)
        starts=[start_cache[t] for t in APPROACH_TICKS]
        critic_starts=[start_cache[t] for t in CRITIC_TICKS]
        def as_state(row):return PhysicalImaginationStateV1.from_record(row,executor_template=template,launch=launch,device='cuda:0')
        timebase=Timebase(**json.loads((ROOT/'configs/supermimic_task_horizon_v2.json').read_text()))
        trainer=PhysicalHorizonTrainer(actor,adapter,as_state(starts[1]),timebase,
            command_delta_budget=args.command_delta_budget,
            target_delta_budget_rad=args.target_delta_budget_rad,
            trainable_names=RECOVERY_NAMES + APPROACH_NAMES,block_name='joint')
        trainer.set_budget_state_library([as_state(row) for row in starts+critic_starts[:4]])
        approach_pool=[start_cache[t] for t in APPROACH_TICKS]
        recovery_pool=[start_cache[t] for t in (272,288,336,384)]
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
        if args.resume_training:
            resumed=torch.load(args.resume_training,map_location='cuda:0',weights_only=False)
            if resumed['adapter_schema']!=adapter.schema or resumed['metadata']['physical_imagination']!=adapter.manifest:
                raise ValueError('Resume physical model/execution source mismatch')
            actor.load_state_dict(resumed['actor'],strict=True);adapter.load_state_dict(resumed['world'],strict=True)
            trainer.critic.load_state_dict(resumed['critic'],strict=True)
            trainer.actor_opt.load_state_dict(resumed['actor_optimizer']);trainer.world_opt.load_state_dict(resumed['world_optimizer'])
            trainer.critic_opt.load_state_dict(resumed['critic_optimizer'])
            for name in ('actor_updates','world_updates','critic_updates','last_critic_loss'):setattr(trainer,name,resumed[name])
            tick=int(resumed['native_ticks'])
            critic_coverage_blocks=int(resumed.get('critic_coverage_blocks',0))
            if trainer.frozen_hash()!=trainer.frozen_actor_hash:raise ValueError('Resume changed frozen actor parameters')
            replay_archives=resumed['replay_archives']
            for directory in replay_archives:
                directory=Path(directory);receipt=json.loads((directory/'manifest.json').read_text())
                for item in receipt['chunks']:
                    path=directory/item['file']
                    if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Resume archive hash mismatch')
                    for row in torch.load(path,map_location='cpu',weights_only=True)['rows']:replay.append(row)
                replay.new_stream()
            replay.generator.set_state(resumed['replay_rng'].cpu())
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
        refresh_records=[]
        class TrainingArchive(PhysicalTransitionArchive):
            def finish_record(self,*values):
                super().finish_record(*values)
                replay.append(self.rows[-1])
                if getattr(self,'capture_refresh',False):
                    refresh_records.append(tensor_tree_copy(self.rows[-1]))
                if len(self.rows)>=32:self.flush()
        archive=TrainingArchive(args.output/'native_transitions',metadata=dict(manifest,control_ticks=0,transitions=0,chunks=[],
            scope='Mixed-policy training transitions, NOT frozen evaluation',actor_update_budget=args.recovery_update_budget+args.approach_update_budget),chunk_ticks=1000000)
        archive.capture_refresh=False
        archive.attach_task_sequence(env);env.physical_transition_archive=archive
        def file_sha(path):
            return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None
        reference_path=args.reference_policy_checkpoint or source_path
        if file_sha(reference_path)!=manifest['source_sha256']:
            raise ValueError('Reference policy checkpoint does not match the verified source actor')
        bad_meta=dict(path=str(args.bad_policy_checkpoint.resolve()),sha256=file_sha(args.bad_policy_checkpoint)) if args.bad_policy_checkpoint else None
        config=dict(update_mode=args.update_mode,ball_side_mode=args.ball_side_mode,recovery_proposal_budget=args.recovery_proposal_budget,approach_proposal_budget=args.approach_proposal_budget,refresh_ticks=args.refresh_ticks,refresh_world_updates=args.refresh_world_updates,refresh_critic_updates=args.refresh_critic_updates,actor_update_budget=(args.recovery_update_budget+args.approach_update_budget if args.update_mode=='sequential' else args.joint_update_budget),joint_update_budget=args.joint_update_budget,recovery_update_budget=args.recovery_update_budget,approach_update_budget=args.approach_update_budget,native_ticks_per_actor_update=4,world_updates_per_actor_update=1,
            world_batch=128,imagination_batch=128,timebase=timebase.manifest(),actor_lr=5e-5,world_lr=2e-4,critic_lr=2e-4,
            policy_anchor=.05,critic_warmup_updates=200,critic_max_loss=50.,max_critic_only_blocks=32,
            command_delta_budget=args.command_delta_budget,target_delta_budget_rad=args.target_delta_budget_rad,
            actor_whitelist={'joint':RECOVERY_NAMES + APPROACH_NAMES,'recovery':RECOVERY_NAMES,'approach':APPROACH_NAMES},source_sha256=manifest['source_sha256'],packed_data_sha256=packed_sha,
            initialization_native_return=expected['native_return_mean'],extra_mpc=0,
            reference_policy=dict(path=str(reference_path),sha256=file_sha(reference_path)),bad_policy=bad_meta,
            sampling='actor: initial/fresh approach; critic: overlapping full-episode current starts, reverse-time order',
            actor_start_ticks=APPROACH_TICKS,critic_start_ticks=CRITIC_TICKS,critic_coverage_period=len(CRITIC_TICKS))
        write(args.output/'config.json',config)
        metadata=dict(init_policy_checkpoint=dict(path=str(source_path),sha256=manifest['source_sha256'],g07_execution_root=str(execution_root)),
                      task_contract=contract,physical_imagination=adapter.manifest)
        def save(name):
            archive.flush()  # Checkpoint never points beyond durable replay.
            atomic_torch_save(dict(algorithm='supermimic_v1',adapter_schema=adapter.schema,actor=actor.state_dict(),
                critic=trainer.critic.state_dict(),world=adapter.state_dict(),actor_optimizer=trainer.actor_opt.state_dict(),
                world_optimizer=trainer.world_opt.state_dict(),critic_optimizer=trainer.critic_opt.state_dict(),
                actor_updates=trainer.actor_updates,actor_attempts=trainer.actor_attempts,
                rejected_actor_updates=trainer.rejected_actor_updates,world_updates=trainer.world_updates,critic_updates=trainer.critic_updates,
                native_ticks=tick,metadata=metadata,config=config,last_critic_loss=trainer.last_critic_loss,
                critic_coverage_blocks=critic_coverage_blocks,
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
        cursor=0;completed_returns=[];completed_discounted=[];phase='LEARNING'
        global_attempt_id=0;global_accepted_id=0;global_rejected=0;global_backtracked=0
        def run_block(block_name, names, budget, proposal_limit):
            nonlocal completed_returns,completed_discounted,obs,tick,cursor
            nonlocal global_attempt_id,global_accepted_id,global_rejected,global_backtracked
            if budget<=0:return 0
            trainer.set_trainable_names(names,block_name)
            block_pool=recovery_pool if block_name=='recovery' else approach_pool
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
                    for _ in range(4):
                        check_stop();action=actor(obs['policy'],sample=False)[0];obs,reward,done,info=wrapper.step(action);tick+=1
                        if not torch.isfinite(obs['policy']).all():raise FloatingPointError('Nonfinite native observation')
                        for raw in env.rally_records[cursor:]:
                            record=score_terminal_episode(raw,dict(protocol,_root=str(ROOT)));record.update(policy_block=block_name,policy_update_at_end=accepted,scope='mixed-policy monitoring, not frozen score');completed_returns.append(record['native_episode_return']);completed_discounted.append(record['native_episode_discounted_return'])
                            with (args.output/'native_episodes.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                        cursor=len(env.rally_records)
                collect_s=time.perf_counter()-started;check_stop()
                window,lanes,sampling=replay.window(timebase.world_steps,128,accepted);world_stats=trainer.fit_world_window(window,launch[lanes]);del window
                pool=replay.new_starts if (block_name=='approach' and replay.new_starts and accepted%2) else block_pool;row=pool[accepted%len(pool)];state=as_state(row);check_stop();stats=trainer.imagine_update(state)
                if stats['update_accepted']:
                    if not any(v>0.0 for v in stats['parameter_max_changes'].values()):
                        raise RuntimeError(f'{block_name} accepted update had zero trainable parameter change; no effective gradient')
                    accepted+=1
                    global_accepted_id+=1
                else:
                    global_rejected+=1
                global_backtracked += int(stats.get('backtracked',False))
                with torch.no_grad():request=actor(state.observation,sample=False)[0];preview,_,_=adapter.executor.forward(state.observation,request,state.clock_goal_context['joint_default'][0],state.clock_goal_context['limits'],state.executor_memory)
                record=dict(**stats,**{k:v for k,v in world_stats.items() if k not in stats},block=block_name,block_accepted_updates=accepted,attempt_id=global_attempt_id,accepted_id=global_accepted_id,global_attempts=global_attempt_id,global_accepted=global_accepted_id,global_rejected=global_rejected,global_backtracked=global_backtracked,native_ticks=tick,native_transitions=tick*128,native_collect_seconds=collect_s, native_reward_mean=float(reward.mean()),completed_native_episodes=len(completed_returns),native_return_mean=sum(completed_returns)/len(completed_returns) if completed_returns else None,native_discounted_return_mean=sum(completed_discounted)/len(completed_discounted) if completed_discounted else None,raw_return_scope='completed mixed-policy monitoring; incomplete episodes excluded',peak_gpu_bytes=torch.cuda.max_memory_allocated(),model_calls=adapter.calls,window_sampling=sampling,imagined_start_source='fresh_native' if pool is replay.new_starts else 'initialization_archive')
                save('latest.pt');write(args.output/'progress.json',record)
                with (args.output/'training_history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                print('PHYSICAL_LEARNING='+json.dumps(record),flush=True)
                if accepted and accepted%16==0:save(f'{block_name}_checkpoint_{accepted:04d}.pt')
            save(f'{block_name}_candidate_{accepted:04d}.pt')
            return accepted
        def refresh_frozen_recovery():
            """Collect one reset-separated full native block before Approach."""
            nonlocal obs,tick
            archive.capture_refresh=True
            refresh_records.clear()
            replay.new_stream()
            obs,_=wrapper.reset()
            start_tick=tick
            with torch.inference_mode():
                for _ in range(args.refresh_ticks):
                    check_stop()
                    action=actor(obs['policy'],sample=False)[0]
                    obs,reward,done,info=wrapper.step(action)
                    tick+=1
                    if not torch.isfinite(obs['policy']).all():
                        raise FloatingPointError('Nonfinite native observation during recovery refresh')
            archive.capture_refresh=False
            archive.flush()
            if len(refresh_records)!=args.refresh_ticks:
                raise RuntimeError(f'Expected {args.refresh_ticks} refresh rows, got {len(refresh_records)}')
            # Same frozen-policy labels, separated by episode and stage. These
            # labels are used only for bounded critic calibration, never actor input.
            rewards=torch.stack([r['reward'].detach().cpu() for r in refresh_records]).double()
            dones=torch.stack([(r['terminated']|r['timeout']).detach().cpu() for r in refresh_records])
            rtg_refresh=torch.zeros_like(rewards)
            running=torch.zeros(rewards.shape[1],dtype=torch.float64)
            for ti in range(args.refresh_ticks-1,-1,-1):
                running=rewards[ti]+timebase.gamma*running*(~dones[ti]).double()
                rtg_refresh[ti]=running
            stage_ticks_refresh=[0,96,192,288,384,480,576,672]
            refresh_world=0
            for update in range(args.refresh_world_updates):
                window,lanes,sampling=replay.window(timebase.world_steps,128,update)
                trainer.fit_world_window(window,launch[lanes]);del window
                refresh_world+=1
            refresh_critic=0
            for update in range(args.refresh_critic_updates):
                ti=stage_ticks_refresh[update%len(stage_ticks_refresh)]
                trainer.imagine_update(as_state(refresh_records[ti]),update_actor=False)
                refresh_critic+=1
            write(args.output/'refresh_manifest.json',dict(
                policy_id='recovery_frozen_refresh',start_tick=start_tick,end_tick=tick,
                ticks=len(refresh_records),transitions=len(refresh_records)*128,
                world_updates=refresh_world,critic_updates=refresh_critic,
                stage_ticks=stage_ticks_refresh,rtg_scope='same frozen recovery policy',
                replay_count=int(replay.count),replay_capacity=int(replay.capacity),
                missing_fields=[]))
            print('PHYSICAL_REFRESH='+json.dumps(dict(ticks=len(refresh_records),transitions=len(refresh_records)*128,world_updates=refresh_world,critic_updates=refresh_critic)),flush=True)
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
                    for _ in range(4):
                        check_stop(); action=actor(obs['policy'],sample=False)[0]; obs,reward,done,info=wrapper.step(action); tick+=1
                        if not torch.isfinite(obs['policy']).all(): raise FloatingPointError('Nonfinite native observation')
                        for raw in env.rally_records[cursor:]:
                            record=score_terminal_episode(raw,dict(protocol,_root=str(ROOT))); record.update(policy_block='joint',policy_update_at_end=accepted,scope='mixed-policy monitoring, not frozen score'); completed_returns.append(record['native_episode_return']); completed_discounted.append(record['native_episode_discounted_return'])
                            with (args.output/'native_episodes.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                        cursor=len(env.rally_records)
                collect_s=time.perf_counter()-started; check_stop()
                window,lanes,sampling=replay.window(timebase.world_steps,128,accepted); world_stats=trainer.fit_world_window(window,launch[lanes]); del window
                row=joint_pool[accepted%len(joint_pool)]; state=as_state(row); check_stop(); stats=trainer.imagine_update(state)
                if stats['update_accepted']:
                    if not any(v>0.0 for v in stats['parameter_max_changes'].values()): raise RuntimeError('joint accepted update had zero trainable parameter change')
                    accepted += 1
                with torch.no_grad(): request=actor(state.observation,sample=False)[0]; preview,_,_=adapter.executor.forward(state.observation,request,state.clock_goal_context['joint_default'][0],state.clock_goal_context['limits'],state.executor_memory)
                record=dict(**stats,**{k:v for k,v in world_stats.items() if k not in stats},block='joint',block_accepted_updates=accepted,native_ticks=tick,native_transitions=tick*128,native_collect_seconds=collect_s,native_reward_mean=float(reward.mean()),completed_native_episodes=len(completed_returns),native_return_mean=sum(completed_returns)/len(completed_returns) if completed_returns else None,native_discounted_return_mean=sum(completed_discounted)/len(completed_discounted) if completed_discounted else None,raw_return_scope='completed mixed-policy monitoring; incomplete episodes excluded',peak_gpu_bytes=torch.cuda.max_memory_allocated(),model_calls=adapter.calls,window_sampling=sampling,imagined_start_source='initialization_archive')
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
            recovery_accepted=run_block('recovery', RECOVERY_NAMES, min(args.recovery_update_budget,args.recovery_proposal_budget), args.recovery_proposal_budget)
            save('recovery_candidate_%04d.pt' % recovery_accepted)
            refresh_world_updates,refresh_critic_updates=refresh_frozen_recovery()
            save('recovery_refresh_candidate_%04d.pt' % recovery_accepted)
            approach_accepted=run_block('approach', APPROACH_NAMES, min(args.approach_update_budget,args.approach_proposal_budget), args.approach_proposal_budget)
            total_accepted=recovery_accepted+approach_accepted
            archive.flush(); save('candidate_%04d.pt' % total_accepted)
            write(args.output/'training_completed.json', dict(
                update_mode='sequential', actor_updates=total_accepted,
                recovery_accepted=recovery_accepted, approach_accepted=approach_accepted,
                proposal_attempts=global_attempt_id, global_accepted=global_accepted_id,
                global_rejected=global_rejected, global_backtracked=global_backtracked,
                native_ticks=tick, actor_attempts=trainer.actor_attempts,
                rejected_actor_updates=trainer.rejected_actor_updates,
                world_updates=trainer.world_updates, critic_updates=trainer.critic_updates,
                refresh_world_updates=refresh_world_updates, refresh_critic_updates=refresh_critic_updates,
                formal_evaluation=False, blocks=['recovery','approach'],
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
                    action=actor(obs['policy'],sample=False)[0]
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
