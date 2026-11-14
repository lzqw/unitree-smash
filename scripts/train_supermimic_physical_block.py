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
    parser.add_argument('--actor-update-budget',type=int,default=256)
    parser.add_argument('--reference-policy-checkpoint',type=Path)
    parser.add_argument('--bad-policy-checkpoint',type=Path)
    parser.add_argument('--video-dir',type=Path)
    parser.add_argument('--command-delta-budget',type=float,default=0.005)
    parser.add_argument('--target-delta-budget-rad',type=float,default=0.04)
    AppLauncher.add_app_launcher_args(parser);args=parser.parse_args()
    if args.actor_update_budget < 1: parser.error('actor-update-budget must be positive')
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
    from quadsmash.rl.supermimic_physical_loop_loading_v1 import load_physical_loop,APPROACH_NAMES
    from quadsmash.rl.supermimic_physical_learner_v1 import PhysicalHorizonTrainer
    from quadsmash.rl.supermimic_physical_loop_replay_v1 import PhysicalLoopReplay,current_start
    from quadsmash.rl.supermimic_physical_imagination_state_v1 import PhysicalImaginationStateV1
    from quadsmash.rl.supermimic_physical_capture_v1 import PhysicalTransitionArchive
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
        replay=PhysicalLoopReplay(package,adapter.task_model.memory_schema);del package
        from quadsmash.rl.supermimic_physical_start_cache_v1 import load_start_cache,APPROACH_TICKS,CRITIC_TICKS
        start_cache=load_start_cache(args.archive,manifest)
        starts=[start_cache[t] for t in APPROACH_TICKS]
        critic_starts=[start_cache[t] for t in CRITIC_TICKS]
        def as_state(row):return PhysicalImaginationStateV1.from_record(row,executor_template=template,launch=launch,device='cuda:0')
        timebase=Timebase(**json.loads((ROOT/'configs/supermimic_task_horizon_v2.json').read_text()))
        trainer=PhysicalHorizonTrainer(actor,adapter,as_state(starts[1]),timebase,
            command_delta_budget=args.command_delta_budget,
            target_delta_budget_rad=args.target_delta_budget_rad)
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
        class TrainingArchive(PhysicalTransitionArchive):
            def finish_record(self,*values):
                super().finish_record(*values)
                replay.append(self.rows[-1])
                if len(self.rows)>=32:self.flush()
        archive=TrainingArchive(args.output/'native_transitions',metadata=dict(manifest,control_ticks=0,transitions=0,chunks=[],
            scope='Mixed-policy training transitions, NOT frozen evaluation',actor_update_budget=256),chunk_ticks=1000000)
        archive.attach_task_sequence(env);env.physical_transition_archive=archive
        def file_sha(path):
            return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None
        reference_path=args.reference_policy_checkpoint or source_path
        if file_sha(reference_path)!=manifest['source_sha256']:
            raise ValueError('Reference policy checkpoint does not match the verified source actor')
        bad_meta=dict(path=str(args.bad_policy_checkpoint.resolve()),sha256=file_sha(args.bad_policy_checkpoint)) if args.bad_policy_checkpoint else None
        config=dict(actor_update_budget=args.actor_update_budget,native_ticks_per_actor_update=4,world_updates_per_actor_update=1,
            world_batch=128,imagination_batch=128,timebase=timebase.manifest(),actor_lr=5e-5,world_lr=2e-4,critic_lr=2e-4,
            policy_anchor=.05,critic_warmup_updates=200,critic_max_loss=50.,max_critic_only_blocks=32,
            command_delta_budget=args.command_delta_budget,target_delta_budget_rad=args.target_delta_budget_rad,
            actor_whitelist=APPROACH_NAMES,source_sha256=manifest['source_sha256'],packed_data_sha256=packed_sha,
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


if __name__=='__main__':main()
