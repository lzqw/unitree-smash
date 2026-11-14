#!/usr/bin/env python3
"""One bounded 256-actor-update physical SuperMimic integration block.

No automatic evaluation, parameter scan, reference trajectory or online MPC.
Fresh native interaction continues across learner updates; no 800-tick wait.
"""
import argparse,hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,default=str)+'\n');temp.replace(path)


def main():
    from isaaclab.app import AppLauncher
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('archive','packed-data','physical-checkpoint','ball-checkpoint','task-checkpoint','safety-mapping','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    AppLauncher.add_app_launcher_args(parser);args=parser.parse_args()
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    write(args.output/'startup.json',dict(argv=sys.argv,cwd=os.getcwd(),executable=sys.executable,
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        dirty_diff=subprocess.check_output(['git','diff'],text=True),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    from quadsmash.rl.supermimic_run_lifecycle import execute_scoped_native,stage
    stage(args.output,'BEFORE_APP');app=AppLauncher(args,fast_shutdown=False).app
    stage(args.output,'AFTER_APP')
    execute_scoped_native(args.output,lambda:run(args),app)


def run(args):
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from quadsmash.rl.supermimic_physical_loop_loading_v1 import load_physical_loop,APPROACH_NAMES
    from quadsmash.rl.supermimic_physical_learner_v1 import PhysicalHorizonTrainer
    from quadsmash.rl.supermimic_physical_loop_replay_v1 import PhysicalLoopReplay,current_start
    from quadsmash.rl.supermimic_physical_imagination_state_v1 import PhysicalImaginationStateV1
    from quadsmash.rl.supermimic_physical_capture_v1 import PhysicalTransitionArchive
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
        starts=[];chunks={}
        for t in (224,240,256,272):
            c=t//32
            if c not in chunks:chunks[c]=torch.load(args.archive/manifest['chunks'][c]['file'],map_location='cpu',weights_only=True)['rows']
            starts.append(current_start(chunks[c][t%32]))
        del chunks
        def as_state(row):return PhysicalImaginationStateV1.from_record(row,executor_template=template,launch=launch,device='cuda:0')
        timebase=Timebase(**json.loads((ROOT/'configs/supermimic_task_horizon_v2.json').read_text()))
        trainer=PhysicalHorizonTrainer(actor,adapter,as_state(starts[1]),timebase)
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
        stage(args.output,'AFTER_RESET')
        class TrainingArchive(PhysicalTransitionArchive):
            def finish_record(self,*values):
                super().finish_record(*values)
                replay.append(self.rows[-1])
                if len(self.rows)>=32:self.flush()
        archive=TrainingArchive(args.output/'native_transitions',metadata=dict(manifest,control_ticks=0,transitions=0,chunks=[],
            scope='Mixed-policy training transitions, NOT frozen evaluation',actor_update_budget=256),chunk_ticks=1000000)
        archive.attach_task_sequence(env);env.physical_transition_archive=archive
        config=dict(actor_update_budget=256,native_ticks_per_actor_update=4,world_updates_per_actor_update=1,
            world_batch=128,imagination_batch=128,timebase=timebase.manifest(),actor_lr=5e-5,world_lr=2e-4,critic_lr=2e-4,
            policy_anchor=.05,critic_warmup_updates=200,critic_max_loss=50.,max_critic_only_blocks=32,
            actor_whitelist=APPROACH_NAMES,source_sha256=manifest['source_sha256'],packed_data_sha256=packed_sha,
            initialization_native_return=expected['native_return_mean'],extra_mpc=0,
            sampling='initial current starts at ticks224/240/256/272; alternate refreshed native approach starts once available')
        write(args.output/'config.json',config)
        metadata=dict(init_policy_checkpoint=dict(path=str(source_path),sha256=manifest['source_sha256'],g07_execution_root=str(execution_root)),
                      task_contract=contract,physical_imagination=adapter.manifest)
        def save(name):
            torch.save(dict(algorithm='supermimic_v1',adapter_schema=adapter.schema,actor=actor.state_dict(),
                critic=trainer.critic.state_dict(),world=adapter.state_dict(),actor_optimizer=trainer.actor_opt.state_dict(),
                world_optimizer=trainer.world_opt.state_dict(),critic_optimizer=trainer.critic_opt.state_dict(),
                actor_updates=trainer.actor_updates,world_updates=trainer.world_updates,critic_updates=trainer.critic_updates,
                native_ticks=tick,metadata=metadata,config=config),args.output/name)
        # A already supplies complete native episodes. This warmup concerns the
        # new contextual critic, not recollecting a fresh 800 ticks each update.
        phase='CRITIC';critic_loss=None
        for warmup in range(32):
            state=as_state(starts[warmup%len(starts)])
            stats=trainer.imagine_update(state,update_actor=False);critic_loss=stats['critic_loss']
            print('PHYSICAL_CRITIC='+json.dumps(stats),flush=True)
            with (args.output/'critic_history.jsonl').open('a') as f:f.write(json.dumps(stats)+'\n')
            if trainer.critic_updates>=200 and critic_loss is not None and critic_loss<=50.:break
        else:raise RuntimeError('Contextual critic quality gate did not pass within finite32-block budget')
        cursor=0;completed_returns=[];completed_discounted=[];phase='LEARNING'
        while trainer.actor_updates<256:
            if shutil.disk_usage(args.output).free<8*1024**3:raise RuntimeError('Disk guard: less than8GiB free')
            started=time.perf_counter()
            with torch.inference_mode():
                for _ in range(4):
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
            window,lanes,sampling=replay.window(timebase.world_steps,128,trainer.actor_updates)
            world_stats=trainer.fit_world_window(window,launch[lanes]);del window
            pool=replay.new_starts if replay.new_starts and trainer.actor_updates%2 else starts
            row=pool[trainer.actor_updates%len(pool)];state=as_state(row)
            stats=trainer.imagine_update(state)
            if not any(v>0 for v in stats['parameter_max_changes'].values()):
                raise RuntimeError('Actor optimizer step made no approach parameter change')
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
            write(args.output/'progress.json',record)
            with (args.output/'training_history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print('PHYSICAL_LEARNING='+json.dumps(record),flush=True)
            if trainer.actor_updates%32==0:save(f'checkpoint_{trainer.actor_updates:04d}.pt')
        archive.flush();save('candidate_0256.pt')
        write(args.output/'training_completed.json',dict(actor_updates=trainer.actor_updates,native_ticks=tick,
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
