#!/usr/bin/env python3
"""Run a bounded 32-case TOI Approach learning check with genuine 32-lane G07 state."""
import json
from functools import partial
from pathlib import Path


def make_replay(packed_path, collection, lanes, reserved, trainer, task, execution):
    import torch
    from quadsmash.rl.supermimic_action_identification_dataset_v1 import _sha256
    from quadsmash.rl.supermimic_approach_data_v1 import recorded_approach_metadata
    from quadsmash.rl.supermimic_physical_loop_replay_v1 import PhysicalLoopReplay
    from quadsmash.rl.supermimic_training_probe_replay import import_training_collection

    packed_path = Path(packed_path)
    receipt = json.loads((packed_path.parent/'completed.json').read_text())
    if receipt['first_completed'] != 128 or _sha256(packed_path) != receipt['sha256']:
        raise ValueError('Initial native panel hash or episode count mismatch')
    package = torch.load(packed_path, map_location='cpu', weights_only=True, mmap=True)
    tensors = package['tensors']
    shape = tensors['state'].shape[:2]
    if shape[1] != 128:
        raise ValueError('Expected original 128-case initialization data')
    subset = {key: value[:, lanes] for key, value in tensors.items()
              if torch.is_tensor(value) and value.shape[:2] == shape}
    schema = trainer.adapter.task_model.memory_schema
    labeler = partial(recorded_approach_metadata, schema=schema, actor=trainer.actor,
                      geometry=trainer.adapter.physical_model.geometry)
    replay = PhysicalLoopReplay({'tensors': subset}, schema, capacity_ticks=2048, approach_labeler=labeler)
    probes = import_training_collection(replay, collection, task, execution, reserved, original_lanes=lanes)
    return replay, dict(initial_sha256=receipt['sha256'], probes=probes)


def learn(args, *, budget_states, env, wrapper, observation, trainer, state, launch, lanes, bank, ck, task, protocol,forward_guard=None,full_launch=None):
    import torch
    from quadsmash.rl.supermimic_action_identification_dataset_v1 import _sha256
    from quadsmash.rl.supermimic_approach_data_v1 import native_approach_metadata
    from quadsmash.rl.supermimic_frozen_refresh_v1 import FrozenRefresh
    from quadsmash.rl.supermimic_native_metrics import summarize_native_episodes
    from quadsmash.rl.supermimic_physical_capture_v1 import PhysicalTransitionArchive
    from quadsmash.rl.supermimic_physical_imagination_state_v1 import PhysicalImaginationStateV1
    from quadsmash.rl.supermimic_physical_loop_loading_v1 import tensor_digest
    from quadsmash.rl.supermimic_physical_run_guard_v1 import atomic_torch_save
    from quadsmash.rl.supermimic_policy_execution_v1 import policy_forward
    from quadsmash.rl.supermimic_run_lifecycle import stage
    from quadsmash.rl.supermimic_terminal_scoring import score_terminal_episode
    from quadsmash.rl.supermimic_training_audit_v1 import rollback_audit, sampling_promotion_gate

    if env.num_envs != 32 or len(lanes) != env.num_envs or args.device.split(':')[0] != 'cuda':
        raise ValueError('The native canary requires exactly 32 GPU environments')
    meta = ck['metadata']
    replay, provenance = make_replay(args.packed_data, args.wm_action_collection, lanes, bank.excluded_lanes,
                                    trainer, task, meta['policy_execution'])
    trainer.world_opt.load_state_dict(ck['world_optimizer'])
    trainer.critic_opt.load_state_dict(ck['critic_optimizer'])
    cfg = ck['config']
    trainer.racket_loss_weight = float(cfg.get('racket_loss_weight', trainer.racket_loss_weight))
    trainer.ball_loss_weight = float(cfg.get('wm_ball_loss_weight', trainer.ball_loss_weight))
    trainer.contact_ball_loss_weight = float(cfg.get('wm_contact_ball_loss_weight', trainer.contact_ball_loss_weight))
    trainer.direct_long_body_supervision=cfg.get('wm_direct_long_body_supervision',False)
    trainer.natural_flight_loss=cfg.get('wm_natural_flight_loss','huber')
    natural_count=cfg.get('wm_natural_flight_batch',0)
    natural=None
    active=None
    active_count=cfg.get('wm_active_flight_batch',0)
    active_launch=full_launch
    if natural_count:
        from quadsmash.rl.supermimic_physical_loop_replay_v1 import ActiveFlightWindows,NaturalFlightWindows
        packed=torch.load(args.packed_data,map_location='cpu',weights_only=True,mmap=True)
        natural=NaturalFlightWindows(packed['tensors'],cfg['wm_holdout_lanes'])
        if ck.get('natural_flight_rng') is None or full_launch is None or len(full_launch)!=128:
            raise ValueError('Missing native anchor sampling state or launch identities')
        natural.generator.set_state(ck['natural_flight_rng'].cpu())
        if active_count:
            source=ck['runtime_validation_archive']
            if _sha256(Path(source['path']))!=source['sha256']:raise ValueError('Frozen flight archive changed')
            saved=torch.load(source['path'],map_location='cpu',weights_only=True,mmap=True)
            active=ActiveFlightWindows(saved['tensors'],cfg['wm_holdout_lanes'],natural.generator)
    if 'native' in ck.get('actor_references', {}):
        trainer.native_accepted_snapshot = ck['actor_references']['native']
    if 'approach' in ck['actor_optimizers']:
        trainer.actor_opt.load_state_dict(ck['actor_optimizers']['approach'])
    elif any(key.endswith('/approach') and block['accepted'] for key, block in ck['sequential_progress']['cycles'].items()):
        raise ValueError('Previously updated Approach head has no Adam state')
    actor = trainer.actor
    adapter = trainer.adapter
    parent_id = _sha256(args.checkpoint)
    frozen_hash = trainer.frozen_hash()
    initial_actor = tensor_digest(actor.state_dict().items())
    initial_critic = tensor_digest(trainer.critic.state_dict().items())
    initial_world = tensor_digest(adapter.state_dict().items())
    stream_version = ck['global_counters']['accepted']
    replay.new_stream(policy_version=stream_version)
    collector = None

    def as_state(row):
        return PhysicalImaginationStateV1.from_record(row, executor_template=env._sm_residual_controller,
                                                      launch=launch, device=args.device)

    class CanaryArchive(PhysicalTransitionArchive):
        def finish_record(self, *values):
            super().finish_record(*values)
            row = self.rows[-1]
            version = stream_version + trainer.actor_updates
            labels = native_approach_metadata(row, actor, adapter.physical_model.geometry)
            row.update(policy_version=version, native_tick=self.ticks-1, stream_id=replay.current_stream,
                       source_id='online', sampling_metadata=labels)
            replay.append(row, policy_version=version, native_tick=self.ticks-1,
                          approach_metadata=labels, retain_start=collector is None)
            if collector is not None:
                collector.append(row)
            if len(self.rows) >= 16:
                self.flush_compressed()

    archive = CanaryArchive(args.output/'native_transitions', metadata=dict(task_contract=task,
        policy_execution=meta['policy_execution'], original_lane_by_replay_lane=lanes, parent_sha256=parent_id),
        chunk_ticks=1_000_000, compact_plan_observations=True)
    archive.attach_task_sequence(env)
    env.physical_transition_archive = archive
    sources = cfg['wm_source_quota']
    events = cfg['wm_approach_stage_quota']
    batch_size = cfg['wm_batch_size']
    if sum(sources.values()) != batch_size or sum(events.values()) != batch_size:
        raise ValueError('Parent WM batch configuration disagrees with quota totals')

    def record(name, value):
        with (args.output/name).open('a') as out:
            out.write(json.dumps(value, allow_nan=False)+'\n')

    def collect():
        nonlocal observation
        with torch.no_grad():
            action = policy_forward(actor, observation['policy'], env.rally, sample=False)[0]
            observation, _, _, _ = wrapper.step(action)
            if not torch.isfinite(observation['policy']).all():
                raise FloatingPointError('Nonfinite canary native observation')

    def update_world():
        for _ in range(4):
            window, ids, stats = replay.window(trainer.timebase.world_steps, batch_size, trainer.world_updates, args.device,
                                                source_quota=sources, stage_quota=events, head='approach',report_loss_groups=True)
            stats['sampling_gate'] = sampling_promotion_gate(stats, 'diagnostic')
            long_native=None
            if natural is not None:
                extra,extra_lanes=natural.sample(natural_count,ck['world_updates']+trainer.world_updates,args.device)
                starts=full_launch[extra_lanes]
                if active_count:
                    flight,flight_lanes=active.sample(active_count,ck['world_updates']+trainer.world_updates,args.device)
                    extra={k:torch.cat((v,flight[k]),1) for k,v in extra.items()}
                    starts=torch.cat((starts,active_launch[flight_lanes]))
                long_native=(extra,starts)
            fit = trainer.fit_world_window(window, launch[ids], loss_groups=stats.pop('loss_groups'),
                                           separate_supervision=cfg.get('wm_separate_supervision',False),long_native=long_native)
            record('wm_batches.jsonl', dict(fit=fit, sampling=stats))

            if forward_guard is not None:
                check=forward_guard.check(adapter)
                record('online_forward_guard.jsonl',dict(check,canary_world_updates=trainer.world_updates))
                if not check['passed']:raise RuntimeError('Canary online WM failed its native forward guard')

    stage(args.output, 'CANARY_TRAINING', original_lanes=lanes, source=provenance, parent_sha256=parent_id)
    root = Path(__file__).resolve().parents[1]
    native_cursor = len(env.rally_records)
    completed_mixed = []
    try:
        trainer.set_budget_state_library([*budget_states, state])
        for attempt in range(1, 25):
            pre_metrics = summarize_native_episodes(completed_mixed)
            for _ in range(16):
                collect()
            update_world()
            for raw in env.rally_records[native_cursor:]:
                score = score_terminal_episode(raw, dict(protocol, _root=str(root)))
                completed_mixed.append(score)
                replay.actor_start_bank.label_terminal(score, replay.current_stream)
            native_cursor = len(env.rally_records)
            checkpoint = args.output/f'pre_proposal_{attempt:02d}.pt'
            atomic_torch_save(dict(actor=actor.state_dict(), optimizer=trainer.actor_opt.state_dict(),
                                   policy_version=stream_version+trainer.actor_updates), checkpoint)
            row, selection = replay.actor_start_bank.sample(attempt)
            trainer.block_attempts = attempt
            stats = trainer.imagine_update(state if row is None else as_state(row))
            if not stats['update_accepted'] or stats['backtrack_alpha'] < 1:
                stats['rollback'] = rollback_audit(phase='approach', update_index=trainer.actor_updates + int(not stats['update_accepted']),
                    proposal_checkpoint=checkpoint, stats=stats, pre_metrics=pre_metrics,
                    post_metrics=summarize_native_episodes(completed_mixed), replay_stream=replay.current_stream)
            record('proposals.jsonl', dict(selection=selection, stats=stats, native_ticks=archive.ticks))
            if stats['update_accepted'] and (stats['tracking_gradient_norm'] <= 0 or stats['final_command_max_change'] <= 0):
                raise RuntimeError('Accepted update lacks TOI tracking or executable action response')
            if trainer.actor_updates == args.canary_updates:
                break
        if trainer.actor_updates != args.canary_updates or trainer.frozen_hash() != frozen_hash:
            raise RuntimeError('Approach update budget or frozen-module check failed')
        version = stream_version + trainer.actor_updates
        actor_id = tensor_digest(actor.state_dict().items())
        if actor_id == initial_actor:
            raise RuntimeError('Unchanged Approach after accepted proposals')
        replay.new_stream(policy_version=version)

        def encode(row):
            return adapter.critic_input(as_state(row)).detach()

        collector = FrozenRefresh(32, 800, trainer.timebase.gamma, actor_id, replay.schema,
                                  stage_ticks=(0, 240, 256, 272, 400, 576, 768), start_encoder=encode)
        # Begin an explicitly new frozen episode on every lane. Mixed-policy returns end at this boundary.
        cursor = len(env.rally_records)
        observation, _ = wrapper.reset()
        for _ in range(800):
            collect()
            if collector.complete.all():
                break
        rtg, receipt = collector.result()
        identities = set(enumerate(map(int, collector.ids.tolist())))
        records = [score_terminal_episode(row, dict(protocol, _root=str(root))) for row in env.rally_records[cursor:]
                   if (int(row['env_id']), int(row['episode_index'])) in identities]
        if (len(records) != 32 or {(int(row['env_id']), int(row['episode_index'])) for row in records} != identities):
            raise RuntimeError('Frozen native scoring lost or repeated an episode')
        if any(row['case_id'] != env.cases[int(row['env_id'])]['case_id'] for row in records):
            raise RuntimeError('Frozen episode was scored against another native case')
        if active_count:
            if set(lanes)&set(cfg['wm_holdout_lanes']):raise ValueError('Canary includes a reserved flight training lane')
            active=ActiveFlightWindows(collector.packed_complete(),[],natural.generator)
            active_launch=launch
        update_world()
        xs, ys = [], []
        for tick, row, mask in collector.calibration_rows():
            xs.append(row['critic_input'][mask]); ys.append(rtg[tick, mask])
        fit = trainer.calibrate_critic_tensors(torch.cat(xs), torch.cat(ys), epochs=2)
        final_critic = tensor_digest(trainer.critic.state_dict().items())
        final_world = tensor_digest(adapter.state_dict().items())
        if (tensor_digest(actor.state_dict().items()) != actor_id or final_critic == initial_critic
                or final_world == initial_world or trainer.frozen_hash() != frozen_hash):
            raise RuntimeError('Frozen Actor or WM/Critic update check failed')
        output = dict(schema='bounded-native-toi-canary-v1', parent_sha256=parent_id,
                      wm_replay_handoff=ck.get('wm_replay_handoff'),
                      actor=actor.state_dict(), critic=trainer.critic.state_dict(), world=adapter.state_dict(),
                      approach_optimizer=trainer.actor_opt.state_dict(), world_optimizer=trainer.world_opt.state_dict(),
                      critic_optimizer=trainer.critic_opt.state_dict(), original_lanes=lanes, promotable=False)
        atomic_torch_save(output, args.output/'canary.pt')
        for row in records:
            record('frozen_scores.jsonl', dict(original_lane=lanes[int(row['env_id'])], native=row))
        result = dict(num_envs=32, original_lanes=lanes, scope='bounded diagnostic on training cases', promotable=False,
                      actor_updates=trainer.actor_updates, world_updates=trainer.world_updates, critic_updates=trainer.critic_updates,
                      contact_gradients='detached', parent_sha256=parent_id, actor_before=initial_actor, actor_after=actor_id,
                      critic_after=final_critic, world_after=final_world, native_transitions=archive.transitions,
                      frozen=receipt, native_scores=summarize_native_episodes(records), rtg_fit=fit,
                      wm_replay_handoff=ck.get('wm_replay_handoff'), world_before=initial_world,
                      candidate_sha256=_sha256(args.output/'canary.pt'))
    finally:
        env.physical_transition_archive = None
        archive.flush_compressed()
    (args.output/'canary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    from quadsmash.rl.supermimic_run_lifecycle import native_process_main
    from audit_supermimic_toi_native_shadow import main
    native_process_main(lambda: main(canary=True))
