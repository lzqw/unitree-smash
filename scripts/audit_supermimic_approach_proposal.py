"""Compare original and flight-priority updates on the first recorded native batch."""
import argparse
import copy
import json
from pathlib import Path

import torch

from quadsmash.physics.full_racket_v4 import load_racket, string_strips
from quadsmash.rl.supermimic_compact_starts import batch_approach_starts, compact_start
from quadsmash.rl.supermimic_physical_imagination_state_v1 import PhysicalImaginationStateV1
from quadsmash.rl.supermimic_physical_learner_v1 import PhysicalHorizonTrainer
from quadsmash.rl.supermimic_physical_loop_loading_v1 import load_physical_loop, tensor_digest
from quadsmash.rl.supermimic_physical_start_cache_v1 import (
    APPROACH_TICKS, CRITIC_TICKS, load_start_cache, training_start_mask,
)
from quadsmash.rl.supermimic_task_space_reference_v1 import TaskSpaceReferenceBankV1
from quadsmash.rl.supermimic_timebase_v2 import Timebase
from quadsmash.rl.supermimic_toi_racket_objective import string_boundary
from scripts.run_supermimic_approach_ball_only import registered_options
from scripts.train_supermimic_physical_block import select_launch_for_start


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cuda:0')
    args = parser.parse_args()

    torch.set_num_threads(4)
    torch.manual_seed(53012)
    run = args.checkpoint.parent
    start = torch.load(run/'pre_block.pt', map_location='cpu', mmap=True, weights_only=False)
    after = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    receipt = read(run/'evaluation_boundary.json')['invocation']
    with (run/'training_history.jsonl').open() as stream:
        history = json.loads(next(stream))
    cfg = after['config']
    original_actor = tensor_digest(start['actor'].items())
    world = tensor_digest(after['world'].items())
    if original_actor != receipt['start']['actor_sha256'] or world != history['model_runtime_state_sha256']:
        raise ValueError('The replay requires the recorded first Actor and first proposal WM')
    if world != after['approach_speed_calibration']['world_sha256']:
        raise ValueError('The recorded release calibration belongs to a different WM')
    options = registered_options(read(run/'startup.json')['argv'], (
        'archive', 'physical-checkpoint', 'ball-checkpoint', 'task-checkpoint',
        'safety-mapping', 'action-ball-checkpoint', 'g07-execution-root',
    ))
    paths = {options[i][2:].replace('-', '_'): options[i + 1] for i in range(0, len(options), 2)}
    paths['execution_root'] = paths.pop('g07_execution_root')
    actor, adapter, template, launch, manifest = load_physical_loop(**paths, device=args.device)
    arm_seconds = after['metadata']['policy_execution']['arm_preparation_seconds']
    actor.load_state_dict(start['actor'], strict=True)
    actor.configure_approach_gate(arm_seconds)
    adapter.load_state_dict(after['world'], strict=True)
    adapter.set_contact_bptt_mode('enabled')
    bank_data = torch.load(cfg['task_reference_bank'], map_location='cpu', weights_only=True)
    bank = TaskSpaceReferenceBankV1.from_payload(bank_data).to(args.device)
    held = cfg['wm_holdout_lanes']
    bank.assert_excludes(held)

    def state(row, selected=None):
        launches = select_launch_for_start(row, launch)
        if selected is not None:
            row, launches = compact_start(row, selected, launches)
        return PhysicalImaginationStateV1.from_record(
            row, executor_template=template, launch=launches, device=args.device,
            array_plan_observations=selected is not None)

    sampling = history['actor_start_sampling']
    starts = []
    for group in sampling['records']:
        labels = group['lane_episode_tick']
        candidates = [entry['start'] for entry in after['actor_start_bank']['entries']
                      if all(int(entry['start']['before']['elapsed_tick'][lane]) == tick and
                             int(entry['start']['before']['episode_index'][lane]) == episode
                             for lane, episode, tick, *_ in labels)]
        if len(candidates) != 1:
            raise ValueError('A recorded first-proposal start is missing or ambiguous')
        record = candidates[0]
        mask = torch.zeros(len(launch), dtype=torch.bool)
        mask[[label[0] for label in labels]] = True
        if not torch.equal(training_start_mask(dict(record, _actor_lane_mask=mask), held), mask):
            raise ValueError('A recorded Actor start overlaps the WM holdout')
        starts.append(state(record, mask))

    clock = Timebase(**read(Path('configs/supermimic_task_horizon_v2.json')))
    phase = after['metadata']['task_contract']['approach_ball_scoring']
    offset = (cfg['intercept_plane_calibration'] or {}).get('offset_m', 0.)
    packed = batch_approach_starts(tuple(starts), max_steps=clock.imagination_steps, dt=clock.control_dt_s,
                                   followthrough_ticks=phase['minimum_followthrough_ticks'], plane_offset_m=offset)
    if [len(row.physical) for row in packed] != sampling['packed_rollout_lane_counts']:
        raise ValueError('Reconstructed proposal uses different horizon batches')
    court = read(Path('configs/court_net_v4.json'))
    racket = Path(paths['execution_root'])/'configs/full_racket_v4_r3.json'
    budget_cache = load_start_cache(paths['archive'], manifest)
    budget_states = [state(budget_cache[tick]) for tick in APPROACH_TICKS + CRITIC_TICKS[:4]]
    trainer = PhysicalHorizonTrainer(
        actor, adapter, budget_states[0], clock, actor_lr=2e-4, reference_bank=bank,
        command_delta_budget=cfg['command_delta_budget'], target_delta_budget_rad=cfg['target_delta_budget_rad'],
        tracking_weight=cfg['tracking_weight'], approach_objective='toi_racket_only',
        toi_racket_weight=cfg['toi_racket_weight'], toi_plane_offset_m=offset)
    trainer.approach_ball_protocol = phase
    trainer.approach_speed_calibration = after['approach_speed_calibration']
    trainer.approach_separation_m = read(Path('configs/shuttle_physics_v2.json'))['separation_hysteresis_m']
    trainer.research_court = court
    trainer.landing_and_first_contact_objectives = True
    trainer.string_strips = len(string_strips(load_racket(racket)))
    trainer.string_boundary = string_boundary(racket, next(actor.parameters()))
    trainer.reference.configure_approach_gate(arm_seconds)
    trainer.reference.load_state_dict(start['actor'], strict=True)
    trainer.anchor_coeff = .01
    trainer.restart_uphill_approach_momentum = cfg['restart_uphill_approach_momentum']
    trainer.set_budget_state_library(budget_states)
    parameters = {name: value.clone() for name, value in actor.state_dict().items()}
    optimizer = copy.deepcopy(trainer.actor_opt.state_dict())
    comparisons = {}
    for mode in ('original', 'flight_priority'):
        actor.load_state_dict(parameters, strict=True)
        trainer.actor_opt.load_state_dict(copy.deepcopy(optimizer))
        trainer.accept_snapshot()
        trainer.prioritize_approach_flight = mode == 'flight_priority'
        stats = trainer.imagine_update(packed)
        if tensor_digest(adapter.state_dict().items()) != world or trainer.frozen_hash() != trainer.frozen_actor_hash:
            raise RuntimeError('A proposal changed the frozen model or policy parameters')
        comparisons[mode] = stats
        print(mode, stats['update_accepted'], stats['objective_before'], stats['objective_after'], flush=True)
    actor.load_state_dict(parameters, strict=True)
    trainer.actor_opt.load_state_dict(optimizer)
    if tensor_digest(actor.state_dict().items()) != original_actor:
        raise RuntimeError('Preflight did not restore the original Actor')
    report = dict(scope='GPU imagination on recorded native starts. No new PhysX steps or deployed checkpoint.',
                  initial_actor_sha256=original_actor, wm_sha256=world, native_start_sampling=sampling,
                  recorded_original_objective_before=history['objective_before'],
                  recorded_original_objective_after=history['objective_after'], **comparisons)
    pending = args.output.with_suffix('.pending')
    pending.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    pending.replace(args.output)


if __name__ == '__main__':
    main()
