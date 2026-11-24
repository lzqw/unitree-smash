"""Audit terminal labels, signed Body support and closed-loop native outcomes."""
import argparse
import json
from pathlib import Path

import torch

from quadsmash.rl.supermimic_action_identification_dataset_v1 import _load_branch
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2
from quadsmash.rl.supermimic_policy_execution_v1 import contract
from quadsmash.rl.supermimic_precontact_body import body_pair_loss
from quadsmash.rl.supermimic_release_calibration import _native_event_columns
from quadsmash.rl.supermimic_task_memory_schema_v1 import MemoryField, TaskMemorySchemaV1
from quadsmash.rl.supermimic_training_probe_replay import training_windows, verified_terminal_records


def outcome(record):
    return {key: record[key] for key in ('case_id', 'outgoing_speed_mps',
            'ball_side_complete_except_recovery', 'rejection_reasons')}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('collection', 'checkpoint', 'output'):
        cli.add_argument('--' + name, type=Path, required=True)
    cli.add_argument('--validation-lane', action='append', type=int, required=True)
    args = cli.parse_args()
    torch.set_num_threads(3)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    actual_task = checkpoint['metadata']['task_contract']
    source_task = {k: v for k, v in actual_task.items() if k != 'approach_ball_scoring'}
    pairs, receipt = training_windows(args.collection, source_task, contract(),
                                       checkpoint['config']['wm_holdout_lanes'])
    if not receipt['terminal_labeled_collection']:
        raise ValueError('Expected new terminal-bound signed native data')
    body = PhysicalWorldV2(actual_task['asset']['sha256']).eval()
    state = {k.removeprefix('physical_model.'): v for k, v in checkpoint['world'].items()
             if k.startswith('physical_model.')}
    body.load_state_dict(state, strict=True)
    cases = json.loads(Path(actual_task['suite']['path']).read_text())['cases']
    launch = torch.tensor([row['launch']['position_m'] + row['launch']['velocity_mps'] for row in cases])
    result = {}
    split = set(args.validation_lane)
    for name, (lanes, signed) in zip(receipt['included_lanes'], pairs, strict=True):
        spec = args.collection / name
        branches = {key: _load_branch(spec, key) for key in receipt['spec_branches'][name]}
        schema = branches['plus']['raw']['wm_window']['schema']
        columns = _native_event_columns(TaskMemorySchemaV1([MemoryField(**f) for f in schema['fields']]))
        records = {key: verified_terminal_records(branch) for key, branch in branches.items()}
        base, repeat = records['nominal_a'], records['nominal_b']
        if set(base) != set(repeat) or any(outcome(row) != outcome(repeat[lane]) for lane, row in base.items()):
            raise ValueError('Nominal native terminal repeat did not match')
        comparisons = {}
        for name_group, subset in [('fit', set(lanes) - split), ('held_interventions', split)]:
            indices = [i for i, lane in enumerate(lanes) if lane in subset]
            if not indices:
                raise ValueError(f'No native lanes for {name_group}')
            with torch.no_grad():
                loss, support, diagnostics = body_pair_loss(body, *(
                    {k: v[:, indices] for k, v in signed[key].items()} for key in ('minus', 'plus')),
                    launch[[lanes[i] for i in indices]], columns)
            comparisons[name_group] = dict(lanes=[lanes[i] for i in indices], support=support,
                                           response_by_horizon=diagnostics,
                                           loss=float(loss) if loss is not None else None)
        selected = records.get('candidate', {})
        compared = {lane: dict(baseline=outcome(row), candidate=outcome(selected[lane]))
                    for lane, row in base.items() if lane in selected}
        result[name] = dict(body_response=comparisons, terminal_count={k: len(v) for k, v in records.items()},
                            native_closed_loop=compared, common_native_cases=len(compared),
                            candidate_success=sum(row['candidate']['ball_side_complete_except_recovery']
                                                  for row in compared.values()),
                            baseline_success=sum(row['baseline']['ball_side_complete_except_recovery']
                                                 for row in compared.values()))
    for name, report in result.items():
        timing = {}
        for branch in receipt['spec_branches'][name]:
            file = args.collection / name / branch / 'shutdown_diagnostic.json'
            completed = json.loads(file.read_text())
            timing[branch] = dict(control_ticks=completed['control_ticks'],
                                  native_loop_wall_seconds=completed['elapsed_s'])
        report['native_timing_by_branch'] = timing
    args.output.write_text(json.dumps(dict(receipt=receipt, groups=result), indent=2, allow_nan=False)+'\n')
    for report in result.values():
        fits = report['body_response']['fit']['response_by_horizon']
        for horizon in (20, 32):
            supported = fits.get(horizon, {})
            if not sum(row['normal_response_measurable'] for row in supported.values()):
                raise ValueError(f'No measured normal-velocity responses in the H{horizon} fitting data')
            if not supported.get('marginal_success', {}).get('pairs'):
                raise ValueError(f'No terminal-labeled marginal success pairs at H{horizon}')


if __name__ == '__main__':
    main()
