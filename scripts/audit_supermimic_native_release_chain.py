"""Evaluate a frozen WM on native releases from its excluded training lanes."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from quadsmash.physics.full_racket_v4 import load_racket, string_strips
from quadsmash.rl.supermimic_archive_storage_v1 import read_archive_rows
from quadsmash.rl.supermimic_physical_loop_loading_v1 import load_physical_loop, tensor_digest
from quadsmash.rl.supermimic_physical_loop_replay_v1 import row_tensors
from quadsmash.rl.supermimic_release_calibration import calibrate_release_speed
from run_supermimic_approach_ball_only import registered_options


FIELDS = ('state', 'next_state', 'values', 'next_values', 'next_valid', 'episode',
          'targets', 'previous_action', 'feet', 'present', 'elapsed', 'terminated', 'timeout')


class MeasuredReplay:
    """Read-only ordered inputs for the same calibration used during training."""
    def __init__(self, directory, scratch, manifest, schema):
        self.schema = schema
        data = {name: [] for name in FIELDS}
        streams, ticks = [], []
        for receipt in manifest['chunks']:
            with read_archive_rows(directory, receipt, scratch=scratch) as rows:
                for row in rows:
                    sample = row_tensors(row, schema)
                    for name in FIELDS:
                        data[name].append(sample[name])
                    streams.append(row['stream_id'])
                    ticks.append(row['native_tick'])
        if len(ticks) != manifest['control_ticks'] or not ticks:
            raise ValueError('Incomplete release audit source')
        data = {name: torch.stack(samples) for name, samples in data.items()}
        sources = torch.ones_like(data['episode'])
        self.inputs = (data, torch.tensor(streams), None, sources, None, torch.tensor(ticks))

    def _ordered(self):
        return self.inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, action='append', required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-cases', type=int, default=32)
    args = parser.parse_args()
    if args.max_cases < 1:
        parser.error('Maximum cases must be positive')
    torch.set_num_threads(2)
    command = json.loads(args.registration.read_text())['command']
    options = registered_options(command, ('archive', 'physical-checkpoint', 'ball-checkpoint',
                                           'task-checkpoint', 'safety-mapping', 'action-ball-checkpoint',
                                           'g07-execution-root'))
    inputs = {flag[2:].replace('-', '_'): value for flag, value in zip(options[::2], options[1::2], strict=True)}
    execution = Path(inputs.pop('g07_execution_root'))
    _, model, _, launch, source = load_physical_loop(**inputs, execution_root=execution, device='cpu')
    model.eval().requires_grad_(False)
    manifest_bytes = (args.archive/'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    task = manifest['task_contract']
    if (manifest['pending_transition'] or 'approach_ball_scoring' not in task or len(launch) != manifest['num_envs']
            or {k: v for k, v in task.items() if k != 'approach_ball_scoring'} != source['task_contract']):
        raise ValueError('Expected a completed native archive on the registered physical task')
    replay = MeasuredReplay(args.archive, args.scratch, manifest, model.task_model.memory_schema)
    racket_config = execution/'configs/full_racket_v4_r3.json'
    strip_count = len(string_strips(load_racket(racket_config)))
    separation = json.loads((execution/'configs/shuttle_physics_v2.json').read_text())['separation_hysteresis_m']
    court = json.loads((execution/'configs/court_net_v4.json').read_text())
    holdout = None
    audits = []
    for path in args.checkpoint:
        checkpoint = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        excluded = set(checkpoint['config']['wm_holdout_lanes'])
        if holdout is None:
            holdout = excluded
        model.load_state_dict(checkpoint['world'], strict=True)
        world = tensor_digest(model.state_dict().items())
        calibration = checkpoint['approach_speed_calibration']
        if (excluded != holdout or not excluded or checkpoint['metadata']['task_contract'] != task
                or checkpoint['metadata']['policy_execution'] != manifest['policy_execution']
                or world != calibration['world_sha256'] or set(calibration['excluded_lanes']) != excluded):
            raise ValueError(f'Incompatible WM, calibration or held-out lanes for {path}')
        measured = calibrate_release_speed(
            model, replay, launch, excluded_lanes=sorted(set(range(len(launch)))-holdout),
            string_strips=strip_count, separation_m=separation,
            threshold=task['approach_ball_scoring']['minimum_outgoing_speed_mps'],
            max_cases=args.max_cases, training_calibration=calibration, flight_court=court,
        )
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        audits.append(dict(checkpoint=str(path), checkpoint_sha256=digest, world_sha256=world, **measured))
        del checkpoint
    result = dict(
        schema='supermimic-native-release-chain-audit-v1', archive=str(args.archive),
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(), holdout_lanes=sorted(holdout),
        scope='Excluded native WM lanes. Three-layer metrics share complete measured landings within each model audit.',
        audits=audits,
    )
    staged = args.output.with_suffix('.tmp')
    staged.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    staged.replace(args.output)


if __name__ == '__main__':
    main()
