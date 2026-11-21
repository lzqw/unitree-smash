"""Attribute a frozen refresh's change in TOI direction to its world model and success references."""
import argparse
import json
from pathlib import Path

import torch

from quadsmash.rl.supermimic_native_pair_capture_v1 import tree_digest
from quadsmash.rl.supermimic_physical_loop_loading_v1 import load_physical_loop, tensor_digest
from quadsmash.rl.supermimic_task_space_reference_v1 import TaskSpaceReferenceBankV1
from scripts.audit_supermimic_paired_toi_objective import score
from scripts.evaluate_supermimic_fresh_wm_panel import digest


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('run', 'before-refresh', 'frozen-episodes', 'development', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run = args.run
    argv = read(run / 'startup.json')['argv']
    inv = read(run / 'training_completed.json')['invocation']
    manifest = read(args.frozen_episodes / 'manifest.json')
    if manifest['complete_lanes'] != 128 or not manifest['actor_unchanged'] or inv['actor_end_sha256'] != manifest['actor_sha256']:
        raise ValueError('Need 128 complete episodes from the frozen closing actor')
    checks = [json.loads(line) for line in (run / 'online_forward_guard.jsonl').read_text().splitlines()]
    stages = {}
    identities = {}
    for name, path in [('before', args.before_refresh), ('after', run / 'latest.pt')]:
        ck = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        world_id = tensor_digest(ck['world'].items())
        if (tensor_digest(ck['actor'].items()) != manifest['actor_sha256']
                or not any(r['passed'] and r['world_updates'] == ck['world_updates']
                           and r['world_sha256'] == world_id for r in checks)):
            raise ValueError('Checkpoint differs from the frozen actor or validated WM')
        identities[name] = dict(checkpoint_sha256=digest(path), world_sha256=world_id, world_updates=ck['world_updates'],
                                reference_sha256=tree_digest(ck['reference_bank']))
        stages[name] = ck
    before, after = stages['before'], stages['after']
    held = set(after['config']['wm_holdout_lanes'])
    meta = after['metadata']['physical_imagination']
    if (before['metadata']['physical_imagination'] != meta
            or before['metadata']['policy_execution'] != after['metadata']['policy_execution']
            or held != set(before['config']['wm_holdout_lanes'])
            or after['world_updates'] - before['world_updates'] != manifest['world_updates']
            or identities['after']['reference_sha256'] != tree_digest(torch.load(
                args.frozen_episodes / 'reference_bank.pt', weights_only=True, map_location='cpu'))):
        raise ValueError('Incompatible sources, holdout, refresh interval or reference bank')
    paths = meta['checkpoint_paths']
    _, adapter, template, launch, contract = load_physical_loop(
        archive=Path(argv[argv.index('--archive') + 1]), safety_mapping=Path(argv[argv.index('--safety-mapping') + 1]),
        physical_checkpoint=paths['physical'], ball_checkpoint=paths['ball'], task_checkpoint=paths['task'],
        action_ball_checkpoint=paths.get('action_ball'), execution_root=meta['g07_execution_root'], device='cpu')
    if meta['checkpoint_sha256'] != adapter.manifest['checkpoint_sha256']:
        raise ValueError('Source checkpoint identity differs')
    adapter.eval().requires_grad_(False)
    development = read(args.development / 'completed.json')
    results = {}
    for world_name, world_stage in stages.items():
        adapter.load_state_dict(world_stage['world'], strict=True)
        results[world_name] = {}
        for bank_name, bank_stage in stages.items():
            bank = TaskSpaceReferenceBankV1.from_payload(bank_stage['reference_bank'])
            bank.assert_excludes(held)
            results[world_name][bank_name] = {
                scale: score(args.development / scale, development[scale], adapter, template, launch, bank,
                             contract['task_contract'], after['metadata']['policy_execution'], held)
                for scale in ('wide', 'half', 'quarter')}
    result = dict(actor_sha256=manifest['actor_sha256'], stages=identities, development=development,
                  world_then_reference=results, actor_updates=0, contact_receipt=False,
                  scope='Historical native commands at three scales on the same development launches, with a fixed actor.')
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
