"""Measure pre-calibration Critic errors on the same frozen episodes used for the post-fit audit."""
import argparse
import json
from pathlib import Path

import torch

from quadsmash.rl.supermimic_dual_head_quality_v1 import critic_holdout
from quadsmash.rl.supermimic_physical_loop_loading_v1 import tensor_digest
from quadsmash.rl.supermimic_v1 import MLP
from scripts.evaluate_supermimic_fresh_wm_panel import digest


def assess(frozen, checkpoint):
    receipt = json.loads((frozen / 'manifest.json').read_text())
    load = dict(mmap=True, map_location='cpu')
    bank = torch.load(frozen / 'native_critic_labels.pt', weights_only=True, **load)
    ck = torch.load(checkpoint, weights_only=False, **load)
    actor = tensor_digest(ck['actor'].items())
    execution = ck['metadata']['policy_execution']
    if not (receipt['actor_unchanged'] and receipt['complete_lanes'] == receipt['complete_episodes'] == 128
            and receipt['actor_sha256'] == bank['actor_sha256'] == actor
            and receipt['policy_id'] == bank['policy_id'] == frozen.name
            and receipt['policy_execution'] == bank['policy_execution'] == execution
            and ck['native_ticks'] == receipt['start_tick']
            and ck['global_counters']['accepted'] == bank['accepted_updates']
            and digest(frozen / 'complete_episodes.pt') == bank['source_sha256']):
        raise ValueError('Frozen labels and checkpoint differ in collection or pre-calibration policy')

    critic = MLP(bank['inputs'].shape[-1], (512, 256, 128), 1)
    critic.load_state_dict(ck['critic'], strict=True)
    critic.eval()
    rows = {}
    for head in ('approach', 'recovery'):
        before = critic_holdout(critic, bank, head, ck['config']['wm_holdout_lanes'],
                                current_actor_sha256=actor, current_execution=execution)
        after = receipt['critic_holdout'][head]
        if not (before['current_policy_matches_labels'] and after['current_policy_matches_labels']
                and after['policy_id'] == bank['policy_id'] and after['label_actor_sha256'] == actor
                and after['label_policy_execution'] == execution
                and before['sample_count'] == after['sample_count'] > 0
                and [list(v) for v in before['identity']] == after['identity']):
            raise ValueError('Frozen Critic held states, policy or execution differ between measurements')
        rows[head] = dict(before=before['metrics'], after=after['metrics'], held_states=before['sample_count'],
                          held_episodes=len({(lane, episode) for lane, episode, tick in before['identity']}))
    return dict(policy_id=bank['policy_id'], actor_sha256=actor, critic_before_sha256=tensor_digest(ck['critic'].items()),
                source_sha256=bank['source_sha256'], start_tick=receipt['start_tick'], end_tick=receipt['end_tick'],
                completed_episodes=receipt['complete_episodes'], critic_updates=receipt['critic_updates'],
                retained_updates=bank['accepted_updates'] - ck['config'].get('research_origin', 0), metrics=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('frozen', type=Path)
    parser.add_argument('checkpoint', type=Path, help='Saved at the control tick immediately preceding the frozen refresh')
    parser.add_argument('output', type=Path)
    parser.add_argument('--tensorboard', type=Path, help='Log paired errors against the retained Actor update count')
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = assess(args.frozen, args.checkpoint)
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        with SummaryWriter(str(args.tensorboard)) as writer:
            for head, values in result['metrics'].items():
                for phase in ('before', 'after'):
                    for metric, value in values[phase].items():
                        if value is not None:
                            writer.add_scalar(f'frozen_rtg/{head}/{phase}/{metric}', value, result['retained_updates'])
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temporary.replace(args.output)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
