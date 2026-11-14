"""Read-only CPU audit of one saved checkpoint/replay; no environment or updates."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from quadsmash.rl.supermimic_v1 import SuperMimicWorldModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(20261114)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    replay = saved['replay']
    data = replay['data']
    world = SuperMimicWorldModel(794, 18, event_dim=replay['event_dim']).eval()
    world.load_state_dict(saved['world_model'], strict=True)
    count = min(replay['ticks_written'], replay['capacity'] // replay['num_envs'])
    assert replay['ticks_written'] == count, 'This diagnostic requires an unwrapped replay'
    starts = torch.randint(0, count - 128 + 1, (512,))
    lanes = torch.randint(0, replay['num_envs'], (512,))
    rows = starts[:, None] + torch.arange(128)
    lane_grid = lanes[:, None].expand_as(rows)
    valid = (replay['episode'][rows, lane_grid] == replay['episode'][starts, lanes, None]).all(1)
    valid &= ~(data['done'][rows[:, :-1], lane_grid[:, :-1], 0] > .5).any(1)
    starts, lanes = starts[valid][:64], lanes[valid][:64]
    assert len(starts) > 0
    pred = data['obs'][starts, lanes].clone()
    event = data['event_state'][starts, lanes].clone()
    reward_pred, reward_true, steps = [], [], []
    with torch.no_grad():
        for k in range(128):
            row = starts + k
            delta, reward, done, logits = world(pred, data['action'][row, lanes], event)
            pred = pred + delta
            # Match the imagination learner's deterministic clock override.
            pred[:, 775] = data['elapsed'][starts, lanes, 0] + (k + 1) * .01
            event = (event + (1 - event) * logits.sigmoid()).clamp(0, 1)
            truth = data['next_obs'][row, lanes]
            reward_pred.append(reward[:, 0])
            reward_true.append(data['reward'][row, lanes, 0])
            if k + 1 in (1, 16, 64, 128):
                phase = pred[:, 770:775]
                steps.append(dict(step=k+1, seconds=(k+1)*.01,
                    obs_mse=float((pred-truth).square().mean()),
                    joint_position_mae_rad=float((pred[:,9:27]-truth[:,9:27]).abs().mean()),
                    ball_relative_position_mae_m=float((pred[:,61:64]-truth[:,61:64]).abs().mean()),
                    gravity_norm_error=float((pred[:,6:9].norm(dim=1)-1).abs().mean()),
                    phase_sum_error=float((phase.sum(1)-1).abs().mean()),
                    phase_outside_0_1_fraction=float(((phase<0)|(phase>1)).float().mean()),
                    final_command_tail_mae=float((pred[:,776:794]-truth[:,776:794]).abs().mean()),
                    event_latch_mae=float((event-data['next_event_state'][row,lanes]).abs().mean())))
    rp, rt = torch.cat(reward_pred), torch.cat(reward_true)
    result = dict(checkpoint=str(args.checkpoint),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        counters=saved['counters'], timebase=saved['timebase'],
        source=saved['metadata'].get('init_policy_checkpoint'),
        replay_transitions=count*replay['num_envs'], replay_fields=list(data),
        windows=len(starts), seed=20261114, horizons=steps,
        reward_mae=float((rp-rt).abs().mean()),
        reward_correlation=float(torch.corrcoef(torch.stack((rp,rt)))[0,1]),
        replay_reward_positive_fraction=float((data['reward']>0).float().mean()),
        scope='In-replay recorded-action open-loop diagnostic; NOT held-out validation, policy rollout, formal evaluation, or causal performance attribution')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
