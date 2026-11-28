"""Check pre-fall Recovery head changes through the same measured G07 history."""
import hashlib
import json

import torch

from quadsmash.control.g07_shared_target_residual_v1 import make_g07_residual_controller
from quadsmash.rl.supermimic_compact_starts import compact_start
from quadsmash.rl.supermimic_g07_memory_preview_v1 import G07MemoryPreviewV1
from quadsmash.rl.supermimic_physical_capture_v1 import restore_archive_execution
from quadsmash.rl.supermimic_physical_loop_loading_v1 import freeze_execution_modules
from quadsmash.rl.supermimic_physical_observation_v1 import JOINT_NAMES
from quadsmash.rl.supermimic_v1 import load_ppo_initialized_actor
from scripts.dynamimic_recovery_validation import policies
from scripts.fit_dynamimic_recovery_wm import read_data


@torch.inference_mode()
def audit(native, checkpoint, companion, candidates):
    data, manifest = read_data(native, ('active','obs','request','release_time'))
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    if digest(checkpoint) != manifest['checkpoint_sha256'] or digest(companion) != manifest['companion_sha256']:
        raise ValueError('Training archive policy or companion mismatch')
    meta = torch.load(checkpoint, mmap=True, weights_only=False, map_location='cpu')['metadata']['init_policy_checkpoint']
    source = meta['path']
    control = make_g07_residual_controller(torch.load(source, weights_only=False, map_location='cpu'),
                                           len(manifest['cases']), 'cpu', meta['g07_execution_root'])
    freeze_execution_modules(control)
    preview = G07MemoryPreviewV1(control)
    limits = torch.tensor(json.loads((native/'actuator_contract.json').read_text())['profiles'][0]['position_limits'])
    n = len(data['live'])
    fault = torch.where(data['active'] & data['body'], torch.arange(n)[:,None], n).amin(0)
    tick = fault - 10
    lane = torch.where((fault<n) & (tick>=0) & data['active'][tick.clamp_min(0), torch.arange(len(tick))])[0]
    tick = tick[lane]
    report = dict(source=digest(checkpoint), cases=len(lane), candidates=[])
    if not len(lane):
        return report
    for candidate in candidates:
        bank = policies(checkpoint,candidate,companion,lambda:load_ppo_initialized_actor(source,'cpu')[0])
        raw = torch.empty(2,len(lane),18)
        for j, policy in enumerate(bank):
            sel = lane%2 == j%2
            if not sel.any():
                continue
            t,l = tick[sel],lane[sel]
            raw[j//2,sel] = policy._gated_distribution(data['obs'][t,l], data['gate'][t,l],
                                                       release_time=data['release_time'][t,l]).loc
        requests = raw.clamp(-1.,1.)
        torch.testing.assert_close(requests[0], data['request'][tick,lane], rtol=0, atol=2e-6)
        output = torch.empty_like(raw)
        valid = torch.zeros(len(lane),dtype=torch.bool)
        for t in tick.unique().tolist():
            start = t//16*16
            path = native/'trajectories'/f'g07_{start:04d}.pt'
            if not path.exists():
                continue
            selection = tick == t
            ids = lane[selection]
            mask = torch.zeros(len(fault),dtype=torch.bool)
            mask[ids] = True
            saved = torch.load(path, mmap=True, weights_only=False, map_location='cpu')
            row,_ = compact_start(dict(before={},observation=data['obs'][start],executor_before=saved),
                                  mask, torch.zeros(len(mask),6))
            memory = restore_archive_execution(control,row['executor_before'],device='cpu',array_plan_observations=True)
            default = data['before'][start,ids[0],13:31] - data['obs'][start,ids[0],9:27]
            for step in range(start,t):
                _,memory,_ = preview.forward(data['obs'][step,ids],data['request'][step,ids],default,limits,memory)
            for j in range(2):
                pred,_,_ = preview.forward(data['obs'][t,ids],requests[j,selection],default,limits,memory)
                output[j,selection] = pred.final_joint_targets
            valid[selection] = True
        errors = (output[0,valid] - data['final_joint_targets'][tick[valid],lane[valid]]).abs().amax(1)
        checked = valid.clone()
        checked[valid] = errors < 1e-4
        result = dict(checkpoint=digest(candidate), available_g07=int(valid.sum()), verified_g07=int(checked.sum()),
                      max_target_replay_error_rad=float(errors.max()) if len(errors) else None, joints={})
        for k,name in enumerate(JOINT_NAMES[:12]):
            before,after = raw[:,checked,k]
            delta = output[1,checked,k] - output[0,checked,k]
            clipped = before.abs() > 1
            cleared = clipped & (after.abs()<1)
            result['joints'][name] = dict(previously_clipped=int(clipped.sum()), cleared=int(cleared.sum()),
                                          newly_clipped=int(((before.abs()<=1)&(after.abs()>1)).sum()),
                                          cleared_with_target_change=int((cleared & (delta.abs()>.001)).sum()),
                                          mean_target_delta_previously_clipped_rad=float(delta[clipped].mean()) if clipped.any() else None)
        report['candidates'].append(result)
    return report


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    from scripts.run_supermimic_recovery_baseline import dump

    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('native','checkpoint','companion','output'):
        parser.add_argument('--'+field, required=True, type=Path)
    parser.add_argument('--candidates', nargs='+', required=True, type=Path)
    args = parser.parse_args()
    dump(args.output, audit(args.native, args.checkpoint, args.companion, args.candidates))
