"""Continue a completed research arm to a cumulative endpoint, then score its frozen policy on the independent 512."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from run_supermimic_research_scale import digest, run_registered_arm


def extension_registration(parent, output, tensorboard, total):
    def read(name):
        return json.loads((parent/name).read_text())

    reg = read('registration.json')
    completed = read('train/training_completed.json')
    previous = reg['expected_retained_updates']
    if (previous, total) not in ((256, 1024), (1024, 2048)):
        raise ValueError('Expected consecutive cumulative research endpoints')
    if (read('exit.json')['returncode'] != 0 or completed['actor_updates'] != previous or
            any(completed[h+'_accepted'] != previous//2 for h in ('recovery', 'approach'))):
        raise ValueError('Parent stage did not finish its retained dual-head budget')
    checkpoint = parent/'train/latest.pt'
    ck = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    schedule = ck['alternating_schedule']
    if (ck['global_counters']['accepted'] != completed['global_accepted'] or
            ck['global_counters']['accepted'] - ck['config']['research_origin'] != previous or
            schedule['running'] or schedule['cursor'] != len(schedule['plan']) or
            schedule['results'][-1]['event']['kind'] != 'finalize' or
            ck['world_updates'] != completed['world_updates'] or ck['critic_updates'] != completed['critic_updates']):
        raise ValueError('Expected the completed post-refresh training checkpoint')
    cmd = list(reg['command'])
    initial = reg.get('initial_checkpoint', cmd[cmd.index('--resume-training')+1])
    if '--resume-reset-schedule' in cmd:
        cmd.remove('--resume-reset-schedule')
    if '--continuous-approach-starts' not in cmd:
        cmd.append('--continuous-approach-starts')
    if '--diagnostic-critic-loss' not in cmd:
        cmd.append('--diagnostic-critic-loss')
    if '--wm-latest-policy-windows' not in cmd:
        cmd.extend(['--wm-latest-policy-windows', '6'])
    if '--wm-stage-quota' in cmd:
        index = cmd.index('--wm-stage-quota') + 1
        quota = dict(item.split('=') for item in cmd[index].split(','))
        if 'post_landing_unload' not in quota and int(quota.get('post_landing_support_loss', 0)) >= 4:
            quota['post_landing_support_loss'] = str(int(quota['post_landing_support_loss']) - 4)
            quota['post_landing_unload'] = '4'
            cmd[index] = ','.join(f'{key}={value}' for key, value in quota.items())
    if '--shuffle-wm-quota-groups' not in cmd:
        cmd.append('--shuffle-wm-quota-groups')
    if '--recovery-prefix-and-string-margin' not in cmd:
        cmd.append('--recovery-prefix-and-string-margin')
    if '--extend-completed-research' not in cmd:
        cmd.append('--extend-completed-research')
    for flag, value in (('--cycles', total//256), ('--resume-training', checkpoint),
                        ('--output', output/'train'), ('--tensorboard-dir', tensorboard/reg['arm'])):
        cmd[cmd.index(flag)+1] = str(value)
    from report_supermimic_native_contact_counts import NativeContactCounts
    # The asynchronous monitor may have exited just before the final archive flush.
    NativeContactCounts(parent).update()
    contacts = read('native_contact_counts.json')
    return dict(reg, command=cmd, initial_checkpoint=str(initial), expected_retained_updates=total,
                max_proposals=2*total, previous_arm=str(parent), previous_checkpoint_sha256=digest(checkpoint),
                previous_heads={h:completed[h+'_accepted'] for h in ('recovery','approach')},
                previous_proposals=sum(x['attempts'] for x in completed['head_counters'].values()),
                previous_native_contacts=dict(env_steps=contacts['durable_env_steps'], counts=contacts['total']),
                learner_rng_at_start='resumed from final checkpoint after frozen refresh')


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for k in ('parent','output','tensorboard'):
        cli.add_argument('--'+k, type=Path, required=True)
    cli.add_argument('--total-retained', type=int, choices=(1024,2048), required=True)
    args = cli.parse_args()
    output = args.output.resolve()
    tensorboard = args.tensorboard.resolve()
    reg = extension_registration(args.parent.resolve(), output, tensorboard, args.total_retained)
    rc = run_registered_arm(output, tensorboard, reg)
    if rc:
        return rc
    cmd = reg['command']
    completed = json.loads((output/'train/training_completed.json').read_text())
    if completed['actor_updates'] != args.total_retained:
        raise RuntimeError('Missing cumulative retained updates at stage end')
    repo = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable,'scripts/evaluate_supermimic_independent512.py',
                    '--checkpoint',str(output/'train/latest.pt'),'--output',str(output/'independent512'),
                    '--g07-execution-root',cmd[cmd.index('--g07-execution-root')+1]], cwd=repo, check=True,
                   env=os.environ | {'PYTHONPATH': '.:scripts'})
    from report_supermimic_research_curve import arm_origin
    from torch.utils.tensorboard import SummaryWriter
    scores = json.loads((output/'independent512/independent512.json').read_text())
    final = torch.load(output/'train/latest.pt', map_location='cpu', mmap=True, weights_only=False)
    origin, native_origin = arm_origin(reg)
    axes = dict(retained_updates=final['global_counters']['accepted']-origin,
                new_native_env_steps=(final['native_ticks']-native_origin)*128,
                training_wall_seconds=int(final['research_elapsed_s']))
    with SummaryWriter(str(tensorboard/(reg['arm']+'_evaluation'))) as writer:
        for metric, value in scores['metrics'].items():
            if isinstance(value, (float, int)):
                for axis, step in axes.items():
                    writer.add_scalar(f'independent512/{axis}/{metric}', value, step)
    return 0


if __name__ == '__main__':
    sys.exit(main())
