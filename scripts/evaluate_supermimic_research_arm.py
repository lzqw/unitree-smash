"""Score saved research milestones on the original fixed panel and log paired outcomes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
from torch.utils.tensorboard import SummaryWriter

from compare_supermimic_fixed_stages import load_run
from quadsmash.rl.supermimic_physical_loop_loading_v1 import tensor_digest
from run_supermimic_research_scale import digest
from report_supermimic_research_curve import arm_origin, axes


def evaluate(folder, step, writer):
    reg=json.loads((folder/'registration.json').read_text())
    cmd=reg['command']
    baseline=Path(reg['baseline_path']).parent
    _,identity,old=load_run(baseline)
    if reg['initial']['actor_sha256']!=old:
        raise ValueError('Zero-update panel belongs to a different actor')
    train=folder/'train'
    checkpoint=train/f'research_{step:04d}.pt'
    ck=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    retained=ck['global_counters']['accepted']-ck['config']['research_origin']
    actor=tensor_digest(ck['actor'].items())
    if retained!=step or ck['config']['training_profile']!='research_scale':
        raise ValueError('Milestone differs from actual retained optimizer count')
    del ck
    folder_eval=folder/f'fixed128_{step:04d}'
    def run(args,log):
        with (folder/log).open('a') as stream:
            subprocess.run([sys.executable,*map(str,args)],cwd=Path(__file__).resolve().parents[1],
                           env=dict(os.environ,PYTHONPATH='.:scripts'),stdout=stream,stderr=subprocess.STDOUT,check=True)
    if not (folder_eval/'completed.json').exists():
        run(['scripts/evaluate_supermimic_v1.py','--checkpoint',checkpoint,'--output',folder_eval,
             '--g07-execution-root',cmd[cmd.index('--g07-execution-root')+1],'--seed',identity['seed'],
             '--physics-device','cuda:0','--device','cuda:0','--headless','--kit_args=--renderer/multiGpu/enabled=false'],
            f'eval_{step}.log')
    metrics=json.loads((folder_eval/'metrics.json').read_text())
    if metrics['actor_state_sha256']!=actor or metrics['evaluated_checkpoint_sha256']!=digest(checkpoint):
        raise ValueError('Native panel did not score this milestone')
    paired=folder/f'paired_{step:04d}.json'
    if not paired.exists():
        run(['scripts/compare_supermimic_fixed_stages.py','--before',baseline,'--after',folder_eval,'--stage',reg['arm']+str(step),
             '--expected-before',old,'--expected-after',actor,'--output',paired],f'paired_{step}.log')
    result=json.loads(paired.read_text())
    if not result['identical_initial_states'] or [r['actor_sha256'] for r in result['native_sources']]!=[old,actor]:
        raise ValueError('Paired panel has the wrong initial states or policy hashes')
    trace=[json.loads(line) for line in (train/'training_history.jsonl').read_text().splitlines()]
    origin = arm_origin(reg)
    row = next(r for r in trace if r['update_accepted'] and r['accepted_id'] == origin[0] + step)
    positions = axes(row, origin, metrics['num_envs'])
    for k in ('qualified_airborne_string_contact','ball_side_complete_except_recovery','complete5',
              'conditional_recovery_count','recovery_given_airborne_string_contact','native_episode_return_mean'):
        if metrics.get(k) is not None:
            for axis,pos in positions.items():writer.add_scalar('fixed128/'+axis+'/'+k,metrics[k],pos)
    writer.flush()


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arm-dir',type=Path,required=True)
    ap.add_argument('--tensorboard',type=Path,required=True)
    ap.add_argument('--steps',nargs='+',type=int,default=[128,256])
    args=ap.parse_args()
    with SummaryWriter(str(args.tensorboard)) as w:
        for step in args.steps:evaluate(args.arm_dir.resolve(),step,w)
