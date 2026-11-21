"""Score saved research milestones on the declared held-out fixed panel and log paired outcomes."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
from torch.utils.tensorboard import SummaryWriter

from compare_supermimic_fixed_stages import load_run
from quadsmash.rl.supermimic_physical_loop_loading_v1 import tensor_digest
from quadsmash.rl.supermimic_independent_panel import fixed128_cases
from run_supermimic_research_scale import digest
from report_supermimic_research_curve import arm_origin, axes


def score_fixed128(folder, checkpoint, checkpoint_sha, actor_sha, execution_root, seed):
    def validate(directory):
        _, identity, actor = load_run(directory)
        cases, marker = fixed128_cases(identity['task']['protocol_sha256'], seed)
        metrics = json.loads((directory/'metrics.json').read_text())
        if (identity['seed'] != seed or identity['task'].get('research_evaluation') != marker
                or sorted((r['case_id'], lane) for lane, r in enumerate(cases)) != identity['assignments']
                or actor != actor_sha or metrics['evaluated_checkpoint_sha256'] != checkpoint_sha):
            raise ValueError('Fixed evaluation policy, case assignment or checkpoint differs')
        return metrics

    with folder.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if folder.exists():
            return validate(folder)
        with tempfile.TemporaryDirectory(dir=folder.parent, prefix='.'+folder.name+'_') as temp:
            run_dir = Path(temp)/'run'
            cmd = [sys.executable, 'scripts/evaluate_supermimic_v1.py', '--checkpoint', str(checkpoint),
                   '--output', str(run_dir), '--g07-execution-root', str(execution_root),
                   '--seed', str(seed), '--research-fixed128', '--physics-device', 'cuda:0', '--device',
                   'cuda:0', '--headless', '--kit_args=--renderer/multiGpu/enabled=false']
            with folder.with_suffix('.log').open('a') as log:
                subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], env=dict(os.environ,PYTHONPATH='.:scripts'),
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            metrics = validate(run_dir)
            run_dir.rename(folder)
            return metrics


def evaluate(folder, step, writer):
    reg=json.loads((folder/'registration.json').read_text())
    cmd=reg['command']
    seed=882072
    execution_root=cmd[cmd.index('--g07-execution-root')+1]
    initial_checkpoint=Path(cmd[cmd.index('--resume-training')+1])
    old=reg['initial']['actor_sha256']
    baseline=folder.parent/'research_fixed128_zero'
    initial_metrics=score_fixed128(baseline, initial_checkpoint, reg['initial']['checkpoint_sha256'], old, execution_root, seed)
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
    metrics=score_fixed128(folder_eval, checkpoint, digest(checkpoint), actor, execution_root, seed)
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
        for axis,pos in positions.items():
            tag='fixed128/'+axis+'/'+k
            if step == 128 and initial_metrics.get(k) is not None:
                writer.add_scalar(tag,initial_metrics[k],0)
            if metrics.get(k) is not None:
                writer.add_scalar(tag,metrics[k],pos)
    writer.flush()


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arm-dir',type=Path,required=True)
    ap.add_argument('--tensorboard',type=Path,required=True)
    ap.add_argument('--steps',nargs='+',type=int,default=[128,256])
    args=ap.parse_args()
    with SummaryWriter(str(args.tensorboard)) as w:
        for step in args.steps:evaluate(args.arm_dir.resolve(),step,w)
