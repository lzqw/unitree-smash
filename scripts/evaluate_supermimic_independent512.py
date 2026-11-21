#!/usr/bin/env python3
"""Evaluate a chosen frozen checkpoint on 512 predeclared launches at the original native timebase."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from compare_supermimic_fixed_stages import load_run
from quadsmash.rl.supermimic_independent_panel import SHA256, load_panel, panel_batch
from quadsmash.rl.supermimic_native_metrics import summarize_native_episodes


def validated_batch(batch, i, checkpoint_sha, panel):
    scores, identity, actor = load_run(batch)
    expected, suite = panel_batch(i, panel['protocol_sha256'], identity['seed'])
    metrics = json.loads((batch/'metrics.json').read_text())
    if (metrics['evaluated_checkpoint_sha256'] != checkpoint_sha
            or identity['task'].get('independent_evaluation') != suite
            or identity['task']['protocol_sha256'] != panel['protocol_sha256']
            or sorted((r['case_id'], lane) for lane,r in enumerate(expected)) != identity['assignments']):
        raise ValueError(f'Independent batch {i} has a different checkpoint, protocol or case assignment')
    return scores, identity, actor


def collect_scores(folder, checkpoint_sha, panel):
    rows = []
    actor = execution = physics = None
    for i in range(4):
        scores, identity, sha = validated_batch(folder/f'batch_{i}',i,checkpoint_sha,panel)
        if i and (actor, execution, physics) != (sha, identity['execution'], identity['physics']):
            raise ValueError('Policy or physics differ across independent batches')
        actor, execution, physics = sha, identity['execution'], identity['physics']
        rows.extend(scores)
    if len({r['case_id'] for r in rows}) != 512:
        raise ValueError('Missing or repeated independent episodes')
    return dict(panel_sha256=SHA256, checkpoint_sha256=checkpoint_sha, actor_sha256=actor,
                protocol_sha256=panel['protocol_sha256'], episodes=len(rows),
                numerical_failures=sum(r['numerical_failures'] for r in rows),
                metrics=summarize_native_episodes(rows), fixed_updates=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','g07-execution-root','output'):
        parser.add_argument('--'+name,required=True,type=Path)
    args = parser.parse_args()
    folder = args.output.resolve()
    folder.mkdir(parents=True,exist_ok=True)
    panel = load_panel()
    with (folder/'evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        with args.checkpoint.open('rb') as stream:
            checkpoint_sha = hashlib.file_digest(stream,'sha256').hexdigest()
        for i in range(4):
            batch = folder/f'batch_{i}'
            if batch.is_dir():
                validated_batch(batch,i,checkpoint_sha,panel)
                continue
            with tempfile.TemporaryDirectory(dir=folder,prefix=f'.batch_{i}_') as temp:
                run_dir = Path(temp)/'run'
                cmd = [sys.executable,'scripts/evaluate_supermimic_v1.py','--checkpoint',str(args.checkpoint.resolve()),
                       '--g07-execution-root',str(args.g07_execution_root.resolve()),'--output',str(run_dir),
                       '--seed',str(panel['native_reset_seed']),'--independent-panel-batch',str(i),
                       '--device','cuda:0','--physics-device','cuda:0','--headless','--kit_args=--renderer/multiGpu/enabled=false']
                with (folder/f'batch_{i}.log').open('a') as log:
                    subprocess.run(cmd,cwd=Path(__file__).resolve().parents[1],env=dict(os.environ,PYTHONPATH='.:scripts'),
                                   stdout=log,stderr=subprocess.STDOUT,check=True)
                validated_batch(run_dir,i,checkpoint_sha,panel)
                run_dir.rename(batch)
        result = collect_scores(folder,checkpoint_sha,panel)
        temp = folder/'independent512.json.tmp'
        temp.write_text(json.dumps(result,indent=2)+'\n')
        temp.replace(folder/'independent512.json')
        print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
