#!/usr/bin/env python3
"""Recover returns from immutable native rewards, never from imagined rewards.

Joins only independently scored terminal episodes present in the supplied
summary. Other completed/partial episodes are counted but never backfilled
with another episode's score. Does not start Isaac or an optimizer.
"""
import argparse,hashlib,json,subprocess
from pathlib import Path
import torch
from quadsmash.rl.supermimic_native_metrics import NativeReturnAccumulator,summarize_native_episodes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--dataset-sha256',required=True)
    p.add_argument('--episodes',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    actual=hashlib.sha256(a.dataset.read_bytes()).hexdigest()
    if actual!=a.dataset_sha256:raise ValueError('Native dataset hash changed')
    data=torch.load(a.dataset,map_location='cpu',weights_only=True)['tensors']
    source=json.loads(a.episodes.read_text())
    lookup={(int(r['env_id']),int(r['episode_index'])):r for r in source}
    if len(lookup)!=len(source):raise ValueError('Duplicate scored episode')
    n=data['reward'].shape[1];acc=NativeReturnAccumulator(n,'cpu');matched=[];terminal_count=0
    for tick in range(len(data['reward'])):
        if tick:
            expected=data['episode'][tick-1]+(data['terminated'][tick-1]|data['timeout'][tick-1]).long()
            if not torch.equal(expected,data['episode'][tick]):raise ValueError('Episode boundary not contiguous')
        acc.add(data['reward'][tick])
        done=data['terminated'][tick]|data['timeout'][tick];ids=torch.where(done)[0]
        values=acc.records(ids);terminal_count+=len(ids)
        for i,value in values.items():
            key=(i,int(data['episode'][tick,i]))
            if key in lookup:
                row=dict(lookup.pop(key),**value,return_source='reconstructed_from_saved_native_rewards',
                         return_terminal_archive_tick=tick)
                matched.append(row)
        acc.reset(ids)
    if lookup:raise ValueError('Missing terminal native rewards for '+str(list(lookup)))
    matched.sort(key=lambda row:(row['env_id'],row['episode_index']))
    summary=summarize_native_episodes(matched)
    summary.update(scope='Historical native reward reconstruction; no new evaluation or policy update',
        dataset=str(a.dataset.resolve()),dataset_sha256=actual,scored_episodes=str(a.episodes.resolve()),
        scored_episodes_sha256=hashlib.sha256(a.episodes.read_bytes()).hexdigest(),
        archive_control_ticks=len(data['reward']),archive_terminal_episodes=terminal_count,
        unscored_terminal_episodes=terminal_count-len(matched),partial_lanes=int((acc.steps>0).sum()),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    (a.output/'episodes_with_returns.json').write_text(json.dumps(matched,indent=2,allow_nan=False)+'\n')
    (a.output/'metrics.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2,allow_nan=False))


if __name__=='__main__':main()
