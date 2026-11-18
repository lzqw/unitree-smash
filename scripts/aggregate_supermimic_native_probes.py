#!/usr/bin/env python3
"""Join compatible immutable paired-native evidence without inventing probes."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from quadsmash.rl.supermimic_native_pair_capture_v1 import BRANCHES,tree_digest


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def aggregate(paths,output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    bundles=[];hashes=set();sources=[]
    for path in map(Path,paths):
        digest=sha(path)
        if digest in hashes:raise ValueError('Duplicate probe bundle cannot increase support')
        hashes.add(digest)
        b=torch.load(path,map_location='cpu',weights_only=True)
        if b.get('schema')!='native-paired-prefix-region-v1':raise ValueError('Not a native paired-prefix bundle')
        if set(b.get('source_paths',{}))!=set(BRANCHES):raise ValueError('Missing original four-branch evidence')
        for branch,raw in b['source_paths'].items():
            if sha(raw)!=b['source_sha256'].get(branch):raise ValueError('Original native probe bytes changed')
        if bundles:
            for key in ('head','runtime_identity'):
                if b[key]!=bundles[0][key]:raise ValueError('Incompatible probe '+key)
            if tree_digest(b['actor_state'])!=tree_digest(bundles[0]['actor_state']):raise ValueError('Probe actor differs')
        n=len(b['pairs']['state'])
        if len(b['sample_ids'])!=n:raise ValueError('Sample identity count mismatch')
        if n!=1:raise ValueError('Aggregate original single-probe bundles, not aggregates')
        bundles.append(b);sources.append(dict(path=str(path.resolve()),sha256=digest))
    if not bundles:raise ValueError('No native probes')
    tensor_keys={k for k,v in bundles[0]['pairs'].items() if torch.is_tensor(v)}
    pairs={}
    for b in bundles:
        if {k for k,v in b['pairs'].items() if torch.is_tensor(v)}!=tensor_keys:
            raise ValueError('Probe tensor schemas differ')
        if b['pairs'].get('restore_mode')!='from-ground-identical-prefix':raise ValueError('Native prefix required')
    for key in tensor_keys:
        shapes={tuple(b['pairs'][key].shape[1:]) for b in bundles}
        if len(shapes)!=1:raise ValueError('Probe tensor shape mismatch: '+key)
        pairs[key]=torch.cat([b['pairs'][key] for b in bundles])
    pairs['restore_mode']='from-ground-identical-prefix'
    for key in ('prefix_sha256','executor_before_sha256'):
        pairs[key]={branch:[b['pairs'][key][branch] for b in bundles] for branch in BRANCHES}
        for values in zip(*pairs[key].values()):
            if not all(values) or len(set(values))!=1:raise ValueError('Probe branch prefixes/memories differ')
    pairs['native_contact_modes']={branch:[b['pairs'].get('native_contact_modes',{}).get(branch) for b in bundles]
                                   for branch in BRANCHES}
    sample_ids=[identity for b in bundles for identity in b['sample_ids']]
    result=dict(schema='native-paired-prefix-region-v1',head=bundles[0]['head'],
        runtime_identity=bundles[0]['runtime_identity'],actor_state=bundles[0]['actor_state'],pairs=pairs,
        sample_ids=sample_ids,command_starts=[row for b in bundles for row in b['command_starts']],
        source_bundles=sources,independent_episode_count=len(set(sample_ids)),
        scope='Measured native probes aggregated; no calibration limits, no automatic pass')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as stream:torch.save(result,stream)
    receipt=dict(path=str(output.resolve()),sha256=sha(output),probe_count=len(sample_ids),
                 independent_episode_count=len(set(sample_ids)),passed=None,source_bundles=sources)
    output.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundles',nargs='+',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    a=parser.parse_args();print(json.dumps(aggregate(a.bundles,a.output),indent=2))
