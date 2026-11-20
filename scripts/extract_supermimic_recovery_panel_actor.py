#!/usr/bin/env python3
"""Extract the saved post-Recovery/pre-Approach weights for fixed-case native evaluation."""
import argparse
import json
from pathlib import Path

import torch

from quadsmash.rl.supermimic_physical_loop_loading_v1 import (
    RECOVERY_NAMES, APPROACH_NAMES, load_ppo_initialized_actor, tensor_digest,
)
from scripts.evaluate_supermimic_fresh_wm_panel import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    torch.set_num_threads(2)
    ck=torch.load(a.checkpoint,weights_only=False,mmap=True,map_location='cpu')
    phases=[(r['event']['head'],r['result']['accepted']) for r in ck['alternating_schedule']['results']
            if r['event']['kind']=='train']
    if phases!=[('recovery',8),('approach',8)] or ck['global_counters']['accepted']!=16:
        raise ValueError('Expected a completed single R8/A8 pair')
    meta=ck['metadata'];src=meta['init_policy_checkpoint']
    init,original=load_ppo_initialized_actor(Path(src['path']),torch.device('cpu'))
    if original['sha256']!=src['sha256']:
        raise ValueError('Initialization changed')
    weights=ck['actor_references']['block']
    initial=init.state_dict()
    if weights.keys()!=initial.keys() or weights.keys()!=ck['actor'].keys():
        raise ValueError('Snapshot has different actor keys')
    changed_r=set();changed_a=set()
    for k,v in weights.items():
        if not torch.equal(v,initial[k]):changed_r.add(k)
        if not torch.equal(v,ck['actor'][k]):changed_a.add(k)
    if not changed_r or not changed_a or changed_r-set(RECOVERY_NAMES) or changed_a-set(APPROACH_NAMES):
        raise ValueError('Saved block snapshot fails independent head isolation')
    receipt=dict(parent_checkpoint_sha256=digest(a.checkpoint),source_field='actor_references.block',
                 pre_actor_sha256=tensor_digest(initial.items()),block_actor_sha256=tensor_digest(weights.items()),
                 post_approach_actor_sha256=tensor_digest(ck['actor'].items()),
                 recovery_parameters_changed=sorted(changed_r),approach_parameters_changed=sorted(changed_a),
                 scope='historical frozen panel; actor-only export')
    a.output.mkdir(parents=True,exist_ok=False)
    torch.save(dict(algorithm=ck['algorithm'],metadata=meta,actor=weights,evaluation_snapshot=receipt),a.output/'actor.pt')
    receipt['export_sha256']=digest(a.output/'actor.pt')
    (a.output/'origin.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':
    main()
