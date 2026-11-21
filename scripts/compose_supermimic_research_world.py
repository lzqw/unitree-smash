"""Add the calibrated research contact module to a trained shared WM, retaining optimizer moments."""
import argparse
import json
from pathlib import Path

import torch

from scripts.evaluate_supermimic_fresh_wm_panel import digest
from quadsmash.rl.supermimic_action_ball_v1 import CalibratedRacketImpulseV13
from quadsmash.rl.supermimic_physical_loop_loading_v1 import load_physical_loop, tensor_digest
from quadsmash.rl.supermimic_physical_run_guard_v1 import atomic_torch_save


def extend_world_optimizer(saved, old, new):
    old_names=list(dict(old.named_parameters()))
    new_names=list(dict(new.named_parameters()))
    if [n for n in new_names if not n.startswith('action_ball_model.')] != old_names:
        raise ValueError('Existing physical parameters changed order or identity')
    group,=saved['param_groups']
    if len(group['params'])!=len(old_names):
        raise ValueError('Old Adam does not cover the source WM')
    ids={n:i for i,n in enumerate(new_names)}
    state={}
    for name,old_id in zip(old_names,group['params'],strict=True):
        if old_id not in saved['state']:
            continue
        old_state=saved['state'][old_id]
        if any(t.shape!=dict(new.named_parameters())[name].shape for k,t in old_state.items() if k.startswith('exp_avg')):
            raise ValueError('Adam moment shape mismatch for '+name)
        state[ids[name]]=old_state
    return dict(state=state,param_groups=[dict(group,params=list(range(len(new_names))))])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','contact','archive','safety-mapping','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    ck=torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=False)
    meta=ck['metadata']['physical_imagination']
    paths=meta['checkpoint_paths']
    if paths.get('action_ball') or args.output.exists():
        raise ValueError('Expected source without added contact branch and a new destination')
    kw=dict(archive=args.archive,physical_checkpoint=paths['physical'],ball_checkpoint=paths['ball'],task_checkpoint=paths['task'],
            execution_root=meta['g07_execution_root'],safety_mapping=args.safety_mapping,device='cpu')
    old=load_physical_loop(**kw)[1]
    new=load_physical_loop(**kw,action_ball_checkpoint=args.contact)[1]
    if old.manifest['checkpoint_sha256']!=meta['checkpoint_sha256'] or not isinstance(new.action_ball_model,CalibratedRacketImpulseV13):
        raise ValueError('Expected the same component sources and a calibrated research contact branch')
    old.load_state_dict(ck['world'],strict=True)
    added={k:v for k,v in new.state_dict().items() if k.startswith('action_ball_model.')}
    new.load_state_dict(ck['world'] | added,strict=True)
    new_optimizer=extend_world_optimizer(ck['world_optimizer'],old,new)
    test_opt=torch.optim.Adam(new.parameters())
    test_opt.load_state_dict(new_optimizer)
    provenance=dict(source_checkpoint_sha256=digest(args.checkpoint),source_world_sha256=tensor_digest(old.state_dict().items()),
                    contact_checkpoint_sha256=digest(args.contact),world_sha256=tensor_digest(new.state_dict().items()),
                    existing_adam_states=len(new_optimizer['state']),actor_sha256=tensor_digest(ck['actor'].items()))
    ck.update(world=new.state_dict(),world_optimizer=new_optimizer,
              wm_fit_provenance=dict(verified=False,reason='research contact branch added',composition=provenance))
    ck['metadata']=dict(ck['metadata'],physical_imagination=dict(meta,**new.manifest),
                        online_forward_validation=None,research_contact_composition=provenance)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    atomic_torch_save(ck,args.output)
    restored=torch.load(args.output,map_location='cpu',mmap=True,weights_only=False)
    new.load_state_dict(restored['world'],strict=True)
    if tensor_digest(new.state_dict().items())!=provenance['world_sha256']:
        raise RuntimeError('Serialized research WM changed')
    args.output.with_suffix('.json').write_text(json.dumps(dict(**provenance,checkpoint_sha256=digest(args.output)),indent=2)+'\n')


if __name__=='__main__':
    main()
