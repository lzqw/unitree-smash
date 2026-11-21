"""Measure Recovery return-direction errors with a measured reward prefix and measured critic input."""
import argparse
import json
from pathlib import Path

import torch

from quadsmash.rl.supermimic_evidence_binding_v1 import complete_native_rtg
from quadsmash.rl.supermimic_physical_loop_loading_v1 import RECOVERY_NAMES, tensor_digest
from quadsmash.rl.supermimic_v1 import MLP
from scripts.evaluate_supermimic_fresh_wm_panel import digest


def load_stage(frozen, checkpoint, snapshots):
    read=lambda p:json.loads(p.read_text())
    ck=torch.load(checkpoint,weights_only=False,mmap=True,map_location='cpu')
    archive_path=frozen/'complete_episodes.pt'
    archive=torch.load(archive_path,weights_only=True,mmap=True,map_location='cpu')
    tail=torch.load(snapshots,weights_only=True,map_location='cpu')
    records={r['env_id']:r for r in read(frozen/'episodes_summary.json')}
    held=set(ck['config']['wm_holdout_lanes'])
    actor=tensor_digest(ck['actor'].items())
    source=digest(archive_path)
    if (len(records)!=128 or source!=tail['source_sha256'] or source!=ck['runtime_validation_archive']['sha256']
            or actor!=tail['actor_sha256'] or actor!=archive['receipt']['actor_sha256']
            or tail['reproduced_sparse_inputs']<=0):
        raise ValueError('Actor, transitions and reconstructed critic inputs differ')
    gamma,={r['native_return_gamma'] for r in records.values()}
    data=archive['tensors']
    rtg,complete=complete_native_rtg(data,gamma)
    torch.testing.assert_close(rtg[0],torch.tensor([records[i]['native_episode_discounted_return'] for i in range(128)],
                                                  dtype=torch.float64),atol=1e-5,rtol=1e-6)
    values={}
    for i,(lane,ep,t) in enumerate(tail['identity']):
        if (ep!=int(data['episode'][t,lane]) or not complete[t,lane] or not complete[0,lane]
                or records[lane]['episode_index']!=ep or records[lane]['actor_sha256']!=actor):
            raise ValueError('Native return or event identity mismatch')
        if not tail['recovery'][i]:
            continue
        offset=t-tail['landing_by_lane'][lane]
        prefix=(data['reward'][:t,lane].double()*gamma**torch.arange(t,dtype=torch.float64)).sum().item()
        if abs(prefix+gamma**t*float(rtg[t,lane])-float(rtg[0,lane]))>1e-7:
            raise ValueError('Measured rewards do not reconstruct the full episode')
        values[lane,offset]=(prefix,gamma**t,i,float(rtg[0,lane]),t)
    return dict(ck=ck,actor=actor,source=source,data=data,held=held,records=records,
                tail=tail,values=values,gamma=gamma)


def summarize(rows):
    nonzero=[r for r in rows if r['native_delta']!=0.]
    def signs(key):
        return sum(r[key]*r['native_delta']>0 for r in nonzero)
    return dict(lanes=len(rows),nonzero=len(nonzero),same_sign_prefix_only=signs('prefix_delta'),
                same_sign_native_prefix_plus_critic=signs('bootstrapped_delta'),
                mean_abs_error=sum(abs(r['bootstrapped_delta']-r['native_delta']) for r in rows)/len(rows) if rows else None)


def fit_paired_tail(model, a, b, common):
    """Fit true signed return differences at paired landing offsets using only non-held episodes."""
    from copy import deepcopy
    from torch.nn import functional as F
    candidate=deepcopy(model)
    for layer in candidate.parameters():
        layer.requires_grad_(False)
    output=list(candidate.modules())[-1]
    if not isinstance(output,torch.nn.Linear):
        raise ValueError('Expected the terminal critic readout')
    output.requires_grad_(True)
    optimizer=torch.optim.Adam(output.parameters(),lr=1e-4)
    train=[]
    for key in sorted(common):
        lane,_=key
        if lane in a['held'] or lane in b['held']:
            continue
        left,right=a['values'][key],b['values'][key]
        # Undo the prefix discount to train the critic in its native value units.
        train.append((left[2],right[2],(left[3]-left[0])/left[1],(right[3]-right[0])/right[1]))
    if not train:
        raise ValueError('No paired training landings')
    ia,ib,ya,yb=zip(*train)
    xa,xb=a['tail']['inputs'][list(ia)],b['tail']['inputs'][list(ib)]
    ya,yb=torch.tensor(ya),torch.tensor(yb)
    same, changed=(xa==xb).all(-1), (ya!=yb)
    with torch.enable_grad():
        for _ in range(8):
            va,vb=candidate(xa).flatten(),candidate(xb).flatten()
            absolute=.5*(F.smooth_l1_loss(va,ya)+F.smooth_l1_loss(vb,yb))
            signed=F.smooth_l1_loss(vb-va,yb-ya)
            optimizer.zero_grad(set_to_none=True)
            (absolute+signed).backward()
            torch.nn.utils.clip_grad_norm_(output.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
    return candidate,dict(steps=8,train_pairs=len(train),independent_training_lanes=len({k[0] for k in common if k[0] not in a['held']}),
                          identical_inputs_different_native_tail=int((same&changed).sum()),parameters='last readout only')


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('before','after','before-frozen','after-frozen','before-snapshots','after-snapshots','output'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--fit-paired-readout',action='store_true',help='Compare an eight-step offline fit on non-held native rollout pairs')
    args=p.parse_args()
    torch.set_num_threads(2)
    a=load_stage(args.before_frozen,args.before,args.before_snapshots)
    b=load_stage(args.after_frozen,args.after,args.after_snapshots)
    if (a['held']!=b['held'] or a['gamma']!=b['gamma'] or a['tail']['offsets']!=b['tail']['offsets']
            or a['ck']['metadata']['policy_execution']!=b['ck']['metadata']['policy_execution']
            or set(a['ck']['actor'])!=set(b['ck']['actor'])
            or any(not torch.equal(v,b['ck']['actor'][k]) for k,v in a['ck']['actor'].items() if k not in RECOVERY_NAMES)
            or any(a['records'][i]['case_id']!=b['records'][i]['case_id'] for i in range(128))):
        raise ValueError('Different native cases, training split, clock or fixed heads')
    torch.testing.assert_close(a['data']['state'][0],b['data']['state'][0],rtol=0,atol=0)
    # One frozen value function for both policies isolates simulated-prefix and value errors.
    model=MLP(b['tail']['inputs'].shape[-1],(512,256,128),1)
    model.load_state_dict(b['ck']['critic'],strict=True)
    overlap=a['values'].keys() & b['values'].keys()
    calibration=None
    if args.fit_paired_readout:
        model,calibration=fit_paired_tail(model,a,b,overlap)
    va,vb=(model(stage['tail']['inputs']).flatten() for stage in (a,b))
    offsets=a['tail']['offsets']
    common=set.intersection(*[{lane for lane,t in a['values'].keys() & b['values'].keys() if t==off and lane in a['held']} for off in offsets])
    panels={}
    for offset in offsets:
        rows=[]
        for lane in sorted(common):
            left=a['values'][lane,offset]
            right=b['values'][lane,offset]
            prefix=right[0]-left[0]
            value=right[1]*float(vb[right[2]])-left[1]*float(va[left[2]])
            rows.append(dict(lane=lane,case_id=b['records'][lane]['case_id'],ticks=[left[4],right[4]],
                             recovery_flip=int(b['records'][lane]['recovery_success'])-int(a['records'][lane]['recovery_success']),
                             native_delta=right[3]-left[3],prefix_delta=prefix,bootstrap_delta=value,
                             bootstrapped_delta=prefix+value))
        panels[offset]=dict(summary=summarize(rows),flips=summarize([r for r in rows if r['recovery_flip']]),cases=rows)
    with args.output.open('x') as f:
        json.dump(dict(archive_sha256=[a['source'],b['source']],actor_sha256=[a['actor'],b['actor']],
                       critic_sha256=tensor_digest(model.state_dict().items()),initial_state_identical_lanes=128,
                       held_lanes=len(a['held']),common_lanes=sorted(common),panels=panels,offline_calibration=calibration,
                       optimizer_updates=8 if calibration else 0, actor_updates=0, deployed=False,
                       scope='Same launch and initial state in two on-policy native refreshes. Only shared held cases observed at all landing offsets. Measured prefixes and post-landing critic inputs are oracle diagnostics.'),
                  f,indent=2,allow_nan=False)


if __name__=='__main__':
    main()
