"""Distill the native-supported part of a Recovery BPTT update on its own branch observations."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts.fit_dynamimic_recovery_wm import read_data
from scripts.fit_dynamimic_recovery_critic import root_termination
from scripts.dynamimic_recovery_validation import policies
from scripts.train_dynamimic_recovery_bptt import mixed_request, recovery_start_groups, stratified_starts
from quadsmash.rl.supermimic_v1 import load_ppo_initialized_actor
from quadsmash.rl.supermimic_physical_loop_loading_v1 import RECOVERY_NAMES


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('before','after','pairs','checkpoint','teacher','companion','output'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    audit = json.loads(args.pairs.read_text())
    if (audit['baseline'],audit['candidate']) != (digest(args.checkpoint),digest(args.teacher)):
        raise ValueError('Behavior checkpoints differ from native pair audit')
    payload = torch.load(args.checkpoint,weights_only=False,mmap=True,map_location='cpu')
    source = Path(payload['metadata']['init_policy_checkpoint']['path'])
    if digest(source) != payload['metadata']['init_policy_checkpoint']['sha256']:
        raise ValueError('Base initialization changed')
    base,comp,teacher,_ = policies(args.checkpoint,args.teacher,args.companion,
                                   lambda:load_ppo_initialized_actor(source,'cpu')[0])
    fields = ('active','obs','request','release_time','strict','relaxed','done','net')
    a,ma = read_data(args.before,fields)
    b,mb = read_data(args.after,fields)
    if ma['cases'] != mb['cases'] or ma['companion_sha256'] != mb['companion_sha256'] or ma['companion_sha256'] != digest(args.companion):
        raise ValueError('Native cases or companion differ')
    if [c['case_id'] for c in audit['cases']] != [c['case_id'] for c in ma['cases']]:
        raise ValueError('Case order changed')
    def outcome(d):
        d['root_terminated'] = root_termination(d['done'],d['after'][...,2]+torch.tensor(ma['origins'])[:,2])
        fault=((d['body']|d['root_terminated']|d['net'])&d['live']).any(0)
        return ((~fault).long() * (1+(d['relaxed']&d['live']).any(0).long()+
                                   (d['strict']&d['live']).any(0).long()))
    oa,ob=outcome(a),outcome(b)
    gates=[d['active'].int().argmax(0) for d in (a,b)]
    lane=torch.arange(len(oa))
    matched=torch.tensor([r['paired_active'] and all(r['prefix_max_abs_'+k]==0 for k in
                         ('before','request','final_joint_targets')) for r in audit['cases']])
    matched &= (gates[0]==gates[1]) & (a['before'][gates[0],lane]==b['before'][gates[1],lane]).all(-1)
    good=(ob>oa)&matched
    train=lane%16>=4
    positive=good & train
    if not positive.any():
        raise ValueError('No independent beneficial native branches')
    head=list(base.actor.recovery.children()); features=[]; targets=[]; weights=[]
    max_error=0.
    # Teacher acts on its own observed trajectory. Reference anchor uses the baseline branch.
    for policy,d,chosen in ((teacher,b,positive),(base,a,train & ~positive)):
        t,l=(d['active'] & chosen & (d['obs'][...,775]-d['release_time']>=.04)).nonzero().T
        counts=l.bincount(minlength=len(lane)).clamp_min(1)
        w=counts[l].double().reciprocal(); weights.append(.5*w/w.sum())
        for ids in torch.arange(len(t)).split(512):
            ti,li=t[ids],l[ids];obs=d['obs'][ti,li]
            raw=mixed_request(policy,comp,obs,li%2==1,d['release_time'][ti,li])
            error=(raw.clamp(-1,1)-d['request'][ti,li]).abs().max()
            max_error=max(max_error,float(error))
            z=base.actor.normalizer(obs);x=z
            for layer in head[:-1]:
                x=layer(x)
            features.append(F.pad(x,(0,1),value=1.))
            targets.append(policy.actor.recovery(z)-base.actor.recovery(z))
    if max_error>2e-6:
        raise ValueError(f'Native action replay mismatch: {max_error}')
    x=torch.cat(features).double();y=torch.cat(targets).double(); w=torch.cat(weights)
    gram=(x.T*w)@x;gram.diagonal().add_(.001)
    delta=torch.linalg.solve(gram,(x.T*w)@y).float()
    snapshot=[int(p.stem.split('_')[-1]) for p in (args.before/'trajectories').glob('g07_*.pt')]
    checks=stratified_starts(recovery_start_groups(a,snapshot,~train),128,torch.Generator().manual_seed(9917))
    t,l=checks.T
    def actions():
        return mixed_request(base,comp,a['obs'][t,l],l%2==1,a['release_time'][t,l]).clamp(-1,1)
    nominal=actions(); weight=head[-1].weight.clone();bias=head[-1].bias.clone()
    args.output.mkdir(exist_ok=False,parents=True)
    report=dict(behavior=audit['baseline'],companion=digest(args.companion),teacher=audit['candidate'],
                positive_cases={s:int((positive & (lane%2==i)).sum()) for i,s in enumerate(('primary','h8d'))},
                candidates={})
    for label,alpha,budget in [('conservative',.5,.02),('moderate',1.,.035)]:
        head[-1].weight.copy_(weight+alpha*delta[:-1].T);head[-1].bias.copy_(bias+alpha*delta[-1])
        rms=float((actions()-nominal).square().mean().sqrt())
        state=base.state_dict()
        if not 0<rms<=budget or any(not torch.equal(v,payload['actor'][k]) for k,v in state.items() if k not in RECOVERY_NAMES):
            raise ValueError('Action budget or freeze failed')
        dest=args.output/(label+'.pt')
        torch.save(dict(payload,actor=state,actor_updates=payload['actor_updates']+1),dest)
        report['candidates'][label]=dict(after=digest(dest),rms=rms,budget=budget,replay_max_error=max_error)
    (args.output/'fit.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
