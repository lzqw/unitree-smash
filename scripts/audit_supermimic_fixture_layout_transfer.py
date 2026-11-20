#!/usr/bin/env python3
"""Compare native signed derivatives across terrain layouts on registered launch families."""
import argparse
import json
from pathlib import Path

import torch

from audit_supermimic_surface_derivative_scales import vector_summary
from evaluate_supermimic_contact_rollout import sha


def magnitude(x):
    n = x.norm(dim=-1)
    return dict(median=float(n.median()), p95=float(n.quantile(.95)), maximum=float(n.max()))


def read(wide, split):
    report = json.loads((wide/(split+'.json')).read_text())
    plan_path = wide/'design.json'
    if report['split'] != split or report['design_sha256'] != sha(plan_path):
        raise ValueError('Mismatched native fixture audit')
    return json.loads(plan_path.read_text()),report,sorted(report['samples'],key=lambda x:(x['seed'],x['lane']))


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('spaced','compact','unloaded-spaced','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--split',choices=('train','development'),required=True)
    a=p.parse_args()
    old, old_audit, old_rows=read(a.spaced,a.split)
    new, new_audit, new_rows=read(a.compact,a.split)
    ids=lambda rows:[(r['seed'],r['lane']) for r in rows]
    if (new['previous_design_sha256'] != sha(a.spaced/'design.json') or
            new['source_teacher_sha256'] != old['source_teacher_sha256'] or ids(old_rows)!=ids(new_rows)
            or new['fixture_layout']!='teacher' or new['expected_origin_spacing_m']!=0.):
        raise ValueError('Unmatched sources or held-out families')
    comparisons={k:[] for k in ('shuttle_position_m','shuttle_velocity_mps','racket_position_m',
                                'racket_linear_velocity_mps','racket_angular_velocity_radps','direction_unit')}
    old_precision=[];new_precision=[];source_hashes={}
    for seed in sorted({row['seed'] for row in old_rows}):
        packets=[]
        for wide, audit in [(a.spaced,old_audit),(a.compact,new_audit)]:
            path=wide/'nominal_a'/f'spec_{seed}'/'trace.pt'
            if sha(path)!=audit['archives'][str(path)]:raise ValueError('Nominal trace changed')
            packets.append(torch.load(path,weights_only=True,mmap=True))
        xo,xn=packets
        zpath=a.unloaded_spaced/'nominal_a'/f'spec_{seed}'/'trace.pt'
        source_hashes[str(zpath)]=sha(zpath)
        zu=torch.load(zpath,weights_only=True,mmap=True)
        if zu['schema']!='native-unloaded-racket-oracle-v1':
            raise ValueError('Expected independent parked shuttle trace for original origins')
        if ids([r for r in old_rows if r['seed']==seed])!=[(seed,i) for i in range(old['lanes'])]:
            raise ValueError('Missing native lanes')
        for packet in (xo,xn,zu):
            if packet['seed']!=seed:raise ValueError('Seed differs')
            torch.testing.assert_close(packet['parent_lane'],xo['parent_lane'],atol=0,rtol=0)
            torch.testing.assert_close(packet['velocity_intervention']['joint_targets'],
                                       xo['velocity_intervention']['joint_targets'],atol=0,rtol=0)
        torch.testing.assert_close(zu['velocity_intervention']['nominal_shuttle'],
                                   xo['velocity_intervention']['nominal_shuttle'],atol=0,rtol=0)
        def tensor(x,k):
            return x['raw']['substep_trace']['tensors'][k][0]
        origin=tensor(zu,'shuttle_state_w')[:,:3]-torch.tensor([0.,0.,5.])
        if origin.abs().max()<16 or torch.count_nonzero(torch.as_tensor(xn['fixture_layout']['origins_w'])):
            raise ValueError('Different registered environment coordinates required')
        so,sn=tensor(xo,'shuttle_state_w'),tensor(xn,'shuttle_state_w')
        for x,values in [(so,old_precision),(sn,new_precision)]:
            coords=x[:,:3].abs()
            values.append((torch.nextafter(coords,torch.full_like(coords,torch.inf))-coords).amax(-1))
        for key,value in [
            ('shuttle_position_m',so[:,:3]-origin-sn[:,:3]),('shuttle_velocity_mps',so[:,7:10]-sn[:,7:10]),
            ('racket_position_m',tensor(xo,'racket_position_w')-origin-tensor(xn,'racket_position_w')),
            ('racket_linear_velocity_mps',tensor(xo,'racket_linear_velocity_w')-tensor(xn,'racket_linear_velocity_w')),
            ('racket_angular_velocity_radps',tensor(xo,'racket_angular_velocity_w')-tensor(xn,'racket_angular_velocity_w')),
            ('direction_unit',xo['velocity_intervention']['direction_world']-xn['velocity_intervention']['direction_world'])]:
            comparisons[key].append(value)
    def delta(rows,key):return torch.tensor([r['delta'][key] for r in rows])
    no,nn=delta(old_rows,'native'),delta(new_rows,'native')
    mo,mn=[delta(rows,'native_racket_analytic_rollout') for rows in (old_rows,new_rows)]
    so,sn=[torch.tensor([r['kind']=='racket_stable' for r in rows]) for rows in (old_rows,new_rows)]
    groups={}
    for name,mask in [('all',torch.ones_like(so)),('both_timing_stable',so&sn),('only_compact_stable',sn&~so),
                      ('only_spaced_stable',so&~sn),('compact_stable',sn),('spaced_stable',so)]:
        groups[name]=dict(rows=int(mask.sum()))
        if mask.any():
            groups[name].update(native_layout_agreement=vector_summary(nn[mask],no[mask]),
                                compact_model=vector_summary(nn[mask],mn[mask]),spaced_model=vector_summary(no[mask],mo[mask]))
    report=dict(split=a.split, paired_cases=len(old_rows), counts=dict(spaced_stable=int(so.sum()),compact_stable=int(sn.sum())),
                groups=groups, local_initial_differences={k:magnitude(torch.cat(v)) for k,v in comparisons.items()},
                world_position_fp32_ulp_median_m={k:float(torch.cat(v).median()) for k,v in [('spaced',old_precision),('compact',new_precision)]},
                parked_shuttle_sources=source_hashes,wide_audit_hashes=[sha(x/(a.split+'.json')) for x in (a.spaced,a.compact)],
                signed_surface_speed_difference_mps=2*new['epsilon_surface_mps'],policy_updates=0,contact_gradients_admitted=False)
    with a.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)


if __name__=='__main__':main()
