#!/usr/bin/env python3
"""Compare analytic local Jacobians, model secants and native signed response on identical racket paths."""
import argparse
import json
from pathlib import Path
import torch
from torch.nn.functional import normalize

from audit_supermimic_surface_derivative_scales import read_material, vector_summary
from calibrate_supermimic_paired_racket_solver import read_split
from evaluate_supermimic_contact_rollout import BallImpulsePriorV1,sha,tick


def interpolate_motion(minus,plus,alpha):
    t=alpha[None,:,None]
    result=[]
    for k,(x,y) in enumerate(zip(minus,plus,strict=True)):
        if k==1:
            y=torch.where((x*y).sum(-1,keepdim=True)<0,-y,y)
            result.append(normalize(torch.lerp(x,y,t),dim=-1))
        else:
            result.append(torch.lerp(x,y,t))
    return result


def local_response(prior,ball,spin,minus,plus,cfg,material,active_fraction):
    alpha=ball.new_full((len(ball),),active_fraction,requires_grad=True)
    motion=interpolate_motion(minus,plus,alpha)
    out,_=tick(prior,ball,[x[0] for x in motion],[x[-1] for x in motion],cfg['fitted'],
               cfg['solver']['pair_contact_offset_m'],measured_path=[x[:-1] for x in motion],
               initial_spin=spin,racket_material=material,compound_racket=True)
    v=out[:,3:]
    # The lanes evolve independently. Include a zero connection for all-free batches.
    components=[]
    for k in range(3):
        g,=torch.autograd.grad(v[:,k].sum()+0*alpha.sum(),alpha,retain_graph=k<2)
        components.append(g)
    return v.detach(),torch.stack(components,-1).detach()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('wide','labels','calibration','material','readout','output'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--device',default='cpu')
    a=p.parse_args();torch.set_num_threads(2)
    labels=json.loads(a.labels.read_text());split=labels['split'];file=a.wide/(split+'.json')
    if split not in ('train','development') or labels['wide_audit_sha256']!=sha(file):
        raise ValueError('Expected matching training or development archive identities')
    if (labels['calibration_sha256'],labels['material_sha256'],labels['coefficient_fit_sha256'])!=(sha(a.calibration),sha(a.material),sha(a.readout)):
        raise ValueError('Different transition used for native derivative comparison')
    rows,streams=read_split(file,a.wide,split)
    if labels['case_ids']!=[[r['seed'],r['lane']] for r in rows]:raise ValueError('Misaligned native labels')
    prov=json.loads((a.wide/'nominal_a/provenance.json').read_text())
    mat=read_material(a.wide,prov,a.material,a.readout,a.calibration).to(a.device)
    prior=BallImpulsePriorV1().to(a.device).eval()
    cfg=json.loads(a.calibration.read_text())
    outputs={k:[] for k in ('native','model_secant','local','branch_negative','branch_positive')}
    for i,seed in enumerate(sorted({r['seed'] for r in rows})):
        ball,v0,minus=streams['minus'][i];other,v1,plus=streams['plus'][i]
        torch.testing.assert_close(ball,other,rtol=0,atol=0)
        packets=[torch.load(a.wide/b/f'spec_{seed}'/'trace.pt',weights_only=True,mmap=True) for b in ('minus','plus')]
        w0,w1=[v['raw']['substep_trace']['tensors']['shuttle_state_w'][0,:,10:13] for v in packets]
        torch.testing.assert_close(w0,w1,atol=0,rtol=0)
        args=prior,ball.to(a.device),w0.to(a.device),[v.to(a.device) for v in minus],[v.to(a.device) for v in plus],cfg,mat
        center,local=local_response(*args,.5)
        endpoints=[]
        for fraction in (0.,1.):
            with torch.no_grad():
                motion=interpolate_motion(args[3],args[4],torch.full_like(args[1][:,0],fraction))
                value,_=tick(prior,args[1],[v[0] for v in motion],[v[-1] for v in motion],cfg['fitted'],
                             cfg['solver']['pair_contact_offset_m'],measured_path=[v[:-1] for v in motion],
                             initial_spin=args[2],racket_material=mat,compound_racket=True)
                endpoints.append(value[:,3:])
        for key,data in zip(outputs,(v1-v0,endpoints[1]-endpoints[0],local,2*(center-endpoints[0]),2*(endpoints[1]-center)),strict=True):
            outputs[key].append(data.cpu())
    data={k:torch.cat(v) for k,v in outputs.items()}
    groups={}
    masks=dict(all=torch.ones(len(rows),dtype=torch.bool),wide_stable=torch.tensor([r['kind']=='racket_stable' for r in rows]),
               native_convergent=torch.tensor(labels['incident_labels']['native_convergent_stable']))
    for name,mask in masks.items():
        finite=torch.stack([v.isfinite().all(-1) for v in data.values()]).all(0)
        report=dict(rows=int(mask.sum()),nonfinite=int((mask&~finite).sum()))
        if (mask&finite).any():
            vals={k:v[mask&finite] for k,v in data.items()}
            for x,y in [('native','local'),('native','model_secant'),('model_secant','local'),('branch_negative','branch_positive')]:
                report[x+'_vs_'+y]=vector_summary(vals[x],vals[y])
        groups[name]=report
    with a.output.open('x') as f:
        json.dump(dict(groups=groups,label_sha256=sha(a.labels),split=split,device=a.device,
                  differentiation='Midpoint of two recorded surface pose and twist sequences; alpha from 0 to 1',
                  units='m/s per signed interval',actor_updates=0,contact_gradients_admitted=False),f,indent=2,allow_nan=False)


if __name__=='__main__':main()
