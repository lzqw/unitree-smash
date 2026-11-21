"""Choose an independent tangential request direction at each recorded training prefix."""
import argparse
import json
from pathlib import Path

import torch
from quadsmash.rl.supermimic_action_identification_dataset_v1 import _load_branch
from quadsmash.rl.supermimic_native_pair_capture_v1 import validate_spec
from quadsmash.rl.supermimic_physical_imagination_state_v1 import PhysicalImaginationStateV1
from quadsmash.rl.supermimic_physical_loop_loading_v1 import load_physical_loop
from quadsmash.rl.supermimic_racket_dynamics_v1 import racket_state
from scripts.declare_supermimic_contact_training_cases import resolve_probe_cases
from scripts.evaluate_supermimic_contact_rollout import sha


def tangent_directions(jac,normal,previous):
    """Maximize tangential center displacement subject to zero normal and old-direction projection."""
    jac,normal,previous=jac.double(),normal.double(),previous.double()
    row=normal[:,None,:]@jac
    constraints=torch.cat((torch.nn.functional.normalize(row,dim=2),previous[:,None,:]),dim=1)
    projector=torch.eye(6,dtype=jac.dtype,device=jac.device)-torch.linalg.pinv(constraints)@constraints
    _,_,vh=torch.linalg.svd(jac@projector)
    direction=torch.nn.functional.normalize((projector@vh[:,0,:,None]).squeeze(-1),dim=1)
    sign=direction.gather(1,direction.abs().argmax(1,keepdim=True)).sign()
    return direction*sign


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    for k in ('collection','support','checkpoint','output'):
        cli.add_argument('--'+k,required=True,type=Path)
    cli.add_argument('--device',choices=['cuda:0','cpu'],default='cuda:0')
    a=cli.parse_args()
    torch.set_num_threads(2)
    read=lambda p:json.loads(p.read_text())
    reg=read(a.collection/'startup.json')
    status=read(a.collection/'status.json')
    support=read(a.support)
    spec=reg['specs'][0]['spec']
    _,case_hash=resolve_probe_cases(spec,[])
    if (case_hash is None or case_hash!=support['case_sha256'] or not support['training_only']
            or not status['status']=='COLLECTION_FINISHED' or sha(a.collection/'status.json')!=support['status_sha256']
            or sha(a.checkpoint)!=reg['source_sha256'] or reg['source_sha256']!=support['source_sha256']):
        raise ValueError('Expected a scored training-only launch and matching behavior checkpoint')
    rec,=[r for r in status['completed'] if r['branch']=='nominal_a']
    folder=a.collection/rec['spec']
    if rec['exit_code'] or sha(folder/'nominal_a/native_probe.pt')!=rec['probe_sha256']:
        raise ValueError('Nominal trace differs')
    source=_load_branch(folder,'nominal_a')
    if source['spec']!=spec:
        raise ValueError('Changed excitation')
    ck=torch.load(a.checkpoint,mmap=True,weights_only=False,map_location='cpu')
    meta=ck['metadata']['physical_imagination']
    paths=meta['checkpoint_paths']
    argv=read(a.checkpoint.parent/'startup.json')['argv']
    opt=lambda k:Path(argv[argv.index('--'+k)+1])
    _,adapter,template,_,_=load_physical_loop(archive=opt('archive'),safety_mapping=opt('safety-mapping'),
        physical_checkpoint=paths['physical'],ball_checkpoint=paths['ball'],task_checkpoint=paths['task'],
        action_ball_checkpoint=paths.get('action_ball'),execution_root=meta['g07_execution_root'],device=a.device)
    if meta['checkpoint_sha256']!=adapter.manifest['checkpoint_sha256']:
        raise ValueError('Executor or dynamics sources changed')
    raw=source['raw']
    cases=spec['contact_training_panel']['cases']
    launch=torch.tensor([c['launch']['position_m']+c['launch']['velocity_mps'] for c in cases])
    s=PhysicalImaginationStateV1.from_record(raw['start'],executor_template=template,launch=launch.to(a.device),device=a.device)
    ids=spec['lanes']
    old=torch.tensor(spec['request_directions'],device=a.device)
    if old[:,:12].count_nonzero() or not torch.allclose(old.norm(dim=1),torch.ones(len(ids),device=a.device)):
        raise ValueError('Expected independent unit-length arm directions')
    req=raw['request_before'].to(a.device)
    with torch.no_grad():
        def targets(u):
            c=s.clock_goal_context
            preview,_,_=adapter.executor.forward(s.observation,u,c['joint_default'][0],c['limits'],s.branch().executor_memory)
            return preview.final_joint_targets[ids,12:]
        recorded=raw['step_sequence'][0]['final_joint_targets'][ids,12:].to(a.device)
        torch.testing.assert_close(targets(req),recorded,rtol=0,atol=2e-6)
        basis=torch.eye(18,device=a.device)[12:]
        eps=spec['epsilon_request']
        actuator=torch.stack([(targets(req+eps*v)-targets(req-eps*v))/(2*eps) for v in basis],dim=-1)
    s0=s.physical[ids].clone().requires_grad_()
    fk=racket_state(s0,adapter.physical_model.geometry)
    j=torch.stack([torch.autograd.grad(fk[:,i].sum(),s0,retain_graph=i<2)[0][:,25:31] for i in range(3)],1)
    normal=fk[:,3:12].reshape(-1,3,3)[:,:,0].detach()
    total=j@actuator
    inactive=total.abs().amax((1,2))<1e-8
    design=torch.where(inactive[:,None,None],j,total)
    d=tangent_directions(design,normal,old[:,12:]).to(req)
    torch.testing.assert_close(d.norm(dim=1),torch.ones(len(ids),device=a.device),atol=1e-5,rtol=0)
    dot=(d*old[:,12:]).sum(-1)
    torch.testing.assert_close(dot,torch.zeros_like(dot),atol=1e-5,rtol=0)
    new=torch.zeros_like(old)
    new[:,12:]=d
    with torch.no_grad():
        offset=torch.zeros_like(req)
        offset[ids]=eps*new
        actual=(targets(req+offset)-targets(req-offset))
        displacement=(j@actual.unsqueeze(-1)).squeeze(-1)
        along=(displacement*normal).sum(-1)
        tangent=(displacement-along[:,None]*normal).norm(dim=1)
    updated=dict(spec,request_directions=new.tolist(),direction_design='training_prefix_tangent_orthogonal',
                 source_direction_seed=spec.get('direction_seed'))
    updated.pop('direction_seed',None)
    validate_spec(updated)
    a.output.mkdir(parents=True,exist_ok=False)
    dest=a.output/(rec['spec']+'.json')
    dest.write_text(json.dumps(updated,indent=2,allow_nan=False)+'\n')
    report=dict(case_sha256=case_hash,seed=reg['seed'],source_status_sha256=support['status_sha256'],
        source_checkpoint_sha256=reg['source_sha256'],source_nominal_sha256=rec['probe_sha256'],
        source_support_sha256=sha(a.support),spec_sha256=sha(dest),lanes=ids,
        normal_displacement_m=along.tolist(),tangential_displacement_m=tangent.tolist(),
        orthogonality_error=float(dot.abs().max()),no_current_drive_response=[lane for lane,empty in zip(ids,inactive.tolist()) if empty],
        scope='Same training launch cases, independent request directions. Endpoint FK of the executed target difference is a design proxy. Actual native response requires four recorded branches.')
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
