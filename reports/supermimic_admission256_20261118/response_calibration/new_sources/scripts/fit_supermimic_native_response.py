#!/usr/bin/env python3
"""One bounded GPU calibration of measured robot/ball action responses.

Task adaptation: measured final-command finite differences supplement native
point/multistep supervision. This is not an Actor update, a native scoring run,
or a passing evidence receipt. No counterfactual response is synthesized.
"""
import argparse,hashlib,json,subprocess,sys,time,traceback
from pathlib import Path
import torch
from quadsmash.rl.supermimic_native_response_data_v1 import load_response_data,normalized_response_loss
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2
from quadsmash.rl.supermimic_ball_residual_v2 import BallResidualV2
from quadsmash.rl.supermimic_action_ball_v1 import ActionConditionedBallV1,contact_neighborhood_pools
from quadsmash.rl.supermimic_ball_sequence_v1 import ball_transition,ball_error
from quadsmash.rl.supermimic_physical_dataset_v1 import valid_windows
from quadsmash.rl.supermimic_physical_loop_loading_v1 import tensor_digest,SOURCE_SHA
from scripts.fit_supermimic_action_ball import metrics,autoregressive_metrics,sha256


def dump(path,value):
    temporary=path.with_suffix(path.suffix+'.pending')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def forward(base,ball,sweep,state,target,other):
    return ball_transition(base,ball,state,target,*other,action_model=sweep)

def aligned_robot(prediction,native):
    same=(prediction[:,3:7]*native[:,3:7]).sum(-1,keepdim=True)>=0
    return torch.cat((prediction[:,:3],torch.where(same,prediction[:,3:7],-prediction[:,3:7]),prediction[:,7:49]),-1)


def robot_point_loss(prediction,native,scale):
    return ((aligned_robot(prediction,native)-native[:,:49])/scale[:49]).square().mean()


def response_summary(base,ball,sweep,pairs,lanes):
    p={k:v[lanes] for k,v in pairs.items()}
    other=[p[k] for k in ('previous_action','feet','present','elapsed','launch')]
    with torch.no_grad():
        out={b:forward(base,ball,sweep,p['state'],p['target_'+b],other)[0] for b in ('nominal_a','minus','plus')}
    true=p['next_plus']-p['next_minus'];pred=out['plus']-out['minus']
    command=(p['target_plus']-p['target_minus']).norm(dim=-1)
    rows=[]
    for i,lane in enumerate(lanes.tolist()):
        item=dict(lane=lane,final_command_pair_delta_rad=float(command[i]),errors={})
        for name,sl in (('joint_position',slice(13,31)),('joint_velocity',slice(31,49)),('root_angular_velocity',slice(10,13)),('ball_velocity',slice(52,55))):
            a=true[i,sl];b=pred[i,sl];an=float(a.norm());bn=float(b.norm())
            item['errors'][name]=dict(native_norm=an,predicted_norm=bn,delta_error_norm=float((b-a).norm()),
                cosine=float((a*b).sum()/(an*bn)) if an>1e-7 and bn>1e-7 else None,
                amplitude_ratio=bn/an if an>1e-7 else None,native_delta=a.tolist(),predicted_delta=b.tolist())
        rows.append(item)
    return dict(rows=rows,count=len(rows),saturated_count=int((command<1e-6).sum()),
                scope='all declared lanes reported; zero responses are not successful gradient probes')


def main(a):
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    a._output_created=True
    if a.updates!=256:raise ValueError('This diagnostic has one predeclared256-update budget; no sweep')
    if not torch.cuda.is_available():raise RuntimeError('GPU required for model fitting')
    torch.set_num_threads(2);torch.manual_seed(20261118);torch.cuda.set_device(0);torch.cuda.init();torch.cuda.reset_peak_memory_stats()
    device=torch.device('cuda:0');started=time.perf_counter()
    native,provenance=load_response_data(a.probe_directory,a.split_manifest)
    pairs={k:v.to(device) for k,v in native.items()}
    train=torch.tensor(provenance['train_lanes'],device=device);val=torch.tensor(provenance['validation_lanes'],device=device)
    source=torch.load(a.physical_checkpoint,map_location='cpu',weights_only=True)
    ball_source=torch.load(a.ball_checkpoint,map_location='cpu',weights_only=True)
    sweep_source=torch.load(a.action_ball_checkpoint,map_location='cpu',weights_only=True)
    if source['schema']!=PhysicalWorldV2.schema or ball_source['schema']!=BallResidualV2.schema or sweep_source['schema']!=ActionConditionedBallV1.schema:
        raise ValueError('Explicit model schema mismatch')
    contract=source['source_manifest']['task_contract']
    if provenance['task_contract']!=contract:raise ValueError('Native response/physical model task contract mismatch')
    if sweep_source['physical_sha256']!=sha256(a.physical_checkpoint) or sweep_source['legacy_ball_sha256']!=sha256(a.ball_checkpoint):
        raise ValueError('Action-ball candidate belongs to a different base model')
    package=torch.load(a.packed_data,map_location='cpu',weights_only=True);cpu=package['tensors']
    manifest=package['source_manifest']
    if manifest['task_contract']!=contract or manifest['source_sha256']!=SOURCE_SHA or manifest.get('policy_execution')!=provenance['policy_execution']:
        raise ValueError('Native archive/probe source or execution differs')
    receipt=json.loads((a.packed_data.parent/'completed.json').read_text())
    if receipt['sha256']!=sha256(a.packed_data) or not bool(package['first_done'].all()):raise ValueError('Packed archive incomplete or changed')
    base=PhysicalWorldV2(contract['asset']['sha256']).to(device);base.load_state_dict(source['model'],strict=True)
    ball=BallResidualV2().to(device);ball.load_state_dict(ball_source['model'],strict=True)
    sweep=ActionConditionedBallV1().to(device);sweep.load_state_dict(sweep_source['model'],strict=True)
    if sweep_source['geometry_hashes']!=sweep.preview.geometry_hashes:raise ValueError('Racket geometry changed')
    base.requires_grad_(False);ball.eval().requires_grad_(False);base.body.requires_grad_(True);sweep.requires_grad_(True)
    contact_hash=tensor_digest(base.contact.state_dict().items());ball_hash=tensor_digest(ball.state_dict().items())
    normalization_hash=tensor_digest((k,v) for k,v in base.state_dict().items() if not k.startswith('body.') and not k.startswith('contact.'))
    before_base=tensor_digest(base.body.state_dict().items());before_sweep=tensor_digest(sweep.state_dict().items())
    needed=('state','next_state','targets','previous_action','feet','present','elapsed','event_state','next_event_state','episode','terminated','timeout')
    data={k:cpu[k].to(device) for k in needed}
    suite=Path(contract['suite']['path'])
    if not suite.is_absolute():suite=Path(__file__).resolve().parents[1]/suite
    if sha256(suite)!=contract['suite']['sha256']:raise ValueError('Task suite changed')
    cases=json.loads(suite.read_text())['cases'];launch=torch.tensor([c['launch']['position_m']+c['launch']['velocity_mps'] for c in cases],device=device)
    edges=(cpu['next_event_state'][...,1]>cpu['event_state'][...,1])|(cpu['next_event_state'][...,6]>cpu['event_state'][...,6])
    eligible=torch.zeros_like(edges);valid=valid_windows(cpu['episode'],cpu['terminated'],cpu['timeout'],16)
    eligible[:len(valid)]=valid;allowed=torch.zeros(128,dtype=torch.bool);allowed[train.cpu()]=True;eligible&=allowed[None]
    pools=contact_neighborhood_pools(edges,cpu['episode'],cpu['terminated'],cpu['timeout'],eligible)
    if any(len(pool)==0 for pool in pools.values()):raise ValueError('Missing source event windows')
    config=dict(argv=sys.argv,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        updates=256,batch=128,window_steps=16,control_dt_s=.01,lr=2e-4,seed=20261118,actor_updates=0,critic_updates=0,
        provenance=provenance,source_sha256={k:sha256(v) for k,v in dict(physical=a.physical_checkpoint,ball=a.ball_checkpoint,action_ball=a.action_ball_checkpoint,packed=a.packed_data).items()},
        optimized='existing robot body and action-conditioned ball only; original schemas/normalization/contact/legacy ball frozen',
        loss='mean16(robot normalized state loss + physical-unit ball error) + measured pair point loss + per-radian robot/ball finite-difference loss; all unit weights',
        sampling=dict(contact=64,pre_contact=32,post_contact=16,background=16),
        validation='original32 heldout lanes excluded from every fitting term; one time/direction is not complete physical admission',
        mode_switch='raw finite-difference transition supervision permitted; never called a smooth Jacobian target',
        reference='failed candidate remains diagnostic; no actor resumes it')
    dump(a.output/'config.json',config)
    initial=dict(legacy=metrics(base,ball,None,data,cpu,val.tolist(),launch),candidate=metrics(base,ball,sweep,data,cpu,val.tolist(),launch),
                 response=response_summary(base,ball,sweep,pairs,val))
    dump(a.output/'validation_initial.json',initial)
    parameters=list(base.body.parameters())+list(sweep.parameters());opt=torch.optim.Adam(parameters,lr=2e-4)
    generator=torch.Generator().manual_seed(20261118)
    other=[pairs[k][train] for k in ('previous_action','feet','present','elapsed','launch')]
    for update in range(256):
        begin=time.perf_counter()
        selected=torch.cat([pools[k][torch.randint(len(pools[k]),(n,),generator=generator)] for k,n in config['sampling'].items()]).to(device)
        t,l=selected.unbind(1);state=data['state'][t,l];feet=data['feet'][t,l].float();present=data['present'][t,l]
        ar=state.new_zeros(())
        for j in range(16):
            state,feet,present=forward(base,ball,sweep,state,data['targets'][t+j,l],[data['previous_action'][t+j,l],feet,present,data['elapsed'][t+j,l],launch[l]])
            truth=data['next_state'][t+j,l]
            ar=ar+(robot_point_loss(state,truth,base.delta_scale)+ball_error(state[:,49:],truth[:,49:]))/16
        predicted={b:forward(base,ball,sweep,pairs['state'][train],pairs['target_'+b][train],other)[0] for b in ('minus','plus')}
        pm,pp=predicted['minus'],predicted['plus'];nm,np=pairs['next_minus'][train],pairs['next_plus'][train]
        point=sum((robot_point_loss(predicted[b],pairs['next_'+b][train],base.delta_scale)+ball_error(predicted[b][:,49:],pairs['next_'+b][train,49:])) for b in ('minus','plus'))/2
        robot_response,support=normalized_response_loss(aligned_robot(pm,nm),aligned_robot(pp,np),nm[:,:49],np[:,:49],pairs['target_minus'][train],pairs['target_plus'][train],base.delta_scale[:49])
        ball_response,_=normalized_response_loss(pm[:,52:55],pp[:,52:55],nm[:,52:55],np[:,52:55],pairs['target_minus'][train],pairs['target_plus'][train],state.new_full((3,),2.))
        if support['supported']==0:raise RuntimeError('No executable native response in calibration set')
        loss=ar+point+robot_response+ball_response
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite native-response loss')
        opt.zero_grad(set_to_none=True);loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(parameters,10.,error_if_nonfinite=True);opt.step();torch.cuda.synchronize()
        row=dict(update=update+1,loss=float(loss.detach()),autoregressive=float(ar.detach()),point=float(point.detach()),robot_response=float(robot_response.detach()),ball_response=float(ball_response.detach()),gradient=float(grad),pair_support=support,seconds=time.perf_counter()-begin,peak_gpu_bytes=torch.cuda.max_memory_allocated(),actor_updates=0)
        with (a.output/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if update%16==0 or update==255:print(json.dumps(row),flush=True)
    if tensor_digest(base.contact.state_dict().items())!=contact_hash or tensor_digest(ball.state_dict().items())!=ball_hash:
        raise RuntimeError('Frozen contact/legacy ball changed')
    if tensor_digest((k,v) for k,v in base.state_dict().items() if not k.startswith('body.') and not k.startswith('contact.'))!=normalization_hash:
        raise RuntimeError('Frozen physical normalization/geometry changed')
    base_path=a.output/'world.pt';ball_path=a.output/'action_ball_candidate.pt'
    clean_source={k:v for k,v in source.items() if 'optimizer' not in k}
    torch.save(dict(clean_source,model=base.state_dict(),world_updates=int(source.get('world_updates',0))+256,
                    response_fit_updates=256,response_fit_config=config,response_optimizer=opt.state_dict(),
                    optimizer_contract='new combined robot-body/action-ball optimizer; old standalone optimizer states deliberately not reused'),base_path)
    new_ball=dict({k:v for k,v in sweep_source.items() if 'optimizer' not in k},model=sweep.state_dict(),
                  physical_sha256=sha256(base_path),updates=int(sweep_source['updates'])+256,response_fit_config=config)
    torch.save(new_ball,ball_path)
    final=dict(candidate=metrics(base,ball,sweep,data,cpu,val.tolist(),launch),response=response_summary(base,ball,sweep,pairs,val),
               autoregressive=autoregressive_metrics(base,ball,sweep,data,cpu,val.tolist(),launch))
    dump(a.output/'validation_final.json',final)
    admission={k:final['candidate'][k]['velocity_rmse']<=initial['legacy'][k]['velocity_rmse'] for k in initial['legacy']}
    dump(a.output/'completed.json',dict(status='FIT_COMPLETED_NOT_ADMITTED',world_response_updates=256,actor_updates=0,critic_updates=0,
        world_sha256=sha256(base_path),action_ball_sha256=sha256(ball_path),
        robot_parameters_changed=tensor_digest(base.body.state_dict().items())!=before_base,
        action_ball_parameters_changed=tensor_digest(sweep.state_dict().items())!=before_sweep,
        adjacent_step_non_degradation=admission,native_admission=False,
        elapsed_s=time.perf_counter()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        scope='one bounded calibrated candidate; no optimizer/policy promotion; independent Recovery and broader contact checks still required'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('probe-directory','split-manifest','physical-checkpoint','ball-checkpoint','action-ball-checkpoint','packed-data','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--updates',type=int,default=256)
    args=parser.parse_args()
    try:main(args)
    except BaseException as e:
        if getattr(args,'_output_created',False):
            dump(args.output/'failure.json',dict(type=type(e).__name__,message=str(e),actor_updates=0))
            (args.output/'failure_traceback.txt').write_text(traceback.format_exc())
        raise
