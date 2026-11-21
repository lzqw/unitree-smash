"""Joint multi-horizon Body calibration on signed commands, natural flight and Recovery."""
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from evaluate_supermimic_contact_rollout import sha
from fit_supermimic_mixed_body_rollout import event_pools
from fit_supermimic_paired_racket_center import take_windows, validate
from quadsmash.rl.supermimic_action_identification_dataset_v1 import _load_branch, _same_contract
from quadsmash.rl.supermimic_native_pair_capture_v1 import BRANCHES
from quadsmash.rl.supermimic_ball_sequence_v1 import ball_transition
from quadsmash.rl.supermimic_physical_dataset_v1 import encode_physical
from quadsmash.rl.supermimic_physical_run_guard_v1 import atomic_torch_save
from quadsmash.rl.supermimic_racket_dynamics_v1 import racket_state, robot_loss

CHECKS = (2, 4, 8, 16)


def fields(state, geometry):
    fk = racket_state(state, geometry)
    return torch.cat((fk[:,:3],fk[:,3:12].reshape(-1,3,3)[:,:,0],fk[:,12:]),1)


def signed_rollout(world,ball,win,launch):
    state,feet,present=(win[k][0] for k in ('state','feet','present'))
    for i in range(len(win['targets'])):
        state,feet,present=ball_transition(world,ball,state,win['targets'][i],win['previous_action'][i],
                                          feet,present,win['elapsed'][i],launch)
        state=torch.cat((state[:,:49],state[:,49:].detach()),1)
        yield state


def development_window(root,task,execution,excluded):
    meta=json.loads((root/'status.json').read_text())
    records=meta['completed']
    if meta['status']!='COLLECTION_FINISHED' or len(records)!=4 or len({r['spec'] for r in records})!=1 or {r['branch'] for r in records}!=set(BRANCHES):
        raise ValueError('Incomplete development pair')
    runs={}
    for r in records:
        folder=root/r['spec']
        if r['exit_code'] or sha(folder/r['branch']/'native_probe.pt')!=r['probe_sha256']:
            raise ValueError('Development archive changed')
        runs[r['branch']]=_load_branch(folder,r['branch'])
    _same_contract(runs)
    spec=runs['plus']['spec']
    ids=spec['lanes']
    if (spec['data_role']!='development_action_response' or len(ids)!=len(set(ids)) or not set(ids)<=set(excluded)
            or runs['plus']['task_contract']!=task or runs['plus']['policy_execution']!=execution):
        raise ValueError('Unexpected held-out physical cases')
    result={}
    for sign in ('minus','plus'):
        raw=runs[sign]['raw']
        initial=raw['start']['before']['physical']
        seq=raw['step_sequence']
        if len(seq)!=16:
            raise ValueError('Unexpected native development horizon')
        result[sign]=dict(state=encode_physical(initial)[ids][None],feet=initial['foot_contact'][ids][None],
            present=initial['ball_present'][ids][None],targets=torch.stack([s['final_joint_targets'][ids] for s in seq]),
            previous_action=torch.stack([initial['final_action'][ids],*[s['final_action'][ids] for s in seq[:-1]]]),
            next_state=torch.stack([s['next_state'][ids] for s in seq]),
            elapsed=torch.stack([initial['elapsed_s'][ids]+i*task['control_dt_s'] for i in range(16)]))
    torch.testing.assert_close(result['minus']['state'],result['plus']['state'],rtol=0,atol=0)
    return ids,result


def recorded_body_inputs(raw, lanes, dt):
    initial=raw['start']['before']['physical']
    steps=raw['step_sequence']
    if len(steps)!=16 or any(step['done'][lanes].any() for step in steps):
        raise ValueError('Expected an uninterrupted 16-step native trajectory')
    elapsed=torch.stack([initial['elapsed_s'][lanes],*[step['elapsed_s'][lanes] for step in steps]])
    torch.testing.assert_close(elapsed.diff(dim=0),torch.full_like(elapsed[:-1],dt),atol=2e-5,rtol=0)
    return dict(state=encode_physical(initial)[lanes][None],feet=initial['foot_contact'][lanes][None],
        present=initial['ball_present'][lanes][None],elapsed=elapsed[:-1],
        targets=torch.stack([step['final_joint_targets'][lanes] for step in steps]),
        previous_action=torch.stack([initial['final_action'][lanes],*[step['final_action'][lanes] for step in steps[:-1]]]),
        next_state=torch.stack([step['next_state'][lanes] for step in steps]))


def additional_body_pairs(collection,support_path,task,execution,stage_hash,excluded):
    from declare_supermimic_contact_training_cases import resolve_probe_cases
    read=lambda p:json.loads(p.read_text())
    support=read(support_path)
    status=read(collection/'status.json')
    spec=read(collection/'startup.json')['specs'][0]['spec']
    suite=task['suite']
    if sha(Path(suite['path']))!=suite['sha256']:
        raise ValueError('Changed original launch suite')
    cases,case_hash=resolve_probe_cases(spec,read(Path(suite['path']))['cases'])
    if (not case_hash or support['case_sha256']!=case_hash or not support['training_only']
            or support['source_sha256']!=stage_hash or support['status_sha256']!=sha(collection/'status.json')
            or support['policy_training_admitted'] or support['contact_receipt'] or support['counts'].get('nominal_mismatch',0)):
        raise ValueError('Different scored native training panel or policy')
    lanes=spec['lanes']
    if set(lanes)&set(excluded) or len(lanes)!=len(set(lanes)):
        raise ValueError('Reserved or repeated training lanes')
    rows=status['completed']
    if (status['status']!='COLLECTION_FINISHED' or len(rows)!=4 or len({r['spec'] for r in rows})!=1
            or {r['branch'] for r in rows}!=set(BRANCHES)):
        raise ValueError('Incomplete native sign or repeat')
    runs={}
    for row in rows:
        folder=collection/row['spec']
        if row['exit_code'] or sha(folder/row['branch']/'native_probe.pt')!=row['probe_sha256']:
            raise ValueError('Changed native trajectory')
        runs[row['branch']]=_load_branch(folder,row['branch'])
    _same_contract(runs)
    expected=dict(task,contact_training_only=True,contact_training_panel_sha256=case_hash)
    for run in runs.values():
        if run['task_contract']!=expected or run['policy_execution']!=execution or run['spec']!=spec:
            raise ValueError('Changed physics, execution or registered excitation')
    seq={b:recorded_body_inputs(run['raw'],lanes,task['control_dt_s']) for b,run in runs.items()}
    for k,x in seq['nominal_a'].items():
        torch.testing.assert_close(x,seq['nominal_b'][k],rtol=0,atol=0)
    for b in ('minus','plus'):
        for k in ('state','feet','present','elapsed','previous_action'):
            torch.testing.assert_close(seq[b][k][0],seq['nominal_a'][k][0],rtol=0,atol=0)
    launch=torch.tensor([cases[i]['launch']['position_m']+cases[i]['launch']['velocity_mps'] for i in lanes])
    return lanes,{b:seq[b] for b in ('minus','plus')},launch,dict(support_sha256=sha(support_path),
        status_sha256=support['status_sha256'],case_sha256=case_hash,case_ids=[cases[i]['case_id'] for i in lanes],
        no_contact_cases=support['counts'].get('no_contact',0))


def merged(signed,indices,device):
    return {k:torch.cat([signed[b][k][:,indices] for b in ('minus','plus')],1).to(device) for k in signed['minus']
            if k in ('state','feet','present','elapsed','targets','previous_action','next_state')}


def paired_terms(world,ball,window,launch,scales,*,training,arm_rows=None,arm_scales=None):
    total=launch.new_zeros(())
    scores={}
    physical_scale=launch.new_tensor([.03]*3+[1.]*3+[1.]*3+[3.]*3)
    for h,state in enumerate(signed_rollout(world,ball,window,launch),1):
        if h not in CHECKS:
            continue
        truth=window['next_state'][h-1]
        got=fields(state,world.geometry)
        actual=fields(truth,world.geometry)
        a,b=got.chunk(2); m,p=actual.chunk(2)
        err=(b-a)-(p-m)
        body=robot_loss(state,truth)
        absolute=F.smooth_l1_loss((got-actual)/physical_scale,torch.zeros_like(got))
        signed=F.smooth_l1_loss(err/scales[h],torch.zeros_like(err))
        if arm_scales is not None:
            q_minus,q_plus=state[:,arm_rows].chunk(2)
            measured_minus,measured_plus=truth[:,arm_rows].chunk(2)
            arm_error=(q_plus-q_minus)-(measured_plus-measured_minus)
            arm_loss=F.smooth_l1_loss(arm_error/arm_scales[h],torch.zeros_like(arm_error))
            signed=.5*(signed+arm_loss)
        gyro=F.smooth_l1_loss(state[:,10:13],truth[:,10:13],beta=.1)
        total=total+body+absolute+gyro+signed
        if not training:
            scores[h]=dict(center_rmse_m=float((got[:,:3]-actual[:,:3]).square().mean().sqrt()),
                body_rmse=float((state[:,:49]-truth[:,:49]).square().mean().sqrt()), gyro_mae=float((state[:,10:13]-truth[:,10:13]).abs().mean()))
            if arm_scales is not None:
                scores[h]['arm_position_signed_rmse_rad']=float(arm_error[:,:6].square().mean().sqrt())
                scores[h]['arm_velocity_signed_rmse_rad_s']=float(arm_error[:,6:].square().mean().sqrt())
            for i,name in enumerate(('center','normal','linear','angular')):
                pred=(b-a)[:,3*i:3*i+3];target=(p-m)[:,3*i:3*i+3]
                scores[h][name+'_signed_rmse']=float((pred-target).square().mean().sqrt())
                supported=target.norm(dim=1)>1.e-7
                scores[h][name+'_supported']=int(supported.sum())
                if supported.any():
                    scores[h][name+'_gain']=float((pred[supported].norm(dim=1)/target[supported].norm(dim=1)).median())
                    scores[h][name+'_cosine']=float(F.cosine_similarity(pred[supported],target[supported]).median())
    return total/len(CHECKS),scores


@torch.no_grad()
def center_timing(world, ball, window, launches, lanes):
    """A positive offset compares a later model prediction with the native position at t."""
    centers=[]
    actual=[]
    components={}
    for i,state in enumerate(signed_rollout(world,ball,window,launches)):
        centers.append(fields(state,world.geometry)[:,:3])
        truth=window['next_state'][i]
        actual.append(fields(truth,world.geometry)[:,:3])
        if i+1 in CHECKS:
            # Replace pose only at readout. The next tick still uses the free rollout.
            components[i+1]={}
            for name,parts in (('predicted',()),('native_root_position',(slice(0,3),)),
                               ('native_root_orientation',(slice(3,7),)),('native_joints',(slice(13,31),)),
                               ('native_root_pose',(slice(0,7),)),('native_all_poses',(slice(0,7),slice(13,31)))):
                probe=state.clone()
                for part in parts:
                    probe[:,part]=truth[:,part]
                center=fields(probe,world.geometry)[:,:3] if parts else centers[-1]
                if name=='native_all_poses':
                    torch.testing.assert_close(center,actual[-1],rtol=0,atol=0)
                lo,hi=(center-actual[-1]).chunk(2)
                components[i+1][name]=dict(midpoint_rmse_m=float(((hi+lo)/2).square().mean().sqrt()),
                                            difference_rmse_m=float((hi-lo).square().mean().sqrt()))
    minus,plus=torch.stack(centers).chunk(2,dim=1)
    nominal_minus,nominal_plus=torch.stack(actual).chunk(2,dim=1)
    # Keep exactly the same interior native ticks and the same cases at all offsets.
    ticks=torch.arange(2,len(centers)-2,device=minus.device)
    offsets=(-2,-1,0,1,2)
    result={}
    for name,prediction,target in (
        ('midpoint',(plus+minus)/2,(nominal_plus+nominal_minus)/2),
        ('difference',plus-minus,nominal_plus-nominal_minus)):
        errors=torch.stack([(prediction[ticks+offset]-target[ticks]).square().mean((0,2)) for offset in offsets])
        result[name]=dict(rmse_m_by_offset={s:float(e.mean().sqrt()) for s,e in zip(offsets,errors,strict=True)},
                         per_lane_rmse_m={lane:row for lane,row in zip(lanes,errors.sqrt().T.tolist(),strict=True)})
    return dict(native_ticks=(ticks+1).tolist(),model_offsets=list(offsets),fk_pose_substitutions=components,**result)


def fit_candidate(args,world,ball,data,launch,eligible,train,receipt,parent,fit,task,execution):
    lanes,signed=train[0]
    reg=json.loads((args.training_collection.parent/'preregistration.json').read_text())
    spec=json.loads((args.training_collection/'startup.json').read_text())['specs'][0]
    if (set(lanes)!=set(spec['spec']['lanes']) or reg['spec_sha256']!=spec['sha256'] or
            reg['actor_checkpoint_sha256']!=receipt['source_checkpoint_sha256'] or
            set(lanes)&set(reg['held_out_lanes'])):
        raise ValueError('Expected every registered training lane and an independent validation set')
    excluded=(~eligible).nonzero().flatten().tolist()
    training_launch=launch[lanes]
    original_lanes=list(lanes)
    extra=None
    if args.additional_training:
        ids,additional,new_launch,extra=additional_body_pairs(args.additional_training,args.additional_support,task,
                                                               execution,fit['stage_checkpoint_sha256'],excluded)
        for sign in ('minus','plus'):
            signed[sign]={k:torch.cat((signed[sign][k][:v.shape[0]],v),dim=1) for k,v in additional[sign].items()}
        training_launch=torch.cat((training_launch,new_launch))
        lanes=lanes+ids
    held,signed_dev=development_window(args.paired_development,task,execution,excluded)
    rng=torch.Generator().manual_seed(61037)
    devwin=merged(signed_dev,torch.arange(len(held)),args.device)
    devlaunch=launch[held].repeat(2,1).to(args.device)
    if args.audit_center_timing:
        index=torch.arange(len(lanes))
        # Native lane numbers repeat across independently seeded training suites.
        case_keys=original_lanes if extra is None else (
            [f"{receipt['status_sha256']}:{lane}" for lane in original_lanes]
            + [f"{extra['status_sha256']}:{lane}" for lane in ids])
        report=dict(training=center_timing(world,ball,merged(signed,index,args.device),training_launch.repeat(2,1).to(args.device),case_keys),
                    development=center_timing(world,ball,devwin,devlaunch,held),source_world_sha256=fit['online_world_sha256'],
                    training_status_sha256=receipt['status_sha256'],development_status_sha256=sha(args.paired_development/'status.json'),
                    additional_training=extra,training_cases=len(case_keys),development_cases=len(held),
                    optimizer_updates=0,final_holdout_opened=False)
        for channel in ('midpoint','difference'):
            a=report['training'][channel]['rmse_m_by_offset']
            b=report['development'][channel]['rmse_m_by_offset']
            offset=min(a,key=lambda o:(a[o],abs(o)))
            offsets=report['development']['model_offsets']
            errors=report['development'][channel]['per_lane_rmse_m'].values()
            gains=[row[offsets.index(0)]-row[offsets.index(offset)] for row in errors]
            report[channel+'_test']=dict(training_selected_offset=offset,dev_rmse_at_selected=b[offset],dev_rmse_at_zero=b[0],
                                         dev_cases_improved=sum(g>0 for g in gains),dev_cases_tied=sum(g==0 for g in gains))
        args.output.mkdir(parents=True)
        (args.output/'timing.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        return
    with torch.no_grad():
        scales={}
        for h in CHECKS:
            m,p=[fields(signed[b]['next_state'][h-1].to(args.device),world.geometry) for b in ('minus','plus')]
            scales[h]=(p-m).reshape(-1,4,3).square().mean((0,2)).sqrt().clamp_min(1.e-5).repeat_interleave(3)
    pools=event_pools(data,eligible,16,True)
    long_pools=event_pools(data,eligible,64,True) if args.long_body_anchor else {}
    long_rng=torch.Generator().manual_seed(9221)
    valid=event_pools(data,~eligible,64,True)
    val={k:tuple(v[torch.randperm(len(v),generator=rng)[:24]].T) for k,v in valid.items()}
    if any(not len(v) for v in pools.values()) or any(not len(x) for x,_ in val.values()):
        raise ValueError('Missing event or recovery windows')
    args.output.mkdir(parents=True)
    world.body.requires_grad_(not args.arm_pose_velocity_readout)
    readout_rows=None
    if args.arm_pose_velocity_readout:
        joints=parent['source_manifest']['joint_names']
        if len(joints)!=18 or len(set(joints))!=18:
            raise ValueError('Expected 18 unique native joint names')
        arm=[joints.index('joint'+str(i)) for i in range(1,7)]
        layer=world.body.net[-1]
        layer.requires_grad_(True)
        readout_rows=[offset+j for offset in (13,31) for j in arm]
        frozen_rows=torch.ones_like(layer.bias,dtype=torch.bool)
        frozen_rows[readout_rows]=False
        original_rows=[x[frozen_rows].detach().clone() for x in (layer.weight,layer.bias)]
    trainable=[v for v in world.body.parameters() if v.requires_grad]
    opt=torch.optim.Adam(trainable,lr=1e-6)
    arm_scales=None
    if args.signed_arm_supervision:
        arm_scales={}
        # Scale each joint only with training traces, including the misses. The floor follows
        # the measured state normalization used by the deployed Body's own residual readout.
        floor=(.01*world.delta_scale[readout_rows]).clamp_min(1e-6)
        for h in CHECKS:
            diff=signed['plus']['next_state'][h-1,:,readout_rows]-signed['minus']['next_state'][h-1,:,readout_rows]
            arm_scales[h]=torch.maximum(diff.to(floor).square().mean(0).sqrt(),floor)
        all_training=merged(signed,torch.arange(len(lanes)),args.device)
    @torch.no_grad()
    def score():
        out=dict(paired=paired_terms(world,ball,devwin,devlaunch,scales,training=False,
                 arm_rows=readout_rows,arm_scales=arm_scales)[1],natural=validate(world,ball,data,val,launch,args.device))
        if arm_scales is not None:
            out['training_paired']=paired_terms(world,ball,all_training,training_launch.repeat(2,1).to(args.device),
                scales,training=False,arm_rows=readout_rows,arm_scales=arm_scales)[1]
        return out
    base=score()
    initial={k:v.detach().clone() for k,v in world.body.named_parameters()}
    best=1.;steps=0;history=[]
    selected=0
    def save():
        cpu_weights={k:v.detach().cpu() for k,v in world.state_dict().items()}
        atomic_torch_save(dict(parent,model=cpu_weights,calibration=dict(parent['calibration'],train_lanes=sorted(set(lanes)),additional_paired_training=extra,
             paired_body_source_world=fit['online_world_sha256'],paired_body_update=steps,arm_readout_rows=readout_rows,
             signed_arm_supervision=args.signed_arm_supervision),
             policy_training_admitted=False),args.output/'world.pt')
    save()
    for epoch in range(4):
        for ix in torch.randperm(len(lanes),generator=rng).split(9):
            steps+=1
            win=merged(signed,ix,args.device)
            loss,_=paired_terms(world,ball,win,training_launch[ix].repeat(2,1).to(args.device),scales,training=True,
                                arm_rows=readout_rows,arm_scales=arm_scales)
            rows=torch.cat([pool[torch.randint(len(pool),(n,),generator=rng)] for pool,n in
                            ((pools['natural'],24),(pools['racket'],6),(pools['recovery'],6))])
            natural={k:v.to(args.device) for k,v in take_windows(data,*rows.T,16).items()}
            other=loss.new_zeros(())
            for h,state in enumerate(signed_rollout(world,ball,natural,launch[rows[:,1]].to(args.device)),1):
                if h in CHECKS:
                    truth=natural['next_state'][h-1]
                    other+=robot_loss(state,truth)+F.smooth_l1_loss(state[:,10:13],truth[:,10:13],beta=.1)
                    other+=F.smooth_l1_loss(fields(state,world.geometry)[:,:3]/.03,fields(truth,world.geometry)[:,:3]/.03)
            anchor=sum((v-initial[k]).square().mean() for k,v in world.body.named_parameters())
            objective=.33*loss+.67*other/len(CHECKS)+.01*anchor
            opt.zero_grad(set_to_none=True);objective.backward()
            if long_pools:
                rows=torch.cat([pool[torch.randint(len(pool),(2,),generator=long_rng)] for pool in long_pools.values()])
                long={k:v.to(args.device) for k,v in take_windows(data,*rows.T,64).items()}
                penalty=loss.new_zeros(())
                for h,state in enumerate(signed_rollout(world,ball,long,launch[rows[:,1]].to(args.device)),1):
                    if h in (32,64):
                        truth=long['next_state'][h-1]
                        penalty+=robot_loss(state,truth)+F.smooth_l1_loss(state[:,10:13],truth[:,10:13],beta=.1)
                        penalty+=F.smooth_l1_loss(fields(state,world.geometry)[:,:3]/.03,fields(truth,world.geometry)[:,:3]/.03)
                (.5*penalty/2).backward()
            if readout_rows is not None:
                layer.weight.grad[frozen_rows]=0
                layer.bias.grad[frozen_rows]=0
            torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True)
            opt.step()
        if readout_rows is not None:
            for param,expected in zip((layer.weight,layer.bias),original_rows,strict=True):
                torch.testing.assert_close(param[frozen_rows],expected,atol=0,rtol=0)
        result=score()
        ratios=[result['paired'][h][k]/max(1.e-12,base['paired'][h][k]) for h in CHECKS
                for k in ('center_rmse_m','center_signed_rmse','normal_signed_rmse','linear_signed_rmse','angular_signed_rmse')]
        protected=max(result['natural'][event][h][k]/max(1.e-12,base['natural'][event][h][k])
                      for event in val for h in (32,64) for k in ('body_rmse','gyro_mae','center_mae','ball_velocity_rmse'))
        value=sum(ratios)/len(ratios)
        history.append(dict(epoch=epoch+1,updates=steps,score=value,max_long_ratio=protected,validation=result))
        if protected<=1 and value<best:
            selected=steps;best=value;save()
        (args.output/'development.json').write_text(json.dumps(history,indent=2,allow_nan=False)+'\n')
    report=dict(source_world_sha256=fit['online_world_sha256'],training=receipt,development_sha256=sha(args.paired_development/'status.json'),
        development_lanes=held,selected_update=selected,updates=steps,long_body_anchor=args.long_body_anchor,baseline=base,history=history,
        arm_readout_rows=readout_rows,signed_arm_scales={h:s.tolist() for h,s in arm_scales.items()} if arm_scales else None,
        trained_lanes_per_epoch=original_lanes,additional_training=extra,training_pairs_per_epoch=len(training_launch),selected_sha256=sha(args.output/'world.pt'),actor_updates=0,critic_updates=0,
        ball_updates=0,final_holdout_opened=False,validation_windows={k:torch.stack(v,1).tolist() for k,v in val.items()})
    (args.output/'fit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
