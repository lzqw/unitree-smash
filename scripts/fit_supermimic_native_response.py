#!/usr/bin/env python3
"""One bounded GPU calibration of measured robot/ball action responses.

Task adaptation: measured final-command finite differences supplement native
point/multistep supervision. This is not an Actor update, a native scoring run,
or a passing evidence receipt. No counterfactual response is synthesized.
"""
import argparse,copy,hashlib,json,subprocess,sys,time,traceback
from pathlib import Path
import torch
from quadsmash.rl.supermimic_native_response_data_v1 import load_response_data,measured_ball_response_support,normalized_response_loss
from quadsmash.rl.supermimic_ball_mode_supervision_v1 import observed_ball_mode_targets,observed_ball_mode_loss
from quadsmash.rl.supermimic_ball_residual_v1 import ball_geometry_features
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2
from quadsmash.rl.supermimic_ball_residual_v2 import BallResidualV2
from quadsmash.rl.supermimic_action_ball_v1 import ActionConditionedBallV1,RacketModeActionBallV2,ImpactTimeActionBallV3,MaterialInertiaActionBallV4,FrameAwareMaterialInertiaActionBallV5,FrameAwareMaterialInertiaActionBallV6,action_ball_from_checkpoint,migrate_material_inertia_v4,contact_neighborhood_pools
from quadsmash.rl.supermimic_ball_sequence_v1 import ball_transition,ball_error
from quadsmash.rl.supermimic_physical_dataset_v1 import valid_windows
from quadsmash.rl.supermimic_physical_loop_loading_v1 import tensor_digest,SOURCE_SHA
from scripts.fit_supermimic_action_ball import metrics,autoregressive_metrics,sha256
from quadsmash.rl.supermimic_response_fit_sources_v1 import load_history,combine_sources,draw_source_windows


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


def compatible_response_sources(reference,others):
    """Multiple measured times/heads share contracts and the full-lane split.

    Native states at different times need not match. Each source independently
    passes its four-branch same-prefix check before this identity comparison.
    """
    keys=('task_contract','policy_execution','actor_state_sha256','train_lanes',
          'validation_lanes','split_sha256')
    for other in others:
        for key in keys:
            if other.get(key)!=reference.get(key) or key not in reference:
                raise ValueError('Native response source contract differs: '+key)
    identities=[(x['spec']['head'],x['spec']['tick'],json.dumps(x['spec'],sort_keys=True))
                for x in (reference,*others)]
    if len(set(identities))!=len(identities):
        raise ValueError('Duplicate native response source')


def measured_foot_labels(directory,provenance):
    """Contact supervision from the already-verified native pair artifacts."""
    result={}
    for branch in ('minus','plus'):
        path=Path(directory)/branch/'native_probe.pt'
        if sha256(path)!=provenance['source_sha256'][branch]:
            raise ValueError('Native contact label source hash changed')
        feet=torch.load(path,map_location='cpu',weights_only=True)['raw']['next_feet']
        if feet.shape!=(128,4) or not torch.isfinite(feet).all() or not ((feet==0)|(feet==1)).all():
            raise ValueError('Expected observed original128 four-foot binary labels')
        result['next_feet_'+branch]=feet.float()
    return result


def foot_prediction_metrics(probability,observed):
    """Read-only boundary contact diagnostic; never a native success label."""
    if probability.shape!=observed.shape or probability.ndim!=2 or probability.shape[1]!=4:
        raise ValueError('Foot prediction/label dimensions differ')
    if not torch.isfinite(probability).all() or not ((probability>=0)&(probability<=1)).all():
        raise ValueError('Invalid predicted foot probabilities')
    if not ((observed==0)|(observed==1)).all():
        raise ValueError('Foot truth must be native binary labels')
    predicted=probability>=.5;truth=observed.bool()
    return dict(lanes=len(observed),boundary_labels=observed.numel(),
        diagnostic_threshold=.5,brier=float((probability-observed.float()).square().mean()),
        positives=int(truth.sum()),false_negative=int((truth&~predicted).sum()),
        false_positive=int((~truth&predicted).sum()),
        scope='recorded control-boundary contact, not substep force or task success')


def response_summary(base,ball,sweep,pairs,lanes):
    p={k:v[lanes] for k,v in pairs.items()}
    other=[p[k] for k in ('previous_action','feet','present','elapsed','launch')]
    with torch.no_grad():
        transition={b:forward(base,ball,sweep,p['state'],p['target_'+b],other) for b in ('nominal_a','minus','plus')}
        out={b:value[0] for b,value in transition.items()}
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
    foot={b:foot_prediction_metrics(transition[b][1],p['next_feet_'+b])
          for b in ('minus','plus') if 'next_feet_'+b in p}
    return dict(rows=rows,count=len(rows),saturated_count=int((command<1e-6).sum()),foot_contact=foot,
                scope='all declared lanes reported; zero responses are not successful gradient probes')


def native_ball_response_validation(base,ball,sweep,pair_sets,lanes):
    """Score measured ball response on lanes excluded from fitting.

    Only rows with a non-zero measured ball response are scored.  A validation
    set containing no such rows is explicitly unusable, rather than silently
    making a candidate look good because both predictions are zero.
    """
    errors=[]
    for pair in pair_sets:
        p={k:v[lanes] for k,v in pair.items()}
        other=[p[k] for k in ('previous_action','feet','present','elapsed','launch')]
        with torch.no_grad():
            minus=forward(base,ball,sweep,p['state'],p['target_minus'],other)[0]
            plus=forward(base,ball,sweep,p['state'],p['target_plus'],other)[0]
        measured=p['next_plus'][:,52:55]-p['next_minus'][:,52:55]
        predicted=plus[:,52:55]-minus[:,52:55]
        mask=measured.norm(dim=-1)>1e-7
        if mask.any():
            errors.append(((predicted[mask]-measured[mask])/2.).square().mean())
    if not errors:
        return float('inf'),0
    return float(torch.stack(errors).mean().item()),sum(int(x.numel()) for x in [
        (p['next_plus'][lanes,52:55]-p['next_minus'][lanes,52:55]).norm(dim=-1)>1e-7
        for p in pair_sets])


def main(a):
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    a._output_created=True
    if a.batch<128 or a.batch%128:raise ValueError('Model batch must be a positive multiple of the original128 quota unit')
    if a.updates!=256:raise ValueError('This diagnostic has one predeclared256-update budget; no sweep')
    if a.min_native_ball_validation_rows<1:raise ValueError('Minimum held-out ball-response support must be positive')
    if not torch.cuda.is_available():raise RuntimeError('GPU required for model fitting')
    torch.set_num_threads(2);torch.manual_seed(20261118);torch.cuda.set_device(0);torch.cuda.init();torch.cuda.reset_peak_memory_stats()
    device=torch.device('cuda:0');started=time.perf_counter()
    native,provenance=load_response_data(a.probe_directory,a.split_manifest)
    pairs={k:v.to(device) for k,v in native.items()}
    extra=[load_response_data(path,a.split_manifest) for path in a.extra_probe_directory]
    compatible_response_sources(provenance,[p for _,p in extra])
    pair_sets=[pairs]+[{k:v.to(device) for k,v in data.items()} for data,_ in extra]
    pair_provenance=[provenance]+[p for _,p in extra]
    pair_names=[str(a.probe_directory.resolve())]+[str(p.resolve()) for p in a.extra_probe_directory]
    if a.repair_foot_contact:
        if not any(p['spec']['head']=='recovery' for p in pair_provenance):
            raise ValueError('Foot-response repair requires measured Recovery pairs')
        for pair,name,p in zip(pair_sets,pair_names,pair_provenance):
            pair.update({k:v.to(device) for k,v in measured_foot_labels(name,p).items()})
    train=torch.tensor(provenance['train_lanes'],device=device);val=torch.tensor(provenance['validation_lanes'],device=device)
    native_ball_support=measured_ball_response_support(pair_sets,val)
    if native_ball_support['rows'] < a.min_native_ball_validation_rows:
        raise RuntimeError(
            'Insufficient held-out native ball-response support: '
            f"{native_ball_support['rows']} rows from {native_ball_support['unique_lanes']} lanes, "
            f"need at least {a.min_native_ball_validation_rows}. Collect probes at event-aligned ticks."
        )
    source=torch.load(a.physical_checkpoint,map_location='cpu',weights_only=True)
    ball_source=torch.load(a.ball_checkpoint,map_location='cpu',weights_only=True)
    sweep_source=torch.load(a.action_ball_checkpoint,map_location='cpu',weights_only=True)
    if source['schema']!=PhysicalWorldV2.schema or ball_source['schema']!=BallResidualV2.schema:
        raise ValueError('Explicit model schema mismatch')
    contract=source['source_manifest']['task_contract']
    if provenance['task_contract']!=contract:raise ValueError('Native response/physical model task contract mismatch')
    if sweep_source['physical_sha256']!=sha256(a.physical_checkpoint) or sweep_source['legacy_ball_sha256']!=sha256(a.ball_checkpoint):
        raise ValueError('Action-ball candidate belongs to a different base model')
    package=torch.load(a.packed_data,map_location='cpu',weights_only=True);cpu=package['tensors']
    manifest=package['source_manifest']
    archive_execution=manifest.get('policy_execution')
    execution_provenance='archive_manifest'
    if archive_execution is None:
        if a.policy_execution is None:
            raise ValueError(
                'Packed archive has no policy_execution. Supply an explicit '
                '--policy-execution sidecar bound to this archive source.'
            )
        sidecar=json.loads(a.policy_execution.read_text())
        if sidecar.get('source_sha256')!=manifest['source_sha256']:
            raise ValueError('Policy execution sidecar source SHA differs from packed archive')
        if sidecar.get('task_contract_sha256') != hashlib.sha256(
                json.dumps(manifest['task_contract'],sort_keys=True,separators=(',',':')).encode()
        ).hexdigest():
            raise ValueError('Policy execution sidecar task contract differs from packed archive')
        archive_execution=sidecar.get('policy_execution')
        execution_provenance='explicit_sidecar'
    if manifest['task_contract']!=contract or manifest['source_sha256']!=SOURCE_SHA or archive_execution!=provenance['policy_execution']:
        raise ValueError('Native archive/probe source or execution differs')
    receipt=json.loads((a.packed_data.parent/'completed.json').read_text())
    if receipt['sha256']!=sha256(a.packed_data) or not bool(package['first_done'].all()):raise ValueError('Packed archive incomplete or changed')
    base=PhysicalWorldV2(contract['asset']['sha256']).to(device);base.load_state_dict(source['model'],strict=True)
    ball=BallResidualV2().to(device);ball.load_state_dict(ball_source['model'],strict=True)
    material_contract=sweep_source.get('material_contract')
    sweep=action_ball_from_checkpoint(sweep_source).to(device)
    sweep.load_state_dict(sweep_source['model'],strict=True)
    if a.frame_aware_action_ball:
        if sweep_source['schema']!=MaterialInertiaActionBallV4.schema:
            raise ValueError('Frame-aware migration requires an explicit V4 material-action source')
        if 'material_contract' not in sweep_source:
            raise ValueError('Frame-aware migration requires the V4 native material contract')
        sweep_cls=FrameAwareMaterialInertiaActionBallV6 if a.analytic_impact_prior else FrameAwareMaterialInertiaActionBallV5
        sweep=sweep_cls(material_contract=sweep_source['material_contract']).to(device)
        sweep.load_state_dict(sweep_source['model'],strict=True)
    if a.condition_action_on_racket_mode:
        if sweep_source['schema']!=ActionConditionedBallV1.schema:
            raise ValueError('Explicit V1-to-V2 migration requires a V1 source')
        sweep=RacketModeActionBallV2().to(device)
        sweep.load_state_dict(sweep_source['model'],strict=True)
    if a.impact_time_state:
        if a.condition_action_on_racket_mode or sweep_source['schema']!=RacketModeActionBallV2.schema:
            raise ValueError('Explicit impact-time migration requires V2 source and no concurrent mode migration')
        sweep=ImpactTimeActionBallV3().to(device)
        sweep.load_state_dict(sweep_source['model'],strict=True)
    if a.native_material_receipt is not None:
        if a.condition_action_on_racket_mode or a.impact_time_state:
            raise ValueError('Native-material migration is a separate explicit V3 revision')
        from quadsmash.rl.supermimic_racket_material_contract_v1 import contract_from_native_receipt
        material_contract=contract_from_native_receipt(a.native_material_receipt)
        if material_contract['asset_sha256']!=contract['asset']['sha256']:raise ValueError('Native material source asset changed')
        sweep=migrate_material_inertia_v4(sweep,material_contract)
    if sweep_source['geometry_hashes']!=sweep.preview.geometry_hashes:raise ValueError('Racket geometry changed')
    base.requires_grad_(False);ball.eval().requires_grad_(False)
    if not a.freeze_physical_response:
        base.body.requires_grad_(True)
    sweep.requires_grad_(True)
    if a.repair_foot_contact:
        base.contact.requires_grad_(True)
    if a.repair_ball_gate:
        ball.gate.requires_grad_(True)
    contact_hash=tensor_digest(base.contact.state_dict().items());ball_hash=tensor_digest(ball.state_dict().items())
    ball_frozen_hash=tensor_digest((k,v) for k,v in ball.state_dict().items() if not k.startswith('gate.'))
    ball_gate_before=tensor_digest(ball.gate.state_dict().items())
    normalization_hash=tensor_digest((k,v) for k,v in base.state_dict().items() if not k.startswith('body.') and not k.startswith('contact.'))
    before_base=tensor_digest(base.body.state_dict().items());before_sweep=tensor_digest(sweep.state_dict().items())
    needed=('state','next_state','targets','previous_action','feet','next_feet','present','elapsed','event_state','next_event_state','episode','terminated','timeout')
    validation_data={k:cpu[k].to(device) for k in needed}
    historical=[];history_provenance=[]
    if a.historical_physical_data is not None:
        historical,history_provenance=load_history(a.historical_physical_data,contract,manifest['joint_names'],sha256(a.split_manifest))
    training_cpu,source_pools,source_ids=combine_sources(cpu,historical,needed,provenance['train_lanes'],16)
    data={k:v.to(device) for k,v in training_cpu.items()}
    suite=Path(contract['suite']['path'])
    if not suite.is_absolute():suite=Path(__file__).resolve().parents[1]/suite
    if sha256(suite)!=contract['suite']['sha256']:raise ValueError('Task suite changed')
    cases=json.loads(suite.read_text())['cases'];launch=torch.tensor([c['launch']['position_m']+c['launch']['velocity_mps'] for c in cases],device=device)
    contact_edges=(training_cpu['next_event_state'][...,1]>training_cpu['event_state'][...,1])
    frame_edges=(training_cpu['next_event_state'][...,6]>training_cpu['event_state'][...,6])
    edges=contact_edges|frame_edges
    config=dict(argv=sys.argv,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        updates=256,batch=a.batch,window_steps=16,control_dt_s=.01,lr=2e-4,seed=20261118,actor_updates=0,critic_updates=0,
        provenance=provenance,source_sha256={k:sha256(v) for k,v in dict(physical=a.physical_checkpoint,ball=a.ball_checkpoint,action_ball=a.action_ball_checkpoint,packed=a.packed_data).items()},
        optimized='existing robot body and action-conditioned ball; original schemas and normalization frozen',
        loss='mean16(robot normalized state loss + physical-unit ball error) + measured pair point loss + per-radian robot/ball finite-difference loss',
        ball_response_weight=float(a.ball_response_weight),
        sampling={k:n*(a.batch//128) for k,n in dict(contact=64,pre_contact=32,post_contact=16,background=16).items()},
        validation='original32 heldout lanes excluded from every fitting term; one time/direction is not complete physical admission',
        mode_switch='raw finite-difference transition supervision permitted; never called a smooth Jacobian target',
        reference='failed candidate remains diagnostic; no actor resumes it')
    config['native_ball_response_support']=native_ball_support
    config['minimum_native_ball_validation_rows']=int(a.min_native_ball_validation_rows)
    config['policy_execution_provenance']=execution_provenance
    config['policy_execution_sidecar']=str(a.policy_execution.resolve()) if a.policy_execution is not None else None
    config['historical_physical_sources']=history_provenance
    config['physical_source_sampling']='initial50/history25/history25' if historical else 'original-only'
    config['historical_scope']='measured physical transitions only; unknown policy identities are not filled, no critic labels or actor starts; every source excludes the same32 fitting lanes'
    config['validation_scope']='unchanged original archive30 event-neighbourhood cases and original native paired-response holdouts; no history-based checkpoint selection'
    config['native_response_sources']=[dict(path=name,provenance=p) for name,p in zip(pair_names,pair_provenance)]
    config['native_response_aggregation']='equal mean over independently verified measured times/heads; identical96/32 lane exclusion in every source'
    config['repair_ball_gate']=bool(a.repair_ball_gate)
    config['action_ball_schema']=sweep.schema
    config['action_ball_source_schema']=sweep_source['schema']
    config['action_ball_migration']=bool(a.condition_action_on_racket_mode or a.impact_time_state or a.native_material_receipt is not None)
    config['material_contract']=getattr(sweep,'material_contract',None)
    if config['material_contract'] is not None:
        config['optimized']+=('; zero-initialized' if a.native_material_receipt is not None else '; existing')+' native-material/articulated-inertia residual head; original material/geometry buffers frozen'
        config['mechanical_scope']='Normal-impact scale and authored free-floating inertia are causal neural features, not a PhysX contact solver. No incoming ball-spin label or future native state is used as forward input.'
    config['impact_time_state']=bool(a.impact_time_state)
    config['impact_time_approximation']='causal current-frame quadratic crossing; discrete first-bin selection, differentiable within-bin root and Hermite physical state; not native substep reconstruction' if a.impact_time_state else None
    config['action_ball_mode_semantics']='current learned racket probability, not native/future contact labels' if getattr(sweep,'requires_racket_probability',False) else 'legacy additive geometry-supported correction'
    config['repair_foot_contact']=bool(a.repair_foot_contact)
    if a.repair_foot_contact:
        config['optimized']+='; existing foot contact head explicitly unfrozen against observed next-foot labels'
        config['loss']+=' + unit-weight native foot BCE on16-step recurrence and paired transitions (same term as main WM learner)'
    else:
        config['optimized']+='; foot contact head frozen'
    if a.repair_ball_gate:
        config['optimized']+='; legacy ball gate explicitly unfrozen, response/normalization still frozen'
        config['loss']+=' + unit-weight observed free/racket gate cross entropy on the same training-only recorded starts'
        config['mode_labels']='Native racket edge or measured free-flight-consistent residual; other transitions unknown. Supervision only, no label as forward input.'
    else:
        config['optimized']+='; legacy ball gate/response frozen'
    dump(a.output/'config.json',config)
    initial=dict(legacy=metrics(base,ball,None,validation_data,cpu,val.tolist(),launch),candidate=metrics(base,ball,sweep,validation_data,cpu,val.tolist(),launch),
                 response=response_summary(base,ball,sweep,pairs,val),
                 responses={name:response_summary(base,ball,sweep,data,val) for name,data in zip(pair_names,pair_sets)})
    dump(a.output/'validation_initial.json',initial)
    parameters=(list(base.body.parameters()) if not a.freeze_physical_response else [])+list(sweep.parameters())
    if a.repair_ball_gate:parameters+=list(ball.gate.parameters())
    if a.repair_foot_contact:parameters+=list(base.contact.parameters())
    opt=torch.optim.Adam(parameters,lr=2e-4)
    generator=torch.Generator().manual_seed(20261118)
    other_sets=[[data[k][train] for k in ('previous_action','feet','present','elapsed','launch')] for data in pair_sets]
    best_validation_score=float('inf')
    best_validation_update=0
    best_validation_rows=0
    best_model_state=None
    best_optimizer_state=None
    validation_history=[]
    for update in range(256):
        begin=time.perf_counter()
        selected,source_sampling=draw_source_windows(source_pools,config['sampling'],generator)
        selected=selected.to(device)
        t,l=selected.unbind(1);state=data['state'][t,l];feet=data['feet'][t,l].float();present=data['present'][t,l]
        ar=state.new_zeros(());contact_loss=state.new_zeros(())
        for j in range(16):
            state,feet,present=forward(base,ball,sweep,state,data['targets'][t+j,l],[data['previous_action'][t+j,l],feet,present,data['elapsed'][t+j,l],launch[l]])
            truth=data['next_state'][t+j,l]
            ar=ar+(robot_point_loss(state,truth,base.delta_scale)+ball_error(state[:,49:],truth[:,49:]))/16
            if a.repair_foot_contact:
                contact_loss=contact_loss+torch.nn.functional.binary_cross_entropy(feet,data['next_feet'][t+j,l].float())/16
        point=ar.new_zeros(());robot_response=ar.new_zeros(());ball_response=ar.new_zeros(())
        supports=[]
        for pair,other in zip(pair_sets,other_sets):
            transition={b:forward(base,ball,sweep,pair['state'][train],pair['target_'+b][train],other) for b in ('minus','plus')}
            predicted={b:x[0] for b,x in transition.items()}
            if a.repair_foot_contact:
                contact_loss=contact_loss+sum(torch.nn.functional.binary_cross_entropy(transition[b][1],pair['next_feet_'+b][train]) for b in ('minus','plus'))/(2*len(pair_sets))
            pm,pp=predicted['minus'],predicted['plus'];nm,np=pair['next_minus'][train],pair['next_plus'][train]
            point=point+sum((robot_point_loss(predicted[b],pair['next_'+b][train],base.delta_scale)+ball_error(predicted[b][:,49:],pair['next_'+b][train,49:])) for b in ('minus','plus'))/2
            robot_i,support=normalized_response_loss(aligned_robot(pm,nm),aligned_robot(pp,np),nm[:,:49],np[:,:49],pair['target_minus'][train],pair['target_plus'][train],base.delta_scale[:49])
            ball_i,_=normalized_response_loss(pm[:,52:55],pp[:,52:55],nm[:,52:55],np[:,52:55],pair['target_minus'][train],pair['target_plus'][train],state.new_full((3,),2.))
            if support['supported']==0:raise RuntimeError('No executable native response in calibration source')
            robot_response=robot_response+robot_i;ball_response=ball_response+ball_i;supports.append(support)
        point=point/len(pair_sets);robot_response=robot_response/len(pair_sets);ball_response=ball_response/len(pair_sets)
        mode_loss=ar.new_zeros(());mode_support=None
        if a.repair_ball_gate:
            # Recorded starts/next states are paired fit examples only. Never
            # use a future event to select an imagined or deployed forward.
            with torch.no_grad():
                observed=data['state'][t,l]
                observed_prior,_=base.prior(observed,data['present'][t,l],data['elapsed'][t,l],launch[l])
                labels=observed_ball_mode_targets(data['next_state'][t,l,49:]-observed_prior[:,49:],
                    contact_edges[t.cpu(),l.cpu()].to(device),data['present'][t,l].bool(),
                    frame_edge=frame_edges[t.cpu(),l.cpu()].to(device))
                mode_features=ball_geometry_features(observed,base.geometry)
            mode_loss,mode_support=observed_ball_mode_loss(ball.gate(ball.normalized(mode_features)),labels,
                                                           allow_extended=True)
            if mode_support['supervised']==0:raise RuntimeError('No observed free/racket mode labels; do not invent free labels')
        # The paired native response term is the only supervision that directly
        # constrains the direction of the ball response to a final-command
        # perturbation.  Keep its historical default at one for compatibility,
        # but allow a single predeclared contact-focused calibration run to give
        # that term the scale it needs relative to the ordinary state loss.
        loss=ar+point+robot_response+(a.ball_response_weight*ball_response)+mode_loss+contact_loss
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite native-response loss')
        opt.zero_grad(set_to_none=True);loss.backward()
        ball_gate_gradient=float(torch.sqrt(sum((p.grad.detach().square().sum()
            for p in ball.gate.parameters() if p.grad is not None),loss.new_zeros(()))))
        foot_contact_gradient=float(torch.sqrt(sum((p.grad.detach().square().sum()
            for p in base.contact.parameters() if p.grad is not None),loss.new_zeros(()))))
        grad=torch.nn.utils.clip_grad_norm_(parameters,10.,error_if_nonfinite=True);opt.step();torch.cuda.synchronize()
        row=dict(update=update+1,loss=float(loss.detach()),autoregressive=float(ar.detach()),point=float(point.detach()),robot_response=float(robot_response.detach()),ball_response=float(ball_response.detach()),ball_response_weight=float(a.ball_response_weight),gradient=float(grad),pair_support=supports[0],pair_sources_support=dict(zip(pair_names,supports)),seconds=time.perf_counter()-begin,peak_gpu_bytes=torch.cuda.max_memory_allocated(),actor_updates=0)
        row['physical_source_sampling']=source_sampling
        row.update(ball_mode_loss=float(mode_loss.detach()),ball_mode_support=mode_support,
                   ball_gate_gradient=ball_gate_gradient,repair_ball_gate=bool(a.repair_ball_gate),
                   contact_loss=float(contact_loss.detach()),foot_contact_gradient=foot_contact_gradient,
                   repair_foot_contact=bool(a.repair_foot_contact))
        if (update+1)%16==0 or update==255:
            validation_score,validation_rows=native_ball_response_validation(
                base,ball,sweep,pair_sets,val)
            validation_record=dict(update=update+1,score=validation_score,
                                   nonzero_native_ball_rows=validation_rows)
            validation_history.append(validation_record)
            row['native_ball_response_validation']=validation_record
            if validation_rows>0 and validation_score<best_validation_score:
                best_validation_score=validation_score
                best_validation_update=update+1
                best_validation_rows=validation_rows
                best_model_state=dict(
                    base=copy.deepcopy(base.state_dict()),
                    ball=copy.deepcopy(ball.state_dict()),
                    sweep=copy.deepcopy(sweep.state_dict()))
                best_optimizer_state=copy.deepcopy(opt.state_dict())
        with (a.output/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if update%16==0 or update==255:print(json.dumps(row),flush=True)
    if best_model_state is None:
        raise RuntimeError('Native response validation had no nonzero held-out ball response rows')
    base.load_state_dict(best_model_state['base'],strict=True)
    ball.load_state_dict(best_model_state['ball'],strict=True)
    sweep.load_state_dict(best_model_state['sweep'],strict=True)
    if best_optimizer_state is not None:
        opt.load_state_dict(best_optimizer_state)
    config['native_ball_response_validation_history']=validation_history
    config['best_validation_update']=best_validation_update
    config['best_native_ball_response_validation']=best_validation_score
    config['best_native_ball_response_rows']=best_validation_rows
    dump(a.output/'config.json',config)
    contact_changed=tensor_digest(base.contact.state_dict().items())!=contact_hash
    if contact_changed and not a.repair_foot_contact:
        raise RuntimeError('Frozen contact changed')
    if a.repair_foot_contact and not contact_changed:
        raise RuntimeError('Requested foot-contact repair made no parameter change')
    if tensor_digest((k,v) for k,v in ball.state_dict().items() if not k.startswith('gate.'))!=ball_frozen_hash:
        raise RuntimeError('Frozen legacy ball response/normalization changed')
    if not a.repair_ball_gate and tensor_digest(ball.state_dict().items())!=ball_hash:
        raise RuntimeError('Frozen legacy ball changed')
    if a.repair_ball_gate and tensor_digest(ball.gate.state_dict().items())==ball_gate_before:
        raise RuntimeError('Requested ball gate repair made no parameter change')
    if tensor_digest((k,v) for k,v in base.state_dict().items() if not k.startswith('body.') and not k.startswith('contact.'))!=normalization_hash:
        raise RuntimeError('Frozen physical normalization/geometry changed')
    base_path=a.output/'world.pt';ball_path=a.output/'action_ball_candidate.pt'
    clean_source={k:v for k,v in source.items() if 'optimizer' not in k}
    torch.save(dict(clean_source,model=base.state_dict(),world_updates=int(source.get('world_updates',0))+256,
                    response_fit_updates=256,response_fit_config=config,response_optimizer=opt.state_dict(),
                    optimizer_contract='new combined optimizer with explicitly declared parameter scope; old standalone optimizer states deliberately not reused'),base_path)
    effective_legacy=a.ball_checkpoint
    if a.repair_ball_gate:
        effective_legacy=a.output/'ball.pt'
        torch.save(dict({k:v for k,v in ball_source.items() if 'optimizer' not in k},model=ball.state_dict(),
                        gate_fit_updates=256,gate_fit_config=config,source_ball_sha256=sha256(a.ball_checkpoint),
                        optimizer_contract='gate optimizer state belongs to the combined response_optimizer in world.pt'),effective_legacy)
    new_ball=dict({k:v for k,v in sweep_source.items() if 'optimizer' not in k},schema=sweep.schema,model=sweep.state_dict(),
                  physical_sha256=sha256(base_path),legacy_ball_sha256=sha256(effective_legacy),
                  updates=int(sweep_source['updates'])+256,response_fit_config=config)
    if material_contract is not None:new_ball['material_contract']=material_contract
    torch.save(new_ball,ball_path)
    final=dict(candidate=metrics(base,ball,sweep,validation_data,cpu,val.tolist(),launch),response=response_summary(base,ball,sweep,pairs,val),
               responses={name:response_summary(base,ball,sweep,pair,val) for name,pair in zip(pair_names,pair_sets)},
               autoregressive=autoregressive_metrics(base,ball,sweep,validation_data,cpu,val.tolist(),launch))
    dump(a.output/'validation_final.json',final)
    admission={k:final['candidate'][k]['velocity_rmse']<=initial['legacy'][k]['velocity_rmse'] for k in initial['legacy']}
    dump(a.output/'completed.json',dict(status='FIT_COMPLETED_NOT_ADMITTED',world_response_updates=256,actor_updates=0,critic_updates=0,
        world_sha256=sha256(base_path),action_ball_sha256=sha256(ball_path),
        robot_parameters_changed=tensor_digest(base.body.state_dict().items())!=before_base,
        action_ball_parameters_changed=tensor_digest(sweep.state_dict().items())!=before_sweep,
        legacy_ball_sha256=sha256(effective_legacy),legacy_ball_path=str(effective_legacy.resolve()),
        legacy_ball_gate_changed=tensor_digest(ball.gate.state_dict().items())!=ball_gate_before,
        repair_ball_gate=bool(a.repair_ball_gate),
        repair_foot_contact=bool(a.repair_foot_contact),foot_contact_parameters_changed=contact_changed,
        adjacent_step_non_degradation=admission,native_admission=False,
        best_validation_update=best_validation_update,
        best_native_ball_response_validation=best_validation_score,
        best_native_ball_response_rows=best_validation_rows,
        elapsed_s=time.perf_counter()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        scope='one bounded calibrated candidate; no optimizer/policy promotion; independent Recovery and broader contact checks still required'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('probe-directory','split-manifest','physical-checkpoint','ball-checkpoint','action-ball-checkpoint','packed-data','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--updates',type=int,default=256)
    parser.add_argument('--min-native-ball-validation-rows',type=int,default=8,
                        help='Minimum nonzero held-out measured ball-response rows before fitting')
    parser.add_argument('--historical-physical-data',type=Path,default=None,help='Explicit verified two-history physics-only package; unknown actor identities never become RTG labels')
    parser.add_argument('--batch',type=int,default=128,help='GPU model batch; multiple of128, original event ratios and16-step horizon preserved')
    parser.add_argument('--native-material-receipt',type=Path,default=None,help='Explicit V3-to-V4 migration using measured native component materials and differentiable authored inertia')
    parser.add_argument('--extra-probe-directory',type=Path,action='append',default=[],
        help='Additional completed native paired time/head with identical actor, physical contract and heldout split')
    parser.add_argument('--repair-ball-gate',action='store_true',
        help='Explicit candidate revision: supervise existing ball gate on observed training-only free/racket transitions; save a new ball.pt')
    parser.add_argument('--repair-foot-contact',action='store_true',
        help='Explicitly fit existing foot-contact head to recorded native labels; requires Recovery pairs')
    parser.add_argument('--frame-aware-action-ball',action='store_true',
        help='Explicit V4-to-V5 migration: condition action response on learned string plus frame impact probability')
    parser.add_argument('--analytic-impact-prior',action='store_true',
        help='Explicit V4-to-V6 migration: add the fixed authored-material/articulated-inertia impulse prior before the learned residual')
    parser.add_argument('--freeze-physical-response',action='store_true',
        help='Freeze the already validated robot response and legacy ball model; fit only the explicit action-response head')
    parser.add_argument('--ball-response-weight',type=float,default=1.0,
        help='Predeclared multiplier for paired native ball-response supervision; default preserves the historical objective')
    parser.add_argument('--condition-action-on-racket-mode',action='store_true',
        help='Explicit V1-to-V2 candidate migration: weight action correction by shared predicted racket mode; no native change')
    parser.add_argument('--impact-time-state',action='store_true',
        help='Explicit V2-to-V3 migration: causal collision-time robot interpolation; no future native input')
    parser.add_argument('--policy-execution',type=Path,default=None,
        help='Explicit execution-contract sidecar required when packed archive lacks policy_execution metadata')
    args=parser.parse_args()
    if not (args.ball_response_weight>0 and torch.isfinite(torch.tensor(args.ball_response_weight))):
        parser.error('ball-response-weight must be finite and positive')
    if args.analytic_impact_prior and not args.frame_aware_action_ball:
        parser.error('analytic-impact-prior requires frame-aware-action-ball')
    try:main(args)
    except BaseException as e:
        if getattr(args,'_output_created',False):
            dump(args.output/'failure.json',dict(type=type(e).__name__,message=str(e),actor_updates=0))
            (args.output/'failure_traceback.txt').write_text(traceback.format_exc())
        raise
