#!/usr/bin/env python3
"""Fit the V13 TOI impulse head with measured racket endpoints and native shuttle targets."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from audit_supermimic_paired_body_transfer import response_metrics
from calibrate_supermimic_paired_racket_solver import read_split
from evaluate_supermimic_contact_rollout import sha
from quadsmash.control.reactive_landing_v1 import quat_matrix
from quadsmash.rl.supermimic_action_ball_v1 import TemporalCausalRacketModeImpulseResidualBallV13 as V13
from quadsmash.rl.supermimic_ball_impulse_prior_v1 import BallImpulsePriorV1
from quadsmash.rl.supermimic_physical_run_guard_v1 import atomic_torch_save


class MeasuredRacket:
    """Supply recorded center/twist in the root slots of the analytic FK interface."""
    ids={'Racket':0}

    def __call__(self, joints, joint_rates, center, rotation, linear, angular):
        return dict(position=center[:,None],rotation=quat_matrix(rotation)[:,None],
                    velocity=linear[:,None],angular_velocity=angular[:,None])


def racket_transition_inputs(prior, ball, stream):
    def frame(t):
        # Only racket pose/twist and incoming shuttle are needed by this analytic prior.
        return torch.cat((*[s[t] for s in stream],ball.new_zeros((len(ball),36)),ball),1)
    present=torch.ones(len(ball),dtype=torch.bool)
    analytic,info=prior(frame(0),MeasuredRacket(),present,frame(-1))
    keys=('active','object_id','normal','time','normal_speed','pre_velocity','pre_position','local','relative_velocity',
          'surface_velocity','racket_axis','impulse_dv','remaining_time')
    row={k:info[k] for k in keys}
    row.update(analytic=analytic[:,3:],analytic_position=analytic[:,:3])
    if any(not torch.isfinite(v).all() for v in row.values()):
        raise ValueError('Nonfinite measured labels or analytic features')
    return row


@torch.no_grad()
def prepare(prior, ball, target, stream):
    row=racket_transition_inputs(prior,ball,stream)
    if not torch.isfinite(target).all():
        raise ValueError('Nonfinite native target')
    return dict(row,target=target)


def concatenate(rows):
    return {k:torch.cat([r[k] for r in rows]) for k in rows[0]}


def fixture(root,split,prior):
    rows,branches=read_split(root/(split+'.json'),root,split)
    data={name:concatenate([prepare(prior,*item) for item in branches[name]]) for name in ('minus','plus')}
    for branch in ('nominal_a','nominal_b'):
        audit=json.loads((root/(split+'.json')).read_text())
        for seed in sorted({r['seed'] for r in rows}):
            path=root/branch/f'spec_{seed}'/'trace.pt'
            if sha(path)!=audit['archives'][str(path)]:raise ValueError('Nominal measurement changed')
    native=torch.tensor([r['delta']['native'] for r in rows])
    torch.testing.assert_close(data['plus']['target']-data['minus']['target'],native)
    return data,[r['kind'] for r in rows],torch.tensor([r['repeat']['native'] for r in rows])


def control_event_groups(signatures_by_tick, branches):
    from quadsmash.rl.supermimic_native_manifold_groups import classify

    groups = []
    for tick in zip(*(signatures_by_tick[b] for b in branches), strict=True):
        groups.extend(classify(events) for events in zip(*tick, strict=True))
    return groups


def g07(root,prior):
    from audit_supermimic_native_window_response import complete_window,window_signature
    from quadsmash.rl.supermimic_action_identification_dataset_v1 import _load_branch,_same_contract
    from quadsmash.rl.supermimic_native_pair_capture_v1 import BRANCHES,perturb_request
    from quadsmash.rl.supermimic_native_manifold_groups import classify,signatures
    status=json.loads((root/'status.json').read_text())
    rows=status['completed']
    if status['status']!='COLLECTION_FINISHED' or len(rows)!=4 or {r['branch'] for r in rows}!=set(BRANCHES) or len({r['spec'] for r in rows})!=1:
        raise ValueError('Missing complete same-spec four-branch capture')
    spec_dir=root/rows[0]['spec']
    for r in rows:
        if sha(spec_dir/r['branch']/'native_probe.pt')!=r['probe_sha256']:raise ValueError('Native trace digest changed')
    runs={b:_load_branch(spec_dir,b) for b in BRANCHES}
    _same_contract(runs)
    specs=runs['nominal_a']['spec']
    lanes=specs['lanes']
    signatures_by_branch={};data={};nominal={};sequences={}
    signatures_by_tick = {}
    for name,run in runs.items():
        traces=complete_window(run['raw'],specs,prior.steps)
        torch.testing.assert_close(run['raw']['request_after'],perturb_request(run['raw']['request_before'],specs,name),atol=0,rtol=0)
        signs=[signatures(t) for t in traces]
        signatures_by_tick[name] = signs
        signatures_by_branch[name]=[window_signature([s[i] for s in signs],prior.steps) for i in range(len(lanes))]
        items=[]
        sequence=[]
        for tr in traces:
            t=tr['tensors'];shuttle=t['shuttle_state_w']
            if abs(tr['physics_dt_s']-prior.sub_dt)>1.e-12:raise ValueError('Different physics clock')
            if name.startswith('nominal'):
                items.append(shuttle[-1,:,7:10])
            else:
                stream=[t[k] for k in ('racket_position_w','racket_quaternion_w','racket_linear_velocity_w','racket_angular_velocity_w')]
                ball=torch.cat((shuttle[0,:,:3],shuttle[0,:,7:10]),1)
                items.append(prepare(prior,ball,shuttle[-1,:,7:10],stream))
                sequence.append((ball,shuttle[-1,:,7:10],stream))

        if name.startswith('nominal'):nominal[name]=torch.cat(items)
        else:
            data[name]=concatenate(items)
            sequences[name]=sequence
    groups=[classify([signatures_by_branch[b][i] for b in BRANCHES]) for i in range(len(lanes))]
    executed=torch.stack([(p['final_joint_targets'][lanes]-m['final_joint_targets'][lanes]).abs().amax(-1)
                          for p,m in zip(runs['plus']['raw']['step_sequence'],runs['minus']['raw']['step_sequence'],strict=True)])
    command_delta=executed.amax(0)
    if not command_delta.gt(0).any():
        raise ValueError('No lane received different executed joint targets')
    # Preserve saturated lanes and report them alongside the physical response.
    execution=dict(lanes=lanes,max_joint_target_difference_rad=command_delta.tolist(),
                   unchanged_lanes=[lane for lane,value in zip(lanes,command_delta.tolist()) if value==0],
                   first_native_loaded_interval={name:[row[3][0] if row[2] else None for row in rows]
                                                 for name,rows in signatures_by_branch.items()})
    tick_groups = control_event_groups(signatures_by_tick, BRANCHES)
    return data,tick_groups,nominal['nominal_b']-nominal['nominal_a'],groups,sequences,execution


def predicted_state(model,prior,row):
    info=model.toi_response(row,torch.ones_like(row['active']))
    dv=info['impulse_residual_dv']
    pos,velocity=prior.advance(row['pre_position'],row['pre_velocity']+row['impulse_dv']+dv,row['remaining_time'])
    return torch.where(info['impulse_residual_active'][:,None],torch.cat((pos,velocity),1),
                       torch.cat((row['analytic_position'],row['analytic']),1))


def prediction(model,prior,row):
    return predicted_state(model,prior,row)[:,3:]


@torch.no_grad()
def free_rollout(model,prior,sequences,groups,repeat,execution,baseline_model=None,*,diagnostic_model_at_tick=None):
    targets={}
    outputs={key:{} for key in ('baseline','candidate')}
    timings={name:{} for name in outputs}
    for b,seq in sequences.items():
        for name in outputs:
            state=seq[0][0]
            first_times=state.new_full((len(state),),float('inf'))
            for tick_index,(_,target,stream) in enumerate(seq):
                row=prepare(prior,state,target,stream)
                loaded=row['active'] & (row['object_id']==0)
                first_times=torch.where(loaded & first_times.isinf(),tick_index*prior.dt+row['time'],first_times)
                response = model if name == 'candidate' else baseline_model
                if name == 'candidate' and diagnostic_model_at_tick is not None:
                    response = diagnostic_model_at_tick(b, tick_index)
                state = (predicted_state(response, prior, row) if response is not None else
                         torch.cat((row['analytic_position'], row['analytic']), 1))
            outputs[name][b]=state[:,3:]
            timings[name][b]=first_times
        targets[b]={'target':seq[-1][1]}
    noise=repeat[-len(groups):]
    report={name:metrics((targets,groups,noise),value) for name,value in outputs.items()}
    cases=[]
    native_delta=targets['plus']['target']-targets['minus']['target']
    for i,lane in enumerate(execution['lanes']):
        native=native_delta[i]
        case=dict(lane=lane,mode=groups[i],native_delta_mps=native.tolist(),repeat_noise_mps=float(noise[i].norm()),
                  executed_target_change_rad=execution['max_joint_target_difference_rad'][i])
        for name,v in outputs.items():
            delta=v['plus'][i]-v['minus'][i]
            event={}
            for branch in ('minus','plus'):
                idx=execution['first_native_loaded_interval'][branch][i]
                predicted=float(timings[name][branch][i])
                # Native contact loads are sampled at the end of each substep.
                event[branch]=dict(native_interval_s=None if idx is None else [idx*prior.sub_dt,(idx+1)*prior.sub_dt],
                                   predicted_toi_s=predicted if torch.isfinite(timings[name][branch][i]) else None)
            supported=native.norm()>max(1e-8,3*case['repeat_noise_mps'])
            case[name]=dict(delta_mps=delta.tolist(),error_norm_mps=float((delta-native).norm()),
                            cosine=float(F.cosine_similarity(delta,native,dim=0)) if supported else None,
                            gain=float(delta.norm()/native.norm()) if supported else None,first_racket_impact=event)
        cases.append(case)
    report['cases']=cases
    return report


def metrics(native,output):
    data,groups,repeat=native
    true=data['plus']['target']-data['minus']['target']
    delta=output['plus']-output['minus']
    stable=torch.tensor([g=='racket_stable' for g in groups])
    miss=torch.tensor([g=='no_contact' for g in groups])
    absolute=torch.cat([output[b]-data[b]['target'] for b in output])
    return dict(all_signed=response_metrics(true,delta,repeat),stable_signed=response_metrics(true[stable],delta[stable],repeat[stable]) if stable.any() else None,
                no_contact_signed_rmse=float((true[miss]-delta[miss]).square().mean().sqrt()) if miss.any() else None,
                absolute_rmse=float(absolute.square().mean().sqrt()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('fixture','calibration','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--calibrated-slip', type=Path, help='Use the admitted native/Body first-impact calibration as a frozen baseline')
    p.add_argument('--evaluate-head',type=Path,help='Evaluate saved fixture weights without running the optimizer')
    p.add_argument('--g07-training',type=Path)
    p.add_argument('--g07-development',type=Path)
    p.add_argument('--device',choices=('cpu','cuda:0'),default='cuda:0')
    a=p.parse_args()
    if a.g07_training and not a.g07_development:p.error('Specify separate train and development G07 collections together')
    torch.manual_seed(20261121);torch.set_num_threads(2)
    cfg=json.loads(a.calibration.read_text())
    prior=BallImpulsePriorV1(normal_impulse_scale=cfg['fitted']['normal_impulse_scale']).eval()
    stages={}
    design=json.loads((a.fixture/'design.json').read_text())
    for split in ('train','development'):
        if not any(v==split for v in design['assignments'].values()):raise ValueError('Missing fixture families')
        stages[split]=fixture(a.fixture,split,prior)
    provenance={s:sha(a.fixture/(s+'.json')) for s in stages}
    group_counts={}
    sequences={}
    unique={}
    execution={}
    if a.g07_training:
        train_reg=json.loads((a.g07_training.parent/'preregistration.json').read_text())
        train_src=json.loads((a.g07_training/'startup.json').read_text())
        dev_src=json.loads((a.g07_development/'startup.json').read_text())
        if (train_reg['role']!='training_action_identification' or train_src['source_sha256']!=train_reg['actor_checkpoint_sha256']
            or len(train_src['specs'])!=1 or train_src['specs'][0]['sha256']!=train_reg['spec_sha256']):
            raise ValueError('G07 training collection differs from registration')
        lanes=set(train_src['specs'][0]['spec']['lanes'])
        dev_lanes=set(dev_src['specs'][0]['spec']['lanes'])
        if lanes & (set(train_reg['held_out_lanes'])|dev_lanes) or not dev_lanes<=set(train_reg['held_out_lanes']):
            raise ValueError('Native lane split overlaps')
    folders={'g07_train':a.g07_training,'g07_development':a.g07_development}
    for label,folder in folders.items():
        if folder is None:
            continue
        data,groups,noise,unique_groups,sequences[label],execution[label]=g07(folder,prior)
        unique[label]=unique_groups
        group_counts[label]={k:unique_groups.count(k) for k in sorted(set(unique_groups))}
        stages[label]=(data,groups,noise)
        provenance[label]=sha(folder/'status.json')
        if label=='g07_train' and not any(unique_groups.count(k) for k in ('racket_stable','toi_switch','manifold_switch')):
            raise ValueError('No comparable paired racket impacts for G07 fitting')
        if label=='g07_train':
            signed=data['plus']['target']-data['minus']['target']
            measurable=signed.norm(dim=-1)>torch.maximum(3*noise.norm(dim=-1),noise.new_tensor(1e-8))
            comparable=torch.tensor([g in ('racket_stable','toi_switch','manifold_switch') for g in groups])
            if not (measurable & comparable).any():
                raise ValueError('No paired racket response above nominal-repeat noise in G07 training data')
    fit_labels=['train']+(['g07_train'] if a.g07_training else [])
    data={b:concatenate([stages[s][0][b] for s in fit_labels]) for b in ('plus','minus')}
    device=a.device
    data={b:{k:v.to(device) for k,v in r.items()} for b,r in data.items()}
    prior=prior.to(device)
    model=V13().to(device)
    analytic = None
    predictor = model
    slip_sha = None
    if a.calibrated_slip:
        from calibrate_supermimic_toi_slip import TOISlip
        slip = json.loads(a.calibrated_slip.read_text())
        if (not a.g07_training or not slip['development_passed'] or not slip['body_racket_response_passed']
                or slip['final_holdout_opened'] or slip['calibration_sha256'] != sha(a.calibration)
                or slip['fixture_sha256'] != {s: provenance[s] for s in ('train', 'development')}
                or slip['g07_training']['status_sha256'] != provenance['g07_train']
                or slip['g07_sha256'] != provenance['g07_development']):
            raise ValueError('Tangential calibration must pass both racket-input checks on these archives')
        config = slip['selected']
        analytic = TOISlip(config['friction'], config['sticking'])
        predictor = TOISlip(config['friction'], config['sticking'], residual=model)
        slip_sha = sha(a.calibrated_slip)
    opt=torch.optim.Adam(model.impulse_residual_head.parameters(),lr=.001)
    n=len(data['plus']['target'])
    updates=0 if a.evaluate_head else 128
    if a.evaluate_head:
        ck=torch.load(a.evaluate_head,map_location='cpu',weights_only=True)
        if (ck['schema']!='supermimic-supervised-native-racket-head-v1' or ck['v13_semantics_version']!=model.V13_SEMANTICS_VERSION
            or ck.get('slip_calibration_sha256')!=slip_sha or ck['analytic']!=prior.calibration_manifest() or not set(fit_labels)<=ck['source_sha256'].keys()
            or any(provenance.get(key)!=digest for key,digest in ck['source_sha256'].items())):
            raise ValueError('Fixture head or source binding changed')
        model.impulse_residual_head.load_state_dict(ck['state'],strict=True)
    for _ in range(updates):
        ix=torch.randint(n,(128,),device=device)
        rows={b:{k:v[ix] for k,v in r.items()} for b,r in data.items()}
        pred={b:prediction(predictor,prior,rows[b]) for b in rows}
        err={b:pred[b]-rows[b]['target'] for b in rows}
        loss=sum(F.smooth_l1_loss(v,torch.zeros_like(v),beta=.1) for v in err.values())/2
        signed=err['plus']-err['minus']
        loss=loss+F.smooth_l1_loss(signed,torch.zeros_like(signed),beta=.05)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite contact fit')
        opt.zero_grad();loss.backward();opt.step()
    report={}
    model=model.cpu().eval();prior=prior.cpu()
    with torch.no_grad():
        for label,native in stages.items():
            row=native[0]
            baseline = {b: prediction(analytic, prior, v) if analytic is not None else v['analytic']
                        for b, v in row.items()}
            report[label]=dict(baseline=metrics(native,baseline),
                               candidate=metrics(native,{b:prediction(predictor,prior,v) for b,v in row.items()}))
    for k,seq in sequences.items():
        report[k+'_free_rollout']=free_rollout(predictor,prior,seq,unique[k],stages[k][2],execution[k],baseline_model=analytic)
    validations=[report['development']]
    if a.g07_development:
        validations.extend([report['g07_development'],report['g07_development_free_rollout']])
    def qualifies(row):
        baseline,v=row['baseline'],row['candidate']
        m=v['stable_signed']
        return bool(m and m.get('supported',0) and m.get('cosine_median',-1)>=.8 and m.get('cosine_p10',-1)>=.5
                    and .5<=m['gain_median']<=2 and m['absolute_rmse']<baseline['stable_signed']['absolute_rmse']
                    and v['absolute_rmse']<=baseline['absolute_rmse'] and v['all_signed']['absolute_rmse']<=baseline['all_signed']['absolute_rmse']
                    and (v['no_contact_signed_rmse'] is None or v['no_contact_signed_rmse']<=baseline['no_contact_signed_rmse']))
    evidence=dict(slip_calibration_sha256=slip_sha,training_device=str(a.device),groups=group_counts,executed_target_audits=execution,scores=report,source_sha256=provenance,design_sha256=sha(a.fixture/'design.json'),
                  calibration_sha256=sha(a.calibration),heldout_opened=False,actor_updates=0,wm_updates=updates,
                  development_passed=all(map(qualifies,validations)),policy_training_admitted=False,
                  scope='V13 TOI-head supervised calibration with measured racket endpoints. G07 intervals use recorded incoming shuttle states.')
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True,exist_ok=False)
    path=a.output/'toi_head.pt'
    if a.evaluate_head:
        path=a.evaluate_head
    else:
        atomic_torch_save(dict(schema='supermimic-supervised-native-racket-head-v1',v13_semantics_version=model.V13_SEMANTICS_VERSION,
                           state=model.impulse_residual_head.state_dict(),analytic=prior.calibration_manifest(),
                           slip_calibration_sha256=slip_sha,source_sha256=provenance),path)
    evidence['checkpoint_sha256']=sha(path)
    (a.output/'result.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
