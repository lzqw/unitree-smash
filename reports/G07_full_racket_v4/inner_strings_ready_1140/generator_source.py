#!/usr/bin/env python3
"""Initialize an inward string target from measured TRAIN arm configurations."""
import argparse,copy,hashlib,json,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from quadsmash.control.full_racket_contact_uv_retarget_v4 import retarget_contact_uv
from quadsmash.control.full_racket_geometry_v4 import FullRacketGeometryV4
from quadsmash.control.full_racket_rally_controller_v4 import FullRacketRallyControllerV4
from quadsmash.control.full_racket_teacher_v4 import LOW,HIGH


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--contacts',type=Path,required=True)
    p.add_argument('--public',type=Path,required=True)
    p.add_argument('--uv-z',type=float,default=.07)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Use a fresh initialization directory')
    torch.set_num_threads(2)
    source=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    assert source['native_execution'] and source['parameters'].shape==(38,)
    assert source['seed'].get('height_intercept') is None
    assert source['seed'].get('walk_heading_gain',0.)==0.
    old_uv=np.array(source['seed']['string_uv_m']);new_uv=np.array([0.,a.uv_z])
    assert 0.<=a.uv_z<old_uv[1]
    events=[c for c in json.loads(a.contacts.read_text()) if c['first_racket_contact'] and c['string_contact']]
    assert len(events)>=3 and all(c['episode_index']==0 for c in events)
    joints=np.array([c['joint_position_rad'][12:] for c in events]);reference=source['parameters'][8:14].numpy()
    lower=np.maximum(-.15,np.array(LOW[8:14])-reference)
    upper=np.minimum(.15,np.array(HIGH[8:14])-reference)
    fit=retarget_contact_uv(joints,old_uv,new_uv,lower,upper,source['seed'].get('face_sign',1))
    prepared=copy.deepcopy(source)
    prepared['seed']['string_uv_m']=new_uv.tolist()
    prepared['seed']['pitch_compensation_basis_rad_per_rad']=FullRacketGeometryV4().pitch_compensation_basis(
        source['seed']['joint_position_rad'],uv=new_uv,face=source['seed'].get('face_sign',1)).tolist()
    prepared['parameters'][8:14]+=torch.tensor(fit['delta_rad'],dtype=torch.float32)
    assert (prepared['parameters'][8:14]>=torch.tensor(LOW[8:14])).all()
    assert (prepared['parameters'][8:14]<=torch.tensor(HIGH[8:14])).all()
    contract=json.loads((ROOT/'reports/G07_full_racket_v4/CPU_REPLAY_AUTHORED_JOINTS_0724.json').read_text())
    names=[f'{leg}_{joint}_joint' for joint in ['hip','thigh','calf'] for leg in ['FL','FR','RL','RR']]+[f'joint{i}' for i in range(1,7)]
    defaults=dict(contract['defaults_before_neutral_arm_override'])
    reset=json.loads((ROOT/'configs/full_racket_reset_v4.json').read_text())['arm_joint_positions_rad']
    defaults.update({f'joint{i+1}':v for i,v in enumerate(reset)})
    default=torch.tensor([defaults[n] for n in names],dtype=torch.float32)
    limits=torch.tensor([contract['limits'][n] for n in names],dtype=torch.float32)
    controllers=[FullRacketRallyControllerV4(c,2,'cpu') for c in [source,prepared]]
    streams=[[],[]]
    with np.load(a.public) as data:
        with torch.inference_mode():
            for obs,valid in zip(data['observations'],data['valid_first_episode']):
                if not valid.all():break
                public=torch.tensor(obs);before=public.clone()
                for controller,stream in zip(controllers,streams):
                    act=controller.action({'policy':public},default,limits)
                    assert torch.isfinite(act).all() and (abs(act)<=1).all()
                    assert torch.equal(public,before)
                    stream.append(act.numpy().copy())
    values=[np.stack(s) for s in streams]
    assert len(values[0])>=300
    assert np.array_equal(values[0][:,:,:12],values[1][:,:,:12])
    assert not np.array_equal(values[0][:,:,12:],values[1][:,:,12:])
    provenance={k:prepared.pop(k) for k in ['selected_source_run','selected_generation','selected_candidate','selected_training_case_index'] if k in prepared}
    prepared.pop('training_result',None)
    prepared.update(native_execution=False,performance_inherited=False,formal_acceptance=False,
        source_policy_provenance=provenance,source_policy_sha256=digest(a.checkpoint),
        initial_timing_scope='Transferred old control-point calibration as initialization only; inward target has not been measured in dynamics.',
        kind='Unexecuted inward string-target initialization with approximate TRAIN joint-offset retarget.')
    a.output.mkdir(parents=True)
    torch.save(prepared,a.output/'policy.pt')
    report=dict(scope='CPU kinematic fit and frozen-input command replay only; no execution or performance inherited.',
        native_execution=False,formal_acceptance=False,old_string_uv_m=old_uv.tolist(),new_string_uv_m=new_uv.tolist(),
        unchanged_asset_sha256=source['asset_sha256'],unchanged_physical_geometry=True,
        measured_TRAIN_configurations=len(joints),joint_offset_delta_rad=fit['delta_rad'].tolist(),
        position_error_before_quantiles_m=np.quantile(fit['position_error_before_m'],[0,.5,1]).tolist(),
        position_error_after_quantiles_m=np.quantile(fit['position_error_after_m'],[0,.5,1]).tolist(),
        face_change_quantiles_deg=np.degrees(np.quantile(fit['normal_error_after_rad'],[0,.5,1])).tolist(),
        retarget_exact=False,replay_ticks=len(values[0]),legs_bitwise_identical_on_frozen_input=True,
        changed_arm_ticks=int(np.any(values[0][:,:,12:]!=values[1][:,:,12:],axis=(1,2)).sum()),
        caveat='Approximate retarget leaves nonzero position error. Coupled ground-start dynamics must measure string contact, outgoing ball, landing and recovery anew.',
        policy_sha256=digest(a.output/'policy.pt'),source_sha256={str(f):digest(f) for f in
            [a.checkpoint,a.contacts,a.public,Path(__file__),ROOT/'quadsmash/control/full_racket_contact_uv_retarget_v4.py']})
    (a.output/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    (a.output/'CALIBRATION.json').write_text(json.dumps(dict(offline_TRAIN_only=True,
        actor_reads_case_id=False,source_env_ids=[c['env_id'] for c in events],joint_position_rad=joints.tolist(),
        fit={k:np.asarray(v).tolist() for k,v in fit.items()}),indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
