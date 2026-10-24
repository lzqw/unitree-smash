#!/usr/bin/env python3
"""Prepare upper-string and coordinated timing TRAIN proposals, without dynamics."""
import argparse,copy
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from quadsmash.control.full_racket_candidate_bank_v4 import validate_candidate_bank
from quadsmash.control.full_racket_teacher_v4 import LOW,HIGH
from quadsmash.control.full_racket_face_feedback_v4 import validate_face_feedback
from quadsmash.physics.full_racket_v4 import load_racket,string_polygon


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--apex-run',type=Path,required=True);p.add_argument('--recovery-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--measured-face-feedback',action='store_true');a=p.parse_args()
    if a.output.exists():raise ValueError('Use a fresh output directory')
    warm=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    bank=json.loads(a.bank.read_text());reference=warm['parameters'].numpy().copy()
    assert len(reference)==32 and bank['seed']==warm['seed']
    apex_population=a.apex_run/'population_000.pt';apex_pose=a.apex_run/'all_first_episodes_000.npz';plan_path=a.apex_run/'causal_plans_000.json'
    measured=torch.load(apex_population,map_location='cpu',weights_only=True)
    assert torch.equal(measured['parameters'][0],warm['parameters']) and measured['asset_sha256']==warm['asset_sha256']
    event=2.75
    for log in json.loads(plan_path.read_text()):
        if 0 in log['env_ids']:
            i=log['env_ids'].index(0);value=log['calibrated_event_time_s'][i]
            if log['valid'][i] and np.isfinite(value) and 2.3<=value<=3.3:event=value
    with np.load(apex_pose) as saved:
        time=saved['episode_clock_s'][:,0]
        valid=saved['valid_first_episode'][:,0]&(time>=2.25)&(time<=2.95)
        indices=np.flatnonzero(valid);index=indices[np.argmax(saved['link_poses'][valid,0,0,2])]
        peak_time=float(time[index]);peak_height=float(saved['link_poses'][index,0,0,2])
    gain=float(warm['seed'].get('leg_event_time_gain',1.))
    local_peak=(peak_time-reference[2]-gain*(event-2.75))*reference[4]
    assert .9<local_peak<1.8
    low=np.array(LOW+[.8,-.12]+[-5.]*6,dtype=np.float32)
    high=np.array(HIGH+[1.45,.12]+[5.]*6,dtype=np.float32);low[22:24]=-.3;high[22:24]=.3
    rows,labels=[],[]
    def add(candidate,label):
        candidate=np.clip(candidate,low,high).astype(np.float32)
        rows.append(candidate);labels.append(label)
    def timed(scale,relative_apex=0.,phase_margin=0.):
        candidate=reference.copy();candidate[4]=scale
        requested_anchor=2.75-local_peak/scale
        anchor=float(np.clip(requested_anchor,low[2]+phase_margin,high[2]-phase_margin))
        raw=anchor+relative_apex
        candidate[2]=np.clip(raw,low[2],high[2])
        return candidate,dict(leg_clock_scale=scale,requested_apex_relative_to_event_s=relative_apex,
            requested_leg_phase_start_s=float(requested_anchor+relative_apex),
            phase_anchor_adjusted_for_original_bounds=bool(anchor!=requested_anchor),
            implied_old_root_peak_relative_to_event_s=float(candidate[2]+local_peak/scale-2.75))
    setups=[(.65,.503,.95),(.80,.55,1.25),(.95,.60,1.55)]
    for scale in [.85,.95,1.05,1.15]:
        for waypoint,speed,ready in setups:
            for pitch in [.2,.8,1.4]:
                candidate,label=timed(scale)
                candidate[0]=speed;candidate[1]=waypoint;candidate[16]=ready;candidate[15]=pitch
                add(candidate,dict(label,kind='coordinated_clock_preparation_pitch',waypoint_m=waypoint,
                    walk_speed=speed,arm_ready_time_s=ready,pitch_compensation_gain=pitch))
    for scale in [.85,.95,1.05,1.15]:
        for relative in [-.04,0.,.04]:
            candidate,label=timed(scale,relative,phase_margin=.04)
            candidate[0]=.55;candidate[1]=.8;candidate[16]=1.25;candidate[15]=.8
            candidate[5]-=.06;candidate[6]+=.08
            add(candidate,dict(label,kind='longer_push_and_apex_alignment',waypoint_m=.8,
                thigh_delta_rad=-.06,calf_delta_rad=.08))
    recovery_path=a.recovery_run/'population_000.pt';recovery=torch.load(recovery_path,map_location='cpu',weights_only=True)
    recovered=[row for row in recovery['rows'] if row['recovery_success']]
    assert len(recovered)==6 and recovery['asset_sha256']==warm['asset_sha256']
    for row in recovered:
        candidate=recovery['parameters'][row['candidate_id']].numpy().copy()
        assert np.array_equal(candidate[26:32],reference[26:32])
        for pitch in [.8,1.4]:
            proposal=candidate.copy();proposal[15]=pitch
            add(proposal,dict(kind='measured_recovered_configuration_new_upper_goal',source_env_id=row['env_id'],
                source_candidate_id=row['candidate_id'],source_training_case=row['source_training_index'],
                source_had_contact=row['had_contact'],pitch_compensation_gain=pitch,performance_inherited=False))
    assert len(rows)==60 and len({tuple(row) for row in rows})==60
    uv=np.array([0.,.10]);racket=load_racket();polygon=string_polygon(racket)
    edges=np.roll(polygon,-1,axis=0)-polygon;delta=uv-polygon
    assert np.all(edges[:,0]*delta[:,1]-edges[:,1]*delta[:,0]>=0)
    assert np.sum((uv/np.asarray(racket['assumptions']['strings_semiaxes_yz_m']))**2)<1.
    warm=copy.deepcopy(warm);warm['seed']=dict(warm['seed'],string_uv_m=uv.tolist());warm.pop('training_result',None)
    if a.measured_face_feedback:
        warm['seed']['measured_face_feedback']=validate_face_feedback({})
    warm.update(kind='Unexecuted upper-string target with coordinated timing/pitch proposals. No native outcome inherited.',
        native_execution=False,performance_inherited=False,formal_acceptance=False)
    bank=copy.deepcopy(bank);bank.update(seed=warm['seed'],reference_parameters=reference.tolist(),parameters=np.stack(rows).tolist(),
        proposal_labels=labels,controller_revision=('upper-string-final-face-projection-v3' if a.measured_face_feedback else 'upper-string-timing-pitch-proposals-v1'),native_execution=False,
        scope='60new proposals within original32parameter bounds, actual unchanged racket geometry and hardware; native outcomes must be remeasured.')
    checked=validate_candidate_bank(bank,warm['seed'],warm['asset_sha256'],warm['parameter_names'],reference,low,high,62)
    assert checked.shape==(60,32) and np.array_equal(checked[:,26:32],np.broadcast_to(reference[26:32],(60,6)))
    sources=[a.checkpoint,a.bank,apex_population,apex_pose,plan_path,recovery_path,Path(__file__),
        ROOT/'quadsmash/control/full_racket_face_feedback_v4.py',ROOT/'quadsmash/control/full_racket_teacher_v4.py']
    provenance={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    warm['initialization_provenance']=provenance;bank['upper_timing_provenance']=provenance
    a.output.mkdir(parents=True);torch.save(warm,a.output/'policy.pt')
    (a.output/'CANDIDATE_BANK.json').write_text(json.dumps(bank,indent=2)+'\n')
    report=dict(created_at=datetime.now(timezone.utc).isoformat(),scope='CPU-only parameter proposal generation and physical strings target validation, not new dynamics.',
        string_uv_target_m=uv.tolist(),target_inside_original_physical_string_polygon=True,original_collision_geometry_and_limits_unchanged=True,
        measured_root_link_peak_time_s=peak_time,measured_root_link_peak_height_m=peak_height,
        original_calibrated_event_s=event,estimated_local_root_link_peak_clock=float(local_peak),
        peak_scope='Observed root-link height is a timing proxy, not a COM jump metric or a guarantee after controller changes.',
        groups=dict(clock_preparation_pitch=36,push_apex_alignment=12,recovered_configuration_pitch=12),
        old_reference_all32parameters_preserved=True,all_six_velocity_directions_preserved=True,
        shared_old_no_ball_calibration_retained_as_initialization_only=True,native_execution=False,formal_acceptance=False,
        measured_face_feedback=warm['seed'].get('measured_face_feedback'),
        parameter15_semantics=('Original pitch gain plus bounded final-face projection gain with measured-normal lead; final commanded point stays within configured margin' if a.measured_face_feedback else 'Original fixed body-pitch compensation gain'),
        policy_sha256=hashlib.sha256((a.output/'policy.pt').read_bytes()).hexdigest(),
        bank_sha256=hashlib.sha256((a.output/'CANDIDATE_BANK.json').read_bytes()).hexdigest(),sources=provenance)
    (a.output/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
