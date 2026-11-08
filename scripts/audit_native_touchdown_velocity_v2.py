"""Compare response errors across observed foot-loading transitions.

Deduplicates event-audit intervals and compares newly loaded feet, avoiding
the confound between a new impact and an already supported foot. Transition
thresholds and post-hoc error bins are diagnostics, not control/success rules.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_native_contact_retry_v3 import initialize
import scripts.audit_native_contact_retry_v3 as setup
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run','configuration','event-audit','output']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();read=lambda p:json.loads(Path(p).read_text())
    def sha(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
        return h.hexdigest()
    audit=read(a.event_audit);meta=read(a.run/'landing_contact_detail.json')
    if a.output.exists() or not meta['complete'] or read(a.run/'job.json')['status']!='SUCCEEDED':
        raise ValueError('Fresh output and completed native telemetry required')
    for path,digest in audit['source_sha256'].items():
        if sha(path)!=digest:raise ValueError('Changed event evidence '+path)
    if audit['source_sha256'].get(str((a.run/'pure_control_commands.npz').resolve()))!=sha(a.run/'pure_control_commands.npz'):
        raise ValueError('Mismatched run and event audit')
    initialize(read(a.configuration),read(a.run/'native_command_inputs.json')['default_joint_position_rad'])
    spheres=setup.CONTROL.geometry.spheres
    if [s['body'] for s in spheres]!=meta['foot_order']:raise ValueError('Foot ordering differs')
    offsets=np.array([s['center_body_m'] for s in spheres]);radii=np.array([s['radius_m'] for s in spheres])
    with np.load(a.run/'landing_contact_detail.npz') as z:
        data={key:z[key].copy() for key in ['foot_pose','foot_twist','normal_force','normal_count','net_peak']}
    unique={}
    for row in audit['rows']:
        key=(row['case'],row['tick'])
        if key not in unique:unique[key]=row
        elif unique[key]['predicted']!=row['predicted']:raise ValueError('Duplicate prediction disagrees')
    rows=[];excluded=dict(nonfoot_or_net=0,no_sampled_loading_transition=0)
    for (case,tick),row in sorted(unique.items()):
        if row['actual_nonfoot_in_interval'] or np.any(data['net_peak'][tick,case]>0):
            excluded['nonfoot_or_net']+=1;continue
        ids=np.arange(tick*10-1,(tick+1)*10)
        pose=data['foot_pose'][ids,case];twist=data['foot_twist'][ids,case]
        quat=pose[..., [4,5,6,3]]
        if np.max(abs(np.linalg.norm(quat,axis=-1)-1))>1e-4:raise ValueError('Invalid foot quaternion')
        center_offset=Rotation.from_quat(quat.reshape(-1,4)).apply(np.tile(offsets,(11,1))).reshape(11,4,3)
        bottom_offset=center_offset.copy();bottom_offset[...,2]-=radii
        velocity=twist[...,:3]+np.cross(twist[...,3:],bottom_offset)
        tangent=np.linalg.norm(velocity[...,:2],axis=-1)
        force=data['normal_force'][ids,case,:,2]
        loaded=(force>1.) & (data['normal_count'][ids,case]>0)
        changes=np.argwhere(loaded[1:] & ~loaded[:-1])
        if not len(changes):excluded['no_sampled_loading_transition']+=1;continue
        events=[]
        for k,j in changes:
            events.append(dict(foot=meta['foot_order'][j],sample_offset_s=(int(k)+1)*.001,
                preceding_tangent_speed_m_s=tangent[k,j],loaded_tangent_speed_m_s=tangent[k+1,j],
                preceding_vertical_velocity_m_s=velocity[k,j,2],
                preceding_normal_force_n=force[k,j],loaded_normal_force_n=force[k+1,j]))
        record=dict(case=case,tick=tick,events=events,
            max_preceding_tangent_speed_m_s=max(e['preceding_tangent_speed_m_s'] for e in events),
            predicted=row['predicted'])
        if row['predicted']:
            for key in ['max_joint_velocity_error_rad_s','angular_velocity_error_rad_s','linear_velocity_error_m_s']:
                record[key]=row[key]
        else:record['rejection']=row.get('rejection')
        rows.append(record)
    bins=[]
    for lo,hi in [(0.,1.),(1.,2.),(2.,3.),(3.,None)]:
        selected=[r for r in rows if r['max_preceding_tangent_speed_m_s']>=lo and (hi is None or r['max_preceding_tangent_speed_m_s']<hi)]
        valid=[r for r in selected if r['predicted']]
        bins.append(dict(lower_m_s=lo,upper_m_s=hi,intervals=len(selected),returned=len(valid),
            joint_velocity_error_median_rad_s=float(np.median([r['max_joint_velocity_error_rad_s'] for r in valid])) if valid else None,
            joint_velocity_error_max_rad_s=max([r['max_joint_velocity_error_rad_s'] for r in valid],default=None),
            numeric_or_model_rejections=len(selected)-len(valid)))
    files=[Path(__file__),a.event_audit,a.configuration,a.run/'landing_contact_detail.npz',a.run/'landing_contact_detail.json']
    result=dict(rows=rows,bins=bins,excluded=excluded,unique_audited_intervals=len(unique),
        source_sha256={str(f.resolve()):sha(f) for f in files},diagnostic_force_threshold_n=1.,
        preceding_sample_is_not_exact_preimpact_state=True,
        sampled_transition_is_not_proof_of_geometric_touchdown=True,
        repeated_cases_not_independent=True,correlation_is_not_unique_cause=True,
        acceptance_changed=False,control_changed=False,native_success=False,valid_expert_labels=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(plain(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps(plain(dict(bins=bins,excluded=excluded,intervals=len(rows)))))


if __name__=='__main__':main()
