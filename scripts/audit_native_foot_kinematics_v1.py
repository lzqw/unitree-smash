"""Inspect measured contact-point geometry and foot motion on audited intervals.

All quantities come from existing native millisecond telemetry and authored
foot spheres. The force threshold selects diagnostic samples only. End-step
velocity is not pre-impact velocity; point means are not centers of pressure.
No physics/control changes or unique-cause inference are made.
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
    for name in ['run','configuration','impulse-audit','output']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();read=lambda f:json.loads(Path(f).read_text())
    def sha(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as file:
            for block in iter(lambda:file.read(1024*1024),b''):h.update(block)
        return h.hexdigest()
    audit=read(a.impulse_audit);meta=read(a.run/'landing_contact_detail.json')
    if a.output.exists() or not meta['complete'] or read(a.run/'job.json')['status']!='SUCCEEDED':
        raise ValueError('Fresh output and completed native evidence required')
    for path,digest in audit['source_sha256'].items():
        if sha(path)!=digest:raise ValueError('Changed impulse evidence '+path)
    initialize(read(a.configuration),read(a.run/'native_command_inputs.json')['default_joint_position_rad'])
    spheres=setup.CONTROL.geometry.spheres
    if [s['body'] for s in spheres]!=meta['foot_order']:
        raise ValueError('Incorrect sphere/sensor order')
    offsets=np.array([s['center_body_m'] for s in spheres])
    radii=np.array([s['radius_m'] for s in spheres])
    with np.load(a.run/'landing_contact_detail.npz') as z:
        data={key:z[key].copy() for key in ['foot_pose','foot_twist','normal_point','friction_point',
            'normal_force','friction_force','normal_count','friction_count']}
    rows=[]
    for interval in audit['rows']:
        case,tick=interval['case'],interval['tick']
        if not interval['actual_clean_ground_and_net_interval']:
            raise ValueError('Clean paired interval required')
        indices=np.arange(tick*10-1,(tick+1)*10)
        pose=data['foot_pose'][indices,case];twist=data['foot_twist'][indices,case]
        quat=pose[..., [4,5,6,3]]
        if np.max(abs(np.linalg.norm(quat,axis=-1)-1))>1e-4:
            raise ValueError('Invalid recorded world quaternion')
        r=Rotation.from_quat(quat.reshape(-1,4))
        center=pose[...,:3]+r.apply(np.tile(offsets,(len(indices),1))).reshape(len(indices),4,3)
        bottom=center.copy();bottom[...,2]-=radii
        velocity=twist[...,:3]+np.cross(twist[...,3:],bottom-pose[...,:3])
        speed=np.linalg.norm(velocity[...,:2],axis=-1)
        force=data['normal_force'][indices,case,:,2]
        loaded=(force[1:]>1.) & (data['normal_count'][indices,case][1:]>0)
        friction_loaded=loaded & (data['friction_count'][indices,case][1:]>0)
        nd=np.linalg.norm((data['normal_point'][indices,case][1:]-bottom[1:])[...,:2],axis=-1)
        fd=np.linalg.norm((data['friction_point'][indices,case][1:]-bottom[1:])[...,:2],axis=-1)
        if not np.any(loaded):raise ValueError('No loaded diagnostic samples')
        feet=[]
        for j,name in enumerate(meta['foot_order']):
            times=np.flatnonzero(loaded[:,j])
            if not len(times):continue
            k=int(times[0])+1
            feet.append(dict(foot=name,first_loaded_sample_offset_s=k*.001,
                preceding_sample_tangent_speed_m_s=speed[k-1,j],
                first_loaded_sample_tangent_speed_m_s=speed[k,j],
                preceding_sample_normal_force_n=force[k-1,j],first_sample_normal_force_n=force[k,j],
                maximum_loaded_tangent_speed_m_s=float(speed[1:,j][loaded[:,j]].max())))
        rows.append(dict(case=case,tick=tick,selected_response_outlier=interval['selected_response_outlier'],
            maximum_loaded_tangent_speed_m_s=float(speed[1:][loaded].max()),
            maximum_normal_point_horizontal_offset_m=float(nd[loaded].max()),
            maximum_friction_point_horizontal_offset_m=float(fd[friction_loaded].max()) if friction_loaded.any() else None,
            minimum_loaded_sphere_gap_m=float(bottom[1:,:,2][loaded].min()),
            maximum_loaded_sphere_gap_m=float(bottom[1:,:,2][loaded].max()),
            per_foot_first_loaded_sample=feet,
            tangent_speed_m_s=speed,normal_force_n=force,loaded_diagnostic_mask=loaded,
            model_impulse_error_ns=interval['total_impulse_error_ns']))
    files=[Path(__file__),a.configuration,a.impulse_audit,a.run/'landing_contact_detail.npz',
        a.run/'landing_contact_detail.json',ROOT/'scripts/native_landing_contact_observer_v2.py']
    files += [Path(m.__file__) for n,m in list(sys.modules.items()) if n.startswith('scripts.') and getattr(m,'__file__',None)]
    result=dict(rows=rows,authored_spheres=spheres,diagnostic_normal_force_threshold_n=1.,
        source_sha256={str(f.resolve()):sha(f) for f in files},acceptance_changed=False,
        native_rerun=False,control_changed=False,model_calibrated=False,valid_expert_labels=False,
        observation_period_s=.001,first_row_is_interval_initial_sample=True,
        sampled_velocity_is_post_physics=True,point_mean_is_not_force_weighted=True,
        preimpact_velocity_or_integrated_work_not_inferred=True,
        selection_is_targeted_not_population_statistics=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(plain(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k not in ['tangent_speed_m_s','normal_force_n','loaded_diagnostic_mask','per_foot_first_loaded_sample']} for r in rows]))


if __name__=='__main__':main()
