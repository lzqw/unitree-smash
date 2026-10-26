"""Measured first contact coordinates; no synthetic hits or new score labels."""
from datetime import datetime,timezone
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[3]
MAIN=ROOT.parent/'unitree_smash'
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(MAIN))
from quadsmash.evaluation_v4 import independent_episode,FAULTS

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert not(HERE/'PLAN.json').exists()
    run=MAIN/'reports/G07_full_racket_v4/body_strike_grid16_fixed_train128_01'
    assert read(run/'job.json')['status']=='SUCCEEDED'
    protocol_path=MAIN/'configs/g07_full_racket_eval_v4_r3.json'
    protocol=read(protocol_path);protocol['_root']=str(MAIN)
    config_path=MAIN/'configs/full_racket_v4_r3.json';config=read(config_path)
    a,b=config['assumptions']['strings_semiaxes_yz_m']
    rows,events=read(run/'episodes.json'),read(run/'contacts.json')
    names=read(run/'geometry.json')['body_names'];racket=names.index('Racket')
    source_paths=[Path(__file__),run/'job.json',run/'episodes.json',run/'contacts.json',run/'geometry.json',config_path,protocol_path]
    plan=dict(created_at=datetime.now(timezone.utc).isoformat(),source_sha256={str(p):sha(p) for p in source_paths},
        use='First actual racket event for each of all90 contacted cases;38 misses retained in funnel.',
        contact_coordinates='Actual collision point transformed with the same recorded racket pose. Projected YZ is descriptive, not a substitute collider or new hit classifier.',
        lateral_descriptive_threshold_m=.06,rigid_point_velocity_tolerance_mps=1e-4,
        observations_not_causes=True,new_native_execution=False,formal_acceptance=False)
    (HERE/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    records=[];velocity_errors=[]
    for row in rows:
        if not row['had_contact']:continue
        hits=[e for e in events if e['env_id']==row['env_id'] and e['episode_index']==0 and e['first_racket_contact']]
        assert len(hits)==1;event=hits[0]
        pose=np.asarray(event['body_link_poses_local'])[racket]
        rotation=Rotation.from_quat(pose[[4,5,6,3]]).inv()
        point=rotation.apply(np.asarray(event['point_m'])-pose[:3])
        ball=rotation.apply(np.asarray(event['ball_position_m'])-pose[:3])
        expected=np.asarray(event['flange_linear_velocity_mps'])+np.cross(
            event['flange_angular_velocity_rad_s'],np.asarray(event['point_m'])-event['flange_position_m'])
        error=float(np.linalg.norm(expected-event['point_velocity_mps']));velocity_errors.append(error)
        records.append(dict(TRAIN=row['source_training_index'],env_id=row['env_id'],episode_index=0,
            actual_string_hit=bool(row['string_hit']),actual_frame_hit=bool(row['frame_hit']),
            first_contact_string=bool(event['string_contact']),first_contact_frame=bool(event['frame_contact']),
            contact_x_m=float(point[0]),contact_y_m=float(point[1]),contact_z_m=float(point[2]),
            ball_center_y_m=float(ball[1]),ball_center_z_m=float(ball[2]),
            projected_ellipse_radius_squared=float((point[1]/a)**2+(point[2]/b)**2),
            flange_plus_angular_contact_velocity_error_mps=error))
    assert len(records)==90 and max(velocity_errors)<=plan['rigid_point_velocity_tolerance_mps']
    groups=[]
    for string_hit in [True,False]:
        group=[x for x in records if x['actual_string_hit']==string_hit]
        y,z=np.asarray([[x['contact_y_m'],x['contact_z_m']] for x in group]).T
        groups.append(dict(actual_string_hit=string_hit,count=len(group),
            median_abs_lateral_contact_m=float(np.median(abs(y))),median_longitudinal_contact_m=float(np.median(z)),
            contact_lateral_abs_over6cm=int((abs(y)>.06).sum()),
            projected_radius_squared_quantiles=np.quantile([x['projected_ellipse_radius_squared'] for x in group],[0,.5,1]).tolist()))
    air=[x for x in rows if independent_episode(x,protocol)['airborne_contact']]
    attack=[x for x in air if x['released'] and np.isfinite(x['outgoing_velocity_mps']).all()
        and np.linalg.norm(x['outgoing_velocity_mps'])>=8 and x['outgoing_velocity_mps'][0]>0 and x['outgoing_velocity_mps'][2]<=-1]
    cross=[x for x in attack if x['crossed_net'] and not any(x[f] for f in FAULTS)]
    target=[x for x in cross if x['landed_target']]
    funnel=dict(total128=len(rows),no_contact=sum(not x['had_contact'] for x in rows),
        frame_only=sum(x['had_contact'] and not x['string_hit'] for x in rows),legal_air=len(air),
        attack=len(attack),valid_cross_after_attack=len(cross),target_after_attack_cross=len(target),
        complete=sum(x['complete_smash'] for x in rows),
        physical_jump_success_among_air=sum(x['jump']['success'] for x in air),
        qualified_recovery=sum(independent_episode(x,protocol)['conditional_recovery'] for x in rows))
    assert [funnel[k] for k in ['no_contact','frame_only','legal_air','attack','valid_cross_after_attack','target_after_attack_cross','complete']]==[38,35,55,19,16,4,2]
    with (HERE/'CONTACT_MAP.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    result=dict(checked_at=datetime.now(timezone.utc).isoformat(),funnel=funnel,groups=groups,
        all90_first_contact_point_velocities_checked=True,max_rigid_velocity_error_mps=max(velocity_errors),
        racket_strings_semiaxes_yz_m=[a,b],geometry_hardware_and_scores_unchanged=True,
        specific_controller_cause_not_established=True,new_native_execution=False,formal_acceptance=False)
    (HERE/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    assert all(sha(Path(n))==digest for n,digest in plan['source_sha256'].items())
    print(json.dumps(result))

if __name__=='__main__':main()
