"""Locate loss of posture in actual contact flights; no new physics execution."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parent
MAIN=HERE.parents[2]
ROOT=MAIN.parent/'unitree_smash_contact_recovery'
RUN=HERE.parent/'early_lateral7_n128_01'


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def pose_features(poses,names):
    root=names.index('base')
    quat=poses[:,root,3:]
    rotation=Rotation.from_quat(quat[:,[1,2,3,0]])
    euler=rotation.as_euler('xyz',degrees=True)
    tilt=np.rad2deg(np.arccos(np.clip(rotation.as_matrix()[:,2,2],-1.,1.)))
    feet=[names.index(x+'_foot') for x in ['FL','FR','RL','RR']]
    return dict(root_roll_deg=euler[:,0],root_pitch_deg=euler[:,1],root_tilt_deg=tilt,
        root_height_m=poses[:,root,2],minimum_foot_center_height_m=poses[:,feet,2].min(1))


def one_sample(clock,features,time):
    k=int(np.argmin(abs(clock-time)))
    assert abs(clock[k]-time)<=.0101,(time,clock[k])
    return dict(requested_clock_s=float(time),actual_sample_clock_s=float(clock[k]),
        **{name:float(value[k]) for name,value in features.items()})


def summarize_sample(samples,key):
    values=np.array([x[key] for x in samples])
    return dict(minimum=float(values.min()),median=float(np.median(values)),maximum=float(values.max()))


def main():
    names=read(RUN/'geometry.json')['body_names']
    sources=[Path(__file__),RUN/'geometry.json',ROOT/'quadsmash/tasks/stand/jump_env.py',
        ROOT/'quadsmash/tasks/stand/jump_metrics.py',ROOT/'quadsmash/tasks/stand/contact_flight_recovery_v5.py']
    # The legacy body_landing field aggregates all non-foot ground sensors.
    # Preserve its native meaning and avoid claiming a particular body part.
    sensor_code=(ROOT/'quadsmash/tasks/stand/jump_env.py').read_text()
    assert "self.ground_sensors=[self.scene.sensors[name] for name in ('body_ground_contact','arm_ground_contact','tool_ground_contact')]" in sensor_code
    assert 'body_force=sum(sensor.data.force_matrix_w.norm(dim=-1).sum((1,2)) for sensor in self.ground_sensors)' in sensor_code
    assert "filter_prim_paths_expr=['/World/Ground/geometry/mesh']" in sensor_code
    runs=[];profiles={};all_samples=[]
    for candidate in [0,1,2]:
        suffix=f'{candidate:03d}'
        row_path,pose_path,contact_path=[RUN/f'{prefix}_{suffix}.{extension}' for prefix,extension in
            [('episodes','json'),('all_first_episodes','npz'),('contacts','json')]]
        rows,events=read(row_path),read(contact_path)
        assert len(rows)==128 and [row['source_training_index'] for row in rows]==list(range(128))
        chosen=[row for row in rows if row['legal_initial_contact'] and row['airborne_string_hit']]
        with np.load(pose_path) as data:
            poses,clock,valid=[data[k].copy() for k in ['link_poses','episode_clock_s','valid_first_episode']]
        assert poses.shape==(400,128,28,7) and clock.shape==valid.shape==(400,128)
        assert (np.diff(clock,axis=0)[valid[1:]&valid[:-1]]>0).all()
        samples=[]
        for row in chosen:
            lane=row['env_id'];record=row['contact_flight_recovery']
            assert row['ground_init'] and record['bound'] and record['qualified_contact_landing']
            keep=valid[:,lane]
            t,p=clock[keep,lane],poses[keep,lane]
            features=pose_features(p,names)
            event=next(e for e in events if e['env_id']==lane and e['episode_index']==row['episode_index']
                and e['first_racket_contact'] and e['string_contact'])
            exact=pose_features(np.asarray(event['body_link_poses_local'])[None],names)
            contact={k:float(v[0]) for k,v in exact.items()}
            landing=record['contact_flight_landing_s']
            phase_samples={name:one_sample(t,features,landing+offset) for name,offset in
                [('before_landing_40ms',-.04),('landing',0.),('after_landing_40ms',.04),('after_landing_160ms',.16)]}
            sample=dict(candidate_id=candidate,source_training_index=lane,episode_index=row['episode_index'],
                exact_contact_clock_s=event['time_s'],exact_contact_posture=contact,
                contact_flight_landing_clock_s=landing,landing_minus_contact_s=landing-event['time_s'],
                phase_samples=phase_samples,native_contact_flight_recovery=record,
                native_terminated=row['jump']['terminated'],last_valid_sample_clock_s=float(t[-1]),
                raw_nonfoot_ground_flag=record['body_landing'],
                native_terminal_root_height_m=record['terminal_root_height_m'])
            samples.append(sample);all_samples.append(sample)
            key=f'candidate{candidate}_TRAIN{lane}'
            profiles[key+'_time_relative_landing_s']=t-landing
            for name,value in features.items():profiles[key+'_'+name]=value
        summary=dict(candidate_id=candidate,original_case_count=128,actual_air_contacts=len(chosen),
            qualified_contact_landings=sum(x['native_contact_flight_recovery']['qualified_contact_landing'] for x in samples),
            native_qualified_recovery=sum(x['native_contact_flight_recovery']['success'] for x in samples),
            native_nonfoot_ground_flags=sum(x['raw_nonfoot_ground_flag'] for x in samples),
            actual_exact_contact_tilt_deg=summarize_sample([x['exact_contact_posture'] for x in samples],'root_tilt_deg'),
            actual_sampled_landing_roll_deg=summarize_sample([x['phase_samples']['landing'] for x in samples],'root_roll_deg'),
            actual_sampled_landing_pitch_deg=summarize_sample([x['phase_samples']['landing'] for x in samples],'root_pitch_deg'),
            actual_sampled_landing_tilt_deg=summarize_sample([x['phase_samples']['landing'] for x in samples],'root_tilt_deg'),
            actual_sampled_postlanding160ms_tilt_deg=summarize_sample([x['phase_samples']['after_landing_160ms'] for x in samples],'root_tilt_deg'),
            samples=samples)
        runs.append(summary);sources.extend([row_path,pose_path,contact_path])
    trace=HERE.parent/'residual_contact_recovery_n128_updates15_02/trajectory_traces/lane0069_episode000000.npz'
    meta_path=trace.with_suffix('.json');meta=read(meta_path)
    assert sha(trace)==meta['data_sha256'] and meta['raw_episode']['contact_flight_recovery']['success']
    with np.load(trace) as data:
        p=data['link_poses'].copy()
        print('recovered_trace_arrays',data.files)
        # Public pack preserves the original clock among its selected fields;
        # unpack using the unchanged project contract, no inferred new timing.
        import sys
        sys.path.insert(0,str(ROOT))
        from quadsmash.rl.episode_trace_v4 import unpack_public
        public=unpack_public(data['packed_public'])
    assert hashlib.sha256(public.tobytes()).hexdigest()==meta['full_public794_sha256']
    t=public[:,775];features=pose_features(p,names)
    landing=meta['raw_episode']['contact_flight_recovery']['contact_flight_landing_s']
    reference=dict(source='Actual exploration TRAIN69; source actor still in critic warmup. No full smash.',
        source_training_index=69,native_recovery=meta['raw_episode']['contact_flight_recovery'],
        sampled_landing=one_sample(t,features,landing),
        sampled_after_landing160ms=one_sample(t,features,landing+.16))
    profiles['recovered69_time_relative_landing_s']=t-landing
    for name,value in features.items():profiles['recovered69_'+name]=value
    sources.extend([trace,meta_path,ROOT/'quadsmash/rl/episode_trace_v4.py'])
    with (HERE/'POSTURE_PROFILES.npz').open('xb') as f:np.savez_compressed(f,**profiles)
    record=dict(checked_at=datetime.now(timezone.utc).isoformat(),runs=runs,recovered_reference=reference,
        actual_air_contact_flights_compared=len(all_samples),all_original384_rows_retained=True,
        body_landing_field_means='Aggregate ground contact of Go2 non-foot links, Piper links, MountPlate and Racket; no individual touching part can be attributed from the aggregate alone.',
        sampled_pose_rate_Hz=50.,recovered_reference_sample_rate_Hz=100.,
        sampling_does_not_replace_native_event_recovery_or_stability_criteria=True,
        causal_body_part_force_or_impulse_attribution_proven=False,
        profiles_sha256=sha(HERE/'POSTURE_PROFILES.npz'),source_sha256={str(p):sha(p) for p in sources},
        new_native_execution=False,formal_acceptance=False,goal_complete=False)
    with (HERE/'ACTUAL_POSTURE_CHECK.json').open('x') as f:f.write(json.dumps(record,indent=2)+'\n')
    for result in runs:print(json.dumps({k:v for k,v in result.items() if k!='samples'}))
    print(json.dumps(reference))


if __name__=='__main__':main()
