"""Compare actual public stability signals; sampled proxies never rescore success."""
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parent
MAIN=HERE.parents[2]
ROOT=MAIN.parent/'unitree_smash_contact_recovery'
RUN=HERE.parent/'early_lateral7_n128_01'
sys.path.insert(0,str(ROOT))
from quadsmash.rl.episode_trace_v4 import unpack_public


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def sampled_runs(mask,clock):
    best=current=0;first=None
    for yes,time in zip(mask,clock):
        current=current+1 if yes else 0
        best=max(best,current)
        if current>=80 and first is None:first=float(time)
    return dict(longest_kinematically_eligible_sampled_duration_s=best*.01,
        first80_consecutive_eligible_samples_end_clock_s=first,
        native_ground_force_history_not_available=True,actual_recovery_not_rescored=True)


def main():
    model_dir=HERE.parent/'centroidal_recovery_model_20261027'
    spec=importlib.util.spec_from_file_location('landing_stability_known_model',model_dir/'known_model.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model=module.KnownModel()
    model_check=read(model_dir/'ACTUAL_MODEL_CHECK.json')
    for n in ['known_model.py','MODEL.json']:
        assert sha(model_dir/n)==model_check['source_sha256'][str(model_dir/n)]
    protocol_path=ROOT/'configs/evaluation_protocol_v2.json'
    protocol=read(protocol_path)['recovery']
    defaults=np.asarray(read(RUN/'INPUTS.json')['default_joint_position_rad'])
    inputs=[];files=[Path(__file__),model_dir/'known_model.py',model_dir/'MODEL.json',model_dir/'ACTUAL_MODEL_CHECK.json',
        protocol_path,RUN/'INPUTS.json',ROOT/'quadsmash/tasks/stand/jump_metrics.py',ROOT/'quadsmash/rl/episode_trace_v4.py']
    for i in [0,1,2]:
        suffix=f'{i:03d}';data_path=RUN/f'public_and_actions_{suffix}.npz';rows_path=RUN/f'episodes_{suffix}.json'
        rows=read(rows_path)
        with np.load(data_path) as data:
            public,valid,indices=[data[k].copy() for k in ['observations','valid_first_episode','training_indices']]
        assert public.shape==(800,6,794) and indices.tolist()==[0,29,69,100,112,123]
        for lane,index in enumerate(indices):
            row=rows[index]
            if row['legal_initial_contact'] and row['airborne_string_hit']:
                inputs.append((f'candidate{i}_TRAIN{index}',public[valid[:,lane],lane],row))
        files.extend([data_path,rows_path])
    trace=HERE.parent/'residual_contact_recovery_n128_updates15_02/trajectory_traces/lane0069_episode000000.npz'
    meta_path=trace.with_suffix('.json');meta=read(meta_path)
    assert sha(trace)==meta['data_sha256']
    with np.load(trace) as data:public=unpack_public(data['packed_public'])
    assert hashlib.sha256(public.tobytes()).hexdigest()==meta['full_public794_sha256']
    inputs.append(('recovered69',public,meta['raw_episode']));files.extend([trace,meta_path])
    results=[];profiles={}
    for key,obs,row in inputs:
        time=obs[:,775]
        assert (np.diff(time)>0).all()
        landing=row['contact_flight_recovery']['contact_flight_landing_s']
        state=model.public_world_state(obs,defaults)
        euler=Rotation.from_matrix(state['root_rotation_world']).as_euler('xyz',degrees=True)
        root_v_world=np.einsum('nij,nj->ni',state['root_rotation_world'],obs[:,:3])
        speed=np.linalg.norm(root_v_world[:,:2],axis=1)
        omega=np.linalg.norm(obs[:,3:6],axis=1)
        joint=np.max(abs(obs[:,27:45]),axis=1)
        feet=np.sum(obs[:,57:61]>.5,axis=1)
        conditions=dict(foot_support=feet>0,
            roll_pitch=abs(euler[:,:2]).max(1)<=protocol['maximum_abs_roll_pitch_deg'],
            horizontal_speed=speed<=protocol['maximum_horizontal_root_speed_m_s'],
            angular_speed=omega<=protocol['maximum_root_angular_speed_rad_s'],
            joint_speed=joint<=protocol['maximum_abs_joint_velocity_rad_s'])
        eligible=(time>=landing)&np.logical_and.reduce(list(conditions.values()))
        first_second=(time>=landing)&(time<landing+1.)
        assert first_second.any()
        result=dict(name=key,source_training_index=row['source_training_index'],actual_public_samples=len(obs),
            native_contact_flight_landing_clock_s=landing,
            native_qualified_recovery=row['contact_flight_recovery']['success'],
            native_nonfoot_ground_flag=row['contact_flight_recovery']['body_landing'],
            fractions_meeting_each_condition_first_second_after_landing={k:float(v[first_second].mean()) for k,v in conditions.items()},
            sampled_kinematic_stability=sampled_runs(eligible,time))
        results.append(result)
        profiles[key+'_time_relative_landing_s']=time-landing
        for name,value in dict(root_roll_deg=euler[:,0],root_pitch_deg=euler[:,1],root_angular_speed_rad_s=omega,
                               supported_foot_count=feet,kinematic_eligibility=eligible).items():
            profiles[key+'_'+name]=value
    with (HERE/'PUBLIC_STABILITY_PROFILES.npz').open('xb') as f:np.savez_compressed(f,**profiles)
    record=dict(checked_at=datetime.now(timezone.utc).isoformat(),results=results,source_sha256={str(p):sha(p) for p in files},
        profiles_sha256=sha(HERE/'PUBLIC_STABILITY_PROFILES.npz'),native_protocol_unchanged=True,
        sampled_predicates_not_native_substep_events=True,no_new_native_execution=True,
        full_success_or_causal_recovery_improvement_claimed=False,formal_acceptance=False)
    with (HERE/'PUBLIC_STABILITY_CHECK.json').open('x') as f:f.write(json.dumps(record,indent=2)+'\n')
    for r in results:
        if r['source_training_index']==69:print(json.dumps(r))


if __name__=='__main__':main()
