#!/usr/bin/env python3
"""Recheck locked native episodes and aggregate the worse of A/B200 panels."""
import argparse,json,sys
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from quadsmash.evaluation_v4 import validate_frozen_inputs,summarize_panel,aggregate_panels,digest
from quadsmash.tasks.stand.full_racket_origin0_v4 import validate_protocol
from scripts.lock_full_racket_origin0_current_policy_v4 import require_ready_training


def verify_original_noise_and_ground(directory,inputs,protocol,panel):
    rng=np.random.default_rng(protocol['initialization']['reset_seeds'][panel])
    amplitudes=protocol['initialization']['perturbations'][panel]
    noise={name:rng.uniform(-amplitudes[name],amplitudes[name],shape) for name,shape in
        [('joint_position_rad',(200,18)),('joint_velocity_rad_s',(200,18)),('root_xy_m',(200,2)),('root_yaw_rad',(200,))]}
    for name,expected in noise.items():
        if not np.array_equal(np.asarray(inputs['reset_noise'][name]),expected):
            raise ValueError('Original formal reset distribution/seed differs: '+name)
    initial=json.loads((directory/'initial_states.json').read_text())
    if len(initial)!=200 or [row['env_id'] for row in initial]!=list(range(200)):
        raise ValueError('Missing actual formal ground states')
    roots=np.asarray([row['root_state_local'] for row in initial])
    positions=np.tile(protocol['initialization']['root_position_m'],(200,1));positions[:,:2]+=noise['root_xy_m']
    quaternion=np.column_stack((np.cos(noise['root_yaw_rad']/2),np.zeros((200,2)),np.sin(noise['root_yaw_rad']/2)))
    default=np.asarray(inputs['default_joint_position_rad']);limits=np.asarray(inputs['joint_limits_rad'])
    q=np.asarray([row['joint_position_rad'] for row in initial]);qd=np.asarray([row['joint_velocity_rad_s'] for row in initial])
    expected_q=np.clip(default+noise['joint_position_rad'],limits[:,0],limits[:,1])
    if (roots.shape!=(200,13) or not np.isfinite(roots).all() or
        abs(roots[:,:3]-positions).max()>1e-5 or abs(roots[:,3:7]-quaternion).max()>1e-5 or abs(roots[:,7:]).max()>1e-5 or
        q.shape!=(200,18) or qd.shape!=(200,18) or not np.isfinite(q).all() or not np.isfinite(qd).all() or
        abs(q-expected_q).max()>1e-5 or abs(qd-noise['joint_velocity_rad_s']).max()>1e-5):
        raise ValueError('Actual formal ground reset differs from original disturbance contract')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--panel-a',type=Path,required=True)
    p.add_argument('--panel-b',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--policy-lock',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--protocol',type=Path,default=ROOT/'configs/g07_full_racket_eval_v4_r3_origin0.json');a=p.parse_args()
    if a.output.exists():raise ValueError('Aggregate output must be new')
    preview_protocol=json.loads(a.protocol.read_text());validate_protocol(preview_protocol)
    preview_lock=json.loads(a.policy_lock.read_text());readiness=preview_lock['TRAIN128_readiness']
    if require_ready_training(ROOT,preview_protocol,a.checkpoint,Path(readiness['evaluation_root']))!=readiness:
        raise ValueError('Actual readiness changed before opening formal cases')
    protocol,lock=validate_frozen_inputs(ROOT,a.protocol,a.policy_lock,a.checkpoint);validate_protocol(protocol);protocol['_root']=str(ROOT)
    if lock.get('numerical_execution')!=protocol['numerical_execution']:raise ValueError('Numerical variant differs')
    panels=[]
    for name,directory in [('A',a.panel_a),('B',a.panel_b)]:
        if json.loads((directory/'job.json').read_text())['status']!='SUCCEEDED':
            raise ValueError('A panel job did not complete successfully')
        if not (directory/'completed.json').is_file():raise ValueError('Missing native completion receipt')
        inputs=json.loads((directory/'INPUTS.json').read_text())
        layout=json.loads((directory/'origin_layout.json').read_text())
        origins=layout['actual_environment_origins_m']
        if len(origins)!=200 or any(row!=[0.,0.,0.] for row in origins):raise ValueError('Formal actual origins differ')
        if inputs.get('origin_spacing_m')!=0. or inputs.get('origin_precision_receipt_sha256')!=lock['origin_precision_receipt_sha256'] or layout['origin_precision_receipt_sha256']!=lock['origin_precision_receipt_sha256']:
            raise ValueError('Panel numerical/precision admission differs')
        verify_original_noise_and_ground(directory,inputs,protocol,name)
        rows=json.loads((directory/'episodes.json').read_text());recorded=json.loads((directory/'metrics.json').read_text())
        job=json.loads((directory/'job.json').read_text());receipt=json.loads((directory/'completed.json').read_text())
        if job['charged_env_steps']!=160000 or not 0<receipt['env_steps']<=160000 or recorded['actual_control_env_steps']!=receipt['env_steps']:
            raise ValueError('Formal native step accounting differs')
        if recorded['policy_sha256']!=lock['policy_sha256'] or recorded['policy_lock_sha256']!=digest(a.policy_lock):
            raise ValueError('Panel was produced under a different policy lock')
        recomputed=summarize_panel(rows,protocol,name)
        for key,value in recomputed.items():
            if recorded[key]!=value:raise ValueError('Panel summary differs from original episodes: '+key)
        panels.append(recorded)
    result=aggregate_panels(panels,protocol)
    result['episode_sources_sha256']={str(d/'episodes.json'):digest(d/'episodes.json') for d in [a.panel_a,a.panel_b]}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
