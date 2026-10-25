"""Audit selected real TRAIN trajectories against original native raw records.

This consumes a completed native job. Controlled CPU fixtures cannot satisfy
the source receipts. It never writes success labels or changes acceptance.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from quadsmash.evaluation_v4 import independent_episode
from quadsmash.rl.episode_trace_v4 import unpack_public,SCHEMA


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def same(left,right):return json.dumps(left,sort_keys=True)==json.dumps(right,sort_keys=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run=args.run.resolve();assert not args.output.exists()
    job=read(run/'job.json');assert job['status']=='SUCCEEDED' and job['exit_code']==0
    assert job['charged_env_steps']==read(run/'completed.json')['env_steps']>0
    protocol_path=ROOT/'configs/g07_full_racket_eval_v4_r3.json';protocol=read(protocol_path);protocol['_root']=str(ROOT)
    inputs=read(run/'INPUTS.json');assert inputs['contract']['protocol_sha256']==sha(protocol_path)
    directory=run/'trajectory_traces';manifest=read(directory/'manifest.json')
    assert manifest['schema']==SCHEMA and manifest['enabled'] and manifest['disabled_error'] is None
    assert manifest['native_execution'] and manifest['provenance']['native_source_confirmed']
    assert manifest['provenance']['protocol_sha256']==sha(protocol_path)
    assert manifest['provenance']['asset_sha256']==protocol['asset']['sha256']
    names=manifest['body_names'];assert names==read(run/'geometry.json')['body_names']
    assert names[0]=='base' and names.count('Racket')==1;tool_index=names.index('Racket')
    rows=read(run/'episodes.json');indexed={(row['env_id'],row['episode_index']):row for row in rows}
    assert len(indexed)==len(rows)==manifest['closed_episodes_observed']
    contact_index={}
    for event in read(run/'contacts.json'):
        contact_index.setdefault((event['env_id'],event['episode_index']),[]).append(event)
    selected=manifest['saved_episodes'];assert selected and len(selected)==manifest['saved_episode_count']
    # Synchronization diagnostics only, not new physical/acceptance tolerances.
    position_tolerance_m=2e-4;normal_tolerance=5e-5
    checked=[];dependencies=[run/'job.json',run/'INPUTS.json',run/'episodes.json',run/'contacts.json',
        run/'geometry.json',run/'geometry.npz',directory/'manifest.json',protocol_path,Path(__file__).resolve()]
    for entry in selected:
        path=directory/entry['data_file'];meta=read(path.with_suffix('.json'));data=np.load(path)
        assert sha(path)==entry['data_sha256']==meta['data_sha256']
        key=(meta['env_id'],meta['episode_index']);row=indexed[key]
        assert same(row,meta['raw_episode']) and meta['native_execution']
        assert same(contact_index.get(key,[]),meta['actual_contact_events'])
        assert row['ground_init'] and row['source_training_index']==meta['source_training_index']
        public=unpack_public(data['packed_public']);length=len(public)
        assert length==entry['samples']==meta['samples']
        assert hashlib.sha256(public.tobytes()).hexdigest()==meta['full_public794_sha256']==entry['full_public794_sha256']
        assert meta['full_public_reconstruction_bitwise_verified']
        assert np.isfinite(public).all() and np.isfinite(data['link_poses']).all()
        assert data['link_poses'].shape==(length,len(names),7)
        assert all(data[name].shape==(length,18) and np.isfinite(data[name]).all()
            for name in ['residual_actions','base_actions','native_actions'])
        assert np.max(np.abs(data['native_actions']))<=1.000001
        assert np.array_equal(data['episode_clock_s'],public[:,775])
        clock=public[:,775];assert abs(clock[0])<=1e-6 and np.max(np.abs(np.diff(clock)-.01))<=2e-6
        assert abs(float(clock[-1])+.01-row['terminal_native_state']['time_s'])<=2e-5
        poses=data['link_poses'];root=poses[:,0,:3]
        inverse=Rotation.from_quat(poses[:,0,3:][:,[1,2,3,0]]).inv()
        local_tool=inverse.apply(poses[:,tool_index,:3]-root)
        tool_error=float(np.max(np.linalg.norm(local_tool-public[:,45:48],axis=1)))
        tool_rotation=Rotation.from_quat(poses[:,tool_index,3:][:,[1,2,3,0]])
        local_normal=inverse.apply(tool_rotation.apply(np.tile([1.,0.,0.],(length,1))))
        normal_error=float(np.max(np.linalg.norm(local_normal-public[:,48:51],axis=1)))
        assert tool_error<=position_tolerance_m and normal_error<=normal_tolerance
        present=public[:,67]>.5;ball_error=None
        assert data['ball_position_m'].shape==(length,3) and np.isfinite(data['ball_position_m']).all()
        if present.any():
            local_ball=inverse.apply(data['ball_position_m']-root)
            ball_error=float(np.max(np.linalg.norm(local_ball[present]-public[present,61:64],axis=1)))
            assert ball_error<=position_tolerance_m
        outcome=independent_episode(row,protocol)
        assert bool(row['complete_smash'])==outcome['complete']
        actual_iterations=np.unique(data['learner_iteration']).tolist()
        assert actual_iterations==meta['actor_iterations_encountered']==entry['actor_iterations_encountered']
        checked.append(dict(env_id=key[0],episode_index=key[1],samples=length,
            actor_iterations_encountered=actual_iterations,independent_outcome=outcome,
            tool_public_pose_max_error_m=tool_error,normal_public_pose_max_error=normal_error,
            ball_public_pose_max_error_m=ball_error,full_public_sha256=meta['full_public794_sha256'],
            exact_raw_episode_and_contacts_match=True))
        dependencies.extend([path,path.with_suffix('.json')])
    result=dict(checked_at=datetime.now(timezone.utc).isoformat(),run=str(run),
        scope='Actual selected changing-policy TRAIN trajectories, original completed native job receipts and raw records. Not frozen-policy performance.',
        selected_episodes_checked=len(checked),all_selected_trajectories_checked=True,
        diagnostic_synchronization_tolerances=dict(position_m=position_tolerance_m,normal=normal_tolerance),
        these_are_acceptance_tolerances=False,episodes=checked,
        source_sha256={str(path):sha(path)for path in dependencies},
        formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    args.output.parent.mkdir(parents=True,exist_ok=True);tmp=args.output.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(args.output)
    print(json.dumps(dict(checked=len(checked),scope=result['scope'])),flush=True)


if __name__=='__main__':main()
