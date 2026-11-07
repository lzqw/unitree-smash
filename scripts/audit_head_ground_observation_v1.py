"""Audit added head-force observations separately from unchanged task labels."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.compare_frozen_complete5_v2 import read_run
from scripts.recorded_nonfoot_contact_history_v1 import RecordedNonfootContactHistory


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise ValueError('Preserve head-force audit')
    read=lambda p:json.loads(Path(p).read_text())
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    job=read(args.run/'job.json')
    if job['status']!='SUCCEEDED' or job['exit_code']!=0: raise ValueError('Completed native execution required')
    protocol_path=ROOT/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'
    protocol=read(protocol_path);protocol['_root']=str(ROOT)
    score,_=read_run(args.run,protocol)
    metadata_path=args.run/'head_ground_observation.json';metadata=read(metadata_path)
    archive=args.run/'head_ground_observation.npz'
    observer=ROOT/'scripts/native_head_ground_observation_v1.py'
    if (not metadata['complete_window'] or metadata['samples']!=8000
            or metadata['data_sha256']!=sha(archive) or metadata['source_sha256']!=sha(observer)
            or metadata['body_names']!=['Head_lower','Head_upper']
            or metadata['ground_filter']!='/World/Ground/geometry/mesh'
            or metadata['original_nonfoot_scoring_changed'] or metadata['action_changed']):
        raise ValueError('Incomplete or mismatched passive head telemetry')
    with np.load(archive,allow_pickle=False) as data:
        frames=data['physics_frame'].copy();forces=data['head_ground_normal_force'].copy()
        positions=data['head_position_world'].copy();episodes=data['episode_index'].copy()
    if (not np.array_equal(frames,np.arange(16,128001,16)) or forces.shape!=(8000,128,2,3)
            or positions.shape!=(8000,128,2,3) or episodes.shape!=(8000,128)
            or not np.isfinite(forces).all() or not np.isfinite(positions).all()):
        raise ValueError('Incomplete or nonfinite native samples')
    rows=read(args.run/'episodes.json');history=RecordedNonfootContactHistory(args.run)
    outcomes={row['source_training_index']:row for row in rows}
    stages={row['training_index']:row for row in score['stages']}
    records=[]
    for case in range(128):
        actual=outcomes[case]
        if actual['env_id']!=case: raise ValueError('Unexpected case/lane mapping')
        live=episodes[:,case]==0
        magnitude=np.linalg.norm(forces[:,case],axis=-1)
        combined=magnitude.sum(axis=1)
        active=live&(combined>1.)
        hits=np.flatnonzero(active)
        t=frames*.0000625
        flight=actual['contact_flight_recovery']
        landing=flight.get('contact_flight_landing_s')
        after_landing=(active&(t>=landing) if flight.get('qualified_contact_landing')
                       and landing is not None and landing>=0 else np.zeros_like(active))
        after_hits=np.flatnonzero(after_landing)
        old=history.first_frame[case]*history.dt
        first=None if not len(hits) else float(t[hits[0]])
        after=None if not len(after_hits) else float(t[after_hits[0]])
        records.append(dict(case=case,complete5=stages[case]['complete5'],
            ball5_except_recovery=stages[case]['ball5_except_recovery'],
            first_sampled_head_force_above_1n_s=first,
            first_sampled_head_force_after_qualified_landing_s=after,
            contact_flight_landing_s=landing,
            original_first_nonfoot_force_s=float(old) if np.isfinite(old) else None,
            sampled_head_precedes_original_event=bool(first is not None and first<old),
            head_normal_force_peak_n=float(combined[live].max(initial=0.)),
            individual_head_peak_normal_force_n=magnitude[live].max(axis=0,initial=0.).tolist(),
            sampled_exceedance_duration_after_landing_s=float(after_landing.sum()*.001),
            lower_head_minimum_center_height_m=float(positions[live,case,0,2].min()) if live.any() else None))
    groups={}
    for name,subset in [('all',records),('original_metric_success',[r for r in records if r['complete5']]),
            ('ball_good_recovery_failure',[r for r in records if r['ball5_except_recovery'] and not r['complete5']])]:
        groups[name]=dict(cases=len(subset),head_force_observed=sum(r['first_sampled_head_force_above_1n_s'] is not None for r in subset),
            head_force_after_landing=sum(r['first_sampled_head_force_after_qualified_landing_s'] is not None for r in subset),
            precedes_original_nonfoot_event=sum(r['sampled_head_precedes_original_event'] for r in subset))
    sources=[Path(__file__),observer,args.run/'job.json',metadata_path,archive,protocol_path,
        args.run/'episodes.json',args.run/'contact_telemetry_metadata.json',args.run/'first_nonfoot_contact_details.json']
    sources += [args.run/name for name in score['source_sha256']]
    result=dict(scope=__doc__,native_execution=True,original_complete5=score['complete5'],groups=groups,records=records,
        source_sha256={str(p.resolve()):sha(p) for p in sources},scoring_changed=False,valid_expert_labels=False,
        limitations=['Instantaneous 1ms samples can miss between-sample contacts; summed exceedance time is not a continuous-window guarantee.',
            'A head-force finding requires revisiting physical admissibility; original task labels are retained for traceability.',
            'This observer does not establish unchanged trajectories: the separate student-shadow audit must also pass.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(groups))


if __name__=='__main__':main()
