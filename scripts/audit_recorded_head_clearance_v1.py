"""Locate head/ground proximity in actual 50Hz pose recordings, not contact forces."""
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
    if args.output.exists(): raise ValueError('Preserve previous result')
    read=lambda p:json.loads(Path(p).read_text())
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    protocol_path=ROOT/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'
    protocol=read(protocol_path);protocol['_root']=str(ROOT)
    score,_=read_run(args.run,protocol)
    shapes_path=ROOT/'configs/go2_piper_authored_ground_geometry_v1.json'
    shapes=read(shapes_path);selected=[s for s in shapes['shapes'] if s['body']=='Head_lower']
    if len(selected)!=1 or selected[0]['kind']!='Sphere' or not np.array_equal(selected[0]['transform'],np.eye(4)):
        raise ValueError('Require original centered lower-head sphere')
    for path,digest in shapes['asset_sha256'].items():
        if sha(path)!=digest: raise ValueError('Changed collider source')
    radius=selected[0]['radius'];geometry_path=args.run/'geometry.json'
    head=read(geometry_path)['body_names'].index('Head_lower')
    archive=args.run/'all_first_episodes.npz'
    with np.load(archive,allow_pickle=False) as data:
        z=data['link_poses'][:,:,head,2].copy()
        valid=data['valid_first_episode'].copy();clock=data['episode_clock_s'].copy()
    if z.shape!=(400,128) or valid.shape!=z.shape or clock.shape!=z.shape:
        raise ValueError('Original 50Hz all128 records required')
    expected=np.broadcast_to(np.arange(400)[:,None]*.02,z.shape)
    if np.max(abs(clock[valid]-expected[valid]))>1e-5:
        raise ValueError('Pose clock differs from control clock')
    outcomes={r['source_training_index']:r for r in read(args.run/'episodes.json')}
    history=RecordedNonfootContactHistory(args.run);rows=[]
    for stage in score['stages']:
        case=stage['training_index'];flight=outcomes[case]['contact_flight_recovery']
        landing=flight.get('contact_flight_landing_s')
        if not flight.get('qualified_contact_landing') or landing is None: continue
        mask=valid[:,case]&(clock[:,case]>=landing)
        gap=z[:,case]-radius;near=np.flatnonzero(mask&(gap<=.001))
        first=None if not len(near) else float(clock[near[0],case])
        original=history.first_frame[case]*history.dt
        rows.append(dict(case=case,complete5=stage['complete5'],ball5_except_recovery=stage['ball5_except_recovery'],
            qualified_landing_s=landing,first_sample_with_head_gap_at_most_1mm_s=first,
            delay_after_landing_s=None if first is None else first-landing,
            minimum_recorded_gap_after_landing_m=float(gap[mask].min()) if mask.any() else None,
            sampled_near_plane_duration_after_landing_s=float((mask&(gap<=.001)).sum()*.02),
            original_first_nonfoot_force_s=float(original) if np.isfinite(original) else None,
            proximity_precedes_original_nonfoot_record=bool(first is not None and first<original)))
    groups={}
    for name,subset in [('qualified_landing',rows),('complete5',[r for r in rows if r['complete5']]),
                       ('ball_good_recovery_failure',[r for r in rows if r['ball5_except_recovery'] and not r['complete5']])]:
        groups[name]=dict(cases=len(subset),head_proximity_observed=sum(r['first_sample_with_head_gap_at_most_1mm_s'] is not None for r in subset),
            precedes_original_nonfoot_record=sum(r['proximity_precedes_original_nonfoot_record'] for r in subset))
    sources=[Path(__file__),protocol_path,shapes_path,geometry_path,archive,
        args.run/'episodes.json',args.run/'contact_telemetry_metadata.json',args.run/'first_nonfoot_contact_details.json']
    sources += [args.run/name for name in score['source_sha256']]
    result=dict(scope=__doc__,groups=groups,records=rows,radius_m=radius,
        diagnostic_proximity_threshold_m=.001,task_thresholds_changed=False,
        source_sha256={str(p.resolve()):sha(p) for p in sources},
        contact_force_measured=False,physical_failure_inferred=False,valid_expert_labels=False,
        limitations=['50Hz geometric proximity cannot establish contact, support force, or exact contact onset.',
            'The original nonfoot sensor grouping omits both head bodies; task labels remain unchanged.',
            'Sampled near-plane duration is not a continuous contact duration.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(groups))


if __name__=='__main__':main()
