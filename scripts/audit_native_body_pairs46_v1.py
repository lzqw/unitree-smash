"""Audit exact controller replay and summarize independently attributed pairs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['run','baseline','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();read=lambda p:json.loads(Path(p).read_text())
    meta=read(a.run/'body_pair_observation.json')
    if read(a.run/'job.json')['status']!='SUCCEEDED':raise ValueError('Completed native run required')
    errors=[]
    for case in range(128):
        name=f'fixed_policy_traces/case_{case:03d}.npz'
        with np.load(a.run/name,allow_pickle=False) as actual,np.load(a.baseline/name,allow_pickle=False) as baseline:
            row=dict(case=case)
            for key in ['packed_public','dispatched_actions']:
                row[key]=dict(shape_equal=actual[key].shape==baseline[key].shape,
                    exactly_equal=np.array_equal(actual[key],baseline[key]),
                    max_error=float(np.max(abs(actual[key]-baseline[key]))) if actual[key].shape==baseline[key].shape else None)
            errors.append(row)
    same=all(v[k]['exactly_equal'] for v in errors for k in ['packed_public','dispatched_actions'])
    # Backend reports either common filters or one filter list per sensor.
    requested=meta['requested_filter_paths'];backend=meta['backend_filter_paths']
    filters_verified=backend==requested or (len(backend)==56 and all(row==requested for row in backend))
    pair_groups=[]
    with np.load(a.run/'body_pair_observation.npz',allow_pickle=False) as data:
        for label,lo,hi in [('model_error_window',56800,61600),('train46_isolation_window',75840,79040)]:
            select=(data['physics_frame']>=lo)&(data['physics_frame']<=hi)
            frames=data['physics_frame'][select];forces=data['normal_force_world_n'][select]
            norms=np.linalg.norm(forces,axis=-1);events=[]
            if filters_verified:
                for i,j in np.argwhere(norms.max(0)>1.):
                    values=norms[:,i,j];ids=np.flatnonzero(values>1.)
                    events.append(dict(sensor=meta['sensor_paths'][i],partner=requested[j],
                        first_sample_s=float(frames[ids[0]]*meta['physics_dt_s']),
                        peak_normal_force_n=float(values.max()),sample_count_over_1n=len(ids)))
            pair_groups.append(dict(name=label,samples=len(frames),pairs=events))
    paths=[a.run/'job.json',a.run/'body_pair_observation.json',a.run/'body_pair_observation.npz',Path(__file__)]
    output=dict(scope=__doc__,observer_exact_replay=same,filter_paths_verified=filters_verified,
        complete_recording=meta['complete_window'],case_replay=errors,groups=pair_groups,
        source_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths},
        valid_expert_labels=False,
        limitations=['Positive pair forces identify observed partners; sampled zeros do not prove absence of all contacts.',
            'Only declared time windows and body pairs are covered.',
            'Exact replay checks passivity against the same controller, not teacher quality or baseline isolation.'])
    a.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:output[k] for k in ['observer_exact_replay','filter_paths_verified','complete_recording']}))
    if not same or not filters_verified or not meta['complete_window']:raise SystemExit(2)


if __name__=='__main__':main()
