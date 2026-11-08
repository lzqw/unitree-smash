"""Audit unchanged execution before interpreting all-environment pair forces."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['run','baseline','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if a.output.exists():raise ValueError('Fresh audit output required')
    pins={};rows=[]
    for run in [a.run,a.baseline]:
        job=json.loads((run/'job.json').read_text())
        if job['status']!='SUCCEEDED' or job['exit_code']!=0:raise ValueError('Complete native jobs required')
        pins[str((run/'job.json').resolve())]=sha(run/'job.json')
    for lane in range(128):
        name=f'fixed_policy_traces/case_{lane:03d}.npz'
        for run in [a.run,a.baseline]:pins[str((run/name).resolve())]=sha(run/name)
        with np.load(a.run/name,allow_pickle=False) as x,np.load(a.baseline/name,allow_pickle=False) as y:
            rows.append(dict(case=lane,exact=all(np.array_equal(x[k],y[k]) for k in ['packed_public','dispatched_actions'])))
    meta=json.loads((a.run/'all_partner_observation.json').read_text());data=a.run/'all_partner_observation.npz'
    if sha(data)!=meta['data_sha256'] or not meta['complete_window'] or not meta['backend_filter_paths_exact']:
        raise ValueError('Incomplete or unmapped telemetry')
    sensors=meta['sensor_paths'];paths=meta['partner_paths'];events=[]
    if len(sensors)!=84 or len(paths)!=3585:raise ValueError('Expected full body coverage')
    with np.load(data,allow_pickle=False) as z:
        frames=z['physics_frame'];force=z['maximum_individual_pair_normal_force_n'];partners=z['maximum_partner_body_index']
        if not np.array_equal(frames,np.r_[np.arange(77120,77601),np.arange(90400,90881)]):
            raise ValueError('Missing physical frames')
        if force.shape!=(962,84,128) or partners.shape!=force.shape or not np.isfinite(force).all():
            raise ValueError('Invalid grouped force coverage')
        maximum=0.
        for sensor in range(84):
            source_lane=meta['lanes'][sensor//28]
            for lane in range(128):
                if lane==source_lane:continue
                values=force[:,sensor,lane];peak=float(values.max());maximum=max(maximum,peak)
                if peak>1e-6:
                    index=int(np.argmax(values));body=int(partners[index,sensor,lane])
                    events.append(dict(sensor=sensors[sensor],partner=paths[lane*28+body],
                        maximum_normal_force_n=peak,peak_physics_frame=int(frames[index]),
                        first_nonzero_frame=int(frames[np.flatnonzero(values>1e-6)[0]])))
    for path in [data,a.run/'all_partner_observation.json',Path(__file__)]:pins[str(path.resolve())]=sha(path)
    for path,h in pins.items():
        if sha(path)!=h:raise ValueError('Evidence changed during audit')
    exact=all(r['exact'] for r in rows)
    result=dict(scope=__doc__,observer_exact_replay=exact,case_replay=rows,
        maximum_cross_environment_individual_normal_force_n=maximum,cross_environment_pairs=events,
        source_sha256=pins,physical_cause_identified=False,valid_expert_labels=False,
        limits=['Only selected sensors, two time windows and normal forces are covered.',
            'Exact controller replay does not replace strict original-student isolation checks.',
            'No maximum over bodies is interpreted as a sum or as friction-force measurement.'])
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(exact=exact,cross_environment_max=maximum)))
    if not exact:raise SystemExit(2)


if __name__=='__main__':main()
