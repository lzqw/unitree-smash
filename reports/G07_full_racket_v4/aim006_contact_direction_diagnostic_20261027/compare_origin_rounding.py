"""Compare local and world float32 Euler rounding with existing native flights."""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RUN = ROOT/'reports/G07_full_racket_v4/contact_aim7_sequential_n128_01'


def read(path):
    return json.loads(path.read_text())


def main():
    output = HERE/'FLOAT32_ORIGIN_COMPARISON.json'; assert not output.exists()
    diagnostic = read(HERE/'RESULT.json')
    selected = [item for item in diagnostic['measurements'] if 'predicted_incoming_velocity_mps' in item]
    rows, events, plans = [read(RUN/name) for name in ['episodes_002.json','contacts_002.json','causal_plans_002.json']]
    first = {}
    for plan in plans:
        for j,lane in enumerate(plan['env_ids']):
            if lane not in first:
                first[lane] = (plan['observation_clock_s'][j],plan['initial_position_m'][j],plan['initial_velocity_mps'][j])
    count = 128
    nr = int(np.ceil(count/int(np.sqrt(count)))); nc = int(np.ceil(count/nr))
    origins = np.array([[(nr-1)/2*16-(i//nc)*16,(i%nc)*16-(nc-1)/2*16,0.] for i in range(count)],dtype=np.float32)
    p = np.array([first[i][1] for i in range(count)],dtype=np.float32)
    v = np.array([first[i][2] for i in range(count)],dtype=np.float32)
    world,local = (p+origins).astype(np.float32),p.copy()
    vw,vl = v.copy(),v.copy()
    dt = np.float32(.0000625);g = np.array([0.,0.,-9.81],dtype=np.float32);k = np.float32(.025442307692307694)
    lanes = [item['TRAIN_index'] for item in selected]
    event_by_lane = {event['env_id']:event for event in events if event['first_racket_contact']
                     and event['episode_index'] == rows[event['env_id']]['episode_index']}
    errors_w,errors_l = {i:[] for i in lanes},{i:[] for i in lanes}
    with np.load(RUN/'all_first_episodes_002.npz') as native:
        for step in range(1,17601):
            vw += (g-k*np.linalg.norm(vw,axis=1,keepdims=True)*vw)*dt;world += vw*dt
            vl += (g-k*np.linalg.norm(vl,axis=1,keepdims=True)*vl)*dt;local += vl*dt
            if step%320 == 160:
                time = 2.01+step*float(dt);index = int(round(time/.02))
                actual = native['ball_position_m'][index]
                for lane in lanes:
                    if time <= event_by_lane[lane]['time_s']-.002:
                        errors_w[lane].append(float(np.linalg.norm((world[lane]-origins[lane])-actual[lane])))
                        errors_l[lane].append(float(np.linalg.norm(local[lane]-actual[lane])))
    results = []
    for item in selected:
        lane = item['TRAIN_index']
        results.append(dict(TRAIN_index=lane,actual_native_episode_index=rows[lane]['episode_index'],
            origin_m=origins[lane].tolist(),ideal_float64_max_error_m=item['maximum_precontact_forecast_position_error_m'],
            float32_world_Euler_max_error_m=max(errors_w[lane]),float32_local_Euler_max_error_m=max(errors_l[lane])))
    keys = ['ideal_float64_max_error_m','float32_world_Euler_max_error_m','float32_local_Euler_max_error_m']
    value = dict(checked_at=datetime.now(timezone.utc).isoformat(),count=len(results),origin_grid_shape=[nr,nc],
        actual_origin_grid_source='/home/autodrive/Isaac/env_isaaclab/lib/python3.11/site-packages/isaacsim/exts/isaacsim.core.cloner/isaacsim/core/cloner/impl/grid_cloner.py',
        results=results,summary={key:dict(median=float(np.median([item[key] for item in results])),
                                            maximum=max(item[key] for item in results)) for key in keys},
        first_inline_attempt_failure='Episode IDs were initially assumed all2. Corrected to each original row actual episode_index before comparing; no data rows were renumbered.',
        scope='CPU rounding surrogate compared with existing native precontact samples. Not a new native precision or collision-isolation result.',
        new_native_execution=False,formal_acceptance=False)
    output.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value['summary'],indent=2),flush=True)


if __name__ == '__main__':
    main()
