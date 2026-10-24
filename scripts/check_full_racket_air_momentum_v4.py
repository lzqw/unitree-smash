#!/usr/bin/env python3
"""Approximate a native pre-contact flight's momentum from50Hz body poses."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--inertials',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--generation',type=int,default=0)
    p.add_argument('--env',type=int,default=12)
    p.add_argument('--start',type=float,default=2.56)
    p.add_argument('--end',type=float,default=2.74)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Use a fresh output')
    poses_path = a.run / f'all_first_episodes_{a.generation:03d}.npz'
    rows_path = a.run / f'episodes_{a.generation:03d}.json'
    geometry_path = a.run / 'geometry.json'
    row = next(x for x in json.loads(rows_path.read_text()) if x['env_id'] == a.env and x['episode_index'] == 0)
    names = json.loads(geometry_path.read_text())['body_names']
    bodies = {x['name']:x for x in json.loads(a.inertials.read_text())['bodies']}
    mass = np.array([bodies[n]['mass_kg'] for n in names])
    com = np.array([bodies[n]['com_body_m'] for n in names])
    inertia = []
    for name in names:
        body = bodies[name]
        q = np.array(body['principal_axes_wxyz'])
        r = Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
        inertia.append(r @ np.diag(body['principal_inertia_kg_m2']) @ r.T)
    inertia = np.array(inertia)
    data = np.load(poses_path)
    times = data['episode_clock_s'][:,a.env]
    ids = np.flatnonzero((times >= a.start) & (times <= a.end))
    assert len(ids) >= 4 and data['valid_first_episode'][ids,a.env].all()
    assert row['jump']['qualified_flight_seen'] and row['jump']['qualified_jump_landed']
    assert row['jump']['takeoff_time_s'] < times[ids[0]] < times[ids[-1]] < row['contact_time_s']
    pose = data['link_poses'][ids,a.env]
    rotations = Rotation.from_quat(pose[:,:,3:][:,:,[1,2,3,0]].reshape(-1,4)).as_matrix().reshape(len(ids),len(names),3,3)
    positions = pose[:,:,:3] + (rotations @ com[None,:,:,None]).squeeze(-1)
    centers = (mass[None,:,None] * positions).sum(1) / mass.sum()
    velocities = np.diff(positions,axis=0) / np.diff(times[ids])[:,None,None]
    system_v = (mass[None,:,None] * velocities).sum(1) / mass.sum()
    momentum = []
    for k in range(len(ids)-1):
        dt = float(times[ids[k+1]] - times[ids[k]])
        delta = Rotation.from_matrix(rotations[k+1] @ rotations[k].transpose(0,2,1)).as_rotvec()
        omega = delta / dt
        mid_rotation = Rotation.from_rotvec(delta * .5).as_matrix() @ rotations[k]
        world_inertia = mid_rotation @ inertia @ mid_rotation.transpose(0,2,1)
        relative = .5 * (positions[k+1] + positions[k]) - .5 * (centers[k+1] + centers[k])
        h = (world_inertia @ omega[:,:,None]).squeeze(-1).sum(0)
        h += np.cross(relative,mass[:,None] * (velocities[k] - system_v[k])).sum(0)
        momentum.append(h)
    momentum = np.array(momentum)
    midpoint_times = .5 * (times[ids][1:] + times[ids][:-1])
    acceleration = np.diff(system_v,axis=0) / np.diff(midpoint_times)[:,None]
    drift = float(np.max(np.linalg.norm(momentum - momentum.mean(0),axis=1)))
    sources = [poses_path,rows_path,geometry_path,a.inertials,Path(__file__)]
    report = dict(created_at=datetime.now(timezone.utc).isoformat(),
                  scope='One native pre-contact flight interval, finite20ms pose derivatives; not exact native velocities or global physics acceptance.',
                  generation=a.generation,env_id=a.env,episode_index=0,body_count=len(names),
                  first_pose_time_s=float(times[ids[0]]),last_pose_time_s=float(times[ids[-1]]),
                  system_com_acceleration_mean_mps2=acceleration.mean(0).tolist(),
                  system_com_acceleration_max_abs_error_from_gravity_mps2=np.max(abs(acceleration-[0,0,-9.81]),axis=0).tolist(),
                  angular_momentum_samples_kg_m2ps=momentum.tolist(),
                  angular_momentum_mean_kg_m2ps=momentum.mean(0).tolist(),
                  angular_momentum_max_norm_drift_from_mean_kg_m2ps=drift,
                  relative_max_momentum_drift=drift / float(np.linalg.norm(momentum.mean(0))),
                  sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},formal_acceptance=False)
    a.output.write_text(json.dumps(report,indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['angular_momentum_samples_kg_m2ps','sources']}))


if __name__ == '__main__':
    main()
