"""Offline model-error diagnostic: future native truth never enters control."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from quadsmash.control.base_contact_prediction_v1 import extrapolate_base
from quadsmash.control.full_racket_rally_controller_v4 import planning_world_pose,public_world_basis_in_body,transported_body_acceleration
from quadsmash.control.contact_first_trajectory_search_v1 import ContactFirstSearchConfig


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve evidence')
    obs=np.load(a.input/'public_observations.npy');poses=np.load(a.input/'link_poses_first2.npy')
    names=json.loads((a.input/'geometry.json').read_text())['body_names'];base_id=names.index('base')
    cfg=ContactFirstSearchConfig();rows=[]
    # Even ticks align exactly with the 20-ms native pose recording.
    for tick in (200,216,232,248):
        current=torch.tensor(obs[tick]);previous=torch.tensor(obs[tick-1])
        dt=current[:,775]-previous[:,775]
        basis=public_world_basis_in_body(current);oldbasis=public_world_basis_in_body(previous)
        linear=transported_body_acceleration(current[:,:3],previous[:,:3],basis,oldbasis,dt,dt>0).numpy()
        angular=transported_body_acceleration(current[:,3:6],previous[:,3:6],basis,oldbasis,dt,dt>0).numpy()
        for lane in range(obs.shape[1]):
            R,root=planning_world_pose(obs[tick,lane],[1.1,0.,1.8])
            start_native=poses[tick//2,lane,base_id]
            for horizon in (.1,.2,.4,.6,.8):
                future=tick+round(horizon/.01);native=poses[future//2,lane,base_id]
                estimate=extrapolate_base(horizon,obs[tick,lane,:3],obs[tick,lane,3:6],
                    np.clip(linear[lane],-cfg.base_linear_acceleration_limit_m_s2,cfg.base_linear_acceleration_limit_m_s2),
                    np.clip(angular[lane],-cfg.base_angular_acceleration_limit_rad_s2,cfg.base_angular_acceleration_limit_rad_s2))
                pred_p=root+R@estimate['translation_current_m'];pred_R=R@estimate['rotation_current_from_future']
                native_R=Rotation.from_quat(np.r_[native[4:7],native[3]]).as_matrix()
                rows.append(dict(tick=tick,lane=lane,horizon_s=horizon,
                    predicted_root_position=pred_p.tolist(),native_root_position=native[:3].tolist(),
                    position_error_m=float(np.linalg.norm(pred_p-native[:3])),
                    rotation_error_rad=float(np.linalg.norm(Rotation.from_matrix(native_R.T@pred_R).as_rotvec())),
                    initial_root_reconstruction_error_m=float(np.linalg.norm(root-start_native[:3]))))
    summary={str(t):dict(position_mean_m=float(np.mean([r['position_error_m'] for r in rows if r['horizon_s']==t])),
        position_max_m=max(r['position_error_m'] for r in rows if r['horizon_s']==t),
        rotation_mean_rad=float(np.mean([r['rotation_error_rad'] for r in rows if r['horizon_s']==t])),
        rotation_max_rad=max(r['rotation_error_rad'] for r in rows if r['horizon_s']==t)) for t in (.1,.2,.4,.6,.8)}
    a.output.write_text(json.dumps(dict(scope='Future truth only for offline error measurement',rows=rows,summary=summary),indent=2)+'\n')
    print(json.dumps(summary))
if __name__=='__main__':main()
