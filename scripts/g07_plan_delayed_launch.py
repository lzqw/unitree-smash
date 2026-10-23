#!/usr/bin/env python3
"""CPU launch/flight seeds for a canonical walking-then-striking episode.

No simulated ball is corrected by this program. All results are predictions
used to choose initial physical launches and later native search targets.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from quadsmash.physics.smash_feasibility import predict_flights

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--walk-seconds',type=float,default=3.4)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
physics=json.loads((ROOT/'configs/shuttle_physics_v2.json').read_text())
drag=physics['quadratic_drag_k_per_m']
arrival=a.walk_seconds+1.485
target_heights=np.repeat([1.60,1.65,1.70],6)
initial_vx=-np.tile([8.,12.,20.,35.,60.,100.],3)
def integrate(vz):
    velocity=np.c_[initial_vx,np.zeros(len(vz)),vz]
    position=np.tile([0.,0.,1.15],(len(vz),1))
    steps=int(np.ceil(arrival/.004))
    dt=arrival/steps
    def acceleration(v):return np.array([0.,0.,-9.81])-drag*np.linalg.norm(v,axis=1,keepdims=True)*v
    for _ in range(steps):
        k1=acceleration(velocity)
        v2=velocity+.5*dt*k1
        k2=acceleration(v2)
        v3=velocity+.5*dt*k2
        k3=acceleration(v3)
        v4=velocity+dt*k3
        k4=acceleration(v4)
        position+=dt*(velocity+2*v2+2*v3+v4)/6
        velocity+=dt*(k1+2*k2+2*k3+k4)/6
    return position,velocity

low=np.zeros(len(initial_vx))
high=np.full(len(initial_vx),150.)
for _ in range(30):
    mid=(low+high)/2
    pos,_=integrate(mid)
    below=pos[:,2]<target_heights
    low=np.where(below,mid,low)
    high=np.where(below,high,mid)
vz=(low+high)/2
position,incoming=integrate(vz)
vx_grid,vz_grid=np.meshgrid(np.arange(5.,16.01,.25),np.arange(-12.,-.99,.25))
outgoing=np.c_[vx_grid.ravel(),np.zeros(vx_grid.size),vz_grid.ravel()]
events=[]
for hit_x in [1.40,1.45,1.48]:
    for hit_z in [1.60,1.65,1.70]:
        prediction=predict_flights([hit_x,0.,hit_z],outgoing)
        feasible=prediction['predicted_legal_attack']&(prediction['conservative_net_exit_margin']>=.01)
        for lane in range(len(initial_vx)):
            if abs(target_heights[lane]-hit_z)>1e-8:continue
            change=outgoing-incoming[lane]
            normals=change/np.linalg.norm(change,axis=1,keepdims=True)
            vn_in=normals@incoming[lane]
            vn_out=np.sum(normals*outgoing,axis=1)
            tool_vn=(vn_out+physics['restitution']*vn_in)/(1+physics['restitution'])
            valid=np.where(feasible&(normals[:,0]>=.5)&(normals[:,2]>0))[0]
            if not len(valid):continue
            i=int(valid[np.argmin(np.abs(tool_vn[valid]))])
            initial=[float(hit_x-position[lane,0]),0.,1.15]
            events.append(dict(hit_position_m=[hit_x,0.,hit_z],arrival_time_s=arrival,
                launch_start_m=initial,launch_velocity_mps=[float(initial_vx[lane]),0.,float(vz[lane])],
                incoming_velocity_mps=incoming[lane].tolist(),outgoing_target_mps=outgoing[i].tolist(),
                ideal_face_normal=normals[i].tolist(),required_tool_normal_speed_mps=float(tool_vn[i]),
                predicted_exit_net_margin_m=float(prediction['conservative_net_exit_margin'][i]),
                predicted_first_landing_m=prediction['landing_position'][i].tolist(),
                arrival_height_error_m=float(abs(position[lane,2]-hit_z)),
                status='ideal frictionless restitution seed, not robot feasibility or native collision success'))
events.sort(key=lambda r:abs(r['required_tool_normal_speed_mps']))
(a.output/'predicted_events.json').write_text(json.dumps(events,indent=2)+'\n')
(a.output/'metrics.json').write_text(json.dumps(dict(kind='CPU necessary-condition planning only',
    walk_seconds=a.walk_seconds,arrival_time_s=arrival,predicted_candidates=len(events),
    events=events[:8]),indent=2)+'\n')
print(json.dumps(dict(predicted_candidates=len(events),best=events[:3])),flush=True)
