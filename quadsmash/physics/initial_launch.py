"""CPU initial-state shooting under the locked drag model, without corrections."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]


def solve_initial_launches(targets,times,initial_vx,initial_z=1.15):
    targets=np.asarray(targets,dtype=float)
    times=np.asarray(times,dtype=float)
    initial_vx=np.asarray(initial_vx,dtype=float)
    if targets.shape!=(len(times),3) or np.any(times<=0):raise ValueError('Invalid launch targets/times')
    cfg=json.loads((ROOT/'configs/shuttle_physics_v2.json').read_text())
    drag=cfg['quadratic_drag_k_per_m']
    wind=np.array(cfg['wind_m_s'])
    steps=int(np.ceil(times.max()/.004))
    dt=times[:,None]/steps
    def integrate(vz):
        velocity=np.c_[initial_vx,np.zeros(len(times)),vz]
        position=np.tile([0.,0.,initial_z],(len(times),1))
        def acceleration(v):
            relative=v-wind
            return np.array([0.,0.,-9.81])-drag*np.linalg.norm(relative,axis=1,keepdims=True)*relative
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
    low=np.zeros(len(times))
    high=np.full(len(times),150.)
    lo_pos,_=integrate(low)
    hi_pos,_=integrate(high)
    if np.any(lo_pos[:,2]>targets[:,2]) or np.any(hi_pos[:,2]<targets[:,2]):raise ValueError('Launch solve is not bracketed')
    for _ in range(30):
        mid=(low+high)/2
        position,_=integrate(mid)
        below=position[:,2]<targets[:,2]
        low=np.where(below,mid,low)
        high=np.where(below,high,mid)
    vz=(low+high)/2
    position,incoming=integrate(vz)
    starts=np.c_[targets[:,0]-position[:,0],targets[:,1],np.full(len(times),initial_z)]
    velocities=np.c_[initial_vx,np.zeros(len(times)),vz]
    if np.any(np.linalg.norm(velocities,axis=1)>cfg['max_speed_m_s']):raise ValueError('Initial launch exceeds locked model range')
    return [dict(env_id=i,shuttle_start=starts[i].tolist(),shuttle_velocity=velocities[i].tolist(),
        arrival_time_s=float(times[i]),target_position_m=targets[i].tolist(),
        predicted_incoming_velocity_mps=incoming[i].tolist(),
        solver_height_residual_m=float(abs(position[i,2]-targets[i,2]))) for i in range(len(times))]
