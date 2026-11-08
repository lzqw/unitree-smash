from types import SimpleNamespace
import json
from pathlib import Path
import numpy as np
from scripts.hip_mount_collision_response_v1 import HipMountCollisionResponseV1


def response():
    record=json.loads((Path(__file__).resolve().parents[1]/'configs/go2_piper_racket_landing_model_v1.json').read_text())
    return HipMountCollisionResponseV1(SimpleNamespace(joint_order=record['joint_order']))


def test_self_contact_reaction_is_coupled_and_does_not_change_hardware_bounds():
    r=response();mass=np.eye(24);mass[0,6]=mass[6,0]=.4
    selection=np.zeros((24,18));selection[6:]=np.eye(18)
    velocity=np.zeros(24);velocity[6]=5.
    q=np.zeros(18);q[0]=.568
    limits=np.tile([-1.0472,1.0472],(18,1));original=limits.copy()
    result=r.step(mass,velocity,np.zeros(24),selection,np.full(18,25.),np.full(18,.5),
        np.zeros(18),np.full(18,23.5),.001,np.zeros((0,24)),np.zeros(0),.65,
        np.full(18,30.),q,limits)
    assert np.array_equal(limits,original)
    assert result['hip_mount_impulse'][0]<0
    assert q[0]+.001*result['velocity'][6]<=.5690785047187993+1e-8
    # Internal joint/mount reaction changes the floating base through M;
    # clipping only the joint velocity would fail this momentum check.
    assert abs(result['velocity'][0])>.1
    assert abs((mass@(result['velocity']-velocity))[0])<1e-9
    assert result['momentum_residual']<1e-9


def test_small_existing_penetration_is_not_teleported_out():
    r=response();q=np.zeros(18);q[0]=.569144
    hardware=np.tile([-1.0472,1.0472],(18,1));dt=.0000625
    bounds=r.interval_bounds(q,hardware,dt)
    assert .5690785047187993<bounds[0,1]<q[0]
    assert bounds[0,1]-q[0]>-1e-6
    assert hardware[0,1]==1.0472
