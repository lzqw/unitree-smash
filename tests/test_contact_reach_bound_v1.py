from types import SimpleNamespace
import numpy as np
from quadsmash.control.contact_reach_bound_v1 import string_strip_reach_bound
from quadsmash.control.full_racket_geometry_v4 import FullRacketGeometryV4
from quadsmash.control.full_racket_kinematics_v4 import FullRacketKinematicsV4
import torch


def test_bound_contains_actual_fk_across_deterministic_angles_and_strip_corners():
    geometry=FullRacketGeometryV4();fk=FullRacketKinematicsV4()
    strip=dict(center=[0.,0.,.07],size=[.0012,.006,.012])
    bound=string_strip_reach_bound(geometry,[0.,0.,0.],strip,ball_radius_m=.025)
    for angle in np.linspace(-2.,2.,9):
        for y,z in ((-.003,.064),(.003,.076)):
            q=torch.tensor([[angle,-angle,.3,-.2,.1,angle]],dtype=torch.float64)
            point=fk(q,torch.tensor([.0256,y,z],dtype=torch.float64))[0][0].numpy()
            assert np.linalg.norm(point-np.asarray(bound['shoulder_body_m']))<=bound['optimistic_reach_upper_bound_m']+1e-12


def test_outside_bound_is_necessary_only_and_allowance_is_explicit():
    model=SimpleNamespace(arm_origin=np.zeros(3),joints=[(np.zeros(3),None),(np.array([1.,0,0]),None)],tool_offset=np.zeros(3),tool_rotation=np.eye(3))
    strip=dict(center=[0.,0.,0.],size=[.001,.01,.01])
    out=string_strip_reach_bound(model,[2.,0.,0.],strip,ball_radius_m=.025)
    assert out['outside_reach_bound'] and out['conditional_translation_lower_bound_m']>.9
    inside=string_strip_reach_bound(model,[.5,0.,0.],strip,ball_radius_m=.025)
    assert not inside['outside_reach_bound'] and not inside['feasible_if_inside']
