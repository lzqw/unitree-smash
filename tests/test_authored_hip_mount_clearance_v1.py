import numpy as np
import pytest
from scripts.authored_hip_mount_clearance_v1 import cylinder_box_gap


def check(center):
    return cylinder_box_gap(np.zeros(3),np.eye(3),1.,1.,2,np.array(center),np.eye(3),.1)


def test_axial_and_radial_separation_and_intersection():
    assert check([0.,0.,2.])['axis_separation_m']==pytest.approx(.9)
    assert check([2.,0.,0.])['radial_gap_m']==pytest.approx(.9)
    assert check([1.,0.,0.])['radial_gap_m']==pytest.approx(-.1)
    assert not check([1.,0.,0.])['separated']


def test_rigid_world_transform_preserves_radial_gap():
    R=np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
    t=np.array([2.,3.,4.]);b=np.array([1.,.2,0.])
    original=cylinder_box_gap(np.zeros(3),np.eye(3),1.,1.,2,b,np.eye(3),.1)
    moved=cylinder_box_gap(t,R,1.,1.,2,t+R@b,R,.1)
    assert moved['radial_gap_m']==pytest.approx(original['radial_gap_m'],abs=1e-10)
