from types import SimpleNamespace
import numpy as np
import pytest
from scripts.diagnostic_nonfoot_spheres_v1 import nonfoot_spheres


def test_authored_sphere_translation_radius_and_foot_exclusion():
    T=np.eye(4);T[3,:3]=[.1,.2,.3]
    sphere=dict(kind='Sphere',body='head',transform=T,radius=.047,path='/head')
    g=SimpleNamespace(shapes=[sphere,dict(sphere,body='foot')])
    result=nonfoot_spheres(g,['foot'])
    assert len(result)==1 and result[0]['radius_m']==.047
    assert result[0]['center_body_m']==[.1,.2,.3]


def test_scaled_ellipsoid_is_not_silently_a_sphere():
    T=np.eye(4);T[0,0]=2.
    g=SimpleNamespace(shapes=[dict(kind='Sphere',body='head',transform=T,radius=.047,path='/head')])
    with pytest.raises(ValueError,match='Nonuniform'):nonfoot_spheres(g,[])
