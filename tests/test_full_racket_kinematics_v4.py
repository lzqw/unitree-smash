import unittest
import numpy as np
import torch
from quadsmash.control.full_racket_geometry_v4 import FullRacketGeometryV4
from quadsmash.control.full_racket_kinematics_v4 import FullRacketKinematicsV4


class FullRacketJacobianTest(unittest.TestCase):
    def test_batch_pose_and_point_normal_jacobians_match_independent_source_urdf(self):
        reference=FullRacketGeometryV4();model=FullRacketKinematicsV4()
        rng=np.random.default_rng(725101)
        joints=rng.uniform(reference.bounds[:,0]+.02,reference.bounds[:,1]-.02,(16,6))
        center,normal,jp,jn=[a.detach().numpy() for a in model(torch.tensor(joints))]
        for i,q in enumerate(joints):
            point,rotation=reference.forward(q,base_position=(0,0,0))
            np.testing.assert_allclose(center[i],point,atol=1e-12)
            np.testing.assert_allclose(normal[i],rotation[:,0],atol=1e-12)
            for j in range(6):
                step=np.eye(6)[j]*1e-6
                hi,rhi=reference.forward(q+step,base_position=(0,0,0))
                lo,rlo=reference.forward(q-step,base_position=(0,0,0))
                np.testing.assert_allclose(jp[i,:,j],(hi-lo)/2e-6,atol=2e-9)
                np.testing.assert_allclose(jn[i,:,j],(rhi[:,0]-rlo[:,0])/2e-6,atol=2e-9)
                np.testing.assert_allclose(jp[i,:,j]+.0256*jn[i,:,j],
                    (hi+.0256*rhi[:,0]-lo-.0256*rlo[:,0])/2e-6,atol=2e-9)

    def test_opposite_face_upper_strings_point_and_jacobian(self):
        reference=FullRacketGeometryV4();model=FullRacketKinematicsV4()
        q=np.array([-.8,1.7,-2.7,.7,-1.1,.4]);offset=torch.tensor([-.0256,0.,.09])
        point,_,jacobian,_=model(torch.tensor(q[None]),offset.double())
        expected,_=reference.contact_center(q,uv=(0.,.09),face=-1,base_position=(0,0,0))
        np.testing.assert_allclose(point[0].numpy(),expected,atol=1e-8)
        expected_jacobian=reference.contact_jacobian(q,uv=(0.,.09),face=-1,base_position=(0,0,0))
        np.testing.assert_allclose(jacobian[0].numpy(),expected_jacobian,atol=1e-8)


if __name__=='__main__':unittest.main()
