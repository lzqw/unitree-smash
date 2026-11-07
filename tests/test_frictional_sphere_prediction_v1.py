import unittest
from pathlib import Path
import numpy as np
import pinocchio as pin
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1,PublicState
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.predict_frictional_spheres_v1 import predict_frictional_spheres,surface_jacobians
from scripts.predict_sphere_normal_impact_v1 import center_normal_jacobians


class SphereContactModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=LoadedPinocchioModelV1()
        with np.load(Path(__file__).parent/'fixtures/native_geometry_sequence_case15.npz') as z:
            cls.q=z['case_015_predicted_q'][0].copy();limits=z['joint_limits_rad'].copy()
        cls.interface=PositionPDInterface(limits,np.r_[np.full(12,25.),np.full(6,80.)],
            np.r_[np.full(12,.5),np.full(6,4.)],np.r_[np.full(12,23.5),np.full(6,100.)],
            np.r_[np.full(12,30.),np.full(6,5.)],.65)
        cls.spheres=[dict(body=n,center_body_m=[-.0020000000949949026,0.,0.],
            radius_m=.02199999988079071) for n in cls.model.feet]

    def test_surface_normal_matches_center_height_jacobian(self):
        full=surface_jacobians(self.model,self.q,self.spheres)
        center=center_normal_jacobians(self.model,self.q,np.zeros(24),self.spheres)
        np.testing.assert_allclose(full[:,2],center,atol=1e-14)

    def test_stale_support_flag_cannot_create_force_in_flight(self):
        m=self.model;q=self.q.copy();q[2]+=1.;v=np.zeros(24)
        s=PublicState(q,v,pin.Quaternion(q[3:7]).matrix(),q[:3].copy(),
            q[m.q_indices].copy(),v[m.v_indices].copy(),np.ones(4,bool),0.)
        a=predict_frictional_spheres(m,self.interface,s,s.joint_position,self.spheres)
        self.assertFalse(a['state'].contacts.any())
        for event in a['events']:np.testing.assert_array_equal(event['normal_impulse_ns'],np.zeros(4))
        np.testing.assert_array_equal(s.q,q)
        self.assertTrue(s.contacts.all())
        s.contacts[:]=False
        b=predict_frictional_spheres(m,self.interface,s,s.joint_position,self.spheres)
        np.testing.assert_array_equal(a['state'].q,b['state'].q)
        np.testing.assert_array_equal(a['state'].v,b['state'].v)


if __name__=='__main__':unittest.main()
