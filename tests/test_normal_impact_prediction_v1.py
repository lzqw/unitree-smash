import unittest
from pathlib import Path
import numpy as np
import pinocchio as pin
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1,PublicState
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.implicit_support_prediction_v1 import implicit_velocity_step
from scripts.predict_sphere_normal_impact_v1 import predict_sphere_normal_impact,center_normal_jacobians
from scripts.audit_sphere_touchdown_predictions_v1 import sphere_gaps


class ImpactMechanics(unittest.TestCase):
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

    def test_center_normal_jacobian_matches_sphere_height(self):
        m=self.model;q=self.q.copy();J=center_normal_jacobians(m,q,np.zeros(24),self.spheres)
        numerical=np.zeros((4,24));epsilon=1e-6
        for k in range(24):
            delta=np.zeros(24);delta[k]=epsilon
            numerical[:,k]=(sphere_gaps(m,pin.integrate(m.model,q,delta),self.spheres)-
                sphere_gaps(m,pin.integrate(m.model,q,-delta),self.spheres))/(2*epsilon)
        np.testing.assert_allclose(J,numerical,atol=1e-8,rtol=0)

    def test_no_contact_force_created_in_flight(self):
        m=self.model;q=self.q.copy();q[2]+=1.;v=np.zeros(24)
        state=PublicState(q,v,pin.Quaternion(q[3:7]).matrix(),q[:3].copy(),
            q[m.q_indices].copy(),v[m.v_indices].copy(),np.zeros(4,bool),0.)
        off=predict_sphere_normal_impact(m,self.interface,state,state.joint_position,self.spheres,detect_impacts=False)
        on=predict_sphere_normal_impact(m,self.interface,state,state.joint_position,self.spheres,detect_impacts=True)
        self.assertEqual(on['events'],[])
        np.testing.assert_array_equal(on['state'].q,off['state'].q)
        np.testing.assert_array_equal(on['state'].v,off['state'].v)

    def test_contact_impulse_changes_coupled_momentum(self):
        mass=np.array([[2.,.3],[.3,1.]])
        v=np.array([.2,-1.]);S=np.array([[1.],[0.]]);J=np.array([[0.,1.]]);dt=.001
        result=implicit_velocity_step(mass,v,np.zeros(2),S,np.array([2.]),np.array([1.]),
            np.array([0.]),np.array([10.]),dt,J,np.array([1.]))
        self.assertGreater(result['force'][0],0.)
        self.assertAlmostEqual(result['velocity'][1],0.,places=12)
        np.testing.assert_allclose(mass@(result['velocity']-v),
            dt*(S@result['torque']+J.T@result['force']),atol=1e-12)
        self.assertLess(.5*result['velocity']@mass@result['velocity'],.5*v@mass@v)


if __name__=='__main__':unittest.main()
