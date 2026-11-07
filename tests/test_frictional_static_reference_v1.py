import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import pinocchio as pin
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1, PublicState
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.frictional_static_reference_v1 import sphere_static_reference
from scripts.predict_frictional_spheres_v1 import surface_jacobians
from scripts.frictional_impulse_step_v1 import frictional_velocity_step


class StaticSurfaceReference(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((Path(__file__).parent/'fixtures/frictional_recovery_static_reference_case3.json').read_text())
        self.model = LoadedPinocchioModelV1()
        self.assertEqual(self.model.sha256, self.data['model_sha256'])
        self.interface = PositionPDInterface(np.asarray(self.data['joint_limits']),
            np.r_[np.full(12,25.),np.full(6,80.)], np.r_[np.full(12,.5),np.full(6,4.)],
            np.r_[np.full(12,23.5),np.full(6,100.)], np.r_[np.full(12,30.),np.full(6,5.)], .65)
        q = np.asarray(self.data['q']); v = np.zeros(24)
        self.state = PublicState(q.copy(), v, pin.Quaternion(q[3:7]).matrix(), q[:3].copy(),
            q[self.model.q_indices].copy(), v[self.model.v_indices].copy(),
            np.asarray(self.data['contacts'], bool), 0.)
        self.state.previous_command_target = np.asarray(self.data['previous_command'])
        self.controller = SimpleNamespace(qp=SimpleNamespace(model=self.model, interface=self.interface))

    def test_surface_preload_holds_static_full_system(self):
        # Checks static force balance against a separate coupled implicit step.
        # Zero gaps are an algebraic equilibrium test, not a native landing.
        result = sphere_static_reference(self.controller, self.state, self.data['spheres'])
        self.assertTrue(result['solved'])
        active = result['support_indices']
        jac = surface_jacobians(self.model, self.state.q, self.data['spheres'])[active].reshape(-1,24)
        snap = self.model.snapshot(self.state.q, self.state.v)
        selection = np.zeros((24,18)); selection[self.model.v_indices,np.arange(18)] = 1.
        answer = frictional_velocity_step(snap['mass_matrix'], self.state.v, snap['nonlinear'], selection,
            self.interface.kp, self.interface.kd, np.asarray(result['target'])-self.state.joint_position,
            self.interface.effort, .000625, jac, np.zeros(len(active)), self.interface.friction)
        self.assertLess(np.max(abs(answer['velocity'])), 2e-6)
        self.assertLess(answer['momentum_residual'], 1e-10)
        self.assertTrue(result['rate_limited_dynamic_validation_required'])

    def test_no_support_cannot_produce_static_gravity_reference(self):
        self.state.contacts[:] = False
        result = sphere_static_reference(self.controller, self.state, self.data['spheres'])
        self.assertFalse(result['solved'])
        self.assertNotIn('target', result)


if __name__ == '__main__':
    unittest.main()
