"""Endpoint admission distinguishes static preload from posture-only success."""
import copy
import unittest
import numpy as np
from tests import test_frictional_static_reference_v1 as static_fixture
from scripts.frictional_static_reference_v1 import sphere_static_reference
from scripts.terminal_settling_constraints_v1 import assess, rank


class TerminalSettling(unittest.TestCase):
    def setUp(self):
        fixture = static_fixture.StaticSurfaceReference()
        fixture.setUp()
        self.fixture = fixture
        preload = sphere_static_reference(fixture.controller, fixture.state, fixture.data['spheres'])
        self.assertTrue(preload['solved'])
        self.result = dict(solved=True, accepted=True, maximum_scaled_violation=0., cost=0.,
            states=[dict(q=fixture.state.q, v=fixture.state.v, contacts=fixture.state.contacts)],
            commands=[preload['target']], terminal=dict(omega=0., horizontal=0.,
                joint_speed=0., vertical_speed=0.))

    def assessed(self, result):
        fixture = self.fixture
        result['settling'] = assess(fixture.controller, fixture.state, result, fixture.data['spheres'])
        return result

    def test_static_preload_passes_but_does_not_certify_dynamic_hold(self):
        result = self.assessed(self.result)
        self.assertFalse(rank(result)[0])
        # The preload QP also regularizes contact forces, so solving it again
        # need not return an identical command even at a static equilibrium.
        self.assertLess(np.max(np.abs(result['settling']['command_gap_rad'])), .01)
        self.assertFalse(result['settling']['dynamic_hold_verified'])

    def test_same_posture_wrong_command_rejected(self):
        result = copy.deepcopy(self.result)
        result['commands'][0][0] += .03
        result = self.assessed(result)
        self.assertTrue(rank(result)[0])
        self.assertGreater(result['settling']['maximum_scaled_violation'], 0.)

    def test_same_posture_residual_motion_rejected(self):
        result = copy.deepcopy(self.result)
        result['states'][0]['v'][0] = .2
        result['terminal']['horizontal'] = .2
        result = self.assessed(result)
        self.assertTrue(rank(result)[0])
        self.assertGreaterEqual(result['settling']['maximum_scaled_violation'], 1.)

    def test_missing_support_cannot_be_a_settling_endpoint(self):
        result = copy.deepcopy(self.result)
        result['states'][0]['contacts'][:] = False
        result = self.assessed(result)
        self.assertFalse(result['settling']['solved'])
        self.assertTrue(rank(result)[0])


if __name__ == '__main__':
    unittest.main()
