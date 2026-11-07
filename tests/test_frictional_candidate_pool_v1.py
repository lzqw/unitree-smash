import copy
import hashlib
import json
from pathlib import Path
import unittest
import numpy as np
import pinocchio as pin
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.loaded_pinocchio_model_v1 import PublicState
from scripts.frictional_candidate_pool_v1 import FrictionalCandidatePoolV1
from scripts.predict_frictional_spheres_v1 import predict_frictional_spheres


class MeasuredStateCandidatePool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).parent/'fixtures/frictional_short_plan_case3.npz'
        cls.meta=json.loads(path.with_suffix('.json').read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest()!=cls.meta['fixture_sha256']:
            raise ValueError('Changed model-only fixture')
        with np.load(path) as z:
            q=z['q'].copy();v=z['v'].copy();contacts=z['contacts'].copy()
            prior=z['previous_command'].copy();cls.commands=z['commands'].copy()
        cls.controller=build_controller(cls.meta['configuration']);m=cls.controller.qp.model
        cls.state=PublicState(q,v,pin.Quaternion(q[3:7]).matrix(),q[:3].copy(),
            q[m.q_indices].copy(),v[m.v_indices].copy(),contacts,0.)
        cls.state.previous_command_target=prior
        cls.pool=FrictionalCandidatePoolV1(cls.meta['configuration'],cls.meta['spheres'],workers=2,timeout_s=30.)

    @classmethod
    def tearDownClass(cls):
        cls.pool.close()
        if any(p.poll() is None for p in cls.pool.processes):
            raise AssertionError('CPU worker leaked after close')

    def request(self,state,lane):
        return dict(lane=lane,state=state,seed=self.commands,correction=np.zeros_like(self.commands),save_states=True)

    def test_separate_workers_reproduce_same_verified_plan(self):
        first,second=self.pool.batch([self.request(self.state,0),self.request(self.state,1)])
        self.assertTrue(first['accepted']);self.assertTrue(second['accepted'])
        np.testing.assert_array_equal(first['commands'],second['commands'])
        np.testing.assert_allclose(first['commands'],self.commands,atol=1e-14,rtol=0)

    def test_new_observation_replaces_prior_prediction_in_reused_worker(self):
        original=self.pool.batch([self.request(self.state,0)])[0]
        moved=copy.deepcopy(self.state);moved.v[3]+=.005
        following=self.pool.batch([self.request(moved,0)])[0]
        self.assertTrue(following['solved'])
        direct=predict_frictional_spheres(self.controller.qp.model,self.controller.qp.interface,
            moved,self.commands[0],self.meta['spheres'])['state']
        np.testing.assert_allclose(following['states'][0]['q'],direct.q,atol=1e-12,rtol=0)
        self.assertGreater(np.max(abs(np.asarray(following['states'][0]['q'])-original['states'][0]['q'])),1e-8)
        np.testing.assert_array_equal(self.state.v,original['input_v'])


if __name__=='__main__':unittest.main()
