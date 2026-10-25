import unittest
import numpy as np

from quadsmash.control.full_racket_height_intercept_v4 import (
    predict_height_intercept,validate_height_intercept)
from quadsmash.control.initial_intercept import predict_initial_intercept


class HeightInterceptTest(unittest.TestCase):
    def setUp(self):
        self.public=np.zeros((3,794))
        self.public[:,8]=-1.
        self.public[:,763]=1.
        self.public[:,760:763]=[1.1,0.,1.8]
        # Ballistic trajectories crossing the original plane at varying
        # heights, with the middle row exactly matching the policy target.
        self.public[:,61:64]=[4.,0.,2.27]
        self.public[:,63]+=[-.12,0.,.12]
        self.public[:,64:67]=[-5.,.1,2.]
        self.public[:,67]=1.
        self.public[:,775]=2.01
        self.physics=dict(wind_m_s=[0.,0.,0.],quadratic_drag_k_per_m=0.)
        t=(4.-1.1)/5.
        self.height=2.27+2*t-4.905*t*t
        self.config=dict(target_height_m=self.height,plane_count=65)

    def predict(self,public=None,config=None):
        return predict_height_intercept(self.public if public is None else public,
            [1.1,0.,1.8],[1.,0.,0.],1.1,self.physics,
            self.config if config is None else config)

    def test_higher_ball_is_intercepted_later_and_lower_ball_earlier(self):
        before=predict_initial_intercept(self.public,[1.1,0.,1.8],[1.,0.,0.],1.1,self.physics,dt=.001,max_seconds=1.5)
        after=self.predict()
        self.assertTrue(after['valid'].all())
        self.assertGreater(after['selected_intercept_plane_x_m'][0],1.1)
        self.assertLess(after['selected_intercept_plane_x_m'][2],1.1)
        self.assertTrue((abs(after['position_m'][:,2]-self.height)<.004).all())
        self.assertLess(after['arrival_time_s'][0],before['arrival_time_s'][0])
        self.assertGreater(after['arrival_time_s'][2],before['arrival_time_s'][2])

    def test_selected_forecast_matches_independent_ballistic_formula(self):
        plan=self.predict();t=plan['arrival_time_s']
        expected=self.public[:,61:64]+self.public[:,64:67]*t[:,None]
        expected[:,2]-=4.905*t*t
        np.testing.assert_allclose(plan['position_m'],expected,atol=2e-6,rtol=0)
        np.testing.assert_allclose(plan['incoming_velocity_mps'][:,2],2.-9.81*t,atol=1e-11,rtol=0)

    def test_zero_adaptation_preserves_the_entire_legacy_prediction(self):
        before=predict_initial_intercept(self.public,[1.1,0.,1.8],[1.,0.,0.],1.1,self.physics,dt=.001,max_seconds=1.5)
        after=self.predict(config=dict(self.config,adaptation_fraction=0.))
        self.assertEqual(set(after),set(before))
        for name in before: np.testing.assert_array_equal(after[name],before[name])

    def test_invalid_trajectories_remain_invalid_and_keep_reference_plane(self):
        public=self.public.copy();public[:,64]=5.
        result=self.predict(public)
        self.assertFalse(result['valid'].any())
        self.assertFalse(result['height_adaptation_available'].any())
        np.testing.assert_array_equal(result['selected_intercept_plane_x_m'],np.full(3,1.1))

    def test_unused_columns_do_not_leak_future_labels_or_mutate_inputs(self):
        expected=self.predict();public=self.public.copy()
        used=np.r_[np.arange(0,9),np.arange(61,67),np.arange(760,766),775]
        public[:,np.setdiff1d(np.arange(794),used)]=np.nan
        before=public.copy();actual=self.predict(public)
        for name in expected: np.testing.assert_array_equal(actual[name],expected[name])
        np.testing.assert_array_equal(public,before)

    def test_unreachable_height_never_moves_interception_outside_original_bounds(self):
        public=self.public.copy();public[:,63]+=1.
        result=self.predict(public,config=dict(self.config,target_height_m=1.95))
        self.assertTrue((result['selected_intercept_plane_x_m']>=.8).all())
        self.assertTrue((result['selected_intercept_plane_x_m']<=1.45).all())
        np.testing.assert_array_equal(result['selected_intercept_plane_x_m'],np.full(3,.8))

    def test_bad_config_is_rejected(self):
        self.assertIsNone(validate_height_intercept(None))
        for change in [dict(target_height_m=np.nan),dict(target_height_m=1.5),
                       dict(plane_bounds_m=[.7,1.45]),dict(plane_count=1),
                       dict(plane_count=4.5),dict(adaptation_fraction=1.1)]:
            with self.assertRaises(ValueError):
                validate_height_intercept(dict(self.config,**change))
        with self.assertRaises(ValueError):
            self.predict(config=dict(self.config,plane_bounds_m=[1.2,1.45],adaptation_fraction=.5))


if __name__=='__main__': unittest.main()
