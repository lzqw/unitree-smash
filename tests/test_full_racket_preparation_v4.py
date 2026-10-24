import unittest
import torch
from quadsmash.control.full_racket_preparation_v4 import validate_preparation,preparation_curve,own_side_reference


class PreparationTest(unittest.TestCase):
    def test_approach_endpoint_positions_and_velocities_are_continuous(self):
        prep=torch.tensor([[1.,0.,1.15]],dtype=torch.float64)
        target=torch.tensor([[1.25,.1,1.82]],dtype=torch.float64)
        velocity=torch.tensor([[1.76,0.,-.165]],dtype=torch.float64)
        event=torch.tensor([2.72],dtype=torch.float64)
        p,v,_=preparation_curve(event-.55,event,prep,target,velocity,.55)
        torch.testing.assert_close(p,prep);torch.testing.assert_close(v,torch.zeros_like(v))
        p,v,_=preparation_curve(event,event,prep,target,velocity,.55)
        torch.testing.assert_close(p,target);torch.testing.assert_close(v,velocity)
        epsilon=1e-7
        before=preparation_curve(event-epsilon,event,prep,target,velocity,.55)
        after=preparation_curve(event+epsilon,event,prep,target,velocity,.55)
        torch.testing.assert_close(before[0],after[0],atol=1e-6,rtol=0)
        torch.testing.assert_close(before[1],after[1],atol=1e-5,rtol=0)

    def test_post_contact_reference_stops_smoothly_and_is_frame_covariant(self):
        prep=torch.tensor([[1.,0.,1.15]],dtype=torch.float64)
        target=torch.tensor([[1.25,.1,1.82]],dtype=torch.float64)
        velocity=torch.tensor([[1.76,0.,-.165]],dtype=torch.float64)
        event=torch.tensor([2.72],dtype=torch.float64)
        p,v,_=preparation_curve(event+.15,event,prep,target,velocity,.55)
        torch.testing.assert_close(p,target+.075*velocity);torch.testing.assert_close(v,torch.zeros_like(v))
        turn=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]],dtype=torch.float64)
        offset=torch.tensor([[.2,.4,-.1]],dtype=torch.float64)
        time=event-.23
        a=preparation_curve(time,event,prep,target,velocity,.55)
        b=preparation_curve(time,event,prep@turn.T+offset,target@turn.T+offset,velocity@turn.T,.55)
        torch.testing.assert_close(b[0],a[0]@turn.T+offset);torch.testing.assert_close(b[1],a[1]@turn.T)

    def test_unhit_command_is_bounded_and_observed_followthrough_is_allowed(self):
        position=torch.tensor([[1.6,.1,1.8],[1.6,.1,1.8]])
        velocity=torch.tensor([[2.,0.,-.3],[2.,0.,-.3]])
        command=torch.tensor([[1.1,0.,1.8],[1.1,0.,1.8]])
        forward=torch.tensor([[1.,0.,0.],[1.,0.,0.]])
        p,v=own_side_reference(position,velocity,command,forward,torch.tensor([False,True]),1.4)
        self.assertAlmostEqual(float(p[0,0]),1.4,places=6);self.assertEqual(float(v[0,0]),0.)
        torch.testing.assert_close(p[1],position[1]);torch.testing.assert_close(v[1],velocity[1])

    def test_optional_configuration_rejects_nonfinite_or_outside_planning_bounds(self):
        self.assertIsNone(validate_preparation(None))
        good={'position_world_m':[1.,0.,1.15]}
        self.assertEqual(validate_preparation(good)['servo_gain'],1.)
        for config in [{'position_world_m':[float('nan'),0.,1.]},dict(good,servo_gain=2.),
                       {'position_world_m':[1.45,0.,1.1]},dict(good,transition_duration_s=.01)]:
            with self.assertRaises(ValueError):validate_preparation(config)


if __name__=='__main__':unittest.main()
