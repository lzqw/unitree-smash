import pytest
import torch

from scripts.audit_supermimic_tangent_magnitude_direction import MeasuredFriction, replace_friction
from scripts.calibrate_supermimic_toi_slip import TOISlip
from scripts.fit_supermimic_toi_friction_radius import SpatialSticking


def test_tangent_replacements_isolate_units_direction_and_magnitude():
    old = torch.tensor([[3.,0,0],[0,0,0],[0,4,0]])
    measured = torch.tensor([[0.,-2,0],[0,0,1],[0,0,0]])
    torch.testing.assert_close(replace_friction(old, measured, 'magnitude'),
                               torch.tensor([[2.,0,0],[0,0,0],[0,0,0]]))
    torch.testing.assert_close(replace_friction(old, measured, 'direction'),
                               torch.tensor([[0.,-3,0],[0,0,0],[0,4,0]]))
    torch.testing.assert_close(replace_friction(old, measured, 'both'), measured)


@pytest.mark.parametrize('slope', [-.75, 0., .75])
def test_radius_calibration_and_native_oracles_respect_contact_scope(slope):
    row = dict(active=torch.tensor([True,True,True,False,True]),object_id=torch.tensor([0,0,2,0,0]),
               normal=torch.tensor([[1.,0,0]]).expand(5,3), relative_velocity=torch.tensor([[-2.,2,3]]).expand(5,3),
               impulse_dv=torch.tensor([[2.,0,0]]).expand(5,3),local=torch.eye(5,3))
    present = torch.tensor([True,True,True,True,False])
    base = TOISlip(.32, 2/7)
    calibrated = SpatialSticking(.32, 2/7, slope, (.0915,.1165))
    new = calibrated.toi_response(row,present)['impulse_residual_dv']
    assert (new.norm(dim=-1)<=.64+1e-6).all()
    torch.testing.assert_close(new[:,0], torch.zeros(5))
    torch.testing.assert_close(new[2:],torch.zeros(3,3))
    if slope==0:
        torch.testing.assert_close(new,base.toi_response(row,present)['impulse_residual_dv'],rtol=0,atol=0)
    oracle = MeasuredFriction(base,torch.ones(5,3),'both').toi_response(row,present)
    torch.testing.assert_close(oracle['impulse_residual_dv'][2:],torch.zeros(3,3))
