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


def test_count_native_support_independent_of_candidate_output():
    from scripts.score_supermimic_training_contact_panel import supported_native
    case = dict(mode='racket_stable', executed_target_change_rad=.002,
                native_delta_mps=[.03, 0., 0.], repeat_noise_mps=.005)
    assert supported_native(case)
    assert not supported_native(dict(case, repeat_noise_mps=.02))
    assert not supported_native(dict(case, executed_target_change_rad=0.))
    assert not supported_native(dict(case, mode='toi_switch'))


def test_normal_oracle_subtracts_existing_impulse_before_the_single_integration():
    row=dict(active=torch.tensor([True,False]),object_id=torch.zeros(2,dtype=torch.long),
             normal=torch.tensor([[1.,0,0]]).expand(2,3),relative_velocity=torch.zeros(2,3),
             impulse_dv=torch.tensor([[5.,0,0],[0.,0,0]]))
    normal=torch.tensor([[4.,1,0],[0.,0,0]])
    friction=torch.tensor([[0.,0,2],[0.,0,0]])
    active=torch.ones(2,dtype=torch.bool)
    oracle=MeasuredFriction(TOISlip(.32,2/7),friction,'all_loads',normal)
    result=row['impulse_dv']+oracle.toi_response(row,active)['impulse_residual_dv']
    torch.testing.assert_close(result, normal+friction)


def test_direction_dependent_sticking_preserves_plane_scope_friction_cap_and_dissipation():
    from scripts.fit_supermimic_toi_friction_radius import DirectionalSticking
    n = torch.nn.functional.normalize(torch.tensor([[1.,.2,.1],[0.,0.,1.],[1.,0.,0.]]),dim=-1)
    velocity = torch.tensor([[-2.,2.,4.],[-1.,0.,1.],[1.,2.,3.]])
    row = dict(normal=n,relative_velocity=velocity,active=torch.ones(3,dtype=torch.bool),
               object_id=torch.tensor([0,0,2]),impulse_dv=n*2,string_axes=torch.eye(3)[:,1:].expand(3,-1,-1))
    present = torch.ones(3,dtype=torch.bool)
    for slope in (-.75,.75):
        dv = DirectionalSticking(.32,2/7,slope).toi_response(row,present)['impulse_residual_dv']
        torch.testing.assert_close((dv*n).sum(1),torch.zeros(3),atol=1e-6,rtol=0.)
        assert (dv.norm(dim=1)<=.640001).all() and (dv*velocity).sum(1).max()<=0
        assert not dv[2].count_nonzero()
    torch.testing.assert_close(DirectionalSticking(.32,2/7,0.).toi_response(row,present)['impulse_residual_dv'],
                               TOISlip(.32,2/7).toi_response(row,present)['impulse_residual_dv'],rtol=0.,atol=0.)


def test_mesh_phase_uses_contact_point_and_only_modulates_string_parts():
    import json
    from pathlib import Path
    from scripts.fit_supermimic_toi_friction_radius import StripSeamSticking
    cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/full_racket_v4_r3.json').read_text())
    model=StripSeamSticking(.9,.25,.125,cfg,.025)
    row=dict(normal=torch.tensor([[1.,0.,0.]]).expand(5,-1), local=torch.zeros(5,3),
             relative_velocity=torch.tensor([[-2.,1.,0.]]).expand(5,-1),
             string_axes=torch.eye(3)[:,1:].expand(5,-1,-1),racket_part=torch.tensor([0,0,0,model.string_parts,0]),
             active=torch.ones(5,dtype=torch.bool),object_id=torch.tensor([0,0,0,0,1]),
             impulse_dv=torch.tensor([[3.,0.,0.]]).expand(5,-1))
    row['local'][:,2]=torch.tensor([0.,model.pitch/2,model.pitch,0.,0.])
    dv=model.toi_response(row,torch.ones(5,dtype=torch.bool))['impulse_residual_dv']
    torch.testing.assert_close(-dv[:,1],torch.tensor([.28125,.21875,.28125,.25,0.]))
