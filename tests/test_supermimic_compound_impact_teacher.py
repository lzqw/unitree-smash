import torch

from scripts.audit_supermimic_compound_impact_teacher import solve_patch
from scripts.audit_supermimic_paired_racket_fixture import classify


def test_single_surface_impulse_and_duplicate_inactive_shape():
    n=torch.tensor([[[0.,0.,1.],[1.,0.,0.]]])
    mask=torch.tensor([[True,False]])
    mu=torch.full((1,2),.5)
    restitution=torch.full_like(mu,.7)
    velocity=torch.tensor([[1.,0.,-2.]])
    normal,tangent=solve_patch(n,mu,restitution,velocity,torch.zeros_like(velocity),
        torch.zeros_like(n),.025,mask,1.,4)
    torch.testing.assert_close(normal,torch.tensor([[0.,0.,3.4]]))
    torch.testing.assert_close(tangent,torch.tensor([[-2/7,0.,0.]]))


def test_fixed_time_but_changed_loaded_points_is_reported_separately():
    base=(False,False,True,(12,),((2,1),))
    changed=(*base[:4],((1,1),))
    assert classify([base]*4)=='racket_stable'
    assert classify([base,base,changed,base])=='manifold_switch'
    assert classify([base,base,(False,False,True,(11,),((2,1),)),base])=='toi_switch'
    assert classify([base,changed,base,base])=='nominal_mismatch'


def test_velocity_intervention_uses_native_initial_twist_and_preserves_later_samples():
    from scripts.collect_supermimic_dynamic_contact_pilot import record_intervened_initial_twist
    trace=dict(tensors={k:torch.zeros(3,2,3) for k in ('racket_linear_velocity_w','racket_angular_velocity_w')},
               contact_metadata={})
    trace['tensors']['racket_linear_velocity_w'][1:]=4.
    intervention=dict(native_velocity_readback={'after':{'racket_link':torch.ones(2,3),'racket_angular':torch.zeros(2,3)}})
    record_intervened_initial_twist(trace,intervention)
    assert trace['tensors']['racket_linear_velocity_w'][0].eq(1).all()
    assert trace['tensors']['racket_linear_velocity_w'][1:].eq(4).all()
    assert intervention['sdk_cached_initial_twist']['racket_link'].eq(0).all()


def test_velocity_pair_compares_link_readback_at_same_forward_boundary():
    import pytest
    from scripts.audit_supermimic_paired_racket_fixture import common_native_twist, verify_common_twist
    def observed(delta):
        before=dict(root_com=torch.zeros(1,6),racket_link=torch.zeros(1,3),
                    racket_angular=torch.zeros(1,3),dof=torch.zeros(1,18))
        after=dict(before)
        after['racket_link']=delta+0.0001
        return dict(delta_world=delta,native_velocity_readback=dict(before=before,after=after))
    ref=common_native_twist(observed(torch.zeros(1,3)))
    plus=common_native_twist(observed(torch.tensor([[0.05,0.,0.]])))
    verify_common_twist(ref,plus)
    plus['after_racket_reference'][0,0]+=.0002
    with pytest.raises(AssertionError):
        verify_common_twist(ref,plus)


def test_coupled_manifold_recovers_two_mass_normal_impulse():
    from scripts.audit_supermimic_articulated_manifold import coupled_impulse
    normal=torch.tensor([[0.,0.,1.]],dtype=torch.float64)
    inverse=torch.zeros(6,6,dtype=normal.dtype)
    inverse[2,2]=2.
    n,f=coupled_impulse(normal,-3*normal,-.025*normal,torch.zeros_like(normal),inverse,
                         radius=.025,mass=.0052,restitution=.72,mu=.08)
    torch.testing.assert_close(n,normal[0]*(3*1.72/(1+.0052*2)))
    torch.testing.assert_close(f,torch.zeros_like(f))


def test_model_only_basin_excludes_switched_contact_part_and_boundary():
    from scripts.audit_supermimic_predictive_surface_basin import stable_prediction
    events=[]
    speeds=[]
    for i,x in enumerate([0.,-.05,.05,-.025,.025,-.0125,.0125]):
        ids=torch.full((2,2),-1,dtype=torch.long)
        ids[0]=0
        if i==6:ids[0,1]=1
        events.append(dict(racket=ids,ground=torch.zeros_like(ids,dtype=torch.bool),net=torch.zeros_like(ids,dtype=torch.bool)))
        speeds.append(torch.tensor([[x,0.,0.],[x,0.,0.]]))
    stable, qualified, derivative=stable_prediction(events,speeds,[.05,.025,.0125])
    assert stable.tolist()==qualified.tolist()==[True,False]
    torch.testing.assert_close(derivative,speeds[2]/.05)


def test_unloaded_native_shadow_is_rejected_as_a_measured_ball_response(tmp_path):
    import pytest
    from quadsmash.rl.supermimic_contact_pilot_v1 import _trace_candidates
    from scripts.calibrate_supermimic_pilot_oracle import trajectory
    packet = {'schema': 'native-unloaded-racket-oracle-v1'}
    path = tmp_path / 'trace.pt'
    torch.save(packet, path)
    with pytest.raises(ValueError):
        _trace_candidates(path, .001, 4)
    with pytest.raises(ValueError):
        trajectory(packet, None)


def test_free_articulation_readout_preserves_uniform_root_velocity_change():
    from scripts.fit_supermimic_unloaded_racket_motion import predict_motion
    t = torch.arange(4) * .0025
    features = torch.zeros(2, 3)
    residual = torch.zeros(4, 12, dtype=torch.double)
    residual[1:, 6] = .2
    model = dict(mean=torch.zeros(3), scale=torch.ones(3),
                 weights=torch.zeros(3, 48, dtype=torch.double), intercept=residual.flatten())
    c = torch.zeros(2, 3)
    q = torch.tensor([[1., 0., 0., 0.]]).expand(2, -1)
    delta = torch.tensor([[.1, -.02, 0.], [-.04, 0., .05]])
    base = predict_motion(model, features, (c, q, c, c), t)
    shifted = predict_motion(model, features, (c, q, delta, c), t)
    torch.testing.assert_close(shifted[0] - base[0], t[:, None, None] * delta)
    torch.testing.assert_close(shifted[2] - base[2], delta.expand_as(base[2]))
    torch.testing.assert_close(shifted[1], base[1])
    torch.testing.assert_close(shifted[3], base[3])


def test_native_fixture_uses_measured_court_origin_and_rejects_wrong_layout():
    import pytest
    from quadsmash.rl.supermimic_fixture_coordinates import fixture_court_origin
    position = torch.tensor([[81.5,0.,1.],[1.5,16.,1.]])
    layout = dict(mode='spaced16',environment_spacing_m=16.,origins_w=[[80.,0.,0.],[0.,16.,0.]])
    packet = dict(fixture_layout=layout)
    origin = fixture_court_origin(packet,position)
    torch.testing.assert_close(position-origin, torch.tensor([[1.5,0.,1.]]).expand_as(position))
    with pytest.raises(ValueError):
        fixture_court_origin(packet, position, dict(fixture_layout='teacher',expected_origin_spacing_m=0.))
    with pytest.raises(ValueError):
        fixture_court_origin({}, position, dict(fixture_layout='teacher',expected_origin_spacing_m=0.))


def test_signed_arm_drive_fit_preserves_leg_readout_and_fits_measured_arm_difference():
    from scripts.fit_supermimic_signed_arm_drive import solve_update

    generator = torch.Generator().manual_seed(73)
    natural = torch.randn(24, 18, 3, dtype=torch.float64, generator=generator)
    delta = torch.randn(32, 18, 3, dtype=torch.float64, generator=generator)
    parent = torch.randn(18, 3, 2, dtype=torch.float64, generator=generator)
    changed = parent.clone()
    changed[12:, 0, 0] += 0.5
    natural_target = torch.einsum('njf,jfo->njo', natural, parent)
    signed_target = torch.einsum('njf,jfo->njo', delta, changed)
    fitted = solve_update(natural, natural_target, delta, signed_target, parent)
    torch.testing.assert_close(fitted[:12], parent[:12], atol=0, rtol=0)
    error = lambda coef: (torch.einsum('njf,jfo->njo', delta, coef) - signed_target).square().sum()
    assert error(fitted) < error(parent)
    unchanged = solve_update(natural, natural_target, torch.zeros_like(delta), torch.zeros_like(signed_target), parent)
    torch.testing.assert_close(unchanged, parent)
