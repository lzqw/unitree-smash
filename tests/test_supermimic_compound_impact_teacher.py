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
