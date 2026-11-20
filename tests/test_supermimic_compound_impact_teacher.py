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


def test_compound_spin_and_closing_speed_gradient_remain_consistent_with_signed_response():
    n = torch.tensor([[[0., 0., 1.], [0., 0., 1.]]], dtype=torch.double)
    mu = torch.full((1, 2), .8, dtype=n.dtype)
    v = torch.tensor([[.2, .4, -2.]], dtype=n.dtype)
    surface = torch.tensor([[[.1, .0, .0], [.1, .0, .0]]], dtype=n.dtype)
    mask = torch.ones(1, 2, dtype=torch.bool)
    radius = .025
    def step(x):
        normal, friction, angular = solve_patch(n, mu, mu, v, torch.zeros_like(v), surface + x, radius, mask,
                                                1., 4, return_spin=True)
        torch.testing.assert_close(angular, torch.linalg.cross(-radius*n[:, 0], friction)/(.4*radius**2))
        return normal + friction
    alpha = torch.tensor(0., dtype=n.dtype, requires_grad=True)
    jac = torch.autograd.functional.jacobian(step, alpha)
    torch.testing.assert_close(jac, (step(alpha + 1.e-6) - step(alpha - 1.e-6))/2.e-6)


def test_compound_contact_mid_substep_uses_remaining_time_and_incoming_surface():
    from scripts.evaluate_supermimic_contact_rollout import BallImpulsePriorV1, tick
    from quadsmash.rl.supermimic_compound_racket import compound_step
    prior = BallImpulsePriorV1().double()
    g = prior.preview
    center = torch.zeros((1,3),dtype=torch.double)
    rotation = torch.tensor([[1.,0.,0.,0.]],dtype=torch.double)
    count = len(g.racket_box_centers)+len(g.racket_round_centers)
    material = torch.tensor([0.,0.,.72],dtype=torch.double).expand(count,-1)
    strip = len(g.racket_box_centers)//2
    p = g.racket_box_centers[strip:strip+1].clone()
    p[:,0] += g.racket_box_halves[strip,0]+g.radius+prior.sub_dt
    v = torch.tensor([[-2.,0.,0.]],dtype=torch.double)
    n,t,spin,active,toi = compound_step(prior,p,v,center,center,rotation,center,center,
        material,.0002,1.,impact_geometry_fraction=1.,return_impact_time=True)
    assert active.item() > 0
    torch.testing.assert_close(toi,torch.tensor([prior.sub_dt/2],dtype=p.dtype),rtol=1e-8,atol=1e-12)
    torch.testing.assert_close(n+t,torch.tensor([[3.44,0.,0.]],dtype=p.dtype))
    # Remove gravity, drag, and court collisions; check the endpoint displacement over this interval.
    prior.gravity.zero_()
    prior.drag = 0.
    center = center + torch.tensor([[0.,0.,4.]],dtype=p.dtype)
    p = p + center
    start = (center,rotation,torch.zeros_like(v),torch.zeros_like(v))
    end,_ = tick(prior,torch.cat((p,v),-1),start,start,dict(normal_impulse_scale=1.,impact_geometry_fraction=1.),
                 .0002,racket_material=material,compound_racket=True)
    expected = p + v*toi[:,None] + (v+n+t)*(prior.dt-toi[:,None])
    torch.testing.assert_close(end[:,:3],expected,rtol=1e-8,atol=1e-10)


def test_compound_basin_requires_same_loaded_shape_and_friction_regime():
    from scripts.audit_supermimic_predictive_surface_basin import stable_prediction
    events=[]
    speeds=[]
    for i,dx in enumerate([0.,-.05,.05,-.025,.025,-.0125,.0125]):
        regime=torch.tensor([[[1,0],[1,0],[1,0]]],dtype=torch.uint8)
        if i==6:
            regime[0,1]=torch.tensor([0,1])
            regime[0,2]=torch.tensor([2,0])
        flag=torch.zeros(1,3,dtype=torch.bool)
        events.append(dict(racket=torch.zeros(1,3,dtype=torch.long),ground=flag,net=flag,shape_regime=regime))
        speeds.append(torch.tensor([[dx,0.,0.]]).expand(3,-1))
    events_stable,derivatives_stable,_=stable_prediction(events,speeds,[.05,.025,.0125])
    assert events_stable.tolist()==derivatives_stable.tolist()==[True,False,False]


def test_finite_racket_mass_reduces_elastic_normal_impulse_at_known_mass_ratio():
    from quadsmash.rl.supermimic_mobile_racket_contact import solve_mobile_patch
    normal=torch.tensor([[[1.,0.,0.]]])
    v=torch.tensor([[-2.,0.,0.]])
    one=torch.ones(1,1)
    mobility=torch.zeros(1,6,6)
    mobility[:,:3,:3]=torch.eye(3)*0.5
    dv,df,spin=solve_mobile_patch(normal,0*one,one,v,torch.zeros_like(v),0*normal,0*normal,.025,
                                  one.bool(),1.,4,mobility,1.)
    torch.testing.assert_close(dv,torch.tensor([[4/1.5,0.,0.]]))
    torch.testing.assert_close(df+spin,0*v)


def test_zero_racket_mobility_matches_prescribed_surface_for_two_contacts():
    from quadsmash.rl.supermimic_mobile_racket_contact import solve_mobile_patch
    normals=torch.tensor([[[1.,0.,0.],[0.,1.,0.]]])
    mu=torch.tensor([[.15,.3]])
    v=torch.tensor([[-3.,-1.,.2]])
    w=torch.tensor([[.5,0.,1.]])
    s=torch.zeros_like(normals)
    active=torch.ones_like(mu,dtype=torch.bool)
    base=solve_patch(normals,mu,mu,v,w,s,.025,active,.96,4,return_spin=True)
    mobile=solve_mobile_patch(normals,mu,mu,v,w,s,s,.025,active,.96,4,torch.zeros(1,6,6),.0052)
    for x,y in zip(base,mobile):
        torch.testing.assert_close(x,y,rtol=1e-5,atol=1e-5)


def test_spin_limit_propagates_free_rotation_without_changing_translational_flight():
    from scripts.evaluate_supermimic_contact_rollout import BallImpulsePriorV1, tick
    prior = BallImpulsePriorV1()
    ball = torch.tensor([[5., 0., 4., 1., 2., 3.]])
    zero = torch.zeros(1,3)
    pose = (zero, torch.tensor([[1.,0.,0.,0.]]), zero, zero)
    spin = torch.tensor([[0.,-120.,160.]])
    kwargs = dict(initial_spin=spin, return_spin=True)
    free, _, original_spin = tick(prior, ball, pose, pose, {}, .0002, **kwargs)
    capped, _, capped_spin = tick(prior, ball, pose, pose, {'pre_solve_spin_limit_rad_s':100.}, .0002, **kwargs)
    torch.testing.assert_close(free, capped)
    torch.testing.assert_close(original_spin, spin)
    torch.testing.assert_close(capped_spin, spin/2)


def test_motion_derivative_uses_each_lanes_complete_motion_and_shortest_quaternion_arc():
    from scripts.audit_supermimic_surface_path_jacobian import interpolate_motion
    alpha = torch.tensor([.3,.7], requires_grad=True)
    p = torch.zeros(3,2,3)
    delta = torch.arange(1,7).reshape(3,2,1).expand_as(p).float()
    q = torch.zeros(3,2,4)
    q[:,:,0] = 1.
    motion = interpolate_motion((p,q,p,p),(delta,-q,delta,delta),alpha)
    torch.testing.assert_close(motion[1],q)
    grad, = torch.autograd.grad(motion[0][:,:,0].sum(), alpha)
    torch.testing.assert_close(grad,delta[:,:,0].sum(0))


def test_frozen_readout_windows_exclude_padding_future_events_and_protected_lanes():
    from scripts.calibrate_supermimic_ball_event_logits import training_groups
    data = {k:torch.zeros(5,2,dtype=torch.bool) for k in ('present','terminated','timeout')}
    data['present'].fill_(True)
    data['episode'] = torch.zeros(5,2,dtype=torch.long)
    data['event_state'] = torch.zeros(5,2,8)
    data['next_event_state'] = torch.zeros(5,2,8)
    data['next_event_state'][3,:,1] = 1
    data['valid_lengths'] = torch.tensor([3,5])
    groups = training_groups(data,[0])
    assert not any(mask.any() for k,mask in groups.items() if k!='ordinary')
    assert groups['ordinary'][:,0].tolist() == [True,True,True,False,False]
    assert not groups['ordinary'][:,1].any()


def test_temporal_patch_restitution_and_no_impulse_outside_contact():
    from quadsmash.rl.supermimic_temporal_contact_patch import solve_temporal_patch
    n = torch.tensor([[[1.,0.,0.]],[[1.,0.,0.]]])
    gaps = torch.zeros(2,1)
    v = torch.tensor([[-2.,0.,0.],[-2.,0.,0.]])
    mask = torch.tensor([[True],[False]])
    normal,tangent,spin = solve_temporal_patch(n,gaps,0*gaps,.72+gaps,v,v*0,n*0,.025,mask,
                                              1.,.0000625,8,2,.05)
    ideal = torch.tensor([[3.44,0.,0.],[0.,0.,0.]])
    refined,_,_ = solve_temporal_patch(n,gaps,0*gaps,.72+gaps,v,v*0,n*0,.025,mask,
                                       1.,.0000625,16,2,.05)
    # A finite displacement solve converges to the isolated Newton rebound.
    assert (refined-ideal).abs().max() < (normal-ideal).abs().max()
    torch.testing.assert_close(refined,ideal)
    torch.testing.assert_close(normal[1],v[1]*0)
    torch.testing.assert_close(tangent+spin,v*0)


def test_native_shape_identity_rejects_ambiguous_point_and_equal_count_shape_switch():
    from scripts.audit_supermimic_native_shape_identity import infer_shape,shape_class
    distances = torch.tensor([[.0001,.00011,.01],[.004,-.0001,.02]])
    assert infer_shape(distances,torch.tensor([.0001,.0003])).tolist()==[-1,1]
    first=[(18,(3,4))]
    assert shape_class([first,first,[(18,(3,5))]])=='changed'
    assert shape_class([first,[(18,(-1,4))]])=='unresolved'
    assert shape_class([first,first])=='consistent'


def test_signed_g07_reset_precedes_earliest_load_in_all_branches():
    from scripts.audit_supermimic_g07_incoming_shuttle import reset_frames
    start=lambda *times:(False,False,bool(times),times,())
    cases=[{'signatures':{'minus':start(319),'plus':start(320)}},
           {'signatures':{'minus':start(),'plus':start()}}]
    assert reset_frames(cases,160).tolist()==[[-1,-1],[160,-1],[319,-1]]


def test_authored_shuttle_inertia_recovers_reference_with_multiple_and_inactive_contacts():
    from quadsmash.rl.supermimic_compound_racket import solve_patch
    normals = torch.eye(3, dtype=torch.float64)[:2].expand(3, 2, 3)
    velocity = torch.tensor([[-2., -1., .45]], dtype=torch.float64).expand(3, 3)
    active = torch.tensor([[True, False], [True, True], [False, False]])
    surface = torch.zeros_like(normals)
    mu = torch.full_like(normals[:, :, 0], .29)
    args = (normals, mu, torch.full_like(mu, .72), velocity, velocity*0,
            surface, .025, active, 1., 8)
    reference = solve_patch(*args, return_spin=True)
    explicit = solve_patch(*args, return_spin=True, inertia_ratio=velocity.new_full((3,), .4))
    for expected, actual in zip(reference, explicit, strict=True):
        torch.testing.assert_close(expected, actual)
        torch.testing.assert_close(actual[-1], torch.zeros_like(actual[-1]))


def test_friction_trace_integrates_native_forces_and_retains_missing_normal_evidence():
    from types import SimpleNamespace
    from scripts.audit_supermimic_friction_load_response import friction_trace
    geometry = SimpleNamespace(**{k: None for k in ('racket_round_centers', 'racket_round_axes',
                               'racket_round_halves', 'racket_round_radii', 'racket_round_capsule')})
    empty = {'lane':torch.empty(0,dtype=torch.long), 'filter_index':torch.empty(0,dtype=torch.long),
             'normal_force_n':torch.empty(0)}
    def sample(force):
        return {'lane':torch.tensor([1]),'force_w_n':torch.tensor([force])}
    trace = {'tensors':{'shuttle_state_w':torch.zeros(3,2,13)},'physics_dt_s':.001,
             'manifold_frames':[empty]*3,
             'racket_friction_frames':[None,sample([0.,2.,0.]),sample([0.,-1.,0.])]}
    response, records = friction_trace(trace, geometry, None, .01)
    torch.testing.assert_close(response['actual'],torch.tensor([[0.,0.,0.],[0.,.1,0.]],dtype=torch.float64))
    assert [r['step'] for r in records] == [0,1]
    assert all(r['lane']==1 and r['normal_points']==0 for r in records)


def test_free_drive_preview_uses_preimpact_pose_and_only_the_unloaded_increment():
    from scripts.audit_supermimic_unloaded_racket_transfer import preimpact_drive_preview
    pose = torch.tensor([[[1.,0.,0.]], [[2.,0.,0.]], [[3.,0.,0.]]])
    quat = torch.ones(3,1,4)
    drive = torch.tensor([[[0.,0.,0.]], [[.1,0.,0.]], [[.4,0.,0.]]])
    output = preimpact_drive_preview([pose,quat,pose,2*pose], [2*pose,quat,drive,3*drive])
    torch.testing.assert_close(output[0],pose[:-1])
    torch.testing.assert_close(output[1],quat[:-1])
    torch.testing.assert_close(output[2],pose[:-1]+drive.diff(dim=0))
    torch.testing.assert_close(output[3],2*pose[:-1]+3*drive.diff(dim=0))


def test_solver_consensus_rejects_a_shifted_impact_or_reversed_response():
    from scripts.audit_supermimic_predictive_surface_basin import solver_consensus
    slots=torch.zeros(2,3,dtype=torch.long)
    event={k:slots.clone() for k in ('racket','ground','net','shape_regime')}
    shifted={k:v.clone() for k,v in event.items()}
    shifted['racket'][1,0]=1
    x=torch.ones(3,3)
    other=x.clone()
    other[1]=-1
    valid=torch.ones(3,dtype=torch.bool)
    assert solver_consensus([event],[shifted],x,other,valid,valid).tolist()==[False,False,True]


def test_friction_load_floor_caps_tangent_without_adding_normal_impulse():
    from quadsmash.rl.supermimic_compound_racket import solve_patch
    from quadsmash.rl.supermimic_temporal_contact_patch import solve_temporal_patch
    normals=torch.tensor([[[1.,0.,0.]],[[1.,0.,0.]]])
    vel=torch.tensor([[0.,1.,0.],[0.,1.,0.]])
    surface=torch.zeros_like(normals)
    active=torch.tensor([[True],[False]])
    friction=torch.full((2,1),.3)
    zero=torch.zeros(2,1)
    opts={'friction_load_floor':torch.full((2,1),.1)}
    for normal,tangent,_ in (
        solve_patch(normals,friction,zero,vel,vel*0,surface,.025,active,1.,4,return_spin=True,**opts),
        solve_temporal_patch(normals,zero,friction,zero,vel,vel*0,surface,.025,active,1.,.0000625,8,2,.05,**opts)):
        torch.testing.assert_close(normal,vel*0)
        assert tangent[0,1]<0 and tangent[0].norm()<=.030001
        torch.testing.assert_close(tangent[1],vel[1]*0)
