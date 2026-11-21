from scripts.audit_supermimic_native_window_response import window_signature


def test_contact_in_earlier_tick_survives_window_with_clean_terminal_frame():
    empty=(False,False,False,(),())
    hit=(False,False,True,(4,5),((1,1),(2,1)))
    assert window_signature([empty,hit,empty],160)==(False,False,True,(164,165),((1,1),(2,1)))


def test_force_on_net_anywhere_in_window_preserved():
    empty=(False,False,False,(),())
    assert window_signature([(False,True,False,(),()),empty],160)[:3]==(False,True,False)


def test_multiscale_derivative_keeps_native_one_sided_switch_out_of_smooth_group():
    from scripts.compare_supermimic_native_window_scales import compare

    reports = []
    for epsilon in (.01,.005,.0025):
        cases = []
        for lane in (96,127):
            difference = [4*epsilon,0.,0.]
            case = dict(lane=lane,group='racket_stable',signatures={'nominal_a': [1]},
                        ticks={'278': dict(native=difference,measured_racket=difference,body_racket=difference,
                            native_nominal=[1.,0.,0.],native_repeat=[0.,0.,0.],
                            native_left=[2*epsilon,0.,0.],
                            native_right=[2*epsilon if lane==96 else -2*epsilon,0.,0.])})
            cases.append(case)
        reports.append(dict(ticks=[278],request_directions=[[1.],[1.]],prefix_sha256='prefix',
                            task_contract='task',body_sha256='body',calibration_sha256='calibration',
                            material_sha256='material',epsilon_request=epsilon,cases=cases))
    groups = compare(reports)
    assert groups['timing_and_manifold_stable']['count']==2
    assert groups['native_convergent_stable']['case_ids']==[96]


def test_scale_reader_uses_compound_material_and_initial_shuttle_spin(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import torch
    import scripts.audit_supermimic_surface_derivative_scales as audit

    path = tmp_path/'plus'/'spec_16'/'trace.pt'
    path.parent.mkdir(parents=True)
    shuttle = torch.zeros(2,1,13)
    shuttle[:,:,10:13] = torch.tensor([1.,2.,3.])
    streams = {key:torch.zeros(2,1,4 if 'quaternion' in key else 3) for key in
               ('racket_position_w','racket_quaternion_w','racket_linear_velocity_w','racket_angular_velocity_w')}
    vector=torch.tensor([[1.,0.,0.]])
    intervention=dict(branch='plus',epsilon_mps=.05,delta_world=.05*vector,direction_world=vector,
                      nominal_shuttle=shuttle[0],joint_targets=vector,nominal_root_com_velocity=vector)
    torch.save(dict(seed=16,parent_family=0,parent_lane=[4],velocity_intervention=intervention,
                    raw=dict(substep_trace=dict(tensors=dict(**streams,shuttle_state_w=shuttle),
                                                count=2,physics_dt_s=.01))),path)
    material=torch.tensor([[.2,.1,.7]])
    def solve(prior,state,start,end,params,offset,previous_patch,**kwargs):
        assert kwargs['compound_racket'] and previous_patch is None
        torch.testing.assert_close(kwargs['initial_spin'],shuttle[0,:,10:13])
        torch.testing.assert_close(kwargs['racket_material'],material)
        return state,None
    monkeypatch.setattr(audit,'tick',solve)
    monkeypatch.setattr(audit,'signatures',lambda trace: [])
    audit.read_trial(tmp_path,16,'plus',.05,{'fitted':{},'solver':{'pair_contact_offset_m':.002}},
                     SimpleNamespace(steps=1,sub_dt=.01),material=material)


def test_crossed_motion_transports_world_angular_velocity_and_is_quaternion_sign_invariant():
    import torch
    from scripts.supermimic_racket_exchange import exchange_racket_path

    zero = torch.zeros(1, 3)
    identity = torch.tensor([[1., 0., 0., 0.]])
    quarter_turn = torch.tensor([[.5**.5, 0., 0., .5**.5]])
    base = (zero, identity, zero, torch.tensor([[1., 0., 0.]]))
    neutral = (zero, identity, zero, zero)
    signed = (zero+1, -quarter_turn, zero+2, zero)
    c, q, v, omega = exchange_racket_path(base, neutral, signed)
    torch.testing.assert_close(c, signed[0])
    torch.testing.assert_close(v, signed[2])
    torch.testing.assert_close(q, quarter_turn)
    torch.testing.assert_close(omega, torch.tensor([[0., 1., 0.]]))
    for observed, expected in zip(exchange_racket_path(base, neutral, neutral), base, strict=True):
        torch.testing.assert_close(observed, expected)


def test_signed_friction_strength_and_rotation_reconstruct_reversal_and_zero_contact():
    import torch
    from scripts.audit_supermimic_g07_impulse_channels import signed_friction_components

    minus=torch.tensor([[1.,0.,0.],[2.,0.,0.],[0.,0.,0.]])
    plus=torch.tensor([[-1.,0.,0.],[3.,0.,0.],[0.,1.,0.]])
    strength,turn=signed_friction_components(minus,plus)
    torch.testing.assert_close(strength+turn,plus-minus)
    torch.testing.assert_close(strength[0],torch.zeros(3))
    torch.testing.assert_close(turn[1],torch.zeros(3))


def test_substep_friction_bound_is_full_for_overlap_and_zero_at_end_of_step():
    import torch
    from quadsmash.rl.supermimic_compound_racket import friction_duration_factor

    gaps=torch.tensor([-.001,0.,.001,.002])
    speed=torch.full_like(gaps,-2.)
    torch.testing.assert_close(friction_duration_factor(gaps,speed,.001,1.),torch.tensor([1.,1.,.5,0.]))
