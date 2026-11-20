from types import SimpleNamespace

import torch

from quadsmash.rl.supermimic_actuator_identification import actuator_contract, drive_features
from quadsmash.rl.supermimic_physical_observation_v1 import JOINT_NAMES


def test_native_drive_rows_follow_joint_names_and_keep_per_lane_gains():
    ids = list(reversed(range(18)))
    d = SimpleNamespace(**{k:torch.ones(3,18) for k in
        ('joint_stiffness','joint_damping','joint_effort_limits','joint_vel_limits')},
        joint_pos_limits=torch.tensor([-1.,1.]).expand(3,18,2))
    d.joint_stiffness[1,17] = 7.
    env = SimpleNamespace(joint_ids=ids, num_envs=3, step_dt=.01,
        robot=SimpleNamespace(data=d,joint_names=list(reversed(JOINT_NAMES)),actuators={}),
        cfg=SimpleNamespace(sim=SimpleNamespace(dt=.001),decimation=10))
    result=actuator_contract(env,{'asset':{'sha256':'example'}})
    assert len(result['profiles'])==2
    assert [result['profiles'][i]['kp'][0] for i in result['lane_profile']]==[1,7,1]


def test_drive_torque_uses_target_error_damping_and_per_joint_saturation():
    state = torch.zeros(2,55)
    state[0,31:49] = .5
    target = torch.full((2,18),.1,requires_grad=True)
    kp = torch.full((18,),10.); kd=torch.ones(18)
    p=dict(kp=kp,kd=kd,effort_limit=torch.ones(18)*.75,velocity_limit=torch.ones(18))
    torque, x=drive_features(state,target,torch.zeros_like(target),torch.zeros_like(target),
                             torch.ones_like(target),p)
    torch.testing.assert_close(torque[:,0],torch.tensor([.5,.75]))
    torch.testing.assert_close(x[:,0,7],torch.tensor([0.,1.]))
    torque.sum().backward()
    torch.testing.assert_close(target.grad[:,0],torch.tensor([10.,0.]))


def test_target_increment_preserves_anchor_at_hold_and_only_replaces_arm_response():
    from scripts.audit_supermimic_pd_racket_transfer import advance_arm
    state = torch.zeros(2, 55)
    previous = torch.zeros(2, 18, dtype=torch.float64)
    target = previous.clone()
    target[1, 12] = .1
    profile = {k:torch.ones(18, dtype=torch.float64) for k in
               ('kp', 'kd', 'effort_limit', 'velocity_limit')}
    features = torch.zeros(18, 13, dtype=torch.float64)
    fit = dict(control_dt_s=.1, coefficients=(features, torch.ones_like(features),
               torch.zeros(18, 13, 2, dtype=torch.float64), torch.ones(18, dtype=torch.float64)))
    held = torch.full_like(state, .5)
    out = held.clone()
    out[:, :25] = 2
    advance_arm(out, state, target, previous, previous, previous, profile, fit, held)
    torch.testing.assert_close(out[0, 25:], held[0, 25:])
    torch.testing.assert_close(out[1, 25], torch.tensor(.5005))
    torch.testing.assert_close(out[1, 43], torch.tensor(.51))
    assert (out[:, :25] == 2).all()


def test_sphere_spin_oracle_uses_post_frame_and_recovers_tangential_impulse():
    from scripts.audit_supermimic_solver_action_response import sphere_tangent_from_spin
    n = torch.tensor([1.,0.,0.])
    dv = torch.tensor([0.,.3,-.2])
    radius = .025
    spin = torch.zeros(3,1,3)
    spin[1:] = -torch.cross(n,dv,dim=0)/(.4*radius)
    normal = torch.zeros_like(spin)
    normal[1] = 2*n
    torch.testing.assert_close(sphere_tangent_from_spin(spin,normal,radius),dv[None])


def test_measured_slip_is_capped_by_normal_loading():
    from scripts.audit_supermimic_solver_action_response import tangent_at_measured_contact
    zero = torch.zeros(2, 2, 3)
    shuttle = torch.zeros(2, 2, 13)
    surface = zero.clone()
    surface[..., 1] = 10
    trace = dict(shuttle_state_w=shuttle, racket_contact_position_w=zero,
                 racket_position_w=zero, racket_linear_velocity_w=surface,
                 racket_angular_velocity_w=zero)
    normal_dv = torch.tensor([[[2., 0., 0.], [0., 0., 0.]]])
    prediction = tangent_at_measured_contact(trace, .025, normal_dv, .1)
    torch.testing.assert_close(prediction, torch.tensor([[0., .2, 0.], [0., 0., 0.]]))


def test_offcenter_contact_normal_impulse_also_changes_tangential_speed():
    from scripts.audit_supermimic_solver_action_response import tangent_from_contact_arm
    radius = .025
    arm = torch.tensor([[-radius,.001,0.],[-radius,0.,0.]])
    velocity = torch.tensor([[-2.,.1,0.],[-2.,.1,0.]])
    normal = torch.tensor([[3.,0.,0.],[3.,0.,0.]])
    impulse = tangent_from_contact_arm(velocity,arm,normal,radius,1.)
    total = impulse+normal
    spin = torch.cross(arm,total,dim=-1)/(.4*radius**2)
    point_velocity = velocity+total+torch.cross(spin,arm,dim=-1)
    torch.testing.assert_close(point_velocity[:,1:], torch.zeros_like(point_velocity[:,1:]),atol=1e-6,rtol=0)
    torch.testing.assert_close(impulse[1,1],torch.tensor(-.1*2/7))
    assert impulse[0,1] < impulse[1,1]


def test_signed_impulse_breakdown_preserves_clock_and_branch_order():
    from scripts.audit_supermimic_solver_action_response import signed_impulse_components
    minus = torch.tensor([[[1., 0., 0.]], [[1., 0., 0.]], [[0., 0., 0.]], [[0., 0., 0.]]])
    plus = torch.tensor([[[0., 2., 0.]], [[0., 0., 0.]], [[0., 0., 3.]], [[0., 0., 0.]]])
    parts = signed_impulse_components(minus, plus)
    torch.testing.assert_close(parts['amplitude_on_shared_substeps'], torch.tensor([[.5, .5, 0.]]))
    torch.testing.assert_close(parts['direction_on_shared_substeps'], torch.tensor([[-1.5, 1.5, 0.]]))
    torch.testing.assert_close(parts['unmatched_substeps'], torch.tensor([[-1., 0., 3.]]))
    torch.testing.assert_close(sum(parts.values()), (plus-minus).sum(0))
    reverse = signed_impulse_components(plus, minus)
    for key in parts:
        torch.testing.assert_close(parts[key], -reverse[key])


def test_native_clock_separates_normal_load_and_surface_direction():
    from types import SimpleNamespace
    from scripts.audit_supermimic_solver_action_response import normal_on_native_clock
    pos = torch.zeros(3,2,3)
    quat = torch.zeros(3,2,4)
    quat[...,0] = 1
    shuttle = torch.zeros(3,2,13)
    shuttle[0,0,7:10] = torch.tensor([-3.,-2.,0.])
    forces = torch.zeros_like(pos)
    forces[1,0,1] = 4.
    def surface(local):
        return local.new_tensor([1.,0.,0.]).expand_as(local), None
    prior = SimpleNamespace(sub_dt=.01,restitution=0.,racket_surface=surface,
                            preview=SimpleNamespace(radius=.02))
    trace = dict(racket_force_w=forces,racket_position_w=pos,racket_quaternion_w=quat,
                 racket_linear_velocity_w=pos,racket_angular_velocity_w=pos,shuttle_state_w=shuttle)
    terms = normal_on_native_clock(trace,prior,{'normal_impulse_scale':1.},.01)
    expected = {'measured_load_geometry_axis':[4.,0.,0.],'model_load_measured_axis':[0.,2.,0.],
                'model_load_geometry_axis':[3.,0.,0.]}
    for name, v in expected.items():
        torch.testing.assert_close(terms[name],torch.tensor([v,[0.,0.,0.]]))


def test_manifold_torque_uses_individual_loaded_points_after_incoming_frame():
    from scripts.audit_supermimic_solver_action_response import manifold_normal_spin
    positions = torch.zeros(2,2,13)
    positions[0,0,0] = 2
    positions[1,0,0] = 5
    measured = torch.zeros(2,2,3)
    measured[1,0,1] = 3
    frame = dict(lane=torch.tensor([0,0,0,1]),filter_index=torch.tensor([0,0,1,0]),
                 normal_force_n=torch.tensor([1.,2.,50.,60.]),
                 normal_w=torch.tensor([[0.,1.,0.]]).expand(4,3),
                 point_w=torch.tensor([[3.,0.,0.],[1.,0.,0.],[0.,0.,0.],[0.,0.,0.]]))
    tr=dict(count=2,physics_dt_s=.1,tensors=dict(shuttle_state_w=positions,racket_force_w=measured),
            manifold_frames=[{},frame])
    spin,count=manifold_normal_spin(tr,0,1.,1.)
    torch.testing.assert_close(spin[1],torch.tensor([0.,0.,-.25]))
    assert count.tolist()==[0,2]
    assert manifold_normal_spin(dict(tr,manifold_frames=[]),0,1.,1.) is None


def test_multiple_friction_constraints_share_one_sphere_momentum():
    from scripts.audit_supermimic_pilot_friction_anchors import sticking_delta
    n=torch.tensor([0.,0.,1.],dtype=torch.float64)
    arm=-.025*n
    slip=torch.tensor([2.,0.,0.],dtype=n.dtype)
    for count in (1,2):
        actual=sticking_delta(arm.expand(count,3),slip.expand(count,3),n,.4*.025**2,10.)
        torch.testing.assert_close(actual,-2/7*slip)
    capped=sticking_delta(arm[None],slip[None],n,.4*.025**2,.1)
    torch.testing.assert_close(capped,-.05*slip)
