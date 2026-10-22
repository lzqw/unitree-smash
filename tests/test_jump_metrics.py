from types import SimpleNamespace
import torch
from quadsmash.tasks.stand import jump_metrics as metrics

PROTOCOL={'jump':{'minimum_total_com_rise_from_takeoff_m':.08,'minimum_continuous_all_feet_airborne_s':.1},
    'recovery':{'continuous_stable_window_s':.8,'maximum_abs_roll_pitch_deg':15.,'maximum_horizontal_root_speed_m_s':.3,
        'maximum_root_angular_speed_rad_s':1.,'maximum_abs_joint_velocity_rad_s':1.,'maximum_body_ground_force_n':1.}}

def new():
    state=SimpleNamespace();metrics.initialize(state,1,'cpu');return state

def tick(s,contact=True,height=.4,clearance=.02,speed=0.,force=0.,dt=.01):
    metrics.advance(s,dt=dt,time=torch.zeros(1),com_height=torch.tensor([height]),
        foot_contact=torch.full((1,4),contact),foot_clearance=torch.tensor([clearance]),roll_pitch=torch.zeros(1,2),
        horizontal_speed=torch.tensor([speed]),angular_speed=torch.zeros(1),joint_speed=torch.zeros(1),
        body_ground_force=torch.tensor([force]),protocol=PROTOCOL)

def qualified_flight(s):
    tick(s)
    tick(s,False,.4,.04)
    for _ in range(12):tick(s,False,.5,.12)
    tick(s)

def test_real_com_flight_and_continuous_recovery():
    s=new();qualified_flight(s)
    for _ in range(80):tick(s)
    assert metrics.success(s).item()

def test_accumulated_short_recovery_windows_do_not_pass():
    s=new();qualified_flight(s)
    for _ in range(5):
        for _ in range(30):tick(s)
        tick(s,speed=.31)
    assert not metrics.success(s).item()
    assert s.recovery_time.item()<.4

def test_height_and_duration_cannot_be_combined_across_different_jumps():
    s=new();tick(s)
    tick(s,False,.4,.04)
    for _ in range(4):tick(s,False,.5,.12)
    tick(s)
    tick(s,False,.4,.04)
    for _ in range(15):tick(s,False,.43,.06)
    tick(s)
    for _ in range(80):tick(s)
    assert s.max_com_rise.item()>.08 and s.airborne_time.item()>.1
    assert not metrics.success(s).item()

def test_no_fake_airborne_from_initial_floating_or_standing():
    s=new()
    for _ in range(200):tick(s,False,2.,1.)
    assert not s.airborne_seen.item() and not metrics.success(s).item()
    s=new()
    for _ in range(200):tick(s)
    assert not metrics.success(s).item()

def test_body_landing_invalidates_success_and_partial_reset_clears():
    s=new();qualified_flight(s)
    tick(s,force=2.)
    for _ in range(100):tick(s)
    assert not metrics.success(s).item()
    metrics.reset(s,torch.tensor([0]))
    assert not s.support_seen.item() and s.max_com_rise.item()==0.

def test_recovery_before_qualified_jump_cannot_count():
    s=new();tick(s);tick(s,False,.4,.04)
    for _ in range(15):tick(s,False,.43,.06)
    for _ in range(100):tick(s)
    assert s.recovery_seen.item()
    qualified_flight(s)
    assert not metrics.success(s).item()
