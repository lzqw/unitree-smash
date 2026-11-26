from types import SimpleNamespace

import torch

from scripts.recovery_native_teachers import NativeTeachers


def test_feedback_applies_only_to_first_episode_released_probe_branches():
    teachers = NativeTeachers('cpu')
    obs = torch.ones(128, 794)
    nominal = torch.zeros(128, 18)
    idx = torch.arange(128)
    had_contact = idx % 5 != 0
    released = idx % 3 != 0
    episode = (idx % 7 == 0).long()
    rally = SimpleNamespace(had_contact=had_contact, released=released)
    action = teachers.apply(0, obs, nominal, rally, episode)
    live = had_contact & released & (episode == 0)
    changed = action.ne(nominal).any(-1)
    assert torch.equal(changed, (idx >= 32) & live)
    assert torch.equal(teachers.frames[0][-1], live)
    assert action.abs().max() <= .16


def test_release_delay_and_airborne_symmetry():
    teacher = NativeTeachers('cpu')
    obs = torch.ones(128, 794)
    obs[:, 57:61] = 0
    action = torch.zeros(128, 18)
    live = torch.ones(128, dtype=torch.bool)
    rally = SimpleNamespace(had_contact=live, released=live)
    for tick in range(5):
        out = teacher.apply(tick, obs, action, rally, live.long()-1)
        assert bool(out[64, 12:].any()) == (tick >= 4)
        assert torch.equal(out[64,4:8],out[64,4].expand(4))
    rally.released = ~live
    teacher.apply(5,obs,action,rally,live.long()-1)
    rally.released = live
    assert not teacher.apply(6,obs,action,rally,live.long()-1)[:,12:].any()


def test_focus_regularizes_all_eleven_ball_frames():
    from scripts.distill_native_recovery import focus_columns
    col = focus_columns(torch.zeros(3, 794))
    for frame in col[:759].view(11,69):
        assert (frame[61:67] == 12).all() and (frame[57:61] == .3).all()
    assert (col[776:] == 10).all()


def test_grounded_feedback_uses_gravity_derivative_signs_and_horizontal_braking():
    x = torch.zeros(4, 794)
    x[:, 57] = 1.
    x[1, 3] = 1
    x[2, 4] = 1
    x[3, 0:2] = 1
    delta = NativeTeachers('cpu').feedback(x)
    assert delta[1, 0] < delta[0, 0] and delta[2, 4] > delta[0, 4]
    assert delta[3, 0] < delta[0, 0] and delta[3, 4] < delta[0, 4]


def test_single_lucky_teacher_does_not_trigger_expansion():
    from scripts.score_recovery_probe_round import compare
    rows = [dict(case_id=str(i % 32), numerical_failures=0, body_ground_seen=True,
                 physical_recovery_success=False, strict_physical_recovery=False,
                 robot_net_fault=False, ball_side_complete_except_recovery=True,
                 qualified_airborne_string_contact=True) for i in range(128)]
    rows[32].update(body_ground_seen=False, physical_recovery_success=True, strict_physical_recovery=True)
    assert not compare(rows)['expand']
    for row in rows[33:40]:
        row.update(body_ground_seen=False, physical_recovery_success=True, strict_physical_recovery=True)
    assert compare(rows)['qualified_branches'] == [1]
