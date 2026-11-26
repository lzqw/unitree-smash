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
