import copy
import pytest
from quadsmash.rl.supermimic_task_contract import require_resume_contract, summarize_episode
from quadsmash.rl.supermimic_task_contract import zero_panel_reset_noise


def terminal_record():
    return dict(env_id=7, episode_index=0, episode_duration_s=8., episode_timeout=True,
                qualified_airborne_string_contact=True,
                ball_side_complete_except_recovery=False, recovery_success=True,
                complete5=False, body_ground_seen=False, rally_fault=False,
                numerical_failures=0, ground_seen=True)


def test_robot_ground_not_inferred_from_shuttle_ground():
    row = summarize_episode(terminal_record())
    assert row['body_ground_seen'] is False
    assert row['ball_side_complete_except_recovery'] is False


def test_missing_independent_audit_is_unknown_not_clean():
    row = summarize_episode(terminal_record())
    assert row['head_contact'] is None
    assert row['independent_nonfoot_contact'] is None


@pytest.mark.parametrize('field', ['body_ground_seen', 'rally_fault', 'episode_index',
                                  'ball_side_complete_except_recovery', 'numerical_failures'])
def test_missing_terminal_evidence_fails(field):
    row = terminal_record()
    del row[field]
    with pytest.raises(ValueError, match='Missing terminal evidence'):
        summarize_episode(row)


def test_preserves_actual_contact_evidence():
    data = terminal_record()
    data.update(head_contact=True, independent_nonfoot_contact=True, body_ground_seen=True)
    row = summarize_episode(data)
    assert row['head_contact'] and row['body_ground_seen'] and row['independent_nonfoot_contact']


def test_resume_requires_same_physical_and_scoring_contract():
    contract = dict(environment_spacing_m=0., ground_size_m=[64.,64.,.1],
                    contact_flight_recovery='contact-flight-recovery-v1', source_policy_sha256='source')
    saved = dict(metadata=dict(task_contract=copy.deepcopy(contract)))
    require_resume_contract(saved, contract)
    for key, value in [('environment_spacing_m',16.), ('contact_flight_recovery',None),
                       ('ground_size_m',[4096.,4096.,.1]), ('source_policy_sha256','other')]:
        changed = {**contract, key:value}
        with pytest.raises(ValueError, match='Resume task contract'):
            require_resume_contract(saved, changed)


def test_legacy_replay_cannot_silently_resume():
    with pytest.raises(ValueError, match='Resume task contract'):
        require_resume_contract(dict(metadata={}), dict(version='new'))


@pytest.mark.parametrize('indices', [list(range(128)), [71, 0, 127], [], [5]])
def test_historical_reset_noise_supports_actual_batched_indexing(indices):
    import numpy as np
    import torch
    noise = zero_panel_reset_noise(128, 882072)
    shapes = {'joint_position_rad':(18,), 'joint_velocity_rad_s':(18,),
              'root_xy_m':(2,), 'root_yaw_rad':()}
    for name, shape in shapes.items():
        # The exact operation that failed at FrozenPanelEnv._reset_idx.
        selected = torch.tensor(noise[name][indices], dtype=torch.float32)
        assert isinstance(noise[name], np.ndarray)
        assert selected.shape == (len(indices), *shape)
        assert torch.isfinite(selected).all() and not selected.any()


def test_reset_noise_partial_modification_does_not_change_other_lanes():
    import numpy as np
    noise = zero_panel_reset_noise(128, 882072)
    noise['joint_position_rad'][[71,127], 0] = [0.1,0.2]
    assert np.count_nonzero(noise['joint_position_rad']) == 2
    assert not noise['joint_position_rad'][[0,70,72]].any()
