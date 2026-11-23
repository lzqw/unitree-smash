import copy
import pytest
from quadsmash.rl.supermimic_task_contract import require_resume_contract, summarize_episode
from quadsmash.rl.supermimic_task_contract import zero_panel_reset_noise
from quadsmash.rl.supermimic_task_contract import SOURCE_SWEEP_AUDIT, validate_sweep_contract


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


def test_task_cfg_spawns_the_registered_asset_from_another_worktree(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from quadsmash.rl.supermimic_task_contract import make_task_cfg, sha

    asset = tmp_path / 'assets' / 'registered.usd'
    asset.parent.mkdir()
    asset.write_text('registered robot')
    cfg = SimpleNamespace(
        robot=SimpleNamespace(spawn=SimpleNamespace(usd_path='/isolated-worktree/missing.usd')),
        scene=SimpleNamespace(env_spacing=0.), sim=SimpleNamespace(dt=.0000625),
        decimation=160, sweep_audit_iterations=4096,
    )
    monkeypatch.setitem(sys.modules, 'quadsmash.tasks.stand.contact_recovery_origin0_v5',
                        SimpleNamespace(contact_recovery_origin0_cfg=lambda *_: cfg))
    contract = dict(contact_flight_recovery='contact-flight-recovery-v1',
                    asset=dict(path='assets/registered.usd', sha256=sha(asset)),
                    ground_size_m=[64., 64., .1], episode_s=8., environment_spacing_m=0.,
                    control_dt_s=.01, sweep_audit=dict(SOURCE_SWEEP_AUDIT))
    result = make_task_cfg(128, contract, {}, 'receipt.json', execution_root=tmp_path)
    assert result.robot.spawn.usd_path == str(asset)
    asset.write_text('changed robot')
    with pytest.raises(ValueError, match='Configured robot asset differs'):
        make_task_cfg(128, contract, {}, 'receipt.json', execution_root=tmp_path)


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


@pytest.mark.parametrize('requested', [None, 4096])
def test_sweep_contract_inherits_without_mutating_source(requested):
    from types import SimpleNamespace
    cfg = SimpleNamespace(sweep_audit_iterations=4096)
    contract = {'sweep_audit':dict(SOURCE_SWEEP_AUDIT)}
    assert validate_sweep_contract(cfg, contract, requested) == 4096
    assert vars(cfg) == {'sweep_audit_iterations':4096}


@pytest.mark.parametrize('requested', [0, -1, 64, 256, 4097])
def test_sweep_override_cannot_change_formal_audit(requested):
    from types import SimpleNamespace
    with pytest.raises(ValueError, match='Sweep override'):
        validate_sweep_contract(SimpleNamespace(sweep_audit_iterations=4096),
                                {'sweep_audit':dict(SOURCE_SWEEP_AUDIT)}, requested)


def test_effective_zero_sweep_setting_is_rejected_before_env_construction():
    from types import SimpleNamespace
    with pytest.raises(ValueError, match='Effective sweep iterations 0'):
        validate_sweep_contract(SimpleNamespace(sweep_audit_iterations=0),
                                {'sweep_audit':dict(SOURCE_SWEEP_AUDIT)})
    with pytest.raises(ValueError, match='source sweep audit contract'):
        validate_sweep_contract(SimpleNamespace(sweep_audit_iterations=4096), {})


def test_sweep_contract_matches_source_factory_literal_and_zero_is_not_disabled():
    import ast
    from pathlib import Path
    import torch
    from quadsmash.physics.full_racket_v4 import swept_string_prism, load_racket
    root = Path(__file__).resolve().parents[1]
    factory = ast.parse((root/'quadsmash/tasks/stand/full_racket_env_v4.py').read_text())
    fn = next(n for n in factory.body if isinstance(n,ast.FunctionDef) and n.name=='full_racket_cfg')
    values = [ast.literal_eval(n.value) for n in ast.walk(fn) if isinstance(n,ast.Assign)
              for target in n.targets if isinstance(target,ast.Attribute) and target.attr=='sweep_audit_iterations']
    assert values == [SOURCE_SWEEP_AUDIT['maximum_iterations']]
    start=torch.tensor([[-.2,0.,0.]])
    center=torch.zeros(1,3); quat=torch.tensor([[1.,0.,0.,0.]])
    hit,_,unresolved = swept_string_prism(start,-start,center,center,quat,quat,load_racket(),.025,iterations=0)
    assert not hit.item() and unresolved.item()
    hit,_,unresolved = swept_string_prism(start,-start,center,center,quat,quat,load_racket(),.025,iterations=4096)
    assert hit.item() and not unresolved.item()
