"""Archive integrity only; synthetic fixtures do not verify native physics."""
import hashlib
import json
from types import SimpleNamespace
import types
from pathlib import Path

import pytest
import torch

from quadsmash.rl.supermimic_physical_capture_v1 import (
    PhysicalTransitionArchive, capture_controller)


def state(tick, episode=0):
    return dict(control_tick=tick, episode_index=torch.tensor([episode, episode]),
                physical={'joint_position': torch.full((2, 18), float(tick))})


def test_terminal_state_and_actual_command_survive_reset_and_chunk_save(tmp_path):
    archive = PhysicalTransitionArchive(tmp_path/'data', metadata={}, chunk_ticks=2)
    requested = torch.full((2, 18), .8)
    final = torch.full((2, 18), .3)
    targets = torch.full((2, 18), .15)
    before, after = state(799), state(800)
    memory = {'tensors': {'clock': torch.tensor([7.99, 7.99])}}
    archive.begin_record(before, requested, torch.zeros(2, 794), memory)
    archive.finish_record(after, final, targets, torch.tensor([1., 2.]),
                          torch.tensor([False, True]), torch.tensor([True, False]), memory)
    # Native buffers are reset/mutated immediately after reward capture.
    for x in (requested, final, targets, after['physical']['joint_position'],
              memory['tensors']['clock']):
        x.zero_()
    after['episode_index'].add_(1)
    archive.flush()
    manifest = json.loads((tmp_path/'data/manifest.json').read_text())
    path = tmp_path/'data'/manifest['chunks'][0]['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['chunks'][0]['sha256']
    row = torch.load(path, weights_only=True)['rows'][0]
    assert row['after']['physical']['joint_position'].eq(800).all()
    assert row['after']['episode_index'].eq(0).all()
    assert row['requested_action'].eq(.8).all()
    assert row['final_action'].eq(.3).all()
    assert row['final_joint_targets'].eq(.15).all()
    assert row['timeout'].tolist() == [True, False]
    assert row['terminated'].tolist() == [False, True]
    assert manifest['transitions'] == 2 and not manifest['pending_transition']


@pytest.mark.parametrize('after,match', [(state(2), 'Nonconsecutive'),
                                       (state(1, 1), 'crossed reset')])
def test_reject_nonconsecutive_or_post_reset_targets(tmp_path, after, match):
    archive = PhysicalTransitionArchive(tmp_path/'data', metadata={})
    archive.begin_record(state(0), torch.zeros(2, 18), torch.zeros(2, 794), {})
    with pytest.raises(RuntimeError, match=match):
        archive.finish_record(after, torch.zeros(2, 18), torch.zeros(2, 18),
                              torch.zeros(2), torch.zeros(2), torch.zeros(2), {})
    assert archive.transitions == 0


def test_controller_memory_isolated_with_aliases_and_omissions():
    child = SimpleNamespace(previous=torch.tensor([.2]), clock=17)
    controller = SimpleNamespace(first=child, second=child,
                                 actor=torch.nn.Linear(2, 2), logs=[{'debug': 'skip'}],
                                 unknown=object())
    controller.loop = controller
    captured = capture_controller(controller)
    child.previous.zero_()
    assert captured['tensors']['controller/first/previous'].item() == pytest.approx(.2)
    assert captured['values']['controller/first/clock'] == 17
    assert captured['aliases']['controller/second'] == 'controller/first'
    assert captured['aliases']['controller/loop'] == 'controller'
    assert captured['omitted']['controller/actor'] == 'network:Linear'
    assert captured['omitted']['controller/unknown'] == 'builtins.object'


def test_refuse_overwrite_and_missing_begin_or_end(tmp_path):
    archive = PhysicalTransitionArchive(tmp_path/'data', metadata={})
    with pytest.raises(FileExistsError):
        PhysicalTransitionArchive(tmp_path/'data', metadata={})
    with pytest.raises(RuntimeError, match='No physical transition begin'):
        archive.finish_record(state(1), None, None, None, None, None, None)
    archive.begin_record(state(0), torch.zeros(2, 18), torch.zeros(2, 794), {})
    with pytest.raises(RuntimeError, match='no pre-reset end'):
        archive.begin_record(state(0), torch.zeros(2, 18), torch.zeros(2, 794), {})


def test_no_dynamic_attributes_or_library_module_graph_is_visited():
    class ExplosiveProxy:
        def __getattribute__(self, name):
            raise ImportError('optional dependency must never be queried')
    module = types.ModuleType('optional_library')
    module.placeholder = ExplosiveProxy()
    root = SimpleNamespace(module=module, proxy=ExplosiveProxy(),
                           callback=lambda: None, state=torch.tensor([1.]))
    snapshot = capture_controller(root)
    assert snapshot['omitted']['controller/module'] == 'code:builtins.module'
    assert snapshot['omitted']['controller/proxy'].endswith('.ExplosiveProxy')
    assert len(snapshot['tensors']) == 1


SOURCE = Path('/home/autodrive/LZQW/unitree_smash/reports/G07_full_racket_v5/'
              'frozen_v2_smooth_approach16_env5120_01/policy.pt')


@pytest.mark.skipif(not SOURCE.exists(), reason='Original controller checkpoint not mounted')
def test_real_controller_memory_capture_before_and_after_cpu_action():
    from quadsmash.control.g07_shared_target_residual_v1 import make_g07_residual_controller
    controller = make_g07_residual_controller(
        torch.load(SOURCE, map_location='cpu', weights_only=False), 2, 'cpu')
    before = capture_controller(controller)
    controller.action({'policy': torch.zeros(2, 794)}, torch.zeros(18),
                      torch.stack((-torch.ones(18), torch.ones(18)), -1))
    after = capture_controller(controller)
    # G07 consumes these logs through plan_cursor; they are causal state.
    controller.reference_controller.logs.append({'env_ids': [0], 'valid': [True]})
    with_plan = capture_controller(controller)
    paths = [path for path, kind in with_plan['object_types'].items()
             if kind.endswith('.FullRacketRallyControllerV4')]
    assert len(paths) == 1
    assert paths[0] + '/logs' not in with_plan['omitted']
    assert with_plan['values'][paths[0] + '/logs/0/valid/0'] is True
    assert 'controller/last_base_action' in after['tensors']
    assert 'controller/mixer/arm_rate/target' in after['tensors']
    assert 'controller/mixer/nonzero_seen' in after['tensors']
    # Cross-check every construction-time tensor from the previous independent
    # inventory, not just the outer mixer. New lazy states are also retained.
    inventory = json.loads((Path(__file__).resolve().parents[1]/
        'reports/supermimic_execution_preview_cpu_20261114/controller_memory_inventory.json').read_text())
    for entry in inventory['attributes']:
        if entry.get('kind') == 'tensor':
            assert entry['path'].replace('.', '/') in before['tensors']
    for snapshot in (before, after):
        assert len(snapshot['object_types']) >= 20
        assert all(reason.startswith(('network:', 'code:')) or reason == 'diagnostic logs'
                   for reason in snapshot['omitted'].values()), snapshot['omitted']


def test_compact_causal_plan_roundtrip_and_mutable_native_buffers():
    import io
    from quadsmash.rl.supermimic_physical_capture_v1 import restore_archive_execution
    from quadsmash.rl.supermimic_physical_capture_v1 import tensor_tree_copy

    from quadsmash.control.full_racket_rally_controller_v4 import FullRacketRallyControllerV4
    controller = object.__new__(FullRacketRallyControllerV4)
    original = torch.linspace(-1.01, 1.09, 2 * 794, dtype=torch.float64).reshape(2, 794).tolist()
    controller.logs = [dict(used_public_observations=original, valid=[True, False])]
    template = object.__new__(FullRacketRallyControllerV4)
    template.logs = []
    baseline = capture_controller(controller)
    cache = {}
    first = capture_controller(controller, plan_observation_cache=cache)
    second = capture_controller(controller, plan_observation_cache=cache)
    key = 'controller/logs/0/used_public_observations'
    field = 'public_plan_observations_f64le_v1'
    assert first[field][key] is second[field][key]
    stream = io.BytesIO()
    torch.save(tensor_tree_copy([first, second], cpu=True), stream)
    stream.seek(0)
    after_load = torch.load(stream, weights_only=True)[1]
    restored = restore_archive_execution(template, after_load)
    assert capture_controller(restored) == baseline
    restored.logs[0]['used_public_observations'][0][0] = 123.
    original[0][0] = 342.
    changed = capture_controller(controller, plan_observation_cache=cache)
    assert changed[field][key] != first[field][key]
    assert capture_controller(restore_archive_execution(template, first)) == baseline
    controller.logs.clear()
    capture_controller(controller, plan_observation_cache=cache)
    assert not cache
