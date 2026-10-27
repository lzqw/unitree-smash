"""Construct shared reference trials over actual admitted controller families."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA = 'g07-public-shared-target-bias-v5'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_source(source, count, device):
    if not source.get('native_execution'):
        raise ValueError('Require an actual source; caller must also admit its independent128')
    schema = source['schema']
    if schema == 'g07-public-stance-recovery-v5':
        directory = HERE.parent/'public_stance_recovery_20261027'
        path = directory/'native_factory.py'
        if source['factory_sha256'] != sha(path) or source['stance_core_sha256'] != sha(directory/'stance_controller.py'):
            raise ValueError('Actual stance source factory/core changed')
        model = HERE.parent/'centroidal_recovery_model_20261027/MODEL.json'
        if source['known_model_sha256'] != sha(model) or source['known_model'] != json.loads(model.read_text()):
            raise ValueError('Actual stance source model changed')
        factory = load('target_preserved_actual_stance', path)
        return factory.make_controller(source['source_public_policy'], count, device, **source['stance_recovery'])
    dispatch = {
        'g07-public-momentum-strike-v5': 'public_stance_recovery_20261027/native_factory.py',
        'g07-public-air-countermotion-v5': 'free_floating_strike_20261027/native_factory.py',
        'g07-shared-native-recovery-prior-v5': 'public_air_countermotion_20261027/native_factory.py',
    }
    if schema in dispatch:
        factory = load('target_preserved_actual_'+schema.replace('-', '_'), HERE.parent/dispatch[schema])
        return factory.make_source(source, count, device)
    factory = load('target_preserved_swing_lateral_source', HERE.parent/'recovery_prior_native_20261027/prior_controller.py')
    return factory.make_base(source, count, device)


def checked_core():
    check = json.loads((HERE/'CPU_SHARED_TARGET_CHECK.json').read_text())
    core = HERE/'shared_target_controller.py'
    if sha(core) != check['source_sha256'][str(core)] or len(check['results']) != 10:
        raise ValueError('Require the unchanged shared-target core and all10 original CPU checks')
    for result in check['results']:
        if not result['bounded_finite'] or not result['actors_and_normalizers_unchanged']:
            raise ValueError('Original shared-target CPU invariants failed')
        if result['candidate']['candidate_id'] == 0 and result['CPU_vs_actual_GPU_max_action_error'] > .0005:
            raise ValueError('Original source replay tolerance failed')
    return load('checked_shared_target_core', core)


def candidates():
    return checked_core().candidates()


def make_controller(source, count, device, *, bias_world_m):
    return checked_core().SharedTargetController(make_source(source, count, device), count, device,
                                                bias_world_m=bias_world_m)
