"""Bind the checked stance proposal to actual source controller families."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
SCHEMA='g07-public-stance-recovery-v5'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def make_source(source,count,device):
    if not source.get('native_execution'):
        raise ValueError('Require an actually executed source; caller must also admit its real cold evaluation')
    if source['schema']=='g07-public-momentum-strike-v5':
        directory=HERE.parent/'free_floating_strike_20261027'
        path=directory/'native_factory.py'
        if source['factory_sha256']!=sha(path) or source['strike_core_sha256']!=sha(directory/'strike_controller.py'):
            raise ValueError('Actual source strike controller changed')
        if source['free_floating_core_sha256']!=sha(directory/'free_floating.py'):
            raise ValueError('Actual source momentum model changed')
        model=HERE.parent/'centroidal_recovery_model_20261027/MODEL.json'
        if source['known_model_sha256']!=sha(model) or source['known_model']!=json.loads(model.read_text()):
            raise ValueError('Actual source physical model changed')
        factory=load('stance_preserved_measured_momentum_strike',path)
        return factory.make_controller(source['source_public_policy'],count,device,**source['momentum_strike'])
    path=HERE.parent/'windowed_swing_20261027/window_controller.py'
    return load('stance_preserved_existing_source_dispatch',path).make_source(source,count,device)


def checked_core():
    check=json.loads((HERE/'CPU_CONTROLLER_CHECK.json').read_text())
    path=HERE/'stance_controller.py'
    if sha(path)!=check['source_sha256'][str(path)] or len(check['results'])!=12:
        raise ValueError('Require unchanged checked stance core and all12 actual-source replays')
    return load('checked_public_stance_core',path)


def candidates():return checked_core().candidates()


def make_controller(source,count,device,*,gain,horizontal_damping):
    return checked_core().StanceRecoveryController(make_source(source,count,device),source['asset_sha256'],
        count,device,gain,horizontal_damping)
