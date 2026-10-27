"""Bind the checked public strike controller to an actually executed source."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
SCHEMA='g07-public-momentum-strike-v5'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_source(source,count,device):
    if not source.get('native_execution'):
        raise ValueError('Require an actually executed source')
    if source['schema']=='g07-public-air-countermotion-v5':
        path=HERE.parent/'public_air_countermotion_20261027/native_factory.py'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=source['factory_sha256']:
            raise ValueError('Measured source countermotion factory changed')
        factory=load('strike_preserved_actual_air_countermotion',path)
        return factory.make_controller(source['source_public_policy'],count,device,
            source['air_countermotion']['maximum_velocity_change_rad_s'])
    # Actual existing swing/lateral families are supported for the CPU
    # factory checks. The queued native source has a stricter admission.
    path=HERE.parent/'windowed_swing_20261027/window_controller.py'
    return load('strike_preserved_existing_source_dispatch',path).make_source(source,count,device)


def checked_core():
    check=json.loads((HERE/'CPU_CONTROLLER_CHECK.json').read_text())
    path=HERE/'strike_controller.py'
    if not check['both_baseline_replays_within_original_tolerance']:
        raise ValueError('Original source command replay tolerance failed')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=check['source_sha256'][str(path)]:
        raise ValueError('CPU-checked strike core changed')
    return load('cpu_checked_momentum_strike_core',path)


def candidates():
    return checked_core().candidates()


def make_controller(source,count,device,*,gain,leg_velocity_delta_cap_rad_s):
    return checked_core().MomentumStrikeController(make_source(source,count,device),
        source['asset_sha256'],count,device,gain,leg_velocity_delta_cap_rad_s)
