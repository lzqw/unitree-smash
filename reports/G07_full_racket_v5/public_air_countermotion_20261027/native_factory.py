"""Construct the checked prototype over actual source checkpoint families."""
import hashlib
import importlib.util
import json
from pathlib import Path

from quadsmash.control.full_racket_origin0_residual_v4 import make_rally_controller

HERE = Path(__file__).resolve().parent
SCHEMA = 'g07-public-air-countermotion-v5'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def candidates():
    return [dict(candidate_id=i, name=name, maximum_velocity_change_rad_s=cap)
            for i,(name,cap) in enumerate([('unchanged_source',0.),('post_hit_legs_cap2',2.),('post_hit_legs_cap5',5.)])]


def make_source(source,count,device):
    if not source.get('native_execution'):
        raise ValueError('Require an actual measured source')
    if source['schema'] == 'g07-shared-native-recovery-prior-v5':
        path = HERE.parent/'recovery_prior_native_20261027/prior_controller.py'
        if hashlib.sha256(path.read_bytes()).hexdigest() != source['factory_sha256']:
            raise ValueError('Measured recovery-prior factory changed')
        factory = load('air_preserved_actual_recovery_prior',path)
        return factory.RecoveryPriorController(source['source_public_policy'],source['seed_metadata'],
            source['seed_arrays'],count,device,**source['recovery_prior'])
    return make_rally_controller(source,count,device)


def make_controller(source,count,device,maximum_velocity_change_rad_s):
    cpu = json.loads((HERE/'CPU_CONTROLLER_CHECK.json').read_text())
    path = HERE/'countermotion.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != cpu['source_sha256'][str(path)]:
        raise ValueError('Checked countermotion core changed')
    core = load('checked_air_countermotion_core',path)
    return core.AirCountermotionController(make_source(source,count,device),source['asset_sha256'],count,device,
        maximum_velocity_change_rad_s)
