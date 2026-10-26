"""Explicit numerical origin variant; physical task and scoring remain exact."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
PROTOCOL='configs/g07_full_racket_eval_v4_r3_origin0.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_protocol(protocol):
    base=json.loads((ROOT/'configs/g07_full_racket_eval_v4_r3.json').read_text())
    original={key:value for key,value in protocol.items() if key not in ('_root','numerical_execution')}
    if original!=base:
        raise ValueError('Origin variant must preserve every original physical, case and scoring field')
    numerical=protocol.get('numerical_execution',{})
    if (numerical.get('variant')!='r3-origin0' or numerical.get('environment_spacing_m')!=0.
            or numerical.get('requires_native_precision_and_collision_isolation_admission') is not True
            or numerical.get('source_performance_inherited') is not False):
        raise ValueError('Require the explicit origin0 numerical contract')


def validate_admission(receipt):
    receipt=Path(receipt)
    data=json.loads(receipt.read_text())
    if data.get('admitted') is not True or data.get('new_native_execution') is not True:
        raise ValueError('Require an actually completed native precision/isolation admission')
    for name,digest in data['source_sha256'].items():
        if sha(name)!=digest:raise ValueError('Native precision evidence changed: '+name)
    compact=data['compact']
    if not (compact['runtime_origin_spacing_m']==0. and compact['compact_precision_gate_passed']
            and compact['collision_isolation_diagnostic_passed'] and compact['numerical_failures']==0
            and compact['max_free_flight_position_error_m']<=.001
            and compact['max_replica_ball_position_spread_m']<=.0001
            and compact['max_replica_root_position_spread_m']<=.001
            and compact['max_shuttle_contact_force_n']<=.001 and compact['unexpected_done_count']==0):
        raise ValueError('Native precision or isolation gate failed')
    if data['baseline']['runtime_origin_spacing_m']!=16.:
        raise ValueError('Missing original16m native comparison')
    for name in ['baseline_run','compact_run']:
        run=Path(data[name]);job=json.loads((run/'job.json').read_text())
        if job['status']!='SUCCEEDED' or job['charged_env_steps']!=6400:
            raise ValueError('Incomplete native engineering probe')
        if json.loads((run/'completed.json').read_text())['env_steps']!=6400:
            raise ValueError('Native engineering probe counters differ')
    return data


def full_racket_origin0_cfg(count,protocol,receipt):
    validate_protocol(protocol)
    validate_admission(receipt)
    from .full_racket_env_v4 import full_racket_cfg
    cfg=full_racket_cfg(count,protocol['physics_dt_s'])
    cfg.scene.env_spacing=0.
    if not cfg.scene.filter_collisions or not cfg.scene.replicate_physics:
        raise ValueError('Require native environment-ID collision isolation')
    return cfg
