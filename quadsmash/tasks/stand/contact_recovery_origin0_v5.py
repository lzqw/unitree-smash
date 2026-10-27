"""Explicit scoring revision on the already admitted, unchanged origin0 physics."""
import copy
import hashlib
import json
from pathlib import Path

from .contact_flight_recovery_v5 import SCHEMA
from .full_racket_origin0_v4 import full_racket_origin0_cfg


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = 'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'
ORIGINAL = 'configs/g07_full_racket_eval_v4_r3_origin0.json'


def validate_recovery_protocol(protocol):
    original_path = ROOT / ORIGINAL
    original = json.loads(original_path.read_text())
    association = protocol.get('recovery_association')
    expected = dict(schema=SCHEMA, source_protocol_sha256=hashlib.sha256(original_path.read_bytes()).hexdigest(),
                    preserves_physics_cases_target_and_thresholds=True,
                    contact_flight_qualification_and_post_landing_continuous_recovery=True)
    if association != expected or protocol['version'] != 'full-racket-singles-rally-v5-contact-flight1-origin0':
        raise ValueError('Require the explicit contact-flight recovery revision')
    restored = copy.deepcopy(protocol)
    restored.pop('_root', None)
    restored.pop('recovery_association')
    if restored.pop('before_v5_policy_training') is not True:
        raise ValueError('Require the versioned protocol established before new training')
    restored['version'] = original['version']
    restored['frozen_at'] = original['frozen_at']
    if restored != original:
        raise ValueError('Recovery association must preserve all other original protocol fields')
    return original


def contact_recovery_origin0_cfg(count, protocol, receipt):
    return full_racket_origin0_cfg(count, validate_recovery_protocol(protocol), receipt)
