"""Deterministic native candidate pairs on a validation panel disjoint from training and final tests."""
import hashlib
import json

import torch

from quadsmash.rl.supermimic_independent_panel import fixed128_cases, load_panel
from quadsmash.rl.supermimic_physical_loop_loading_v1 import RECOVERY_NAMES
from quadsmash.rl.supermimic_policy_execution_v1 import resolve_contract
from scripts.run_supermimic_recovery_baseline import verify


def panel_layout(validation, training, protocol_sha, seed):
    rows = json.loads(validation.read_text())['cases']
    chosen = [c for panel in ('A','B') for c in [r for r in rows if r['panel'] == panel][:32]]
    forbidden = (json.loads(training.read_text())['cases'] + load_panel()['cases'] +
                 fixed128_cases(protocol_sha, seed)[0])
    def launches(cases):
        return {json.dumps(row['launch'], sort_keys=True) for row in cases}
    if (len(chosen) != 64 or len(launches(chosen)) != 64 or len({r['case_id'] for r in chosen}) != 64
            or launches(chosen) & launches(forbidden)):
        raise ValueError('Validation requires 64 distinct launches disjoint from training and final panels')
    return chosen * 2


def policies(primary_path, alternate_path, companion_path, load):
    verify([primary_path, alternate_path, companion_path], synchronize_recovery=True)
    primary, alternate, companion = [
        torch.load(p, weights_only=False, mmap=True, map_location='cpu') for p in
        (primary_path, alternate_path, companion_path)]
    if primary['actor'].keys() != alternate['actor'].keys() or any(
            not torch.equal(x, alternate['actor'][k]) for k,x in primary['actor'].items() if k not in RECOVERY_NAMES):
        raise ValueError('Recovery candidates differ outside the six permitted tensors')
    bank = []
    for candidate in (primary, alternate):
        for use_companion in (False, True):
            actor = load()
            actor.load_state_dict(candidate['actor'], strict=True)
            if use_companion:
                actor.actor.approach.load_state_dict({k.removeprefix('actor.approach.'):v
                    for k,v in companion['actor'].items() if k.startswith('actor.approach.')})
            actor.configure_approach_gate(resolve_contract(candidate)[0].get('arm_preparation_seconds'))
            actor.eval()
            bank.append(actor)
    return bank


def paired_requests(bank, obs, gate, release):
    out = obs.new_empty((128,18))
    for idx, policy in enumerate(bank):
        first = 64*(idx//2)
        lanes = slice(first+idx%2, first+64, 2)
        out[lanes] = policy._gated_distribution(obs[lanes], gate[lanes], release_time=release[lanes]).loc.clamp(-1,1)
    return out


def report(rows, checkpoint, alternate, companion, initial_states):
    if len(rows) != 128 or [r['case_id'] for r in rows[:64]] != [r['case_id'] for r in rows[64:]]:
        raise ValueError('Incomplete native pair')
    # The native constructor and wrapper reset can each emit a row. Inspect the
    # last initial condition per lane, then bind that panel across native runs.
    initial = {r['env_id']:{k:v for k,v in r.items() if k!='env_id'} for r in initial_states}
    if set(initial) != set(range(128)) or any(initial[i] != initial[64+i] for i in range(64)):
        raise ValueError('Candidate lanes have different native reset states')
    identity = hashlib.sha256(json.dumps([initial[i] for i in range(64)],sort_keys=True).encode()).hexdigest()
    metrics = []
    for group in (rows[:64], rows[64:]):
        if any(r['numerical_failures'] for r in group):
            raise ValueError('Invalid native evaluation')
        contact = [r for r in group if r['qualified_airborne_string_contact']]
        metrics.append(dict(qualified=len(contact), body=sum(r['body_ground_seen'] for r in contact),
            strict=sum(r['strict_physical_recovery'] is True for r in contact),
            relaxed=sum(r['physical_recovery_success'] is True for r in contact),
            net=sum(r['robot_net_fault'] for r in group), ball=sum(r['ball_side_complete_except_recovery'] for r in group)))
    return dict(schema='dynamimic-paired-native64-v1', checkpoints=[hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (checkpoint,alternate)], companion_sha256=hashlib.sha256(companion.read_bytes()).hexdigest(),
        metrics=metrics, reset_sha256=identity, case_ids=[r['case_id'] for r in rows[:64]], companion_lanes=list(range(1,64,2)))
