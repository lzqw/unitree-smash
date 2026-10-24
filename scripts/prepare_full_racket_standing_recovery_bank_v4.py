#!/usr/bin/env python3
"""Prepare existing standing-controller gains after a measured attacking stroke."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quadsmash.control.full_racket_candidate_bank_v4 import validate_candidate_bank
from quadsmash.control.full_racket_teacher_v4 import LOW, HIGH
from quadsmash.control.full_racket_recovery_search_v4 import (
    extend_recovery_parameters, RECOVERY_PARAMETER_NAMES, RECOVERY_LOW, RECOVERY_HIGH)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['attack-checkpoint', 'recovered-checkpoint', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh output directory')
    attack = torch.load(args.attack_checkpoint, map_location='cpu', weights_only=True)
    recovered = torch.load(args.recovered_checkpoint, map_location='cpu', weights_only=True)
    assert len(attack['parameters']) == len(recovered['parameters']) == 32
    assert attack['asset_sha256'] == recovered['asset_sha256'] and attack['seed'] == recovered['seed']
    legacy = attack['backbone']['recovery_parameters']
    assert torch.equal(legacy, recovered['backbone']['recovery_parameters'])
    attacking_cases = [r for r in attack['training_result'] if r['attack_outgoing'] and r['crossed_net']]
    recovering_cases = [r for r in recovered['training_result'] if r['airborne_string_hit'] and r['recovery_success']]
    assert attacking_cases and recovering_cases
    reference = extend_recovery_parameters(attack['parameters'], legacy).numpy()
    weak_reference = extend_recovery_parameters(recovered['parameters'], legacy).numpy()
    names = attack['parameter_names'] + RECOVERY_PARAMETER_NAMES
    lower = np.array(LOW + [.8, -.12] + [-5.]*6 + RECOVERY_LOW, dtype=np.float32)
    upper = np.array(HIGH + [1.45, .12] + [5.]*6 + RECOVERY_HIGH, dtype=np.float32)
    lower[22:24], upper[22:24], lower[3] = -.3, .3, 1.30
    proposals, labels = [], []
    for handover in [1.425, 1.45, 1.50, 1.55]:
        for actor_gain in [.4, .65, 1.]:
            for front_back_delta in [-.06, -.03, 0., .03, .06]:
                p = reference.copy()
                p[3], p[33], p[36] = handover, float(legacy[1])+front_back_delta, actor_gain
                proposals.append(p)
                labels.append(dict(kind='attacking_motion_standing_controller_adaptation',
                    recovery_local_time_s=handover, standing_actor_gain=actor_gain,
                    standing_front_back_thigh_delta_rad=front_back_delta, performance_inherited=False))
    proposals.append(weak_reference)
    labels.append(dict(kind='actual_cem17_airborne_string_qualified_recovery_reference', performance_inherited=False))
    proposals = np.stack(proposals)
    assert proposals.shape == (61, 37) and len({tuple(row) for row in proposals}) == 61
    fixed32 = [i for i in range(32) if i != 3]
    np.testing.assert_array_equal(proposals[:60, fixed32], np.tile(reference[fixed32], (60, 1)))
    assert np.max(abs(proposals[:,26:32]*proposals[:,7:8])) <= 5.
    warm = copy.deepcopy(attack)
    warm['parameters'] = torch.tensor(reference)
    warm['parameter_names'] = names
    warm.pop('training_result', None)
    warm.update(native_execution=False, formal_acceptance=False, performance_inherited=False,
        kind='Unexecuted37-coordinate search: all32attacking reference parameters retained, five existing standing-controller parameters appended at legacy values. New proposals change existing leg-target policy, never hardware/task gates.')
    bank = dict(schema='full-racket-candidate-bank-v4', asset_sha256=warm['asset_sha256'],
        seed=warm['seed'], parameter_names=names, reference_parameters=reference.tolist(),
        parameters=proposals.tolist(), proposal_labels=labels,
        controller_revision='searched-existing-standing-recovery37-v1',
        native_execution=False, performance_inherited=False)
    validate_candidate_bank(bank, warm['seed'], warm['asset_sha256'], names, reference, lower, upper, 62)
    paths = [args.attack_checkpoint, args.recovered_checkpoint, Path(__file__),
             ROOT/'quadsmash/control/full_racket_recovery_search_v4.py',
             ROOT/'quadsmash/control/full_racket_teacher_v4.py',
             ROOT/'quadsmash/control/full_racket_arm_search_v4.py']
    provenance = {str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    warm['standing_recovery_search_provenance'] = provenance
    bank['standing_recovery_search_provenance'] = provenance
    args.output.mkdir(parents=True)
    torch.save(warm, args.output/'policy.pt')
    (args.output/'CANDIDATE_BANK.json').write_text(json.dumps(bank,indent=2)+'\n')
    report = dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='CPU-only unexecuted standing-recovery37search preparation.60proposals retain all attacking coordinates except handover;61st repeats actual weaker contact/recovery parameters. Coupled performance must be remeasured.',
        parameter_count=37, old32_reference_coordinates_preserved_exactly=True,
        appended_legacy_recovery_values=legacy.tolist(),
        new_policy_coordinate_names=RECOVERY_PARAMETER_NAMES,
        new_algorithm_lower_bounds=RECOVERY_LOW,new_algorithm_upper_bounds=RECOVERY_HIGH,
        first60_changed_coordinates=[3,33,36], first60_original32_except_handover_fixed=True,
        controlled_local_handover_times_s=[1.425,1.45,1.50,1.55],
        controlled_standing_actor_gains=[.4,.65,1.],
        controlled_standing_front_back_thigh_delta_rad=[-.06,-.03,0.,.03,.06],
        actual_airborne_recovery_reference=dict(env_id=recovering_cases[0]['env_id'],
            recovery_s=recovering_cases[0]['jump']['recovery_s'],
            outgoing_velocity_mps=recovering_cases[0]['outgoing_velocity_mps'],
            net_blocked_fault=recovering_cases[0]['net_blocked_fault'],
            complete_smash=recovering_cases[0]['complete_smash']),
        all_physical_assets_hardware_limits_and_original_success_gates_unchanged=True,
        recommended_flags=['--search-arm-velocity','--search-recovery-parameters',
                           '--retain-contact-recovery','--recovery-search-min','1.30'],
        policy_sha256=hashlib.sha256((args.output/'policy.pt').read_bytes()).hexdigest(),
        bank_sha256=hashlib.sha256((args.output/'CANDIDATE_BANK.json').read_bytes()).hexdigest(),
        native_execution=False,formal_acceptance=False,source_sha256=provenance)
    (args.output/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'}),flush=True)


if __name__ == '__main__':
    main()
