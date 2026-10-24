"""Prepare a TRAIN grid of the existing Cartesian feedback coordinate.

This only writes policy inputs. It does not change the controller, physical
model, hardware limits, task labels, or frozen evaluation cases.
"""
import copy
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from quadsmash.control.full_racket_candidate_bank_v4 import validate_candidate_bank
from quadsmash.control.full_racket_contact_normal_search_v4 import NORMAL_LOW, NORMAL_HIGH
from quadsmash.control.full_racket_parameter_layout_v4 import controller_parameter_names
from quadsmash.control.full_racket_recovery_search_v4 import RECOVERY_LOW, RECOVERY_HIGH
from quadsmash.control.full_racket_teacher_v4 import LOW, HIGH

OUT = Path(__file__).parent
assert not (OUT / 'policy.pt').exists()
parent = ROOT / 'reports/G07_full_racket_v4/slew50_near_attack_gen0_c54_1540/policy_evidence.pt'
assert hashlib.sha256(parent.read_bytes()).hexdigest() == '634da4f6fb65298ccf5afb50e7807324efc0bab3add5abe0b0064c26aeaa0a68'
source = torch.load(parent, map_location='cpu', weights_only=True)
assert source['native_execution'] and source['seed']['arm_target_rate_limit_rad_s'] == 5.
names = controller_parameter_names(search_arm_velocity=True, search_recovery_parameters=True,
                                   search_contact_normal=True)
assert source['parameter_names'] == names
reference = source['parameters'].numpy().copy()
lower = np.array(LOW + [.8, -.12] + [-5.] * 6 + RECOVERY_LOW + NORMAL_LOW, dtype=np.float32)
upper = np.array(HIGH + [1.45, .12] + [5.] * 6 + RECOVERY_HIGH + NORMAL_HIGH, dtype=np.float32)
lower[3] = 1.20
lower[22:24], upper[22:24] = -.30, .30
rows, labels = [], []
for servo in [.35, .55, .75, 1.]:
    for lead in [0., .01, .02, .03, .04]:
        for normal in [.04, .08, .12]:
            row = reference.copy()
            row[[14, 18, 37]] = [lead, servo, normal]
            rows.append(row)
            labels.append(dict(cartesian_gain=servo, arm_lead_s=lead,
                               normal_downward_bias_rad=normal, performance_inherited=False))
rows = np.array(rows, dtype=np.float32)
assert rows.shape == (60, 38) and len(set(map(tuple, rows))) == 60
fixed = [i for i in range(38) if i not in [14, 18, 37]]
np.testing.assert_array_equal(rows[:, fixed], np.tile(reference[fixed], (60, 1)))
assert (abs(rows[:, 26:32] * rows[:, 7:8]) <= 5).all()
warm = copy.deepcopy(source)
warm.pop('training_result', None)
warm.update(native_execution=False, performance_inherited=False, formal_acceptance=False,
            kind='Unexecuted existing Cartesian-gain/timing/normal grid; source outcomes do not transfer.')
bank = dict(schema='full-racket-candidate-bank-v4', asset_sha256=warm['asset_sha256'],
            seed=warm['seed'], parameter_names=names, reference_parameters=reference.tolist(),
            parameters=rows.tolist(), proposal_labels=labels,
            controller_revision='existing-cartesian-feedback38-grid-v1',
            native_execution=False, performance_inherited=False)
validate_candidate_bank(bank, warm['seed'], warm['asset_sha256'], names, reference, lower, upper, 62)
torch.save(warm, OUT / 'policy.pt')
(OUT / 'CANDIDATE_BANK.json').write_text(json.dumps(bank, indent=2) + '\n')
sources = [parent, Path(__file__), ROOT / 'quadsmash/control/full_racket_teacher_v4.py',
           ROOT / 'quadsmash/control/full_racket_candidate_bank_v4.py']
report = dict(created_at=datetime.now(timezone.utc).isoformat(),
              scope='CPU-prepared input only, no new native performance. Tests the already implemented public-state Cartesian feedback gain, with the same bounded software slew and all physical/task conditions.',
              candidates=60, parameter_width=38, changed_coordinates=[14, 18, 37],
              cartesian_gains=[.35, .55, .75, 1.], arm_leads_s=[0., .01, .02, .03, .04],
              normal_biases_rad=[.04, .08, .12], software_arm_target_rate_rad_s=5.,
              all_other_parameters_fixed=True, native_execution=False, performance_inherited=False,
              formal_acceptance=False,
              source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              policy_sha256=hashlib.sha256((OUT / 'policy.pt').read_bytes()).hexdigest(),
              bank_sha256=hashlib.sha256((OUT / 'CANDIDATE_BANK.json').read_bytes()).hexdigest())
(OUT / 'REPORT.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
