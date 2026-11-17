#!/usr/bin/env bash
set -Eeuo pipefail
# Code-only delivery: this launcher is not invoked by the CPU validation suite.
# A supplied contact receipt is a prerequisite, not generated or marked passed here.
export SM_TRAINING_PROFILE=large
export SM_CYCLES="${SM_CYCLES:-4}"
export SM_RECOVERY_UPDATES="${SM_RECOVERY_UPDATES:-64}"
export SM_APPROACH_UPDATES="${SM_APPROACH_UPDATES:-64}"
export SM_RECOVERY_PROPOSALS="${SM_RECOVERY_PROPOSALS:-256}"
export SM_APPROACH_PROPOSALS="${SM_APPROACH_PROPOSALS:-256}"
: "${SM_CONTACT_VALIDATION_RECEIPT:?supply checkpoint/contract-matched contact validation evidence}"
: "${SM_TASK_REFERENCE_BANK:?supply independently scored v2 reference bank}"
: "${SM_TRACKING_WEIGHT:?declare calibrated tracking weight}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/scripts/run_supermimic_phase1_long_v1.sh"
