"""Full-remainder shooting with explicit preservation/decay of actuator preload."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import search_foot_anchored_recovery_v1 as driver
from scripts.loaded_foot_recovery_reference_v2 import (
    LoadedFootRecoveryReferenceV2,LoadedFootRecoveryFeedbackV2,
    PARAMETER_NAMES,LOWER,UPPER,NOMINAL,
)

# Imported also by spawned workers, ensuring the same declared policy family.
driver.FootAnchoredRecoveryReferenceV1=LoadedFootRecoveryReferenceV2
driver.FootAnchoredRecoveryFeedbackV1=LoadedFootRecoveryFeedbackV2
driver.PARAMETER_NAMES=PARAMETER_NAMES
driver.LOWER=LOWER;driver.UPPER=UPPER;driver.NOMINAL=NOMINAL

if __name__=='__main__':
    # Register this executable before source pinning, including worker setup.
    import scripts.search_loaded_foot_recovery_v2
    driver.main()
