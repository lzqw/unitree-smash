"""Full-remainder shooting over a measured-support feedback controller."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import search_foot_anchored_recovery_v1 as driver
from scripts.feedback_foot_recovery_reference_v3 import (
    FeedbackFootRecoveryReferenceV3,FeedbackFootRecoveryPolicyV3,
    PARAMETER_NAMES,LOWER,UPPER,NOMINAL,
)

driver.FootAnchoredRecoveryReferenceV1=FeedbackFootRecoveryReferenceV3
driver.FootAnchoredRecoveryFeedbackV1=FeedbackFootRecoveryPolicyV3
driver.PARAMETER_NAMES=PARAMETER_NAMES
driver.LOWER=LOWER;driver.UPPER=UPPER;driver.NOMINAL=NOMINAL

if __name__=='__main__':
    import scripts.search_feedback_foot_recovery_v3
    driver.main()
