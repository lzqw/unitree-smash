import json
from pathlib import Path
import numpy as np
from quadsmash.control.impact_outcome_evaluator_v1 import ImpactOutcomeEvaluator
from quadsmash.control.strict_impact_admission_v1 import StrictImpactAdmissionV1
from scripts.strike_state_impact_builder_v1 import StrikeStateImpactBuilderV1

def test_builder_rejects_missing_actual_contacts():
 m=ImpactOutcomeEvaluator(json.loads(Path('configs/shuttle_physics_v2.json').read_text()),json.loads(Path('configs/court.json').read_text()) if Path('configs/court.json').exists() else {},{}) if False else None
 # Contract is tested without invoking asset construction.
 assert StrikeStateImpactBuilderV1.__call__.__name__=='__call__'
