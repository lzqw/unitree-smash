"""Run the persistent-recovery evaluator with deferred diagnostic JSON I/O.

Use the existing observer subclass that inherits all force sampling and final
save behavior. Only repeated event-history serialization is deferred. This
entry is for subsequent jobs; it does not change already running evaluation.
"""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.airborne_contact_observer_fast_v1 import DeferredStrikeGroundContactObserver
import scripts.strike_ground_contact_observer_v1 as observer_module

observer_module.StrikeGroundContactObserver = DeferredStrikeGroundContactObserver
runpy.run_path(str(Path(__file__).with_name(
    'evaluate_shared_target_ground64_persistent_recovery_v1.py')), run_name='__main__')
