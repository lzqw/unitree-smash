"""Reuse the bounded timed shooting solver with a declared separation predictor.

The process-local function substitution changes only submitted prediction jobs;
no shared source file or native controller is changed. The imported module is
included in the existing solver's pre/post source-hash audit.
"""
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import scripts.current_state_timed_recovery_planner_v4 as planning
    from scripts.separated_timed_prediction_v1 import predict_plan
    from scripts.solve_native_timed_position_suffix_v1 import main as solve
    original=planning.predict_plan
    try:
        planning.predict_plan=predict_plan
        solve()
    finally:
        planning.predict_plan=original


if __name__=='__main__':
    # Import under its canonical name as well, so this driver is source-pinned
    # by the reused solver before any worker starts.
    from scripts.solve_native_separated_timed_suffix_v1 import main
    main()
