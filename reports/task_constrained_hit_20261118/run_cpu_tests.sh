#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 bash scripts/python.sh -m pytest -q \
 tests/test_contact_replan_gate_v1.py tests/test_contact_pose_reachability_v1.py \
 tests/test_inverse_impact_target_v1.py tests/test_causal_final_command_history_v1.py \
 tests/test_contact_first_base_transport_v1.py tests/test_probe_rally_export_v1.py \
 tests/test_contact_first_trajectory_search_v1.py tests/test_joint_prefix_execution_v1.py \
 tests/test_impact_outcome_evaluator_v1.py tests/test_contact_first_fullbody_evaluator_v1.py \
 tests/test_unilateral_implicit_prediction_v1.py tests/test_position_anchored_contact_search_v1.py tests/test_constrained_contact_normal_v1.py tests/test_velocity_aware_contact_pose_v1.py tests/test_offline_reference_bundle_v1.py tests/test_impact_feasible_rate_v1.py tests/test_task_constrained_hit_v2.py
