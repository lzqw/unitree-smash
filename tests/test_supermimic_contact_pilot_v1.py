import json

import torch

from quadsmash.rl.supermimic_contact_pilot_v1 import _trace_candidates, build_pilot


def _trace(path, *, touching=True):
    frames, lanes = 12, 2
    state = torch.zeros(frames, lanes, 13)
    state[..., 2] = .5
    state[..., 7] = -4.
    state[..., 9] = 1.
    racket = torch.zeros(frames, lanes, 3)
    racket[..., 0] = .02
    quat = torch.zeros(frames, lanes, 4)
    quat[..., 0] = 1.
    lin = torch.zeros_like(racket)
    ang = torch.zeros_like(racket)
    force = torch.zeros_like(racket)
    point = torch.zeros_like(racket)
    point[..., 0] = .025
    if touching:
        force[4:6, 0, 0] = 10.
    trace = dict(schema="native-one-control-tick-substeps-v1", physics_dt_s=.001,
                 tensors={"shuttle_state_w": state,
                          "racket_position_w": racket,
                          "racket_quaternion_w": quat,
                          "racket_linear_velocity_w": lin,
                          "racket_angular_velocity_w": ang,
                          "racket_force_w": force,
                          "racket_contact_position_w": point})
    torch.save(trace, path)


def test_pilot_selects_dynamic_contact_and_near_miss(tmp_path):
    source = tmp_path / "trace.pt"
    _trace(source)
    manifest = build_pilot(source, tmp_path / "pilot", contacts=1, negatives=1)
    assert manifest["selected_contacts"] == 1
    assert manifest["selected_negatives"] == 1
    payload = torch.load(tmp_path / "pilot" / "pilot.pt", weights_only=True)
    contact = next(row for row in payload["rows"] if row["contact_object"] == "racket")
    assert contact["toi_frame"] == 4
    assert contact["contact_count_substeps"] == 2
    assert contact["normal_impulse_ns"] > 0
    assert len(contact["outgoing_direction_w"]) == 3


def test_pilot_reports_insufficient_candidates_without_duplication(tmp_path):
    source = tmp_path / "trace.pt"
    _trace(source, touching=False)
    manifest = build_pilot(source, tmp_path / "pilot", contacts=2, negatives=100)
    assert manifest["selected_contacts"] == 0
    assert manifest["selected_negatives"] <= manifest["candidate_negatives"]
    assert manifest["pilot_status"] == "INSUFFICIENT_CANDIDATES"
    recorded = json.loads((tmp_path / "pilot" / "manifest.json").read_text())
    assert recorded["schema"] == "supermimic-racket-ball-contact-pilot-v1"


def test_contact_matrix_keeps_multiple_slots_and_samples_lane_negatives(tmp_path):
    source = tmp_path / "trace.pt"
    _trace(source)
    payload = torch.load(source, weights_only=False)
    force = payload["tensors"]["racket_force_w"]
    matrix = torch.zeros(force.shape[0], force.shape[1], 2, 3)
    matrix[..., 0, :] = force
    matrix[4:6, 0, 1, 1] = 7.
    payload["tensors"]["contact_force_matrix_w"] = matrix
    torch.save(payload, source)
    rows, sensitivity = _trace_candidates(source, .001, 1, (.0001, .001, .005, .01))
    contact = next(row for row in rows if row["contact_object"] == "racket")
    assert contact["contact_slot_count"] == 2
    assert contact["contact_slot_indices"] == [0, 1]
    assert contact["slot_impulse_ns"][1][1] > 0
    assert any(row["contact_object"] == "no_contact" and row["lane"] == 0 for row in rows)
    assert set(sensitivity) == {"0.0001", "0.001", "0.005", "0.01"}


def test_native_filters_separate_ground_and_robot_net_from_racket(tmp_path):
    path = tmp_path / "trace.pt"
    _trace(path, touching=False)
    trace = torch.load(path, weights_only=True)
    tensors = trace["tensors"]
    matrix = torch.zeros(12, 2, 1, 2, 3)
    matrix[1, 0, 0, 1, 2] = 20.0
    matrix[7, 0, 0, 0, 0] = 10.0
    matrix[7, 0, 0, 1, 2] = 20.0
    matrix[8, 1, 0, 0, 0] = 11.0
    points = torch.full_like(matrix, float("nan"))
    points[7, 0, 0, 1] = 40.0
    inverse = torch.zeros(12, 2, 1, 3, 3)
    inverse[7, 0, 0, 0, 2] = 15.0
    inverse[8, 1, 0, -1, 2] = 16.0
    tensors.update(contact_force_matrix_w=matrix, contact_position_matrix_w=points,
                   net_force_matrix_w=inverse)
    trace["contact_metadata"] = {"racket_body_name": "Racket"}
    torch.save(trace, path)
    rows, sensitivity = _trace_candidates(path, .001, 1, (.001,))
    contacts = [row for row in rows if row["contact_object"] == "racket"]
    assert [(row["lane"], row["toi_frame"]) for row in contacts] == [(0, 7), (1, 8)]
    assert sensitivity["0.001"]["contact_intervals"] == 2
    assert contacts[0]["interference_objects"] == ["ground"]
    assert contacts[1]["interference_objects"] == ["net"]
    assert contacts[0]["gap_source"] == "analytic_racket_geometry"
    assert contacts[0]["racket_body_name"] == "Racket"
    assert contacts[0]["racket_part"] == "unknown"
    torch.testing.assert_close(torch.tensor(contacts[0]["impulse_ns"]), torch.tensor([.01, 0., 0.]))


def test_deterministic_replays_count_once_and_changed_velocity_survives(tmp_path):
    from quadsmash.rl.supermimic_contact_pilot_v1 import _deduplicate
    path = tmp_path / 'trace.pt'
    _trace(path)
    rows, _ = _trace_candidates(path, .001, 1)
    row = next(r for r in rows if r['contact_object'] == 'racket')
    replay = dict(row, source_sha256='repeat', capture_id='another', event_id='nominal_b')
    varied = dict(replay, toi_post_substep={**replay['toi_post_substep'], 'ball_state_w': [9.] * 13})
    unique, count = _deduplicate([row, replay, varied])
    assert unique == [row, varied]
    assert count == 1
