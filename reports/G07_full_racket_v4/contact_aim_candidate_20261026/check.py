"""Check new aiming commands on existing real public inputs, without physics."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
MAIN = ROOT.parent/'unitree_smash'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(MAIN))
from quadsmash.control.full_racket_body_residual_v4 import make_rally_controller
from quadsmash.control.full_racket_geometry_v4 import FullRacketGeometryV4
from scripts.initialize_centered_policy_v4 import same_tree

module_path = ROOT/'quadsmash/control/full_racket_contact_aim_v4.py'
spec = importlib.util.spec_from_file_location('new_contact_aim_proposal', module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (HERE/'PLAN.json').exists(), 'Preserve previous evidence'
    owner = MAIN/'reports/G07_full_racket_v4/body_residual48_after_grid16_queue01'
    admission = read(owner/'ADMISSION_DECISION.json')
    assert all(sha(MAIN/name) == digest for name, digest in admission['source_hashes'].items())
    source_path = Path(admission['source']['policy'])
    run = Path(admission['source']['evaluation'])
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    before = torch.load(source_path, map_location='cpu', weights_only=False)
    manifest_path = MAIN/'reports/G07_full_racket_v4/body_residual_leg15_fresh_n2048_updates48_01/trajectory_traces/manifest.json'
    provenance = read(manifest_path)['provenance']
    (HERE/'ACTUAL_JOINT_PROVENANCE.json').write_text(json.dumps(provenance, indent=2)+'\n')
    default = torch.tensor(provenance['default_joint_position_rad'], dtype=torch.float32)
    limits = torch.tensor(provenance['joint_limits_rad'], dtype=torch.float32)
    stream_path = run/'public_and_actions_first2.npz'
    with np.load(stream_path) as data:
        public, actions, valid = [data[k].copy() for k in ('observations', 'actions', 'valid_first_episode')]
    assert public.shape[1:] == (2, 794) and valid.shape == public.shape[:2]
    sources = [Path(__file__), module_path, source_path, stream_path,
               HERE/'ACTUAL_JOINT_PROVENANCE.json', owner/'ADMISSION_DECISION.json']
    plan = dict(created_at=datetime.now(timezone.utc).isoformat(), source_sha256={str(p):sha(p) for p in sources},
        hypothesis='Measured35 frame-only events cluster near lateral edges; test more interior analytic aiming points without changing physical strings, source actors, body timing or scoring.',
        candidates_uv_m=module.aim_candidates(), baseline_action_tolerance=5e-4,
        matched_reference_must_be_bitwise_identical=True, analytic_jacobian_tolerance=1e-7,
        source_physical_contract='Original v4-r3 asset, 5rad/s arm target limit, original TRAIN128 and all acceptance checks.',
        new_native_execution=False, formal_acceptance=False)
    (HERE/'PLAN.json').write_text(json.dumps(plan, indent=2)+'\n')
    base = make_rally_controller(source, 2, 'cpu')
    candidates = [module.ContactAimControllerV4(source, uv, 2, 'cpu') for uv in module.aim_candidates()]
    maximum_reference = maximum_native = 0.
    changes = [[] for _ in candidates]
    slew_errors = []
    previous_targets = [None for _ in candidates]
    previous_clocks = None
    with torch.inference_mode():
        for obs, recorded, keep in zip(public, actions, valid):
            tensor = torch.from_numpy(obs)
            reference = base.action({'policy':tensor}, default, limits)
            if keep.any():
                maximum_native = max(maximum_native, float(abs(reference.numpy()[keep]-recorded[keep]).max()))
            for i, candidate in enumerate(candidates):
                action = candidate.action({'policy':tensor}, default, limits)
                assert action.shape == (2, 18) and torch.isfinite(action).all() and (action.abs() <= 1).all()
                if keep.any():
                    change = abs(action.numpy()[keep]-reference.numpy()[keep])
                    changes[i].append(change)
                    if i == 0:
                        maximum_reference = max(maximum_reference, float(change.max()))
                targets = limits[:,0] + .5*(action+1)*(limits[:,1]-limits[:,0])
                if previous_clocks is not None:
                    consecutive = keep & (obs[:,775] > previous_clocks) & (obs[:,775]-previous_clocks < .011)
                    if consecutive.any():
                        step = abs(targets.numpy()[consecutive,12:]-previous_targets[i][consecutive,12:])
                        slew_errors.append(float((step-.05).max()))
                previous_targets[i] = targets.numpy().copy()
            previous_clocks = obs[:,775].copy()
    assert maximum_reference == 0 and maximum_native <= plan['baseline_action_tolerance']
    assert max(slew_errors) <= 2e-6, max(slew_errors)
    geometry = FullRacketGeometryV4()
    # Independent source-URDF finite differences at actual recorded arm states.
    samples = public.reshape(-1, 794)
    samples = samples[(samples[:,775] > 2.3) & (samples[:,775] < 3.)][::20]
    assert len(samples) >= 6
    jacobian_error = 0.
    center_guard_error = 0.
    for candidate, uv in zip(candidates, module.aim_candidates()):
        teacher = candidate.reference.teacher
        kinematics = teacher.kinematics.to(dtype=torch.float64)
        q = default[12:].double()[None] + torch.from_numpy(samples[:,21:27]).double()
        point, normal, jp, jn = kinematics(q, teacher.contact_offset.double())
        for i, joints in enumerate(q.numpy()):
            finite = geometry.contact_jacobian(joints, uv=uv, face=1, base_position=(0,0,0))
            jacobian_error = max(jacobian_error, float(abs(jp[i].numpy()-finite).max()))
        original = base.reference_controller.teacher.kinematics.to(dtype=torch.float64)
        # Calls without a contact offset are used by the full physical-tool
        # support guard; changing the aiming point must leave them identical.
        center_guard_error = max(center_guard_error, float((kinematics(q)[0]-original(q)[0]).abs().max()))
    assert jacobian_error <= plan['analytic_jacobian_tolerance'] and center_guard_error == 0
    assert same_tree(source, before), 'Never mutate the embedded measured checkpoint'
    assert all(sha(Path(p)) == digest for p,digest in plan['source_sha256'].items())
    assert all(sha(MAIN/name) == digest for name,digest in admission['source_hashes'].items())
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(),
        actual_valid_public_rows=int(valid.sum()), actual_arm_states_for_jacobian_check=len(samples),
        matched_reference_max_normalized_action_error=maximum_reference,
        reference_CPU_vs_native_max_normalized_action_error=maximum_native,
        maximum_independent_position_jacobian_error=jacobian_error,
        physical_support_kinematics_max_error=center_guard_error,
        maximum_arm_target_slew_excess_rad=max(slew_errors),
        candidates=[dict(uv_m=uv, max_normalized_action_change=float(np.concatenate(change).max()),
                         mean_abs_normalized_action_change=float(np.concatenate(change).mean()))
                    for uv,change in zip(module.aim_candidates(), changes)],
        embedded_measured_source_unchanged=True, original407_active_inputs_unchanged=True,
        source_actor_weights_and_normalizers_unchanged=True,
        scope='CPU command and source-URDF checks on real previously recorded inputs; no new physical rollout, hit or success claim.',
        new_native_execution=False, formal_acceptance=False)
    (HERE/'RESULT.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
