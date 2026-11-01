"""Geometry evidence for a ground-extent-only native diagnostic.

The robot, upper ground surface, thickness, material and controller stay fixed.
Changing remote horizontal ground boundaries is an explicitly labelled numeric
experiment, not a new task score or a reinterpretation of historical failures.
"""
import json
from pathlib import Path

import numpy as np

from scripts.run_landing_validation_sequence_v1 import sha


def validate_bounds(lower, upper, dimensions):
    lower, upper, dimensions = map(lambda x: np.asarray(x, dtype=np.float64),
                                   (lower, upper, dimensions))
    if any(x.shape != (3,) or not np.isfinite(x).all() for x in [lower, upper, dimensions]):
        raise ValueError('Invalid actual ground bounds')
    if (not np.allclose(upper-lower, dimensions, atol=1e-6, rtol=0)
            or not np.allclose((upper+lower)[:2], 0., atol=1e-6, rtol=0)
            or abs(upper[2]) > 1e-6 or abs(lower[2]+.1) > 1e-6):
        raise ValueError('Ground dimensions, top surface or thickness differ from the declared diagnostic')


def observe_ground(stage, original_dimensions, requested_dimensions):
    from pxr import Usd, UsdGeom, UsdPhysics
    prim = stage.GetPrimAtPath('/World/Ground/geometry/mesh')
    if not prim.IsValid() or not prim.HasAPI(UsdPhysics.CollisionAPI):
        raise ValueError('Missing actual ground collision primitive')
    bound = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy']).ComputeWorldBound(prim)
    aligned = bound.ComputeAlignedRange()
    lower, upper = list(aligned.GetMin()), list(aligned.GetMax())
    validate_bounds(lower, upper, requested_dimensions)
    attributes = {}
    current = prim
    while current.IsValid() and str(current.GetPath()).startswith('/World/Ground'):
        for attr in current.GetAttributes():
            if attr.GetName().startswith(('physics:', 'physxCollision:', 'xformOp:', 'size')):
                value = attr.Get()
                attributes[str(current.GetPath())+'.'+attr.GetName()] = str(value)
        current = current.GetParent()
    return dict(scope='Actual USD collision geometry; not a cooked PhysX-shape equivalence claim',
        prim_path=str(prim.GetPath()), prim_type=prim.GetTypeName(),
        original_configured_dimensions_m=list(original_dimensions),
        requested_dimensions_m=list(requested_dimensions),
        actual_world_lower_m=lower, actual_world_upper_m=upper,
        actual_collision_enabled=bool(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()),
        authored_and_schema_attributes=attributes,
        only_horizontal_extent_changed=True, formal_acceptance=False)


def compare_completed(source, baseline):
    from scripts.audit_standing_contact_telemetry_v1 import audit_telemetry
    source, baseline = Path(source), Path(baseline)
    result = audit_telemetry(source)
    original = audit_telemetry(baseline)
    ground = json.loads((source/'ground_geometry.json').read_text())
    validate_bounds(ground['actual_world_lower_m'], ground['actual_world_upper_m'],
                    ground['requested_dimensions_m'])
    if not ground['actual_collision_enabled']:
        raise ValueError('Ground collision disabled')
    one, two = [json.loads((p/'INPUTS.json').read_text()) for p in [baseline, source]]
    for key in ['protocol_sha256','source_indices_per_group','groups','ground_noise',
                'ground_noise_seed','first_pair_nominal','control_dt_s','physics_dt_s']:
        if one[key] != two[key]:
            raise ValueError('Ground comparison changes '+key)
    for name in ['initial_states.json','controller_settings.json']:
        if json.loads((source/name).read_text()) != json.loads((baseline/name).read_text()):
            raise ValueError('Matched comparison differs in '+name)
    for p in [source, baseline]:
        admission = json.loads((p/'native_admission.json').read_text())
        if not admission['passed']:
            raise ValueError('Loaded physical model admission failed')
    with np.load(source/'actual_first_episode_streams.npz') as data:
        poses = data['link_poses']; valid = data['valid_first_episode']
        extent = np.max(np.abs(poses[..., :2])[valid], axis=(0,1))
    half = np.asarray(ground['requested_dimensions_m'][:2])/2
    if np.any(extent+1. >= half):
        raise ValueError('Robot link origins approach a changed horizontal boundary')
    result['ground_extent_comparison'] = dict(
        baseline=str(baseline), baseline_counts=original['counts'], candidate_counts=result['counts'],
        actual_ground=ground, maximum_recorded_link_origin_abs_xy_m=extent.tolist(),
        boundary_check_scope='Recorded link origins plus conservative1m margin; not full swept geometry certification',
        paired_initial_states_equal=True, controller_settings_equal=True,
        baseline_telemetry=original['additional_telemetry'],
        old_results_not_relabelled=True, physical_cause_not_automatically_certified=True,
        landing_or_smash_success_claimed=False)
    result['evidence_sha256']['ground_geometry.json'] = sha(source/'ground_geometry.json')
    result['baseline_evidence_sha256'] = original['evidence_sha256']
    return result
