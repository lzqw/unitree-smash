"""Read-only terminal-error decomposition, never a controller input."""
import gzip,json
from pathlib import Path
import numpy as np
import pinocchio as pin
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
from scripts.audit_prediction_contact_geometry_v2 import racket_contact_state
root=Path(__file__).resolve().parent
path=root/'REPLAY.json'
x=json.loads(path.read_text()) if path.exists() else json.load(gzip.open(str(path)+'.gz','rt'))
rows=[]
for call in x['physical_calls']:
    r=call['result'];m=call['metadata'];trace=r.get('state_trace') or []
    if not trace or 'actual_racket_position_world_m' not in r:continue
    model=LoadedPinocchioModelV1();q=np.asarray(trace[-1]['q']);v=np.asarray(trace[-1]['v'])
    q_ideal=q.copy();q_ideal[model.q_indices[12:]]=np.asarray(call['arm_sequence_rad'])[m['contact_index']]
    offset=np.asarray(m['contact_point_local']).copy();offset[0]+=.025  # original ball-center proxy
    actual=racket_contact_state(model,q,v,offset)[0].copy()
    ideal=racket_contact_state(model,q_ideal,v,offset)[0].copy()
    rotation=np.asarray(m['planning_to_world_rotation'])
    origin=np.asarray(r['geometry_witness_initial']['root_position_world_m'])
    callback_error=float(np.linalg.norm(rotation.T@(actual-origin)-np.asarray(r['actual_racket_position_world_m'])))
    if callback_error>1e-5:raise ValueError(f'Terminal FK/callback mismatch: {callback_error}')
    arm_effect=rotation.T@(actual-ideal)
    actual_c=np.asarray(r['actual_racket_position_world_m']);planned=np.asarray(m['achieved_position_current_frame_m'])
    base_effect=actual_c-planned-arm_effect
    residual=planned-np.asarray(m['target_position_world_m'])
    error=actual_c-np.asarray(m['target_position_world_m'])
    assert np.max(abs(error-base_effect-arm_effect-residual))<1e-12
    rows.append(dict(lane=call['lane'],contact_time_s=m['contact_time_s'],
        callback_fk_error_m=callback_error,
        decomposition_order='hold actual base, replace only terminal arm angles with planned endpoint',
        total_error_m=error.tolist(),total_error_norm_m=float(np.linalg.norm(error)),
        arm_tracking_effect_m=arm_effect.tolist(),arm_tracking_effect_norm_m=float(np.linalg.norm(arm_effect)),
        remaining_base_transport_effect_m=base_effect.tolist(),remaining_base_transport_effect_norm_m=float(np.linalg.norm(base_effect)),
        initial_kinematic_residual_m=residual.tolist(),initial_kinematic_residual_norm_m=float(np.linalg.norm(residual)),
        caution='Vector decomposition conditional on FK/frame equivalence already audited; component norms are not additive. Not native truth or a base-correction command.'))
(root/'TERMINAL_ATTRIBUTION.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
