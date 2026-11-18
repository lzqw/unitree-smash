"""Read-only terminal sensitivity and reach diagnostics, never an action source."""
import gzip,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
from quadsmash.control.full_racket_geometry_v4 import FullRacketGeometryV4
from quadsmash.control.contact_reach_bound_v1 import string_strip_reach_bound
out=Path(__file__).parent;root=out.resolve().parents[1]
x=json.loads((out/'RESULTS.json').read_text());base=x['trials'][0]
m=json.load(gzip.open(root/'reports/experimental_fullbody_gate_20261118/REPLAY.json.gz','rt'))['physical_calls'][0]['metadata']
initial=json.loads((root/'reports/push_off_task_20261118/DIAGNOSED.json').read_text())['rows'][0]
q0=np.array(initial['initial_q']);ball=q0[:3]+Rotation.from_quat(q0[3:7]).apply(m['target_position_world_m'])
radii=np.array(x['parameter_radii_rad']);J=np.zeros((9,len(radii)));r0=np.array(base['result']['normalized_residual'])
for trial in x['trials']:
 if trial['label'].startswith('derivative_') and trial['valid']:
  j=int(trial['label'].split('_')[1]);J[:,j]=(np.array(trial['result']['normalized_residual'])-r0)/trial['amplitude_rad'][j]
rows=[];model=LoadedPinocchioModelV1()
for index in sorted(set([0,x['selected_index'],len(x['trials'])-1])):
 trial=x['trials'][index];result=trial['result']
 if not trial['valid']:continue
 end=result['trajectory'][-1];q=np.array(end['q']);R=Rotation.from_quat(q[3:7]).as_matrix();target=R.T@(ball-q[:3])
 snapshot=model.snapshot(q,np.array(end['v']))
 rows.append(dict(index=index,label=trial['label'],cost=trial['cost'],terminal_errors=result['terminal_errors'],
  racket_velocity_at_planned_event_current_frame_m_s=(np.array(result['normalized_residual'][6:])*.75+np.array(m['retargeted_velocity_current_frame_m_s'])).tolist(),root_position=q[:3].tolist(),root_rpy_rad=Rotation.from_matrix(R).as_euler('xyz').tolist(),
  actual_arm_q=q[model.q_indices[12:]].tolist(),actual_arm_v=np.array(end['v'])[model.v_indices[12:]].tolist(),momentum=snapshot['momentum_world'].tolist(),reach_bound=string_strip_reach_bound(FullRacketGeometryV4(),target,m['task_string_strip'],ball_radius_m=m['task_physics']['collision_radius_m']),
  max_command_error_rad=max(s['command_error_rad'] for s in result['trajectory']),
  support_basis_nodes=[s['node'] for s in result['trajectory'] if s['weight']!=0],
  changed_command_nodes=[s['node'] for s,b in zip(result['trajectory'],base['result']['trajectory']) if not np.array_equal(s['final_target_rad'],b['final_target_rad'])],
  arm_commands_exact=all(np.array_equal(s['final_target_rad'][12:],b['final_target_rad'][12:]) for s,b in zip(result['trajectory'],base['result']['trajectory']))))
proposal=x['trials'][-1]
summary=dict(rows=rows,normalized_response_jacobian=J.tolist(),singular_values=np.linalg.svd(J,compute_uv=False).tolist(),scaled_parameter_singular_values=np.linalg.svd(J*radii,compute_uv=False).tolist(),
 local_linear_position_axis_change_upper_m=(.035*np.sum(abs(J[:3])*radii,axis=1)).tolist(),
 linearization_scope='first-order fixed temporal basis, not a global robot reach bound',
 prediction_error_norm=(float(np.linalg.norm(np.array(proposal['result']['normalized_residual'])-x['predicted_residual'])) if proposal['label']=='proposal' and proposal['valid'] else None))
(out/'ANALYSIS.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='normalized_response_jacobian'},indent=2))
