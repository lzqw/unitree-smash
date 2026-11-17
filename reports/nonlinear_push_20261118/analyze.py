"""Read-only paired model response, not a native counterfactual."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
from quadsmash.control.full_racket_geometry_v4 import FullRacketGeometryV4
from quadsmash.control.contact_reach_bound_v1 import string_strip_reach_bound
import gzip
root=Path(__file__).resolve().parents[2];out=Path(__file__).resolve().parent
metadata=json.load(gzip.open(root/'reports/experimental_fullbody_gate_20261118/REPLAY.json.gz','rt'))['physical_calls'][0]['metadata']
initial=json.loads((root/'reports/push_off_task_20261118/DIAGNOSED.json').read_text())['rows'][0]
q0=np.asarray(initial['initial_q']);R0=Rotation.from_quat(q0[3:7]).as_matrix()
ball=q0[:3]+R0@np.asarray(metadata['target_position_world_m'])
model=LoadedPinocchioModelV1();rows=[]
for name,path in [('fixed8',root/'reports/push_off_task_20261118/PREFIX_IMPLICIT.json'),('contact_latched',root/'reports/contact_gated_push_20261118/RESULTS.json'),('momentum_preserved',root/'reports/momentum_push_20261118/RESULTS.json'),('nonlinear_corrected',out/'RESULTS.json')]:
    r=json.loads(path.read_text())['results'][0];trace=r['trajectory'];end=trace[-1]
    q=np.asarray(end['q']);v=np.asarray(end['v']);snap=model.snapshot(q,v)
    R=Rotation.from_quat(q[3:7]).as_matrix();target=R.T@(ball-q[:3])
    bound=string_strip_reach_bound(FullRacketGeometryV4(),target,metadata['task_string_strip'],ball_radius_m=metadata['task_physics']['collision_radius_m'])
    flight=next((s['elapsed_s'] for s in trace if not any(s['contacts'])),None)
    decisions=[s for s in trace if s.get('window',{}).get('enabled',False)]
    rows.append(dict(mode=name,steps=r['steps'],adopted=r['adopted_qp_commands'],
        terminal_errors=r['terminal_errors'],root_position=q[:3].tolist(),
        root_rpy_rad=Rotation.from_matrix(R).as_euler('xyz').tolist(),
        root_angular_velocity_world=(R@v[3:6]).tolist(),
        com_position=snap['com_world'].tolist(),com_velocity=snap['com_velocity_world'].tolist(),
        centroidal_momentum=snap['momentum_world'].tolist(),first_all_feet_released_s=flight,
        reach_bound=bound,terminal_contact_passed=r['terminal_contact_passed'],
        max_roundtrip_error_rad=max(s['command_error_rad'] for s in trace),
        enabled_nodes=[s['node'] for s in decisions],
        cpu_only=True))
(out/'COMPARISON.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
