"""Requested/QP/nonlinear momentum-rate comparison in world coordinates."""
import json
from pathlib import Path
import numpy as np
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
from scripts.push_off_support_qp_v1 import PushOffSupportQPV1
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from types import SimpleNamespace
path=Path(__file__).parent;model=LoadedPinocchioModelV1()
initial=json.loads(Path('reports/push_off_task_20261118/DIAGNOSED.json').read_text())['rows'][0]
q=np.array(initial['initial_q']);v=np.array(initial['initial_v'])
interface=PositionPDInterface(**json.loads(Path('reports/codex_teacher_joint_eval_20261116/controller.json').read_text())['interface'])
qp=PushOffSupportQPV1(interface);rows=[]
for step in json.loads((path/'RESULTS.json').read_text())['results'][0]['trajectory']:
    before=model.snapshot(q,v);after=model.snapshot(np.asarray(step['q']),np.asarray(step['v']))
    plan=step['qp']
    if plan['accepted']:
        requested=np.array(plan['desired_angular_momentum_rate_world']);qp._momentum_rate=requested.copy()
        mat,target=qp.angular_task(SimpleNamespace(q=q,v=v),before,{})
        predicted=requested+before['centroidal_inertia_world']@(mat@np.array(plan['acceleration'])-target)
        actual=(after['momentum_world'][3:]-before['momentum_world'][3:])/.01
        rows.append(dict(node=step['node'],requested_Nm=requested.tolist(),qp_Nm=predicted.tolist(),
            nonlinear_average_Nm=actual.tolist(),qp_to_nonlinear_error_Nm=(predicted-actual).tolist()))
    q=np.array(step['q']);v=np.array(step['v'])
(path/'MOMENTUM_AUDIT.json').write_text(json.dumps(dict(scope='CPU model only, not measured native torque',rows=rows),indent=2)+'\n')
print('accepted calls',len(rows),'max QP/nonlinear difference Nm',max(np.linalg.norm(r['qp_to_nonlinear_error_Nm']) for r in rows))
