"""Read-only position-stop response near the rejected measured-400 suffix."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.check_constrained_recovery_suffix_v1 import source_context,state_from_row
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.predict_speed_limited_spheres_v1 import predict_speed_limited_spheres
from scripts.audit_joint_sequence_proposal_v1 import plain

read=lambda p:json.loads(p.read_text());out=Path(__file__).parent/'stop412.json'
if out.exists():raise ValueError('Fresh evidence required')
setup=ROOT/'reports/landing_speed_limited_family_search_20261108/plan.json'
source=read(Path(read(setup)['source']));_,_,_,_,context=source_context(source['configuration']['source']);initialize(*context)
w=_WORKER;model=w['controller'].qp.model;interface=w['controller'].qp.interface
trace_path=ROOT/'reports/native_timed85_all128_20261108_01/sequence_native_trace.npz'
with np.load(trace_path,allow_pickle=False) as z:trace={k:z[k].copy() for k in z.files}
j=model.joint_order.index('RL_calf_joint');qi=model.q_indices[j];vi=model.v_indices[j];rows=[]
for tick in range(408,417):
    i=tick-280
    state=state_from_row(dict(q=trace['q'][i],v=trace['v'][i],contacts=trace['contacts'][i],target=trace['previous_target_rad'][i]))
    prediction=predict_speed_limited_spheres(model,interface,state,trace['final_target_rad'][i],w['spheres'],geometry=w['geometry'],substeps=160)
    following=prediction['state']
    rows.append(dict(tick=tick,q=trace['q'][i,qi],v=trace['v'][i,vi],command=trace['final_target_rad'][i,j],
        next_actual_q=trace['q'][i+1,qi],next_predicted_q=following.q[qi],
        next_actual_v=trace['v'][i+1,vi],next_predicted_v=following.v[vi],
        predicted_joint_position_violation_rad=prediction['diagnostics']['joint_position_violation_rad']))
paths=[setup,trace_path,Path(__file__),ROOT/'reports/landing_native_timed85_suffixes_20261108/tick_400.json']
paths += [Path(m.__file__).resolve() for n,m in sys.modules.items() if n.startswith('scripts.') and getattr(m,'__file__',None)]
result=dict(joint=model.joint_order[j],limits_rad=interface.limits[j],rows=rows,
    source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    scope=__doc__,new_native_execution=False,valid_expert_labels=False,
    limitation='Position stop is not in the predictor. This diagnostic does not identify all other PhysX model differences.')
out.write_text(json.dumps(plain(result),indent=2,allow_nan=False)+'\n')
print(json.dumps(plain(rows[3])))
