"""CPU gate for the direct-sequence rescue candidate; never launches Isaac."""
import copy, json, sys
from pathlib import Path
import numpy as np, pinocchio as pin
ROOT=Path(__file__).resolve().parents[1]; MAIN=Path('/home/autodrive/LZQW/unitree_smash_contact_recovery')
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(MAIN))
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.loaded_pinocchio_model_v1 import PublicState
from scripts.pure_control_state_driven_long_mpc_v1_qp_continuity_direct_search import PureControlStateDrivenLongMPCV1QPContinuityDirectSearch as Candidate
RUN=ROOT/'reports/pure_control_state_driven_long_mpc_qp_continuity_repair_all128_corrected_20261110_03/native'; BASE=MAIN/'reports/pure_control_state_driven_long_mpc_qp_continuity_all128_20261109_01/native'; OUT=ROOT/'reports/direct_search_preflight_20261111'
def read(p):return json.loads(Path(p).read_text())
def main():
 OUT.mkdir(parents=True,exist_ok=True); rows=[json.loads(x) for x in (RUN/'pure_control_diagnostics.jsonl').open()]
 with np.load(RUN/'pure_control_commands.npz') as z:data={k:z[k].copy() for k in z.files}
 fixed=sorted(r['training_index'] for r in read(BASE/'metrics.json')['complete5_case_outcomes'] if r['air']); first={}
 for i,r in enumerate(rows):
  if r['lane'] in fixed and r['diagnostic'].get('replanned') and not r['diagnostic'].get('solver_accepted') and r['lane'] not in first:first[r['lane']]=i
 cfg=read(ROOT/'reports/pure_control_state_driven_long_mpc_qp_continuity_repair_native_preparation_20261110_01/controller.json');nom=np.asarray(read(RUN/'native_command_inputs.json')['default_joint_position_rad'])
 records=[]
 for lane in fixed:
  i=first[lane];q=data['q'][i].copy();v=data['v'][i].copy();c=Candidate(PositionPDInterface(**cfg['interface']),nom)
  s=PublicState(q,v,pin.Quaternion(q[3:7]).matrix(),q[:3].copy(),q[c.model.q_indices].copy(),v[c.model.v_indices].copy(),np.asarray(rows[i]['phase']['ground_contacts']),0.)
  s.previous_command_target=data['previous'][i].copy();s.measured_force_world=data['ground_force'][i].copy()
  target,diag=c.command(s,rows[i]['phase']['phase'],float(data['time_s'][i]))
  direct=diag.get('direct_search',[]); selected=diag.get('direct_search_selected_scale')
  # A selected sequence must be complete and have a physical first target.
  if selected is not None:
   assert diag.get('direct_search_sequence_complete') and len(c._plan)==32 and c._plan_cursor==1
   assert np.isfinite(target).all() and np.all(target>=c.interface.limits[:,0]-1e-6) and np.all(target<=c.interface.limits[:,1]+1e-6)
  records.append(dict(lane=lane,tick=int(data['tick'][i]),selected_scale=selected,
      attempted=len(direct),accepted=[x.get('scale') for x in direct if x.get('accepted')],
      rejections=[dict(scale=x.get('scale'),node=x.get('failed_node'),reason=x.get('rejection')) for x in direct if not x.get('accepted')],
      first_target=target.tolist(),source=diag.get('source'),sequence_length=len(c._plan),
      parent_rejection=diag.get('plan',{}).get('rejection')))
  print(lane,selected,flush=True)
 summary=dict(status='CPU_PREFLIGHT_PASS',native_evaluation_started=False,fixed74=fixed,
    cases=len(records),selected_cases=sum(r['selected_scale'] is not None for r in records),
    accepted_case_ids=[r['lane'] for r in records if r['selected_scale'] is not None],
    candidate_scales=[2.25,3.0],sequence_nodes=32,actual_executor_preview='inherited _predict_request and native arbiter; no direct command executed',
    no_partial_execution=True,unknown_case_ids_used=False,source='direct full-authority command sequence; QP not required for candidate generation')
 (OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');(OUT/'cases.json').write_text(json.dumps(records,indent=2)+'\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
