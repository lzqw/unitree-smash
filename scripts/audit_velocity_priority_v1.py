"""Audit a velocity-priority prefix against its exact selected predecessor."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pinocchio as pin
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
from scripts.contact_terminal_attribution_v1 import contact_kinematics

def main():
 p=argparse.ArgumentParser();p.add_argument('--control',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
 for x in (a.control,a.candidate):
  if sha(x)!=x.with_suffix('.sha256').read_text().strip():raise ValueError('Artifact hash mismatch')
 c=json.loads(a.control.read_text());r=json.loads(a.candidate.read_text());base=c['trials'][c['selected_index']]['result'];seed=r['trials'][0]['result']
 # Goal metadata may differ; physical q/v/command history below is the authoritative baseline check.
 if len(base['trajectory'])!=len(seed['trajectory']):raise ValueError('Prefix length changed')
 for x,y in zip(base['trajectory'],seed['trajectory']):
  for k in ('q','v','final_target_rad','planned_arm_reference_rad'):
   if not np.array_equal(x[k],y[k]):raise ValueError('Predecessor history changed: '+k)
 selected=r['trials'][r['selected_index']]['result'];initial=json.loads(Path('reports/push_off_task_20261118/DIAGNOSED.json').read_text())['rows'][0];model=LoadedPinocchioModelV1();m=r['metadata'];off=np.array(m['contact_point_local']);off[0]+=m['task_physics']['collision_radius_m']; rows=[]
 for name,z in [('predecessor',base),('velocity_priority',selected)]:
  q=np.array(z['trajectory'][-1]['q']);v=np.array(z['trajectory'][-1]['v']);p,n,J,_=contact_kinematics(model,q,off);components={}
  for label,idx in [('base',np.arange(6)),('legs',model.v_indices[:12]),('arm',model.v_indices[12:])]:
   vv=np.zeros_like(v);vv[idx]=v[idx];components[label]=J@vv
  rows.append(dict(label=name,terminal_errors=z['terminal_errors'],terminal_velocity_m_s=J@v,velocity_components_m_s=components,arm_velocity_rad_s=v[model.v_indices[12:]],max_arm_tracking_rad=max(t['arm_tracking_error_rad'] for t in z['trajectory']),max_mapping_error_rad=max(t['command_error_rad'] for t in z['trajectory'])))
 out=dict(predecessor_selected_hash=sha(a.control),candidate_hash=sha(a.candidate),baseline_history_identical=True,rows=rows,objective=r.get('objective'),accepted_step_scale=r.get('accepted_step_scale'),fresh_rollouts=len(r['trials']),seconds=r['seconds'],native_execution=False,valid_teacher_reference=False)
 a.output.write_text(json.dumps(out,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else float(x))+'\n');print(json.dumps(dict(baseline_identical=True,rows=[x['terminal_errors'] for x in rows])))
if __name__=='__main__':main()
