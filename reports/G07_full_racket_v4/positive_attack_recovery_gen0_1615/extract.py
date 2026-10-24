"""Freeze the two actually executed attacking-recovery candidates, including failures."""
import copy,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from quadsmash.evaluation_v4 import independent_episode
RUN=ROOT/'reports/G07_full_racket_v4/ball_slew50_near_attack_positive38_cem24';OUT=Path(__file__).parent
population_path=RUN/'population_000.pt';rows_path=RUN/'episodes_000.json'
state=torch.load(population_path,map_location='cpu',weights_only=False);rows=json.loads(rows_path.read_text())
assert json.dumps(rows,sort_keys=True)==json.dumps(state['rows'],sort_keys=True)
protocol=json.loads((ROOT/'configs/g07_full_racket_eval_v4_r3.json').read_text());protocol['_root']=str(ROOT)
assert state['candidate_count']==64 and state['training_case_indices']==[1,65]
with np.load(RUN/'arm_targets_000.npz') as data:vel=data['observed_velocity_rad_s'].copy();valid=data['valid_first_episode'].copy()
for candidate in [36,46]:
 out=OUT/f'candidate{candidate}';assert not out.exists();out.mkdir()
 group=rows[2*candidate:2*candidate+2];assert all(r['candidate_id']==candidate for r in group)
 policy=copy.deepcopy(state['controller_template'])
 for key in ['mean','sigma','rng','generations','scores','rows','best_parameters','best_result','reference_parameters']:
  policy.pop(key,None)
 policy.update(schema='full-racket-ball-cem-v4',parameters=state['parameters'][candidate].clone(),parameter_names=state['parameter_names'],training_suite=state['training_suite'],training_case_indices=state['training_case_indices'],training_result=group,native_execution=True,performance_inherited=False,formal_acceptance=False,selected_source_run=RUN.name,selected_generation=0,selected_candidate=candidate,kind='One actually executed CEM24candidate with TRAIN65attacking air+recovery but original target missed. Paired TRAIN1frame/body failure preserved; no full or distribution pass.')
 torch.save(policy,out/'policy_evidence.pt')
 checks=[]
 for row in group:
  (out/f'NATIVE_EPISODE_{row["env_id"]}.json').write_text(json.dumps(row,indent=2)+'\n')
  value=independent_episode(dict(row,case_id=f'TRAIN{row["source_training_index"]}_G0_ENV{row["env_id"]}',ground_init=True),protocol)
  v=np.array(row['outgoing_velocity_mps']);speed=float(np.linalg.norm(v))
  checks.append(dict(env_id=row['env_id'],source_training_index=row['source_training_index'],airborne_contact=value['airborne_contact'],conditional_recovery=value['conditional_recovery'],complete=value['complete'],fault=value['fault'],attack_outgoing=row['attack_outgoing'],actual_outgoing_velocity_mps=v.tolist(),outgoing_norm_mps=speed,first_landing_m=row['first_landing_m'],target_x_overshoot_m=max(row['first_landing_m'][0]-4.,0.) if np.isfinite(row['first_landing_m'][0]) else None,recovery_s=row['jump']['recovery_s'],maximum_body_ground_force_n=row['jump']['maximum_body_ground_force_n'],sampled100Hz_arm_speed_peak_rad_s=float(abs(vel[:,row['env_id']])[valid[:,row['env_id']]].max())))
 assert checks[1]['airborne_contact'] and checks[1]['conditional_recovery'] and checks[1]['attack_outgoing'] and not checks[1]['complete']
 report=dict(created_at=datetime.now(timezone.utc).isoformat(),scope=policy['kind'],candidate_id=candidate,optimizer_generation_winner=candidate==36,checks=checks,policy_sha256=hashlib.sha256((out/'policy_evidence.pt').read_bytes()).hexdigest(),native_execution=True,formal_acceptance=False,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [population_path,rows_path,Path(__file__)]})
 (out/'MILESTONE.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
