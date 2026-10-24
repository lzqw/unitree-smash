import copy,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,torch
ROOT=Path.cwd();sys.path.insert(0,str(ROOT));from quadsmash.evaluation_v4 import independent_episode
RUN=ROOT/'reports/G07_full_racket_v4/contact_lead_gain_normal_candidate43_cem28';OUT=Path(__file__).parent
state=torch.load(RUN/'population_003.pt',map_location='cpu',weights_only=False);rows=json.loads((RUN/'episodes_003.json').read_text());protocol=json.loads((ROOT/'configs/g07_full_racket_eval_v4_r3.json').read_text());protocol['_root']=str(ROOT)
with np.load(RUN/'arm_targets_003.npz') as d:vel=d['observed_velocity_rad_s'];valid=d['valid_first_episode']
candidate=39;group=[r for r in rows if r['candidate_id']==candidate];assert len(group)==2;out=OUT/f'candidate{candidate}';out.mkdir(exist_ok=True)
policy=copy.deepcopy(state['controller_template'])
for key in ['mean','sigma','rng','generations','scores','rows','best_parameters','best_result','reference_parameters']:policy.pop(key,None)
policy.update(schema='full-racket-ball-cem-v4',parameters=state['parameters'][candidate].clone(),parameter_names=state['parameter_names'],training_suite=state['training_suite'],training_case_indices=state['training_case_indices'],training_result=group,native_execution=True,performance_inherited=False,formal_acceptance=False,selected_source_run=RUN.name,selected_generation=3,selected_candidate=candidate,kind='Actually executed CEM28 generation3 second complete-smash candidate. Native TRAIN65 complete flag true; paired TRAIN1 failure preserved. No formal panel pass.')
torch.save(policy,out/'policy_evidence.pt');checks=[]
for row in group:
 (out/f'NATIVE_EPISODE_{row["env_id"]}.json').write_text(json.dumps(row,indent=2)+'\n');v=independent_episode(dict(row,case_id=f'TRAIN{row["source_training_index"]}_G3_ENV{row["env_id"]}',ground_init=True),protocol);o=np.array(row['outgoing_velocity_mps']);checks.append(dict(env_id=row['env_id'],source_training_index=row['source_training_index'],airborne_contact=v['airborne_contact'],conditional_recovery=v['conditional_recovery'],complete=v['complete'],fault=v['fault'],attack_outgoing=row['attack_outgoing'],actual_outgoing_velocity_mps=o.tolist(),outgoing_norm_mps=float(np.linalg.norm(o)),first_landing_m=row['first_landing_m'],landed_target=row['landed_target'],recovery_s=row['jump']['recovery_s'],maximum_body_ground_force_n=row['jump']['maximum_body_ground_force_n'],sampled100Hz_arm_speed_peak_rad_s=float(abs(vel[:,row['env_id']])[valid[:,row['env_id']]].max())))
assert checks[1]['complete'] is True and checks[0]['complete'] is False
report=dict(created_at=datetime.now(timezone.utc).isoformat(),scope=policy['kind'],candidate_id=candidate,checks=checks,policy_sha256=hashlib.sha256((out/'policy_evidence.pt').read_bytes()).hexdigest(),native_execution=True,formal_acceptance=False,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [RUN/'population_003.pt',RUN/'episodes_003.json',Path(__file__)]});(out/'MILESTONE.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
