import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,torch
ROOT=Path.cwd();RUN=ROOT/'reports/G07_full_racket_v4/ball_relative_prepose38_cem25b';OUT=Path(__file__).parent
assert not (OUT/'REPORT.json').exists()
progress=json.loads((RUN/'progress.json').read_text());start=torch.load(RUN/'generation_start_000.pt',map_location='cpu',weights_only=False)
assert progress['generation']==0 and progress['control_ticks']==100 and progress['first_episodes_complete']==0
assert start['parameter_names'][-1]=='contact_normal_downward_pitch_bias_rad' and start['search_coordinate_indices']==[7,14,18,37]
assert start['controller_template']['seed']['relative_contact_prepose']['frame']=='robot_root_body'
assert start['controller_template']['seed']['relative_contact_prepose']['provenance']['reference_env_id']==109
bank=ROOT/'reports/G07_full_racket_v4/relative_prepose38_ready04_1615/CANDIDATE_BANK.json';assert hashlib.sha256(bank.read_bytes()).hexdigest()==start['candidate_bank_sha256']
with np.load(RUN/'partial_public_and_poses.npz') as d:
 assert d['actions'].shape==(100,2,18) and np.isfinite(d['actions']).all() and abs(d['actions']).max()<=1
with np.load(RUN/'all_first_episodes.partial.npz') as d:
 assert d['link_poses'].shape==(50,128,28,7) and np.isfinite(d['link_poses']).all()
with np.load(RUN/'arm_targets.partial.npz') as d:
 assert d['targets_before_slew_rad'].shape==(100,128,6) and np.isfinite(d['targets_after_slew_rad']).all()
report=dict(verified_at=datetime.now(timezone.utc).isoformat(),scope='CEM25b first100 native ticks before ball contact. Relative public-forecast preparation is present in the actual runtime template; no performance or formal result is proven.',generation=0,control_ticks=100,actual_population_shape=[64,38],candidate_bank_rows=60,search_coordinates=[7,14,18,37],relative_prepose_reference_env_id=109,relative_prepose_gain=float(start['controller_template']['seed']['causal_arm_prepose_gain']),relative_prepose_reference_body_m=start['controller_template']['seed']['relative_contact_prepose']['reference_target_body_m'],all_actions_finite_bounded=True,all_native_streams_finite=True,first_episodes_ended=0,contact_events=0,software_arm_target_rate_rad_s=5.,native_execution=True,formal_acceptance=False,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [RUN/'generation_start_000.pt',RUN/'partial_public_and_poses.npz',RUN/'all_first_episodes.partial.npz',RUN/'arm_targets.partial.npz',bank,Path(__file__)]})
(OUT/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
