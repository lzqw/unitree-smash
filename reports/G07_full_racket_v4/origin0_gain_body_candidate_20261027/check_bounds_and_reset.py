from pathlib import Path
import hashlib,json,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from quadsmash.control.full_racket_gain_body_adaptation_v4 import GainBodyAdaptationControllerV4
proposal=json.loads((HERE/'PROPOSALS.json').read_text())
source=torch.load(proposal['source_policy'],map_location='cpu',weights_only=False)
run=Path(proposal['source_evaluation'])
with np.load(run/'public_and_actions_first2.npz') as a:
 public=torch.from_numpy(a['observations'][250].copy())
metadata=json.loads(Path('/home/autodrive/LZQW/unitree_smash/reports/G07_full_racket_v4/body_residual_leg15_fresh_n2048_updates48_01/trajectory_traces/manifest.json').read_text())['provenance']
default,limits=[torch.tensor(metadata[k]) for k in ['default_joint_position_rad','joint_limits_rad']]
for candidate in proposal['candidates']:
 c=GainBodyAdaptationControllerV4(source,candidate['gains'],2,'cpu',candidate['body_deltas'])
 c.action({'policy':public},default,limits)
 assert c.reference.planned.all()
 untouched=c.reference.parameters[1].clone()
 c.reset(torch.tensor([0]))
 assert not c.reference.planned[0] and c.reference.planned[1]
 assert torch.equal(c.reference.parameters[0],c.runtime_parameters[0])
 assert torch.equal(c.reference.parameters[1],untouched)
invalid=[{'18':.1},{'2':99.},{'3':float('nan')},{'bogus':1.},{'2':[1.]}]
for value in invalid:
 try:GainBodyAdaptationControllerV4(source,[[1.,1.],[1.,1.]],2,'cpu',value)
 except ValueError:pass
 else:raise AssertionError('Invalid body adaptation accepted')
p=HERE/'BODY_BOUNDS_AND_RESET_CHECK.json';assert not p.exists()
p.write_text(json.dumps(dict(candidates_after_causal_planning_checked=8,partial_reset_preserves_other_lane=True,original_algorithm_bounds_preserved=True,invalid_body_deltas_rejected=5,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),new_native_execution=False,formal_cases_read=False),indent=2)+'\n')
print(p.read_text())
