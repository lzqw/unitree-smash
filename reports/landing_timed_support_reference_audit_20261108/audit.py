"""Descriptive model-trajectory support/reference comparison, not native proof."""
import hashlib,json
from pathlib import Path
import sys
import numpy as np
import pinocchio as pin
ROOT=Path('/home/autodrive/LZQW/unitree_smash_contact_recovery');sys.path.insert(0,str(ROOT))
from scripts.native_timed_experimental_recovery_v1 import NativeTimedExperimentalRecoveryV1
from scripts.loaded_pinocchio_model_v1 import PublicState
from scripts.timed_whole_body_recovery_v1 import build_timed_plan
from scripts.landing_reference_tracking_v1 import LandingReferenceTrackingV1
from scripts.landing_support_mpc_v2 import _support_margin
from scripts.search_preserved_early_support_v1 import joint_factory
from functools import partial
import torch

out=Path(__file__).parent;result_path=out/'result.json'
if result_path.exists():raise ValueError('Fresh result required')
generation=ROOT/'reports/landing_timed_support85_20261108/generation_00.json'
bundle=ROOT/'reports/landing_native_timed85_preparation_20261108/bundle'
m=json.loads((bundle/'manifest.json').read_text());g=json.loads(generation.read_text())
selected=min(g['fine'],key=lambda r:r['result']['rank'])
adapter=NativeTimedExperimentalRecoveryV1(bundle,85,torch.tensor(m['configuration']['interface']['limits']),m['policy_sha256'],out)
with np.load(bundle/m['bundle_name'],allow_pickle=False) as z:
 rq=z['tracking_reference_q'].copy();rv=z['tracking_reference_v'].copy()
worker=dict(controller=adapter.controller,initial=adapter.initial,factory=partial(joint_factory,reference_q=rq,reference_v=rv,gains=m['tracking_gains']))
policy=build_timed_plan(worker,selected['plan']);tracker=policy
while not isinstance(tracker,LandingReferenceTrackingV1):tracker=tracker.base
q=np.asarray(m['query']['q']);v=np.asarray(m['query']['v']);contacts=np.asarray(m['query']['contacts'],bool);previous=np.asarray(m['query']['target']);rows=[]
for step,row in enumerate(selected['result']['trajectory']):
 R=pin.Quaternion(q[3:7]).matrix();state=PublicState(q,v,R,q[:3].copy(),q[adapter.model.q_indices].copy(),v[adapter.model.v_indices].copy(),contacts,0.)
 state.previous_command_target=previous.copy()
 target,_=policy.action(80+step,state,previous)
 if np.max(abs(target-row['target']))>1e-12:raise ValueError('Reference command reconstruction differs')
 snapshot,feet,active=adapter.controller._support_geometry(state)
 com=snapshot['com_world'];margin=float(_support_margin(com,active))
 record=dict(tick=280+step,contacts=int(contacts.sum()),com=com.tolist(),com_velocity=snapshot['com_velocity_world'].tolist(),
  support_margin_m=margin if np.isfinite(margin) else None,tracking_active=tracker.last_diagnostic['active'])
 if tracker.last_diagnostic['active']:
  translation=R@np.asarray(tracker.last_diagnostic['requested_virtual_root_delta_local'])[:3]
  virtual=float(_support_margin(com+translation,active))
  record.update(virtual_translation_world=translation.tolist(),virtual_support_margin_m=virtual if np.isfinite(virtual) else None,
   root_reference_error_world=tracker.last_diagnostic['position_error_world_m'],active_feet=active.tolist())
 rows.append(record)
 q=np.asarray(row['q']);v=np.asarray(row['v']);contacts=np.asarray(row['contacts'],bool);previous=np.asarray(row['target'])
eligible=[r for r in rows if r['contacts']>=3 and r['tracking_active'] and r['support_margin_m'] is not None]
worse=[r for r in eligible if r['virtual_support_margin_m']<r['support_margin_m']-1e-6]
outside=[r for r in eligible if r['support_margin_m']<0]
paths=[generation,bundle/'manifest.json',Path(__file__)]
result=dict(scope=__doc__,rows=rows,support_reference_steps=len(eligible),virtual_margin_reduced_steps=len(worse),
 outside_support_steps=len(outside),outside_and_virtual_worse=sum(r['support_margin_m']<0 for r in worse),
 command_reconstruction_passed=True,native_execution=False,valid_expert_labels=False,
 limitations=['Virtual root translation is not the actual COM displacement produced by the leg commands.',
 'Current model contact support and fixed reference alone do not prove a cause of failure.'],
 source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
result_path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:result[k] for k in ['support_reference_steps','virtual_margin_reduced_steps','outside_support_steps','outside_and_virtual_worse']}))
print(json.dumps(rows[-1]))
