#!/usr/bin/env python3
"""Build an offline case-conditioned expert-bank selector manifest.

The manifest records which measured native policy succeeded on each fixed TRAIN
case and the first public ball feature available after launch. It is an
analysis/selector input, not a success-rate claim or a runtime oracle.
"""
import argparse, hashlib, json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from quadsmash.rl.episode_trace_v4 import unpack_public

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--source-index',type=Path,required=True);p.add_argument('--baseline-evaluation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists(): raise ValueError('fresh output required')
 idx=json.loads(a.source_index.read_text())
 candidates=sorted(idx['sources'],key=lambda s:(-s['complete5'],s['evaluation']))
 by_case={}
 for source in candidates:
  policy=Path(source['policy'])
  if not policy.exists() or sha(policy)!=source['policy_sha256']: continue
  payload=torch.load(policy,map_location='cpu',weights_only=False)
  if payload.get('schema')!='g07-shared-target-residual-ppo-v1' or not payload.get('native_execution'):
   continue
  for case in source['complete_ids']:
   by_case.setdefault(int(case),source)
 # Read the real first-episode trace used by the current 27/128 candidate.
 manifest=json.loads((a.baseline_evaluation/'fixed_policy_traces/manifest.json').read_text())
 features={}
 for entry in manifest['episodes']:
  case=int(entry['source_training_index']);path=a.baseline_evaluation/'fixed_policy_traces'/entry['data_file']
  if sha(path)!=entry['data_sha256']: raise ValueError('trace changed')
  with np.load(path,allow_pickle=False) as z: public=unpack_public(z['packed_public'])
  present=np.flatnonzero(public[:,67]>.5)
  if not len(present): continue
  row=public[int(present[0])]
  # First causal ball/goal feature: current relative ball position/velocity,
  # remaining goal command, and phase clock. No future outcome is included.
  feature=np.concatenate([row[61:67],row[759:770],[row[775]]]).astype(np.float32)
  features[case]=dict(feature=feature.tolist(),first_ball_present_sample=int(present[0]))
 if set(by_case)-set(features): raise ValueError('Missing public ball feature for successful cases')
 cases=[]
 for case in sorted(features):
  source=by_case.get(case)
  cases.append(dict(training_index=case,feature=features[case]['feature'],
      first_ball_present_sample=features[case]['first_ball_present_sample'],
      expert_policy=str(source['policy']) if source else None,
      expert_policy_sha256=source['policy_sha256'] if source else None,
      expert_evaluation=source['evaluation'] if source else None,
      expert_complete5=source['complete5'] if source else None,
      selector_label_available=source is not None))
 result=dict(schema='case-conditioned-native-expert-bank-v1',source_index_sha256=sha(a.source_index),
     baseline_evaluation=str(a.baseline_evaluation.resolve()),baseline_policy_sha256=manifest['provenance']['policy_sha256'],
     cases=cases,covered_cases=sum(c['selector_label_available'] for c in cases),total_cases=128,
     selector_training_cases=len(cases),runtime_selection_not_evaluated=True,
     oracle_union_not_a_policy_success_rate=True,formal_acceptance=False,
     scope='Fixed TRAIN128 native measured-policy case coverage; feature is causal first-ball observation. '
           'This manifest supports an expert-bank selector analysis and does not claim generalized autonomous performance.')
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'covered_cases':result['covered_cases'],'selector_training_cases':len(cases),'total_cases':128,'sha256':sha(a.output)}))
if __name__=='__main__': main()
