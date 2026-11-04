#!/usr/bin/env python3
"""Distill causal expert-bank residuals into one public-observation head."""
import argparse, hashlib, io, json
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from quadsmash.rl.episode_trace_v4 import unpack_public

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(794,128),nn.ELU(),nn.Linear(128,64),nn.ELU(),nn.Linear(64,18))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x): return self.net(x)

class Distilled(nn.Module):
    def __init__(self,candidate,mean,std,head):
        super().__init__();self.candidate=candidate;self.head=head
        self.register_buffer('mean',mean);self.register_buffer('std',std)
    def forward(self,raw):
        base=self.candidate(raw)
        delta=self.head((raw-self.mean)/self.std)
        return base+raw[:,67:68].clamp(0.,1.)*delta

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def collect(root,trace_manifest,metrics,bank_manifest,experts,candidate):
    m=json.loads(Path(trace_manifest).read_text());met=json.loads(Path(metrics).read_text())
    bank=json.loads(Path(bank_manifest).read_text())
    selected=met['expert_bank_selection']['selected_case_ids']
    by_case={x['training_index']:x for x in bank['cases']}
    xs=[];ys=[]
    for entry,case in zip(m['episodes'],selected):
        path=Path(trace_manifest).parent/entry['data_file']
        with np.load(path,allow_pickle=False) as z: raw=unpack_public(z['packed_public']).astype(np.float32)
        raw_t=torch.from_numpy(raw)
        with torch.inference_mode(): base=candidate(raw_t).numpy()
        expert_path=by_case.get(int(case),{}).get('expert_policy')
        if expert_path:
            if expert_path not in experts:
                payload=torch.load(expert_path,map_location='cpu',weights_only=False)
                experts[expert_path]=torch.jit.load(io.BytesIO(payload['residual_actor_jit']),map_location='cpu').eval()
            with torch.inference_mode(): target=experts[expert_path](raw_t).numpy()
            ball=raw[:,67:68].clip(0.,1.)
            y=base+ball*(target-base)
        else: y=base
        xs.append(raw);ys.append(y-base)
    return np.concatenate(xs),np.concatenate(ys)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--candidate-policy',type=Path,required=True)
    p.add_argument('--bank-manifest',type=Path,required=True)
    p.add_argument('--train-evaluation',type=Path,required=True)
    p.add_argument('--validation-evaluation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=1800)
    p.add_argument('--seed',type=int,default=882130)
    a=p.parse_args()
    if a.output.exists(): raise ValueError('fresh output required')
    torch.manual_seed(a.seed);torch.set_num_threads(1)
    candidate_payload=torch.load(a.candidate_policy,map_location='cpu',weights_only=False)
    candidate=torch.jit.load(io.BytesIO(candidate_payload['residual_actor_jit']),map_location='cpu').eval()
    state=candidate.state_dict();mean=state['normalizer._mean'].float();std=state['normalizer._std'].float().clamp_min(1e-6)
    experts={}
    train_x,train_y=collect(a.train_evaluation,a.train_evaluation/'fixed_policy_traces/manifest.json',a.train_evaluation/'metrics.json',a.bank_manifest,experts,candidate)
    val_x,val_y=collect(a.validation_evaluation,a.validation_evaluation/'fixed_policy_traces/manifest.json',a.validation_evaluation/'metrics.json',a.bank_manifest,experts,candidate)
    x=torch.from_numpy(train_x);y=torch.from_numpy(train_y);vx=torch.from_numpy(val_x);vy=torch.from_numpy(val_y)
    z=(x-mean)/std;vz=(vx-mean)/std
    head=Head();opt=torch.optim.Adam(head.parameters(),lr=2e-4);hist=[];g=torch.Generator().manual_seed(a.seed)
    for u in range(1,a.updates+1):
        ids=torch.randint(len(z),(2048,),generator=g);pred=head(z[ids]);loss=(pred-y[ids]).square().mean()
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step()
        if u%300==0 or u==a.updates:
            with torch.inference_mode():
                tr=(head(z)-y);va=(head(vz)-vy)
                rec=dict(update=u,train_rmse=float(tr.square().mean().sqrt()),val_rmse=float(va.square().mean().sqrt()),val_p95=float(va.abs().flatten().quantile(.95)))
                hist.append(rec);print(json.dumps(rec),flush=True)
    wrapper=Distilled(candidate,mean,std,head).eval();scripted=torch.jit.script(wrapper)
    a.output.mkdir(parents=True);blob=io.BytesIO();torch.jit.save(scripted,blob)
    result=dict(candidate_payload);result['residual_actor_jit']=blob.getvalue();result['native_execution']=False;result['performance_inherited']=False;result['formal_acceptance']=False
    meta=dict(result.get('residual_training_metadata',{}),offline_trace_finetune=True,case_expert_distillation=True,
              distillation_updates=a.updates,distillation_train_samples=len(x),distillation_validation_samples=len(vx),
              distillation_manifest_sha256=sha(a.bank_manifest),distilled_actor_jit_sha256=hashlib.sha256(blob.getvalue()).hexdigest())
    result['residual_training_metadata']=meta
    torch.save(result,a.output/'policy.pt')
    (a.output/'RESULT.json').write_text(json.dumps(dict(schema='offline-case-expert-distillation-v1',candidate_policy_sha256=sha(a.candidate_policy),bank_manifest_sha256=sha(a.bank_manifest),train_samples=len(x),validation_samples=len(vx),history=hist,native_execution=False,closed_loop_success_tested=False,policy_sha256=sha(a.output/'policy.pt')),indent=2)+'\n')
    print(json.dumps(dict(completed=True,train_samples=len(x),validation_samples=len(vx),policy_sha256=sha(a.output/'policy.pt'))))
if __name__=='__main__': main()
