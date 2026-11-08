"""Reproduce each actually executed feedback layer on recorded native states.

This is command attribution, not a counterfactual rollout: an upstream command
increment may be clipped or cancelled by later layers. Recorded future states
are never presented as an online reference or certified expert demonstration.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pinocchio as pin
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.native_timed_experimental_recovery_v1 import NativeTimedExperimentalRecoveryV1
from scripts.loaded_pinocchio_model_v1 import PublicState
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['run','bundle','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh output required')
    read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    job=read(a.run/'job.json');m=read(a.bundle/'manifest.json');execution=read(a.run/'sequence_native_result.json')
    if job['status']!='SUCCEEDED' or job['exit_code']!=0 or not execution['query_match']['passed']:
        raise ValueError('Completed native execution with actual matching query required')
    adapter=NativeTimedExperimentalRecoveryV1(a.bundle,m['case'],
        torch.tensor(m['configuration']['interface']['limits']),m['policy_sha256'],a.output.parent)
    model=adapter.model
    with np.load(a.run/'sequence_native_trace.npz',allow_pickle=False) as z:trace={k:z[k].copy() for k in z.files}
    ticks=np.array([r['tick'] for r in execution['records']],int)
    if execution['case']!=m['case'] or not np.array_equal(ticks,np.arange(280,280+len(ticks))):
        raise ValueError('Continuous original episode required')
    nodes=[];node=adapter.feedback
    while node is not None:
        nodes.append(node);node=getattr(node,'base',None)
    captured={}
    def wrapped(original,index,node):
        def call(*args,**kwargs):
            target,mode=original(*args,**kwargs)
            captured[index]=dict(target=np.asarray(target).copy(),mode=mode,
                diagnostic=copy.deepcopy(getattr(node,'last_diagnostic',None)))
            return target,mode
        return call
    for i,node in enumerate(nodes):node.action=wrapped(node.action,i,node)
    records=[];largest=0.
    for i,tick in enumerate(ticks):
        q=trace['q'][i].copy();v=trace['v'][i].copy();previous=trace['previous_target_rad'][i].copy()
        state=PublicState(q,v,pin.Quaternion(q[3:7]).matrix(),q[:3].copy(),
            q[model.q_indices].copy(),v[model.v_indices].copy(),trace['contacts'][i].copy(),0.)
        state.previous_command_target=previous;captured.clear()
        target,mode=adapter.feedback.action(80+i,state,previous)
        error=float(np.max(abs(target-trace['requested_target_rad'][i])));largest=max(largest,error)
        if error>1e-10:raise ValueError('Layer replay does not reproduce executed request')
        layers=[]
        for index,item in sorted(captured.items()):
            row=dict(layer=type(nodes[index]).__name__,mode=item['mode'],diagnostic=item['diagnostic'])
            if index+1 in captured:
                delta=item['target']-captured[index+1]['target']
                row.update(increment_rad=delta.tolist(),max_abs_increment_rad=float(np.max(abs(delta))))
            layers.append(row)
        records.append(dict(control_tick=int(tick),contacts=state.contacts.tolist(),layers=layers))
    summaries=[]
    for node in nodes:
        name=type(node).__name__;rows=[x for r in records for x in r['layers'] if x['layer']==name]
        active=[r for r in rows if r['diagnostic'] and r['diagnostic'].get('active')]
        summary=dict(layer=name,recorded_steps=len(rows),diagnostic_active_steps=len(active),
            maximum_increment_rad=max([r.get('max_abs_increment_rad',0.) for r in rows],default=0.))
        for key in ['position_error_world_m','attitude_error_world_rad','error_world_rad']:
            values=[np.linalg.norm(r['diagnostic'][key]) for r in active if key in r['diagnostic']]
            if values:summary[key]=dict(median=float(np.median(values)),maximum=float(max(values)))
        if hasattr(node,'max_joint_delta_rad'):summary['configured_bound_rad']=node.max_joint_delta_rad
        # Both kinematic correction classes expose their effective limits differently.
        bound=float(node.limits[2]) if name=='LandingReferenceTrackingV1' else getattr(node,'maximum_joint_delta_rad',None)
        if bound is not None:
            summary['configured_bound_rad']=bound
            summary['steps_reaching_bound']=sum(r.get('max_abs_increment_rad',0.)>=bound-1e-9 for r in rows)
        summaries.append(summary)
    paths=[Path(__file__),a.run/'job.json',a.run/'sequence_native_result.json',a.run/'sequence_native_trace.npz',a.bundle/'manifest.json']
    result=dict(scope=__doc__,case=m['case'],passed_exact_command_replay=True,
        maximum_command_replay_error_rad=largest,steps=len(records),layers=summaries,records=records,
        causal_effect_of_each_layer_established=False,valid_expert_labels=False,
        source_sha256={**m['source_sha256'],**{str(f.resolve()):sha(f) for f in paths}})
    tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(plain(result),indent=2,allow_nan=False)+'\n');tmp.replace(a.output)
    print(json.dumps({k:v for k,v in result.items() if k not in ['records','source_sha256']}))


if __name__=='__main__':main()
