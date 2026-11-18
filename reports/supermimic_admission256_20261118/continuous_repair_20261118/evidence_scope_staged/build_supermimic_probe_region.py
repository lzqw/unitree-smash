#!/usr/bin/env python3
"""Bundle FOUR actual same-prefix native diagnostics, never synthesize PhysX responses.

This constructs immutable evidence, not an automatic passing receipt. The quality
receipt must declare independently calibrated bounds and reference this file hash.
"""
import argparse,hashlib,json
from pathlib import Path
import torch
from quadsmash.rl.supermimic_native_pair_capture_v1 import BRANCHES,tree_digest,probe_lanes,validate_spec
from quadsmash.rl.supermimic_physical_dataset_v1 import encode_physical
from quadsmash.rl.supermimic_physical_loop_loading_v1 import tensor_digest

def build(paths,identity,output,selected_lanes=None):
    runs={k:torch.load(paths[k],map_location='cpu',weights_only=True) for k in BRANCHES}
    base=runs['nominal_a']
    for branch,run in runs.items():
        if run.get('schema')!='native-paired-prefix-run-v1' or run.get('branch')!=branch:raise ValueError('Not an actual branch record')
        for key in ('spec','seed','task_contract','policy_execution','physics','prefix_sha256','executor_before_sha256','execution_code_sha256'):
            if run.get(key)!=base.get(key):raise ValueError('Native paired-prefix mismatch: '+key)
        if any(run.get(k)!=0 for k in ('actor_updates','world_updates','critic_updates')):raise ValueError('Probe weights were updated')
        if tree_digest(run['actor_state'])!=tree_digest(base['actor_state']):raise ValueError('Different native actors')
    from quadsmash.rl.supermimic_evidence_binding_v1 import native_code_compatibility
    code_scope=native_code_compatibility(base['execution_code_sha256'],identity['execution_code_sha256'])
    if not code_scope['compatible']:raise ValueError('Runtime/probe native execution code differs: '+str(code_scope))
    if tensor_digest(base['actor_state'].items())!=identity['actor_state_sha256']:raise ValueError('Runtime identity actor mismatch')
    from quadsmash.rl.supermimic_evidence_binding_v1 import json_sha
    if json_sha(base['task_contract'])!=identity['task_contract_sha256'] or base['policy_execution']!=identity['policy_execution']:
        raise ValueError('Probe/runtime task or execution mismatch')
    validate_spec(base['spec'])
    row=base['raw']['start'];physical=row['before']['physical'];intervention_lanes=probe_lanes(base['spec'])
    lanes=intervention_lanes if selected_lanes is None else list(selected_lanes)
    if not lanes or any(type(i)is not int or i not in intervention_lanes for i in lanes) or len(set(lanes))!=len(lanes):
        raise ValueError('Evidence lanes must be unique measured intervention lanes')
    sl=torch.tensor(lanes)
    for run in runs.values():
        if tree_digest(run['raw']['start'])!=tree_digest(row):raise ValueError('Different full native start state')
    suite=base['task_contract']['suite'];path=Path(suite['path'])
    if not path.is_absolute():path=Path(__file__).resolve().parents[1]/path
    if hashlib.sha256(path.read_bytes()).hexdigest()!=suite['sha256']:raise ValueError('Suite hash mismatch')
    cases=json.loads(path.read_text())['cases'];launch=torch.tensor([c['launch']['position_m']+c['launch']['velocity_mps'] for c in cases])
    pairs=dict(state=encode_physical(physical)[sl],previous_action=physical['final_action'][sl],
        feet=physical['foot_contact'][sl].float(),present=physical['ball_present'][sl],elapsed=physical['elapsed_s'][sl],launch=launch[sl],
        restore_mode=base['restore_mode'],prefix_sha256={k:v['prefix_sha256'] for k,v in runs.items()},
        executor_before_sha256={k:v['executor_before_sha256'] for k,v in runs.items()})
    for branch,run in runs.items():pairs['native_'+branch]=run['raw']['next_state'][sl]
    for branch in ('minus','plus'):pairs['targets_'+branch]=runs[branch]['raw']['final_joint_targets'][sl]
    def modes(run,lane):
        task=run['raw']['task_after']
        result={k:(v[lane].tolist() if isinstance(v,torch.Tensor) and v.ndim else v) for k,v in task.items()
                if k in ('had_contact','released','frame_hit','legal_initial_contact','string_hit','airborne_string_hit','rally_fault')}
        if 'next_feet' in run['raw']:result['foot_contact']=run['raw']['next_feet'][lane].bool().tolist()
        recovery=run['raw'].get('recovery_after',{})
        if 'body_ground_seen' in recovery:result['body_ground_seen']=bool(recovery['body_ground_seen'][lane])
        return result
    pairs['native_contact_modes']={k:([modes(v,lane) for lane in lanes] if len(lanes)>1 else modes(v,lanes[0])) for k,v in runs.items()}
    sample_ids=[f"{identity['actor_state_sha256']}:{base['seed']}:{lane}:{int(row['before']['episode_index'][lane])}" for lane in lanes]
    evidence=dict(sample_ids=sample_ids,lane_ids=lanes,intervention_lanes=intervention_lanes,schema='native-paired-prefix-region-v1',runtime_identity=identity,actor_state=base['actor_state'],pairs=pairs,
        command_starts=[dict(start=row,final_normalized=base['raw']['final_normalized'],final_joint_targets=base['raw']['final_joint_targets'],
                            valid_lanes=base['raw'].get('command_valid_lanes',torch.ones(len(launch),dtype=torch.bool)))],
        source_sha256={k:hashlib.sha256(Path(p).read_bytes()).hexdigest() for k,p in paths.items()},
        source_paths={k:str(Path(p).resolve()) for k,p in paths.items()},
        native_recorded_code_sha256=base['execution_code_sha256'],runtime_model_code_scope=code_scope,
        head=base['spec']['head'],scope='measured same-tick native probes; no automatic pass or calibrated validity bounds')
    output=Path(output)
    with output.open('xb') as f:torch.save(evidence,f)
    return dict(path=str(output),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),native_probe_count=len(lanes),passed=None)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for branch in BRANCHES:p.add_argument('--'+branch.replace('_','-'),type=Path,required=True)
    p.add_argument('--runtime-identity',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--selected-lanes',type=int,nargs='+',help='Predeclared calibration OR heldout subset, never selected by measured response')
    a=p.parse_args();print(json.dumps(build({k:getattr(a,k) for k in BRANCHES},json.loads(a.runtime_identity.read_text()),a.output,a.selected_lanes),indent=2))
