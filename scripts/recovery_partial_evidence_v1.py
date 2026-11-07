"""Save diagnostic prefixes without weakening complete-run admission.

Only records earlier than a completed env.step count are marked executed.
Plans may include a pending/unexecuted query. No checkpoint restores PhysX.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


NUMERIC=('q','v','contacts','foot_force_world_n','student_action','final_action',
         'previous_target_rad','requested_target_rad','final_target_rad')


def plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    return value


def immutable_json(path,value):
    raw=(json.dumps(plain(value),sort_keys=True,allow_nan=False)+'\n').encode()
    if path.exists():
        if path.read_bytes()!=raw:raise ValueError('Published diagnostic mutated: '+str(path))
    else:
        temp=path.with_suffix('.tmp');temp.write_bytes(raw);temp.replace(path)
    return hashlib.sha256(raw).hexdigest()


def save_partial_recovery(replay,completed_control_ticks):
    if type(completed_control_ticks) is not int or not 0<=completed_control_ticks<=800:
        raise ValueError('Actual completed env.step count required')
    records=replay.records
    ticks=[r['control_tick'] for r in records]
    if ticks!=list(range(replay.start,replay.start+len(records))):
        raise ValueError('Noncontiguous recorded control requests')
    executed=[r for r in records if r['control_tick']<completed_control_ticks]
    if not records and not getattr(replay,'plans',[]):return None
    output=Path(replay.output)/'recovery_partial'
    output.mkdir(exist_ok=True)
    plans=[]
    for plan in getattr(replay,'plans',[]):
        path=output/f"plan_{plan['tick']:06d}.json"
        plans.append(dict(tick=plan['tick'],file=path.name,sha256=immutable_json(path,plan),
                          any_step_completed_after_query=plan['tick']<completed_control_ticks))
    data={key:np.stack([r[key] for r in executed]) for key in NUMERIC} if executed else {}
    data['control_tick']=np.asarray([r['control_tick'] for r in executed],dtype=np.int64)
    path=output/f'trace_{completed_control_ticks:06d}.npz'
    if path.exists():
        with np.load(path,allow_pickle=False) as prior:
            if set(prior.files)!=set(data) or any(not np.array_equal(prior[k],v) for k,v in data.items()):
                raise ValueError('Published executed prefix changed')
    else:
        temp=path.with_suffix('.tmp')
        with temp.open('wb') as f:np.savez_compressed(f,**data)
        temp.replace(path)
    report=dict(schema='diagnostic_recovery_prefix_v1',case=replay.case,start_tick=replay.start,
        requested_sequence_steps=replay.steps,completed_control_ticks=completed_control_ticks,
        executed_recovery_records=len(executed),pending_request_records=len(records)-len(executed),
        records=[{k:v for k,v in r.items() if k not in NUMERIC} for r in executed],
        planning_records=plans,trace_file=path.name,trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        query_match=replay.query_check,full_run_admission=False,valid_expert_labels=False,
        physical_state_resume_supported=False,
        scope='Completed-step prefix only; model acceptance is not native recovery success')
    revision=hashlib.sha256((json.dumps(plain(report),sort_keys=True,allow_nan=False)+'\n').encode()).hexdigest()[:12]
    report_path=output/f'prefix_{completed_control_ticks:06d}_{revision}.json'
    digest=immutable_json(report_path,report)
    latest=dict(file=report_path.name,sha256=digest,full_run_admission=False,valid_expert_labels=False)
    temp=output/'latest.tmp';temp.write_text(json.dumps(latest,indent=2)+'\n');temp.replace(output/'latest.json')
    return report
