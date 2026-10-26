"""Validate a completed original128 body-policy evaluation before new learning."""
import hashlib
import json
from pathlib import Path
from quadsmash.control.full_racket_body_residual_v4 import validate_frozen_baseline
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases,summarize_training


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_frozen_body_source(path,checkpoint,evaluation,root,protocol_sha256):
    root,path,evaluation=Path(root),Path(path),Path(evaluation)
    validate_frozen_baseline(checkpoint)
    read=lambda name:json.loads((evaluation/name).read_text())
    inputs,metrics,job=read('INPUTS.json'),read('metrics.json'),read('job.json')
    assert job['status']=='SUCCEEDED' and job['exit_code']==0 and job['receipt_present']
    assert read('completed.json')['env_steps']==job['charged_env_steps']==102400
    assert inputs['protocol_sha256']==metrics['protocol_sha256']==protocol_sha256
    assert inputs['policy_sha256']==metrics['policy_sha256']==sha(path)
    assert inputs['training_indices']==list(range(128)) and inputs['formal_cases_read'] is False
    assert all(value==0 for value in inputs['reset_noise_amplitudes'].values()) and metrics['noise_level']=='none'
    for name,digest in read('source_hashes.json').items():
        if name.startswith('quadsmash/'):assert sha(root/name)==digest,name
    protocol_path=root/'configs/g07_full_racket_eval_v4_r3.json'
    assert sha(protocol_path)==protocol_sha256
    protocol=json.loads(protocol_path.read_text());protocol['_root']=str(root)
    assert sha(root/protocol['asset']['path'])==protocol['asset']['sha256']==checkpoint['asset_sha256']
    for name,digest in protocol['physical_configs'].items():assert sha(root/name)==digest,name
    cases,indices=training_cases(root,protocol)
    summary=summarize_training(read('episodes.json'),cases,indices,protocol)
    assert summary['numerical_failures']==0 and all(metrics[key]==value for key,value in summary.items())
    return dict(source_policy_sha256=sha(path),source_evaluation=str(evaluation),
        source_original_TRAIN128_summary=summary,
        source_evidence_sha256={name:sha(evaluation/name) for name in
            ['INPUTS.json','metrics.json','episodes.json','job.json','completed.json','source_hashes.json']},
        scope='Actual fixed body teacher provenance; new zero residual inherits no task score.',
        both_fixed_source_actors_preserved=True,formal_cases_read=False,formal_acceptance=False)
