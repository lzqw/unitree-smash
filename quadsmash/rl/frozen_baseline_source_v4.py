"""Read-only completed native source receipts for a fixed public-policy base."""
import hashlib
import json
from pathlib import Path

from quadsmash.control.full_racket_residual_v4 import validate_frozen_baseline


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen_baseline_source(path,checkpoint,protocol_sha256):
    path=Path(path);run=path.parent
    validate_frozen_baseline(checkpoint)
    read=lambda name:json.loads((run/name).read_text())
    job,receipt,inputs,metrics=[read(name)for name in ['job.json','completed.json','INPUTS.json','metrics.json']]
    if job['status']!='SUCCEEDED' or job['exit_code']!=0 or not job['receipt_present']:
        raise ValueError('Frozen learned base must come from a completed native job')
    if not 0<receipt['env_steps']==job['charged_env_steps']:
        raise ValueError('Frozen base native step receipts disagree')
    measured=checkpoint.get('residual_training_metadata',{}).get('actual_control_env_steps',
        checkpoint.get('native_trial_execution',{}).get('actual_control_env_steps'))
    if measured!=receipt['env_steps']:raise ValueError('Frozen actor metadata differs from its source job steps')
    source_protocol=inputs.get('contract',{}).get('protocol_sha256',inputs.get('protocol_sha256'))
    if source_protocol!=protocol_sha256:raise ValueError('Frozen base belongs to a different original protocol')
    policy_sha=sha(path)
    if metrics['policy_sha256']!=policy_sha:
        trial=checkpoint.get('native_trial_execution',{})
        if (trial.get('proposal_sha256')!=metrics['policy_sha256']
                or trial.get('metrics_sha256')!=sha(run/'metrics.json')
                or trial.get('episodes_sha256')!=sha(run/'episodes.json')
                or trial.get('actual_control_env_steps')!=receipt['env_steps']):
            raise ValueError('Executed feedback base lacks matching original native records')
    return dict(source_policy_sha256=policy_sha,source_run=str(run),
        source_job_sha256=sha(run/'job.json'),source_metrics_sha256=sha(run/'metrics.json'),
        source_completed_control_env_steps=receipt['env_steps'],protocol_sha256=protocol_sha256,
        scope='Completed native source provenance only; new centered learner inherits no performance.')
