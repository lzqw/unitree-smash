"""Run one registered 256-update paired-block research arm from the composed WM."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

if __package__:
    from .train_supermimic_physical_block import LEARNER_SEED, NATIVE_TRAIN128_SEED
else:
    from train_supermimic_physical_block import LEARNER_SEED, NATIVE_TRAIN128_SEED


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('source-registration','composition','contact','output','tensorboard'):
        ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--arm',choices=('G0','G1'),required=True)
    a=ap.parse_args()
    src=json.loads(a.source_registration.read_text())
    composition=json.loads(a.composition.with_suffix('.json').read_text())
    if (digest(a.composition)!=composition['checkpoint_sha256'] or digest(a.contact)!=composition['contact_checkpoint_sha256']
        or src['source_checkpoint_sha256']!=composition['source_checkpoint_sha256']
        or src['actor_sha256']!=composition['actor_sha256']):
        raise ValueError('Incompatible initial policy, replay or contact package')
    cmd=src['command'].copy()
    if '--resume-reset-schedule' not in cmd or '--wm-separate-supervision' not in cmd:
        raise ValueError('Expected separate world supervision and a fresh schedule')
    for flag,v in dict(output=a.output/'train',training_profile='research_scale',training_stage='research_contact',
                       contact_bptt_mode='enabled',sequential_layout='short-alternating',cycles=1,
                       resume_start_block='recovery',resume_training=a.composition,action_ball_checkpoint=a.contact,
                       recovery_update_budget=128,approach_update_budget=128,recovery_proposal_budget=256,
                       approach_proposal_budget=256,head_block_size=8,refresh_mode='pair',native_ticks_per_actor_update=128,
                       world_updates_per_actor_update=8,command_delta_budget=.02,target_delta_budget_rad=.08,
                       toi_effective_start_lanes=16,recovery_effective_start_lanes=16,wm_natural_flight_batch=4,
                       wm_active_flight_batch=4,wm_source_quota='recent_online=24,older_online=8,initial_archive=32,action_identification=26,contact_event=38',
                       research_contact_gradient=a.arm,tensorboard_dir=a.tensorboard/a.arm).items():
        flag='--'+flag.replace('_','-')
        if flag in cmd:cmd[cmd.index(flag)+1]=str(v)
        else:cmd.extend([flag,str(v)])
    a.output.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,PYTHONPATH='.:scripts',OMP_NUM_THREADS='2',QUADSMASH_ZSTD_ARCHIVE='1')
    registration=dict(command=cmd,arm=a.arm,initial=composition,expected_retained_updates=256,
                      max_proposals=512,train_case_seed=NATIVE_TRAIN128_SEED,learner_initial_seed=LEARNER_SEED,
                      learner_rng_at_start='restored from composition checkpoint',compression='zstd',
                      previous_registration_sha256=digest(a.source_registration),
                      baseline_path=cmd[cmd.index('--paired-evaluation-baseline')+1])
    (a.output/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    started=time.time()
    repo = Path(__file__).resolve().parents[1]
    with (a.output/'train.log').open('w') as log:
        with subprocess.Popen([sys.executable,*cmd],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT) as trainer:
            with (a.output/'curve_export.log').open('w') as curve_log, (a.output/'resources.log').open('w') as resource_log:
                monitor = subprocess.Popen([sys.executable,'scripts/report_supermimic_research_curve.py',
                    '--arm-dir',str(a.output.resolve()),'--tensorboard',str((a.tensorboard/(a.arm+'_learning')).resolve()),
                    '--follow-pid',str(trainer.pid)],cwd=repo,env=env,stdout=curve_log,stderr=subprocess.STDOUT)
                resources = subprocess.Popen([sys.executable,'scripts/monitor_supermimic_research_gpu.py',
                    '--pid',str(trainer.pid),'--output',str((a.tensorboard/(a.arm+'_resources')).resolve())],
                    cwd=repo,env=env,stdout=resource_log,stderr=subprocess.STDOUT)
                result = trainer.wait()
                resource_result = resources.wait()
                monitor_result = monitor.wait()
    (a.output/'exit.json').write_text(json.dumps(dict(returncode=result,curve_export_returncode=monitor_result,
        resource_monitor_returncode=resource_result,
        wall_seconds=time.time()-started))+'\n')
    return result


if __name__=='__main__':
    sys.exit(main())
