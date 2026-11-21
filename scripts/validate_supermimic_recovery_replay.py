#!/usr/bin/env python3
"""Bind a continuous Recovery WM replay to a new same-policy native evaluation before Approach."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from scripts.evaluate_supermimic_fresh_wm_panel import digest


def read(path):
    return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('run','output','anchor','contact-calibration','paired-result'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--previous-panel',type=Path,action='append',required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--learning-rate',type=float,required=True)
    parser.add_argument('--stop-receipt',type=Path,help='Deliberate post-refresh stopping point before the scheduled budget')
    parser.add_argument('--active-flight-batch',type=int,default=0)
    parser.add_argument('--direct-long-body-supervision',action='store_true')
    args=parser.parse_args()
    run=args.run.resolve()
    output=args.output.resolve()
    cfg=read(run/'config.json')
    per=cfg['recovery_update_budget']
    cycles=cfg['cycles']
    paired=read(args.paired_result)
    if args.stop_receipt:
        import torch
        stop=read(args.stop_receipt)
        checkpoint=Path(stop['checkpoint']).resolve(strict=True)
        if checkpoint.parent!=run or digest(checkpoint)!=stop['checkpoint_sha256']:
            raise ValueError('Stopping receipt checkpoint changed')
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
        accepted=ck['global_counters']['accepted']
        cycles,remainder=divmod(accepted,per)
        if (not 0<cycles<=cfg['cycles'] or remainder or cfg['approach_update_budget']!=0
                or checkpoint.name!=f'cycle_{cycles:02d}_recovery_refresh_candidate_{per:04d}.pt'):
            raise ValueError('Stopping receipt is not a completed Recovery refresh boundary')
        del ck
    else:
        completed=read(run/'training_completed.json')
        if completed['recovery_accepted']!=per*cycles or completed['approach_accepted']!=0:
            raise ValueError('Scheduled Recovery run is incomplete')
    if (paired['case_count']!=128 or not paired['identical_initial_states'] or not paired['paired_case_identity_verified']):
        raise ValueError('Recovery budget or fixed native pair is incomplete')
    captures=sorted((read(f)['start_tick'],f) for f in run.glob('recovery_frozen_*/manifest.json'))
    if len(captures)!=cycles:raise ValueError('A complete frozen native refresh is required after each Recovery block')
    original=read(run/'startup.json')['argv']
    def option(key):
        return Path(original[original.index('--'+key)+1])
    triples=[(f,run/f'cycle_{i:02d}_recovery_candidate_{per:04d}.pt',
               run/f'cycle_{i:02d}_recovery_refresh_candidate_{per:04d}.pt') for i,(_,f) in enumerate(captures,1)]
    manifest,pre,checkpoint=triples[-1]
    if read(manifest)['actor_sha256']!=paired['native_sources'][1]['actor_sha256']:
        raise ValueError('Final Recovery policy differs from the case evaluation')
    output.mkdir(parents=True,exist_ok=False)
    repo=Path(__file__).resolve().parents[1]
    def execute(script,log,*parameters):
        with (output/(log+'.log')).open('x') as stream:
            subprocess.run([sys.executable,str(repo/'scripts'/(script+'.py')),*map(str,parameters)],check=True,
                           cwd=repo,env=dict(os.environ,PYTHONPATH='.:scripts',OMP_NUM_THREADS='2'),
                           stdout=stream,stderr=subprocess.STDOUT)
    candidate=output/'wm_replay'
    history=[param for triple in triples[:-1] for param in ('--prior-refresh',*triple)]
    direct=['--direct-long-body-supervision'] if args.direct_long_body_supervision else []
    execute('replay_supermimic_recovery_wm_updates','fit','--run',run,'--checkpoint',pre,
            '--refresh-manifest',manifest,'--post-refresh-checkpoint',checkpoint,*history,
            '--output',candidate,'--learning-rate',args.learning_rate,'--save-optimizer',
            '--natural-flight-batch',16,'--natural-flight-loss','mse','--separate-supervision',
            '--active-flight-batch',args.active_flight_batch,*direct,
            '--device','cuda:0','--cuda-memory-gib',2,'--cpu-threads',3)
    panel=output/'panel.json';capture=output/'native';initial=output/'initial.json';score=output/'candidate.json'
    previous=[arg for f in args.previous_panel for arg in ('--previous-panel',f)]
    execute('declare_supermimic_body_transfer_panel','declare','--actor',checkpoint,'--anchor',args.anchor,
            '--contact-calibration',args.contact_calibration,'--body',option('physical-checkpoint'),
            '--ball-candidate',option('ball-checkpoint'),'--online-candidate',candidate,*previous,
            '--seed',args.seed,'--cases-per-panel',16,'--output',panel)
    common=['--anchor',args.anchor,'--candidate',checkpoint,'--panel',panel,'--archive',option('archive'),
            '--safety-mapping',option('safety-mapping')]
    execute('evaluate_supermimic_fresh_wm_panel','native',*common,'--output',capture,
            '--headless','--device','cuda:0','--kit_args=--renderer/multiGpu/enabled=false')
    execute('rescore_supermimic_fresh_wm_panel','initial',*common,'--capture',capture,'--body-candidate',option('physical-checkpoint'),
            '--ball-candidate',option('ball-checkpoint'),'--initial-physical-components','--output',initial)
    execute('audit_supermimic_online_stage_forward','candidate','--replayed-world',candidate,'--panel',panel,
            '--checkpoint',checkpoint,'--capture',capture,'--initial-score',initial,'--archive',option('archive'),
            '--safety-mapping',option('safety-mapping'),'--compare-components','--output',score)
    result=read(score)
    fit=read(candidate/'fit.json')
    if (result['native_behavior_actor_sha256']!=paired['native_sources'][1]['actor_sha256']
            or result['replay_fit_sha256']!=digest(candidate/'fit.json')
            or fit['optimizer_file_sha256']!=digest(candidate/'online_world_optimizer.pt')):
        raise ValueError('Scored policy or optimizer does not match')
    guard=output/'forward_guard.json'
    passed=result['forward_guard_vs_formal_anchor']['passed']
    device_check=None
    if passed:
        execute('supermimic_online_forward_guard','guard','--reference-score',initial,
                '--behavior-checkpoint',checkpoint,'--panel',panel,'--capture-metrics',capture/'metrics.json','--output',guard)
        try:
            execute('audit_supermimic_recovery_start_gradients','gpu_handoff','--checkpoint',checkpoint,
                    '--archive',option('archive'),'--safety-mapping',option('safety-mapping'),'--device','cuda:0',
                    '--wm-forward-guard',guard,'--resume-wm-replay',candidate,'--output',output/'loaded_recovery.json')
        except subprocess.CalledProcessError:
            device_check=False
        else:
            device_check=read(output/'loaded_recovery.json')['forward_check']['passed']
        passed=bool(device_check)
    summary=dict(passed=passed,cpu_forward_passed=result['forward_guard_vs_formal_anchor']['passed'],
                 gpu_load_and_forward_passed=device_check,body_forward_vs_formal=result['forward_guard_vs_formal_anchor']['body'],
                 body_forward_vs_deployed=result['forward_guard_vs_block_initial']['body'],candidate=str(candidate),checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),
                 guard=str(guard) if passed else None,guard_sha256=digest(guard) if passed else None,
                 score_sha256=digest(score),world_sha256=fit['online_world_sha256'],actor_sha256=result['stage_actor_sha256'],
                 scope='Same-policy prospective WM forward evaluation. Native response and policy promotion are evaluated separately.')
    (output/'decision.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    if not passed:raise SystemExit('Continuous Recovery WM failed prospective forward checks')


if __name__=='__main__':main()
