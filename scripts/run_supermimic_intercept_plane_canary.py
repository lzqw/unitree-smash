"""Run one detached-contact Approach proposal from a verified frozen policy and evaluate the same 128 cases."""
import argparse
import json
import os
import time
from pathlib import Path
import subprocess
import sys

import torch

from quadsmash.rl.supermimic_physical_loop_loading_v1 import tensor_digest
from quadsmash.rl.supermimic_toi_racket_objective import calibrated_plane_offset
from scripts.compare_supermimic_fixed_stages import load_run
from scripts.evaluate_supermimic_fresh_wm_panel import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('parent','evaluation','panel','output'):
        parser.add_argument('--'+key,required=True,type=Path)
    parser.add_argument('--after-collection',type=Path,help='Share the GPU after this already running native collection')
    args=parser.parse_args()
    read=lambda path:json.loads(path.read_text())
    parent=args.parent
    root=args.panel
    checkpoint=parent/'latest.pt'
    fit=root/'contact_plane.json'
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    shadow=read(root/'advanced_shadow.json')
    actor=tensor_digest(ck['actor'].items())
    offset=calibrated_plane_offset(read(fit),ck['config']['wm_holdout_lanes'],digest(checkpoint))
    _,identity,base=load_run(args.evaluation)
    if (base!=actor or actor!=shadow['actor_sha256'] or not shadow['passed'] or shadow['actor_updates']
            or tensor_digest(ck['world'].items())!=shadow['world_sha256'] or shadow['toi_plane_offset_m']!=offset
            or shadow['independent_lanes_with_gradient']!=4 or not shadow['toi_coverage_preserved']):
        raise ValueError('Fresh matched four-start shadow and actual formal baseline required')
    audit=read(root/'advanced_native_clock.json')
    nominal=read(root/'advanced_native_outcomes.json')
    if (audit['intercept_plane_calibration_sha256']!=digest(fit) or audit['actor_sha256']!=actor
            or audit['stages']['after']['world_sha256']!=shadow['world_sha256']
            or nominal['costs_sha256']!=digest(root/'advanced_native_clock.json')
            or nominal['formal_episodes_sha256']!=digest(args.evaluation/'episodes.jsonl')):
        raise ValueError('Native measurements do not refer to the candidate objective')
    command=read(parent/'startup.json')['argv'].copy()
    def get(key):
        return command[command.index('--'+key)+1]
    if (get('training-profile')!='bounded' or get('contact-bptt-mode')!='detached'
            or get('approach-objective')!='toi_racket_only' or '--resume-reset-schedule' not in command):
        raise ValueError('Expected bounded TOI-only continuation with new counters')
    if '--use-substep-toi-references' in command:
        raise ValueError('The preregistered comparison uses the incumbent reference bank')
    args.output.mkdir(parents=True,exist_ok=False)
    train=args.output/'train'
    after=args.output/'fixed128'
    for key,value in dict(output=train,cycles=1,recovery_update_budget=0,recovery_proposal_budget=1,
             approach_update_budget=1,approach_proposal_budget=3,resume_training=checkpoint,
             resume_start_block='approach',paired_evaluation_baseline=args.evaluation/'episodes_summary.json',
             intercept_plane_calibration=fit,toi_effective_start_lanes=4).items():
        flag='--'+key.replace('_','-')
        if flag in command:
            command[command.index(flag)+1]=str(value)
        else:
            command.extend([flag,str(value)])
    reg=dict(command=command,source_checkpoint_sha256=digest(checkpoint),actor_sha256=actor,
             world_sha256=shadow['world_sha256'],shadow_sha256=digest(root/'advanced_shadow.json'),
             timing_sha256=digest(fit),held_native_sha256=digest(root/'advanced_native_clock.json'),
             primary='qualified_airborne_string_contact',secondary=['ball_side_complete_except_recovery',
             'complete5','conditional_recovery_count','native_episode_return_mean'],max_accepted=1,large_training=False)
    (args.output/'registered.json').write_text(json.dumps(reg,indent=2)+'\n')
    def run(argv,log):
        with (args.output/log).open('x') as f:
            subprocess.run([sys.executable,*argv],check=True,stdout=f,stderr=subprocess.STDOUT)
    if args.after_collection:
        status=args.after_collection/'status.json'
        def finished():
            if not status.exists():
                return False
            try:
                state=read(status)['status']
            except json.JSONDecodeError:
                return False  # The collector writes progress while the last process is exiting.
            if state not in ('RUNNING','COLLECTION_FINISHED'):
                raise RuntimeError('Earlier native collection failed: '+state)
            return state=='COLLECTION_FINISHED'
        if not finished():
            procs=[]
            for proc in Path('/proc').glob('[0-9]*'):
                try:
                    cmd=(proc/'cmdline').read_bytes().split(b'\0')
                    if b'scripts/run_supermimic_event_aligned_native_probes.py' in cmd and os.fsencode(args.after_collection.resolve()) in cmd:
                        procs.append(proc)
                except (PermissionError,FileNotFoundError,ProcessLookupError):
                    continue
            if len(procs)!=1 and not finished():
                raise RuntimeError('Expected one live collector at the registered destination')
            if procs:
                proc,=procs
                identity_token=(proc/'stat').read_text().split(') ')[1].split()[19]
                while not finished():
                    if not proc.exists() or (proc/'stat').read_text().split(') ')[1].split()[19]!=identity_token:
                        if finished():
                            break
                        raise RuntimeError('Native collector exited before completion')
                    time.sleep(10)
    run(command,'train.log')
    inv=read(train/'training_completed.json')['invocation']
    if (inv['start']['actor_sha256']!=actor or inv['actor_updates']!=1 or inv['heads']['approach']['accepted']!=1
            or inv['heads']['recovery']['attempts'] or inv['actor_end_sha256']==actor
            or inv['approach_objective']!='toi_racket_only'):
        raise ValueError('Expected one real bounded Approach update')
    saved=torch.load(train/'latest.pt',weights_only=False,map_location='cpu',mmap=True)
    if saved['config']['intercept_plane_calibration']['sha256']!=digest(fit) or saved['config']['toi_effective_start_lanes']!=4:
        raise ValueError('Fitted objective or sample count changed')
    run(['scripts/evaluate_supermimic_v1.py','--checkpoint',train/'latest.pt','--output',after,'--g07-execution-root',
         get('g07-execution-root'),'--seed',str(identity['seed']),'--physics-device','cuda:0','--device','cuda:0',
         '--headless','--kit_args=--renderer/multiGpu/enabled=false'],'evaluation.log')
    run(['scripts/compare_supermimic_fixed_stages.py','--before',args.evaluation,'--after',after,
         '--stage','advanced_contact_plane_one_approach','--expected-before',actor,'--expected-after',inv['actor_end_sha256'],
         '--output',args.output/'paired.json'],'paired.log')


if __name__=='__main__':
    main()
