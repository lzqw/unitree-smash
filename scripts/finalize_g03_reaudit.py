#!/usr/bin/env python3
"""Build G03 evidence from native rollout, full PPO and interruption records."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from quadsmash.tasks.stand.observation_contract import OBSERVATION_SIZE,FRAME_FIELDS,FRAME_SIZE,HISTORY_LENGTH

def load(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('reports/G03'))
    parser.add_argument('--interrupt-name',default='interrupt')
    parser.add_argument('--resume-interrupt-name',default='resume_interrupt')
    args=parser.parse_args();run=ROOT/args.run_dir;output=ROOT/args.output;output.mkdir(parents=True,exist_ok=True)
    stages={name:load(run/name/'interface.json') for name in ('jump','virtual','ball')}
    train,reload,interrupted,resumed=[load(run/name/'resume.json') for name in ('train','reload',args.interrupt_name,args.resume_interrupt_name)]
    interruption=load(run/args.interrupt_name/'interruption.json')
    protocol=load(ROOT/'configs/evaluation_protocol_v2.json')
    tests=ET.parse(run/'cpu_tests.xml').getroot().findall('.//testsuite')
    cpu_ok=sum(int(t.attrib['tests']) for t in tests)>=4 and all(int(t.attrib.get(k,0))==0 for t in tests for k in ('errors','failures','skipped'))
    checks=[]
    tolerance=protocol['checkpoint_restore_tolerances']
    for result,source in ((reload,train),(resumed,interrupted)):
        r=result['resume_metrics']
        checks.append(r['rng_exact'] and r['observation_error']<tolerance['initial_observation_max_abs']
            and r['deterministic_action_error']<tolerance['initial_action_max_abs']
            and r['root_position_error_m']<tolerance['root_position_after_5_steps_m']
            and r['joint_position_error_rad']<tolerance['joint_position_after_5_steps_rad']
            and r['root_velocity_error_m_s']<tolerance['root_velocity_after_5_steps_m_s']
            and r['restored_optimizer_states']>0 and r['restored_iteration']==source['completed_iteration']
            and result['completed_iteration']==source['completed_iteration']+result['updates'])
    metrics=dict(env_contract_ok=all(r['shape'][1]==OBSERVATION_SIZE and r['shape'][0]>=32 and r['numerical_failures']==0 for r in stages.values())
            and stages['jump']['steps']>=2000 and train['updates']>=10 and train['finite_gradients'],
        partial_reset_isolation_ok=all(r['partial_reset_isolation_ok'] for r in stages.values()),
        timeout_bootstrap_ok=cpu_ok and all(r['timeout_count']>0 and r['final_reset_observation_max_difference']>.01 for r in stages.values()),
        checkpoint_resume_ok=all(checks) and interrupted['stop_requested'] and interruption['checkpoint_saved'],
        evaluation_protocol_frozen=protocol['frozen_before_corrected_G04_training'] and len(protocol['panels'])==2
            and all(p['episodes']==200 for p in protocol['panels']) and not protocol['holdout']['results_accessed'],
        backend='Isaac Lab / GPU PhysX',observation_dim=OBSERVATION_SIZE,policy_steps=stages['jump']['steps'],
        num_envs=stages['jump']['shape'][0],fresh_ppo_updates=train['updates'],resumed_updates=reload['updates']+resumed['updates'],
        reload_metrics=reload['resume_metrics'],interrupt_reload_metrics=resumed['resume_metrics'])
    names=('jump','virtual','ball','train','reload',args.interrupt_name,args.resume_interrupt_name)
    jobs=[load(run/name/'job.json') for name in names]
    if any(j['status']!='SUCCEEDED' for j in jobs):raise ValueError('failed job is not passing evidence')
    interface=dict(version='native-policy-v2',observation_size=OBSERVATION_SIZE,frame_size=FRAME_SIZE,history_frames=HISTORY_LENGTH,
        history_span_s=.10,frame_fields=FRAME_FIELDS,tail_fields={'goal_time_position_normal_velocity_mask':11,'observed_phase_one_hot':5,'elapsed_time':1,'previous_action':18},
        frame='body frame at root link; relative point velocity subtracts reference translation and omega cross offset',
        quaternion='wxyz',action_dim=18,action_order='12 audited Go2 joints then PiPER joint1..joint6',
        stage_shapes='identical Jump/Virtual/Ball, absent ball and command fields masked and zeroed',
        timeouts='capture final observation before automatic reset; bootstrap gamma*V(final), never true terminals',
        resume='full model/normalizers/optimizer/learning-rate schedule/counters/RNG; public checkpoint.pt starts explicitly reset ground episodes after a completed PPO update; interrupted unfinished episodes are counted as aborted, not successes. Diagnostic model_i files are not resumable.',
        resume_tolerances=protocol['checkpoint_restore_tolerances'],asset_sha256=digest(ROOT/'assets/quadsmash_standard_v2.usd'))
    (output/'interface_contract.json').write_text(json.dumps(interface,indent=2)+'\n')
    (output/'evaluation_manifest.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (output/'smoke_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    (output/'commands.log').write_text('\n'.join(json.dumps(j['command']) for j in jobs)+'\n')
    (output/'REPORT.md').write_text(f'''# G03 native interface and PPO re-audit

The public smoke command ran {metrics['num_envs']} native environments for {metrics['policy_steps']} policy steps, plus virtual and real-ball interface probes. All stages use the same {OBSERVATION_SIZE}-float observation, with audited joint order, body-frame reference-point velocities, tool state, contact, ball and history-valid masks, event commands, measured phase, elapsed clock and prior action. Partial resets isolate physical state/history. Timeout final observations are captured before reset; a known-value regression distinguishes V(final)=7 from V(previous)=2 and V(reset)=-50 and excludes simultaneous true terminals.

Stock locked RSL-RL PPO performed {train['updates']} fresh updates with finite losses, gradients and parameters. The adapter changes timeout reward augmentation and checkpointing, not the PPO optimizer/update equations. Independent processes restored model, normalizers, optimizer, scheduler rate, iteration counts and RNG streams. Public checkpoint.pt explicitly starts new ground episodes after a completed PPO update; unfinished episodes are counted as aborted and excluded from success statistics. The saved canonical root/joints/targets, task buffers and history are restored exactly. Diagnostic model_i files retain arbitrary states for inspection or weight initialization and are rejected for full resume.

Earlier in-contact physical restores exceeded the unchanged continuation tolerances because PhysX solver caches are inaccessible. Those failures remain recorded. The corrected public checkpoint protocol avoids promising arbitrary mid-contact replay. Exact initial observation/action and RNG checks precede five steps with identical recorded actions under the original tolerances; policy sensitivity during replay is also reported. No recorded reference actions are used in PPO training, and no bitwise physical replay is claimed.

A controlled SIGTERM was sent through the owned GPU guard after two updates. The native loop completed its current update, saved atomically and exited; a separate process loaded that interrupted checkpoint and continued. Stale PID protection compares Linux boot ID, process start tick, UID, command hash, cwd and run directory. Missing/stale identity is never permission to kill another process.

Frozen development panels each contain 200 normal ground initializations. Seeds, perturbation ranges, recovery/jump criteria and future G10 holdout generation rules are in evaluation_manifest.json and configs/evaluation_protocol_v2.json. No G10 results were accessed. The maximum recovery joint speed (1 rad/s) is now explicit alongside the original roll/pitch, root speed, angular speed and continuous .8s window.

Raw native data/checkpoints/logs/source snapshots are under {args.run_dir}. Earlier reduced NumPy and incomplete native evidence remains invalidated. This proves engineering readiness only; G04's old root-height and accumulated-recovery success calculation still needs replacement before corrected task training/evaluation. No jump/contact/smash success rate is claimed here.
''')
    files=[output/name for name in ('REPORT.md','commands.log','interface_contract.json','evaluation_manifest.json','smoke_metrics.json')]
    files += [ROOT/'configs/evaluation_protocol_v2.json',run/'cpu_tests.xml',run/args.interrupt_name/'interruption.json']
    for name in names:
        files += [run/name/p for p in ('job.json','stdout.log','source_snapshot.zip','interface.json' if name in stages else 'resume.json')]
        if name not in stages:files += [run/name/'checkpoint.pt',run/name/'continuation_reference.pt']
    files += [ROOT/'tests'/name for name in ('test_native_observations.py','test_native_timeout_bootstrap.py','test_process_identity.py')]
    git=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True)
    evidence=dict(schema_version=1,goal_id='G03',task_id='environment_ppo_smoke',evidence_kind='observed',
        contract_sha256=digest(ROOT/'configs/acceptance.json'),metrics=metrics,
        run=dict(command='scripts/smoke.py; actual native commands in reports/G03/commands.log',
            git_commit=git.stdout.strip() if git.returncode==0 else 'uncommitted-worktree-source-snapshots-attached',
            started_at=min(j['started_at'] for j in jobs),finished_at=max(j['finished_at'] for j in jobs),
            gpu_hours=sum(j['charged_gpu_hours'] for j in jobs),env_steps=sum(j['charged_env_steps'] for j in jobs)),
        artifacts=[dict(path=str(p.relative_to(ROOT)),sha256=digest(p),bytes=p.stat().st_size) for p in dict.fromkeys(files)])
    temp=output/'evidence.json.tmp';temp.write_text(json.dumps(evidence,indent=2)+'\n');temp.replace(output/'evidence.json')
    print(json.dumps(metrics,indent=2))
    return 0 if all(metrics[k] for k in ('env_contract_ok','partial_reset_isolation_ok','timeout_bootstrap_ok','checkpoint_resume_ok','evaluation_protocol_frozen')) else 1

if __name__=='__main__':raise SystemExit(main())
