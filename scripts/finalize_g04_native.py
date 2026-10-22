#!/usr/bin/env python3
"""Publish an auditable G04 result; failed learning remains a failed result."""
import argparse,csv,hashlib,json,shlex,shutil,subprocess
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic_json(p,value):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eval-dir',type=Path,required=True)
    p.add_argument('--training-dir',type=Path,required=True)
    p.add_argument('--foot-audit-dir',type=Path)
    p.add_argument('--state',choices=['IN_PROGRESS','FAILED','BUDGET_EXHAUSTED','PASSED'],required=True)
    args=p.parse_args();run=args.eval_dir.resolve();out=ROOT/'reports/G04'
    measured=json.loads((run/'metrics.json').read_text());manifest=json.loads((run/'eval_manifest.json').read_text())
    job=json.loads((run/'job.json').read_text());checkpoint=Path(manifest['checkpoint'])
    assert job['status']=='SUCCEEDED' and not (run/'error.txt').exists()
    assert digest(checkpoint)==manifest['checkpoint_sha256']==measured['checkpoint_sha256']
    rows=list(csv.DictReader((run/'episodes.csv').open()))
    protocol=json.loads((ROOT/'configs/evaluation_protocol_v2.json').read_text())
    rates={}
    for panel in protocol['panels']:
        selected=[r for r in rows if r['panel']==panel['id']]
        assert len(selected)==200 and len({r['scenario'] for r in selected})==200
        rates[panel['id']]=sum(r['success']=='True' and r['ground_init']=='True' for r in selected)/200
        for r in selected:
            if r['success']=='True':
                assert r['body_landing']=='False' and r['qualified_jump_landed']=='True' and r['stable_recovery']=='True'
                assert float(r['total_com_rise_m'])>=.08 and float(r['airborne_s'])>=.1-1e-6 and float(r['recovery_s'])>=.8-1e-5
    metrics={**measured,'sample_count':len(rows),'panel_success_rates':rates,
        'ground_init_fraction':sum(r['ground_init']=='True' for r in rows)/len(rows),
        'jump_recovery_success_rate':min(rates.values())}
    assert metrics==measured, 'published metrics differ from CSV recomputation'
    passed=len(rows)==400 and metrics['ground_init_fraction']==1. and min(rates.values())>=.9 and metrics['numerical_failures']==0
    if args.state=='PASSED' and not passed:raise ValueError('cannot publish PASSED below frozen thresholds')
    ledger=json.loads((ROOT/'status/GPU_LEDGER.json').read_text())
    records=[r for r in ledger['runs'] if r['goal']=='G04']
    if any(r['status']=='RUNNING' for r in records):raise ValueError('wait for active GPU jobs before finalizing cumulative cost')
    hours=sum(r['charged_gpu_hours'] for r in records);steps=sum(r['charged_env_steps'] for r in records)
    artifact=ROOT/'artifacts/G04/policy.pt';artifact.parent.mkdir(parents=True,exist_ok=True)
    if checkpoint.resolve()!=artifact.resolve():shutil.copy2(checkpoint,artifact)
    for name in ('episodes.csv','initial_states.csv','eval_manifest.json','metrics.json'):
        shutil.copy2(run/name,out/name)
    commands=[' '.join(shlex.quote(x) for x in r['command']) for r in records if 'command' in r]
    (out/'commands.log').write_text('\n'.join(commands)+'\n')
    failures={
        'body_contact':sum(r['body_landing']=='True' for r in rows),
        'no_qualified_jump_landing':sum(r['qualified_jump_landed']!='True' for r in rows),
        'no_contiguous_recovery':sum(r['stable_recovery']!='True' for r in rows)}
    config_path=manifest.get('config','configs/jump_ppo_v2b.json')
    (out/'REPORT.md').write_text(f'''# G04 — {args.state}

The independent native evaluation measured {metrics['sample_count']} ground-initialized episodes. Panel success fractions: {json.dumps(rates)}; the required minimum-panel score is {min(rates.values()):.4f} (threshold 0.90). Numerical failures: {metrics['numerical_failures']}. This is a development evaluation, not the unopened G10 holdout. Engineering execution and PPO updates alone do not establish task success.

Frozen evaluation protocol: configs/evaluation_protocol_v2.json. The entire robot's mass-weighted COM must rise 0.08 m after actual takeoff in the same flight that has at least 0.10 s continuous all-feet clearance. After landing, posture, horizontal/root angular/joint velocity and non-foot-ground contact constraints must hold for a continuous 0.80 s. Metrics are sampled every physics substep; body, arm, leg non-foot links, mounting plate and racket ground contacts invalidate success. The two panels contain exactly 200 unique scenarios each; initial physical states are retained in initial_states.csv. Deterministic PPO inference uses no assisting controller.

Observed failure counts (overlapping): {json.dumps(failures)}. Qualified jump landing fraction: {metrics['qualified_jump_landed_rate']:.4f}; maximum observed COM rise: {metrics['maximum_com_rise_m']:.5f} m. These results support neither a claim of trained smashing nor a proof that the hardware is incapable.

Audited standard PiPER and full payload remain in assets/quadsmash_standard_v2.usd, total mass about 20.6469917 kg; original geometry and actuator limits retained. Separate bounded-action engineering search reports/G04/capability_all_body_v2 found 17 qualified jump landings in 256 trajectories but zero complete recovery successes. That search is not a learned policy result. Earlier three training revisions used invalid physics and remain archived with their costs. The first corrected baseline also omitted the specified unsafe-body-contact reward cost; it was safely stopped, retained, and explicitly corrected in jump_ppo_v2b.json. No acceptance threshold was reduced.

Checkpoint: artifacts/G04/policy.pt, SHA256 {digest(artifact)}. Source evaluation: {run.relative_to(ROOT)}; training run: {args.training_dir}. Each native job retains command, source snapshot/hashes, logs, seeds and receipt. Full resumable checkpoint.pt files begin at a canonical ground episode boundary after a completed PPO update, preserving optimizer/normalizers/RNG/counters; arbitrary in-contact PhysX replay is not claimed.

G04 cumulative charged GPU0 walltime: {hours:.6f} h / 8 h. Conservatively charged all-environment interactions: {steps:,} / 200,000,000. This includes failed attempts, diagnostics, evaluation and a historical reserve of 2 h / 25 M interactions; the reserve is not a measured historical timing claim. G05+ is not authorized.

Evaluation configuration: {config_path}. Subsequent shaping changes and their hypotheses are retained in status/DECISIONS.md and each run's source snapshot. Raw action saturation counters from runs before wrapper clipping was removed (including corrected_v2b_main01 training) are invalid; actual actuator torque/speed records are retained separately.

Reproduce: python3 scripts/evaluate.py --stage jump_recover --checkpoint artifacts/G04/policy.pt --config {config_path} --suite dev_jump_v2 --episodes 400 --num-envs 200 --reset-mode ground --output reports/G04/NEW_EVALUATION_DIRECTORY
Verification: python3 tools/verify_goal.py --root . --goal G04 (expected to fail when measured performance is below threshold).
''')
    comparison=out/'trial_comparison.json'
    if comparison.is_file():
        trials=json.loads(comparison.read_text())
        table=['\n## Independent development evaluations\n',
            '| Training run | Minimum panel success | Non-foot contact | Learned noise std |',
            '|---|---:|---:|---:|']
        for trial in trials:
            sigma=trial['noise_std_mean']
            noise='not logged' if sigma is None else f'{sigma:.4f}'
            table.append(f"| {trial['training_run']} | {trial['minimum_success']:.1%} | {trial['body_contact']:.1%} | {noise} |")
        table.append('\nEach row is a separate 400-episode development evaluation. Sparse height/flight events with body contact or no recovery remain failures. Growing policy noise and the gap between stochastic training and deterministic evaluation motivate the proposed low-noise test; they do not prove an exclusive cause. The proposal is not an observed improvement and requires the explicit extra-revision authorization described in additional_revision_proposal.json.\n')
        with (out/'REPORT.md').open('a') as stream:stream.write('\n'.join(table))
    audit_files=[]
    if args.foot_audit_dir is not None:
        audit=args.foot_audit_dir.resolve()
        audit_job=json.loads((audit/'job.json').read_text())
        audit_metrics=json.loads((audit/'metrics.json').read_text())
        assert audit_job['status']=='SUCCEEDED' and not (audit/'error.txt').exists()
        groups=audit_metrics['groups']
        substeps=sum(group['substeps'] for group in groups.values())
        mismatches=sum(group['airborne_mismatch_substeps'] for group in groups.values())
        delta=max(group['maximum_vector_discrepancy_n'] for group in groups.values())
        replay=groups['existing_capability_trajectories']
        with (out/'REPORT.md').open('a') as stream:
            stream.write(f'\n## Supplementary foot-contact audit\n\n'
                f'In {substeps:,} environment physics substeps, the production net foot-force sensor '
                f'and four independent ground-filtered foot sensors had maximum force-vector discrepancy '
                f'{delta:.6g} N and {mismatches} airborne-label disagreements. Controls included zero-action '
                f'standing, replayed existing bounded-action trajectories, deterministic policy inference '
                f'and stochastic policy inference. This finite diagnostic does not prove agreement in '
                f'all possible configurations; no production sensor or threshold was changed.\n\n'
                f'Replay of selected existing engineering trajectories yielded '
                f'{replay["qualified_jump_landings"]}/{replay["completed_episodes"]} qualified jump landings '
                f'and {replay["complete_successes"]}/{replay["completed_episodes"]} complete recoveries. '
                f'These curated controller trajectories are not a learned-policy evaluation and do not '
                f'change the failed independent 400-episode PPO result. Source: {audit.relative_to(ROOT)}.\n')
        audit_files=[audit/name for name in ('metrics.json','per_environment.csv','episodes.csv',
            'job.json','source_hashes.json','source_snapshot.zip','REPORT.md')]
    files=[out/n for n in ('REPORT.md','commands.log','episodes.csv','initial_states.csv','eval_manifest.json','metrics.json')]
    files += [artifact,ROOT/'configs/evaluation_protocol_v2.json',run/'job.json',run/'source_hashes.json',run/'source_snapshot.zip']
    files += [f for f in (out/'trial_comparison.csv',comparison,out/'training_curve_live.png',out/'engineering_tests.xml') if f.is_file()]
    files += audit_files
    revision=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True)
    commit=revision.stdout.strip() if revision.returncode==0 else 'unborn; exact sources preserved in source_snapshot.zip'
    evidence={'schema_version':1,'goal_id':'G04','task_id':'jump_recover_ground_v1','evidence_kind':'observed',
        'contract_sha256':digest(ROOT/'configs/acceptance.json'),'metrics':metrics,
        'run':{'command':shlex.join(job['command']),'git_commit':commit,
            'started_at':job['started_at'],'finished_at':job['finished_at'],'gpu_hours':hours,'env_steps':steps},
        'artifacts':[{'path':str(f.relative_to(ROOT)),'sha256':digest(f)} for f in files]}
    atomic_json(out/'evidence.json',evidence)
    if args.state=='PASSED':
        subprocess.run(['python3','tools/verify_goal.py','--root','.','--goal','G04'],cwd=ROOT,check=True)
    status_path=ROOT/'status/STATUS.json';status=json.loads(status_path.read_text())
    status['goals']['G04'].update(state=args.state,evidence='reports/G04/evidence.json',
        notes=f'真实双面板评测400回合，较差面板成功率{min(rates.values()):.1%}；累计GPU0记账{hours:.4f}/8小时。参见原始CSV和报告，不把训练执行视为策略达标。')
    atomic_json(status_path,status)
    print(json.dumps({'state':args.state,'metrics':metrics,'gpu_hours':hours,'env_steps':steps},ensure_ascii=False))
if __name__=='__main__':main()
