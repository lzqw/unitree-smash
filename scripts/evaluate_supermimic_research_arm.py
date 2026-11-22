"""Score saved research milestones on the declared held-out fixed panel and log paired outcomes."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
from torch.utils.tensorboard import SummaryWriter

from compare_supermimic_fixed_stages import load_run, paired_task_outcomes
from quadsmash.rl.supermimic_physical_loop_loading_v1 import tensor_digest
from quadsmash.rl.supermimic_independent_panel import fixed128_cases
from run_supermimic_research_scale import digest
from report_supermimic_research_curve import arm_origin, axes


def score_fixed128(folder, checkpoint, checkpoint_sha, actor_sha, execution_root, seed):
    def validate(directory):
        _, identity, actor = load_run(directory)
        cases, marker = fixed128_cases(identity['task']['protocol_sha256'], seed)
        metrics = json.loads((directory/'metrics.json').read_text())
        if (identity['seed'] != seed or identity['task'].get('research_evaluation') != marker
                or sorted((r['case_id'], lane) for lane, r in enumerate(cases)) != identity['assignments']
                or actor != actor_sha or metrics['evaluated_checkpoint_sha256'] != checkpoint_sha):
            raise ValueError('Fixed evaluation policy, case assignment or checkpoint differs')
        return metrics

    with folder.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if folder.exists():
            return validate(folder)
        with tempfile.TemporaryDirectory(dir=folder.parent, prefix='.'+folder.name+'_') as temp:
            run_dir = Path(temp)/'run'
            cmd = [sys.executable, 'scripts/evaluate_supermimic_v1.py', '--checkpoint', str(checkpoint),
                   '--output', str(run_dir), '--g07-execution-root', str(execution_root),
                   '--seed', str(seed), '--research-fixed128', '--physics-device', 'cuda:0', '--device',
                   'cuda:0', '--headless', '--kit_args=--renderer/multiGpu/enabled=false']
            with folder.with_suffix('.log').open('a') as log:
                subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], env=dict(os.environ,PYTHONPATH='.:scripts'),
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            metrics = validate(run_dir)
            run_dir.rename(folder)
            return metrics


def paired_scalars(result):
    outcomes = result.get('paired_task_outcomes') or paired_task_outcomes(result)
    recovery = result['paired_recovery']
    metrics = {}
    for name, comparison in {**outcomes['binary'], 'common_contact_recovery': recovery}.items():
        metrics[name+'/gained_cases'] = len(comparison['improved_cases'])
        metrics[name+'/lost_cases'] = len(comparison['regressed_cases'])
    for field in ('common_contacts', 'recovered_before', 'recovered_after'):
        metrics['common_contact_recovery/'+field] = recovery[field]
    for name, cases in result.get('landing_diagnostics', {}).get('failure_flips', {}).items():
        metrics['native_failure_flips/'+name] = len(cases)
    conditions = result.get('ball_side_flip_conditions', {})
    if conditions.get('available'):
        for group in ('improved_cases', 'regressed_cases'):
            counts = conditions[group]['sole_condition_counts']
            for reason in ('speed_threshold', 'released_downward_forward', 'actual_net_and_target'):
                metrics[f'ball_side/{group}/sole_{reason}'] = counts.get(reason, 0)
    change = outcomes['native_return']
    low, high = change['descriptive_paired_case_bootstrap_95']
    metrics.update({'native_return/mean_delta': change['mean'], 'native_return/median_delta': change['median'],
                    'native_return/mean_delta_ci_low': low, 'native_return/mean_delta_ci_high': high})
    return metrics


def command_scalars(report, *, before_checkpoint, after_checkpoint, before_actor, after_actor, archive_hash):
    expected = (before_checkpoint, after_checkpoint, before_actor, after_actor, archive_hash)
    actual = tuple(report[k] for k in ('before_checkpoint_sha256','after_checkpoint_sha256',
                                       'before_actor_sha256','after_actor_sha256','archive_manifest_sha256'))
    if actual != expected:
        raise ValueError('Fixed command audit belongs to different checkpoints, policies or state archive')
    metrics = {'actor_parameter_rms': report['parameter_delta']['rms']}
    for window, observations in report['groups'].items():
        for gate in ('recovery_only','approach_only','both_heads_active','both_heads_inactive'):
            group = observations['groups'][gate]
            prefix = f'{window}/{gate}/'
            metrics[prefix+'rows'] = group['rows']
            if group['rows']:
                metrics[prefix+'final_target_rms_rad'] = group['final_target_rad']['rms']
                metrics[prefix+'changed_rows'] = group['final_target_rad']['changed_rows']
                metrics[prefix+'preclamp_kl_per_dim'] = group['gaussian_kl_per_dim']
    return metrics


def training_interactions(native_origin, end_tick, lanes, refreshes):
    """Count physics steps through this point, including the whole asynchronous frozen rollout."""
    if lanes <= 0 or end_tick < native_origin:
        raise ValueError('Invalid native interaction range')
    frozen_ticks = 0
    labeled_steps = 0
    last_end = native_origin
    for receipt in sorted(refreshes, key=lambda r: r['start_tick']):
        start, end = receipt['start_tick'], receipt['end_tick']
        if start >= end_tick:
            break
        if (start < last_end or end <= start or end > end_tick
                or end - start != receipt['native_ticks'] or receipt['complete_lanes'] != lanes
                or len(receipt['lengths']) != lanes
                or any(n <= 0 or n > end-start for n in receipt['lengths'])):
            raise ValueError('Frozen receipt does not match the completed training interval')
        frozen_ticks += end - start
        labeled_steps += sum(receipt['lengths'])
        last_end = end
    total = (end_tick - native_origin) * lanes
    return dict(training_total=total, online_mixed=total - frozen_ticks * lanes,
                frozen_all=frozen_ticks * lanes, frozen_in_complete_rtg_episodes=labeled_steps)


def checkpoint_interactions(checkpoint, native_origin, lanes):
    receipts = []
    for entry in checkpoint['alternating_schedule']['results']:
        if entry['event']['kind'] == 'refresh':
            result = entry['result']
            receipt = json.loads(Path(result['manifest_path']).read_text())
            if (receipt['policy_id'] != result['policy_id'] or receipt['actor_sha256'] != result['actor_sha256']
                    or receipt['schedule_event_id'] != entry['event']['id']):
                raise ValueError('Frozen collection identity differs from saved schedule')
            receipts.append(receipt)
    return training_interactions(native_origin, checkpoint['native_ticks'], lanes, receipts)


def log_fixed_scores(writer, metrics, positions):
    for name in ('qualified_airborne_string_contact', 'ball_side_complete_except_recovery', 'complete5',
                 'body_ground_seen', 'conditional_recovery_count', 'recovery_given_airborne_string_contact',
                 'native_episode_return_mean'):
        if metrics.get(name) is not None:
            for axis, position in positions.items():
                writer.add_scalar(f'fixed128/{axis}/{name}', metrics[name], position)


def evaluate(folder, step, writer):
    reg=json.loads((folder/'registration.json').read_text())
    cmd=reg['command']
    seed=882072
    execution_root=cmd[cmd.index('--g07-execution-root')+1]
    initial_checkpoint=Path(reg.get('initial_checkpoint', cmd[cmd.index('--resume-training')+1]))
    old=reg['initial']['actor_sha256']
    baseline=folder.parent/'research_fixed128_zero'
    initial_metrics=score_fixed128(baseline, initial_checkpoint, reg['initial']['checkpoint_sha256'], old, execution_root, seed)
    if step == 0:
        log_fixed_scores(writer, initial_metrics, dict(retained_updates=0, new_native_env_steps=0, training_wall_seconds=0))
        writer.flush()
        return
    train=folder/'train'
    checkpoint=train/f'research_{step:04d}.pt'
    ck=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    retained=ck['global_counters']['accepted']-ck['config']['research_origin']
    actor=tensor_digest(ck['actor'].items())
    if retained!=step or ck['config']['training_profile']!='research_scale':
        raise ValueError('Milestone differs from actual retained optimizer count')
    milestone_clock = {key: ck[key] for key in ('native_ticks', 'alternating_schedule')}
    del ck
    folder_eval=folder/f'fixed128_{step:04d}'
    def run(args,log):
        with (folder/log).open('a') as stream:
            subprocess.run([sys.executable,*map(str,args)],cwd=Path(__file__).resolve().parents[1],
                           env=dict(os.environ,PYTHONPATH='.:scripts'),stdout=stream,stderr=subprocess.STDOUT,check=True)
    checkpoint_sha = digest(checkpoint)
    metrics=score_fixed128(folder_eval, checkpoint, checkpoint_sha, actor, execution_root, seed)
    paired=folder/f'paired_{step:04d}.json'
    if not paired.exists():
        run(['scripts/compare_supermimic_fixed_stages.py','--before',baseline,'--after',folder_eval,'--stage',reg['arm']+str(step),
             '--expected-before',old,'--expected-after',actor,'--output',paired],f'paired_{step}.log')
    result=json.loads(paired.read_text())
    if not result['identical_initial_states'] or [r['actor_sha256'] for r in result['native_sources']]!=[old,actor]:
        raise ValueError('Paired panel has the wrong initial states or policy hashes')
    trace=[json.loads(line) for line in (train/'training_history.jsonl').read_text().splitlines()]
    origin = arm_origin(reg)
    row = next(r for r in trace if r['update_accepted'] and r['accepted_id'] == origin[0] + step)
    positions = axes(row, origin, metrics['num_envs'])
    lanes = metrics['num_envs']
    budget = dict(checkpoint_sha256=checkpoint_sha,
                  heldout_evaluation_env_steps=lanes * metrics['control_ticks'],
                  shared_zero_evaluation_env_steps=initial_metrics['num_envs'] * initial_metrics['control_ticks'])
    try:
        if milestone_clock['native_ticks'] != row['native_ticks']:
            raise ValueError('Milestone and Actor log native step counts differ')
        budget['at_actor_checkpoint'] = checkpoint_interactions(milestone_clock, origin[1], lanes)
        if (train/'training_completed.json').is_file():
            final = torch.load(train/'latest.pt', map_location='cpu', mmap=True, weights_only=False)
            completed = json.loads((train/'training_completed.json').read_text())
            if final['native_ticks'] != completed['native_ticks']:
                raise ValueError('Final native step count differs from training summary')
            budget['at_training_end'] = checkpoint_interactions(final, origin[1], lanes)
            del final
    except (OSError, KeyError, ValueError) as error:
        # Missing historical accounting must not discard genuine fixed-panel scores.
        budget['accounting_issue'] = str(error)
    budget_path = folder/f'interaction_budget_{step:04d}.json'
    budget_path.write_text(json.dumps(budget, indent=2)+'\n')
    for name, value in budget.get('at_actor_checkpoint', {}).items():
        writer.add_scalar('interactions_at_checkpoint/'+name, value, step)
    if step == 128:
        log_fixed_scores(writer, initial_metrics, dict.fromkeys(positions, 0))
    log_fixed_scores(writer, metrics, positions)
    for name, value in paired_scalars(result).items():
        for axis, position in positions.items():
            writer.add_scalar(f'fixed128_paired/{axis}/{name}', value, position)
    archive = Path(cmd[cmd.index('--archive')+1])
    command_file = folder/f'fixed_commands_{step:04d}.json'
    if not command_file.is_file():
        with tempfile.TemporaryDirectory(dir=folder) as tmp:
            path = Path(tmp)/'audit.json'
            run(['scripts/audit_supermimic_paired_g07_commands.py','--before',initial_checkpoint,'--after',checkpoint,
                 '--archive',archive,'--safety-mapping',cmd[cmd.index('--safety-mapping')+1],'--output',path],
                f'fixed_commands_{step:04d}.log')
            path.replace(command_file)
    commands = command_scalars(json.loads(command_file.read_text()),before_checkpoint=reg['initial']['checkpoint_sha256'],
                               after_checkpoint=checkpoint_sha,before_actor=old,after_actor=actor,
                               archive_hash=digest(archive/'manifest.json'))
    for name, value in commands.items():
        for axis, position in positions.items():
            writer.add_scalar(f'fixed_commands/{axis}/{name}', value, position)
    writer.flush()


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arm-dir',type=Path,required=True)
    ap.add_argument('--tensorboard',type=Path,required=True)
    ap.add_argument('--steps',nargs='+',type=int,default=[128,256])
    args=ap.parse_args()
    with SummaryWriter(str(args.tensorboard)) as w:
        for step in args.steps:evaluate(args.arm_dir.resolve(),step,w)
