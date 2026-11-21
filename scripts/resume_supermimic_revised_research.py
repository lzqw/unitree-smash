"""Fork the two corrected research arms from a scored common native policy."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from compare_supermimic_fixed_stages import first_episode_failure_flips, load_run
from run_supermimic_research_scale import digest, run_registered_arm
from quadsmash.rl.supermimic_physical_loop_loading_v1 import tensor_digest
from quadsmash.rl.supermimic_physical_run_guard_v1 import atomic_torch_save


def read(path):
    return json.loads(path.read_text())


def run(script, *arguments):
    subprocess.run([sys.executable, 'scripts/' + script, *map(str, arguments)], check=True,
                   cwd=Path(__file__).resolve().parents[1], env=dict(os.environ, PYTHONPATH='.:scripts'))


def native_flips(before, after):
    def first_rows(directory):
        records = [json.loads(line) for line in (directory/'episodes.jsonl').read_text().splitlines()]
        return {r['case_id']:r for r in records if r['episode_index'] == 0}
    a,b = first_rows(before),first_rows(after)
    if len(a) != 128 or a.keys() != b.keys():
        raise ValueError('Incomplete matched first-episode records')
    return first_episode_failure_flips(a, b, read(before/'task_contract.json')['physics_dt_s'])



def recover_proposal_entry(parent):
    """Salvage only a stop at proposal entry, before the first optimizer/collect call."""
    source = parent/'train/interrupted.pt'
    last = read(parent/'train/progress.json')
    fail = read(parent/'train/failure.json')
    ck = torch.load(source, map_location='cpu', mmap=True, weights_only=False)
    counters = ck['global_counters']
    progress = ck['sequential_progress']['cycles'][last['block_key']+'/'+last['block']]
    if not (fail['type'] == 'InterruptedError' and fail['message'] == 'SIGTERM'
            and ck['actor_updates'] == progress['accepted'] == last['block_accepted_updates']
            and progress['proposal_in_flight'] and progress['attempts'] == last['block_attempts']+1
            and counters['attempts'] == last['attempt_id']+1
            and counters['accepted'] == last['accepted_id'] and counters['rejected'] == last['global_rejected']
            and fail['native_ticks'] == ck['native_ticks'] == last['native_ticks']
            and ck['world_updates'] == last['world_updates'] and ck['critic_updates'] == last['critic_updates']
            and tensor_digest(ck['world'].items()) == last['model_runtime_state_sha256']):
        raise ValueError('Uncommitted optimizer or collection work at interrupt')
    counters['attempts'] -= 1
    progress['attempts'] -= 1
    ck['actor_attempts'] = progress['attempts']
    progress['proposal_in_flight'] = False
    return ck, source


def recover_paused_block(parent, pause):
    # Only the Replay from the last native collection crosses this boundary.
    # Actor, WM, Critic and all optimizer states come from the scored block entry.
    source = Path(pause['checkpoint']).resolve()
    if source != (parent/'train/pre_block.pt').resolve():
        raise ValueError('Expected the paused arm block entry')
    ck = torch.load(source, map_location='cpu', mmap=True, weights_only=False)
    if tensor_digest(ck['actor'].items()) != pause['checkpoint_actor']:
        raise ValueError('Paused actor does not match the saved block')
    interrupted = torch.load(parent/'train/interrupted.pt', map_location='cpu', mmap=True, weights_only=False)
    fail = read(parent/'train/failure.json')
    last_archive = (parent/'train/native_transitions').resolve()
    before, after = (state['replay_archive_receipts'][-1]['chunks'] for state in (ck, interrupted))
    n = sum(c['ticks'] for c in after[len(before):])
    if (fail.get('message') != 'SIGTERM' or ck['replay_archives'] != interrupted['replay_archives']
            or Path(ck['replay_archives'][-1]).resolve() != last_archive
            or interrupted['replay_archive_receipts'][:-1] != ck['replay_archive_receipts'][:-1]
            or after[:len(before)] != before or n != ck['config']['native_ticks_per_actor_update']
            or interrupted['native_ticks'] != ck['native_ticks'] + n
            or pause['body_audit']['source_chunks'] != [c['sha256'] for c in after]):
        raise ValueError('Native archive does not match the observed pause interval')
    state = interrupted['replay_state']
    versions = interrupted['replay_metadata']['policy_versions']
    rows = (torch.arange(n) + state['write_pos'] - n) % len(versions)
    if not (versions[rows] == ck['global_counters']['accepted']).all():
        raise ValueError('Additional native transitions do not belong to the scored policy')
    for key in ('native_ticks','replay_state','replay_metadata','replay_archives','replay_archive_receipts',
                'actor_start_bank','dynamic_stage_starts'):
        ck[key] = interrupted[key]
    return ck, source


def compose(parent, destination, pause=None):
    ck, source = recover_paused_block(parent, pause) if pause else recover_proposal_entry(parent)
    reg = read(parent/'registration.json')
    original = Path(reg.get('initial_checkpoint', reg['command'][reg['command'].index('--resume-training')+1]))
    base_dir = Path(reg['baseline_path']).parent
    evaluation = Path(pause['evaluation_dir']) if pause else parent/'pre_revision_fixed128'
    _, _, old = load_run(base_dir)
    _, _, actor = load_run(evaluation)
    if actor != tensor_digest(ck['actor'].items()) or old != reg['initial']['actor_sha256']:
        raise ValueError('Interruption does not match frozen native policy')
    paired = parent/'pre_revision_paired.json'
    if not paired.exists():
        run('compare_supermimic_fixed_stages.py', '--before', base_dir, '--after', evaluation,
            '--stage', 'pre_revision', '--expected-before', old, '--expected-after', actor, '--output', paired)
    result = read(paired)
    if not result['identical_initial_states'] or [r['actor_sha256'] for r in result['native_sources']] != [old, actor]:
        raise ValueError('Frozen comparison identity mismatch')
    pre, post = (read(d/'metrics.json') for d in (base_dir, evaluation))
    if pause and post['evaluated_checkpoint_sha256'] != digest(source):
        raise ValueError('Frozen evaluation used a different block checkpoint')
    outcomes = result['paired_task_outcomes']
    delta = outcomes['native_return']
    # Do not roll back on the rolling mixed-policy scalar. Use the complete paired panel.
    down = post['complete5'] < pre['complete5'] or post['qualified_airborne_string_contact'] < pre['qualified_airborne_string_contact']
    if pause:
        recovery = result['paired_recovery']
        down |= (post['body_ground_seen'] > pre['body_ground_seen']
                 or recovery['recovered_after'] < recovery['recovered_before'])
    rollback = down and delta['descriptive_paired_case_bootstrap_95'][1] < 0
    history = [json.loads(line) for line in (parent/'train/training_history.jsonl').read_text().splitlines()]
    history = [row for row in history if row['attempt_id'] <= ck['global_counters']['attempts']]
    if not history or history[-1]['accepted_id'] != ck['global_counters']['accepted']:
        raise ValueError('Training history does not reproduce the scored prefix')
    if pause and (sum(r['update_accepted'] for r in history)
                  != ck['global_counters']['accepted'] - ck['config']['research_origin']):
        raise ValueError('Incomplete common policy prefix')
    if any(row.get('contact_aux_weight', 0) != 0 for row in history):
        raise ValueError('G0 and G1 must fork before any auxiliary gradient was applied')
    if rollback:
        baseline = torch.load(original, weights_only=False, map_location='cpu', mmap=True)
        if tensor_digest(baseline['actor'].items()) != old:
            raise ValueError('Original Actor differs from the paired baseline')
        for key in ('actor', 'actor_optimizer', 'actor_optimizers', 'actor_references'):
            ck[key] = baseline[key]
    evidence = dict(source_checkpoint_sha256=digest(source), original_checkpoint=str(original),
                    measured_actor=actor, starting_actor=tensor_digest(ck['actor'].items()),
                    paired_sha256=digest(paired), rollback_to_baseline=rollback, inherited_updates=len([r for r in history if r['update_accepted']]),
                    common_prefix_aux_updates=0, native_ticks=ck['native_ticks'],
                    previous_heads={h: sum(r['block']==h and r['update_accepted'] for r in history) for h in ('recovery','approach')},
                    previous_proposals=len(history), fresh_native_stream=True)
    evidence.update(native_flips=native_flips(base_dir,evaluation),
                    frozen_scores={label:{k:scores[k] for k in ('qualified_airborne_string_contact',
                        'ball_side_complete_except_recovery','conditional_recovery_count','conditional_recovery_denominator',
                        'complete5','body_ground_seen','native_episode_return_mean')} for label,scores in (('baseline',pre),('pre_revision',post))})
    if pause:
        evidence.update(pre_auxiliary_checkpoint=str(source), extra_native_ticks=ck['config']['native_ticks_per_actor_update'],
                        pause_actor=actor, policy_prefix_retained=not rollback)
    ck['metadata']['research_revision_handoff'] = evidence
    ck['research_elapsed_s'] = 0. if rollback else (ck['research_elapsed_s'] if 'research_elapsed_s' in ck else read(parent/'exit.json')['wall_seconds'])
    destination.mkdir(exist_ok=False, parents=True)
    checkpoint = destination/'handoff.pt'
    atomic_torch_save(ck, checkpoint)
    evidence['checkpoint_sha256'] = digest(checkpoint)
    (destination/'handoff.json').write_text(json.dumps(evidence, indent=2)+'\n')
    return reg, checkpoint, evidence


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--parent', required=True, type=Path)
    cli.add_argument('--destination', required=True, type=Path)
    cli.add_argument('--tensorboard', required=True, type=Path)
    cli.add_argument('--native-pause', type=Path, help='Use the scored block entry and the durable native collection at a Return pause')
    args = cli.parse_args()
    root = args.destination.resolve()
    parent = args.parent.resolve()
    pause = read(args.native_pause) if args.native_pause else None
    reg, checkpoint, evidence = compose(parent, root, pause)
    for arm in ('G1', 'G0'):
        cmd = list(reg['command'])
        original = evidence['original_checkpoint']
        if '--resume-reset-schedule' in cmd and not evidence['rollback_to_baseline']:
            cmd.remove('--resume-reset-schedule')
        output = root/arm
        for opt, value in (('--resume-training',checkpoint),('--output',output/'train'),
                           ('--research-contact-gradient',arm),('--tensorboard-dir',args.tensorboard/arm)):
            cmd[cmd.index(opt)+1] = str(value)
        for option in ('--landing-and-first-contact-objectives', '--diagnostic-critic-loss',
                       '--continuous-approach-starts', '--shuffle-wm-quota-groups'):
            if option not in cmd:
                cmd.append(option)
        quota = ('uniform=24,pre_touchdown=8,touchdown=24,post_touchdown=16,'
                 'post_landing_support_loss=20,pre_body_ground=16,recovery_failure=12,stable_recovery=8')
        if pause:
            if '--recovery-rollout-landing' not in cmd:
                cmd.append('--recovery-rollout-landing')
            quota = quota.replace('post_landing_support_loss=20','post_landing_support_loss=16,post_landing_unload=4')
        if '--wm-stage-quota' in cmd:
            cmd[cmd.index('--wm-stage-quota')+1] = quota
        else:
            cmd.extend(('--wm-stage-quota', quota))
        before = reg['initial'].copy()
        if evidence['rollback_to_baseline']:
            before.update(checkpoint_sha256=evidence['checkpoint_sha256'], actor_sha256=evidence['starting_actor'])
        contacts = read(parent/'native_contact_counts.json')
        new = dict(reg, arm=arm, command=cmd, initial=before, revision=evidence,
                   initial_checkpoint=str(checkpoint if evidence['rollback_to_baseline'] else original),
                   previous_heads={} if evidence['rollback_to_baseline'] else evidence['previous_heads'],
                   previous_proposals=0 if evidence['rollback_to_baseline'] else evidence['previous_proposals'],
                   previous_native_contacts={} if evidence['rollback_to_baseline'] else dict(
                       env_steps=contacts['durable_env_steps'],counts=contacts['total']))
        if not new['previous_heads']:
            new['previous_heads']={'approach':0,'recovery':0}
        rc = run_registered_arm(output, args.tensorboard, new)
        if rc:
            raise SystemExit(rc)
        run('evaluate_supermimic_research_arm.py','--arm-dir',output,'--tensorboard',args.tensorboard/(arm+'_evaluation'))
    for step in (128, 256):
        before, after = (root/a/f'fixed128_{step:04d}' for a in ('G0','G1'))
        run('compare_supermimic_fixed_stages.py','--before',before,'--after',after,'--stage',f'G1_vs_G0_{step}',
            '--expected-before',load_run(before)[2],'--expected-after',load_run(after)[2],
            '--output',root/f'G1_vs_G0_{step:04d}.json')


if __name__ == '__main__':
    main()
