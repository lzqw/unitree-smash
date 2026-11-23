"""Build an Approach learning curve from frozen panels and their training archives."""
import argparse
import hashlib
import json
from pathlib import Path

from compare_supermimic_approach_ball_only import compare
from compare_supermimic_fixed_stages import load_run
from run_supermimic_approach_ball_only import verify_frozen_milestone, verify_training_milestone
from quadsmash.rl.supermimic_independent_panel import FIXED128_SHA256
from quadsmash.rl.supermimic_terminal_scoring import APPROACH_BALL_SCORING


def read_json(directory, name):
    return json.loads((directory/name).read_text())


def read_rows(directory, name):
    return [json.loads(line) for line in (directory/name).read_text().splitlines()]


def frozen_panel(evaluation, arm):
    rows, identity, actor = load_run(evaluation)
    metrics = read_json(evaluation, 'metrics.json')
    expected_gate = .25 if arm == 'E1' else None
    if (identity['scoring'] != APPROACH_BALL_SCORING
            or identity['task'].get('research_evaluation', {}).get('panel_sha256') != FIXED128_SHA256
            or identity['execution'].get('arm_preparation_seconds') != expected_gate
            or metrics['numerical_failures'] or any(row['numerical_failures'] for row in rows)):
        raise ValueError(f'Invalid frozen {arm} development panel at {evaluation}')
    command = read_json(evaluation, 'startup.json')['argv']
    checkpoint = Path(command[command.index('--checkpoint') + 1]).resolve(strict=True)
    return checkpoint, metrics, actor


def training_cost(directory):
    archive = read_json(directory/'native_transitions', 'manifest.json')
    ticks = sum(chunk['ticks'] for chunk in archive['chunks'])
    if ticks <= 0 or archive['num_envs'] != 128 or archive['control_ticks'] != ticks or archive['transitions'] != ticks*128:
        raise ValueError(f'Incomplete native archive at {directory}')

    proposals = read_rows(directory, 'training_history.jsonl')
    episodes = read_rows(directory, 'native_episodes.jsonl')
    identities = {(row['env_id'], row['episode_index']) for row in episodes}
    if len(identities) != len(episodes) or any(row['scoring_schema'] != APPROACH_BALL_SCORING for row in episodes):
        raise ValueError(f'Invalid native episode records at {directory}')
    lifecycle = {row['stage']: row['time_ns'] for row in read_rows(directory, 'lifecycle.jsonl')}
    wall = (lifecycle['CLEANUP_FINISHED'] - lifecycle['BEFORE_APP']) / 1e9
    native_seconds = sum(row['native_collect_seconds'] for row in proposals)
    if not 0 < native_seconds <= wall:
        raise ValueError(f'Invalid run timing at {directory}')

    valid_hits = [row for row in episodes if row['qualified_airborne_string_contact'] and not row['numerical_failures']]
    cost = dict(
        new_native_control_ticks=ticks, new_native_transitions=archive['transitions'],
        training_wall_seconds=wall, native_collection_seconds=native_seconds,
        new_archive_bytes=sum(chunk['bytes'] for chunk in archive['chunks']),
        accepted_approach_updates=sum(row['update_accepted'] for row in proposals),
        actor_attempts=len(proposals), actor_bptt_seconds=sum(row['bptt_seconds'] for row in proposals),
        world_refresh_seconds=sum(row.get('world_update_seconds', 0.) for row in proposals),
        actor_peak_torch_allocated_bytes=max(row['peak_gpu_bytes'] for row in proposals),
        completed_native_episodes=len(episodes),
        qualified_native_contact_episodes=len(valid_hits),
        unique_qualified_native_case_ids=len({row['case_id'] for row in valid_hits}),
        native_numerical_failures=sum(row['numerical_failures'] for row in episodes),
    )
    breakdown = [row['proposal_wall_breakdown'] for row in proposals if row.get('proposal_wall_breakdown')]
    cost['actor_proposal_wall_components'] = dict(
        measured_attempts=len(breakdown),
        seconds={key: sum(sample[key] for sample in breakdown) for key in breakdown[0]} if breakdown else None,
    )
    return cost


def curve(arm, baseline, milestones, *, final_selection=False):
    checkpoint, metrics, actor = frozen_panel(baseline, arm)
    with checkpoint.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != metrics['evaluated_checkpoint_sha256']:
            raise ValueError('Baseline checkpoint changed after evaluation')
    pivot = dict(
        accepted_updates=0, checkpoint=str(checkpoint), actor_sha256=actor,
        checkpoint_sha256=metrics['evaluated_checkpoint_sha256'], cases=128,
        valid_returns=metrics['ball_side_complete_except_recovery'],
        cumulative_training_seconds=0., cumulative_new_native_transitions=0,
    )
    points = [pivot]
    for retained, train, evaluation in milestones:
        prior = points[-1]
        source = Path(prior['checkpoint'])
        previous_updates = prior['accepted_updates']
        current, invocation = verify_training_milestone(train, retained, source)
        evaluated, _, _ = frozen_panel(evaluation, arm)
        cost = training_cost(train)
        if (current != evaluated or invocation['start']['actor_sha256'] != prior['actor_sha256']
                or retained <= previous_updates or cost['accepted_approach_updates'] != retained - previous_updates):
            raise ValueError(f'A{retained} is not a verified continuation from A{previous_updates}')
        evidence = verify_frozen_milestone(evaluation, current, invocation)
        paired = compare(baseline, evaluation)
        wall = cost['training_wall_seconds']
        collection = cost['native_collection_seconds']
        points.append(dict(
            **evidence, accepted_updates=retained, checkpoint=str(current),
            cumulative_new_native_transitions=prior['cumulative_new_native_transitions'] + cost['new_native_transitions'],
            cumulative_training_seconds=prior['cumulative_training_seconds'] + wall,
            cumulative_eval_seconds=prior.get('cumulative_eval_seconds', 0.) + evidence['evaluation_seconds'],
            stage_training=cost,
            efficiency=dict(
                effective_actor_updates_per_training_hour=3600*(retained-previous_updates)/wall,
                observed_qualified_contact_episodes_per_collection_minute=60*cost['qualified_native_contact_episodes']/collection,
                completed_ball_episodes_per_collection_hour=3600*cost['completed_native_episodes']/collection,
                new_native_transitions_per_collection_second=cost['new_native_transitions']/collection,
            ),
            frozen_outcome=paired['after'],
            improved_cases=paired['improved_cases'], regressed_cases=paired['regressed_cases'],
            paired_exact_two_sided_p=paired['paired_exact_two_sided_p'],
        ))

    result = dict(schema='supermimic-approach-learning-curve-v1', arm=arm,
                  fixed_development_panel_sha256=FIXED128_SHA256, baseline=str(baseline), points=points)
    if final_selection:
        if arm != 'E1' or not {64, 128, 256}.issubset({point['accepted_updates'] for point in points}):
            raise ValueError('Final E1 selection requires verified A64, A128 and A256')
        selected = max(points, key=lambda point: (point['valid_returns'], -point['accepted_updates']))
        result['final_selection'] = dict(
            criterion='maximum development ball-side successes, then earliest verified checkpoint',
            **{k: selected[k] for k in ('checkpoint', 'checkpoint_sha256', 'actor_sha256', 'accepted_updates', 'valid_returns')},
            independent_panel_scored=False,
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('E0', 'E1'), required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--stage', nargs=3, action='append', required=True, metavar=('UPDATES', 'TRAIN_DIR', 'EVAL_DIR'))
    parser.add_argument('--final-selection', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    stages = [(int(retained), Path(train), Path(evaluation)) for retained, train, evaluation in args.stage]
    report = curve(args.arm, args.baseline, stages, final_selection=args.final_selection)
    staged = args.output.with_suffix('.tmp')
    staged.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    staged.replace(args.output)


if __name__ == '__main__':
    main()
