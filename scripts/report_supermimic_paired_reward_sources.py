"""Account for native Return on the same completed first episodes in a mixed research run."""
import argparse
import json
import math
from pathlib import Path

import torch

from quadsmash.rl.supermimic_archive_storage_v1 import read_archive_rows
from scripts.compare_supermimic_fixed_stages import load_run
from scripts.report_supermimic_online_flights import flight_flags, reference_actor


def terminal_terms(episode):
    flags = flight_flags(episode)
    return dict(body=-40. * episode['body_ground_seen'],
                qualified_contact=20. * flags['qualified_contact'],
                cross=5. * flags['scoring_cross'], target=15. * flags['scoring_target'])


def report(arm, resume_reference=None):
    directory = arm/'train/native_transitions'
    manifest = json.loads((directory/'manifest.json').read_text())
    registration = json.loads((arm/'registration.json').read_text())
    resumed = resume_reference is not None
    baseline = Path(resume_reference) if resumed else Path(registration['baseline_path']).parent
    expected_actor = reference_actor(registration, baseline, resumed)
    scored_rows, identity, actor = load_run(baseline)
    if (actor != expected_actor or identity['seed'] != manifest['seed']
            or identity['execution'] != manifest['policy_execution'] or identity['task'] != manifest['task_contract']):
        raise ValueError('Baseline differs from online execution')
    def first(path, *, live=False):
        with path.open() as stream:
            rows = (json.loads(line) for line in stream if not live or line.endswith('\n'))
            return [r for r in rows if r['episode_index']==0]
    original_rows = first(baseline/'episodes.jsonl')
    initial = {r['env_id']:r for r in original_rows}
    if len(initial) != len(original_rows) or sorted((r['case_id'],lane) for lane,r in initial.items()) != identity['assignments']:
        raise ValueError('Ambiguous baseline episode identity')
    if any(initial[r['env_id']]['native_episode_return'] != r['native_episode_return'] for r in scored_rows):
        raise ValueError('Baseline episode Return differs from its frozen score')
    completed = first(arm/'train/native_episodes.jsonl', live=True)
    if len({r['env_id'] for r in completed}) != len(completed):
        raise ValueError('Duplicate online first episode')
    # A live terminal row can reach disk before its native chunk is sealed.
    completed = [r for r in completed if r['native_episode_length_steps'] <= manifest['control_ticks']]
    if not completed:
        raise ValueError('No complete first episode in the committed native archive')
    for r in completed:
        if (r['case_id'] != initial[r['env_id']]['case_id'] or r['scoring_schema'] != identity['scoring']
                or r['scope'] != 'mixed-policy monitoring, not frozen score'):
            raise ValueError('Incompatible online episode')
    lanes = torch.tensor([r['env_id'] for r in completed])
    lengths = torch.tensor([r['native_episode_length_steps'] for r in completed])
    counts = torch.zeros_like(lengths)
    total = torch.zeros(len(lanes),dtype=torch.float64)
    components = {}
    first_body_penalty = {}
    sources = []
    for receipt in manifest['chunks']:
        with read_archive_rows(directory,receipt,scratch=directory) as rows:
            for row in rows:
                valid = row['before']['episode_index'][lanes]==0
                if not (row['before']['elapsed_tick'][lanes][valid] == counts[valid]).all():
                    raise ValueError('Nonconsecutive native episode clock')
                counts += valid.long()
                total += row['reward'].reshape(-1)[lanes].double()*valid
                before = row['before']
                body = row['reward_components']['body'].reshape(-1)[lanes]
                for i in torch.where(valid & (body < 0))[0].tolist():
                    lane = int(lanes[i])
                    if lane not in first_body_penalty:
                        task = before['task']
                        first_body_penalty[lane] = dict(
                            episode_tick=int(before['elapsed_tick'][lane]),
                            native_tick=row.get('native_tick'), policy_version=row.get('policy_version'),
                            actor_sha256=row.get('actor_state_sha256'),
                            recovery_gate=bool(task['had_contact'][lane] and task['released'][lane]))
                for key,value in row['reward_components'].items():
                    if key not in components:
                        components[key] = torch.zeros_like(total)
                    components[key] += value.reshape(-1)[lanes].double()*valid
        sources.append(receipt['sha256'])
        if torch.equal(counts,lengths):
            break
    if not torch.equal(counts,lengths):
        raise ValueError('Archive does not cover the declared episode lengths')
    torch.testing.assert_close(total,sum(components.values()),rtol=0,atol=1e-4)
    terms = []
    for i,r in enumerate(completed):
        original = initial[r['env_id']]
        current = r['native_episode_return']
        if not math.isclose(current,float(total[i]),abs_tol=1e-6):
            raise ValueError('Archived rewards disagree with terminal Return')
        pre = terminal_terms(original)
        post = {k:float(v[i]) for k,v in components.items()}
        delta = {k:post[k]-v for k,v in pre.items()}
        change = current-original['native_episode_return']
        delta['other_terms_combined'] = change-sum(delta.values())
        terms.append(dict(case_id=r['case_id'],env_id=r['env_id'],before=original['native_episode_return'],
                          after=current,difference=change,baseline_reconstructed_terms=pre,
                          online_measured_terms=post,difference_by_term=delta,
                          first_body_penalty=first_body_penalty.get(r['env_id']),
                          behavior_at_terminal=r.get('behavior_at_terminal')))
    mean = {k:sum(r['difference_by_term'][k] for r in terms)/len(terms) for k in terms[0]['difference_by_term']}
    return dict(scope='First completed same-case TRAIN128 episodes. Baseline body, qualified contact, cross and target use native terminal latches and one-shot weights. Online terms are summed from the archive. Other baseline terms cannot be individually reconstructed. Mixed behavior, termination order, no frozen-policy comparison.',
                reference_scope='scored_resume' if resumed else 'experiment_start',
                reference_checkpoint_sha256=registration['revision']['source_checkpoint_sha256'] if resumed else None,
                baseline_actor_sha256=actor,source_chunk_sha256=sources,cases=len(terms),
                mean_before=sum(r['before'] for r in terms)/len(terms),mean_after=sum(r['after'] for r in terms)/len(terms),
                mean_difference_by_term=mean,records=terms)


def publish(result, directory):
    from torch.utils.tensorboard import SummaryWriter
    with SummaryWriter(str(directory)) as writer:
        step=result['cases']
        prefix = ('paired_return_difference_vs_resume' if result.get('reference_scope') == 'scored_resume'
                  else 'paired_return_difference')
        for key,value in result['mean_difference_by_term'].items():
            writer.add_scalar(prefix+'/'+key,value,step)
        writer.add_scalar(prefix+'/total',result['mean_after']-result['mean_before'],step)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm-dir',type=Path,required=True)
    parser.add_argument('--resume-reference',type=Path,help='Frozen evaluation directory of the scored resume checkpoint')
    parser.add_argument('--tensorboard',type=Path)
    args=parser.parse_args()
    result=report(args.arm_dir,args.resume_reference)
    if args.tensorboard:
        publish(result,args.tensorboard)
    print(json.dumps(result,indent=2,allow_nan=False))
