"""Project one registered research arm from its first completed R8/A8 proposal sequence."""
import argparse
import json
import time
from pathlib import Path

from scripts.report_supermimic_research_curve import read_new


def first_pair_budget(plan, rows, historical):
    blocks = [e for e in plan if e['kind'] == 'train']
    first = blocks[:2]
    if len(first) != 2 or [e['head'] for e in first] != ['recovery', 'approach']:
        raise ValueError('Expected alternating Recovery and Approach blocks')
    wall = 0.
    costs = {}
    for event in first:
        head = event['head']
        proposals = [r for r in rows if (r['block_key'], r['block']) == (event['block_key'], head)]
        count = sum(r['update_accepted'] for r in proposals)
        if count < event['target']:
            return None
        if count != event['target'] or proposals[-1]['block_accepted_updates'] != count:
            raise ValueError('First-pair history disagrees with the schedule')
        end = proposals[-1]['research_elapsed_s']
        costs[head] = dict(proposals=len(proposals), retained=count, wall_seconds=end-wall,
                           **{k: sum(r.get(k, 0.) for r in proposals) for k in (
                               'native_collect_seconds','world_update_seconds','bptt_seconds','critic_seconds')})
        wall = end
    regular = sum(costs[e['head']]['wall_seconds'] * e['target'] / costs[e['head']]['retained']
                  for e in blocks)
    refreshes = sum(e['kind'] == 'refresh' for e in plan)
    frozen = refreshes * historical['frozen_collection']['proposal_record_to_episodes_file_seconds']
    # The registered panel evaluator scores two checkpoints offline per research arm.
    fixed = 2 * historical['evaluation']['elapsed_s']
    return dict(first_pair=costs, first_pair_wall_seconds=wall, planned_retained=sum(e['target'] for e in blocks),
                remaining_regular_hours_at_first_pair=(regular-wall)/3600,
                historical_full_collection_hours=frozen/3600, historical_fixed_panel_hours=fixed/3600,
                estimated_remaining_hours_after_first_pair=(regular-wall+frozen+fixed)/3600,
                frozen_collections=refreshes, fixed_panels=2,
                scope='One arm after its first R/A pair. Includes retries in measured blocks. '
                      'Start-up after replay restore is included in the first Recovery block. '
                      'Uses historical native collection and evaluation costs. Additional refresh fitting, '
                      'checkpoint writes after the last proposal and future G1 contact-gradient cost may change elapsed time.')


def write_budget(arm):
    rows, _ = read_new(arm/'train/training_history.jsonl', 0)
    if not rows or not any(r['block']=='approach' for r in rows):
        return None
    plan = json.loads((arm/'train/config.json').read_text())['alternating_plan']
    reference = Path(__file__).resolve().parents[1]/'reports/supermimic_r16_and_contact_fit_20261121/runtime_cost_snapshot.json'
    budget = first_pair_budget(plan, rows, json.loads(reference.read_text()))
    if budget is not None:
        path = arm/'first_pair_budget.json'
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(budget, indent=2)+'\n')
        temp.replace(path)
    return budget


if __name__ == '__main__':
    import psutil
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm-dir', type=Path, required=True)
    parser.add_argument('--follow-pid', type=int)
    args = parser.parse_args()
    owner = psutil.Process(args.follow_pid) if args.follow_pid else None
    while True:
        result = write_budget(args.arm_dir)
        if result is not None:
            print(json.dumps(result), flush=True)
            break
        if owner is None or not owner.is_running() or owner.status() == psutil.STATUS_ZOMBIE:
            break
        time.sleep(30)
