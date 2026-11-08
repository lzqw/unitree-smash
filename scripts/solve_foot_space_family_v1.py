"""Re-optimize the complete recovery on a preselected failure-family state.

This uses a passive native state bank, not a simulator reset or a new physical
success. The complete TRAIN87 controller is a warm start, not a fixed action
template presumed valid on all failures. Original constraints remain active.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_constrained_recovery_suffix_v1 import source_context, state_from_row
from scripts.recovery_family_queries_v1 import transfer_reference
from scripts.native_recovery_query_bank_v1 import state_from_sample
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.current_state_recovery_planner_v3 import CurrentStateRecoveryPlannerV3
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['plan', 'search', 'domain', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--case', type=int, required=True)
    parser.add_argument('--max-seconds', type=float, default=600.)
    args = parser.parse_args()
    if args.output.exists() or not 120 <= args.max_seconds <= 1800:
        raise ValueError('Fresh output and bounded search required')
    read = lambda p: json.loads(Path(p).read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    plan, search = read(args.plan), read(args.search)
    selection = read(plan['selection'])
    selected = [r for r in selection['selected'] if r['case'] == args.case]
    if len(selected) != 1 or args.case == search['case'] or not search['verified']:
        raise ValueError('Require an original preselected family outside the solved nominal case')
    audit = read(Path(plan['bank_followup'])/'query_bank_audit.json')
    bank = Path(plan['bank_run'])
    metadata = read(bank/'recovery_query_bank.json')
    if not audit['passed'] or not metadata['complete_window']:
        raise ValueError('Complete strictly passive native state bank required')
    source = read(plan['source'])
    _, _, _, geometry, context = source_context(source['configuration']['source'])
    initialize(*context)
    context, _ = transfer_reference(context, state_from_row(search['query']))
    model = _WORKER['controller'].qp.model
    with np.load(bank/'recovery_query_bank.npz', allow_pickle=False) as z:
        indices = np.flatnonzero(z['control_ticks'] == 280)
        if len(indices) != 1:
            raise ValueError('Missing unique original query tick')
        sample = {k: z[k][int(indices[0]), args.case].copy() for k in z.files if k != 'control_ticks'}
    if (not sample['unfinished_first_episode'] or sample['episode_index'] != 0
            or sample['phase'][3] <= .5):
        raise ValueError('Require a real first-episode posthit airborne query')
    measured = state_from_sample(model, sample, metadata['origins_world'][args.case])
    query = dict(q=measured.q, v=measured.v, contacts=measured.contacts,
                 target=measured.previous_command_target)
    domain_path = args.domain/f'case_{args.case:03d}.json'
    domain = read(domain_path)
    if domain['case'] != args.case or domain['verified']:
        raise ValueError('Require the previously measured unsolved policy-domain case')
    for key in query:
        if not np.array_equal(query[key], domain['query'][key]):
            raise ValueError('Native query differs from domain probe: '+key)
    context, transfer = transfer_reference(context, measured)
    state = state_from_row(query)
    warm = dict(contract='foot_space_current_state_plan_v1',
        original_knots_rad=search['original_knots'],
        transition_knots_rad=search['verified']['replay']['transition_knots'],
        braking=search['braking'], corrections=[])
    pins = dict(search['source_sha256'])
    pins.update(audit['source_sha256'])
    paths = [args.plan, args.search, Path(plan['selection']), domain_path,
        bank/'recovery_query_bank.json', bank/'recovery_query_bank.npz', Path(__file__).resolve()]
    paths += [Path(m.__file__).resolve() for name, m in list(sys.modules.items())
              if name.startswith('scripts.') and getattr(m, '__file__', None)]
    for path in paths:
        pins[str(path.resolve())] = sha(path)
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError('Changed input '+path)
    planner = CurrentStateRecoveryPlannerV3(context, geometry, workers=2,
        max_seconds=args.max_seconds, seed=882122)
    try:
        result = planner.solve(state, 280, warm, preceding_stable_ticks=0)
    finally:
        planner.close()
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError('Input changed during optimization '+path)
    result.update(case=args.case, selection=selected[0], source_sha256=pins,
        reference_transfer=transfer, query_is_native=True, simulator_state_reset=False,
        native_execution=False, valid_expert_labels=False,
        prior_unmodified_controller_completed_steps=domain['result']['completed_model_steps'],
        physical_impossibility_claim=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(plain(result), indent=2, allow_nan=False)+'\n')
    temporary.replace(args.output)
    print(json.dumps(dict(case=args.case, accepted=result['accepted'],
        wall_seconds=result['wall_seconds'], stages=[r['stage'] for r in result['records']])))


if __name__ == '__main__':
    main()
