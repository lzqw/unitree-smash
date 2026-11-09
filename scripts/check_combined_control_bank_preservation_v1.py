"""Preserve the complete verified V20 controller when expanding its initial bank.

This is a frozen-history command-preservation check, not a new physics run.
It does not add case IDs to the controller or assert future robustness.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.pure_control_preserved_foot_replanning_v20 import PureControlPreservedFootReplanningV20
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.audit_actual_handoff_model_v1 import state_from_record


def replay(cfg, nominal, library, case, records, deadline):
    controller = PureControlPreservedFootReplanningV20(
        PositionPDInterface(**cfg['interface']), nominal,
        library, cfg['model_replanning_library'])
    maximum = 0.
    for row in records:
        if time.monotonic() > deadline:
            raise TimeoutError('Bounded command replay expired')
        state = state_from_record(controller.model, row)
        q, v = state.q.copy(), state.v.copy()
        target, diagnostic = controller.command(state, row['phase']['phase'], row['time_s'])
        if not np.array_equal(q, state.q) or not np.array_equal(v, state.v):
            raise ValueError('Controller modified the observed state')
        error = float(np.max(abs(target-row['requested'])))
        if (error > 2e-6 or bool(diagnostic['replanned']) != row['expected_replanned']
                or diagnostic.get('replanning_source_key') != row['expected_replanning_source_key']):
            raise ValueError('Protected native command changed: '+str((case,row['tick'],error)))
        maximum = max(maximum, error)
    return dict(case=case, first_tick=records[0]['tick'], last_tick=records[-1]['tick'],
        commands_checked=len(records), maximum_request_error_rad=maximum,
        replanning_selected=any(r['expected_replanned'] for r in records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['baseline', 'configuration', 'library', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    read = lambda p: json.loads(Path(p).read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    baseline = read(args.baseline)
    if args.output.exists() or baseline['version'] != 'pure_control_preserved_foot_replanning_v20':
        raise ValueError('Fresh output and verified complete V20 baseline required')
    for data in [baseline, read(args.library)]:
        for filename,digest in data['source_sha256'].items():
            if sha(filename) != digest:
                raise ValueError('Changed source '+filename)
    run = Path(baseline['native_run'])
    if read(run/'job.json')['status'] != 'SUCCEEDED':
        raise ValueError('Completed native history required')
    cfg = read(args.configuration)
    old_cfg = read(baseline['configuration'])
    for key in set(cfg) | set(old_cfg):
        if key not in ['method', 'model_reference_library'] and cfg.get(key) != old_cfg.get(key):
            raise ValueError('Changed protected execution setting '+key)
    old_bank = read(old_cfg['model_reference_library'])
    bank = read(args.library)
    if bank['references'][:len(old_bank['references'])] != old_bank['references']:
        raise ValueError('Existing initial references changed')
    if Path(cfg['model_reference_library']).resolve() != args.library.resolve():
        raise ValueError('Configuration does not use the checked expanded bank')
    nominal = read(run/'native_command_inputs.json')['default_joint_position_rad']
    with np.load(run/'pure_control_commands.npz') as data:
        arrays = {key:data[key].copy() for key in data.files}
    diagnostics = [json.loads(line) for line in (run/'pure_control_diagnostics.jsonl').read_text().splitlines()]
    if len(diagnostics) != len(arrays['lane']):
        raise ValueError('Misaligned native command log')
    jobs = []
    for case in baseline['clean_stable_cases']:
        ids = np.flatnonzero(arrays['lane'] == case)
        if len(ids) < 2 or arrays['tick'][ids[-1]] != 799 or np.any(np.diff(arrays['tick'][ids]) != 1):
            raise ValueError('Missing complete native recovery history')
        rows = []
        for index in ids:
            index = int(index)
            diagnostic = diagnostics[index]
            if (diagnostic['tick'],diagnostic['lane']) != (int(arrays['tick'][index]),case):
                raise ValueError('Misaligned native observation')
            row = {key:arrays[key][index].copy() for key in ['q','v','previous','ground_force','requested']}
            row.update(tick=int(arrays['tick'][index]),time_s=float(arrays['time_s'][index]),phase=diagnostic['phase'],
                expected_replanned=bool(diagnostic['diagnostic']['replanned']),
                expected_replanning_source_key=diagnostic['diagnostic'].get('replanning_source_key'))
            rows.append(row)
        jobs.append((case,rows))
    # Instantiate once to collect every local module dependency before pinning.
    PureControlPreservedFootReplanningV20(PositionPDInterface(**cfg['interface']), nominal,
        args.library, cfg['model_replanning_library'])
    files = [Path(__file__),args.baseline,args.configuration,args.library,Path(cfg['model_replanning_library']),
        run/'job.json',run/'native_command_inputs.json',run/'pure_control_commands.npz',run/'pure_control_diagnostics.jsonl']
    files += [Path(module.__file__) for name,module in list(sys.modules.items())
        if name.startswith('scripts.') and getattr(module,'__file__',None)]
    pins = {str(path.resolve()):sha(path) for path in files}
    deadline = time.monotonic()+1200
    with ProcessPoolExecutor(max_workers=min(4,len(jobs)),mp_context=mp.get_context('spawn')) as pool:
        futures = [pool.submit(replay,cfg,nominal,args.library,case,rows,deadline) for case,rows in jobs]
        checks = [future.result() for future in futures]
    for filename,digest in pins.items():
        if sha(filename) != digest:
            raise ValueError('Changed replay input '+filename)
    result = dict(passed=True,checks=checks,source_sha256=pins,
        native_execution=False,native_success=False,valid_expert_labels=False,
        scope='All verified V20 landing segments, including their replanning decisions; not new physics')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    temporary.replace(args.output)
    print(json.dumps(dict(passed=True,checks=checks,native_execution=False)))


if __name__ == '__main__':
    main()
