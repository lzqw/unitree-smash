"""Replay the identical optimized commands with coupled joint-stop reactions.

This changes the predictive constraint dynamics, not native limits or the
physical scene. It is a full-remainder model diagnostic, not expert data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_constrained_recovery_suffix_v1 import source_context, state_from_row
from scripts.recovery_family_queries_v1 import transfer_reference
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.current_state_timed_recovery_planner_v4 import predict_plan
from scripts.timed_whole_body_recovery_v1 import build_timed_plan
from scripts.predict_speed_limited_spheres_v1 import predict_speed_limited_spheres
from scripts.hip_mount_path_constraint_v1 import HipMountPathConstraintV1
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['model-plan', 'warm-search', 'search', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Fresh diagnostic output required')
    read = lambda path: json.loads(Path(path).read_text())
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    search = read(args.search); setup = read(args.model_plan)
    source = read(setup['source']); warm = read(args.warm_search)
    *_, geometry, context = source_context(source['configuration']['source'])
    initialize(*context)
    context, _ = transfer_reference(context, state_from_row(warm['query']))
    context, _ = transfer_reference(context, state_from_row(search['query']))
    initialize(*context)
    pins = dict(search['source_sha256'])
    for path in [args.model_plan, args.warm_search, args.search, Path(__file__)]:
        pins[str(path.resolve())] = sha(path)
    for name, module in list(sys.modules.items()):
        if name.startswith('scripts.') and getattr(module, '__file__', None):
            path = str(Path(module.__file__).resolve()); digest = sha(path)
            if path in pins and pins[path] != digest:
                raise ValueError('Changed source ' + path)
            pins[path] = digest
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError('Changed evidence ' + path)
    # Identify the limit on the last rejected interval using the same
    # predictive model and pre-command state as the search.
    old = search['best']['result']; boundary = None
    if old['failure'] and old['trajectory']:
        previous = old['trajectory'][-1]
        state = state_from_row(previous)
        policy = build_timed_plan(_WORKER, search['best']['plan'])
        target, _ = policy.action(80 + old['completed_model_steps'], state,
                                 state.previous_command_target.copy())
        controller = _WORKER['controller']; interface = controller.qp.interface
        model = controller.qp.model
        next_state = predict_speed_limited_spheres(model, interface, state, target,
            _WORKER['spheres'], substeps=160,
            geometry=HipMountPathConstraintV1(_WORKER['geometry'], geometry, margin_rad=.02))
        q = next_state['state'].joint_position
        violation = np.maximum(interface.limits[:, 0] - q, q - interface.limits[:, 1])
        boundary = dict(control_tick=280 + old['completed_model_steps'],
            interval_diagnostics=next_state['diagnostics'],
            joints=[dict(name=model.joint_order[i], position_rad=float(q[i]),
                lower_rad=float(interface.limits[i, 0]), upper_rad=float(interface.limits[i, 1]),
                command_rad=float(target[i]), violation_rad=float(violation[i]))
                for i in np.flatnonzero(violation > 3e-4)])
    row = predict_plan(search['best']['plan'], search['query'], 280, 0,
                       geometry, 160, time.monotonic() + 300.)
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError('Source changed during replay ' + path)
    result = dict(scope=__doc__, case=search['case'], replay=row, rejected_interval=boundary,
        source_sha256=pins, physical_limits_changed=False,
        native_execution=False, valid_expert_labels=False)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(plain(result), indent=2, allow_nan=False) + '\n')
    temporary.replace(args.output)
    print(json.dumps(plain({k: v for k, v in row['result'].items()
                           if k in ['accepted', 'completed_model_steps', 'failure', 'wall_seconds']})))


if __name__ == '__main__':
    main()
