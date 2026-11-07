"""Full remaining model episodes from measured queries, without outcome promotion."""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_terminal_feedback_native_queries_v1 as queries
from scripts import stress_terminal_support_commands_v1 as stress
from scripts.arm_velocity_recovery_feedback_v1 import ArmVelocityRecoveryFeedbackV1
from scripts.search_terminal_prefix_arm_margin_v1 import ArmPrefixCandidate
from scripts.audit_native_frictional_recovery_v1 import state_at


def evaluate(task):
    controller, base, _, _, spheres, geometry, protocol = stress.CONTEXT
    policy = ArmVelocityRecoveryFeedbackV1(
        ArmPrefixCandidate(base, controller.qp.interface, -.006), base, controller.qp.interface,
        damping_nm_per_rad_s=task['damping'])
    state = state_at(controller.qp.model, queries.TRACE, task['tick']-queries.NATIVE_START)
    states = []; failure = None; peak = 0.; commands = []; started = time.monotonic()
    for tick in range(task['tick'], 800):
        target, _ = policy.action(tick-460, state, state.previous_command_target)
        try:
            item = stress.predict_frictional_spheres(controller.qp.model, controller.qp.interface,
                state, target, spheres, geometry=geometry, substeps=task['substeps'])
        except (ValueError, np.linalg.LinAlgError) as error:
            failure = dict(tick=tick, error=repr(error)); break
        d = item['diagnostics']
        peak = max(peak, d['joint_velocity_violation_rad_s'])
        bad = {k:v for k,v in d.items() if
            (k == 'nonfoot_ground_violation_m' and v > 1e-6) or
            (k in ['joint_position_violation_rad', 'joint_velocity_violation_rad_s', 'motor_torque_violation_nm'] and v > 3e-4)}
        if bad:
            failure = dict(tick=tick, violations=bad, first_ground_violation=item['first_ground_violation']); break
        state = item['state']; state.previous_command_target = target.copy()
        states.append(copy.deepcopy(state)); commands.append(target.tolist())
    window = stress.kinematic_window(controller.qp.model, states, protocol, .01)
    return dict(task=task, valid_steps=len(states), needed_steps=800-task['tick'], failure=failure,
        model_success=failure is None and len(states)==800-task['tick'] and window['longest_model_kinematic_window_s']>=.8,
        model_window=window, maximum_velocity_violation_rad_s=peak,
        maximum_applied_offset_rad=policy.maximum_applied_offset, clipped_requests=policy.clipped_requests,
        commands=commands, wall_seconds=time.monotonic()-started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['bundle','warm-plan','validation','sphere-audit','native-run','output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--damping', type=float, nargs='+', default=[0., .5, 1.])
    parser.add_argument('--substeps', type=int, default=16)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('Preserve prior evidence')
    paths = tuple(str(x) for x in [args.bundle,args.warm_plan,args.validation,args.sphere_audit,args.native_run])
    queries.initialize(*paths)
    files = [Path(__file__),args.bundle/'manifest.json',args.warm_plan,args.validation,args.sphere_audit,
             args.native_run/'sequence_native_trace.npz',args.native_run/'sequence_native_result.json']
    files += [Path(v.__file__) for k,v in list(sys.modules.items()) if k.startswith('scripts.') and getattr(v,'__file__',None)]
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    hashes = {str(p.resolve()):sha(p) for p in files}
    tasks = [dict(tick=tick, damping=gain, substeps=args.substeps) for tick in [460,473,478] for gain in args.damping]
    rows = []
    with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context('spawn'),
                             initializer=queries.initialize, initargs=paths) as pool:
        for row in pool.map(evaluate,tasks):
            rows.append(row)
            print(json.dumps({k:v for k,v in row.items() if k!='commands'}),flush=True)
            temporary = args.output.with_suffix('.partial.tmp')
            temporary.write_text(json.dumps(dict(rows=rows, complete=False),allow_nan=False)+'\n')
            temporary.replace(args.output.with_suffix('.partial.json'))
    for path,digest in hashes.items():
        if sha(path)!=digest: raise ValueError('Input changed '+path)
    output = dict(scope=__doc__,rows=rows,source_sha256=hashes,complete=True,
                  native_execution=False,valid_expert_labels=False,unseen_case_generalization_proven=False)
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')


if __name__=='__main__': main()
