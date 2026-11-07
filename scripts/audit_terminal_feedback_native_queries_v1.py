"""Model continuation from states actually reached by a prior native controller.

No new native rollout, state teleport, outcome promotion, or expert label.
The controller retains the original episode phase and actual previous command.
"""
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

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import stress_terminal_support_commands_v1 as stress
from scripts.search_terminal_prefix_arm_margin_v1 import ArmPrefixCandidate
from scripts.audit_native_frictional_recovery_v1 import state_at

TRACE=None
NATIVE_START=None


def initialize(bundle,warm,validation,sphere,native_run):
    global TRACE,NATIVE_START
    stress.initialize(bundle,warm,validation,sphere)
    native_run=Path(native_run)
    result=json.loads((native_run/'sequence_native_result.json').read_text())
    job=json.loads((native_run/'job.json').read_text())
    if result['case']!=3 or result['start_tick']!=460 or job['status']!='SUCCEEDED' or job['exit_code']!=0:
        raise ValueError('Require complete raw native execution; strict audit is not promoted')
    NATIVE_START=result['start_tick']
    with np.load(native_run/'sequence_native_trace.npz',allow_pickle=False) as z:
        TRACE={key:z[key].copy() for key in z.files}


def evaluate(task):
    controller,base,_,_,spheres,geometry,protocol=stress.CONTEXT
    policy=ArmPrefixCandidate(base,controller.qp.interface,task['arm_correction_rad'])
    state=state_at(controller.qp.model,TRACE,task['query_tick']-NATIVE_START)
    initial=copy.deepcopy(state);states=[];failure=None;maximum_command_step=0.;started=time.monotonic()
    for tick in range(task['query_tick'],800):
        target,_=policy.action(tick-460,state,state.previous_command_target)
        maximum_command_step=max(maximum_command_step,float(np.max(abs(target-state.previous_command_target))))
        try:
            item=stress.predict_frictional_spheres(controller.qp.model,controller.qp.interface,state,
                                                  target,spheres,geometry=geometry,substeps=task['substeps'])
        except (ValueError,np.linalg.LinAlgError) as error:
            failure=dict(tick=tick,prediction_error=repr(error));break
        d=item['diagnostics']
        bad={key:value for key,value in d.items() if (key=='nonfoot_ground_violation_m' and value>1e-6)
             or (key in ['joint_position_violation_rad','joint_velocity_violation_rad_s','motor_torque_violation_nm']
                 and value>3e-4)}
        if bad:
            failure=dict(tick=tick,violations=bad,first_ground_violation=item['first_ground_violation']);break
        state=item['state'];state.previous_command_target=target.copy();states.append(copy.deepcopy(state))
    window=stress.kinematic_window(controller.qp.model,states,protocol,.01)
    return dict(task=task,needed_steps=800-task['query_tick'],valid_steps=len(states),failure=failure,
        model_success=failure is None and len(states)==800-task['query_tick']
        and window['longest_model_kinematic_window_s']>=.8,model_window=window,
        maximum_command_step_rad=maximum_command_step,
        initial_q=initial.q.tolist(),initial_v=initial.v.tolist(),initial_contacts=initial.contacts.tolist(),
        actual_previous_command=initial.previous_command_target.tolist(),wall_seconds=time.monotonic()-started,
        native_continuation_executed=False,valid_expert_labels=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['bundle','warm-plan','validation','sphere-audit','native-run','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Preserve previous evidence')
    paths=tuple(str(path) for path in [args.bundle,args.warm_plan,args.validation,args.sphere_audit,args.native_run])
    initialize(*paths)
    files=[Path(__file__),args.bundle/'manifest.json',args.warm_plan,args.validation,args.sphere_audit]
    files += [args.native_run/name for name in ['job.json','sequence_native_trace.npz','sequence_native_result.json']]
    files += [Path(value.__file__) for key,value in list(sys.modules.items())
              if key.startswith('scripts.') and getattr(value,'__file__',None)]
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    hashes={str(path.resolve()):sha(path) for path in files}
    tasks=[dict(query_tick=tick,arm_correction_rad=correction,substeps=substeps)
           for tick in [460,473,478] for correction in [0.,-.006] for substeps in [16,160]]
    rows=[];started=time.monotonic()
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize,initargs=paths) as pool:
        for row in pool.map(evaluate,tasks):
            rows.append(row);print(json.dumps({key:value for key,value in row.items()
                if key not in ['initial_q','initial_v','initial_contacts','actual_previous_command']}),flush=True)
            temporary=args.output.with_suffix('.partial.tmp')
            temporary.write_text(json.dumps(dict(rows=rows,complete=False),indent=2)+'\n')
            temporary.replace(args.output.with_suffix('.partial.json'))
    for path,digest in hashes.items():
        if sha(path)!=digest:raise ValueError('Input changed: '+path)
    output=dict(scope=__doc__,rows=rows,complete=True,source_sha256=hashes,wall_seconds=time.monotonic()-started,
        native_continuation_executed=False,valid_expert_labels=False,
        preceding_native_strict_audit_promoted=False,unseen_case_generalization_proven=False)
    temporary=args.output.with_suffix('.tmp');temporary.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    temporary.replace(args.output)


if __name__=='__main__':main()
