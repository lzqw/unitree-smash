"""Compare original and cached contact solves over full recorded MPC continuations.

Only this standalone CPU process swaps a function during the candidate replay;
the original module is restored afterwards. No native controller is changed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.audit_native_frictional_recovery_v1 import state_at
from scripts.native_terminal_attitude_v1 import load_verified_policy
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.feedback_tail_rollout_v1 import evaluate_feedback_tail
from scripts.cached_frictional_impulse_v1 import frictional_velocity_step_cached
import scripts.predict_frictional_spheres_v1 as predictor


def maximum_state_error(first,second):
    if len(first)!=len(second): raise ValueError('Different continuation lengths')
    errors=dict(q=0.,v=0.)
    for a,b in zip(first,second):
        if not np.array_equal(a['contacts'],b['contacts']): raise ValueError('Predicted contact sequence differs')
        for name in errors: errors[name]=max(errors[name],float(np.max(abs(np.asarray(a[name])-b[name]))))
    return errors


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['bundle','warm-plan','validation','sphere-audit','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--plans',type=Path,nargs='+',required=True)
    args=parser.parse_args()
    if args.output.exists() or not 1<=len(args.plans)<=3: raise ValueError('New bounded benchmark required')
    read=lambda p:json.loads(Path(p).read_text())
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest=read(args.bundle/'manifest.json');archive=args.bundle/manifest['bundle_name']
    if sha(archive)!=manifest['bundle_sha256']: raise ValueError('Changed source bundle')
    with np.load(archive,allow_pickle=False) as data: query=data['case_003_query_public'].copy()
    controller=build_controller(manifest['configuration']);model=controller.qp.model
    policy,_=load_verified_policy(model,controller,query,args.warm_plan,args.validation)
    geometry_path=ROOT/'configs/go2_piper_authored_ground_geometry_v1.json'
    protocol_path=ROOT/'configs/evaluation_protocol_v2.json'
    geometry=AuthoredGroundClearance(geometry_path,model,margin_m=0.)
    protocol=read(protocol_path)['recovery'];spheres=read(args.sphere_audit)['spheres']
    sources=[Path(__file__),args.bundle/'manifest.json',archive,args.warm_plan,args.validation,
        args.sphere_audit,geometry_path,protocol_path,model.path,*args.plans]
    sources += [Path(m.__file__) for n,m in list(sys.modules.items()) if n.startswith('scripts.') and getattr(m,'__file__',None)]
    hashes={str(p.resolve()):sha(p) for p in sources}
    results=[];original_function=predictor.frictional_velocity_step
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for path in args.plans:
        plan=read(path);decision=plan['decision'];tick=plan['tick']
        if (not decision['accepted'] or decision['planner_version']!='feedback_tail_candidate_selection_v1'
                or decision['horizon_steps']!=80 or decision['observed_control_tick']!=tick):
            raise ValueError('Require an accepted observed-state feedback-tail plan')
        trace={name:np.asarray([plan[field]]) for name,field in [('q','observed_q'),('v','observed_v'),
            ('contacts','observed_contacts'),('previous_target_rad','actual_previous_target_rad')]}
        state=state_at(model,trace,0);commands=np.asarray(decision['commands'])
        baseline=evaluate_feedback_tail(controller,state,policy,spheres,geometry,protocol,
            control_tick=tick,prefix_commands=commands,substeps=160,time_limit_s=300.)
        print(json.dumps(dict(tick=tick,stage='reference_finished',accepted=baseline['accepted'],seconds=baseline['wall_seconds'])),flush=True)
        with patch.object(predictor,'frictional_velocity_step',frictional_velocity_step_cached):
            candidate=evaluate_feedback_tail(controller,state,policy,spheres,geometry,protocol,
                control_tick=tick,prefix_commands=commands,substeps=160,time_limit_s=300.)
        if predictor.frictional_velocity_step is not original_function: raise ValueError('Reference function not restored')
        saved=decision['predicted_states']+decision['continuation_states']
        reference_errors=maximum_state_error(baseline['predicted_states'],saved)
        errors=maximum_state_error(baseline['predicted_states'],candidate['predicted_states'])
        if (not baseline['accepted'] or not candidate['accepted'] or max(reference_errors.values())>1e-10
                or max(errors.values())>1e-10
                or baseline['trailing_model_kinematic_window_s']!=candidate['trailing_model_kinematic_window_s']):
            raise ValueError('Full continuation equivalence failed')
        command_error=float(np.max(abs(candidate['commands']-baseline['commands'])))
        if command_error>1e-10: raise ValueError('Feedback actions differ')
        keys=['required_model_steps','completed_model_steps','terminal','terminal_violations',
              'trailing_model_kinematic_window_s','path_diagnostics','wall_seconds']
        row=dict(tick=tick,source_plan=str(path.resolve()),reference={k:baseline[k] for k in keys},
            candidate={k:candidate[k] for k in keys},maximum_state_error=errors,
            reference_to_saved_plan_error=reference_errors,maximum_command_error_rad=command_error,
            runtime_ratio_reference_over_candidate=baseline['wall_seconds']/candidate['wall_seconds'],
            all_predicted_contact_flags_equal=True)
        results.append(row);print(json.dumps(row),flush=True)
        args.output.with_suffix('.partial.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    for path,digest in hashes.items():
        if sha(path)!=digest: raise ValueError('Changed benchmark source '+path)
    args.output.write_text(json.dumps(dict(scope=__doc__,results=results,source_sha256=hashes,
        native_execution=False,valid_expert_labels=False,production_controller_changed=False,
        integration_substeps=160,original_iteration_limits_and_tolerances=True,
        limitations=['Timing is one sequential CPU comparison per query, not a statistical throughput or real-time guarantee.',
            'Numerical agreement on these recorded queries does not prove agreement on every state.',
            'No prediction model fidelity, recovery success or new native score follows from this speed comparison.']),
        indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
