"""Check model-only repeated replanning from a native measured initial query.

Only the initial state is measured. All subsequent states are predictions,
never represented as native observations or successful recovery episodes.
Failure preserves the exact state/command history for a follow-up search.
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
from scripts.frictional_receding_recovery_v1 import FrictionalRecedingRecoveryV1
from scripts.native_frictional_recovery_v1 import shifted_frictional_seed
from scripts.extend_frictional_recovery_v1 import set_state, replay
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['bundle','sphere-audit','query-result','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous result')
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    query = json.loads(args.query_result.read_text())
    manifest = json.loads((args.bundle/'manifest.json').read_text())
    sphere = json.loads(args.sphere_audit.read_text())
    if not query['accepted'] or query['horizon_steps'] != 80 or query['native_execution']:
        raise ValueError('Require accepted model-only measured-query result')
    hashes = dict(query['source_sha256'])
    hashes.update(sphere['source_sha256'])
    for p,h in hashes.items():
        if sha(p) != h:
            raise ValueError('Changed declared input '+p)
    trace_path = Path(query['measured_query_source'])
    if sha(trace_path) != query['query_source_sha256']:
        raise ValueError('Changed original native observation')
    model_source = trace_path.parent/'sequence_native_result.json'
    native_result = json.loads(model_source.read_text())
    case = native_result['case']; start = native_result['start_tick']
    row = next(r for r in manifest['rows'] if r['case']==case)
    if start != row['query_tick'] or native_result['bundle_sha256'] != manifest['bundle_sha256']:
        raise ValueError('Different query/bundle')
    remaining = 800-start
    if not 80 <= remaining <= 800 or remaining % 10:
        raise ValueError('Require complete original control horizon')
    planner = FrictionalRecedingRecoveryV1(manifest['configuration'],sphere['spheres'],horizon=80)
    c = planner.controller; model = c.qp.model
    geometry = AuthoredGroundClearance(ROOT/'configs/go2_piper_authored_ground_geometry_v1.json',model,margin_m=0.)
    with np.load(args.bundle/manifest['bundle_name'],allow_pickle=False) as z:
        template = model.state_from_public(z[f'case_{case:03d}_query_public'],
            np.asarray(manifest['configuration']['posture_target']))
    with np.load(trace_path,allow_pickle=False) as z:
        state = set_state(template,model,z['q'][0],z['v'][0],z['contacts'][0],z['previous_target_rad'][0])
    for key,value in [('initial_q',state.q),('initial_v',state.v),('previous_command',state.previous_command_target)]:
        if not np.array_equal(query[key],value):
            raise ValueError('Measured initial query differs: '+key)
    files = [Path(__file__),args.query_result,args.sphere_audit,args.bundle/'manifest.json',
        args.bundle/manifest['bundle_name'],trace_path,model_source,model.path]
    files += [Path(v.__file__) for k,v in list(sys.modules.items()) if k.startswith('scripts.') and getattr(v,'__file__',None)]
    for p in files:
        hashes[str(p.resolve())] = sha(p)
    warm = np.asarray(query['commands']); records=[]; states=[]; commands=[]
    began = time.monotonic(); failed = None
    def save(complete):
        record = dict(scope=__doc__,case=case,start_tick=start,needed_steps=remaining,
            predicted_executed_steps=len(commands),reaches_original_end=complete,
            records=records,commands=commands,states=states,failed_query=failed,
            source_sha256=hashes,wall_seconds=time.monotonic()-began,
            native_execution=False,valid_teacher=False,teacher_off_learning=False)
        path=args.output if complete or failed is not None else args.output.with_suffix('.partial.json')
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(plain(record),indent=2,allow_nan=False)+'\n');tmp.replace(path)
    try:
        while len(commands) < remaining:
            elapsed=len(commands)
            # This is a model-only horizon test, not the future frozen student's
            # actions: use the shifted verified first request as nominal input.
            decision=planner.solve(state,warm[0],warm)
            records.append(dict(offset=elapsed,decision=decision))
            print(json.dumps(dict(offset=elapsed,accepted=decision['accepted'],
                wall_seconds=decision['wall_seconds'],reason=decision.get('reason'))),flush=True)
            if not decision['accepted']:
                failed=dict(q=state.q,v=state.v,contacts=state.contacts,
                    previous_command=state.previous_command_target,warm_commands=warm)
                save(False);break
            segment=np.asarray(decision['commands'])[:10]
            verified=replay(c,state,state.previous_command_target,segment,sphere['spheres'],geometry)
            for i,following in enumerate(verified):
                expected=decision['predicted_states'][i]
                if max(np.max(abs(following.q-expected['q'])),np.max(abs(following.v-expected['v'])))>1e-10:
                    raise ValueError('Executed model prefix differs from accepted prediction')
                states.append(dict(q=following.q,v=following.v,contacts=following.contacts))
            commands.extend(segment);state=verified[-1]
            warm=shifted_frictional_seed(decision['commands'],10,80,state.previous_command_target)
            save(len(commands)==remaining)
        for p,h in hashes.items():
            if sha(p)!=h:raise ValueError('Input changed during repeated prediction '+p)
    finally:
        planner.close()


if __name__=='__main__':
    main()
