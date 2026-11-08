"""Reference matching and real full-command solver checks; not native success."""
import argparse,ast,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.pure_control_trajectory_recovery_v8 import PureControlTrajectoryRecoveryV8
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.full_authority_position_interface_v1 import FullAuthorityCommandArbiterV1
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['entry','configuration','run','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();read=lambda p:json.loads(Path(p).read_text());cfg=read(a.configuration)
    if a.output.exists():raise ValueError('Fresh check required')
    text=a.entry.read_text();ast.parse(text)
    if text.count('action=native_replay.action(tick,obs,action,finished)')!=1 or 'native_pure_control_trajectory_recovery_v8' not in text:
        raise ValueError('Wrong native entry')
    block=text[text.index('action=native_replay.action(tick,obs,action,finished)'):text.index('observation, _, done, _ = wrapped.step(action)')]
    if 'action = ' in block or 'action=' in block.split('\n',1)[1]:raise ValueError('Second writer after controller')
    nominal=read(a.run/'native_command_inputs.json')['default_joint_position_rad']
    def controller():return PureControlTrajectoryRecoveryV8(PositionPDInterface(**cfg['interface']),nominal,cfg['model_reference_library'])
    c=controller();rows=[]
    for source in c.library.records:
        test=controller();initial=test.library.state(source,-1)
        chosen,score=test.library.match(initial)
        if chosen is None or score>1e-10:raise ValueError('Exact initial mapping failed')
        expected_key=c.library.match(initial)[0]['source_key']
        source_case=source['source_case'];source['source_case']=-999
        if c.library.match(initial)[0]['source_key']!=expected_key:raise ValueError('Runtime matching uses case ID')
        source['source_case']=source_case
        changed=test.library.state(source,-1);changed.q[0]+=.3;changed.root_position_world[0]+=.3
        if test.library.match(changed)[0] is not None:raise ValueError('Uncovered state silently extrapolated')
        record=dict(source_key=source['source_key'],match_score=score,checks=[])
        for k in [-1,19,49,59]:
            state=test.library.state(source,k);q=state.q.copy();v=state.v.copy()
            time_s=100.+(k+1)*.01
            phase='BUFFER_LOAD' if state.contacts.any() else 'AIR_PREPARE'
            target,diagnostic=test.command(state,phase,time_s)
            final,mapping=FullAuthorityCommandArbiterV1(test.interface.limits).select(None,target,state.previous_command_target,owns_recovery=True)
            if not np.array_equal(q,state.q) or not np.array_equal(v,state.v) or not np.array_equal(target,final):
                raise ValueError('State changed or command overwritten: '+str(dict(source=source['source_key'],step=k,q=float(np.max(abs(q-state.q))),v=float(np.max(abs(v-state.v))),command=float(np.max(abs(target-final))),diagnostic=diagnostic['source'])))
            if not np.isfinite(target).all() or np.any(target<test.interface.limits[:,0]) or np.any(target>test.interface.limits[:,1]):
                raise ValueError('Invalid original-interface command')
            if diagnostic['reference_active']:
                errors=diagnostic['reference']
                if errors['model_position_tracking_error_m']>1e-10 or errors['model_attitude_tracking_error_rad']>1e-10:
                    raise ValueError('Reference tracking is one timestep ahead of the current state')
            record['checks'].append(dict(reference_step=k,solver_accepted=diagnostic['solver_accepted'],
                command_source=diagnostic['source'],reference_active=diagnostic['reference_active']))
        if record['checks'][-1]['reference_active']:raise ValueError('Reference remains active after its declared horizon')
        rows.append(record)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(plain(dict(passed=True,records=rows,native_execution=False,
        local_match_not_coverage_proof=True,landing_success_not_inferred=True)),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True,reference_count=len(rows),command_checks=sum(len(r['checks']) for r in rows))))


if __name__=='__main__':main()
