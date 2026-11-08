"""Check original limits, causal matching and one final native command writer."""
import argparse
import ast
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.pure_control_stance_foot_reference_v15 import PureControlStanceFootReferenceV15
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.full_authority_position_interface_v1 import FullAuthorityCommandArbiterV1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['entry','configuration','run','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();read=lambda f:json.loads(f.read_text());cfg=read(a.configuration)
    if a.output.exists():raise ValueError('Fresh check required')
    text=a.entry.read_text();ast.parse(text)
    invocation='action=native_replay.action(tick,obs,action,finished)'
    if text.count(invocation)!=1 or 'native_pure_control_stance_foot_reference_v15' not in text:
        raise ValueError('Wrong controller entry')
    block=text[text.index(invocation):text.index('observation, _, done, _ = wrapped.step(action)')]
    if 'action = ' in block or 'action=' in block.split('\n',1)[1]:
        raise ValueError('Second action writer')
    nominal=read(a.run/'native_command_inputs.json')['default_joint_position_rad']
    def make():return PureControlStanceFootReferenceV15(PositionPDInterface(**cfg['interface']),nominal,cfg['model_reference_library'])
    c=make();checks=0
    for source in c.library.records:
        test=make();initial=test.library.state(source,-1)
        selected,score=test.library.match(initial)
        if selected is None or score>1e-10:raise ValueError('Exact source not matched')
        key=selected['source_key'];old=source['source_case'];source['source_case']=-999
        if c.library.match(initial)[0]['source_key']!=key:raise ValueError('Case-ID-dependent matching')
        source['source_case']=old
        for k in [-1,len(source['trajectory'])-2,len(source['trajectory'])-1]:
            state=test.library.state(source,k);q,v=state.q.copy(),state.v.copy()
            target,d=test.command(state,'BUFFER_LOAD' if state.contacts.any() else 'AIR_PREPARE',100.+(k+1)*.01)
            final,_=FullAuthorityCommandArbiterV1(test.interface.limits).select(None,target,state.previous_command_target,owns_recovery=True)
            if not np.array_equal(q,state.q) or not np.array_equal(v,state.v) or not np.array_equal(final,target):
                raise ValueError('State mutation or command overwrite')
            if not np.isfinite(target).all() or np.any(target<test.interface.limits[:,0]) or np.any(target>test.interface.limits[:,1]):
                raise ValueError('Original joint bounds violated')
            checks+=1
    result=dict(passed=True,reference_count=len(c.library.records),command_checks=checks,
        native_execution=False,landing_success_not_inferred=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
