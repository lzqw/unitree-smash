"""Plan full recovery on four preregistered measured native failure states.

This is CPU model search following passive native capture, not native reset,
success evidence, expert data, or a real-time control benchmark.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.recovery_family_queries_v1 import transfer_reference
from scripts.native_recovery_query_bank_v1 import state_from_sample
from scripts.check_constrained_recovery_suffix_v1 import source_context, state_from_row
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.current_state_recovery_service_v1 import CurrentStateRecoveryServiceV1
from scripts.audit_joint_sequence_proposal_v1 import plain
from scripts.queue_native_terminal_support_v1 import identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    args = p.parse_args(); plan = json.loads(args.plan.read_text()); out = args.plan.parent
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    read = lambda p: json.loads(Path(p).read_text())
    def write(name, value):
        path = out/name; tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(plain(value), indent=2, allow_nan=False)+'\n'); tmp.replace(path)
    def check():
        for name, expected in plan['source_sha256'].items():
            if sha(name) != expected: raise ValueError('Changed input: '+name)
    with (out/'owner.json').open('x') as stream:
        json.dump(dict(pid=os.getpid(),plan_sha256=sha(args.plan)),stream)
    try:
        check(); began=time.monotonic()
        owner=plan['wait_process']
        write('status.json',dict(state='WAITING_FOR_VERIFIED_NATIVE_BANK',pid=os.getpid(),owner=owner))
        while True:
            live=identity(owner['pid'])
            if live is None: break
            if live!=owner['start_ticks']: raise ValueError('Waiting process identity changed')
            if time.monotonic()-began>7200: raise TimeoutError('Native bank wait expired')
            time.sleep(5)
        check()
        bank=Path(plan['bank_run']); follow=Path(plan['bank_followup'])
        if read(follow/'status.json')['state']!='COMPLETED_DIAGNOSTIC':
            raise ValueError('Native bank pipeline did not complete its unchanged strict audit')
        audit=read(follow/'query_bank_audit.json'); meta=read(bank/'recovery_query_bank.json')
        if not audit['passed'] or not meta['complete_window'] or sha(bank/'recovery_query_bank.npz')!=meta['data_sha256']:
            raise ValueError('Native state bank failed passivity/completeness checks')
        for name, expected in audit['source_sha256'].items():
            if sha(name)!=expected: raise ValueError('Changed native query evidence')
        selection=read(plan['selection']); source=read(plan['source'])
        _,knots,_,geometry,context=source_context(source['configuration']['source'])
        initialize(*context); model=_WORKER['controller'].qp.model
        if list(model.joint_order)!=meta['joint_names']: raise ValueError('Changed joint order')
        pins=dict(plan['source_sha256']); pins.update(audit['source_sha256'])
        for name,module in list(sys.modules.items()):
            if name.startswith('scripts.') and getattr(module,'__file__',None):
                path=Path(module.__file__).resolve(); pins[str(path)]=sha(path)
        rows=[]
        with np.load(bank/'recovery_query_bank.npz',allow_pickle=False) as z:
            indices=np.flatnonzero(z['control_ticks']==280)
            if len(indices)!=1: raise ValueError('Missing measured query time')
            index=int(indices[0])
            for choice in selection['selected']:
                lane=choice['case']
                sample={k:z[k][index,lane].copy() for k in z.files if k!='control_ticks'}
                if not sample['unfinished_first_episode'] or sample['episode_index']!=0 or sample['phase'][3]<=.5:
                    raise ValueError('Selected query is not observed first-episode post-hit flight')
                measured=state_from_sample(model,sample,meta['origins_world'][lane])
                query=dict(q=measured.q.copy(),v=measured.v.copy(),contacts=measured.contacts.copy(),
                    target=measured.previous_command_target.copy())
                # Match the same redundant representation used by workers.
                state=state_from_row(query)
                case_context,alignment=transfer_reference(context,state)
                # A new case has not executed the old TRAIN3 suffix patch.
                # Begin with a zero current-epoch patch and the actual target.
                warm=dict(original_knots_rad=knots,corrections=[dict(control_tick=280,
                    previous_command=state.previous_command_target.copy(),knots_rad=np.zeros((6,18)))])
                write('status.json',dict(state='CPU_MODEL_SEARCH',case=lane,pid=os.getpid(),
                    completed_cases=len(rows),max_seconds_per_case=600,native_execution=False))
                service=CurrentStateRecoveryServiceV1(case_context,geometry,pins,workers=4,max_seconds=600.)
                try: result=service.solve(state,280,warm)
                finally: service.close()
                for key in query:
                    if not np.array_equal(query[key],result['query'][key]):
                        raise ValueError('Measured state/history changed')
                row=dict(selection=choice,reference_transfer=alignment,result=result)
                rows.append(row); write(f'case_{lane:03d}.json',row)
                print(json.dumps(dict(case=lane,accepted=result['accepted'],seconds=result['wall_seconds'])),flush=True)
        check()
        for name,expected in pins.items():
            if sha(name)!=expected: raise ValueError('Source changed during search')
        write('result.json',dict(scope=__doc__,selection=selection,source_sha256=pins,
            rows=[dict(case=r['selection']['case'],accepted=r['result']['accepted'],
                file=f"case_{r['selection']['case']:03d}.json",sha256=sha(out/f"case_{r['selection']['case']:03d}.json")) for r in rows],
            native_execution=False,valid_expert_labels=False,automatic_promotion=False))
        write('status.json',dict(state='COMPLETED_MODEL_SEARCH',cases=len(rows),
            model_accepted=sum(r['result']['accepted'] for r in rows),native_execution=False))
    except BaseException as error:
        write('status.json',dict(state='FAILED',error=repr(error),pid=os.getpid()));raise


if __name__=='__main__':main()
