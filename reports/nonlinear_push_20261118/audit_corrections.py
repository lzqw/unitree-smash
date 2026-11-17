import json
from pathlib import Path
import numpy as np
out=Path(__file__).parent
result=json.loads((out/'RESULTS.json').read_text())['results'][0]
rows=[]
for s in result['trajectory']:
    c=s.get('nonlinear_correction')
    if c is None:continue
    proposal=next((e for e in c['evaluations'] if e['kind']=='proposal'),None)
    original=np.asarray(s['qp']['joint_position_target']);final=np.asarray(s['final_target_rad'])
    assert np.array_equal(final[12:],original[12:])
    if c['accepted']:assert np.array_equal(final,np.asarray(c['command']))
    else:assert np.array_equal(final,original)
    rows.append(dict(node=s['node'],accepted=c['accepted'],rejection=c.get('rejection'),
        cost_before=c.get('cost_before'),cost_after=c.get('cost_after'),
        calls=len(c['evaluations']),failed_evaluations=sum(not e['valid'] for e in c['evaluations']),
        maximum_command_delta_rad=float(np.max(abs(final-original))),
        normalized_linear_prediction_error=(float(np.linalg.norm((np.asarray(proposal['response'])-c['linear_predicted_response'])/c['scales'])) if proposal and proposal['valid'] else None)))
summary=dict(correction_calls=len(rows),adopted=sum(r['accepted'] for r in rows),
    model_evaluations=sum(r['calls'] for r in rows),failed_evaluations=sum(r['failed_evaluations'] for r in rows),
    rows=rows,cpu_only=True)
(out/'CORRECTION_AUDIT.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
