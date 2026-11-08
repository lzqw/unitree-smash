"""Evaluate authored geometry on measured states; this is not contact-force evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.check_constrained_recovery_suffix_v1 import source_context
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.native_recovery_query_bank_v1 import state_from_sample


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['plan','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if a.output.exists():raise ValueError('Fresh diagnostic required')
    plan=read(a.plan);bank=Path(plan['bank_run']);follow=Path(plan['bank_followup'])
    audit=read(follow/'query_bank_audit.json');meta=read(bank/'recovery_query_bank.json')
    if not audit['passed']:raise ValueError('Verified passive bank required')
    for path,h in audit['source_sha256'].items():
        if sha(path)!=h:raise ValueError('Changed recorded evidence')
    source=read(plan['source']);_,_,_,_,context=source_context(source['configuration']['source']);initialize(*context)
    model=_WORKER['controller'].qp.model;geometry=_WORKER['geometry'];selected=read(plan['selection'])['selected']
    bodies=[s['body'] for s in geometry.shapes];heads=np.array([b.startswith('Head') for b in bodies])
    if not heads.any():raise ValueError('Head geometry absent')
    rows=[]
    with np.load(bank/'recovery_query_bank.npz',allow_pickle=False) as archive:
        z={k:archive[k] for k in archive.files}
        for choice in selected:
            lane=choice['case'];samples=[]
            for index,tick in enumerate(z['control_ticks']):
                if tick<280 or not z['unfinished_first_episode'][index,lane]:continue
                sample={k:z[k][index,lane] for k in z if k!='control_ticks'}
                state=state_from_sample(model,sample,meta['origins_world'][lane]);gaps,_=geometry.evaluate(state.q)
                idx=int(gaps.argmin());samples.append(dict(control_tick=int(tick),minimum_gap_m=float(gaps[idx]),
                    closest_body=bodies[idx],head_gap_m=float(gaps[heads].min())))
            first=next((r for r in samples if r['minimum_gap_m'] < -1e-6),None)
            first_head=next((r for r in samples if r['head_gap_m'] < -1e-6),None)
            rows.append(dict(case=lane,selection=choice,samples=len(samples),initial=samples[0],
                first_negative_authored_gap=first,first_negative_head_gap=first_head,
                minimum_head_gap_m=min(r['head_gap_m'] for r in samples),
                early_samples=[r for r in samples if r['control_tick']<=320]))
    files=[a.plan,Path(plan['selection']),follow/'query_bank_audit.json',bank/'recovery_query_bank.npz',
           geometry.path,Path(__file__),ROOT/'scripts/authored_ground_clearance_v1.py']
    result=dict(scope=__doc__,rows=rows,native_states_used=True,native_contact_forces_measured=False,
        formal_scoring_changed=False,negative_gap_threshold_m=-1e-6,
        limits=['Analytic authored support is not equivalent to cooked PhysX collision or a force sensor.',
                '10 ms state sampling can miss intervening contacts.',
                'These are original-student states, not candidate-controller trajectories.'],
        source_sha256={str(p.resolve()):sha(p) for p in files})
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{k:r[k] for k in ['case','initial','first_negative_authored_gap','first_negative_head_gap']} for r in rows]))


if __name__=='__main__':main()
