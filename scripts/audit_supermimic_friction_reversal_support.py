"""Count independent signed slip reversals in authenticated training and development contact fixtures."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

import torch

from audit_supermimic_loaded_contact_history import summarize_trace
from evaluate_supermimic_contact_rollout import sha
from quadsmash.rl.supermimic_ball_impulse_prior_v1 import BallImpulsePriorV1


def single_load(events):
    if len(events)!=1:
        return None
    e=events[0]
    signed=e['signed_post_slip_along_incident_mps']
    if not signed or any(v is None for v in signed):
        return None
    return dict(normal_points=e['normal_points'],friction_points=e['friction_points'],
                incident_tangent_mps=statistics.mean(e['tangent_speed_pre_post_mps'][0]),
                signed_outgoing_mps=statistics.mean(signed),
                min_signed_mps=min(signed),max_signed_mps=max(signed),
                every_anchor_reversed=all(v<0 for v in signed),any_anchor_reversed=any(v<0 for v in signed),
                min_separation_m=min(e['separation_m']),max_separation_m=max(e['separation_m']))


def compare(rows):
    counts=Counter()
    pairs=[]
    for row in rows:
        counts['cases']+=1
        if row['kind']!='racket_stable':
            continue
        counts['racket_stable']+=1
        minus,plus=row['minus'],row['plus']
        if minus is None or plus is None:
            continue
        counts['single_substep_with_normals_and_slip']+=1
        if not row['above_repeat_noise']:
            continue
        counts['response_above_repeat_noise']+=1
        if any(v['every_anchor_reversed']!=v['any_anchor_reversed'] for v in (minus,plus)):
            counts['mixed_sign_within_patch']+=1
            continue
        counts['uniform_anchor_signs']+=1
        if minus['every_anchor_reversed']!=plus['every_anchor_reversed']:
            counts['signed_pair_reversal_switch']+=1
            pairs.append([row['seed'],row['lane']])
            repeated = [row[k] for k in ('nominal_a','nominal_b')]
            if all(repeated):
                jitter=max(abs(repeated[0][key]-repeated[1][key]) for key in ('min_signed_mps','max_signed_mps'))
                margin=min(abs(b[key]) for b in (minus,plus) for key in ('min_signed_mps','max_signed_mps'))
                if margin>max(1.e-6,3*jitter):
                    counts['switch_beyond_nominal_slip_jitter']+=1
                    if margin>.05:
                        counts['switch_both_branches_over_0p05_mps']+=1
    return dict(counts=dict(counts),switches=pairs,distinct_switch_seeds=sorted({s for s,_ in pairs}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('fit','fixture','output'):
        p.add_argument('--'+k,type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(2)
    read=lambda f:json.loads(f.read_text())
    fit=read(args.fit)
    design=read(args.fixture/'design.json')
    prior=BallImpulsePriorV1()
    physics=Path(__file__).resolve().parents[1]/'configs/shuttle_physics_v2.json'
    if sha(physics)!=fit['physics_sha256']:
        raise ValueError('Unexpected physics source')
    mass=read(physics)['mass_kg']
    result={}
    for split in ('train','development'):
        audit_file=args.fixture/(split+'.json')
        audit=read(audit_file)
        if (sha(audit_file)!=fit['fixture_sha256'][split] or audit['design_sha256']!=sha(args.fixture/'design.json') or
                audit['split']!=split or audit['policy_training_admitted']):
            raise ValueError('Unexpected split or permission')
        samples={(v['seed'],v['lane']):v for v in audit['samples']}
        result_rows=[]
        for seed,role in design['assignments'].items():
            if role!=split:
                continue
            seed=int(seed)
            summaries={}
            for name in ('nominal_a','nominal_b','minus','plus'):
                path=args.fixture/name/f'spec_{seed}'/'trace.pt'
                if sha(path)!=audit['archives'][str(path)]:
                    raise ValueError('Raw fixture changed')
                trace=torch.load(path,map_location='cpu',mmap=True,weights_only=True)['raw']['substep_trace']
                if len(trace['manifold_frames'])!=prior.steps+1 or abs(trace['physics_dt_s']-prior.sub_dt)>1.e-12:
                    raise ValueError('Unexpected contact timebase')
                events=[[] for _ in range(design['lanes'])]
                summarize_trace(trace,0,mass,events,prior.preview)
                summaries[name]=[single_load(items) for items in events]
            for lane in range(design['lanes']):
                case=samples[(seed,lane)]
                delta=torch.tensor(case['delta']['native']).norm().item()
                noise=torch.tensor(case['repeat']['native']).norm().item()
                result_rows.append(dict(seed=seed,lane=lane,kind=case['kind'],above_repeat_noise=delta>max(3*noise,1.e-8),
                                        **{name:values[lane] for name,values in summaries.items()}))
        if len(result_rows)!=len(audit['samples']):
            raise ValueError('Unexpected duplicate or missing case')
        result[split]=dict(summary=compare(result_rows),cases=result_rows,split_sha256=sha(audit_file))
    result.update(final_holdout_opened=False,scope='Observed single-substep patch slip, signed along the incident relative tangent per anchor. A measured reversal is not a solver stick/slip flag.')
    with args.output.open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False)


if __name__=='__main__':
    main()
