#!/usr/bin/env python3
"""Recheck locked native episodes and aggregate the worse of A/B200 panels."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from quadsmash.evaluation_v4 import validate_frozen_inputs,summarize_panel,aggregate_panels,digest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--panel-a',type=Path,required=True)
    p.add_argument('--panel-b',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--policy-lock',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--protocol',type=Path,default=ROOT/'configs/g07_full_racket_eval_v4_r3.json');a=p.parse_args()
    if a.output.exists():raise ValueError('Aggregate output must be new')
    protocol,lock=validate_frozen_inputs(ROOT,a.protocol,a.policy_lock,a.checkpoint);protocol['_root']=str(ROOT)
    panels=[]
    for name,directory in [('A',a.panel_a),('B',a.panel_b)]:
        if json.loads((directory/'job.json').read_text())['status']!='SUCCEEDED':
            raise ValueError('A panel job did not complete successfully')
        if not (directory/'completed.json').is_file():raise ValueError('Missing native completion receipt')
        rows=json.loads((directory/'episodes.json').read_text());recorded=json.loads((directory/'metrics.json').read_text())
        if recorded['policy_sha256']!=lock['policy_sha256'] or recorded['policy_lock_sha256']!=digest(a.policy_lock):
            raise ValueError('Panel was produced under a different policy lock')
        recomputed=summarize_panel(rows,protocol,name)
        for key,value in recomputed.items():
            if recorded[key]!=value:raise ValueError('Panel summary differs from original episodes: '+key)
        panels.append(recorded)
    result=aggregate_panels(panels,protocol)
    result['episode_sources_sha256']={str(d/'episodes.json'):digest(d/'episodes.json') for d in [a.panel_a,a.panel_b]}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
