#!/usr/bin/env python3
"""Time-align legacy native prediction-error records; no new simulation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from collections import defaultdict
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--runs',nargs='+',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
if args.output.exists():raise RuntimeError('Preserve prior output')
results=[]
for directory in args.runs:
 name=str(directory)
 path=directory/'landing_mpc_records.json';rows=json.loads(path.read_text());history={};expected={};ended=set();matched=[];allerrors=[];pairs=[]
 for r in rows:
  c=r['case'];clock=r['clock_s'];tick=round(clock/.01)
  if c in history and tick<history[c]:ended.add(c)
  history[c]=tick
  if c in ended:continue
  e=r.get('prediction_joint_position_error_max_rad')
  if e is not None:allerrors.append(e)
  if tick==expected.get(c) and e is not None:
   matched.append(e);pairs.append(dict(case=c,clock_s=clock,max_joint_error_rad=e))
  if r.get('applied_first_step') and not r.get('held_first_action') and 'predicted_first_joint_position_rad' in r:expected[c]=tick+1
 def stats(x):return dict(n=len(x),mean=float(np.mean(x)),median=float(np.median(x)),p95=float(np.percentile(x,95))) if x else None
 results.append(dict(run=name,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),scope='First chronological clock segment per case; legacy logs lack episode IDs. Time-matched scalar telemetry, no new native run.',unfiltered_stale_including_held=stats(allerrors),one_tick_matches=stats(matched),pairs=pairs))
p=args.output
p.parent.mkdir(parents=True,exist_ok=True)
if p.exists():raise RuntimeError('Preserve prior output')
p.write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps([{k:v for k,v in r.items() if k!='pairs'} for r in results],indent=2))
