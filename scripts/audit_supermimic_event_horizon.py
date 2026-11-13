#!/usr/bin/env python3
"""Describe measured task durations without running physics or choosing actions.

Use explicit ROBOT takeoff/touchdown fields. Shuttle landing time must never be
used as robot touchdown. Missing fields are reported, not inferred from labels.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    loc = (len(values)-1) * p
    lo, hi = math.floor(loc), math.ceil(loc)
    return values[lo] + (values[hi]-values[lo])*(loc-lo)


def audit(rows, fields, horizons=(.32,.64,.96,1.28,1.6)):
    result = {}
    pairs = [('takeoff','hit'),('hit','touchdown'),('touchdown','stable')]
    for begin,end in pairs:
        durations = []
        missing = invalid = 0
        for row in rows:
            try:
                a,b = float(row[fields[begin]]), float(row[fields[end]])
            except (KeyError,TypeError,ValueError):
                missing += 1; continue
            if not math.isfinite(a) or not math.isfinite(b) or b < a:
                invalid += 1; continue
            durations.append(b-a)
        result[f'{begin}_to_{end}'] = {
            'fields': [fields[begin],fields[end]], 'count':len(durations),
            'missing':missing, 'invalid':invalid,
            'p50_s':percentile(durations,.5), 'p90_s':percentile(durations,.9),
            'p95_s':percentile(durations,.95),
            'fraction_within_horizon':{str(h):sum(v<=h for v in durations)/len(durations)
                                       if durations else None for h in horizons}}
    return {'measured_only':True,'duration_coverage_is_not_prediction_accuracy':True,'intervals':result}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--takeoff-key',required=True)
    parser.add_argument('--hit-key',required=True)
    parser.add_argument('--touchdown-key',required=True)
    parser.add_argument('--stable-key',required=True)
    args=parser.parse_args()
    if args.input.suffix == '.csv':
        with args.input.open(newline='') as stream:
            rows=list(csv.DictReader(stream))
    else:
        rows=json.loads(args.input.read_text())
        if isinstance(rows,dict):
            rows=rows.get('episodes',list(rows.values()))
    if not isinstance(rows,list) or not all(isinstance(r,dict) for r in rows):
        raise ValueError('Expected episode rows')
    fields={name:getattr(args,name+'_key') for name in ('takeoff','hit','touchdown','stable')}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(audit(rows,fields),indent=2)+'\n')


if __name__=='__main__':
    main()
