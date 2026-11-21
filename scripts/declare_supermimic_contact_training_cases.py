"""Declare independent training-only G07 launches and signed arm excitation for contact identification."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from quadsmash.rl.supermimic_native_pair_capture_v1 import validate_spec


def case_digest(cases):
    return hashlib.sha256(json.dumps(cases, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def resolve_probe_cases(spec, original):
    panel = spec.get('contact_training_panel')
    if panel is None:
        return original, None
    if (spec.get('data_role') != 'contact_model_training' or spec.get('capture_wm_window')
            or not spec.get('capture_substeps') or panel['case_sha256'] != case_digest(panel['cases'])):
        raise ValueError('Contact training requires an identified native substep panel outside policy replay')
    cases = panel['cases']
    def keys(rows):
        return {tuple(row['launch']['position_m'] + row['launch']['velocity_mps']) for row in rows}
    if (len(cases) != 128 or len(keys(cases)) != 128 or len({r['case_id'] for r in cases}) != 128
            or keys(cases) & keys(original)):
        raise ValueError('Contact training cases must be independent of the original cases')
    return cases, panel['case_sha256']


def main():
    from scripts.freeze_full_racket_protocol_v4 import incoming_cases
    parser = argparse.ArgumentParser(description=__doc__)
    for k in ('source-spec','config','output'):
        parser.add_argument('--'+k,type=Path,required=True)
    parser.add_argument('--reserved-panel',type=Path,action='append',default=[])
    parser.add_argument('--seed',required=True,type=int)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    read=lambda p: json.loads(p.read_text())
    spec=validate_spec(read(args.source_spec))
    cfg=read(args.config)
    eligible=sorted(set(range(128))-set(cfg['wm_holdout_lanes']))
    if sorted(spec['lanes'])!=eligible:
        raise ValueError('Source spec includes held lanes or misses training lanes')
    cases=incoming_cases(64,args.seed,'A')+incoming_cases(64,args.seed+1,'B')
    rng=np.random.default_rng(args.seed+2)
    arm=rng.normal(size=(len(eligible),6))
    directions=np.zeros((len(eligible),18))
    directions[:,12:]=arm/np.linalg.norm(arm,axis=1,keepdims=True)
    panel=dict(seed=args.seed,cases=cases,case_sha256=case_digest(cases))
    spec=dict(spec,contact_training_panel=panel,lanes=eligible,request_directions=directions.tolist(),
              data_role='contact_model_training',capture_wm_window=False)
    validate_spec(spec)
    protocol=read(root/'configs/g07_full_racket_eval_v4.json')
    reserved=list(args.reserved_panel)
    for suite in [protocol['training_suite'],*protocol['formal_suites'].values()]:
        path=root/suite['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=suite['sha256']:
            raise ValueError('Original suite changed')
        reserved.append(path)
    for path in reserved:
        resolve_probe_cases(spec,read(path)['cases'])
    args.output.mkdir(parents=True,exist_ok=False)
    path=args.output/f"spec_00_tick{spec['tick']}.json"
    path.write_text(json.dumps(spec,indent=2,allow_nan=False)+'\n')
    manifest=dict(case_sha256=panel['case_sha256'],spec_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  reserved={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in reserved},lanes=eligible,
                  scope='Contact model training only. Count usable independent cases after four signed/repeated native runs.')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    main()
