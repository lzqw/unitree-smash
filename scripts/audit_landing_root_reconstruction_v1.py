#!/usr/bin/env python3
"""Compare reconstructed planning root poses with independently saved native poses.

This checks position/orientation at matched recorded ticks, not unrecorded PhysX
contact internals, actuator state, or future dynamics accuracy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1,rotation_wxyz
from quadsmash.rl.episode_trace_v4 import unpack_public


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve previous audit')
    read=lambda name:json.loads((a.run/name).read_text())
    if read('job.json')['status']!='SUCCEEDED':raise ValueError('Native run incomplete')
    records=read('landing_mpc_records.json');keys={(r['case'],r['control_tick']) for r in records if r['episode_index']==0 and r['control_tick']%20==0}
    default=np.asarray(read('native_command_inputs.json')['default_joint_position_rad'])
    base=read('geometry.json')['body_names'].index('base');model=LoadedPinocchioModelV1()
    paths=[a.run/x for x in ('job.json','landing_mpc_records.json','native_command_inputs.json','geometry.json','all_first_episodes.npz')]
    rows=[]
    with np.load(a.run/'all_first_episodes.npz') as z:
        poses=z['link_poses'];clocks=z['episode_clock_s'];valid=z['valid_first_episode']
    for case in sorted({case for case,tick in keys}):
        path=a.run/'fixed_policy_traces'/f'case_{case:03d}.npz';paths.append(path)
        with np.load(path) as z:public=unpack_public(z['packed_public'])
        for tick in sorted(t for c,t in keys if c==case):
            samples=np.flatnonzero(valid[:,case] & (abs(clocks[:,case]-.01*tick)<2e-6))
            if len(samples)!=1:raise ValueError('Missing or ambiguous native pose clock')
            if abs(public[tick,775]-.01*tick)>2e-6:raise ValueError('Public clock mismatch')
            native=poses[samples[0],case,base];state=model.state_from_public(public[tick],default)
            rows.append(dict(case=case,tick=tick,
                position_max_error_m=float(np.max(abs(state.q[:3]-native[:3]))),
                rotation_matrix_max_error=float(np.max(abs(state.root_rotation_world-rotation_wxyz(native[3:]))))))
    result=dict(scope=__doc__,run=str(a.run.resolve()),samples=len(rows),records=rows,
        position_max_error_m=max(r['position_max_error_m'] for r in rows),
        rotation_matrix_max_error=max(r['rotation_matrix_max_error'] for r in rows),
        source_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths+[Path(__file__),model.path]})
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','source_sha256','scope')}))


if __name__=='__main__':main()
