#!/usr/bin/env python3
"""Recompute a net diagnostic from measured native trajectories, not final vx."""
import argparse
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--input',type=Path,nargs='+',required=True)
a=p.parse_args()
for directory in a.input:
    raw=json.loads((directory/'metrics.json').read_text())
    trace=json.loads((directory/'trace.json').read_text())
    position=np.asarray([t['position'] for t in trace])
    velocity=np.asarray([t['velocity'] for t in trace])
    reflected=((position[:,:,0]>1.4)&(velocity[:,:,0]<-.1)).any(0)
    maximum=position[:,:,0].max(0)
    geometry=np.asarray([g['net_center_relative_to_environment_m'] for g in raw['geometry']])
    centered=np.linalg.norm(geometry-[1.5,0.,.762],axis=1)<1e-5
    blocked=(maximum<1.535)&reflected&centered
    result=dict(kind='CPU recomputation of the saved native below-net test',
        native_trace=str(directory/'trace.json'),num_envs=raw['num_envs'],
        correctly_located=int(centered.sum()),first_net_reflections=int(reflected.sum()),
        blocked=int(blocked.sum()),all_nets_block=bool(blocked.all()),
        note='Original final-vx criterion can be invalidated by later robot/ground rebounds; original raw metrics preserved.',
        lanes=[dict(env_id=i,net_center=geometry[i].tolist(),maximum_x_m=float(maximum[i]),
            net_side_reversal=bool(reflected[i]),blocked=bool(blocked[i])) for i in range(len(blocked))])
    (directory/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='lanes'}),flush=True)
