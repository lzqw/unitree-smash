"""Check ground-contact semantics on actual recorded collision measurements."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.native_ground_support_state_v1 import ground_contact_update


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh evidence required')
    path=a.run/'filtered_foot_pairs.npz';meta=json.loads((a.run/'filtered_contact_observation.json').read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if sha(path)!=meta['data_sha256'][path.name] or not meta['high_rate_complete']:
        raise ValueError('Complete immutable native measurements required')
    with np.load(path,allow_pickle=False) as z:
        frame=z['physics_frame'];normal=z['pair_normal_force_world_n']
        row=int(np.flatnonzero(frame==46791)[0]);ground=normal[row,0,:,0].copy()
        all_pairs=normal[row,0].sum(1)
    contacts,_=ground_contact_update(np.ones(4,bool),ground)
    net_contacts,_=ground_contact_update(np.zeros(4,bool),all_pairs)
    if contacts.any() or net_contacts.tolist()!=[True,True,False,False] or all_pairs[1,2]<=5.:
        raise ValueError('Recorded self-contact counterexample changed')
    # Semantic checks: retain a genuinely observed light contact, allow
    # loading to rise, remove zero-force contact even after high load.
    previous=np.zeros(4,bool);trace=[]
    for f in [2.,.75,6.,.25,0.]:
        forces=np.zeros((4,3));forces[0,2]=f
        previous,modes=ground_contact_update(previous,forces)
        trace.append(dict(force_n=f,contact=bool(previous[0]),loaded=bool(modes['loaded'][0])))
    if [r['contact'] for r in trace]!=[True,True,True,False,False] or trace[0]['loaded']:
        raise ValueError('Light contact or release semantics differ')
    result=dict(passed=True,native_data_used=True,new_native_execution=False,
        recorded_frame=46791,ground_force_world_n=ground.tolist(),net_force_world_n=all_pairs.tolist(),
        ground_contacts=contacts.tolist(),net_contacts=net_contacts.tolist(),load_release_sequence=trace,
        controller_stability_validated=False,source_sha256={str(p.resolve()):sha(p) for p in
            [path,Path(__file__),ROOT/'scripts/native_ground_support_state_v1.py']})
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,native_counterexample_excluded=True,light_contact_kept=True)))


if __name__=='__main__':main()
