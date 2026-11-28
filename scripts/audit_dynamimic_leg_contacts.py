"""Compare authored Go2 collision surfaces with native first-body events and H16 forecasts."""
import argparse
import json
from pathlib import Path

import torch

from quadsmash.rl.dynamimic_lagrangian_v1 import mv
from quadsmash.rl.dynamimic_ground_clearance import Go2GroundClearance
from quadsmash.rl.supermimic_full_body_geometry_v1 import FullBodyGeometryV1
from scripts.fit_dynamimic_recovery_wm import DynaMimicRecoveryWorldModel, local_robot_state, read_data, rotation


@torch.no_grad()
def audit(args):
    data, manifest = read_data(args.native, ('active', 'strict', 'relaxed', 'done', 'net'))
    contract = json.loads((args.native/'actuator_contract.json').read_text())
    model = DynaMimicRecoveryWorldModel(args.asset, contract)
    model.load_state_dict(torch.load(args.wm/'ensemble.pt', weights_only=True, map_location='cpu'))
    geom = FullBodyGeometryV1(model.body.asset_sha256)
    geom.enable_depth_batches()
    shapes = Go2GroundClearance(geom)
    def surface(q, v):
        rot = rotation(q[:,3:7])
        fk = geom(q[:,7:],v[:,6:],q[:,:3],q[:,3:7],mv(rot,v[:,:3]),mv(rot,v[:,3:6]))
        return shapes(fk['position'], fk['rotation']).min(1)
    live = data['active']
    tick,lane = live.nonzero().T
    measured = torch.full(live.shape,float('inf'))
    source = torch.zeros_like(live,dtype=torch.long)
    for chunk in torch.arange(len(tick)).split(1024):
        t,l = tick[chunk], lane[chunk]
        gap,link = surface(*local_robot_state(data['after'][t,l]))
        measured[t,l],source[t,l]=gap,link
    event = data['body'] & live
    fault = event.any(0)
    n = len(live)
    clock = torch.arange(n)[:,None]
    first = torch.where(event,clock,n).amin(0)
    landing = torch.where(data['landed'] & data['live'],clock,n).amin(0)
    relaxed = torch.where(data['relaxed'] & live, clock,n).amin(0)
    root_only = data['done'] & (data['after'][...,2]+torch.tensor(manifest['origins'])[:,2] < .12)
    clean = ~((root_only | data['net'] | data['body']) & data['live']).any(0)
    profile=torch.tensor(contract['lane_profile'])
    report={}
    for split,mask in [('training',torch.arange(len(fault))%16>=2),('held',torch.arange(len(fault))%16<2)]:
        has_data=mask & live.any(0)
        actual=fault & has_data
        near=(measured<=.001).any(0)
        report[split]=dict(native_falls=int(actual.sum()), matched_onset=int((event & (measured<=.001))[:,has_data].sum()),
                           nonfall_cases=int((has_data & ~fault).sum()), nonfall_flagged=int((has_data & ~fault & near).sum()),
                           nonfall_entered_10mm=int((has_data & ~fault & (measured<.01).any(0)).sum()), events={})
        for label,times,take in [('early_body',first,actual & (first-landing<100)),
                                 ('late_body',first,actual & (first-landing>=100)),
                                 ('clean_first_relaxed',relaxed,clean & has_data & (relaxed<n))]:
            rows=take.nonzero().flatten()
            if not len(rows):
                continue
            t=times[rows]
            gap=measured[t,rows]
            ids,cnt=source[t,rows].unique(return_counts=True)
            bodies={}
            for i,k in zip(ids.tolist(),cnt.tolist()):
                name=shapes.names[i]
                bodies[name]=bodies.get(name,0)+k
            row=dict(cases=len(rows), closest_surfaces=bodies, median_mm=float(gap.median()*1000), forecasts={})
            for h in (4,8,16):
                start=t-h+1
                valid=(start>=0) & live[start.clamp_min(0),rows]
                s,l=start[valid],rows[valid]
                members=[]
                if not len(l):
                    row['forecasts'][h] = dict(cases=0, members=[])
                    continue
                for member in range(3):
                    q,v=local_robot_state(data['before'][s,l])
                    minimum=q.new_full((len(q),),float('inf'))
                    for i in range(h):
                        q,v,_=model(q,v,data['final_joint_targets'][s+i,l],member,profile[l])
                        pred,_=surface(q,v)
                        minimum=torch.minimum(minimum,pred)
                    members.append(dict(entered_10mm=int((minimum<.01).sum()), entered_1mm=int((minimum<=.001).sum()),
                                        endpoint_mae_mm=float((pred-measured[s+h-1,l]).abs().mean()*1000)))
                row['forecasts'][h]=dict(cases=len(l),members=members)
            report[split]['events'][label]=row
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('native','wm','asset','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    args.output.write_text(json.dumps(audit(args),indent=2)+'\n')
