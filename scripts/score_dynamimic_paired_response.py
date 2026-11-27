"""Score the native body response to the executed Recovery update from identical release states."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from scripts.compare_dynamimic_native_behavior import first_tick
from scripts.fit_dynamimic_recovery_wm import DynaMimicRecoveryWorldModel, local_robot_state, read_data, rotation


def response(q, v):
    return {'gravity_direction': rotation(q[:, 3:7])[:, 2],
            'root_velocity_local': v[:, :3], 'angular_velocity_local': v[:, 3:6],
            'joint_position': q[:, 7:]}


def metrics(pred, truth):
    # Dot products aggregate signal energy without dividing by tiny per-case responses.
    energy = truth.square().sum()
    size = pred.square().sum()
    dot = (pred * truth).sum()
    return dict(cases=len(pred), native_rms=float(truth.square().mean().sqrt()),
                rmse=float((pred-truth).square().mean().sqrt()),
                cosine=float(dot/(energy*size).sqrt()) if energy*size > 0 else None,
                projected_gain=float(dot/energy) if energy > 0 else None)


@torch.no_grad()
def score(args):
    m = json.loads(args.pairs.read_text())
    a, baseline = read_data(args.before, ('active',))
    b, candidate = read_data(args.after, ('active',))
    if (m['baseline'] != baseline['checkpoint_sha256'] or m['candidate'] != candidate['checkpoint_sha256']
            or [r['case_id'] for r in m['cases']] != [r['case_id'] for r in baseline['cases']]
            or baseline['cases'] != candidate['cases']):
        raise ValueError('Paired audit and native archives disagree')
    fit = json.loads((args.wm/'validation.json').read_text())
    weights = args.wm/'ensemble.pt'
    if (not fit['passed'] or hashlib.sha256(weights.read_bytes()).hexdigest() != fit['ensemble_sha256']
            or fit['native_checkpoint'] != baseline['checkpoint_sha256']):
        raise ValueError('Expected the validated ensemble from the baseline behavior')
    contract = json.loads((args.before/'actuator_contract.json').read_text())
    model = DynaMimicRecoveryWorldModel(args.asset, contract)
    model.load_state_dict(torch.load(weights, weights_only=True, map_location='cpu'))
    model.eval()
    ta, tb = [first_tick(d['active']) for d in (a,b)]
    lanes = torch.arange(len(ta))
    start = ta.clamp_max(min(len(a['active']), len(b['active']))-1)
    same = (a['before'][start,lanes] == b['before'][start,lanes]).all(-1)
    same &= torch.tensor([all(r['prefix_max_abs_'+k] == 0 for k in ('before','request','final_joint_targets'))
                          for r in m['cases']])
    same &= (ta == tb) & a['active'][start,lanes] & b['active'][start,lanes]
    out = dict(exact_prefix_cases=int(same.sum()), total_cases=len(lanes), horizons={})
    for h in (4,8,16,32,48):
        valid = same & (start+h <= min(len(a['active']), len(b['active'])))
        times = (start[None,:] + torch.arange(h)[:,None]).clamp_max(min(len(a['active']),len(b['active']))-1)
        valid &= (a['active'][times,lanes] & b['active'][times,lanes]).all(0)
        l = lanes[valid]; t = start[l]
        if not len(l):
            out['horizons'][h] = dict(cases=0)
            continue
        profile = torch.tensor(contract['lane_profile'])[l]
        q0,v0 = local_robot_state(a['before'][t,l])
        measured = [response(*local_robot_state(d['after'][t+h-1,l])) for d in (a,b)]
        truth = {k:measured[1][k]-measured[0][k] for k in measured[0]}
        members = []
        for member in range(3):
            branches = []
            for data in (a,b):
                q,v = q0,v0
                for i in range(h):
                    q,v,_ = model(q,v,data['final_joint_targets'][t+i,l],member,profile)
                branches.append(response(q,v))
            predicted = {k:branches[1][k]-x for k,x in branches[0].items()}
            members.append({group:{k:metrics(x[sel],truth[k][sel]) for k,x in predicted.items()}
                            for group,sel in [('train',l%16>=4),('held',l%16<2)] if sel.any()})
        out['horizons'][h] = dict(cases=len(l), members=members)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('before','after','pairs','wm','asset','output'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(score(args), indent=2)+'\n')
