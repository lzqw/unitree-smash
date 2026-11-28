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
    pulse = 'pulse' in m
    schema = 'dynamimic-native-recovery-pulse-v1' if pulse else 'dynamimic-native-recovery-v1'
    b, candidate = read_data(args.after, ('active',), expected_schema=schema)
    if baseline['cases'] != candidate['cases'] or baseline['companion_sha256'] != candidate['companion_sha256']:
        raise ValueError('Different launches or Approach mixture')
    if pulse:
        if (m['incumbent'] != baseline['checkpoint_sha256'] or m['incumbent'] != candidate['checkpoint_sha256']
                or m['pulse'] != candidate['pulse']):
            raise ValueError('Native intervention and prefix audit disagree')
    elif (m['baseline'] != baseline['checkpoint_sha256'] or m['candidate'] != candidate['checkpoint_sha256']
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
    quantities = response
    if getattr(args, 'ground_surfaces', False):
        from quadsmash.rl.dynamimic_ground_clearance import Go2GroundClearance
        from quadsmash.rl.supermimic_full_body_geometry_v1 import FullBodyGeometryV1
        from quadsmash.rl.dynamimic_lagrangian_v1 import mv
        geometry = FullBodyGeometryV1(model.body.asset_sha256)
        geometry.enable_depth_batches()
        shapes = Go2GroundClearance(geometry)
        def quantities(q, v):
            rot = rotation(q[:,3:7])
            fk = geometry(q[:,7:],v[:,6:],q[:,:3],q[:,3:7],mv(rot,v[:,:3]),mv(rot,v[:,3:6]))
            gaps = shapes(fk['position'], fk['rotation'])
            return dict(go2_floor_gap_m=gaps.amin(-1), left_rear_thigh_gap_m=gaps[:,shapes.left_rear_thigh])

    lanes = torch.arange(len(baseline['cases']))
    if pulse:
        from scripts.dynamimic_intervention_timing import pulse_ticks
        start = pulse_ticks(m['pulse'],len(lanes)).clamp_min(0)
        eligible = set(m['eligible_case_ids'])
        same = torch.tensor([row['case_id'] in eligible for row in baseline['cases']])
        if int(same.sum()) != m['eligible'] or len(eligible) != m['eligible']:
            raise ValueError('Native prefix audit case identities differ')
        horizons = [h for h in (4,8,16) if h <= m['pulse']['duration_ticks']]
    else:
        ta, tb = [first_tick(d['active']) for d in (a,b)]
        start = ta.clamp_max(min(len(a['active']), len(b['active']))-1)
        same = (a['before'][start,lanes] == b['before'][start,lanes]).all(-1)
        same &= torch.tensor([all(r['prefix_max_abs_'+k] == 0 for k in ('before','request','final_joint_targets'))
                              for r in m['cases']])
        same &= (ta == tb) & a['active'][start,lanes] & b['active'][start,lanes]
        horizons = (4,8,16,32,48)
    out = dict(exact_prefix_cases=int(same.sum()), total_cases=len(lanes), horizons={})
    for h in horizons:
        valid = same & (start+h <= min(len(a['active']), len(b['active'])))
        times = (start[None,:] + torch.arange(h)[:,None]).clamp_max(min(len(a['active']),len(b['active']))-1)
        valid &= (a['active'][times,lanes] & b['active'][times,lanes]).all(0)
        l = lanes[valid]; t = start[l]
        if not len(l):
            out['horizons'][h] = dict(cases=0)
            continue
        profile = torch.tensor(contract['lane_profile'])[l]
        q0,v0 = local_robot_state(a['before'][t,l])
        measured = [quantities(*local_robot_state(d['after'][t+h-1,l])) for d in (a,b)]
        truth = {k:measured[1][k]-measured[0][k] for k in measured[0]}
        members = []
        for member in range(3):
            branches = []
            for data in (a,b):
                q,v = q0,v0
                for i in range(h):
                    q,v,_ = model(q,v,data['final_joint_targets'][t+i,l],member,profile)
                branches.append(quantities(q,v))
            predicted = {k:branches[1][k]-x for k,x in branches[0].items()}
            groups = {}
            for group, sel in [('train',l%16>=4), ('held',l%16<2)]:
                if not sel.any():
                    continue
                groups[group] = {}
                for key, x in predicted.items():
                    result = metrics(x[sel], truth[key][sel])
                    if getattr(args, 'ground_surfaces', False):
                        material = sel & (truth[key].abs() > .001)
                        result.update(measured_over_1mm=int(material.sum()),
                                      sign_correct_over_1mm=int((x[material]*truth[key][material] > 0).sum()))
                    groups[group][key] = result
            members.append(groups)
        out['horizons'][h] = dict(cases=len(l), members=members)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('before','after','pairs','wm','asset','output'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--ground-surfaces', action='store_true')
    args = parser.parse_args()
    args.output.write_text(json.dumps(score(args), indent=2)+'\n')
