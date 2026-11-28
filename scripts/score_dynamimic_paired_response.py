"""Score the native body response to the executed Recovery update from identical release states."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from scripts.compare_dynamimic_native_behavior import first_tick
from scripts.fit_dynamimic_recovery_wm import DynaMimicRecoveryWorldModel, local_robot_state, read_data, rotation, trunk_innovation


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
    timed = pulse and 'start_ticks' in m['pulse']
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
        geometry = FullBodyGeometryV1(model.body.asset_sha256)
        geometry.enable_depth_batches()
        shapes = Go2GroundClearance(geometry)
        def quantities(q, v):
            gaps = shapes.monitored_gaps(geometry, q, v)
            result = dict(go2_floor_gap_m=gaps[:,0], left_rear_thigh_gap_m=gaps[:,1])
            if timed:
                result.update(response(q, v))
            return result

    air_audit = getattr(args, 'air_preparation', False)
    if air_audit:
        if pulse or getattr(args, 'ground_surfaces', False):
            raise ValueError('Air preparation audit uses a complete paired policy rollout')
        from quadsmash.rl.supermimic_full_body_geometry_v1 import FullBodyGeometryV1
        from quadsmash.rl.dynamimic_lagrangian_v1 import mv
        geometry = FullBodyGeometryV1(model.body.asset_sha256)
        geometry.enable_depth_batches()
        def quantities(q, v):
            rot = rotation(q[:,3:7])
            feet = geometry(q[:,7:],v[:,6:],q[:,:3],q[:,3:7],mv(rot,v[:,:3]),mv(rot,v[:,3:6]))['feet']
            return dict(**response(q,v), rear_minus_front_height=(
                feet[:,2:,2].mean(1) - feet[:,:2,2].mean(1)).unsqueeze(-1))

    lanes = torch.arange(len(baseline['cases']))
    if pulse:
        from scripts.dynamimic_intervention_timing import pulse_ticks
        start = pulse_ticks(m['pulse'],len(lanes)).clamp_min(0)
        eligible = set(m['eligible_case_ids'])
        same = torch.tensor([row['case_id'] in eligible for row in baseline['cases']])
        if int(same.sum()) != m['eligible'] or len(eligible) != m['eligible']:
            raise ValueError('Native prefix audit case identities differ')
        horizons = [h for h in (4,8,16) if h <= m['pulse']['duration_ticks']]
        if timed:
            # The policy switches back to the incumbent after 16 ticks. Each branch
            # then uses its own recorded closed-loop commands up to the nominal fall.
            horizons.append(m['pulse']['duration_ticks'] + 16)
    else:
        ta, tb = [first_tick(d['active']) for d in (a,b)]
        start = ta.clamp_max(min(len(a['active']), len(b['active']))-1)
        same = (a['before'][start,lanes] == b['before'][start,lanes]).all(-1)
        same &= torch.tensor([all(r['prefix_max_abs_'+k] == 0 for k in ('before','request','final_joint_targets'))
                              for r in m['cases']])
        same &= (ta == tb) & a['active'][start,lanes] & b['active'][start,lanes]
        horizons = (4,8,16,24,32,48) if air_audit else (4,8,16,32,48)
    out = dict(exact_prefix_cases=int(same.sum()), total_cases=len(lanes), horizons={})
    baseline_fall = (a['active'] & a['body']).any(0)
    for h in horizons:
        valid = same & (start+h <= min(len(a['active']), len(b['active'])))
        times = (start[None,:] + torch.arange(h)[:,None]).clamp_max(min(len(a['active']),len(b['active']))-1)
        valid &= (a['active'][times,lanes] & b['active'][times,lanes]).all(0)
        l = lanes[valid]; t = start[l]
        coverage = {}
        if timed:
            held = lanes % 16 < 2
            for label, group in [('held_prefall', held & baseline_fall), ('held_control', held & ~baseline_fall)]:
                coverage[label] = dict(matched=int((same & group).sum()), active_both=int((valid & group).sum()))
        if not len(l):
            out['horizons'][h] = dict(cases=0, **({'coverage': coverage} if timed else {}))
            continue
        profile = torch.tensor(contract['lane_profile'])[l]
        q0,v0 = local_robot_state(a['before'][t,l])
        measured = [quantities(*local_robot_state(d['after'][t+h-1,l])) for d in (a,b)]
        truth = {k:measured[1][k]-measured[0][k] for k in measured[0]}
        splits = {'train': l % 16 >= 4, 'held': l % 16 < 2}
        if air_audit:
            # Native landed is latched after the control step. Partition on the
            # baseline outcome to avoid selecting for candidate-induced changes.
            airborne = (t > 0) & ~a['landed'][(t-1).clamp_min(0),l]
            touched = a['landed'][t+h-1,l].bool()
            for split, mask in tuple(splits.items()):
                splits[split+'_airborne'] = mask & airborne & ~touched
                splits[split+'_crossing'] = mask & airborne & touched
        if timed:
            # Use only the incumbent outcome to identify the scheduled window class.
            for split, mask in tuple(splits.items()):
                splits[split+'_prefall'] = mask & baseline_fall[l]
                splits[split+'_control'] = mask & ~baseline_fall[l]
        support_audit = timed and getattr(args, 'support_audit', False)
        if support_audit:
            # Compare the four observed contacts at every control boundary.
            # Equal masks do not imply equal substep impulses.
            feet_a, feet_b = [d['feet'][times[:, l], l].bool() for d in (a, b)]
            changed = (feet_a != feet_b).any(-1).any(0)
            for partition in ('train_prefall', 'held_prefall'):
                splits[partition+'_changed_support'] = splits[partition] & changed
                splits[partition+'_same_support'] = splits[partition] & ~changed
        members = []
        for member in range(3):
            branches = []
            for data in (a,b):
                q,v = q0,v0
                for i in range(h):
                    q,v,_ = model(q,v,data['final_joint_targets'][t+i,l],member,profile)
                branches.append(quantities(q,v))
            predicted = {k:branches[1][k]-x for k,x in branches[0].items()}
            if support_audit:
                # Remove pre-existing branch differences. Scoring only endpoint
                # velocities after re-anchoring would give credit for inherited motion.
                increments = [trunk_innovation(model, d, t+h, l, profile, member) for d in (a,b)]
                for name, sl in [('linear',slice(0,3)), ('angular',slice(3,6))]:
                    key = 'last4_world_'+name+'_innovation'
                    predicted[key] = increments[1][0][:,sl] - increments[0][0][:,sl]
                    truth[key] = increments[1][1][:,sl] - increments[0][1][:,sl]
            groups = {}
            for group, sel in splits.items():
                if not sel.any():
                    continue
                groups[group] = {}
                for key, x in predicted.items():
                    result = metrics(x[sel], truth[key][sel])
                    if key == 'rear_minus_front_height':
                        for cutoff_mm in (.1,1.):
                            material = sel & (truth[key].abs().flatten() > cutoff_mm*.001)
                            result[f'over_{cutoff_mm:g}mm'] = int(material.sum())
                            result[f'correct_sign_over_{cutoff_mm:g}mm'] = int((x[material]*truth[key][material] > 0).sum())
                    if key in ('go2_floor_gap_m', 'left_rear_thigh_gap_m'):
                        material = sel & (truth[key].abs() > .001)
                        result.update(measured_over_1mm=int(material.sum()),
                                      sign_correct_over_1mm=int((x[material]*truth[key][material] > 0).sum()))
                    groups[group][key] = result
            members.append(groups)
        out['horizons'][h] = dict(cases=len(l), members=members)
        if timed:
            out['horizons'][h]['coverage'] = coverage
            out['horizons'][h]['incumbent_continuation_ticks'] = max(0, h-m['pulse']['duration_ticks'])
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('before','after','pairs','wm','asset','output'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--ground-surfaces', action='store_true')
    parser.add_argument('--air-preparation', action='store_true')
    parser.add_argument('--support-audit', action='store_true',
                        help='Split timed pre-fall responses by observed foot contact sequence')
    args = parser.parse_args()
    args.output.write_text(json.dumps(score(args), indent=2)+'\n')
