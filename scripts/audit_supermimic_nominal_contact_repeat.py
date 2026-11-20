#!/usr/bin/env python3
"""Measure repeat noise and one-tick native momentum balance before scoring signed interventions."""
import argparse
import json
from pathlib import Path

import torch

from quadsmash.rl.supermimic_action_identification_dataset_v1 import _load_branch, _sha256
from quadsmash.rl.supermimic_native_pair_capture_v1 import tree_digest


def compare(folder):
    status = json.loads((folder.parent/'status.json').read_text())
    receipts = {row['branch']: row for row in status['completed'] if row['spec'] == folder.name}
    runs = {}
    hashes = {}
    for branch in ('nominal_a', 'nominal_b'):
        digest = _sha256(folder/branch/'native_probe.pt')
        if receipts[branch]['exit_code'] != 0 or receipts[branch]['probe_sha256'] != digest:
            raise ValueError(f'Invalid completed {branch} capture')
        hashes[branch] = digest
        runs[branch] = _load_branch(folder, branch)
    first, repeat = runs['nominal_a'], runs['nominal_b']
    if any(first[k] != repeat[k] for k in ('spec','task_contract','policy_execution')):
        raise ValueError('Nominal runs used different cases, physics or execution')
    if tree_digest(first['actor_state']) != tree_digest(repeat['actor_state']):
        raise ValueError('Different nominal policies')
    lanes = first['spec']['lanes']
    physics = json.loads((Path(__file__).resolve().parents[1]/'configs/shuttle_physics_v2.json').read_text())
    traces = {b: run['raw']['substep_trace'] for b, run in runs.items()}
    trace = traces['nominal_a']
    dt = physics['native_physics_dt_s']
    count = round(physics['policy_dt_s']/dt)+1
    if any(t['count'] != count or t['physics_dt_s'] != dt or
           t['contact_metadata']['friction_source'] != 'physx_filtered_racket_pair' for t in traces.values()):
        raise ValueError('A complete control tick of filtered native friction is required')

    def field(branch, key):
        return traces[branch]['tensors'][key][:, lanes]

    errors = {}
    for key in ('shuttle_state_w','racket_position_w','racket_quaternion_w','racket_friction_force_w',
                'racket_force_w','racket_contact_position_w','joint_position'):
        x, y = field('nominal_a',key), field('nominal_b',key)
        if not torch.equal(torch.isnan(x),torch.isnan(y)) or x.isinf().any() or y.isinf().any():
            raise ValueError(f'Nominal validity differs or infinite value in {key}')
        errors[key] = float((x.nan_to_num()-y.nan_to_num()).abs().max())

    vel = field('nominal_a','shuttle_state_w')[:,:,7:10].double()
    rel = vel[:-1] - vel.new_tensor(physics['wind_m_s'])
    free = (vel.new_tensor([0.,0.,-9.81])-physics['quadratic_drag_k_per_m']*rel.norm(dim=-1,keepdim=True)*rel)*dt
    def dv(key):
        return field('nominal_a',key)[1:].double().sum(0) * (dt/physics['mass_kg'])
    normal, friction = dv('racket_force_w'), dv('racket_friction_force_w')
    residual = vel[-1]-vel[0]-free.sum(0)-normal
    other = {k:float(field('nominal_a',k)[1:].abs().max()) for k in ('ground_force_w','net_force_matrix_w')}
    return dict(scope='Measured repeatability and force balance; signed action response evaluated separately',
                lanes=lanes, hashes=hashes, frames=count, physics_dt_s=dt, mass_kg=physics['mass_kg'],
                sensors=trace['contact_metadata'], max_repeat_abs_error=errors, other_normal_max_N=other,
                samples=[dict(lane=lane, normal_dv_mps=float(normal[i].norm()),
                              measured_friction_dv_mps=float(friction[i].norm()),
                              residual_without_friction_mps=float(residual[i].norm()),
                              residual_after_measured_friction_mps=float((residual[i]-friction[i]).norm()))
                         for i,lane in enumerate(lanes)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec-directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = compare(args.spec_directory)
    with args.output.open('x') as out:
        json.dump(result,out,indent=2,allow_nan=False)
        out.write('\n')
