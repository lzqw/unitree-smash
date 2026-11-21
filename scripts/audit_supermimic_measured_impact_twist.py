"""Teacher-only intervention on the incident racket velocity in signed native G07 windows."""
import argparse
import json
from pathlib import Path

import torch

from calibrate_supermimic_toi_slip import TOISlip
from evaluate_supermimic_contact_rollout import sha
from fit_supermimic_native_racket_toi_head import g07, metrics, predicted_state, prepare
from quadsmash.rl.supermimic_ball_impulse_prior_v1 import BallImpulsePriorV1


def incident_index(time_in_step, tick, dt, steps, first_load):
    """Force frame i+1 is produced by interval i; pose/twist i is still pre-impact."""
    predicted = tick*steps + torch.floor(time_in_step / dt).long().clamp(0, steps-1)
    return torch.minimum(predicted, first_load), predicted


@torch.no_grad()
def measure(prior, model, seqs, groups, noise, commands, *, measured):
    outputs, timing, target = {}, {}, {}
    for branch, sequence in seqs.items():
        state = sequence[0][0]
        lanes = torch.arange(len(state))
        linear, angular = (torch.cat([item[2][channel][:-1] for item in sequence]) for channel in (2,3))
        last = len(linear)-1
        first = torch.tensor([last if i is None else i for i in commands['first_native_loaded_interval'][branch]])
        used = torch.zeros(len(state),dtype=torch.bool)
        shifts = []
        for tick, (_, native_target, stream) in enumerate(sequence):
            row = prepare(prior, state, native_target, stream)
            impact = row['active'] & (row['object_id'] == 0)
            apply = impact & ~used
            used |= impact
            if measured and apply.any():
                index, predicted = incident_index(row['time'],tick,prior.sub_dt,prior.steps,first)
                # Keep the model's lever arm, geometry, normal, restitution and TOI fixed.
                # Recover the identical arm from the modeled contact center and shuttle position
                # in local coordinates under the same interpolated rotation as the deployed prior.
                from quadsmash.control.reactive_landing_v1 import quat_matrix
                alpha = (row['time']/prior.dt).clamp(0,1)
                rotation = prior._interpolate_rotation(quat_matrix(stream[1][0]), quat_matrix(stream[1][-1]), alpha[:,None])
                arm = (rotation @ row['local'][...,None]).squeeze(-1) - prior.preview.radius*row['normal']
                surface = linear[index,lanes] + torch.linalg.cross(angular[index,lanes],arm)
                relative = row['pre_velocity'] - surface
                speed = (relative*row['normal']).sum(-1)
                dv = -(1+prior.restitution)*prior.normal_impulse_scale*speed.clamp_max(0)[:,None]*row['normal']
                for name, updated in [('surface_velocity',surface), ('relative_velocity',relative), ('impulse_dv',dv)]:
                    row[name] = torch.where(apply[:,None], updated, row[name])
                shifts += [(commands['lanes'][i],int(predicted[i]),int(index[i])) for i in torch.where(apply)[0].tolist()]
            state = predicted_state(model, prior, row)
        outputs[branch] = state[:,3:]
        target[branch] = {'target':sequence[-1][1]}
        timing[branch] = shifts
    return dict(metrics=metrics((target, groups, noise[-len(groups):]), outputs), first_impact_frames=timing,
                signed_prediction_mps=(outputs['plus']-outputs['minus']).tolist())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ('fit','calibration','development','output'):
        parser.add_argument('--'+k,type=Path,required=True)
    a=parser.parse_args()
    torch.set_num_threads(2)
    fit=json.loads(a.fit.read_text())
    if fit['calibration_sha256'] != sha(a.calibration) or fit['g07_sha256'] != sha(a.development/'wide/status.json'):
        raise ValueError('Calibration provenance differs')
    prior=BallImpulsePriorV1(normal_impulse_scale=json.loads(a.calibration.read_text())['fitted']['normal_impulse_scale'])
    model=TOISlip(fit['selected']['friction'],fit['selected']['sticking'])
    scores={}
    hashes=json.loads((a.development/'completed.json').read_text())
    for scale in ('wide','half','quarter'):
        root=a.development/scale
        if sha(root/'status.json') != hashes[scale]:
            raise ValueError('Changed native branches')
        _,_,noise,groups,seq,cmds=g07(root,prior)
        if set(cmds['lanes']) & set(fit['g07_training']['commands']['lanes']):
            raise ValueError('Training overlap')
        base=measure(prior,model,seq,groups,noise,cmds,measured=False)
        if scale=='wide' and base['metrics'] != fit['g07']['candidate']:
            raise ValueError('Original deployable baseline does not reproduce')
        oracle=measure(prior,model,seq,groups,noise,cmds,measured=True)
        scores[scale]=dict(baseline=base, measured_incident_twist=oracle,lanes=cmds['lanes'],groups=groups)
    with a.output.open('x') as f:
        json.dump(dict(calibration_sha256=sha(a.fit),sources=hashes,scales=scores,optimizer_updates=0,
                       scope='First predicted racket impact only, using recorded pre-load twist. Native diagnosis only.'),f,indent=2,allow_nan=False)


if __name__=='__main__':
    main()
