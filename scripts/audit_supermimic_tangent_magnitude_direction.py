"""Separate first-impact friction magnitude and direction errors using recorded substep forces."""
import argparse
import json
from pathlib import Path

import torch

from quadsmash.rl.supermimic_action_identification_dataset_v1 import _load_branch
from quadsmash.rl.supermimic_ball_impulse_prior_v1 import BallImpulsePriorV1
from scripts.audit_supermimic_native_window_response import complete_window
from scripts.calibrate_supermimic_toi_slip import TOISlip
from scripts.evaluate_supermimic_contact_rollout import sha
from scripts.fit_supermimic_native_racket_toi_head import g07, free_rollout


def replace_friction(predicted, measured, mode):
    """Oracle replacements on a loaded racket tick, with no change to contact timing or normal impulse."""
    magnitude = torch.linalg.vector_norm(predicted, dim=-1, keepdim=True)
    measured_magnitude = torch.linalg.vector_norm(measured, dim=-1, keepdim=True)
    if mode == 'magnitude':
        return measured_magnitude * (predicted / magnitude.clamp_min(1e-12))
    if mode == 'direction':
        # Direction is undefined for a zero measurement. Preserve the baseline in that case.
        return torch.where(measured_magnitude > 1e-12, magnitude * (measured/measured_magnitude.clamp_min(1e-12)), predicted)
    if mode == 'both':
        return measured
    raise ValueError(mode)


class MeasuredFriction:
    def __init__(self, base, measured, mode):
        self.base = base
        self.measured = measured
        self.mode = mode

    def toi_response(self, row, present):
        info = self.base.toi_response(row, present)
        dv = replace_friction(info['impulse_residual_dv'], self.measured, self.mode)
        return dict(impulse_residual_active=info['impulse_residual_active'],
                    impulse_residual_dv=torch.where(info['impulse_residual_active'][:,None], dv, 0.))


def run(panel, prior, model, mass):
    _, _, noise, groups, sequences, execution = g07(panel, prior)
    completed = json.loads((panel/'status.json').read_text())['completed']
    folder = panel/completed[0]['spec']
    measured = {}
    for branch in ('minus','plus'):
        run = _load_branch(folder,branch)
        windows = complete_window(run['raw'],run['spec'],prior.steps)
        measured[branch] = []
        for win in windows:
            forces = win['tensors']['racket_friction_force_w'][1:]
            if win['physics_dt_s']!=prior.sub_dt or forces.shape!=(prior.steps,len(execution['lanes']),3) or not forces.isfinite().all():
                raise ValueError('Friction labels differ from the native lane order or physics clock')
            measured[branch].append(forces.sum(0)*(prior.sub_dt/mass))
    scores = {}
    baseline = None
    for mode in ('magnitude','direction','both'):
        result = free_rollout(model,prior,sequences,groups,noise,execution,baseline_model=model,
                         diagnostic_model_at_tick=lambda b,j:MeasuredFriction(model,measured[b][j],mode))
        if baseline is not None and baseline != result['baseline']:
            raise RuntimeError('Different baseline for paired component substitutions')
        baseline = result['baseline']
        scores[mode] = dict(metrics=result['candidate'],cases=result['cases'])
    return dict(status_sha256=sha(panel/'status.json'), baseline=baseline, substitutions=scores)


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('fit','calibration','training','development','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    read=lambda p:json.loads(p.read_text())
    physics=Path(__file__).resolve().parents[1]/'configs/shuttle_physics_v2.json'
    fit=read(args.fit)
    dev=read(args.development/'completed.json')
    if (sha(physics)!=fit['physics_sha256'] or sha(args.calibration)!=fit['calibration_sha256']
            or sha(args.training/'status.json')!=fit['g07_training']['status_sha256']
            or dev['wide']!=fit['g07_sha256']):
        raise ValueError('Calibration or case provenance differs')
    prior=BallImpulsePriorV1(normal_impulse_scale=read(args.calibration)['fitted']['normal_impulse_scale']).eval()
    model=TOISlip(fit['selected']['friction'],fit['selected']['sticking'])
    panels=dict(training=args.training,**{k:args.development/k for k in ('wide','half','quarter')})
    scores={}
    for name,panel in panels.items():
        if name!='training' and sha(panel/'status.json')!=dev[name]:
            raise ValueError('Development evidence differs')
        scores[name]=run(panel,prior,model,read(physics)['mass_kg'])
    if scores['wide']['baseline']!=fit['g07']['candidate']:
        raise ValueError('Calibrated baseline did not reproduce')
    with args.output.open('x') as stream:
        json.dump(dict(panels=scores,fit_sha256=sha(args.fit),actor_updates=0,v13_updates=0,
                       interpretation='Post-hoc native force labels. Oracle changes integrated friction at predicted racket TOI only. Three development scales share independent cases.'),
                  stream,indent=2,allow_nan=False)


if __name__=='__main__':
    main()
