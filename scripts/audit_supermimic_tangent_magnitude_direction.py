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
    def __init__(self, base, measured, mode, measured_normal=None):
        self.base = base
        self.measured = measured
        self.mode = mode
        self.measured_normal = measured_normal

    def toi_response(self, row, present):
        info = self.base.toi_response(row, present)
        if self.mode.startswith('normal_'):
            mode = self.mode.removeprefix('normal_')
            analytic = row['impulse_dv']
            # predicted_state adds analytic once before integrating the remaining time.
            normal_correction = replace_friction(analytic, self.measured_normal, mode) - analytic
            dv = normal_correction + info['impulse_residual_dv']
        elif self.mode=='all_loads':
            dv = self.measured_normal - row['impulse_dv'] + self.measured
        else:
            dv = replace_friction(info['impulse_residual_dv'], self.measured, self.mode)
        return dict(impulse_residual_active=info['impulse_residual_active'],
                    impulse_residual_dv=torch.where(info['impulse_residual_active'][:,None], dv, 0.))


def anchor_impulses(window, physics, friction, lanes):
    """Sum each native-geometry teacher over isolated racket interactions in this control tick."""
    from scripts.audit_supermimic_pilot_friction_anchors import inspect
    names=('native','radial_teacher','centroid_teacher','anchor_teacher')
    values={name:torch.zeros(lanes,3,dtype=torch.float64) for name in names}
    for event in inspect(window,physics,friction):
        for name,array in values.items():
            array[event['lane']]+=torch.as_tensor(event[name],dtype=array.dtype)
    return values


def run(panel, prior, model, mass, include_normal=False, *, anchor_physics=None):
    _, _, noise, groups, sequences, execution = g07(panel, prior)
    completed = json.loads((panel/'status.json').read_text())['completed']
    folder = panel/completed[0]['spec']
    measured = {}
    anchors = {}
    for branch in ('minus','plus'):
        run = _load_branch(folder,branch)
        windows = complete_window(run['raw'],run['spec'],prior.steps)
        measured[branch] = []
        anchors[branch] = []
        for win in windows:
            forces = win['tensors']['racket_friction_force_w'][1:]
            if win['physics_dt_s']!=prior.sub_dt or forces.shape!=(prior.steps,len(execution['lanes']),3) or not forces.isfinite().all():
                raise ValueError('Friction labels differ from the native lane order or physics clock')
            normal = win['tensors']['racket_force_w'][1:]
            if normal.shape!=forces.shape or not torch.isfinite(normal).all():
                raise ValueError('Invalid filtered normal force')
            if anchor_physics is not None:
                estimates=anchor_impulses(win,anchor_physics,model.friction,len(execution['lanes']))
                anchors[branch].append(estimates)
            measured[branch].append(tuple(x.sum(0)*(prior.sub_dt/mass) for x in (forces,normal)))
    scores = {}
    baseline = None
    modes = ('magnitude','direction','both') + (('normal_magnitude','normal_direction','normal_both','all_loads') if include_normal else ())
    for mode in modes:
        result = free_rollout(model,prior,sequences,groups,noise,execution,baseline_model=model,
                         diagnostic_model_at_tick=lambda b,j:MeasuredFriction(model,measured[b][j][0],mode,measured[b][j][1]))
        if baseline is not None and baseline != result['baseline']:
            raise RuntimeError('Different baseline for paired component substitutions')
        baseline = result['baseline']
        scores[mode] = dict(metrics=result['candidate'],cases=result['cases'])
    report=dict(status_sha256=sha(panel/'status.json'),baseline=baseline,substitutions=scores)
    if anchor_physics is not None:
        report['anchor_mechanics']={}
        # Preserve the measured tangential impulse from any simultaneous/nonisolated contacts
        # in every teacher arm. Only the isolated-racket contribution is exchanged.
        for method in ('radial_teacher','centroid_teacher','anchor_teacher'):
            def at_tick(branch,tick):
                obs=anchors[branch][tick]
                real,normal=measured[branch][tick]
                estimate=real+(obs[method]-obs['native']).to(real)
                return MeasuredFriction(model,estimate,'both',normal)
            result=free_rollout(model,prior,sequences,groups,noise,execution,baseline_model=model,
                                diagnostic_model_at_tick=at_tick)
            if result['baseline']!=baseline:
                raise ValueError('Baseline changed during measured-anchor comparison')
            report['anchor_mechanics'][method]=dict(metrics=result['candidate'],cases=result['cases'])
    return report


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('fit','calibration','training','development','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--new-training-support',type=Path,help='Score only this fresh signed training collection, bound to its support report')
    parser.add_argument('--include-normal',action='store_true',help='Also substitute measured normal load components when friction alone leaves a signed error')
    parser.add_argument('--anchor-mechanics',action='store_true',help='Test sphere inertia and native friction anchors at measured loading times')
    args=parser.parse_args()
    torch.set_num_threads(2)
    read=lambda p:json.loads(p.read_text())
    physics=Path(__file__).resolve().parents[1]/'configs/shuttle_physics_v2.json'
    fit=read(args.fit)
    dev=read(args.development/'completed.json')
    if (sha(physics)!=fit['physics_sha256'] or sha(args.calibration)!=fit['calibration_sha256']
            or dev['wide']!=fit['g07_sha256']):
        raise ValueError('Calibration or case provenance differs')
    prior=BallImpulsePriorV1(normal_impulse_scale=read(args.calibration)['fitted']['normal_impulse_scale']).eval()
    model=TOISlip(fit['selected']['friction'],fit['selected']['sticking'])
    panels=dict(training=args.training,**{k:args.development/k for k in ('wide','half','quarter')})
    if args.new_training_support:
        support=read(args.new_training_support)
        if (not support['training_only'] or support['policy_training_admitted'] or support['contact_receipt']
                or support['status_sha256']!=sha(args.training/'status.json') or support['physics_fit_sha256']!=sha(args.fit)):
            raise ValueError('Different fresh training panel')
        panels={'training':args.training}
    elif sha(args.training/'status.json')!=fit['g07_training']['status_sha256']:
        raise ValueError('Original training panel changed')
    scores={}
    for name,panel in panels.items():
        if name!='training' and sha(panel/'status.json')!=dev[name]:
            raise ValueError('Development evidence differs')
        scores[name]=run(panel,prior,model,read(physics)['mass_kg'],include_normal=args.include_normal,
                         anchor_physics=read(physics) if args.anchor_mechanics else None)
    expected=support['native_racket']['candidate'] if args.new_training_support else fit['g07']['candidate']
    reproduced=scores['training' if args.new_training_support else 'wide']['baseline']
    if reproduced!=expected:
        raise ValueError('Calibrated baseline did not reproduce')
    with args.output.open('x') as stream:
        json.dump(dict(panels=scores,fit_sha256=sha(args.fit),actor_updates=0,v13_updates=0,
                       interpretation='Post-hoc native force labels. Oracle changes integrated friction at predicted racket TOI only. Three development scales share independent cases.'),
                  stream,indent=2,allow_nan=False)


if __name__=='__main__':
    main()
