"""Fit one bounded slip correction for the predicted gap at the first discrete contact-offset sample."""
import argparse
import json
from pathlib import Path

import torch

from calibrate_supermimic_toi_slip import (
    TOISlip, body_transfer, evaluate, friction_errors, integrated_friction, signed_training_capture,
)
from evaluate_supermimic_contact_rollout import sha
from fit_supermimic_native_racket_toi_head import fixture, free_rollout, g07
from quadsmash.rl.supermimic_ball_impulse_prior_v1 import BallImpulsePriorV1


def onset_separation(row, offset, dt):
    """Local incident velocity approximation. No observed force, gap, spin or outcome is used."""
    closing = (-row['normal_speed']).clamp_min(0.)
    toi = row['time']
    crossing = (toi - offset / closing.clamp_min(1e-8)).clamp_min(0.)
    sampled = torch.ceil(crossing / dt) * dt
    gap = closing * (toi - sampled)
    return torch.where((closing > 1e-8) & (toi > 0), gap, torch.zeros_like(gap))


class SeparationSlip(TOISlip):
    def __init__(self, friction, sticking, slope, offset, dt):
        super().__init__(friction, sticking)
        self.slope = slope
        self.offset = offset
        self.dt = dt

    def toi_response(self, row, present):
        normal = row['normal']
        velocity = row['relative_velocity']
        slip = velocity - (velocity * normal).sum(-1, keepdim=True) * normal
        separation = onset_separation(row, self.offset, self.dt)
        stick = self.sticking * (1 + self.slope * (separation / self.offset).clamp(-1., 1.))
        cap = self.friction * row['impulse_dv'].norm(dim=-1)
        scale = torch.minimum(stick, cap / slip.norm(dim=-1).clamp_min(1e-12))
        active = row['active'] & (row['object_id'] == 0) & present.reshape(-1).bool()
        return dict(impulse_residual_active=active,
                    impulse_residual_dv=-slip * (active * scale)[:, None])


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for k in ('fit','calibration','fixture','training','development','body','online-body','stage','output'):
        parser.add_argument('--'+k, required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    read = lambda f: json.loads(f.read_text())
    fit = read(args.fit)
    physics = Path(__file__).resolve().parents[1] / 'configs/shuttle_physics_v2.json'
    ids = read(args.development / 'completed.json')
    if (sha(args.calibration) != fit['calibration_sha256'] or sha(physics) != fit['physics_sha256']
            or fit['g07_sha256'] != ids['wide'] or fit['g07_training']['status_sha256'] != sha(args.training/'status.json')
            or any(sha(args.fixture/(s+'.json')) != fit['fixture_sha256'][s] for s in ('train','development'))):
        raise ValueError('Calibration or training source changed')
    cfg = read(args.calibration)
    prior = BallImpulsePriorV1(normal_impulse_scale=cfg['fitted']['normal_impulse_scale']).eval()
    offset = cfg['solver']['pair_contact_offset_m']
    base = TOISlip(fit['selected']['friction'],fit['selected']['sticking'])
    def candidate(slope):
        return SeparationSlip(base.friction,base.sticking,slope,offset,prior.sub_dt)
    data = fixture(args.fixture,'train',prior)
    labels = integrated_friction(args.fixture,'train',read(physics),prior)
    signed,forces,identity = signed_training_capture(args.training,args.development/'wide',prior,read(physics))
    sources = [(data,labels),(signed,forces)]
    refs = [friction_errors(base,d,v) for d,v in sources]
    grid = []
    for slope in (0.,-.5,-.25,-.125,-.0625,.0625,.125,.25,.5):
        errors = [friction_errors(candidate(slope),d,v) for d,v in sources]
        ratios = [[row[k]/ref[k] for k in ref if ref[k] and ref[k]>0] for row,ref in zip(errors,refs,strict=True)]
        grid.append(dict(slope=slope,score=sum(sum(r)/len(r) for r in ratios)/len(ratios),sources=errors))
    if abs(grid[0]['score'] - 1.) > 1e-6:
        raise ValueError('Zero correction differs from the baseline')
    best = min(grid,key=lambda row:row['score'])
    model = candidate(best['slope'])
    dev = fixture(args.fixture,'development',prior)
    fixture_score = {k:evaluate(prior,m,dev) for k,m in [('baseline',base),('candidate',model)]}
    results = {}
    initial = None
    for key,divisor in [('wide',1),('half',2),('quarter',4)]:
        root = args.development / key
        if sha(root/'status.json') != ids[key]:
            raise ValueError('Development source differs')
        startup = read(root/'startup.json')
        spec = startup['specs'][0]['spec'].copy()
        signature = (startup['source_sha256'],startup['seed'],spec.pop('epsilon_request')*divisor,spec)
        if initial is None:
            initial = signature
        if initial!=signature:
            raise ValueError('Source, direction or held cases differ across scales')
        _,_,noise,groups,seq,commands = g07(root,prior)
        report = free_rollout(model,prior,seq,groups,noise,commands,baseline_model=base)
        body = body_transfer(root,args.body,prior,model,groups,noise,seq,commands,args.online_body,args.stage)
        if body['replay_world_sha256'] != fit['online_body_racket']['replay_world_sha256']:
            raise ValueError('Body validation source changed')
        if key=='wide' and report['baseline']!=fit['g07']['candidate']:
            raise ValueError('Native baseline no longer reproduced')
        results[key] = dict(native_racket=report,body=body)
    result = dict(physics_fit_sha256=sha(args.fit),sources=identity,development_status_sha256=ids,
                  calibration_sha256=sha(args.calibration),pair_contact_offset_m=offset,selected=best,grid=grid,
                  fixture_development=fixture_score,scales=results,final_holdout_opened=False,
                  actor_updates=0,v13_updates=0,scope='Single-impact forward diagnostic with a predicted discrete onset separation; no gradient admission.')
    with args.output.open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False)


if __name__=='__main__':
    main()
