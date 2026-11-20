#!/usr/bin/env python3
"""Compare racket paths on native misses selected as stable strikes by the frozen surface model."""
import argparse
import json
from pathlib import Path

import torch

from audit_supermimic_predictive_surface_basin import motion_from_initial, read_material
from evaluate_supermimic_contact_rollout import BallImpulsePriorV1, sha, tick
from fit_supermimic_unloaded_racket_motion import RACKET_FIELDS, motion_features, predict_motion
from quadsmash.rl.supermimic_ball_geometry_preview_v1 import box_gap, round_gap
from quadsmash.rl.supermimic_contact_pilot_v1 import _quat_rotate_inverse
from quadsmash.rl.supermimic_fixture_coordinates import fixture_court_origin


def ball_surface_gap(g, position, center, rotation):
    local=_quat_rotate_inverse(rotation,position-center)
    box=box_gap(local,g.racket_box_centers,g.racket_box_halves)
    rounded=round_gap(local,g.racket_round_centers,g.racket_round_axes,g.racket_round_halves,
                      g.racket_round_radii,g.racket_round_capsule)
    return torch.cat((box,rounded),-1).amin(-1)-g.radius


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('fixture','audit','labels','selection','motion','material','readout','calibration','output'):
        p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args()
    torch.set_num_threads(2)
    read=lambda path:json.loads(path.read_text())
    audit,labels,selection=map(read,(a.audit,a.labels,a.selection))
    if (audit['split'] not in ('train','development') or sha(a.audit)!=labels['wide_audit_sha256']
            or sha(a.labels)!=selection['labels_sha256'] or sha(a.calibration)!=selection['calibration_sha256']
            or selection['material_sha256']!=sha(a.material) or selection['coefficient_fit_sha256']!=sha(a.readout)
            or selection['motion_readout_sha256'] is not None):
        raise ValueError('Expected constant-twist case selection on this frozen calibration or development panel')
    group=selection['groups']['predicted_cross_solver_consensus_single_shape']
    chosen=set(map(tuple,group['selected_case_ids']))
    misses=[r for r in audit['samples'] if r['kind']=='no_contact' and (r['seed'],r['lane']) in chosen]
    if len(misses)!=group['false_positive_no_native_racket']:
        raise ValueError('Native miss identities do not match the reported selection')
    cfg=read(a.calibration)
    prior=BallImpulsePriorV1().eval()
    times=torch.arange(prior.steps)*prior.sub_dt
    prov=read(a.fixture/'nominal_a/provenance.json')
    table=read_material(a.fixture,prov,a.material,a.readout,a.calibration)
    motion=torch.load(a.motion,weights_only=True)
    teacher_path=Path(prov['args'][prov['args'].index('--teacher')+1])
    if (motion['schema']!='unloaded-articulation-motion-readout-v1' or motion['design_sha256']!=sha(a.fixture/'design.json')
            or motion['teacher_sha256']!=sha(teacher_path) or motion['physics_dt_s']!=prior.sub_dt):
        raise ValueError('Unloaded motion model is not bound to this fixture')
    teacher=torch.load(teacher_path,weights_only=True,mmap=True)
    results=[]
    for case in misses:
        lane=case['lane']
        for branch in ('nominal_a','nominal_b','minus','plus'):
            path=a.fixture/branch/f"spec_{case['seed']}"/'trace.pt'
            if sha(path)!=audit['archives'][str(path)]:raise ValueError('Changed native branch')
            packet=torch.load(path,weights_only=True,mmap=True)
            native=packet['raw']['substep_trace']['tensors']
            stream=[native[k][:,lane:lane+1] for k in RACKET_FIELDS]
            shuttle=native['shuttle_state_w'][:,lane:lane+1]
            origin=fixture_court_origin(packet,native['shuttle_state_w'][0,:,:3])[lane:lane+1]
            stream[0]=stream[0]-origin
            actual_position=shuttle[:,:,:3]-origin
            ball=torch.cat((actual_position[0],shuttle[0,:,7:10]),-1)
            start=[v[0] for v in stream]
            motion_input=motion_features(packet,teacher)[lane:lane+1]
            paths=dict(constant_twist=motion_from_initial(*start,times),measured=[v[:-1] for v in stream],
                       learned_unloaded=predict_motion(motion,motion_input,start,times))
            peak=float(native['racket_force_w'][:,lane].norm(dim=-1).max())
            if peak>.001:raise ValueError('Native miss had a loaded racket contact')
            entry=dict(seed=case['seed'],lane=lane,branch=branch,native_racket_peak_n=peak)
            for method,preview in paths.items():
                events={}
                predicted,_=tick(prior,ball,start,[v[-1] for v in stream],cfg['fitted'],cfg['solver']['pair_contact_offset_m'],
                                 measured_path=preview,event_trace=events,racket_material=table,initial_spin=shuttle[0,:,10:13],compound_racket=True)
                hits=(events['racket'][:,0]>=0).nonzero().flatten()
                gap=ball_surface_gap(prior.preview,actual_position[:-1,0],preview[0][:,0],preview[1][:,0])
                entry[method]=dict(predicted_hit_substeps=hits.tolist(),min_gap_on_measured_ball_m=float(gap.min()),
                                   final_ball_velocity_error_mps=(predicted[:,3:]-shuttle[-1,:,7:10])[0].tolist())
            results.append(entry)
    with a.output.open('x') as f:
        json.dump(dict(cases=results,selected_misses=len(misses),selection_sha256=sha(a.selection),
                       motion_sha256=sha(a.motion),scope='Native no-contact cases among model-selected development or calibration strikes',
                       contact_gradients_admitted=False),f,allow_nan=False,indent=2)
        f.write('\n')


if __name__=='__main__':main()
