#!/usr/bin/env python3
"""Use a collision-free native racket trajectory to predict the loaded fixture's ball response."""
import argparse
import json
from pathlib import Path

import torch
from quadsmash.rl.supermimic_fixture_coordinates import fixture_court_origin

from audit_supermimic_paired_racket_fixture import verify_initial_intervention, common_native_twist, verify_common_twist
from audit_supermimic_surface_derivative_scales import vector_summary, read_material
from evaluate_supermimic_contact_rollout import BallImpulsePriorV1, sha, tick


def preimpact_drive_preview(loaded, free):
    """At each measured loaded pose, add one unloaded substep of actuator-induced twist."""
    if len(loaded) != 4 or len(free) != 4 or len(loaded[0]) < 2:
        raise ValueError('Expected position, orientation, linear and angular velocity at substep boundaries')
    if any(x.shape != y.shape or x.shape[:2] != loaded[0].shape[:2]
           for x, y in zip(loaded, free, strict=True)):
        raise ValueError('Mismatched native racket timelines')
    return [loaded[0][:-1], loaded[1][:-1],
            loaded[2][:-1] + free[2].diff(dim=0), loaded[3][:-1] + free[3].diff(dim=0)]


@torch.no_grad()
def main():
    cli=argparse.ArgumentParser(description=__doc__)
    for key in ('loaded','unloaded','wide-audit','labels','calibration','output'):
        cli.add_argument('--'+key,type=Path,required=True)
    cli.add_argument('--compound-material',type=Path)
    cli.add_argument('--material-readout',type=Path)
    cli.add_argument('--drive-preview',action='store_true',
                     help='Also test the next unloaded drive increment at the measured incoming contact pose')
    a=cli.parse_args();torch.set_num_threads(2)
    if a.material_readout and not a.compound_material:cli.error('Source material missing')
    cfg=json.loads(a.calibration.read_text())
    wide=json.loads(a.wide_audit.read_text());labels=json.loads(a.labels.read_text())
    if (wide['split'] not in ('train','development') or labels['wide_audit_sha256']!=sha(a.wide_audit)
            or labels['calibration_sha256']!=sha(a.calibration)):
        raise ValueError('Different native cases or solver')
    rows=sorted(wide['samples'],key=lambda r:(r['seed'],r['lane']))
    if labels['case_ids']!=[[r['seed'],r['lane']] for r in rows]:raise ValueError('Mismatched case labels')
    prior=BallImpulsePriorV1().eval()
    prov=json.loads((a.loaded/'nominal_a/provenance.json').read_text())
    material=read_material(a.loaded,prov,a.compound_material,a.material_readout,a.calibration)
    mat_sha=sha(a.compound_material) if a.compound_material else None
    fit_sha=sha(a.material_readout) if a.material_readout else None
    if labels.get('material_sha256')!=mat_sha or labels.get('coefficient_fit_sha256')!=fit_sha:
        raise ValueError('Native derivative study and transfer must reference the same solver')
    branch_values={};loaded_values={};paths={}
    impulse_dv=[]
    drive_values={}
    keys=('racket_position_w','racket_quaternion_w','racket_linear_velocity_w','racket_angular_velocity_w')
    collector_hashes = {}
    for b in ('minus','plus','nominal_a'):
        lp,up=[json.loads((p/b/'provenance.json').read_text()) for p in (a.loaded,a.unloaded)]
        for source, provenance in [('loaded', lp), ('unloaded', up)]:
            if collector_hashes.setdefault(source, provenance['collector_sha256']) != provenance['collector_sha256']:
                raise ValueError('Collector changed within the paired branches')
        for key in ('teacher_sha256','trace_recorder_sha256','lanes','batches','seed'):
            if lp[key]!=up[key]:raise ValueError('Different setup: '+key)
        batch=[];loaded_batch=[];drive_batch=[]
        for seed in sorted({r['seed'] for r in rows}):
            load=a.loaded/b/f'spec_{seed}'/'trace.pt';unload=a.unloaded/b/f'spec_{seed}'/'trace.pt'
            if sha(load)!=wide['archives'][str(load)]:raise ValueError('Native target changed')
            paths[str(unload)]=sha(unload)
            orig,shadow=[torch.load(p,weights_only=True,mmap=True) for p in (load,unload)]
            if orig['schema']!='native-controlled-contact-trial-v1' or shadow['schema']!='native-unloaded-racket-oracle-v1':
                raise ValueError('Wrong fixture pairing')
            for x in (orig, shadow):
                intervention = x['velocity_intervention']
                if x['seed'] != seed or intervention['branch'] != b or x['parent_family'] != seed % 16:
                    raise ValueError('Native branch or launch family mismatch')
                verify_initial_intervention(x['raw']['substep_trace'], intervention)
            if [r['lane'] for r in rows if r['seed'] == seed] != list(range(lp['lanes'])):
                raise ValueError('Expected all registered lanes in physical row order')
            vo,vs=orig['velocity_intervention'],shadow['velocity_intervention']
            if not vs['unloaded_racket_oracle'] or vo.get('unloaded_racket_oracle'):
                raise ValueError('Expected the ball intervention only in the unloaded fixture')
            verify_common_twist(common_native_twist(vo),common_native_twist(vs))
            for k in ('nominal_shuttle','direction_world','joint_targets','nominal_root_com_velocity','delta_world'):
                torch.testing.assert_close(vo[k],vs[k],rtol=0,atol=0)
            torch.testing.assert_close(orig['parent_lane'],shadow['parent_lane'],rtol=0,atol=0)
            a0,a1=[x['raw']['substep_trace']['tensors'] for x in (orig,shadow)]
            for k in keys:torch.testing.assert_close(a0[k][0],a1[k][0],rtol=0,atol=0)
            for k in ('racket_force_w','racket_friction_force_w','net_force_matrix_w','ground_force_w'):
                if a1[k].norm(dim=-1).max()>.001:raise ValueError('Shadow racket saw external shuttle contact')
            s=a0['shuttle_state_w'][0];ball=torch.cat((s[:,:3],s[:,7:10]),-1)
            origin=fixture_court_origin(orig,ball[:,:3])
            torch.testing.assert_close(origin,fixture_court_origin(shadow,ball[:,:3]),rtol=0,atol=0)
            ball[:,:3]-=origin
            for fields,output in ((a1,batch),(a0,loaded_batch)):
                stream=[fields[k] for k in keys]
                stream[0]=stream[0]-origin
                speed,_=tick(prior,ball,[x[0] for x in stream],[x[-1] for x in stream],cfg['fitted'],
                    cfg['solver']['pair_contact_offset_m'],None if material is not None else torch.zeros(len(ball),dtype=torch.bool),
                    measured_path=[x[:-1] for x in stream],initial_spin=s[:,10:13] if material is not None else None,
                    compound_racket=material is not None,racket_material=material)
                output.append(speed[:,3:])
            if a.drive_preview:
                preview = preimpact_drive_preview([a0[k] for k in keys], [a1[k] for k in keys])
                preview[0] = preview[0]-origin
                out,_=tick(prior,ball,[v[0] for v in preview],[v[-1] for v in preview],cfg['fitted'],
                           cfg['solver']['pair_contact_offset_m'],None if material is not None else torch.zeros(len(ball),dtype=torch.bool),
                           measured_path=preview,initial_spin=s[:,10:13] if material is not None else None,
                           compound_racket=material is not None,racket_material=material)
                drive_batch.append(out[:,3:])
            if b=='nominal_a':
                impulse_dv.append((a0['racket_linear_velocity_w']-a1['racket_linear_velocity_w']).norm(dim=-1).amax(0))
        branch_values[b]=torch.cat(batch)
        loaded_values[b]=torch.cat(loaded_batch)
        if a.drive_preview:drive_values[b]=torch.cat(drive_batch)
    d=branch_values['plus']-branch_values['minus']
    n=torch.tensor([r['delta']['native'] for r in rows]);teacher=loaded_values['plus']-loaded_values['minus']
    groups={}
    for label,mask in [('all',torch.ones(len(rows),dtype=torch.bool)),('wide_stable',torch.tensor([r['kind']=='racket_stable' for r in rows])),
                      ('native_convergent',torch.tensor(labels['incident_labels']['native_convergent_stable']))]:
        if mask.any():
            impact=torch.cat(impulse_dv)[mask]
            groups[label]=dict(loaded=vector_summary(n[mask],teacher[mask]),unloaded=vector_summary(n[mask],d[mask]),
                              peak_racket_recoil_median_mps=float(impact.median()),peak_racket_recoil_p90_mps=float(impact.quantile(.9)))
            if a.drive_preview:
                drive=(drive_values['plus']-drive_values['minus'])[mask]
                groups[label]['loaded_with_unloaded_next_twist_increment']=vector_summary(n[mask],drive)
    with a.output.open('x') as out:
        json.dump(dict(groups=groups,split=wide['split'],unloaded_hashes=paths,collector_hashes=collector_hashes,loaded_audit_sha256=sha(a.wide_audit),
                  material_sha256=mat_sha,coefficient_fit_sha256=fit_sha,labels_sha256=sha(a.labels),model_calibration_sha256=sha(a.calibration),policy_updates=0,contact_bptt_admitted=False,
                  scope='Free articulated racket path supplied by separate native simulation; ball initial state and all outcomes refer to the loaded fixture'),out,indent=2,allow_nan=False)


if __name__=='__main__':main()
