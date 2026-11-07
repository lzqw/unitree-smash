"""Audit a specific omitted self-collision candidate in native/model trajectories.

Authored primitive overlap is separate from native contact-force evidence.
No filter, asset, policy, physics parameter or old acceptance result is changed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.authored_hip_mount_clearance_v1 import AuthoredHipMountClearanceV1


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh audit required')
    read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    bundle=ROOT/'reports/landing_early_support_bundle_20261108/manifest.json';manifest=read(bundle)
    controller=build_controller(manifest['configuration']);model=controller.qp.model
    geometry=ROOT/'configs/go2_piper_authored_ground_geometry_v1.json';checker=AuthoredHipMountClearanceV1(geometry,model)
    run=ROOT/'reports/native_early_support_all128_20261108_02';trace=run/'sequence_native_trace.npz'
    finepath=ROOT/'reports/landing_early_references_20261108/refine_support_tail_fine160_v1.json'
    searchpath=ROOT/'reports/landing_early_references_20261108/preserved_early_support_search_v1.json'
    fine=read(finepath);search=read(searchpath)
    with np.load(trace,allow_pickle=False) as data:native_q=data['q'].copy()
    traces=[('native_early_reference',native_q,280),
        ('original_model_nominal',np.asarray([r['q'] for r in fine['independent_replay']['trajectory']]),281)]
    for row in search['final_checks']:
        if row['result']['accepted']:
            traces.append(('refined_model_'+row['probe']['name'],np.asarray([r['q'] for r in row['result']['trajectory']]),281))
    paths=[bundle,geometry,trace,run/'job.json',finepath,searchpath,Path(__file__),ROOT/'scripts/authored_hip_mount_clearance_v1.py']
    hashes=dict(manifest['source_sha256']);hashes.update({str(f):sha(f) for f in paths})
    for f,h in hashes.items():
        if sha(f)!=h:raise ValueError('Changed source '+f)
    summaries=[];detail=[]
    for name,qs,start in traces:
        overlaps=[];minima={h:None for h in checker.hips}
        for offset,q in enumerate(qs):
            rows=checker.evaluate(q)
            for row in rows:
                g=row['radial_gap_m']
                if g is not None and (minima[row['hip']] is None or g<minima[row['hip']]):minima[row['hip']]=g
                if g is not None and g < -1e-6:
                    overlaps.append(dict(tick=start+offset,hip=row['hip'],radial_gap_m=g))
            if name=='native_early_reference' and 350<=start+offset<=380:
                detail.append(dict(tick=start+offset,hip_angles_rad=q[model.q_indices[:4]].tolist(),pairs=rows))
        summaries.append(dict(name=name,state_samples=len(qs),sampled_overlap_count=len(overlaps),
            first_sampled_overlap=overlaps[0] if overlaps else None,
            first_by_hip={h:next((r for r in overlaps if r['hip']==h),None) for h in checker.hips},
            minimum_radial_gap_in_axial_slab_m=minima))
    # The mount is rigidly attached to the base, and each hip collider moves
    # with its hip joint. Verify numerical invariance on actual configurations
    # before expressing the observed authored boundary in one joint coordinate.
    q0=native_q[0].copy();boundaries=[]
    for j,hip in enumerate(checker.hips):
        qi=model.q_indices[j];errors=[]
        for actual in native_q[::37]:
            fixed=q0.copy();fixed[qi]=actual[qi]
            x=checker.evaluate(actual,[hip])[0];y=checker.evaluate(fixed,[hip])[0]
            if (x['radial_gap_m'] is None)!=(y['radial_gap_m'] is None):raise ValueError('Unexpected multijoint dependence')
            errors.append(abs((x['radial_gap_m'] if x['radial_gap_m'] is not None else x['axis_separation_m'])-
                              (y['radial_gap_m'] if y['radial_gap_m'] is not None else y['axis_separation_m'])))
        if max(errors)>1e-10:raise ValueError('Single-joint clearance representation invalid')
        def intersect(angle):
            q=q0.copy();q[qi]=angle;row=checker.evaluate(q,[hip])[0]
            return row['radial_gap_m'] is not None and row['radial_gap_m']<=0.
        grid=np.linspace(*controller.qp.interface.limits[j],201);flags=[intersect(x) for x in grid];roots=[]
        for i in range(len(grid)-1):
            if flags[i]==flags[i+1]:continue
            low,high=float(grid[i]),float(grid[i+1]);left=flags[i]
            for _ in range(32):
                mid=(low+high)/2
                if intersect(mid)==left:low=mid
                else:high=mid
            roots.append(dict(angle_rad=(low+high)/2,overlap_on_lower_side=left))
        boundaries.append(dict(hip=hip,joint=model.joint_order[j],physical_joint_limits_rad=controller.qp.interface.limits[j].tolist(),
            authored_intersection_boundaries=roots,coordinate_invariance_max_error_m=max(errors)))
    for f,h in hashes.items():
        if sha(f)!=h:raise ValueError('Input changed during audit '+f)
    output=dict(scope=__doc__,source_sha256=hashes,summaries=summaries,native_window=detail,angular_boundaries=boundaries,
        native_force_confirmation=False,native_execution=False,valid_expert_labels=False,
        limitations=['Authored primitive geometry may differ from cooked geometry/contact offsets.',
            'Pair-force observation is required to attribute actual native impulses.',
            'These are sampled configurations, not continuous trajectory collision guarantees.',
            'Single-joint independence checked numerically on recorded configurations; this is not a full self-collision model.',
            'Existing 7/13 acceptance omitted this collision pair and must not be interpreted as full collision feasibility.'])
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(summaries=summaries,angular_boundaries=boundaries)))


if __name__=='__main__':main()
