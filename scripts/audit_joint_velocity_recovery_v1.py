"""Offline matched joint-velocity objective experiment; frozen native files stay unchanged.

The local QP and nonlinear merit both gain the same all-joint velocity cost.
This process-local experiment does not replace the queued evaluator or alter
hardware, contact, geometry or original terminal acceptance thresholds.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import inspect
import json
import multiprocessing
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import scripts.landing_joint_sequence_proposal_v1 as proposal_module
import scripts.landing_sequential_recovery_search_v1 as search_module
from scripts.audit_geometry_recovery_task_v1 import query as original_query
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.audit_joint_sequence_proposal_v1 import plain


def generated_proposal_source(weight):
    if not np.isfinite(weight) or weight<=0:raise ValueError('Positive objective weight required')
    source=inspect.getsource(proposal_module.propose_sequence)
    anchor='            task(vm[3:6], vb[3:6], np.zeros(3), 2.*w)'
    if source.count(anchor)!=1:raise ValueError('Original objective shape changed')
    return source.replace(anchor,anchor+f'\n            task(S.T@vm, S.T@vb, np.zeros(18), {float(weight)!r}*w)')


def query(args):
    bundle,geometry,row,weight,source=args
    if source!=generated_proposal_source(weight):raise ValueError('Generated objective differs')
    namespace=dict(vars(proposal_module));exec(compile(source,'<joint-velocity-objective>','exec'),namespace)
    original_proposal=search_module.propose_sequence;original_rollout=search_module.rollout_for_search
    def weighted_rollout(controller,initial,commands,reference):
        result=original_rollout(controller,initial,commands,reference);H=len(result['profile'])
        result['joint_velocity_cost']=sum(.5*weight*(10. if i==H-1 else 1./H)*
            np.sum(n['following'].joint_velocity**2) for i,n in enumerate(result['profile']))
        result['cost']+=result['joint_velocity_cost'];return result
    search_module.propose_sequence=namespace['propose_sequence']
    search_module.rollout_for_search=weighted_rollout
    try:result=original_query((bundle,geometry,row))
    finally:
        search_module.propose_sequence=original_proposal;search_module.rollout_for_search=original_rollout
    manifest=json.loads((bundle/'manifest.json').read_text());cfg=manifest['configuration'];c=build_controller(cfg)
    c.sequence_geometry=AuthoredGroundClearance(geometry,c.qp.model,margin_m=0.)
    key=f"case_{row['case']:03d}"
    with np.load(bundle/manifest['bundle_name'],allow_pickle=False) as z:
        state=c.qp.model.state_from_public(z[key+'_query_public'],np.asarray(cfg['posture_target']))
        limits=np.asarray(cfg['interface']['limits'])
        state.previous_command_target=limits[:,0]+.5*(z[key+'_previous_action']+1.)*np.diff(limits).ravel()
        reference=limits[:,0]+.5*(z[key+'_query_student_action']+1.)*np.diff(limits).ravel()
    final=original_rollout(c,state,np.asarray(result['final_search_commands']),reference)['profile'][-1]['following']
    result.update(joint_velocity_weight=weight,terminal_joint_velocity_rad_s=final.joint_velocity.tolist(),
        maximum_terminal_leg_speed_rad_s=float(max(abs(final.joint_velocity[:12]))),
        maximum_terminal_arm_speed_rad_s=float(max(abs(final.joint_velocity[12:]))),
        terminal_root_vertical_speed_mps=float((final.root_rotation_world@final.v[:3])[2]),
        native_execution=False)
    return plain(result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--geometry',type=Path,required=True)
    p.add_argument('--start-report',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--weight',type=float,default=2.);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve experiment evidence')
    start=json.loads(a.start_report.read_text())
    for name,digest in start['source_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed start source '+name)
    source=generated_proposal_source(a.weight);a.output.parent.mkdir(parents=True,exist_ok=True)
    generated=a.output.with_suffix('.proposal.py');generated.write_text(source)
    rows=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        for row in pool.map(query,[(a.bundle,a.geometry,r,a.weight,source) for r in start['records']]):
            rows.append(row);print(json.dumps({k:row[k] for k in ['case','accepted','maximum_terminal_leg_speed_rad_s','maximum_terminal_arm_speed_rad_s','final_ground_violation_m']}),flush=True)
            a.output.with_suffix('.partial.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    files=[Path(__file__),a.start_report,a.bundle/'manifest.json',a.geometry,generated,
        ROOT/'configs/go2_piper_racket_landing_model_v1.json']
    files += [Path(m.__file__) for n,m in sys.modules.items() if n.startswith('scripts.') and getattr(m,'__file__',None)]
    result=dict(scope=__doc__,native_execution=False,joint_velocity_weight=a.weight,
        accepted_model_candidates=sum(r['accepted'] for r in rows),records=rows,
        source_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
