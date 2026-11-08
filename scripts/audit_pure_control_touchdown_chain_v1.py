"""Retrospective physical landing chronology, without altering acceptance."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
from scripts.hip_mount_planning_constraints_v1 import HipMountPlanningConstraintsV1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['run','audit','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();read=lambda f:json.loads(Path(f).read_text())
    if a.output.exists():raise ValueError('Fresh evidence output required')
    job=read(a.run/'job.json')
    if job['status']!='SUCCEEDED':raise ValueError('Completed native execution required')
    def arrays(name):
        with np.load(a.run/name) as z:return {k:z[k].copy() for k in z.files}
    d=arrays('pure_control_commands.npz');f=arrays('pure_control_ground_forces.npz')
    m=read(a.run/'pure_control_ground_forces.json');audit=read(a.audit);metrics=read(a.run/'metrics.json')
    rows=[json.loads(x) for x in (a.run/'pure_control_diagnostics.jsonl').read_text().splitlines()]
    model=LoadedPinocchioModelV1();hip=HipMountPlanningConstraintsV1(model)
    names=m['body_names'];feet=[names.index(n) for n in model.feet];heads=[names.index(n) for n in ['Head_lower','Head_upper']]
    others=[i for i in range(len(names)) if i not in feet+heads];cases=[];dt=m['physics_rate_aggregation_s']
    for case in audit['cases']:
        lane=case['case'];ids=np.flatnonzero(d['lane']==lane)
        if not len(ids):continue
        begin,end=int(d['tick'][ids[0]]),int(d['tick'][ids[-1]])+1
        def first(indices):
            frames=f['interval_first_frame_over1n'][begin:end,lane][:,indices]
            entries=np.argwhere(frames>=0)
            if not len(entries):return None
            ij=min(entries,key=lambda ij:frames[tuple(ij)])
            frame=int(frames[tuple(ij)])
            return dict(physics_frame=frame,time_s=frame*dt,body=names[indices[int(ij[1])]])
        touchdown=first(feet);head=first(heads);other=first(others)
        positions=d['q'][ids][:,model.q_indices];gaps=np.asarray([hip.gaps(q) for q in positions])
        near=np.flatnonzero(gaps.min(1)<.001)
        at_touchdown=None
        if touchdown:
            tick=int(np.floor(touchdown['time_s']/.01))
            j=int(np.argmin(abs(d['tick'][ids]-tick)))
            at_touchdown=dict(sample_time_s=float(d['time_s'][ids[j]]),front_hip_rad=positions[j,:2].tolist(),
                calf_rad=positions[j,8:12].tolist(),joint_velocity=d['v'][ids[j],model.v_indices].tolist(),
                hip_mount_minimum_gap_rad=float(gaps[j].min()))
        pre=[int(i) for i in ids if not touchdown or float(d['time_s'][i])<touchdown['time_s']]
        cases.append(dict(case=lane,takeover_time_s=float(d['time_s'][ids[0]]),first_foot_ground=touchdown,
            first_head_ground=head,first_other_ground=other,
            head_after_first_foot_ms=(head['time_s']-touchdown['time_s'])*1000 if head and touchdown else None,
            first_near_hip_mount_s=float(d['time_s'][ids[near[0]]]) if len(near) else None,
            at_touchdown=at_touchdown,pre_touchdown_commands=len(pre),
            pre_touchdown_rejected_qp=sum(not rows[i]['diagnostic']['solver_accepted'] for i in pre),
            no_recorded_nonfoot_ground_force=case['no_recorded_nonfoot_ground_force'],
            longest_sampled_stable_s=case['longest_control_sample_stable_s'],
            final_sampled_stable_s=case['final_control_sample_stable_s']))
    delays=[r['head_after_first_foot_ms'] for r in cases if r['head_after_first_foot_ms'] is not None]
    summary=dict(owned=len(cases),head_over1n=sum(r['first_head_ground'] is not None for r in cases),
        other_nonfoot_over1n=sum(r['first_other_ground'] is not None for r in cases),
        any_near_hip_mount=sum(r['first_near_hip_mount_s'] is not None for r in cases),
        near_hip_before_touchdown=sum(r['first_near_hip_mount_s'] is not None and r['first_foot_ground'] is not None and
            r['first_near_hip_mount_s']<=r['first_foot_ground']['time_s'] for r in cases),
        no_recorded_nonfoot=sum(r['no_recorded_nonfoot_ground_force'] for r in cases),
        sampled_stable_08s=[r['case'] for r in cases if r['longest_sampled_stable_s']>=.8],
        clean_and_sampled_stable_08s=[r['case'] for r in cases if r['no_recorded_nonfoot_ground_force'] and r['longest_sampled_stable_s']>=.8],
        head_vs_foot_delay_ms=dict(minimum=min(delays),median=float(np.median(delays)),maximum=max(delays)) if delays else None)
    fields=['complete5','airborne_contacts','conditional_recovery_count','actual_racket_contacts','crossed_net','landed_target','numerical_failures','fault_counts']
    output=dict(summary=summary,cases=cases,metrics={k:metrics[k] for k in fields},
        ball5_except_recovery=sum(r['ball5_except_recovery'] for r in metrics['complete5_case_outcomes']),
        execution=audit['execution'],gpu_hours=job['charged_gpu_hours'],control_env_steps=job['charged_env_steps'],
        diagnostic_force_threshold_n=1.,hip_near_boundary_diagnostic_rad=.001,formal_acceptance_unchanged=True,
        sampled_stability_not_strict_certification=True,
        source_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),a.audit,a.run/'pure_control_commands.npz',a.run/'pure_control_ground_forces.npz',a.run/'job.json']})
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(summary=summary,metrics=output['metrics'],ball5_except_recovery=output['ball5_except_recovery'])))


if __name__=='__main__':main()
