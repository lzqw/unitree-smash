"""Compact evidence snapshot; live job progress is explicitly time stamped."""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    def read(path):return json.loads((ROOT/path).read_text())
    batch=Path('reports/pure_control_release_bank_search_20261108')
    cases=[read(path.relative_to(ROOT)) for path in sorted((ROOT/batch).glob('case_*.json'))]
    failures=Counter();heads=Counter();steps=[]
    for row in cases:
        fine=row.get('fine')
        if fine is None:failures['no_fine_replay']+=1;continue
        steps.append(fine['steps']);f=fine.get('failure') or {};error=f.get('error','')
        if f.get('reason')=='physical_path_violation':
            failures['physical_path_violation']+=1;heads[(f.get('first_ground_violation') or {}).get('body','other')]+=1
        elif 'did not converge' in error:failures['contact_solver_not_converged']+=1
        elif 'authored_net_collision' in error:failures['authored_net_collision']+=1
        else:failures[f.get('reason','passed')]+=1
    impulse_path=Path('reports/pure_control_failure_chain_20261108/clean_ground_impulse_balance.json')
    model_path=batch/'result.json'
    prep=Path('reports/pure_control_contact_detail_preparation_20261108_01')
    run=Path('reports/pure_control_contact_detail_all128_20261108_01')
    replay=dict(result=read(model_path),coarse_candidate_count=sum(len(r['coarse_candidates']) for r in cases),
        fine_failure_counts=dict(failures),first_nonfoot_ground_bodies=dict(heads),
        fine_steps=steps,case_details=[dict(case=r['case'],fine_steps=r['fine']['steps'],
        fine_failure=r['fine']['failure']) for r in cases if r.get('fine')])
    evidence=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Pure-control landing; no new RL/BC/distillation',
        release_bank_search=replay,impulse_balance=read(impulse_path),
        native_diagnostic_status=read(prep/'status.json'),native_progress_snapshot=read(run/'pure_control_progress.json'),
        followup_status=read(Path('reports/pure_control_contact_detail_audit_20261108_01/status.json')),
        new_native_result_available=(ROOT/run/'job.json').exists() and read(run/'job.json')['status']=='SUCCEEDED',
        unrestricted_interface=dict(implemented=True,native_executed=False,physical_limits_unchanged=True,
            replaces='Previous-command arm software slew only in a future separately versioned trial'),
        source_sha256={str(path):hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in
            [impulse_path,model_path,prep/'plan.json',Path(__file__).relative_to(ROOT)]},
        current_running_run_uses_original_controller_and_arm_slew=True,
        results_are_not_expert_labels=True)
    output=ROOT/'docs/pure_control_contact_impulse_and_search_20261108.json'
    output.write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output),cases=len(cases),failures=failures,
        native_progress=evidence['native_progress_snapshot'])))


if __name__=='__main__':main()
