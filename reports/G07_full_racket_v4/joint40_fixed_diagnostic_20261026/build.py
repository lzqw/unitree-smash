"""Retain all128 paired frozen TRAIN outcomes for the new learner8 and40."""
from datetime import datetime,timezone
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
REPORTS=ROOT/'reports/G07_full_racket_v4';sys.path.insert(0,str(ROOT))
from quadsmash.evaluation_v4 import independent_episode,FAULTS


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output=HERE/'REPORT.json';assert not output.exists()
    path=REPORTS/'residual_resume24_queue01/run.py'
    spec=importlib.util.spec_from_file_location('preserved_checks',path)
    core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    runs=[REPORTS/'residual_ppo_joint_placement_actor32_n2048_probe8_fixed_train128_01',
        REPORTS/'residual_ppo_joint_placement_actor32_n2048_iter40_fixed_train128_01']
    policies=[REPORTS/'residual_ppo_joint_placement_actor32_n2048_probe8_01/policy.pt',
        REPORTS/'residual_ppo_joint_placement_actor32_n2048_resume32_01/policy.pt']
    summaries=[core.verify_evaluation(run,policy)for run,policy in zip(runs,policies)]
    protocol=read(ROOT/'configs/g07_full_racket_eval_v4_r3.json');protocol['_root']=str(ROOT)
    rows=[read(run/'episodes.json')for run in runs];pairs=[]
    for index,(before,after)in enumerate(zip(*rows)):
        assert before['source_training_index']==after['source_training_index']==index
        value=dict(TRAIN_index=index,case_id=before['case_id'])
        for iteration,row in [(8,before),(40,after)]:
            checked=independent_episode(row,protocol);v=np.asarray(row['outgoing_velocity_mps'])
            value.update({f'{name}_iter{iteration}':item for name,item in checked.items()})
            value.update({f'had_contact_iter{iteration}':row['had_contact'],
                f'frame_hit_iter{iteration}':row['frame_hit'],f'string_hit_iter{iteration}':row['string_hit'],
                f'outgoing_norm_mps_iter{iteration}':float(np.linalg.norm(v))if np.isfinite(v).all()else None,
                f'outgoing_vz_mps_iter{iteration}':float(v[2])if np.isfinite(v).all()else None,
                f'recovery_s_iter{iteration}':row['jump']['recovery_s'],
                f'faults_iter{iteration}':','.join(name for name in FAULTS if row[name])})
        pairs.append(value)
    table=HERE/'all128_paired_outcomes.csv'
    with table.open('x',newline='')as stream:
        writer=csv.DictWriter(stream,fieldnames=list(pairs[0]));writer.writeheader();writer.writerows(pairs)
    value=dict(checked_at=datetime.now(timezone.utc).isoformat(),
        scope='All128 original cases under two frozen policies of the same new learner. TRAIN diagnostics only, no A/B cases or statistical superiority claim.',
        original_frozen8=summaries[0],continued_frozen40=summaries[1],
        no_TRAIN_cases_removed=True,all128_paired_csv=str(table.relative_to(ROOT)),paired_csv_sha256=sha(table),
        lost_airborne_cases=[p['TRAIN_index']for p in pairs if p['airborne_contact_iter8']and not p['airborne_contact_iter40']],
        gained_airborne_cases=[p['TRAIN_index']for p in pairs if p['airborne_contact_iter40']and not p['airborne_contact_iter8']],
        training_success_case_retests=[pairs[65],pairs[6]],
        changing_policy_training_counts=read(REPORTS/'residual_ppo_joint_placement_actor32_n2048_resume32_01/metrics.json')['counts'],
        changing_policy_complete_count_not_frozen_performance=True,
        export_and_action_fidelity='reports/G07_full_racket_v4/joint40_export_diagnostic_20261026/REPORT.json',
        measured_full_body_drift='reports/G07_full_racket_v4/joint40_export_diagnostic_20261026/ACTUAL_MOTION_COMPARISON.json',
        decision='Reject learner40 as the next frozen candidate and do not extend its same-reward training blindly. Retain learner8 and finish both unchanged causal tangent probes; then use actual outcomes to select the next native training method. The separate v3 reward/trajectory candidate remains unexecuted.',
        original_formal_thresholds_unchanged=True,
        source_sha256={str(p.relative_to(ROOT)):sha(p)for p in [Path(__file__).resolve(),*policies,*[run/'episodes.json'for run in runs],*[run/'metrics.json'for run in runs]]},
        formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    output.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(dict(report=str(output),all128_paired=True,frozen40_airborne=0,frozen40_complete=0)),flush=True)


if __name__=='__main__':main()
