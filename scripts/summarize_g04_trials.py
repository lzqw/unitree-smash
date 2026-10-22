#!/usr/bin/env python3
"""Summarize actual independent evaluations without selecting only good runs."""
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TRIALS=[('corrected_v2_smoke','corrected_smoke_eval_v2'),
    ('corrected_v2b_main01','corrected_v2b_main01_eval'),
    ('corrected_v2c_main01','corrected_v2c_main01_eval'),
    *[(f'corrected_v2c_continue_{i:02d}',f'corrected_v2c_continue_{i:02d}_eval') for i in (1,2,3)]]
rows=[]
for train,ev in TRIALS:
    path=ROOT/f'reports/G04/{ev}/metrics.json'
    if not path.is_file():continue
    metric=json.loads(path.read_text())
    progress=ROOT/f'runs/G04/{train}/native/training_progress.jsonl'
    last=json.loads(progress.read_text().splitlines()[-1])
    rows.append(dict(training_run=train,evaluation=ev,global_iteration=last['iteration'],
        actual_local_training_interactions=last['train_env_steps_this_run'],episodes=metric['sample_count'],
        panel_a_success=metric['panel_success_rates']['dev_jump_v2_a'],
        panel_b_success=metric['panel_success_rates']['dev_jump_v2_b'],
        minimum_success=metric['jump_recovery_success_rate'],body_contact=metric['body_landing_rate'],
        qualified_flight_seen=metric.get('qualified_flight_seen_rate'),
        qualified_landing=metric['qualified_jump_landed_rate'],maximum_com_rise=metric['maximum_com_rise_m'],
        numerical_failures=metric['numerical_failures'],noise_std_mean=last['losses'].get('action_std_mean'),
        checkpoint_sha256=metric['checkpoint_sha256']))
out=ROOT/'reports/G04'
with (out/'trial_comparison.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(out/'trial_comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
