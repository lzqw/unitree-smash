"""Compare the same original128 TRAIN cases after actual fixed leg06 evaluation."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REPORTS = ROOT/'reports/G07_full_racket_v4'
sys.path.insert(0,str(ROOT))
from quadsmash.evaluation_v4 import independent_episode
spec = importlib.util.spec_from_file_location('frozen_leg06_bottleneck_reference',
    REPORTS/'centered_leg06_after075_intermediates_pilot8_queue01/run.py')
reference = importlib.util.module_from_spec(spec); spec.loader.exec_module(reference)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze():
    output = HERE/'ACTUAL_FROZEN128_SOURCE_COMPARISON.json'; assert not output.exists()
    protocol = reference.core.read(ROOT/'configs/g07_full_racket_eval_v4_r3.json'); protocol['_root'] = str(ROOT)
    court = reference.core.read(ROOT/'configs/court_net_v4.json')
    pairs = [('source0758',REPORTS/'residual_ppo_centered_placement075_n2048_probe8_fixed_train128_01',
        REPORTS/'residual_ppo_centered_placement075_n2048_probe8_01/policy.pt'),
        ('new_leg06_8',reference.EVALUATION,reference.TRAIN/'policy.pt')]
    results, flags, sources = {}, {}, {}
    for name,folder,policy in pairs:
        summary = reference.core.verify_evaluation(folder,policy)
        assert reference.core.read(folder/'INPUTS.json')['reset_noise_seed'] == 724201
        rows = reference.core.read(folder/'episodes.json'); gate_rows, velocities = [], []
        for index,row in enumerate(rows):
            assert row['source_training_index'] == row['env_id'] == index and row['episode_index'] == 0
            x = independent_episode(row,protocol)
            v = np.asarray(row['outgoing_velocity_mps']); landing = np.asarray(row['first_landing_m'])
            attack = bool(x['airborne_contact'] and row['released'] and np.isfinite(v).all()
                and np.linalg.norm(v) >= 8. and v[0] > 0. and v[2] <= -1.)
            target = bool(np.isfinite(landing).all() and court['target_x_limits_m'][0] <= landing[0] <= court['target_x_limits_m'][1]
                and abs(landing[1]) <= court['target_y_half_width_m'])
            chain = bool(attack and row['crossed_net'] and row['landed_target'] and target and not x['fault'])
            gate_rows.append(dict(had_contact=bool(row['had_contact']),air=x['airborne_contact'],
                air_and_attack=attack,air_attack_crossed_target_without_fault=chain,
                air_and_qualified_recovery=x['conditional_recovery'],complete=x['complete'],
                any_fault=x['fault'],terminated_with_body_landing=bool(row['jump']['terminated'] and row['jump']['body_landing'])))
            if x['airborne_contact'] and np.isfinite(v).all(): velocities.append([float(np.linalg.norm(v)),float(v[2])])
        keys = list(gate_rows[0]); flags[name] = gate_rows
        results[name] = dict(original_TRAIN128_summary=summary,
            sequential_and_separate_gate_counts={key:sum(row[key] for row in gate_rows) for key in keys},
            actual_air_outgoing_speed_and_vz=velocities)
        for path in [policy,folder/'job.json',folder/'completed.json',folder/'metrics.json',folder/'INPUTS.json',folder/'episodes.json']:
            sources[str(path.relative_to(ROOT))] = sha(path)
    transitions = {}
    for key in flags['source0758'][0]:
        old,new = flags['source0758'],flags['new_leg06_8']
        transitions[key] = dict(retained=[i for i in range(128) if old[i][key] and new[i][key]],
            gained=[i for i in range(128) if not old[i][key] and new[i][key]],
            lost=[i for i in range(128) if old[i][key] and not new[i][key]])
    sources[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(),results=results,
        same128_case_transitions=transitions,source_sha256=sources,
        scope='Two actual original128 TRAIN fixed policies, same seed/no noise. Diagnostic gates do not replace independent complete predicate. One development comparison, no causal isolation or formal A/B score.',
        new_native_execution=False,formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({name:item['sequential_and_separate_gate_counts'] for name,item in results.items()}),flush=True)


if __name__ == '__main__': analyze()
