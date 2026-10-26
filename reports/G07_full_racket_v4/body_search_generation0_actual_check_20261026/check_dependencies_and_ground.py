"""Record consumed config custody and actual equal ground resets separately."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
MAIN = ROOT.parent/'unitree_smash'
HERE = Path(__file__).resolve().parent
REPORTS = MAIN/'reports/G07_full_racket_v4'


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = HERE/'DEPENDENCIES_AND_GROUND_CHECK.json';assert not out.exists()
    plan_path = REPORTS/'fixed_actor_body_cem_gen2_queue01/PLAN.json'
    plan = read(plan_path)
    for name,digest in plan['source_hashes'].items():assert sha(MAIN/name) == digest,name
    new_path = REPORTS/'fixed_actor_body_cem1280_gen2_01/initial_ground_states_000.json'
    old_path = REPORTS/'residual_ppo_centered_placement075_n2048_probe8_fixed_train128_01/initial_states.json'
    new,old = read(new_path),read(old_path)
    old_q,old_v,old_root = (np.asarray([x[key] for x in old]) for key in
        ['joint_position_rad','joint_velocity_rad_s','root_state_local'])
    q,v,root = (np.asarray(new[key]) for key in
        ['joint_position_rad','joint_velocity_rad_s','root_state_local'])
    assert q.shape == v.shape == (1280,18) and root.shape == (1280,13)
    values = [dict(candidate=i,joint_position_max_error_rad=float(abs(q[i*128:(i+1)*128]-old_q).max()),
        joint_velocity_max_error_rad_s=float(abs(v[i*128:(i+1)*128]-old_v).max()),
        root_state_max_error=float(abs(root[i*128:(i+1)*128]-old_root).max())) for i in range(2)]
    assert all(x['joint_position_max_error_rad'] == x['joint_velocity_max_error_rad_s'] == x['root_state_max_error'] == 0. for x in values)
    new_geometry = REPORTS/'fixed_actor_body_cem1280_gen2_01/geometry.json'
    old_geometry = REPORTS/'residual_ppo_centered_placement075_n2048_probe8_fixed_train128_01/geometry.json'
    assert read(new_geometry)['body_names'] == read(old_geometry)['body_names']
    configs = sorted(MAIN.glob('configs/*.json'))
    # All runtime config hashes agree with the prelaunch native custody plan,
    # which remains held by the running native parent and CPU video observer.
    fingerprints = {str(path):sha(path) for path in [Path(__file__),plan_path,new_path,old_path,
        new_geometry,old_geometry,HERE/'RESULT.json',*configs]}
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(),source_sha256=fingerprints,
        all380_native_bound_inputs_still_equal_prelaunch_plan=True,
        source_plan_sha256=sha(plan_path),actual_equal_initial_states=values,
        body_name_order_identical=True,
        source_list_scope='Supplements the analysis manifest with actual configs, both ground-state files and body ordering. No rerun or altered numerical result.',
        native_world_placement_or_numeric_cause_established=False,
        new_native_execution=False,formal_cases_read=False,formal_acceptance=False)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(actual_ground_state_differences=values,all_native_bound_inputs_unchanged=True)),flush=True)


if __name__ == '__main__':main()
