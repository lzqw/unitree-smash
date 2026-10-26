"""Validate completed native48 training and portable weights from real bytes.

This checks changing-policy training evidence only. It never opens formal
case files or converts training counts into frozen-policy performance.
"""
import argparse
import contextlib
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

TRAIN = 'reports/G07_full_racket_v4/body_residual_leg15_fresh_n2048_updates48_01'
HERE_REL = 'reports/G07_full_racket_v4/body48_completed_training_delivery_20261026'
STEPS = 48*2048*128


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate(root, require_bundled=False):
    root = root.resolve()
    sys.path[:0] = [str(root/'vendor/rsl_rl'), str(root)]
    import numpy as np
    import torch
    import rsl_rl
    from rsl_rl.modules import ActorCritic
    from tensordict import TensorDict
    from quadsmash.evaluation_v4 import independent_episode
    from quadsmash.control.full_racket_train_evaluation_v4 import training_cases, summarize_training
    from quadsmash.rl.episode_trace_v4 import unpack_public
    from scripts.initialize_centered_policy_v4 import same_tree

    bundled = Path(rsl_rl.__file__).resolve().is_relative_to(root/'vendor/rsl_rl')
    assert bundled or not require_bundled
    run, here = root/TRAIN, root/HERE_REL
    inputs, metrics, job = [read(run/name) for name in ['INPUTS.json','metrics.json','job.json']]
    assert job['status'] == 'SUCCEEDED' and job['exit_code'] == 0 and job['receipt_present']
    assert job['charged_env_steps'] == read(run/'completed.json')['env_steps'] == STEPS
    assert inputs['num_envs'] == 2048 and inputs['requested_updates'] == 48
    assert inputs['training_indices'] == list(range(128))
    protocol_path = root/'configs/g07_full_racket_eval_v4_r3.json'
    protocol = read(protocol_path)
    protocol['_root'] = str(root)
    assert inputs['contract']['protocol_sha256'] == sha(protocol_path)
    for name, digest in protocol['physical_configs'].items():
        assert sha(root/name) == digest, name
    assert sha(root/protocol['asset']['path']) == protocol['asset']['sha256']
    for name, digest in read(run/'source_hashes.json').items():
        assert sha(root/name) == digest, name

    native = torch.load(run/'checkpoint.pt', map_location='cpu', weights_only=False)
    public = torch.load(run/'policy.pt', map_location='cpu', weights_only=False)
    assert native['iter'] == 48 and native['native']['canonical_restart']
    assert native['native']['total_timesteps'] == STEPS
    assert not native['native']['environment']['tensors']['episode_length_buf'].any()
    state = native['model_state_dict']
    assert int(state['actor_obs_normalizer.count']) == int(state['critic_obs_normalizer.count']) == STEPS
    assert all(torch.isfinite(value).all() for value in state.values() if torch.is_tensor(value))
    configuration = read(root/'configs/g07_full_racket_body_residual_leg15_v4.json')
    assert sha(root/'configs/g07_full_racket_body_residual_leg15_v4.json') == inputs['configuration_sha256']
    options = dict(configuration['runner']['policy'])
    options.pop('class_name')
    with contextlib.redirect_stdout(io.StringIO()):
        actor = ActorCritic(TensorDict({'policy':torch.zeros(2,794)}, batch_size=[2]),
            configuration['runner']['obs_groups'], 18, **options)
    actor.load_state_dict(state, strict=True)
    actor.eval()
    names = [name for name,_ in actor.named_parameters()]
    initial = torch.load(run/'fresh_initial_model_state.pt', map_location='cpu', weights_only=False)
    start = initial['model_state_dict']
    assert initial['actual_control_env_steps'] == initial['optimizer_entries'] == 0
    assert initial['base_policy_sha256'] == inputs['contract']['base_policy_sha256']
    assert initial['fixed_source_residual_actor_count'] == 2
    assert int(start['actor_obs_normalizer.count']) == int(start['critic_obs_normalizer.count']) == 0
    assert torch.allclose(start['log_std'].exp(), torch.tensor([.2]*12+[.1]*6), atol=1e-7, rtol=0)
    actor_keys = [name for name in start if name.startswith('actor.')]
    assert not start[actor_keys[-1]].any() and not start[actor_keys[-2]].any()
    optimizer = native['optimizer_state_dict']
    ids = [value for group in optimizer['param_groups'] for value in group['params']]
    assert len(ids) == len(names) == len(optimizer['state']) == 17
    adam = {name:int(optimizer['state'][index]['step']) for name,index in zip(names,ids)}
    assert all(value == (960 if name.startswith('critic.') else 820) for name,value in adam.items())
    assert all(torch.isfinite(v).all() for s in optimizer['state'].values() for v in s.values() if torch.is_tensor(v))

    diagnostics = []
    for iteration in range(1,49):
        path = run/f'ppo/model_{iteration-1}.pt'
        diagnostic = torch.load(path, map_location='cpu', weights_only=False)
        model = diagnostic['model_state_dict']
        assert diagnostic['iter'] == iteration and not diagnostic['native']['canonical_restart']
        assert diagnostic['native']['total_timesteps'] == iteration*262144
        assert int(model['actor_obs_normalizer.count']) == int(model['critic_obs_normalizer.count']) == iteration*262144
        assert len(diagnostic['optimizer_state_dict']['state']) == (8 if iteration <= 7 else 17)
        assert all(torch.isfinite(v).all() for v in model.values() if torch.is_tensor(v))
        opt = diagnostic['optimizer_state_dict']
        ordered = [index for group in opt['param_groups'] for index in group['params']]
        for name,index in zip(names,ordered):
            if name.startswith('critic.') or iteration > 7:
                assert int(opt['state'][index]['step']) == (iteration*20 if name.startswith('critic.') else (iteration-7)*20)
        unchanged = all(torch.equal(model[name],start[name]) for name in [*actor_keys,'log_std'])
        assert unchanged == (iteration <= 7)
        diagnostics.append(dict(iteration=iteration,sha256=sha(path)))
    assert metrics['completed_updates'] == 48 and len(metrics['history']) == 48
    assert [item['actor_warmup'] for item in metrics['history']] == [True]*7+[False]*41
    assert metrics['history'][-1]['actual_control_env_steps'] == STEPS

    ground = read(run/'initial_ground_states.json')
    roots = np.asarray(ground['root_state_local'])
    assert roots.shape == (2048,13) and np.isfinite(roots).all()
    assert np.max(abs(roots[:,:3]-protocol['initialization']['root_position_m'])) < 1e-5
    assert np.max(abs(roots[:,3:7]-[1,0,0,0])) < 1e-5 and np.max(abs(roots[:,7:])) < 1e-5
    assert np.max(abs(np.asarray(ground['joint_velocity_rad_s']))) < 1e-5
    assert ground['all_episode_clocks_zero'] and ground['source_training_indices'] == list(range(128))*16
    rows = read(run/'episodes.json')
    seen, outcomes = [0]*2048, []
    for row in rows:
        lane = row['env_id']
        assert row['episode_index'] == seen[lane] and row['source_training_index'] == lane%128 and row['ground_init']
        seen[lane] += 1
        outcomes.append(independent_episode(row,protocol))
    assert seen == native['native']['environment']['tensors']['episode_index'].tolist()
    counts = dict(completed_episodes=len(rows),air=sum(x['airborne_contact'] for x in outcomes),
        conditional_recovery_count=sum(x['conditional_recovery'] for x in outcomes),
        complete=sum(x['complete'] for x in outcomes),numerical_failures=sum(x['numerical_failures'] for x in rows))
    assert counts == metrics['counts'] and counts['numerical_failures'] == 0

    assert public['schema'] == 'full-racket-body-residual-ppo-v4' and public['native_execution']
    assert not public['performance_inherited'] and sha(run/'policy.pt') == metrics['policy_sha256']
    check_inputs = read(here/'CHECK_INPUTS.json')
    source_rel = check_inputs['source_policy_relative_path']
    source = torch.load(root/source_rel, map_location='cpu', weights_only=False)
    assert sha(root/source_rel) == inputs['contract']['base_policy_sha256']
    assert same_tree(public['base_controller'],source)
    source_run = root/check_inputs['source_evaluation_relative_path']
    source_job = read(source_run/'job.json')
    assert source_job['status'] == 'SUCCEEDED' and source_job['exit_code'] == 0
    assert source_job['charged_env_steps'] == read(source_run/'completed.json')['env_steps'] == 102400
    cases,indices = training_cases(root,protocol)
    source_summary = summarize_training(read(source_run/'episodes.json'),cases,indices,protocol)
    assert all(read(source_run/'metrics.json')[key] == value for key,value in source_summary.items())
    assert read(source_run/'INPUTS.json')['policy_sha256'] == sha(root/source_rel)
    manifest = read(run/'trajectory_traces/manifest.json')
    assert manifest['disabled_error'] is None and manifest['closed_episodes_observed'] == len(rows)
    samples = []
    for entry in manifest['saved_episodes']:
        path = run/'trajectory_traces'/entry['data_file']
        meta = read(path.with_suffix('.json'))
        assert sha(path) == entry['data_sha256'] == meta['data_sha256']
        with np.load(path) as data:
            actual = unpack_public(data['packed_public'])
        assert hashlib.sha256(actual.tobytes()).hexdigest() == meta['full_public794_sha256']
        samples.append(actual)
    witness = torch.from_numpy(np.concatenate(samples))
    portable = torch.jit.load(io.BytesIO(public['residual_actor_jit']),map_location='cpu').eval()
    with torch.inference_mode():
        expected = actor.act_inference(TensorDict({'policy':witness},batch_size=[len(witness)]))
        error = float((portable(witness)-expected).abs().max())
    assert error <= 2e-5
    with (here/'NATIVE48_EPISODES.csv').open(newline='') as stream:
        table = list(csv.DictReader(stream))
    assert len(table) == len(rows)
    for key, value in [('canonical_complete',counts['complete']),('canonical_airborne_contact',counts['air']),
                        ('canonical_conditional_recovery',counts['conditional_recovery_count'])]:
        assert sum(int(row[key]) for row in table) == value
    assert all((int(item['env_id']),int(item['episode_index'])) == (row['env_id'],row['episode_index'])
               for item,row in zip(table,rows))
    return dict(checked_at=datetime.now(timezone.utc).isoformat(),root=str(root),
        scope='Completed changing-policy native48 course only; no frozen performance or formal acceptance.',
        independently_recomputed_training_counts=counts,all2048_episode_chains_exact=True,
        canonical_checkpoint_sha256=sha(run/'checkpoint.pt'),public_policy_sha256=sha(run/'policy.pt'),
        actual_public_witness_rows=len(witness),canonical_vs_public_actor_max_error=error,
        diagnostics=diagnostics,actual_Adam_steps=adam,bundled_RSL_used=bundled,
        source_original_TRAIN128_recomputed=source_summary,
        original_source_teacher_exact=True,new_native_execution=False,formal_cases_read=False,
        formal_acceptance=False,goal_complete=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--require-bundled',action='store_true')
    args = parser.parse_args()
    assert not args.output.exists()
    result = validate(args.root,args.require_bundled)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['canonical_vs_public_actor_max_error','actual_public_witness_rows','bundled_RSL_used']}))
