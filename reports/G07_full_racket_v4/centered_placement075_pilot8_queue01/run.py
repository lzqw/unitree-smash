"""Conditional measured-actor075 pilot through the original main GPU0 ledger.

Inspection is read-only. Preparation/native execution is main-worktree only,
after the original intermediate selector has actually exited.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
MAIN = ROOT.parent/'unitree_smash'
CANDIDATE = ROOT.parent/'unitree_smash_centered_transfer'
REPORTS = ROOT/'reports/G07_full_racket_v4'
SELECTOR = MAIN/'reports/G07_full_racket_v4/centered_intermediate_selection_queue01'
PYTHON = '/home/autodrive/Isaac/env_isaaclab/bin/python'
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('preserved_centered_pilot_template', MAIN/'reports/G07_full_racket_v4/centered_residual_pilot8_queue02/run.py')
template = importlib.util.module_from_spec(spec)
spec.loader.exec_module(template)
core = template.core
TRAIN = REPORTS/'residual_ppo_centered_placement075_n2048_probe8_01'
EVALUATION = REPORTS/'residual_ppo_centered_placement075_n2048_probe8_fixed_train128_01'
BASE = REPORTS/'residual_ppo_joint_placement_actor32_n2048_probe8_01/policy.pt'
CONFIG = ROOT/'configs/g07_full_racket_residual_ppo_centered_placement075_v4.json'
PLAN = HERE/'PREDECLARED_PLAN.json'
STATE = HERE/'state.json'
VIDEO = REPORTS/'video/Current_effect_centered_placement075_pilot8_20261026.mp4'
SNAPSHOT = REPORTS/'video_snapshot_centered_placement075_pilot8_20261026'
VIDEO_SPEC = REPORTS/'video_spec_centered_placement075_pilot8_20261026.json'
STEPS = 8*2048*128
for name in ['ROOT','HERE','REPORTS','TRAIN','EVALUATION','BASE','CONFIG','PLAN','STATE','VIDEO','SNAPSHOT','VIDEO_SPEC']:
    setattr(template, name, globals()[name])


def inspect_selection():
    state = core.read(SELECTOR/'state.json')
    active = [row for row in core.read(MAIN/'status/GPU_LEDGER.json')['runs'] if row['status'] == 'RUNNING']
    owner_live = core.live(1697096, 'centered_intermediate_selection_queue01/run.py')
    return dict(observed_at=datetime.now(timezone.utc).isoformat(), actual_original_owner_live=owner_live,
                original_selector_state=state['state'], original_selector_iteration=state.get('iteration'),
                original_result_exists=(SELECTOR/'RESULT.json').exists(),
                active_project_jobs=[{key:row.get(key) for key in ['pid','child_pid','output','status']} for row in active],
                main_worktree=ROOT == MAIN,
                source_owner_and_ledger_clear=(not owner_live and not active), new_native_execution=False)


def require_completed_selection():
    status = inspect_selection()
    assert status['source_owner_and_ledger_clear'], 'Original owner or main ledger job is still live'
    assert status['original_selector_state'] == 'COMPLETED_INTERMEDIATE_SELECTION'
    selected = core.read(SELECTOR/'RESULT.json')
    assert {entry['iteration'] for entry in selected['all_candidates']} == {8,16,24,32,40}
    assert len(selected['all_candidates']) == 5
    ranked = max(selected['all_candidates'], key=lambda e:(e['summary']['complete'], e['summary']['attack_outgoing'],
                 e['summary']['conditional_recovery_count'], e['summary']['airborne_contacts'], -e['iteration']))
    assert ranked == selected['selected']
    assert core.sha(Path(ranked['policy'])) == selected['selected_policy_sha256']
    # All five complete assessments must still independently match the declared ranking.
    for entry in selected['all_candidates']:
        evaluation = source_evaluation(entry['iteration'])
        summary = core.verify_evaluation(evaluation, Path(entry['policy']))
        assert summary == entry['summary'] and summary['numerical_failures'] == 0
        assert core.read(evaluation/'INPUTS.json')['reset_noise_seed'] == 724201
        assert core.read(evaluation/'metrics.json')['noise_seed'] == 724201
    return selected


def source_evaluation(iteration):
    reports = MAIN/'reports/G07_full_racket_v4'
    if iteration == 8:
        return reports/'residual_ppo_centered_base8_n2048_probe8_fixed_train128_02'
    if iteration == 40:
        return reports/'residual_ppo_centered_base8_n2048_iter40_fixed_train128_01'
    assert iteration in [16,24,32]
    return reports/f'residual_ppo_centered_base8_diag_iter{iteration}_fixed_train128_01'


def ready(summary):
    return (summary['complete_success_rate'] >= .50 and summary['airborne_contact_rate'] >= .70
            and summary['conditional_recovery_rate'] is not None
            and summary['conditional_recovery_rate'] >= .85 and summary['numerical_failures'] == 0)


def commands(selected):
    train = template.command(TRAIN,7200,STEPS,'scripts/train_full_racket_residual_ppo_v4.py',
        ['--base-checkpoint',str(BASE),'--config',str(CONFIG),'--num-envs','2048','--updates','8',
         '--initialize-centered-policy',selected['policy'],'--centered-source-evaluation',
         str(source_evaluation(selected['iteration'])),'--seed','748901'])
    evaluate = template.command(EVALUATION,2700,102400,'scripts/evaluate_full_racket_train_v4.py',
        ['--checkpoint',str(TRAIN/'policy.pt'),'--noise','none','--seed','724201'])
    return train,evaluate


def sources(selected):
    paths = list(ROOT.glob('quadsmash/**/*.py'))+list(ROOT.glob('configs/*.json'))+list(ROOT.glob('assets/*.usd'))
    paths += [ROOT/'scripts'/name for name in ['run_gpu_job.py','train_full_racket_residual_ppo_v4.py',
        'initialize_centered_policy_v4.py','evaluate_full_racket_train_v4.py','export_full_racket_train_csv_v4.py',
        'check_full_racket_training_traces_v4.py','prepare_full_racket_latest_video_v4.py',
        'render_full_racket_latest_effect_v4.py','render_full_racket_r3_effect.py','render_current_v4_progress.py']]
    paths += [Path(__file__), BASE, Path(selected['policy']), SELECTOR/'RESULT.json', SELECTOR/'PREDECLARED_PLAN.json',
              ROOT/'reports/G07_full_racket_v4/residual_resume24_queue01/run.py',
              ROOT/'reports/G07_full_racket_v4/centered_residual_pilot8_queue02/run.py',
              ROOT/'reports/G07_full_racket_v4/joint_placement_iter40_video_queue01/run.py',
              ROOT/'assets/vendor/go2_flat_baryn20251217/policy.pt']
    urdf = ROOT/'assets/vendor/piper_isaac_sim/piper_description/urdf/piper_no_gripper_description.urdf'
    paths.append(urdf)
    paths += [p for p in (urdf.parent.parent/'meshes').rglob('*') if p.is_file() and p.suffix.lower() in ['.stl','.dae','.obj']]
    protocol = core.read(ROOT/'configs/g07_full_racket_eval_v4_r3.json')
    paths.append(ROOT/protocol['training_suite']['path'])
    for folder in [BASE.parent, source_evaluation(selected['iteration'])]:
        paths += [folder/name for name in ['job.json','completed.json','INPUTS.json','metrics.json','source_hashes.json'] if (folder/name).exists()]
    paths += [source_evaluation(selected['iteration'])/'episodes.json']
    paths += [Path(name) for name in core.read(BASE.parent/'asset_dependencies.json')['resolved_layers']]
    return {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):core.sha(p) for p in sorted(set(paths))}


def prepare():
    assert ROOT == MAIN, 'Native preparation/execution belongs to the original main worktree'
    selection = require_completed_selection()
    assert not any(p.exists() for p in [PLAN,STATE,TRAIN,EVALUATION,VIDEO,SNAPSHOT,VIDEO_SPEC])
    declaration_path = CANDIDATE/'reports/G07_full_racket_v4/centered_placement075_candidate_20261026/PREDECLARED_PLAN.json'
    declaration = core.read(declaration_path)
    assert core.sha(CONFIG) == declaration['configuration_sha256']
    expected_runtime = {}
    for name in ['scripts/initialize_centered_policy_v4.py','scripts/train_full_racket_residual_ppo_v4.py']:
        digest = declaration['source_sha256'][str(CANDIDATE/name)]
        assert core.sha(ROOT/name) == digest, 'Prepared transfer runtime has not been integrated: '+name
        expected_runtime[name] = digest
    # The original selector checked its complete source map before terminal exit.
    # Only the explicitly integrated trainer is allowed to differ afterwards.
    for name,digest in core.read(SELECTOR/'PREDECLARED_PLAN.json')['source_hashes'].items():
        target = Path(name) if Path(name).is_absolute() else ROOT/name
        assert core.sha(target) == expected_runtime.get(name,digest), 'Unexpected integration change: '+name
    selected = selection['selected']
    train,evaluate = commands(selected)
    expected = sources(selected)
    template.check_sources(expected)
    template.save_once(PLAN,dict(created_at=datetime.now(timezone.utc).isoformat(), source_hashes=expected,
        declaration_sha256=core.sha(declaration_path), source_selection_sha256=core.sha(SELECTOR/'RESULT.json'),
        selected=selected, selected_policy_sha256=selection['selected_policy_sha256'],
        source_evaluation=str(source_evaluation(selected['iteration'])),
        source_ready_for_formal_review=ready(selected['summary']),
        training_command=train,evaluation_command=evaluate,config_sha256=core.sha(CONFIG),
        initialization='Fresh learner with measured outer actor and its normalizer; exact same fixed old8 base.',
        fresh_critic_exploration_Adam_RNG_counters_and_actual_ground=True,
        actor_normalizer_accounting='Measured sourceM plus new steps; critic normalization starts0.',
        source_custody='Hold all current runtime/config/assets/selected-source inputs until actual training, fixed evaluation and CPU video exit.',
        video_preselection_before_new_training=[19,23,107,0,8,58],original_formal_thresholds_unchanged=True,
        automatic_longer_training=False,formal_cases_read=False,formal_acceptance=False,goal_complete=False))
    print(json.dumps(dict(prepared=str(PLAN),sources=len(expected),source_iteration=selected['iteration'],new_native_execution=False)))


def validate_training_inputs(inputs):
    assert inputs['initialization']['mode'] == 'fresh_learner_with_measured_centered_public_actor', 'Require the new measured-actor stage, never relabel an old strict-resume job'
    assert inputs['resume_checkpoint_sha256'] is None and inputs['initial_actor_checkpoint_sha256'] is None
    assert not inputs['resize_ground_requested']
    assert inputs['configuration_sha256'] == core.sha(CONFIG)
    plan = core.read(PLAN)
    assert inputs['initial_centered_policy_sha256'] == plan['selected_policy_sha256']
    assert inputs['centered_source_evaluation'] == plan['source_evaluation']
    init = inputs['initialization']
    assert init['source_policy_sha256'] == plan['selected_policy_sha256'] and init['initial_iteration'] == 0
    assert init['num_envs'] == 2048 and init['source_public_policy_is_not_a_restart_checkpoint']
    assert init['source_performance_is_not_inherited'] and init['performance_inherited'] is False
    for key in ['critic_transferred','critic_normalizer_transferred','optimizer_transferred','source_rng_restored',
                'old_physical_state_restored','previous_learning_counters_restored','exploration_log_std_transferred']:
        assert init[key] is False, key
    contract = inputs['contract']
    assert contract['base_controller_mode'] == 'frozen_public_policy' and contract['base_policy_sha256'] == core.sha(BASE)
    assert contract['joint_placement_reward'] == {'weight':20.,'distance_scale_m':.75}
    assert contract['seed'] == 748901 and inputs['training_indices'] == list(range(128))
    assert inputs['num_envs'] == 2048 and inputs['requested_updates'] == 8
    return init


def optimizer_steps(native):
    model,optimizer = native['model_state_dict'],native['optimizer_state_dict']
    ids = [i for group in optimizer['param_groups'] for i in group['params']]
    names = [name for name in model if name == 'log_std' or name.startswith(('actor.','critic.'))]
    assert len(ids) == len(names) == 17
    return {name:int(optimizer['state'][key]['step']) for key,name in zip(ids,names) if key in optimizer['state']}


def verify_first_update():
    import numpy as np
    import torch
    inputs = core.read(TRAIN/'INPUTS.json');init = validate_training_inputs(inputs)
    native = torch.load(TRAIN/'ppo/model_0.pt',map_location='cpu',weights_only=False)
    assert native['iter'] == 1 and not native['native']['canonical_restart']
    assert native['native']['total_timesteps'] == 262144 and native['native']['environment']['num_envs'] == 2048
    model = native['model_state_dict']
    assert all(torch.isfinite(value).all() for value in model.values() if torch.is_tensor(value))
    assert int(model['actor_obs_normalizer.count']) == init['source_actor_normalizer_count']+262144
    assert int(model['critic_obs_normalizer.count']) == 262144
    steps = optimizer_steps(native)
    assert len(steps) == 8 and all(name.startswith('critic.') and value == 20 for name,value in steps.items())
    assert all(torch.isfinite(value).all() for state in native['optimizer_state_dict']['state'].values() for value in state.values() if torch.is_tensor(value))
    source = torch.load(Path(core.read(PLAN)['selected']['policy']),map_location='cpu',weights_only=False)
    actor = torch.jit.load(io.BytesIO(source['residual_actor_jit']),map_location='cpu').eval().state_dict()
    assert all(torch.equal(model['actor.'+name[2:]],value) for name,value in actor.items() if name.startswith('1.'))
    assert torch.allclose(model['log_std'].exp(),torch.full_like(model['log_std'],.1),atol=1e-7,rtol=0)
    ground = core.read(TRAIN/'initial_ground_states.json')
    assert ground['all_episode_clocks_zero'] and ground['source_training_indices'] == list(range(128))*16
    roots = np.asarray(ground['root_state_local'])
    protocol = core.read(ROOT/'configs/g07_full_racket_eval_v4_r3.json')
    assert roots.shape == (2048,13) and np.isfinite(roots).all()
    assert np.max(abs(roots[:,:3]-protocol['initialization']['root_position_m'])) <= 1e-5
    assert np.max(abs(roots[:,3:7]-[1.,0.,0.,0.])) <= 1e-5 and np.max(abs(roots[:,7:])) <= 1e-5
    assert np.max(abs(np.asarray(ground['joint_velocity_rad_s']))) <= 1e-5
    active = [row for row in core.read(ROOT/'status/GPU_LEDGER.json')['runs'] if row['status'] == 'RUNNING']
    assert len(active) == 1 and Path(active[0]['output']) == TRAIN
    for key in ['pid','child_pid']:
        assert core.live(active[0][key],str(TRAIN)), 'Actual native source no longer live during first-update check'
    progress = core.read(TRAIN/'progress.json')
    assert progress['numerical_failures'] == 0
    recording = progress['trajectory_recording']
    assert recording['enabled'] and recording['disabled_error'] is None and recording['verified_full_public_rows'] >= 262144
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(),actual_native_child_pid=active[0]['child_pid'],
                  actual_completed_iteration=1,actual_control_env_steps=262144,source_policy_sha256=init['source_policy_sha256'],
                  source_actor_normalizer_count=init['source_actor_normalizer_count'],
                  actor_normalizer_count=init['source_actor_normalizer_count']+262144,critic_normalizer_count=262144,
                  copied_mean_actor_weights_unchanged_through_critic_warmup=True,fresh_critic_Adam_states=8,
                  fresh_critic_Adam_steps=20,fresh_exploration_std=.1,real_public_trace_recorder_enabled=True,
                  first_diagnostic_sha256=core.sha(TRAIN/'ppo/model_0.pt'),formal_acceptance=False,goal_complete=False)
    template.save_once(HERE/'FIRST_NATIVE_UPDATE_CHECK.json',result)


def guarded_training(command,expected):
    assert not template.stopping and not TRAIN.exists();template.check_sources(expected)
    checked = False
    with (HERE/'RUNNING_NATIVE_PLACEMENT075_PILOT8.guard.log').open('x') as log:
        template.child = subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2'),
                                          stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        template.write('RUNNING_NATIVE_PLACEMENT075_PILOT8',guard_pid=template.child.pid)
        try:
            while template.child.poll() is None:
                if template.stopping:
                    template.stop()
                progress = TRAIN/'progress.json'
                if not checked and progress.exists() and core.read(progress)['update'] >= 1:
                    verify_first_update();checked = True
                time.sleep(2)
            code = template.child.returncode
        except BaseException:
            template.stop()
            try:
                template.child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass  # Root must inspect this actual guard PID; never auto-restart.
            raise
    assert code == 0 and not template.stopping and checked, 'Preserve actual job/first-update failure; no automatic restart'
    template.check_sources(expected)
    return core.verify_job(TRAIN,STEPS)


def verify_training(job):
    import numpy as np
    import torch
    from scripts.initialize_centered_policy_v4 import same_tree
    from quadsmash.evaluation_v4 import independent_episode
    inputs,metrics = core.read(TRAIN/'INPUTS.json'),core.read(TRAIN/'metrics.json')
    init = validate_training_inputs(inputs)
    native = torch.load(TRAIN/'checkpoint.pt',map_location='cpu',weights_only=False)
    public = torch.load(TRAIN/'policy.pt',map_location='cpu',weights_only=False)
    assert native['iter'] == 8 and native['native']['canonical_restart'] and native['native']['total_timesteps'] == STEPS
    assert native['native']['environment']['num_envs'] == 2048
    assert not native['native']['environment']['tensors']['episode_length_buf'].any()
    model = native['model_state_dict']
    assert all(torch.isfinite(v).all() for v in model.values() if torch.is_tensor(v))
    assert int(model['actor_obs_normalizer.count']) == init['source_actor_normalizer_count']+STEPS
    assert int(model['critic_obs_normalizer.count']) == STEPS
    optimizer = native['optimizer_state_dict']
    steps = optimizer_steps(native)
    assert len(steps) == len(optimizer['state']) == 17
    for state in optimizer['state'].values():
        assert all(torch.isfinite(v).all() for v in state.values() if torch.is_tensor(v))
    assert all(value == (160 if name.startswith('critic.') else 120) for name,value in steps.items())
    assert native['infos']['learner_initialization'] == metrics['initialization'] == init
    assert native['infos']['residual_contract'] == inputs['contract']
    base = torch.load(BASE,map_location='cpu',weights_only=False)
    assert public['native_execution'] and public['residual_base_mode'] == 'frozen_public_policy' and not public['performance_inherited']
    assert same_tree(public['base_controller'],base)
    meta = public['residual_training_metadata']
    assert meta['completed_updates'] == 8 and meta['actual_control_env_steps'] == STEPS and meta['initialization'] == init
    assert meta['export_actor_max_error'] <= 2e-5 and hashlib.sha256(public['residual_actor_jit']).hexdigest() == meta['actor_jit_sha256']
    assert core.sha(TRAIN/'policy.pt') == metrics['policy_sha256']
    assert metrics['completed_updates'] == len(metrics['history']) == 8
    assert [entry['update'] for entry in metrics['history']] == list(range(1,9))
    assert [entry['actor_warmup'] for entry in metrics['history']] == [True,True]+[False]*6
    assert metrics['history'][-1]['actual_control_env_steps'] == STEPS and core.finite(metrics['history'])
    source = torch.load(Path(core.read(PLAN)['selected']['policy']),map_location='cpu',weights_only=False)
    actor = torch.jit.load(io.BytesIO(source['residual_actor_jit']),map_location='cpu').eval().state_dict()
    assert any(not torch.equal(model['actor.'+name[2:]],v) for name,v in actor.items() if name.startswith('1.'))
    protocol = core.read(ROOT/'configs/g07_full_racket_eval_v4_r3.json');protocol['_root'] = str(ROOT)
    ground = core.read(TRAIN/'initial_ground_states.json');roots = np.asarray(ground['root_state_local'])
    assert ground['all_episode_clocks_zero'] and ground['initialization'] == init
    assert roots.shape == (2048,13) and np.isfinite(roots).all()
    assert np.max(abs(roots[:,:3]-protocol['initialization']['root_position_m'])) <= 1e-5
    assert np.max(abs(roots[:,3:7]-[1.,0.,0.,0.])) <= 1e-5 and np.max(abs(roots[:,7:])) <= 1e-5
    assert np.max(abs(np.asarray(ground['joint_velocity_rad_s']))) <= 1e-5
    assert ground['source_training_indices'] == list(range(128))*16
    rows = core.read(TRAIN/'episodes.json');assert core.sha(TRAIN/'episodes.json') == core.sha(TRAIN/'episodes.partial.json')
    seen = [0]*2048
    for row in rows:
        lane = row['env_id'];assert 0 <= lane < 2048 and row['episode_index'] == seen[lane]
        assert row['ground_init'] and row['source_training_index'] == lane%128
        seen[lane] += 1
    terminal = native['native']['environment']['tensors']['episode_index'].cpu().tolist()
    assert seen == terminal and min(seen) >= 1
    outcomes = [independent_episode(row,protocol) for row in rows]
    counts = dict(completed_episodes=len(rows),air=sum(v['airborne_contact'] for v in outcomes),
                  conditional_recovery_count=sum(v['conditional_recovery'] for v in outcomes),
                  complete=sum(v['complete'] for v in outcomes),numerical_failures=sum(row['numerical_failures'] for row in rows))
    assert counts == metrics['counts'] and counts['numerical_failures'] == 0
    return dict(checked_at=datetime.now(timezone.utc).isoformat(),actual_control_env_steps=STEPS,actual_GPU_hours=job['charged_gpu_hours'],
                checkpoint_sha256=core.sha(TRAIN/'checkpoint.pt'),policy_sha256=core.sha(TRAIN/'policy.pt'),
                source_public_policy_sha256=init['source_policy_sha256'],source_actor_normalizer_count=init['source_actor_normalizer_count'],
                actor_normalizer_count=init['source_actor_normalizer_count']+STEPS,critic_normalizer_count=STEPS,
                fresh_actor_Adam_steps=120,fresh_critic_Adam_steps=160,all2048_fresh_episode_chains_exact=True,
                independent_changing_policy_counts=counts,frozen_policy_performance=False,formal_acceptance=False,goal_complete=False)


def video_spec(summary,source_iteration):
    result = template.video_spec(summary)
    for case in result['cases']:
        case['subtitle'] = f'落点反馈0.75｜阶段学习器8｜来源{source_iteration}平均动作初始化'
    result.update(scope='New frozen075 pilot8 initialized from a measured outer actor, not source performance or formal A/B.',
                  opening_title='落点反馈修订｜新阶段PPO8',
                  opening_lines=[f'从已测{source_iteration}版外层平均动作初始化；新学习器8次更新。',
                                 f'全128真实结果：空中{summary["airborne_contacts"]}、恢复{summary["conditional_recovery_count"]}、完整{summary["complete"]}。',
                                 '原六例全部保留；来源成绩不继承，原成功条件保持。'],
                  evaluation_footer=f'落点反馈075｜新阶段PPO8｜空中{summary["airborne_contacts"]}、恢复{summary["conditional_recovery_count"]}、完整{summary["complete"]}｜非正式A/B')
    result['caveats'][2] = 'New075 reward stage with only the source outer actor/normalizer copied; fresh critic/Adam/RNG/counters, one unchanged fixed old8 base.'
    return result


def main():
    assert ROOT == MAIN and PLAN.exists() and not STATE.exists()
    selection = require_completed_selection();plan = core.read(PLAN)
    assert plan['selected'] == selection['selected'] and plan['selected_policy_sha256'] == selection['selected_policy_sha256']
    expected = dict(plan['source_hashes']);expected[str(PLAN.relative_to(ROOT))] = core.sha(PLAN)
    template.check_sources(expected)
    if plan['source_ready_for_formal_review']:
        template.write('READY_FOR_ORIGINAL_FORMAL_REVIEW',new_training_skipped=True,selected=plan['selected'])
        return
    train,evaluate = commands(plan['selected'])
    assert train == plan['training_command'] and evaluate == plan['evaluation_command']
    job = guarded_training(train,expected)
    receipt = verify_training(job);template.save_once(HERE/'TRAINING_RECEIPT.json',receipt)
    audit_code = template.cpu([PYTHON,'scripts/check_full_racket_training_traces_v4.py','--run',str(TRAIN),
        '--output',str(HERE/'TRAINING_TRACE_AUDIT.json')],'trace_audit',required=False)
    template.save_once(HERE/'TRACE_AUDIT_ATTEMPT.json',dict(exit_code=audit_code,passed=audit_code == 0,formal_acceptance=False))
    template.guarded(evaluate,EVALUATION,102400,'RUNNING_FIXED_PLACEMENT075_TRAIN128',expected)
    summary = core.verify_evaluation(EVALUATION,TRAIN/'policy.pt');template.save_once(HERE/'FIXED_TRAIN128_RESULT.json',summary)
    for run,label in [(TRAIN,'placement075_training_episodes'),(EVALUATION,'placement075_fixed_episodes')]:
        template.cpu([PYTHON,'scripts/export_full_racket_train_csv_v4.py','--run',str(run),'--output-prefix',str(HERE/label)],label)
    template.save_once(VIDEO_SPEC,video_spec(summary,plan['selected']['iteration']))
    template.write('PREPARING_ACTUAL_FROZEN_VIDEO',independent_fixed_TRAIN128=summary)
    template.cpu([PYTHON,'scripts/prepare_full_racket_latest_video_v4.py','--spec',str(VIDEO_SPEC),'--output',str(SNAPSHOT)],'video_prepare')
    template.write('RENDERING_ACTUAL_VIDEO_ON_CPU')
    template.cpu([PYTHON,'scripts/render_full_racket_latest_effect_v4.py','--snapshot',str(SNAPSHOT),'--output',str(VIDEO)],'video_render')
    spec = importlib.util.spec_from_file_location('preserved_video_decode',REPORTS/'joint_placement_iter40_video_queue01/run.py')
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    template.cpu([PYTHON,'-c',module.DECODE,str(VIDEO)],'video_decode')
    template.check_sources(expected)
    template.write('READY_FOR_VISUAL_REVIEW',training=receipt,independent_fixed_TRAIN128=summary,
        selected_training_trace_audit_passed=audit_code == 0,video=str(VIDEO),video_sha256=core.sha(VIDEO),
        next_action='READY_FOR_ORIGINAL_FORMAL_REVIEW' if ready(summary) and audit_code == 0 else 'CONTINUE_FROM_ACTUAL_PILOT8_EVIDENCE',
        automatic_longer_training=False,automatic_upload_performed=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect-only',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    args = parser.parse_args()
    if args.inspect_only:
        print(json.dumps(inspect_selection()))
    else:
        assert ROOT == MAIN, 'Use the integrated original main worktree for native preparation/execution'
        with (HERE/'observer.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if args.prepare_only:
                prepare()
            else:
                signal.signal(signal.SIGTERM,template.stop);signal.signal(signal.SIGINT,template.stop)
                try:
                    main()
                except BaseException as error:
                    (HERE/'error.txt').write_text(traceback.format_exc())
                    template.write('FAILED_NATIVE_OR_FOLLOWUP',error=repr(error),
                        last_child_pid=template.child.pid if template.child is not None else None,
                        last_child_still_live=template.child is not None and template.child.poll() is None,
                        training_terminal_receipt_exists=(TRAIN/'job.json').exists())
                    raise
