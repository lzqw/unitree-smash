"""Guarded native Recovery Dyna blocks with deterministic paired promotion."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.run_recovery_ppo_round import compare as compare_fixed128
from scripts.run_recovery_training_round import run_stage
from scripts.run_supermimic_recovery_baseline import dump, summarize


def choose_candidate(baseline, metrics):
    accepted = []
    for i, score in enumerate(metrics):
        if (score['qualified'] == baseline['qualified'] and score['net'] == 0 and score['ball'] >= baseline['ball']
                and (score['body'] < baseline['body'] or score['strict'] > baseline['strict']
                     or score['relaxed'] > baseline['relaxed'])):
            accepted.append(i)
    return max(accepted, key=lambda i:(-metrics[i]['body'], metrics[i]['strict'], metrics[i]['relaxed'],
                                       metrics[i]['ball'], -i), default=None)


def first_block_candidate_to_score(selected, scores):
    if selected is not None:
        return selected
    # On rollback, score one rejected proposal for diagnosis. This result
    # cannot promote it; the incumbent already has a full fixed128 baseline.
    return max(range(len(scores)), key=lambda i: (-scores[i]['net'], scores[i]['ball'],
               scores[i]['qualified'], -scores[i]['body'], scores[i]['strict'], scores[i]['relaxed'], -i))


def passes_fixed128(score):
    return (score['qualified_contacts'] == 108 and score['relaxed_ball_side'] >= 95 and score['robot_net'] == 0
            and score['body_ground_given_contact'] <= 63 and score['strict_physical_recovery'] >= 16
            and score['relaxed_physical_recovery'] >= 35)


def retention_candidates(folder, behavior, companion):
    receipt = json.loads((folder/'fit.json').read_text())
    if receipt['behavior'] != behavior or receipt['companion'] != companion:
        raise ValueError('Retention fit belongs to a different behavior or companion')
    paths = []
    for label, budget in (('conservative', .020), ('moderate', .035)):
        path = folder/(label+'.pt')
        fit = receipt['candidates'][label]
        if (hashlib.sha256(path.read_bytes()).hexdigest() != fit['after']
                or not 0 < fit['rms'] <= budget or not 0 <= fit['replay_max_error'] <= 2e-6):
            raise ValueError(f'Invalid {label} retention candidate or action replay')
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','companion','panel','validation','asset','baseline','output','gpu-lock'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--initial-native', type=Path, help='Reuse a completed first native collection')
    parser.add_argument('--prior-native', type=Path, action='append', default=[],
                        help='Earlier native training archives for the first WM refresh')
    parser.add_argument('--initial-wm', type=Path, help='Accepted first-block WM folder for the same native archive')
    parser.add_argument('--initial-critic', type=Path, help='Native tail fit folder for the same behavior policy')
    parser.add_argument('--initial-validation-baseline', type=Path, help='Completed two-slot baseline for the same native validation setup')
    parser.add_argument('--retain-success-actions', action='store_true',
                        help='Fit the Recovery output layer to preserve successful native actions after BPTT')
    parser.add_argument('--initial-retention', type=Path,
                        help='Reuse previously fitted and replay-verified first-block retention candidates')
    parser.add_argument('--native-body-fault', action='store_true')
    parser.add_argument('--blocks', type=int, default=8, choices=range(1,9))
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if (args.initial_wm or args.initial_critic) and not args.initial_native:
        parser.error('First-block fit reuse requires its native archive')
    if args.initial_retention and not (args.retain_success_actions and args.initial_native
                                      and args.initial_wm and args.initial_critic):
        parser.error('Retention reuse requires the original native, WM and Critic and --retain-success-actions')
    if args.prior_native and (args.initial_wm or len(args.prior_native) > 3):
        parser.error('Up to three prior archives may be used when fitting the first WM')
    root = args.output
    root.mkdir(parents=True,exist_ok=False)
    status = root/'status.json'
    native_flags = ['scripts/evaluate_supermimic_v1.py','--headless','--physics-device',args.device,
                    '--device',args.device,'--kit_args=--renderer/multiGpu/enabled=false','--recovery-full-tail']
    validate_flags = ['--dynamimic-validation',args.validation,'--dynamimic-companion',args.companion,
                      '--dynamimic-exclude-training',args.panel]
    current = args.checkpoint
    confirmed = current
    incumbent = None
    history = list(args.prior_native)
    old_wm, old_critic = None,None

    with args.gpu_lock.open('a') as lock:
        try:
            for block in range(1,args.blocks+1):
                folder = root/f'block_{block:02d}'
                folder.mkdir()

                def run(label, command):
                    return run_stage([str(arg) for arg in command],folder/label,lock,status)

                native = args.initial_native if block==1 and args.initial_native else run('native',[
                    *native_flags,'--checkpoint',current,'--recovery-ppo-panel',args.panel,
                    '--recovery-probe-companion',args.companion,'--dynamimic-capture'])
                manifest = json.loads((native/'trajectories/manifest.json').read_text())
                current_sha = hashlib.sha256(current.read_bytes()).hexdigest()
                if manifest['checkpoint_sha256'] != current_sha or manifest['companion_sha256'] != hashlib.sha256(args.companion.read_bytes()).hexdigest():
                    raise ValueError('Native collection policy mismatch')
                critic = (args.initial_critic if block == 1 else None) or run('value',['scripts/fit_dynamimic_recovery_critic.py','--checkpoint',current,
                    '--native',native, '--device',args.device, *(['--resume',old_critic/'critic.pt'] if old_critic else [])])
                wm = (args.initial_wm if block == 1 else None) or run('wm',['scripts/fit_dynamimic_recovery_wm.py','--asset',args.asset,'--native',native,
                    '--device',args.device,*(['--resume',old_wm/'ensemble.pt'] if old_wm else []),
                    *(x for source in history[-3:] for x in ('--history-native',source))])
                validation = json.loads((wm/'validation.json').read_text())
                if not validation['passed']:
                    dump(status,dict(state='wm_gate_failed', block=block, retained=str(confirmed),
                                     native=str(native), evidence=str(wm/'validation.json')))
                    return
                old_wm, old_critic = wm, critic
                history.append(native)
                retention = args.initial_retention if block == 1 else None
                if retention is None:
                    fit = run('bptt',['scripts/train_dynamimic_recovery_bptt.py','--native',native,
                        '--wm',wm,'--critic',critic,'--asset',args.asset,'--checkpoint',current,
                        '--companion',args.companion,'--device',args.device,
                        *(['--native-body-fault'] if args.native_body_fault else [])])
                    if args.retain_success_actions:
                        retention = run('retention',['scripts/fit_dynamimic_recovery_retention.py','--native',native,
                            '--checkpoint',current,'--companion',args.companion,'--bptt',fit])
                if retention is not None:
                    candidates = retention_candidates(retention, current_sha, manifest['companion_sha256'])
                else:
                    candidates = [fit/'conservative.pt',fit/'moderate.pt']
                if incumbent is None:
                    base = (args.initial_validation_baseline if block == 1 else None) or run('frozen_native64',[*native_flags,*validate_flags,'--checkpoint',current,
                                                   '--dynamimic-alternate',current])
                    incumbent = json.loads((base/'dyna_validation.json').read_text())
                    if incumbent['checkpoints'] != [current_sha, current_sha]:
                        raise ValueError('Both validation reference slots must use the incumbent')
                    # Each half keeps its own lane positions for the paired comparison.
                evaluated = run('candidates_native64',[*native_flags,*validate_flags,'--checkpoint',candidates[0],
                                                        '--dynamimic-alternate',candidates[1]])
                result = json.loads((evaluated/'dyna_validation.json').read_text())
                if base is not None and any((base/file).read_bytes() != (evaluated/file).read_bytes()
                                            for file in ('task_contract.json', 'actuator_contract.json', 'policy_execution.json')):
                    raise ValueError('Candidate and reference validation contracts differ')
                if any(result[k] != incumbent[k] for k in ('case_ids','reset_sha256','companion_sha256')):
                    raise ValueError('Native promotion panel changed')
                indices = [i for i in range(2) if choose_candidate(incumbent['metrics'][i], [result['metrics'][i]]) is not None]
                selected = max(indices, key=lambda i:(-result['metrics'][i]['body'],result['metrics'][i]['strict'],
                                                       result['metrics'][i]['relaxed'],-i),default=None)
                previous = current
                if selected is not None:
                    current = candidates[selected]
                    incumbent = None  # Rescore the promoted policy in both native lane positions.
                decision = dict(comparison=result, selected=selected, previous=str(previous),
                                native64_choice=str(current), retained=str(confirmed))
                dump(folder/'decision.json', decision)
                if block==1 or block%2==0 or block==args.blocks:
                    checkpoint_to_score = (candidates[first_block_candidate_to_score(selected, result['metrics'])]
                                           if block == 1 else current)
                    decision.update(fixed128_checkpoint=str(checkpoint_to_score),
                                    fixed128_diagnostic_only=block == 1 and selected is None)
                    dump(folder/'decision.json', decision)
                    frozen = run('fixed128',[*native_flags,'--checkpoint',checkpoint_to_score,'--research-fixed128'])
                    if (frozen/'initial_states.json').read_bytes() != (args.baseline/'initial_states.json').read_bytes():
                        raise ValueError('Fixed128 starts changed')
                    score = summarize(frozen)
                    dump(folder/'fixed128.json',score)
                    compare_fixed128(args.baseline, frozen, folder/'fixed128_paired.json')
                    passed = passes_fixed128(score) and not decision['fixed128_diagnostic_only']
                    decision['fixed128_passed'] = passed
                    if block == 1:
                        decision['first_block_passed'] = passed
                    if passed:
                        confirmed = current
                    else:
                        current = confirmed
                        incumbent = None
                    decision['retained'] = str(confirmed)
                    dump(folder/'decision.json', decision)
                    if block == 1 and not passed:
                        dump(status,dict(state='first_block_native_gate_failed',evidence=str(folder/'fixed128.json'),
                                         retained=str(confirmed)))
                        return
            dump(status,dict(state='blocks_evaluated', retained=str(current), blocks=args.blocks))
        except Exception as exc:
            dump(status,dict(state='stopped',reason=str(exc),retained=str(confirmed)))
            raise


if __name__ == '__main__':
    main()
