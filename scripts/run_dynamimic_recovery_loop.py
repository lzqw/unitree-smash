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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','companion','panel','validation','asset','baseline','output','gpu-lock'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--initial-native', type=Path, help='Reuse a completed first native collection')
    parser.add_argument('--blocks', type=int, default=8, choices=range(1,9))
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True,exist_ok=False)
    status = root/'status.json'
    native_flags = ['scripts/evaluate_supermimic_v1.py','--headless','--physics-device',args.device,
                    '--device',args.device,'--kit_args=--renderer/multiGpu/enabled=false','--recovery-full-tail']
    validate_flags = ['--dynamimic-validation',args.validation,'--dynamimic-companion',args.companion,
                      '--dynamimic-exclude-training',args.panel]
    current = args.checkpoint
    incumbent = None
    history = []
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
                critic = run('value',['scripts/fit_dynamimic_recovery_critic.py','--checkpoint',current,
                    '--native',native, '--device',args.device, *(['--resume',old_critic/'critic.pt'] if old_critic else [])])
                wm = run('wm',['scripts/fit_dynamimic_recovery_wm.py','--asset',args.asset,'--native',native,
                    '--device',args.device,*(['--resume',old_wm/'ensemble.pt'] if old_wm else []),
                    *(x for source in history[-3:] for x in ('--history-native',source))])
                validation = json.loads((wm/'validation.json').read_text())
                if not validation['passed']:
                    dump(status,dict(state='wm_gate_failed', block=block, retained=str(current),
                                     native=str(native), evidence=str(wm/'validation.json')))
                    return
                old_wm, old_critic = wm, critic
                history.append(native)
                fit = run('bptt',['scripts/train_dynamimic_recovery_bptt.py','--native',native,
                    '--wm',wm,'--critic',critic,'--asset',args.asset,'--checkpoint',current,
                    '--companion',args.companion,'--device',args.device])
                if incumbent is None:
                    base = run('frozen_native64',[*native_flags,*validate_flags,'--checkpoint',current,
                                                   '--dynamimic-alternate',current])
                    incumbent = json.loads((base/'dyna_validation.json').read_text())
                    # Each half keeps its own lane positions for the paired comparison.
                candidates = [fit/'conservative.pt',fit/'moderate.pt']
                evaluated = run('candidates_native64',[*native_flags,*validate_flags,'--checkpoint',candidates[0],
                                                        '--dynamimic-alternate',candidates[1]])
                result = json.loads((evaluated/'dyna_validation.json').read_text())
                if any(result[k] != incumbent[k] for k in ('case_ids','reset_sha256','companion_sha256')):
                    raise ValueError('Native promotion panel changed')
                indices = [i for i in range(2) if choose_candidate(incumbent['metrics'][i], [result['metrics'][i]]) is not None]
                selected = max(indices, key=lambda i:(-result['metrics'][i]['body'],result['metrics'][i]['strict'],
                                                       result['metrics'][i]['relaxed'],-i),default=None)
                previous = current
                if selected is not None:
                    current = candidates[selected]
                    incumbent = None  # Rescore the promoted policy in both native lane positions.
                dump(folder/'decision.json',dict(comparison=result, selected=selected, previous=str(previous),
                                                 retained=str(current)))
                if block==1 or block%2==0 or block==args.blocks:
                    frozen = run('fixed128',[*native_flags,'--checkpoint',current,'--research-fixed128'])
                    if (frozen/'initial_states.json').read_bytes() != (args.baseline/'initial_states.json').read_bytes():
                        raise ValueError('Fixed128 starts changed')
                    score = summarize(frozen)
                    dump(folder/'fixed128.json',score)
                    compare_fixed128(args.baseline, frozen, folder/'fixed128_paired.json')
                    if block==1 and not (score['body_ground_given_contact']<=63 and score['strict_physical_recovery']>=16
                            and score['relaxed_physical_recovery']>=35 and score['relaxed_ball_side']>=95
                            and score['robot_net']==0 and score['qualified_contacts']==108):
                        dump(status,dict(state='first_block_native_gate_failed',evidence=str(folder/'fixed128.json'),
                                         retained=str(args.checkpoint)))
                        return
            dump(status,dict(state='blocks_evaluated', retained=str(current), blocks=args.blocks))
        except Exception as exc:
            dump(status,dict(state='stopped',reason=str(exc),retained=str(current)))
            raise


if __name__ == '__main__':
    main()
