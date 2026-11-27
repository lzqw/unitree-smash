"""GPU-guarded native PPO training followed by frozen fixed128 evaluation."""
import argparse
import json
from pathlib import Path

from scripts.run_recovery_training_round import run_stage
from scripts.run_supermimic_recovery_baseline import dump, summarize, verify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','companion','panel','baseline','output','gpu-lock'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--iterations', type=int, default=1)
    p.add_argument('--ground-prefix', action='store_true')
    args = p.parse_args()
    if args.iterations < 1:
        p.error('iterations must be positive')
    verify([args.checkpoint,args.companion], synchronize_recovery=True)
    args.output.mkdir(parents=True,exist_ok=False)
    status = args.output/'status.json'
    command = ['scripts/evaluate_supermimic_v1.py','--headless','--physics-device','cuda:0','--device','cuda:0',
               '--kit_args=--renderer/multiGpu/enabled=false','--recovery-full-tail']
    try:
        with args.gpu_lock.open('a') as lock:
            policy = args.checkpoint
            resume = []
            for start in range(0, args.iterations, 5):
                count = min(5, args.iterations-start)
                block = args.output / f'block_{start+count:02d}'
                block.mkdir()
                train = run_stage([*command,'--checkpoint',str(policy),'--recovery-ppo-panel',str(args.panel),
                                   '--recovery-probe-companion',str(args.companion),
                                   '--recovery-ppo-iterations',str(count),*resume,
                                   *(['--recovery-ground-prefix'] if args.ground_prefix else [])],block/'train',lock,status)
                if (train/'early_stop.json').exists():
                    dump(status, dict(state='stopped_by_held32', training=str(train),
                                      evidence=json.loads((train/'early_stop.json').read_text())))
                    return
                policy = train/'policy.pt'
                resume = ['--recovery-ppo-resume',str(train/'optimizer.pt')]
                eval_dir = run_stage([*command,'--checkpoint',str(policy),'--research-fixed128'],
                                     block/'fixed128',lock,status)
                compare(args.baseline, eval_dir, block/'comparison.json')
        compare(args.baseline, eval_dir, args.output/'comparison.json')
        dump(status,dict(state='complete',training=str(train),evaluation=str(eval_dir)))
    except Exception as exc:
        dump(status,dict(state='stopped',reason=str(exc)))
        raise


def compare(baseline, eval_dir, output):
    if (baseline/'initial_states.json').read_bytes() != (eval_dir/'initial_states.json').read_bytes():
        raise ValueError('Fixed128 initial states differ')
    before = json.loads((baseline/'episodes_summary.json').read_text())
    after = json.loads((eval_dir/'episodes_summary.json').read_text())
    if [r['case_id'] for r in before] != [r['case_id'] for r in after]:
        raise ValueError('Fixed128 case identities differ')
    flips = {}
    for k in ('physical_recovery_success','strict_physical_recovery','body_ground_seen',
              'qualified_airborne_string_contact','robot_net_fault','ball_side_complete_except_recovery','complete5'):
        flips[k] = {name:[b['case_id'] for b,a in zip(before,after) if bool(b[k])==x and bool(a[k])==y]
                    for name,x,y in [('gained',False,True),('lost',True,False)]}
    dump(output,dict(before=summarize(baseline),after=summarize(eval_dir),flips=flips))

if __name__ == '__main__':
    main()
