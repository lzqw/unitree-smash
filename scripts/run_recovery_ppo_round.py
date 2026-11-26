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
    args = p.parse_args()
    verify([args.checkpoint,args.companion])
    args.output.mkdir(parents=True,exist_ok=False)
    status = args.output/'status.json'
    command = ['scripts/evaluate_supermimic_v1.py','--headless','--physics-device','cuda:0','--device','cuda:0',
               '--kit_args=--renderer/multiGpu/enabled=false','--recovery-full-tail']
    try:
        with args.gpu_lock.open('a') as lock:
            train = run_stage([*command,'--checkpoint',str(args.checkpoint),'--recovery-ppo-panel',str(args.panel),
                               '--recovery-probe-companion',str(args.companion)],args.output/'train',lock,status)
            eval_dir = run_stage([*command,'--checkpoint',str(train/'policy.pt'),'--research-fixed128'],
                                 args.output/'fixed128',lock,status)
        if (args.baseline/'initial_states.json').read_bytes() != (eval_dir/'initial_states.json').read_bytes():
            raise ValueError('Fixed128 initial states differ')
        before = json.loads((args.baseline/'episodes_summary.json').read_text())
        after = json.loads((eval_dir/'episodes_summary.json').read_text())
        if [r['case_id'] for r in before] != [r['case_id'] for r in after]:
            raise ValueError('Fixed128 case identities differ')
        flips = {}
        for k in ('physical_recovery_success','strict_physical_recovery','body_ground_seen',
                  'qualified_airborne_string_contact','robot_net_fault','ball_side_complete_except_recovery','complete5'):
            flips[k] = {name:[b['case_id'] for b,a in zip(before,after) if bool(b[k])==x and bool(a[k])==y]
                        for name,x,y in [('gained',False,True),('lost',True,False)]}
        dump(args.output/'comparison.json',dict(before=summarize(args.baseline),after=summarize(eval_dir),flips=flips))
        dump(status,dict(state='complete',training=str(train),evaluation=str(eval_dir)))
    except Exception as exc:
        dump(status,dict(state='stopped',reason=str(exc)))
        raise


if __name__ == '__main__':
    main()
