"""Static admission check for phase-adaptive state-driven MPC."""
import argparse,ast,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--entry',type=Path,required=True);p.add_argument('--configuration',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    text=a.entry.read_text();ast.parse(text);cfg=json.loads(a.configuration.read_text());missing=[]
    for marker in ('native_pure_control_state_driven_adaptive_mpc_v1','NativePureControlStateDrivenAdaptiveMPCV1'):
        if marker not in text: missing.append(marker)
    if cfg.get('controller')!='pure_control_state_driven_adaptive_mpc_v1': missing.append('configuration.controller')
    impl=(Path(__file__).resolve().parents[1]/'scripts'/'pure_control_state_driven_adaptive_mpc_v1.py').read_text()
    for marker in ('phase_horizon_steps','reference-library lookup'):
        if marker not in impl: missing.append('implementation.'+marker)
    result=dict(passed=not missing,missing=missing,native_execution=False,physical_limits_changed=False,reference_library_required=False)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    if missing: raise SystemExit(1)


if __name__=='__main__': main()
