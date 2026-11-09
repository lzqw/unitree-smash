"""Static admission check for the extended library-independent MPC."""
import argparse
import ast
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--entry', type=Path, required=True)
    p.add_argument('--configuration', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    text = a.entry.read_text()
    ast.parse(text)
    cfg = json.loads(a.configuration.read_text())
    missing = []
    for marker in ('native_pure_control_state_driven_long_mpc_v2',
                   'NativePureControlStateDrivenLongMPCV2'):
        if marker not in text: missing.append(marker)
    if cfg.get('controller') != 'pure_control_state_driven_long_mpc_v2':
        missing.append('configuration.controller=state_driven_long_mpc_v2')
    impl = (Path(__file__).resolve().parents[1] / 'scripts' /
            'pure_control_state_driven_long_mpc_v2.py').read_text()
    for marker in ('horizon_steps = 64', 'PureControlStateDrivenLongMPCV1'):
        if marker not in impl: missing.append('implementation.' + marker)
    result = dict(passed=not missing, missing=missing,
                  native_execution=False, physical_limits_changed=False,
                  reference_library_required=False)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    if missing: raise SystemExit(1)


if __name__ == '__main__': main()
