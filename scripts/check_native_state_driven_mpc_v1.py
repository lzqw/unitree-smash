"""Static admission checks for the library-independent state-driven MPC."""
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
    markers = [
        'native_pure_control_state_driven_mpc_v1',
        'NativePureControlStateDrivenMPCV1',
    ]
    missing = [m for m in markers if m not in text]
    cfg = json.loads(a.configuration.read_text())
    if cfg.get('controller') != 'pure_control_state_driven_mpc_v1':
        missing.append('configuration.controller=state_driven_mpc_v1')
    root = Path(__file__).resolve().parents[1]
    impl = (root / 'scripts' / 'pure_control_state_driven_mpc_v1.py').read_text()
    adapter = (root / 'scripts' / 'native_pure_control_state_driven_mpc_v1.py').read_text()
    for marker in ('reference_library_required=False', 'train_case_required=False',
                   'state_driven_mpc:', 'predict_self_contact_spheres'):
        if marker not in impl + adapter:
            missing.append('implementation.' + marker)
    result = dict(passed=not missing, missing=missing,
                  native_execution=False, physical_limits_changed=False,
                  reference_library_required=False)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    if missing:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
