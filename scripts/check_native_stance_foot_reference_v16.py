"""Static admission checks for the command-continuous V16 native entry."""
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
    required = [
        'native_pure_control_stance_foot_reference_v16',
        'NativePureControlStanceFootReferenceV16',
        'pure_control_stance_foot_reference_v16',
    ]
    missing = [x for x in required if x not in text]
    cfg = json.loads(a.configuration.read_text())
    if cfg.get('controller') != 'pure_control_stance_foot_reference_v16':
        missing.append('configuration.controller=v16')
    root = Path(__file__).resolve().parents[1]
    implementation = (root / 'scripts' / 'pure_control_stance_foot_reference_v16.py').read_text()
    for marker in ('command_continuous_fallback', 'fallback_certified=False',
                   'hold_last_admitted_target_with_velocity_damping'):
        if marker not in implementation:
            missing.append('implementation.' + marker)
    result = dict(passed=not missing, missing=missing,
                  native_execution=False, physical_limits_changed=False)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    if missing:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
