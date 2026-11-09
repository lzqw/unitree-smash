import argparse
import ast
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--entry', type=Path, required=True)
p.add_argument('--configuration', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()

entry = a.entry.read_text()
ast.parse(entry)
cfg = json.loads(a.configuration.read_text())
missing = []
for marker in (
    'pure_control_state_driven_long_mpc_v1_qp_continuity_hold',
    'NativePureControlStateDrivenLongMPCV1QPContinuityHold',
):
    if marker not in entry:
        missing.append(marker)
if cfg.get('controller') != 'pure_control_state_driven_long_mpc_v1_qp_continuity_hold':
    missing.append('configuration.controller')
result = dict(
    passed=not missing,
    missing=missing,
    native_execution=False,
    physical_limits_changed=False,
    reference_library_required=False,
)
a.output.write_text(json.dumps(result, indent=2) + '\n')
if missing:
    raise SystemExit(1)
