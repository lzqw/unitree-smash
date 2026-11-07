"""Run the matched recovery refinement with explicit terminal inequalities."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import scripts.audit_microstep_task_refinement_v1 as experiment
import scripts.audit_microstep_recovery_v1 as continuation
from scripts.terminal_recovery_constraints_v1 import make_proposal

# This standalone process (including spawned workers) owns these bindings.
# The existing files and running native jobs are unchanged.
experiment.make_proposal=make_proposal
continuation.make_proposal=make_proposal

if __name__=='__main__':
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--output',type=Path,required=True)
    args,_=parser.parse_known_args()
    experiment.main()
    result=json.loads(args.output.read_text())
    result.update(explicit_terminal_recovery_inequalities=True,
        linear_terminal_interior_margin=1e-5,formal_terminal_conditions_unchanged=True)
    result['source_sha256'][str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
