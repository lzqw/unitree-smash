"""Save current-state command integrity separately from cohort isolation."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_native_current_state_dispatch_v1 import verify_feedback_dispatch


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run','bundle','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve prior audit')
    result=verify_feedback_dispatch(a.run,a.bundle)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
