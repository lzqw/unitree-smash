#!/usr/bin/env python3
"""Verify every completed candidate now, including the final cycle; no training."""
import argparse,json
from pathlib import Path
from quadsmash.rl.supermimic_long_run_contract_v1 import verify_native_promotion
from quadsmash.rl.supermimic_policy_execution_v1 import contract


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metrics',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    try:result=dict(accepted=True,**verify_native_promotion(a.metrics,a.checkpoint,expected_execution=contract()))
    except (ValueError,KeyError,TypeError) as error:result=dict(accepted=False,error_type=type(error).__name__,reason=str(error))
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if not result['accepted']:raise SystemExit(21)
if __name__=='__main__':main()
