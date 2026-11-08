"""Controlled model comparison with identical foot/body goals and feedback.

Only the retained actuator command preload fraction changes. Predictions
start at the same measured state/history and keep original physical limits.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import search_loaded_foot_recovery_v2  # Same configuration in spawned workers.
from scripts.search_foot_anchored_recovery_v1 import evaluate
from scripts.check_constrained_recovery_suffix_v1 import source_context
from scripts.search_preserved_early_support_v1 import initialize
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh comparison required')
    read=lambda f:json.loads(Path(f).read_text());sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    source=read(a.source)
    # Partial snapshots are explicit, pinned inputs; not final search results.
    params=np.asarray(source['best']['parameters'])
    inputs=read(a.source.parent/'inputs.json');original=read(inputs['source_inputs'])
    cfg=original['configuration'];query=original['query'];tick=int(cfg['tick'])
    root=read(read(cfg['model_plan'])['source'])
    _,_,_,audit,context=source_context(root['configuration']['source'])
    params_zero=params.copy();params_zero[15]=0.
    pins=dict(inputs['source_sha256'])
    for f in [a.source,Path(__file__),ROOT/'scripts/search_loaded_foot_recovery_v2.py']:
        pins[str(f.resolve())]=sha(f)
    for f,h in pins.items():
        if sha(f)!=h:raise ValueError('Changed input '+f)
    began=time.monotonic();deadline=began+180.
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=context) as pool:
        jobs=[pool.submit(evaluate,x,query,tick,audit,160,deadline) for x in [params,params_zero]]
        rows=[job.result() for job in jobs]
    for f,h in pins.items():
        if sha(f)!=h:raise ValueError('Changed comparison source '+f)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(plain(dict(scope=__doc__,source_sha256=pins,
        with_preload=rows[0],without_preload=rows[1],only_changed_parameter='command_preload_fraction',
        wall_seconds=time.monotonic()-began,native_execution=False,valid_expert_labels=False)),indent=2)+'\n')
    print(json.dumps([dict(preload=x['parameters'][15],steps=x['result']['completed_model_steps'],failure=x['result']['failure']) for x in rows]))


if __name__=='__main__':main()
