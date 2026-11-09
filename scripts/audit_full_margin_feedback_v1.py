"""Run full V15 feedback only for complete, improved margin-search candidates.

Original actual initial states, physical limits and full episode endpoint are
required. Model improvement is not a native result or a valid expert label.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.evaluate_state_settling_feedback_v1 import evaluate
from scripts.retry_budgeted_settling_feedback_v1 import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['search','configuration','run','output']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();read=lambda f:json.loads(Path(f).read_text())
    result=read(a.search/'result.json');inputs=read(a.search/'inputs.json')
    if a.output.exists() or result['state']!='COMPLETED_FULL_RECOVERY_MARGIN_SEARCH':
        raise ValueError('Fresh output and completed margin search required')
    if read(a.run/'job.json')['status']!='SUCCEEDED':raise ValueError('Completed native history required')
    pins=dict(inputs['source_sha256'])
    for f,h in pins.items():
        if sha(f)!=h:raise ValueError('Changed source '+f)
    if pins.get(str(a.configuration.resolve()))!=sha(a.configuration):raise ValueError('Keep the original interface')
    cfg=read(a.configuration);nominal=read(a.run/'native_command_inputs.json')['default_joint_position_rad']
    selected=[];excluded=[]
    with np.load(a.run/'pure_control_commands.npz') as z:
        for item in result['results']:
            key=item['state_key'];path=a.search/('state_'+key.replace(':','_')+'.json');data=read(path);fine=data['fine'];record=data['initial_actual_state']
            improved=(item['complete_stable_model'] and item['improved'] and
                      item['best_minimum_nonfoot_m']>item['baseline_minimum_nonfoot_m']+1e-6)
            if not improved:
                excluded.append(dict(state_key=key,reason='No verified complete model margin improvement'));continue
            if (data['state_key']!=key or data['native_success'] or data['valid_expert_labels'] or
                not fine['model_path_complete'] or fine['failure'] is not None or fine['terminal_stable_s']<.8 or fine['substeps']!=160):
                raise ValueError('Incomplete or incorrectly labelled candidate')
            if (len(fine['trajectory'])!=800-record['tick'] or
                any(row['step']!=i for i,row in enumerate(fine['trajectory'])) or
                fine['sampled_minimum_nonfoot_m']!=item['best_minimum_nonfoot_m']):
                raise ValueError('Candidate and final summary differ')
            ids=np.flatnonzero((z['lane']==record['case'])&(z['tick']==record['tick']))
            if len(ids)!=1 or any(not np.array_equal(z[k][ids[0]],record[k]) for k in ['q','v','previous','ground_force']):
                raise ValueError('Candidate is not rooted at the actual release')
            selected.append((path,data))
    if not selected:raise ValueError('No improved candidates; feedback should not be launched')
    a.output.mkdir(parents=True);snapdir=a.output/'candidate_snapshots';snapdir.mkdir()
    paths=[]
    for path,data in selected:
        # Explicit metadata for future readers; target trajectories are intact.
        data['fine'].update(nodes_describe_dense_correction=True,nodes_describe_suffix_only=False,
            command_representation='full model-feedback warmstart plus PCHIP full18 correction',prefix_steps=0)
        destination=snapdir/path.name;write(destination,data);paths.append(destination)
    files=[Path(__file__),a.search/'inputs.json',a.search/'result.json',a.search/'seed_check.json',a.configuration,
           a.run/'job.json',a.run/'native_command_inputs.json',a.run/'pure_control_commands.npz',*paths]+[p for p,_ in selected]
    files += [Path(m.__file__) for n,m in list(sys.modules.items()) if n.startswith('scripts.') and getattr(m,'__file__',None)]
    pins.update({str(f.resolve()):sha(f) for f in files})
    write(a.output/'inputs.json',dict(source_sha256=pins,excluded=excluded,selected=[data['state_key'] for _,data in selected],
        physical_substeps=160,scoring_substeps=16,max_seconds=2000,workers=3,controller='V15',
        native_execution=False,training=False,acceptance_changed=False))
    start=time.monotonic();rows=[]
    def progress(state):write(a.output/'status.json',dict(state=state,pid=os.getpid(),completed=len(rows),total=len(paths),results=rows))
    progress('RUNNING_REPLANNED_FEEDBACK_MODEL')
    with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn')) as pool:
        jobs=[pool.submit(evaluate,cfg,nominal,str(path),'plan_end',start+2000) for path in paths]
        for job in as_completed(jobs):
            row=job.result();write(a.output/('state_'+row['state_key'].replace(':','_')+'.json'),row)
            rows.append({k:v for k,v in row.items() if k not in ['trajectory','initial_actual_state']})
            progress('RUNNING_REPLANNED_FEEDBACK_MODEL')
    for f,h in pins.items():
        if sha(f)!=h:raise ValueError('Changed feedback input '+f)
    final=dict(state='COMPLETED_REPLANNED_FEEDBACK_MODEL',results=rows,excluded=excluded,
        elapsed_s=time.monotonic()-start,phase_logging='post_interval',native_success=False,valid_expert_labels=False)
    write(a.output/'result.json',final);progress(final['state']);print(json.dumps(final))


if __name__=='__main__':main()
