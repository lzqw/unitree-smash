"""Use a future equilibrium reference through the original command-rate limit.

The static reference may be more than one command tick away. Only its causal
rate-limited trajectory is tested or optimized, under unchanged physics.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import scripts.audit_gravity_seeded_continuation_v1 as experiment
from scripts.audit_terminal_gravity_balance_v1 import gravity_target


def future_equilibrium_reference(controller,state):
    result=gravity_target(controller,state)
    if result['solved']:return result
    interface=controller.qp.interface
    # Request a future reference, not an executable instantaneous action.
    saved=interface.admitted_target_bounds
    interface.admitted_target_bounds=lambda q,dt:(interface.limits[:,0].copy(),interface.limits[:,1].copy())
    try:future=gravity_target(controller,state)
    finally:interface.admitted_target_bounds=saved
    future.update(future_reference_only=True,one_tick_static_result=result,requires_original_rate_limited_transition=True)
    return future


def rate_admitted_seed(controller,state,target,steps):
    interface=controller.qp.interface;previous=state.previous_command_target.copy();rows=[]
    target=np.asarray(target);dt=controller.settings.control_dt_s
    if np.any(target<interface.limits[:,0]) or np.any(target>interface.limits[:,1]):raise ValueError('Reference outside hardware range')
    for _ in range(steps):
        command=target.copy();command[12:]=np.clip(command[12:],previous[12:]-interface.arm_rate*dt,previous[12:]+interface.arm_rate*dt)
        rows.append(command);previous=command
    return np.asarray(rows)


source=inspect.getsource(experiment.query)
anchor="seed=np.tile(proposal['target'],(40,1));"
if source.count(anchor)!=1:raise ValueError('Changed continuation seed interface')
source=source.replace(anchor,"seed=rate_admitted_seed(c,state,proposal['target'],40);")
namespace=dict(vars(experiment));namespace.update(gravity_target=future_equilibrium_reference,rate_admitted_seed=rate_admitted_seed)
exec(compile(source,'<ramped-gravity-continuation>','exec'),namespace)
generated_query=namespace['query']


def query(args):return generated_query(args)


experiment.query=query
if __name__=='__main__':
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--output',type=Path,required=True)
    args,_=parser.parse_known_args();experiment.main()
    result=json.loads(args.output.read_text());result.update(original_arm_rate_enforced=True,
        future_static_reference_is_not_instantaneous_action=True)
    result['source_sha256'][str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
