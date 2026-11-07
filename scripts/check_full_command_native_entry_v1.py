#!/usr/bin/env python3
"""Execute the actual evaluator factory AST without importing Isaac."""
import argparse
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import scripts.check_interval_command_native_entry_v1 as previous_check
from scripts.landing_full_command_interval_mpc_v1 import LandingFullCommandIntervalMPCV1


def check(source):
    previous_check.check(source)
    tree=ast.parse(source.read_text());values={}
    for node in tree.body:
        if not isinstance(node,ast.Expr) or not isinstance(node.value,ast.Call):continue
        call=node.value
        if not isinstance(call.func,ast.Attribute) or call.func.attr!='add_argument':continue
        if not call.args or not isinstance(call.args[0],ast.Constant):continue
        name=call.args[0].value
        if not isinstance(name,str) or not name.startswith('--landing-mpc-'):continue
        kw={x.arg:x.value for x in call.keywords}
        if 'default' in kw:values[name[2:].replace('-','_')]=ast.literal_eval(kw['default'])
        elif 'action' in kw and ast.literal_eval(kw['action'])=='store_true':values[name[2:].replace('-','_')]=False
    assert values['landing_mpc_full_command_damper'] is False
    assert sum(k.arg=='landing_mpc_full_command_damper' for n in ast.walk(tree)
               if isinstance(n,ast.Call) for k in n.keywords)==2
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SharedTargetResidualController')
    init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    assert 'landing_mpc_full_command_damper' in [a.arg for a in init.args.args+init.args.kwonlyargs]
    factory=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='configure_landing_mpc')
    namespace=vars(previous_check).copy();namespace['LandingFullCommandIntervalMPCV1']=LandingFullCommandIntervalMPCV1
    exec(compile(ast.Module(body=[factory],type_ignores=[]),str(source),'exec'),namespace)
    values.update(landing_mpc_cases={1},landing_mpc_command_history_interface=True,
        landing_mpc_continuous_reference=True,landing_mpc_support_margin=True,
        landing_mpc_executor_replay=True,landing_mpc_arm_target_rate=10.,_landing_mpc_horizon_steps=40,
        landing_mpc_interval_command_qp=True,landing_mpc_interval_active_set=True,
        landing_mpc_free_arm=True,landing_mpc_full_command_damper=True)
    subject=SimpleNamespace(**values)
    namespace['configure_landing_mpc'](subject,np.tile([-4.,4.],(18,1)),np.zeros(18))
    assert type(subject.landing_mpc) is LandingFullCommandIntervalMPCV1
    assert not subject.landing_mpc.qp.preserve_arm_command
    assert subject.landing_mpc.qp.damper_gain_s_inv==20.
    assert subject.landing_mpc.qp.substeps==16
    assert subject.landing_mpc.qp.interface.arm_rate==10.
    np.testing.assert_array_equal(subject.landing_mpc.qp.interface.velocity[12:],np.full(6,5.))
    print('Explicit full-command factory, default legacy branches, physical limits and manifest plumbing passed; no native execution')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'scripts/evaluate_shared_target_ground64_persistent_recovery_v1.py')
    check(p.parse_args().source)
