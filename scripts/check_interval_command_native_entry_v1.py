#!/usr/bin/env python3
"""Check the real evaluator factory/flag plumbing without starting Isaac."""
import argparse
import ast
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.landing_command_history_mpc_v1 import LandingCommandHistoryMPCV1
from scripts.landing_interval_command_mpc_v1 import LandingIntervalCommandMPCV1
from scripts.landing_support_mpc_v1 import LandingSupportMPCV1,LandingMPCSettingsV1
from scripts.landing_support_mpc_v2 import LandingSupportMPCV2,LandingMPCSettingsV2
from scripts.landing_support_mpc_v3 import LandingSupportMPCV3
from scripts.whole_body_support_qp_v1 import PositionPDInterface,SupportSettings
from scripts.interval_command_support_qp_v1 import IntervalCommandSupportQP


def check(source):
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
    assert 'landing_mpc_interval_command_qp' in values,'Explicit parser option missing'
    values.update(landing_mpc_cases={1},landing_mpc_command_history_interface=True,
        landing_mpc_continuous_reference=True,landing_mpc_support_margin=True,
        landing_mpc_executor_replay=True,landing_mpc_arm_target_rate=10.,_landing_mpc_horizon_steps=2)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SharedTargetResidualController')
    init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    assert 'landing_mpc_interval_command_qp' in [a.arg for a in init.args.args+init.args.kwonlyargs]
    assert sum(k.arg=='landing_mpc_interval_command_qp' for n in ast.walk(tree)
               if isinstance(n,ast.Call) for k in n.keywords)==2,'Input manifest and constructor must both record flag'
    factory=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='configure_landing_mpc')
    namespace=globals().copy()
    exec(compile(ast.Module(body=[factory],type_ignores=[]),str(source),'exec'),namespace)
    for enabled in (False,True):
        values['landing_mpc_interval_command_qp']=enabled;subject=SimpleNamespace(**values)
        namespace['configure_landing_mpc'](subject,np.tile([-4.,4.],(18,1)),np.zeros(18))
        assert type(subject.landing_mpc) is (LandingIntervalCommandMPCV1 if enabled else LandingCommandHistoryMPCV1)
        assert isinstance(subject.landing_mpc.qp,IntervalCommandSupportQP)==enabled
        assert subject.landing_mpc.qp.interface.arm_rate==10.
        np.testing.assert_array_equal(subject.landing_mpc.qp.interface.velocity[12:],np.full(6,5.))
    print('Explicit interval-command flag, baseline branch and physical/interface limits checked; no native execution')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'scripts/evaluate_shared_target_ground64_persistent_recovery_v1.py')
    check(p.parse_args().source)
