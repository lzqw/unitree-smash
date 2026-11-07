#!/usr/bin/env python3
"""Check original MPC AST preservation and execute the proposed dispatch block.

This validates integration semantics without importing Isaac. It is not an
executed recovery experiment or a controller effectiveness claim.
"""
import argparse
import ast
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.check_full_command_native_entry_v1 import check as check_factory
from quadsmash.control.learned_recovery_command_v1 import NativeRecoveryCommandAdapter


def ast_hash(body):
    return hashlib.sha256(ast.dump(ast.Module(body=body,type_ignores=[]),include_attributes=False).encode()).hexdigest()


def check(source):
    check_factory(source)
    tree=ast.parse(source.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SharedTargetResidualController')
    action=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='action')
    correction=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='apply_landing_mpc')
    assert ast_hash(action.body[:9])=='49acd1e34215f944f46b770fce35743c081a7f1319b4fa54bb1f3e6a4eb04f1d'
    assert ast.unparse(action.body[9].test)=='defer_landing_mpc'
    assert ast.unparse(action.body[10].value)=='self.apply_landing_mpc(observation, default, limits, action)'
    protected=[]
    for node in correction.body:
        if isinstance(node,ast.If) and ast.unparse(node.test)=='eligible_mask is not None':continue
        protected.append(node)
    assert ast_hash(protected)=='ffcf092b1747be8afd2563759970d74fa94fca34dcffcc1d92aaaa000ece2ffa'
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    admission=next(n for n in main.body if isinstance(n,ast.If) and ast.unparse(n.test)=='args.learned_recovery_mpc_correction')
    admitted=dict(learned_recovery_mpc_correction=True,learned_recovery_command_head=Path('head.pt'),
        landing_mpc_cases=list(range(128)),landing_mpc_command_history_interface=True,
        landing_mpc_continuous_reference=True,landing_mpc_torque_overlay=False)
    code=compile(ast.Module(body=[admission],type_ignores=[]),str(source),'exec')
    exec(code,dict(args=SimpleNamespace(**admitted)))
    for field,value in [('learned_recovery_command_head',None),('landing_mpc_cases',[3,8]),
                        ('landing_mpc_command_history_interface',False),('landing_mpc_torque_overlay',True)]:
        try:exec(code,dict(args=SimpleNamespace(**(admitted|{field:value}))))
        except ValueError:pass
        else:raise AssertionError('Missing admission restriction: '+field)
    tickloop=next(n for n in ast.walk(main) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='tick')
    first=next(i for i,n in enumerate(tickloop.body) if isinstance(n,ast.If) and ast.unparse(n.test)=='args.learned_recovery_mpc_correction')
    dispatch=compile(ast.Module(body=tickloop.body[first:first+2],type_ignores=[]),str(source),'exec')
    class Head(torch.nn.Module):
        def forward(self,x):return torch.full((len(x),18),.01)
    limits=torch.tensor([[-2.,2.]]*18);public=torch.zeros((2,794));base=torch.zeros((2,18))
    adapter=NativeRecoveryCommandAdapter(Head(),2,'cpu',limits)
    class Controller:
        def __init__(self):self.requests=0;self.corrections=0
        def action(self,obs,default,limits,*,defer_landing_mpc=False):
            assert defer_landing_mpc;self.requests+=1;return base.clone()
        def apply_landing_mpc(self,obs,default,limits,proposal,*,eligible_mask):
            assert eligible_mask.tolist()==[True,False]
            torch.testing.assert_close(proposal[0],torch.full((18,),.02))
            proposal[0,:12]=.08;proposal[0,12:]=.04;self.corrections+=1;return proposal
    controller=Controller();env=dict(torch=torch,args=SimpleNamespace(**admitted),controller=controller,
        observation={'policy':public},obs=public,default=torch.zeros(18),limits=limits,
        learned_recovery=adapter,finished=torch.zeros(2,dtype=torch.bool),learned_records=[])
    exec(dispatch,env)
    public[:,775]=.01;public[0,773]=1;exec(dispatch,env)
    public[:,775]=.02;public[:,773]=0;public[:,774]=1;exec(dispatch,env)
    assert controller.requests==3 and controller.corrections==1
    record=env['learned_records'][-1]
    torch.testing.assert_close(torch.from_numpy(record['student_command'][0]),torch.full((18,),.02))
    torch.testing.assert_close(torch.from_numpy(record['final_command'][0,:12]),torch.full((12,),.08))
    torch.testing.assert_close(adapter.previous,env['action'])
    assert record['teacher_changed'].tolist()==[True,False]
    # Execute the real filter with a current foot-contact phase but ineligible
    # lanes: neither its contact latch nor a QP may activate before the hit.
    ns=dict(torch=torch);exec(compile(ast.Module(body=[correction],type_ignores=[]),str(source),'exec'),ns)
    subject=SimpleNamespace(landing_mpc=object(),landing_mpc_latched_post_touchdown_ticks=800,
        _landing_mpc_latched_post_touchdown=torch.zeros(2,dtype=torch.bool),
        _landing_mpc_latched_age=torch.zeros(2,dtype=torch.long),
        _landing_mpc_finished_first_episode=torch.zeros(2,dtype=torch.bool))
    output=ns['apply_landing_mpc'](subject,{'policy':public},torch.zeros(18),limits,base.clone(),
        eligible_mask=torch.zeros(2,dtype=torch.bool))
    torch.testing.assert_close(output,base)
    assert not subject._landing_mpc_latched_post_touchdown.any()
    print('Original request/filter AST preserved; actual dispatch order, causal eligibility, final history and explicit assisted admission checked')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'scripts/evaluate_shared_target_ground64_persistent_recovery_v1.py')
    check(p.parse_args().source)
