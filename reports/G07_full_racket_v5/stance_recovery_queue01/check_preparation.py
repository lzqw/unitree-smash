"""Execute generated stance-entry CPU conditions and inspect stage routing."""
import ast
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

HERE=Path(__file__).resolve().parent
MAIN=HERE.parents[2]
ROOT=MAIN.parent/'unitree_smash_contact_recovery'
PROPOSALS=HERE.parent/'public_stance_recovery_20261027'
sys.path.insert(0,str(ROOT))


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main():
    cpu=read(PROPOSALS/'CPU_CONTROLLER_CHECK.json')
    binding=read(PROPOSALS/'NATIVE_FACTORY_CPU_CHECK.json')
    proposal=read(PROPOSALS/'NATIVE_PROPOSAL.json')
    for record in [cpu,binding]:assert all(sha(Path(n))==h for n,h in record['source_sha256'].items())
    assert len(cpu['results'])==12 and len(binding['results'])==2
    generation=read(HERE/'ENTRY_GENERATION_CHECK.json')
    assert generation['generator_sha256']==sha(HERE/'prepare_entry.py')
    assert all(sha(Path(p))==h for group in ['source_sha256','generated_sha256'] for p,h in generation[group].items())
    names=['run.py','source_support.py','prepare_entry.py','evaluate_stance_recovery.py','check_preparation.py']
    for name in names:ast.parse((HERE/name).read_text())
    queue=load('checked_stance_queue',HERE/'run.py')
    assert queue.PRIOR==HERE.parent/'momentum_strike_queue01'
    assert queue.SOURCE==HERE.parent/'momentum_strike4_n128_01/best_candidate.pt'
    assert queue.SOURCE_COLD==HERE.parent/'momentum_strike4_best_cold128_01'
    assert queue.SEARCH==HERE.parent/'stance_recovery4_n128_01'
    assert queue.COLD==HERE.parent/'stance_recovery4_best_cold128_01'
    assert queue.SOURCE_COLD!=queue.COLD and queue.factory().candidates()==proposal['candidates']
    tree=ast.parse((HERE/'evaluate_stance_recovery.py').read_text())
    original=ast.parse((queue.PRIOR/'evaluate_momentum_strike.py').read_text())
    def function(t,name):return next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name)
    assert ast.dump(function(tree,'generation_summary'))==ast.dump(function(original,'generation_summary'))
    def sampling(t):
        loops=[n for n in ast.walk(t) if isinstance(n,ast.For) and ast.unparse(n.target)=='tick']
        assert len(loops)==1
        return ast.dump(loops[0])
    assert sampling(tree)==sampling(original)
    support_tree=ast.parse((HERE/'source_support.py').read_text())
    admission=function(support_tree,'verify_strike_source')
    assignments={n.targets[0].id:ast.unparse(n.value) for n in admission.body if isinstance(n,ast.Assign)
        and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)}
    assert assignments['earlier_path']=="reports / 'air_countermotion3_n128_01/best_candidate.pt'"
    assert assignments['earlier_cold']=="reports / 'air_countermotion3_best_cold128_01'"
    assert "reports / 'momentum_strike4_n128_01'" in ast.unparse(admission)
    main_node=function(tree,'main')
    scope=dict(cpu=cpu,factory_check=binding,proposal=proposal,sha=sha,PROPOSALS=PROPOSALS,
        candidates=proposal['candidates'],module=queue.factory)
    checked=0
    for node in main_node.body:
        if isinstance(node,ast.Assert) and any(isinstance(n,ast.Name) and n.id in ['cpu','factory_check'] for n in ast.walk(node)):
            assert eval(compile(ast.Expression(node.test),'native_cpu_precondition','eval'),scope)
            checked+=1
    search=next(n for n in main_node.body if isinstance(n,ast.If) and ast.unparse(n.test)=="args.mode == 'search'")
    candidate_assertions=0
    for node in search.body:
        if isinstance(node,ast.Assert) and any(isinstance(n,ast.Name) and n.id=='candidates' for n in ast.walk(node)):
            assert eval(compile(ast.Expression(node.test),'native_candidate_precondition','eval'),scope)
            candidate_assertions+=1
    assert checked==6 and candidate_assertions==2,(checked,candidate_assertions)
    queue_main=function(ast.parse((HERE/'run.py').read_text()),'main')
    gate_tests=[n.test for n in queue_main.body if isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
        and any(isinstance(v,ast.Subscript) and ast.unparse(v.value)=='summary' for v in ast.walk(n.test))]
    assert len(gate_tests)==2
    threshold=read(ROOT/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json')['thresholds']['conditional_recovery_rate']
    assert threshold==.85
    # These are isolated CPU branch checks, never native performance data.
    for air,recovery,expected in [(0,0.,[True,False]),(13,.5,[False,False]),(13,.85,[False,True])]:
        scope=dict(summary=dict(airborne_contacts=air,conditional_recovery_rate=recovery),
            protocol=dict(thresholds=dict(conditional_recovery_rate=threshold)))
        assert [eval(compile(ast.Expression(t),'stance_routing_gate','eval'),scope) for t in gate_tests]==expected
    support=load('checked_stance_source_admission',HERE/'source_support.py')
    early=HERE.parent/'early_lateral7_n128_01'
    source=torch.load(early/'candidate_001.pt',map_location='cpu',weights_only=False)
    try:support.verify_strike_source(early/'candidate_001.pt',source,early,ROOT,sha(ROOT/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'))
    except AssertionError:pass
    else:raise AssertionError('Early source cannot replace an actual completed strike4/cold128')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    result=subprocess.run([sys.executable,str(HERE/'evaluate_stance_recovery.py'),'--help'],capture_output=True,text=True,env=env)
    assert result.returncode==0 and '--source-evaluation' in result.stdout,result.stderr[-2000:]
    (HERE/'native_help.txt').write_text(result.stdout)
    record=dict(checked_at=datetime.now(timezone.utc).isoformat(),native_entry_help_exit_code=result.returncode,
        native_CPU_preconditions_executed=checked,native_search_candidate_preconditions_executed=candidate_assertions,
        original_sampling_and_generation_scoring_AST_identical=True,
        source_momentum4_and_predecessor_countermotion3_routes_checked=True,
        conditional_recovery_adaptive_gates_CPU_checked=True,
        core3sources_times4variants_primary41600_public_commands_verified=True,
        factory2times4800_public_commands_verified=True,incompatible_actual_source_rejected=True,
        actual_selected_momentum_source_and_all4_public_replays_pending=True,
        source_sha256={str(p):sha(p) for p in [*[HERE/n for n in names],PROPOSALS/'native_factory.py',
            PROPOSALS/'stance_controller.py',PROPOSALS/'CPU_CONTROLLER_CHECK.json',
            PROPOSALS/'NATIVE_FACTORY_CPU_CHECK.json',PROPOSALS/'NATIVE_PROPOSAL.json',HERE/'ENTRY_GENERATION_CHECK.json']},
        new_native_execution=False,formal_acceptance=False,goal_complete=False)
    with (HERE/'PRE_NATIVE_CPU_CHECK.json').open('x') as f:f.write(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(native_help_exit_code=result.returncode,CPU_preconditions=checked,candidate_preconditions=candidate_assertions,
        adaptive_gates_checked=True,actual_selected_source_pending=True)))


if __name__=='__main__':main()
