"""Verify the generated native wiring and execute its actual CPU conditions."""
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
PROPOSALS=HERE.parent/'free_floating_strike_20261027'
sys.path.insert(0,str(ROOT))


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    cpu=read(PROPOSALS/'CPU_CONTROLLER_CHECK.json')
    factory_check=read(PROPOSALS/'NATIVE_FACTORY_CPU_CHECK.json')
    proposal=read(PROPOSALS/'PROPOSAL.json')
    for record in [cpu,factory_check]:
        assert all(sha(Path(p))==h for p,h in record['source_sha256'].items())
    assert len(cpu['results'])==8 and cpu['both_baseline_replays_within_original_tolerance']
    generation=read(HERE/'ENTRY_GENERATION_CHECK.json')
    assert generation['generator_sha256']==sha(HERE/'prepare_entry.py')
    assert all(sha(Path(p))==h for group in ['source_sha256','generated_sha256'] for p,h in generation[group].items())
    names=['run.py','source_support.py','prepare_entry.py','evaluate_momentum_strike.py','check_preparation.py']
    for name in names:ast.parse((HERE/name).read_text())
    queue=load('checked_momentum_queue',HERE/'run.py')
    assert queue.PRIOR==HERE.parent/'air_countermotion_queue01'
    assert queue.SOURCE==HERE.parent/'air_countermotion3_n128_01/best_candidate.pt'
    assert queue.SOURCE_COLD==HERE.parent/'air_countermotion3_best_cold128_01'
    assert queue.SEARCH==HERE.parent/'momentum_strike4_n128_01'
    assert queue.COLD==HERE.parent/'momentum_strike4_best_cold128_01'
    assert queue.SOURCE_COLD!=queue.COLD and queue.factory().candidates()==proposal['candidates']
    tree=ast.parse((HERE/'evaluate_momentum_strike.py').read_text())
    original=ast.parse((queue.PRIOR/'evaluate_air_countermotion.py').read_text())
    def function(t,name):return next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name)
    assert ast.dump(function(tree,'generation_summary'))==ast.dump(function(original,'generation_summary'))
    def sample_loop(t):
        nodes=[n for n in ast.walk(t) if isinstance(n,ast.For) and ast.unparse(n.target)=='tick']
        assert len(nodes)==1
        return ast.dump(nodes[0])
    assert sample_loop(tree)==sample_loop(original)
    main_node=function(tree,'main')
    scope=dict(cpu=cpu,factory_check=factory_check,proposal=proposal,sha=sha,PROPOSALS=PROPOSALS,
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
    support=load('checked_momentum_source_admission',HERE/'source_support.py')
    live=HERE.parent/'early_lateral7_n128_01'
    source=torch.load(live/'candidate_001.pt',map_location='cpu',weights_only=False)
    try:
        support.verify_countermotion_source(live/'candidate_001.pt',source,live,ROOT,
            sha(ROOT/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'))
    except AssertionError:pass
    else:raise AssertionError('Early search cannot replace completed countermotion/cold source')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    result=subprocess.run([sys.executable,str(HERE/'evaluate_momentum_strike.py'),'--help'],
        capture_output=True,text=True,env=env)
    assert result.returncode==0 and '--source-evaluation' in result.stdout,result.stderr[-2000:]
    (HERE/'native_help.txt').write_text(result.stdout)
    record=dict(checked_at=datetime.now(timezone.utc).isoformat(),native_entry_help_exit_code=result.returncode,
        native_CPU_preconditions_executed=checked,native_search_candidate_preconditions_executed=candidate_assertions,
        original_native_sampling_loop_and_generation_scoring_AST_identical=True,
        core_2sources_times4variants_times4800_public_replays_verified=True,
        factory_3times4800_public_replays_verified=True,incompatible_actual_source_rejected=True,
        actual_selected_countermotion_source_all4_replays_pending=True,
        source_cold_and_future_output_directories_distinct=True,
        source_sha256={str(p):sha(p) for p in [*[HERE/n for n in names],PROPOSALS/'native_factory.py',
            PROPOSALS/'strike_controller.py',PROPOSALS/'free_floating.py',PROPOSALS/'CPU_CONTROLLER_CHECK.json',
            PROPOSALS/'NATIVE_FACTORY_CPU_CHECK.json',PROPOSALS/'PROPOSAL.json',HERE/'ENTRY_GENERATION_CHECK.json']},
        new_native_execution=False,formal_acceptance=False)
    with (HERE/'PRE_NATIVE_CPU_CHECK.json').open('x') as stream:stream.write(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(native_help_exit_code=result.returncode,CPU_preconditions_executed=checked,
        candidate_preconditions_executed=candidate_assertions,actual_selected_source_pending=True)))


if __name__=='__main__':main()
