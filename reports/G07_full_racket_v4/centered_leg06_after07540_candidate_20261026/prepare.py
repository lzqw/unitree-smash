"""Declare a seven-source optional pilot before actual075 stage40 results."""
from datetime import datetime,timezone
import copy,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;MAIN=ROOT.parent/'unitree_smash'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    output=HERE/'CONDITIONAL_PLAN.json';assert not output.exists()
    reports=MAIN/'reports/G07_full_racket_v4';owner=reports/'centered_placement075_resume32_queue02'
    future_eval=reports/'residual_ppo_centered_placement075_n2048_iter40_fixed_train128_02'
    assert not (future_eval/'metrics.json').exists(),'Preserve actual declaration timing before075stage40 assessment'
    for pid,needle in [(1734658,'centered_placement075_resume32_queue02/run.py'),(1734680,'residual_ppo_centered_placement075_n2048_resume32_02')]:
        p=Path('/proc')/str(pid)/'cmdline';assert p.exists()and needle.encode()in p.read_bytes()
    held=dict(read(owner/'PREDECLARED_PLAN.json')['source_hashes']);held[str((owner/'PREDECLARED_PLAN.json').relative_to(MAIN))]=sha(owner/'PREDECLARED_PLAN.json')
    for name,digest in held.items():
        p=Path(name)if Path(name).is_absolute()else MAIN/name;assert sha(p)==digest,name
    old_path=HERE.parent/'centered_leg06_candidate_20261026/CONDITIONAL_PLAN.json';old=read(old_path)
    workflow=HERE.parent/'centered_leg06_after07540_pilot8_queue01';check=read(workflow/'ACTUAL_CPU_PIPELINE_CHECK.json')
    assert check['actual_live_owner_blocks_selection'] and check['actual07540_frozen_results_still_absent']
    assert check['actual_strict075_inputs_rejected_as_new_action_unit_stage']
    assert len(check['actual_measured_source_CPU_witnesses'])==3
    assert all(v['transfer']['actual_witness_physical_mean_max_error_rad']==0 for v in check['actual_measured_source_CPU_witnesses'])
    original=read(reports/'centered_intermediate_selection_queue01/RESULT.json')
    known=[dict(candidate_id=f'centered{e["iteration"]}',policy=e['policy'],summary=e['summary'])for e in sorted(original['all_candidates'],key=lambda e:e['iteration'])]
    known.append(dict(candidate_id='placement075_stage8',policy=str(reports/'residual_ppo_centered_placement075_n2048_probe8_01/policy.pt'),summary=read(reports/'centered_placement075_pilot8_queue01/FIXED_TRAIN128_RESULT.json')))
    new=copy.deepcopy(old);new.update(created_at=datetime.now(timezone.utc).isoformat(),status='OPTIONAL_CPU_PREPARED_NOT_SELECTED_NOT_MAIN_INTEGRATED_NOT_NATIVE_RUN',
        prior_declaration_preserved=str(old_path),prior_declaration_sha256=sha(old_path),
        decision_timing='After completed075 stage8 frozen128 and actual strict continuation early update11; before075 stage40 completion and frozen128. Current continuation remains selected and running.',
        evidence_scope='Known075 stage8 is0complete/26air/19conditional recoveries/5attacks. Early strict continuation counts are a censored changing-policy prefix, not frozen performance.',
        CPU_witness40_is_not_selected_future_source=True,main_held_sources_unchanged=len(held))
    select=new['source_selection_after_actual075_completion'];select['candidate_order_for_final_tie']=[v['candidate_id']for v in known]+['placement075_stage40'];select['known_original_candidates']=known
    select['pending_candidate']=dict(candidate_id='placement075_stage40',policy=str(reports/'residual_ppo_centered_placement075_n2048_resume32_02/policy.pt'),evaluation=str(future_eval))
    select['required']='Actual continuation observer1734658 exits READY_FOR_VISUAL_REVIEW, successful canonical40 and selected-trajectory audit; recompute all seven full original128 assessments and matching native receipts. Any source reaching original50/70/85/NF0 skips this revision for formal review. Otherwise use the declared ranking; do not discard a worse current40 result.'
    new['launch_requirements']=['Actual1734658 continuation training/fixed/video observer exits successfully; original main ledger has no RUNNING job. Hold all269 sources plus current plan until then.',
        'Current32-update training must have successful original guard receipt, matching final canonical40/public hashes and passed selected-trajectory audit.',
        'Recompute all seven original128-case zero-noise sources, preserving old five-source attack-first result and old six-source candidate declaration.',
        'Only after selecting this ordinary method, integrate the reviewed helper/trainer/config into original main; run relevant checks there, bind actual selected witness and all data/assets before GPU0 launch.',
        'Use original main GPU0 shared guard/ledger, preserve charges, start no candidate-local native job and no automatic longer training or upload.']
    new['planned_native_pilot']['output']=str(reports/'residual_ppo_centered_leg06_after07540_n2048_probe8_01')
    new['subsequent_assessment']['output']=str(reports/'residual_ppo_centered_leg06_after07540_n2048_probe8_fixed_train128_01')
    new['source_sha256']={str(p):sha(p)for p in [Path(__file__),old_path,workflow/'run.py',workflow/'check.py',workflow/'ACTUAL_CPU_PIPELINE_CHECK.json',owner/'PREDECLARED_PLAN.json',reports/'centered_intermediate_selection_queue01/RESULT.json',reports/'centered_placement075_pilot8_queue01/FIXED_TRAIN128_RESULT.json',ROOT/'configs/g07_full_racket_residual_ppo_centered_leg06_placement075_v4.json',ROOT/'scripts/initialize_centered_action_units_v4.py',ROOT/'scripts/train_full_racket_residual_ppo_v4.py',ROOT/'tests/test_initialize_centered_action_units_v4.py']}
    with output.open('x')as f:json.dump(new,f,indent=2);f.write('\n')
    print(json.dumps(dict(status=new['status'],known_sources=len(known),pending_sources=1,main_sources_unchanged=len(held),new_native_execution=False)))
if __name__=='__main__':main()
