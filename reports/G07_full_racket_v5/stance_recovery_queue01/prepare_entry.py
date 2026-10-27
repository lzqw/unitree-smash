"""Derive bounded stance trials from the held original128 native episode loop."""
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'momentum_strike_queue01'
PROPOSALS=HERE.parent/'public_stance_recovery_20261027'
MODEL=HERE.parent/'centroidal_recovery_model_20261027'


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def once(text,old,new):
    assert text.count(old)==1,(old,text.count(old))
    return text.replace(old,new)


def assignment(text,name,value):
    nodes=[n for n in ast.parse(text).body if isinstance(n,ast.Assign) and len(n.targets)==1
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id==name]
    assert len(nodes)==1 and nodes[0].lineno==nodes[0].end_lineno
    lines=text.splitlines(True);lines[nodes[0].lineno-1]=name+' = '+value+'\n'
    return ''.join(lines)


def source_support():
    text=(PRIOR/'source_support.py').read_text().split('\ndef replay_selected_public(',1)[0]
    text=text.replace('verify_countermotion_source','verify_strike_source')
    text=text.replace('Admit completed countermotion3/cold128 before any strike GPU job.',
        'Admit completed momentum-strike4/cold128 before any stance GPU job.')
    text=once(text,"reports/'public_air_countermotion_20261027/native_factory.py'", "reports/'free_floating_strike_20261027/native_factory.py'")
    text=once(text,"source['schema']=='g07-public-air-countermotion-v5'", "source['schema']=='g07-public-momentum-strike-v5'")
    text=once(text,"core_path=reports/'public_air_countermotion_20261027/countermotion.py'", "core_path=reports/'free_floating_strike_20261027/strike_controller.py'")
    text=once(text,"assert source['counter_core_sha256']==sha(core_path)",
        "assert source['strike_core_sha256']==sha(core_path)\n    free_core_path=reports/'free_floating_strike_20261027/free_floating.py'\n    assert source['free_floating_core_sha256']==sha(free_core_path)")
    text=text.replace("reports/'recovery_prior9_n128_01/best_candidate.pt'", "reports/'air_countermotion3_n128_01/best_candidate.pt'")
    text=text.replace("reports/'recovery_prior9_best_cold128_01'", "reports/'air_countermotion3_best_cold128_01'")
    text=once(text,"reports/'air_countermotion_queue01/source_support.py'", "reports/'momentum_strike_queue01/source_support.py'")
    text=once(text,'support.verify_prior_source(', 'support.verify_countermotion_source(')
    text=text.replace("reports/'air_countermotion3_n128_01'", "reports/'momentum_strike4_n128_01'")
    text=text.replace('<=307200','<=409600').replace('list(range(3))','list(range(4))')
    text=once(text,'files=[path,factory_path,model_path,core_path,support_path,protocol_path,',
        'files=[path,factory_path,model_path,core_path,free_core_path,support_path,protocol_path,')
    text=text.replace('selected_countermotion_candidate_id','selected_strike_candidate_id')
    text=text.replace('previous_recovery_source_check','previous_countermotion_source_check')
    text=text.replace('all3_actual_countermotion_search_summaries_independently_recomputed',
        'all4_actual_strike_search_summaries_independently_recomputed')
    text+='''
def replay_selected_public(path,source,evaluation,root,factory):
    from scripts.initialize_centered_policy_v4 import same_tree
    check=verify_strike_source(path,source,evaluation,root,
        sha(Path(root)/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'))
    saved=copy.deepcopy(source)
    inputs=read(Path(evaluation)/'INPUTS.json')
    defaults,limits=[torch.tensor(inputs[k],dtype=torch.float32) for k in ['default_joint_position_rad','joint_limits_rad']]
    selected=check['selected_strike_candidate_id']
    with np.load(Path(evaluation)/f'public_and_actions_{selected:03d}.npz') as data:
        public,actions,valid=[data[k].copy() for k in ['observations','actions','valid_first_episode']]
        assert data['training_indices'].tolist()==[0,29,69,100,112,123]
    assert public.shape[1:]==(6,794) and 0<len(public)<=800 and valid.any()
    results=[]
    for candidate in factory.candidates():
        kwargs={k:candidate[k] for k in ['gain','horizontal_damping']}
        controller=factory.make_controller(source,6,'cpu',**kwargs)
        reference=factory.make_source(source,6,'cpu')
        previous=previous_target=None
        maximum=max_leg_delta=max_arm_rate=0.;active_rows=0
        digest=hashlib.sha256()
        for value,native,keep in zip(public,actions,valid):
            obs=torch.from_numpy(value);unchanged=obs.clone();reset=torch.zeros(6,dtype=torch.bool)
            if previous is not None:
                reset=obs[:,775]<previous-1e-5
                ids=torch.where(reset)[0]
                if len(ids):controller.reset(ids);reference.reset(ids)
            expected=reference.action({'policy':obs},defaults,limits)
            actual=controller.action({'policy':obs},defaults,limits)
            assert torch.equal(obs,unchanged)
            assert actual.shape==(6,18) and torch.isfinite(actual).all() and (abs(actual)<=1).all()
            assert torch.equal(actual[:,12:],expected[:,12:])
            if candidate['candidate_id']==0:
                assert torch.equal(actual,expected)
                if keep.any():maximum=max(maximum,float(abs(actual.numpy()[keep]-native[keep]).max()))
            else:
                diagnostic=controller.last_diagnostic
                active=diagnostic['active'];seen=diagnostic['hit_seen'].numpy()
                contacts=value[:,57:61]>.5
                height=1.80-np.sum((-value[:,6:9])*value[:,760:763],axis=1)
                assert not np.any(active & (~seen | (~contacts.any(1)) | (height<.14) | (-value[:,8]<=.25)))
                assert torch.equal(actual[~active],expected[~active])
                active_rows+=int(active.sum())
                for foot,ids in enumerate(controller.requester.legs):
                    mask=~contacts[:,foot]
                    assert torch.equal(actual[mask][:,ids],expected[mask][:,ids])
            target=limits[:,0]+.5*(actual+1)*(limits[:,1]-limits[:,0])
            base_target=limits[:,0]+.5*(expected+1)*(limits[:,1]-limits[:,0])
            max_leg_delta=max(max_leg_delta,float(abs(target[:,:12]-base_target[:,:12]).max()))
            if previous_target is not None and (~reset).any():
                max_arm_rate=max(max_arm_rate,float(abs(target[~reset,12:]-previous_target[~reset,12:]).max())/.01)
            digest.update(actual.numpy().tobytes())
            previous=obs[:,775].clone();previous_target=target.clone()
        assert maximum<=5e-4 and max_leg_delta<=.15+1e-5 and max_arm_rate<=5.0005
        assert same_tree(source,saved)
        results.append(dict(candidate_id=candidate['candidate_id'],actual_public_rows=int(np.prod(public.shape[:2])),
            actual_valid_public_rows=int(valid.sum()),active_post_hit_stance_rows=active_rows,
            CPU_vs_selected_source_GPU_max_action_error=maximum if candidate['candidate_id']==0 else None,
            maximum_leg_target_delta_rad=max_leg_delta,maximum_arm_target_rate_rad_s=max_arm_rate,
            arm_pre_hit_and_nonstance_commands_preserved=True,command_stream_sha256=digest.hexdigest(),
            source_weights_and_public_inputs_unchanged=True))
    return dict(source_check=check,actual_selected_source_all4_public_replays=results,
        active_branch_coverage_is_reported_not_assumed=True,new_native_execution=False,formal_acceptance=False)
'''
    ast.parse(text)
    return text


def main():
    cpu=read(PROPOSALS/'CPU_CONTROLLER_CHECK.json');binding=read(PROPOSALS/'NATIVE_FACTORY_CPU_CHECK.json')
    assert len(cpu['results'])==12 and len(binding['results'])==2
    assert all(sha(Path(p))==h for record in [cpu,binding] for p,h in record['source_sha256'].items())
    candidates=[x['candidate'] for x in cpu['results'][:4]]
    assert candidates==[x['candidate'] for x in cpu['results'][4:8]]==[x['candidate'] for x in cpu['results'][8:]]
    proposal=dict(prepared_at=datetime.now(timezone.utc).isoformat(),candidates=candidates,
        original_CPU_proposal_sha256=sha(PROPOSALS/'PROPOSAL.json'),factory_sha256=sha(PROPOSALS/'native_factory.py'),
        core_sha256=sha(PROPOSALS/'stance_controller.py'),CPU_check_sha256=sha(PROPOSALS/'CPU_CONTROLLER_CHECK.json'),
        model_sha256=sha(MODEL/'MODEL.json'),algorithm='Public post-hit observed stance-leg bounded root orientation/velocity correction; original arm/pre-hit/nonstance commands and physical limits preserved.',
        actual_source_pending_momentum4_and_independent_cold=True,conditional_recovery_deficit_required=True,
        physics_rules_cases_target_and_thresholds_unchanged=True,native_execution=False,formal_acceptance=False)
    text=(PRIOR/'evaluate_momentum_strike.py').read_text()
    text=text.replace('verify_countermotion_source','verify_strike_source')
    text=text.replace('free_floating_strike_20261027','public_stance_recovery_20261027')
    text=text.replace("SCHEMA = 'g07-public-momentum-strike-v5'","SCHEMA = 'g07-public-stance-recovery-v5'")
    text=text.replace('PROPOSAL.json','NATIVE_PROPOSAL.json')
    start=text.index("    assert factory_check['factory_sha256']");end=text.index("    if args.mode == 'search':",start)
    text=text[:start]+'''    assert factory_check['factory_sha256'] == sha(PROPOSALS / 'native_factory.py')
    assert len(cpu['results']) == 12 and len(proposal['candidates']) == 4
    assert all(r['maximum_source_normalized_action_change'] == 0. and r['maximum_original_native_action_difference'] <= 5e-4 for r in cpu['results'] if r['candidate']['candidate_id'] == 0)
    assert all(r['original_actors_normalizers_and_public_inputs_unchanged'] for r in cpu['results'])
    assert all(r['arm_and_pre_hit_and_nonstance_leg_commands_bitwise_unchanged'] for r in cpu['results'])
    assert all(r['checked_core_command_stream_bitwise_identical'] and (r['actual_baseline_GPU_max_error'] is None or r['actual_baseline_GPU_max_error'] <= 5e-4) for r in factory_check['results'])
'''+text[end:]
    text=once(text,"        assert candidates == [r['candidate'] for r in cpu['results'][:4]] == [r['candidate'] for r in cpu['results'][4:]]",
        "        assert candidates == [r['candidate'] for r in cpu['results'][:4]] == [r['candidate'] for r in cpu['results'][4:8]] == [r['candidate'] for r in cpu['results'][8:]]")
    text=text.replace("name='independent_cold_selected_momentum_strike',\n                           **checkpoint['momentum_strike']", "name='independent_cold_selected_stance_recovery',\n                           **checkpoint['stance_recovery']")
    text=once(text,"assert checkpoint['strike_core_sha256'] == sha(PROPOSALS / 'strike_controller.py')\n            assert checkpoint['free_floating_core_sha256'] == sha(PROPOSALS / 'free_floating.py')",
        "assert checkpoint['stance_core_sha256'] == sha(PROPOSALS / 'stance_controller.py')")
    text=text.replace("['gain', 'leg_velocity_delta_cap_rad_s']","['gain', 'horizontal_damping']")
    text=text.replace('momentum_strike={key: candidate[key]', 'stance_recovery={key: candidate[key]')
    text=once(text,"strike_core_sha256=sha(PROPOSALS / 'strike_controller.py'),\n                    free_floating_core_sha256=sha(PROPOSALS / 'free_floating.py'),",
        "stance_core_sha256=sha(PROPOSALS / 'stance_controller.py'),")
    start=text.index('    extra = [');end=text.index('    extra_hashes = ',start)
    text=text[:start]+'''    extra = [Path(__file__), HERE / 'run.py', HERE / 'prepare_entry.py', HERE / 'check_preparation.py',
             HERE / 'PLAN.json', HERE / 'PRE_NATIVE_CPU_CHECK.json', HERE / 'source_support.py',
             HERE / 'SELECTED_SOURCE_CPU_CHECK.json', PROPOSALS / 'native_factory.py',
             PROPOSALS / 'stance_controller.py', PROPOSALS / 'CPU_CONTROLLER_CHECK.json',
             PROPOSALS / 'NATIVE_FACTORY_CPU_CHECK.json', PROPOSALS / 'NATIVE_PROPOSAL.json',
             HERE.parent / 'free_floating_strike_20261027/native_factory.py',
             HERE.parent / 'free_floating_strike_20261027/strike_controller.py',
             HERE.parent / 'free_floating_strike_20261027/free_floating.py',
             HERE.parent / 'free_floating_strike_20261027/ACTUAL_CHECK.json',
             HERE.parent / 'free_floating_strike_20261027/CPU_CONTROLLER_CHECK.json',
             HERE.parent / 'free_floating_strike_20261027/PROPOSAL.json',
             HERE.parent / 'public_air_countermotion_20261027/native_factory.py',
             HERE.parent / 'public_air_countermotion_20261027/countermotion.py',
             HERE.parent / 'centroidal_recovery_model_20261027/known_model.py',
             HERE.parent / 'centroidal_recovery_model_20261027/MODEL.json',
             HERE.parent / 'centroidal_recovery_model_20261027/ACTUAL_MODEL_CHECK.json',
             HERE.parent / 'recovery_prior_native_20261027/prior_controller.py',
             HERE.parent / 'recovery_motion_prior_20261027/SEED_RECEIPT.json',
             HERE.parent / 'recovery_motion_prior_20261027/recorded_recovery_prior.npz',
             HERE.parent / 'windowed_swing_20261027/window_controller.py',
             HERE.parent / 'early_tangent_intercept_20261027/early_lateral_controller.py',
             HERE.parent / 'origin0_measured_swing_velocity_20261027/swing_controller.py']
'''+text[end:]
    text=text.replace('Execute public pre-hit conserved-momentum strike proposals','Execute public post-hit stance-leg recovery proposals')
    text=text.replace('native_momentum_strike_controller','native_stance_recovery_controller')
    queue=(PRIOR/'run.py').read_text()
    for name,value in dict(PRIOR="OUTPUTS / 'momentum_strike_queue01'",SOURCE="OUTPUTS / 'momentum_strike4_n128_01/best_candidate.pt'",
        SOURCE_COLD="OUTPUTS / 'momentum_strike4_best_cold128_01'",PROPOSALS="OUTPUTS / 'public_stance_recovery_20261027'",
        SEARCH="OUTPUTS / 'stance_recovery4_n128_01'",COLD="OUTPUTS / 'stance_recovery4_best_cold128_01'").items():queue=assignment(queue,name,value)
    for old,new in [('evaluate_momentum_strike.py','evaluate_stance_recovery.py'),('1973878','1984009'),
        ('verify_countermotion_source','verify_strike_source'),('PROPOSAL.json','NATIVE_PROPOSAL.json'),
        ('WAITING_FOR_CONFIRMED_COUNTERMOTION3_OWNER_AND_COLD128','WAITING_FOR_CONFIRMED_MOMENTUM4_OWNER_AND_COLD128'),
        ('COMPLETED_AIR_COUNTERMOTION3_AND_INDEPENDENT_COLD128','COMPLETED_MOMENTUM_STRIKE4_AND_INDEPENDENT_COLD128'),
        ('RUNNING_MOMENTUM_STRIKE_','RUNNING_STANCE_RECOVERY_'),
        ('CHECKING_SELECTED_SOURCE_CPU_BEFORE_MOMENTUM_STRIKE_NATIVE','CHECKING_SELECTED_SOURCE_CPU_BEFORE_STANCE_RECOVERY_NATIVE'),
        ("state('COMPLETED_MOMENTUM_STRIKE4_AND_INDEPENDENT_COLD128', final_summary", "state('COMPLETED_STANCE_RECOVERY4_AND_INDEPENDENT_COLD128', final_summary")]:
        assert old in queue,old;queue=queue.replace(old,new)
    start=queue.index('    files = [');end=queue.index('    for path in files:',start)
    queue=queue[:start]+'''    files = [Path(__file__), HERE / 'evaluate_stance_recovery.py', HERE / 'prepare_entry.py',
        HERE / 'ENTRY_GENERATION_CHECK.json', HERE / 'source_support.py', HERE / 'PRE_NATIVE_CPU_CHECK.json',
        HERE / 'check_preparation.py', PRIOR / 'PLAN.json', PROPOSALS / 'native_factory.py',
        PROPOSALS / 'stance_controller.py', PROPOSALS / 'CPU_CONTROLLER_CHECK.json',
        PROPOSALS / 'check_controller.py', PROPOSALS / 'PROPOSAL.json', PROPOSALS / 'NATIVE_PROPOSAL.json',
        PROPOSALS / 'check_native_factory.py', PROPOSALS / 'NATIVE_FACTORY_CPU_CHECK.json']
'''+queue[end:]
    queue=once(queue,"    assert len(read(PROPOSALS / 'CPU_CONTROLLER_CHECK.json')['results']) == 8\n    assert read(PROPOSALS / 'CPU_CONTROLLER_CHECK.json')['both_baseline_replays_within_original_tolerance']",
        "    assert len(read(PROPOSALS / 'CPU_CONTROLLER_CHECK.json')['results']) == 12")
    queue=queue.replace('actual_source_pending_countermotion3_and_independent_cold','actual_source_pending_momentum4_and_independent_cold')
    queue=queue.replace('all3 countermotion original128 trials','all4 momentum-strike original128 trials')
    queue=queue.replace('Public pre-hit air-only conserved-momentum strike correction; original complete-tool guard,18joint limits and.15rad mixer.',
        'Public post-hit observed stance-leg correction; original source arms, pre-hit path, nonstance targets and physical limits preserved.')
    queue=queue.replace('After countermotion3/cold128 exits, execute4 public conserved-momentum strike proposals.',
        'After momentum-strike4/cold128 exits, conditionally execute4 public stance recovery proposals.')
    start=queue.index("    if read(PRIOR / 'state.json')['state'] == 'SKIPPED_SOURCE_TRAIN_GATES_MET_FORMAL_STILL_REQUIRED':")
    end=queue.index('    source_job = ',start)
    queue=queue[:start]+'''    parent_state = read(PRIOR / 'state.json')['state']
    if parent_state.startswith('SKIPPED_'):
        state('SKIPPED_PARENT_SOURCE_STAGE', parent_state=parent_state, formal_still_required=True)
        return
    assert parent_state == 'COMPLETED_MOMENTUM_STRIKE4_AND_INDEPENDENT_COLD128'
'''+queue[end:]
    needle="    state('CHECKING_SELECTED_SOURCE_CPU_BEFORE_STANCE_RECOVERY_NATIVE', source_policy_sha256=sha(SOURCE))"
    queue=once(queue,needle,"""    if summary['airborne_contacts'] == 0:
        state('SKIPPED_NO_ACTUAL_AIR_CONTACTS_CONTACT_ADAPTATION_REQUIRED', summary=summary)
        return
    if summary['conditional_recovery_rate'] >= protocol['thresholds']['conditional_recovery_rate']:
        state('SKIPPED_RECOVERY_GATE_MET_FOCUS_CONTACT_OR_ATTACK', summary=summary)
        return
"""+needle)
    support=source_support()
    for content in [text,queue,support]:ast.parse(content)
    with (PROPOSALS/'NATIVE_PROPOSAL.json').open('x') as f:f.write(json.dumps(proposal,indent=2)+'\n')
    generated={'evaluate_stance_recovery.py':text,'run.py':queue,'source_support.py':support}
    for name,content in generated.items():
        with (HERE/name).open('x') as f:f.write(content)
    record=dict(prepared_at=datetime.now(timezone.utc).isoformat(),generator_sha256=sha(Path(__file__)),
        source_sha256={str(PRIOR/n):sha(PRIOR/n) for n in ['evaluate_momentum_strike.py','run.py','source_support.py']},
        generated_sha256={str(HERE/n):sha(HERE/n) for n in generated},
        original_native_sampling_scoring_and128_cases_preserved=True,
        no_new_native_execution=True,formal_acceptance=False)
    with (HERE/'ENTRY_GENERATION_CHECK.json').open('x') as f:f.write(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(generated=list(generated),new_native_execution=False)))


if __name__=='__main__':main()
