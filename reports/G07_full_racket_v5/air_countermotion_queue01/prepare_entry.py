"""Derive three countermotion trials from the unchanged native episode loop."""
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'recovery_prior_queue01'
PROPOSALS=HERE.parent/'public_air_countermotion_20261027'
MODEL=HERE.parent/'centroidal_recovery_model_20261027'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def once(text,old,new):
    assert text.count(old)==1,(old,text.count(old))
    return text.replace(old,new)


def assignment(text,name,value):
    nodes=[n for n in ast.parse(text).body if isinstance(n,ast.Assign) and len(n.targets)==1
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id==name]
    assert len(nodes)==1 and nodes[0].lineno==nodes[0].end_lineno
    lines=text.splitlines(True);lines[nodes[0].lineno-1]=name+' = '+value+'\n'
    return ''.join(lines)


def main():
    cpu=json.loads((PROPOSALS/'CPU_CONTROLLER_CHECK.json').read_text())
    factory_check=json.loads((PROPOSALS/'NATIVE_FACTORY_CPU_CHECK.json').read_text())
    assert len(cpu['results'])==3 and factory_check['actual_baseline_CPU_vs_GPU_max_error']<=5e-4
    assert all(sha(Path(p))==h for x in [cpu,factory_check] for p,h in x['source_sha256'].items())
    candidates=[dict(candidate_id=i,name=name,maximum_velocity_change_rad_s=cap) for i,(name,cap) in
        enumerate([('unchanged_source',0.),('post_hit_legs_cap2',2.),('post_hit_legs_cap5',5.)])]
    proposal=dict(prepared_at=datetime.now(timezone.utc).isoformat(),candidates=candidates,
        factory_sha256=sha(PROPOSALS/'native_factory.py'),core_sha256=sha(PROPOSALS/'countermotion.py'),
        CPU_check_sha256=sha(PROPOSALS/'CPU_CONTROLLER_CHECK.json'),model_sha256=sha(MODEL/'MODEL.json'),
        algorithm='Bounded public post-hit air-only leg momentum redistribution; require feet off, root height>=.30m and body-up-z>.25. Original.15rad residual mixer,30rad/s leg hardware and5rad/s unchanged arm commands.',
        source_policy_selection_pending_actual_recovery9_and_cold128=True,
        physics_rules_cases_target_and_thresholds_unchanged=True,native_execution=False,formal_acceptance=False)
    with (PROPOSALS/'PROPOSAL.json').open('x') as stream:stream.write(json.dumps(proposal,indent=2)+'\n')
    text=(PRIOR/'evaluate_recovery_prior.py').read_text()
    text=text.replace('verify_window_source','verify_prior_source')
    text=text.replace('recovery_prior_native_20261027','public_air_countermotion_20261027')
    text=text.replace('prior_controller.py','native_factory.py')
    text=text.replace("SCHEMA = 'g07-shared-native-recovery-prior-v5'","SCHEMA = 'g07-public-air-countermotion-v5'")
    start=text.index("    assert cpu['factory_sha256']")
    end=text.index("    if args.mode == 'search':",start)
    text=text[:start]+'''    factory_check = read(PROPOSALS / 'NATIVE_FACTORY_CPU_CHECK.json')
    assert factory_check['factory_sha256'] == sha(PROPOSALS / 'native_factory.py')
    assert len(cpu['results']) == len(proposal['candidates']) == 3
    assert cpu['results'][0]['maximum_source_action_difference'] == 0.
    assert cpu['results'][0]['maximum_original_GPU_action_difference'] <= 5e-4
    assert factory_check['actual_baseline_CPU_vs_GPU_max_error'] <= 5e-4
    assert all(row['public_inputs_and_source_weights_unchanged'] for row in cpu['results'])
'''+text[end:]
    text=once(text,"        assert candidates == cpu['candidates'] == module().candidates()",
        "        assert candidates == module().candidates()\n        assert [c['maximum_velocity_change_rad_s'] for c in candidates] == [r['maximum_velocity_change_rad_s'] for r in cpu['results']]")
    text=text.replace("len(search_metrics['history']) == 9","len(search_metrics['history']) == 3")
    text=text.replace('<= 921600','<= 307200')
    text=text.replace("name='independent_cold_selected_recovery_prior',\n                           **checkpoint['recovery_prior']", "name='independent_cold_selected_air_countermotion',\n                           **checkpoint['air_countermotion']")
    start=text.index('        seed_metadata, seed_arrays = factory.load_seed()')
    end=text.index('        for candidate in candidates:',start)
    text=text[:start]+'''        model_record = read(HERE.parent / 'centroidal_recovery_model_20261027/MODEL.json')
        if args.mode == 'cold':
            assert checkpoint['known_model_sha256'] == sha(HERE.parent / 'centroidal_recovery_model_20261027/MODEL.json')
            assert checkpoint['known_model'] == model_record
            assert checkpoint['counter_core_sha256'] == sha(PROPOSALS / 'countermotion.py')
'''+text[end:]
    text=once(text,"controller = factory.RecoveryPriorController(base, seed_metadata, seed_arrays, 128, env.device,\n                **{key: candidate[key] for key in ['group', 'table', 'alignment', 'gain']})",
        "controller = factory.make_controller(base, 128, env.device, candidate['maximum_velocity_change_rad_s'])")
    start=text.index("                    recovery_prior={key:")
    end=text.index(' protocol_sha256=sha(protocol_path),',start)
    text=text[:start]+'''                    air_countermotion=dict(maximum_velocity_change_rad_s=candidate['maximum_velocity_change_rad_s']),
                    known_model=model_record,
                    known_model_sha256=sha(HERE.parent / 'centroidal_recovery_model_20261027/MODEL.json'),
                    counter_core_sha256=sha(PROPOSALS / 'countermotion.py'),'''+text[end:]
    text=once(text,"HERE.parent / 'recovery_motion_prior_20261027/recorded_recovery_prior.npz']",
        """HERE.parent / 'recovery_motion_prior_20261027/recorded_recovery_prior.npz',
             HERE.parent / 'recovery_prior_native_20261027/prior_controller.py',
             PROPOSALS / 'countermotion.py', PROPOSALS / 'NATIVE_FACTORY_CPU_CHECK.json',
             HERE.parent / 'centroidal_recovery_model_20261027/known_model.py',
             HERE.parent / 'centroidal_recovery_model_20261027/MODEL.json',
             HERE.parent / 'centroidal_recovery_model_20261027/ACTUAL_MODEL_CHECK.json']""")
    text=text.replace('Execute shared recovery-prior proposals','Execute public air-only countermotion proposals')
    text=text.replace('uses nine sequential','uses three sequential')
    text=text.replace('native_recovery_prior_controller','native_air_countermotion_controller')
    ast.parse(text)
    with (HERE/'evaluate_air_countermotion.py').open('x') as stream:stream.write(text)
    text=(PRIOR/'run.py').read_text()
    values=dict(PRIOR="OUTPUTS / 'recovery_prior_queue01'",SOURCE="OUTPUTS / 'recovery_prior9_n128_01/best_candidate.pt'",
        SOURCE_COLD="OUTPUTS / 'recovery_prior9_best_cold128_01'",PROPOSALS="OUTPUTS / 'public_air_countermotion_20261027'",
        SEARCH="OUTPUTS / 'air_countermotion3_n128_01'",COLD="OUTPUTS / 'air_countermotion3_best_cold128_01'")
    for name,value in values.items():text=assignment(text,name,value)
    changes=[('evaluate_recovery_prior.py','evaluate_air_countermotion.py'),('prior_controller.py','native_factory.py'),
        ('1966524','1968858'),('verify_window_source','verify_prior_source'),
        ('WAITING_FOR_CONFIRMED_WINDOW7_OWNER_AND_COLD128','WAITING_FOR_CONFIRMED_RECOVERY9_OWNER_AND_COLD128'),
        ("assert read(PRIOR / 'state.json')['state'] == 'COMPLETED_WINDOWED_SWING7_AND_INDEPENDENT_COLD128'",
         "assert read(PRIOR / 'state.json')['state'] == 'COMPLETED_RECOVERY_PRIOR9_AND_INDEPENDENT_COLD128'"),
        ('RUNNING_RECOVERY_PRIOR_','RUNNING_AIR_COUNTERMOTION_'),
        ('CHECKING_SELECTED_SOURCE_CPU_BEFORE_RECOVERY_PRIOR_NATIVE','CHECKING_SELECTED_SOURCE_CPU_BEFORE_AIR_COUNTERMOTION_NATIVE'),
        ("state('COMPLETED_RECOVERY_PRIOR9_AND_INDEPENDENT_COLD128', final_summary", "state('COMPLETED_AIR_COUNTERMOTION3_AND_INDEPENDENT_COLD128', final_summary"),
        ('list(range(9))','list(range(3))'),("['results']) == 9","['results']) == 3"),
        ('env_steps_cap=921600, max_seconds=21600','env_steps_cap=307200, max_seconds=9000'),
        ("job(SEARCH, 'search', SOURCE, 921600, 21600, expected)","job(SEARCH, 'search', SOURCE, 307200, 9000, expected)")]
    for old,new in changes:
        assert old in text,old
        text=text.replace(old,new)
    start=text.index('    files = [');end=text.index('    for path in files:',start)
    text=text[:start]+'''    expected.update(read(PROPOSALS / 'NATIVE_FACTORY_CPU_CHECK.json')['source_sha256'])
    files = [Path(__file__), HERE / 'evaluate_air_countermotion.py', HERE / 'prepare_entry.py',
        HERE / 'ENTRY_GENERATION_CHECK.json', HERE / 'source_support.py', HERE / 'PRE_NATIVE_CPU_CHECK.json',
        HERE / 'check_preparation.py', PRIOR / 'PLAN.json', PROPOSALS / 'native_factory.py',
        PROPOSALS / 'countermotion.py', PROPOSALS / 'check_controller.py', PROPOSALS / 'CPU_CONTROLLER_CHECK.json',
        PROPOSALS / 'PROPOSAL.json', PROPOSALS / 'check_native_factory.py', PROPOSALS / 'NATIVE_FACTORY_CPU_CHECK.json']
'''+text[end:]
    text=text.replace('actual_source_pending_window7_and_independent_cold','actual_source_pending_recovery9_and_independent_cold')
    text=text.replace('all7 windowed-swing original128 trials','all9 recovery-prior original128 trials')
    text=text.replace('Shared bounded actual recovery residual traces with group and causal alignment ablations; original.15rad residual mixer and5rad/s arm rate limit.',
        'Public post-hit air-only bounded leg countermotion; validated known28-body model; original.15rad mixer and hardware limits, arms unchanged.')
    text=text.replace('After windowed-swing7/cold128 exits, execute9 shared recovery-prior proposals.',
        'After recovery-prior9/cold128 exits, execute3 air-only countermotion proposals.')
    ast.parse(text)
    with (HERE/'run.py').open('x') as stream:stream.write(text)
    record=dict(prepared_at=datetime.now(timezone.utc).isoformat(),generator_sha256=sha(Path(__file__)),
        source_sha256={str(PRIOR/n):sha(PRIOR/n) for n in ['evaluate_recovery_prior.py','run.py']},
        generated_sha256={str(HERE/n):sha(HERE/n) for n in ['evaluate_air_countermotion.py','run.py']},
        original_native_sampling_scoring_and_128_cases_preserved=True,new_native_execution=False,formal_acceptance=False)
    with (HERE/'ENTRY_GENERATION_CHECK.json').open('x') as stream:stream.write(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__=='__main__':main()
