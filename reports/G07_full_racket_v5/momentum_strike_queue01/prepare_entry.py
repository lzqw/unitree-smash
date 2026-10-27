"""Derive4 conserved-momentum strike trials from the held native episode loop."""
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'air_countermotion_queue01'
PROPOSALS=HERE.parent/'free_floating_strike_20261027'
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


def main():
    cpu=read(PROPOSALS/'CPU_CONTROLLER_CHECK.json')
    factory_check=read(PROPOSALS/'NATIVE_FACTORY_CPU_CHECK.json')
    assert len(cpu['results'])==8 and cpu['both_baseline_replays_within_original_tolerance']
    assert factory_check['actual_baseline_CPU_vs_GPU_max_error']<=5e-4
    assert all(sha(Path(p))==h for record in [cpu,factory_check] for p,h in record['source_sha256'].items())
    candidates=[r['candidate'] for r in cpu['results'][:4]]
    assert candidates==[r['candidate'] for r in cpu['results'][4:]]
    proposal=dict(prepared_at=datetime.now(timezone.utc).isoformat(),candidates=candidates,
        factory_sha256=sha(PROPOSALS/'native_factory.py'),core_sha256=sha(PROPOSALS/'strike_controller.py'),
        free_floating_core_sha256=sha(PROPOSALS/'free_floating.py'),
        CPU_check_sha256=sha(PROPOSALS/'CPU_CONTROLLER_CHECK.json'),model_sha256=sha(MODEL/'MODEL.json'),
        algorithm='Bounded pre-hit air-only public strike correction with full-body linear/angular momentum coupling, original tool guard, joint limits and.15rad residual mixer.',
        source_policy_selection_pending_actual_countermotion3_and_cold128=True,
        physics_rules_cases_target_and_thresholds_unchanged=True,native_execution=False,formal_acceptance=False)
    with (PROPOSALS/'PROPOSAL.json').open('x') as stream:stream.write(json.dumps(proposal,indent=2)+'\n')
    text=(PRIOR/'evaluate_air_countermotion.py').read_text()
    text=text.replace('verify_prior_source','verify_countermotion_source')
    text=text.replace('public_air_countermotion_20261027','free_floating_strike_20261027')
    text=text.replace("SCHEMA = 'g07-public-air-countermotion-v5'","SCHEMA = 'g07-public-momentum-strike-v5'")
    start=text.index("    assert factory_check['factory_sha256']")
    end=text.index("    if args.mode == 'search':",start)
    text=text[:start]+'''    assert factory_check['factory_sha256'] == sha(PROPOSALS / 'native_factory.py')
    assert len(cpu['results']) == 8 and len(proposal['candidates']) == 4
    assert cpu['both_baseline_replays_within_original_tolerance']
    assert all(r['maximum_normalized_action_change'] == 0. for r in cpu['results'] if r['candidate']['candidate_id'] == 0)
    assert factory_check['actual_baseline_CPU_vs_GPU_max_error'] <= 5e-4
    assert all(r['source_checkpoint_and_public_inputs_unchanged'] for r in cpu['results'])
'''+text[end:]
    text=once(text,"        assert [c['maximum_velocity_change_rad_s'] for c in candidates] == [r['maximum_velocity_change_rad_s'] for r in cpu['results']]",
        "        assert candidates == [r['candidate'] for r in cpu['results'][:4]] == [r['candidate'] for r in cpu['results'][4:]]")
    text=text.replace("len(search_metrics['history']) == 3","len(search_metrics['history']) == 4")
    text=text.replace('<= 307200','<= 409600')
    text=text.replace("name='independent_cold_selected_air_countermotion',\n                           **checkpoint['air_countermotion']",
        "name='independent_cold_selected_momentum_strike',\n                           **checkpoint['momentum_strike']")
    text=once(text,"assert checkpoint['counter_core_sha256'] == sha(PROPOSALS / 'countermotion.py')",
        "assert checkpoint['strike_core_sha256'] == sha(PROPOSALS / 'strike_controller.py')\n            assert checkpoint['free_floating_core_sha256'] == sha(PROPOSALS / 'free_floating.py')")
    text=once(text,"controller = factory.make_controller(base, 128, env.device, candidate['maximum_velocity_change_rad_s'])",
        "controller = factory.make_controller(base, 128, env.device,\n                **{key: candidate[key] for key in ['gain', 'leg_velocity_delta_cap_rad_s']})")
    text=once(text,"air_countermotion=dict(maximum_velocity_change_rad_s=candidate['maximum_velocity_change_rad_s']),",
        "momentum_strike={key: candidate[key] for key in ['gain', 'leg_velocity_delta_cap_rad_s']},")
    text=once(text,"counter_core_sha256=sha(PROPOSALS / 'countermotion.py'),",
        "strike_core_sha256=sha(PROPOSALS / 'strike_controller.py'),\n                    free_floating_core_sha256=sha(PROPOSALS / 'free_floating.py'),")
    text=text.replace("PROPOSALS / 'countermotion.py'","PROPOSALS / 'strike_controller.py'")
    text=once(text,"HERE.parent / 'centroidal_recovery_model_20261027/ACTUAL_MODEL_CHECK.json']",
        """HERE.parent / 'centroidal_recovery_model_20261027/ACTUAL_MODEL_CHECK.json',
             PROPOSALS / 'free_floating.py', PROPOSALS / 'ACTUAL_CHECK.json',
             HERE.parent / 'public_air_countermotion_20261027/native_factory.py',
             HERE.parent / 'public_air_countermotion_20261027/countermotion.py']""")
    # Two distinct factory paths share a basename. Preserve both by using
    # a path-derived unique filename and an explicit layout manifest.
    text=once(text,"    for path in extra:\n        directory = args.output / 'extra_execution_sources'\n        directory.mkdir(exist_ok=True)\n        shutil.copyfile(path, directory / path.name)",
        """    layout = {}
    for path in extra:
        directory = args.output / 'extra_execution_sources'
        directory.mkdir(exist_ok=True)
        name = hashlib.sha256(str(path).encode()).hexdigest()[:16] + '_' + path.name
        assert name not in layout.values()
        layout[str(path)] = name
        shutil.copyfile(path, directory / name)
        assert sha(directory / name) == extra_hashes[str(path)]
    atomic(args.output / 'extra_execution_source_layout.json', layout)""")
    text=text.replace('Execute public air-only countermotion proposals','Execute public pre-hit conserved-momentum strike proposals')
    text=text.replace('uses three sequential','uses four sequential')
    text=text.replace('native_air_countermotion_controller','native_momentum_strike_controller')
    ast.parse(text)
    with (HERE/'evaluate_momentum_strike.py').open('x') as stream:stream.write(text)
    text=(PRIOR/'run.py').read_text()
    assignments=dict(PRIOR="OUTPUTS / 'air_countermotion_queue01'",SOURCE="OUTPUTS / 'air_countermotion3_n128_01/best_candidate.pt'",
        SOURCE_COLD="OUTPUTS / 'air_countermotion3_best_cold128_01'",PROPOSALS="OUTPUTS / 'free_floating_strike_20261027'",
        SEARCH="OUTPUTS / 'momentum_strike4_n128_01'",COLD="OUTPUTS / 'momentum_strike4_best_cold128_01'")
    for name,value in assignments.items():text=assignment(text,name,value)
    replacements=[('evaluate_air_countermotion.py','evaluate_momentum_strike.py'),('1968858','1973878'),
        ('verify_prior_source','verify_countermotion_source'),
        ('WAITING_FOR_CONFIRMED_RECOVERY9_OWNER_AND_COLD128','WAITING_FOR_CONFIRMED_COUNTERMOTION3_OWNER_AND_COLD128'),
        ("assert read(PRIOR / 'state.json')['state'] == 'COMPLETED_RECOVERY_PRIOR9_AND_INDEPENDENT_COLD128'",
         "assert read(PRIOR / 'state.json')['state'] == 'COMPLETED_AIR_COUNTERMOTION3_AND_INDEPENDENT_COLD128'"),
        ('RUNNING_AIR_COUNTERMOTION_','RUNNING_MOMENTUM_STRIKE_'),
        ('CHECKING_SELECTED_SOURCE_CPU_BEFORE_AIR_COUNTERMOTION_NATIVE','CHECKING_SELECTED_SOURCE_CPU_BEFORE_MOMENTUM_STRIKE_NATIVE'),
        ("state('COMPLETED_AIR_COUNTERMOTION3_AND_INDEPENDENT_COLD128', final_summary",
         "state('COMPLETED_MOMENTUM_STRIKE4_AND_INDEPENDENT_COLD128', final_summary"),
        ('list(range(3))','list(range(4))'),
        ('env_steps_cap=307200, max_seconds=9000','env_steps_cap=409600, max_seconds=12600'),
        ("job(SEARCH, 'search', SOURCE, 307200, 9000, expected)","job(SEARCH, 'search', SOURCE, 409600, 12600, expected)")]
    for old,new in replacements:
        assert old in text,old
        text=text.replace(old,new)
    text=once(text,"    assert len(read(PROPOSALS / 'CPU_CONTROLLER_CHECK.json')['results']) == 3",
        "    assert len(read(PROPOSALS / 'CPU_CONTROLLER_CHECK.json')['results']) == 8\n    assert read(PROPOSALS / 'CPU_CONTROLLER_CHECK.json')['both_baseline_replays_within_original_tolerance']")
    text=once(text,"PROPOSALS / 'countermotion.py', PROPOSALS / 'check_controller.py', PROPOSALS / 'CPU_CONTROLLER_CHECK.json',",
        "PROPOSALS / 'strike_controller.py', PROPOSALS / 'free_floating.py', PROPOSALS / 'ACTUAL_CHECK.json',\n        PROPOSALS / 'analyze_actual.py', PROPOSALS / 'check_controller.py', PROPOSALS / 'CPU_CONTROLLER_CHECK.json',")
    text=text.replace('actual_source_pending_recovery9_and_independent_cold','actual_source_pending_countermotion3_and_independent_cold')
    text=text.replace('all9 recovery-prior original128 trials','all3 countermotion original128 trials')
    text=text.replace('Public post-hit air-only bounded leg countermotion; validated known28-body model; original.15rad mixer and hardware limits, arms unchanged.',
        'Public pre-hit air-only conserved-momentum strike correction; original complete-tool guard,18joint limits and.15rad mixer.')
    text=text.replace('After recovery-prior9/cold128 exits, execute3 air-only countermotion proposals.',
        'After countermotion3/cold128 exits, execute4 public conserved-momentum strike proposals.')
    needle="    assert read(PRIOR / 'state.json')['state'] == 'COMPLETED_AIR_COUNTERMOTION3_AND_INDEPENDENT_COLD128'"
    text=once(text,needle,"""    if read(PRIOR / 'state.json')['state'] == 'SKIPPED_SOURCE_TRAIN_GATES_MET_FORMAL_STILL_REQUIRED':
        state('SKIPPED_PARENT_SOURCE_TRAIN_GATES_MET_FORMAL_STILL_REQUIRED')
        return
"""+needle)
    ast.parse(text)
    with (HERE/'run.py').open('x') as stream:stream.write(text)
    record=dict(prepared_at=datetime.now(timezone.utc).isoformat(),generator_sha256=sha(Path(__file__)),
        source_sha256={str(PRIOR/n):sha(PRIOR/n) for n in ['evaluate_air_countermotion.py','run.py']},
        generated_sha256={str(HERE/n):sha(HERE/n) for n in ['evaluate_momentum_strike.py','run.py']},
        original_native_sampling_scoring_and_128_cases_preserved=True,unique_extra_source_layout_added=True,
        new_native_execution=False,formal_acceptance=False)
    with (HERE/'ENTRY_GENERATION_CHECK.json').open('x') as stream:stream.write(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(generated=list(record['generated_sha256']),native_execution=False)))


if __name__=='__main__':main()
