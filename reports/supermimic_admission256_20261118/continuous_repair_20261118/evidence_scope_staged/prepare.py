from pathlib import Path
root=Path.cwd();out=Path(__file__).parent
p=root/'quadsmash/rl/supermimic_evidence_binding_v1.py'
s=p.read_text()
helper='''# Only prediction modules absent from the real native action/physics path may
# differ when immutable native responses are re-evaluated. Unknown/new files,
# actor, G07, mixer, rate limits and execution preview remain exact checks.
MODEL_ONLY_CODE = frozenset((
    'supermimic_ball_sequence_v1.py', 'supermimic_action_ball_v1.py',
    'supermimic_articulated_inertia_v1.py', 'supermimic_racket_material_contract_v1.py',
    'supermimic_impact_timing_v1.py',
))
NATIVE_REQUIRED_CODE = frozenset((
    'supermimic_policy_execution_v1.py', 'supermimic_v1.py',
    'control/g07_shared_target_residual_v1.py', 'control/full_racket_residual_v4.py',
    'control/full_racket_rally_controller_v4.py', 'control/full_racket_arm_target_rate_v4.py',
    'control/persistent_recovery_assistance_v1.py',
))


def native_code_compatibility(recorded, current):
    """Source applicability only; never a model accuracy/policy admission pass.

    The complete old and new maps are retained. A model code change requires
    current-model native-response rechecking in assess_region/probe_applicability.
    """
    if not isinstance(recorded, dict) or not isinstance(current, dict) or not recorded or not current:
        return dict(compatible=False, reason='missing-code-identity', differences={})
    diff={k:dict(recorded=recorded.get(k), current=current.get(k))
          for k in recorded.keys() | current.keys() if recorded.get(k)!=current.get(k)}
    if not diff:
        return dict(compatible=True, reason='exact-source-identity', differences={})
    missing=sorted(NATIVE_REQUIRED_CODE - (recorded.keys() & current.keys()))
    forbidden=sorted(set(diff)-MODEL_ONLY_CODE)
    return dict(compatible=not missing and not forbidden,
        reason='prediction-code-only-requires-current-response-check' if not missing and not forbidden else 'native-or-unknown-code-changed',
        missing_native_files=missing, forbidden_changes=forbidden, differences=diff,
        scope='native measured transition can be re-used; current model accuracy is not certified')


def static_identity_failures(origin, current):
    failures=[k for k in ('schema','task_contract_sha256','policy_execution') if origin.get(k)!=current.get(k)]
    if not native_code_compatibility(origin.get('execution_code_sha256'),current.get('execution_code_sha256'))['compatible']:
        failures.append('execution_code_sha256')
    return failures


'''
s=s.replace('def runtime_identity(',helper+'def runtime_identity(',1)
s=s.replace("    static=('execution_code_sha256','schema','task_contract_sha256','policy_execution')", "    static_failures=static_identity_failures(declared or {},current)")
s=s.replace("and not any(k in mismatch for k in static))", "and not static_failures)")
(out/p.name).write_text(s)
p=root/'quadsmash/rl/supermimic_probe_region_v1.py';s=p.read_text()
s=s.replace("    failures=[key for key in STATIC if origin.get(key)!=current.get(key)]", "    from .supermimic_evidence_binding_v1 import static_identity_failures, native_code_compatibility\n    failures=static_identity_failures(origin,current)\n    code_scope=native_code_compatibility(origin.get('execution_code_sha256'),current.get('execution_code_sha256'))")
s=s.replace("        origin_identity=origin,accepted_update_age=age,", "        origin_identity=origin,code_scope=code_scope,accepted_update_age=age,")
(out/p.name).write_text(s)
p=root/'scripts/build_supermimic_probe_region.py';s=p.read_text()
s=s.replace("    if base['execution_code_sha256']!=identity['execution_code_sha256']:raise ValueError('Runtime/probe code differs')", "    from quadsmash.rl.supermimic_evidence_binding_v1 import native_code_compatibility\n    code_scope=native_code_compatibility(base['execution_code_sha256'],identity['execution_code_sha256'])\n    if not code_scope['compatible']:raise ValueError('Runtime/probe native execution code differs: '+str(code_scope))")
s=s.replace("        source_paths={k:str(Path(p).resolve()) for k,p in paths.items()},", "        source_paths={k:str(Path(p).resolve()) for k,p in paths.items()},\n        native_recorded_code_sha256=base['execution_code_sha256'],runtime_model_code_scope=code_scope,")
(out/p.name).write_text(s)
print('Staged 3 files only; running production sources unchanged')
