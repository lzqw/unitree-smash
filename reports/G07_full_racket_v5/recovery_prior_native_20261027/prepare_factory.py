"""Extend only source dispatch; retain the published recovery-prior algorithm."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / 'recovery_motion_prior_20261027/prior_controller.py'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    text = ORIGINAL.read_text()
    text = text.replace('def load_seed(directory=HERE):',
        "def load_seed(directory=HERE.parent / 'recovery_motion_prior_20261027'):")
    marker = "    if source.get('schema') in ['g07-measured-swing-contact-recovery-v5', 'g07-public-early-lateral-intercept-v5']:"
    assert text.count(marker) == 1
    branch = '''    if source.get('schema') == 'g07-windowed-swing-contact-recovery-v5':
        path = HERE.parent / 'windowed_swing_20261027/window_controller.py'
        if hashlib.sha256(path.read_bytes()).hexdigest() != source['factory_sha256']:
            raise ValueError('Measured windowed-swing factory changed')
        spec = importlib.util.spec_from_file_location('preserved_windowed_swing_factory', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.WindowedSwingController(source['source_public_policy'], count, device,
            source['windowed_swing']['joint_velocity_rad_s'])
'''
    text = text.replace(marker, branch+marker)
    before, after = ast.parse(ORIGINAL.read_text()), ast.parse(text)
    names = ['RecoveryPriorController', 'candidates']
    for name in names:
        a = next(node for node in before.body if getattr(node, 'name', None) == name)
        b = next(node for node in after.body if getattr(node, 'name', None) == name)
        assert ast.dump(a) == ast.dump(b)
    target = HERE / 'prior_controller.py'
    with target.open('x') as stream:
        stream.write(text)
    record = dict(prepared_at=datetime.now(timezone.utc).isoformat(), original_source_sha256=sha(ORIGINAL),
        derived_source_sha256=sha(target), generator_sha256=sha(Path(__file__)),
        prior_core_and_candidate_definitions_AST_identical=True,
        changed_scope='Add admission of actual windowed-swing source; resolve existing immutable seed directory.',
        native_execution=False, formal_acceptance=False)
    with (HERE / 'FACTORY_GENERATION_CHECK.json').open('x') as stream:
        stream.write(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
