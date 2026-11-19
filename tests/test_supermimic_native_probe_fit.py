from scripts.fit_supermimic_native_probe_response import validate_probe_identity_reuse, concat_pairs
from scripts.fit_supermimic_native_probe_response import metrics
import torch


def _identity(*, actor='a', task='t', execution='e', version=0, native='n', model='m'):
    from quadsmash.rl.supermimic_evidence_binding_v1 import NATIVE_REQUIRED_CODE
    code = {name: native for name in NATIVE_REQUIRED_CODE}
    code['supermimic_action_ball_v1.py'] = model
    return dict(schema='supermimic-runtime-identity-v1', actor_state_sha256=actor,
                task_contract_sha256=task, policy_execution=execution,
                policy_version=version, execution_code_sha256=code)


def test_native_label_fit_allows_prediction_code_revision():
    rows = [{'runtime_identity': _identity(model='old')},
            {'runtime_identity': _identity(model='new')}]
    result = validate_probe_identity_reuse(rows)
    assert result['prediction_code_reuse']
    assert result['bundle_identity_count'] == 2


def test_native_label_fit_rejects_execution_code_revision():
    rows = [{'runtime_identity': _identity(native='old')},
            {'runtime_identity': _identity(native='new')}]
    try:
        validate_probe_identity_reuse(rows)
    except ValueError as exc:
        assert 'Native execution identity mismatch' in str(exc)
    else:
        raise AssertionError('Expected native execution identity mismatch')


def test_response_metrics_report_ball_components_and_zero_baseline():
    native_plus = torch.zeros(1, 55)
    native_plus[:, 52:55] = torch.tensor([[1., 2., 3.]])
    pairs = {'native_plus': native_plus, 'native_minus': torch.zeros(1, 55)}
    result = metrics(torch.zeros(1, 3), pairs)
    assert result['component_rmse'] == [1., 2., 3.]
    assert result['zero_response_baseline_rmse'] > 0


def test_native_contact_metadata_marks_only_verified_zero_response_as_negative():
    state = torch.zeros(2, 55)
    zeros = torch.zeros(2, 55)
    pairs = {
        'state': state,
        'previous_action': torch.zeros(2, 18),
        'feet': torch.zeros(2, 4),
        'present': torch.ones(2, dtype=torch.bool),
        'elapsed': torch.zeros(2),
        'launch': torch.zeros(2, 6),
        'targets_minus': torch.zeros(2, 18),
        'targets_plus': torch.zeros(2, 18),
        'native_minus': zeros.clone(),
        'native_plus': zeros.clone(),
        'native_contact_modes': {
            'minus': [{'had_contact': False}, {'had_contact': True}],
            'plus': [{'had_contact': False}, {'had_contact': True}],
        },
    }
    result = concat_pairs([{'pairs': pairs}], torch.device('cpu'))
    assert result['verified_no_contact'].tolist() == [True, False]
