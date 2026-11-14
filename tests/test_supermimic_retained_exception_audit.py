import json
from quadsmash.rl.supermimic_retained_exception_audit import capture_retained_exceptions


def test_audit_finds_retained_exception_without_clearing_trace_or_dumping_values(tmp_path):
    retained = {}
    def retain_error():
        private_value = 'DO_NOT_SERIALIZE_THIS_VALUE'
        try:
            raise OSError(private_value)
        except OSError as error:
            retained['owned_exception'] = error
    retain_error()
    before = retained['owned_exception'].__traceback__
    count = capture_retained_exceptions(tmp_path)
    text = (tmp_path/'retained_exceptions.json').read_text()
    data = json.loads(text)
    assert count == data['count']
    assert retained['owned_exception'].__traceback__ is before
    assert 'DO_NOT_SERIALIZE_THIS_VALUE' not in text
    matches = [row for row in data['exceptions'] if any(
        frame['frame']['function'] == 'retain_error' for frame in row['frames'])]
    assert len(matches) == 1
    assert any('owned_exception' in owner.get('matching_keys', []) for owner in matches[0]['owners'])
    assert matches[0]['frames'][0]['frame']['local_types']['private_value'] == 'builtins.str'
