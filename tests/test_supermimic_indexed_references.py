import json
import sqlite3
import pytest
from scripts.build_supermimic_indexed_references import indexed_terminal_records


def fixture():
    connection=sqlite3.connect(':memory:')
    connection.execute('CREATE TABLE episodes (id INTEGER, record_json TEXT, scoring_schema TEXT)')
    connection.execute('CREATE TABLE occurrences (episode INTEGER, archive TEXT, identity_known INTEGER, raw_available INTEGER)')
    record=dict(env_id=3,episode_index=0,scoring_schema='historical-independent-complete5-v1',complete5=True)
    connection.execute('INSERT INTO episodes VALUES (?,?,?)',(1,json.dumps(record),record['scoring_schema']))
    connection.execute('INSERT INTO occurrences VALUES (1,?,1,1)',('/archive',))
    return connection,record


def test_index_requires_exact_score_and_archive_not_just_success_flag():
    connection,record=fixture()
    assert indexed_terminal_records(connection,'/archive',[record])==[record]
    for archive,rows in [('/another',[record]),('/archive',[dict(record,complete5=False)]),('/archive',[record,record])]:
        with pytest.raises(ValueError):indexed_terminal_records(connection,archive,rows)


def test_unidentified_or_missing_raw_cannot_be_used_as_tracking_reference():
    connection,record=fixture()
    connection.execute('UPDATE occurrences SET identity_known=0')
    with pytest.raises(ValueError):indexed_terminal_records(connection,'/archive',[record])


def test_training_archive_cannot_inherit_frozen_actor_identity(tmp_path):
    from scripts.index_supermimic_success_archive import archive_actor_sha256
    from quadsmash.rl.supermimic_physical_capture_v1 import PhysicalTransitionArchive

    initial = dict(policy_id='frozen-evaluation', actor_state_sha256='initial')
    assert archive_actor_sha256(initial) == 'initial'
    legacy = dict(initial, replay_source_schema=['initial_archive','online'], replay_stage_schema=['uniform'])
    assert archive_actor_sha256(legacy) is None
    current = dict(behavior_policy_scope='per_transition', actor_state_sha256=None,
                   scope='Native training transitions with per-row behavior identity',
                   initial_archive_identity=initial)
    PhysicalTransitionArchive(tmp_path/'archive', metadata=current)
    saved = json.loads((tmp_path/'archive/manifest.json').read_text())
    assert saved['scope'] == current['scope']
    assert saved['initial_archive_identity'] == initial
    assert archive_actor_sha256(saved) is None
