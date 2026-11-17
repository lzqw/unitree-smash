import importlib.util,json,sqlite3
from pathlib import Path

def load_script():
    p=Path(__file__).resolve().parents[1]/'scripts/index_supermimic_success_archive.py'
    spec=importlib.util.spec_from_file_location('inventory',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_inventory_deduplicates_reports_but_never_admits_unknown_trajectory(tmp_path):
    m=load_script();root=tmp_path/'source';root.mkdir()
    good=dict(case_id='case1',episode_index=0,complete5=True,scoring_schema='historical-independent-complete5-v1')
    unknown=dict(case_id='case2',episode_index=0,complete5=None)
    (root/'episodes_summary.json').write_text(json.dumps([good,unknown]))
    (root/'episodes_copy.json').write_text(json.dumps([good]))
    (root/'policy.pt').write_bytes(b'not a pickle, must never load')
    result=m.build([root],tmp_path/'db')
    assert result['episode_records']==2 and result['success_evidence_records']==1
    assert result['linked_success_records']==0 and result['admitted_to_training']==0
    db=sqlite3.connect(tmp_path/'db/success_trajectories.sqlite3')
    assert db.execute('SELECT COUNT(*) FROM occurrences').fetchone()[0]==3
    assert db.execute('SELECT complete5 FROM episodes WHERE case_id="case2"').fetchone()[0] is None

def test_missing_archive_chunks_are_not_raw_data(tmp_path):
    m=load_script();root=tmp_path/'run';raw=root/'physical_transitions';raw.mkdir(parents=True)
    (root/'episodes_summary.json').write_text(json.dumps([dict(env_id=1,complete5=True)]))
    (raw/'manifest.json').write_text(json.dumps(dict(chunks=[dict(file='missing.pt',sha256='x')],actor_state_sha256='a',policy_execution={'gate':'causal'},task_contract={'asset':'a'})))
    result=m.build([root],tmp_path/'db')
    assert result['archives']==1 and result['linked_success_records']==0
    assert result['admitted_to_training']==0

def test_resume_preserves_partial_inventory_and_records_unreadable_directory(tmp_path,monkeypatch):
    m=load_script();root=tmp_path/'source';root.mkdir();bad=root/'broken';bad.mkdir()
    (root/'episodes_summary.json').write_text(json.dumps([dict(case_id='x',complete5=True)]))
    original=Path.is_symlink
    def check(path):
        if path==bad:raise OSError('simulated unavailable mount')
        return original(path)
    monkeypatch.setattr(Path,'is_symlink',check)
    result=m.build([root],tmp_path/'db')
    assert any('unavailable mount' in e for e in result['errors'])
    # Simulate interrupted scan after its last committed chunk, no completed report.
    (tmp_path/'db/inventory.json').unlink()
    resumed=m.build([root],tmp_path/'db',resume=True)
    assert resumed['episode_records']==1 and resumed['success_evidence_records']==1
