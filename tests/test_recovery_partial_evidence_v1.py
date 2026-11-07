import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.recovery_partial_evidence_v1 import NUMERIC, save_partial_recovery


def replay(tmp_path):
    records=[]
    for tick in [460,461,462]:
        records.append(dict(control_tick=tick,mode='teacher',
                            **{name:np.array([tick],dtype=float) for name in NUMERIC}))
    return SimpleNamespace(output=tmp_path,records=records,plans=[dict(tick=460,decision={'accepted':True})],
                           start=460,case=3,steps=340,query_check={'passed':True})


def test_pending_request_is_never_saved_as_executed(tmp_path):
    r=replay(tmp_path);before=copy.deepcopy(r.records)
    result=save_partial_recovery(r,462)
    assert result['executed_recovery_records']==2 and result['pending_request_records']==1
    assert not result['full_run_admission'] and not result['valid_expert_labels']
    with np.load(tmp_path/'recovery_partial'/result['trace_file']) as x:
        assert x['control_tick'].tolist()==[460,461]
    assert all(np.array_equal(a['q'],b['q']) for a,b in zip(before,r.records))


def test_pending_plan_after_same_completed_count_gets_separate_revision(tmp_path):
    r=replay(tmp_path);save_partial_recovery(r,462)
    first=json.loads((tmp_path/'recovery_partial/latest.json').read_text())['file']
    r.plans.append(dict(tick=462,decision={'accepted':False}))
    result=save_partial_recovery(r,462)
    second=json.loads((tmp_path/'recovery_partial/latest.json').read_text())['file']
    assert first!=second
    assert not result['planning_records'][-1]['any_step_completed_after_query']
    assert (tmp_path/'recovery_partial'/first).exists()


def test_mutated_published_plan_and_gap_are_rejected(tmp_path):
    r=replay(tmp_path);save_partial_recovery(r,462)
    r.plans[0]['decision']['accepted']=False
    with pytest.raises(ValueError,match='mutated'):
        save_partial_recovery(r,463)
    r.records[1]['control_tick']=465
    with pytest.raises(ValueError,match='Noncontiguous'):
        save_partial_recovery(r,463)


def test_partial_completion_never_emits_complete_artifacts(tmp_path):
    r=replay(tmp_path);save_partial_recovery(r,463)
    assert not (tmp_path/'completed.json').exists()
    assert not (tmp_path/'sequence_native_result.json').exists()
    assert not (tmp_path/'metrics.json').exists()
