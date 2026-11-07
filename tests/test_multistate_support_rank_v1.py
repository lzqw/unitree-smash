from scripts.refine_multistate_support_v1 import candidate_rank


def result(ok,steps=520):
    return dict(accepted=ok,completed_model_steps=steps,rank=[0 if ok else 1,520-steps,0,1.],cost=1.,failure=None)


def test_two_new_successes_cannot_compensate_for_lost_nominal():
    preserved=[result(True),result(False,100),result(False,120)]
    lost=[result(False,500),result(True),result(True)]
    assert candidate_rank(preserved)<candidate_rank(lost)
    improved=[result(True),result(True),result(False,120)]
    assert candidate_rank(improved)<candidate_rank(preserved)


def test_compute_expiration_never_becomes_a_physical_rank():
    values=[result(True),result(False,100),result(False,120)]
    values[1]['failure']={'reason':'declared_compute_budget'}
    assert candidate_rank(values) is None
