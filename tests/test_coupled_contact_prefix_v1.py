from types import SimpleNamespace
import numpy as np
from scripts.coupled_contact_prefix_v1 import CoupledContactRollout,arm_basis
from scripts.terminal_push_refinement_v1 import refine_terminal


def fixture():
    obj=object.__new__(CoupledContactRollout)
    obj.commands=np.zeros((29,18));obj.arm_reference=np.zeros((30,6))
    obj.interface=SimpleNamespace(limits=np.tile([-1.,1.],(18,1)))
    return obj


def test_arm_endpoint_position_and_rate_are_independent_and_start_continuous():
    assert arm_basis(0)==(0,0)
    assert arm_basis(1)==(1,0)
    eps=1e-6
    fp,gp=arm_basis(eps)
    assert abs(fp/eps)<4e-6 and abs(gp/eps)<2e-6
    fm,gm=arm_basis(1-eps)
    assert abs((1-fm)/eps)<4e-6
    assert abs(-gm/eps-1)<3e-6


def test_leg_ownership_and_arm_reference_are_updated_without_mutation():
    obj=fixture();a=np.r_[np.full(12,.02),np.full(6,.05),np.full(6,.03)]
    old=obj.commands.copy();arm=obj.arm_reference.copy()
    u,q=obj.command_at_node(a,obj.commands[28],28,0.)
    np.testing.assert_array_equal(u[:12],np.zeros(12))
    np.testing.assert_allclose(u[12:],.05);np.testing.assert_array_equal(u[12:],q)
    np.testing.assert_array_equal(obj.commands,old);np.testing.assert_array_equal(obj.arm_reference,arm)
    assert obj.amplitude_bounds().shape==(24,2)


def test_all_joint_directions_can_contribute_to_one_prefix_step():
    rng=np.random.default_rng(12);matrix=rng.normal(size=(9,24))
    def evaluate(a):return dict(path_valid=True,normalized_residual=matrix@a-.05)
    result=refine_terminal(evaluate,parameter_radii=CoupledContactRollout.radii)
    assert result['improved'] and len(result['trials'])==26
    assert len(result['usable_columns'])==24
    a=result['trials'][result['selected_index']]['amplitude_rad']
    assert np.all(abs(a)<=CoupledContactRollout.radii+1e-12)
    assert np.linalg.norm(a[:12])>0 and np.linalg.norm(a[12:18])>0 and np.linalg.norm(a[18:])>0
    assert not result['valid_teacher_reference']


def test_arm_position_bounds_choose_rate_basis_direction_correctly():
    obj=fixture();obj.commands[:,12:]=1.;obj.arm_reference[:,0:]=1.
    b=obj.amplitude_bounds()
    assert np.all(b[12:18,1]==0)  # Positive endpoint offsets are forbidden.
    assert np.all(b[18:24,0]==0)  # Negative g(s) reverses the allowed parameter side.


def test_combined_arm_constraints_equal_actual_command_and_reference_offsets():
    obj=fixture();a=np.r_[np.full(12,.01),np.full(6,.08),np.full(6,-.04)]
    C,low,high=obj.combined_arm_constraints()
    actual=np.array([obj.command_at_node(a,obj.commands[k],k,0.)[0][12:] for k in range(29)])
    np.testing.assert_allclose((C@a)[:29*6].reshape(29,6),actual,atol=1e-14)
    assert np.all(C@a>=low) and np.all(C@a<=high)


def test_optimizer_enforces_combined_parameter_constraint_without_clipping():
    def evaluate(a):return dict(path_valid=True,normalized_residual=np.r_[a[:2]-.02,np.zeros(7)])
    C=np.zeros((1,12));C[0,:2]=1.
    result=refine_terminal(evaluate,linear_constraints=(C,np.array([-.1]),np.array([.01])))
    assert result['improved']
    final=result['trials'][result['selected_index']]['amplitude_rad']
    assert final[:2].sum()<=.010000001
    assert result['linear_constraint_rows']==1


def test_cache_signature_changes_when_rollout_semantics_change(monkeypatch):
    from scripts.coupled_contact_prefix_v1 import evaluation_signature
    from scripts.terminal_push_refinement_v1 import TerminalPushRollout
    before=evaluation_signature()
    monkeypatch.setattr(CoupledContactRollout,'command_at_node',TerminalPushRollout.command_at_node)
    after=evaluation_signature()
    assert before['rollout_method_sha256']!=after['rollout_method_sha256']
    assert before['source_sha256']==after['source_sha256']
