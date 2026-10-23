from types import SimpleNamespace
from quadsmash.rl.native_checkpoint import interface


def test_absolute_position_policy_has_an_explicit_distinct_action_contract(tmp_path):
    asset=tmp_path/'asset.usd'
    asset.write_text('pinned test asset')
    env=SimpleNamespace(step_dt=.01,robot=SimpleNamespace(joint_names=['one']),
        joint_ids=[0],cfg=SimpleNamespace(robot=SimpleNamespace(spawn=SimpleNamespace(usd_path=asset))))
    assert interface(env)['action_family']=='jump-event-v1'
    env.action_family='absolute-joint-limits-v1'
    assert interface(env)['action_family']=='absolute-joint-limits-v1'
