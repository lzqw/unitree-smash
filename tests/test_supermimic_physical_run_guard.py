import torch
from quadsmash.rl.supermimic_physical_run_guard_v1 import critic_ready,atomic_torch_save


def test_quality_gate_distinguishes_zero_missing_and_nonfinite():
    assert critic_ready(200,0.)
    assert critic_ready(200,50.)
    assert not critic_ready(199,0.)
    assert not critic_ready(200,None)
    assert not critic_ready(None,0.)
    assert not critic_ready(200,float('nan'))
    assert not critic_ready(200,float('inf'))
    assert not critic_ready(200,50.0001)
    assert critic_ready(0,0.,required=0)


def test_atomic_checkpoint_preserves_real_update_counter(tmp_path):
    path=tmp_path/'latest.pt'
    atomic_torch_save(dict(actor_updates=8,last_critic_loss=82.45,weight=torch.ones(3)),path)
    stored=torch.load(path,weights_only=True)
    assert stored['actor_updates']==8 and stored['last_critic_loss']==82.45
    assert not path.with_name('latest.pt.pending').exists()
    atomic_torch_save(dict(actor_updates=9,last_critic_loss=0.,weight=torch.zeros(3)),path)
    assert torch.load(path,weights_only=True)['actor_updates']==9
