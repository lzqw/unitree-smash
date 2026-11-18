import copy
import pytest
import torch
from scripts.aggregate_supermimic_native_probes import aggregate,sha
from quadsmash.rl.supermimic_native_pair_capture_v1 import BRANCHES


def make(tmp_path,index,head='approach',episode=None):
    files={branch:tmp_path/f'{index}_{branch}.raw' for branch in BRANCHES}
    for branch,path in files.items():path.write_bytes(f'synthetic-unit-test-{index}-{branch}'.encode())
    p=dict(state=torch.zeros(1,55),restore_mode='from-ground-identical-prefix',
           prefix_sha256={b:f'prefix{index}' for b in BRANCHES},executor_before_sha256={b:f'memory{index}' for b in BRANCHES},
           native_contact_modes={b:dict(had_contact=True) for b in BRANCHES})
    obj=dict(schema='native-paired-prefix-region-v1',head=head,runtime_identity={'actor':'fixed'},actor_state={'w':torch.ones(1)},
             pairs=p,sample_ids=[str(index if episode is None else episode)],command_starts=[{'unit-test-start':index}],
             source_paths={k:str(v) for k,v in files.items()},source_sha256={k:sha(v) for k,v in files.items()})
    path=tmp_path/f'{index}.pt';torch.save(obj,path);return path,obj


def test_native_aggregation_preserves_per_probe_prefixes_and_distinct_episode_count(tmp_path):
    a,_=make(tmp_path,1);b,_=make(tmp_path,2,episode=1)
    receipt=aggregate([a,b],tmp_path/'joined.pt')
    assert receipt['probe_count']==2 and receipt['independent_episode_count']==1
    assert receipt['passed'] is None
    joined=torch.load(tmp_path/'joined.pt',weights_only=True)
    assert joined['pairs']['prefix_sha256']['plus']==['prefix1','prefix2']
    assert len(joined['command_starts'])==2


def test_duplicate_or_changed_raw_evidence_and_mixed_heads_are_rejected(tmp_path):
    a,_=make(tmp_path,1);b,_=make(tmp_path,2,head='recovery')
    with pytest.raises(ValueError,match='Duplicate'):aggregate([a,a],tmp_path/'duplicate.pt')
    with pytest.raises(ValueError,match='head'):aggregate([a,b],tmp_path/'mixed.pt')
    (tmp_path/'1_plus.raw').write_bytes(b'changed')
    with pytest.raises(ValueError,match='bytes changed'):aggregate([a],tmp_path/'changed.pt')
