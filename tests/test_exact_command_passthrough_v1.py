import numpy as np
import torch
from quadsmash.control.exact_command_passthrough_v1 import exact_admitted_action


def test_noop_retains_original_bits_despite_roundtrip_error():
    original=torch.linspace(-.9,.9,18,dtype=torch.float32)
    requested=np.linspace(-2.,1.,18,dtype=np.float32)
    rounded=original.numpy().astype(float)+1e-8
    result=exact_admitted_action(original,requested,requested.astype(float),rounded)
    assert torch.equal(result,original)
    assert result.data_ptr()!=original.data_ptr()


def test_real_admission_still_changes_only_selected_joints():
    original=torch.linspace(-.9,.9,18,dtype=torch.float32)
    requested=np.linspace(-2.,1.,18,dtype=np.float32)
    admitted=requested.astype(float);admitted[[12,17]]-=.1
    encoded=original.numpy().astype(float)+1e-8;encoded[[12,17]]=[.1,.2]
    result=exact_admitted_action(original,requested,admitted,encoded)
    unchanged=torch.ones(18,dtype=torch.bool);unchanged[[12,17]]=False
    assert torch.equal(result[unchanged],original[unchanged])
    torch.testing.assert_close(result[[12,17]],torch.tensor([.1,.2]),rtol=0,atol=0)
