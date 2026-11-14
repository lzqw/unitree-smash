#!/usr/bin/env python3
"""Saved-state full G07 partial-reset/branch isolation; no native steps."""
import argparse,hashlib,json
from pathlib import Path
import torch
from quadsmash.control.g07_shared_target_residual_v1 import make_g07_residual_controller
from quadsmash.rl.supermimic_g07_memory_preview_v1 import clone_execution_graph,restore_captured_execution,G07MemoryPreviewV1
from quadsmash.rl.supermimic_physical_loop_loading_v1 import freeze_execution_modules
from quadsmash.rl.supermimic_physical_capture_v1 import capture_controller


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    m=json.loads((args.archive/'manifest.json').read_text())
    source=Path(m['source_checkpoint']);assert hashlib.sha256(source.read_bytes()).hexdigest()==m['source_sha256']
    template=make_g07_residual_controller(torch.load(source,map_location='cpu',weights_only=False),128,torch.device('cpu'),'/home/autodrive/LZQW/unitree_smash')
    freeze_execution_modules(template)
    row=torch.load(args.archive/m['chunks'][8]['file'],map_location='cpu',weights_only=True)['rows'][4]
    memory=restore_captured_execution(template,row['executor_before']);snapshot=capture_controller(memory)
    native=clone_execution_graph(memory);branch=clone_execution_graph(memory,learning=True);sibling=clone_execution_graph(memory)
    ids=torch.tensor([0,3]);native.reset(ids);branch.reset(ids)
    obs=row['observation'];action=row['requested_action'];default=row['before']['physical']['joint_default'][0];limits=row['before']['auxiliary']['command_limits']
    expected=native.apply_residual({'policy':obs},action,default,limits)
    actual,_,_=G07MemoryPreviewV1(template).forward(obs,action,default,limits,branch)
    torch.testing.assert_close(actual.final_normalized,expected,atol=2e-6,rtol=0)
    pristine=clone_execution_graph(memory).apply_residual({'policy':obs},action,default,limits)
    other=sibling.apply_residual({'policy':obs},action,default,limits)
    torch.testing.assert_close(pristine,other,atol=0,rtol=0)
    after=capture_controller(memory)
    for key,value in snapshot['tensors'].items():torch.testing.assert_close(after['tensors'][key],value,atol=0,rtol=0,equal_nan=True)
    assert snapshot['values']==after['values']
    result=dict(tick=260,reset_lanes=ids.tolist(),partial_reset_command_error=float((actual.final_normalized-expected).abs().max()),
        untouched_branch_error=float((pristine-other).abs().max()),source_memory_unchanged=True,
        source_tensor_fields=len(snapshot['tensors']),native_steps=0,actor_updates=0)
    (args.output/'completed.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
