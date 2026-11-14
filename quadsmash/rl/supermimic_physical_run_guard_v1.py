"""Explicit quality gates and atomic checkpoints; no simulator dependency."""
import math,os
from pathlib import Path
import torch


def critic_ready(updates,loss,required=200,maximum=50.):
    if updates is None or loss is None:return False
    return int(updates)>=required and math.isfinite(float(loss)) and float(loss)<=maximum


def atomic_torch_save(payload,path):
    path=Path(path);temporary=path.with_name(path.name+'.pending')
    with temporary.open('wb') as stream:
        torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)
