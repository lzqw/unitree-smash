"""Check native packed friction-buffer reduction without a physics mock."""
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.native_ground_pair_observer_v1 import aggregate_friction


def main():
    forces = torch.tensor([[0., 2., 0.], [1., 0., 0.], [float('nan')] * 3, [0., 0., 3.]])
    points = torch.tensor([[1., 0., 0.], [0., 2., 0.], [float('nan')] * 3, [0., 1., 0.]])
    counts = torch.tensor([[2, 0, 1, 0]])
    starts = torch.tensor([[0, 0, 3, 999]])
    f, m, bad = aggregate_friction(forces, points, counts, starts, torch.zeros(3))
    assert torch.equal(f, torch.tensor([[1., 2., 0.], [0., 0., 0.], [0., 0., 3.], [0., 0., 0.]]))
    assert torch.equal(m, torch.tensor([[0., 0., 0.], [0., 0., 0.], [3., 0., 0.], [0., 0., 0.]]))
    assert not bool(bad)
    _, _, overflow = aggregate_friction(forces, points, torch.tensor([[2]]), torch.tensor([[3]]), torch.zeros(3))
    assert bool(overflow)
    print(json.dumps(dict(passed=True, padding_ignored=True, overflow_detected=True,
        native_backend_exercised=False, controller_validated=False)))


if __name__ == '__main__':
    main()
