#!/usr/bin/env python3

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from bounded_waypoint_student import BoundedWaypointStudent


def main() -> None:
    torch.manual_seed(7)
    mean = np.linspace(-1.0, 1.0, 39, dtype=np.float32)
    std = np.linspace(0.1, 2.0, 39, dtype=np.float32)
    model = BoundedWaypointStudent(mean, std)
    observations = torch.randn(32, 39) * 100.0
    actions = model(observations)
    assert actions.shape == (32, 4)
    assert torch.all(actions <= 1.0)
    assert torch.all(actions >= -1.0)

    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "student.pt"
        torch.save(model.checkpoint_payload(), path)
        restored = BoundedWaypointStudent.from_checkpoint(path, torch.device("cpu"))
        assert torch.allclose(actions, restored(observations), atol=1e-7)

    print("bounded waypoint student tests passed")


if __name__ == "__main__":
    main()
