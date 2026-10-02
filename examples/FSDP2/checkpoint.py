# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/pytorch/examples/tree/main/distributed/FSDP2
# (BSD 3-Clause License, Copyright (c) 2017, Pytorch contributors).

import os
import re
from contextlib import AbstractContextManager

import torch
from torch.distributed.fsdp import FSDPModule
from torch_checkpointing import CheckpointManager, ItemSpec
from torch_checkpointing.default_resharder import DefaultResharder
from torch_checkpointing.state_transformations import (
    optimizer_transform_post,
    optimizer_transform_pre,
)

STEP_DIR = re.compile(r"step_(\d+)")


def get_latest_step(folder: str) -> int | None:
    """Return the highest `n` among the `step_<n>` checkpoints in `folder`, if any."""
    if not os.path.isdir(folder):
        return None
    steps = [
        int(match.group(1))
        for name in os.listdir(folder)
        if (match := STEP_DIR.fullmatch(name))
    ]
    return max(steps, default=None)


def init_optimizer_state(optim: torch.optim.Optimizer) -> None:
    """Create the optimizer's lazily initialized state (e.g. Adam's `exp_avg`).

    Loading copies saved tensors into existing ones, so the state has to exist
    first. Takes one step with zero gradients and a learning rate of 0, which
    leaves the parameters unchanged.
    """
    for group in optim.param_groups:
        for param in group["params"]:
            param.grad = torch.zeros_like(param)
    lrs = [group["lr"] for group in optim.param_groups]
    for group in optim.param_groups:
        group["lr"] = 0.0
    optim.step()
    for group, lr in zip(optim.param_groups, lrs):
        group["lr"] = lr
    optim.zero_grad(set_to_none=True)


class Checkpointer:
    """Saves and loads sharded model and optimizer state with torch_checkpointing.

    Each rank writes only its own shards, in a background process, so `save()`
    returns before the checkpoint is written. Loading reshards the saved state
    onto the current device mesh, so a run can resume on a different number of
    GPUs than the one that saved it.
    """

    def __init__(self, folder: str):
        self.folder = folder
        # The step is recorded in the checkpoint's name rather than in the
        # checkpoint: an item without a resharder is saved per rank, so a run with
        # more ranks would find no copy of it for its extra ranks.
        self.last_step = get_latest_step(folder)
        self.manager = CheckpointManager(
            CheckpointManager.Config(
                items={
                    "model": ItemSpec(resharder=DefaultResharder()),
                    "optimizer": ItemSpec(resharder=DefaultResharder()),
                },
            )
        )

    def path(self, step: int) -> str:
        return os.path.join(self.folder, f"step_{step}")

    def load(self, model: FSDPModule, optim: torch.optim.Optimizer) -> None:
        """Load the latest checkpoint into `model` and `optim`."""
        assert self.last_step is not None, "no checkpoint to load"
        init_optimizer_state(optim)
        restored = self.manager.load(
            self.path(self.last_step),
            into={
                # Sharded DTensors, copied into in place.
                "model": model.state_dict(),
                # Keyed by parameter name instead of index, so that it can be resharded.
                "optimizer": optimizer_transform_pre(optim.state_dict()),
            },
            strict=True,
        )
        optim.load_state_dict(optimizer_transform_post(restored["optimizer"]))

    def save(self, model: FSDPModule, optim: torch.optim.Optimizer, step: int) -> str:
        """Start saving a checkpoint for `step`; returns its path."""
        path = self.path(step)
        self.manager.save(
            path,
            {
                "model": model.state_dict(),
                "optimizer": optimizer_transform_pre(optim.state_dict()),
            },
        )
        return path

    def lock(self) -> AbstractContextManager:
        """Hold while updating parameters, so a save never sees a half-applied step."""
        return self.manager.lock()

    def close(self) -> None:
        """Wait for the last save to finish. Call before destroying the process group."""
        self.manager.close()
