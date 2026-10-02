# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/pytorch/examples/tree/main/distributed/FSDP2
# (BSD 3-Clause License, Copyright (c) 2017, Pytorch contributors).

import argparse
import os

import torch
from checkpoint import Checkpointer
from model import ModelArgs, Transformer
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from utils import inspect_mixed_precision, inspect_model


def verify_min_gpu_count(min_gpus: int = 2) -> bool:
    """verification that we have at least 2 gpus to run dist examples"""
    has_gpu = torch.accelerator.is_available()
    gpu_count = torch.accelerator.device_count()
    return has_gpu and gpu_count >= min_gpus


def set_modules_to_forward_prefetch(model, num_to_forward_prefetch):
    for i, layer in enumerate(model.layers):
        if i >= len(model.layers) - num_to_forward_prefetch:
            break
        layers_to_prefetch = [
            model.layers[i + j] for j in range(1, num_to_forward_prefetch + 1)
        ]
        layer.set_modules_to_forward_prefetch(layers_to_prefetch)


def set_modules_to_backward_prefetch(model, num_to_backward_prefetch):
    for i, layer in enumerate(model.layers):
        if i < num_to_backward_prefetch:
            continue
        layers_to_prefetch = [
            model.layers[i - j] for j in range(1, num_to_backward_prefetch + 1)
        ]
        layer.set_modules_to_backward_prefetch(layers_to_prefetch)


def main(args):
    _min_gpu_count = 2
    if not verify_min_gpu_count(min_gpus=_min_gpu_count):
        print(
            f"Unable to locate sufficient {_min_gpu_count} gpus to run this example. Exiting."
        )
        exit()
    rank = int(os.environ["LOCAL_RANK"])
    device_type = torch.accelerator.current_accelerator()
    device = torch.device(f"{device_type}:{rank}")
    torch.accelerator.set_device_index(rank)
    print(f"Running on rank {rank} on device {device}")

    backend = torch.distributed.get_default_backend_for_device(device)
    torch.distributed.init_process_group(backend=backend, device_id=device)
    is_rank_zero = torch.distributed.get_rank() == 0

    torch.manual_seed(0)
    vocab_size = 1024
    batch_size = 32
    seq_len = 64
    model_args = ModelArgs(
        n_layers=10,
        n_heads=4,
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        dropout_p=0,
    )
    with torch.device("meta"):
        model = Transformer(model_args)
    fsdp_kwargs = {}
    if args.mixed_precision:
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
    for layer in model.layers:
        fully_shard(layer, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)

    inspect_model(model)

    if args.explicit_prefetching:
        set_modules_to_forward_prefetch(model, num_to_forward_prefetch=2)
        set_modules_to_backward_prefetch(model, num_to_backward_prefetch=2)

    # Loading copies into existing tensors, so allocate the (sharded) parameters
    # either way.
    model.to_empty(device=device)
    # Named parameters put parameter names into the optimizer's state dict, which
    # the checkpoint uses to key optimizer state.
    optim = torch.optim.Adam(model.named_parameters(), lr=1e-2)

    checkpointer = Checkpointer(args.checkpoint_dir)
    if checkpointer.last_step is None:
        model.reset_parameters()
        start_step = 0
    else:
        checkpointer.load(model, optim)
        start_step = checkpointer.last_step + 1
        if is_rank_zero:
            print(f"Resumed from {checkpointer.path(checkpointer.last_step)}")

    if args.mixed_precision:
        inspect_mixed_precision(model)

    end_step = start_step + args.steps
    for step in range(start_step, end_step):
        if args.explicit_prefetching:
            model.unshard()
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        loss = model(x).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        with checkpointer.lock():
            optim.step()
        optim.zero_grad()

        if (step + 1) % args.save_every == 0 or step == end_step - 1:
            # Returns once the state is staged; training continues while it is written.
            path = checkpointer.save(model, optim, step)
            if is_rank_zero:
                print(f"Step {step}: saving {path}")

    checkpointer.close()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PyTorch FSDP2 example with torch_checkpointing"
    )
    parser.add_argument("--explicit-prefetching", action="store_true", default=False)
    parser.add_argument("--mixed-precision", action="store_true", default=False)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--steps", type=int, default=10, help="training steps per run")
    parser.add_argument("--save-every", type=int, default=5)
    args = parser.parse_args()

    main(args)
