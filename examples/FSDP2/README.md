# FSDP2 with `torch_checkpointing`

Trains a small transformer with [FSDP2](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)
and checkpoints it with `torch_checkpointing`. It is adapted from the
[PyTorch FSDP2 example](https://github.com/pytorch/examples/tree/main/distributed/FSDP2),
which gathers the full model and optimizer state onto rank 0 and saves it there.
Here, instead:

- every rank saves only its own shards of the model and optimizer state;
- saves are asynchronous: `save()` returns once the state is copied off the GPU,
  and a background process writes it while training continues;
- loading reshards onto the current device mesh, so a run can resume on a
  different number of GPUs.

The checkpointing code is in [`checkpoint.py`](./checkpoint.py).

## Running

Needs at least 2 GPUs. From the repository root:

```bash
uv run torchrun --nproc_per_node 2 examples/FSDP2/example.py
```

Without uv, install `torch_checkpointing` and run
`torchrun --nproc_per_node 2 example.py` from this directory.

Each run trains 10 steps, saving every 5 and at the end (`--steps`, `--save-every`), to
`checkpoints/step_<n>` under the current directory (`--checkpoint-dir`). If a
checkpoint already exists, the run resumes from the newest one. To resume on a
different number of GPUs, change `--nproc_per_node`:

```bash
uv run torchrun --nproc_per_node 4 examples/FSDP2/example.py
```

To enable explicit prefetching:

```bash
uv run torchrun --nproc_per_node 2 examples/FSDP2/example.py --explicit-prefetching
```

To enable mixed precision:

```bash
uv run torchrun --nproc_per_node 2 examples/FSDP2/example.py --mixed-precision
```
