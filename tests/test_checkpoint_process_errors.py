# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pickle
from functools import partial
from unittest import mock

import pytest
from torch_checkpointing.checkpoint_base import CheckpointWriteInfo
from torch_checkpointing.checkpoint_process import (
    CheckpointProcess,
    CheckpointProcessConfig,
    RequestType,
    WorkerResponse,
)
from torch_checkpointing.checkpoint_writer import (
    CheckpointWriterArgs,
    CheckpointWriterConfig,
)
from torch_checkpointing.storage.filesystem import LocalFileSystemStorageConfig
from torch_checkpointing.types import RankInfo


class CallbackError(Exception):
    pass


class TwoArgumentError(Exception):
    def __init__(self, message, detail):
        super().__init__(message, detail)


def _raise_worker_error(error_type, *args):
    message = "worker rejected checkpoint"
    if error_type == "local":

        class LocalError(Exception):
            pass

        raise LocalError(message)
    if error_type is TwoArgumentError:
        raise TwoArgumentError(message, "detail")
    error = error_type(message)
    # Transport must not serialize the exception instance itself.
    error.unpicklable = lambda: None
    raise error


def _initialize():
    pass


@pytest.mark.parametrize(
    "error_type, expected_type, during_init",
    [
        pytest.param(FileExistsError, FileExistsError, False, id="builtin"),
        pytest.param(CallbackError, CallbackError, False, id="custom"),
        pytest.param(EOFError, EOFError, False, id="application-pipe-error"),
        pytest.param(TwoArgumentError, RuntimeError, False, id="constructor-fallback"),
        pytest.param("local", RuntimeError, False, id="unpicklable-type"),
        pytest.param(CallbackError, CallbackError, True, id="initialization"),
    ],
)
def test_worker_exception_type_and_traceback(
    tmp_path, error_type, expected_type, during_init
):
    rank_info = RankInfo(
        global_rank=0, global_world_size=1, role_rank=0, role_world_size=1
    )
    fail = partial(_raise_worker_error, error_type)
    process = CheckpointProcess(
        rank_info=rank_info,
        config=CheckpointProcessConfig(),
        subprocess_init_fn=fail if during_init else _initialize,
        subprocess_init_args=(),
        checkpoint_writer_args=CheckpointWriterArgs(
            config=CheckpointWriterConfig(),
            rank_info=rank_info,
            storage_config=LocalFileSystemStorageConfig(use_direct_io=False),
            pre_write_callback=fail,
        ),
    )
    destination = tmp_path / "checkpoint"
    try:
        with pytest.raises(expected_type, match="rejected checkpoint") as caught:
            if during_init:
                process.wait_for_init()
            else:
                process.write(
                    CheckpointWriteInfo(checkpoint_items={}), str(destination)
                ).result(timeout=30)
        assert type(caught.value) is expected_type
        assert "_raise_worker_error" in str(caught.value)
    finally:
        process.close()


def test_unavailable_exception_type_retains_worker_error():
    process = CheckpointProcess.__new__(CheckpointProcess)
    process._parent_end = mock.Mock()
    process._parent_end.recv.return_value = WorkerResponse(
        RequestType.WRITE_CHECKPOINT,
        success=False,
        error_msg="original worker traceback",
        serialized_exception_type=pickle.dumps(CallbackError),
    )
    # The exception's defining module may be unavailable in the parent.
    with mock.patch(
        "torch_checkpointing.checkpoint_process.pickle.loads",
        side_effect=ModuleNotFoundError("callback module unavailable"),
    ):
        with pytest.raises(RuntimeError, match="original worker traceback"):
            process._recv()
