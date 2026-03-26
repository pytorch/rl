# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Testing utilities for TorchRL.

This module provides helper classes and utilities for testing TorchRL functionality,
particularly for distributed and Ray-based tests that require importable classes.
"""

from torchrl.testing.assertions import (
    check_rollout_consistency_multikey_env,
    rand_reset,
    rollout_consistency_assertion,
)
from torchrl.testing.env_creators import (
    get_transform_out,
    make_envs,
    make_multithreaded_env,
)
from torchrl.testing.gym_helpers import (
    BREAKOUT_VERSIONED,
    CARTPOLE_VERSIONED,
    CLIFFWALKING_VERSIONED,
    HALFCHEETAH_VERSIONED,
    PENDULUM_VERSIONED,
    PONG_VERSIONED,
)
from torchrl.testing.llm_mocks import (
    DummyStrDataLoader,
    DummyTensorDataLoader,
    MockTransformerConfig,
    MockTransformerModel,
    MockTransformerOutput,
)
from torchrl.testing.mocking_classes import FastImageEnv
from torchrl.testing.modules import (
    BiasModule,
    call_value_nets,
    LSTMNet,
    NonSerializableBiasModule,
)
from torchrl.testing.ray_helpers import (
    WorkerTransformerDoubleBuffer,
    WorkerTransformerNCCL,
    WorkerVLLMDoubleBuffer,
    WorkerVLLMNCCL,
)
from torchrl.testing.utils import (
    capture_log_records,
    dtype_fixture,
    generate_seeds,
    get_available_devices,
    get_default_devices,
    IS_WIN,
    make_tc,
    mp_ctx,
    PYTHON_3_9,
    retry,
    set_global_var,
)

__all__ = [
    # Assertions
    "check_rollout_consistency_multikey_env",
    "rand_reset",
    "rollout_consistency_assertion",
    # Environment creators
    "get_transform_out",
    "make_envs",
    "make_multithreaded_env",
    # Gym helpers
    "BREAKOUT_VERSIONED",
    "CARTPOLE_VERSIONED",
    "CLIFFWALKING_VERSIONED",
    "HALFCHEETAH_VERSIONED",
    "PENDULUM_VERSIONED",
    "PONG_VERSIONED",
    # LLM mocks
    "DummyStrDataLoader",
    "DummyTensorDataLoader",
    "MockTransformerConfig",
    "MockTransformerModel",
    "MockTransformerOutput",
    # Mocking classes
    "FastImageEnv",
    # Modules
    "BiasModule",
    "call_value_nets",
    "LSTMNet",
    "NonSerializableBiasModule",
    # Ray helpers
    "WorkerTransformerDoubleBuffer",
    "WorkerTransformerNCCL",
    "WorkerVLLMDoubleBuffer",
    "WorkerVLLMNCCL",
    # Utils
    "capture_log_records",
    "dtype_fixture",
    "generate_seeds",
    "get_available_devices",
    "get_default_devices",
    "IS_WIN",
    "make_tc",
    "mp_ctx",
    "PYTHON_3_9",
    "retry",
    "set_global_var",
]
