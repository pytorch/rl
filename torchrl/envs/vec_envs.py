# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import warnings

warnings.warn("vec_env.py has moved to batch_envs.py.", category=DeprecationWarning)

from .batched_envs import *  # noqa: F403, F401
