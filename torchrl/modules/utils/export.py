# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase, TensorDictSequential
from tensordict.utils import NestedKey, unravel_key

from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules.tensordict_module.rnn import set_recurrent_mode


def _onnx_name(key: NestedKey) -> str:
    """Name a graph input or output after its key, as tensordict's ``dispatch`` does."""
    key = unravel_key(key)
    return key if isinstance(key, str) else "_".join(key)


def export_onnx_policy(
    policy: TensorDictModuleBase,
    path: str | Path,
    example: TensorDictBase,
    *,
    out_keys: Sequence[NestedKey] | None = None,
    opset_version: int = 18,
) -> Path:
    """Export a policy to ONNX by tracing it with :func:`torch.onnx.export`.

    The graph runs the policy on one unbatched step under
    ``ExplorationType.DETERMINISTIC``. Its inputs are ``policy.in_keys`` and
    its outputs are ``out_keys``, in that order, each named after its key with
    nested keys joined by ``"_"``. A recurrent policy takes its state and
    ``is_init`` as inputs and returns the next state. Env transforms are not
    exported, so observation normalization must be part of ``policy``.
    Requires the ``onnx`` extra (``pip install torchrl[onnx]``).

    Args:
        policy (TensorDictModuleBase): the policy to export.
        path (str or Path): file to write.
        example (TensorDictBase): unbatched input to trace with, holding every
            entry of ``policy.in_keys``.

    Keyword Args:
        out_keys (sequence of NestedKey, optional): entries the graph returns,
            in order. Defaults to ``"action"`` followed by ``("next", key)`` for
            every input key the policy also writes under ``"next"``, i.e. its
            recurrent state.
        opset_version (int, optional): ONNX opset version. Defaults to ``18``.

    Returns:
        The path written.

    Raises:
        ValueError: if ``example`` has a batch dimension.

    Examples:
        >>> import torch
        >>> from tensordict.nn import (
        ...     AddStateIndependentNormalScale,
        ...     TensorDictModule,
        ...     TensorDictSequential,
        ... )
        >>> from torch import nn
        >>> from torchrl.envs import ObservationNorm
        >>> from torchrl.modules import (
        ...     export_onnx_policy,
        ...     LSTMModule,
        ...     MLP,
        ...     ProbabilisticActor,
        ...     TanhNormal,
        ... )
        >>> lstm = LSTMModule(input_size=61, hidden_size=64, in_key="observation", out_key="features")
        >>> head = TensorDictModule(
        ...     nn.Sequential(MLP(in_features=64, out_features=14, num_cells=[64]), AddStateIndependentNormalScale(14)),
        ...     in_keys=["features"],
        ...     out_keys=["loc", "scale"],
        ... )
        >>> actor = ProbabilisticActor(
        ...     TensorDictSequential(lstm, head),
        ...     in_keys=["loc", "scale"],
        ...     distribution_class=TanhNormal,
        ... )
        >>> norm = ObservationNorm(loc=torch.zeros(61), scale=torch.ones(61), in_keys=["observation"])
        >>> policy = TensorDictSequential(norm, actor)
        >>> example = lstm.make_tensordict_primer().primers.zero()
        >>> example["observation"] = torch.zeros(61)
        >>> example["is_init"] = torch.zeros(1, dtype=torch.bool)
        >>> export_onnx_policy(policy, "walk.onnx", example)  # doctest: +SKIP
        PosixPath('walk.onnx')
    """
    if example.batch_dims:
        raise ValueError(
            f"example must be unbatched, got batch_size={example.batch_size}."
        )
    if out_keys is None:
        out_keys = ["action"] + [
            unravel_key(("next", key))
            for key in policy.in_keys
            if unravel_key(("next", key)) in policy.out_keys
        ]
    graph = TensorDictSequential(policy, selected_out_keys=list(out_keys))
    inputs = {_onnx_name(key): example.get(key) for key in graph.in_keys}
    path = Path(path)
    with set_exploration_type(ExplorationType.DETERMINISTIC), set_recurrent_mode(
        False
    ), torch.no_grad():
        torch.onnx.export(
            graph,
            (),
            path,
            kwargs=inputs,
            output_names=[_onnx_name(key) for key in graph.out_keys],
            opset_version=opset_version,
            dynamo=True,
            external_data=False,
            verbose=False,
        )
    return path
