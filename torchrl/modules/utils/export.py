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

    The graph runs the policy on one unbatched step. Its inputs are
    ``policy.in_keys`` and its outputs are ``out_keys``, in that order, each
    named after its key with nested keys joined by ``"_"``. A recurrent policy
    takes its state and ``is_init`` as inputs and returns the next state. Env
    transforms are not exported, so observation normalization must be part of
    ``policy``. Requires the ``onnx`` extra (``pip install torchrl[onnx]``).

    The policy is traced as configured and nothing is overridden. Call
    ``policy.eval()`` first: a policy in training mode is exported with its
    training behavior, such as dropout, and PyTorch warns about it. Set the
    exploration type and recurrent mode around the call, as in the example.

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
        ValueError: if ``example`` has a batch dimension, or if two input or
            output keys get the same name.
        RuntimeError: if the exported graph does not hold one distinct input
            or output per key, so it cannot be named after them.

    Examples:
        >>> import torch
        >>> from tensordict.nn import (
        ...     AddStateIndependentNormalScale,
        ...     TensorDictModule,
        ...     TensorDictSequential,
        ... )
        >>> from torch import nn
        >>> from torchrl.envs import ObservationNorm
        >>> from torchrl.envs.utils import ExplorationType, set_exploration_type
        >>> from torchrl.modules import (
        ...     export_onnx_policy,
        ...     LSTMModule,
        ...     MLP,
        ...     ProbabilisticActor,
        ...     set_recurrent_mode,
        ...     TanhNormal,
        ... )
        >>> lstm = LSTMModule(input_size=61, hidden_size=64, in_key="observation_norm", out_key="features")
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
        >>> norm = ObservationNorm(
        ...     loc=torch.zeros(61), scale=torch.ones(61), in_keys=["observation"], out_keys=["observation_norm"]
        ... )
        >>> policy = TensorDictSequential(norm, actor).eval()
        >>> example = lstm.make_tensordict_primer().primers.zero()
        >>> example["observation"] = torch.zeros(61)
        >>> example["is_init"] = torch.zeros(1, dtype=torch.bool)
        >>> with set_exploration_type(ExplorationType.DETERMINISTIC), set_recurrent_mode(False):  # doctest: +SKIP
        ...     export_onnx_policy(policy, "walk.onnx", example)
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
    graph.training = policy.training
    input_names = [_onnx_name(key) for key in graph.in_keys]
    output_names = [_onnx_name(key) for key in graph.out_keys]
    names = input_names + output_names
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(
            f"Graph inputs and outputs need distinct names, but {duplicates} is used "
            f"more than once (in_keys={graph.in_keys}, out_keys={graph.out_keys}). "
            "Rename one of the keys, or write a transformed entry to its own key "
            "instead of in place."
        )
    with torch.no_grad():
        program = torch.onnx.export(
            graph,
            (),
            kwargs={
                name: example.get(key) for name, key in zip(input_names, graph.in_keys)
            },
            opset_version=opset_version,
            dynamo=True,
            verbose=False,
        )
    _name_graph(program.model.graph, input_names, output_names)
    path = Path(path)
    program.save(path, external_data=False)
    return path


def _name_graph(graph, input_names: list[str], output_names: list[str]) -> None:
    """Name graph inputs and outputs after their keys, renaming generated values in the way."""
    ends = dict(zip(graph.inputs, input_names)) | dict(zip(graph.outputs, output_names))
    if (len(graph.inputs), len(graph.outputs), len(ends)) != (
        len(input_names),
        len(output_names),
        len(input_names) + len(output_names),
    ):
        raise RuntimeError(
            f"The exported graph does not hold one distinct value per key: it has "
            f"{len(graph.inputs)} inputs and {len(graph.outputs)} outputs for the "
            f"names {input_names} and {output_names}."
        )
    reserved = set(ends.values())
    values = [
        *graph.initializers.values(),
        *(value for node in graph for value in node.outputs),
    ]
    taken = reserved | {value.name for value in values}
    for value in values:
        if value not in ends and value.name in reserved:
            index = 0
            while f"{value.name}_{index}" in taken:
                index += 1
            value.name = f"{value.name}_{index}"
            taken.add(value.name)
    for value, name in ends.items():
        value.name = name
