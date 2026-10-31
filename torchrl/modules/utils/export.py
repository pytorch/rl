# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase
from tensordict.utils import NestedKey, unravel_key
from torch import nn

from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules.tensordict_module.rnn import (
    GRUModule,
    LSTMModule,
    set_recurrent_mode,
)


class _ExplicitStatePolicy(nn.Module):
    """Tensor-in, tensor-out view of a TensorDict policy with its LSTM state as explicit inputs and outputs."""

    def __init__(
        self,
        policy: TensorDictModuleBase,
        observation_key: NestedKey,
        action_key: NestedKey,
        lstm: LSTMModule | None,
    ):
        super().__init__()
        self.policy = policy
        self.observation_key = observation_key
        self.action_key = action_key
        self.state_keys = [] if lstm is None else lstm.in_keys[1:3]
        self.next_state_keys = [] if lstm is None else lstm.out_keys[1:3]
        self.is_init_key = None if lstm is None else lstm.in_keys[3]

    def forward(
        self, obs: torch.Tensor, *state: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        tensordict = TensorDict({self.observation_key: obs}, batch_size=obs.shape[:1])
        for key, value in zip(self.state_keys, state):
            tensordict.set(key, value.transpose(0, 1))
        if self.is_init_key is not None:
            tensordict.set(
                self.is_init_key,
                torch.zeros(obs.shape[0], 1, dtype=torch.bool, device=obs.device),
            )
        tensordict = self.policy(tensordict)
        action = tensordict.get(self.action_key)
        if not self.next_state_keys:
            return action
        return (
            action,
            *(tensordict.get(key).transpose(0, 1) for key in self.next_state_keys),
        )


def export_onnx_policy(
    policy: TensorDictModuleBase,
    path: str | Path,
    observation_shape: Sequence[int],
    *,
    observation_key: NestedKey | None = None,
    action_key: NestedKey = "action",
    opset_version: int = 18,
) -> Path:
    """Export a feed-forward or LSTM policy to ONNX in the ``rsl_rl`` / ``mjlab`` layout.

    The graph computes the deterministic action for one observation:
    ``obs -> actions``, or ``(obs, h_in, c_in) -> (actions, h_out, c_out)``
    for a policy with one :class:`~torchrl.modules.LSTMModule`. Env transforms
    are not exported, so observation normalization must be part of ``policy``.
    Requires the ``onnx`` extra (``pip install torchrl[onnx]``).

    Args:
        policy (TensorDictModuleBase): the policy to export.
        path (str or Path): file to write.
        observation_shape (Sequence[int]): shape of one observation without
            the batch dimension, e.g. ``(61,)``.

    Keyword Args:
        observation_key (NestedKey, optional): key the policy reads the
            observation from. Defaults to the policy's only non-recurrent input.
        action_key (NestedKey, optional): key the policy writes the action to.
            Defaults to ``"action"``.
        opset_version (int, optional): ONNX opset version. Defaults to ``18``.

    Returns:
        The path written.

    Raises:
        NotImplementedError: if ``policy`` contains a
            :class:`~torchrl.modules.GRUModule` or more than one
            :class:`~torchrl.modules.LSTMModule`.

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
        >>> export_onnx_policy(policy, "walk.onnx", (61,))  # doctest: +SKIP
        PosixPath('walk.onnx')
    """
    if any(isinstance(module, GRUModule) for module in policy.modules()):
        raise NotImplementedError(
            "The explicit-state ONNX layout carries LSTM states (h, c); "
            "GRUModule policies cannot be exported with it."
        )
    lstms = [module for module in policy.modules() if isinstance(module, LSTMModule)]
    if len(lstms) > 1:
        raise NotImplementedError(
            f"The explicit-state ONNX layout carries one LSTM state, but the "
            f"policy holds {len(lstms)} LSTMModules."
        )
    lstm = lstms[0] if lstms else None
    if observation_key is None:
        recurrent_keys = set() if lstm is None else set(lstm.in_keys[1:])
        candidates = [key for key in policy.in_keys if key not in recurrent_keys]
        if len(candidates) != 1:
            raise ValueError(
                "Cannot tell which input holds the observation among "
                f"{candidates}; pass observation_key."
            )
        observation_key = candidates[0]
    graph = _ExplicitStatePolicy(
        deepcopy(policy).cpu(),
        unravel_key(observation_key),
        unravel_key(action_key),
        lstm,
    ).eval()
    example = [torch.zeros(1, *observation_shape)]
    input_names, output_names = ["obs"], ["actions"]
    if lstm is not None:
        primer = lstm.make_tensordict_primer()
        for key in lstm.in_keys[1:3]:
            num_layers, hidden_size = primer.primers[key].shape[-2:]
            example.append(torch.zeros(num_layers, 1, hidden_size))
        input_names += ["h_in", "c_in"]
        output_names += ["h_out", "c_out"]
    path = Path(path)
    with set_exploration_type(ExplorationType.DETERMINISTIC), set_recurrent_mode(
        False
    ), torch.no_grad():
        torch.onnx.export(
            graph,
            tuple(example),
            path,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            dynamo=True,
            external_data=False,
            verbose=False,
        )
    return path
