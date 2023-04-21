# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional, Tuple

import torch

from tensordict.nn import TensorDictModuleBase as ModuleBase

from tensordict.tensordict import NO_DEFAULT, TensorDictBase
from tensordict.utils import prod

from torch import nn


class LSTMModule(ModuleBase):
    """An embedder for an LSTM module.

    This class adds the following functionality to :class:`torch.nn.LSTM`:

    - Compatibility with TensorDict: the hidden states are reshaped to match
      the tensordict batch size.
    - Optional multi-step execution: with torch.nn, one has to choose between
      :class:`torch.nn.LSTMCell` and :class:`torch.nn.LSTM`, the former being
      compatible with single step inputs and the latter being compatible with
      multi-step. This class enables both usages.


    After construction, the module is *not* set in temporal mode, ie. it will
    expect single steps inputs.

    If in temporal mode, it is expected that the last dimension of the tensordict
    marks the number of steps. There is no constrain on the dimensionality of the
    tensordict (except that it must be greater than one for temporal inputs).

    Args:
        lstm (torch.nn.Module): a :class:`torch.nn.LSTM` instance. For simplicity, this class does not
            handle the construction of the module.
        in_keys (list of str): a triplet of strings corresponding to the input value,
            first and second hidden key.
        out_keys (list of str): a triplet of strings corresponding to the output value,
            first and second hidden key.
            .. note::
              For a better integration with TorchRL's environments, the best naming
              for the output hidden key is ``("next", "hidden0")`` or similar, such
              that the hidden values are passed from step to step during a rollout.

    Attributes:
        temporal_mode: Returns the temporal mode of the module.

    Methods:
        set_temporal_mode: controls whether the module should be executed in
            temporal mode.

    Examples:
        >>> from torchrl.envs import TransformedEnv, InitTracker
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> from torchrl.modules import MLP
        >>> from torch import nn
        >>> from tensordict.nn import TensorDictSequential as Seq, TensorDictModule as Mod
        >>> env = TransformedEnv(GymEnv("Pendulum-v1"), InitTracker())
        >>> lstm = nn.LSTM(input_size=env.observation_spec["observation"].shape[-1], hidden_size=64, batch_first=True)
        >>> lstm_module = LSTMModule(lstm, in_keys=["observation", "hidden0", "hidden1"], out_keys=["intermediate", ("next", "hidden0"), ("next", "hidden1")])
        >>> mlp = MLP(num_cells=[64], out_features=1)
        >>> policy = Seq(lstm_module, Mod(mlp, in_keys=["intermediate"], out_keys=["action"]))
        >>> policy(env.reset())
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                intermediate: Tensor(shape=torch.Size([64]), device=cpu, dtype=torch.float32, is_shared=False),
                is_init: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        hidden0: Tensor(shape=torch.Size([1, 64]), device=cpu, dtype=torch.float32, is_shared=False),
                        hidden1: Tensor(shape=torch.Size([1, 64]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([]),
                    device=cpu,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=cpu,
            is_shared=False)

    """

    def __init__(self, lstm: nn.LSTM, in_keys, out_keys):
        super().__init__()
        if len(in_keys) != 3:
            raise ValueError(
                f"LSTMModule expects 3 inputs: a value, and two hidden states. Got in_keys {in_keys} instead."
            )
        if len(out_keys) != 3:
            raise ValueError(
                f"LSTMModule expects 3 outputs: a value, and two hidden states. Got in_keys {in_keys} instead."
            )
        if not lstm.batch_first:
            raise ValueError("The input lstm must have batch_first=True")
        self.lstm = lstm
        self.in_keys = in_keys
        self.out_keys = out_keys
        self._temporal_mode = False

    @property
    def temporal_mode(self):
        return self._temporal_mode

    @temporal_mode.setter
    def temporal_mode(self, value):
        raise RuntimeError("temporal_mode cannot be changed in-place. Call `module.set")

    def set_temporal_mode(self, mode: bool = True):
        """Returns a new copy of the module that shares the same lstm model but with a different ``temporal_mode`` attribute (if it differs).

        A copy is created such that the module can be used with divergent behaviour
        in various parts of the code (inference vs training):

        Examples:
            >>> from torchrl.envs import TransformedEnv, InitTracker, step_mdp
            >>> from torchrl.envs.libs.gym import GymEnv
            >>> from torchrl.modules import MLP
            >>> from tensordict import TensorDict
            >>> from torch import nn
            >>> from tensordict.nn import TensorDictSequential as Seq, TensorDictModule as Mod
            >>> env = TransformedEnv(GymEnv("Pendulum-v1"), InitTracker())
            >>> lstm = nn.LSTM(input_size=env.observation_spec["observation"].shape[-1], hidden_size=64, batch_first=True)
            >>> lstm_module = LSTMModule(lstm, in_keys=["observation", "hidden0", "hidden1"], out_keys=["intermediate", ("next", "hidden0"), ("next", "hidden1")])
            >>> mlp = MLP(num_cells=[64], out_features=1)
            >>> # building two policies with different behaviours:
            >>> policy_inference = Seq(lstm_module, Mod(mlp, in_keys=["intermediate"], out_keys=["action"]))
            >>> policy_training = Seq(lstm_module.set_temporal_mode(True), Mod(mlp, in_keys=["intermediate"], out_keys=["action"]))
            >>> traj_td = env.rollout(3) # some random temporal data
            >>> traj_td = policy_training(traj_td)
            >>> # let's check that both return the same results
            >>> td_inf = TensorDict({}, traj_td.shape[:-1])
            >>> for td in traj_td.unbind(-1):
            ...     td_inf = td_inf.update(td.select("is_init", "observation", ("next", "observation")))
            ...     td_inf = policy_inference(td_inf)
            ...     td_inf = step_mdp(td_inf)
            ...
            >>> torch.testing.assert_close(td_inf["hidden0"], traj_td[..., -1]["next", "hidden0"])
        """
        if mode is self._temporal_mode:
            return self
        out = LSTMModule(self.lstm, in_keys=self.in_keys, out_keys=self.out_keys)
        out._temporal_mode = mode
        return out

    def forward(self, tensordict: TensorDictBase):
        # we want to get an error if the value input is missing, but not the hidden states
        defaults = [NO_DEFAULT, None, None]
        shape = tensordict.shape
        tensordict_shaped = tensordict
        if self.temporal_mode:
            # if less than 2 dims, unsqueeze
            while tensordict_shaped.ndim < 2:
                tensordict_shaped = tensordict_shaped.unsqueeze(0)
            if tensordict_shaped.ndim > 2:
                dims_to_flatten = tensordict_shaped.ndim - 2
                nelts = prod(tensordict_shaped.shape[: dims_to_flatten + 1])
                tensordict_shaped = tensordict_shaped.apply(
                    lambda value: value.flatten(0, dims_to_flatten),
                    batch_size=[nelts, tensordict_shaped.shape[-1]],
                )
        else:
            tensordict_shaped = tensordict.view(-1).unsqueeze(-1)
        value, hidden0, hidden1 = (
            tensordict_shaped.get(key, default)
            for key, default in zip(self.in_keys, defaults)
        )
        is_init = tensordict_shaped.get("is_init")
        if is_init.any() and hidden0 is not None:
            hidden0[is_init] = 0
            hidden1[is_init] = 0
        val, hidden0, hidden1 = self._lstm(value, hidden0, hidden1)
        tensordict_shaped.set(self.out_keys[0], val)
        tensordict_shaped.set(self.out_keys[1], hidden0)
        tensordict_shaped.set(self.out_keys[2], hidden1)
        if shape != tensordict_shaped.shape:
            tensordict.update(tensordict_shaped.reshape(shape))
        return tensordict

    def _lstm(
        self,
        input: torch.Tensor,
        hidden0_in: Optional[torch.Tensor] = None,
        hidden1_in: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if input.ndim != 3:
            raise ValueError("the input does not have 3 dimensions.")

        batch, steps = input.shape[:2]
        if not self.temporal_mode and steps != 1:
            raise ValueError("Expected a single step")

        if hidden1_in is None and hidden0_in is None:
            shape = (batch, steps)
            hidden0_in, hidden1_in = [
                torch.zeros(
                    *shape,
                    self.lstm.num_layers,
                    self.lstm.hidden_size,
                    device=input.device,
                    dtype=input.dtype,
                )
                for _ in range(2)
            ]
        elif hidden1_in is None or hidden0_in is None:
            raise RuntimeError(
                f"got type(hidden0)={type(hidden0_in)} and type(hidden1)={type(hidden1_in)}"
            )

        # we only need the first hidden state
        _hidden0_in = hidden0_in[:, 0]
        _hidden1_in = hidden1_in[:, 0]
        hidden = (
            _hidden0_in.transpose(-3, -2).contiguous(),
            _hidden1_in.transpose(-3, -2).contiguous(),
        )

        y, hidden = self.lstm(input, hidden)
        # dim 0 in hidden is num_layers, but that will conflict with tensordict
        hidden = tuple(_h.transpose(0, 1) for _h in hidden)

        out = [y, *hidden]
        # we pad the hidden states with zero to make tensordict happy
        for i in range(1, 3):
            out[i] = torch.stack(
                [torch.zeros_like(out[i]) for _ in range(input.shape[1] - 1)]
                + [out[i]],
                1,
            )
        return tuple(out)
