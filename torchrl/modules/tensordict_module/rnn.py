# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
from typing import Optional, Tuple

import torch
from tensordict import TensorDictBase, unravel_key_list

from tensordict.nn import TensorDictModuleBase as ModuleBase

from tensordict.tensordict import NO_DEFAULT
from tensordict.utils import prod

from torch import nn

from torchrl.data import UnboundedContinuousTensorSpec
from torchrl.objectives.value.functional import (
    _inv_pad_sequence,
    _split_and_pad_sequence,
)
from torchrl.objectives.value.utils import _get_num_per_traj_init


class LSTMModule(ModuleBase):
    """An embedder for an LSTM module.

    This class adds the following functionality to :class:`torch.nn.LSTM`:

    - Compatibility with TensorDict: the hidden states are reshaped to match
      the tensordict batch size.
    - Optional multi-step execution: with torch.nn, one has to choose between
      :class:`torch.nn.LSTMCell` and :class:`torch.nn.LSTM`, the former being
      compatible with single step inputs and the latter being compatible with
      multi-step. This class enables both usages.


    After construction, the module is *not* set in recurrent mode, ie. it will
    expect single steps inputs.

    If in recurrent mode, it is expected that the last dimension of the tensordict
    marks the number of steps. There is no constrain on the dimensionality of the
    tensordict (except that it must be greater than one for temporal inputs).

    .. note::
      This class can handle multiple consecutive trajectories along the time dimension
      *but* the final hidden values should not be trusted in those cases (ie. they
      should not be re-used for a consecutive trajectory).
      The reason is that LSTM returns only the last hidden value, which for the
      padded inputs we provide can correspont to a 0-filled input.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two LSTMs together to form a `stacked LSTM`,
            with the second LSTM taking in outputs of the first LSTM and
            computing the final results. Default: 1
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            LSTM layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0

    Keyword Args:
        in_key (str or tuple of str): the input key of the module. Exclusive use
            with ``in_keys``. If provided, the recurrent keys are assumed to be
            ["recurrent_state_h", "recurrent_state_c"] and the ``in_key`` will be
            appended before these.
        in_keys (list of str): a triplet of strings corresponding to the input value,
            first and second hidden key. Exclusive with ``in_key``.
        out_key (str or tuple of str): the output key of the module. Exclusive use
            with ``out_keys``. If provided, the recurrent keys are assumed to be
            [("next", "recurrent_state_h"), ("next", "recurrent_state_c")]
            and the ``out_key`` will be
            appended before these.
        out_keys (list of str): a triplet of strings corresponding to the output value,
            first and second hidden key.
            .. note::
              For a better integration with TorchRL's environments, the best naming
              for the output hidden key is ``("next", <custom_key>)``, such
              that the hidden values are passed from step to step during a rollout.
        device (torch.device or compatible): the device of the module.
        lstm (torch.nn.LSTM, optional): an LSTM instance to be wrapped.
            Exclusive with other nn.LSTM arguments.

    Attributes:
        recurrent_mode: Returns the recurrent mode of the module.

    Methods:
        set_recurrent_mode: controls whether the module should be executed in
            recurrent mode.

    Examples:
        >>> from torchrl.envs import TransformedEnv, InitTracker
        >>> from torchrl.envs import GymEnv
        >>> from torchrl.modules import MLP
        >>> from torch import nn
        >>> from tensordict.nn import TensorDictSequential as Seq, TensorDictModule as Mod
        >>> env = TransformedEnv(GymEnv("Pendulum-v1"), InitTracker())
        >>> lstm_module = LSTMModule(
        ...     input_size=env.observation_spec["observation"].shape[-1],
        ...     hidden_size=64,
        ...     in_keys=["observation", "rs_h", "rs_c"],
        ...     out_keys=["intermediate", ("next", "rs_h"), ("next", "rs_c")])
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
                        rs_c: Tensor(shape=torch.Size([1, 64]), device=cpu, dtype=torch.float32, is_shared=False),
                        rs_h: Tensor(shape=torch.Size([1, 64]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([]),
                    device=cpu,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
                terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([]),
            device=cpu,
            is_shared=False)

    """

    DEFAULT_IN_KEYS = ["recurrent_state_h", "recurrent_state_c"]
    DEFAULT_OUT_KEYS = [("next", "recurrent_state_h"), ("next", "recurrent_state_c")]

    def __init__(
        self,
        input_size: int = None,
        hidden_size: int = None,
        num_layers: int = 1,
        bias: bool = True,
        batch_first=True,
        dropout=0,
        proj_size=0,
        bidirectional=False,
        *,
        in_key=None,
        in_keys=None,
        out_key=None,
        out_keys=None,
        device=None,
        lstm=None,
    ):
        super().__init__()
        if lstm is not None:
            if not lstm.batch_first:
                raise ValueError("The input lstm must have batch_first=True.")
            if lstm.bidirectional:
                raise ValueError("The input lstm cannot be bidirectional.")
            if input_size is not None or hidden_size is not None:
                raise ValueError(
                    "An LSTM instance cannot be passed along with class argument."
                )
        else:
            if not batch_first:
                raise ValueError("The input lstm must have batch_first=True.")
            if bidirectional:
                raise ValueError("The input lstm cannot be bidirectional.")
            lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bias=bias,
                dropout=dropout,
                proj_size=proj_size,
                device=device,
                batch_first=True,
                bidirectional=False,
            )
        if not ((in_key is None) ^ (in_keys is None)):
            raise ValueError(
                f"Either in_keys or in_key must be specified but not both or none. Got {in_keys} and {in_key} respectively."
            )
        elif in_key:
            in_keys = [in_key, *self.DEFAULT_IN_KEYS]

        if not ((out_key is None) ^ (out_keys is None)):
            raise ValueError(
                f"Either out_keys or out_key must be specified but not both or none. Got {out_keys} and {out_key} respectively."
            )
        elif out_key:
            out_keys = [out_key, *self.DEFAULT_OUT_KEYS]

        in_keys = unravel_key_list(in_keys)
        out_keys = unravel_key_list(out_keys)
        if not isinstance(in_keys, (tuple, list)) or (
            len(in_keys) != 3 and not (len(in_keys) == 4 and in_keys[-1] == "is_init")
        ):
            raise ValueError(
                f"LSTMModule expects 3 inputs: a value, and two hidden states (and potentially an 'is_init' marker). Got in_keys {in_keys} instead."
            )
        if not isinstance(out_keys, (tuple, list)) or len(out_keys) != 3:
            raise ValueError(
                f"LSTMModule expects 3 outputs: a value, and two hidden states. Got out_keys {out_keys} instead."
            )
        self.lstm = lstm
        if "is_init" not in in_keys:
            in_keys = in_keys + ["is_init"]
        self.in_keys = in_keys
        self.out_keys = out_keys
        self._recurrent_mode = False

    def make_tensordict_primer(self):
        from torchrl.envs.transforms.transforms import TensorDictPrimer

        def make_tuple(key):
            if isinstance(key, tuple):
                return key
            return (key,)

        out_key1 = make_tuple(self.out_keys[1])
        in_key1 = make_tuple(self.in_keys[1])
        out_key2 = make_tuple(self.out_keys[2])
        in_key2 = make_tuple(self.in_keys[2])
        if out_key1 != ("next", *in_key1) or out_key2 != ("next", *in_key2):
            raise RuntimeError(
                "make_tensordict_primer is supposed to work with in_keys/out_keys that "
                "have compatible names, ie. the out_keys should be named after ('next', <in_key>). Got "
                f"in_keys={self.in_keys} and out_keys={self.out_keys} instead."
            )
        return TensorDictPrimer(
            {
                in_key1: UnboundedContinuousTensorSpec(
                    shape=(self.lstm.num_layers, self.lstm.hidden_size)
                ),
                in_key2: UnboundedContinuousTensorSpec(
                    shape=(self.lstm.num_layers, self.lstm.hidden_size)
                ),
            }
        )

    @property
    def recurrent_mode(self):
        return self._recurrent_mode

    @recurrent_mode.setter
    def recurrent_mode(self, value):
        raise RuntimeError(
            "recurrent_mode cannot be changed in-place. Call `module.set"
        )

    @property
    def temporal_mode(self):
        warnings.warn(
            "temporal_mode is deprecated, use recurrent_mode instead.",
            category=DeprecationWarning,
        )
        return self.recurrent_mode

    def set_recurrent_mode(self, mode: bool = True):
        """Returns a new copy of the module that shares the same lstm model but with a different ``recurrent_mode`` attribute (if it differs).

        A copy is created such that the module can be used with divergent behaviour
        in various parts of the code (inference vs training):

        Examples:
            >>> from torchrl.envs import TransformedEnv, InitTracker, step_mdp
            >>> from torchrl.envs import GymEnv
            >>> from torchrl.modules import MLP
            >>> from tensordict import TensorDict
            >>> from torch import nn
            >>> from tensordict.nn import TensorDictSequential as Seq, TensorDictModule as Mod
            >>> env = TransformedEnv(GymEnv("Pendulum-v1"), InitTracker())
            >>> lstm = nn.LSTM(input_size=env.observation_spec["observation"].shape[-1], hidden_size=64, batch_first=True)
            >>> lstm_module = LSTMModule(lstm=lstm, in_keys=["observation", "hidden0", "hidden1"], out_keys=["intermediate", ("next", "hidden0"), ("next", "hidden1")])
            >>> mlp = MLP(num_cells=[64], out_features=1)
            >>> # building two policies with different behaviours:
            >>> policy_inference = Seq(lstm_module, Mod(mlp, in_keys=["intermediate"], out_keys=["action"]))
            >>> policy_training = Seq(lstm_module.set_recurrent_mode(True), Mod(mlp, in_keys=["intermediate"], out_keys=["action"]))
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
        if mode is self._recurrent_mode:
            return self
        out = LSTMModule(lstm=self.lstm, in_keys=self.in_keys, out_keys=self.out_keys)
        out._recurrent_mode = mode
        return out

    def forward(self, tensordict: TensorDictBase):
        # we want to get an error if the value input is missing, but not the hidden states
        defaults = [NO_DEFAULT, None, None]
        shape = tensordict.shape
        tensordict_shaped = tensordict
        if self.recurrent_mode:
            # if less than 2 dims, unsqueeze
            ndim = tensordict_shaped.get(self.in_keys[0]).ndim
            while ndim < 3:
                tensordict_shaped = tensordict_shaped.unsqueeze(0)
                ndim += 1
            if ndim > 3:
                dims_to_flatten = ndim - 3
                # we assume that the tensordict can be flattened like this
                nelts = prod(tensordict_shaped.shape[: dims_to_flatten + 1])
                tensordict_shaped = tensordict_shaped.apply(
                    lambda value: value.flatten(0, dims_to_flatten),
                    batch_size=[nelts, tensordict_shaped.shape[-1]],
                )
        else:
            tensordict_shaped = tensordict.reshape(-1).unsqueeze(-1)

        is_init = tensordict_shaped.get("is_init").squeeze(-1)
        splits = None
        if self.recurrent_mode and is_init[..., 1:].any():
            # if we have consecutive trajectories, things get a little more complicated
            # we have a tensordict of shape [B, T]
            # we will split / pad things such that we get a tensordict of shape
            # [N, T'] where T' <= T and N >= B is the new batch size, such that
            # each index of N is an independent trajectory. We'll need to keep
            # track of the indices though, as we want to put things back together in the end.
            splits = _get_num_per_traj_init(is_init)
            tensordict_shaped_shape = tensordict_shaped.shape
            tensordict_shaped = _split_and_pad_sequence(
                tensordict_shaped.select(*self.in_keys, strict=False), splits
            )
            is_init = tensordict_shaped.get("is_init").squeeze(-1)

        value, hidden0, hidden1 = (
            tensordict_shaped.get(key, default)
            for key, default in zip(self.in_keys, defaults)
        )
        batch, steps = value.shape[:2]
        device = value.device
        dtype = value.dtype
        # packed sequences do not help to get the accurate last hidden values
        # if splits is not None:
        #     value = torch.nn.utils.rnn.pack_padded_sequence(value, splits, batch_first=True)
        if is_init.any() and hidden0 is not None:
            hidden0[is_init] = 0
            hidden1[is_init] = 0
        val, hidden0, hidden1 = self._lstm(
            value, batch, steps, device, dtype, hidden0, hidden1
        )
        tensordict_shaped.set(self.out_keys[0], val)
        tensordict_shaped.set(self.out_keys[1], hidden0)
        tensordict_shaped.set(self.out_keys[2], hidden1)
        if splits is not None:
            # let's recover our original shape
            tensordict_shaped = _inv_pad_sequence(tensordict_shaped, splits).reshape(
                tensordict_shaped_shape
            )

        if shape != tensordict_shaped.shape or tensordict_shaped is not tensordict:
            tensordict.update(tensordict_shaped.reshape(shape))
        return tensordict

    def _lstm(
        self,
        input: torch.Tensor,
        batch,
        steps,
        device,
        dtype,
        hidden0_in: Optional[torch.Tensor] = None,
        hidden1_in: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if not self.recurrent_mode and steps != 1:
            raise ValueError("Expected a single step")

        if hidden1_in is None and hidden0_in is None:
            shape = (batch, steps)
            hidden0_in, hidden1_in = [
                torch.zeros(
                    *shape,
                    self.lstm.num_layers,
                    self.lstm.hidden_size,
                    device=device,
                    dtype=dtype,
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
                [torch.zeros_like(out[i]) for _ in range(steps - 1)] + [out[i]],
                1,
            )
        return tuple(out)


class GRUModule(ModuleBase):
    """An embedder for an GRU module.

    This class adds the following functionality to :class:`torch.nn.GRU`:

    - Compatibility with TensorDict: the hidden states are reshaped to match
      the tensordict batch size.
    - Optional multi-step execution: with torch.nn, one has to choose between
      :class:`torch.nn.GRUCell` and :class:`torch.nn.GRU`, the former being
      compatible with single step inputs and the latter being compatible with
      multi-step. This class enables both usages.


    After construction, the module is *not* set in recurrent mode, ie. it will
    expect single steps inputs.

    If in recurrent mode, it is expected that the last dimension of the tensordict
    marks the number of steps. There is no constrain on the dimensionality of the
    tensordict (except that it must be greater than one for temporal inputs).

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two GRUs together to form a `stacked GRU`,
            with the second GRU taking in outputs of the first GRU and
            computing the final results. Default: 1
        bias: If ``False``, then the layer does not use bias weights.
            Default: ``True``
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            GRU layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        proj_size: If ``> 0``, will use GRU with projections of corresponding size. Default: 0

    Keyword Args:
        in_key (str or tuple of str): the input key of the module. Exclusive use
            with ``in_keys``. If provided, the recurrent keys are assumed to be
            ["recurrent_state"] and the ``in_key`` will be
            appended before this.
        in_keys (list of str): a pair of strings corresponding to the input value and recurrent entry.
            Exclusive with ``in_key``.
        out_key (str or tuple of str): the output key of the module. Exclusive use
            with ``out_keys``. If provided, the recurrent keys are assumed to be
            [("recurrent_state")] and the ``out_key`` will be
            appended before these.
        out_keys (list of str): a pair of strings corresponding to the output value,
            first and second hidden key.
            .. note::
              For a better integration with TorchRL's environments, the best naming
              for the output hidden key is ``("next", <custom_key>)``, such
              that the hidden values are passed from step to step during a rollout.
        device (torch.device or compatible): the device of the module.
        gru (torch.nn.GRU, optional): a GRU instance to be wrapped.
            Exclusive with other nn.GRU arguments.

    Attributes:
        recurrent_mode: Returns the recurrent mode of the module.

    Methods:
        set_recurrent_mode: controls whether the module should be executed in
            recurrent mode.

    Examples:
        >>> from torchrl.envs import TransformedEnv, InitTracker
        >>> from torchrl.envs import GymEnv
        >>> from torchrl.modules import MLP
        >>> from torch import nn
        >>> from tensordict.nn import TensorDictSequential as Seq, TensorDictModule as Mod
        >>> env = TransformedEnv(GymEnv("Pendulum-v1"), InitTracker())
        >>> gru_module = GRUModule(
        ...     input_size=env.observation_spec["observation"].shape[-1],
        ...     hidden_size=64,
        ...     in_keys=["observation", "rs"],
        ...     out_keys=["intermediate", ("next", "rs")])
        >>> mlp = MLP(num_cells=[64], out_features=1)
        >>> policy = Seq(gru_module, Mod(mlp, in_keys=["intermediate"], out_keys=["action"]))
        >>> policy(env.reset())
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                intermediate: Tensor(shape=torch.Size([64]), device=cpu, dtype=torch.float32, is_shared=False),
                is_init: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        rs: Tensor(shape=torch.Size([1, 64]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([]),
                    device=cpu,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([]),
            device=cpu,
            is_shared=False)
        >>> gru_module_training = gru_module.set_recurrent_mode()
        >>> policy_training = Seq(gru_module, Mod(mlp, in_keys=["intermediate"], out_keys=["action"]))
        >>> traj_td = env.rollout(3) # some random temporal data
        >>> traj_td = policy_training(traj_td)
        >>> print(traj_td)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                intermediate: Tensor(shape=torch.Size([3, 64]), device=cpu, dtype=torch.float32, is_shared=False),
                is_init: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        is_init: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([3, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        rs: Tensor(shape=torch.Size([3, 1, 64]), device=cpu, dtype=torch.float32, is_shared=False),
                        terminated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([3]),
                    device=cpu,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([3, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                terminated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([3]),
            device=cpu,
            is_shared=False)

    """

    DEFAULT_IN_KEYS = ["recurrent_state"]
    DEFAULT_OUT_KEYS = [("next", "recurrent_state")]

    def __init__(
        self,
        input_size: int = None,
        hidden_size: int = None,
        num_layers: int = 1,
        bias: bool = True,
        batch_first=True,
        dropout=0,
        bidirectional=False,
        *,
        in_key=None,
        in_keys=None,
        out_key=None,
        out_keys=None,
        device=None,
        gru=None,
    ):
        super().__init__()
        if gru is not None:
            if not gru.batch_first:
                raise ValueError("The input gru must have batch_first=True.")
            if gru.bidirectional:
                raise ValueError("The input gru cannot be bidirectional.")
            if input_size is not None or hidden_size is not None:
                raise ValueError(
                    "An GRU instance cannot be passed along with class argument."
                )
        else:
            if not batch_first:
                raise ValueError("The input gru must have batch_first=True.")
            if bidirectional:
                raise ValueError("The input gru cannot be bidirectional.")
            gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bias=bias,
                dropout=dropout,
                device=device,
                batch_first=True,
                bidirectional=False,
            )
        if not ((in_key is None) ^ (in_keys is None)):
            raise ValueError(
                f"Either in_keys or in_key must be specified but not both or none. Got {in_keys} and {in_key} respectively."
            )
        elif in_key:
            in_keys = [in_key, *self.DEFAULT_IN_KEYS]

        if not ((out_key is None) ^ (out_keys is None)):
            raise ValueError(
                f"Either out_keys or out_key must be specified but not both or none. Got {out_keys} and {out_key} respectively."
            )
        elif out_key:
            out_keys = [out_key, *self.DEFAULT_OUT_KEYS]

        in_keys = unravel_key_list(in_keys)
        out_keys = unravel_key_list(out_keys)
        if not isinstance(in_keys, (tuple, list)) or (
            len(in_keys) != 2 and not (len(in_keys) == 3 and in_keys[-1] == "is_init")
        ):
            raise ValueError(
                f"GRUModule expects 3 inputs: a value, and two hidden states (and potentially an 'is_init' marker). Got in_keys {in_keys} instead."
            )
        if not isinstance(out_keys, (tuple, list)) or len(out_keys) != 2:
            raise ValueError(
                f"GRUModule expects 3 outputs: a value, and two hidden states. Got out_keys {out_keys} instead."
            )
        self.gru = gru
        if "is_init" not in in_keys:
            in_keys = in_keys + ["is_init"]
        self.in_keys = in_keys
        self.out_keys = out_keys
        self._recurrent_mode = False

    def make_tensordict_primer(self):
        from torchrl.envs import TensorDictPrimer

        def make_tuple(key):
            if isinstance(key, tuple):
                return key
            return (key,)

        out_key1 = make_tuple(self.out_keys[1])
        in_key1 = make_tuple(self.in_keys[1])
        if out_key1 != ("next", *in_key1):
            raise RuntimeError(
                "make_tensordict_primer is supposed to work with in_keys/out_keys that "
                "have compatible names, ie. the out_keys should be named after ('next', <in_key>). Got "
                f"in_keys={self.in_keys} and out_keys={self.out_keys} instead."
            )
        return TensorDictPrimer(
            {
                in_key1: UnboundedContinuousTensorSpec(
                    shape=(self.gru.num_layers, self.gru.hidden_size)
                ),
            }
        )

    @property
    def recurrent_mode(self):
        return self._recurrent_mode

    @recurrent_mode.setter
    def recurrent_mode(self, value):
        raise RuntimeError(
            "recurrent_mode cannot be changed in-place. Call `module.set"
        )

    @property
    def temporal_mode(self):
        warnings.warn(
            "temporal_mode is deprecated, use recurrent_mode instead.",
            category=DeprecationWarning,
        )
        return self.recurrent_mode

    def set_recurrent_mode(self, mode: bool = True):
        """Returns a new copy of the module that shares the same gru model but with a different ``recurrent_mode`` attribute (if it differs).

        A copy is created such that the module can be used with divergent behaviour
        in various parts of the code (inference vs training):

        Examples:
            >>> from torchrl.envs import GymEnv, TransformedEnv, InitTracker, step_mdp
            >>> from torchrl.modules import MLP
            >>> from tensordict import TensorDict
            >>> from torch import nn
            >>> from tensordict.nn import TensorDictSequential as Seq, TensorDictModule as Mod
            >>> env = TransformedEnv(GymEnv("Pendulum-v1"), InitTracker())
            >>> gru = nn.GRU(input_size=env.observation_spec["observation"].shape[-1], hidden_size=64, batch_first=True)
            >>> gru_module = GRUModule(gru=gru, in_keys=["observation", "hidden"], out_keys=["intermediate", ("next", "hidden")])
            >>> mlp = MLP(num_cells=[64], out_features=1)
            >>> # building two policies with different behaviours:
            >>> policy_inference = Seq(gru_module, Mod(mlp, in_keys=["intermediate"], out_keys=["action"]))
            >>> policy_training = Seq(gru_module.set_recurrent_mode(True), Mod(mlp, in_keys=["intermediate"], out_keys=["action"]))
            >>> traj_td = env.rollout(3) # some random temporal data
            >>> traj_td = policy_training(traj_td)
            >>> # let's check that both return the same results
            >>> td_inf = TensorDict({}, traj_td.shape[:-1])
            >>> for td in traj_td.unbind(-1):
            ...     td_inf = td_inf.update(td.select("is_init", "observation", ("next", "observation")))
            ...     td_inf = policy_inference(td_inf)
            ...     td_inf = step_mdp(td_inf)
            ...
            >>> torch.testing.assert_close(td_inf["hidden"], traj_td[..., -1]["next", "hidden"])
        """
        if mode is self._recurrent_mode:
            return self
        out = GRUModule(gru=self.gru, in_keys=self.in_keys, out_keys=self.out_keys)
        out._recurrent_mode = mode
        return out

    def forward(self, tensordict: TensorDictBase):
        # we want to get an error if the value input is missing, but not the hidden states
        defaults = [NO_DEFAULT, None]
        shape = tensordict.shape
        tensordict_shaped = tensordict
        if self.recurrent_mode:
            # if less than 2 dims, unsqueeze
            ndim = tensordict_shaped.get(self.in_keys[0]).ndim
            while ndim < 3:
                tensordict_shaped = tensordict_shaped.unsqueeze(0)
                ndim += 1
            if ndim > 3:
                dims_to_flatten = ndim - 3
                # we assume that the tensordict can be flattened like this
                nelts = prod(tensordict_shaped.shape[: dims_to_flatten + 1])
                tensordict_shaped = tensordict_shaped.apply(
                    lambda value: value.flatten(0, dims_to_flatten),
                    batch_size=[nelts, tensordict_shaped.shape[-1]],
                )
        else:
            tensordict_shaped = tensordict.reshape(-1).unsqueeze(-1)

        is_init = tensordict_shaped.get("is_init").squeeze(-1)
        splits = None
        if self.recurrent_mode and is_init[..., 1:].any():
            # if we have consecutive trajectories, things get a little more complicated
            # we have a tensordict of shape [B, T]
            # we will split / pad things such that we get a tensordict of shape
            # [N, T'] where T' <= T and N >= B is the new batch size, such that
            # each index of N is an independent trajectory. We'll need to keep
            # track of the indices though, as we want to put things back together in the end.
            splits = _get_num_per_traj_init(is_init)
            tensordict_shaped_shape = tensordict_shaped.shape
            tensordict_shaped = _split_and_pad_sequence(
                tensordict_shaped.select(*self.in_keys, strict=False), splits
            )
            is_init = tensordict_shaped.get("is_init").squeeze(-1)

        value, hidden = (
            tensordict_shaped.get(key, default)
            for key, default in zip(self.in_keys, defaults)
        )
        batch, steps = value.shape[:2]
        device = value.device
        dtype = value.dtype
        # packed sequences do not help to get the accurate last hidden values
        # if splits is not None:
        #     value = torch.nn.utils.rnn.pack_padded_sequence(value, splits, batch_first=True)
        if is_init.any() and hidden is not None:
            hidden[is_init] = 0
        val, hidden = self._gru(value, batch, steps, device, dtype, hidden)
        tensordict_shaped.set(self.out_keys[0], val)
        tensordict_shaped.set(self.out_keys[1], hidden)
        if splits is not None:
            # let's recover our original shape
            tensordict_shaped = _inv_pad_sequence(tensordict_shaped, splits).reshape(
                tensordict_shaped_shape
            )

        if shape != tensordict_shaped.shape or tensordict_shaped is not tensordict:
            tensordict.update(tensordict_shaped.reshape(shape))
        return tensordict

    def _gru(
        self,
        input: torch.Tensor,
        batch,
        steps,
        device,
        dtype,
        hidden_in: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if not self.recurrent_mode and steps != 1:
            raise ValueError("Expected a single step")

        if hidden_in is None:
            shape = (batch, steps)
            hidden_in = torch.zeros(
                *shape,
                self.gru.num_layers,
                self.gru.hidden_size,
                device=device,
                dtype=dtype,
            )

        # we only need the first hidden state
        _hidden_in = hidden_in[:, 0]
        hidden = _hidden_in.transpose(-3, -2).contiguous()

        y, hidden = self.gru(input, hidden)
        # dim 0 in hidden is num_layers, but that will conflict with tensordict
        hidden = hidden.transpose(0, 1)

        # we pad the hidden states with zero to make tensordict happy
        hidden = torch.stack(
            [torch.zeros_like(hidden) for _ in range(steps - 1)] + [hidden],
            1,
        )
        out = [y, hidden]
        return tuple(out)
