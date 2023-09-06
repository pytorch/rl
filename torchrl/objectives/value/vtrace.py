# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List, Optional, Union, Tuple

import torch
from tensordict.nn import (
    dispatch,
    is_functional,
    set_skip_existing,
    TensorDictModule,
    TensorDictModuleBase,
)
from tensordict.tensordict import TensorDictBase
from tensordict.utils import NestedKey
from torch import nn, Tensor
from torchrl.objectives.utils import hold_out_net
from torchrl.objectives.value.advantages import ValueEstimatorBase, _self_set_skip_existing, _self_set_grad_enabled, _call_value_nets
from torchrl.objectives.value.functional import _transpose_time, SHAPE_ERR


def _c_val(
    log_pi: torch.Tensor,
    log_mu: torch.Tensor,
    c: Union[float, torch.Tensor] = 1,
) -> torch.Tensor:
    return (log_pi - log_mu).clamp_max(math.log(c)).exp().unsqueeze(-1)


def _dv_val(
    rewards: torch.Tensor,
    vals: torch.Tensor,
    next_vals: torch.Tensor,
    gamma: Union[float, torch.Tensor],
    rho_bar: Union[float, torch.Tensor],
    log_pi: torch.Tensor,
    log_mu: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rho = _c_val(log_pi, log_mu, rho_bar)
    deltas = rho * (rewards + gamma * next_vals - vals)
    return deltas, rho


def _vtrace(
    rewards: torch.Tensor,
    vals: torch.Tensor,
    log_pi: torch.Tensor,
    log_mu: torch.Tensor,
    gamma: Union[torch.Tensor, float],
    rho_bar: Union[float, torch.Tensor] = 1.0,
    c_bar: Union[float, torch.Tensor] = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:

    T = vals.shape[1]
    if not isinstance(gamma, torch.Tensor):
        gamma = torch.full_like(vals, gamma)

    dv, rho = _dv_val(rewards, vals, gamma, rho_bar, log_pi, log_mu)
    c = _c_val(log_pi, log_mu, c_bar)

    v_out = []
    v_out.append(vals[:, -1] + dv[:, -1])
    for t in range(T - 2, -1, -1):
        _v_out = (
            vals[:, t] + dv[:, t] + gamma[:, t] * c[:, t] * (v_out[-1] - vals[:, t + 1])
        )
        v_out.append(_v_out)
    v_out = torch.stack(list(reversed(v_out)), 1)  # values
    return v_out, rho

# @_transpose_time
def vtrace_correction(
        gamma: float,
        log_pi: torch.Tensor,
        log_mu: torch.Tensor,
        state_value: torch.Tensor,
        next_state_value: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        rho_bar: Union[float, torch.Tensor] = 1.0,
        c_bar: Union[float, torch.Tensor] = 1.0,
        time_dim: int = -2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """V-Trace off-policy correction method.

    Refer to "IMPALA: Scalable Distributed Deep-RL with Importance Weighted  Actor-Learner Architectures"
    https://arxiv.org/abs/1802.01561 for more context.

    Args:
        gamma (scalar): exponential mean discount.
        log_pi (Tensor): log probability of taking actions in the environment.
        log_mu (Tensor): log probability of taking actions in the environment.
        state_value (Tensor): value function result with old_state input.
        next_state_value (Tensor): value function result with new_state input.
        reward (Tensor): reward of taking actions in the environment.
        done (Tensor): boolean flag for end of episode.
        rho_bar (Union[float, Tensor]): clipping parameter for importance weights.
        c_bar (Union[float, Tensor]): clipping parameter for importance weights.
        time_dim (int): dimension where the time is unrolled. Defaults to -2.

    All tensors (values, reward and done) must have shape
    ``[*Batch x TimeSteps x *F]``, with ``*F`` feature dimensions.

    """

    if not (next_state_value.shape == state_value.shape == reward.shape == done.shape):
        raise RuntimeError(SHAPE_ERR)

    dtype = next_state_value.dtype
    device = state_value.device

    delta, rho = _dv_val(reward, state_value, next_state_value, gamma, rho_bar, log_pi, log_mu)
    cs = _c_val(log_pi, log_mu, c_bar)

    not_done = (~done).int()
    *batch_size, time_steps, lastdim = not_done.shape
    acc = 0
    v_out = torch.empty(*batch_size, time_steps, lastdim, device=device, dtype=dtype)
    gnotdone = gamma * not_done
    import ipdb; ipdb.set_trace()  # TODO: Review!
    for t in reversed(range(time_steps)):
        import ipdb; ipdb.set_trace()  # TODO: Review!
        acc = delta[..., t, :] + (gnotdone[..., t, :] * acc * cs[t])
        v_out[..., t, :].copy_(acc + state_value[..., t, :])

    advantage = rho * (reward + gamma * v_out - state_value) # TODO: Review!

    return advantage, v_out


class VTrace(ValueEstimatorBase):
    """A class wrapper around V-Trace estimate functional.

    Refer to "IMPALA: Scalable Distributed Deep-RL with Importance Weighted  Actor-Learner Architectures"
    https://arxiv.org/abs/1802.01561 for more context.

    Args:
        gamma (scalar): exponential mean discount.
        value_network (TensorDictModule): value operator used to retrieve the value estimates.
        average_gae (bool): if ``True``, the resulting GAE values will be standardized.
            Default is ``False``.
        differentiable (bool, optional): if ``True``, gradients are propagated through
            the computation of the value function. Default is ``False``.

            .. note::
              The proper way to make the function call non-differentiable is to
              decorate it in a `torch.no_grad()` context manager/decorator or
              pass detached parameters for functional modules.

        # vectorized (bool, optional): whether to use the vectorized version of the
        #     lambda return. Default is `True`.

        skip_existing (bool, optional): if ``True``, the value network will skip
            modules which outputs are already present in the tensordict.
            Defaults to ``None``, ie. the value of :func:`tensordict.nn.skip_existing()`
            is not affected.
            Defaults to "state_value".
        advantage_key (str or tuple of str, optional): [Deprecated] the key of
            the advantage entry.  Defaults to ``"advantage"``.
        value_target_key (str or tuple of str, optional): [Deprecated] the key
            of the advantage entry.  Defaults to ``"value_target"``.
        value_key (str or tuple of str, optional): [Deprecated] the value key to
            read from the input tensordict.  Defaults to ``"state_value"``.

        # shifted (bool, optional): if ``True``, the value and next value are
        #     estimated with a single call to the value network. This is faster
        #     but is only valid whenever (1) the ``"next"`` value is shifted by
        #     only one time step (which is not the case with multi-step value
        #     estimation, for instance) and (2) when the parameters used at time
        #     ``t`` and ``t+1`` are identical (which is not the case when target
        #     parameters are to be used). Defaults to ``False``.

    VTrace will return an :obj:`"advantage"` entry containing the advantage value. It will also
    return a :obj:`"value_target"` entry with the V-Trace target value.

    # Finally, if :obj:`gradient_mode` is ``True``,
    # an additional and differentiable :obj:`"value_error"` entry will be returned,
    # which simple represents the difference between the return and the value network
    # output (i.e. an additional distance loss should be applied to that signed value).

    # .. note::
    #   As other advantage functions do, if the ``value_key`` is already present
    #   in the input tensordict, the VTrace module will ignore the calls to the value
    #   network (if any) and use the provided value instead.

    """

    def __init__(
            self,
            *,
            gamma: Union[float, torch.Tensor],
            rho_bar: Union[float, torch.Tensor] = 1.0,
            c_bar: Union[float, torch.Tensor] = 1.0,
            actor_network: TensorDictModule = None,
            value_network: TensorDictModule,
            average_gae: bool = False,
            differentiable: bool = False,
            vectorized: bool = False,  # TODO: Review!
            skip_existing: Optional[bool] = None,
            advantage_key: NestedKey = None,
            value_target_key: NestedKey = None,
            value_key: NestedKey = None,
            shifted: bool = False,
    ):
        super().__init__(
            shifted=shifted,
            value_network=value_network,
            differentiable=differentiable,
            advantage_key=advantage_key,
            value_target_key=value_target_key,
            value_key=value_key,
            skip_existing=skip_existing,
        )
        try:
            device = next(value_network.parameters()).device
        except (AttributeError, StopIteration):
            device = torch.device("cpu")
        self.register_buffer("gamma", torch.tensor(gamma, device=device))
        self.register_buffer("rho_bar", torch.tensor(rho_bar, device=device))
        self.register_buffer("c_bar", torch.tensor(c_bar, device=device))
        self.average_gae = average_gae
        self.vectorized = vectorized
        self.actor_network = actor_network

    @_self_set_skip_existing
    @_self_set_grad_enabled
    @dispatch
    def forward(
            self,
            tensordict: TensorDictBase,
            *unused_args,
            params: Optional[List[Tensor]] = None,
            target_params: Optional[List[Tensor]] = None,
    ) -> TensorDictBase:
        """Computes the V-Trace correction given the data in tensordict.

        If a functional module is provided, a nested TensorDict containing the parameters
        (and if relevant the target parameters) can be passed to the module.

        Args:
            tensordict (TensorDictBase): A TensorDict containing the data
                (an observation key, "action", "reward", "done" and "next" tensordict state
                as returned by the environment) necessary to compute the value estimates and the GAE.
                The data passed to this module should be structured as :obj:`[*B, T, F]` where :obj:`B` are
                the batch size, :obj:`T` the time dimension and :obj:`F` the feature dimension(s).
            params (TensorDictBase, optional): A nested TensorDict containing the params
                to be passed to the functional value network module.
            target_params (TensorDictBase, optional): A nested TensorDict containing the
                target params to be passed to the functional value network module.

        Returns:
            An updated TensorDict with an advantage and a value_error keys as defined in the constructor.

        Examples:
            >>> from tensordict import TensorDict
            >>> value_net = TensorDictModule(
            ...     nn.Linear(3, 1), in_keys=["obs"], out_keys=["state_value"]
            ... )
            >>> module = VTrace(
            ...     gamma=0.98,
            ...     value_network=value_net,
            ...     differentiable=False,
            ... )
            >>> obs, next_obs = torch.randn(2, 1, 10, 3)
            >>> reward = torch.randn(1, 10, 1)
            >>> done = torch.zeros(1, 10, 1, dtype=torch.bool)
            >>> tensordict = TensorDict({"obs": obs, "next": {"obs": next_obs}, "done": done, "reward": reward}, [1, 10])
            >>> _ = module(tensordict)
            >>> assert "advantage" in tensordict.keys()

        The module supports non-tensordict (i.e. unpacked tensordict) inputs too:

        Examples:
            >>> value_net = TensorDictModule(
            ...     nn.Linear(3, 1), in_keys=["obs"], out_keys=["state_value"]
            ... )
            >>> module = VTrace(
            ...     gamma=0.98,
            ...     value_network=value_net,
            ...     differentiable=False,
            ... )
            >>> obs, next_obs = torch.randn(2, 1, 10, 3)
            >>> reward = torch.randn(1, 10, 1)
            >>> done = torch.zeros(1, 10, 1, dtype=torch.bool)
            >>> advantage, value_target = module(obs=obs, reward=reward, done=done, next_obs=next_obs)

        """
        if tensordict.batch_dims < 1:
            raise RuntimeError(
                "Expected input tensordict to have at least one dimensions, got "
                f"tensordict.batch_size = {tensordict.batch_size}"
            )
        reward = tensordict.get(("next", self.tensor_keys.reward))
        device = reward.device
        gamma = self.gamma.to(device)
        steps_to_next_obs = tensordict.get(self.tensor_keys.steps_to_next_obs, None)
        if steps_to_next_obs is not None:
            gamma = gamma ** steps_to_next_obs.view_as(reward)

        # Make sure we have the value and next value
        if self.value_network is not None:
            if params is not None:
                params = params.detach()
                if target_params is None:
                    target_params = params.clone(False)
            with hold_out_net(self.value_network):
                # we may still need to pass gradient, but we don't want to assign grads to
                # value net params
                value, next_value = _call_value_nets(
                    value_net=self.value_network,
                    data=tensordict,
                    params=params,
                    next_params=target_params,
                    single_call=self.shifted,
                    value_key=self.tensor_keys.value,
                    detach_next=True,
                )
        else:
            value = tensordict.get(self.tensor_keys.value)
            next_value = tensordict.get(("next", self.tensor_keys.value))

        # Make sure we have the new log prob and the old log prob
        if self.actor_network is not None:
            import ipdb; ipdb.set_trace()
            raise NotImplementedError
        else:
            # log_pi = tensordict.get(self.tensor_keys.log_pi)  # new / local log prob
            log_pi = tensordict.get("sample_log_prob")
            # log_mu = tensordict.get(self.tensor_keys.log_mu)  # old / distributed log prob
            log_mu = tensordict.get("sample_log_prob")

        done = tensordict.get(("next", self.tensor_keys.done))
        if self.vectorized:
            raise NotImplementedError
        else:
            adv, value_target = vtrace_correction(
                gamma,
                log_pi,
                log_mu,
                value,
                next_value,
                reward,
                done,
                rho_bar=self.rho_bar,
                c_bar=self.c_bar,
                time_dim=tensordict.ndim - 1,
            )

        if self.average_gae:
            loc = adv.mean()
            scale = adv.std().clamp_min(1e-4)
            adv = adv - loc
            adv = adv / scale

        tensordict.set(self.tensor_keys.advantage, adv)
        tensordict.set(self.tensor_keys.value_target, value_target)

        return tensordict

