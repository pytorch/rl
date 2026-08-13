# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import Literal

import torch
from tensordict import TensorDictBase
from tensordict.nn import ProbabilisticTensorDictSequential, TensorDictModule
from torch import Tensor

from torchrl.data.tensor_specs import TensorSpec
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives.sac import compute_rsample_log_prob, SACLoss
from torchrl.objectives.utils import ValueEstimators


class TQCLoss(SACLoss):
    r"""Truncated Quantile Critics loss.

    TQC critics predict return quantiles instead of one value. Bellman targets
    pool every target critic's atoms, discard the highest from the shared pool,
    and train each critic on what remains. The actor uses the full mixture.

    See `Kuznetsov et al. (2020), Controlling Overestimation Bias with
    Truncated Mixture of Continuous Distributional Quantile Critics
    <https://arxiv.org/abs/2005.04269>`_.

    Args:
        actor_network (ProbabilisticTensorDictSequential): policy used to
            sample actions and score their entropy.
        qvalue_network (TensorDictModule or list of TensorDictModule):
            critic whose last output dimension holds return atoms. One module
            is copied across the ensemble; a list supplies each critic, as in
            :class:`~torchrl.objectives.SACLoss`.

    Keyword Args:
        num_qvalue_nets (int, optional): ensemble size for one critic module.
            Defaults to ``5``.
        top_quantiles_to_drop_per_net (int, optional): upper atoms discarded per
            critic from the pooled target. The total is this value times the
            ensemble size. Defaults to ``2``.
        alpha_init (float, optional): initial entropy temperature. Defaults to
            ``1.0``.
        min_alpha (float, optional): temperature floor. Defaults to ``None``.
        max_alpha (float, optional): temperature ceiling. Defaults to ``None``.
        action_spec (TensorSpec, optional): action domain for automatic target
            entropy. Defaults to the actor's spec.
        fixed_alpha (bool, optional): disable temperature learning. Defaults to
            ``False``.
        target_entropy (float or ``"auto"``, optional): entropy target.
            Defaults to ``"auto"``.
        delay_qvalue (bool, optional): maintain delayed target critics.
            Defaults to ``True``.
        separate_losses (bool, optional): exclude shared actor parameters from
            critic training. Defaults to ``False``.
        reduction (str, optional): ``"none"``, ``"mean"``, or
            ``"sum"``. Defaults to ``"mean"``.
        deactivate_vmap (bool, optional): loop over critics instead of using
            ``vmap``. Defaults to ``False``.
        skip_done_states (bool, optional): skip terminal next-state evaluation.
            Defaults to ``False``.
        use_prioritized_weights (bool or ``"auto"``, optional): use replay
            weights when present. Defaults to ``"auto"``.
        scalar_output_mode (str, optional): scalar handling when
            ``reduction="none"``. See :class:`~torchrl.objectives.SACLoss`.

    .. note::
        Among TorchRL's built-in value estimators, TQC supports only
        :class:`~torchrl.objectives.value.TD0Estimator`. Reward and termination
        leaves are explicitly expanded to the target-atom shape so the value
        estimator retains its strict shape checks.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import NormalParamExtractor, TensorDictModule
        >>> from torchrl.data import Bounded
        >>> from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
        >>> from torchrl.modules.distributions import TanhNormal
        >>> from torchrl.objectives import SoftUpdate, TQCLoss
        >>> n_obs, n_act, n_quantiles = 3, 2, 8
        >>> action_spec = Bounded(-1, 1, (n_act,))
        >>> actor_net = nn.Sequential(
        ...     nn.Linear(n_obs, 2 * n_act), NormalParamExtractor()
        ... )
        >>> actor = ProbabilisticActor(
        ...     TensorDictModule(
        ...         actor_net,
        ...         in_keys=["observation"],
        ...         out_keys=["loc", "scale"],
        ...     ),
        ...     in_keys=["loc", "scale"],
        ...     spec=action_spec,
        ...     distribution_class=TanhNormal,
        ... )
        >>> critic = ValueOperator(
        ...     MLP(
        ...         in_features=n_obs + n_act,
        ...         out_features=n_quantiles,
        ...         num_cells=[],
        ...     ),
        ...     in_keys=["observation", "action"],
        ... )
        >>> loss = TQCLoss(actor, critic, num_qvalue_nets=2)
        >>> loss.make_value_estimator(gamma=0.99)
        >>> target_updater = SoftUpdate(loss, eps=0.995)
        >>> batch = TensorDict(
        ...     {
        ...         "observation": torch.randn(4, n_obs),
        ...         "action": action_spec.rand((4,)),
        ...         "next": {
        ...             "observation": torch.randn(4, n_obs),
        ...             "reward": torch.randn(4, 1),
        ...             "done": torch.zeros(4, 1, dtype=torch.bool),
        ...             "terminated": torch.zeros(4, 1, dtype=torch.bool),
        ...         },
        ...     },
        ...     batch_size=[4],
        ... )
        >>> output = loss(batch)
        >>> output.get("loss_qvalue").shape
        torch.Size([])
    """

    SUPPORTED_VALUE_ESTIMATORS = (ValueEstimators.TD0,)

    def __init__(
        self,
        actor_network: ProbabilisticTensorDictSequential,
        qvalue_network: TensorDictModule | list[TensorDictModule],
        *,
        num_qvalue_nets: int = 5,
        top_quantiles_to_drop_per_net: int = 2,
        alpha_init: float = 1.0,
        min_alpha: float | None = None,
        max_alpha: float | None = None,
        action_spec: TensorSpec | None = None,
        fixed_alpha: bool = False,
        target_entropy: Literal["auto"] | float = "auto",
        delay_qvalue: bool = True,
        separate_losses: bool = False,
        reduction: Literal["none", "mean", "sum"] | None = None,
        deactivate_vmap: bool = False,
        skip_done_states: bool = False,
        use_prioritized_weights: Literal["auto"] | bool = "auto",
        scalar_output_mode: Literal["exclude", "non_tensor"] | None = None,
    ) -> None:
        if num_qvalue_nets < 1:
            raise ValueError("num_qvalue_nets must be greater than zero.")
        if top_quantiles_to_drop_per_net < 0:
            raise ValueError(
                "top_quantiles_to_drop_per_net must be greater than or equal to zero."
            )
        self.top_quantiles_to_drop_per_net = top_quantiles_to_drop_per_net
        super().__init__(
            actor_network=actor_network,
            qvalue_network=qvalue_network,
            num_qvalue_nets=num_qvalue_nets,
            loss_function="smooth_l1",
            alpha_init=alpha_init,
            min_alpha=min_alpha,
            max_alpha=max_alpha,
            action_spec=action_spec,
            fixed_alpha=fixed_alpha,
            target_entropy=target_entropy,
            delay_actor=False,
            delay_qvalue=delay_qvalue,
            separate_losses=separate_losses,
            reduction=reduction,
            deactivate_vmap=deactivate_vmap,
            skip_done_states=skip_done_states,
            use_prioritized_weights=use_prioritized_weights,
            scalar_output_mode=scalar_output_mode,
        )

    def actor_loss(
        self, tensordict: TensorDictBase
    ) -> tuple[Tensor, dict[str, Tensor]]:
        weights = self._maybe_get_priority_weight(tensordict)
        with (
            set_exploration_type(ExplorationType.RANDOM),
            self.actor_network_params.to_module(
                self.actor_network, preserve_module_state=False
            ),
        ):
            distribution = self.actor_network.get_dist(tensordict)
            action, log_prob = compute_rsample_log_prob(distribution)

        critic_input = tensordict.select(*self.qvalue_network.in_keys, strict=False)
        critic_input.set(self.tensor_keys.action, action)
        critic_output = self._vmap_qnetworkN0(
            critic_input, self._cached_detached_qvalue_params
        )
        quantiles = critic_output.get(self.tensor_keys.state_action_value)
        expected_ndim = tensordict.ndim + 2
        if quantiles.ndim != expected_ndim:
            raise RuntimeError(
                "The TQC critic must output one quantile dimension after the "
                f"TensorDict batch dimensions, but got shape {quantiles.shape}."
            )
        # Policy gradients use the whole mixture; only Bellman targets are cut.
        qvalue = quantiles.mean(dim=(0, -1))
        if log_prob.shape != qvalue.shape:
            raise RuntimeError(
                f"Actor log-probability and critic value shapes differ: "
                f"{log_prob.shape} and {qvalue.shape}."
            )
        loss_actor = self._alpha * log_prob - qvalue
        loss_actor = self._reduce_loss(
            loss_actor, tensordict=tensordict, weights=weights
        )
        return loss_actor, {"log_prob": log_prob.detach()}

    def compute_target(self, tensordict: TensorDictBase) -> Tensor:
        steps_key = self.value_estimator.tensor_keys.steps_to_next_obs
        target_tensordict = tensordict.select("next", steps_key, strict=False).clone()
        with (
            torch.no_grad(),
            set_exploration_type(ExplorationType.RANDOM),
            self.actor_network_params.to_module(
                self.actor_network, preserve_module_state=False
            ),
        ):
            next_tensordict = target_tensordict.get("next")
            selection = None
            selected_tensordict = next_tensordict
            if self.skip_done_states:
                terminated = next_tensordict.get(self.tensor_keys.terminated)
                if terminated.any():
                    selection = ~terminated.squeeze(-1)
                    selected_tensordict = next_tensordict[selection]
            distribution = self.actor_network.get_dist(selected_tensordict)
            if selected_tensordict.batch_size.numel():
                action, log_prob = compute_rsample_log_prob(distribution)
            else:
                action = distribution.rsample()
                log_prob = None
            selected_tensordict.set(self.tensor_keys.action, action)
            critic_output = self._vmap_qnetworkN0(
                selected_tensordict, self.target_qvalue_network_params
            )
            quantiles = critic_output.get(self.tensor_keys.state_action_value)
            expected_ndim = selected_tensordict.ndim + 2
            if quantiles.ndim != expected_ndim:
                raise RuntimeError(
                    "The TQC critic must output one quantile dimension after the "
                    f"TensorDict batch dimensions, but got shape {quantiles.shape}."
                )
            if log_prob is None:
                log_prob = quantiles.new_zeros(quantiles.shape[1:-1])
            # Pool critics before trimming; trimming each critic is a different method.
            quantiles = quantiles.movedim(0, -2).flatten(-2)
            quantiles_to_drop = (
                self.top_quantiles_to_drop_per_net * self.num_qvalue_nets
            )
            quantiles_to_keep = quantiles.shape[-1] - quantiles_to_drop
            if quantiles_to_keep < 1:
                raise ValueError(
                    "top_quantiles_to_drop_per_net must be smaller than the "
                    f"critic's number of quantiles ({quantiles.shape[-1] // self.num_qvalue_nets})."
                )
            quantiles = torch.sort(quantiles, dim=-1).values[..., :quantiles_to_keep]
            if log_prob.shape != quantiles.shape[:-1]:
                raise RuntimeError(
                    f"Actor log-probability and target critic batch shapes differ: "
                    f"{log_prob.shape} and {quantiles.shape[:-1]}."
                )
            next_value = quantiles - self._alpha * log_prob.unsqueeze(-1)
            if selection is not None:
                full_shape = (*tensordict.batch_size, next_value.shape[-1])
                selection = selection.unsqueeze(-1).expand(full_shape)
                next_value = next_value.new_zeros(full_shape).masked_scatter_(
                    selection, next_value
                )

            # TD0 checks shapes strictly, so every retained atom gets its own entry.
            for key in (
                self.tensor_keys.reward,
                self.tensor_keys.terminated,
            ):
                nested_key = ("next", *key) if isinstance(key, tuple) else ("next", key)
                value = target_tensordict.get(nested_key)
                while value.ndim < next_value.ndim:
                    value = value.unsqueeze(-1)
                target_tensordict.set(nested_key, value.expand_as(next_value))
            steps = target_tensordict.get(steps_key, None)
            if steps is not None:
                while steps.ndim < next_value.ndim:
                    steps = steps.unsqueeze(-1)
                target_tensordict.set(steps_key, steps.expand_as(next_value))
            return self.value_estimator.value_estimate(
                target_tensordict, next_value=next_value
            )

    def qvalue_v2_loss(
        self, tensordict: TensorDictBase
    ) -> tuple[Tensor, dict[str, Tensor]]:
        weights = self._maybe_get_priority_weight(tensordict)
        target_quantiles = self.compute_target(tensordict)
        critic_input = tensordict.select(*self.qvalue_network.in_keys, strict=False)
        critic_output = self._vmap_qnetworkN0(critic_input, self.qvalue_network_params)
        quantiles = critic_output.get(self.tensor_keys.state_action_value)
        expected_ndim = tensordict.ndim + 2
        if quantiles.ndim != expected_ndim:
            raise RuntimeError(
                "The TQC critic must output one quantile dimension after the "
                f"TensorDict batch dimensions, but got shape {quantiles.shape}."
            )
        if target_quantiles.shape[:-1] != quantiles.shape[1:-1]:
            raise RuntimeError(
                f"Target and predicted critic batch shapes differ: "
                f"{target_quantiles.shape[:-1]} and {quantiles.shape[1:-1]}."
            )

        # Regress every predicted quantile against every retained target atom.
        pairwise_delta = target_quantiles.unsqueeze(0).unsqueeze(
            -2
        ) - quantiles.unsqueeze(-1)
        absolute_delta = pairwise_delta.abs()
        huber_loss = torch.where(
            absolute_delta <= 1,
            0.5 * pairwise_delta.square(),
            absolute_delta - 0.5,
        )
        num_quantiles = quantiles.shape[-1]
        quantile_fractions = (
            torch.arange(num_quantiles, device=quantiles.device, dtype=quantiles.dtype)
            + 0.5
        ) / num_quantiles
        quantile_fractions = quantile_fractions.view(
            *([1] * (pairwise_delta.ndim - 2)), num_quantiles, 1
        )
        quantile_weights = (
            quantile_fractions - (pairwise_delta.detach() < 0).to(quantiles.dtype)
        ).abs()
        loss_qvalue = (quantile_weights * huber_loss).mean(dim=(-1, -2)).sum(0)
        loss_qvalue = self._reduce_loss(
            loss_qvalue, tensordict=tensordict, weights=weights
        )
        td_error = absolute_delta.detach().mean(dim=(-1, -2)).max(0).values
        return loss_qvalue, {"td_error": td_error}
