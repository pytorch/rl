# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import warnings

import torch
from packaging import version
from tensordict.nn import (
    NormalParamExtractor,
    TensorDictModule,
    TensorDictModuleBase,
    TensorDictSequential,
)
from torch import nn
from torch.nn import GRUCell

from torchrl._utils import _maybe_record_function_decorator
from torchrl.modules.models.models import MLP


class _Contiguous(nn.Module):
    """Helper module to ensure contiguous memory layout for torch.compile compatibility."""

    def forward(self, x):
        return x.contiguous()


UNSQUEEZE_RNN_INPUT = version.parse(torch.__version__) < version.parse("1.11")


class DreamerActor(nn.Module):
    """Dreamer actor network.

    This network is used to predict the action distribution given the
    the stochastic state and the deterministic belief at the current
    time step.
    It outputs the mean and the scale of the action distribution.

    Reference: https://arxiv.org/abs/1912.01603

    Args:
        out_features (int): Number of output features.
        depth (int, optional): Number of hidden layers.
            Defaults to 4.
        num_cells (int, optional): Number of hidden units per layer.
            Defaults to 200.
        activation_class (nn.Module, optional): Activation class.
            Defaults to nn.ELU.
        std_bias (:obj:`float`, optional): Bias of the softplus transform.
            Defaults to 5.0.
        std_min_val (:obj:`float`, optional): Minimum value of the standard deviation.
            Defaults to 1e-4.
        device (torch.device, optional): Device to create the module on.
            Defaults to None (uses default device).
    """

    def __init__(
        self,
        out_features,
        depth=4,
        num_cells=200,
        activation_class=nn.ELU,
        std_bias=5.0,
        std_min_val=1e-4,
        device=None,
    ):
        super().__init__()
        self.backbone = MLP(
            out_features=2 * out_features,
            depth=depth,
            num_cells=num_cells,
            activation_class=activation_class,
            device=device,
        )
        self.backbone.append(
            NormalParamExtractor(
                scale_mapping=f"biased_softplus_{std_bias}_{std_min_val}",
                # scale_mapping="relu",
            ),
        )

    def forward(self, state, belief):
        loc, scale = self.backbone(state, belief)
        return loc, scale


class ObsEncoder(nn.Module):
    """Observation encoder network.

    Takes a pixel observation and encodes it into a latent space.

    Reference: https://arxiv.org/abs/1803.10122

    Args:
        channels (int, optional): Number of hidden units in the first layer.
            Defaults to 32.
        num_layers (int, optional): Depth of the network. Defaults to 4.
        in_channels (int, optional): Number of input channels. If None, uses LazyConv2d.
            Defaults to None for backward compatibility.
        device (torch.device, optional): Device to create the module on.
            Defaults to None (uses default device).
    """

    def __init__(self, channels=32, num_layers=4, in_channels=None, depth=None, device=None):
        if depth is not None:
            warnings.warn(
                f"The depth argument in {type(self)} will soon be deprecated and "
                f"used for the depth of the network instead. Please use channels "
                f"for the layer size and num_layers for the depth until depth "
                f"replaces num_layers."
            )
            channels = depth
        if num_layers < 1:
            raise RuntimeError("num_layers cannot be smaller than 1.")
        super().__init__()
        # Use explicit Conv2d if in_channels provided, else LazyConv2d for backward compat
        if in_channels is not None:
            first_conv = nn.Conv2d(in_channels, channels, 4, stride=2, device=device)
        else:
            first_conv = nn.LazyConv2d(channels, 4, stride=2, device=device)
        # _Contiguous after ReLU ensures NCHW layout for torch.compile inductor compatibility
        layers = [
            first_conv,
            nn.ReLU(),
            _Contiguous(),
        ]
        k = 1
        for _ in range(1, num_layers):
            layers += [
                nn.Conv2d(channels * k, channels * (k * 2), 4, stride=2, device=device),
                nn.ReLU(),
                _Contiguous(),
            ]
            k = k * 2
        self.encoder = nn.Sequential(*layers)

    @_maybe_record_function_decorator("obs_encoder/forward")
    def forward(self, observation):
        *batch_sizes, C, H, W = observation.shape
        # Flatten batch dims -> encoder -> unflatten batch dims
        if batch_sizes:
            observation = observation.flatten(0, len(batch_sizes) - 1).contiguous()

        obs_encoded = self.encoder(observation)

        obs_encoded = obs_encoded.flatten(1)  # flatten spatial dims
        if batch_sizes:
            obs_encoded = obs_encoded.unflatten(0, batch_sizes).contiguous()

        return obs_encoded


class ObsDecoder(nn.Module):
    """Observation decoder network.

    Takes the deterministic state and the stochastic belief and decodes it into a pixel observation.

    Reference: https://arxiv.org/abs/1803.10122

    Args:
        channels (int, optional): Number of hidden units in the last layer.
            Defaults to 32.
        num_layers (int, optional): Depth of the network. Defaults to 4.
        kernel_sizes (int or list of int, optional): the kernel_size of each layer.
            Defaults to ``[5, 5, 6, 6]`` if num_layers if 4, else ``[5] * num_layers``.
        latent_dim (int, optional): Input dimension (state_dim + rnn_hidden_dim).
            If None, uses LazyLinear. Defaults to None for backward compatibility.
        device (torch.device, optional): Device to create the module on.
            Defaults to None (uses default device).
    """

    def __init__(self, channels=32, num_layers=4, kernel_sizes=None, latent_dim=None, depth=None, device=None):
        if depth is not None:
            warnings.warn(
                f"The depth argument in {type(self)} will soon be deprecated and "
                f"used for the depth of the network instead. Please use channels "
                f"for the layer size and num_layers for the depth until depth "
                f"replaces num_layers."
            )
            channels = depth
        if num_layers < 1:
            raise RuntimeError("num_layers cannot be smaller than 1.")

        super().__init__()
        # Use explicit Linear if latent_dim provided, else LazyLinear for backward compat
        linear_out = channels * 8 * 2 * 2
        if latent_dim is not None:
            first_linear = nn.Linear(latent_dim, linear_out, device=device)
        else:
            first_linear = nn.LazyLinear(linear_out, device=device)
        self.state_to_latent = nn.Sequential(
            first_linear,
            nn.ReLU(),
        )
        if kernel_sizes is None and num_layers == 4:
            kernel_sizes = [5, 5, 6, 6]
        elif kernel_sizes is None:
            kernel_sizes = 5
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * num_layers
        # _Contiguous after ReLU ensures NCHW layout for torch.compile inductor compatibility
        layers = [
            nn.ReLU(),
            _Contiguous(),
            nn.ConvTranspose2d(channels, 3, kernel_sizes[-1], stride=2, device=device),
        ]
        kernel_sizes = kernel_sizes[:-1]
        k = 1
        for j in range(1, num_layers):
            if j != num_layers - 1:
                layers = [
                    nn.ConvTranspose2d(
                        channels * k * 2, channels * k, kernel_sizes[-1], stride=2, device=device
                    ),
                ] + layers
                kernel_sizes = kernel_sizes[:-1]
                k = k * 2
                layers = [nn.ReLU(), _Contiguous()] + layers
            else:
                # Use explicit ConvTranspose2d - input is always channels * 8 from state_to_latent
                layers = [
                    nn.ConvTranspose2d(linear_out, channels * k, kernel_sizes[-1], stride=2, device=device)
                ] + layers

        self.decoder = nn.Sequential(*layers)
        self._depth = channels

    @_maybe_record_function_decorator("obs_decoder/forward")
    def forward(self, state, rnn_hidden):
        latent = self.state_to_latent(torch.cat([state, rnn_hidden], dim=-1))

        *batch_sizes, D = latent.shape
        # Flatten batch dims -> decoder -> unflatten batch dims
        if batch_sizes:
            latent = latent.flatten(0, len(batch_sizes) - 1)
        latent = latent.unsqueeze(-1).unsqueeze(-1).contiguous()  # add spatial dims

        obs_decoded = self.decoder(latent)

        if batch_sizes:
            obs_decoded = obs_decoded.unflatten(0, batch_sizes).contiguous()

        return obs_decoded


class RSSMRollout(TensorDictModuleBase):
    """Rollout the RSSM network.

    Given a set of encoded observations and actions, this module will rollout the RSSM network to compute all the intermediate
    states and beliefs.
    The previous posterior is used as the prior for the next time step.
    The forward method returns a stack of all intermediate states and beliefs.

    Reference: https://arxiv.org/abs/1811.04551

    Args:
        rssm_prior (TensorDictModule): Prior network.
        rssm_posterior (TensorDictModule): Posterior network.
        use_scan (bool, optional): If ``True``, uses ``torch._higher_order_ops.scan`` instead of
            an explicit for-loop. This enables end-to-end torch.compile without graph breaks.
            Note that scan is a prototype feature in PyTorch and may have limitations with
            autograd. Defaults to ``False``.


    """

    def __init__(
        self,
        rssm_prior: TensorDictModule,
        rssm_posterior: TensorDictModule,
        use_scan: bool = False,
        *,
        compile_step: bool = False,
        compile_backend: str = "inductor",
        compile_mode: str | None = "reduce-overhead",
    ):
        super().__init__()
        _module = TensorDictSequential(rssm_prior, rssm_posterior)
        self.in_keys = _module.in_keys
        self.out_keys = _module.out_keys
        self.rssm_prior = rssm_prior
        self.rssm_posterior = rssm_posterior
        self.use_scan = use_scan
        self.compile_step = compile_step
        self.compile_backend = compile_backend
        self.compile_mode = compile_mode

        # Optionally compile the per-timestep step function (TensorDict in/out).
        # This keeps the outer Python loop but reduces per-step overhead and can
        # fuse parts of the prior/posterior computation.
        self._loop_step = self._loop_step_impl
        if compile_step:
            # Compile the bound method (self is captured), resulting in a callable with
            # signature: (tensordict_t, prior_n, posterior_n) -> None.
            self._loop_step = torch.compile(
                self._loop_step_impl,
                backend=compile_backend,
                mode=compile_mode,
                fullgraph=True,
            )
        
        # Create a stable combine_fn for scan that won't trigger recompilation.
        # This must be created once and reused - recreating it causes recompiles.
        if use_scan:
            self._init_scan_combine_fn()

    def _init_scan_combine_fn(self):
        """Initialize the combine_fn for scan, caching it to avoid recompilation."""
        # Capture module references in the closure - these are stable across calls
        rssm_prior = self.rssm_prior.module
        rssm_posterior = self.rssm_posterior.module

        def _combine_fn_no_noise(carry, xs):
            """Combine function for scan without noise (standard training path)."""
            state, belief = carry
            action, enc_lat = xs

            prior_mean, prior_std, _, next_belief = rssm_prior(state, belief, action)
            posterior_mean, posterior_std, next_state = rssm_posterior(next_belief, enc_lat)

            next_carry = (next_state.clone(), next_belief.clone())
            outputs = (
                prior_mean.clone(),
                prior_std.clone(),
                next_belief.clone(),
                posterior_mean.clone(),
                posterior_std.clone(),
                next_state.clone(),
            )
            return next_carry, outputs

        self._scan_combine_fn_no_noise = _combine_fn_no_noise

    def _loop_step_impl(
        self,
        tensordict_t,
        next_td,
        prior_n: torch.Tensor,
        posterior_n: torch.Tensor,
    ):
        """Executes a single RSSM step for one timestep.

        Args:
            tensordict_t: Current timestep tensordict (mutated in-place).
            next_td: Next timestep tensordict to prepare, or None if last step.
            prior_n: Noise tensor for prior sampling.
            posterior_n: Noise tensor for posterior sampling.

        Returns:
            Tuple of (output_td, prepared_next_td). If `next_td` is None,
            `prepared_next_td` is also None.
        """
        # Set noise for TensorDictModule wrappers.
        tensordict_t.set("prior_noise", prior_n)
        tensordict_t.set("posterior_noise", posterior_n)

        # Prior: p(s_{t+1} | s_t, a_t, b_t)
        self.rssm_prior(tensordict_t)
        # Posterior: q(s_{t+1} | b_{t+1}, o_{t+1})
        self.rssm_posterior(tensordict_t)

        # Copy for output (preserves this timestep's state before we propagate).
        out = tensordict_t.copy()

        # Propagate state/belief to the next tensordict for the next iteration.
        if next_td is not None:
            next_state = tensordict_t.get(("next", "state"))
            next_belief = tensordict_t.get(("next", "belief"))
            next_td.set("state", next_state)
            next_td.set("belief", next_belief)

        return out, next_td

    @_maybe_record_function_decorator("rssm_rollout/forward")
    def forward(self, tensordict, *, prior_noise=None, posterior_noise=None):
        """Runs a rollout of simulated transitions in the latent space given a sequence of actions and environment observations.

        The rollout requires a belief and posterior state primer.

        At each step, two probability distributions are built and sampled from:

        - A prior distribution p(s_{t+1} | s_t, a_t, b_t) where b_t is a
            deterministic transform of the form b_t(s_{t-1}, a_{t-1}). The
            previous state s_t is sampled according to the posterior
            distribution (see below), creating a chain of posterior-to-priors
            that accumulates evidence to compute a prior distribution over
            the current event distribution:
            p(s_{t+1} s_t | o_t, a_t, s_{t-1}, a_{t-1}) = p(s_{t+1} | s_t, a_t, b_t) q(s_t | b_t, o_t)

        - A posterior distribution of the form q(s_{t+1} | b_{t+1}, o_{t+1})
            which amends to q(s_{t+1} | s_t, a_t, o_{t+1})

        Args:
            tensordict (TensorDict): Input tensordict with shape ``(*batch, time_steps)`` containing:
                - ``"state"``: Initial state tensor with shape ``(*batch, time_steps, state_dim)``.
                - ``"belief"``: Initial belief tensor with shape ``(*batch, time_steps, rnn_hidden_dim)``.
                - ``"action"``: Action tensor with shape ``(*batch, time_steps, action_dim)``.
                - ``("next", "encoded_latents")``: Encoded observations with shape ``(*batch, time_steps, obs_embed_dim)``.
            prior_noise (torch.Tensor, optional): Pre-computed noise tensor for prior sampling with shape
                ``(*batch, time_steps, state_dim)``. If ``None``, noise is sampled internally using
                ``torch.randn_like``. Passing noise enables deterministic execution for testing.
                Defaults to ``None``.
            posterior_noise (torch.Tensor, optional): Pre-computed noise tensor for posterior sampling with
                shape ``(*batch, time_steps, state_dim)``. If ``None``, noise is sampled internally using
                ``torch.randn_like``. Passing noise enables deterministic execution for testing.
                Defaults to ``None``.

        Returns:
            TensorDict: Output tensordict with the same batch size containing prior/posterior
                means, stds, beliefs, and states for each timestep.

        """
        if self.use_scan:
            return self._forward_scan(tensordict, prior_noise=prior_noise, posterior_noise=posterior_noise)
        return self._forward_loop(tensordict, prior_noise=prior_noise, posterior_noise=posterior_noise)

    def _forward_loop(self, tensordict, *, prior_noise=None, posterior_noise=None):
        """Forward pass using explicit for-loop with graph breaks.

        Args:
            tensordict (TensorDict): Input tensordict.
            prior_noise (torch.Tensor, optional): Noise tensor with shape ``(*batch, time_steps, state_dim)``.
            posterior_noise (torch.Tensor, optional): Noise tensor with shape ``(*batch, time_steps, state_dim)``.

        Note:
            When noise tensors are provided, they are inserted into the tensordict at
            ``"prior_noise"`` and ``"posterior_noise"`` keys. The TensorDictModule wrappers
            must have these keys in their ``in_keys`` (mapped to the ``noise`` argument)
            for the noise to be passed through. With ``strict=False`` (default), missing
            noise keys will pass ``None`` to the underlying module.
        """
        tensordict_out = []
        *batch, time_steps = tensordict.shape
        time_dim = len(batch)

        update_values = tensordict.exclude(*self.out_keys).unbind(-1)
        _tensordict = update_values[0]

        # Pre-create noise tensors if not provided, for consistency and to avoid per-timestep
        # `select(...)`/`randn(...)` overhead.
        state0 = _tensordict.get("state")
        state_dim = state0.shape[-1]
        if prior_noise is None:
            prior_noise = torch.randn(
                (*state0.shape[:-1], time_steps, state_dim),
                device=state0.device,
                dtype=state0.dtype,
            )
        if posterior_noise is None:
            posterior_noise = torch.randn(
                (*state0.shape[:-1], time_steps, state_dim),
                device=state0.device,
                dtype=state0.dtype,
            )

        # Unbind once: avoid calling select(time_dim, t) in the loop.
        prior_noise_steps = prior_noise.unbind(time_dim)
        posterior_noise_steps = posterior_noise.unbind(time_dim)

        for t in range(time_steps):
            # Get next timestep's base tensordict (or None if last step).
            next_td = update_values[t + 1] if t < time_steps - 1 else None

            # Run compiled step: prior, posterior, copy output, propagate state to next_td.
            out_td, _tensordict = self._loop_step(
                _tensordict, next_td, prior_noise_steps[t], posterior_noise_steps[t]
            )
            tensordict_out.append(out_td)

        out = torch.stack(tensordict_out, tensordict.ndim - 1)

        return out

    def _forward_scan(self, tensordict, *, prior_noise=None, posterior_noise=None):
        """Forward pass using torch._higher_order_ops.scan for torch.compile compatibility.

        This implementation avoids graph breaks by using the scan higher-order op,
        enabling end-to-end compilation of the RSSM rollout.

        Args:
            tensordict (TensorDict): Input tensordict.
            prior_noise (torch.Tensor, optional): Noise tensor with shape ``(*batch, time_steps, state_dim)``.
            posterior_noise (torch.Tensor, optional): Noise tensor with shape ``(*batch, time_steps, state_dim)``.
        """
        from tensordict import TensorDict
        from torch._higher_order_ops.scan import scan

        *batch, time_steps = tensordict.shape

        # Extract the raw modules from TensorDictModule wrappers
        rssm_prior_module = self.rssm_prior.module
        rssm_posterior_module = self.rssm_posterior.module

        # time_dim is where the time dimension sits in the data tensors
        # tensordict.shape = (*batch, time_steps), so time is at len(batch)
        time_dim = len(batch)

        # Extract initial carry: only the FIRST timestep's state and belief
        # Data tensors have shape (*batch, time_steps, feature_dim)
        # We select the first timestep to get (*batch, feature_dim)
        # Must make contiguous to match the stride/memory_format of carry output from combine_fn
        init_state = tensordict.get("state").select(time_dim, 0).contiguous()
        init_belief = tensordict.get("belief").select(time_dim, 0).contiguous()

        # Extract inputs for each timestep (xs)
        # These have shape (*batch, time_steps, feature_dim)
        actions = tensordict.get("action")
        encoded_latents = tensordict.get(("next", "encoded_latents"))

        # Move time dimension to first position for scan (scan operates on dim=0 by default)
        # From (*batch, time_steps, feature_dim) -> (time_steps, *batch, feature_dim)
        actions_t = actions.movedim(time_dim, 0)
        encoded_latents_t = encoded_latents.movedim(time_dim, 0)

        # Prepare noise tensors if provided
        if prior_noise is not None:
            prior_noise_t = prior_noise.movedim(time_dim, 0)
        else:
            prior_noise_t = None

        if posterior_noise is not None:
            posterior_noise_t = posterior_noise.movedim(time_dim, 0)
        else:
            posterior_noise_t = None

        # Build xs tuple - include noise if provided
        if prior_noise_t is not None and posterior_noise_t is not None:
            xs = (actions_t, encoded_latents_t, prior_noise_t, posterior_noise_t)

            def combine_fn(carry, xs):
                state, belief = carry
                action, enc_lat, prior_n, posterior_n = xs

                prior_mean, prior_std, _, next_belief = rssm_prior_module(
                    state, belief, action, noise=prior_n
                )
                posterior_mean, posterior_std, next_state = rssm_posterior_module(
                    next_belief, enc_lat, noise=posterior_n
                )

                next_carry = (next_state.clone(), next_belief.clone())
                outputs = (
                    prior_mean.clone(),
                    prior_std.clone(),
                    next_belief.clone(),
                    posterior_mean.clone(),
                    posterior_std.clone(),
                    next_state.clone(),
                )
                return next_carry, outputs

        elif prior_noise_t is not None:
            xs = (actions_t, encoded_latents_t, prior_noise_t)

            def combine_fn(carry, xs):
                state, belief = carry
                action, enc_lat, prior_n = xs

                prior_mean, prior_std, _, next_belief = rssm_prior_module(
                    state, belief, action, noise=prior_n
                )
                posterior_mean, posterior_std, next_state = rssm_posterior_module(
                    next_belief, enc_lat
                )

                next_carry = (next_state.clone(), next_belief.clone())
                outputs = (
                    prior_mean.clone(),
                    prior_std.clone(),
                    next_belief.clone(),
                    posterior_mean.clone(),
                    posterior_std.clone(),
                    next_state.clone(),
                )
                return next_carry, outputs

        elif posterior_noise_t is not None:
            xs = (actions_t, encoded_latents_t, posterior_noise_t)

            def combine_fn(carry, xs):
                state, belief = carry
                action, enc_lat, posterior_n = xs

                prior_mean, prior_std, _, next_belief = rssm_prior_module(
                    state, belief, action
                )
                posterior_mean, posterior_std, next_state = rssm_posterior_module(
                    next_belief, enc_lat, noise=posterior_n
                )

                next_carry = (next_state.clone(), next_belief.clone())
                outputs = (
                    prior_mean.clone(),
                    prior_std.clone(),
                    next_belief.clone(),
                    posterior_mean.clone(),
                    posterior_std.clone(),
                    next_state.clone(),
                )
                return next_carry, outputs

        else:
            # Use the cached combine_fn to avoid recompilation
            xs = (actions_t, encoded_latents_t)
            combine_fn = self._scan_combine_fn_no_noise

        # Run scan
        # Use eager backend for scan's internal compilation to avoid slow
        # Inductor compilation that scales O(sequence_length).
        # See: https://github.com/pytorch/pytorch/issues/...
        init = (init_state, init_belief)

        # Temporarily patch _maybe_compile_and_run_fn to use eager backend
        from torch._higher_order_ops import utils as hop_utils

        _orig_maybe_compile = hop_utils._maybe_compile_and_run_fn

        def _eager_compile_and_run_fn(fn, *args):
            if not torch.compiler.is_dynamo_compiling():
                # Force eager backend to avoid slow Inductor compilation
                with torch._dynamo.utils.disable_cache_limit():
                    return torch.compile(fn, backend="eager", fullgraph=True)(*args)
            else:
                return fn(*args)

        hop_utils._maybe_compile_and_run_fn = _eager_compile_and_run_fn
        try:
            final_carry, stacked_outputs = scan(combine_fn, init, xs, dim=0)
        finally:
            hop_utils._maybe_compile_and_run_fn = _orig_maybe_compile

        # Unpack stacked outputs
        # Each output has shape (time_steps, *batch, feature_dim)
        (
            prior_means,
            prior_stds,
            beliefs,
            posterior_means,
            posterior_stds,
            states,
        ) = stacked_outputs

        # Move time dimension back to original position
        # From (time_steps, *batch, feature_dim) -> (*batch, time_steps, feature_dim)
        prior_means = prior_means.movedim(0, time_dim)
        prior_stds = prior_stds.movedim(0, time_dim)
        beliefs = beliefs.movedim(0, time_dim)
        posterior_means = posterior_means.movedim(0, time_dim)
        posterior_stds = posterior_stds.movedim(0, time_dim)
        states = states.movedim(0, time_dim)

        # Also need the original input keys in the output
        # Get the input data excluding output keys
        base_td = tensordict.exclude(*self.out_keys).clone()

        # Construct output tensordict with the same structure as the loop version
        out = TensorDict(
            {
                **base_td.to_dict(),
                ("next", "prior_mean"): prior_means,
                ("next", "prior_std"): prior_stds,
                ("next", "belief"): beliefs,
                ("next", "posterior_mean"): posterior_means,
                ("next", "posterior_std"): posterior_stds,
                ("next", "state"): states,
            },
            batch_size=tensordict.batch_size,
            device=tensordict.device,
        )

        return out


class RSSMPrior(nn.Module):
    """The prior network of the RSSM.

    This network takes as input the previous state and belief and the current action.
    It returns the next prior state and belief, as well as the parameters of the prior state distribution.
    State is by construction stochastic and belief is deterministic. In "Dream to control", these are called "deterministic state " and "stochastic state", respectively.

    Reference: https://arxiv.org/abs/1811.04551

    Args:
        action_spec (TensorSpec): Action spec.
        hidden_dim (int, optional): Number of hidden units in the linear network. Input size of the recurrent network.
            Defaults to 200.
        rnn_hidden_dim (int, optional): Number of hidden units in the recurrent network. Also size of the belief.
            Defaults to 200.
        state_dim (int, optional): Size of the state.
            Defaults to 30.
        scale_lb (:obj:`float`, optional): Lower bound of the scale of the state distribution.
            Defaults to 0.1.
        action_dim (int, optional): Dimension of the action. If provided along with state_dim,
            uses explicit Linear instead of LazyLinear. Defaults to None for backward compatibility.
        device (torch.device, optional): Device to create the module on.
            Defaults to None (uses default device).


    """

    def __init__(
        self,
        action_spec,
        hidden_dim=200,
        rnn_hidden_dim=200,
        state_dim=30,
        scale_lb=0.1,
        action_dim=None,
        device=None,
    ):
        super().__init__()

        # Prior - use explicit Linear if action_dim provided, else LazyLinear
        self.rnn = GRUCell(hidden_dim, rnn_hidden_dim, device=device)
        if action_dim is not None:
            projector_in = state_dim + action_dim
            first_linear = nn.Linear(projector_in, hidden_dim, device=device)
        else:
            first_linear = nn.LazyLinear(hidden_dim, device=device)
        self.action_state_projector = nn.Sequential(first_linear, nn.ELU())
        self.rnn_to_prior_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, device=device),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * state_dim, device=device),
            NormalParamExtractor(
                scale_lb=scale_lb,
                scale_mapping="softplus",
            ),
        )

        self.state_dim = state_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.action_shape = action_spec.shape

    @_maybe_record_function_decorator("rssm_prior/forward")
    def forward(self, state, belief, action, *, noise=None):
        """Forward pass of the prior network.

        Args:
            state (torch.Tensor): Previous state tensor with shape ``(*batch, state_dim)``.
            belief (torch.Tensor): Previous belief/hidden state tensor with shape ``(*batch, rnn_hidden_dim)``.
            action (torch.Tensor): Action tensor with shape ``(*batch, action_dim)``.
            noise (torch.Tensor, optional): Pre-computed noise tensor for sampling with shape
                ``(*batch, state_dim)``. If ``None``, noise is sampled using ``torch.randn_like``.
                Passing noise enables deterministic execution for testing. Defaults to ``None``.

        Returns:
            tuple: A tuple of (prior_mean, prior_std, state, belief) where:
                - prior_mean (torch.Tensor): Mean of the prior distribution.
                - prior_std (torch.Tensor): Standard deviation of the prior distribution.
                - state (torch.Tensor): Sampled state from the prior.
                - belief (torch.Tensor): Updated belief/hidden state.
        """
        projector_input = torch.cat([state, action], dim=-1)
        action_state = self.action_state_projector(projector_input)
        unsqueeze = False
        if UNSQUEEZE_RNN_INPUT and action_state.ndimension() == 1:
            if belief is not None:
                belief = belief.unsqueeze(0)
            action_state = action_state.unsqueeze(0)
            unsqueeze = True

        # GRUCell can have issues with bfloat16 autocast on some GPU/cuBLAS combinations.
        # Run the RNN in full precision to avoid CUBLAS_STATUS_INVALID_VALUE errors.
        dtype = action_state.dtype
        device_type = action_state.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            belief = self.rnn(action_state.float(), belief.float() if belief is not None else None)
        belief = belief.to(dtype)
        if unsqueeze:
            belief = belief.squeeze(0)

        prior_mean, prior_std = self.rnn_to_prior_projector(belief)

        if noise is None:
            noise = torch.randn_like(prior_std)
        state = prior_mean + noise * prior_std

        return prior_mean, prior_std, state, belief


class RSSMPosterior(nn.Module):
    """The posterior network of the RSSM.

    This network takes as input the belief and the associated encoded observation.
    It returns the parameters of the posterior as well as a state sampled according to this distribution.

    Reference: https://arxiv.org/abs/1811.04551

    Args:
        hidden_dim (int, optional): Number of hidden units in the linear network.
            Defaults to 200.
        state_dim (int, optional): Size of the state.
            Defaults to 30.
        scale_lb (:obj:`float`, optional): Lower bound of the scale of the state distribution.
            Defaults to 0.1.
        rnn_hidden_dim (int, optional): Dimension of the belief/rnn hidden state.
            If provided along with obs_embed_dim, uses explicit Linear. Defaults to None.
        obs_embed_dim (int, optional): Dimension of the observation embedding.
            If provided along with rnn_hidden_dim, uses explicit Linear. Defaults to None.
        device (torch.device, optional): Device to create the module on.
            Defaults to None (uses default device).

    """

    def __init__(self, hidden_dim=200, state_dim=30, scale_lb=0.1, rnn_hidden_dim=None, obs_embed_dim=None, device=None):
        super().__init__()
        # Use explicit Linear if both dims provided, else LazyLinear for backward compat
        if rnn_hidden_dim is not None and obs_embed_dim is not None:
            projector_in = rnn_hidden_dim + obs_embed_dim
            first_linear = nn.Linear(projector_in, hidden_dim, device=device)
        else:
            first_linear = nn.LazyLinear(hidden_dim, device=device)
        self.obs_rnn_to_post_projector = nn.Sequential(
            first_linear,
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * state_dim, device=device),
            NormalParamExtractor(
                scale_lb=scale_lb,
                scale_mapping="softplus",
            ),
        )
        self.hidden_dim = hidden_dim

    @_maybe_record_function_decorator("rssm_posterior/forward")
    def forward(self, belief, obs_embedding, *, noise=None):
        """Forward pass of the posterior network.

        Args:
            belief (torch.Tensor): Belief/hidden state tensor from the prior with shape
                ``(*batch, rnn_hidden_dim)``.
            obs_embedding (torch.Tensor): Encoded observation tensor with shape
                ``(*batch, obs_embed_dim)``.
            noise (torch.Tensor, optional): Pre-computed noise tensor for sampling with shape
                ``(*batch, state_dim)``. If ``None``, noise is sampled using ``torch.randn_like``.
                Passing noise enables deterministic execution for testing. Defaults to ``None``.

        Returns:
            tuple: A tuple of (posterior_mean, posterior_std, state) where:
                - posterior_mean (torch.Tensor): Mean of the posterior distribution.
                - posterior_std (torch.Tensor): Standard deviation of the posterior distribution.
                - state (torch.Tensor): Sampled state from the posterior.
        """
        posterior_mean, posterior_std = self.obs_rnn_to_post_projector(
            torch.cat([belief, obs_embedding], dim=-1)
        )

        if noise is None:
            noise = torch.randn_like(posterior_std)
        state = posterior_mean + noise * posterior_std

        return posterior_mean, posterior_std, state
