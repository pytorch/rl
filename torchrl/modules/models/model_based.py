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

# from torchrl.modules.tensordict_module.rnn import GRUCell
from torch.nn import GRUCell

from torchrl.modules.models.models import MLP

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

    def __init__(
        self, channels=32, num_layers=4, in_channels=None, depth=None, device=None
    ):
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
        layers = [
            first_conv,
            nn.ReLU(),
        ]
        k = 1
        for _ in range(1, num_layers):
            layers += [
                nn.Conv2d(channels * k, channels * (k * 2), 4, stride=2, device=device),
                nn.ReLU(),
            ]
            k = k * 2
        self.encoder = nn.Sequential(*layers)

    def forward(self, observation):
        *batch_sizes, C, H, W = observation.shape
        if len(batch_sizes) == 0:
            end_dim = 0
        else:
            end_dim = len(batch_sizes) - 1
        observation = torch.flatten(observation, start_dim=0, end_dim=end_dim)
        obs_encoded = self.encoder(observation)
        latent = obs_encoded.reshape(*batch_sizes, -1)
        return latent


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

    def __init__(
        self,
        channels=32,
        num_layers=4,
        kernel_sizes=None,
        latent_dim=None,
        depth=None,
        device=None,
    ):
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
        layers = [
            nn.ReLU(),
            nn.ConvTranspose2d(channels, 3, kernel_sizes[-1], stride=2, device=device),
        ]
        kernel_sizes = kernel_sizes[:-1]
        k = 1
        for j in range(1, num_layers):
            if j != num_layers - 1:
                layers = [
                    nn.ConvTranspose2d(
                        channels * k * 2,
                        channels * k,
                        kernel_sizes[-1],
                        stride=2,
                        device=device,
                    ),
                ] + layers
                kernel_sizes = kernel_sizes[:-1]
                k = k * 2
                layers = [nn.ReLU()] + layers
            else:
                # Use explicit ConvTranspose2d - input is always channels * 8 from state_to_latent
                layers = [
                    nn.ConvTranspose2d(
                        linear_out,
                        channels * k,
                        kernel_sizes[-1],
                        stride=2,
                        device=device,
                    )
                ] + layers

        self.decoder = nn.Sequential(*layers)
        self._depth = channels

    def forward(self, state, rnn_hidden):
        latent = self.state_to_latent(torch.cat([state, rnn_hidden], dim=-1))
        *batch_sizes, D = latent.shape
        latent = latent.view(-1, D, 1, 1)
        obs_decoded = self.decoder(latent)
        _, C, H, W = obs_decoded.shape
        obs_decoded = obs_decoded.view(*batch_sizes, C, H, W)
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
        use_scan (bool, optional): If True, uses torch._higher_order_ops.scan for
            the rollout loop. This is more torch.compile friendly but may have
            different performance characteristics. Defaults to False.
        compile_step (bool, optional): If True, compiles the individual step function.
            Only used when use_scan=False. Defaults to False.
        compile_backend (str, optional): Backend to use for compilation.
            Defaults to "inductor".
        compile_mode (str, optional): Mode to use for compilation.
            Defaults to None (uses PyTorch default).


    """

    def __init__(
        self,
        rssm_prior: TensorDictModule,
        rssm_posterior: TensorDictModule,
        use_scan: bool = False,
        compile_step: bool = False,
        compile_backend: str = "inductor",
        compile_mode: str | None = None,
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
        self._compiled_step = None

    def _get_step_fn(self):
        """Get the step function, optionally compiled."""
        if self.compile_step and self._compiled_step is None:
            self._compiled_step = torch.compile(
                self._step,
                backend=self.compile_backend,
                mode=self.compile_mode,
            )
        return self._compiled_step if self.compile_step else self._step

    def _step(self, _tensordict):
        """Single RSSM step: prior + posterior."""
        self.rssm_prior(_tensordict)
        self.rssm_posterior(_tensordict)
        return _tensordict

    def forward(self, tensordict):
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

        """
        if self.use_scan:
            return self._forward_scan(tensordict)
        return self._forward_loop(tensordict)

    def _forward_loop(self, tensordict):
        """Traditional loop-based forward."""
        tensordict_out = []
        *batch, time_steps = tensordict.shape

        update_values = tensordict.exclude(*self.out_keys).unbind(-1)
        _tensordict = update_values[0]
        step_fn = self._get_step_fn()

        for t in range(time_steps):
            _tensordict = step_fn(_tensordict)

            tensordict_out.append(_tensordict)
            if t < time_steps - 1:
                # Translate ("next", *) to the non-next key required for the current step input
                _tensordict = _tensordict.select(*self.in_keys, strict=False)
                _tensordict = update_values[t + 1].update(_tensordict)

        out = torch.stack(tensordict_out, tensordict.ndim - 1)
        return out

    def _forward_scan(self, tensordict):
        """Scan-based forward using torch._higher_order_ops.scan.

        This is more torch.compile friendly as it avoids Python control flow.
        """
        from torch._higher_order_ops.scan import scan

        *batch, time_steps = tensordict.shape

        update_values = tensordict.exclude(*self.out_keys).unbind(-1)
        init_td = update_values[0]

        # Stack the update values for scan input
        stacked_updates = torch.stack(list(update_values), dim=0)

        def scan_fn(carry, x):
            # carry is the current tensordict, x is the update for this step
            _td = x.update(carry.select(*self.in_keys, strict=False))
            self.rssm_prior(_td)
            self.rssm_posterior(_td)
            # Return output and new carry
            return _td, _td

        # Run scan
        _, outputs = scan(scan_fn, [init_td], [stacked_updates])

        # outputs is stacked along dim 0, move to time dimension
        out = outputs.transpose(0, tensordict.ndim - 1)
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

    def forward(self, state, belief, action, noise=None):
        """Forward pass through the prior network.

        Args:
            state: Previous stochastic state.
            belief: Previous deterministic belief.
            action: Action to condition on.
            noise: Optional pre-sampled noise for the prior state.
                If None, samples from standard normal. Used for deterministic testing.

        Returns:
            Tuple of (prior_mean, prior_std, state, belief).
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
            belief = self.rnn(
                action_state.float(), belief.float() if belief is not None else None
            )
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

    def __init__(
        self,
        hidden_dim=200,
        state_dim=30,
        scale_lb=0.1,
        rnn_hidden_dim=None,
        obs_embed_dim=None,
        device=None,
    ):
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

    def forward(self, belief, obs_embedding, noise=None):
        """Forward pass through the posterior network.

        Args:
            belief: Deterministic belief from the prior.
            obs_embedding: Encoded observation.
            noise: Optional pre-sampled noise for the posterior state.
                If None, samples from standard normal. Used for deterministic testing.

        Returns:
            Tuple of (posterior_mean, posterior_std, state).
        """
        posterior_mean, posterior_std = self.obs_rnn_to_post_projector(
            torch.cat([belief, obs_embedding], dim=-1)
        )
        if noise is None:
            noise = torch.randn_like(posterior_std)
        state = posterior_mean + noise * posterior_std
        return posterior_mean, posterior_std, state
