# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import re

from numbers import Number

import numpy as np
import pytest
import torch
from packaging import version
from tensordict import TensorDict
from torch import nn
from torchrl.data.tensor_specs import Bounded, Composite
from torchrl.modules import (
    CEMPlanner,
    DTActor,
    GRU,
    GRUCell,
    LSTM,
    LSTMCell,
    MultiAgentConvNet,
    MultiAgentMLP,
    OnlineDTActor,
    QMixer,
    SafeModule,
    TanhModule,
    ValueOperator,
    VDNMixer,
)
from torchrl.modules.distributions.utils import safeatanh, safetanh
from torchrl.modules.models import (
    BatchRenorm1d,
    Conv3dNet,
    ConvNet,
    MLP,
    NoisyLazyLinear,
    NoisyLinear,
)
from torchrl.modules.models.decision_transformer import (
    _has_transformers,
    DecisionTransformer,
)
from torchrl.modules.models.model_based import (
    DreamerActor,
    ObsDecoder,
    ObsEncoder,
    RSSMPosterior,
    RSSMPrior,
    RSSMRollout,
)
from torchrl.modules.models.multiagent import MultiAgentNetBase
from torchrl.modules.models.utils import SquashDims
from torchrl.modules.planners.mppi import MPPIPlanner
from torchrl.objectives.value import TDLambdaEstimator

from torchrl.testing import get_default_devices, retry

from torchrl.testing.mocking_classes import MockBatchedUnLockedEnv


@pytest.fixture
def double_prec_fixture():
    dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)
    yield
    torch.set_default_dtype(dtype)


class TestMLP:
    @pytest.mark.parametrize("in_features", [3, 10, None])
    @pytest.mark.parametrize("out_features", [3, (3, 10)])
    @pytest.mark.parametrize("depth, num_cells", [(3, 32), (None, (32, 32, 32))])
    @pytest.mark.parametrize(
        "activation_class, activation_kwargs",
        [(nn.ReLU, {"inplace": True}), (nn.ReLU, {}), (nn.PReLU, {})],
    )
    @pytest.mark.parametrize(
        "norm_class, norm_kwargs",
        [
            (nn.LazyBatchNorm1d, {}),
            (nn.BatchNorm1d, {"num_features": 32}),
            (nn.LayerNorm, {"normalized_shape": 32}),
        ],
    )
    @pytest.mark.parametrize("dropout", [0.0, 0.5])
    @pytest.mark.parametrize("bias_last_layer", [True, False])
    @pytest.mark.parametrize("single_bias_last_layer", [True, False])
    @pytest.mark.parametrize("layer_class", [nn.Linear, NoisyLinear])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_mlp(
        self,
        in_features,
        out_features,
        depth,
        num_cells,
        activation_class,
        activation_kwargs,
        dropout,
        bias_last_layer,
        norm_class,
        norm_kwargs,
        single_bias_last_layer,
        layer_class,
        device,
        seed=0,
    ):
        torch.manual_seed(seed)
        batch = 2
        mlp = MLP(
            in_features=in_features,
            out_features=out_features,
            depth=depth,
            num_cells=num_cells,
            activation_class=activation_class,
            activation_kwargs=activation_kwargs,
            norm_class=norm_class,
            norm_kwargs=norm_kwargs,
            dropout=dropout,
            bias_last_layer=bias_last_layer,
            single_bias_last_layer=False,
            layer_class=layer_class,
            device=device,
        )
        if in_features is None:
            in_features = 5
        x = torch.randn(batch, in_features, device=device)
        y = mlp(x)
        out_features = (
            [out_features] if isinstance(out_features, Number) else out_features
        )
        assert y.shape == torch.Size([batch, *out_features])

    def test_kwargs(self):
        def make_activation(shift):
            return lambda x: x + shift

        def layer(*args, **kwargs):
            linear = nn.Linear(*args, **kwargs)
            linear.weight.data.copy_(torch.eye(4))
            return linear

        in_features = 4
        out_features = 4
        num_cells = [4, 4, 4]
        mlp = MLP(
            in_features=in_features,
            out_features=out_features,
            num_cells=num_cells,
            activation_class=make_activation,
            activation_kwargs=[{"shift": 0}, {"shift": 1}, {"shift": 2}],
            layer_class=layer,
            layer_kwargs=[{"bias": False}] * 4,
            bias_last_layer=False,
        )
        x = torch.zeros(4)
        y = mlp(x)
        for i, module in enumerate(mlp.modules()):
            if isinstance(module, nn.Linear):
                assert (module.weight == torch.eye(4)).all(), i
                assert module.bias is None, i
        assert (y == 3).all()


@pytest.mark.parametrize("in_features", [3, 10, None])
@pytest.mark.parametrize(
    "input_size, depth, num_cells, kernel_sizes, strides, paddings, expected_features",
    [(100, None, None, 3, 1, 0, 32 * 94 * 94), (100, 3, 32, 3, 1, 1, 32 * 100 * 100)],
)
@pytest.mark.parametrize(
    "activation_class, activation_kwargs",
    [(nn.ReLU, {"inplace": True}), (nn.ReLU, {}), (nn.PReLU, {})],
)
@pytest.mark.parametrize(
    "norm_class, norm_kwargs",
    [(None, None), (nn.LazyBatchNorm2d, {}), (nn.BatchNorm2d, {"num_features": 32})],
)
@pytest.mark.parametrize("bias_last_layer", [True, False])
@pytest.mark.parametrize(
    "aggregator_class, aggregator_kwargs",
    [(SquashDims, {})],
)
@pytest.mark.parametrize("squeeze_output", [False])
@pytest.mark.parametrize("device", get_default_devices())
@pytest.mark.parametrize("batch", [(2,), (2, 2)])
def test_convnet(
    batch,
    in_features,
    depth,
    num_cells,
    kernel_sizes,
    strides,
    paddings,
    activation_class,
    activation_kwargs,
    norm_class,
    norm_kwargs,
    bias_last_layer,
    aggregator_class,
    aggregator_kwargs,
    squeeze_output,
    device,
    input_size,
    expected_features,
    seed=0,
):
    torch.manual_seed(seed)
    convnet = ConvNet(
        in_features=in_features,
        depth=depth,
        num_cells=num_cells,
        kernel_sizes=kernel_sizes,
        strides=strides,
        paddings=paddings,
        activation_class=activation_class,
        activation_kwargs=activation_kwargs,
        norm_class=norm_class,
        norm_kwargs=norm_kwargs,
        bias_last_layer=bias_last_layer,
        aggregator_class=aggregator_class,
        aggregator_kwargs=aggregator_kwargs,
        squeeze_output=squeeze_output,
        device=device,
    )
    if in_features is None:
        in_features = 5
    x = torch.randn(*batch, in_features, input_size, input_size, device=device)
    y = convnet(x)
    assert y.shape == torch.Size([*batch, expected_features])


class TestConv3d:
    @pytest.mark.parametrize("in_features", [3, 10, None])
    @pytest.mark.parametrize(
        "input_size, depth, num_cells, kernel_sizes, strides, paddings, expected_features",
        [
            (10, None, None, 3, 1, 0, 32 * 4 * 4 * 4),
            (10, 3, 32, 3, 1, 1, 32 * 10 * 10 * 10),
        ],
    )
    @pytest.mark.parametrize(
        "activation_class, activation_kwargs",
        [(nn.ReLU, {"inplace": True}), (nn.ReLU, {}), (nn.PReLU, {})],
    )
    @pytest.mark.parametrize(
        "norm_class, norm_kwargs",
        [
            (None, None),
            (nn.LazyBatchNorm3d, {}),
            (nn.BatchNorm3d, {"num_features": 32}),
        ],
    )
    @pytest.mark.parametrize("bias_last_layer", [True, False])
    @pytest.mark.parametrize(
        "aggregator_class, aggregator_kwargs",
        [(SquashDims, None)],
    )
    @pytest.mark.parametrize("squeeze_output", [False])
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("batch", [(2,), (2, 2)])
    def test_conv3dnet(
        self,
        batch,
        in_features,
        depth,
        num_cells,
        kernel_sizes,
        strides,
        paddings,
        activation_class,
        activation_kwargs,
        norm_class,
        norm_kwargs,
        bias_last_layer,
        aggregator_class,
        aggregator_kwargs,
        squeeze_output,
        device,
        input_size,
        expected_features,
        seed=0,
    ):
        torch.manual_seed(seed)
        conv3dnet = Conv3dNet(
            in_features=in_features,
            depth=depth,
            num_cells=num_cells,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            activation_class=activation_class,
            activation_kwargs=activation_kwargs,
            norm_class=norm_class,
            norm_kwargs=norm_kwargs,
            bias_last_layer=bias_last_layer,
            aggregator_class=aggregator_class,
            aggregator_kwargs=aggregator_kwargs,
            squeeze_output=squeeze_output,
            device=device,
        )
        if in_features is None:
            in_features = 5
        x = torch.randn(
            *batch, in_features, input_size, input_size, input_size, device=device
        )
        y = conv3dnet(x)
        assert y.shape == torch.Size([*batch, expected_features])
        with pytest.raises(ValueError, match="must have at least 4 dimensions"):
            conv3dnet(torch.randn(3, 16, 16))

    def test_errors(self):
        with pytest.raises(
            ValueError, match="Null depth is not permitted with Conv3dNet"
        ):
            conv3dnet = Conv3dNet(
                in_features=5,
                num_cells=32,
                depth=0,
            )
        with pytest.raises(
            ValueError, match="depth=None requires one of the input args"
        ):
            conv3dnet = Conv3dNet(
                in_features=5,
                num_cells=32,
                depth=None,
            )
        with pytest.raises(
            ValueError, match="consider matching or specifying a constant num_cells"
        ):
            conv3dnet = Conv3dNet(
                in_features=5,
                num_cells=[32],
                depth=None,
                kernel_sizes=[3, 3],
            )


@pytest.mark.parametrize(
    "layer_class",
    [
        NoisyLinear,
        NoisyLazyLinear,
    ],
)
@pytest.mark.parametrize("device", get_default_devices())
def test_noisy(layer_class, device, seed=0):
    torch.manual_seed(seed)
    layer = layer_class(3, 4, device=device)
    x = torch.randn(10, 3, device=device)
    y1 = layer(x)
    layer.reset_noise()
    y2 = layer(x)
    y3 = layer(x)
    torch.testing.assert_close(y2, y3)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(y1, y2)


@pytest.mark.parametrize("device", get_default_devices())
@pytest.mark.parametrize("batch_size", [3, 5])
class TestPlanner:
    def test_CEM_model_free_env(self, device, batch_size, seed=1):
        env = MockBatchedUnLockedEnv(device=device)
        torch.manual_seed(seed)
        planner = CEMPlanner(
            env,
            planning_horizon=10,
            optim_steps=2,
            num_candidates=100,
            top_k=2,
        )
        td = env.reset(TensorDict(batch_size=batch_size).to(device))
        td_copy = td.clone()
        td = planner(td)
        assert (
            td.get("action").shape[-len(env.action_spec.shape) :]
            == env.action_spec.shape
        )
        assert env.action_spec.is_in(td.get("action"))

        for key in td.keys():
            if key != "action":
                assert torch.allclose(td[key], td_copy[key])

    def test_MPPI(self, device, batch_size, seed=1):
        torch.manual_seed(seed)
        env = MockBatchedUnLockedEnv(device=device)
        value_net = nn.LazyLinear(1, device=device)
        value_net = ValueOperator(value_net, in_keys=["observation"])
        advantage_module = TDLambdaEstimator(
            gamma=0.99,
            lmbda=0.95,
            value_network=value_net,
        )
        value_net(env.reset())
        planner = MPPIPlanner(
            env,
            advantage_module,
            temperature=1.0,
            planning_horizon=10,
            optim_steps=2,
            num_candidates=100,
            top_k=2,
        )
        td = env.reset(TensorDict(batch_size=batch_size).to(device))
        td_copy = td.clone()
        td = planner(td)
        assert (
            td.get("action").shape[-len(env.action_spec.shape) :]
            == env.action_spec.shape
        )
        assert env.action_spec.is_in(td.get("action"))

        for key in td.keys():
            if key != "action":
                assert torch.allclose(td[key], td_copy[key])


@pytest.mark.parametrize("device", get_default_devices())
@pytest.mark.parametrize("batch_size", [[], [3], [5]])
@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse("1.11.0"),
    reason="""Dreamer works with batches of null to 2 dimensions. Torch < 1.11
requires one-dimensional batches (for RNN and Conv nets for instance). If you'd like
to see torch < 1.11 supported for dreamer, please submit an issue.""",
)
class TestDreamerComponents:
    @pytest.mark.parametrize("out_features", [3, 5])
    @pytest.mark.parametrize("temporal_size", [[], [2], [4]])
    def test_dreamer_actor(self, device, batch_size, temporal_size, out_features):
        actor = DreamerActor(
            out_features,
        ).to(device)
        emb = torch.randn(*batch_size, *temporal_size, 15, device=device)
        state = torch.randn(*batch_size, *temporal_size, 2, device=device)
        loc, scale = actor(emb, state)
        assert loc.shape == (*batch_size, *temporal_size, out_features)
        assert scale.shape == (*batch_size, *temporal_size, out_features)
        assert torch.all(scale > 0)

    @pytest.mark.parametrize("depth", [32, 64])
    @pytest.mark.parametrize("temporal_size", [[], [2], [4]])
    def test_dreamer_encoder(self, device, temporal_size, batch_size, depth):
        encoder = ObsEncoder(channels=depth).to(device)
        obs = torch.randn(*batch_size, *temporal_size, 3, 64, 64, device=device)
        emb = encoder(obs)
        assert emb.shape == (*batch_size, *temporal_size, depth * 8 * 4)

    @pytest.mark.parametrize("depth", [32, 64])
    @pytest.mark.parametrize("stoch_size", [10, 20])
    @pytest.mark.parametrize("deter_size", [20, 30])
    @pytest.mark.parametrize("temporal_size", [[], [2], [4]])
    def test_dreamer_decoder(
        self, device, batch_size, temporal_size, depth, stoch_size, deter_size
    ):
        decoder = ObsDecoder(channels=depth).to(device)
        stoch_state = torch.randn(
            *batch_size, *temporal_size, stoch_size, device=device
        )
        det_state = torch.randn(*batch_size, *temporal_size, deter_size, device=device)
        obs = decoder(stoch_state, det_state)
        assert obs.shape == (*batch_size, *temporal_size, 3, 64, 64)

    @pytest.mark.parametrize("stoch_size", [10, 20])
    @pytest.mark.parametrize("deter_size", [20, 30])
    @pytest.mark.parametrize("action_size", [3, 6])
    def test_rssm_prior(self, device, batch_size, stoch_size, deter_size, action_size):
        action_spec = Bounded(shape=(action_size,), dtype=torch.float32, low=-1, high=1)
        rssm_prior = RSSMPrior(
            action_spec,
            hidden_dim=stoch_size,
            rnn_hidden_dim=stoch_size,
            state_dim=deter_size,
        ).to(device)
        state = torch.randn(*batch_size, deter_size, device=device)
        action = torch.randn(*batch_size, action_size, device=device)
        belief = torch.randn(*batch_size, stoch_size, device=device)
        prior_mean, prior_std, next_state, belief = rssm_prior(state, belief, action)
        assert prior_mean.shape == (*batch_size, deter_size)
        assert prior_std.shape == (*batch_size, deter_size)
        assert next_state.shape == (*batch_size, deter_size)
        assert belief.shape == (*batch_size, stoch_size)
        assert torch.all(prior_std > 0)

    @pytest.mark.parametrize("stoch_size", [10, 20])
    @pytest.mark.parametrize("deter_size", [20, 30])
    def test_rssm_posterior(self, device, batch_size, stoch_size, deter_size):
        rssm_posterior = RSSMPosterior(
            hidden_dim=stoch_size,
            state_dim=deter_size,
        ).to(device)
        belief = torch.randn(*batch_size, stoch_size, device=device)
        obs_emb = torch.randn(*batch_size, 1024, device=device)
        # Init of lazy linears
        _ = rssm_posterior(belief.clone(), obs_emb.clone())

        torch.manual_seed(0)
        posterior_mean, posterior_std, next_state = rssm_posterior(
            belief.clone(), obs_emb.clone()
        )
        assert posterior_mean.shape == (*batch_size, deter_size)
        assert posterior_std.shape == (*batch_size, deter_size)
        assert next_state.shape == (*batch_size, deter_size)
        assert torch.all(posterior_std > 0)

        torch.manual_seed(0)
        posterior_mean_bis, posterior_std_bis, next_state_bis = rssm_posterior(
            belief.clone(), obs_emb.clone()
        )
        assert torch.allclose(posterior_mean, posterior_mean_bis)
        assert torch.allclose(posterior_std, posterior_std_bis)
        assert torch.allclose(next_state, next_state_bis)

    @pytest.mark.parametrize("stoch_size", [10, 20])
    @pytest.mark.parametrize("deter_size", [20, 30])
    @pytest.mark.parametrize("temporal_size", [2, 4])
    @pytest.mark.parametrize("action_size", [3, 6])
    def test_rssm_rollout(
        self, device, batch_size, temporal_size, stoch_size, deter_size, action_size
    ):
        action_spec = Bounded(shape=(action_size,), dtype=torch.float32, low=-1, high=1)
        rssm_prior = RSSMPrior(
            action_spec,
            hidden_dim=stoch_size,
            rnn_hidden_dim=stoch_size,
            state_dim=deter_size,
        ).to(device)
        rssm_posterior = RSSMPosterior(
            hidden_dim=stoch_size,
            state_dim=deter_size,
        ).to(device)

        rssm_rollout = RSSMRollout(
            SafeModule(
                rssm_prior,
                in_keys=["state", "belief", "action"],
                out_keys=[
                    ("next", "prior_mean"),
                    ("next", "prior_std"),
                    "_",
                    ("next", "belief"),
                ],
            ),
            SafeModule(
                rssm_posterior,
                in_keys=[("next", "belief"), ("next", "encoded_latents")],
                out_keys=[
                    ("next", "posterior_mean"),
                    ("next", "posterior_std"),
                    ("next", "state"),
                ],
            ),
        )

        state = torch.randn(*batch_size, temporal_size, deter_size, device=device)
        belief = torch.randn(*batch_size, temporal_size, stoch_size, device=device)
        action = torch.randn(*batch_size, temporal_size, action_size, device=device)
        obs_emb = torch.randn(*batch_size, temporal_size, 1024, device=device)

        tensordict = TensorDict(
            {
                "state": state.clone(),
                "action": action.clone(),
                "next": {
                    "encoded_latents": obs_emb.clone(),
                    "belief": belief.clone(),
                },
            },
            device=device,
            batch_size=torch.Size([*batch_size, temporal_size]),
        )
        ## Init of lazy linears
        _ = rssm_rollout(tensordict.clone())
        torch.manual_seed(0)
        rollout = rssm_rollout(tensordict)
        assert rollout["next", "prior_mean"].shape == (
            *batch_size,
            temporal_size,
            deter_size,
        )
        assert rollout["next", "prior_std"].shape == (
            *batch_size,
            temporal_size,
            deter_size,
        )
        assert rollout["next", "state"].shape == (
            *batch_size,
            temporal_size,
            deter_size,
        )
        assert rollout["next", "belief"].shape == (
            *batch_size,
            temporal_size,
            stoch_size,
        )
        assert rollout["next", "posterior_mean"].shape == (
            *batch_size,
            temporal_size,
            deter_size,
        )
        assert rollout["next", "posterior_std"].shape == (
            *batch_size,
            temporal_size,
            deter_size,
        )
        assert torch.all(rollout["next", "prior_std"] > 0)
        assert torch.all(rollout["next", "posterior_std"] > 0)

        state[..., 1:, :] = 0
        belief[..., 1:, :] = 0
        # Only the first state is used for the prior. The rest are recomputed

        tensordict_bis = TensorDict(
            {
                "state": state.clone(),
                "action": action.clone(),
                "next": {"encoded_latents": obs_emb.clone(), "belief": belief.clone()},
            },
            device=device,
            batch_size=torch.Size([*batch_size, temporal_size]),
        )
        torch.manual_seed(0)
        rollout_bis = rssm_rollout(tensordict_bis)

        assert torch.allclose(
            rollout["next", "prior_mean"], rollout_bis["next", "prior_mean"]
        ), (rollout["next", "prior_mean"] - rollout_bis["next", "prior_mean"]).norm()
        assert torch.allclose(
            rollout["next", "prior_std"], rollout_bis["next", "prior_std"]
        )
        assert torch.allclose(rollout["next", "state"], rollout_bis["next", "state"])
        assert torch.allclose(rollout["next", "belief"], rollout_bis["next", "belief"])
        assert torch.allclose(
            rollout["next", "posterior_mean"], rollout_bis["next", "posterior_mean"]
        )
        assert torch.allclose(
            rollout["next", "posterior_std"], rollout_bis["next", "posterior_std"]
        )


class TestTanh:
    def test_errors(self):
        with pytest.raises(
            ValueError, match="in_keys and out_keys should have the same length"
        ):
            TanhModule(in_keys=["a", "b"], out_keys=["a"])
        with pytest.raises(ValueError, match=r"The minimum value \(-2\) provided"):
            spec = Bounded(-1, 1, shape=())
            TanhModule(in_keys=["act"], low=-2, spec=spec)
        with pytest.raises(ValueError, match=r"The maximum value \(-2\) provided to"):
            spec = Bounded(-1, 1, shape=())
            TanhModule(in_keys=["act"], high=-2, spec=spec)
        with pytest.raises(ValueError, match="Got high < low"):
            TanhModule(in_keys=["act"], high=-2, low=-1)

    def test_minmax(self):
        mod = TanhModule(
            in_keys=["act"],
            high=2,
        )
        assert isinstance(mod.act_high, torch.Tensor)
        mod = TanhModule(
            in_keys=["act"],
            low=-2,
        )
        assert isinstance(mod.act_low, torch.Tensor)
        mod = TanhModule(
            in_keys=["act"],
            high=np.ones((1,)),
        )
        assert isinstance(mod.act_high, torch.Tensor)
        mod = TanhModule(
            in_keys=["act"],
            low=-np.ones((1,)),
        )
        assert isinstance(mod.act_low, torch.Tensor)

    @pytest.mark.parametrize("clamp", [True, False])
    def test_boundaries(self, clamp):
        torch.manual_seed(0)
        eps = torch.finfo(torch.float).resolution
        for _ in range(10):
            min, max = (5 * torch.randn(2)).sort()[0]
            mod = TanhModule(in_keys=["act"], low=min, high=max, clamp=clamp)
            assert mod.non_trivial
            td = TensorDict({"act": (2 * torch.rand(100) - 1) * 10}, [])
            mod(td)
            # we should have a good proportion of samples close to the boundaries
            assert torch.isclose(td["act"], max).any()
            assert torch.isclose(td["act"], min).any()
            if not clamp:
                assert (td["act"] <= max + eps).all()
                assert (td["act"] >= min - eps).all()
            else:
                assert (td["act"] < max + eps).all()
                assert (td["act"] > min - eps).all()

    @pytest.mark.parametrize("out_keys", [[("a", "c"), "b"], None])
    @pytest.mark.parametrize("has_spec", [[True, True], [True, False], [False, False]])
    def test_multi_inputs(self, out_keys, has_spec):
        in_keys = [("x", "z"), "y"]
        real_out_keys = out_keys if out_keys is not None else in_keys

        if any(has_spec):
            spec = {}
            if has_spec[0]:
                spec.update({real_out_keys[0]: Bounded(-2.0, 2.0, shape=())})
                low, high = -2.0, 2.0
            if has_spec[1]:
                spec.update({real_out_keys[1]: Bounded(-3.0, 3.0, shape=())})
                low, high = None, None
            spec = Composite(spec)
        else:
            spec = None
            low, high = -2.0, 2.0

        mod = TanhModule(
            in_keys=in_keys,
            out_keys=out_keys,
            low=low,
            high=high,
            spec=spec,
            clamp=False,
        )
        data = TensorDict({in_key: torch.randn(100) * 100 for in_key in in_keys}, [])
        mod(data)
        assert all(out_key in data.keys(True, True) for out_key in real_out_keys)
        eps = torch.finfo(torch.float).resolution

        for out_key in real_out_keys:
            key = out_key if isinstance(out_key, str) else "_".join(out_key)
            low_key = f"{key}_low"
            high_key = f"{key}_high"
            min, max = getattr(mod, low_key), getattr(mod, high_key)
            assert torch.isclose(data[out_key], max).any()
            assert torch.isclose(data[out_key], min).any()
            assert (data[out_key] <= max + eps).all()
            assert (data[out_key] >= min - eps).all()


class TestMultiAgent:
    def _get_mock_input_td(
        self, n_agents, n_agents_inputs, state_shape=(64, 64, 3), T=None, batch=(2,)
    ):
        if T is not None:
            batch = batch + (T,)
        obs = torch.randn(*batch, n_agents, n_agents_inputs)
        state = torch.randn(*batch, *state_shape)

        td = TensorDict(
            {
                "agents": TensorDict(
                    {"observation": obs},
                    [*batch, n_agents],
                ),
                "state": state,
            },
            batch_size=batch,
        )
        return td

    @retry(AssertionError, 5)
    @pytest.mark.parametrize("n_agents", [1, 3])
    @pytest.mark.parametrize("share_params", [True, False])
    @pytest.mark.parametrize("centralized", [True, False])
    @pytest.mark.parametrize("n_agent_inputs", [6, None])
    @pytest.mark.parametrize("batch", [(4,), (4, 3), ()])
    def test_multiagent_mlp(
        self,
        n_agents,
        centralized,
        share_params,
        batch,
        n_agent_inputs,
        n_agent_outputs=2,
    ):
        torch.manual_seed(1)
        mlp = MultiAgentMLP(
            n_agent_inputs=n_agent_inputs,
            n_agent_outputs=n_agent_outputs,
            n_agents=n_agents,
            centralized=centralized,
            share_params=share_params,
            depth=2,
        )
        if n_agent_inputs is None:
            n_agent_inputs = 6
        td = self._get_mock_input_td(n_agents, n_agent_inputs, batch=batch)
        obs = td.get(("agents", "observation"))

        out = mlp(obs)
        assert out.shape == (*batch, n_agents, n_agent_outputs)
        for i in range(n_agents):
            if centralized and share_params:
                assert torch.allclose(out[..., i, :], out[..., 0, :])
            else:
                for j in range(i + 1, n_agents):
                    assert not torch.allclose(out[..., i, :], out[..., j, :])

        obs[..., 0, 0] += 1
        out2 = mlp(obs)
        for i in range(n_agents):
            if centralized:
                # a modification to the input of agent 0 will impact all agents
                assert not torch.allclose(out[..., i, :], out2[..., i, :])
            elif i > 0:
                assert torch.allclose(out[..., i, :], out2[..., i, :])

        obs = (
            torch.randn(*batch, 1, n_agent_inputs)
            .expand(*batch, n_agents, n_agent_inputs)
            .clone()
        )
        out = mlp(obs)
        for i in range(n_agents):
            if share_params:
                # same input same output
                assert torch.allclose(out[..., i, :], out[..., 0, :])
            else:
                for j in range(i + 1, n_agents):
                    # same input different output
                    assert not torch.allclose(out[..., i, :], out[..., j, :])
        pattern = rf"""MultiAgentMLP\(
    MLP\(
      \(0\): Linear\(in_features=\d+, out_features=32, bias=True\)
      \(1\): Tanh\(\)
      \(2\): Linear\(in_features=32, out_features=32, bias=True\)
      \(3\): Tanh\(\)
      \(4\): Linear\(in_features=32, out_features=2, bias=True\)
    \),
    n_agents={n_agents},
    share_params={share_params},
    centralized={centralized},
    agent_dim={-2}\)"""
        assert re.match(pattern, str(mlp), re.DOTALL)

    @retry(AssertionError, 5)
    @pytest.mark.parametrize("n_agents", [1, 3])
    @pytest.mark.parametrize("share_params", [True, False])
    @pytest.mark.parametrize("centralized", [True, False])
    @pytest.mark.parametrize("n_agent_inputs", [6, None])
    @pytest.mark.parametrize("batch", [(4,), (4, 3), ()])
    def test_multiagent_mlp_init(
        self,
        n_agents,
        centralized,
        share_params,
        batch,
        n_agent_inputs,
        n_agent_outputs=2,
    ):
        torch.manual_seed(1)
        mlp = MultiAgentMLP(
            n_agent_inputs=n_agent_inputs,
            n_agent_outputs=n_agent_outputs,
            n_agents=n_agents,
            centralized=centralized,
            share_params=share_params,
            depth=2,
        )
        for m in mlp.modules():
            if isinstance(m, nn.Linear):
                assert not isinstance(m.weight, nn.Parameter)
                assert m.weight.device == torch.device("meta")
                break
        else:
            raise RuntimeError("could not find a Linear module")
        if n_agent_inputs is None:
            n_agent_inputs = 6
        td = self._get_mock_input_td(n_agents, n_agent_inputs, batch=batch)
        obs = td.get(("agents", "observation"))
        mlp(obs)
        snet = mlp.get_stateful_net()
        assert snet is not mlp._empty_net

        def zero_inplace(mod):
            if hasattr(mod, "weight"):
                mod.weight.data *= 0
            if hasattr(mod, "bias"):
                mod.bias.data *= 0

        snet.apply(zero_inplace)
        assert (mlp.params == 0).all()

        def one_outofplace(mod):
            if hasattr(mod, "weight"):
                mod.weight = nn.Parameter(torch.ones_like(mod.weight.data))
            if hasattr(mod, "bias"):
                mod.bias = nn.Parameter(torch.ones_like(mod.bias.data))

        snet.apply(one_outofplace)
        assert (mlp.params == 0).all()
        mlp.from_stateful_net(snet)
        assert (mlp.params == 1).all()

    @retry(AssertionError, 5)
    @pytest.mark.parametrize("n_agents", [3])
    @pytest.mark.parametrize("share_params", [True])
    @pytest.mark.parametrize("centralized", [True])
    @pytest.mark.parametrize("n_agent_inputs", [6])
    @pytest.mark.parametrize("batch", [(4,)])
    @pytest.mark.parametrize("tdparams", [True, False])
    def test_multiagent_mlp_tdparams(
        self,
        n_agents,
        centralized,
        share_params,
        batch,
        n_agent_inputs,
        tdparams,
        n_agent_outputs=2,
    ):
        torch.manual_seed(1)
        mlp = MultiAgentMLP(
            n_agent_inputs=n_agent_inputs,
            n_agent_outputs=n_agent_outputs,
            n_agents=n_agents,
            centralized=centralized,
            share_params=share_params,
            depth=2,
            use_td_params=tdparams,
        )
        if tdparams:
            assert list(mlp._empty_net.parameters()) == []
            assert list(mlp.params.parameters()) == list(mlp.parameters())
        else:
            assert list(mlp._empty_net.parameters()) == list(mlp.parameters())
            assert not hasattr(mlp.params, "parameters")
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            return
        mlp = nn.Sequential(mlp)
        mlp.to(device)
        param_set = set(mlp.parameters())
        for p in mlp[0].params.values(True, True):
            assert p in param_set

    def test_multiagent_mlp_lazy(self):
        mlp = MultiAgentMLP(
            n_agent_inputs=None,
            n_agent_outputs=6,
            n_agents=3,
            centralized=True,
            share_params=False,
            depth=2,
        )
        optim = torch.optim.SGD(mlp.parameters(), lr=1e-3)
        for p in mlp.parameters():
            if isinstance(p, torch.nn.parameter.UninitializedParameter):
                break
        else:
            raise AssertionError("No UninitializedParameter found")
        for p in optim.param_groups[0]["params"]:
            if isinstance(p, torch.nn.parameter.UninitializedParameter):
                break
        else:
            raise AssertionError("No UninitializedParameter found")
        for _ in range(2):
            td = self._get_mock_input_td(3, 4, batch=(10,))
            obs = td.get(("agents", "observation"))
            out = mlp(obs)
            assert (
                not mlp.params[0]
                .apply(lambda x, y: torch.isclose(x, y), mlp.params[1])
                .any()
            )
            out.mean().backward()
            optim.step()
        for p in mlp.parameters():
            if isinstance(p, torch.nn.parameter.UninitializedParameter):
                raise AssertionError("UninitializedParameter found")
        for p in optim.param_groups[0]["params"]:
            if isinstance(p, torch.nn.parameter.UninitializedParameter):
                raise AssertionError("UninitializedParameter found")

    @pytest.mark.parametrize("n_agents", [1, 3])
    @pytest.mark.parametrize("share_params", [True, False])
    @pytest.mark.parametrize("centralized", [True, False])
    def test_multiagent_reset_mlp(
        self,
        n_agents,
        centralized,
        share_params,
    ):
        actor_net = MultiAgentMLP(
            n_agent_inputs=4,
            n_agent_outputs=6,
            num_cells=(4, 4),
            n_agents=n_agents,
            centralized=centralized,
            share_params=share_params,
        )
        params_before = actor_net.params.clone()
        actor_net.reset_parameters()
        params_after = actor_net.params
        assert not params_before.apply(
            lambda x, y: torch.isclose(x, y), params_after, batch_size=[]
        ).any()
        if params_after.numel() > 1:
            assert (
                not params_after[0]
                .apply(lambda x, y: torch.isclose(x, y), params_after[1], batch_size=[])
                .any()
            )

    @pytest.mark.parametrize("share_params", [True, False])
    @pytest.mark.parametrize("agent_dim", [1, -3])
    def test_multiagent_custom_agent_dim(self, share_params, agent_dim):
        """Test that custom agent_dim values work correctly.

        Regression test for https://github.com/pytorch/rl/issues/3288
        """
        n_agents = 3
        obs_dim = 5
        seq_len = 6
        output_dim = 4

        class SingleAgentMLP(nn.Module):
            def __init__(self, in_dim, out_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, 32),
                    nn.Tanh(),
                    nn.Linear(32, out_dim),
                )

            def forward(self, x):
                return self.net(x)

        class MultiAgentPolicyNet(MultiAgentNetBase):
            def __init__(
                self,
                obs_dim,
                output_dim,
                n_agents,
                share_params,
                agent_dim,
                device=None,
            ):
                self.obs_dim = obs_dim
                self.output_dim = output_dim
                self._agent_dim = agent_dim

                super().__init__(
                    n_agents=n_agents,
                    centralized=False,
                    share_params=share_params,
                    agent_dim=agent_dim,
                    device=device,
                )

            def _build_single_net(self, *, device, **kwargs):
                net = SingleAgentMLP(self.obs_dim, self.output_dim)
                return net.to(device) if device is not None else net

            def _pre_forward_check(self, inputs):
                if inputs.shape[self._agent_dim] != self.n_agents:
                    raise ValueError(
                        f"Multi-agent network expected input with shape[{self._agent_dim}]={self.n_agents},"
                        f" but got {inputs.shape}"
                    )
                return inputs

        policy_net = MultiAgentPolicyNet(
            obs_dim=obs_dim,
            output_dim=output_dim,
            n_agents=n_agents,
            share_params=share_params,
            agent_dim=agent_dim,
        )

        # Input shape: (batch, n_agents, seq_len, obs_dim) with agents at dim 1
        batch_size = 4
        obs = torch.randn(batch_size, n_agents, seq_len, obs_dim)
        out = policy_net(obs)

        # Output should preserve agent dimension position
        expected_shape = (batch_size, n_agents, seq_len, output_dim)
        assert (
            out.shape == expected_shape
        ), f"Expected {expected_shape}, got {out.shape}"

        # Verify different agents produce different outputs (unless share_params with same input)
        if not share_params:
            for i in range(n_agents):
                for j in range(i + 1, n_agents):
                    assert not torch.allclose(out[:, i], out[:, j])

    @pytest.mark.parametrize("n_agents", [1, 3])
    @pytest.mark.parametrize("share_params", [True, False])
    @pytest.mark.parametrize("centralized", [True, False])
    @pytest.mark.parametrize("channels", [3, None])
    @pytest.mark.parametrize("batch", [(4,), (4, 3), ()])
    def test_multiagent_cnn(
        self,
        n_agents,
        centralized,
        share_params,
        batch,
        channels,
        x=15,
        y=15,
    ):
        torch.manual_seed(0)
        cnn = MultiAgentConvNet(
            n_agents=n_agents,
            centralized=centralized,
            share_params=share_params,
            in_features=channels,
            kernel_sizes=3,
        )
        if channels is None:
            channels = 3
        td = TensorDict(
            {
                "agents": TensorDict(
                    {"observation": torch.randn(*batch, n_agents, channels, x, y)},
                    [*batch, n_agents],
                )
            },
            batch_size=batch,
        )
        obs = td[("agents", "observation")]
        out = cnn(obs)
        assert out.shape[:-1] == (*batch, n_agents)
        if centralized and share_params:
            torch.testing.assert_close(out, out[..., :1, :].expand_as(out))
        else:
            for i in range(n_agents):
                for j in range(i + 1, n_agents):
                    assert not torch.allclose(out[..., i, :], out[..., j, :])
        obs[..., 0, 0, 0, 0] += 1
        out2 = cnn(obs)
        if centralized:
            # a modification to the input of agent 0 will impact all agents
            assert not torch.isclose(out, out2).all()
        elif n_agents > 1:
            assert not torch.isclose(out[..., 0, :], out2[..., 0, :]).all()
            torch.testing.assert_close(out[..., 1:, :], out2[..., 1:, :])

        obs = torch.randn(*batch, 1, channels, x, y).expand(
            *batch, n_agents, channels, x, y
        )
        out = cnn(obs)
        for i in range(n_agents):
            if share_params:
                # same input same output
                assert torch.allclose(out[..., i, :], out[..., 0, :])
            else:
                for j in range(i + 1, n_agents):
                    # same input different output
                    assert not torch.allclose(out[..., i, :], out[..., j, :])

    def test_multiagent_cnn_lazy(self):
        torch.manual_seed(42)
        n_agents = 5
        n_channels = 3
        cnn = MultiAgentConvNet(
            n_agents=n_agents,
            centralized=False,
            share_params=False,
            in_features=None,
            kernel_sizes=3,
        )
        optim = torch.optim.SGD(cnn.parameters(), lr=1e-3)
        for p in cnn.parameters():
            if isinstance(p, torch.nn.parameter.UninitializedParameter):
                break
        else:
            raise AssertionError("No UninitializedParameter found")
        for p in optim.param_groups[0]["params"]:
            if isinstance(p, torch.nn.parameter.UninitializedParameter):
                break
        else:
            raise AssertionError("No UninitializedParameter found")
        for _ in range(2):
            td = TensorDict(
                {
                    "agents": TensorDict(
                        {"observation": torch.randn(4, n_agents, n_channels, 15, 15)},
                        [4, 5],
                    )
                },
                batch_size=[4],
            )
            obs = td[("agents", "observation")]
            out = cnn(obs)
            assert (
                not cnn.params[0]
                .apply(lambda x, y: torch.isclose(x, y), cnn.params[1])
                .any()
            )
            out.mean().backward()
            optim.step()
        for p in cnn.parameters():
            if isinstance(p, torch.nn.parameter.UninitializedParameter):
                raise AssertionError("UninitializedParameter found")
        for p in optim.param_groups[0]["params"]:
            if isinstance(p, torch.nn.parameter.UninitializedParameter):
                raise AssertionError("UninitializedParameter found")

    @pytest.mark.parametrize("n_agents", [1, 3])
    @pytest.mark.parametrize("share_params", [True, False])
    @pytest.mark.parametrize("centralized", [True, False])
    def test_multiagent_reset_cnn(
        self,
        n_agents,
        centralized,
        share_params,
    ):
        torch.manual_seed(42)
        actor_net = MultiAgentConvNet(
            in_features=4,
            num_cells=[5, 5],
            n_agents=n_agents,
            centralized=centralized,
            share_params=share_params,
        )
        params_before = actor_net.params.clone()
        actor_net.reset_parameters()
        params_after = actor_net.params
        assert not params_before.apply(
            lambda x, y: torch.isclose(x, y), params_after, batch_size=[]
        ).any()
        if params_after.numel() > 1:
            assert (
                not params_after[0]
                .apply(lambda x, y: torch.isclose(x, y), params_after[1], batch_size=[])
                .any()
            )

    @pytest.mark.parametrize("n_agents", [1, 3])
    @pytest.mark.parametrize("batch", [(10,), (10, 3), ()])
    def test_vdn(self, n_agents, batch):
        torch.manual_seed(0)
        mixer = VDNMixer(n_agents=n_agents, device="cpu")

        td = self._get_mock_input_td(n_agents, batch=batch, n_agents_inputs=1)
        obs = td.get(("agents", "observation"))
        assert obs.shape == (*batch, n_agents, 1)
        out = mixer(obs)
        assert out.shape == (*batch, 1)
        assert torch.equal(obs.sum(-2), out)

    @pytest.mark.parametrize("n_agents", [1, 3])
    @pytest.mark.parametrize("batch", [(10,), (10, 3), ()])
    @pytest.mark.parametrize("state_shape", [(64, 64, 3), (10,)])
    def test_qmix(self, n_agents, batch, state_shape):
        torch.manual_seed(0)
        mixer = QMixer(
            n_agents=n_agents,
            state_shape=state_shape,
            mixing_embed_dim=32,
            device="cpu",
        )

        td = self._get_mock_input_td(
            n_agents, batch=batch, n_agents_inputs=1, state_shape=state_shape
        )
        obs = td.get(("agents", "observation"))
        state = td.get("state")
        assert obs.shape == (*batch, n_agents, 1)
        assert state.shape == (*batch, *state_shape)
        out = mixer(obs, state)
        assert out.shape == (*batch, 1)

    @pytest.mark.parametrize("mixer", ["qmix", "vdn"])
    def test_mixer_malformed_input(
        self, mixer, n_agents=3, batch=(32,), state_shape=(64, 64, 3)
    ):
        td = self._get_mock_input_td(
            n_agents, batch=batch, n_agents_inputs=3, state_shape=state_shape
        )
        if mixer == "qmix":
            mixer = QMixer(
                n_agents=n_agents,
                state_shape=state_shape,
                mixing_embed_dim=32,
                device="cpu",
            )
        else:
            mixer = VDNMixer(n_agents=n_agents, device="cpu")
        obs = td.get(("agents", "observation"))
        state = td.get("state")

        if mixer.needs_state:
            with pytest.raises(
                ValueError,
                match="Mixer that needs state was passed more than 2 inputs",
            ):
                mixer(obs)
        else:
            with pytest.raises(
                ValueError,
                match="Mixer that doesn't need state was passed more than 1 input",
            ):
                mixer(obs, state)

        in_put = [obs, state] if mixer.needs_state else [obs]
        with pytest.raises(
            ValueError,
            match="Mixer network expected chosen_action_value with last 2 dimensions",
        ):
            mixer(*in_put)
        if mixer.needs_state:
            state_diff = state.unsqueeze(-1)
            with pytest.raises(
                ValueError,
                match="Mixer network expected state with ending shape",
            ):
                mixer(obs, state_diff)

        td = self._get_mock_input_td(
            n_agents, batch=batch, n_agents_inputs=1, state_shape=state_shape
        )
        obs = td.get(("agents", "observation"))
        state = td.get("state")
        obs = obs.sum(-2)
        in_put = [obs, state] if mixer.needs_state else [obs]
        with pytest.raises(
            ValueError,
            match="Mixer network expected chosen_action_value with last 2 dimensions",
        ):
            mixer(*in_put)

        obs = td.get(("agents", "observation"))
        state = td.get("state")
        in_put = [obs, state] if mixer.needs_state else [obs]
        mixer(*in_put)


@pytest.mark.skipif(torch.__version__ < "2.0", reason="torch 2.0 is required")
@pytest.mark.parametrize("use_vmap", [False, True])
@pytest.mark.parametrize("scale", range(10))
def test_tanh_atanh(use_vmap, scale):
    if use_vmap:
        try:
            from torch import vmap
        except ImportError:
            try:
                from functorch import vmap
            except ImportError:
                raise pytest.skip("functorch not found")

    torch.manual_seed(0)
    x = (torch.randn(10, dtype=torch.double) * scale).requires_grad_(True)
    if not use_vmap:
        y = safetanh(x, 1e-6)
    else:
        y = vmap(safetanh, (0, None))(x, 1e-6)

    if not use_vmap:
        xp = safeatanh(y, 1e-6)
    else:
        xp = vmap(safeatanh, (0, None))(y, 1e-6)

    xp.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


@pytest.mark.skipif(
    not _has_transformers, reason="transformers needed for TestDecisionTransformer"
)
class TestDecisionTransformer:
    def test_init(self):
        DecisionTransformer(
            3,
            4,
        )
        with pytest.raises(TypeError):
            DecisionTransformer(3, 4, config="some_str")
        DecisionTransformer(
            3,
            4,
            config=DecisionTransformer.DTConfig(
                n_layer=2, n_embd=16, n_positions=16, n_inner=16, n_head=2
            ),
        )

    @pytest.mark.parametrize("batch_dims", [[], [3], [3, 4]])
    def test_exec(self, batch_dims, T=5):
        observations = torch.randn(*batch_dims, T, 3)
        actions = torch.randn(*batch_dims, T, 4)
        r2go = torch.randn(*batch_dims, T, 1)
        model = DecisionTransformer(
            3,
            4,
            config=DecisionTransformer.DTConfig(
                n_layer=2, n_embd=16, n_positions=16, n_inner=16, n_head=2
            ),
        )
        out = model(observations, actions, r2go)
        assert out.shape == torch.Size([*batch_dims, T, 16])

    @pytest.mark.parametrize("batch_dims", [[], [3], [3, 4]])
    def test_dtactor(self, batch_dims, T=5):
        dtactor = DTActor(
            3,
            4,
            transformer_config=DecisionTransformer.DTConfig(
                n_layer=2, n_embd=16, n_positions=16, n_inner=16, n_head=2
            ),
        )
        observations = torch.randn(*batch_dims, T, 3)
        actions = torch.randn(*batch_dims, T, 4)
        r2go = torch.randn(*batch_dims, T, 1)
        out = dtactor(observations, actions, r2go)
        assert out.shape == torch.Size([*batch_dims, T, 4])

    @pytest.mark.parametrize("batch_dims", [[], [3], [3, 4]])
    def test_onlinedtactor(self, batch_dims, T=5):
        dtactor = OnlineDTActor(
            3,
            4,
            transformer_config=DecisionTransformer.DTConfig(
                n_layer=2, n_embd=16, n_positions=16, n_inner=16, n_head=2
            ),
        )
        observations = torch.randn(*batch_dims, T, 3)
        actions = torch.randn(*batch_dims, T, 4)
        r2go = torch.randn(*batch_dims, T, 1)
        mu, sig = dtactor(observations, actions, r2go)
        assert mu.shape == torch.Size([*batch_dims, T, 4])
        assert sig.shape == torch.Size([*batch_dims, T, 4])
        assert (dtactor.log_std_min < sig.log()).all()
        assert (dtactor.log_std_max > sig.log()).all()


@pytest.mark.parametrize("device", get_default_devices())
@pytest.mark.parametrize("bias", [True, False])
def test_python_lstm_cell(device, bias):
    lstm_cell1 = LSTMCell(10, 20, device=device, bias=bias)
    lstm_cell2 = nn.LSTMCell(10, 20, device=device, bias=bias)

    lstm_cell1.load_state_dict(lstm_cell2.state_dict())

    # Make sure parameters match
    for (k1, v1), (k2, v2) in zip(
        lstm_cell1.named_parameters(), lstm_cell2.named_parameters()
    ):
        assert k1 == k2, f"Parameter names do not match: {k1} != {k2}"
        assert (
            v1.shape == v2.shape
        ), f"Parameter shapes do not match: {k1} shape {v1.shape} != {k2} shape {v2.shape}"

    # Run loop
    input = torch.randn(2, 3, 10, device=device)
    h0 = torch.randn(3, 20, device=device)
    c0 = torch.randn(3, 20, device=device)
    with torch.no_grad():
        for i in range(input.size()[0]):
            h1, c1 = lstm_cell1(input[i], (h0, c0))
            h2, c2 = lstm_cell2(input[i], (h0, c0))

            # Make sure the final hidden states have the same shape
            assert h1.shape == h2.shape
            assert c1.shape == c2.shape
            torch.testing.assert_close(h1, h2)
            torch.testing.assert_close(c1, c2)
            h0 = h1
            c0 = c1


@pytest.mark.parametrize("device", get_default_devices())
@pytest.mark.parametrize("bias", [True, False])
def test_python_gru_cell(device, bias):
    gru_cell1 = GRUCell(10, 20, device=device, bias=bias)
    gru_cell2 = nn.GRUCell(10, 20, device=device, bias=bias)

    gru_cell2.load_state_dict(gru_cell1.state_dict())

    # Make sure parameters match
    for (k1, v1), (k2, v2) in zip(
        gru_cell1.named_parameters(), gru_cell2.named_parameters()
    ):
        assert k1 == k2, f"Parameter names do not match: {k1} != {k2}"
        assert (v1 == v2).all()
        assert (
            v1.shape == v2.shape
        ), f"Parameter shapes do not match: {k1} shape {v1.shape} != {k2} shape {v2.shape}"

    # Run loop
    input = torch.randn(2, 3, 10, device=device)
    h0 = torch.zeros(3, 20, device=device)
    with torch.no_grad():
        for i in range(input.size()[0]):
            h1 = gru_cell1(input[i], h0)
            h2 = gru_cell2(input[i], h0)

            # Make sure the final hidden states have the same shape
            assert h1.shape == h2.shape
            torch.testing.assert_close(h1, h2)
            h0 = h1


@pytest.mark.parametrize("device", get_default_devices())
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("dropout", [0.0, 0.5])
@pytest.mark.parametrize("num_layers", [1, 2])
def test_python_lstm(device, bias, dropout, batch_first, num_layers):
    B = 5
    T = 3
    lstm1 = LSTM(
        input_size=10,
        hidden_size=20,
        num_layers=num_layers,
        device=device,
        bias=bias,
        batch_first=batch_first,
    )
    lstm2 = nn.LSTM(
        input_size=10,
        hidden_size=20,
        num_layers=num_layers,
        device=device,
        bias=bias,
        batch_first=batch_first,
    )

    lstm2.load_state_dict(lstm1.state_dict())

    # Make sure parameters match
    for (k1, v1), (k2, v2) in zip(lstm1.named_parameters(), lstm2.named_parameters()):
        assert k1 == k2, f"Parameter names do not match: {k1} != {k2}"
        assert (
            v1.shape == v2.shape
        ), f"Parameter shapes do not match: {k1} shape {v1.shape} != {k2} shape {v2.shape}"

    if batch_first:
        input = torch.randn(B, T, 10, device=device)
    else:
        input = torch.randn(T, B, 10, device=device)

    h0 = torch.randn(num_layers, 5, 20, device=device)
    c0 = torch.randn(num_layers, 5, 20, device=device)

    # Test without hidden states
    with torch.no_grad():
        output1, (h1, c1) = lstm1(input)
        output2, (h2, c2) = lstm2(input)

    assert h1.shape == h2.shape
    assert c1.shape == c2.shape
    assert output1.shape == output2.shape
    if dropout == 0.0:
        torch.testing.assert_close(output1, output2)
        torch.testing.assert_close(h1, h2)
        torch.testing.assert_close(c1, c2)

    # Test with hidden states
    with torch.no_grad():
        output1, (h1, c1) = lstm1(input, (h0, c0))
        output2, (h2, c2) = lstm1(input, (h0, c0))

    assert h1.shape == h2.shape
    assert c1.shape == c2.shape
    assert output1.shape == output2.shape
    if dropout == 0.0:
        torch.testing.assert_close(output1, output2)
        torch.testing.assert_close(h1, h2)
        torch.testing.assert_close(c1, c2)


@pytest.mark.parametrize("device", get_default_devices())
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("dropout", [0.0, 0.5])
@pytest.mark.parametrize("num_layers", [1, 2])
def test_python_gru(device, bias, dropout, batch_first, num_layers):
    B = 5
    T = 3
    gru1 = GRU(
        input_size=10,
        hidden_size=20,
        num_layers=num_layers,
        device=device,
        bias=bias,
        batch_first=batch_first,
    )
    gru2 = nn.GRU(
        input_size=10,
        hidden_size=20,
        num_layers=num_layers,
        device=device,
        bias=bias,
        batch_first=batch_first,
    )
    gru2.load_state_dict(gru1.state_dict())

    # Make sure parameters match
    for (k1, v1), (k2, v2) in zip(gru1.named_parameters(), gru2.named_parameters()):
        assert k1 == k2, f"Parameter names do not match: {k1} != {k2}"
        torch.testing.assert_close(v1, v2)
        assert (
            v1.shape == v2.shape
        ), f"Parameter shapes do not match: {k1} shape {v1.shape} != {k2} shape {v2.shape}"

    if batch_first:
        input = torch.randn(B, T, 10, device=device)
    else:
        input = torch.randn(T, B, 10, device=device)

    h0 = torch.randn(num_layers, 5, 20, device=device)

    # Test without hidden states
    with torch.no_grad():
        output1, h1 = gru1(input)
        output2, h2 = gru2(input)

    assert h1.shape == h2.shape
    assert output1.shape == output2.shape
    if dropout == 0.0:
        torch.testing.assert_close(output1, output2)
        torch.testing.assert_close(h1, h2)

    # Test with hidden states
    with torch.no_grad():
        output1, h1 = gru1(input, h0)
        output2, h2 = gru2(input, h0)

    assert h1.shape == h2.shape
    assert output1.shape == output2.shape
    if dropout == 0.0:
        torch.testing.assert_close(output1, output2)
        torch.testing.assert_close(h1, h2)


class TestBatchRenorm:
    @pytest.mark.parametrize("num_steps", [0, 5])
    @pytest.mark.parametrize("smooth", [False, True])
    def test_batchrenorm(self, num_steps, smooth):
        torch.manual_seed(0)
        bn = torch.nn.BatchNorm1d(5, momentum=0.1, eps=1e-5)
        brn = BatchRenorm1d(
            5,
            momentum=0.1,
            eps=1e-5,
            warmup_steps=num_steps,
            max_d=10000,
            max_r=10000,
            smooth=smooth,
        )
        bn.train()
        brn.train()
        data_train = torch.randn(100, 5).split(25)
        data_test = torch.randn(100, 5)
        for i, d in enumerate(data_train):
            b = bn(d)
            a = brn(d)
            if num_steps > 0 and (
                (i < num_steps and not smooth) or (i == 0 and smooth)
            ):
                torch.testing.assert_close(a, b)
            else:
                assert not torch.isclose(a, b).all(), i

        bn.eval()
        brn.eval()
        torch.testing.assert_close(bn(data_test), brn(data_test))


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
