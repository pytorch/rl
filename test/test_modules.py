# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
from numbers import Number

import pytest
import torch
from _utils_internal import get_available_devices
from mocking_classes import MockBatchedUnLockedEnv
from packaging import version
from tensordict import TensorDict
from torch import nn
from torchrl.data.tensor_specs import (
    DiscreteTensorSpec,
    NdBoundedTensorSpec,
    OneHotDiscreteTensorSpec,
)
from torchrl.modules import (
    ActorValueOperator,
    CEMPlanner,
    LSTMNet,
    ProbabilisticActor,
    QValueActor,
    TensorDictModule,
    ValueOperator,
)
from torchrl.modules.models import ConvNet, MLP, NoisyLazyLinear, NoisyLinear
from torchrl.modules.models.model_based import (
    DreamerActor,
    ObsDecoder,
    ObsEncoder,
    RSSMPosterior,
    RSSMPrior,
    RSSMRollout,
)
from torchrl.modules.models.utils import SquashDims


@pytest.fixture
def double_prec_fixture():
    dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)
    yield
    torch.set_default_dtype(dtype)


@pytest.mark.parametrize("in_features", [3, 10, None])
@pytest.mark.parametrize("out_features", [3, (3, 10)])
@pytest.mark.parametrize("depth, num_cells", [(3, 32), (None, (32, 32, 32))])
@pytest.mark.parametrize(
    "activation_class, activation_kwargs",
    [(nn.ReLU, {"inplace": True}), (nn.ReLU, {}), (nn.PReLU, {})],
)
@pytest.mark.parametrize(
    "norm_class, norm_kwargs",
    [(nn.LazyBatchNorm1d, {}), (nn.BatchNorm1d, {"num_features": 32})],
)
@pytest.mark.parametrize("bias_last_layer", [True, False])
@pytest.mark.parametrize("single_bias_last_layer", [True, False])
@pytest.mark.parametrize("layer_class", [nn.Linear, NoisyLinear])
@pytest.mark.parametrize("device", get_available_devices())
def test_mlp(
    in_features,
    out_features,
    depth,
    num_cells,
    activation_class,
    activation_kwargs,
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
        bias_last_layer=bias_last_layer,
        single_bias_last_layer=False,
        layer_class=layer_class,
        device=device,
    )
    if in_features is None:
        in_features = 5
    x = torch.randn(batch, in_features, device=device)
    y = mlp(x)
    out_features = [out_features] if isinstance(out_features, Number) else out_features
    assert y.shape == torch.Size([batch, *out_features])


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
@pytest.mark.parametrize("device", get_available_devices())
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


@pytest.mark.parametrize(
    "layer_class",
    [
        NoisyLinear,
        NoisyLazyLinear,
    ],
)
@pytest.mark.parametrize("device", get_available_devices())
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


@pytest.mark.parametrize("device", get_available_devices())
def test_value_based_policy(device):
    torch.manual_seed(0)
    obs_dim = 4
    action_dim = 5
    action_spec = OneHotDiscreteTensorSpec(action_dim)

    def make_net():
        net = MLP(in_features=obs_dim, out_features=action_dim, depth=2, device=device)
        for mod in net.modules():
            if hasattr(mod, "bias") and mod.bias is not None:
                mod.bias.data.zero_()
        return net

    actor = QValueActor(spec=action_spec, module=make_net(), safe=True)
    obs = torch.zeros(2, obs_dim, device=device)
    td = TensorDict(batch_size=[2], source={"observation": obs})
    action = actor(td).get("action")
    assert (action.sum(-1) == 1).all()

    actor = QValueActor(spec=action_spec, module=make_net(), safe=False)
    obs = torch.randn(2, obs_dim, device=device)
    td = TensorDict(batch_size=[2], source={"observation": obs})
    action = actor(td).get("action")
    assert (action.sum(-1) == 1).all()

    actor = QValueActor(spec=action_spec, module=make_net(), safe=False)
    obs = torch.zeros(2, obs_dim, device=device)
    td = TensorDict(batch_size=[2], source={"observation": obs})
    action = actor(td).get("action")
    with pytest.raises(AssertionError):
        assert (action.sum(-1) == 1).all()


@pytest.mark.parametrize("device", get_available_devices())
def test_value_based_policy_categorical(device):
    torch.manual_seed(0)
    obs_dim = 4
    action_dim = 5
    action_spec = DiscreteTensorSpec(action_dim)

    def make_net():
        net = MLP(in_features=obs_dim, out_features=action_dim, depth=2, device=device)
        for mod in net.modules():
            if hasattr(mod, "bias") and mod.bias is not None:
                mod.bias.data.zero_()
        return net

    actor = QValueActor(
        spec=action_spec, module=make_net(), safe=True, action_space="categorical"
    )
    obs = torch.zeros(2, obs_dim, device=device)
    td = TensorDict(batch_size=[2], source={"observation": obs})
    action = actor(td).get("action")
    assert (0 <= action).all() and (action < action_dim).all()

    actor = QValueActor(
        spec=action_spec, module=make_net(), safe=False, action_space="categorical"
    )
    obs = torch.randn(2, obs_dim, device=device)
    td = TensorDict(batch_size=[2], source={"observation": obs})
    action = actor(td).get("action")
    assert (0 <= action).all() and (action < action_dim).all()


@pytest.mark.parametrize("device", get_available_devices())
def test_actorcritic(device):
    common_module = TensorDictModule(
        spec=None, module=nn.Linear(3, 4), in_keys=["obs"], out_keys=["hidden"]
    ).to(device)
    module = TensorDictModule(nn.Linear(4, 5), in_keys=["hidden"], out_keys=["param"])
    policy_operator = ProbabilisticActor(
        spec=None, module=module, dist_in_keys=["param"], return_log_prob=True
    ).to(device)
    value_operator = ValueOperator(nn.Linear(4, 1), in_keys=["hidden"]).to(device)
    op = ActorValueOperator(
        common_operator=common_module,
        policy_operator=policy_operator,
        value_operator=value_operator,
    ).to(device)
    td = TensorDict(
        source={"obs": torch.randn(4, 3)},
        batch_size=[
            4,
        ],
    ).to(device)
    td_total = op(td.clone())
    policy_op = op.get_policy_operator()
    td_policy = policy_op(td.clone())
    value_op = op.get_value_operator()
    td_value = value_op(td)
    torch.testing.assert_close(td_total.get("action"), td_policy.get("action"))
    torch.testing.assert_close(
        td_total.get("sample_log_prob"), td_policy.get("sample_log_prob")
    )
    torch.testing.assert_close(td_total.get("state_value"), td_value.get("state_value"))

    value_params = set(
        list(op.get_value_operator().parameters()) + list(op.module[0].parameters())
    )
    value_params2 = set(value_op.parameters())
    assert len(value_params.difference(value_params2)) == 0 and len(
        value_params.intersection(value_params2)
    ) == len(value_params)

    policy_params = set(
        list(op.get_policy_operator().parameters()) + list(op.module[0].parameters())
    )
    policy_params2 = set(policy_op.parameters())
    assert len(policy_params.difference(policy_params2)) == 0 and len(
        policy_params.intersection(policy_params2)
    ) == len(policy_params)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("out_features", [3, 4])
@pytest.mark.parametrize("hidden_size", [8, 9])
@pytest.mark.parametrize("num_layers", [1, 2])
@pytest.mark.parametrize("has_precond_hidden", [True, False])
def test_lstm_net(
    device,
    out_features,
    hidden_size,
    num_layers,
    has_precond_hidden,
    double_prec_fixture,
):

    torch.manual_seed(0)
    batch = 5
    time_steps = 6
    in_features = 7
    net = LSTMNet(
        out_features,
        {
            "input_size": hidden_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
        },
        {"out_features": hidden_size},
        device=device,
    )
    # test single step vs multi-step
    x = torch.randn(batch, time_steps, in_features, device=device)
    x_unbind = x.unbind(1)
    tds_loop = []
    if has_precond_hidden:
        hidden0_out0, hidden1_out0 = torch.randn(
            2, batch, time_steps, num_layers, hidden_size, device=device
        )
        hidden0_out0[:, 1:] = 0.0
        hidden1_out0[:, 1:] = 0.0
        hidden0_out = hidden0_out0[:, 0]
        hidden1_out = hidden1_out0[:, 0]
    else:
        hidden0_out, hidden1_out = None, None
        hidden0_out0, hidden1_out0 = None, None

    for _x in x_unbind:
        y, hidden0_in, hidden1_in, hidden0_out, hidden1_out = net(
            _x, hidden0_out, hidden1_out
        )
        td = TensorDict(
            {
                "y": y,
                "hidden0_in": hidden0_in,
                "hidden1_in": hidden1_in,
                "hidden0_out": hidden0_out,
                "hidden1_out": hidden1_out,
            },
            [batch],
        )
        tds_loop.append(td)
    tds_loop = torch.stack(tds_loop, 1)

    y, hidden0_in, hidden1_in, hidden0_out, hidden1_out = net(
        x, hidden0_out0, hidden1_out0
    )
    tds_vec = TensorDict(
        {
            "y": y,
            "hidden0_in": hidden0_in,
            "hidden1_in": hidden1_in,
            "hidden0_out": hidden0_out,
            "hidden1_out": hidden1_out,
        },
        [batch, time_steps],
    )
    torch.testing.assert_close(tds_vec["y"], tds_loop["y"])
    torch.testing.assert_close(
        tds_vec["hidden0_out"][:, -1], tds_loop["hidden0_out"][:, -1]
    )
    torch.testing.assert_close(
        tds_vec["hidden1_out"][:, -1], tds_loop["hidden1_out"][:, -1]
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("out_features", [3, 5])
@pytest.mark.parametrize("hidden_size", [3, 5])
def test_lstm_net_nobatch(device, out_features, hidden_size):
    time_steps = 6
    in_features = 4
    net = LSTMNet(
        out_features,
        {"input_size": hidden_size, "hidden_size": hidden_size},
        {"out_features": hidden_size},
        device=device,
    )
    # test single step vs multi-step
    x = torch.randn(time_steps, in_features, device=device)
    x_unbind = x.unbind(0)
    tds_loop = []
    hidden0_in, hidden1_in, hidden0_out, hidden1_out = [
        None,
    ] * 4
    for _x in x_unbind:
        y, hidden0_in, hidden1_in, hidden0_out, hidden1_out = net(
            _x, hidden0_out, hidden1_out
        )
        td = TensorDict(
            {
                "y": y,
                "hidden0_in": hidden0_in,
                "hidden1_in": hidden1_in,
                "hidden0_out": hidden0_out,
                "hidden1_out": hidden1_out,
            },
            [],
        )
        tds_loop.append(td)
    tds_loop = torch.stack(tds_loop, 0)

    y, hidden0_in, hidden1_in, hidden0_out, hidden1_out = net(x.unsqueeze(0))
    tds_vec = TensorDict(
        {
            "y": y,
            "hidden0_in": hidden0_in,
            "hidden1_in": hidden1_in,
            "hidden0_out": hidden0_out,
            "hidden1_out": hidden1_out,
        },
        [1, time_steps],
    ).squeeze(0)
    torch.testing.assert_close(tds_vec["y"], tds_loop["y"])
    torch.testing.assert_close(tds_vec["hidden0_out"][-1], tds_loop["hidden0_out"][-1])
    torch.testing.assert_close(tds_vec["hidden1_out"][-1], tds_loop["hidden1_out"][-1])


class TestPlanner:
    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("batch_size", [3, 5])
    def test_CEM_model_free_env(self, device, batch_size, seed=1):
        env = MockBatchedUnLockedEnv(device=device)
        torch.manual_seed(seed)
        planner = CEMPlanner(
            env,
            planning_horizon=10,
            optim_steps=2,
            num_candidates=100,
            num_top_k_candidates=2,
        )
        td = env.reset(TensorDict({}, batch_size=batch_size).to(device))
        td_copy = td.clone()
        td = planner(td)
        assert td.get("action").shape[1:] == env.action_spec.shape

        assert env.action_spec.is_in(td.get("action"))

        for key in td.keys():
            if key != "action":
                assert torch.allclose(td[key], td_copy[key])


@pytest.mark.parametrize("device", get_available_devices())
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
        encoder = ObsEncoder(depth=depth).to(device)
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
        decoder = ObsDecoder(depth=depth).to(device)
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
        action_spec = NdBoundedTensorSpec(
            shape=(action_size,), dtype=torch.float32, minimum=-1, maximum=1
        )
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
        action_spec = NdBoundedTensorSpec(
            shape=(action_size,), dtype=torch.float32, minimum=-1, maximum=1
        )
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
            TensorDictModule(
                rssm_prior,
                in_keys=["state", "belief", "action"],
                out_keys=[
                    ("next", "prior_mean"),
                    ("next", "prior_std"),
                    "_",
                    ("next", "belief"),
                ],
            ),
            TensorDictModule(
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


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
