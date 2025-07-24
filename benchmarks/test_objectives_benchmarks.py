# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse

import pytest
import torch
from packaging import version

from tensordict import TensorDict
from tensordict.nn import (
    composite_lp_aggregate,
    InteractionType,
    NormalParamExtractor,
    ProbabilisticTensorDictModule as ProbMod,
    ProbabilisticTensorDictSequential as ProbSeq,
    TensorDictModule as Mod,
    TensorDictSequential as Seq,
)
from torch.nn import functional as F
from torchrl.data.tensor_specs import Bounded, Unbounded
from torchrl.modules import MLP, QValueActor, TanhNormal
from torchrl.objectives import (
    A2CLoss,
    ClipPPOLoss,
    CQLLoss,
    DDPGLoss,
    DQNLoss,
    IQLLoss,
    REDQLoss,
    ReinforceLoss,
    SACLoss,
    TD3Loss,
)
from torchrl.objectives.deprecated import REDQLoss_deprecated
from torchrl.objectives.value import GAE
from torchrl.objectives.value.functional import (
    generalized_advantage_estimate,
    td0_return_estimate,
    td1_return_estimate,
    td_lambda_return_estimate,
    vec_generalized_advantage_estimate,
    vec_td1_return_estimate,
    vec_td_lambda_return_estimate,
)

TORCH_VERSION = torch.__version__
FULLGRAPH = version.parse(".".join(TORCH_VERSION.split(".")[:3])) >= version.parse(
    "2.5.0"
)  # Anything from 2.5, incl. nightlies, allows for fullgraph


# @pytest.fixture(scope="module", autouse=True)
# def set_default_device():
#     cur_device = getattr(torch, "get_default_device", lambda: torch.device("cpu"))()
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     torch.set_default_device(device)
#     yield
#     torch.set_default_device(cur_device)


class setup_value_fn:
    def __init__(self, has_lmbda, has_state_value):
        self.has_lmbda = has_lmbda
        self.has_state_value = has_state_value

    def __call__(
        self,
        b=300,
        t=500,
        d=1,
        gamma=0.95,
        lmbda=0.95,
    ):
        torch.manual_seed(0)
        device = "cuda:0" if torch.cuda.device_count() else "cpu"
        values = torch.randn(b, t, d, device=device)
        next_values = torch.randn(b, t, d, device=device)
        reward = torch.randn(b, t, d, device=device)
        done = torch.zeros(b, t, d, dtype=torch.bool, device=device).bernoulli_(0.1)
        kwargs = {
            "gamma": gamma,
            "next_state_value": next_values,
            "reward": reward,
            "done": done,
        }
        if self.has_lmbda:
            kwargs["lmbda"] = lmbda

        if self.has_state_value:
            kwargs["state_value"] = values

        return ((), kwargs)


@pytest.mark.parametrize(
    "val_fn,has_lmbda,has_state_value",
    [
        [generalized_advantage_estimate, True, True],
        [vec_generalized_advantage_estimate, True, True],
        [td0_return_estimate, False, False],
        [td1_return_estimate, False, False],
        [vec_td1_return_estimate, False, False],
        [td_lambda_return_estimate, True, False],
        [vec_td_lambda_return_estimate, True, False],
    ],
)
def test_values(benchmark, val_fn, has_lmbda, has_state_value):
    benchmark.pedantic(
        val_fn,
        setup=setup_value_fn(
            has_lmbda=has_lmbda,
            has_state_value=has_state_value,
        ),
        iterations=1,
        rounds=50,
    )


@pytest.mark.parametrize(
    "gae_fn,gamma_tensor,batches,timesteps",
    [
        [generalized_advantage_estimate, False, 1, 512],
        [vec_generalized_advantage_estimate, True, 1, 512],
        [vec_generalized_advantage_estimate, False, 1, 512],
        [vec_generalized_advantage_estimate, True, 32, 512],
        [vec_generalized_advantage_estimate, False, 32, 512],
    ],
)
def test_gae_speed(benchmark, gae_fn, gamma_tensor, batches, timesteps):
    size = (batches, timesteps, 1)

    torch.manual_seed(0)
    device = "cuda:0" if torch.cuda.device_count() else "cpu"
    values = torch.randn(*size, device=device)
    next_values = torch.randn(*size, device=device)
    reward = torch.randn(*size, device=device)
    done = torch.zeros(*size, dtype=torch.bool, device=device).bernoulli_(0.1)

    gamma = 0.99
    if gamma_tensor:
        gamma = torch.full(size, gamma, device=device)
    lmbda = 0.95

    benchmark(
        gae_fn,
        gamma=gamma,
        lmbda=lmbda,
        state_value=values,
        next_state_value=next_values,
        reward=reward,
        done=done,
    )


def _maybe_compile(fn, compile, td, fullgraph=FULLGRAPH, warmup=3):
    if compile:
        if isinstance(compile, str):
            fn = torch.compile(fn, mode=compile, fullgraph=fullgraph)
        else:
            fn = torch.compile(fn, fullgraph=fullgraph)

        for _ in range(warmup):
            fn(td)

    return fn


@pytest.mark.parametrize("backward", [None, "backward"])
@pytest.mark.parametrize("compile", [False, True, "reduce-overhead"])
def test_dqn_speed(
    benchmark, backward, compile, n_obs=8, n_act=4, depth=3, ncells=128, batch=128
):
    if compile:
        torch._dynamo.reset_code_caches()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = MLP(
        in_features=n_obs,
        out_features=n_act,
        depth=depth,
        num_cells=ncells,
        device=device,
    )
    action_space = "one-hot"
    mod = QValueActor(net, in_keys=["obs"], action_space=action_space)
    loss = DQNLoss(value_network=mod, action_space=action_space)
    td = TensorDict(
        {
            "obs": torch.randn(batch, n_obs),
            "action": F.one_hot(torch.randint(n_act, (batch,))),
            "next": {
                "obs": torch.randn(batch, n_obs),
                "done": torch.zeros(batch, 1, dtype=torch.bool),
                "reward": torch.randn(batch, 1),
            },
        },
        [batch],
        device=device,
    )
    loss(td)

    loss = _maybe_compile(loss, compile, td)

    if backward:

        def loss_and_bw(td):
            losses = loss(td)
            sum(
                [val for key, val in losses.items() if key.startswith("loss")]
            ).backward()

        benchmark.pedantic(
            loss_and_bw,
            args=(td,),
            setup=loss.zero_grad,
            iterations=1,
            warmup_rounds=5,
            rounds=50,
        )
    else:
        benchmark(loss, td)


@pytest.mark.parametrize("backward", [None, "backward"])
@pytest.mark.parametrize("compile", [False, True, "reduce-overhead"])
def test_ddpg_speed(
    benchmark, backward, compile, n_obs=8, n_act=4, ncells=128, batch=128, n_hidden=64
):
    if compile:
        torch._dynamo.reset_code_caches()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    common = MLP(
        num_cells=ncells,
        in_features=n_obs,
        depth=3,
        out_features=n_hidden,
        device=device,
    )
    actor = MLP(
        num_cells=ncells,
        in_features=n_hidden,
        depth=2,
        out_features=n_act,
        device=device,
    )
    value = MLP(
        in_features=n_hidden + n_act,
        num_cells=ncells,
        depth=2,
        out_features=1,
        device=device,
    )
    batch = [batch]
    td = TensorDict(
        {
            "obs": torch.randn(*batch, n_obs),
            "action": torch.randn(*batch, n_act),
            "done": torch.zeros(*batch, 1, dtype=torch.bool),
            "next": {
                "obs": torch.randn(*batch, n_obs),
                "reward": torch.randn(*batch, 1),
                "done": torch.zeros(*batch, 1, dtype=torch.bool),
            },
        },
        batch,
        device=device,
    )
    common = Mod(common, in_keys=["obs"], out_keys=["hidden"])
    actor_head = Mod(actor, in_keys=["hidden"], out_keys=["action"])
    actor = Seq(common, actor_head)
    value = Mod(value, in_keys=["hidden", "action"], out_keys=["state_action_value"])
    value(actor(td))

    loss = DDPGLoss(actor, value)

    loss(td)

    loss = _maybe_compile(loss, compile, td)

    if backward:

        def loss_and_bw(td):
            losses = loss(td)
            sum(
                [val for key, val in losses.items() if key.startswith("loss")]
            ).backward()

        benchmark.pedantic(
            loss_and_bw,
            args=(td,),
            setup=loss.zero_grad,
            iterations=1,
            warmup_rounds=5,
            rounds=50,
        )
    else:
        benchmark(loss, td)


@pytest.mark.parametrize("backward", [None, "backward"])
@pytest.mark.parametrize("compile", [False, True, "reduce-overhead"])
def test_sac_speed(
    benchmark, backward, compile, n_obs=8, n_act=4, ncells=128, batch=128, n_hidden=64
):
    if compile:
        torch._dynamo.reset_code_caches()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    common = MLP(
        num_cells=ncells,
        in_features=n_obs,
        depth=3,
        out_features=n_hidden,
        device=device,
    )
    actor_net = MLP(
        num_cells=ncells,
        in_features=n_hidden,
        depth=2,
        out_features=2 * n_act,
        device=device,
    )
    value = MLP(
        in_features=n_hidden + n_act,
        num_cells=ncells,
        depth=2,
        out_features=1,
        device=device,
    )
    batch = [batch]
    td = TensorDict(
        {
            "obs": torch.randn(*batch, n_obs),
            "action": torch.randn(*batch, n_act),
            "done": torch.zeros(*batch, 1, dtype=torch.bool),
            "next": {
                "obs": torch.randn(*batch, n_obs),
                "reward": torch.randn(*batch, 1),
                "done": torch.zeros(*batch, 1, dtype=torch.bool),
            },
        },
        batch,
        device=device,
    )
    common = Mod(common, in_keys=["obs"], out_keys=["hidden"])
    actor = ProbSeq(
        common,
        Mod(actor_net, in_keys=["hidden"], out_keys=["param"]),
        Mod(NormalParamExtractor(), in_keys=["param"], out_keys=["loc", "scale"]),
        ProbMod(
            in_keys=["loc", "scale"],
            out_keys=["action"],
            distribution_class=TanhNormal,
            distribution_kwargs={"safe_tanh": False},
        ),
    )
    value_head = Mod(
        value, in_keys=["hidden", "action"], out_keys=["state_action_value"]
    )
    value = Seq(common, value_head)
    value(actor(td.clone()))

    loss = SACLoss(actor, value, action_spec=Unbounded(shape=(n_act,)))

    loss(td)

    loss = _maybe_compile(loss, compile, td)

    if backward:

        def loss_and_bw(td):
            losses = loss(td)
            sum(
                [val for key, val in losses.items() if key.startswith("loss")]
            ).backward()

        benchmark.pedantic(
            loss_and_bw,
            args=(td,),
            setup=loss.zero_grad,
            iterations=1,
            warmup_rounds=5,
            rounds=50,
        )
    else:
        benchmark(loss, td)


# FIXME: fix this
@pytest.mark.skipif(torch.cuda.is_available(), reason="Currently fails on GPU")
@pytest.mark.parametrize("backward", [None, "backward"])
@pytest.mark.parametrize("compile", [False, True, "reduce-overhead"])
def test_redq_speed(
    benchmark, backward, compile, n_obs=8, n_act=4, ncells=128, batch=128, n_hidden=64
):
    if compile:
        torch._dynamo.reset_code_caches()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    common = MLP(
        num_cells=ncells,
        in_features=n_obs,
        depth=3,
        out_features=n_hidden,
        device=device,
    )
    actor_net = MLP(
        num_cells=ncells,
        in_features=n_hidden,
        depth=2,
        out_features=2 * n_act,
        device=device,
    )
    value = MLP(
        in_features=n_hidden + n_act,
        num_cells=ncells,
        depth=2,
        out_features=1,
        device=device,
    )
    batch = [batch]
    td = TensorDict(
        {
            "obs": torch.randn(*batch, n_obs),
            "action": torch.randn(*batch, n_act),
            "done": torch.zeros(*batch, 1, dtype=torch.bool),
            "next": {
                "obs": torch.randn(*batch, n_obs),
                "reward": torch.randn(*batch, 1),
                "done": torch.zeros(*batch, 1, dtype=torch.bool),
            },
        },
        batch,
        device=device,
    )
    common = Mod(common, in_keys=["obs"], out_keys=["hidden"])
    actor = ProbSeq(
        common,
        Mod(actor_net, in_keys=["hidden"], out_keys=["param"]),
        Mod(NormalParamExtractor(), in_keys=["param"], out_keys=["loc", "scale"]),
        ProbMod(
            in_keys=["loc", "scale"],
            out_keys=["action"],
            distribution_class=TanhNormal,
            return_log_prob=True,
            distribution_kwargs={"safe_tanh": False},
        ),
    )
    value_head = Mod(
        value, in_keys=["hidden", "action"], out_keys=["state_action_value"]
    )
    value = Seq(common, value_head)
    value(actor(td.copy()))

    loss = REDQLoss(actor, value, action_spec=Unbounded(shape=(n_act,)))

    loss(td)
    loss = _maybe_compile(loss, compile, td)

    if backward:

        def loss_and_bw(td):
            losses = loss(td)
            totalloss = sum(
                [val for key, val in losses.items() if key.startswith("loss")]
            )
            totalloss.backward()

        loss_and_bw(td)

        benchmark.pedantic(
            loss_and_bw,
            args=(td,),
            setup=loss.zero_grad,
            iterations=1,
            warmup_rounds=5,
            rounds=50,
        )
    else:
        benchmark(loss, td)


@pytest.mark.parametrize("backward", [None, "backward"])
@pytest.mark.parametrize("compile", [False, True, "reduce-overhead"])
def test_redq_deprec_speed(
    benchmark, backward, compile, n_obs=8, n_act=4, ncells=128, batch=128, n_hidden=64
):
    if compile:
        torch._dynamo.reset_code_caches()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    common = MLP(
        num_cells=ncells,
        in_features=n_obs,
        depth=3,
        out_features=n_hidden,
        device=device,
    )
    actor_net = MLP(
        num_cells=ncells,
        in_features=n_hidden,
        depth=2,
        out_features=2 * n_act,
        device=device,
    )
    value = MLP(
        in_features=n_hidden + n_act,
        num_cells=ncells,
        depth=2,
        out_features=1,
        device=device,
    )
    batch = [batch]
    td = TensorDict(
        {
            "obs": torch.randn(*batch, n_obs),
            "action": torch.randn(*batch, n_act),
            "done": torch.zeros(*batch, 1, dtype=torch.bool),
            "next": {
                "obs": torch.randn(*batch, n_obs),
                "reward": torch.randn(*batch, 1),
                "done": torch.zeros(*batch, 1, dtype=torch.bool),
            },
        },
        batch,
        device=device,
    )
    common = Mod(common, in_keys=["obs"], out_keys=["hidden"])
    actor = ProbSeq(
        common,
        Mod(actor_net, in_keys=["hidden"], out_keys=["param"]),
        Mod(NormalParamExtractor(), in_keys=["param"], out_keys=["loc", "scale"]),
        ProbMod(
            in_keys=["loc", "scale"],
            out_keys=["action"],
            distribution_class=TanhNormal,
            return_log_prob=True,
            distribution_kwargs={"safe_tanh": False},
        ),
    )
    value_head = Mod(
        value, in_keys=["hidden", "action"], out_keys=["state_action_value"]
    )
    value = Seq(common, value_head)
    value(actor(td.copy()))

    loss = REDQLoss_deprecated(actor, value, action_spec=Unbounded(shape=(n_act,)))

    loss(td)

    loss = _maybe_compile(loss, compile, td)

    if backward:

        def loss_and_bw(td):
            losses = loss(td)
            sum(
                [val for key, val in losses.items() if key.startswith("loss")]
            ).backward()

        benchmark.pedantic(
            loss_and_bw,
            args=(td,),
            setup=loss.zero_grad,
            iterations=1,
            warmup_rounds=5,
            rounds=50,
        )
    else:
        benchmark(loss, td)


@pytest.mark.parametrize("backward", [None, "backward"])
@pytest.mark.parametrize("compile", [False, True, "reduce-overhead"])
def test_td3_speed(
    benchmark, backward, compile, n_obs=8, n_act=4, ncells=128, batch=128, n_hidden=64
):
    if compile:
        torch._dynamo.reset_code_caches()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    common = MLP(
        num_cells=ncells,
        in_features=n_obs,
        depth=3,
        out_features=n_hidden,
        device=device,
    )
    actor_net = MLP(
        num_cells=ncells,
        in_features=n_hidden,
        depth=2,
        out_features=2 * n_act,
        device=device,
    )
    value = MLP(
        in_features=n_hidden + n_act,
        num_cells=ncells,
        depth=2,
        out_features=1,
        device=device,
    )
    batch = [batch]
    td = TensorDict(
        {
            "obs": torch.randn(*batch, n_obs),
            "action": torch.randn(*batch, n_act),
            "done": torch.zeros(*batch, 1, dtype=torch.bool),
            "next": {
                "obs": torch.randn(*batch, n_obs),
                "reward": torch.randn(*batch, 1),
                "done": torch.zeros(*batch, 1, dtype=torch.bool),
            },
        },
        batch,
        device=device,
    )
    common = Mod(common, in_keys=["obs"], out_keys=["hidden"])
    actor = ProbSeq(
        common,
        Mod(actor_net, in_keys=["hidden"], out_keys=["param"]),
        Mod(NormalParamExtractor(), in_keys=["param"], out_keys=["loc", "scale"]),
        ProbMod(
            in_keys=["loc", "scale"],
            out_keys=["action"],
            distribution_class=TanhNormal,
            distribution_kwargs={"safe_tanh": False},
            return_log_prob=True,
            default_interaction_type=InteractionType.DETERMINISTIC,
        ),
    )
    value_head = Mod(
        value, in_keys=["hidden", "action"], out_keys=["state_action_value"]
    )
    value = Seq(common, value_head)
    value(actor(td.clone()))

    loss = TD3Loss(
        actor,
        value,
        action_spec=Bounded(shape=(n_act,), low=-1, high=1),
    )

    loss(td)

    loss = _maybe_compile(loss, compile, td)

    if backward:

        def loss_and_bw(td):
            losses = loss(td)
            sum(
                [val for key, val in losses.items() if key.startswith("loss")]
            ).backward()

        benchmark.pedantic(
            loss_and_bw,
            args=(td,),
            setup=loss.zero_grad,
            iterations=1,
            warmup_rounds=5,
            rounds=50,
        )
    else:
        benchmark.pedantic(loss, args=(td,), rounds=100, iterations=10)


@pytest.mark.parametrize("backward", [None, "backward"])
@pytest.mark.parametrize("compile", [False, True, "reduce-overhead"])
def test_cql_speed(
    benchmark, backward, compile, n_obs=8, n_act=4, ncells=128, batch=128, n_hidden=64
):
    if compile:
        torch._dynamo.reset_code_caches()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    common = MLP(
        num_cells=ncells,
        in_features=n_obs,
        depth=3,
        out_features=n_hidden,
        device=device,
    )
    actor_net = MLP(
        num_cells=ncells,
        in_features=n_hidden,
        depth=2,
        out_features=2 * n_act,
        device=device,
    )
    value = MLP(
        in_features=n_hidden + n_act,
        num_cells=ncells,
        depth=2,
        out_features=1,
        device=device,
    )
    batch = [batch]
    td = TensorDict(
        {
            "obs": torch.randn(*batch, n_obs),
            "action": torch.randn(*batch, n_act),
            "done": torch.zeros(*batch, 1, dtype=torch.bool),
            "next": {
                "obs": torch.randn(*batch, n_obs),
                "reward": torch.randn(*batch, 1),
                "done": torch.zeros(*batch, 1, dtype=torch.bool),
            },
        },
        batch,
        device=device,
    )
    common = Mod(common, in_keys=["obs"], out_keys=["hidden"])
    actor = ProbSeq(
        common,
        Mod(actor_net, in_keys=["hidden"], out_keys=["param"]),
        Mod(NormalParamExtractor(), in_keys=["param"], out_keys=["loc", "scale"]),
        ProbMod(
            in_keys=["loc", "scale"],
            out_keys=["action"],
            distribution_class=TanhNormal,
            distribution_kwargs={"safe_tanh": False},
        ),
    )
    value_head = Mod(
        value, in_keys=["hidden", "action"], out_keys=["state_action_value"]
    )
    value = Seq(common, value_head)
    value(actor(td.copy()))

    loss = CQLLoss(actor, value, action_spec=Unbounded(shape=(n_act,)))

    loss(td)

    loss = _maybe_compile(loss, compile, td)

    if backward:

        def loss_and_bw(td):
            losses = loss(td)
            sum(
                [val for key, val in losses.items() if key.startswith("loss")]
            ).backward()

        benchmark.pedantic(
            loss_and_bw,
            args=(td,),
            setup=loss.zero_grad,
            iterations=1,
            warmup_rounds=5,
            rounds=50,
        )
    else:
        benchmark(loss, td)


@pytest.mark.parametrize("backward", [None, "backward"])
@pytest.mark.parametrize("compile", [False, True, "reduce-overhead"])
def test_a2c_speed(
    benchmark,
    backward,
    compile,
    n_obs=8,
    n_act=4,
    n_hidden=64,
    ncells=128,
    batch=128,
    T=10,
):
    if compile:
        torch._dynamo.reset_code_caches()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    common_net = MLP(
        num_cells=ncells,
        in_features=n_obs,
        depth=3,
        out_features=n_hidden,
        device=device,
    )
    actor_net = MLP(
        num_cells=ncells,
        in_features=n_hidden,
        depth=2,
        out_features=2 * n_act,
        device=device,
    )
    value_net = MLP(
        in_features=n_hidden,
        num_cells=ncells,
        depth=2,
        out_features=1,
        device=device,
    )
    batch = [batch, T]
    if composite_lp_aggregate():
        raise RuntimeError(
            "Expected composite_lp_aggregate() to return False. Use set_composite_lp_aggregate or COMPOSITE_LP_AGGREGATE env variable."
        )
    td = TensorDict(
        {
            "obs": torch.randn(*batch, n_obs),
            "action": torch.randn(*batch, n_act),
            "action_log_prob": torch.randn(*batch),
            "done": torch.zeros(*batch, 1, dtype=torch.bool),
            "next": {
                "obs": torch.randn(*batch, n_obs),
                "reward": torch.randn(*batch, 1),
                "done": torch.zeros(*batch, 1, dtype=torch.bool),
            },
        },
        batch,
        names=[None, "time"],
        device=device,
    )
    common = Mod(common_net, in_keys=["obs"], out_keys=["hidden"])
    actor = ProbSeq(
        common,
        Mod(actor_net, in_keys=["hidden"], out_keys=["param"]),
        Mod(NormalParamExtractor(), in_keys=["param"], out_keys=["loc", "scale"]),
        ProbMod(
            in_keys=["loc", "scale"],
            out_keys=["action"],
            distribution_class=TanhNormal,
            distribution_kwargs={"safe_tanh": False},
        ),
    )
    critic = Seq(common, Mod(value_net, in_keys=["hidden"], out_keys=["state_value"]))
    actor(td.clone())
    critic(td.clone())

    loss = A2CLoss(actor_network=actor, critic_network=critic)
    advantage = GAE(
        value_network=critic, gamma=0.99, lmbda=0.95, shifted=True, device=device
    )
    advantage(td)
    loss(td)

    loss = _maybe_compile(loss, compile, td)

    if backward:

        def loss_and_bw(td):
            losses = loss(td)
            sum(
                [val for key, val in losses.items() if key.startswith("loss")]
            ).backward()

        benchmark.pedantic(
            loss_and_bw,
            args=(td,),
            setup=loss.zero_grad,
            iterations=1,
            warmup_rounds=5,
            rounds=50,
        )
    else:
        benchmark(loss, td)


@pytest.mark.parametrize("backward", [None, "backward"])
@pytest.mark.parametrize("compile", [False, True, "reduce-overhead"])
def test_ppo_speed(
    benchmark,
    backward,
    compile,
    n_obs=8,
    n_act=4,
    n_hidden=64,
    ncells=128,
    batch=128,
    T=10,
):
    if compile:
        torch._dynamo.reset_code_caches()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    common_net = MLP(
        num_cells=ncells,
        in_features=n_obs,
        depth=3,
        out_features=n_hidden,
        device=device,
    )
    actor_net = MLP(
        num_cells=ncells,
        in_features=n_hidden,
        depth=2,
        out_features=2 * n_act,
        device=device,
    )
    value_net = MLP(
        in_features=n_hidden,
        num_cells=ncells,
        depth=2,
        out_features=1,
        device=device,
    )
    batch = [batch, T]
    if composite_lp_aggregate():
        raise RuntimeError(
            "Expected composite_lp_aggregate() to return False. Use set_composite_lp_aggregate or COMPOSITE_LP_AGGREGATE env variable."
        )
    td = TensorDict(
        {
            "obs": torch.randn(*batch, n_obs),
            "action": torch.randn(*batch, n_act),
            "action_log_prob": torch.randn(*batch),
            "done": torch.zeros(*batch, 1, dtype=torch.bool),
            "next": {
                "obs": torch.randn(*batch, n_obs),
                "reward": torch.randn(*batch, 1),
                "done": torch.zeros(*batch, 1, dtype=torch.bool),
            },
        },
        batch,
        names=[None, "time"],
        device=device,
    )
    common = Mod(common_net, in_keys=["obs"], out_keys=["hidden"])
    actor = ProbSeq(
        common,
        Mod(actor_net, in_keys=["hidden"], out_keys=["param"]),
        Mod(NormalParamExtractor(), in_keys=["param"], out_keys=["loc", "scale"]),
        ProbMod(
            in_keys=["loc", "scale"],
            out_keys=["action"],
            distribution_class=TanhNormal,
            distribution_kwargs={"safe_tanh": False},
        ),
    )
    critic = Seq(common, Mod(value_net, in_keys=["hidden"], out_keys=["state_value"]))
    actor(td.clone())
    critic(td.clone())

    loss = ClipPPOLoss(actor_network=actor, critic_network=critic)
    advantage = GAE(
        value_network=critic, gamma=0.99, lmbda=0.95, shifted=True, device=device
    )
    advantage(td)
    loss(td)

    loss = _maybe_compile(loss, compile, td)

    if backward:

        def loss_and_bw(td):
            losses = loss(td)
            sum(
                [val for key, val in losses.items() if key.startswith("loss")]
            ).backward()

        benchmark.pedantic(
            loss_and_bw,
            args=(td,),
            setup=loss.zero_grad,
            iterations=1,
            warmup_rounds=5,
            rounds=50,
        )
    else:
        benchmark(loss, td)


@pytest.mark.parametrize("backward", [None, "backward"])
@pytest.mark.parametrize("compile", [False, True, "reduce-overhead"])
def test_reinforce_speed(
    benchmark,
    backward,
    compile,
    n_obs=8,
    n_act=4,
    n_hidden=64,
    ncells=128,
    batch=128,
    T=10,
):
    if compile:
        torch._dynamo.reset_code_caches()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    common_net = MLP(
        num_cells=ncells,
        in_features=n_obs,
        depth=3,
        out_features=n_hidden,
        device=device,
    )
    actor_net = MLP(
        num_cells=ncells,
        in_features=n_hidden,
        depth=2,
        out_features=2 * n_act,
        device=device,
    )
    value_net = MLP(
        in_features=n_hidden,
        num_cells=ncells,
        depth=2,
        out_features=1,
        device=device,
    )
    batch = [batch, T]
    if composite_lp_aggregate():
        raise RuntimeError(
            "Expected composite_lp_aggregate() to return False. Use set_composite_lp_aggregate or COMPOSITE_LP_AGGREGATE env variable."
        )
    td = TensorDict(
        {
            "obs": torch.randn(*batch, n_obs),
            "action": torch.randn(*batch, n_act),
            "action_log_prob": torch.randn(*batch),
            "done": torch.zeros(*batch, 1, dtype=torch.bool),
            "next": {
                "obs": torch.randn(*batch, n_obs),
                "reward": torch.randn(*batch, 1),
                "done": torch.zeros(*batch, 1, dtype=torch.bool),
            },
        },
        batch,
        names=[None, "time"],
        device=device,
    )
    common = Mod(common_net, in_keys=["obs"], out_keys=["hidden"])
    actor = ProbSeq(
        common,
        Mod(actor_net, in_keys=["hidden"], out_keys=["param"]),
        Mod(NormalParamExtractor(), in_keys=["param"], out_keys=["loc", "scale"]),
        ProbMod(
            in_keys=["loc", "scale"],
            out_keys=["action"],
            distribution_class=TanhNormal,
            distribution_kwargs={"safe_tanh": False},
        ),
    )
    critic = Seq(common, Mod(value_net, in_keys=["hidden"], out_keys=["state_value"]))
    actor(td.clone())
    critic(td.clone())

    loss = ReinforceLoss(actor_network=actor, critic_network=critic)
    advantage = GAE(
        value_network=critic, gamma=0.99, lmbda=0.95, shifted=True, device=device
    )
    advantage(td)
    loss(td)

    loss = _maybe_compile(loss, compile, td)

    if backward:

        def loss_and_bw(td):
            losses = loss(td)
            sum(
                [val for key, val in losses.items() if key.startswith("loss")]
            ).backward()

        benchmark.pedantic(
            loss_and_bw,
            args=(td,),
            setup=loss.zero_grad,
            iterations=1,
            warmup_rounds=5,
            rounds=50,
        )
    else:
        benchmark(loss, td)


@pytest.mark.parametrize("backward", [None, "backward"])
@pytest.mark.parametrize("compile", [False, True, "reduce-overhead"])
def test_iql_speed(
    benchmark,
    backward,
    compile,
    n_obs=8,
    n_act=4,
    n_hidden=64,
    ncells=128,
    batch=128,
    T=10,
):
    if compile:
        torch._dynamo.reset_code_caches()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    common_net = MLP(
        num_cells=ncells,
        in_features=n_obs,
        depth=3,
        out_features=n_hidden,
        device=device,
    )
    actor_net = MLP(
        num_cells=ncells,
        in_features=n_hidden,
        depth=2,
        out_features=2 * n_act,
        device=device,
    )
    value_net = MLP(
        in_features=n_hidden,
        num_cells=ncells,
        depth=2,
        out_features=1,
        device=device,
    )
    qvalue_net = MLP(
        in_features=n_hidden + n_act,
        num_cells=ncells,
        depth=2,
        out_features=1,
        device=device,
    )
    batch = [batch, T]
    if composite_lp_aggregate():
        raise RuntimeError(
            "Expected composite_lp_aggregate() to return False. Use set_composite_lp_aggregate or COMPOSITE_LP_AGGREGATE env variable."
        )
    td = TensorDict(
        {
            "obs": torch.randn(*batch, n_obs),
            "action": torch.randn(*batch, n_act),
            "action_log_prob": torch.randn(*batch),
            "done": torch.zeros(*batch, 1, dtype=torch.bool),
            "next": {
                "obs": torch.randn(*batch, n_obs),
                "reward": torch.randn(*batch, 1),
                "done": torch.zeros(*batch, 1, dtype=torch.bool),
            },
        },
        batch,
        names=[None, "time"],
        device=device,
    )
    common = Mod(common_net, in_keys=["obs"], out_keys=["hidden"])
    actor = ProbSeq(
        common,
        Mod(actor_net, in_keys=["hidden"], out_keys=["param"]),
        Mod(NormalParamExtractor(), in_keys=["param"], out_keys=["loc", "scale"]),
        ProbMod(
            in_keys=["loc", "scale"],
            out_keys=["action"],
            distribution_class=TanhNormal,
            distribution_kwargs={"safe_tanh": False},
        ),
    )
    value = Seq(common, Mod(value_net, in_keys=["hidden"], out_keys=["state_value"]))
    qvalue = Seq(
        common,
        Mod(qvalue_net, in_keys=["hidden", "action"], out_keys=["state_action_value"]),
    )
    qvalue(actor(td.clone()))
    value(td.clone())

    loss = IQLLoss(actor_network=actor, value_network=value, qvalue_network=qvalue)
    loss(td)

    loss = _maybe_compile(loss, compile, td)

    if backward:

        def loss_and_bw(td):
            losses = loss(td)
            sum(
                [val for key, val in losses.items() if key.startswith("loss")]
            ).backward()

        benchmark.pedantic(
            loss_and_bw,
            args=(td,),
            setup=loss.zero_grad,
            iterations=1,
            warmup_rounds=5,
            rounds=50,
        )
    else:
        benchmark(loss, td)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main(
        [__file__, "--capture", "no", "--exitfirst", "--benchmark-group-by", "func"]
        + unknown
    )
