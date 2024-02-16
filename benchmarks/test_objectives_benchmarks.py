# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse

import pytest
import torch

from tensordict import TensorDict
from tensordict.nn import (
    NormalParamExtractor,
    ProbabilisticTensorDictModule as ProbMod,
    ProbabilisticTensorDictSequential as ProbSeq,
    TensorDictModule as Mod,
    TensorDictSequential as Seq,
)
from torch.nn import functional as F
from torchrl.data.tensor_specs import BoundedTensorSpec, UnboundedContinuousTensorSpec
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


def test_dqn_speed(benchmark, n_obs=8, n_act=4, depth=3, ncells=128, batch=128):
    net = MLP(in_features=n_obs, out_features=n_act, depth=depth, num_cells=ncells)
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
    )
    loss(td)
    benchmark(loss, td)


def test_ddpg_speed(benchmark, n_obs=8, n_act=4, ncells=128, batch=128, n_hidden=64):
    common = MLP(
        num_cells=ncells,
        in_features=n_obs,
        depth=3,
        out_features=n_hidden,
    )
    actor = MLP(
        num_cells=ncells,
        in_features=n_hidden,
        depth=2,
        out_features=n_act,
    )
    value = MLP(
        in_features=n_hidden + n_act,
        num_cells=ncells,
        depth=2,
        out_features=1,
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
    )
    common = Mod(common, in_keys=["obs"], out_keys=["hidden"])
    actor_head = Mod(actor, in_keys=["hidden"], out_keys=["action"])
    actor = Seq(common, actor_head)
    value = Mod(value, in_keys=["hidden", "action"], out_keys=["state_action_value"])
    value(actor(td))

    loss = DDPGLoss(actor, value)

    loss(td)
    benchmark(loss, td)


def test_sac_speed(benchmark, n_obs=8, n_act=4, ncells=128, batch=128, n_hidden=64):
    common = MLP(
        num_cells=ncells,
        in_features=n_obs,
        depth=3,
        out_features=n_hidden,
    )
    actor_net = MLP(
        num_cells=ncells,
        in_features=n_hidden,
        depth=2,
        out_features=2 * n_act,
    )
    value = MLP(
        in_features=n_hidden + n_act,
        num_cells=ncells,
        depth=2,
        out_features=1,
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
        ),
    )
    value_head = Mod(
        value, in_keys=["hidden", "action"], out_keys=["state_action_value"]
    )
    value = Seq(common, value_head)
    value(actor(td))

    loss = SACLoss(
        actor, value, action_spec=UnboundedContinuousTensorSpec(shape=(n_act,))
    )

    loss(td)
    benchmark(loss, td)


def test_redq_speed(benchmark, n_obs=8, n_act=4, ncells=128, batch=128, n_hidden=64):
    common = MLP(
        num_cells=ncells,
        in_features=n_obs,
        depth=3,
        out_features=n_hidden,
    )
    actor_net = MLP(
        num_cells=ncells,
        in_features=n_hidden,
        depth=2,
        out_features=2 * n_act,
    )
    value = MLP(
        in_features=n_hidden + n_act,
        num_cells=ncells,
        depth=2,
        out_features=1,
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
        ),
    )
    value_head = Mod(
        value, in_keys=["hidden", "action"], out_keys=["state_action_value"]
    )
    value = Seq(common, value_head)
    value(actor(td))

    loss = REDQLoss(
        actor, value, action_spec=UnboundedContinuousTensorSpec(shape=(n_act,))
    )

    loss(td)
    benchmark(loss, td)


def test_redq_deprec_speed(
    benchmark, n_obs=8, n_act=4, ncells=128, batch=128, n_hidden=64
):
    common = MLP(
        num_cells=ncells,
        in_features=n_obs,
        depth=3,
        out_features=n_hidden,
    )
    actor_net = MLP(
        num_cells=ncells,
        in_features=n_hidden,
        depth=2,
        out_features=2 * n_act,
    )
    value = MLP(
        in_features=n_hidden + n_act,
        num_cells=ncells,
        depth=2,
        out_features=1,
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
        ),
    )
    value_head = Mod(
        value, in_keys=["hidden", "action"], out_keys=["state_action_value"]
    )
    value = Seq(common, value_head)
    value(actor(td))

    loss = REDQLoss_deprecated(
        actor, value, action_spec=UnboundedContinuousTensorSpec(shape=(n_act,))
    )

    loss(td)
    benchmark(loss, td)


def test_td3_speed(benchmark, n_obs=8, n_act=4, ncells=128, batch=128, n_hidden=64):
    common = MLP(
        num_cells=ncells,
        in_features=n_obs,
        depth=3,
        out_features=n_hidden,
    )
    actor_net = MLP(
        num_cells=ncells,
        in_features=n_hidden,
        depth=2,
        out_features=2 * n_act,
    )
    value = MLP(
        in_features=n_hidden + n_act,
        num_cells=ncells,
        depth=2,
        out_features=1,
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
        ),
    )
    value_head = Mod(
        value, in_keys=["hidden", "action"], out_keys=["state_action_value"]
    )
    value = Seq(common, value_head)
    value(actor(td))

    loss = TD3Loss(
        actor,
        value,
        action_spec=BoundedTensorSpec(shape=(n_act,), low=-1, high=1),
    )

    loss(td)
    benchmark.pedantic(loss, args=(td,), rounds=100, iterations=10)


def test_cql_speed(benchmark, n_obs=8, n_act=4, ncells=128, batch=128, n_hidden=64):
    common = MLP(
        num_cells=ncells,
        in_features=n_obs,
        depth=3,
        out_features=n_hidden,
    )
    actor_net = MLP(
        num_cells=ncells,
        in_features=n_hidden,
        depth=2,
        out_features=2 * n_act,
    )
    value = MLP(
        in_features=n_hidden + n_act,
        num_cells=ncells,
        depth=2,
        out_features=1,
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
    )
    common = Mod(common, in_keys=["obs"], out_keys=["hidden"])
    actor = ProbSeq(
        common,
        Mod(actor_net, in_keys=["hidden"], out_keys=["param"]),
        Mod(NormalParamExtractor(), in_keys=["param"], out_keys=["loc", "scale"]),
        ProbMod(
            in_keys=["loc", "scale"], out_keys=["action"], distribution_class=TanhNormal
        ),
    )
    value_head = Mod(
        value, in_keys=["hidden", "action"], out_keys=["state_action_value"]
    )
    value = Seq(common, value_head)
    value(actor(td))

    loss = CQLLoss(
        actor, value, action_spec=UnboundedContinuousTensorSpec(shape=(n_act,))
    )

    loss(td)
    benchmark(loss, td)


def test_a2c_speed(
    benchmark, n_obs=8, n_act=4, n_hidden=64, ncells=128, batch=128, T=10
):
    common_net = MLP(
        num_cells=ncells,
        in_features=n_obs,
        depth=3,
        out_features=n_hidden,
    )
    actor_net = MLP(
        num_cells=ncells,
        in_features=n_hidden,
        depth=2,
        out_features=2 * n_act,
    )
    value_net = MLP(
        in_features=n_hidden,
        num_cells=ncells,
        depth=2,
        out_features=1,
    )
    batch = [batch, T]
    td = TensorDict(
        {
            "obs": torch.randn(*batch, n_obs),
            "action": torch.randn(*batch, n_act),
            "sample_log_prob": torch.randn(*batch),
            "done": torch.zeros(*batch, 1, dtype=torch.bool),
            "next": {
                "obs": torch.randn(*batch, n_obs),
                "reward": torch.randn(*batch, 1),
                "done": torch.zeros(*batch, 1, dtype=torch.bool),
            },
        },
        batch,
        names=[None, "time"],
    )
    common = Mod(common_net, in_keys=["obs"], out_keys=["hidden"])
    actor = ProbSeq(
        common,
        Mod(actor_net, in_keys=["hidden"], out_keys=["param"]),
        Mod(NormalParamExtractor(), in_keys=["param"], out_keys=["loc", "scale"]),
        ProbMod(
            in_keys=["loc", "scale"], out_keys=["action"], distribution_class=TanhNormal
        ),
    )
    critic = Seq(common, Mod(value_net, in_keys=["hidden"], out_keys=["state_value"]))
    actor(td.clone())
    critic(td.clone())

    loss = A2CLoss(actor_network=actor, critic_network=critic)
    advantage = GAE(value_network=critic, gamma=0.99, lmbda=0.95, shifted=True)
    advantage(td)
    loss(td)
    benchmark(loss, td)


def test_ppo_speed(
    benchmark, n_obs=8, n_act=4, n_hidden=64, ncells=128, batch=128, T=10
):
    common_net = MLP(
        num_cells=ncells,
        in_features=n_obs,
        depth=3,
        out_features=n_hidden,
    )
    actor_net = MLP(
        num_cells=ncells,
        in_features=n_hidden,
        depth=2,
        out_features=2 * n_act,
    )
    value_net = MLP(
        in_features=n_hidden,
        num_cells=ncells,
        depth=2,
        out_features=1,
    )
    batch = [batch, T]
    td = TensorDict(
        {
            "obs": torch.randn(*batch, n_obs),
            "action": torch.randn(*batch, n_act),
            "sample_log_prob": torch.randn(*batch),
            "done": torch.zeros(*batch, 1, dtype=torch.bool),
            "next": {
                "obs": torch.randn(*batch, n_obs),
                "reward": torch.randn(*batch, 1),
                "done": torch.zeros(*batch, 1, dtype=torch.bool),
            },
        },
        batch,
        names=[None, "time"],
    )
    common = Mod(common_net, in_keys=["obs"], out_keys=["hidden"])
    actor = ProbSeq(
        common,
        Mod(actor_net, in_keys=["hidden"], out_keys=["param"]),
        Mod(NormalParamExtractor(), in_keys=["param"], out_keys=["loc", "scale"]),
        ProbMod(
            in_keys=["loc", "scale"], out_keys=["action"], distribution_class=TanhNormal
        ),
    )
    critic = Seq(common, Mod(value_net, in_keys=["hidden"], out_keys=["state_value"]))
    actor(td.clone())
    critic(td.clone())

    loss = ClipPPOLoss(actor_network=actor, critic_network=critic)
    advantage = GAE(value_network=critic, gamma=0.99, lmbda=0.95, shifted=True)
    advantage(td)
    loss(td)
    benchmark(loss, td)


def test_reinforce_speed(
    benchmark, n_obs=8, n_act=4, n_hidden=64, ncells=128, batch=128, T=10
):
    common_net = MLP(
        num_cells=ncells,
        in_features=n_obs,
        depth=3,
        out_features=n_hidden,
    )
    actor_net = MLP(
        num_cells=ncells,
        in_features=n_hidden,
        depth=2,
        out_features=2 * n_act,
    )
    value_net = MLP(
        in_features=n_hidden,
        num_cells=ncells,
        depth=2,
        out_features=1,
    )
    batch = [batch, T]
    td = TensorDict(
        {
            "obs": torch.randn(*batch, n_obs),
            "action": torch.randn(*batch, n_act),
            "sample_log_prob": torch.randn(*batch),
            "done": torch.zeros(*batch, 1, dtype=torch.bool),
            "next": {
                "obs": torch.randn(*batch, n_obs),
                "reward": torch.randn(*batch, 1),
                "done": torch.zeros(*batch, 1, dtype=torch.bool),
            },
        },
        batch,
        names=[None, "time"],
    )
    common = Mod(common_net, in_keys=["obs"], out_keys=["hidden"])
    actor = ProbSeq(
        common,
        Mod(actor_net, in_keys=["hidden"], out_keys=["param"]),
        Mod(NormalParamExtractor(), in_keys=["param"], out_keys=["loc", "scale"]),
        ProbMod(
            in_keys=["loc", "scale"], out_keys=["action"], distribution_class=TanhNormal
        ),
    )
    critic = Seq(common, Mod(value_net, in_keys=["hidden"], out_keys=["state_value"]))
    actor(td.clone())
    critic(td.clone())

    loss = ReinforceLoss(actor_network=actor, critic_network=critic)
    advantage = GAE(value_network=critic, gamma=0.99, lmbda=0.95, shifted=True)
    advantage(td)
    loss(td)
    benchmark(loss, td)


def test_iql_speed(
    benchmark, n_obs=8, n_act=4, n_hidden=64, ncells=128, batch=128, T=10
):
    common_net = MLP(
        num_cells=ncells,
        in_features=n_obs,
        depth=3,
        out_features=n_hidden,
    )
    actor_net = MLP(
        num_cells=ncells,
        in_features=n_hidden,
        depth=2,
        out_features=2 * n_act,
    )
    value_net = MLP(
        in_features=n_hidden,
        num_cells=ncells,
        depth=2,
        out_features=1,
    )
    qvalue_net = MLP(
        in_features=n_hidden + n_act,
        num_cells=ncells,
        depth=2,
        out_features=1,
    )
    batch = [batch, T]
    td = TensorDict(
        {
            "obs": torch.randn(*batch, n_obs),
            "action": torch.randn(*batch, n_act),
            "sample_log_prob": torch.randn(*batch),
            "done": torch.zeros(*batch, 1, dtype=torch.bool),
            "next": {
                "obs": torch.randn(*batch, n_obs),
                "reward": torch.randn(*batch, 1),
                "done": torch.zeros(*batch, 1, dtype=torch.bool),
            },
        },
        batch,
        names=[None, "time"],
    )
    common = Mod(common_net, in_keys=["obs"], out_keys=["hidden"])
    actor = ProbSeq(
        common,
        Mod(actor_net, in_keys=["hidden"], out_keys=["param"]),
        Mod(NormalParamExtractor(), in_keys=["param"], out_keys=["loc", "scale"]),
        ProbMod(
            in_keys=["loc", "scale"], out_keys=["action"], distribution_class=TanhNormal
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
    benchmark(loss, td)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
