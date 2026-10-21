# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import math

import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn

from torchrl.data import LazyTensorStorage, SubgoalTriplet, TensorDictReplayBuffer
from torchrl.envs.utils import ExplorationType
from torchrl.modules import IndependentNormal, ProbabilisticActor
from torchrl.objectives import SoftUpdate, TransitiveQLoss

OBS, ACT = 3, 2


class ConstantLogit(nn.Module):
    """A critic returning the same logit for every input."""

    def __init__(self, logit: float):
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(logit))

    def forward(self, observation, goal, action):
        return self.logit.expand(*observation.shape[:-1], 1)


class GoalSumLogit(nn.Module):
    """A critic whose logit is the sum of its goal input."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, observation, goal, action):
        return self.scale * goal.sum(-1, keepdim=True)


class LinearLogit(nn.Module):
    """A trainable critic on the concatenated inputs."""

    def __init__(self, goal_dim: int = OBS):
        super().__init__()
        self.net = nn.Linear(OBS + goal_dim + ACT, 1)

    def forward(self, observation, goal, action):
        return self.net(torch.cat([observation, goal, action], -1))


class ZeroMeanActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(2 * OBS, ACT)
        nn.init.zeros_(self.net.weight)
        nn.init.zeros_(self.net.bias)

    def forward(self, observation, goal):
        loc = self.net(torch.cat([observation, goal], -1))
        return loc, torch.ones_like(loc)


def make_actor():
    return ProbabilisticActor(
        TensorDictModule(ZeroMeanActor(), ["observation", "goal"], ["loc", "scale"]),
        in_keys=["loc", "scale"],
        distribution_class=IndependentNormal,
        default_interaction_type=ExplorationType.DETERMINISTIC,
    )


def make_critics(module_cls, *args):
    return [
        TensorDictModule(
            module_cls(*args),
            ["observation", "goal", "action"],
            ["state_action_value"],
        )
        for _ in range(2)
    ]


def make_batch(subgoal_offsets, goal_offsets, policy_goal=None):
    n = len(subgoal_offsets)
    offsets = torch.stack(
        [
            torch.zeros(n, dtype=torch.long),
            torch.as_tensor(subgoal_offsets),
            torch.as_tensor(goal_offsets),
            torch.ones(n, dtype=torch.long),
        ],
        -1,
    )
    batch = TensorDict(
        {
            "observation": torch.randn(n, 4, OBS),
            "action": torch.rand(n, 4, ACT) * 2 - 1,
            "step_offset": offsets,
        },
        batch_size=[n, 4],
    )
    if policy_goal is not None:
        batch["goals", "oracle"] = policy_goal
    return batch


def bce(logit, target):
    return -(
        target * math.log(torch.sigmoid(torch.tensor(logit)))
        + (1 - target) * math.log(1 - torch.sigmoid(torch.tensor(logit)))
    )


class TestTransitiveQLoss:
    @pytest.mark.parametrize("distance_weight", [0.0, 1.0])
    def test_critic_target_and_weights(self, distance_weight):
        """Targets use gamma**offset at horizons <= 1 and the target critic product otherwise."""
        gamma, kappa, logit = 0.9, 0.7, 2.0
        loss = TransitiveQLoss(
            make_actor(),
            make_critics(ConstantLogit, logit),
            discount=gamma,
            expectile=kappa,
            distance_weight=distance_weight,
        )
        q = 1 / (1 + math.exp(-logit))
        cases = [(0, 1, gamma), (1, 2, gamma**2), (0, 5, q), (2, 5, q * q)]
        batch = make_batch([c[0] for c in cases], [c[1] for c in cases])
        expected = []
        for _, _, target in cases:
            weight = kappa if target >= q else 1 - kappa
            weight *= (1 + math.log(target) / math.log(gamma)) ** -distance_weight
            expected.append(weight * float(bce(logit, target)))
        out = loss(batch)
        assert out["loss_qvalue"].item() == pytest.approx(
            sum(expected) / len(expected), rel=1e-5
        )

    def test_actor_loss(self):
        """Normalized critic term plus the behavior negative log-likelihood of a unit Gaussian."""
        alpha, logit = 0.5, -3.0
        loss = TransitiveQLoss(
            make_actor(), make_critics(ConstantLogit, logit), alpha=alpha
        )
        batch = make_batch([0, 1, 2], [3, 4, 5])
        action = batch["action"][:, 0]
        nll = 0.5 * action.pow(2).sum(-1) + 0.5 * ACT * math.log(2 * math.pi)
        out = loss(batch)
        assert out["loss_actor"].item() == pytest.approx(
            1.0 + alpha * nll.mean().item(), rel=1e-5
        )

    def test_oracle_distillation_reads_policy_goal(self):
        """The oracle critic sees the nested policy-goal entry of the goal record."""
        policy_goal = torch.randn(5, 4, OBS)
        loss = TransitiveQLoss(
            make_actor(),
            make_critics(ConstantLogit, 1.0),
            oracle_qvalue_network=make_critics(GoalSumLogit),
        )
        loss.set_keys(policy_goal=("goals", "oracle"))
        out = loss(make_batch([0] * 5, [3] * 5, policy_goal))
        oracle_logit = policy_goal[:, 2].sum(-1)
        q = torch.sigmoid(torch.tensor(1.0))
        expected = nn.functional.binary_cross_entropy_with_logits(
            oracle_logit, q.expand_as(oracle_logit)
        )
        assert out["loss_oracle_qvalue"].item() == pytest.approx(
            expected.item(), rel=1e-5
        )

    def test_invalid_anchors_do_not_contribute(self):
        """Anchors SubgoalTriplet flags invalid leave the critic and actor losses unchanged.

        The flat sample is fed as the buffer returns it; single-record episodes
        are the anchors the unit marks invalid.
        """
        torch.manual_seed(0)
        done = torch.zeros(24, 1, dtype=torch.bool)
        done[[0, 8, 9, 10, 22, 23]] = True
        rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(24),
            batch_size=64,
            sample_unit=SubgoalTriplet(goal_discount=0.9),
        )
        rb.extend(
            TensorDict(
                {
                    "observation": torch.randn(24, OBS),
                    "action": torch.rand(24, ACT) * 2 - 1,
                    ("next", "done"): done,
                },
                batch_size=[24],
            )
        )
        batch = rb.sample()
        records = batch.unflatten(0, (batch.shape[0] // 4, 4))
        valid = records["validity_mask"][:, 0]
        assert valid.any() and not valid.all()
        loss = TransitiveQLoss(make_actor(), make_critics(GoalSumLogit), alpha=0.5)
        masked = loss(batch)
        reference = loss(records[valid].exclude("validity_mask"))
        for key in ("loss_qvalue", "loss_actor"):
            torch.testing.assert_close(masked[key], reference[key])

    def test_oracle_ensemble_size_must_match(self):
        with pytest.raises(ValueError, match="as many members as the critic"):
            TransitiveQLoss(
                make_actor(),
                make_critics(LinearLogit),
                oracle_qvalue_network=make_critics(LinearLogit) * 2,
            )

    def test_compile_matches_eager(self):
        """Compiled updates follow the eager losses, gradients and target parameters."""
        torch.manual_seed(0)
        losses = []
        for _ in range(2):
            torch.manual_seed(0)
            loss = TransitiveQLoss(
                make_actor(),
                make_critics(LinearLogit),
                oracle_qvalue_network=make_critics(LinearLogit),
                distance_weight=1.0,
            )
            loss.set_keys(policy_goal=("goals", "oracle"))
            losses.append(loss)
        losses[1].load_state_dict(losses[0].state_dict())
        updaters = [SoftUpdate(loss, tau=0.05) for loss in losses]
        optimizers = [torch.optim.Adam(loss.parameters(), lr=1e-2) for loss in losses]
        torch._dynamo.reset()
        calls = (losses[0], torch.compile(losses[1], fullgraph=True))
        for _ in range(3):
            batch = make_batch([0, 1, 2, 3], [1, 3, 5, 7], torch.randn(4, 4, OBS))
            results, gradients = [], []
            for loss, call, optimizer, updater in zip(
                losses, calls, optimizers, updaters
            ):
                result = call(batch.clone())
                optimizer.zero_grad(set_to_none=True)
                sum(v for k, v in result.items() if k.startswith("loss_")).backward()
                gradients.append([p.grad.clone() for p in loss.parameters()])
                results.append(result.detach())
                optimizer.step()
                updater.step()
            torch.testing.assert_close(results[0], results[1], atol=1e-5, rtol=1e-4)
            torch.testing.assert_close(gradients[0], gradients[1], atol=1e-5, rtol=1e-4)
            torch.testing.assert_close(
                losses[0].state_dict(), losses[1].state_dict(), atol=1e-5, rtol=1e-4
            )


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
