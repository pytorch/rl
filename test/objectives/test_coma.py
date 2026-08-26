import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.modules import OneHotCategorical, ProbabilisticActor

from torchrl.objectives.multiagent import COMALoss

from torchrl.objectives.multiagent.coma import (
    add_action_without_self,
    add_joint_observation,
    add_masked_joint_action,
)


class FixedActionValue(nn.Module):
    def __init__(self, values: torch.Tensor):
        super().__init__()
        self.values = nn.Parameter(values.clone())

    def forward(self, observation, action_without_self):
        del action_without_self
        return self.values.expand(*observation.shape[:-1], -1)


def _one_hot(index, n_actions=3):
    return torch.nn.functional.one_hot(torch.as_tensor(index), n_actions).to(torch.float)


def _make_loss(
    gamma=0.5, qvalue_loss_coef=0.5, entropy_coef=0.0, n_step=1, normalize_advantage=False,
):
    obs_dim = 4
    n_actions = 3
    actor_net = nn.Linear(obs_dim, n_actions)
    nn.init.zeros_(actor_net.weight)
    nn.init.zeros_(actor_net.bias)
    actor_module = TensorDictModule(
        actor_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "logits")],
    )
    actor = ProbabilisticActor(
        module=actor_module,
        in_keys=[("agents", "logits")],
        out_keys=[("agents", "action")],
        distribution_class=OneHotCategorical,
        return_log_prob=True,
    )
    qvalue_module = TensorDictModule(
        FixedActionValue(torch.tensor([1.0, 2.0, 4.0])),
        in_keys=[("agents", "observation"), ("agents", "action_without_self")],
        out_keys=[("agents", "action_value")],
    )
    return COMALoss(
        actor_network=actor,
        qvalue_network=qvalue_module,
        gamma=gamma,
        qvalue_loss_coef=qvalue_loss_coef,
        entropy_coef=entropy_coef,
        n_step=n_step,
        normalize_advantage=normalize_advantage,
    )


def test_add_action_without_self_flattens_other_agents_actions():
    action = _one_hot([[0, 1, 2]], n_actions=3)
    tensordict = TensorDict({("agents", "action"): action}, batch_size=[1])

    add_action_without_self(tensordict)

    expected = torch.tensor(
        [
            [
                [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ]
        ]
    )
    torch.testing.assert_close(tensordict.get(("agents", "action_without_self")), expected)


def test_coma_loss_uses_counterfactual_baseline_and_qvalue_target():
    loss = _make_loss(qvalue_loss_coef=0.25, entropy_coef=0.0)
    action = _one_hot([[0, 2]], n_actions=3)
    tensordict = TensorDict(
        {
            ("agents", "observation"): torch.zeros(1, 2, 4),
            ("agents", "action"): action,
            "value_target": torch.zeros(1, 2, 1),
        },
        batch_size=[1],
    )
    add_action_without_self(tensordict)

    loss_values = loss(tensordict)

    expected_advantage = torch.tensor([1.0, 4.0]).mean() - torch.tensor([1.0, 2.0, 4.0]).mean()
    expected_qvalue_loss = ((torch.tensor([1.0, 4.0]) ** 2).mean()) * 0.25
    torch.testing.assert_close(loss_values["advantage"], expected_advantage)
    torch.testing.assert_close(loss_values["loss_qvalue"], expected_qvalue_loss)
    assert set(loss.out_keys).issubset(set(loss_values.keys()))


def test_compute_value_target_bootstraps_along_time_dimension():
    loss = _make_loss(gamma=0.5)
    actions = _one_hot([[[0, 0], [1, 1], [2, 2]]], n_actions=3)
    tensordict = TensorDict(
        {
            ("agents", "observation"): torch.zeros(1, 3, 2, 4),
            ("agents", "action"): actions,
            "next": {
                "agents": {
                    "reward": torch.zeros(1, 3, 2, 1),
                    "done": torch.tensor([[[[False], [False]], [[False], [False]], [[True], [True]]]]),
                }
            },
        },
        batch_size=[1, 3],
    )
    add_action_without_self(tensordict)

    loss.compute_value_target(tensordict)

    expected = torch.tensor([[[[1.0], [1.0]], [[2.0], [2.0]], [[0.0], [0.0]]]])
    torch.testing.assert_close(tensordict["value_target"], expected)


def test_compute_value_target_supports_nstep_returns():
    """n_step=2 accumulates two rewards then bootstraps, stopping at done."""
    loss = _make_loss(gamma=0.5, n_step=2)
    actions = _one_hot([[[0, 0], [1, 1], [2, 2]]], n_actions=3)
    tensordict = TensorDict(
        {
            ("agents", "observation"): torch.zeros(1, 3, 2, 4),
            ("agents", "action"): actions,
            "next": {
                "agents": {
                    "reward": torch.ones(1, 3, 2, 1),
                    "done": torch.tensor([[[[False], [False]], [[False], [False]], [[True], [True]]]]),
                }
            },
        },
        batch_size=[1, 3],
    )
    add_action_without_self(tensordict)

    loss.compute_value_target(tensordict)

    # G2(t0) = r0 + g*r1 + g^2*Q(t2) = 1 + 0.5 + 0.25*4 = 2.5
    # G2(t1) = r1 + g*r2 = 1.5 (done at t2 stops the bootstrap)
    # G2(t2) = r2 = 1.0
    expected = torch.tensor([[[[2.5], [2.5]], [[1.5], [1.5]], [[1.0], [1.0]]]])
    torch.testing.assert_close(tensordict["value_target"], expected)


def test_add_joint_observation_repeats_team_observation_per_agent():
    observation = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    tensordict = TensorDict({("agents", "observation"): observation}, batch_size=[1])

    add_joint_observation(tensordict)

    expected = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]])
    torch.testing.assert_close(tensordict.get(("agents", "joint_observation")), expected)


def test_add_masked_joint_action_zeroes_own_action_block():
    action = _one_hot([[0, 2]], n_actions=3)
    tensordict = TensorDict({("agents", "action"): action}, batch_size=[1])

    add_masked_joint_action(tensordict)

    expected = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ]
    )
    torch.testing.assert_close(tensordict.get(("agents", "masked_joint_action")), expected)


def test_normalize_advantage_zero_centres_the_actor_signal():
    """With a uniform policy, log-probs are constant, so the actor loss equals
    -log(1/3) * mean(A); standardised advantages have zero mean, so the
    normalized actor loss must vanish while the raw one does not."""
    action = _one_hot([[0, 2]], n_actions=3)
    data = {
        ("agents", "observation"): torch.zeros(1, 2, 4),
        ("agents", "action"): action,
        "value_target": torch.zeros(1, 2, 1),
    }
    raw = TensorDict(dict(data), batch_size=[1])
    add_action_without_self(raw)
    norm = TensorDict(dict(data), batch_size=[1])
    add_action_without_self(norm)

    raw_loss = _make_loss(normalize_advantage=False)(raw)
    norm_loss = _make_loss(normalize_advantage=True)(norm)

    assert abs(raw_loss["loss_actor"].item()) > 1e-3
    torch.testing.assert_close(norm_loss["loss_actor"], torch.tensor(0.0), atol=1e-5, rtol=0)


def test_diagnostics_report_q_contrast_measures():
    """Flat-critic instrumentation: own-action spread, others-sensitivity,
    and empirical target contrast by chosen action."""
    loss = _make_loss()
    action = _one_hot([[0, 2]], n_actions=3)
    tensordict = TensorDict(
        {
            ("agents", "observation"): torch.zeros(1, 2, 4),
            ("agents", "action"): action,
            ("agents", "logits"): torch.zeros(1, 2, 3),
            "value_target": torch.tensor([[[0.5], [2.5]]]),
        },
        batch_size=[1],
    )
    add_action_without_self(tensordict)

    diag = loss.diagnostics(tensordict)

    # FixedActionValue outputs [1, 2, 4] for every input:
    # population std of [1, 2, 4] = sqrt(14/9)
    expected_spread = torch.tensor([1.0, 2.0, 4.0]).std(correction=0)
    torch.testing.assert_close(diag["action_value_spread"], expected_spread.expand(1, 2))
    # the fixed critic ignores other agents' actions entirely
    torch.testing.assert_close(diag["others_sensitivity"], torch.zeros(1, 2))
    # targets grouped by chosen action: {action0: 0.5, action2: 2.5} -> pop std 1.0
    torch.testing.assert_close(diag["target_contrast"], torch.tensor(1.0))


def test_compute_value_target_bootstraps_along_time_only_rollout():
    """A [T, n_agents, 1] rollout from an unbatched collector has no leading
    batch dim: time must resolve to dim 0, not dim 1 (the agent dim), or the
    bootstrap shift silently mixes agents together instead of time steps."""
    loss = _make_loss(gamma=0.5)
    actions = _one_hot([[0, 0], [1, 1], [2, 2]], n_actions=3)
    tensordict = TensorDict(
        {
            ("agents", "observation"): torch.zeros(3, 2, 4),
            ("agents", "action"): actions,
            "next": {
                "agents": {
                    "reward": torch.zeros(3, 2, 1),
                    "done": torch.tensor(
                        [[[False], [False]], [[False], [False]], [[True], [True]]]
                    ),
                }
            },
        },
        batch_size=[3],
    )
    add_action_without_self(tensordict)

    loss.compute_value_target(tensordict)

    expected = torch.tensor([[[1.0], [1.0]], [[2.0], [2.0]], [[0.0], [0.0]]])
    torch.testing.assert_close(tensordict["value_target"], expected)


def test_compute_value_target_marks_non_terminal_tail_as_invalid():
    """A rollout window that simply ends (``done=False`` throughout, i.e. it
    was truncated mid-episode rather than reaching a real terminal state) has
    no known next transition beyond it. The zero-filled bootstrap there must
    not be trusted as a real target, so the last ``n_step`` positions are
    flagged invalid via ``shifted_valid`` instead of silently biasing the
    return low."""
    loss = _make_loss(gamma=0.5, n_step=1)
    actions = _one_hot([[[0, 0], [1, 1], [2, 2]]], n_actions=3)
    tensordict = TensorDict(
        {
            ("agents", "observation"): torch.zeros(1, 3, 2, 4),
            ("agents", "action"): actions,
            "next": {
                "agents": {
                    "reward": torch.zeros(1, 3, 2, 1),
                    "done": torch.zeros(1, 3, 2, 1, dtype=torch.bool),
                }
            },
        },
        batch_size=[1, 3],
    )
    add_action_without_self(tensordict)

    loss.compute_value_target(tensordict)

    expected_valid = torch.tensor(
        [[[[True], [True]], [[True], [True]], [[False], [False]]]]
    )
    torch.testing.assert_close(tensordict["shifted_valid"], expected_valid)


def test_forward_excludes_shifted_invalid_positions_from_qvalue_loss():
    """``compute_value_target``'s ``shifted_valid`` mask must be honored
    automatically by ``forward`` (via ``LossModule._reduce_loss``): the
    non-terminal tail's zero-filled target must not pull the qvalue loss even
    though it is numerically way off."""
    loss = _make_loss(gamma=0.5, qvalue_loss_coef=1.0, n_step=1)
    actions = _one_hot([[0], [2]], n_actions=3)
    tensordict = TensorDict(
        {
            ("agents", "observation"): torch.zeros(2, 1, 4),
            ("agents", "action"): actions,
            "next": {
                "agents": {
                    "reward": torch.zeros(2, 1, 1),
                    "done": torch.zeros(2, 1, 1, dtype=torch.bool),
                }
            },
        },
        batch_size=[2],
    )
    add_action_without_self(tensordict)
    loss.compute_value_target(tensordict)

    loss_values = loss(tensordict)

    # chosen_action_value = [1, 4]; value_target = [2, 0] (t1's target is the
    # zero-filled, invalid one). Only t0's (1 - 2) ** 2 = 1 should count;
    # t1's (4 - 0) ** 2 = 16 must be excluded by ``shifted_valid``.
    torch.testing.assert_close(loss_values["loss_qvalue"], torch.tensor(1.0))


def test_forward_excludes_positions_marked_by_collector_mask():
    """``("collector", "mask")`` (written by a ``SliceSampler`` with
    ``pad_output=True`` to flag padded steps) must be honored automatically as
    well, independently of ``shifted_valid`` / ``compute_value_target``."""
    loss = _make_loss(qvalue_loss_coef=1.0)
    action = _one_hot([[0, 2]], n_actions=3)
    tensordict = TensorDict(
        {
            ("agents", "observation"): torch.zeros(1, 2, 4),
            ("agents", "action"): action,
            "value_target": torch.tensor([[[2.0], [2.0]]]),
            ("collector", "mask"): torch.tensor([[True, False]]),
        },
        batch_size=[1],
    )
    add_action_without_self(tensordict)

    loss_values = loss(tensordict)

    # chosen_action_value = [1, 4]; only the first (unmasked) position's
    # (1 - 2) ** 2 = 1 counts. The second position's (4 - 2) ** 2 = 4 is
    # excluded by the collector mask.
    torch.testing.assert_close(loss_values["loss_qvalue"], torch.tensor(1.0))


def test_compute_value_target_respects_flat_and_nested_reward_done_keys():
    """Overriding ``reward``/``done`` via ``set_keys`` with a flat string (not
    the default nested tuple) must not be shredded into a per-character tuple
    such as ``("next", "r", "e", "w", "a", "r", "d")``."""
    for reward_key, done_key in [
        ("reward", "done"),
        (("data", "reward"), ("data", "done")),
    ]:
        loss = _make_loss(gamma=0.5)
        loss.set_keys(reward=reward_key, done=done_key)
        actions = _one_hot([[[0, 0], [1, 1]]], n_actions=3)
        tensordict = TensorDict(
            {
                ("agents", "observation"): torch.zeros(1, 2, 2, 4),
                ("agents", "action"): actions,
            },
            batch_size=[1, 2],
        )
        tensordict.set(("next", reward_key), torch.zeros(1, 2, 2, 1))
        tensordict.set(
            ("next", done_key),
            torch.tensor([[[[False], [False]], [[True], [True]]]]),
        )
        add_action_without_self(tensordict)

        loss.compute_value_target(tensordict)

        expected = torch.tensor([[[[1.0], [1.0]], [[0.0], [0.0]]]])
        torch.testing.assert_close(tensordict["value_target"], expected)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])

