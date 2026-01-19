# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Assertions and validation utilities for TorchRL tests."""

from __future__ import annotations

import torch
from tensordict import TensorDict

__all__ = [
    "check_rollout_consistency_multikey_env",
    "rand_reset",
    "rollout_consistency_assertion",
]


def rollout_consistency_assertion(
    rollout, *, done_key="done", observation_key="observation", done_strict=False
):
    """Test that observations in 'next' match observations in the next root tensordict.

    Verifies consistency: when done is False the next observation should match,
    and when done is True they should differ (indicating a reset occurred).

    Args:
        rollout: The rollout tensordict to validate.
        done_key: The key for the done signal.
        observation_key: The key for observations.
        done_strict: If True, raise an error if no done is detected.
    """
    done = rollout[..., :-1]["next", done_key].squeeze(-1)
    # data resulting from step, when it's not done
    r_not_done = rollout[..., :-1]["next"][~done]
    # data resulting from step, when it's not done, after step_mdp
    r_not_done_tp1 = rollout[:, 1:][~done]
    torch.testing.assert_close(
        r_not_done[observation_key],
        r_not_done_tp1[observation_key],
        msg=f"Key {observation_key} did not match",
    )

    if done_strict and not done.any():
        raise RuntimeError("No done detected, test could not complete.")
    if done.any():
        # data resulting from step, when it's done
        r_done = rollout[..., :-1]["next"][done]
        # data resulting from step, when it's done, after step_mdp and reset
        r_done_tp1 = rollout[..., 1:][done]
        # check that at least one obs after reset does not match the version before reset
        assert not torch.isclose(
            r_done[observation_key], r_done_tp1[observation_key]
        ).all()


def rand_reset(env):
    """Generate a tensordict with reset keys that mimic the done spec.

    Values are drawn at random until at least one reset is present.

    Args:
        env: The environment to generate reset keys for.

    Returns:
        A TensorDict containing the reset signals.
    """
    full_done_spec = env.full_done_spec
    result = {}
    for reset_key, list_of_done in zip(env.reset_keys, env.done_keys_groups):
        val = full_done_spec[list_of_done[0]].rand()
        while not val.any():
            val = full_done_spec[list_of_done[0]].rand()
        result[reset_key] = val
    # create a data structure that keeps the batch size of the nested specs
    result = (
        full_done_spec.zero().update(result).exclude(*full_done_spec.keys(True, True))
    )
    return result


def check_rollout_consistency_multikey_env(td: TensorDict, max_steps: int):
    """Check rollout consistency for environments with multiple observation/action keys.

    Validates that:
    - Done and reset behavior is correct for root, nested_1, and nested_2
    - Observations update correctly based on actions
    - Rewards are computed correctly

    Args:
        td: The rollout tensordict to validate.
        max_steps: The maximum steps before done in the environment.
    """
    index_batch_size = (0,) * (len(td.batch_size) - 1)

    # Check done and reset for root
    observation_is_max = td["next", "observation"][..., 0, 0, 0] == max_steps + 1
    next_is_done = td["next", "done"][index_batch_size][:-1].squeeze(-1)
    assert (td["next", "done"][observation_is_max]).all()
    assert (~td["next", "done"][~observation_is_max]).all()
    # Obs after done is 0
    assert (td["observation"][index_batch_size][1:][next_is_done] == 0).all()
    # Obs after not done is previous obs
    assert (
        td["observation"][index_batch_size][1:][~next_is_done]
        == td["next", "observation"][index_batch_size][:-1][~next_is_done]
    ).all()
    # Check observation and reward update with count action for root
    action_is_count = td["action"].long().argmax(-1).to(torch.bool)
    assert (
        td["next", "observation"][action_is_count]
        == td["observation"][action_is_count] + 1
    ).all()
    assert (td["next", "reward"][action_is_count] == 1).all()
    # Check observation and reward do not update with no-count action for root
    assert (
        td["next", "observation"][~action_is_count]
        == td["observation"][~action_is_count]
    ).all()
    assert (td["next", "reward"][~action_is_count] == 0).all()

    # Check done and reset for nested_1
    observation_is_max = td["next", "nested_1", "observation"][..., 0] == max_steps + 1
    # done at the root always prevail
    next_is_done = td["next", "done"][index_batch_size][:-1].squeeze(-1)
    assert (td["next", "nested_1", "done"][observation_is_max]).all()
    assert (~td["next", "nested_1", "done"][~observation_is_max]).all()
    # Obs after done is 0
    assert (
        td["nested_1", "observation"][index_batch_size][1:][next_is_done] == 0
    ).all()
    # Obs after not done is previous obs
    assert (
        td["nested_1", "observation"][index_batch_size][1:][~next_is_done]
        == td["next", "nested_1", "observation"][index_batch_size][:-1][~next_is_done]
    ).all()
    # Check observation and reward update with count action for nested_1
    action_is_count = td["nested_1"]["action"].to(torch.bool)
    assert (
        td["next", "nested_1", "observation"][action_is_count]
        == td["nested_1", "observation"][action_is_count] + 1
    ).all()
    assert (td["next", "nested_1", "gift"][action_is_count] == 1).all()
    # Check observation and reward do not update with no-count action for nested_1
    assert (
        td["next", "nested_1", "observation"][~action_is_count]
        == td["nested_1", "observation"][~action_is_count]
    ).all()
    assert (td["next", "nested_1", "gift"][~action_is_count] == 0).all()

    # Check done and reset for nested_2
    observation_is_max = td["next", "nested_2", "observation"][..., 0] == max_steps + 1
    # done at the root always prevail
    next_is_done = td["next", "done"][index_batch_size][:-1].squeeze(-1)
    assert (td["next", "nested_2", "done"][observation_is_max]).all()
    assert (~td["next", "nested_2", "done"][~observation_is_max]).all()
    # Obs after done is 0
    assert (
        td["nested_2", "observation"][index_batch_size][1:][next_is_done] == 0
    ).all()
    # Obs after not done is previous obs
    assert (
        td["nested_2", "observation"][index_batch_size][1:][~next_is_done]
        == td["next", "nested_2", "observation"][index_batch_size][:-1][~next_is_done]
    ).all()
    # Check observation and reward update with count action for nested_2
    action_is_count = td["nested_2"]["azione"].squeeze(-1).to(torch.bool)
    assert (
        td["next", "nested_2", "observation"][action_is_count]
        == td["nested_2", "observation"][action_is_count] + 1
    ).all()
    assert (td["next", "nested_2", "reward"][action_is_count] == 1).all()
    # Check observation and reward do not update with no-count action for nested_2
    assert (
        td["next", "nested_2", "observation"][~action_is_count]
        == td["nested_2", "observation"][~action_is_count]
    ).all()
    assert (td["next", "nested_2", "reward"][~action_is_count] == 0).all()
