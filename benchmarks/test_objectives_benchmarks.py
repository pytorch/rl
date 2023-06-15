# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse

import pytest
import torch
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
    print(size)

    torch.manual_seed(0)
    device = "cuda:0" if torch.cuda.device_count() else "cpu"
    values = torch.randn(*size, device=device)
    next_values = torch.randn(*size, device=device)
    reward = torch.randn(*size, device=device)
    done = torch.zeros(*size, dtype=torch.bool, device=device).bernoulli_(0.1)

    gamma = 0.99
    if gamma_tensor:
        gamma = torch.full(size, gamma)
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


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
