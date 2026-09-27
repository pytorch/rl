# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict
from torch import nn

from torchrl import objectives, timeit
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer

from torchrl.modules import (
    FlowMatchingModel,
    FlowMatchingPolicy,
    MLP,
    OneStepPolicy,
    ValueOperator,
)
from torchrl.testing import get_default_devices
from torchrl.trainers import algorithms


class TestFlow:
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("compiled", [False, True])
    def test_sampling(self, benchmark, device, compiled):
        model = FlowMatchingModel(
            nn.Sequential(
                nn.Linear(21, 64, device=device),
                nn.Tanh(),
                nn.Linear(64, 4, device=device),
            ),
            4,
        )
        observation = torch.randn(64, 16, device=device)
        noise = torch.randn(64, 4, device=device)
        torch._dynamo.reset()
        call = torch.compile(model, fullgraph=True) if compiled else model

        def run():
            action = call(observation, noise)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            return action

        with torch.no_grad():
            run()
            benchmark(run)

    @pytest.mark.skipif(
        not hasattr(algorithms, "FQLTrainer"),
        reason="FQLTrainer is unavailable on the benchmark baseline revision",
    )
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("compiled", [False, True])
    def test_fql_update(self, benchmark, device, compiled):
        """Full FQL update on a fixed synthetic antmaze-shaped replay buffer."""
        torch.manual_seed(0)
        batch_size, replay_size, observation_dim, action_dim = 256, 32768, 29, 8
        precision = torch.get_float32_matmul_precision()
        num_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision("highest")

        def network(inputs: int, outputs: int, layer_norm: bool = False) -> MLP:
            def activation() -> nn.Module:
                layers = [nn.GELU(approximate="tanh")]
                if layer_norm:
                    layers.append(nn.LayerNorm(512, eps=1e-6, device=device))
                return nn.Sequential(*layers)

            model = MLP(
                inputs,
                outputs,
                num_cells=[512] * 4,
                activation_class=activation,
                device=device,
            )
            for layer in model.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
            return model

        try:
            loss = objectives.FQLLoss(
                FlowMatchingPolicy(
                    network(observation_dim + action_dim + 1, action_dim), action_dim
                ),
                OneStepPolicy(
                    network(observation_dim + action_dim, action_dim), action_dim
                ),
                [
                    ValueOperator(
                        network(observation_dim + action_dim, 1, layer_norm=True),
                        in_keys=["observation", "action"],
                    )
                    for _ in range(2)
                ],
                alpha=10.0,
                q_aggregation="min",
            )
            loss.make_value_estimator(gamma=0.99)
            # CPU replay includes the host-to-device transfer in each timed update.
            replay = TensorDictReplayBuffer(
                storage=LazyTensorStorage(replay_size, device="cpu"),
                batch_size=batch_size,
            )
            replay.extend(
                TensorDict(
                    {
                        "observation": torch.randn(
                            replay_size, observation_dim, device="cpu"
                        ),
                        "action": torch.randn(
                            replay_size, action_dim, device="cpu"
                        ).clamp(-1, 1),
                        ("next", "observation"): torch.randn(
                            replay_size, observation_dim, device="cpu"
                        ),
                        ("next", "reward"): -torch.ones(replay_size, 1, device="cpu"),
                        ("next", "done"): torch.zeros(
                            replay_size, 1, dtype=torch.bool, device="cpu"
                        ),
                        ("next", "terminated"): torch.zeros(
                            replay_size, 1, dtype=torch.bool, device="cpu"
                        ),
                    },
                    [replay_size],
                )
            )
            trainer = algorithms.FQLTrainer(
                loss_module=loss,
                optimizer=torch.optim.Adam(loss.parameters(), lr=3e-4),
                replay_buffer=replay,
                target_net_updater=objectives.SoftUpdate(loss, tau=0.005),
                offline_steps=0,
                device=device,
                compile_loss=compiled,
                auto_log_optim_steps=False,
            )

            def update() -> None:
                trainer.update()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)

            torch._dynamo.reset()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            with timeit("fql_first_update", sync=device.type == "cuda") as timer:
                update()
                first_update = timer.elapsed()
            benchmark.extra_info.update(
                first_update_seconds=first_update,
                batch_size=batch_size,
                replay_size=replay_size,
                observation_dim=observation_dim,
                action_dim=action_dim,
                hidden_dims=[512] * 4,
                flow_steps=10,
                matmul_precision="highest",
                workload="synthetic",
            )
            benchmark.pedantic(update, rounds=200, iterations=1, warmup_rounds=10)
            if device.type == "cuda":
                benchmark.extra_info[
                    "peak_allocated_bytes"
                ] = torch.cuda.max_memory_allocated(device)
        finally:
            torch.set_float32_matmul_precision(precision)
            torch.set_num_threads(num_threads)


if __name__ == "__main__":
    pytest.main([__file__, "--benchmark-only"])
