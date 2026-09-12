# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Compare per-step-state and parallel compact-window GTrXL training on CPU."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch
from tensordict import TensorDict
from torch.utils.benchmark import Timer
from torchrl.modules import GTrXL, set_recurrent_mode, TransformerModule


def train_window(module, data):
    module.zero_grad(set_to_none=True)
    with set_recurrent_mode(True):
        module(data.clone())["features"].square().mean().backward()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    results = []
    for batch, length, memory_len, width in [
        (1, 8, 8, 16),
        (32, 32, 16, 32),
        (32, 64, 64, 64),
    ]:
        torch.manual_seed(0)
        module = TransformerModule(
            transformer=GTrXL(7, width, 2, memory_len=memory_len),
            in_keys=["observation", "state"],
            out_keys=["features", ("next", "state")],
        )
        state = module.transformer.state_spec.zero([batch])
        state["memory"].normal_()
        state["valid"].fill_(True)
        observation = torch.randn(batch, length, 7)
        is_init = torch.zeros(batch, length, 1, dtype=torch.bool)
        compact = TensorDict(
            {"observation": observation, "is_init": is_init, "state": state}, [batch]
        )
        dense = TensorDict(
            {
                "observation": observation,
                "is_init": is_init,
                "state": state.unsqueeze(-1).expand(batch, length),
            },
            [batch, length],
        )
        with set_recurrent_mode(True):
            expected = module(dense.clone())["features"]
            actual = module(compact.clone())["features"]
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        for layout, data in (("per_step", dense), ("compact", compact)):
            measurement = Timer(
                stmt="train_window(module, data)",
                globals={"train_window": train_window, "module": module, "data": data},
                num_threads=1,
            ).blocked_autorange(min_run_time=1)
            memory_bytes = batch * 2 * memory_len * width * observation.element_size()
            results.append(
                {
                    "layout": layout,
                    "batch": batch,
                    "window_length": length,
                    "memory_len": memory_len,
                    "hidden_size": width,
                    "latency_ms": measurement.median * 1000,
                    "transitions_per_second": batch * length / measurement.median,
                    "carry_payload_bytes": memory_bytes
                    * (length if layout == "per_step" else 1),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "platform": platform.platform(),
                "torch_version": torch.__version__,
                "num_threads": 1,
                "description": "Clone input, forward and backward with identical weights/tensors; no optimizer or collection. Plain TensorDict in both paths.",
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
