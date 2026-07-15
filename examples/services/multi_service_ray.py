# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Run the multi-service training example with Ray.

The logger, replay buffer, and inference policy are Ray service owners. The
driver runs the same ordinary TensorDict loop as the other examples through
their restricted component clients.

Install Ray and run from the repository root:

    pip install ray
    python examples/services/multi_service_ray.py
"""

from __future__ import annotations

import argparse

from multi_service_utils import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--log-dir", default="/tmp/torchrl-service-example/ray")
    args = parser.parse_args()
    run_training(
        service_backend="ray",
        steps=args.steps,
        batch_size=args.batch_size,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()
