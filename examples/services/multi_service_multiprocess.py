# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Run the multi-service training example with process owners.

Policy inference and CSV logging each run in a spawned process. Actor threads
receive restricted process clients and share the driver's direct replay
buffer, since replay buffers do not yet expose a process service backend.

Run from the repository root:

    python examples/services/multi_service_multiprocess.py
"""

from __future__ import annotations

import argparse

from multi_service_utils import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-actors", type=int, default=4)
    parser.add_argument("--steps-per-actor", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--log-dir", default="/tmp/torchrl-service-example/multiprocess"
    )
    args = parser.parse_args()
    run_training(
        service_backend="process",
        num_actors=args.num_actors,
        steps_per_actor=args.steps_per_actor,
        batch_size=args.batch_size,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()
