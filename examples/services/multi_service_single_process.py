# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Run the multi-service training example in one process.

The inference server runs in a background thread. Logger and replay-buffer
clients are direct identity clients. The TensorDict training loop is identical
to the process and Ray examples.

Run from the repository root:

    python examples/services/multi_service_single_process.py
"""

from __future__ import annotations

import argparse

from multi_service_utils import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--log-dir", default="/tmp/torchrl-service-example/single-process"
    )
    args = parser.parse_args()
    run_training(
        service_backend="direct",
        steps=args.steps,
        batch_size=args.batch_size,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()
