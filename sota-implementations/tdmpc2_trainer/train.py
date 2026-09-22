# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import random

import hydra
import numpy as np
import torch
from torchrl.trainers.algorithms.configs import instantiate_trainer


def _set_seed(seed: int) -> None:
    """Seed the random generators used during model construction."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg):
    """Instantiate and run the configured TD-MPC2 trainer."""
    _set_seed(int(cfg.trainer.seed))
    trainer = instantiate_trainer(cfg)
    with trainer.stop_on_signal():
        trainer.train()


if __name__ == "__main__":
    main()
