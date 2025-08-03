# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import hydra

from torchrl.trainers.algorithms.configs.common import Config


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg: Config):
    print(f"{cfg=}")
    trainer = hydra.utils.instantiate(cfg.trainer)
    trainer.train()

if __name__ == "__main__":
    main()
