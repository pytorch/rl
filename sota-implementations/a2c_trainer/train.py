# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import hydra
import torchrl
from torchrl.trainers.algorithms.configs import (  # noqa: F401
    A2CTrainerConfig,
    instantiate_trainer,
)


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg):
    def print_reward(td):
        torchrl.logger.info(f"reward: {td['next', 'reward'].mean(): 4.4f}")

    trainer = instantiate_trainer(cfg)
    trainer.register_op(dest="batch_process", op=print_reward)
    with trainer.stop_on_signal():
        trainer.train()


if __name__ == "__main__":
    main()
