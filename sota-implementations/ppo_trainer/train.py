# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import hydra
import torchrl


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg):
    def print_reward(td):
        torchrl.logger.info(f"reward: {td['next', 'reward'].mean(): 4.4f}")

    trainer = hydra.utils.instantiate(cfg.trainer)
    trainer.register_op(dest="batch_process", op=print_reward)
    trainer.train()


if __name__ == "__main__":
    main()
