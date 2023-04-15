# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import hydra
from utils import (
    make_collector,
    make_ppo_model,
)


@hydra.main(config_path=".", config_name="config")
def main(cfg: "DictConfig"):  # noqa: F821

    model_device = cfg.optim.device
    actor, critic = make_ppo_model(cfg)
    actor = actor.to(model_device)
    collector = make_collector(cfg, policy=actor)

    # Main loop
    for data in collector:
        episode_rewards = data["next"]["episode_reward"][data["next"]["done"]]
        if len(episode_rewards) > 0:
            import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    main()
