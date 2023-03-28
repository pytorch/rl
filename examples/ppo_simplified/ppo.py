# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""PPO Example.

This is a self-contained example of a PPO training script.

It works across Gym and DM-control over a variety of tasks.

Both state and pixel-based environments are supported.

The helper functions are coded in the utils.py associated with this script.
"""

import hydra
import tqdm
from torchrl.trainers.helpers.envs import correct_for_frame_skip

from utils import (
    get_stats,
    make_collector,
    make_ppo_model,
    make_policy,  # needed ???
    make_logger,
    make_loss,
    make_optim,
    make_recorder,
    make_advantage_module
)


@hydra.main(config_path=".", config_name="config")
def main(cfg: "DictConfig"):  # noqa: F821

    cfg = correct_for_frame_skip(cfg)
    model_device = cfg.optim.device

    actor_network, value_network = make_ppo_model(cfg)
    actor_network = actor_network.to(model_device)
    value_network = value_network.to(model_device)

    policy, critic = make_policy(cfg.model, actor_network)
    collector = make_collector(cfg, policy=policy)
    # loss, adv_module = make_loss(cfg.loss, actor_network, value_network)
    # optim = make_optim(cfg.optim, actor_network, value_network)
    logger = make_logger(cfg.logger)
    # recorder = make_recorder(cfg, logger, policy)

    # optim_steps_per_batch = cfg.optim.optim_steps_per_batch
    # batch_size = cfg.optim.batch_size
    # record_interval = cfg.recorder.interval

    pbar = tqdm.tqdm(total=cfg.collector.total_frames)
    collected_frames = 0

    # Main loop
    r0 = None
    l0 = None
    for data in collector:
        import ipdb; ipdb.set_trace()
        frames_in_batch = data.numel()
        collected_frames += frames_in_batch
        pbar.update(data.numel())


if __name__ == "__main__":
    main()
