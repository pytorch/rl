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
)


@hydra.main(config_path=".", config_name="config")
def main(cfg: "DictConfig"):  # noqa: F821

    # cfg = correct_for_frame_skip(cfg)
    model_device = cfg.optim.device

    actor, critic = make_ppo_model(cfg)
    actor = actor.to(model_device)
    critic = critic.to(model_device)

    collector = make_collector(cfg, policy=actor)
    loss, adv_module = make_loss(cfg.loss, actor_network=actor, value_network=critic)
    optim = make_optim(cfg.optim, actor_network=actor, value_network=critic)
    logger = make_logger(cfg.logger)
    recorder = make_recorder(cfg, logger, actor)

    batch_size = cfg.optim.batch_size
    record_interval = cfg.recorder.interval

    pbar = tqdm.tqdm(total=cfg.collector.total_frames)
    collected_frames = 0

    # Main loop
    r0 = None
    l0 = None
    for data in collector:
        frames_in_batch = data.numel()
        collected_frames += frames_in_batch
        pbar.update(data.numel())
        data_view = data.reshape(-1)
        data_view = adv_module(data_view)  # TODO: modify after losses refactor
        import ipdb; ipdb.set_trace()
        data_view = loss(data_view)


if __name__ == "__main__":
    main()
