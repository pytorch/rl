# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import hydra
import torch
from models.actor_critic import init_actor_critic
from torchrl.data.rlhf.utils import AdaptiveKLController, RolloutFromModel

from torchrl.record.loggers import get_logger

from tqdm import tqdm

from utils import (
    flatten_td,
    freeze_layers,
    get_prompt_loaders,
    make_evaluator,
    make_loss,
    make_optimizer,
    make_ref_model,
    make_replay_buffer,
    make_reward_model,
    make_sub_replay_buffer,
    resolve_name_or_path,
    setup,
    TrainLogger,
)


@hydra.main(version_base="1.1", config_path="config", config_name="train_rlhf")
def main(cfg):

    # ============ Retrieve config ============ #
    #############################################

    # make path absolute
    cfg.model.name_or_path = resolve_name_or_path(cfg.model.name_or_path)

    # Get some constants: number of iters, grad clip...
    batch_size = cfg.data.batch_size
    num_rollouts_per_epoch = cfg.train.ppo.num_rollouts_per_epoch
    collection_iters = num_rollouts_per_epoch // batch_size

    grad_clip = cfg.train.grad_clip
    max_epochs = cfg.train.max_epochs

    ppo_batch_size = cfg.train.ppo.ppo_batch_size
    ppo_num_epochs = cfg.train.ppo.ppo_num_epochs

    device = cfg.sys.device

    # ============ Instantiate utils ============ #
    ###############################################
    ctx = setup(cfg.sys)

    logger = get_logger(
        logger_type=cfg.io.logger,
        logger_name="./log",
        experiment_name="torchrlhf-gpt2",
        wandb_kwargs={
            "config": dict(cfg),
            "project": cfg.io.project_name,
            "group": cfg.io.group_name,
        },
    )

    # =============== Dataloaders =============== #
    ###############################################
    # We use prompts to get generated data from the generative model

    train_prompt_loader, val_prompt_loader = get_prompt_loaders(cfg.data, cfg.sys)

    # ================= Models ================= #
    ##############################################
    # Actor (gen model) - critic (value predictor)
    actor, critic, critic_head, model = init_actor_critic(cfg.model, cfg.sys)
    # Freeze initial model to use as ref
    ref_model = make_ref_model(model, sys_cfg=cfg.sys)
    # Freeze layers of the model -- can be customized
    freeze_layers(model)

    reward_model = make_reward_model(reward_model_cfg=cfg.reward_model, sys_cfg=cfg.sys)

    # ================= Loss and optimizer ================= #
    ##########################################################
    loss_fn, advantage = make_loss(actor, critic, critic_head)

    optimizer, lr_scheduler = make_optimizer(cfg.train, loss_fn)

    # ================= Replay buffer ================= #
    #####################################################
    rb = make_replay_buffer(cfg.train.ppo, cfg.data)

    # ================= Data collector ================= #
    ######################################################
    #
    # Because we interact with HuggingFace's transformers models,
    # using a Gym-like API (querying steps etc) introduces some
    # extra code that we can spare.
    #
    kl_scheduler = AdaptiveKLController(init_kl_coef=0.1, target=6, horizon=10000)
    rollout_from_model = RolloutFromModel(
        model,
        ref_model,
        reward_model,
        kl_scheduler=kl_scheduler,
        num_steps=collection_iters,
    )

    # ================= Evaluation utils ================= #
    ########################################################
    evaluator = make_evaluator(
        ppo_cfg=cfg.train.ppo,
        io_cfg=cfg.io,
        model_cfg=cfg.model,
        train_cfg=cfg.train,
        val_prompt_loader=val_prompt_loader,
        model=model,
        ref_model=ref_model,
        reward_model=reward_model,
        ctx=ctx,
        logger=logger,
    )

    # ================= Training loop ================= #
    #####################################################

    stats_logger = TrainLogger(
        collection_iters, log_interval=cfg.io.log_interval, logger=logger
    )
    pbar = tqdm(total=max_epochs * collection_iters)
    for _ in range(max_epochs):
        # ----------------- 1. Collect data, fill replay buffer ----------------- #
        # it's possible we didn't fill the replay buffer in the last iteration if
        # generation stopped early, so we empty first before repopulating
        rb.empty()
        for _ in range(collection_iters):
            batch = next(train_prompt_loader)
            td = rollout_from_model.rollout_from_data(batch)
            with torch.no_grad(), ctx:
                # TODO: moving this to within epoch
                advantage(td)
            rb.extend(flatten_td(td))
            stats_logger(td)
        stats_logger.aggregate()
        stats_logger.log()

        rollout_from_model.step_scheduler()

        # ----------------- 2. Feed model ----------------- #
        for batch in rb:
            rb_ppo = make_sub_replay_buffer(batch, batch_size=ppo_batch_size)
            for _ in range(ppo_num_epochs):  # PPO epochs
                optimizer.zero_grad()
                for minibatch in rb_ppo:  # GO over RB
                    minibatch = minibatch.to(device, non_blocking=True)
                    with ctx:
                        loss_vals = loss_fn(minibatch)
                    loss_val = sum(
                        value
                        for key, value in loss_vals.items()
                        if key.startswith("loss")
                    )
                    loss_val.backward()
                    torch.nn.utils.clip_grad_norm_(loss_fn.parameters(), grad_clip)
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
            pbar.update(1)

            # ----------------- 3. Possibly evaluate ----------------- #
            evaluator.maybe_evaluate()


if __name__ == "__main__":
    main()
