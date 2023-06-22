# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from copy import deepcopy

import numpy as np
import torch

import wandb
from models.actor_critic import init_actor_critic
from models.reward import init_reward_model

from omegaconf import OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchrl.data import LazyTensorStorage
from torchrl.data.replay_buffers import (
    SamplerWithoutReplacement,
    TensorDictReplayBuffer,
)
from torchrl.data.rlhf.dataset import get_dataloader
from torchrl.data.rlhf.prompt import PromptData
from torchrl.data.rlhf.utils import RolloutFromModel

from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm
from transformers import GenerationConfig, GPT2Tokenizer
from utils import get_file_logger, resolve_name_or_path, setup


def flatten_td(td):
    # our tensordict has shape [B, T] where B = batch_size and T = trajectory length
    # some trajectories may have stopped (reached EOS) before generating T tokens
    # this function truncates and concatenates the trajectories, resulting in a
    # tensordict that has shape [N] where N <= B * T.
    done = td["next", "done"]
    mask = torch.zeros_like(done)
    mask[..., 1:, :] = done[..., :-1, :]  # shift by one
    mask = ~mask.cumsum(-2).bool().squeeze()
    return td[mask]


class AdaptiveKLController:
    """Adaptive KL Controller as described in Ziegler et al. "Fine-Tuning Language Models from Human Preferences"
    Reference: Section 2.2 https://arxiv.org/pdf/1909.08593.pdf#page=2
    Source: https://github.com/openai/lm-human-preferences/blob/master/lm_human_preferences/train_policy.py
    """

    def __init__(self, init_kl_coef: float, target: float, horizon: int):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current: float, n_steps: int):
        """Returns adaptively updated KL coefficient, βₜ₊₁.
        Arguments:
            current: The current KL value between the newest policy and the initial policy.
        """
        proportional_error = np.clip(current / self.target - 1, -0.2, 0.2)  # ϵₜ
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult  # βₜ₊₁
        return self.value


def create_reward_estimator(
    eval_iters, episode_length, reward_model, batch, ctx, logger=None, ref_model=None
):
    """Create a function to estimate the reward via sampling.

    This function creates a new function which, given a model and a dataloader, will
    perform multiple rollouts using the model and data sampled from the dataloader then
    average the accumulated rewards.

    For debugging purposes, we also generate responses to a fixed prompt so that the
    quality of the model can be visually assessed during training.
    """
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    test_rindex = batch.prompt_rindex[0]
    test_prompt_ids = batch.input_ids[:1, :test_rindex]
    test_label_ids = batch.input_ids[:1, test_rindex:]
    generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id, max_new_tokens=episode_length
    )
    test_prompt = tokenizer.decode(test_prompt_ids[0, :test_rindex].tolist())
    test_label = tokenizer.decode(
        test_label_ids[0, test_label_ids[0] != tokenizer.pad_token_id].tolist()
    )
    _, test_label_reward = reward_model(
        input_ids=batch.input_ids[:1], attention_mask=batch.attention_mask[:1]
    )

    @torch.no_grad()
    def estimate_reward(model, dataloader):
        rollout_from_model = RolloutFromModel(model, ref_model, reward_model)
        rewards = torch.zeros(eval_iters)
        for k in range(eval_iters):
            batch = next(dataloader)
            # NOTE: disable kl for evaluation
            td = rollout_from_model.rollout_from_data(batch, kl_coef=0.0)
            rewards[k] = td.get(("next", "reward")).sum(dim=1).mean().item()
        test_reward = rewards.mean()

        if logger:
            response_ids = model.generate(
                input_ids=test_prompt_ids, generation_config=generation_config
            )
            with ctx:
                _, response_reward = reward_model(
                    input_ids=response_ids,
                    attention_mask=(response_ids != tokenizer.pad_token_id).to(
                        torch.int64
                    ),
                )
            reward = (response_reward - test_label_reward).item()
            response_ids = response_ids[0, test_rindex:]
            response = tokenizer.decode(
                response_ids[response_ids != tokenizer.eos_token_id].tolist()
            )
            string_to_write = (
                f"Query:\n{test_prompt}\n"
                f"Response:\n{response}\n"
                f"Actual response:\n{test_label}\n"
                f"{reward=:4.4f}, "
                f"{test_reward=:4.4f}\n"
                f"====================================================\n"
            )
            logger.info(string_to_write)

        return test_reward

    return estimate_reward


# @hydra.main(version_base="1.1", config_path="config", config_name="train_rlhf")
def main():
    cfg = OmegaConf.load("config/train_rlhf.yaml")
    wandb.init(
        # set the wandb project where this run will be logged
        project="rlhf-training",
        # track hyperparameters and run metadata
        config=cfg,
    )
    query_logger = get_file_logger("query_logger", "rlhf_query_logger.log")
    val_reward_logger = get_file_logger("val_reward_logger", "rlhf_valid_rewards.log")

    data_cfg = cfg.data
    model_cfg = cfg.model
    reward_model_cfg = cfg.reward_model
    train_cfg = cfg.train
    ppo_cfg = train_cfg.ppo

    eval_interval = cfg.io.eval_interval
    log_interval = cfg.io.log_interval
    eval_iters = cfg.io.eval_iters

    rlhf_out_dir = model_cfg.out_dir
    transformer_name_or_path = model_cfg.name_or_path
    dropout = model_cfg.dropout

    batch_size = data_cfg.batch_size

    grad_clip = train_cfg.grad_clip
    max_epochs = train_cfg.max_epochs
    always_save_checkpoint = train_cfg.always_save_checkpoint

    episode_length = ppo_cfg.episode_length
    ppo_batch_size = ppo_cfg.ppo_batch_size
    ppo_num_epochs = ppo_cfg.ppo_num_epochs
    num_rollouts_per_epoch = ppo_cfg.num_rollouts_per_epoch

    device = cfg.sys.device
    dtype = cfg.sys.dtype
    compile_ = cfg.sys.compile

    ctx = setup(device, dtype)

    train_loader = get_dataloader(
        data_cfg.batch_size,
        data_cfg.block_size,
        PromptData,
        device,
        dataset_name="CarperAI/openai_summarize_tldr",
        split="train",
    )
    val_loader = get_dataloader(
        data_cfg.batch_size,
        data_cfg.block_size,
        PromptData,
        device,
        dataset_name="CarperAI/openai_summarize_tldr",
        split="valid",
    )

    actor, critic, critic_head, model = init_actor_critic(
        resolve_name_or_path(transformer_name_or_path), dropout, device, compile_
    )
    ref_model = deepcopy(model).to("cuda:1")
    ref_model.requires_grad_(False)
    layers = model.transformer.h
    num_layers = len(layers)
    num_unfrozen = int(0.3 * num_layers)
    for layer in layers[:-num_unfrozen]:
        layer.requires_grad_(False)

    reward_model = init_reward_model(
        reward_model_path=resolve_name_or_path(reward_model_cfg.name_or_path),
        device=device,
        compile_=compile_,
    )
    reward_model.eval()
    reward_model.requires_grad_(False)

    adv_fn = GAE(
        value_network=critic, gamma=0.99, lmbda=0.95, average_gae=True, shifted=True
    )
    loss_fn = ClipPPOLoss(actor, critic_head)

    test_prompt = next(val_loader)
    estimate_reward = create_reward_estimator(
        eval_iters,
        episode_length,
        reward_model,
        test_prompt,
        ctx,
        logger=query_logger,
        ref_model=ref_model,
    )

    optimizer = torch.optim.AdamW(
        [p for p in loss_fn.parameters() if p.requires_grad], **train_cfg.optimizer
    )
    scheduler = None
    if train_cfg.decay_lr:
        scheduler = CosineAnnealingLR(optimizer, **train_cfg.scheduler)

    rb = TensorDictReplayBuffer(
        storage=LazyTensorStorage(episode_length * num_rollouts_per_epoch),
        batch_size=episode_length * batch_size,
        sampler=SamplerWithoutReplacement(),
        prefetch=10,
    )
    rb_ppo = TensorDictReplayBuffer(
        storage=LazyTensorStorage(episode_length * batch_size),
        batch_size=ppo_batch_size,
        sampler=SamplerWithoutReplacement(),
        prefetch=10,
    )

    rollout_from_model = RolloutFromModel(model, ref_model, reward_model)

    best_val_reward = float("-inf")
    it = 0  # it is equivalent to batch_size number of episodes
    with tqdm(total=int(max_epochs * num_rollouts_per_epoch / batch_size)) as pbar:
        for _epoch in range(1, max_epochs + 1):
            rb.empty()
            rollout_rewards = []
            rollout_kl = []
            kl_controller = AdaptiveKLController(0.1, 6, 10000)
            for _ in range(0, num_rollouts_per_epoch, batch_size):
                batch = next(train_loader)
                td = rollout_from_model.rollout_from_data(
                    batch, kl_coef=kl_controller.value
                )
                with torch.no_grad(), ctx:
                    # moving this to within epoch
                    adv_fn(td)
                # it's possible we didn't fill the replay buffer in the last iteration if
                # generation stopped early, so we empty first before repopulating
                rb.extend(flatten_td(td))
                done = td.get(("next", "done"))
                next_reward = td.get(("next", "reward_raw"))[done]
                next_kl = td.get(("next", "reward_kl"))[done]
                rollout_rewards.append(next_reward.mean().cpu().item())
                rollout_kl.append(next_kl.mean().cpu().item())
            rollout_reward = torch.tensor(rollout_rewards).mean().cpu().item()
            rollout_kl_reward = torch.tensor(rollout_kl).mean().cpu().item()
            # recover true kl
            rollout_kl = -rollout_kl_reward / kl_controller.value
            kl_controller.update(rollout_kl, num_rollouts_per_epoch / batch_size)

            # FIXME: THIS PPO CYCLE WAS DIFFERENT wrt trlx. @tcbegley please double check
            # they sample batch_size from rb and then do minibatches ppo_batch_size within
            if it % log_interval == 0:
                val_reward_logger.info(
                    f"TRAIN: {it=}: {rollout_reward=:.4f} {rollout_kl_reward=:.4f} {rollout_kl=:.4f}"
                )
                wandb.log(
                    {
                        "rollout_reward": rollout_reward,
                        "rollout_kl_reward": rollout_kl_reward,
                        "rollout_kl": rollout_kl,
                    },
                    step=it,
                )
                pbar.set_description(f"TRAIN: {it=}: {rollout_reward=:.4f}")

            for batch in rb:
                rb_ppo.empty()
                rb_ppo.extend(batch)
                for _ in range(ppo_num_epochs):  # PPO epochs
                    optimizer.zero_grad()
                    # why don't we optimize at each step? Is accumulating grads better?
                    # usually more small steps is better than a giant one
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
                    if scheduler is not None:
                        scheduler.step()
                it += 1
                pbar.update(1)
                if it % eval_interval == 0:
                    val_reward = estimate_reward(model, val_loader)
                    val_reward_logger.info(f"VALID: {it=}: {val_reward=:.4f}")
                    wandb.log({"val_reward": val_reward}, step=it)
                    pbar.set_description(f"VALID: {it=}: {val_reward=:.4f}")
                    if val_reward > best_val_reward or always_save_checkpoint:
                        best_val_reward = val_reward
                        if it > 0:
                            val_reward_logger.info(
                                f"saving checkpoint to {rlhf_out_dir}"
                            )
                            model.save_pretrained(rlhf_out_dir)


if __name__ == "__main__":
    main()
