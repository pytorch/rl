# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import contextlib
import logging
from contextlib import nullcontext
from copy import deepcopy

import torch
import torch._dynamo

from hydra.utils import to_absolute_path
from models.reward import init_reward_model

from tensordict import TensorDict
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchrl.data import (
    LazyTensorStorage,
    RolloutFromModel,
    TensorDictReplayBuffer,
    TensorStorage,
)
from torchrl.data.replay_buffers import SamplerWithoutReplacement
from torchrl.data.rlhf.dataset import get_dataloader
from torchrl.data.rlhf.prompt import PromptData
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from torchrl.record.loggers import Logger
from transformers import GenerationConfig, GPT2Tokenizer


class TestPromptLogger:
    def __init__(self, batch, reward_model, logger, episode_length):
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        test_rindex = batch.prompt_rindex[0]
        test_prompt_ids = batch.input_ids[:1, :test_rindex]
        test_label_ids = batch.input_ids[:1, test_rindex:]
        test_prompt = tokenizer.decode(test_prompt_ids[0, :test_rindex].tolist())
        test_label = tokenizer.decode(
            test_label_ids[0, test_label_ids[0] != tokenizer.pad_token_id].tolist()
        )
        _, test_label_reward = reward_model(
            input_ids=batch.input_ids[:1], attention_mask=batch.attention_mask[:1]
        )
        self.generation_config = GenerationConfig(
            pad_token_id=tokenizer.pad_token_id, max_new_tokens=episode_length
        )
        self.test_prompt_ids = test_prompt_ids
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.test_label_reward = test_label_reward
        self.test_rindex = test_rindex
        self.test_prompt = test_prompt
        self.test_label = test_label
        self.logger = logger

    def log(self, model):
        response_ids = model.generate(
            input_ids=self.test_prompt_ids, generation_config=self.generation_config
        )
        _, response_reward = self.reward_model(
            input_ids=response_ids,
            attention_mask=(response_ids != self.tokenizer.pad_token_id).to(
                torch.int64
            ),
        )
        reward = (response_reward - self.test_label_reward).item()
        response_ids = response_ids[0, self.test_rindex :]
        response = self.tokenizer.decode(
            response_ids[response_ids != self.tokenizer.eos_token_id].tolist()
        )
        string_to_write = (
            f"Query:\n{self.test_prompt}\n"
            f"Response:\n{response}\n"
            f"Actual response:\n{self.test_label}\n"
            f"{reward=:4.4f}\n"
            f"====================================================\n"
        )
        self.logger.info(string_to_write)


class TrainLogger:
    def __init__(self, size: int, log_interval: int, logger: Logger):
        self.data = TensorDict({}, [size])
        self.counter = 0
        self.log_interval = log_interval
        self.logger = logger
        self.it = -1

    def __call__(self, data):
        done = data.get(("next", "done"))
        td_done = data[done.view(data.shape)]
        next_reward = td_done.get(("next", "reward_raw"))
        next_kl = td_done.get(("next", "reward_kl"))
        self.data[self.counter]["next_reward"] = next_reward.mean().cpu()
        self.data[self.counter]["next_kl"] = next_kl.mean().cpu()
        self.counter += 1

    def aggregate(self):
        result = {}
        for key, item in self.data.items():
            result[key] = item.mean()
        self.aggregated_data = TensorDict(result, [])

    def log(self):
        self.it += 1
        if self.it % self.log_interval == 0:
            for key, item in self.aggregated_data.items():
                self.logger.log_scalar(key, item)


class Evaluator:
    def __init__(
        self,
        *,
        reward_estimator,
        model,
        prompt_logger,
        io_cfg,
        val_reward_logger,
        val_loader,
        rlhf_out_dir,
        always_save_checkpoint=False,
        ctx=None,
        logger=None,
    ):
        self.reward_estimator = reward_estimator
        self.model = model
        self.prompt_logger = prompt_logger
        self.io_cfg = io_cfg
        self.eval_interval = io_cfg.eval_interval
        self.log_interval = io_cfg.log_interval
        self.eval_iters = io_cfg.eval_iters
        if ctx is None:
            ctx = contextlib.nullcontext()
        self.ctx = ctx
        self.val_reward_logger = val_reward_logger
        self.val_loader = val_loader
        self.always_save_checkpoint = always_save_checkpoint
        self.rlhf_out_dir = rlhf_out_dir
        self.logger = logger

        self.best_val_reward = -float("inf")
        self.it = 0

    def maybe_evaluate(self):
        self.it += 1
        if self.it % self.eval_interval == 0:
            with self.ctx:
                val_reward = self.reward_estimator(self.model, self.val_loader)
                self.prompt_logger.log(self.model)
            self.val_reward_logger.info(f"VALID: {self.it=}: {val_reward=:.4f}")
            self.logger.log_scalar("val_reward", val_reward, step=self.it)
            # pbar.set_description(f"VALID: {it=}: {val_reward=:.4f}")
            if val_reward > self.best_val_reward:
                self.best_val_reward = val_reward
            if self.always_save_checkpoint:
                if self.it > 0:
                    self.val_reward_logger.info(
                        f"saving checkpoint to {self.rlhf_out_dir}"
                    )
                    self.model.save_pretrained(self.rlhf_out_dir)


class RewardEstimator:
    """Create a class to estimate the reward via sampling.

    This class exposes a call method which, given a model and a dataloader, will
    perform multiple rollouts using the model and data sampled from the dataloader then
    average the accumulated rewards.

    For debugging purposes, we also generate responses to a fixed prompt so that the
    quality of the model can be visually assessed during training.

    """

    def __init__(self, eval_iters, episode_length, reward_model, ref_model):
        """
        Args:
            eval_iters (int): number of batches on which we would like to estimate reward

            episode_length (int): max number of generated new tokens

            reward_model (GPT2RewardModel): reward model

            ref_model (GPT2LMHeadModel): original transformer model that it is used to
                correctly compute kl component of reward.
        """
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.eval_iters = eval_iters
        self.episode_length = episode_length

    @torch.no_grad()
    def __call__(self, model, dataloader):
        rollout_from_model = RolloutFromModel(
            model,
            self.ref_model,
            self.reward_model,
            kl_coef=0,  # disable KL for evaluation
            max_new_tokens=self.episode_length,
        )
        rewards = torch.zeros(self.eval_iters)
        for k in range(self.eval_iters):
            batch = next(dataloader)
            td = rollout_from_model.rollout_from_data(batch)
            rewards[k] = td.get(("next", "reward")).sum(dim=1).mean().item()
        test_reward = rewards.mean()

        return test_reward


def resolve_name_or_path(name_or_path):
    """Hydra changes the working directory, so we need to absolutify paths."""
    if not name_or_path:
        return None
    if name_or_path.startswith("./") or name_or_path.startswith("/"):
        return to_absolute_path(name_or_path)
    return name_or_path


def get_file_logger(name, filename, level=logging.DEBUG):
    """
    Set up logger that will log to the given filename.
    """
    logger = logging.getLogger(name)
    handler = logging.FileHandler(filename)
    handler.setFormatter(
        # logging.Formatter("%(asctime)s, %(name)s %(levelname)s %(message)s")
        logging.Formatter("%(asctime)s - %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def setup(sys_cfg):
    """
    Set manual seed, configure backend and autocasting.
    """
    device = sys_cfg.device
    dtype = sys_cfg.dtype

    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    torch._dynamo.config.cache_size_limit = 256

    if "cuda" not in device:
        return nullcontext()

    return torch.amp.autocast(device_type="cuda", dtype=getattr(torch, dtype))


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


def make_evaluator(
    ppo_cfg,
    io_cfg,
    model_cfg,
    train_cfg,
    val_prompt_loader,
    model,
    ref_model,
    reward_model,
    ctx,
    logger,
):
    query_logger = get_file_logger("query_logger", "rlhf_query_logger.log")
    val_reward_logger = get_file_logger("val_reward_logger", "rlhf_valid_rewards.log")
    episode_length = ppo_cfg.episode_length
    rlhf_out_dir = model_cfg.out_dir
    always_save_checkpoint = train_cfg.always_save_checkpoint

    test_prompt = next(val_prompt_loader)
    prompt_logger = TestPromptLogger(
        batch=test_prompt,
        reward_model=reward_model,
        logger=query_logger,
        episode_length=episode_length,
    )
    reward_estimator = RewardEstimator(
        io_cfg.eval_iters, episode_length, reward_model, ref_model
    )

    evaluator = Evaluator(
        reward_estimator=reward_estimator,
        model=model,
        prompt_logger=prompt_logger,
        io_cfg=io_cfg,
        val_reward_logger=val_reward_logger,
        val_loader=val_prompt_loader,
        rlhf_out_dir=rlhf_out_dir,
        always_save_checkpoint=always_save_checkpoint,
        ctx=ctx,
        logger=logger,
    )
    return evaluator


def make_replay_buffer(ppo_cfg, data_cfg):
    return TensorDictReplayBuffer(
        storage=LazyTensorStorage(
            ppo_cfg.episode_length * ppo_cfg.num_rollouts_per_epoch
        ),
        batch_size=ppo_cfg.episode_length * data_cfg.batch_size,
        sampler=SamplerWithoutReplacement(),
        prefetch=10,
    )


def get_prompt_loaders(data_cfg, sys_cfg):
    train_prompt_loader = get_dataloader(
        data_cfg.batch_size,
        data_cfg.block_size,
        PromptData,
        sys_cfg.device,
        dataset_name="CarperAI/openai_summarize_tldr",
        split="train",
        num_workers=data_cfg.num_workers,
    )
    val_prompt_loader = get_dataloader(
        data_cfg.batch_size,
        data_cfg.block_size,
        PromptData,
        sys_cfg.device,
        dataset_name="CarperAI/openai_summarize_tldr",
        split="valid",
        num_workers=data_cfg.num_workers,
    )
    return train_prompt_loader, val_prompt_loader


def make_ref_model(model, sys_cfg):
    device = sys_cfg.ref_device
    ref_model = deepcopy(model).to(device)
    ref_model.requires_grad_(False)
    return ref_model


def freeze_layers(model):
    layers = model.transformer.h
    num_layers = len(layers)
    num_unfrozen = int(0.3 * num_layers)
    for layer in layers[:-num_unfrozen]:
        layer.requires_grad_(False)


def make_reward_model(reward_model_cfg, sys_cfg):
    device = sys_cfg.device
    compile_model = sys_cfg.compile
    reward_model = init_reward_model(
        reward_model_path=resolve_name_or_path(reward_model_cfg.name_or_path),
        device=device,
        compile_model=compile_model,
    )
    reward_model.eval()
    reward_model.requires_grad_(False)
    return reward_model


def make_loss(actor, critic, critic_head):
    advantage = GAE(
        value_network=critic, gamma=0.99, lmbda=0.95, average_gae=True, shifted=True
    )
    loss_fn = ClipPPOLoss(actor, critic_head)
    return loss_fn, advantage


def make_optimizer(train_cfg, loss_fn):
    optimizer = torch.optim.AdamW(
        [p for p in loss_fn.parameters() if p.requires_grad], **train_cfg.optimizer
    )
    scheduler = None
    if train_cfg.decay_lr:
        scheduler = CosineAnnealingLR(optimizer, **train_cfg.scheduler)
    return optimizer, scheduler


def make_sub_replay_buffer(data, batch_size):
    """A zero-copy sub-replay buffer."""
    # We expect some overhead due to the instantiation of the rb, storage and sampler
    # but hopefully these shouldn't be as big as copying data.
    # An optimized version of this would cache the rb, storage container and sampler and
    # just rewire to the new data location.
    storage = TensorStorage(data.exclude("index"))
    rb = TensorDictReplayBuffer(
        storage=storage, batch_size=batch_size, sampler=SamplerWithoutReplacement()
    )
    return rb
