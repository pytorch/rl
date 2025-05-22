# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""

# TODO: make sure VLLM_USE_V1=0

$ python -m pip install peft
$ python -m pip install bitsandbytes
$ python -m pip install flash_attn
$ python -m pip install datasets

"""
from __future__ import annotations

import gc
import os
from argparse import ArgumentParser

import torch

import tqdm

from grpo_utils import get_train_model
from torchrl import logger as torchrl_logger
from torchrl.collectors.llm import LLMCollector
from torchrl.data import LazyStackStorage, ReplayBuffer, SamplerWithoutReplacement
from torchrl.envs import AsyncEnvPool, EnvBase, Transform

from torchrl.envs.llm.libs import make_mlgym

from torchrl.objectives.llm.grpo import GRPOLoss, MCAdvantage
from torchrl.record import WandbLogger

parser = ArgumentParser()
parser.add_argument("--dataset", type=str, default="gsm8k")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--repeats", type=int, default=16)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps_per_batch", type=int, default=64)
parser.add_argument("--optim_batch_size", type=int, default=4)
# parser.add_argument("--model_name", type=str, default="gpt2")
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-7B-Instruct")
parser.add_argument("--compile", action="store_true")
parser.add_argument("--clip_grad_norm", type=float, default=0.5)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--kl_coef", type=float, default=1e-2)


parser.add_argument("--gpu_memory_utilization", type=float, default=0.5)

torch.set_default_dtype(torch.bfloat16)

torch.set_default_device("cuda:0")

if not os.getenv("VLLM_USE_V1", "0"):
    raise ValueError("VLLM_USE_V1=0 not set")


def make_env(transform: Transform | None = None) -> EnvBase:
    env = make_mlgym()
    if transform is not None:
        env = env.append_transform(transform)
    return env


if __name__ == "__main__":
    import ray

    ray.init()

    args = parser.parse_args()

    policy_training, train_tokenizer = get_train_model(args, [0])

    # Here we have only one policy
    policy = policy_training

    # Env
    # trsf =         KLRewardTransform(
    #         actor=ref_model,
    #         coef=args.kl_coef,
    #         device=ref_device,
    #         add_to_reward=False,
    #     )

    env = AsyncEnvPool([make_mlgym] * args.num_envs, backend="multiprocessing")

    # replay buffer
    rb = ReplayBuffer(
        storage=LazyStackStorage(args.steps_per_batch),
        sampler=SamplerWithoutReplacement(),
        batch_size=args.optim_batch_size,
    )
    rb.append_transform(MCAdvantage(grpo_size=args.repeats))

    # Collector
    collector = LLMCollector(
        env,
        policy=policy,
        dialog_turns_per_batch=args.steps_per_batch,
        total_dialog_turns=1_000_000,
        async_envs=True,
        replay_buffer=rb,
    )
    for d in collector:
        print(d)
        print(rb[:])
        break

    # Loss module
    loss_fn = GRPOLoss(actor_network=policy_training, kl_to_ref_coeff=args.kl_coef)

    if args.compile:
        loss_fn = torch.compile(loss_fn)

    # TODO: foreach=False to avoid "Tensors of the same index must be on the same device" error due to "auto" device map
    optim = torch.optim.AdamW(policy_training.model.parameters(), lr=args.lr)
    logger = WandbLogger(exp_name=args.model_name)

    for i, trajs in enumerate(collector):
        torchrl_logger.info(f"Collected batch {i}: {trajs=}")

        # rb.empty()
        trajs = trajs.reshape(-1)
        rb.extend(trajs)

        # logging
        reward = torch.cat(rb[:].get(("next", "reward"), as_list=True)).mean()
        advantage = torch.cat(rb[:].get("advantage", as_list=True)).mean()
        kl_penalty = torch.cat(rb[:].get(("next", "kl_penalty"), as_list=True)).mean()
        seq_length = []
        for t in rb[:].get("tokens_response", as_list=True):
            seq_length.append(t.numel())
        seq_length = torch.tensor(seq_length, dtype=torch.float).mean()

        if not reward:
            # no use in training a model without reward
            torchrl_logger.info("no reward - skipping")
            torch.cuda.empty_cache()  # TODO: Test if this is needed
            continue
        logger.log_scalar("reward", reward)
        logger.log_scalar("advantage", advantage)
        logger.log_scalar("kl_penalty", kl_penalty)
        logger.log_scalar("seq_length", seq_length)

        torchrl_logger.info(f"reward: {reward: 4.4f}")
        for i in range(args.epochs):
            torchrl_logger.info(f"epoch: {i}")
            pbar = tqdm.tqdm(total=len(rb) // args.optim_batch_size)
            for batch in rb:
                pbar.update(1)
                optim.zero_grad()
                batch = batch.to(train_devices[0])
                loss = loss_fn(batch)
                loss_val = loss.mean(reduce=True)
                loss_val.backward()
                gn = torch.nn.utils.clip_grad_norm_(
                    policy_training.model.parameters(), args.clip_grad_norm
                )
                optim.step()

                logger.log_scalar("ESS", loss.ESS)
                logger.log_scalar("loss_objective", loss.loss_objective)
                logger.log_scalar("clip_fraction", loss.clip_fraction)
                logger.log_scalar("kl_approx", loss.kl_approx)
                logger.log_scalar("grad_norm", gn)
                logger.log_scalar("entropy", loss.loss_entropy.mean())
                logger.log_scalar("kl_to_ref", loss.kl_to_ref.mean())
                logger.log_scalar("loss_kl_to_ref", loss.loss_kl_to_ref.mean())

                # scaler.update()

                gc.collect()
                torch.cuda.empty_cache()

        torchrl_logger.info("Updating weights...")
        collector.update_policy_weights_(
            policy_weights=policy_training.model.merge_and_unload().state_dict(),
            worker_ids=[0],
        )
        gc.collect()
        torch.cuda.empty_cache()
