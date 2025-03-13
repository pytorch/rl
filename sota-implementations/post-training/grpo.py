# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from argparse import ArgumentParser

import torch
from datasets import load_dataset
from tensordict import TensorDict
from torch.utils.data import DataLoader
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyStackStorage, ReplayBuffer, SamplerWithoutReplacement
from torchrl.envs import DataLoadingPrimer, KLRewardTransform, LLMEnv, StepCounter, Tokenizer
from torchrl.modules import from_hf_transformers
from torchrl.objectives import ClipPPOLoss, ReinforceLoss
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
from grpo_utils import ShapedCorrectnessReward, PrepareQuestion
from torch.utils._pytree import tree_map

parser = ArgumentParser()
parser.add_argument("--dataset", type=str, default="gsm8k")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--repeats", type=int, default=10)
parser.add_argument("--steps_per_batch", type=int, default=16)
parser.add_argument("--optim_batch_size", type=int, default=4)

def compute_mc_advantage(trajectories):
    # Get the question
    answer = trajectories["answer"]
    # Identify indices where the answers match
    answer_ids = tree_map(lambda string: hash(string), answer)
    answer_ids = torch.tensor(answer_ids)
    print("answer_ids", answer_ids)
    unique_qs = answer_ids.view(-1).unique()
    trajectories["advantage"] = trajectories["next", "reward"] * 0
    for u in unique_qs:
        idx =  answer_ids == u
        rewards = trajectories[idx]["next", "reward"]
        rewards = (rewards - rewards.mean()) / rewards.std().clamp(min=1e-4)
        print("rewards", rewards)
        trajectories.set_at_("advantage", rewards, idx)
    return trajectories

if __name__ == "__main__":
    args = parser.parse_args()
    # Create env instance:
    #  - Load the gsm8k dataset
    dataset = load_dataset(args.dataset, "main")
    train_dataset = dataset["train"]

    def collate_fn(batch):
        batch = torch.stack([TensorDict.from_dict(_batch) for _batch in batch])
        batch.rename_key_("question", "text")
        return batch

    # LLM
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel(GPT2Config())

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Env
    dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    env = LLMEnv.from_dataloader(
        dataloader=dataloader,
        tokenizer=tokenizer,
        str2str=True,
        batch_size=(args.batch_size * args.repeats,),
        repeats=args.repeats,
    )
    for i, trsf in enumerate(env.transform):
        if isinstance(trsf, DataLoadingPrimer):
            env.insert_transform(i, PrepareQuestion())
            break

    # Finally, we want the env to stop after the first step
    env.append_transform(StepCounter(max_steps=1))

    print("env", env)
    print(env.reset())

    policy = from_hf_transformers(
        model,
        tokenizer=tokenizer,
        from_text=False,
        generate=True,
        return_log_probs=True,
    )

    # Reward transform
    env.append_transform(ShapedCorrectnessReward(tokenizer=tokenizer))

    # Ref model
    ref_model = GPT2LMHeadModel(GPT2Config())
    ref_model = from_hf_transformers(
        ref_model,
        tokenizer=tokenizer,
        from_text=False,
        generate=False,
        return_log_probs=True,
    )
    env.append_transform(KLRewardTransform(actor=ref_model, coef=0.1, log_prob_key="log_probs"))

    # replay buffer
    rb = ReplayBuffer(storage=LazyStackStorage(args.steps_per_batch), sampler=SamplerWithoutReplacement(), batch_size=args.optim_batch_size)

    # Collector
    collector = SyncDataCollector(
        env, policy, frames_per_batch=args.steps_per_batch, total_frames=1_000_000,
    )

    # Loss module
    policy_traning = from_hf_transformers(
        model,
        tokenizer=tokenizer,
        from_text=False,
        generate=False,
        return_log_probs=True,
    )
    loss_fn = ClipPPOLoss(
        actor_network=policy_traning,
        critic_network=None,
        critic_coef=0.0,
        functional=False,
    )
    loss_fn.set_keys(sample_log_prob="log_probs")
    loss_fn._set_in_keys()
    optim = torch.optim.Adam(loss_fn.parameters())

    # loss_fn = ReinforceLoss(
    #     actor_network=policy,
    #     critic_network=None,
    #     critic_coef=0.0,
    # )

    for trajs in collector:
        trajs = trajs.reshape(-1)
        print('trajs from collector', trajs)
        trajs = compute_mc_advantage(trajs)
        rb.extend(trajs)
        for i in range(args.epochs):
            for batch in rb:
                print('running loss with batch', batch)
                loss = loss_fn(batch)
                loss_val = loss.mean(reduce=True)
                loss_val.backward()
                optim.step()
                optim.zero_grad()
