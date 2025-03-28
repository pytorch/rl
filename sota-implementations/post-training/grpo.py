# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from argparse import ArgumentParser
import os
from torchrl._utils import logger as torchrl_logger
import torch
from datasets import load_dataset
from grpo_utils import (
    HF2vLLMLocalWeightUpdater,
    PrepareQuestion,
    ShapedCorrectnessReward,
)
from tensordict import TensorDict
from torch.utils._pytree import tree_map
from torch.utils.data import DataLoader
from torchrl.collectors import SyncDataCollector
from torchrl.data import (
    LazyStackStorage,
    ReplayBuffer,
    SamplerWithoutReplacement,
)
from torchrl.envs import KLRewardTransform, LLMEnv, StepCounter
from torchrl.modules import TransformersWrapper, vLLMWrapper
from torchrl.objectives import ClipPPOLoss
from torchrl.record import WandbLogger
from transformers import GPT2LMHeadModel, Qwen2ForCausalLM
from vllm import LLM

parser = ArgumentParser()
parser.add_argument("--dataset", type=str, default="gsm8k")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--repeats", type=int, default=16)
parser.add_argument("--steps_per_batch", type=int, default=16)
parser.add_argument("--optim_batch_size", type=int, default=4)
# parser.add_argument("--model_name", type=str, default="gpt2")
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B")


def compute_mc_advantage(trajectories):
    # Get the question
    answer = trajectories["answer"]
    # Identify indices where the answers match
    answer_ids = tree_map(lambda string: hash(string), answer)
    answer_ids = torch.tensor(answer_ids)
    unique_qs = answer_ids.view(-1).unique()
    trajectories["advantage"] = trajectories["next", "reward"] * 0
    for u in unique_qs:
        idx = answer_ids == u
        rewards = trajectories[idx]["next", "reward"]
        rewards = (rewards - rewards.mean()) / rewards.std().clamp(min=1e-4)
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
    # inference_model = GPT2LMHeadModel(GPT2Config())
    devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    inference_model = LLM(args.model_name)
    tokenizer = inference_model.get_tokenizer()
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = devices
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

    # Env
    dataloader = DataLoader(  # noqa: TOR401
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    env = LLMEnv.from_dataloader(
        dataloader=dataloader,
        # tokenizer=tokenizer,
        str2str=True,
        batch_size=(args.batch_size,),
        repeats=args.repeats,
        group_repeats=True,
    )
    env.insert_transform(0, PrepareQuestion())

    # Finally, we want the env to stop after the first step
    env.append_transform(StepCounter(max_steps=1))

    policy = vLLMWrapper(
        inference_model,
        tokenizer=tokenizer,
        from_text=True,
        generate=True,
        # vLLM log-probs are a bit screwed up, we could use something else
        return_log_probs=True,
    )

    # Reward transform
    env.append_transform(ShapedCorrectnessReward(tokenizer=tokenizer))

    # Ref model
    devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    if args.model_name == "Qwen/Qwen2.5-3B":
        ref_model = Qwen2ForCausalLM.from_pretrained(args.model_name).eval()
    else:
        ref_model = GPT2LMHeadModel.from_pretrained(args.model_name).eval()
    # Detach weights
    TensorDict.from_module(ref_model).data.to_module(ref_model)
    ref_model = TransformersWrapper(
        ref_model,
        tokenizer=tokenizer,
        from_text=False,
        generate=False,
        return_log_probs=True,
    )
    if devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = devices
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

    env.append_transform(
        KLRewardTransform(actor=ref_model, coef=0.1, log_prob_key="log_probs")
    )

    # replay buffer
    rb = ReplayBuffer(
        storage=LazyStackStorage(args.steps_per_batch),
        sampler=SamplerWithoutReplacement(),
        batch_size=args.optim_batch_size,
    )

    # Collector
    if args.model_name == "Qwen/Qwen2.5-3B":
        train_model = Qwen2ForCausalLM.from_pretrained(args.model_name).eval()
        train_model.gradient_checkpointing_enable()
    else:
        train_model = GPT2LMHeadModel.from_pretrained(args.model_name).eval()
    collector = SyncDataCollector(
        env,
        policy,
        frames_per_batch=args.steps_per_batch,
        total_frames=1_000_000,
        local_weight_updater=HF2vLLMLocalWeightUpdater(
            hf_model=train_model, vllm_model=inference_model
        ),
        use_buffers=False,
    )
    # # Try to update the weights
    # torchrl_logger.info('updating weights...')
    # collector.update_policy_weights_()
    # # make sure we can collect data
    # for batch in collector:
    #     print('first batch', batch)
    #     break

    # Loss module
    policy_training = TransformersWrapper(
        train_model,
        tokenizer=tokenizer,
        # We have the tokens, let's just use them
        from_text=False,
        generate=False,
        return_log_probs=True,
    )
    loss_fn = ClipPPOLoss(
        actor_network=policy_training,
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

    logger = WandbLogger(exp_name=args.model_name)
    for i, trajs in enumerate(collector):
        print("Collected batch", i)
        trajs = trajs.reshape(-1)
        trajs = compute_mc_advantage(trajs)
        rb.extend(trajs)
        # logging
        reward = torch.cat(rb[:].get(("next", "reward"), as_list=True)).mean()
        logger.log_scalar("reward", reward)
        for _ in range(args.epochs):
            for batch in rb:
                loss = loss_fn(batch)
                loss_val = loss.mean(reduce=True)
                loss_val.backward()
                optim.step()
                optim.zero_grad(set_to_none=True)
        collector.update_policy_weights_()
