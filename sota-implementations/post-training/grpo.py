# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import gc
from argparse import ArgumentParser

import torch

import tqdm
from datasets import load_dataset
from grpo_utils import (
    cuda_visible_devices,
    get_unsloth_model,
    HF2vLLMLocalWeightUpdater,
    PrepareQuestion,
    ShapedCorrectnessReward,
)
from tensordict import TensorDict
from torch.utils._pytree import tree_map
from torch.utils.data import DataLoader
from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyStackStorage, ReplayBuffer, SamplerWithoutReplacement
from torchrl.envs import KLRewardTransform, LLMEnv, StepCounter
from torchrl.modules import TransformersWrapper, vLLMWrapper
from torchrl.objectives import ClipPPOLoss
from torchrl.record import WandbLogger
from vllm import LLM

parser = ArgumentParser()
parser.add_argument("--dataset", type=str, default="gsm8k")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--repeats", type=int, default=16)
parser.add_argument("--steps_per_batch", type=int, default=16)
parser.add_argument("--optim_batch_size", type=int, default=4)
# parser.add_argument("--model_name", type=str, default="gpt2")
parser.add_argument("--model_name", type=str, default="unsloth/Qwen2.5-3B")
parser.add_argument("--compile", action="store_true")

# torch.set_default_dtype(torch.bfloat16)

torch.set_default_device("cuda:0")


def make_device_splits():
    # devices = list(range(torch.cuda.device_count()))
    # train_devices = devices[1:-1]
    # vllm_device = devices[0]
    # ref_device = devices[-1]
    devices = list(range(torch.cuda.device_count()))
    train_devices = devices[:-2]
    vllm_device = devices[-1]
    ref_device = devices[-2]
    return train_devices, ref_device, vllm_device


def compute_mc_advantage(trajectories):
    # Get the question
    answer = trajectories["answer"]
    # Identify indices where the answers match
    answer_ids = tree_map(lambda string: hash(string), answer)
    answer_ids = torch.tensor(answer_ids)
    unique_qs = answer_ids.view(-1).unique()
    trajectories["advantage"] = trajectories.get(
        ("next", "reward"), as_nested_tensor=True, layout=torch.strided
    )
    for u in unique_qs:
        idx = (answer_ids == u).nonzero(as_tuple=True)[0]
        rewards = trajectories[idx].get(
            ("next", "reward"), as_nested_tensor=True, layout=torch.strided
        )
        rewards = (rewards - rewards.values().mean()) / rewards.values().std().clamp(
            min=1e-4
        )
        trajectories.set_at_("advantage", rewards, idx)
    return trajectories


if __name__ == "__main__":
    args = parser.parse_args()

    train_devices, ref_device, vllm_device = make_device_splits()

    # Load the train model first, since unsloth does some disturbing monkey patching
    if args.model_name.startswith("unsloth"):
        train_model, train_tokenizer = get_unsloth_model(
            args.model_name, devices=train_devices
        )
        # train_tokenizer = train_tokenizer.to(train_devices[0])
    elif args.model_name == "Qwen/Qwen2.5-3B":
        from transformers import Qwen2ForCausalLM

        train_model = Qwen2ForCausalLM.from_pretrained(args.model_name).eval()
        train_model.gradient_checkpointing_enable()
    elif "gpt2" in args.model_name:
        # for debugging
        from transformers import GPT2LMHeadModel

        train_model = GPT2LMHeadModel.from_pretrained(args.model_name).eval()
    else:
        raise NotImplementedError

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
    with torch.device(f"cuda:{vllm_device}"):
        if args.model_name.startswith("unsloth"):
            model_name = args.model_name
            model_name = model_name.replace("unsloth", "Qwen")
        with cuda_visible_devices([vllm_device]):
            inference_model = LLM(model_name, gpu_memory_utilization=0.5)
        inference_tokenizer = inference_model.get_tokenizer()
        inference_tokenizer.pad_token = inference_tokenizer.eos_token
        inference_tokenizer.padding_side = "left"

    # Env
    seed = int(torch.empty((), dtype=torch.int64).random_().item())
    generator = torch.Generator(device=torch.get_default_device())
    generator.manual_seed(seed)

    dataloader = DataLoader(  # noqa: TOR401
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        generator=generator,
    )
    env = LLMEnv.from_dataloader(
        dataloader=dataloader,
        # tokenizer=tokenizer,
        str2str=True,
        batch_size=(args.batch_size,),
        repeats=args.repeats,
        group_repeats=True,
        # assign_reward=True,
    )
    env.insert_transform(0, PrepareQuestion())

    # Finally, we want the env to stop after the first step
    env.append_transform(StepCounter(max_steps=1))

    policy = vLLMWrapper(
        inference_model,
        tokenizer=inference_tokenizer,
        from_text=True,
        generate=True,
        # vLLM log-probs are a bit screwed up, we could use something else
        return_log_probs=True,
        generate_kwargs={"max_tokens": 512},
    )

    # Reward transform
    env.append_transform(ShapedCorrectnessReward(tokenizer=inference_tokenizer))

    # Ref model
    with torch.device(f"cuda:{ref_device}"):
        if args.model_name.startswith("unsloth"):
            model_name = args.model_name
            model_name = model_name.replace("unsloth", "Qwen")
        if model_name == "Qwen/Qwen2.5-3B":
            from transformers import Qwen2ForCausalLM

            ref_model = Qwen2ForCausalLM.from_pretrained(model_name).eval()
        elif "gpt2" in model_name:
            from transformers import GPT2LMHeadModel

            ref_model = GPT2LMHeadModel.from_pretrained(model_name).eval()
        else:
            raise NotImplementedError(model_name)
        # Detach weights
        TensorDict.from_module(ref_model).data.to_module(ref_model)
        ref_model = TransformersWrapper(
            ref_model,
            tokenizer=inference_tokenizer,
            from_text=False,
            generate=False,
            return_log_probs=True,
        )

    env.append_transform(
        KLRewardTransform(
            actor=ref_model,
            coef=0.1,
            log_prob_key="log_probs",
            functional=False,
            device=ref_device,
        )
    )

    # replay buffer
    rb = ReplayBuffer(
        storage=LazyStackStorage(args.steps_per_batch),
        sampler=SamplerWithoutReplacement(),
        batch_size=args.optim_batch_size,
    )

    # Collector
    collector = SyncDataCollector(
        env,
        policy,
        frames_per_batch=args.steps_per_batch,
        total_frames=1_000_000,
        local_weight_updater=HF2vLLMLocalWeightUpdater(
            hf_model=train_model,
            vllm_model=inference_model,
            use_unsloth=args.model_name.startswith("unsloth"),
        ),
        use_buffers=False,
    )
    collector.update_policy_weights_()

    # Loss module
    policy_training = TransformersWrapper(
        train_model,
        tokenizer=inference_tokenizer,
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
    # We don't want to use the string action but the tokens
    loss_fn._set_in_keys()
    loss_fn.set_keys(sample_log_prob="log_probs", action="tokens_response")
    if args.compile:
        loss_fn = torch.compile(loss_fn)
    # TODO: foreach=False to avoid "Tensors of the same index must be on the same device" error due to "auto" device map
    optim = torch.optim.AdamW(loss_fn.parameters(), foreach=False)

    # loss_fn = ReinforceLoss(
    #     actor_network=policy,
    #     critic_network=None,
    #     critic_coef=0.0,
    # )

    logger = WandbLogger(exp_name=args.model_name)
    for i, trajs in enumerate(collector):
        torchrl_logger.info(f"Collected batch {i}")
        torchrl_logger.info(f"trajs {trajs}")
        trajs = trajs.reshape(-1)
        trajs = compute_mc_advantage(trajs)
        rb.extend(trajs)
        # logging
        reward = torch.cat(rb[:].get(("next", "reward"), as_list=True)).mean()
        if not reward:
            # no use in training a model without reward
            torchrl_logger.info("no reward - skipping")
            torch.cuda.empty_cache()  # TODO: Test if this is needed
            continue
        logger.log_scalar("reward", reward)
        torchrl_logger.info(f"reward: {reward: 4.4f}")
        for i in range(args.epochs):
            torchrl_logger.info(f"epoch: {i}")
            for batch in tqdm.tqdm(rb):
                batch = batch.to(train_devices[0])
                loss = loss_fn(batch)
                loss_val = loss.mean(reduce=True)
                loss_val.backward()
                optim.step()
                optim.zero_grad(set_to_none=True)

                gc.collect()
                torch.cuda.empty_cache()

        collector.update_policy_weights_()
        gc.collect()
        torch.cuda.empty_cache()
