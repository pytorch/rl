import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import tiktoken
import torch
import torch.nn as nn
from datasets import load_dataset
from model import RLHF
from shared import (
    create_infinite_dataloader,
    create_lr_scheduler,
    init_model,
    load_checkpoint,
    setup,
)
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from tensordict.prototype import tensorclass
from torch import nn
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset
from tqdm import tqdm
from utils import init_ddp, load_and_update_config


HERE = Path(__file__).parent


class Collate(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = torch.device(device)

    def __call__(self, batch):
        batch = torch.stack(batch, dim=0).contiguous()
        batch.batch_size = []
        if self.device.type == "cuda":
            batch = batch.pin_memory()
        return batch.to(self.device)


@tensorclass
class Data:
    prompt: torch.Tensor
    target: torch.Tensor
    logits: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None


class PairedDataset(Dataset):
    def __init__(self, path, block_size):
        self._memmap = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def __getitem__(self, idx):
        return Data(
            prompt=torch.from_numpy(
                self._memmap[idx : idx + self.block_size].astype(np.int64)
            ),
            target=torch.from_numpy(
                self._memmap[idx + 1 : idx + self.block_size + 1].astype(np.int64)
            ),
            batch_size=[self.block_size],
        )

    def __len__(self):
        # how many sequences of length block_size + 1 can we extract from the data?
        # the valid starting points for such a sequence are those tokens that aren't in
        # the final block_size positions. so it's just the length of the overall
        # sequence minus the block_size
        return len(self._memmap) - self.block_size


def create_datasets(config):
    data_dir = HERE / "nanoGPT" / "data" / config["dataset"]
    train_data = PairedDataset(data_dir / "train.bin", block_size=config["block_size"])
    val_data = PairedDataset(data_dir / "val.bin", block_size=config["block_size"])

    return train_data, val_data


def get_dataloaders(config):
    train_data, val_data = create_datasets(config)

    train_loader = create_infinite_dataloader(
        train_data, config, Collate(config["device"])
    )
    val_loader = create_infinite_dataloader(val_data, config, Collate(config["device"]))

    return train_loader, val_loader


def train(config):
    # TODO: clean up...train should do just the training.
    # model creation, data loading etc. should be performed outside
    # plus align all script to have same structure and order of calls

    # model init
    # FIXME: Don't like this. include it into model
    model, model_kwargs = init_model(config)
    rl_model = RLHF(model, "RL", discrete_reward=config["discrete_reward"])

    # The current approach is to use a separate reward model because otherwise optimisation of the reward model changes upstream parameters impacting performance of the multihead
    # I therefore load the language model from 'out_dir' and the reward model from 'out_dir_multihead'

    # TODO: move into model
    if config["init_multihead_from"] == "scratch":
        print("initializing multihead from scratch")
    else:
        if config["init_multihead_from"] == "resume":
            print(f"Resuming training from {config['out_dir']}")
            # resume training from a checkpoint.
            ckpt_path = os.path.join(config["out_dir"], "ckpt.pt")
            checkpoint = torch.load(ckpt_path, map_location=config["device"])
            state_dict = checkpoint["model"]
            # fix the keys of the state dictionary :(
            # honestly no idea how checkpoints sometimes get this prefix, have to debug more
            unwanted_prefix = "_orig_mod."
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
            rl_model.load_state_dict(state_dict)

    # TODO: this is different from pg. both should be refactored
    if config["separate_reward_model"]:
        print("Reward model instantiated as copy")
        import copy

        reward_model = copy.deepcopy(model)

        print(f"Resuming reward model from {config['out_dir_multihead']}")

        reward_model = RLHF(
            reward_model, "RL", discrete_reward=config["discrete_reward"]
        )
        # resume training from a checkpoint.
        ckpt_path = os.path.join(config["out_dir_multihead"], "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=config["device"])
        state_dict = checkpoint["model"]
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefixes = ["_orig_mod.", "module."]
        for unwanted_prefix in unwanted_prefixes:
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        reward_model.load_state_dict(state_dict)
    else:
        reward_model = rl_model
    rl_model.to(config["device"])
    reward_model.to(config["device"])

    rl_model = TensorDictModule(rl_model, ["prompt", "target"], ["logits", "loss"])

    gumbel_optimizer = torch.optim.AdamW(rl_model.parameters(), lr=1e-3)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(config["dtype"] == "float16"))
    enc = tiktoken.get_encoding("gpt2")

    last_time = time.time()
    rews_all = []
    max_iters = 100000

    # FIXME: why no compile in this train??

    train_loader, val_loader = get_dataloaders(config)

    # training loop
    next_batch = next(train_loader)  # fetch the very first batch

    # TODO: why loading batches if then not used?
    next_batch.prompt = torch.zeros(
        (next_batch.prompt.shape[0], 1), dtype=torch.long
    ).to(
        config["device"]
    )  # for now there is no prompt

    t0 = time.time()
    for iter in range(max_iters):
        for micro_step in range(config["gradient_accumulation_steps"]):
            batch = next_batch
            if config["is_ddp"]:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                rl_model.require_backward_grad_sync = (
                    micro_step == config["gradient_accumulation_steps"] - 1
                )
            with ctx:
                states, rewards = rl_model.module.generate_gumbel(
                    batch.prompt,
                    config["episode_length"],
                    config["device"],
                    config["block_size"],
                    reward_model=reward_model,
                )
                mean_reward = rewards.mean()
                loss = -mean_reward
                # # immediately async prefetch next batch while model is doing the forward pass on the GPU
                next_batch = next(train_loader)  # fetch the very first batch
                # backward pass, with gradient scaling if training in fp16
                scaler.scale(loss).backward()

        # clip the gradient
        if config["grad_clip"] != 0.0:
            scaler.unscale_(gumbel_optimizer)
            torch.nn.utils.clip_grad_norm_(rl_model.parameters(), config["grad_clip"])
        # step the optimizer and scaler if training in fp16
        scaler.step(gumbel_optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        gumbel_optimizer.zero_grad(set_to_none=True)

        rews_all.append(mean_reward.detach().cpu().numpy())
        eval_interval = config["eval_interval"]
        if iter % eval_interval == 0:
            t1 = time.time()
            print("############## EVALUATION ################")
            print(f"iter: {iter}, time: {t1-t0}")
            print(f"rets: {np.mean(rews_all[-eval_interval:])}")
            if config["verbose"]:
                print("Prompt:")
                print(batch.prompt)
                current_time = time.time()
                # print(current_time - last_time)
                last_time = current_time
                # TODO: why generate and not generate_gumble like previous?
                text = rl_model.generate(
                    batch.prompt,
                    config["episode_length"],
                    config["device"],
                    config["block_size"],
                    reward_model=reward_model,
                    hard_code_reward=config["hard_code_reward"],
                )[0]
                print("-----------")
                print("Generated:")
                print("TEXT", text)
                for i in range(len(text)):
                    text_i = text[i, :]
                    print(f"TEXT[{i}, :]: {text_i}")
                    # print(reward(text_i))
                    try:
                        print("DECODED:", enc.decode(text_i.tolist()))
                    except:
                        continue
                print("-+-+-+-+-+-+-+-+-+-+")


if __name__ == "__main__":
    config = load_and_update_config("config/train_rl.yaml")
    config.update(init_ddp(config["backend"], config["device"]))

    ctx = setup(config)
    train(config)

    if config["is_ddp"]:
        destroy_process_group()
