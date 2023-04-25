import os
from pathlib import Path
from typing import Optional

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from model import RLHF

from shared import create_infinite_dataloader, create_lr_scheduler, init_model, setup
from tensordict.nn import TensorDictModule
from tensordict.prototype import tensorclass
from tqdm import tqdm
from utils import load_and_update_config

HERE = Path(__file__).parent


class Collate(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = torch.device(device)

    def __call__(self, batch):
        if self.device.type == "cuda":
            batch = batch.pin_memory()
        return batch.to(self.device)


@tensorclass
class PairwiseDataset:
    chosen: torch.Tensor
    rejected: torch.Tensor
    reward: Optional[torch.Tensor] = None

    @classmethod
    def from_dataset(cls, dataset, max_length):
        # TODO: check dtypes
        data = cls(
            chosen=torch.zeros(len(dataset), max_length, dtype=torch.int32),
            rejected=torch.zeros(len(dataset), max_length, dtype=torch.int32),
            batch_size=[len(dataset)],
        )
        enc = tiktoken.get_encoding("gpt2")
        i = 0

        for sample in tqdm(dataset, total=len(dataset)):
            prompt = sample["prompt"]
            chosen = sample["chosen"]
            rejected = sample["rejected"]

            if len(chosen.split()) < 5 or len(rejected.split()) < 5:
                continue

            chosen = "\n".join([prompt, chosen])
            rejected = "\n".join([prompt, rejected])

            chosen = enc.encode(
                "<|startoftext|>" + chosen + "<|endoftext|>", allowed_special="all"
            )[-max_length:]
            rejected = enc.encode(
                "<|startoftext|>" + rejected + "<|endoftext|>", allowed_special="all"
            )[-max_length:]

            if chosen == rejected:
                continue

            data[i] = cls(
                chosen=F.pad(
                    torch.Tensor(chosen), (max_length - len(chosen), 0), value=0
                ),
                rejected=F.pad(
                    torch.Tensor(rejected), (max_length - len(rejected), 0), value=0
                ),
                batch_size=[],
            )
            i += 1

        # index because we will have skipped some datapoints
        return data[:i]


def create_datasets(config):
    # Make pairwise datasets for training
    print("Creating pairwise datasets")
    data_path = "CarperAI/openai_summarize_comparisons"
    train_data = PairwiseDataset.from_dataset(
        load_dataset(data_path, split="train"), max_length=config["block_size"]
    )
    val_data = PairwiseDataset.from_dataset(
        load_dataset(data_path, split="test"), max_length=config["block_size"]
    )

    return train_data, val_data


def get_dataloaders(config):
    train_data, val_data = create_datasets(config)
    train_data.memmap_()
    val_data.memmap_()

    train_loader = create_infinite_dataloader(
        train_data, config, Collate(config["device"])
    )
    val_loader = create_infinite_dataloader(val_data, config, Collate(config["device"]))

    return train_loader, val_loader


# helps estimate an arbitrarily accurate loss over either split using many batches
def create_loss_estimator(config):
    @torch.no_grad()
    def estimate_loss(model, dataloader):
        losses = torch.zeros(config["eval_iters"])
        for k in range(config["eval_iters"]):
            batch = next(dataloader)
            with ctx:
                reward_chosen = model(batch.chosen)
                reward_rejected = model(batch.rejected)
                loss = -torch.log(torch.sigmoid(reward_chosen - reward_rejected)).mean()
            losses[k] = loss.item()
        return losses.mean()

    return estimate_loss


def train_reward_model(config):
    # TODO: clean up...train should do just the training.
    # model creation, data loading etc. should be performed outside
    # plus align all script to have same structure and order of calls

    # GET DATA
    train_loader, val_loader = get_dataloaders(config)

    # FIXME: Don't like this. include it into model
    model, model_kwargs = init_model(config)
    model = RLHF(model, mode="reward", discrete_reward=False)

    print("Config of model: ", model.config)

    if not os.path.exists(config["out_dir_reward"]):
        print(f"Create {config['out_dir_reward']}")
        os.mkdir(config["out_dir_reward"])

    if config["init_multihead_from"] == "scratch":
        print("initializing multihead from scratch")
    else:
        if config["init_multihead_from"] == "resume":
            print(f"Resuming training from {config['out_dir_reward']}")
            # resume training from a checkpoint.
            ckpt_path = os.path.join(config["out_dir_reward"], "ckpt.pt")
            checkpoint = torch.load(ckpt_path, map_location=config["device"])
            state_dict = checkpoint["model"]
            # fix the keys of the state dictionary :(
            # honestly no idea how checkpoints sometimes get this prefix, have to debug more
            unwanted_prefixes = ["_orig_mod.", "model."]
            for unwanted_prefix in unwanted_prefixes:
                for k in list(state_dict):
                    if k.startswith(unwanted_prefix):
                        state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
            model.load_state_dict(state_dict)

    model.to(config["device"])

    # compile the model
    if config["compile"]:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)  # requires PyTorch 2.0

    model = TensorDictModule(model, in_keys=["input"], out_keys=["reward"])
    # FIXME: which one?
    # optimizer = torch.optim.AdamW(model.model.reward_head.parameters(), lr=1e-3)
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-4)

    # training loop
    local_iter_num = 0  # number of iterations in the lifetime of this process
    config["running_mfu"] = -1.0
    raw_model = model.module
    loss = None
    # these will already have been set if resuming from previous checkpoint
    iter_num = config.setdefault("iter_num", 0)
    best_val_loss = config.setdefault("best_val_loss", 1e9)

    estimate_loss = create_loss_estimator(config)

    if config["decay_lr"]:
        lr_scheduler = create_lr_scheduler(config)
    else:

        def lr_scheduler(_):
            return config["learning_rate"]

    while True:
        # determine and set the learning rate for this iteration
        lr = lr_scheduler(iter_num)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # # every once in a while evaluate the loss on train and val sets
        if iter_num % config["eval_interval"] == 0:
            model.eval()
            losses = {
                "train": estimate_loss(model, train_loader),
                "val": estimate_loss(model, val_loader),
            }
            model.train()
            print(
                f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            if losses["val"] < best_val_loss or config["always_save_checkpoint"]:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_kwargs": model_kwargs,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": config,
                    }
                    print(f"saving checkpoint to {config['out_dir_reward']}")
                    torch.save(
                        checkpoint, os.path.join(config["out_dir_reward"], "ckpt.pt")
                    )
        if iter_num == 0 and config["eval_only"]:
            break

        batch = next(train_loader)

        # TODO: check why is different from std model (missing micro gradients)

        # TODO: combine evaluate_loss function with this. it's almost the same thing
        # evaluate the loss
        reward_chosen = model(batch.chosen)
        reward_rejected = model(batch.rejected)
        loss = -torch.log(torch.sigmoid(reward_chosen - reward_rejected)).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > config["max_iters"]:
            break


if __name__ == "__main__":
    config = load_and_update_config("config/train_reward.yaml")

    ctx = setup(config)
    train_reward_model(config)
