import os
import time
from pathlib import Path

import torch

from data import get_reward_dataloaders
from models.reward import init_reward_model
from shared import create_lr_scheduler, setup
from utils import load_and_update_config

HERE = Path(__file__).parent


# helps estimate an arbitrarily accurate loss over either split using many batches
def create_loss_estimator(config, ctx):
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


def main():
    config = load_and_update_config("config/train_reward.yaml")
    ctx = setup(config)

    # ######## INIT MODELS ########
    model, model_kwargs = init_reward_model(config)

    # ######## INIT TRAINING FUNCTIONS ########

    optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-4)
    lr_scheduler = create_lr_scheduler(config)

    estimate_loss = create_loss_estimator(config, ctx)

    train_loader, val_loader = get_reward_dataloaders(config)

    # these will already have been set if resuming from previous checkpoint
    iter_num = config.setdefault("iter_num", 0)
    best_val_loss = config.setdefault("best_val_loss", 1e9)

    # ######## TRAINING LOOP ########
    t0 = time.time()
    for it in range(iter_num, config["max_iters"]):
        # get and update the learning rate
        lr = lr_scheduler(it)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        batch = next(train_loader)

        # TODO: check why is different from std model (missing micro gradients)

        # evaluate the loss
        reward_chosen = model(batch.chosen)
        reward_rejected = model(batch.rejected)
        loss = -torch.log(torch.sigmoid(reward_chosen - reward_rejected)).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # ########### EVALUATE MODEL AND CHECKPOINT ###############
        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if it % config["eval_interval"] == 0:
            # evaluate the loss on train/val sets and write checkpoints
            val_loss = estimate_loss(model, val_loader)
            train_loss = estimate_loss(model, train_loader)
            print(
                f"Evaluation: iter {it}: train loss {train_loss:.4f}, val loss {val_loss:.4f}"
            )
            if val_loss < best_val_loss or config["always_save_checkpoint"]:
                best_val_loss = val_loss
                if it > 0:
                    checkpoint = {
                        "model": model.module._orig_mod.state_dict()
                        if config["compile"]
                        else model.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_kwargs": model_kwargs,
                        "iter_num": it,
                        "best_val_loss": best_val_loss,
                        "config": config,
                    }
                    print(f"saving checkpoint to {config['out_dir_reward']}")
                    torch.save(
                        checkpoint, os.path.join(config["out_dir_reward"], "ckpt.pt")
                    )
        elif it % config["log_interval"] == 0:
            # loss as float. note: this is a CPU-GPU sync point
            lossf = batch.loss.item()
            print(f"iter {it}: train loss {lossf:.4f}, time {dt*1000:.2f}ms")


if __name__ == "__main__":
    main()
