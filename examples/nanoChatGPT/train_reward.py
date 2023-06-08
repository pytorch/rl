import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from data import get_reward_dataloaders
from models.reward import init_reward_model
from utils import create_lr_scheduler, load_and_update_config, setup

HERE = Path(__file__).parent


def _accuracy(chosen_end_scores, rejected_end_scores):
    return (
        sum(chosen_end_scores > rejected_end_scores) / len(rejected_end_scores)
    ).item()


# TODO: eliminate redundant repeated definition
# helps estimate an arbitrarily accurate loss over either split using many batches
def create_loss_estimator(config, ctx):
    @torch.no_grad()
    def estimate_loss(model, dataloader):
        model.eval()
        losses = torch.zeros(config["eval_iters"])
        accs = torch.zeros(config["eval_iters"])
        for k in range(config["eval_iters"]):
            chosen_batch, rejected_batch = next(dataloader)
            with ctx:
                model(chosen_batch)
                model(rejected_batch)
            losses[k] = model.compute_reward_loss(chosen_batch, rejected_batch).item()
            accs[k] = _accuracy(chosen_batch.end_scores, rejected_batch.end_scores)
        model.train()
        return losses.mean(), accs.mean()

    return estimate_loss


def main():
    config = load_and_update_config("config/train_reward.yaml")
    ctx = setup(config)

    # ######## INIT MODELS ########
    model = init_reward_model(config)

    # Freeze the first 70% of the hidden layers of the reward model backbone
    layers = model.transformer.h
    num_layers = len(layers)
    num_unfrozen = int(0.3 * num_layers)
    for layer in layers[:-num_unfrozen]:
        layer.requires_grad_(False)

    # ######## INIT TRAINING FUNCTIONS ########

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    lr_scheduler = create_lr_scheduler(config)

    estimate_loss = create_loss_estimator(config, ctx)

    train_loader, val_loader = get_reward_dataloaders(config)

    # these will already have been set if resuming from previous checkpoint
    iter_num = config.setdefault("iter_num", 0)
    best_val_loss = config.setdefault("best_val_loss", 1e9)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # ######## TRAINING LOOP ########
    t0 = time.time()
    for it in range(iter_num, config["max_iters"]):
        # get and update the learning rate
        lr = lr_scheduler(it)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        chosen_batch, rejected_batch = next(train_loader)

        # TODO: check why is different from std model (missing micro gradients)

        model(chosen_batch)
        model(rejected_batch)
        optimizer.zero_grad(set_to_none=True)
        loss = model.compute_reward_loss(chosen_batch, rejected_batch)
        loss.backward()
        optimizer.step()

        # ########### EVALUATE MODEL AND CHECKPOINT ###############
        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if it % config["eval_interval"] == 0:
            # evaluate the loss on train/val sets and write checkpoints
            val_loss, val_acc = estimate_loss(model, val_loader)
            train_loss, train_acc = estimate_loss(model, train_loader)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            print(
                f"Evaluation: iter {it}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, "
                f"train acc {train_acc:.4f}, val acc {val_acc:.4f}"
            )
            if val_loss < best_val_loss or config["always_save_checkpoint"]:
                best_val_loss = val_loss
                if it > 0:
                    checkpoint = {
                        "optimizer": optimizer.state_dict(),
                        "iter_num": it,
                        "best_val_loss": best_val_loss,
                        "config": config,
                    }
                    print(f"saving checkpoint to {config['out_dir_reward']}")
                    model.module.save_pretrained(config["out_dir_reward"])
                    torch.save(
                        checkpoint,
                        os.path.join(config["out_dir_reward"], "ckpt_status.pt"),
                    )
            train_losses.append(train_loss)
            val_losses.append(val_loss)

        elif it % config["log_interval"] == 0:
            # loss as float. note: this is a CPU-GPU sync point
            lossf = loss.item()
            train_losses.append(lossf)
            train_accs.append(
                _accuracy(chosen_batch.end_scores, rejected_batch.end_scores)
            )
            print(
                f"iter {it}: train loss {lossf:.4f}, accuracy {train_accs[-1]:.4f} time {dt*1000:.2f}ms"
            )

    f, ax = plt.subplots(figsize=(8, 6))
    plt.title("Reward Model: Loss")
    ax.plot(
        np.arange(0, config["max_iters"], config["log_interval"]),
        train_losses,
        label="train loss",
    )
    ax.plot(
        np.arange(0, config["max_iters"], config["eval_interval"]),
        val_losses,
        label="valid loss",
    )
    ax.set_yscale("log")
    ax.legend()

    f.savefig("figures/reward_curve_loss.png", dpi=150)

    f, ax = plt.subplots(figsize=(8, 6))
    plt.title("Reward Model: Accuracy")
    ax.plot(
        np.arange(0, config["max_iters"], config["log_interval"]),
        train_accs,
        label="train accs",
    )
    ax.plot(
        np.arange(0, config["max_iters"], config["eval_interval"]),
        val_accs,
        label="valid accs",
    )
    ax.set_yscale("log")
    ax.legend()

    f.savefig("figures/reward_curve_accuracy.png", dpi=150)


if __name__ == "__main__":
    main()
