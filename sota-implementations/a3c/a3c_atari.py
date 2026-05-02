from __future__ import annotations

from copy import deepcopy

import hydra
import torch

import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim
import tqdm
from tensordict import from_module

from torchrl.collectors import SyncDataCollector
from torchrl.objectives import A2CLoss
from torchrl.objectives.value.advantages import GAE

from torchrl.record.loggers import generate_exp_name, get_logger
from utils_atari import make_parallel_env, make_ppo_models, SharedAdam


torch.set_float32_matmul_precision("high")


class A3CWorker(mp.Process):
    def __init__(
        self, name, cfg, global_actor, global_critic, optimizer, use_logger=False
    ):
        super().__init__()
        self.name = name
        self.cfg = cfg

        self.optimizer = optimizer

        self.device = cfg.loss.device or torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        self.frame_skip = 4
        self.total_frames = cfg.collector.total_frames // self.frame_skip
        self.frames_per_batch = cfg.collector.frames_per_batch // self.frame_skip
        self.mini_batch_size = cfg.loss.mini_batch_size // self.frame_skip
        self.test_interval = cfg.logger.test_interval // self.frame_skip

        self.global_actor = global_actor
        self.global_critic = global_critic
        self.local_actor = self.copy_model(global_actor)
        self.local_critic = self.copy_model(global_critic)

        logger = None
        if use_logger and cfg.logger.backend:
            exp_name = generate_exp_name(
                "A3C", f"{cfg.logger.exp_name}_{cfg.env.env_name}"
            )
            logger = get_logger(
                cfg.logger.backend,
                logger_name="a3c",
                experiment_name=exp_name,
                wandb_kwargs={
                    "config": dict(cfg),
                    "project": cfg.logger.project_name,
                    "group": cfg.logger.group_name,
                },
            )

        self.logger = logger

        self.adv_module = GAE(
            gamma=cfg.loss.gamma,
            lmbda=cfg.loss.gae_lambda,
            value_network=self.local_critic,
            average_gae=True,
            vectorized=not cfg.compile.compile,
            device=self.device,
        )
        self.loss_module = A2CLoss(
            actor_network=self.local_actor,
            critic_network=self.local_critic,
            loss_critic_type=cfg.loss.loss_critic_type,
            entropy_coef=cfg.loss.entropy_coef,
            critic_coef=cfg.loss.critic_coef,
        )

        self.adv_module.set_keys(done="end-of-life", terminated="end-of-life")
        self.loss_module.set_keys(done="end-of-life", terminated="end-of-life")

    def copy_model(self, model):
        td_params = from_module(model)
        td_new_params = td_params.data.clone()
        td_new_params = td_new_params.apply(
            lambda p0, p1: torch.nn.Parameter(p0)
            if isinstance(p1, torch.nn.Parameter)
            else p0,
            td_params,
        )
        with td_params.data.to("meta").to_module(model):
            # we don't copy any param here
            new_model = deepcopy(model)
        td_new_params.to_module(new_model)
        return new_model

    def update(self, batch, max_grad_norm=None):
        if max_grad_norm is None:
            max_grad_norm = self.cfg.optim.max_grad_norm

        loss = self.loss_module(batch)
        loss_sum = loss["loss_critic"] + loss["loss_objective"] + loss["loss_entropy"]
        loss_sum.backward()

        for local_param, global_param in zip(
            self.local_actor.parameters(), self.global_actor.parameters()
        ):
            global_param._grad = local_param.grad

        for local_param, global_param in zip(
            self.local_critic.parameters(), self.global_critic.parameters()
        ):
            global_param._grad = local_param.grad

        gn = torch.nn.utils.clip_grad_norm_(
            self.loss_module.parameters(), max_norm=max_grad_norm
        )

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        return (
            loss.select("loss_critic", "loss_entropy", "loss_objective")
            .detach()
            .set("grad_norm", gn)
        )

    def run(self):
        cfg = self.cfg

        collector = SyncDataCollector(
            create_env_fn=make_parallel_env(
                cfg.env.env_name,
                num_envs=cfg.env.num_envs,
                device=self.device,
                gym_backend=cfg.env.backend,
            ),
            policy=self.local_actor,
            frames_per_batch=self.frames_per_batch,
            total_frames=self.total_frames,
            device=self.device,
            storing_device=self.device,
            policy_device=self.device,
            compile_policy=False,
            cudagraph_policy=False,
        )

        collected_frames = 0
        num_network_updates = 0
        pbar = tqdm.tqdm(total=self.total_frames)
        num_mini_batches = self.frames_per_batch // self.mini_batch_size
        total_network_updates = (
            self.total_frames // self.frames_per_batch
        ) * num_mini_batches
        lr = cfg.optim.lr

        c_iter = iter(collector)
        total_iter = len(collector)

        for _ in range(total_iter):
            data = next(c_iter)

            metrics_to_log = {}
            frames_in_batch = data.numel()
            collected_frames += self.frames_per_batch * self.frame_skip
            pbar.update(frames_in_batch)

            episode_rewards = data["next", "episode_reward"][data["next", "terminated"]]
            if len(episode_rewards) > 0:
                episode_length = data["next", "step_count"][data["next", "terminated"]]
                metrics_to_log["train/reward"] = episode_rewards.mean().item()
                metrics_to_log[
                    "train/episode_length"
                ] = episode_length.sum().item() / len(episode_length)

            with torch.no_grad():
                data = self.adv_module(data)
            data_reshape = data.reshape(-1)
            losses = []

            mini_batches = data_reshape.split(self.mini_batch_size)
            for batch in mini_batches:
                alpha = 1.0
                if cfg.optim.anneal_lr:
                    alpha = 1 - (num_network_updates / total_network_updates)
                    for group in self.optimizer.param_groups:
                        group["lr"] = lr * alpha

                num_network_updates += 1
                loss = self.update(batch).clone()
                losses.append(loss)

            losses = torch.stack(losses).float().mean()

            for key, value in losses.items():
                metrics_to_log[f"train/{key}"] = value.item()

            metrics_to_log["train/lr"] = lr * alpha

            # Logging only on the first worker in the dashboard.
            # Alternatively, you can use a distributed logger, or aggregate metrics from all workers.
            if self.logger:
                for key, value in metrics_to_log.items():
                    self.logger.log_scalar(key, value, collected_frames)
        collector.shutdown()


@hydra.main(config_path="", config_name="config_atari", version_base="1.1")
def main(cfg: DictConfig):  # noqa: F821

    global_actor, global_critic, global_critic_head = make_ppo_models(
        cfg.env.env_name, device=cfg.loss.device, gym_backend=cfg.env.backend
    )
    global_model = nn.ModuleList([global_actor, global_critic_head])
    global_model.share_memory()
    optimizer = SharedAdam(global_model.parameters(), lr=cfg.optim.lr)

    num_workers = cfg.multiprocessing.num_workers

    workers = [
        A3CWorker(
            f"worker_{i}",
            cfg,
            global_actor,
            global_critic,
            optimizer,
            use_logger=i == 0,
        )
        for i in range(num_workers)
    ]
    [w.start() for w in workers]
    [w.join() for w in workers]


if __name__ == "__main__":
    main()
