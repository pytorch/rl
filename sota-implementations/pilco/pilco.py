import hydra
import tensordict
import torch
from omegaconf import DictConfig

from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torchrl._utils import get_available_device
from torchrl.envs import EnvBase
from torchrl.envs.utils import RandomPolicy
from torchrl.record.loggers import generate_exp_name, get_logger, Logger

from utils import (
    BoTorchGPWorldModel,
    ImaginedEnv,
    make_env,
    pendulum_cost,
    RBFController,
)


def pilco_loop(
    cfg: DictConfig, env: EnvBase, logger: Logger | None = None
) -> TensorDictModule:
    obs_dim = env.observation_spec["observation"].shape[-1]
    action_dim = env.action_spec.shape[-1]

    random_policy = RandomPolicy(action_spec=env.action_spec)
    rollout = env.rollout(
        max_steps=cfg.pilco.initial_rollout_length,
        policy=random_policy,
        break_when_all_done=False,
        break_when_any_done=False,
    )

    base_policy = (
        RBFController(
            input_dim=obs_dim,
            output_dim=action_dim,
            n_basis=cfg.pilco.policy_n_basis,
            max_action=env.action_spec.high,
        )
        .to(env.device)
        .double()
    )
    policy_module = TensorDictModule(
        module=base_policy,
        in_keys=[("observation", "mean"), ("observation", "var")],
        out_keys=[
            ("action", "mean"),
            ("action", "var"),
            ("action", "cross_covariance"),
        ],
    )
    optimizer = torch.optim.Adam(policy_module.parameters(), lr=cfg.optim.policy_lr)

    dtype = torch.float64
    initial_observation = TensorDict(
        {
            ("observation", "mean"): torch.zeros(
                obs_dim, device=env.device, dtype=dtype
            ),
            ("observation", "var"): torch.eye(obs_dim, device=env.device, dtype=dtype)
            * 1e-3,
        }
    )

    for epoch in range(cfg.pilco.epochs):
        base_world_model = BoTorchGPWorldModel(
            obs_dim=obs_dim, action_dim=action_dim
        ).to(env.device)
        base_world_model.fit(rollout)
        base_world_model.freeze_and_detach()

        world_model_module = TensorDictModule(
            module=base_world_model,
            in_keys=["action", "observation"],
            out_keys=[("next_observation", "mean"), ("next_observation", "var")],
        )

        imagined_env = ImaginedEnv(
            world_model_module=world_model_module,
            base_env=env,
        )
        reset_td = initial_observation.expand(*imagined_env.batch_size)

        for step in range(cfg.pilco.policy_training_steps):
            logger_step = (epoch * cfg.pilco.policy_training_steps) + step
            optimizer.zero_grad()

            imagined_data = imagined_env.rollout(
                max_steps=cfg.pilco.horizon,
                policy=policy_module,
                tensordict=reset_td,
            )

            obs = imagined_data["observation"]
            cost = pendulum_cost(obs)
            loss = cost.mean()

            loss.backward()
            optimizer.step()

            if logger:
                logger.log_scalar(
                    "train/trajectory_cost", loss.item(), step=logger_step
                )

        def policy_for_env(td: TensorDictBase) -> TensorDictBase:
            obs = td["observation"]
            device, dtype = obs.device, obs.dtype

            is_unbatched = obs.ndim == 1
            if is_unbatched:
                obs = obs.unsqueeze(0)

            batch_shape = obs.shape[:-1]
            D = obs.shape[-1]

            policy_in = TensorDict(
                {
                    "observation": TensorDict(
                        {
                            "mean": obs,
                            "var": torch.zeros(
                                (*batch_shape, D, D), device=device, dtype=dtype
                            ),
                        },
                        batch_size=batch_shape,
                    )
                },
                batch_size=batch_shape,
                device=device,
            )

            policy_out = policy_module(policy_in)
            action_mean = policy_out["action", "mean"]

            if is_unbatched:
                action_mean = action_mean.squeeze(0)

            td["action"] = action_mean
            return td

        test_rollout = env.rollout(
            max_steps=1000, policy=policy_for_env, break_when_any_done=True
        )

        reward = test_rollout["episode_reward"][-1].item()
        steps = test_rollout["step_count"].max().item()

        if logger:
            logger.log_scalar("eval/reward", reward, step=logger_step)
            logger.log_scalar("eval/steps", steps, step=logger_step)

        rollout = tensordict.cat([rollout, test_rollout], dim=0)

        if len(rollout) > cfg.pilco.max_rollout_length:
            rollout = rollout[-cfg.pilco.max_rollout_length :]

    return policy_module


@hydra.main(config_path="", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    device = torch.device(cfg.device) if cfg.device else get_available_device()

    env = make_env(cfg.env.env_name, device, from_pixels=cfg.logger.video)

    if cfg.logger.backend:
        exp_name = generate_exp_name("PILCO", cfg.env.env_name)
        logger = get_logger(
            cfg.logger.backend,
            logger_name="pilco",
            experiment_name=exp_name,
            wandb_kwargs={
                "config": dict(cfg),
                "project": cfg.logger.project_name,
                "group": cfg.logger.group_name,
            },
        )

    pilco_loop(cfg, env, logger=logger)

    if not env.is_closed:
        env.close()


if __name__ == "__main__":
    main()
