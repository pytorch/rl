import hydra
import tensordict
import torch
from omegaconf import DictConfig

from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl._utils import get_available_device
from torchrl.envs import EnvBase, TransformedEnv
from torchrl.envs.model_based import ImaginedEnv
from torchrl.envs.transforms import MeanActionSelector
from torchrl.envs.utils import RandomPolicy
from torchrl.modules.models import GPWorldModel, RBFController
from torchrl.objectives import ExponentialQuadraticCost
from torchrl.record.loggers import generate_exp_name, get_logger, Logger

from utils import make_env


def evaluate_policy(
    env: TransformedEnv,
    policy: TensorDictModule,
    eval_horizon: int,
    step: int,
    logger: Logger | None = None,
) -> TensorDict:
    test_rollout = env.rollout(
        max_steps=eval_horizon,
        policy=policy,
        break_when_any_done=True,
    )

    reward = test_rollout["episode_reward"][-1].tolist()
    steps = test_rollout["step_count"].max().tolist()
    angles = test_rollout["observation", "mean"][..., 1].abs()
    max_abs_angle = angles.max().tolist()
    mean_abs_angle = angles.mean().tolist()

    if logger:
        logger.log_scalar("eval/reward", reward, step=step)
        logger.log_scalar("eval/steps", steps, step=step)
        logger.log_scalar("eval/max_abs_angle", max_abs_angle, step=step)
        logger.log_scalar("eval/mean_abs_angle", mean_abs_angle, step=step)

    return test_rollout


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

    eval_env = TransformedEnv(env, MeanActionSelector())
    evaluate_policy(
        eval_env,
        policy_module,
        eval_horizon=cfg.pilco.eval_horizon,
        step=0,
        logger=logger,
    )

    cost_module = ExponentialQuadraticCost(reduction="none").to(env.device)
    for epoch in range(cfg.pilco.epochs):
        base_world_model = GPWorldModel(obs_dim=obs_dim, action_dim=action_dim).to(
            env.device
        )
        base_world_model.fit(rollout.to(dtype))
        base_world_model.eval()

        imagined_env = ImaginedEnv(
            world_model_module=base_world_model,
            base_env=env,
        )
        reset_td = initial_observation.expand(*imagined_env.batch_size)

        for step in range(cfg.pilco.policy_training_steps):
            logger_step = (epoch * cfg.pilco.policy_training_steps) + step + 1
            optimizer.zero_grad()

            imagined_data = imagined_env.rollout(
                max_steps=cfg.pilco.horizon,
                policy=policy_module,
                tensordict=reset_td,
            )

            loss_td = cost_module(imagined_data)
            loss = loss_td.get("loss_cost").sum(dim=-1).mean()

            loss.backward()
            optimizer.step()

            if logger:
                logger.log_scalar(
                    "train/trajectory_cost", loss.item(), step=logger_step
                )

            if logger_step % cfg.pilco.eval_interval == 0:
                evaluate_policy(
                    eval_env,
                    policy_module,
                    eval_horizon=cfg.pilco.eval_horizon,
                    step=logger_step,
                    logger=logger,
                )

        should_log_final_eval = logger_step % cfg.pilco.eval_interval != 0
        test_rollout = evaluate_policy(
            eval_env,
            policy_module,
            eval_horizon=cfg.pilco.eval_horizon,
            step=logger_step,
            logger=logger if should_log_final_eval else None,
        )

        test_rollout.set("observation", test_rollout.get(("observation", "mean")))
        test_rollout.set("action", test_rollout.get(("action", "mean")))
        test_rollout.set(
            ("next", "observation"), test_rollout.get(("next", "observation", "mean"))
        )

        test_rollout = test_rollout.select(
            *rollout.keys(include_nested=True, leaves_only=True)
        )
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
