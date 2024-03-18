# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time

import hydra
import torch

from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torch.distributions import Categorical, OneHotCategorical
from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.modules.models.multiagent import MultiAgentMLP
from torchrl.objectives import DiscreteSACLoss, SACLoss, SoftUpdate, ValueEstimators
from utils.logging import init_logging, log_evaluation, log_training
from utils.utils import DoneTransform


def rendering_callback(env, td):
    env.frames.append(env.render(mode="rgb_array", agent_index_focus=None))


@hydra.main(version_base="1.1", config_path="", config_name="sac")
def train(cfg: "DictConfig"):  # noqa: F821
    # Device
    cfg.train.device = "cpu" if not torch.cuda.device_count() else "cuda:0"
    cfg.env.device = cfg.train.device

    # Seeding
    torch.manual_seed(cfg.seed)

    # Sampling
    cfg.env.vmas_envs = cfg.collector.frames_per_batch // cfg.env.max_steps
    cfg.collector.total_frames = cfg.collector.frames_per_batch * cfg.collector.n_iters
    cfg.buffer.memory_size = cfg.collector.frames_per_batch

    # Create env and env_test
    env = VmasEnv(
        scenario=cfg.env.scenario_name,
        num_envs=cfg.env.vmas_envs,
        continuous_actions=cfg.env.continuous_actions,
        max_steps=cfg.env.max_steps,
        device=cfg.env.device,
        seed=cfg.seed,
        categorical_actions=cfg.env.categorical_actions,
        # Scenario kwargs
        **cfg.env.scenario,
    )
    env = TransformedEnv(
        env,
        RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
    )

    env_test = VmasEnv(
        scenario=cfg.env.scenario_name,
        num_envs=cfg.eval.evaluation_episodes,
        continuous_actions=cfg.env.continuous_actions,
        max_steps=cfg.env.max_steps,
        device=cfg.env.device,
        seed=cfg.seed,
        # Scenario kwargs
        **cfg.env.scenario,
    )

    # Policy
    if cfg.env.continuous_actions:
        actor_net = nn.Sequential(
            MultiAgentMLP(
                n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
                n_agent_outputs=2 * env.action_spec.shape[-1],
                n_agents=env.n_agents,
                centralised=False,
                share_params=cfg.model.shared_parameters,
                device=cfg.train.device,
                depth=2,
                num_cells=256,
                activation_class=nn.Tanh,
            ),
            NormalParamExtractor(),
        )
        policy_module = TensorDictModule(
            actor_net,
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "loc"), ("agents", "scale")],
        )

        policy = ProbabilisticActor(
            module=policy_module,
            spec=env.unbatched_action_spec,
            in_keys=[("agents", "loc"), ("agents", "scale")],
            out_keys=[env.action_key],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "min": env.unbatched_action_spec[("agents", "action")].space.low,
                "max": env.unbatched_action_spec[("agents", "action")].space.high,
            },
            return_log_prob=True,
        )

        # Critic
        module = MultiAgentMLP(
            n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1]
            + env.action_spec.shape[-1],  # Q critic takes action and value
            n_agent_outputs=1,
            n_agents=env.n_agents,
            centralised=cfg.model.centralised_critic,
            share_params=cfg.model.shared_parameters,
            device=cfg.train.device,
            depth=2,
            num_cells=256,
            activation_class=nn.Tanh,
        )
        value_module = ValueOperator(
            module=module,
            in_keys=[("agents", "observation"), env.action_key],
            out_keys=[("agents", "state_action_value")],
        )
    else:
        actor_net = nn.Sequential(
            MultiAgentMLP(
                n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
                n_agent_outputs=env.action_spec.space.n,
                n_agents=env.n_agents,
                centralised=False,
                share_params=cfg.model.shared_parameters,
                device=cfg.train.device,
                depth=2,
                num_cells=256,
                activation_class=nn.Tanh,
            ),
        )
        policy_module = TensorDictModule(
            actor_net,
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "logits")],
        )
        policy = ProbabilisticActor(
            module=policy_module,
            spec=env.unbatched_action_spec,
            in_keys=[("agents", "logits")],
            out_keys=[env.action_key],
            distribution_class=OneHotCategorical
            if not cfg.env.categorical_actions
            else Categorical,
            return_log_prob=True,
        )

        # Critic
        module = MultiAgentMLP(
            n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
            n_agent_outputs=env.action_spec.space.n,
            n_agents=env.n_agents,
            centralised=cfg.model.centralised_critic,
            share_params=cfg.model.shared_parameters,
            device=cfg.train.device,
            depth=2,
            num_cells=256,
            activation_class=nn.Tanh,
        )
        value_module = ValueOperator(
            module=module,
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "action_value")],
        )

    collector = SyncDataCollector(
        env,
        policy,
        device=cfg.env.device,
        storing_device=cfg.train.device,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        postproc=DoneTransform(reward_key=env.reward_key, done_keys=env.done_keys),
    )

    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(cfg.buffer.memory_size, device=cfg.train.device),
        sampler=SamplerWithoutReplacement(),
        batch_size=cfg.train.minibatch_size,
    )

    if cfg.env.continuous_actions:
        loss_module = SACLoss(
            actor_network=policy,
            qvalue_network=value_module,
            delay_qvalue=True,
            action_spec=env.unbatched_action_spec,
        )
        loss_module.set_keys(
            state_action_value=("agents", "state_action_value"),
            action=env.action_key,
            reward=env.reward_key,
            done=("agents", "done"),
            terminated=("agents", "terminated"),
        )
    else:
        loss_module = DiscreteSACLoss(
            actor_network=policy,
            qvalue_network=value_module,
            delay_qvalue=True,
            num_actions=env.action_spec.space.n,
            action_space=env.unbatched_action_spec,
        )
        loss_module.set_keys(
            action_value=("agents", "action_value"),
            action=env.action_key,
            reward=env.reward_key,
            done=("agents", "done"),
            terminated=("agents", "terminated"),
        )

    loss_module.make_value_estimator(ValueEstimators.TD0, gamma=cfg.loss.gamma)
    target_net_updater = SoftUpdate(loss_module, eps=1 - cfg.loss.tau)

    optim = torch.optim.Adam(loss_module.parameters(), cfg.train.lr)

    # Logging
    if cfg.logger.backend:
        model_name = (
            ("Het" if not cfg.model.shared_parameters else "")
            + ("MA" if cfg.model.centralised_critic else "I")
            + "SAC"
        )
        logger = init_logging(cfg, model_name)

    total_time = 0
    total_frames = 0
    sampling_start = time.time()
    for i, tensordict_data in enumerate(collector):
        torchrl_logger.info(f"\nIteration {i}")

        sampling_time = time.time() - sampling_start

        current_frames = tensordict_data.numel()
        total_frames += current_frames
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view)

        training_tds = []
        training_start = time.time()
        for _ in range(cfg.train.num_epochs):
            for _ in range(cfg.collector.frames_per_batch // cfg.train.minibatch_size):
                subdata = replay_buffer.sample()
                loss_vals = loss_module(subdata)
                training_tds.append(loss_vals.detach())

                loss_value = (
                    loss_vals["loss_actor"]
                    + loss_vals["loss_alpha"]
                    + loss_vals["loss_qvalue"]
                )

                loss_value.backward()

                total_norm = torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), cfg.train.max_grad_norm
                )
                training_tds[-1].set("grad_norm", total_norm.mean())

                optim.step()
                optim.zero_grad()
                target_net_updater.step()

        collector.update_policy_weights_()

        training_time = time.time() - training_start

        iteration_time = sampling_time + training_time
        total_time += iteration_time
        training_tds = torch.stack(training_tds)

        # More logs
        if cfg.logger.backend:
            log_training(
                logger,
                training_tds,
                tensordict_data,
                sampling_time,
                training_time,
                total_time,
                i,
                current_frames,
                total_frames,
                step=i,
            )

        if (
            cfg.eval.evaluation_episodes > 0
            and i % cfg.eval.evaluation_interval == 0
            and cfg.logger.backend
        ):
            evaluation_start = time.time()
            with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
                env_test.frames = []
                rollouts = env_test.rollout(
                    max_steps=cfg.env.max_steps,
                    policy=policy,
                    callback=rendering_callback,
                    auto_cast_to_device=True,
                    break_when_any_done=False,
                    # We are running vectorized evaluation we do not want it to stop when just one env is done
                )

                evaluation_time = time.time() - evaluation_start

                log_evaluation(logger, rollouts, env_test, evaluation_time, step=i)

        if cfg.logger.backend == "wandb":
            logger.experiment.log({}, commit=True)
        sampling_start = time.time()


if __name__ == "__main__":
    train()
