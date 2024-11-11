# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Any

import omegaconf

import torch

import torch.nn
import torch.optim

from tensordict.nn import AddStateIndependentNormalScale, TensorDictModule
from torch import nn
from torchrl.collectors import DataCollectorBase
from torchrl.data import Composite, LazyMemmapStorage, TensorDictReplayBuffer, SamplerWithoutReplacement
from torchrl.data import ReplayBuffer
from torchrl.data.tensor_specs import CategoricalBox, Composite
from torchrl.envs import (
    ClipTransform,
    DoubleToFloat,
    ExplorationType,
    RewardSum,
    StepCounter,
    TransformedEnv,
    VecNorm, EnvBase,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs import (
    CatFrames,
    DoubleToFloat,
    EndOfLifeTransform,
    EnvCreator,
    ExplorationType,
    GrayScale,
    GymEnv,
    NoopResetEnv,
    ParallelEnv,
    RenameTransform,
    Resize,
    RewardSum,
    SignTransform,
    StepCounter,
    ToTensorImage,
    TransformedEnv,
    VecNorm,
)
from torchrl.modules import (
    ActorValueOperator,
    ConvNet,
    MLP,
    OneHotCategorical,
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
)
from torchrl.objectives import PPOLoss, LossModule, ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.record import VideoRecorder
from torchrl.record.loggers import Logger
from torchrl.trainers import Trainer

@dataclass
class EnvConfig(omegaconf.DictConfig):
    env_name: str
    num_envs: int = 0

@dataclass
class CollectorConfig(omegaconf.DictConfig):
    frames_per_batch: int
    total_frames: int
@dataclass
class LoggerConfig(omegaconf.DictConfig):
    backend: str
    project_name: str
    group_name: Optional[str]
    exp_name: str
    test_interval: int
    num_test_episodes: int
    video: bool
@dataclass
class OptimConfig(omegaconf.DictConfig):
    lr: float
    weight_decay: float
    anneal_lr: bool
@dataclass
class LossConfig(omegaconf.DictConfig):
    gamma: float
    mini_batch_size: int
    ppo_epochs: int
    gae_lambda: float
    clip_epsilon: float
    anneal_clip_epsilon: bool
    critic_coef: float
    entropy_coef: float
    loss_critic_type: str
@dataclass
class PPOConfig(omegaconf.DictConfig):
    env: EnvConfig
    collector: CollectorConfig
    logger: LoggerConfig
    optim: OptimConfig
    loss: LossConfig

class PPOTrainer(Trainer):
    def __init__(
        self,
        cfg: PPOConfig | None,
        *,
        env: Any | EnvConfig | None = None,
        loss: PPOLoss | LossConfig | None = None,
        policy: nn.Module | omegaconf.DictConfig | None = None,
        critic: nn.Module | omegaconf.DictConfig | None = None,
        replay_buffer: ReplayBuffer | omegaconf.DictConfig | None = None,
        collector: DataCollectorBase | CollectorConfig | None = None,
        optimizer: torch.optim.Optimizer | OptimConfig | None = None,
        logger: Logger | LoggerConfig | None = None
    ):
        cfg = self._parse_cfg(cfg)
        self._env = self._parse_env(env, cfg)
        self._loss = self._parse_loss(loss, cfg)


    @classmethod
    def _parse_cfg(cls, cfg: PPOConfig | None):
        if cfg is None:
            cfg = cls.default_config()
        return cfg

    @classmethod
    def _parse_env(cls, env: Any | EnvConfig | None, cfg: PPOConfig | None=None):
        if env is None:
            return cls.make_env(cfg.env.env_name, cfg.env.num_envsgg)
        elif isinstance(env, omegaconf.DictConfig):
            return cls.make_env(env.env_name)
        elif isinstance(env, EnvBase):
            return env
        else:
            return _autowrap_env(env)

    @classmethod
    def _parse_loss(cls, loss, cfg: LossConfig | None=None):
        if loss is None:
            return cls.make_loss(cfg.loss)
        elif isinstance(loss, omegaconf.DictConfig):
            return cls.make_loss(loss)
        elif isinstance(loss, LossModule):
            return loss
        else:
            raise TypeError(f"Unsupported loss type: {type(loss)} in {cls.__name__}._parse_loss.")

    @classmethod
    def default_config(cls) -> omegaconf.DictConfig:
        raise NotImplementedError

    @classmethod
    def make_env(cls, env_name: str) -> EnvBase:
        raise NotImplementedError

class ContinuousControlPPOTrainer(PPOTrainer):
    @classmethod
    def default_config(cls) -> omegaconf.DictConfig:
        from omegaconf import OmegaConf

        return OmegaConf.load("config_mujoco.yaml")

    # ====================================================================
    # Environment utils
    # --------------------------------------------------------------------
    @classmethod
    def make_env(
        cls, env_name="HalfCheetah-v4", device="cpu", from_pixels: bool = False
    ):
        env = GymEnv(
            env_name, device=device, from_pixels=from_pixels, pixels_only=False
        )
        env = TransformedEnv(env)
        env.append_transform(VecNorm(in_keys=["observation"], decay=0.99999, eps=1e-2))
        env.append_transform(ClipTransform(in_keys=["observation"], low=-10, high=10))
        env.append_transform(RewardSum())
        env.append_transform(StepCounter())
        env.append_transform(DoubleToFloat(in_keys=["observation"]))
        return env

    # ====================================================================
    # Model utils
    # --------------------------------------------------------------------
    @classmethod
    def make_ppo_models_state(cls, proof_environment):

        # Define input shape
        input_shape = proof_environment.observation_spec["observation"].shape

        # Define policy output distribution class
        num_outputs = proof_environment.action_spec.shape[-1]
        distribution_class = TanhNormal
        distribution_kwargs = {
            "low": proof_environment.action_spec.space.low,
            "high": proof_environment.action_spec.space.high,
            "tanh_loc": False,
        }

        # Define policy architecture
        policy_mlp = MLP(
            in_features=input_shape[-1],
            activation_class=torch.nn.Tanh,
            out_features=num_outputs,  # predict only loc
            num_cells=[64, 64],
        )

        # Initialize policy weights
        for layer in policy_mlp.modules():
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.orthogonal_(layer.weight, 1.0)
                layer.bias.data.zero_()

        # Add state-independent normal scale
        policy_mlp = torch.nn.Sequential(
            policy_mlp,
            AddStateIndependentNormalScale(
                proof_environment.action_spec.shape[-1], scale_lb=1e-8
            ),
        )

        # Add probabilistic sampling of the actions
        policy_module = ProbabilisticActor(
            TensorDictModule(
                module=policy_mlp,
                in_keys=["observation"],
                out_keys=["loc", "scale"],
            ),
            in_keys=["loc", "scale"],
            spec=Composite(action=proof_environment.action_spec),
            distribution_class=distribution_class,
            distribution_kwargs=distribution_kwargs,
            return_log_prob=True,
            default_interaction_type=ExplorationType.RANDOM,
        )

        # Define value architecture
        value_mlp = MLP(
            in_features=input_shape[-1],
            activation_class=torch.nn.Tanh,
            out_features=1,
            num_cells=[64, 64],
        )

        # Initialize value weights
        for layer in value_mlp.modules():
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.orthogonal_(layer.weight, 0.01)
                layer.bias.data.zero_()

        # Define value module
        value_module = ValueOperator(
            value_mlp,
            in_keys=["observation"],
        )

        return policy_module, value_module

    @classmethod
    def make_ppo_models(cls, env_config: EnvConfig):
        proof_environment = cls.make_env(env_config)
        actor, critic = cls.make_ppo_models_state(proof_environment)
        return actor, critic

    # ====================================================================
    # Evaluation utils
    # --------------------------------------------------------------------

    @classmethod
    def dump_video(cls, module):
        if isinstance(module, VideoRecorder):
            module.dump()

    @classmethod
    def eval_model(cls, actor, test_env, num_episodes=3):
        test_rewards = []
        for _ in range(num_episodes):
            td_test = test_env.rollout(
                policy=actor,
                auto_reset=True,
                auto_cast_to_device=True,
                break_when_any_done=True,
                max_steps=10_000_000,
            )
            reward = td_test["next", "episode_reward"][td_test["next", "done"]]
            test_rewards.append(reward.cpu())
            test_env.apply(cls.dump_video)
        del td_test
        return torch.cat(test_rewards, 0).mean()

    # ====================================================================
    # Data (replay buffer) utils
    # --------------------------------------------------------------------

    def make_rb(self, cfg: PPOConfig) -> ReplayBuffer:
        # Create data buffer
        sampler = SamplerWithoutReplacement()
        data_buffer = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(cfg.collector.frames_per_batch),
            sampler=sampler,
            batch_size=cfg.loss.mini_batch_size,
        )
        return data_buffer

    # ====================================================================
    # Loss utils
    # --------------------------------------------------------------------

    def make_loss(self, *, loss_cfg: LossConfig | None = None, actor: TensorDictModule, critic: TensorDictModule):

        # Create loss and adv modules
        adv_module = GAE(
            gamma=loss_cfg.gamma,
            lmbda=loss_cfg.gae_lambda,
            value_network=critic,
            average_gae=False,
        )

        loss_module = ClipPPOLoss(
            actor_network=actor,
            critic_network=critic,
            clip_epsilon=loss_cfg.clip_epsilon,
            loss_critic_type=loss_cfg.loss_critic_type,
            entropy_coef=loss_cfg.entropy_coef,
            critic_coef=loss_cfg.critic_coef,
            normalize_advantage=True,
        )
        return adv_module, loss_module


    # ====================================================================
    # Optim utils
    # --------------------------------------------------------------------

    def make_optimizer(self, *, optimizer_cfg: OptimConfig, actor: TensorDictModule, critic: TensorDictModule):
        # Create optimizers
        actor_optim = torch.optim.Adam(actor.parameters(), lr=optimizer_cfg.lr, eps=1e-5)
        critic_optim = torch.optim.Adam(critic.parameters(), lr=optimizer_cfg.lr, eps=1e-5)
        return actor_optim, critic_optim

class AtariPPOTrainer(PPOTrainer):

    @classmethod
    def default_config(cls) -> omegaconf.OmegaConf:
        from omegaconf import OmegaConf

        return OmegaConf.load("config_atari.yaml")

    # ====================================================================
    # Environment utils
    # --------------------------------------------------------------------

    @classmethod
    def make_base_env(
        cls, env_name="BreakoutNoFrameskip-v4", frame_skip=4, is_test=False
    ):
        env = GymEnv(
            env_name,
            frame_skip=frame_skip,
            from_pixels=True,
            pixels_only=False,
            device="cpu",
        )
        env = TransformedEnv(env)
        env.append_transform(NoopResetEnv(noops=30, random=True))
        if not is_test:
            env.append_transform(EndOfLifeTransform())
        return env

    @classmethod
    def make_parallel_env(cls, env_name, num_envs, device, is_test=False):
        env = ParallelEnv(
            num_envs,
            EnvCreator(cls.make_base_env),
            serial_for_single=True,
            device=device,
        )
        env = TransformedEnv(env)
        env.append_transform(
            RenameTransform(in_keys=["pixels"], out_keys=["pixels_int"])
        )
        env.append_transform(ToTensorImage(in_keys=["pixels_int"], out_keys=["pixels"]))
        env.append_transform(GrayScale())
        env.append_transform(Resize(84, 84))
        env.append_transform(CatFrames(N=4, dim=-3))
        env.append_transform(RewardSum())
        env.append_transform(StepCounter(max_steps=4500))
        if not is_test:
            env.append_transform(SignTransform(in_keys=["reward"]))
        env.append_transform(DoubleToFloat())
        env.append_transform(VecNorm(in_keys=["pixels"]))
        return env

    # ====================================================================
    # Model utils
    # --------------------------------------------------------------------

    @classmethod
    def make_ppo_modules_pixels(cls, proof_environment):

        # Define input shape
        input_shape = proof_environment.observation_spec["pixels"].shape

        # Define distribution class and kwargs
        if isinstance(proof_environment.action_spec.space, CategoricalBox):
            num_outputs = proof_environment.action_spec.space.n
            distribution_class = OneHotCategorical
            distribution_kwargs = {}
        else:  # is ContinuousBox
            num_outputs = proof_environment.action_spec.shape
            distribution_class = TanhNormal
            distribution_kwargs = {
                "low": proof_environment.action_spec.space.low,
                "high": proof_environment.action_spec.space.high,
            }

        # Define input keys
        in_keys = ["pixels"]

        # Define a shared Module and TensorDictModule (CNN + MLP)
        common_cnn = ConvNet(
            activation_class=torch.nn.ReLU,
            num_cells=[32, 64, 64],
            kernel_sizes=[8, 4, 3],
            strides=[4, 2, 1],
        )
        common_cnn_output = common_cnn(torch.ones(input_shape))
        common_mlp = MLP(
            in_features=common_cnn_output.shape[-1],
            activation_class=torch.nn.ReLU,
            activate_last_layer=True,
            out_features=512,
            num_cells=[],
        )
        common_mlp_output = common_mlp(common_cnn_output)

        # Define shared net as TensorDictModule
        common_module = TensorDictModule(
            module=torch.nn.Sequential(common_cnn, common_mlp),
            in_keys=in_keys,
            out_keys=["common_features"],
        )

        # Define on head for the policy
        policy_net = MLP(
            in_features=common_mlp_output.shape[-1],
            out_features=num_outputs,
            activation_class=torch.nn.ReLU,
            num_cells=[],
        )
        policy_module = TensorDictModule(
            module=policy_net,
            in_keys=["common_features"],
            out_keys=["logits"],
        )

        # Add probabilistic sampling of the actions
        policy_module = ProbabilisticActor(
            policy_module,
            in_keys=["logits"],
            spec=Composite(action=proof_environment.action_spec),
            distribution_class=distribution_class,
            distribution_kwargs=distribution_kwargs,
            return_log_prob=True,
            default_interaction_type=ExplorationType.RANDOM,
        )

        # Define another head for the value
        value_net = MLP(
            activation_class=torch.nn.ReLU,
            in_features=common_mlp_output.shape[-1],
            out_features=1,
            num_cells=[],
        )
        value_module = ValueOperator(
            value_net,
            in_keys=["common_features"],
        )

        return common_module, policy_module, value_module

    @classmethod
    def make_ppo_models(cls, env_name):

        proof_environment = cls.make_parallel_env(env_name, 1, device="cpu")
        common_module, policy_module, value_module = cls.make_ppo_modules_pixels(
            proof_environment
        )

        # Wrap modules in a single ActorCritic operator
        actor_critic = ActorValueOperator(
            common_operator=common_module,
            policy_operator=policy_module,
            value_operator=value_module,
        )

        with torch.no_grad():
            td = proof_environment.rollout(max_steps=100, break_when_any_done=False)
            td = actor_critic(td)
            del td

        actor = actor_critic.get_policy_operator()
        critic = actor_critic.get_value_operator()

        del proof_environment

        return actor, critic

    # ====================================================================
    # Evaluation utils
    # --------------------------------------------------------------------

    @classmethod
    def dump_video(cls, module):
        if isinstance(module, VideoRecorder):
            module.dump()

    @classmethod
    def eval_model(cls, actor, test_env, num_episodes=3):
        test_rewards = []
        for _ in range(num_episodes):
            td_test = test_env.rollout(
                policy=actor,
                auto_reset=True,
                auto_cast_to_device=True,
                break_when_any_done=True,
                max_steps=10_000_000,
            )
            test_env.apply(dump_video)
            reward = td_test["next", "episode_reward"][td_test["next", "done"]]
            test_rewards.append(reward.cpu())
        del td_test
        return torch.cat(test_rewards, 0).mean()
