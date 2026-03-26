# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import copy
from functools import partial

import torch
from omegaconf import OmegaConf
from tensordict.nn import (
    InteractionType,
    ProbabilisticTensorDictSequential,
    TensorDictModule,
    TensorDictModuleWrapper,
)
from torch import distributions as d, nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchrl._utils import logger as torchrl_logger, VERBOSE
from torchrl.collectors import DataCollectorBase
from torchrl.data import (
    LazyMemmapStorage,
    MultiStep,
    PrioritizedSampler,
    RandomSampler,
    ReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.data.utils import DEVICE_TYPING
from torchrl.envs import (
    CatFrames,
    CatTensors,
    CenterCrop,
    Compose,
    DMControlEnv,
    DoubleToFloat,
    env_creator,
    EnvBase,
    EnvCreator,
    FlattenObservation,
    GrayScale,
    gSDENoise,
    GymEnv,
    InitTracker,
    NoopResetEnv,
    ObservationNorm,
    ParallelEnv,
    Resize,
    RewardScaling,
    StepCounter,
    ToTensorImage,
    TransformedEnv,
    VecNorm,
)
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import (
    ActorCriticOperator,
    ActorValueOperator,
    DdpgCnnActor,
    DdpgCnnQNet,
    MLP,
    NoisyLinear,
    NormalParamExtractor,
    ProbabilisticActor,
    SafeModule,
    SafeSequential,
    TanhNormal,
    ValueOperator,
)
from torchrl.modules.distributions.continuous import SafeTanhTransform
from torchrl.modules.models.exploration import LazygSDEModule
from torchrl.objectives import HardUpdate, LossModule, SoftUpdate, TargetNetUpdater
from torchrl.objectives.deprecated import REDQLoss_deprecated
from torchrl.record.loggers import Logger
from torchrl.record.recorder import VideoRecorder
from torchrl.trainers.helpers import sync_async_collector, sync_sync_collector
from torchrl.trainers.trainers import (
    BatchSubSampler,
    ClearCudaCache,
    CountFramesLog,
    LogScalar,
    LogValidationReward,
    ReplayBufferTrainer,
    RewardNormalizer,
    Trainer,
    UpdateWeights,
)

LIBS = {
    "gym": GymEnv,
    "dm_control": DMControlEnv,
}
ACTIVATIONS = {
    "elu": nn.ELU,
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
}
OPTIMIZERS = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
    "adamax": optim.Adamax,
}


def correct_for_frame_skip(cfg: DictConfig) -> DictConfig:  # noqa: F821
    """Correct the arguments for the input frame_skip, by dividing all the arguments that reflect a count of frames by the frame_skip.

    This is aimed at avoiding unknowingly over-sampling from the environment, i.e. targeting a total number of frames
    of 1M but actually collecting frame_skip * 1M frames.

    Args:
        cfg (DictConfig): DictConfig containing some frame-counting argument, including:
            "max_frames_per_traj", "total_frames", "frames_per_batch", "record_frames", "annealing_frames",
            "init_random_frames", "init_env_steps"

    Returns:
         the input DictConfig, modified in-place.

    """

    def _hasattr(field):
        local_cfg = cfg
        fields = field.split(".")
        for f in fields:
            if not hasattr(local_cfg, f):
                return False
            local_cfg = getattr(local_cfg, f)
        else:
            return True

    def _getattr(field):
        local_cfg = cfg
        fields = field.split(".")
        for f in fields:
            local_cfg = getattr(local_cfg, f)
        return local_cfg

    def _setattr(field, val):
        local_cfg = cfg
        fields = field.split(".")
        for f in fields[:-1]:
            local_cfg = getattr(local_cfg, f)
        setattr(local_cfg, field[-1], val)

    # Adapt all frame counts wrt frame_skip
    frame_skip = cfg.env.frame_skip
    if frame_skip != 1:
        fields = [
            "collector.max_frames_per_traj",
            "collector.total_frames",
            "collector.frames_per_batch",
            "logger.record_frames",
            "exploration.annealing_frames",
            "collector.init_random_frames",
            "env.init_env_steps",
            "env.noops",
        ]
        for field in fields:
            if _hasattr(cfg, field):
                _setattr(field, _getattr(field) // frame_skip)
    return cfg


def make_trainer(
    collector: DataCollectorBase,
    loss_module: LossModule,
    recorder: EnvBase | None,
    target_net_updater: TargetNetUpdater | None,
    policy_exploration: TensorDictModuleWrapper | TensorDictModule | None,
    replay_buffer: ReplayBuffer | None,
    logger: Logger | None,
    cfg: DictConfig,  # noqa: F821
) -> Trainer:
    """Creates a Trainer instance given its constituents.

    Args:
        collector (DataCollectorBase): A data collector to be used to collect data.
        loss_module (LossModule): A TorchRL loss module
        recorder (EnvBase, optional): a recorder environment.
        target_net_updater (TargetNetUpdater): A target network update object.
        policy_exploration (TDModule or TensorDictModuleWrapper): a policy to be used for recording and exploration
            updates (should be synced with the learnt policy).
        replay_buffer (ReplayBuffer): a replay buffer to be used to collect data.
        logger (Logger): a Logger to be used for logging.
        cfg (DictConfig): a DictConfig containing the arguments of the script.

    Returns:
        A trainer built with the input objects. The optimizer is built by this helper function using the cfg provided.

    Examples:
        >>> import torch
        >>> import tempfile
        >>> from torchrl.trainers.loggers import TensorboardLogger
        >>> from torchrl.trainers import Trainer
        >>> from torchrl.envs import EnvCreator
        >>> from torchrl.collectors import SyncDataCollector
        >>> from torchrl.data import TensorDictReplayBuffer
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> from torchrl.modules import TensorDictModuleWrapper, SafeModule, ValueOperator, EGreedyWrapper
        >>> from torchrl.objectives.common import LossModule
        >>> from torchrl.objectives.utils import TargetNetUpdater
        >>> from torchrl.objectives import DDPGLoss
        >>> env_maker = EnvCreator(lambda: GymEnv("Pendulum-v0"))
        >>> env_proof = env_maker()
        >>> obs_spec = env_proof.observation_spec
        >>> action_spec = env_proof.action_spec
        >>> net = torch.nn.Linear(env_proof.observation_spec.shape[-1], action_spec.shape[-1])
        >>> net_value = torch.nn.Linear(env_proof.observation_spec.shape[-1], 1)  # for the purpose of testing
        >>> policy = SafeModule(action_spec, net, in_keys=["observation"], out_keys=["action"])
        >>> value = ValueOperator(net_value, in_keys=["observation"], out_keys=["state_action_value"])
        >>> collector = SyncDataCollector(env_maker, policy, total_frames=100)
        >>> loss_module = DDPGLoss(policy, value, gamma=0.99)
        >>> recorder = env_proof
        >>> target_net_updater = None
        >>> policy_exploration = EGreedyWrapper(policy)
        >>> replay_buffer = TensorDictReplayBuffer()
        >>> dir = tempfile.gettempdir()
        >>> logger = TensorboardLogger(exp_name=dir)
        >>> trainer = make_trainer(collector, loss_module, recorder, target_net_updater, policy_exploration,
        ...    replay_buffer, logger)
        >>> torchrl_logger.info(trainer)

    """

    optimizer = OPTIMIZERS[cfg.optim.optimizer](
        loss_module.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        eps=cfg.optim.eps,
        **OmegaConf.to_container(cfg.optim.kwargs),
    )
    device = next(loss_module.parameters()).device
    if cfg.optim.lr_scheduler == "cosine":
        optim_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=int(
                cfg.collector.total_frames
                / cfg.collector.frames_per_batch
                * cfg.optim.steps_per_batch
            ),
        )
    elif cfg.optim.lr_scheduler == "":
        optim_scheduler = None
    else:
        raise NotImplementedError(f"lr scheduler {cfg.optim.lr_scheduler}")

    if VERBOSE:
        torchrl_logger.info(
            f"collector = {collector}; \n"
            f"loss_module = {loss_module}; \n"
            f"recorder = {recorder}; \n"
            f"target_net_updater = {target_net_updater}; \n"
            f"policy_exploration = {policy_exploration}; \n"
            f"replay_buffer = {replay_buffer}; \n"
            f"logger = {logger}; \n"
            f"cfg = {cfg}; \n"
        )

    if logger is not None:
        # log hyperparams
        logger.log_hparams(cfg)

    trainer = Trainer(
        collector=collector,
        frame_skip=cfg.env.frame_skip,
        total_frames=cfg.collector.total_frames * cfg.env.frame_skip,
        loss_module=loss_module,
        optimizer=optimizer,
        logger=logger,
        optim_steps_per_batch=cfg.optim.steps_per_batch,
        clip_grad_norm=cfg.optim.clip_grad_norm,
        clip_norm=cfg.optim.clip_norm,
    )

    if torch.cuda.device_count() > 0:
        trainer.register_op("pre_optim_steps", ClearCudaCache(1))

    trainer.register_op("batch_process", lambda batch: batch.cpu())

    if replay_buffer is not None:
        # replay buffer is used 2 or 3 times: to register data, to sample
        # data and to update priorities
        rb_trainer = ReplayBufferTrainer(
            replay_buffer,
            cfg.buffer.batch_size,
            flatten_tensordicts=True,
            memmap=False,
            device=device,
        )

        trainer.register_op("batch_process", rb_trainer.extend)
        trainer.register_op("process_optim_batch", rb_trainer.sample)
        trainer.register_op("post_loss", rb_trainer.update_priority)
    else:
        # trainer.register_op("batch_process", mask_batch)
        trainer.register_op(
            "process_optim_batch",
            BatchSubSampler(
                batch_size=cfg.buffer.batch_size, sub_traj_len=cfg.buffer.sub_traj_len
            ),
        )
        trainer.register_op("process_optim_batch", lambda batch: batch.to(device))

    if optim_scheduler is not None:
        trainer.register_op("post_optim", optim_scheduler.step)

    if target_net_updater is not None:
        trainer.register_op("post_optim", target_net_updater.step)

    if cfg.env.normalize_rewards_online:
        # if used the running statistics of the rewards are computed and the
        # rewards used for training will be normalized based on these.
        reward_normalizer = RewardNormalizer(
            scale=cfg.env.normalize_rewards_online_scale,
            decay=cfg.env.normalize_rewards_online_decay,
        )
        trainer.register_op("batch_process", reward_normalizer.update_reward_stats)
        trainer.register_op("process_optim_batch", reward_normalizer.normalize_reward)

    if policy_exploration is not None and hasattr(policy_exploration, "step"):
        trainer.register_op(
            "post_steps", policy_exploration.step, frames=cfg.collector.frames_per_batch
        )

    trainer.register_op(
        "post_steps_log", lambda *cfg: {"lr": optimizer.param_groups[0]["lr"]}
    )

    if recorder is not None:
        # create recorder object
        recorder_obj = LogValidationReward(
            record_frames=cfg.logger.record_frames,
            frame_skip=cfg.env.frame_skip,
            policy_exploration=policy_exploration,
            environment=recorder,
            record_interval=cfg.logger.record_interval,
            log_keys=cfg.logger.recorder_log_keys,
        )
        # register recorder
        trainer.register_op(
            "post_steps_log",
            recorder_obj,
        )
        # call recorder - could be removed
        recorder_obj(None)
        # create explorative recorder - could be optional
        recorder_obj_explore = LogValidationReward(
            record_frames=cfg.logger.record_frames,
            frame_skip=cfg.env.frame_skip,
            policy_exploration=policy_exploration,
            environment=recorder,
            record_interval=cfg.logger.record_interval,
            exploration_type=ExplorationType.RANDOM,
            suffix="exploration",
            out_keys={("next", "reward"): "r_evaluation_exploration"},
        )
        # register recorder
        trainer.register_op(
            "post_steps_log",
            recorder_obj_explore,
        )
        # call recorder - could be removed
        recorder_obj_explore(None)

    trainer.register_op(
        "post_steps", UpdateWeights(collector, update_weights_interval=1)
    )

    trainer.register_op("pre_steps_log", LogScalar())
    trainer.register_op("pre_steps_log", CountFramesLog(frame_skip=cfg.env.frame_skip))

    return trainer


def make_redq_model(
    proof_environment: EnvBase,
    cfg: DictConfig,  # noqa: F821
    device: DEVICE_TYPING = "cpu",
    in_keys: Sequence[str] | None = None,
    actor_net_kwargs=None,
    qvalue_net_kwargs=None,
    observation_key=None,
    **kwargs,
) -> nn.ModuleList:
    """Actor and Q-value model constructor helper function for REDQ.

    Follows default parameters proposed in REDQ original paper: https://openreview.net/pdf?id=AY8zfZm0tDd.
    Other configurations can easily be implemented by modifying this function at will.
    A single instance of the Q-value model is returned. It will be multiplicated by the loss function.

    Args:
        proof_environment (EnvBase): a dummy environment to retrieve the observation and action spec
        cfg (DictConfig): contains arguments of the REDQ script
        device (torch.device, optional): device on which the model must be cast. Default is "cpu".
        in_keys (iterable of strings, optional): observation key to be read by the actor, usually one of
            `'observation_vector'` or `'pixels'`. If none is provided, one of these two keys is chosen
             based on the `cfg.from_pixels` argument.
        actor_net_kwargs (dict, optional): kwargs of the actor MLP.
        qvalue_net_kwargs (dict, optional): kwargs of the qvalue MLP.

    Returns:
         A nn.ModuleList containing the actor, qvalue operator(s) and the value operator.

    """
    torch.manual_seed(cfg.seed)
    tanh_loc = cfg.network.tanh_loc
    default_policy_scale = cfg.network.default_policy_scale
    gSDE = cfg.exploration.gSDE

    action_spec = proof_environment.action_spec_unbatched

    if actor_net_kwargs is None:
        actor_net_kwargs = {}
    if qvalue_net_kwargs is None:
        qvalue_net_kwargs = {}

    linear_layer_class = (
        torch.nn.Linear
        if not cfg.exploration.noisy
        else partial(NoisyLinear, use_exploration_type=True)
    )

    out_features_actor = (2 - gSDE) * action_spec.shape[-1]
    if cfg.env.from_pixels:
        if in_keys is None:
            in_keys_actor = ["pixels"]
        else:
            in_keys_actor = in_keys
        actor_net_kwargs_default = {
            "mlp_net_kwargs": {
                "layer_class": linear_layer_class,
                "activation_class": ACTIVATIONS[cfg.network.activation],
            },
            "conv_net_kwargs": {
                "activation_class": ACTIVATIONS[cfg.network.activation]
            },
        }
        actor_net_kwargs_default.update(actor_net_kwargs)
        actor_net = DdpgCnnActor(out_features_actor, **actor_net_kwargs_default)
        gSDE_state_key = "hidden"
        out_keys_actor = ["param", "hidden"]

        value_net_default_kwargs = {
            "mlp_net_kwargs": {
                "layer_class": linear_layer_class,
                "activation_class": ACTIVATIONS[cfg.network.activation],
            },
            "conv_net_kwargs": {
                "activation_class": ACTIVATIONS[cfg.network.activation]
            },
        }
        value_net_default_kwargs.update(qvalue_net_kwargs)

        in_keys_qvalue = ["pixels", "action"]
        qvalue_net = DdpgCnnQNet(**value_net_default_kwargs)
    else:
        if in_keys is None:
            in_keys_actor = ["observation_vector"]
        else:
            in_keys_actor = in_keys

        actor_net_kwargs_default = {
            "num_cells": [cfg.network.actor_cells] * cfg.network.actor_depth,
            "out_features": out_features_actor,
            "activation_class": ACTIVATIONS[cfg.network.activation],
        }
        actor_net_kwargs_default.update(actor_net_kwargs)
        actor_net = MLP(**actor_net_kwargs_default)
        out_keys_actor = ["param"]
        gSDE_state_key = in_keys_actor[0]

        qvalue_net_kwargs_default = {
            "num_cells": [cfg.network.qvalue_cells] * cfg.network.qvalue_depth,
            "out_features": 1,
            "activation_class": ACTIVATIONS[cfg.network.activation],
        }
        qvalue_net_kwargs_default.update(qvalue_net_kwargs)
        qvalue_net = MLP(
            **qvalue_net_kwargs_default,
        )
        in_keys_qvalue = in_keys_actor + ["action"]

    dist_class = TanhNormal
    dist_kwargs = {
        "low": action_spec.space.low,
        "high": action_spec.space.high,
        "tanh_loc": tanh_loc,
    }

    if not gSDE:
        actor_net = nn.Sequential(
            actor_net,
            NormalParamExtractor(
                scale_mapping=f"biased_softplus_{default_policy_scale}",
                scale_lb=cfg.network.scale_lb,
            ),
        )
        actor_module = SafeModule(
            actor_net,
            in_keys=in_keys_actor,
            out_keys=["loc", "scale"] + out_keys_actor[1:],
        )

    else:
        actor_module = SafeModule(
            actor_net,
            in_keys=in_keys_actor,
            out_keys=["action"] + out_keys_actor[1:],  # will be overwritten
        )

        if action_spec.domain == "continuous":
            min = action_spec.space.low
            max = action_spec.space.high
            transform = SafeTanhTransform()
            if (min != -1).any() or (max != 1).any():
                transform = d.ComposeTransform(
                    transform,
                    d.AffineTransform(loc=(max + min) / 2, scale=(max - min) / 2),
                )
        else:
            raise RuntimeError("cannot use gSDE with discrete actions")

        actor_module = SafeSequential(
            actor_module,
            SafeModule(
                LazygSDEModule(transform=transform, device=device),
                in_keys=["action", gSDE_state_key, "_eps_gSDE"],
                out_keys=["loc", "scale", "action", "_eps_gSDE"],
            ),
        )

    actor = ProbabilisticActor(
        spec=action_spec,
        in_keys=["loc", "scale"],
        module=actor_module,
        distribution_class=dist_class,
        distribution_kwargs=dist_kwargs,
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=True,
    )
    qvalue = ValueOperator(
        in_keys=in_keys_qvalue,
        module=qvalue_net,
    )
    model = nn.ModuleList([actor, qvalue]).to(device)

    # init nets
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = proof_environment.fake_tensordict()
        td = td.unsqueeze(-1)
        td = td.to(device)
        for net in model:
            net(td)
    del td
    return model


def transformed_env_constructor(
    cfg: DictConfig,  # noqa: F821
    video_tag: str = "",
    logger: Logger | None = None,
    stats: dict | None = None,
    norm_obs_only: bool = False,
    use_env_creator: bool = False,
    custom_env_maker: Callable | None = None,
    custom_env: EnvBase | None = None,
    return_transformed_envs: bool = True,
    action_dim_gsde: int | None = None,
    state_dim_gsde: int | None = None,
    batch_dims: int | None = 0,
    obs_norm_state_dict: dict | None = None,
) -> Callable | EnvCreator:
    """Returns an environment creator from an argparse.Namespace built with the appropriate parser constructor.

    Args:
        cfg (DictConfig): a DictConfig containing the arguments of the script.
        video_tag (str, optional): video tag to be passed to the Logger object
        logger (Logger, optional): logger associated with the script
        stats (dict, optional): a dictionary containing the :obj:`loc` and :obj:`scale` for the `ObservationNorm` transform
        norm_obs_only (bool, optional): If `True` and `VecNorm` is used, the reward won't be normalized online.
            Default is `False`.
        use_env_creator (bool, optional): whether the `EnvCreator` class should be used. By using `EnvCreator`,
            one can make sure that running statistics will be put in shared memory and accessible for all workers
            when using a `VecNorm` transform. Default is `True`.
        custom_env_maker (callable, optional): if your env maker is not part
            of torchrl env wrappers, a custom callable
            can be passed instead. In this case it will override the
            constructor retrieved from `args`.
        custom_env (EnvBase, optional): if an existing environment needs to be
            transformed_in, it can be passed directly to this helper. `custom_env_maker`
            and `custom_env` are exclusive features.
        return_transformed_envs (bool, optional): if ``True``, a transformed_in environment
            is returned.
        action_dim_gsde (int, Optional): if gSDE is used, this can present the action dim to initialize the noise.
            Make sure this is indicated in environment executed in parallel.
        state_dim_gsde: if gSDE is used, this can present the state dim to initialize the noise.
            Make sure this is indicated in environment executed in parallel.
        batch_dims (int, optional): number of dimensions of a batch of data. If a single env is
            used, it should be 0 (default). If multiple envs are being transformed in parallel,
            it should be set to 1 (or the number of dims of the batch).
        obs_norm_state_dict (dict, optional): the state_dict of the ObservationNorm transform to be loaded into the
            environment
    """

    def make_transformed_env(**kwargs) -> TransformedEnv:
        env_name = cfg.env.name
        env_task = cfg.env.task
        env_library = LIBS[cfg.env.library]
        frame_skip = cfg.env.frame_skip
        from_pixels = cfg.env.from_pixels
        categorical_action_encoding = cfg.env.categorical_action_encoding

        if custom_env is None and custom_env_maker is None:
            if cfg.collector.device in ("", None):
                device = "cpu" if not torch.cuda.is_available() else "cuda:0"
            elif isinstance(cfg.collector.device, str):
                device = cfg.collector.device
            elif isinstance(cfg.collector.device, Sequence):
                device = cfg.collector.device[0]
            else:
                raise ValueError(
                    "collector_device must be either a string or a sequence of strings"
                )
            env_kwargs = {
                "env_name": env_name,
                "device": device,
                "frame_skip": frame_skip,
                "from_pixels": from_pixels or len(video_tag),
                "pixels_only": from_pixels,
            }
            if env_library is GymEnv:
                env_kwargs.update(
                    {"categorical_action_encoding": categorical_action_encoding}
                )
            elif categorical_action_encoding:
                raise NotImplementedError(
                    "categorical_action_encoding=True is currently only compatible with GymEnvs."
                )
            if env_library is DMControlEnv:
                env_kwargs.update({"task_name": env_task})
            env_kwargs.update(kwargs)
            env = env_library(**env_kwargs)
        elif custom_env is None and custom_env_maker is not None:
            env = custom_env_maker(**kwargs)
        elif custom_env_maker is None and custom_env is not None:
            env = custom_env
        else:
            raise RuntimeError("cannot provide both custom_env and custom_env_maker")

        if cfg.env.noops and custom_env is None:
            # this is a bit hacky: if custom_env is not None, it is probably a ParallelEnv
            # that already has its NoopResetEnv set for the contained envs.
            # There is a risk however that we're just skipping the NoopsReset instantiation
            env = TransformedEnv(env, NoopResetEnv(cfg.env.noops))
        if not return_transformed_envs:
            return env

        return make_env_transforms(
            env,
            cfg,
            video_tag,
            logger,
            env_name,
            stats,
            norm_obs_only,
            env_library,
            action_dim_gsde,
            state_dim_gsde,
            batch_dims=batch_dims,
            obs_norm_state_dict=obs_norm_state_dict,
        )

    if use_env_creator:
        return env_creator(make_transformed_env)
    return make_transformed_env


def get_norm_state_dict(env):
    """Gets the normalization loc and scale from the env state_dict."""
    sd = env.state_dict()
    sd = {
        key: val
        for key, val in sd.items()
        if key.endswith("loc") or key.endswith("scale")
    }
    return sd


def initialize_observation_norm_transforms(
    proof_environment: EnvBase,
    num_iter: int = 1000,
    key: str | tuple[str, ...] = None,
):
    """Calls :obj:`ObservationNorm.init_stats` on all uninitialized :obj:`ObservationNorm` instances of a :obj:`TransformedEnv`.

    If an :obj:`ObservationNorm` already has non-null :obj:`loc` or :obj:`scale`, a call to :obj:`initialize_observation_norm_transforms` will be a no-op.
    Similarly, if the transformed environment does not contain any :obj:`ObservationNorm`, a call to this function will have no effect.
    If no key is provided but the observations of the :obj:`EnvBase` contains more than one key, an exception will
    be raised.

    Args:
        proof_environment (EnvBase instance, optional): if provided, this env will
            be used to execute the rollouts. If not, it will be created using
            the cfg object.
        num_iter (int): Number of iterations used for initializing the :obj:`ObservationNorms`
        key (str, optional): if provided, the stats of this key will be gathered.
            If not, it is expected that only one key exists in `env.observation_spec`.

    """
    if not isinstance(proof_environment.transform, Compose) and not isinstance(
        proof_environment.transform, ObservationNorm
    ):
        return

    if key is None:
        keys = list(proof_environment.base_env.observation_spec.keys(True, True))
        key = keys.pop()
        if len(keys):
            raise RuntimeError(
                f"More than one key exists in the observation_specs: {[key] + keys} were found, "
                "thus initialize_observation_norm_transforms cannot infer which to compute the stats of."
            )

    if isinstance(proof_environment.transform, Compose):
        for transform in proof_environment.transform:
            if isinstance(transform, ObservationNorm) and not transform.initialized:
                transform.init_stats(num_iter=num_iter, key=key)
    elif not proof_environment.transform.initialized:
        proof_environment.transform.init_stats(num_iter=num_iter, key=key)


def parallel_env_constructor(
    cfg: DictConfig, **kwargs  # noqa: F821
) -> ParallelEnv | EnvCreator:
    """Returns a parallel environment from an argparse.Namespace built with the appropriate parser constructor.

    Args:
        cfg (DictConfig): config containing user-defined arguments
        kwargs: keyword arguments for the `transformed_env_constructor` method.
    """
    batch_transform = cfg.env.batch_transform
    if not batch_transform:
        raise NotImplementedError(
            "batch_transform must be set to True for the recorder to be synced "
            "with the collection envs."
        )
    if cfg.collector.env_per_collector == 1:
        kwargs.update({"cfg": cfg, "use_env_creator": True})
        make_transformed_env = transformed_env_constructor(**kwargs)
        return make_transformed_env
    kwargs.update({"cfg": cfg, "use_env_creator": True})
    make_transformed_env = transformed_env_constructor(
        return_transformed_envs=not batch_transform, **kwargs
    )
    parallel_env = ParallelEnv(
        num_workers=cfg.collector.env_per_collector,
        create_env_fn=make_transformed_env,
        create_env_kwargs=None,
        serial_for_single=True,
        pin_memory=False,
    )
    if batch_transform:
        kwargs.update(
            {
                "cfg": cfg,
                "use_env_creator": False,
                "custom_env": parallel_env,
                "batch_dims": 1,
            }
        )
        env = transformed_env_constructor(**kwargs)()
        return env
    return parallel_env


def retrieve_observation_norms_state_dict(proof_environment: TransformedEnv):
    """Traverses the transforms of the environment and retrieves the :obj:`ObservationNorm` state dicts.

    Returns a list of tuple (idx, state_dict) for each :obj:`ObservationNorm` transform in proof_environment
    If the environment transforms do not contain any :obj:`ObservationNorm`, returns an empty list

    Args:
        proof_environment (EnvBase instance, optional): the :obj:``TransformedEnv` to retrieve the :obj:`ObservationNorm`
            state dict from
    """
    obs_norm_state_dicts = []

    if isinstance(proof_environment.transform, Compose):
        for idx, transform in enumerate(proof_environment.transform):
            if isinstance(transform, ObservationNorm):
                obs_norm_state_dicts.append((idx, transform.state_dict()))

    if isinstance(proof_environment.transform, ObservationNorm):
        obs_norm_state_dicts.append((0, proof_environment.transform.state_dict()))

    return obs_norm_state_dicts


def make_env_transforms(
    env,
    cfg,
    video_tag,
    logger,
    env_name,
    stats,
    norm_obs_only,
    env_library,
    action_dim_gsde,
    state_dim_gsde,
    batch_dims=0,
    obs_norm_state_dict=None,
):
    """Creates the typical transforms for and env."""
    env = TransformedEnv(env)

    from_pixels = cfg.env.from_pixels
    vecnorm = cfg.env.vecnorm
    norm_rewards = vecnorm and cfg.env.norm_rewards
    _norm_obs_only = norm_obs_only or not norm_rewards
    reward_scaling = cfg.env.reward_scaling
    reward_loc = cfg.env.reward_loc

    if len(video_tag):
        center_crop = cfg.env.center_crop
        if center_crop:
            center_crop = center_crop[0]
        env.append_transform(
            VideoRecorder(
                logger=logger,
                tag=f"{video_tag}_{env_name}_video",
                center_crop=center_crop,
            ),
        )

    if from_pixels:
        if not cfg.env.catframes:
            raise RuntimeError(
                "this env builder currently only accepts positive catframes values"
                "when pixels are being used."
            )
        env.append_transform(ToTensorImage())
        if cfg.env.center_crop:
            env.append_transform(CenterCrop(*cfg.env.center_crop))
        env.append_transform(Resize(cfg.env.image_size, cfg.env.image_size))
        if cfg.env.grayscale:
            env.append_transform(GrayScale())
        env.append_transform(FlattenObservation(0, -3, allow_positive_dim=True))
        env.append_transform(CatFrames(N=cfg.env.catframes, in_keys=["pixels"], dim=-3))
        if stats is None and obs_norm_state_dict is None:
            obs_stats = {}
        elif stats is None:
            obs_stats = copy(obs_norm_state_dict)
        else:
            obs_stats = copy(stats)
        obs_stats["standard_normal"] = True
        obs_norm = ObservationNorm(**obs_stats, in_keys=["pixels"])
        env.append_transform(obs_norm)
    if norm_rewards:
        reward_scaling = 1.0
        reward_loc = 0.0
    if norm_obs_only:
        reward_scaling = 1.0
        reward_loc = 0.0
    if reward_scaling is not None:
        env.append_transform(RewardScaling(reward_loc, reward_scaling))

    if not from_pixels:
        selected_keys = [
            key
            for key in env.observation_spec.keys(True, True)
            if ("pixels" not in key) and (key not in env.state_spec.keys(True, True))
        ]

        # even if there is a single tensor, it'll be renamed in "observation_vector"
        out_key = "observation_vector"
        env.append_transform(CatTensors(in_keys=selected_keys, out_key=out_key))

        if not vecnorm:
            if stats is None and obs_norm_state_dict is None:
                _stats = {}
            elif stats is None:
                _stats = copy(obs_norm_state_dict)
            else:
                _stats = copy(stats)
            _stats.update({"standard_normal": True})
            obs_norm = ObservationNorm(
                **_stats,
                in_keys=[out_key],
            )
            env.append_transform(obs_norm)
        else:
            env.append_transform(
                VecNorm(
                    in_keys=[out_key, "reward"] if not _norm_obs_only else [out_key],
                    decay=0.9999,
                )
            )

        env.append_transform(DoubleToFloat())

        if hasattr(cfg, "catframes") and cfg.env.catframes:
            env.append_transform(
                CatFrames(N=cfg.env.catframes, in_keys=[out_key], dim=-1)
            )

    else:
        env.append_transform(DoubleToFloat())

    if hasattr(cfg, "gSDE") and cfg.exploration.gSDE:
        env.append_transform(
            gSDENoise(action_dim=action_dim_gsde, state_dim=state_dim_gsde)
        )

    env.append_transform(StepCounter())
    env.append_transform(InitTracker())

    return env


def make_redq_loss(model, cfg) -> tuple[REDQLoss_deprecated, TargetNetUpdater | None]:
    """Builds the REDQ loss module."""
    loss_kwargs = {}
    loss_kwargs.update({"loss_function": cfg.loss.loss_function})
    loss_kwargs.update({"delay_qvalue": cfg.loss.type == "double"})
    loss_class = REDQLoss_deprecated
    if isinstance(model, ActorValueOperator):
        actor_model = model.get_policy_operator()
        qvalue_model = model.get_value_operator()
    elif isinstance(model, ActorCriticOperator):
        raise RuntimeError(
            "Although REDQ Q-value depends upon selected actions, using the"
            "ActorCriticOperator will lead to resampling of the actions when"
            "computing the Q-value loss, which we don't want. Please use the"
            "ActorValueOperator instead."
        )
    else:
        actor_model, qvalue_model = model

    loss_module = loss_class(
        actor_network=actor_model,
        qvalue_network=qvalue_model,
        num_qvalue_nets=cfg.loss.num_q_values,
        gSDE=cfg.exploration.gSDE,
        **loss_kwargs,
    )
    loss_module.make_value_estimator(gamma=cfg.loss.gamma)
    target_net_updater = make_target_updater(cfg, loss_module)
    return loss_module, target_net_updater


def make_target_updater(
    cfg: DictConfig, loss_module: LossModule  # noqa: F821
) -> TargetNetUpdater | None:
    """Builds a target network weight update object."""
    if cfg.loss.type == "double":
        if not cfg.loss.hard_update:
            target_net_updater = SoftUpdate(
                loss_module, eps=1 - 1 / cfg.loss.value_network_update_interval
            )
        else:
            target_net_updater = HardUpdate(
                loss_module,
                value_network_update_interval=cfg.loss.value_network_update_interval,
            )
    else:
        if cfg.hard_update:
            raise RuntimeError(
                "hard/soft-update are supposed to be used with double SAC loss. "
                "Consider using --loss=double or discarding the hard_update flag."
            )
        target_net_updater = None
    return target_net_updater


def make_collector_offpolicy(
    make_env: Callable[[], EnvBase],
    actor_model_explore: TensorDictModuleWrapper | ProbabilisticTensorDictSequential,
    cfg: DictConfig,  # noqa: F821
    make_env_kwargs: dict | None = None,
) -> DataCollectorBase:
    """Returns a data collector for off-policy sota-implementations.

    Args:
        make_env (Callable): environment creator
        actor_model_explore (SafeModule): Model instance used for evaluation and exploration update
        cfg (DictConfig): config for creating collector object
        make_env_kwargs (dict): kwargs for the env creator

    """
    if cfg.collector.async_collection:
        collector_helper = sync_async_collector
    else:
        collector_helper = sync_sync_collector

    if cfg.collector.multi_step:
        ms = MultiStep(
            gamma=cfg.loss.gamma,
            n_steps=cfg.collector.n_steps_return,
        )
    else:
        ms = None

    env_kwargs = {}
    if make_env_kwargs is not None and isinstance(make_env_kwargs, dict):
        env_kwargs.update(make_env_kwargs)
    elif make_env_kwargs is not None:
        env_kwargs = make_env_kwargs
    if cfg.collector.device in ("", None):
        cfg.collector.device = "cpu" if not torch.cuda.is_available() else "cuda:0"
    else:
        cfg.collector.device = (
            cfg.collector.device
            if len(cfg.collector.device) > 1
            else cfg.collector.device[0]
        )
    collector_helper_kwargs = {
        "env_fns": make_env,
        "env_kwargs": env_kwargs,
        "policy": actor_model_explore,
        "max_frames_per_traj": cfg.collector.max_frames_per_traj,
        "frames_per_batch": cfg.collector.frames_per_batch,
        "total_frames": cfg.collector.total_frames,
        "postproc": ms,
        "num_env_per_collector": 1,
        # we already took care of building the make_parallel_env function
        "num_collectors": -cfg.num_workers // -cfg.collector.env_per_collector,
        "device": cfg.collector.device,
        "init_random_frames": cfg.collector.init_random_frames,
        "split_trajs": True,
        # trajectories must be separated if multi-step is used
    }

    collector = collector_helper(**collector_helper_kwargs)
    collector.set_seed(cfg.seed)
    return collector


def make_replay_buffer(
    device: DEVICE_TYPING, cfg: DictConfig  # noqa: F821
) -> ReplayBuffer:  # noqa: F821
    """Builds a replay buffer using the config built from ReplayArgsConfig."""
    device = torch.device(device)
    if not cfg.buffer.prb:
        sampler = RandomSampler()
    else:
        sampler = PrioritizedSampler(
            max_capacity=cfg.buffer.size,
            alpha=0.7,
            beta=0.5,
        )
    buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(
            cfg.buffer.size,
            scratch_dir=cfg.buffer.scratch_dir,
        ),
        sampler=sampler,
        pin_memory=device != torch.device("cpu"),
        prefetch=cfg.buffer.prefetch,
        batch_size=cfg.buffer.batch_size,
    )
    return buffer
