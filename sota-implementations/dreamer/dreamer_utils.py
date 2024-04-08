# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import functools
import tempfile
from contextlib import nullcontext

import torch

import torch.nn as nn
from tensordict.nn import InteractionType
from torchrl.collectors import SyncDataCollector
from torchrl.data import SliceSampler, TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyMemmapStorage

from torchrl.data.tensor_specs import CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs import ParallelEnv

from torchrl.envs.env_creator import EnvCreator
from torchrl.envs.libs.dm_control import DMControlEnv
from torchrl.envs.libs.gym import GymEnv, set_gym_backend
from torchrl.envs.model_based.dreamer import DreamerEnv
from torchrl.envs.transforms import (
    Compose,
    DoubleToFloat,
    # ExcludeTransform,
    FrameSkipTransform,
    GrayScale,
    ObservationNorm,
    RandomCropTensorDict,
    Resize,
    RewardSum,
    ToTensorImage,
    TransformedEnv,
)
from torchrl.envs.transforms.transforms import (
    ExcludeTransform,
    RenameTransform,
    StepCounter,
    TensorDictPrimer,
)
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import (
    MLP,
    SafeModule,
    SafeProbabilisticModule,
    SafeProbabilisticTensorDictSequential,
    SafeSequential,
)
from torchrl.modules.distributions import IndependentNormal, TanhNormal
from torchrl.modules.models.model_based import (
    DreamerActor,
    ObsDecoder,
    ObsEncoder,
    RSSMPosterior,
    RSSMPrior,
    RSSMRollout,
)
from torchrl.modules.tensordict_module.exploration import AdditiveGaussianWrapper
from torchrl.modules.tensordict_module.world_models import WorldModelWrapper


def _make_env(cfg, device):
    lib = cfg.env.backend
    if lib in ("gym", "gymnasium"):
        with set_gym_backend(lib):
            return GymEnv(
                cfg.env.name,
                device=device,
            )
    elif lib == "dm_control":
        return DMControlEnv(cfg.env.name, cfg.env.task, from_pixels=cfg.env.from_pixels)
    else:
        raise NotImplementedError(f"Unknown lib {lib}.")


def transform_env(cfg, env, parallel_envs, dummy=False):
    env = TransformedEnv(env)
    if cfg.env.from_pixels:
        # transforms pixel from 0-255 to 0-1 (uint8 to float32)
        env.append_transform(
            RenameTransform(in_keys=["pixels"], out_keys=["pixels_int"])
        )
        env.append_transform(
            ToTensorImage(from_int=True, in_keys=["pixels_int"], out_keys=["pixels"])
        )
        if cfg.env.grayscale:
            env.append_transform(GrayScale())

        image_size = cfg.env.image_size
        env.append_transform(Resize(image_size, image_size))

    env.append_transform(DoubleToFloat())
    env.append_transform(RewardSum())
    env.append_transform(FrameSkipTransform(cfg.env.frame_skip))
    env.append_transform(StepCounter(cfg.env.horizon))
    if dummy:
        default_dict = {
            "state": UnboundedContinuousTensorSpec(shape=(cfg.networks.state_dim)),
            "belief": UnboundedContinuousTensorSpec(
                shape=(cfg.networks.rssm_hidden_dim)
            ),
        }
    else:
        default_dict = {
            "state": UnboundedContinuousTensorSpec(
                shape=(parallel_envs, cfg.networks.state_dim)
            ),
            "belief": UnboundedContinuousTensorSpec(
                shape=(parallel_envs, cfg.networks.rssm_hidden_dim)
            ),
        }
    env.append_transform(
        TensorDictPrimer(random=False, default_value=0, **default_dict)
    )

    return env


def make_environments(cfg, device, parallel_envs=1):
    """Make environments for training and evaluation."""
    func = functools.partial(_make_env, cfg=cfg, device=device)
    train_env = ParallelEnv(
        parallel_envs,
        EnvCreator(func),
        serial_for_single=True,
    )
    train_env = transform_env(cfg, train_env, parallel_envs)
    train_env.set_seed(cfg.env.seed)
    eval_env = ParallelEnv(
        parallel_envs,
        EnvCreator(func),
        serial_for_single=True,
    )
    eval_env = transform_env(cfg, eval_env, parallel_envs)
    eval_env.set_seed(cfg.env.seed + 1)
    check_env_specs(train_env)
    check_env_specs(eval_env)
    return train_env, eval_env


def make_dreamer(
    config,
    device,
    action_key: str = "action",
    value_key: str = "state_value",
    use_decoder_in_env: bool = False,
):
    test_env = _make_env(config, device="cpu")
    test_env = transform_env(config, test_env, parallel_envs=1, dummy=True)
    # Make encoder and decoder
    if config.env.from_pixels:
        encoder = ObsEncoder()
        decoder = ObsDecoder()
        observation_in_key = "pixels"
        obsevation_out_key = "reco_pixels"
    else:
        encoder = MLP(
            out_features=1024,
            depth=2,
            num_cells=config.networks.hidden_dim,
            activation_class=get_activation(config.networks.activation),
        )
        decoder = MLP(
            out_features=test_env.observation_spec["observation"].shape[-1],
            depth=2,
            num_cells=config.networks.hidden_dim,
            activation_class=get_activation(config.networks.activation),
        )
        # if config.env.backend == "dm_control":
        #     observation_in_key = ("position", "velocity")
        #     obsevation_out_key = "reco_observation"
        # else:
        observation_in_key = "observation"
        obsevation_out_key = "reco_observation"

    # Make RSSM
    rssm_prior = RSSMPrior(
        hidden_dim=config.networks.rssm_hidden_dim,
        rnn_hidden_dim=config.networks.rssm_hidden_dim,
        state_dim=config.networks.state_dim,
        action_spec=test_env.action_spec,
    )
    rssm_posterior = RSSMPosterior(
        hidden_dim=config.networks.rssm_hidden_dim, state_dim=config.networks.state_dim
    )
    # Make reward module
    reward_module = MLP(
        out_features=1,
        depth=2,
        num_cells=config.networks.hidden_dim,
        activation_class=get_activation(config.networks.activation),
    )

    # Make combined world model
    world_model = _dreamer_make_world_model(
        encoder,
        decoder,
        rssm_prior,
        rssm_posterior,
        reward_module,
        observation_in_key=observation_in_key,
        observation_out_key=obsevation_out_key,
    )
    world_model.to(device)

    # Initialize world model
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        tensordict = (
            test_env.rollout(5, auto_cast_to_device=True)
            .unsqueeze(-1)
            .to(world_model.device)
        )
        tensordict = tensordict.to_tensordict()
        world_model(tensordict)

    # Create model-based environment
    model_based_env = _dreamer_make_mbenv(
        reward_module=reward_module,
        rssm_prior=rssm_prior,
        decoder=decoder,
        observation_out_key=obsevation_out_key,
        test_env=test_env,
        use_decoder_in_env=use_decoder_in_env,
        state_dim=config.networks.state_dim,
        rssm_hidden_dim=config.networks.rssm_hidden_dim,
    )

    def detach_state_and_belief(data):
        data.set("state", data.get("state").detach())
        data.set("belief", data.get("belief").detach())
        return data

    model_based_env = model_based_env.append_transform(detach_state_and_belief)
    check_env_specs(model_based_env)

    # Make actor
    actor_simulator, actor_realworld = _dreamer_make_actors(
        encoder=encoder,
        observation_in_key=observation_in_key,
        rssm_prior=rssm_prior,
        rssm_posterior=rssm_posterior,
        mlp_num_units=config.networks.hidden_dim,
        activation=get_activation(config.networks.activation),
        action_key=action_key,
        test_env=test_env,
    )
    # Exploration noise to be added to the actor_realworld
    actor_realworld = AdditiveGaussianWrapper(
        actor_realworld,
        sigma_init=1.0,
        sigma_end=1.0,
        annealing_num_steps=1,
        mean=0.0,
        std=config.networks.exploration_noise,
    )

    # Make Critic
    value_model = _dreamer_make_value_model(
        hidden_dim=config.networks.hidden_dim,
        activation=config.networks.activation,
        value_key=value_key,
    )

    actor_simulator.to(device)
    value_model.to(device)
    actor_realworld.to(device)
    model_based_env.to(device)

    # Initialize model-based environment, actor and critic
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        tensordict = (
            model_based_env.fake_tensordict().unsqueeze(-1).to(value_model.device)
        )
        tensordict = tensordict
        tensordict = actor_simulator(tensordict)
        value_model(tensordict)

    return world_model, model_based_env, actor_simulator, value_model, actor_realworld


def make_collector(cfg, train_env, actor_model_explore):
    """Make collector."""
    collector = SyncDataCollector(
        train_env,
        actor_model_explore,
        init_random_frames=cfg.collector.init_random_frames,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        device=cfg.collector.device,
    )
    collector.set_seed(cfg.env.seed)

    return collector


def make_replay_buffer(
    batch_size,
    *,
    batch_seq_len,
    buffer_size=1000000,
    buffer_scratch_dir=None,
    device="cpu",
    prefetch=3,
    pixel_obs=True,
    grayscale=True,
    image_size,
):
    with (
        tempfile.TemporaryDirectory()
        if buffer_scratch_dir is None
        else nullcontext(buffer_scratch_dir)
    ) as scratch_dir:
        transforms = None
        if pixel_obs:

            def check_no_pixels(data):
                assert "pixels" not in data.keys()
                return data

            transforms = Compose(
                ExcludeTransform("pixels", ("next", "pixels"), inverse=True),
                check_no_pixels,  # will be called only during forward
                ToTensorImage(
                    in_keys=["pixels_int", ("next", "pixels_int")],
                    out_keys=["pixels", ("next", "pixels")],
                ),
            )
            if grayscale:
                transforms.append(GrayScale(in_keys=["pixels", ("next", "pixels")]))
            transforms.append(
                Resize(image_size, image_size, in_keys=["pixels", ("next", "pixels")])
            )

        replay_buffer = TensorDictReplayBuffer(
            pin_memory=False,
            prefetch=prefetch,
            storage=LazyMemmapStorage(
                buffer_size,
                scratch_dir=scratch_dir,
                device=device,
                ndim=2,
            ),
            sampler=SliceSampler(
                slice_len=batch_seq_len,
                strict_length=False,
                traj_key=("collector", "traj_ids"),
            ),
            transform=transforms,
            batch_size=batch_size,
        )
        return replay_buffer


def _dreamer_make_value_model(
    hidden_dim: int = 400, activation: str = "elu", value_key: str = "state_value"
):
    value_model = MLP(
        out_features=1,
        depth=3,
        num_cells=hidden_dim,
        activation_class=get_activation(activation),
    )
    value_model = SafeProbabilisticTensorDictSequential(
        SafeModule(
            value_model,
            in_keys=["state", "belief"],
            out_keys=["loc"],
        ),
        SafeProbabilisticModule(
            in_keys=["loc"],
            out_keys=[value_key],
            distribution_class=IndependentNormal,
            distribution_kwargs={"scale": 1.0, "event_dim": 1},
        ),
    )

    return value_model


def _dreamer_make_actors(
    encoder,
    observation_in_key,
    rssm_prior,
    rssm_posterior,
    mlp_num_units,
    activation,
    action_key,
    test_env,
):
    actor_module = DreamerActor(
        out_features=test_env.action_spec.shape[-1],
        depth=3,
        num_cells=mlp_num_units,
        activation_class=activation,
    )
    actor_simulator = _dreamer_make_actor_sim(action_key, test_env, actor_module)
    actor_realworld = _dreamer_make_actor_real(
        encoder,
        observation_in_key,
        rssm_prior,
        rssm_posterior,
        actor_module,
        action_key,
        test_env,
    )
    return actor_simulator, actor_realworld


def _dreamer_make_actor_sim(action_key, proof_environment, actor_module):
    actor_simulator = SafeProbabilisticTensorDictSequential(
        SafeModule(
            actor_module,
            in_keys=["state", "belief"],
            out_keys=["loc", "scale"],
            spec=CompositeSpec(
                **{
                    "loc": UnboundedContinuousTensorSpec(
                        proof_environment.action_spec.shape,
                        device=proof_environment.action_spec.device,
                    ),
                    "scale": UnboundedContinuousTensorSpec(
                        proof_environment.action_spec.shape,
                        device=proof_environment.action_spec.device,
                    ),
                }
            ),
        ),
        SafeProbabilisticModule(
            in_keys=["loc", "scale"],
            out_keys=[action_key],
            default_interaction_type=InteractionType.RANDOM,
            distribution_class=TanhNormal,
            distribution_kwargs={"tanh_loc": True},
            spec=CompositeSpec(**{action_key: proof_environment.action_spec}),
        ),
    )
    return actor_simulator


def _dreamer_make_actor_real(
    encoder,
    observation_in_key,
    rssm_prior,
    rssm_posterior,
    actor_module,
    action_key,
    proof_environment,
):
    # actor for real world: interacts with states ~ posterior
    # Out actor differs from the original paper where first they compute prior and posterior and then act on it
    # but we found that this approach worked better.
    actor_realworld = SafeSequential(
        SafeModule(
            encoder,
            in_keys=[observation_in_key],
            out_keys=["encoded_latents"],
        ),
        SafeModule(
            rssm_posterior,
            in_keys=["belief", "encoded_latents"],
            out_keys=[
                "_",
                "_",
                "state",
            ],
        ),
        SafeProbabilisticTensorDictSequential(
            SafeModule(
                actor_module,
                in_keys=["state", "belief"],
                out_keys=["loc", "scale"],
                spec=CompositeSpec(
                    **{
                        "loc": UnboundedContinuousTensorSpec(
                            proof_environment.action_spec.shape,
                        ),
                        "scale": UnboundedContinuousTensorSpec(
                            proof_environment.action_spec.shape,
                        ),
                    }
                ),
            ),
            SafeProbabilisticModule(
                in_keys=["loc", "scale"],
                out_keys=[action_key],
                default_interaction_type=InteractionType.MODE,
                distribution_class=TanhNormal,
                distribution_kwargs={"tanh_loc": True},
                spec=CompositeSpec(
                    **{action_key: proof_environment.action_spec.to("cpu")}
                ),
            ),
        ),
        SafeModule(
            rssm_prior,
            in_keys=["state", "belief", action_key],
            out_keys=[
                "_",
                "_",
                "_",  # we don't need the prior state
                ("next", "belief"),
            ],
        ),
    )
    return actor_realworld


def _dreamer_make_mbenv(
    reward_module,
    rssm_prior,
    test_env,
    decoder,
    observation_out_key: str = "reco_pixels",
    use_decoder_in_env: bool = False,
    state_dim: int = 30,
    rssm_hidden_dim: int = 200,
):
    # MB environment
    if use_decoder_in_env:
        mb_env_obs_decoder = SafeModule(
            decoder,
            in_keys=[("next", "state"), ("next", "belief")],
            out_keys=[("next", observation_out_key)],
        )
    else:
        mb_env_obs_decoder = None

    transition_model = SafeSequential(
        SafeModule(
            rssm_prior,
            in_keys=["state", "belief", "action"],
            out_keys=[
                "_",
                "_",
                "state",
                "belief",
            ],
        ),
    )

    reward_model = SafeProbabilisticTensorDictSequential(
        SafeModule(
            reward_module,
            in_keys=["state", "belief"],
            out_keys=["loc"],
        ),
        SafeProbabilisticModule(
            in_keys=["loc"],
            out_keys=["reward"],
            distribution_class=IndependentNormal,
            distribution_kwargs={"scale": 1.0, "event_dim": 1},
        ),
    )

    model_based_env = DreamerEnv(
        world_model=WorldModelWrapper(
            transition_model,
            reward_model,
        ),
        prior_shape=torch.Size([state_dim]),
        belief_shape=torch.Size([rssm_hidden_dim]),
        obs_decoder=mb_env_obs_decoder,
    )

    model_based_env.set_specs_from_env(test_env)
    return model_based_env


def _dreamer_make_world_model(
    encoder,
    decoder,
    rssm_prior,
    rssm_posterior,
    reward_module,
    observation_in_key: str = "pixels",
    observation_out_key: str = "reco_pixels",
):
    # World Model and reward model
    rssm_rollout = RSSMRollout(
        SafeModule(
            rssm_prior,
            in_keys=["state", "belief", "action"],
            out_keys=[
                ("next", "prior_mean"),
                ("next", "prior_std"),
                "_",
                ("next", "belief"),
            ],
        ),
        SafeModule(
            rssm_posterior,
            in_keys=[("next", "belief"), ("next", "encoded_latents")],
            out_keys=[
                ("next", "posterior_mean"),
                ("next", "posterior_std"),
                ("next", "state"),
            ],
        ),
    )
    event_dim = 3 if observation_out_key == "reco_pixels" else 1  # 3 for RGB
    decoder = SafeProbabilisticTensorDictSequential(
        SafeModule(
            decoder,
            in_keys=[("next", "state"), ("next", "belief")],
            out_keys=["loc"],
        ),
        SafeProbabilisticModule(
            in_keys=["loc"],
            out_keys=[("next", observation_out_key)],
            distribution_class=IndependentNormal,
            distribution_kwargs={"scale": 1.0, "event_dim": event_dim},
        ),
    )

    transition_model = SafeSequential(
        SafeModule(
            encoder,
            in_keys=[("next", observation_in_key)],
            out_keys=[("next", "encoded_latents")],
        ),
        rssm_rollout,
        decoder,
    )

    reward_model = SafeProbabilisticTensorDictSequential(
        SafeModule(
            reward_module,
            in_keys=[("next", "state"), ("next", "belief")],
            out_keys=[("next", "loc")],
        ),
        SafeProbabilisticModule(
            in_keys=[("next", "loc")],
            out_keys=[("next", "reward")],
            distribution_class=IndependentNormal,
            distribution_kwargs={"scale": 1.0, "event_dim": 1},
        ),
    )

    world_model = WorldModelWrapper(
        transition_model,
        reward_model,
    )
    return world_model


def log_metrics(logger, metrics, step):
    for metric_name, metric_value in metrics.items():
        logger.log_scalar(metric_name, metric_value, step)


def get_activation(name):
    if name == "relu":
        return nn.ReLU
    elif name == "tanh":
        return nn.Tanh
    elif name == "leaky_relu":
        return nn.LeakyReLU
    elif name == "elu":
        return nn.ELU
    else:
        raise NotImplementedError
