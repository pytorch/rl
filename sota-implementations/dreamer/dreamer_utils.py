# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import functools
import tempfile
from contextlib import nullcontext

import torch
from torchrl import logger as torchrl_logger

import torch.nn as nn
from tensordict import NestedKey
from tensordict.nn import (
    InteractionType,
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
    TensorDictModule,
    TensorDictSequential,
)
from torchrl.collectors import MultiCollector

from torchrl.data import (
    Composite,
    LazyMemmapStorage,
    SliceSampler,
    TensorDictReplayBuffer,
    Unbounded,
)

from torchrl.envs import (
    Compose,
    DeviceCastTransform,
    DMControlEnv,
    DoubleToFloat,
    DreamerDecoder,
    DreamerEnv,
    EnvCreator,
    ExcludeTransform,
    # ExcludeTransform,
    FrameSkipTransform,
    GrayScale,
    GymEnv,
    ParallelEnv,
    RenameTransform,
    Resize,
    RewardSum,
    set_gym_backend,
    StepCounter,
    TensorDictPrimer,
    ToTensorImage,
    TransformedEnv,
)
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import (
    AdditiveGaussianModule,
    DreamerActor,
    IndependentNormal,
    MLP,
    ObsDecoder,
    ObsEncoder,
    RSSMPosterior,
    RSSMPrior,
    RSSMRollout,
    SafeModule,
    SafeProbabilisticModule,
    SafeProbabilisticTensorDictSequential,
    SafeSequential,
    TanhNormal,
    WorldModelWrapper,
)
from torchrl.record import VideoRecorder


def _make_env(cfg, device, from_pixels=False):
    lib = cfg.env.backend
    if lib in ("gym", "gymnasium"):
        with set_gym_backend(lib):
            env = GymEnv(
                cfg.env.name,
                device=device,
                from_pixels=cfg.env.from_pixels or from_pixels,
                pixels_only=cfg.env.from_pixels,
            )
    elif lib == "dm_control":
        env = DMControlEnv(
            cfg.env.name,
            cfg.env.task,
            from_pixels=cfg.env.from_pixels or from_pixels,
            pixels_only=cfg.env.from_pixels,
            device=device,
        )
    else:
        raise NotImplementedError(f"Unknown lib {lib}.")
    default_dict = {
        "state": Unbounded(shape=(cfg.networks.state_dim,)),
        "belief": Unbounded(shape=(cfg.networks.rssm_hidden_dim,)),
    }
    env = env.append_transform(
        TensorDictPrimer(random=False, default_value=0, **default_dict)
    )
    return env


def transform_env(cfg, env):
    if not isinstance(env, TransformedEnv):
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

    return env


def make_environments(cfg, parallel_envs=1, logger=None):
    """Make environments for training and evaluation.

    Returns:
        train_env_factory: A callable that creates a training environment (for MultiCollector)
        eval_env: The evaluation environment instance
    """

    def train_env_factory():
        """Factory function for creating training environments."""
        func = functools.partial(
            _make_env, cfg=cfg, device=_default_device(cfg.env.device)
        )
        train_env = ParallelEnv(
            parallel_envs,
            EnvCreator(func),
            serial_for_single=True,
        )
        train_env = transform_env(cfg, train_env)
        train_env.set_seed(cfg.env.seed)
        return train_env

    # Create eval env directly (not a factory)
    func = functools.partial(
        _make_env,
        cfg=cfg,
        device=_default_device(cfg.env.device),
        from_pixels=cfg.logger.video,
    )
    eval_env = ParallelEnv(
        1,
        EnvCreator(func),
        serial_for_single=True,
    )
    eval_env = transform_env(cfg, eval_env)
    eval_env.set_seed(cfg.env.seed + 1)
    if cfg.logger.video:
        eval_env.insert_transform(0, VideoRecorder(logger, tag="eval/video"))

    # Check specs on a temporary train env
    temp_train_env = train_env_factory()
    check_env_specs(temp_train_env)
    temp_train_env.close()
    del temp_train_env

    check_env_specs(eval_env)
    return train_env_factory, eval_env


def dump_video(module):
    if isinstance(module, VideoRecorder):
        module.dump()


def _compute_encoder_output_size(image_size, channels=32, num_layers=4):
    """Compute the flattened output size of ObsEncoder."""
    # Compute spatial size after each conv layer (kernel=4, stride=2)
    size = image_size
    for _ in range(num_layers):
        size = (size - 4) // 2 + 1
    # Final channels = channels * 2^(num_layers-1)
    final_channels = channels * (2 ** (num_layers - 1))
    return final_channels * size * size


def make_dreamer(
    cfg,
    device,
    action_key: str = "action",
    value_key: str = "state_value",
    use_decoder_in_env: bool = False,
    compile: bool = True,
    logger=None,
):
    test_env = _make_env(cfg, device="cpu")
    test_env = transform_env(cfg, test_env)
    
    # Get dimensions for explicit module instantiation (avoids lazy modules)
    state_dim = cfg.networks.state_dim
    rssm_hidden_dim = cfg.networks.rssm_hidden_dim
    action_dim = test_env.action_spec.shape[-1]
    
    # Make encoder and decoder
    if cfg.env.from_pixels:
        # Determine input channels (1 for grayscale, 3 for RGB)
        in_channels = 1 if cfg.env.grayscale else 3
        image_size = cfg.env.image_size
        
        # Compute encoder output size for explicit posterior input
        obs_embed_dim = _compute_encoder_output_size(image_size, channels=32, num_layers=4)
        
        encoder = ObsEncoder(in_channels=in_channels)
        decoder = ObsDecoder(latent_dim=state_dim + rssm_hidden_dim)
        observation_in_key = "pixels"
        observation_out_key = "reco_pixels"
    else:
        obs_embed_dim = 1024  # MLP output size
        encoder = MLP(
            out_features=obs_embed_dim,
            depth=2,
            num_cells=cfg.networks.hidden_dim,
            activation_class=get_activation(cfg.networks.activation),
        )
        decoder = MLP(
            out_features=test_env.observation_spec["observation"].shape[-1],
            depth=2,
            num_cells=cfg.networks.hidden_dim,
            activation_class=get_activation(cfg.networks.activation),
        )
        observation_in_key = "observation"
        observation_out_key = "reco_observation"

    # Make RSSM with explicit input sizes (no lazy modules)
    rssm_prior = RSSMPrior(
        hidden_dim=rssm_hidden_dim,
        rnn_hidden_dim=rssm_hidden_dim,
        state_dim=state_dim,
        action_spec=test_env.action_spec,
        action_dim=action_dim,
    )
    rssm_posterior = RSSMPosterior(
        hidden_dim=rssm_hidden_dim,
        state_dim=state_dim,
        rnn_hidden_dim=rssm_hidden_dim,
        obs_embed_dim=obs_embed_dim,
    )

    # When use_scan=True, replace C++ GRU with Python-based GRU for torch.compile compatibility.
    # The C++ GRU (cuBLAS) cannot be traced by torch.compile when used inside scan on GPU.
    if cfg.networks.use_scan:
        from torchrl.modules.tensordict_module.rnn import GRUCell as PythonGRUCell

        old_rnn = rssm_prior.rnn
        python_rnn = PythonGRUCell(old_rnn.input_size, old_rnn.hidden_size)
        python_rnn.load_state_dict(old_rnn.state_dict())
        rssm_prior.rnn = python_rnn
        torchrl_logger.info("Switched RSSMPrior to Python-based GRU for scan compatibility")
    # Make reward module
    reward_module = MLP(
        out_features=1,
        depth=2,
        num_cells=cfg.networks.hidden_dim,
        activation_class=get_activation(cfg.networks.activation),
    )

    # Make combined world model
    world_model = _dreamer_make_world_model(
        encoder,
        decoder,
        rssm_prior,
        rssm_posterior,
        reward_module,
        observation_in_key=observation_in_key,
        observation_out_key=observation_out_key,
        use_scan=cfg.networks.use_scan,
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
        observation_out_key=observation_out_key,
        test_env=test_env,
        use_decoder_in_env=use_decoder_in_env,
        state_dim=cfg.networks.state_dim,
        rssm_hidden_dim=cfg.networks.rssm_hidden_dim,
    )

    # def detach_state_and_belief(data):
    #     data.set("state", data.get("state").detach())
    #     data.set("belief", data.get("belief").detach())
    #     return data
    #
    # model_based_env = model_based_env.append_transform(detach_state_and_belief)
    check_env_specs(model_based_env)

    # Make actor
    actor_simulator, actor_realworld = _dreamer_make_actors(
        encoder=encoder,
        observation_in_key=observation_in_key,
        rssm_prior=rssm_prior,
        rssm_posterior=rssm_posterior,
        mlp_num_units=cfg.networks.hidden_dim,
        activation=get_activation(cfg.networks.activation),
        action_key=action_key,
        test_env=test_env,
    )
    # Exploration noise to be added to the actor_realworld
    actor_realworld = TensorDictSequential(
        actor_realworld,
        AdditiveGaussianModule(
            spec=test_env.action_spec,
            sigma_init=1.0,
            sigma_end=1.0,
            annealing_num_steps=1,
            mean=0.0,
            std=cfg.networks.exploration_noise,
            device=device,
        ),
    )

    # Make Critic
    value_model = _dreamer_make_value_model(
        hidden_dim=cfg.networks.hidden_dim,
        activation=cfg.networks.activation,
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

    if cfg.logger.video:
        model_based_env_eval = model_based_env.append_transform(DreamerDecoder())

        def float_to_int(data):
            reco_pixels_float = data.get("reco_pixels")
            reco_pixels = (reco_pixels_float * 255).floor()
            # assert (reco_pixels < 256).all() and (reco_pixels > 0).all(), (reco_pixels.min(), reco_pixels.max())
            reco_pixels = reco_pixels.to(torch.uint8)
            data.set("reco_pixels_float", reco_pixels_float)
            return data.set("reco_pixels", reco_pixels)

        model_based_env_eval.append_transform(float_to_int)
        model_based_env_eval.append_transform(
            VideoRecorder(
                logger=logger, tag="eval/simulated_rendering", in_keys=["reco_pixels"]
            )
        )

    else:
        model_based_env_eval = None
    return (
        world_model,
        model_based_env,
        model_based_env_eval,
        actor_simulator,
        value_model,
        actor_realworld,
    )


def make_collector(cfg, train_env_factory, actor_model_explore):
    """Make async multi-collector for parallel data collection.

    Args:
        cfg: Configuration object
        train_env_factory: A callable that creates a training environment
        actor_model_explore: The exploration policy

    Returns:
        MultiCollector in async mode with multiple worker processes
    """
    num_collectors = cfg.collector.num_collectors

    collector = MultiCollector(
        create_env_fn=[train_env_factory] * num_collectors,
        policy=actor_model_explore,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        init_random_frames=cfg.collector.init_random_frames,
        policy_device=_default_device(cfg.collector.device),
        storing_device="cpu",
        sync=False,  # Async mode for overlapping collection with training
        update_at_each_batch=True,
    )
    collector.set_seed(cfg.env.seed)

    return collector


def make_storage_transform(
    *,
    pixel_obs=True,
    grayscale=True,
    image_size,
):
    """Create transforms to be applied at extend-time (once per frame).

    These heavy transforms (ToTensorImage, GrayScale, Resize) are applied once
    when data is added to the buffer, rather than on every sample.
    """
    if not pixel_obs:
        return None

    storage_transforms = Compose(
        ExcludeTransform("pixels", ("next", "pixels"), inverse=True),
        ToTensorImage(
            in_keys=["pixels_int", ("next", "pixels_int")],
            out_keys=["pixels", ("next", "pixels")],
        ),
    )
    if grayscale:
        storage_transforms.append(GrayScale(in_keys=["pixels", ("next", "pixels")]))
    storage_transforms.append(
        Resize(image_size, image_size, in_keys=["pixels", ("next", "pixels")])
    )
    return storage_transforms


def make_replay_buffer(
    *,
    batch_size,
    batch_seq_len,
    buffer_size=1000000,
    buffer_scratch_dir=None,
    device=None,
    prefetch=8,
    pixel_obs=True,
    grayscale=True,
    image_size,
):
    """Create replay buffer with minimal sample-time transforms.

    Heavy image transforms are expected to be applied at extend-time using
    make_storage_transform(). Only DeviceCastTransform is applied at sample-time.

    Note: We don't compile the SliceSampler because:
    1. Sampler operations (index computation) happen on CPU and are already fast
    2. torch.compile with inductor has bugs with the sampler's vectorized int64 operations
    """
    with (
        tempfile.TemporaryDirectory()
        if buffer_scratch_dir is None
        else nullcontext(buffer_scratch_dir)
    ) as scratch_dir:
        # Sample-time transforms: only device transfer (fast)
        sample_transforms = Compose(
            DeviceCastTransform(device=device),
        )

        replay_buffer = TensorDictReplayBuffer(
            pin_memory=False,
            prefetch=prefetch,
            storage=LazyMemmapStorage(
                buffer_size,
                scratch_dir=scratch_dir,
                device="cpu",
                ndim=2,
            ),
            sampler=SliceSampler(
                slice_len=batch_seq_len,
                strict_length=False,
                traj_key=("collector", "traj_ids"),
                cache_values=True,
                # Don't compile the sampler - inductor has C++ codegen bugs for int64 ops
            ),
            transform=sample_transforms,
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
    value_model = ProbabilisticTensorDictSequential(
        TensorDictModule(
            value_model,
            in_keys=["state", "belief"],
            out_keys=["loc"],
        ),
        ProbabilisticTensorDictModule(
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
            spec=Composite(
                **{
                    "loc": Unbounded(
                        proof_environment.action_spec_unbatched.shape,
                        device=proof_environment.action_spec_unbatched.device,
                    ),
                    "scale": Unbounded(
                        proof_environment.action_spec_unbatched.shape,
                        device=proof_environment.action_spec_unbatched.device,
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
            spec=Composite(**{action_key: proof_environment.action_spec_unbatched}),
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
                spec=Composite(
                    **{
                        "loc": Unbounded(
                            proof_environment.action_spec_unbatched.shape,
                        ),
                        "scale": Unbounded(
                            proof_environment.action_spec_unbatched.shape,
                        ),
                    }
                ),
            ),
            SafeProbabilisticModule(
                in_keys=["loc", "scale"],
                out_keys=[action_key],
                default_interaction_type=InteractionType.DETERMINISTIC,
                distribution_class=TanhNormal,
                distribution_kwargs={"tanh_loc": True},
                spec=proof_environment.full_action_spec_unbatched.to("cpu"),
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
            in_keys=["state", "belief"],
            out_keys=[observation_out_key],
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
    observation_in_key: NestedKey = "pixels",
    observation_out_key: NestedKey = "reco_pixels",
    use_scan: bool = False,
):
    # World Model and reward model
    # Note: in_keys uses dict form with out_to_in_map=True to map function args to tensordict keys.
    # {"noise": "prior_noise"} means: read "prior_noise" from tensordict, pass as `noise` kwarg.
    # With strict=False (default), missing noise keys pass None to the module.
    rssm_rollout = RSSMRollout(
        TensorDictModule(
            rssm_prior,
            in_keys={"state": "state", "belief": "belief", "action": "action", "noise": "prior_noise"},
            out_keys=[
                ("next", "prior_mean"),
                ("next", "prior_std"),
                "_",
                ("next", "belief"),
            ],
            out_to_in_map=True,
        ),
        TensorDictModule(
            rssm_posterior,
            in_keys={"belief": ("next", "belief"), "obs_embedding": ("next", "encoded_latents"), "noise": "posterior_noise"},
            out_keys=[
                ("next", "posterior_mean"),
                ("next", "posterior_std"),
                ("next", "state"),
            ],
            out_to_in_map=True,
        ),
        use_scan=use_scan,
    )
    event_dim = 3 if observation_out_key == "reco_pixels" else 1  # 3 for RGB
    decoder = ProbabilisticTensorDictSequential(
        TensorDictModule(
            decoder,
            in_keys=[("next", "state"), ("next", "belief")],
            out_keys=["loc"],
        ),
        ProbabilisticTensorDictModule(
            in_keys=["loc"],
            out_keys=[("next", observation_out_key)],
            distribution_class=IndependentNormal,
            distribution_kwargs={"scale": 1.0, "event_dim": event_dim},
        ),
    )

    transition_model = TensorDictSequential(
        TensorDictModule(
            encoder,
            in_keys=[("next", observation_in_key)],
            out_keys=[("next", "encoded_latents")],
        ),
        rssm_rollout,
        decoder,
    )

    reward_model = ProbabilisticTensorDictSequential(
        TensorDictModule(
            reward_module,
            in_keys=[("next", "state"), ("next", "belief")],
            out_keys=[("next", "loc")],
        ),
        ProbabilisticTensorDictModule(
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


def _default_device(device=None):
    if device in ("", None):
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)
