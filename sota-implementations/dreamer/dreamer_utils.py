# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import functools
import tempfile
from contextlib import nullcontext

import torch
import torch.nn as nn
from tensordict import NestedKey, TensorDictBase
from tensordict.nn import (
    InteractionType,
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
    TensorDictModule,
    TensorDictSequential,
)
from torchrl import logger as torchrl_logger
from torchrl._utils import set_profiling_enabled
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
    DMControlEnv,
    DoubleToFloat,
    DreamerDecoder,
    DreamerEnv,
    EnvCreator,
    ExcludeTransform,
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
    Transform,
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


def allocate_collector_devices(
    num_collectors: int, training_device: torch.device
) -> list[torch.device]:
    """Allocate CUDA devices for collectors, reserving cuda:0 for training.

    Device allocation strategy:
    - Training always uses cuda:0
    - Collectors use cuda:1, cuda:2, ..., cuda:N-1 if available
    - If only 1 CUDA device, colocate training and inference on cuda:0
    - If num_collectors >= num_cuda_devices, raise an exception

    Args:
        num_collectors: Number of collector workers requested
        training_device: The device used for training (determines if CUDA is used)

    Returns:
        List of devices for each collector worker

    Raises:
        ValueError: If num_collectors >= num_cuda_devices (no device left for training)
    """
    if training_device.type != "cuda":
        # CPU training: all collectors on CPU
        return [torch.device("cpu")] * num_collectors

    num_cuda_devices = torch.cuda.device_count()

    if num_cuda_devices == 0:
        # No CUDA devices available, fall back to CPU
        return [torch.device("cpu")] * num_collectors

    if num_cuda_devices == 1:
        # Single GPU: colocate training and inference
        torchrl_logger.info(
            f"Single CUDA device available. Colocating {num_collectors} collectors "
            "with training on cuda:0"
        )
        return [torch.device("cuda:0")] * num_collectors

    # Multiple GPUs available
    # Reserve cuda:0 for training, use cuda:1..cuda:N-1 for inference
    inference_devices = num_cuda_devices - 1  # Devices available for collectors

    if num_collectors > inference_devices:
        raise ValueError(
            f"Requested {num_collectors} collectors but only {inference_devices} "
            f"CUDA devices available for inference (cuda:1 to cuda:{num_cuda_devices - 1}). "
            f"cuda:0 is reserved for training. Either reduce num_collectors to "
            f"{inference_devices} or add more GPUs."
        )

    # Distribute collectors across available inference devices (round-robin)
    collector_devices = []
    for i in range(num_collectors):
        device_idx = (i % inference_devices) + 1  # +1 to skip cuda:0
        collector_devices.append(torch.device(f"cuda:{device_idx}"))

    device_str = ", ".join(str(d) for d in collector_devices)
    torchrl_logger.info(
        f"Allocated {num_collectors} collectors to devices: [{device_str}]. "
        f"Training on cuda:0."
    )

    return collector_devices


class DreamerProfiler:
    """Helper class for PyTorch profiling in Dreamer training.

    Encapsulates profiler setup, stepping, and trace export logic.

    Args:
        cfg: Hydra config with profiling section.
        device: Training device (used to determine CUDA profiling).
        pbar: Progress bar to update total when profiling.
    """

    def __init__(self, cfg, device, pbar=None, *, compile_warmup: int = 0):
        self.enabled = cfg.profiling.enabled
        self.cfg = cfg
        self.total_optim_steps = 0
        self._profiler = None
        self._stopped = False
        self._compile_warmup = compile_warmup

        # Enable detailed profiling instrumentation in torchrl when profiling
        set_profiling_enabled(self.enabled)

        if not self.enabled:
            return

        # Override total_optim_steps for profiling runs
        torchrl_logger.info(
            f"Profiling enabled: running {cfg.profiling.total_optim_steps} optim steps "
            f"(skip_first={cfg.profiling.skip_first}, warmup={cfg.profiling.warmup_steps}, "
            f"active={cfg.profiling.active_steps})"
        )
        if pbar is not None:
            pbar.total = cfg.profiling.total_optim_steps

        # Setup profiler schedule
        # - skip_first: steps to skip entirely (no profiling)
        # - warmup: steps to warm up profiler (data discarded)
        # - active: steps to actually profile (data kept)
        #
        # When torch.compile is enabled via compile_with_warmup, the first `compile_warmup`
        # calls run eagerly and the *next* call typically triggers compilation. Profiling
        # these steps is usually undesirable because it captures compilation overhead and
        # non-representative eager execution.
        #
        # Therefore we automatically extend skip_first by (compile_warmup + 1) optim steps.
        extra_skip = self._compile_warmup + 1 if self._compile_warmup else 0
        skip_first = cfg.profiling.skip_first + extra_skip
        profiler_schedule = torch.profiler.schedule(
            skip_first=skip_first,
            wait=0,
            warmup=cfg.profiling.warmup_steps,
            active=cfg.profiling.active_steps,
            repeat=1,
        )

        # Determine profiler activities
        activities = [torch.profiler.ProfilerActivity.CPU]
        if cfg.profiling.profile_cuda and device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        self._profiler = torch.profiler.profile(
            activities=activities,
            schedule=profiler_schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_logs")
            if not cfg.profiling.trace_file
            else None,
            record_shapes=cfg.profiling.record_shapes,
            profile_memory=cfg.profiling.profile_memory,
            with_stack=cfg.profiling.with_stack,
            with_flops=cfg.profiling.with_flops,
        )
        self._profiler.start()

    def step(self) -> bool:
        """Step the profiler and check if profiling is complete.

        Returns:
            True if profiling is complete and training should exit.
        """
        if not self.enabled or self._stopped:
            return False

        self.total_optim_steps += 1
        self._profiler.step()

        # Check if we should stop profiling
        extra_skip = self._compile_warmup + 1 if self._compile_warmup else 0
        target_steps = (
            self.cfg.profiling.skip_first
            + extra_skip
            + self.cfg.profiling.warmup_steps
            + self.cfg.profiling.active_steps
        )
        if self.total_optim_steps >= target_steps:
            torchrl_logger.info(
                f"Profiling complete after {self.total_optim_steps} optim steps. "
                f"Exporting trace to {self.cfg.profiling.trace_file}"
            )
            self._profiler.stop()
            self._stopped = True
            # Export trace if trace_file is set
            if self.cfg.profiling.trace_file:
                self._profiler.export_chrome_trace(self.cfg.profiling.trace_file)
            return True

        return False

    def should_exit(self) -> bool:
        """Check if training loop should exit due to profiling completion."""
        if not self.enabled:
            return False
        extra_skip = self._compile_warmup + 1 if self._compile_warmup else 0
        target_steps = (
            self.cfg.profiling.skip_first
            + extra_skip
            + self.cfg.profiling.warmup_steps
            + self.cfg.profiling.active_steps
        )
        return self.total_optim_steps >= target_steps


class GPUImageTransform(Transform):
    """Composite transform that processes images on GPU for faster execution.

    This transform:
    1. Moves pixels_int to GPU
    2. Runs ToTensorImage (permute + divide by 255)
    3. Optionally runs GrayScale
    4. Runs Resize
    5. Keeps output on GPU for fast policy inference

    This avoids device mismatch issues by not using DeviceCastTransform on the
    full tensordict - only the pixel processing happens on GPU.
    """

    def __init__(
        self,
        device: torch.device,
        image_size: int,
        grayscale: bool = False,
        in_key: str = "pixels_int",
        out_key: str = "pixels",
    ):
        super().__init__(in_keys=[in_key], out_keys=[out_key])
        self.device = device
        self.image_size = image_size
        self.grayscale = grayscale
        self.in_key = in_key
        self.out_key = out_key

    def _apply_transform(self, pixels_int: torch.Tensor) -> torch.Tensor:
        # Move to GPU
        pixels = pixels_int.to(self.device)
        # ToTensorImage: permute W x H x C -> C x W x H and normalize
        pixels = pixels.permute(*list(range(pixels.ndimension() - 3)), -1, -3, -2)
        pixels = pixels.float().div(255)
        # GrayScale
        if self.grayscale:
            pixels = pixels.mean(dim=-3, keepdim=True)
        # Resize using interpolate
        if pixels.shape[-2:] != (self.image_size, self.image_size):
            # Add batch dim if needed for interpolate
            needs_squeeze = pixels.ndim == 3
            if needs_squeeze:
                pixels = pixels.unsqueeze(0)
            pixels = torch.nn.functional.interpolate(
                pixels,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            if needs_squeeze:
                pixels = pixels.squeeze(0)
        return pixels

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return self._call(tensordict_reset)

    def transform_observation_spec(self, observation_spec):
        # Update the spec for the output key
        # Note: Keep spec on CPU to match other specs in Composite
        # The actual transform will put data on GPU, but spec device must be uniform
        from torchrl.data import Unbounded

        in_spec = observation_spec[self.in_key]
        # Output shape: (C, H, W) where C=1 if grayscale else 3
        out_channels = 1 if self.grayscale else 3
        out_shape = (
            *in_spec.shape[:-3],
            out_channels,
            self.image_size,
            self.image_size,
        )
        # Use in_spec.device to maintain device consistency in Composite
        out_spec = Unbounded(
            shape=out_shape, dtype=torch.float32, device=in_spec.device
        )
        observation_spec[self.out_key] = out_spec
        return observation_spec


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
        # Gym doesn't support native frame_skip, apply transform inside worker
        if cfg.env.frame_skip > 1:
            env = TransformedEnv(env, FrameSkipTransform(cfg.env.frame_skip))
    elif lib == "dm_control":
        env = DMControlEnv(
            cfg.env.name,
            cfg.env.task,
            from_pixels=cfg.env.from_pixels or from_pixels,
            pixels_only=cfg.env.from_pixels,
            device=device,
            frame_skip=cfg.env.frame_skip,  # Native frame skip inside worker
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


def transform_env(cfg, env, device=None):
    """Apply transforms to environment.

    Args:
        cfg: Config object
        env: The environment to transform
        device: If specified and is a CUDA device, use GPU-accelerated image
            processing which is ~50-100x faster than CPU.
    """
    if not isinstance(env, TransformedEnv):
        env = TransformedEnv(env)
    if cfg.env.from_pixels:
        # Rename original pixels for processing
        env.append_transform(
            RenameTransform(in_keys=["pixels"], out_keys=["pixels_int"])
        )

        # Use GPU-accelerated image processing if device is CUDA
        if device is not None and str(device).startswith("cuda"):
            env.append_transform(
                GPUImageTransform(
                    device=device,
                    image_size=cfg.env.image_size,
                    grayscale=cfg.env.grayscale,
                    in_key="pixels_int",
                    out_key="pixels",
                )
            )
        else:
            # CPU fallback: use standard transforms
            env.append_transform(
                ToTensorImage(
                    from_int=True, in_keys=["pixels_int"], out_keys=["pixels"]
                )
            )
            if cfg.env.grayscale:
                env.append_transform(GrayScale())
            env.append_transform(Resize(cfg.env.image_size, cfg.env.image_size))

    env.append_transform(DoubleToFloat())
    env.append_transform(RewardSum())
    # Note: FrameSkipTransform is now applied inside workers (in _make_env) to avoid
    # extra IPC round-trips. DMControl uses native frame_skip, Gym uses the transform.
    env.append_transform(StepCounter(cfg.env.horizon))

    return env


def make_environments(cfg, parallel_envs=1, logger=None):
    """Make environments for training and evaluation.

    Returns:
        train_env_factory: A callable that creates a training environment (for MultiCollector)
        eval_env: The evaluation environment instance
    """

    def train_env_factory():
        """Factory function for creating training environments.

        Note: This factory runs inside collector worker processes. We use
        CUDA if available for GPU-accelerated image transforms (ToTensorImage,
        Resize) which are ~50-100x faster than CPU. The cfg.env.device setting
        is ignored in favor of auto-detecting CUDA availability.
        """
        # Use CUDA for transforms if available, regardless of cfg.env.device
        # This is critical: image transforms (Resize, ToTensorImage) are ~50-100x
        # faster on GPU. DMControl/Gym render on CPU, but we move to GPU for transforms.
        transform_device = _default_device(None)  # Returns CUDA if available
        # Base env still uses cfg.env.device for compatibility
        env_device = _default_device(cfg.env.device)
        func = functools.partial(_make_env, cfg=cfg, device=env_device)
        train_env = ParallelEnv(
            parallel_envs,
            EnvCreator(func),
            serial_for_single=True,
        )
        # Pass transform_device to enable GPU-accelerated image transforms
        train_env = transform_env(cfg, train_env, device=transform_device)
        train_env.set_seed(cfg.env.seed)
        return train_env

    # Create eval env directly (not a factory)
    # Use CUDA for transforms if available, regardless of cfg.env.device
    transform_device = _default_device(None)  # Returns CUDA if available
    env_device = _default_device(cfg.env.device)
    func = functools.partial(
        _make_env,
        cfg=cfg,
        device=env_device,
        from_pixels=cfg.logger.video,
    )
    eval_env = ParallelEnv(
        1,
        EnvCreator(func),
        serial_for_single=True,
    )
    # Pass transform_device to enable GPU-accelerated image transforms
    eval_env = transform_env(cfg, eval_env, device=transform_device)
    eval_env.set_seed(cfg.env.seed + 1)
    if cfg.logger.video:
        eval_env.insert_transform(
            0,
            VideoRecorder(
                logger,
                tag="eval/video",
                in_keys=["pixels"],
                skip=cfg.logger.video_skip,
            ),
        )

    # Check specs on a temporary train env
    temp_train_env = train_env_factory()
    check_env_specs(temp_train_env)
    temp_train_env.close()
    del temp_train_env

    check_env_specs(eval_env)
    return train_env_factory, eval_env


def dump_video(module, step: int | None = None):
    """Dump video from VideoRecorder transforms.

    Args:
        module: The transform module to check.
        step: Optional step to log the video at. If not provided,
            the VideoRecorder uses its internal counter.
    """
    if isinstance(module, VideoRecorder):
        module.dump(step=step)


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
        obs_embed_dim = _compute_encoder_output_size(
            image_size, channels=32, num_layers=4
        )

        encoder = ObsEncoder(in_channels=in_channels, device=device)
        decoder = ObsDecoder(latent_dim=state_dim + rssm_hidden_dim, device=device)

        observation_in_key = "pixels"
        observation_out_key = "reco_pixels"
    else:
        obs_embed_dim = 1024  # MLP output size
        encoder = MLP(
            out_features=obs_embed_dim,
            depth=2,
            num_cells=cfg.networks.hidden_dim,
            activation_class=get_activation(cfg.networks.activation),
            device=device,
        )
        decoder = MLP(
            out_features=test_env.observation_spec["observation"].shape[-1],
            depth=2,
            num_cells=cfg.networks.hidden_dim,
            activation_class=get_activation(cfg.networks.activation),
            device=device,
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
        device=device,
    )
    rssm_posterior = RSSMPosterior(
        hidden_dim=rssm_hidden_dim,
        state_dim=state_dim,
        rnn_hidden_dim=rssm_hidden_dim,
        obs_embed_dim=obs_embed_dim,
        device=device,
    )

    # When use_scan=True or rssm_rollout.compile=True, replace C++ GRU with Python-based GRU
    # for torch.compile compatibility. The C++ GRU (cuBLAS) cannot be traced by torch.compile.
    if cfg.networks.use_scan or cfg.networks.rssm_rollout.compile:
        from torchrl.modules.tensordict_module.rnn import GRUCell as PythonGRUCell

        old_rnn = rssm_prior.rnn
        python_rnn = PythonGRUCell(
            old_rnn.input_size, old_rnn.hidden_size, device=device
        )
        python_rnn.load_state_dict(old_rnn.state_dict())
        rssm_prior.rnn = python_rnn
        torchrl_logger.info(
            "Switched RSSMPrior to Python-based GRU for torch.compile compatibility"
        )
    # Make reward module
    reward_module = MLP(
        out_features=1,
        depth=2,
        num_cells=cfg.networks.hidden_dim,
        activation_class=get_activation(cfg.networks.activation),
        device=device,
    )

    # Make combined world model (modules already on device)
    world_model = _dreamer_make_world_model(
        encoder,
        decoder,
        rssm_prior,
        rssm_posterior,
        reward_module,
        observation_in_key=observation_in_key,
        observation_out_key=observation_out_key,
        use_scan=cfg.networks.use_scan,
        rssm_rollout_compile=cfg.networks.rssm_rollout.compile,
        rssm_rollout_compile_backend=cfg.networks.rssm_rollout.compile_backend,
        rssm_rollout_compile_mode=cfg.networks.rssm_rollout.compile_mode,
    )

    # Initialize world model (already on device)
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        tensordict = (
            test_env.rollout(5, auto_cast_to_device=True).unsqueeze(-1).to(device)
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

    # Make actor (modules already on device)
    actor_simulator, actor_realworld = _dreamer_make_actors(
        encoder=encoder,
        observation_in_key=observation_in_key,
        rssm_prior=rssm_prior,
        rssm_posterior=rssm_posterior,
        mlp_num_units=cfg.networks.hidden_dim,
        activation=get_activation(cfg.networks.activation),
        action_key=action_key,
        test_env=test_env,
        device=device,
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

    # Make Critic (on device)
    value_model = _dreamer_make_value_model(
        hidden_dim=cfg.networks.hidden_dim,
        activation=cfg.networks.activation,
        value_key=value_key,
        device=device,
    )

    # Move model_based_env to device (it contains references to modules already on device)
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
                logger=logger,
                tag="eval/simulated_video",
                in_keys=["reco_pixels"],
                skip=cfg.logger.video_skip,
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


def make_collector(
    cfg,
    train_env_factory,
    actor_model_explore,
    training_device: torch.device,
    replay_buffer=None,
    storage_transform=None,
    track_policy_version=False,
):
    """Make async multi-collector for parallel data collection.

    Args:
        cfg: Configuration object
        train_env_factory: A callable that creates a training environment
        actor_model_explore: The exploration policy
        training_device: Device used for training (used to allocate collector devices)
        replay_buffer: Optional replay buffer for true async collection with start()
        storage_transform: Optional transform to apply before storing in buffer
        track_policy_version: If True, track policy version using integer versioning.
            Can also be a PolicyVersion instance for custom versioning.

    Returns:
        MultiCollector in async mode with multiple worker processes

    Device allocation:
        - If training on CUDA with multiple GPUs: collectors use cuda:1, cuda:2, etc.
        - If training on CUDA with single GPU: collectors colocate on cuda:0
        - If training on CPU: collectors use CPU
    """
    num_collectors = cfg.collector.num_collectors
    init_random_frames = (
        cfg.collector.init_random_frames
        if not cfg.profiling.enabled
        else cfg.profiling.collector.init_random_frames_override
    )

    # Allocate devices for collectors (reserves cuda:0 for training if multi-GPU)
    collector_devices = allocate_collector_devices(num_collectors, training_device)

    collector = MultiCollector(
        create_env_fn=[train_env_factory] * num_collectors,
        policy=actor_model_explore,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=-1,  # Run indefinitely until async_shutdown() is called
        init_random_frames=init_random_frames,
        policy_device=collector_devices,
        env_device=collector_devices,  # Match env output device to policy device for CUDA transforms
        storing_device="cpu",
        sync=False,  # Async mode for overlapping collection with training
        update_at_each_batch=False,  # We manually call update_policy_weights_() in training loop
        replay_buffer=replay_buffer,
        postproc=storage_transform,
        track_policy_version=track_policy_version,
        # Skip fake data initialization - storage handles coordination
        local_init_rb=True,
    )
    collector.set_seed(cfg.env.seed)

    return collector


def make_storage_transform(
    *,
    pixel_obs=True,
    grayscale=True,
    image_size,
    gpu_transforms=False,
):
    """Create transforms to be applied at extend-time (once per frame).

    Args:
        pixel_obs: Whether observations are pixel-based.
        grayscale: Whether to convert to grayscale.
        image_size: Target image size.
        gpu_transforms: If True, skip heavy image transforms (ToTensorImage,
            GrayScale, Resize) since they're already applied by GPUImageTransform
            in the environment. Only ExcludeTransform is applied to filter keys.
    """
    if not pixel_obs:
        return None

    # When GPU transforms are enabled, GPUImageTransform already processes
    # pixels_int -> pixels with normalization, grayscale, and resize.
    # We only need to filter out the intermediate pixels_int key.
    if gpu_transforms:
        storage_transforms = Compose(
            # Just exclude pixels_int, keep everything else including processed pixels
            ExcludeTransform("pixels_int", ("next", "pixels_int")),
        )
        return storage_transforms

    # CPU fallback: apply heavy transforms at storage time
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


def _to_device(td, device):
    return td.to(device=device, non_blocking=True)


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
            functools.partial(_to_device, device=device),
        )

        replay_buffer = TensorDictReplayBuffer(
            prefetch=prefetch,
            storage=LazyMemmapStorage(
                buffer_size,
                scratch_dir=scratch_dir,
                device="cpu",
                ndim=2,
                shared_init=True,  # Allow remote processes to initialize storage
            ),
            sampler=SliceSampler(
                slice_len=batch_seq_len,
                strict_length=False,
                traj_key=("collector", "traj_ids"),
                cache_values=False,  # Disabled for async collection (cache not synced across processes)
                use_gpu=device.type == "cuda"
                if device is not None
                else False,  # Speed up trajectory computation on GPU
            ),
            transform=sample_transforms,
            batch_size=batch_size,
        )
        return replay_buffer


def _dreamer_make_value_model(
    hidden_dim: int = 400,
    activation: str = "elu",
    value_key: str = "state_value",
    device=None,
):
    value_model = MLP(
        out_features=1,
        depth=3,
        num_cells=hidden_dim,
        activation_class=get_activation(activation),
        device=device,
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
    device=None,
):
    actor_module = DreamerActor(
        out_features=test_env.action_spec.shape[-1],
        depth=3,
        num_cells=mlp_num_units,
        activation_class=activation,
        device=device,
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
    rssm_rollout_compile: bool = False,
    rssm_rollout_compile_backend: str = "inductor",
    rssm_rollout_compile_mode: str | None = "reduce-overhead",
):
    # World Model and reward model
    # Note: in_keys uses dict form with out_to_in_map=True to map function args to tensordict keys.
    # {"noise": "prior_noise"} means: read "prior_noise" from tensordict, pass as `noise` kwarg.
    # With strict=False (default), missing noise keys pass None to the module.
    rssm_rollout = RSSMRollout(
        TensorDictModule(
            rssm_prior,
            in_keys={
                "state": "state",
                "belief": "belief",
                "action": "action",
                "noise": "prior_noise",
            },
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
            in_keys={
                "belief": ("next", "belief"),
                "obs_embedding": ("next", "encoded_latents"),
                "noise": "posterior_noise",
            },
            out_keys=[
                ("next", "posterior_mean"),
                ("next", "posterior_std"),
                ("next", "state"),
            ],
            out_to_in_map=True,
        ),
        use_scan=use_scan,
        compile_step=rssm_rollout_compile,
        compile_backend=rssm_rollout_compile_backend,
        compile_mode=rssm_rollout_compile_mode,
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
