import torch.nn
import torch.optim
from tensordict.nn import TensorDictModule

from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.data.datasets.d4rl import D4RLExperienceReplay
from torchrl.data.replay_buffers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.samplers import RandomSampler
from torchrl.envs import (
    CatFrames,
    Compose,
    EnvCreator,
    ExcludeTransform,
    NoopResetEnv,
    ObservationNorm,
    ParallelEnv,
    Reward2GoTransform,
    StepCounter,
    TargetReturn,
    TensorDictPrimer,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.envs.libs.dm_control import DMControlEnv
from torchrl.envs.utils import set_exploration_mode
from torchrl.modules import DTActor, ProbabilisticActor, TanhNormal
from torchrl.objectives import SoftUpdate, TD3Loss
from torchrl.record.loggers import generate_exp_name, get_logger
from torchrl.trainers.helpers.envs import LIBS


DEFAULT_REWARD_SCALING = {
    "Hopper-v1": 5,
    "Walker2d-v1": 5,
    "HalfCheetah-v1": 5,
    "cheetah": 5,
    "Ant-v2": 5,
    "Humanoid-v2": 20,
    "humanoid": 100,
}

# ====================================================================
# Environment utils
# -----------------


def make_base_env(env_cfg, from_pixels=None):
    env_library = LIBS[env_cfg.env_library]
    env_name = env_cfg.env_name
    frame_skip = env_cfg.frame_skip
    if from_pixels is None:
        from_pixels = env_cfg.from_pixels

    env_kwargs = {
        "env_name": env_name,
        "frame_skip": frame_skip,
        "from_pixels": from_pixels,  # for rendering
        "pixels_only": False,
    }
    if env_library is DMControlEnv:
        env_task = env_cfg.env_task
        env_kwargs.update({"task_name": env_task})
    env = env_library(**env_kwargs)
    if env_cfg.noop > 1:
        env = TransformedEnv(env, NoopResetEnv(env_cfg.noop))
    return env


def make_transformed_env(base_env, env_cfg):
    return make_transformed_env_states(base_env, env_cfg)


def make_transformed_env_states(base_env, env_cfg):
    transformed_env = TransformedEnv(base_env)

    transformed_env.append_transform(StepCounter())
    # Only needed if ordering True -> Default is False
    # transformed_env.append_transform(
    #     RenameTransform(["step_count"], ["timesteps"], create_copy=True)
    # )
    transformed_env.append_transform(
        TargetReturn(
            200 * 0.01, out_keys=["return_to_go"]
        )  # WATCH OUT FOR THE SCALING!
    )
    # transformed_env.append_transform(SCALE)
    transformed_env.append_transform(TensorDictPrimer(action=base_env.action_spec))
    # transformed_env.append_transform(TensorDictPrimer(padding_mask=env.action_spec))

    transformed_env.append_transform(UnsqueezeTransform(-2, in_keys=["observation"]))
    transformed_env.append_transform(
        CatFrames(in_keys=["observation"], N=env_cfg.stacked_frames, dim=-2)
    )

    transformed_env.append_transform(UnsqueezeTransform(-2, in_keys=["action"]))
    transformed_env.append_transform(
        CatFrames(in_keys=["action"], N=env_cfg.stacked_frames, dim=-2)
    )

    transformed_env.append_transform(UnsqueezeTransform(-2, in_keys=["return_to_go"]))
    transformed_env.append_transform(
        CatFrames(in_keys=["return_to_go"], N=env_cfg.stacked_frames, dim=-2)
    )
    # Only needed if ordering True -> Default is False
    # transformed_env.append_transform(UnsqueezeTransform(-2, in_keys=["timesteps"]))
    # transformed_env.append_transform(
    #     CatFrames(in_keys=["timesteps"], N=env_cfg.stacked_frames, dim=-2)
    # )

    return transformed_env


def make_parallel_env(env_cfg, state_dict):
    num_envs = env_cfg.num_envs
    env = make_transformed_env(
        ParallelEnv(num_envs, EnvCreator(lambda: make_base_env(env_cfg))), env_cfg
    )
    for t in env.transform:
        if isinstance(t, ObservationNorm):
            t.init_stats(3, cat_dim=1, reduce_dim=[0, 1])
    env.load_state_dict(state_dict)
    return env


def make_test_env(env_cfg):
    env_cfg.num_envs = 1
    state_dict = get_stats(env_cfg)
    env = make_parallel_env(env_cfg, state_dict=state_dict)
    return env


def get_stats(env_cfg):
    env = make_transformed_env(make_base_env(env_cfg), env_cfg)
    init_stats(env, env_cfg.n_samples_stats)
    return env.state_dict()


def init_stats(env, n_samples_stats):
    for t in env.transform:
        if isinstance(t, ObservationNorm):
            t.init_stats(n_samples_stats)


# ====================================================================
# Collector and replay buffer
# ---------------------------


def make_collector(cfg, policy):
    env_cfg = cfg.env
    collector_cfg = cfg.collector
    collector_class = SyncDataCollector
    state_dict = get_stats(env_cfg)
    # to exclude inference target returns
    exclude = ExcludeTransform("return_to_go")  # next return to go
    collector = collector_class(
        make_parallel_env(env_cfg, state_dict=state_dict),
        policy,
        frames_per_batch=collector_cfg.frames_per_batch,
        total_frames=collector_cfg.total_frames,
        device=collector_cfg.collector_devices,
        max_frames_per_traj=collector_cfg.max_frames_per_traj,
        postproc=exclude,
    )
    return collector


def make_replay_buffer(rb_cfg):
    r2g = Reward2GoTransform(gamma=1.0, out_keys=["return_to_go"])
    transforms = [r2g]
    sampler = RandomSampler()
    return TensorDictReplayBuffer(
        storage=LazyMemmapStorage(rb_cfg.capacity),
        sampler=sampler,
        transform=Compose(*transforms),
    )


def make_offline_replay_buffer(rb_cfg):
    r2g = Reward2GoTransform(gamma=1.0, out_keys=["return_to_go"])
    data = D4RLExperienceReplay(
        rb_cfg.dataset,
        split_trajs=False,
        batch_size=rb_cfg.batch_size,
        sampler=SamplerWithoutReplacement(drop_last=False),
        transform=r2g,
    )
    # data.append_transform(
    #     Reward2GoTransform(gamma=1.0, out_keys=["return_to_go"])
    # )
    # data.append_transform(

    # )
    # data.append_transform(

    # )

    return data


# ====================================================================
# Model
# -----
#
# We give one version of the model for learning from pixels, and one for state.
# TorchRL comes in handy at this point, as the high-level interactions with
# these models is unchanged, regardless of the modality.
#


def make_decision_transformer_model(cfg):
    env_cfg = cfg.env
    # model_cfg = cfg.model
    proof_environment = make_transformed_env(make_base_env(env_cfg), env_cfg)
    # we must initialize the observation norm transform
    init_stats(proof_environment, n_samples_stats=3, from_pixels=env_cfg.from_pixels)

    action_spec = proof_environment.action_spec

    in_keys = [
        "observation",
        "action",
        "return_to_go",
        # "timesteps",
    ]  # return_to_go, timesteps

    actor_net = DTActor(action_dim=1)

    dist_class = TanhNormal
    dist_kwargs = {
        "min": -1.0,
        "max": 1.0,
        "tanh_loc": False,
    }

    actor_module = TensorDictModule(
        actor_net, in_keys=in_keys, out_keys=["loc", "scale"]  # , "hidden_state"],
    )
    actor = ProbabilisticActor(
        spec=action_spec,
        in_keys=["loc", "scale"],  # , "hidden_state"],
        out_keys=["action", "log_prob"],  # , "hidden_state"],
        module=actor_module,
        distribution_class=dist_class,
        distribution_kwargs=dist_kwargs,
        default_interaction_mode="random",
        cache_dist=True,
        return_log_prob=False,
    )

    # init the lazy layers
    with torch.no_grad(), set_exploration_mode("random"):
        td = proof_environment.rollout(max_steps=1000)
        print(td)
        actor(td)

    return actor


# ====================================================================
# Decision Transformer Loss
# ---------


def make_loss(loss_cfg, actor_network):
    loss = TD3Loss(
        actor_network,
        gamma=loss_cfg.gamma,
        loss_function=loss_cfg.loss_function,
    )
    target_net_updater = SoftUpdate(loss, 1 - loss_cfg.tau)
    target_net_updater.init_()
    return loss, target_net_updater


def make_dt_optimizer(optim_cfg, actor_network):
    optimizer = torch.optim.Adam(
        actor_network.parameters(),
        lr=optim_cfg.lr,
        weight_decay=optim_cfg.weight_decay,
    )
    return optimizer


# ====================================================================
# Logging and recording
# ---------------------


def make_logger(logger_cfg):
    exp_name = generate_exp_name("TD3", logger_cfg.exp_name)
    logger_cfg.exp_name = exp_name
    logger = get_logger(logger_cfg.backend, logger_name="td3", experiment_name=exp_name)
    return logger
