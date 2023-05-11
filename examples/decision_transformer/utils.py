import torch.nn
import torch.optim
from tensordict.nn import TensorDictModule

from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.data.datasets.d4rl import D4RLExperienceReplay
from torchrl.data.replay_buffers import SamplerWithoutReplacement
from torchrl.envs import (
    CatFrames,
    Compose,
    DoubleToFloat,
    EnvCreator,
    ExcludeTransform,
    NoopResetEnv,
    ObservationNorm,
    ParallelEnv,
    Reward2GoTransform,
    RewardScaling,
    RewardSum,
    TargetReturn,
    TensorDictPrimer,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.envs.libs.dm_control import DMControlEnv
from torchrl.envs.utils import set_exploration_mode
from torchrl.modules import DTActor, ProbabilisticActor, TanhNormal
from torchrl.modules.tensordict_module import DecisionTransformerInferenceWrapper
from torchrl.objectives import OnlineDTLoss
from torchrl.record.loggers import generate_exp_name, get_logger
from torchrl.trainers.helpers.envs import LIBS


# ====================================================================
# Environment utils
# -----------------


def make_base_env(env_cfg):
    env_library = LIBS[env_cfg.env_library]
    env_name = env_cfg.env_name
    frame_skip = env_cfg.frame_skip

    env_kwargs = {
        "env_name": env_name,
        "frame_skip": frame_skip,
    }
    if env_library is DMControlEnv:
        env_task = env_cfg.env_task
        env_kwargs.update({"task_name": env_task})
    env = env_library(**env_kwargs)
    if env_cfg.noop > 1:
        env = TransformedEnv(env, NoopResetEnv(env_cfg.noop))
    return env


def make_transformed_env(base_env, env_cfg, train=False):
    transformed_env = TransformedEnv(base_env)
    if train:
        transformed_env.append_transform(
            TargetReturn(env_cfg.collect_target_return, out_keys=["return_to_go"])
        )
    else:
        transformed_env.append_transform(
            TargetReturn(env_cfg.eval_target_return, out_keys=["return_to_go"])
        )
    transformed_env.append_transform(
        RewardScaling(
            loc=0,
            scale=env_cfg.reward_scaling,
            in_keys="return_to_go",
            standard_normal=False,
        )
    )
    transformed_env.append_transform(
        RewardScaling(
            loc=0, scale=env_cfg.reward_scaling, in_keys="reward", standard_normal=False
        )
    )
    transformed_env.append_transform(TensorDictPrimer(action=base_env.action_spec))

    transformed_env.append_transform(
        DoubleToFloat(
            in_keys=["observation"],
            in_keys_inv=[],
        )
    )
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
    if train:
        transformed_env.append_transform(RewardSum())

    return transformed_env


def make_parallel_env(env_cfg, state_dict, train=False):
    if train:
        num_envs = env_cfg.num_train_envs
    else:
        num_envs = env_cfg.num_eval_envs
    env = make_transformed_env(
        ParallelEnv(num_envs, EnvCreator(lambda: make_base_env(env_cfg))),
        env_cfg,
        train,
    )
    for t in env.transform:
        if isinstance(t, ObservationNorm):
            t.init_stats(3, cat_dim=1, reduce_dim=[0, 1])
    env.load_state_dict(state_dict)
    return env


def make_env(env_cfg, train=False):
    state_dict = get_stats(env_cfg)
    env = make_parallel_env(env_cfg, state_dict=state_dict, train=train)
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
    exclude_target_return = ExcludeTransform(
        "return_to_go",
        ("next", "return_to_go"),
        ("next", "action"),
        ("next", "observation"),
        "scale",
        "loc",
    )
    cat = CatFrames(in_keys=["action"], N=20, dim=-2, padding="zeros")
    transforms = Compose(
        exclude_target_return,
        cat,
    )
    collector_cfg = cfg.collector
    collector_class = SyncDataCollector
    collector = collector_class(
        make_env(cfg.env, train=True),
        policy,
        frames_per_batch=collector_cfg.frames_per_batch,
        total_frames=collector_cfg.total_frames,
        device=collector_cfg.collector_devices,
        max_frames_per_traj=collector_cfg.max_frames_per_traj,
        postproc=transforms,
    )
    return collector


def make_offline_replay_buffer(rb_cfg, reward_scaling):
    r2g = Reward2GoTransform(gamma=1.0, out_keys=["return_to_go"])
    reward_scale = RewardScaling(
        loc=0, scale=reward_scaling, in_keys="return_to_go", standard_normal=False
    )
    catframes = CatFrames(
        in_keys=["action", "observation", "return_to_go"],
        N=rb_cfg.stacked_frames,
        dim=-2,
        padding="zeros",
    )

    d2f = DoubleToFloat(
        in_keys=["observation", ("next", "observation")],
        in_keys_inv=[],
    )
    exclude = ExcludeTransform(
        "next_observations",
        "timeout",
        "terminal",
        "info",
        ("next", "timeout"),
        ("next", "terminal"),
        ("next", "observation"),
        ("next", "info"),
    )

    transforms = Compose(
        d2f,
        r2g,
        reward_scale,
        catframes,
        exclude,
    )
    data = D4RLExperienceReplay(
        rb_cfg.dataset,
        split_trajs=False,
        batch_size=rb_cfg.batch_size,
        sampler=SamplerWithoutReplacement(drop_last=False),
        transform=transforms,
    )
    # TODO: add obsnorm here

    return data


def make_online_replay_buffer(offline_buffer, rb_cfg, reward_scaling=0.001):
    offline_data = offline_buffer.sample(100000)
    offline_data.del_("return_to_go")
    offline_data.del_("index")  # delete

    r2g = Reward2GoTransform(gamma=1.0, out_keys=["return_to_go"])
    reward_scale = RewardScaling(
        loc=0, scale=reward_scaling, in_keys="return_to_go", standard_normal=False
    )
    catframes = CatFrames(
        in_keys=["return_to_go"], N=rb_cfg.stacked_frames, dim=-2, padding="zeros"
    )
    transforms = Compose(
        r2g,
        reward_scale,
        catframes,
    )
    storage = LazyMemmapStorage(
        rb_cfg.capacity, rb_cfg.buffer_scratch_dir, device=rb_cfg.device
    )

    replay_buffer = TensorDictReplayBuffer(
        pin_memory=False,
        prefetch=rb_cfg.prefetch,
        transform=transforms,
        storage=storage,
        batch_size=rb_cfg.batch_size,
    )
    # init buffer with offline data
    # replay_buffer.extend(offline_data.clone().detach().to_tensordict())

    return replay_buffer


# ====================================================================
# Model
# -----


def make_decision_transformer_model(cfg):
    env_cfg = cfg.env
    proof_environment = make_transformed_env(make_base_env(env_cfg), env_cfg)

    action_spec = proof_environment.action_spec
    for key, value in proof_environment.observation_spec.items():
        if key == "observation":
            state_dim = value.shape[-1]
    in_keys = [
        "observation",
        "action",
        "return_to_go",
    ]

    actor_net = DTActor(
        state_dim=state_dim,
        action_dim=action_spec.shape[-1],
        transformer_config=cfg.transformer,
    )

    actor_module = TensorDictModule(
        actor_net,
        in_keys=in_keys,
        out_keys=[
            "loc",
            "scale",
        ],
    )
    dist_class = TanhNormal
    dist_kwargs = {
        "min": -1.0,
        "max": 1.0,
        "tanh_loc": False,
    }

    actor = ProbabilisticActor(
        spec=action_spec,
        in_keys=["loc", "scale"],
        out_keys=["action", "log_prob"],
        module=actor_module,
        distribution_class=dist_class,
        distribution_kwargs=dist_kwargs,
        default_interaction_mode="random",
        cache_dist=False,
        return_log_prob=False,
    )

    # init the lazy layers
    with torch.no_grad(), set_exploration_mode("random"):
        td = proof_environment.rollout(max_steps=100)
        td["action"] = td["next", "action"]
        actor(td)

    inference_actor = DecisionTransformerInferenceWrapper(
        actor,
    )
    return inference_actor, actor


# ====================================================================
# Online Decision Transformer Loss
# ---------


def make_loss(loss_cfg, actor_network):
    loss = OnlineDTLoss(
        actor_network,
        loss_cfg.alpha_init,
    )
    return loss


def make_dt_optimizer(optim_cfg, actor_network, loss):
    # Should be Lambda Optimizer
    dt_optimizer = torch.optim.Adam(
        actor_network.parameters(),
        lr=optim_cfg.lr,
        weight_decay=optim_cfg.weight_decay,
        eps=1.0e-8,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        dt_optimizer, lambda steps: min((steps + 1) / optim_cfg.warmup_steps, 1)
    )

    log_temp_optimizer = torch.optim.Adam(
        [loss.log_alpha],
        lr=1e-4,
        betas=[0.9, 0.999],
    )

    return dt_optimizer, log_temp_optimizer, scheduler


# ====================================================================
# Logging and recording
# ---------------------


def make_logger(logger_cfg):
    exp_name = generate_exp_name("OnlineDecisionTransformer", logger_cfg.exp_name)
    logger_cfg.exp_name = exp_name
    logger = get_logger(logger_cfg.backend, logger_name="oDT", experiment_name=exp_name)
    return logger
