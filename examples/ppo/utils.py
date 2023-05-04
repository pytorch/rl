import torch.nn
import torch.optim
from tensordict.nn import TensorDictModule

from torchrl.collectors import SyncDataCollector
from torchrl.data import CompositeSpec, LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement

from torchrl.data.tensor_specs import DiscreteBox
from torchrl.envs import (
    CatFrames,
    CatTensors,
    DoubleToFloat,
    EnvCreator,
    ExplorationType,
    GrayScale,
    NoopResetEnv,
    ObservationNorm,
    ParallelEnv,
    Resize,
    RewardScaling,
    RewardSum,
    StepCounter,
    ToTensorImage,
    TransformedEnv,
)
from torchrl.envs.libs.dm_control import DMControlEnv
from torchrl.modules import (
    ActorValueOperator,
    ConvNet,
    MLP,
    NormalParamWrapper,
    OneHotCategorical,
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
)
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value.advantages import GAE
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
    env_kwargs = {
        "env_name": env_cfg.env_name,
        "frame_skip": env_cfg.frame_skip,
        "from_pixels": env_cfg.from_pixels
        if from_pixels is None
        else from_pixels,  # for rendering
        "pixels_only": False,
    }
    if env_library is DMControlEnv:
        env_task = env_cfg.env_task
        env_kwargs.update({"task_name": env_task})
    env = env_library(**env_kwargs)
    return env


def make_transformed_env(base_env, env_cfg):
    if env_cfg.noop > 1:
        base_env = TransformedEnv(env=base_env, transform=NoopResetEnv(env_cfg.noop))
    from_pixels = env_cfg.from_pixels
    if from_pixels:
        return make_transformed_env_pixels(base_env, env_cfg)
    else:
        return make_transformed_env_states(base_env, env_cfg)


def make_transformed_env_pixels(base_env, env_cfg):
    if not isinstance(env_cfg.reward_scaling, float):
        env_cfg.reward_scaling = DEFAULT_REWARD_SCALING.get(env_cfg.env_name, 5.0)

    env_library = LIBS[env_cfg.env_library]
    env = TransformedEnv(base_env)

    reward_scaling = env_cfg.reward_scaling
    env.append_transform(RewardScaling(0.0, reward_scaling))

    double_to_float_list = []
    double_to_float_inv_list = []

    env.append_transform(ToTensorImage())
    env.append_transform(GrayScale())
    env.append_transform(Resize(84, 84))
    env.append_transform(CatFrames(N=4, dim=-3))
    env.append_transform(RewardSum())
    env.append_transform(StepCounter())

    if env_library is DMControlEnv:
        double_to_float_list += [
            "reward",
        ]
        double_to_float_inv_list += ["action"]  # DMControl requires double-precision
    env.append_transform(
        DoubleToFloat(
            in_keys=double_to_float_list, in_keys_inv=double_to_float_inv_list
        )
    )
    return env


def make_transformed_env_states(base_env, env_cfg):
    if not isinstance(env_cfg.reward_scaling, float):
        env_cfg.reward_scaling = DEFAULT_REWARD_SCALING.get(env_cfg.env_name, 5.0)

    env_library = LIBS[env_cfg.env_library]
    env = TransformedEnv(base_env)

    reward_scaling = env_cfg.reward_scaling

    env.append_transform(RewardScaling(0.0, reward_scaling))

    double_to_float_list = []
    double_to_float_inv_list = []

    # we concatenate all the state vectors
    # even if there is a single tensor, it'll be renamed in "observation_vector"
    selected_keys = [
        key for key in env.observation_spec.keys(True, True) if key != "pixels"
    ]
    out_key = "observation_vector"
    env.append_transform(CatTensors(in_keys=selected_keys, out_key=out_key))
    env.append_transform(RewardSum())
    env.append_transform(StepCounter())
    # obs_norm = ObservationNorm(in_keys=[out_key])
    # env.append_transform(obs_norm)

    if env_library is DMControlEnv:
        double_to_float_list += [
            "reward",
        ]
        double_to_float_inv_list += ["action"]  # DMControl requires double-precision
        double_to_float_list += ["observation_vector"]
    else:
        double_to_float_list += ["observation_vector"]
    env.append_transform(
        DoubleToFloat(
            in_keys=double_to_float_list, in_keys_inv=double_to_float_inv_list
        )
    )
    return env


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


def get_stats(env_cfg):
    env = make_transformed_env(make_base_env(env_cfg), env_cfg)
    return env.state_dict()


def init_stats(env, n_samples_stats, from_pixels):
    for t in env.transform:
        if isinstance(t, ObservationNorm):
            if from_pixels:
                t.init_stats(
                    n_samples_stats,
                    cat_dim=-3,
                    reduce_dim=(-1, -2, -3),
                    keep_dims=(-1, -2, -3),
                )
            else:
                t.init_stats(n_samples_stats)


def make_test_env(env_cfg):
    env_cfg.num_envs = 1
    state_dict = get_stats(env_cfg)
    env = make_parallel_env(env_cfg, state_dict=state_dict)
    return env


# ====================================================================
# Collector and replay buffer
# ---------------------------


def make_collector(cfg, policy):
    env_cfg = cfg.env
    collector_cfg = cfg.collector
    collector_class = SyncDataCollector
    state_dict = get_stats(env_cfg)
    collector = collector_class(
        make_parallel_env(env_cfg, state_dict=state_dict),
        policy,
        frames_per_batch=collector_cfg.frames_per_batch,
        total_frames=collector_cfg.total_frames,
        device=collector_cfg.collector_device,
        max_frames_per_traj=collector_cfg.max_frames_per_traj,
    )
    return collector


def make_data_buffer(cfg):
    cfg_collector = cfg.collector
    cfg_loss = cfg.loss
    sampler = SamplerWithoutReplacement()
    return TensorDictReplayBuffer(
        storage=LazyMemmapStorage(cfg_collector.frames_per_batch),
        sampler=sampler,
        batch_size=cfg_loss.mini_batch_size,
    )


# ====================================================================
# Model
# -----
#
# We give one version of the model for learning from pixels, and one for state.
# TorchRL comes in handy at this point, as the high-level interactions with
# these models is unchanged, regardless of the modality.


def make_ppo_models(cfg):

    env_cfg = cfg.env
    from_pixels = env_cfg.from_pixels
    proof_environment = make_transformed_env(make_base_env(env_cfg), env_cfg)

    if not from_pixels:
        # we must initialize the observation norm transform
        # init_stats(
        #     proof_environment, n_samples_stats=3, from_pixels=env_cfg.from_pixels
        # )
        common_module, policy_module, value_module = make_ppo_modules_state(
            proof_environment
        )
    else:
        common_module, policy_module, value_module = make_ppo_modules_pixels(
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
    critic = actor_critic.get_value_head()

    return actor, critic


def make_ppo_modules_state(proof_environment):

    # Define input shape
    env_specs = proof_environment.specs
    input_shape = env_specs["output_spec"]["observation"]["observation_vector"].shape

    # Define distribution class and kwargs
    continuous_actions = False
    if isinstance(env_specs["input_spec"]["action"].space, DiscreteBox):
        num_outputs = env_specs["input_spec"]["action"].space.n
        distribution_class = OneHotCategorical
        distribution_kwargs = {}
    else:  # is ContinuousBox
        continuous_actions = True
        num_outputs = env_specs["input_spec"]["action"].shape[-1] * 2
        distribution_class = TanhNormal
        distribution_kwargs = {
            "min": env_specs["input_spec"]["action"].space.minimum,
            "max": env_specs["input_spec"]["action"].space.maximum,
            "tanh_loc": False,
        }

    # Define input keys
    in_keys = ["observation_vector"]
    shared_features_size = 256

    # Define a shared Module and TensorDictModule
    common_mlp = MLP(
        in_features=input_shape[-1],
        activation_class=torch.nn.Tanh,
        activate_last_layer=True,
        out_features=shared_features_size,
        num_cells=[64, 64],
    )
    common_module = TensorDictModule(
        module=common_mlp,
        in_keys=in_keys,
        out_keys=["common_features"],
    )

    # Define on head for the policy
    policy_net = MLP(
        in_features=shared_features_size, out_features=num_outputs, num_cells=[]
    )
    if continuous_actions:
        policy_net = NormalParamWrapper(policy_net)

    policy_module = TensorDictModule(
        module=policy_net,
        in_keys=["common_features"],
        out_keys=["loc", "scale"] if continuous_actions else ["logits"],
    )

    # Add probabilistic sampling of the actions
    policy_module = ProbabilisticActor(
        policy_module,
        in_keys=["loc", "scale"] if continuous_actions else ["logits"],
        spec=CompositeSpec(action=env_specs["input_spec"]["action"]),
        safe=True,
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    # Define another head for the value
    value_net = MLP(in_features=shared_features_size, out_features=1, num_cells=[])
    value_module = ValueOperator(
        value_net,
        in_keys=["common_features"],
    )

    return common_module, policy_module, value_module


def make_ppo_modules_pixels(proof_environment):

    # Define input shape
    env_specs = proof_environment.specs
    input_shape = env_specs["output_spec"]["observation"]["pixels"].shape

    # Define distribution class and kwargs
    if isinstance(env_specs["input_spec"]["action"].space, DiscreteBox):
        num_outputs = env_specs["input_spec"]["action"].space.n
        distribution_class = OneHotCategorical
        distribution_kwargs = {}
    else:  # is ContinuousBox
        num_outputs = env_specs["input_spec"]["action"].shape
        distribution_class = TanhNormal
        distribution_kwargs = {
            "min": env_specs["input_spec"]["action"].space.minimum,
            "max": env_specs["input_spec"]["action"].space.maximum,
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
        num_cells=[256],
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
        spec=CompositeSpec(action=env_specs["input_spec"]["action"]),
        safe=True,
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    # Define another head for the value
    value_net = MLP(
        in_features=common_mlp_output.shape[-1], out_features=1, num_cells=[256]
    )
    value_module = ValueOperator(
        value_net,
        in_keys=["common_features"],
    )

    return common_module, policy_module, value_module


# ====================================================================
# PPO Loss
# ---------


def make_advantage_module(loss_cfg, value_network):
    advantage_module = GAE(
        gamma=loss_cfg.gamma,
        lmbda=loss_cfg.gae_lamdda,
        value_network=value_network,
        average_gae=True,
    )
    return advantage_module


def make_loss(loss_cfg, actor_network, value_network):
    advantage_module = make_advantage_module(loss_cfg, value_network)
    loss_module = ClipPPOLoss(
        actor=actor_network,
        critic=value_network,
        clip_epsilon=loss_cfg.clip_epsilon,
        loss_critic_type=loss_cfg.loss_critic_type,
        entropy_coef=loss_cfg.entropy_coef,
        critic_coef=loss_cfg.critic_coef,
        normalize_advantage=True,
    )
    loss_module.make_value_estimator(gamma=loss_cfg.gamma)
    return loss_module, advantage_module


def make_optim(optim_cfg, actor_network, value_network):
    optim = torch.optim.Adam(
        list(actor_network.parameters()) + list(value_network.parameters()),
        lr=optim_cfg.lr,
        weight_decay=optim_cfg.weight_decay,
    )
    return optim


# ====================================================================
# Logging and recording
# ---------------------


def make_logger(logger_cfg):
    exp_name = generate_exp_name("PPO", logger_cfg.exp_name)
    logger_cfg.exp_name = exp_name
    logger = get_logger(logger_cfg.backend, logger_name="ppo", experiment_name=exp_name)
    return logger
