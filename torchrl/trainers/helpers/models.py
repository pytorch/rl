# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import itertools
from dataclasses import dataclass

import torch
from tensordict import set_lazy_legacy
from tensordict.nn import InteractionType
from torch import nn
from torchrl.data.tensor_specs import (
    CompositeSpec,
    DiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.data.utils import DEVICE_TYPING
from torchrl.envs.common import EnvBase
from torchrl.envs.model_based.dreamer import DreamerEnv
from torchrl.envs.transforms import TensorDictPrimer, TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import (
    NoisyLinear,
    SafeModule,
    SafeProbabilisticModule,
    SafeProbabilisticTensorDictSequential,
    SafeSequential,
)
from torchrl.modules.distributions import (
    Delta,
    OneHotCategorical,
    TanhDelta,
    TanhNormal,
)
from torchrl.modules.models.model_based import (
    DreamerActor,
    ObsDecoder,
    ObsEncoder,
    RSSMPosterior,
    RSSMPrior,
    RSSMRollout,
)
from torchrl.modules.models.models import DuelingCnnDQNet, DuelingMlpDQNet, MLP
from torchrl.modules.tensordict_module import (
    Actor,
    DistributionalQValueActor,
    QValueActor,
)
from torchrl.modules.tensordict_module.world_models import WorldModelWrapper
from torchrl.trainers.helpers import transformed_env_constructor

DISTRIBUTIONS = {
    "delta": Delta,
    "tanh-normal": TanhNormal,
    "categorical": OneHotCategorical,
    "tanh-delta": TanhDelta,
}

ACTIVATIONS = {
    "elu": nn.ELU,
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
}


def make_dqn_actor(
    proof_environment: EnvBase, cfg: "DictConfig", device: torch.device  # noqa: F821
) -> Actor:
    """DQN constructor helper function.

    Args:
        proof_environment (EnvBase): a dummy environment to retrieve the observation and action spec.
        cfg (DictConfig): contains arguments of the DQN script
        device (torch.device): device on which the model must be cast

    Returns:
         A DQN policy operator.

    Examples:
        >>> from torchrl.trainers.helpers.models import make_dqn_actor, DiscreteModelConfig
        >>> from torchrl.trainers.helpers.envs import EnvConfig
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> from torchrl.envs.transforms import ToTensorImage, TransformedEnv
        >>> import hydra
        >>> from hydra.core.config_store import ConfigStore
        >>> import dataclasses
        >>> proof_environment = TransformedEnv(GymEnv("ALE/Pong-v5",
        ...    pixels_only=True), ToTensorImage())
        >>> device = torch.device("cpu")
        >>> config_fields = [(config_field.name, config_field.type, config_field) for config_cls in
        ...                    (DiscreteModelConfig, EnvConfig)
        ...                   for config_field in dataclasses.fields(config_cls)]
        >>> Config = dataclasses.make_dataclass(cls_name="Config", fields=config_fields)
        >>> cs = ConfigStore.instance()
        >>> cs.store(name="config", node=Config)
        >>> with initialize(config_path=None):
        >>>     cfg = compose(config_name="config")
        >>> actor = make_dqn_actor(proof_environment, cfg, device)
        >>> td = proof_environment.reset()
        >>> print(actor(td))
        TensorDict(
            fields={
                done: Tensor(torch.Size([1]), dtype=torch.bool),
                pixels: Tensor(torch.Size([3, 210, 160]), dtype=torch.float32),
                action: Tensor(torch.Size([6]), dtype=torch.int64),
                action_value: Tensor(torch.Size([6]), dtype=torch.float32),
                chosen_action_value: Tensor(torch.Size([1]), dtype=torch.float32)},
            batch_size=torch.Size([]),
            device=cpu,
            is_shared=False)


    """
    env_specs = proof_environment.specs

    atoms = cfg.atoms if cfg.distributional else None
    linear_layer_class = torch.nn.Linear if not cfg.noisy else NoisyLinear

    action_spec = env_specs["input_spec", "full_action_spec", "action"]
    if action_spec.domain != "discrete":
        raise ValueError(
            f"env {proof_environment} has an action domain "
            f"{action_spec.domain} which is incompatible with "
            f"DQN. Make sure your environment has a discrete "
            f"domain."
        )

    if cfg.from_pixels:
        net_class = DuelingCnnDQNet
        default_net_kwargs = {
            "cnn_kwargs": {
                "bias_last_layer": True,
                "depth": None,
                "num_cells": [32, 64, 64],
                "kernel_sizes": [8, 4, 3],
                "strides": [4, 2, 1],
            },
            "mlp_kwargs": {"num_cells": 512, "layer_class": linear_layer_class},
        }
        in_key = "pixels"

    else:
        net_class = DuelingMlpDQNet
        default_net_kwargs = {
            "mlp_kwargs_feature": {},  # see class for details
            "mlp_kwargs_output": {"num_cells": 512, "layer_class": linear_layer_class},
        }
        # automatically infer in key
        (in_key,) = itertools.islice(
            env_specs["output_spec", "full_observation_spec"], 1
        )

    actor_class = QValueActor
    actor_kwargs = {}

    if isinstance(action_spec, DiscreteTensorSpec):
        # if action spec is modeled as categorical variable, we still need to have features equal
        # to the number of possible choices and also set categorical behavioural for actors.
        actor_kwargs.update({"action_space": "categorical"})
        out_features = env_specs["input_spec", "full_action_spec", "action"].space.n
    else:
        out_features = action_spec.shape[0]

    if cfg.distributional:
        if not atoms:
            raise RuntimeError(
                "Expected atoms to be a positive integer, " f"got {atoms}"
            )
        vmin = -3
        vmax = 3

        out_features = (atoms, out_features)
        support = torch.linspace(vmin, vmax, atoms)
        actor_class = DistributionalQValueActor
        actor_kwargs.update({"support": support})
        default_net_kwargs.update({"out_features_value": (atoms, 1)})

    net = net_class(
        out_features=out_features,
        **default_net_kwargs,
    )

    model = actor_class(
        module=net,
        spec=CompositeSpec(action=action_spec),
        in_keys=[in_key],
        safe=True,
        **actor_kwargs,
    ).to(device)

    # init
    with torch.no_grad():
        td = proof_environment.fake_tensordict()
        td = td.unsqueeze(-1)
        model(td.to(device))
    return model


@set_lazy_legacy(False)
def make_dreamer(
    cfg: "DictConfig",  # noqa: F821
    proof_environment: EnvBase = None,
    device: DEVICE_TYPING = "cpu",
    action_key: str = "action",
    value_key: str = "state_value",
    use_decoder_in_env: bool = False,
    obs_norm_state_dict=None,
) -> nn.ModuleList:
    """Create Dreamer components.

    Args:
        cfg (DictConfig): Config object.
        proof_environment (EnvBase): Environment to initialize the model.
        device (DEVICE_TYPING, optional): Device to use.
            Defaults to "cpu".
        action_key (str, optional): Key to use for the action.
            Defaults to "action".
        value_key (str, optional): Key to use for the value.
            Defaults to "state_value".
        use_decoder_in_env (bool, optional): Whether to use the decoder in the model based dreamer env.
            Defaults to False.
        obs_norm_state_dict (dict, optional): the state_dict of the ObservationNorm transform used
            when proof_environment is missing. Defaults to None.

    Returns:
        nn.TensorDictModel: Dreamer World model.
        nn.TensorDictModel: Dreamer Model based environnement.
        nn.TensorDictModel: Dreamer Actor the world model space.
        nn.TensorDictModel: Dreamer Value model.
        nn.TensorDictModel: Dreamer Actor for the real world space.

    """
    proof_env_is_none = proof_environment is None
    if proof_env_is_none:
        proof_environment = transformed_env_constructor(
            cfg=cfg, use_env_creator=False, obs_norm_state_dict=obs_norm_state_dict
        )()

    # Modules
    obs_encoder = ObsEncoder()
    obs_decoder = ObsDecoder()

    rssm_prior = RSSMPrior(
        hidden_dim=cfg.rssm_hidden_dim,
        rnn_hidden_dim=cfg.rssm_hidden_dim,
        state_dim=cfg.state_dim,
        action_spec=proof_environment.action_spec,
    )
    rssm_posterior = RSSMPosterior(
        hidden_dim=cfg.rssm_hidden_dim, state_dim=cfg.state_dim
    )
    reward_module = MLP(
        out_features=1, depth=2, num_cells=cfg.mlp_num_units, activation_class=nn.ELU
    )

    world_model = _dreamer_make_world_model(
        obs_encoder, obs_decoder, rssm_prior, rssm_posterior, reward_module
    ).to(device)
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        tensordict = proof_environment.fake_tensordict().unsqueeze(-1)
        tensordict = tensordict.to(device)
        world_model(tensordict)

    model_based_env = _dreamer_make_mbenv(
        reward_module,
        rssm_prior,
        obs_decoder,
        proof_environment,
        use_decoder_in_env,
        cfg.state_dim,
        cfg.rssm_hidden_dim,
    )
    model_based_env = model_based_env.to(device)

    actor_simulator, actor_realworld = _dreamer_make_actors(
        obs_encoder,
        rssm_prior,
        rssm_posterior,
        cfg.mlp_num_units,
        action_key,
        proof_environment,
    )
    actor_simulator = actor_simulator.to(device)

    value_model = _dreamer_make_value_model(cfg.mlp_num_units, value_key)
    value_model = value_model.to(device)
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        tensordict = model_based_env.fake_tensordict().unsqueeze(-1)
        tensordict = tensordict.to(device)
        tensordict = actor_simulator(tensordict)
        value_model(tensordict)

    actor_realworld = actor_realworld.to(device)
    if proof_env_is_none:
        proof_environment.close()
        torch.cuda.empty_cache()
        del proof_environment

    del tensordict
    return world_model, model_based_env, actor_simulator, value_model, actor_realworld


def _dreamer_make_world_model(
    obs_encoder, obs_decoder, rssm_prior, rssm_posterior, reward_module
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

    transition_model = SafeSequential(
        SafeModule(
            obs_encoder,
            in_keys=[("next", "pixels")],
            out_keys=[("next", "encoded_latents")],
        ),
        rssm_rollout,
        SafeModule(
            obs_decoder,
            in_keys=[("next", "state"), ("next", "belief")],
            out_keys=[("next", "reco_pixels")],
        ),
    )
    reward_model = SafeModule(
        reward_module,
        in_keys=[("next", "state"), ("next", "belief")],
        out_keys=[("next", "reward")],
    )
    world_model = WorldModelWrapper(
        transition_model,
        reward_model,
    )
    return world_model


def _dreamer_make_actors(
    obs_encoder,
    rssm_prior,
    rssm_posterior,
    mlp_num_units,
    action_key,
    proof_environment,
):
    actor_module = DreamerActor(
        out_features=proof_environment.action_spec.shape[0],
        depth=3,
        num_cells=mlp_num_units,
        activation_class=nn.ELU,
    )
    actor_simulator = _dreamer_make_actor_sim(
        action_key, proof_environment, actor_module
    )
    actor_realworld = _dreamer_make_actor_real(
        obs_encoder,
        rssm_prior,
        rssm_posterior,
        actor_module,
        action_key,
        proof_environment,
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
    obs_encoder, rssm_prior, rssm_posterior, actor_module, action_key, proof_environment
):
    # actor for real world: interacts with states ~ posterior
    # Out actor differs from the original paper where first they compute prior and posterior and then act on it
    # but we found that this approach worked better.
    actor_realworld = SafeSequential(
        SafeModule(
            obs_encoder,
            in_keys=["pixels"],
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


def _dreamer_make_value_model(mlp_num_units, value_key):
    # actor for simulator: interacts with states ~ prior
    value_model = SafeModule(
        MLP(
            out_features=1,
            depth=3,
            num_cells=mlp_num_units,
            activation_class=nn.ELU,
        ),
        in_keys=["state", "belief"],
        out_keys=[value_key],
    )
    return value_model


def _dreamer_make_mbenv(
    reward_module,
    rssm_prior,
    obs_decoder,
    proof_environment,
    use_decoder_in_env,
    state_dim,
    rssm_hidden_dim,
):
    # MB environment
    if use_decoder_in_env:
        mb_env_obs_decoder = SafeModule(
            obs_decoder,
            in_keys=[("next", "state"), ("next", "belief")],
            out_keys=[("next", "reco_pixels")],
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
    reward_model = SafeModule(
        reward_module,
        in_keys=["state", "belief"],
        out_keys=["reward"],
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

    model_based_env.set_specs_from_env(proof_environment)
    model_based_env = TransformedEnv(model_based_env)
    default_dict = {
        "state": UnboundedContinuousTensorSpec(state_dim),
        "belief": UnboundedContinuousTensorSpec(rssm_hidden_dim),
        # "action": proof_environment.action_spec,
    }
    model_based_env.append_transform(
        TensorDictPrimer(random=False, default_value=0, **default_dict)
    )
    return model_based_env


@dataclass
class DreamerConfig:
    """Dreamer model config struct."""

    batch_length: int = 50
    state_dim: int = 30
    rssm_hidden_dim: int = 200
    mlp_num_units: int = 400
    grad_clip: int = 100
    world_model_lr: float = 6e-4
    actor_value_lr: float = 8e-5
    imagination_horizon: int = 15
    model_device: str = ""
    # Decay of the reward moving averaging
    exploration: str = "additive_gaussian"
    # One of "additive_gaussian", "ou_exploration" or ""


@dataclass
class REDQModelConfig:
    """REDQ model config struct."""

    annealing_frames: int = 1000000
    # float of frames used for annealing of the OrnsteinUhlenbeckProcess. Default=1e6.
    noisy: bool = False
    # whether to use NoisyLinearLayers in the value network.
    ou_exploration: bool = False
    # wraps the policy in an OU exploration wrapper, similar to DDPG. SAC being designed for
    # efficient entropy-based exploration, this should be left for experimentation only.
    ou_sigma: float = 0.2
    # Ornstein-Uhlenbeck sigma
    ou_theta: float = 0.15
    # Aimed at superseeding --ou_exploration.
    distributional: bool = False
    # whether a distributional loss should be used (TODO: not implemented yet).
    atoms: int = 51
    # number of atoms used for the distributional loss (TODO)
    gSDE: bool = False
    # if True, exploration is achieved using the gSDE technique.
    tanh_loc: bool = False
    # if True, uses a Tanh-Normal transform for the policy location of the form
    # upscale * tanh(loc/upscale) (only available with TanhTransform and TruncatedGaussian distributions)
    default_policy_scale: float = 1.0
    # Default policy scale parameter
    distribution: str = "tanh_normal"
    # if True, uses a Tanh-Normal-Tanh distribution for the policy
    actor_cells: int = 256
    # cells of the actor
    qvalue_cells: int = 256
    # cells of the qvalue net
    scale_lb: float = 0.1
    # min value of scale
    value_cells: int = 256
    # cells of the value net
    activation: str = "tanh"
    # activation function, either relu or elu or tanh, Default=tanh


@dataclass
class ContinuousModelConfig:
    """Continuous control model config struct."""

    annealing_frames: int = 1000000
    # float of frames used for annealing of the OrnsteinUhlenbeckProcess. Default=1e6.
    noisy: bool = False
    # whether to use NoisyLinearLayers in the value network.
    ou_exploration: bool = False
    # wraps the policy in an OU exploration wrapper, similar to DDPG. SAC being designed for
    # efficient entropy-based exploration, this should be left for experimentation only.
    ou_sigma: float = 0.2
    # Ornstein-Uhlenbeck sigma
    ou_theta: float = 0.15
    # Aimed at superseeding --ou_exploration.
    distributional: bool = False
    # whether a distributional loss should be used (TODO: not implemented yet).
    atoms: int = 51
    # number of atoms used for the distributional loss (TODO)
    gSDE: bool = False
    # if True, exploration is achieved using the gSDE technique.
    tanh_loc: bool = False
    # if True, uses a Tanh-Normal transform for the policy location of the form
    # upscale * tanh(loc/upscale) (only available with TanhTransform and TruncatedGaussian distributions)
    default_policy_scale: float = 1.0
    # Default policy scale parameter
    distribution: str = "tanh_normal"
    # if True, uses a Tanh-Normal-Tanh distribution for the policy
    lstm: bool = False
    # if True, uses an LSTM for the policy.
    shared_mapping: bool = False
    # if True, the first layers of the actor-critic are shared.
    actor_cells: int = 256
    # cells of the actor
    qvalue_cells: int = 256
    # cells of the qvalue net
    scale_lb: float = 0.1
    # min value of scale
    value_cells: int = 256
    # cells of the value net
    activation: str = "tanh"
    # activation function, either relu or elu or tanh, Default=tanh


@dataclass
class DiscreteModelConfig:
    """Discrete model config struct."""

    annealing_frames: int = 1000000
    # Number of frames used for annealing of the EGreedy exploration. Default=1e6.
    noisy: bool = False
    # whether to use NoisyLinearLayers in the value network
    distributional: bool = False
    # whether a distributional loss should be used.
    atoms: int = 51
    # number of atoms used for the distributional loss
